"""
ml/train_wesad.py
-----------------
End-to-end training pipeline for PanicGuard using the WESAD dataset.

Steps:
  1. Extract features from all WESAD subjects -> data/wesad_features.csv
  2. Train Bi-LSTM + Attention classifier     -> models/bilstm.keras + .tflite
  3. Train LSTM Autoencoder (normal data)     -> models/autoencoder.keras + .tflite
  4. Evaluate and print classification report

Usage:
  python -m ml.train_wesad
  python -m ml.train_wesad --wesad e:/WESAD --epochs 50 --batch-size 32
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Train PanicGuard models on WESAD")
    parser.add_argument("--wesad",        default="e:/WESAD", help="WESAD root folder")
    parser.add_argument("--data-dir",     default="data",     help="Output directory for CSVs")
    parser.add_argument("--model-dir",    default="models",   help="Output directory for models")
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip feature extraction if CSVs already exist")
    args = parser.parse_args()

    os.makedirs(args.data_dir,  exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    features_csv = os.path.join(args.data_dir,  "wesad_features.csv")
    normal_csv   = os.path.join(args.data_dir,  "wesad_features_normal.csv")
    lstm_path    = os.path.join(args.model_dir, "bilstm.keras")
    ae_path      = os.path.join(args.model_dir, "autoencoder.keras")
    cnn_path     = os.path.join(args.model_dir, "cnn.keras")

    # Step 1: Feature extraction
    if args.skip_extract and os.path.exists(features_csv):
        print("[Step 1] Skipping extraction - using existing", features_csv)
    else:
        print("[Step 1] Extracting features from WESAD ...")
        from ml.wesad_loader import build_dataset
        build_dataset(args.wesad, features_csv)

    # Step 2: Train Bi-LSTM
    print("\n[Step 2] Training Bi-LSTM + Attention ->", lstm_path)
    _train_bilstm(features_csv, lstm_path, args.epochs, args.batch_size)

    # Step 3: Train Autoencoder
    print("\n[Step 3] Training LSTM Autoencoder ->", ae_path)
    _train_autoencoder(normal_csv, ae_path, args.epochs, args.batch_size)

    # Step 4: Train CNN
    print("\n[Step 4] Training 1D CNN ->", cnn_path)
    _train_cnn(features_csv, cnn_path, args.epochs, args.batch_size)

    print("\nTraining complete.")
    print("  Bi-LSTM    :", lstm_path)
    print("  Autoencoder:", ae_path)
    print("  CNN        :", cnn_path)


def _train_bilstm(data_path: str, out_path: str, epochs: int, batch_size: int):
    import pickle
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    from ml.predictive_model import build_bilstm_attention
    from ml.inference import FeatureVector
    from ml.sequence_buffer import SEQ_LEN, N_FEATURES

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} windows  stress_ratio={df['label'].mean():.3f}")

    X_raw = df[FeatureVector.FEATURE_NAMES].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.float32)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    scaler_path = out_path.replace(".keras", "_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print("  Scaler saved:", scaler_path)

    # Each CSV row is already a 60s feature window extracted by wesad_loader.
    # Expand to (N, SEQ_LEN, N_FEATURES) by repeating the feature vector
    # SEQ_LEN times along the time axis so the Bi-LSTM input shape is satisfied.
    X_arr = np.repeat(X_sc[:, np.newaxis, :], SEQ_LEN, axis=1).astype(np.float32)
    y_arr = y_raw
    print(f"  Samples: {len(X_arr)}  stress_ratio={y_arr.mean():.3f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=42)

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    cw = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
    print(f"  Class weights: {cw}")

    model = build_bilstm_attention(seq_len=SEQ_LEN, n_features=N_FEATURES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                out_path, save_best_only=True, monitor="val_auc", mode="max", verbose=1),
        ],
    )

    y_pred_prob = model.predict(X_val, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    print("\n  Validation results:")
    print(classification_report(y_val.astype(int), y_pred, target_names=["normal", "stress"]))
    print(f"  ROC-AUC: {roc_auc_score(y_val, y_pred_prob):.4f}")

    tflite_path = out_path.replace(".keras", ".tflite")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv._experimental_lower_tensor_list_ops = False
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())
    print("  TFLite saved:", tflite_path)


def _train_autoencoder(data_path: str, out_path: str, epochs: int, batch_size: int):
    import pickle
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from ml.predictive_model import build_autoencoder
    from ml.inference import FeatureVector
    from ml.sequence_buffer import SEQ_LEN

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} normal windows")

    X_raw = df[FeatureVector.FEATURE_NAMES].values.astype(np.float32)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    # Each CSV row is already a 60s feature window — expand to (N, SEQ_LEN, N_FEATURES)
    X_arr = np.repeat(X_sc[:, np.newaxis, :], SEQ_LEN, axis=1).astype(np.float32)
    print(f"  Samples: {len(X_arr)}")

    autoencoder, _ = build_autoencoder(seq_len=SEQ_LEN)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_arr, X_arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                out_path, save_best_only=True, monitor="val_loss", verbose=1),
        ],
    )

    reconstructed = autoencoder.predict(X_arr, verbose=0)
    errors    = np.mean((X_arr - reconstructed) ** 2, axis=(1, 2))
    threshold = float(np.percentile(errors, 95))
    print(f"  Anomaly threshold (95th pct): {threshold:.6f}")

    thr_path = out_path.replace(".keras", "_threshold.pkl")
    with open(thr_path, "wb") as f:
        pickle.dump(threshold, f)
    print("  Threshold saved:", thr_path)

    tflite_path = out_path.replace(".keras", ".tflite")
    conv = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv._experimental_lower_tensor_list_ops = False
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())
    print("  TFLite saved:", tflite_path)


def _train_cnn(data_path: str, out_path: str, epochs: int, batch_size: int):
    import pickle
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    from ml.predictive_model import build_cnn
    from ml.inference import FeatureVector
    from ml.sequence_buffer import SEQ_LEN, N_FEATURES

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} windows  stress_ratio={df['label'].mean():.3f}")

    scaler = pickle.load(open(out_path.replace("cnn.keras", "bilstm_scaler.pkl"), "rb"))
    X_raw  = df[FeatureVector.FEATURE_NAMES].values.astype(np.float32)
    y_raw  = df["label"].values.astype(np.float32)
    X_sc   = scaler.transform(X_raw)
    X_arr  = np.repeat(X_sc[:, np.newaxis, :], SEQ_LEN, axis=1).astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr, y_raw, test_size=0.2, stratify=y_raw, random_state=42)

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    cw = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
    print(f"  Class weights: {cw}")

    model = build_cnn(seq_len=SEQ_LEN, n_features=N_FEATURES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                out_path, save_best_only=True, monitor="val_auc", mode="max", verbose=1),
        ],
    )

    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    print("\n  CNN Validation results:")
    print(classification_report(y_val.astype(int), y_pred, target_names=["normal", "stress"]))
    print(f"  ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")

    tflite_path = out_path.replace(".keras", ".tflite")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())
    print("  CNN TFLite saved:", tflite_path)


if __name__ == "__main__":
    main()
