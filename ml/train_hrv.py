"""
ml/train_hrv.py
---------------
Train a scikit-learn classifier on the HRV Stress Dataset.

Dataset: c:/Users/DELL/Downloads/archive (1)/hrv dataset/data/final/
  train.csv / test.csv
  34 HRV features + condition label (no stress / interruption / time pressure / amusement)

Label mapping:
  no stress  -> 0
  everything else (interruption, time pressure, amusement) -> 1 (stress/arousal)

Features used (overlap with PanicGuard FeatureVector):
  HR       -> hr_mean
  RMSSD    -> rmssd
  SDSD     -> sdnn (proxy)
  LF_HF    -> hr_resp_coupling (proxy)
  sampen   -> used as anomaly feature
  + all other 34 HRV features as extended input

Output:
  models/hrv_panic_classifier.pkl  -- sklearn Pipeline (scaler + RF)
  models/hrv_panic_classifier_report.txt

Usage:
  python -m ml.train_hrv
  python -m ml.train_hrv --data-dir "c:/Users/DELL/Downloads/archive (1)/hrv dataset/data/final"
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Train PanicGuard sklearn model on HRV dataset")
    parser.add_argument(
        "--data-dir",
        default=r"c:/Users/DELL/Downloads/archive (1)/hrv dataset/data/final",
        help="Path to folder containing train.csv and test.csv",
    )
    parser.add_argument("--model-dir", default="models", help="Output directory for model")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, "hrv_panic_classifier.pkl")
    report_path = os.path.join(args.model_dir, "hrv_panic_classifier_report.txt")

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

    train_csv = os.path.join(args.data_dir, "train.csv")
    test_csv  = os.path.join(args.data_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    print(f"Train: {len(train_df)} rows   Test: {len(test_df)} rows")
    print("Conditions:", train_df["condition"].value_counts().to_dict())

    # Binary label: no stress = 0, any stress/arousal = 1
    def binarize(df):
        df = df.copy()
        df["label"] = (df["condition"] != "no stress").astype(int)
        return df

    train_df = binarize(train_df)
    test_df  = binarize(test_df)

    feature_cols = [c for c in train_df.columns if c not in ("condition", "label", "datasetId")]

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["label"].values
    X_test  = test_df[feature_cols].values.astype(float)
    y_test  = test_df["label"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Train stress ratio: {y_train.mean():.3f}   Test: {y_test.mean():.3f}")

    # Pipeline: scale -> select top-20 -> RandomForest
    pipeline = Pipeline([
        ("scaler",    StandardScaler()),
        ("selector",  SelectKBest(f_classif, k=20)),
        ("clf",       RandomForestClassifier(
                          n_estimators=args.n_estimators,
                          max_features="sqrt",
                          class_weight="balanced",
                          random_state=42,
                          n_jobs=-1,
                      )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["no stress", "stress"])
    auc    = roc_auc_score(y_test, y_pred_prob)
    cm     = confusion_matrix(y_test, y_pred)

    print("\nTest results:")
    print(report)
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)

    # Save model
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved: {out_path}")

    # Save report
    with open(report_path, "w") as f:
        f.write(f"HRV Stress Dataset - PanicGuard Classifier\n")
        f.write(f"Features: {feature_cols}\n\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")
        f.write(f"Confusion matrix:\n{cm}\n")
    print(f"Report saved: {report_path}")

    print("\nTo use in PanicGuard:")
    print(f"  classifier = PanicClassifier(bus, backend='sklearn', model_path='{out_path}')")


if __name__ == "__main__":
    main()
