"""
ml/predictive_model.py
──────────────────────
Full AI processing layer — two models + fusion engine.

Model 1 — Bi-LSTM + Attention  (supervised, time-series prediction)
    Input : (batch, SEQ_LEN=60, N_FEATURES=13)  — 60s biosignal window at 1Hz
    Output: panic probability 0.0–1.0 for next HORIZON_SECS (120s)

    Why Bi-LSTM?
      Forward LSTM  — learns how signals build up toward panic
      Backward LSTM — learns what patterns precede panic in reverse
      Attention     — weights which timesteps matter most (e.g. the HR spike
                      at t-45s matters more than baseline noise at t-60s)

Model 2 — Autoencoder  (unsupervised, anomaly detection)
    Input : (batch, SEQ_LEN=60, N_FEATURES=13)  — same window
    Output: reconstruction error (anomaly score 0.0–1.0)

    Why Autoencoder?
      Trained only on NORMAL physiology — learns to reconstruct calm patterns.
      When panic physiology appears, reconstruction error spikes → anomaly.
      Catches sudden abnormal changes that the LSTM hasn't seen before.

Decision Fusion Engine
    Combines both scores:
        IF lstm_prob > 0.70  OR  anomaly_score > threshold  → HIGH risk
        ELSE                                                 → LOW risk
    Supports weighted scoring + per-user personalised thresholds.

Training:
    # Train Bi-LSTM
    python -m ml.predictive_model --train lstm --data data/episodes.csv

    # Train Autoencoder (normal data only)
    python -m ml.predictive_model --train autoencoder --data data/normal.csv

Inference (runtime — no model files needed, uses rule fallback):
    engine = FusionEngine(bus)
"""

import time
import collections
import numpy as np
from loguru import logger

from core.event_bus import EventBus, Topics
from ml.inference import FeatureVector
from ml.sequence_buffer import SequenceBuffer, SEQ_LEN, N_FEATURES, HORIZON_SECS


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Bi-LSTM + Attention
# ─────────────────────────────────────────────────────────────────────────────

def build_bilstm_attention(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES,
                           dropout: float = 0.3):
    """
    Bi-LSTM + Attention architecture.

    Block 1 — CNN front-end (local pattern extraction):
        Conv1D(32, k=5, causal) → BN → MaxPool(2) → Dropout
        Conv1D(64, k=3, causal) → BN → MaxPool(2) → Dropout

    Block 2 — Bi-LSTM (temporal modelling):
        Bidirectional(LSTM(128, return_sequences=True)) → Dropout
        Bidirectional(LSTM(64,  return_sequences=True)) → Dropout

    Block 3 — Attention (focus on most panic-predictive timesteps):
        Dense(1) → softmax over time → weighted sum of LSTM outputs

    Head:
        Dense(32, relu) → Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(seq_len, n_features), name="biosignal_seq")

    # CNN front-end
    x = layers.Conv1D(32, kernel_size=5, padding="causal", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    # Bi-LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    # Attention — learn which timesteps predict panic
    attn_weights = layers.Dense(1, activation="tanh")(x)          # (batch, T, 1)
    attn_weights = layers.Softmax(axis=1)(attn_weights)            # normalise over time
    x = layers.Multiply()([x, attn_weights])                       # weighted LSTM output
    x = layers.GlobalAveragePooling1D()(x)                         # context vector (replaces Lambda)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="panic_prob")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="BiLSTM_Attention")


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Autoencoder (anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES,
                      latent_dim: int = 16):
    """
    LSTM Autoencoder trained on NORMAL physiology only.

    Encoder: LSTM(64) → LSTM(latent_dim)   — compress to latent representation
    Decoder: RepeatVector → LSTM(64, return_seq) → TimeDistributed Dense
             — reconstruct the original sequence

    At inference:
        reconstruction_error = MSE(input, reconstructed)
        High error → pattern is abnormal → anomaly detected
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(seq_len, n_features), name="ae_input")

    # Encoder
    x = layers.LSTM(64, return_sequences=True)(inp)
    encoded = layers.LSTM(latent_dim, return_sequences=False, name="latent")(x)

    # Decoder
    x = layers.RepeatVector(seq_len)(encoded)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(x)

    autoencoder = tf.keras.Model(inputs=inp, outputs=decoded, name="LSTM_Autoencoder")
    encoder     = tf.keras.Model(inputs=inp, outputs=encoded,  name="Encoder")

    return autoencoder, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — 1D CNN (local spike pattern detection)
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES,
              dropout: float = 0.3):
    """
    Deep 1D CNN for panic detection.

    Why CNN for biosignals?
      - Detects LOCAL patterns: sudden HR spike, SCR burst, breathing fragmentation
      - Faster than LSTM (no recurrence, fully parallelizable)
      - Residual connections prevent vanishing gradients in deep networks
      - Complementary to LSTM: CNN catches sharp transient events,
        LSTM catches long-range temporal buildup

    Architecture:
      Input (batch, SEQ_LEN, N_FEATURES)
        -> Conv block 1: Conv1D(64, k=3) + BN + ReLU + Dropout
        -> Conv block 2: Conv1D(128, k=3) + BN + ReLU + Dropout  [residual]
        -> Conv block 3: Conv1D(256, k=3) + BN + ReLU + Dropout  [residual]
        -> GlobalMaxPool  (picks the most activated timestep per filter)
        -> Dense(64, relu) -> Dropout -> Dense(1, sigmoid)

    GlobalMaxPool vs GlobalAvgPool:
      MaxPool keeps the PEAK activation — ideal for detecting spike events
      (a single SCR burst or HR spike should fire the classifier)
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(seq_len, n_features), name="cnn_input")

    # Block 1
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Block 2 with residual
    res = layers.Conv1D(128, kernel_size=1, padding="same")(x)   # projection
    x   = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Add()([x, res])
    x   = layers.Activation("relu")(x)

    # Block 3 with residual
    res = layers.Conv1D(256, kernel_size=1, padding="same")(x)
    x   = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Conv1D(256, kernel_size=3, padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Add()([x, res])
    x   = layers.Activation("relu")(x)

    # Global max pooling — captures peak activation (spike detection)
    x = layers.GlobalMaxPooling1D()(x)

    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="cnn_panic_prob")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="PanicCNN")


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback (no model needed — works on day 1)
# ─────────────────────────────────────────────────────────────────────────────

class TrendScorer:
    """
    Linear regression slope scorer across all 13 features.
    Used when no trained model is available.
    """
    IDX = {"hr_mean": 0, "rmssd": 2, "scl_mean": 4,
           "resp_rate": 7, "resp_regularity": 8, "hr_resp_coupling": 12}

    def predict(self, seq: np.ndarray) -> float:
        t = np.arange(seq.shape[0], dtype=float)

        def slope(i):
            y = seq[:, i]
            tm, ym = t.mean(), y.mean()
            den = ((t - tm) ** 2).sum()
            return float(((t - tm) * (y - ym)).sum() / den) if den > 1e-9 else 0.0

        score  = 0.30 * min(1.0, max(0.0,  slope(self.IDX["hr_mean"])       / 0.5))
        score += 0.25 * min(1.0, max(0.0, -slope(self.IDX["rmssd"])         / 0.3))
        score += 0.20 * min(1.0, max(0.0,  slope(self.IDX["scl_mean"])      / 0.05))
        score += 0.15 * min(1.0, max(0.0,  slope(self.IDX["resp_rate"])     / 0.2))
        score += 0.10 * min(1.0, max(0.0, -slope(self.IDX["hr_resp_coupling"]) / 0.005))
        return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Decision Fusion Engine
# ─────────────────────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Subscribes to ml.features.
    Runs Bi-LSTM (Model 1) + Autoencoder (Model 2) on every sliding window.
    Combines scores and publishes:
        ml.prediction_score  — Bi-LSTM probability
        ml.anomaly_score     — Autoencoder reconstruction error (normalised)
        ml.fusion_risk       — final HIGH / LOW decision dict
        ml.prediction_alert  — fired when fusion risk is HIGH

    Fusion logic:
        IF lstm_prob > lstm_threshold  OR  anomaly_score > anomaly_threshold
            → FUSION_RISK = HIGH
        Weighted score = w_lstm * lstm_prob + w_anomaly * anomaly_score

    Personalised thresholds loaded from user baseline profile.
    """

    # Default fusion thresholds (override via baseline config)
    LSTM_THRESHOLD    = 0.55
    ANOMALY_THRESHOLD = 0.50
    CNN_THRESHOLD     = 0.55
    W_LSTM            = 0.40   # Bi-LSTM weight
    W_CNN             = 0.25   # CNN weight
    W_ANOMALY         = 0.20   # Autoencoder weight
    W_RULE            = 0.15   # Rule-based weight
    SMOOTH_LEN        = 5

    def __init__(self, bus: EventBus,
                 lstm_path:        str | None = None,
                 autoencoder_path: str | None = None,
                 cnn_path:         str | None = None,
                 scaler_path:      str | None = None,
                 baseline:         dict | None = None):
        self.bus = bus
        self._seq_buf     = SequenceBuffer()
        self._lstm_scores: collections.deque = collections.deque(maxlen=self.SMOOTH_LEN)
        self._ae_scores:   collections.deque = collections.deque(maxlen=self.SMOOTH_LEN)
        self._cnn_scores:  collections.deque = collections.deque(maxlen=self.SMOOTH_LEN)

        self._lstm_model = None
        self._ae_model   = None
        self._cnn_model  = None
        self._scaler     = None
        self._ae_threshold = None
        self._backend    = "rules"
        self._trend      = TrendScorer()
        self._last_rule_score: float = 0.0   # updated by PanicClassifier via event bus

        # Apply personalised thresholds from user baseline
        bl = baseline or {}
        self._lstm_thr   = bl.get("lstm_threshold",    self.LSTM_THRESHOLD)
        self._ae_thr_mul = bl.get("anomaly_sensitivity", 1.0)  # multiplier on learned threshold

        if lstm_path:
            self._load_lstm(lstm_path, scaler_path)
        if autoencoder_path:
            self._load_autoencoder(autoencoder_path)
        if cnn_path:
            self._load_cnn(cnn_path)

        bus.subscribe(Topics.FEATURES,    self._on_features)
        bus.subscribe(Topics.PANIC_SCORE, self._on_rule_score)
        logger.info(f"[FusionEngine] started  backend={self._backend}  "
                    f"lstm_thr={self._lstm_thr:.2f}  ae_thr_mul={self._ae_thr_mul:.2f}")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_lstm(self, path: str, scaler_path: str | None):
        import pickle
        if scaler_path:
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info(f"[FusionEngine] scaler loaded from {scaler_path}")

        if path.endswith(".tflite"):
            self._lstm_model = self._load_tflite(path)
            self._backend = "tflite"
        else:
            import tensorflow as tf
            import keras
            keras.config.enable_unsafe_deserialization()
            self._lstm_model = tf.keras.models.load_model(path)
            self._backend = "keras"
        logger.info(f"[FusionEngine] Bi-LSTM loaded from {path}")

    def _load_autoencoder(self, path: str):
        # Load threshold alongside model (saved during training)
        import pickle
        thr_path = path.replace(".keras", "_threshold.pkl").replace(".tflite", "_threshold.pkl")
        try:
            with open(thr_path, "rb") as f:
                self._ae_threshold = pickle.load(f)
            logger.info(f"[FusionEngine] AE threshold={self._ae_threshold:.4f}")
        except FileNotFoundError:
            self._ae_threshold = 0.05   # fallback default
            logger.warning("[FusionEngine] AE threshold file not found, using default 0.05")

        if path.endswith(".tflite"):
            self._ae_model = self._load_tflite(path)
        else:
            import tensorflow as tf
            import keras
            keras.config.enable_unsafe_deserialization()
            self._ae_model = tf.keras.models.load_model(path)
        logger.info(f"[FusionEngine] Autoencoder loaded from {path}")

    def _load_tflite(self, path: str):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        interp = tflite.Interpreter(model_path=path)
        interp.allocate_tensors()
        return interp

    def _load_cnn(self, path: str):
        if path.endswith(".tflite"):
            self._cnn_model = self._load_tflite(path)
        else:
            import tensorflow as tf
            import keras
            keras.config.enable_unsafe_deserialization()
            self._cnn_model = tf.keras.models.load_model(path)
        logger.info(f"[FusionEngine] CNN loaded from {path}")

    # ── Rule score capture ────────────────────────────────────────────────────

    async def _on_rule_score(self, topic: str, score: float):
        self._last_rule_score = score

    # ── Feature handler ───────────────────────────────────────────────────────

    async def _on_features(self, topic: str, fv: FeatureVector):
        self._seq_buf.push(fv)

        if not self._seq_buf.window_ready():
            await self.bus.publish(Topics.PREDICTION_SCORE, 0.0)
            await self.bus.publish(Topics.ANOMALY_SCORE,    0.0)
            return

        seq = self._seq_buf.get_array()   # (1, SEQ_LEN, N_FEATURES)

        if self._scaler is not None:
            seq = self._scaler.transform(
                seq.reshape(-1, N_FEATURES)
            ).reshape(seq.shape).astype(np.float32)

        lstm_prob    = self._infer_lstm(seq)
        anomaly_norm = self._infer_autoencoder(seq)
        cnn_prob     = self._infer_cnn(seq)

        # Smooth all scores
        self._lstm_scores.append(lstm_prob)
        self._ae_scores.append(anomaly_norm)
        self._cnn_scores.append(cnn_prob)
        lstm_smooth = float(np.mean(self._lstm_scores))
        ae_smooth   = float(np.mean(self._ae_scores))
        cnn_smooth  = float(np.mean(self._cnn_scores))

        # Weighted fusion: 40% LSTM + 25% CNN + 20% Anomaly + 15% Rule
        rule_score   = self._last_rule_score
        fusion_score = (self.W_LSTM    * lstm_smooth
                      + self.W_CNN     * cnn_smooth
                      + self.W_ANOMALY * ae_smooth
                      + self.W_RULE    * rule_score)

        # Decision: HIGH if any model fires or fusion is high
        risk_high = (lstm_smooth  > self._lstm_thr) or \
                    (cnn_smooth   > self.CNN_THRESHOLD) or \
                    (ae_smooth    > self.ANOMALY_THRESHOLD * self._ae_thr_mul) or \
                    (fusion_score > 0.60)
        risk_level = "HIGH" if risk_high else "LOW"

        logger.debug(
            f"[FusionEngine] lstm={lstm_smooth:.3f}  anomaly={ae_smooth:.3f}  "
            f"fusion={fusion_score:.3f}  risk={risk_level}"
        )

        await self.bus.publish(Topics.PREDICTION_SCORE, round(lstm_smooth,  4))
        await self.bus.publish(Topics.ANOMALY_SCORE,    round(ae_smooth,    4))
        await self.bus.publish(Topics.FUSION_RISK, {
            "level":   risk_level,
            "score":   round(fusion_score, 4),
            "lstm":    round(lstm_smooth,  4),
            "cnn":     round(cnn_smooth,   4),
            "anomaly": round(ae_smooth,    4),
            "rule":    round(rule_score,   4),
            "timestamp": time.time(),
        })

        if risk_high:
            await self.bus.publish(Topics.PREDICTION_ALERT, {
                "score":        round(fusion_score, 4),
                "horizon_secs": HORIZON_SECS,
                "timestamp":    time.time(),
                "message": (
                    f"Pre-panic risk HIGH — "
                    f"LSTM={lstm_smooth:.2f}  anomaly={ae_smooth:.2f}  "
                    f"attack likely within {HORIZON_SECS // 60} min"
                ),
            })

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _infer_lstm(self, seq: np.ndarray) -> float:
        if self._lstm_model is None:
            return self._trend.predict(seq[0])
        if self._backend == "keras":
            try:
                return float(self._lstm_model.predict(seq, verbose=0)[0][0])
            except Exception as e:
                logger.warning(f"[FusionEngine] LSTM error: {e}")
                return 0.0
        return self._tflite_run(self._lstm_model, seq)

    def _infer_cnn(self, seq: np.ndarray) -> float:
        if self._cnn_model is None:
            # Fallback: use z-score spike detection as CNN proxy
            arr = seq[0]
            spikes = np.abs((arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8))
            return float(min(1.0, spikes.max(axis=0).mean() / 4.0))
        if self._backend == "keras":
            try:
                return float(self._cnn_model.predict(seq, verbose=0)[0][0])
            except Exception as e:
                logger.warning(f"[FusionEngine] CNN error: {e}")
                return 0.0
        return self._tflite_run(self._cnn_model, seq)

    def _infer_autoencoder(self, seq: np.ndarray) -> float:
        if self._ae_model is None:
            # Rule-based anomaly: z-score of current window vs its own mean
            arr = seq[0]
            z = np.abs((arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8))
            raw_error = float(z.mean())
            return min(1.0, raw_error / 3.0)   # normalise: z=3 → full anomaly

        if self._backend == "keras":
            try:
                reconstructed = self._ae_model.predict(seq, verbose=0)
                mse = float(np.mean((seq - reconstructed) ** 2))
            except Exception as e:
                logger.warning(f"[FusionEngine] AE error: {e}")
                return 0.0
        else:
            reconstructed = self._tflite_run(self._ae_model, seq, output_shape=seq.shape)
            mse = float(np.mean((seq - reconstructed) ** 2))

        # Normalise against learned threshold: error/threshold capped at 1
        thr = self._ae_threshold or 0.05
        return min(1.0, mse / (thr * self._ae_thr_mul))

    def _tflite_run(self, interp, seq: np.ndarray,
                    output_shape: tuple | None = None) -> float | np.ndarray:
        try:
            inp_det  = interp.get_input_details()[0]
            out_det  = interp.get_output_details()[0]
            interp.set_tensor(inp_det["index"], seq)
            interp.invoke()
            result = interp.get_tensor(out_det["index"])
            if output_shape:
                return result.reshape(output_shape)
            return float(result[0][0])
        except Exception as e:
            logger.warning(f"[FusionEngine] TFLite error: {e}")
            return 0.0 if output_shape is None else np.zeros(output_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_bilstm(data_path: str, out_path: str, epochs: int = 50, batch_size: int = 32):
    """
    Train Bi-LSTM + Attention on labeled episode data.

    CSV columns: timestamp, <13 feature cols>, label
        label = 1 if panic occurs within HORIZON_SECS of window end, else 0

    Steps:
      1. Load + normalize features
      2. Build 60s sliding windows with 50% overlap
      3. Label each window (future panic within horizon)
      4. Train with class weighting + early stopping
      5. Save .keras + export .tflite
    """
    import pandas as pd
    import pickle
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from ml.sequence_buffer import STEP

    df = pd.read_csv(data_path)
    X_raw = df[FeatureVector.FEATURE_NAMES].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.float32)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    scaler_path = out_path.replace(".keras", "_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Sliding windows with 50% overlap
    X_wins, y_wins = [], []
    for i in range(0, len(X_sc) - SEQ_LEN - HORIZON_SECS, STEP):
        X_wins.append(X_sc[i: i + SEQ_LEN])
        future = y_raw[i + SEQ_LEN: i + SEQ_LEN + HORIZON_SECS]
        y_wins.append(1.0 if future.max() > 0 else 0.0)

    X_arr = np.array(X_wins, dtype=np.float32)
    y_arr = np.array(y_wins, dtype=np.float32)
    logger.info(f"[Train LSTM] windows={len(X_arr)}  panic_ratio={y_arr.mean():.3f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=42)

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    cw = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}

    model = build_bilstm_attention()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, class_weight=cw,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                out_path, save_best_only=True, monitor="val_auc", mode="max"),
        ],
    )

    # TFLite export
    tflite_path = out_path.replace(".keras", ".tflite")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())
    logger.info(f"[Train LSTM] saved: {out_path}  {tflite_path}")


def train_autoencoder(data_path: str, out_path: str, epochs: int = 50, batch_size: int = 32):
    """
    Train LSTM Autoencoder on NORMAL physiology data only (no panic labels needed).

    Steps:
      1. Load + normalize normal data
      2. Build sliding windows
      3. Train autoencoder to reconstruct normal patterns
      4. Compute 95th-percentile reconstruction error as anomaly threshold
      5. Save model + threshold
    """
    import pandas as pd
    import pickle
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from ml.sequence_buffer import STEP

    df = pd.read_csv(data_path)
    X_raw = df[FeatureVector.FEATURE_NAMES].values.astype(np.float32)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    X_wins = [X_sc[i: i + SEQ_LEN]
              for i in range(0, len(X_sc) - SEQ_LEN, STEP)]
    X_arr  = np.array(X_wins, dtype=np.float32)
    logger.info(f"[Train AE] windows={len(X_arr)}")

    autoencoder, _ = build_autoencoder()
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_arr, X_arr,
        epochs=epochs, batch_size=batch_size, validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True),
        ],
    )
    autoencoder.save(out_path)

    # Compute anomaly threshold = 95th percentile of training reconstruction errors
    reconstructed = autoencoder.predict(X_arr, verbose=0)
    errors = np.mean((X_arr - reconstructed) ** 2, axis=(1, 2))
    threshold = float(np.percentile(errors, 95))

    thr_path = out_path.replace(".keras", "_threshold.pkl")
    with open(thr_path, "wb") as f:
        pickle.dump(threshold, f)

    # TFLite export
    tflite_path = out_path.replace(".keras", ".tflite")
    conv = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())

    logger.info(f"[Train AE] threshold={threshold:.5f}  saved: {out_path}  {tflite_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PanicGuard — AI model training")
    parser.add_argument("--train",      choices=["lstm", "autoencoder"], required=True)
    parser.add_argument("--data",       required=True,  help="Path to CSV data file")
    parser.add_argument("--out",        required=True,  help="Output .keras model path")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.train == "lstm":
        train_bilstm(args.data, args.out, args.epochs, args.batch_size)
    elif args.train == "autoencoder":
        train_autoencoder(args.data, args.out, args.epochs, args.batch_size)
