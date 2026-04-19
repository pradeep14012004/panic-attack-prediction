"""
ml/feature_extractor.py + ml/inference.py
──────────────────────────────────────────
Two-stage ML pipeline:

1. FeatureExtractor — consumes raw sensor samples from event bus,
   maintains rolling windows, computes clinical biosignal features
   (RMSSD, SCR peaks, respiratory irregularity, etc.)

2. PanicClassifier — runs on extracted feature vectors.
   Supports:
     - Rule-based classifier (no training needed, works on day 1)
     - scikit-learn model (loaded from .pkl)
     - TensorFlow Lite (loaded from .tflite — for Raspberry Pi / Jetson)

Features extracted (13 total):
    hr_mean, hr_std, rmssd, sdnn,
    scl_mean, scr_peak_rate, scr_amplitude_mean,
    resp_rate, resp_regularity, resp_depth,
    temp_delta,
    motion_rms,
    hr_resp_coupling   (physiological coherence)
"""

import asyncio
import time
import math
import collections
from dataclasses import dataclass
from loguru import logger

from core.event_bus import EventBus, Topics
from sensors.ppg import PPGSample, GSRSample, RespirationSample, IMUSample


# ── Feature vector ────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    timestamp: float
    hr_mean: float          # beats per minute
    hr_std: float           # BPM std dev over window
    rmssd: float            # root mean square successive differences (HRV)
    sdnn: float             # std dev of NN intervals
    scl_mean: float         # tonic skin conductance level (µS)
    scr_peak_rate: float    # skin conductance responses per minute
    scr_amplitude: float    # mean SCR amplitude (µS)
    resp_rate: float        # breaths per minute
    resp_regularity: float  # 0–1
    resp_depth: float       # 0–1
    temp_delta: float       # °C from personal baseline
    motion_rms: float       # RMS acceleration (g)
    hr_resp_coupling: float # 0–1, physiological coherence

    def to_list(self) -> list[float]:
        return [
            self.hr_mean, self.hr_std, self.rmssd, self.sdnn,
            self.scl_mean, self.scr_peak_rate, self.scr_amplitude,
            self.resp_rate, self.resp_regularity, self.resp_depth,
            self.temp_delta, self.motion_rms, self.hr_resp_coupling,
        ]

    FEATURE_NAMES = [
        "hr_mean", "hr_std", "rmssd", "sdnn",
        "scl_mean", "scr_peak_rate", "scr_amplitude",
        "resp_rate", "resp_regularity", "resp_depth",
        "temp_delta", "motion_rms", "hr_resp_coupling",
    ]


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Listens to all sensor topics.
    Every WINDOW_SECS seconds, computes a FeatureVector and publishes it.
    """
    WINDOW_SECS   = 30     # feature window length
    PUBLISH_HZ    = 1      # how often to emit features (1/sec)

    def __init__(self, bus: EventBus, baseline_temp: float = 33.5):
        self.bus = bus
        self._baseline_temp = baseline_temp

        # Rolling windows
        maxlen = int(self.WINDOW_SECS * 25)   # 25Hz max
        self._hr_window:   collections.deque = collections.deque(maxlen=maxlen)
        self._rr_window:   collections.deque = collections.deque(maxlen=100)
        self._gsr_window:  collections.deque = collections.deque(maxlen=int(self.WINDOW_SECS * 10))
        self._resp_window: collections.deque = collections.deque(maxlen=int(self.WINDOW_SECS * 5))
        self._imu_window:  collections.deque = collections.deque(maxlen=maxlen)
        self._temp_window: collections.deque = collections.deque(maxlen=60)

        self._prev_hr: float | None = None
        self._running = False

        bus.subscribe(Topics.PPG_RAW,         self._on_ppg)
        bus.subscribe(Topics.GSR_RAW,         self._on_gsr)
        bus.subscribe(Topics.RESPIRATION_RAW, self._on_resp)
        bus.subscribe(Topics.IMU_RAW,         self._on_imu)
        bus.subscribe(Topics.TEMP_RAW,        self._on_temp)

    async def _on_ppg(self, topic: str, sample: PPGSample):
        self._hr_window.append(sample.heart_rate)
        if self._prev_hr is not None:
            rr = 60.0 / sample.heart_rate  # RR interval in seconds
            self._rr_window.append(rr)
        self._prev_hr = sample.heart_rate

    async def _on_gsr(self, topic: str, sample: GSRSample):
        self._gsr_window.append(sample.conductance_us)

    async def _on_resp(self, topic: str, sample: RespirationSample):
        self._resp_window.append(sample)

    async def _on_imu(self, topic: str, sample: IMUSample):
        mag = math.sqrt(sample.accel_x**2 + sample.accel_y**2 + sample.accel_z**2)
        self._imu_window.append(mag)

    async def _on_temp(self, topic: str, temp_celsius: float):
        self._temp_window.append(temp_celsius)

    def compute(self) -> FeatureVector | None:
        if len(self._hr_window) < 10:
            return None

        hr_vals = list(self._hr_window)
        rr_vals = list(self._rr_window)
        gsr_vals = list(self._gsr_window)
        resp_vals = list(self._resp_window)
        imu_vals = list(self._imu_window) or [9.81]   # 1g if no IMU
        temp_vals = list(self._temp_window)

        # HRV features
        hr_mean = _mean(hr_vals)
        hr_std  = _std(hr_vals)
        if len(rr_vals) >= 2:
            rmssd = _rmssd(rr_vals) * 1000   # convert to ms
            sdnn  = _std(rr_vals) * 1000
        else:
            rmssd, sdnn = 40.0, 50.0         # healthy defaults

        # GSR features
        scl_mean = _mean(gsr_vals) if gsr_vals else 2.5
        scr_peak_rate, scr_amplitude = _scr_features(gsr_vals)

        # Respiration features
        if resp_vals:
            resp_rate        = _mean([r.rate_bpm    for r in resp_vals])
            resp_regularity  = _mean([r.regularity  for r in resp_vals])
            resp_depth       = _mean([r.depth        for r in resp_vals])
        else:
            resp_rate, resp_regularity, resp_depth = 14.0, 0.9, 0.7

        # Temperature delta from personal baseline
        temp_delta = (_mean(temp_vals) - self._baseline_temp) if temp_vals else 0.0

        # Motion RMS (detects walking, exercise — confounders)
        motion_rms = math.sqrt(_mean([v**2 for v in imu_vals]))

        # HR–respiration coupling (RSA — respiratory sinus arrhythmia)
        # High RSA = parasympathetic tone = calm. Low = sympathetic activation.
        hr_resp_coupling = _compute_rsa(hr_vals, resp_rate)

        return FeatureVector(
            timestamp=time.time(),
            hr_mean=round(hr_mean, 2),
            hr_std=round(hr_std, 2),
            rmssd=round(rmssd, 2),
            sdnn=round(sdnn, 2),
            scl_mean=round(scl_mean, 3),
            scr_peak_rate=round(scr_peak_rate, 3),
            scr_amplitude=round(scr_amplitude, 3),
            resp_rate=round(resp_rate, 2),
            resp_regularity=round(resp_regularity, 3),
            resp_depth=round(resp_depth, 3),
            temp_delta=round(temp_delta, 2),
            motion_rms=round(motion_rms, 3),
            hr_resp_coupling=round(hr_resp_coupling, 3),
        )

    async def run(self):
        self._running = True
        interval = 1.0 / self.PUBLISH_HZ
        while self._running:
            await asyncio.sleep(interval)
            fv = self.compute()
            if fv:
                await self.bus.publish(Topics.FEATURES, fv)

    def stop(self):
        self._running = False


# ── Panic Classifier ──────────────────────────────────────────────────────────

class PanicClassifier:
    """
    Receives FeatureVectors, outputs panic score 0.0–1.0.

    Backends:
        'rules'    — explainable rule-based classifier (use for prototyping)
        'sklearn'  — scikit-learn model loaded from .pkl
        'tflite'   — TensorFlow Lite model (.tflite) for MCU/Pi
    """
    def __init__(self, bus: EventBus, backend: str = "rules",
                 model_path: str | None = None,
                 baseline: dict | None = None):
        self.bus = bus
        self._backend = backend
        self._model = None
        self._baseline = baseline or {}   # personalised offsets

        if backend == "sklearn":
            self._load_sklearn(model_path)
        elif backend == "tflite":
            self._load_tflite(model_path)

        # Smoothing buffer — prevents jitter from triggering interventions
        self._score_buffer: collections.deque = collections.deque(maxlen=5)

        bus.subscribe(Topics.FEATURES, self._on_features)

    def _load_sklearn(self, path: str):
        import pickle
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info(f"[Classifier] sklearn model loaded from {path}")

    def _load_tflite(self, path: str):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        self._model = tflite.Interpreter(model_path=path)
        self._model.allocate_tensors()
        logger.info(f"[Classifier] TFLite model loaded from {path}")

    async def _on_features(self, topic: str, fv: FeatureVector):
        score = self._infer(fv)
        self._score_buffer.append(score)
        smoothed = _mean(list(self._score_buffer))
        logger.debug(f"[Classifier] score={smoothed:.3f}  raw={score:.3f}  "
                     f"hr={fv.hr_mean:.0f}bpm  rmssd={fv.rmssd:.0f}ms  "
                     f"scl={fv.scl_mean:.2f}µS")
        await self.bus.publish(Topics.PANIC_SCORE, round(smoothed, 4))

    def _infer(self, fv: FeatureVector) -> float:
        if self._backend == "rules":
            return self._rule_based(fv)
        elif self._backend == "sklearn":
            return self._sklearn_infer(fv)
        elif self._backend == "tflite":
            return self._tflite_infer(fv)
        return 0.0

    def _rule_based(self, fv: FeatureVector) -> float:
        """
        Clinically-informed rule-based scorer.
        Each criterion contributes a weight. Sum is normalised to 0–1.
        Transparent, debuggable, works without any training data.

        References:
          - Tiller et al., Frontiers in Psychology (2014) — HRV during panic
          - Can et al., J. Biomedical Informatics (2020) — GSR thresholds
          - Barnett & Gotlib (1988) — respiratory rate in panic disorder
        """
        score = 0.0

        # HR elevation (weight: 0.25)
        personal_hr_baseline = self._baseline.get("hr_resting", 70.0)
        hr_excess = max(0, fv.hr_mean - personal_hr_baseline)
        score += 0.25 * min(1.0, hr_excess / 45.0)   # 45 BPM excess → full contribution

        # HRV suppression — low RMSSD = sympathetic dominance (weight: 0.25)
        personal_rmssd_baseline = self._baseline.get("rmssd_resting", 50.0)
        rmssd_drop = max(0, personal_rmssd_baseline - fv.rmssd)
        score += 0.25 * min(1.0, rmssd_drop / personal_rmssd_baseline)

        # GSR elevation (weight: 0.20)
        personal_scl_baseline = self._baseline.get("scl_resting", 2.5)
        scl_excess = max(0, fv.scl_mean - personal_scl_baseline)
        score += 0.20 * min(1.0, scl_excess / 8.0)

        # SCR peak rate (weight: 0.10)
        score += 0.10 * min(1.0, fv.scr_peak_rate / 3.0)  # 3 peaks/min → full

        # Rapid, irregular breathing (weight: 0.15)
        resp_score = 0.0
        if fv.resp_rate > 18:
            resp_score += min(1.0, (fv.resp_rate - 18) / 14)  # 14bpm excess → full
        resp_score += (1.0 - fv.resp_regularity)
        score += 0.15 * (resp_score / 2.0)

        # Motion gate: suppress score if heavy exercise detected
        if fv.motion_rms > 12.0:     # ~1.2g sustained = jogging
            score *= 0.4

        return max(0.0, min(1.0, score))

    def _sklearn_infer(self, fv: FeatureVector) -> float:
        import numpy as np
        try:
            # Try HRV adapter first (34-feature RF model)
            from ml.sklearn_adapter import feature_vector_to_hrv
            x = feature_vector_to_hrv(fv).reshape(1, -1)
        except Exception:
            x = np.array([fv.to_list()], dtype=float)
        try:
            proba = self._model.predict_proba(x)[0]
            return float(proba[1])
        except Exception as e:
            logger.warning(f"[Classifier] sklearn error: {e}")
            return 0.0

    def _tflite_infer(self, fv: FeatureVector) -> float:
        import numpy as np
        inp  = self._model.get_input_details()[0]
        outp = self._model.get_output_details()[0]
        x = np.array([fv.to_list()], dtype=np.float32)
        self._model.set_tensor(inp["index"], x)
        self._model.invoke()
        result = self._model.get_tensor(outp["index"])
        return float(result[0][0])


# ── Signal processing helpers ─────────────────────────────────────────────────

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def _rmssd(rr_intervals: list[float]) -> float:
    if len(rr_intervals) < 2:
        return 0.0
    diffs = [rr_intervals[i+1] - rr_intervals[i] for i in range(len(rr_intervals)-1)]
    return math.sqrt(_mean([d**2 for d in diffs]))

def _scr_features(gsr_vals: list[float]) -> tuple[float, float]:
    """Detect skin conductance response peaks (phasic component)."""
    if len(gsr_vals) < 10:
        return 0.0, 0.0
    m = _mean(gsr_vals)
    threshold = m + _std(gsr_vals) * 0.5
    peaks, amplitudes = [], []
    above = False
    for i, v in enumerate(gsr_vals):
        if v > threshold and not above:
            peaks.append(i)
            amplitudes.append(v - m)
            above = True
        elif v <= threshold:
            above = False
    window_minutes = len(gsr_vals) / 10.0 / 60.0
    peak_rate = (len(peaks) / window_minutes) if window_minutes > 0 else 0.0
    amplitude = _mean(amplitudes) if amplitudes else 0.0
    return round(peak_rate, 3), round(amplitude, 3)

def _compute_rsa(hr_vals: list[float], resp_rate: float) -> float:
    """
    Simplified respiratory sinus arrhythmia index.
    Real RSA uses spectral analysis at respiratory frequency band.
    This approximation checks HR variability synchronised to breathing cycle.
    """
    if len(hr_vals) < 20 or resp_rate <= 0:
        return 0.5
    # Breathing cycle length in samples (at 25Hz)
    cycle = int(25 * (60.0 / resp_rate))
    if cycle >= len(hr_vals):
        return 0.5
    # Compute HR range within each breathing cycle
    ranges = []
    for i in range(0, len(hr_vals) - cycle, cycle):
        chunk = hr_vals[i:i + cycle]
        ranges.append(max(chunk) - min(chunk))
    rsa = _mean(ranges)
    # Normalise: healthy RSA is ~5–15 BPM. Low RSA (<3) = low parasympathetic.
    return max(0.0, min(1.0, rsa / 15.0))
