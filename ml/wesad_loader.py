"""
ml/wesad_loader.py
──────────────────
Loads WESAD dataset (.pkl files) and extracts the 13 features used by
PanicGuard's ML pipeline, producing a labeled CSV ready for training.

WESAD wrist signals used:
  BVP  @ 64 Hz  → HR (bpm) via peak detection, RMSSD, SDNN
  EDA  @ 4 Hz   → SCL mean, SCR peak rate, SCR amplitude
  TEMP @ 4 Hz   → temp_delta (from subject baseline)
  ACC  @ 32 Hz  → motion_rms

Labels (WESAD protocol):
  1 = baseline  → label 0 (normal)
  2 = stress    → label 1 (panic-like arousal)
  3 = amusement → label 0
  4 = meditation→ label 0
  0, 6, 7       → skipped (transient / undefined)

Output CSV columns:
  timestamp, hr_mean, hr_std, rmssd, sdnn,
  scl_mean, scr_peak_rate, scr_amplitude,
  resp_rate, resp_regularity, resp_depth,
  temp_delta, motion_rms, hr_resp_coupling, label
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

# ── Sampling rates ────────────────────────────────────────────────────────────
BVP_FS   = 64    # Hz
EDA_FS   = 4     # Hz
TEMP_FS  = 4     # Hz
ACC_FS   = 32    # Hz
LABEL_FS = 700   # Hz (chest label sampling rate in WESAD)

# ── Window config (must match sequence_buffer.py) ─────────────────────────────
WIN_SEC  = 60    # feature window length in seconds
STEP_SEC = 30    # 50% overlap


def _hr_features(bvp: np.ndarray, fs: int = BVP_FS):
    """Extract HR mean, std, RMSSD, SDNN from BVP window."""
    bvp = bvp.flatten()
    # Detect systolic peaks
    min_dist = int(fs * 0.4)   # min 0.4s between beats (max 150 bpm)
    peaks, _ = find_peaks(bvp, distance=min_dist, height=np.percentile(bvp, 60))
    if len(peaks) < 4:
        return 75.0, 5.0, 40.0, 50.0

    rr_sec = np.diff(peaks) / fs          # RR intervals in seconds
    hr_vals = 60.0 / rr_sec               # instantaneous HR in bpm

    hr_mean = float(np.mean(hr_vals))
    hr_std  = float(np.std(hr_vals, ddof=1)) if len(hr_vals) > 1 else 0.0

    diffs = np.diff(rr_sec)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) * 1000   # ms
    sdnn  = float(np.std(rr_sec, ddof=1)) * 1000         # ms

    return (
        np.clip(hr_mean, 30, 220),
        np.clip(hr_std,  0,  50),
        np.clip(rmssd,   0, 200),
        np.clip(sdnn,    0, 200),
    )


def _eda_features(eda: np.ndarray):
    """Extract SCL mean, SCR peak rate (per min), SCR amplitude from EDA window."""
    eda = eda.flatten()
    if len(eda) < 4:
        return 2.5, 0.0, 0.0

    scl_mean = float(np.mean(eda))
    threshold = scl_mean + np.std(eda) * 0.5

    peaks, props = find_peaks(eda, height=threshold, distance=int(EDA_FS * 1.0))
    win_min = len(eda) / EDA_FS / 60.0
    scr_rate = len(peaks) / win_min if win_min > 0 else 0.0
    scr_amp  = float(np.mean(props["peak_heights"] - scl_mean)) if len(peaks) else 0.0

    return float(np.clip(scl_mean, 0, 30)), float(np.clip(scr_rate, 0, 20)), float(np.clip(scr_amp, 0, 15))


def _motion_rms(acc: np.ndarray) -> float:
    mag = np.sqrt((acc ** 2).sum(axis=1))
    return float(np.sqrt(np.mean(mag ** 2)))


def _hr_resp_coupling(hr_mean: float, resp_rate: float = 15.0) -> float:
    """Simplified RSA proxy: higher HR variability at respiratory frequency → higher coupling."""
    # Without a real respiration signal we use a physiological heuristic:
    # normal RSA is inversely related to HR (higher HR → lower parasympathetic)
    rsa = max(0.0, min(1.0, (100.0 - hr_mean) / 60.0))
    return round(rsa, 3)


def load_subject(pkl_path: str) -> pd.DataFrame:
    """
    Load one WESAD subject .pkl and return a DataFrame of feature windows.
    Each row = one 60s window at 1Hz feature rate.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist  = data["signal"]["wrist"]
    labels = data["label"]          # shape (N_chest_samples,) at 700 Hz

    bvp  = wrist["BVP"].flatten()   # 64 Hz
    eda  = wrist["EDA"].flatten()   # 4 Hz
    temp = wrist["TEMP"].flatten()  # 4 Hz
    acc  = wrist["ACC"]             # (N, 3) at 32 Hz

    total_sec = len(bvp) / BVP_FS
    temp_baseline = float(np.median(temp))

    rows = []
    for t_start in range(0, int(total_sec) - WIN_SEC, STEP_SEC):
        t_end = t_start + WIN_SEC

        # Slice each signal to the window
        bvp_win  = bvp [int(t_start * BVP_FS)  : int(t_end * BVP_FS)]
        eda_win  = eda [int(t_start * EDA_FS)   : int(t_end * EDA_FS)]
        temp_win = temp[int(t_start * TEMP_FS)  : int(t_end * TEMP_FS)]
        acc_win  = acc [int(t_start * ACC_FS)   : int(t_end * ACC_FS)]

        # Majority label for this window (from chest label signal at 700 Hz)
        lbl_win = labels[int(t_start * LABEL_FS): int(t_end * LABEL_FS)]
        valid   = lbl_win[lbl_win > 0]
        if len(valid) == 0:
            continue
        majority = int(np.bincount(valid).argmax())
        if majority not in (1, 2, 3, 4):
            continue

        # Map to binary: stress (2) → 1, everything else → 0
        label = 1 if majority == 2 else 0

        # Extract features
        hr_mean, hr_std, rmssd, sdnn = _hr_features(bvp_win)
        scl_mean, scr_rate, scr_amp  = _eda_features(eda_win)
        temp_delta = float(np.mean(temp_win)) - temp_baseline
        motion     = _motion_rms(acc_win) if len(acc_win) > 0 else 1.0
        hr_resp    = _hr_resp_coupling(hr_mean)

        rows.append({
            "timestamp":       float(t_start),
            "hr_mean":         round(hr_mean,  2),
            "hr_std":          round(hr_std,   2),
            "rmssd":           round(rmssd,    2),
            "sdnn":            round(sdnn,     2),
            "scl_mean":        round(scl_mean, 3),
            "scr_peak_rate":   round(scr_rate, 3),
            "scr_amplitude":   round(scr_amp,  3),
            "resp_rate":       15.0,          # wrist device has no resp sensor
            "resp_regularity": 0.9,
            "resp_depth":      0.7,
            "temp_delta":      round(temp_delta, 2),
            "motion_rms":      round(motion,  3),
            "hr_resp_coupling": hr_resp,
            "label":           label,
        })

    return pd.DataFrame(rows)


def build_dataset(wesad_root: str, out_csv: str):
    """
    Iterate all subject folders in wesad_root, extract features, save CSV.
    Also saves a separate normal-only CSV for autoencoder training.
    """
    root = Path(wesad_root)
    subjects = sorted(root.glob("S*/S*.pkl"))
    if not subjects:
        raise FileNotFoundError(f"No .pkl files found under {wesad_root}")

    all_dfs = []
    for pkl in subjects:
        print(f"  Loading {pkl.name} …", end=" ", flush=True)
        df = load_subject(str(pkl))
        print(f"{len(df)} windows  (stress={df['label'].sum()})")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"Dataset saved: {out_csv}  ({len(combined)} windows, stress_ratio={combined['label'].mean():.3f})")

    # Normal-only CSV for autoencoder
    normal_csv = out_csv.replace(".csv", "_normal.csv")
    combined[combined["label"] == 0].to_csv(normal_csv, index=False)
    print(f"Normal-only  : {normal_csv}  ({(combined['label']==0).sum()} windows)")

    return out_csv, normal_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wesad",  default="e:/WESAD",          help="WESAD root folder")
    p.add_argument("--out",    default="data/wesad_features.csv")
    args = p.parse_args()
    build_dataset(args.wesad, args.out)
