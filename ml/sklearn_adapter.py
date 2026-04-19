"""
ml/sklearn_adapter.py
---------------------
Bridges the live FeatureVector (13 biosignal features) to the 34-feature
HRV dataset format expected by the trained RandomForest classifier.

The HRV dataset features that overlap with FeatureVector are mapped directly.
The remaining features are estimated from the available signals using
physiologically-grounded approximations.

Feature mapping:
  FeatureVector       -> HRV dataset column
  hr_mean             -> HR
  rmssd               -> RMSSD
  sdnn                -> SDRR
  hr_std              -> SDRR (std of HR, proxy)
  scl_mean            -> (no direct match, used in composite)
  hr_resp_coupling    -> LF_HF (inverse proxy)
  resp_rate           -> (used in derived features)
"""

import math
import numpy as np
from ml.inference import FeatureVector


# Column order must exactly match the HRV dataset train.csv
HRV_FEATURE_COLS = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD",
    "HR", "pNN25", "pNN50", "SD1", "SD2",
    "KURT", "SKEW",
    "MEAN_REL_RR", "MEDIAN_REL_RR", "SDRR_REL_RR", "RMSSD_REL_RR",
    "SDSD_REL_RR", "SDRR_RMSSD_REL_RR", "KURT_REL_RR", "SKEW_REL_RR",
    "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU", "HF", "HF_PCT", "HF_NU",
    "TP", "LF_HF", "HF_LF", "sampen", "higuci",
]


def feature_vector_to_hrv(fv: FeatureVector) -> np.ndarray:
    """
    Convert a live FeatureVector to the 34-column HRV feature array
    expected by the sklearn RandomForest classifier.

    Directly mapped:
      HR, RMSSD, SDRR (from sdnn), SD1 (from rmssd), SD2 (from sdnn)

    Derived / approximated:
      MEAN_RR  = 60000 / HR  (ms)
      SDSD     ~ RMSSD  (SDSD ≈ RMSSD for stationary signals)
      pNN25/50 estimated from RMSSD using empirical relationship
      LF/HF    estimated from hr_resp_coupling
      sampen   estimated from motion_rms (proxy for signal complexity)
    """
    hr      = max(30.0, fv.hr_mean)
    rmssd   = fv.rmssd          # ms
    sdnn    = fv.sdnn            # ms
    rsa     = fv.hr_resp_coupling  # 0-1, high = parasympathetic

    # RR interval statistics (ms)
    mean_rr   = 60000.0 / hr
    median_rr = mean_rr * (1 + 0.01 * (fv.hr_std / max(hr, 1)))
    sdrr      = sdnn
    sdsd      = rmssd * 0.98    # SDSD ≈ RMSSD for short windows

    sdrr_rmssd = sdrr / max(rmssd, 0.1)

    # pNN25 / pNN50 — empirical relationship with RMSSD
    # From Mietus et al. (2002): pNN50 ≈ 0.5 * (RMSSD - 15) clamped 0-100
    pnn50 = max(0.0, min(100.0, 0.5 * (rmssd - 15.0)))
    pnn25 = max(0.0, min(100.0, pnn50 * 1.8))

    # Poincare plot: SD1 = RMSSD/sqrt(2), SD2 = sqrt(2*SDNN^2 - RMSSD^2/2)
    sd1 = rmssd / math.sqrt(2)
    sd2_sq = max(0.0, 2 * sdnn**2 - (rmssd**2) / 2)
    sd2 = math.sqrt(sd2_sq)

    # Distribution shape (approximate — calm signals are more Gaussian)
    kurt = 0.5 if rsa > 0.6 else 2.0
    skew = 0.1 if rsa > 0.6 else 0.8

    # Relative RR (normalised by mean)
    mean_rel_rr   = 0.0
    median_rel_rr = 0.0
    sdrr_rel      = sdrr / max(mean_rr, 1)
    rmssd_rel     = rmssd / max(mean_rr, 1)
    sdsd_rel      = sdsd  / max(mean_rr, 1)
    sdrr_rmssd_rel = sdrr_rel / max(rmssd_rel, 0.001)
    kurt_rel      = kurt
    skew_rel      = skew

    # Spectral features — estimated from RSA (hr_resp_coupling)
    # High RSA → high HF (parasympathetic), low LF/HF ratio
    # Low RSA  → high LF, high LF/HF (sympathetic dominance = stress)
    total_power = (sdnn ** 2) * 4   # rough estimate
    hf_pct  = max(5.0,  min(60.0, rsa * 60.0))
    lf_pct  = max(10.0, min(80.0, (1 - rsa) * 70.0 + 10.0))
    vlf_pct = max(5.0,  100.0 - hf_pct - lf_pct)

    tp  = max(100.0, total_power)
    hf  = tp * hf_pct  / 100.0
    lf  = tp * lf_pct  / 100.0
    vlf = tp * vlf_pct / 100.0

    lf_nu  = lf  / max(lf + hf, 1) * 100.0
    hf_nu  = hf  / max(lf + hf, 1) * 100.0
    lf_hf  = lf  / max(hf, 0.1)
    hf_lf  = hf  / max(lf, 0.1)

    # Non-linear features
    # Sample entropy: low during stress (more regular), high at rest
    sampen = max(0.1, min(3.0, 2.5 * rsa - 0.3 * fv.scr_peak_rate))
    # Higuchi fractal dimension: ~1.0-2.0, higher = more complex
    higuci = max(1.0, min(2.0, 1.0 + 0.5 * rsa))

    vec = np.array([
        mean_rr, median_rr, sdrr, rmssd, sdsd, sdrr_rmssd,
        hr, pnn25, pnn50, sd1, sd2,
        kurt, skew,
        mean_rel_rr, median_rel_rr, sdrr_rel, rmssd_rel,
        sdsd_rel, sdrr_rmssd_rel, kurt_rel, skew_rel,
        vlf, vlf_pct, lf, lf_pct, lf_nu, hf, hf_pct, hf_nu,
        tp, lf_hf, hf_lf, sampen, higuci,
    ], dtype=np.float32)

    return vec
