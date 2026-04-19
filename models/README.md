# Model files are excluded from git (too large).
# After cloning, train models by running:
#
#   python -m ml.train_wesad --wesad e:/WESAD
#   python -m ml.train_hrv
#
# This will generate:
#   models/bilstm.keras
#   models/bilstm_scaler.pkl
#   models/autoencoder.keras
#   models/autoencoder_threshold.pkl
#   models/hrv_panic_classifier.pkl
