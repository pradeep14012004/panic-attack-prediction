import sys, os, warnings
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from ml.inference import FeatureVector
from ml.sequence_buffer import SEQ_LEN

print("\n" + "="*55)
print("  PanicGuard — Model Accuracy Report")
print("="*55)

# ── 1. HRV RandomForest ───────────────────────────────────
print("\n[1] HRV RandomForest (sklearn)")
print("-"*40)
clf  = pickle.load(open('models/hrv_panic_classifier.pkl','rb'))
test = pd.read_csv(r'c:/Users/DELL/Downloads/archive (1)/hrv dataset/data/final/test.csv')
test['label'] = (test['condition'] != 'no stress').astype(int)
feat = [c for c in test.columns if c not in ('condition','label','datasetId')]
X, y = test[feat].values, test['label'].values
yp   = clf.predict(X)
ypr  = clf.predict_proba(X)[:,1]
print(f"  Accuracy : {(yp==y).mean()*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc_score(y, ypr):.4f}")
print(f"  Precision: {(yp[y==1]==1).mean()*100:.2f}%  (stress class)")
print(f"  Recall   : {(yp[y==1]==1).sum()/y.sum()*100:.2f}%  (stress class)")
print(classification_report(y, yp, target_names=['no stress','stress']))

# ── 2. Bi-LSTM + Attention ────────────────────────────────
print("\n[2] Bi-LSTM + Attention (WESAD dataset)")
print("-"*40)
df     = pd.read_csv('data/wesad_features.csv')
scaler = pickle.load(open('models/bilstm_scaler.pkl','rb'))
X_raw  = df[FeatureVector.FEATURE_NAMES].values.astype(np.float32)
y_raw  = df['label'].values.astype(np.float32)
X_sc   = scaler.transform(X_raw)
X_arr  = np.repeat(X_sc[:,np.newaxis,:], SEQ_LEN, axis=1).astype(np.float32)
_, X_val, _, y_val = train_test_split(X_arr, y_raw, test_size=0.2, stratify=y_raw, random_state=42)

lstm   = tf.keras.models.load_model('models/bilstm.keras')
y_prob = lstm.predict(X_val, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
print(f"  Accuracy : {(y_pred==y_val).mean()*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc_score(y_val, y_prob):.4f}")
print(classification_report(y_val.astype(int), y_pred, target_names=['normal','stress']))

# ── 3. LSTM Autoencoder ───────────────────────────────────
print("\n[3] LSTM Autoencoder — Anomaly Detection (WESAD)")
print("-"*40)
ae  = tf.keras.models.load_model('models/autoencoder.keras')
thr = pickle.load(open('models/autoencoder_threshold.pkl','rb'))

X_norm   = pd.read_csv('data/wesad_features_normal.csv')[FeatureVector.FEATURE_NAMES].values.astype(np.float32)
X_stress = df[df['label']==1][FeatureVector.FEATURE_NAMES].values.astype(np.float32)
Xn = np.repeat(scaler.transform(X_norm)[:,np.newaxis,:],   SEQ_LEN, axis=1).astype(np.float32)
Xs = np.repeat(scaler.transform(X_stress)[:,np.newaxis,:], SEQ_LEN, axis=1).astype(np.float32)

en = np.mean((Xn - ae.predict(Xn, verbose=0))**2, axis=(1,2))
es = np.mean((Xs - ae.predict(Xs, verbose=0))**2, axis=(1,2))
ya = np.concatenate([np.zeros(len(en)), np.ones(len(es))])
ea = np.concatenate([en, es])
yp = (ea > thr).astype(int)

print(f"  Threshold: {thr:.6f}")
print(f"  Accuracy : {(yp==ya).mean()*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc_score(ya, ea):.4f}")
print(classification_report(ya.astype(int), yp, target_names=['normal','stress']))

print("="*55)
print("  Summary")
print("="*55)
print(f"  HRV RandomForest  Accuracy: {(clf.predict(test[feat].values)==test['label'].values).mean()*100:.1f}%   AUC: {roc_auc_score(test['label'].values, clf.predict_proba(test[feat].values)[:,1]):.4f}")
print(f"  Bi-LSTM+Attention Accuracy: {(y_pred==y_val).mean()*100:.1f}%   AUC: {roc_auc_score(y_val, y_prob):.4f}")
print(f"  LSTM Autoencoder  Accuracy: {(yp==ya).mean()*100:.1f}%   AUC: {roc_auc_score(ya, ea):.4f}")
print("="*55)
