# 🧠 NeuroCalm – Real-Time Panic Detection & Response System



## 🚀 Features

- 📡 Real-time biosignal monitoring (Heart Rate, GSR, Respiration)
- 🤖 Multi-model AI detection system
- 🔗 Intelligent fusion engine
- 🔄 State-based system behavior
- 🌿 Automated calming interventions
- 📊 Live monitoring dashboard
- ☁️ Cloud integration via MQTT

---

## 🧩 System Architecture

✔ Event-driven architecture (Event Bus)  
✔ Scalable & modular design  
✔ Easy to upgrade individual components  

---

## 📡 Sensors

| Sensor      | Frequency | Measures                  |
|------------|----------|---------------------------|
| PPG        | 25 Hz    | Heart rate, SpO2          |
| GSR        | 10 Hz    | Skin conductance (stress) |
| Respiration| 5 Hz     | Breathing patterns        |

---

## 🧮 Feature Extraction

The system computes **13 physiological features**, including:

- HRV (Heart Rate Variability)  
- RMSSD, SDNN  
- LF/HF ratio  
- Skin conductance level  
- Breathing rate & irregularity  
- Motion filtering  
- Heart–respiration coupling  

---

## 🤖 Machine Learning Models

### 1. Rule-Based Model
- Instant decision making  
- No training required  
- Clinically inspired thresholds  

### 2. Bi-LSTM + Attention
- Captures temporal dependencies  
- High recall for panic detection  
- Trained on WESAD dataset  

### 3. Autoencoder
- Anomaly detection model  
- Trained on normal data only  
- Detects unseen panic patterns  

### 4. Random Forest (HRV-based)
- Uses HRV features  
- Fast and highly accurate on structured data  

---

## 📊 Model Accuracy Results

### 🔹 HRV Random Forest
- **Accuracy:** 100%  
- **ROC-AUC:** 1.0000  
- **Precision / Recall / F1:** 100%  

📌 Dataset: HRV Stress Dataset (41,033 samples)

💡 Insight:  
Clean feature separation (RMSSD, LF/HF, pNN50) leads to near-perfect classification.

⚠️ Note:  
Performance may vary in real-world noisy conditions.

---

### 🔹 Bi-LSTM + Attention
- **Accuracy:** 94.5%  
- **ROC-AUC:** 0.9845  

| Class  | Precision | Recall | F1   |
|--------|----------|--------|------|
| Normal | 1.00     | 0.93   | 0.96 |
| Stress | 0.81     | 0.99   | 0.89 |

💡 Key Insight:
- 99% recall → almost zero missed panic events  
- Designed for safety-first detection  

---

### 🔹 LSTM Autoencoder
- **Accuracy:** 79.5%  
- **ROC-AUC:** 0.7677  

| Class  | Precision | Recall |
|--------|----------|--------|
| Normal | 0.83     | 0.93   |
| Stress | 0.57     | 0.31   |

💡 Insight:
- Not a classifier  
- Works as anomaly detector  
- Detects unknown panic patterns  

---

## 🔗 Fusion Engine

### Why Fusion?
- LSTM → temporal learning  
- Autoencoder → anomaly detection  
- Rule-based → fast baseline  

👉 Result: **robust and reliable prediction**

---

## 🔄 State Machine

| State    | Description        |
|----------|--------------------|
| IDLE     | Normal condition   |
| ALERT    | Early warning      |
| EPISODE  | Panic detected     |
| SOS      | Emergency alert    |
| COOLDOWN | Recovery phase     |

---

## 🌿 Interventions

| Type       | Action                          |
|------------|---------------------------------|
| Haptic     | Guided breathing vibrations     |
| Audio      | Calming sounds                  |
| Water Mist | Physiological calming effect    |
| LED        | Visual status indication        |
| SOS        | Emergency contact alert         |

---

## 📊 Dashboard

- Real-time panic score  
- Sensor readings  
- Model outputs  
- Episode logs  
- WebSocket-based live updates  

---

## ☁️ Cloud Integration (MQTT Topics)

---

## 🧪 Datasets Used

- WESAD Dataset  
- HRV Stress Dataset  

---

python main.py

http://localhost:8080
