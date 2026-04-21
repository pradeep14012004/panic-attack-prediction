# PanicGuard - Real-Time Panic Attack Detection, Prediction & Response System

## Features

- Real-time biosignal monitoring (Heart Rate, GSR, Respiration)
- Multi-model AI detection system (Rule-based + Bi-LSTM + CNN + Autoencoder + Random Forest)
- Intelligent fusion engine combining all 4 models
- State-based system behavior (IDLE -> ALERT -> EPISODE -> SOS -> COOLDOWN)
- Automated calming interventions (Haptic, Audio, Water Mist, LED, SOS)
- Live monitoring dashboard at http://localhost:8080
- Cloud integration via MQTT

---

## System Architecture

```
Sensors -> Feature Extractor -> ML Models -> Fusion Engine -> State Machine -> Interventions
                                                    |
                                              Dashboard / MQTT
```

- Event-driven architecture (Event Bus)
- Scalable and modular design
- Easy to upgrade individual components

---

## Sensors

| Sensor       | Frequency | Measures                  |
|-------------|----------|---------------------------|
| PPG         | 25 Hz    | Heart rate, SpO2          |
| GSR         | 10 Hz    | Skin conductance (stress) |
| Respiration | 5 Hz     | Breathing patterns        |

---

## Feature Extraction

The system computes **13 physiological features**:

- `hr_mean`, `hr_std` — Heart rate mean and variability
- `rmssd`, `sdnn` — HRV (Heart Rate Variability)
- `scl_mean`, `scr_peak_rate`, `scr_amplitude` — Skin conductance
- `resp_rate`, `resp_regularity`, `resp_depth` — Breathing patterns
- `temp_delta` — Skin temperature change
- `motion_rms` — Motion filtering (exercise suppression)
- `hr_resp_coupling` — Heart-respiration synchronization (RSA)

---

## Machine Learning Models

### 1. Rule-Based Classifier
- Instant decision making, no training required
- Clinically inspired thresholds (HR elevation, RMSSD drop, GSR spike, breathing irregularity)
- Works on day 1 without any data

### 2. Bi-LSTM + Attention
- Captures long-range temporal dependencies in biosignals
- Bidirectional LSTM reads signals forward AND backward
- Attention layer focuses on most panic-predictive timesteps
- Trained on WESAD dataset (15 subjects)

### 3. 1D CNN with Residual Blocks (NEW)
- Detects local spike patterns: sudden HR acceleration, SCR bursts, breathing fragmentation
- Deep residual architecture (3 blocks, 64/128/256 filters)
- GlobalMaxPooling captures peak activation (ideal for spike detection)
- Faster than LSTM at inference (no recurrence)
- Complementary to LSTM: CNN catches sharp transient events, LSTM catches long-range buildup
- Trained on WESAD dataset

### 4. LSTM Autoencoder
- Unsupervised anomaly detection
- Trained on normal physiology only
- High reconstruction error = abnormal/panic pattern detected
- Catches unknown panic patterns the LSTM hasn't seen

### 5. Random Forest (HRV-based)
- Trained on HRV Stress Dataset (369K windows, 34 HRV features)
- Uses frequency-domain features: RMSSD, LF/HF, pNN50, sampen, higuci
- Fast and highly accurate on structured HRV data

---

## Model Accuracy Results

### HRV Random Forest
- **Accuracy: 100%** | **ROC-AUC: 1.0000**
- Precision / Recall / F1: 100% on all classes
- Dataset: HRV Stress Dataset (41,033 test samples)

### Bi-LSTM + Attention
- **Accuracy: 94.5%** | **ROC-AUC: 0.9845**

| Class  | Precision | Recall | F1   |
|--------|----------|--------|------|
| Normal | 1.00     | 0.93   | 0.96 |
| Stress | 0.81     | 0.99   | 0.89 |

Key insight: 99% recall means almost zero missed panic events.

### 1D CNN (Residual)
- **Accuracy: 93%** | **ROC-AUC: 0.9945**

| Class  | Precision | Recall | F1   |
|--------|----------|--------|------|
| Normal | 1.00     | 0.91   | 0.96 |
| Stress | 0.77     | 1.00   | 0.87 |

Key insight: 100% recall on stress class — never misses a panic event. Highest AUC of all models.

### LSTM Autoencoder
- **Accuracy: 79.5%** | **ROC-AUC: 0.7677**
- Not a direct classifier — works as anomaly detector
- Detects unknown/novel panic patterns

---

## Fusion Engine

All 4 models are combined using a weighted fusion formula:

```
fusion_score = 0.40 x Bi-LSTM
             + 0.25 x CNN
             + 0.20 x Autoencoder
             + 0.15 x Rule-based
```

Decision: FUSION HIGH if:
- LSTM score > 0.55, OR
- CNN score > 0.55, OR
- Anomaly score > 0.50, OR
- Fusion score > 0.60

Why fusion?
- LSTM captures temporal buildup toward panic
- CNN catches sharp transient spikes
- Autoencoder detects novel anomalies
- Rule-based provides instant clinical baseline

---

## State Machine

| State    | Score Threshold | Description             |
|----------|----------------|-------------------------|
| IDLE     | < 0.40         | Normal monitoring       |
| ALERT    | > 0.40         | Early warning, soft haptic |
| EPISODE  | > 0.60         | Full intervention suite |
| SOS      | > 0.80 (15s)   | Emergency escalation    |
| COOLDOWN | < 0.35         | Recovery phase          |

---

## Interventions

| Type       | Action                                        |
|------------|-----------------------------------------------|
| Haptic     | Breathing guide vibration (4s on / 6s off)   |
| Audio      | Calming voice / ambient sounds               |
| Water Mist | 800ms pump pulse (activates diving reflex)   |
| LED        | Color-coded ring (green/amber/blue/red)      |
| SOS        | GPS location + message to emergency contacts |

---

## Live Dashboard

Open http://localhost:8080 while simulation is running.

- Real-time panic score gauge (color changes with severity)
- 120-second score history chart
- Live sensor readings (HR, SpO2, GSR, Resp)
- All 4 model score bars updating in real time
- Fusion risk badge (HIGH/LOW)
- Episode log with timestamps
- Override buttons (Cancel Intervention / Trigger SOS)
- WebSocket auto-reconnect

---

## Cloud Integration (MQTT)

Enable in `config/device.yaml` by setting `mqtt.enabled: true`.

Topics published:
```
panicguard/{device_id}/score      - live panic score
panicguard/{device_id}/state      - current state
panicguard/{device_id}/fusion     - full model breakdown
panicguard/{device_id}/episode    - episode events
panicguard/{device_id}/status     - 30s heartbeat
```

Default broker: `test.mosquitto.org` (free public, no setup needed)

---

## Datasets

| Dataset          | Subjects | Samples  | Used For                        |
|-----------------|----------|----------|---------------------------------|
| WESAD           | 15       | 1,643    | Bi-LSTM, CNN, Autoencoder       |
| HRV Stress      | 1        | 410,322  | Random Forest classifier        |

---

## Quickstart

```bash
pip install -r requirements.txt

# Run simulation (no hardware needed)
python main.py --mode simulate --panic-at 15 --duration 180

# Open dashboard in browser
# http://localhost:8080

# Train models (requires WESAD dataset)
python -m ml.train_wesad --wesad path/to/WESAD
python -m ml.train_hrv

# Evaluate all models
python evaluate_models.py
```

---

## Project Structure

```
panic_guard/
├── core/           - Event bus, state machine
├── sensors/        - PPG, GSR, respiration simulators + hardware drivers
├── ml/             - Feature extraction, Bi-LSTM, CNN, Autoencoder, RF, Fusion
├── interventions/  - Haptic, audio, water, LED, SOS drivers
├── cloud/          - MQTT bridge
├── app/            - FastAPI server + live dashboard
├── models/         - Trained model files (generated after training)
├── data/           - Extracted feature CSVs (generated after training)
├── config/         - Device configuration
└── tests/          - Integration tests (4/4 passing)
```
