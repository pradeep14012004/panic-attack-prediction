# PanicGuard — Panic Attack Detection & Intervention System

A modular, hardware-agnostic software stack for wearable panic attack detection.
Runs on Raspberry Pi, Jetson Nano, ESP32 (MicroPython), or any POSIX system in simulation mode.

## Architecture

```
panic_guard/
├── core/           # Event bus, state machine, data pipeline
├── sensors/        # Hardware-agnostic sensor drivers + simulators
├── ml/             # Feature extraction + TinyML inference engine
├── interventions/  # Haptic, audio, water-mist, LED, SOS actuators
├── cloud/          # BLE bridge, MQTT/HTTP sync, caregiver dashboard API
├── app/            # Companion app WebSocket server + REST API
├── config/         # Per-user baseline profiles
└── tests/          # Unit + integration tests with simulated signals
```

## Quickstart (simulation mode — no hardware needed)

```bash
pip install -r requirements.txt
python main.py --mode simulate --user demo_user
```

## Hardware integration

Set sensor backend in `config/device.yaml`:
- `backend: simulate`   — synthetic biosignals (development)
- `backend: max30102`   — PPG via I2C (Raspberry Pi)
- `backend: serial`     — UART stream from Arduino/ESP32
- `backend: ble`        — BLE characteristic polling

## Intervention outputs

| Intervention  | Interface       | Notes                              |
|---------------|-----------------|-------------------------------------|
| Haptic        | GPIO PWM / I2C  | DRV2605L or direct ERM motor        |
| Water mist    | GPIO relay      | Pump on GPIO pin, timed pulse       |
| Audio         | BLE A2DP / ALSA | Earbuds or onboard speaker          |
| LED           | GPIO / I2C      | NeoPixel or simple LED ring         |
| SOS           | HTTP / SMS      | Twilio or ntfy.sh push              |
