"""
main.py
-------
PanicGuard entry point.

Usage:
    python main.py --mode simulate --panic-at 15 --duration 60
    python main.py --mode hardware --config config/device.yaml
    python main.py --mode api-only
"""

import asyncio
import argparse
import yaml
from loguru import logger

from core.event_bus import EventBus
from core.state_machine import PanicStateMachine
from sensors.ppg import create_sensor
from ml.inference import FeatureExtractor, PanicClassifier
from ml.predictive_model import FusionEngine
from interventions.drivers import create_intervention_suite
from app.server import app as fastapi_app, init_server


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {path} not found, using defaults")
        return {}


async def run_simulation(panic_at: float = 15.0, duration: float = 90.0):
    logger.info("=" * 55)
    logger.info("  PanicGuard  --  Simulation Mode")
    logger.info(f"  Panic injected at t={panic_at}s  |  Duration={duration}s")
    logger.info("=" * 55)

    bus = EventBus()

    state_machine = PanicStateMachine(bus)

    # Use a shorter window in simulation so features appear quickly
    fex = FeatureExtractor(bus)
    fex.WINDOW_SECS = 10      # 10s window instead of 30s
    fex.PUBLISH_HZ  = 2       # publish features 2x/sec

    # Load trained sklearn model for rule+ML combined scoring
    classifier = PanicClassifier(
        bus,
        backend="sklearn",
        model_path="models/hrv_panic_classifier.pkl",
    )

    # Load trained Bi-LSTM + CNN + Autoencoder — all fully wired
    fusion_engine = FusionEngine(
        bus,
        lstm_path="models/bilstm.keras",
        autoencoder_path="models/autoencoder.keras",
        cnn_path="models/cnn.keras",
        scaler_path="models/bilstm_scaler.pkl",
    )
    interventions = create_intervention_suite(bus, {
        "haptic": {"backend": "simulate"},
        "audio":  {"backend": "simulate"},
        "water":  {"backend": "simulate"},
        "led":    {"backend": "simulate"},
        "sos":    {"backend": "simulate", "contacts": [
            {"name": "Emergency Contact", "ntfy_topic": "demo-sos"}
        ]},
    })

    sensors = [
        create_sensor("ppg",         "simulate", bus, inject_panic_at=panic_at),
        create_sensor("gsr",         "simulate", bus, inject_panic_at=panic_at),
        create_sensor("respiration", "simulate", bus, inject_panic_at=panic_at),
    ]

    # Score printer
    async def print_score(topic, score):
        bar = "#" * int(score * 30)
        logger.info(f"  Score={score:.3f}  [{bar:<30}]  state={state_machine.ctx.current_state.value}")
    bus.subscribe("ml.panic_score", print_score)

    logger.info("Starting sensors and feature extractor ...")
    logger.info(f"Dashboard at http://localhost:8080  (open in browser)")

    init_server(bus)

    import uvicorn
    server_cfg = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8080,
                                log_level="warning", loop="asyncio")
    server = uvicorn.Server(server_cfg)

    async def serve_safe():
        try:
            await server.serve()
        except (OSError, SystemExit) as e:
            logger.warning(f"API server could not start (port busy?): {e}")
        except Exception:
            pass  # suppress CancelledError traceback on clean shutdown

    tasks = [asyncio.create_task(t) for t in [
        *[s.run() for s in sensors],
        fex.run(),
        serve_safe(),
    ]]

    await asyncio.sleep(duration)

    logger.info("=" * 55)
    logger.info("  Simulation complete — shutting down")
    logger.info(f"  Final state : {state_machine.ctx.current_state.value}")
    logger.info(f"  Peak score  : {state_machine.ctx.peak_score:.3f}")
    logger.info(f"  Episodes    : {state_machine.ctx.episode_count_today}")
    logger.info("=" * 55)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_hardware(config_path: str = "config/device.yaml"):
    config = load_config(config_path)
    logger.info(f"Hardware mode  config={config_path}")

    bus = EventBus()
    state_machine = PanicStateMachine(bus)
    fex = FeatureExtractor(
        bus,
        baseline_temp=config.get("baseline", {}).get("temp", 33.5)
    )
    classifier = PanicClassifier(
        bus,
        backend=config.get("ml", {}).get("backend", "rules"),
        model_path=config.get("ml", {}).get("model_path"),
        baseline=config.get("baseline", {}),
    )
    pred_cfg = config.get("ml", {}).get("predictor", {})
    fusion_engine = FusionEngine(
        bus,
        lstm_path=pred_cfg.get("lstm_path"),
        autoencoder_path=pred_cfg.get("autoencoder_path"),
        cnn_path=pred_cfg.get("cnn_path"),
        scaler_path=pred_cfg.get("scaler_path"),
        baseline=config.get("baseline", {}),
    )
    interventions = create_intervention_suite(bus, config.get("interventions", {}))

    # Optional MQTT cloud bridge
    mqtt_cfg = config.get("mqtt", {})
    mqtt_bridge = None
    if mqtt_cfg.get("enabled", False):
        from cloud.mqtt_bridge import MQTTBridge
        mqtt_bridge = MQTTBridge(
            bus,
            broker=mqtt_cfg.get("broker", "test.mosquitto.org"),
            port=mqtt_cfg.get("port", 1883),
            device_id=mqtt_cfg.get("device_id", "panicguard-01"),
            username=mqtt_cfg.get("username"),
            password=mqtt_cfg.get("password"),
            tls=mqtt_cfg.get("tls", False),
        )
        logger.info(f"[MQTT] bridge enabled -> {mqtt_cfg.get('broker')}")

    sensor_backend = config.get("sensor_backend", "simulate")
    sensors = [
        create_sensor("ppg",         sensor_backend, bus),
        create_sensor("gsr",         sensor_backend, bus),
        create_sensor("respiration", sensor_backend, bus),
    ]

    init_server(bus)

    import uvicorn
    server_cfg = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8080,
                                log_level="warning", loop="asyncio")
    server = uvicorn.Server(server_cfg)

    tasks = [s.run() for s in sensors] + [fex.run(), server.serve()]
    if mqtt_bridge:
        tasks.append(mqtt_bridge.start())

    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="PanicGuard")
    parser.add_argument("--mode",      choices=["simulate", "hardware", "api-only"],
                        default="simulate")
    parser.add_argument("--config",    default="config/device.yaml")
    parser.add_argument("--panic-at",  type=float, default=15.0,
                        help="Seconds after start to inject panic (simulate)")
    parser.add_argument("--duration",  type=float, default=90.0,
                        help="How long to run the simulation (seconds)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    import sys, io
    # Force stdout to UTF-8 so arrows/emoji in log messages don't crash on Windows
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    if args.mode == "simulate":
        asyncio.run(run_simulation(panic_at=args.panic_at, duration=args.duration))
    elif args.mode == "hardware":
        asyncio.run(run_hardware(args.config))
    elif args.mode == "api-only":
        import uvicorn
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
