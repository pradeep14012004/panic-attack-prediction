"""
core/event_bus.py
─────────────────
Async publish-subscribe event bus.
All modules communicate through here — sensors publish, ML subscribes,
interventions subscribe to ML output. No direct module coupling.

Usage:
    bus = EventBus()
    bus.subscribe("sensor.ppg", my_handler)
    await bus.publish("sensor.ppg", payload)
"""

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine
from loguru import logger


class EventBus:
    def __init__(self):
        # topic -> list of async handler coroutines
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[tuple[str, Any]] = []
        self._max_history = 500

    def subscribe(self, topic: str, handler: Callable[..., Coroutine]):
        """Register an async handler for a topic. Supports wildcards: 'sensor.*'"""
        self._subscribers[topic].append(handler)
        logger.debug(f"[EventBus] {handler.__name__} subscribed to '{topic}'")

    def unsubscribe(self, topic: str, handler: Callable):
        self._subscribers[topic] = [
            h for h in self._subscribers[topic] if h != handler
        ]

    async def publish(self, topic: str, payload: Any = None):
        """Fire all handlers for the topic (and matching wildcards)."""
        self._history.append((topic, payload))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        handlers = list(self._subscribers.get(topic, []))

        # Wildcard support: 'sensor.*' matches 'sensor.ppg', 'sensor.gsr', etc.
        prefix = topic.rsplit(".", 1)[0] + ".*"
        handlers += self._subscribers.get(prefix, [])
        handlers += self._subscribers.get("*", [])

        if not handlers:
            return

        await asyncio.gather(
            *[self._safe_call(h, topic, payload) for h in handlers],
            return_exceptions=True,
        )

    async def _safe_call(self, handler: Callable, topic: str, payload: Any):
        try:
            await handler(topic, payload)
        except Exception as e:
            logger.error(f"[EventBus] Handler '{handler.__name__}' raised: {e}")

    def get_history(self, topic_filter: str | None = None) -> list:
        if topic_filter:
            return [(t, p) for t, p in self._history if t == topic_filter]
        return list(self._history)


# Topics used across the system
class Topics:
    # Sensor raw readings
    PPG_RAW         = "sensor.ppg"
    GSR_RAW         = "sensor.gsr"
    RESPIRATION_RAW = "sensor.respiration"
    TEMP_RAW        = "sensor.temperature"
    IMU_RAW         = "sensor.imu"

    # Processed features (output of feature extractor)
    FEATURES        = "ml.features"

    # Inference result
    PANIC_SCORE       = "ml.panic_score"        # float 0.0–1.0  (current state)
    PANIC_STATE       = "ml.panic_state"        # PanicState enum

    # Predictive model output (Bi-LSTM + Attention pre-panic forecasting)
    PREDICTION_SCORE  = "ml.prediction_score"   # float 0.0–1.0  (upcoming panic probability)
    PREDICTION_ALERT  = "ml.prediction_alert"   # dict {score, horizon_secs, message}

    # Autoencoder anomaly detection
    ANOMALY_SCORE     = "ml.anomaly_score"       # float reconstruction error (normalised 0–1)

    # Fusion engine final decision
    FUSION_RISK       = "ml.fusion_risk"         # dict {level: HIGH|LOW, score, lstm, anomaly}

    # Intervention commands
    HAPTIC_CMD      = "intervention.haptic"
    AUDIO_CMD       = "intervention.audio"
    WATER_CMD       = "intervention.water"
    LED_CMD         = "intervention.led"
    SOS_CMD         = "intervention.sos"

    # System
    SYSTEM_STATUS   = "system.status"
    USER_CANCEL     = "user.cancel"          # manual override button
    USER_FEEDBACK   = "user.feedback"        # was this a real attack?
