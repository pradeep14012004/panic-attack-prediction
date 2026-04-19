"""
cloud/mqtt_bridge.py
--------------------
Publishes PanicGuard events to an MQTT broker.
Subscribes to incoming commands (override, baseline update) from the broker.

Topics published (device -> broker):
  panicguard/{device_id}/score        float  panic score 0.0-1.0
  panicguard/{device_id}/state        str    idle|alert|episode|sos|cooldown
  panicguard/{device_id}/fusion       json   {level, score, lstm, anomaly}
  panicguard/{device_id}/sensors/ppg  json   {heart_rate, spo2}
  panicguard/{device_id}/sensors/gsr  json   {conductance_us}
  panicguard/{device_id}/episode      json   {ts, state, score}
  panicguard/{device_id}/status       json   heartbeat every 30s

Topics subscribed (broker -> device):
  panicguard/{device_id}/cmd/cancel       -> publish USER_CANCEL
  panicguard/{device_id}/cmd/sos          -> publish SOS_CMD
  panicguard/{device_id}/cmd/baseline     -> update baseline thresholds

Usage:
  bridge = MQTTBridge(bus, broker="mqtt.example.com", device_id="pg-001")
  await bridge.start()

Broker options (free/self-hosted):
  - test.mosquitto.org  (public test broker, no auth)
  - broker.hivemq.com   (public test broker)
  - Self-hosted: mosquitto on Raspberry Pi
"""

import asyncio
import json
import time
from loguru import logger
from core.event_bus import EventBus, Topics
from core.state_machine import PanicState


class MQTTBridge:
    def __init__(
        self,
        bus: EventBus,
        broker: str = "test.mosquitto.org",
        port: int = 1883,
        device_id: str = "panicguard-01",
        username: str | None = None,
        password: str | None = None,
        tls: bool = False,
    ):
        self.bus       = bus
        self.broker    = broker
        self.port      = port
        self.device_id = device_id
        self._username = username
        self._password = password
        self._tls      = tls
        self._client   = None
        self._running  = False

        self._prefix = f"panicguard/{device_id}"

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self):
        """Connect to broker and start publishing. Call as asyncio task."""
        try:
            import asyncio_mqtt as aiomqtt
        except ImportError:
            logger.error("[MQTT] asyncio-mqtt not installed. Run: pip install asyncio-mqtt")
            return

        self._running = True
        logger.info(f"[MQTT] connecting to {self.broker}:{self.port}  device={self.device_id}")

        # Subscribe to event bus topics
        self.bus.subscribe(Topics.PANIC_SCORE,    self._on_score)
        self.bus.subscribe(Topics.PANIC_STATE,    self._on_state)
        self.bus.subscribe(Topics.FUSION_RISK,    self._on_fusion)
        self.bus.subscribe("sensor.*",            self._on_sensor)

        tls_params = aiomqtt.TLSParameters() if self._tls else None

        async with aiomqtt.Client(
            hostname=self.broker,
            port=self.port,
            username=self._username,
            password=self._password,
            tls_params=tls_params,
            keepalive=60,
            will=aiomqtt.Will(
                topic=f"{self._prefix}/status",
                payload=json.dumps({"online": False, "ts": time.time()}),
                qos=1,
                retain=True,
            ),
        ) as client:
            self._client = client
            logger.info(f"[MQTT] connected to {self.broker}")

            # Announce online
            await self._publish("status", {"online": True, "ts": time.time()}, retain=True)

            # Subscribe to incoming commands
            await client.subscribe(f"{self._prefix}/cmd/#")
            logger.info(f"[MQTT] subscribed to {self._prefix}/cmd/#")

            # Run heartbeat + message loop concurrently
            await asyncio.gather(
                self._heartbeat_loop(client),
                self._message_loop(client),
            )

    async def stop(self):
        self._running = False
        if self._client:
            await self._publish("status", {"online": False, "ts": time.time()}, retain=True)

    # ── Publish helpers ───────────────────────────────────────────────────────

    async def _publish(self, subtopic: str, payload, retain: bool = False, qos: int = 0):
        if self._client is None:
            return
        try:
            msg = json.dumps(payload) if not isinstance(payload, str) else payload
            await self._client.publish(
                f"{self._prefix}/{subtopic}",
                payload=msg.encode(),
                qos=qos,
                retain=retain,
            )
        except Exception as e:
            logger.warning(f"[MQTT] publish error: {e}")

    # ── Event bus handlers ────────────────────────────────────────────────────

    async def _on_score(self, topic: str, score: float):
        await self._publish("score", str(round(score, 4)))

    async def _on_state(self, topic: str, state: PanicState):
        await self._publish("state", state.value, retain=True, qos=1)
        if state in (PanicState.EPISODE, PanicState.SOS):
            await self._publish("episode", {
                "ts": time.time(), "state": state.value,
            }, qos=1)

    async def _on_fusion(self, topic: str, payload: dict):
        await self._publish("fusion", payload)

    async def _on_sensor(self, topic: str, sample):
        sensor = topic.replace("sensor.", "")
        d = sample.__dict__ if hasattr(sample, "__dict__") else {"value": sample}
        # Only publish key fields to keep bandwidth low
        if sensor == "ppg":
            await self._publish(f"sensors/ppg", {
                "hr": d.get("heart_rate"), "spo2": d.get("spo2")
            })
        elif sensor == "gsr":
            await self._publish(f"sensors/gsr", {
                "conductance": d.get("conductance_us")
            })

    # ── Incoming command loop ─────────────────────────────────────────────────

    async def _message_loop(self, client):
        try:
            async for msg in client.messages:
                topic = str(msg.topic)
                try:
                    payload = json.loads(msg.payload.decode())
                except Exception:
                    payload = msg.payload.decode()

                cmd = topic.split("/cmd/")[-1] if "/cmd/" in topic else None
                if cmd == "cancel":
                    logger.info("[MQTT] received cancel command")
                    await self.bus.publish(Topics.USER_CANCEL)
                elif cmd == "sos":
                    logger.info("[MQTT] received SOS command")
                    await self.bus.publish(Topics.SOS_CMD, {"action": "escalate"})
                elif cmd == "baseline" and isinstance(payload, dict):
                    logger.info(f"[MQTT] baseline update: {payload}")
                    await self.bus.publish("system.baseline_update", payload)
                else:
                    logger.debug(f"[MQTT] unknown command: {cmd}  payload={payload}")
        except Exception as e:
            if self._running:
                logger.warning(f"[MQTT] message loop error: {e}")

    async def _heartbeat_loop(self, client):
        while self._running:
            await asyncio.sleep(30)
            await self._publish("status", {
                "online": True,
                "ts": time.time(),
                "device_id": self.device_id,
            })
