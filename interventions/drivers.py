"""
interventions/drivers.py
─────────────────────────
Hardware-agnostic intervention drivers.
Each driver listens to its command topic and dispatches to
the appropriate backend (GPIO, BLE, HTTP, etc.)

All drivers have a `simulate=True` mode that logs instead of
touching hardware — safe for development on any machine.
"""

import asyncio
import time
from loguru import logger
from core.event_bus import EventBus, Topics


# ── Haptic Driver ─────────────────────────────────────────────────────────────

class HapticDriver:
    """
    Drives a haptic motor for breathing guidance and grounding patterns.

    Backends:
        simulate — logs pattern to console
        gpio     — PWM on GPIO pin (Raspberry Pi)
        drv2605l — I2C haptic driver IC (better waveforms)

    Breathing guide pattern:
        4s vibrate (inhale) → 6s off (exhale) → repeat
        This matches the box breathing protocol used in panic therapy.
    """
    PATTERNS = {
        "breathing_guide": [
            {"on": True,  "duration": 4.0, "intensity": 0.7},
            {"on": False, "duration": 2.0},
            {"on": True,  "duration": 1.5, "intensity": 0.4},
            {"on": False, "duration": 6.0},
        ],
        "soft_pulse": [
            {"on": True,  "duration": 0.5, "intensity": 0.3},
            {"on": False, "duration": 1.5},
        ],
        "grounding": [
            {"on": True,  "duration": 0.3, "intensity": 1.0},
            {"on": False, "duration": 0.3},
        ] * 5,
        "stop": [],
    }

    def __init__(self, bus: EventBus, backend: str = "simulate",
                 gpio_pin: int = 18, i2c_address: int = 0x5A):
        self.bus = bus
        self._backend = backend
        self._gpio_pin = gpio_pin
        self._i2c_address = i2c_address
        self._task: asyncio.Task | None = None
        self._pwm = None

        bus.subscribe(Topics.HAPTIC_CMD, self._on_cmd)
        self._init_hardware()

    def _init_hardware(self):
        if self._backend == "gpio":
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self._gpio_pin, GPIO.OUT)
                self._pwm = GPIO.PWM(self._gpio_pin, 200)  # 200Hz carrier
                self._pwm.start(0)
                logger.info(f"[Haptic] GPIO PWM on pin {self._gpio_pin}")
            except ImportError:
                logger.warning("[Haptic] RPi.GPIO not available, falling back to simulate")
                self._backend = "simulate"
        elif self._backend == "drv2605l":
            logger.info(f"[Haptic] DRV2605L at I2C 0x{self._i2c_address:02X}")

    async def _on_cmd(self, topic: str, cmd: dict):
        pattern_name = cmd.get("pattern", "stop")
        active = cmd.get("active", True)

        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.sleep(0.05)

        if not active or pattern_name == "stop":
            await self._set_motor(0)
            return

        pattern = self.PATTERNS.get(pattern_name, [])
        if pattern:
            self._task = asyncio.create_task(self._run_pattern(pattern, pattern_name))

    async def _run_pattern(self, steps: list, name: str):
        logger.info(f"[Haptic] playing pattern '{name}'")
        try:
            while True:   # loop pattern until cancelled
                for step in steps:
                    if step["on"]:
                        await self._set_motor(step.get("intensity", 0.8))
                    else:
                        await self._set_motor(0)
                    await asyncio.sleep(step["duration"])
        except asyncio.CancelledError:
            await self._set_motor(0)

    async def _set_motor(self, intensity: float):
        """intensity 0.0–1.0"""
        if self._backend == "simulate":
            if intensity > 0:
                logger.debug(f"[Haptic] 🔴 VIBRATE  intensity={intensity:.0%}")
            else:
                logger.debug(f"[Haptic] ⬛ OFF")
        elif self._backend == "gpio" and self._pwm:
            self._pwm.ChangeDutyCycle(intensity * 100)
        elif self._backend == "drv2605l":
            await self._drv2605l_set(intensity)

    async def _drv2605l_set(self, intensity: float):
        """DRV2605L I2C control (simplified)."""
        try:
            import smbus2
            bus = smbus2.SMBus(1)
            waveform = int(intensity * 117)  # waveforms 1–117 in ROM
            bus.write_byte_data(self._i2c_address, 0x04, max(1, waveform))
            bus.write_byte_data(self._i2c_address, 0x0C, 0xFF)  # GO register
        except Exception as e:
            logger.warning(f"[Haptic] DRV2605L error: {e}")


# ── Audio Driver ──────────────────────────────────────────────────────────────

class AudioDriver:
    """
    Plays calming audio through BLE earbuds or onboard speaker.

    Backends:
        simulate — logs to console
        alsa     — Linux audio via subprocess (aplay)
        bluetooth — BLE A2DP via bluetoothctl + aplay

    Audio files live in assets/audio/.
    User can upload personal voice recordings via the companion app.
    """
    DEFAULT_TRACKS = {
        "calm_voice":  "assets/audio/calm_voice.wav",
        "box_breath":  "assets/audio/box_breathing.wav",
        "nature":      "assets/audio/rain_ambient.wav",
    }

    def __init__(self, bus: EventBus, backend: str = "simulate",
                 bt_device: str | None = None):
        self.bus = bus
        self._backend = backend
        self._bt_device = bt_device    # MAC address e.g. "AA:BB:CC:DD:EE:FF"
        self._process = None
        self._custom_tracks: dict[str, str] = {}

        bus.subscribe(Topics.AUDIO_CMD, self._on_cmd)

    def register_voice_recording(self, name: str, path: str):
        """Companion app calls this when user uploads a recording."""
        self._custom_tracks[name] = path
        logger.info(f"[Audio] registered voice recording '{name}' → {path}")

    async def _on_cmd(self, topic: str, cmd: dict):
        action = cmd.get("action", "stop")
        if action == "play":
            track = cmd.get("track", "calm_voice")
            await self._play(track)
        elif action == "stop":
            await self._stop()

    async def _play(self, track: str):
        path = self._custom_tracks.get(track) or self.DEFAULT_TRACKS.get(track)
        if not path:
            logger.warning(f"[Audio] unknown track '{track}'")
            return

        await self._stop()   # stop any current playback

        if self._backend == "simulate":
            logger.info(f"[Audio] 🎵 PLAY  track='{track}'  path={path}")
        elif self._backend == "alsa":
            import subprocess
            self._process = subprocess.Popen(["aplay", "-q", path])
            logger.info(f"[Audio] playing via ALSA: {path}")
        elif self._backend == "bluetooth":
            import subprocess
            # Pair and connect earbuds, then play via ALSA with BT sink
            logger.info(f"[Audio] playing via Bluetooth {self._bt_device}: {path}")
            self._process = subprocess.Popen(
                ["aplay", "-D", f"bluealsa:DEV={self._bt_device}", path]
            )

    async def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process = None
        if self._backend == "simulate":
            logger.debug("[Audio] ⬛ STOP")


# ── Water Mist Driver ─────────────────────────────────────────────────────────

class WaterMistDriver:
    """
    Triggers a micro peristaltic pump for grounding water mist.
    The cold stimulus activates the mammalian diving reflex —
    a hard-wired parasympathetic response that drops heart rate within seconds.

    Hardware: 5V peristaltic pump on GPIO relay. 800ms pulse = ~0.1ml mist.
    Safe for wrist or face application.
    """
    def __init__(self, bus: EventBus, backend: str = "simulate",
                 gpio_pin: int = 24, max_pulses_per_hour: int = 6):
        self.bus = bus
        self._backend = backend
        self._gpio_pin = gpio_pin
        self._max_per_hour = max_pulses_per_hour
        self._pulse_times: list[float] = []

        bus.subscribe(Topics.WATER_CMD, self._on_cmd)
        self._init_hardware()

    def _init_hardware(self):
        if self._backend == "gpio":
            try:
                import RPi.GPIO as GPIO
                GPIO.setup(self._gpio_pin, GPIO.OUT, initial=GPIO.LOW)
                logger.info(f"[Water] pump relay on GPIO {self._gpio_pin}")
            except ImportError:
                self._backend = "simulate"

    async def _on_cmd(self, topic: str, cmd: dict):
        action = cmd.get("action", "mist")
        if action != "mist":
            return
        duration_ms = cmd.get("duration_ms", 800)
        await self._pulse(duration_ms / 1000.0)

    async def _pulse(self, duration_sec: float):
        # Rate limit — don't spray more than N times per hour
        now = time.time()
        self._pulse_times = [t for t in self._pulse_times if now - t < 3600]
        if len(self._pulse_times) >= self._max_per_hour:
            logger.warning("[Water] rate limit reached, skipping mist")
            return

        self._pulse_times.append(now)

        if self._backend == "simulate":
            logger.info(f"[Water] 💧 MIST  duration={duration_sec*1000:.0f}ms")
            await asyncio.sleep(duration_sec)
            logger.debug("[Water] ⬛ PUMP OFF")
        elif self._backend == "gpio":
            import RPi.GPIO as GPIO
            GPIO.output(self._gpio_pin, GPIO.HIGH)
            await asyncio.sleep(duration_sec)
            GPIO.output(self._gpio_pin, GPIO.LOW)


# ── LED Driver ────────────────────────────────────────────────────────────────

class LEDDriver:
    """
    Drives an LED ring (NeoPixel) or single RGB LED.
    Color encodes panic state — visible biofeedback for the user.

    Modes:
        idle_green   — calm, monitoring (slow breathe pulse)
        pulse_amber  — early warning
        pulse_blue   — active episode (calming color)
        sos_red      — emergency
    """
    MODES = {
        "idle_green":  {"r": 0,   "g": 40,  "b": 10,  "pulse": True,  "hz": 0.15},
        "pulse_amber": {"r": 200, "g": 100, "b": 0,   "pulse": True,  "hz": 0.5},
        "pulse_blue":  {"r": 0,   "g": 80,  "b": 200, "pulse": True,  "hz": 0.25},
        "sos_red":     {"r": 255, "g": 0,   "b": 0,   "pulse": True,  "hz": 2.0},
    }

    def __init__(self, bus: EventBus, backend: str = "simulate",
                 gpio_pin: int = 12, num_pixels: int = 12):
        self.bus = bus
        self._backend = backend
        self._gpio_pin = gpio_pin
        self._num_pixels = num_pixels
        self._strip = None
        self._task: asyncio.Task | None = None

        bus.subscribe(Topics.LED_CMD, self._on_cmd)
        self._init_hardware()

    def _init_hardware(self):
        if self._backend == "neopixel":
            try:
                import board, neopixel
                self._strip = neopixel.NeoPixel(
                    getattr(board, f"D{self._gpio_pin}"),
                    self._num_pixels, brightness=0.5, auto_write=False
                )
                logger.info(f"[LED] NeoPixel ring {self._num_pixels}px on D{self._gpio_pin}")
            except Exception:
                self._backend = "simulate"

    async def _on_cmd(self, topic: str, cmd: dict):
        mode = cmd.get("mode", "idle_green")
        if self._task and not self._task.done():
            self._task.cancel()
        config = self.MODES.get(mode, self.MODES["idle_green"])
        self._task = asyncio.create_task(self._animate(config, mode))

    async def _animate(self, cfg: dict, name: str):
        import math
        try:
            while True:
                # Pulse brightness using sine wave
                t = time.time()
                brightness = (math.sin(2 * 3.14159 * cfg["hz"] * t) + 1) / 2
                r = int(cfg["r"] * brightness)
                g = int(cfg["g"] * brightness)
                b = int(cfg["b"] * brightness)
                await self._set_color(r, g, b, name)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            await self._set_color(0, 0, 0, "off")

    async def _set_color(self, r: int, g: int, b: int, label: str):
        if self._backend == "simulate":
            pass  # only log on state change to avoid spam
        elif self._backend == "neopixel" and self._strip:
            for i in range(self._num_pixels):
                self._strip[i] = (r, g, b)
            self._strip.show()

    import math


# ── SOS Driver ────────────────────────────────────────────────────────────────

class SOSDriver:
    """
    Emergency escalation — sends GPS coordinates + alert message
    to emergency contacts via HTTP (ntfy.sh push) or SMS (Twilio).

    Backends:
        simulate — logs to console
        ntfy     — free push notifications via ntfy.sh
        twilio   — SMS via Twilio API (requires credentials)
    """
    def __init__(self, bus: EventBus, backend: str = "simulate",
                 contacts: list[dict] | None = None,
                 ntfy_topic: str = "panic-guard-sos",
                 twilio_creds: dict | None = None):
        self.bus = bus
        self._backend = backend
        self._contacts = contacts or []
        self._ntfy_topic = ntfy_topic
        self._twilio = twilio_creds or {}
        self._last_sos: float | None = None
        self._cooldown_secs = 300   # don't re-alert within 5 min

        bus.subscribe(Topics.SOS_CMD, self._on_cmd)
        bus.subscribe(Topics.USER_CANCEL, self._on_cancel)

    async def _on_cmd(self, topic: str, cmd: dict):
        if cmd.get("action") != "escalate":
            return

        now = time.time()
        if self._last_sos and (now - self._last_sos) < self._cooldown_secs:
            logger.info("[SOS] cooldown active, skipping")
            return

        self._last_sos = now
        gps = await self._get_gps()
        message = self._format_message(gps)
        logger.warning(f"[SOS] 🆘 ESCALATING — {message}")

        if self._backend == "simulate":
            logger.warning(f"[SOS] Would send to {len(self._contacts)} contacts: {message}")
        elif self._backend == "ntfy":
            await self._send_ntfy(message)
        elif self._backend == "twilio":
            await self._send_twilio(message)

    async def _on_cancel(self, topic: str, _payload=None):
        logger.info("[SOS] User cancelled — sending 'false alarm' to contacts")
        if self._backend == "ntfy":
            await self._send_ntfy("✅ False alarm — the person is okay.")

    async def _get_gps(self) -> dict:
        """Get GPS from a gpsd daemon or return mock location."""
        try:
            import gpsd
            gpsd.connect()
            packet = gpsd.get_current()
            return {"lat": packet.lat, "lon": packet.lon, "accuracy": packet.error.get("s", 50)}
        except Exception:
            return {"lat": None, "lon": None, "accuracy": None}

    def _format_message(self, gps: dict) -> str:
        base = "⚠️ PanicGuard SOS — the person wearing this device may be experiencing a severe panic attack."
        if gps["lat"]:
            base += f"\n📍 Location: https://maps.google.com/?q={gps['lat']},{gps['lon']}"
        base += "\n\nIf you cannot reach them, call emergency services."
        return base

    async def _send_ntfy(self, message: str):
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                for contact in self._contacts:
                    # Each contact has their own ntfy subscription
                    topic = contact.get("ntfy_topic", self._ntfy_topic)
                    await client.post(
                        f"https://ntfy.sh/{topic}",
                        content=message,
                        headers={"Title": "PanicGuard SOS", "Priority": "urgent", "Tags": "rotating_light"},
                    )
            logger.info(f"[SOS] ntfy alerts sent to {len(self._contacts)} contacts")
        except Exception as e:
            logger.error(f"[SOS] ntfy send failed: {e}")

    async def _send_twilio(self, message: str):
        try:
            from twilio.rest import Client
            client = Client(self._twilio["account_sid"], self._twilio["auth_token"])
            for contact in self._contacts:
                if phone := contact.get("phone"):
                    client.messages.create(
                        body=message,
                        from_=self._twilio["from_number"],
                        to=phone,
                    )
            logger.info(f"[SOS] SMS sent via Twilio to {len(self._contacts)} contacts")
        except Exception as e:
            logger.error(f"[SOS] Twilio send failed: {e}")


# ── Driver factory ────────────────────────────────────────────────────────────

def create_intervention_suite(bus: EventBus, config: dict) -> dict:
    """
    Instantiates all intervention drivers from a config dict.

    config example:
        {
          "haptic":  {"backend": "gpio",     "gpio_pin": 18},
          "audio":   {"backend": "alsa"},
          "water":   {"backend": "gpio",     "gpio_pin": 24},
          "led":     {"backend": "neopixel", "gpio_pin": 12, "num_pixels": 12},
          "sos":     {"backend": "ntfy",     "contacts": [...]}
        }
    """
    import math  # needed in LEDDriver
    return {
        "haptic": HapticDriver(bus, **config.get("haptic", {"backend": "simulate"})),
        "audio":  AudioDriver(bus,  **config.get("audio",  {"backend": "simulate"})),
        "water":  WaterMistDriver(bus, **config.get("water", {"backend": "simulate"})),
        "led":    LEDDriver(bus,    **config.get("led",    {"backend": "simulate"})),
        "sos":    SOSDriver(bus,    **config.get("sos",    {"backend": "simulate"})),
    }
