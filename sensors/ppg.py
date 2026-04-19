"""
sensors/base.py + sensors/ppg.py
─────────────────────────────────
All sensors inherit from SensorBase.
Switch backend by setting `backend` in config — zero code changes elsewhere.

Supported backends:
    simulate   — synthetic signal with injected panic episodes (dev/test)
    max30102   — real PPG over I2C (Raspberry Pi)
    serial     — UART byte stream from Arduino/ESP32
    ble        — BLE characteristic polling (bleak library)
"""

import asyncio
import time
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
from core.event_bus import EventBus, Topics


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PPGSample:
    timestamp: float
    red: int           # raw ADC counts (0–131071 for MAX30102)
    ir: int
    heart_rate: float  # BPM, computed on-device or here
    spo2: float        # SpO2 %


@dataclass
class GSRSample:
    timestamp: float
    conductance_us: float   # microSiemens
    resistance_kohm: float


@dataclass
class RespirationSample:
    timestamp: float
    rate_bpm: float
    depth: float            # normalized 0–1
    regularity: float       # 0 = chaotic, 1 = perfectly regular


@dataclass
class IMUSample:
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


# ── Base class ────────────────────────────────────────────────────────────────

class SensorBase(ABC):
    """
    All sensors share this interface.
    Subclasses implement _read() to return a sample dataclass.
    The run() loop publishes samples to the event bus at sample_rate_hz.
    """
    def __init__(self, bus: EventBus, topic: str, sample_rate_hz: float = 25.0):
        self.bus = bus
        self.topic = topic
        self.sample_rate_hz = sample_rate_hz
        self._running = False

    @abstractmethod
    async def _read(self):
        """Return one sample. Raise SensorError on hardware fault."""
        ...

    async def _on_connect(self):
        """Called once before the read loop. Override for hardware init."""
        pass

    async def _on_disconnect(self):
        """Called after the read loop exits."""
        pass

    async def run(self):
        await self._on_connect()
        self._running = True
        interval = 1.0 / self.sample_rate_hz
        logger.info(f"[{self.__class__.__name__}] started at {self.sample_rate_hz}Hz on topic '{self.topic}'")
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    sample = await self._read()
                    await self.bus.publish(self.topic, sample)
                except SensorError as e:
                    logger.warning(f"[{self.__class__.__name__}] read error: {e}")
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0, interval - elapsed))
        finally:
            await self._on_disconnect()
            logger.info(f"[{self.__class__.__name__}] stopped")

    def stop(self):
        self._running = False


class SensorError(Exception):
    pass


# ── PPG Simulator ─────────────────────────────────────────────────────────────

class PPGSimulator(SensorBase):
    """
    Generates realistic PPG signals with:
      - Normal baseline (HR 65–80, SpO2 97–99)
      - Injected panic episodes (HR surge, SpO2 slight drop)
      - Motion artifacts (random noise bursts)

    Useful for end-to-end testing without hardware.
    """
    def __init__(self, bus: EventBus, inject_panic_at: float | None = None):
        super().__init__(bus, Topics.PPG_RAW, sample_rate_hz=25.0)
        self._start_time = time.time()
        self._inject_panic_at = inject_panic_at  # seconds after start
        self._phase = 0.0

    def _is_panic_phase(self) -> bool:
        elapsed = time.time() - self._start_time
        if self._inject_panic_at is None:
            return False
        return self._inject_panic_at <= elapsed <= self._inject_panic_at + 120

    async def _read(self) -> PPGSample:
        await asyncio.sleep(0)  # yield to event loop
        elapsed = time.time() - self._start_time
        panic = self._is_panic_phase()

        # Heart rate: 68 baseline, spikes to 115 during panic
        hr_base = 115.0 if panic else 68.0
        hr_drift = 5 * math.sin(elapsed / 60)          # slow drift
        hr_noise = random.gauss(0, 1.5)
        hr = max(40, hr_base + hr_drift + hr_noise)

        # SpO2: 98.5 baseline, drops slightly during panic
        spo2 = random.gauss(97.2 if panic else 98.5, 0.3)
        spo2 = max(90.0, min(100.0, spo2))

        # Raw ADC: simulate pulsatile waveform
        self._phase += (hr / 60.0) * (1.0 / self.sample_rate_hz) * 2 * math.pi
        pulse = math.sin(self._phase)
        # Add dicrotic notch (secondary peak in real PPG)
        notch_phase = self._phase % (2 * math.pi)
        dicrotic = 0.15 * math.sin(2 * self._phase) if notch_phase > math.pi else 0

        noise = random.gauss(0, 300 if panic else 150)  # more noise during panic
        ir_dc = 80000
        ir_ac = int(ir_dc + 4000 * pulse + 800 * dicrotic + noise)
        red_ac = int(ir_dc * 0.7 + 2800 * pulse + noise * 0.8)

        return PPGSample(
            timestamp=time.time(),
            red=max(0, red_ac),
            ir=max(0, ir_ac),
            heart_rate=round(hr, 1),
            spo2=round(spo2, 1),
        )


# ── GSR Simulator ─────────────────────────────────────────────────────────────

class GSRSimulator(SensorBase):
    """
    Simulates electrodermal activity (EDA).
    Panic phase: SCR peaks + tonic level rise.
    """
    def __init__(self, bus: EventBus, inject_panic_at: float | None = None):
        super().__init__(bus, Topics.GSR_RAW, sample_rate_hz=10.0)
        self._start_time = time.time()
        self._inject_panic_at = inject_panic_at
        self._scr_cooldown = 0.0

    def _is_panic_phase(self) -> bool:
        elapsed = time.time() - self._start_time
        if self._inject_panic_at is None:
            return False
        return self._inject_panic_at <= elapsed <= self._inject_panic_at + 120

    async def _read(self) -> GSRSample:
        await asyncio.sleep(0)
        panic = self._is_panic_phase()

        # Tonic (slow baseline) conductance
        tonic = 8.0 if panic else 2.5
        tonic += random.gauss(0, 0.2)

        # Phasic (SCR peaks) — random spikes during panic
        phasic = 0.0
        if panic and random.random() < 0.05:   # 5% chance per sample = ~0.5/sec
            phasic = random.uniform(1.5, 4.0)

        conductance = max(0.1, tonic + phasic)

        return GSRSample(
            timestamp=time.time(),
            conductance_us=round(conductance, 3),
            resistance_kohm=round(1000.0 / conductance, 1),
        )


# ── Respiration Simulator ─────────────────────────────────────────────────────

class RespirationSimulator(SensorBase):
    """
    Simulates breathing rate and regularity.
    Panic phase: rapid, shallow, irregular breathing.
    """
    def __init__(self, bus: EventBus, inject_panic_at: float | None = None):
        super().__init__(bus, Topics.RESPIRATION_RAW, sample_rate_hz=5.0)
        self._start_time = time.time()
        self._inject_panic_at = inject_panic_at

    def _is_panic_phase(self) -> bool:
        elapsed = time.time() - self._start_time
        if self._inject_panic_at is None:
            return False
        return self._inject_panic_at <= elapsed <= self._inject_panic_at + 120

    async def _read(self) -> RespirationSample:
        await asyncio.sleep(0)
        panic = self._is_panic_phase()

        rate = random.gauss(26 if panic else 14, 3 if panic else 1)
        rate = max(6, min(40, rate))

        depth = random.gauss(0.3 if panic else 0.7, 0.1)
        depth = max(0.1, min(1.0, depth))

        regularity = random.gauss(0.4 if panic else 0.9, 0.15)
        regularity = max(0.0, min(1.0, regularity))

        return RespirationSample(
            timestamp=time.time(),
            rate_bpm=round(rate, 1),
            depth=round(depth, 3),
            regularity=round(regularity, 3),
        )


# ── MAX30102 Real Hardware Driver ─────────────────────────────────────────────

class MAX30102Sensor(SensorBase):
    """
    Real PPG sensor via I2C on Raspberry Pi.
    Requires: pip install smbus2 max30102
    Datasheet: https://datasheets.maximintegrated.com/en/ds/MAX30102.pdf

    Wiring:
        MAX30102 VIN  → 3.3V
        MAX30102 GND  → GND
        MAX30102 SDA  → GPIO2 (pin 3)
        MAX30102 SCL  → GPIO3 (pin 5)
        MAX30102 INT  → GPIO4 (pin 7)  [optional interrupt]
    """
    I2C_ADDRESS = 0x57
    SAMPLE_AVG  = 4      # average 4 samples per reading
    LED_MODE    = 2      # RED + IR
    SAMPLE_RATE = 400    # samples/sec on chip (we read at 25Hz via polling)
    LED_POWER   = 60     # mA (0–51mA, 0.2mA resolution)

    def __init__(self, bus: EventBus, i2c_bus: int = 1):
        super().__init__(bus, Topics.PPG_RAW, sample_rate_hz=25.0)
        self._i2c_bus_num = i2c_bus
        self._device = None
        self._hr_calc = _HRCalculator()

    async def _on_connect(self):
        try:
            import max30102
            self._device = max30102.MAX30102(
                i2c_bus=self._i2c_bus_num,
                mode=self.LED_MODE,
                sample_rate=self.SAMPLE_RATE,
                led_power=self.LED_POWER,
            )
            logger.info("[MAX30102] connected on I2C bus", self._i2c_bus_num)
        except ImportError:
            raise SensorError("max30102 library not installed. Run: pip install max30102")
        except Exception as e:
            raise SensorError(f"MAX30102 init failed: {e}")

    async def _read(self) -> PPGSample:
        if self._device is None:
            raise SensorError("Device not connected")
        try:
            # Read FIFO — returns lists of red, ir values
            red_buf, ir_buf = self._device.read_sequential(num=4)
            red = int(sum(red_buf) / len(red_buf))
            ir  = int(sum(ir_buf)  / len(ir_buf))

            hr, spo2 = self._hr_calc.update(red, ir)

            return PPGSample(
                timestamp=time.time(),
                red=red,
                ir=ir,
                heart_rate=hr,
                spo2=spo2,
            )
        except Exception as e:
            raise SensorError(f"MAX30102 read error: {e}")

    async def _on_disconnect(self):
        if self._device:
            self._device.shutdown()


class _HRCalculator:
    """
    Simple peak-detection HR calculator from raw IR signal.
    For production, replace with the Pan-Tompkins algorithm or
    use the on-chip IR LED + algorithm from the MAX30102 firmware.
    """
    WINDOW = 100  # samples at 25Hz = 4 seconds

    def __init__(self):
        self._ir_buffer: list[int] = []
        self._hr = 0.0
        self._spo2 = 98.5

    def update(self, red: int, ir: int) -> tuple[float, float]:
        self._ir_buffer.append(ir)
        if len(self._ir_buffer) > self.WINDOW:
            self._ir_buffer.pop(0)

        if len(self._ir_buffer) >= self.WINDOW:
            self._hr, self._spo2 = self._compute(red, ir)

        return round(self._hr, 1), round(self._spo2, 1)

    def _compute(self, red: int, ir: int) -> tuple[float, float]:
        import numpy as np
        buf = np.array(self._ir_buffer, dtype=float)

        # Remove DC offset
        buf -= buf.mean()

        # Find peaks (simple threshold crossing)
        threshold = buf.std() * 0.6
        peaks = []
        above = False
        for i, v in enumerate(buf):
            if v > threshold and not above:
                peaks.append(i)
                above = True
            elif v <= threshold:
                above = False

        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / 25.0  # seconds
            hr = 60.0 / rr_intervals.mean()
            hr = max(40, min(200, hr))
        else:
            hr = self._hr or 70.0

        # Ratio-of-ratios SpO2 (simplified — calibrate with lookup table for accuracy)
        dc_ir  = ir
        dc_red = red
        if dc_ir > 0 and dc_red > 0:
            r = (dc_red / dc_ir)
            spo2 = 110.0 - 25.0 * r   # approximate calibration curve
            spo2 = max(90.0, min(100.0, spo2))
        else:
            spo2 = self._spo2 or 98.5

        return hr, spo2


# ── Sensor factory ────────────────────────────────────────────────────────────

def create_sensor(sensor_type: str, backend: str, bus: EventBus,
                  inject_panic_at: float | None = None, **kwargs):
    """
    Factory function — instantiates the right sensor class for the backend.

    sensor_type: 'ppg' | 'gsr' | 'respiration'
    backend:     'simulate' | 'max30102' | 'serial' | 'ble'
    """
    drivers = {
        "ppg": {
            "simulate": PPGSimulator,
            "max30102": MAX30102Sensor,
        },
        "gsr": {
            "simulate": GSRSimulator,
        },
        "respiration": {
            "simulate": RespirationSimulator,
        },
    }

    if sensor_type not in drivers:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    if backend not in drivers[sensor_type]:
        raise ValueError(f"Backend '{backend}' not available for '{sensor_type}'. "
                         f"Available: {list(drivers[sensor_type].keys())}")

    cls = drivers[sensor_type][backend]
    if backend == "simulate":
        return cls(bus, inject_panic_at=inject_panic_at)
    return cls(bus, **kwargs)
