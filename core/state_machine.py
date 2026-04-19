"""
core/state_machine.py
─────────────────────
Manages the device lifecycle and panic episode state.

States:
    IDLE         — normal operation, monitoring in background
    ALERT        — score rising, early-warning interventions active
    EPISODE      — confirmed panic, full intervention suite active
    SOS          — sustained severe episode, emergency escalation
    COOLDOWN     — episode resolved, gradual return to IDLE
    CANCELLED    — user pressed override button
"""

import asyncio
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from loguru import logger
from core.event_bus import EventBus, Topics


class PanicState(Enum):
    IDLE     = "idle"
    ALERT    = "alert"       # score 0.40–0.60
    EPISODE  = "episode"     # score 0.60–0.80
    SOS      = "sos"         # score >0.80 sustained
    COOLDOWN = "cooldown"
    CANCELLED= "cancelled"


@dataclass
class StateContext:
    current_state: PanicState = PanicState.IDLE
    entered_at: float = field(default_factory=time.time)
    panic_score: float = 0.0
    peak_score: float = 0.0
    episode_count_today: int = 0
    last_episode_at: float | None = None

    def time_in_state(self) -> float:
        return time.time() - self.entered_at


# Score thresholds for state transitions
THRESHOLDS = {
    "alert_enter":    0.40,
    "episode_enter":  0.60,
    "sos_enter":      0.80,
    "sos_sustained":  15.0,   # seconds at >0.80 before SOS fires
    "cooldown_enter": 0.35,   # score drops below this → cooldown
    "idle_enter":     0.25,   # score drops below this during cooldown → idle
    "cooldown_secs":  120,    # minimum cooldown duration
}


class PanicStateMachine:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.ctx = StateContext()
        self._sos_high_since: float | None = None

        bus.subscribe(Topics.PANIC_SCORE,      self.on_score)
        bus.subscribe(Topics.USER_CANCEL,        self.on_user_cancel)
        bus.subscribe(Topics.PREDICTION_ALERT,   self.on_prediction_alert)
        bus.subscribe(Topics.FUSION_RISK,        self.on_fusion_risk)

    async def on_fusion_risk(self, topic: str, payload: dict):
        """
        Decision Fusion Engine output.
        HIGH risk while IDLE → escalate to PRE_ALERT (soft warning).
        HIGH risk while ALERT → push straight to EPISODE (anomaly confirmed).
        """
        level  = payload.get("level", "LOW")
        score  = payload.get("score", 0.0)
        lstm   = payload.get("lstm",  0.0)
        anomaly = payload.get("anomaly", 0.0)

        if level != "HIGH":
            return

        state = self.ctx.current_state
        logger.warning(
            f"[StateMachine] FUSION HIGH  score={score:.2f}  "
            f"lstm={lstm:.2f}  anomaly={anomaly:.2f}  state={state.value}"
        )

        if state == PanicState.IDLE:
            # Pre-panic warning — soft haptic + amber LED
            await self.bus.publish(Topics.HAPTIC_CMD, {"pattern": "soft_pulse", "active": True})
            await self.bus.publish(Topics.LED_CMD,    {"mode": "pre_alert_amber"})
            await self._transition(PanicState.ALERT)

        elif state == PanicState.ALERT:
            # Both models agree — escalate immediately
            if lstm > 0.70 and anomaly > 0.50:
                await self._transition(PanicState.EPISODE)

    async def on_prediction_alert(self, topic: str, payload: dict):
        """Bi-LSTM + Autoencoder flagged a pre-panic trajectory."""
        if self.ctx.current_state == PanicState.IDLE:
            score   = payload.get("score", 0.0)
            horizon = payload.get("horizon_secs", 120)
            logger.warning(
                f"[StateMachine] PRE-PANIC ALERT  score={score:.2f}  "
                f"predicted within {horizon}s"
            )
            await self.bus.publish(Topics.HAPTIC_CMD, {"pattern": "soft_pulse", "active": True})
            await self.bus.publish(Topics.LED_CMD,    {"mode": "pre_alert_amber"})

    async def on_score(self, topic: str, score: float):
        self.ctx.panic_score = score
        self.ctx.peak_score = max(self.ctx.peak_score, score)
        await self._evaluate_transitions(score)

    async def on_user_cancel(self, topic: str, _payload=None):
        logger.info("[StateMachine] User cancelled intervention")
        await self._transition(PanicState.CANCELLED)
        # Return to IDLE after 30s cooldown
        await asyncio.sleep(30)
        await self._transition(PanicState.IDLE)

    async def _evaluate_transitions(self, score: float):
        state = self.ctx.current_state

        if state == PanicState.IDLE:
            if score >= THRESHOLDS["alert_enter"]:
                await self._transition(PanicState.ALERT)

        elif state == PanicState.ALERT:
            if score >= THRESHOLDS["episode_enter"]:
                await self._transition(PanicState.EPISODE)
            elif score < THRESHOLDS["cooldown_enter"]:
                await self._transition(PanicState.COOLDOWN)

        elif state == PanicState.EPISODE:
            if score >= THRESHOLDS["sos_enter"]:
                # Track how long we've been in SOS territory
                if self._sos_high_since is None:
                    self._sos_high_since = time.time()
                elif time.time() - self._sos_high_since >= THRESHOLDS["sos_sustained"]:
                    await self._transition(PanicState.SOS)
            else:
                self._sos_high_since = None
                if score < THRESHOLDS["cooldown_enter"]:
                    await self._transition(PanicState.COOLDOWN)

        elif state == PanicState.SOS:
            if score < THRESHOLDS["cooldown_enter"]:
                await self._transition(PanicState.COOLDOWN)

        elif state == PanicState.COOLDOWN:
            elapsed = self.ctx.time_in_state()
            if score < THRESHOLDS["idle_enter"] and elapsed >= THRESHOLDS["cooldown_secs"]:
                await self._transition(PanicState.IDLE)
            elif score >= THRESHOLDS["episode_enter"]:
                # Relapse — jump back to episode
                await self._transition(PanicState.EPISODE)

    async def _transition(self, new_state: PanicState):
        old_state = self.ctx.current_state
        if old_state == new_state:
            return

        logger.info(f"[StateMachine] {old_state.value} → {new_state.value}  (score={self.ctx.panic_score:.2f})")

        self.ctx.current_state = new_state
        self.ctx.entered_at = time.time()
        self._sos_high_since = None

        if new_state in (PanicState.EPISODE, PanicState.SOS):
            self.ctx.episode_count_today += 1
            self.ctx.last_episode_at = time.time()

        # Reset peak when returning to idle
        if new_state == PanicState.IDLE:
            self.ctx.peak_score = 0.0

        await self.bus.publish(Topics.PANIC_STATE, new_state)

        # Fire intervention commands based on new state
        await self._dispatch_interventions(new_state)

    async def _dispatch_interventions(self, state: PanicState):
        if state == PanicState.ALERT:
            await self.bus.publish(Topics.HAPTIC_CMD, {"pattern": "breathing_guide", "active": True})
            await self.bus.publish(Topics.LED_CMD,    {"mode": "pulse_amber"})

        elif state == PanicState.EPISODE:
            await self.bus.publish(Topics.HAPTIC_CMD, {"pattern": "breathing_guide", "active": True})
            await self.bus.publish(Topics.AUDIO_CMD,  {"action": "play", "track": "calm_voice"})
            await self.bus.publish(Topics.WATER_CMD,  {"action": "mist", "duration_ms": 800})
            await self.bus.publish(Topics.LED_CMD,    {"mode": "pulse_blue"})

        elif state == PanicState.SOS:
            await self.bus.publish(Topics.SOS_CMD,    {"action": "escalate"})
            await self.bus.publish(Topics.LED_CMD,    {"mode": "sos_red"})

        elif state in (PanicState.COOLDOWN, PanicState.IDLE, PanicState.CANCELLED):
            await self.bus.publish(Topics.HAPTIC_CMD, {"pattern": "stop", "active": False})
            await self.bus.publish(Topics.AUDIO_CMD,  {"action": "stop"})
            await self.bus.publish(Topics.LED_CMD,    {"mode": "idle_green"})
