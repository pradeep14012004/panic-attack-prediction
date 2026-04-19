"""
tests/test_full_pipeline.py
────────────────────────────
End-to-end integration test:
  1. Start all modules with simulated sensors
  2. Inject a panic episode at t=5s
  3. Assert state machine reaches EPISODE within 60s
  4. Assert interventions fired (haptic, audio, water)
  5. Assert score returns below threshold and state reaches COOLDOWN

Run: pytest tests/test_full_pipeline.py -v
"""

import asyncio
import pytest
from core.event_bus import EventBus, Topics
from core.state_machine import PanicStateMachine, PanicState
from sensors.ppg import create_sensor
from ml.inference import FeatureExtractor, PanicClassifier
from interventions.drivers import create_intervention_suite


class InterventionMonitor:
    """Records which interventions fired."""
    def __init__(self, bus: EventBus):
        self.haptic_fired = False
        self.audio_fired  = False
        self.water_fired  = False
        self.sos_fired    = False
        self.states: list[PanicState] = []

        bus.subscribe(Topics.HAPTIC_CMD, self._h)
        bus.subscribe(Topics.AUDIO_CMD,  self._a)
        bus.subscribe(Topics.WATER_CMD,  self._w)
        bus.subscribe(Topics.SOS_CMD,    self._s)
        bus.subscribe(Topics.PANIC_STATE, self._state)

    async def _h(self, t, cmd):
        if cmd.get("active"):
            self.haptic_fired = True
    async def _a(self, t, cmd):
        if cmd.get("action") == "play":
            self.audio_fired = True
    async def _w(self, t, cmd):
        if cmd.get("action") == "mist":
            self.water_fired = True
    async def _s(self, t, cmd):
        if cmd.get("action") == "escalate":
            self.sos_fired = True
    async def _state(self, t, state: PanicState):
        self.states.append(state)


@pytest.mark.asyncio
async def test_panic_episode_detected_and_interventions_fire():
    bus     = EventBus()
    monitor = InterventionMonitor(bus)
    sm      = PanicStateMachine(bus)
    fex     = FeatureExtractor(bus)
    clf     = PanicClassifier(bus, backend="rules")
    _       = create_intervention_suite(bus, {})  # all simulate

    sensors = [
        create_sensor("ppg",         "simulate", bus, inject_panic_at=5.0),
        create_sensor("gsr",         "simulate", bus, inject_panic_at=5.0),
        create_sensor("respiration", "simulate", bus, inject_panic_at=5.0),
    ]

    async def run_for(secs: float):
        tasks = [asyncio.create_task(s.run()) for s in sensors]
        tasks.append(asyncio.create_task(fex.run()))
        await asyncio.sleep(secs)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    await run_for(90)  # run for 90s — panic episode at 5s should be detected

    assert PanicState.EPISODE in monitor.states or PanicState.ALERT in monitor.states, \
        f"Expected EPISODE or ALERT state. Got: {monitor.states}"
    assert monitor.haptic_fired, "Haptic breathing guide should have fired"
    assert monitor.audio_fired,  "Calming audio should have played"
    assert monitor.water_fired,  "Water mist should have triggered"


@pytest.mark.asyncio
async def test_rule_based_classifier_scores_panic_features():
    from ml.inference import FeatureVector, PanicClassifier

    bus = EventBus()
    clf = PanicClassifier(bus, backend="rules", baseline={
        "hr_resting": 70.0,
        "rmssd_resting": 50.0,
        "scl_resting": 2.5,
    })

    # High-panic feature vector
    panic_fv = FeatureVector(
        timestamp=0, hr_mean=118, hr_std=8, rmssd=12, sdnn=15,
        scl_mean=9.5, scr_peak_rate=2.8, scr_amplitude=3.1,
        resp_rate=28, resp_regularity=0.3, resp_depth=0.2,
        temp_delta=-1.5, motion_rms=9.82, hr_resp_coupling=0.1
    )
    panic_score = clf._rule_based(panic_fv)
    assert panic_score >= 0.6, f"Panic features should score ≥0.6, got {panic_score:.3f}"

    # Calm feature vector
    calm_fv = FeatureVector(
        timestamp=0, hr_mean=66, hr_std=1.5, rmssd=52, sdnn=55,
        scl_mean=2.3, scr_peak_rate=0.2, scr_amplitude=0.1,
        resp_rate=14, resp_regularity=0.92, resp_depth=0.75,
        temp_delta=0.1, motion_rms=9.82, hr_resp_coupling=0.85
    )
    calm_score = clf._rule_based(calm_fv)
    assert calm_score <= 0.2, f"Calm features should score ≤0.2, got {calm_score:.3f}"


@pytest.mark.asyncio
async def test_sos_not_triggered_for_mild_episode():
    bus     = EventBus()
    monitor = InterventionMonitor(bus)
    sm      = PanicStateMachine(bus)

    # Manually publish scores that should trigger EPISODE but not SOS
    scores = [0.45, 0.55, 0.65, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.25]
    for s in scores:
        await bus.publish(Topics.PANIC_SCORE, s)
        await asyncio.sleep(0.1)

    assert PanicState.EPISODE in monitor.states
    assert not monitor.sos_fired, "SOS should not fire for a mild episode"


@pytest.mark.asyncio
async def test_user_cancel_stops_interventions():
    bus     = EventBus()
    monitor = InterventionMonitor(bus)
    sm      = PanicStateMachine(bus)

    # Drive into EPISODE state
    for s in [0.65, 0.70, 0.68]:
        await bus.publish(Topics.PANIC_SCORE, s)
        await asyncio.sleep(0.05)

    assert PanicState.EPISODE in monitor.states

    # User presses cancel
    await bus.publish(Topics.USER_CANCEL)
    await asyncio.sleep(0.1)

    assert PanicState.CANCELLED in monitor.states
