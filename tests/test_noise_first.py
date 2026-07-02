from datetime import datetime

from interaction_sensing.noise import (
    NoiseFirstPolicy,
    NoiseObservation,
    NoiseSource,
    ObservabilityState,
)


def test_co_moving_foreground_becomes_audit_priority_not_a_deleted_frame() -> None:
    observation = NoiseObservation(
        timestamp=datetime(2026, 7, 2, 12, 0, 0),
        source=NoiseSource.CO_MOVING_FOREGROUND,
        confidence=0.9,
    )
    decision = NoiseFirstPolicy().decide(observation)
    assert decision.state is ObservabilityState.AUDIT_PRIORITY
    assert decision.capture_audit is True
    assert decision.record_high_resolution_context is True
    assert decision.false_event_risk > decision.missed_event_risk


def test_occlusion_marks_high_miss_risk() -> None:
    observation = NoiseObservation(
        timestamp=datetime(2026, 7, 2, 12, 0, 0),
        source=NoiseSource.OCCLUSION,
        confidence=1.0,
    )
    decision = NoiseFirstPolicy().decide(observation)
    assert decision.state is ObservabilityState.UNOBSERVABLE
    assert decision.missed_event_risk == 0.85
    assert decision.attribution_risk == 0.55


def test_stable_scene_stays_clean() -> None:
    observation = NoiseObservation(
        timestamp=datetime(2026, 7, 2, 12, 0, 0),
        source=NoiseSource.STABLE_SCENE,
        confidence=0.95,
    )
    decision = NoiseFirstPolicy().decide(observation)
    assert decision.state is ObservabilityState.CLEAN
    assert decision.capture_audit is False
