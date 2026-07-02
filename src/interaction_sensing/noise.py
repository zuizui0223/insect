"""Noise-first contracts for auditable ecological sensing.

The primary object of this module is not a flower, insect, or interaction. It
is the *observation condition*: which scene processes are likely to create a
false event, a missed event, or an attribution error in a later ecological
measurement.

Targets and actors are optional downstream applications of this noise field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class NoiseSource(str, Enum):
    STABLE_SCENE = "stable_scene"
    GLOBAL_CAMERA_SHAKE = "global_camera_shake"
    CO_MOVING_FOREGROUND = "co_moving_foreground"
    BACKGROUND_VEGETATION_MOTION = "background_vegetation_motion"
    ILLUMINATION_TRANSIENT = "illumination_transient"
    SHADOW_TRANSIENT = "shadow_transient"
    OCCLUSION = "occlusion"
    BLUR_OR_FOCUS_LOSS = "blur_or_focus_loss"
    LENS_CONTAMINATION = "lens_contamination"
    MULTI_OBJECT_CLUTTER = "multi_object_clutter"
    UNKNOWN = "unknown"


class ObservabilityState(str, Enum):
    CLEAN = "clean"
    CONFOUNDED = "confounded"
    UNOBSERVABLE = "unobservable"
    AUDIT_PRIORITY = "audit_priority"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class NoiseObservation:
    """Condition record for one frame or fixed time window.

    `sensor_scores` can contain IMX500 model outputs for scene-noise classes.
    The host-side quantities remain explicit because temporal noise processes
    such as global shake and coherent foreground motion cannot be inferred
    reliably from one image alone.
    """

    timestamp: datetime
    source: NoiseSource
    confidence: float
    frame_index: int | None = None
    global_motion_score: float | None = None
    coherent_foreground_motion_score: float | None = None
    local_relative_motion_score: float | None = None
    illumination_change: float | None = None
    blur_score: float | None = None
    occlusion_score: float | None = None
    clutter_score: float | None = None
    sensor_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["source"] = self.source.value
        return payload


@dataclass(frozen=True, slots=True)
class ObservabilityDecision:
    state: ObservabilityState
    false_event_risk: float
    missed_event_risk: float
    attribution_risk: float
    capture_audit: bool
    record_high_resolution_context: bool
    reasons: tuple[str, ...]


@dataclass(slots=True)
class NoiseFirstPolicy:
    """Transparent baseline policy for turning a noise record into actions.

    This is not a final learned model. It ensures that the system records noise
    and uncertainty instead of silently deleting inconvenient frames.
    """

    high_risk_threshold: float = 0.60
    unobservable_threshold: float = 0.85

    def decide(self, observation: NoiseObservation) -> ObservabilityDecision:
        source = observation.source
        confidence = observation.confidence
        false_risk = 0.0
        missed_risk = 0.0
        attribution_risk = 0.0
        reasons: list[str] = []

        if source in {NoiseSource.GLOBAL_CAMERA_SHAKE, NoiseSource.CO_MOVING_FOREGROUND}:
            false_risk = 0.85 * confidence
            missed_risk = 0.25 * confidence
            reasons.append("motion may be explained by camera or foreground displacement")
        elif source is NoiseSource.BACKGROUND_VEGETATION_MOTION:
            false_risk = 0.70 * confidence
            attribution_risk = 0.40 * confidence
            reasons.append("background motion can create false local candidates")
        elif source in {NoiseSource.ILLUMINATION_TRANSIENT, NoiseSource.SHADOW_TRANSIENT}:
            false_risk = 0.65 * confidence
            missed_risk = 0.45 * confidence
            reasons.append("photometric change can alter motion and detector outputs")
        elif source is NoiseSource.OCCLUSION:
            missed_risk = 0.85 * confidence
            attribution_risk = 0.55 * confidence
            reasons.append("occlusion can hide or merge observed entities")
        elif source in {NoiseSource.BLUR_OR_FOCUS_LOSS, NoiseSource.LENS_CONTAMINATION}:
            missed_risk = 0.80 * confidence
            false_risk = 0.25 * confidence
            reasons.append("image degradation reduces observable detail")
        elif source is NoiseSource.MULTI_OBJECT_CLUTTER:
            attribution_risk = 0.85 * confidence
            false_risk = 0.30 * confidence
            reasons.append("multiple moving objects make causal attribution ambiguous")
        elif source is NoiseSource.UNKNOWN:
            false_risk = missed_risk = attribution_risk = 0.50
            reasons.append("unclassified scene condition")
        else:
            reasons.append("no dominant noise source inferred")

        maximum_risk = max(false_risk, missed_risk, attribution_risk)
        if maximum_risk >= self.unobservable_threshold:
            state = ObservabilityState.UNOBSERVABLE
        elif maximum_risk >= self.high_risk_threshold:
            state = ObservabilityState.AUDIT_PRIORITY
        elif maximum_risk > 0.20:
            state = ObservabilityState.CONFOUNDED
        else:
            state = ObservabilityState.CLEAN

        return ObservabilityDecision(
            state=state,
            false_event_risk=false_risk,
            missed_event_risk=missed_risk,
            attribution_risk=attribution_risk,
            capture_audit=state in {ObservabilityState.AUDIT_PRIORITY, ObservabilityState.UNKNOWN},
            record_high_resolution_context=state in {ObservabilityState.AUDIT_PRIORITY, ObservabilityState.UNOBSERVABLE},
            reasons=tuple(reasons),
        )
