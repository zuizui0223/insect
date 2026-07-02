"""Noise-first, error-aware sensing for complex natural scenes."""

from .domain import (
    AuditRecord,
    BBox,
    Candidate,
    ErrorClass,
    InteractionEvent,
    InteractionState,
    SceneState,
    TargetSpec,
)
from .noise import (
    NoiseFirstPolicy,
    NoiseObservation,
    NoiseSource,
    ObservabilityDecision,
    ObservabilityState,
)

__all__ = [
    "AuditRecord",
    "BBox",
    "Candidate",
    "ErrorClass",
    "InteractionEvent",
    "InteractionState",
    "NoiseFirstPolicy",
    "NoiseObservation",
    "NoiseSource",
    "ObservabilityDecision",
    "ObservabilityState",
    "SceneState",
    "TargetSpec",
]

__version__ = "0.1.0"
