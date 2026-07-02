"""Error-aware sensing of target--actor interactions in natural scenes."""

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

__all__ = [
    "AuditRecord",
    "BBox",
    "Candidate",
    "ErrorClass",
    "InteractionEvent",
    "InteractionState",
    "SceneState",
    "TargetSpec",
]

__version__ = "0.1.0"
