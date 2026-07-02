"""Taxon-agnostic data contracts for interaction sensing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class BBox:
    """Pixel-aligned rectangle in left, top, right, bottom order."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("BBox must have positive width and height")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def intersection_area(self, other: "BBox") -> float:
        width = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        height = max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return width * height

    def iou(self, other: "BBox") -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return 0.0 if union <= 0 else intersection / union

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class InteractionState(str, Enum):
    OUTSIDE = "outside"
    APPROACH = "approach"
    CONTEXT_ENTRY = "context_entry"
    TARGET_CONTACT = "target_contact"
    ACCESS_ZONE_ENTRY = "access_zone_entry"
    DEPARTED = "departed"
    UNKNOWN = "unknown"


class ErrorClass(str, Enum):
    FP_MOTION = "FP_MOTION"
    FP_NONINTERACTION = "FP_NONINTERACTION"
    FP_WRONG_TARGET = "FP_WRONG_TARGET"
    FN_MISSED = "FN_MISSED"
    FN_OCCLUDED = "FN_OCCLUDED"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    ID_SWAP = "ID_SWAP"
    TARGET_DRIFT = "TARGET_DRIFT"
    STATE_ERROR = "STATE_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class TargetSpec:
    """A focal object plus nested researcher-defined interaction zones."""

    target_id: str
    target_type: str
    core_zone: BBox
    context_zone: BBox | None = None
    access_zone: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id cannot be empty")
        if not self.target_type.strip():
            raise ValueError("target_type cannot be empty")
        if self.context_zone is None:
            self.context_zone = self.core_zone
        if self.access_zone is not None and not self.core_zone.contains(self.access_zone.centroid):
            raise ValueError("access_zone should be located within the focal target")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "core_zone": self.core_zone.to_dict(),
            "context_zone": None if self.context_zone is None else self.context_zone.to_dict(),
            "access_zone": None if self.access_zone is None else self.access_zone.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SceneState:
    timestamp: datetime
    target_id: str
    target_motion_score: float | None = None
    camera_motion_score: float | None = None
    illumination_mean: float | None = None
    illumination_change: float | None = None
    glare_score: float | None = None
    rain_score: float | None = None
    target_density: float | None = None
    occlusion_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(slots=True)
class Candidate:
    timestamp: datetime
    bbox: BBox
    relative_motion_score: float
    candidate_id: str = field(default_factory=lambda: uuid4().hex)
    track_id: str | None = None
    objectness_score: float | None = None
    verifier_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        return self.bbox.centroid

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "timestamp": self.timestamp.isoformat(),
            "bbox": self.bbox.to_dict(),
            "relative_motion_score": self.relative_motion_score,
            "track_id": self.track_id,
            "objectness_score": self.objectness_score,
            "verifier_label": self.verifier_label,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class InteractionEvent:
    target_id: str
    start_time: datetime
    state: InteractionState
    event_id: str = field(default_factory=lambda: uuid4().hex)
    actor_track_id: str | None = None
    end_time: datetime | None = None
    max_state: InteractionState | None = None
    attribution_score: float | None = None
    verification_score: float | None = None
    clip_id: str | None = None
    pipeline_id: str | None = None
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_state is None:
            self.max_state = self.state

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return max(0.0, (self.end_time - self.start_time).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_time"] = self.start_time.isoformat()
        payload["end_time"] = None if self.end_time is None else self.end_time.isoformat()
        payload["state"] = self.state.value
        payload["max_state"] = None if self.max_state is None else self.max_state.value
        payload["duration_seconds"] = self.duration_seconds
        return payload


@dataclass(slots=True)
class AuditRecord:
    clip_id: str
    sampled_at: datetime
    sampling_probability: float
    audit_id: str = field(default_factory=lambda: uuid4().hex)
    truth_event_count: int | None = None
    truth_target_ids: list[str] = field(default_factory=list)
    error_classes: list[ErrorClass] = field(default_factory=list)
    reviewer: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.sampling_probability <= 1.0:
            raise ValueError("sampling_probability must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "clip_id": self.clip_id,
            "sampled_at": self.sampled_at.isoformat(),
            "sampling_probability": self.sampling_probability,
            "truth_event_count": self.truth_event_count,
            "truth_target_ids": self.truth_target_ids,
            "error_classes": [item.value for item in self.error_classes],
            "reviewer": self.reviewer,
            "notes": self.notes,
            "metadata": self.metadata,
        }


def json_ready(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert supported domain values to a serialisable mapping."""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported value type: {type(value)!r}")
