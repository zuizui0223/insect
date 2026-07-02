"""Target-relative coordinates and motion gating.

These utilities do not identify an actor. They separate motion that is explained
by a focal target's estimated displacement from motion that remains relative to
that target. The same functions are used by the synthetic benchmark and are
intended for the future live tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from interaction_sensing.domain import BBox


Point = tuple[float, float]
Vector = tuple[float, float]


def subtract_vectors(left: Vector, right: Vector) -> Vector:
    return (left[0] - right[0], left[1] - right[1])


def vector_norm(vector: Vector) -> float:
    return hypot(vector[0], vector[1])


def translate_bbox(bbox: BBox, displacement: Vector) -> BBox:
    """Translate a target-zone box into the current image coordinate frame."""

    dx, dy = displacement
    return BBox(bbox.left + dx, bbox.top + dy, bbox.right + dx, bbox.bottom + dy)


@dataclass(frozen=True, slots=True)
class TargetPose:
    """Current target centre and a displacement estimated from the previous frame."""

    center: Point
    displacement: Vector
    confidence: float = 1.0


@dataclass(slots=True)
class TargetMotionEstimator:
    """Minimal translation-only estimator interface for a target tracker.

    A future optical-flow or keypoint tracker should provide its centre estimate
    to this class. Keeping the motion contract this small makes the downstream
    interaction logic independent of which visual tracker is used.
    """

    previous_center: Point | None = None

    def update(self, center: Point, *, confidence: float = 1.0) -> TargetPose:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if self.previous_center is None:
            displacement = (0.0, 0.0)
        else:
            displacement = (center[0] - self.previous_center[0], center[1] - self.previous_center[1])
        self.previous_center = center
        return TargetPose(center=center, displacement=displacement, confidence=confidence)


def relative_motion_magnitude(candidate_displacement: Vector, target_displacement: Vector) -> float:
    """Magnitude of candidate motion after removing estimated target motion."""

    return vector_norm(subtract_vectors(candidate_displacement, target_displacement))
