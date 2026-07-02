"""Zone geometry and taxon-agnostic interaction-state assignment."""

from __future__ import annotations

from interaction_sensing.domain import BBox, Candidate, InteractionState, TargetSpec


def expand_bbox(
    bbox: BBox,
    ratio: float,
    *,
    frame_shape: tuple[int, int] | None = None,
) -> BBox:
    """Expand a box symmetrically, optionally clipping to ``(height, width)``."""

    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    dx = bbox.width * ratio
    dy = bbox.height * ratio
    left, top = bbox.left - dx, bbox.top - dy
    right, bottom = bbox.right + dx, bbox.bottom + dy
    if frame_shape is not None:
        height, width = frame_shape
        left, right = max(0.0, left), min(float(width), right)
        top, bottom = max(0.0, top), min(float(height), bottom)
    return BBox(left, top, right, bottom)


def classify_candidate_state(candidate: Candidate, target: TargetSpec) -> InteractionState:
    """Assign a geometric state without assuming anything about actor taxon.

    A centroid inside an access zone is stronger evidence than merely being inside
    the broad context zone. This is deliberately a geometric baseline; later
    trajectory and visual-verification modules can replace or enrich it.
    """

    point = candidate.centroid
    if target.access_zone is not None and target.access_zone.contains(point):
        return InteractionState.ACCESS_ZONE_ENTRY
    if target.core_zone.contains(point):
        return InteractionState.TARGET_CONTACT
    if target.context_zone is not None and target.context_zone.contains(point):
        return InteractionState.CONTEXT_ENTRY
    return InteractionState.OUTSIDE
