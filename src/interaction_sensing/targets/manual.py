"""Manual target initialisation without a taxon-specific detector."""

from __future__ import annotations

from interaction_sensing.domain import BBox, TargetSpec
from interaction_sensing.targets.zones import expand_bbox


def build_target_from_boxes(
    *,
    target_id: str,
    target_type: str,
    core_zone: BBox,
    context_expand_ratio: float = 0.2,
    access_zone: BBox | None = None,
    frame_shape: tuple[int, int] | None = None,
    metadata: dict | None = None,
) -> TargetSpec:
    """Create a target from a user-drawn box and optional nested access box.

    Parameters
    ----------
    frame_shape:
        Optional ``(height, width)`` used to clip the context zone to image bounds.
    """

    context_zone = expand_bbox(core_zone, context_expand_ratio, frame_shape=frame_shape)
    return TargetSpec(
        target_id=target_id,
        target_type=target_type,
        core_zone=core_zone,
        context_zone=context_zone,
        access_zone=access_zone,
        metadata={} if metadata is None else dict(metadata),
    )
