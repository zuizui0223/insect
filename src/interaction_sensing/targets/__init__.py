"""Target specification, zones, and target-relative helpers."""

from .manual import build_target_from_boxes
from .zones import classify_candidate_state, expand_bbox

__all__ = ["build_target_from_boxes", "classify_candidate_state", "expand_bbox"]
