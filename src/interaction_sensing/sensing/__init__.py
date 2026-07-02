"""Low-cost sensing primitives and scene-state features."""

from .motion import MOG2MotionExtractor
from .scene_state import SceneStateEstimator
from .stabilise import (
    TargetMotionEstimator,
    TargetPose,
    relative_motion_magnitude,
    translate_bbox,
)

__all__ = [
    "MOG2MotionExtractor",
    "SceneStateEstimator",
    "TargetMotionEstimator",
    "TargetPose",
    "relative_motion_magnitude",
    "translate_bbox",
]
