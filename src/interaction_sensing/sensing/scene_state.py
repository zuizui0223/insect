"""Cheap scene covariates retained even when no event is detected."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from interaction_sensing.domain import SceneState


class SceneStateEstimator:
    """Estimate illumination and local target-change scores from successive crops.

    This is a baseline for the future observability model, not a wind classifier.
    It deliberately records the raw quantities needed to learn condition-specific
    error rates from audit clips.
    """

    def __init__(self) -> None:
        self._previous_gray: Any | None = None

    def update(self, crop: Any, *, timestamp: datetime, target_id: str) -> SceneState:
        try:
            import cv2  # type: ignore
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install interaction-sensing[runtime] to estimate scene state") from exc

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        change = None
        target_motion = None
        if self._previous_gray is not None and self._previous_gray.shape == gray.shape:
            difference = cv2.absdiff(gray, self._previous_gray)
            change = float(difference.mean())
            target_motion = float((difference > 20).mean())
        glare = float((gray >= 245).mean())
        self._previous_gray = gray.copy()
        return SceneState(
            timestamp=timestamp,
            target_id=target_id,
            target_motion_score=target_motion,
            illumination_mean=mean,
            illumination_change=change,
            glare_score=glare,
            metadata={"scene_estimator": "frame_difference_baseline", "shape": list(gray.shape)},
        )
