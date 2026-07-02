"""Baseline motion extraction inside a target-relative crop.

This module intentionally preserves the current MOG2-style baseline. It labels
motion candidates, not biological interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from interaction_sensing.domain import BBox, Candidate


@dataclass(slots=True)
class MotionResult:
    candidates: list[Candidate]
    foreground_ratio: float
    mask: Any


class MOG2MotionExtractor:
    """Extract connected foreground components from a local target crop."""

    def __init__(
        self,
        *,
        history: int = 120,
        var_threshold: float = 40.0,
        min_area: int = 80,
        resize_to: tuple[int, int] = (320, 240),
    ) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on runtime extra
            raise ImportError("Install interaction-sensing[runtime] to use MOG2MotionExtractor") from exc
        self.cv2 = cv2
        self.min_area = min_area
        self.resize_to = resize_to
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def extract(
        self,
        frame: Any,
        roi: BBox,
        *,
        timestamp: datetime,
    ) -> MotionResult:
        """Return motion candidates in full-frame coordinates."""

        cv2 = self.cv2
        height, width = frame.shape[:2]
        left = max(0, int(roi.left))
        top = max(0, int(roi.top))
        right = min(width, int(roi.right))
        bottom = min(height, int(roi.bottom))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return MotionResult(candidates=[], foreground_ratio=0.0, mask=None)

        crop_height, crop_width = crop.shape[:2]
        small = cv2.resize(crop, self.resize_to)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.subtractor.apply(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = mask.copy()
        candidates: list[Candidate] = []
        scale_x = crop_width / self.resize_to[0]
        scale_y = crop_height / self.resize_to[1]
        for label_index in range(1, count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < self.min_area:
                clean[labels == label_index] = 0
                continue
            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            bbox = BBox(
                left + x * scale_x,
                top + y * scale_y,
                left + (x + component_width) * scale_x,
                top + (y + component_height) * scale_y,
            )
            candidates.append(
                Candidate(
                    timestamp=timestamp,
                    bbox=bbox,
                    relative_motion_score=area / float(clean.size),
                    metadata={"component_area": area, "extractor": "mog2"},
                )
            )
        ratio = float(cv2.countNonZero(clean)) / float(clean.size)
        return MotionResult(candidates=candidates, foreground_ratio=ratio, mask=clean)
