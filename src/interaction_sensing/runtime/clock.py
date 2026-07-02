"""Deterministic source-time clock for files and live cameras."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(slots=True)
class SourceClock:
    """Map frame indices and optional capture positions to monotone timestamps."""

    started_at: datetime
    fps: float
    _last_seconds: float = field(default=-1.0, init=False)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    def timestamp(
        self,
        *,
        frame_index: int,
        position_msec: float | None = None,
    ) -> tuple[datetime, float]:
        fallback = frame_index / self.fps
        from_position = None
        if position_msec is not None and position_msec >= 0:
            from_position = position_msec / 1000.0
        seconds = from_position if from_position is not None else fallback
        # Webcam backends commonly report zero for every frame. Do not let that
        # collapse a real event stream to a single timestamp.
        if frame_index > 0 and seconds <= self._last_seconds:
            seconds = fallback
        seconds = max(seconds, self._last_seconds, 0.0)
        self._last_seconds = seconds
        return self.started_at + timedelta(seconds=seconds), seconds
