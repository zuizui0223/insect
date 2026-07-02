"""Bounded pre-event frame storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    timestamp: datetime
    frame: Any


class FrameRingBuffer:
    """Keep the immediately preceding frames needed to interpret event onset."""

    def __init__(self, max_frames: int) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self._frames: deque[BufferedFrame] = deque(maxlen=max_frames)

    def append(self, frame: Any, *, timestamp: datetime) -> None:
        self._frames.append(BufferedFrame(timestamp=timestamp, frame=frame))

    def snapshot(self) -> list[BufferedFrame]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterable[BufferedFrame]:
        return iter(self._frames)
