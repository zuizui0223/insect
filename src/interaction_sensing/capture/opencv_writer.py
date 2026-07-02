"""OpenCV writers for raw event and audit clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


class OpenCVClipWriter:
    """Write raw BGR frames without overlay graphics used during live display."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        frame_shape: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(codec) != 4:
            raise ValueError("codec must have four characters")
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime extra
            raise ImportError("Install interaction-sensing[runtime] to write clips") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._height, self._width = frame_shape
        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (self._width, self._height))
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Could not open video writer: {self.path}")
        self.frames_written = 0
        self.closed = False

    def write(self, frame: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed clip")
        height, width = frame.shape[:2]
        if (height, width) != (self._height, self._width):
            raise ValueError(
                f"Frame shape {(height, width)} does not match writer shape {(self._height, self._width)}"
            )
        self._writer.write(frame)
        self.frames_written += 1

    def write_many(self, frames: Iterable[Any]) -> None:
        for frame in frames:
            self.write(frame)

    def close(self) -> None:
        if not self.closed:
            self._writer.release()
            self.closed = True

    def __enter__(self) -> "OpenCVClipWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
