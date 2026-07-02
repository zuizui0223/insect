"""Small OpenCV video recorder utilities used by baseline runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from interaction_sensing.capture.ring_buffer import BufferedFrame


@dataclass(slots=True)
class VideoClipRecorder:
    """Write event or audit clips without coupling capture to event logic."""

    output_dir: Path
    fps: float
    codec: str = "mp4v"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def open(self, *, clip_id: str, frame_shape: tuple[int, int]) -> "OpenVideoClip":
        return OpenVideoClip(output_dir=self.output_dir, clip_id=clip_id, fps=self.fps, codec=self.codec, frame_shape=frame_shape)

    def write_clip(self, *, clip_id: str, frames: Iterable[BufferedFrame], frame_shape: tuple[int, int]) -> Path:
        clip = self.open(clip_id=clip_id, frame_shape=frame_shape)
        try:
            for buffered in frames:
                clip.write(buffered.frame)
        finally:
            clip.close()
        return clip.path


class OpenVideoClip:
    def __init__(self, *, output_dir: Path, clip_id: str, fps: float, codec: str, frame_shape: tuple[int, int]) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install interaction-sensing[runtime] to write video clips") from exc
        self.path = output_dir / f"{clip_id}.mp4"
        height, width = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.path}")

    def write(self, frame: Any) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "OpenVideoClip":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
