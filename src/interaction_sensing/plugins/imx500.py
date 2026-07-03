"""Sony IMX500 adapter for noise-first ecological sensing.

The IMX500 is used primarily as a *sensor-side scene-noise and observability
monitor*. It should identify conditions that make later ecological measurement
unreliable: shake, co-moving foreground, clutter, photometric transients,
occlusion, image degradation, or unknown conditions.

Object detections remain an optional compatibility feature. A neural-network
object detection never directly establishes a biological interaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from interaction_sensing.domain import BBox, Candidate


class ModelRole(str, Enum):
    NOISE_STATE = "noise_state"
    OBSERVABILITY_QUALITY = "observability_quality"
    SCENE_EMBEDDING = "scene_embedding"
    OBJECT_PROPOSAL = "object_proposal"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class SensorROI:
    """Absolute IMX500 inference ROI in full-sensor pixel coordinates."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("SensorROI width and height must be positive")
        if self.left < 0 or self.top < 0:
            raise ValueError("SensorROI offsets must be non-negative")

    @classmethod
    def from_bbox(cls, bbox: BBox) -> "SensorROI":
        return cls(
            left=round(bbox.left),
            top=round(bbox.top),
            width=round(bbox.width),
            height=round(bbox.height),
        )

    def to_bbox(self) -> BBox:
        return BBox(self.left, self.top, self.left + self.width, self.top + self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def clipped(self, sensor_size: tuple[int, int]) -> "SensorROI":
        """Clip to ``(width, height)`` while preserving a non-empty rectangle."""

        sensor_width, sensor_height = sensor_size
        left = min(max(0, self.left), max(0, sensor_width - 1))
        top = min(max(0, self.top), max(0, sensor_height - 1))
        right = min(sensor_width, max(left + 1, self.left + self.width))
        bottom = min(sensor_height, max(top + 1, self.top + self.height))
        return SensorROI(left, top, right - left, bottom - top)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IMX500Detection:
    """Optional object proposal, never a biological event by itself."""

    bbox: BBox
    category: int
    confidence: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "category": self.category,
            "confidence": self.confidence,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class IMX500InferenceRecord:
    """Immutable sensor-side record, independent from biological labels."""

    timestamp: datetime
    model_path: str
    model_role: ModelRole
    inference_roi: SensorROI | None
    detections: tuple[IMX500Detection, ...]
    frame_index: int
    kpi: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_path": self.model_path,
            "model_role": self.model_role.value,
            "inference_roi": None if self.inference_roi is None else self.inference_roi.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
            "frame_index": self.frame_index,
            "kpi": self.kpi,
            "metadata": self.metadata,
        }


def detections_as_candidates(record: IMX500InferenceRecord) -> list[Candidate]:
    """Optional bridge from object proposals into the generic candidate layer.

    This is intentionally not used by the noise-first core. It exists only for
    ablations in which object proposals are compared after scene observability
    has already been measured.
    """

    return [
        Candidate(
            timestamp=record.timestamp,
            bbox=detection.bbox,
            relative_motion_score=0.0,
            objectness_score=detection.confidence,
            verifier_label=detection.label,
            metadata={
                "source": "imx500",
                "model_path": record.model_path,
                "model_role": record.model_role.value,
                "category": detection.category,
                "sensor_frame_index": record.frame_index,
                "sensor_roi": None if record.inference_roi is None else record.inference_roi.to_dict(),
                "kpi": record.kpi,
            },
        )
        for detection in record.detections
    ]


class InferenceDecoder(Protocol):
    def decode(self, outputs: Sequence[Any], metadata: Any, imx500: Any, picam2: Any) -> list[IMX500Detection]:
        """Decode model-specific outputs into optional coordinate-aware proposals."""


@dataclass(slots=True)
class SSDDetectionDecoder:
    """Optional decoder for common `(boxes, scores, classes)` IMX500 SSD output.

    It is retained solely to verify device metadata and to support object-model
    ablations. Custom noise-state classifiers should use their own lightweight
    decoder and write a `NoiseObservation` alongside the inference record.
    """

    confidence_threshold: float = 0.25
    labels: dict[int, str] = field(default_factory=dict)

    def decode(self, outputs: Sequence[Any], metadata: Any, imx500: Any, picam2: Any) -> list[IMX500Detection]:
        if outputs is None or len(outputs) < 3:
            raise ValueError("SSD decoder expects boxes, scores, and classes outputs")
        # Picamera2 IMX500.get_outputs(add_batch=False) returns boxes (N, 4),
        # scores (N,) and classes (N,) with no batch axis. add_batch=True adds a
        # leading singleton axis; strip it only when it is genuinely present.
        boxes = _strip_batch(outputs[0], expected_ndim=2)
        scores = _strip_batch(outputs[1], expected_ndim=1)
        classes = _strip_batch(outputs[2], expected_ndim=1)
        detections: list[IMX500Detection] = []
        for coords, score, category in zip(boxes, scores, classes):
            confidence = float(score)
            if confidence < self.confidence_threshold:
                continue
            # convert_inference_coords returns a (x, y, width, height) tuple in
            # output-image pixels.
            x, y, width, height = imx500.convert_inference_coords(coords, metadata, picam2)
            bbox = BBox(float(x), float(y), float(x) + float(width), float(y) + float(height))
            category_id = int(category)
            detections.append(
                IMX500Detection(
                    bbox=bbox,
                    category=category_id,
                    confidence=confidence,
                    label=self.labels.get(category_id),
                )
            )
        return detections


def _strip_batch(value: Any, *, expected_ndim: int) -> Any:
    """Remove a leading singleton batch axis if one is present.

    ``expected_ndim`` is the shape rank the tensor should have without a batch
    axis (2 for ``(N, 4)`` boxes, 1 for ``(N,)`` scores/classes). Only a leading
    axis of length 1 is stripped, so a real detection count of 1 is preserved.
    """

    ndim = getattr(value, "ndim", None)
    if ndim == expected_ndim + 1:
        try:
            if value.shape[0] == 1:
                return value[0]
        except (IndexError, AttributeError, TypeError):
            pass
    return value


@dataclass(slots=True)
class TargetAwareROIController:
    """Optional sensor ROI controller for controlled ablations.

    Core noise monitoring should normally inspect a fixed scene/context region,
    because moving a crop to follow presumed targets can conceal the very
    clutter and motion processes being measured. This controller remains for
    later controlled comparisons, not as the default observation design.
    """

    sensor_size: tuple[int, int]
    min_update_iou: float = 0.90
    _current_roi: SensorROI | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_update_iou <= 1.0:
            raise ValueError("min_update_iou must lie in [0, 1]")

    @property
    def current_roi(self) -> SensorROI | None:
        return self._current_roi

    def update_target_context(self, context_box: BBox) -> tuple[SensorROI, bool]:
        proposed = SensorROI.from_bbox(context_box).clipped(self.sensor_size)
        if self._current_roi is not None and self._current_roi.to_bbox().iou(proposed.to_bbox()) >= self.min_update_iou:
            return self._current_roi, False
        self._current_roi = proposed
        return proposed, True


class InferenceNDJSONLogger:
    """Append complete sensor records for later noise and audit analysis."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: IMX500InferenceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


class IMX500Runtime:
    """Minimal Picamera2/IMX500 lifecycle wrapper for hardware-in-the-loop tests."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        decoder: InferenceDecoder,
        model_role: ModelRole = ModelRole.EXPERIMENTAL,
        main_size: tuple[int, int] = (2028, 1520),
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"IMX500 model package not found: {self.model_path}")
        self.decoder = decoder
        self.model_role = model_role
        self.main_size = main_size
        self.imx500: Any | None = None
        self.picam2: Any | None = None
        self._frame_index = 0

    def start(self) -> None:
        """Initialise IMX500 *before* Picamera2, as required by Picamera2."""

        try:
            from picamera2 import Picamera2  # type: ignore
            from picamera2.devices.imx500 import IMX500  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware runtime only
            raise ImportError(
                "IMX500Runtime requires Picamera2 with IMX500 support. "
                "Install Raspberry Pi OS camera packages and imx500-all on the Pi."
            ) from exc
        self.imx500 = IMX500(str(self.model_path))
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": self.main_size})
        self.picam2.configure(config)
        self.picam2.start()

    def set_inference_roi(self, roi: SensorROI) -> None:
        if self.imx500 is None:
            raise RuntimeError("Call start() before setting IMX500 ROI")
        self.imx500.set_inference_roi_abs(roi.as_tuple())

    def capture_inference(self, *, inference_roi: SensorROI | None = None) -> IMX500InferenceRecord:
        if self.picam2 is None or self.imx500 is None:
            raise RuntimeError("Call start() before capture_inference()")
        request = self.picam2.capture_request()
        try:
            metadata = request.get_metadata()
            outputs = self.imx500.get_outputs(metadata)
            # On real hardware get_outputs() returns None while the network
            # firmware is still uploading (first frames) and intermittently
            # when a frame carries no inference tensor. That is a valid
            # observation about the sensor, not an error: record the frame with
            # no detections rather than crashing the decoder.
            inference_available = outputs is not None
            detections = (
                self.decoder.decode(outputs, metadata, self.imx500, self.picam2)
                if inference_available
                else []
            )
            try:
                kpi = dict(self.imx500.get_kpi_info(metadata) or {})
            except Exception:
                kpi = {}
            self._frame_index += 1
            return IMX500InferenceRecord(
                timestamp=datetime.now(timezone.utc),
                model_path=str(self.model_path),
                model_role=self.model_role,
                inference_roi=inference_roi,
                detections=tuple(detections),
                frame_index=self._frame_index,
                kpi=kpi,
                metadata={"inference_available": inference_available},
            )
        finally:
            request.release()

    def stop(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

    def __enter__(self) -> "IMX500Runtime":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
