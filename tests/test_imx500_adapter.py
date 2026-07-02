from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from interaction_sensing.domain import BBox
from interaction_sensing.plugins.imx500 import (
    IMX500Detection,
    IMX500InferenceRecord,
    ModelRole,
    SSDDetectionDecoder,
    SensorROI,
    TargetAwareROIController,
    detections_as_candidates,
)


@dataclass
class _Converted:
    x: float
    y: float
    width: float
    height: float


class _Converter:
    def convert_inference_coords(self, coords, metadata, picam2):
        assert metadata == {"frame": 1}
        assert picam2 == "camera"
        return _Converted(*coords)


class _BatchedTensor:
    """Minimal NumPy-like batch wrapper used to test decoder logic off-device."""

    ndim = 2

    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, index):
        return self.batch[index]


def test_sensor_roi_clips_to_sensor_bounds() -> None:
    roi = SensorROI(4000, 3000, 200, 200).clipped((4056, 3040))
    assert roi.as_tuple() == (4000, 3000, 56, 40)


def test_roi_controller_only_updates_after_meaningful_change() -> None:
    controller = TargetAwareROIController(sensor_size=(4056, 3040), min_update_iou=0.90)
    first, changed_first = controller.update_target_context(BBox(100, 100, 300, 300))
    second, changed_second = controller.update_target_context(BBox(102, 102, 302, 302))
    assert changed_first is True
    assert changed_second is False
    assert first == second


def test_ssd_decoder_converts_coords_filters_confidence_and_labels() -> None:
    decoder = SSDDetectionDecoder(confidence_threshold=0.5, labels={1: "actor_candidate"})
    detections = decoder.decode(
        outputs=[
            _BatchedTensor([[(10.0, 20.0, 30.0, 40.0), (0.0, 0.0, 5.0, 5.0)]]),
            _BatchedTensor([[0.9, 0.2]]),
            _BatchedTensor([[1, 2]]),
        ],
        metadata={"frame": 1},
        imx500=_Converter(),
        picam2="camera",
    )
    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "actor_candidate"
    assert detection.bbox == BBox(10.0, 20.0, 40.0, 60.0)
    assert detection.confidence == pytest.approx(0.9)


def test_imx500_detections_are_candidates_not_interaction_events() -> None:
    record = IMX500InferenceRecord(
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        model_path="models/actor.rpk",
        model_role=ModelRole.ACTOR_PROPOSAL,
        inference_roi=SensorROI(100, 100, 300, 300),
        detections=(IMX500Detection(BBox(110, 120, 130, 145), 1, 0.8, "actor_candidate"),),
        frame_index=7,
    )
    candidates = detections_as_candidates(record)
    assert len(candidates) == 1
    assert candidates[0].objectness_score == pytest.approx(0.8)
    assert candidates[0].metadata["model_role"] == "actor_proposal"
    assert candidates[0].metadata["source"] == "imx500"
