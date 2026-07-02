from dataclasses import dataclass

import pytest

from interaction_sensing.domain import BBox
from interaction_sensing.plugins.imx500 import (
    SSDDetectionDecoder,
    SensorROI,
    TargetAwareROIController,
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
            [[(10.0, 20.0, 30.0, 40.0), (0.0, 0.0, 5.0, 5.0)]],
            [[0.9, 0.2]],
            [[1, 2]],
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
