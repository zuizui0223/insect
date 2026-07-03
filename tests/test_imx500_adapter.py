from datetime import datetime, timezone

import numpy as np
import pytest

from interaction_sensing.domain import BBox
from interaction_sensing.plugins.imx500 import (
    IMX500Detection,
    IMX500InferenceRecord,
    IMX500Runtime,
    ModelRole,
    SSDDetectionDecoder,
    SensorROI,
    TargetAwareROIController,
    detections_as_candidates,
)


class _Converter:
    """Mimics Picamera2 IMX500.convert_inference_coords: returns an (x, y, w, h) tuple."""

    def convert_inference_coords(self, coords, metadata, picam2):
        assert metadata == {"frame": 1}
        assert picam2 == "camera"
        return (10, 20, 30, 40)


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
    # Hardware shape (add_batch=False): boxes (N, 4), scores (N,), classes (N,).
    decoder = SSDDetectionDecoder(confidence_threshold=0.5, labels={1: "hardware_probe_label"})
    detections = decoder.decode(
        outputs=[
            np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.05, 0.05]], dtype=np.float32),
            np.array([0.9, 0.2], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
        ],
        metadata={"frame": 1},
        imx500=_Converter(),
        picam2="camera",
    )
    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "hardware_probe_label"
    assert detection.bbox == BBox(10.0, 20.0, 40.0, 60.0)  # (x, y, x+w, y+h)
    assert detection.confidence == pytest.approx(0.9)


def test_ssd_decoder_strips_leading_batch_axis() -> None:
    # add_batch=True shape: boxes (1, N, 4), scores (1, N), classes (1, N).
    decoder = SSDDetectionDecoder(confidence_threshold=0.5)
    detections = decoder.decode(
        outputs=[
            np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32),
            np.array([[0.9]], dtype=np.float32),
            np.array([[1]], dtype=np.int32),
        ],
        metadata={"frame": 1},
        imx500=_Converter(),
        picam2="camera",
    )
    assert len(detections) == 1
    assert detections[0].bbox == BBox(10.0, 20.0, 40.0, 60.0)


class _FakeRequest:
    def __init__(self, metadata):
        self._metadata = metadata
        self.released = False

    def get_metadata(self):
        return self._metadata

    def release(self):
        self.released = True


class _FakePicam2:
    def __init__(self, metadata):
        self._metadata = metadata
        self.last_request = None

    def capture_request(self):
        self.last_request = _FakeRequest(self._metadata)
        return self.last_request


class _FakeIMX500:
    """Mimics Picamera2's IMX500: get_outputs() is None before the network is ready."""

    def __init__(self, outputs):
        self._outputs = outputs

    def get_outputs(self, metadata):
        return self._outputs

    def get_kpi_info(self, metadata):
        return None


class _RaisingDecoder:
    def decode(self, outputs, metadata, imx500, picam2):
        raise AssertionError("decoder must not run when no inference tensor is present")


def _runtime_with_fakes(tmp_path, *, outputs, decoder) -> IMX500Runtime:
    model = tmp_path / "fake.rpk"
    model.write_bytes(b"")
    runtime = IMX500Runtime(model_path=model, decoder=decoder)
    runtime.picam2 = _FakePicam2(metadata={"frame": 1})
    runtime.imx500 = _FakeIMX500(outputs)
    return runtime


def test_capture_records_frame_with_no_inference_tensor(tmp_path) -> None:
    runtime = _runtime_with_fakes(tmp_path, outputs=None, decoder=_RaisingDecoder())
    record = runtime.capture_inference()
    assert record.detections == ()
    assert record.metadata["inference_available"] is False
    assert record.frame_index == 1
    assert runtime.picam2.last_request.released is True


def test_capture_decodes_when_inference_tensor_present(tmp_path) -> None:
    outputs = [
        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([1], dtype=np.int32),
    ]
    decoder = SSDDetectionDecoder(confidence_threshold=0.5)
    runtime = _runtime_with_fakes(tmp_path, outputs=outputs, decoder=decoder)
    runtime.imx500.convert_inference_coords = lambda coords, metadata, picam2: (10, 20, 30, 40)
    record = runtime.capture_inference()
    assert record.metadata["inference_available"] is True
    assert len(record.detections) == 1


def test_optional_object_proposals_remain_candidates_not_events() -> None:
    record = IMX500InferenceRecord(
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        model_path="models/ablation.rpk",
        model_role=ModelRole.OBJECT_PROPOSAL,
        inference_roi=SensorROI(100, 100, 300, 300),
        detections=(IMX500Detection(BBox(110, 120, 130, 145), 1, 0.8, "object"),),
        frame_index=7,
    )
    candidates = detections_as_candidates(record)
    assert len(candidates) == 1
    assert candidates[0].objectness_score == pytest.approx(0.8)
    assert candidates[0].metadata["model_role"] == "object_proposal"
    assert candidates[0].metadata["source"] == "imx500"
