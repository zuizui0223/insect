"""Run an IMX500 model as an auditable sensor-side proposal stream.

This command is a hardware-in-the-loop probe. It does not claim that a detected
object is an ecological interaction. It records every decoded sensor proposal
so target attribution and manual audits can happen downstream.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from interaction_sensing.domain import BBox
from interaction_sensing.plugins.imx500 import (
    IMX500Runtime,
    InferenceNDJSONLogger,
    ModelRole,
    SSDDetectionDecoder,
    SensorROI,
    TargetAwareROIController,
)


def _load_labels(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    labels: dict[int, str] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        label = line.strip()
        if label:
            labels[index] = label
    return labels


def _roi_from_args(values: list[int] | None) -> SensorROI | None:
    if values is None:
        return None
    left, top, width, height = values
    return SensorROI(left, top, width, height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Packaged IMX500 .rpk model")
    parser.add_argument("--decoder", choices=["ssd"], default="ssd")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--model-role",
        choices=[role.value for role in ModelRole],
        default=ModelRole.EXPERIMENTAL.value,
        help="Semantic role in the sensing pipeline, not a biological conclusion",
    )
    parser.add_argument(
        "--sensor-roi",
        nargs=4,
        type=int,
        default=None,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Optional absolute full-sensor crop used as IMX500 inference ROI",
    )
    parser.add_argument("--sensor-width", type=int, default=4056)
    parser.add_argument("--sensor-height", type=int, default=3040)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("runs/imx500/inferences.ndjson"))
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    if args.interval_seconds < 0:
        raise ValueError("interval-seconds cannot be negative")

    labels = _load_labels(args.labels)
    decoder = SSDDetectionDecoder(confidence_threshold=args.confidence, labels=labels)
    requested_roi = _roi_from_args(args.sensor_roi)
    logger = InferenceNDJSONLogger(args.output)
    role = ModelRole(args.model_role)

    with IMX500Runtime(model_path=args.model, decoder=decoder, model_role=role) as runtime:
        roi = None
        if requested_roi is not None:
            controller = TargetAwareROIController(sensor_size=(args.sensor_width, args.sensor_height))
            roi, _ = controller.update_target_context(requested_roi.to_bbox())
            runtime.set_inference_roi(roi)
        for _ in range(args.frames):
            record = runtime.capture_inference(inference_roi=roi)
            logger.write(record)
            print(
                f"frame={record.frame_index} detections={len(record.detections)} "
                f"roi={None if roi is None else roi.as_tuple()}"
            )
            if args.interval_seconds:
                time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
