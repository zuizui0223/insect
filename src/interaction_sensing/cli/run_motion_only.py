"""Run the error-aware motion-only baseline from a camera or video source."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from interaction_sensing.config import MotionOnlySettings
from interaction_sensing.domain import BBox
from interaction_sensing.runtime import MotionOnlyRunConfig, MotionOnlyRuntime


def _parse_xywh(value: str) -> BBox:
    try:
        x, y, width, height = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected x,y,width,height") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return BBox(x, y, x + width, y + height)


def _parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the motion-only baseline with a manual focal target, raw event clips, "
            "random audit clips, and an SQLite event ledger."
        )
    )
    parser.add_argument("--source", required=True, type=_parse_source, help="Video path or numeric camera index")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target-type", default="focal_target")
    parser.add_argument(
        "--core-zone",
        required=True,
        type=_parse_xywh,
        metavar="X,Y,W,H",
        help="Manual focal target rectangle in pixels from the first frame",
    )
    parser.add_argument(
        "--access-zone",
        type=_parse_xywh,
        metavar="X,Y,W,H",
        help="Optional nested access rectangle in pixels",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/motion_only.toml"),
        help="TOML baseline settings",
    )
    parser.add_argument("--display", action="store_true", help="Show a live diagnostic overlay; raw clips remain unannotated")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames (useful for tests)")
    parser.add_argument("--motion-threshold", type=float, help="Override foreground-ratio threshold")
    parser.add_argument("--audit-probability", type=float, help="Override audit selection probability per window")
    parser.add_argument("--audit-window-seconds", type=float, help="Override duration between audit selection windows")
    parser.add_argument("--audit-clip-seconds", type=float, help="Override audit clip length")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = MotionOnlySettings.from_toml(args.config)
    overrides: dict[str, float] = {}
    if args.motion_threshold is not None:
        overrides["foreground_ratio_threshold"] = args.motion_threshold
    if args.audit_probability is not None:
        overrides["audit_probability_per_window"] = args.audit_probability
    if args.audit_window_seconds is not None:
        overrides["audit_window_seconds"] = args.audit_window_seconds
    if args.audit_clip_seconds is not None:
        overrides["audit_clip_seconds"] = args.audit_clip_seconds
    if overrides:
        settings = replace(settings, **overrides)

    summary = MotionOnlyRuntime(
        MotionOnlyRunConfig(
            source=args.source,
            output_dir=args.output_dir,
            target_id=args.target_id,
            target_type=args.target_type,
            core_zone=args.core_zone,
            access_zone=args.access_zone,
            settings=settings,
            display=args.display,
            max_frames=args.max_frames,
            source_label=str(args.source),
        )
    ).run()
    print(
        f"run={summary.run_id} frames={summary.frames_processed} "
        f"events={summary.events_started} audits={summary.audits_started} "
        f"output={summary.output_dir}"
    )


if __name__ == "__main__":
    main()
