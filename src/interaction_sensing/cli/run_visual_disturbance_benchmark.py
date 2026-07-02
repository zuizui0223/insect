"""Run the rendered image-based disturbance-reference benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from interaction_sensing.simulation import (
    VisualBenchmarkConfig,
    run_visual_benchmark,
    write_visual_benchmark,
)


def _parse_scales(values: list[str]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales or any(scale <= 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("nuisance scales must be positive")
    return scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/visual_disturbance_benchmark"))
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--calibration-replicates", type=int, default=12)
    parser.add_argument("--test-replicates", type=int, default=18)
    parser.add_argument("--nuisance-scales", nargs="+", default=["0.65", "1.0", "1.35"])
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--reference-delay-frames", type=int, default=7)
    parser.add_argument("--alignment-search-radius", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test grid")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VisualBenchmarkConfig(
        frames=min(args.frames, 100) if args.quick else args.frames,
        calibration_replicates=3 if args.quick else args.calibration_replicates,
        test_replicates=5 if args.quick else args.test_replicates,
        nuisance_scales=_parse_scales(args.nuisance_scales),
        target_recall=args.target_recall,
        reference_delay_frames=args.reference_delay_frames,
        alignment_search_radius=args.alignment_search_radius,
        seed=args.seed,
    )
    results, thresholds = run_visual_benchmark(config)
    outputs = write_visual_benchmark(args.output_dir, results, config, thresholds)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"held-out rows: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
