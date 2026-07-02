"""Create a randomised, target-agnostic NoiseBench recording plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from interaction_sensing.noisebench import NoiseBenchConfig, build_noisebench_plan, write_noisebench_plan


def _parse_intensities(values: list[str]) -> tuple[float, ...]:
    intensities = tuple(float(value) for value in values)
    if not intensities:
        raise argparse.ArgumentTypeError("at least one intensity is required")
    return intensities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/noisebench_plan"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--frame-rate", type=float, default=15.0)
    parser.add_argument("--intensities", nargs="+", default=["0.30", "0.60", "0.90"])
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--no-mixed", action="store_true", help="Generate controls and single disturbances only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = NoiseBenchConfig(
        replicates=args.replicates,
        duration_seconds=args.duration_seconds,
        frame_rate=args.frame_rate,
        intensities=_parse_intensities(args.intensities),
        include_mixed_disturbance=not args.no_mixed,
        seed=args.seed,
    )
    plan = build_noisebench_plan(config)
    outputs = write_noisebench_plan(args.output_dir, plan)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"scenarios: {plan.scenario_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
