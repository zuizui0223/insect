"""Run the pre-field synthetic observability benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from interaction_sensing.simulation import BenchmarkConfig, run_benchmark, write_benchmark


def _numbers(values: list[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/synthetic_benchmark"))
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--wind", nargs="+", default=["0", "4", "10"])
    parser.add_argument("--neighbour-distance", nargs="+", default=["32", "64", "120"])
    parser.add_argument("--tracker-error", nargs="+", default=["0", "0.5", "2"])
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--audit-probability", type=float, default=0.10)
    parser.add_argument("--quick", action="store_true", help="Use a small grid for rapid smoke testing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        config = BenchmarkConfig(
            frames=min(args.frames, 240),
            replicates=min(args.replicates, 3),
            wind_amplitudes=(0.0, 10.0),
            neighbour_distances=(32.0, 120.0),
            tracker_error_sds=(0.0, 1.0),
            seed=args.seed,
            audit_probability=args.audit_probability,
        )
    else:
        config = BenchmarkConfig(
            frames=args.frames,
            replicates=args.replicates,
            wind_amplitudes=_numbers(args.wind),
            neighbour_distances=_numbers(args.neighbour_distance),
            tracker_error_sds=_numbers(args.tracker_error),
            seed=args.seed,
            audit_probability=args.audit_probability,
        )
    results = run_benchmark(config)
    outputs = write_benchmark(args.output_dir, results, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"rows: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
