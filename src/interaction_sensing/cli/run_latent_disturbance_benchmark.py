"""Run the target-agnostic latent-disturbance benchmark and locked evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from interaction_sensing.simulation import (
    LatentBenchmarkConfig,
    LatentEvaluationConfig,
    evaluate_latent_results,
    run_latent_benchmark,
    write_latent_benchmark,
    write_latent_evaluation,
)


def _parse_scales(values: list[str]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales or any(value <= 0.0 for value in scales):
        raise argparse.ArgumentTypeError("nuisance scales must be positive")
    return scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latent_disturbance_benchmark"))
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--calibration-replicates", type=int, default=24)
    parser.add_argument("--test-replicates", type=int, default=32)
    parser.add_argument("--nuisance-scales", nargs="+", default=["0.55", "1.0", "1.45"])
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--audit-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--recall-noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test grid")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibration_replicates = 4 if args.quick else args.calibration_replicates
    test_replicates = 6 if args.quick else args.test_replicates
    frames = min(args.frames, 240) if args.quick else args.frames
    bootstrap_resamples = min(args.bootstrap_resamples, 300) if args.quick else args.bootstrap_resamples
    config = LatentBenchmarkConfig(
        frames=frames,
        calibration_replicates=calibration_replicates,
        test_replicates=test_replicates,
        nuisance_scales=_parse_scales(args.nuisance_scales),
        target_recall=args.target_recall,
        audit_fraction=args.audit_fraction,
        seed=args.seed,
    )
    results, thresholds = run_latent_benchmark(config)
    benchmark_outputs = write_latent_benchmark(args.output_dir, results, config, thresholds)
    evaluation_config = LatentEvaluationConfig(
        bootstrap_resamples=bootstrap_resamples,
        recall_noninferiority_margin=args.recall_noninferiority_margin,
        seed=args.seed,
    )
    effects, failure_map = evaluate_latent_results(results, evaluation_config)
    evaluation_outputs = write_latent_evaluation(args.output_dir, effects, failure_map, evaluation_config)
    for name, path in {**benchmark_outputs, **evaluation_outputs}.items():
        print(f"{name}: {path}")
    print(f"test rows: {len(results)}")
    print(f"failure-map cells: {len(failure_map)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
