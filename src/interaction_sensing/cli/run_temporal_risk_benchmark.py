"""Train and evaluate the lightweight temporal nuisance-risk model."""

from __future__ import annotations

import argparse
from pathlib import Path

from interaction_sensing.simulation import (
    TemporalRiskBenchmarkConfig,
    TemporalRiskModelConfig,
    VisualBenchmarkConfig,
    run_temporal_risk_benchmark,
    write_temporal_risk_benchmark,
)


def _parse_scales(values: list[str]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales or any(scale <= 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("nuisance scales must be positive")
    return scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/temporal_risk_benchmark"))
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--calibration-replicates", type=int, default=12)
    parser.add_argument("--test-replicates", type=int, default=18)
    parser.add_argument("--nuisance-scales", nargs="+", default=["0.65", "1.0", "1.35"])
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--candidate-recall", type=float, default=0.97)
    parser.add_argument("--window-frames", type=int, default=6)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--quick", action="store_true", help="Run a compact smoke-test grid")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visual = VisualBenchmarkConfig(
        frames=min(args.frames, 100) if args.quick else args.frames,
        calibration_replicates=4 if args.quick else args.calibration_replicates,
        test_replicates=6 if args.quick else args.test_replicates,
        nuisance_scales=_parse_scales(args.nuisance_scales),
        target_recall=args.target_recall,
        seed=args.seed,
    )
    model = TemporalRiskModelConfig(
        window_frames=args.window_frames,
        hidden_units=args.hidden_units,
        epochs=min(args.epochs, 120) if args.quick else args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    config = TemporalRiskBenchmarkConfig(
        visual=visual,
        candidate_recall=args.candidate_recall,
        model=model,
    )
    run = run_temporal_risk_benchmark(config)
    outputs = write_temporal_risk_benchmark(args.output_dir, run, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"held-out rows: {len(run.results)}")
    print(f"training samples: {run.training_summary.samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
