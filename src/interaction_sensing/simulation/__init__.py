"""Reproducible, truth-labelled pre-field benchmarks for interaction sensing."""

from .benchmark import BenchmarkConfig, BenchmarkResult, run_benchmark, write_benchmark
from .latent_benchmark import (
    LatentBenchmarkConfig,
    LatentBenchmarkResult,
    LatentPolicy,
    run_latent_benchmark,
    write_latent_benchmark,
)
from .latent_disturbance import LatentDisturbanceConfig, LatentDisturbanceWorld, ReferenceMode
from .latent_evaluation import (
    FailureMapCell,
    LatentEvaluationConfig,
    PairedEffect,
    evaluate_latent_results,
    write_latent_evaluation,
)
from .world import ScenarioConfig

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "FailureMapCell",
    "LatentBenchmarkConfig",
    "LatentBenchmarkResult",
    "LatentDisturbanceConfig",
    "LatentDisturbanceWorld",
    "LatentEvaluationConfig",
    "LatentPolicy",
    "PairedEffect",
    "ReferenceMode",
    "ScenarioConfig",
    "evaluate_latent_results",
    "run_benchmark",
    "run_latent_benchmark",
    "write_benchmark",
    "write_latent_benchmark",
    "write_latent_evaluation",
]
