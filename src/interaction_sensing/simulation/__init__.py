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
from .temporal_risk import (
    TemporalRiskModel,
    TemporalRiskModelConfig,
    TemporalRiskTrainingSummary,
    fit_temporal_risk_model,
    temporal_reference_matrix,
)
from .temporal_risk_benchmark import (
    TemporalRiskBenchmarkConfig,
    TemporalRiskBenchmarkResult,
    TemporalRiskCalibration,
    TemporalRiskPolicy,
    TemporalRiskRun,
    run_temporal_risk_benchmark,
    write_temporal_risk_benchmark,
)
from .temporal_risk_evaluation import (
    TemporalRiskEvaluationConfig,
    TemporalRiskFailureMapCell,
    TemporalRiskPairedEffect,
    evaluate_temporal_risk_results,
    write_temporal_risk_evaluation,
)
from .visual_benchmark import (
    VisualBenchmarkConfig,
    VisualBenchmarkResult,
    VisualFeatureFrame,
    VisualPolicy,
    estimate_global_shift,
    extract_visual_features,
    run_visual_benchmark,
    write_visual_benchmark,
)
from .visual_world import RenderedFrame, VisualDisturbanceWorld, VisualWorldConfig
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
    "RenderedFrame",
    "ScenarioConfig",
    "TemporalRiskBenchmarkConfig",
    "TemporalRiskBenchmarkResult",
    "TemporalRiskCalibration",
    "TemporalRiskEvaluationConfig",
    "TemporalRiskFailureMapCell",
    "TemporalRiskModel",
    "TemporalRiskModelConfig",
    "TemporalRiskPairedEffect",
    "TemporalRiskPolicy",
    "TemporalRiskRun",
    "TemporalRiskTrainingSummary",
    "VisualBenchmarkConfig",
    "VisualBenchmarkResult",
    "VisualDisturbanceWorld",
    "VisualFeatureFrame",
    "VisualPolicy",
    "VisualWorldConfig",
    "estimate_global_shift",
    "evaluate_latent_results",
    "evaluate_temporal_risk_results",
    "extract_visual_features",
    "fit_temporal_risk_model",
    "run_benchmark",
    "run_latent_benchmark",
    "run_temporal_risk_benchmark",
    "run_visual_benchmark",
    "temporal_reference_matrix",
    "write_benchmark",
    "write_latent_benchmark",
    "write_latent_evaluation",
    "write_temporal_risk_benchmark",
    "write_temporal_risk_evaluation",
    "write_visual_benchmark",
]
