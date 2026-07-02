import numpy as np

from interaction_sensing.simulation import (
    TemporalRiskBenchmarkConfig,
    TemporalRiskModelConfig,
    VisualBenchmarkConfig,
    fit_temporal_risk_model,
    run_temporal_risk_benchmark,
    temporal_reference_matrix,
    write_temporal_risk_benchmark,
)
from interaction_sensing.simulation.visual_benchmark import VisualFeatureFrame


def _frame(*, local: float, robust: float, single: float, coherence: float, y: int, x: int) -> VisualFeatureFrame:
    return VisualFeatureFrame(
        frame_index=0,
        true_local_event=False,
        raw_local_evidence=local,
        stabilised_local_evidence=local,
        single_reference=single,
        robust_reference=robust,
        delayed_reference=0.0,
        mismatched_reference=0.0,
        global_shift_y=y,
        global_shift_x=x,
        global_shift_error=0.0,
        reference_coherence=coherence,
    )


def test_temporal_reference_matrix_excludes_focal_local_evidence() -> None:
    low_local = _frame(local=0.01, robust=0.2, single=0.1, coherence=0.05, y=1, x=-1)
    high_local = _frame(local=9.0, robust=0.2, single=0.1, coherence=0.05, y=1, x=-1)
    low = temporal_reference_matrix([low_local, low_local], [1], window_frames=2)
    high = temporal_reference_matrix([high_local, high_local], [1], window_frames=2)
    assert np.array_equal(low, high)


def test_tiny_mlp_learns_separable_scene_context_risk() -> None:
    negative = np.tile(np.asarray([-1.0, -0.5, 0.0, 0.0]), (24, 1))
    positive = np.tile(np.asarray([1.0, 0.8, 1.0, 1.0]), (24, 1))
    features = np.concatenate((negative, positive), axis=0)
    labels = np.asarray([0.0] * len(negative) + [1.0] * len(positive))
    model, summary = fit_temporal_risk_model(
        features,
        labels,
        TemporalRiskModelConfig(hidden_units=6, epochs=250, learning_rate=0.06, seed=7),
    )
    probabilities = model.predict_proba(features)
    assert summary.positive_samples == 24
    assert probabilities[:24].mean() < 0.25
    assert probabilities[24:].mean() > 0.75


def test_temporal_risk_benchmark_writes_model_and_held_out_results(tmp_path) -> None:
    config = TemporalRiskBenchmarkConfig(
        visual=VisualBenchmarkConfig(
            frames=150,
            calibration_replicates=4,
            test_replicates=6,
            nuisance_scales=(0.8, 1.2),
            seed=63,
        ),
        candidate_recall=0.97,
        model=TemporalRiskModelConfig(hidden_units=10, epochs=160, learning_rate=0.05, seed=63),
    )
    run = run_temporal_risk_benchmark(config)
    policies = {result.policy for result in run.results}
    assert "N0_robust_visual_reference" in policies
    assert "N1_rule_risk_gate" in policies
    assert "N1_temporal_mlp_risk_gate" in policies
    assert run.training_summary.samples > 0
    assert 0 < run.training_summary.positive_samples < run.training_summary.samples
    outputs = write_temporal_risk_benchmark(tmp_path, run, config)
    assert all(path.exists() for path in outputs.values())
    assert outputs["model"].suffix == ".npz"
    assert "N1_temporal_mlp_risk_gate" in outputs["metrics"].read_text(encoding="utf-8")
    assert "does not consume focal local evidence" in outputs["report"].read_text(encoding="utf-8")
