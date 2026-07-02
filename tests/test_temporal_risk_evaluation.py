from interaction_sensing.simulation import (
    TemporalRiskBenchmarkConfig,
    TemporalRiskEvaluationConfig,
    TemporalRiskModelConfig,
    VisualBenchmarkConfig,
    evaluate_temporal_risk_results,
    run_temporal_risk_benchmark,
    write_temporal_risk_evaluation,
)


def test_temporal_risk_evaluator_uses_matched_blocks_and_writes_failure_map(tmp_path) -> None:
    benchmark = TemporalRiskBenchmarkConfig(
        visual=VisualBenchmarkConfig(
            frames=140,
            calibration_replicates=4,
            test_replicates=6,
            nuisance_scales=(0.8, 1.2),
            seed=101,
        ),
        candidate_recall=0.97,
        model=TemporalRiskModelConfig(hidden_units=10, epochs=140, learning_rate=0.05, seed=101),
    )
    run = run_temporal_risk_benchmark(benchmark)
    config = TemporalRiskEvaluationConfig(bootstrap_resamples=300, seed=101)
    effects, failure_map = evaluate_temporal_risk_results(run.results, config)

    n1_vs_n0 = [
        effect
        for effect in effects
        if effect.condition == "nuisance_scale=1.2"
        and effect.intervention == "N1_temporal_mlp_risk_gate"
        and effect.comparator == "N0_robust_visual_reference"
        and effect.metric == "false_event_reduction"
    ]
    assert len(n1_vs_n0) == 1
    assert n1_vs_n0[0].blocks == 6
    assert n1_vs_n0[0].ci_low <= n1_vs_n0[0].estimate <= n1_vs_n0[0].ci_high

    n1_vs_rule = [
        effect
        for effect in effects
        if effect.condition == "nuisance_scale=0.8"
        and effect.intervention == "N1_temporal_mlp_risk_gate"
        and effect.comparator == "N1_rule_risk_gate"
        and effect.metric == "false_event_reduction"
    ]
    assert len(n1_vs_rule) == 1
    assert {cell.nuisance_scale for cell in failure_map} == {0.8, 1.2}
    assert all(cell.blocks == 6 for cell in failure_map)

    outputs = write_temporal_risk_evaluation(tmp_path, effects, failure_map, config)
    assert all(path.exists() for path in outputs.values())
    assert "N1_temporal_mlp_risk_gate" in outputs["paired_effects"].read_text(encoding="utf-8")
    assert "mlp_false_event_reduction_vs_rule" in outputs["failure_map"].read_text(encoding="utf-8")
    assert "Frames are not independent" in outputs["report"].read_text(encoding="utf-8")
