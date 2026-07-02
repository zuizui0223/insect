from interaction_sensing.simulation import (
    LatentBenchmarkConfig,
    LatentEvaluationConfig,
    evaluate_latent_results,
    run_latent_benchmark,
    write_latent_evaluation,
)


def test_paired_evaluator_reports_reference_specific_effects(tmp_path) -> None:
    benchmark = LatentBenchmarkConfig(
        frames=420,
        calibration_replicates=6,
        test_replicates=10,
        nuisance_scales=(0.7, 1.1),
        seed=21,
    )
    results, _ = run_latent_benchmark(benchmark)
    config = LatentEvaluationConfig(bootstrap_resamples=300, seed=21)
    effects, failure_map = evaluate_latent_results(results, config)

    robust_vs_raw = [
        effect
        for effect in effects
        if effect.condition == "nuisance_scale=1.1"
        and effect.intervention == "N0_robust_reference"
        and effect.comparator == "B0_raw_motion"
        and effect.metric == "false_event_reduction"
    ]
    assert len(robust_vs_raw) == 1
    assert robust_vs_raw[0].estimate > 0.0
    assert robust_vs_raw[0].blocks == 10

    specificity = [
        effect
        for effect in effects
        if effect.condition == "nuisance_scale=1.1"
        and effect.intervention == "N0_robust_reference"
        and effect.comparator == "NC_time_shifted_reference"
        and effect.metric == "false_event_reduction"
    ]
    assert len(specificity) == 1
    assert specificity[0].estimate > 0.0
    assert {cell.nuisance_scale for cell in failure_map} == {0.7, 1.1}

    outputs = write_latent_evaluation(tmp_path, effects, failure_map, config)
    assert all(path.exists() for path in outputs.values())
    assert "false_event_reduction" in outputs["paired_effects"].read_text(encoding="utf-8")
    assert "reference_specificity" in outputs["failure_map"].read_text(encoding="utf-8")
    assert "Frames are not treated as independent" in outputs["report"].read_text(encoding="utf-8")
