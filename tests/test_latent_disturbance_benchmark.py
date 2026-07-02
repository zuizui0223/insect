from statistics import mean

from interaction_sensing.simulation import (
    LatentBenchmarkConfig,
    LatentDisturbanceConfig,
    LatentDisturbanceWorld,
    ReferenceMode,
    run_latent_benchmark,
    write_latent_benchmark,
)


def _mean_metric(results, policy: str, metric: str) -> float:
    values = [getattr(result, metric) for result in results if result.policy == policy]
    numeric = [float(value) for value in values if value is not None]
    assert numeric
    return mean(numeric)


def test_world_has_reproducible_correct_and_broken_references() -> None:
    config = LatentDisturbanceConfig(frames=40, seed=11)
    first = LatentDisturbanceWorld(config)
    second = LatentDisturbanceWorld(config)
    assert first.frames == second.frames
    frame_index = 20
    assert first.reference(frame_index, ReferenceMode.CORRECT) == second.reference(frame_index, ReferenceMode.CORRECT)
    assert first.reference(frame_index, ReferenceMode.CORRECT) != first.reference(
        frame_index,
        ReferenceMode.SPATIALLY_MISMATCHED,
    )
    assert first.reference(frame_index, ReferenceMode.CORRECT) != first.reference(
        frame_index,
        ReferenceMode.TIME_SHIFTED,
    )


def test_correct_reference_reduces_false_events_and_broken_reference_loses_benefit() -> None:
    config = LatentBenchmarkConfig(
        frames=420,
        calibration_replicates=6,
        test_replicates=10,
        nuisance_scales=(1.0,),
        seed=51,
    )
    results, thresholds = run_latent_benchmark(config)
    raw_fpr = _mean_metric(results, "B0_raw_motion", "false_event_rate")
    correct_fpr = _mean_metric(results, "N0_robust_reference", "false_event_rate")
    shifted_fpr = _mean_metric(results, "NC_time_shifted_reference", "false_event_rate")
    assert thresholds["N0_robust_reference"] > 0.0
    assert correct_fpr < raw_fpr
    assert correct_fpr < shifted_fpr
    assert _mean_metric(results, "N0_robust_reference", "recall") > 0.70


def test_writer_emits_held_out_metrics_and_locked_thresholds(tmp_path) -> None:
    config = LatentBenchmarkConfig(
        frames=180,
        calibration_replicates=3,
        test_replicates=4,
        nuisance_scales=(0.6, 1.2),
        seed=8,
    )
    results, thresholds = run_latent_benchmark(config)
    outputs = write_latent_benchmark(tmp_path, results, config, thresholds)
    assert all(path.exists() for path in outputs.values())
    assert "NC_time_shifted_reference" in outputs["metrics"].read_text(encoding="utf-8")
    assert "calibration" in outputs["report"].read_text(encoding="utf-8")
    assert "N2_risk_guided_audit" in outputs["thresholds"].read_text(encoding="utf-8")
