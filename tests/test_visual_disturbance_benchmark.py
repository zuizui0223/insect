from statistics import mean

import numpy as np

from interaction_sensing.simulation import (
    VisualBenchmarkConfig,
    VisualDisturbanceWorld,
    VisualWorldConfig,
    extract_visual_features,
    run_visual_benchmark,
    write_visual_benchmark,
)


def _mean_metric(results, policy: str, metric: str) -> float:
    values = [getattr(result, metric) for result in results if result.policy == policy]
    numeric = [float(value) for value in values if value is not None]
    assert numeric
    return mean(numeric)


def test_visual_world_and_pixel_features_are_reproducible() -> None:
    config = VisualWorldConfig(frames=60, seed=17)
    first = VisualDisturbanceWorld(config)
    second = VisualDisturbanceWorld(config)
    assert len(first.frames_data) == 60
    assert np.array_equal(first.frames_data[12].image, second.frames_data[12].image)
    features = extract_visual_features(first, reference_delay_frames=5, alignment_search_radius=4)
    assert len(features) == 59
    assert any(feature.true_local_event for feature in features)
    assert any(abs(feature.global_shift_x) + abs(feature.global_shift_y) > 0 for feature in features)
    assert all(np.isfinite(feature.risk_proxy) for feature in features)


def test_correct_visual_reference_beats_raw_and_time_shifted_controls() -> None:
    config = VisualBenchmarkConfig(
        frames=220,
        calibration_replicates=6,
        test_replicates=10,
        nuisance_scales=(1.0,),
        seed=31,
    )
    results, thresholds = run_visual_benchmark(config)
    raw_fpr = _mean_metric(results, "B0_raw_pixel_difference", "false_event_rate")
    robust_fpr = _mean_metric(results, "N0_robust_visual_reference", "false_event_rate")
    delayed_fpr = _mean_metric(results, "NC_time_shifted_visual_reference", "false_event_rate")
    mismatch_fpr = _mean_metric(results, "NC_spatially_mismatched_visual_reference", "false_event_rate")
    assert thresholds["N0_robust_visual_reference"] > 0.0
    assert robust_fpr < raw_fpr
    assert robust_fpr < delayed_fpr
    assert robust_fpr < mismatch_fpr
    assert _mean_metric(results, "N0_robust_visual_reference", "recall") > 0.70


def test_visual_benchmark_writer_records_image_derived_scope(tmp_path) -> None:
    config = VisualBenchmarkConfig(
        frames=100,
        calibration_replicates=3,
        test_replicates=4,
        nuisance_scales=(0.7, 1.2),
        seed=41,
    )
    results, thresholds = run_visual_benchmark(config)
    outputs = write_visual_benchmark(tmp_path, results, config, thresholds)
    assert all(path.exists() for path in outputs.values())
    assert "N0_robust_visual_reference" in outputs["metrics"].read_text(encoding="utf-8")
    report = outputs["report"].read_text(encoding="utf-8")
    assert "rendered pixels" in report
    assert "time-shifted" in report
