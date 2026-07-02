from interaction_sensing.evaluation import fit_audit_calibration
from interaction_sensing.simulation import BenchmarkConfig, run_benchmark
from interaction_sensing.sensing import relative_motion_magnitude


def test_target_relative_motion_removes_co_moving_displacement() -> None:
    assert relative_motion_magnitude((3.5, -1.0), (3.5, -1.0)) == 0.0
    assert relative_motion_magnitude((4.5, -1.0), (3.5, -1.0)) == 1.0


def test_target_relative_policy_reduces_wind_false_events_with_perfect_tracking() -> None:
    results = run_benchmark(
        BenchmarkConfig(
            frames=240,
            replicates=1,
            wind_amplitudes=(10.0,),
            neighbour_distances=(120.0,),
            tracker_error_sds=(0.0,),
            focal_event_start_rate=0.0,
            neighbour_event_start_rate=0.0,
            pass_by_start_rate=0.0,
            shadow_start_rate=0.0,
            audit_probability=0.5,
            seed=11,
        )
    )
    by_policy = {result.policy: result for result in results}
    assert by_policy["fixed_context"].plant_motion_false_events > 0
    assert by_policy["target_relative_attribution"].plant_motion_false_events == 0


def test_target_relative_policy_reduces_close_neighbour_wrong_target_events() -> None:
    results = run_benchmark(
        BenchmarkConfig(
            frames=300,
            replicates=1,
            wind_amplitudes=(0.0,),
            neighbour_distances=(32.0,),
            tracker_error_sds=(0.0,),
            focal_event_start_rate=0.0,
            neighbour_event_start_rate=0.25,
            pass_by_start_rate=0.0,
            shadow_start_rate=0.0,
            audit_probability=0.5,
            seed=12,
        )
    )
    by_policy = {result.policy: result for result in results}
    assert by_policy["fixed_context"].wrong_target_events > 0
    assert by_policy["target_relative_attribution"].wrong_target_events < by_policy["fixed_context"].wrong_target_events


def test_audit_calibration_recovers_binary_truth_count() -> None:
    # p = 0.8, phi = 0.1 in the labelled audit windows.
    audit_rows = [(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] + [(False, False)] * 9
    calibration = fit_audit_calibration(audit_rows)
    estimate = calibration.corrected_truth_count(total_windows=100, observed_positive_windows=38)
    assert calibration.detection_probability == 0.8
    assert calibration.false_positive_probability == 0.1
    assert estimate == 40.0
