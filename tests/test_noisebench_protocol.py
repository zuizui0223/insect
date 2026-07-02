from interaction_sensing.noisebench import (
    NoiseBenchConfig,
    NoiseBenchPlan,
    PerturbationKind,
    build_noisebench_plan,
    write_noisebench_plan,
)


def test_noisebench_includes_controls_single_and_mixed_conditions() -> None:
    config = NoiseBenchConfig(replicates=1, intensities=(0.5,), include_mixed_disturbance=True, seed=1)
    plan = build_noisebench_plan(config)
    scenario_ids = {scenario.scenario_id for scenario in plan.scenarios}
    assert "stable-r1" in scenario_ids
    assert any("camera_shake" in scenario_id for scenario_id in scenario_ids)
    assert any(scenario.metadata["condition_family"] == "mixed" for scenario in plan.scenarios)
    assert all(scenario.is_target_agnostic for scenario in plan.scenarios)


def test_noisebench_plan_is_seed_reproducible() -> None:
    config = NoiseBenchConfig(replicates=1, intensities=(0.3, 0.9), seed=7)
    first = build_noisebench_plan(config)
    second = build_noisebench_plan(config)
    assert [scenario.scenario_id for scenario in first.scenarios] == [scenario.scenario_id for scenario in second.scenarios]


def test_single_perturbation_has_known_error_hypothesis() -> None:
    plan = build_noisebench_plan(NoiseBenchConfig(replicates=1, intensities=(0.5,), include_mixed_disturbance=False))
    shake = next(scenario for scenario in plan.scenarios if "camera_shake" in scenario.scenario_id)
    perturbation = shake.perturbations[0]
    assert perturbation.kind is PerturbationKind.CAMERA_SHAKE
    assert set(perturbation.kind.expected_error_channels) == {"false_event", "missed_event"}
    assert perturbation.end_seconds <= shake.duration_seconds


def test_noisebench_writer_creates_preregistered_manifest_and_windows(tmp_path) -> None:
    plan = build_noisebench_plan(NoiseBenchConfig(replicates=1, intensities=(0.5,), include_mixed_disturbance=False))
    stable = next(scenario for scenario in plan.scenarios if scenario.scenario_id == "stable-r1")
    stable_first = NoiseBenchPlan(config=plan.config, scenarios=(stable, *[s for s in plan.scenarios if s != stable]))
    outputs = write_noisebench_plan(tmp_path, stable_first)
    assert all(path.exists() for path in outputs.values())
    manifest = outputs["manifest"].read_text(encoding="utf-8")
    windows = outputs["windows"].read_text(encoding="utf-8")
    assert "target_agnostic" in manifest
    assert "noise_source" in manifest
    assert "active_noise_sources" in windows
    assert "camera_shake" in manifest
