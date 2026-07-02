"""Run truth-labelled pre-field comparisons of observation policies."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Iterable

from interaction_sensing.evaluation import fit_audit_calibration

from .policies import FixedContextPolicy, TargetRelativeAttributionPolicy
from .world import LatentKind, ScenarioConfig, SyntheticWorld


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Factorial benchmark grid and stochastic replication settings."""

    frames: int = 900
    replicates: int = 20
    wind_amplitudes: tuple[float, ...] = (0.0, 4.0, 10.0)
    neighbour_distances: tuple[float, ...] = (32.0, 64.0, 120.0)
    tracker_error_sds: tuple[float, ...] = (0.0, 0.5, 2.0)
    seed: int = 20260702
    audit_probability: float = 0.10
    candidate_detection_probability: float = 0.90
    focal_event_start_rate: float = 0.018
    neighbour_event_start_rate: float = 0.014
    pass_by_start_rate: float = 0.012
    shadow_start_rate: float = 0.006

    def __post_init__(self) -> None:
        if self.frames <= 0 or self.replicates <= 0:
            raise ValueError("frames and replicates must be positive")
        if not self.wind_amplitudes or not self.neighbour_distances or not self.tracker_error_sds:
            raise ValueError("benchmark grids cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    scenario_name: str
    replicate: int
    policy: str
    wind_amplitude: float
    neighbour_distance: float
    tracker_error_sd: float
    truth_focal_events: int
    observed_focal_events: int
    true_positive_events: int
    false_events: int
    missed_events: int
    wrong_target_events: int
    plant_motion_false_events: int
    ambiguous_events: int
    precision: float | None
    recall: float | None
    true_focal_windows: int
    observed_focal_windows: int
    raw_window_bias: float | None
    audit_windows: int
    audit_detection_probability: float | None
    audit_false_positive_probability: float | None
    audit_adjusted_truth_windows: float | None
    audit_adjusted_window_bias: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(config: BenchmarkConfig = BenchmarkConfig()) -> list[BenchmarkResult]:
    """Run all scenario cells and return one row per policy/replicate.

    The same latent world and sampled observations are supplied to both policies
    within each replicate. Differences therefore reflect policy design rather
    than different synthetic data draws.
    """

    results: list[BenchmarkResult] = []
    scenario_index = 0
    for wind in config.wind_amplitudes:
        for neighbour_distance in config.neighbour_distances:
            for tracker_error in config.tracker_error_sds:
                for replicate in range(config.replicates):
                    seed = config.seed + scenario_index * 100_000 + replicate
                    scenario = ScenarioConfig(
                        name=f"wind={wind:g}|neighbour={neighbour_distance:g}|tracker_sd={tracker_error:g}",
                        frames=config.frames,
                        wind_amplitude=wind,
                        neighbour_distance=neighbour_distance,
                        target_tracker_error_sd=tracker_error,
                        candidate_detection_probability=config.candidate_detection_probability,
                        focal_event_start_rate=config.focal_event_start_rate,
                        neighbour_event_start_rate=config.neighbour_event_start_rate,
                        pass_by_start_rate=config.pass_by_start_rate,
                        shadow_start_rate=config.shadow_start_rate,
                        audit_probability=config.audit_probability,
                        seed=seed,
                    )
                    results.extend(_run_scenario(scenario, replicate=replicate))
                scenario_index += 1
    return results


def _run_scenario(scenario: ScenarioConfig, *, replicate: int) -> list[BenchmarkResult]:
    world = SyntheticWorld(scenario)
    policies = [
        FixedContextPolicy(scenario),
        TargetRelativeAttributionPolicy(scenario, seed=scenario.seed + 17),
    ]
    states = {policy.name: _PolicyAccumulator(policy.name, world, scenario) for policy in policies}
    audit_rng = Random(scenario.seed + 29)

    for target_frame, observations in world.iter_frames():
        truth_focal_window = any(
            event.kind is LatentKind.FOCAL_INTERACTION and event.active(target_frame.frame_index)
            for event in world.events
        )
        audit_this_window = audit_rng.random() < scenario.audit_probability
        for policy in policies:
            policy.begin_frame(target_frame)
            decisions = [policy.decide(observation, target_frame) for observation in observations]
            state = states[policy.name]
            system_focal_window = any(decision.focal_counted for decision in decisions)
            state.observe_window(
                truth_focal=truth_focal_window,
                system_focal=system_focal_window,
                audited=audit_this_window,
            )
            for observation, decision in zip(observations, decisions):
                state.observe_event(observation.event_id, observation.kind, decision.focal_counted, decision.ambiguous)

    return [state.to_result(scenario, replicate) for state in states.values()]


@dataclass(slots=True)
class _PolicyAccumulator:
    policy: str
    world: SyntheticWorld
    scenario: ScenarioConfig
    event_focal_counted: dict[str, bool] = field(default_factory=dict)
    event_ambiguous: dict[str, bool] = field(default_factory=dict)
    true_focal_windows: int = 0
    observed_focal_windows: int = 0
    audit_rows: list[tuple[bool, bool]] = field(default_factory=list)

    def observe_event(self, event_id: str, kind: LatentKind, focal_counted: bool, ambiguous: bool) -> None:
        if focal_counted:
            self.event_focal_counted[event_id] = True
        else:
            self.event_focal_counted.setdefault(event_id, False)
        if ambiguous:
            self.event_ambiguous[event_id] = True
        else:
            self.event_ambiguous.setdefault(event_id, False)

    def observe_window(self, *, truth_focal: bool, system_focal: bool, audited: bool) -> None:
        self.true_focal_windows += int(truth_focal)
        self.observed_focal_windows += int(system_focal)
        if audited:
            self.audit_rows.append((truth_focal, system_focal))

    def to_result(self, scenario: ScenarioConfig, replicate: int) -> BenchmarkResult:
        focal_events = [event for event in self.world.events if event.kind is LatentKind.FOCAL_INTERACTION]
        nonfocal_events = [event for event in self.world.events if event.kind is not LatentKind.FOCAL_INTERACTION]
        true_positive = sum(self.event_focal_counted.get(event.event_id, False) for event in focal_events)
        false_events = sum(self.event_focal_counted.get(event.event_id, False) for event in nonfocal_events)
        wrong_target = sum(
            self.event_focal_counted.get(event.event_id, False)
            for event in self.world.events
            if event.kind is LatentKind.NEIGHBOUR_INTERACTION
        )
        plant_motion_false = sum(
            self.event_focal_counted.get(event.event_id, False)
            for event in self.world.events
            if event.kind is LatentKind.TARGET_SWAY
        )
        ambiguous = sum(self.event_ambiguous.get(event.event_id, False) for event in self.world.events)
        observed_focal_events = true_positive + false_events
        precision = None if observed_focal_events == 0 else true_positive / observed_focal_events
        recall = None if not focal_events else true_positive / len(focal_events)
        raw_window_bias = _relative_bias(self.observed_focal_windows, self.true_focal_windows)
        calibration = fit_audit_calibration(self.audit_rows)
        adjusted = calibration.corrected_truth_count(
            total_windows=scenario.frames,
            observed_positive_windows=self.observed_focal_windows,
        )
        adjusted_bias = None if adjusted is None else _relative_bias(adjusted, self.true_focal_windows)
        return BenchmarkResult(
            scenario_name=scenario.name,
            replicate=replicate,
            policy=self.policy,
            wind_amplitude=scenario.wind_amplitude,
            neighbour_distance=scenario.neighbour_distance,
            tracker_error_sd=scenario.target_tracker_error_sd,
            truth_focal_events=len(focal_events),
            observed_focal_events=observed_focal_events,
            true_positive_events=true_positive,
            false_events=false_events,
            missed_events=len(focal_events) - true_positive,
            wrong_target_events=wrong_target,
            plant_motion_false_events=plant_motion_false,
            ambiguous_events=ambiguous,
            precision=precision,
            recall=recall,
            true_focal_windows=self.true_focal_windows,
            observed_focal_windows=self.observed_focal_windows,
            raw_window_bias=raw_window_bias,
            audit_windows=calibration.audit_windows,
            audit_detection_probability=calibration.detection_probability,
            audit_false_positive_probability=calibration.false_positive_probability,
            audit_adjusted_truth_windows=adjusted,
            audit_adjusted_window_bias=adjusted_bias,
        )


def _relative_bias(estimate: float, truth: int) -> float | None:
    if truth <= 0:
        return None
    return (estimate - truth) / truth


def write_benchmark(
    output_dir: str | Path,
    results: Iterable[BenchmarkResult],
    config: BenchmarkConfig,
) -> dict[str, Path]:
    """Write tidy replicate results, scenario summaries, and a readable report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_rows = [result.to_dict() for result in results]
    metrics_path = output / "scenario_metrics.csv"
    _write_csv(metrics_path, result_rows)

    summary_rows = _summarise(result_rows)
    summary_path = output / "benchmark_summary.csv"
    _write_csv(summary_path, summary_rows)

    assumptions_path = output / "assumptions.json"
    assumptions_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output / "benchmark_report.md"
    report_path.write_text(_render_report(config, summary_rows), encoding="utf-8")
    return {
        "scenario_metrics": metrics_path,
        "benchmark_summary": summary_path,
        "assumptions": assumptions_path,
        "report": report_path,
    }


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouping = ("policy", "wind_amplitude", "neighbour_distance", "tracker_error_sd")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in grouping)].append(row)

    numeric_fields = [
        "truth_focal_events",
        "observed_focal_events",
        "true_positive_events",
        "false_events",
        "missed_events",
        "wrong_target_events",
        "plant_motion_false_events",
        "ambiguous_events",
        "precision",
        "recall",
        "true_focal_windows",
        "observed_focal_windows",
        "raw_window_bias",
        "audit_windows",
        "audit_detection_probability",
        "audit_false_positive_probability",
        "audit_adjusted_truth_windows",
        "audit_adjusted_window_bias",
    ]
    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(grouping, key))
        row["replicates"] = len(group)
        for field in numeric_fields:
            values = [float(item[field]) for item in group if item[field] is not None]
            row[field] = None if not values else mean(values)
        summary.append(row)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(config: BenchmarkConfig, summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Synthetic observability benchmark",
        "",
        "## Scope",
        "",
        "This report is a **pre-field mechanistic stress test**, not evidence of field generalisation. It compares a fixed-context motion baseline against a target-relative, attribution-aware policy under fully known synthetic truth.",
        "",
        "## Benchmark grid",
        "",
        f"- Replicates per scenario: {config.replicates}",
        f"- Frames per replicate: {config.frames}",
        f"- Wind amplitudes: {', '.join(map(str, config.wind_amplitudes))}",
        f"- Neighbour distances: {', '.join(map(str, config.neighbour_distances))}",
        f"- Tracker-error SDs: {', '.join(map(str, config.tracker_error_sds))}",
        "",
        "## Summary by condition",
        "",
        "| policy | wind | neighbour distance | tracker SD | precision | recall | false events | wrong-target events | plant-motion false events | raw window bias | audit-adjusted bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {wind_amplitude:.2f} | {neighbour_distance:.2f} | {tracker_error_sd:.2f} | {precision} | {recall} | {false_events:.2f} | {wrong_target_events:.2f} | {plant_motion_false_events:.2f} | {raw_window_bias} | {audit_adjusted_window_bias} |".format(
                policy=row["policy"],
                wind_amplitude=float(row["wind_amplitude"]),
                neighbour_distance=float(row["neighbour_distance"]),
                tracker_error_sd=float(row["tracker_error_sd"]),
                precision=_fmt(row["precision"]),
                recall=_fmt(row["recall"]),
                false_events=float(row["false_events"] or 0.0),
                wrong_target_events=float(row["wrong_target_events"] or 0.0),
                plant_motion_false_events=float(row["plant_motion_false_events"] or 0.0),
                raw_window_bias=_fmt(row["raw_window_bias"]),
                audit_adjusted_window_bias=_fmt(row["audit_adjusted_window_bias"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A useful method result is a pattern, not one favourable cell: as wind increases, fixed-context false events should rise; as neighbour distance decreases, wrong-target risk should rise; and the proposed policy should reduce those errors while retaining focal recall under non-zero tracker error. Audit-adjusted bias should be interpreted only when the audit sample contains both truth-positive and truth-negative windows.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"
