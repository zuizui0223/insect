"""Monte Carlo benchmark for reference-guided latent-disturbance inference.

This benchmark deliberately tests a causal mechanism rather than an object
recogniser.  Every policy sees the same local observation stream.  Only the
reference condition changes: correct shared-scene references, weak single-region
references, or explicitly broken reference controls.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .latent_disturbance import LatentDisturbanceConfig, LatentDisturbanceWorld, ReferenceMode


class LatentPolicy(str, Enum):
    RAW_MOTION = "B0_raw_motion"
    SINGLE_REFERENCE = "B2_single_reference"
    ROBUST_REFERENCE = "N0_robust_reference"
    QUALITY_AWARE = "N1_quality_aware"
    RISK_GUIDED_AUDIT = "N2_risk_guided_audit"
    TIME_SHIFTED_REFERENCE = "NC_time_shifted_reference"
    SPATIALLY_MISMATCHED_REFERENCE = "NC_spatially_mismatched_reference"
    DEGRADED_REFERENCE = "NC_degraded_reference"


@dataclass(frozen=True, slots=True)
class PolicySpec:
    policy: LatentPolicy
    reference_mode: ReferenceMode
    cancel_reference: bool
    use_quality_hint: bool
    uses_risk_guided_audit: bool


POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec(LatentPolicy.RAW_MOTION, ReferenceMode.ABSENT, False, False, False),
    PolicySpec(LatentPolicy.SINGLE_REFERENCE, ReferenceMode.SINGLE_REGION, True, False, False),
    PolicySpec(LatentPolicy.ROBUST_REFERENCE, ReferenceMode.CORRECT, True, False, False),
    PolicySpec(LatentPolicy.QUALITY_AWARE, ReferenceMode.CORRECT, True, True, False),
    PolicySpec(LatentPolicy.RISK_GUIDED_AUDIT, ReferenceMode.CORRECT, True, True, True),
    PolicySpec(LatentPolicy.TIME_SHIFTED_REFERENCE, ReferenceMode.TIME_SHIFTED, True, False, False),
    PolicySpec(LatentPolicy.SPATIALLY_MISMATCHED_REFERENCE, ReferenceMode.SPATIALLY_MISMATCHED, True, False, False),
    PolicySpec(LatentPolicy.DEGRADED_REFERENCE, ReferenceMode.DEGRADED, True, False, False),
)


@dataclass(frozen=True, slots=True)
class LatentBenchmarkConfig:
    """Predeclared Monte Carlo conditions and split settings."""

    frames: int = 900
    calibration_replicates: int = 24
    test_replicates: int = 32
    nuisance_scales: tuple[float, ...] = (0.55, 1.0, 1.45)
    target_recall: float = 0.85
    audit_fraction: float = 0.10
    seed: int = 20260702
    base_world: LatentDisturbanceConfig = LatentDisturbanceConfig()

    def __post_init__(self) -> None:
        if self.frames <= 0 or self.calibration_replicates <= 0 or self.test_replicates <= 0:
            raise ValueError("frames and replicate counts must be positive")
        if not self.nuisance_scales or any(scale <= 0.0 for scale in self.nuisance_scales):
            raise ValueError("nuisance_scales must contain positive values")
        if not 0.0 < self.target_recall < 1.0:
            raise ValueError("target_recall must lie in (0, 1)")
        if not 0.0 < self.audit_fraction < 1.0:
            raise ValueError("audit_fraction must lie in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["base_world"] = self.base_world.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class LatentBenchmarkResult:
    split: str
    nuisance_scale: float
    replicate: int
    policy: str
    reference_mode: str
    threshold: float
    true_event_windows: int
    non_event_windows: int
    true_positive_windows: int
    false_positive_windows: int
    recall: float | None
    false_event_rate: float | None
    event_signal_distortion: float | None
    risk_brier_score: float
    risk_expected_calibration_error: float
    nuisance_dominant_windows: int
    audit_windows: int
    audit_failure_yield: float | None
    uniform_failure_yield: float
    audit_lift: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ScoredFrame:
    true_event: bool
    true_signal: float
    score: float
    risk: float
    nuisance_dominant: bool


def run_latent_benchmark(config: LatentBenchmarkConfig = LatentBenchmarkConfig()) -> tuple[list[LatentBenchmarkResult], dict[str, float]]:
    """Run a held-out Monte Carlo benchmark with calibration-only thresholds.

    Thresholds are selected using calibration worlds only, at a predeclared true
    event recall.  Test worlds are seeded separately and never used to tune a
    policy.  Broken-reference policies are causal negative controls: they use
    identical local observations but invalid references.
    """

    calibration_worlds = list(_worlds(config, split="calibration"))
    thresholds = {
        spec.policy.value: _threshold_at_target_recall(
            [frame for world in calibration_worlds for frame in _score_world(world, spec)],
            config.target_recall,
        )
        for spec in POLICIES
    }

    results: list[LatentBenchmarkResult] = []
    for nuisance_scale, replicate, world in _worlds(config, split="test"):
        for spec in POLICIES:
            scored = _score_world(world, spec)
            results.append(
                _evaluate_world(
                    scored=scored,
                    threshold=thresholds[spec.policy.value],
                    spec=spec,
                    nuisance_scale=nuisance_scale,
                    replicate=replicate,
                    audit_fraction=config.audit_fraction,
                    split="test",
                )
            )
    return results, thresholds


def _worlds(
    config: LatentBenchmarkConfig,
    *,
    split: str,
) -> Iterable[tuple[float, int, LatentDisturbanceWorld]]:
    replicate_count = config.calibration_replicates if split == "calibration" else config.test_replicates
    split_offset = 0 if split == "calibration" else 10_000_000
    for scale_index, scale in enumerate(config.nuisance_scales):
        for replicate in range(replicate_count):
            seed = config.seed + split_offset + scale_index * 100_000 + replicate
            base = config.base_world
            world_config = replace(
                base,
                name=f"{split}|scale={scale:g}|replicate={replicate}",
                frames=config.frames,
                global_motion_sd=base.global_motion_sd * scale,
                sway_amplitude=base.sway_amplitude * scale,
                photometric_amplitude=base.photometric_amplitude * scale,
                nuisance_dominant_threshold=base.nuisance_dominant_threshold * scale,
                seed=seed,
            )
            yield scale, replicate, LatentDisturbanceWorld(world_config)


def _score_world(world: LatentDisturbanceWorld, spec: PolicySpec) -> list[_ScoredFrame]:
    frames: list[_ScoredFrame] = []
    threshold = world.config.nuisance_dominant_threshold
    for frame in world.iter_frames():
        reference = world.reference(frame.frame_index, spec.reference_mode)
        score = frame.raw_local_evidence - reference if spec.cancel_reference else frame.raw_local_evidence
        risk = _predicted_risk(
            raw_evidence=frame.raw_local_evidence,
            reference=reference,
            quality_hint=frame.quality_hint,
            uses_quality_hint=spec.use_quality_hint,
            has_reference=spec.reference_mode is not ReferenceMode.ABSENT,
            centre=threshold,
        )
        frames.append(
            _ScoredFrame(
                true_event=frame.true_local_event,
                true_signal=frame.true_local_signal,
                score=score,
                risk=risk,
                nuisance_dominant=(not frame.true_local_event)
                and abs(frame.nuisance_contribution) >= threshold,
            )
        )
    return frames


def _predicted_risk(
    *,
    raw_evidence: float,
    reference: float,
    quality_hint: float,
    uses_quality_hint: bool,
    has_reference: bool,
    centre: float,
) -> float:
    """Transparent risk proxy; later hardware models can replace this component."""

    if not has_reference:
        evidence = abs(raw_evidence)
    elif uses_quality_hint:
        evidence = 0.55 * abs(reference) + 0.45 * quality_hint
    else:
        evidence = abs(reference)
    return _sigmoid((evidence - centre) / max(0.20, centre * 0.35))


def _threshold_at_target_recall(scored: list[_ScoredFrame], target_recall: float) -> float:
    positives = sorted(abs(frame.score) for frame in scored if frame.true_event)
    if not positives:
        raise ValueError("calibration worlds contain no true local events")
    index = min(len(positives) - 1, max(0, int((1.0 - target_recall) * len(positives))))
    return positives[index]


def _evaluate_world(
    *,
    scored: list[_ScoredFrame],
    threshold: float,
    spec: PolicySpec,
    nuisance_scale: float,
    replicate: int,
    audit_fraction: float,
    split: str,
) -> LatentBenchmarkResult:
    predictions = [abs(frame.score) >= threshold for frame in scored]
    true_events = [frame for frame in scored if frame.true_event]
    non_events = [frame for frame in scored if not frame.true_event]
    true_positive = sum(predicted and frame.true_event for frame, predicted in zip(scored, predictions))
    false_positive = sum(predicted and not frame.true_event for frame, predicted in zip(scored, predictions))
    recall = None if not true_events else true_positive / len(true_events)
    false_event_rate = None if not non_events else false_positive / len(non_events)
    distortion_values = [abs(frame.score - frame.true_signal) for frame in true_events]
    risk_labels = [float(frame.nuisance_dominant) for frame in scored]
    risks = [frame.risk for frame in scored]
    audit_count = max(1, round(len(scored) * audit_fraction))
    ranked = sorted(range(len(scored)), key=lambda index: risks[index], reverse=True)
    audited = ranked[:audit_count]
    audit_yield = mean(risk_labels[index] for index in audited) if audited else None
    uniform_yield = mean(risk_labels) if risk_labels else 0.0
    audit_lift = None if uniform_yield == 0.0 or audit_yield is None else audit_yield / uniform_yield
    return LatentBenchmarkResult(
        split=split,
        nuisance_scale=nuisance_scale,
        replicate=replicate,
        policy=spec.policy.value,
        reference_mode=spec.reference_mode.value,
        threshold=threshold,
        true_event_windows=len(true_events),
        non_event_windows=len(non_events),
        true_positive_windows=true_positive,
        false_positive_windows=false_positive,
        recall=recall,
        false_event_rate=false_event_rate,
        event_signal_distortion=None if not distortion_values else mean(distortion_values),
        risk_brier_score=_brier_score(risks, risk_labels),
        risk_expected_calibration_error=_expected_calibration_error(risks, risk_labels),
        nuisance_dominant_windows=int(sum(risk_labels)),
        audit_windows=audit_count,
        audit_failure_yield=audit_yield,
        uniform_failure_yield=uniform_yield,
        audit_lift=audit_lift,
    )


def _brier_score(probabilities: list[float], labels: list[float]) -> float:
    if not probabilities:
        return 0.0
    return mean((probability - label) ** 2 for probability, label in zip(probabilities, labels))


def _expected_calibration_error(probabilities: list[float], labels: list[float], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    error = 0.0
    count = len(probabilities)
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if low <= probability < high or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = mean(probabilities[index] for index in members)
        frequency = mean(labels[index] for index in members)
        error += len(members) / count * abs(confidence - frequency)
    return error


def _sigmoid(value: float) -> float:
    bounded = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + __import__("math").exp(-bounded))


def write_latent_benchmark(
    output_dir: str | Path,
    results: Iterable[LatentBenchmarkResult],
    config: LatentBenchmarkConfig,
    thresholds: dict[str, float],
) -> dict[str, Path]:
    """Write tidy held-out results and a compact, auditable report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_rows = [result.to_dict() for result in results]
    metrics_path = output / "latent_disturbance_metrics.csv"
    _write_csv(metrics_path, result_rows)

    summary_rows = _summarise(result_rows)
    summary_path = output / "latent_disturbance_summary.csv"
    _write_csv(summary_path, summary_rows)

    config_path = output / "latent_disturbance_config.json"
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    thresholds_path = output / "calibration_thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output / "latent_disturbance_report.md"
    report_path.write_text(_render_report(config, thresholds, summary_rows), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "config": config_path,
        "thresholds": thresholds_path,
        "report": report_path,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouping = ("policy", "reference_mode", "nuisance_scale")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in grouping)].append(row)
    numeric = [
        "threshold",
        "true_event_windows",
        "non_event_windows",
        "true_positive_windows",
        "false_positive_windows",
        "recall",
        "false_event_rate",
        "event_signal_distortion",
        "risk_brier_score",
        "risk_expected_calibration_error",
        "nuisance_dominant_windows",
        "audit_windows",
        "audit_failure_yield",
        "uniform_failure_yield",
        "audit_lift",
    ]
    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(grouping, key))
        row["replicates"] = len(group)
        for field in numeric:
            values = [float(item[field]) for item in group if item[field] is not None]
            row[field] = None if not values else mean(values)
        summary.append(row)
    return summary


def _render_report(
    config: LatentBenchmarkConfig,
    thresholds: dict[str, float],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Latent-disturbance reference benchmark",
        "",
        "## Scope",
        "",
        "This is a truth-labelled Monte Carlo test of a causal proposition: a correct scene reference can cancel shared nuisance motion while preserving independent local signal. It is not an organism-recognition or field-generalisation result.",
        "",
        "## Locked evaluation design",
        "",
        f"- Calibration replicates per nuisance scale: {config.calibration_replicates}",
        f"- Held-out test replicates per nuisance scale: {config.test_replicates}",
        f"- Frames per recording block: {config.frames}",
        f"- Nuisance scales: {', '.join(map(str, config.nuisance_scales))}",
        f"- Thresholds calibrated to target true-event recall: {config.target_recall:.2f}",
        f"- Audit budget: {config.audit_fraction:.0%} of windows",
        "",
        "## Causal negative controls",
        "",
        "The time-shifted, spatially mismatched, and degraded-reference policies receive the identical local observation stream as the robust-reference policy. They differ only in whether the reference retains the correct shared cause.",
        "",
        "## Calibration thresholds",
        "",
        "```text",
        *[f"{policy}: {threshold:.4f}" for policy, threshold in sorted(thresholds.items())],
        "```",
        "",
        "## Held-out summary",
        "",
        "| policy | reference | scale | recall | false-event rate | signal distortion | risk Brier | risk ECE | audit yield | audit lift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {reference_mode} | {nuisance_scale:.2f} | {recall} | {false_event_rate} | {event_signal_distortion} | {risk_brier_score:.3f} | {risk_expected_calibration_error:.3f} | {audit_failure_yield} | {audit_lift} |".format(
                **{
                    **row,
                    "recall": _format(row["recall"]),
                    "false_event_rate": _format(row["false_event_rate"]),
                    "event_signal_distortion": _format(row["event_signal_distortion"]),
                    "audit_failure_yield": _format(row["audit_failure_yield"]),
                    "audit_lift": _format(row["audit_lift"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A valid mechanistic result requires robust correct-reference cancellation to reduce false-event rate at matched recall, while broken-reference controls lose that advantage and independent local signal remains detectable. Risk-guided audit is evaluated separately by its yield under a fixed audit budget.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
