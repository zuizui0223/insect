"""Monte Carlo benchmark for reference-guided latent-disturbance inference.

Every policy receives the same local observation stream. The only intervention is
whether its reference retains the correct shared nuisance cause, is weak, or is
explicitly broken. Thresholds are learned on calibration worlds and evaluated
only on separately seeded held-out worlds.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import exp
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


def run_latent_benchmark(
    config: LatentBenchmarkConfig = LatentBenchmarkConfig(),
) -> tuple[list[LatentBenchmarkResult], dict[str, float]]:
    """Evaluate policies on a held-out Monte Carlo split.

    Thresholds are selected from calibration worlds only. Time-shifted,
    spatially-mismatched, and degraded references are causal negative controls:
    their local observation stream is identical to that of the correct-reference
    policy.
    """

    calibration_worlds = [world for _, _, world in _worlds(config, split="calibration")]
    thresholds: dict[str, float] = {}
    for spec in POLICIES:
        calibration_frames = [
            frame
            for world in calibration_worlds
            for frame in _score_world(world, spec)
        ]
        thresholds[spec.policy.value] = _threshold_at_target_recall(
            calibration_frames,
            config.target_recall,
        )

    results: list[LatentBenchmarkResult] = []
    for nuisance_scale, replicate, world in _worlds(config, split="test"):
        for spec in POLICIES:
            results.append(
                _evaluate_world(
                    scored=_score_world(world, spec),
                    threshold=thresholds[spec.policy.value],
                    spec=spec,
                    nuisance_scale=nuisance_scale,
                    replicate=replicate,
                    audit_fraction=config.audit_fraction,
                )
            )
    return results, thresholds


def _worlds(
    config: LatentBenchmarkConfig,
    *,
    split: str,
) -> Iterable[tuple[float, int, LatentDisturbanceWorld]]:
    count = config.calibration_replicates if split == "calibration" else config.test_replicates
    split_offset = 0 if split == "calibration" else 10_000_000
    for scale_index, scale in enumerate(config.nuisance_scales):
        for replicate in range(count):
            base = config.base_world
            world_config = replace(
                base,
                name=f"{split}|scale={scale:g}|replicate={replicate}",
                frames=config.frames,
                global_motion_sd=base.global_motion_sd * scale,
                sway_amplitude=base.sway_amplitude * scale,
                photometric_amplitude=base.photometric_amplitude * scale,
                nuisance_dominant_threshold=base.nuisance_dominant_threshold * scale,
                seed=config.seed + split_offset + scale_index * 100_000 + replicate,
            )
            yield scale, replicate, LatentDisturbanceWorld(world_config)


def _score_world(world: LatentDisturbanceWorld, spec: PolicySpec) -> list[_ScoredFrame]:
    nuisance_threshold = world.config.nuisance_dominant_threshold
    output: list[_ScoredFrame] = []
    for frame in world.iter_frames():
        reference = world.reference(frame.frame_index, spec.reference_mode)
        score = frame.raw_local_evidence - reference if spec.cancel_reference else frame.raw_local_evidence
        output.append(
            _ScoredFrame(
                true_event=frame.true_local_event,
                true_signal=frame.true_local_signal,
                score=score,
                risk=_predicted_risk(
                    raw_evidence=frame.raw_local_evidence,
                    reference=reference,
                    quality_hint=frame.quality_hint,
                    uses_quality_hint=spec.use_quality_hint,
                    has_reference=spec.reference_mode is not ReferenceMode.ABSENT,
                    centre=nuisance_threshold,
                ),
                nuisance_dominant=(not frame.true_local_event)
                and abs(frame.nuisance_contribution) >= nuisance_threshold,
            )
        )
    return output


def _predicted_risk(
    *,
    raw_evidence: float,
    reference: float,
    quality_hint: float,
    uses_quality_hint: bool,
    has_reference: bool,
    centre: float,
) -> float:
    """Transparent proxy that a later ML/DL nuisance-state model can replace."""

    if not has_reference:
        evidence = abs(raw_evidence)
    elif uses_quality_hint:
        evidence = 0.55 * abs(reference) + 0.45 * quality_hint
    else:
        evidence = abs(reference)
    return _sigmoid((evidence - centre) / max(0.20, centre * 0.35))


def _threshold_at_target_recall(scored: list[_ScoredFrame], target_recall: float) -> float:
    positive_scores = sorted(abs(frame.score) for frame in scored if frame.true_event)
    if not positive_scores:
        raise ValueError("calibration worlds contain no true local events")
    index = min(
        len(positive_scores) - 1,
        max(0, int((1.0 - target_recall) * len(positive_scores))),
    )
    return positive_scores[index]


def _evaluate_world(
    *,
    scored: list[_ScoredFrame],
    threshold: float,
    spec: PolicySpec,
    nuisance_scale: float,
    replicate: int,
    audit_fraction: float,
) -> LatentBenchmarkResult:
    predictions = [abs(frame.score) >= threshold for frame in scored]
    true_events = [frame for frame in scored if frame.true_event]
    non_events = [frame for frame in scored if not frame.true_event]
    true_positive = sum(predicted and frame.true_event for frame, predicted in zip(scored, predictions))
    false_positive = sum(predicted and not frame.true_event for frame, predicted in zip(scored, predictions))
    risks = [frame.risk for frame in scored]
    risk_labels = [float(frame.nuisance_dominant) for frame in scored]
    audit_count = max(1, round(len(scored) * audit_fraction))
    audited = sorted(range(len(scored)), key=lambda index: risks[index], reverse=True)[:audit_count]
    audit_yield = mean(risk_labels[index] for index in audited) if audited else None
    uniform_yield = mean(risk_labels) if risk_labels else 0.0
    distortion = [abs(frame.score - frame.true_signal) for frame in true_events]
    return LatentBenchmarkResult(
        split="test",
        nuisance_scale=nuisance_scale,
        replicate=replicate,
        policy=spec.policy.value,
        reference_mode=spec.reference_mode.value,
        threshold=threshold,
        true_event_windows=len(true_events),
        non_event_windows=len(non_events),
        true_positive_windows=true_positive,
        false_positive_windows=false_positive,
        recall=None if not true_events else true_positive / len(true_events),
        false_event_rate=None if not non_events else false_positive / len(non_events),
        event_signal_distortion=None if not distortion else mean(distortion),
        risk_brier_score=_brier_score(risks, risk_labels),
        risk_expected_calibration_error=_expected_calibration_error(risks, risk_labels),
        nuisance_dominant_windows=int(sum(risk_labels)),
        audit_windows=audit_count,
        audit_failure_yield=audit_yield,
        uniform_failure_yield=uniform_yield,
        audit_lift=None if audit_yield is None or uniform_yield == 0.0 else audit_yield / uniform_yield,
    )


def _brier_score(probabilities: list[float], labels: list[float]) -> float:
    return 0.0 if not probabilities else mean((p - y) ** 2 for p, y in zip(probabilities, labels))


def _expected_calibration_error(probabilities: list[float], labels: list[float], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    error = 0.0
    for bin_index in range(bins):
        low, high = bin_index / bins, (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if low <= probability < high or (bin_index == bins - 1 and probability == 1.0)
        ]
        if members:
            error += len(members) / len(probabilities) * abs(
                mean(probabilities[index] for index in members)
                - mean(labels[index] for index in members)
            )
    return error


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + exp(-value))


def write_latent_benchmark(
    output_dir: str | Path,
    results: Iterable[LatentBenchmarkResult],
    config: LatentBenchmarkConfig,
    thresholds: dict[str, float],
) -> dict[str, Path]:
    """Write tidy held-out metrics, a summary, and all locked assumptions."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_rows = [result.to_dict() for result in results]
    metrics_path = output / "latent_disturbance_metrics.csv"
    summary_path = output / "latent_disturbance_summary.csv"
    config_path = output / "latent_disturbance_config.json"
    thresholds_path = output / "calibration_thresholds.json"
    report_path = output / "latent_disturbance_report.md"
    _write_csv(metrics_path, result_rows)
    summary_rows = _summarise(result_rows)
    _write_csv(summary_path, summary_rows)
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    thresholds_path.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
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
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    grouping = ("policy", "reference_mode", "nuisance_scale")
    for row in rows:
        groups[tuple(row[field] for field in grouping)].append(row)
    numeric = [field for field in rows[0] if field not in {*grouping, "split", "replicate"}]
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(grouping, key))
        summary["replicates"] = len(group)
        for field in numeric:
            values = [float(row[field]) for row in group if row[field] is not None]
            summary[field] = None if not values else mean(values)
        output.append(summary)
    return output


def _render_report(
    config: LatentBenchmarkConfig,
    thresholds: dict[str, float],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Latent-disturbance reference benchmark",
        "",
        "This is a held-out Monte Carlo test of a causal proposition: a correct scene reference can cancel shared nuisance motion while preserving independent local signal. It is not a target-recognition or field-generalisation result.",
        "",
        "## Locked evaluation design",
        f"- calibration replicates / scale: {config.calibration_replicates}",
        f"- held-out test replicates / scale: {config.test_replicates}",
        f"- frames / recording block: {config.frames}",
        f"- nuisance scales: {', '.join(map(str, config.nuisance_scales))}",
        f"- calibration recall target: {config.target_recall:.2f}",
        f"- audit budget: {config.audit_fraction:.0%}",
        "",
        "## Calibration thresholds",
        "```text",
        *[f"{policy}: {threshold:.4f}" for policy, threshold in sorted(thresholds.items())],
        "```",
        "",
        "## Held-out summary",
        "| policy | reference | scale | recall | false-event rate | distortion | Brier | ECE | audit yield | audit lift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {reference_mode} | {nuisance_scale:.2f} | {recall} | {false_event_rate} | {event_signal_distortion} | {risk_brier_score:.3f} | {risk_expected_calibration_error:.3f} | {audit_failure_yield} | {audit_lift} |".format(
                **{
                    **row,
                    "recall": _fmt(row["recall"]),
                    "false_event_rate": _fmt(row["false_event_rate"]),
                    "event_signal_distortion": _fmt(row["event_signal_distortion"]),
                    "audit_failure_yield": _fmt(row["audit_failure_yield"]),
                    "audit_lift": _fmt(row["audit_lift"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "Correct robust-reference cancellation must reduce false-event rate at matched recall, while time-shifted, mismatched, and degraded references lose that benefit. Risk-guided auditing is evaluated separately at the fixed audit budget.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
