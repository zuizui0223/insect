"""Review-oriented evaluation for latent-disturbance benchmark results.

The independent unit is a matched Monte Carlo recording block, identified by
``nuisance_scale × replicate``. Frames are never treated as independent
replicates. This module therefore computes paired effect estimates and
percentile bootstrap intervals across blocks.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Callable, Iterable

from .latent_benchmark import LatentBenchmarkResult, LatentPolicy


@dataclass(frozen=True, slots=True)
class LatentEvaluationConfig:
    """Locked rules for interpreting held-out latent-benchmark effects."""

    bootstrap_resamples: int = 2_000
    confidence_level: float = 0.95
    recall_noninferiority_margin: float = 0.05
    minimum_false_event_reduction: float = 0.0
    seed: int = 20260702

    def __post_init__(self) -> None:
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if self.recall_noninferiority_margin < 0.0:
            raise ValueError("recall_noninferiority_margin must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedEffect:
    """Beneficial paired difference: positive values always favour intervention."""

    condition: str
    intervention: str
    comparator: str
    metric: str
    blocks: int
    estimate: float
    ci_low: float
    ci_high: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureMapCell:
    """Per-condition conclusion without p-value-only decision making."""

    nuisance_scale: float
    blocks: int
    false_event_reduction_vs_raw: float
    false_event_reduction_vs_raw_ci_low: float
    false_event_reduction_vs_raw_ci_high: float
    recall_change_vs_raw: float
    recall_change_vs_raw_ci_low: float
    recall_change_vs_raw_ci_high: float
    reference_specificity_reduction: float
    reference_specificity_ci_low: float
    reference_specificity_ci_high: float
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_EFFECTS: tuple[tuple[str, Callable[[LatentBenchmarkResult, LatentBenchmarkResult], float], str], ...] = (
    (
        "false_event_reduction",
        lambda intervention, comparator: _required(comparator.false_event_rate) - _required(intervention.false_event_rate),
        "positive means fewer false events",
    ),
    (
        "recall_change",
        lambda intervention, comparator: _required(intervention.recall) - _required(comparator.recall),
        "positive means greater true-signal recall",
    ),
    (
        "signal_distortion_reduction",
        lambda intervention, comparator: _required(comparator.event_signal_distortion)
        - _required(intervention.event_signal_distortion),
        "positive means less true-signal distortion",
    ),
    (
        "risk_brier_reduction",
        lambda intervention, comparator: comparator.risk_brier_score - intervention.risk_brier_score,
        "positive means better risk calibration",
    ),
    (
        "risk_ece_reduction",
        lambda intervention, comparator: comparator.risk_expected_calibration_error
        - intervention.risk_expected_calibration_error,
        "positive means better risk calibration",
    ),
    (
        "audit_failure_yield_gain",
        lambda intervention, comparator: _required(intervention.audit_failure_yield)
        - _required(comparator.audit_failure_yield),
        "positive means more nuisance-dominant failures per fixed audit budget",
    ),
)


def evaluate_latent_results(
    results: Iterable[LatentBenchmarkResult],
    config: LatentEvaluationConfig = LatentEvaluationConfig(),
) -> tuple[list[PairedEffect], list[FailureMapCell]]:
    """Produce paired effects and a condition-specific failure map.

    Required comparisons are deliberately narrow:

    * N0 robust reference versus B0 raw motion: does cancellation help?
    * N0 robust reference versus time-shifted reference: is the effect reference
      specific rather than an accidental threshold effect?

    N1 and N2 remain available in the generic paired-effect table so a later
    learned nuisance-state model can be evaluated without redesigning the
    analysis contract.
    """

    rows = list(results)
    if not rows:
        raise ValueError("results cannot be empty")
    _assert_held_out(rows)
    effects: list[PairedEffect] = []
    comparisons = (
        (LatentPolicy.ROBUST_REFERENCE.value, LatentPolicy.RAW_MOTION.value),
        (LatentPolicy.QUALITY_AWARE.value, LatentPolicy.RAW_MOTION.value),
        (LatentPolicy.RISK_GUIDED_AUDIT.value, LatentPolicy.RAW_MOTION.value),
        (LatentPolicy.ROBUST_REFERENCE.value, LatentPolicy.TIME_SHIFTED_REFERENCE.value),
        (LatentPolicy.ROBUST_REFERENCE.value, LatentPolicy.SPATIALLY_MISMATCHED_REFERENCE.value),
        (LatentPolicy.ROBUST_REFERENCE.value, LatentPolicy.DEGRADED_REFERENCE.value),
    )
    for condition, condition_rows in _condition_groups(rows).items():
        for intervention, comparator in comparisons:
            effects.extend(
                _paired_effects_for_comparison(
                    condition=condition,
                    rows=condition_rows,
                    intervention=intervention,
                    comparator=comparator,
                    config=config,
                )
            )
    failure_map = _build_failure_map(effects, config)
    return effects, failure_map


def write_latent_evaluation(
    output_dir: str | Path,
    effects: Iterable[PairedEffect],
    failure_map: Iterable[FailureMapCell],
    config: LatentEvaluationConfig,
) -> dict[str, Path]:
    """Write machine-readable paired effects and a paper-ready condition report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effect_rows = [effect.to_dict() for effect in effects]
    map_rows = [cell.to_dict() for cell in failure_map]
    effects_path = output / "latent_paired_effects.csv"
    map_path = output / "latent_failure_map.csv"
    config_path = output / "latent_evaluation_config.json"
    report_path = output / "latent_evaluation_report.md"
    _write_csv(effects_path, effect_rows)
    _write_csv(map_path, map_rows)
    config_path.write_text(_json(config.to_dict()), encoding="utf-8")
    report_path.write_text(_render_report(effect_rows, map_rows, config), encoding="utf-8")
    return {
        "paired_effects": effects_path,
        "failure_map": map_path,
        "config": config_path,
        "report": report_path,
    }


def _condition_groups(results: list[LatentBenchmarkResult]) -> dict[str, list[LatentBenchmarkResult]]:
    groups: dict[str, list[LatentBenchmarkResult]] = defaultdict(list)
    groups["all_held_out_conditions"].extend(results)
    for row in results:
        groups[f"nuisance_scale={row.nuisance_scale:g}"].append(row)
    return groups


def _paired_effects_for_comparison(
    *,
    condition: str,
    rows: list[LatentBenchmarkResult],
    intervention: str,
    comparator: str,
    config: LatentEvaluationConfig,
) -> list[PairedEffect]:
    paired = _pair_rows(rows, intervention=intervention, comparator=comparator)
    output: list[PairedEffect] = []
    for metric, difference, direction in _EFFECTS:
        deltas = [difference(intervention_row, comparator_row) for intervention_row, comparator_row in paired]
        estimate, low, high = _bootstrap_interval(
            deltas,
            resamples=config.bootstrap_resamples,
            confidence_level=config.confidence_level,
            seed=_stable_seed(config.seed, condition, intervention, comparator, metric),
        )
        output.append(
            PairedEffect(
                condition=condition,
                intervention=intervention,
                comparator=comparator,
                metric=metric,
                blocks=len(deltas),
                estimate=estimate,
                ci_low=low,
                ci_high=high,
                direction=direction,
            )
        )
    return output


def _pair_rows(
    rows: list[LatentBenchmarkResult],
    *,
    intervention: str,
    comparator: str,
) -> list[tuple[LatentBenchmarkResult, LatentBenchmarkResult]]:
    keyed: dict[tuple[float, int], dict[str, LatentBenchmarkResult]] = defaultdict(dict)
    for row in rows:
        keyed[(row.nuisance_scale, row.replicate)][row.policy] = row
    pairs: list[tuple[LatentBenchmarkResult, LatentBenchmarkResult]] = []
    missing: list[tuple[float, int]] = []
    for key, policies in sorted(keyed.items()):
        if intervention not in policies or comparator not in policies:
            missing.append(key)
            continue
        pairs.append((policies[intervention], policies[comparator]))
    if missing:
        raise ValueError(f"missing paired policies for {intervention} vs {comparator}: {missing[:3]}")
    if not pairs:
        raise ValueError(f"no paired rows for {intervention} vs {comparator}")
    return pairs


def _bootstrap_interval(
    values: list[float],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    estimate = mean(values)
    rng = Random(seed)
    draws = sorted(
        mean(values[rng.randrange(len(values))] for _ in range(len(values)))
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence_level) / 2.0
    low_index = max(0, min(resamples - 1, int(alpha * (resamples - 1))))
    high_index = max(0, min(resamples - 1, int((1.0 - alpha) * (resamples - 1))))
    return estimate, draws[low_index], draws[high_index]


def _build_failure_map(effects: list[PairedEffect], config: LatentEvaluationConfig) -> list[FailureMapCell]:
    lookup = {
        (effect.condition, effect.intervention, effect.comparator, effect.metric): effect
        for effect in effects
    }
    conditions = sorted(condition for condition in {effect.condition for effect in effects} if condition.startswith("nuisance_scale="))
    cells: list[FailureMapCell] = []
    for condition in conditions:
        raw = _require_effect(
            lookup,
            condition,
            LatentPolicy.ROBUST_REFERENCE.value,
            LatentPolicy.RAW_MOTION.value,
            "false_event_reduction",
        )
        recall = _require_effect(
            lookup,
            condition,
            LatentPolicy.ROBUST_REFERENCE.value,
            LatentPolicy.RAW_MOTION.value,
            "recall_change",
        )
        specificity = _require_effect(
            lookup,
            condition,
            LatentPolicy.ROBUST_REFERENCE.value,
            LatentPolicy.TIME_SHIFTED_REFERENCE.value,
            "false_event_reduction",
        )
        status, reason = _classify_condition(raw, recall, specificity, config)
        cells.append(
            FailureMapCell(
                nuisance_scale=_parse_scale(condition),
                blocks=raw.blocks,
                false_event_reduction_vs_raw=raw.estimate,
                false_event_reduction_vs_raw_ci_low=raw.ci_low,
                false_event_reduction_vs_raw_ci_high=raw.ci_high,
                recall_change_vs_raw=recall.estimate,
                recall_change_vs_raw_ci_low=recall.ci_low,
                recall_change_vs_raw_ci_high=recall.ci_high,
                reference_specificity_reduction=specificity.estimate,
                reference_specificity_ci_low=specificity.ci_low,
                reference_specificity_ci_high=specificity.ci_high,
                status=status,
                reason=reason,
            )
        )
    return cells


def _classify_condition(
    raw: PairedEffect,
    recall: PairedEffect,
    specificity: PairedEffect,
    config: LatentEvaluationConfig,
) -> tuple[str, str]:
    if raw.ci_low <= config.minimum_false_event_reduction:
        return "no_supported_cancellation_advantage", "false-event reduction interval includes the preregistered minimum"
    if recall.ci_low < -config.recall_noninferiority_margin:
        return "recall_cost", "recall loss exceeds the preregistered noninferiority margin"
    if specificity.ci_low <= 0.0:
        return "reference_non_specific", "correct reference is not clearly better than time-shifted reference"
    return "mechanism_supported", "false-event reduction, recall preservation, and reference specificity all meet the locked criteria"


def _require_effect(
    lookup: dict[tuple[str, str, str, str], PairedEffect],
    condition: str,
    intervention: str,
    comparator: str,
    metric: str,
) -> PairedEffect:
    try:
        return lookup[(condition, intervention, comparator, metric)]
    except KeyError as error:
        raise ValueError(f"required effect missing: {condition}, {intervention}, {comparator}, {metric}") from error


def _parse_scale(condition: str) -> float:
    return float(condition.removeprefix("nuisance_scale="))


def _assert_held_out(rows: list[LatentBenchmarkResult]) -> None:
    unexpected = sorted({row.split for row in rows if row.split != "test"})
    if unexpected:
        raise ValueError(f"evaluator accepts held-out test rows only, received: {unexpected}")


def _stable_seed(base: int, *parts: str) -> int:
    total = base
    for part in parts:
        for character in part:
            total = (total * 131 + ord(character)) % 2_147_483_647
    return total


def _required(value: float | None) -> float:
    if value is None:
        raise ValueError("metric is undefined for a paired block")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_report(
    effects: list[dict[str, Any]],
    failure_map: list[dict[str, Any]],
    config: LatentEvaluationConfig,
) -> str:
    lines = [
        "# Latent benchmark paired evaluation",
        "",
        "## Independent unit",
        "",
        "Each paired Monte Carlo recording block (`nuisance scale × replicate`) is one observation. Frames are not treated as independent replicates.",
        "",
        "## Locked interpretation rules",
        "",
        f"- Percentile bootstrap resamples: {config.bootstrap_resamples}",
        f"- Confidence level: {config.confidence_level:.0%}",
        f"- Minimum false-event reduction: {config.minimum_false_event_reduction:.3f}",
        f"- Recall noninferiority margin: {-config.recall_noninferiority_margin:.3f}",
        "",
        "## Condition failure map",
        "",
        "| nuisance scale | blocks | false-event reduction vs raw (95% interval) | recall change vs raw (95% interval) | specificity reduction vs time-shifted (95% interval) | status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in failure_map:
        lines.append(
            "| {nuisance_scale:.2f} | {blocks} | {false_event_reduction_vs_raw:.3f} [{false_event_reduction_vs_raw_ci_low:.3f}, {false_event_reduction_vs_raw_ci_high:.3f}] | {recall_change_vs_raw:.3f} [{recall_change_vs_raw_ci_low:.3f}, {recall_change_vs_raw_ci_high:.3f}] | {reference_specificity_reduction:.3f} [{reference_specificity_ci_low:.3f}, {reference_specificity_ci_high:.3f}] | {status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Reading the map",
            "",
            "`mechanism_supported` requires all three: reduction of false events versus raw motion, no recall loss beyond the locked margin, and an advantage over a time-shifted reference. Other labels are diagnostic outcomes, not failures to be hidden.",
            "",
            "## All paired effects",
            "",
            "| condition | intervention | comparator | metric | blocks | beneficial effect | 95% interval |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            "| {condition} | {intervention} | {comparator} | {metric} | {blocks} | {estimate:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |".format(**row)
        )
    return "\n".join(lines) + "\n"
