"""Paired held-out evaluation for the N1 temporal nuisance-risk model.

The independent unit is a matched rendered recording block,
``nuisance_scale × replicate``.  Frames are correlated within a block and are
never used as independent observations for uncertainty intervals.

The evaluator answers three distinct questions:

1. Does N0 reference-guided cancellation improve on raw pixel differencing?
2. Does N1 temporal risk gating improve on N0 at a protected recall level?
3. Does the learned N1 model add value beyond a transparent rule-risk gate?

All results are effects plus percentile bootstrap intervals, not frame-level
p-values or selected mean plots.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Callable, Iterable

from .temporal_risk_benchmark import TemporalRiskBenchmarkResult, TemporalRiskPolicy


@dataclass(frozen=True, slots=True)
class TemporalRiskEvaluationConfig:
    """Locked effect and interpretation rules for the N1 benchmark."""

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
class TemporalRiskPairedEffect:
    """Positive effect values always favour the intervention."""

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
class TemporalRiskFailureMapCell:
    """One condition-specific, non-p-hacked conclusion about N1."""

    nuisance_scale: float
    blocks: int
    n0_false_event_reduction_vs_raw: float
    n0_false_event_reduction_ci_low: float
    n0_false_event_reduction_ci_high: float
    mlp_false_event_reduction_vs_n0: float
    mlp_false_event_reduction_ci_low: float
    mlp_false_event_reduction_ci_high: float
    mlp_recall_change_vs_n0: float
    mlp_recall_change_ci_low: float
    mlp_recall_change_ci_high: float
    mlp_false_event_reduction_vs_rule: float
    mlp_false_event_reduction_vs_rule_ci_low: float
    mlp_false_event_reduction_vs_rule_ci_high: float
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _MetricSpec:
    name: str
    difference: Callable[[TemporalRiskBenchmarkResult, TemporalRiskBenchmarkResult], float | None]
    direction: str


_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        "false_event_reduction",
        lambda intervention, comparator: _required(comparator.false_event_rate)
        - _required(intervention.false_event_rate),
        "positive means fewer false events",
    ),
    _MetricSpec(
        "recall_change",
        lambda intervention, comparator: _required(intervention.recall) - _required(comparator.recall),
        "positive means greater true-event recall",
    ),
    _MetricSpec(
        "precision_gain",
        lambda intervention, comparator: _required(intervention.precision) - _required(comparator.precision),
        "positive means a greater accepted-event precision",
    ),
    _MetricSpec(
        "abstention_rate_change",
        lambda intervention, comparator: _abstention_rate(intervention) - _abstention_rate(comparator),
        "positive means more candidate abstention",
    ),
    _MetricSpec(
        "audit_priority_yield_gain",
        lambda intervention, comparator: _optional_difference(
            intervention.audit_priority_yield,
            comparator.audit_priority_yield,
        ),
        "positive means more false candidates recovered per fixed top-10% audit budget",
    ),
)


_COMPARISONS: tuple[tuple[str, str], ...] = (
    (TemporalRiskPolicy.ROBUST_REFERENCE.value, TemporalRiskPolicy.RAW_PIXEL_DIFFERENCE.value),
    (TemporalRiskPolicy.RULE_RISK_GATE.value, TemporalRiskPolicy.ROBUST_REFERENCE.value),
    (TemporalRiskPolicy.TEMPORAL_MLP_RISK_GATE.value, TemporalRiskPolicy.ROBUST_REFERENCE.value),
    (TemporalRiskPolicy.TEMPORAL_MLP_RISK_GATE.value, TemporalRiskPolicy.RULE_RISK_GATE.value),
)


def evaluate_temporal_risk_results(
    results: Iterable[TemporalRiskBenchmarkResult],
    config: TemporalRiskEvaluationConfig = TemporalRiskEvaluationConfig(),
) -> tuple[list[TemporalRiskPairedEffect], list[TemporalRiskFailureMapCell]]:
    """Compute paired block-level effects and condition-specific N1 failure map."""

    rows = list(results)
    if not rows:
        raise ValueError("results cannot be empty")
    _assert_held_out(rows)
    effects: list[TemporalRiskPairedEffect] = []
    for condition, condition_rows in _condition_groups(rows).items():
        for intervention, comparator in _COMPARISONS:
            effects.extend(
                _effects_for_comparison(
                    condition=condition,
                    rows=condition_rows,
                    intervention=intervention,
                    comparator=comparator,
                    config=config,
                )
            )
    return effects, _build_failure_map(effects, config)


def write_temporal_risk_evaluation(
    output_dir: str | Path,
    effects: Iterable[TemporalRiskPairedEffect],
    failure_map: Iterable[TemporalRiskFailureMapCell],
    config: TemporalRiskEvaluationConfig,
) -> dict[str, Path]:
    """Write paired effects, a failure map, configuration, and paper-ready table."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effect_rows = [effect.to_dict() for effect in effects]
    map_rows = [cell.to_dict() for cell in failure_map]
    effects_path = output / "temporal_risk_paired_effects.csv"
    map_path = output / "temporal_risk_failure_map.csv"
    config_path = output / "temporal_risk_evaluation_config.json"
    report_path = output / "temporal_risk_evaluation_report.md"
    _write_csv(effects_path, effect_rows)
    _write_csv(map_path, map_rows)
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(effect_rows, map_rows, config), encoding="utf-8")
    return {
        "paired_effects": effects_path,
        "failure_map": map_path,
        "config": config_path,
        "report": report_path,
    }


def _condition_groups(
    rows: list[TemporalRiskBenchmarkResult],
) -> dict[str, list[TemporalRiskBenchmarkResult]]:
    groups: dict[str, list[TemporalRiskBenchmarkResult]] = defaultdict(list)
    groups["all_held_out_conditions"].extend(rows)
    for row in rows:
        groups[f"nuisance_scale={row.nuisance_scale:g}"].append(row)
    return groups


def _effects_for_comparison(
    *,
    condition: str,
    rows: list[TemporalRiskBenchmarkResult],
    intervention: str,
    comparator: str,
    config: TemporalRiskEvaluationConfig,
) -> list[TemporalRiskPairedEffect]:
    pairs = _pair_rows(rows, intervention=intervention, comparator=comparator)
    output: list[TemporalRiskPairedEffect] = []
    for spec in _METRICS:
        deltas = [spec.difference(intervention_row, comparator_row) for intervention_row, comparator_row in pairs]
        if any(value is None for value in deltas):
            # Metrics such as audit yield are undefined for policies that do not
            # emit risk scores. Do not silently replace missing values with zero.
            continue
        values = [float(value) for value in deltas if value is not None]
        estimate, ci_low, ci_high = _bootstrap_interval(
            values,
            resamples=config.bootstrap_resamples,
            confidence_level=config.confidence_level,
            seed=_stable_seed(config.seed, condition, intervention, comparator, spec.name),
        )
        output.append(
            TemporalRiskPairedEffect(
                condition=condition,
                intervention=intervention,
                comparator=comparator,
                metric=spec.name,
                blocks=len(values),
                estimate=estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                direction=spec.direction,
            )
        )
    return output


def _pair_rows(
    rows: list[TemporalRiskBenchmarkResult],
    *,
    intervention: str,
    comparator: str,
) -> list[tuple[TemporalRiskBenchmarkResult, TemporalRiskBenchmarkResult]]:
    keyed: dict[tuple[float, int], dict[str, TemporalRiskBenchmarkResult]] = defaultdict(dict)
    for row in rows:
        keyed[(row.nuisance_scale, row.replicate)][row.policy] = row
    pairs: list[tuple[TemporalRiskBenchmarkResult, TemporalRiskBenchmarkResult]] = []
    missing: list[tuple[float, int]] = []
    for key, by_policy in sorted(keyed.items()):
        if intervention not in by_policy or comparator not in by_policy:
            missing.append(key)
            continue
        pairs.append((by_policy[intervention], by_policy[comparator]))
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
        raise ValueError("cannot bootstrap an empty value list")
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


def _build_failure_map(
    effects: list[TemporalRiskPairedEffect],
    config: TemporalRiskEvaluationConfig,
) -> list[TemporalRiskFailureMapCell]:
    lookup = {
        (effect.condition, effect.intervention, effect.comparator, effect.metric): effect
        for effect in effects
    }
    conditions = sorted(
        condition
        for condition in {effect.condition for effect in effects}
        if condition.startswith("nuisance_scale=")
    )
    cells: list[TemporalRiskFailureMapCell] = []
    for condition in conditions:
        n0 = _require_effect(
            lookup,
            condition,
            TemporalRiskPolicy.ROBUST_REFERENCE.value,
            TemporalRiskPolicy.RAW_PIXEL_DIFFERENCE.value,
            "false_event_reduction",
        )
        mlp_n0 = _require_effect(
            lookup,
            condition,
            TemporalRiskPolicy.TEMPORAL_MLP_RISK_GATE.value,
            TemporalRiskPolicy.ROBUST_REFERENCE.value,
            "false_event_reduction",
        )
        recall = _require_effect(
            lookup,
            condition,
            TemporalRiskPolicy.TEMPORAL_MLP_RISK_GATE.value,
            TemporalRiskPolicy.ROBUST_REFERENCE.value,
            "recall_change",
        )
        mlp_rule = _require_effect(
            lookup,
            condition,
            TemporalRiskPolicy.TEMPORAL_MLP_RISK_GATE.value,
            TemporalRiskPolicy.RULE_RISK_GATE.value,
            "false_event_reduction",
        )
        status, reason = _classify(n0, mlp_n0, recall, mlp_rule, config)
        cells.append(
            TemporalRiskFailureMapCell(
                nuisance_scale=_parse_scale(condition),
                blocks=mlp_n0.blocks,
                n0_false_event_reduction_vs_raw=n0.estimate,
                n0_false_event_reduction_ci_low=n0.ci_low,
                n0_false_event_reduction_ci_high=n0.ci_high,
                mlp_false_event_reduction_vs_n0=mlp_n0.estimate,
                mlp_false_event_reduction_ci_low=mlp_n0.ci_low,
                mlp_false_event_reduction_ci_high=mlp_n0.ci_high,
                mlp_recall_change_vs_n0=recall.estimate,
                mlp_recall_change_ci_low=recall.ci_low,
                mlp_recall_change_ci_high=recall.ci_high,
                mlp_false_event_reduction_vs_rule=mlp_rule.estimate,
                mlp_false_event_reduction_vs_rule_ci_low=mlp_rule.ci_low,
                mlp_false_event_reduction_vs_rule_ci_high=mlp_rule.ci_high,
                status=status,
                reason=reason,
            )
        )
    return cells


def _classify(
    n0: TemporalRiskPairedEffect,
    mlp_n0: TemporalRiskPairedEffect,
    recall: TemporalRiskPairedEffect,
    mlp_rule: TemporalRiskPairedEffect,
    config: TemporalRiskEvaluationConfig,
) -> tuple[str, str]:
    if n0.ci_low <= config.minimum_false_event_reduction:
        return "n0_not_supported", "N0 does not show the locked false-event reduction versus raw motion"
    if recall.ci_low < -config.recall_noninferiority_margin:
        return "recall_cost", "MLP recall loss exceeds the locked noninferiority margin"
    if mlp_n0.ci_low <= config.minimum_false_event_reduction:
        return "no_mlp_increment_over_n0", "MLP does not clearly reduce false events beyond N0"
    if mlp_rule.ci_low <= config.minimum_false_event_reduction:
        return "no_mlp_increment_over_rule", "MLP does not clearly reduce false events beyond the transparent rule gate"
    return "mlp_increment_supported", "N0 baseline, recall preservation, and learned increment all meet locked criteria"


def _required(value: float | None) -> float:
    if value is None:
        raise ValueError("paired metric is undefined")
    return value


def _optional_difference(intervention: float | None, comparator: float | None) -> float | None:
    if intervention is None or comparator is None:
        return None
    return intervention - comparator


def _abstention_rate(row: TemporalRiskBenchmarkResult) -> float:
    if row.candidate_windows <= 0:
        raise ValueError("candidate_windows must be positive for abstention rate")
    return row.abstained_candidate_windows / row.candidate_windows


def _require_effect(
    lookup: dict[tuple[str, str, str, str], TemporalRiskPairedEffect],
    condition: str,
    intervention: str,
    comparator: str,
    metric: str,
) -> TemporalRiskPairedEffect:
    try:
        return lookup[(condition, intervention, comparator, metric)]
    except KeyError as error:
        raise ValueError(f"required effect missing: {condition}, {intervention}, {comparator}, {metric}") from error


def _parse_scale(condition: str) -> float:
    return float(condition.removeprefix("nuisance_scale="))


def _assert_held_out(rows: list[TemporalRiskBenchmarkResult]) -> None:
    unexpected = sorted({row.split for row in rows if row.split != "test"})
    if unexpected:
        raise ValueError(f"evaluator accepts held-out test rows only, received: {unexpected}")


def _stable_seed(base: int, *parts: str) -> int:
    value = base
    for part in parts:
        for character in part:
            value = (value * 131 + ord(character)) % 2_147_483_647
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


def _render_report(
    effects: list[dict[str, Any]],
    failure_map: list[dict[str, Any]],
    config: TemporalRiskEvaluationConfig,
) -> str:
    lines = [
        "# N1 temporal nuisance-risk paired evaluation",
        "",
        "## Independent unit",
        "",
        "One matched rendered recording block (`nuisance scale × replicate`) is one observation. Frames are not independent replicates and are never bootstrapped as if they were.",
        "",
        "## Locked interpretation rules",
        "",
        f"- Percentile bootstrap resamples: {config.bootstrap_resamples}",
        f"- Confidence level: {config.confidence_level:.0%}",
        f"- Minimum false-event reduction: {config.minimum_false_event_reduction:.3f}",
        f"- Recall noninferiority margin: {-config.recall_noninferiority_margin:.3f}",
        "",
        "## N1 failure map",
        "",
        "| nuisance scale | blocks | N0 false-event reduction vs raw (95% interval) | MLP false-event reduction vs N0 (95% interval) | MLP recall change vs N0 (95% interval) | MLP false-event reduction vs rule (95% interval) | status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in failure_map:
        lines.append(
            "| {nuisance_scale:.2f} | {blocks} | {n0_false_event_reduction_vs_raw:.3f} [{n0_false_event_reduction_ci_low:.3f}, {n0_false_event_reduction_ci_high:.3f}] | {mlp_false_event_reduction_vs_n0:.3f} [{mlp_false_event_reduction_ci_low:.3f}, {mlp_false_event_reduction_ci_high:.3f}] | {mlp_recall_change_vs_n0:.3f} [{mlp_recall_change_ci_low:.3f}, {mlp_recall_change_ci_high:.3f}] | {mlp_false_event_reduction_vs_rule:.3f} [{mlp_false_event_reduction_vs_rule_ci_low:.3f}, {mlp_false_event_reduction_vs_rule_ci_high:.3f}] | {status} |".format(**row)
        )
    lines.extend(
        [
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
    lines.extend(
        [
            "",
            "## Reading the map",
            "",
            "`mlp_increment_supported` requires: N0 must first beat raw pixel differencing; MLP must reduce false events beyond N0; MLP must not lose recall beyond the locked margin; and MLP must add false-event reduction beyond the transparent rule gate. Other statuses are findings to report, not cases to tune away.",
        ]
    )
    return "\n".join(lines) + "\n"
