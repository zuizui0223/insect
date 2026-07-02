"""Held-out benchmark for learned temporal nuisance-risk gating.

The learned model is deliberately downstream of reference-guided cancellation:

```text
rendered pixels
  -> N0 robust shared-scene residual candidate
  -> temporal scene-context model predicts false-candidate risk
  -> accept, abstain, or prioritise audit
```

The model never sees focal local evidence, target identity, hidden disturbance
variables, or future frames. It sees only a causal history of image-derived
reference-region features ending at the candidate frame.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np

from .temporal_risk import (
    TemporalRiskModel,
    TemporalRiskModelConfig,
    TemporalRiskTrainingSummary,
    fit_temporal_risk_model,
    temporal_reference_matrix,
)
from .visual_benchmark import (
    VisualBenchmarkConfig,
    VisualFeatureFrame,
    VisualPolicy,
    extract_visual_features,
    threshold_at_target_recall,
)
from .visual_world import VisualDisturbanceWorld


class TemporalRiskPolicy(str, Enum):
    RAW_PIXEL_DIFFERENCE = "B0_raw_pixel_difference"
    ROBUST_REFERENCE = "N0_robust_visual_reference"
    RULE_RISK_GATE = "N1_rule_risk_gate"
    TEMPORAL_MLP_RISK_GATE = "N1_temporal_mlp_risk_gate"


@dataclass(frozen=True, slots=True)
class TemporalRiskBenchmarkConfig:
    """Predeclared training, calibration, and held-out evaluation settings."""

    visual: VisualBenchmarkConfig = VisualBenchmarkConfig()
    candidate_recall: float = 0.97
    model: TemporalRiskModelConfig = TemporalRiskModelConfig()

    def __post_init__(self) -> None:
        if not self.visual.target_recall < self.candidate_recall < 1.0:
            raise ValueError("candidate_recall must be greater than target_recall and below 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual": self.visual.to_dict(),
            "candidate_recall": self.candidate_recall,
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TemporalRiskCalibration:
    """All decision thresholds are fitted from calibration worlds only."""

    raw_event_threshold: float
    robust_event_threshold: float
    candidate_event_threshold: float
    rule_risk_threshold: float
    model_risk_threshold: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalRiskBenchmarkResult:
    split: str
    nuisance_scale: float
    replicate: int
    policy: str
    event_threshold: float
    risk_threshold: float | None
    true_event_windows: int
    non_event_windows: int
    accepted_true_events: int
    accepted_false_events: int
    candidate_windows: int
    abstained_candidate_windows: int
    recall: float | None
    false_event_rate: float | None
    precision: float | None
    audit_priority_yield: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalRiskRun:
    """Result bundle including the learned model for reproducible inspection."""

    results: tuple[TemporalRiskBenchmarkResult, ...]
    calibration: TemporalRiskCalibration
    model: TemporalRiskModel
    training_summary: TemporalRiskTrainingSummary


@dataclass(frozen=True, slots=True)
class _VisualBlock:
    nuisance_scale: float
    replicate: int
    features: tuple[VisualFeatureFrame, ...]


def run_temporal_risk_benchmark(
    config: TemporalRiskBenchmarkConfig = TemporalRiskBenchmarkConfig(),
) -> TemporalRiskRun:
    """Fit a temporal MLP on calibration worlds and evaluate held-out worlds.

    The learning task is deliberately narrow: among a high-recall N0 residual
    candidate stream, predict whether a candidate is a false event using only
    temporal scene-reference history. The MLP's risk-gating threshold is chosen
    to preserve the predeclared total true-event recall on calibration data.
    """

    calibration_blocks = list(_visual_blocks(config.visual, split="calibration"))
    test_blocks = list(_visual_blocks(config.visual, split="test"))
    calibration_features = [frame for block in calibration_blocks for frame in block.features]
    raw_event_threshold = threshold_at_target_recall(
        [frame.raw_local_evidence for frame in calibration_features if frame.true_local_event],
        config.visual.target_recall,
    )
    robust_event_threshold = threshold_at_target_recall(
        [_robust_score(frame) for frame in calibration_features if frame.true_local_event],
        config.visual.target_recall,
    )
    candidate_event_threshold = threshold_at_target_recall(
        [_robust_score(frame) for frame in calibration_features if frame.true_local_event],
        config.candidate_recall,
    )

    train_x, train_y = _candidate_training_data(calibration_blocks, candidate_event_threshold, config.model)
    model, training_summary = fit_temporal_risk_model(train_x, train_y, config.model)
    rule_risk_threshold = _risk_threshold_for_target_recall(
        calibration_blocks,
        candidate_event_threshold,
        config.visual.target_recall,
        risk_provider=lambda features, indices: np.asarray([features[index].risk_proxy for index in indices]),
    )
    model_risk_threshold = _risk_threshold_for_target_recall(
        calibration_blocks,
        candidate_event_threshold,
        config.visual.target_recall,
        risk_provider=lambda features, indices: model.predict_proba(
            temporal_reference_matrix(features, indices, window_frames=config.model.window_frames)
        ),
    )
    calibration = TemporalRiskCalibration(
        raw_event_threshold=raw_event_threshold,
        robust_event_threshold=robust_event_threshold,
        candidate_event_threshold=candidate_event_threshold,
        rule_risk_threshold=rule_risk_threshold,
        model_risk_threshold=model_risk_threshold,
    )

    results: list[TemporalRiskBenchmarkResult] = []
    for block in test_blocks:
        results.append(
            _evaluate_policy(
                block,
                TemporalRiskPolicy.RAW_PIXEL_DIFFERENCE,
                event_threshold=raw_event_threshold,
                risk_threshold=None,
                accept_mask=_raw_acceptance(block.features, raw_event_threshold),
            )
        )
        results.append(
            _evaluate_policy(
                block,
                TemporalRiskPolicy.ROBUST_REFERENCE,
                event_threshold=robust_event_threshold,
                risk_threshold=None,
                accept_mask=_robust_acceptance(block.features, robust_event_threshold),
            )
        )
        candidate_mask = _robust_acceptance(block.features, candidate_event_threshold)
        candidate_indices = np.flatnonzero(candidate_mask).tolist()
        rule_risks = np.asarray([block.features[index].risk_proxy for index in candidate_indices])
        rule_acceptance = _gate_candidates(candidate_mask, candidate_indices, rule_risks, rule_risk_threshold)
        results.append(
            _evaluate_policy(
                block,
                TemporalRiskPolicy.RULE_RISK_GATE,
                event_threshold=candidate_event_threshold,
                risk_threshold=rule_risk_threshold,
                accept_mask=rule_acceptance,
                risk_values=rule_risks,
            )
        )
        model_risks = model.predict_proba(
            temporal_reference_matrix(
                block.features,
                candidate_indices,
                window_frames=config.model.window_frames,
            )
        )
        model_acceptance = _gate_candidates(candidate_mask, candidate_indices, model_risks, model_risk_threshold)
        results.append(
            _evaluate_policy(
                block,
                TemporalRiskPolicy.TEMPORAL_MLP_RISK_GATE,
                event_threshold=candidate_event_threshold,
                risk_threshold=model_risk_threshold,
                accept_mask=model_acceptance,
                risk_values=model_risks,
            )
        )
    return TemporalRiskRun(
        results=tuple(results),
        calibration=calibration,
        model=model,
        training_summary=training_summary,
    )


def write_temporal_risk_benchmark(
    output_dir: str | Path,
    run: TemporalRiskRun,
    config: TemporalRiskBenchmarkConfig,
) -> dict[str, Path]:
    """Write results, learned weights, thresholds, and a claim-bounded report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in run.results]
    metrics_path = output / "temporal_risk_metrics.csv"
    summary_path = output / "temporal_risk_summary.csv"
    config_path = output / "temporal_risk_config.json"
    calibration_path = output / "temporal_risk_calibration.json"
    training_path = output / "temporal_risk_training_summary.json"
    model_path = output / "temporal_risk_model.npz"
    report_path = output / "temporal_risk_report.md"
    _write_csv(metrics_path, rows)
    summary = _summarise(rows)
    _write_csv(summary_path, summary)
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(run.calibration.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    training_path.write_text(json.dumps(run.training_summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    run.model.save(model_path)
    report_path.write_text(_render_report(config, run, summary), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "config": config_path,
        "calibration": calibration_path,
        "training": training_path,
        "model": model_path,
        "report": report_path,
    }


def _visual_blocks(
    config: VisualBenchmarkConfig,
    *,
    split: str,
) -> Iterable[_VisualBlock]:
    count = config.calibration_replicates if split == "calibration" else config.test_replicates
    split_offset = 0 if split == "calibration" else 20_000_000
    for scale_index, scale in enumerate(config.nuisance_scales):
        for replicate in range(count):
            base = config.base_world
            world_config = replace(
                base,
                name=f"{split}|temporal_scale={scale:g}|replicate={replicate}",
                frames=config.frames,
                camera_motion_sd=base.camera_motion_sd * scale,
                sway_amplitude=base.sway_amplitude * scale,
                shadow_amplitude=base.shadow_amplitude * scale,
                illumination_sd=base.illumination_sd * scale,
                seed=config.seed + split_offset + scale_index * 100_000 + replicate,
            )
            world = VisualDisturbanceWorld(world_config)
            features = tuple(
                extract_visual_features(
                    world,
                    reference_delay_frames=config.reference_delay_frames,
                    alignment_search_radius=config.alignment_search_radius,
                )
            )
            yield _VisualBlock(nuisance_scale=scale, replicate=replicate, features=features)


def _candidate_training_data(
    blocks: list[_VisualBlock],
    candidate_event_threshold: float,
    model_config: TemporalRiskModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for block in blocks:
        candidate_indices = [
            index
            for index, frame in enumerate(block.features)
            if _robust_score(frame) >= candidate_event_threshold
        ]
        if not candidate_indices:
            continue
        matrices.append(
            temporal_reference_matrix(
                block.features,
                candidate_indices,
                window_frames=model_config.window_frames,
            )
        )
        labels.append(
            np.asarray(
                [float(not block.features[index].true_local_event) for index in candidate_indices],
                dtype=np.float64,
            )
        )
    if not matrices:
        raise ValueError("calibration worlds produced no high-recall residual candidates")
    return np.concatenate(matrices, axis=0), np.concatenate(labels, axis=0)


def _risk_threshold_for_target_recall(
    blocks: list[_VisualBlock],
    candidate_event_threshold: float,
    target_recall: float,
    *,
    risk_provider: Callable[[tuple[VisualFeatureFrame, ...], list[int]], np.ndarray],
) -> float:
    total_true_events = sum(
        int(frame.true_local_event)
        for block in blocks
        for frame in block.features
    )
    true_candidate_risks: list[float] = []
    for block in blocks:
        indices = [
            index
            for index, frame in enumerate(block.features)
            if _robust_score(frame) >= candidate_event_threshold
        ]
        risks = risk_provider(block.features, indices)
        if risks.size != len(indices):
            raise ValueError("risk provider must emit one risk per candidate")
        true_candidate_risks.extend(
            float(risk)
            for index, risk in zip(indices, risks)
            if block.features[index].true_local_event
        )
    required = ceil(target_recall * total_true_events)
    if len(true_candidate_risks) < required:
        raise ValueError("candidate threshold cannot reach requested total true-event recall")
    ordered = sorted(true_candidate_risks)
    return ordered[required - 1]


def _raw_acceptance(features: tuple[VisualFeatureFrame, ...], threshold: float) -> np.ndarray:
    return np.asarray([frame.raw_local_evidence >= threshold for frame in features], dtype=bool)


def _robust_acceptance(features: tuple[VisualFeatureFrame, ...], threshold: float) -> np.ndarray:
    return np.asarray([_robust_score(frame) >= threshold for frame in features], dtype=bool)


def _gate_candidates(
    candidate_mask: np.ndarray,
    candidate_indices: list[int],
    risks: np.ndarray,
    risk_threshold: float,
) -> np.ndarray:
    if risks.size != len(candidate_indices):
        raise ValueError("risks must match candidate indices")
    accepted = np.zeros(candidate_mask.size, dtype=bool)
    for index, risk in zip(candidate_indices, risks):
        if risk <= risk_threshold:
            accepted[index] = True
    return accepted


def _evaluate_policy(
    block: _VisualBlock,
    policy: TemporalRiskPolicy,
    *,
    event_threshold: float,
    risk_threshold: float | None,
    accept_mask: np.ndarray,
    risk_values: np.ndarray | None = None,
) -> TemporalRiskBenchmarkResult:
    features = block.features
    if accept_mask.size != len(features):
        raise ValueError("acceptance mask must match feature frames")
    true_event_windows = sum(frame.true_local_event for frame in features)
    non_event_windows = len(features) - true_event_windows
    accepted_true = sum(
        bool(accepted) and frame.true_local_event
        for frame, accepted in zip(features, accept_mask)
    )
    accepted_false = sum(
        bool(accepted) and not frame.true_local_event
        for frame, accepted in zip(features, accept_mask)
    )
    candidates = _robust_acceptance(features, event_threshold)
    candidate_count = int(np.sum(candidates))
    abstained = candidate_count - int(np.sum(accept_mask))
    precision_denominator = accepted_true + accepted_false
    audit_yield = None
    if risk_values is not None and risk_values.size:
        candidate_indices = np.flatnonzero(candidates).tolist()
        high_risk_count = max(1, ceil(0.10 * len(candidate_indices)))
        ranked = sorted(
            zip(candidate_indices, risk_values),
            key=lambda item: item[1],
            reverse=True,
        )[:high_risk_count]
        audit_yield = mean(float(not features[index].true_local_event) for index, _ in ranked)
    return TemporalRiskBenchmarkResult(
        split="test",
        nuisance_scale=block.nuisance_scale,
        replicate=block.replicate,
        policy=policy.value,
        event_threshold=event_threshold,
        risk_threshold=risk_threshold,
        true_event_windows=true_event_windows,
        non_event_windows=non_event_windows,
        accepted_true_events=accepted_true,
        accepted_false_events=accepted_false,
        candidate_windows=candidate_count,
        abstained_candidate_windows=abstained,
        recall=None if true_event_windows == 0 else accepted_true / true_event_windows,
        false_event_rate=None if non_event_windows == 0 else accepted_false / non_event_windows,
        precision=None if precision_denominator == 0 else accepted_true / precision_denominator,
        audit_priority_yield=audit_yield,
    )


def _robust_score(frame: VisualFeatureFrame) -> float:
    return frame.stabilised_local_evidence - frame.robust_reference


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), float(row["nuisance_scale"]))].append(row)
    numeric = (
        "event_threshold",
        "risk_threshold",
        "true_event_windows",
        "non_event_windows",
        "accepted_true_events",
        "accepted_false_events",
        "candidate_windows",
        "abstained_candidate_windows",
        "recall",
        "false_event_rate",
        "precision",
        "audit_priority_yield",
    )
    output: list[dict[str, Any]] = []
    for (policy, scale), values in sorted(grouped.items()):
        row: dict[str, Any] = {"policy": policy, "nuisance_scale": scale, "replicates": len(values)}
        for field in numeric:
            finite = [float(value[field]) for value in values if value[field] is not None]
            row[field] = None if not finite else mean(finite)
        output.append(row)
    return output


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
    config: TemporalRiskBenchmarkConfig,
    run: TemporalRiskRun,
    summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Learned temporal nuisance-risk benchmark",
        "",
        "## Scope",
        "",
        "This benchmark evaluates a small temporal neural model that estimates whether a high-recall N0 residual candidate is likely false from image-derived background-reference history. The model does not consume focal local evidence, object identity, bounding boxes, hidden renderer state, or future frames.",
        "",
        "## Locked calibration",
        "",
        f"- Target total true-event recall: {config.visual.target_recall:.2f}",
        f"- High-recall candidate stream: {config.candidate_recall:.2f}",
        f"- Temporal history: {config.model.window_frames} frame pairs",
        f"- MLP hidden units: {config.model.hidden_units}",
        f"- Training samples: {run.training_summary.samples}",
        f"- False-candidate labels in calibration: {run.training_summary.positive_samples}",
        "",
        "```text",
        f"raw event threshold: {run.calibration.raw_event_threshold:.6f}",
        f"N0 robust threshold: {run.calibration.robust_event_threshold:.6f}",
        f"N1 candidate threshold: {run.calibration.candidate_event_threshold:.6f}",
        f"rule-risk gate: {run.calibration.rule_risk_threshold:.6f}",
        f"MLP-risk gate: {run.calibration.model_risk_threshold:.6f}",
        "```",
        "",
        "## Held-out summary",
        "",
        "| policy | nuisance scale | recall | false-event rate | precision | abstained candidates | audit-priority false yield |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {policy} | {nuisance_scale:.2f} | {recall} | {false_event_rate} | {precision} | {abstained_candidate_windows} | {audit_priority_yield} |".format(
                **{
                    **row,
                    "recall": _fmt(row["recall"]),
                    "false_event_rate": _fmt(row["false_event_rate"]),
                    "precision": _fmt(row["precision"]),
                    "abstained_candidate_windows": _fmt(row["abstained_candidate_windows"]),
                    "audit_priority_yield": _fmt(row["audit_priority_yield"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "The learned MLP is an N1 observability-risk layer, not an organism detector. A useful result requires it to reduce false-event rate versus N0 robust reference and versus a transparent rule-risk gate at matched calibrated recall. Any recall cost, condition-specific failure, or lack of improvement must be reported rather than tuned away.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
