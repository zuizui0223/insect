"""Image-derived visual benchmark for reference-guided nuisance cancellation.

All policy inputs in this module are estimated from rendered pixels. Hidden
renderer variables are used only after inference to score whether a detected
local pulse was genuine. This is the bridge between the scalar causal benchmark
and later real camera video.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from random import Random
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from .visual_world import RenderedFrame, VisualDisturbanceWorld, VisualWorldConfig


class VisualPolicy(str, Enum):
    RAW_PIXEL_DIFFERENCE = "B0_raw_pixel_difference"
    GLOBAL_STABILISED = "B1_global_stabilised"
    SINGLE_REFERENCE = "B2_single_visual_reference"
    ROBUST_REFERENCE = "N0_robust_visual_reference"
    TIME_SHIFTED_REFERENCE = "NC_time_shifted_visual_reference"
    SPATIALLY_MISMATCHED_REFERENCE = "NC_spatially_mismatched_visual_reference"


@dataclass(frozen=True, slots=True)
class VisualFeatureFrame:
    """Pixel-derived features for one frame pair; truth retained for scoring."""

    frame_index: int
    true_local_event: bool
    raw_local_evidence: float
    stabilised_local_evidence: float
    single_reference: float
    robust_reference: float
    delayed_reference: float
    mismatched_reference: float
    global_shift_y: int
    global_shift_x: int
    global_shift_error: float
    reference_coherence: float

    @property
    def risk_proxy(self) -> float:
        """Transparent image-derived high-risk signal for future audit policies."""

        return abs(self.robust_reference) + self.reference_coherence + 0.04 * (
            abs(self.global_shift_y) + abs(self.global_shift_x)
        )


@dataclass(frozen=True, slots=True)
class VisualBenchmarkConfig:
    """Held-out visual-rendering benchmark settings."""

    frames: int = 240
    calibration_replicates: int = 12
    test_replicates: int = 18
    nuisance_scales: tuple[float, ...] = (0.65, 1.0, 1.35)
    target_recall: float = 0.85
    reference_delay_frames: int = 7
    alignment_search_radius: int = 4
    seed: int = 20260702
    base_world: VisualWorldConfig = VisualWorldConfig()

    def __post_init__(self) -> None:
        if self.frames < 20:
            raise ValueError("frames must be at least 20")
        if self.calibration_replicates < 1 or self.test_replicates < 1:
            raise ValueError("replicate counts must be positive")
        if not self.nuisance_scales or any(value <= 0.0 for value in self.nuisance_scales):
            raise ValueError("nuisance_scales must be positive")
        if not 0.0 < self.target_recall < 1.0:
            raise ValueError("target_recall must lie in (0, 1)")
        if self.reference_delay_frames < 1:
            raise ValueError("reference_delay_frames must be positive")
        if self.alignment_search_radius < 0:
            raise ValueError("alignment_search_radius must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["base_world"] = self.base_world.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class VisualBenchmarkResult:
    split: str
    nuisance_scale: float
    replicate: int
    policy: str
    threshold: float
    true_event_windows: int
    non_event_windows: int
    true_positive_windows: int
    false_positive_windows: int
    recall: float | None
    false_event_rate: float | None
    mean_global_shift_error: float
    mean_reference_coherence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_POLICY_SCORE = {
    VisualPolicy.RAW_PIXEL_DIFFERENCE: lambda frame: frame.raw_local_evidence,
    VisualPolicy.GLOBAL_STABILISED: lambda frame: frame.stabilised_local_evidence,
    VisualPolicy.SINGLE_REFERENCE: lambda frame: frame.stabilised_local_evidence - frame.single_reference,
    VisualPolicy.ROBUST_REFERENCE: lambda frame: frame.stabilised_local_evidence - frame.robust_reference,
    VisualPolicy.TIME_SHIFTED_REFERENCE: lambda frame: frame.stabilised_local_evidence - frame.delayed_reference,
    VisualPolicy.SPATIALLY_MISMATCHED_REFERENCE: lambda frame: frame.stabilised_local_evidence - frame.mismatched_reference,
}


def extract_visual_features(
    world: VisualDisturbanceWorld,
    *,
    reference_delay_frames: int = 7,
    alignment_search_radius: int = 4,
) -> list[VisualFeatureFrame]:
    """Extract all nuisance references exclusively from adjacent rendered frames.

    The function intentionally accepts only ``VisualDisturbanceWorld`` images and
    ROIs. It never reads ``latent_illumination`` or ``latent_sway`` while
    estimating a reference. Those hidden fields are not part of the output.
    """

    rendered = world.frames_data
    features: list[VisualFeatureFrame] = []
    reference_history: list[float] = []
    for previous, current in zip(rendered[:-1], rendered[1:]):
        shift_y, shift_x = estimate_global_shift(
            previous.image,
            current.image,
            world.reference_regions,
            search_radius=alignment_search_radius,
        )
        stabilised_current = shift_image(current.image, -shift_y, -shift_x)
        raw_difference = current.image - previous.image
        stabilised_difference = stabilised_current - previous.image
        local_y, local_x = world.local_region
        raw_local = float(np.mean(raw_difference[local_y, local_x]))
        stabilised_local = float(np.mean(stabilised_difference[local_y, local_x]))
        reference_values = [
            float(np.mean(stabilised_difference[region_y, region_x]))
            for region_y, region_x in world.reference_regions
        ]
        robust_reference = float(median(reference_values))
        reference_history.append(robust_reference)
        delayed_reference = (
            reference_history[-reference_delay_frames - 1]
            if len(reference_history) > reference_delay_frames
            else 0.0
        )
        mismatch_y, mismatch_x = world.mismatched_region
        mismatch_reference = float(np.mean(stabilised_difference[mismatch_y, mismatch_x]))
        expected_y = current.camera_shift[0] - previous.camera_shift[0]
        expected_x = current.camera_shift[1] - previous.camera_shift[1]
        features.append(
            VisualFeatureFrame(
                frame_index=current.frame_index,
                true_local_event=current.true_local_event,
                raw_local_evidence=raw_local,
                stabilised_local_evidence=stabilised_local,
                single_reference=reference_values[0],
                robust_reference=robust_reference,
                delayed_reference=float(delayed_reference),
                mismatched_reference=mismatch_reference,
                global_shift_y=shift_y,
                global_shift_x=shift_x,
                global_shift_error=float(abs(shift_y - expected_y) + abs(shift_x - expected_x)),
                reference_coherence=float(np.mean(np.abs(np.asarray(reference_values) - robust_reference))),
            )
        )
    return features


def estimate_global_shift(
    previous: np.ndarray,
    current: np.ndarray,
    reference_regions: tuple[tuple[slice, slice], ...],
    *,
    search_radius: int,
) -> tuple[int, int]:
    """Estimate integer camera displacement by matching only background regions.

    This is a deliberately lightweight reference estimator rather than a hidden
    optical-flow oracle. Later deployments can replace it with optical flow,
    IMU, or a learned temporal encoder while preserving the benchmark contract.
    """

    if previous.shape != current.shape:
        raise ValueError("previous and current frames must have identical shapes")
    mask = np.zeros(previous.shape, dtype=bool)
    for region_y, region_x in reference_regions:
        mask[region_y, region_x] = True
    if not np.any(mask):
        raise ValueError("reference regions cannot be empty")
    best_error = float("inf")
    best_shift = (0, 0)
    for shift_y in range(-search_radius, search_radius + 1):
        for shift_x in range(-search_radius, search_radius + 1):
            aligned = shift_image(current, -shift_y, -shift_x)
            error = float(np.mean((aligned[mask] - previous[mask]) ** 2))
            if error < best_error:
                best_error = error
                best_shift = (shift_y, shift_x)
    return best_shift


def shift_image(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    return np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)


def run_visual_benchmark(
    config: VisualBenchmarkConfig = VisualBenchmarkConfig(),
) -> tuple[list[VisualBenchmarkResult], dict[str, float]]:
    """Run held-out rendered-video comparisons with calibration-only thresholds."""

    calibration_features = [
        feature
        for _, _, world in _worlds(config, split="calibration")
        for feature in extract_visual_features(
            world,
            reference_delay_frames=config.reference_delay_frames,
            alignment_search_radius=config.alignment_search_radius,
        )
    ]
    thresholds = {
        policy.value: threshold_at_target_recall(
            [_POLICY_SCORE[policy](feature) for feature in calibration_features if feature.true_local_event],
            config.target_recall,
        )
        for policy in VisualPolicy
    }
    results: list[VisualBenchmarkResult] = []
    for scale, replicate, world in _worlds(config, split="test"):
        features = extract_visual_features(
            world,
            reference_delay_frames=config.reference_delay_frames,
            alignment_search_radius=config.alignment_search_radius,
        )
        for policy in VisualPolicy:
            results.append(
                evaluate_visual_features(
                    features,
                    policy=policy,
                    threshold=thresholds[policy.value],
                    nuisance_scale=scale,
                    replicate=replicate,
                )
            )
    return results, thresholds


def evaluate_visual_features(
    features: list[VisualFeatureFrame],
    *,
    policy: VisualPolicy,
    threshold: float,
    nuisance_scale: float,
    replicate: int,
) -> VisualBenchmarkResult:
    """Score one policy on a single held-out rendered recording block."""

    scores = [_POLICY_SCORE[policy](feature) for feature in features]
    predictions = [score >= threshold for score in scores]
    true_events = [feature for feature in features if feature.true_local_event]
    non_events = [feature for feature in features if not feature.true_local_event]
    true_positive = sum(
        predicted and feature.true_local_event
        for feature, predicted in zip(features, predictions)
    )
    false_positive = sum(
        predicted and not feature.true_local_event
        for feature, predicted in zip(features, predictions)
    )
    return VisualBenchmarkResult(
        split="test",
        nuisance_scale=nuisance_scale,
        replicate=replicate,
        policy=policy.value,
        threshold=threshold,
        true_event_windows=len(true_events),
        non_event_windows=len(non_events),
        true_positive_windows=true_positive,
        false_positive_windows=false_positive,
        recall=None if not true_events else true_positive / len(true_events),
        false_event_rate=None if not non_events else false_positive / len(non_events),
        mean_global_shift_error=mean(feature.global_shift_error for feature in features),
        mean_reference_coherence=mean(feature.reference_coherence for feature in features),
    )


def threshold_at_target_recall(positive_scores: list[float], target_recall: float) -> float:
    """Calibrate a one-sided event threshold using calibration worlds only."""

    if not positive_scores:
        raise ValueError("calibration worlds contain no true local events")
    ordered = sorted(positive_scores)
    index = min(
        len(ordered) - 1,
        max(0, int((1.0 - target_recall) * len(ordered))),
    )
    return ordered[index]


def write_visual_benchmark(
    output_dir: str | Path,
    results: Iterable[VisualBenchmarkResult],
    config: VisualBenchmarkConfig,
    thresholds: dict[str, float],
) -> dict[str, Path]:
    """Write all held-out visual benchmark rows and a compact report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    metrics_path = output / "visual_disturbance_metrics.csv"
    summary_path = output / "visual_disturbance_summary.csv"
    config_path = output / "visual_disturbance_config.json"
    thresholds_path = output / "visual_calibration_thresholds.json"
    report_path = output / "visual_disturbance_report.md"
    _write_csv(metrics_path, rows)
    summary_rows = _summarise(rows)
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


def _worlds(
    config: VisualBenchmarkConfig,
    *,
    split: str,
) -> Iterable[tuple[float, int, VisualDisturbanceWorld]]:
    count = config.calibration_replicates if split == "calibration" else config.test_replicates
    split_offset = 0 if split == "calibration" else 20_000_000
    for scale_index, scale in enumerate(config.nuisance_scales):
        for replicate in range(count):
            base = config.base_world
            world_config = replace(
                base,
                name=f"{split}|visual_scale={scale:g}|replicate={replicate}",
                frames=config.frames,
                camera_motion_sd=base.camera_motion_sd * scale,
                sway_amplitude=base.sway_amplitude * scale,
                shadow_amplitude=base.shadow_amplitude * scale,
                illumination_sd=base.illumination_sd * scale,
                seed=config.seed + split_offset + scale_index * 100_000 + replicate,
            )
            yield scale, replicate, VisualDisturbanceWorld(world_config)


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouping = ("policy", "nuisance_scale")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in grouping)].append(row)
    numeric = [
        "threshold",
        "true_event_windows",
        "non_event_windows",
        "true_positive_windows",
        "false_positive_windows",
        "recall",
        "false_event_rate",
        "mean_global_shift_error",
        "mean_reference_coherence",
    ]
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        summary = dict(zip(grouping, key))
        summary["replicates"] = len(values)
        for field in numeric:
            numeric_values = [float(row[field]) for row in values if row[field] is not None]
            summary[field] = None if not numeric_values else mean(numeric_values)
        output.append(summary)
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
    config: VisualBenchmarkConfig,
    thresholds: dict[str, float],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Rendered visual-disturbance benchmark",
        "",
        "## Scope",
        "",
        "All nuisance references are extracted from rendered pixels. Hidden camera, illumination, and vegetation state are retained only for test truth and are never consumed by a policy.",
        "",
        "## Design",
        "",
        f"- Calibration replicates per scale: {config.calibration_replicates}",
        f"- Held-out test replicates per scale: {config.test_replicates}",
        f"- Frames per block: {config.frames}",
        f"- Nuisance scales: {', '.join(map(str, config.nuisance_scales))}",
        f"- Calibration target recall: {config.target_recall:.2f}",
        f"- Time-shifted control delay: {config.reference_delay_frames} frames",
        "",
        "## Calibration thresholds",
        "```text",
        *[f"{policy}: {value:.5f}" for policy, value in sorted(thresholds.items())],
        "```",
        "",
        "## Held-out summary",
        "",
        "| policy | nuisance scale | recall | false-event rate | global-shift error | reference coherence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {nuisance_scale:.2f} | {recall} | {false_event_rate} | {mean_global_shift_error:.3f} | {mean_reference_coherence:.4f} |".format(
                **{
                    **row,
                    "recall": _format(row["recall"]),
                    "false_event_rate": _format(row["false_event_rate"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "The correct visual reference must reduce false-event rate at matched recall relative to raw pixel differencing, while time-shifted and physically mismatched visual references lose that advantage. This is still a controlled rendered-video result, not field generalisation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
