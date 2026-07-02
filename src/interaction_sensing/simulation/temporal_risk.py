"""Lightweight temporal ML for nuisance-risk estimation from visual references.

This is intentionally not an object recogniser. The model consumes short
sequences of *scene-level* visual features derived from reference regions:
robust/single reference changes, cross-reference disagreement, and image-derived
camera displacement. It never receives a flower, insect, class label, bounding
box, or hidden renderer nuisance variable as an input.

The reference implementation is a small two-layer NumPy MLP so its input/output
contract is transparent and can later be reimplemented on Pi / IMX500 tooling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .visual_benchmark import VisualFeatureFrame


@dataclass(frozen=True, slots=True)
class TemporalRiskModelConfig:
    """Training and architecture settings for the small temporal MLP."""

    window_frames: int = 6
    hidden_units: int = 16
    epochs: int = 350
    learning_rate: float = 0.04
    l2: float = 0.0005
    seed: int = 20260702

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalRiskTrainingSummary:
    """Training summary saved alongside a reproducible model artifact."""

    samples: int
    positive_samples: int
    feature_count: int
    final_loss: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalRiskModel:
    """A compact dense temporal risk model with explicit normalisation state."""

    config: TemporalRiskModelConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(false candidate | temporal scene-reference history)."""

        values = _validate_feature_matrix(features)
        if values.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"feature width {values.shape[1]} does not match model width {self.feature_mean.size}"
            )
        standardised = (values - self.feature_mean) / self.feature_scale
        hidden = np.maximum(0.0, standardised @ self.hidden_weights + self.hidden_bias)
        logits = hidden @ self.output_weights + self.output_bias
        return _sigmoid(logits)

    def save(self, path: str | Path) -> Path:
        """Persist a dependency-free artifact suitable for later conversion tooling."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            config=np.asarray([self.config.to_dict()], dtype=object),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            hidden_weights=self.hidden_weights,
            hidden_bias=self.hidden_bias,
            output_weights=self.output_weights,
            output_bias=np.asarray([self.output_bias], dtype=np.float64),
        )
        return output


def temporal_reference_matrix(
    features: Iterable[VisualFeatureFrame],
    indices: Iterable[int],
    *,
    window_frames: int,
) -> np.ndarray:
    """Create temporal scene-reference inputs without focal local evidence.

    Each row is a causal history ending at a candidate frame. The five channels
    are all image-derived scene quantities:

    ```text
    robust shared-scene change
    single-region change
    cross-reference disagreement
    estimated global shift y
    estimated global shift x
    ```

    Focal local evidence is intentionally excluded. The model learns whether the
    *observation context* resembles a false-candidate condition rather than
    re-learning a target detector.
    """

    history = list(features)
    selected = list(indices)
    if not history:
        raise ValueError("features cannot be empty")
    if window_frames < 2:
        raise ValueError("window_frames must be at least 2")
    if any(index < 0 or index >= len(history) for index in selected):
        raise IndexError("candidate index is outside feature history")

    channels = np.asarray(
        [
            [
                frame.robust_reference,
                frame.single_reference,
                frame.reference_coherence,
                float(frame.global_shift_y),
                float(frame.global_shift_x),
            ]
            for frame in history
        ],
        dtype=np.float64,
    )
    rows: list[np.ndarray] = []
    for index in selected:
        start = max(0, index - window_frames + 1)
        segment = channels[start : index + 1]
        if segment.shape[0] < window_frames:
            padding = np.repeat(segment[:1], window_frames - segment.shape[0], axis=0)
            segment = np.concatenate((padding, segment), axis=0)
        rows.append(segment.reshape(-1))
    return np.asarray(rows, dtype=np.float64)


def fit_temporal_risk_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: TemporalRiskModelConfig = TemporalRiskModelConfig(),
) -> tuple[TemporalRiskModel, TemporalRiskTrainingSummary]:
    """Fit the MLP using weighted binary cross-entropy.

    Labels are supplied by the benchmark harness after it has fixed a candidate
    definition on calibration worlds. The model itself sees only temporal visual
    reference features.
    """

    values = _validate_feature_matrix(features)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if target.size != values.shape[0]:
        raise ValueError("labels must have one value per feature row")
    if not np.all(np.isin(target, (0.0, 1.0))):
        raise ValueError("labels must be binary")
    positive_samples = int(np.sum(target))
    if positive_samples == 0 or positive_samples == target.size:
        raise ValueError("training requires both false and true candidate examples")

    feature_mean = values.mean(axis=0)
    feature_scale = values.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    x = (values - feature_mean) / feature_scale
    rng = np.random.default_rng(config.seed)
    hidden_weights = rng.normal(
        0.0,
        1.0 / np.sqrt(x.shape[1]),
        size=(x.shape[1], config.hidden_units),
    )
    hidden_bias = np.zeros(config.hidden_units, dtype=np.float64)
    output_weights = rng.normal(
        0.0,
        1.0 / np.sqrt(config.hidden_units),
        size=config.hidden_units,
    )
    output_bias = 0.0

    positive_weight = target.size / (2.0 * positive_samples)
    negative_weight = target.size / (2.0 * (target.size - positive_samples))
    sample_weight = np.where(target == 1.0, positive_weight, negative_weight)
    final_loss = float("nan")
    for _ in range(config.epochs):
        hidden_pre = x @ hidden_weights + hidden_bias
        hidden = np.maximum(0.0, hidden_pre)
        logits = hidden @ output_weights + output_bias
        probability = _sigmoid(logits)
        weighted_error = (probability - target) * sample_weight / sample_weight.mean()
        grad_output_weights = hidden.T @ weighted_error / target.size + config.l2 * output_weights
        grad_output_bias = float(np.mean(weighted_error))
        grad_hidden = weighted_error[:, None] * output_weights[None, :]
        grad_hidden_pre = grad_hidden * (hidden_pre > 0.0)
        grad_hidden_weights = x.T @ grad_hidden_pre / target.size + config.l2 * hidden_weights
        grad_hidden_bias = grad_hidden_pre.mean(axis=0)
        hidden_weights -= config.learning_rate * grad_hidden_weights
        hidden_bias -= config.learning_rate * grad_hidden_bias
        output_weights -= config.learning_rate * grad_output_weights
        output_bias -= config.learning_rate * grad_output_bias
        final_loss = _weighted_bce(probability, target, sample_weight)

    model = TemporalRiskModel(
        config=config,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_weights=hidden_weights,
        hidden_bias=hidden_bias,
        output_weights=output_weights,
        output_bias=float(output_bias),
    )
    return model, TemporalRiskTrainingSummary(
        samples=int(target.size),
        positive_samples=positive_samples,
        feature_count=int(values.shape[1]),
        final_loss=float(final_loss),
    )


def _validate_feature_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return values


def _sigmoid(values: np.ndarray) -> np.ndarray:
    bounded = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _weighted_bce(probability: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(weights * (labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
    )
