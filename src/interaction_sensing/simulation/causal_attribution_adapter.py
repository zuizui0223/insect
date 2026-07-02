"""Adapters from visual-reference features to causal-attribution evidence.

The adapter is intentionally explicit about normalisation scales. It receives
only features already estimated from pixels plus an optional N1 false-candidate
risk score. Hidden renderer variables and biological labels are not inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from interaction_sensing.sensing.causal_attribution import CausalAttributionEvidence

from .visual_benchmark import VisualFeatureFrame


@dataclass(frozen=True, slots=True)
class VisualAttributionScale:
    """Calibration scales for converting raw visual features to [0, 1] evidence.

    These defaults support deterministic rendered-video experiments only. A real
    camera deployment must fit/lock equivalent scales on calibration footage and
    persist the fitted values with each run.
    """

    residual_support_scale: float = 0.045
    reference_coherence_scale: float = 0.050
    shift_error_scale: float = 3.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.residual_support_scale <= 0.0:
            raise ValueError("residual_support_scale must be positive")
        if self.reference_coherence_scale <= 0.0:
            raise ValueError("reference_coherence_scale must be positive")
        if self.shift_error_scale <= 0.0:
            raise ValueError("shift_error_scale must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evidence_from_visual_feature(
    feature: VisualFeatureFrame,
    *,
    false_candidate_risk: float,
    scale: VisualAttributionScale = VisualAttributionScale(),
) -> CausalAttributionEvidence:
    """Build a target-free attribution record from image-derived visual features.

    ``local_residual_support`` uses the N0 residual after robust-reference
    subtraction. ``shared_explanation_fraction`` asks how much of the stabilised
    local change could be accounted for by the robust shared reference. The
    remaining channels report whether reference regions agree and whether image
    alignment was reliable enough to interpret a local residual.
    """

    _validate_probability("false_candidate_risk", false_candidate_risk)
    local_change = abs(feature.stabilised_local_evidence)
    shared_change = abs(feature.robust_reference)
    residual_change = abs(feature.stabilised_local_evidence - feature.robust_reference)
    local_residual_support = _saturate(residual_change / scale.residual_support_scale)
    shared_explanation_fraction = _saturate(shared_change / (local_change + scale.epsilon))
    reference_agreement = 1.0 - _saturate(
        feature.reference_coherence / scale.reference_coherence_scale
    )
    shift_quality = 1.0 - _saturate(feature.global_shift_error / scale.shift_error_scale)
    observability_quality = reference_agreement * shift_quality
    return CausalAttributionEvidence(
        local_residual_support=local_residual_support,
        shared_explanation_fraction=shared_explanation_fraction,
        reference_agreement=reference_agreement,
        observability_quality=observability_quality,
        false_candidate_risk=false_candidate_risk,
        metadata={
            "source": "visual_reference_adapter_v1",
            "frame_index": feature.frame_index,
            "stabilised_local_evidence": feature.stabilised_local_evidence,
            "robust_reference": feature.robust_reference,
            "reference_coherence": feature.reference_coherence,
            "global_shift_error": feature.global_shift_error,
            "scale": scale.to_dict(),
        },
    )


def _saturate(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
