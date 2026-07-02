"""Attribution of local visual change to shared disturbance or local residual evidence.

This module intentionally does *not* infer an animal's purpose, species, or
mental state. It makes a narrower observation-process decision:

* is an apparent local change largely explained by a shared exogenous field?
* does independent local residual evidence remain after accounting for that field?
* is the scene coupled/ambiguous, or too degraded to interpret?

The output is an auditable observation state and action, not a biological label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CausalAttributionState(str, Enum):
    """Observation-process states; none of these is a species or intent label."""

    INDEPENDENT_LOCAL = "independent_local"
    SHARED_EXOGENOUS = "shared_exogenous"
    COUPLED_OR_AMBIGUOUS = "coupled_or_ambiguous"
    UNOBSERVABLE = "unobservable"


class AttributionAction(str, Enum):
    """Downstream handling for an attribution state."""

    COUNT_AS_INDEPENDENT_EVIDENCE = "count_as_independent_evidence"
    RETAIN_CONTEXT_NO_COUNT = "retain_context_no_count"
    PRIORITISE_AUDIT = "prioritise_audit"
    MARK_OBSERVABILITY_GAP = "mark_observability_gap"


@dataclass(frozen=True, slots=True)
class CausalAttributionEvidence:
    """Normalised, target-agnostic evidence available at one candidate window.

    ``local_residual_support`` represents how much candidate-like local change
    remains after reference-guided cancellation.  ``shared_explanation_fraction``
    represents the fraction of the raw local change that a contemporaneous shared
    scene field can explain.  These are observation quantities, not statements
    about a target's purpose.
    """

    local_residual_support: float
    shared_explanation_fraction: float
    reference_agreement: float
    observability_quality: float
    false_candidate_risk: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "local_residual_support",
            "shared_explanation_fraction",
            "reference_agreement",
            "observability_quality",
            "false_candidate_risk",
        ):
            _validate_probability(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalAttributionConfig:
    """Predeclared state thresholds to calibrate before a deployment.

    Defaults express a conservative routing policy. They are not biological
    decision thresholds and must be calibrated from held-out/physical audit data
    before a field deployment can use them to suppress observations.
    """

    minimum_observability_quality: float = 0.45
    minimum_reference_agreement: float = 0.55
    minimum_residual_for_independent: float = 0.60
    maximum_shared_explanation_for_independent: float = 0.35
    maximum_false_candidate_risk_for_independent: float = 0.35
    minimum_shared_explanation: float = 0.70
    minimum_residual_for_coupled: float = 0.35
    minimum_risk_for_coupled: float = 0.35

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _validate_probability(name, value)
        if self.minimum_residual_for_coupled > self.minimum_residual_for_independent:
            raise ValueError("coupled residual threshold cannot exceed independent threshold")
        if self.minimum_shared_explanation < self.maximum_shared_explanation_for_independent:
            raise ValueError("shared explanation thresholds overlap")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalAttribution:
    """A serialisable attribution outcome for one candidate observation window."""

    state: CausalAttributionState
    action: AttributionAction
    evidence: CausalAttributionEvidence
    reasons: tuple[str, ...]
    policy_version: str = "causal-attribution-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "action": self.action.value,
            "evidence": self.evidence.to_dict(),
            "reasons": list(self.reasons),
            "policy_version": self.policy_version,
        }


class CausalAttributor:
    """Route an apparent local change without equating it to a biological target.

    The policy deliberately favours ``COUPLED_OR_AMBIGUOUS`` over a forced
    independent/shared decision whenever a local residual and a shared field are
    both plausible.  This preserves context for audit instead of silently
    deleting potentially meaningful but confounded observations.
    """

    def __init__(self, config: CausalAttributionConfig | None = None) -> None:
        self.config = config if config is not None else CausalAttributionConfig()

    def attribute(self, evidence: CausalAttributionEvidence) -> CausalAttribution:
        cfg = self.config
        if evidence.observability_quality < cfg.minimum_observability_quality:
            return self._unobservable(
                evidence,
                "observability_quality_below_minimum",
            )
        if evidence.reference_agreement < cfg.minimum_reference_agreement:
            return self._ambiguous(
                evidence,
                "reference_regions_do_not_agree",
            )

        independent_support = (
            evidence.local_residual_support >= cfg.minimum_residual_for_independent
            and evidence.shared_explanation_fraction <= cfg.maximum_shared_explanation_for_independent
            and evidence.false_candidate_risk <= cfg.maximum_false_candidate_risk_for_independent
        )
        if independent_support:
            return CausalAttribution(
                state=CausalAttributionState.INDEPENDENT_LOCAL,
                action=AttributionAction.COUNT_AS_INDEPENDENT_EVIDENCE,
                evidence=evidence,
                reasons=(
                    "strong_local_residual_after_shared_cancellation",
                    "shared_explanation_below_independent_limit",
                    "false_candidate_risk_below_independent_limit",
                ),
            )

        shared_support = (
            evidence.local_residual_support < cfg.minimum_residual_for_coupled
            and evidence.shared_explanation_fraction >= cfg.minimum_shared_explanation
        )
        if shared_support:
            return CausalAttribution(
                state=CausalAttributionState.SHARED_EXOGENOUS,
                action=AttributionAction.RETAIN_CONTEXT_NO_COUNT,
                evidence=evidence,
                reasons=(
                    "local_change_largely_explained_by_shared_scene_field",
                    "insufficient_independent_local_residual",
                ),
            )

        coupled_support = (
            evidence.local_residual_support >= cfg.minimum_residual_for_coupled
            and (
                evidence.shared_explanation_fraction > cfg.maximum_shared_explanation_for_independent
                or evidence.false_candidate_risk >= cfg.minimum_risk_for_coupled
            )
        )
        if coupled_support:
            return self._ambiguous(
                evidence,
                "local_residual_and_shared_or_risk_evidence_coexist",
            )

        return self._ambiguous(
            evidence,
            "insufficient_evidence_for_independent_or_shared_attribution",
        )

    @staticmethod
    def _unobservable(evidence: CausalAttributionEvidence, reason: str) -> CausalAttribution:
        return CausalAttribution(
            state=CausalAttributionState.UNOBSERVABLE,
            action=AttributionAction.MARK_OBSERVABILITY_GAP,
            evidence=evidence,
            reasons=(reason,),
        )

    @staticmethod
    def _ambiguous(evidence: CausalAttributionEvidence, reason: str) -> CausalAttribution:
        return CausalAttribution(
            state=CausalAttributionState.COUPLED_OR_AMBIGUOUS,
            action=AttributionAction.PRIORITISE_AUDIT,
            evidence=evidence,
            reasons=(reason,),
        )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
