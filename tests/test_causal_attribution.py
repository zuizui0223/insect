from interaction_sensing.sensing import (
    AttributionAction,
    CausalAttributionConfig,
    CausalAttributionEvidence,
    CausalAttributionState,
    CausalAttributor,
)
from interaction_sensing.simulation.causal_attribution_adapter import evidence_from_visual_feature
from interaction_sensing.simulation.visual_benchmark import VisualFeatureFrame


def _evidence(**overrides: float) -> CausalAttributionEvidence:
    values = {
        "local_residual_support": 0.80,
        "shared_explanation_fraction": 0.10,
        "reference_agreement": 0.90,
        "observability_quality": 0.90,
        "false_candidate_risk": 0.10,
    }
    values.update(overrides)
    return CausalAttributionEvidence(**values)


def test_independent_local_state_requires_residual_and_low_shared_explanation() -> None:
    record = CausalAttributor().attribute(_evidence())
    assert record.state is CausalAttributionState.INDEPENDENT_LOCAL
    assert record.action is AttributionAction.COUNT_AS_INDEPENDENT_EVIDENCE
    assert "strong_local_residual_after_shared_cancellation" in record.reasons


def test_shared_exogenous_state_never_becomes_a_biological_count() -> None:
    record = CausalAttributor().attribute(
        _evidence(
            local_residual_support=0.12,
            shared_explanation_fraction=0.89,
            false_candidate_risk=0.82,
        )
    )
    assert record.state is CausalAttributionState.SHARED_EXOGENOUS
    assert record.action is AttributionAction.RETAIN_CONTEXT_NO_COUNT


def test_coupled_or_ambiguous_routes_to_audit_instead_of_forced_binary_label() -> None:
    record = CausalAttributor().attribute(
        _evidence(
            local_residual_support=0.66,
            shared_explanation_fraction=0.63,
            false_candidate_risk=0.56,
        )
    )
    assert record.state is CausalAttributionState.COUPLED_OR_AMBIGUOUS
    assert record.action is AttributionAction.PRIORITISE_AUDIT


def test_low_observability_overrides_other_evidence() -> None:
    record = CausalAttributor().attribute(_evidence(observability_quality=0.15))
    assert record.state is CausalAttributionState.UNOBSERVABLE
    assert record.action is AttributionAction.MARK_OBSERVABILITY_GAP


def test_custom_thresholds_are_validated() -> None:
    try:
        CausalAttributionConfig(
            minimum_residual_for_coupled=0.8,
            minimum_residual_for_independent=0.6,
        )
    except ValueError as error:
        assert "coupled residual threshold" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid threshold ordering was accepted")


def test_visual_adapter_uses_reference_features_not_hidden_renderer_truth() -> None:
    feature = VisualFeatureFrame(
        frame_index=13,
        true_local_event=True,
        raw_local_evidence=0.80,
        stabilised_local_evidence=0.08,
        single_reference=0.04,
        robust_reference=0.05,
        delayed_reference=0.02,
        mismatched_reference=0.01,
        global_shift_y=1,
        global_shift_x=-1,
        global_shift_error=0.20,
        reference_coherence=0.01,
    )
    evidence = evidence_from_visual_feature(feature, false_candidate_risk=0.22)
    assert evidence.local_residual_support > 0.0
    assert 0.0 <= evidence.shared_explanation_fraction <= 1.0
    assert evidence.reference_agreement > 0.0
    assert evidence.observability_quality > 0.0
    assert evidence.metadata["source"] == "visual_reference_adapter_v1"
    assert "true_local_event" not in evidence.metadata
