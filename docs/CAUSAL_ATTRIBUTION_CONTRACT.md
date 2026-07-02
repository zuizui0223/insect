# Shared-cause causal attribution contract

## The central question

A conventional detector begins with:

```text
what object is present?
```

The noise-first observation layer begins earlier:

```text
can an apparent local change be explained by a shared exogenous scene field,
or does independent local residual evidence remain after accounting for it?
```

This is not an attempt to read an insect's intention. It is an
**observation-process attribution**.

```text
insect, pollen, animal, moving marker, or other focal target
  may create a local change from a locally generated source

wind, camera displacement, shadow, exposure change, occlusion
  may create changes shared across many image regions
```

The proposed system does not assume that every local change belongs to one of
these pure cases. It retains confounded cases rather than forcing a binary label.

## Four output states

```text
independent_local
  A local residual remains after reference-guided cancellation; shared
  explanation and false-candidate risk are low enough under a locked policy.

shared_exogenous
  The apparent local change is largely explained by a contemporaneous shared
  scene field and little independent local residual remains.

coupled_or_ambiguous
  Both local-residual and shared/risk evidence are plausible, or the reference
  regions disagree. This includes a true local event observed during strong
  wind/shadow/camera disturbance, not merely false positives.

unobservable
  Image quality is too poor to support a stable attribution.
```

These are **not** classes such as `insect`, `flower`, `wind`, or `no insect`.
They record whether a visual observation can be treated as independent evidence.

## Required decision actions

```text
independent_local
  -> count_as_independent_evidence

shared_exogenous
  -> retain_context_no_count

coupled_or_ambiguous
  -> prioritise_audit

unobservable
  -> mark_observability_gap
```

`retain_context_no_count` does not mean delete. A clip or compressed context may
be retained for auditing and calibration. `prioritise_audit` is essential: it
prevents a potential biological event observed under an external disturbance
from being silently discarded.

## Evidence record

The state machine receives five normalised quantities:

```text
local_residual_support
  Strength of local change left after N0 shared-scene cancellation.

shared_explanation_fraction
  Fraction of raw local change accounted for by a contemporaneous shared field.

reference_agreement
  Whether independent background reference regions tell a consistent story.

observability_quality
  Whether alignment/reference quality permits interpretation at all.

false_candidate_risk
  N1 probability-like risk from the causal history of background disturbance.
```

The runtime contract is therefore:

```text
pixels
  -> reference features
  -> N0 shared-scene cancellation
  -> N1 false-candidate risk
  -> causal attribution state + action
  -> optional object detector / biological interpretation
```

## Notation

For a local region at time \(t\):

\[
I_t^{local} = L_t + N_t^{shared} + \epsilon_t
\]

where:

```text
I_t^local      observed local change
L_t            independent local residual source
N_t^shared     shared exogenous disturbance contribution
ε_t            sensor / compression / unmodelled error
```

The system does not identify \(L_t\) with an insect. It asks whether the
observed local change has enough residual support after estimating
\(N_t^{shared}\) to be carried forward as independent observation evidence.

## Why purpose is not the model target

An insect often has a locally generated movement trajectory, but that is not a
reliable observable definition of purpose. An insect may be wind-driven; a
shadow may be regular and predictable; a moving plant can have strong local
structure. The deployable distinction is instead:

```text
locally independent residual evidence
  versus
change explained by a shared external field
```

The `coupled_or_ambiguous` state is the guardrail against overclaiming this
separation.

## Calibration boundary

The thresholds in `CausalAttributionConfig` are conservative defaults for the
rendered benchmark. They must be fitted on calibration footage and frozen before
reporting held-out physical or field results. No state may be used to suppress
biological observations without retaining an auditable record of the evidence,
policy version, and action.

## Claim boundary

The contract supports a methodological claim about observation attribution. It
does not establish that a system knows intent, distinguishes every insect from
every disturbance, or can make unreviewed ecological counts in new habitats.
Physical NoiseBench and target-specific field audits remain necessary.
