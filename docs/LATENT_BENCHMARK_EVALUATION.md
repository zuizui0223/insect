# Latent benchmark evaluation contract

## Why an evaluator is needed

A causal simulator can still be overclaimed if results are summarised only as
mean accuracy or selected figures. This evaluator fixes the unit of inference,
the effect directions, the comparison set, and the interpretation rule before
later model changes.

## Independent unit

```text
one matched Monte Carlo recording block
= nuisance scale × replicate
```

All policies receive the same latent world within a block. Frames are correlated
observations within that block and are never used as independent statistical
replicates.

## Effect direction

Every reported paired effect is expressed so that a positive value favours the
intervention:

```text
false-event reduction       comparator FPR - intervention FPR
recall change               intervention recall - comparator recall
signal-distortion reduction comparator distortion - intervention distortion
risk-calibration reduction  comparator score - intervention score
audit-yield gain            intervention yield - comparator yield
```

The central comparisons are:

```text
N0 robust reference vs B0 raw motion
N0 robust reference vs NC time-shifted reference
```

The first asks whether cancellation helps. The second asks whether that help is
specific to a reference retaining the correct shared cause.

## Bootstrap intervals

For every condition and comparison, the evaluator resamples matched recording
blocks with replacement and reports percentile bootstrap intervals. It does not
report a frame-level p-value.

```text
latent_paired_effects.csv
  all effect estimates and intervals

latent_failure_map.csv
  one diagnostic conclusion per nuisance intensity

latent_evaluation_report.md
  paper-ready table with all conditions
```

## Failure-map labels

These are diagnostics, not labels to hide when unfavourable.

```text
mechanism_supported
  false-event reduction exceeds the locked minimum;
  recall loss stays inside the noninferiority margin;
  correct reference is better than a time-shifted reference.

no_supported_cancellation_advantage
  the interval for false-event reduction includes the locked minimum.

recall_cost
  the lower interval for recall change falls below the locked margin.

reference_non_specific
  correct reference does not clearly beat a time-shifted reference.
```

The default recall noninferiority margin is `-0.05`. This is a methodological
contract, not an assertion that a 5% reduction is harmless for every future
biological task. A downstream application may specify a stricter task-specific
margin, but it must not overwrite the original result.

## How this supports the paper

The paper can report all nuisance scales, including conditions where the method
fails. The mechanistic claim is supported only when the correct reference wins
against both raw motion and a deliberately broken reference while preserving the
independent local signal.

This evaluator still does not establish field generalisation. Its role is to
make the simulation result falsifiable, effect-size based, and resistant to
post-hoc metric selection before physical NoiseBench and hardware validation are
added.
