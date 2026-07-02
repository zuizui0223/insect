# Pre-field validation strategy

## Aim

Before collecting biological field data, establish whether the proposed
observation design has a **mechanistic advantage** over a fixed-region,
motion-only camera trigger under controlled, labelled conditions.

This document distinguishes what can be demonstrated before fieldwork from what
requires field validation.

## The claim ladder

### Claim A — architecture-level advantage: testable before fieldwork

The proposed architecture should reduce predictable error mechanisms when the
underlying truth is known:

1. target-relative motion should suppress false triggers caused by target sway;
2. multi-target geometry should reduce wrong-target attribution in dense displays;
3. nested interaction zones should separate pass-by/context entry from target
   contact/access;
4. randomly sampled audit clips should expose and partially correct raw-event
   bias.

The synthetic benchmark implements these falsifiable tests with a fixed random
seed and exports every assumption, truth label, system decision, and metric.

### Claim B — sensing feasibility: testable before fieldwork

Use controlled desktop or greenhouse clips in which target position, actor path,
wind surrogate, illumination, and neighbouring targets can be manipulated.
The same event ledger and audit schema used in simulation must be used here.

### Claim C — ecological validity: requires field data

Only field validation can establish whether the design recovers real
flower--visitor interaction rates, visitor behaviour, or treatment effects in
natural backgrounds. Simulation is **not** evidence of field generalisation.

## Benchmark comparison

```text
Fixed-context baseline
  fixed initial target context
  any detected motion in that context is counted as focal activity

Target-relative attribution-aware proposal
  current target coordinates are estimated each frame
  co-moving target sway is removed from relative motion
  candidates are assigned across focal and neighbouring targets
  ambiguous candidates are retained as ambiguous rather than counted as focal
  focal interaction requires core-zone contact or access-zone entry
```

The benchmark evaluates both systems against known truth under a factorial
scenario matrix:

```text
wind / target sway
neighbour distance and display overlap
pass-by rate
neighbour-visitor rate
target-tracker error
detection miss probability
false-candidate probability
```

## Required output

Every benchmark run writes:

- `scenario_metrics.csv`: one row per replicate and scenario;
- `benchmark_summary.csv`: mean metrics per scenario and policy;
- `benchmark_report.md`: a human-readable comparison;
- `assumptions.json`: complete input configuration.

Minimum endpoints:

```text
true focal interaction count
observed focal-event count
precision
recall
wrong-target attribution rate
plant-motion false-event rate
ambiguous-event rate
raw count bias
audit-adjusted count bias
```

## Pre-field decision criterion

Do not claim superiority merely because the proposed system has a higher
synthetic F1 score. Advance to controlled video and field deployment only when:

1. errors change in the predicted direction as wind and target overlap increase;
2. the proposed policy reduces the relevant error source without an unacceptable
   loss of focal-event recall;
3. audit-adjusted estimates are closer to truth than raw observed counts;
4. performance remains beneficial under non-zero target-tracker error.

The important result is not "AI succeeds", but:

> A sensing design that represents target motion, competing targets, and audit
> uncertainty produces less biased interaction estimates under the conditions
> that make fixed-ROI monitoring fail.
