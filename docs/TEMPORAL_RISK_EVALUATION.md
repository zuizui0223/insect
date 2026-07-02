# N1 temporal-risk evaluation contract

## The question being tested

The N1 temporal MLP is useful only if it adds information after N0
reference-guided cancellation. The question is not whether a neural network can
fit rendered data. It is:

> At a protected total true-event recall, does a learned temporal scene-context
> risk gate reduce false candidates more than N0 alone and more than a transparent
> rule-based gate using the same observation context?

## Independent unit

```text
one matched held-out rendered recording block
= nuisance scale × replicate
```

Every policy sees the same rendering within a block. Individual frames are
correlated samples within that recording and must not be analysed as independent
replicates. Percentile bootstrap resampling therefore resamples blocks.

## Predeclared paired comparisons

```text
N0 robust reference
  vs B0 raw pixel difference

N1 rule gate
  vs N0 robust reference

N1 temporal MLP gate
  vs N0 robust reference

N1 temporal MLP gate
  vs N1 rule gate
```

Effects are always oriented so positive values favour the intervention.

```text
false-event reduction
  comparator false-event rate - intervention false-event rate

recall change
  intervention recall - comparator recall

precision gain
  intervention precision - comparator precision
```

For two risk-gating policies sharing the same high-recall candidate stream, the
evaluator also records candidate abstention change and top-10% audit-yield gain.
Those quantities are not interpreted for B0 or N0, because those policies do
not share the risk-gating candidate threshold.

## Locked N1 failure map

One map cell is emitted for each held-out nuisance scale. It contains:

```text
N0 false-event reduction vs B0
N1-MLP false-event reduction vs N0
N1-MLP recall change vs N0
N1-MLP false-event reduction vs N1-rule
```

The labels are deliberately diagnostic rather than promotional:

```text
n0_not_supported
  N0 does not show the locked false-event reduction versus raw differencing.

recall_cost
  MLP recall loss exceeds the locked noninferiority margin.

no_mlp_increment_over_n0
  MLP does not clearly reduce false events beyond N0.

no_mlp_increment_over_rule
  MLP does not clearly reduce false events beyond the transparent rule gate.

mlp_increment_supported
  all of the above checks are satisfied.
```

Default rules:

```text
95% percentile bootstrap interval
minimum false-event reduction = 0
recall noninferiority margin = -0.05
```

The margin is a benchmark-level contract, not a claim that a 5% recall loss is
biologically acceptable in every later application. Biological deployments must
set stricter task-specific limits before using N1 to exclude observations.

## Outputs

Running:

```bash
interaction-temporal-risk-benchmark \
  --output-dir runs/temporal_risk_benchmark
```

now writes in addition to the original benchmark artifacts:

```text
temporal_risk_paired_effects.csv
temporal_risk_failure_map.csv
temporal_risk_evaluation_config.json
temporal_risk_evaluation_report.md
```

## Claim boundary

This evaluator can support a claim about held-out rendered-video conditions. It
does not establish that N1 generalises to physical camera data, weather, new
habitats, or biological targets. The next evidence layer is a controlled
physical NoiseBench in which N0/N1 features and the same evaluator are fed from
real video and independently recorded perturbation truth.
