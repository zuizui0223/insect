# Latent disturbance inference benchmark

## The methodological shift

Most target-first computer vision asks:

```text
what object is visible?
where is it?
did it move?
```

This benchmark asks an earlier question:

```text
is an apparent local change independent,
or is it a visible signature of an unobserved external disturbance?
```

Wind, camera movement, passing shadow, exposure variation, blur, and partial
occlusion may not be directly visible as named objects. Their **spatiotemporal
consequences** are visible: scene-wide correlation, coherent motion, non-rigid
sway, shared luminance change, and loss of observable detail.

The initial simulation abstracts those mechanisms into a latent nuisance field:

\[
I_t = S_t + N_t + \epsilon_t
\]

where:

```text
I_t  observed local evidence
S_t  independent local signal that should be retained
N_t  shared nuisance field
ε_t  observation / sensor noise
```

The benchmark contains no flower, insect, species identity, or object detector.
It therefore tests whether a sensing architecture can separate *source of
change* before asking what the changing object is.

## Reference-guided cancellation

Background regions provide a reference channel for the nuisance contribution.
The core comparison is:

```text
raw local evidence
  versus
raw local evidence - robust shared-scene reference
```

A separate quality side channel represents a future temporal ML/DL or IMX500
nuisance-state model. It is used for risk estimation and adaptive audit choice,
not as a biological label.

## Causal negative controls

The benchmark deliberately includes references that are wrong in known ways:

```text
absent reference
single weak reference region
time-shifted reference
spatially mismatched reference
degraded reference
```

All policies see the **identical local observation stream**. Only the reference
condition changes. A valid mechanistic result therefore requires:

```text
correct robust reference
  -> lower false-event rate at matched true-signal recall

broken reference
  -> loss or substantial reduction of that advantage
```

If a method wins equally with mismatched or time-shifted references, its claimed
reference-guided mechanism has not been demonstrated.

## Predeclared outputs

The CLI separates calibration worlds from held-out test worlds.

```bash
interaction-latent-benchmark --output-dir runs/latent_benchmark
```

It writes:

```text
latent_disturbance_metrics.csv
latent_disturbance_summary.csv
latent_disturbance_config.json
calibration_thresholds.json
latent_disturbance_report.md
```

Thresholds are chosen only in calibration worlds at a fixed true-event recall.
The held-out worlds are used for all reported performance comparisons.

## Primary endpoints

```text
false-event rate at matched true-local-signal recall
local-signal distortion after cancellation
risk Brier score and expected calibration error
audit failure yield and lift at a fixed audit budget
```

The audit target is a nuisance-dominant non-event window. The method should not
silently discard such windows; it should identify them as high-risk and make
them available for audit or high-resolution context capture.

## Relation to later video and field work

This benchmark is not a claim that a system already recognises insects, flowers,
or ecological interactions. It establishes the causal observation layer:

```text
latent disturbance inference
  -> nuisance contribution estimate
  -> residual local evidence
  -> observability risk
  -> audit decision
```

Later video and field work can attach YOLO, a classifier, or a biological event
model downstream. Those models should consume the observability record rather
than treating every visual change as direct biological evidence.
