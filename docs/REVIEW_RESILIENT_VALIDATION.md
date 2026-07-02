# Review-resilient validation standard

## Non-negotiable distinction

A paper can be review-resistant **without biological field observations**, but
not without evidence. The evidence must be a combination of:

```text
known-truth synthetic tests
+ controlled physical perturbation recordings
+ hardware-in-the-loop deployment measurements
+ held-out outdoor operating conditions
```

The resulting paper can claim that the method improves *observation quality,
noise cancellation, risk calibration, and audit efficiency*. It cannot claim
that it already improves a biological population, visitor, or interaction
estimate until a target-specific downstream validation is performed.

## What is being compared

All methods must consume the **same recorded low-cost stream**, run under the
same storage and energy budget, and be evaluated on the same predeclared truth
windows.

```text
B0  fixed-interval recording
B1  conventional motion trigger / thresholding
B2  conventional preprocessing or global stabilisation only
B3  optional object detector confidence threshold

N0  reference-guided noise cancellation without learned scene model
N1  N0 + learned IMX500 noise-state / quality representation
N2  N1 + risk-guided audit and high-resolution-context capture
```

Object detectors are optional baselines: NoiseBench must be able to establish
its primary claims even when no biological detector is present.

## Primary endpoints

Predeclare only three primary outcomes.

### P1 — nuisance-cancellation benefit

At matched true-change recall, reduce false event windows relative to B1/B2.

```text
false-event rate at matched recall
paired difference per independent recording block
```

### P2 — observability-risk calibration

Predict whether a window is susceptible to a false event, missed event, or
wrong attribution better than a global confidence / motion threshold.

```text
Brier score
expected calibration error
area under precision-recall curve for high-risk windows
false reassurance rate
```

### P3 — audit efficiency under a fixed budget

At equal energy and storage budget, retrieve more independently verified
high-risk / failure windows than uniform or event-only audit sampling.

```text
verified failure windows per GB
verified failure windows per Wh
recall of unobservable windows at fixed audit budget
```

All other metrics are secondary and cannot be used to replace a failed primary
endpoint.

## Independent experimental unit

Frames are observations, not independent replicates. The independent unit for
statistical inference is a predeclared **recording block**:

```text
device × physical scene/background × recording session × perturbation schedule
```

Within each block, all compared methods operate on identical frames. Analyse
paired block-level contrasts; do not report frame-level p-values as evidence of
system superiority.

## Recommended inference

For each primary endpoint:

```text
report effect size
report 95% confidence / credible interval
use paired hierarchical bootstrap or blocked permutation across recording blocks
report all condition-specific estimates, not only pooled mean
```

A mixed-effects model may be used when enough blocks exist:

```text
endpoint ~ method * noise family * intensity
         + (1 | device) + (1 | physical scene) + (1 | session)
```

The interaction terms are essential. A single pooled advantage can conceal a
method that fails under the conditions it was designed to solve.

## Required validation tiers

### Tier 0 — analytic and synthetic truth

Use simulated motion/photometric fields with exact nuisance and target signal.
Test whether reference-guided subtraction removes the nuisance component without
removing a controlled independent local signal.

Required negative controls:

```text
time-shifted reference signal
spatially mismatched reference signal
reference channel with injected error
noise source absent
```

The proposed method must lose its advantage when its reference is deliberately
broken. Otherwise the claimed mechanism is not demonstrated.

### Tier 1 — controlled physical NoiseBench

Use the preregistered NoiseBench manifest. Record real camera data under camera
shake, foreground/background motion, lighting, shadows, occlusion, blur,
contamination, clutter, and mixed disturbances.

This tier supplies real optics, sensor artefacts, compression, and hardware
latency that synthetic tests cannot reproduce.

### Tier 2 — hardware-in-the-loop feasibility

Run N0/N1/N2 on the Raspberry Pi + IMX500 platform. Report:

```text
end-to-end latency
frame / inference throughput
power draw and Wh per recording hour
storage consumption
thermal stability
crash / recovery rate
continuous operation duration
```

These are deployability outcomes, not optional engineering details.

### Tier 3 — held-out outdoor operating conditions

Collect non-biological outdoor scenes that were never used for development:
new site/background, day, lighting regime, wind regime, and disturbance mix.
Known perturbations may still be introduced, but the physical context must be
unseen. This is the minimum transfer test for a field-deployment claim.

## Anti-overfitting safeguards

```text
freeze the benchmark taxonomy and primary endpoints before model tuning
separate development, validation, and locked test sessions by day and scene
hold out whole devices and whole physical scenes where possible
hold out at least one mixed-noise combination and one outdoor location
never tune thresholds using the locked test set
publish manifest, raw metadata, code, seeds, and failed runs
```

## What reviewers should be able to falsify

A reviewer must be able to ask — and the paper must answer — all of these:

```text
Does the proposed method still win when the sensor reference is wrong?
Does it preserve true local changes while cancelling coherent nuisance motion?
Does it work on unseen physical conditions, not only synthetic overlays?
Does risk calibration hold under mixed disturbance?
Does audit prioritisation win after equalising GB and Wh?
Are noisy windows retained instead of excluded post hoc?
Does the on-device system run long enough and cheaply enough for field use?
```

## Claim ladder for the first paper

### Defensible after Tiers 0–3

> We introduce a target-agnostic, reference-guided noise-cancellation and
> observability-calibration framework that reduces spurious visual events,
> quantifies observation risk, and improves audit efficiency under controlled
> and held-out outdoor disturbances on low-power edge hardware.

### Not defensible yet

```text
works for all ecological targets
improves visitor or pollination estimates
eliminates observation error in natural field studies
universally transfers across cameras, habitats, and weather
```

## Publication-ready artifact set

```text
protocol / preregistration
NoiseBench manifest and truth windows
raw low-cost streams and high-resolution audits
IMX500 outputs and Pi temporal features
resource logs
all baseline and proposed-method outputs
analysis notebooks with locked test split
negative-control experiments
failure analysis gallery
reproducible release with versioned model packages
```
