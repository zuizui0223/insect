# N1 temporal nuisance-risk model

## What is being learned

N1 is the first learned component of the noise-first architecture. It is not a
flower detector, insect classifier, object detector, or a generic weather
predictor.

Its task is deliberately narrower:

> Among high-recall local residual candidates produced by N0 reference-guided
> cancellation, estimate the probability that the candidate is false from the
> recent history of scene-level visual disturbance features.

```text
rendered / camera pixels
  -> image-derived background reference features
  -> high-recall N0 candidate
  -> temporal MLP: P(false candidate | scene context)
  -> count / abstain / audit
```

The first model is a two-layer dense neural network, implemented with NumPy.
This keeps the model small and its feature contract explicit before a later
TFLite, IMX500, or learned image-embedding implementation is attempted.

## Inputs: what the MLP is allowed to see

Each candidate receives a causal history of several frame pairs. The only input
channels are image-derived, scene-level quantities:

```text
robust shared-scene reference change
single-reference-region change
cross-reference disagreement
estimated global displacement y
estimated global displacement x
```

The model does **not** receive:

```text
focal local evidence
flower / insect crop
class label
bounding box
hidden camera displacement
hidden illumination, shadow, or vegetation state
future frames
```

Excluding focal local evidence is intentional. Otherwise the MLP could become a
small target detector and the method would drift back toward YOLO-style object
recognition. N1 instead learns whether the *observation context* makes an N0
candidate unreliable.

## Supervision target

The calibration set first creates a high-recall N0 candidate stream. A candidate
is labelled positive for N1 when it is a candidate but has no true independent
local event:

```text
N1 label = false N0 residual candidate
```

This target is available in rendered/simulated worlds. It is not a claim that
field systems know the true label online. In a later physical NoiseBench, the
same label comes from independently controlled truth windows and audit footage.

## Calibration protocol

All thresholds and all MLP weights use calibration worlds only.

```text
B0 raw event threshold
  calibrated at target true-event recall

N0 robust residual threshold
  calibrated at target true-event recall

N1 candidate threshold
  calibrated at higher recall (default 0.97)

N1 rule / MLP risk gate
  calibrated to restore the locked total target recall
```

Held-out worlds are separately seeded and not used to select feature channels,
weights, event thresholds, or risk thresholds.

## Why include a transparent rule gate?

N1 is compared not only to N0 but also to a non-learned risk gate based on the
same scene features.

```text
N0  reference-guided residual candidate
N1-rule  hand-specified risk proxy gate
N1-MLP   learned temporal risk gate
```

This prevents a weak claim such as “any risk threshold improves results.” A
learned model must improve the false-event / recall trade-off relative to both
N0 and the transparent rule under held-out conditions.

## Outputs

```bash
interaction-temporal-risk-benchmark \
  --output-dir runs/temporal_risk_benchmark
```

writes:

```text
temporal_risk_metrics.csv
temporal_risk_summary.csv
temporal_risk_config.json
temporal_risk_calibration.json
temporal_risk_training_summary.json
temporal_risk_model.npz
temporal_risk_report.md
```

The `.npz` contains normalisation state and dense weights. It is a reproducible
reference artifact, not yet a deployable IMX500 package.

## Claim boundary

A successful result supports the following narrow claim:

> A compact temporal neural model using only image-derived background-reference
> histories can identify false residual candidates more efficiently than a
> non-learned risk proxy under held-out rendered observation disturbances.

It does not yet demonstrate:

```text
real-camera performance
field transfer
wind or weather prediction
species recognition
biological interaction inference
IMX500 deployment or TFLite conversion
```

Those need the next layers: controlled physical NoiseBench, hardware-in-the-loop
measurements on zuizui2, and application-specific biological audits.
