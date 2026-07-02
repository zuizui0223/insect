# NoiseBench: controlled perturbation protocol

## Purpose

NoiseBench is the first empirical layer of this project. It tests whether an
edge sensing system can recognise and preserve **observation disturbances**
before any target-specific ecological detector is trusted.

```text
Not: can the camera identify a flower or insect?
But: can the camera identify when its own observations are likely to be wrong,
why they are likely to be wrong, and when it should retain extra evidence?
```

The benchmark is intentionally target-agnostic: no focal organism, species
label, or biological interaction is required.

## Primary hypotheses

### H1 — noise-state observability

The system can distinguish stable windows from controlled disturbance types and
reports uncertainty for mixed or unfamiliar conditions.

### H2 — error-channel prediction

The inferred noise state predicts the expected observation error channel:

```text
false event
missed event
wrong attribution
```

The evaluation is not limited to noise-class accuracy. A model that recognises
an artefact but cannot predict its effect on observation reliability has not
solved the methodological problem.

### H3 — audit efficiency

Noise-aware audit decisions retrieve more high-risk windows per unit storage and
per unit energy than uniform or event-trigger-only recording.

### H4 — no silent deletion

High-noise windows are retained as `confounded`, `audit_priority`,
`unobservable`, or `unknown`; they are never removed without a record.

## Core disturbance matrix

| Family | Physical perturbation | Primary expected error channel(s) |
|---|---|---|
| Stable control | Fixed scene | none |
| Camera motion | Controlled mount displacement | false event; missed event |
| Co-moving foreground | Near-lens foliage/card motion | false event |
| Background vegetation motion | Fan-driven flexible background | false event; wrong attribution |
| Illumination transient | Dimmable light / lamp shutter | false event; missed event |
| Shadow transient | Moving opaque flag | false event; missed event |
| Occlusion | Partial opaque / translucent cover | missed event; wrong attribution |
| Blur / focus loss | Focus offset / diffusion film | missed event |
| Lens contamination | Reversible droplet / particulate surrogate | missed event |
| Multi-object clutter | Overlapping moving markers / surrogates | false event; wrong attribution |
| Mixed disturbance | Two overlapping controlled perturbations | all channels / unknown |

The table expresses **mechanistic hypotheses**, not fixed truths about every
future deployment. The magnitude of each error is calibrated later for each
specific downstream sensing task.

## Recording structure

One run has a clean lead-in, a time-bounded perturbation, and a clean recovery.
The plan randomises run order while retaining a deterministic seed. Every run
stores:

```text
scenario ID and randomisation order
perturbation source, intensity, start, end, apparatus, and operator instruction
raw low-cost image stream
IMX500 inference output and KPI metadata
Pi-side temporal features
battery, storage, camera configuration
noise-state output
observability decision
audit/context capture decision
```

The `interaction-noisebench-plan` command creates this manifest *before
recording*, avoiding retrospective relabelling.

## First field-side implementation

```bash
interaction-noisebench-plan \
  --replicates 3 \
  --duration-seconds 30 \
  --frame-rate 15 \
  --output-dir runs/noisebench_2026-07-02
```

This writes:

```text
noisebench_manifest.csv  one row per planned recording / perturbation
noisebench_windows.csv   one-second truth windows for later model matching
noisebench_config.json   randomisation and design assumptions
noisebench_protocol.md   portable operator-facing protocol
```

## IMX500 role

IMX500 should host a lightweight model that produces a scene-noise or
observability-quality representation. The Pi should add temporal variables that
single frames cannot resolve reliably:

```text
IMX500: scene quality, clutter, occlusion, blur, photometric disturbance,
        contamination, unknown / out-of-distribution

Pi:     global displacement, coherent foreground motion, local relative motion,
        illumination change, temporal persistence
```

The initial official SSD `.rpk` is only a firmware/metadata smoke test. It is
not part of the scientific NoiseBench score.

## Evaluation hierarchy

### Level 1 — perturbation recognition

```text
balanced accuracy / macro F1
multi-label score for mixed conditions
intensity monotonicity
unknown-condition rejection
```

### Level 2 — observation-risk calibration

```text
calibration of false-event risk
calibration of missed-event risk
calibration of attribution-risk
Brier score / expected calibration error
condition-specific uncertainty
```

### Level 3 — resource-aware audit value

```text
high-risk windows retrieved per GB
high-risk windows retrieved per Wh
recall of unobservable windows
false reassurance rate: clean prediction during high-risk truth
```

### Level 4 — downstream task transfer

Only after Levels 1–3, introduce a generic zone-entry or object-event task.
Then test whether noise-aware recording and modelling reduce raw-count bias
relative to a fixed-interval or fixed-ROI baseline.

Biological flower--visitor scenes are one future task in Level 4, not the
benchmark definition.

## What NoiseBench can and cannot prove

NoiseBench can establish a reproducible, target-independent sensing and audit
method. It cannot by itself establish that any species detector works in a wild
community, nor that a particular ecological interaction rate is unbiased.

Its scientific value is to make later target-specific claims conditional on an
explicit, measured observation process rather than on unrecorded camera
failures.
