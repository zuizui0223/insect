# interaction-sensing

## Purpose

`interaction-sensing` is an experimental codebase for **noise-first, error-aware ecological sensing**.

The central question is not merely whether a camera can identify a flower, insect, animal, or other focal object. In natural scenes, camera motion, coherent foreground sway, shadows, illumination transients, occlusion, blur, lens contamination, and object clutter can create or hide observations before a target detector ever makes a decision.

This repository therefore asks:

> Can an autonomous edge camera recognise when and why its own observations are unreliable, preserve that uncertainty as data, and use it to adapt auditing and downstream ecological inference?

## Central inversion

```text
Conventional target-first pipeline
  detect target -> detect organism -> count event -> treat noise as nuisance

Noise-first pipeline
  characterise scene disturbance -> estimate observation risks
  -> preserve uncertainty / prioritise audit -> optionally interpret targets
```

Flowers, insects, nests, fruits, leaves, camera-trap animals, and other focal entities are downstream applications. They are not the core methodological object.

## Core outputs

The primary output is an **observability record** for each frame or time window:

```text
noise source
noise confidence
false-event risk
missed-event risk
attribution-risk
observability state
whether high-resolution context should be retained
whether an independent audit clip should be captured
```

A noisy interval is never silently dropped:

```text
clean          -> ordinary low-cost monitoring
confounded     -> retain noise metadata
high risk      -> prioritise audit and high-resolution context
unobservable   -> retain the condition; do not force a biological conclusion
unknown        -> preserve uncertainty and audit
```

## Function-first architecture

```text
1. scene-noise / observability measurement
2. temporal motion and photometric features
3. false-event / missed-event / attribution-risk estimation
4. adaptive context and independent-audit recording
5. noise-aware error modelling and calibration
6. optional target specification and localisation
7. optional event segmentation and interaction-zone states
8. optional actor guild or taxon recognition
```

Target and organism models are retained as optional ablations and applications. They must never turn a detection directly into an ecological claim.

## What is implemented now

- target-agnostic `NoiseObservation`, `NoiseSource`, `ObservabilityState`, and transparent risk policy;
- IMX500 hardware adapter that records model role, ROI, outputs, and KPI metadata;
- target-agnostic **NoiseBench** protocol generator for controlled perturbations;
- target-relative motion primitives, event data contracts, audit sampling, SQLite ledger, and error-evaluation utilities;
- legacy motion / detector / classifier scripts retained as explicit ablation baselines;
- a previous interaction-level synthetic benchmark, now treated as a downstream stress test rather than the primary method.

## Repository map

- `src/interaction_sensing/noise.py` — noise source, observability record, and transparent risk policy.
- `src/interaction_sensing/noisebench/` — pre-recording manifest generation for target-agnostic controlled perturbation experiments.
- `src/interaction_sensing/plugins/imx500.py` — optional IMX500 hardware / metadata adapter.
- `src/interaction_sensing/` — reusable contracts for targets, candidates, events, audits, ledgers, simulation, and evaluation.
- `docs/NOISE_FIRST_METHOD.md` — the conceptual framework.
- `docs/NOISEBENCH_PROTOCOL.md` — controlled perturbation protocol and endpoints.
- `docs/IMX500_DEPLOYMENT.md` — IMX500 as a scene-observability sensor.
- `docs/REPOSITORY_OVERVIEW.md` — index of every doc in `docs/`, plus a one-page map of the whole repository.
- `analysis/` — scripts that turn a recorded event ledger and audit annotations into observability/error estimates; see `analysis/README.md`.
- `configs/` — named baseline and NoiseBench pipeline configurations (`.toml`).
- `models/` — trained weights for the legacy Cirsium YOLO target-detection baseline; see `models/README.md`.
- `legacy/` — original prototype scripts, organised as runtime, target detection, recognition, and data utilities; see `legacy/README.md`.
- `tests/` — pytest suite for the `interaction_sensing` package.

## Quick start

```bash
python -m pip install -e ".[runtime,analysis,dev]"
pytest
```

## Build a NoiseBench recording plan

NoiseBench needs no organism data. It creates a randomised, reproducible schedule for stable controls, camera shake, foreground/background motion, lighting changes, shadows, occlusion, blur, lens contamination, clutter, and mixed disturbances.

```bash
interaction-noisebench-plan \
  --replicates 3 \
  --duration-seconds 30 \
  --frame-rate 15 \
  --output-dir runs/noisebench_plan
```

The output is created **before recording**:

```text
noisebench_manifest.csv  randomised recording order and perturbation truth
noisebench_windows.csv   one-second truth windows for later matching
noisebench_config.json   design assumptions and seed
noisebench_protocol.md   operator-facing protocol
```

See `docs/NOISEBENCH_PROTOCOL.md` for the experimental matrix and endpoints.

## IMX500 role

The Raspberry Pi AI Camera is intended first as a low-power scene-noise / image-quality sensor:

```text
stable scene
photometric disturbance
occlusion
blur or focus loss
lens contamination
clutter / overlap
unknown condition
```

The Pi supplements that with temporal features that one frame cannot resolve, such as global camera displacement and coherent foreground motion. Generic SSD models are only hardware/metadata smoke tests, not ecological detectors.

## Downstream applications

After NoiseBench establishes the observation process, target-specific tasks can test whether noise-aware sensing changes conclusions:

```text
object-zone entry
flower--visitor recording
nest entrance activity
fruit or leaf use
camera-trap species observations
```

The key endpoint is not classifier F1 alone. It is whether a noise-aware system better predicts and corrects false events, missed events, and attribution errors under finite storage and power.

## Status

This is an active research prototype. The next empirical milestone is a hardware-in-the-loop NoiseBench dataset and a lightweight IMX500 noise-state model; no field ecological claim should be made before that calibration exists.
