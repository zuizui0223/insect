# Target architecture

## Refactor goal

Move from experiment-named scripts to a reusable package for **target--actor interaction sensing under observation error**.

This first phase is documentation-only. Existing scripts remain runnable as historical baselines. Later phases should migrate one function at a time while preserving baseline comparisons.

## Proposed package layout

```text
src/interaction_sensing/
  config/
    schema.py                 # validated run, camera, target, and model config

  targets/
    manual.py                 # user-defined target / zones
    detector.py               # optional learned target proposal
    selector.py               # target identity selection and persistence
    tracker.py                # target-relative motion / deformation tracking
    zones.py                  # context, contact, and access zones

  sensing/
    frames.py                 # desktop / Picamera2 frame-source adapters
    stabilise.py              # camera and target-relative transforms
    motion.py                 # baseline MOG2 and candidate extraction
    candidates.py             # blob geometry and candidate scoring
    scene_state.py            # wind, light, target-motion, density covariates

  interaction/
    states.py                 # approach -> entry -> contact -> access -> departure
    tracks.py                 # actor tracks and multi-actor logic
    attribution.py            # focal versus neighbour target assignment
    segment.py                # event start / continuation / stop rules

  verification/
    objectness.py             # actor/non-actor/unknown verifier
    recognition.py            # optional guild or taxon recogniser
    calibration.py            # threshold / score calibration

  capture/
    ring_buffer.py            # pre-event video buffer
    recorder.py               # clip writer and adaptive quality policies
    audit_sampler.py          # random high-quality audit clips

  data/
    events.py                 # event ledger schema and writers
    clips.py                  # file layout and immutable IDs
    annotations.py            # manual truth / error labels

  evaluation/
    matching.py               # system-to-truth event matching
    errors.py                 # FP, FN, wrong-target, split, merge calculations
    ablation.py               # compare sensing pipelines on same clips
    blocked_splits.py         # site/camera/target/day-held-out evaluation
    observability.py          # conditional error surfaces and correction models

  plugins/
    cirsium_yolo.py           # current flower detector adapter
    insect3_tflite.py         # current 3-guild classifier adapter

  cli/
    run.py
    audit.py
    evaluate.py
```

## Interfaces that must not depend on taxon

### Target specification

```python
TargetSpec(
    target_id,
    target_type,
    bbox_or_polygon,
    context_zone,
    contact_zone,
    access_zone,
)
```

A target may come from a human click, polygon drawing, marker, or detector. All later functions receive the same `TargetSpec`.

### Scene state

```python
SceneState(
    timestamp,
    target_motion_score,
    camera_motion_score,
    illumination_score,
    glare_score,
    rainfall_score,
    target_density,
    occlusion_score,
)
```

Scene state must be written for every time block, including blocks with no candidate event.

### Candidate

```python
Candidate(
    candidate_id,
    timestamp,
    bbox,
    relative_motion_score,
    objectness_score,
    track_id,
)
```

A candidate is not yet a visitor and not yet a focal interaction.

### Interaction event

```python
InteractionEvent(
    event_id,
    target_id,
    actor_track_id,
    start_time,
    end_time,
    max_state,
    attribution_score,
    verification_score,
    clip_id,
    model_version,
)
```

### Audit record

```python
AuditRecord(
    audit_id,
    clip_id,
    sampling_probability,
    truth_event_count,
    truth_target_ids,
    error_classes,
    reviewer,
)
```

## Non-breaking migration order

### Phase 0 — document and freeze baselines

- Keep existing desktop and Pi scripts runnable.
- Record their exact settings in versioned config files.
- Treat these as ablation baselines, not discarded prototypes.

### Phase 1 — separate reusable logic

Extract functions without changing behavior:

```text
detect target
expand ROI
validate target geometry
stable-target state
MOG2 candidate extraction
connected-component filtering
candidate crop expansion
video start / stop
frame-source adapters
```

### Phase 2 — create structured event logging

Add a ledger before altering model logic. Each clip must have a target ID, source frame times, parameters, trigger path, and model versions.

### Phase 3 — add audit sampling and truth matching

Implement random clips and a review table. Report errors for the three current baselines:

```text
motion only
motion -> YOLO detector
motion -> 3-guild classifier
```

### Phase 4 — introduce target-relative sensing

Add target tracking and local stabilisation. Compare it directly to the fixed-ROI MOG2 baseline under wind and target displacement.

### Phase 5 — introduce interaction attribution

Model whether candidates enter the context, contact, and access zones of the focal target versus a neighbour. Do not require taxon identification.

### Phase 6 — conditional observability model

Fit detection and wrong-target attribution probabilities as functions of scene state. Report corrected target-level interaction estimates and uncertainty.

## Deprecated naming

The following names describe a temporary organism or implementation rather than the function:

```text
flower
cirsium
insect3
record_pi
```

They should remain in `plugins/` or `legacy/`, not in core module names. Core names should use:

```text
target
actor
candidate
interaction
audit
attribution
observability
```

## Success criterion

A new plant, object, or ecological target should be usable by defining a target and zones, without retraining the core sensing and event functions. Optional recognition models may improve interpretation but must not determine whether a target-level interaction event exists.