# Analysis skeleton

The analysis is deliberately organised around **observation error**, not model accuracy alone.

## Runtime-to-analysis workflow

```text
1. Run `interaction-motion-only`.
   -> raw event clips, raw random audit clips, events.sqlite, run_manifest.json

2. Export the ledger.
   -> python analysis/00_export_ledger.py --ledger RUN/events.sqlite --output-dir RUN/exports

3. Annotate the random audit clips and selected event clips.
   -> truth event count, focal target identity, interaction state, error class

4. Match automatic and truth events.
   -> python analysis/01_validate_audit.py ...

5. Estimate condition-specific observability.
   -> python analysis/02_observability_by_condition.py ...
```

## Required inputs

1. `events.sqlite` — automatic event ledger written by each pipeline.
2. `audit_annotations.csv` — independently sampled clips with human truth labels.
3. `target_metadata.csv` — target identity, target type, morphology or treatment, and spatial context.
4. Optional `scene_covariates.csv` — wind, illumination, target motion, density, occlusion, and weather.

## Analysis order

```text
00_export_ledger.py
  Export events, audits, targets, and scene states from the authoritative SQLite ledger.

01_validate_audit.py
  Match automatic and truth events by target and time.
  Write false event, missed event, wrong target, split, merge, and unknown labels.

02_observability_by_condition.py
  Estimate detection and false-event rates by scene-state strata.
  Export an interaction observability table.
```

## Central quantities

For each target/time block, retain separate estimates for:

```text
N = true focal interactions
p = probability that a true focal interaction is recorded and attributed correctly
phi = false focal-event rate
Y = observed system events
```

Raw `Y` is not the biological endpoint. The later statistical model should estimate a corrected `N` and uncertainty conditional on scene state.

## Blocking principle

Never use a random image-level split to claim ecological generalisation. Hold out at least one of:

```text
site
camera
day / weather period
target individual
target morphology
```

The question is whether the *interaction estimate* transfers, not whether an image classifier can recognise a familiar background.
