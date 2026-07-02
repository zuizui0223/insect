# interaction-sensing

## Purpose

`interaction-sensing` is an experimental codebase for **error-aware ecological interaction sensing**.

The central question is not only whether a camera can identify a flower or an insect. In complex natural scenes, many objects move, many potential targets co-occur, and an automated system can:

- miss a real interaction,
- record a non-interaction as an interaction,
- assign a visitor to the wrong target,
- split one interaction into multiple events, or
- merge multiple visitors into one event.

This repository is organised around the functions needed to measure, explain, and eventually correct those errors.

## Core principle

The unit of observation is a **target--actor interaction event**:

> an actor enters, contacts, remains near, or accesses a researcher-defined target.

A target can be a flower, inflorescence, fruit, leaf, nest entrance, bait station, or another focal biological structure. Actor taxon identification is optional rather than required for the core event record.

## Function-first architecture

```text
1. target specification
2. target localisation or tracking
3. local coordinate stabilisation
4. relative-motion candidate extraction
5. interaction-zone state estimation
6. event segmentation and adaptive recording
7. random audit recording
8. manual annotation and error taxonomy
9. detection / attribution error modelling
10. optional actor guild or taxon recognition
```

The first nine functions are the core method. Flower detection, Cirsium-specific models, and three-class insect recognition are optional plugins retained as prototypes.

## What is implemented now

The reusable package provides the first, deliberately small backbone:

- taxon-agnostic target, candidate, scene-state, event, and audit-record data contracts;
- manual target and nested-zone specification;
- local MOG2 motion extraction as a reproducible baseline;
- target-relative motion primitives;
- target assignment that retains ambiguous neighbouring targets rather than forcing a false certainty;
- interaction-event segmentation and a pre-event ring buffer;
- independent random audit sampling;
- SQLite event ledger;
- target/time-aware audit matching, error summaries, and condition-stratified observability summaries;
- a runnable motion-only baseline CLI that writes the new ledger format;
- a truth-labelled synthetic benchmark that stress-tests the method before fieldwork.

The legacy scripts are retained as explicit ablation baselines:

```text
motion only
motion -> detector
motion -> classifier
```

## Repository map

- `src/interaction_sensing/` — reusable package for targets, candidates, interaction events, audit capture, ledgers, simulation, and evaluation.
- `analysis/` — audit matching and condition-specific observability analysis skeleton.
- `configs/baselines/` — versioned settings for the historical ablation pipelines.
- `legacy/` — original prototype scripts, now organised as runtime, target-detection, recognition, and data utilities.
- `docs/PREFIELD_VALIDATION.md` — the claim ladder and pre-field benchmark protocol.
- `docs/FUNCTION_INVENTORY.md` — current functions and their role in the new system.
- `docs/ERROR_TAXONOMY.md` — error classes and minimum audit-annotation fields.
- `docs/TARGET_ARCHITECTURE.md` — migration plan and target package layout.

## Quick start

```bash
python -m pip install -e ".[runtime,analysis,dev]"
pytest
```

## Run the motion-only baseline

Use a manually defined target box. The runner does not need flower or insect models.

```bash
interaction-motion-baseline \
  --source path/to/video.mp4 \
  --target-id flower_001 \
  --target-type flower \
  --target-bbox 420 180 720 520 \
  --access-bbox 520 260 620 360 \
  --ledger runs/motion_baseline/events.sqlite \
  --clips-dir runs/motion_baseline/clips \
  --write-clips \
  --audit-probability 0.05 \
  --audit-window-seconds 60
```

The output is not a biological conclusion yet. It is a structured record of:

- target metadata;
- motion-triggered candidate interaction events;
- event clips, if requested;
- random audit clips, if requested;
- an SQLite ledger for later human truth matching and error analysis.

## Run the pre-field synthetic benchmark

The benchmark has complete ground truth and compares a fixed-context baseline
against a target-relative, multi-target, ambiguity-preserving policy. It tests
wind-driven target motion, neighbouring targets, pass-bys, shadows, tracker
error, detection misses, and audit correction.

```bash
# Fast smoke test
interaction-sim-benchmark --quick --output-dir runs/synthetic_quick

# Default factorial benchmark
interaction-sim-benchmark --output-dir runs/synthetic_benchmark
```

Every run writes:

```text
scenario_metrics.csv    # replicate-level outcomes
benchmark_summary.csv   # policy × scenario means
benchmark_report.md     # human-readable result table
assumptions.json        # complete simulation settings
```

A synthetic win is a **mechanistic result**, not field validation. The method
is useful only when its predicted advantage persists under non-zero tracker
error and later transfers to controlled and field video. See
`docs/PREFIELD_VALIDATION.md` for the evidence ladder.

## Research direction

The method will be evaluated by its ability to recover ecological interaction estimates, not only image-level accuracy. Key outputs include:

- event recall and false-event rate;
- wrong-target attribution rate;
- duplicate / merged-event rate;
- how error changes with wind, light, target motion, target density, overlap, and background complexity;
- whether corrected target-level interaction rates reproduce conclusions from continuous-video ground truth.

## Status

This is an active research prototype. The `feature/synthetic-observability-benchmark` branch adds the first pre-field benchmark for the core methodological claim; the default branch contains the runnable motion baseline and organised legacy layout.
