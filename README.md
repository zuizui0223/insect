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
- target assignment that retains ambiguous neighbouring targets rather than forcing a false certainty;
- interaction-event segmentation and a pre-event ring buffer;
- independent random audit sampling;
- SQLite event ledger;
- target/time-aware audit matching, error summaries, and condition-stratified observability summaries.

The legacy scripts are retained as explicit ablation baselines:

```text
motion only
motion -> detector
motion -> classifier
```

## Repository map

- `src/interaction_sensing/` — reusable package for targets, candidates, interaction events, audit capture, ledgers, and evaluation.
- `analysis/` — audit matching and condition-specific observability analysis skeleton.
- `configs/baselines/` — versioned settings for the historical ablation pipelines.
- `legacy/` — original prototype scripts, now organised as runtime, target-detection, recognition, and data utilities.
- `docs/FUNCTION_INVENTORY.md` — current functions and their role in the new system.
- `docs/ERROR_TAXONOMY.md` — error classes and minimum audit-annotation fields.
- `docs/TARGET_ARCHITECTURE.md` — migration plan and target package layout.

## Quick start

```bash
python -m pip install -e ".[runtime,analysis,dev]"
pytest
```

## Research direction

The method will be evaluated by its ability to recover ecological interaction estimates, not only image-level accuracy. Key outputs include:

- event recall and false-event rate;
- wrong-target attribution rate;
- duplicate / merged-event rate;
- how error changes with wind, light, target motion, target density, overlap, and background complexity;
- whether corrected target-level interaction rates reproduce conclusions from continuous-video ground truth.

## Status

This is an active research prototype. The `refactor/interaction-sensing-architecture` branch contains the initial package and analysis skeleton; the default branch still contains the historical flat-script layout until this refactor is reviewed and merged.
