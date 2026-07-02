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

## Implemented baseline: manual target + motion-only capture

The first runnable baseline connects the prior motion detector to the new research architecture:

```text
manual target + nested zones
  -> fixed local ROI
  -> MOG2 motion candidates
  -> geometric interaction state
  -> event segmentation + raw pre-event clip
  -> SQLite event ledger
  -> independent random raw audit clips
  -> later truth matching and observability analysis
```

A motion event is **not** treated as proof of an ecological interaction. Its event state, raw clip, target ID, scene state, and capture path are stored so that it can later be audited as true focal interaction, near-target pass, wrong-target interaction, plant-motion false event, missed event, split, merge, or unknown.

### Quick start

```bash
python -m pip install -e ".[runtime,dev]"

interaction-motion-only \
  --source path/to/video.mp4 \
  --output-dir runs/demo_001 \
  --target-id flower_001 \
  --target-type flower \
  --core-zone 420,180,220,220 \
  --access-zone 470,230,100,100 \
  --display
```

`--core-zone` and optional `--access-zone` use `x,y,width,height` pixel coordinates from the first frame. The wider context zone is created automatically using `configs/baselines/motion_only.toml`.

This produces:

```text
runs/demo_001/
  run_manifest.json
  events.sqlite
  events/                 # raw clips started by the motion baseline
  audits/                 # raw random clips sampled independently of triggers
```

Export the ledger before annotation or analysis:

```bash
python analysis/00_export_ledger.py \
  --ledger runs/demo_001/events.sqlite \
  --output-dir runs/demo_001/exports
```

## Legacy ablation baselines

The historical scripts are retained as explicit comparison conditions:

```text
motion only
motion -> detector
motion -> classifier
```

They live under `legacy/` and are not the public API.

## Repository map

- `src/interaction_sensing/` — reusable package for targets, candidates, interaction events, audit capture, ledgers, and evaluation.
- `analysis/` — ledger export, audit matching, and condition-specific observability analysis.
- `configs/baselines/` — versioned settings for the historical ablation pipelines.
- `legacy/` — original prototype scripts, organised as runtime, target-detection, recognition, and data utilities.
- `docs/FUNCTION_INVENTORY.md` — current functions and their role in the new system.
- `docs/ERROR_TAXONOMY.md` — error classes and minimum audit-annotation fields.
- `docs/TARGET_ARCHITECTURE.md` — migration plan and target package layout.

## Research direction

The method will be evaluated by its ability to recover ecological interaction estimates, not only image-level accuracy. Key outputs include:

- event recall and false-event rate;
- wrong-target attribution rate;
- duplicate / merged-event rate;
- how error changes with wind, light, target motion, target density, overlap, and background complexity;
- whether corrected target-level interaction rates reproduce conclusions from continuous-video ground truth.

## Status

The manual-target motion-only baseline is runnable. It deliberately uses a fixed target coordinate system, so wind-driven target movement remains a measurable error source rather than a solved problem. The next technical comparison is target-relative stabilisation versus this fixed-ROI baseline under controlled and field wind conditions.
