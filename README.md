# interaction-sensing

## Purpose

`interaction-sensing` is an experimental codebase for **error-aware ecological interaction sensing**.

The central question is not only whether a camera can identify a flower or an insect. In complex natural scenes, many objects move, many potential targets co-occur, and an automated system can:

- miss a real interaction,
- record a non-interaction as an interaction,
- assign a visitor to the wrong target,
- split one interaction into multiple events, or
- merge multiple visitors into one event.

This repository is being reorganized around the functions needed to measure, explain, and eventually correct those errors.

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

The first nine functions are the core method. Flower detection, Cirsium-specific models, and three-class insect recognition are optional plugins that currently serve as prototypes.

## What exists today

The legacy scripts already contain working prototypes for:

- detecting a focal flower and expanding it to a local region of interest;
- periodically refreshing and stabilising a target detection;
- detecting motion only inside a target region;
- filtering small motion components and extracting moving candidate boxes;
- cascading motion triggers into a heavier verifier;
- recording only around detected candidate events;
- running the system both on desktop OpenCV video input and Raspberry Pi Picamera2;
- preparing iNaturalist image data, training classifiers, evaluating them, and converting them to TFLite.

The current implementation is intentionally retained as a legacy baseline for future ablation experiments:

```text
motion only
motion -> detector
motion -> classifier
```

## Research direction

The method will be evaluated by its ability to recover ecological interaction estimates, not only by image-level accuracy. Key outputs include:

- event recall and false-event rate;
- wrong-target attribution rate;
- duplicate / merged-event rate;
- how error changes with wind, light, target motion, target density, overlap, and background complexity;
- whether corrected target-level interaction rates reproduce conclusions from continuous-video ground truth.

## Repository map

- `docs/FUNCTION_INVENTORY.md` — existing functions, their current scripts, and their role in the future system.
- `docs/ERROR_TAXONOMY.md` — proposed error classes and the minimum event annotation scheme.
- `docs/TARGET_ARCHITECTURE.md` — non-breaking refactor plan and target package layout.

## Status

This is an active research prototype. The main branch contains historical experiment scripts. The `refactor/interaction-sensing-architecture` branch first documents the function-first architecture without changing runtime behavior.