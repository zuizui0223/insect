# Repository overview

A one-page map of `interaction-sensing`, for orientation before reading any
single document in detail. See the root `README.md` for the full purpose
statement and quick start.

## What this repository is

`interaction-sensing` studies whether an autonomous edge camera can recognise
*when and why* its own observations of ecological interactions (e.g.
flower-visitor events) are unreliable, preserve that uncertainty as data, and
use it to drive adaptive auditing — rather than starting from "detect the
target, count the event" and treating scene noise as a nuisance to filter out.
Target/organism recognition (Cirsium flowers, insect classes) is a downstream
application of the method, not its core object.

## Directory map

| Path | Contents |
|---|---|
| `src/interaction_sensing/` | The current package: noise/observability contracts, NoiseBench protocol generator, target/event/audit/ledger contracts, simulation and evaluation utilities, CLI entry points, optional IMX500 plugin. |
| `tests/` | pytest suite covering the package above (37 tests as of this writing). |
| `configs/` | Named `.toml` pipeline configurations for baselines and NoiseBench. |
| `analysis/` | Post-hoc scripts that turn a recorded event ledger + audit annotations into observability/error estimates. See `analysis/README.md`. |
| `docs/` | Design and methodology documents (index below). |
| `legacy/` | Original prototype scripts (runtime cascades, target detection training/export, classifier training, data prep), preserved as baselines — not the public API. See `legacy/README.md`. |
| `models/` | Trained weights (`best.pt`, `last.pt`, `best.onnx`) for the legacy Cirsium YOLO target-detection baseline. See `models/README.md`. |

## Entry points

- Install: `python -m pip install -e ".[runtime,analysis,dev]"`
- Test: `pytest`
- CLI scripts (registered in `pyproject.toml`): `interaction-motion-baseline`,
  `interaction-sim-benchmark`, `interaction-imx500-probe`,
  `interaction-noisebench-plan`, `interaction-latent-benchmark`,
  `interaction-visual-benchmark`, `interaction-temporal-risk-benchmark`.

## Docs index

| Doc | Covers |
|---|---|
| `NOISE_FIRST_METHOD.md` | The core noise-first conceptual framework and inversion from target-first pipelines. |
| `NOISEBENCH_PROTOCOL.md` | Controlled-perturbation recording protocol and evaluation endpoints. |
| `IMX500_DEPLOYMENT.md` | Using the Raspberry Pi AI Camera (IMX500) as a scene-observability sensor. |
| `TARGET_ARCHITECTURE.md` | Refactor goal: from experiment-named scripts to a reusable target-actor interaction package. |
| `FUNCTION_INVENTORY.md` | Component-by-component inventory of what each part of the system does, which legacy script it came from, and its future role. |
| `ERROR_TAXONOMY.md` | Taxonomy distinguishing visual detection errors from interaction-attribution errors. |
| `CAUSAL_ATTRIBUTION_CONTRACT.md` | Shared-cause causal attribution contract for the sensing pipeline. |
| `TEMPORAL_NUISANCE_RISK_MODEL.md` | The N1 temporal nuisance-risk learned model. |
| `TEMPORAL_RISK_EVALUATION.md` | Evaluation contract for whether N1 adds information over the baseline. |
| `LATENT_DISTURBANCE_METHOD.md` | Latent-disturbance causal inference benchmark methodology. |
| `LATENT_BENCHMARK_EVALUATION.md` | Evaluation contract for the latent-disturbance benchmark. |
| `VISUAL_DISTURBANCE_BENCHMARK.md` | Rendered visual-disturbance benchmark, extending the scalar latent benchmark. |
| `PREFIELD_VALIDATION.md` | Validation strategy to run before collecting biological field data. |
| `REVIEW_RESILIENT_VALIDATION.md` | Standard for a review-resilient validation without requiring field observations. |
| `FIVE_RIG_DEPLOYMENT_DESIGN.md` | Deployment design across the five physical Raspberry Pi field rigs. |
| `METHODOLOGY_CONTRIBUTION.md` | Statement of the project's methodological contribution vs. prior flower-visitor systems. |
| `LITERATURE_POSITION.md` | Working literature map situating NoiseBench and noise-first sensing. |

## Current status

- Implemented: target-agnostic noise/observability contracts and risk policy,
  the IMX500 hardware adapter, the NoiseBench protocol generator, motion
  primitives, event/audit/ledger contracts, and error-evaluation utilities.
- Retained as explicit baselines, not the public API: the legacy
  motion/detector/classifier scripts and the Cirsium YOLO weights in `models/`.
- The earlier interaction-level synthetic benchmark is now a downstream stress
  test rather than the primary method.
