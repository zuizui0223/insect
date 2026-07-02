# Target--actor attribution error taxonomy

## Why this taxonomy exists

A vision system can correctly recognise an insect while still making the wrong ecological claim. The central record must therefore distinguish **visual detection** from **attribution of an interaction to a focal target**.

This document defines labels for continuous-video audit annotation and for later error modelling.

## The event truth table

Each candidate is annotated on two axes.

### Axis 1: Was there an actor?

- `actor_present`
- `no_actor`
- `unknown_actor_presence`

### Axis 2: What was its relation to the focal target?

- `outside_target_context`
- `approach`
- `context_entry`
- `target_contact`
- `access_zone_entry`
- `interaction_uncertain`

An event cannot be called a focal interaction solely because an actor was detected in the image.

## Error classes

| Code | Name | Definition | Consequence if ignored |
|---|---|---|---|
| `FP_MOTION` | plant / scene-motion false event | Wind, rain, shadow, glare, or camera motion triggers an event without an actor | Inflates activity estimates under noisy conditions |
| `FP_NONINTERACTION` | actor-present non-interaction | An actor is present but does not enter the focal interaction zone | Treats fly-bys or nearby resting as focal use |
| `FP_WRONG_TARGET` | wrong-target attribution | Actor interacts with a neighbouring target but is assigned to the focal one | Biases target-level comparisons, especially in dense displays |
| `FN_MISSED` | missed true interaction | True focal interaction occurs but no event is recorded | Underestimates activity, possibly condition-dependently |
| `FN_OCCLUDED` | occluded true interaction | Interaction is visible in principle but hidden by target structure, neighbours, or another actor | Biases morphology or density comparisons |
| `SPLIT` | one true event split into multiple records | System stops and restarts within one continuous interaction | Inflates event counts and reduces estimated duration |
| `MERGE` | multiple true events merged into one record | Multiple arrivals or actors are recorded as a single event | Underestimates event number and visitor turnover |
| `ID_SWAP` | actor identity exchange | A tracked actor is confused with another actor | Biases trajectories and duration estimates |
| `TARGET_DRIFT` | target coordinate error | Target region fails to follow wind-driven movement or re-detection changes identity | Converts target motion into visitor motion or changes focal target |
| `STATE_ERROR` | interaction-state error | Approach, contact, and access are confused | Biases behavioural mechanism inference |
| `UNKNOWN` | non-resolvable clip | Clip quality prevents a defensible label | Must remain explicit rather than forced into a biological class |

## Minimum manual annotation fields

```text
clip_id
camera_id
site_id
target_id
target_type
datetime_start
datetime_end
truth_actor_present
truth_interaction_state
truth_target_id
n_actors
n_true_events
system_event_id
system_target_id
system_state
error_class
visibility
occlusion_score
wind_state
illumination_state
target_motion_score
target_density
reviewer_id
annotation_version
```

## Scene-state covariates

The following variables are expected to influence both detection and attribution.

- wind and target displacement;
- illumination, shadow movement, glare, and time of day;
- rainfall and droplets;
- camera vibration;
- target size, orientation, and visual contrast;
- target density, overlap, and background complexity;
- actor size, speed, and approach direction;
- simultaneous actors;
- target occlusion.

These covariates should be retained even when no event is detected, because they determine whether apparent biological differences could be observation differences.

## Core estimands

For a target `i` in time block `t`, distinguish:

```text
N_it = true focal interactions
p_it = probability a true interaction is detected and attributed correctly
phi_it = rate of false focal events
Y_it = observed focal events
```

The project should report all three observable error components:

```text
false-event rate
miss rate
wrong-target attribution rate
```

The eventual ecological quantity is not raw `Y_it`, but a corrected estimate of `N_it` with uncertainty.

## Annotation priority

Start with a small but high-quality audit dataset. The first labels need not include species names. The essential distinction is:

```text
no actor
actor nearby only
actor on the wrong target
actor interacting with focal target
unresolvable
```

Taxon or guild labels can be added later where image quality supports them.