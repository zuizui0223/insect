# Methodological contribution

## The problem this project addresses

Most automated flower--visitor systems begin with a recognition question:

```text
Can the model find an insect?
Can it identify a taxon or guild?
Can it count detections cheaply?
```

Those are necessary capabilities, but they do not resolve the observation
problem in a dense natural scene. A system may identify an insect correctly and
still make the wrong ecological statement because it:

- assigns an actor to the wrong flower;
- treats wind-driven flower movement as a visitor;
- treats a pass-by as target use;
- silently misses interaction events in conditions correlated with a focal
  treatment or morphology;
- reports raw detections as though they were true interaction counts.

## Proposed contribution

`interaction-sensing` defines the observation unit as a **target--actor
interaction event**, not a detected taxon.

```text
target specification
  -> target-relative candidate motion
  -> competing-target attribution
  -> nested interaction states
  -> adaptive capture
  -> random audit clips
  -> error-aware interaction estimate
```

The core can operate without a target-specific detector or actor classifier at
deployment. Recognition is an optional annotation layer, not the criterion that
creates an interaction event.

## Four testable design claims

### 1. Target-relative motion is more informative than fixed image motion

When a target moves due to wind or camera displacement, raw image motion is not
evidence of a visitor. Candidate motion should be interpreted relative to an
estimated target displacement.

**Expected failure mode:** the benefit declines when target tracking becomes
noisy. The benchmark must show this degradation rather than hide it.

### 2. Explicit competing targets reduce wrong-target attribution

In a multi-flower display, the closest detected movement is not automatically
an event on the chosen focal flower. The system should assign candidates across
all visible targets and preserve ambiguity when assignment is not defensible.

**Expected failure mode:** at severe target overlap, the policy should emit more
`ambiguous_target` records. A lower count is not automatically better when it
comes from forced assignment.

### 3. Interaction zones are more interpretable than a single ROI

A broad context zone can record approach; a core zone can record contact proxy;
an access zone can record a stronger entry proxy. The system should retain the
state rather than collapse every movement into a visit.

**Expected failure mode:** a poorly placed manual zone reduces recall. Zone
setup therefore belongs in calibration and audit, not hidden configuration.

### 4. Audits convert trigger errors into estimable uncertainty

Trigger-selected clips cannot reveal events that were missed. Randomly sampled
clips are necessary to estimate detection and false-positive probabilities.
Raw counts and audit-adjusted estimates must remain distinct outputs.

**Expected failure mode:** with too few audit windows, correction is unstable or
undefined. The system should return uncertainty / NA rather than a falsely
precise corrected count.

## What the synthetic benchmark can establish

The benchmark can show that, under known truth and explicit error mechanisms,
the architecture behaves as designed:

```text
wind ↑             -> fixed-context plant-motion false events ↑
neighbour distance ↓ -> fixed-context wrong-target attribution ↑
tracking error ↑   -> target-relative advantage may diminish
random audits      -> raw count bias can be diagnosed and partly corrected
```

This is a methodological, causal stress test. It is not evidence that the
system already works in arbitrary field scenes.

## What will make the method persuasive

A strong pre-field package should provide all of the following:

1. reproducible scenario definitions and random seeds;
2. complete latent truth, not only final metric plots;
3. a baseline that represents common fixed-ROI practice;
4. ablations that remove target motion compensation, multi-target attribution,
   or audits one at a time;
5. deliberate tracker-error and overlap stress tests;
6. outputs expressed as interaction-estimate bias, not only classifier F1;
7. a direct bridge from synthetic data to controlled video and then field audit.

The methodological paper is therefore not:

> We built a better flower/insect detector.

It is:

> We specify and validate an observation process that makes automated
> interaction estimates auditable under target motion, competing targets, and
> domain shift.
