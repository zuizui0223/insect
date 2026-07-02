# Noise-first ecological sensing

## Central inversion

The method does **not** begin with:

```text
find a flower
find an insect
count an interaction
remove nuisance noise afterwards
```

It begins with:

```text
characterise the noise field
estimate which error mechanism is active
quantify observability and attribution risk
preserve uncertainty and collect audit evidence
only then interpret optional biological observations
```

Flowers, insects, nest entrances, fruit, leaves, camera traps, and other focal
objects are applications. They are not the centre of the method.

## The actual research question

> Which natural-scene noise processes create false ecological observations,
> missed observations, or false causal attribution; can a low-power edge camera
> identify those processes early enough to adapt recording and produce a
> calibrated observation-quality record?

The primary output is therefore an **observability record**, not a species label
or visit count.

```text
window_id
  scene noise source
  confidence
  false-event risk
  missed-event risk
  attribution risk
  observability state
  audit-capture decision
  high-resolution-context decision
```

## Noise taxonomy

The method treats the following as first-class empirical variables:

```text
stable_scene
global_camera_shake
co_moving_foreground
background_vegetation_motion
illumination_transient
shadow_transient
occlusion
blur_or_focus_loss
lens_contamination
multi_object_clutter
unknown
```

These are not discarded as preprocessing residues. They are the phenomena that
make automated ecological inference unreliable.

## Why IMX500 belongs here

The Raspberry Pi AI Camera provides an on-sensor neural-network path. In this
project, its primary role is a low-power **noise-state / scene-quality monitor**
that can operate continuously without making the Raspberry Pi run a heavy model
on every frame.

The IMX500 model should first classify or score broad scene conditions such as:

```text
clean / stable
photometric disturbance
clutter / overlap
occlusion
focus or lens degradation
unknown / out-of-distribution
```

The Pi complements that snapshot-level model with temporal quantities that need
frame history:

```text
global frame displacement
coherent foreground displacement
local relative motion
illumination change
blur / focus proxy
occlusion proxy
```

The fusion gives a noise source and risks. The system then decides whether to:

```text
continue low-cost monitoring
record high-resolution context
sample an independent audit clip
mark a time window as unobservable
```

## No silent deletion

A noisy interval must not simply disappear. That would produce precisely the
selection bias this method is designed to expose.

```text
clean                -> normal observation
confounded           -> retain observation plus noise state
high-risk            -> retain observation and prioritise audit/context capture
unobservable         -> record condition, do not make a strong biological claim
unknown              -> retain and audit rather than force a class
```

## Where flowers and insects return

Later, flower--visitor scenes provide a demanding application because all major
noise mechanisms occur together: wind, flower sway, dense displays, shadows,
small moving organisms, and occlusion.

The application question is not merely:

> Did the camera detect a visitor?

It is:

> Does the relationship between a biological treatment and an observed visitor
> rate remain after accounting for a noise field that may change detectability,
> false-event rate, and target attribution?

## Evidence ladder

### Before fieldwork

1. Create labelled noise perturbation datasets with no biological target
   required: camera shake, artificial foliage sway, shadows, glare, blur,
   occlusion, clutter, lens droplets.
2. Test whether IMX500 + host temporal features recover noise source and risk.
3. Test whether adaptive audit selection finds high-risk windows better than
   uniform recording at the same storage/power budget.
4. Demonstrate that a downstream event counter becomes biased when noise is
   ignored and that the noise record predicts the bias.

### Controlled biological video

Use flowers and moving surrogates or real visitors only to test whether a
noise-aware observation quality record predicts false/missed/ambiguous events.

### Field deployment

Use audit clips to fit condition-specific error models and evaluate whether
noise-adjusted ecological estimates change conclusions relative to raw camera
counts.
