# Rendered visual-disturbance benchmark

## Why add rendered frames?

The scalar latent benchmark proves a causal idea under complete control:

```text
local observation = independent signal + shared nuisance + noise
```

That alone does not prove that a camera can recover a useful reference from
pixels. This benchmark introduces a deliberately small but genuine visual
measurement problem:

```text
rendered frame sequence
  -> image-derived global alignment
  -> image-derived local and background changes
  -> reference-guided local residual
  -> held-out truth evaluation
```

No flower, insect, or taxon label is present. The independent local pulse is a
stand-in for any small, local change that downstream biological systems may want
to interpret.

## Rendered mechanisms

Every rendered sequence mixes the following sources:

```text
static textured scene
non-rigid vegetation-like layers with different sway trajectories
moving spatial shadow
persistent global illumination variation
integer-pixel camera displacement
sensor noise
independent bright local pulse
```

The generator keeps the causal state for assessment, but feature extraction does
not read it. In particular, it does not access hidden illumination, sway, or
camera variables to construct a reference.

## Pixel-derived references

For two adjacent frames, the pipeline:

1. estimates camera displacement by searching the integer alignment that best
   matches **background reference regions**;
2. aligns the current frame;
3. measures signed local change in a fixed central region;
4. measures signed changes in four reference regions;
5. uses their median as a robust shared-scene reference.

```text
raw evidence          = local difference before alignment
stabilised evidence   = local difference after image-derived alignment
N0 residual evidence  = stabilised evidence - median(background differences)
```

This deliberately exposes the mechanism to failure. Image alignment can be
wrong; a background reference can be unrepresentative; shadow and non-rigid
motion can violate shared-change assumptions.

## Negative controls

The benchmark does not compare N0 only against raw differencing. It includes
references whose *pixels are visible* but whose causal relation is broken.

```text
time-shifted visual reference
  previous background change rather than contemporaneous change

spatially mismatched visual reference
  a visible calibration panel that moves with camera displacement but does not
  share illumination, shadow, or vegetation-sway mechanisms
```

All policies receive the exact same rendered frames. If either broken reference
matches the robust reference, the claimed shared-scene reference mechanism is
not supported.

## Locked split

Thresholds are calibrated from separately seeded calibration worlds at a fixed
true-local-event recall. Results are reported only from held-out rendered worlds.

```bash
interaction-visual-benchmark --output-dir runs/visual_disturbance_benchmark
```

Outputs:

```text
visual_disturbance_metrics.csv
visual_disturbance_summary.csv
visual_disturbance_config.json
visual_calibration_thresholds.json
visual_disturbance_report.md
```

## Claim boundary

Passing this benchmark would support:

> Under rendered natural-scene-like disturbances, image-derived shared-scene
> references reduce spurious local visual events relative to raw pixel
> differencing, and this advantage is lost with temporally or spatially invalid
> references.

It does not support:

```text
field generalisation
wind detection from arbitrary natural video
superiority over all optical-flow methods
biological event accuracy
pollinator or interaction inference
```

The next evidence layer is controlled physical NoiseBench video, where the same
features are extracted from real camera frames rather than renderer output.
