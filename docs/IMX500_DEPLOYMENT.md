# IMX500 deployment path

## Role in this project

The Raspberry Pi AI Camera (Sony IMX500) is a **noise-state and observability
sensor**, not a flower detector, insect detector, or pollinator counter.

```text
IMX500 scene-noise / quality model
  + Pi temporal measurements
  -> noise source and observability risks
  -> adaptive context capture and independent audit
  -> later, optional ecological observation corrected for those risks
```

The first output is a record of *whether and why the scene is measurable*, not
what species is present.

## What the sensor should help measure

The IMX500 is well suited to continuous low-power scores for image conditions
that can be recognised from the current scene. The host complements it with
frame-history features.

```text
IMX500 model candidates
  clean / stable scene
  photometric disturbance
  occlusion
  image degradation
  scene clutter / overlap
  lens contamination
  unknown / out-of-distribution

Pi temporal measurements
  global camera displacement
  coherent foreground motion
  local relative motion
  illumination change
  blur / focus proxy
```

The combined record is saved as a `NoiseObservation`, then converted into:

```text
false-event risk
missed-event risk
attribution risk
observability state
whether to save high-resolution context
whether to prioritise an independent audit clip
```

## No silent filtering

A high-noise frame must not be deleted and forgotten. It is evidence about the
observation process.

```text
clean          -> ordinary low-cost recording
confounded     -> retain noise metadata with downstream observations
audit_priority -> retain metadata and capture audit/context video
unobservable   -> record the condition; do not force a biological conclusion
unknown        -> preserve uncertainty and audit
```

## Current hardware adapter

```text
src/interaction_sensing/noise.py
  NoiseSource               explicit noise taxonomy
  NoiseObservation          per-frame / per-window noise record
  NoiseFirstPolicy          transparent risk and capture policy

src/interaction_sensing/plugins/imx500.py
  SensorROI                 optional fixed context crop in full-sensor pixels
  IMX500Runtime             Picamera2 lifecycle and output metadata capture
  InferenceNDJSONLogger     complete sensor log
  SSDDetectionDecoder       optional hardware / object-proposal compatibility

interaction-imx500-probe
  hardware and metadata probe, not ecological inference
```

`TargetAwareROIController` remains only for controlled ablations. It is not the
default method because moving the sensor crop to follow a presumed target can
hide the broader clutter, sway, and neighbouring-object processes that this
method aims to measure.

## Model roles

The code now records one of these roles:

```text
noise_state
observability_quality
scene_embedding
object_proposal        optional ablation only
experimental
```

The first custom IMX500 model should be a compact **noise-state / image-quality
classifier**, not a flower or insect model. Object proposal may later be added
only to test whether downstream biological measurements improve after the noise
field is known.

## Pi setup and hardware check

On the Pi, update the system and install IMX500 firmware/models:

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all
sudo reboot
```

Install this repository in editable mode:

```bash
cd ~/insect
python -m pip install -e ".[dev]"
```

Use the supplied generic SSD package only to check firmware loading, output
metadata, full-sensor coordinate conversion, NDJSON logging, and sensor ROI
behaviour. Its generic object labels are not scientific observations:

```bash
interaction-imx500-probe \
  --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk \
  --model-role experimental \
  --frames 300 \
  --output runs/imx500/hardware_probe.ndjson
```

An optional fixed scene context uses **full-sensor** pixels:

```bash
interaction-imx500-probe \
  --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk \
  --sensor-roi 900 600 1800 1400 \
  --frames 300 \
  --output runs/imx500/context_probe.ndjson
```

The standard full sensor size is 4056 × 3040; verify the actual size on the
running system before fixing a field configuration.

## First pre-field data collection should be noise-only

Collect controlled clips with no biological target requirement:

```text
camera wobble
foreground / vegetation sway
moving shadows
rapid sun–cloud brightness changes
blur and defocus
partial occlusion
water or dust on lens
single object versus cluttered overlapping objects
```

For every clip retain the known perturbation label, IMX500 output/KPI, host-side
temporal features, storage/power use, and whether the noise-first policy chose
audit or high-resolution context capture.

This dataset directly tests the method's claim: can the camera describe the
conditions under which automatic ecological observations become unreliable?

## What object models are for

The legacy Cirsium IMX export and any later insect/object detector are optional
stress-test instruments. They can be useful **after** the noise field exists,
for testing questions such as:

```text
Does a detector's false-event rate increase in clutter, shake, or shadow?
Does a noise-aware policy flag those failures before an ecological comparison is made?
Do raw biological counts change after excluding or modelling unobservable windows?
```

They are not the core method and should not define the first custom IMX500
model.
