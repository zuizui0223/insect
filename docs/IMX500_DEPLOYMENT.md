# IMX500 deployment path

## Role in this project

The Raspberry Pi AI Camera (Sony IMX500) is not used as a black-box
"pollinator recogniser". It is the **sensor-side proposal layer** in an
auditable interaction-sensing pipeline:

```text
manual target or target detector
  -> target context in full-sensor coordinates
  -> IMX500 inference ROI
  -> sensor-side model proposals
  -> Pi-side coordinate conversion and target attribution
  -> interaction-zone state / event segmentation
  -> event clips + random audit clips
  -> corrected ecological estimate
```

A neural-network detection is never equivalent to `visit`, `contact`, or
`pollination`.

## Why IMX500 matters

A conventional Raspberry Pi camera sends frames to the Pi, where the Pi must
run neural-network inference. With IMX500, the sensor creates the neural-network
input tensor and runs inference on its on-module accelerator; the Pi receives
inference outputs and performs the necessary host-side post-processing.

That division is valuable for this project because the Pi can spend its compute
budget on what the sensor does **not** solve:

- tracking a focal target through wind and camera disturbance;
- assigning candidate actors to the focal target versus neighbouring targets;
- retaining `ambiguous_target` rather than forced assignment;
- recording pre-event context and independent audit clips;
- storing scene state and uncertainty.

## Three model roles

### 0. Hardware / metadata proof

Use an official pre-packaged SSD `.rpk` only to verify that the camera firmware,
Picamera2 metadata path, coordinate conversion, sensor ROI, and NDJSON logging
work. Its generic labels are **not ecological labels** and must not be used to
claim insect or visit detection.

### 1. Target-proposal model

A flower / inflorescence detector proposes target candidates or reacquires a
target after tracking loss. The legacy Cirsium model can be used only as a
hardware integration prototype. It should remain an optional species-specific
plugin, not the core method.

### 2. Actor-proposal model

The first ecological model should be deliberately broad:

```text
actor_candidate
non_actor_or_background
unknown
```

It should produce a candidate proposal inside the focal target context. It does
not establish target use; the Pi-side geometry and event state machine determine
whether the proposal is focal, neighbour-associated, pass-by, or uncertain.

A future object detector can improve trajectories and multi-actor separation,
but an actor/non-actor/unknown classifier inside a manually supplied target
context is the more realistic first custom IMX500 model when box annotations are
still scarce.

## One inference ROI, many biological targets

The IMX500 inference crop is a sensor-level resource. Treat it as one active
region, not one independent neural network per flower. For a dense display:

```text
1. define one target cluster / context ROI on the sensor;
2. detect or track individual flowers on the Pi-side preview stream;
3. attribute IMX500 actor proposals among those flower targets;
4. save ambiguous cases for review rather than switching the model between
   flowers every frame.
```

Do not continuously swap models in the field. Model firmware loading is a
startup/runtime operation; a field run should normally select one model role and
keep it stable. Test ROI-update latency separately before relying on frequent
sensor-crop updates.

## Current repository components

```text
src/interaction_sensing/plugins/imx500.py
  SensorROI                  absolute full-sensor inference crop
  TargetAwareROIController   safe, change-aware ROI updates
  SSDDetectionDecoder        adapter for standard boxes/scores/classes outputs
  IMX500Runtime              Picamera2 lifecycle and tensor metadata capture
  InferenceNDJSONLogger      complete per-inference sensor record

interaction-imx500-probe
  hardware-in-the-loop test runner
```

## Pi setup and first hardware check

On the Pi, update the system and install the IMX500 firmware/models:

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all
sudo reboot
```

Then install the repository package in editable mode:

```bash
cd ~/interaction-sensing
python -m pip install -e ".[dev]"
```

Run the probe with the supplied generic SSD package only to validate the
hardware/metadata pipeline:

```bash
interaction-imx500-probe \
  --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk \
  --decoder ssd \
  --model-role experimental \
  --frames 300 \
  --output runs/imx500/ssd_probe.ndjson
```

A target-context crop is expressed in **full-sensor** pixels, not preview
pixels:

```bash
interaction-imx500-probe \
  --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk \
  --sensor-roi 900 600 1800 1400 \
  --frames 300 \
  --output runs/imx500/context_probe.ndjson
```

The standard full sensor size is 4056 × 3040. Confirm the actual full sensor
resolution from the running model before hard-coding a field configuration.

## Model packaging: status of the legacy Cirsium export

`legacy/target_detection/export_cirsium_imx.py` exports an Ultralytics model to
an intermediate IMX representation. It is **not** a complete deployment record.
A reproducible IMX500 model release must also retain:

```text
source checkpoint and labels
training-data provenance and split
input size and preprocessing
Edge-MDT conversion version and parameters
final .rpk firmware package
post-processing decoder and confidence threshold
sensor ROI policy
field/audit calibration results
```

The official IMX500 deployment chain is:

```text
PyTorch or TensorFlow model
  -> Edge-MDT quantisation / compression / conversion
  -> final firmware package on Raspberry Pi
  -> .rpk loaded by IMX500 at runtime
```

## What to measure before field deployment

Use `interaction-imx500-probe` to record an NDJSON stream while varying:

- fixed versus manually moved target ROI;
- camera/target sway;
- full sensor versus focal context crop;
- lighting and focus;
- inference confidence threshold;
- model startup time and sensor KPI metadata.

This first dataset is a **hardware-in-the-loop observability dataset**. It can
validate runtime behaviour and power/throughput trade-offs, but it cannot yet
prove ecological accuracy without independently labelled flower--visitor video.
