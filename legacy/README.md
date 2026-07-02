# Legacy scripts

These scripts are preserved exactly as the historical prototypes that motivated
`interaction_sensing`. They are **baseline implementations**, not abandoned
work.

## Runtime baselines

| Legacy path | Baseline role |
|---|---|
| `runtime/flower_roi_live.py` | target localisation visual check |
| `runtime/flower_roi_motion_record.py` | motion-only event capture |
| `runtime/flower_motion_insect_record.py` | motion -> detector -> capture |
| `runtime/flower_motion_insect3_record.py` | motion -> 3-group classifier -> capture (desktop) |
| `runtime/flower_motion_insect3_record_pi.py` | same cascade on Raspberry Pi / TFLite |
| `runtime/detect_flower_roi.py` | single-image target ROI check |
| `runtime/visit_event_detector_mobilenetv3_tflite.py` | flower YOLO -> ROI motion -> MobileNetV3 TFLite classifier -> record (Pi field variant) |

## Model and data prototypes

The remaining files preserve Cirsium target detection, iNaturalist ingestion,
image preprocessing, CNN training, TFLite conversion, and evaluation. They are
useful as optional target/actor-recognition plugins but do not define the new
core method.

| Legacy path | Baseline role |
|---|---|
| `recognition/train_insect3_mobilenetv3.py` | trains the 3-class MobileNetV3 insect classifier from a pre-downloaded iNaturalist manifest |

The Cirsium YOLO weights these scripts expect at local paths (`best.pt`,
`last.pt`, `best.onnx`) are checked in under `models/`, see `models/README.md`.

## Important

Legacy scripts retain absolute local paths and organism-specific names. Do not
use them as the public API. New experiments should use the package under
`src/interaction_sensing/` and register their exact legacy-style settings as a
named pipeline configuration.
