# Function inventory

This inventory describes the repository by **what each component does**, not by the current focal species or model result.

## A. Target definition and localisation

| Function | Current implementation | Legacy scripts | Keep? | Future role |
|---|---|---|---|---|
| Target detection | YOLO detects the highest-confidence Cirsium flower | `detect_flower_roi.py`, `flower_roi_live.py`, `flower_roi_motion_record.py`, `flower_motion_insect*_record*.py` | Yes | Optional automatic target initialiser |
| Target selection | Highest-confidence candidate is selected | same | Replace | Allow manual target choice, multiple targets, and deterministic target IDs |
| Target region expansion | Bounding box is padded by a fixed ratio | same | Yes | Target context zone, configurable by target type |
| Target stability gate | Detection is accepted only after repeated detections | `flower_motion_insect3_record.py`, `flower_motion_insect3_record_pi.py` | Yes | General target-presence state machine |
| Target-loss gate | Recording is disabled after repeated missed detections | `flower_motion_insect3_record.py`, `flower_motion_insect3_record_pi.py` | Yes | Explicit target visibility / occlusion state |

### Gap

The present target is a fixed image-coordinate ROI refreshed periodically. It does not estimate the target's motion between detector calls, so wind-driven target displacement can become false motion inside the ROI.

## B. Scene stabilisation and motion candidate extraction

| Function | Current implementation | Legacy scripts | Keep? | Future role |
|---|---|---|---|---|
| Local ROI processing | Only the padded target ROI is analysed | `flower_roi_motion_record.py` onward | Yes | Main computational-saving principle |
| Low-resolution gating | ROI is resized before background subtraction | `flower_motion_insect*_record*.py` | Yes | Low-cost candidate proposal stage |
| Background subtraction | OpenCV MOG2 foreground mask | `flower_roi_motion_record.py` onward | Yes, as baseline | Baseline candidate extractor |
| Morphological filtering | Opening / closing reduces isolated mask noise | same | Yes | Baseline noise filter |
| Blob extraction | Connected components yield candidate bounding boxes | `flower_motion_insect*_record*.py` | Yes | Candidate objects for later tracking and annotation |
| Motion burden score | Foreground-pixel ratio is thresholded | same | Yes | Diagnostic feature, not a biological event label |

### Gap

This layer currently detects **image motion**, not motion relative to a moving target. It therefore mixes target sway, neighbouring vegetation, illumination shifts, and visitor motion.

## C. Candidate verification

| Function | Current implementation | Legacy scripts | Keep? | Future role |
|---|---|---|---|---|
| Detector-based verification | A YOLO insect detector runs after a motion trigger | `flower_motion_insect_record.py` | Keep as optional plugin | Objectness / actor verifier |
| Classifier-based verification | Largest motion crop is assigned to one of three insect groups | `flower_motion_insect3_record.py`, `_pi.py` | Keep as optional plugin | Actor guild label, not event definition |
| Confidence thresholding | High-confidence output starts or maintains recording | same | Yes | Calibrated decision threshold, conditional on noise state |
| Brief post-detection tracking | Last verified crop persists for a short interval | `flower_motion_insect3_record_pi.py` | Replace | Multi-object target-relative trajectory tracking |

### Gap

The verifier has no explicit `non-actor`, `wrong-target`, `occluded`, or `unknown` class. A high-confidence taxon prediction therefore does not establish that an actor interacted with the focal target.

## D. Event segmentation and capture

| Function | Current implementation | Legacy scripts | Keep? | Future role |
|---|---|---|---|---|
| Triggered recording | Video starts after motion or verified actor detection | all `*_record*.py` | Yes | Event capture policy |
| Stop rule | Recording ends after a quiet interval | same | Yes | Event end state, evaluated against truth |
| Maximum duration guard | Pi runtime stops events after 60 seconds | `flower_motion_insect3_record_pi.py` | Yes | Operational storage control |
| Desktop camera runtime | OpenCV `VideoCapture` input | desktop scripts | Yes | Local development backend |
| Raspberry Pi runtime | Picamera2 input and TFLite inference | `flower_motion_insect3_record_pi.py` | Yes | Field deployment backend |
| Visual diagnostics | Draw target, ROI, blobs, crops, status text | runtime scripts | Yes | Debug and annotation QA layer |

### Gap

Recording begins after a trigger decision; the present code does not preserve a pre-event buffer. It also saves video filenames but no structured event ledger.

## E. Data and model utilities

| Function | Current implementation | Legacy scripts | Keep? | Future role |
|---|---|---|---|---|
| Public-image download | Resumable iNaturalist download with metadata | `inat_download_resume.py` | Yes | Optional recogniser-data acquisition |
| Image preprocessing | Resize images and map metadata to labels | `local_preprocess.py` | Keep but isolate | Generic image-data ingestion utility |
| Train/test split | Stratified random image split | `local_split_prepare.py`, `make_insect3_dataset.py` | Replace for ecological evaluation | Site / camera / target / day blocked split |
| Data augmentation | Geometric image augmentation | `local_augment_prepare.py`, training scripts | Keep but revise | Noise-aware augmentation library |
| Classifier training | Keras CNN training | `local_train_cnn.py`, `train_insect3_classifier.py` | Optional plugin | Actor guild / taxon recogniser |
| Model conversion | Keras-to-TFLite, including INT8 | `local_convert_tflite.py`, `convert_insect3_to_tflite.py` | Yes | Edge deployment adapter |
| Classifier evaluation | Accuracy, reports, confusion matrices | `eval_insect3_classifier.py`, `local_tflite_evaluate.py` | Keep but expand | Conditional error evaluation by scene state |
| Focal-target detector training | Prepare / split / train Cirsium YOLO | `prepare_cirsium_all.py`, `build_cirsium_yolo_dataset.py`, `train_cirsium_yolo.py`, `predict_cirsium_yolo.py` | Optional plugin | Automatic target proposal / benchmark baseline |

## F. Missing functions required by the new research question

These are not implementation failures; they define the next research modules.

1. **Manual target initialisation** — define target, surrounding zone, and access zone without a taxon-specific detector.
2. **Target-relative stabilisation** — estimate target displacement or deformation, then distinguish co-moving vegetation from independent visitor motion.
3. **Interaction-zone state machine** — represent approach, zone entry, contact, access, departure, and uncertainty.
4. **Multi-target / multi-actor identity handling** — retain target IDs and resolve whether an actor belongs to the focal target or a neighbour.
5. **Error taxonomy and annotation interface** — label false triggers, missed events, wrong-target attribution, splits, merges, and occlusion.
6. **Pre-event ring buffer** — preserve a short time window before trigger confirmation.
7. **Random audit sampler** — retain high-quality non-triggered clips for unbiased error estimation.
8. **Structured event ledger** — write target IDs, settings, scene state, event states, predictions, files, and reviewer decisions to CSV or SQLite.
9. **Observation model** — estimate condition-dependent detection and attribution probabilities rather than treating all observed events as truth.
10. **Ablation harness** — run motion-only, motion-plus-detector, motion-plus-classifier, and future target-relative variants on the same labelled videos.

## Interpretation

The reusable core is already present: target localisation, local candidate extraction, staged verification, and adaptive capture. The central refactor is to treat taxon recognition as optional output while elevating event attribution and error measurement to first-class functions.