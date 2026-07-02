# Working literature position: NoiseBench and noise-first sensing

## Scope of this document

This is a **working research map**, not a completed systematic review or a claim
that no related work exists. It records the current differentiation of this
repository and the evidence needed to test it.

## Four adjacent research streams

### 1. Ecological observation-error modelling

Ecological monitoring has long recognised that observations can contain false
positives and false negatives. Recent hierarchical work explicitly models both
forms of error and connects different occurrence-data designs through a common
observation-process framework.

```text
Strength: statistical correction of imperfect observations.
Typical gap for this project: error probabilities are usually modelled after
observation, rather than measured as a time-resolved visual scene state that can
change capture and audit decisions during deployment.
```

Working reference:

- Abubakari, K. et al. (2026). *False Positives, False Negatives, and the
  Detection-Only Problem: A Hierarchical Model for Species Occurrence with
  Observation Error*. arXiv:2606.26323.

### 2. Target-first ecological computer vision and edge AI

Camera-trap and insect-monitoring systems generally improve target detection,
classification, or trigger efficiency. Low-power CNN triggers can reduce storage
and energy use by distinguishing insects from background, while object detectors
can locate animals in camera-trap images.

```text
Strength: scalable target recognition and capture efficiency.
Typical gap for this project: scene disturbances are mostly treated as
background to suppress, not as explicit empirical variables that predict
false-event, missed-event, and attribution risks across target tasks.
```

Working references:

- Schneider, S., Taylor, G. W., & Kremer, S. C. (2018). *Deep Learning Object
  Detection Methods for Ecological Camera Trap Data*. arXiv:1803.10842.
- Gardiner, R., Rowands, S., & Simmons, B. I. (2024). *Towards Scalable Insect
  Monitoring: Ultra-Lightweight CNNs as On-Device Triggers for Insect Camera
  Traps*. arXiv:2411.14467.

### 3. Domain shift and scene dependence in wildlife vision

Camera-trap recognition generalises poorly across new locations, in part because
backgrounds, environmental conditions, and species composition change. Recent
work attempts to learn geographically invariant image representations.

```text
Strength: demonstrates that visual environment matters to recognition.
Typical gap for this project: domain shift is commonly evaluated as a final
accuracy drop, not decomposed into physically enactable noise mechanisms that
can be measured, audited, and acted on at deployment time.
```

Working references:

- Beery, S., van Horn, G., & Perona, P. (2018). *Recognition in Terra
  Incognita*. arXiv:1807.04975.
- Beery, S., Morris, D., & Yang, S. (2019). *Efficient Pipeline for Camera Trap
  Image Review*. arXiv:1907.06772.
- Santamaria, J., Isaza, C., & Giraldo, J. (2026). *WildIng: A Wildlife Image
  Invariant Representation Model for Geographical Domain Shift*.
  arXiv:2601.00993.

### 4. AI error and downstream ecological inference

Recent studies stress that conventional image-level accuracy may not represent
whether a model preserves ecological estimates. Others combine AI confidence
with human annotations to improve inference and uncertainty quantification.

```text
Strength: connects AI error to ecological conclusions and statistical inference.
Typical gap for this project: these approaches begin with AI outputs and manual
labels; they do not yet establish a target-agnostic edge layer that characterises
scene conditions, prioritises audits, and preserves unobservable windows before
raw AI counts are treated as ecological data.
```

Working references:

- Pantazis, O. et al. (2024). *Deep learning-based ecological analysis of
  camera trap images is impacted by training data quality and size*.
  arXiv:2408.14348.
- Cohen, A. et al. (2026). *Improving ecological inference and uncertainty
  quantification from camera trap data through the fusion of AI confidences and
  manual annotations*. arXiv:2605.13660.

## NoiseBench's proposed contribution

NoiseBench is not proposed as another image-quality dataset and not as a new
organism detector. Its contribution is the connection:

```text
controlled physical disturbance truth
  -> edge-estimated scene noise / observability state
  -> false-event, missed-event, and attribution-risk prediction
  -> adaptive context capture and independent audit sampling
  -> later target-specific error model and ecological inference
```

The claimed reusable layer is:

```text
noise taxonomy
+ target-agnostic controlled perturbation protocol
+ observation-risk record
+ audit-selection policy
+ task-specific calibration of risk -> error magnitude
```

## What must be demonstrated before a strong novelty claim

1. **Coverage:** the controlled perturbation taxonomy covers a substantial
   fraction of observable failure modes across at least two unrelated downstream
   tasks.
2. **Prediction:** noise state predicts actual false-event, missed-event, or
   attribution error better than a global confidence threshold alone.
3. **Actionability:** risk-guided audit capture retrieves more annotated failure
   windows per GB and per Wh than uniform or event-only capture.
4. **Transfer:** the scene-noise representation transfers better across target
   tasks than a target-trained detector's confidence alone.
5. **Inference value:** downstream estimates change less under held-out
   perturbations when the observation-risk record is used.

## Conservative paper claim

Until those tests are complete, the defensible claim is:

> NoiseBench provides a reproducible, target-agnostic protocol and software
> architecture for measuring observation disturbances and auditing their effects
> on autonomous ecological sensing.

The stronger claim — that this architecture improves ecological inference across
biological targets — requires the controlled and field validation stages.
