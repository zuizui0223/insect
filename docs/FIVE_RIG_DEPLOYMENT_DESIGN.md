# Five-rig deployment design: one IMX500, five field nodes

## Hardware reality

The current deployment has five Raspberry Pi rigs:

```text
zuizui2  Raspberry Pi + Sony IMX500 AI Camera
zuizui   standard field node
zuizui3  standard field node
zuizui4  standard field node
zuizui5  standard field node
```

Only `zuizui2` has an AI Camera. The study must therefore **not** describe the
five rigs as five independent IMX500 replications, and must not estimate a
general device-to-device IMX500 variance component.

This is not a fatal limitation. It changes the design into three distinct
claims, each with an appropriate unit of replication.

## Three evidence streams

### A. Algorithmic / methodological superiority

**Question:** Does reference-guided nuisance cancellation and risk-guided audit
improve observation quality relative to conventional processing?

**Evidence unit:** repeated recording blocks, not cameras.

All method variants run on the exact same raw stream from `zuizui2`:

```text
B0  fixed-interval capture
B1  conventional motion trigger
B2  global stabilisation / thresholding
N0  reference-guided cancellation with host temporal references
N1  N0 + IMX500 noise-state side channel
N2  N1 + risk-guided audit and high-resolution context capture
```

Because every method sees identical frames, any difference in false events,
true-change recall, risk calibration, or audit yield cannot be attributed to a
different scene or camera view. Repeated blocks span day, physical background,
noise family, intensity, and mixed perturbations.

**This is the primary evidence for the method paper.**

### B. IMX500 implementation feasibility

**Question:** Can one real IMX500 edge rig execute the proposed pipeline long
enough, cheaply enough, and reliably enough for a field deployment?

**Evidence unit:** repeated operational sessions on `zuizui2`.

Measure:

```text
end-to-end latency
inference and capture throughput
power / Wh per hour
storage / GB per hour
thermal state
crash and recovery events
continuous operation duration
sensor output / KPI availability
```

This demonstrates feasibility on the tested IMX500 platform. It is not a claim
about all IMX500 units or all edge cameras.

### C. Five-node field-operational robustness

**Question:** Does the architecture remain operable as a multi-node ecological
camera deployment when one node has the IMX500 enhancement and four nodes use
the host-side fallback?

```text
zuizui2  N1/N2: IMX500 noise-state side channel + host temporal references
others   N0: host temporal references + the same audit / ledger contracts
```

All five nodes should share:

```text
clock synchronisation policy
manifest and run identifiers
scene-state / observability schema
adaptive recording policy interface
audit schedule and ledger schema
power, storage, thermal, and restart logs
```

The four standard nodes are not failed versions of the method. They are the
portable host-only implementation and provide heterogeneous operational contexts
for the common data contracts.

## Controlled experiment layout

### 1. Same-stream ablation on zuizui2

This is the clean comparison. Record each NoiseBench block once with `zuizui2`,
then replay its low-cost stream through B0–N2 offline. For power/latency, run
predeclared live modes sequentially in randomly ordered, matched blocks.

Do not compare a scene seen by `zuizui2` against a different scene seen by a
standard rig as the primary effectiveness test.

### 2. Physical replication through blocks, not AI units

Use independent blocks defined as:

```text
session day × physical scene/background × perturbation schedule × run replicate
```

Minimum balanced design:

```text
>= 3 days
>= 3 physical scene/background arrangements
>= 3 repeats per noise family × intensity
stable, single-noise, and mixed-noise conditions
one locked held-out outdoor scene/day
```

The exact number should be increased after a pilot variance estimate. The
important point is that frames within one clip are not independent replicates.

### 3. Standard-node companion recordings

Use `zuizui`, `zuizui3`, `zuizui4`, and `zuizui5` in parallel for:

```text
host-only N0 feasibility
clock / logging / audit reliability
battery and storage variation
long-duration field operations
outdoor background diversity
failure and restart behaviour
```

Where camera viewpoints differ, treat node identity and scene as deployment
contexts, not as a direct causal comparison with zuizui2.

## Outdoor pre-deployment rotation

To avoid permanently confounding `zuizui2` with one favourable location:

```text
Day 1: zuizui2 at site A
Day 2: zuizui2 at site B
Day 3: zuizui2 at site C
...
```

Rotate the IMX500 rig among representative locations or controlled outdoor
setups. The four standard rigs can remain at companion stations and provide
concurrent weather / operational context. Randomise or counterbalance the order
of sites and recording modes.

## Paper claims and prohibited claims

### Defensible

> On a Raspberry Pi AI Camera testbed, reference-guided nuisance cancellation
> and noise-aware audit selection improved predeclared observation-quality
> endpoints relative to matched baselines under controlled and held-out outdoor
> disturbances.

> The host-side noise-aware architecture operated across five Raspberry Pi field
> nodes, with an IMX500-enhanced mode on one node and a portable fallback mode on
> four standard nodes.

### Do not claim

```text
five independent IMX500 replications
universal superiority of IMX500 cameras
hardware-general performance across all AI cameras
five-node causal comparison of IMX500 versus non-IMX500
biological interaction accuracy before the downstream field study
```

## How the later biological study cites this paper

The field ecology paper can state:

```text
Camera observations were collected with a prevalidated noise-aware sensing
architecture. The prior methods study established controlled nuisance-cancellation,
observability-risk recording, and audit-selection behaviour before biological
interpretation.
```

The field paper must still perform task-specific audits for its own biological
endpoint, because error magnitudes are target and scene dependent.
