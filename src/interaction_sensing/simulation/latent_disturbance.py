"""Truth-labelled latent-disturbance worlds for noise-cancellation benchmarks.

The simulator intentionally has no flowers, insects, or object classes.  It
creates an observed local signal as a mixture of:

    independent local signal + shared nuisance field + sensor noise

The nuisance field is latent: a downstream system never receives its true
components.  It receives only noisy spatial reference channels, a noisy scene-
quality side channel, and the local observation.  This lets a benchmark test
whether correct references reduce apparent events without suppressing the local
signal, and whether deliberately broken references lose that benefit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import pi, sin
from random import Random
from statistics import median
from typing import Iterator


class ReferenceMode(str, Enum):
    """Reference-channel conditions used for causal negative controls."""

    ABSENT = "absent"
    SINGLE_REGION = "single_region"
    CORRECT = "correct"
    TIME_SHIFTED = "time_shifted"
    SPATIALLY_MISMATCHED = "spatially_mismatched"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class LatentDisturbanceConfig:
    """Parameters for one reproducible, target-agnostic observation world."""

    name: str = "default"
    frames: int = 900
    frame_rate: float = 15.0
    reference_regions: int = 11
    event_start_rate: float = 0.018
    event_min_frames: int = 3
    event_max_frames: int = 8
    event_amplitude: float = 2.50
    local_observation_noise_sd: float = 0.22
    reference_noise_sd: float = 0.18
    global_motion_sd: float = 0.62
    global_motion_persistence: float = 0.86
    sway_amplitude: float = 0.90
    sway_period_frames: int = 73
    photometric_amplitude: float = 0.55
    photometric_period_frames: int = 127
    local_sway_gain: float = 1.20
    local_photometric_gain: float = 0.80
    quality_noise_sd: float = 0.16
    degraded_reference_noise_sd: float = 0.85
    reference_delay_frames: int = 11
    nuisance_dominant_threshold: float = 0.95
    seed: int = 1

    def __post_init__(self) -> None:
        if self.frames <= 0 or self.frame_rate <= 0:
            raise ValueError("frames and frame_rate must be positive")
        if self.reference_regions < 3:
            raise ValueError("reference_regions must be at least 3 for robust aggregation")
        if not 0.0 <= self.event_start_rate <= 1.0:
            raise ValueError("event_start_rate must lie in [0, 1]")
        if self.event_min_frames <= 0 or self.event_max_frames < self.event_min_frames:
            raise ValueError("event duration bounds are invalid")
        if self.event_amplitude <= 0.0:
            raise ValueError("event_amplitude must be positive")
        if self.sway_period_frames <= 0 or self.photometric_period_frames <= 0:
            raise ValueError("periods must be positive")
        if self.reference_delay_frames < 1:
            raise ValueError("reference_delay_frames must be at least 1")
        if self.nuisance_dominant_threshold <= 0.0:
            raise ValueError("nuisance_dominant_threshold must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisturbanceFrame:
    """One observation window with hidden causal truth retained for evaluation."""

    frame_index: int
    true_local_event: bool
    true_local_signal: float
    global_motion: float
    coherent_sway: float
    photometric_shift: float
    nuisance_contribution: float
    raw_local_evidence: float
    reference_values: tuple[float, ...]
    mismatched_reference: float
    degraded_reference: float
    quality_hint: float

    @property
    def true_nuisance_dominant(self) -> bool:
        return (not self.true_local_event) and abs(self.nuisance_contribution) >= 0.0


class LatentDisturbanceWorld:
    """Generate a latent nuisance field and independent local events.

    A ``ReferenceMode.CORRECT`` reference is a robust aggregation of background
    regions that share the same latent global/sway/photometric causes as the
    local observation.  Other modes either remove, degrade, delay, or replace
    that reference with an independent process.
    """

    def __init__(self, config: LatentDisturbanceConfig) -> None:
        self.config = config
        self._rng = Random(config.seed)
        self._reference_sway_gains = tuple(
            self._rng.uniform(0.55, 1.25) for _ in range(config.reference_regions)
        )
        self._reference_photo_gains = tuple(
            self._rng.uniform(0.30, 1.10) for _ in range(config.reference_regions)
        )
        self._frames = tuple(self._generate_frames())

    @property
    def frames(self) -> tuple[DisturbanceFrame, ...]:
        return self._frames

    def iter_frames(self) -> Iterator[DisturbanceFrame]:
        yield from self._frames

    def reference(self, frame_index: int, mode: ReferenceMode) -> float:
        frame = self._frames[frame_index]
        if mode is ReferenceMode.ABSENT:
            return 0.0
        if mode is ReferenceMode.SINGLE_REGION:
            return frame.reference_values[0]
        if mode is ReferenceMode.CORRECT:
            return float(median(frame.reference_values))
        if mode is ReferenceMode.DEGRADED:
            return frame.degraded_reference
        if mode is ReferenceMode.SPATIALLY_MISMATCHED:
            return frame.mismatched_reference
        if mode is ReferenceMode.TIME_SHIFTED:
            delayed = max(0, frame_index - self.config.reference_delay_frames)
            return float(median(self._frames[delayed].reference_values))
        raise ValueError(f"unsupported reference mode: {mode}")

    def _generate_frames(self) -> Iterator[DisturbanceFrame]:
        cfg = self.config
        global_state = 0.0
        mismatched_state = 0.0
        event_remaining = 0
        event_amplitude = 0.0
        for frame_index in range(cfg.frames):
            global_state = (
                cfg.global_motion_persistence * global_state
                + self._rng.gauss(0.0, cfg.global_motion_sd)
            )
            mismatched_state = 0.79 * mismatched_state + self._rng.gauss(0.0, cfg.global_motion_sd)
            coherent_sway = cfg.sway_amplitude * sin(2.0 * pi * frame_index / cfg.sway_period_frames)
            photometric_shift = cfg.photometric_amplitude * sin(
                2.0 * pi * frame_index / cfg.photometric_period_frames + pi / 5.0
            )

            if event_remaining <= 0 and self._rng.random() < cfg.event_start_rate:
                event_remaining = self._rng.randint(cfg.event_min_frames, cfg.event_max_frames)
                event_amplitude = cfg.event_amplitude * self._rng.uniform(0.78, 1.22)
            true_local_event = event_remaining > 0
            true_local_signal = event_amplitude if true_local_event else 0.0
            if event_remaining > 0:
                event_remaining -= 1

            nuisance = (
                global_state
                + cfg.local_sway_gain * coherent_sway
                + cfg.local_photometric_gain * photometric_shift
            )
            raw = true_local_signal + nuisance + self._rng.gauss(0.0, cfg.local_observation_noise_sd)
            references = tuple(
                global_state
                + sway_gain * coherent_sway
                + photo_gain * photometric_shift
                + self._rng.gauss(0.0, cfg.reference_noise_sd)
                for sway_gain, photo_gain in zip(self._reference_sway_gains, self._reference_photo_gains)
            )
            robust_reference = float(median(references))
            quality_magnitude = abs(global_state) + abs(coherent_sway) + abs(photometric_shift)
            quality_hint = max(0.0, quality_magnitude + self._rng.gauss(0.0, cfg.quality_noise_sd))
            yield DisturbanceFrame(
                frame_index=frame_index,
                true_local_event=true_local_event,
                true_local_signal=true_local_signal,
                global_motion=global_state,
                coherent_sway=coherent_sway,
                photometric_shift=photometric_shift,
                nuisance_contribution=nuisance,
                raw_local_evidence=raw,
                reference_values=references,
                mismatched_reference=mismatched_state + self._rng.gauss(0.0, cfg.reference_noise_sd),
                degraded_reference=robust_reference + self._rng.gauss(0.0, cfg.degraded_reference_noise_sd),
                quality_hint=quality_hint,
            )
