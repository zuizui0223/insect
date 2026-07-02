"""Truth-labelled synthetic scenes for pre-field observation benchmarks.

The simulator intentionally models *error mechanisms*, not insect appearance.
Its unit is a moving candidate in a multi-target scene with known truth:

- focal interaction;
- neighbour-target interaction;
- pass-by in the focal context;
- target sway;
- shadow / illumination artefact.

This makes the benchmark a controlled falsification tool for sensing design.
It must not be interpreted as field-generalisation evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import pi, sin
from random import Random
from typing import Iterator
from uuid import uuid4

from interaction_sensing.domain import BBox


Point = tuple[float, float]
Vector = tuple[float, float]


class LatentKind(str, Enum):
    FOCAL_INTERACTION = "focal_interaction"
    NEIGHBOUR_INTERACTION = "neighbour_interaction"
    PASS_BY = "pass_by"
    TARGET_SWAY = "target_sway"
    SHADOW = "shadow"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """One controlled field-like scenario.

    Rates are start probabilities per frame. They are deliberately explicit so
    that every performance difference is traceable to a stated assumption.
    """

    name: str = "default"
    frames: int = 900
    frame_rate: float = 15.0
    focal_center: Point = (160.0, 160.0)
    target_size: float = 36.0
    context_expand_ratio: float = 1.0
    access_ratio: float = 0.34
    neighbour_distance: float = 64.0
    wind_amplitude: float = 8.0
    wind_period_frames: int = 90
    target_tracker_error_sd: float = 2.0
    candidate_detection_probability: float = 0.90
    focal_event_start_rate: float = 0.018
    neighbour_event_start_rate: float = 0.014
    pass_by_start_rate: float = 0.012
    shadow_start_rate: float = 0.006
    event_min_frames: int = 3
    event_max_frames: int = 8
    sway_burst_frames: int = 32
    sway_burst_interval_frames: int = 120
    relative_motion_threshold: float = 0.85
    actor_relative_speed: float = 2.3
    ambiguity_margin: float = 0.10
    audit_probability: float = 0.10
    seed: int = 1

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        if self.neighbour_distance <= 0:
            raise ValueError("neighbour_distance must be positive")
        if self.wind_period_frames <= 0:
            raise ValueError("wind_period_frames must be positive")
        if self.event_min_frames <= 0 or self.event_max_frames < self.event_min_frames:
            raise ValueError("event frame bounds are invalid")
        for value_name in (
            "candidate_detection_probability",
            "focal_event_start_rate",
            "neighbour_event_start_rate",
            "pass_by_start_rate",
            "shadow_start_rate",
            "audit_probability",
        ):
            value = getattr(self, value_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{value_name} must lie in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TargetFrame:
    frame_index: int
    focal_center: Point
    neighbour_center: Point
    focal_displacement: Vector
    neighbour_displacement: Vector


@dataclass(frozen=True, slots=True)
class LatentEvent:
    event_id: str
    kind: LatentKind
    start_frame: int
    end_frame: int
    anchor: Point
    relative_velocity: Vector

    @property
    def is_focal_truth(self) -> bool:
        return self.kind is LatentKind.FOCAL_INTERACTION

    @property
    def is_nonfocal_truth(self) -> bool:
        return not self.is_focal_truth

    def active(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """A detected candidate with hidden truth retained only for benchmarking."""

    event_id: str
    kind: LatentKind
    frame_index: int
    center: Point
    displacement: Vector
    bbox: BBox

    @property
    def true_focal_interaction(self) -> bool:
        return self.kind is LatentKind.FOCAL_INTERACTION


class SyntheticWorld:
    """Generates moving targets and latent events with reproducible truth labels."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.rng = Random(config.seed)
        self.events = self._sample_events()

    def target_frame(self, frame_index: int) -> TargetFrame:
        cfg = self.config
        phase = 2.0 * pi * frame_index / cfg.wind_period_frames
        previous_phase = 2.0 * pi * max(frame_index - 1, 0) / cfg.wind_period_frames

        focal_offset = (
            cfg.wind_amplitude * sin(phase),
            0.60 * cfg.wind_amplitude * sin(phase + pi / 3.0),
        )
        previous_focal_offset = (
            cfg.wind_amplitude * sin(previous_phase),
            0.60 * cfg.wind_amplitude * sin(previous_phase + pi / 3.0),
        )
        neighbour_offset = (
            0.78 * cfg.wind_amplitude * sin(phase + pi / 4.0),
            0.55 * cfg.wind_amplitude * sin(phase + pi / 2.0),
        )
        previous_neighbour_offset = (
            0.78 * cfg.wind_amplitude * sin(previous_phase + pi / 4.0),
            0.55 * cfg.wind_amplitude * sin(previous_phase + pi / 2.0),
        )
        focal = (cfg.focal_center[0] + focal_offset[0], cfg.focal_center[1] + focal_offset[1])
        neighbour_base = (cfg.focal_center[0] + cfg.neighbour_distance, cfg.focal_center[1] + cfg.target_size * 0.15)
        neighbour = (neighbour_base[0] + neighbour_offset[0], neighbour_base[1] + neighbour_offset[1])
        return TargetFrame(
            frame_index=frame_index,
            focal_center=focal,
            neighbour_center=neighbour,
            focal_displacement=(focal_offset[0] - previous_focal_offset[0], focal_offset[1] - previous_focal_offset[1]),
            neighbour_displacement=(neighbour_offset[0] - previous_neighbour_offset[0], neighbour_offset[1] - previous_neighbour_offset[1]),
        )

    def active_observations(self, frame_index: int, target_frame: TargetFrame) -> list[CandidateObservation]:
        observations: list[CandidateObservation] = []
        for event in self.events:
            if not event.active(frame_index):
                continue
            if self.rng.random() > self.config.candidate_detection_probability:
                continue
            observation = self._observation_for_event(event, frame_index, target_frame)
            observations.append(observation)
        return observations

    def _sample_events(self) -> list[LatentEvent]:
        cfg = self.config
        events: list[LatentEvent] = []
        for frame_index in range(cfg.frames):
            self._maybe_add(events, LatentKind.FOCAL_INTERACTION, frame_index, cfg.focal_event_start_rate)
            self._maybe_add(events, LatentKind.NEIGHBOUR_INTERACTION, frame_index, cfg.neighbour_event_start_rate)
            self._maybe_add(events, LatentKind.PASS_BY, frame_index, cfg.pass_by_start_rate)
            self._maybe_add(events, LatentKind.SHADOW, frame_index, cfg.shadow_start_rate)
        if cfg.wind_amplitude > 0:
            for start in range(1, cfg.frames, cfg.sway_burst_interval_frames):
                end = min(cfg.frames - 1, start + cfg.sway_burst_frames - 1)
                events.append(
                    LatentEvent(
                        event_id=f"sway-{start}",
                        kind=LatentKind.TARGET_SWAY,
                        start_frame=start,
                        end_frame=end,
                        anchor=(0.35 * cfg.target_size, -0.20 * cfg.target_size),
                        relative_velocity=(0.0, 0.0),
                    )
                )
        return events

    def _maybe_add(self, events: list[LatentEvent], kind: LatentKind, start_frame: int, rate: float) -> None:
        if self.rng.random() >= rate:
            return
        cfg = self.config
        duration = self.rng.randint(cfg.event_min_frames, cfg.event_max_frames)
        end_frame = min(cfg.frames - 1, start_frame + duration - 1)
        if kind is LatentKind.FOCAL_INTERACTION:
            anchor = self._inside_core_anchor()
            velocity = self._random_vector(cfg.actor_relative_speed)
        elif kind is LatentKind.NEIGHBOUR_INTERACTION:
            anchor = self._inside_core_anchor()
            velocity = self._random_vector(cfg.actor_relative_speed)
        elif kind is LatentKind.PASS_BY:
            anchor = self._inside_context_not_core_anchor()
            velocity = self._random_vector(cfg.actor_relative_speed * 1.25)
        else:  # shadow
            anchor = self._inside_core_anchor()
            velocity = self._random_vector(cfg.actor_relative_speed * 0.5)
        events.append(
            LatentEvent(
                event_id=uuid4().hex,
                kind=kind,
                start_frame=start_frame,
                end_frame=end_frame,
                anchor=anchor,
                relative_velocity=velocity,
            )
        )

    def _inside_core_anchor(self) -> Point:
        radius = self.config.target_size * 0.22
        return (self.rng.uniform(-radius, radius), self.rng.uniform(-radius, radius))

    def _inside_context_not_core_anchor(self) -> Point:
        core_half = self.config.target_size / 2.0
        context_half = core_half * (1.0 + 2.0 * self.config.context_expand_ratio)
        for _ in range(100):
            x = self.rng.uniform(-context_half * 0.92, context_half * 0.92)
            y = self.rng.uniform(-context_half * 0.92, context_half * 0.92)
            if abs(x) > core_half * 1.05 or abs(y) > core_half * 1.05:
                return (x, y)
        return (context_half * 0.8, 0.0)

    def _random_vector(self, magnitude: float) -> Vector:
        return (self.rng.uniform(-magnitude, magnitude), self.rng.uniform(-magnitude, magnitude))

    def _observation_for_event(
        self,
        event: LatentEvent,
        frame_index: int,
        target_frame: TargetFrame,
    ) -> CandidateObservation:
        cfg = self.config
        elapsed = frame_index - event.start_frame
        if event.kind is LatentKind.FOCAL_INTERACTION:
            base = target_frame.focal_center
            target_velocity = target_frame.focal_displacement
        elif event.kind is LatentKind.NEIGHBOUR_INTERACTION:
            base = target_frame.neighbour_center
            target_velocity = target_frame.neighbour_displacement
        elif event.kind is LatentKind.PASS_BY:
            base = target_frame.focal_center
            target_velocity = target_frame.focal_displacement
        elif event.kind is LatentKind.TARGET_SWAY:
            base = target_frame.focal_center
            target_velocity = target_frame.focal_displacement
        else:  # shadow
            base = target_frame.focal_center
            target_velocity = (0.0, 0.0)

        center = (
            base[0] + event.anchor[0] + event.relative_velocity[0] * elapsed,
            base[1] + event.anchor[1] + event.relative_velocity[1] * elapsed,
        )
        if event.kind is LatentKind.SHADOW:
            displacement = event.relative_velocity
        else:
            displacement = (
                target_velocity[0] + event.relative_velocity[0],
                target_velocity[1] + event.relative_velocity[1],
            )
        half = max(2.0, cfg.target_size * 0.075)
        return CandidateObservation(
            event_id=event.event_id,
            kind=event.kind,
            frame_index=frame_index,
            center=center,
            displacement=displacement,
            bbox=BBox(center[0] - half, center[1] - half, center[0] + half, center[1] + half),
        )

    def iter_frames(self) -> Iterator[tuple[TargetFrame, list[CandidateObservation]]]:
        for frame_index in range(self.config.frames):
            frame = self.target_frame(frame_index)
            yield frame, self.active_observations(frame_index, frame)
