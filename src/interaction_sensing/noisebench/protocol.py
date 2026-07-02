"""Target-agnostic NoiseBench scenario definitions.

NoiseBench is not a biological benchmark. It is a controlled perturbation
protocol for asking whether a sensor can characterise when observations become
unreliable, why they become unreliable, and which windows deserve audit.

Every generated run has explicit, reproducible perturbation truth. A biological
object may be added to a later downstream experiment, but it is not required by
this protocol or by its primary outcome measures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from random import Random
from typing import Any
from uuid import uuid4

from interaction_sensing.noise import NoiseSource


class PerturbationKind(str, Enum):
    STABLE_CONTROL = "stable_control"
    CAMERA_SHAKE = "camera_shake"
    CO_MOVING_FOREGROUND = "co_moving_foreground"
    BACKGROUND_VEGETATION_MOTION = "background_vegetation_motion"
    ILLUMINATION_TRANSIENT = "illumination_transient"
    SHADOW_TRANSIENT = "shadow_transient"
    OCCLUSION = "occlusion"
    BLUR_OR_FOCUS_LOSS = "blur_or_focus_loss"
    LENS_CONTAMINATION = "lens_contamination"
    MULTI_OBJECT_CLUTTER = "multi_object_clutter"
    MIXED_DISTURBANCE = "mixed_disturbance"

    @property
    def primary_noise_source(self) -> NoiseSource:
        mapping = {
            PerturbationKind.STABLE_CONTROL: NoiseSource.STABLE_SCENE,
            PerturbationKind.CAMERA_SHAKE: NoiseSource.GLOBAL_CAMERA_SHAKE,
            PerturbationKind.CO_MOVING_FOREGROUND: NoiseSource.CO_MOVING_FOREGROUND,
            PerturbationKind.BACKGROUND_VEGETATION_MOTION: NoiseSource.BACKGROUND_VEGETATION_MOTION,
            PerturbationKind.ILLUMINATION_TRANSIENT: NoiseSource.ILLUMINATION_TRANSIENT,
            PerturbationKind.SHADOW_TRANSIENT: NoiseSource.SHADOW_TRANSIENT,
            PerturbationKind.OCCLUSION: NoiseSource.OCCLUSION,
            PerturbationKind.BLUR_OR_FOCUS_LOSS: NoiseSource.BLUR_OR_FOCUS_LOSS,
            PerturbationKind.LENS_CONTAMINATION: NoiseSource.LENS_CONTAMINATION,
            PerturbationKind.MULTI_OBJECT_CLUTTER: NoiseSource.MULTI_OBJECT_CLUTTER,
            PerturbationKind.MIXED_DISTURBANCE: NoiseSource.UNKNOWN,
        }
        return mapping[self]

    @property
    def expected_error_channels(self) -> tuple[str, ...]:
        """Mechanistic hypotheses, not labels from an AI model."""

        mapping = {
            PerturbationKind.STABLE_CONTROL: (),
            PerturbationKind.CAMERA_SHAKE: ("false_event", "missed_event"),
            PerturbationKind.CO_MOVING_FOREGROUND: ("false_event",),
            PerturbationKind.BACKGROUND_VEGETATION_MOTION: ("false_event", "wrong_attribution"),
            PerturbationKind.ILLUMINATION_TRANSIENT: ("false_event", "missed_event"),
            PerturbationKind.SHADOW_TRANSIENT: ("false_event", "missed_event"),
            PerturbationKind.OCCLUSION: ("missed_event", "wrong_attribution"),
            PerturbationKind.BLUR_OR_FOCUS_LOSS: ("missed_event",),
            PerturbationKind.LENS_CONTAMINATION: ("missed_event",),
            PerturbationKind.MULTI_OBJECT_CLUTTER: ("false_event", "wrong_attribution"),
            PerturbationKind.MIXED_DISTURBANCE: ("false_event", "missed_event", "wrong_attribution"),
        }
        return mapping[self]


@dataclass(frozen=True, slots=True)
class Perturbation:
    """A time-bounded, physically enactable scene disturbance with known truth."""

    kind: PerturbationKind
    intensity: float
    start_seconds: float
    duration_seconds: float
    apparatus: str
    protocol_instruction: str
    perturbation_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must lie in [0, 1]")
        if self.start_seconds < 0.0:
            raise ValueError("start_seconds must be non-negative")
        if self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")
        if not self.apparatus.strip():
            raise ValueError("apparatus cannot be empty")
        if not self.protocol_instruction.strip():
            raise ValueError("protocol_instruction cannot be empty")

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["noise_source"] = self.kind.primary_noise_source.value
        payload["expected_error_channels"] = list(self.kind.expected_error_channels)
        payload["end_seconds"] = self.end_seconds
        return payload


@dataclass(frozen=True, slots=True)
class NoiseBenchScenario:
    """One target-agnostic controlled recording run."""

    scenario_id: str
    replicate: int
    duration_seconds: float
    frame_rate: float
    perturbations: tuple[Perturbation, ...]
    scene_description: str = "generic natural-background scene; no biological focal target required"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id cannot be empty")
        if self.replicate < 1:
            raise ValueError("replicate must be >= 1")
        if self.duration_seconds <= 0.0 or self.frame_rate <= 0.0:
            raise ValueError("duration_seconds and frame_rate must be positive")
        for perturbation in self.perturbations:
            if perturbation.end_seconds > self.duration_seconds + 1e-9:
                raise ValueError("perturbation extends beyond scenario duration")

    @property
    def frame_count(self) -> int:
        return round(self.duration_seconds * self.frame_rate)

    @property
    def is_target_agnostic(self) -> bool:
        return True

    def to_manifest_rows(self) -> list[dict[str, Any]]:
        base = {
            "scenario_id": self.scenario_id,
            "replicate": self.replicate,
            "duration_seconds": self.duration_seconds,
            "frame_rate": self.frame_rate,
            "frame_count": self.frame_count,
            "scene_description": self.scene_description,
            "target_agnostic": True,
        }
        if not self.perturbations:
            return [{**base, "kind": PerturbationKind.STABLE_CONTROL.value, "intensity": 0.0}]
        return [{**base, **perturbation.to_dict()} for perturbation in self.perturbations]


@dataclass(frozen=True, slots=True)
class NoiseBenchConfig:
    """Controlled benchmark design, kept small enough for field-side execution."""

    replicates: int = 3
    duration_seconds: float = 30.0
    frame_rate: float = 15.0
    intensities: tuple[float, ...] = (0.30, 0.60, 0.90)
    include_mixed_disturbance: bool = True
    seed: int = 20260702

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if self.duration_seconds <= 0.0 or self.frame_rate <= 0.0:
            raise ValueError("duration_seconds and frame_rate must be positive")
        if not self.intensities:
            raise ValueError("intensities cannot be empty")
        if any(not 0.0 < value <= 1.0 for value in self.intensities):
            raise ValueError("intensities must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NoiseBenchPlan:
    config: NoiseBenchConfig
    scenarios: tuple[NoiseBenchScenario, ...]

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for order, scenario in enumerate(self.scenarios, start=1):
            for row in scenario.to_manifest_rows():
                rows.append({"recording_order": order, **row})
        return rows


def build_noisebench_plan(config: NoiseBenchConfig = NoiseBenchConfig()) -> NoiseBenchPlan:
    """Generate a randomised protocol with stable, single, and mixed disturbances.

    Every single-disturbance run contains one perturbation centred within a fixed
    recording. Mixed runs deliberately overlap two mechanisms so models cannot
    succeed merely by recognising isolated laboratory-like artefacts.
    """

    scenarios: list[NoiseBenchScenario] = []
    single_kinds = [
        PerturbationKind.CAMERA_SHAKE,
        PerturbationKind.CO_MOVING_FOREGROUND,
        PerturbationKind.BACKGROUND_VEGETATION_MOTION,
        PerturbationKind.ILLUMINATION_TRANSIENT,
        PerturbationKind.SHADOW_TRANSIENT,
        PerturbationKind.OCCLUSION,
        PerturbationKind.BLUR_OR_FOCUS_LOSS,
        PerturbationKind.LENS_CONTAMINATION,
        PerturbationKind.MULTI_OBJECT_CLUTTER,
    ]
    for replicate in range(1, config.replicates + 1):
        scenarios.append(_stable_scenario(config, replicate))
        for kind in single_kinds:
            for intensity in config.intensities:
                scenarios.append(_single_scenario(config, kind, intensity, replicate))
        if config.include_mixed_disturbance:
            for intensity in config.intensities:
                scenarios.extend(_mixed_scenarios(config, intensity, replicate))

    rng = Random(config.seed)
    rng.shuffle(scenarios)
    return NoiseBenchPlan(config=config, scenarios=tuple(scenarios))


def _stable_scenario(config: NoiseBenchConfig, replicate: int) -> NoiseBenchScenario:
    return NoiseBenchScenario(
        scenario_id=f"stable-r{replicate}",
        replicate=replicate,
        duration_seconds=config.duration_seconds,
        frame_rate=config.frame_rate,
        perturbations=(),
        metadata={"condition_family": "control"},
    )


def _single_scenario(
    config: NoiseBenchConfig,
    kind: PerturbationKind,
    intensity: float,
    replicate: int,
) -> NoiseBenchScenario:
    duration = _perturbation_duration(config.duration_seconds, intensity)
    start = (config.duration_seconds - duration) / 2.0
    perturbation = Perturbation(
        kind=kind,
        intensity=intensity,
        start_seconds=start,
        duration_seconds=duration,
        **_apparatus_for(kind, intensity),
    )
    return NoiseBenchScenario(
        scenario_id=f"{kind.value}-i{intensity:.2f}-r{replicate}",
        replicate=replicate,
        duration_seconds=config.duration_seconds,
        frame_rate=config.frame_rate,
        perturbations=(perturbation,),
        metadata={"condition_family": "single"},
    )


def _mixed_scenarios(config: NoiseBenchConfig, intensity: float, replicate: int) -> list[NoiseBenchScenario]:
    pairs = [
        (PerturbationKind.CAMERA_SHAKE, PerturbationKind.SHADOW_TRANSIENT),
        (PerturbationKind.CO_MOVING_FOREGROUND, PerturbationKind.MULTI_OBJECT_CLUTTER),
        (PerturbationKind.OCCLUSION, PerturbationKind.BLUR_OR_FOCUS_LOSS),
    ]
    duration = _perturbation_duration(config.duration_seconds, intensity)
    start = (config.duration_seconds - duration) / 2.0
    scenarios: list[NoiseBenchScenario] = []
    for left, right in pairs:
        left_perturbation = Perturbation(
            kind=left,
            intensity=intensity,
            start_seconds=start,
            duration_seconds=duration,
            **_apparatus_for(left, intensity),
        )
        right_perturbation = Perturbation(
            kind=right,
            intensity=intensity,
            start_seconds=start + duration * 0.20,
            duration_seconds=duration * 0.80,
            **_apparatus_for(right, intensity),
        )
        scenarios.append(
            NoiseBenchScenario(
                scenario_id=f"mixed-{left.value}__{right.value}-i{intensity:.2f}-r{replicate}",
                replicate=replicate,
                duration_seconds=config.duration_seconds,
                frame_rate=config.frame_rate,
                perturbations=(left_perturbation, right_perturbation),
                metadata={"condition_family": "mixed", "mixed_label": PerturbationKind.MIXED_DISTURBANCE.value},
            )
        )
    return scenarios


def _perturbation_duration(recording_duration: float, intensity: float) -> float:
    # Stronger perturbations are sustained longer, but always leave clean context
    # before and after the event for temporal model calibration.
    return min(recording_duration * 0.70, max(recording_duration * 0.20, recording_duration * (0.20 + 0.35 * intensity)))


def _apparatus_for(kind: PerturbationKind, intensity: float) -> dict[str, str]:
    level = f"intensity {intensity:.2f}"
    mapping: dict[PerturbationKind, tuple[str, str]] = {
        PerturbationKind.CAMERA_SHAKE: (
            "repeatable camera mount displacement rig",
            f"Apply controlled horizontal/vertical camera displacement at {level}; retain clean lead-in and recovery.",
        ),
        PerturbationKind.CO_MOVING_FOREGROUND: (
            "foreground card or artificial foliage mounted near the lens",
            f"Move the foreground element coherently across the scene at {level} without changing camera pose.",
        ),
        PerturbationKind.BACKGROUND_VEGETATION_MOTION: (
            "fan plus background foliage or flexible surrogate",
            f"Apply airflow to background elements at {level}; keep camera and foreground support fixed.",
        ),
        PerturbationKind.ILLUMINATION_TRANSIENT: (
            "dimmable LED panel or controlled lamp shutter",
            f"Create a scene-wide brightness transition at {level} while geometry remains fixed.",
        ),
        PerturbationKind.SHADOW_TRANSIENT: (
            "opaque flag between light source and scene",
            f"Sweep a moving shadow across the scene at {level} without changing global camera pose.",
        ),
        PerturbationKind.OCCLUSION: (
            "opaque or semi-transparent occluder",
            f"Introduce partial scene occlusion covering a defined fraction at {level}.",
        ),
        PerturbationKind.BLUR_OR_FOCUS_LOSS: (
            "repeatable focus offset or optical diffusion filter",
            f"Apply a calibrated focus offset / diffusion condition at {level}.",
        ),
        PerturbationKind.LENS_CONTAMINATION: (
            "removable transparent film with droplets or particulate surrogate",
            f"Place a reversible contamination surrogate over part of the optical path at {level}.",
        ),
        PerturbationKind.MULTI_OBJECT_CLUTTER: (
            "multiple moving markers or overlapping natural-material surrogates",
            f"Introduce overlapping moving elements at {level} while preserving a fixed camera pose.",
        ),
    }
    apparatus, instruction = mapping[kind]
    return {"apparatus": apparatus, "protocol_instruction": instruction}
