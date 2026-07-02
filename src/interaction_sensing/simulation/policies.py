"""Observation policies compared by the synthetic benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import hypot
from random import Random

from interaction_sensing.domain import BBox, Candidate, TargetSpec
from interaction_sensing.interaction import assign_target
from interaction_sensing.sensing import relative_motion_magnitude
from interaction_sensing.targets import build_target_from_boxes, classify_candidate_state

from .world import CandidateObservation, ScenarioConfig, TargetFrame


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    label: str
    focal_counted: bool
    ambiguous: bool = False


class FixedContextPolicy:
    """Historical fixed-ROI baseline.

    It represents a common design choice: define one broad context region once,
    then count any candidate motion appearing there as focal activity.
    """

    name = "fixed_context"

    def __init__(self, config: ScenarioConfig) -> None:
        initial = _target_spec(config, "focal", config.focal_center)
        self.fixed_context = initial.context_zone or initial.core_zone

    def decide(self, observation: CandidateObservation, _: TargetFrame) -> PolicyDecision:
        if self.fixed_context.contains(observation.center):
            return PolicyDecision(label="focal_context_event", focal_counted=True)
        return PolicyDecision(label="outside_fixed_context", focal_counted=False)


class TargetRelativeAttributionPolicy:
    """Target-following, multi-target, ambiguity-preserving observation policy.

    The policy assumes an estimated focal and neighbour centre for the current
    frame. The benchmark perturbs those estimates with configurable tracker
    error; therefore it does not silently assume perfect tracking.
    """

    name = "target_relative_attribution"

    def __init__(self, config: ScenarioConfig, *, seed: int) -> None:
        self.config = config
        self.rng = Random(seed)
        self._previous_focal_estimate: tuple[float, float] | None = None

    def decide(self, observation: CandidateObservation, target_frame: TargetFrame) -> PolicyDecision:
        focal_center = self._estimate_center(target_frame.focal_center)
        neighbour_center = self._estimate_center(target_frame.neighbour_center)
        if self._previous_focal_estimate is None:
            focal_displacement = (0.0, 0.0)
        else:
            focal_displacement = (
                focal_center[0] - self._previous_focal_estimate[0],
                focal_center[1] - self._previous_focal_estimate[1],
            )
        self._previous_focal_estimate = focal_center

        residual = relative_motion_magnitude(observation.displacement, focal_displacement)
        if residual < self.config.relative_motion_threshold:
            return PolicyDecision(label="co_moving_or_static", focal_counted=False)

        focal = _target_spec(self.config, "focal", focal_center)
        neighbour = _target_spec(self.config, "neighbour", neighbour_center)
        timestamp = datetime(2026, 1, 1) + timedelta(seconds=observation.frame_index / self.config.frame_rate)
        candidate = Candidate(
            timestamp=timestamp,
            bbox=observation.bbox,
            relative_motion_score=residual,
            metadata={"synthetic_event_id": observation.event_id},
        )
        assignment = assign_target(candidate, [focal, neighbour], ambiguity_margin=self.config.ambiguity_margin)
        if assignment.status == "ambiguous_target":
            return PolicyDecision(label="ambiguous_target", focal_counted=False, ambiguous=True)
        if assignment.target_id != "focal":
            return PolicyDecision(label="neighbour_or_outside", focal_counted=False)

        state = classify_candidate_state(candidate, focal)
        if state.value in {"target_contact", "access_zone_entry"}:
            return PolicyDecision(label=state.value, focal_counted=True)
        return PolicyDecision(label=state.value, focal_counted=False)

    def _estimate_center(self, center: tuple[float, float]) -> tuple[float, float]:
        sd = self.config.target_tracker_error_sd
        if sd <= 0:
            return center
        return (center[0] + self.rng.gauss(0.0, sd), center[1] + self.rng.gauss(0.0, sd))


def _target_spec(config: ScenarioConfig, target_id: str, center: tuple[float, float]) -> TargetSpec:
    half = config.target_size / 2.0
    core = BBox(center[0] - half, center[1] - half, center[0] + half, center[1] + half)
    access_half = max(2.0, half * config.access_ratio)
    access = BBox(
        center[0] - access_half,
        center[1] - access_half,
        center[0] + access_half,
        center[1] + access_half,
    )
    return build_target_from_boxes(
        target_id=target_id,
        target_type="synthetic_target",
        core_zone=core,
        access_zone=access,
        context_expand_ratio=config.context_expand_ratio,
    )
