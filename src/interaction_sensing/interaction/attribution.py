"""Assign candidates to a focal target or an explicit competing neighbour."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from interaction_sensing.domain import Candidate, TargetSpec


@dataclass(frozen=True, slots=True)
class AttributionDecision:
    target_id: str | None
    score: float
    status: str
    competing_target_ids: tuple[str, ...] = ()


def _membership_score(candidate: Candidate, target: TargetSpec) -> float:
    point = candidate.centroid
    if target.access_zone is not None and target.access_zone.contains(point):
        return 1.0
    if target.core_zone.contains(point):
        return 0.8
    if target.context_zone is not None and target.context_zone.contains(point):
        return 0.45
    overlap = candidate.bbox.iou(target.context_zone or target.core_zone)
    return 0.25 * overlap


def assign_target(candidate: Candidate, targets: Iterable[TargetSpec], *, ambiguity_margin: float = 0.1) -> AttributionDecision:
    """Choose a target from geometry while preserving ambiguity explicitly."""

    scored = sorted(
        ((target.target_id, _membership_score(candidate, target)) for target in targets),
        key=lambda item: item[1],
        reverse=True,
    )
    if not scored or scored[0][1] <= 0.0:
        return AttributionDecision(target_id=None, score=0.0, status="outside_all_targets")
    winning_id, winning_score = scored[0]
    competitors = tuple(target_id for target_id, score in scored[1:] if winning_score - score <= ambiguity_margin and score > 0)
    if competitors:
        return AttributionDecision(
            target_id=winning_id,
            score=winning_score,
            status="ambiguous_target",
            competing_target_ids=competitors,
        )
    return AttributionDecision(target_id=winning_id, score=winning_score, status="assigned")
