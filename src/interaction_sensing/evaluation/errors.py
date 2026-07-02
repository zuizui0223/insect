"""Translate matched audit events into ecological observation-error summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from interaction_sensing.evaluation.matching import EventMatch, match_events


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    n_truth: int
    n_system: int
    matched: int
    false_events: int
    missed_events: int
    detection_recall: float | None
    positive_predictive_value: float | None
    wrong_target_events: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def summarise_errors(system_events: Iterable[dict], truth_events: Iterable[dict]) -> ErrorSummary:
    system = list(system_events)
    truth = list(truth_events)
    matches, false_events, missed_events = match_events(system, truth)
    matched = len(matches)
    recall = None if not truth else matched / len(truth)
    ppv = None if not system else matched / len(system)
    return ErrorSummary(
        n_truth=len(truth),
        n_system=len(system),
        matched=matched,
        false_events=len(false_events),
        missed_events=len(missed_events),
        detection_recall=recall,
        positive_predictive_value=ppv,
    )
