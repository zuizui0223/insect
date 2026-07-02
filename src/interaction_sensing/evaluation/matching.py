"""Temporal target-aware matching between system and audit truth events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EventMatch:
    system_event_id: str
    truth_event_id: str
    target_id: str
    overlap_seconds: float
    temporal_iou: float


def _parse_time(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _temporal_iou(left: dict, right: dict) -> tuple[float, float]:
    start_left = _parse_time(left["start_time"])
    end_left = _parse_time(left.get("end_time")) or start_left
    start_right = _parse_time(right["start_time"])
    end_right = _parse_time(right.get("end_time")) or start_right
    assert start_left is not None and end_left is not None
    assert start_right is not None and end_right is not None
    overlap = max(0.0, (min(end_left, end_right) - max(start_left, start_right)).total_seconds())
    union = max(end_left, end_right) - min(start_left, start_right)
    union_seconds = max(union.total_seconds(), 1e-9)
    return overlap, overlap / union_seconds


def match_events(
    system_events: Iterable[dict],
    truth_events: Iterable[dict],
    *,
    min_temporal_iou: float = 0.1,
) -> tuple[list[EventMatch], list[dict], list[dict]]:
    """Greedily match events only when target identity and time overlap agree."""

    remaining_truth = list(truth_events)
    matches: list[EventMatch] = []
    unmatched_system: list[dict] = []
    for system in system_events:
        best_index: int | None = None
        best_score = -1.0
        best_overlap = 0.0
        for index, truth in enumerate(remaining_truth):
            if system.get("target_id") != truth.get("target_id"):
                continue
            overlap, score = _temporal_iou(system, truth)
            if score > best_score:
                best_index, best_score, best_overlap = index, score, overlap
        if best_index is None or best_score < min_temporal_iou:
            unmatched_system.append(system)
            continue
        truth = remaining_truth.pop(best_index)
        matches.append(
            EventMatch(
                system_event_id=str(system["event_id"]),
                truth_event_id=str(truth["event_id"]),
                target_id=str(system["target_id"]),
                overlap_seconds=best_overlap,
                temporal_iou=best_score,
            )
        )
    return matches, unmatched_system, remaining_truth
