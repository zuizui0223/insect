"""Condition-stratified observability summaries.

This module deliberately starts with transparent grouped rates. Hierarchical
observation models belong on top of this audited table, not in the runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ObservabilityCell:
    condition_key: tuple[tuple[str, Any], ...]
    truth_events: int
    detected_events: int
    false_events: int

    @property
    def detection_probability(self) -> float | None:
        return None if self.truth_events == 0 else self.detected_events / self.truth_events

    @property
    def false_event_rate(self) -> float | None:
        total = self.detected_events + self.false_events
        return None if total == 0 else self.false_events / total

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["condition_key"] = dict(self.condition_key)
        payload["detection_probability"] = self.detection_probability
        payload["false_event_rate"] = self.false_event_rate
        return payload


def summarise_observability(rows: Iterable[dict[str, Any]], *, condition_fields: list[str]) -> list[ObservabilityCell]:
    """Group manually audited rows by explicit scene-condition fields.

    Each row should contain a truth label and a system label:
    `truth_focal_event` and `system_focal_event`. Missing values are ignored.
    """

    grouped: dict[tuple[tuple[str, Any], ...], dict[str, int]] = {}
    for row in rows:
        if row.get("truth_focal_event") is None or row.get("system_focal_event") is None:
            continue
        key = tuple((field, row.get(field)) for field in condition_fields)
        counts = grouped.setdefault(key, {"truth": 0, "detected": 0, "false": 0})
        truth = bool(row["truth_focal_event"])
        system = bool(row["system_focal_event"])
        if truth:
            counts["truth"] += 1
        if truth and system:
            counts["detected"] += 1
        if not truth and system:
            counts["false"] += 1
    return [
        ObservabilityCell(
            condition_key=key,
            truth_events=counts["truth"],
            detected_events=counts["detected"],
            false_events=counts["false"],
        )
        for key, counts in sorted(grouped.items(), key=lambda item: repr(item[0]))
    ]
