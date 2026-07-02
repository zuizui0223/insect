#!/usr/bin/env python3
"""Build a transparent audit-validation table from automatic and human events.

Input CSVs require at least:
  event_id,target_id,start_time,end_time

Truth rows may additionally contain `truth_target_id`; when it differs from the
system target, the resulting system event is labelled as wrong-target evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from interaction_sensing.evaluation.matching import match_events


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-events", type=Path, required=True)
    parser.add_argument("--truth-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-temporal-iou", type=float, default=0.1)
    args = parser.parse_args()

    system = read_csv(args.system_events)
    truth = read_csv(args.truth_events)
    matches, unmatched_system, unmatched_truth = match_events(
        system, truth, min_temporal_iou=args.min_temporal_iou
    )
    match_by_system = {match.system_event_id: match for match in matches}
    truth_by_id = {str(row["event_id"]): row for row in truth}

    rows: list[dict[str, object]] = []
    for row in system:
        event_id = str(row["event_id"])
        match = match_by_system.get(event_id)
        if match is None:
            rows.append({**row, "audit_class": "false_event", "matched_truth_event_id": ""})
            continue
        truth_row = truth_by_id[match.truth_event_id]
        truth_target = truth_row.get("truth_target_id") or truth_row.get("target_id")
        audit_class = "matched" if truth_target == row.get("target_id") else "wrong_target"
        rows.append(
            {
                **row,
                "audit_class": audit_class,
                "matched_truth_event_id": match.truth_event_id,
                "temporal_iou": match.temporal_iou,
            }
        )
    for row in unmatched_truth:
        rows.append({**row, "audit_class": "missed_event", "matched_truth_event_id": ""})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "matched": len(matches),
        "false_events": len(unmatched_system),
        "missed_events": len(unmatched_truth),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
