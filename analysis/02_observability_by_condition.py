#!/usr/bin/env python3
"""Summarise audited observability by user-selected scene covariates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from interaction_sensing.evaluation.observability import summarise_observability


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-table", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.audit_table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["truth_focal_event"] = as_bool(row.get("truth_focal_event", ""))
        row["system_focal_event"] = as_bool(row.get("system_focal_event", ""))

    cells = summarise_observability(rows, condition_fields=args.conditions)
    payload = [cell.to_dict() for cell in cells]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
