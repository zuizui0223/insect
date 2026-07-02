#!/usr/bin/env python3
"""Export the SQLite runtime ledger to reviewable CSV files.

The raw event ledger remains authoritative. CSV export is a convenience layer for
annotation and downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from interaction_sensing.data.ledger import EventLedger


def _serialise_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list))
        else value
        for key, value in row.items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [_serialise_row(row) for row in rows]
    fieldnames = sorted({key for row in serialised for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialised)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with EventLedger(args.ledger) as ledger:
        _write_csv(args.output_dir / "targets.csv", ledger.fetch_targets())
        _write_csv(args.output_dir / "events.csv", ledger.fetch_events())
        _write_csv(args.output_dir / "audits.csv", ledger.fetch_audits())
        _write_csv(args.output_dir / "scene_states.csv", ledger.fetch_scene_states())


if __name__ == "__main__":
    main()
