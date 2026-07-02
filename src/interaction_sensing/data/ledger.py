"""SQLite event ledger with immutable IDs and JSON payloads."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from interaction_sensing.domain import AuditRecord, InteractionEvent, SceneState, TargetSpec, json_ready


class EventLedger:
    """Small dependency-free store shared by runtime, audit, and analysis code."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._initialise()

    def _initialise(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS targets (
                target_id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scene_states (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scene_states_target_time
                ON scene_states(target_id, timestamp);
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                state TEXT NOT NULL,
                max_state TEXT,
                pipeline_id TEXT,
                clip_id TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_target_time
                ON events(target_id, start_time);
            CREATE TABLE IF NOT EXISTS audits (
                audit_id TEXT PRIMARY KEY,
                clip_id TEXT NOT NULL,
                sampled_at TEXT NOT NULL,
                sampling_probability REAL NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _payload(value: Any) -> str:
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)

    def register_target(self, target: TargetSpec) -> None:
        self.connection.execute(
            """INSERT INTO targets(target_id, target_type, payload_json)
               VALUES (?, ?, ?)
               ON CONFLICT(target_id) DO UPDATE SET
                 target_type=excluded.target_type,
                 payload_json=excluded.payload_json""",
            (target.target_id, target.target_type, self._payload(target)),
        )
        self.connection.commit()

    def write_scene_state(self, state: SceneState) -> None:
        self.connection.execute(
            "INSERT INTO scene_states(target_id, timestamp, payload_json) VALUES (?, ?, ?)",
            (state.target_id, state.timestamp.isoformat(), self._payload(state)),
        )
        self.connection.commit()

    def write_event(self, event: InteractionEvent) -> None:
        payload = event.to_dict()
        self.connection.execute(
            """INSERT INTO events(
                   event_id, target_id, start_time, end_time, state, max_state,
                   pipeline_id, clip_id, payload_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id) DO UPDATE SET
                   end_time=excluded.end_time,
                   state=excluded.state,
                   max_state=excluded.max_state,
                   clip_id=excluded.clip_id,
                   payload_json=excluded.payload_json""",
            (
                event.event_id,
                event.target_id,
                event.start_time.isoformat(),
                None if event.end_time is None else event.end_time.isoformat(),
                event.state.value,
                None if event.max_state is None else event.max_state.value,
                event.pipeline_id,
                event.clip_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.connection.commit()

    def write_audit(self, audit: AuditRecord) -> None:
        self.connection.execute(
            """INSERT INTO audits(audit_id, clip_id, sampled_at, sampling_probability, payload_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(audit_id) DO UPDATE SET payload_json=excluded.payload_json""",
            (
                audit.audit_id,
                audit.clip_id,
                audit.sampled_at.isoformat(),
                audit.sampling_probability,
                self._payload(audit),
            ),
        )
        self.connection.commit()

    def fetch_events(self, *, target_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT payload_json FROM events"
        params: tuple[Any, ...] = ()
        if target_id is not None:
            query += " WHERE target_id = ?"
            params = (target_id,)
        query += " ORDER BY start_time"
        rows = self.connection.execute(query, params).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EventLedger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
