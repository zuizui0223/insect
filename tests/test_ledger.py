from datetime import datetime, timezone

from interaction_sensing.data.ledger import EventLedger
from interaction_sensing.domain import AuditRecord, BBox, InteractionEvent, InteractionState, SceneState, TargetSpec


def test_ledger_returns_all_runtime_record_types(tmp_path) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    target = TargetSpec("target-1", "flower", BBox(0, 0, 10, 10))
    event = InteractionEvent("target-1", timestamp, InteractionState.CONTEXT_ENTRY)
    audit = AuditRecord("audits/audit.mp4", timestamp, 0.1)
    state = SceneState(timestamp, "target-1", illumination_mean=42.0)

    with EventLedger(tmp_path / "events.sqlite") as ledger:
        ledger.register_target(target)
        ledger.write_event(event)
        ledger.write_audit(audit)
        ledger.write_scene_state(state)

        assert ledger.fetch_targets()[0]["target_id"] == "target-1"
        assert ledger.fetch_events()[0]["event_id"] == event.event_id
        assert ledger.fetch_audits()[0]["audit_id"] == audit.audit_id
        assert ledger.fetch_scene_states()[0]["illumination_mean"] == 42.0
