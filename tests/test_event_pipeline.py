from datetime import datetime, timedelta

from interaction_sensing.data import EventLedger
from interaction_sensing.domain import BBox, InteractionState, TargetSpec
from interaction_sensing.interaction import EventSegmenter, assign_target
from interaction_sensing.targets.zones import classify_candidate_state
from interaction_sensing.domain import Candidate


def test_segmenter_closes_quiet_event() -> None:
    start = datetime(2026, 1, 1, 12, 0, 0)
    segmenter = EventSegmenter(quiet_seconds=1.0)
    update = segmenter.observe(
        target_id="target-1",
        actor_track_id=None,
        timestamp=start,
        state=InteractionState.CONTEXT_ENTRY,
        pipeline_id="motion_only_v1",
    )
    assert update.started is not None
    ended = segmenter.close_quiet(now=start + timedelta(seconds=2))
    assert len(ended) == 1
    assert ended[0].duration_seconds == 0.0
    assert ended[0].pipeline_id == "motion_only_v1"


def test_assign_target_preserves_ambiguity() -> None:
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    candidate = Candidate(timestamp=timestamp, bbox=BBox(9, 9, 11, 11), relative_motion_score=0.2)
    target_a = TargetSpec(target_id="a", target_type="flower", core_zone=BBox(0, 0, 12, 12))
    target_b = TargetSpec(target_id="b", target_type="flower", core_zone=BBox(8, 8, 20, 20))
    decision = assign_target(candidate, [target_a, target_b], ambiguity_margin=0.5)
    assert decision.status == "ambiguous_target"
    assert decision.competing_target_ids


def test_ledger_round_trip_event(tmp_path) -> None:
    start = datetime(2026, 1, 1, 12, 0, 0)
    target = TargetSpec(target_id="target-1", target_type="flower", core_zone=BBox(0, 0, 10, 10))
    candidate = Candidate(timestamp=start, bbox=BBox(1, 1, 2, 2), relative_motion_score=0.1)
    assert classify_candidate_state(candidate, target) is InteractionState.TARGET_CONTACT

    segmenter = EventSegmenter(quiet_seconds=1.0)
    update = segmenter.observe(
        target_id=target.target_id,
        actor_track_id=None,
        timestamp=start,
        state=InteractionState.TARGET_CONTACT,
        pipeline_id="motion_only_v1",
    )
    event = update.active
    assert event is not None
    event.end_time = start + timedelta(seconds=3)

    ledger = EventLedger(tmp_path / "events.sqlite")
    ledger.register_target(target)
    ledger.write_event(event)
    events = ledger.fetch_events(target_id="target-1")
    ledger.close()

    assert len(events) == 1
    assert events[0]["target_id"] == "target-1"
    assert events[0]["state"] == "target_contact"
    assert events[0]["pipeline_id"] == "motion_only_v1"
