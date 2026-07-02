from datetime import datetime, timedelta, timezone

from interaction_sensing.domain import InteractionState
from interaction_sensing.interaction.segment import EventSegmenter


def test_event_segmenter_force_close_uses_requested_end_time() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    segmenter = EventSegmenter(quiet_seconds=2.0)
    update = segmenter.observe(
        target_id="target-1",
        actor_track_id=None,
        timestamp=start,
        state=InteractionState.CONTEXT_ENTRY,
    )
    assert update.started is not None

    end = start + timedelta(seconds=5)
    closed = segmenter.close_key(target_id="target-1", actor_track_id=None, end_time=end)

    assert closed is update.started
    assert closed is not None
    assert closed.end_time == end
    assert closed.state is InteractionState.DEPARTED
