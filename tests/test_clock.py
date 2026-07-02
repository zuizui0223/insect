from datetime import datetime, timezone

from interaction_sensing.runtime.clock import SourceClock


def test_source_clock_falls_back_when_camera_position_stalls() -> None:
    clock = SourceClock(started_at=datetime(2026, 1, 1, tzinfo=timezone.utc), fps=20.0)
    first_time, first_seconds = clock.timestamp(frame_index=0, position_msec=0.0)
    second_time, second_seconds = clock.timestamp(frame_index=1, position_msec=0.0)

    assert first_seconds == 0.0
    assert second_seconds == 0.05
    assert (second_time - first_time).total_seconds() == 0.05
