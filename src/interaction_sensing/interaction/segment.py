"""Segment attributed candidate states into auditable interaction events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from interaction_sensing.domain import InteractionEvent, InteractionState
from interaction_sensing.interaction.states import stronger_state


@dataclass(slots=True)
class EventUpdate:
    started: InteractionEvent | None = None
    active: InteractionEvent | None = None
    ended: InteractionEvent | None = None


class EventSegmenter:
    """One-state-machine baseline per `(target_id, actor_track_id)` key."""

    def __init__(self, *, quiet_seconds: float = 2.0) -> None:
        if quiet_seconds <= 0:
            raise ValueError("quiet_seconds must be positive")
        self.quiet_period = timedelta(seconds=quiet_seconds)
        self._active: dict[tuple[str, str | None], InteractionEvent] = {}
        self._last_seen: dict[tuple[str, str | None], datetime] = {}

    def observe(
        self,
        *,
        target_id: str,
        actor_track_id: str | None,
        timestamp: datetime,
        state: InteractionState,
        attribution_score: float | None = None,
        verification_score: float | None = None,
        pipeline_id: str | None = None,
    ) -> EventUpdate:
        key = (target_id, actor_track_id)
        event = self._active.get(key)
        if event is None:
            event = InteractionEvent(
                target_id=target_id,
                actor_track_id=actor_track_id,
                start_time=timestamp,
                state=state,
                attribution_score=attribution_score,
                verification_score=verification_score,
                pipeline_id=pipeline_id,
            )
            self._active[key] = event
            self._last_seen[key] = timestamp
            return EventUpdate(started=event, active=event)

        event.max_state = stronger_state(event.max_state or event.state, state)
        event.state = state
        event.attribution_score = attribution_score if attribution_score is not None else event.attribution_score
        event.verification_score = verification_score if verification_score is not None else event.verification_score
        self._last_seen[key] = timestamp
        return EventUpdate(active=event)

    def close_key(
        self,
        *,
        target_id: str,
        actor_track_id: str | None,
        end_time: datetime | None = None,
    ) -> InteractionEvent | None:
        """Force-close one active event, for capture guards or source shutdown."""

        key = (target_id, actor_track_id)
        event = self._active.pop(key, None)
        last_seen = self._last_seen.pop(key, None)
        if event is None:
            return None
        event.end_time = end_time or last_seen or event.start_time
        event.state = InteractionState.DEPARTED
        return event

    def close_quiet(self, *, now: datetime) -> list[InteractionEvent]:
        """Close events that were not refreshed during the quiet interval."""

        ended: list[InteractionEvent] = []
        for key, last_seen in list(self._last_seen.items()):
            if now - last_seen <= self.quiet_period:
                continue
            event = self.close_key(target_id=key[0], actor_track_id=key[1], end_time=last_seen)
            if event is not None:
                ended.append(event)
        return ended

    def close_all(self, *, now: datetime) -> list[InteractionEvent]:
        ended: list[InteractionEvent] = []
        for target_id, actor_track_id in list(self._active):
            event = self.close_key(target_id=target_id, actor_track_id=actor_track_id, end_time=now)
            if event is not None:
                ended.append(event)
        return ended
