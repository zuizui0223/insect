"""State ordering for interaction events."""

from interaction_sensing.domain import InteractionState

_STATE_RANK = {
    InteractionState.OUTSIDE: 0,
    InteractionState.APPROACH: 1,
    InteractionState.CONTEXT_ENTRY: 2,
    InteractionState.TARGET_CONTACT: 3,
    InteractionState.ACCESS_ZONE_ENTRY: 4,
    InteractionState.DEPARTED: 0,
    InteractionState.UNKNOWN: -1,
}


def state_rank(state: InteractionState) -> int:
    return _STATE_RANK[state]


def stronger_state(current: InteractionState, observed: InteractionState) -> InteractionState:
    """Return the state with stronger evidence of focal interaction."""

    return observed if state_rank(observed) > state_rank(current) else current
