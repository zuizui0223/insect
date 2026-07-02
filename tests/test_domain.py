from datetime import datetime

from interaction_sensing.domain import BBox, Candidate, InteractionState, TargetSpec
from interaction_sensing.targets.zones import classify_candidate_state


def test_nested_zones_assign_strongest_state() -> None:
    target = TargetSpec(
        target_id="target-1",
        target_type="flower",
        core_zone=BBox(10, 10, 30, 30),
        context_zone=BBox(0, 0, 40, 40),
        access_zone=BBox(15, 15, 25, 25),
    )
    candidate = Candidate(
        timestamp=datetime.now(),
        bbox=BBox(17, 17, 19, 19),
        relative_motion_score=0.3,
    )
    assert classify_candidate_state(candidate, target) is InteractionState.ACCESS_ZONE_ENTRY
