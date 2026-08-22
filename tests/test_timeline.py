from __future__ import annotations

import pytest

from backend.adapters.base import EnvironmentSnapshot
from ui.timeline import Timeline, TimelineFrame


def _frame(number: int) -> TimelineFrame:
    return TimelineFrame(
        snapshot=EnvironmentSnapshot(
            environment="test",
            frame=number,
            state={"frame": number},
            environment_rng_state=(number,),
        )
    )


def test_select_is_clamped_and_replace_updates_exact_frame() -> None:
    timeline = Timeline()
    timeline.reset(_frame(0))
    timeline.append(_frame(1))

    assert timeline.select(99).snapshot.frame == 1
    assert timeline.select(-5).snapshot.frame == 0
    replacement = TimelineFrame(
        snapshot=timeline.current.snapshot,
        actions={"robot_1": "LEFT"},
    )
    timeline.replace_current(replacement)
    assert timeline.current.actions == {"robot_1": "LEFT"}


def test_stepping_from_history_discards_old_future() -> None:
    timeline = Timeline()
    timeline.reset(_frame(0))
    timeline.append(_frame(1))
    timeline.append(_frame(2))
    timeline.select(0)

    timeline.append(_frame(10))

    assert timeline.count == 2
    assert timeline.current.snapshot.frame == 10
    assert timeline.max_index == 1


def test_timeline_resolves_exact_simulator_frame_ids() -> None:
    timeline = Timeline()
    timeline.reset(_frame(3))
    timeline.append(_frame(8))

    assert timeline.simulator_frame(8).snapshot.frame == 8
    with pytest.raises(KeyError, match="no simulator frame"):
        timeline.simulator_frame(7)


def test_timeline_frame_keeps_pre_transition_decision_snapshot() -> None:
    visible = _frame(8)
    decision = _frame(7).snapshot
    aligned = TimelineFrame(
        snapshot=visible.snapshot,
        decision_snapshot=decision,
        actions={"robot_2": "LEFT"},
    )

    assert aligned.snapshot.frame == 8
    assert aligned.decision_snapshot is not None
    assert aligned.decision_snapshot.frame == 7
