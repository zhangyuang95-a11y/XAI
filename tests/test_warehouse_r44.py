from copy import deepcopy

import pytest

from backend.warehouse_r44_runtime import R44WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.r44_charger import R44_OBSERVATION_FEATURE_NAMES
from env.warehouse_native.partners import _goals
from ui.warehouse_alignment_r42_release import R44UnifiedExplainer


def _env(*, occupant="robot_2", occupant_battery=51,
         teammate_battery=20, teammate_position=(5, 2)):
    env = R44WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    env.reset(seed=44)
    state = env.get_state()
    other = "robot_1" if occupant == "robot_2" else "robot_2"
    state.by_id(occupant).position = env.layout.charger_position
    state.by_id(occupant).battery = float(occupant_battery)
    state.by_id(occupant).active = True
    state.by_id(other).position = teammate_position
    state.by_id(other).battery = float(teammate_battery)
    state.by_id(other).active = True
    env.set_state(state)
    return env, other


def _wait(env, occupant="robot_2", other="robot_1"):
    return env.step({occupant: "WAIT", other: "WAIT"})


def _penalties(info):
    return [event for event in info["events"]
            if event["event"] == "charger_occupancy_penalty"]


@pytest.mark.parametrize("before,expected", [
    (49, False), (50, False), (51, True), (53, True),
])
def test_post_charge_boundary_is_immediate(before, expected):
    env, other = _env(occupant_battery=before)
    observations, _rewards, _terminated, _truncated, info = _wait(
        env, "robot_2", other
    )
    assert env.state.by_id("robot_2").battery == min(100, before + 10)
    assert bool(_penalties(info)) is expected
    # The observation returned by step is the same post-rule observation a
    # direct call returns; the penalized bit cannot lag by one frame.
    assert (observations["robot_2"] == env.observations()["robot_2"]).all()


def test_one_penalty_per_occupancy_and_departure_rearms():
    env, other = _env(occupant_battery=53)
    first = _wait(env, "robot_2", other)[-1]
    assert _penalties(first)[0]["occupant_battery_after"] == 63
    score = env.state.user_score
    assert not _penalties(_wait(env, "robot_2", other)[-1])
    assert env.state.user_score == score - 1
    env.step({"robot_1": "WAIT", "robot_2": "UP"})
    env.step({"robot_1": "WAIT", "robot_2": "DOWN"})
    assert _penalties(_wait(env, "robot_2", other)[-1])


def test_collision_and_no_safe_exit_do_not_penalize_or_rearm():
    env, _other = _env(occupant_battery=53)
    collision = env.step({"robot_1": "RIGHT", "robot_2": "WAIT"})[-1]
    assert collision["robot_collision"] is True
    assert not _penalties(collision)
    # Remaining on the same occupancy after a collision can still receive its
    # one penalty; the collision did not act as a false departure.
    assert _penalties(_wait(env)[-1])
    assert not _penalties(_wait(env)[-1])

    blocked, other = _env(occupant_battery=53)
    blocked._safe_departures = lambda _state, _agent_id: ()
    assert not _penalties(_wait(blocked, "robot_2", other)[-1])


def test_snapshot_preserves_occupancy_flag_and_observation():
    env, other = _env(occupant_battery=53)
    _wait(env, "robot_2", other)
    snapshot = deepcopy(env.snapshot())
    restored = R44WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert not _penalties(_wait(restored, "robot_2", other)[-1])
    start = len(restored.feature_names) - len(R44_OBSERVATION_FEATURE_NAMES)
    assert restored.observations()["robot_2"][start + 1] == 1.0


def test_unified_parser_binds_person_and_time():
    mine = R44UnifiedExplainer.parse_question({
        "question": "我现在需要充电吗？", "intent_id": None,
    })
    assert mine["subject"] == "robot_1"
    assert mine["temporal"] == "current"
    future = R44UnifiedExplainer.parse_question({
        "question": "如果我接下来连续等待三步会怎样？",
        "intent_id": "counterfactual",
    })
    assert future == {"intent_id": "counterfactual", "subject": "robot_2",
                      "temporal": "future", "steps": 3}
    past = R44UnifiedExplainer.parse_question({
        "question": "如果我刚才改成向左会怎样？",
        "intent_id": "counterfactual",
    })
    assert past["temporal"] == "past"


def test_partner_budget_reads_three_percent_environment_configuration():
    two = R44WarehouseEnv(collaborative_study_config(move_battery_cost=2.0))
    three = R44WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    two.reset(seed=0)
    three.reset(seed=0)
    for env in (two, three):
        state = env.get_state()
        state.by_id("robot_2").battery = 34.0
        env.set_state(state)
    assert _goals(two, "skilled")["robot_2"] != two.layout.charger_position
    assert _goals(three, "skilled")["robot_2"] == three.layout.charger_position
