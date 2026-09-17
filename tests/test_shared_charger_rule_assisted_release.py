from __future__ import annotations

from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.runtime_coordination import select_human_ai_action


def _charger_state(occupant_battery: float, teammate_battery: float):
    env = WarehouseMultiAgentEnv(collaborative_study_config())
    env.reset(seed=901)
    state = env.get_state()
    state.by_id("robot_1").position = env.layout.charger_position
    state.by_id("robot_1").battery = occupant_battery
    state.by_id("robot_2").position = (4, 2)
    state.by_id("robot_2").battery = teammate_battery
    env.state = state
    return env


def test_shared_charger_penalty_is_immediate_and_continuous_occupancy_is_idempotent():
    env = _charger_state(53.0, 19.0)
    _, _, _, _, info = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert env.state.by_id("robot_1").battery == 63.0
    assert info["shared_charger_penalty_count"] == 1
    assert env.state.score_breakdown["shared_charger_occupancy"] == -50.0
    assert env.state.last_rule_events[0]["event_id"]

    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert env.state.score_breakdown["shared_charger_occupancy"] == -50.0


def test_shared_charger_thresholds_and_departure_reset():
    for battery, expected in ((49.0, 0), (50.0, 0), (51.0, 1)):
        env = _charger_state(battery, 19.0)
        _, _, _, _, info = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
        assert info["shared_charger_penalty_count"] == expected

    env = _charger_state(53.0, 20.0)
    _, _, _, _, info = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert info["shared_charger_penalty_count"] == 0

    env = _charger_state(53.0, 19.0)
    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    env.step({"robot_1": "UP", "robot_2": "WAIT"})
    assert env.state.shared_charger_penalty_occupants == ()


def test_rule_assisted_selection_records_nn_and_controller_actions():
    env = WarehouseMultiAgentEnv(collaborative_study_config())
    env.reset(seed=902)
    selected, runtime = select_human_ai_action(env, "WAIT")
    assert runtime["participant_action_known_at_decision_time"] is False
    assert runtime["policy_actions"]["robot_2"] == "WAIT"
    assert runtime["selected_actions"]["robot_2"] == selected
    assert "controller_reason" in runtime
    assert all("score" in candidate for candidate in runtime["ai_action_candidates"])
