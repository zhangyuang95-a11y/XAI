from copy import deepcopy

from backend.training.warehouse_r45_adaptation import (
    _coordination_teacher_action, _teacher_action,
)
from backend.warehouse_r45_runtime import R45WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.r45_energy import selected_energy_budget
from ui.warehouse_alignment_r42_release import R44UnifiedExplainer


def _env():
    env = R45WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    env.reset(seed=4501)
    return env


def test_budget_uses_one_three_percent_contract():
    env = _env()
    row = selected_energy_budget(env, "robot_2")
    assert row["required_battery"] == 3.0 * (
        row["route_steps"] + env.config.charge_release_hysteresis_steps + 4
    )


def test_energy_cycle_features_are_real_and_resumable():
    env = _env()
    before = env.observations()["robot_2"].copy()
    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert len(env.state.by_id("robot_2").recent_positions) == 1
    snapshot = env.snapshot()
    restored = _env()
    restored.restore(deepcopy(snapshot))
    assert restored.feature_names == env.feature_names
    assert (restored.observations()["robot_2"] == env.observations()["robot_2"]).all()
    assert not (before == env.observations()["robot_2"]).all()


def test_teacher_uses_bounded_clearance_when_player_blocks_improving_step():
    env = _env()
    state = env.get_state()
    learner, player = state.by_id("robot_2"), state.by_id("robot_1")
    learner.position = (4, 2)
    learner.battery = 100.0
    player.position = (3, 2)
    task = next(task for task in state.tasks if task.active)
    task.pickup_position = (2, 2)
    task.delivery_position = (1, 3)
    env.set_state(state)
    action = _teacher_action(env)
    assert action == "DOWN"


def test_next_task_energy_is_not_counterfactual():
    parsed = R44UnifiedExplainer.parse_question({
        "question": "How much battery does the next task need?", "language": "en"
    })
    assert parsed["intent_id"] == "energy_budget"


def test_collision_recovery_is_an_explicit_actor_input():
    env = _env()
    index = list(env.feature_names).index("energy_cycle.last_robot_collision")
    assert env.observations()["robot_2"][index] == 0.0
    state = env.get_state()
    state.last_robot_collision_event = True
    env.set_state(state)
    assert env.observations()["robot_2"][index] == 1.0
    assert _coordination_teacher_action(env) in {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}
