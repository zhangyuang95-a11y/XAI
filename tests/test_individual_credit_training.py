from __future__ import annotations

from argparse import Namespace

import pytest

from backend.training.warehouse_options import skill_retention_weight
from env.warehouse.coordination import (
    stable_coordination_actions,
    stable_coordination_goal_overrides,
)
from env.warehouse.domain import DeliveryTask, WarehouseConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.observations import _coordination_features
from env.warehouse.policy_metrics import EfficiencyMetrics
from env.warehouse.rewards import RewardConfig
from env.warehouse.transition_audit import necessary_teammate_route_clearance


def test_loaded_route_clearance_is_not_an_avoidable_detour() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=10))
    environment.reset(seed=1540)
    state = environment.get_state()
    for task, agent_id, position, pickup, goal in (
        (state.tasks[0], "robot_1", (3, 4), (5, 0), (6, 6)),
        (state.tasks[1], "robot_2", (4, 4), (2, 8), (1, 1)),
    ):
        agent = state.by_id(agent_id)
        task.status = "carried"
        task.carrier_agent_id = agent_id
        task.pickup_position = pickup
        task.delivery_position = goal
        agent.position = position
        agent.battery = 90.0
        agent.carrying_task_id = task.task_id
        agent.route_commitment_task_id = task.task_id
        agent.navigation_goal_kind = "delivery"
        agent.navigation_goal_position = goal
    environment.set_state(state)
    before = environment.get_state()

    assert necessary_teammate_route_clearance(
        environment,
        before,
        before.by_id("robot_2"),
    )
    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "RIGHT"}
    )

    assert "robot_2" not in info["avoidable_detour_agents"]
    assert "robot_2" not in info[
        "avoidable_loaded_delivery_detour_agents"
    ]


def test_participant_actions_are_excluded_from_actor_efficiency_metrics() -> None:
    metrics = EfficiencyMetrics()
    metrics.update_step(
        {
            "individual_progress_rewards": {"robot_1": 1.0, "robot_2": 2.0},
            "counterfactual_regret_units": {"robot_1": 3.0, "robot_2": 4.0},
            "counterfactual_regret_penalty_rewards": {
                "robot_1": -0.3,
                "robot_2": -0.4,
            },
            "repeated_avoidable_wait_penalty_rewards": {
                "robot_1": -0.2,
                "robot_2": -0.1,
            },
            "avoidable_wait_agents": ("robot_1", "robot_2"),
            "avoidable_detour_agents": ("robot_1", "robot_2"),
            "avoidable_loaded_delivery_detour_agents": (
                "robot_1",
                "robot_2",
            ),
            "avoidable_wait_streaks": {"robot_1": 7, "robot_2": 2},
        },
        excluded_agent_ids=("robot_1",),
    )

    assert metrics.progress_rewards == {"robot_1": 0.0, "robot_2": 2.0}
    assert metrics.avoidable_wait_counts == {"robot_1": 0, "robot_2": 1}
    assert metrics.maximum_wait_streaks == {"robot_1": 0, "robot_2": 2}
    assert metrics.loaded_detour_counts == {"robot_1": 0, "robot_2": 1}


def test_skill_retention_decays_linearly_to_zero_by_half_training() -> None:
    args = Namespace(
        episodes=100,
        skill_retention_weight=1.0,
        skill_retention_fade_end=0.5,
        skill_retention_fixed=False,
    )
    assert skill_retention_weight(args, 0) == pytest.approx(1.0)
    assert skill_retention_weight(args, 25) == pytest.approx(0.5)
    assert skill_retention_weight(args, 50) == pytest.approx(0.0)
    assert skill_retention_weight(args, 100) == pytest.approx(0.0)


def test_legacy_retention_is_available_only_for_explicit_ablation() -> None:
    args = Namespace(
        episodes=100,
        skill_retention_weight=5.0,
        skill_retention_fade_end=0.5,
        skill_retention_fixed=True,
    )
    assert skill_retention_weight(args, 0) == pytest.approx(5.0)
    assert skill_retention_weight(args, 100) == pytest.approx(5.0)


def test_old_reward_version_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="Unsupported reward version"):
        RewardConfig(version="warehouse_safe_mission_reward_v18_efficiency_penalties")


def test_legacy_team_credit_requires_an_explicit_ablation_flag() -> None:
    legacy = RewardConfig(
        individual_credit_enabled=False,
        progress_scale=2.0,
        coordination_clearance_cost=16.0,
        avoidable_wait_cost=0.01,
        mission_regression_scale=1.0,
    )
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8, reward=legacy))
    environment.reset(seed=7)
    _, rewards, _, _, info = environment.step(
        {"robot_1": "UP", "robot_2": "WAIT"}
    )
    assert rewards["robot_1"] == pytest.approx(rewards["robot_2"])
    assert info["training_reward"] == pytest.approx(rewards["robot_1"])


def _install_loaded_delivery_charger_conflict(
    environment: WarehouseMultiAgentEnv,
) -> None:
    state = environment.get_state()
    state.tasks = [
        DeliveryTask(
            "task_3",
            (6, 8),
            (1, 4),
            status="carried",
            carrier_agent_id="robot_2",
            created_frame=14,
            claimed_frame=14,
        ),
        DeliveryTask("task_5", (2, 8), (5, 1), created_frame=26),
    ]
    robot_one = state.by_id("robot_1")
    robot_one.position = (4, 4)
    robot_one.battery = 44.0
    robot_one.carrying_task_id = None
    robot_one.route_commitment_task_id = "task_5"
    robot_one.navigation_goal_kind = "charge"
    robot_one.navigation_goal_position = environment.layout.charger_position
    robot_one.charge_mode_active = True
    robot_two = state.by_id("robot_2")
    robot_two.position = (6, 4)
    robot_two.battery = 44.0
    robot_two.carrying_task_id = "task_3"
    robot_two.route_commitment_task_id = "task_3"
    robot_two.navigation_goal_kind = "delivery"
    robot_two.navigation_goal_position = (1, 4)
    robot_two.charge_mode_active = False
    environment.set_state(state)


def test_actor_priority_and_teacher_share_loaded_delivery_precedence() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=1)
    _install_loaded_delivery_charger_conflict(environment)
    state = environment.get_state()

    overrides = stable_coordination_goal_overrides(environment)
    robot_one_features = _coordination_features(
        state,
        "robot_1",
        environment.config,
    )
    robot_two_features = _coordination_features(
        state,
        "robot_2",
        environment.config,
    )

    assert overrides["robot_1"] == environment.layout.charger_position
    assert overrides["robot_2"] == (1, 4)
    assert robot_one_features[6:8] == [0.0, 1.0]
    assert robot_two_features[6:8] == [1.0, 0.0]
    assert stable_coordination_actions(environment) == {
        "robot_1": "WAIT",
        "robot_2": "UP",
    }


def test_joint_wait_credit_detects_state_only_coordinated_escape() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2)
    _install_loaded_delivery_charger_conflict(environment)

    _, rewards, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    assert info["joint_wait_escape_actions"] == {
        "robot_1": "RIGHT",
        "robot_2": "UP",
    }
    assert info["avoidable_wait_agents"] == ()
    assert info["counterfactual_regret_units"] == {
        "robot_1": pytest.approx(2.0),
        "robot_2": pytest.approx(2.0),
    }
    assert rewards["robot_1"] < 0.0
    assert rewards["robot_2"] < 0.0


def test_critical_charger_survival_precedes_loaded_delivery() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=3)
    state = environment.get_state()
    state.tasks = [
        DeliveryTask(
            "task_1",
            (6, 8),
            (1, 1),
            status="carried",
            carrier_agent_id="robot_2",
        ),
        DeliveryTask("task_2", (2, 8), (5, 1)),
    ]
    robot_one = state.by_id("robot_1")
    robot_one.position = (1, 3)
    robot_one.battery = 16.0
    robot_one.route_commitment_task_id = "task_2"
    robot_one.navigation_goal_kind = "charge"
    robot_one.navigation_goal_position = environment.layout.charger_position
    robot_one.charge_mode_active = True
    robot_two = state.by_id("robot_2")
    robot_two.position = (1, 4)
    robot_two.battery = 36.0
    robot_two.carrying_task_id = "task_1"
    robot_two.route_commitment_task_id = "task_1"
    robot_two.navigation_goal_kind = "delivery"
    robot_two.navigation_goal_position = (1, 1)
    environment.set_state(state)
    state = environment.get_state()

    assert _coordination_features(
        state,
        "robot_1",
        environment.config,
    )[6:8] == [1.0, 0.0]
    assert _coordination_features(
        state,
        "robot_2",
        environment.config,
    )[6:8] == [0.0, 1.0]
    # The critical robot cannot condition on the loaded peer's private
    # current-frame action.  The peer clears first; the critical robot enters
    # the just-vacated cell only on the next frozen transition.
    assert stable_coordination_actions(environment) == {
        "robot_1": "WAIT",
        "robot_2": "UP",
    }


def test_charge_mode_cancels_off_station_when_shared_state_becomes_safe() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=4)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    robot.position = (6, 4)
    robot.battery = 100.0
    robot.charge_mode_active = True
    robot.navigation_goal_kind = "charge"
    robot.navigation_goal_position = environment.layout.charger_position

    environment._refresh_navigation_goals(state)

    assert robot.charge_mode_active is False
    assert robot.navigation_goal_kind == "wait"
    assert robot.navigation_goal_position == robot.position


def test_teacher_prioritizes_the_more_urgent_empty_charger_entry() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=15038)
    while environment.get_state().frame < 90:
        actions = stable_coordination_actions(environment)
        _, _, terminated, truncated, info = environment.step(actions)
        assert not info.get("shutdowns")
        assert not terminated
        assert not truncated
    while True:
        actions = stable_coordination_actions(environment)
        _, _, terminated, truncated, info = environment.step(actions)
        assert not info.get("shutdowns")
        if terminated or truncated:
            break
    assert environment.get_state().total_deliveries >= 1
