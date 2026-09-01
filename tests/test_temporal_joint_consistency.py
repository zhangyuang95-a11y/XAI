from __future__ import annotations

from copy import deepcopy

from env.warehouse.coordination_plan import frozen_joint_coordination_plan
from env.warehouse.domain import WarehouseConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import COMPACT_STAGGERED_8X9_LAYOUT
from env.warehouse.navigation import ACTIONS


def _legacy_config(**overrides) -> WarehouseConfig:
    return WarehouseConfig(
        rows=COMPACT_STAGGERED_8X9_LAYOUT.rows,
        cols=COMPACT_STAGGERED_8X9_LAYOUT.cols,
        map_layout_id=COMPACT_STAGGERED_8X9_LAYOUT.layout_id,
        **overrides,
    )


def _occupied_route_environment() -> WarehouseMultiAgentEnv:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=31_001)
    state = environment.get_state()
    priority = state.by_id("robot_1")
    clearing = state.by_id("robot_2")
    priority.position = (4, 4)
    clearing.position = (5, 4)
    priority.battery = 80.0
    clearing.battery = 80.0
    task = state.tasks[0]
    task.status = "carried"
    task.carrier_agent_id = priority.agent_id
    task.delivery_position = (5, 2)
    priority.carrying_task_id = task.task_id
    environment.set_state(state)
    return environment


def _only_allowed_action(observation: object) -> str:
    mask = observation[-len(ACTIONS) :]
    allowed = [action for action, value in zip(ACTIONS, mask) if value > 0.5]
    assert len(allowed) == 1
    return allowed[0]


def test_occupied_route_plan_clears_then_preserves_priority_followthrough() -> None:
    environment = _occupied_route_environment()
    state = environment.get_state()
    plan = state.active_coordination_plan
    assert plan is not None
    assert plan["phase"] == "CLEAR_CELL"
    assert plan["priority_agent_id"] == "robot_1"
    assert plan["clearing_agent_id"] == "robot_2"
    assert plan["moving_action"] == "DOWN"

    observations = environment.observations()
    assert _only_allowed_action(observations["robot_1"]) == "WAIT"
    assert _only_allowed_action(observations["robot_2"]) == "DOWN"
    _, _, _, _, clear_info = environment.step(
        {"robot_1": "WAIT", "robot_2": "DOWN"}
    )
    assert clear_info["unexplained_reversal_agents"] == ()

    state = environment.get_state()
    assert state.active_coordination_plan is not None
    assert state.active_coordination_plan["phase"] == "PASS_THROUGH"
    observations = environment.observations()
    assert _only_allowed_action(observations["robot_1"]) == "DOWN"
    assert _only_allowed_action(observations["robot_2"]) == "WAIT"
    _, _, _, _, pass_info = environment.step(
        {"robot_1": "DOWN", "robot_2": "WAIT"}
    )
    assert environment.get_state().by_id("robot_1").position == (5, 4)
    assert pass_info["unexplained_reversal_agents"] == ()
    assert any(
        event.get("event") == "coordination_yield"
        for event in pass_info["coordination_events"]
    )


def test_followthrough_plan_is_cancelled_when_priority_goal_changes() -> None:
    environment = _occupied_route_environment()
    environment.step({"robot_1": "WAIT", "robot_2": "DOWN"})
    state = environment.get_state()
    assert state.active_coordination_plan is not None
    assert state.active_coordination_plan["phase"] == "PASS_THROUGH"

    stale = deepcopy(state)
    priority = stale.by_id("robot_1")
    priority.carrying_task_id = None
    priority.route_commitment_task_id = None
    priority.goal_type = "SELECT_TASK"
    priority.goal_id = None
    plan = frozen_joint_coordination_plan(
        stale,
        environment.config,
        requires_charge={agent.agent_id: False for agent in stale.agents},
    )
    assert plan is None


def test_critical_charger_handoff_is_one_joint_causal_plan() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=31_002)
    state = environment.get_state()
    occupant = state.by_id("robot_1")
    waiter = state.by_id("robot_2")
    occupant.position = environment.layout.charger_position
    occupant.battery = 36.0
    occupant.charge_mode_active = True
    waiter.position = (6, 4)
    waiter.battery = 18.0
    waiter.charge_mode_active = True
    environment.set_state(state)

    plan = environment.get_state().active_coordination_plan
    assert plan is not None
    assert plan["reason_code"] == "critical_charger_route_clearance"
    _, _, _, _, info = environment.step(
        {"robot_1": str(plan["moving_action"]), "robot_2": "WAIT"}
    )
    departure = next(
        event
        for event in info["energy_events"]
        if event.get("event") == "charger_departure"
    )
    assert departure["premature"] is False
    trace = info["decision_trace"]
    assert trace["fact_valid"] is True
    assert trace["fact_validation_failures"] == ()
    assert trace["agents"]["robot_1"]["primary_reason_code"] == (
        "CLEAR_TEAMMATE_ROUTE"
    )
    assert trace["agents"]["robot_2"]["primary_reason_code"] == (
        "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE"
    )


def test_immediate_reverse_without_lifecycle_event_is_audited() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=31_003)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    state.by_id("robot_1").position = (5, 4)
    state.by_id("robot_2").position = (1, 4)
    environment.set_state(state)

    environment.step({"robot_1": "DOWN", "robot_2": "WAIT"})
    # Remove the task-rematching lifecycle event created by the first move;
    # this test isolates a genuine reversal with no new goal or obstacle.
    assert environment.state is not None
    environment.state.by_id("robot_1").goal_since = -10
    environment.state.by_id("robot_1").goal_switch_reason = "state_restore"
    _, _, _, _, info = environment.step(
        {"robot_1": "UP", "robot_2": "WAIT"}
    )
    assert info["unexplained_reversal_agents"] == ("robot_1",)
    assert info["temporal_consistency_penalty_rewards"]["robot_1"] < 0.0


def test_energy_infeasible_pickup_is_rejected_before_first_task_step() -> None:
    """Do not take an A-directed step and discover the charge need later."""

    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=31_006)
    state = environment.get_state()
    charging = state.by_id("robot_1")
    teammate = state.by_id("robot_2")
    pickup_task, teammate_task = state.tasks
    charging.position = (5, 4)
    charging.battery = 10.0
    charging.route_commitment_task_id = pickup_task.task_id
    pickup_task.pickup_position = (3, 4)
    pickup_task.delivery_position = (3, 0)
    teammate.position = (1, 0)
    teammate_task.status = "carried"
    teammate_task.carrier_agent_id = teammate.agent_id
    teammate_task.delivery_position = (2, 8)
    teammate.carrying_task_id = teammate_task.task_id
    environment.set_state(state)

    before = environment.get_state().by_id("robot_1")
    assert before.goal_type == "GO_TO_CHARGER"
    assert before.navigation_goal_kind == "charge"
    assert _only_allowed_action(environment.observations()["robot_1"]) == "DOWN"

    environment.step({"robot_1": "DOWN", "robot_2": "WAIT"})
    after = environment.get_state().by_id("robot_1")
    assert after.goal_type == "GO_TO_CHARGER"
    assert _only_allowed_action(environment.observations()["robot_1"]) == "DOWN"


def test_persistent_pickup_goals_are_distinct_and_observable() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    observations, _ = environment.reset(seed=31_004)
    state = environment.get_state()
    goals = [
        agent.goal_id
        for agent in state.agents
        if agent.goal_type == "GO_TO_PICKUP"
    ]
    assert len(goals) == len(set(goals))
    for agent in state.agents:
        if agent.goal_type == "GO_TO_PICKUP":
            assert agent.goal_id is not None
            assert observations[agent.agent_id].shape == observations[
                "robot_1"
            ].shape


def test_loaded_robot_departing_charger_is_explained_by_delivery_progress() -> None:
    """A charger tile alone must not fabricate a charge-completion reason."""

    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=31_005)
    state = environment.get_state()
    loaded = state.by_id("robot_2")
    teammate = state.by_id("robot_1")
    teammate.position = (1, 4)
    loaded.position = environment.layout.charger_position
    loaded.battery = 36.0
    loaded.charge_mode_active = False
    task = state.tasks[0]
    task.status = "carried"
    task.carrier_agent_id = loaded.agent_id
    task.delivery_position = (4, 7)
    loaded.carrying_task_id = task.task_id
    environment.set_state(state)

    before = environment.get_state().by_id("robot_2")
    assert before.navigation_goal_kind == "delivery"
    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "UP"}
    )

    trace = info["decision_trace"]
    assert trace["fact_validation_failures"] == ()
    assert trace["agents"]["robot_2"]["primary_reason_code"] == (
        "DELIVERY_ROUTE_PROGRESS"
    )
