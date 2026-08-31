from __future__ import annotations

from env.warehouse.domain import WarehouseConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import COMPACT_STAGGERED_8X9_LAYOUT
from env.warehouse.evaluation_diagnostics import (
    avoidable_loaded_delivery_detour_agents,
)


def _diagnostic_config(**overrides: object) -> WarehouseConfig:
    values: dict[str, object] = {
        "rows": COMPACT_STAGGERED_8X9_LAYOUT.rows,
        "cols": COMPACT_STAGGERED_8X9_LAYOUT.cols,
        "map_layout_id": COMPACT_STAGGERED_8X9_LAYOUT.layout_id,
    }
    values.update(overrides)
    return WarehouseConfig(**values)


def _loaded_state(environment: WarehouseMultiAgentEnv):
    environment.reset(seed=91_001)
    state = environment.get_state()
    for agent, task in zip(state.agents, state.tasks):
        task.status = "carried"
        task.carrier_agent_id = agent.agent_id
        task.claimed_frame = 1
        agent.carrying_task_id = task.task_id
        agent.navigation_goal_kind = "delivery"
        agent.navigation_goal_position = task.delivery_position
        agent.battery = 78.0
    return state


def test_strict_metric_exempts_public_single_lane_clearance() -> None:
    environment = WarehouseMultiAgentEnv(_diagnostic_config(horizon=120))
    state = _loaded_state(environment)
    outer = state.by_id("robot_1")
    clearing = state.by_id("robot_2")
    outer.position = (3, 2)
    outer.navigation_goal_position = (4, 7)
    clearing.position = (3, 4)
    clearing.navigation_goal_position = (3, 1)
    environment.set_state(state)

    assert avoidable_loaded_delivery_detour_agents(
        environment,
        state,
        {"robot_1": "RIGHT", "robot_2": "UP"},
    ) == ()


def test_strict_metric_still_reports_unrelated_loaded_regression() -> None:
    environment = WarehouseMultiAgentEnv(_diagnostic_config(horizon=120))
    state = _loaded_state(environment)
    loaded = state.by_id("robot_1")
    peer = state.by_id("robot_2")
    loaded.position = (3, 4)
    loaded.navigation_goal_position = (3, 1)
    peer.position = (5, 4)
    peer.navigation_goal_position = (5, 1)
    environment.set_state(state)

    assert avoidable_loaded_delivery_detour_agents(
        environment,
        state,
        {"robot_1": "UP", "robot_2": "WAIT"},
    ) == ("robot_1",)


def test_strict_metric_uses_actor_visible_commitment_during_persistent_clearance() -> None:
    environment = WarehouseMultiAgentEnv(_diagnostic_config(horizon=120))
    state = _loaded_state(environment)
    clearing = state.by_id("robot_1")
    outer = state.by_id("robot_2")
    clearing.position = (3, 4)
    clearing.navigation_goal_position = (5, 3)
    clearing.last_executed_action = "UP"
    outer_task = state.task_by_id(outer.carrying_task_id)
    outer_task.status = "available"
    outer_task.carrier_agent_id = None
    outer_task.claimed_frame = None
    outer_task.pickup_position = (3, 1)
    outer.position = (4, 5)
    outer.carrying_task_id = None
    outer.route_commitment_task_id = outer_task.task_id
    outer.navigation_goal_kind = "wait"
    outer.navigation_goal_position = outer.position
    outer.last_executed_action = "WAIT"
    environment.set_state(state)

    assert avoidable_loaded_delivery_detour_agents(
        environment,
        state,
        {"robot_1": "LEFT", "robot_2": "LEFT"},
    ) == ()


def test_strict_metric_exempts_best_exit_for_urgent_charger_queue() -> None:
    environment = WarehouseMultiAgentEnv(_diagnostic_config(horizon=120))
    state = _loaded_state(environment)
    waiter = state.by_id("robot_1")
    occupant = state.by_id("robot_2")
    waiter.position = (5, 4)
    waiter.battery = 18.0
    waiter_task = state.task_by_id(waiter.carrying_task_id)
    waiter_task.status = "available"
    waiter_task.carrier_agent_id = None
    waiter_task.claimed_frame = None
    waiter.carrying_task_id = None
    waiter.navigation_goal_kind = "charge"
    waiter.navigation_goal_position = environment.layout.charger_position
    occupant.position = environment.layout.charger_position
    occupant.battery = 54.0
    state.task_by_id(occupant.carrying_task_id).delivery_position = (1, 2)
    occupant.navigation_goal_position = (1, 2)
    environment.set_state(state)

    assert avoidable_loaded_delivery_detour_agents(
        environment,
        state,
        {"robot_1": "WAIT", "robot_2": "LEFT"},
    ) == ()
