"""Validation and text rendering helpers for warehouse state snapshots."""

from __future__ import annotations

import math
from typing import Any

from .domain import WarehouseState
from .navigation import (
    ACTIONS,
    is_passable,
    pickup_pairs,
    shortest_path_distance,
)


def validate_warehouse_state(
    environment: Any,
    state: WarehouseState,
) -> tuple[str, ...]:
    """Return every state-contract error without mutating the environment."""

    errors: list[str] = []
    ids = [agent.agent_id for agent in state.agents]
    if ids != list(environment.agent_ids):
        errors.append("state must contain robot_1 and robot_2 in stable order")
    positions = [agent.position for agent in state.agents]
    if len(positions) != len(set(positions)):
        errors.append("robots cannot overlap")
    for agent in state.agents:
        if not is_passable(agent.position, environment.config.map_layout_id):
            errors.append(f"{agent.agent_id} is outside a passable aisle")
        if not 0.0 <= agent.battery <= 100.0:
            errors.append(f"{agent.agent_id} battery is outside [0, 100]")
        if agent.last_action not in ACTIONS:
            errors.append(f"{agent.agent_id} has an invalid requested action")
        if agent.last_executed_action not in ACTIONS:
            errors.append(f"{agent.agent_id} has an invalid executed action")
    active_ids = [task.task_id for task in state.tasks]
    completed_ids = [task.task_id for task in state.completed_tasks]
    if len(active_ids) != environment.config.active_task_count:
        errors.append("state must keep exactly two active tasks")
    if len(set((*active_ids, *completed_ids))) != len(active_ids) + len(
        completed_ids
    ):
        errors.append("task IDs must be unique")
    endpoints: list[tuple[int, int]] = []
    pickup_positions = {
        access
        for _, access in pickup_pairs(environment.config.map_layout_id)
    }
    for task in state.tasks:
        if task.status not in {"available", "carried"}:
            errors.append(f"active task {task.task_id} has invalid status")
        if task.pickup_position not in pickup_positions:
            errors.append(f"task {task.task_id} pickup is not shelf-adjacent")
        if not is_passable(
            task.delivery_position,
            environment.config.map_layout_id,
        ):
            errors.append(f"task {task.task_id} delivery is not passable")
        if task.pickup_position == task.delivery_position:
            errors.append(f"task {task.task_id} has identical endpoints")
        if (
            shortest_path_distance(
                task.pickup_position,
                task.delivery_position,
                environment.config.map_layout_id,
            )
            < environment.config.minimum_task_distance
        ):
            errors.append(f"task {task.task_id} is shorter than the minimum")
        endpoints.extend((task.pickup_position, task.delivery_position))
        if task.status == "available" and task.carrier_agent_id is not None:
            errors.append(f"available task {task.task_id} has a carrier")
        if task.status == "carried":
            if task.carrier_agent_id not in ids:
                errors.append(f"carried task {task.task_id} has no valid carrier")
            elif (
                state.by_id(task.carrier_agent_id).carrying_task_id
                != task.task_id
            ):
                errors.append(f"task {task.task_id} and carrier disagree")
    if len(endpoints) != len(set(endpoints)):
        errors.append("active task endpoints must be unique")
    carried_ids = [
        agent.carrying_task_id
        for agent in state.agents
        if agent.carrying_task_id is not None
    ]
    if len(carried_ids) != len(set(carried_ids)):
        errors.append("a task cannot be carried by two robots")
    if state.total_deliveries != len(state.completed_tasks):
        errors.append("total deliveries must equal completed task history")
    expected_score = sum(float(value) for value in state.score_breakdown.values())
    if not math.isclose(state.user_score, expected_score, abs_tol=1e-6):
        errors.append("user score must equal its component breakdown")
    return tuple(errors)


def render_ascii_state(
    environment: Any,
    state: WarehouseState,
) -> tuple[str, ...]:
    """Render one state using the same canonical MapLayout as execution."""

    grid = [
        [
            "."
            if is_passable((row, column), environment.config.map_layout_id)
            else "S"
            for column in range(environment.layout.cols)
        ]
        for row in range(environment.layout.rows)
    ]
    charger = environment.layout.charger_position
    grid[charger[0]][charger[1]] = "C"
    for row, column in environment.layout.robot_start_positions:
        grid[row][column] = "W"
    for index, task in enumerate(sorted(state.tasks, key=lambda item: item.task_id)):
        if task.status == "available":
            row, column = task.pickup_position
            grid[row][column] = chr(ord("A") + index)
        row, column = task.delivery_position
        grid[row][column] = chr(ord("a") + index)
    for index, agent in enumerate(state.agents, start=1):
        row, column = agent.position
        grid[row][column] = str(index) if agent.active else "X"
    return tuple("".join(row) for row in grid)
