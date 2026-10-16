"""Frozen route-goal selection shared by scoring and offline supervision."""

from __future__ import annotations

from typing import Any

from .domain import WarehouseState


def frozen_route_goal(
    environment: Any,
    state: WarehouseState,
    agent_id: str,
    *,
    prioritize_old_tasks: bool = False,
) -> tuple[int, int] | None:
    """Choose one route goal from the immutable transition-start snapshot."""

    agent = state.by_id(agent_id)
    if not agent.active:
        return None
    if agent.carrying_task_id:
        task = state.task_by_id(agent.carrying_task_id)
        return (
            task.delivery_position
            if environment._task_is_directly_energy_safe(state, agent, task)
            else environment.layout.charger_position
        )

    available = sorted(
        (task for task in state.tasks if task.status == "available"),
        key=lambda task: task.task_id,
    )
    if not available:
        return None
    assignments = environment._frozen_task_assignments(
        state,
        prioritize_old_tasks=prioritize_old_tasks,
    )
    assigned = assignments.get(agent_id)
    if prioritize_old_tasks and assigned is not None:
        # Starvation-correction supervision deliberately overrides a younger
        # unclaimed route commitment. Runtime calls leave this flag false, so
        # ordinary frame-to-frame task persistence is unchanged.
        return assigned.pickup_position
    committed_task_id = agent.route_commitment_task_id
    if committed_task_id is None and agent.goal_type == "GO_TO_PICKUP":
        committed_task_id = agent.goal_id
    committed = next(
        (task for task in available if task.task_id == committed_task_id),
        None,
    )
    if committed is not None:
        return (
            committed.pickup_position
            if environment._task_is_directly_energy_safe(state, agent, committed)
            else environment.layout.charger_position
        )
    if assigned is not None:
        return assigned.pickup_position

    assigned_task_ids = {task.task_id for task in assignments.values()}
    remaining = [
        task for task in available if task.task_id not in assigned_task_ids
    ]
    if remaining and not any(
        environment._task_is_directly_energy_safe(state, agent, task)
        for task in remaining
    ):
        return environment.layout.charger_position
    return None
