"""Shared transition-local mission facts for runtime, reward, and audit.

This module is intentionally below both ``credit_assignment`` and
``transition_audit`` in the dependency graph.  Mission freezing is causal
state interpretation, not training credit and not post-transition auditing;
placing it here prevents those two consumers from importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import DeliveryTask, WarehouseState
from .energy_management import charger_service_required


@dataclass(frozen=True)
class FrozenMission:
    """One transition-local mission shared by every decision consumer."""

    goal_kind: str
    goal_position: tuple[int, int]
    task: DeliveryTask | None = None


def frozen_training_missions(
    environment: Any,
    state: WarehouseState,
) -> dict[str, FrozenMission | None]:
    """Match each robot once at transition start and freeze that mission."""

    assignments = environment._frozen_task_assignments(
        state,
        prioritize_old_tasks=True,
    )
    available = {
        task.task_id: task
        for task in state.tasks
        if task.status == "available"
    }
    reserved_task_ids = {task.task_id for task in assignments.values()}
    fallback_task_ids: set[str] = set()
    missions: dict[str, FrozenMission | None] = {}
    for agent in state.agents:
        if not agent.active:
            missions[agent.agent_id] = None
            continue
        if agent.carrying_task_id is not None:
            task = state.task_by_id(agent.carrying_task_id)
        else:
            # A route commitment is public state created by this robot's
            # previous executed movement.  Do not silently rematch it for one
            # audit frame; runtime, reward, and explanation must all consume
            # the same task.
            task = (
                available.get(agent.route_commitment_task_id)
                if agent.route_commitment_task_id is not None
                else None
            )
            if (
                task is None
                and agent.goal_type == "GO_TO_PICKUP"
                and agent.goal_id is not None
            ):
                task = available.get(agent.goal_id)
            if task is None:
                task = assignments.get(agent.agent_id)
            if task is None and charger_service_required(
                environment,
                state,
                agent,
            ):
                task = min(
                    (
                        item
                        for item in available.values()
                        if item.task_id not in reserved_task_ids
                        and item.task_id not in fallback_task_ids
                    ),
                    key=lambda item: (
                        environment._safe_task_cost(state, agent, item),
                        item.task_id,
                    ),
                    default=None,
                )
                if task is None:
                    task = min(
                        available.values(),
                        key=lambda item: (
                            environment._safe_task_cost(state, agent, item),
                            item.task_id,
                        ),
                        default=None,
                    )
                if task is not None:
                    fallback_task_ids.add(task.task_id)
        if task is None:
            missions[agent.agent_id] = None
            continue
        if charger_service_required(environment, state, agent):
            missions[agent.agent_id] = FrozenMission(
                "charge",
                environment.layout.charger_position,
                task,
            )
        elif agent.carrying_task_id is not None:
            missions[agent.agent_id] = FrozenMission(
                "delivery",
                task.delivery_position,
                task,
            )
        else:
            missions[agent.agent_id] = FrozenMission(
                "pickup",
                task.pickup_position,
                task,
            )
    return missions
