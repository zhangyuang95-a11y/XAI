"""Frozen task-goal matching helpers for offline coordination labels."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

from .energy_management import charger_service_required
from .environment import MOVE_DELTAS, WarehouseMultiAgentEnv, shortest_path_distance


def claim_safe_distance(
    environment: WarehouseMultiAgentEnv,
    agent: Any,
    origin: tuple[int, int],
    goal: tuple[int, int],
    available_pickups: set[tuple[int, int]],
) -> int:
    """Shortest route that cannot accidentally claim another open task."""

    if (
        agent.carrying_task_id is not None
        or (
            agent.navigation_goal_kind != "pickup"
            and goal not in available_pickups
        )
    ):
        return shortest_path_distance(
            origin,
            goal,
            environment.config.map_layout_id,
        )
    forbidden = available_pickups - (
        {goal} if goal in available_pickups else set()
    )
    if origin == goal:
        return 0
    queue = deque(((origin, 0),))
    visited = {origin}
    while queue:
        position, distance = queue.popleft()
        for delta in MOVE_DELTAS.values():
            candidate = (position[0] + delta[0], position[1] + delta[1])
            if (
                candidate in visited
                or candidate in forbidden
                or not environment.layout.is_passable(candidate)
            ):
                continue
            if candidate == goal:
                return distance + 1
            visited.add(candidate)
            queue.append((candidate, distance + 1))
    return 10_000


def stable_coordination_goal_overrides(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    """Return one frozen, charge-safe task matching for offline labels."""

    state = environment.get_state()
    if goal_overrides is None:
        overrides: dict[str, tuple[int, int]] = {}
        for agent in state.agents:
            if charger_service_required(environment, state, agent):
                overrides[agent.agent_id] = environment.layout.charger_position
                continue
            goal = environment._frozen_route_goal(
                state,
                agent.agent_id,
                prioritize_old_tasks=True,
            )
            overrides[agent.agent_id] = (
                goal if goal is not None else agent.position
            )
    else:
        overrides = dict(goal_overrides)
    available_pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    return {
        agent_id: goal
        for agent_id, goal in overrides.items()
        if not (
            goal in available_pickups
            and claim_safe_distance(
                environment,
                state.by_id(agent_id),
                state.by_id(agent_id).position,
                goal,
                available_pickups,
            )
            >= 10_000
        )
    }
