"""One public energy budget used by r4.5 training, inference and explanations."""
from __future__ import annotations

import math
from typing import Any

from env.warehouse.navigation import shortest_path_distance


VERSION = "warehouse-r45-energy-budget.v2"
CLEARANCE_RESERVE_STEPS = 4.0


def _distance(env: Any, left, right) -> float:
    return float(shortest_path_distance(left, right, env.config.map_layout_id))


def task_energy_budget(env: Any, agent_id: str, task: Any) -> dict:
    """Return a complete work-and-return estimate from the current public state."""
    agent = env.state.by_id(agent_id)
    carrying = agent.carrying_task_id == task.task_id
    pickup_steps = 0.0 if carrying else _distance(
        env, agent.position, task.pickup_position
    )
    delivery_steps = _distance(
        env,
        agent.position if carrying else task.pickup_position,
        task.delivery_position,
    )
    return_steps = _distance(
        env, task.delivery_position, env.layout.charger_position
    )
    reserve_steps = (float(env.config.charge_release_hysteresis_steps)
                     + CLEARANCE_RESERVE_STEPS)
    route_steps = pickup_steps + delivery_steps + return_steps
    total_steps = route_steps + reserve_steps
    required = float(env.config.move_battery_cost) * total_steps
    reachable = all(math.isfinite(value) for value in (
        pickup_steps, delivery_steps, return_steps
    ))
    return {
        "version": VERSION,
        "agent_id": agent_id,
        "task_id": task.task_id,
        "carrying": carrying,
        "pickup_steps": int(pickup_steps) if math.isfinite(pickup_steps) else None,
        "delivery_steps": int(delivery_steps) if math.isfinite(delivery_steps) else None,
        "return_steps": int(return_steps) if math.isfinite(return_steps) else None,
        "reserve_steps": reserve_steps,
        "route_steps": route_steps,
        "total_steps": total_steps,
        "required_battery": required,
        "current_battery": float(agent.battery),
        "battery_margin": float(agent.battery) - required,
        "reachable": reachable,
    }


def candidate_energy_budgets(env: Any, agent_id: str) -> list[dict]:
    agent = env.state.by_id(agent_id)
    if agent.carrying_task_id:
        tasks = [env.state.task_by_id(agent.carrying_task_id)]
    else:
        tasks = [task for task in env.state.tasks if task.status == "available"]
    rows = [task_energy_budget(env, agent_id, task) for task in tasks]
    return sorted(rows, key=lambda row: (
        not row["reachable"], row["required_battery"], row["task_id"]
    ))


def selected_energy_budget(env: Any, agent_id: str) -> dict | None:
    rows = candidate_energy_budgets(env, agent_id)
    return rows[0] if rows else None


__all__ = [
    "VERSION", "CLEARANCE_RESERVE_STEPS", "task_energy_budget", "candidate_energy_budgets",
    "selected_energy_budget",
]
