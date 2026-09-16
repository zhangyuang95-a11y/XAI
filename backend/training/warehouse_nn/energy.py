"""One public, task-state-aware energy contract for training and evaluation."""
from __future__ import annotations

import math
from typing import Any

from env.warehouse.navigation import shortest_path_distance


ENERGY_VERSION = "warehouse-nn-energy.v1"


def _distance(env: Any, start, end) -> float:
    return float(shortest_path_distance(start, end, env.config.map_layout_id))


def task_budget(env: Any, agent_id: str, task: Any) -> dict[str, Any]:
    """Describe a task without assigning it to the Actor.

    A task is executable by this robot only when it already carries it or when
    it is available and the robot is empty.  A task carried by the teammate is
    retained as an observable status but never becomes this robot's work budget.
    """
    agent = env.state.by_id(agent_id)
    other = next(item for item in env.state.agents if item.agent_id != agent_id)
    carried_self = task.status == "carried" and task.carrier_agent_id == agent_id
    carried_other = task.status == "carried" and task.carrier_agent_id == other.agent_id
    eligible = bool(carried_self or (task.status == "available" and agent.carrying_task_id is None))
    pickup_steps = 0.0 if carried_self else _distance(env, agent.position, task.pickup_position)
    delivery_steps = _distance(
        env, agent.position if carried_self else task.pickup_position,
        task.delivery_position,
    )
    return_steps = _distance(env, task.delivery_position, env.layout.charger_position)
    charger_steps = _distance(env, agent.position, env.layout.charger_position)
    reserve_steps = float(env.config.mission_reserve_steps + env.config.charge_release_hysteresis_steps)
    direct_route = pickup_steps + delivery_steps + return_steps + reserve_steps
    from_charger_route = (
        (0.0 if carried_self else _distance(env, env.layout.charger_position, task.pickup_position))
        + _distance(env, env.layout.charger_position if carried_self else task.pickup_position,
                    task.delivery_position)
        + return_steps + reserve_steps
    )
    pickup_to_charger = (
        charger_steps if carried_self
        else _distance(env, agent.position, task.pickup_position)
        + _distance(env, task.pickup_position, env.layout.charger_position)
    )
    charger_to_delivery_and_back = (
        _distance(env, env.layout.charger_position, task.delivery_position)
        + return_steps + reserve_steps
    )
    finite = all(math.isfinite(value) for value in (
        pickup_steps, delivery_steps, return_steps, charger_steps, from_charger_route,
        pickup_to_charger, charger_to_delivery_and_back,
    ))
    cost = float(env.config.move_battery_cost)
    direct_required = cost * direct_route if finite and eligible else None
    from_charger_required = cost * from_charger_route if finite and eligible else None
    return {
        "version": ENERGY_VERSION,
        "task_id": task.task_id,
        "status": task.status,
        "carrier_agent_id": task.carrier_agent_id,
        "available": task.status == "available",
        "carried_self": carried_self,
        "carried_other": carried_other,
        "eligible": eligible,
        "reachable": finite,
        "pickup_steps": pickup_steps if finite else None,
        "delivery_steps": delivery_steps if finite else None,
        "return_steps": return_steps if finite else None,
        "charger_steps": charger_steps if finite else None,
        "reserve_steps": reserve_steps,
        "direct_required_battery": direct_required,
        "from_charger_required_battery": from_charger_required,
        "direct_margin": None if direct_required is None else float(agent.battery) - direct_required,
        "charge_needed": None if direct_required is None else max(0.0, direct_required - float(agent.battery)),
        "single_charge_feasible": bool(from_charger_required is not None and from_charger_required <= 100.0),
        "mid_charge_feasible": bool(
            eligible and finite
            and pickup_to_charger * cost <= float(agent.battery)
            and charger_to_delivery_and_back * cost <= 100.0
        ),
        "pre_mid_charge_required_battery": pickup_to_charger * cost if finite and eligible else None,
        "post_mid_charge_required_battery": charger_to_delivery_and_back * cost if finite and eligible else None,
    }


def task_budgets(env: Any, agent_id: str) -> list[dict[str, Any]]:
    return [task_budget(env, agent_id, task) for task in sorted(
        env.state.tasks, key=lambda item: int(item.task_id.rsplit("_", 1)[-1])
    )]


__all__ = ["ENERGY_VERSION", "task_budget", "task_budgets"]
