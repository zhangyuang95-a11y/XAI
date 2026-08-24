"""Observable warehouse collaboration, energy and task context builders."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from backend.adapters.base import EnvironmentSnapshot
from env.warehouse.environment import (
    MOVE_DELTAS,
    AgentState,
    WarehouseMultiAgentEnv,
    WarehouseState,
    shortest_path_distance,
)


@dataclass(frozen=True)
class WarehousePolicyState:
    snapshot: EnvironmentSnapshot
    agent_id: str


def _manhattan(
    left: tuple[int, int],
    right: tuple[int, int],
) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _action_zh(action: str) -> str:
    return {
        "UP": "向上",
        "DOWN": "向下",
        "LEFT": "向左",
        "RIGHT": "向右",
        "WAIT": "等待",
    }.get(str(action), str(action))


def _goal_label(goal: str, language: str) -> str:
    labels = {
        "pickup": ("认领取货", "claim pickup"),
        "delivery": ("完成配送", "deliver cargo"),
        "charge": ("充电", "charge"),
        "wait": ("等待", "wait"),
    }
    zh, en = labels.get(str(goal), (str(goal), str(goal)))
    return zh if language == "zh-CN" else en


def _post_charge_task_context(
    state: WarehouseState,
    agent: AgentState,
    environment: WarehouseMultiAgentEnv,
) -> dict[str, Any]:
    """Describe visible shared-task work after charging without exposing goals."""

    if agent.carrying_task_id:
        task = state.task_by_id(agent.carrying_task_id)
        return {
            "kind": "delivery",
            "task_id": task.task_id,
            "task_slot": _task_slot(state, task.task_id),
            "endpoint": task.delivery_position,
            "endpoint_kind": "B",
        }

    # Estimate the next pickup under the current shared-task state by giving
    # the selected robot enough energy and rerunning the environment's
    # deterministic assignment on a copy.  No navigation-goal field is placed
    # in participant-facing evidence.
    candidate = deepcopy(state)
    candidate_agent = candidate.by_id(agent.agent_id)
    candidate_agent.battery = 100.0
    environment._refresh_navigation_goals(candidate)
    if candidate_agent.navigation_goal_kind == "pickup":
        task = next(
            (
                item
                for item in candidate.tasks
                if item.status == "available"
                and item.pickup_position
                == candidate_agent.navigation_goal_position
            ),
            None,
        )
        if task is not None:
            return {
                "kind": "pickup",
                "task_id": task.task_id,
                "task_slot": _task_slot(state, task.task_id),
                "endpoint": task.pickup_position,
                "endpoint_kind": "A",
            }
    return {"kind": "shared_assignment"}


def _movement_work_context(
    state: WarehouseState,
    agent: AgentState,
    environment: WarehouseMultiAgentEnv,
) -> dict[str, Any]:
    """Bind a successful move to visible task or charging work."""

    # Charging takes precedence over cargo state.  A low-battery carrier may
    # deliberately move away from B on its way to the charger; describing that
    # move against the delivery endpoint would invert the observed progress
    # and give participants a false reason for the action.
    if agent.navigation_goal_kind == "charge":
        return {
            "kind": "charge",
            "endpoint": environment.layout.charger_position,
            "battery_before": float(agent.battery),
            "next_task": _post_charge_task_context(
                state,
                agent,
                environment,
            ),
        }
    if agent.carrying_task_id:
        task = state.task_by_id(agent.carrying_task_id)
        return {
            "kind": "delivery",
            "task_id": task.task_id,
            "task_slot": _task_slot(state, task.task_id),
            "endpoint": task.delivery_position,
            "endpoint_kind": "B",
        }
    if agent.navigation_goal_kind == "pickup":
        task = next(
            (
                item
                for item in state.tasks
                if item.status == "available"
                and item.pickup_position == agent.navigation_goal_position
            ),
            None,
        )
        if task is not None:
            return {
                "kind": "pickup",
                "task_id": task.task_id,
                "task_slot": _task_slot(state, task.task_id),
                "endpoint": task.pickup_position,
                "endpoint_kind": "A",
            }
    return {"kind": "reposition"}


def _task_slot(state: WarehouseState, task_id: str | None) -> int | None:
    for index, task in enumerate(state.tasks, start=1):
        if task.task_id == task_id:
            return index
    return None


def _agent_task_role(
    state: WarehouseState,
    agent: AgentState,
    *,
    charger_position: tuple[int, int],
) -> dict[str, Any]:
    """Return the visible shared-task role currently held by one robot."""

    if agent.carrying_task_id:
        task = state.task_by_id(agent.carrying_task_id)
        return {
            "kind": "delivery",
            "task_id": task.task_id,
            "task_slot": _task_slot(state, task.task_id),
            "endpoint": task.delivery_position,
            "endpoint_kind": "B",
        }
    if agent.navigation_goal_kind == "pickup":
        task = next(
            (
                item
                for item in state.tasks
                if item.status == "available"
                and item.pickup_position == agent.navigation_goal_position
            ),
            None,
        )
        if task is not None:
            return {
                "kind": "pickup",
                "task_id": task.task_id,
                "task_slot": _task_slot(state, task.task_id),
                "endpoint": task.pickup_position,
                "endpoint_kind": "A",
            }
    if agent.navigation_goal_kind == "charge":
        return {"kind": "charge", "endpoint": charger_position}
    return {"kind": "wait"}


def _energy_decision_context(
    state: WarehouseState,
    agent: AgentState,
    environment: WarehouseMultiAgentEnv,
    *,
    executed_action: str,
) -> dict[str, Any]:
    role = _agent_task_role(
        state,
        agent,
        charger_position=environment.layout.charger_position,
    )
    post_charge_role = _post_charge_task_context(state, agent, environment)
    task_id = str(role.get("task_id", "") or post_charge_role.get("task_id", ""))
    task = state.task_by_id(task_id) if task_id else None
    required_energy = None
    minimum_departure_battery = None
    charge_waits_remaining = 0
    projected_departure_battery = float(agent.battery)
    if task is not None:
        required_energy = (
            environment._mission_route_steps(
                state,
                agent,
                task,
                origin=agent.position,
            )
            * environment.config.move_battery_cost
        )
        minimum_departure_battery = (
            environment._mission_route_steps(
                state,
                agent,
                task,
                origin=environment.layout.charger_position,
            )
            * environment.config.move_battery_cost
        )
        travel_to_charger = shortest_path_distance(
            agent.position,
            environment.layout.charger_position,
            environment.config.map_layout_id,
        )
        battery_at_charger = max(
            0.0,
            float(agent.battery)
            - travel_to_charger * environment.config.move_battery_cost,
        )
        charge_waits_remaining = math.ceil(
            max(0.0, minimum_departure_battery - battery_at_charger)
            / environment.config.charge_per_wait
        )
        projected_departure_battery = min(
            100.0,
            battery_at_charger
            + charge_waits_remaining * environment.config.charge_per_wait,
        )
    task_role = role if role.get("kind") != "charge" else {
        **post_charge_role,
        "task_slot": _task_slot(state, task_id) if task_id else None,
    }
    return {
        "battery": float(agent.battery),
        "move_battery_cost": float(environment.config.move_battery_cost),
        "charge_per_wait": float(environment.config.charge_per_wait),
        "requires_charge": bool(environment._requires_charge(state, agent)),
        "required_safe_energy": required_energy,
        "minimum_safe_departure_battery": minimum_departure_battery,
        "charge_waits_remaining": int(charge_waits_remaining),
        "projected_departure_battery": projected_departure_battery,
        "at_charger": agent.position == environment.layout.charger_position,
        "executed_action": str(executed_action),
        "task_role": task_role,
    }


def _safe_assignment_breakdown(
    state: WarehouseState,
    agent: AgentState,
    task: Any,
    environment: WarehouseMultiAgentEnv,
) -> dict[str, Any]:
    """Expose the route and charging components behind one assignment cost."""

    origin = agent.position
    layout_id = environment.config.map_layout_id
    charger_position = environment.layout.charger_position
    current_to_delivery = shortest_path_distance(
        origin, task.delivery_position, layout_id
    )
    current_to_pickup = shortest_path_distance(
        origin, task.pickup_position, layout_id
    )
    pickup_to_delivery = shortest_path_distance(
        task.pickup_position,
        task.delivery_position,
        layout_id,
    )
    delivery_leg = (
        current_to_delivery
        if agent.carrying_task_id
        else current_to_pickup + pickup_to_delivery
    )
    return_leg = shortest_path_distance(
        task.delivery_position,
        charger_position,
        layout_id,
    )
    direct_travel = delivery_leg + return_leg
    direct_safe_steps = direct_travel + environment.config.mission_reserve_steps
    direct_energy = direct_safe_steps * environment.config.move_battery_cost
    if agent.battery >= direct_energy:
        route_legs = (
            (
                {"kind": "current_to_delivery", "cells": int(current_to_delivery)},
            )
            if agent.carrying_task_id
            else (
                {"kind": "current_to_pickup", "cells": int(current_to_pickup)},
                {"kind": "pickup_to_delivery", "cells": int(pickup_to_delivery)},
            )
        ) + ({"kind": "delivery_to_charger", "cells": int(return_leg)},)
        return {
            "agent_id": agent.agent_id,
            "task_id": task.task_id,
            "task_slot": _task_slot(state, task.task_id),
            "travel_cells": int(direct_travel),
            "charge_waits": 0,
            "charge_first": False,
            "current_battery": float(agent.battery),
            "route_legs": route_legs,
        }

    charger_distance = shortest_path_distance(origin, charger_position, layout_id)
    battery_at_charger = max(
        0.0,
        agent.battery
        - charger_distance * environment.config.move_battery_cost,
    )
    charger_to_delivery = shortest_path_distance(
        charger_position,
        task.delivery_position,
        layout_id,
    )
    charger_to_pickup = shortest_path_distance(
        charger_position,
        task.pickup_position,
        layout_id,
    )
    from_charger_delivery = (
        charger_to_delivery
        if agent.carrying_task_id
        else charger_to_pickup + pickup_to_delivery
    )
    charged_travel = charger_distance + from_charger_delivery + return_leg
    charged_safe_steps = (
        from_charger_delivery
        + return_leg
        + environment.config.mission_reserve_steps
    )
    energy_deficit = max(
        0.0,
        charged_safe_steps * environment.config.move_battery_cost
        - battery_at_charger,
    )
    charge_waits = math.ceil(
        energy_deficit / environment.config.charge_per_wait
    )
    route_legs = (
        {"kind": "current_to_charger", "cells": int(charger_distance)},
        *(
            ({"kind": "charger_to_delivery", "cells": int(charger_to_delivery)},)
            if agent.carrying_task_id
            else (
                {"kind": "charger_to_pickup", "cells": int(charger_to_pickup)},
                {"kind": "pickup_to_delivery", "cells": int(pickup_to_delivery)},
            )
        ),
        {"kind": "delivery_to_charger", "cells": int(return_leg)},
    )
    return {
        "agent_id": agent.agent_id,
        "task_id": task.task_id,
        "task_slot": _task_slot(state, task.task_id),
        "travel_cells": int(charged_travel),
        "charge_waits": int(charge_waits),
        "charge_first": True,
        "current_battery": float(agent.battery),
        "route_legs": route_legs,
    }


def _collaboration_context(
    state: WarehouseState,
    agent: AgentState,
    environment: WarehouseMultiAgentEnv,
    *,
    proposed_action: str,
    executed_action: str,
    executed_actions: Mapping[str, str],
    action_resolution: Mapping[str, Any],
) -> dict[str, Any]:
    teammate = next(item for item in state.agents if item.agent_id != agent.agent_id)
    target_role = _agent_task_role(
        state,
        agent,
        charger_position=environment.layout.charger_position,
    )
    teammate_role = _agent_task_role(
        state,
        teammate,
        charger_position=environment.layout.charger_position,
    )
    roles = {
        agent.agent_id: target_role,
        teammate.agent_id: teammate_role,
    }

    joint_selected_cost = None
    joint_swapped_cost = None
    joint_selected_breakdown = None
    joint_swapped_breakdown = None
    if (
        target_role.get("kind") == "pickup"
        and teammate_role.get("kind") == "pickup"
        and target_role.get("task_id") != teammate_role.get("task_id")
    ):
        target_task = state.task_by_id(str(target_role["task_id"]))
        teammate_task = state.task_by_id(str(teammate_role["task_id"]))
        joint_selected_cost = environment._safe_task_cost(
            state,
            agent,
            target_task,
            position=agent.position,
        ) + environment._safe_task_cost(
            state,
            teammate,
            teammate_task,
            position=teammate.position,
        )
        joint_swapped_cost = environment._safe_task_cost(
            state,
            agent,
            teammate_task,
            position=agent.position,
        ) + environment._safe_task_cost(
            state,
            teammate,
            target_task,
            position=teammate.position,
        )
        selected_entries = (
            _safe_assignment_breakdown(
                state,
                agent,
                target_task,
                environment,
            ),
            _safe_assignment_breakdown(
                state,
                teammate,
                teammate_task,
                environment,
            ),
        )
        swapped_entries = (
            _safe_assignment_breakdown(
                state,
                agent,
                teammate_task,
                environment,
            ),
            _safe_assignment_breakdown(
                state,
                teammate,
                target_task,
                environment,
            ),
        )
        joint_selected_breakdown = {
            "assignments": selected_entries,
            "total_travel_cells": sum(
                int(item["travel_cells"]) for item in selected_entries
            ),
            "total_charge_waits": sum(
                int(item["charge_waits"]) for item in selected_entries
            ),
        }
        joint_swapped_breakdown = {
            "assignments": swapped_entries,
            "total_travel_cells": sum(
                int(item["travel_cells"]) for item in swapped_entries
            ),
            "total_charge_waits": sum(
                int(item["charge_waits"]) for item in swapped_entries
            ),
        }

    constrained_actions: list[str] = []
    other_action = str(executed_actions.get(teammate.agent_id, "WAIT"))
    other_delta = MOVE_DELTAS.get(other_action, (0, 0))
    other_target = (
        teammate.position[0] + other_delta[0],
        teammate.position[1] + other_delta[1],
    )
    for action, delta in MOVE_DELTAS.items():
        candidate = (agent.position[0] + delta[0], agent.position[1] + delta[1])
        if (
            candidate == teammate.position
            or candidate == other_target
            or (candidate == teammate.position and other_target == agent.position)
        ):
            constrained_actions.append(action)

    return {
        "target_agent": agent.agent_id,
        "teammate_agent": teammate.agent_id,
        "target_role": target_role,
        "teammate_role": teammate_role,
        "roles": roles,
        "teammate_position": teammate.position,
        "teammate_distance": _manhattan(agent.position, teammate.position),
        "teammate_constrained_actions": tuple(dict.fromkeys(constrained_actions)),
        "teammate_executed_action": other_action,
        "proposed_action": str(proposed_action),
        "executed_action": str(executed_action),
        "teammate_directly_limited_action": bool(
            str(proposed_action) != str(executed_action)
            and str(action_resolution.get("blocked_reason", ""))
            in {"same_target", "swap", "occupied_stationary", "robot_collision"}
        ),
        "action_resolution": dict(action_resolution),
        "joint_selected_safe_actions": joint_selected_cost,
        "joint_swapped_safe_actions": joint_swapped_cost,
        "joint_selected_breakdown": joint_selected_breakdown,
        "joint_swapped_breakdown": joint_swapped_breakdown,
        "single_cargo_capacity": True,
    }


def _task_record(task: Any) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "pickup_position": task.pickup_position,
        "delivery_position": task.delivery_position,
        "status": task.status,
        "carrier_agent_id": task.carrier_agent_id,
        "created_frame": task.created_frame,
        "claimed_frame": task.claimed_frame,
        "delivered_frame": task.delivered_frame,
    }


def _task_state(state: WarehouseState) -> dict[str, Any]:
    return {
        "active_tasks": [_task_record(task) for task in state.tasks],
        "completed_task_count": len(state.completed_tasks),
        "total_deliveries": state.total_deliveries,
        "carrying_task_by_robot": {
            agent.agent_id: agent.carrying_task_id
            for agent in state.agents
        },
    }


def _charging_state(
    state: WarehouseState,
    charger_position: tuple[int, int],
) -> dict[str, Any]:
    occupant = next(
        (agent.agent_id for agent in state.agents if agent.position == charger_position),
        None,
    )
    return {
        "position": charger_position,
        "occupant_agent_id": occupant,
        "battery_by_robot": {
            agent.agent_id: float(agent.battery)
            for agent in state.agents
        },
    }


def _transition_events(info: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    events = [dict(item) for item in info.get("task_changes", ())]
    events.extend(
        dict(item)
        for item in info.get("coordination_events", ())
        if isinstance(item, Mapping)
    )
    events.extend(
        dict(item)
        for item in info.get("energy_events", ())
        if isinstance(item, Mapping)
    )
    if info.get("robot_collision_event"):
        events.append(
            {
                "event": "robot_collision",
                "agents": tuple(info.get("collisions", ())),
                "conflict_kind": info.get("robot_collision_kind"),
                "intended_targets": dict(info.get("intended_targets", {})),
                "proposed_actions": dict(info.get("proposed_actions", {})),
                "executed_actions": dict(info.get("executed_actions", {})),
            }
        )
    for agent_id in info.get("shutdowns", ()):
        events.append({"event": "battery_shutdown", "agent_id": agent_id})
    gained_by_agent = dict(info.get("charger_energy_gained_by_agent", {}))
    if gained_by_agent:
        events.extend(
            {
                "event": "charging",
                "agent_id": str(agent_id),
                "energy_gained": float(energy_gained),
            }
            for agent_id, energy_gained in sorted(gained_by_agent.items())
            if float(energy_gained) > 0.0
        )
    resolution = dict(info.get("action_resolution", {}))
    for agent_id, value in resolution.items():
        if value.get("environment_changed_action"):
            events.append(
                {
                    "event": "action_blocked",
                    "agent_id": agent_id,
                    "requested_action": value.get("requested_action"),
                    "executed_action": value.get("executed_action"),
                    "reason": value.get("blocked_reason"),
                }
            )
    return tuple(events)
