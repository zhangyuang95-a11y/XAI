"""Short-horizon reservations for the offline coordination teacher.

These helpers derive one bounded joint instruction from the frozen pre-move
state.  Keeping them separate from the route-selection teacher makes the
reservation protocol independently testable and prevents ``coordination.py``
from becoming an unbounded collection of unrelated geometry rules.
"""

from __future__ import annotations

from typing import Any, Mapping

from .coordination_priority import single_lane_egress_agent_id
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from .transition_audit import action_is_robustly_safe


ARCHIVED_8X9_LAYOUT_ID = "warehouse_staggered_aisles_8x9_v1_three_cell_exit"


def canonical_shortest_route(
    environment: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    horizon: int,
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic short route prefix derived only from ``S_t``."""

    current = start
    route: list[tuple[int, int]] = [current]
    for _ in range(max(0, int(horizon))):
        current_distance = shortest_path_distance(
            current, goal, environment.config.map_layout_id
        )
        candidates: list[tuple[int, tuple[int, int]]] = []
        for action_index, action in enumerate(ACTIONS):
            delta = MOVE_DELTAS.get(action)
            if delta is None:
                continue
            target = (current[0] + delta[0], current[1] + delta[1])
            if (
                environment.layout.is_passable(target)
                and shortest_path_distance(
                    target, goal, environment.config.map_layout_id
                )
                < current_distance
            ):
                candidates.append((action_index, target))
        if not candidates:
            break
        _, current = min(candidates)
        route.append(current)
        if current == goal:
            break
    return tuple(route)


def short_horizon_charger_reservation_actions(
    environment: Any,
    *,
    goal_overrides: Mapping[str, tuple[int, int]],
    priority_agent: Any,
    priority_basis: str,
    horizon: int = 4,
) -> dict[str, str] | None:
    """Park a robot before it enters an approaching charger's bottleneck."""

    state = environment.get_state()
    charger = environment.layout.charger_position
    if priority_basis == "charger_exit":
        approaching = tuple(
            agent
            for agent in state.agents
            if agent.agent_id != priority_agent.agent_id
            and goal_overrides.get(agent.agent_id) == charger
            and shortest_path_distance(
                agent.position,
                charger,
                environment.config.map_layout_id,
            )
            <= horizon + 1
        )
        if len(approaching) != 1:
            return None
        priority_agent = approaching[0]
        priority_basis = "urgent_charger_route"
    if priority_basis not in {
        "critical_charger_route",
        "urgent_charger_route",
        "charger_route",
        "lower_energy_charger_waiter",
        "charger_clearance_commitment",
    }:
        return None
    if (
        goal_overrides.get(priority_agent.agent_id) != charger
        or priority_agent.position == charger
    ):
        return None
    yielding = next(
        agent for agent in state.agents if agent.agent_id != priority_agent.agent_id
    )
    if (
        yielding.carrying_task_id is not None
        and priority_basis != "critical_charger_route"
    ):
        return None

    yielding_goal = goal_overrides.get(
        yielding.agent_id, yielding.navigation_goal_position
    )
    priority_route = canonical_shortest_route(
        environment,
        priority_agent.position,
        charger,
        horizon=horizon + 1,
    )
    yielding_route = canonical_shortest_route(
        environment,
        yielding.position,
        yielding_goal,
        horizon=horizon,
    )
    reserved = set(priority_route[1:])
    if not reserved.intersection(yielding_route[1:]):
        return None

    priority_progress: list[tuple[int, str]] = []
    current_priority_distance = shortest_path_distance(
        priority_agent.position, charger, environment.config.map_layout_id
    )
    for action_index, action in enumerate(ACTIONS):
        delta = MOVE_DELTAS.get(action)
        if delta is None:
            continue
        target = (
            priority_agent.position[0] + delta[0],
            priority_agent.position[1] + delta[1],
        )
        if (
            environment.layout.is_passable(target)
            and shortest_path_distance(
                target, charger, environment.config.map_layout_id
            )
            < current_priority_distance
        ):
            remaining_battery = (
                priority_agent.battery - environment.config.move_battery_cost
            )
            required_after = (
                shortest_path_distance(
                    target,
                    charger,
                    environment.config.map_layout_id,
                )
                * environment.config.move_battery_cost
            )
            if remaining_battery <= 0.0 or remaining_battery + 1e-8 < required_after:
                continue
            priority_progress.append((action_index, action))
    if not priority_progress:
        return None
    _, priority_action = min(priority_progress)

    candidates: list[tuple[int, int, int, str]] = []
    for action_index, action in enumerate(ACTIONS):
        delta = MOVE_DELTAS.get(action)
        target = yielding.position
        if delta is not None:
            target = (
                yielding.position[0] + delta[0],
                yielding.position[1] + delta[1],
            )
            if (
                not environment.layout.is_passable(target)
                or yielding.battery <= environment.config.move_battery_cost
            ):
                continue
        if target == charger and yielding_goal != charger:
            continue
        trial = {
            priority_agent.agent_id: priority_action,
            yielding.agent_id: action,
        }
        _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
        if invalid or collision:
            continue
        committed_task = next(
            (
                task
                for task in state.tasks
                if task.task_id
                == (yielding.route_commitment_task_id or yielding.goal_id)
                and task.status == "available"
            ),
            None,
        )
        if committed_task is not None:
            remaining_battery = yielding.battery - (
                environment.config.move_battery_cost
                if action in MOVE_DELTAS
                else 0.0
            )
            required_after = (
                environment._mission_route_steps(
                    state,
                    yielding,
                    committed_task,
                    origin=target,
                )
                * environment.config.move_battery_cost
            )
            if remaining_battery + 1e-8 < required_after:
                continue
        candidate_route = canonical_shortest_route(
            environment, target, yielding_goal, horizon=horizon
        )
        first_overlap = next(
            (
                index
                for index, position in enumerate(candidate_route)
                if position in reserved
            ),
            horizon + 2,
        )
        mission_distance = shortest_path_distance(
            target, yielding_goal, environment.config.map_layout_id
        )
        reverse = int(
            yielding.last_executed_action in MOVE_DELTAS
            and MOVE_DELTAS.get(action)
            == tuple(
                -value for value in MOVE_DELTAS[yielding.last_executed_action]
            )
        )
        candidates.append(
            (
                -first_overlap,
                mission_distance,
                reverse * len(ACTIONS) + action_index,
                action,
            )
        )
    if not candidates:
        return None
    _, _, _, yielding_action = min(candidates)
    return {
        priority_agent.agent_id: priority_action,
        yielding.agent_id: yielding_action,
    }


def single_lane_egress_actions(
    environment: Any,
    *,
    goal_overrides: Mapping[str, tuple[int, int]],
    priority_basis: str,
) -> dict[str, str] | None:
    """Return one causal phase of a shelf-end egress handshake."""

    if priority_basis != "single_lane_egress":
        return None
    state = environment.get_state()
    layout = environment.layout
    layout_id = environment.config.map_layout_id
    egress_id = single_lane_egress_agent_id(
        state,
        environment.config,
        goal_positions=goal_overrides,
    )
    if egress_id is None:
        return None
    egress = state.by_id(egress_id)
    clearing = next(agent for agent in state.agents if agent.agent_id != egress_id)
    spine_column = layout.charger_position[1]
    inward_action = "RIGHT" if egress.position[1] < spine_column else "LEFT"
    held = {agent.agent_id: "WAIT" for agent in state.agents}
    inward_delta = MOVE_DELTAS[inward_action]
    inward_target = (
        egress.position[0] + inward_delta[0],
        egress.position[1] + inward_delta[1],
    )
    trial = dict(held)
    trial[egress.agent_id] = inward_action
    _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
    can_advance = bool(
        inward_target != clearing.position
        and not collision
        and egress.agent_id not in invalid
    )
    if can_advance and layout_id == ARCHIVED_8X9_LAYOUT_ID:
        participant_priority = bool(
            state.participant_controlled_agent_id == egress.agent_id
        )
        if participant_priority or action_is_robustly_safe(
            environment,
            state,
            held,
            egress.agent_id,
            inward_action,
        ):
            return {
                agent.agent_id: (
                    inward_action if agent.agent_id == egress.agent_id else "WAIT"
                )
                for agent in state.agents
            }
    elif can_advance:
        joint = {
            agent.agent_id: (
                inward_action if agent.agent_id == egress.agent_id else "WAIT"
            )
            for agent in state.agents
        }
        clearing_goal = goal_overrides.get(
            clearing.agent_id, clearing.navigation_goal_position
        )
        clearing_distance = shortest_path_distance(
            clearing.position,
            clearing_goal,
            layout_id,
        )
        parallel: list[tuple[int, str]] = []
        for action_index, action in enumerate(ACTIONS):
            delta = MOVE_DELTAS.get(action)
            if delta is None:
                continue
            target = (
                clearing.position[0] + delta[0],
                clearing.position[1] + delta[1],
            )
            if (
                not layout.is_passable(target)
                or shortest_path_distance(target, clearing_goal, layout_id)
                >= clearing_distance
                or clearing.battery <= environment.config.move_battery_cost
            ):
                continue
            trial = dict(joint)
            trial[clearing.agent_id] = action
            _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
            if not invalid and not collision:
                parallel.append((action_index, action))
        if parallel:
            _, joint[clearing.agent_id] = min(parallel)
        return joint

    available_pickups = {
        task.pickup_position for task in state.tasks if task.status == "available"
    }
    clearance: list[tuple[int, int, int, int, str]] = []
    clearing_mask = environment.action_masks()[clearing.agent_id]
    current_spine_distance = abs(clearing.position[1] - spine_column)
    for action_index, (action, allowed) in enumerate(zip(ACTIONS, clearing_mask)):
        if allowed <= 0.5 or action not in MOVE_DELTAS:
            continue
        delta = MOVE_DELTAS[action]
        target = (
            clearing.position[0] + delta[0],
            clearing.position[1] + delta[1],
        )
        target_spine_distance = abs(target[1] - spine_column)
        moves_toward_or_off_spine = bool(
            target_spine_distance < current_spine_distance
            or current_spine_distance == 0
        )
        trial = dict(held)
        trial[clearing.agent_id] = action
        _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
        if (
            not moves_toward_or_off_spine
            or (target in available_pickups and clearing.carrying_task_id is not None)
            or clearing.battery <= environment.config.move_battery_cost
            or collision
            or clearing.agent_id in invalid
        ):
            continue
        clearance.append(
            (
                int(
                    clearing.last_executed_action in MOVE_DELTAS
                    and action != clearing.last_executed_action
                ),
                -shortest_path_distance(target, egress.position, layout_id),
                target_spine_distance,
                action_index,
                action,
            )
        )
    if clearance:
        _, _, _, _, action = min(clearance)
        return {
            agent.agent_id: (
                action if agent.agent_id == clearing.agent_id else "WAIT"
            )
            for agent in state.agents
        }
    return held
