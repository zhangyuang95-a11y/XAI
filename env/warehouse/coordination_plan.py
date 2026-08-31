"""Frozen-state joint coordination plans used by traces and explanations.

The plan is evidence, not a post-policy controller: it is computed before the
current actions exist and never rewrites them.  Independent Actors can observe
the same priority features and align with the plan; the transition audit then
records whether both selected actions actually did so.
"""

from __future__ import annotations

from typing import Any, Mapping

from .coordination_priority import (
    coordination_priority,
    imminent_head_on_encounter,
)
from .domain import WarehouseConfig, WarehouseState
from .layouts import get_map_layout
from .navigation import (
    ACTIONS,
    MOVE_DELTAS,
    legal_action_mask,
    shortest_path_distance,
)


def _goal_progress_actions(
    state: WarehouseState,
    config: WarehouseConfig,
    *,
    goals: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[tuple[int, str, tuple[int, int]], ...]]:
    """Return each robot's legal one-step shortest-route progress actions."""

    layout = get_map_layout(config.map_layout_id)
    result: dict[str, tuple[tuple[int, str, tuple[int, int]], ...]] = {}
    for agent in state.agents:
        goal = goals[agent.agent_id]
        current_distance = shortest_path_distance(
            agent.position,
            goal,
            config.map_layout_id,
        )
        candidates: list[tuple[int, str, tuple[int, int]]] = []
        for action_index, action in enumerate(ACTIONS):
            delta = MOVE_DELTAS.get(action)
            if delta is None:
                continue
            target = (
                agent.position[0] + delta[0],
                agent.position[1] + delta[1],
            )
            if not layout.is_passable(target):
                continue
            if shortest_path_distance(
                target,
                goal,
                config.map_layout_id,
            ) < current_distance:
                candidates.append((action_index, action, target))
        result[agent.agent_id] = tuple(candidates)
    return result


def _independent_parallel_progress_exists(
    state: WarehouseState,
    progress: Mapping[str, tuple[tuple[int, str, tuple[int, int]], ...]],
) -> bool:
    """Whether both robots can advance into distinct, initially free cells.

    A right-of-way plan is unnecessary when two mission-progress moves are
    already independent.  Requiring initially free targets preserves the
    conservative frozen-state hand-off used when one robot needs the other's
    currently occupied cell; that case receives an explicit clearance plan.
    """

    active = tuple(agent for agent in state.agents if agent.active)
    if len(active) != 2:
        return False
    first, second = active
    for _, _, first_target in progress.get(first.agent_id, ()):
        for _, _, second_target in progress.get(second.agent_id, ()):
            if (
                first_target != second_target
                and first_target != second.position
                and second_target != first.position
            ):
                return True
    return False


def frozen_joint_coordination_plan(
    state: WarehouseState,
    config: WarehouseConfig,
    *,
    requires_charge: Mapping[str, bool],
) -> dict[str, Any] | None:
    """Return one non-contradictory occupied-route clearance plan for ``S_t``."""

    active = tuple(agent for agent in state.agents if agent.active)
    if len(active) != 2:
        return None
    stored = state.active_coordination_plan
    if stored is not None:
        priority_agent = state.by_id(str(stored["priority_agent_id"]))
        current_priority_goal_id = (
            priority_agent.carrying_task_id
            or priority_agent.route_commitment_task_id
            or priority_agent.goal_id
        )
        planned_priority_goal_id = stored.get("priority_goal_id")
        if (
            planned_priority_goal_id is not None
            and current_priority_goal_id != planned_priority_goal_id
        ):
            # A clearance plan cannot force follow-through toward a task that
            # was claimed or completed during its first phase.
            return None
        phase = str(stored.get("phase", "CLEAR_CELL"))
        if phase == "CLEAR_CELL":
            return dict(stored)
        if phase == "SINGLE_STEP":
            return dict(stored)
        if phase == "PASS_THROUGH":
            priority_id = str(stored["priority_agent_id"])
            clearing_id = str(stored["clearing_agent_id"])
            priority_agent = state.by_id(priority_id)
            occupied_position = tuple(stored["occupied_position"])
            delta = (
                occupied_position[0] - priority_agent.position[0],
                occupied_position[1] - priority_agent.position[1],
            )
            moving_action = next(
                (
                    action
                    for action, action_delta in MOVE_DELTAS.items()
                    if action_delta == delta
                ),
                None,
            )
            if moving_action is None:
                return None
            return {
                **dict(stored),
                "plan_kind": "priority_followthrough",
                "phase": "PASS_THROUGH",
                "waiting_agent_id": clearing_id,
                "moving_agent_id": priority_id,
                "moving_action": moving_action,
                "moving_target": occupied_position,
                "yielding_agent_id": clearing_id,
                "reason_code": "priority_route_followthrough",
                "expected_duration_frames": 1,
                "completion_condition": "priority_robot_enters_cleared_route",
                "resume_condition": "priority_robot_has_passed_clearing_robot",
            }
        return None
    layout = get_map_layout(config.map_layout_id)
    charger_handoff_needed = bool(
        any(
            agent.position == layout.charger_position
            for agent in active
        )
        and any(
            agent.position != layout.charger_position
            and requires_charge.get(agent.agent_id, False)
            and shortest_path_distance(
                agent.position,
                layout.charger_position,
                config.map_layout_id,
            )
            == 1
            for agent in active
        )
    )
    goals: dict[str, tuple[int, int]] = {}
    kinds: dict[str, str] = {}
    for agent in active:
        if agent.navigation_goal_kind == "charge":
            goals[agent.agent_id] = agent.navigation_goal_position
            kinds[agent.agent_id] = "charge"
        elif agent.carrying_task_id is not None:
            goals[agent.agent_id] = state.task_by_id(
                agent.carrying_task_id
            ).delivery_position
            kinds[agent.agent_id] = "delivery"
        elif (
            agent.route_commitment_task_id is not None
            or (
                agent.goal_type == "GO_TO_PICKUP"
                and agent.goal_id is not None
            )
        ):
            task_id = agent.route_commitment_task_id or agent.goal_id
            committed = next(
                (
                    task
                    for task in state.tasks
                    if task.task_id == task_id
                    and task.status == "available"
                ),
                None,
            )
            if committed is not None:
                goals[agent.agent_id] = committed.pickup_position
                kinds[agent.agent_id] = "pickup"
                continue
            goals[agent.agent_id] = agent.navigation_goal_position
            kinds[agent.agent_id] = agent.navigation_goal_kind
        else:
            goals[agent.agent_id] = agent.navigation_goal_position
            kinds[agent.agent_id] = agent.navigation_goal_kind
    progress_actions = _goal_progress_actions(
        state,
        config,
        goals=goals,
    )
    active_progress_targets = {
        agent.agent_id: {
            target
            for _, _, target in progress_actions.get(agent.agent_id, ())
        }
        for agent in active
    }
    shared_progress_targets = set.intersection(
        *(active_progress_targets[agent.agent_id] for agent in active)
    )
    # Do not manufacture a single-lane conflict from mere proximity.  If
    # both frozen goals have progress moves into different free cells, both
    # Actors can advance simultaneously without a right-of-way handshake.
    if _independent_parallel_progress_exists(state, progress_actions):
        return None
    priority = coordination_priority(
        state,
        config,
        goal_positions=goals,
        goal_kinds=kinds,
        requires_charge=requires_charge,
        imminent_head_on=imminent_head_on_encounter(state, config, goals),
    )
    # A robot whose only progress cell is occupied must wait until that cell
    # is visibly cleared.  When the occupant is also the frozen priority robot
    # and can progress away, retain the causal plan for exactly that clearing
    # step instead of dropping the reason one frame too early.
    for blocked in active:
        occupant = next(
            agent for agent in active if agent.agent_id != blocked.agent_id
        )
        blocked_targets = {
            target
            for _, _, target in progress_actions.get(blocked.agent_id, ())
        }
        occupant_progress = tuple(
            item
            for item in progress_actions.get(occupant.agent_id, ())
            if item[2] != blocked.position
        )
        if (
            blocked_targets != {occupant.position}
            or not occupant_progress
            or priority.agent_id != occupant.agent_id
        ):
            continue
        _, moving_action, moving_target = min(occupant_progress)
        priority_goal_id = (
            occupant.carrying_task_id
            or occupant.route_commitment_task_id
            or occupant.goal_id
        )
        return {
            "plan_id": (
                f"coord:{state.episode_id}:{state.frame}:release:"
                f"{occupant.agent_id}:{blocked.agent_id}"
            ),
            "plan_kind": "occupied_route_release",
            "phase": "SINGLE_STEP",
            "priority_agent_id": occupant.agent_id,
            "waiting_agent_id": blocked.agent_id,
            "moving_agent_id": occupant.agent_id,
            "moving_action": moving_action,
            "moving_target": moving_target,
            "yielding_agent_id": blocked.agent_id,
            "occupied_position": occupant.position,
            "priority_basis": priority.basis,
            "priority_goal_id": priority_goal_id,
            "reason_code": "occupied_route_release",
            "expected_duration_frames": 1,
            "completion_condition": "occupied_position_cleared",
            "resume_condition": "waiting_robot_route_cell_is_free",
            "derived_from_frame": state.frame,
        }
    waiting = state.by_id(priority.agent_id)
    clearing = next(
        agent for agent in active if agent.agent_id != waiting.agent_id
    )
    priority_goal_id = (
        waiting.carrying_task_id
        or waiting.route_commitment_task_id
        or waiting.goal_id
    )
    goal = goals[waiting.agent_id]
    progress_cells = {
        target
        for _, _, target in progress_actions.get(waiting.agent_id, ())
    }
    head_on = imminent_head_on_encounter(state, config, goals)
    occupied_progress = clearing.position in progress_cells
    uniquely_blocked = occupied_progress and progress_cells == {clearing.position}
    if (
        state.frame < state.coordination_plan_cooldown_until
        and not uniquely_blocked
        and not charger_handoff_needed
        and not head_on
        and not shared_progress_targets
    ):
        return None
    if not uniquely_blocked:
        clearing_static_mask = legal_action_mask(
            state,
            clearing,
            config.map_layout_id,
        )
        clearing_reachable_targets = {
            (
                clearing.position
                if action == "WAIT"
                else (
                    clearing.position[0] + MOVE_DELTAS[action][0],
                    clearing.position[1] + MOVE_DELTAS[action][1],
                )
            )
            for action, allowed in zip(ACTIONS, clearing_static_mask)
            if allowed > 0.5
        }
        potential_same_target_conflict = bool(
            progress_cells & clearing_reachable_targets
        )
        if not head_on and not potential_same_target_conflict:
            return None
        progress_actions: list[tuple[int, str, tuple[int, int]]] = []
        for action_index, action in enumerate(ACTIONS):
            delta = MOVE_DELTAS.get(action)
            if delta is None:
                continue
            target = (
                waiting.position[0] + delta[0],
                waiting.position[1] + delta[1],
            )
            if target in progress_cells and target != clearing.position:
                progress_actions.append((action_index, action, target))
        if not progress_actions:
            return None
        _, moving_action, moving_target = min(progress_actions)
        plan_id = (
            f"coord:{state.episode_id}:{state.frame}:head_on:"
            f"{waiting.agent_id}:{clearing.agent_id}"
        )
        return {
            "plan_id": plan_id,
            "plan_kind": (
                "head_on_priority"
                if head_on
                else "same_target_priority"
            ),
            "phase": "SINGLE_STEP",
            "priority_agent_id": waiting.agent_id,
            "waiting_agent_id": clearing.agent_id,
            "moving_agent_id": waiting.agent_id,
            "moving_action": moving_action,
            "moving_target": moving_target,
            "yielding_agent_id": clearing.agent_id,
            "priority_basis": priority.basis,
            "priority_goal_id": priority_goal_id,
            "reason_code": (
                "head_on_priority_passage"
                if head_on
                else "same_target_priority_passage"
            ),
            "expected_duration_frames": 1,
            "completion_condition": "priority_robot_advances",
            "resume_condition": "robots_no_longer_approaching_head_on",
            "derived_from_frame": state.frame,
        }

    clearing_goal = goals[clearing.agent_id]
    downstream_distance = shortest_path_distance(
        clearing.position,
        goal,
        config.map_layout_id,
    )
    downstream_progress_cells = {
        candidate
        for delta in MOVE_DELTAS.values()
        if layout.is_passable(
            candidate := (
                clearing.position[0] + delta[0],
                clearing.position[1] + delta[1],
            )
        )
        and shortest_path_distance(
            candidate,
            goal,
            config.map_layout_id,
        )
        < downstream_distance
    }
    candidates: list[tuple[int, int, int, str, tuple[int, int]]] = []
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
            or target == waiting.position
            or target == clearing.position
        ):
            continue
        candidates.append(
            (
                int(target in downstream_progress_cells),
                shortest_path_distance(
                    target,
                    clearing_goal,
                    config.map_layout_id,
                ),
                action_index,
                action,
                target,
            )
        )
    if not candidates:
        return None
    _, _, _, clearing_action, clearing_target = min(candidates)
    allowed_clearing_actions = tuple(item[3] for item in candidates)
    allowed_clearing_targets = tuple(item[4] for item in candidates)
    critical_bases = {
        "critical_charger_route",
        "urgent_charger_route",
        "lower_energy_charger_waiter",
        "charger_clearance_commitment",
    }
    reason_code = (
        "critical_charger_route_clearance"
        if priority.basis in critical_bases
        else "occupied_route_clearance"
    )
    plan_id = (
        f"coord:{state.episode_id}:{state.frame}:"
        f"{waiting.agent_id}:{clearing.agent_id}:{clearing.position[0]}:"
        f"{clearing.position[1]}"
    )
    return {
        "plan_id": plan_id,
        "plan_kind": "occupied_route_clearance",
        "phase": "CLEAR_CELL",
        "priority_agent_id": waiting.agent_id,
        "waiting_agent_id": waiting.agent_id,
        "clearing_agent_id": clearing.agent_id,
        "yielding_agent_id": clearing.agent_id,
        "moving_agent_id": clearing.agent_id,
        "occupied_position": clearing.position,
        "clearing_action": clearing_action,
        "clearing_target": clearing_target,
        "allowed_clearing_actions": allowed_clearing_actions,
        "allowed_clearing_targets": allowed_clearing_targets,
        "moving_action": clearing_action,
        "moving_target": clearing_target,
        "priority_basis": priority.basis,
        "priority_goal_id": priority_goal_id,
        "reason_code": reason_code,
        "expected_duration_frames": 1,
        "completion_condition": "occupied_position_cleared",
        "resume_condition": "priority_robot_enters_cleared_route",
        "derived_from_frame": state.frame,
    }


def coordination_plan_execution_event(
    plan: Mapping[str, Any] | None,
    *,
    requested_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
    intended_targets: Mapping[str, tuple[int, int]],
) -> dict[str, Any] | None:
    """Audit whether a pre-action plan was jointly followed this transition."""

    if plan is None:
        return None
    waiting_id = str(plan["waiting_agent_id"])
    moving_id = str(plan["moving_agent_id"])
    expected_move = str(plan["moving_action"])
    if str(plan.get("phase")) == "CLEAR_CELL":
        allowed_actions = {
            str(action)
            for action in plan.get(
                "allowed_clearing_actions",
                (expected_move,),
            )
        }
        allowed_targets = {
            tuple(target)
            for target in plan.get(
                "allowed_clearing_targets",
                (plan["moving_target"],),
            )
        }
        aligned = bool(
            str(executed_actions.get(waiting_id, "WAIT")) == "WAIT"
            and str(executed_actions.get(moving_id, "WAIT")) in allowed_actions
            and tuple(intended_targets.get(moving_id, ())) in allowed_targets
        )
    else:
        aligned = bool(
            str(executed_actions.get(waiting_id, "WAIT")) == "WAIT"
            and str(executed_actions.get(moving_id, "WAIT")) == expected_move
            and tuple(intended_targets.get(moving_id, ()))
            == tuple(plan["moving_target"])
        )
    return {
        "event": "joint_coordination_plan",
        **dict(plan),
        "requested_actions": dict(requested_actions),
        "executed_actions": dict(executed_actions),
        "execution_aligned": aligned,
        "completed": bool(
            aligned and str(plan.get("phase")) != "CLEAR_CELL"
        ),
    }
