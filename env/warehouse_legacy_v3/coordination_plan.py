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


def _canonical_short_route(
    start: tuple[int, int],
    goal: tuple[int, int],
    config: WarehouseConfig,
    *,
    horizon: int,
) -> tuple[tuple[int, int], ...]:
    layout = get_map_layout(config.map_layout_id)
    current = start
    route: list[tuple[int, int]] = [current]
    for _ in range(max(0, int(horizon))):
        distance = shortest_path_distance(current, goal, config.map_layout_id)
        candidates: list[tuple[int, tuple[int, int]]] = []
        for action_index, action in enumerate(ACTIONS):
            delta = MOVE_DELTAS.get(action)
            if delta is None:
                continue
            target = (current[0] + delta[0], current[1] + delta[1])
            if (
                layout.is_passable(target)
                and shortest_path_distance(target, goal, config.map_layout_id)
                < distance
            ):
                candidates.append((action_index, target))
        if not candidates:
            break
        _, current = min(candidates)
        route.append(current)
        if current == goal:
            break
    return tuple(route)


def _task_energy_after(
    state: WarehouseState,
    config: WarehouseConfig,
    agent: Any,
    position: tuple[int, int],
) -> float | None:
    task_id = agent.route_commitment_task_id or agent.goal_id
    task = next(
        (
            item
            for item in state.tasks
            if item.task_id == task_id and item.status == "available"
        ),
        None,
    )
    if task is None:
        return None
    layout = get_map_layout(config.map_layout_id)
    steps = (
        shortest_path_distance(position, task.pickup_position, config.map_layout_id)
        + shortest_path_distance(
            task.pickup_position, task.delivery_position, config.map_layout_id
        )
        + shortest_path_distance(
            task.delivery_position,
            layout.charger_position,
            config.map_layout_id,
        )
        + config.mission_reserve_steps
    )
    return float(steps * config.move_battery_cost)


def _short_horizon_charger_plan(
    state: WarehouseState,
    config: WarehouseConfig,
    *,
    goals: Mapping[str, tuple[int, int]],
    priority: Any,
    horizon: int = 4,
) -> dict[str, Any] | None:
    """Return one joint reservation step before a charger-route conflict."""

    layout = get_map_layout(config.map_layout_id)
    priority_agent = state.by_id(priority.agent_id)
    priority_basis = str(priority.basis)
    if priority_basis == "charger_exit":
        approaching = tuple(
            agent
            for agent in state.agents
            if agent.agent_id != priority_agent.agent_id
            and goals.get(agent.agent_id) == layout.charger_position
            and shortest_path_distance(
                agent.position, layout.charger_position, config.map_layout_id
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
        goals.get(priority_agent.agent_id) != layout.charger_position
        or priority_agent.position == layout.charger_position
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
    yielding_goal = goals[yielding.agent_id]
    priority_route = _canonical_short_route(
        priority_agent.position,
        layout.charger_position,
        config,
        horizon=horizon + 1,
    )
    yielding_route = _canonical_short_route(
        yielding.position, yielding_goal, config, horizon=horizon
    )
    reserved = set(priority_route[1:])
    if not reserved.intersection(yielding_route[1:]):
        return None

    current_priority_distance = shortest_path_distance(
        priority_agent.position, layout.charger_position, config.map_layout_id
    )
    priority_actions: list[tuple[int, str, tuple[int, int]]] = []
    for action_index, action in enumerate(ACTIONS):
        delta = MOVE_DELTAS.get(action)
        if delta is None:
            continue
        target = (
            priority_agent.position[0] + delta[0],
            priority_agent.position[1] + delta[1],
        )
        if (
            layout.is_passable(target)
            and shortest_path_distance(
                target, layout.charger_position, config.map_layout_id
            )
            < current_priority_distance
        ):
            remaining = priority_agent.battery - config.move_battery_cost
            required = (
                shortest_path_distance(
                    target,
                    layout.charger_position,
                    config.map_layout_id,
                )
                * config.move_battery_cost
            )
            if remaining <= 0.0 or remaining + 1e-8 < required:
                continue
            priority_actions.append((action_index, action, target))
    if not priority_actions:
        return None
    _, priority_action, priority_target = min(priority_actions)

    candidates: list[tuple[int, int, int, str, tuple[int, int]]] = []
    for action_index, action in enumerate(ACTIONS):
        delta = MOVE_DELTAS.get(action)
        target = yielding.position
        if delta is not None:
            target = (
                yielding.position[0] + delta[0],
                yielding.position[1] + delta[1],
            )
            if not layout.is_passable(target) or yielding.battery <= config.move_battery_cost:
                continue
        if (
            target == priority_target
            or target == priority_agent.position
            or (
                priority_target == yielding.position
                and target == priority_agent.position
            )
        ):
            continue
        if target == layout.charger_position and yielding_goal != layout.charger_position:
            # The charger is the reserved destination, not a clearance cell.
            # Allow the yielding robot to use either apron, never the single
            # station itself.
            continue
        required = _task_energy_after(state, config, yielding, target)
        remaining = yielding.battery - (
            config.move_battery_cost if action in MOVE_DELTAS else 0.0
        )
        if required is not None and remaining + 1e-8 < required:
            continue
        candidate_route = _canonical_short_route(
            target, yielding_goal, config, horizon=horizon
        )
        first_overlap = next(
            (
                index
                for index, position in enumerate(candidate_route)
                if position in reserved
            ),
            horizon + 2,
        )
        candidates.append(
            (
                -first_overlap,
                shortest_path_distance(target, yielding_goal, config.map_layout_id),
                action_index,
                action,
                target,
            )
        )
    if not candidates:
        return None
    _, _, _, yielding_action, yielding_target = min(candidates)
    # Frozen-state simultaneous motion never assumes that a currently
    # occupied cell will be released later in the same joint command.  This
    # matters especially in Human-AI rounds: the participant's current-frame
    # action is private, so an AI may not enter the participant's cell merely
    # because the public plan asked the participant to leave it.  Split that
    # hand-off into a visible clearance step followed by charger progress.
    occupied_handoff = priority_target == yielding.position
    joint_actions = {
        priority_agent.agent_id: "WAIT" if occupied_handoff else priority_action,
        yielding.agent_id: yielding_action,
    }
    moving_agent = yielding if occupied_handoff else priority_agent
    moving_action = yielding_action if occupied_handoff else priority_action
    moving_target = yielding_target if occupied_handoff else priority_target
    return {
        "plan_id": (
            f"coord:{state.episode_id}:{state.frame}:horizon-charge:"
            f"{priority_agent.agent_id}:{yielding.agent_id}"
        ),
        "plan_kind": "short_horizon_charger_reservation",
        "phase": "CLEAR_CELL" if occupied_handoff else "JOINT_STEP",
        "priority_agent_id": priority_agent.agent_id,
        "yielding_agent_id": yielding.agent_id,
        "waiting_agent_id": (
            priority_agent.agent_id if occupied_handoff else yielding.agent_id
        ),
        "moving_agent_id": moving_agent.agent_id,
        "moving_action": moving_action,
        "moving_target": moving_target,
        "reserved_agent_action": yielding_action,
        "reserved_agent_target": yielding_target,
        "joint_actions": joint_actions,
        **(
            {
                "clearing_agent_id": yielding.agent_id,
                "occupied_position": yielding.position,
                "clearing_action": yielding_action,
                "clearing_target": yielding_target,
                "allowed_clearing_actions": (yielding_action,),
                "allowed_clearing_targets": (yielding_target,),
            }
            if occupied_handoff
            else {}
        ),
        "priority_basis": priority_basis,
        "priority_goal_id": None,
        "reason_code": "anticipated_charger_corridor_reservation",
        "lookahead_steps": horizon,
        "expected_duration_frames": 1,
        "completion_condition": (
            "priority_route_cell_cleared"
            if occupied_handoff
            else "joint_reservation_step_completed"
        ),
        "resume_condition": "short_routes_no_longer_overlap",
        "derived_from_frame": state.frame,
    }


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
        if phase in {"SINGLE_STEP", "JOINT_STEP"}:
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
            joint_actions = {
                agent.agent_id: (
                    moving_action
                    if agent.agent_id == priority_id
                    else "WAIT"
                )
                for agent in state.agents
            }
            return {
                **dict(stored),
                "plan_kind": "priority_followthrough",
                "phase": "PASS_THROUGH",
                "waiting_agent_id": clearing_id,
                "moving_agent_id": priority_id,
                "moving_action": moving_action,
                "moving_target": occupied_position,
                "yielding_agent_id": clearing_id,
                # A stored CLEAR_CELL plan carries the previous phase's
                # joint action.  Replace it atomically when roles change;
                # otherwise every consumer that correctly prioritizes the
                # public joint contract repeats the clearing move and the
                # supposed waiter walks away again.
                "joint_actions": joint_actions,
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
    archived_8x9 = (
        config.map_layout_id
        == "warehouse_staggered_aisles_8x9_v1_three_cell_exit"
    )
    if archived_8x9 and _independent_parallel_progress_exists(
        state, progress_actions
    ):
        return None
    priority = coordination_priority(
        state,
        config,
        goal_positions=goals,
        goal_kinds=kinds,
        requires_charge=requires_charge,
        imminent_head_on=imminent_head_on_encounter(state, config, goals),
    )
    participant_id = state.participant_controlled_agent_id
    if participant_id is not None and not archived_8x9:
        # Human-AI decisions cannot condition on the participant's private
        # current command.  When a loaded (or genuinely charger-critical) AI
        # has priority but every mission-progress cell could also be entered
        # by a currently submittable participant action, publish a one-step
        # right-of-way reservation from S_t.  The UI/guard removes only that
        # contested target from this frame's participant choices; all other
        # actions remain available.  Without this causal public reservation a
        # loaded AI can WAIT forever under worst-case safety even while the
        # participant visibly waits, as happened in acceptance seed 51054.
        ai_agent = next(
            (
                agent
                for agent in active
                if agent.agent_id != participant_id
            ),
            None,
        )
        participant = state.by_id(participant_id)
        priority_bases = {
            "loaded_delivery",
            "critical_charger_route",
            "urgent_charger_route",
            "lower_energy_charger_waiter",
            "charger_clearance_commitment",
        }
        if (
            ai_agent is not None
            and priority.agent_id == ai_agent.agent_id
            and (
                ai_agent.carrying_task_id is not None
                or str(priority.basis) in priority_bases
            )
        ):
            participant_mask = legal_action_mask(
                state,
                participant,
                config.map_layout_id,
            )
            participant_targets = {
                (
                    participant.position
                    if action == "WAIT"
                    else (
                        participant.position[0] + MOVE_DELTAS[action][0],
                        participant.position[1] + MOVE_DELTAS[action][1],
                    )
                )
                for action, allowed in zip(ACTIONS, participant_mask)
                if allowed > 0.5
            }
            ai_progress = tuple(
                item
                for item in progress_actions.get(ai_agent.agent_id, ())
                if item[2] != participant.position
            )
            uncontested = tuple(
                item for item in ai_progress if item[2] not in participant_targets
            )
            if ai_progress and not uncontested:
                _, moving_action, moving_target = min(ai_progress)
                priority_goal_id = (
                    ai_agent.carrying_task_id
                    or ai_agent.route_commitment_task_id
                    or ai_agent.goal_id
                )
                return {
                    "plan_id": (
                        f"coord:{state.episode_id}:{state.frame}:human-reserve:"
                        f"{ai_agent.agent_id}:{participant.agent_id}:"
                        f"{moving_target[0]}:{moving_target[1]}"
                    ),
                    "plan_kind": "participant_avoids_priority_cell",
                    "phase": "SINGLE_STEP",
                    "priority_agent_id": ai_agent.agent_id,
                    "waiting_agent_id": participant.agent_id,
                    "moving_agent_id": ai_agent.agent_id,
                    "moving_action": moving_action,
                    "moving_target": moving_target,
                    "yielding_agent_id": participant.agent_id,
                    "priority_basis": priority.basis,
                    "priority_goal_id": priority_goal_id,
                    "reason_code": "public_priority_cell_reservation",
                    "expected_duration_frames": 1,
                    "completion_condition": "priority_robot_advances",
                    "resume_condition": "reserved_cell_released_next_frame",
                    "derived_from_frame": state.frame,
                }
    # Do not turn a predicted route overlap into a one-frame right-of-way
    # contract.  Those speculative reservations were recreated on every
    # frame and could force a robot to leave its committed route, producing
    # exactly the DOWN->UP / LEFT->RIGHT cycles the public audit is meant to
    # reject.  The authoritative runtime evaluates all 25 atomic joint
    # actions and scores the resulting next-state bottleneck directly.  A
    # persistent plan is reserved for an occupied unique next cell or an
    # immediate same-target/head-on conflict below.
    # Do not manufacture a single-lane conflict from mere proximity.  If
    # both frozen goals have progress moves into different free cells, both
    # Actors can advance simultaneously without a right-of-way handshake.
    if _independent_parallel_progress_exists(state, progress_actions):
        return None
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
        if archived_8x9:
            # Preserve the archived study artifact's conservative contract;
            # it is not served by the new 6x7 causal joint runtime.
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
            real_shared_target = bool(
                progress_cells & clearing_reachable_targets
            )
        else:
            # A peer merely *could* enter one of these cells; that is not
            # evidence that the 6x7 frozen joint decision competes for it.
            # Only shared mission-progress targets warrant priority.
            real_shared_target = bool(shared_progress_targets)
        if not head_on and not real_shared_target:
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
        participant_id = state.participant_controlled_agent_id
        if (
            not archived_8x9
            and
            participant_id is not None
            and clearing.agent_id == participant_id
            and waiting.agent_id != participant_id
        ):
            # In Human-AI play the AI cannot assume that the participant will
            # obey a same-frame WAIT while it enters a contested cell.  Make
            # the clearance an observable first phase instead: the human
            # leaves every target that could conflict with the AI's next
            # move, then the AI passes on the following frame.  This preserves
            # S_t causality without trapping both sides in repeated WAITs.
            participant_clearance: list[
                tuple[int, int, int, str, tuple[int, int]]
            ] = []
            clearing_goal = goals[clearing.agent_id]
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
                    or target == moving_target
                    or clearing.battery <= config.move_battery_cost
                    or (
                        target == layout.charger_position
                        and clearing_goal != layout.charger_position
                    )
                ):
                    continue
                participant_clearance.append(
                    (
                        shortest_path_distance(
                            target,
                            clearing_goal,
                            config.map_layout_id,
                        ),
                        -shortest_path_distance(
                            target,
                            waiting.position,
                            config.map_layout_id,
                        ),
                        action_index,
                        action,
                        target,
                    )
                )
            if participant_clearance:
                _, _, _, clearing_action, clearing_target = min(
                    participant_clearance
                )
                plan_id = (
                    f"coord:{state.episode_id}:{state.frame}:human-clear:"
                    f"{waiting.agent_id}:{clearing.agent_id}"
                )
                return {
                    "plan_id": plan_id,
                    "plan_kind": "participant_clearance_before_ai_pass",
                    "phase": "CLEAR_CELL",
                    "priority_agent_id": waiting.agent_id,
                    "waiting_agent_id": waiting.agent_id,
                    "clearing_agent_id": clearing.agent_id,
                    "yielding_agent_id": clearing.agent_id,
                    "moving_agent_id": clearing.agent_id,
                    # The contested next cell is intentionally recorded here;
                    # PASS_THROUGH will move the AI into it only after the
                    # participant's clearance is visible in the next S_t.
                    "occupied_position": moving_target,
                    "clearing_action": clearing_action,
                    "clearing_target": clearing_target,
                    "allowed_clearing_actions": (clearing_action,),
                    "allowed_clearing_targets": (clearing_target,),
                    "moving_action": clearing_action,
                    "moving_target": clearing_target,
                    "priority_basis": priority.basis,
                    "priority_goal_id": priority_goal_id,
                    "reason_code": "unknown_participant_action_clearance",
                    "expected_duration_frames": 2,
                    "completion_condition": (
                        "participant_clears_contested_next_cell"
                    ),
                    "resume_condition": "ai_enters_observably_cleared_cell",
                    "derived_from_frame": state.frame,
                }
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
    agent_execution_aligned: dict[str, bool] = {}
    agent_request_aligned: dict[str, bool] = {}
    if isinstance(plan.get("joint_actions"), Mapping):
        expected_joint = {
            str(agent_id): str(action)
            for agent_id, action in dict(plan["joint_actions"]).items()
        }
        agent_execution_aligned = {
            agent_id: str(executed_actions.get(agent_id, "WAIT")) == action
            for agent_id, action in expected_joint.items()
        }
        agent_request_aligned = {
            agent_id: str(requested_actions.get(agent_id, "WAIT")) == action
            for agent_id, action in expected_joint.items()
        }
        aligned = all(agent_execution_aligned.values())
    elif str(plan.get("phase")) == "CLEAR_CELL":
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
        agent_execution_aligned = {
            waiting_id: str(executed_actions.get(waiting_id, "WAIT")) == "WAIT",
            moving_id: bool(
                str(executed_actions.get(moving_id, "WAIT")) in allowed_actions
                and tuple(intended_targets.get(moving_id, ())) in allowed_targets
            ),
        }
        agent_request_aligned = {
            waiting_id: str(requested_actions.get(waiting_id, "WAIT")) == "WAIT",
            moving_id: str(requested_actions.get(moving_id, "WAIT"))
            in allowed_actions,
        }
        aligned = all(agent_execution_aligned.values())
    else:
        agent_execution_aligned = {
            waiting_id: str(executed_actions.get(waiting_id, "WAIT")) == "WAIT",
            moving_id: bool(
                str(executed_actions.get(moving_id, "WAIT")) == expected_move
                and tuple(intended_targets.get(moving_id, ()))
                == tuple(plan["moving_target"])
            ),
        }
        agent_request_aligned = {
            waiting_id: str(requested_actions.get(waiting_id, "WAIT")) == "WAIT",
            moving_id: str(requested_actions.get(moving_id, "WAIT"))
            == expected_move,
        }
        aligned = all(agent_execution_aligned.values())
    return {
        "event": "joint_coordination_plan",
        **dict(plan),
        "requested_actions": dict(requested_actions),
        "executed_actions": dict(executed_actions),
        # Human-AI decisions are sampled before the participant's private
        # current-frame action exists.  Preserve whether each robot followed
        # its own causal instruction even when the other party deviates; the
        # joint flag remains the stricter all-parties contract.
        "agent_execution_aligned": agent_execution_aligned,
        "agent_request_aligned": agent_request_aligned,
        "execution_aligned": aligned,
        "completed": bool(
            aligned and str(plan.get("phase")) != "CLEAR_CELL"
        ),
    }
