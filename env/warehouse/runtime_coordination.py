"""Authoritative frozen-state runtime coordination for the public study.

The neural Actors still provide the policy proposal.  This module turns that
proposal into one causally safe public decision using only ``S_t``.  AI-AI
rounds optimize the complete 5x5 joint action set.  Human-AI rounds choose
Robot 2 before the participant command is known and prove the choice against
every action the participant may causally submit from the same state.
"""

from __future__ import annotations

from typing import Any, Mapping

from .coordination import (
    stable_coordination_actions,
)
from .frozen_missions import frozen_training_missions
from .energy_management import (
    charger_departure_progress,
    charger_handoff_clearance_action,
)
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from .transition_audit import (
    necessary_participant_standoff_clearance,
    necessary_teammate_route_clearance,
)


def _target(agent: Any, action: str) -> tuple[int, int]:
    delta = MOVE_DELTAS.get(action)
    if delta is None:
        return agent.position
    return (agent.position[0] + delta[0], agent.position[1] + delta[1])


def causal_participant_actions(environment: Any) -> tuple[str, ...]:
    """Statically legal actions a participant may choose from ``S_t``.

    Teammate occupancy and right-of-way reservations deliberately do not mask
    participant commands.  In a simultaneous Human-AI round, the participant
    owns Robot 1's decision: a command that conflicts with Robot 2 must reach
    the environment unchanged so the atomic joint-motion resolver can record
    the collision and apply its score penalty.  Only walls and map boundaries
    are excluded here when Robot 2 evaluates counterfactual human actions.
    """

    state = environment.get_state()
    participant_id = environment.config.human_agent_id
    result = [
        action
        for action, allowed in zip(
            ACTIONS,
            environment.action_masks()[participant_id],
        )
        if allowed > 0.5
    ]
    return tuple(result or ("WAIT",))


def guard_participant_action(
    environment: Any,
    requested_action: str,
) -> tuple[str, dict[str, Any]]:
    """Preserve every recognized participant command for atomic resolution.

    This function remains as an evidence boundary for callers, but it is not
    a collision-avoidance filter.  Static impossibilities are resolved by the
    environment (and reported as invalid moves); robot conflicts are resolved
    as collisions with the configured score penalty.
    """

    requested = str(requested_action)
    selected = requested if requested in ACTIONS else "WAIT"
    statically_legal = causal_participant_actions(environment)
    state = environment.get_state()
    return selected, {
        "requested_action": requested,
        "selected_action": selected,
        "legal_actions": list(statically_legal),
        "statically_legal": requested in statically_legal,
        "collision_protection_applied": False,
        "blocked": selected != requested,
        "blocked_reason": (
            "unknown_participant_action"
            if selected != requested
            else None
        ),
        "decision_state_positions": {
            agent.agent_id: list(agent.position) for agent in state.agents
        },
    }


def _goal_positions(environment: Any) -> dict[str, tuple[int, int]]:
    state = environment.get_state()
    missions = frozen_training_missions(environment, state)
    return {
        agent.agent_id: (
            missions[agent.agent_id].goal_position
            if missions.get(agent.agent_id) is not None
            else agent.navigation_goal_position
        )
        for agent in state.agents
    }


def _progress_positions_from(
    environment: Any,
    position: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    distance = shortest_path_distance(
        position, goal, environment.config.map_layout_id
    )
    return tuple(
        target
        for delta in MOVE_DELTAS.values()
        if environment.layout.is_passable(
            target := (position[0] + delta[0], position[1] + delta[1])
        )
        and shortest_path_distance(
            target, goal, environment.config.map_layout_id
        )
        < distance
    )


def _canonical_progress_route(
    environment: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    horizon: int,
) -> tuple[tuple[int, int], ...]:
    current = start
    route: list[tuple[int, int]] = []
    for _ in range(max(0, int(horizon))):
        candidates = _progress_positions_from(environment, current, goal)
        if not candidates:
            break
        current = min(candidates)
        route.append(current)
        if current == goal:
            break
    return tuple(route)


def _will_need_charge_after_delivery(
    environment: Any,
    state: Any,
    agent: Any,
    position_after_action: tuple[int, int],
    action: str,
) -> bool:
    """Predict a carried task's observable post-delivery charge transition."""

    if agent.carrying_task_id is None:
        return False
    task = state.task_by_id(agent.carrying_task_id)
    delivery_steps = shortest_path_distance(
        position_after_action,
        task.delivery_position,
        environment.config.map_layout_id,
    )
    projected_battery = float(
        agent.battery
        - (environment.config.move_battery_cost if action in MOVE_DELTAS else 0.0)
        - delivery_steps * environment.config.move_battery_cost
    )
    available = tuple(item for item in state.tasks if item.status == "available")
    if not available:
        required = (
            shortest_path_distance(
                task.delivery_position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            + environment.config.mission_reserve_steps
        ) * environment.config.move_battery_cost
        return projected_battery + 1e-8 < required
    required = min(
        (
            shortest_path_distance(
                task.delivery_position,
                candidate.pickup_position,
                environment.config.map_layout_id,
            )
            + shortest_path_distance(
                candidate.pickup_position,
                candidate.delivery_position,
                environment.config.map_layout_id,
            )
            + shortest_path_distance(
                candidate.delivery_position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            + environment.config.mission_reserve_steps
        )
        * environment.config.move_battery_cost
        for candidate in available
    )
    return projected_battery + 1e-8 < required


def _committed_goal(environment: Any, state: Any, agent: Any) -> tuple[int, int]:
    """Return the exact S_t mission consumed later by audit and explanation."""

    mission = frozen_training_missions(environment, state).get(agent.agent_id)
    return (
        mission.goal_position
        if mission is not None
        else agent.navigation_goal_position
    )


def _joint_candidate_evidence(
    environment: Any,
    *,
    policy_actions: Mapping[str, str],
    optimizer_actions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate all joint actions and attach a transparent lexicographic score."""

    state = environment.get_state()
    goals = _goal_positions(environment)
    charging_agents = tuple(
        agent
        for agent in state.agents
        if environment._requires_charge(state, agent)
    )
    lowest_energy_charger = (
        min(charging_agents, key=lambda agent: (agent.battery, agent.agent_id))
        if charging_agents
        else None
    )
    safe: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    left_id, right_id = environment.agent_ids
    for left_index, left_action in enumerate(ACTIONS):
        for right_index, right_action in enumerate(ACTIONS):
            actions = {left_id: left_action, right_id: right_action}
            targets, _, invalid, collision, collision_kind, _ = (
                environment._resolve_motion(state, actions)
            )
            if invalid or collision:
                rejected.append(
                    {
                        "actions": dict(actions),
                        "reason": (
                            f"collision:{collision_kind}"
                            if collision
                            else "invalid_static_move"
                        ),
                        "invalid_agents": sorted(invalid),
                    }
                )
                continue
            progress_count = 0
            commitment_regressions = 0
            loaded_regressions = 0
            loaded_progress = 0
            charging_regressions = 0
            charging_progress = 0
            low_energy_priority_regressions = 0
            low_energy_priority_progress = 0
            charger_access_blocking = 0
            route_clearance_failures = 0
            future_head_on_bottlenecks = 0
            anticipated_charger_route_blocking = 0
            total_distance = 0
            waits = 0
            reversals = 0
            short_cycles = 0
            policy_changes = 0
            distances: dict[str, dict[str, int]] = {}
            for agent in state.agents:
                goal = goals.get(agent.agent_id, agent.navigation_goal_position)
                before_distance = shortest_path_distance(
                    agent.position, goal, environment.config.map_layout_id
                )
                after_distance = shortest_path_distance(
                    targets[agent.agent_id], goal, environment.config.map_layout_id
                )
                distances[agent.agent_id] = {
                    "before": int(before_distance),
                    "after": int(after_distance),
                }
                productive_charging = bool(
                    actions[agent.agent_id] == "WAIT"
                    and agent.position == environment.layout.charger_position
                    and agent.battery < 100.0
                )
                progress_count += int(
                    after_distance < before_distance or productive_charging
                )
                commitment_regressions += int(
                    goal != agent.position and after_distance > before_distance
                )
                loaded_regressions += int(
                    agent.carrying_task_id is not None
                    and after_distance > before_distance
                )
                loaded_progress += int(
                    agent.carrying_task_id is not None
                    and after_distance < before_distance
                )
                charging_regressions += int(
                    environment._requires_charge(state, agent)
                    and after_distance > before_distance
                )
                charging_progress += int(
                    environment._requires_charge(state, agent)
                    and after_distance < before_distance
                )
                low_energy_priority_regressions += int(
                    lowest_energy_charger is not None
                    and agent.agent_id == lowest_energy_charger.agent_id
                    and after_distance > before_distance
                )
                low_energy_priority_progress += int(
                    lowest_energy_charger is not None
                    and agent.agent_id == lowest_energy_charger.agent_id
                    and after_distance < before_distance
                )
                charger_access_blocking += int(
                    lowest_energy_charger is not None
                    and agent.agent_id != lowest_energy_charger.agent_id
                    and agent.position == environment.layout.charger_position
                    and actions[agent.agent_id] == "WAIT"
                    and lowest_energy_charger.position
                    != environment.layout.charger_position
                )
                route_clearance_failures += int(
                    necessary_teammate_route_clearance(
                        environment,
                        state,
                        agent,
                    )
                    and actions[agent.agent_id] == "WAIT"
                )
                total_distance += int(after_distance)
                waits += int(actions[agent.agent_id] == "WAIT" and not productive_charging)
                reversals += int(
                    {
                        "UP": "DOWN",
                        "DOWN": "UP",
                        "LEFT": "RIGHT",
                        "RIGHT": "LEFT",
                    }.get(agent.last_executed_action)
                    == actions[agent.agent_id]
                )
                short_cycles += int(
                    actions[agent.agent_id] in MOVE_DELTAS
                    and targets[agent.agent_id]
                    in tuple(agent.recent_positions[:-1])[-5:]
                    and after_distance >= before_distance
                )
                policy_changes += int(
                    str(policy_actions.get(agent.agent_id, "WAIT"))
                    != actions[agent.agent_id]
                )
            # Reject a superficially attractive move that creates the exact
            # occupied-route stand-off the next frame.  This detects only a
            # mutual unique-next-cell dependency, so ordinary same-direction
            # pipeline movement remains valid.
            left_agent = state.by_id(left_id)
            right_agent = state.by_id(right_id)
            left_future_progress = _progress_positions_from(
                environment, targets[left_id], goals[left_id]
            )
            right_future_progress = _progress_positions_from(
                environment, targets[right_id], goals[right_id]
            )
            future_head_on_bottlenecks = int(
                left_future_progress == (targets[right_id],)
                and right_future_progress == (targets[left_id],)
            )
            for arriving_id, other_id in (
                (left_id, right_id),
                (right_id, left_id),
            ):
                arriving = state.by_id(arriving_id)
                if not _will_need_charge_after_delivery(
                    environment,
                    state,
                    arriving,
                    targets[arriving_id],
                    actions[arriving_id],
                ):
                    continue
                task = state.task_by_id(arriving.carrying_task_id)
                route_to_delivery = _canonical_progress_route(
                    environment,
                    targets[arriving_id],
                    task.delivery_position,
                    horizon=4,
                )
                if task.delivery_position not in route_to_delivery and (
                    targets[arriving_id] != task.delivery_position
                ):
                    continue
                charger_route = _canonical_progress_route(
                    environment,
                    task.delivery_position,
                    environment.layout.charger_position,
                    horizon=6,
                )
                anticipated_charger_route_blocking += int(
                    targets[other_id] in charger_route
                )
            selected_by_optimizer = all(
                actions[agent_id] == str(optimizer_actions.get(agent_id, "WAIT"))
                for agent_id in environment.agent_ids
            )
            # This is the public lexicographic contract.  The stable teacher
            # supplies frozen task matching and is only the final semantic
            # tie-break; it cannot override a safer or more productive pair.
            score = [
                commitment_regressions,
                loaded_regressions,
                -loaded_progress,
                low_energy_priority_regressions,
                charger_access_blocking,
                route_clearance_failures,
                future_head_on_bottlenecks,
                anticipated_charger_route_blocking,
                -low_energy_priority_progress,
                charging_regressions,
                -charging_progress,
                -progress_count,
                total_distance,
                waits,
                reversals,
                short_cycles,
                policy_changes,
                0 if selected_by_optimizer else 1,
                left_index,
                right_index,
            ]
            safe.append(
                {
                    "actions": dict(actions),
                    "targets": {
                        agent_id: list(position)
                        for agent_id, position in targets.items()
                    },
                    "distances": distances,
                    "score": score,
                    "score_breakdown": {
                        "committed_goal_regressions": commitment_regressions,
                        "loaded_delivery_regressions": loaded_regressions,
                        "loaded_delivery_progress": loaded_progress,
                        "low_energy_priority_regressions": low_energy_priority_regressions,
                        "charger_access_blocking": charger_access_blocking,
                        "route_clearance_failures": route_clearance_failures,
                        "future_head_on_bottlenecks": future_head_on_bottlenecks,
                        "anticipated_charger_route_blocking": (
                            anticipated_charger_route_blocking
                        ),
                        "low_energy_priority_progress": low_energy_priority_progress,
                        "charging_route_regressions": charging_regressions,
                        "charging_route_progress": charging_progress,
                        "progressing_agents": progress_count,
                        "total_goal_distance": total_distance,
                        "noncharging_waits": waits,
                        "immediate_reversals": reversals,
                        "short_cycles": short_cycles,
                        "matches_joint_optimizer": selected_by_optimizer,
                        "policy_action_changes": policy_changes,
                    },
                }
            )
    return safe, rejected


def select_ai_ai_joint_actions(
    environment: Any,
    policy_actions: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Select one safe, productive joint action from the frozen AI-AI state."""

    proposed = {
        agent_id: str(policy_actions.get(agent_id, "WAIT"))
        for agent_id in environment.agent_ids
    }
    optimized = stable_coordination_actions(environment)
    safe, rejected = _joint_candidate_evidence(
        environment,
        policy_actions=proposed,
        optimizer_actions=optimized,
    )
    optimized_safe = next(
        (
            item
            for item in safe
            if all(
                item["actions"][agent_id] == optimized[agent_id]
                for agent_id in environment.agent_ids
            )
        ),
        None,
    )
    def dominates_teacher(candidate: Mapping[str, Any]) -> bool:
        if optimized_safe is None or candidate is optimized_safe:
            return False
        candidate_breakdown = dict(candidate.get("score_breakdown", {}))
        teacher_breakdown = dict(optimized_safe.get("score_breakdown", {}))
        protected = (
            "loaded_delivery_regressions",
            "charging_route_regressions",
            "low_energy_priority_regressions",
            "charger_access_blocking",
            "route_clearance_failures",
            "future_head_on_bottlenecks",
            "anticipated_charger_route_blocking",
            "immediate_reversals",
            "short_cycles",
        )
        if any(
            int(candidate_breakdown.get(key, 0))
            > int(teacher_breakdown.get(key, 0))
            for key in protected
        ):
            return False
        candidate_distances = dict(candidate.get("distances", {}))
        teacher_distances = dict(optimized_safe.get("distances", {}))
        if any(
            int(candidate_distances[agent_id]["after"])
            > int(teacher_distances[agent_id]["after"])
            for agent_id in environment.agent_ids
        ):
            return False
        strict_progress = any(
            int(candidate_distances[agent_id]["after"])
            < int(teacher_distances[agent_id]["after"])
            for agent_id in environment.agent_ids
        )
        strict_efficiency = bool(
            int(candidate_breakdown.get("noncharging_waits", 0))
            < int(teacher_breakdown.get("noncharging_waits", 0))
            or int(candidate_breakdown.get("immediate_reversals", 0))
            < int(teacher_breakdown.get("immediate_reversals", 0))
            or int(candidate_breakdown.get("short_cycles", 0))
            < int(teacher_breakdown.get("short_cycles", 0))
            or any(
                int(candidate_breakdown.get(key, 0))
                < int(teacher_breakdown.get(key, 0))
                for key in protected
            )
        )
        adds_wait = int(candidate_breakdown.get("noncharging_waits", 0)) > int(
            teacher_breakdown.get("noncharging_waits", 0)
        )
        removes_regression_or_cycle = bool(
            int(candidate_breakdown.get("committed_goal_regressions", 0))
            < int(teacher_breakdown.get("committed_goal_regressions", 0))
            or int(candidate_breakdown.get("immediate_reversals", 0))
            < int(teacher_breakdown.get("immediate_reversals", 0))
            or int(candidate_breakdown.get("short_cycles", 0))
            < int(teacher_breakdown.get("short_cycles", 0))
        )
        if adds_wait and not removes_regression_or_cycle:
            return False
        return strict_progress or strict_efficiency

    dominating = [item for item in safe if dominates_teacher(item)]
    selected_record = (
        min(dominating, key=lambda item: item["score"])
        if dominating
        else optimized_safe
        if optimized_safe is not None
        else min(safe, key=lambda item: item["score"])
    )
    selected = dict(selected_record["actions"])
    return selected, {
        "mode": "ai_ai_joint_optimizer",
        "policy_actions": proposed,
        "selected_actions": dict(selected),
        "safe_joint_actions": safe,
        "rejected_joint_actions": rejected,
        "selected_joint_action": selected_record,
        "teacher_actions": dict(optimized),
        "teacher_actions_were_safe": optimized_safe is not None,
        "teacher_dominated": bool(dominating),
        "dominating_joint_actions": dominating,
        "selection_changed_policy": selected != proposed,
        "same_frozen_state": True,
    }


def select_human_ai_action(
    environment: Any,
    preferred_action: str,
) -> tuple[str, dict[str, Any]]:
    """Choose Robot 2 before observing the participant's current command."""

    state = environment.get_state()
    ai = state.by_id("robot_2")
    participant_actions = causal_participant_actions(environment)
    masks = environment.action_masks()
    goal = _committed_goal(environment, state, ai)
    public_coordination_action = stable_coordination_actions(environment)["robot_2"]
    participant = state.by_id(environment.config.human_agent_id)
    active_plan = state.active_coordination_plan or {}
    charger_handoff_action = charger_handoff_clearance_action(
        environment,
        state,
        ai,
        participant,
    )
    ai_is_planned_waiter = bool(
        str(active_plan.get("waiting_agent_id", "")) == ai.agent_id
        and str(active_plan.get("moving_agent_id", "")) != ai.agent_id
        and public_coordination_action == "WAIT"
        and str(active_plan.get("phase", ""))
        in {"CLEAR_CELL", "PASS_THROUGH", "SINGLE_STEP", "JOINT_STEP"}
    )
    ai_is_planned_clearer = bool(
        str(active_plan.get("phase", "")) == "CLEAR_CELL"
        and str(active_plan.get("moving_agent_id", "")) == ai.agent_id
        and str(active_plan.get("yielding_agent_id", "")) == ai.agent_id
        and public_coordination_action in MOVE_DELTAS
    )
    allowed_clearing_actions = {
        str(action)
        for action in active_plan.get(
            "allowed_clearing_actions",
            (public_coordination_action,),
        )
    }
    allowed_clearing_targets = {
        tuple(target)
        for target in active_plan.get(
            "allowed_clearing_targets",
            (active_plan.get("moving_target", ()),),
        )
        if isinstance(target, (list, tuple)) and len(target) == 2
    }
    physical_clearance_required = bool(
        ai_is_planned_clearer
        or (
            ai.carrying_task_id is None
            and (
                charger_handoff_action is not None
                or (
                    not (
                        ai.position == environment.layout.charger_position
                        and environment._requires_charge(state, ai)
                    )
                    and (
                        necessary_teammate_route_clearance(
                            environment, state, ai
                        )
                        or necessary_participant_standoff_clearance(
                            environment, state, ai
                        )
                    )
                )
            )
        )
    )
    recent_goal_event = bool(
        int(ai.goal_since) >= int(state.frame) - 1
        and str(ai.goal_switch_reason)
        in {
            "pickup_completed",
            "delivery_completed",
            "task_completed_or_unavailable",
            "task_claimed_by_teammate",
            "charge_release_threshold_met",
            "energy_route_infeasible",
            "energy_safe_task_committed",
            "joint_coordination_plan_started",
            "joint_coordination_plan_completed",
        }
    )
    last_action_was_participant_clearance = any(
        str(event.get("event", "")) == "participant_standoff_clearance"
        and str(event.get("agent_id", "")) == ai.agent_id
        for event in state.last_coordination_events
        if isinstance(event, Mapping)
    )
    candidates: list[dict[str, Any]] = []
    for action_index, (action, allowed) in enumerate(
        zip(ACTIONS, masks["robot_2"])
    ):
        if allowed <= 0.5:
            continue
        conflicts: list[dict[str, str]] = []
        for participant_action in participant_actions:
            _, _, invalid, collision, collision_kind, _ = environment._resolve_motion(
                state,
                {"robot_1": participant_action, "robot_2": action},
            )
            if invalid or collision:
                conflicts.append(
                    {
                        "participant_action": participant_action,
                        "kind": str(collision_kind or "invalid_move"),
                    }
                )
        target = _target(ai, action)
        satisfies_planned_clearance = bool(
            ai_is_planned_clearer
            and action in allowed_clearing_actions
            and target in allowed_clearing_targets
        )
        before_distance = shortest_path_distance(
            ai.position, goal, environment.config.map_layout_id
        )
        after_distance = shortest_path_distance(
            target, goal, environment.config.map_layout_id
        )
        remaining = ai.battery - (
            environment.config.move_battery_cost if action in MOVE_DELTAS else 0.0
        )
        energy_violation = int(
            action in MOVE_DELTAS
            and (
                remaining < 0.0
                or (
                    remaining <= 0.0
                    and target != environment.layout.charger_position
                )
                or (
                    ai.navigation_goal_kind == "charge"
                    and remaining + 1e-8
                    < after_distance * environment.config.move_battery_cost
                )
            )
        )
        recent_unproductive_charger_reentry = int(
            target == environment.layout.charger_position
            and ai.position != environment.layout.charger_position
            and ai.last_charger_departure_frame is not None
            and 0 <= state.frame - ai.last_charger_departure_frame <= 6
            and not any(charger_departure_progress(state, ai))
            # Re-entry hysteresis prevents optional charger oscillation; it
            # must never outrank a newly proven energy requirement.  A noisy
            # participant can delay or displace the AI after departure, so a
            # genuinely critical return is new causal evidence rather than
            # an unproductive cycle.
            and not environment._requires_charge(state, ai)
        )
        reversal = int(
            not recent_goal_event
            and not last_action_was_participant_clearance
            and
            {
                "UP": "DOWN",
                "DOWN": "UP",
                "LEFT": "RIGHT",
                "RIGHT": "LEFT",
            }.get(ai.last_executed_action)
            == action
        )
        nonprogress_move = int(
            action in MOVE_DELTAS and after_distance >= before_distance
        )
        score = [
            int(bool(conflicts)),
            len(conflicts),
            energy_violation,
            recent_unproductive_charger_reentry,
            int(ai_is_planned_waiter and action != "WAIT"),
            int(
                ai_is_planned_clearer
                and not satisfies_planned_clearance
            ),
            int(physical_clearance_required and action == "WAIT"),
            int(after_distance > before_distance),
            nonprogress_move,
            after_distance,
            int(action == "WAIT"),
            # Once an action is proven safe and makes strict mission
            # progress, that progress outranks the cosmetic cost of reversing
            # a prior clearance/lifecycle step.  Putting reversal before goal
            # distance caused the AI to WAIT despite a safe shorter route.
            reversal,
            int(action != public_coordination_action),
            int(action != str(preferred_action)),
            action_index,
        ]
        candidates.append(
            {
                "action": action,
                "target": list(target),
                "safe_for_all_participant_actions": not conflicts,
                "collision_counterfactuals": conflicts,
                "distance_before": int(before_distance),
                "distance_after": int(after_distance),
                "energy_violation": bool(energy_violation),
                "recent_unproductive_charger_reentry": bool(
                    recent_unproductive_charger_reentry
                ),
                "satisfies_planned_clearance": satisfies_planned_clearance,
                "nonprogress_move": bool(nonprogress_move),
                "score": score,
            }
        )
    selected_record = min(candidates, key=lambda item: item["score"])
    selected = str(selected_record["action"])
    return selected, {
        "mode": "human_ai_robust_selection",
        "policy_actions": {"robot_2": str(preferred_action)},
        "selected_actions": {"robot_2": selected},
        "participant_action_known_at_decision_time": False,
        "participant_legal_actions": list(participant_actions),
        "public_coordination_action": public_coordination_action,
        "physical_clearance_required": physical_clearance_required,
        "ai_is_planned_waiter": ai_is_planned_waiter,
        "ai_is_planned_clearer": ai_is_planned_clearer,
        "charger_handoff_action": charger_handoff_action,
        "recent_goal_event": recent_goal_event,
        "last_action_was_participant_clearance": (
            last_action_was_participant_clearance
        ),
        "ai_action_candidates": candidates,
        "selected_ai_action": selected_record,
        "selection_changed_policy": selected != str(preferred_action),
        "same_frozen_state": True,
    }
