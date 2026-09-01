"""Cross-frame decision diagnostics without policy or environment imports."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .navigation import MOVE_DELTAS, shortest_path_distance


OPPOSITE_ACTION = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}


def transition_temporal_violations(
    environment: Any,
    previous: Any,
    next_state: Any,
    *,
    requested_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
    pickup_agents: Iterable[str] = (),
    delivery_agents: Iterable[str] = (),
    coordination_events: Iterable[Mapping[str, Any]] = (),
    energy_events: Iterable[Mapping[str, Any]] = (),
) -> dict[str, tuple[str, ...]]:
    """Audit unexplained reversals, short cycles, and late goal switches."""

    pickup = set(pickup_agents)
    delivery = set(delivery_agents)
    # Coordination records are hypotheses, not exemptions.  The physical
    # checks below must independently prove that a detour or reversal cleared
    # a route from S_t.
    prior_participant_clearance_agents = {
        str(event.get("agent_id", ""))
        for event in previous.last_coordination_events
        if str(event.get("event", "")) == "participant_standoff_clearance"
    }
    energy_progress = {
        str(event.get("agent_id", ""))
        for event in energy_events
        if str(event.get("event", ""))
        in {"charger_departure", "charger_productive_return"}
    }
    unexplained_reversals: list[str] = []
    short_cycles: list[str] = []
    invalid_goal_switches: list[str] = []
    for before in previous.agents:
        agent_id = before.agent_id
        after = next_state.by_id(agent_id)
        action = str(executed_actions.get(agent_id, "WAIT"))
        requested = str(requested_actions.get(agent_id, action))
        participant_standoff_clearance = False
        teammate_route_clearance = False
        if action in MOVE_DELTAS:
            # This import stays local because transition_audit also owns the
            # frozen-state robust-action helper.  A retreat after an observed
            # participant WAIT is an explained response to public S_t, not a
            # policy cycle caused by seeing the participant's new command.
            from .transition_audit import (
                necessary_participant_standoff_clearance,
            )

            participant_standoff_clearance = bool(
                necessary_participant_standoff_clearance(
                    environment,
                    previous,
                    before,
                    candidate_action=action,
                )
            )
            from .transition_audit import necessary_teammate_route_clearance

            teammate_route_clearance = bool(
                necessary_teammate_route_clearance(
                    environment,
                    previous,
                    before,
                )
            )
        recent_lifecycle_switch = bool(
            before.goal_since >= previous.frame - 1
            and before.goal_switch_reason
            in {
                "pickup_completed",
                "delivery_completed",
                "charge_release_threshold_met",
                "joint_coordination_plan_completed",
                "task_claimed_by_teammate",
                "energy_route_infeasible",
                "energy_safe_task_committed",
            }
        )
        allowed_event = bool(
            agent_id in pickup
            or agent_id in delivery
            or agent_id in energy_progress
            or requested != action
            or recent_lifecycle_switch
            or participant_standoff_clearance
            or teammate_route_clearance
            or (
                action == OPPOSITE_ACTION.get(before.last_executed_action)
                and agent_id in prior_participant_clearance_agents
            )
        )
        immediate_reverse = bool(
            action == OPPOSITE_ACTION.get(before.last_executed_action)
            and len(before.recent_positions) >= 2
            and after.position == before.recent_positions[-2]
        )
        if immediate_reverse and not allowed_event:
            unexplained_reversals.append(agent_id)
        goal_position = None
        if before.navigation_goal_kind == "charge":
            goal_position = before.navigation_goal_position
        elif before.carrying_task_id is not None:
            goal_position = previous.task_by_id(
                before.carrying_task_id
            ).delivery_position
        elif before.route_commitment_task_id is not None:
            committed = next(
                (
                    task
                    for task in previous.tasks
                    if task.task_id == before.route_commitment_task_id
                ),
                None,
            )
            if committed is not None:
                goal_position = committed.pickup_position
        elif before.goal_type == "GO_TO_PICKUP" and before.goal_id is not None:
            selected = next(
                (
                    task
                    for task in previous.tasks
                    if task.task_id == before.goal_id
                    and task.status == "available"
                ),
                None,
            )
            if selected is not None:
                goal_position = selected.pickup_position
        route_progress = bool(
            goal_position is not None
            and shortest_path_distance(
                after.position,
                goal_position,
                environment.config.map_layout_id,
            )
            < shortest_path_distance(
                before.position,
                goal_position,
                environment.config.map_layout_id,
            )
        )
        prior_positions = tuple(before.recent_positions[:-1])[-5:]
        if (
            action in MOVE_DELTAS
            and after.position in prior_positions
            and not allowed_event
            and goal_position is not None
            and not route_progress
            and before.goal_type == after.goal_type
            and before.goal_id == after.goal_id
        ):
            short_cycles.append(agent_id)

        goal_changed = bool(
            before.goal_type != after.goal_type
            or before.goal_id != after.goal_id
        )
        if not goal_changed:
            continue
        reason = str(after.goal_switch_reason)
        valid_reasons = {
            "pickup_completed",
            "delivery_completed",
            "task_completed_or_unavailable",
            "task_claimed_by_teammate",
            "charge_release_threshold_met",
            "joint_coordination_plan_started",
            "joint_coordination_plan_completed",
            "energy_safe_task_committed",
            "energy_route_infeasible",
        }
        invalid = reason not in valid_reasons
        if (
            reason == "energy_route_infeasible"
            and before.navigation_goal_kind != "charge"
            and not environment._requires_charge(previous, before)
            and action in MOVE_DELTAS
            and not allowed_event
        ):
            invalid = True
        if invalid:
            invalid_goal_switches.append(agent_id)
    return {
        "unexplained_reversal_agents": tuple(sorted(unexplained_reversals)),
        "short_cycle_agents": tuple(sorted(set(short_cycles))),
        "invalid_goal_switch_agents": tuple(
            sorted(set(invalid_goal_switches))
        ),
    }
