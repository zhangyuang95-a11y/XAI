"""Persistent mission goals and frozen joint-plan lifecycle helpers.

The environment owns transition order; this module owns the cross-frame intent
that must survive those transitions.  Keeping the two concerns separate makes
it explicit that goals are derived from the frozen state and are not inferred
again after the first robot moves.
"""

from __future__ import annotations

from typing import Any, Mapping

from .coordination_plan import frozen_joint_coordination_plan
from .domain import WarehouseState
from .energy_management import (
    charger_handoff_clearance_action,
    should_continue_charge_mode,
)
from .navigation import MOVE_DELTAS, shortest_path_distance


def refresh_navigation_goals(environment: Any, state: WarehouseState) -> None:
    """Refresh delivery/charger modes while preserving pickup commitments."""

    for agent in state.agents:
        if not agent.active:
            agent.navigation_goal_kind = "wait"
            agent.navigation_goal_position = agent.position
            continue
        if agent.charge_mode_active:
            if should_continue_charge_mode(environment, state, agent):
                agent.navigation_goal_kind = "charge"
                agent.navigation_goal_position = environment.layout.charger_position
                continue
            agent.charge_mode_active = False
        if environment._requires_charge(state, agent):
            agent.charge_mode_active = True
            agent.navigation_goal_kind = "charge"
            agent.navigation_goal_position = environment.layout.charger_position
            continue
        if agent.carrying_task_id:
            task = state.task_by_id(agent.carrying_task_id)
            agent.navigation_goal_kind = "delivery"
            agent.navigation_goal_position = task.delivery_position
            continue
        agent.navigation_goal_kind = "wait"
        agent.navigation_goal_position = agent.position


def assign_persistent_pickup_goals(
    environment: Any,
    state: WarehouseState,
) -> None:
    """Lock one distinct, energy-safe shared pickup goal per empty robot."""

    available = {
        task.task_id: task for task in state.tasks if task.status == "available"
    }
    strong_owner_by_task: dict[str, str] = {}
    for agent in sorted(state.agents, key=lambda item: item.agent_id):
        committed = agent.route_commitment_task_id
        if committed in available and committed not in strong_owner_by_task:
            strong_owner_by_task[str(committed)] = agent.agent_id

    reserved: set[str] = set(strong_owner_by_task)
    goal_owner_by_task: dict[str, str] = {}
    for agent in sorted(state.agents, key=lambda item: item.agent_id):
        existing = agent.route_commitment_task_id
        if existing in available:
            if strong_owner_by_task.get(str(existing)) != agent.agent_id:
                agent.route_commitment_task_id = None
            else:
                continue
        if (
            agent.goal_type == "GO_TO_PICKUP"
            and agent.goal_id in available
            and environment._task_is_directly_energy_safe(
                state,
                agent,
                available[str(agent.goal_id)],
            )
        ):
            goal_id = str(agent.goal_id)
            if goal_id in reserved or goal_id in goal_owner_by_task:
                agent.goal_type = "SELECT_TASK"
                agent.goal_id = None
            else:
                goal_owner_by_task[goal_id] = agent.agent_id
                reserved.add(goal_id)
        elif agent.carrying_task_id is None:
            agent.goal_id = None
            if agent.goal_type == "GO_TO_PICKUP":
                agent.goal_type = "SELECT_TASK"

    for agent in sorted(state.agents, key=lambda item: item.agent_id):
        if (
            not agent.active
            or agent.carrying_task_id is not None
            or agent.navigation_goal_kind == "charge"
            or agent.route_commitment_task_id in available
            or (
                agent.goal_type == "GO_TO_PICKUP"
                and agent.goal_id in available
            )
        ):
            continue
        candidates = [
            task
            for task in available.values()
            if task.task_id not in reserved
            and environment._task_is_directly_energy_safe(state, agent, task)
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda task: (
                environment._safe_task_cost(state, agent, task),
                -max(0, state.frame - task.created_frame),
                task.task_id,
            ),
        )
        agent.goal_type = "GO_TO_PICKUP"
        agent.goal_id = selected.task_id
        reserved.add(selected.task_id)


def synchronize_persistent_goals(
    environment: Any,
    previous: WarehouseState | None,
    state: WarehouseState,
    *,
    reset_reason: str | None = None,
    coordination_plan: Mapping[str, Any] | None = None,
) -> None:
    """Persist explicit goal phase and record a causal switch reason."""

    for agent in state.agents:
        before = previous.by_id(agent.agent_id) if previous is not None else None
        plan_role = None
        participates_in_plan = False
        if coordination_plan is not None:
            plan_ids = {
                str(coordination_plan.get("priority_agent_id", "")),
                str(coordination_plan.get("clearing_agent_id", "")),
                str(coordination_plan.get("yielding_agent_id", "")),
            }
            participates_in_plan = agent.agent_id in plan_ids
            is_yielding = (
                str(coordination_plan.get("yielding_agent_id", ""))
                == agent.agent_id
            )
            is_blocked_charger_priority = bool(
                str(coordination_plan.get("phase")) == "CLEAR_CELL"
                and str(coordination_plan.get("priority_agent_id"))
                == agent.agent_id
                and agent.navigation_goal_kind == "charge"
            )
            if is_blocked_charger_priority:
                plan_role = "QUEUED_FOR_CHARGER"
            elif is_yielding:
                plan_role = "YIELDING"
        if not agent.active:
            goal_type, goal_id = "IDLE", None
        elif plan_role is not None:
            goal_type = plan_role
            goal_id = (
                agent.route_commitment_task_id
                or agent.carrying_task_id
                or agent.goal_id
            )
        elif agent.navigation_goal_kind == "charge":
            goal_type = (
                "CHARGING"
                if agent.position == environment.layout.charger_position
                else "GO_TO_CHARGER"
            )
            goal_id = None
        elif agent.carrying_task_id is not None:
            goal_type, goal_id = "GO_TO_DROPOFF", agent.carrying_task_id
        elif agent.route_commitment_task_id is not None:
            goal_type, goal_id = "GO_TO_PICKUP", agent.route_commitment_task_id
        elif agent.goal_type == "GO_TO_PICKUP" and agent.goal_id is not None:
            goal_type, goal_id = "GO_TO_PICKUP", agent.goal_id
        else:
            goal_type, goal_id = "SELECT_TASK", None

        same_goal = bool(
            before is not None
            and before.goal_type == goal_type
            and before.goal_id == goal_id
        )
        if same_goal:
            agent.goal_since = before.goal_since
            agent.goal_switch_reason = before.goal_switch_reason
        else:
            reason = reset_reason
            if reason is None and before is not None:
                available_ids = {
                    task.task_id
                    for task in state.tasks
                    if task.status == "available"
                }
                prior_pickup_id = before.route_commitment_task_id or before.goal_id
                concurrent_claim_loss = any(
                    task.status == "carried"
                    and task.carrier_agent_id != agent.agent_id
                    and task.claimed_frame == state.frame
                    and shortest_path_distance(
                        agent.position,
                        task.pickup_position,
                        environment.config.map_layout_id,
                    )
                    < shortest_path_distance(
                        before.position,
                        task.pickup_position,
                        environment.config.map_layout_id,
                    )
                    for task in state.tasks
                )
                if (
                    agent.carrying_task_id is not None
                    and before.carrying_task_id is None
                ):
                    reason = "pickup_completed"
                elif (
                    before.carrying_task_id is not None
                    and agent.carrying_task_id is None
                ):
                    reason = "delivery_completed"
                elif before.carrying_task_id is None and (
                    concurrent_claim_loss
                    or (
                        prior_pickup_id is not None
                        and prior_pickup_id not in available_ids
                    )
                ):
                    reason = "task_claimed_by_teammate"
                elif (
                    before.goal_type in {"YIELDING", "QUEUED_FOR_CHARGER"}
                    and plan_role is None
                ):
                    reason = "joint_coordination_plan_completed"
                elif goal_type in {
                    "GO_TO_CHARGER",
                    "CHARGING",
                    "QUEUED_FOR_CHARGER",
                }:
                    reason = "energy_route_infeasible"
                elif before.goal_type in {
                    "GO_TO_CHARGER",
                    "CHARGING",
                    "QUEUED_FOR_CHARGER",
                }:
                    reason = "charge_release_threshold_met"
                elif goal_type == "YIELDING":
                    reason = "joint_coordination_plan_started"
                elif goal_type == "GO_TO_PICKUP":
                    reason = "energy_safe_task_committed"
                elif goal_type == "SELECT_TASK":
                    reason = "task_completed_or_unavailable"
            agent.goal_since = state.frame
            agent.goal_switch_reason = reason or "state_restore"
        agent.goal_type = goal_type
        agent.goal_id = goal_id
        agent.charging_reason = (
            "insufficient_energy_for_safe_mission"
            if goal_type in {
                "GO_TO_CHARGER",
                "QUEUED_FOR_CHARGER",
                "CHARGING",
            }
            else None
        )
        agent.yielding_plan_id = (
            str(coordination_plan.get("plan_id"))
            if coordination_plan is not None and participates_in_plan
            else None
        )
        history = tuple(before.recent_positions if before is not None else ())
        if not history or history[-1] != agent.position:
            history += (agent.position,)
        agent.recent_positions = history[-6:]
        goal_history = tuple(
            before.recent_goal_types if before is not None else ()
        )
        if not goal_history or goal_history[-1] != goal_type:
            goal_history += (goal_type,)
        agent.recent_goal_types = goal_history[-6:]


def frozen_coordination_plan(
    environment: Any,
    state: WarehouseState,
) -> dict[str, Any] | None:
    """Compute the one coordination plan visible in frozen ``S_t``."""

    return frozen_joint_coordination_plan(
        state,
        environment.config,
        requires_charge={
            agent.agent_id: bool(
                agent.navigation_goal_kind == "charge"
                and agent.navigation_goal_position
                == environment.layout.charger_position
            )
            for agent in state.agents
        },
    )


def _human_ai_charger_handoff_plan(
    environment: Any,
    state: WarehouseState,
) -> dict[str, Any] | None:
    """Publish a two-phase charger handoff before either side acts.

    Human-AI motion cannot use an atomic occupied-cell handoff because the AI
    must be selected without observing the participant's current command.  A
    sufficiently charged station occupant therefore clears in one public
    frame, and the more depleted waiter enters only from the following
    frozen state.  The shared energy helper is deliberately authoritative:
    ordinary route-clearance heuristics must not evict a robot that still
    needs the charger itself.
    """

    participant_id = state.participant_controlled_agent_id
    if participant_id is None:
        return None
    occupant = next(
        (
            agent
            for agent in state.agents
            if agent.active
            and agent.position == environment.layout.charger_position
        ),
        None,
    )
    if occupant is None:
        return None
    waiter = next(
        (
            agent
            for agent in state.agents
            if agent.active and agent.agent_id != occupant.agent_id
        ),
        None,
    )
    if waiter is None:
        return None
    clearing_action = charger_handoff_clearance_action(
        environment,
        state,
        occupant,
        waiter,
    )
    delta = MOVE_DELTAS.get(str(clearing_action))
    if delta is None:
        return None
    clearing_target = (
        occupant.position[0] + delta[0],
        occupant.position[1] + delta[1],
    )
    priority_goal_id = (
        waiter.carrying_task_id
        or waiter.route_commitment_task_id
        or waiter.goal_id
    )
    return {
        "plan_id": (
            f"coord:{state.episode_id}:{state.frame}:charger-handoff:"
            f"{waiter.agent_id}:{occupant.agent_id}"
        ),
        "plan_kind": "charger_handoff_clearance",
        "phase": "CLEAR_CELL",
        "priority_agent_id": waiter.agent_id,
        "waiting_agent_id": waiter.agent_id,
        "clearing_agent_id": occupant.agent_id,
        "yielding_agent_id": occupant.agent_id,
        "moving_agent_id": occupant.agent_id,
        "occupied_position": occupant.position,
        "clearing_action": str(clearing_action),
        "clearing_target": clearing_target,
        "allowed_clearing_actions": (str(clearing_action),),
        "allowed_clearing_targets": (clearing_target,),
        "moving_action": str(clearing_action),
        "moving_target": clearing_target,
        "joint_actions": {
            agent.agent_id: (
                str(clearing_action)
                if agent.agent_id == occupant.agent_id
                else "WAIT"
            )
            for agent in state.agents
        },
        "priority_basis": "lower_energy_charger_waiter",
        "priority_goal_id": priority_goal_id,
        "reason_code": "charger_handoff_clearance",
        "expected_duration_frames": 2,
        "completion_condition": "charger_cell_cleared",
        "resume_condition": "priority_robot_enters_charger",
        "derived_from_frame": state.frame,
    }


def prepare_coordination_plan(
    environment: Any,
    state: WarehouseState,
) -> dict[str, Any] | None:
    """Freeze a newly derived plan before either actor is evaluated.

    Human-AI rounds publish the plan in ``S_t``.  A public priority target may
    disable only the participant command that would enter that reserved cell;
    the AI still never observes the participant's private current-frame
    command.
    """
    # Preserve an already-published plan verbatim.  For a fresh Human-AI
    # state, however, the energy-authoritative charger handoff must outrank
    # generic occupied-route clearance.  The latter sees the same occupied
    # cell but does not know whether the occupant has enough energy to leave
    # and return, which previously evicted an equally depleted AI from the
    # station and created an UP->DOWN cycle.
    plan = (
        frozen_coordination_plan(environment, state)
        if state.active_coordination_plan is not None
        else _human_ai_charger_handoff_plan(environment, state)
    )
    if plan is None:
        plan = frozen_coordination_plan(environment, state)
    if plan is None:
        state.active_coordination_plan = None
        return None
    if state.active_coordination_plan is None:
        first_actions = dict(plan.get("joint_actions", {}))
        if not first_actions:
            first_actions = {
                agent.agent_id: (
                    str(plan.get("moving_action", "WAIT"))
                    if agent.agent_id == str(plan.get("moving_agent_id", ""))
                    else "WAIT"
                )
                for agent in state.agents
            }
        sequence = [
            {
                "step": 0,
                "phase": str(plan.get("phase", "CLEAR_CELL")),
                "joint_actions": first_actions,
                "completion_condition": str(
                    plan.get("completion_condition", "joint_step_completed")
                ),
            }
        ]
        if str(plan.get("phase", "")) == "CLEAR_CELL":
            priority = state.by_id(str(plan.get("priority_agent_id")))
            occupied = tuple(plan.get("occupied_position", ()))
            delta = (
                occupied[0] - priority.position[0],
                occupied[1] - priority.position[1],
            )
            pass_action = next(
                (
                    action
                    for action, action_delta in MOVE_DELTAS.items()
                    if action_delta == delta
                ),
                "WAIT",
            )
            sequence.append(
                {
                    "step": 1,
                    "phase": "PASS_THROUGH",
                    "joint_actions": {
                        agent.agent_id: (
                            pass_action
                            if agent.agent_id == priority.agent_id
                            else "WAIT"
                        )
                        for agent in state.agents
                    },
                    "completion_condition": "priority_robot_enters_cleared_route",
                }
            )
        plan = {
            **plan,
            "phase": str(plan.get("phase", "CLEAR_CELL")),
            "started_frame": state.frame,
            "current_plan_step": 0,
            "planned_action_sequence": sequence,
            "release_condition": str(
                plan.get("resume_condition", "required_route_cell_released")
            ),
            "invalidation_conditions": (
                "priority_goal_changed",
                "planned_action_became_unsafe",
                "required_cell_already_released",
                "participant_deviated_from_public_plan",
            ),
        }
    state.active_coordination_plan = dict(plan)
    return dict(plan)


def advance_coordination_plan(
    previous: WarehouseState,
    next_state: WarehouseState,
    execution: Mapping[str, Any] | None,
) -> None:
    """Advance a frozen two-phase plan without recomputing its roles."""

    active = previous.active_coordination_plan
    next_state.active_coordination_plan = None
    if active is None or execution is None:
        return
    if not bool(execution.get("execution_aligned", False)):
        return
    phase = str(active.get("phase", "CLEAR_CELL"))
    if phase == "CLEAR_CELL":
        next_state.active_coordination_plan = {
            **dict(active),
            "phase": "PASS_THROUGH",
            "phase_started_frame": next_state.frame,
            "current_plan_step": int(active.get("current_plan_step", 0)) + 1,
        }
        return
    next_state.coordination_plan_cooldown_until = next_state.frame + 2


def update_route_commitments(
    environment: Any,
    previous: WarehouseState,
    next_state: WarehouseState,
    executed: Mapping[str, str],
) -> None:
    """Persist task intent revealed by an actor's own successful movement."""

    previous_available = {
        task.task_id: task
        for task in previous.tasks
        if task.status == "available"
    }
    next_available = {
        task.task_id: task
        for task in next_state.tasks
        if task.status == "available"
    }
    for agent in next_state.agents:
        before_agent = previous.by_id(agent.agent_id)
        if agent.carrying_task_id is not None:
            agent.route_commitment_task_id = agent.carrying_task_id
            continue
        persistent_goal = (
            before_agent.goal_id
            if before_agent.goal_type in {"GO_TO_PICKUP", "YIELDING"}
            and before_agent.goal_id in next_available
            else None
        )
        existing = (
            before_agent.route_commitment_task_id
            if before_agent.route_commitment_task_id in next_available
            else persistent_goal
        )
        agent.route_commitment_task_id = None
        if (
            before_agent.carrying_task_id is not None
            or before_agent.navigation_goal_kind == "charge"
            or before_agent.charge_mode_active
            or executed.get(agent.agent_id) not in MOVE_DELTAS
        ):
            if existing in next_available:
                agent.route_commitment_task_id = existing
            continue
        progress_candidates: list[tuple[int, int, int, str]] = []
        for task_id, task in previous_available.items():
            if task_id not in next_available:
                continue
            before_distance = shortest_path_distance(
                before_agent.position,
                task.pickup_position,
                environment.config.map_layout_id,
            )
            after_distance = shortest_path_distance(
                agent.position,
                task.pickup_position,
                environment.config.map_layout_id,
            )
            progress = before_distance - after_distance
            if progress <= 0:
                continue
            age = max(0, previous.frame - task.created_frame)
            progress_candidates.append((-progress, after_distance, -age, task_id))
        if not progress_candidates:
            if existing in next_available:
                agent.route_commitment_task_id = existing
            continue
        progress_candidates.sort()
        best = progress_candidates[0]
        equally_informative = [
            item for item in progress_candidates if item[0] == best[0]
        ]
        if len(equally_informative) == 1:
            inferred = best[3]
            teammate = next(
                item
                for item in previous.agents
                if item.agent_id != before_agent.agent_id
            )
            teammate_committed_task_id = teammate.route_commitment_task_id or (
                teammate.goal_id
                if teammate.goal_type in {"GO_TO_PICKUP", "YIELDING"}
                else None
            )
            if existing in next_available:
                agent.route_commitment_task_id = existing
            elif inferred != teammate_committed_task_id:
                agent.route_commitment_task_id = inferred
        elif existing in next_available:
            agent.route_commitment_task_id = existing
