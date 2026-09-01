"""Stable public transition information and causal decision audit payloads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .decision_protocol import DECISION_AUDIT_SCHEMA, canonical_sha256
from .energy_management import charge_release_evidence
from .frozen_missions import frozen_training_missions
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance


def necessary_teammate_route_clearance(
    environment: Any,
    state: Any,
    clearing_agent: Any,
) -> bool:
    """Whether S_t requires this robot to vacate a peer's unique next cell.

    This is a frozen-state topology test, not a prediction of the peer's
    current-frame action.  A move away from the clearing robot's own mission
    is therefore legitimate coordination rather than an avoidable detour when
    its present cell is the teammate's only shortest-path progress cell.
    """

    # Use the same frozen mission consumed by the runtime joint selector,
    # reward audit, and DecisionTrace.  ``navigation_goal_position`` can be
    # temporarily replaced by the robot's current cell while a public
    # YIELDING plan is active; using that display goal here made a physically
    # necessary retreat look like an unexplained reversal/short cycle.
    missions = frozen_training_missions(environment, state)
    for teammate in state.agents:
        if teammate.agent_id == clearing_agent.agent_id or not teammate.active:
            continue
        mission = missions.get(teammate.agent_id)
        goal = (
            mission.goal_position
            if mission is not None
            else teammate.navigation_goal_position
        )
        current_distance = shortest_path_distance(
            teammate.position,
            goal,
            environment.config.map_layout_id,
        )
        if current_distance <= 0:
            continue
        progress_positions = tuple(
            candidate
            for delta in MOVE_DELTAS.values()
            if environment.layout.is_passable(
                candidate := (
                    teammate.position[0] + delta[0],
                    teammate.position[1] + delta[1],
                )
            )
            and shortest_path_distance(
                candidate,
                goal,
                environment.config.map_layout_id,
            )
            < current_distance
        )
        if progress_positions == (clearing_agent.position,):
            return True
    return False


def necessary_participant_standoff_clearance(
    environment: Any,
    state: Any,
    clearing_agent: Any,
    *,
    candidate_action: str | None = None,
) -> bool:
    """Whether one causal retreat is required after a participant stand-off.

    A cautious participant can repeatedly WAIT while another robot remains
    within two path steps even when that robot is not on the participant's
    unique shortest path.  After a stationary joint transition is visible in
    ``S_t``, one move that is safe against every legal participant action and
    increases separation is necessary yielding, not an avoidable detour.  No
    current-frame participant action is read here.
    """

    participant_id = getattr(state, "participant_controlled_agent_id", None)
    if (
        participant_id is None
        or clearing_agent.agent_id == participant_id
        or int(getattr(state, "ineffective_joint_wait_streak", 0)) < 1
        or not clearing_agent.active
        or clearing_agent.battery <= environment.config.move_battery_cost
    ):
        return False
    participant = state.by_id(participant_id)
    if not participant.active or participant.last_executed_action != "WAIT":
        return False
    current_separation = shortest_path_distance(
        participant.position,
        clearing_agent.position,
        environment.config.map_layout_id,
    )
    if current_separation > 2:
        return False

    actions = (
        (candidate_action,)
        if candidate_action is not None
        else tuple(MOVE_DELTAS)
    )
    from .observations import _actor_action_mask

    own_mask = _actor_action_mask(
        state,
        clearing_agent,
        environment.config,
    )
    for action in actions:
        if action not in MOVE_DELTAS:
            continue
        if own_mask[ACTIONS.index(action)] <= 0.5:
            continue
        delta = MOVE_DELTAS[action]
        target = (
            clearing_agent.position[0] + delta[0],
            clearing_agent.position[1] + delta[1],
        )
        if clearing_agent.navigation_goal_kind == "charge":
            remaining_battery = (
                clearing_agent.battery - environment.config.move_battery_cost
            )
            required_to_charger = (
                shortest_path_distance(
                    target,
                    environment.layout.charger_position,
                    environment.config.map_layout_id,
                )
                * environment.config.move_battery_cost
            )
            if remaining_battery + 1e-8 < required_to_charger:
                continue
        if shortest_path_distance(
            participant.position,
            target,
            environment.config.map_layout_id,
        ) <= current_separation:
            continue
        requested = {agent.agent_id: "WAIT" for agent in state.agents}
        if action_is_robustly_safe(
            environment,
            state,
            requested,
            clearing_agent.agent_id,
            action,
        ):
            return True
    return False


def wait_is_robustly_safe(
    environment: Any,
    state: Any,
    requested_actions: Mapping[str, str],
    agent_id: str,
) -> bool:
    """Whether WAIT avoids collision for every legal peer action from S_t."""

    return action_is_robustly_safe(
        environment,
        state,
        requested_actions,
        agent_id,
        "WAIT",
    )


def action_is_robustly_safe(
    environment: Any,
    state: Any,
    requested_actions: Mapping[str, str],
    agent_id: str,
    candidate_action: str,
) -> bool:
    """Check one action against every legal simultaneous peer action."""

    teammate_id = next(
        agent.agent_id for agent in state.agents if agent.agent_id != agent_id
    )
    for teammate_action in ACTIONS:
        trial = dict(requested_actions)
        trial[agent_id] = candidate_action
        trial[teammate_id] = teammate_action
        _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
        if teammate_id in invalid:
            continue
        if agent_id in invalid or collision:
            return False
    return True


def environment_info(
    environment: Any,
    *,
    reward_breakdown: Mapping[str, float] | None,
    collisions: tuple[str, ...],
    shutdowns: tuple[str, ...],
) -> dict[str, Any]:
    state = environment.state
    if state is None:
        raise RuntimeError("Environment has not been reset.")
    return {
        "contract_version": "collaborative_delivery_v1",
        "episode_id": state.episode_id,
        "frame": state.frame,
        "total_deliveries": state.total_deliveries,
        "per_agent_deliveries": {
            agent.agent_id: agent.deliveries_completed for agent in state.agents
        },
        "collisions": collisions,
        "shutdowns": shutdowns,
        "reward_breakdown": dict(reward_breakdown or {}),
        "terminal_reason": state.terminal_reason,
        "active_task_count": len(state.tasks),
        "tasks": [
            {
                "task_id": task.task_id,
                "pickup_position": task.pickup_position,
                "delivery_position": task.delivery_position,
                "status": task.status,
                "carrier_agent_id": task.carrier_agent_id,
                "created_frame": task.created_frame,
                "claimed_frame": task.claimed_frame,
                "delivered_frame": task.delivered_frame,
            }
            for task in state.tasks
        ],
        "user_score": state.user_score,
        "score_breakdown": dict(state.score_breakdown),
        "human_route_regret_units": state.human_route_regret_units,
        "robot_collision_events": state.robot_collision_events,
        "invalid_move_count": state.invalid_move_count,
    }


def joint_decision_audit(
    *,
    previous: Any,
    next_state: Any,
    pre_move_observations: Mapping[str, Any],
    raw_actions: Mapping[str, str],
    decision_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build immutable S_t/action/S_t+1 evidence with protected invariants."""

    return {
        **dict(decision_metadata or {}),
        "schema_version": DECISION_AUDIT_SCHEMA,
        "episode_id": int(previous.episode_id),
        "decision_frame": int(previous.frame),
        "outcome_frame": int(next_state.frame),
        "pre_move_state_sha256": canonical_sha256(previous),
        "pre_move_observation_sha256": {
            agent_id: canonical_sha256(observation)
            for agent_id, observation in sorted(pre_move_observations.items())
        },
        "joint_action": dict(raw_actions),
        "post_move_state_sha256": canonical_sha256(next_state),
        "same_pre_move_state": True,
        "environment_step_calls": 1,
    }


def _best_alternative_action(
    selected: str,
    distribution: Mapping[str, Any],
) -> str | None:
    actions = tuple(str(item) for item in distribution.get("actions", ACTIONS))
    probabilities = tuple(
        float(item) for item in distribution.get("probabilities", ())
    )
    ranked = sorted(
        zip(probabilities, actions),
        key=lambda item: (-item[0], item[1]),
    )
    return next((action for _, action in ranked if action != selected), None)


def _verified_wait_progress_alternative(
    environment: Any,
    previous: Any,
    agent_id: str,
    executed_actions: Mapping[str, str],
    coordination_plan: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Prove one safe progress move before calling a WAIT avoidable."""

    before = previous.by_id(agent_id)
    if str(executed_actions.get(agent_id, "WAIT")) != "WAIT":
        return None
    # A plan is evidence to inspect, never a blanket exemption.  The physical
    # counterfactual below still audits plan-aligned WAITs against S_t.
    if before.position == environment.layout.charger_position and before.battery < 100.0:
        return None

    task = None
    if before.carrying_task_id is not None:
        task = previous.task_by_id(before.carrying_task_id)
        goal = task.delivery_position
    elif before.route_commitment_task_id is not None:
        task = next(
            (
                item
                for item in previous.tasks
                if item.task_id == before.route_commitment_task_id
                and item.status == "available"
            ),
            None,
        )
        goal = task.pickup_position if task is not None else before.navigation_goal_position
    elif before.navigation_goal_kind == "charge":
        goal = environment.layout.charger_position
    else:
        return None
    current_distance = shortest_path_distance(
        before.position,
        goal,
        environment.config.map_layout_id,
    )
    from .observations import _actor_action_mask

    actor_mask = _actor_action_mask(
        previous,
        before,
        environment.config,
    )
    candidates: list[tuple[int, int, str, tuple[int, int], float]] = []
    for action_index, action in enumerate(ACTIONS):
        if actor_mask[action_index] <= 0.5:
            continue
        delta = MOVE_DELTAS.get(action)
        if delta is None or before.battery <= environment.config.move_battery_cost:
            continue
        trial = dict(executed_actions)
        trial[agent_id] = action
        # A past joint outcome is known when explaining it, but the Actor did
        # not know the teammate's private current-frame action at decision
        # time.  Calling WAIT avoidable therefore requires an alternative
        # that would have remained safe for every legal simultaneous peer
        # action from the same frozen state S_t.
        if not action_is_robustly_safe(
            environment,
            previous,
            executed_actions,
            agent_id,
            action,
        ):
            continue
        targets, _, invalid, collision, _, _ = environment._resolve_motion(
            previous, trial
        )
        if collision or invalid:
            continue
        target = targets[agent_id]
        next_distance = shortest_path_distance(
            target,
            goal,
            environment.config.map_layout_id,
        )
        if next_distance >= current_distance:
            continue
        remaining = before.battery - environment.config.move_battery_cost
        if task is not None:
            required = (
                environment._mission_route_steps(
                    previous,
                    before,
                    task,
                    origin=target,
                )
                * environment.config.move_battery_cost
            )
        else:
            required = (
                next_distance + environment.config.mission_reserve_steps
            ) * environment.config.move_battery_cost
        if remaining + 1e-8 < required:
            continue
        candidates.append(
            (next_distance, action_index, action, target, float(required))
        )
    if not candidates:
        return None
    next_distance, _, action, target, required = min(candidates)
    return {
        "verified": True,
        "action": action,
        "target": target,
        "distance_before": int(current_distance),
        "distance_after": int(next_distance),
        "battery_before": float(before.battery),
        "battery_after": float(
            before.battery - environment.config.move_battery_cost
        ),
        "required_energy_after": required,
    }


def _decision_reason_code(
    environment: Any,
    previous: Any,
    next_state: Any,
    agent_id: str,
    requested_action: str,
    executed_action: str,
    coordination_plan: Mapping[str, Any] | None,
    wait_counterfactual: Mapping[str, Any] | None = None,
    runtime_decision: Mapping[str, Any] | None = None,
) -> str:
    before = previous.by_id(agent_id)
    after = next_state.by_id(agent_id)
    runtime = dict(runtime_decision or {})
    if (
        executed_action == "WAIT"
        and str(runtime.get("mode", "")) == "human_ai_robust_selection"
        and agent_id == "robot_2"
    ):
        candidates = tuple(
            item
            for item in runtime.get("ai_action_candidates", ())
            if isinstance(item, Mapping)
        )
        if any(
            str(item.get("action", "")) in MOVE_DELTAS
            and int(item.get("distance_after", 0))
            < int(item.get("distance_before", 0))
            and bool(item.get("collision_counterfactuals"))
            for item in candidates
        ):
            return "WAIT_FOR_UNKNOWN_PARTICIPANT_ACTION"
    if requested_action != executed_action:
        return "SAFETY_RULE_BLOCKED"
    plan_aligned_for_agent = bool(
        coordination_plan is not None
        and (
            bool(coordination_plan.get("execution_aligned", False))
            or bool(
                dict(coordination_plan.get("agent_execution_aligned", {})).get(
                    agent_id, False
                )
            )
        )
    )
    if coordination_plan is not None and plan_aligned_for_agent:
        if str(coordination_plan.get("plan_kind")) == (
            "short_horizon_charger_reservation"
        ):
            if str(coordination_plan.get("priority_agent_id")) == agent_id:
                return "PRIORITY_ROUTE_PROGRESS"
            return (
                "WAIT_FOR_PRIORITY_PASSAGE"
                if executed_action == "WAIT"
                else "CLEAR_TEAMMATE_ROUTE"
            )
        if str(coordination_plan.get("waiting_agent_id")) == agent_id:
            plan_kind = str(coordination_plan.get("plan_kind"))
            if plan_kind == "same_target_priority":
                return "WAIT_FOR_CONFLICTING_TARGET"
            if plan_kind in {"head_on_priority", "priority_followthrough"}:
                return "WAIT_FOR_PRIORITY_PASSAGE"
            return "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE"
        if (
            str(coordination_plan.get("moving_agent_id")) == agent_id
            and str(coordination_plan.get("plan_kind"))
            == "occupied_route_clearance"
        ):
            return "CLEAR_TEAMMATE_ROUTE"
        if str(coordination_plan.get("moving_agent_id")) == agent_id:
            return "PRIORITY_ROUTE_PROGRESS"
    if (
        executed_action == "WAIT"
        and before.position == environment.layout.charger_position
        and after.battery > before.battery
    ):
        return "CONTINUE_CHARGING"
    if (
        executed_action == "WAIT"
        and before.navigation_goal_kind == "charge"
        and before.position != environment.layout.charger_position
        and any(
            teammate.active
            and teammate.agent_id != agent_id
            and teammate.position == environment.layout.charger_position
            for teammate in previous.agents
        )
    ):
        return "WAIT_FOR_CHARGER_AVAILABILITY"
    if (
        before.position == environment.layout.charger_position
        and after.position != before.position
        and (
            before.charge_mode_active
            or before.navigation_goal_kind == "charge"
            or before.goal_type in {"CHARGING", "LEAVE_CHARGER"}
        )
    ):
        release_threshold = float(
            charge_release_evidence(environment, previous, before)[
                "release_threshold"
            ]
        )
        return (
            "LEAVE_CHARGER_THRESHOLD_MET"
            if before.battery + 1e-8 >= release_threshold
            else "PREMATURE_CHARGER_DEPARTURE"
        )
    if before.navigation_goal_kind == "charge":
        return (
            "CHARGER_ROUTE_PROGRESS"
            if shortest_path_distance(
                after.position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            < shortest_path_distance(
                before.position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            else "CHARGER_ROUTE_WAIT_OR_DETOUR"
        )
    if executed_action == "WAIT":
        return (
            "AVOIDABLE_WAIT_SAFE_PROGRESS_AVAILABLE"
            if wait_counterfactual is not None
            and bool(wait_counterfactual.get("verified", False))
            else "WAIT_NO_VERIFIED_CAUSE"
        )
    if before.carrying_task_id is not None:
        task = previous.task_by_id(before.carrying_task_id)
        return (
            "DELIVERY_ROUTE_PROGRESS"
            if shortest_path_distance(
                after.position,
                task.delivery_position,
                environment.config.map_layout_id,
            )
            < shortest_path_distance(
                before.position,
                task.delivery_position,
                environment.config.map_layout_id,
            )
            else "POLICY_MISSION_DETOUR"
        )
    if before.route_commitment_task_id is not None:
        task = next(
            (
                item
                for item in previous.tasks
                if item.task_id == before.route_commitment_task_id
            ),
            None,
        )
        if task is not None:
            return (
                "PICKUP_ROUTE_PROGRESS"
                if shortest_path_distance(
                    after.position,
                    task.pickup_position,
                    environment.config.map_layout_id,
                )
                < shortest_path_distance(
                    before.position,
                    task.pickup_position,
                    environment.config.map_layout_id,
                )
                else "POLICY_MISSION_DETOUR"
            )
    return "ENERGY_SAFE_TASK_SELECTION"


def validate_decision_trace(
    environment: Any,
    previous: Any,
    next_state: Any,
    trace: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return factual failures that make a generated reason unsafe to show."""

    failures: list[str] = []
    if trace.get("pre_state_hash") != canonical_sha256(previous):
        failures.append("pre_state_hash_mismatch")
    if trace.get("post_state_hash") != canonical_sha256(next_state):
        failures.append("post_state_hash_mismatch")
    if int(trace.get("decision_frame", -1)) + 1 != int(
        trace.get("outcome_frame", -1)
    ):
        failures.append("non_atomic_frame_boundary")
    for agent_id, decision in dict(trace.get("agents", {})).items():
        before = previous.by_id(agent_id)
        after = next_state.by_id(agent_id)
        action = str(decision.get("resolved_action", "WAIT"))
        delta = MOVE_DELTAS.get(action)
        expected = (
            before.position
            if delta is None
            else (before.position[0] + delta[0], before.position[1] + delta[1])
        )
        if action in MOVE_DELTAS and after.position != expected:
            failures.append(f"{agent_id}:action_position_mismatch")
        if action == "WAIT" and after.position != before.position:
            failures.append(f"{agent_id}:wait_position_changed")
        reason = str(decision.get("primary_reason_code", ""))
        plan = decision.get("joint_coordination_plan")
        if reason in {
            "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE",
            "WAIT_FOR_CONFLICTING_TARGET",
            "WAIT_FOR_PRIORITY_PASSAGE",
            "CLEAR_TEAMMATE_ROUTE",
            "PRIORITY_ROUTE_PROGRESS",
        } and not (
            isinstance(plan, Mapping)
            and (
                bool(plan.get("execution_aligned", False))
                or bool(
                    dict(plan.get("agent_execution_aligned", {})).get(
                        agent_id, False
                    )
                )
            )
        ):
            failures.append(f"{agent_id}:coordination_reason_without_plan")
        if reason == "WAIT_FOR_UNKNOWN_PARTICIPANT_ACTION":
            uncertainty = decision.get("human_action_uncertainty", {})
            if not (
                isinstance(uncertainty, Mapping)
                and uncertainty.get("participant_action_known_at_decision_time")
                is False
                and bool(uncertainty.get("collision_counterfactuals"))
            ):
                failures.append(f"{agent_id}:human_wait_without_counterfactual")
        if reason == "LEAVE_CHARGER_THRESHOLD_MET":
            charging = decision.get("charging_state", {})
            threshold = float(charging.get("release_threshold", 101.0))
            if before.position != environment.layout.charger_position:
                failures.append(f"{agent_id}:charger_departure_source_mismatch")
            if before.battery + 1e-8 < threshold:
                failures.append(f"{agent_id}:charger_release_below_threshold")
        if reason == "PREMATURE_CHARGER_DEPARTURE":
            charging = decision.get("charging_state", {})
            threshold = float(charging.get("release_threshold", 101.0))
            if before.position != environment.layout.charger_position:
                failures.append(f"{agent_id}:premature_departure_source_mismatch")
            if before.battery + 1e-8 >= threshold:
                failures.append(f"{agent_id}:false_premature_departure_claim")
        if reason in {
            "CHARGER_ROUTE_PROGRESS",
            "DELIVERY_ROUTE_PROGRESS",
            "PICKUP_ROUTE_PROGRESS",
        }:
            effect = decision.get("direct_effect", {})
            if int(effect.get("distance_after", 0)) >= int(
                effect.get("distance_before", 0)
            ):
                failures.append(f"{agent_id}:false_progress_claim")
    return tuple(failures)


def build_decision_trace(
    environment: Any,
    *,
    previous: Any,
    next_state: Any,
    raw_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
    action_resolution: Mapping[str, Mapping[str, Any]],
    decision_metadata: Mapping[str, Any] | None,
    frozen_missions: Mapping[str, Any],
    coordination_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one trace whose causes are restricted to the frozen ``S_t``."""

    metadata = dict(decision_metadata or {})
    distributions = dict(metadata.get("action_distributions", {}))
    policy_actions = dict(metadata.get("policy_actions", {}))
    selected_actions = dict(metadata.get("selected_actions", {}))
    runtime_decision = dict(metadata.get("runtime_decision", {}))
    safe_joint_actions = tuple(
        item
        for item in runtime_decision.get("safe_joint_actions", ())
        if isinstance(item, Mapping)
    )
    rejected_joint_actions = tuple(
        item
        for item in runtime_decision.get("rejected_joint_actions", ())
        if isinstance(item, Mapping)
    )
    agents: dict[str, Any] = {}
    for before in previous.agents:
        agent_id = before.agent_id
        after = next_state.by_id(agent_id)
        distribution = dict(distributions.get(agent_id, {}))
        battery_feasibility = []
        for task in sorted(previous.tasks, key=lambda item: item.task_id):
            if task.status not in {"available", "carried"}:
                continue
            if task.status == "carried" and task.carrier_agent_id != agent_id:
                continue
            route_steps = environment._mission_route_steps(
                previous,
                before,
                task,
                origin=before.position,
            )
            required = route_steps * environment.config.move_battery_cost
            battery_feasibility.append(
                {
                    "task_id": task.task_id,
                    "route_steps": float(route_steps),
                    "required_energy": float(required),
                    "battery": float(before.battery),
                    "energy_slack": float(before.battery - required),
                    "safe": bool(before.battery + 1e-8 >= required),
                }
            )
        frozen_mission = frozen_missions.get(agent_id)
        if isinstance(frozen_mission, Mapping):
            goal_position = tuple(frozen_mission.get("goal_position", before.position))
            resolved_goal_kind = str(frozen_mission.get("goal_kind", "wait"))
            resolved_goal_id = frozen_mission.get("task_id")
        else:
            goal_position = before.navigation_goal_position
            resolved_goal_kind = before.navigation_goal_kind
            resolved_goal_id = before.route_commitment_task_id or before.goal_id
        distance_before = shortest_path_distance(
            before.position,
            goal_position,
            environment.config.map_layout_id,
        )
        distance_after = shortest_path_distance(
            after.position,
            goal_position,
            environment.config.map_layout_id,
        )
        policy_action = str(policy_actions.get(agent_id, raw_actions.get(agent_id, "WAIT")))
        selected = str(selected_actions.get(agent_id, raw_actions.get(agent_id, "WAIT")))
        wait_counterfactual = _verified_wait_progress_alternative(
            environment,
            previous,
            agent_id,
            executed_actions,
            coordination_plan,
        )
        distribution_actions = tuple(
            str(action) for action in distribution.get("actions", ACTIONS)
        )
        distribution_mask = tuple(
            float(allowed)
            for allowed in distribution.get(
                "action_mask", (1.0,) * len(distribution_actions)
            )
        )
        legal_actions = tuple(
            action
            for action, allowed in zip(distribution_actions, distribution_mask)
            if allowed > 0.5
        )
        ai_candidates = tuple(
            item
            for item in runtime_decision.get("ai_action_candidates", ())
            if isinstance(item, Mapping)
        )
        if str(runtime_decision.get("mode", "")) == "human_ai_robust_selection" and agent_id == "robot_2":
            safe_actions = tuple(
                str(item.get("action"))
                for item in ai_candidates
                if bool(item.get("safe_for_all_participant_actions", False))
            )
            collision_counterfactuals = tuple(
                {
                    "ai_action": str(item.get("action", "")),
                    **dict(counterfactual),
                }
                for item in ai_candidates
                for counterfactual in item.get("collision_counterfactuals", ())
                if isinstance(counterfactual, Mapping)
            )
        elif safe_joint_actions:
            safe_actions = tuple(sorted({
                str(item.get("actions", {}).get(agent_id, "WAIT"))
                for item in safe_joint_actions
                if isinstance(item.get("actions"), Mapping)
            }))
            collision_counterfactuals = ()
        else:
            safe_actions = legal_actions
            collision_counterfactuals = ()
        agents[agent_id] = {
            "base_goal": {
                "goal_type": before.goal_type,
                "goal_id": before.goal_id,
                "goal_since": int(before.goal_since),
                "navigation_kind": before.navigation_goal_kind,
                "position": before.navigation_goal_position,
            },
            # Backward-compatible alias for archived explanation consumers.
            "frozen_goal": {
                "goal_type": before.goal_type,
                "goal_id": before.goal_id,
                "goal_since": int(before.goal_since),
                "navigation_kind": before.navigation_goal_kind,
                "position": before.navigation_goal_position,
            },
            "resolved_goal": {
                "goal_type": resolved_goal_kind,
                "goal_id": resolved_goal_id,
                "position": goal_position,
            },
            "committed_task": (
                before.route_commitment_task_id or before.goal_id
            ),
            "carrying_task_id": before.carrying_task_id,
            "battery_feasibility": battery_feasibility,
            "charging_state": {
                "active": bool(before.charge_mode_active),
                "at_charger": bool(
                    before.position == environment.layout.charger_position
                ),
                "battery": float(before.battery),
                "requires_charge": bool(
                    before.navigation_goal_kind == "charge"
                ),
                **charge_release_evidence(environment, previous, before),
                "reason": before.charging_reason,
            },
            "joint_coordination_plan": (
                dict(coordination_plan) if coordination_plan is not None else None
            ),
            "candidate_actions": list(
                distribution.get("actions", ACTIONS)
            ),
            "legal_actions": list(legal_actions),
            "safe_actions": list(safe_actions),
            "policy_logits": list(distribution.get("logits", ())),
            "policy_probabilities": list(
                distribution.get("probabilities", ())
            ),
            "safety_mask": list(
                distribution.get("action_mask", ())
            ),
            "policy_action": policy_action,
            "selected_action": selected,
            "executed_action": str(executed_actions.get(agent_id, "WAIT")),
            "resolved_action": str(
                executed_actions.get(agent_id, "WAIT")
            ),
            "action_resolution": dict(action_resolution.get(agent_id, {})),
            "primary_reason_code": _decision_reason_code(
                environment,
                previous,
                next_state,
                agent_id,
                selected,
                str(executed_actions.get(agent_id, "WAIT")),
                coordination_plan,
                wait_counterfactual,
                runtime_decision,
            ),
            "wait_counterfactual": wait_counterfactual,
            "alternative_action": _best_alternative_action(
                selected,
                distribution,
            ),
            "goal_switch_reason": after.goal_switch_reason,
            "resulting_goal": {
                "goal_type": after.goal_type,
                "goal_id": after.goal_id,
                "goal_since": int(after.goal_since),
            },
            "direct_effect": {
                "position_before": before.position,
                "position_after": after.position,
                "goal_position": goal_position,
                "distance_before": int(distance_before),
                "distance_after": int(distance_after),
                "battery_before": float(before.battery),
                "battery_after": float(after.battery),
            },
            "frozen_mission": frozen_missions.get(agent_id),
            "human_action_uncertainty": {
                "participant_action_known_at_decision_time": runtime_decision.get(
                    "participant_action_known_at_decision_time"
                ),
                "participant_legal_actions": list(
                    runtime_decision.get("participant_legal_actions", ())
                ),
                "collision_counterfactuals": list(collision_counterfactuals),
            },
            "rejected_action_reasons": [
                {
                    "action": str(item.get("action", "")),
                    "collision_counterfactuals": list(
                        item.get("collision_counterfactuals", ())
                    ),
                    "energy_violation": bool(item.get("energy_violation", False)),
                    "recent_unproductive_charger_reentry": bool(
                        item.get("recent_unproductive_charger_reentry", False)
                    ),
                }
                for item in ai_candidates
                if str(item.get("action", "")) != selected
            ],
        }
    trace: dict[str, Any] = {
        "schema_version": "warehouse-decision-trace.v3",
        "episode_id": int(previous.episode_id),
        "decision_frame": int(previous.frame),
        "outcome_frame": int(next_state.frame),
        "pre_state_hash": canonical_sha256(previous),
        "post_state_hash": canonical_sha256(next_state),
        "same_frozen_state_for_all_agents": True,
        "environment_step_calls": 1,
        "decision_source": metadata.get("decision_source", "unspecified"),
        "pre_state": asdict(previous),
        "policy_actions": policy_actions,
        "selected_actions": selected_actions or dict(raw_actions),
        "executed_actions": dict(executed_actions),
        "safe_joint_actions": list(safe_joint_actions),
        "rejected_joint_actions": list(rejected_joint_actions),
        "runtime_decision": runtime_decision,
        "tasks": tuple(
            {
                "task_id": task.task_id,
                "status": task.status,
                "carrier_agent_id": task.carrier_agent_id,
                "pickup_position": task.pickup_position,
                "delivery_position": task.delivery_position,
            }
            for task in sorted(previous.tasks, key=lambda item: item.task_id)
        ),
        "agents": agents,
    }
    failures = validate_decision_trace(
        environment,
        previous,
        next_state,
        trace,
    )
    trace["fact_validation_failures"] = failures
    trace["fact_valid"] = not failures
    return trace
