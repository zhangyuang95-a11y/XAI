"""Stable public transition information and causal decision audit payloads."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_protocol import DECISION_AUDIT_SCHEMA, canonical_sha256
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

    for teammate in state.agents:
        if teammate.agent_id == clearing_agent.agent_id or not teammate.active:
            continue
        goal = teammate.navigation_goal_position
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
    own_mask = environment.action_masks()[clearing_agent.agent_id]
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
