"""Strict frozen-state diagnostics shared by evaluation and trace tooling."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .coordination import is_necessary_urgent_charger_clearance
from .coordination_priority import single_lane_egress_agent_id
from .navigation import MOVE_DELTAS, shortest_path_distance
from .observations import _actor_visible_goal
from .transition_audit import (
    action_is_robustly_safe,
    necessary_participant_standoff_clearance,
    necessary_teammate_route_clearance,
    wait_is_robustly_safe,
)


def _necessary_single_lane_clearance(
    environment: Any,
    state: Any,
    actions: Mapping[str, str],
    agent: Any,
) -> bool:
    """Whether ``agent`` performs the public shelf-arm egress clearance."""

    goals = {
        item.agent_id: _actor_visible_goal(state, item)[1]
        for item in state.agents
    }
    egress_id = single_lane_egress_agent_id(
        state,
        environment.config,
        goal_positions=goals,
    )
    if egress_id is None or agent.agent_id == egress_id:
        return False
    action = str(actions.get(agent.agent_id, "WAIT"))
    delta = MOVE_DELTAS.get(action)
    if delta is None:
        return False
    egress = state.by_id(egress_id)
    spine_column = environment.layout.charger_position[1]
    inward_action = "RIGHT" if egress.position[1] < spine_column else "LEFT"
    inward_delta = MOVE_DELTAS[inward_action]
    inward_target = (
        egress.position[0] + inward_delta[0],
        egress.position[1] + inward_delta[1],
    )
    held = {item.agent_id: "WAIT" for item in state.agents}
    if (
        inward_target != agent.position
        and action_is_robustly_safe(
            environment,
            state,
            held,
            egress_id,
            inward_action,
        )
    ):
        return False
    target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
    current_spine_distance = abs(agent.position[1] - spine_column)
    target_spine_distance = abs(target[1] - spine_column)
    return bool(
        target_spine_distance < current_spine_distance
        or current_spine_distance == 0
    )


def _necessary_preemptive_charger_clearance(
    environment: Any,
    state: Any,
    actions: Mapping[str, str],
    agent: Any,
) -> bool:
    """Whether a charged occupant exits before an urgent peer reaches it."""

    action = str(actions.get(agent.agent_id, "WAIT"))
    delta = MOVE_DELTAS.get(action)
    charger = environment.layout.charger_position
    if delta is None or agent.position != charger or environment._requires_charge(state, agent):
        return False
    teammate = next(
        item for item in state.agents if item.agent_id != agent.agent_id
    )
    if (
        not environment._requires_charge(state, teammate)
        or shortest_path_distance(
            teammate.position,
            charger,
            environment.config.map_layout_id,
        )
        > 2
    ):
        return False
    target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
    candidates: list[int] = []
    held = {item.agent_id: "WAIT" for item in state.agents}
    for candidate_action, move in MOVE_DELTAS.items():
        candidate = (agent.position[0] + move[0], agent.position[1] + move[1])
        if (
            environment.layout.is_passable(candidate)
            and action_is_robustly_safe(
                environment,
                state,
                held,
                agent.agent_id,
                candidate_action,
            )
        ):
            candidates.append(
                shortest_path_distance(
                    candidate,
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
            )
    return bool(
        candidates
        and environment.layout.is_passable(target)
        and shortest_path_distance(
            target,
            agent.navigation_goal_position,
            environment.config.map_layout_id,
        )
        == min(candidates)
    )


def avoidable_loaded_delivery_detour_agents(
    environment: Any,
    state: Any,
    actions: Mapping[str, str],
    *,
    excluded_agent_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return loaded robots that move away when robust WAIT was available.

    The check uses only the frozen transition-start state and the already
    locked joint action. Necessary charger, teammate-route, and participant
    clearance is excluded before testing the WAIT counterfactual.
    """

    excluded = set(excluded_agent_ids)
    final_targets = environment._resolve_motion(state, actions)[0]
    avoidable: list[str] = []
    for agent in state.agents:
        if (
            agent.agent_id in excluded
            or agent.carrying_task_id is None
            or agent.navigation_goal_kind != "delivery"
            or environment._requires_charge(state, agent)
        ):
            continue
        current_distance = shortest_path_distance(
            agent.position,
            agent.navigation_goal_position,
            environment.config.map_layout_id,
        )
        final_distance = shortest_path_distance(
            final_targets[agent.agent_id],
            agent.navigation_goal_position,
            environment.config.map_layout_id,
        )
        if final_distance <= current_distance:
            continue
        if is_necessary_urgent_charger_clearance(environment, state, agent):
            continue
        if _necessary_preemptive_charger_clearance(
            environment,
            state,
            actions,
            agent,
        ):
            continue
        if _necessary_single_lane_clearance(
            environment,
            state,
            actions,
            agent,
        ):
            continue
        if necessary_teammate_route_clearance(environment, state, agent):
            continue
        if necessary_participant_standoff_clearance(
            environment,
            state,
            agent,
            candidate_action=actions[agent.agent_id],
        ):
            continue
        held_actions = {**actions, agent.agent_id: "WAIT"}
        if (
            not environment._resolve_motion(state, held_actions)[3]
            and wait_is_robustly_safe(
                environment,
                state,
                actions,
                agent.agent_id,
            )
        ):
            avoidable.append(agent.agent_id)
    return tuple(avoidable)
