"""Final efficiency guard for offline teacher labels only."""

from __future__ import annotations

from typing import Any, Mapping

from . import credit_assignment as credit
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance


def _urgent_charge(environment: Any, agent: Any) -> bool:
    if agent.navigation_goal_kind != "charge":
        return False
    distance = shortest_path_distance(
        agent.position,
        agent.navigation_goal_position,
        environment.config.map_layout_id,
    )
    slack = agent.battery - distance * environment.config.move_battery_cost
    return bool(slack <= environment.config.charge_per_wait)


def teacher_efficiency_guard(
    environment: Any,
    proposed_actions: Mapping[str, str],
) -> dict[str, str]:
    """Remove avoidable stalls and loaded detours from every teacher branch."""

    state = environment.get_state()
    actions = {
        agent_id: str(proposed_actions.get(agent_id, "WAIT"))
        for agent_id in environment.agent_ids
    }
    if not environment.config.teacher_efficiency_guard_enabled:
        return actions

    # When two depleted robots reach an empty single-cell charger together,
    # reserve it for the one with the least remaining energy after entry.
    # Geometric tie-breaking alone can otherwise strand the urgent robot on
    # the apron with exactly one move of battery left.
    if not any(
        agent.position == environment.layout.charger_position
        for agent in state.agents
    ):
        entry_options: list[tuple[float, str, str]] = []
        for agent in state.agents:
            if not environment._requires_charge(state, agent):
                continue
            for action, delta in MOVE_DELTAS.items():
                target = (
                    agent.position[0] + delta[0],
                    agent.position[1] + delta[1],
                )
                if target == environment.layout.charger_position:
                    entry_options.append(
                        (
                            agent.battery
                            - environment.config.move_battery_cost,
                            agent.agent_id,
                            action,
                        )
                    )
        if len(entry_options) > 1:
            _, urgent_id, entry_action = min(entry_options)
            reserved = {
                agent_id: (
                    entry_action
                    if agent_id == urgent_id
                    else "WAIT"
                )
                for agent_id in environment.agent_ids
            }
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                reserved,
            )
            if not collision and not invalid:
                actions = reserved

    # Do not teach an undercharged occupant to leave an otherwise uncontested
    # station.  That one premature step makes the next frozen mission point
    # straight back to the charger and creates a deterministic two-step loop.
    for agent in state.agents:
        if (
            agent.position != environment.layout.charger_position
            or actions[agent.agent_id] not in MOVE_DELTAS
            or not environment._requires_charge(state, agent)
        ):
            continue
        inbound_queue = any(
            teammate.agent_id != agent.agent_id
            and environment._requires_charge(state, teammate)
            and shortest_path_distance(
                teammate.position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            <= 2
            for teammate in state.agents
        )
        held = dict(actions)
        held[agent.agent_id] = "WAIT"
        _, _, invalid, collision, _, _ = environment._resolve_motion(state, held)
        if not inbound_queue and not collision and agent.agent_id not in invalid:
            actions = held

    def diagnose(current: Mapping[str, str]):
        targets, executed, _, _, collision_kind, intended = (
            environment._resolve_motion(state, current)
        )
        events = environment._coordination_events(
            state,
            current,
            executed,
            intended,
            collision_kind,
        )
        missions = credit.frozen_training_missions(environment, state)
        return missions, credit.counterfactual_action_regrets(
            environment,
            state,
            current,
            executed,
            targets,
            missions,
            events,
        )

    _, diagnosis = diagnose(actions)
    for agent_id in diagnosis[3]:
        held = dict(actions)
        held[agent_id] = "WAIT"
        _, _, invalid, collision, _, _ = environment._resolve_motion(state, held)
        if not collision and agent_id not in invalid:
            actions = held

    # Replace WAIT only when the exact teammate-conditioned resolver has a
    # collision-free move with strictly lower frozen-goal distance.
    for _ in range(2):
        missions, diagnosis = diagnose(actions)
        changed = False
        for agent_id in diagnosis[1]:
            agent = state.by_id(agent_id)
            mission = missions.get(agent_id)
            if mission is None:
                continue
            candidates: list[tuple[float, int, str]] = []
            for action_index, action in enumerate(ACTIONS):
                if action == "WAIT":
                    continue
                trial = dict(actions)
                trial[agent_id] = action
                targets, _, invalid, collision, _, _ = environment._resolve_motion(
                    state,
                    trial,
                )
                if collision or agent_id in invalid:
                    continue
                candidates.append(
                    (
                        credit.mission_goal_distance(
                            environment,
                            state,
                            agent,
                            mission,
                            targets[agent_id],
                        ),
                        action_index,
                        action,
                    )
                )
            if candidates:
                actions[agent_id] = min(candidates)[2]
                changed = True
        if not changed:
            break

    if all(action == "WAIT" for action in actions.values()) and not any(
        agent.position == environment.layout.charger_position
        and agent.battery < 100.0
        for agent in state.agents
    ):
        # Adjacent follow-through can require both robots to move.  Enumerate
        # the 5x5 joint set and retain energy safety and charger-cycle guards.
        missions = credit.frozen_training_missions(environment, state)
        joint_candidates: list[tuple[float, int, int, dict[str, str]]] = []
        left_id, right_id = environment.agent_ids
        for left_index, left_action in enumerate(ACTIONS):
            for right_index, right_action in enumerate(ACTIONS):
                candidate = {left_id: left_action, right_id: right_action}
                targets, _, invalid, collision, _, _ = environment._resolve_motion(
                    state,
                    candidate,
                )
                if collision or invalid:
                    continue
                if any(
                    agent.position != environment.layout.charger_position
                    and targets[agent.agent_id]
                    == environment.layout.charger_position
                    and agent.last_charger_departure_frame is not None
                    and state.frame - agent.last_charger_departure_frame <= 6
                    and agent.deliveries_completed
                    <= agent.deliveries_at_last_charger_departure
                    and agent.carrying_task_id
                    == agent.carrying_task_at_last_charger_departure
                    for agent in state.agents
                ):
                    continue
                weighted_progress = 0.0
                unsafe_urgent_move = False
                for agent in state.agents:
                    mission = missions.get(agent.agent_id)
                    if mission is None:
                        continue
                    before = credit.mission_goal_distance(
                        environment,
                        state,
                        agent,
                        mission,
                        agent.position,
                    )
                    after = credit.mission_goal_distance(
                        environment,
                        state,
                        agent,
                        mission,
                        targets[agent.agent_id],
                    )
                    urgent = _urgent_charge(environment, agent)
                    weight = (
                        4.0
                        if urgent
                        else 3.0
                        if agent.carrying_task_id is not None
                        else 1.0
                    )
                    weighted_progress += weight * (before - after)
                    if urgent:
                        remaining = agent.battery - (
                            environment.config.move_battery_cost
                            if candidate[agent.agent_id] in MOVE_DELTAS
                            else 0.0
                        )
                        required = (
                            shortest_path_distance(
                                targets[agent.agent_id],
                                environment.layout.charger_position,
                                environment.config.map_layout_id,
                            )
                            + environment.config.battery_safety_margin
                        ) * environment.config.move_battery_cost
                        unsafe_urgent_move = unsafe_urgent_move or (
                            after > before or remaining < required
                        )
                if unsafe_urgent_move or weighted_progress <= 0.0:
                    continue
                joint_candidates.append(
                    (-weighted_progress, left_index, right_index, candidate)
                )
        if joint_candidates:
            actions = min(joint_candidates, key=lambda item: item[:3])[3]

    # A robot that just yielded the single charger must remain on the apron
    # until it has made mission progress.  Re-entering immediately blocks the
    # inbound teammate and is the second source of teacher return cycles.
    missions = credit.frozen_training_missions(environment, state)
    targets, _, _, _, _, _ = environment._resolve_motion(state, actions)
    for agent in state.agents:
        made_progress = bool(
            agent.deliveries_completed > agent.deliveries_at_last_charger_departure
            or (
                agent.carrying_task_id is not None
                and agent.carrying_task_id
                != agent.carrying_task_at_last_charger_departure
            )
        )
        if (
            agent.position == environment.layout.charger_position
            or targets[agent.agent_id] != environment.layout.charger_position
            or agent.last_charger_departure_frame is None
            or state.frame - agent.last_charger_departure_frame > 6
            or made_progress
        ):
            continue
        mission = missions.get(agent.agent_id)
        alternatives: list[tuple[float, int, str]] = []
        for index, action in enumerate(ACTIONS):
            trial = dict(actions)
            trial[agent.agent_id] = action
            trial_targets, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if (
                collision
                or agent.agent_id in invalid
                or trial_targets[agent.agent_id]
                == environment.layout.charger_position
            ):
                continue
            distance = (
                credit.mission_goal_distance(
                    environment,
                    state,
                    agent,
                    mission,
                    trial_targets[agent.agent_id],
                )
                if mission is not None
                else 0.0
            )
            alternatives.append((distance, index, action))
        if alternatives:
            actions[agent.agent_id] = min(alternatives)[2]
    return actions
