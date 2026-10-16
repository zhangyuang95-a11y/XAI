"""Final efficiency guard for offline teacher labels only."""

from __future__ import annotations

from typing import Any, Mapping

from . import credit_assignment as credit
from .energy_management import (
    charger_departure_progress,
    charger_service_required,
)
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
            or not charger_service_required(environment, state, agent)
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

    def best_robust_progress_action(
        current: Mapping[str, str],
        agent_id: str,
        mission: credit.FrozenMission,
    ) -> str | None:
        """Return a decentralized-safe action that strictly advances mission.

        The teacher may know the complete joint label, but the independent
        Actor does not observe its teammate's private action for the current
        frame.  Candidate supervision therefore has to remain safe for every
        legal simultaneous teammate action from the same frozen state S_t.
        """

        agent = state.by_id(agent_id)
        reserved_charger_cells: set[tuple[int, int]] = set()
        for teammate in state.agents:
            if (
                teammate.agent_id == agent_id
                or not environment._requires_charge(state, teammate)
            ):
                continue
            teammate_distance = shortest_path_distance(
                teammate.position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            reserved_charger_cells.add(environment.layout.charger_position)
            for action_delta in MOVE_DELTAS.values():
                candidate = (
                    teammate.position[0] + action_delta[0],
                    teammate.position[1] + action_delta[1],
                )
                if (
                    shortest_path_distance(
                        teammate.position,
                        candidate,
                        environment.config.map_layout_id,
                    )
                    + shortest_path_distance(
                        candidate,
                        environment.layout.charger_position,
                        environment.config.map_layout_id,
                    )
                    == teammate_distance
                ):
                    reserved_charger_cells.add(candidate)
        before = credit.mission_goal_distance(
            environment,
            state,
            agent,
            mission,
            agent.position,
        )
        candidates: list[tuple[float, int, str]] = []
        for action_index, action in enumerate(ACTIONS):
            if action == "WAIT" or not credit.action_is_robustly_safe(
                environment,
                state,
                current,
                agent_id,
                action,
            ):
                continue
            trial = dict(current)
            trial[agent_id] = action
            targets, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if collision or agent_id in invalid:
                continue
            if (
                mission.goal_kind != "charge"
                and targets[agent_id] in reserved_charger_cells
            ):
                continue
            remaining_battery = agent.battery - environment.config.move_battery_cost
            if remaining_battery <= 0.0:
                # Reaching zero ends the round before a robot can receive a
                # charge.  A geometrically shorter action is therefore not a
                # feasible counterfactual.
                continue
            if mission.goal_kind == "charge":
                required_after = (
                    shortest_path_distance(
                        targets[agent_id],
                        environment.layout.charger_position,
                        environment.config.map_layout_id,
                    )
                    * environment.config.move_battery_cost
                )
            elif mission.task is not None:
                required_after = (
                    environment._mission_route_steps(
                        state,
                        agent,
                        mission.task,
                        origin=targets[agent_id],
                    )
                    * environment.config.move_battery_cost
                )
            else:
                required_after = (
                    shortest_path_distance(
                        targets[agent_id],
                        mission.goal_position,
                        environment.config.map_layout_id,
                    )
                    * environment.config.move_battery_cost
                )
            if remaining_battery + 1e-8 < required_after:
                # Do not replace a necessary hold with a move that makes the
                # frozen mission energetically impossible on the next frame.
                continue
            after = credit.mission_goal_distance(
                environment,
                state,
                agent,
                mission,
                targets[agent_id],
            )
            if after + 1e-9 < before:
                candidates.append((after, action_index, action))
        return min(candidates)[2] if candidates else None

    missions, diagnosis = diagnose(actions)
    # Correct every geometric detour, not just the loaded subset.  Prefer a
    # robust mission-progressing move; otherwise hold when WAIT is safe.  The
    # previous implementation merely counted ordinary empty-robot detours,
    # allowing them to leak into the imitation labels.
    for agent_id in (*diagnosis[2], *diagnosis[3]):
        mission = missions.get(agent_id)
        replacement = (
            best_robust_progress_action(actions, agent_id, mission)
            if mission is not None
            else None
        )
        trial = dict(actions)
        trial[agent_id] = replacement or "WAIT"
        _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
        if not collision and agent_id not in invalid:
            actions = trial

    # Replace WAIT only when the exact teammate-conditioned resolver has a
    # collision-free move with strictly lower frozen-goal distance.
    for _ in range(2):
        missions, diagnosis = diagnose(actions)
        changed = False
        for agent_id in diagnosis[1]:
            mission = missions.get(agent_id)
            if mission is None:
                continue
            replacement = best_robust_progress_action(
                actions,
                agent_id,
                mission,
            )
            if replacement is not None:
                actions[agent_id] = replacement
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
                charging_ids = {
                    agent.agent_id
                    for agent in state.agents
                    if environment._requires_charge(state, agent)
                }
                if charging_ids and any(
                    agent.agent_id not in charging_ids
                    and agent.position != environment.layout.charger_position
                    and targets[agent.agent_id]
                    == environment.layout.charger_position
                    for agent in state.agents
                ):
                    # The single charger is a reserved resource while a peer
                    # has a verified charging mission.  An unrelated robot
                    # may use the apron but may not steal the charger cell.
                    continue
                if any(
                    agent.position != environment.layout.charger_position
                    and targets[agent.agent_id]
                    == environment.layout.charger_position
                    and agent.last_charger_departure_frame is not None
                    and state.frame - agent.last_charger_departure_frame <= 6
                    and not any(charger_departure_progress(state, agent))
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
                            + environment.config.mission_reserve_steps
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
        made_progress = any(charger_departure_progress(state, agent))
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
    # A decentralized robot cannot know that its teammate will vacate an
    # occupied cell in the current frame.  Earlier supervision labelled
    # simultaneous follow-through (one robot leaves while the other enters
    # its S_t cell), which is collision-free only after observing the peer's
    # still-private action.  Hold the follower for one frozen transition and
    # let it enter on the next frame.  This is an offline label invariant, not
    # a runtime action rewrite.
    targets, _, _, _, _, _ = environment._resolve_motion(state, actions)
    occupied_followers = {
        agent.agent_id
        for agent in state.agents
        for teammate in state.agents
        if teammate.agent_id != agent.agent_id
        and targets[agent.agent_id] == teammate.position
        and targets[agent.agent_id] != agent.position
    }
    if occupied_followers:
        actions = {
            agent_id: ("WAIT" if agent_id in occupied_followers else action)
            for agent_id, action in actions.items()
        }

    # Every offline label must belong to the Actor's static action support.
    # Special charger branches may return before the generic joint search and
    # a later efficiency rewrite can otherwise leave a wall-facing move in a
    # rare scenario.  Such a label is impossible for the masked Actor to
    # represent and makes cross-entropy overflow.  Fall back to WAIT for an
    # invalid mover; if that exposes an occupied-stationary conflict, hold the
    # complete joint action for this one supervision row.
    _, _, invalid, collision, _, _ = environment._resolve_motion(state, actions)
    if invalid:
        actions = {
            agent_id: ("WAIT" if agent_id in invalid else action)
            for agent_id, action in actions.items()
        }
        _, _, _, collision, _, _ = environment._resolve_motion(state, actions)
    if collision:
        actions = {agent_id: "WAIT" for agent_id in environment.agent_ids}

    # The occupied-follower and invalid-label protections above intentionally
    # run after the main efficiency pass.  They can introduce a new WAIT, so
    # the completed label must be audited once more.  Replace only waits for
    # which a strictly progressing action is robust to every legal teammate
    # action; this preserves the no-future-observation contract.
    for _ in range(2):
        missions, diagnosis = diagnose(actions)
        changed = False
        for agent_id in diagnosis[1]:
            mission = missions.get(agent_id)
            if mission is None:
                continue
            replacement = best_robust_progress_action(
                actions,
                agent_id,
                mission,
            )
            if replacement is None:
                continue
            trial = dict(actions)
            trial[agent_id] = replacement
            _, _, invalid, collision, _, _ = environment._resolve_motion(state, trial)
            if collision or agent_id in invalid:
                continue
            actions = trial
            changed = True
        if not changed:
            break
    return actions
