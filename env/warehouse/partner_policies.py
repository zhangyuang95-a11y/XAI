"""Causal participant surrogates used only for training and evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .coordination import (
    is_necessary_urgent_charger_clearance,
    stable_coordination_actions,
)
from .coordination_priority import single_lane_egress_agent_id
from .energy_management import charger_service_required
from .environment import WarehouseMultiAgentEnv
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from .transition_audit import (
    necessary_participant_standoff_clearance,
    necessary_teammate_route_clearance,
)


PARTNER_PROFILES = (
    "coordinated",
    "goal_directed",
    "cautious",
    "hesitant",
)


def _goal_directed_action(environment: WarehouseMultiAgentEnv) -> str:
    state = environment.get_state()
    agent = state.by_id(environment.config.human_agent_id)
    teammate = next(
        item for item in state.agents if item.agent_id != agent.agent_id
    )
    # The public charge mode is already frozen in S_t and must dominate an
    # offline task matching. Calling ``_frozen_route_goal`` directly can
    # otherwise match an old pickup to an undercharged participant that is
    # still sitting on the station, making the surrogate leave prematurely
    # and return two frames later. This uses no current-frame action.
    goal = (
        environment.layout.charger_position
        if charger_service_required(environment, state, agent)
        else environment._frozen_route_goal(
            state,
            agent.agent_id,
            prioritize_old_tasks=True,
        )
    )
    if goal is None:
        return "WAIT"
    mask = environment.action_masks()[agent.agent_id]
    candidates: list[tuple[int, int, str]] = []
    for order, (action, allowed) in enumerate(zip(ACTIONS, mask)):
        if allowed <= 0.5:
            continue
        if action in MOVE_DELTAS:
            delta = MOVE_DELTAS[action]
            target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
        else:
            target = agent.position
        # A non-coordinated participant cannot assume the teammate will leave
        # its current cell in this simultaneous decision.  Entering it would
        # encode knowledge of an unobserved current action and create an
        # adversarial occupied-stationary collision loop.
        if target == teammate.position and target != agent.position:
            continue
        candidates.append(
            (
                shortest_path_distance(
                    target,
                    goal,
                    environment.config.map_layout_id,
                ),
                order,
                action,
            )
        )
    if not candidates:
        return "WAIT"
    if state.ineffective_joint_wait_streak > 0:
        moving = [item for item in candidates if item[2] != "WAIT"]
        if moving:
            # A goal-directed participant may make one causal retreat after a
            # previously observed joint stall.  Persisting with WAIT forever
            # would be an adversarial frozen policy rather than a plausible
            # participant profile.
            return min(moving)[2]
    return min(candidates)[2]


def participant_surrogate_action(
    environment: WarehouseMultiAgentEnv,
    *,
    profile: str,
    rng: np.random.Generator,
    coordinated_actions: Mapping[str, str] | None = None,
) -> str:
    """Choose from S_t without observing either Actor's current action."""

    probabilities = participant_surrogate_distribution(
        environment,
        profile=profile,
        coordinated_actions=coordinated_actions,
    )
    return str(rng.choice(ACTIONS, p=probabilities))


def participant_surrogate_distribution(
    environment: WarehouseMultiAgentEnv,
    *,
    profile: str,
    coordinated_actions: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Return a profile's action distribution using only frozen ``S_t``."""

    name = str(profile)
    if name not in PARTNER_PROFILES:
        raise ValueError(f"Unknown participant surrogate profile: {name!r}")
    if name == "coordinated":
        shared_actions = (
            dict(coordinated_actions)
            if coordinated_actions is not None
            else stable_coordination_actions(environment)
        )
        action = shared_actions[environment.config.human_agent_id]
        result = np.zeros(len(ACTIONS), dtype=np.float64)
        result[ACTIONS.index(action)] = 1.0
        return result

    directed = _goal_directed_action(environment)
    state = environment.get_state()
    human = state.by_id(environment.config.human_agent_id)
    teammate = next(
        agent for agent in state.agents if agent.agent_id != human.agent_id
    )
    dual_charger_clearance = bool(
        charger_service_required(environment, state, human)
        and charger_service_required(environment, state, teammate)
        and human.position != environment.layout.charger_position
        and teammate.position != environment.layout.charger_position
    )
    delivery_clearance = bool(
        teammate.carrying_task_id is not None
        and teammate.navigation_goal_kind == "delivery"
        and 0
        < shortest_path_distance(
            teammate.position,
            teammate.navigation_goal_position,
            environment.config.map_layout_id,
        )
        <= 2
        and shortest_path_distance(
            human.position,
            teammate.navigation_goal_position,
            environment.config.map_layout_id,
        )
        <= 2
    )
    single_lane_clearance = bool(
        single_lane_egress_agent_id(
            state,
            environment.config,
            goal_positions={
                agent.agent_id: environment._frozen_route_goal(
                    state,
                    agent.agent_id,
                    prioritize_old_tasks=True,
                )
                or agent.navigation_goal_position
                for agent in state.agents
            },
        )
        is not None
    )
    ai_charger_distance = shortest_path_distance(
        teammate.position,
        environment.layout.charger_position,
        environment.config.map_layout_id,
    )
    ai_critical_charger_priority = bool(
        charger_service_required(environment, state, teammate)
        and (
            teammate.battery
            - ai_charger_distance * environment.config.move_battery_cost
        )
        <= (
            environment.config.mission_reserve_steps
            * environment.config.move_battery_cost
        )
    )
    shared_actions = (
        dict(coordinated_actions)
        if coordinated_actions is not None
        else stable_coordination_actions(environment)
        if (
            dual_charger_clearance
            or delivery_clearance
            or single_lane_clearance
            or ai_critical_charger_priority
        )
        else None
    )
    coordinated_human = (
        shared_actions[human.agent_id] if shared_actions is not None else "WAIT"
    )
    if ai_critical_charger_priority:
        # A participant can see the same public emergency right-of-way bit as
        # Robot 2. Respect the complete frozen-state coordination action so a
        # goal-directed surrogate cannot independently enter the critical
        # robot's only remaining charger step.
        directed = coordinated_human
    if dual_charger_clearance and shared_actions is not None:
        # Every supported participant can derive the same charger order from
        # frozen S_t.  Respect both halves of that public protocol: a
        # lower-priority participant may have to clear, or it may have to
        # WAIT while the priority AI advances.  Restricting this to movement
        # let a goal-directed participant move into the exact cell that the
        # priority robot had been trained to enter.
        directed = coordinated_human
    if single_lane_clearance and shared_actions is not None:
        # On a one-cell aisle arm, every profile respects the public egress
        # order.  Goal-directed participants still follow their own route
        # everywhere else; cautious and hesitant profiles retain their timing
        # variation below.
        directed = coordinated_human
    if (
        coordinated_human in MOVE_DELTAS
        and delivery_clearance
    ):
        delta = MOVE_DELTAS[coordinated_human]
        target = (
            human.position[0] + delta[0],
            human.position[1] + delta[1],
        )
        if shortest_path_distance(
            target,
            teammate.navigation_goal_position,
            environment.config.map_layout_id,
        ) > shortest_path_distance(
            human.position,
            teammate.navigation_goal_position,
            environment.config.map_layout_id,
        ):
            # A participant that can see it is occupying the AI carrier's
            # final approach can clear it without seeing the AI's private
            # current-frame action.  Cautious/hesitant timing remains applied
            # below, but their route intent no longer reverses immediately.
            directed = coordinated_human
    if name == "goal_directed":
        result = np.zeros(len(ACTIONS), dtype=np.float64)
        result[ACTIONS.index(directed)] = 1.0
        return result

    if name == "cautious":
        # A cautious participant yields only when its directed action has a
        # collision counterfactual against a legal teammate action in S_t.
        # Proximity alone is not a reason to freeze: the old blanket two-cell
        # rule made ordinary following and narrow dead ends unsolvable.
        teammate_mask = environment.action_masks()[teammate.agent_id]
        collision_counterfactual = any(
            allowed > 0.5
            and environment._resolve_motion(
                state,
                {
                    human.agent_id: directed,
                    teammate.agent_id: teammate_action,
                },
            )[3]
            for teammate_action, allowed in zip(ACTIONS, teammate_mask)
        )
        action = (
            "WAIT"
            if (
                directed != "WAIT"
                and collision_counterfactual
                and state.ineffective_joint_wait_streak == 0
            )
            else directed
        )
        result = np.zeros(len(ACTIONS), dtype=np.float64)
        result[ACTIONS.index(action)] = 1.0
        return result

    # Hesitation is structured human timing noise rather than an arbitrary
    # teleport or an action selected after observing Robot 2's current move.
    result = np.zeros(len(ACTIONS), dtype=np.float64)
    directed_probability = (
        0.95 if state.ineffective_joint_wait_streak > 0 else 0.75
    )
    result[ACTIONS.index(directed)] += directed_probability
    result[ACTIONS.index("WAIT")] += 1.0 - directed_probability
    return result


def robust_partner_robot_two_action(
    environment: WarehouseMultiAgentEnv,
    *,
    preferred_action: str | None = None,
    profile_probabilities: np.ndarray | None = None,
) -> str:
    """Return a zero-worst-profile-risk Robot 2 training label from S_t."""

    state = environment.get_state()
    agent = state.by_id("robot_2")
    if profile_probabilities is None:
        coordinated_actions = stable_coordination_actions(environment)
        profiles = np.stack(
            [
                participant_surrogate_distribution(
                    environment,
                    profile=name,
                    coordinated_actions=coordinated_actions,
                )
                for name in PARTNER_PROFILES
            ]
        )
    else:
        profiles = np.asarray(profile_probabilities, dtype=np.float64)
        if profiles.shape != (len(PARTNER_PROFILES), len(ACTIONS)):
            raise ValueError("profile_probabilities has an invalid shape.")
    collision = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.float64)
    for first_index, first_action in enumerate(ACTIONS):
        for second_index, second_action in enumerate(ACTIONS):
            collision[first_index, second_index] = float(
                environment._resolve_motion(
                    state,
                    {"robot_1": first_action, "robot_2": second_action},
                )[3]
            )
    goal = (
        environment.layout.charger_position
        if charger_service_required(environment, state, agent)
        else environment._frozen_route_goal(
            state,
            "robot_2",
            prioritize_old_tasks=True,
        )
    )
    candidates: list[tuple[float, int, int, int]] = []
    for index, (action, allowed) in enumerate(
        zip(ACTIONS, environment.action_masks()["robot_2"])
    ):
        if allowed <= 0.5:
            continue
        if action in MOVE_DELTAS and agent.battery <= environment.config.move_battery_cost:
            continue
        risk = float(
            np.max(
                np.einsum(
                    "pi,i->p",
                    profiles,
                    collision[:, index],
                )
            )
        )
        if action in MOVE_DELTAS:
            delta = MOVE_DELTAS[action]
            target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
            if agent.navigation_goal_kind == "charge":
                remaining_battery = (
                    agent.battery - environment.config.move_battery_cost
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
        else:
            target = agent.position
        distance = (
            shortest_path_distance(target, goal, environment.config.map_layout_id)
            if goal is not None
            else 0
        )
        candidates.append((risk, distance, int(action == "WAIT"), index))
    if not candidates:
        return "WAIT"
    minimum_risk = min(item[0] for item in candidates)
    safest = [item for item in candidates if item[0] <= minimum_risk + 1e-8]
    by_action = {ACTIONS[item[3]]: item for item in safest}
    teammate = state.by_id(environment.config.human_agent_id)
    recent_departure = agent.last_charger_departure_frame
    station_clearance_active = bool(
        charger_service_required(environment, state, teammate)
        and shortest_path_distance(
            teammate.position,
            environment.layout.charger_position,
            environment.config.map_layout_id,
        )
        <= 6
        and (
            agent.position == environment.layout.charger_position
            or (
                recent_departure is not None
                and 0 <= state.frame - recent_departure <= 6
            )
        )
    )
    public_clearance = str(preferred_action or "WAIT")
    if (
        station_clearance_active
        and public_clearance != "WAIT"
        and public_clearance in by_action
    ):
        # Once a public charger handoff starts, keep following its safe
        # clearance phase until the queued teammate reaches the station. A
        # fresh pickup tie-break must not send the former occupant straight
        # back onto the charger one frame after it departed.
        return public_clearance
    dual_charger_clearance = bool(
        charger_service_required(environment, state, agent)
        and charger_service_required(environment, state, teammate)
        and agent.position != environment.layout.charger_position
        and teammate.position != environment.layout.charger_position
    )
    coordinated_robot_two = (
        stable_coordination_actions(environment)["robot_2"]
        if dual_charger_clearance
        else "WAIT"
    )
    if (
        dual_charger_clearance
        and coordinated_robot_two != "WAIT"
        and coordinated_robot_two in by_action
    ):
        # A two-phase station handoff can require the yielding robot to move
        # one cell farther from the charger before the priority robot enters.
        # It is still the minimum-risk action from S_t and must not be rejected
        # by the ordinary one-step distance tie-break.
        return coordinated_robot_two
    if (
        agent.position == environment.layout.charger_position
        and charger_service_required(environment, state, agent)
        and "WAIT" in by_action
    ):
        # Productive charging normally dominates movement, but a public
        # lower-energy handoff is the one exception. The coordinated action
        # was derived from S_t and may be used only when it is also in the
        # minimum worst-profile-risk set. Earlier code returned WAIT here
        # before consulting the handoff, making the occupant charge through
        # several avoidable frames while its depleted peer was queued.
        handoff = str(preferred_action or "WAIT")
        if handoff != "WAIT" and handoff in by_action:
            return handoff
        coordinated_handoff = stable_coordination_actions(environment)["robot_2"]
        if (
            coordinated_handoff != "WAIT"
            and coordinated_handoff in by_action
        ):
            return coordinated_handoff
        return "WAIT"
    current_distance = (
        shortest_path_distance(
            agent.position,
            goal,
            environment.config.map_layout_id,
        )
        if goal is not None
        else 0
    )
    preferred = by_action.get(str(preferred_action))
    if preferred is not None and preferred[2] == 0 and preferred[1] <= current_distance:
        return str(preferred_action)
    coordinated = stable_coordination_actions(environment)["robot_2"]
    if (
        coordinated != "WAIT"
        and coordinated in by_action
        and by_action[coordinated][1] <= current_distance
    ):
        return coordinated
    moves = [item for item in safest if item[2] == 0]
    progress_moves = [item for item in moves if item[1] <= current_distance]
    if progress_moves:
        selected = min(progress_moves, key=lambda item: (item[1], item[3]))
        return ACTIONS[selected[3]]
    standoff_clearance_moves = [
        item
        for item in moves
        if necessary_participant_standoff_clearance(
            environment,
            state,
            agent,
            candidate_action=ACTIONS[item[3]],
        )
    ]
    if standoff_clearance_moves:
        def standoff_rank(
            item: tuple[float, int, int, int],
        ) -> tuple[int, int, int]:
            action = ACTIONS[item[3]]
            delta = MOVE_DELTAS[action]
            target = (
                agent.position[0] + delta[0],
                agent.position[1] + delta[1],
            )
            separation = shortest_path_distance(
                target,
                teammate.position,
                environment.config.map_layout_id,
            )
            return (-separation, item[1], item[3])

        selected = min(standoff_clearance_moves, key=standoff_rank)
        return ACTIONS[selected[3]]
    if (
        agent.navigation_goal_kind == "charge"
        and "WAIT" in by_action
        and not necessary_teammate_route_clearance(
            environment,
            state,
            agent,
        )
    ):
        # Do not spend scarce reserve retreating from a safe charger queue.
        # A retreat is justified only when the frozen teammate route proves
        # that this robot is the blocker, as in a cautious-partner stand-off.
        return "WAIT"
    if (
        agent.carrying_task_id is not None
        and "WAIT" in by_action
        and not is_necessary_urgent_charger_clearance(
            environment,
            state,
            agent,
        )
        and not necessary_teammate_route_clearance(
            environment,
            state,
            agent,
        )
    ):
        # A collision-safe hold is preferable to an avoidable loaded route
        # regression.  The participant will decide again from the next S_t;
        # only a proved route/charger clearance may justify backing away.
        return "WAIT"
    # Empty robots may need one safe retreat to break a cautious-partner
    # stand-off.  This is an offline label, never a runtime action rewrite.
    selected = min(moves or safest, key=lambda item: (item[1], item[3]))
    return ACTIONS[selected[3]]


def sampled_partner_profile(rng: np.random.Generator) -> str:
    return str(rng.choice(PARTNER_PROFILES))
