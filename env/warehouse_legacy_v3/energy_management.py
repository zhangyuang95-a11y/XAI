"""Pure energy-planning helpers shared by warehouse transition code."""

from __future__ import annotations

from typing import Any

from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance


def charger_route_slack(
    config: Any,
    *,
    position: tuple[int, int],
    battery: float,
    charger_position: tuple[int, int],
) -> float:
    """Energy left on arrival at the charger from one frozen state."""

    distance = shortest_path_distance(
        position,
        charger_position,
        config.map_layout_id,
    )
    return float(battery - distance * config.move_battery_cost)


def charger_route_is_critical(
    config: Any,
    *,
    position: tuple[int, int],
    battery: float,
    charger_position: tuple[int, int],
) -> bool:
    """Whether a charging route has consumed its mission safety reserve.

    The old implementation used three different thresholds in the priority,
    teacher, and reward paths.  In particular, a robot with exactly the
    configured mission reserve was treated as ordinary charging by the public
    priority feature but urgent by later training heuristics.  That mismatch
    produced the right-then-left charger detour.  This single predicate is
    intentionally state-only and is shared by every decision consumer.
    """

    return bool(
        charger_route_slack(
            config,
            position=position,
            battery=battery,
            charger_position=charger_position,
        )
        <= config.mission_reserve_steps * config.move_battery_cost
    )


def charger_departure_progress(
    state: Any,
    agent: Any,
) -> tuple[bool, bool]:
    """Return mission and coordination progress since a charger departure.

    A robot that temporarily clears the station for a depleted teammate must
    be allowed to return after that teammate has charged.  Counting only the
    clearing robot's pickups/deliveries misclassifies the causal two-phase
    handoff as a charger loop and can leave both robots waiting beside an
    empty station.  Every input here belongs to the current frozen state.
    """

    departure = agent.last_charger_departure_frame
    if departure is None:
        return False, False
    elapsed = max(0, int(state.frame) - int(departure))
    mission = bool(
        agent.deliveries_completed > agent.deliveries_at_last_charger_departure
        or (
            agent.carrying_task_id is not None
            and agent.carrying_task_id
            != agent.carrying_task_at_last_charger_departure
        )
    )
    coordination = bool(
        state.total_deliveries > agent.team_deliveries_at_last_charger_departure
        or any(
            teammate.agent_id != agent.agent_id
            and (
                teammate.charger_wait_streak > 0
                or 0 < teammate.steps_since_charging < elapsed
            )
            for teammate in state.agents
        )
    )
    return mission, coordination


def charger_handoff_clearance_action(
    environment: Any,
    state: Any,
    occupant: Any,
    waiter: Any,
) -> str | None:
    """Return a safe lateral charger handoff from one frozen state.

    A station occupant may still be below its own hysteretic release target
    while a teammate with even less energy has reached the charger apron.  In
    that case an unconditional joint WAIT starves the more urgent robot and
    teaches the Actor to hold the station.  This predicate is shared by
    offline supervision and dense training credit so the same ``S_t`` cannot
    be labelled both necessary and avoidable.

    The waiter remains stationary for this transition.  It can enter only on
    the following frozen state, after the occupant has actually cleared the
    charger; this preserves the conservative no-future-knowledge protocol.
    """

    charger = environment.layout.charger_position
    if (
        not occupant.active
        or not waiter.active
        or occupant.agent_id == waiter.agent_id
        or occupant.position != charger
        or waiter.position == charger
        or shortest_path_distance(
            waiter.position,
            charger,
            environment.config.map_layout_id,
        )
        != 1
        or not environment._requires_charge(state, waiter)
        or waiter.battery >= occupant.battery
        or (
            occupant.battery - waiter.battery
            < environment.config.charge_per_wait
        )
    ):
        return None

    # The clearing robot must retain enough energy for the lateral move and a
    # later return after its teammate has charged.  A more depleted occupant
    # therefore keeps the station even when both robots are energy-critical.
    remaining_energy = occupant.battery - environment.config.move_battery_cost
    if remaining_energy < environment.config.move_battery_cost:
        return None

    mission_goal = charger
    if occupant.carrying_task_id is not None:
        mission_goal = state.task_by_id(
            occupant.carrying_task_id
        ).delivery_position
    elif occupant.route_commitment_task_id is not None:
        committed = next(
            (
                task
                for task in state.tasks
                if task.task_id == occupant.route_commitment_task_id
                and task.status == "available"
            ),
            None,
        )
        if committed is not None:
            mission_goal = committed.pickup_position

    home_column = environment.layout.robot_start_positions[
        environment.agent_ids.index(occupant.agent_id)
    ][1]
    candidates: list[tuple[int, int, int, str]] = []
    for action_index, action in enumerate(ACTIONS):
        delta = MOVE_DELTAS.get(action)
        if delta is None or delta[0] != 0:
            continue
        target = (
            occupant.position[0] + delta[0],
            occupant.position[1] + delta[1],
        )
        if not environment.layout.is_passable(target) or target == waiter.position:
            continue
        joint_action = {
            agent.agent_id: (
                action if agent.agent_id == occupant.agent_id else "WAIT"
            )
            for agent in state.agents
        }
        _, _, invalid, collision, _, _ = environment._resolve_motion(
            state,
            joint_action,
        )
        if collision or occupant.agent_id in invalid:
            continue
        candidates.append(
            (
                shortest_path_distance(
                    target,
                    mission_goal,
                    environment.config.map_layout_id,
                ),
                abs(target[1] - home_column),
                action_index,
                action,
            )
        )
    return min(candidates)[3] if candidates else None


def charger_queue_clearance_delay(
    environment: Any,
    state: Any,
    positions: Any,
    active_agents: list[Any],
    clearance_cap: float,
) -> float:
    """Measure state-only charger blocking under the shared priority rule."""

    charger = environment.layout.charger_position
    occupant = next(
        (
            agent
            for agent in active_agents
            if positions.get(agent.agent_id, agent.position) == charger
        ),
        None,
    )
    if occupant is None:
        return 0.0
    queued = next(
        (
            agent
            for agent in active_agents
            if agent.agent_id != occupant.agent_id
        ),
        None,
    )
    if queued is None:
        return 0.0
    queued_position = positions.get(queued.agent_id, queued.position)
    blocked = bool(
        environment._requires_charge(state, queued, position=queued_position)
        and shortest_path_distance(
            queued_position,
            charger,
            environment.config.map_layout_id,
        )
        <= 2
        and (
            not environment._requires_charge(state, occupant, position=charger)
            or charger_handoff_clearance_action(
                environment,
                state,
                occupant,
                queued,
            )
            is not None
        )
    )
    if not blocked:
        return 0.0
    return (
        min(float(clearance_cap), 2.0)
        if environment.config.reward.individual_credit_enabled
        else float(clearance_cap)
    )


def charger_reentry_event(
    agent: Any,
    *,
    elapsed: int,
    completed_mission_progress: bool,
    completed_coordination_progress: bool = False,
) -> dict[str, Any] | None:
    """Classify a return within six steps using actual mission progress."""

    if elapsed > 6:
        return None
    if completed_mission_progress or completed_coordination_progress:
        event = "charger_productive_return"
    else:
        event = "charger_return_cycle"
    return {
        "event": event,
        "agent_id": agent.agent_id,
        "steps_since_departure": int(elapsed),
        "battery": float(agent.battery),
        "productive_reason": (
            # If the station was used by the teammate during the absence,
            # that observable handoff is the more specific causal reason for
            # the short return. Mission geometry may also have improved, but
            # must not hide the completed coordination event.
            "coordination"
            if completed_coordination_progress
            else "mission"
            if completed_mission_progress
            else None
        ),
    }


def charge_release_evidence(
    environment: Any,
    state: Any,
    agent: Any,
) -> dict[str, Any]:
    """Return the auditable components of the charger release threshold.

    Participant-facing explanations must not present a derived threshold as a
    magic constant.  Keep the controlling task, route legs, safety reserve,
    and anti-oscillation margin together so every consumer verbalizes the same
    calculation that controls charge mode.
    """

    if agent.carrying_task_id:
        tasks = [state.task_by_id(agent.carrying_task_id)]
    else:
        tasks = [task for task in state.tasks if task.status == "available"]
    candidates: list[dict[str, Any]] = []
    if tasks:
        # An empty robot's route commitment is not task ownership.  Its peer
        # can still reach that shared A point first, so release with enough
        # energy for every currently available alternative.  This prevents a
        # one-step departure followed by a return when the preferred A point
        # is claimed concurrently.
        for task in tasks:
            origin = environment.layout.charger_position
            pickup_steps = (
                0
                if agent.carrying_task_id
                else shortest_path_distance(
                    origin,
                    task.pickup_position,
                    environment.config.map_layout_id,
                )
            )
            delivery_steps = shortest_path_distance(
                origin if agent.carrying_task_id else task.pickup_position,
                task.delivery_position,
                environment.config.map_layout_id,
            )
            return_steps = shortest_path_distance(
                task.delivery_position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            route_steps = environment._mission_route_steps(
                state,
                agent,
                task,
                origin=origin,
            )
            candidates.append(
                {
                    "task_id": task.task_id,
                    "pickup_steps": float(pickup_steps),
                    "delivery_steps": float(delivery_steps),
                    "return_steps": float(return_steps),
                    "mission_reserve_steps": float(
                        environment.config.mission_reserve_steps
                    ),
                    "route_steps": float(route_steps),
                    "route_energy": float(
                        route_steps * environment.config.move_battery_cost
                    ),
                }
            )
        controlling = max(
            candidates,
            key=lambda item: (float(item["route_energy"]), str(item["task_id"])),
        )
        required = float(controlling["route_energy"])
    else:
        required = (
            environment.config.mission_reserve_steps
            * environment.config.move_battery_cost
        )
        controlling = {
            "task_id": None,
            "pickup_steps": 0.0,
            "delivery_steps": 0.0,
            "return_steps": 0.0,
            "mission_reserve_steps": float(
                environment.config.mission_reserve_steps
            ),
            "route_steps": float(environment.config.mission_reserve_steps),
            "route_energy": float(required),
        }
    hysteresis_steps = float(environment.config.charge_release_hysteresis_steps)
    hysteresis_energy = float(
        hysteresis_steps * environment.config.move_battery_cost
    )
    # In Human-AI play, an AI leaving the station on the exact ordinary
    # threshold can immediately meet a low-energy participant on the only
    # approach lane.  One public two-phase clearance then consumes the whole
    # ordinary coordination reserve and forces a return.  When that conflict
    # is already visible in S_t, retain exactly one additional charging wait
    # before departure.  This is state-derived energy planning—not an action
    # override—and is exposed verbatim in DecisionTrace/explanations.
    coordination_contention_energy = 0.0
    participant_id = state.participant_controlled_agent_id
    if (
        participant_id is not None
        and agent.agent_id != participant_id
        and agent.position == environment.layout.charger_position
    ):
        participant = state.by_id(participant_id)
        participant_distance = shortest_path_distance(
            participant.position,
            environment.layout.charger_position,
            environment.config.map_layout_id,
        )
        if (
            participant.position != environment.layout.charger_position
            and participant_distance
            <= int(environment.config.coordination_energy_reserve_steps) + 2
            and environment._requires_charge(state, participant)
        ):
            coordination_contention_energy = float(
                environment.config.charge_per_wait
            )
    coordination_contention_steps = float(
        coordination_contention_energy
        / environment.config.move_battery_cost
    )
    threshold = float(
        min(
            100.0,
            required
            + hysteresis_energy
            + coordination_contention_energy,
        )
    )
    return {
        **controlling,
        "candidate_tasks": tuple(candidates),
        "move_battery_cost": float(environment.config.move_battery_cost),
        "hysteresis_steps": hysteresis_steps,
        "hysteresis_energy": hysteresis_energy,
        "coordination_contention_steps": coordination_contention_steps,
        "coordination_contention_energy": coordination_contention_energy,
        "release_threshold": threshold,
    }


def charge_release_energy(environment: Any, state: Any, agent: Any) -> float:
    """Return the hysteretic energy threshold for leaving charge mode."""

    return float(
        charge_release_evidence(environment, state, agent)["release_threshold"]
    )


def should_continue_charge_mode(environment: Any, state: Any, agent: Any) -> bool:
    """Keep hysteresis only while the frozen state still needs the station."""

    if agent.position == environment.layout.charger_position:
        return bool(agent.battery < charge_release_energy(environment, state, agent))
    # A task can disappear while the robot clears a queue.  Cancelling a now
    # unnecessary return prevents it from re-blocking a more critical peer.
    return bool(environment._requires_charge(state, agent))


def charger_service_required(environment: Any, state: Any, agent: Any) -> bool:
    """Return the single frozen-state predicate for necessary charging.

    ``_requires_charge`` is the *entry* threshold: it becomes false as soon as
    one mission is barely feasible.  A robot already in charge mode instead
    owns the station until the higher hysteretic release threshold is met.
    Teacher labels, counterfactual credit, and policy metrics must use that
    same distinction or a productive charging WAIT is relabelled as an
    avoidable departure between those two thresholds.
    """

    if agent.charge_mode_active or agent.navigation_goal_kind == "charge":
        return should_continue_charge_mode(environment, state, agent)
    return bool(environment._requires_charge(state, agent))
