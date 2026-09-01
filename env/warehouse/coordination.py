"""Offline deterministic teacher used only to label MAPPO training data.

Nothing in this module is permitted to participate in environment, evaluation,
reference-rollout, or deployed action execution.
"""

from __future__ import annotations

from typing import Any, Mapping

from .coordination_priority import (
    coordination_priority,
    imminent_head_on_encounter,
    single_lane_egress_agent_id,
)
from .coordination_goals import (
    claim_safe_distance as _claim_safe_distance,
    stable_coordination_goal_overrides,
)
from .coordination_reservations import (
    short_horizon_charger_reservation_actions as _short_horizon_charger_reservation_actions,
    single_lane_egress_actions as _single_lane_egress_actions,
)
from .energy_management import (
    charger_departure_progress,
    charger_handoff_clearance_action,
    charger_route_is_critical,
    charger_service_required,
)
from .environment import ACTIONS, MOVE_DELTAS, WarehouseMultiAgentEnv, shortest_path_distance
from .teacher_efficiency import (
    teacher_efficiency_guard as _raw_teacher_efficiency_guard,
)
from .transition_audit import (
    action_is_robustly_safe,
    necessary_teammate_route_clearance,
)


_ARCHIVED_8X9_LAYOUT_ID = (
    "warehouse_staggered_aisles_8x9_v1_three_cell_exit"
)


def _teacher_efficiency_guard(
    environment: WarehouseMultiAgentEnv,
    actions: Mapping[str, str],
) -> dict[str, str]:
    """Audit production labels without rewriting archived fixture semantics.

    The 8x9 map is retained only for coordinate-specific historical invariant
    tests.  Its narrow-aisle protocol intentionally holds a follower for an
    extra public clearance phase, while the production 6x7 protocol allows
    separated routes to move in parallel.  Running the new counterfactual
    efficiency rewrite over the archived labels silently changed those
    historical contracts, so preserve them at this compatibility boundary.
    """

    frozen = {str(agent_id): str(action) for agent_id, action in actions.items()}
    if environment.config.map_layout_id == _ARCHIVED_8X9_LAYOUT_ID:
        return frozen
    return _raw_teacher_efficiency_guard(environment, frozen)


def _urgent_charge(
    environment: WarehouseMultiAgentEnv,
    agent: Any,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    override = (goal_overrides or {}).get(agent.agent_id)
    if override is not None and override != environment.layout.charger_position:
        return False
    if agent.navigation_goal_kind != "charge":
        return False
    return charger_route_is_critical(
        environment.config,
        position=agent.position,
        battery=agent.battery,
        charger_position=environment.layout.charger_position,
    )


def _priority_agent_and_basis(
    environment: WarehouseMultiAgentEnv,
    *,
    imminent_head_on: bool,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[Any, str]:
    """Return the robot with right of way and the observable reason.

    This is deliberately shared by action selection and explanation evidence.
    A verbalizer must never infer priority from a robot number or from which
    action happened to be changed after collision resolution.
    """

    state = environment.get_state()
    layout = environment.layout
    goals = {
        agent.agent_id: (goal_overrides or {}).get(
            agent.agent_id,
            agent.navigation_goal_position,
        )
        for agent in state.agents
    }
    goal_kinds = {
        agent.agent_id: (
            "charge"
            if goals[agent.agent_id] == layout.charger_position
            and charger_service_required(environment, state, agent)
            else "delivery"
            if agent.carrying_task_id is not None
            else "pickup"
            if goals[agent.agent_id] != agent.position
            else agent.navigation_goal_kind
        )
        for agent in state.agents
    }
    decision = coordination_priority(
        state,
        environment.config,
        goal_positions=goals,
        goal_kinds=goal_kinds,
        requires_charge={
            agent.agent_id: charger_service_required(environment, state, agent)
            for agent in state.agents
        },
        imminent_head_on=imminent_head_on,
    )
    return state.by_id(decision.agent_id), decision.basis


def _clear_head_on_encounter(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    state = environment.get_state()
    goals = {
        agent.agent_id: (goal_overrides or {}).get(
            agent.agent_id,
            agent.navigation_goal_position,
        )
        for agent in state.agents
    }
    return imminent_head_on_encounter(
        state,
        environment.config,
        goals,
    )


def _shortest_progress_positions(
    environment: WarehouseMultiAgentEnv,
    agent: Any,
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return passable neighbouring cells on a shortest path to ``goal``."""

    current_distance = shortest_path_distance(
        agent.position,
        goal,
        environment.config.map_layout_id,
    )
    return tuple(
        sorted(
            target
            for row_delta, column_delta in MOVE_DELTAS.values()
            if environment.layout.is_passable(
                target := (
                    agent.position[0] + row_delta,
                    agent.position[1] + column_delta,
                )
            )
            and shortest_path_distance(
                target,
                goal,
                environment.config.map_layout_id,
            )
            < current_distance
        )
    )


def is_necessary_urgent_charger_clearance(
    environment: WarehouseMultiAgentEnv,
    state: Any,
    clearing_agent: Any,
) -> bool:
    """Whether one detour is required to unblock a charger handoff.

    Besides a mathematically urgent route, the robot occupying the charger
    must clear once a teammate whose current safe-mission goal is charging has
    reached the apron entrance.  The latter may still have more than one
    charge-wait of slack, but holding the station would create a permanent
    queue and is therefore not an avoidable delivery detour.
    """

    return any(
        teammate.agent_id != clearing_agent.agent_id
        and (
            (
                _urgent_charge(environment, teammate)
                and clearing_agent.position
                in _shortest_progress_positions(
                    environment,
                    teammate,
                    teammate.navigation_goal_position,
                )
            )
            or (
                clearing_agent.position == environment.layout.charger_position
                and charger_handoff_clearance_action(
                    environment,
                    state,
                    clearing_agent,
                    teammate,
                )
                is not None
            )
        )
        for teammate in state.agents
    )


def _shortest_progress_from_position(
    environment: WarehouseMultiAgentEnv,
    position: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Shortest-path neighbours from a candidate future position."""

    current_distance = shortest_path_distance(
        position,
        goal,
        environment.config.map_layout_id,
    )
    return tuple(
        target
        for row_delta, column_delta in MOVE_DELTAS.values()
        if environment.layout.is_passable(
            target := (
                position[0] + row_delta,
                position[1] + column_delta,
            )
        )
        and shortest_path_distance(
            target,
            goal,
            environment.config.map_layout_id,
        )
        < current_distance
    )


def _has_side_clearance_from_position(
    environment: WarehouseMultiAgentEnv,
    position: tuple[int, int],
    teammate_position: tuple[int, int],
    available_pickups: set[tuple[int, int]],
) -> bool:
    """Whether ``position`` has an ordinary lateral escape from a shared line."""

    if position[0] == teammate_position[0]:
        line_axis = 0
    elif position[1] == teammate_position[1]:
        line_axis = 1
    else:
        return False
    return any(
        target[line_axis] != position[line_axis]
        and environment.layout.is_passable(target)
        and target != teammate_position
        and target not in available_pickups
        and target != environment.layout.charger_position
        for row_delta, column_delta in MOVE_DELTAS.values()
        if (
            target := (
                position[0] + row_delta,
                position[1] + column_delta,
            )
        )
    )


def _shared_route_bottleneck(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    """Whether both shortest routes currently require the same next cell."""

    state = environment.get_state()
    left, right = state.agents
    overrides = dict(goal_overrides or {})
    left_goal = overrides.get(left.agent_id, left.navigation_goal_position)
    right_goal = overrides.get(right.agent_id, right.navigation_goal_position)
    left_progress = set(_shortest_progress_positions(environment, left, left_goal))
    right_progress = set(_shortest_progress_positions(environment, right, right_goal))
    return bool(left_progress.intersection(right_progress))


def _priority_route_occupied(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    """Whether a mission-critical robot needs its teammate to vacate now."""

    state = environment.get_state()
    overrides = dict(goal_overrides or {})
    for mover in state.agents:
        if mover.carrying_task_id is None and mover.navigation_goal_kind != "charge":
            continue
        blocker = next(
            agent for agent in state.agents if agent.agent_id != mover.agent_id
        )
        mover_goal = overrides.get(mover.agent_id, mover.navigation_goal_position)
        if blocker.position not in _shortest_progress_positions(
            environment,
            mover,
            mover_goal,
        ):
            continue
        blocker_goal = overrides.get(
            blocker.agent_id,
            blocker.navigation_goal_position,
        )
        if any(
            target != mover.position
            for target in _shortest_progress_positions(
                environment,
                blocker,
                blocker_goal,
            )
        ):
            return True
    return False


def _reserved_side_clearance(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[str, tuple[int, int]] | None:
    """Keep a yielding robot out of a reserved corridor until passage ends.

    A head-on manoeuvre often moves the lower-priority robot one cell into a
    side branch.  In that next state the robots are no longer aligned, so a
    memoryless controller used to send the yielding robot straight back into
    the corridor.  One frame later it had to yield again, creating a
    battery-draining two-cell oscillation.  The reservation is reconstructed
    from geometry: if the yielding robot's next mission step re-enters a cell
    on the priority robot's imminent shortest route, it holds its current
    cleared position until the priority robot has passed that cell.
    """

    state = environment.get_state()
    overrides = dict(goal_overrides or {})
    left, right = state.agents
    if (
        left.position[0] == right.position[0]
        or left.position[1] == right.position[1]
    ):
        # A reservation represents a robot that has already stepped out of a
        # shared line.  Applying it while the robots are still aligned turned
        # an ordinary open perpendicular route into a forced WAIT (notably the
        # full-battery robot beside the charger at demonstration reset).
        return None
    priority_agent, _ = _priority_agent_and_basis(
        environment,
        imminent_head_on=False,
        goal_overrides=overrides,
    )
    yielding_agent = next(
        agent
        for agent in state.agents
        if agent.agent_id != priority_agent.agent_id
    )
    yielding_goal = overrides.get(
        yielding_agent.agent_id,
        yielding_agent.navigation_goal_position,
    )
    priority_goal = overrides.get(
        priority_agent.agent_id,
        priority_agent.navigation_goal_position,
    )
    priority_route_distance = shortest_path_distance(
        priority_agent.position,
        priority_goal,
        environment.config.map_layout_id,
    )
    # A robot already standing on the priority route has not cleared it yet;
    # collision/head-on handling must move it rather than reserve that cell.
    yielding_on_priority_route = bool(
        shortest_path_distance(
            priority_agent.position,
            yielding_agent.position,
            environment.config.map_layout_id,
        )
        + shortest_path_distance(
            yielding_agent.position,
            priority_goal,
            environment.config.map_layout_id,
        )
        == priority_route_distance
    )
    if yielding_on_priority_route:
        return None
    for reentry in _shortest_progress_positions(
        environment,
        yielding_agent,
        yielding_goal,
    ):
        distance_to_reentry = shortest_path_distance(
            priority_agent.position,
            reentry,
            environment.config.map_layout_id,
        )
        reentry_to_goal = shortest_path_distance(
            reentry,
            priority_goal,
            environment.config.map_layout_id,
        )
        if (
            distance_to_reentry <= 6
            and distance_to_reentry + reentry_to_goal
            == priority_route_distance
        ):
            return yielding_agent.agent_id, yielding_agent.position
    return None


def _ordinary_clearance_position(
    environment: WarehouseMultiAgentEnv,
    yielding_agent: Any,
    priority_agent: Any,
    available_pickups: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the nearest ordinary side aisle that clears the shared line.

    The map has no designated yield-bay cells.  During an actual head-on
    encounter, the lower-priority robot instead uses the closest regular
    passable cell outside the robots' current row or column.  Open pickup
    points are excluded so clearing the corridor cannot steal a shared task.
    """

    if yielding_agent.position[0] == priority_agent.position[0]:
        axis = 0
        line_coordinate = yielding_agent.position[0]
    elif yielding_agent.position[1] == priority_agent.position[1]:
        axis = 1
        line_coordinate = yielding_agent.position[1]
    else:
        return None
    occupied = {agent.position for agent in environment.get_state().agents}
    candidates = tuple(
        position
        for position in environment.layout.passable_positions
        if position[axis] != line_coordinate
        and position not in occupied
        and position not in available_pickups
        and position != environment.layout.charger_position
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda position: (
            shortest_path_distance(
                yielding_agent.position,
                position,
                environment.config.map_layout_id,
            ),
            shortest_path_distance(
                position,
                yielding_agent.navigation_goal_position,
                environment.config.map_layout_id,
            ),
            position,
        ),
    )


def _immediate_clearance_positions(
    environment: WarehouseMultiAgentEnv,
    agent: Any,
    teammate: Any,
    available_pickups: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return ordinary adjacent aisle cells that immediately leave the line."""

    if agent.position[0] == teammate.position[0]:
        line_axis = 0
    elif agent.position[1] == teammate.position[1]:
        line_axis = 1
    else:
        return ()
    candidates: list[tuple[int, int]] = []
    for row_delta, column_delta in MOVE_DELTAS.values():
        target = (
            agent.position[0] + row_delta,
            agent.position[1] + column_delta,
        )
        if (
            target[line_axis] != agent.position[line_axis]
            and environment.layout.is_passable(target)
            and target != teammate.position
            and target not in available_pickups
            and target != environment.layout.charger_position
        ):
            candidates.append(target)
    return tuple(sorted(candidates))


def _retreat_clearance_position(
    environment: WarehouseMultiAgentEnv,
    yielding_agent: Any,
    priority_agent: Any,
    available_pickups: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the end of a free branch behind the yielding robot.

    On the shelf-only map a lower-priority robot can already be inside a
    horizontal branch when a loaded robot needs to enter that branch.  There
    is no lateral cell to step into, so the correct manoeuvre is to keep
    retreating away from the loaded robot until the latter can pass.  Treating
    the priority robot's junction as the only escape caused the two robots to
    alternate in and out of the corridor instead.
    """

    if yielding_agent.position[0] == priority_agent.position[0]:
        axis = 1
    elif yielding_agent.position[1] == priority_agent.position[1]:
        axis = 0
    else:
        return None
    offset = (
        -1
        if yielding_agent.position[axis] < priority_agent.position[axis]
        else 1
    )
    occupied = {
        agent.position
        for agent in environment.get_state().agents
        if agent.agent_id != yielding_agent.agent_id
    }
    current = yielding_agent.position
    candidates: list[tuple[int, int]] = []
    while True:
        target = list(current)
        target[axis] += offset
        position = (target[0], target[1])
        if (
            not environment.layout.is_passable(position)
            or position in occupied
            or position in available_pickups
            or position == environment.layout.charger_position
        ):
            break
        candidates.append(position)
        current = position
    if candidates:
        return candidates[-1]
    # Reaching the branch end is itself a valid cleared state once at least
    # one free cell separates the robots.  Keep waiting there while the
    # priority robot enters, instead of immediately walking back into it.
    if abs(yielding_agent.position[axis] - priority_agent.position[axis]) >= 2:
        return yielding_agent.position
    return None


def stable_coordination_actions(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, str]:
    """Choose offline supervision labels for one shared-task joint state.

    The environment deliberately leaves available tasks unassigned, so its
    public navigation goal is ``wait`` for an empty robot. Offline callers
    that omit explicit goals must therefore freeze the optimal shared-task
    matching; otherwise relabeling would teach both robots to wait and erase
    the Actor's pickup skill. This helper never rewrites runtime actions.
    """

    state = environment.get_state()
    layout = environment.layout
    layout_id = environment.config.map_layout_id
    plan = state.active_coordination_plan
    if (
        plan is not None
        and (
            state.participant_controlled_agent_id is None
            or layout_id != _ARCHIVED_8X9_LAYOUT_ID
        )
        and isinstance(plan.get("joint_actions"), Mapping)
    ):
        return {
            agent_id: str(plan["joint_actions"].get(agent_id, "WAIT"))
            for agent_id in environment.agent_ids
        }
    if (
        plan is not None
        and (
            state.participant_controlled_agent_id is None
            or layout_id != _ARCHIVED_8X9_LAYOUT_ID
        )
        and str(plan.get("moving_agent_id", ""))
        in environment.agent_ids
        and str(plan.get("moving_action", "")) in ACTIONS
    ):
        # Offline labels, runtime observations, and DecisionTrace must consume
        # the same frozen multi-frame contract. Re-ranking from the changed
        # geometry after CLEAR_CELL is exactly what used to flip priority and
        # produce mutually contradictory "I yielded for you" explanations.
        return {
            agent_id: (
                str(plan["moving_action"])
                if agent_id == str(plan["moving_agent_id"])
                else "WAIT"
            )
            for agent_id in environment.agent_ids
        }
    if (
        plan is not None
        and layout_id == _ARCHIVED_8X9_LAYOUT_ID
        and state.participant_controlled_agent_id is not None
        and str(plan.get("priority_basis", ""))
        == "charger_clearance_commitment"
        and str(plan.get("moving_agent_id", ""))
        != state.participant_controlled_agent_id
        and tuple(plan.get("moving_target", ())) == layout.charger_position
    ):
        # Historical 8x9 hand-off: after the participant visibly departs the
        # station apron, the AI's already-public next step into the charger is
        # deterministic while the participant holds.  Consuming that frozen
        # plan prevents the generic scorer from inventing an unrelated
        # parallel participant move.
        moving_id = str(plan["moving_agent_id"])
        return {
            agent_id: (
                str(plan["moving_action"])
                if agent_id == moving_id
                else "WAIT"
            )
            for agent_id in environment.agent_ids
        }
    overrides = stable_coordination_goal_overrides(
        environment,
        goal_overrides=goal_overrides,
    )
    imminent_head_on = _clear_head_on_encounter(
        environment,
        goal_overrides=overrides,
    )
    charging_agents = tuple(
        agent
        for agent in state.agents
        if overrides.get(agent.agent_id) == layout.charger_position
        and charger_service_required(environment, state, agent)
    )
    charger_exit_agents = tuple(
        agent
        for agent in state.agents
        if agent.position == layout.charger_position
        and not charger_service_required(environment, state, agent)
        and overrides.get(agent.agent_id) not in {
            None,
            layout.charger_position,
        }
    )
    priority_agent, priority_basis = _priority_agent_and_basis(
        environment,
        imminent_head_on=imminent_head_on,
        goal_overrides=overrides,
    )
    # A future route overlap is not an immediate conflict and must not become
    # a fresh single-step reservation on every frame.  Production runtime
    # performs exhaustive atomic joint-action lookahead; the teacher below
    # only establishes right-of-way for concrete occupied/same-target/head-on
    # conflicts.
    # Finish the second half of an observed delivery-cell clearance before a
    # fresh single-lane tie-break can assign right-of-way back to the robot
    # that just vacated B.  Both Actors can derive this reservation from S_t
    # (positions plus previous executed actions); no current-frame action is
    # observed.  The exact joint resolver remains the final safety proof.
    for event in state.last_coordination_events:
        if str(event.get("event", "")) != "occupied_cell_clearance_wait":
            continue
        waiting_id = str(event.get("waiting_agent_id", ""))
        clearing_id = str(event.get("clearing_agent_id", ""))
        if waiting_id not in environment.agent_ids or clearing_id not in environment.agent_ids:
            continue
        waiting_agent = state.by_id(waiting_id)
        clearing_agent = state.by_id(clearing_id)
        occupied_position = tuple(event.get("occupied_position", ()))
        if (
            waiting_agent.carrying_task_id is None
            or waiting_agent.navigation_goal_kind != "delivery"
            or waiting_agent.navigation_goal_position != occupied_position
            or clearing_agent.position == occupied_position
        ):
            continue
        for action in ACTIONS:
            if action not in MOVE_DELTAS:
                continue
            delta = MOVE_DELTAS[action]
            if (
                waiting_agent.position[0] + delta[0],
                waiting_agent.position[1] + delta[1],
            ) != occupied_position:
                continue
            followthrough = {
                agent.agent_id: (
                    action if agent.agent_id == waiting_id else "WAIT"
                )
                for agent in state.agents
            }
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                followthrough,
            )
            if not invalid and not collision:
                return _teacher_efficiency_guard(environment, followthrough)
    single_lane_actions = _single_lane_egress_actions(
        environment,
        goal_overrides=overrides,
        priority_basis=priority_basis,
    )
    if single_lane_actions is not None:
        return _teacher_efficiency_guard(environment, single_lane_actions)
    # Two depleted robots approaching the single station need a causal,
    # multi-frame handshake.  The priority robot may advance only when its
    # next charger step is safe against *every* legal action of the peer from
    # S_t.  Otherwise the lower-priority robot moves one cell farther away
    # from the station while the priority robot visibly waits.  Repeating this
    # rule clears the conflicting side of the three-cell exit before entry
    # and avoids the old
    # approach/retreat ping-pong that consumed the priority robot's reserve.
    if (
        len(charging_agents) == 2
        and all(
            agent.position != layout.charger_position
            for agent in charging_agents
        )
    ):
        yielding_agent = next(
            agent
            for agent in charging_agents
            if agent.agent_id != priority_agent.agent_id
        )
        held = {agent.agent_id: "WAIT" for agent in state.agents}
        priority_distance = shortest_path_distance(
            priority_agent.position,
            layout.charger_position,
            layout_id,
        )
        priority_progress: list[tuple[int, str]] = []
        committed_priority_progress: list[tuple[int, str]] = []
        priority_mask = environment.action_masks()[priority_agent.agent_id]
        for action_index, (action, allowed) in enumerate(
            zip(ACTIONS, priority_mask)
        ):
            if allowed <= 0.5 or action not in MOVE_DELTAS:
                continue
            delta = MOVE_DELTAS[action]
            target = (
                priority_agent.position[0] + delta[0],
                priority_agent.position[1] + delta[1],
            )
            if shortest_path_distance(
                target,
                layout.charger_position,
                layout_id,
            ) >= priority_distance:
                continue
            trial = dict(held)
            trial[priority_agent.agent_id] = action
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if collision or priority_agent.agent_id in invalid:
                continue
            committed_priority_progress.append((action_index, action))
            if not action_is_robustly_safe(
                    environment,
                    state,
                    held,
                    priority_agent.agent_id,
                    action,
                ):
                continue
            priority_progress.append((action_index, action))
        if priority_progress:
            _, action = min(priority_progress)
            return _teacher_efficiency_guard(environment, {
                agent.agent_id: (
                    action if agent.agent_id == priority_agent.agent_id else "WAIT"
                )
                for agent in state.agents
            })

        yielding_distance = shortest_path_distance(
            yielding_agent.position,
            layout.charger_position,
            layout_id,
        )
        yielding_priority_distance = shortest_path_distance(
            yielding_agent.position,
            priority_agent.position,
            layout_id,
        )
        yielding_on_priority_charger_route = bool(
            yielding_agent.position[0] == priority_agent.position[0]
            and
            priority_distance
            == yielding_priority_distance + yielding_distance
        )
        priority_route_blocked_by_yielding = bool(
            yielding_agent.position
            in _shortest_progress_positions(
                environment,
                priority_agent,
                layout.charger_position,
            )
        )
        available_pickups = {
            task.pickup_position
            for task in state.tasks
            if task.status == "available"
        }
        clearance: list[tuple[int, int, int, str]] = []
        yielding_mask = environment.action_masks()[yielding_agent.agent_id]
        for action_index, (action, allowed) in enumerate(
            zip(ACTIONS, yielding_mask)
        ):
            if allowed <= 0.5 or action not in MOVE_DELTAS:
                continue
            delta = MOVE_DELTAS[action]
            target = (
                yielding_agent.position[0] + delta[0],
                yielding_agent.position[1] + delta[1],
            )
            target_charger_distance = shortest_path_distance(
                target,
                layout.charger_position,
                layout_id,
            )
            target_priority_distance = shortest_path_distance(
                target,
                priority_agent.position,
                layout_id,
            )
            # On a one-cell arm, the lower-priority robot sometimes has no
            # side exit and must travel ahead toward the shared apron before
            # it can step aside.  That is still genuine clearance when it
            # lies on the priority route and strictly increases separation.
            # Conversely, moving farther from the charger but *toward* the
            # priority robot is never clearance; it caused the pair to meet
            # in the middle and repeat a same-target collision forever.
            clears_along_single_lane = bool(
                yielding_on_priority_charger_route
                and target_charger_distance < yielding_distance
            )
            remaining = (
                yielding_agent.battery - environment.config.move_battery_cost
            )
            # When the yielding robot physically occupies the priority
            # robot's only charger-progress cell, refusing a side step in
            # order to preserve the ordinary reserve creates an unsatisfiable
            # WAIT/WAIT state.  Permit the clearance while retaining one move
            # of survival energy beyond the route.  In all other cases keep
            # the full study reserve.
            return_reserve_steps = (
                1.0
                if priority_route_blocked_by_yielding
                else environment.config.mission_reserve_steps
            )
            required_return = (
                target_charger_distance + return_reserve_steps
            ) * environment.config.move_battery_cost
            trial = dict(held)
            trial[yielding_agent.agent_id] = action
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if (
                (
                    yielding_on_priority_charger_route
                    and target_priority_distance <= yielding_priority_distance
                )
                or (
                    target_charger_distance <= yielding_distance
                    and not clears_along_single_lane
                )
                or target in available_pickups
                or remaining + 1e-8 < required_return
                or collision
                or yielding_agent.agent_id in invalid
            ):
                continue
            clearance.append(
                (
                    -target_charger_distance,
                    -target_priority_distance,
                    action_index,
                    action,
                )
            )
        yielding_delta = MOVE_DELTAS.get(yielding_agent.last_executed_action)
        yielding_just_vacated_charger = bool(
            yielding_delta is not None
            and yielding_delta[0] == 0
            and (
                yielding_agent.position[0] - yielding_delta[0],
                yielding_agent.position[1] - yielding_delta[1],
            )
            == layout.charger_position
        )
        if (
            clearance
            and committed_priority_progress
            and priority_basis == "charger_clearance_commitment"
            and yielding_just_vacated_charger
        ):
            # Complete an observed two-phase handoff. The yielding robot's
            # previous horizontal departure is already part of S_t, so both
            # agents can derive this reservation without seeing either
            # current action. Entering the now-empty station while the former
            # occupant continues away prevents the occupant's charge mode
            # from pulling it back into an otherwise artificial return loop.
            for _, _, _, clearance_action in sorted(clearance):
                for _, progress_action in sorted(committed_priority_progress):
                    followthrough = {
                        yielding_agent.agent_id: clearance_action,
                        priority_agent.agent_id: progress_action,
                    }
                    _, _, invalid, collision, _, _ = environment._resolve_motion(
                        state,
                        followthrough,
                    )
                    if not invalid and not collision:
                        return _teacher_efficiency_guard(
                            environment,
                            followthrough,
                        )
        if clearance:
            _, _, _, action = min(clearance)
            return _teacher_efficiency_guard(environment, {
                agent.agent_id: (
                    action if agent.agent_id == yielding_agent.agent_id else "WAIT"
                )
                for agent in state.agents
            })
        if committed_priority_progress:
            # In a one-cell approach there are states where the yielding
            # robot cannot spend another clearance move without losing its
            # own charger reserve.  Both robots can nevertheless derive the
            # same public order from S_t: the priority robot advances and the
            # yielding robot waits.  This is a simultaneous state-based
            # commitment, not observation of the peer's current action.
            _, action = min(committed_priority_progress)
            return _teacher_efficiency_guard(environment, {
                agent.agent_id: (
                    action
                    if agent.agent_id == priority_agent.agent_id
                    else "WAIT"
                )
                for agent in state.agents
            })
        return _teacher_efficiency_guard(environment, held)
    # A delivery point or its final approach can be occupied by the other
    # loaded robot at a T-junction.  Resolve that conflict with the same
    # causal two-phase protocol used at the charger: first move the
    # lower-priority robot far enough away that the delivery step is robust to
    # every legal peer action, then allow the carrier to advance on the next
    # frozen state.  Remembering the prior clearance direction in S_t prevents
    # the teacher/participant surrogate from alternating left-right forever.
    priority_goal = overrides.get(
        priority_agent.agent_id,
        priority_agent.navigation_goal_position,
    )
    yielding_agent = next(
        agent for agent in state.agents if agent.agent_id != priority_agent.agent_id
    )
    priority_goal_distance = shortest_path_distance(
        priority_agent.position,
        priority_goal,
        layout_id,
    )
    yielding_goal_distance = shortest_path_distance(
        yielding_agent.position,
        priority_goal,
        layout_id,
    )
    if (
        priority_agent.carrying_task_id is not None
        and priority_agent.navigation_goal_kind == "delivery"
        and 0 < priority_goal_distance <= 2
        and yielding_goal_distance <= 2
    ):
        public_delivery_clearance_committed = any(
            str(event.get("event", ""))
            == "occupied_cell_clearance_wait"
            and str(event.get("waiting_agent_id", ""))
            == priority_agent.agent_id
            and str(event.get("clearing_agent_id", ""))
            == yielding_agent.agent_id
            for event in state.last_coordination_events
        )
        held = {agent.agent_id: "WAIT" for agent in state.agents}
        progress: list[tuple[int, str]] = []
        priority_mask = environment.action_masks()[priority_agent.agent_id]
        for action_index, (action, allowed) in enumerate(
            zip(ACTIONS, priority_mask)
        ):
            if allowed <= 0.5 or action not in MOVE_DELTAS:
                continue
            delta = MOVE_DELTAS[action]
            target = (
                priority_agent.position[0] + delta[0],
                priority_agent.position[1] + delta[1],
            )
            trial = dict(held)
            trial[priority_agent.agent_id] = action
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            participant_priority = bool(
                state.participant_controlled_agent_id
                == priority_agent.agent_id
            )
            if (
                shortest_path_distance(target, priority_goal, layout_id)
                >= priority_goal_distance
                or collision
                or priority_agent.agent_id in invalid
                or (
                    not participant_priority
                    and not public_delivery_clearance_committed
                    and not action_is_robustly_safe(
                        environment,
                        state,
                        held,
                        priority_agent.agent_id,
                        action,
                    )
                )
            ):
                continue
            progress.append((action_index, action))
        if progress:
            _, action = min(progress)
            return _teacher_efficiency_guard(environment, {
                agent.agent_id: (
                    action if agent.agent_id == priority_agent.agent_id else "WAIT"
                )
                for agent in state.agents
            })

        available_pickups = {
            task.pickup_position
            for task in state.tasks
            if task.status == "available"
        }
        priority_progress_positions = set(
            _shortest_progress_positions(
                environment,
                priority_agent,
                priority_goal,
            )
        )
        clearance: list[tuple[int, int, int, int, str]] = []
        yielding_mask = environment.action_masks()[yielding_agent.agent_id]
        for action_index, (action, allowed) in enumerate(
            zip(ACTIONS, yielding_mask)
        ):
            if allowed <= 0.5 or action not in MOVE_DELTAS:
                continue
            delta = MOVE_DELTAS[action]
            target = (
                yielding_agent.position[0] + delta[0],
                yielding_agent.position[1] + delta[1],
            )
            target_clearance = shortest_path_distance(
                target,
                priority_goal,
                layout_id,
            )
            trial = dict(held)
            trial[yielding_agent.agent_id] = action
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if (
                target_clearance <= yielding_goal_distance
                or target in priority_progress_positions
                or (
                    target in available_pickups
                    and yielding_agent.carrying_task_id is not None
                )
                or yielding_agent.battery <= environment.config.move_battery_cost
                or collision
                or yielding_agent.agent_id in invalid
            ):
                continue
            own_goal = overrides.get(
                yielding_agent.agent_id,
                yielding_agent.navigation_goal_position,
            )
            clearance.append(
                (
                    -int(action == yielding_agent.last_executed_action),
                    -target_clearance,
                    shortest_path_distance(target, own_goal, layout_id),
                    action_index,
                    action,
                )
            )
        if clearance:
            _, _, _, _, action = min(clearance)
            return _teacher_efficiency_guard(environment, {
                agent.agent_id: (
                    action if agent.agent_id == yielding_agent.agent_id else "WAIT"
                )
                for agent in state.agents
            })
        return _teacher_efficiency_guard(environment, held)
    delivery_commitment_agent = (
        priority_agent
        if priority_agent.carrying_task_id is not None
        and priority_agent.navigation_goal_kind == "delivery"
        else None
    )
    # Atomic charger handoff: an energy-safe robot on the station must never
    # wait behind its own full battery while an energy-critical teammate is
    # parked in the single approach cell.  Generic mission scoring can prefer
    # joint WAIT when an open pickup is an articulation point, recreating the
    # exact full-robot/low-robot deadlock that the two-cell apron was added to
    # remove.  Move the departing robot into the better side cell now; on the
    # following frame it can go up through the apron while the teammate enters
    # the charger.  Endpoint exclusions guarantee that neither side cell can
    # accidentally claim a package.
    approach_position = (
        layout.charger_position[0] - 1,
        layout.charger_position[1],
    )
    approaching_charger = tuple(
        agent
        for agent in charging_agents
        if agent.position == approach_position
    )
    adjacent_charger_waiters = tuple(
        agent
        for agent in charging_agents
        if agent.position != layout.charger_position
        and shortest_path_distance(
            agent.position,
            layout.charger_position,
            layout_id,
        )
        == 1
    )
    charging_occupants = tuple(
        agent
        for agent in state.agents
        if agent.position == layout.charger_position
        and charger_service_required(environment, state, agent)
    )
    station_occupants = tuple(
        agent
        for agent in state.agents
        if agent.position == layout.charger_position
    )
    if station_occupants and adjacent_charger_waiters:
        occupant = station_occupants[0]
        waiter = adjacent_charger_waiters[0]
        handoff_action = charger_handoff_clearance_action(
            environment,
            state,
            occupant,
            waiter,
        )
        if handoff_action is not None:
            return _teacher_efficiency_guard(
                environment,
                {
                    occupant.agent_id: handoff_action,
                    waiter.agent_id: "WAIT",
                },
            )
    if charging_occupants and adjacent_charger_waiters:
        # A robot that has not yet reached a safe departure state owns the
        # single charger unless the shared frozen-state handoff predicate has
        # already proved that its lower-energy teammate should receive it.
        return _teacher_efficiency_guard(
            environment,
            {
                charging_occupants[0].agent_id: "WAIT",
                adjacent_charger_waiters[0].agent_id: "WAIT",
            },
        )
    idle_occupants = tuple(
        agent
        for agent in state.agents
        if agent.position == layout.charger_position
        and agent.battery < 100.0
        and overrides.get(agent.agent_id) in {None, layout.charger_position}
    )
    if idle_occupants and adjacent_charger_waiters:
        # A nominally energy-safe occupant can still lack a feasible teacher
        # assignment (for example while the other A point blocks its aisle).
        # Charge until an actionable commitment exists before handing over;
        # otherwise it steps aside and immediately returns after the queue.
        return _teacher_efficiency_guard(
            environment,
            {
                idle_occupants[0].agent_id: "WAIT",
                adjacent_charger_waiters[0].agent_id: "WAIT",
            },
        )
    safe_occupants = tuple(
        agent
        for agent in charger_exit_agents
    )
    if safe_occupants:
        occupant = safe_occupants[0]
        prospective_delivery_inbound = tuple(
            agent
            for agent in state.agents
            if agent.agent_id != occupant.agent_id
            and agent.carrying_task_id is not None
            and agent.navigation_goal_kind == "delivery"
            and agent.battery <= 30.0
            and shortest_path_distance(
                agent.position,
                agent.navigation_goal_position,
                layout_id,
            )
            <= 1
            and shortest_path_distance(
                agent.navigation_goal_position,
                layout.charger_position,
                layout_id,
            )
            <= 6
        )
        if prospective_delivery_inbound:
            inbound_delivery = prospective_delivery_inbound[0]
            progress_positions = _shortest_progress_positions(
                environment,
                inbound_delivery,
                inbound_delivery.navigation_goal_position,
            )
            progress_action = next(
                (
                    action
                    for action in ACTIONS
                    if action in MOVE_DELTAS
                    and (
                        inbound_delivery.position[0] + MOVE_DELTAS[action][0],
                        inbound_delivery.position[1] + MOVE_DELTAS[action][1],
                    )
                    in progress_positions
                ),
                "WAIT",
            )
            return _teacher_efficiency_guard(
                environment,
                {
                    occupant.agent_id: "WAIT",
                    inbound_delivery.agent_id: progress_action,
                },
            )
        inbound = tuple(
            agent
            for agent in charging_agents
            if agent.agent_id != occupant.agent_id
            and shortest_path_distance(
                agent.position,
                layout.charger_position,
                layout_id,
            )
            <= 6
        )
        if inbound:
            # Do not depart up the centre line just before an energy-critical
            # teammate arrives.  That creates a head-on encounter, forces the
            # departing robot to consume reserve in a side step, and can make
            # it return to the charger within six frames.  Clear directly into
            # a side apron only after the remaining mission is still safe from
            # that side cell; otherwise one energy-neutral charge wait is the
            # correct supervision label.
            task = None
            if occupant.carrying_task_id is not None:
                task = state.task_by_id(occupant.carrying_task_id)
            else:
                occupant_goal = overrides.get(occupant.agent_id)
                task = next(
                    (
                        item
                        for item in state.tasks
                        if item.status == "available"
                        and item.pickup_position == occupant_goal
                    ),
                    None,
                )
            clearance_candidates: list[tuple[float, int, str]] = []
            if task is not None:
                for action_index, action in enumerate(ACTIONS):
                    if action not in MOVE_DELTAS or MOVE_DELTAS[action][0] != 0:
                        continue
                    delta = MOVE_DELTAS[action]
                    target = (
                        occupant.position[0] + delta[0],
                        occupant.position[1] + delta[1],
                    )
                    required = (
                        environment._mission_route_steps(
                            state,
                            occupant,
                            task,
                            origin=target,
                        )
                        * environment.config.move_battery_cost
                    )
                    remaining = (
                        occupant.battery - environment.config.move_battery_cost
                    )
                    if (
                        layout.is_passable(target)
                        and target != inbound[0].position
                        and remaining >= required
                    ):
                        clearance_candidates.append((required, action_index, action))
            if clearance_candidates:
                _, _, clearance_action = min(clearance_candidates)
                return _teacher_efficiency_guard(
                    environment,
                    {
                        occupant.agent_id: clearance_action,
                        inbound[0].agent_id: "WAIT",
                    },
                )
            return _teacher_efficiency_guard(
                environment,
                {
                    occupant.agent_id: "WAIT",
                    inbound[0].agent_id: "WAIT",
                },
            )
    if approaching_charger:
        charging_agent = approaching_charger[0]
        side_agent = next(
            (
                agent
                for agent in state.agents
                if agent.agent_id != charging_agent.agent_id
                and agent.position[0] == layout.charger_position[0]
                and abs(
                    agent.position[1] - layout.charger_position[1]
                ) == 1
            ),
            None,
        )
        if side_agent is not None:
            simultaneous_handoff = {
                charging_agent.agent_id: "DOWN",
                side_agent.agent_id: "UP",
            }
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                simultaneous_handoff,
            )
            if not collision and not invalid:
                return _teacher_efficiency_guard(
                    environment,
                    simultaneous_handoff,
                )
    if (
        charger_exit_agents
        and approaching_charger
    ):
        exiting_agent = charger_exit_agents[0]
        teammate = next(
            agent
            for agent in state.agents
            if agent.agent_id != exiting_agent.agent_id
        )
        handoff_candidates: list[
            tuple[int, int, dict[str, str]]
        ] = []
        for action_index, action in enumerate(ACTIONS):
            if action not in MOVE_DELTAS or MOVE_DELTAS[action][0] != 0:
                continue
            actions = {
                agent.agent_id: "WAIT"
                for agent in state.agents
            }
            actions[exiting_agent.agent_id] = action
            actions[teammate.agent_id] = "WAIT"
            targets, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                actions,
            )
            if collision or exiting_agent.agent_id in invalid:
                continue
            handoff_candidates.append(
                (
                    shortest_path_distance(
                        targets[exiting_agent.agent_id],
                        overrides.get(
                            exiting_agent.agent_id,
                            exiting_agent.navigation_goal_position,
                        ),
                        layout_id,
                    ),
                    action_index,
                    actions,
                )
            )
        if handoff_candidates:
            return _teacher_efficiency_guard(
                environment,
                min(handoff_candidates, key=lambda item: item[:2])[2],
            )

    available_pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    clear_corridor = imminent_head_on
    side_reservation = _reserved_side_clearance(
        environment,
        goal_overrides=overrides,
    )
    lower_priority_agent = next(
        agent for agent in state.agents if agent.agent_id != priority_agent.agent_id
    )
    clearance_agent = lower_priority_agent
    clearance_position: tuple[int, int] | None = None
    if side_reservation is not None:
        clearance_agent = state.by_id(side_reservation[0])
        clearance_position = side_reservation[1]
        clear_corridor = True
    elif clear_corridor:
        # First keep the lower-priority robot responsible for clearing the
        # route.  It may step sideways at a junction or retreat to the end of
        # the branch it already occupies.  Divert the priority robot only when
        # neither option exists.
        immediate = _immediate_clearance_positions(
            environment,
            lower_priority_agent,
            priority_agent,
            available_pickups,
        )
        if immediate:
            clearance_position = min(
                immediate,
                key=lambda position: (
                    shortest_path_distance(
                        position,
                        lower_priority_agent.navigation_goal_position,
                        layout_id,
                    ),
                    position,
                ),
            )
        else:
            clearance_position = _retreat_clearance_position(
                environment,
                lower_priority_agent,
                priority_agent,
                available_pickups,
            )
        if clearance_position is None:
            priority_clearance = _immediate_clearance_positions(
                environment,
                priority_agent,
                lower_priority_agent,
                available_pickups,
            )
            if priority_clearance:
                clearance_agent = priority_agent
                clearance_position = min(
                    priority_clearance,
                    key=lambda position: (
                        shortest_path_distance(
                            position,
                            priority_agent.navigation_goal_position,
                            layout_id,
                        ),
                        position,
                    ),
                )
        if clearance_position is None:
            clearance_position = _ordinary_clearance_position(
                environment,
                lower_priority_agent,
                priority_agent,
                available_pickups,
            )
            clearance_agent = lower_priority_agent
    legal_by_agent: dict[str, list[str]] = {}
    for agent in state.agents:
        agent_id = agent.agent_id
        # Per-agent masks conservatively mark the teammate's current cell as
        # blocked. Joint offline labels may enter that cell only when the
        # teammate leaves simultaneously. The exact joint resolver rejects
        # same-cell, occupied, and swap cases.
        legal_by_agent[agent_id] = [
            action
            for action in ACTIONS
            if action == "WAIT"
            or (
                action in MOVE_DELTAS
                and environment.layout.is_passable(
                    (
                        agent.position[0] + MOVE_DELTAS[action][0],
                        agent.position[1] + MOVE_DELTAS[action][1],
                    )
                )
            )
        ]
    left_id, right_id = environment.agent_ids
    candidates: list[tuple[tuple[float, int, int], dict[str, str]]] = []
    for left_index, left_action in enumerate(legal_by_agent[left_id]):
        for right_index, right_action in enumerate(legal_by_agent[right_id]):
            actions = {left_id: left_action, right_id: right_action}
            targets, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                actions,
            )
            if collision or invalid:
                continue
            charging_ids = {
                agent.agent_id
                for agent in state.agents
                if overrides.get(agent.agent_id) == layout.charger_position
                and environment._requires_charge(state, agent)
            }
            if layout_id != _ARCHIVED_8X9_LAYOUT_ID and charging_ids and any(
                agent.agent_id not in charging_ids
                and agent.position != layout.charger_position
                and targets[agent.agent_id] == layout.charger_position
                for agent in state.agents
            ):
                # Keep the station cell free for a robot whose frozen goal is
                # charging.  This fixes the outer-apron case where the idle
                # peer moved into the charger while the critical robot tried
                # to approach it.
                continue
            if any(
                actions[agent.agent_id] in MOVE_DELTAS
                and agent.battery <= environment.config.move_battery_cost
                for agent in state.agents
            ):
                # Never turn a survivable wait into a move that immediately
                # reaches zero battery.
                continue
            if any(
                _urgent_charge(
                    environment,
                    agent,
                    goal_overrides=overrides,
                )
                and shortest_path_distance(
                    targets[agent.agent_id],
                    layout.charger_position,
                    layout_id,
                )
                > shortest_path_distance(
                    agent.position,
                    layout.charger_position,
                    layout_id,
                )
                and (
                    agent.battery
                    - (
                        environment.config.move_battery_cost
                        if actions[agent.agent_id] in MOVE_DELTAS
                        else 0.0
                    )
                    < (
                        shortest_path_distance(
                            targets[agent.agent_id],
                            layout.charger_position,
                            layout_id,
                        )
                        + environment.config.mission_reserve_steps
                    )
                    * environment.config.move_battery_cost
                )
                for agent in state.agents
            ):
                # Reject only a clearance move that would violate the exact
                # charger route plus reserve.  Treating every one-cell detour
                # by an "urgent" robot as forbidden forced loaded teammates to
                # reverse and re-enter the charger.  A side step that preserves
                # the safety budget is valid coordination supervision.
                continue
            score = 0.0
            for agent in state.agents:
                action = actions[agent.agent_id]
                goal = overrides.get(
                    agent.agent_id,
                    agent.navigation_goal_position,
                )
                if (
                    agent.navigation_goal_kind == "charge"
                    and overrides.get(agent.agent_id, layout.charger_position)
                    == layout.charger_position
                ):
                    charger_distance = _claim_safe_distance(
                        environment,
                        agent,
                        targets[agent.agent_id],
                        layout.charger_position,
                        available_pickups,
                    )
                    charge_weight = (
                        1_000.0
                        if agent.agent_id == priority_agent.agent_id
                        else 100.0
                    )
                    # Once charging is required, charger progress dominates
                    # ordinary delivery geometry. Weighting
                    # both robots, with the stable priority robot first,
                    # prevents the alternating up/down pattern that can drain
                    # a queue before either robot reaches the station.
                    score += charge_weight * charger_distance
                    if action == "WAIT" and agent.position != goal:
                        score += charge_weight * 0.25
                if (
                    agent.carrying_task_id is None
                    and targets[agent.agent_id] in available_pickups
                    and targets[agent.agent_id] != goal
                ):
                    # Entering an available A point claims it immediately.  A
                    # large deterministic cost makes an AI wait or take an
                    # alternative aisle until the robot assigned to that job
                    # has claimed it, instead of silently stealing the job.
                    score += 1_000.0
                at_goal = agent.position == goal
                if at_goal:
                    score += 0.0 if action == "WAIT" else 4.0
                else:
                    mission_distance = _claim_safe_distance(
                        environment,
                        agent,
                        targets[agent.agent_id],
                        goal,
                        available_pickups,
                    )
                    if mission_distance >= 10_000:
                        # Another open pickup is an articulation point on the
                        # route.  Hold position while the newly matched robot
                        # claims that blocking job; on the next state the cell
                        # becomes traversable without stealing it.
                        score += 0.0 if action == "WAIT" else 4.0
                    else:
                        mission_weight = (
                            1_000.0
                            if delivery_commitment_agent is not None
                            and agent.agent_id
                            == delivery_commitment_agent.agent_id
                            else (
                                2.0
                                if agent.agent_id == priority_agent.agent_id
                                else 1.0
                            )
                        )
                        score += mission_weight * mission_distance
                    if action == "WAIT" and mission_distance < 10_000:
                        score += 0.25 * (
                            1_000.0
                            if delivery_commitment_agent is not None
                            and agent.agent_id
                            == delivery_commitment_agent.agent_id
                            else (
                                2.0
                                if agent.agent_id == priority_agent.agent_id
                                else 1.0
                            )
                        )
            if all(action == "WAIT" for action in actions.values()) and not any(
                agent.position == layout.charger_position
                and agent.battery < 100.0
                for agent in state.agents
            ):
                # A joint stationary state without charging cannot make any
                # delivery progress.  Prefer a collision-free escape move so
                # the deterministic guard cannot create its own wait deadlock.
                score += 5_000.0
            if clear_corridor and clearance_position is not None:
                # A large but finite geometric term sends only the
                # lower-priority robot toward an ordinary side aisle while
                # the priority robot continues on its delivery line.
                passing_agent = next(
                    agent
                    for agent in state.agents
                    if agent.agent_id != clearance_agent.agent_id
                )
                charge_before_clearing = bool(
                    clearance_agent.position == layout.charger_position
                    and actions[clearance_agent.agent_id] == "WAIT"
                    and clearance_agent.battery < 100.0
                    and targets[passing_agent.agent_id]
                    != layout.charger_position
                )
                # When the yielding robot is already on the charger and the
                # priority robot is still one cell away, use this otherwise
                # idle step to charge before clearing.  An immediate side-step
                # can consume the last delivery reserve and incorrectly flip
                # a loaded robot onto a longer charge-first route.
                if not charge_before_clearing:
                    # Clearance is a hard temporal reservation, not a soft
                    # preference.  A 1,000x delivery-progress term otherwise
                    # pulled the yielding robot straight back into the lane on
                    # the very next frame and recreated the oscillation.
                    score += 10_000.0 * shortest_path_distance(
                        targets[clearance_agent.agent_id],
                        clearance_position,
                        layout_id,
                    )
                passing_goal = overrides.get(
                    passing_agent.agent_id,
                    passing_agent.navigation_goal_position,
                )
                score += 10.0 * shortest_path_distance(
                    targets[passing_agent.agent_id],
                    passing_goal,
                    layout_id,
                )
                if actions[passing_agent.agent_id] == "WAIT":
                    score += 5.0
            if (
                _urgent_charge(
                    environment,
                    priority_agent,
                    goal_overrides=overrides,
                )
                and priority_agent.position != layout.charger_position
            ):
                current_charger_distance = _claim_safe_distance(
                    environment,
                    priority_agent,
                    priority_agent.position,
                    layout.charger_position,
                    available_pickups,
                )
                next_charger_distance = _claim_safe_distance(
                    environment,
                    priority_agent,
                    targets[priority_agent.agent_id],
                    layout.charger_position,
                    available_pickups,
                )
                # A single unnecessary detour can consume the last two units
                # of reserve.  Make progress by the energy-critical robot the
                # dominant tie-breaker while still rejecting collisions.
                score += 20.0 * next_charger_distance
                if next_charger_distance >= current_charger_distance:
                    score += 20.0
            charging_before_clearance = any(
                agent.position == layout.charger_position
                and actions[agent.agent_id] == "WAIT"
                and agent.battery < 100.0
                for agent in state.agents
            )
            if (
                not charging_before_clearance
                and any(
                    _urgent_charge(
                        environment,
                        agent,
                        goal_overrides=overrides,
                    )
                    for agent in state.agents
                )
            ):
                next_progress = {
                    agent.agent_id: set(
                        _shortest_progress_from_position(
                            environment,
                            targets[agent.agent_id],
                            overrides.get(
                                agent.agent_id,
                                agent.navigation_goal_position,
                            ),
                        )
                    )
                    for agent in state.agents
                }
                next_route_conflict = bool(
                    next_progress[left_id].intersection(next_progress[right_id])
                    or (
                        targets[right_id] in next_progress[left_id]
                        and targets[left_id] in next_progress[right_id]
                    )
                )
                can_clear_next = any(
                    agent.navigation_goal_kind != "charge"
                    and _has_side_clearance_from_position(
                        environment,
                        targets[agent.agent_id],
                        targets[
                            next(
                                other.agent_id
                                for other in state.agents
                                if other.agent_id != agent.agent_id
                            )
                        ],
                        available_pickups,
                    )
                    for agent in state.agents
                )
                urgent_progressing = any(
                    _urgent_charge(
                        environment,
                        agent,
                        goal_overrides=overrides,
                    )
                    and shortest_path_distance(
                        targets[agent.agent_id],
                        layout.charger_position,
                        layout_id,
                    )
                    < shortest_path_distance(
                        agent.position,
                        layout.charger_position,
                        layout_id,
                    )
                    for agent in state.agents
                )
                if (
                    next_route_conflict
                    and not can_clear_next
                    and not urgent_progressing
                ):
                    # Avoid entering a state whose deterministic shortest
                    # moves collide on the following frame.  Clearing one
                    # step earlier replaces an advance/reverse pair with one
                    # purposeful side move or an energy-neutral wait.  Never
                    # postpone an urgent charger's only safe forward move;
                    # the teammate can clear on the following frame.
                    score += 10_000.0
            for urgent_agent in state.agents:
                if not _urgent_charge(
                    environment,
                    urgent_agent,
                    goal_overrides=overrides,
                ):
                    continue
                current_urgent_distance = _claim_safe_distance(
                    environment,
                    urgent_agent,
                    urgent_agent.position,
                    layout.charger_position,
                    available_pickups,
                )
                next_urgent_distance = _claim_safe_distance(
                    environment,
                    urgent_agent,
                    targets[urgent_agent.agent_id],
                    layout.charger_position,
                    available_pickups,
                )
                remaining_battery = (
                    urgent_agent.battery
                    - (
                        environment.config.move_battery_cost
                        if actions[urgent_agent.agent_id] in MOVE_DELTAS
                        else 0.0
                    )
                )
                required_battery = (
                    next_urgent_distance
                    + environment.config.mission_reserve_steps
                ) * environment.config.move_battery_cost
                if (
                    next_urgent_distance > current_urgent_distance
                    and remaining_battery < required_battery
                ):
                    # Penalize only clearance that breaks the safety reserve.
                    score += 2_000.0
            for loaded_agent in state.agents:
                if (
                    loaded_agent.carrying_task_id is None
                    or loaded_agent.navigation_goal_kind != "delivery"
                    or environment._requires_charge(state, loaded_agent)
                ):
                    continue
                current_delivery_distance = shortest_path_distance(
                    loaded_agent.position,
                    loaded_agent.navigation_goal_position,
                    layout_id,
                )
                next_delivery_distance = shortest_path_distance(
                    targets[loaded_agent.agent_id],
                    loaded_agent.navigation_goal_position,
                    layout_id,
                )
                if next_delivery_distance <= current_delivery_distance:
                    continue
                held_actions = dict(actions)
                held_actions[loaded_agent.agent_id] = "WAIT"
                _, _, held_invalid, held_collision, _, _ = (
                    environment._resolve_motion(state, held_actions)
                )
                if (
                    not held_collision
                    and not held_invalid
                    and not is_necessary_urgent_charger_clearance(
                        environment,
                        state,
                        loaded_agent,
                    )
                    and not necessary_teammate_route_clearance(
                        environment,
                        state,
                        loaded_agent,
                    )
                ):
                    # A loaded robot must not spend two battery units moving
                    # away from its delivery when holding for this one joint
                    # step is collision-free.  This keeps the offline label
                    # generator aligned with the formal detour metric and
                    # makes the teammate clear first.  It is training-only;
                    # no runtime action is ever rewritten by this function.
                    score += 100_000.0
            for departing_agent in state.agents:
                if (
                    departing_agent.position == layout.charger_position
                    or targets[departing_agent.agent_id]
                    != layout.charger_position
                    or departing_agent.last_charger_departure_frame is None
                    or state.frame - departing_agent.last_charger_departure_frame > 6
                ):
                    continue
                made_mission_progress, made_coordination_progress = (
                    charger_departure_progress(state, departing_agent)
                )
                if not (made_mission_progress or made_coordination_progress):
                    # Prefer the ordinary side apron over re-entering the
                    # charger during a clearance.  The latter is recorded as
                    # an unproductive departure/return cycle and was present
                    # in early-return supervision despite zero collisions.
                    score += 200_000.0
            candidates.append(((score, left_index, right_index), actions))
    if not candidates:
        selected = {
            agent_id: "WAIT"
            for agent_id in environment.agent_ids
        }
    else:
        selected = min(candidates, key=lambda item: item[0])[1]
    return _teacher_efficiency_guard(
        environment,
        selected,
    )
