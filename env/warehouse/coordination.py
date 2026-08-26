"""Offline deterministic teacher used only to label MAPPO training data.

Nothing in this module is permitted to participate in environment, evaluation,
reference-rollout, or deployed action execution.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping


from .environment import ACTIONS, MOVE_DELTAS, WarehouseMultiAgentEnv, shortest_path_distance
from .teacher_efficiency import teacher_efficiency_guard as _teacher_efficiency_guard


def _claim_safe_distance(
    environment: WarehouseMultiAgentEnv,
    agent: Any,
    origin: tuple[int, int],
    goal: tuple[int, int],
    available_pickups: set[tuple[int, int]],
) -> int:
    """Shortest route that cannot accidentally claim another open task.

    Entering an available pickup is an irreversible claim, so a geometrically
    short path through a teammate's A point is not a legal route for the
    current assignment.  Treat those cells as temporary obstacles while the
    task is unclaimed.  Delivery and charging routes are unaffected.
    """

    if (
        agent.carrying_task_id is not None
        or (
            agent.navigation_goal_kind != "pickup"
            and goal not in available_pickups
        )
    ):
        return shortest_path_distance(
            origin,
            goal,
            environment.config.map_layout_id,
        )
    forbidden = available_pickups - ({goal} if goal in available_pickups else set())
    if origin == goal:
        return 0
    queue = deque(((origin, 0),))
    visited = {origin}
    while queue:
        position, distance = queue.popleft()
        for delta in MOVE_DELTAS.values():
            candidate = (position[0] + delta[0], position[1] + delta[1])
            if (
                candidate in visited
                or candidate in forbidden
                or not environment.layout.is_passable(candidate)
            ):
                continue
            if candidate == goal:
                return distance + 1
            visited.add(candidate)
            queue.append((candidate, distance + 1))
    # A task endpoint can be an articulation point.  This assignment is not
    # safely reachable until the other pickup has been claimed.
    return 10_000


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
    goal_distance = shortest_path_distance(
        agent.position,
        agent.navigation_goal_position,
        environment.config.map_layout_id,
    )
    charger_slack = (
        agent.battery - goal_distance * environment.config.move_battery_cost
    )
    return bool(charger_slack <= environment.config.charge_per_wait)


def _priority_key(
    environment: WarehouseMultiAgentEnv,
    agent: Any,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[int, int, int, int, str]:
    goal = (goal_overrides or {}).get(
        agent.agent_id,
        agent.navigation_goal_position,
    )
    goal_distance = shortest_path_distance(
        agent.position,
        goal,
        environment.config.map_layout_id,
    )
    return (
        # A loaded robot whose public navigation goal is still delivery has
        # already passed the safe-energy check.  Preserve that delivery
        # commitment ahead of a teammate that can safely wait on its charger
        # route without spending energy.  Truly unavoidable charger conflicts
        # are still resolved by the joint safety costs below.
        -int(
            agent.carrying_task_id is not None
            and agent.navigation_goal_kind == "delivery"
        ),
        -int(
            _urgent_charge(
                environment,
                agent,
                goal_overrides=goal_overrides,
            )
        ),
        -int(agent.carrying_task_id is not None),
        goal_distance,
        agent.agent_id,
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
    charging_agents = tuple(
        agent
        for agent in state.agents
        if agent.navigation_goal_kind == "charge"
        and (goal_overrides or {}).get(
            agent.agent_id,
            environment.layout.charger_position,
        )
        == environment.layout.charger_position
    )
    charger_exit_agents = tuple(
        agent
        for agent in state.agents
        if agent.position == layout.charger_position
        and not environment._requires_charge(state, agent)
        and (
            goal_overrides is None
            or (goal_overrides or {}).get(agent.agent_id)
            not in {None, layout.charger_position}
        )
    )
    loaded_delivery_agents = tuple(
        agent
        for agent in state.agents
        if agent.carrying_task_id is not None
        and agent.navigation_goal_kind == "delivery"
    )
    if charger_exit_agents:
        candidates = charger_exit_agents
        selection_mode = "charger_exit"
    elif loaded_delivery_agents:
        # Waiting is energy-neutral, while making a loaded robot reverse can
        # consume the exact reserve that made its delivery safe and send it
        # back to the charger.  A charging teammate therefore waits and lets
        # the committed delivery pass; true motion conflicts are still handled
        # by the joint candidate search below.
        candidates = loaded_delivery_agents
        selection_mode = "loaded_delivery"
    elif imminent_head_on and charging_agents:
        candidates = charging_agents
        selection_mode = "charger_route"
    else:
        candidates = tuple(state.agents)
        selection_mode = "mission"
    selected = min(
        candidates,
        key=lambda agent: _priority_key(
            environment,
            agent,
            goal_overrides=goal_overrides,
        ),
    )
    if selection_mode == "charger_exit":
        basis = "charger_exit"
    elif selection_mode == "loaded_delivery":
        basis = "loaded_delivery"
    elif selection_mode == "charger_route":
        basis = "urgent_charger_route"
    elif (
        selected.carrying_task_id is not None
        and selected.navigation_goal_kind == "delivery"
    ):
        basis = "loaded_delivery"
    elif _urgent_charge(
        environment,
        selected,
        goal_overrides=goal_overrides,
    ):
        basis = "urgent_charger_route"
    elif selected.carrying_task_id is not None:
        basis = "loaded_robot"
    else:
        basis = "shorter_route_or_stable_tie_break"
    return selected, basis


def _clear_head_on_encounter(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    state = environment.get_state()
    left, right = state.agents
    goals = {
        agent.agent_id: (goal_overrides or {}).get(
            agent.agent_id,
            agent.navigation_goal_position,
        )
        for agent in state.agents
    }
    aligned = (
        left.position[0] == right.position[0]
        or left.position[1] == right.position[1]
    )
    direct_distance = abs(left.position[0] - right.position[0]) + abs(
        left.position[1] - right.position[1]
    )
    if left.position[0] == right.position[0]:
        axis = 1
    elif left.position[1] == right.position[1]:
        axis = 0
    else:
        axis = -1
    coordinate_approach = bool(
        axis >= 0
        and (
            goals[left.agent_id][axis] - left.position[axis]
        )
        * (right.position[axis] - left.position[axis]) > 0
        and (
            goals[right.agent_id][axis] - right.position[axis]
        )
        * (left.position[axis] - right.position[axis]) > 0
    )
    topology_approach = False
    if axis >= 0:
        progress_targets: dict[str, tuple[tuple[int, int], ...]] = {}
        for agent in state.agents:
            current_goal_distance = shortest_path_distance(
                agent.position,
                goals[agent.agent_id],
                environment.config.map_layout_id,
            )
            candidates = []
            for row_delta, column_delta in MOVE_DELTAS.values():
                target = (
                    agent.position[0] + row_delta,
                    agent.position[1] + column_delta,
                )
                if (
                    environment.layout.is_passable(target)
                    and shortest_path_distance(
                        target,
                        goals[agent.agent_id],
                        environment.config.map_layout_id,
                    )
                    < current_goal_distance
                ):
                    candidates.append(target)
            progress_targets[agent.agent_id] = tuple(candidates)
        # A topological head-on exists only when *every* shortest-progress
        # choice of both robots continues along the shared line.  Using
        # ``any`` here falsely reserved a robot's current cell whenever one
        # shortest route approached its teammate, even if an equally short
        # perpendicular route was open.  The offline coordination teacher
        # then labelled the open move as WAIT and the Actor learned to idle
        # beside a charging teammate.  If either robot has a shortest route
        # that already leaves the shared line, ordinary joint scoring can use
        # it without invoking right-of-way clearance.
        topology_approach = bool(
            progress_targets[left.agent_id]
            and progress_targets[right.agent_id]
            and all(
                (target[axis] - left.position[axis])
                * (right.position[axis] - left.position[axis])
                > 0
                for target in progress_targets[left.agent_id]
            )
            and all(
                (target[axis] - right.position[axis])
                * (left.position[axis] - right.position[axis])
                > 0
                for target in progress_targets[right.agent_id]
            )
        )
    return bool(
        aligned
        and (coordinate_approach or topology_approach)
        and shortest_path_distance(
            left.position,
            right.position,
            environment.config.map_layout_id,
        )
        == direct_distance
        # Coordinate only immediately before a collision.  Longer geometric
        # lookahead made a loaded robot retreat even though the charging
        # teammate could wait without spending energy.
        and direct_distance <= 2
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
                and teammate.navigation_goal_kind == "charge"
                and teammate.position
                == (
                    environment.layout.charger_position[0] - 1,
                    environment.layout.charger_position[1],
                )
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
    charging_agents = tuple(
        agent
        for agent in state.agents
        if agent.navigation_goal_kind == "charge"
        and overrides.get(agent.agent_id, environment.layout.charger_position)
        == environment.layout.charger_position
    )
    priority_agent = min(
        charging_agents or tuple(state.agents),
        key=lambda agent: _priority_key(
            environment,
            agent,
            goal_overrides=overrides,
        ),
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



def stable_coordination_goal_overrides(
    environment: WarehouseMultiAgentEnv,
    *,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    """Return the exact temporary mission goals used by the offline teacher.

    The environment deliberately leaves available tasks unassigned, so its
    public navigation goal is ``wait`` for an empty robot. Offline callers
    that omit explicit goals must therefore freeze the optimal shared-task
    matching.  Explanation code can record this mapping at decision time
    instead of later misreporting the public ``wait`` goal as the reason for
    an action.
    """

    state = environment.get_state()
    if goal_overrides is None:
        overrides = {
            agent.agent_id: goal
            for agent in state.agents
            if (
                goal := environment._frozen_route_goal(
                    state,
                    agent.agent_id,
                    prioritize_old_tasks=True,
                )
            )
            is not None
        }
    else:
        overrides = dict(goal_overrides)
    available_pickups_for_matching = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    # When one open pickup is an articulation point on the route to another,
    # the farther robot must hold until the nearer task is physically claimed.
    # Otherwise it is guaranteed either to steal the teammate's task or to
    # consume its exact post-charge reserve in a later retreat.  Removing this
    # one-frame teacher goal does not reserve or assign either task in state.
    return {
        agent_id: goal
        for agent_id, goal in overrides.items()
        if not (
            goal in available_pickups_for_matching
            and _claim_safe_distance(
                environment,
                state.by_id(agent_id),
                state.by_id(agent_id).position,
                goal,
                available_pickups_for_matching,
            )
            >= 10_000
        )
    }


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
        if agent.navigation_goal_kind == "charge"
        and overrides.get(agent.agent_id, layout.charger_position)
        == layout.charger_position
    )
    charger_exit_agents = tuple(
        agent
        for agent in state.agents
        if agent.position == layout.charger_position
        and not environment._requires_charge(state, agent)
        and overrides.get(agent.agent_id) not in {
            None,
            layout.charger_position,
        }
    )
    priority_agent, _ = _priority_agent_and_basis(
        environment,
        imminent_head_on=imminent_head_on,
        goal_overrides=overrides,
    )
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
        and environment._requires_charge(state, agent)
    )
    if charging_occupants and adjacent_charger_waiters:
        # A robot that has not yet reached a safe departure state owns the
        # single charger for this frame.  Earlier teacher labels let a loaded
        # teammate displace it at the apron, producing a *premature* departure
        # followed by a measured return cycle.  Waiting at the apron costs no
        # energy and is therefore the only commitment-consistent label.
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
                        + environment.config.battery_safety_margin
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
                    + environment.config.battery_safety_margin
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
                made_mission_progress = bool(
                    departing_agent.deliveries_completed
                    > departing_agent.deliveries_at_last_charger_departure
                    or (
                        departing_agent.carrying_task_id is not None
                        and departing_agent.carrying_task_id
                        != departing_agent.carrying_task_at_last_charger_departure
                    )
                )
                if not made_mission_progress:
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


def _human_intent_task_override(
    environment: WarehouseMultiAgentEnv,
    proposed_actions: Mapping[str, str],
    fixed_agent_ids: tuple[str, ...],
) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """Infer an unambiguous participant pickup choice for the AI to avoid.

    Shared jobs remain unowned until pickup.  This is only a one-step
    coordination target: it does not reserve a task or change environment
    state.  An override is created only when the participant's requested move
    makes strictly more path progress toward one available pickup than every
    other pickup, so a common corridor move cannot be mistaken for intent.
    """

    if len(fixed_agent_ids) != 1:
        return {}, {}
    state = environment.get_state()
    human_id = fixed_agent_ids[0]
    if human_id not in environment.agent_ids:
        return {}, {}
    human = state.by_id(human_id)
    ai = next(agent for agent in state.agents if agent.agent_id != human_id)
    available = sorted(
        (task for task in state.tasks if task.status == "available"),
        key=lambda task: task.task_id,
    )
    requested = str(proposed_actions.get(human_id, "WAIT"))
    if (
        len(available) != 2
        or requested not in MOVE_DELTAS
        or human.carrying_task_id is not None
        or ai.carrying_task_id is not None
        or not human.active
        or not ai.active
        or ai.navigation_goal_kind == "charge"
    ):
        return {}, {}
    delta = MOVE_DELTAS[requested]
    requested_target = (
        human.position[0] + delta[0],
        human.position[1] + delta[1],
    )
    if not environment.layout.is_passable(requested_target):
        return {}, {}
    progress = []
    for task in available:
        before = shortest_path_distance(
            human.position,
            task.pickup_position,
            environment.config.map_layout_id,
        )
        after = shortest_path_distance(
            requested_target,
            task.pickup_position,
            environment.config.map_layout_id,
        )
        progress.append((before - after, task))
    progress.sort(key=lambda item: (-item[0], item[1].task_id))
    if progress[0][0] <= progress[1][0]:
        return {}, {}
    participant_task = progress[0][1]
    ai_task = next(
        task for task in available if task.task_id != participant_task.task_id
    )
    if ai.navigation_goal_position == ai_task.pickup_position:
        return {}, {}
    return (
        {ai.agent_id: ai_task.pickup_position},
        {ai.agent_id: ai_task.task_id},
    )


def _claim_safe_assignment_override(
    environment: WarehouseMultiAgentEnv,
    proposed_actions: Mapping[str, str],
) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """Repair a two-task match whose pickup order would cause a false race.

    The environment's energy-aware matcher compares total mission cost, but an
    open A point is also an automatic-claim cell.  When that cell lies on the
    route to the other A point, the nominally cheaper assignment can be
    operationally impossible.  This small exact matcher compares both task
    permutations using claim-safe path lengths and swaps responsibility only
    when the alternative is strictly shorter.
    """

    state = environment.get_state()
    agents = sorted(
        (
            agent
            for agent in state.agents
            if agent.active
            and agent.carrying_task_id is None
            and agent.navigation_goal_kind == "pickup"
        ),
        key=lambda agent: agent.agent_id,
    )
    tasks = sorted(
        (task for task in state.tasks if task.status == "available"),
        key=lambda task: task.task_id,
    )
    if len(agents) != 2 or len(tasks) != 2:
        return {}, {}
    pickups = {task.pickup_position for task in tasks}
    proposed_targets = environment._resolve_motion(
        state,
        proposed_actions,
    )[0]
    if not any(
        proposed_targets[agent.agent_id] != agent.position
        and proposed_targets[agent.agent_id] in pickups
        and proposed_targets[agent.agent_id] != agent.navigation_goal_position
        for agent in agents
    ):
        return {}, {}

    def assignment_cost(assigned: tuple[Any, Any]) -> int:
        return sum(
            _claim_safe_distance(
                environment,
                agent,
                agent.position,
                task.pickup_position,
                pickups,
            )
            for agent, task in zip(agents, assigned)
        )

    direct = (tasks[0], tasks[1])
    crossed = (tasks[1], tasks[0])
    candidates = sorted(
        (
            (assignment_cost(direct), tuple(task.task_id for task in direct), direct),
            (assignment_cost(crossed), tuple(task.task_id for task in crossed), crossed),
        ),
        key=lambda item: (item[0], item[1]),
    )
    selected_cost, _, selected = candidates[0]
    current_by_goal = {task.pickup_position: task for task in tasks}
    current = tuple(current_by_goal.get(agent.navigation_goal_position) for agent in agents)
    if any(task is None for task in current):
        return {}, {}
    current_tasks = tuple(task for task in current if task is not None)
    current_cost = assignment_cost(current_tasks)  # type: ignore[arg-type]
    if selected_cost >= current_cost or selected == current_tasks:
        return {}, {}
    return (
        {
            agent.agent_id: task.pickup_position
            for agent, task in zip(agents, selected)
        },
        {
            agent.agent_id: task.task_id
            for agent, task in zip(agents, selected)
        },
    )


def _mission_correction_reason(
    environment: WarehouseMultiAgentEnv,
    proposed: Mapping[str, str],
    corrected: Mapping[str, str],
    *,
    fixed_agent_ids: tuple[str, ...],
    goal_overrides: Mapping[str, tuple[int, int]],
) -> str | None:
    """Return why the deterministic alternative strictly dominates proposal."""

    state = environment.get_state()
    proposed_targets = environment._resolve_motion(
        state,
        proposed,
    )[0]
    corrected_collision = environment._resolve_motion(
        state,
        corrected,
    )[3]
    if corrected_collision:
        return None
    available_pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    wrong_pickup = False
    for agent in state.agents:
        if not agent.active or agent.agent_id in fixed_agent_ids:
            continue
        goal = goal_overrides.get(
            agent.agent_id,
            agent.navigation_goal_position,
        )
        proposed_target = proposed_targets[agent.agent_id]
        wrong_pickup = wrong_pickup or bool(
            agent.carrying_task_id is None
            and proposed_target in available_pickups
            and proposed_target != goal
        )
    if wrong_pickup:
        return "task_deconfliction"
    if goal_overrides:
        return "task_deconfliction"
    return None


def _mission_progress_offenders(
    environment: WarehouseMultiAgentEnv,
    proposed: Mapping[str, str],
    teacher: Mapping[str, str],
    *,
    fixed_agent_ids: tuple[str, ...],
    goal_overrides: Mapping[str, tuple[int, int]],
) -> tuple[str, ...]:
    """AI robots whose proposed step is a verified detour or idle stall."""

    state = environment.get_state()
    proposed_targets = environment._resolve_motion(state, proposed)[0]
    teacher_targets = environment._resolve_motion(state, teacher)[0]
    available_pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    offenders: list[str] = []
    for agent in state.agents:
        if (
            not agent.active
            or agent.agent_id in fixed_agent_ids
            or agent.navigation_goal_kind == "charge"
        ):
            continue
        goal = goal_overrides.get(
            agent.agent_id,
            agent.navigation_goal_position,
        )
        current_distance = _claim_safe_distance(
            environment,
            agent,
            agent.position,
            goal,
            available_pickups,
        )
        proposed_distance = _claim_safe_distance(
            environment,
            agent,
            proposed_targets[agent.agent_id],
            goal,
            available_pickups,
        )
        teacher_distance = _claim_safe_distance(
            environment,
            agent,
            teacher_targets[agent.agent_id],
            goal,
            available_pickups,
        )
        moved_away = proposed_distance > current_distance
        # A loaded robot that still has enough energy to deliver and retain a
        # safe charger route must not spend that reserve oscillating or
        # waiting.  The old low-battery exemption (<40) allowed exactly that:
        # the Actor could waste energy until `_requires_charge` flipped and
        # then take a longer charge-first route.  Correct a no-progress action
        # only when the collision-free joint teacher has a strictly closer
        # step; real yielding or a blocked route therefore remains untouched.
        stalled_safe_delivery = bool(
            agent.carrying_task_id is not None
            and not environment._requires_charge(state, agent)
            and proposed_distance >= current_distance
            and teacher_distance < current_distance
        )
        if (
            (moved_away and teacher_distance < proposed_distance)
            or stalled_safe_delivery
        ):
            offenders.append(agent.agent_id)
    return tuple(offenders)
