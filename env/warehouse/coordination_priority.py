"""State-only right-of-way evidence shared by observations and supervision.

The functions in this module never choose or rewrite an action.  They derive a
stable coordination role from the frozen transition-start state so the Actor
features and the offline label generator cannot silently use contradictory
priority rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .domain import WarehouseConfig, WarehouseState
from .energy_management import (
    charger_departure_progress,
    charger_route_is_critical,
)
from .layouts import get_map_layout
from .navigation import MOVE_DELTAS, shortest_path_distance


@dataclass(frozen=True)
class CoordinationPriority:
    """Observable right-of-way role for one frozen joint state."""

    agent_id: str
    basis: str


def imminent_head_on_encounter(
    state: WarehouseState,
    config: WarehouseConfig,
    goal_positions: Mapping[str, tuple[int, int]],
) -> bool:
    """Return whether both shortest routes meet within the next two cells."""

    active = tuple(agent for agent in state.agents if agent.active)
    if len(active) != 2:
        return False
    left, right = active
    left_goal = goal_positions.get(left.agent_id, left.navigation_goal_position)
    right_goal = goal_positions.get(right.agent_id, right.navigation_goal_position)
    same_row = left.position[0] == right.position[0]
    same_column = left.position[1] == right.position[1]
    if not (same_row or same_column):
        return False
    axis = 1 if same_row else 0
    direct_distance = abs(left.position[axis] - right.position[axis])
    if not 0 < direct_distance <= 2:
        return False
    if (
        shortest_path_distance(
            left.position,
            right.position,
            config.map_layout_id,
        )
        != direct_distance
    ):
        return False

    coordinate_approach = bool(
        (left_goal[axis] - left.position[axis])
        * (right.position[axis] - left.position[axis])
        > 0
        and (right_goal[axis] - right.position[axis])
        * (left.position[axis] - right.position[axis])
        > 0
    )
    layout = get_map_layout(config.map_layout_id)
    progress_targets: dict[str, tuple[tuple[int, int], ...]] = {}
    for agent, goal in ((left, left_goal), (right, right_goal)):
        current_distance = shortest_path_distance(
            agent.position,
            goal,
            config.map_layout_id,
        )
        candidates: list[tuple[int, int]] = []
        for row_delta, column_delta in MOVE_DELTAS.values():
            target = (
                agent.position[0] + row_delta,
                agent.position[1] + column_delta,
            )
            if (
                layout.is_passable(target)
                and shortest_path_distance(
                    target,
                    goal,
                    config.map_layout_id,
                )
                < current_distance
            ):
                candidates.append(target)
        progress_targets[agent.agent_id] = tuple(candidates)
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
    return bool(coordinate_approach or topology_approach)


def single_lane_egress_agent_id(
    state: WarehouseState,
    config: WarehouseConfig,
    *,
    goal_positions: Mapping[str, tuple[int, int]],
) -> str | None:
    """Return the robot trapped behind its peer on one horizontal aisle arm.

    Every compact work aisle has one closed shelf end and opens onto the
    central spine.  When two robots occupy the same arm, the robot farther
    from the spine cannot step aside; the inner robot must clear toward (and,
    if necessary, off) the spine.  This priority is derived entirely from
    S_t topology and remains stable while the outer robot exits the arm.
    """

    active = tuple(agent for agent in state.agents if agent.active)
    if len(active) != 2:
        return None
    layout = get_map_layout(config.map_layout_id)
    spine_column = layout.charger_position[1]
    non_work_rows = {
        layout.charger_position[0],
        *(position[0] for position in layout.robot_exit_positions),
    }
    def needs_inward_progress(agent: object) -> bool:
        own_offset = agent.position[1] - spine_column
        if own_offset == 0:
            return False
        inward_delta = 1 if own_offset < 0 else -1
        inward = (agent.position[0], agent.position[1] + inward_delta)
        goal = goal_positions.get(
            agent.agent_id,
            agent.navigation_goal_position,
        )
        return bool(
            layout.is_passable(inward)
            and shortest_path_distance(
                inward,
                goal,
                config.map_layout_id,
            )
            < shortest_path_distance(
                agent.position,
                goal,
                config.map_layout_id,
            )
        )

    candidates: list[str] = []
    for agent in active:
        peer = next(item for item in active if item.agent_id != agent.agent_id)
        if (
            agent.position[0] != peer.position[0]
            or agent.position[0] in non_work_rows
        ):
            continue
        own_offset = agent.position[1] - spine_column
        peer_offset = peer.position[1] - spine_column
        if (
            own_offset == 0
            or own_offset * peer_offset < 0
            or abs(own_offset) <= abs(peer_offset)
        ):
            continue
        if not needs_inward_progress(agent):
            continue
        candidates.append(agent.agent_id)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return None

    # The clearing phase spans more than one frame. Once the inner robot has
    # moved vertically off the spine, the two robots are no longer on the
    # same row, but returning immediately would recreate the exact two-state
    # oscillation this priority is meant to prevent. Preserve the public
    # egress role using only actions already executed in S_t. No current-frame
    # action or future state is consulted.
    persistent: list[str] = []
    for agent in active:
        peer = next(item for item in active if item.agent_id != agent.agent_id)
        if (
            agent.position[0] in non_work_rows
            or agent.position[1] == spine_column
            or peer.position[1] != spine_column
            or not 0 < abs(peer.position[0] - agent.position[0]) <= 2
            or not needs_inward_progress(agent)
        ):
            continue
        own_offset = agent.position[1] - spine_column
        inward_action = "RIGHT" if own_offset < 0 else "LEFT"
        peer_delta = MOVE_DELTAS.get(peer.last_executed_action)
        peer_just_cleared = False
        if peer_delta is not None and peer_delta[0] != 0:
            previous_peer_row = peer.position[0] - peer_delta[0]
            previous_gap = abs(previous_peer_row - agent.position[0])
            current_gap = abs(peer.position[0] - agent.position[0])
            peer_just_cleared = bool(
                previous_gap <= 1 and current_gap > previous_gap
            )
        peer_holds_clearance = bool(
            peer.last_executed_action == "WAIT"
            and agent.last_executed_action in {inward_action, "WAIT"}
        )
        if peer_just_cleared or peer_holds_clearance:
            persistent.append(agent.agent_id)
    return persistent[0] if len(persistent) == 1 else None


def coordination_priority(
    state: WarehouseState,
    config: WarehouseConfig,
    *,
    goal_positions: Mapping[str, tuple[int, int]],
    goal_kinds: Mapping[str, str],
    requires_charge: Mapping[str, bool],
    imminent_head_on: bool,
) -> CoordinationPriority:
    """Select one stable priority role without observing either current action.

    A robot that is ready to leave the single charger clears it first.  A
    committed delivery then takes precedence over a robot that can wait on a
    charger route without spending energy.  Charging wins an actual imminent
    head-on encounter only when there is no loaded delivery commitment.
    """

    active = tuple(agent for agent in state.agents if agent.active)
    if not active:
        # A terminal observation can be requested after a random-policy
        # baseline shuts both robots down.  No future action will be selected;
        # keep the feature shape deterministic with the stable robot IDs.
        active = tuple(state.agents)
    layout = get_map_layout(config.map_layout_id)

    def goal(agent_id: str) -> tuple[int, int]:
        return goal_positions.get(
            agent_id,
            state.by_id(agent_id).navigation_goal_position,
        )

    def goal_kind(agent_id: str) -> str:
        return str(
            goal_kinds.get(
                agent_id,
                state.by_id(agent_id).navigation_goal_kind,
            )
        )

    charger_exit = tuple(
        agent
        for agent in active
        if agent.position == layout.charger_position
        and not bool(requires_charge.get(agent.agent_id, False))
        and goal(agent.agent_id) != layout.charger_position
    )
    loaded_delivery = tuple(
        agent
        for agent in active
        if agent.carrying_task_id is not None
        and goal_kind(agent.agent_id) == "delivery"
        and goal(agent.agent_id) != layout.charger_position
    )
    charging = tuple(
        agent
        for agent in active
        if goal_kind(agent.agent_id) == "charge"
        and goal(agent.agent_id) == layout.charger_position
    )
    critical_charging = tuple(
        agent
        for agent in charging
        if charger_route_is_critical(
            config,
            position=agent.position,
            battery=agent.battery,
            charger_position=layout.charger_position,
        )
    )
    charger_occupant = next(
        (
            agent
            for agent in active
            if agent.position == layout.charger_position
        ),
        None,
    )
    lower_energy_charger_waiters = tuple(
        agent
        for agent in charging
        if charger_occupant is not None
        and agent.agent_id != charger_occupant.agent_id
        and shortest_path_distance(
            agent.position,
            layout.charger_position,
            config.map_layout_id,
        )
        == 1
        and agent.battery < charger_occupant.battery
        and (
            charger_occupant.battery - agent.battery
            >= config.charge_per_wait
        )
        and charger_occupant.battery >= 2.0 * config.move_battery_cost
        and any(
            layout.is_passable(
                (
                    layout.charger_position[0],
                    layout.charger_position[1] + column_delta,
                )
            )
            for column_delta in (-1, 1)
        )
    )
    single_lane_egress_id = single_lane_egress_agent_id(
        state,
        config,
        goal_positions=goal_positions,
    )
    charger_clearance_commitments: list[object] = []
    if (
        len(charging) == 2
        and all(agent.position != layout.charger_position for agent in charging)
    ):
        for priority in charging:
            yielding = next(
                agent
                for agent in charging
                if agent.agent_id != priority.agent_id
            )
            yielding_delta = MOVE_DELTAS.get(yielding.last_executed_action)
            priority_delta = MOVE_DELTAS.get(priority.last_executed_action)
            yielding_moved_away = False
            if yielding_delta is not None:
                previous_yielding = (
                    yielding.position[0] - yielding_delta[0],
                    yielding.position[1] - yielding_delta[1],
                )
                yielding_moved_away = bool(
                    shortest_path_distance(
                        yielding.position,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                    > shortest_path_distance(
                        previous_yielding,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                    and priority.last_executed_action == "WAIT"
                )
            priority_moved_closer = False
            if priority_delta is not None:
                previous_priority = (
                    priority.position[0] - priority_delta[0],
                    priority.position[1] - priority_delta[1],
                )
                priority_moved_closer = bool(
                    shortest_path_distance(
                        priority.position,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                    < shortest_path_distance(
                        previous_priority,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                )
            yielding_moved_closer = False
            if yielding_delta is not None:
                previous_yielding = (
                    yielding.position[0] - yielding_delta[0],
                    yielding.position[1] - yielding_delta[1],
                )
                yielding_moved_closer = bool(
                    shortest_path_distance(
                        yielding.position,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                    < shortest_path_distance(
                        previous_yielding,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                )
            priority_charger_distance = shortest_path_distance(
                priority.position,
                layout.charger_position,
                config.map_layout_id,
            )
            yielding_charger_distance = shortest_path_distance(
                yielding.position,
                layout.charger_position,
                config.map_layout_id,
            )
            priority_leads_parallel_convoy = bool(
                priority_charger_distance < yielding_charger_distance
                and yielding_charger_distance
                == shortest_path_distance(
                    yielding.position,
                    priority.position,
                    config.map_layout_id,
                )
                + priority_charger_distance
            )
            parallel_convoy_progress = bool(
                priority_moved_closer
                and yielding_moved_closer
                and priority_leads_parallel_convoy
            )
            if (
                yielding_moved_away
                or (
                    priority_moved_closer
                    and (
                        yielding.last_executed_action == "WAIT"
                        or parallel_convoy_progress
                    )
                )
            ):
                charger_clearance_commitments.append(priority)
    if lower_energy_charger_waiters:
        # This priority is observable from S_t and matches the two-phase
        # occupied-station handoff used by reward and offline supervision.
        # It must precede loaded-delivery and charger-exit priority, otherwise
        # the Actor feature says the occupant owns the station while its label
        # says to clear for the lower-energy teammate.
        candidates = lower_energy_charger_waiters
        selection_mode = "lower_energy_charger_waiter"
    elif charger_exit:
        candidates = charger_exit
        selection_mode = "charger_exit"
    elif charger_occupant in charging and any(
        agent.agent_id != charger_occupant.agent_id
        and shortest_path_distance(
            agent.position,
            layout.charger_position,
            config.map_layout_id,
        )
        == 1
        for agent in charging
    ):
        # Until the shared handoff predicate sees a meaningful energy gap,
        # the undercharged occupant owns the station. Otherwise a two-unit
        # difference swaps both robots after every single charge wait.
        candidates = (charger_occupant,)
        selection_mode = "charger_occupant"
    elif len(charger_clearance_commitments) == 1:
        # A visible clearance move in S_t commits the same charger priority
        # until it passes the yielding robot. Re-ranking by battery slack
        # after every two-unit move otherwise flips the roles and creates a
        # three-state energy-draining oscillation.
        candidates = tuple(charger_clearance_commitments)
        selection_mode = "charger_clearance_commitment"
    elif critical_charging:
        # A robot with no clearance reserve left cannot safely retreat for a
        # loaded peer.  Give it the route before the conflict becomes an
        # unrecoverable zero-battery queue; ordinary charging still yields to
        # an energy-safe loaded delivery below.
        candidates = critical_charging
        selection_mode = "critical_charger_route"
    elif single_lane_egress_id is not None:
        candidates = (state.by_id(single_lane_egress_id),)
        selection_mode = "single_lane_egress"
    elif loaded_delivery:
        candidates = loaded_delivery
        selection_mode = "loaded_delivery"
    elif charging:
        # Two robots committed to the same single-cell station form a queue
        # even when the staggered apron means they are not geometrically
        # head-on yet. A lone charging robot also owns an empty approach over
        # an unloaded peer with no urgent mission; loaded delivery priority
        # has already been handled above. Give the least-slack charger the
        # route before the queue becomes a zero-battery deadlock.
        candidates = charging
        selection_mode = "charger_route"
    else:
        active_missions = tuple(
            agent
            for agent in active
            if goal_kind(agent.agent_id) != "wait"
            and goal(agent.agent_id) != agent.position
        )
        candidates = active_missions or active
        selection_mode = "mission"

    def urgent_charge(agent_id: str) -> bool:
        if (
            goal_kind(agent_id) != "charge"
            or goal(agent_id) != layout.charger_position
        ):
            return False
        agent = state.by_id(agent_id)
        return charger_route_is_critical(
            config,
            position=agent.position,
            battery=agent.battery,
            charger_position=layout.charger_position,
        )

    def recent_unproductive_departure(agent_id: str) -> bool:
        agent = state.by_id(agent_id)
        if agent.last_charger_departure_frame is None:
            return False
        if state.frame - agent.last_charger_departure_frame > 6:
            return False
        return not any(charger_departure_progress(state, agent))

    selected = min(
        candidates,
        key=lambda agent: (
            -int(
                agent.carrying_task_id is not None
                and goal_kind(agent.agent_id) == "delivery"
            ),
            -int(urgent_charge(agent.agent_id)),
            int(
                selection_mode in {
                    "critical_charger_route",
                    "charger_route",
                    "lower_energy_charger_waiter",
                }
                and recent_unproductive_departure(agent.agent_id)
            ),
            (
                agent.battery
                - shortest_path_distance(
                    agent.position,
                    layout.charger_position,
                    config.map_layout_id,
                )
                * config.move_battery_cost
                if selection_mode in {
                    "critical_charger_route",
                    "charger_route",
                    "lower_energy_charger_waiter",
                }
                else 0.0
            ),
            -int(agent.carrying_task_id is not None),
            shortest_path_distance(
                agent.position,
                goal(agent.agent_id),
                config.map_layout_id,
            ),
            agent.agent_id,
        ),
    )
    if selection_mode == "lower_energy_charger_waiter":
        basis = "lower_energy_charger_waiter"
    elif selection_mode == "charger_exit":
        basis = "charger_exit"
    elif selection_mode == "critical_charger_route":
        basis = "critical_charger_route"
    elif selection_mode == "charger_occupant":
        basis = "charger_occupant"
    elif selection_mode == "charger_clearance_commitment":
        basis = "charger_clearance_commitment"
    elif selection_mode == "loaded_delivery":
        basis = "loaded_delivery"
    elif selection_mode == "charger_route":
        basis = "urgent_charger_route"
    elif selection_mode == "single_lane_egress":
        basis = "single_lane_egress"
    elif (
        selected.carrying_task_id is not None
        and goal_kind(selected.agent_id) == "delivery"
    ):
        basis = "loaded_delivery"
    elif urgent_charge(selected.agent_id):
        basis = "urgent_charger_route"
    elif selected.carrying_task_id is not None:
        basis = "loaded_robot"
    else:
        basis = "shorter_route_or_stable_tie_break"
    return CoordinationPriority(selected.agent_id, basis)
