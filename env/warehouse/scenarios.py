"""Deterministic environment-state curricula and evaluation scenarios.

These helpers only construct initial environment states.  They never select,
replace, or submit an action; every action after construction comes from the
MAPPO Actor (or from the explicitly simulated participant in noisy evaluation).
"""

from __future__ import annotations

from itertools import combinations

from .environment import WarehouseMultiAgentEnv
from .navigation import pickup_pairs, shortest_path_distance


def _uses_compact_staggered_layout(
    environment: WarehouseMultiAgentEnv,
) -> bool:
    """Return whether the active map follows the compact staggered grammar."""

    layout = environment.layout
    return (
        ("warehouse_staggered" in layout.layout_id or "staggered_loop" in layout.layout_id)
        and len(layout.robot_exit_positions) == 3
    )


def _uses_six_by_seven_layout(environment: WarehouseMultiAgentEnv) -> bool:
    return environment.layout.rows == 6 and environment.layout.cols == 7


def _loop_layout(environment: WarehouseMultiAgentEnv) -> bool:
    return "staggered_loop_aisles" in environment.layout.layout_id


def _straight_passable_triples(
    environment: WarehouseMultiAgentEnv,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...
]:
    """Return ordered straight three-cell paths in the active topology."""

    triples = []
    for row, column in environment.layout.passable_positions:
        for row_delta, column_delta in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            triple = (
                (row, column),
                (row + row_delta, column + column_delta),
                (row + 2 * row_delta, column + 2 * column_delta),
            )
            if all(environment.layout.is_passable(position) for position in triple):
                triples.append(triple)
    return tuple(triples)


def _passable_neighbors(
    environment: WarehouseMultiAgentEnv,
    position: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    row, column = position
    return tuple(
        candidate
        for candidate in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
        if environment.layout.is_passable(candidate)
    )


def _compact_same_target_conflicts(
    environment: WarehouseMultiAgentEnv,
) -> tuple[
    tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        bool,
        bool,
    ],
    ...,
]:
    """Derive contested-junction states from the active immutable map.

    Each robot begins at a different neighbour of one T junction and has a
    delivery whose shortest path enters that same junction.  The construction
    deliberately avoids action labels: the offline teacher still supplies the
    collision-free target during training.
    """

    layout = environment.layout
    excluded_goals = {
        layout.charger_position,
        *layout.robot_start_positions,
        *layout.robot_exit_positions,
        *layout.task_endpoint_exclusions,
    }
    passable_goals = tuple(
        position
        for position in layout.passable_positions
        if position not in excluded_goals
    )
    specifications = []
    for target in layout.passable_positions:
        # The charger apron is covered by dedicated queue/commitment curricula;
        # same-target examples here focus on warehouse work-aisle junctions.
        if target in {
            layout.charger_position,
            *layout.robot_start_positions,
            *layout.robot_exit_positions,
        }:
            continue
        neighbors = _passable_neighbors(environment, target)
        if len(neighbors) < 3:
            continue
        for first_position, second_position in combinations(neighbors, 2):
            first_candidates = sorted(
                (
                    goal
                    for goal in passable_goals
                    if goal not in {target, first_position, second_position}
                    and shortest_path_distance(
                        first_position,
                        goal,
                        environment.config.map_layout_id,
                    )
                    == 1
                    + shortest_path_distance(
                        target,
                        goal,
                        environment.config.map_layout_id,
                    )
                ),
                key=lambda goal: (
                    -shortest_path_distance(
                        target,
                        goal,
                        environment.config.map_layout_id,
                    ),
                    goal,
                ),
            )
            second_candidates = sorted(
                (
                    goal
                    for goal in passable_goals
                    if goal not in {
                        target,
                        first_position,
                        second_position,
                    }
                    and shortest_path_distance(
                        second_position,
                        goal,
                        environment.config.map_layout_id,
                    )
                    == 1
                    + shortest_path_distance(
                        target,
                        goal,
                        environment.config.map_layout_id,
                    )
                ),
                key=lambda goal: (
                    -shortest_path_distance(
                        target,
                        goal,
                        environment.config.map_layout_id,
                    ),
                    goal,
                ),
            )
            if not first_candidates or not second_candidates:
                continue
            first_goal = first_candidates[0]
            second_goal = next(
                (goal for goal in second_candidates if goal != first_goal),
                None,
            )
            if second_goal is None:
                continue
            specifications.append(
                (
                    first_position,
                    second_position,
                    first_goal,
                    second_goal,
                    True,
                    True,
                )
            )
    if not specifications:
        raise ValueError("Compact layout has no derivable contested junctions.")
    return tuple(specifications)


def apply_head_on_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    reverse: bool,
    variant: int = 0,
) -> None:
    """Place two loaded robots in an opposing compact-corridor state.

    ``variant`` covers both vertical spine encounters and the horizontal
    one-cell work aisles.  Training only the former left a systematic blind
    spot: two loaded robots could meet in row 4, where the lower-priority
    robot must retreat to a side junction before the priority robot may enter
    its frozen-state cell.  The catalog is state construction only and never
    participates in runtime action selection.
    """

    state = environment.get_state()
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    if _loop_layout(environment):
        upper, lower = (3, 3), (3, 4)
        if reverse:
            robot_one.position, robot_two.position = lower, upper
            destinations = ((3, 1), (3, 7))
            pickups = ((1, 7), (1, 1))
        else:
            robot_one.position, robot_two.position = upper, lower
            destinations = ((3, 7), (3, 1))
            pickups = ((1, 1), (1, 7))
    elif _uses_compact_staggered_layout(environment):
        # (first position, second position, goal beyond second, goal beyond
        # first, pickup for first carrier, pickup for second carrier).
        compact_encounters = (
            (
                ((1, 3), (2, 3), (3, 0), (0, 3), (1, 0), (2, 6)),
                ((2, 3), (2, 4), (2, 6), (1, 0), (3, 0), (0, 3)),
                ((4, 2), (4, 3), (4, 4), (3, 0), (1, 0), (2, 6)),
            )
            if _uses_six_by_seven_layout(environment)
            else (
                ((3, 4), (4, 4), (6, 6), (1, 1), (1, 0), (6, 8)),
                ((4, 4), (4, 5), (4, 7), (3, 1), (1, 0), (6, 8)),
                ((3, 3), (3, 4), (4, 7), (3, 1), (6, 8), (1, 0)),
                ((2, 4), (2, 5), (2, 7), (1, 1), (1, 0), (6, 8)),
                ((5, 3), (5, 4), (4, 7), (5, 1), (6, 8), (1, 0)),
                ((6, 4), (6, 5), (6, 7), (5, 1), (1, 0), (6, 8)),
            )
        )
        (
            first,
            second,
            beyond_second,
            beyond_first,
            first_pickup,
            second_pickup,
        ) = compact_encounters[int(variant) % len(compact_encounters)]
        if reverse:
            robot_one.position, robot_two.position = second, first
            destinations = (beyond_first, beyond_second)
            pickups = (second_pickup, first_pickup)
        else:
            robot_one.position, robot_two.position = first, second
            destinations = (beyond_second, beyond_first)
            pickups = (first_pickup, second_pickup)
    elif reverse:
        robot_one.position, robot_two.position = (4, 5), (3, 5)
        destinations = ((1, 0), (4, 10))
        pickups = ((6, 10), (7, 0))
    else:
        robot_one.position, robot_two.position = (3, 5), (4, 5)
        destinations = ((4, 10), (1, 0))
        pickups = ((6, 10), (7, 0))
    for agent, task, destination, pickup in zip(
        (robot_one, robot_two),
        tasks,
        destinations,
        pickups,
    ):
        task.status = "carried"
        task.carrier_agent_id = agent.agent_id
        task.claimed_frame = state.frame
        task.delivery_position = destination
        task.pickup_position = pickup
        agent.carrying_task_id = task.task_id
        agent.battery = 100.0
    environment.set_state(state)


def apply_charger_handoff_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    occupant_agent_id: str,
    queued_battery: float,
    occupant_battery: float = 100.0,
    occupant_carrying: bool = False,
    queued_carrying: bool = False,
) -> None:
    """Place two robots in a charger queue without selecting their actions."""

    if occupant_agent_id not in environment.agent_ids:
        raise ValueError("Unknown charger occupant.")
    if not 0.0 < queued_battery < 100.0:
        raise ValueError("queued_battery must be between zero and 100.")
    if not 0.0 < occupant_battery <= 100.0:
        raise ValueError("occupant_battery must be in (0, 100].")
    state = environment.get_state()
    queued_agent_id = next(
        agent_id
        for agent_id in environment.agent_ids
        if agent_id != occupant_agent_id
    )
    occupant = state.by_id(occupant_agent_id)
    queued = state.by_id(queued_agent_id)
    occupant.position = environment.layout.charger_position
    occupant.battery = float(occupant_battery)
    occupant.carrying_task_id = None
    queued.position = (
        environment.layout.charger_position[0] - 1,
        environment.layout.charger_position[1],
    )
    queued.battery = float(queued_battery)
    queued.carrying_task_id = None
    carry_flags = {
        occupant.agent_id: bool(occupant_carrying),
        queued.agent_id: bool(queued_carrying),
    }
    for agent, task in zip(
        sorted(state.agents, key=lambda item: item.agent_id),
        sorted(state.tasks, key=lambda item: item.task_id),
    ):
        if not carry_flags[agent.agent_id]:
            continue
        task.status = "carried"
        task.carrier_agent_id = agent.agent_id
        task.claimed_frame = state.frame
        agent.carrying_task_id = task.task_id
    environment.set_state(state)


_DELIVERY_GOAL_CLEARANCE_LANES = (
    # (row, direction away from the central spine).  The trailing loaded
    # robot is one cell behind a teammate currently occupying its drop-off.
    # Both robots can move in the listed direction in the same transition.
    (1, -1),
    (2, 1),
    (3, -1),
    (4, 1),
    (5, -1),
    (6, 1),
    (7, -1),
)

_DELIVERY_GOAL_CLEARANCE_BATTERIES = (
    (92.0, 88.0),
    (64.0, 58.0),
    (40.0, 34.0),
    (32.0, 26.0),
)


def apply_delivery_goal_clearance_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    variant: int,
) -> None:
    """Place two loaded robots in a follow-through delivery state.

    One robot is a single step from its B point, which is temporarily
    occupied by its loaded teammate.  The teammate has an unobstructed route
    farther along the branch.  Under the causal decentralized protocol the
    teammate leaves first and the trailing robot enters the vacated B cell
    from the next frozen state; it never relies on the peer's private current
    action.

    This function constructs state only.  It never selects or submits an
    action and is therefore safe for pure-Actor curriculum rollouts.
    """

    state = environment.get_state()
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    if len(tasks) < 2:
        raise ValueError("The goal-clearance curriculum requires two tasks.")
    lanes = _DELIVERY_GOAL_CLEARANCE_LANES
    center_column = 5
    compact_triples = None
    if _uses_compact_staggered_layout(environment):
        compact_triples = tuple(
            triple
            for triple in _straight_passable_triples(environment)
            if triple[1] not in {
                environment.layout.charger_position,
                *environment.layout.robot_start_positions,
                *environment.layout.robot_exit_positions,
            }
            and triple[2] not in environment.layout.dead_end_positions
        )
        if not compact_triples:
            raise ValueError("Compact layout has no delivery-clearance triples.")
        lanes = tuple(range(len(compact_triples)))
    lane_index = int(variant) % len(lanes)
    lane = lanes[lane_index]
    battery_index = (
        int(variant) // len(lanes)
    ) % len(_DELIVERY_GOAL_CLEARANCE_BATTERIES)
    batteries = _DELIVERY_GOAL_CLEARANCE_BATTERIES[battery_index]

    if compact_triples is not None:
        trailing_position, occupied_delivery, teammate_delivery = compact_triples[
            lane_index
        ]
    else:
        row, direction = lane
        trailing_position = (row, center_column)
        occupied_delivery = (row, center_column + direction)
        # A second step away from the spine is sufficient to make the teammate
        # vacate the contested B cell while keeping the lower-battery variants
        # on a genuinely delivery-safe route instead of silently switching them
        # to a charge goal.
        teammate_delivery = (row, center_column + 2 * direction)
    agents = sorted(state.agents, key=lambda item: item.agent_id)
    trailing, teammate = agents
    trailing.position = trailing_position
    teammate.position = occupied_delivery
    trailing.battery = float(batteries[0])
    teammate.battery = float(batteries[1])

    deliveries = (occupied_delivery, teammate_delivery)
    reserved: set[tuple[int, int]] = set(deliveries)
    pickup_candidates = sorted(
        {
            access
            for _, access in pickup_pairs(environment.config.map_layout_id)
            if access not in environment.layout.task_endpoint_exclusions
        }
    )
    for agent, task, delivery in zip(
        (trailing, teammate),
        tasks,
        deliveries,
    ):
        pickup = next(
            candidate
            for candidate in pickup_candidates
            if candidate not in reserved
            and shortest_path_distance(
                candidate,
                delivery,
                environment.config.map_layout_id,
            )
            >= environment.config.minimum_task_distance
        )
        reserved.add(pickup)
        task.status = "carried"
        task.carrier_agent_id = agent.agent_id
        task.claimed_frame = state.frame
        task.pickup_position = pickup
        task.delivery_position = delivery
        agent.carrying_task_id = task.task_id
    environment.set_state(state)


_EMPTY_DELIVERY_CLEARANCE_GEOMETRIES = (
    # (empty occupant, loaded teammate, occupant's committed pickup).
    ((5, 3), (5, 4), (5, 1)),
    ((3, 3), (3, 4), (3, 1)),
    ((2, 5), (2, 4), (2, 7)),
    ((4, 5), (4, 4), (4, 7)),
)


def apply_empty_delivery_clearance_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    variant: int,
) -> None:
    """Place an empty robot on a loaded teammate's immediate B point.

    The empty robot has a stable pickup commitment in the only useful
    clearance direction.  The loaded robot must wait for one frozen-state
    transition and may enter the vacated delivery cell only on the following
    frame.  Identity and aisle variants prevent the Actor from memorising one
    robot number or coordinate.
    """

    if not _uses_compact_staggered_layout(environment):
        raise ValueError("Empty-delivery clearance requires the compact layout.")
    state = environment.get_state()
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    if len(tasks) < 2:
        raise ValueError("Empty-delivery clearance requires two tasks.")
    endpoint_candidates = tuple(
        sorted(
            {
                access
                for _, access in pickup_pairs(environment.config.map_layout_id)
                if access not in environment.layout.task_endpoint_exclusions
            }
        )
    )
    geometries = _EMPTY_DELIVERY_CLEARANCE_GEOMETRIES
    if _uses_six_by_seven_layout(environment):
        geometries = (
            ((3, 1), (3, 2), (3, 0)),
            ((2, 4), (2, 3), (2, 6)),
            ((1, 2), (1, 3), (1, 0)),
        )
    elif _loop_layout(environment):
        excluded = {
            environment.layout.charger_position,
            *environment.layout.robot_start_positions,
            *environment.layout.robot_exit_positions,
        }
        geometries = tuple(
            (occupant, loaded, committed_pickup)
            for loaded, occupant, committed_pickup in _straight_passable_triples(
                environment
            )
            if not ({loaded, occupant, committed_pickup} & excluded)
            and occupant not in environment.layout.dead_end_positions
            and committed_pickup in endpoint_candidates
        )
        if not geometries:
            raise ValueError("Loop layout has no empty-delivery clearance geometry.")
    geometry_count = len(geometries)
    occupant_position, loaded_position, committed_pickup = (
        geometries[int(variant) % geometry_count]
    )
    reverse_identity = bool((int(variant) // geometry_count) % 2)
    agents = sorted(state.agents, key=lambda item: item.agent_id)
    occupant = agents[int(reverse_identity)]
    loaded = agents[1 - int(reverse_identity)]
    loaded_task, empty_task = tasks[:2]

    reserved = {
        occupant_position,
        loaded_position,
        committed_pickup,
        environment.layout.charger_position,
    }
    loaded_pickup = max(
        (
            endpoint
            for endpoint in endpoint_candidates
            if endpoint not in reserved
            and shortest_path_distance(
                endpoint,
                occupant_position,
                environment.config.map_layout_id,
            )
            >= environment.config.minimum_task_distance
        ),
        key=lambda endpoint: (
            shortest_path_distance(
                endpoint,
                occupant_position,
                environment.config.map_layout_id,
            ),
            endpoint,
        ),
    )
    reserved.add(loaded_pickup)
    empty_delivery = max(
        (
            endpoint
            for endpoint in endpoint_candidates
            if endpoint not in reserved
            and shortest_path_distance(
                committed_pickup,
                endpoint,
                environment.config.map_layout_id,
            )
            >= environment.config.minimum_task_distance
        ),
        key=lambda endpoint: (
            shortest_path_distance(
                committed_pickup,
                endpoint,
                environment.config.map_layout_id,
            ),
            endpoint,
        ),
    )

    occupant.position = occupant_position
    occupant.battery = 100.0
    occupant.carrying_task_id = None
    occupant.route_commitment_task_id = empty_task.task_id
    occupant.charge_mode_active = False
    loaded.position = loaded_position
    loaded.battery = 100.0
    loaded.carrying_task_id = loaded_task.task_id
    loaded.route_commitment_task_id = None
    loaded.charge_mode_active = False

    loaded_task.status = "carried"
    loaded_task.carrier_agent_id = loaded.agent_id
    loaded_task.claimed_frame = state.frame
    loaded_task.pickup_position = loaded_pickup
    loaded_task.delivery_position = occupant_position
    empty_task.status = "available"
    empty_task.carrier_agent_id = None
    empty_task.claimed_frame = None
    empty_task.pickup_position = committed_pickup
    empty_task.delivery_position = empty_delivery
    environment.set_state(state)


def apply_dual_charger_approach_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    variant: int,
) -> None:
    """Place two low-energy robots at different charger entrances.

    The side-apron robot is one move from the station and has priority; the
    robot in one of the three real exit cells either waits or makes a
    nonconflicting approach without entering the same charger cell.
    Covering left, centre, and right exit approaches is important: ordinary
    rollouts can put both low-energy robots in one side column even though the
    shortest-path-only curriculum originally exercised the centre cell.  The
    state exercises a queue *before* either robot occupies the charger, which
    is distinct from the existing occupied-station handoff curriculum.
    """

    if not _uses_compact_staggered_layout(environment):
        raise ValueError("Dual charger approach requires the compact layout.")
    state = environment.get_state()
    agents = sorted(state.agents, key=lambda item: item.agent_id)
    approach_geometries = (
        (-1, 0),
        (1, 0),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    )
    geometry_count = len(approach_geometries)
    side, queued_column_offset = approach_geometries[
        int(variant) % geometry_count
    ]
    reverse_identity = bool((int(variant) // geometry_count) % 2)
    battery_profile = (
        (14.0, 22.0),
        (20.0, 28.0),
    )[(int(variant) // (2 * geometry_count)) % 2]
    entrant = agents[int(reverse_identity)]
    queued = agents[1 - int(reverse_identity)]
    charger_row, charger_column = environment.layout.charger_position
    entrant.position = (charger_row, charger_column + side)
    entrant.battery = battery_profile[0]
    queued.position = (
        charger_row - 1,
        charger_column + queued_column_offset,
    )
    queued.battery = battery_profile[1]
    for agent in (entrant, queued):
        agent.carrying_task_id = None
        agent.route_commitment_task_id = None
        agent.charge_mode_active = True
        agent.last_action = "WAIT"
        agent.last_executed_action = "WAIT"
    environment.set_state(state)


def apply_outer_exit_charger_approach_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    variant: int,
) -> None:
    """Place one urgent robot at an outer exit beside an idle teammate.

    The urgent robot must first traverse horizontally to the central exit and
    can enter the charger only on the following frozen-state transition.  This
    is the ordinary-rollout geometry that differs from a direct side-apron
    admission: the teammate below is not charging and should simply hold its
    cell while the urgent robot crosses above it.
    """

    if not _uses_compact_staggered_layout(environment):
        raise ValueError("Outer-exit charger approach requires the compact layout.")
    state = environment.get_state()
    agents = sorted(state.agents, key=lambda item: item.agent_id)
    base_variant = int(variant) % 12
    side = -1 if base_variant % 2 == 0 else 1
    reverse_identity = bool((base_variant // 2) % 2)
    urgent_battery = (12.0, 16.0, 20.0)[(base_variant // 4) % 3]
    urgent = agents[int(reverse_identity)]
    idle = agents[1 - int(reverse_identity)]
    charger_row, charger_column = environment.layout.charger_position
    urgent.position = (charger_row - 1, charger_column + side)
    urgent.battery = urgent_battery
    urgent.carrying_task_id = None
    urgent.route_commitment_task_id = None
    urgent.charge_mode_active = True
    idle.position = (charger_row, charger_column + side)
    idle.battery = 64.0
    idle.carrying_task_id = None
    idle.route_commitment_task_id = None
    idle.charge_mode_active = False
    for agent in (urgent, idle):
        agent.last_action = "WAIT"
        agent.last_executed_action = "WAIT"
    wait_history = int(variant) % 8
    urgent.avoidable_wait_streak = wait_history
    idle.avoidable_wait_streak = wait_history
    state.ineffective_joint_wait_streak = wait_history
    phase = (0, 32, 64, 88)[(int(variant) // 12) % 4]
    state.frame = min(phase, max(0, environment.config.horizon - 4))
    environment.set_state(state)


_SAME_TARGET_CONFLICTS = (
    ((3, 4), (4, 5), (4, 9), (2, 5), True, True),
    ((4, 6), (3, 5), (3, 0), (5, 5), True, True),
    ((5, 4), (4, 5), (4, 9), (6, 5), True, True),
    ((6, 6), (5, 5), (5, 0), (7, 5), True, True),
    ((7, 4), (6, 5), (6, 10), (8, 5), True, True),
    ((8, 4), (7, 5), (8, 10), (8, 7), True, True),
    # Failure-mined geometries from direct-neural deterministic rollouts.
    ((3, 5), (2, 6), (1, 4), (1, 5), True, True),
    ((7, 4), (7, 5), (5, 1), (7, 4), True, True),
    ((4, 5), (4, 7), (4, 9), (7, 2), False, True),
    ((7, 5), (6, 6), (3, 2), (1, 4), False, True),
    # Deterministic rollout failures: delivery/charge, charge/delivery, and
    # delivery/pickup pairs all approaching the same cell.
    ((5, 5), (6, 6), (8, 9), (6, 10), True, False),
    ((5, 0), (5, 2), (7, 0), (5, 1), False, True),
    ((3, 5), (2, 6), (2, 6), (6, 6), True, False),
    # Direct-Actor failures mined from independent deterministic evaluation:
    # two robots converge on (5, 5), either while a loaded robot has delivery
    # priority or while its teammate is making an urgent charger approach.
    ((4, 5), (6, 5), (6, 6), (4, 5), False, True),
    ((8, 5), (6, 5), (3, 1), (1, 0), True, False),
)

_CONFLICT_BATTERY_PROFILES = (
    (100.0, 100.0),
    (82.0, 82.0),
    (74.0, 70.0),
    (52.0, 40.0),
    (40.0, 52.0),
    (32.0, 28.0),
    (28.0, 32.0),
    (18.0, 24.0),
    (46.0, 12.0),
    (12.0, 46.0),
    (56.0, 52.0),
    (52.0, 56.0),
)


def apply_same_target_conflict_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    variant: int,
) -> None:
    """Place two loaded robots one move from the same contested cell.

    This is a state-only curriculum.  It deliberately spans several aisle
    junctions and both robot orderings so the Actor must learn the observable
    right-of-way relation instead of memorising one head-on coordinate.
    """

    state = environment.get_state()
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    if len(tasks) < 2:
        raise ValueError("The conflict curriculum requires two active tasks.")
    base_count = len(_SAME_TARGET_CONFLICTS)
    specifications = _SAME_TARGET_CONFLICTS
    if _uses_compact_staggered_layout(environment):
        specifications = _compact_same_target_conflicts(environment)
        base_count = len(specifications)
    specification = specifications[int(variant) % base_count]
    (
        first_position,
        second_position,
        first_goal,
        second_goal,
        first_carrying,
        second_carrying,
    ) = specification
    ordering = (int(variant) // base_count) % 2
    battery_profile = _CONFLICT_BATTERY_PROFILES[
        (int(variant) // (2 * base_count))
        % len(_CONFLICT_BATTERY_PROFILES)
    ]
    if ordering:
        first_position, second_position = second_position, first_position
        first_goal, second_goal = second_goal, first_goal
        first_carrying, second_carrying = second_carrying, first_carrying
        battery_profile = tuple(reversed(battery_profile))
    excluded = {
        first_position,
        second_position,
        first_goal,
        second_goal,
        environment.layout.charger_position,
    }
    far_endpoints: list[tuple[int, int]] = []
    shelf_access_positions = {
        access
        for _, access in pickup_pairs(environment.config.map_layout_id)
    }
    for goal in (first_goal, second_goal):
        far_endpoint = next(
            position
            for position in environment.layout.passable_positions
            if position in shelf_access_positions
            if position not in excluded
            and position not in far_endpoints
            and position not in environment.layout.task_endpoint_exclusions
            and shortest_path_distance(
                position,
                goal,
                environment.config.map_layout_id,
            )
            >= environment.config.minimum_task_distance
        )
        far_endpoints.append(far_endpoint)
    for agent, task, position, goal, carrying, far_endpoint, battery in zip(
        sorted(state.agents, key=lambda item: item.agent_id),
        tasks,
        (first_position, second_position),
        (first_goal, second_goal),
        (first_carrying, second_carrying),
        far_endpoints,
        battery_profile,
    ):
        agent.position = position
        agent.battery = float(battery)
        if carrying:
            agent.carrying_task_id = task.task_id
            task.status = "carried"
            task.carrier_agent_id = agent.agent_id
            task.claimed_frame = state.frame
            task.pickup_position = far_endpoint
            task.delivery_position = goal
        else:
            agent.carrying_task_id = None
            task.status = "available"
            task.carrier_agent_id = None
            task.claimed_frame = None
            task.pickup_position = goal
            task.delivery_position = far_endpoint
    environment.set_state(state)


_CRITICAL_CHARGER_APPROACH_POSITIONS = (
    (6, 5),
    (7, 5),
    (8, 5),
    (9, 4),
    (9, 6),
)


def apply_critical_charger_approach_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    approaching_agent_id: str,
    variant: int,
) -> None:
    """Place one robot on a last-safe approach to the free charger.

    This helper constructs training/evaluation state only.  It does not choose
    or submit an action, so the resulting transition is still Actor-controlled.
    """

    if approaching_agent_id not in environment.agent_ids:
        raise ValueError("Unknown approaching robot.")
    state = environment.get_state()
    approaching = state.by_id(approaching_agent_id)
    teammate = next(
        agent for agent in state.agents if agent.agent_id != approaching_agent_id
    )
    approach_positions = _CRITICAL_CHARGER_APPROACH_POSITIONS
    if _loop_layout(environment):
        approach_positions = tuple(
            position
            for position in sorted(
                environment.layout.passable_positions,
                key=lambda item: (
                    shortest_path_distance(
                        item,
                        environment.layout.charger_position,
                        environment.config.map_layout_id,
                    ),
                    item,
                ),
            )
            if position not in {
                environment.layout.charger_position,
                *environment.layout.robot_start_positions,
            }
        )[:8]
    elif _uses_compact_staggered_layout(environment):
        charger = environment.layout.charger_position
        approach_positions = tuple(
            position
            for position in sorted(
                environment.layout.passable_positions,
                key=lambda item: (
                    shortest_path_distance(
                        item,
                        charger,
                        environment.config.map_layout_id,
                    ),
                    item,
                ),
            )
            if position not in {
                charger,
                *environment.layout.robot_start_positions,
            }
        )[:8]
    position = approach_positions[
        int(variant) % len(approach_positions)
    ]
    distance = shortest_path_distance(
        position,
        environment.layout.charger_position,
        environment.config.map_layout_id,
    )
    reserve_steps = (
        1
        + (int(variant) // len(approach_positions)) % 3
    )
    approaching.position = position
    approaching.battery = float(
        (distance + reserve_steps) * environment.config.move_battery_cost
    )
    teammate.position = max(
        (
            candidate
            for candidate in environment.layout.passable_positions
            if candidate not in {position, environment.layout.charger_position}
        ),
        key=lambda candidate: shortest_path_distance(
            position,
            candidate,
            environment.config.map_layout_id,
        ),
    )
    teammate.battery = 100.0
    approaching.carrying_task_id = None
    teammate.carrying_task_id = None
    if (int(variant) // 3) % 2:
        task = sorted(state.tasks, key=lambda item: item.task_id)[0]
        task.status = "carried"
        task.carrier_agent_id = approaching.agent_id
        task.claimed_frame = state.frame
        reserved_endpoints = {
            endpoint
            for other_task in state.tasks
            for endpoint in (
                other_task.pickup_position,
                other_task.delivery_position,
            )
            if other_task.task_id != task.task_id
        }
        reserved_endpoints.add(task.pickup_position)
        task.delivery_position = max(
            (
                endpoint
                for endpoint in environment.layout.passable_positions
                if endpoint
                not in {
                    position,
                    teammate.position,
                    environment.layout.charger_position,
                }
                and endpoint not in reserved_endpoints
                and endpoint not in environment.layout.task_endpoint_exclusions
                and shortest_path_distance(
                    task.pickup_position,
                    endpoint,
                    environment.config.map_layout_id,
                )
                >= environment.config.minimum_task_distance
            ),
            key=lambda endpoint: shortest_path_distance(
                position,
                endpoint,
                environment.config.map_layout_id,
            ),
        )
        approaching.carrying_task_id = task.task_id
    environment.set_state(state)


def apply_charger_commitment_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    agent_id: str,
    variant: int,
) -> None:
    """Construct pre-departure and just-departed charger commitment states.

    No action is selected here.  The history fields are ordinary observations
    that let the neural Actor distinguish "continue necessary charging" from
    "leave with enough energy" and remember a recent departure.
    """

    if agent_id not in environment.agent_ids:
        raise ValueError("Unknown charger commitment robot.")
    state = environment.get_state()
    state.frame = 20 + int(variant) % 20
    agent = state.by_id(agent_id)
    teammate = next(item for item in state.agents if item.agent_id != agent_id)
    charger_row, charger_column = environment.layout.charger_position
    phase = (int(variant) // 6) % 4
    battery = (22.0, 32.0, 42.0, 52.0, 62.0, 72.0)[int(variant) % 6]
    positions = (
        environment.layout.charger_position,
        (charger_row - 1, charger_column),
        (charger_row, charger_column - 1),
        (charger_row, charger_column + 1),
    )
    agent.position = positions[phase]
    recently_departed = phase != 0
    departure_actions = ("WAIT", "UP", "LEFT", "RIGHT")
    agent.last_action = departure_actions[phase]
    agent.last_executed_action = agent.last_action
    agent.last_battery_delta = (
        -environment.config.move_battery_cost if recently_departed else 10.0
    )
    elapsed = 1 + (int(variant) // 24) % 6
    agent.steps_since_charging = elapsed if recently_departed else 0
    agent.charger_wait_streak = 0 if recently_departed else 1 + int(variant) % 3
    agent.last_charger_departure_frame = (
        state.frame - elapsed if recently_departed else None
    )
    teammate.position = max(
        (
            position
            for position in environment.layout.passable_positions
            if position not in {agent.position, environment.layout.charger_position}
        ),
        key=lambda position: shortest_path_distance(
            agent.position,
            position,
            environment.config.map_layout_id,
        ),
    )
    teammate.battery = 100.0
    for item in state.agents:
        item.carrying_task_id = None
        item.route_commitment_task_id = None
        item.charge_mode_active = False
    if recently_departed:
        # A synthetic departure must be energy-feasible.  Earlier curriculum
        # variants placed a 20%-battery robot outside the charger, so the
        # correct teacher label was immediate re-entry and the Actor was
        # inadvertently trained to create the very six-step loops measured by
        # the release gate.  Commit to the cheapest available task and give
        # enough energy to continue it from every exit cell.
        available = tuple(task for task in state.tasks if task.status == "available")
        if available:
            task = min(
                available,
                key=lambda item: (
                    environment._mission_route_steps(
                        state,
                        agent,
                        item,
                        origin=agent.position,
                    ),
                    item.task_id,
                ),
            )
            required = (
                environment._mission_route_steps(
                    state,
                    agent,
                    task,
                    origin=agent.position,
                )
                * environment.config.move_battery_cost
            )
            agent.route_commitment_task_id = task.task_id
            agent.battery = min(
                100.0,
                max(
                    battery,
                    required
                    + environment.config.charge_release_hysteresis_steps
                    * environment.config.move_battery_cost,
                ),
            )
        else:
            agent.battery = max(battery, 60.0)
    else:
        agent.battery = battery
        agent.charge_mode_active = True
    environment.set_state(state)


def apply_task_commitment_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    variant: int,
) -> None:
    """Construct shared old/new task states for implicit neural division."""

    state = environment.get_state()
    state.frame = 60
    ordering = int(variant) % 2
    if _loop_layout(environment):
        positions = ((3, 3), (5, 4)) if not ordering else ((5, 4), (3, 3))
    elif _uses_compact_staggered_layout(environment):
        if _uses_six_by_seven_layout(environment):
            positions = (
                ((1, 3), (3, 2))
                if not ordering
                else ((3, 2), (1, 3))
            )
        else:
            positions = ((2, 4), (4, 4)) if not ordering else ((4, 4), (2, 4))
    else:
        positions = ((4, 5), (6, 5)) if not ordering else ((6, 5), (4, 5))
    for agent, position in zip(
        sorted(state.agents, key=lambda item: item.agent_id),
        positions,
    ):
        agent.position = position
        agent.battery = (76.0, 84.0)[int(agent.agent_id[-1]) - 1]
        agent.carrying_task_id = None
    old_task, new_task = sorted(state.tasks, key=lambda item: item.task_id)
    if not _uses_compact_staggered_layout(environment):
        old_task.pickup_position = (7, 0) if not ordering else (2, 10)
        old_task.delivery_position = (2, 8) if not ordering else (7, 2)
    old_task.created_frame = 0
    old_task.status = "available"
    old_task.carrier_agent_id = None
    old_task.claimed_frame = None
    if not _uses_compact_staggered_layout(environment):
        new_task.pickup_position = (6, 10) if not ordering else (3, 0)
        new_task.delivery_position = (1, 2) if not ordering else (6, 8)
    new_task.created_frame = 56
    new_task.status = "available"
    new_task.carrier_agent_id = None
    new_task.claimed_frame = None
    environment.set_state(state)
