"""Deterministic environment-state curricula and evaluation scenarios.

These helpers only construct initial environment states.  They never select,
replace, or submit an action; every action after construction comes from the
MAPPO Actor (or from the explicitly simulated participant in noisy evaluation).
"""

from __future__ import annotations

from .environment import WarehouseMultiAgentEnv
from .navigation import pickup_pairs, shortest_path_distance


def apply_head_on_scenario(
    environment: WarehouseMultiAgentEnv,
    *,
    reverse: bool,
) -> None:
    """Place two loaded robots in a reproducible opposing corridor state."""

    state = environment.get_state()
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    if reverse:
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
    farther along the branch, so the efficient joint transition is for the
    teammate to leave and the trailing robot to enter the vacated B cell.

    This function constructs state only.  It never selects or submits an
    action and is therefore safe for pure-Actor curriculum rollouts.
    """

    state = environment.get_state()
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    if len(tasks) < 2:
        raise ValueError("The goal-clearance curriculum requires two tasks.")
    lane_index = int(variant) % len(_DELIVERY_GOAL_CLEARANCE_LANES)
    row, direction = _DELIVERY_GOAL_CLEARANCE_LANES[lane_index]
    battery_index = (
        int(variant) // len(_DELIVERY_GOAL_CLEARANCE_LANES)
    ) % len(_DELIVERY_GOAL_CLEARANCE_BATTERIES)
    batteries = _DELIVERY_GOAL_CLEARANCE_BATTERIES[battery_index]

    trailing_position = (row, 5)
    occupied_delivery = (row, 5 + direction)
    # A second step away from the spine is sufficient to make the teammate
    # vacate the contested B cell while keeping the lower-battery variants on
    # a genuinely delivery-safe route instead of silently switching them to
    # a charge goal.
    teammate_delivery = (row, 5 + 2 * direction)
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
    specification = _SAME_TARGET_CONFLICTS[int(variant) % base_count]
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
    for goal in (first_goal, second_goal):
        far_endpoint = next(
            position
            for position in environment.layout.passable_positions
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
    position = _CRITICAL_CHARGER_APPROACH_POSITIONS[
        int(variant) % len(_CRITICAL_CHARGER_APPROACH_POSITIONS)
    ]
    distance = shortest_path_distance(
        position,
        environment.layout.charger_position,
        environment.config.map_layout_id,
    )
    reserve_steps = (
        1
        + (int(variant) // len(_CRITICAL_CHARGER_APPROACH_POSITIONS)) % 3
    )
    approaching.position = position
    approaching.battery = float(
        (distance + reserve_steps) * environment.config.move_battery_cost
    )
    teammate.position = (1, 0) if position != (1, 0) else (1, 1)
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
    just_departed = bool((int(variant) // 4) % 2)
    battery = (22.0, 32.0, 42.0, 52.0)[int(variant) % 4]
    agent.position = (
        (environment.layout.charger_position[0] - 1, environment.layout.charger_position[1])
        if just_departed
        else environment.layout.charger_position
    )
    agent.battery = battery - (environment.config.move_battery_cost if just_departed else 0.0)
    agent.last_action = "UP" if just_departed else "WAIT"
    agent.last_executed_action = agent.last_action
    agent.last_battery_delta = -2.0 if just_departed else 10.0
    agent.steps_since_charging = 1 if just_departed else 0
    agent.charger_wait_streak = 0 if just_departed else 1 + int(variant) % 3
    agent.last_charger_departure_frame = state.frame if just_departed else None
    teammate.position = (1, 0)
    teammate.battery = 100.0
    for item in state.agents:
        item.carrying_task_id = None
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
    positions = ((4, 5), (6, 5)) if not ordering else ((6, 5), (4, 5))
    for agent, position in zip(
        sorted(state.agents, key=lambda item: item.agent_id),
        positions,
    ):
        agent.position = position
        agent.battery = (76.0, 84.0)[int(agent.agent_id[-1]) - 1]
        agent.carrying_task_id = None
    old_task, new_task = sorted(state.tasks, key=lambda item: item.task_id)
    old_task.pickup_position = (7, 0) if not ordering else (2, 10)
    old_task.delivery_position = (2, 8) if not ordering else (7, 2)
    old_task.created_frame = 0
    old_task.status = "available"
    old_task.carrier_agent_id = None
    old_task.claimed_frame = None
    new_task.pickup_position = (6, 10) if not ordering else (3, 0)
    new_task.delivery_position = (1, 2) if not ordering else (6, 8)
    new_task.created_frame = 56
    new_task.status = "available"
    new_task.carrier_agent_id = None
    new_task.claimed_frame = None
    environment.set_state(state)
