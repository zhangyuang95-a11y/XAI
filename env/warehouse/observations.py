"""Actor and centralized-critic observations for collaborative delivery."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import OBSERVATION_CONTRACT_VERSION
from .coordination_priority import (
    coordination_priority,
    imminent_head_on_encounter,
)
from .domain import WarehouseConfig, WarehouseState
from .navigation import (
    ACTIONS,
    CHARGER_POSITION,
    MOVE_DELTAS,
    is_passable,
    is_shelf,
    legal_action_mask,
    shortest_path_distance,
)
from .layouts import get_map_layout


NAVIGATION_GOAL_KINDS = ("pickup", "delivery", "charge", "wait")


def teammate_goal_kind_start(
    *,
    max_agents: int,
    active_task_count: int,
    action_dim: int,
) -> int:
    """Return the first teammate goal one-hot offset in a local observation."""

    return (
        34
        + int(max_agents)
        + 3 * int(action_dim)
        + 23 * int(active_task_count)
    )


def own_frames_since_charger_departure_index(*, action_dim: int) -> int:
    """Return the own recent-energy departure-age feature offset."""

    return 24 + 2 * int(action_dim) + 5


def teammate_steps_since_charging_index(
    *,
    max_agents: int,
    active_task_count: int,
    action_dim: int,
) -> int:
    """Return the teammate recent-energy charge-age feature offset."""

    return (
        teammate_goal_kind_start(
            max_agents=max_agents,
            active_task_count=active_task_count,
            action_dim=action_dim,
        )
        + len(NAVIGATION_GOAL_KINDS)
        + int(action_dim)
        + 13
    )


def teammate_legal_action_mask_start(
    *,
    max_agents: int,
    active_task_count: int,
    action_dim: int,
) -> int:
    """Return the first frozen teammate legal-action bit."""

    return (
        teammate_goal_kind_start(
            max_agents=max_agents,
            active_task_count=active_task_count,
            action_dim=action_dim,
        )
        + len(NAVIGATION_GOAL_KINDS)
        + 7
    )


def _normalize_delta(value: int, limit: int) -> float:
    return max(-1.0, min(1.0, value / max(1, limit - 1)))


def _actor_visible_goal(
    state: WarehouseState,
    agent: Any,
) -> tuple[str, tuple[int, int]]:
    """Expose only a pickup intent already chosen by the neural Actor.

    Available tasks remain unassigned and unreserved.  The extra goal exists
    only after this robot's own executed movement created a non-binding route
    commitment, and disappears as soon as that task is no longer available.
    Charging and carried-delivery modes always take precedence.
    """

    if (
        agent.navigation_goal_kind == "wait"
        and agent.carrying_task_id is None
        and (
            agent.route_commitment_task_id is not None
            or (
                agent.goal_type == "GO_TO_PICKUP"
                and agent.goal_id is not None
            )
        )
    ):
        task_id = agent.route_commitment_task_id or agent.goal_id
        task = next(
            (
                item
                for item in state.tasks
                if item.task_id == task_id
                and item.status == "available"
            ),
            None,
        )
        if task is not None:
            return "pickup", task.pickup_position
    return agent.navigation_goal_kind, agent.navigation_goal_position


def _actor_action_mask(
    state: WarehouseState,
    agent: Any,
    config: WarehouseConfig,
) -> list[float]:
    """Return the frozen-state safety mask, including an AI-AI joint plan.

    This is decided before either current action exists. Participant rounds
    retain the ordinary geometry mask so a human action is never represented
    as compliance with an AI coordination plan.
    """

    static = legal_action_mask(state, agent, config.map_layout_id)
    plan = state.active_coordination_plan
    if state.participant_controlled_agent_id is not None:
        return static
    if plan is None:
        other = next(
            item for item in state.agents if item.agent_id != agent.agent_id
        )
        _, goal = _actor_visible_goal(state, agent)
        current_distance = shortest_path_distance(
            agent.position,
            goal,
            config.map_layout_id,
        )
        progress_actions: dict[str, tuple[int, int]] = {}
        for action, allowed in zip(ACTIONS, static):
            if action not in MOVE_DELTAS or allowed <= 0.5:
                continue
            target = (
                agent.position[0] + MOVE_DELTAS[action][0],
                agent.position[1] + MOVE_DELTAS[action][1],
            )
            if shortest_path_distance(
                target,
                goal,
                config.map_layout_id,
            ) < current_distance:
                progress_actions[action] = target
        progress_targets = set(progress_actions.values())
        other_static = legal_action_mask(state, other, config.map_layout_id)
        other_targets = {
            action: (
                other.position
                if action == "WAIT"
                else (
                    other.position[0] + MOVE_DELTAS[action][0],
                    other.position[1] + MOVE_DELTAS[action][1],
                )
            )
            for action, allowed in zip(ACTIONS, other_static)
            if allowed > 0.5
        }
        robust_progress_actions = {
            action
            for action, target in progress_actions.items()
            if all(
                target != other_target
                and not (
                    target == other.position
                    and other_target == agent.position
                )
                for other_target in other_targets.values()
            )
        }
        if robust_progress_actions:
            return [
                1.0 if action in robust_progress_actions else 0.0
                for action in ACTIONS
            ]
        # A temporary step away from a locked goal is not a solution when the
        # only progress cell is occupied by the teammate. Wait for the frozen
        # blocker to move; a genuine clearance manoeuvre has an explicit plan
        # and is handled below instead.
        if progress_targets == {other.position} and static[ACTIONS.index("WAIT")] > 0.5:
            return [
                1.0 if action == "WAIT" else 0.0
                for action in ACTIONS
            ]
        return static
    phase = str(plan.get("phase", ""))
    expected: str | None = None
    if phase == "CLEAR_CELL":
        if str(plan.get("priority_agent_id")) == agent.agent_id:
            expected = "WAIT"
        elif str(plan.get("clearing_agent_id")) == agent.agent_id:
            expected = str(plan.get("moving_action", ""))
    elif phase in {"PASS_THROUGH", "SINGLE_STEP"}:
        if str(plan.get("moving_agent_id")) == agent.agent_id:
            expected = str(plan.get("moving_action", ""))
        elif str(plan.get("waiting_agent_id")) == agent.agent_id:
            expected = "WAIT"
    if expected not in ACTIONS:
        return static
    index = ACTIONS.index(expected)
    if static[index] <= 0.5:
        return static
    return [
        1.0 if action_index == index else 0.0
        for action_index in range(len(ACTIONS))
    ]


def _route_energy_features(
    state: WarehouseState,
    agent_id: str,
    task: Any,
    config: WarehouseConfig,
) -> tuple[float, float, float]:
    """Observable energy need, slack, and charge waits for one task slot."""

    agent = state.by_id(agent_id)
    layout = get_map_layout(config.map_layout_id)
    if task.status == "carried" and task.carrier_agent_id != agent_id:
        return 0.0, 0.0, 0.0
    if agent.carrying_task_id == task.task_id:
        route_steps = shortest_path_distance(
            agent.position,
            task.delivery_position,
            config.map_layout_id,
        )
        route_steps += shortest_path_distance(
            task.delivery_position,
            layout.charger_position,
            config.map_layout_id,
        )
        route_from_charger = shortest_path_distance(
            layout.charger_position,
            task.delivery_position,
            config.map_layout_id,
        ) + shortest_path_distance(
            task.delivery_position,
            layout.charger_position,
            config.map_layout_id,
        )
    else:
        route_steps = shortest_path_distance(
            agent.position,
            task.pickup_position,
            config.map_layout_id,
        ) + shortest_path_distance(
            task.pickup_position,
            task.delivery_position,
            config.map_layout_id,
        ) + shortest_path_distance(
            task.delivery_position,
            layout.charger_position,
            config.map_layout_id,
        )
        route_from_charger = shortest_path_distance(
            layout.charger_position,
            task.pickup_position,
            config.map_layout_id,
        ) + shortest_path_distance(
            task.pickup_position,
            task.delivery_position,
            config.map_layout_id,
        ) + shortest_path_distance(
            task.delivery_position,
            layout.charger_position,
            config.map_layout_id,
        )
    route_steps += config.mission_reserve_steps
    route_from_charger += config.mission_reserve_steps
    required_energy = route_steps * config.move_battery_cost
    direct_slack = agent.battery - required_energy
    charger_distance = shortest_path_distance(
        agent.position,
        layout.charger_position,
        config.map_layout_id,
    )
    battery_at_charger = max(
        0.0,
        agent.battery - charger_distance * config.move_battery_cost,
    )
    wait_count = math.ceil(
        max(
            0.0,
            route_from_charger * config.move_battery_cost - battery_at_charger,
        )
        / config.charge_per_wait
    )
    if direct_slack >= 0.0:
        wait_count = 0
    return (
        min(1.0, required_energy / 100.0),
        max(-1.0, min(1.0, direct_slack / 100.0)),
        min(1.0, wait_count / 10.0),
    )


def _recent_energy_features(
    state: WarehouseState,
    agent_id: str,
    config: WarehouseConfig,
) -> tuple[float, ...]:
    agent = state.by_id(agent_id)
    frames_since_departure = (
        config.horizon
        if agent.last_charger_departure_frame is None
        else max(0, state.frame - agent.last_charger_departure_frame)
    )
    return (
        *(1.0 if agent.last_executed_action == action else 0.0 for action in ACTIONS),
        max(-1.0, min(1.0, agent.last_battery_delta / config.charge_per_wait)),
        min(1.0, agent.steps_since_charging / max(1, config.horizon)),
        min(1.0, agent.charger_wait_streak / 10.0),
        float(agent.charge_mode_active),
        min(1.0, agent.avoidable_wait_streak / 5.0),
        min(1.0, frames_since_departure / max(1, config.horizon)),
        float(frames_since_departure <= 4),
    )


def _local_patch(
    state: WarehouseState,
    agent_id: str,
    config: WarehouseConfig,
) -> list[float]:
    agent = state.by_id(agent_id)
    layout = get_map_layout(config.map_layout_id)
    other_positions = {
        item.position
        for item in state.agents
        if item.agent_id != agent_id
    }
    pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    deliveries = {task.delivery_position for task in state.tasks}
    values: list[float] = []
    for row_delta in range(-config.local_patch_radius, config.local_patch_radius + 1):
        for column_delta in range(-config.local_patch_radius, config.local_patch_radius + 1):
            position = (
                agent.position[0] + row_delta,
                agent.position[1] + column_delta,
            )
            if not is_passable(position, config.map_layout_id):
                code = 1.0
            elif position in other_positions:
                code = 0.6
            elif position == layout.charger_position:
                code = 0.4
            elif position in layout.dead_end_positions:
                code = 0.32
            elif position in pickups:
                code = 0.25
            elif position in deliveries:
                code = 0.2
            else:
                code = 0.0
            values.append(code)
    return values


def _passable_neighbors(
    position: tuple[int, int],
    config: WarehouseConfig,
) -> tuple[tuple[int, int], ...]:
    candidates = (
        (position[0] - 1, position[1]),
        (position[0] + 1, position[1]),
        (position[0], position[1] - 1),
        (position[0], position[1] + 1),
    )
    return tuple(
        item for item in candidates if is_passable(item, config.map_layout_id)
    )


def _corridor_axis(
    position: tuple[int, int],
    config: WarehouseConfig,
) -> str:
    layout = get_map_layout(config.map_layout_id)
    neighbors = _passable_neighbors(position, config)
    if len(neighbors) != 2:
        return "junction"
    rows = {item[0] for item in neighbors}
    columns = {item[1] for item in neighbors}
    if len(rows) == 1:
        return "horizontal"
    if len(columns) == 1:
        return "vertical"
    return "turn"


def _coordination_features(
    state: WarehouseState,
    agent_id: str,
    config: WarehouseConfig,
) -> list[float]:
    layout = get_map_layout(config.map_layout_id)
    agent = state.by_id(agent_id)
    other = next(item for item in state.agents if item.agent_id != agent_id)
    own_goal_kind, own_goal_position = _actor_visible_goal(state, agent)
    other_goal_kind, other_goal_position = _actor_visible_goal(state, other)
    own_axis = _corridor_axis(agent.position, config)
    other_axis = _corridor_axis(other.position, config)
    same_axis = own_axis in {"horizontal", "vertical"} and own_axis == other_axis
    aligned = (
        (own_axis == "horizontal" and agent.position[0] == other.position[0])
        or (own_axis == "vertical" and agent.position[1] == other.position[1])
    )
    distance = abs(agent.position[0] - other.position[0]) + abs(
        agent.position[1] - other.position[1]
    )
    nearest_dead_end = min(
        (
            shortest_path_distance(
                agent.position,
                bay,
                config.map_layout_id,
            )
            for bay in layout.dead_end_positions
            if bay != other.position
        ),
        default=config.rows * config.cols,
    )
    visible_goals = {
        agent.agent_id: own_goal_position,
        other.agent_id: other_goal_position,
    }
    visible_goal_kinds = {
        agent.agent_id: own_goal_kind,
        other.agent_id: other_goal_kind,
    }
    priority = coordination_priority(
        state,
        config,
        goal_positions=visible_goals,
        goal_kinds=visible_goal_kinds,
        requires_charge={
            agent.agent_id: bool(
                agent.navigation_goal_kind == "charge"
                or agent.charge_mode_active
            ),
            other.agent_id: bool(
                other.navigation_goal_kind == "charge"
                or other.charge_mode_active
            ),
        },
        imminent_head_on=imminent_head_on_encounter(
            state,
            config,
            visible_goals,
        ),
    )
    if state.active_coordination_plan is not None:
        planned_priority = str(
            state.active_coordination_plan.get("priority_agent_id", "")
        )
        if planned_priority in {agent.agent_id, other.agent_id}:
            own_priority = planned_priority == agent.agent_id
        else:
            own_priority = priority.agent_id == agent.agent_id
    else:
        own_priority = priority.agent_id == agent.agent_id
    axis = (
        1
        if agent.position[0] == other.position[0]
        else 0 if agent.position[1] == other.position[1] else -1
    )
    approaching = bool(
        axis >= 0
        and (
            own_goal_position[axis] - agent.position[axis]
        )
        * (other.position[axis] - agent.position[axis]) > 0
        and (
            other_goal_position[axis] - other.position[axis]
        )
        * (agent.position[axis] - other.position[axis]) > 0
    )
    action_deltas = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "WAIT": (0, 0),
    }
    other_possible_targets = {
        (
            other.position[0] + action_deltas[action][0],
            other.position[1] + action_deltas[action][1],
        )
        for action, allowed in zip(
            ACTIONS,
            _actor_action_mask(state, other, config),
        )
        if allowed > 0.5
    }
    available_pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    own_current_goal_distance = shortest_path_distance(
        agent.position,
        own_goal_position,
        config.map_layout_id,
    )
    other_current_goal_distance = shortest_path_distance(
        other.position,
        other_goal_position,
        config.map_layout_id,
    )
    action_targets: dict[str, tuple[int, int]] = {}
    other_action_targets: dict[str, tuple[int, int]] = {}
    for action in ACTIONS:
        delta = action_deltas[action]
        candidate = (
            agent.position[0] + delta[0],
            agent.position[1] + delta[1],
        )
        action_targets[action] = (
            candidate if layout.is_passable(candidate) else agent.position
        )
        other_candidate = (
            other.position[0] + delta[0],
            other.position[1] + delta[1],
        )
        other_action_targets[action] = (
            other_candidate
            if layout.is_passable(other_candidate)
            else other.position
        )
    stable_tasks = sorted(state.tasks, key=lambda item: item.task_id)[
        : config.active_task_count
    ]

    def task_action_features(
        acting_agent: Any,
        target: tuple[int, int],
    ) -> tuple[float, ...]:
        """Per-task neural action evidence in stable task-slot order.

        A route commitment is useful memory, but it must not hide an older
        shared task.  These features let the Actor itself compare progress
        toward both task slots and retarget; no assignment or action is
        produced here.
        """

        result: list[float] = []
        for task in stable_tasks:
            if task.status == "available":
                task_goal = task.pickup_position
                relevant = True
            elif task.carrier_agent_id == acting_agent.agent_id:
                task_goal = task.delivery_position
                relevant = True
            else:
                task_goal = acting_agent.position
                relevant = False
            before_distance = shortest_path_distance(
                acting_agent.position,
                task_goal,
                config.map_layout_id,
            )
            after_distance = shortest_path_distance(
                target,
                task_goal,
                config.map_layout_id,
            )
            remaining_battery = max(
                0.0,
                acting_agent.battery
                - (
                    config.move_battery_cost
                    if target != acting_agent.position
                    else 0.0
                ),
            )
            if task.status == "available":
                route_steps = (
                    shortest_path_distance(
                        target,
                        task.pickup_position,
                        config.map_layout_id,
                    )
                    + shortest_path_distance(
                        task.pickup_position,
                        task.delivery_position,
                        config.map_layout_id,
                    )
                    + shortest_path_distance(
                        task.delivery_position,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                    + config.mission_reserve_steps
                )
            elif task.carrier_agent_id == acting_agent.agent_id:
                route_steps = (
                    shortest_path_distance(
                        target,
                        task.delivery_position,
                        config.map_layout_id,
                    )
                    + shortest_path_distance(
                        task.delivery_position,
                        layout.charger_position,
                        config.map_layout_id,
                    )
                    + config.mission_reserve_steps
                )
            else:
                route_steps = 0.0
            required_energy = route_steps * config.move_battery_cost
            energy_deficit = max(0.0, required_energy - remaining_battery)
            result.extend(
                (
                    after_distance / float(config.rows * config.cols)
                    if relevant
                    else 0.0,
                    float(
                        max(-1, min(1, before_distance - after_distance))
                    )
                    if relevant
                    else 0.0,
                    min(
                        1.0,
                        max(0, state.frame - task.created_frame)
                        / max(1, config.reward.task_age_priority_horizon),
                    )
                    if task.status == "available"
                    else 0.0,
                    min(1.0, required_energy / 100.0) if relevant else 0.0,
                    max(
                        -1.0,
                        min(1.0, (remaining_battery - required_energy) / 100.0),
                    )
                    if relevant
                    else 0.0,
                    min(
                        1.0,
                        math.ceil(energy_deficit / config.charge_per_wait)
                        / 10.0,
                    )
                    if relevant
                    else 0.0,
                )
            )
        result.extend(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            * (config.active_task_count - len(stable_tasks))
        )
        return tuple(result)
    values = [
        float(agent.position in layout.dead_end_positions),
        float(own_axis in {"horizontal", "vertical"}),
        nearest_dead_end / float(config.rows * config.cols),
        float(other_axis in {"horizontal", "vertical"}),
        float(same_axis and aligned),
        float(same_axis and aligned and approaching and distance <= 6),
        float(own_priority),
        float(not own_priority),
    ]
    for action in ACTIONS:
        delta = action_deltas[action]
        target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
        values.extend(
            (
                float(target in other_possible_targets),
                float(
                    target == other.position
                    and agent.position in other_possible_targets
                ),
            )
        )
    distance_scale = float(config.rows * config.cols)
    for action in ACTIONS:
        target = action_targets[action]
        goal_distance = shortest_path_distance(
            target,
            own_goal_position,
            config.map_layout_id,
        )
        values.extend(
            (
                goal_distance / distance_scale,
                shortest_path_distance(
                    target,
                    layout.charger_position,
                    config.map_layout_id,
                )
                / distance_scale,
                float(max(-1, min(1, own_current_goal_distance - goal_distance))),
                float(target == other.position),
                float(
                    target in available_pickups
                    and target != own_goal_position
                ),
                float(target == layout.charger_position),
                float(agent.position == layout.charger_position),
                float(target == agent.position),
                max(
                    0.0,
                    agent.battery
                    - (
                        config.move_battery_cost
                        if target != agent.position
                        else 0.0
                    ),
                )
                / 100.0,
                *task_action_features(agent, target),
            )
        )
    for action in ACTIONS:
        target = other_action_targets[action]
        goal_distance = shortest_path_distance(
            target,
            other_goal_position,
            config.map_layout_id,
        )
        values.extend(
            (
                goal_distance / distance_scale,
                shortest_path_distance(
                    target,
                    layout.charger_position,
                    config.map_layout_id,
                )
                / distance_scale,
                float(max(-1, min(1, other_current_goal_distance - goal_distance))),
                float(target == agent.position),
                float(
                    target in available_pickups
                    and target != other_goal_position
                ),
                float(target == layout.charger_position),
                float(other.position == layout.charger_position),
                float(target == other.position),
                max(
                    0.0,
                    other.battery
                    - (
                        config.move_battery_cost
                        if target != other.position
                        else 0.0
                    ),
                )
                / 100.0,
                *task_action_features(other, target),
            )
        )
    for own_action in ACTIONS:
        own_target = action_targets[own_action]
        for teammate_action in ACTIONS:
            teammate_target = other_action_targets[teammate_action]
            values.append(
                float(
                    own_target == teammate_target
                    or (
                        own_target == other.position
                        and teammate_target == agent.position
                    )
                    or (
                        own_target == other.position
                        and teammate_target == other.position
                    )
                    or (
                        teammate_target == agent.position
                        and own_target == agent.position
                    )
                )
            )
    return values


def _canonical_team_context(
    state: WarehouseState,
    config: WarehouseConfig,
) -> list[float]:
    """Encode the complete observable team state in one stable coordinate frame.

    Relative per-robot features remain useful for parameter sharing, but a
    narrow fixed map has discontinuous topology.  This canonical block lets
    both calls to the shared Actor reason from the same robot/task ordering;
    the existing agent-identity one-hot still tells the network which action
    it is producing.  It contains state only and never an expert action or a
    post-policy correction.
    """

    values: list[float] = []
    tasks = sorted(state.tasks, key=lambda item: item.task_id)[
        : config.active_task_count
    ]
    task_ids = [task.task_id for task in tasks]
    for teammate in sorted(state.agents, key=lambda item: item.agent_id):
        teammate_goal_kind, teammate_goal_position = _actor_visible_goal(
            state,
            teammate,
        )
        carry_slot = (
            task_ids.index(teammate.carrying_task_id)
            if teammate.carrying_task_id in task_ids
            else -1
        )
        commitment_slot = (
            task_ids.index(teammate.route_commitment_task_id)
            if teammate.route_commitment_task_id in task_ids
            else -1
        )
        values.extend(
            (
                teammate.position[0] / max(1, config.rows - 1),
                teammate.position[1] / max(1, config.cols - 1),
                teammate.battery / 100.0,
                float(teammate.active),
                float(carry_slot < 0),
                *(
                    1.0 if carry_slot == index else 0.0
                    for index in range(config.active_task_count)
                ),
                float(commitment_slot < 0),
                *(
                    1.0 if commitment_slot == index else 0.0
                    for index in range(config.active_task_count)
                ),
                *(
                    1.0 if teammate_goal_kind == kind else 0.0
                    for kind in NAVIGATION_GOAL_KINDS
                ),
                teammate_goal_position[0] / max(1, config.rows - 1),
                teammate_goal_position[1] / max(1, config.cols - 1),
                shortest_path_distance(
                    teammate.position,
                    teammate_goal_position,
                    config.map_layout_id,
                )
                / float(config.rows * config.cols),
                *(
                    1.0 if teammate.last_action == action else 0.0
                    for action in ACTIONS
                ),
            )
        )
    for task in tasks:
        values.extend(
            (
                float(task.status == "available"),
                float(task.status == "carried"),
                float(task.carrier_agent_id is None),
                float(task.carrier_agent_id == "robot_1"),
                float(task.carrier_agent_id == "robot_2"),
                task.pickup_position[0] / max(1, config.rows - 1),
                task.pickup_position[1] / max(1, config.cols - 1),
                task.delivery_position[0] / max(1, config.rows - 1),
                task.delivery_position[1] / max(1, config.cols - 1),
            )
        )
    values.extend(
        [0.0]
        * ((config.active_task_count - len(tasks)) * 9)
    )
    return values


def local_observation(
    state: WarehouseState,
    agent_id: str,
    config: WarehouseConfig,
) -> np.ndarray:
    agent = state.by_id(agent_id)
    layout = get_map_layout(config.map_layout_id)
    agent_index = int(agent_id.rsplit("_", 1)[1]) - 1
    actor_goal_kind, actor_goal_position = _actor_visible_goal(state, agent)
    values: list[float] = [
        agent.position[0] / max(1, config.rows - 1),
        agent.position[1] / max(1, config.cols - 1),
        agent.battery / 100.0,
        float(agent.active),
        float(agent.carrying_task_id is not None),
        _normalize_delta(layout.charger_position[0] - agent.position[0], config.rows),
        _normalize_delta(layout.charger_position[1] - agent.position[1], config.cols),
        float(agent.position == layout.charger_position),
        min(1.0, state.frame / max(1, config.horizon)),
        float(state.last_robot_collision_event),
        min(1.0, state.ineffective_joint_wait_streak / 8.0),
        float(state.ineffective_joint_wait_streak > 0),
        float(state.ineffective_joint_wait_streak >= 3),
        *(
            1.0 if actor_goal_kind == kind else 0.0
            for kind in NAVIGATION_GOAL_KINDS
        ),
        _normalize_delta(
            actor_goal_position[0] - agent.position[0],
            config.rows,
        ),
        _normalize_delta(
            actor_goal_position[1] - agent.position[1],
            config.cols,
        ),
        shortest_path_distance(
            agent.position,
            actor_goal_position,
            config.map_layout_id,
        )
        / float(config.rows * config.cols),
        *(1.0 if index == agent_index else 0.0 for index in range(config.max_agents)),
        float(state.participant_controlled_agent_id == agent_id),
        float(
            state.participant_controlled_agent_id is not None
            and state.participant_controlled_agent_id != agent_id
        ),
        *(1.0 if agent.last_action == action else 0.0 for action in ACTIONS),
        *_recent_energy_features(state, agent_id, config),
    ]
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    teammate = next(item for item in state.agents if item.agent_id != agent_id)
    for task in tasks[: config.active_task_count]:
        values.extend(
            (
                float(task.status == "available"),
                float(task.status == "carried"),
                float(task.carrier_agent_id is None),
                float(task.carrier_agent_id == agent_id),
                float(
                    task.carrier_agent_id is not None
                    and task.carrier_agent_id != agent_id
                ),
                float(
                    (agent.route_commitment_task_id or agent.goal_id)
                    == task.task_id
                ),
                float(
                    (teammate.route_commitment_task_id or teammate.goal_id)
                    == task.task_id
                ),
                _normalize_delta(
                    task.pickup_position[0] - agent.position[0],
                    config.rows,
                ),
                _normalize_delta(
                    task.pickup_position[1] - agent.position[1],
                    config.cols,
                ),
                _normalize_delta(
                    task.delivery_position[0] - agent.position[0],
                    config.rows,
                ),
                _normalize_delta(
                    task.delivery_position[1] - agent.position[1],
                    config.cols,
                ),
                shortest_path_distance(
                    task.pickup_position,
                    task.delivery_position,
                    config.map_layout_id,
                )
                / float(config.rows * config.cols),
                shortest_path_distance(
                    agent.position,
                    task.pickup_position,
                    config.map_layout_id,
                )
                / float(config.rows * config.cols),
                shortest_path_distance(
                    agent.position,
                    task.delivery_position,
                    config.map_layout_id,
                )
                / float(config.rows * config.cols),
                min(1.0, max(0, state.frame - task.created_frame) / max(1, config.horizon)),
                shortest_path_distance(
                    teammate.position,
                    task.pickup_position,
                    config.map_layout_id,
                )
                / float(config.rows * config.cols),
                shortest_path_distance(
                    teammate.position,
                    task.delivery_position,
                    config.map_layout_id,
                )
                / float(config.rows * config.cols),
                *_route_energy_features(state, agent_id, task, config),
                *_route_energy_features(state, teammate.agent_id, task, config),
            )
        )
    values.extend(
        [0.0]
        * ((config.active_task_count - len(tasks)) * 23)
    )
    others = sorted(
        (item for item in state.agents if item.agent_id != agent_id),
        key=lambda item: item.agent_id,
    )
    for other in others:
        other_goal_kind, other_goal_position = _actor_visible_goal(state, other)
        teammate_goal_distance = shortest_path_distance(
            other.position,
            other_goal_position,
            config.map_layout_id,
        )
        teammate_charger_distance = shortest_path_distance(
            other.position,
            layout.charger_position,
            config.map_layout_id,
        )
        teammate_charger_slack = (
            other.battery
            - teammate_charger_distance * config.move_battery_cost
        )
        values.extend(
            (
                _normalize_delta(other.position[0] - agent.position[0], config.rows),
                _normalize_delta(other.position[1] - agent.position[1], config.cols),
                other.battery / 100.0,
                float(other.active),
                float(other.carrying_task_id is not None),
                *(1.0 if other.last_action == action else 0.0 for action in ACTIONS),
                *(
                    1.0 if other_goal_kind == kind else 0.0
                    for kind in NAVIGATION_GOAL_KINDS
                ),
                _normalize_delta(
                    other_goal_position[0] - other.position[0],
                    config.rows,
                ),
                _normalize_delta(
                    other_goal_position[1] - other.position[1],
                    config.cols,
                ),
                teammate_goal_distance / float(config.rows * config.cols),
                float(other.position == other_goal_position),
                float(other.position == layout.charger_position),
                max(-1.0, min(1.0, teammate_charger_slack / 100.0)),
                float(
                    other_goal_kind == "charge"
                    and teammate_charger_slack <= config.charge_per_wait
                ),
                *_actor_action_mask(state, other, config),
                *_recent_energy_features(state, other.agent_id, config),
            )
        )
    values.extend(
        [0.0]
        * (
            (config.max_agents - 1 - len(others))
            * (19 + len(NAVIGATION_GOAL_KINDS) + 3 * len(ACTIONS))
        )
    )
    values.extend(_canonical_team_context(state, config))
    values.extend(_coordination_features(state, agent_id, config))
    values.extend(_local_patch(state, agent_id, config))
    values.extend(_actor_action_mask(state, agent, config))
    return np.asarray(values, dtype=np.float32)


def all_local_observations(
    state: WarehouseState,
    config: WarehouseConfig,
) -> dict[str, np.ndarray]:
    return {
        agent.agent_id: local_observation(state, agent.agent_id, config)
        for agent in state.agents
    }


def observation_dim(config: WarehouseConfig) -> int:
    own = (
        14
        + len(NAVIGATION_GOAL_KINDS)
        + 5
        + config.max_agents
        + 2 * len(ACTIONS)
        + 6
    )
    tasks = config.active_task_count * 23
    teammate = (
        (config.max_agents - 1)
        * (19 + len(NAVIGATION_GOAL_KINDS) + 3 * len(ACTIONS))
    )
    canonical_team = (
        config.max_agents
        * (
            4
            + 1
            + config.active_task_count
            + 1
            + config.active_task_count
            + len(NAVIGATION_GOAL_KINDS)
            + 3
            + len(ACTIONS)
        )
        + config.active_task_count * 9
    )
    coordination = (
        8
        + 2 * len(ACTIONS)
        + 2
        * (9 + 6 * config.active_task_count)
        * len(ACTIONS)
        + len(ACTIONS) ** 2
    )
    patch = (2 * config.local_patch_radius + 1) ** 2
    return (
        own
        + tasks
        + teammate
        + canonical_team
        + coordination
        + patch
        + len(ACTIONS)
    )


def global_observation(
    state: WarehouseState,
    config: WarehouseConfig,
) -> np.ndarray:
    locals_by_id = all_local_observations(state, config)
    flat: list[float] = []
    for agent_id in ("robot_1", "robot_2"):
        flat.extend(locals_by_id[agent_id].tolist())
    flat.extend(
        (
            state.frame / max(1, config.horizon),
            min(
                1.0,
                state.total_deliveries
                / max(1, config.horizon // config.minimum_task_distance),
            ),
            sum(agent.active for agent in state.agents) / config.max_agents,
            float(
                any(
                    agent.position
                    == get_map_layout(config.map_layout_id).charger_position
                    for agent in state.agents
                )
            ),
            min(1.0, state.robot_collision_events / max(1, config.horizon)),
            sum(task.status == "available" for task in state.tasks)
            / config.active_task_count,
            sum(task.status == "carried" for task in state.tasks)
            / config.active_task_count,
            max(-1.0, min(1.0, state.user_score / 1000.0)),
        )
    )
    return np.asarray(flat, dtype=np.float32)


def global_state_dim(config: WarehouseConfig) -> int:
    return config.max_agents * observation_dim(config) + 8


def observation_schema(config: WarehouseConfig) -> dict[str, Any]:
    return {
        "contract_version": OBSERVATION_CONTRACT_VERSION,
        "local_dim": observation_dim(config),
        "global_dim": global_state_dim(config),
        "robot_count": 2,
        "participant_robot": "robot_1",
        "ai_robot": "robot_2",
        "local_patch_radius": config.local_patch_radius,
        "map_layout_id": config.map_layout_id,
        "task_slots": {
            "count": config.active_task_count,
            "stable_order": "task_id",
            "fields": (
                "available",
                "carried",
                "unowned",
                "owned_by_self",
                "owned_by_teammate",
                "self_route_commitment",
                "teammate_route_commitment",
                "pickup_row_delta",
                "pickup_column_delta",
                "delivery_row_delta",
                "delivery_column_delta",
                "pickup_to_delivery_path_distance",
                "self_to_pickup_path_distance",
                 "self_to_delivery_path_distance",
                 "task_age",
                 "teammate_to_pickup_path_distance",
                 "teammate_to_delivery_path_distance",
                 "self_required_route_energy",
                 "self_route_energy_slack",
                 "self_necessary_charge_waits",
                 "teammate_required_route_energy",
                 "teammate_route_energy_slack",
                 "teammate_necessary_charge_waits",
             ),
         },
        "navigation_goal_fields": (
            "goal_kind_one_hot",
            "goal_row_delta",
            "goal_column_delta",
            "goal_path_distance",
        ),
         "last_robot_collision_event": True,
         "recent_energy_memory_fields": (
             "last_executed_action_one_hot",
             "last_battery_delta",
             "steps_since_charging",
             "charger_wait_streak",
             "charge_mode_active",
             "avoidable_wait_streak",
             "frames_since_charger_departure",
             "departed_charger_within_four_steps",
         ),
        "ineffective_joint_wait_streak_fields": (
            "normalized_to_eight_steps",
            "positive",
            "at_least_three",
        ),
        "corridor_coordination_fields": (
            "self_in_topological_dead_end_legacy_slot",
            "self_in_single_lane",
            "nearest_topological_dead_end_distance_legacy_slot",
            "teammate_in_single_lane",
            "same_corridor_segment",
            "head_on_risk",
            "self_has_priority",
            "teammate_has_priority",
            "candidate_same_target_and_swap_conflict_per_action",
            "self_candidate_goal_charger_progress_and_claim_features",
            "teammate_candidate_goal_charger_progress_and_claim_features",
            "joint_action_collision_matrix",
        ),
        "teammate_fields": (
            "relative_row",
            "relative_column",
            "battery",
            "active",
            "carrying_shared_task",
            "previous_action_one_hot",
            "navigation_goal_kind_one_hot",
            "navigation_goal_row_delta_from_teammate",
            "navigation_goal_column_delta_from_teammate",
            "navigation_goal_path_distance",
            "at_navigation_goal",
            "at_charger",
            "charger_energy_slack",
            "urgent_charge_route",
            "static_action_mask",
        ),
        "canonical_team_context": {
            "robot_order": ("robot_1", "robot_2"),
            "task_order": "task_id",
            "contains_expert_or_program_action": False,
        },
        "agent_identity": {"type": "one_hot", "size": 2},
        "control_provenance": {
            "fields": ("self_is_participant", "teammate_is_participant"),
            "known_before_episode": True,
            "contains_current_action": False,
        },
        "legal_action_mask": {
            "offset_from_end": len(ACTIONS),
            "actions": ACTIONS,
        },
    }
