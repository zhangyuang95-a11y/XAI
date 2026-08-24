"""Training-only individual credit assignment for warehouse transitions.

The environment owns state transitions and the participant-facing score.  This
module deliberately owns only dense training credit, so offline teacher code
does not become a runtime dependency of the environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping

from .domain import AgentState, DeliveryTask, WarehouseState
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance


@dataclass(frozen=True)
class FrozenMission:
    """One transition-local mission used for stable efficiency credit."""

    goal_kind: str
    goal_position: tuple[int, int]
    task: DeliveryTask | None = None


def frozen_training_missions(
    environment: Any,
    state: WarehouseState,
) -> dict[str, FrozenMission | None]:
    """Match each robot once at transition start and freeze that mission."""

    assignments = environment._frozen_task_assignments(
        state,
        prioritize_old_tasks=True,
    )
    available = {
        task.task_id: task
        for task in state.tasks
        if task.status == "available"
    }
    reserved_task_ids = {task.task_id for task in assignments.values()}
    fallback_task_ids: set[str] = set()
    missions: dict[str, FrozenMission | None] = {}
    for agent in state.agents:
        if not agent.active:
            missions[agent.agent_id] = None
            continue
        if agent.carrying_task_id is not None:
            task = state.task_by_id(agent.carrying_task_id)
        else:
            task = assignments.get(agent.agent_id)
            if task is None and agent.route_commitment_task_id in available:
                committed = available[agent.route_commitment_task_id]
                assigned_elsewhere = {
                    item.task_id
                    for other_id, item in assignments.items()
                    if other_id != agent.agent_id
                }
                if committed.task_id not in assigned_elsewhere:
                    task = committed
            if task is None and environment._requires_charge(state, agent):
                task = min(
                    (
                        item
                        for item in available.values()
                        if item.task_id not in reserved_task_ids
                        and item.task_id not in fallback_task_ids
                    ),
                    key=lambda item: (
                        environment._safe_task_cost(state, agent, item),
                        item.task_id,
                    ),
                    default=None,
                )
                if task is None:
                    task = min(
                        available.values(),
                        key=lambda item: (
                            environment._safe_task_cost(state, agent, item),
                            item.task_id,
                        ),
                        default=None,
                    )
                if task is not None:
                    fallback_task_ids.add(task.task_id)
        if task is None:
            missions[agent.agent_id] = None
            continue
        if environment._requires_charge(state, agent):
            missions[agent.agent_id] = FrozenMission(
                "charge",
                environment.layout.charger_position,
                task,
            )
        elif agent.carrying_task_id is not None:
            missions[agent.agent_id] = FrozenMission(
                "delivery",
                task.delivery_position,
                task,
            )
        else:
            missions[agent.agent_id] = FrozenMission(
                "pickup",
                task.pickup_position,
                task,
            )
    return missions


def frozen_mission_cost(
    environment: Any,
    state: WarehouseState,
    agent_id: str,
    mission: FrozenMission | None,
) -> float | None:
    """Return safe actions remaining for a transition-frozen mission."""

    if mission is None:
        return None
    agent = state.by_id(agent_id)
    if not agent.active:
        return None
    task = mission.task
    if task is None:
        return float(
            shortest_path_distance(
                agent.position,
                mission.goal_position,
                environment.config.map_layout_id,
            )
        )
    live_task = next(
        (
            item
            for item in (*state.tasks, *state.completed_tasks)
            if item.task_id == task.task_id
        ),
        None,
    )
    if live_task is not None and live_task.status == "delivered":
        return 0.0 if live_task.carrier_agent_id == agent_id else None
    if (
        live_task is not None
        and live_task.status == "carried"
        and live_task.carrier_agent_id != agent_id
    ):
        return None
    return environment._safe_task_cost(
        state,
        agent,
        live_task if live_task is not None else task,
        position=agent.position,
    )


def mission_goal_distance(
    environment: Any,
    state: WarehouseState,
    agent: AgentState,
    mission: FrozenMission,
    position: tuple[int, int],
) -> float:
    """Distance for one-step counterfactuals without crossing another A."""

    if (
        mission.goal_kind == "pickup"
        and mission.task is not None
        and mission.task.status == "available"
    ):
        return environment._claim_safe_pickup_distance(
            state,
            agent,
            mission.task,
            position=position,
        )
    return float(
        shortest_path_distance(
            position,
            mission.goal_position,
            environment.config.map_layout_id,
        )
    )


def _urgent_charge(environment: Any, agent: AgentState) -> bool:
    if agent.navigation_goal_kind != "charge":
        return False
    distance = shortest_path_distance(
        agent.position,
        agent.navigation_goal_position,
        environment.config.map_layout_id,
    )
    slack = agent.battery - distance * environment.config.move_battery_cost
    return bool(slack <= environment.config.charge_per_wait)


def _shortest_progress_positions(
    environment: Any,
    agent: AgentState,
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    current = shortest_path_distance(
        agent.position,
        goal,
        environment.config.map_layout_id,
    )
    return tuple(
        sorted(
            candidate
            for delta in MOVE_DELTAS.values()
            if environment.layout.is_passable(
                candidate := (
                    agent.position[0] + delta[0],
                    agent.position[1] + delta[1],
                )
            )
            and shortest_path_distance(
                candidate,
                goal,
                environment.config.map_layout_id,
            )
            < current
        )
    )


def necessary_urgent_charger_clearance(
    environment: Any,
    state: WarehouseState,
    clearing_agent: AgentState,
) -> bool:
    """Whether a detour is required to unblock an urgent charger handoff."""

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


def counterfactual_action_regrets(
    environment: Any,
    state: WarehouseState,
    requested_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
    actual_targets: Mapping[str, tuple[int, int]],
    missions: Mapping[str, FrozenMission | None],
    coordination_events: tuple[dict[str, Any], ...],
) -> tuple[
    dict[str, float],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, float],
]:
    """Compute per-robot one-step regret with the teammate action fixed."""

    exempt_agents: set[str] = set()
    for event in coordination_events:
        kind = str(event.get("event", ""))
        if kind == "coordination_yield":
            yielding_id = str(event.get("yielding_agent_id", ""))
            passing_id = str(event.get("passing_agent_id", ""))
            if (
                str(requested_actions.get(yielding_id, "WAIT")) != "WAIT"
                or str(requested_actions.get(passing_id, "WAIT"))
                in MOVE_DELTAS
            ):
                exempt_agents.add(yielding_id)
        elif kind == "charger_queue":
            exempt_agents.add(str(event.get("waiting_agent_id", "")))

    regrets = {agent.agent_id: 0.0 for agent in state.agents}
    best_distances: dict[str, float] = {}
    avoidable_waits: list[str] = []
    detours: list[str] = []
    loaded_detours: list[str] = []
    for agent in state.agents:
        mission = missions.get(agent.agent_id)
        if not agent.active or mission is None:
            continue
        current_distance = mission_goal_distance(
            environment, state, agent, mission, agent.position
        )
        chosen_distance = mission_goal_distance(
            environment,
            state,
            agent,
            mission,
            actual_targets[agent.agent_id],
        )
        candidate_distances: list[float] = []
        for candidate_action in ACTIONS:
            trial = dict(requested_actions)
            trial[agent.agent_id] = candidate_action
            targets, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if agent.agent_id in invalid or collision:
                continue
            candidate_distances.append(
                mission_goal_distance(
                    environment,
                    state,
                    agent,
                    mission,
                    targets[agent.agent_id],
                )
            )
        if not candidate_distances:
            best_distances[agent.agent_id] = chosen_distance
            continue
        best_distance = min(candidate_distances)
        best_distances[agent.agent_id] = best_distance
        action = str(requested_actions.get(agent.agent_id, "WAIT"))
        exempt = agent.agent_id in exempt_agents
        if (
            action == "WAIT"
            and agent.position == environment.layout.charger_position
            and environment._requires_charge(state, agent)
        ):
            exempt = True
        if (
            not exempt
            and action in MOVE_DELTAS
            and agent.position == environment.layout.charger_position
            and any(
                teammate.agent_id != agent.agent_id
                and environment._requires_charge(state, teammate)
                and shortest_path_distance(
                    teammate.position,
                    environment.layout.charger_position,
                    environment.config.map_layout_id,
                )
                <= 6
                for teammate in state.agents
            )
        ):
            exempt = True
        if (
            not exempt
            and action in MOVE_DELTAS
            and agent.carrying_task_id is not None
        ):
            exempt = necessary_urgent_charger_clearance(
                environment,
                state,
                agent,
            )
        if not exempt and chosen_distance > best_distance:
            regrets[agent.agent_id] = min(
                2.0,
                max(0.0, float(chosen_distance - best_distance)),
            )
        executed = str(executed_actions.get(agent.agent_id, "WAIT"))
        if action == "WAIT" and executed == "WAIT" and regrets[agent.agent_id] > 0:
            avoidable_waits.append(agent.agent_id)
        if action in MOVE_DELTAS and regrets[agent.agent_id] > 0:
            detours.append(agent.agent_id)
            if (
                agent.carrying_task_id is not None
                and mission.goal_kind == "delivery"
                and chosen_distance > current_distance
            ):
                loaded_detours.append(agent.agent_id)
    return (
        regrets,
        tuple(sorted(avoidable_waits)),
        tuple(sorted(detours)),
        tuple(sorted(loaded_detours)),
        best_distances,
    )


def safe_delivery_completion_steps(
    environment: Any,
    agent: AgentState,
    task: DeliveryTask,
) -> float:
    """Shortest safe post-claim work for the path-efficiency audit."""

    delivery_distance = shortest_path_distance(
        agent.position,
        task.delivery_position,
        environment.config.map_layout_id,
    )
    delivery_to_charger = shortest_path_distance(
        task.delivery_position,
        environment.layout.charger_position,
        environment.config.map_layout_id,
    )
    required_energy = (
        delivery_distance
        + delivery_to_charger
        + environment.config.mission_reserve_steps
    ) * environment.config.move_battery_cost
    if agent.battery >= required_energy:
        return float(delivery_distance)
    charger_distance = shortest_path_distance(
        agent.position,
        environment.layout.charger_position,
        environment.config.map_layout_id,
    )
    battery_at_charger = max(
        0.0,
        agent.battery - charger_distance * environment.config.move_battery_cost,
    )
    charger_to_delivery = shortest_path_distance(
        environment.layout.charger_position,
        task.delivery_position,
        environment.config.map_layout_id,
    )
    charged_required = (
        charger_to_delivery
        + delivery_to_charger
        + environment.config.mission_reserve_steps
    ) * environment.config.move_battery_cost
    waits = math.ceil(
        max(0.0, charged_required - battery_at_charger)
        / environment.config.charge_per_wait
    )
    return float(charger_distance + waits + charger_to_delivery)


def individual_credit_components(
    environment: Any,
    *,
    terminated: bool,
    score_delta: float,
    next_state: WarehouseState,
    mission_costs_before: Mapping[str, float | None],
    mission_costs_after: Mapping[str, float | None],
    coordination_cost_before: float,
    counterfactual_regret_units: Mapping[str, float],
    avoidable_wait_agents: tuple[str, ...],
) -> dict[str, Any]:
    """Build all reward components without touching the user score."""

    config = environment.config.reward
    progress_units = {
        agent.agent_id: (
            0.0
            if terminated
            or mission_costs_before[agent.agent_id] is None
            or mission_costs_after[agent.agent_id] is None
            else float(
                mission_costs_before[agent.agent_id]
                - mission_costs_after[agent.agent_id]
            )
        )
        for agent in next_state.agents
    }
    progress_rewards = {
        agent_id: config.progress_scale * units / 100.0
        for agent_id, units in progress_units.items()
    }
    coordination_cost_after = environment._coordination_delay_cost(
        next_state,
        {agent.agent_id: agent.position for agent in next_state.agents},
    )
    raw_coordination_reward = (
        0.0
        if terminated
        else (coordination_cost_before - coordination_cost_after) / 100.0
    )
    coordination_reward = max(
        -config.coordination_progress_cap,
        min(config.coordination_progress_cap, raw_coordination_reward),
    )
    regret_penalties = {
        agent.agent_id: (
            0.0
            if terminated
            else -config.counterfactual_regret_cost
            * counterfactual_regret_units[agent.agent_id]
        )
        for agent in next_state.agents
    }
    repeated_wait_penalties = {}
    flat_wait_penalties = {}
    for agent in next_state.agents:
        streak_units = min(
            max(0, agent.avoidable_wait_streak - 1),
            config.avoidable_wait_streak_cap,
        )
        repeated_wait_penalties[agent.agent_id] = (
            0.0
            if terminated
            else -config.avoidable_wait_streak_cost * streak_units
        )
        flat_wait_penalties[agent.agent_id] = (
            -config.avoidable_wait_cost
            if not terminated and agent.agent_id in avoidable_wait_agents
            else 0.0
        )
    base_reward = score_delta / 100.0
    rewards = {
        agent.agent_id: (
            base_reward
            + progress_rewards[agent.agent_id]
            + coordination_reward
            + regret_penalties[agent.agent_id]
            + repeated_wait_penalties[agent.agent_id]
            + flat_wait_penalties[agent.agent_id]
        )
        for agent in next_state.agents
    }
    return {
        "base_training_reward": base_reward,
        "rewards": rewards,
        "training_reward": sum(rewards.values()) / len(rewards),
        "individual_progress_units": progress_units,
        "individual_progress_rewards": progress_rewards,
        "coordination_cost_after": coordination_cost_after,
        "coordination_progress_reward": coordination_reward,
        "counterfactual_regret_penalty_rewards": regret_penalties,
        "repeated_avoidable_wait_penalty_rewards": repeated_wait_penalties,
        "flat_avoidable_wait_penalty_rewards": flat_wait_penalties,
    }


def transition_credit_components(
    environment: Any,
    *,
    terminated: bool,
    score_delta: float,
    previous_state: WarehouseState,
    next_state: WarehouseState,
    requested_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
    mission_costs_before: Mapping[str, float | None],
    mission_costs_after: Mapping[str, float | None],
    coordination_cost_before: float,
    counterfactual_regret_units: Mapping[str, float],
    avoidable_wait_agents: tuple[str, ...],
    assignment_potential_before: float,
    assignment_potential_after: float,
) -> dict[str, Any]:
    """Select production individual credit or the isolated legacy ablation."""

    config = environment.config.reward
    if config.individual_credit_enabled:
        result = individual_credit_components(
            environment,
            terminated=terminated,
            score_delta=score_delta,
            next_state=next_state,
            mission_costs_before=mission_costs_before,
            mission_costs_after=mission_costs_after,
            coordination_cost_before=coordination_cost_before,
            counterfactual_regret_units=counterfactual_regret_units,
            avoidable_wait_agents=avoidable_wait_agents,
        )
        result["potential_shaping_reward"] = (
            sum(result["individual_progress_rewards"].values())
            / len(result["rewards"])
            + result["coordination_progress_reward"]
        )
        result["avoidable_wait_penalty_reward"] = sum(
            result["repeated_avoidable_wait_penalty_rewards"][agent.agent_id]
            + result["flat_avoidable_wait_penalty_rewards"][agent.agent_id]
            for agent in next_state.agents
        ) / len(result["rewards"])
        result["mission_regression_units"] = 0.0
        result["mission_regression_penalty_reward"] = 0.0
        return result

    legacy_wait_agents = environment._avoidable_wait_agents(
        previous_state,
        requested_actions,
        executed_actions,
    )
    shaping = (
        0.0
        if terminated
        else config.progress_scale
        * (assignment_potential_before - assignment_potential_after)
        / 100.0
    )
    wait_penalty = (
        0.0 if terminated else config.avoidable_wait_cost * len(legacy_wait_agents)
    )
    regression_units = (
        0.0
        if terminated
        else max(0.0, assignment_potential_after - assignment_potential_before)
    )
    regression_penalty = (
        config.mission_regression_scale * regression_units / 100.0
    )
    base_reward = score_delta / 100.0
    team_reward = base_reward + shaping - wait_penalty - regression_penalty
    agent_ids = tuple(agent.agent_id for agent in next_state.agents)
    zeros = {agent_id: 0.0 for agent_id in agent_ids}
    return {
        "base_training_reward": base_reward,
        "rewards": {agent_id: team_reward for agent_id in agent_ids},
        "training_reward": team_reward,
        "individual_progress_units": {agent_id: 0.0 for agent_id in agent_ids},
        "individual_progress_rewards": {
            agent_id: shaping for agent_id in agent_ids
        },
        "coordination_cost_after": environment._coordination_delay_cost(
            next_state,
            {agent.agent_id: agent.position for agent in next_state.agents},
        ),
        "coordination_progress_reward": 0.0,
        "counterfactual_regret_penalty_rewards": dict(zeros),
        "repeated_avoidable_wait_penalty_rewards": dict(zeros),
        "flat_avoidable_wait_penalty_rewards": {
            agent_id: (
                -config.avoidable_wait_cost
                if agent_id in legacy_wait_agents and not terminated
                else 0.0
            )
            for agent_id in agent_ids
        },
        "potential_shaping_reward": shaping,
        "avoidable_wait_penalty_reward": -wait_penalty,
        "mission_regression_units": regression_units,
        "mission_regression_penalty_reward": -regression_penalty,
    }


def serialized_frozen_missions(
    missions: Mapping[str, FrozenMission | None],
) -> dict[str, dict[str, Any] | None]:
    return {
        agent_id: (
            {
                "goal_kind": mission.goal_kind,
                "goal_position": mission.goal_position,
                "task_id": mission.task.task_id if mission.task is not None else None,
            }
            if mission is not None
            else None
        )
        for agent_id, mission in missions.items()
    }


def completed_delivery_path_metrics(state: WarehouseState) -> dict[str, Any]:
    actual = {
        task.task_id: float(task.delivered_frame - task.claimed_frame)
        for task in state.completed_tasks
        if task.delivered_frame is not None and task.claimed_frame is not None
    }
    shortest = {
        task.task_id: float(task.shortest_safe_delivery_steps)
        for task in state.completed_tasks
        if task.shortest_safe_delivery_steps is not None
    }
    eligible_actual = sum(
        actual[task_id] for task_id in shortest if task_id in actual
    )
    return {
        "completed_delivery_actual_steps": actual,
        "completed_delivery_shortest_safe_steps": shortest,
        "path_efficiency_actual_over_shortest_safe": (
            eligible_actual / max(1.0, sum(shortest.values()))
        ),
    }


def measured_head_on_clearance_delay(
    environment: Any,
    yielding_position: tuple[int, int],
    priority_position: tuple[int, int],
    *,
    same_row: bool,
    clearance_cap: float,
) -> float:
    """Measure the smallest move-out/move-back clearance on the real map."""

    if not environment.config.reward.individual_credit_enabled:
        return float(clearance_cap)
    queue = deque(((yielding_position, 0),))
    visited = {yielding_position}
    while queue:
        candidate, distance = queue.popleft()
        if distance > 0:
            cleared = (
                candidate[0] != priority_position[0]
                if same_row
                else candidate[1] != priority_position[1]
            )
            if cleared:
                return min(clearance_cap, float(2 * distance))
        if distance >= max(1, int(math.ceil(clearance_cap / 2.0))):
            continue
        for delta in MOVE_DELTAS.values():
            neighbor = (
                candidate[0] + delta[0],
                candidate[1] + delta[1],
            )
            if (
                neighbor in visited
                or neighbor == priority_position
                or not environment.layout.is_passable(neighbor)
            ):
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return float(clearance_cap)


def measured_charger_clearance_delay(
    environment: Any,
    clearance_cap: float,
) -> float:
    return (
        min(clearance_cap, 1.0)
        if environment.config.reward.individual_credit_enabled
        else float(clearance_cap)
    )
