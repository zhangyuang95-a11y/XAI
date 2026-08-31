"""Two-robot collaborative delivery environment used by the user study."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from itertools import product
import math
import random
from typing import Any, Mapping

from . import credit_assignment as credit
from .contracts import ENVIRONMENT_VERSION
from .coordination_plan import coordination_plan_execution_event
from .domain import AgentState, DeliveryTask, WarehouseConfig, WarehouseState
from .energy_management import (charge_release_energy, charger_departure_progress,
    charger_queue_clearance_delay, charger_reentry_event)
from .goal_management import (
    advance_coordination_plan, assign_persistent_pickup_goals,
    frozen_coordination_plan as derive_frozen_coordination_plan,
    prepare_coordination_plan, refresh_navigation_goals,
    synchronize_persistent_goals, update_route_commitments,
)
from .temporal_audit import transition_temporal_violations
from .layouts import get_map_layout
from .navigation import (
    ACTIONS,
    CHARGER_POSITION,
    COLS,
    MOVE_DELTAS,
    ROWS,
    SHELF_COLUMNS,
    SHELF_POSITIONS,
    SHELF_ROWS,
    WAITING_POSITIONS,
    WAITING_ZONE,
    all_passable_positions,
    assigned_waiting_positions,
    in_bounds,
    is_passable,
    is_shelf,
    legal_action_mask,
    pickup_pairs,
    shortest_path_distance,
)
from .transition_audit import (
    action_is_robustly_safe,
    build_decision_trace,
    environment_info,
    joint_decision_audit,
    necessary_participant_standoff_clearance,
)
from .state_support import render_ascii_state, validate_warehouse_state
from .route_goals import frozen_route_goal


class WarehouseMultiAgentEnv:
    """Deterministic-seeded cooperative Markov game for two moving robots."""

    environment_name = ENVIRONMENT_VERSION

    def __init__(self, config: WarehouseConfig | None = None) -> None:
        self.config = config or WarehouseConfig()
        self.layout = get_map_layout(self.config.map_layout_id)
        self._rng = random.Random(self.config.seed)
        self._episode_counter = 0
        self.state: WarehouseState | None = None

    @property
    def agent_ids(self) -> tuple[str, str]:
        return ("robot_1", "robot_2")

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self._episode_counter += 1
        excluded = {
            self.layout.charger_position,
            *self.layout.robot_start_positions,
            *self.layout.task_endpoint_exclusions,
        }
        tasks: list[DeliveryTask] = []
        next_task_index = 1
        for _ in range(self.config.active_task_count):
            task = self._sample_delivery_job(
                task_index=next_task_index,
                created_frame=0,
                excluded_positions=excluded,
            )
            tasks.append(task)
            excluded.update((task.pickup_position, task.delivery_position))
            next_task_index += 1
        agents = [
            AgentState(
                agent_id=agent_id,
                position=self.layout.robot_start_positions[index],
                heading=self._rng.choice(tuple(MOVE_DELTAS)),
            )
            for index, agent_id in enumerate(self.agent_ids)
        ]
        self.state = WarehouseState(
            episode_id=self._episode_counter,
            frame=0,
            agents=agents,
            tasks=tasks,
            next_task_index=next_task_index,
        )
        self._refresh_navigation_goals(self.state)
        assign_persistent_pickup_goals(self, self.state)
        # Task reservations change which remaining missions are energy-safe
        # for the teammate. Recompute physical mode from that same frozen
        # assignment before publishing the observation; otherwise an
        # unassigned robot can take one task-directed step and discover only
        # on the next frame that it actually needed to charge.
        self._refresh_navigation_goals(self.state)
        synchronize_persistent_goals(
            self,
            None,
            self.state,
            reset_reason="episode_reset",
        )
        initial_plan = prepare_coordination_plan(self, self.state)
        if initial_plan is not None:
            synchronize_persistent_goals(
                self,
                None,
                self.state,
                reset_reason="episode_reset",
                coordination_plan=initial_plan,
            )
        return self.observations(), environment_info(
            self,
            reward_breakdown=None,
            collisions=(),
            shutdowns=(),
        )

    def _sample_delivery_job(
        self,
        *,
        task_index: int,
        created_frame: int,
        excluded_positions: set[tuple[int, int]],
    ) -> DeliveryTask:
        pickup_candidates = sorted(
            {
                access
                for _, access in pickup_pairs(self.config.map_layout_id)
                if access not in excluded_positions
                and access not in self.layout.pickup_endpoint_exclusions
                # Entering A claims automatically.  A transit-aisle pickup can
                # therefore be stolen unavoidably by a robot returning from B
                # or travelling to the charger.  Shelf-aisle dead ends remain
                # ordinary A points but are never on another mission's route.
                and access in self.layout.dead_end_positions
            }
        )
        delivery_candidates = [
            position
            for position in all_passable_positions(self.config.map_layout_id)
            if position not in excluded_positions
            and position
            not in {
                self.layout.charger_position,
                *self.layout.robot_start_positions,
                *self.layout.dead_end_positions,
            }
        ]
        self._rng.shuffle(pickup_candidates)
        self._rng.shuffle(delivery_candidates)
        for pickup in pickup_candidates:
            for delivery in delivery_candidates:
                if delivery == pickup:
                    continue
                if (
                    shortest_path_distance(
                        pickup,
                        delivery,
                        self.config.map_layout_id,
                    )
                    < self.config.minimum_task_distance
                ):
                    continue
                return DeliveryTask(
                    task_id=f"task_{task_index}",
                    pickup_position=pickup,
                    delivery_position=delivery,
                    created_frame=created_frame,
                )
        raise RuntimeError("Could not sample a valid shared A-to-B delivery task.")

    def _mission_route_steps(
        self,
        state: WarehouseState,
        agent: AgentState,
        task: DeliveryTask,
        *,
        origin: tuple[int, int],
    ) -> float:
        """Movement plus reserve needed to deliver and retain a charger route."""

        if agent.carrying_task_id:
            mission_steps = shortest_path_distance(
                origin, task.delivery_position, self.config.map_layout_id
            )
        else:
            mission_steps = shortest_path_distance(
                origin, task.pickup_position, self.config.map_layout_id
            )
            mission_steps += shortest_path_distance(
                task.pickup_position,
                task.delivery_position,
                self.config.map_layout_id,
            )
        return float(
            mission_steps
            + shortest_path_distance(
                task.delivery_position,
                self.layout.charger_position,
                self.config.map_layout_id,
            )
            + self.config.mission_reserve_steps
        )

    def _safe_task_cost(
        self,
        state: WarehouseState,
        agent: AgentState,
        task: DeliveryTask,
        *,
        position: tuple[int, int] | None = None,
    ) -> float:
        """Estimated safe actions, including only the charging waits still needed.

        The direct and charge-first branches meet at exactly the same cost when
        the final necessary charging wait is completed.  This prevents a goal
        switch at the charger from creating an artificial shaping penalty.
        """

        origin = position or agent.position
        direct_steps = self._mission_route_steps(
            state,
            agent,
            task,
            origin=origin,
        )
        direct_energy = direct_steps * self.config.move_battery_cost
        if agent.battery >= direct_energy:
            return direct_steps

        charger_distance = shortest_path_distance(
            origin,
            self.layout.charger_position,
            self.config.map_layout_id,
        )
        battery_at_charger = max(
            0.0,
            agent.battery - charger_distance * self.config.move_battery_cost,
        )
        charged_route_steps = self._mission_route_steps(
            state,
            agent,
            task,
            origin=self.layout.charger_position,
        )
        charged_route_energy = charged_route_steps * self.config.move_battery_cost
        energy_deficit = max(0.0, charged_route_energy - battery_at_charger)
        necessary_waits = math.ceil(energy_deficit / self.config.charge_per_wait)
        return float(charger_distance + necessary_waits + charged_route_steps)

    def _priority_safe_task_cost(
        self,
        state: WarehouseState,
        agent: AgentState,
        task: DeliveryTask,
        *,
        position: tuple[int, int] | None = None,
        task_age_frame: int | None = None,
    ) -> float:
        """Training-only safe work with bounded urgency for old shared tasks."""

        cost = self._safe_task_cost(state, agent, task, position=position)
        if task.status != "available":
            return cost
        age_frame = state.frame if task_age_frame is None else task_age_frame
        age_fraction = min(
            1.0,
            max(0, age_frame - task.created_frame)
            / self.config.reward.task_age_priority_horizon,
        )
        return float(
            cost * (1.0 + self.config.reward.task_age_priority_scale * age_fraction)
        )

    def _requires_charge(
        self,
        state: WarehouseState,
        agent: AgentState,
        *,
        position: tuple[int, int] | None = None,
    ) -> bool:
        """Return whether every available safe mission requires charging first."""

        origin = position or agent.position
        if agent.carrying_task_id:
            tasks = [state.task_by_id(agent.carrying_task_id)]
        elif agent.route_commitment_task_id is not None:
            committed = next(
                (
                    task
                    for task in state.tasks
                    if task.task_id == agent.route_commitment_task_id
                    and task.status == "available"
                ),
                None,
            )
            tasks = (
                [committed]
                if committed is not None
                else [
                    task for task in state.tasks if task.status == "available"
                ]
            )
        elif agent.goal_type == "GO_TO_PICKUP" and agent.goal_id is not None:
            selected = next(
                (
                    task
                    for task in state.tasks
                    if task.task_id == agent.goal_id
                    and task.status == "available"
                ),
                None,
            )
            tasks = (
                [selected]
                if selected is not None
                else [
                    task for task in state.tasks if task.status == "available"
                ]
            )
        else:
            available_tasks = [
                task for task in state.tasks if task.status == "available"
            ]
            teammate_reservations = {
                teammate.route_commitment_task_id or teammate.goal_id
                for teammate in state.agents
                if teammate.agent_id != agent.agent_id
                and teammate.carrying_task_id is None
            }
            unreserved = [
                task
                for task in available_tasks
                if task.task_id not in teammate_reservations
            ]
            tasks = unreserved or available_tasks
        if not tasks:
            required = (
                shortest_path_distance(
                    origin,
                    self.layout.charger_position,
                    self.config.map_layout_id,
                )
                + self.config.mission_reserve_steps
            ) * self.config.move_battery_cost
            return agent.battery < required
        return all(
            agent.battery
            < self._mission_route_steps(state, agent, task, origin=origin)
            * self.config.move_battery_cost
            for task in tasks
        )

    def _task_is_directly_energy_safe(
        self,
        state: WarehouseState,
        agent: AgentState,
        task: DeliveryTask,
        *,
        position: tuple[int, int] | None = None,
    ) -> bool:
        """Whether ``agent`` can finish ``task`` and retain the safety reserve.

        This is deliberately the same direct-route test used by
        :meth:`_requires_charge`.  Matching a robot to a task that fails this
        test was the source of the leave-charger/return-one-step-later cycle:
        the old matcher established that *some* task was safe, then assigned a
        different, unsafe task because it lowered the team distance.
        """

        origin = agent.position if position is None else position
        required_energy = (
            self._mission_route_steps(state, agent, task, origin=origin)
            * self.config.move_battery_cost
        )
        return bool(agent.battery >= required_energy)

    def _claim_safe_pickup_distance(
        self,
        state: WarehouseState,
        agent: AgentState,
        task: DeliveryTask,
        *,
        position: tuple[int, int] | None = None,
    ) -> float:
        """Distance to A without crossing a different unclaimed A point."""

        origin = agent.position if position is None else position
        forbidden = {
            item.pickup_position
            for item in state.tasks
            if item.status == "available" and item.task_id != task.task_id
        }
        queue = deque(((origin, 0),))
        visited = {origin}
        pickup_distance: int | None = None
        while queue:
            current, distance = queue.popleft()
            if current == task.pickup_position:
                pickup_distance = distance
                break
            for delta in MOVE_DELTAS.values():
                candidate = (
                    current[0] + delta[0],
                    current[1] + delta[1],
                )
                if (
                    candidate in visited
                    or candidate in forbidden
                    or not self.layout.is_passable(candidate)
                ):
                    continue
                visited.add(candidate)
                queue.append((candidate, distance + 1))
        if pickup_distance is None:
            return float("inf")
        return float(pickup_distance)

    def _claim_safe_assignment_route_steps(
        self,
        state: WarehouseState,
        agent: AgentState,
        task: DeliveryTask,
        *,
        position: tuple[int, int] | None = None,
    ) -> float:
        """Mission steps without crossing a different unclaimed A point."""

        origin = agent.position if position is None else position
        if agent.carrying_task_id is not None:
            return self._mission_route_steps(
                state,
                agent,
                task,
                origin=origin,
            )
        pickup_distance = self._claim_safe_pickup_distance(
            state,
            agent,
            task,
            position=origin,
        )
        if pickup_distance == float("inf"):
            return float("inf")
        return float(
            pickup_distance
            + shortest_path_distance(
                task.pickup_position,
                task.delivery_position,
                self.config.map_layout_id,
            )
            + shortest_path_distance(
                task.delivery_position,
                self.layout.charger_position,
                self.config.map_layout_id,
            )
            + self.config.mission_reserve_steps
        )

    def _frozen_task_assignments(
        self,
        state: WarehouseState,
        *,
        prioritize_old_tasks: bool = False,
    ) -> dict[str, DeliveryTask]:
        """Return one atomic, energy-feasible shared-task matching.

        The result is a transition-local planning snapshot, not environment
        ownership: tasks remain unassigned until a robot physically reaches A.
        Every enumerated pair is energy-safe, so matching and the charge
        decision cannot disagree.  The offline teacher may also take an old
        task first and then recharge while carrying it, provided A and the
        charger remain safely reachable.  This prevents a long shared job from
        being skipped forever merely because its full A-to-B route is long.
        """

        agents = tuple(
            sorted(
                (
                    agent
                    for agent in state.agents
                    if agent.active and agent.carrying_task_id is None
                ),
                key=lambda agent: agent.agent_id,
            )
        )
        tasks = tuple(
            sorted(
                (task for task in state.tasks if task.status == "available"),
                key=lambda task: task.task_id,
            )
        )
        if not agents or not tasks:
            return {}

        safe_pairs: set[tuple[str, str]] = set()
        for agent in agents:
            for task in tasks:
                route_steps = self._claim_safe_assignment_route_steps(
                    state,
                    agent,
                    task,
                )
                pickup_distance = self._claim_safe_pickup_distance(
                    state,
                    agent,
                    task,
                )
                if route_steps == float("inf") or pickup_distance == float("inf"):
                    continue
                departure_reserve = (
                    2.0 * self.config.move_battery_cost
                    if prioritize_old_tasks
                    and agent.position == self.layout.charger_position
                    else 0.0
                )
                claim_commitment_reserve = (
                    self.config.charge_per_wait
                    if prioritize_old_tasks
                    and agent.position == self.layout.charger_position
                    else 0.0
                )
                full_mission_energy = (
                    route_steps * self.config.move_battery_cost
                    + departure_reserve
                )
                claim_then_charge_energy = (
                    (
                        pickup_distance
                        + shortest_path_distance(
                            task.pickup_position,
                            self.layout.charger_position,
                            self.config.map_layout_id,
                        )
                        + self.config.mission_reserve_steps
                    )
                    * self.config.move_battery_cost
                    + claim_commitment_reserve
                )
                if agent.battery >= full_mission_energy or (
                    prioritize_old_tasks
                    and agent.battery >= claim_then_charge_energy
                ):
                    safe_pairs.add((agent.agent_id, task.task_id))
        candidates: list[tuple[tuple[Any, ...], dict[str, DeliveryTask]]] = []
        for choices in product((None, *tasks), repeat=len(agents)):
            selected = tuple(task for task in choices if task is not None)
            if len({task.task_id for task in selected}) != len(selected):
                continue
            if any(
                task is not None
                and (agent.agent_id, task.task_id) not in safe_pairs
                for agent, task in zip(agents, choices)
            ):
                continue
            assignments = {
                agent.agent_id: task
                for agent, task in zip(agents, choices)
                if task is not None
            }
            assigned_tasks = tuple(assignments.values())
            assigned_count = len(assigned_tasks)
            ages = tuple(
                max(0, state.frame - task.created_frame)
                for task in assigned_tasks
            )
            overdue_count = sum(
                age >= self.config.reward.task_age_priority_horizon
                for age in ages
            )
            projected_starvation = sum(
                max(
                    0.0,
                    max(0, state.frame - task.created_frame)
                    + self._claim_safe_pickup_distance(
                        state,
                        agent,
                        task,
                    )
                    - self.config.reward.task_age_priority_horizon,
                )
                for agent in agents
                if (task := assignments.get(agent.agent_id)) is not None
            )
            recent_departure_commitments = sum(
                agent.agent_id in assignments
                and agent.last_charger_departure_frame is not None
                and state.frame - agent.last_charger_departure_frame <= 6
                for agent in agents
            )
            route_cost = sum(
                self._claim_safe_assignment_route_steps(
                    state,
                    agent,
                    assignments[agent.agent_id],
                    position=agent.position,
                )
                for agent in agents
                if agent.agent_id in assignments
            )
            stable_ids = tuple(
                assignments.get(agent.agent_id).task_id
                if agent.agent_id in assignments
                else "~"
                for agent in agents
            )
            preserved_neural_commitments = sum(
                assignments.get(agent.agent_id) is not None
                and assignments[agent.agent_id].task_id
                == agent.route_commitment_task_id
                for agent in agents
            )
            if prioritize_old_tasks:
                score: tuple[Any, ...] = (
                    # A valid Actor-visible route commitment is episode
                    # memory. It must outrank assignment count and task age;
                    # otherwise reward and offline labels silently retarget a
                    # robot while its observation still exposes the committed
                    # pickup. Task coverage and age remain authoritative only
                    # among assignments preserving the same commitments.
                    -preserved_neural_commitments,
                    -assigned_count,
                    -overdue_count,
                    -sum(ages),
                    projected_starvation,
                    -recent_departure_commitments,
                    route_cost,
                    stable_ids,
                )
            else:
                score = (
                    -preserved_neural_commitments,
                    -assigned_count,
                    route_cost,
                    stable_ids,
                )
            candidates.append((score, assignments))
        if not candidates:
            return {}
        return min(candidates, key=lambda item: item[0])[1]

    def _refresh_navigation_goals(self, state: WarehouseState) -> None:
        """Compatibility wrapper for adapters that restore edited states."""

        refresh_navigation_goals(self, state)
    def observations(self) -> dict[str, Any]:
        self._require_state()
        from .observations import all_local_observations

        return all_local_observations(self.state, self.config)

    def action_masks(self) -> dict[str, tuple[float, ...]]:
        self._require_state()
        return {
            agent.agent_id: legal_action_mask(
                self.state,
                agent,
                self.config.map_layout_id,
            )
            for agent in self.state.agents
        }

    def global_state(self) -> Any:
        self._require_state()
        from .observations import global_observation

        return global_observation(self.state, self.config)

    def _resolve_motion(
        self,
        state: WarehouseState,
        actions: Mapping[str, str],
    ) -> tuple[
        dict[str, tuple[int, int]],
        dict[str, str],
        set[str],
        bool,
        str | None,
        dict[str, tuple[int, int]],
    ]:
        targets: dict[str, tuple[int, int]] = {}
        executed: dict[str, str] = {}
        invalid: set[str] = set()
        for agent in state.agents:
            requested = str(actions.get(agent.agent_id, "WAIT"))
            if requested not in ACTIONS:
                raise ValueError(f"Unknown warehouse action: {requested}")
            target = agent.position
            if agent.active and requested in MOVE_DELTAS:
                delta = MOVE_DELTAS[requested]
                candidate = (agent.position[0] + delta[0], agent.position[1] + delta[1])
                if is_passable(candidate, self.config.map_layout_id):
                    target = candidate
                else:
                    invalid.add(agent.agent_id)
            targets[agent.agent_id] = target
            executed[agent.agent_id] = requested if target != agent.position else "WAIT"

        left, right = state.agents
        intended_targets = dict(targets)
        same_target = targets[left.agent_id] == targets[right.agent_id]
        swap = (
            targets[left.agent_id] == right.position
            and targets[right.agent_id] == left.position
        )
        robot_collision = bool(
            left.active
            and right.active
            and (same_target or swap)
        )
        collision_kind: str | None = None
        if robot_collision:
            if swap:
                collision_kind = "swap"
            elif (
                targets[left.agent_id] == left.position
                or targets[right.agent_id] == right.position
            ):
                collision_kind = "occupied_stationary"
            else:
                collision_kind = "same_target"
        if robot_collision:
            for agent in state.agents:
                targets[agent.agent_id] = agent.position
                executed[agent.agent_id] = "WAIT"
        return (
            targets,
            executed,
            invalid,
            robot_collision,
            collision_kind,
            intended_targets,
        )

    def _coordination_events(
        self,
        state: WarehouseState,
        actions: Mapping[str, str],
        executed: Mapping[str, str],
        intended_targets: Mapping[str, tuple[int, int]],
        collision_kind: str | None,
    ) -> tuple[dict[str, Any], ...]:
        """Describe observable corridor coordination without changing score."""

        left, right = state.agents
        events: list[dict[str, Any]] = []
        if collision_kind is not None:
            events.append(
                {
                    "event": "collision_risk",
                    "conflict_kind": collision_kind,
                    "agents": self.agent_ids,
                    "proposed_actions": dict(actions),
                    "intended_targets": dict(intended_targets),
                }
            )
            return tuple(events)

        same_row = left.position[0] == right.position[0]
        same_column = left.position[1] == right.position[1]
        line_distance = abs(left.position[0] - right.position[0]) + abs(
            left.position[1] - right.position[1]
        )
        clear_line = (
            (same_row or same_column)
            and shortest_path_distance(
                left.position,
                right.position,
                self.config.map_layout_id,
            ) == line_distance
        )
        next_distance = abs(
            intended_targets[left.agent_id][0]
            - intended_targets[right.agent_id][0]
        ) + abs(
            intended_targets[left.agent_id][1]
            - intended_targets[right.agent_id][1]
        )
        both_requested_move = all(
            str(actions.get(agent_id, "WAIT")) in MOVE_DELTAS
            for agent_id in self.agent_ids
        )
        if clear_line and both_requested_move and next_distance < line_distance:
            events.append(
                {
                    "event": "head_on_conflict_risk",
                    "agents": self.agent_ids,
                    "distance": line_distance,
                    "proposed_actions": dict(actions),
                    "intended_targets": dict(intended_targets),
                }
            )

        # A yield label requires a verified one-step collision counterfactual.
        # Proximity, an ordinary wait, or charging alone is never enough.  This
        # is explanation evidence only and does not alter either Actor action.
        for yielding in state.agents:
            passing = right if yielding.agent_id == left.agent_id else left
            if (
                yielding.position == self.layout.charger_position
                or (
                    self._requires_charge(state, yielding)
                    and passing.position == self.layout.charger_position
                )
            ):
                continue
            actual_distance = shortest_path_distance(
                intended_targets[yielding.agent_id],
                yielding.navigation_goal_position,
                self.config.map_layout_id,
            )
            candidates: list[tuple[int, str, str]] = []
            for candidate_action in MOVE_DELTAS:
                if candidate_action == str(actions.get(yielding.agent_id, "WAIT")):
                    continue
                candidate_joint = dict(actions)
                candidate_joint[yielding.agent_id] = candidate_action
                (
                    candidate_targets,
                    _,
                    invalid_candidates,
                    candidate_collision,
                    candidate_kind,
                    _,
                ) = self._resolve_motion(state, candidate_joint)
                if yielding.agent_id in invalid_candidates or not candidate_collision:
                    continue
                candidate_position = (
                    yielding.position[0] + MOVE_DELTAS[candidate_action][0],
                    yielding.position[1] + MOVE_DELTAS[candidate_action][1],
                )
                candidate_distance = shortest_path_distance(
                    candidate_position,
                    yielding.navigation_goal_position,
                    self.config.map_layout_id,
                )
                if candidate_distance <= actual_distance:
                    candidates.append(
                        (candidate_distance, candidate_action, str(candidate_kind))
                    )
                del candidate_targets
            if not candidates:
                continue
            _, candidate_action, candidate_kind = min(candidates)
            events.append(
                {
                    "event": "coordination_yield",
                    "yielding_agent_id": yielding.agent_id,
                    "passing_agent_id": passing.agent_id,
                    "candidate_action": candidate_action,
                    "candidate_conflict_kind": candidate_kind,
                    "proposed_actions": dict(actions),
                    "executed_actions": dict(executed),
                    "intended_targets": dict(intended_targets),
                }
            )

        for agent in state.agents:
            teammate = right if agent.agent_id == left.agent_id else left
            if (
                agent.position != self.layout.charger_position
                and str(executed.get(agent.agent_id, "WAIT")) == "WAIT"
                and self._requires_charge(state, agent)
                and teammate.position == self.layout.charger_position
                and str(executed.get(teammate.agent_id, "WAIT")) == "WAIT"
            ):
                events.append(
                    {
                        "event": "charger_queue",
                        "waiting_agent_id": agent.agent_id,
                        "occupant_agent_id": teammate.agent_id,
                    }
                )
        return tuple(events)

    def _assignment_potential(
        self,
        state: WarehouseState,
        positions: Mapping[str, tuple[int, int]],
        *,
        excluded_task_ids: set[str] | frozenset[str] = frozenset(),
        task_age_frame: int | None = None,
    ) -> float:
        """Minimum safe mission cost, including observable coordination delay.

        The coordination term is part of the same state potential as route and
        charging work.  It therefore telescopes over a state loop and cannot be
        collected repeatedly as a fixed yielding or charger bonus.  Its only
        purpose is to make clearing a real head-on blockage or a single-charger
        queue cheaper than preserving that blocked state.
        """

        potential = 0.0
        free_agents: list[AgentState] = []
        for agent in state.agents:
            if not agent.active:
                continue
            position = positions.get(agent.agent_id, agent.position)
            if agent.carrying_task_id:
                task = state.task_by_id(agent.carrying_task_id)
                if task.task_id not in excluded_task_ids:
                    potential += self._priority_safe_task_cost(
                        state,
                        agent,
                        task,
                        position=position,
                        task_age_frame=task_age_frame,
                    )
            else:
                free_agents.append(agent)
        available = sorted(
            (
                task
                for task in state.tasks
                if task.status == "available" and task.task_id not in excluded_task_ids
            ),
            key=lambda task: task.task_id,
        )
        if not free_agents or not available:
            return float(
                potential + self._coordination_delay_cost(state, positions)
            )
        costs = {
            (agent.agent_id, task.task_id): self._priority_safe_task_cost(
                state,
                agent,
                task,
                position=positions.get(agent.agent_id, agent.position),
                task_age_frame=task_age_frame,
            )
            for agent in free_agents
            for task in available
        }
        if len(free_agents) == 1:
            potential += min(
                costs[(free_agents[0].agent_id, task.task_id)]
                for task in available
            )
        elif len(available) == 1:
            potential += min(
                costs[(agent.agent_id, available[0].task_id)]
                for agent in free_agents
            )
        else:
            left, right = free_agents[:2]
            first, second = available[:2]
            potential += min(
                costs[(left.agent_id, first.task_id)]
                + costs[(right.agent_id, second.task_id)],
                costs[(left.agent_id, second.task_id)]
                + costs[(right.agent_id, first.task_id)],
            )
        return float(
            potential + self._coordination_delay_cost(state, positions)
        )

    def _coordination_delay_cost(
        self,
        state: WarehouseState,
        positions: Mapping[str, tuple[int, int]],
    ) -> float:
        """Estimate state-only clearance work for a blocked two-robot route."""

        active = [agent for agent in state.agents if agent.active]
        if len(active) != 2:
            return 0.0
        left, right = active
        left_position = positions.get(left.agent_id, left.position)
        right_position = positions.get(right.agent_id, right.position)
        delay = 0.0
        clearance_cost = self.config.reward.coordination_clearance_cost
        priority_agent = min(
            active,
            key=lambda agent: (
                -int(agent.carrying_task_id is not None),
                shortest_path_distance(
                    positions.get(agent.agent_id, agent.position),
                    agent.navigation_goal_position,
                    self.config.map_layout_id,
                ),
                agent.agent_id,
            ),
        )
        expected_yielding_agent_id = next(
            agent.agent_id
            for agent in active
            if agent.agent_id != priority_agent.agent_id
        )
        just_cleared_head_on = any(
            str(event.get("event", "")) == "coordination_yield"
            and str(event.get("yielding_agent_id", ""))
            == expected_yielding_agent_id
            for event in state.last_coordination_events
        )

        same_row = left_position[0] == right_position[0]
        same_column = left_position[1] == right_position[1]
        if same_row or same_column:
            axis = 1 if same_row else 0
            separation = abs(
                left_position[axis] - right_position[axis]
            )
            unobstructed = (
                shortest_path_distance(
                    left_position,
                    right_position,
                    self.config.map_layout_id,
                )
                == separation
            )
            left_toward_right = (
                left.navigation_goal_position[axis] - left_position[axis]
            ) * (
                right_position[axis] - left_position[axis]
            ) > 0
            right_toward_left = (
                right.navigation_goal_position[axis] - right_position[axis]
            ) * (
                left_position[axis] - right_position[axis]
            ) > 0
            if (
                unobstructed
                and 0 < separation <= 6
                and left_toward_right
                and right_toward_left
                and not just_cleared_head_on
            ):
                yielding = state.by_id(expected_yielding_agent_id)
                delay += credit.measured_head_on_clearance_delay(
                    self,
                    positions.get(yielding.agent_id, yielding.position),
                    positions.get(priority_agent.agent_id, priority_agent.position),
                    same_row=same_row,
                    clearance_cap=clearance_cost,
                )

        delay += charger_queue_clearance_delay(
            self,
            state,
            positions,
            active,
            clearance_cost,
        )
        return float(delay)

    def _human_route_regret(
        self,
        state: WarehouseState,
        actions: Mapping[str, str],
    ) -> float:
        """Return the participant's one-step regret against a frozen objective.

        Detour score is deliberately *not* based on the team's assignment
        potential.  Re-solving the two-robot assignment for every candidate
        action lets a one-cell move switch the matching and attributes the
        resulting whole-team cost jump to the participant.  Instead, the
        transition-start state selects one immediate route objective (charger,
        carried delivery, or the participant's member of the optimal matching)
        and every legal candidate is compared against that same objective.
        Consequently a single grid move can differ from the best legal move by
        at most two path steps.
        """

        human = state.by_id(self.config.human_agent_id)
        requested = str(actions.get(human.agent_id, "WAIT"))
        if not human.active:
            return 0.0
        # A charger wait is never classified as a participant detour.  When
        # charging is necessary it is safety work; when it is unnecessary the
        # ordinary time cost is sufficient and avoids double-counting.
        if requested == "WAIT" and human.position == self.layout.charger_position:
            return 0.0
        if requested in MOVE_DELTAS:
            delta = MOVE_DELTAS[requested]
            candidate = (human.position[0] + delta[0], human.position[1] + delta[1])
            if not is_passable(candidate, self.config.map_layout_id):
                return 0.0
        chosen_targets, _, _, chosen_collision, _, _ = self._resolve_motion(
            state, actions
        )
        if chosen_collision:
            return 0.0

        robot_two = state.by_id("robot_2")
        ai_requested = str(actions.get("robot_2", "WAIT"))
        ai_target = chosen_targets["robot_2"]
        # Waiting or stepping aside to clear the AI's intended route is not a detour.
        if (
            ai_requested in MOVE_DELTAS
            and ai_target == human.position
            and chosen_targets[human.agent_id] != human.position
        ):
            return 0.0
        del robot_two

        route_goal = self._frozen_route_goal(state, human.agent_id)
        if route_goal is None:
            return 0.0
        chosen = shortest_path_distance(
            chosen_targets[human.agent_id],
            route_goal,
            self.config.map_layout_id,
        )
        alternatives: list[float] = []
        for candidate_action in ACTIONS:
            candidate_actions = dict(actions)
            candidate_actions[human.agent_id] = candidate_action
            targets, _, invalid, collision, _, _ = self._resolve_motion(
                state, candidate_actions
            )
            if human.agent_id in invalid or collision:
                continue
            alternatives.append(
                shortest_path_distance(
                    targets[human.agent_id],
                    route_goal,
                    self.config.map_layout_id,
                )
            )
        if not alternatives:
            return 0.0
        return min(2.0, max(0.0, float(chosen - min(alternatives))))

    def _avoidable_wait_agents(
        self,
        state: WarehouseState,
        requested_actions: Mapping[str, str],
        executed_actions: Mapping[str, str],
    ) -> tuple[str, ...]:
        """Identify voluntary WAITs that forgo safe one-step route progress.

        This detector is used only by the training reward.  A wait is exempt
        when charging is still required, when no frozen mission goal exists,
        or when every closer move would be invalid or conflict with the
        teammate's simultaneously requested action.  It therefore does not
        penalize charging, yielding, blocked actions, or unavoidable queues.
        """

        avoidable: list[str] = []
        for agent in state.agents:
            if (
                not agent.active
                or str(requested_actions.get(agent.agent_id, "WAIT")) != "WAIT"
                or str(executed_actions.get(agent.agent_id, "WAIT")) != "WAIT"
            ):
                continue
            if (
                agent.position == self.layout.charger_position
                and (
                    agent.charge_mode_active
                    or agent.navigation_goal_kind == "charge"
                )
            ):
                continue
            goal = self._frozen_route_goal(
                state,
                agent.agent_id,
                prioritize_old_tasks=True,
            )
            if goal is None:
                continue
            current_distance = shortest_path_distance(
                agent.position,
                goal,
                self.config.map_layout_id,
            )
            for candidate_action in MOVE_DELTAS:
                if not action_is_robustly_safe(
                    self,
                    state,
                    requested_actions,
                    agent.agent_id,
                    candidate_action,
                ):
                    continue
                trial_actions = dict(requested_actions)
                trial_actions[agent.agent_id] = candidate_action
                targets, _, invalid, collision, _, _ = self._resolve_motion(
                    state,
                    trial_actions,
                )
                if agent.agent_id in invalid or collision:
                    continue
                target = targets[agent.agent_id]
                if target == agent.position:
                    continue
                candidate_distance = shortest_path_distance(
                    target,
                    goal,
                    self.config.map_layout_id,
                )
                if candidate_distance < current_distance:
                    avoidable.append(agent.agent_id)
                    break
        return tuple(sorted(avoidable))

    def _frozen_route_goal(
        self,
        state: WarehouseState,
        agent_id: str,
        *,
        prioritize_old_tasks: bool = False,
    ) -> tuple[int, int] | None:
        """Choose an agent route goal once from the frozen pre-move state."""

        return frozen_route_goal(
            self,
            state,
            agent_id,
            prioritize_old_tasks=prioritize_old_tasks,
        )


    def step(
        self,
        actions: Mapping[str, str],
        *,
        decision_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        self._require_state()
        if self.state.terminated or self.state.truncated:
            raise RuntimeError("Cannot step a completed warehouse episode.")
        previous = deepcopy(self.state)
        pre_move_observations = self.observations()
        raw_actions = {
            agent.agent_id: str(actions.get(agent.agent_id, "WAIT"))
            for agent in previous.agents
        }
        frozen_missions = credit.frozen_training_missions(self, previous)
        frozen_mission_costs_before = {
            agent.agent_id: credit.frozen_mission_cost(
                self,
                previous,
                agent.agent_id,
                frozen_missions[agent.agent_id],
            )
            for agent in previous.agents
        }
        coordination_cost_before = self._coordination_delay_cost(
            previous,
            {agent.agent_id: agent.position for agent in previous.agents},
        )
        potential_before = self._assignment_potential(
            previous,
            {agent.agent_id: agent.position for agent in previous.agents},
        )
        route_regret = (
            self._human_route_regret(previous, raw_actions)
            if self.config.participant_detour_scoring
            else 0.0
        )
        (
            targets,
            executed,
            invalid,
            robot_collision,
            collision_kind,
            intended_targets,
        ) = self._resolve_motion(previous, raw_actions)
        coordination_events = self._coordination_events(
            previous,
            raw_actions,
            executed,
            intended_targets,
            collision_kind,
        )
        # A generic one-step collision counterfactual is not sufficient to
        # claim that a robot intentionally yielded.  Retain only a plan that
        # was derived before either current action existed, then audit whether
        # both independently selected actions actually followed it.
        coordination_events = tuple(
            event
            for event in coordination_events
            if str(event.get("event", "")) != "coordination_yield"
        )
        frozen_coordination_plan = derive_frozen_coordination_plan(self, previous)
        plan_execution = coordination_plan_execution_event(
            frozen_coordination_plan,
            requested_actions=raw_actions,
            executed_actions=executed,
            intended_targets=intended_targets,
        )
        if plan_execution is not None:
            coordination_events += (plan_execution,)
            if bool(plan_execution.get("completed", False)):
                coordination_events += (
                    {
                        "event": "coordination_yield",
                        "plan_id": plan_execution["plan_id"],
                        "yielding_agent_id": plan_execution.get(
                            "yielding_agent_id",
                            plan_execution.get("clearing_agent_id"),
                        ),
                        "passing_agent_id": plan_execution[
                            "priority_agent_id"
                        ],
                        "candidate_action": plan_execution["moving_action"],
                        "candidate_conflict_kind": plan_execution[
                            "plan_kind"
                        ],
                        "reason_code": plan_execution["reason_code"],
                        "proposed_actions": dict(raw_actions),
                        "executed_actions": dict(executed),
                        "intended_targets": dict(intended_targets),
                    },
                )
        for agent in previous.agents:
            action = str(executed.get(agent.agent_id, "WAIT"))
            if action not in MOVE_DELTAS:
                continue
            if necessary_participant_standoff_clearance(
                self,
                previous,
                agent,
                candidate_action=action,
            ):
                coordination_events += (
                    {
                        "event": "participant_standoff_clearance",
                        "agent_id": agent.agent_id,
                        "action": action,
                        "derived_from_frame": previous.frame,
                        "reason_code": "observed_participant_stall_clearance",
                    },
                )
        coordination_events += credit.occupied_cell_clearance_events(
            self,
            previous,
            executed,
            intended_targets,
        )
        (
            counterfactual_regret_units,
            avoidable_wait_agents,
            detour_agents,
            loaded_detour_agents,
            best_counterfactual_distances,
            joint_wait_escape_actions,
        ) = credit.counterfactual_action_regrets(
            self,
            previous,
            raw_actions,
            executed,
            targets,
            frozen_missions,
            coordination_events,
        )

        next_state = deepcopy(previous)
        credit.extend_safe_path_baseline_for_clearance(
            self,
            previous,
            next_state,
            raw_actions,
            executed,
            targets,
            coordination_events,
            avoidable_wait_agents,
            tuple(sorted(loaded_detour_agents)),
            robot_collision,
        )
        next_state.frame += 1
        advance_coordination_plan(previous, next_state, plan_execution)
        next_state.last_robot_collision_event = robot_collision
        next_state.last_robot_collision_kind = collision_kind
        next_state.last_coordination_events = coordination_events
        if robot_collision:
            next_state.collision_count += 1
            next_state.robot_collision_events += 1
        next_state.invalid_move_count += len(invalid)
        charger_energy_gained = 0.0
        charger_energy_gained_by_agent: dict[str, float] = {}
        energy_events: list[dict[str, Any]] = []
        for agent in next_state.agents:
            previous_agent = previous.by_id(agent.agent_id)
            previous_position = previous_agent.position
            previous_battery = previous_agent.battery
            requested = raw_actions[agent.agent_id]
            action = executed[agent.agent_id]
            agent.last_action = requested
            agent.last_executed_action = action
            agent.position = targets[agent.agent_id]
            if action in MOVE_DELTAS:
                agent.battery = max(
                    0.0,
                    agent.battery - self.config.move_battery_cost,
                )
                agent.heading = action
            elif (
                action == "WAIT"
                and agent.position == self.layout.charger_position
                and agent.battery < 100.0
            ):
                before = agent.battery
                agent.battery = min(100.0, agent.battery + self.config.charge_per_wait)
                gained = agent.battery - before
                charger_energy_gained += gained
                if gained > 0.0:
                    charger_energy_gained_by_agent[agent.agent_id] = gained
            agent.last_battery_delta = agent.battery - previous_battery
            if agent.last_battery_delta > 0.0:
                agent.steps_since_charging = 0
                agent.charger_wait_streak = previous_agent.charger_wait_streak + 1
            else:
                agent.steps_since_charging = min(
                    self.config.horizon,
                    previous_agent.steps_since_charging + 1,
                )
                agent.charger_wait_streak = 0
            if (
                previous_position == self.layout.charger_position
                and agent.position != self.layout.charger_position
            ):
                agent.last_charger_departure_frame = next_state.frame
                agent.deliveries_at_last_charger_departure = (
                    previous_agent.deliveries_completed
                )
                agent.team_deliveries_at_last_charger_departure = (
                    previous.total_deliveries
                )
                agent.carrying_task_at_last_charger_departure = (
                    previous_agent.carrying_task_id
                )
                energy_events.append(
                    {
                        "event": "charger_departure",
                        "agent_id": agent.agent_id,
                        "battery": float(agent.battery),
                        "premature": bool(
                            self._requires_charge(
                                previous,
                                previous_agent,
                                position=self.layout.charger_position,
                            )
                            and not (
                                plan_execution is not None
                                and bool(
                                    plan_execution.get(
                                        "execution_aligned",
                                        False,
                                    )
                                )
                                and str(
                                    plan_execution.get("phase", "")
                                ) == "CLEAR_CELL"
                                and str(
                                    plan_execution.get(
                                        "moving_agent_id",
                                        "",
                                    )
                                ) == agent.agent_id
                            )
                        ),
                    }
                )
            elif (
                previous_position != self.layout.charger_position
                and agent.position == self.layout.charger_position
                and previous_agent.last_charger_departure_frame is not None
            ):
                elapsed = next_state.frame - previous_agent.last_charger_departure_frame
                (
                    completed_mission_progress,
                    completed_coordination_progress,
                ) = charger_departure_progress(
                    previous,
                    previous_agent,
                )
                frozen_goal = self._frozen_route_goal(
                    previous,
                    previous_agent.agent_id,
                    prioritize_old_tasks=True,
                )
                route_progress = bool(
                    frozen_goal is not None
                    and frozen_goal != self.layout.charger_position
                    and shortest_path_distance(
                        self.layout.charger_position,
                        frozen_goal,
                        self.config.map_layout_id,
                    )
                    < shortest_path_distance(
                        previous_agent.position,
                        frozen_goal,
                        self.config.map_layout_id,
                    )
                )
                reentry = charger_reentry_event(
                    agent,
                    elapsed=elapsed,
                    completed_mission_progress=(
                        completed_mission_progress or route_progress
                    ),
                    completed_coordination_progress=(
                        completed_coordination_progress
                        or bool(
                            plan_execution is not None
                            and plan_execution.get(
                                "execution_aligned",
                                False,
                            )
                            and str(
                                plan_execution.get("moving_agent_id", "")
                            ) == agent.agent_id
                        )
                    ),
                )
                if reentry is not None:
                    energy_events.append(reentry)

        claimed_tasks: list[DeliveryTask] = []
        delivered_tasks: list[DeliveryTask] = []
        replacement_tasks: list[DeliveryTask] = []
        pickup_agents: set[str] = set()
        delivery_agents: set[str] = set()
        for agent in next_state.agents:
            if agent.carrying_task_id:
                carried = next_state.task_by_id(agent.carrying_task_id)
                if agent.position == carried.delivery_position:
                    carried.status = "delivered"
                    carried.delivered_frame = next_state.frame
                    delivered_tasks.append(carried)
                    agent.carrying_task_id = None
                    agent.deliveries_completed += 1
                    next_state.total_deliveries += 1
                    delivery_agents.add(agent.agent_id)
            if agent.carrying_task_id is None:
                available_here = sorted(
                    (
                        task
                        for task in next_state.tasks
                        if task.status == "available"
                        and task.pickup_position == agent.position
                    ),
                    key=lambda task: task.task_id,
                )
                if available_here:
                    claimed = available_here[0]
                    claimed.status = "carried"
                    claimed.carrier_agent_id = agent.agent_id
                    claimed.claimed_frame = next_state.frame
                    agent.carrying_task_id = claimed.task_id
                    claimed.claimed_battery = float(agent.battery)
                    (
                        claimed.shortest_safe_delivery_steps,
                        claimed.safe_path_charge_planned,
                    ) = credit.safe_delivery_completion_plan(
                        self,
                        agent,
                        claimed,
                    )
                    pickup_agents.add(agent.agent_id)
                    claimed_tasks.append(claimed)

        if delivered_tasks:
            delivered_ids = {task.task_id for task in delivered_tasks}
            next_state.tasks = [
                task for task in next_state.tasks if task.task_id not in delivered_ids
            ]
            next_state.completed_tasks.extend(delivered_tasks)
            for _ in delivered_tasks:
                excluded = {
                    self.layout.charger_position,
                    *self.layout.robot_start_positions,
                    *self.layout.task_endpoint_exclusions,
                    *(agent.position for agent in next_state.agents),
                    *(
                        endpoint
                        for task in next_state.tasks
                        for endpoint in (task.pickup_position, task.delivery_position)
                    ),
                }
                replacement = self._sample_delivery_job(
                    task_index=next_state.next_task_index,
                    created_frame=next_state.frame,
                    excluded_positions=excluded,
                )
                next_state.next_task_index += 1
                next_state.tasks.append(replacement)
                replacement_tasks.append(replacement)

        update_route_commitments(self, previous, next_state, executed)

        ineffective_joint_wait = bool(
            all(action == "WAIT" for action in executed.values())
            and charger_energy_gained <= 0.0
            and not claimed_tasks
            and not delivered_tasks
        )
        next_state.ineffective_joint_wait_streak = (
            previous.ineffective_joint_wait_streak + 1
            if ineffective_joint_wait
            else 0
        )
        for agent in next_state.agents:
            before_agent = previous.by_id(agent.agent_id)
            agent.avoidable_wait_streak = (
                before_agent.avoidable_wait_streak + 1
                if agent.agent_id in avoidable_wait_agents
                else 0
            )

        shutdown_agents = [
            agent.agent_id
            for agent in next_state.agents
            if agent.active and agent.battery <= 0.0
        ]
        for agent_id in shutdown_agents:
            next_state.by_id(agent_id).active = False
        next_state.shutdown_count += len(shutdown_agents)

        score_components = {
            "delivery": self.config.delivery_points * len(delivered_tasks),
            "robot_collision": (
                self.config.robot_collision_points if robot_collision else 0.0
            ),
            "shutdown": self.config.shutdown_points * len(shutdown_agents),
            "time": self.config.step_points,
            "human_detour": self.config.human_detour_points_per_unit * route_regret,
        }
        if shutdown_agents and next_state.frame < self.config.horizon:
            score_components["time"] += self.config.step_points * (
                self.config.horizon - next_state.frame
            )
        score_delta = float(sum(score_components.values()))
        next_state.user_score += score_delta
        for name, value in score_components.items():
            next_state.score_breakdown[name] += float(value)
        next_state.human_route_regret_units += route_regret

        terminated = bool(shutdown_agents)
        truncated = bool(next_state.frame >= self.config.horizon and not terminated)
        reason = "battery_shutdown" if terminated else "horizon" if truncated else None
        next_state.terminated = terminated
        next_state.truncated = truncated
        next_state.terminal_reason = reason
        self._refresh_navigation_goals(next_state)
        assign_persistent_pickup_goals(self, next_state)
        self._refresh_navigation_goals(next_state)
        next_coordination_plan = prepare_coordination_plan(self, next_state)
        synchronize_persistent_goals(
            self,
            previous,
            next_state,
            coordination_plan=next_coordination_plan,
        )
        frozen_mission_costs_after = {
            agent.agent_id: credit.frozen_mission_cost(
                self,
                next_state,
                agent.agent_id,
                frozen_missions[agent.agent_id],
            )
            for agent in next_state.agents
        }
        potential_after = self._assignment_potential(
            next_state,
            {agent.agent_id: agent.position for agent in next_state.agents},
            excluded_task_ids={task.task_id for task in replacement_tasks},
            task_age_frame=previous.frame,
        )
        charger_return_cycle_agents = tuple(
            sorted(
                str(event["agent_id"])
                for event in energy_events
                if event.get("event") == "charger_return_cycle"
            )
        )
        frozen_assignees_by_task = {
            mission.task.task_id: agent_id
            for agent_id, mission in frozen_missions.items()
            if mission is not None and mission.task is not None
        }
        starving_task_ids = tuple(
            sorted(
                task.task_id
                for task in next_state.tasks
                if task.status == "available"
                and next_state.frame - task.created_frame > 40
                and not any(
                    agent.route_commitment_task_id == task.task_id
                    for agent in next_state.agents
                    if agent.active
                )
                and (
                    assignee_id := frozen_assignees_by_task.get(task.task_id)
                )
                is not None
                and counterfactual_regret_units.get(assignee_id, 0.0) > 0.0
            )
        )
        temporal_violations = transition_temporal_violations(
            self,
            previous,
            next_state,
            requested_actions=raw_actions,
            executed_actions=executed,
            pickup_agents=pickup_agents,
            delivery_agents=delivery_agents,
            coordination_events=coordination_events,
            energy_events=energy_events,
        )
        reward_credit = credit.transition_credit_components(
            self,
            terminated=terminated,
            score_delta=score_delta,
            previous_state=previous,
            next_state=next_state,
            requested_actions=raw_actions,
            executed_actions=executed,
            mission_costs_before=frozen_mission_costs_before,
            mission_costs_after=frozen_mission_costs_after,
            coordination_cost_before=coordination_cost_before,
            counterfactual_regret_units=counterfactual_regret_units,
            avoidable_wait_agents=avoidable_wait_agents,
            loaded_detour_agents=tuple(sorted(loaded_detour_agents)),
            coordination_events=coordination_events,
            charger_return_cycle_agents=charger_return_cycle_agents,
            starving_task_ids=starving_task_ids,
            assignment_potential_before=potential_before,
            assignment_potential_after=potential_after,
            unexplained_reversal_agents=temporal_violations[
                "unexplained_reversal_agents"
            ],
            short_cycle_agents=temporal_violations[
                "short_cycle_agents"
            ],
            invalid_goal_switch_agents=temporal_violations[
                "invalid_goal_switch_agents"
            ],
        )
        rewards = reward_credit["rewards"]
        potential_shaping = reward_credit["potential_shaping_reward"]
        avoidable_wait_penalty = -reward_credit["avoidable_wait_penalty_reward"]
        mission_regression_units = reward_credit["mission_regression_units"]
        mission_regression_penalty = -reward_credit[
            "mission_regression_penalty_reward"
        ]
        next_state.last_rewards = dict(rewards)
        self.state = next_state

        collision_agents = self.agent_ids if robot_collision else ()
        action_resolution = {
            agent.agent_id: {
                "requested_action": raw_actions[agent.agent_id],
                "executed_action": executed[agent.agent_id],
                "position": previous.by_id(agent.agent_id).position,
                "intended_target": intended_targets[agent.agent_id],
                "teammate_position": next(
                    other.position
                    for other in previous.agents
                    if other.agent_id != agent.agent_id
                ),
                "teammate_requested_action": next(
                    raw_actions[other.agent_id]
                    for other in previous.agents
                    if other.agent_id != agent.agent_id
                ),
                "teammate_intended_target": next(
                    intended_targets[other.agent_id]
                    for other in previous.agents
                    if other.agent_id != agent.agent_id
                ),
                "collision_kind": collision_kind,
                "environment_changed_action": (
                    raw_actions[agent.agent_id] != executed[agent.agent_id]
                ),
                "blocked_reason": (
                    collision_kind or "robot_collision"
                    if robot_collision
                    else "static_obstacle"
                    if agent.agent_id in invalid
                    else None
                ),
            }
            for agent in next_state.agents
        }
        decision_trace = build_decision_trace(
            self,
            previous=previous,
            next_state=next_state,
            raw_actions=raw_actions,
            executed_actions=executed,
            action_resolution=action_resolution,
            decision_metadata=decision_metadata,
            frozen_missions=credit.serialized_frozen_missions(frozen_missions),
            coordination_plan=plan_execution,
        )
        info = environment_info(
            self,
            reward_breakdown=score_components,
            collisions=collision_agents,
            shutdowns=tuple(sorted(shutdown_agents)),
        )
        info.update(
            {
                "proposed_actions": dict(raw_actions),
                "executed_actions": dict(executed),
                "pickup_agents": tuple(sorted(pickup_agents)),
                "delivery_agents": tuple(sorted(delivery_agents)),
                "delivered_task_ids": tuple(sorted(task.task_id for task in delivered_tasks)),
                "claimed_task_ids": tuple(sorted(task.task_id for task in claimed_tasks)),
                "created_task_ids": tuple(sorted(task.task_id for task in replacement_tasks)),
                "task_changes": tuple(
                    [
                        {
                            "event": "claimed",
                            "task_id": task.task_id,
                            "carrier_agent_id": task.carrier_agent_id,
                            "frame": task.claimed_frame,
                        }
                        for task in claimed_tasks
                    ]
                    + [
                        {
                            "event": "delivered",
                            "task_id": task.task_id,
                            "carrier_agent_id": task.carrier_agent_id,
                            "frame": task.delivered_frame,
                        }
                        for task in delivered_tasks
                    ]
                    + [
                        {
                            "event": "created",
                            "task_id": task.task_id,
                            "pickup_position": task.pickup_position,
                            "delivery_position": task.delivery_position,
                            "frame": task.created_frame,
                        }
                        for task in replacement_tasks
                    ]
                ),
                "invalid_move_agents": tuple(sorted(invalid)),
                "robot_collision_event": robot_collision,
                "robot_collision_kind": collision_kind,
                "intended_targets": dict(intended_targets),
                "coordination_events": coordination_events,
                "route_regret": {self.config.human_agent_id: route_regret},
                "assignment_potential_before": potential_before,
                "assignment_potential_after": potential_after,
                "safe_mission_potential_before": potential_before,
                "safe_mission_potential_after": potential_after,
                "frozen_missions": credit.serialized_frozen_missions(frozen_missions),
                "frozen_mission_costs_before": frozen_mission_costs_before,
                "frozen_mission_costs_after": frozen_mission_costs_after,
                "base_training_reward": reward_credit["base_training_reward"],
                "potential_shaping_reward": potential_shaping,
                "avoidable_wait_penalty_reward": -avoidable_wait_penalty,
                "mission_regression_penalty_reward": -mission_regression_penalty,
                "mission_regression_units": mission_regression_units,
                "training_reward": reward_credit["training_reward"],
                "training_rewards": dict(rewards),
                "individual_progress_units": reward_credit["individual_progress_units"],
                "individual_progress_rewards": reward_credit["individual_progress_rewards"],
                "coordination_cost_before": coordination_cost_before,
                "coordination_cost_after": reward_credit["coordination_cost_after"],
                "coordination_progress_reward": reward_credit[
                    "coordination_progress_reward"
                ],
                "counterfactual_regret_units": counterfactual_regret_units,
                "counterfactual_regret_penalty_rewards": (
                    reward_credit["counterfactual_regret_penalty_rewards"]
                ),
                "best_counterfactual_distances": best_counterfactual_distances,
                "joint_wait_escape_actions": dict(joint_wait_escape_actions),
                "repeated_avoidable_wait_penalty_rewards": (
                    reward_credit["repeated_avoidable_wait_penalty_rewards"]
                ),
                "flat_avoidable_wait_penalty_rewards": (
                    reward_credit["flat_avoidable_wait_penalty_rewards"]
                ),
                "causal_efficiency_penalty_rewards": (
                    reward_credit["causal_efficiency_penalty_rewards"]
                ),
                "temporal_consistency_penalty_rewards": (
                    reward_credit["temporal_consistency_penalty_rewards"]
                ),
                **temporal_violations,
                "avoidable_wait_streaks": {
                    agent.agent_id: agent.avoidable_wait_streak
                    for agent in next_state.agents
                },
                "avoidable_wait_agents": tuple(sorted(avoidable_wait_agents)),
                "avoidable_detour_agents": tuple(sorted(detour_agents)),
                "avoidable_loaded_delivery_detour_agents": tuple(
                    sorted(loaded_detour_agents)
                ),
                **credit.completed_delivery_path_metrics(next_state),
                "charger_energy_gained": charger_energy_gained,
                "charger_energy_gained_by_agent": dict(
                    charger_energy_gained_by_agent
                ),
                "charger_used": charger_energy_gained > 0.0,
                "energy_events": tuple(energy_events),
                "available_task_ages": {
                    task.task_id: max(0, next_state.frame - task.created_frame)
                    for task in next_state.tasks
                    if task.status == "available"
                },
                "starving_task_ids": starving_task_ids,
                "starving_task_assignees": {
                    task_id: frozen_assignees_by_task[task_id]
                    for task_id in starving_task_ids
                },
                "ineffective_joint_wait_streak": (
                    next_state.ineffective_joint_wait_streak
                ),
                "action_resolution": action_resolution,
                "decision_trace": decision_trace,
                "decision_audit": joint_decision_audit(
                    previous=previous,
                    next_state=next_state,
                    pre_move_observations=pre_move_observations,
                    raw_actions=raw_actions,
                    decision_metadata=decision_metadata,
                ),
                "terminal_reason": reason,
            }
        )
        return self.observations(), rewards, terminated, truncated, info

    def get_state(self) -> WarehouseState:
        self._require_state()
        return deepcopy(self.state)

    def set_state(self, state: WarehouseState) -> None:
        candidate = deepcopy(state)
        self._refresh_navigation_goals(candidate)
        assign_persistent_pickup_goals(self, candidate)
        self._refresh_navigation_goals(candidate)
        candidate.active_coordination_plan = None
        restored_plan = prepare_coordination_plan(self, candidate)
        synchronize_persistent_goals(
            self,
            None,
            candidate,
            reset_reason="state_restore",
            coordination_plan=restored_plan,
        )
        errors = self.validate_state(candidate)
        if errors:
            raise ValueError("Invalid warehouse state: " + "; ".join(errors))
        self.state = candidate

    def get_rng_state(self) -> object:
        return self._rng.getstate()

    def set_rng_state(self, state: object) -> None:
        self._rng.setstate(state)

    def validate_state(self, state: WarehouseState) -> tuple[str, ...]:
        return validate_warehouse_state(self, state)

    def render_ascii(self, state: WarehouseState | None = None) -> tuple[str, ...]:
        current = state or self.state
        if current is None:
            raise RuntimeError("Environment has not been reset.")
        return render_ascii_state(self, current)

    def _require_state(self) -> None:
        if self.state is None:
            raise RuntimeError("Call reset before accessing the environment state.")
