"""Training-only individual credit assignment for warehouse transitions.

The environment owns state transitions and the participant-facing score.  This
module deliberately owns only dense training credit, so offline teacher code
does not become a runtime dependency of the environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping

from .domain import AgentState, DeliveryTask, WarehouseState
from .coordination_priority import single_lane_egress_agent_id
from .energy_management import (
    charger_handoff_clearance_action,
    charger_route_is_critical,
    charger_service_required,
)
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from .transition_audit import (
    action_is_robustly_safe,
    necessary_participant_standoff_clearance,
    necessary_teammate_route_clearance,
    wait_is_robustly_safe,
)


_CAUSAL_6X7_LAYOUT_ID = (
    "warehouse_staggered_aisles_6x7_v2_three_cell_exit_no_cross"
)


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
            # A route commitment is public state created by this robot's
            # previous executed movement.  Re-running the global matching and
            # silently replacing it for one audit frame makes the teacher,
            # reward, and explanation disagree about the frozen task and was
            # the source of the task-9 -> task-8 -> charger oscillation.
            task = (
                available.get(agent.route_commitment_task_id)
                if agent.route_commitment_task_id is not None
                else None
            )
            if (
                task is None
                and agent.goal_type == "GO_TO_PICKUP"
                and agent.goal_id is not None
            ):
                task = available.get(agent.goal_id)
            if task is None:
                task = assignments.get(agent.agent_id)
            if task is None and charger_service_required(
                environment,
                state,
                agent,
            ):
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
        if charger_service_required(environment, state, agent):
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
    if mission.goal_kind == "charge":
        # Count both travel to the station and the charging waits still needed
        # for the frozen task.  Distance alone becomes zero on arrival and
        # incorrectly gives a necessary charging WAIT no progress credit.
        # Excluding the post-charge route itself keeps this an immediate
        # charge-phase objective and avoids rewarding a premature departure.
        charger_distance = shortest_path_distance(
            agent.position,
            mission.goal_position,
            environment.config.map_layout_id,
        )
        task = mission.task
        if task is None:
            return float(charger_distance)
        live_task = next(
            (
                item
                for item in (*state.tasks, *state.completed_tasks)
                if item.task_id == task.task_id
            ),
            task,
        )
        battery_at_charger = max(
            0.0,
            agent.battery
            - charger_distance * environment.config.move_battery_cost,
        )
        charged_route_steps = environment._mission_route_steps(
            state,
            agent,
            live_task,
            origin=mission.goal_position,
        )
        required_energy = (
            charged_route_steps * environment.config.move_battery_cost
        )
        remaining_waits = math.ceil(
            max(0.0, required_energy - battery_at_charger)
            / environment.config.charge_per_wait
        )
        return float(charger_distance + remaining_waits)
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

    if mission.goal_kind == "charge":
        # ``FrozenMission.task`` records which mission needs energy; it does
        # not change the immediate route target.  Counterfactual credit must
        # therefore compare candidate positions with the charger until the
        # charge-first phase has actually completed.
        return float(
            shortest_path_distance(
                position,
                mission.goal_position,
                environment.config.map_layout_id,
            )
        )
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
    return charger_route_is_critical(
        environment.config,
        position=agent.position,
        battery=agent.battery,
        charger_position=environment.layout.charger_position,
    )


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


def _best_joint_wait_escape(
    environment: Any,
    state: WarehouseState,
    requested_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
) -> tuple[dict[str, str], float]:
    """Find a collision-free team improvement from a frozen joint WAIT.

    The ordinary regret audit deliberately asks whether one robot can make
    progress under every possible peer action.  That is the correct causal
    test for a unilateral WAIT, but it cannot recognize a narrow-corridor
    state where both robots must move together to clear the route.  This
    bounded 5x5 enumeration uses only S_t, never either robot's future action,
    and returns training credit rather than a runtime action replacement.
    """

    if not all(
        str(requested_actions.get(agent.agent_id, "WAIT")) == "WAIT"
        and str(executed_actions.get(agent.agent_id, "WAIT")) == "WAIT"
        for agent in state.agents
        if agent.active
    ):
        return {}, 0.0
    positions = {agent.agent_id: agent.position for agent in state.agents}
    baseline = environment._assignment_potential(state, positions)
    agent_ids = tuple(agent.agent_id for agent in state.agents)
    charger_handoffs: dict[str, tuple[str, str]] = {}
    for occupant in state.agents:
        if occupant.position != environment.layout.charger_position:
            continue
        waiter = next(
            agent
            for agent in state.agents
            if agent.agent_id != occupant.agent_id
        )
        clearance_action = charger_handoff_clearance_action(
            environment,
            state,
            occupant,
            waiter,
        )
        if clearance_action is not None:
            charger_handoffs[occupant.agent_id] = (
                waiter.agent_id,
                clearance_action,
            )
    if not charger_handoffs and any(
        agent.position == environment.layout.charger_position
        and charger_service_required(environment, state, agent)
        for agent in state.agents
    ):
        # Productive charging is not an ineffective joint stall.  In the
        # inverse-priority case the queued robot must preserve its energy at
        # the apron; moving it away merely creates a second charger approach
        # and taught the Actor to oscillate above a correctly retained station.
        return {}, 0.0
    candidates: list[tuple[tuple[float, int, int, int], dict[str, str]]] = []
    for left_index, right_index in product(range(len(ACTIONS)), repeat=2):
        actions = {
            agent_ids[0]: ACTIONS[left_index],
            agent_ids[1]: ACTIONS[right_index],
        }
        if all(action == "WAIT" for action in actions.values()):
            continue
        targets, _, invalid, collision, _, _ = environment._resolve_motion(
            state,
            actions,
        )
        if collision or invalid:
            continue
        if charger_handoffs and any(
            actions[occupant_id] != clearance_action
            or actions[waiter_id] != "WAIT"
            for occupant_id, (waiter_id, clearance_action) in (
                charger_handoffs.items()
            )
        ):
            # A frozen-state handoff is deliberately two phase: the lower
            # battery waiter cannot infer that the occupant will clear and
            # must remain at the apron for this transition.
            continue
        if any(
            targets[agent.agent_id] == teammate.position
            and targets[agent.agent_id] != agent.position
            for agent in state.agents
            for teammate in state.agents
            if teammate.agent_id != agent.agent_id
        ):
            # A unilateral Actor cannot rely on the peer's still-private
            # current-frame action to vacate its S_t cell.  Such a joint move
            # is not a safe counterfactual for declaring WAIT avoidable.
            continue
        if any(
            actions[agent.agent_id] in MOVE_DELTAS
            and agent.battery <= environment.config.move_battery_cost
            for agent in state.agents
        ):
            continue
        if any(
            agent.position == environment.layout.charger_position
            and charger_service_required(environment, state, agent)
            and actions[agent.agent_id] != "WAIT"
            and (
                agent.agent_id not in charger_handoffs
                or actions[agent.agent_id]
                != charger_handoffs[agent.agent_id][1]
            )
            for agent in state.agents
        ):
            # Charging that is still required is productive work, not a stall.
            continue
        avoidable_loaded_regressions = 0
        for agent in state.agents:
            if (
                agent.carrying_task_id is None
                or agent.navigation_goal_kind != "delivery"
                or environment._requires_charge(state, agent)
                or actions[agent.agent_id] not in MOVE_DELTAS
            ):
                continue
            current_distance = shortest_path_distance(
                agent.position,
                agent.navigation_goal_position,
                environment.config.map_layout_id,
            )
            next_distance = shortest_path_distance(
                targets[agent.agent_id],
                agent.navigation_goal_position,
                environment.config.map_layout_id,
            )
            if (
                next_distance > current_distance
                and not necessary_urgent_charger_clearance(
                    environment,
                    state,
                    agent,
                )
                and not necessary_teammate_route_clearance(
                    environment,
                    state,
                    agent,
                )
                and not necessary_participant_standoff_clearance(
                    environment,
                    state,
                    agent,
                    candidate_action=actions[agent.agent_id],
                )
            ):
                avoidable_loaded_regressions += 1
        if avoidable_loaded_regressions:
            continue
        potential = environment._assignment_potential(state, targets)
        if potential >= baseline - 1e-9:
            continue
        moving_count = sum(action != "WAIT" for action in actions.values())
        candidates.append(
            (
                (potential, moving_count, left_index, right_index),
                actions,
            )
        )
    if not candidates:
        return {}, 0.0
    rank, selected = min(candidates, key=lambda item: item[0])
    improvement = min(2.0, max(0.0, baseline - rank[0]))
    return dict(selected), float(improvement)


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
    dict[str, str],
]:
    """Compute per-robot one-step regret with the teammate action fixed."""

    exempt_agents: set[str] = set()
    coordination_yield_agents: set[str] = set()
    verified_plan_yield_agents: set[str] = set()
    for event in coordination_events:
        kind = str(event.get("event", ""))
        if kind == "joint_coordination_plan":
            individually_aligned = event.get("agent_request_aligned", {})
            if not isinstance(individually_aligned, Mapping):
                individually_aligned = event.get("agent_execution_aligned", {})
            if isinstance(individually_aligned, Mapping):
                for agent_id, aligned in individually_aligned.items():
                    if bool(aligned):
                        exempt_agents.add(str(agent_id))
                        verified_plan_yield_agents.add(str(agent_id))
        elif kind == "coordination_yield":
            yielding_id = str(event.get("yielding_agent_id", ""))
            passing_id = str(event.get("passing_agent_id", ""))
            if (
                str(requested_actions.get(yielding_id, "WAIT")) != "WAIT"
                or str(requested_actions.get(passing_id, "WAIT"))
                in MOVE_DELTAS
            ):
                exempt_agents.add(yielding_id)
                coordination_yield_agents.add(yielding_id)
                if event.get("plan_id"):
                    verified_plan_yield_agents.add(yielding_id)
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
        actor_mask: list[float] | None = None
        if environment.config.map_layout_id == _CAUSAL_6X7_LAYOUT_ID:
            # Production regret compares only actions the deployed Actor can
            # actually submit from this frozen observation. A public joint
            # reservation may intentionally make WAIT the sole safe choice;
            # treating masked actions as alternatives falsely labels that
            # causal wait as inefficient. Archived layouts retain their
            # historical geometry-only audit below.
            from .observations import _actor_action_mask

            actor_mask = _actor_action_mask(
                state,
                agent,
                environment.config,
            )
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
            if (
                actor_mask is not None
                and actor_mask[ACTIONS.index(candidate_action)] <= 0.5
            ):
                continue
            trial = dict(requested_actions)
            trial[agent.agent_id] = candidate_action
            targets, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                trial,
            )
            if agent.agent_id in invalid or collision:
                continue
            candidate_target = targets[agent.agent_id]
            remaining_battery = (
                agent.battery - environment.config.move_battery_cost
                if candidate_action in MOVE_DELTAS
                else agent.battery
            )
            if mission.goal_kind == "charge":
                required_energy = (
                    shortest_path_distance(
                        candidate_target,
                        mission.goal_position,
                        environment.config.map_layout_id,
                    )
                    * environment.config.move_battery_cost
                )
            elif mission.task is not None:
                required_energy = (
                    environment._mission_route_steps(
                        state,
                        agent,
                        mission.task,
                        origin=candidate_target,
                    )
                    * environment.config.move_battery_cost
                )
            else:
                required_energy = (
                    shortest_path_distance(
                        candidate_target,
                        mission.goal_position,
                        environment.config.map_layout_id,
                    )
                    * environment.config.move_battery_cost
                )
            if remaining_battery + 1e-8 < required_energy:
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
        charger_handoff_action = None
        if action == "WAIT" and agent.position == environment.layout.charger_position:
            teammate = next(
                item
                for item in state.agents
                if item.agent_id != agent.agent_id
            )
            charger_handoff_action = charger_handoff_clearance_action(
                environment,
                state,
                agent,
                teammate,
            )
        if action == "WAIT":
            robust_distances: list[float] = []
            for candidate_action in ACTIONS:
                if (
                    actor_mask is not None
                    and actor_mask[ACTIONS.index(candidate_action)] <= 0.5
                ):
                    continue
                if not action_is_robustly_safe(
                    environment,
                    state,
                    requested_actions,
                    agent.agent_id,
                    candidate_action,
                ):
                    continue
                trial = dict(requested_actions)
                trial[agent.agent_id] = candidate_action
                targets = environment._resolve_motion(state, trial)[0]
                candidate_target = targets[agent.agent_id]
                remaining_battery = (
                    agent.battery - environment.config.move_battery_cost
                    if candidate_action in MOVE_DELTAS
                    else agent.battery
                )
                if mission.goal_kind == "charge":
                    required_energy = (
                        shortest_path_distance(
                            candidate_target,
                            mission.goal_position,
                            environment.config.map_layout_id,
                        )
                        * environment.config.move_battery_cost
                    )
                elif mission.task is not None:
                    required_energy = (
                        environment._mission_route_steps(
                            state,
                            agent,
                            mission.task,
                            origin=candidate_target,
                        )
                        * environment.config.move_battery_cost
                    )
                else:
                    required_energy = (
                        shortest_path_distance(
                            candidate_target,
                            mission.goal_position,
                            environment.config.map_layout_id,
                        )
                        * environment.config.move_battery_cost
                    )
                if remaining_battery + 1e-8 < required_energy:
                    continue
                robust_distances.append(
                    mission_goal_distance(
                        environment,
                        state,
                        agent,
                        mission,
                        targets[agent.agent_id],
                    )
                )
            if not robust_distances or min(robust_distances) >= chosen_distance:
                exempt = True
            else:
                best_distance = min(robust_distances)
                best_distances[agent.agent_id] = best_distance
                if (
                    agent.agent_id in coordination_yield_agents
                    and agent.agent_id not in verified_plan_yield_agents
                ):
                    # A same-target conflict with one candidate action does
                    # not justify WAIT when another action is safe against
                    # every legal teammate move and advances the same frozen
                    # mission.  Keeping the exemption here created the extra
                    # wait after a channel had already separated.
                    exempt = False
        if (
            action == "WAIT"
            and agent.position == environment.layout.charger_position
            and charger_service_required(environment, state, agent)
            and charger_handoff_action is None
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
        if not exempt and action in MOVE_DELTAS:
            exempt = bool(
                necessary_participant_standoff_clearance(
                    environment,
                    state,
                    agent,
                    candidate_action=action,
                )
            )
        if (
            not exempt
            and action in MOVE_DELTAS
            and agent.carrying_task_id is not None
        ):
            exempt = bool(
                necessary_urgent_charger_clearance(
                    environment,
                    state,
                    agent,
                )
                or necessary_teammate_route_clearance(
                    environment,
                    state,
                    agent,
                )
            )
        if (
            not exempt
            and action in MOVE_DELTAS
            and chosen_distance > best_distance
            and not wait_is_robustly_safe(
                environment,
                state,
                requested_actions,
                agent.agent_id,
            )
        ):
            exempt = True
        if not exempt and chosen_distance > best_distance:
            regrets[agent.agent_id] = min(
                2.0,
                max(0.0, float(chosen_distance - best_distance)),
            )
        if charger_handoff_action is not None:
            # Clearing the single station is coordination progress rather than
            # geometric progress toward the occupant's temporary charge goal.
            # Give it one bounded unit so WAIT supervision cannot contradict
            # the queue potential and the offline handoff label.
            regrets[agent.agent_id] = max(regrets[agent.agent_id], 1.0)
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
    joint_escape_actions, joint_escape_units = _best_joint_wait_escape(
        environment,
        state,
        requested_actions,
        executed_actions,
    )
    if joint_escape_actions:
        for agent in state.agents:
            if joint_escape_actions.get(agent.agent_id, "WAIT") == "WAIT":
                continue
            if (
                agent.position == environment.layout.charger_position
                and charger_service_required(environment, state, agent)
                and charger_handoff_clearance_action(
                    environment,
                    state,
                    agent,
                    next(
                        teammate
                        for teammate in state.agents
                        if teammate.agent_id != agent.agent_id
                    ),
                )
                != joint_escape_actions.get(agent.agent_id)
            ):
                # Defensive invariant: coordinated-escape credit must never
                # re-add an occupant whose WAIT is productive hysteretic
                # charging.  A proved two-phase handoff remains creditable.
                continue
            if agent.agent_id in avoidable_waits or regrets[agent.agent_id] > 0.0:
                # Preserve the ordinary one-agent regret magnitude when that
                # causal test already explains the WAIT.  Joint credit exists
                # only to fill the coordinated-clearance blind spot.
                continue
            regrets[agent.agent_id] = max(
                regrets[agent.agent_id],
                joint_escape_units,
            )
            # A coordinated joint escape is useful offline supervision and
            # still receives bounded regret credit, but it is not an
            # *avoidable* unilateral WAIT: neither independent Actor may rely
            # on the peer's private current-frame action.  The public
            # ineffective-joint-wait streak and deadlock metric continue to
            # expose failures of the multi-frame handshake.

    return (
        regrets,
        tuple(sorted(set(avoidable_waits))),
        tuple(sorted(detours)),
        tuple(sorted(loaded_detours)),
        best_distances,
        joint_escape_actions,
    )


def safe_delivery_completion_plan(
    environment: Any,
    agent: AgentState,
    task: DeliveryTask,
) -> tuple[float, bool]:
    """Return claim-time safe work and whether it already plans charging."""

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
        return float(delivery_distance), False
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
    return float(charger_distance + waits + charger_to_delivery), True


def safe_delivery_completion_steps(
    environment: Any,
    agent: AgentState,
    task: DeliveryTask,
) -> float:
    """Shortest safe post-claim work for the path-efficiency audit."""

    return safe_delivery_completion_plan(environment, agent, task)[0]


def occupied_cell_clearance_events(
    environment: Any,
    state: WarehouseState,
    executed_actions: Mapping[str, str],
    intended_targets: Mapping[str, tuple[int, int]],
) -> tuple[dict[str, Any], ...]:
    """Describe a necessary one-frame hold before entering a peer's S_t cell."""

    events: list[dict[str, Any]] = []
    for follower in state.agents:
        teammate = next(
            agent for agent in state.agents if agent.agent_id != follower.agent_id
        )
        if (
            follower.carrying_task_id is None
            or follower.navigation_goal_kind != "delivery"
            or str(executed_actions.get(follower.agent_id)) != "WAIT"
            or str(executed_actions.get(teammate.agent_id)) not in MOVE_DELTAS
            or intended_targets[teammate.agent_id] == teammate.position
        ):
            continue
        current_distance = shortest_path_distance(
            follower.position,
            follower.navigation_goal_position,
            environment.config.map_layout_id,
        )
        if (
            shortest_path_distance(
                teammate.position,
                follower.navigation_goal_position,
                environment.config.map_layout_id,
            )
            != current_distance - 1
        ):
            continue
        events.append(
            {
                "event": "occupied_cell_clearance_wait",
                "waiting_agent_id": follower.agent_id,
                "clearing_agent_id": teammate.agent_id,
                "occupied_position": teammate.position,
            }
        )
    return tuple(events)


def extend_safe_path_baseline_for_clearance(
    environment: Any,
    previous: WarehouseState,
    next_state: WarehouseState,
    requested_actions: Mapping[str, str],
    executed_actions: Mapping[str, str],
    actual_targets: Mapping[str, tuple[int, int]],
    coordination_events: tuple[Mapping[str, Any], ...],
    avoidable_wait_agents: tuple[str, ...],
    loaded_detour_agents: tuple[str, ...],
    transition_had_collision: bool,
) -> None:
    """Extend a safe path only for causally proved dynamic work.

    The claim-time denominator remains the immutable shortest safe plan.
    Mandatory yielding can add elapsed work on a shared narrow map, while an
    avoidable WAIT/detour/cycle must never make its own metric easier.  An
    initially unplanned charging phase is eligible only when accumulated
    mandatory clearance energy fully explains the current route-energy
    deficit.
    """

    avoidable = set(avoidable_wait_agents) | set(loaded_detour_agents)
    necessary_agents: set[str] = set()
    for event in coordination_events:
        kind = str(event.get("event", ""))
        if kind == "coordination_yield":
            necessary_agents.add(str(event.get("yielding_agent_id", "")))
        elif kind in {"charger_queue", "occupied_cell_clearance_wait"}:
            necessary_agents.add(str(event.get("waiting_agent_id", "")))

    for before_agent in previous.agents:
        task_id = before_agent.carrying_task_id
        if task_id is None:
            continue
        task = next_state.task_by_id(task_id)
        if task.shortest_safe_delivery_steps is None:
            continue
        action = str(executed_actions.get(before_agent.agent_id, "WAIT"))
        current_distance = shortest_path_distance(
            before_agent.position,
            task.delivery_position,
            environment.config.map_layout_id,
        )
        next_distance = shortest_path_distance(
            actual_targets[before_agent.agent_id],
            task.delivery_position,
            environment.config.map_layout_id,
        )
        excess_work = min(
            2.0,
            max(0.0, 1.0 - float(current_distance - next_distance)),
        )
        event_proved_necessary = bool(
            before_agent.agent_id in necessary_agents
            and before_agent.agent_id not in avoidable
        )
        requested = str(requested_actions.get(before_agent.agent_id, "WAIT"))
        directly_proved_safe_work = bool(
            not transition_had_collision
            and requested == action
            and before_agent.navigation_goal_kind == "delivery"
            and (
                (
                    action == "WAIT"
                    and before_agent.agent_id not in avoidable_wait_agents
                )
                or (
                    action in MOVE_DELTAS
                    and before_agent.agent_id not in loaded_detour_agents
                )
            )
        )
        # Counterfactual credit already searches every legal action from S_t.
        # A loaded WAIT or non-progress move that it cannot classify as
        # avoidable is mandatory decentralized safety work. Count that exact
        # elapsed excess in the safe denominator; otherwise participant
        # uncertainty is reported as route inefficiency even though no Actor
        # may inspect the participant's private current-frame action.
        necessary = event_proved_necessary or directly_proved_safe_work
        if necessary and excess_work > 0.0:
            task.shortest_safe_delivery_steps += excess_work
            task.safe_path_clearance_extension_steps += excess_work
            if action in MOVE_DELTAS:
                task.safe_path_clearance_energy_budget += (
                    excess_work * environment.config.move_battery_cost
                )
            continue

        planned_charge = bool(task.safe_path_charge_planned)
        if planned_charge or before_agent.agent_id in avoidable:
            task.safe_path_unplanned_charge_active = False
            continue
        if (
            before_agent.navigation_goal_kind == "charge"
            and charger_service_required(environment, previous, before_agent)
        ):
            delivery_to_charger = shortest_path_distance(
                task.delivery_position,
                environment.layout.charger_position,
                environment.config.map_layout_id,
            )
            required_energy = (
                current_distance
                + delivery_to_charger
                + environment.config.mission_reserve_steps
            ) * environment.config.move_battery_cost
            energy_deficit = max(0.0, required_energy - before_agent.battery)
            if (
                energy_deficit > 0.0
                and energy_deficit
                <= task.safe_path_clearance_energy_budget + 1e-8
            ):
                task.safe_path_unplanned_charge_active = True
        else:
            task.safe_path_unplanned_charge_active = False
        if task.safe_path_unplanned_charge_active and excess_work > 0.0:
            task.shortest_safe_delivery_steps += excess_work
            task.safe_path_unplanned_charge_extension_steps += excess_work


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
    loaded_detour_agents: tuple[str, ...],
    necessary_clearance_agents: tuple[str, ...],
    charger_return_cycle_agents: tuple[str, ...],
    starving_task_ids: tuple[str, ...],
    unexplained_reversal_agents: tuple[str, ...],
    short_cycle_agents: tuple[str, ...],
    invalid_goal_switch_agents: tuple[str, ...],
) -> dict[str, Any]:
    """Build all reward components without touching the user score."""

    config = environment.config.reward
    clearance_set = set(necessary_clearance_agents)
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
    for agent_id in clearance_set:
        if agent_id in progress_units:
            progress_units[agent_id] = max(0.0, progress_units[agent_id])
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
    loaded_detour_set = set(loaded_detour_agents)
    charger_cycle_set = set(charger_return_cycle_agents)
    causal_efficiency_penalties = {
        agent.agent_id: (
            -config.loaded_detour_cost
            * int(agent.agent_id in loaded_detour_set)
            -config.charger_return_cycle_cost
            * int(agent.agent_id in charger_cycle_set)
            -config.starving_task_cost * len(starving_task_ids)
        )
        for agent in next_state.agents
    }
    reversal_set = set(unexplained_reversal_agents)
    short_cycle_set = set(short_cycle_agents)
    invalid_switch_set = set(invalid_goal_switch_agents)
    temporal_consistency_penalties = {
        agent.agent_id: (
            -config.unexplained_reversal_cost
            * int(agent.agent_id in reversal_set)
            -config.short_cycle_cost
            * int(agent.agent_id in short_cycle_set)
            -config.invalid_goal_switch_cost
            * int(agent.agent_id in invalid_switch_set)
        )
        for agent in next_state.agents
    }
    rewards = {
        agent.agent_id: (
            base_reward
            + progress_rewards[agent.agent_id]
            + coordination_reward
            + regret_penalties[agent.agent_id]
            + repeated_wait_penalties[agent.agent_id]
            + flat_wait_penalties[agent.agent_id]
            + causal_efficiency_penalties[agent.agent_id]
            + temporal_consistency_penalties[agent.agent_id]
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
        "causal_efficiency_penalty_rewards": causal_efficiency_penalties,
        "temporal_consistency_penalty_rewards": temporal_consistency_penalties,
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
    loaded_detour_agents: tuple[str, ...],
    coordination_events: tuple[Mapping[str, Any], ...],
    charger_return_cycle_agents: tuple[str, ...],
    starving_task_ids: tuple[str, ...],
    assignment_potential_before: float,
    assignment_potential_after: float,
    unexplained_reversal_agents: tuple[str, ...],
    short_cycle_agents: tuple[str, ...],
    invalid_goal_switch_agents: tuple[str, ...],
) -> dict[str, Any]:
    """Select production individual credit or the isolated legacy ablation."""

    config = environment.config.reward
    if config.individual_credit_enabled:
        necessary_clearance_agent_ids = {
                    str(event.get(key, ""))
                    for event in coordination_events
                    for key in ("yielding_agent_id", "clearing_agent_id")
                    if str(event.get("event", ""))
                    in {"coordination_yield", "occupied_cell_clearance_wait"}
                    and str(event.get(key, ""))
                }
        # A shelf-arm egress handshake can require more than one clearance
        # move.  After the first move there may be no immediate collision
        # counterfactual, but the public S_t topology still reserves the arm
        # for the trapped robot.  Do not give the clearing carrier negative
        # progress for faithfully maintaining that causal reservation.
        egress_agent_id = single_lane_egress_agent_id(
            previous_state,
            environment.config,
            goal_positions={
                agent.agent_id: agent.navigation_goal_position
                for agent in previous_state.agents
            },
        )
        if egress_agent_id is not None:
            necessary_clearance_agent_ids.update(
                agent.agent_id
                for agent in previous_state.agents
                if agent.agent_id != egress_agent_id
                and requested_actions.get(agent.agent_id) in MOVE_DELTAS
            )
        necessary_clearance_agents = tuple(
            sorted(necessary_clearance_agent_ids)
        )
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
            loaded_detour_agents=loaded_detour_agents,
            necessary_clearance_agents=necessary_clearance_agents,
            charger_return_cycle_agents=charger_return_cycle_agents,
            starving_task_ids=starving_task_ids,
            unexplained_reversal_agents=unexplained_reversal_agents,
            short_cycle_agents=short_cycle_agents,
            invalid_goal_switch_agents=invalid_goal_switch_agents,
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
        "causal_efficiency_penalty_rewards": dict(zeros),
        "temporal_consistency_penalty_rewards": dict(zeros),
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
