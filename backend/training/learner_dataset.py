"""Actor-visited dataset aggregation for direct-neural warehouse training."""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from env.warehouse.coordination import (
    is_necessary_urgent_charger_clearance,
    stable_coordination_actions,
)
from env.warehouse.environment import (
    WarehouseConfig,
    WarehouseMultiAgentEnv,
    shortest_path_distance,
)
from env.warehouse.mappo import MAPPOPolicy
from env.warehouse.navigation import ACTIONS, all_passable_positions
from env.warehouse.partner_policies import (
    PARTNER_PROFILES,
    participant_surrogate_action,
    robust_partner_robot_two_action,
)
from env.warehouse.policy import MISSION_INTENT_NAMES, independent_actor_input
from env.warehouse.transition_audit import (
    necessary_participant_standoff_clearance,
    necessary_teammate_route_clearance,
)
from env.warehouse.scenarios import (
    apply_charger_commitment_scenario,
    apply_charger_handoff_scenario,
    apply_critical_charger_approach_scenario,
    apply_delivery_goal_clearance_scenario,
    apply_dual_charger_approach_scenario,
    apply_empty_delivery_clearance_scenario,
    apply_head_on_scenario,
    apply_outer_exit_charger_approach_scenario,
    apply_same_target_conflict_scenario,
    apply_task_commitment_scenario,
)


def safe_navigation_teacher_actions(
    environment: WarehouseMultiAgentEnv,
) -> dict[str, str]:
    """Construct offline action labels without touching runtime execution."""

    state = environment.get_state()
    goals = {
        agent.agent_id: goal
        for agent in state.agents
        if (goal := environment._frozen_route_goal(state, agent.agent_id))
        is not None
    }
    return stable_coordination_actions(environment, goal_overrides=goals)


def teacher_mission_intent_label(
    environment: WarehouseMultiAgentEnv,
    state: object,
    agent_id: str,
) -> int:
    """Return an offline target for the Actor's neural mission head."""

    agent = state.by_id(agent_id)
    goal = environment._frozen_route_goal(
        state,
        agent_id,
        prioritize_old_tasks=True,
    )
    if goal == environment.layout.charger_position:
        return MISSION_INTENT_NAMES.index("charge")
    if agent.carrying_task_id is not None:
        task = state.task_by_id(agent.carrying_task_id)
        if goal == task.delivery_position:
            return MISSION_INTENT_NAMES.index("delivery")
    for slot, task in enumerate(sorted(state.tasks, key=lambda item: item.task_id)[:2]):
        if task.status == "available" and goal == task.pickup_position:
            return slot
    return MISSION_INTENT_NAMES.index("wait")


def collect_critical_coordination_dataset(
    environment_config: WarehouseConfig,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect stratified energy, commitment and conflict pretraining rows."""

    if sample_count <= 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            0,
        )
    environment = WarehouseMultiAgentEnv(environment_config)
    rows: list[np.ndarray] = []
    actions: list[int] = []
    intents: list[int] = []
    charge_samples = 0
    case = 0
    while len(rows) < sample_count:
        environment.reset(seed=seed + case)
        kind = case % 11
        scenario_index = case // 11
        if kind == 0:
            apply_head_on_scenario(
                environment,
                reverse=bool(scenario_index % 2),
                variant=scenario_index // 2,
            )
            rollout_steps = 8
        elif kind == 1:
            handoff_profiles = (
                (2.0, 18.0, True, True),
                (12.0, 20.0, True, True),
                (20.0, 24.0, True, True),
                (44.0, 16.0, False, True),
                (58.0, 24.0, True, True),
                (70.0, 28.0, True, False),
                (100.0, 12.0, False, False),
                (36.0, 30.0, False, True),
            )
            (
                occupant_battery,
                queued_battery,
                occupant_carrying,
                queued_carrying,
            ) = handoff_profiles[scenario_index % len(handoff_profiles)]
            apply_charger_handoff_scenario(
                environment,
                occupant_agent_id=environment.agent_ids[scenario_index % 2],
                queued_battery=queued_battery,
                occupant_battery=occupant_battery,
                occupant_carrying=occupant_carrying,
                queued_carrying=queued_carrying,
            )
            rollout_steps = 6
        elif kind == 2:
            apply_same_target_conflict_scenario(
                environment,
                variant=scenario_index,
            )
            rollout_steps = 4
        elif kind == 3:
            apply_critical_charger_approach_scenario(
                environment,
                approaching_agent_id=environment.agent_ids[scenario_index % 2],
                variant=scenario_index,
            )
            rollout_steps = 6
        elif kind == 4:
            apply_delivery_goal_clearance_scenario(
                environment,
                variant=scenario_index,
            )
            rollout_steps = 4
        elif kind == 5:
            apply_charger_commitment_scenario(
                environment,
                agent_id=environment.agent_ids[scenario_index % 2],
                variant=scenario_index,
            )
            rollout_steps = 8
        elif kind == 6:
            apply_task_commitment_scenario(
                environment,
                variant=scenario_index,
            )
            rollout_steps = 12
        elif kind == 7:
            apply_empty_delivery_clearance_scenario(
                environment,
                variant=scenario_index,
            )
            rollout_steps = 4
        elif kind == 8:
            apply_dual_charger_approach_scenario(
                environment,
                variant=scenario_index,
            )
            rollout_steps = 6
        elif kind == 9:
            apply_outer_exit_charger_approach_scenario(
                environment,
                variant=scenario_index,
            )
            rollout_steps = 6
        else:
            # Rehearse the exact fixed-demo departure geometry that previously
            # made the Actor idle at full battery while its teammate approached
            # the charger.  This is an offline weight-update row only: the
            # teacher is never called by deployed action paths.
            environment.reset(seed=42_041)
            departure_state = environment.get_state()
            departure_state.by_id("robot_2").battery = 35.0
            environment.set_state(departure_state)
            rollout_steps = 8
        participant_mode = bool(scenario_index % 2)
        participant_profile = PARTNER_PROFILES[
            (scenario_index // 2) % len(PARTNER_PROFILES)
        ]
        participant_rng = np.random.default_rng(seed + 20_000_000 + case)
        if participant_mode:
            participant_state = environment.get_state()
            participant_state.participant_controlled_agent_id = (
                environment.config.human_agent_id
            )
            environment.set_state(participant_state)
        for _ in range(rollout_steps):
            local = environment.observations()
            state = environment.get_state()
            labels = safe_navigation_teacher_actions(environment)
            if participant_mode:
                labels["robot_2"] = robust_partner_robot_two_action(environment)
            for agent in state.agents:
                if participant_mode and agent.agent_id == "robot_1":
                    continue
                rows.append(
                    independent_actor_input(local[agent.agent_id])
                )
                actions.append(ACTIONS.index(labels[agent.agent_id]))
                intents.append(
                    teacher_mission_intent_label(
                        environment,
                        state,
                        agent.agent_id,
                    )
                )
                charge_samples += int(agent.navigation_goal_kind == "charge")
                if len(rows) >= sample_count:
                    break
            if len(rows) >= sample_count:
                break
            executed = dict(labels)
            if participant_mode:
                executed["robot_1"] = participant_surrogate_action(
                    environment,
                    profile=participant_profile,
                    rng=participant_rng,
                )
            _, _, terminated, truncated, _ = environment.step(executed)
            if terminated or truncated:
                break
        case += 1
    return (
        np.stack(rows),
        np.asarray(actions, dtype=np.int64),
        np.asarray(intents, dtype=np.int64),
        charge_samples,
    )


def best_unilateral_mission_action(
    environment: WarehouseMultiAgentEnv,
    state: object,
    *,
    agent_id: str,
) -> str:
    """Return a causal, collision-free mission correction for supervision.

    The peer baseline is derived only from the same frozen ``S_t`` through the
    deterministic coordination teacher.  This helper deliberately accepts no
    current-frame Actor proposal: conditioning a label on the peer's sampled
    ``a_t`` would give the decentralized Actor an unobservable target and
    recreate the sequential/future-peek dependency forbidden at runtime.  The
    returned action remains an offline label and is never submitted to the
    environment.
    """

    state_only_actions = stable_coordination_actions(environment)
    agent = state.by_id(agent_id)
    teammate = next(
        item for item in state.agents if item.agent_id != agent_id
    )
    if environment._requires_charge(state, agent):
        goal = environment.layout.charger_position
    elif agent.carrying_task_id is not None:
        goal = state.task_by_id(agent.carrying_task_id).delivery_position
    else:
        goal = agent.navigation_goal_position
    available_pickups = {
        task.pickup_position
        for task in state.tasks
        if task.status == "available"
    }
    action_mask = environment.action_masks()[agent_id]
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for action_index, (action, allowed) in enumerate(zip(ACTIONS, action_mask)):
        if allowed <= 0.5:
            continue
        trial_actions = dict(state_only_actions)
        trial_actions[agent_id] = action
        targets, _, invalid, collision, _, _ = environment._resolve_motion(
            state,
            trial_actions,
        )
        if collision or agent_id in invalid:
            continue
        target = targets[agent_id]
        if target == teammate.position:
            # Occupancy belongs to frozen S_t.  Never train the Actor to enter
            # the peer's current cell by assuming that its predicted a_t will
            # vacate the cell; the peer must clear it in an earlier frame.
            continue
        if (
            agent.carrying_task_id is None
            and agent.navigation_goal_kind == "pickup"
            and target in available_pickups
            and target != goal
        ):
            # Do not use a geometrically short action that would irreversibly
            # claim the teammate's currently assigned open task.
            continue
        distance = shortest_path_distance(
            target,
            goal,
            environment.config.map_layout_id,
        )
        # Prefer actual progress over an equally distant WAIT. ACTIONS order
        # remains the final deterministic tie-break for symmetric routes.
        stationary = int(target == agent.position)
        candidates.append(((distance, stationary, action_index), action))
    if not candidates:
        return "WAIT"
    return min(candidates, key=lambda item: item[0])[1]


def configure_learner_state_head_on(
    environment: WarehouseMultiAgentEnv,
    *,
    reverse: bool,
    variant: int = 0,
) -> None:
    """Install one of the two symmetric loaded corridor encounters."""

    apply_head_on_scenario(environment, reverse=reverse, variant=variant)


def collect_learner_state_relabel_dataset(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    sample_count: int,
    seed: int,
    include_teammate_labels: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]
):
    """Label Actor-visited states without submitting expert actions.

    Robot 2 receives counterfactual neural-response examples for every legal
    current robot-1 action.  Proxy-human actions are inputs only; they are not
    used as Actor labels.  All environment transitions use the Actor pair.
    """

    empty_coverage = {
        "head_on_rows": 0,
        "charger_handoff_rows": 0,
        "noisy_teammate_rows": 0,
        "counterfactual_teammate_rows": 0,
        "ordinary_rows": 0,
        "collision_rows": 0,
        "predicted_collision_rows": 0,
        "joint_wait_rows": 0,
        "avoidable_loaded_detour_rows": 0,
        "charger_queue_rows": 0,
        "critical_energy_rows": 0,
        "junction_conflict_rows": 0,
        "delivery_goal_clearance_rows": 0,
        "empty_delivery_clearance_rows": 0,
        "dual_charger_approach_rows": 0,
        "outer_exit_charger_approach_rows": 0,
        "policy_mismatch_rows": 0,
    }
    if sample_count <= 0:
        empty = (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            empty_coverage,
        )
        if include_teammate_labels:
            return (empty[0], empty[1], empty[1].copy(), empty[2], empty[3])
        return empty

    environment = WarehouseMultiAgentEnv(environment_config)
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    targets: list[int] = []
    teammate_targets: list[int] = []
    categories: list[str] = []
    coverage = dict(empty_coverage)
    episode = 0
    while len(observations) < sample_count:
        environment.reset(seed=seed + episode)
        scenario_kind = (
            "head_on",
            "junction_conflict",
            "charger_handoff",
            "critical_approach",
            "noisy_teammate",
            "ordinary",
            "junction_conflict",
            "critical_approach",
            "noisy_teammate",
            "head_on",
            "ordinary",
            "charger_handoff",
            "junction_conflict",
            "noisy_teammate",
            "critical_approach",
            "noisy_teammate",
            "delivery_goal_clearance",
            "delivery_goal_clearance",
            "empty_delivery_clearance",
            "empty_delivery_clearance",
            "dual_charger_approach",
            "dual_charger_approach",
            "outer_exit_charger_approach",
            "outer_exit_charger_approach",
        )[episode % 24]
        head_on = scenario_kind == "head_on"
        junction_conflict = scenario_kind == "junction_conflict"
        charger_handoff = scenario_kind == "charger_handoff"
        critical_approach = scenario_kind == "critical_approach"
        noisy_teammate = scenario_kind == "noisy_teammate"
        delivery_goal_clearance = scenario_kind == "delivery_goal_clearance"
        empty_delivery_clearance = (
            scenario_kind == "empty_delivery_clearance"
        )
        dual_charger_approach = scenario_kind == "dual_charger_approach"
        outer_exit_charger_approach = (
            scenario_kind == "outer_exit_charger_approach"
        )
        if head_on:
            configure_learner_state_head_on(
                environment,
                reverse=bool((episode // 2) % 2),
                variant=episode // 24,
            )
        elif junction_conflict:
            apply_same_target_conflict_scenario(
                environment,
                variant=seed + episode - 1,
            )
        elif charger_handoff:
            handoff_profiles = (
                (2.0, 18.0, True, True),
                (12.0, 20.0, True, True),
                (20.0, 24.0, True, True),
                (44.0, 16.0, False, True),
                (58.0, 24.0, True, True),
                (70.0, 28.0, True, False),
                (100.0, 12.0, False, False),
                (36.0, 30.0, False, True),
            )
            charger_variant = (
                (episode // 24) * 2 + int(episode % 24 > 2)
            ) % len(handoff_profiles)
            (
                occupant_battery,
                queued_battery,
                occupant_carrying,
                queued_carrying,
            ) = handoff_profiles[charger_variant]
            apply_charger_handoff_scenario(
                environment,
                occupant_agent_id=environment.agent_ids[episode % 2],
                queued_battery=queued_battery,
                occupant_battery=occupant_battery,
                occupant_carrying=occupant_carrying,
                queued_carrying=queued_carrying,
            )
        elif critical_approach:
            apply_critical_charger_approach_scenario(
                environment,
                approaching_agent_id=environment.agent_ids[episode % 2],
                variant=seed + episode - 1,
            )
        elif delivery_goal_clearance:
            apply_delivery_goal_clearance_scenario(
                environment,
                variant=seed + episode - 1,
            )
        elif empty_delivery_clearance:
            apply_empty_delivery_clearance_scenario(
                environment,
                variant=seed + episode - 1,
            )
        elif dual_charger_approach:
            apply_dual_charger_approach_scenario(
                environment,
                variant=seed + episode - 1,
            )
        elif outer_exit_charger_approach:
            apply_outer_exit_charger_approach_scenario(
                environment,
                variant=seed + episode - 1,
            )
        else:
            state = environment.get_state()
            if rng.random() < 0.50:
                selected = int(rng.integers(0, len(state.agents)))
                state.agents[selected].battery = float(rng.uniform(15.0, 45.0))
                environment.set_state(state)
        partner_profile = PARTNER_PROFILES[episode % len(PARTNER_PROFILES)]
        partner_rng = np.random.default_rng(seed + 21_000_000 + episode)
        if noisy_teammate:
            participant_state = environment.get_state()
            participant_state.participant_controlled_agent_id = (
                environment.config.human_agent_id
            )
            environment.set_state(participant_state)
        episode += 1
        observations_by_agent = environment.observations()
        repeated_joint_state = 0
        previous_joint_signature: tuple[object, ...] | None = None
        for scenario_step in range(
            min(environment_config.horizon, 64 if head_on else 120)
        ):
            state = environment.get_state()
            participant_overrides: dict[str, str] = {}
            proxy_override = False
            if noisy_teammate:
                participant_overrides["robot_1"] = participant_surrogate_action(
                    environment,
                    profile=partner_profile,
                    rng=partner_rng,
                )
                proxy_override = True
            proposed, _ = policy.act(
                observations_by_agent,
                environment.global_state(),
                deterministic=True,
            )
            # A noisy proxy participant is applied only after both independent
            # Actor outputs have been computed from the frozen state.
            proposed.update(participant_overrides)
            teacher_actions = stable_coordination_actions(environment)
            if noisy_teammate:
                teacher_actions["robot_2"] = robust_partner_robot_two_action(
                    environment,
                    preferred_action=proposed["robot_2"],
                )
            (
                predicted_targets,
                _,
                _,
                predicted_collision,
                _,
                _,
            ) = environment._resolve_motion(state, proposed)

            def ineffective_stall(
                actions: Mapping[str, str],
                targets_by_agent: Mapping[str, tuple[int, int]],
            ) -> bool:
                all_stationary = all(
                    targets_by_agent[agent.agent_id] == agent.position
                    for agent in state.agents
                )
                useful_charge = any(
                    actions[agent.agent_id] == "WAIT"
                    and agent.position == environment.layout.charger_position
                    and agent.battery < 100.0
                    for agent in state.agents
                )
                return bool(all_stationary and not useful_charge)

            predicted_stall = ineffective_stall(proposed, predicted_targets)

            def avoidable_loaded_detour_agents(
                actions: Mapping[str, str],
                targets_by_agent: Mapping[str, tuple[int, int]],
                *,
                eligible_agent_ids: set[str] | None = None,
            ) -> tuple[str, ...]:
                offenders: list[str] = []
                for agent in state.agents:
                    if (
                        eligible_agent_ids is not None
                        and agent.agent_id not in eligible_agent_ids
                    ):
                        continue
                    if (
                        agent.carrying_task_id is None
                        or agent.navigation_goal_kind != "delivery"
                        or environment._requires_charge(state, agent)
                    ):
                        continue
                    current_distance = shortest_path_distance(
                        agent.position,
                        agent.navigation_goal_position,
                        environment.config.map_layout_id,
                    )
                    next_distance = shortest_path_distance(
                        targets_by_agent[agent.agent_id],
                        agent.navigation_goal_position,
                        environment.config.map_layout_id,
                    )
                    if next_distance <= current_distance:
                        continue
                    held_actions = dict(actions)
                    held_actions[agent.agent_id] = "WAIT"
                    held_collision = environment._resolve_motion(
                        state,
                        held_actions,
                    )[3]
                    if (
                        not held_collision
                        and not is_necessary_urgent_charger_clearance(
                            environment,
                            state,
                            agent,
                        )
                        and not necessary_teammate_route_clearance(
                            environment,
                            state,
                            agent,
                        )
                    ):
                        offenders.append(agent.agent_id)
                return tuple(offenders)

            predicted_detour_agents = avoidable_loaded_detour_agents(
                proposed,
                predicted_targets,
            )
            predicted_detour_agent_ids = set(predicted_detour_agents)
            energy_state = bool(
                charger_handoff
                or critical_approach
                or dual_charger_approach
                or outer_exit_charger_approach
                or any(
                    agent.navigation_goal_kind == "charge"
                    or agent.battery <= 30.0
                    for agent in state.agents
                )
            )
            charger_queue_state = bool(
                dual_charger_approach
                or outer_exit_charger_approach
                or (
                    any(
                    agent.position == environment.layout.charger_position
                    for agent in state.agents
                    )
                    and any(
                        agent.position != environment.layout.charger_position
                        and agent.navigation_goal_kind == "charge"
                        for agent in state.agents
                    )
                )
            )
            critical_energy_state = any(
                agent.navigation_goal_kind == "charge"
                and agent.battery <= 16.0
                for agent in state.agents
            )
            junction_conflict_state = bool(
                junction_conflict and scenario_step < 4
            )

            def append_row(
                row: np.ndarray,
                label: str,
                *,
                teammate_label: str | None,
                actor_action: str,
                row_collision: bool,
                row_stall: bool,
                row_loaded_detour: bool,
                counterfactual: bool,
            ) -> None:
                if len(observations) >= sample_count:
                    return
                if row_collision or state.last_robot_collision_event:
                    category = "collision"
                elif row_stall:
                    category = "joint_wait"
                elif row_loaded_detour:
                    category = "loaded_detour"
                elif charger_queue_state:
                    category = "charger_queue"
                elif critical_energy_state:
                    category = "critical_energy"
                elif junction_conflict_state:
                    category = "junction_conflict"
                elif delivery_goal_clearance or empty_delivery_clearance:
                    # Reuse the strongly supervised junction-conflict replay
                    # bucket.  The dedicated coverage counter below still
                    # distinguishes these follow-through states for audits.
                    category = "junction_conflict"
                elif noisy_teammate or counterfactual:
                    category = "teammate_response"
                elif energy_state:
                    category = "energy"
                elif head_on:
                    category = "head_on"
                else:
                    category = "ordinary"
                observations.append(row)
                targets.append(ACTIONS.index(label))
                teammate_targets.append(
                    ACTIONS.index(teammate_label)
                    if teammate_label in ACTIONS
                    else -1
                )
                categories.append(category)
                coverage["head_on_rows"] += int(head_on)
                coverage["charger_handoff_rows"] += int(charger_handoff)
                coverage["noisy_teammate_rows"] += int(noisy_teammate)
                coverage["counterfactual_teammate_rows"] += int(counterfactual)
                coverage["ordinary_rows"] += int(
                    not head_on
                    and not charger_handoff
                    and not noisy_teammate
                    and not delivery_goal_clearance
                    and not empty_delivery_clearance
                    and not dual_charger_approach
                    and not outer_exit_charger_approach
                )
                coverage["collision_rows"] += int(
                    state.last_robot_collision_event
                )
                coverage["predicted_collision_rows"] += int(row_collision)
                coverage["joint_wait_rows"] += int(row_stall)
                coverage["avoidable_loaded_detour_rows"] += int(
                    row_loaded_detour
                )
                coverage["charger_queue_rows"] += int(charger_queue_state)
                coverage["critical_energy_rows"] += int(
                    critical_energy_state
                )
                coverage["junction_conflict_rows"] += int(
                    junction_conflict_state
                )
                coverage["delivery_goal_clearance_rows"] += int(
                    delivery_goal_clearance
                )
                coverage["empty_delivery_clearance_rows"] += int(
                    empty_delivery_clearance
                )
                coverage["dual_charger_approach_rows"] += int(
                    dual_charger_approach
                )
                coverage["outer_exit_charger_approach_rows"] += int(
                    outer_exit_charger_approach
                )
                coverage["policy_mismatch_rows"] += int(
                    actor_action != label
                )

            robot_one_label: str | None = None
            robot_one_correctable_detour = False
            if not proxy_override:
                robot_one_label = teacher_actions["robot_1"]
                robot_one_correctable_detour = False
                if "robot_1" in predicted_detour_agent_ids:
                    detour_correction = best_unilateral_mission_action(
                        environment,
                        state,
                        agent_id="robot_1",
                    )
                    robot_one_label = detour_correction
                    robot_one_correctable_detour = True
                elif predicted_stall and proposed["robot_1"] == "WAIT":
                    unilateral_escape = best_unilateral_mission_action(
                        environment,
                        state,
                        agent_id="robot_1",
                    )
                    if unilateral_escape != "WAIT":
                        robot_one_label = unilateral_escape
            # Robot 2 receives one label for S_t. The label and the Actor row
            # are deliberately not conditioned on robot 1's current action.
            robot_two_label = teacher_actions["robot_2"]
            robot_two_correctable_detour = False
            if noisy_teammate:
                # Keep the S_t-only robust participant label authoritative.
                # Applying the generic detour/stall correction afterwards
                # used to replace it with actions that were unsafe for a
                # supported participant profile.
                pass
            elif "robot_2" in predicted_detour_agent_ids:
                detour_correction = best_unilateral_mission_action(
                    environment,
                    state,
                    agent_id="robot_2",
                )
                robot_two_label = detour_correction
                robot_two_correctable_detour = True
            elif predicted_stall and proposed["robot_2"] == "WAIT":
                unilateral_escape = best_unilateral_mission_action(
                    environment,
                    state,
                    agent_id="robot_2",
                )
                if unilateral_escape != "WAIT":
                    robot_two_label = unilateral_escape
            if robot_one_label is not None:
                append_row(
                    independent_actor_input(observations_by_agent["robot_1"]),
                    robot_one_label,
                    teammate_label=robot_two_label,
                    actor_action=proposed["robot_1"],
                    row_collision=bool(predicted_collision),
                    row_stall=bool(predicted_stall),
                    row_loaded_detour=robot_one_correctable_detour,
                    counterfactual=False,
                )
            append_row(
                independent_actor_input(observations_by_agent["robot_2"]),
                robot_two_label,
                teammate_label=(
                    robot_one_label
                    if robot_one_label is not None
                    else participant_overrides.get("robot_1")
                ),
                actor_action=proposed["robot_2"],
                row_collision=bool(predicted_collision),
                row_stall=bool(predicted_stall),
                row_loaded_detour=robot_two_correctable_detour,
                counterfactual=False,
            )
            if len(observations) >= sample_count:
                break
            joint_signature = tuple(
                (
                    agent.position,
                    round(agent.battery, 3),
                    agent.carrying_task_id,
                    agent.navigation_goal_kind,
                    agent.navigation_goal_position,
                    proposed[agent.agent_id],
                )
                for agent in state.agents
            )
            if joint_signature == previous_joint_signature:
                repeated_joint_state += 1
            else:
                repeated_joint_state = 0
                previous_joint_signature = joint_signature
            observations_by_agent, _, terminated, truncated, _ = environment.step(
                proposed
            )
            if terminated or truncated or repeated_joint_state >= 3:
                break
    result = (
        np.stack(observations),
        np.asarray(targets, dtype=np.int64),
        np.asarray(categories, dtype="<U32"),
        coverage,
    )
    if include_teammate_labels:
        return (
            result[0],
            result[1],
            np.asarray(teammate_targets, dtype=np.int64),
            result[2],
            result[3],
        )
    return result


def collect_loaded_detour_correction_dataset(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    sample_count: int,
    maximum_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Mine rare loaded detours from unmodified deterministic Actor rollouts.

    The general learner-state collector records every visited state, so a
    roughly one-in-a-thousand loaded detour can be absent from an entire
    relabel round.  This pass scans a wider, disjoint seed range and stores
    only those rare mistakes.  Environment transitions still receive the
    Actor's joint action unchanged; the unilateral mission action is computed
    afterwards as an offline supervised label and is never executed.
    """

    requested = max(0, int(sample_count))
    episode_limit = max(0, int(maximum_episodes))
    empty = (
        np.empty((0, 0), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype="<U32"),
        {
            "detour_search_episodes": 0,
            "detour_search_actor_steps": 0,
            "detour_correction_rows": 0,
            "detour_expert_actions_submitted": 0,
        },
    )
    if requested == 0 or episode_limit == 0:
        return empty

    environment = WarehouseMultiAgentEnv(environment_config)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    actor_steps = 0
    episodes_scanned = 0
    for episode in range(episode_limit):
        observations, _ = environment.reset(seed=int(seed) + episode)
        episodes_scanned += 1
        while len(rows) < requested:
            state = environment.get_state()
            actions, _ = policy.act(
                observations,
                environment.global_state(),
                deterministic=True,
            )
            actor_steps += 1
            targets = environment._resolve_motion(state, actions)[0]
            detouring_agent_ids: set[str] = set()
            for agent in state.agents:
                if (
                    agent.carrying_task_id is None
                    or agent.navigation_goal_kind != "delivery"
                    or environment._requires_charge(state, agent)
                ):
                    continue
                distance_before = shortest_path_distance(
                    agent.position,
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
                distance_after = shortest_path_distance(
                    targets[agent.agent_id],
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
                if distance_after <= distance_before:
                    continue
                held_actions = dict(actions)
                held_actions[agent.agent_id] = "WAIT"
                if (
                    environment._resolve_motion(state, held_actions)[3]
                    or is_necessary_urgent_charger_clearance(
                        environment,
                        state,
                        agent,
                    )
                    or necessary_teammate_route_clearance(
                        environment,
                        state,
                        agent,
                    )
                    or necessary_participant_standoff_clearance(
                        environment,
                        state,
                        agent,
                        candidate_action=actions[agent.agent_id],
                    )
                ):
                    continue
                detouring_agent_ids.add(agent.agent_id)

            if detouring_agent_ids:
                # Correct the *joint* response.  A unilateral WAIT label can
                # turn A->B->A oscillation into WAIT/WAIT; the paired teacher
                # rows instead teach both independent decisions from the same
                # corrected pre-move state. The
                # pair remains offline supervision and is never executed.
                correction_actions = stable_coordination_actions(environment)
                if not environment._resolve_motion(
                    state,
                    correction_actions,
                )[3]:
                    for agent in sorted(
                        state.agents,
                        key=lambda item: item.agent_id,
                    ):
                        correction = correction_actions[agent.agent_id]
                        if agent.agent_id in detouring_agent_ids:
                            correction_targets = environment._resolve_motion(
                                state,
                                correction_actions,
                            )[0]
                            distance_before = shortest_path_distance(
                                agent.position,
                                agent.navigation_goal_position,
                                environment.config.map_layout_id,
                            )
                            corrected_distance = shortest_path_distance(
                                correction_targets[agent.agent_id],
                                agent.navigation_goal_position,
                                environment.config.map_layout_id,
                            )
                            if corrected_distance > distance_before:
                                unilateral = best_unilateral_mission_action(
                                    environment,
                                    state,
                                    agent_id=agent.agent_id,
                                )
                                if unilateral != correction:
                                    correction = unilateral
                                    correction_actions = dict(correction_actions)
                                    correction_actions[agent.agent_id] = unilateral
                        rows.append(
                            independent_actor_input(observations[agent.agent_id])
                        )
                        labels.append(ACTIONS.index(correction))
                        if len(rows) >= requested:
                            break
            # The transition contract is the central invariant of this
            # miner: only the original Actor action reaches the environment.
            observations, _, terminated, truncated, _ = environment.step(actions)
            if terminated or truncated:
                break
        if len(rows) >= requested:
            break

    if not rows:
        return (
            empty[0],
            empty[1],
            empty[2],
            {
                "detour_search_episodes": episodes_scanned,
                "detour_search_actor_steps": actor_steps,
                "detour_correction_rows": 0,
                "detour_expert_actions_submitted": 0,
            },
        )
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.full(len(rows), "loaded_detour", dtype="<U32"),
        {
            "detour_search_episodes": episodes_scanned,
            "detour_search_actor_steps": actor_steps,
            "detour_correction_rows": len(rows),
            "detour_expert_actions_submitted": 0,
        },
    )


def collect_actor_collision_correction_dataset(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    sample_count: int,
    maximum_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Mine unique Actor collision states for offline paired supervision.

    Half of the scan uses ordinary deterministic rollouts and half starts from
    the broad same-target curriculum.  In both cases the environment receives
    the Actor's original joint action unchanged.  A collision-free pair is
    computed only after observing the Actor proposal and is stored as two
    independent pre-move-state training rows; it is never executed here.
    """

    requested = max(0, int(sample_count))
    episode_limit = max(0, int(maximum_episodes))
    empty_coverage = {
        "collision_search_episodes": 0,
        "collision_search_actor_steps": 0,
        "unique_predicted_collision_states": 0,
        "collision_correction_rows": 0,
        "collision_expert_actions_submitted": 0,
    }
    if requested == 0 or episode_limit == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            empty_coverage,
        )

    environment = WarehouseMultiAgentEnv(environment_config)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    seen: set[tuple[object, ...]] = set()
    actor_steps = 0
    episodes_scanned = 0
    for episode in range(episode_limit):
        observations, _ = environment.reset(seed=int(seed) + episode)
        if episode % 2 == 0:
            apply_same_target_conflict_scenario(
                environment,
                variant=int(seed) + episode,
            )
            observations = environment.observations()
        episodes_scanned += 1
        repeated_signature: tuple[object, ...] | None = None
        repeated_count = 0
        while len(rows) < requested:
            state = environment.get_state()
            actor_actions, _ = policy.act(
                observations,
                environment.global_state(),
                deterministic=True,
            )
            actor_steps += 1
            _, _, _, predicted_collision, _, _ = environment._resolve_motion(
                state,
                actor_actions,
            )
            state_signature = tuple(
                (
                    agent.agent_id,
                    agent.position,
                    round(float(agent.battery), 3),
                    agent.carrying_task_id,
                    agent.navigation_goal_kind,
                    agent.navigation_goal_position,
                    actor_actions[agent.agent_id],
                )
                for agent in sorted(state.agents, key=lambda item: item.agent_id)
            )
            if predicted_collision and state_signature not in seen:
                correction_actions = stable_coordination_actions(environment)
                correction_collision = environment._resolve_motion(
                    state,
                    correction_actions,
                )[3]
                if not correction_collision:
                    seen.add(state_signature)
                    for agent in sorted(
                        state.agents,
                        key=lambda item: item.agent_id,
                    ):
                        rows.append(
                            independent_actor_input(observations[agent.agent_id])
                        )
                        labels.append(
                            ACTIONS.index(correction_actions[agent.agent_id])
                        )
                        if len(rows) >= requested:
                            break

            if state_signature == repeated_signature:
                repeated_count += 1
            else:
                repeated_signature = state_signature
                repeated_count = 0
            # This is the critical contract: collision labels are offline.
            # The transition still receives exactly the Actor proposal.
            observations, _, terminated, truncated, _ = environment.step(
                actor_actions
            )
            if terminated or truncated or repeated_count >= 3:
                break
        if len(rows) >= requested:
            break

    coverage = {
        "collision_search_episodes": episodes_scanned,
        "collision_search_actor_steps": actor_steps,
        "unique_predicted_collision_states": len(seen),
        "collision_correction_rows": len(rows),
        "collision_expert_actions_submitted": 0,
    }
    if not rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            coverage,
        )
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.full(len(rows), "collision", dtype="<U32"),
        coverage,
    )


def collect_actor_commitment_failure_dataset(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    charger_cycle_samples: int,
    task_starvation_samples: int,
    maximum_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Mine causal Actor states for energy loops and old-task starvation.

    The environment always receives the deterministic Actor pair.  Teacher
    actions are computed before the transition and retained only as offline
    supervised labels.  When a failure becomes observable, the preceding
    causal window is copied into a dedicated replay category so a handful of
    critical errors cannot disappear inside aggregate imitation accuracy.
    """

    cycle_limit = max(0, int(charger_cycle_samples))
    starvation_limit = max(0, int(task_starvation_samples))
    episode_limit = max(0, int(maximum_episodes))
    coverage = {
        "commitment_search_episodes": 0,
        "commitment_search_actor_steps": 0,
        "premature_departures_found": 0,
        "charger_return_cycles_found": 0,
        "unnecessary_charge_waits_found": 0,
        "starving_tasks_found": 0,
        "charger_cycle_correction_rows": 0,
        "task_starvation_correction_rows": 0,
        "commitment_expert_actions_submitted": 0,
    }
    if episode_limit == 0 or (cycle_limit == 0 and starvation_limit == 0):
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            coverage,
        )

    environment = WarehouseMultiAgentEnv(environment_config)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    categories: list[str] = []
    # Keep the complete causal horizon for a task that can first become
    # starving after forty frames.  Each row also records the Actor label and
    # the teacher's task target, so replay contains the *decision that caused*
    # a failure rather than ten mostly-correct states after it was inevitable.
    history: deque[dict[str, object]] = deque(maxlen=45)
    seen: set[tuple[str, bytes, int]] = set()

    def category_count(category: str) -> int:
        return categories.count(category)

    def append_window(
        entries: tuple[dict[str, object], ...],
        category: str,
        limit: int,
        *,
        target_agent_ids: set[str] | None = None,
        starving_task_ids: set[str] | None = None,
    ) -> None:
        for entry in entries:
            if starving_task_ids is not None and not (
                starving_task_ids
                & set(entry["available_task_ids"])
            ):
                continue
            for item in entry["rows"]:
                agent_id, row, label, actor_label, empty, teacher_task_id = item
                if category_count(category) >= limit:
                    return
                if target_agent_ids is not None and agent_id not in target_agent_ids:
                    continue
                # Only disagreements are causal correction samples. Copying
                # every correct step made the rare category report 100%
                # accuracy while leaving the actual departure/retarget error
                # untouched on the next distribution.
                if int(actor_label) == int(label):
                    continue
                # A masked Actor cannot learn an action outside the static
                # support encoded in this exact observation.  Keep failure
                # mining defensive even if a future teacher branch regresses.
                if float(row[-len(ACTIONS) + int(label)]) <= 0.5:
                    continue
                if category == "task_starvation":
                    if not bool(empty):
                        continue
                    if (
                        starving_task_ids is not None
                        and teacher_task_id not in starving_task_ids
                    ):
                        continue
                key = (category, row.tobytes(), int(label))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                labels.append(int(label))
                categories.append(category)

    for episode in range(episode_limit):
        observations, _ = environment.reset(seed=int(seed) + episode)
        history.clear()
        coverage["commitment_search_episodes"] += 1
        while (
            category_count("charger_cycle") < cycle_limit
            or category_count("task_starvation") < starvation_limit
        ):
            state = environment.get_state()
            actor_actions, _ = policy.act(
                observations,
                environment.global_state(),
                deterministic=True,
            )
            teacher_actions = stable_coordination_actions(environment)
            teacher_task_by_agent: dict[str, str | None] = {}
            for agent_id in environment.agent_ids:
                goal = environment._frozen_route_goal(
                    state,
                    agent_id,
                    prioritize_old_tasks=True,
                )
                teacher_task_by_agent[agent_id] = next(
                    (
                        task.task_id
                        for task in state.tasks
                        if task.status == "available"
                        and task.pickup_position == goal
                    ),
                    None,
                )
            current_entry: dict[str, object] = {
                "available_task_ids": tuple(
                    task.task_id
                    for task in state.tasks
                    if task.status == "available"
                ),
                "rows": tuple(
                    (
                        agent_id,
                        independent_actor_input(observations[agent_id]),
                        ACTIONS.index(teacher_actions[agent_id]),
                        ACTIONS.index(actor_actions[agent_id]),
                        state.by_id(agent_id).carrying_task_id is None,
                        teacher_task_by_agent[agent_id],
                    )
                    for agent_id in environment.agent_ids
                ),
            }
            history.append(current_entry)
            coverage["commitment_search_actor_steps"] += 1

            unnecessary_wait = any(
                actor_actions[agent.agent_id] == "WAIT"
                and agent.position == environment.layout.charger_position
                and not environment._requires_charge(
                    state,
                    agent,
                    position=environment.layout.charger_position,
                )
                for agent in state.agents
            )
            if unnecessary_wait and category_count("charger_cycle") < cycle_limit:
                coverage["unnecessary_charge_waits_found"] += 1
                append_window((current_entry,), "charger_cycle", cycle_limit)

            observations, _, terminated, truncated, info = environment.step(
                actor_actions
            )
            premature_events = tuple(
                event
                for event in info.get("energy_events", ())
                if event.get("event") == "charger_departure"
                and bool(event.get("premature", False))
            )
            cycle_events = tuple(
                event
                for event in info.get("energy_events", ())
                if event.get("event") == "charger_return_cycle"
            )
            if premature_events:
                coverage["premature_departures_found"] += len(
                    premature_events
                )
                append_window(
                    (current_entry,),
                    "charger_cycle",
                    cycle_limit,
                    target_agent_ids={
                        str(event["agent_id"])
                        for event in premature_events
                    },
                )
            if cycle_events:
                coverage["charger_return_cycles_found"] += len(cycle_events)
                append_window(
                    tuple(history)[-7:],
                    "charger_cycle",
                    cycle_limit,
                    target_agent_ids={
                        str(event["agent_id"])
                        for event in cycle_events
                    },
                )

            starving = tuple(info.get("starving_task_ids", ()))
            near_starvation = any(
                int(age)
                >= max(1, environment.config.reward.task_age_priority_horizon - 4)
                for age in info.get("available_task_ages", {}).values()
            )
            if (
                (starving or near_starvation)
                and category_count("task_starvation") < starvation_limit
            ):
                near_starving_ids = {
                    task_id
                    for task_id, age in info.get(
                        "available_task_ages", {}
                    ).items()
                    if int(age)
                    >= max(
                        1,
                        environment.config.reward.task_age_priority_horizon - 4,
                    )
                }
                coverage["starving_tasks_found"] += max(
                    len(starving),
                    len(near_starving_ids),
                )
                append_window(
                    tuple(history),
                    "task_starvation",
                    starvation_limit,
                    starving_task_ids=set(starving) or near_starving_ids,
                )
            if terminated or truncated:
                break
        if (
            category_count("charger_cycle") >= cycle_limit
            and category_count("task_starvation") >= starvation_limit
        ):
            break

    coverage["charger_cycle_correction_rows"] = category_count(
        "charger_cycle"
    )
    coverage["task_starvation_correction_rows"] = category_count(
        "task_starvation"
    )
    if not rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            coverage,
        )
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(categories, dtype="<U32"),
        coverage,
    )


def collect_commitment_curriculum_dataset(
    environment_config: WarehouseConfig,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Create diverse offline energy/old-task commitment supervision.

    Unlike failure mining, this curriculum does not wait for a rare mistake.
    It parameterizes charger histories and shared-task ages across independently
    sampled task geometry.  Teacher actions advance only this private dataset
    environment; the resulting labels are never a deployed controller.
    """

    requested = max(0, int(sample_count))
    coverage = {
        "commitment_curriculum_rows": 0,
        "charger_commitment_curriculum_rows": 0,
        "task_starvation_curriculum_rows": 0,
        "commitment_curriculum_expert_actions_submitted_to_runtime": 0,
    }
    if requested == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            coverage,
        )

    environment = WarehouseMultiAgentEnv(environment_config)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    categories: list[str] = []
    episode = 0
    passable = tuple(
        position
        for position in all_passable_positions(
            environment_config.map_layout_id
        )
        if position != environment.layout.charger_position
    )
    while len(rows) < requested:
        observations, _ = environment.reset(seed=int(seed) + episode)
        category = "charger_cycle" if episode % 2 == 0 else "task_starvation"
        if category == "charger_cycle":
            apply_charger_commitment_scenario(
                environment,
                agent_id=environment.agent_ids[(episode // 2) % 2],
                variant=episode // 2,
            )
            # The scenario mutates S_t after reset.  Supervision must pair the
            # teacher label with an observation regenerated from that exact
            # same state; retaining reset's observation leaks a different
            # time slice into the training row.
            observations = environment.observations()
            rollout_steps = 10
        else:
            state = environment.get_state()
            state.frame = 36 + (episode % 25)
            old_task, new_task = sorted(
                state.tasks,
                key=lambda item: item.task_id,
            )
            old_task.created_frame = 0
            new_task.created_frame = max(0, state.frame - 4)
            for task in (old_task, new_task):
                task.status = "available"
                task.carrier_agent_id = None
                task.claimed_frame = None
            rng = np.random.default_rng(int(seed) + 1_000_000 + episode)
            excluded = {
                old_task.pickup_position,
                new_task.pickup_position,
            }
            positions = [item for item in passable if item not in excluded]
            selected = rng.choice(len(positions), size=2, replace=False)
            for index, agent in enumerate(
                sorted(state.agents, key=lambda item: item.agent_id)
            ):
                agent.position = positions[int(selected[index])]
                agent.battery = float(54 + (episode * 7 + index * 11) % 47)
                agent.carrying_task_id = None
                # Half the variants start with a neural commitment to the
                # newer task. The Actor must learn to retarget from age and
                # route evidence rather than having the environment clear it.
                agent.route_commitment_task_id = (
                    new_task.task_id
                    if (episode + index) % 3 == 0
                    else None
                )
            environment.set_state(state)
            observations = environment.observations()
            rollout_steps = 14
        episode += 1

        for _ in range(rollout_steps):
            state = environment.get_state()
            teacher_actions = stable_coordination_actions(environment)
            for agent_id in environment.agent_ids:
                rows.append(
                    independent_actor_input(observations[agent_id])
                )
                labels.append(ACTIONS.index(teacher_actions[agent_id]))
                categories.append(category)
                if len(rows) >= requested:
                    break
            if len(rows) >= requested:
                break
            observations, _, terminated, truncated, _ = environment.step(
                teacher_actions
            )
            if terminated or truncated:
                break

    coverage["commitment_curriculum_rows"] = len(rows)
    coverage["charger_commitment_curriculum_rows"] = categories.count(
        "charger_cycle"
    )
    coverage["task_starvation_curriculum_rows"] = categories.count(
        "task_starvation"
    )
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(categories, dtype="<U32"),
        coverage,
    )
