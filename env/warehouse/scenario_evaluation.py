"""Closed-loop release evaluations for compact warehouse scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .decision_protocol import distribution_decision_metadata
from .energy_management import charger_handoff_clearance_action
from .environment import WarehouseConfig, WarehouseMultiAgentEnv
from .policy import MAPPOPolicy
from .scenarios import (
    apply_charger_handoff_scenario,
    apply_dual_charger_approach_scenario,
    apply_empty_delivery_clearance_scenario,
    apply_outer_exit_charger_approach_scenario,
)


_OCCUPIED_CHARGER_PROFILES = (
    # occupant battery, queued battery, occupant carrying, queued carrying,
    # expected charger priority
    (58.0, 36.0, True, True, "waiter"),
    (70.0, 28.0, True, False, "waiter"),
    (44.0, 16.0, False, True, "waiter"),
    (100.0, 12.0, False, False, "waiter"),
    (20.0, 24.0, True, True, "occupant"),
    (12.0, 20.0, True, True, "occupant"),
    (2.0, 18.0, True, True, "occupant"),
    (30.0, 36.0, False, True, "occupant"),
)


def _configure_valid_occupied_charger_case(
    environment: WarehouseMultiAgentEnv,
    *,
    episode: int,
) -> tuple[str, str, str, str | None, int]:
    """Create one genuine frozen charger-priority case.

    Random task endpoints change the exact energy needed for a mission.  The
    profile therefore starts at the requested battery and, only when needed,
    lowers the queued robot in two-unit increments until the state really is
    a charger queue.  The adjustment is reported by the evaluator rather than
    silently treating a non-charging teammate as a handoff success.
    """

    profile = _OCCUPIED_CHARGER_PROFILES[episode % len(_OCCUPIED_CHARGER_PROFILES)]
    occupant_battery, queued_battery, occupant_carrying, queued_carrying, priority = profile
    occupant_id = environment.agent_ids[(episode // len(_OCCUPIED_CHARGER_PROFILES)) % 2]
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id=occupant_id,
        occupant_battery=occupant_battery,
        queued_battery=queued_battery,
        occupant_carrying=occupant_carrying,
        queued_carrying=queued_carrying,
    )
    queued_id = next(agent_id for agent_id in environment.agent_ids if agent_id != occupant_id)
    adjustments = 0
    for _ in range(16):
        state = environment.get_state()
        occupant = state.by_id(occupant_id)
        waiter = state.by_id(queued_id)
        action = charger_handoff_clearance_action(
            environment,
            state,
            occupant,
            waiter,
        )
        if priority == "waiter" and action is not None:
            return occupant_id, queued_id, priority, action, adjustments
        if (
            priority == "occupant"
            and environment._requires_charge(state, occupant)
            and environment._requires_charge(state, waiter)
            and action is None
        ):
            return occupant_id, queued_id, priority, None, adjustments
        if priority == "waiter":
            waiter.battery = max(4.0, waiter.battery - 2.0)
        else:
            occupant.battery = max(2.0, occupant.battery - 2.0)
            waiter.battery = max(occupant.battery + 2.0, waiter.battery - 2.0)
        environment.set_state(state)
        adjustments += 1
    raise RuntimeError("Could not construct a genuine occupied-charger priority case.")


def evaluate_occupied_charger_handoff_scenarios(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Require causal handoff or retention at an already occupied charger."""

    successes = collisions = shutdowns = deadlocks = priority_violations = 0
    adjusted_profiles = 0
    completion_steps: list[int] = []
    failure_details: list[dict[str, Any]] = []
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        environment.reset(seed=seed + episode)
        (
            occupant_id,
            waiter_id,
            priority,
            _clearance_action,
            adjustments,
        ) = _configure_valid_occupied_charger_case(environment, episode=episode)
        adjusted_profiles += int(adjustments > 0)
        inference = policy.fork_for_inference(seed=seed + episode + 29_000_000)
        observations = environment.observations()
        first_step_valid = False
        first_actions: dict[str, str] = {}
        first_charged_agent_id: str | None = None
        collided = shutdown = completed = False
        wait_streak = 0
        for step in range(min(16, environment_config.horizon)):
            decision_state = environment.get_state()
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=True,
                decision_key=(decision_state.episode_id, decision_state.frame),
            )
            if step == 0:
                first_actions = dict(actions)
                _, _, invalid, collision, _, _ = environment._resolve_motion(
                    decision_state,
                    actions,
                )
                causal_handoff = bool(
                    actions[occupant_id] in {"LEFT", "RIGHT"}
                    and occupant_id not in invalid
                    and not collision
                )
                first_step_valid = bool(
                    actions[waiter_id] == "WAIT"
                    and (
                        causal_handoff
                        if priority == "waiter"
                        else actions[occupant_id] == "WAIT"
                    )
                )
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="formal_occupied_charger_handoff_actor",
                ),
            )
            collided = collided or bool(info.get("robot_collision_event", False))
            shutdown = shutdown or bool(info.get("shutdowns", ()))
            charged = tuple(info.get("charger_energy_gained_by_agent", {}))
            if charged and first_charged_agent_id is None:
                first_charged_agent_id = str(charged[0])
            expected_first = waiter_id if priority == "waiter" else occupant_id
            completed = first_charged_agent_id == expected_first
            wait_streak = int(environment.get_state().ineffective_joint_wait_streak)
            if completed or collided or shutdown or wait_streak >= 8 or terminated or truncated:
                if completed:
                    completion_steps.append(step + 1)
                break
        success = bool(
            first_step_valid
            and completed
            and not collided
            and not shutdown
            and wait_streak < 8
        )
        successes += int(success)
        collisions += int(collided)
        shutdowns += int(shutdown)
        deadlocks += int(wait_streak >= 8)
        priority_violations += int(not first_step_valid or not completed)
        if not success:
            failure_details.append(
                {
                    "seed": int(seed + episode),
                    "episode": int(episode),
                    "occupant_agent_id": occupant_id,
                    "waiter_agent_id": waiter_id,
                    "expected_priority": priority,
                    "first_actions": first_actions,
                    "first_step_valid": first_step_valid,
                    "completed": completed,
                    "first_charged_agent_id": first_charged_agent_id,
                    "collided": collided,
                    "shutdown": shutdown,
                    "deadlocked": wait_streak >= 8,
                    "profile_adjustments": adjustments,
                }
            )
    count = max(1, episodes)
    return {
        "episodes": float(episodes),
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "shutdown_rate": shutdowns / count,
        "deadlock_rate": deadlocks / count,
        "priority_violation_rate": priority_violations / count,
        "adjusted_profile_rate": adjusted_profiles / count,
        "mean_completion_steps": (
            float(np.mean(completion_steps)) if completion_steps else 0.0
        ),
        "failure_seeds": [item["seed"] for item in failure_details],
        "failure_details": failure_details,
    }


def evaluate_empty_delivery_clearance_scenarios(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Require an empty occupant to clear B before its loaded peer enters."""

    successes = 0
    collisions = 0
    deadlocks = 0
    completion_steps: list[int] = []
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        environment.reset(seed=seed + episode)
        apply_empty_delivery_clearance_scenario(environment, variant=episode)
        initial_state = environment.get_state()
        loaded = next(
            agent
            for agent in initial_state.agents
            if agent.carrying_task_id is not None
        )
        task_id = str(loaded.carrying_task_id)
        delivery_position = initial_state.task_by_id(task_id).delivery_position
        occupant = next(
            agent
            for agent in initial_state.agents
            if agent.agent_id != loaded.agent_id
        )
        inference = policy.fork_for_inference(seed=seed + episode + 29_000_000)
        observations = environment.observations()
        wait_streak = int(initial_state.ineffective_joint_wait_streak)
        collided = False
        completed = False
        occupant_cleared_first = False
        for step in range(min(16, environment_config.horizon)):
            decision_state = environment.get_state()
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=True,
                decision_key=(decision_state.episode_id, decision_state.frame),
            )
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="formal_empty_delivery_clearance_actor",
                ),
            )
            next_state = environment.get_state()
            collided = collided or bool(info.get("robot_collision_event", False))
            occupant_cleared_first = occupant_cleared_first or (
                next_state.by_id(occupant.agent_id).position != delivery_position
                and next_state.by_id(loaded.agent_id).position != delivery_position
            )
            completed = any(
                task.task_id == task_id for task in next_state.completed_tasks
            )
            wait_streak = int(
                environment.get_state().ineffective_joint_wait_streak
            )
            if completed or collided or wait_streak >= 8 or terminated or truncated:
                if completed:
                    completion_steps.append(step + 1)
                break
        successes += int(completed and occupant_cleared_first and not collided)
        collisions += int(collided)
        deadlocks += int(wait_streak >= 8)
    count = max(1, episodes)
    return {
        "episodes": float(episodes),
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "deadlock_rate": deadlocks / count,
        "mean_completion_steps": (
            float(np.mean(completion_steps)) if completion_steps else 0.0
        ),
    }


def evaluate_dual_charger_approach_scenarios(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Require causal queueing when both robots approach the single charger."""

    successes = collisions = shutdowns = deadlocks = return_cycles = 0
    both_charged_steps: list[int] = []
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        environment.reset(seed=seed + episode)
        apply_dual_charger_approach_scenario(environment, variant=episode)
        initial_state = environment.get_state()
        priority_agent_id = min(
            initial_state.agents,
            key=lambda agent: (agent.battery, agent.agent_id),
        ).agent_id
        inference = policy.fork_for_inference(seed=seed + episode + 29_000_000)
        observations = environment.observations()
        charged_agents: list[str] = []
        wait_streak = int(initial_state.ineffective_joint_wait_streak)
        collided = shutdown = False
        episode_cycles = 0
        for step in range(min(32, environment_config.horizon)):
            decision_state = environment.get_state()
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=True,
                decision_key=(decision_state.episode_id, decision_state.frame),
            )
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="formal_dual_charger_approach_actor",
                ),
            )
            collided = collided or bool(info.get("robot_collision_event", False))
            shutdown = shutdown or bool(info.get("shutdowns", ()))
            for agent_id in info.get("charger_energy_gained_by_agent", {}):
                if agent_id not in charged_agents:
                    charged_agents.append(str(agent_id))
            episode_cycles += sum(
                str(event.get("event", "")) == "charger_return_cycle"
                for event in info.get("energy_events", ())
                if isinstance(event, Mapping)
            )
            wait_streak = int(
                environment.get_state().ineffective_joint_wait_streak
            )
            if (
                len(charged_agents) == len(environment.agent_ids)
                or collided
                or shutdown
                or wait_streak >= 8
                or terminated
                or truncated
            ):
                if len(charged_agents) == len(environment.agent_ids):
                    both_charged_steps.append(step + 1)
                break
        success = bool(
            len(charged_agents) == len(environment.agent_ids)
            and charged_agents[0] == priority_agent_id
            and not collided
            and not shutdown
            and wait_streak < 8
            and episode_cycles == 0
        )
        successes += int(success)
        collisions += int(collided)
        shutdowns += int(shutdown)
        deadlocks += int(wait_streak >= 8)
        return_cycles += episode_cycles
    count = max(1, episodes)
    return {
        "episodes": float(episodes),
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "shutdown_rate": shutdowns / count,
        "deadlock_rate": deadlocks / count,
        "return_cycles_per_episode": return_cycles / count,
        "mean_both_charged_steps": (
            float(np.mean(both_charged_steps)) if both_charged_steps else 0.0
        ),
    }


def evaluate_outer_exit_charger_approach_scenarios(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Require an urgent outer-exit robot to reach the charger causally."""

    successes = collisions = shutdowns = deadlocks = 0
    completion_steps: list[int] = []
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        environment.reset(seed=seed + episode)
        apply_outer_exit_charger_approach_scenario(environment, variant=episode)
        initial_state = environment.get_state()
        urgent_agent_id = next(
            agent.agent_id
            for agent in initial_state.agents
            if agent.position[0] == environment.layout.charger_position[0] - 1
        )
        inference = policy.fork_for_inference(seed=seed + episode + 29_000_000)
        observations = environment.observations()
        wait_streak = int(initial_state.ineffective_joint_wait_streak)
        collided = shutdown = completed = False
        first_charged_agent_id: str | None = None
        for step in range(min(16, environment_config.horizon)):
            decision_state = environment.get_state()
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=True,
                decision_key=(decision_state.episode_id, decision_state.frame),
            )
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="formal_outer_exit_charger_approach_actor",
                ),
            )
            collided = collided or bool(info.get("robot_collision_event", False))
            shutdown = shutdown or bool(info.get("shutdowns", ()))
            charged = tuple(info.get("charger_energy_gained_by_agent", {}))
            if charged and first_charged_agent_id is None:
                first_charged_agent_id = str(charged[0])
            completed = first_charged_agent_id == urgent_agent_id
            wait_streak = int(
                environment.get_state().ineffective_joint_wait_streak
            )
            if completed or collided or shutdown or wait_streak >= 8 or terminated or truncated:
                if completed:
                    completion_steps.append(step + 1)
                break
        successes += int(completed and not collided and not shutdown and wait_streak < 8)
        collisions += int(collided)
        shutdowns += int(shutdown)
        deadlocks += int(wait_streak >= 8)
    count = max(1, episodes)
    return {
        "episodes": float(episodes),
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "shutdown_rate": shutdowns / count,
        "deadlock_rate": deadlocks / count,
        "mean_completion_steps": (
            float(np.mean(completion_steps)) if completion_steps else 0.0
        ),
    }
