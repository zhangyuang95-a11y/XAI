"""Preflight the offline supervision policy before training MAPPO.

This evaluator never participates in deployed action execution.  It exists to
prevent a structurally bad teacher from being distilled into a fresh Actor.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from statistics import mean

from env.warehouse.coordination import (
    _clear_head_on_encounter,
    _priority_agent_and_basis,
    _reserved_side_clearance,
    stable_coordination_actions,
)
from env.warehouse.domain import WarehouseConfig
from env.warehouse.environment import WarehouseMultiAgentEnv


def evaluate_teacher(*, episodes: int, seed_start: int) -> dict[str, object]:
    deliveries: list[int] = []
    return_episodes = 0
    return_cycles = 0
    starvation_episodes = 0
    shutdown_episodes = 0
    collision_episodes = 0
    deadlock_episodes = 0
    action_metrics: Counter[str] = Counter()
    maximum_avoidable_wait_streaks = {
        "robot_1": 0,
        "robot_2": 0,
    }
    path_actual_steps = 0.0
    path_shortest_safe_steps = 0.0
    return_cycle_details: list[dict[str, object]] = []
    for seed in range(seed_start, seed_start + episodes):
        environment = WarehouseMultiAgentEnv(
            WarehouseConfig(participant_detour_scoring=False)
        )
        environment.reset(seed=seed)
        returned = False
        starved = False
        shutdown = False
        collided = False
        deadlocked = False
        recent_steps: deque[dict[str, object]] = deque(maxlen=7)
        while True:
            before = environment.get_state()
            overrides = {
                agent.agent_id: goal
                for agent in before.agents
                if (
                    goal := environment._frozen_route_goal(
                        before,
                        agent.agent_id,
                        prioritize_old_tasks=True,
                    )
                )
                is not None
            }
            imminent_head_on = _clear_head_on_encounter(
                environment,
                goal_overrides=overrides,
            )
            priority_agent, priority_basis = _priority_agent_and_basis(
                environment,
                imminent_head_on=imminent_head_on,
                goal_overrides=overrides,
            )
            actions = stable_coordination_actions(environment)
            recent_steps.append(
                {
                    "frame": before.frame,
                    "actions": dict(actions),
                    "priority_agent": priority_agent.agent_id,
                    "priority_basis": priority_basis,
                    "imminent_head_on": imminent_head_on,
                    "side_reservation": _reserved_side_clearance(
                        environment,
                        goal_overrides=overrides,
                    ),
                    "agents": {
                        agent.agent_id: {
                            "position": agent.position,
                            "battery": agent.battery,
                            "carrying": agent.carrying_task_id,
                            "commitment": agent.route_commitment_task_id,
                            "goal_kind": agent.navigation_goal_kind,
                            "goal_position": agent.navigation_goal_position,
                            "requires_charge": environment._requires_charge(
                                before,
                                agent,
                            ),
                        }
                        for agent in before.agents
                    },
                }
            )
            _, _, terminated, truncated, info = environment.step(actions)
            for agent_id in environment.agent_ids:
                action_metrics[f"{agent_id}_steps"] += 1
                action_metrics[f"{agent_id}_waits"] += int(
                    str(actions.get(agent_id, "WAIT")) == "WAIT"
                )
                action_metrics[f"{agent_id}_avoidable_waits"] += int(
                    agent_id in info.get("avoidable_wait_agents", ())
                )
                action_metrics[f"{agent_id}_detours"] += int(
                    agent_id in info.get("avoidable_detour_agents", ())
                )
                action_metrics[f"{agent_id}_loaded_detours"] += int(
                    agent_id
                    in info.get(
                        "avoidable_loaded_delivery_detour_agents",
                        (),
                    )
                )
                maximum_avoidable_wait_streaks[agent_id] = max(
                    maximum_avoidable_wait_streaks[agent_id],
                    int(
                        info.get("avoidable_wait_streaks", {}).get(
                            agent_id,
                            0,
                        )
                    ),
                )
            action_metrics["ineffective_joint_wait_steps"] += int(
                int(info.get("ineffective_joint_wait_streak", 0)) > 0
            )
            episode_cycles = sum(
                event.get("event") == "charger_return_cycle"
                for event in info.get("energy_events", ())
            )
            return_cycles += episode_cycles
            for event in info.get("energy_events", ()):
                if event.get("event") != "charger_return_cycle":
                    continue
                agent_id = str(event.get("agent_id"))
                before_agent = before.by_id(agent_id)
                after_agent = environment.get_state().by_id(agent_id)
                return_cycle_details.append(
                    {
                        "seed": seed,
                        "frame": environment.get_state().frame,
                        "agent_id": agent_id,
                        "actions": dict(actions),
                        "event": dict(event),
                        "before_position": before_agent.position,
                        "before_battery": before_agent.battery,
                        "before_commitment": before_agent.route_commitment_task_id,
                        "after_position": after_agent.position,
                        "after_battery": after_agent.battery,
                        "after_commitment": after_agent.route_commitment_task_id,
                        "recent_steps": list(recent_steps),
                    }
                )
            returned = returned or episode_cycles > 0
            starved = starved or bool(info.get("starving_task_ids", ()))
            shutdown = shutdown or bool(info.get("shutdowns", ()))
            collided = collided or bool(info.get("collisions", ()))
            deadlocked = deadlocked or int(
                info.get("ineffective_joint_wait_streak", 0)
            ) >= 8
            if terminated or truncated:
                break
        final_state = environment.get_state()
        for task in final_state.completed_tasks:
            if (
                task.delivered_frame is None
                or task.claimed_frame is None
                or task.shortest_safe_delivery_steps is None
            ):
                continue
            path_actual_steps += float(task.delivered_frame - task.claimed_frame)
            path_shortest_safe_steps += float(task.shortest_safe_delivery_steps)
        deliveries.append(environment.get_state().total_deliveries)
        return_episodes += int(returned)
        starvation_episodes += int(starved)
        shutdown_episodes += int(shutdown)
        collision_episodes += int(collided)
        deadlock_episodes += int(deadlocked)
    denominator = max(1, episodes)
    report: dict[str, object] = {
        "kind": "offline_teacher_preflight",
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + episodes - 1,
        "mean_deliveries": mean(deliveries) if deliveries else 0.0,
        "minimum_deliveries": min(deliveries, default=0),
        "maximum_deliveries": max(deliveries, default=0),
        "charger_departure_return_cycle_episode_rate": (
            return_episodes / denominator
        ),
        "charger_departure_return_cycles": return_cycles,
        "task_starvation_episode_rate": starvation_episodes / denominator,
        "shutdown_episode_rate": shutdown_episodes / denominator,
        "collision_episode_rate": collision_episodes / denominator,
        "deadlock_episode_rate": deadlock_episodes / denominator,
        "per_agent_action_metrics": {
            agent_id: {
                "steps": action_metrics[f"{agent_id}_steps"],
                "waits": action_metrics[f"{agent_id}_waits"],
                "wait_rate": (
                    action_metrics[f"{agent_id}_waits"]
                    / max(1, action_metrics[f"{agent_id}_steps"])
                ),
                "avoidable_waits": action_metrics[
                    f"{agent_id}_avoidable_waits"
                ],
                "avoidable_wait_rate": (
                    action_metrics[f"{agent_id}_avoidable_waits"]
                    / max(1, action_metrics[f"{agent_id}_steps"])
                ),
                "maximum_avoidable_wait_streak": (
                    maximum_avoidable_wait_streaks[agent_id]
                ),
                "detours": action_metrics[f"{agent_id}_detours"],
                "loaded_detours": action_metrics[
                    f"{agent_id}_loaded_detours"
                ],
            }
            for agent_id in environment.agent_ids
        },
        "avoidable_wait_rate": (
            sum(
                action_metrics[f"{agent_id}_avoidable_waits"]
                for agent_id in environment.agent_ids
            )
            / max(
                1,
                sum(
                    action_metrics[f"{agent_id}_steps"]
                    for agent_id in environment.agent_ids
                ),
            )
        ),
        "avoidable_loaded_delivery_detours": sum(
            action_metrics[f"{agent_id}_loaded_detours"]
            for agent_id in environment.agent_ids
        ),
        "ineffective_joint_wait_steps": action_metrics[
            "ineffective_joint_wait_steps"
        ],
        "path_efficiency_actual_over_shortest_safe": (
            path_actual_steps / max(1.0, path_shortest_safe_steps)
        ),
        "path_actual_steps": path_actual_steps,
        "path_shortest_safe_steps": path_shortest_safe_steps,
        "charger_return_cycle_details": return_cycle_details,
    }
    report["acceptance"] = {
        "charger_departure_return_cycle_rate_le_0_01": (
            report["charger_departure_return_cycle_episode_rate"] <= 0.01
        ),
        "task_starvation_episode_rate_le_0_05": (
            report["task_starvation_episode_rate"] <= 0.05
        ),
        "shutdown_episode_rate_eq_0": report["shutdown_episode_rate"] == 0.0,
        "collision_episode_rate_eq_0": report["collision_episode_rate"] == 0.0,
        "deadlock_episode_rate_le_0_01": report["deadlock_episode_rate"] <= 0.01,
        "avoidable_wait_rate_le_0_005": report["avoidable_wait_rate"] <= 0.005,
        "avoidable_loaded_delivery_detours_eq_0": (
            report["avoidable_loaded_delivery_detours"] == 0
        ),
    }
    report["accepted"] = all(report["acceptance"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=15000)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    report = evaluate_teacher(episodes=args.episodes, seed_start=args.seed_start)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
