"""Audit deterministic MAPPO failures without changing submitted actions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from backend.artifacts import CollaborativeArtifactPaths
from env.warehouse.coordination import (
    is_necessary_urgent_charger_clearance,
    stable_coordination_actions,
)
from env.warehouse.contracts import ARTIFACT_NAMESPACE
from env.warehouse.environment import WarehouseMultiAgentEnv, shortest_path_distance
from env.warehouse.policy import MAPPOPolicy


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = CollaborativeArtifactPaths.under(
    PROJECT_ROOT,
    ARTIFACT_NAMESPACE,
).model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=102_026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-detours", action="store_true")
    parser.add_argument("--formal-detours-only", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--max-detour-records", type=int, default=50)
    parser.add_argument("--max-collision-records", type=int, default=10)
    args = parser.parse_args()

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    deadlocks: list[dict[str, object]] = []
    detours: list[dict[str, object]] = []
    collisions: list[dict[str, object]] = []
    shutdowns: list[dict[str, object]] = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=episode_seed)
        wait_streak = 0
        deadlock_recorded = False
        while True:
            before = deepcopy(environment.get_state())
            actions, _ = policy.act(
                observations,
                environment.global_state(),
                deterministic=True,
            )
            final_targets = environment._resolve_motion(before, actions)[0]
            teacher_actions = (
                None
                if args.skip_teacher
                else stable_coordination_actions(environment)
            )
            teacher_targets = (
                None
                if teacher_actions is None
                else environment._resolve_motion(before, teacher_actions)[0]
            )
            for agent in before.agents:
                if (
                    not args.include_detours
                    or
                    len(detours) >= int(args.max_detour_records)
                    or agent.carrying_task_id is None
                    or agent.navigation_goal_kind != "delivery"
                    or environment._requires_charge(before, agent)
                ):
                    continue
                current_distance = shortest_path_distance(
                    agent.position,
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
                final_distance = shortest_path_distance(
                    final_targets[agent.agent_id],
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
                if final_distance > current_distance:
                    held_actions = dict(actions)
                    held_actions[agent.agent_id] = "WAIT"
                    held_collision = environment._resolve_motion(
                        before,
                        held_actions,
                    )[3]
                    necessary_charger_clearance = (
                        is_necessary_urgent_charger_clearance(
                            environment,
                            before,
                            agent,
                        )
                    )
                    formal_avoidable = bool(
                        not held_collision
                        and not necessary_charger_clearance
                    )
                    if args.formal_detours_only and not formal_avoidable:
                        continue
                    teacher_distance = (
                        None
                        if teacher_targets is None
                        else shortest_path_distance(
                            teacher_targets[agent.agent_id],
                            agent.navigation_goal_position,
                            environment.config.map_layout_id,
                        )
                    )
                    detours.append(
                        {
                            "seed": episode_seed,
                            "frame": before.frame,
                            "agent": agent.agent_id,
                            "position": agent.position,
                            "battery": agent.battery,
                            "task": agent.carrying_task_id,
                            "action": actions[agent.agent_id],
                            "network_actions": dict(actions),
                            "counterfactual_teacher_action": (
                                None
                                if teacher_actions is None
                                else teacher_actions[agent.agent_id]
                            ),
                            "counterfactual_teacher_actions": (
                                None
                                if teacher_actions is None
                                else dict(teacher_actions)
                            ),
                            "distance_before": current_distance,
                            "distance_after": final_distance,
                            "teacher_distance_after": teacher_distance,
                            "counterfactual_teacher_detours": (
                                None
                                if teacher_distance is None
                                else teacher_distance > current_distance
                            ),
                            "held_action_would_collide": bool(held_collision),
                            "necessary_urgent_charger_clearance": bool(
                                necessary_charger_clearance
                            ),
                            "formal_avoidable_detour": formal_avoidable,
                            "teammate_position": next(
                                item.position
                                for item in before.agents
                                if item.agent_id != agent.agent_id
                            ),
                            "agents_before": [
                                {
                                    "id": item.agent_id,
                                    "position": item.position,
                                    "battery": item.battery,
                                    "carrying": item.carrying_task_id,
                                }
                                for item in before.agents
                            ],
                            "tasks_before": [
                                {
                                    "id": task.task_id,
                                    "pickup": task.pickup_position,
                                    "delivery": task.delivery_position,
                                    "status": task.status,
                                    "carrier": task.carrier_agent_id,
                                }
                                for task in before.tasks
                            ],
                        }
                    )
            observations, _, terminated, truncated, info = environment.step(actions)
            if (
                bool(info.get("robot_collision_event"))
                and len(collisions) < int(args.max_collision_records)
            ):
                collisions.append(
                    {
                        "seed": episode_seed,
                        "frame": before.frame,
                        "network_actions": dict(actions),
                        "counterfactual_teacher_actions": (
                            None
                            if teacher_actions is None
                            else dict(teacher_actions)
                        ),
                        "collision_kind": info.get("robot_collision_kind"),
                        "action_resolution": dict(
                            info.get("action_resolution", {})
                        ),
                        "agents_before": [
                            {
                                "id": agent.agent_id,
                                "position": agent.position,
                                "battery": agent.battery,
                                "carrying": agent.carrying_task_id,
                                "navigation_goal_kind": agent.navigation_goal_kind,
                                "navigation_goal_position": (
                                    agent.navigation_goal_position
                                ),
                            }
                            for agent in before.agents
                        ],
                    }
                )
            if (
                terminated
                and environment.get_state().terminal_reason
                == "battery_shutdown"
            ):
                shutdowns.append(
                    {
                        "seed": episode_seed,
                        "frame": before.frame,
                        "network_actions": dict(actions),
                        "agents_before": [
                            {
                                "id": agent.agent_id,
                                "position": agent.position,
                                "battery": agent.battery,
                                "carrying": agent.carrying_task_id,
                                "navigation_goal_kind": agent.navigation_goal_kind,
                                "navigation_goal_position": (
                                    agent.navigation_goal_position
                                ),
                            }
                            for agent in before.agents
                        ],
                    }
                )
            ineffective_wait = bool(
                all(
                    str(action) == "WAIT"
                    for action in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
            )
            wait_streak = wait_streak + 1 if ineffective_wait else 0
            if wait_streak >= 8 and not deadlock_recorded:
                state = environment.get_state()
                deadlocks.append(
                    {
                        "seed": episode_seed,
                        "frame": state.frame,
                        "network_actions": dict(actions),
                        "action_resolution": dict(
                            info.get("action_resolution", {})
                        ),
                        "agents": [
                            {
                                "id": agent.agent_id,
                                "position": agent.position,
                                "battery": agent.battery,
                                "carrying": agent.carrying_task_id,
                                "navigation_goal_kind": agent.navigation_goal_kind,
                                "navigation_goal_position": (
                                    agent.navigation_goal_position
                                ),
                            }
                            for agent in state.agents
                        ],
                    }
                )
                deadlock_recorded = True
            if terminated or truncated:
                break
    print(
        json.dumps(
            {
                "execution": "direct_mappo_actor",
                "deadlock_count": len(deadlocks),
                "deadlocks": deadlocks,
                "collision_record_count": len(collisions),
                "collisions": collisions,
                "shutdown_count": len(shutdowns),
                "shutdowns": shutdowns,
                "detour_examples": detours,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
