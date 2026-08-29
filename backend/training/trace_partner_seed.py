"""Replay one exact multi-partner evaluation episode with an audit trace."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.coordination import (
    _clear_head_on_encounter,
    _priority_agent_and_basis,
    stable_coordination_goal_overrides,
)
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.coordination_priority import single_lane_egress_agent_id
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.evaluation_diagnostics import (
    avoidable_loaded_delivery_detour_agents,
)
from env.warehouse.partner_policies import participant_surrogate_action
from env.warehouse.policy import MAPPOPolicy


def _state_payload(state: Any) -> dict[str, Any]:
    return {
        "episode_id": int(state.episode_id),
        "frame": int(state.frame),
        "ineffective_joint_wait_streak": int(
            state.ineffective_joint_wait_streak
        ),
        "agents": [asdict(agent) for agent in state.agents],
        "tasks": [asdict(task) for task in state.tasks],
        "completed_tasks": [asdict(task) for task in state.completed_tasks],
    }


def replay_partner_episode(
    policy: MAPPOPolicy,
    *,
    base_seed: int,
    episode_index: int,
    profile: str,
) -> dict[str, Any]:
    """Match ``evaluate_policy`` RNG consumption through one episode index."""

    participant_rng = np.random.default_rng(base_seed + 17_000_000)
    participant_mode = profile != "ai_ai"
    selected_trace: list[dict[str, Any]] = []
    selected_summary: dict[str, Any] = {}
    for episode in range(episode_index + 1):
        episode_seed = base_seed + episode
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=episode_seed)
        if participant_mode:
            participant_state = environment.get_state()
            participant_state.participant_controlled_agent_id = (
                environment.config.human_agent_id
            )
            environment.set_state(participant_state)
            observations = environment.observations()
        inference = policy.fork_for_inference(
            seed=episode_seed + 29_000_000
        )
        trace: list[dict[str, Any]] = []
        collision_events = return_cycles = 0
        deadlocked = False
        while True:
            before = environment.get_state()
            public_coordination_actions = stable_coordination_actions(
                environment
            )
            participant_action = (
                participant_surrogate_action(
                    environment,
                    profile=profile,
                    rng=participant_rng,
                )
                if participant_mode
                else None
            )
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=False,
                decision_key=(before.episode_id, before.frame),
            )
            participant_overrides = {}
            if participant_action is not None:
                actions[environment.config.human_agent_id] = participant_action
                participant_overrides = {
                    environment.config.human_agent_id: participant_action
                }
            strict_loaded_detours = avoidable_loaded_delivery_detour_agents(
                environment,
                before,
                actions,
                excluded_agent_ids=participant_overrides,
            )
            diagnostic_priority = None
            diagnostic_basis = None
            diagnostic_coordination_features: dict[str, list[float]] = {}
            if episode == episode_index:
                goal_overrides = stable_coordination_goal_overrides(environment)
                diagnostic_priority, diagnostic_basis = _priority_agent_and_basis(
                    environment,
                    imminent_head_on=_clear_head_on_encounter(
                        environment,
                        goal_overrides=goal_overrides,
                    ),
                    goal_overrides=goal_overrides,
                )
                diagnostic_coordination_features = {
                    agent_id: [
                        float(value)
                        for value in observations[agent_id][
                            inference.network.coordination_start :
                            inference.network.coordination_start + 8
                        ]
                    ]
                    for agent_id in observations
                }
                diagnostic_coordination_features["single_lane_egress_agent_id"] = (
                    single_lane_egress_agent_id(
                        before,
                        environment.config,
                        goal_positions=goal_overrides,
                    )
                )
                diagnostic_coordination_features["robot_2_structural"] = {
                    "charge_goal": float(observations["robot_2"][15]),
                    "teammate_charge_goal": float(
                        observations["robot_2"][
                            inference.network.teammate_charge_goal_index
                        ]
                    ),
                    "self_at_charger": float(
                        observations["robot_2"][
                            inference.network.own_action_features_start + 6
                        ]
                    ),
                    "teammate_at_charger": float(
                        observations["robot_2"][
                            inference.network.teammate_action_features_start + 6
                        ]
                    ),
                    "own_frames_since_charger_departure": float(
                        observations["robot_2"][
                            inference.network.own_frames_since_charger_departure_index
                        ]
                    ),
                    "teammate_steps_since_charging": float(
                        observations["robot_2"][
                            inference.network.teammate_steps_since_charging_index
                        ]
                    ),
                    "own_action_targets_charger": [
                        float(
                            observations["robot_2"][
                                inference.network.own_action_features_start
                                + action_index
                                * inference.network.per_action_feature_dim
                                + 5
                            ]
                        )
                        for action_index in range(inference.network.action_dim)
                    ],
                }
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="partner_seed_diagnostic",
                    participant_overrides=participant_overrides,
                ),
            )
            if episode == episode_index:
                trace.append(
                    {
                        "before": _state_payload(before),
                        "participant_action": participant_action,
                        "public_coordination_actions": dict(
                            public_coordination_actions
                        ),
                        "public_priority": {
                            "agent_id": diagnostic_priority.agent_id,
                            "basis": diagnostic_basis,
                        },
                        "actor_coordination_features": (
                            diagnostic_coordination_features
                        ),
                        "actor_actions_before_override": {
                            agent_id: distribution.proposed_action
                            for agent_id, distribution in distributions.items()
                        },
                        "actor_probabilities": {
                            agent_id: list(distribution.probabilities)
                            for agent_id, distribution in distributions.items()
                        },
                        "submitted_joint_action": dict(actions),
                        "executed_joint_action": dict(
                            info.get("executed_actions", {})
                        ),
                        "intended_targets": dict(
                            info.get("intended_targets", {})
                        ),
                        "collision": bool(
                            info.get("robot_collision_event", False)
                        ),
                        "collision_kind": info.get(
                            "robot_collision_kind"
                        ),
                        "coordination_events": list(
                            info.get("coordination_events", ())
                        ),
                        "energy_events": list(info.get("energy_events", ())),
                        "avoidable_wait_agents": list(
                            info.get("avoidable_wait_agents", ())
                        ),
                        "starving_task_ids": list(
                            info.get("starving_task_ids", ())
                        ),
                        "starving_task_assignees": dict(
                            info.get("starving_task_assignees", {})
                        ),
                        "counterfactual_regret_units": dict(
                            info.get("counterfactual_regret_units", {})
                        ),
                        "loaded_detour_agents": list(
                            info.get(
                                "avoidable_loaded_delivery_detour_agents",
                                (),
                            )
                        ),
                        "strict_loaded_delivery_detour_agents": list(
                            strict_loaded_detours
                        ),
                        "after": _state_payload(environment.get_state()),
                    }
                )
            collision_events += int(
                bool(info.get("robot_collision_event", False))
            )
            return_cycles += sum(
                str(event.get("event", "")) == "charger_return_cycle"
                for event in info.get("energy_events", ())
            )
            deadlocked = deadlocked or (
                environment.get_state().ineffective_joint_wait_streak >= 8
            )
            if terminated or truncated:
                if episode == episode_index:
                    final = environment.get_state()
                    selected_trace = trace
                    selected_summary = {
                        "seed": episode_seed,
                        "profile": profile,
                        "frames": int(final.frame),
                        "deliveries": int(final.total_deliveries),
                        "collision_events": collision_events,
                        "return_cycles": return_cycles,
                        "deadlocked": deadlocked,
                        "terminal_reason": final.terminal_reason,
                    }
                break
    return {"summary": selected_summary, "trace": selected_trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episode_index < 0:
        parser.error("episode index must be non-negative")
    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    payload = replay_partner_episode(
        policy,
        base_seed=int(args.base_seed),
        episode_index=int(args.episode_index),
        profile=str(args.profile),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
