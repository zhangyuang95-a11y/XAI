"""Evaluate and package the formal staggered-map Actor release."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.training.export_numpy_actor import export_numpy_actor
from env.warehouse.contracts import MODEL_VERSION
from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.mappo import MAPPOPolicy, evaluate_policy
from ui.tutorial import TUTORIAL_SEED


EVALUATION_FIELDS = (
    "mean_deliveries",
    "deliveries_per_100_steps",
    "collision_episode_rate",
    "mean_robot_collision_events",
    "maximum_robot_collision_events",
    "repeated_collision_episode_rate",
    "shutdown_episode_rate",
    "deadlock_episode_rate",
    "avoidable_wait_rate",
    "mean_charger_use_steps",
    "path_efficiency_actual_over_shortest_safe",
    "mean_post_policy_action_interventions",
    "avoidable_mission_detour_steps",
    "charger_departure_return_cycle_episode_rate",
    "task_starvation_episode_rate",
    "avoidable_loaded_delivery_detour_steps",
)

# Fresh post-formal smoke ranges.  These sit beyond the complete declared
# training/development ledger and the final formal protocol rooted at 23M, so
# packaging cannot reuse evidence that influenced training or release.
RELEASE_AI_SEED_STARTS = (25_000_004, 25_100_004)
RELEASE_NOISY_SEED_START = 25_200_004


def _compact_evaluation(values: dict[str, Any]) -> dict[str, float]:
    return {field: float(values[field]) for field in EVALUATION_FIELDS}


def _tutorial_summary(policy: MAPPOPolicy) -> dict[str, Any]:
    config = replace(
        collaborative_study_config(), participant_detour_scoring=False
    )
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=TUTORIAL_SEED)
    state = environment.get_state()
    state.by_id("robot_2").battery = 35.0
    environment.set_state(state)
    observations = environment.observations()
    events: set[str] = set()
    collisions = 0
    charging_steps = 0
    inference = policy.fork_for_inference(seed=TUTORIAL_SEED)
    while True:
        actions, _ = inference.act(
            observations,
            environment.global_state(),
            deterministic=True,
        )
        observations, _, terminated, truncated, info = environment.step(actions)
        collisions += int(bool(info.get("robot_collision_event", False)))
        charging_steps += int(bool(info.get("charger_used", False)))
        for group in (
            info.get("task_changes", ()),
            info.get("coordination_events", ()),
            info.get("energy_events", ()),
        ):
            for event in group:
                if isinstance(event, dict):
                    events.add(str(event.get("event", "")))
        if terminated or truncated:
            break
    final = environment.get_state()
    return {
        "seed": TUTORIAL_SEED,
        "steps": final.frame,
        "deliveries": final.total_deliveries,
        "robot_collision_events": collisions,
        "shutdowns": final.shutdown_count,
        "charging_steps": charging_steps,
        "events": sorted(event for event in events if event),
        "terminal_reason": final.terminal_reason,
    }


def _validated_formal_evaluation(
    path: Path,
    source: Path,
) -> dict[str, Any]:
    """Require the strict 1,000 + 1,000 gate before packaging a release."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("acceptance_checks", {})
    if (
        payload.get("model_version") != MODEL_VERSION
        or int(payload.get("episodes_per_condition", 0)) < 1_000
        or int(payload.get("multi_partner_episodes", 0)) < 1_000
        or not bool(payload.get("formal_candidate", False))
        or not isinstance(checks, dict)
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise RuntimeError("Strict formal evaluation has not passed.")
    expected_hash = payload.get("artifact_hashes", {}).get("model")
    actual_hash = sha256(source.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise RuntimeError("Formal evaluation checkpoint hash does not match source.")
    return payload


def _release_gate(seed_ranges: dict[str, dict[str, float]]) -> dict[str, bool]:
    values = tuple(seed_ranges.values())
    return {
        "mean_deliveries_ge_6": all(item["mean_deliveries"] >= 6.0 for item in values),
        "collision_episode_rate_le_0_01": all(
            item["collision_episode_rate"] <= 0.01 for item in values
        ),
        "maximum_collision_events_le_1": all(
            item["maximum_robot_collision_events"] <= 1.0 for item in values
        ),
        "repeated_collision_rate_eq_0": all(
            item["repeated_collision_episode_rate"] == 0.0 for item in values
        ),
        "shutdown_episode_rate_le_0_01": all(
            item["shutdown_episode_rate"] <= 0.01 for item in values
        ),
        "deadlock_episode_rate_le_0_01": all(
            item["deadlock_episode_rate"] <= 0.01 for item in values
        ),
        "avoidable_wait_rate_le_0_005": all(
            item["avoidable_wait_rate"] <= 0.005 for item in values
        ),
        "charger_use_positive": all(
            item["mean_charger_use_steps"] > 0.0 for item in values
        ),
        "path_efficiency_le_1_10": all(
            item["path_efficiency_actual_over_shortest_safe"] <= 1.10
            for item in values
        ),
        "charger_return_cycle_rate_le_0_01": all(
            item["charger_departure_return_cycle_episode_rate"] <= 0.01
            for item in values
        ),
        "task_starvation_rate_le_0_05": all(
            item["task_starvation_episode_rate"] <= 0.05 for item in values
        ),
        "avoidable_loaded_delivery_detours_eq_0": all(
            item["avoidable_loaded_delivery_detour_steps"] == 0.0
            for item in values
        ),
        "avoidable_mission_detours_eq_0": all(
            item["avoidable_mission_detour_steps"] == 0.0
            for item in values
        ),
        "post_policy_action_interventions_eq_0": all(
            item["mean_post_policy_action_interventions"] == 0.0
            for item in values
        ),
    }


def _release_acceptance_gates(
    tutorial: dict[str, int],
    supplemental_gates: dict[str, bool],
) -> dict[str, bool]:
    """Combine formal, tutorial, and fresh-seed smoke requirements."""

    return {
        "strict_formal_evaluation_passed": True,
        "supplemental_seed_ranges_pass": bool(
            supplemental_gates and all(supplemental_gates.values())
        ),
        "tutorial_has_120_steps": tutorial["steps"] == 120,
        "tutorial_has_delivery": tutorial["deliveries"] > 0,
        "tutorial_has_charging": tutorial["charging_steps"] > 0,
        "tutorial_has_no_shutdown": tutorial["shutdowns"] == 0,
    }


def release(
    source: Path,
    formal_evaluation: Path,
    checkpoint_output: Path,
    actor_output: Path,
    evaluation_output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    formal = _validated_formal_evaluation(formal_evaluation, source)
    policy = MAPPOPolicy.load(source, device=device)
    config = collaborative_study_config()
    ranges: dict[str, dict[str, float]] = {}
    for start in RELEASE_AI_SEED_STARTS:
        values = evaluate_policy(
            policy,
            config,
            episodes=100,
            seed=start,
            deterministic=False,
        )
        ranges[f"{start}-{start + 99}"] = _compact_evaluation(values)
        print(json.dumps({start: ranges[f"{start}-{start + 99}"]}), flush=True)
    noisy = _compact_evaluation(
        evaluate_policy(
            policy,
            config,
            episodes=100,
            seed=RELEASE_NOISY_SEED_START,
            noisy_teammate_probability=0.20,
            deterministic=False,
        )
    )
    tutorial = _tutorial_summary(policy)
    supplemental_gates = _release_gate(ranges)
    gates = _release_acceptance_gates(tutorial, supplemental_gates)
    if not all(gates.values()):
        raise RuntimeError(f"Formal release gates failed: {gates}")

    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    training_metadata = deepcopy(source_payload.get("training_metadata", {}))
    training_metadata["formal_evaluation"] = formal
    training_metadata["release_smoke_evaluation"] = {
        "protocol": {
            "execution": "independent simultaneous stochastic actor",
            "episodes_per_seed_range": 100,
            "post_policy_action_interventions": 0,
        },
        "seed_ranges": ranges,
        "noisy_teammate_20_percent": noisy,
        "supplemental_gates": supplemental_gates,
        "tutorial": tutorial,
    }
    policy.save(checkpoint_output, training_metadata=training_metadata)
    export_numpy_actor(checkpoint_output, actor_output)
    report = {
        "model_version": MODEL_VERSION,
        "checkpoint": checkpoint_output.name,
        "checkpoint_sha256": sha256(checkpoint_output.read_bytes()).hexdigest(),
        "render_actor": actor_output.name,
        "render_actor_sha256": sha256(actor_output.read_bytes()).hexdigest(),
        "formal_evaluation": str(formal_evaluation.resolve()),
        "formal_evaluation_sha256": sha256(
            formal_evaluation.read_bytes()
        ).hexdigest(),
        "protocol": training_metadata["release_smoke_evaluation"]["protocol"],
        "seed_ranges": ranges,
        "noisy_teammate_20_percent": noisy,
        "release_gates": gates,
        "supplemental_gates": supplemental_gates,
        "tutorial": tutorial,
    }
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--formal-evaluation", type=Path, required=True)
    parser.add_argument("checkpoint_output", type=Path)
    parser.add_argument("actor_output", type=Path)
    parser.add_argument("evaluation_output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            release(
                args.source,
                args.formal_evaluation,
                args.checkpoint_output,
                args.actor_output,
                args.evaluation_output,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
