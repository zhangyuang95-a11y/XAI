"""Evaluate the formal two-robot policy on fixed independent seed ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.artifact_contracts import (
    validate_posthoc_rcpd_metadata,
    validate_reference_trajectory_manifest,
)
from backend.artifacts import CollaborativeArtifactPaths, file_sha256
from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    ARTIFACT_NAMESPACE,
    FORMAL_ACCEPTANCE_CHECKS,
    RUNTIME_CONTROLLER,
)
from env.warehouse.mappo import (
    evaluate_head_on_yield_scenarios,
    evaluate_policy,
    evaluate_random_policy,
)
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.seed_calibration import load_parallel_seed_library
from env.warehouse.regressions import evaluate_seed_42027_detour_regressions


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = CollaborativeArtifactPaths.under(
    PROJECT_ROOT,
    ARTIFACT_NAMESPACE,
)
DEFAULT_ROOT = DEFAULT_ARTIFACTS.root


def _bootstrap_difference_lower(
    left: list[float],
    right: list[float],
    *,
    seed: int,
    samples: int = 5000,
) -> float:
    rng = np.random.default_rng(seed)
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        differences[index] = (
            rng.choice(left_values, size=len(left_values), replace=True).mean()
            - rng.choice(right_values, size=len(right_values), replace=True).mean()
        )
    return float(np.quantile(differences, 0.025))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AI-AI, noisy-teammate, and random warehouse policies."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_ARTIFACTS.model),
    )
    parser.add_argument("--episodes", type=int, default=200)
    # The previous candidate was diagnosed on the 500004 seed family.  A
    # genuinely fresh family is required after using that result to improve
    # training, so the next formal gate starts above every training/offline
    # seed interval and the post-hoc RCPD interval.
    parser.add_argument("--seed", type=int, default=12_000_004)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACTS.formal_evaluation),
    )
    parser.add_argument(
        "--training-summary",
        default=str(DEFAULT_ARTIFACTS.training_summary),
    )
    parser.add_argument("--program", default=str(DEFAULT_ARTIFACTS.rcpd_program))
    parser.add_argument(
        "--reference-trajectory",
        default=str(DEFAULT_ARTIFACTS.reference_trajectory),
    )
    parser.add_argument(
        "--parallel-seed-library",
        default=str(DEFAULT_ARTIFACTS.parallel_seed_pairs),
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    config = policy.environment_config
    ai_ai = evaluate_policy(
        policy,
        config,
        episodes=args.episodes,
        seed=args.seed,
    )
    noisy = evaluate_policy(
        policy,
        config,
        episodes=args.episodes,
        seed=args.seed + 50_000,
        noisy_teammate_probability=0.20,
    )
    random_policy = evaluate_random_policy(
        config,
        episodes=args.episodes,
        seed=args.seed + 100_000,
    )
    head_on = evaluate_head_on_yield_scenarios(
        policy,
        config,
        episodes=max(20, min(args.episodes, 200)),
        seed=args.seed + 150_000,
    )
    summary_path = Path(args.training_summary)
    training_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    explanation_eligible = bool(
        training_summary.get("program_regularization", {}).get(
            "explanation_eligible", False
        )
    )
    artifact_contract_errors: dict[str, str] = {}
    try:
        program_payload = json.loads(Path(args.program).read_text(encoding="utf-8"))
        validate_posthoc_rcpd_metadata(program_payload.get("metadata", {}))
        posthoc_rcpd_contract_valid = True
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        posthoc_rcpd_contract_valid = False
        artifact_contract_errors["program"] = str(exc)
    try:
        reference_payload = json.loads(
            Path(args.reference_trajectory).read_text(encoding="utf-8")
        )
        validate_reference_trajectory_manifest(
            reference_payload,
            model_version=policy.model_version,
            environment_version=WarehouseMultiAgentEnv.environment_name,
            map_layout_id=policy.environment_config.map_layout_id,
        )
        reference_contract_valid = True
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        reference_contract_valid = False
        artifact_contract_errors["reference_trajectory"] = str(exc)
    try:
        load_parallel_seed_library(args.parallel_seed_library)
        parallel_seed_contract_valid = True
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parallel_seed_contract_valid = False
        artifact_contract_errors["parallel_seed_library"] = str(exc)
    formal_ranges = (
        (args.seed, args.episodes),
        (args.seed + 50_000, args.episodes),
        (args.seed + 100_000, args.episodes),
        (args.seed + 150_000, max(20, min(args.episodes, 200))),
    )

    def disjoint(left: tuple[int, int], right: tuple[int, int]) -> bool:
        left_start, left_end_exclusive = left
        right_start, right_end_exclusive = right
        return bool(
            left_end_exclusive > left_start
            and right_end_exclusive > right_start
            and (
                left_end_exclusive <= right_start
                or right_end_exclusive <= left_start
            )
        )

    ledger = training_summary.get("seed_ledger", {})
    raw_reserved = (
        ledger.get("reserved_intervals", ()) if isinstance(ledger, dict) else ()
    )
    try:
        reserved_ranges = tuple(
            (int(item["start"]), int(item["end_exclusive"]))
            for item in raw_reserved
            if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError):
        reserved_ranges = ()
    formal_exclusive_ranges = tuple(
        (start, start + count) for start, count in formal_ranges
    )
    independent_seed_ranges = bool(
        isinstance(ledger, dict)
        and ledger.get("schema") == "warehouse-training-seed-ledger.v1"
        and len(reserved_ranges) == len(raw_reserved)
        and bool(reserved_ranges)
        and all(
            disjoint(reserved_range, formal_range)
            for reserved_range in reserved_ranges
            for formal_range in formal_exclusive_ranges
        )
    )
    delivery_difference_lower = _bootstrap_difference_lower(
        ai_ai["delivery_samples"],
        random_policy["delivery_samples"],
        seed=args.seed + 1,
    )
    score_difference_lower = _bootstrap_difference_lower(
        ai_ai["user_score_samples"],
        random_policy["user_score_samples"],
        seed=args.seed + 2,
    )
    detour_regressions = evaluate_seed_42027_detour_regressions(config)
    checks = {
        "episodes_per_condition_ge_200": args.episodes >= 200,
        "formal_seed_ranges_disjoint_from_training": independent_seed_ranges,
        "shutdown_episode_rate_le_0_05": ai_ai["shutdown_episode_rate"] <= 0.05,
        "charger_utilization_positive": ai_ai["charger_utilization_rate"] > 0.0,
        "mean_minimum_battery_positive": ai_ai["mean_minimum_battery"] > 0.0,
        "collision_episode_rate_le_0_05": ai_ai["collision_episode_rate"] <= 0.05,
        "maximum_collision_events_per_episode_le_1": (
            ai_ai["maximum_robot_collision_events"] <= 1
        ),
        "repeated_collision_episode_rate_eq_0": (
            ai_ai["repeated_collision_episode_rate"] == 0.0
        ),
        "deadlock_episode_rate_le_0_05": ai_ai["deadlock_episode_rate"] <= 0.05,
        "head_on_yield_success_ge_0_90": head_on["success_rate"] >= 0.90,
        "delivery_bootstrap_lower_positive": delivery_difference_lower > 0.0,
        "score_bootstrap_lower_positive": score_difference_lower > 0.0,
        "noisy_delivery_episode_rate_ge_0_80": noisy["delivery_episode_rate"] >= 0.80,
        "charger_departure_return_cycle_rate_le_0_01": (
            ai_ai["charger_departure_return_cycle_episode_rate"] <= 0.01
        ),
        "task_starvation_episode_rate_le_0_05": (
            ai_ai["task_starvation_episode_rate"] <= 0.05
        ),
        "seed_42027_detour_regressions_pass": bool(
            detour_regressions["passed"]
        ),
        "avoidable_loaded_delivery_detours_eq_0": (
            ai_ai["avoidable_loaded_delivery_detour_steps"] == 0
        ),
        "ai_ai_post_policy_action_interventions_eq_0": (
            ai_ai["mean_post_policy_action_interventions"] == 0.0
        ),
        "noisy_post_policy_action_interventions_eq_0": (
            noisy["mean_post_policy_action_interventions"] == 0.0
        ),
        "posthoc_rcpd_artifact_contract_valid": posthoc_rcpd_contract_valid,
        "pure_neural_reference_artifact_contract_valid": (
            reference_contract_valid
        ),
        "parallel_seed_artifact_contract_valid": parallel_seed_contract_valid,
        "explanation_eligible": explanation_eligible,
    }
    if set(checks) != set(FORMAL_ACCEPTANCE_CHECKS):
        raise RuntimeError("Formal acceptance check contract is incomplete.")
    artifact_paths = {
        "model": Path(args.checkpoint),
        "program": Path(args.program),
        "training_summary": summary_path,
        "parallel_seed_library": Path(args.parallel_seed_library),
        "reference_trajectory": Path(args.reference_trajectory),
    }
    artifact_hashes: dict[str, str | None] = {}
    for name, path in artifact_paths.items():
        try:
            artifact_hashes[name] = file_sha256(path)
        except OSError as exc:
            artifact_hashes[name] = None
            artifact_contract_errors.setdefault(name, str(exc))
    payload = {
        "model_version": policy.model_version,
        "runtime_controller": RUNTIME_CONTROLLER,
        "action_execution_version": ACTION_EXECUTION_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes_per_condition": args.episodes,
        "seed_ranges": {
            "ai_ai": [formal_ranges[0][0], formal_ranges[0][1]],
            "noisy_teammate": [formal_ranges[1][0], formal_ranges[1][1]],
            "random": [formal_ranges[2][0], formal_ranges[2][1]],
            "head_on": [formal_ranges[3][0], formal_ranges[3][1]],
        },
        "training_seed_ledger": ledger,
        "artifact_contracts": {
            "program": str(Path(args.program).resolve()),
            "reference_trajectory": str(Path(args.reference_trajectory).resolve()),
            "parallel_seed_library": str(Path(args.parallel_seed_library).resolve()),
            "errors": artifact_contract_errors,
        },
        "artifact_hashes": artifact_hashes,
        "ai_ai": ai_ai,
        "noisy_teammate_20_percent": noisy,
        "random": random_policy,
        "standard_head_on": head_on,
        "bootstrap_95_percent_lower_bounds": {
            "ai_ai_minus_random_deliveries": delivery_difference_lower,
            "ai_ai_minus_random_user_score": score_difference_lower,
        },
        "seed_42027_detour_regressions": detour_regressions,
        "acceptance_checks": checks,
        "formal_candidate": all(checks.values()),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
