"""Focused offline correction for the two-phase occupied-charger handoff."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.artifacts import file_sha256
from backend.training.learner_replay import (
    fit_actor_supervised,
    supervised_category_accuracy,
)
from backend.training.seed_ledger import reserve_evaluation_seed_span
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.navigation import ACTIONS
from env.warehouse.policy import MAPPOPolicy, independent_actor_input
from env.warehouse.scenario_evaluation import (
    _configure_valid_occupied_charger_case,
)
from env.warehouse.environment import WarehouseMultiAgentEnv


@dataclass(frozen=True)
class HandoffRefitConfig:
    episodes: int = 2_048
    epochs: int = 12
    learning_rate: float = 5e-6
    margin: float = 2.0
    margin_weight: float = 2.0
    seed: int = 12_700_004
    parameter_scope: str = "structured"


def collect_handoff_rows(
    policy: MAPPOPolicy,
    *,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Label only S_t initial handoff states; no teacher action is executed."""

    rows: list[np.ndarray] = []
    labels: list[int] = []
    categories: list[str] = []
    label_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    adjusted_profiles = 0
    actor_mismatches = 0
    for episode in range(int(episodes)):
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        environment.reset(seed=int(seed) + episode)
        _, _, priority, _, adjustments = _configure_valid_occupied_charger_case(
            environment,
            episode=episode,
        )
        adjusted_profiles += int(adjustments > 0)
        priority_counts[priority] += 1
        observations = environment.observations()
        actor_actions, _ = policy.act(
            observations,
            environment.global_state(),
            deterministic=True,
        )
        teacher = stable_coordination_actions(environment)
        for agent_id in environment.agent_ids:
            label = str(teacher[agent_id])
            rows.append(independent_actor_input(observations[agent_id]))
            labels.append(ACTIONS.index(label))
            categories.append("charger_queue")
            label_counts[label] += 1
            actor_mismatches += int(actor_actions[agent_id] != label)
    return (
        np.stack(rows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        np.asarray(categories, dtype="<U32"),
        {
            "episodes": int(episodes),
            "rows": len(rows),
            "actor_mismatches_before": actor_mismatches,
            "label_counts": dict(sorted(label_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
            "adjusted_profile_episodes": adjusted_profiles,
            "teacher_actions_submitted_to_environment": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("training_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=2_048)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--margin-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=12_700_004)
    parser.add_argument("--diagnostic-seed", type=int)
    parser.add_argument("--diagnostic-episodes", type=int, default=0)
    parser.add_argument(
        "--diagnostic-seed-range",
        action="append",
        nargs=2,
        type=int,
        default=[],
        metavar=("START", "COUNT"),
    )
    parser.add_argument(
        "--development-diagnostic",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--parameter-scope",
        choices=("structured", "action_heads_only"),
        default="structured",
    )
    args = parser.parse_args()
    config = HandoffRefitConfig(
        episodes=int(args.episodes),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        margin=float(args.margin),
        margin_weight=float(args.margin_weight),
        seed=int(args.seed),
        parameter_scope=str(args.parameter_scope),
    )
    if config.episodes <= 0 or config.epochs <= 0:
        parser.error("episodes and epochs must be positive")
    if (args.diagnostic_seed is None) != (args.diagnostic_episodes <= 0):
        parser.error(
            "diagnostic-seed and a positive diagnostic-episodes are required together"
        )
    source_metadata = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    ).get("training_metadata", {})
    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    rows, labels, categories, collection = collect_handoff_rows(
        policy,
        episodes=config.episodes,
        seed=config.seed,
    )
    before = supervised_category_accuracy(policy, rows, labels, categories)
    correction = np.ones(len(rows), dtype=bool)
    wait_mask = labels == ACTIONS.index("WAIT")
    fit = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=config.epochs,
        batch_size=256,
        learning_rate=config.learning_rate,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=0.0,
        escape_wait_weight=0.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=config.margin,
        correction_weight=config.margin_weight,
        correction_mask=correction,
        wait_margin=config.margin,
        wait_weight=config.margin_weight,
        wait_margin_mask=wait_mask,
        seed=config.seed,
        parameter_scope=config.parameter_scope,
    )
    after = supervised_category_accuracy(policy, rows, labels, categories)
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    source_seed_ledger = (
        source_metadata.get("seed_ledger", {})
        if isinstance(source_metadata, dict)
        else {}
    )
    seed_ledger = source_seed_ledger or summary.get("seed_ledger", {})
    if args.diagnostic_seed is not None:
        seed_ledger = reserve_evaluation_seed_span(
            seed_ledger,
            {
                "targeted_handoff_diagnosis": [
                    int(args.diagnostic_seed),
                    int(args.diagnostic_episodes),
                ]
            },
            name=(
                "targeted_occupied_charger_handoff_development_diagnosis_"
                f"{int(args.diagnostic_seed)}"
            ),
        )
    for diagnostic_seed, diagnostic_episodes in args.diagnostic_seed_range:
        if diagnostic_episodes <= 0:
            parser.error("diagnostic seed range counts must be positive")
        seed_ledger = reserve_evaluation_seed_span(
            seed_ledger,
            {
                "targeted_handoff_diagnosis": [
                    int(diagnostic_seed),
                    int(diagnostic_episodes),
                ]
            },
            name=(
                "targeted_occupied_charger_handoff_development_diagnosis_"
                f"{int(diagnostic_seed)}"
            ),
        )
    seed_ledger = reserve_evaluation_seed_span(
        seed_ledger,
        {"targeted_handoff_refit": [config.seed, config.episodes]},
        name=f"targeted_occupied_charger_handoff_refit_{config.seed}",
    )
    diagnostic_evaluations: list[dict[str, Any]] = []
    for diagnostic_path in args.development_diagnostic:
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        checks = diagnostic.get("acceptance_checks", {})
        diagnostic_evaluations.append(
            {
                "path": str(diagnostic_path.resolve()),
                "sha256": file_sha256(diagnostic_path),
                "formal_candidate": bool(diagnostic.get("formal_candidate", False)),
                "failed_acceptance_checks": sorted(
                    key
                    for key, passed in checks.items()
                    if not bool(passed)
                    and key
                    not in {
                        "episodes_per_condition_ge_1000",
                        "multi_partner_episodes_ge_1000",
                    }
                ),
            }
        )
    report = {
        "config": asdict(config),
        "collection": collection,
        "accuracy_before": before,
        "accuracy_after": after,
        "fit": fit,
        "execution_contract": "offline_actor_weight_update_only",
        "teacher_actions_submitted_to_environment": 0,
        "used_rejected_formal_observations": False,
        "formal_seed_use": False,
        "development_diagnostics": diagnostic_evaluations,
        "seed_ledger": seed_ledger,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "seed_ledger": seed_ledger,
            "rejected_formal_evaluations": list(
                summary.get("rejected_formal_evaluations", [])
            ),
            "occupied_charger_handoff_refit": report,
        },
    )
    dataset = args.output.with_suffix(".dataset.npz")
    np.savez_compressed(
        dataset,
        rows=rows,
        labels=labels,
        categories=categories,
    )
    report_path = args.output.with_suffix(".handoff_refit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
