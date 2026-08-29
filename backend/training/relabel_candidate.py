"""Offline DAgger calibration for a completed warehouse Actor candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from backend.training.warehouse import _LearnerStateRelabeler, build_parser
from env.warehouse.policy import MAPPOPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10_200_000)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.rounds <= 0 or args.samples < 4 or args.epochs <= 0:
        parser.error("rounds and epochs must be positive; samples must be >= 4")

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    training_args = build_parser().parse_args([])
    training_args.learner_state_relabel_samples = int(args.samples)
    training_args.learner_state_relabel_replay_capacity = 65_536
    training_args.learner_state_relabel_epochs = int(args.epochs)
    training_args.learner_state_relabel_lr = float(args.learning_rate)
    training_args.learner_state_parameter_scope = "all"
    training_args.learner_state_detour_samples = 128
    training_args.learner_state_detour_search_episodes = 256
    training_args.learner_state_collision_samples = 512
    training_args.learner_state_collision_search_episodes = 512
    training_args.learner_state_charger_cycle_samples = 256
    training_args.learner_state_task_starvation_samples = 256
    training_args.learner_state_commitment_search_episodes = 512
    training_args.learner_state_commitment_curriculum_samples = 8192
    training_args.behavior_cloning_batch_size = 512
    training_args.learner_state_non_wait_margin = 0.0
    training_args.learner_state_non_wait_weight = 0.0
    training_args.learner_state_escape_wait_margin = 5.0
    training_args.learner_state_escape_wait_weight = 2.0
    training_args.learner_state_correction_margin = 8.0
    training_args.learner_state_correction_weight = 5.0
    training_args.learner_state_wait_margin = 5.0
    training_args.learner_state_wait_weight = 2.0

    relabeler = _LearnerStateRelabeler(
        policy,
        policy.environment_config,
        training_args,
    )
    rounds: list[dict[str, object]] = []
    for index in range(int(args.rounds)):
        result = relabeler.run_round(
            seed=int(args.seed) + index * 100_000,
        )
        result["round"] = index + 1
        rounds.append(result)
        print(
            json.dumps(
                {
                    "round": index + 1,
                    "samples": result.get("samples"),
                    "accuracy_before": result.get("accuracy_before"),
                    "accuracy_after": result.get("accuracy_after"),
                    "minimum_critical_category_accuracy": result.get(
                        "minimum_critical_category_accuracy"
                    ),
                    "collision_rows": result.get("targeted_collision_rows"),
                    "final_loss": result.get("final_loss"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    source_metadata = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    report = {
        "execution_contract": "offline_actor_weight_update_only",
        "expert_actions_submitted_to_environment": 0,
        "source": str(args.checkpoint),
        "rounds": rounds,
    }
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "learner_state_candidate_relabeling": report,
        },
    )
    report_path = Path(args.output).with_suffix(".relabel.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
