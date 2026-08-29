"""Refit a warehouse Actor from a saved, causally collected imitation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.training.learner_replay import (
    fit_actor_supervised,
    supervised_category_accuracy,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse.policy import MAPPOPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--teammate-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--parameter-scope",
        choices=(
            "all",
            "structured",
            "actor_without_teammate_predictor",
            "action_heads_only",
            "teammate_predictor_only",
            "teammate_context_predictor_only",
        ),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=10_530_000)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as archive:
        rows = archive["rows"].copy()
        labels = archive["labels"].copy()
        teammate_labels = archive["teammate_labels"].copy()
        categories = archive["categories"].copy()
    label_sets: dict[bytes, set[int]] = {}
    for row, label in zip(rows, labels):
        label_sets.setdefault(row.tobytes(), set()).add(int(label))
    ambiguous_inputs = {
        key for key, values in label_sets.items() if len(values) > 1
    }
    keep = np.fromiter(
        (row.tobytes() not in ambiguous_inputs for row in rows),
        dtype=bool,
        count=len(rows),
    )
    ambiguous_rows_removed = int(np.sum(~keep))
    rows = rows[keep]
    labels = labels[keep]
    teammate_labels = teammate_labels[keep]
    categories = categories[keep]
    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    fit = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=int(args.epochs),
        batch_size=512,
        learning_rate=float(args.learning_rate),
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=float(args.margin),
        escape_wait_weight=float(args.margin_weight),
        escape_wait_mask=categories == "joint_wait",
        correction_margin=float(args.margin),
        correction_weight=float(args.margin_weight),
        correction_mask=np.ones(len(rows), dtype=bool),
        wait_margin=float(args.margin),
        wait_weight=float(args.margin_weight),
        wait_margin_mask=labels == ACTIONS.index("WAIT"),
        seed=int(args.seed),
        parameter_scope=str(args.parameter_scope),
        teammate_labels=teammate_labels,
        teammate_loss_weight=float(args.teammate_loss_weight),
        action_loss_weight=float(args.action_loss_weight),
    )
    report = {
        "execution_contract": "offline_actor_weight_update_only",
        "expert_actions_submitted_to_environment": 0,
        "source": str(args.checkpoint),
        "dataset": str(args.dataset),
        "rows": int(len(rows)),
        "ambiguous_inputs_removed": int(len(ambiguous_inputs)),
        "ambiguous_rows_removed": ambiguous_rows_removed,
        "category_accuracy": supervised_category_accuracy(
            policy, rows, labels, categories
        ),
        **fit,
    }
    source_metadata = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "saved_dataset_refit": report,
        },
    )
    report_path = Path(args.output).with_suffix(".refit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(report_path)


if __name__ == "__main__":
    main()
