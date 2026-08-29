"""High-margin offline imitation calibration for a warehouse Actor."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.partner_policies import (
    PARTNER_PROFILES,
    participant_surrogate_action,
    robust_partner_robot_two_action,
)
from backend.training.learner_dataset import (
    collect_actor_collision_correction_dataset,
    collect_learner_state_relabel_dataset,
)
from backend.training.learner_replay import (
    fit_actor_supervised,
    supervised_category_accuracy,
)
from backend.training.warehouse import _collect_behavior_cloning_dataset
from env.warehouse.navigation import ACTIONS
from env.warehouse.observations import observation_dim
from env.warehouse.policy import MAPPOPolicy, independent_actor_input


def _collect_central_teacher_dagger_dataset(
    policy: MAPPOPolicy,
    *,
    sample_count: int,
    seed: int,
    partner_probability: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    if sample_count <= 0:
        return (
            np.empty(
                (0, observation_dim(policy.environment_config)),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U32"),
            {"episodes": 0, "actor_steps": 0, "failure_rows": 0},
        )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    teammate_labels: list[int] = []
    categories: list[str] = []
    rng = np.random.default_rng(int(seed) + 17_000_000)
    episode = 0
    actor_steps = 0
    failure_rows = 0
    while len(rows) < int(sample_count):
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=int(seed) + episode)
        mode_rng = np.random.default_rng(int(seed) + episode + 41_000_000)
        profile = (
            PARTNER_PROFILES[episode % len(PARTNER_PROFILES)]
            if mode_rng.random() < float(partner_probability)
            else None
        )
        if profile is not None:
            participant_state = environment.get_state()
            participant_state.participant_controlled_agent_id = (
                environment.config.human_agent_id
            )
            environment.set_state(participant_state)
            observations = environment.observations()
        inference = policy.fork_for_inference(seed=int(seed) + episode + 29_000_000)
        precursor_history: deque[
            tuple[tuple[np.ndarray, int, int], ...]
        ] = deque(maxlen=8)
        while len(rows) < int(sample_count):
            state = environment.get_state()
            participant_action = (
                participant_surrogate_action(
                    environment,
                    profile=profile,
                    rng=rng,
                )
                if profile is not None
                else None
            )
            actions, _ = inference.act(
                observations,
                environment.global_state(),
                deterministic=False,
                decision_key=(state.episode_id, state.frame),
            )
            if participant_action is not None:
                actions[environment.config.human_agent_id] = participant_action
            teacher = stable_coordination_actions(environment)
            if profile is not None:
                teacher["robot_2"] = robust_partner_robot_two_action(
                    environment,
                    preferred_action=actions["robot_2"],
                )
            predicted_collision = bool(
                environment._resolve_motion(state, actions)[3]
            )
            ineffective_joint_wait = bool(
                all(action == "WAIT" for action in actions.values())
                and not all(action == "WAIT" for action in teacher.values())
            )
            category = (
                "collision"
                if predicted_collision
                else "joint_wait"
                if ineffective_joint_wait
                else "central_dagger"
            )
            agent_ids = ("robot_2",) if profile is not None else environment.agent_ids
            frame_entries: list[tuple[np.ndarray, int, int]] = []
            for agent_id in agent_ids:
                row = independent_actor_input(observations[agent_id])
                label = ACTIONS.index(teacher[agent_id])
                teammate_id = "robot_2" if agent_id == "robot_1" else "robot_1"
                teammate_label = (
                    ACTIONS.index(participant_action)
                    if profile is not None and agent_id == "robot_2"
                    else ACTIONS.index(teacher[teammate_id])
                )
                rows.append(row)
                labels.append(label)
                teammate_labels.append(teammate_label)
                categories.append(category)
                frame_entries.append((row, label, teammate_label))
                failure_rows += int(category != "central_dagger")
                if len(rows) >= int(sample_count):
                    break
            if category != "central_dagger":
                for historical_frame in precursor_history:
                    for row, label, teammate_label in historical_frame:
                        if len(rows) >= int(sample_count):
                            break
                        rows.append(row.copy())
                        labels.append(label)
                        teammate_labels.append(teammate_label)
                        categories.append(f"{category}_precursor")
                        failure_rows += 1
                    if len(rows) >= int(sample_count):
                        break
            precursor_history.append(tuple(frame_entries))
            observations, _, terminated, truncated, info = environment.step(actions)
            charger_cycle = any(
                event.get("event") == "charger_return_cycle"
                for event in info.get("energy_events", ())
                if isinstance(event, dict)
            )
            if charger_cycle:
                for historical_frame in precursor_history:
                    for row, label, teammate_label in historical_frame:
                        if len(rows) >= int(sample_count):
                            break
                        rows.append(row.copy())
                        labels.append(label)
                        teammate_labels.append(teammate_label)
                        categories.append("charger_cycle_precursor")
                        failure_rows += 1
                    if len(rows) >= int(sample_count):
                        break
            actor_steps += 1
            if terminated or truncated:
                break
        episode += 1
    return (
        np.stack(rows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        np.asarray(teammate_labels, dtype=np.int64),
        np.asarray(categories, dtype="<U32"),
        {
            "episodes": episode,
            "actor_steps": actor_steps,
            "failure_rows": failure_rows,
            "teacher_actions_submitted_to_environment": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=10_480_000)
    parser.add_argument("--teacher-samples", type=int, default=65_536)
    parser.add_argument("--learner-samples", type=int, default=16_384)
    parser.add_argument("--central-dagger-samples", type=int, default=0)
    parser.add_argument(
        "--central-dagger-partner-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--dagger-failure-repeat", type=int, default=1)
    parser.add_argument("--teammate-loss-weight", type=float, default=0.0)
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
            "teammate_forecast_only",
            "participant_partner_only",
        ),
        default="all",
    )
    parser.add_argument("--collision-samples", type=int, default=512)
    parser.add_argument("--collision-search-episodes", type=int, default=1024)
    parser.add_argument("--collision-repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=8.0)
    parser.add_argument("--margin-weight", type=float, default=5.0)
    args = parser.parse_args()
    if (
        args.teacher_samples <= 0
        or args.epochs <= 0
        or args.learner_samples < 0
        or args.central_dagger_samples < 0
    ):
        parser.error(
            "teacher samples and epochs must be positive; learner samples cannot be negative"
        )
    if args.collision_repeat <= 0:
        parser.error("collision repeat must be positive")
    if args.dagger_failure_repeat <= 0:
        parser.error("DAgger failure repeat must be positive")
    if not 0.0 <= args.central_dagger_partner_probability <= 1.0:
        parser.error("central DAgger partner probability must be in [0, 1]")

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    teacher_rows, teacher_labels, _, *_ = _collect_behavior_cloning_dataset(
        policy.environment_config,
        sample_count=int(args.teacher_samples),
        seed=int(args.seed),
    )
    if args.learner_samples:
        learner_rows, learner_labels, learner_categories, learner_coverage = (
            collect_learner_state_relabel_dataset(
                policy,
                policy.environment_config,
                sample_count=int(args.learner_samples),
                seed=int(args.seed) + 100_000,
            )
        )
    else:
        learner_rows = np.empty((0, teacher_rows.shape[1]), dtype=np.float32)
        learner_labels = np.empty((0,), dtype=np.int64)
        learner_categories = np.empty((0,), dtype="<U32")
        learner_coverage = {"disabled": True}
    (
        dagger_rows,
        dagger_labels,
        dagger_teammate_labels,
        dagger_categories,
        dagger_coverage,
    ) = (
        _collect_central_teacher_dagger_dataset(
            policy,
            sample_count=int(args.central_dagger_samples),
            seed=int(args.seed) + 200_000,
            partner_probability=float(
                args.central_dagger_partner_probability
            ),
        )
    )
    collision_rows, collision_labels, collision_categories, collision_coverage = (
        collect_actor_collision_correction_dataset(
            policy,
            policy.environment_config,
            sample_count=int(args.collision_samples),
            maximum_episodes=int(args.collision_search_episodes),
            seed=int(args.seed) + 40_000_000,
        )
    )
    row_groups = [teacher_rows, learner_rows]
    label_groups = [teacher_labels, learner_labels]
    teammate_label_groups = [
        np.full(len(teacher_rows), -1, dtype=np.int64),
        np.full(len(learner_rows), -1, dtype=np.int64),
    ]
    category_groups = [
        np.full(len(teacher_rows), "ordinary", dtype="<U32"),
        learner_categories,
    ]
    if len(dagger_rows):
        failure_mask = dagger_categories != "central_dagger"
        row_groups.append(dagger_rows[~failure_mask])
        label_groups.append(dagger_labels[~failure_mask])
        teammate_label_groups.append(
            dagger_teammate_labels[~failure_mask]
        )
        category_groups.append(dagger_categories[~failure_mask])
        if bool(failure_mask.any()):
            row_groups.append(
                np.repeat(
                    dagger_rows[failure_mask],
                    int(args.dagger_failure_repeat),
                    axis=0,
                )
            )
            label_groups.append(
                np.repeat(
                    dagger_labels[failure_mask],
                    int(args.dagger_failure_repeat),
                    axis=0,
                )
            )
            teammate_label_groups.append(
                np.repeat(
                    dagger_teammate_labels[failure_mask],
                    int(args.dagger_failure_repeat),
                    axis=0,
                )
            )
            category_groups.append(
                np.repeat(
                    dagger_categories[failure_mask],
                    int(args.dagger_failure_repeat),
                    axis=0,
                )
            )
    if len(collision_rows):
        row_groups.append(
            np.repeat(collision_rows, int(args.collision_repeat), axis=0)
        )
        label_groups.append(
            np.repeat(collision_labels, int(args.collision_repeat), axis=0)
        )
        teammate_label_groups.append(
            np.full(
                len(collision_labels) * int(args.collision_repeat),
                -1,
                dtype=np.int64,
            )
        )
        category_groups.append(
            np.repeat(collision_categories, int(args.collision_repeat), axis=0)
        )
    rows = np.concatenate(row_groups, axis=0)
    labels = np.concatenate(label_groups, axis=0)
    teammate_labels = np.concatenate(teammate_label_groups, axis=0)
    categories = np.concatenate(category_groups, axis=0)
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
    dropped_ambiguous_rows = int(np.sum(~keep))
    if dropped_ambiguous_rows:
        rows = rows[keep]
        labels = labels[keep]
        teammate_labels = teammate_labels[keep]
        categories = categories[keep]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = output_path.with_suffix(".dataset.npz")
    np.savez_compressed(
        dataset_path,
        rows=rows,
        labels=labels,
        teammate_labels=teammate_labels,
        categories=categories,
    )
    label_sets = {}
    for row, label in zip(rows, labels):
        label_sets.setdefault(row.tobytes(), set()).add(int(label))
    conflicting_inputs = sum(len(values) > 1 for values in label_sets.values())
    if conflicting_inputs:
        raise RuntimeError("Ambiguous-input filtering failed closed.")
    correction_mask = np.ones(len(rows), dtype=bool)
    escape_wait_mask = categories == "joint_wait"
    wait_margin_mask = labels == ACTIONS.index("WAIT")
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
        escape_wait_mask=escape_wait_mask,
        correction_margin=float(args.margin),
        correction_weight=float(args.margin_weight),
        correction_mask=correction_mask,
        wait_margin=float(args.margin),
        wait_weight=float(args.margin_weight),
        wait_margin_mask=wait_margin_mask,
        seed=int(args.seed),
        parameter_scope=str(args.parameter_scope),
        teammate_labels=teammate_labels,
        teammate_loss_weight=float(args.teammate_loss_weight),
        action_loss_weight=float(args.action_loss_weight),
    )
    observations = torch.as_tensor(
        rows, dtype=torch.float32, device=policy.device
    )
    targets = torch.as_tensor(
        labels, dtype=torch.long, device=policy.device
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(observations)
        probabilities = torch.softmax(logits, dim=-1)
        target_probability = float(
            probabilities.gather(1, targets[:, None]).mean().cpu()
        )
        minimum_target_probability = float(
            probabilities.gather(1, targets[:, None]).min().cpu()
        )
    report = {
        "execution_contract": "offline_actor_weight_update_only",
        "expert_actions_submitted_to_environment": 0,
        "source": str(args.checkpoint),
        "teacher_rows": int(len(teacher_rows)),
        "learner_rows": int(len(learner_rows)),
        "central_dagger_rows": int(len(dagger_rows)),
        "dagger_failure_repeat": int(args.dagger_failure_repeat),
        "dagger_coverage": dagger_coverage,
        "collision_rows": int(len(collision_rows)),
        "effective_collision_rows": int(
            len(collision_rows) * int(args.collision_repeat)
        ),
        "collision_repeat": int(args.collision_repeat),
        "unique_inputs": int(len(label_sets)),
        "conflicting_inputs": int(conflicting_inputs),
        "ambiguous_inputs_removed": int(len(ambiguous_inputs)),
        "ambiguous_rows_removed": dropped_ambiguous_rows,
        "dataset": str(dataset_path),
        "learner_coverage": learner_coverage,
        "collision_coverage": collision_coverage,
        "category_accuracy": supervised_category_accuracy(
            policy, rows, labels, categories
        ),
        "mean_target_probability": target_probability,
        "minimum_target_probability": minimum_target_probability,
        "margin": float(args.margin),
        "margin_weight": float(args.margin_weight),
        **fit,
    }
    source_metadata = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "high_margin_imitation": report,
        },
    )
    report_path = Path(args.output).with_suffix(".imitation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(report_path)


if __name__ == "__main__":
    main()
