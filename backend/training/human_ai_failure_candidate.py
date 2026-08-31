"""Offline correction of causal Robot-2 failures on Human-AI trajectories.

The current participant action is applied only after the Actor has sampled
from S_t.  Labels come from the S_t-only robust coordination teacher and are
never submitted to the environment.  When an observable failure occurs, the
current row and up to four causal precursor rows are retained for a narrow
action-head update anchored by ordinary production-map imitation rows.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from backend.training.learner_dataset import actor_supported_teacher_action
from backend.training.learner_replay import fit_actor_supervised
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.partner_policies import robust_partner_robot_two_action
from env.warehouse.policy import MAPPOPolicy, independent_actor_input


@dataclass(frozen=True)
class HumanAIFailureConfig:
    seeds: tuple[int, ...]
    participant_noise_probability: float = 0.35
    precursor_frames: int = 4
    anchor_rows: int = 8_192
    failure_repeat: int = 12
    epochs: int = 8
    learning_rate: float = 1e-5
    margin: float = 3.0
    margin_weight: float = 2.0
    seed: int = 31_200_000


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not seeds:
        raise ValueError("At least one Human-AI failure seed is required.")
    return seeds


def _failure_categories(
    info: Mapping[str, Any],
    *,
    ineffective_wait_streak: int,
) -> tuple[str, ...]:
    categories: list[str] = []
    if bool(info.get("robot_collision_event", False)):
        categories.append("collision")
    if "robot_2" in info.get("avoidable_wait_agents", ()):
        categories.append("avoidable_wait")
    if "robot_2" in info.get("avoidable_detour_agents", ()):
        categories.append("avoidable_detour")
    if "robot_2" in info.get("unexplained_reversal_agents", ()):
        categories.append("unexplained_reversal")
    if "robot_2" in info.get("short_cycle_agents", ()):
        categories.append("short_cycle")
    if "robot_2" in info.get("invalid_goal_switch_agents", ()):
        categories.append("invalid_goal_switch")
    if ineffective_wait_streak >= 3:
        categories.append("joint_wait")
    return tuple(categories)


def collect_human_ai_failure_rows(
    policy: MAPPOPolicy,
    seeds: Iterable[int],
    *,
    participant_noise_probability: float,
    precursor_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    categories: list[str] = []
    counts: dict[str, int] = {}
    actor_steps = 0

    for seed in (int(value) for value in seeds):
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=seed)
        participant_state = environment.get_state()
        participant_state.participant_controlled_agent_id = (
            environment.config.human_agent_id
        )
        environment.set_state(participant_state)
        observations = environment.observations()
        inference = policy.fork_for_inference(seed=seed + 13_000_000)
        rng = np.random.default_rng(seed + 999)
        history: deque[tuple[np.ndarray, int]] = deque(
            maxlen=max(1, int(precursor_frames) + 1)
        )
        ineffective_wait_streak = 0

        while True:
            state = environment.get_state()
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=False,
                decision_key=(state.episode_id, state.frame),
            )
            preferred = stable_coordination_actions(environment)["robot_2"]
            robust_label = robust_partner_robot_two_action(
                environment,
                preferred_action=preferred,
            )
            # Worst-case one-step robustness would WAIT forever when the
            # participant could hypothetically enter the priority robot's next
            # cell on every future frame.  Once two stationary outcomes are
            # already visible in S_t and the public coordination teacher has a
            # non-WAIT priority move, train that causal deadlock escape.  This
            # uses observed history only; the participant's current command is
            # still sampled afterwards.
            if (
                int(getattr(state, "ineffective_joint_wait_streak", 0)) >= 2
                and preferred != "WAIT"
            ):
                robust_label = preferred
            robust_label = actor_supported_teacher_action(
                observations["robot_2"],
                robust_label,
            )
            current = (
                independent_actor_input(observations["robot_2"]).astype(
                    np.float32,
                    copy=True,
                ),
                ACTIONS.index(robust_label),
            )
            history.append(current)

            participant_action = stable_coordination_actions(environment)["robot_1"]
            if rng.random() < float(participant_noise_probability):
                mask = environment.action_masks()["robot_1"]
                legal = [
                    action
                    for action, allowed in zip(ACTIONS, mask)
                    if allowed > 0.5
                ]
                participant_action = str(rng.choice(legal))
            actions["robot_1"] = participant_action
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="human_ai_failure_mining",
                    participant_overrides={"robot_1": participant_action},
                ),
            )
            actor_steps += 1
            ineffective_joint_wait = bool(
                all(
                    str(action) == "WAIT"
                    for action in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
            )
            ineffective_wait_streak = (
                ineffective_wait_streak + 1 if ineffective_joint_wait else 0
            )
            observed = _failure_categories(
                info,
                ineffective_wait_streak=ineffective_wait_streak,
            )
            if observed:
                category = "+".join(observed)
                for row, label in history:
                    rows.append(row.copy())
                    labels.append(int(label))
                    categories.append(category)
                for item in observed:
                    counts[item] = counts.get(item, 0) + 1
            if terminated or truncated:
                break

    if not rows:
        raise RuntimeError("The supplied trajectories contained no attributable failures.")
    row_array = np.stack(rows).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    category_array = np.asarray(categories, dtype="<U128")
    label_sets: dict[bytes, set[int]] = {}
    for row, label in zip(row_array, label_array):
        label_sets.setdefault(row.tobytes(), set()).add(int(label))
    ambiguous = {key for key, values in label_sets.items() if len(values) > 1}
    keep = np.fromiter(
        (row.tobytes() not in ambiguous for row in row_array),
        dtype=bool,
        count=len(row_array),
    )
    return row_array[keep], label_array[keep], category_array[keep], {
        "seeds": len(tuple(int(value) for value in seeds)),
        "actor_steps": actor_steps,
        "raw_rows": len(row_array),
        "retained_rows": int(np.sum(keep)),
        "ambiguous_inputs_removed": len(ambiguous),
        "events": counts,
        "teacher_actions_submitted_to_environment": 0,
        "participant_action_visible_to_actor_before_decision": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("anchor_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--participant-noise", type=float, default=0.35)
    parser.add_argument("--precursor-frames", type=int, default=4)
    parser.add_argument("--anchor-rows", type=int, default=8_192)
    parser.add_argument("--failure-repeat", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--margin-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=31_200_000)
    args = parser.parse_args()
    config = HumanAIFailureConfig(
        seeds=_parse_seeds(args.seeds),
        participant_noise_probability=float(args.participant_noise),
        precursor_frames=int(args.precursor_frames),
        anchor_rows=int(args.anchor_rows),
        failure_repeat=int(args.failure_repeat),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        margin=float(args.margin),
        margin_weight=float(args.margin_weight),
        seed=int(args.seed),
    )
    if not 0.0 <= config.participant_noise_probability <= 1.0:
        parser.error("participant noise must be in [0, 1]")
    if config.anchor_rows < 0 or config.failure_repeat <= 0 or config.epochs <= 0:
        parser.error("anchor rows cannot be negative; repeat and epochs must be positive")

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    failure_rows, failure_labels, failure_categories, coverage = (
        collect_human_ai_failure_rows(
            policy,
            config.seeds,
            participant_noise_probability=config.participant_noise_probability,
            precursor_frames=config.precursor_frames,
        )
    )
    with np.load(args.anchor_dataset, allow_pickle=False) as archive:
        anchor_rows = np.asarray(archive["rows"], dtype=np.float32)
        anchor_labels = np.asarray(archive["labels"], dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    anchor_count = min(config.anchor_rows, len(anchor_rows))
    indices = (
        rng.choice(len(anchor_rows), size=anchor_count, replace=False)
        if anchor_count
        else np.empty((0,), dtype=np.int64)
    )
    repeated_rows = np.repeat(failure_rows, config.failure_repeat, axis=0)
    repeated_labels = np.repeat(failure_labels, config.failure_repeat, axis=0)
    repeated_categories = np.repeat(
        failure_categories,
        config.failure_repeat,
        axis=0,
    )
    rows = np.concatenate((anchor_rows[indices], repeated_rows), axis=0)
    labels = np.concatenate((anchor_labels[indices], repeated_labels), axis=0)
    categories = np.concatenate(
        (
            np.full(anchor_count, "anchor", dtype="<U128"),
            repeated_categories,
        ),
        axis=0,
    )
    wait_index = ACTIONS.index("WAIT")
    joint_wait_mask = np.char.find(categories, "joint_wait") >= 0
    # This calibration changes only the isolated deadlock-escape head.  Rows
    # outside an observable joint-wait standoff are retained as regression
    # evidence but cannot activate the causal gate or update ordinary action
    # scoring.
    correction_mask = joint_wait_mask
    fit = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=config.epochs,
        batch_size=256,
        learning_rate=config.learning_rate,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=config.margin,
        escape_wait_weight=config.margin_weight,
        escape_wait_mask=joint_wait_mask & (labels != wait_index),
        correction_margin=config.margin,
        correction_weight=config.margin_weight,
        correction_mask=correction_mask,
        wait_margin=config.margin,
        wait_weight=config.margin_weight,
        wait_margin_mask=joint_wait_mask & (labels == wait_index),
        seed=config.seed,
        parameter_scope="deadlock_escape_only",
    )
    report = {
        "config": asdict(config),
        "coverage": coverage,
        "anchor_rows": anchor_count,
        "failure_rows": len(failure_rows),
        "effective_failure_rows": len(repeated_rows),
        "execution_contract": "offline_actor_weight_update_only",
        "teacher_actions_submitted_to_environment": 0,
        "fit": fit,
    }
    source_metadata = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    ).get("training_metadata", {})
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "human_ai_failure_refit": report,
        },
    )
    np.savez_compressed(
        args.output.with_suffix(".dataset.npz"),
        rows=rows,
        labels=labels,
        categories=categories,
    )
    args.output.with_suffix(".human_ai_failure.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
