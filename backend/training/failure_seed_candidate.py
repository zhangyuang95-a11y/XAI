"""Focused offline Actor correction from rejected development seed traces."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from backend.training.learner_replay import fit_actor_supervised
from backend.training.learner_dataset import actor_supported_teacher_action
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.policy import MAPPOPolicy, independent_actor_input


@dataclass(frozen=True)
class FailureSeedConfig:
    seeds: tuple[int, ...]
    anchor_rows: int = 16_384
    failure_repeat: int = 16
    epochs: int = 10
    learning_rate: float = 1e-5
    margin: float = 4.0
    margin_weight: float = 2.0
    seed: int = 10_700_000
    human_ai: bool = False
    participant_noise_probability: float = 0.35
    stochastic_actor: bool = False


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not seeds:
        raise ValueError("At least one rejected development seed is required.")
    return seeds


def _event_category(info: dict[str, Any]) -> str | None:
    if bool(info.get("robot_collision_event", False)):
        return "collision"
    if any(
        event.get("event") == "charger_return_cycle"
        for event in info.get("energy_events", ())
        if isinstance(event, dict)
    ):
        return "charger_cycle"
    if info.get("avoidable_loaded_delivery_detour_agents", ()):
        return "loaded_detour"
    return None


def collect_failure_rows(
    policy: MAPPOPolicy,
    seeds: Iterable[int],
    *,
    human_ai: bool = False,
    participant_noise_probability: float = 0.35,
    stochastic_actor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Relabel only causal precursors of observed Actor failures."""

    seed_values = tuple(int(seed) for seed in seeds)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    categories: list[str] = []
    event_counts = {
        "joint_wait": 0,
        "charger_cycle": 0,
        "loaded_detour": 0,
        "collision": 0,
        "avoidable_detour": 0,
        "reversal": 0,
        "short_cycle": 0,
        "invalid_goal_switch": 0,
    }
    actor_steps = 0
    for seed in seed_values:
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=int(seed))
        if human_ai:
            participant_state = environment.get_state()
            participant_state.participant_controlled_agent_id = (
                policy.environment_config.human_agent_id
            )
            environment.set_state(participant_state)
            observations = environment.observations()
        inference = policy.fork_for_inference(
            seed=int(seed) + (13_000_000 if human_ai else 29_000_000)
        )
        participant_rng = np.random.default_rng(int(seed) + 999)
        history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=8)
        while True:
            state = environment.get_state()
            actor_actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=not stochastic_actor,
                decision_key=(state.episode_id, state.frame),
            )
            teacher = stable_coordination_actions(environment)
            audited_agent_ids = (
                ("robot_2",) if human_ai else environment.agent_ids
            )
            frame_rows = np.stack(
                [
                    independent_actor_input(observations[agent_id])
                    for agent_id in audited_agent_ids
                ]
            ).astype(np.float32, copy=False)
            frame_labels = np.asarray(
                [
                    ACTIONS.index(
                        actor_supported_teacher_action(
                            observations[agent_id],
                            teacher[agent_id],
                        )
                    )
                    for agent_id in audited_agent_ids
                ],
                dtype=np.int64,
            )
            history.append((frame_rows, frame_labels))
            participant_action: str | None = None
            if human_ai:
                participant_action = teacher["robot_1"]
                if participant_rng.random() < participant_noise_probability:
                    participant_mask = environment.action_masks()["robot_1"]
                    legal = [
                        action
                        for action, allowed in zip(ACTIONS, participant_mask)
                        if allowed > 0.5
                    ]
                    participant_action = str(participant_rng.choice(legal))
                actor_actions["robot_1"] = participant_action
            observations, _, terminated, truncated, info = environment.step(
                actor_actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source=(
                        "rejected_human_ai_seed_actor_failure_mining"
                        if human_ai
                        else "rejected_seed_actor_failure_mining"
                    ),
                    participant_overrides=(
                        {"robot_1": participant_action}
                        if participant_action is not None
                        else None
                    ),
                ),
            )
            actor_steps += 1
            if human_ai:
                temporal_categories = (
                    ("joint_wait", "avoidable_wait_agents"),
                    ("avoidable_detour", "avoidable_detour_agents"),
                    ("reversal", "unexplained_reversal_agents"),
                    ("short_cycle", "short_cycle_agents"),
                    ("invalid_goal_switch", "invalid_goal_switch_agents"),
                )
                for category_name, info_key in temporal_categories:
                    if "robot_2" not in info.get(info_key, ()):
                        continue
                    source = (
                        tuple(history)
                        if category_name in {"reversal", "short_cycle"}
                        else ((frame_rows, frame_labels),)
                    )
                    for historical_rows, historical_labels in source:
                        rows.extend(historical_rows)
                        labels.extend(historical_labels)
                        categories.extend(
                            (category_name,) * len(historical_rows)
                        )
                    event_counts[category_name] += 1
                if terminated or truncated:
                    break
                continue
            ineffective_joint_wait = bool(
                all(
                    action == "WAIT"
                    for action in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
                and any(action != "WAIT" for action in teacher.values())
            )
            if ineffective_joint_wait or info.get("avoidable_wait_agents", ()):
                rows.extend(frame_rows)
                labels.extend(frame_labels)
                categories.extend(("joint_wait", "joint_wait"))
                event_counts["joint_wait"] += 1
            category = _event_category(info)
            if category is not None:
                # A return cycle or collision is caused by a short action
                # sequence, not only by the final re-entry frame.  Preserve
                # the preceding frozen states and their safe teacher labels.
                for historical_rows, historical_labels in history:
                    rows.extend(historical_rows)
                    labels.extend(historical_labels)
                    categories.extend((category, category))
                event_counts[category] += 1
            if terminated or truncated:
                break
    if not rows:
        raise RuntimeError("The supplied rejected seeds contained no failures.")
    row_array = np.stack(rows).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    category_array = np.asarray(categories, dtype="<U32")
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
        "seeds": len(seed_values),
        "actor_steps": actor_steps,
        "raw_rows": len(row_array),
        "retained_rows": int(np.sum(keep)),
        "ambiguous_inputs_removed": len(ambiguous),
        "events": event_counts,
        "teacher_actions_submitted_to_environment": 0,
        "human_ai": bool(human_ai),
        "participant_noise_probability": float(
            participant_noise_probability if human_ai else 0.0
        ),
        "stochastic_actor": bool(stochastic_actor),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("anchor_dataset")
    parser.add_argument("output")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchor-rows", type=int, default=16_384)
    parser.add_argument("--failure-repeat", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--margin-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=10_700_000)
    parser.add_argument("--human-ai", action="store_true")
    parser.add_argument("--participant-noise", type=float, default=0.35)
    parser.add_argument("--stochastic-actor", action="store_true")
    args = parser.parse_args()
    config = FailureSeedConfig(
        seeds=_parse_seeds(args.seeds),
        anchor_rows=int(args.anchor_rows),
        failure_repeat=int(args.failure_repeat),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        margin=float(args.margin),
        margin_weight=float(args.margin_weight),
        seed=int(args.seed),
        human_ai=bool(args.human_ai),
        participant_noise_probability=float(args.participant_noise),
        stochastic_actor=bool(args.stochastic_actor),
    )
    if config.anchor_rows < 0 or config.failure_repeat <= 0 or config.epochs <= 0:
        parser.error("anchor rows cannot be negative; repeat and epochs must be positive")
    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    failure_rows, failure_labels, failure_categories, coverage = collect_failure_rows(
        policy,
        config.seeds,
        human_ai=config.human_ai,
        participant_noise_probability=config.participant_noise_probability,
        stochastic_actor=config.stochastic_actor,
    )
    anchor = np.load(args.anchor_dataset, allow_pickle=False)
    anchor_rows = np.asarray(anchor["rows"], dtype=np.float32)
    anchor_labels = np.asarray(anchor["labels"], dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    count = min(config.anchor_rows, len(anchor_rows))
    indices = rng.choice(len(anchor_rows), size=count, replace=False) if count else ()
    repeated_rows = np.repeat(failure_rows, config.failure_repeat, axis=0)
    repeated_labels = np.repeat(failure_labels, config.failure_repeat, axis=0)
    repeated_categories = np.repeat(
        failure_categories, config.failure_repeat, axis=0
    )
    rows = np.concatenate((anchor_rows[indices], repeated_rows), axis=0)
    labels = np.concatenate((anchor_labels[indices], repeated_labels), axis=0)
    categories = np.concatenate(
        (np.full(count, "anchor", dtype="<U32"), repeated_categories), axis=0
    )
    failure_mask = categories != "anchor"
    wait_index = ACTIONS.index("WAIT")
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
        escape_wait_mask=(categories == "joint_wait") & (labels != wait_index),
        correction_margin=config.margin,
        correction_weight=config.margin_weight,
        correction_mask=failure_mask,
        wait_margin=config.margin,
        wait_weight=config.margin_weight,
        wait_margin_mask=failure_mask & (labels == wait_index),
        seed=config.seed,
        parameter_scope="action_heads_only",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_metadata = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    report = {
        "config": asdict(config),
        "coverage": coverage,
        "anchor_rows": count,
        "failure_rows": len(failure_rows),
        "effective_failure_rows": len(repeated_rows),
        "execution_contract": "offline_actor_weight_update_only",
        "expert_actions_submitted_to_environment": 0,
        "fit": fit,
    }
    policy.save(
        output,
        training_metadata={
            "base_training": dict(source_metadata),
            "rejected_seed_failure_refit": report,
        },
    )
    dataset_path = output.with_suffix(".dataset.npz")
    np.savez_compressed(
        dataset_path, rows=rows, labels=labels, categories=categories
    )
    report_path = output.with_suffix(".failure_refit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
