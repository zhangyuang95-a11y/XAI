"""Narrow offline calibration for safe WAIT under charger-route priority."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.artifacts import file_sha256
from backend.training.seed_ledger import reserve_evaluation_seed_span
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.policy import MAPPOPolicy, independent_actor_input


@dataclass(frozen=True)
class PriorityYieldConfig:
    episodes: int = 4_096
    anchor_rows: int = 16_384
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 5e-6
    wait_margin: float = 8.0
    wait_margin_weight: float = 4.0
    anchor_kl_weight: float = 12.0
    seed: int = 16_000_004


def _configure_priority_yield_case(
    environment: WarehouseMultiAgentEnv,
    *,
    episode: int,
) -> tuple[str, str]:
    """Create a frozen, symmetric critical-route case without choosing actions."""

    state = environment.get_state()
    priority_id = environment.agent_ids[episode % 2]
    yielding_id = next(
        agent_id for agent_id in environment.agent_ids if agent_id != priority_id
    )
    priority = state.by_id(priority_id)
    yielding = state.by_id(yielding_id)
    from_right = bool((episode // 2) % 2 == 0)
    priority.position = (6, 6) if from_right else (6, 3)
    yielding.position = (6, 4) if from_right else (6, 5)
    priority.battery = float(8 + 2 * ((episode // 4) % 4))
    priority.carrying_task_id = None
    priority.route_commitment_task_id = None
    priority.charge_mode_active = True
    priority.steps_since_charging = 18 + episode % 12
    priority.last_action = "WAIT" if episode % 3 == 0 else (
        "LEFT" if from_right else "RIGHT"
    )
    priority.last_executed_action = priority.last_action

    task = sorted(state.tasks, key=lambda item: item.task_id)[0]
    yielding.battery = float(36 + 2 * ((episode // 16) % 24))
    yielding.carrying_task_id = task.task_id
    yielding.route_commitment_task_id = task.task_id
    yielding.charge_mode_active = False
    yielding.steps_since_charging = 4 + episode % 28
    yielding.last_action = ("LEFT", "RIGHT", "WAIT")[episode % 3]
    yielding.last_executed_action = yielding.last_action
    task.status = "carried"
    task.carrier_agent_id = yielding_id
    task.claimed_frame = state.frame
    task.claimed_battery = yielding.battery
    task.pickup_position = (2, 8) if from_right else (1, 0)
    task.delivery_position = priority.position
    other = sorted(state.tasks, key=lambda item: item.task_id)[1]
    other.status = "available"
    other.carrier_agent_id = None
    other.claimed_frame = None
    other.pickup_position = (3, 0) if from_right else (4, 8)
    other.delivery_position = (5, 2) if from_right else (2, 6)
    environment.set_state(state)
    return priority_id, yielding_id


def collect_priority_yield_rows(
    policy: MAPPOPolicy,
    *,
    episodes: int,
    seed: int,
    only_actor_wait_preferred: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    priority_counts = {agent_id: 0 for agent_id in ("robot_1", "robot_2")}
    skipped = 0
    actor_non_wait_skipped = 0
    wait_index = ACTIONS.index("WAIT")
    requested_rows = int(episodes)
    attempt = 0
    maximum_attempts = max(requested_rows, requested_rows * 8)
    while len(rows) < requested_rows and attempt < maximum_attempts:
        episode = attempt
        attempt += 1
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        environment.reset(seed=int(seed) + episode)
        priority_id, yielding_id = _configure_priority_yield_case(
            environment,
            episode=episode,
        )
        teacher = stable_coordination_actions(environment)
        if teacher[yielding_id] != "WAIT" or teacher[priority_id] == "WAIT":
            skipped += 1
            continue
        observations = environment.observations()
        actor_actions, _ = policy.act(
            observations,
            environment.global_state(),
            deterministic=True,
        )
        if only_actor_wait_preferred and actor_actions[yielding_id] != "WAIT":
            actor_non_wait_skipped += 1
            continue
        row = independent_actor_input(observations[yielding_id])
        if float(row[-len(ACTIONS) + wait_index]) <= 0.5:
            skipped += 1
            continue
        rows.append(row)
        labels.append(wait_index)
        priority_counts[priority_id] += 1
    if not rows:
        raise RuntimeError("No valid critical charger-route yielding rows were built.")
    return (
        np.stack(rows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        {
            "requested_episodes": int(episodes),
            "attempted_cases": attempt,
            "rows": len(rows),
            "skipped": skipped,
            "actor_non_wait_skipped": actor_non_wait_skipped,
            "priority_counts": priority_counts,
            "teacher_actions_submitted_to_environment": 0,
        },
    )


def _fit_with_anchor_kl(
    policy: MAPPOPolicy,
    target_rows: np.ndarray,
    anchor_rows: np.ndarray,
    config: PriorityYieldConfig,
) -> dict[str, float]:
    targets = torch.as_tensor(target_rows, dtype=torch.float32, device=policy.device)
    anchors = torch.as_tensor(anchor_rows, dtype=torch.float32, device=policy.device)
    wait_index = ACTIONS.index("WAIT")
    with torch.no_grad():
        source_anchor_probabilities = torch.softmax(
            policy.masked_actor_logits(anchors), dim=-1
        )

    def target_metrics() -> tuple[float, float]:
        with torch.no_grad():
            probabilities = torch.softmax(policy.masked_actor_logits(targets), dim=-1)
        wait = probabilities[:, wait_index]
        return float(wait.mean().cpu()), float(wait.min().cpu())

    wait_mean_before, wait_min_before = target_metrics()
    parameters = tuple(policy.network.action_scorer.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    rng = np.random.default_rng(config.seed + 1)
    final_loss = torch.tensor(0.0)
    for _ in range(int(config.epochs)):
        target_order = rng.permutation(len(targets))
        anchor_order = rng.permutation(len(anchors))
        steps = max(
            1,
            (max(len(targets), len(anchors)) + config.batch_size - 1)
            // config.batch_size,
        )
        for step in range(steps):
            target_indices = target_order[
                (step * config.batch_size) % len(targets) :
                (step * config.batch_size) % len(targets) + config.batch_size
            ]
            if len(target_indices) == 0:
                target_indices = target_order[: config.batch_size]
            anchor_indices = anchor_order[
                (step * config.batch_size) % len(anchors) :
                (step * config.batch_size) % len(anchors) + config.batch_size
            ]
            if len(anchor_indices) == 0:
                anchor_indices = anchor_order[: config.batch_size]
            target_index = torch.as_tensor(target_indices, device=policy.device)
            anchor_index = torch.as_tensor(anchor_indices, device=policy.device)
            logits = policy.masked_actor_logits(targets[target_index])
            target = torch.full(
                (len(target_index),), wait_index, dtype=torch.long, device=policy.device
            )
            cross_entropy = F.cross_entropy(logits, target)
            strongest_non_wait = torch.cat(
                (logits[:, :wait_index], logits[:, wait_index + 1 :]), dim=-1
            ).max(dim=-1).values
            margin = torch.relu(
                float(config.wait_margin)
                - (logits[:, wait_index] - strongest_non_wait)
            ).mean()
            anchor_logits = policy.masked_actor_logits(anchors[anchor_index])
            anchor_kl = F.kl_div(
                F.log_softmax(anchor_logits, dim=-1),
                source_anchor_probabilities[anchor_index],
                reduction="batchmean",
            )
            loss = (
                cross_entropy
                + float(config.wait_margin_weight) * margin
                + float(config.anchor_kl_weight) * anchor_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,
                policy.algorithm_config.max_grad_norm,
            )
            optimizer.step()
            final_loss = loss.detach()
    wait_mean_after, wait_min_after = target_metrics()
    with torch.no_grad():
        after_anchor = torch.softmax(policy.masked_actor_logits(anchors), dim=-1)
    anchor_probability_l1 = float(
        torch.abs(after_anchor - source_anchor_probabilities).sum(dim=-1).mean().cpu()
    )
    return {
        "target_wait_probability_mean_before": wait_mean_before,
        "target_wait_probability_min_before": wait_min_before,
        "target_wait_probability_mean_after": wait_mean_after,
        "target_wait_probability_min_after": wait_min_after,
        "anchor_probability_mean_l1_change": anchor_probability_l1,
        "final_loss": float(final_loss.cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("training_summary", type=Path)
    parser.add_argument("anchor_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=4_096)
    parser.add_argument("--anchor-rows", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--wait-margin", type=float, default=8.0)
    parser.add_argument("--wait-margin-weight", type=float, default=4.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=16_000_004)
    args = parser.parse_args()
    config = PriorityYieldConfig(
        episodes=args.episodes,
        anchor_rows=args.anchor_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wait_margin=args.wait_margin,
        wait_margin_weight=args.wait_margin_weight,
        anchor_kl_weight=args.anchor_kl_weight,
        seed=args.seed,
    )
    if min(config.episodes, config.anchor_rows, config.epochs, config.batch_size) <= 0:
        parser.error("episodes, anchor rows, epochs, and batch size must be positive")
    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    rows, labels, collection = collect_priority_yield_rows(
        policy,
        episodes=config.episodes,
        seed=config.seed,
    )
    anchor = np.load(args.anchor_dataset, allow_pickle=False)
    all_anchor_rows = np.asarray(anchor["rows"], dtype=np.float32)
    rng = np.random.default_rng(config.seed + 2)
    count = min(config.anchor_rows, len(all_anchor_rows))
    indices = rng.choice(len(all_anchor_rows), size=count, replace=False)
    fit = _fit_with_anchor_kl(policy, rows, all_anchor_rows[indices], config)
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    ledger = summary.get("seed_ledger", {})
    if config.seed != 16_000_004:
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_priority_yield_refit": [16_000_004, 4_096]},
            name="rejected_priority_yield_refit_16000004",
        )
    if config.seed not in {16_000_004, 16_010_004}:
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_priority_yield_refit": [16_010_004, 4_096]},
            name="rejected_priority_yield_refit_16010004",
        )
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_priority_yield_development": [16_300_004, 190_100]},
            name="rejected_priority_yield_development_16300004",
        )
    ledger = reserve_evaluation_seed_span(
        ledger,
        {"superseded_formal": [15_000_004, 190_200]},
        name="superseded_formal_evaluation_15000004",
    )
    ledger = reserve_evaluation_seed_span(
        ledger,
        {
            "release_smoke_ai_1": [15_500_004, 100],
            "release_smoke_ai_2": [15_600_004, 100],
            "release_smoke_noisy": [15_700_004, 100],
        },
        name="rejected_release_smoke_evaluation_15500004",
    )
    ledger = reserve_evaluation_seed_span(
        ledger,
        {"priority_yield_refit": [config.seed, config.episodes]},
        name=f"targeted_priority_yield_refit_{config.seed}",
    )
    superseded_formal_path = (
        args.training_summary.parent
        / "superseded_formal_evaluation_seed_15000004.json"
    )
    if not superseded_formal_path.is_file():
        raise FileNotFoundError(
            "The superseded 15M formal report must be preserved before refitting."
        )
    report = {
        "config": asdict(config),
        "collection": collection,
        "fit": fit,
        "execution_contract": "offline_actor_weight_update_only",
        "teacher_actions_submitted_to_environment": 0,
        "used_formal_or_smoke_observations_for_weight_updates": False,
        "anchor_dataset": str(args.anchor_dataset.resolve()),
        "anchor_dataset_sha256": file_sha256(args.anchor_dataset),
        "superseded_formal_evaluation": {
            "path": str(superseded_formal_path.resolve()),
            "sha256": file_sha256(superseded_formal_path),
            "reason": "candidate changed after an independent release smoke failure",
        },
        "seed_ledger": ledger,
    }
    source_metadata = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(
        args.output,
        training_metadata={
            "base_training": deepcopy(source_metadata),
            "seed_ledger": ledger,
            "rejected_formal_evaluations": deepcopy(
                summary.get("rejected_formal_evaluations", [])
            ),
            "superseded_formal_evaluations": [
                report["superseded_formal_evaluation"]
            ],
            "priority_yield_refit": report,
        },
    )
    np.savez_compressed(
        args.output.with_suffix(".dataset.npz"),
        rows=rows,
        labels=labels,
    )
    args.output.with_suffix(".priority_yield_refit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
