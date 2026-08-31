"""Narrow offline Actor calibration for causal one-lane deadlock clearance.

The rejected formal evaluation is diagnosis evidence only.  Weight updates use
fresh, structurally generated states from a separately reserved seed range.
The offline teacher is never submitted to ``environment.step``.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
class DeadlockBreakConfig:
    cases: int = 8_192
    anchor_rows: int = 16_384
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-6
    move_margin: float = 3.0
    move_margin_weight: float = 2.0
    anchor_kl_weight: float = 25.0
    seed: int = 20_000_004


def _configure_structural_case(
    environment: WarehouseMultiAgentEnv,
    *,
    case: int,
) -> str:
    """Place robots in a causal contested-aisle state using only S_t fields."""

    state = environment.get_state()
    family_index = case % 3
    if family_index == 0:
        horizontal_pairs = tuple(
            ((row, column), (row, column + 1))
            for row in range(environment.layout.rows)
            for column in range(environment.layout.cols - 1)
            if environment.layout.is_passable((row, column))
            and environment.layout.is_passable((row, column + 1))
            and (row, column) not in environment.layout.robot_start_positions
            and (row, column + 1) not in environment.layout.robot_start_positions
        )
        positions = list(horizontal_pairs[(case // 3) % len(horizontal_pairs)])
        family = "horizontal_single_lane"
    elif family_index == 1:
        spine_column = environment.layout.charger_position[1]
        spine_pairs = tuple(
            ((row, spine_column), (row + 1, spine_column))
            for row in range(environment.layout.rows - 1)
            if environment.layout.is_passable((row, spine_column))
            and environment.layout.is_passable((row + 1, spine_column))
        )
        positions = list(spine_pairs[(case // 3) % len(spine_pairs)])
        family = "vertical_spine"
    else:
        exits = tuple(sorted(environment.layout.robot_exit_positions))
        exit_pairs = tuple(zip(exits, exits[1:], strict=False))
        positions = list(exit_pairs[(case // 3) % len(exit_pairs)])
        family = "three_cell_exit"
    if (case // 36) % 2:
        positions.reverse()

    for agent_index, (agent, position) in enumerate(
        zip(state.agents, positions, strict=True)
    ):
        agent.position = position
        agent.battery = float(20 + 2 * ((case + 7 * agent_index) % 41))
        agent.charge_mode_active = False
        agent.last_action = ACTIONS[(case // 5 + agent_index) % len(ACTIONS)]
        agent.last_executed_action = ACTIONS[
            (case // 7 + 2 * agent_index) % len(ACTIONS)
        ]
        agent.steps_since_charging = case % 31
        agent.carrying_task_id = None
        agent.route_commitment_task_id = None

    # The correction is deliberately gated by public history already present
    # in S_t.  Ordinary first-contact navigation must remain untouched; a
    # mover is supervised only after three or more observed stationary joint
    # transitions prove that the pair is actually stuck.
    state.ineffective_joint_wait_streak = 3 + case % 5

    first_task, second_task = state.tasks
    for task in state.tasks:
        task.status = "available"
        task.carrier_agent_id = None
        task.claimed_frame = None
        task.claimed_battery = None

    cargo_pattern = (case // 9) % 4
    if cargo_pattern in {1, 3}:
        first_task.status = "carried"
        first_task.carrier_agent_id = state.agents[0].agent_id
        first_task.claimed_frame = state.frame
        first_task.claimed_battery = state.agents[0].battery
        state.agents[0].carrying_task_id = first_task.task_id
        state.agents[0].route_commitment_task_id = first_task.task_id
    else:
        state.agents[0].route_commitment_task_id = first_task.task_id
    if cargo_pattern in {2, 3}:
        second_task.status = "carried"
        second_task.carrier_agent_id = state.agents[1].agent_id
        second_task.claimed_frame = state.frame
        second_task.claimed_battery = state.agents[1].battery
        state.agents[1].carrying_task_id = second_task.task_id
        state.agents[1].route_commitment_task_id = second_task.task_id
    else:
        state.agents[1].route_commitment_task_id = second_task.task_id

    environment.set_state(state)
    return family


def collect_deadlock_break_rows(
    policy: MAPPOPolicy,
    *,
    cases: int,
    seed: int,
    require_actor_joint_wait: bool = True,
    horizontal_nondual_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect non-WAIT teacher rows where the Actor otherwise double-waits."""

    rows: list[np.ndarray] = []
    labels: list[int] = []
    family_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    teacher_single_move_states = 0
    actor_joint_wait_states = 0
    excluded_other_families = 0
    excluded_dual_loaded = 0
    for case in range(int(cases)):
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        environment.reset(seed=int(seed) + case)
        family = _configure_structural_case(environment, case=case)
        configured_state = environment.get_state()
        if horizontal_nondual_only and family != "horizontal_single_lane":
            excluded_other_families += 1
            continue
        if horizontal_nondual_only and all(
            agent.carrying_task_id is not None
            for agent in configured_state.agents
        ):
            # Standard head-on scenarios deliberately contain two loaded
            # robots.  Leave that already passing distribution anchored and
            # correct only the one-empty/one-loaded shelf-arm stalls that
            # dominate the rejected formal failures.
            excluded_dual_loaded += 1
            continue
        teacher = stable_coordination_actions(environment)
        moving_agents = tuple(
            agent_id for agent_id, action in teacher.items() if action != "WAIT"
        )
        if len(moving_agents) != 1:
            continue
        teacher_single_move_states += 1
        observations = environment.observations()
        actor_actions, _ = policy.act(
            observations,
            environment.global_state(),
            deterministic=True,
        )
        actor_joint_wait = all(action == "WAIT" for action in actor_actions.values())
        actor_joint_wait_states += int(actor_joint_wait)
        if require_actor_joint_wait and not actor_joint_wait:
            continue
        for agent_id in environment.agent_ids:
            label = teacher[agent_id]
            rows.append(independent_actor_input(observations[agent_id]))
            labels.append(ACTIONS.index(label))
            label_counts[label] += 1
        family_counts[family] += 1
    if not rows:
        raise RuntimeError("No structural double-WAIT clearance rows were found.")
    return (
        np.stack(rows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        {
            "requested_cases": int(cases),
            "rows": len(rows),
            "teacher_single_move_states": teacher_single_move_states,
            "actor_joint_wait_states": actor_joint_wait_states,
            "horizontal_nondual_only": bool(horizontal_nondual_only),
            "excluded_other_families": excluded_other_families,
            "excluded_dual_loaded": excluded_dual_loaded,
            "family_counts": dict(sorted(family_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "teacher_actions_submitted_to_environment": 0,
        },
    )


def _fit_with_anchor_kl(
    policy: MAPPOPolicy,
    target_rows: np.ndarray,
    target_labels: np.ndarray,
    anchor_rows: np.ndarray,
    config: DeadlockBreakConfig,
) -> dict[str, float]:
    targets = torch.as_tensor(target_rows, dtype=torch.float32, device=policy.device)
    labels = torch.as_tensor(target_labels, dtype=torch.long, device=policy.device)
    anchors = torch.as_tensor(anchor_rows, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        source_anchor_probabilities = torch.softmax(
            policy.masked_actor_logits(anchors), dim=-1
        )

    def metrics() -> tuple[float, float, float]:
        with torch.no_grad():
            probabilities = torch.softmax(policy.masked_actor_logits(targets), dim=-1)
        row_indices = torch.arange(len(labels), device=policy.device)
        selected = probabilities[row_indices, labels]
        accuracy = (probabilities.argmax(dim=-1) == labels).float().mean()
        return (
            float(selected.mean().cpu()),
            float(selected.min().cpu()),
            float(accuracy.cpu()),
        )

    probability_mean_before, probability_min_before, accuracy_before = metrics()
    parameters = tuple(policy.network.deadlock_escape_action_head.parameters())
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
            selected_labels = labels[target_index]
            logits = policy.masked_actor_logits(targets[target_index])
            cross_entropy = F.cross_entropy(logits, selected_labels)
            row_indices = torch.arange(len(selected_labels), device=policy.device)
            target_logits = logits[row_indices, selected_labels]
            alternatives = logits.clone()
            alternatives[row_indices, selected_labels] = -torch.inf
            strongest_alternative = alternatives.max(dim=-1).values
            margin = torch.relu(
                float(config.move_margin)
                - (target_logits - strongest_alternative)
            ).mean()
            anchor_logits = policy.masked_actor_logits(anchors[anchor_index])
            anchor_kl = F.kl_div(
                F.log_softmax(anchor_logits, dim=-1),
                source_anchor_probabilities[anchor_index],
                reduction="batchmean",
            )
            loss = (
                cross_entropy
                + float(config.move_margin_weight) * margin
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
    probability_mean_after, probability_min_after, accuracy_after = metrics()
    with torch.no_grad():
        after_anchor = torch.softmax(policy.masked_actor_logits(anchors), dim=-1)
    anchor_probability_l1 = float(
        torch.abs(after_anchor - source_anchor_probabilities).sum(dim=-1).mean().cpu()
    )
    return {
        "target_move_probability_mean_before": probability_mean_before,
        "target_move_probability_min_before": probability_min_before,
        "target_accuracy_before": accuracy_before,
        "target_move_probability_mean_after": probability_mean_after,
        "target_move_probability_min_after": probability_min_after,
        "target_accuracy_after": accuracy_after,
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
    parser.add_argument("--cases", type=int, default=8_192)
    parser.add_argument("--anchor-rows", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--move-margin", type=float, default=3.0)
    parser.add_argument("--move-margin-weight", type=float, default=2.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20_000_004)
    args = parser.parse_args()
    config = DeadlockBreakConfig(
        cases=args.cases,
        anchor_rows=args.anchor_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        move_margin=args.move_margin,
        move_margin_weight=args.move_margin_weight,
        anchor_kl_weight=args.anchor_kl_weight,
        seed=args.seed,
    )
    if min(config.cases, config.anchor_rows, config.epochs, config.batch_size) <= 0:
        parser.error("cases, anchor rows, epochs, and batch size must be positive")

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    rows, labels, collection = collect_deadlock_break_rows(
        policy,
        cases=config.cases,
        seed=config.seed,
    )
    anchor = np.load(args.anchor_dataset, allow_pickle=False)
    all_anchor_rows = np.asarray(anchor["rows"], dtype=np.float32)
    rng = np.random.default_rng(config.seed + 2)
    count = min(config.anchor_rows, len(all_anchor_rows))
    indices = rng.choice(len(all_anchor_rows), size=count, replace=False)
    fit = _fit_with_anchor_kl(
        policy,
        rows,
        labels,
        all_anchor_rows[indices],
        config,
    )

    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    rejected_report_path = (
        args.training_summary.parent
        / "rejected_formal_evaluation_seed_18000004.json"
    )
    if not rejected_report_path.is_file():
        raise FileNotFoundError("The rejected 18M formal report must be preserved.")
    rejected_record = next(
        (
            item
            for item in summary.get("rejected_formal_evaluations", [])
            if item.get("name") == "rejected_formal_evaluation_18000004"
        ),
        None,
    )
    if rejected_record is None:
        raise ValueError("The rejected 18M formal report is absent from the ledger.")
    if rejected_record.get("sha256") != file_sha256(rejected_report_path):
        raise ValueError("The rejected 18M report hash does not match the ledger.")
    ledger = reserve_evaluation_seed_span(
        summary.get("seed_ledger", {}),
        {"deadlock_break_refit": [config.seed, config.cases]},
        name=f"targeted_deadlock_break_refit_{config.seed}",
    )
    rejected_development: dict[str, Any] | None = None
    if config.seed != 20_000_004:
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_deadlock_break_refit": [20_000_004, 8_192]},
            name="rejected_deadlock_break_refit_20000004",
        )
        ledger = reserve_evaluation_seed_span(
            ledger,
            {
                "ai_ai": [20_300_004, 100],
                "coordinated": [20_320_004, 100],
                "goal_directed": [20_340_004, 100],
                "cautious": [20_360_004, 100],
                "hesitant": [20_380_004, 100],
                "random": [20_400_004, 100],
                "head_on": [20_450_004, 100],
                "empty_delivery_clearance": [20_460_004, 100],
                "dual_charger_approach": [20_470_004, 100],
                "outer_exit_charger_approach": [20_480_004, 100],
                "occupied_charger_handoff": [20_490_004, 100],
            },
            name="rejected_deadlock_break_development_20300004",
        )
        rejected_development_path = (
            args.checkpoint.parent.parent
            / "v64_deadlock_break1"
            / "development_100x400.json"
        )
        if not rejected_development_path.is_file():
            raise FileNotFoundError(
                "The rejected first deadlock-break development report is missing."
            )
        rejected_development = {
            "path": str(rejected_development_path.resolve()),
            "sha256": file_sha256(rejected_development_path),
            "used_for_weight_updates": False,
            "failed_checks": ["avoidable_loaded_delivery_detours_eq_0"],
        }
    rejected_second_development: dict[str, Any] | None = None
    if config.seed not in {20_000_004, 20_100_004}:
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_deadlock_break_refit": [20_100_004, 8_192]},
            name="rejected_deadlock_break_refit_20100004",
        )
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"deadlock_break_screen": [20_600_004, 100]},
            name="rejected_deadlock_break_screen_20600004",
        )
        ledger = reserve_evaluation_seed_span(
            ledger,
            {
                "ai_ai": [20_700_004, 100],
                "coordinated": [20_720_004, 100],
                "goal_directed": [20_740_004, 100],
                "cautious": [20_760_004, 100],
                "hesitant": [20_780_004, 100],
                "random": [20_800_004, 100],
                "head_on": [20_850_004, 100],
                "empty_delivery_clearance": [20_860_004, 100],
                "dual_charger_approach": [20_870_004, 100],
                "outer_exit_charger_approach": [20_880_004, 100],
                "occupied_charger_handoff": [20_890_004, 100],
            },
            name="rejected_deadlock_break_development_20700004",
        )
        second_development_path = (
            args.checkpoint.parent.parent
            / "v64_deadlock_break2"
            / "development_100x400.json"
        )
        if not second_development_path.is_file():
            raise FileNotFoundError(
                "The rejected second deadlock-break development report is missing."
            )
        rejected_second_development = {
            "path": str(second_development_path.resolve()),
            "sha256": file_sha256(second_development_path),
            "used_for_weight_updates": False,
            "failed_checks": ["head_on_yield_success_ge_0_90"],
        }
    rejected_third_screen: dict[str, Any] | None = None
    if config.seed not in {20_000_004, 20_100_004, 20_200_004}:
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_deadlock_break_refit": [20_200_004, 24_576]},
            name="rejected_deadlock_break_refit_20200004",
        )
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"deadlock_break_ai_screen": [20_900_004, 100]},
            name="rejected_deadlock_break_ai_screen_20900004",
        )
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"deadlock_break_head_on_screen": [20_950_004, 200]},
            name="rejected_deadlock_break_head_on_screen_20950004",
        )
        rejected_third_screen = {
            "used_for_weight_updates": False,
            "failed_checks": [
                "head_on_yield_success_ge_0_90",
                "avoidable_loaded_delivery_detours_eq_0",
            ],
            "ai_ai_deadlock_episode_rate": 0.01,
            "head_on_yield_success_rate": 0.745,
        }
    rejected_zero_gate_refit: dict[str, Any] | None = None
    if config.seed not in {
        20_000_004,
        20_100_004,
        20_200_004,
        21_000_004,
    }:
        ledger = reserve_evaluation_seed_span(
            ledger,
            {"rejected_zero_gate_refit": [21_000_004, 24_576]},
            name="rejected_zero_gate_refit_21000004",
        )
        rejected_zero_gate_refit = {
            "used_for_weight_updates": False,
            "reason": "the initial topology gate was identically zero on the target rows",
            "target_accuracy_before": 0.5,
            "target_accuracy_after": 0.5,
            "anchor_probability_mean_l1_change": 0.0,
        }
    report = {
        "config": asdict(config),
        "collection": collection,
        "fit": fit,
        "execution_contract": "offline_actor_weight_update_only",
        "teacher_actions_submitted_to_environment": 0,
        "used_formal_observations_for_weight_updates": False,
        "used_development_observations_for_weight_updates": False,
        "rejected_formal_diagnosis": {
            "path": str(rejected_report_path.resolve()),
            "sha256": file_sha256(rejected_report_path),
            "failed_checks": rejected_record.get("failed_checks", []),
            "used_for_weight_updates": False,
        },
        "anchor_dataset": str(args.anchor_dataset.resolve()),
        "anchor_dataset_sha256": file_sha256(args.anchor_dataset),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "rejected_prior_development": rejected_development,
        "rejected_second_development": rejected_second_development,
        "rejected_third_screen": rejected_third_screen,
        "rejected_zero_gate_refit": rejected_zero_gate_refit,
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
            "superseded_formal_evaluations": deepcopy(
                summary.get("superseded_formal_evaluations", [])
            ),
            "deadlock_break_refit": report,
        },
    )
    np.savez_compressed(
        args.output.with_suffix(".dataset.npz"),
        rows=rows,
        labels=labels,
    )
    args.output.with_suffix(".deadlock_break_refit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
