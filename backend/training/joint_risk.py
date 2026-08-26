"""Post-MAPPO collision-risk calibration for two independent Actors.

Both distributions are evaluated from the same frozen pre-move state.  The
loss never receives either robot's sampled current action: it minimizes the
expected collision probability ``p1.T @ C(S_t) @ p2`` while a KL anchor keeps
the already learned delivery and charging behaviour stable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.policy import MAPPOPolicy


@dataclass(frozen=True)
class JointRiskFineTuneConfig:
    rows: int = 18_000
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 5e-6
    joint_expected_collision_weight: float = 8.0
    ordinary_kl_weight: float = 1.0
    teacher_anchor_weight: float = 0.10
    avoidable_joint_wait_weight: float = 1.0
    optimization_temperature: float = 1.5
    seed: int = 740_000


@dataclass(frozen=True)
class JointRiskDataset:
    robot_one_observations: np.ndarray
    robot_two_observations: np.ndarray
    collision_matrices: np.ndarray
    teacher_labels: np.ndarray
    collection: dict[str, Any]


def _collision_matrix(policy: MAPPOPolicy, observation: np.ndarray) -> np.ndarray:
    start = policy.network.joint_collision_matrix_start
    action_dim = len(ACTIONS)
    return np.asarray(
        observation[start : start + action_dim**2],
        dtype=np.float32,
    ).reshape(action_dim, action_dim)


def collect_joint_risk_dataset(
    policy: MAPPOPolicy,
    config: JointRiskFineTuneConfig,
) -> JointRiskDataset:
    """Collect paired observations before one direct Actor joint step."""

    environment_config = policy.environment_config
    expected_config = collaborative_study_config(
        horizon=environment_config.horizon,
        seed=environment_config.seed,
    )
    if environment_config != expected_config:
        raise ValueError("The policy does not use the current formal study map.")

    robot_one_rows: list[np.ndarray] = []
    robot_two_rows: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    teacher_rows: list[tuple[int, int]] = []
    episode = 0
    collision_opportunity_rows = 0
    sampled_collision_steps = 0
    shutdown_episodes = 0
    while len(robot_one_rows) < int(config.rows):
        seed = int(config.seed) + episode
        environment = WarehouseMultiAgentEnv(environment_config)
        observations, _ = environment.reset(seed=seed)
        rollout_policy = policy.fork_for_inference(seed=seed + 17_000_000)
        while len(robot_one_rows) < int(config.rows):
            # Freeze both independent observations and the offline teacher
            # target before sampling either current action.
            first = np.asarray(observations["robot_1"], dtype=np.float32).copy()
            second = np.asarray(observations["robot_2"], dtype=np.float32).copy()
            matrix = _collision_matrix(policy, first)
            teacher = stable_coordination_actions(environment)
            robot_one_rows.append(first)
            robot_two_rows.append(second)
            matrices.append(matrix)
            teacher_rows.append(
                (ACTIONS.index(teacher["robot_1"]), ACTIONS.index(teacher["robot_2"]))
            )
            collision_opportunity_rows += int(bool(np.any(matrix > 0.5)))

            actions, _ = rollout_policy.act(
                observations,
                environment.global_state(),
                deterministic=False,
            )
            observations, _, terminated, truncated, info = environment.step(actions)
            sampled_collision_steps += int(
                bool(info.get("robot_collision_event", False))
            )
            if terminated or truncated:
                shutdown_episodes += int(
                    environment.get_state().terminal_reason == "battery_shutdown"
                )
                break
        episode += 1

    first_array = np.stack(robot_one_rows).astype(np.float32, copy=False)
    second_array = np.stack(robot_two_rows).astype(np.float32, copy=False)
    matrix_array = np.stack(matrices).astype(np.float32, copy=False)
    labels = np.asarray(teacher_rows, dtype=np.int64)
    return JointRiskDataset(
        robot_one_observations=first_array,
        robot_two_observations=second_array,
        collision_matrices=matrix_array,
        teacher_labels=labels,
        collection={
            "rows": len(first_array),
            "episodes": episode,
            "collision_opportunity_rows": collision_opportunity_rows,
            "sampled_collision_steps": sampled_collision_steps,
            "shutdown_episodes": shutdown_episodes,
            "action_source": "independent_actor_from_same_pre_move_state",
            "teacher_actions_submitted_to_environment": 0,
        },
    )


def _probabilities(
    policy: MAPPOPolicy,
    observations: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = policy.masked_actor_logits(observations)
    return logits, torch.softmax(logits / float(temperature), dim=-1)


def fine_tune_joint_risk(
    policy: MAPPOPolicy,
    dataset: JointRiskDataset,
    config: JointRiskFineTuneConfig,
) -> dict[str, Any]:
    """Optimize expected joint risk with KL and safe-teacher anchors."""

    first = torch.as_tensor(
        dataset.robot_one_observations,
        dtype=torch.float32,
        device=policy.device,
    )
    second = torch.as_tensor(
        dataset.robot_two_observations,
        dtype=torch.float32,
        device=policy.device,
    )
    matrices = torch.as_tensor(
        dataset.collision_matrices,
        dtype=torch.float32,
        device=policy.device,
    )
    labels = torch.as_tensor(
        dataset.teacher_labels,
        dtype=torch.long,
        device=policy.device,
    )
    with torch.no_grad():
        base_first_logits = policy.masked_actor_logits(first)
        base_second_logits = policy.masked_actor_logits(second)
        base_first = torch.softmax(base_first_logits, dim=-1)
        base_second = torch.softmax(base_second_logits, dim=-1)

    parameters = tuple(policy.network.actor_parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    rng = np.random.default_rng(int(config.seed) + 1)
    history: list[dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        order = rng.permutation(len(first))
        totals = {
            "loss": 0.0,
            "joint_expected_collision": 0.0,
            "ordinary_kl": 0.0,
            "teacher_anchor": 0.0,
            "avoidable_joint_wait": 0.0,
        }
        batches = 0
        policy.network.train()
        for start in range(0, len(order), int(config.batch_size)):
            indices = torch.as_tensor(
                order[start : start + int(config.batch_size)],
                dtype=torch.long,
                device=policy.device,
            )
            first_logits, first_probabilities = _probabilities(
                policy,
                first[indices],
                temperature=config.optimization_temperature,
            )
            second_logits, second_probabilities = _probabilities(
                policy,
                second[indices],
                temperature=config.optimization_temperature,
            )
            expected_collision = torch.einsum(
                "bi,bij,bj->b",
                first_probabilities,
                matrices[indices],
                second_probabilities,
            ).mean()
            first_log = F.log_softmax(first_logits, dim=-1)
            second_log = F.log_softmax(second_logits, dim=-1)
            ordinary_kl = 0.5 * (
                F.kl_div(first_log, base_first[indices], reduction="batchmean")
                + F.kl_div(second_log, base_second[indices], reduction="batchmean")
            )
            teacher_anchor = 0.5 * (
                F.cross_entropy(first_logits, labels[indices, 0])
                + F.cross_entropy(second_logits, labels[indices, 1])
            )
            wait_index = ACTIONS.index("WAIT")
            teacher_requires_progress = ~(
                (labels[indices, 0] == wait_index)
                & (labels[indices, 1] == wait_index)
            )
            joint_wait_probabilities = (
                first_probabilities[:, wait_index]
                * second_probabilities[:, wait_index]
            )
            avoidable_joint_wait = (
                joint_wait_probabilities[teacher_requires_progress].mean()
                if bool(teacher_requires_progress.any())
                else torch.zeros((), dtype=first.dtype, device=policy.device)
            )
            loss = (
                float(config.joint_expected_collision_weight) * expected_collision
                + float(config.ordinary_kl_weight) * ordinary_kl
                + float(config.teacher_anchor_weight) * teacher_anchor
                + float(config.avoidable_joint_wait_weight) * avoidable_joint_wait
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            batches += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["joint_expected_collision"] += float(
                expected_collision.detach().cpu()
            )
            totals["ordinary_kl"] += float(ordinary_kl.detach().cpu())
            totals["teacher_anchor"] += float(teacher_anchor.detach().cpu())
            totals["avoidable_joint_wait"] += float(
                avoidable_joint_wait.detach().cpu()
            )
        row = {key: value / max(1, batches) for key, value in totals.items()}
        row["epoch"] = float(epoch + 1)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    policy.network.eval()
    return {
        "config": asdict(config),
        "collection": dict(dataset.collection),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=18_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=740_000)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0.10)
    parser.add_argument("--avoidable-joint-wait-weight", type=float, default=1.0)
    args = parser.parse_args()

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    configuration = JointRiskFineTuneConfig(
        rows=int(args.rows),
        epochs=int(args.epochs),
        seed=int(args.seed),
        teacher_anchor_weight=float(args.teacher_anchor_weight),
        avoidable_joint_wait_weight=float(args.avoidable_joint_wait_weight),
    )
    dataset = collect_joint_risk_dataset(policy, configuration)
    report = fine_tune_joint_risk(policy, dataset, configuration)
    source_metadata = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "joint_risk_finetuning": report,
        },
    )
    report_path = Path(args.output).with_suffix(".joint_risk.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
