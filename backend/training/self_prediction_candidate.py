"""Distil simultaneous peer-Actor distributions into the context predictor.

Both target distributions are computed in one batched forward pass from the
same frozen S_t.  Targets are offline supervision only; environment actions
remain the source Actor's independently sampled actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy, independent_actor_input


def _collect(
    policy: MAPPOPolicy,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episodes = 0
    actor_steps = 0
    collision_risk_rows = 0
    while len(rows) < samples:
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=seed + episodes)
        inference = policy.fork_for_inference(seed=seed + episodes + 29_000_000)
        while len(rows) < samples:
            state = environment.get_state()
            batch = np.stack(
                [
                    independent_actor_input(observations[agent_id])
                    for agent_id in environment.agent_ids
                ]
            ).astype(np.float32, copy=False)
            tensor = torch.as_tensor(
                batch,
                dtype=torch.float32,
                device=policy.device,
            )
            with torch.no_grad():
                probabilities = torch.softmax(
                    policy.masked_actor_logits(tensor),
                    dim=-1,
                ).detach().cpu().numpy()
            rows.extend((batch[0], batch[1]))
            targets.extend((probabilities[1], probabilities[0]))
            top_actions = {
                agent_id: policy.action_names[int(np.argmax(probabilities[index]))]
                for index, agent_id in enumerate(environment.agent_ids)
            }
            collision_risk_rows += 2 * int(
                bool(environment._resolve_motion(state, top_actions)[3])
            )
            actions, _ = inference.act(
                observations,
                environment.global_state(),
                deterministic=False,
                decision_key=(state.episode_id, state.frame),
            )
            observations, _, terminated, truncated, _ = environment.step(actions)
            actor_steps += 1
            if terminated or truncated:
                break
        episodes += 1
    return (
        np.stack(rows[:samples]).astype(np.float32, copy=False),
        np.stack(targets[:samples]).astype(np.float32, copy=False),
        {
            "episodes": episodes,
            "actor_steps": actor_steps,
            "collision_risk_rows": collision_risk_rows,
            "post_policy_action_interventions": 0,
            "target_source": "paired_frozen_actor_distributions",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=10_540_000)
    args = parser.parse_args()
    if args.samples <= 0 or args.epochs <= 0:
        parser.error("samples and epochs must be positive")

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    rows, targets, coverage = _collect(
        policy,
        samples=int(args.samples),
        seed=int(args.seed),
    )
    dataset_path = Path(args.output).with_suffix(".dataset.npz")
    np.savez_compressed(dataset_path, rows=rows, target_probabilities=targets)
    observations = torch.as_tensor(rows, dtype=torch.float32, device=policy.device)
    target_tensor = torch.as_tensor(
        targets, dtype=torch.float32, device=policy.device
    )

    def metrics() -> tuple[float, float]:
        with torch.no_grad():
            logits = policy.network.teammate_action_logits(observations)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            kl = torch.sum(
                target_tensor
                * (torch.log(target_tensor.clamp_min(1e-8)) - log_probabilities),
                dim=-1,
            ).mean()
            accuracy = (
                logits.argmax(dim=-1) == target_tensor.argmax(dim=-1)
            ).float().mean()
        return float(kl.cpu()), float(accuracy.cpu())

    kl_before, accuracy_before = metrics()
    optimizer = torch.optim.Adam(
        policy.network.teammate_context_predictor.parameters(),
        lr=float(args.learning_rate),
    )
    rng = np.random.default_rng(int(args.seed) + 71)
    policy.network.train()
    final_loss = 0.0
    for _ in range(int(args.epochs)):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), 512):
            indices = torch.as_tensor(
                order[start : start + 512],
                dtype=torch.long,
                device=policy.device,
            )
            logits = policy.network.teammate_action_logits(observations[indices])
            loss = -torch.sum(
                target_tensor[indices] * torch.log_softmax(logits, dim=-1),
                dim=-1,
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.network.teammate_context_predictor.parameters(), 1.0
            )
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    policy.network.eval()
    kl_after, accuracy_after = metrics()
    report = {
        "execution_contract": "offline_simultaneous_self_prediction_only",
        "source": str(args.checkpoint),
        "dataset": str(dataset_path),
        "samples": int(len(rows)),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "kl_before": kl_before,
        "kl_after": kl_after,
        "argmax_accuracy_before": accuracy_before,
        "argmax_accuracy_after": accuracy_after,
        "final_cross_entropy": final_loss,
        **coverage,
    }
    source_metadata = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "simultaneous_self_prediction": report,
        },
    )
    report_path = Path(args.output).with_suffix(".self_prediction.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(report_path)


if __name__ == "__main__":
    main()
