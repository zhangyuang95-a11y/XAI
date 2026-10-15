"""Train a shared raw-state Actor by offline imitation then two-NN self-play.

Offline teacher labels initialize weights only.  Every formal PPO transition
uses both Actor outputs unchanged; no participant surrogate enters rollouts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.domain import collaborative_study_config, participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import (
    AGENTS, MODEL_VERSION, OBSERVATION_DIM, OBSERVATION_VERSION,
    action_masks, observations, FrozenPureNN,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ACTOR = ROOT / "output/deployment/warehouse_pure_nn_v1_actor.npz"


class PureNN(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * OBSERVATION_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def outputs(self, features: torch.Tensor, masks: torch.Tensor):
        logits = self.actor(features)
        return logits.masked_fill(masks <= 0.5, -1.0e9)


def teacher_rows(episodes: int, seed: int):
    rows, masks, labels = [], [], []
    for episode in range(episodes):
        env = WarehouseMultiAgentEnv(collaborative_study_config())
        env.reset(seed=seed + episode)
        while not (env.state.terminated or env.state.truncated):
            obs = observations(env.state)
            mask = action_masks(env.state)
            suggested = stable_coordination_actions(env)
            actions = {}
            for row, agent_id in enumerate(AGENTS):
                action = suggested[agent_id]
                if mask[row, ACTIONS.index(action)] <= 0.5:
                    raise RuntimeError("offline teacher proposed a physically illegal action")
                rows.append(obs[row])
                masks.append(mask[row])
                labels.append(ACTIONS.index(action))
                actions[agent_id] = action
            env.step(actions)
    return (torch.as_tensor(np.asarray(rows)),
            torch.as_tensor(np.asarray(masks)),
            torch.as_tensor(labels, dtype=torch.long))


def pretrain(model: PureNN, *, episodes: int, epochs: int, seed: int):
    features, masks, labels = teacher_rows(episodes, seed + 10000)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(labels), generator=generator).split(512):
            loss = nn.functional.cross_entropy(model.outputs(features[indices], masks[indices]), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        accuracy = float((model.outputs(features, masks).argmax(-1) == labels).float().mean())
    return {"rows": len(labels), "epochs": epochs, "training_label_accuracy": accuracy}, (features, masks, labels)


def collect_episode(model: PureNN, *, seed: int, generator: torch.Generator):
    env = WarehouseMultiAgentEnv(collaborative_study_config())
    env.reset(seed=seed)
    records = []
    decisions = 0
    while not (env.state.terminated or env.state.truncated):
        obs = observations(env.state)
        mask = action_masks(env.state)
        teacher = stable_coordination_actions(env)
        teacher_indices = np.asarray([ACTIONS.index(teacher[agent_id]) for agent_id in AGENTS], dtype=np.int64)
        collision_matrix = np.asarray([
            [float(env._resolve_motion(env.state, {"robot_1": first, "robot_2": second})[3])
             for second in ACTIONS]
            for first in ACTIONS
        ], dtype=np.float32)
        with torch.no_grad():
            t_obs = torch.from_numpy(obs)
            t_mask = torch.from_numpy(mask)
            logits = model.outputs(t_obs, t_mask)
            probabilities = torch.softmax(logits, dim=-1)
            selected = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            logp = torch.log(probabilities.gather(-1, selected[:, None]).squeeze(-1).clamp_min(1e-8))
            value = float(model.critic(t_obs.flatten()[None]).item())
        actions = {agent_id: ACTIONS[int(selected[row])] for row, agent_id in enumerate(AGENTS)}
        _, rewards, terminated, truncated, info = env.step(actions)
        if info["proposed_actions"] != actions:
            raise RuntimeError("formal Warehouse self-play action changed before physics")
        records.append((obs, mask, selected.numpy(), logp.numpy(), value,
                        teacher_indices, collision_matrix,
                        float(np.mean(list(rewards.values())))))
        decisions += 2
        if terminated or truncated:
            break
    return records, {"seed": seed, "steps": len(records), "actor_decisions": decisions,
                     "proxy_human_overrides": 0, "post_policy_interventions": 0,
                     "deliveries": env.state.total_deliveries,
                     "collisions": env.state.robot_collision_events,
                     "shutdowns": env.state.shutdown_count,
                     "team_reward": sum(item[-1] for item in records)}


def ppo_update(model: PureNN, optimizer, episodes, teacher_data, *, gamma: float = 0.99,
               gae_lambda: float = 0.95, epochs: int = 4):
    chunks = []
    for records in episodes:
        advantage = 0.0
        next_value = 0.0
        enriched = []
        for obs, mask, action, logp, value, teacher_actions, collision_matrix, reward in reversed(records):
            delta = reward + gamma * next_value - value
            advantage = delta + gamma * gae_lambda * advantage
            enriched.append((obs, mask, action, logp, value, advantage, advantage + value,
                             teacher_actions, collision_matrix))
            next_value = value
        chunks.extend(reversed(enriched))
    if not chunks:
        return 0.0
    obs = torch.from_numpy(np.stack([item[0] for item in chunks]))
    masks = torch.from_numpy(np.stack([item[1] for item in chunks]))
    actions = torch.as_tensor(np.stack([item[2] for item in chunks]), dtype=torch.long)
    rollout_teacher = torch.as_tensor(np.stack([item[7] for item in chunks]), dtype=torch.long)
    collision_matrices = torch.from_numpy(np.stack([item[8] for item in chunks]))
    old_logp = torch.from_numpy(np.stack([item[3] for item in chunks]))
    advantages = torch.as_tensor([item[5] for item in chunks], dtype=torch.float32)
    targets = torch.as_tensor([item[6] for item in chunks], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    total_loss = 0.0
    teacher_obs, teacher_masks, teacher_labels = teacher_data
    for _ in range(epochs):
        logits = model.outputs(obs, masks)
        policy = Categorical(logits=logits)
        logp = policy.log_prob(actions)
        ratio = torch.exp(logp - old_logp)
        advantage = advantages[:, None]
        actor_loss = -torch.minimum(ratio * advantage,
                                    ratio.clamp(0.8, 1.2) * advantage).mean()
        values = model.critic(obs.flatten(start_dim=1)).squeeze(-1)
        critic_loss = nn.functional.mse_loss(values, targets)
        teacher_indices = torch.randint(len(teacher_labels), (512,))
        imitation_loss = nn.functional.cross_entropy(
            model.outputs(teacher_obs[teacher_indices], teacher_masks[teacher_indices]),
            teacher_labels[teacher_indices])
        rollout_imitation = nn.functional.cross_entropy(logits.flatten(end_dim=1), rollout_teacher.flatten())
        chance = policy.probs
        collision_risk = (chance[:, 0, :, None] * chance[:, 1, None, :] * collision_matrices).sum((-1, -2)).mean()
        loss = (actor_loss + 0.5 * critic_loss - 0.005 * policy.entropy().mean()
                + 0.25 * imitation_loss + 0.75 * rollout_imitation
                + 3.0 * collision_risk)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / epochs


def export_actor(model: PureNN, path: Path, *, training_seed: int,
                 training_episodes: int, checkpoint_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = (model.actor[0], model.actor[2], model.actor[4])
    tensors = {}
    for index, layer in enumerate(layers):
        tensors[f"w{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        tensors[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    metadata = {"model_version": MODEL_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "observation_dim": OBSERVATION_DIM,
                "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
                "action_names": list(ACTIONS),
                "training_seed": training_seed,
                "training_episodes": training_episodes,
                "checkpoint_sha256": sha256(checkpoint_path.read_bytes()).hexdigest()}
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **tensors)
    frozen = FrozenPureNN.load(path)
    sample = np.random.default_rng(9).normal(size=(5, OBSERVATION_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = model.actor(torch.from_numpy(sample)).numpy()
    if not np.allclose(frozen.logits(sample), expected, atol=1e-5):
        raise RuntimeError("PyTorch/NumPy pure NN logits differ")
    return {"path": str(path), "sha256": frozen.artifact_sha256,
            "maximum_logit_difference": float(np.max(np.abs(frozen.logits(sample) - expected)))}


def evaluate_model(model: PureNN, *, seeds):
    layers = (model.actor[0], model.actor[2], model.actor[4])
    weights = {}
    for index, layer in enumerate(layers):
        weights[f"w{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    actor = FrozenPureNN(weights, "development-only")
    rows = []
    for seed in seeds:
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=int(seed))
        while not (env.state.terminated or env.state.truncated):
            actions, _ = actor.act(env.state, base_seed=int(seed))
            env.step(actions)
        rows.append((env.state.total_deliveries,
                     env.state.robot_collision_events, env.state.shutdown_count))
    summary = {"mean_deliveries": float(np.mean([row[0] for row in rows])),
               "mean_collisions": float(np.mean([row[1] for row in rows])),
               "mean_shutdowns": float(np.mean([row[2] for row in rows]))}
    quality = summary["mean_deliveries"] - 0.25 * summary["mean_collisions"] - 0.5 * summary["mean_shutdowns"]
    return quality, summary


def train(*, episodes: int, bc_episodes: int, bc_epochs: int, seed: int,
          output: Path, checkpoint: Path, report: Path,
          init_checkpoint: Path | None = None):
    if min(episodes, bc_episodes, bc_epochs) <= 0:
        raise ValueError("training episode and epoch counts must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if init_checkpoint is None:
        model = PureNN()
        pretraining, teacher_data = pretrain(model, episodes=bc_episodes, epochs=bc_epochs, seed=seed)
    else:
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("model_version") != MODEL_VERSION or
            payload.get("observation_version") != OBSERVATION_VERSION or
            payload.get("selected_episode") != 0):
            raise ValueError("initial checkpoint must be the fresh pure-NN imitation model")
        hidden = payload["state_dict"]["actor.0.bias"].numel()
        model = PureNN(hidden=hidden)
        model.load_state_dict(payload["state_dict"])
        teacher_data = teacher_rows(bc_episodes, seed + 10000)
        pretraining = {"initial_checkpoint_sha256": sha256(init_checkpoint.read_bytes()).hexdigest(),
                       "source": "fresh_raw_state_imitation", "teacher_rows": len(teacher_data[2])}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    development = []
    best_state = None
    best_quality = float("-inf")
    best_episode = 0
    def check_candidate(count):
        nonlocal best_state, best_quality, best_episode
        quality, sample = evaluate_model(model, seeds=range(100, 108))
        development.append({"episode": count, **sample, "quality": quality})
        if count > 0 and quality > best_quality:
            best_quality, best_episode = quality, count
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    check_candidate(0)
    for start in range(0, episodes, 8):
        batches = []
        for index in range(start, min(episodes, start + 8)):
            rows, summary = collect_episode(model, seed=seed + index, generator=generator)
            batches.append(rows)
            history.append(summary)
        ppo_update(model, optimizer, batches, teacher_data)
        if len(history) % 40 == 0 or len(history) == episodes:
            check_candidate(len(history))
            print(json.dumps({"episodes": len(history),
                              "mean_deliveries_last_40": float(np.mean([row["deliveries"] for row in history[-40:]])),
                              "development": development[-1]}), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_version": MODEL_VERSION, "observation_version": OBSERVATION_VERSION,
                "training_seed": seed, "training_episodes": episodes,
                "selected_episode": best_episode, "development_quality": best_quality,
                "proxy_human_overrides": 0, "post_policy_interventions": 0,
                "actor_decisions": sum(row["actor_decisions"] for row in history),
                "state_dict": model.state_dict()}, checkpoint)
    artifact = export_actor(model, output, training_seed=seed,
                            training_episodes=episodes, checkpoint_path=checkpoint)
    results = {"model_version": MODEL_VERSION, "pretraining": pretraining,
               "formal_training_episodes": episodes,
               "actor_decisions": sum(row["actor_decisions"] for row in history),
               "selected_episode": best_episode, "development": development,
               "proxy_human_overrides": 0, "post_policy_interventions": 0,
               "mean_training_deliveries_last_40": float(np.mean([row["deliveries"] for row in history[-40:]])),
               "artifact": artifact, "episodes": history}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--bc-episodes", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "output/deployment/warehouse_pure_nn_v1.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "output/deployment/warehouse_pure_nn_v1_training.json")
    args = parser.parse_args()
    result = train(episodes=args.episodes, bc_episodes=args.bc_episodes,
                   bc_epochs=args.bc_epochs, seed=args.seed, output=args.output,
                   checkpoint=args.checkpoint, report=args.report,
                   init_checkpoint=args.init_checkpoint)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
