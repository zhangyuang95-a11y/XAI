"""
train_rl.py -- Train a DQN agent for the Pac-Man XAI maze.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agent import DQNNetwork, HeuristicAgent, MODEL_VERSION, mask_invalid_q_values
from environment import ACTION_NAMES, MazeEnvironment, shielded_action_mask
from training_game_viewer import TrainingGameViewer

WARM_START_EPOCHS = 8
BOOTSTRAP_HOLDOUT_FRACTION = 0.10
EXPERT_GUIDANCE_EPISODES = 600
EXPERT_ROLLOUT_START_PROB = 0.70
BC_LOSS_START_WEIGHT = 0.50
FIXED_TRAINING_SUMMARY = "11x11 maze, 2 monsters, 1500 episodes"


@dataclass
class EpisodeStats:
    reward: float
    steps: int
    won: bool
    truncated: bool
    loss: float


@dataclass
class EvalStats:
    episode: int
    avg_reward: float
    win_rate: float
    avg_steps: float
    truncation_rate: float


@dataclass
class ImitationSample:
    observation: np.ndarray
    action_idx: int
    valid_mask: np.ndarray


@dataclass
class WarmStartStats:
    losses: list[float]
    holdout_accuracy: float | None
    train_samples: int
    holdout_samples: int


@dataclass
class ImitationDataset:
    observations: np.ndarray
    actions: np.ndarray
    valid_masks: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty imitation dataset")
        sample_size = min(batch_size, len(self))
        indices = np.random.choice(len(self), size=sample_size, replace=False)
        return (
            self.observations[indices],
            self.actions[indices],
            self.valid_masks[indices],
        )


class TrainingVisualizer:
    def __init__(
        self,
        plot_path: Path,
        metrics_path: Path,
        show_plot: bool,
        refresh_every: int,
    ):
        self.plot_path = plot_path
        self.metrics_path = metrics_path
        self.show_plot = show_plot
        self.refresh_every = max(1, refresh_every)
        self.csv_fields = [
            "kind",
            "episode",
            "reward",
            "avg_reward",
            "win_rate",
            "avg_steps",
            "loss",
            "epsilon",
            "truncation_rate",
            "train_monsters",
            "reward_preset",
            "encoder",
            "bootstrap_phase",
            "eval_scope",
            "eval_monsters",
        ]
        self.train_episodes: list[int] = []
        self.train_rewards: list[float] = []
        self.train_avg_rewards: list[float] = []
        self.train_win_rates: list[float] = []
        self.train_avg_steps: list[float] = []
        self.train_losses: list[float] = []
        self.epsilons: list[float] = []
        self.eval_stats: list[EvalStats] = []
        self._updates = 0

        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()

        self.enabled = False
        self._plt = None
        self._figure = None
        self._axes = None

        try:
            import matplotlib

            if not self.show_plot:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._plt = plt
            if self.show_plot:
                plt.ion()
            self._figure, self._axes = plt.subplots(2, 2, figsize=(13, 8))
            self._figure.suptitle("Pac-Man DQN Training Dashboard", fontsize=14)
            self.enabled = True
        except Exception as exc:
            print(f"[plot ] visualization disabled: {exc}")

    def _init_csv(self) -> None:
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fields)
            writer.writeheader()

    def record_bootstrap_episode(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        win_rate: float,
        avg_steps: float,
        train_monsters: int,
        reward_preset: str,
        encoder: str,
        bootstrap_phase: str,
    ) -> None:
        self._append_csv(
            {
                "kind": "bootstrap",
                "episode": episode,
                "reward": reward,
                "avg_reward": avg_reward,
                "win_rate": win_rate,
                "avg_steps": avg_steps,
                "loss": "",
                "epsilon": "",
                "truncation_rate": "",
                "train_monsters": train_monsters,
                "reward_preset": reward_preset,
                "encoder": encoder,
                "bootstrap_phase": bootstrap_phase,
                "eval_scope": "",
                "eval_monsters": "",
            }
        )

    def record_train_episode(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        win_rate: float,
        avg_steps: float,
        loss: float,
        epsilon: float,
        train_monsters: int,
        reward_preset: str,
        encoder: str,
        bootstrap_phase: str,
    ) -> None:
        self.train_episodes.append(episode)
        self.train_rewards.append(reward)
        self.train_avg_rewards.append(avg_reward)
        self.train_win_rates.append(win_rate)
        self.train_avg_steps.append(avg_steps)
        self.train_losses.append(loss)
        self.epsilons.append(epsilon)
        self._append_csv(
            {
                "kind": "train",
                "episode": episode,
                "reward": reward,
                "avg_reward": avg_reward,
                "win_rate": win_rate,
                "avg_steps": avg_steps,
                "loss": loss,
                "epsilon": epsilon,
                "truncation_rate": "",
                "train_monsters": train_monsters,
                "reward_preset": reward_preset,
                "encoder": encoder,
                "bootstrap_phase": bootstrap_phase,
                "eval_scope": "",
                "eval_monsters": "",
            }
        )
        if episode % self.refresh_every == 0:
            self.refresh()

    def record_eval(
        self,
        episode: int,
        metrics: dict[str, float],
        reward_preset: str,
        encoder: str,
        eval_scope: str,
        eval_monsters: int,
    ) -> None:
        if eval_scope == "eval_final":
            self.eval_stats.append(
                EvalStats(
                    episode=episode,
                    avg_reward=metrics["avg_reward"],
                    win_rate=metrics["win_rate"],
                    avg_steps=metrics["avg_steps"],
                    truncation_rate=metrics["truncation_rate"],
                )
            )
        self._append_csv(
            {
                "kind": "eval",
                "episode": episode,
                "reward": "",
                "avg_reward": metrics["avg_reward"],
                "win_rate": metrics["win_rate"],
                "avg_steps": metrics["avg_steps"],
                "loss": "",
                "epsilon": "",
                "truncation_rate": metrics["truncation_rate"],
                "train_monsters": "",
                "reward_preset": reward_preset,
                "encoder": encoder,
                "bootstrap_phase": "",
                "eval_scope": eval_scope,
                "eval_monsters": eval_monsters,
            }
        )
        self.refresh(force=True)

    def _append_csv(self, row: dict) -> None:
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fields)
            writer.writerow(row)

    def refresh(self, force: bool = False) -> None:
        self._updates += 1
        if not force and self._updates % self.refresh_every != 0:
            return
        if not self.enabled:
            return

        assert self._axes is not None
        axes = self._axes.flatten()
        for axis in axes:
            axis.clear()
            axis.grid(alpha=0.25)

        eval_episodes = [item.episode for item in self.eval_stats]
        eval_rewards = [item.avg_reward for item in self.eval_stats]
        eval_win_rates = [item.win_rate * 100.0 for item in self.eval_stats]
        eval_steps = [item.avg_steps for item in self.eval_stats]

        axes[0].plot(self.train_episodes, self.train_rewards, color="#60a5fa", alpha=0.35, label="Train reward")
        axes[0].plot(self.train_episodes, self.train_avg_rewards, color="#1d4ed8", linewidth=2, label="Rolling avg")
        if eval_episodes:
            axes[0].plot(eval_episodes, eval_rewards, color="#f59e0b", linewidth=2, marker="o", label="Eval avg")
        axes[0].set_title("Reward")
        axes[0].set_xlabel("Episode")
        axes[0].legend(loc="best")

        axes[1].plot(self.train_episodes, [rate * 100.0 for rate in self.train_win_rates], color="#22c55e", linewidth=2, label="Rolling train win rate")
        if eval_episodes:
            axes[1].plot(eval_episodes, eval_win_rates, color="#ef4444", linewidth=2, marker="o", label="Eval win rate")
        axes[1].set_title("Win Rate (%)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylim(0, 100)
        axes[1].legend(loc="best")

        axes[2].plot(self.train_episodes, self.train_losses, color="#8b5cf6", linewidth=2, label="Loss")
        axes[2].plot(self.train_episodes, self.epsilons, color="#64748b", linewidth=2, label="Epsilon")
        axes[2].set_title("Loss / Exploration")
        axes[2].set_xlabel("Episode")
        axes[2].legend(loc="best")

        axes[3].plot(self.train_episodes, self.train_avg_steps, color="#06b6d4", linewidth=2, label="Rolling avg steps")
        if eval_episodes:
            axes[3].plot(eval_episodes, eval_steps, color="#f97316", linewidth=2, marker="o", label="Eval avg steps")
        axes[3].set_title("Episode Length")
        axes[3].set_xlabel("Episode")
        axes[3].legend(loc="best")

        self._figure.tight_layout()
        self._figure.subplots_adjust(top=0.90)
        self._figure.savefig(self.plot_path, dpi=140)

        if self.show_plot:
            self._figure.canvas.draw_idle()
            self._plt.pause(0.001)

    def close(self) -> None:
        self.refresh(force=True)
        if self.enabled and self.show_plot:
            self._plt.ioff()
            self._plt.show()


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs: deque[np.ndarray] = deque(maxlen=capacity)
        self.actions: deque[int] = deque(maxlen=capacity)
        self.rewards: deque[float] = deque(maxlen=capacity)
        self.next_obs: deque[np.ndarray] = deque(maxlen=capacity)
        self.dones: deque[bool] = deque(maxlen=capacity)
        self.next_masks: deque[np.ndarray] = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.obs.append(obs.astype(np.float32, copy=False))
        self.actions.append(int(action_idx))
        self.rewards.append(float(reward))
        self.next_obs.append(next_obs.astype(np.float32, copy=False))
        self.dones.append(bool(done))
        self.next_masks.append(next_mask.astype(np.bool_, copy=False))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = np.random.choice(len(self.obs), size=batch_size, replace=False)
        obs = np.stack([self.obs[idx] for idx in indices])
        actions = np.array([self.actions[idx] for idx in indices], dtype=np.int64)
        rewards = np.array([self.rewards[idx] for idx in indices], dtype=np.float32)
        next_obs = np.stack([self.next_obs[idx] for idx in indices])
        dones = np.array([self.dones[idx] for idx in indices], dtype=np.float32)
        next_masks = np.stack([self.next_masks[idx] for idx in indices])
        return obs, actions, rewards, next_obs, dones, next_masks

    def __len__(self) -> int:
        return len(self.obs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fixed Pac-Man RL configuration.",
        epilog=f"Fixed configuration: {FIXED_TRAINING_SUMMARY}. No runtime options are exposed.",
    )
    return parser


def build_fixed_args() -> argparse.Namespace:
    return argparse.Namespace(
        episodes=1500,
        grid_size=11,
        num_monsters=2,
        max_steps=0,
        seed=42,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        buffer_size=100000,
        warmup_steps=2000,
        target_update=500,
        train_frequency=4,
        random_episode_fraction=0.0,
        epsilon_start=0.35,
        epsilon_end=0.02,
        epsilon_decay=25000,
        hidden_dims=[256, 128],
        encoder="cnn",
        reward_preset="stable",
        bootstrap_episodes=80,
        bootstrap_noise=0.05,
        curriculum_monsters=[1, 2],
        curriculum_boundaries=[300],
        eval_every=50,
        eval_episodes=8,
        converge_win_rate=0.7,
        converge_truncation_rate=0.1,
        converge_consecutive_evals=2,
        log_every=10,
        save_path="models/dqn_pacman.pt",
        plot_path="artifacts/training_progress.png",
        metrics_path="artifacts/training_metrics.csv",
        plot_refresh_every=5,
        game_refresh_steps=1,
        device="auto",
        show_plot=True,
        show_game=True,
        stop_on_converge=True,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def epsilon_by_step(global_step: int, start: float, end: float, decay: int) -> float:
    if decay <= 0:
        return end
    progress = np.exp(-global_step / decay)
    return float(end + (start - end) * progress)


def normalize_curriculum(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    monsters = [int(item) for item in args.curriculum_monsters]
    boundaries = [int(item) for item in args.curriculum_boundaries]
    if not monsters:
        monsters = [args.num_monsters]
    if len(boundaries) != max(0, len(monsters) - 1):
        raise ValueError(
            f"curriculum-boundaries must have exactly len(curriculum-monsters)-1 items: "
            f"got monsters={monsters}, boundaries={boundaries}"
        )
    if any(boundary <= 0 for boundary in boundaries):
        raise ValueError(f"curriculum-boundaries must be positive, got {boundaries}")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"curriculum-boundaries must be strictly increasing, got {boundaries}")
    monsters[-1] = int(args.num_monsters)
    return monsters, boundaries


def curriculum_monsters_for_episode(
    episode: int,
    monsters: list[int],
    boundaries: list[int],
) -> int:
    for stage_monsters, boundary in zip(monsters, boundaries):
        if episode <= boundary:
            return stage_monsters
    return monsters[-1]


def linear_schedule_by_episode(
    episode: int,
    total_episodes: int,
    start: float,
    end: float,
) -> float:
    if total_episodes <= 1:
        return float(end)
    if episode <= 1:
        return float(start)
    if episode >= total_episodes:
        return float(end)
    progress = (episode - 1) / float(total_episodes - 1)
    return float(start + (end - start) * progress)


def empty_imitation_dataset() -> ImitationDataset:
    return ImitationDataset(
        observations=np.empty((0, 0), dtype=np.float32),
        actions=np.empty((0,), dtype=np.int64),
        valid_masks=np.empty((0, len(ACTION_NAMES)), dtype=np.bool_),
    )


def build_imitation_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    valid_masks: np.ndarray,
) -> ImitationDataset:
    return ImitationDataset(
        observations=observations.astype(np.float32, copy=False),
        actions=actions.astype(np.int64, copy=False),
        valid_masks=valid_masks.astype(np.bool_, copy=False),
    )


def split_imitation_samples(
    samples: list[ImitationSample],
    holdout_fraction: float,
    seed: int,
) -> tuple[ImitationDataset, ImitationDataset]:
    if not samples:
        return empty_imitation_dataset(), empty_imitation_dataset()

    observations = np.stack([item.observation for item in samples]).astype(np.float32, copy=False)
    actions = np.array([item.action_idx for item in samples], dtype=np.int64)
    valid_masks = np.stack([item.valid_mask for item in samples]).astype(np.bool_, copy=False)

    holdout_fraction = min(0.5, max(0.0, holdout_fraction))
    holdout_size = 0
    if len(samples) > 1 and holdout_fraction > 0.0:
        holdout_size = min(len(samples) - 1, max(1, int(round(len(samples) * holdout_fraction))))

    if holdout_size == 0:
        return (
            build_imitation_dataset(observations, actions, valid_masks),
            empty_imitation_dataset(),
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    holdout_indices = indices[:holdout_size]
    train_indices = indices[holdout_size:]
    return (
        build_imitation_dataset(observations[train_indices], actions[train_indices], valid_masks[train_indices]),
        build_imitation_dataset(observations[holdout_indices], actions[holdout_indices], valid_masks[holdout_indices]),
    )


def behavior_cloning_loss(
    policy_net: DQNNetwork,
    observations: np.ndarray,
    actions: np.ndarray,
    valid_masks: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
    valid_masks_tensor = torch.as_tensor(valid_masks, dtype=torch.bool, device=device)
    logits = policy_net(obs_tensor)
    masked_logits = mask_invalid_q_values(logits, valid_masks_tensor)
    return F.cross_entropy(masked_logits, actions_tensor)


def evaluate_imitation_accuracy(
    policy_net: DQNNetwork,
    dataset: ImitationDataset,
    batch_size: int,
    device: torch.device,
) -> float | None:
    if len(dataset) == 0:
        return None

    policy_net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = start + batch_size
            observations = dataset.observations[start:end]
            actions = dataset.actions[start:end]
            valid_masks = dataset.valid_masks[start:end]
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            valid_masks_tensor = torch.as_tensor(valid_masks, dtype=torch.bool, device=device)
            logits = policy_net(obs_tensor)
            predictions = mask_invalid_q_values(logits, valid_masks_tensor).argmax(dim=1).cpu().numpy()
            correct += int(np.sum(predictions == actions))
            total += int(actions.shape[0])
    policy_net.train()
    return correct / max(1, total)


def bootstrap_stage_schedule(monsters: list[int], total_episodes: int) -> list[tuple[int, int]]:
    if total_episodes <= 0:
        return []

    unique_monsters = list(dict.fromkeys(int(monster) for monster in monsters))
    base = total_episodes // len(unique_monsters)
    remainder = total_episodes % len(unique_monsters)
    schedule: list[tuple[int, int]] = []
    for idx, monster_count in enumerate(unique_monsters):
        episode_count = base + (1 if idx < remainder else 0)
        if episode_count > 0:
            schedule.append((monster_count, episode_count))
    return schedule


def reached_convergence(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        metrics["win_rate"] >= args.converge_win_rate
        and metrics["truncation_rate"] <= args.converge_truncation_rate
    )


def select_action(
    policy_net: DQNNetwork,
    observation: np.ndarray,
    valid_mask_np: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: random.Random,
) -> tuple[int, np.ndarray]:
    valid_indices = np.flatnonzero(valid_mask_np)
    if len(valid_indices) == 0:
        valid_indices = np.array([ACTION_NAMES.index("STAY")])

    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(obs_tensor).squeeze(0).cpu().numpy()

    if rng.random() < epsilon:
        action_idx = int(rng.choice(valid_indices.tolist()))
        return action_idx, q_values

    masked_q = np.where(valid_mask_np, q_values, -1e9)
    action_idx = int(masked_q.argmax())
    return action_idx, q_values


def select_heuristic_action(
    heuristic_agent: HeuristicAgent,
    state: dict,
    valid_mask_np: np.ndarray,
    noise: float,
    rng: random.Random,
) -> tuple[int, int]:
    valid_indices = np.flatnonzero(valid_mask_np)
    if len(valid_indices) == 0:
        stay_idx = ACTION_NAMES.index("STAY")
        return stay_idx, stay_idx

    heuristic_action = heuristic_agent.choose_action(state)
    heuristic_idx = ACTION_NAMES.index(heuristic_action)
    if valid_mask_np[heuristic_idx]:
        target_idx = heuristic_idx
    else:
        target_idx = int(valid_indices[0])

    executed_idx = target_idx
    if rng.random() < noise:
        executed_idx = int(rng.choice(valid_indices.tolist()))
    return executed_idx, target_idx


def optimize_step(
    policy_net: DQNNetwork,
    target_net: DQNNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    expert_dataset: ImitationDataset | None = None,
    bc_weight: float = 0.0,
) -> tuple[float, float, float]:
    obs, actions, rewards, next_obs, dones, next_masks = replay_buffer.sample(batch_size)

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
    next_masks_tensor = torch.as_tensor(next_masks, dtype=torch.bool, device=device)

    q_values = policy_net(obs_tensor).gather(1, actions_tensor).squeeze(1)

    with torch.no_grad():
        next_policy_q = policy_net(next_obs_tensor)
        next_policy_q = mask_invalid_q_values(next_policy_q, next_masks_tensor)
        next_actions = next_policy_q.argmax(dim=1, keepdim=True)
        next_target_q = target_net(next_obs_tensor)
        next_target_q = mask_invalid_q_values(next_target_q, next_masks_tensor)
        next_state_values = next_target_q.gather(1, next_actions).squeeze(1)
        targets = rewards_tensor + (1.0 - dones_tensor) * gamma * next_state_values

    rl_loss = F.smooth_l1_loss(q_values, targets)
    total_loss = rl_loss
    bc_loss_value = 0.0
    if expert_dataset is not None and len(expert_dataset) > 0 and bc_weight > 0.0:
        bc_obs, bc_actions, bc_masks = expert_dataset.sample(batch_size)
        bc_loss = behavior_cloning_loss(policy_net, bc_obs, bc_actions, bc_masks, device)
        total_loss = total_loss + bc_weight * bc_loss
        bc_loss_value = float(bc_loss.item())

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(total_loss.item()), float(rl_loss.item()), bc_loss_value


def run_imitation_warm_start(
    policy_net: DQNNetwork,
    optimizer: torch.optim.Optimizer,
    train_dataset: ImitationDataset,
    holdout_dataset: ImitationDataset,
    batch_size: int,
    device: torch.device,
    epochs: int = WARM_START_EPOCHS,
) -> WarmStartStats:
    if len(train_dataset) == 0:
        return WarmStartStats(losses=[], holdout_accuracy=None, train_samples=0, holdout_samples=len(holdout_dataset))

    losses: list[float] = []

    for _ in range(epochs):
        indices = np.random.permutation(len(train_dataset))
        epoch_losses: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            loss = behavior_cloning_loss(
                policy_net,
                train_dataset.observations[batch_indices],
                train_dataset.actions[batch_indices],
                train_dataset.valid_masks[batch_indices],
                device,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    holdout_accuracy = evaluate_imitation_accuracy(policy_net, holdout_dataset, batch_size=batch_size, device=device)
    return WarmStartStats(
        losses=losses,
        holdout_accuracy=holdout_accuracy,
        train_samples=len(train_dataset),
        holdout_samples=len(holdout_dataset),
    )


def collect_bootstrap_experience(
    args: argparse.Namespace,
    replay_buffer: ReplayBuffer,
    visualizer: TrainingVisualizer,
    rng: random.Random,
) -> tuple[list[ImitationSample], int]:
    if args.bootstrap_episodes <= 0:
        return [], 0

    stage_schedule = bootstrap_stage_schedule(args.curriculum_monsters, args.bootstrap_episodes)
    if not stage_schedule:
        return [], 0

    env = MazeEnvironment(
        grid_size=args.grid_size,
        num_monsters=stage_schedule[0][0],
        seed=args.seed + 80_000,
        max_steps=args.max_steps or None,
        reward_preset=args.reward_preset,
    )
    heuristic_agent = HeuristicAgent()
    recent_stats: deque[EpisodeStats] = deque(maxlen=max(1, args.log_every))
    samples: list[ImitationSample] = []
    total_transitions = 0

    print(
        f"[boot ] heuristic bootstrap: episodes={args.bootstrap_episodes} "
        f"noise={args.bootstrap_noise:.1%} stages={stage_schedule}"
    )
    global_episode = 0
    for stage_idx, (stage_monsters, stage_episodes) in enumerate(stage_schedule, start=1):
        env.num_monsters = stage_monsters
        print(
            f"[boot ] stage={stage_idx}/{len(stage_schedule)} "
            f"monsters={stage_monsters} episodes={stage_episodes}"
        )
        for _ in range(stage_episodes):
            global_episode += 1
            observation = env.reset_rl(seed=args.seed + 80_000 + global_episode)
            state = env.get_state()
            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done:
                mask = shielded_action_mask(state)
                action_idx, target_action_idx = select_heuristic_action(
                    heuristic_agent,
                    state,
                    mask,
                    args.bootstrap_noise,
                    rng,
                )
                action = ACTION_NAMES[action_idx]
                samples.append(
                    ImitationSample(
                        observation=observation.astype(np.float32, copy=False),
                        action_idx=target_action_idx,
                        valid_mask=mask.astype(np.bool_, copy=False),
                    )
                )

                next_observation, reward, done, info = env.step_rl(action)
                next_state = info["state"]
                replay_buffer.add(
                    observation,
                    action_idx,
                    reward,
                    next_observation,
                    done,
                    shielded_action_mask(next_state),
                )
                observation = next_observation
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_transitions += 1

            recent_stats.append(
                EpisodeStats(
                    reward=episode_reward,
                    steps=episode_steps,
                    won=state["game_state"].value == "won",
                    truncated=info["truncated"],
                    loss=0.0,
                )
            )
            avg_reward = float(np.mean([item.reward for item in recent_stats]))
            avg_steps = float(np.mean([item.steps for item in recent_stats]))
            win_rate = float(np.mean([1.0 if item.won else 0.0 for item in recent_stats]))
            visualizer.record_bootstrap_episode(
                episode=global_episode,
                reward=episode_reward,
                avg_reward=avg_reward,
                win_rate=win_rate,
                avg_steps=avg_steps,
                train_monsters=stage_monsters,
                reward_preset=args.reward_preset,
                encoder=args.encoder,
                bootstrap_phase="collect",
            )
            if global_episode % args.log_every == 0 or global_episode == 1:
                print(
                    f"[boot ] episode={global_episode:4d}/{args.bootstrap_episodes} "
                    f"reward={episode_reward:7.2f} avg_reward={avg_reward:7.2f} "
                    f"win_rate={win_rate:5.2%} avg_steps={avg_steps:6.1f} "
                    f"train_monsters={stage_monsters}"
                )

    return samples, total_transitions


def top_q_values(q_values: np.ndarray, valid_mask_np: np.ndarray, top_k: int = 4) -> list[tuple[str, float]]:
    masked = np.where(valid_mask_np, q_values, -1e9)
    ranked = np.argsort(masked)[::-1]
    items: list[tuple[str, float]] = []
    for idx in ranked:
        if masked[idx] <= -1e8:
            continue
        items.append((ACTION_NAMES[int(idx)], float(masked[idx])))
        if len(items) >= top_k:
            break
    return items


def evaluate_policy(
    policy_net: DQNNetwork,
    args: argparse.Namespace,
    input_dim: int,
    device: torch.device,
    num_monsters: int | None = None,
    viewer: TrainingGameViewer | None = None,
) -> dict[str, float]:
    del input_dim
    policy_net.eval()
    eval_monsters = args.num_monsters if num_monsters is None else int(num_monsters)
    eval_env = MazeEnvironment(
        grid_size=args.grid_size,
        num_monsters=eval_monsters,
        seed=args.seed + 10_000,
        max_steps=args.max_steps or None,
        reward_preset=args.reward_preset,
    )

    rewards: list[float] = []
    wins = 0
    steps_list: list[int] = []
    truncations = 0

    for episode in range(args.eval_episodes):
        observation = eval_env.reset_rl(seed=args.seed + 10_000 + episode)
        state = eval_env.get_state()
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            mask = shielded_action_mask(state)
            action_idx, q_values = select_action(policy_net, observation, mask, 0.0, device, random.Random(episode))
            action = ACTION_NAMES[action_idx]
            observation, reward, done, info = eval_env.step_rl(action)
            state = info["state"]
            episode_reward += reward
            episode_steps += 1

            if viewer is not None and (episode_steps % max(1, args.game_refresh_steps) == 0 or done):
                viewer.update(
                    state,
                    mode="eval",
                    episode=episode,
                    total_episodes=args.eval_episodes,
                    episode_step=episode_steps,
                    global_step=0,
                    epsilon=0.0,
                    episode_reward=episode_reward,
                    last_reward=reward,
                    action=action,
                    q_values=top_q_values(q_values, mask),
                )

        rewards.append(episode_reward)
        steps_list.append(episode_steps)
        if state["game_state"].value == "won":
            wins += 1
        if info["truncated"]:
            truncations += 1

    policy_net.train()
    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "win_rate": wins / max(1, args.eval_episodes),
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "truncation_rate": truncations / max(1, args.eval_episodes),
    }


def record_evaluation(
    visualizer: TrainingVisualizer,
    episode: int,
    total_episodes: int,
    metrics: dict[str, float],
    reward_preset: str,
    encoder: str,
    eval_scope: str,
    eval_monsters: int,
) -> None:
    print(
        f"[{eval_scope}] episode={episode:4d}/{total_episodes} "
        f"monsters={eval_monsters} avg_reward={metrics['avg_reward']:7.2f} "
        f"win_rate={metrics['win_rate']:5.2%} avg_steps={metrics['avg_steps']:6.1f} "
        f"truncation_rate={metrics['truncation_rate']:5.2%}"
    )
    visualizer.record_eval(
        episode,
        metrics,
        reward_preset=reward_preset,
        encoder=encoder,
        eval_scope=eval_scope,
        eval_monsters=eval_monsters,
    )


def save_checkpoint(
    path: Path,
    policy_net: DQNNetwork,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    global_step: int,
    episode: int,
    best_metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_env = MazeEnvironment(
        grid_size=args.grid_size,
        num_monsters=args.num_monsters,
        seed=args.seed,
        max_steps=args.max_steps or None,
        reward_preset=args.reward_preset,
    )
    checkpoint = {
        "model_state": policy_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_dim": policy_net.input_dim,
        "action_names": ACTION_NAMES,
        "hidden_dims": tuple(args.hidden_dims),
        "encoder_type": policy_net.encoder_type,
        "grid_size": args.grid_size,
        "spatial_channels": getattr(policy_net, "spatial_channels", 8),
        "scalar_dim": getattr(policy_net, "scalar_dim", 7),
        "model_version": MODEL_VERSION,
        "metadata": {
            "algorithm": "DQN",
            "grid_size": args.grid_size,
            "num_monsters": args.num_monsters,
            "max_steps": checkpoint_env.max_steps,
            "seed": args.seed,
            "random_episode_fraction": args.random_episode_fraction,
            "reward_preset": args.reward_preset,
            "encoder_type": policy_net.encoder_type,
            "spatial_channels": getattr(policy_net, "spatial_channels", 8),
            "scalar_dim": getattr(policy_net, "scalar_dim", 7),
            "curriculum_monsters": list(args.curriculum_monsters),
            "curriculum_boundaries": list(args.curriculum_boundaries),
            "bootstrap_episodes": args.bootstrap_episodes,
            "bootstrap_noise": args.bootstrap_noise,
            "warm_start_epochs": WARM_START_EPOCHS,
            "bootstrap_holdout_fraction": BOOTSTRAP_HOLDOUT_FRACTION,
            "expert_guidance_episodes": EXPERT_GUIDANCE_EPISODES,
            "expert_rollout_start_prob": EXPERT_ROLLOUT_START_PROB,
            "bc_loss_start_weight": BC_LOSS_START_WEIGHT,
            "converge_win_rate": args.converge_win_rate,
            "converge_truncation_rate": args.converge_truncation_rate,
            "converge_consecutive_evals": args.converge_consecutive_evals,
            "stop_on_converge": args.stop_on_converge,
            "episode": episode,
            "global_step": global_step,
            "best_metrics": best_metrics or {},
            "reward_config": checkpoint_env.reward_config,
        },
    }
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    curriculum_monsters, curriculum_boundaries = normalize_curriculum(args)
    args.curriculum_monsters = curriculum_monsters
    args.curriculum_boundaries = curriculum_boundaries

    env = MazeEnvironment(
        grid_size=args.grid_size,
        num_monsters=curriculum_monsters[0],
        seed=args.seed,
        max_steps=args.max_steps or None,
        reward_preset=args.reward_preset,
    )
    initial_obs = env.reset_rl(seed=args.seed)
    input_dim = int(initial_obs.shape[0])
    output_dim = len(ACTION_NAMES)
    hidden_dims = tuple(args.hidden_dims)

    policy_net = DQNNetwork(
        input_dim,
        output_dim,
        hidden_dims=hidden_dims,
        encoder_type=args.encoder,
        grid_size=args.grid_size,
    ).to(device)
    target_net = DQNNetwork(
        input_dim,
        output_dim,
        hidden_dims=hidden_dims,
        encoder_type=args.encoder,
        grid_size=args.grid_size,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size)
    save_path = Path(args.save_path)
    visualizer = TrainingVisualizer(
        plot_path=Path(args.plot_path),
        metrics_path=Path(args.metrics_path),
        show_plot=args.show_plot,
        refresh_every=args.plot_refresh_every,
    )
    game_viewer = TrainingGameViewer(
        grid_size=args.grid_size,
        danger_radius=3,
        enabled=args.show_game,
    )
    print(f"[plot ] dashboard -> {visualizer.plot_path}")
    print(f"[plot ] metrics   -> {visualizer.metrics_path}")
    if args.show_plot:
        print("[plot ] live window enabled")
    else:
        print("[plot ] running headless, PNG will still update on disk")
    if args.show_game:
        print("[game ] live game window enabled")
    else:
        print("[game ] live game window disabled")
    print(
        f"[train] encoder={args.encoder} reward_preset={args.reward_preset} "
        f"curriculum={curriculum_monsters} boundaries={curriculum_boundaries}"
    )

    global_step = 0
    policy_step = 0
    optimize_step_count = 0
    best_score = float("-inf")
    best_metrics: dict[str, float] | None = None
    recent_stats: deque[EpisodeStats] = deque(maxlen=max(1, args.log_every))
    expert_agent = HeuristicAgent()
    consecutive_converged_evals = 0
    bootstrap_samples, bootstrap_steps = collect_bootstrap_experience(
        args,
        replay_buffer,
        visualizer,
        rng,
    )
    global_step = bootstrap_steps
    expert_train_dataset, expert_holdout_dataset = split_imitation_samples(
        bootstrap_samples,
        holdout_fraction=BOOTSTRAP_HOLDOUT_FRACTION,
        seed=args.seed,
    )
    warm_start_stats = run_imitation_warm_start(
        policy_net,
        optimizer,
        expert_train_dataset,
        expert_holdout_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    if warm_start_stats.losses:
        print("[boot ] imitation warm start losses=" + ", ".join(f"{loss:.4f}" for loss in warm_start_stats.losses))
        if warm_start_stats.holdout_accuracy is not None:
            print(
                f"[boot ] holdout accuracy={warm_start_stats.holdout_accuracy:5.2%} "
                f"train_samples={warm_start_stats.train_samples} holdout_samples={warm_start_stats.holdout_samples}"
            )
        target_net.load_state_dict(policy_net.state_dict())

        stage_monsters = curriculum_monsters[0]
        stage_metrics = evaluate_policy(
            policy_net,
            args,
            input_dim,
            device,
            num_monsters=stage_monsters,
            viewer=None,
        )
        record_evaluation(
            visualizer,
            episode=0,
            total_episodes=args.episodes,
            metrics=stage_metrics,
            reward_preset=args.reward_preset,
            encoder=args.encoder,
            eval_scope="eval_stage",
            eval_monsters=stage_monsters,
        )
        if stage_monsters == args.num_monsters:
            final_metrics = dict(stage_metrics)
        else:
            final_metrics = evaluate_policy(
                policy_net,
                args,
                input_dim,
                device,
                num_monsters=args.num_monsters,
                viewer=game_viewer if args.show_game else None,
            )
        record_evaluation(
            visualizer,
            episode=0,
            total_episodes=args.episodes,
            metrics=final_metrics,
            reward_preset=args.reward_preset,
            encoder=args.encoder,
            eval_scope="eval_final",
            eval_monsters=args.num_monsters,
        )
        initial_score = final_metrics["win_rate"] * 100.0 + final_metrics["avg_reward"]
        best_score = initial_score
        best_metrics = final_metrics
        save_checkpoint(save_path, policy_net, optimizer, args, global_step, 0, best_metrics)
        consecutive_converged_evals = 1 if reached_convergence(final_metrics, args) else 0
        print(f"[save ] bootstrap checkpoint -> {save_path}")
    random_episode_fraction = min(1.0, max(0.0, args.random_episode_fraction))
    random_episodes = min(
        args.episodes,
        max(1 if args.episodes > 0 and random_episode_fraction > 0 else 0, int(args.episodes * random_episode_fraction)),
    )
    if random_episodes > 0:
        print(
            f"[train] random collection phase: first {random_episodes}/{args.episodes} episodes "
            f"({random_episode_fraction:.0%}) run with purely random actions"
        )
    else:
        print("[train] random collection phase disabled")

    for episode in range(1, args.episodes + 1):
        train_monsters = curriculum_monsters_for_episode(episode, curriculum_monsters, curriculum_boundaries)
        env.num_monsters = train_monsters
        observation = env.reset_rl(seed=args.seed + episode)
        state = env.get_state()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        losses: list[float] = []
        in_random_phase = episode <= random_episodes
        expert_rollout_prob = 0.0 if in_random_phase else linear_schedule_by_episode(
            episode,
            EXPERT_GUIDANCE_EPISODES,
            EXPERT_ROLLOUT_START_PROB,
            0.0,
        )
        bc_weight = 0.0 if in_random_phase else linear_schedule_by_episode(
            episode,
            EXPERT_GUIDANCE_EPISODES,
            BC_LOSS_START_WEIGHT,
            0.0,
        )

        while not done:
            epsilon = 1.0 if in_random_phase else epsilon_by_step(
                policy_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay
            )
            mask = shielded_action_mask(state)
            policy_action_idx, q_values = select_action(
                policy_net,
                observation,
                mask,
                1.0 if in_random_phase else epsilon,
                device,
                rng,
            )
            action_idx = policy_action_idx
            if expert_rollout_prob > 0.0 and rng.random() < expert_rollout_prob:
                action_idx, _ = select_heuristic_action(
                    expert_agent,
                    state,
                    mask,
                    0.0,
                    rng,
                )
            action = ACTION_NAMES[action_idx]

            next_observation, reward, done, info = env.step_rl(action)
            next_state = info["state"]
            replay_buffer.add(
                observation,
                action_idx,
                reward,
                next_observation,
                done,
                shielded_action_mask(next_state),
            )

            if (
                not in_random_phase
                and len(replay_buffer) >= max(args.batch_size, args.warmup_steps)
                and global_step % args.train_frequency == 0
            ):
                loss_value, _, _ = optimize_step(
                    policy_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    args.batch_size,
                    args.gamma,
                    device,
                    expert_dataset=expert_train_dataset if len(expert_train_dataset) > 0 else None,
                    bc_weight=bc_weight,
                )
                losses.append(loss_value)
                optimize_step_count += 1

            if optimize_step_count > 0 and optimize_step_count % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            observation = next_observation
            state = next_state
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            if not in_random_phase:
                policy_step += 1

            if args.show_game and (episode_steps % max(1, args.game_refresh_steps) == 0 or done):
                game_viewer.update(
                    state,
                    mode="train",
                    episode=episode,
                    total_episodes=args.episodes,
                    episode_step=episode_steps,
                    global_step=global_step,
                    epsilon=epsilon,
                    episode_reward=episode_reward,
                    last_reward=reward,
                    action=action,
                    q_values=top_q_values(q_values, mask),
                )

        recent_stats.append(
            EpisodeStats(
                reward=episode_reward,
                steps=episode_steps,
                won=state["game_state"].value == "won",
                truncated=info["truncated"],
                loss=float(np.mean(losses)) if losses else 0.0,
            )
        )

        avg_reward = float(np.mean([item.reward for item in recent_stats]))
        avg_steps = float(np.mean([item.steps for item in recent_stats]))
        win_rate = float(np.mean([1.0 if item.won else 0.0 for item in recent_stats]))
        avg_loss = float(np.mean([item.loss for item in recent_stats]))
        visualizer.record_train_episode(
            episode=episode,
            reward=episode_reward,
            avg_reward=avg_reward,
            win_rate=win_rate,
            avg_steps=avg_steps,
            loss=avg_loss,
            epsilon=epsilon,
            train_monsters=train_monsters,
            reward_preset=args.reward_preset,
            encoder=args.encoder,
            bootstrap_phase="off",
        )

        if episode % args.log_every == 0 or episode == 1:
            print(
                f"[train] episode={episode:4d}/{args.episodes} "
                f"reward={episode_reward:7.2f} avg_reward={avg_reward:7.2f} "
                f"win_rate={win_rate:5.2%} avg_steps={avg_steps:6.1f} "
                f"epsilon={epsilon:5.3f} loss={avg_loss:7.4f} "
                f"phase={'random' if in_random_phase else 'learn'} "
                f"train_monsters={train_monsters} "
                f"expert_rollout={expert_rollout_prob:4.2f} bc_weight={bc_weight:4.2f}"
            )

        if (not in_random_phase and episode % args.eval_every == 0) or episode == args.episodes:
            stage_metrics = evaluate_policy(
                policy_net,
                args,
                input_dim,
                device,
                num_monsters=train_monsters,
                viewer=None,
            )
            record_evaluation(
                visualizer,
                episode=episode,
                total_episodes=args.episodes,
                metrics=stage_metrics,
                reward_preset=args.reward_preset,
                encoder=args.encoder,
                eval_scope="eval_stage",
                eval_monsters=train_monsters,
            )

            if train_monsters == args.num_monsters:
                final_metrics = dict(stage_metrics)
            else:
                final_metrics = evaluate_policy(
                    policy_net,
                    args,
                    input_dim,
                    device,
                    num_monsters=args.num_monsters,
                    viewer=game_viewer if args.show_game else None,
                )
            record_evaluation(
                visualizer,
                episode=episode,
                total_episodes=args.episodes,
                metrics=final_metrics,
                reward_preset=args.reward_preset,
                encoder=args.encoder,
                eval_scope="eval_final",
                eval_monsters=args.num_monsters,
            )

            score = final_metrics["win_rate"] * 100.0 + final_metrics["avg_reward"]
            if score > best_score:
                best_score = score
                best_metrics = final_metrics
                save_checkpoint(save_path, policy_net, optimizer, args, global_step, episode, best_metrics)
                print(f"[save ] new best checkpoint -> {save_path}")
            if reached_convergence(final_metrics, args):
                consecutive_converged_evals += 1
                print(
                    f"[stop ] convergence counter="
                    f"{consecutive_converged_evals}/{args.converge_consecutive_evals}"
                )
            else:
                consecutive_converged_evals = 0
            if args.stop_on_converge and consecutive_converged_evals >= args.converge_consecutive_evals:
                print("[stop ] convergence reached on eval_final, stopping early")
                break

    if best_metrics is None:
        stage_monsters = curriculum_monsters_for_episode(max(1, args.episodes), curriculum_monsters, curriculum_boundaries)
        stage_metrics = evaluate_policy(
            policy_net,
            args,
            input_dim,
            device,
            num_monsters=stage_monsters,
            viewer=None,
        )
        record_evaluation(
            visualizer,
            episode=args.episodes,
            total_episodes=args.episodes,
            metrics=stage_metrics,
            reward_preset=args.reward_preset,
            encoder=args.encoder,
            eval_scope="eval_stage",
            eval_monsters=stage_monsters,
        )
        if stage_monsters == args.num_monsters:
            best_metrics = dict(stage_metrics)
        else:
            best_metrics = evaluate_policy(
                policy_net,
                args,
                input_dim,
                device,
                num_monsters=args.num_monsters,
                viewer=game_viewer if args.show_game else None,
            )
        record_evaluation(
            visualizer,
            episode=args.episodes,
            total_episodes=args.episodes,
            metrics=best_metrics,
            reward_preset=args.reward_preset,
            encoder=args.encoder,
            eval_scope="eval_final",
            eval_monsters=args.num_monsters,
        )
        save_checkpoint(save_path, policy_net, optimizer, args, global_step, args.episodes, best_metrics)
    game_viewer.close()
    visualizer.close()
    return save_path


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    args = build_fixed_args()
    print(f"[cfg  ] fixed configuration -> {FIXED_TRAINING_SUMMARY}")
    checkpoint_path = train(args)
    print(f"[done ] best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
