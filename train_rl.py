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

from agent import DQNNetwork, MODEL_VERSION, mask_invalid_q_values
from environment import ACTION_NAMES, MazeEnvironment, valid_action_mask
from training_game_viewer import TrainingGameViewer


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
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "kind",
                    "episode",
                    "reward",
                    "avg_reward",
                    "win_rate",
                    "avg_steps",
                    "loss",
                    "epsilon",
                    "truncation_rate",
                ],
            )
            writer.writeheader()

    def record_train_episode(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        win_rate: float,
        avg_steps: float,
        loss: float,
        epsilon: float,
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
            }
        )
        if episode % self.refresh_every == 0:
            self.refresh()

    def record_eval(self, episode: int, metrics: dict[str, float]) -> None:
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
            }
        )
        self.refresh(force=True)

    def _append_csv(self, row: dict) -> None:
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "kind",
                "episode",
                "reward",
                "avg_reward",
                "win_rate",
                "avg_steps",
                "loss",
                "epsilon",
                "truncation_rate",
            ])
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


def add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{dest: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DQN Pac-Man agent.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--num-monsters", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means grid_size*grid_size*4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=40000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="models/dqn_pacman.pt")
    parser.add_argument("--plot-path", type=str, default="artifacts/training_progress.png")
    parser.add_argument("--metrics-path", type=str, default="artifacts/training_metrics.csv")
    parser.add_argument("--plot-refresh-every", type=int, default=5)
    parser.add_argument("--game-refresh-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    add_bool_flag(parser, "show-plot", default=True, help_text="Show the live matplotlib dashboard.")
    add_bool_flag(parser, "show-game", default=True, help_text="Show the live Pac-Man training window.")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def epsilon_by_step(global_step: int, start: float, end: float, decay: int) -> float:
    if decay <= 0:
        return end
    progress = np.exp(-global_step / decay)
    return float(end + (start - end) * progress)


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


def optimize_step(
    policy_net: DQNNetwork,
    target_net: DQNNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
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

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


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
    viewer: TrainingGameViewer | None = None,
) -> dict[str, float]:
    del input_dim
    policy_net.eval()
    eval_env = MazeEnvironment(
        grid_size=args.grid_size,
        num_monsters=args.num_monsters,
        seed=args.seed + 10_000,
        max_steps=args.max_steps or None,
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
            mask = valid_action_mask(state)
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
    checkpoint = {
        "model_state": policy_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_dim": next(policy_net.parameters()).shape[1],
        "action_names": ACTION_NAMES,
        "hidden_dims": tuple(args.hidden_dims),
        "model_version": MODEL_VERSION,
        "metadata": {
            "algorithm": "DQN",
            "grid_size": args.grid_size,
            "num_monsters": args.num_monsters,
            "seed": args.seed,
            "episode": episode,
            "global_step": global_step,
            "best_metrics": best_metrics or {},
            "reward_config": MazeEnvironment(
                grid_size=args.grid_size,
                num_monsters=args.num_monsters,
                seed=args.seed,
                max_steps=args.max_steps or None,
            ).reward_config,
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

    env = MazeEnvironment(
        grid_size=args.grid_size,
        num_monsters=args.num_monsters,
        seed=args.seed,
        max_steps=args.max_steps or None,
    )
    initial_obs = env.reset_rl(seed=args.seed)
    input_dim = int(initial_obs.shape[0])
    output_dim = len(ACTION_NAMES)
    hidden_dims = tuple(args.hidden_dims)

    policy_net = DQNNetwork(input_dim, output_dim, hidden_dims=hidden_dims).to(device)
    target_net = DQNNetwork(input_dim, output_dim, hidden_dims=hidden_dims).to(device)
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

    global_step = 0
    best_score = float("-inf")
    best_metrics: dict[str, float] | None = None
    recent_stats: deque[EpisodeStats] = deque(maxlen=max(1, args.log_every))

    for episode in range(1, args.episodes + 1):
        observation = env.reset_rl(seed=args.seed + episode)
        state = env.get_state()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        losses: list[float] = []

        while not done:
            epsilon = epsilon_by_step(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay)
            mask = valid_action_mask(state)
            action_idx, q_values = select_action(policy_net, observation, mask, epsilon, device, rng)
            action = ACTION_NAMES[action_idx]

            next_observation, reward, done, info = env.step_rl(action)
            next_state = info["state"]
            replay_buffer.add(observation, action_idx, reward, next_observation, done, info["valid_action_mask"])

            if len(replay_buffer) >= max(args.batch_size, args.warmup_steps) and global_step % args.train_frequency == 0:
                loss_value = optimize_step(
                    policy_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    args.batch_size,
                    args.gamma,
                    device,
                )
                losses.append(loss_value)

            if global_step > 0 and global_step % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            observation = next_observation
            state = next_state
            episode_reward += reward
            episode_steps += 1
            global_step += 1

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
        )

        if episode % args.log_every == 0 or episode == 1:
            print(
                f"[train] episode={episode:4d}/{args.episodes} "
                f"reward={episode_reward:7.2f} avg_reward={avg_reward:7.2f} "
                f"win_rate={win_rate:5.2%} avg_steps={avg_steps:6.1f} "
                f"epsilon={epsilon:5.3f} loss={avg_loss:7.4f}"
            )

        if episode % args.eval_every == 0 or episode == args.episodes:
            metrics = evaluate_policy(policy_net, args, input_dim, device, viewer=game_viewer if args.show_game else None)
            score = metrics["win_rate"] * 100.0 + metrics["avg_reward"]
            print(
                f"[eval ] episode={episode:4d}/{args.episodes} "
                f"avg_reward={metrics['avg_reward']:7.2f} "
                f"win_rate={metrics['win_rate']:5.2%} "
                f"avg_steps={metrics['avg_steps']:6.1f} "
                f"truncation_rate={metrics['truncation_rate']:5.2%}"
            )
            visualizer.record_eval(episode, metrics)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                save_checkpoint(save_path, policy_net, optimizer, args, global_step, episode, best_metrics)
                print(f"[save ] new best checkpoint -> {save_path}")

    if best_metrics is None:
        best_metrics = evaluate_policy(policy_net, args, input_dim, device, viewer=game_viewer if args.show_game else None)
        save_checkpoint(save_path, policy_net, optimizer, args, global_step, args.episodes, best_metrics)
        visualizer.record_eval(args.episodes, best_metrics)
    game_viewer.close()
    visualizer.close()
    return save_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    checkpoint_path = train(args)
    print(f"[done ] best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
