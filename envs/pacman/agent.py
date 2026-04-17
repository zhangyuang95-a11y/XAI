"""
agent.py -- Heuristic planner and trainable DQN inference agent.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .environment import (
    ACTION_NAMES,
    ACTION_TO_INDEX,
    DIRECTIONS,
    PATH,
    WALL,
    available_actions_from_state,
    encode_state_vector,
    estimate_action_risks,
    get_relative_direction,
    manhattan_distance,
    nearest_monster_distance,
    shielded_action_mask,
    target_position_from_state,
    valid_action_mask,
)

MODEL_VERSION = 2


class DQNNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        encoder_type: str = "mlp",
        grid_size: int | None = None,
        spatial_channels: int = 8,
        scalar_dim: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        self.spatial_channels = spatial_channels
        self.scalar_dim = scalar_dim

        if encoder_type == "mlp":
            layers: list[nn.Module] = []
            last_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(nn.ReLU())
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, output_dim))
            self.net = nn.Sequential(*layers)
            self.grid_size = grid_size
            self.spatial_input_dim = 0
            self.conv = None
            self.head = None
            return

        if encoder_type != "cnn":
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.grid_size = grid_size or self._infer_grid_size(input_dim, spatial_channels, scalar_dim)
        self.spatial_input_dim = spatial_channels * self.grid_size * self.grid_size
        expected_input_dim = self.spatial_input_dim + scalar_dim
        if expected_input_dim != input_dim:
            raise ValueError(
                f"CNN encoder expects input_dim={expected_input_dim}, got {input_dim} "
                f"(grid_size={self.grid_size}, spatial_channels={spatial_channels}, scalar_dim={scalar_dim})"
            )

        self.conv = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_output_dim = 64 * self.grid_size * self.grid_size
        layers = []
        last_dim = conv_output_dim + scalar_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.head = nn.Sequential(*layers)
        self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "mlp":
            assert self.net is not None
            return self.net(x)

        assert self.conv is not None and self.head is not None
        spatial = x[:, : self.spatial_input_dim].reshape(
            x.shape[0], self.spatial_channels, self.grid_size, self.grid_size
        )
        scalar = x[:, self.spatial_input_dim : self.spatial_input_dim + self.scalar_dim]
        conv_features = self.conv(spatial).reshape(x.shape[0], -1)
        return self.head(torch.cat([conv_features, scalar], dim=1))

    @staticmethod
    def _infer_grid_size(input_dim: int, spatial_channels: int, scalar_dim: int) -> int:
        spatial_input_dim = input_dim - scalar_dim
        if spatial_input_dim <= 0 or spatial_input_dim % spatial_channels != 0:
            raise ValueError(
                f"Cannot infer grid size from input_dim={input_dim}, "
                f"spatial_channels={spatial_channels}, scalar_dim={scalar_dim}"
            )
        side = math.isqrt(spatial_input_dim // spatial_channels)
        if side * side * spatial_channels != spatial_input_dim:
            raise ValueError(
                f"Input dimension does not describe a square spatial tensor: input_dim={input_dim}, "
                f"spatial_channels={spatial_channels}, scalar_dim={scalar_dim}"
            )
        return side


def mask_invalid_q_values(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    invalid_fill = torch.full_like(q_values, -1e9)
    return torch.where(valid_mask, q_values, invalid_fill)


class HeuristicAgent:
    def __init__(
        self,
        danger_radius: int = 3,
        danger_penalty: float = 80.0,
        safety_min_distance: int = 2,
    ):
        self.danger_radius = danger_radius
        self.danger_penalty = danger_penalty
        self.safety_min_distance = safety_min_distance
        self._last_reasoning = ""

    def choose_action(self, state: dict) -> str:
        player = state["player_pos"]
        exit_pos = state["exit_pos"]
        grid = state["grid"]
        grid_size = state["grid_size"]
        monsters = state["monsters"]
        dots = state.get("dots", frozenset())
        shield_mask = shielded_action_mask(state, minimum_monster_distance=self.safety_min_distance)
        shielded_actions = {
            ACTION_NAMES[idx] for idx, allowed in enumerate(shield_mask) if allowed
        }

        if dots:
            target = self._find_nearest_dot(state)
            if target is None:
                target = exit_pos
            phase = "dots"
        else:
            target = exit_pos
            phase = "exit"

        danger = set()
        for _, mr, mc in monsters:
            for row in range(max(0, mr - self.danger_radius), min(grid_size, mr + self.danger_radius + 1)):
                for col in range(max(0, mc - self.danger_radius), min(grid_size, mc + self.danger_radius + 1)):
                    if manhattan_distance((row, col), (mr, mc)) <= self.danger_radius:
                        danger.add((row, col))

        start_row, start_col = player
        open_set = []
        heapq.heappush(open_set, (manhattan_distance(player, target), 0.0, start_row, start_col, None))
        best_g: dict[tuple[int, int], float] = {}

        result_action = None
        while open_set:
            f_score, g_score, row, col, first_action = heapq.heappop(open_set)
            del f_score
            if (row, col) == target:
                result_action = first_action
                break
            if (row, col) in best_g and best_g[(row, col)] <= g_score:
                continue
            best_g[(row, col)] = g_score

            for action_name, (dr, dc) in DIRECTIONS.items():
                if action_name == "STAY":
                    continue
                if (row, col) == player and action_name not in shielded_actions:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                if grid[nr][nc] == WALL:
                    continue
                step_cost = 1.0
                if (nr, nc) in danger:
                    step_cost += self.danger_penalty
                if any((nr, nc) == (mr, mc) for _, mr, mc in monsters):
                    step_cost += self.danger_penalty * 10
                new_g = g_score + step_cost
                new_h = manhattan_distance((nr, nc), target)
                new_first = first_action if first_action is not None else action_name
                heapq.heappush(open_set, (new_g + new_h, new_g, nr, nc, new_first))

        if result_action is None:
            result_action = self._fallback_action(state, shielded_actions=shielded_actions)

        if result_action not in shielded_actions and shielded_actions:
            result_action = self._fallback_action(state, shielded_actions=shielded_actions)

        self._last_reasoning = self._build_reasoning(state, result_action, phase, target)
        return result_action

    def _find_nearest_dot(self, state: dict) -> tuple[int, int] | None:
        player = state["player_pos"]
        dots = state.get("dots", frozenset())
        grid = state["grid"]
        grid_size = state["grid_size"]

        if not dots:
            return None

        visited = {player}
        queue = deque([player])
        while queue:
            row, col = queue.popleft()
            if (row, col) in dots:
                return row, col
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < grid_size
                    and 0 <= nc < grid_size
                    and grid[nr][nc] == PATH
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return min(dots, key=lambda dot: manhattan_distance(player, dot))

    def get_action_risks(self, state: dict) -> dict[str, float]:
        return estimate_action_risks(state, danger_radius=self.danger_radius)

    def get_reasoning(self, state: dict, chosen_action: str) -> str:
        del state, chosen_action
        return self._last_reasoning

    def _fallback_action(self, state: dict, shielded_actions: set[str] | None = None) -> str:
        player = state["player_pos"]
        grid = state["grid"]
        grid_size = state["grid_size"]
        monsters = state["monsters"]
        best_action = "STAY"
        best_distance = -1
        for action_name, (dr, dc) in DIRECTIONS.items():
            if shielded_actions is not None and action_name not in shielded_actions:
                continue
            nr, nc = player[0] + dr, player[1] + dc
            if action_name != "STAY":
                if not (0 <= nr < grid_size and 0 <= nc < grid_size) or grid[nr][nc] == WALL:
                    continue
            else:
                nr, nc = player
            distance = min((manhattan_distance((nr, nc), (mr, mc)) for _, mr, mc in monsters), default=999)
            if distance > best_distance:
                best_distance = distance
                best_action = action_name
        return best_action

    def _build_reasoning(self, state: dict, action: str, phase: str, target: tuple[int, int]) -> str:
        player = state["player_pos"]
        exit_pos = state["exit_pos"]
        monsters = state["monsters"]
        collected = state.get("collected_dots", 0)
        total = state.get("total_dots", 0)

        exit_dir = get_relative_direction(player, exit_pos)
        exit_dist = manhattan_distance(player, exit_pos)

        nearest_id, nearest_dist = -1, float("inf")
        for mid, mr, mc in monsters:
            distance = manhattan_distance(player, (mr, mc))
            if distance < nearest_dist:
                nearest_dist = distance
                nearest_id = mid

        parts = [f"A* chose {action}"]
        if phase == "dots":
            target_dir = get_relative_direction(player, target)
            target_dist = manhattan_distance(player, target)
            parts.append(f"collecting dots ({collected}/{total}), nearest dot {target_dir} dist={target_dist}")
        else:
            parts.append(f"all dots collected, heading to exit {exit_dir} dist={exit_dist}")

        if nearest_dist <= self.danger_radius + 1:
            parts.append(f"avoiding monster #{nearest_id} at dist={nearest_dist}")

        return "; ".join(parts)


class RLAgent:
    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str] = None,
        danger_radius: int = 3,
        safety_min_distance: int = 2,
        epsilon: float = 0.0,
    ):
        self.model_path = str(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.input_dim = int(checkpoint["input_dim"])
        self.action_names = tuple(checkpoint.get("action_names", ACTION_NAMES))
        self.metadata = dict(checkpoint.get("metadata", {}))
        self.danger_radius = danger_radius
        self.safety_min_distance = safety_min_distance
        self.epsilon = epsilon
        hidden_dims = tuple(checkpoint.get("hidden_dims", (512, 256, 128)))
        self.encoder_type = checkpoint.get("encoder_type", self.metadata.get("encoder_type", "mlp"))
        grid_size = checkpoint.get("grid_size", self.metadata.get("grid_size"))
        self.grid_size = int(grid_size) if grid_size is not None else None
        self.spatial_channels = int(checkpoint.get("spatial_channels", self.metadata.get("spatial_channels", 8)))
        self.scalar_dim = int(checkpoint.get("scalar_dim", self.metadata.get("scalar_dim", 7)))
        self.metadata.setdefault("encoder_type", self.encoder_type)
        self.metadata.setdefault("spatial_channels", self.spatial_channels)
        self.metadata.setdefault("scalar_dim", self.scalar_dim)
        self.metadata.setdefault("model_version", checkpoint.get("model_version", MODEL_VERSION))
        if self.grid_size is not None:
            self.metadata.setdefault("grid_size", self.grid_size)

        self._rng = random.Random(self.metadata.get("seed", 0))
        self._last_reasoning = ""

        self.policy_net = DQNNetwork(
            self.input_dim,
            len(self.action_names),
            hidden_dims=hidden_dims,
            encoder_type=self.encoder_type,
            grid_size=self.grid_size,
            spatial_channels=self.spatial_channels,
            scalar_dim=self.scalar_dim,
        ).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state"])
        self.policy_net.eval()

    def choose_action(self, state: dict) -> str:
        observation = encode_state_vector(state, danger_radius=self.danger_radius)
        valid_mask_np = shielded_action_mask(state, minimum_monster_distance=self.safety_min_distance)
        valid_actions = [action for action, allowed in zip(self.action_names, valid_mask_np) if allowed]
        if not valid_actions:
            valid_actions = ["STAY"]

        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            action = self._rng.choice(valid_actions)
            q_values = self._predict_q_values(observation)
            self._last_reasoning = self._build_reasoning(state, action, q_values, valid_mask_np, exploratory=True)
            return action

        q_values = self._predict_q_values(observation)
        masked = np.where(valid_mask_np, q_values, -1e9)
        action_idx = int(masked.argmax())
        action = self.action_names[action_idx]
        self._last_reasoning = self._build_reasoning(state, action, q_values, valid_mask_np, exploratory=False)
        return action

    def _predict_q_values(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor).squeeze(0).cpu().numpy()
        return q_values

    def get_action_risks(self, state: dict) -> dict[str, float]:
        return estimate_action_risks(state, danger_radius=self.danger_radius)

    def score_action_values(self, state: dict) -> dict[str, float]:
        observation = encode_state_vector(state, danger_radius=self.danger_radius)
        q_values = self._predict_q_values(observation)
        valid_mask_np = shielded_action_mask(state, minimum_monster_distance=self.safety_min_distance)
        masked = np.where(valid_mask_np, q_values, -1e9)
        return {
            action_name: float(score)
            for action_name, score in zip(self.action_names, masked)
            if score > -1e8
        }

    def get_reasoning(self, state: dict, chosen_action: str) -> str:
        del state, chosen_action
        return self._last_reasoning

    def _build_reasoning(
        self,
        state: dict,
        action: str,
        q_values: np.ndarray,
        valid_mask_np: np.ndarray,
        exploratory: bool,
    ) -> str:
        masked = np.where(valid_mask_np, q_values, -1e9)
        ranked_indices = list(np.argsort(masked)[::-1])
        top_bits = []
        for idx in ranked_indices[:3]:
            if masked[idx] <= -1e8:
                continue
            top_bits.append(f"{self.action_names[idx]}={masked[idx]:.2f}")

        phase = "dots" if state.get("dots") else "exit"
        target = target_position_from_state(state)
        target_dir = get_relative_direction(state["player_pos"], target)
        target_dist = manhattan_distance(state["player_pos"], target)
        nearest_threat = nearest_monster_distance(state)
        mode = "exploration" if exploratory else "policy"
        return (
            f"DQN chose {action} via {mode}; "
            f"phase={phase}; target={target_dir} dist={target_dist}; "
            f"nearest_monster_dist={nearest_threat}; "
            f"q_values[{', '.join(top_bits)}]"
        )
