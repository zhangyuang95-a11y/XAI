"""
agent.py -- Heuristic planner and trainable DQN inference agent.
"""

from __future__ import annotations

import heapq
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from environment import (
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
    target_position_from_state,
    valid_action_mask,
)


MODEL_VERSION = 1


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mask_invalid_q_values(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    invalid_fill = torch.full_like(q_values, -1e9)
    return torch.where(valid_mask, q_values, invalid_fill)


class HeuristicAgent:
    def __init__(self, danger_radius: int = 3, danger_penalty: float = 80.0):
        self.danger_radius = danger_radius
        self.danger_penalty = danger_penalty
        self._last_reasoning = ""

    def choose_action(self, state: dict) -> str:
        player = state["player_pos"]
        exit_pos = state["exit_pos"]
        grid = state["grid"]
        grid_size = state["grid_size"]
        monsters = state["monsters"]
        dots = state.get("dots", frozenset())

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
            result_action = self._fallback_action(state)

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

    def _fallback_action(self, state: dict) -> str:
        player = state["player_pos"]
        grid = state["grid"]
        grid_size = state["grid_size"]
        monsters = state["monsters"]
        best_action = "STAY"
        best_distance = -1
        for action_name, (dr, dc) in DIRECTIONS.items():
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
        epsilon: float = 0.0,
    ):
        self.model_path = str(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.input_dim = int(checkpoint["input_dim"])
        self.action_names = tuple(checkpoint.get("action_names", ACTION_NAMES))
        self.metadata = checkpoint.get("metadata", {})
        self.danger_radius = danger_radius
        self.epsilon = epsilon
        self._rng = random.Random(self.metadata.get("seed", 0))
        self._last_reasoning = ""

        hidden_dims = tuple(checkpoint.get("hidden_dims", (512, 256, 128)))
        self.policy_net = DQNNetwork(self.input_dim, len(self.action_names), hidden_dims=hidden_dims).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state"])
        self.policy_net.eval()

    def choose_action(self, state: dict) -> str:
        observation = encode_state_vector(state, danger_radius=self.danger_radius)
        valid_mask_np = valid_action_mask(state)
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
