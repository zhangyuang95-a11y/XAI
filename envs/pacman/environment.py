"""
environment.py -- Pac-Man style maze environment with RL-friendly helpers.
"""

from __future__ import annotations

import enum
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

WALL = 1
PATH = 0

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "STAY": (0, 0),
}

ACTION_NAMES = tuple(DIRECTIONS.keys())
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
LEGACY_REWARD_CONFIG = {
    "step": -0.08,
    "invalid_move": -0.55,
    "stay": -0.25,
    "stall": -0.12,
    "dot": 1.4,
    "unlock_exit": 6.0,
    "win": 32.0,
    "win_speed_bonus": 18.0,
    "lose": -30.0,
    "timeout": -18.0,
    "target_progress": 0.24,
    "target_regress": -0.18,
    "monster_closer": -0.14,
    "monster_farther": 0.03,
    "danger_zone": -0.22,
    "time_pressure": -0.12,
}
STABLE_REWARD_CONFIG = {
    "step": -0.08,
    "invalid_move": -0.55,
    "stay": -0.25,
    "stall": -0.12,
    "dot": 1.4,
    "unlock_exit": 6.0,
    "win": 32.0,
    "win_speed_bonus": 18.0,
    "lose": -30.0,
    "timeout": -30.0,
    "target_potential": 0.35,
    "monster_closer": -0.14,
    "monster_farther": 0.03,
    "danger_zone": -0.15,
    "time_pressure": -0.05,
}
REWARD_PRESETS = {
    "legacy": LEGACY_REWARD_CONFIG,
    "stable": STABLE_REWARD_CONFIG,
}
DEFAULT_REWARD_CONFIG = LEGACY_REWARD_CONFIG


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_relative_direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    if dr == 0 and dc == 0:
        return "same"
    ns = "north" if dr < 0 else ("south" if dr > 0 else "")
    ew = "west" if dc < 0 else ("east" if dc > 0 else "")
    return f"{ns}-{ew}" if ns and ew else (ns or ew)


def shortest_path_distances(
    grid: list[list[int]],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    if not grid or not grid[0]:
        return {}

    rows = len(grid)
    cols = len(grid[0])
    row, col = start
    if not (0 <= row < rows and 0 <= col < cols) or grid[row][col] != PATH:
        return {}

    distances = {start: 0}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        next_distance = distances[(row, col)] + 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and grid[nr][nc] == PATH
                and (nr, nc) not in distances
            ):
                distances[(nr, nc)] = next_distance
                queue.append((nr, nc))
    return distances


def shortest_path_distance(
    grid: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> int | None:
    return shortest_path_distances(grid, start).get(goal)


def nearest_dot_position(
    state: dict,
    distances: dict[tuple[int, int], int] | None = None,
) -> tuple[int, int] | None:
    player = state["player_pos"]
    dots = state.get("dots", frozenset())
    if not dots:
        return None
    if distances is None:
        distances = shortest_path_distances(state["grid"], player)
    reachable = [dot for dot in dots if dot in distances]
    if not reachable:
        return None
    return min(reachable, key=lambda dot: (distances[dot], manhattan_distance(player, dot)))


def target_position_from_state(
    state: dict,
    distances: dict[tuple[int, int], int] | None = None,
) -> tuple[int, int]:
    nearest_dot = nearest_dot_position(state, distances=distances)
    return nearest_dot if nearest_dot is not None else state["exit_pos"]


def objective_distance(
    state: dict,
    distances: dict[tuple[int, int], int] | None = None,
) -> int:
    if distances is None:
        distances = shortest_path_distances(state["grid"], state["player_pos"])
    target = target_position_from_state(state, distances=distances)
    return distances.get(target, state["grid_size"] * state["grid_size"])


def nearest_monster_distance(state: dict) -> int:
    player = state["player_pos"]
    monsters = state["monsters"]
    return min((manhattan_distance(player, (mr, mc)) for _, mr, mc in monsters), default=999)


def available_actions_from_state(state: dict) -> list[str]:
    grid = state["grid"]
    grid_size = state["grid_size"]
    player = state["player_pos"]
    actions = ["STAY"]
    for name, (dr, dc) in DIRECTIONS.items():
        if name == "STAY":
            continue
        nr, nc = player[0] + dr, player[1] + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr][nc] == PATH:
            actions.append(name)
    return actions


def valid_action_mask(state: dict) -> np.ndarray:
    allowed = set(available_actions_from_state(state))
    return np.array([action in allowed for action in ACTION_NAMES], dtype=np.bool_)


def projected_player_position(state: dict, action_name: str) -> tuple[int, int] | None:
    grid = state["grid"]
    grid_size = state["grid_size"]
    player = state["player_pos"]
    dr, dc = DIRECTIONS[action_name]
    nr, nc = player[0] + dr, player[1] + dc

    if action_name == "STAY":
        return player
    if not (0 <= nr < grid_size and 0 <= nc < grid_size):
        return None
    if grid[nr][nc] != PATH:
        return None
    return nr, nc


def shielded_action_mask(state: dict, minimum_monster_distance: int = 2) -> np.ndarray:
    valid_mask = valid_action_mask(state)
    monsters = state["monsters"]
    shielded_mask = np.zeros_like(valid_mask)

    for idx, action_name in enumerate(ACTION_NAMES):
        if not valid_mask[idx]:
            continue
        next_pos = projected_player_position(state, action_name)
        if next_pos is None:
            continue
        nearest_dist = min(
            (manhattan_distance(next_pos, (mr, mc)) for _, mr, mc in monsters),
            default=999,
        )
        if nearest_dist >= minimum_monster_distance:
            shielded_mask[idx] = True

    return shielded_mask if shielded_mask.any() else valid_mask


def estimate_action_risks(state: dict, danger_radius: int = 3) -> dict[str, float]:
    player = state["player_pos"]
    grid = state["grid"]
    grid_size = state["grid_size"]
    monsters = state["monsters"]
    risks: dict[str, float] = {}

    for action_name, (dr, dc) in DIRECTIONS.items():
        nr, nc = player[0] + dr, player[1] + dc
        if not (0 <= nr < grid_size and 0 <= nc < grid_size) or grid[nr][nc] == WALL:
            if action_name == "STAY":
                nr, nc = player
            else:
                continue

        min_dist = min((manhattan_distance((nr, nc), (mr, mc)) for _, mr, mc in monsters), default=999)
        if min_dist == 0:
            risk = 1.0
        elif min_dist <= danger_radius + 1:
            risk = max(0.0, 1.0 - min_dist / (danger_radius + 2))
        else:
            risk = 0.0
        risks[action_name] = round(risk, 3)
    return risks


def encode_state_vector(state: dict, danger_radius: int = 3) -> np.ndarray:
    grid = state["grid"]
    size = state["grid_size"]
    player = state["player_pos"]
    exit_pos = state["exit_pos"]
    dots = state.get("dots", frozenset())
    monsters = state["monsters"]

    wall_channel = np.zeros((size, size), dtype=np.float32)
    path_channel = np.zeros((size, size), dtype=np.float32)
    player_channel = np.zeros((size, size), dtype=np.float32)
    exit_channel = np.zeros((size, size), dtype=np.float32)
    dots_channel = np.zeros((size, size), dtype=np.float32)
    monsters_channel = np.zeros((size, size), dtype=np.float32)
    danger_channel = np.zeros((size, size), dtype=np.float32)
    exit_open_channel = np.full((size, size), 1.0 if state.get("exit_open", False) else 0.0, dtype=np.float32)

    for row in range(size):
        for col in range(size):
            if grid[row][col] == WALL:
                wall_channel[row, col] = 1.0
            else:
                path_channel[row, col] = 1.0

    player_channel[player[0], player[1]] = 1.0
    exit_channel[exit_pos[0], exit_pos[1]] = 1.0

    for row, col in dots:
        dots_channel[row, col] = 1.0

    for _, mr, mc in monsters:
        monsters_channel[mr, mc] = 1.0
        for row in range(max(0, mr - danger_radius), min(size, mr + danger_radius + 1)):
            for col in range(max(0, mc - danger_radius), min(size, mc + danger_radius + 1)):
                if manhattan_distance((row, col), (mr, mc)) <= danger_radius:
                    danger_channel[row, col] = 1.0

    nearest_dot = nearest_dot_position(state)
    nearest_dot_dist = manhattan_distance(player, nearest_dot) if nearest_dot is not None else size * 2
    nearest_monster_dist = nearest_monster_distance(state)
    exit_dist = manhattan_distance(player, exit_pos)
    max_distance = max(1, size * 2)
    total_dots = max(1, state.get("total_dots", 1))

    step_budget = max(1, int(state.get("max_steps", size * size * 2)))
    scalar_features = np.array(
        [
            state.get("step_count", 0) / step_budget,
            state.get("collected_dots", 0) / total_dots,
            len(dots) / total_dots,
            1.0 if state.get("exit_open", False) else 0.0,
            nearest_dot_dist / max_distance,
            exit_dist / max_distance,
            min(nearest_monster_dist, max_distance) / max_distance,
        ],
        dtype=np.float32,
    )

    spatial = np.stack(
        [
            wall_channel,
            path_channel,
            player_channel,
            exit_channel,
            dots_channel,
            monsters_channel,
            danger_channel,
            exit_open_channel,
        ],
        axis=0,
    ).reshape(-1)
    return np.concatenate([spatial, scalar_features], dtype=np.float32)


@dataclass
class Monster:
    id: int
    row: int
    col: int
    _move_cooldown: int = 0

    def random_move(
        self,
        grid: list[list[int]],
        rng: random.Random,
        move_interval: int = 2,
        stay_weight: int = 3,
    ) -> None:
        self._move_cooldown += 1
        if self._move_cooldown < move_interval:
            return
        self._move_cooldown = 0
        candidates = [(self.row, self.col)] * stay_weight
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = self.row + dr, self.col + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == PATH:
                candidates.append((nr, nc))
        self.row, self.col = rng.choice(candidates)


class GameState(enum.Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    WON = "won"
    LOST = "lost"


class MazeEnvironment:
    def __init__(
        self,
        grid_size: int = 15,
        num_monsters: int = 4,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        reward_config: Optional[dict[str, float]] = None,
        reward_preset: str = "legacy",
    ):
        self.grid_size = grid_size if grid_size % 2 == 1 else grid_size + 1
        self.num_monsters = num_monsters
        self.max_steps = max_steps or self.grid_size * self.grid_size * 2
        if reward_preset not in REWARD_PRESETS:
            raise ValueError(f"Unknown reward preset: {reward_preset}")
        self.reward_preset = reward_preset
        self.reward_config = {**REWARD_PRESETS[reward_preset], **(reward_config or {})}
        self.rng = random.Random(seed)
        self._init_game()

    def _init_game(self) -> None:
        for _ in range(20):
            self.grid = self._generate_maze()
            self.start = (self.grid_size - 2, 1)
            self.exit = (1, self.grid_size - 2)
            self.grid[self.start[0]][self.start[1]] = PATH
            self.grid[self.exit[0]][self.exit[1]] = PATH
            if self._bfs_solvable():
                break
        else:
            self._force_path()

        self.dots: set[tuple[int, int]] = set()
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == PATH and (row, col) not in (self.start, self.exit):
                    self.dots.add((row, col))
        self.total_dots = len(self.dots)
        self.collected_dots = 0

        self.monsters: list[Monster] = []
        self._place_monsters()
        self.player_pos = self.start
        self.step_count = 0
        self.game_state = GameState.READY

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = random.Random(seed)
        self._init_game()
        return self.get_state()

    def reset_rl(self, seed: Optional[int] = None) -> np.ndarray:
        return encode_state_vector(self.reset(seed=seed))

    def _generate_maze(self) -> list[list[int]]:
        size = self.grid_size
        grid = [[WALL] * size for _ in range(size)]
        stack = [(1, 1)]
        grid[1][1] = PATH
        while stack:
            row, col = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = row + dr, col + dc
                if 1 <= nr < size - 1 and 1 <= nc < size - 1 and grid[nr][nc] == WALL:
                    neighbors.append((nr, nc, row + dr // 2, col + dc // 2))
            if neighbors:
                nr, nc, wr, wc = self.rng.choice(neighbors)
                grid[wr][wc] = PATH
                grid[nr][nc] = PATH
                stack.append((nr, nc))
            else:
                stack.pop()

        extra_openings = size * size // 8
        walls = []
        for row in range(2, size - 2):
            for col in range(2, size - 2):
                if grid[row][col] != WALL:
                    continue
                if grid[row - 1][col] == PATH and grid[row + 1][col] == PATH:
                    walls.append((row, col))
                elif grid[row][col - 1] == PATH and grid[row][col + 1] == PATH:
                    walls.append((row, col))
        self.rng.shuffle(walls)
        for row, col in walls[:extra_openings]:
            grid[row][col] = PATH
        return grid

    def _bfs_solvable(self) -> bool:
        visited = {self.start}
        queue = deque([self.start])
        while queue:
            row, col = queue.popleft()
            if (row, col) == self.exit:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and self.grid[nr][nc] == PATH
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _force_path(self) -> None:
        sr, sc = self.start
        er, ec = self.exit
        row, col = sr, sc
        while row != er:
            self.grid[row][col] = PATH
            row += -1 if row > er else 1
        while col != ec:
            self.grid[row][col] = PATH
            col += -1 if col > ec else 1
        self.grid[er][ec] = PATH

    def _place_monsters(self) -> None:
        min_dist_from_start = 6
        min_dist_from_exit = 3
        path_cells = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (
                    self.grid[row][col] == PATH
                    and manhattan_distance((row, col), self.start) >= min_dist_from_start
                    and manhattan_distance((row, col), self.exit) >= min_dist_from_exit
                    and (row, col) not in (self.start, self.exit)
                ):
                    path_cells.append((row, col))
        self.rng.shuffle(path_cells)
        count = min(self.num_monsters, len(path_cells))
        self.monsters = [Monster(id=idx, row=row, col=col) for idx, (row, col) in enumerate(path_cells[:count])]

    @property
    def exit_open(self) -> bool:
        return not self.dots

    def step(self, action: str) -> dict:
        state, _ = self._transition(action)
        return state

    def step_rl(self, action: str) -> tuple[np.ndarray, float, bool, dict]:
        next_state, transition = self._transition(action)
        reward, breakdown = self._compute_reward(transition)
        truncated = False
        if not transition["won"] and not transition["lost"] and next_state["step_count"] >= self.max_steps:
            truncated = True
            reward += self.reward_config["timeout"]
            breakdown["timeout"] = self.reward_config["timeout"]
        done = transition["won"] or transition["lost"] or truncated
        info = {
            "state": next_state,
            "valid_action_mask": valid_action_mask(next_state),
            "reward_breakdown": breakdown,
            "truncated": truncated,
        }
        return encode_state_vector(next_state), float(reward), done, info

    def _transition(self, action: str) -> tuple[dict, dict]:
        if self.game_state in (GameState.WON, GameState.LOST):
            state = self.get_state()
            transition = self._build_transition_summary(state, state, action, False, False, False)
            transition["lost"] = self.game_state == GameState.LOST
            transition["won"] = self.game_state == GameState.WON
            return state, transition

        self.game_state = GameState.RUNNING
        before_state = self.get_state()
        before_pos = self.player_pos
        before_dots = len(self.dots)
        before_exit_open = self.exit_open

        for monster in self.monsters:
            monster.random_move(self.grid, self.rng)

        monster_collision = any((monster.row, monster.col) == self.player_pos for monster in self.monsters)
        valid_move = True
        if not monster_collision:
            dr, dc = DIRECTIONS.get(action, (0, 0))
            nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
            if action != "STAY" and not (
                0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.grid[nr][nc] == PATH
            ):
                valid_move = False
            else:
                self.player_pos = (nr, nc) if action != "STAY" else self.player_pos

        dot_collected = False
        if not monster_collision and self.player_pos in self.dots:
            self.dots.remove(self.player_pos)
            self.collected_dots += 1
            dot_collected = True

        player_collision = any((monster.row, monster.col) == self.player_pos for monster in self.monsters)
        won = self.player_pos == self.exit and self.exit_open and not player_collision and not monster_collision
        lost = monster_collision or player_collision

        if lost:
            self.game_state = GameState.LOST
        elif won:
            self.game_state = GameState.WON

        self.step_count += 1
        after_state = self.get_state()
        transition = self._build_transition_summary(
            before_state,
            after_state,
            action,
            valid_move,
            dot_collected,
            (not before_exit_open) and after_state["exit_open"],
        )
        transition["lost"] = lost
        transition["won"] = won
        transition["monster_collision"] = monster_collision
        transition["player_collision"] = player_collision
        transition["dots_before"] = before_dots
        transition["dots_after"] = len(self.dots)
        transition["player_before"] = before_pos
        transition["player_after"] = self.player_pos
        return after_state, transition

    def _build_transition_summary(
        self,
        before_state: dict,
        after_state: dict,
        action: str,
        valid_move: bool,
        dot_collected: bool,
        exit_opened: bool,
    ) -> dict:
        before_distances = shortest_path_distances(before_state["grid"], before_state["player_pos"])
        after_distances = shortest_path_distances(after_state["grid"], after_state["player_pos"])
        return {
            "action": action,
            "valid_move": valid_move,
            "dot_collected": dot_collected,
            "exit_opened": exit_opened,
            "lost": False,
            "won": False,
            "target_distance_before": objective_distance(before_state, distances=before_distances),
            "target_distance_after": objective_distance(after_state, distances=after_distances),
            "nearest_monster_before": nearest_monster_distance(before_state),
            "nearest_monster_after": nearest_monster_distance(after_state),
            "used_stay": action == "STAY",
            "before_state": before_state,
            "after_state": after_state,
        }

    def _compute_reward(self, transition: dict) -> tuple[float, dict[str, float]]:
        cfg = self.reward_config
        reward = cfg["step"]
        breakdown: dict[str, float] = {"step": cfg["step"]}
        step_ratio = min(1.0, transition["after_state"]["step_count"] / max(1, self.max_steps))

        if cfg.get("time_pressure", 0.0):
            time_penalty = cfg["time_pressure"] * (0.35 + 0.65 * step_ratio)
            reward += time_penalty
            breakdown["time_pressure"] = time_penalty

        if not transition["valid_move"] and transition["action"] != "STAY":
            reward += cfg["invalid_move"]
            breakdown["invalid_move"] = cfg["invalid_move"]

        if transition["used_stay"]:
            reward += cfg["stay"]
            breakdown["stay"] = cfg["stay"]

        distance_delta = transition["target_distance_before"] - transition["target_distance_after"]
        potential_scale = cfg.get("target_potential")
        if potential_scale is not None and distance_delta != 0:
            delta_reward = potential_scale * distance_delta
            reward += delta_reward
            breakdown["target_progress" if delta_reward > 0 else "target_regress"] = delta_reward
        elif distance_delta > 0:
            delta_reward = cfg["target_progress"] * distance_delta
            reward += delta_reward
            breakdown["target_progress"] = delta_reward
        elif distance_delta < 0:
            delta_penalty = cfg["target_regress"] * abs(distance_delta)
            reward += delta_penalty
            breakdown["target_regress"] = delta_penalty
        elif not transition["dot_collected"] and not transition["exit_opened"] and not transition["won"]:
            reward += cfg["stall"]
            breakdown["stall"] = cfg["stall"]

        monster_delta = transition["nearest_monster_after"] - transition["nearest_monster_before"]
        if monster_delta > 0:
            monster_reward = cfg["monster_farther"] * min(monster_delta, 2)
            reward += monster_reward
            breakdown["monster_farther"] = monster_reward
        elif monster_delta < 0:
            monster_penalty = cfg["monster_closer"] * min(abs(monster_delta), 2)
            reward += monster_penalty
            breakdown["monster_closer"] = monster_penalty

        if transition["nearest_monster_after"] <= 2 and not transition["lost"]:
            reward += cfg["danger_zone"]
            breakdown["danger_zone"] = cfg["danger_zone"]

        if transition["dot_collected"]:
            reward += cfg["dot"]
            breakdown["dot"] = cfg["dot"]

        if transition["exit_opened"]:
            reward += cfg["unlock_exit"]
            breakdown["unlock_exit"] = cfg["unlock_exit"]

        if transition["won"]:
            reward += cfg["win"]
            breakdown["win"] = cfg["win"]
            speed_bonus = cfg["win_speed_bonus"] * (1.0 - step_ratio)
            reward += speed_bonus
            breakdown["win_speed_bonus"] = speed_bonus

        if transition["lost"]:
            reward += cfg["lose"]
            breakdown["lose"] = cfg["lose"]

        return reward, breakdown

    def get_state(self) -> dict:
        return {
            "grid": self.grid,
            "grid_size": self.grid_size,
            "player_pos": self.player_pos,
            "exit_pos": self.exit,
            "monsters": [(monster.id, monster.row, monster.col) for monster in self.monsters],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "game_state": self.game_state,
            "start_pos": self.start,
            "dots": frozenset(self.dots),
            "total_dots": self.total_dots,
            "collected_dots": self.collected_dots,
            "exit_open": self.exit_open,
        }

    def get_available_actions(self) -> list[str]:
        return available_actions_from_state(self.get_state())


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    env = MazeEnvironment(grid_size=11, num_monsters=2, seed=42, reward_preset="stable")
    state = env.get_state()
    print(f"Grid: {state['grid_size']}x{state['grid_size']}, Dots: {state['total_dots']}, Monsters: {len(state['monsters'])}")
    print(f"Exit open: {state['exit_open']}")
    obs = env.reset_rl(seed=42)
    next_obs, reward, done, info = env.step_rl("RIGHT")
    print(f"Observation dim: {obs.shape[0]}")
    print(f"Sample reward: {reward:.3f}, done={done}, truncated={info['truncated']}")
