"""
environment.py — Pac-Man 风格迷宫环境

迷宫使用递归回溯法生成并打通额外墙壁创建环路。
所有通道格初始放置豆子，玩家必须收集全部豆子后出口才打开。
怪物在通道上随机移动（有冷却和原地不动倾向）。
"""

from __future__ import annotations

import enum
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

WALL = 1
PATH = 0

DIRECTIONS = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
    "STAY":  (0, 0),
}


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


@dataclass
class Monster:
    id: int
    row: int
    col: int
    _move_cooldown: int = 0

    def random_move(self, grid: list[list[int]], rng: random.Random,
                    move_interval: int = 2, stay_weight: int = 3) -> None:
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

    def __init__(self, grid_size: int = 21, num_monsters: int = 8,
                 seed: Optional[int] = None):
        self.grid_size = grid_size if grid_size % 2 == 1 else grid_size + 1
        self.num_monsters = num_monsters
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

        # Pac-Man dots: all PATH cells except start and exit
        self.dots: set[tuple[int, int]] = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] == PATH and (r, c) != self.start and (r, c) != self.exit:
                    self.dots.add((r, c))
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

    def _generate_maze(self) -> list[list[int]]:
        size = self.grid_size
        grid = [[WALL] * size for _ in range(size)]
        stack = [(1, 1)]
        grid[1][1] = PATH
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < size - 1 and 1 <= nc < size - 1 and grid[nr][nc] == WALL:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
            if neighbors:
                nr, nc, wr, wc = self.rng.choice(neighbors)
                grid[wr][wc] = PATH
                grid[nr][nc] = PATH
                stack.append((nr, nc))
            else:
                stack.pop()

        # Break extra walls to create loops
        extra_openings = size * size // 8
        walls = []
        for r in range(2, size - 2):
            for c in range(2, size - 2):
                if grid[r][c] == WALL:
                    if grid[r - 1][c] == PATH and grid[r + 1][c] == PATH:
                        walls.append((r, c))
                    elif grid[r][c - 1] == PATH and grid[r][c + 1] == PATH:
                        walls.append((r, c))
        self.rng.shuffle(walls)
        for r, c in walls[:extra_openings]:
            grid[r][c] = PATH
        return grid

    def _bfs_solvable(self) -> bool:
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.exit:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                        and self.grid[nr][nc] == PATH and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _force_path(self) -> None:
        sr, sc = self.start
        er, ec = self.exit
        r, c = sr, sc
        while r != er:
            self.grid[r][c] = PATH
            r += -1 if r > er else 1
        while c != ec:
            self.grid[r][c] = PATH
            c += -1 if c > ec else 1
        self.grid[er][ec] = PATH

    def _place_monsters(self) -> None:
        min_dist_from_start = 6
        min_dist_from_exit = 3
        path_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (self.grid[r][c] == PATH
                        and manhattan_distance((r, c), self.start) >= min_dist_from_start
                        and manhattan_distance((r, c), self.exit) >= min_dist_from_exit
                        and (r, c) != self.start and (r, c) != self.exit):
                    path_cells.append((r, c))
        self.rng.shuffle(path_cells)
        n = min(self.num_monsters, len(path_cells))
        self.monsters = [Monster(id=i, row=r, col=c) for i, (r, c) in enumerate(path_cells[:n])]

    @property
    def exit_open(self) -> bool:
        return len(self.dots) == 0

    def step(self, action: str) -> dict:
        if self.game_state in (GameState.WON, GameState.LOST):
            return self.get_state()
        self.game_state = GameState.RUNNING

        # 1. Move monsters
        for m in self.monsters:
            m.random_move(self.grid, self.rng)

        # 2. Check if monster moved into player
        for m in self.monsters:
            if (m.row, m.col) == self.player_pos:
                self.game_state = GameState.LOST
                self.step_count += 1
                return self.get_state()

        # 3. Move player
        dr, dc = DIRECTIONS.get(action, (0, 0))
        nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
        if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                and self.grid[nr][nc] == PATH):
            self.player_pos = (nr, nc)

        # 4. Collect dot
        if self.player_pos in self.dots:
            self.dots.remove(self.player_pos)
            self.collected_dots += 1

        # 5. Check collision with monster
        for m in self.monsters:
            if (m.row, m.col) == self.player_pos:
                self.game_state = GameState.LOST
                self.step_count += 1
                return self.get_state()

        # 6. Check win (exit open AND player on exit)
        if self.player_pos == self.exit and self.exit_open:
            self.game_state = GameState.WON
            self.step_count += 1
            return self.get_state()

        self.step_count += 1
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "grid": self.grid,
            "grid_size": self.grid_size,
            "player_pos": self.player_pos,
            "exit_pos": self.exit,
            "monsters": [(m.id, m.row, m.col) for m in self.monsters],
            "step_count": self.step_count,
            "game_state": self.game_state,
            "start_pos": self.start,
            "dots": frozenset(self.dots),
            "total_dots": self.total_dots,
            "collected_dots": self.collected_dots,
            "exit_open": self.exit_open,
        }

    def get_available_actions(self) -> list[str]:
        actions = ["STAY"]
        r, c = self.player_pos
        for name, (dr, dc) in DIRECTIONS.items():
            if name == "STAY":
                continue
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                    and self.grid[nr][nc] == PATH):
                actions.append(name)
        return actions


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    env = MazeEnvironment(grid_size=21, num_monsters=8, seed=42)
    s = env.get_state()
    print(f"Grid: {s['grid_size']}x{s['grid_size']}, Dots: {s['total_dots']}, Monsters: {len(s['monsters'])}")
    print(f"Exit open: {s['exit_open']}")
