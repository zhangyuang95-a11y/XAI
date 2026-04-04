"""
agent.py — Multi-objective A* Agent (Pac-Man style)

Phase 1: dots remain → A* toward nearest reachable dot (with monster avoidance)
Phase 2: all dots collected → A* toward exit (with monster avoidance)

Interface (for RL replacement):
  choose_action(state) -> str
  get_action_risks(state) -> dict
  get_reasoning(state, action) -> str
"""

from __future__ import annotations

import heapq
from collections import deque
from environment import DIRECTIONS, WALL, PATH, manhattan_distance, get_relative_direction


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
        exit_open = state.get("exit_open", True)

        # Determine target
        if dots:
            target = self._find_nearest_dot(state)
            if target is None:
                target = exit_pos
            phase = "dots"
        else:
            target = exit_pos
            phase = "exit"

        # Build danger zone
        danger = set()
        for _, mr, mc in monsters:
            for r in range(max(0, mr - self.danger_radius),
                           min(grid_size, mr + self.danger_radius + 1)):
                for c in range(max(0, mc - self.danger_radius),
                               min(grid_size, mc + self.danger_radius + 1)):
                    if manhattan_distance((r, c), (mr, mc)) <= self.danger_radius:
                        danger.add((r, c))

        # A* search toward target
        start_r, start_c = player
        open_set = []
        h0 = manhattan_distance(player, target)
        heapq.heappush(open_set, (h0, 0.0, start_r, start_c, None))
        best_g: dict[tuple[int, int], float] = {}

        result_action = None
        while open_set:
            f, g, r, c, first_act = heapq.heappop(open_set)
            if (r, c) == target:
                result_action = first_act
                break
            if (r, c) in best_g and best_g[(r, c)] <= g:
                continue
            best_g[(r, c)] = g

            for action_name, (dr, dc) in DIRECTIONS.items():
                if action_name == "STAY":
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                if grid[nr][nc] == WALL:
                    continue
                step_cost = 1.0
                if (nr, nc) in danger:
                    step_cost += self.danger_penalty
                for _, mr, mc in monsters:
                    if (nr, nc) == (mr, mc):
                        step_cost += self.danger_penalty * 10
                new_g = g + step_cost
                new_h = manhattan_distance((nr, nc), target)
                new_f = new_g + new_h
                new_first = first_act if first_act is not None else action_name
                heapq.heappush(open_set, (new_f, new_g, nr, nc, new_first))

        if result_action is None:
            result_action = self._fallback_action(state)

        self._last_reasoning = self._build_reasoning(state, result_action, danger, phase, target)
        return result_action

    def _find_nearest_dot(self, state: dict) -> tuple[int, int] | None:
        """BFS find nearest reachable dot from player position."""
        player = state["player_pos"]
        dots = state.get("dots", frozenset())
        grid = state["grid"]
        grid_size = state["grid_size"]

        if not dots:
            return None

        visited = set()
        queue = deque([player])
        visited.add(player)
        while queue:
            r, c = queue.popleft()
            if (r, c) in dots:
                return (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < grid_size and 0 <= nc < grid_size
                        and grid[nr][nc] == PATH and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        # Fallback: manhattan nearest
        return min(dots, key=lambda d: manhattan_distance(player, d))

    def get_action_risks(self, state: dict) -> dict[str, float]:
        player = state["player_pos"]
        grid = state["grid"]
        grid_size = state["grid_size"]
        monsters = state["monsters"]
        risks = {}
        for action_name, (dr, dc) in DIRECTIONS.items():
            nr, nc = player[0] + dr, player[1] + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size) or grid[nr][nc] == WALL:
                if action_name == "STAY":
                    nr, nc = player
                else:
                    continue
            min_dist = float("inf")
            for _, mr, mc in monsters:
                d = manhattan_distance((nr, nc), (mr, mc))
                min_dist = min(min_dist, d)
            if min_dist == 0:
                risk = 1.0
            elif min_dist <= self.danger_radius + 1:
                risk = max(0.0, 1.0 - min_dist / (self.danger_radius + 2))
            else:
                risk = 0.0
            risks[action_name] = round(risk, 3)
        return risks

    def get_reasoning(self, state: dict, chosen_action: str) -> str:
        return self._last_reasoning

    def _fallback_action(self, state: dict) -> str:
        player = state["player_pos"]
        grid = state["grid"]
        grid_size = state["grid_size"]
        monsters = state["monsters"]
        best_action, best_dist = "STAY", -1
        for action_name, (dr, dc) in DIRECTIONS.items():
            nr, nc = player[0] + dr, player[1] + dc
            if action_name != "STAY":
                if not (0 <= nr < grid_size and 0 <= nc < grid_size) or grid[nr][nc] == WALL:
                    continue
            else:
                nr, nc = player
            min_d = min((manhattan_distance((nr, nc), (mr, mc)) for _, mr, mc in monsters), default=999)
            if min_d > best_dist:
                best_dist = min_d
                best_action = action_name
        return best_action

    def _build_reasoning(self, state: dict, action: str, danger: set,
                         phase: str, target: tuple[int, int]) -> str:
        player = state["player_pos"]
        exit_pos = state["exit_pos"]
        monsters = state["monsters"]
        collected = state.get("collected_dots", 0)
        total = state.get("total_dots", 0)

        exit_dir = get_relative_direction(player, exit_pos)
        exit_dist = manhattan_distance(player, exit_pos)

        nearest_id, nearest_dist = -1, float("inf")
        for mid, mr, mc in monsters:
            d = manhattan_distance(player, (mr, mc))
            if d < nearest_dist:
                nearest_dist = d
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
