"""
evidence_recorder.py — 滚动 30 步结构化证据记录器（含豆子收集信息）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from environment import manhattan_distance, get_relative_direction, DIRECTIONS, WALL


@dataclass(frozen=True)
class EvidenceRecord:
    step: int
    player_pos: tuple[int, int]
    exit_pos: tuple[int, int]
    monster_positions: tuple[tuple[int, int, int], ...]
    exit_direction: str
    exit_distance: int
    nearest_monster_id: int
    nearest_monster_direction: str
    nearest_monster_distance: int
    monster_distances: tuple[tuple[int, int], ...]
    available_actions: tuple[str, ...]
    chosen_action: str
    collision_risks: tuple[tuple[str, float], ...]
    has_safer_alternative: bool
    reasoning: str
    # Dot fields
    dots_remaining: int
    dots_collected: int
    total_dots: int
    nearest_dot_direction: str
    nearest_dot_distance: int
    exit_open: bool


class EvidenceRecorder:

    def __init__(self, max_history: int = 30):
        self._history: deque[EvidenceRecord] = deque(maxlen=max_history)

    def record(self, state: dict, agent, chosen_action: str) -> EvidenceRecord:
        player = state["player_pos"]
        exit_pos = state["exit_pos"]
        monsters = state["monsters"]

        exit_dir = get_relative_direction(player, exit_pos)
        exit_dist = manhattan_distance(player, exit_pos)

        m_dists = []
        nearest_id, nearest_dist = -1, 999999
        for mid, mr, mc in monsters:
            d = manhattan_distance(player, (mr, mc))
            m_dists.append((mid, d))
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = mid

        nearest_dir = "none"
        for mid, mr, mc in monsters:
            if mid == nearest_id:
                nearest_dir = get_relative_direction(player, (mr, mc))
                break

        risks = agent.get_action_risks(state)
        risk_tuples = tuple(sorted(risks.items()))
        chosen_risk = risks.get(chosen_action, 0.0)
        has_safer = any(r < chosen_risk for a, r in risks.items() if a != chosen_action)

        grid = state["grid"]
        grid_size = state["grid_size"]
        avail = ["STAY"]
        for name, (dr, dc) in DIRECTIONS.items():
            if name == "STAY":
                continue
            nr, nc = player[0] + dr, player[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr][nc] != WALL:
                avail.append(name)

        reasoning = agent.get_reasoning(state, chosen_action)

        # Dot info
        dots = state.get("dots", frozenset())
        dots_remaining = len(dots)
        dots_collected = state.get("collected_dots", 0)
        total_dots = state.get("total_dots", 0)
        exit_open = state.get("exit_open", True)

        nearest_dot_dist = -1
        nearest_dot_dir = "none"
        if dots:
            best_d = 999999
            best_pos = None
            for dp in dots:
                d = manhattan_distance(player, dp)
                if d < best_d:
                    best_d = d
                    best_pos = dp
            if best_pos is not None:
                nearest_dot_dist = best_d
                nearest_dot_dir = get_relative_direction(player, best_pos)

        record = EvidenceRecord(
            step=state["step_count"],
            player_pos=player,
            exit_pos=exit_pos,
            monster_positions=tuple(tuple(m) for m in monsters),
            exit_direction=exit_dir,
            exit_distance=exit_dist,
            nearest_monster_id=nearest_id,
            nearest_monster_direction=nearest_dir,
            nearest_monster_distance=nearest_dist,
            monster_distances=tuple(m_dists),
            available_actions=tuple(avail),
            chosen_action=chosen_action,
            collision_risks=risk_tuples,
            has_safer_alternative=has_safer,
            reasoning=reasoning,
            dots_remaining=dots_remaining,
            dots_collected=dots_collected,
            total_dots=total_dots,
            nearest_dot_direction=nearest_dot_dir,
            nearest_dot_distance=nearest_dot_dist,
            exit_open=exit_open,
        )
        self._history.append(record)
        return record

    def get_latest(self) -> Optional[EvidenceRecord]:
        return self._history[-1] if self._history else None

    def get_recent(self, n: int) -> list[EvidenceRecord]:
        items = list(self._history)
        return items[-n:] if n < len(items) else items

    def get_all(self) -> list[EvidenceRecord]:
        return list(self._history)

    def __len__(self) -> int:
        return len(self._history)
