"""Offline-only, memoryless Warehouse BC labels from the v6 public input.

This teacher is never imported by the frozen Actor or runtime adapter. Its
entire input is the same pre-action observation and physical action mask given
to the two learned Actors; old hidden goal/route history cannot change a label.
"""
from __future__ import annotations

import numpy as np

from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse.public_observation_v6 import (
    OBSERVATION_DIM, OBSERVATION_FEATURE_NAMES,
)


_INDEX = {name: index for index, name in enumerate(OBSERVATION_FEATURE_NAMES)}
_CHARGER = STUDY_MAP_LAYOUT.charger_position


def _position(row: np.ndarray, prefix: str) -> tuple[int, int]:
    return (int(round(float(row[_INDEX[f"{prefix}_row"]]) * (STUDY_MAP_LAYOUT.rows - 1))),
            int(round(float(row[_INDEX[f"{prefix}_column"]]) * (STUDY_MAP_LAYOUT.cols - 1))))


def _distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    return shortest_path_distance(start, end, STUDY_MAP_LAYOUT.layout_id)


def _order(row: np.ndarray, index: int):
    prefix = f"order_{index}"
    return {
        "pickup": _position(row, prefix + "_pickup"),
        "delivery": _position(row, prefix + "_delivery"),
        "available": row[_INDEX[prefix + "_available"]] > 0.5,
    }


def _destination(start: tuple[int, int], action: str) -> tuple[int, int]:
    if action == "WAIT":
        return start
    move = MOVE_DELTAS[action]
    return start[0] + move[0], start[1] + move[1]


def public_bc_teacher(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return two legal offline BC labels, reading no environment state."""
    data, legal = np.asarray(obs, dtype=np.float32), np.asarray(masks, dtype=np.float32)
    if (data.shape != (2, OBSERVATION_DIM) or legal.shape != (2, len(ACTIONS))
            or not np.isfinite(data).all() or not np.isfinite(legal).all()
            or np.any(legal.sum(-1) < 1.0)):
        raise ValueError("invalid public Warehouse BC input")
    positions = [_position(data[index], "self") for index in (0, 1)]
    batteries = [float(data[index, _INDEX["self_battery_fraction"]] * 100.0)
                 for index in (0, 1)]
    orders = [_order(data[0], index) for index in (0, 1)]
    carried = [next((task for task in (0, 1)
                     if data[index, _INDEX[f"self_carrying_task_{task}"]] > 0.5), None)
               for index in (0, 1)]
    goals: list[tuple[int, int] | None] = [None, None]
    assigned: set[int] = set()
    for role in (0, 1):
        if carried[role] is not None:
            task = int(carried[role])
            goals[role] = orders[task]["delivery"]
            assigned.add(task)
    free = [role for role in (0, 1) if carried[role] is None]
    available = [task for task in (0, 1)
                 if orders[task]["available"] and task not in assigned]
    if len(free) == 2 and len(available) == 2:
        direct = sum(_distance(positions[role], orders[available[role]]["pickup"])
                     for role in (0, 1))
        swapped = sum(_distance(positions[role], orders[available[1 - role]]["pickup"])
                      for role in (0, 1))
        assignment = available if direct <= swapped else available[::-1]
        for role, task in zip(free, assignment, strict=True):
            goals[role] = orders[task]["pickup"]
    else:
        for role in free:
            if available:
                task = min(available,
                           key=lambda item: (_distance(positions[role], orders[item]["pickup"]), item))
                goals[role] = orders[task]["pickup"]
                available.remove(task)
    wait_to_charge = [False, False]
    for role in (0, 1):
        charger_distance = _distance(positions[role], _CHARGER)
        capacity = batteries[role] / 2.0
        mission_margin = min(
            (float(data[role, _INDEX[f"order_{task}_self_charger_return_margin"]]) * 50.0
             for task in (0, 1) if carried[role] == task or
             (carried[role] is None and orders[task]["available"])),
            default=capacity - charger_distance,
        )
        need_charge = capacity <= charger_distance + 4 or mission_margin <= 4
        if positions[role] == _CHARGER and batteries[role] < 80:
            need_charge = True
        if need_charge:
            goals[role] = _CHARGER
            wait_to_charge[role] = positions[role] == _CHARGER and batteries[role] < 80
    candidates = [[index for index, allowed in enumerate(legal[role]) if allowed > 0.5]
                  for role in (0, 1)]
    best_pair, best_score = (len(ACTIONS) - 1, len(ACTIONS) - 1), -float("inf")
    for first in candidates[0]:
        for second in candidates[1]:
            indices = (first, second)
            actions = (ACTIONS[first], ACTIONS[second])
            destinations = [_destination(positions[role], actions[role]) for role in (0, 1)]
            conflict = (destinations[0] == destinations[1]
                        or (destinations[0] == positions[1]
                            and destinations[1] == positions[0]))
            score = -50.0 * float(conflict)
            for role in (0, 1):
                goal = goals[role]
                if wait_to_charge[role]:
                    score += 6.0 if actions[role] == "WAIT" else -6.0
                elif goal is None:
                    score += 1.0 if actions[role] == "WAIT" else -0.1
                else:
                    before, after = _distance(positions[role], goal), _distance(destinations[role], goal)
                    score += 3.0 * (before - after)
                    if before == 0 and actions[role] == "WAIT":
                        score += 1.0
                if actions[role] != "WAIT":
                    score -= 0.01
            if score > best_score or (score == best_score and indices < best_pair):
                best_pair, best_score = indices, score
    return np.asarray(best_pair, dtype=np.int64)
