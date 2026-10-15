"""Development prototype: public-state, multi-step joint BC teacher.

This module is intentionally not imported by the candidate-02 training
closure. It explores whether a coupled path plan avoids one-step corridor
deadlocks before a new candidate is registered.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse.public_observation_v6 import OBSERVATION_DIM
from .warehouse_v6_teacher import (
    _CHARGER, _INDEX, _destination, _distance, _order, _position,
    public_bc_teacher,
)


_CELLS = tuple((row, col) for row in range(STUDY_MAP_LAYOUT.rows)
               for col in range(STUDY_MAP_LAYOUT.cols)
               if (row, col) not in STUDY_MAP_LAYOUT.blocked_positions)
_CELL_SET = frozenset(_CELLS)
_STATES = tuple((first, second) for first in _CELLS for second in _CELLS
                if first != second)
_REVERSE: dict[tuple, list[tuple]] = {state: [] for state in _STATES}
_EDGES: dict[tuple, list[tuple[tuple[int, int], tuple]]] = {}
for _state in _STATES:
    _options = []
    for _a in range(len(ACTIONS)):
        _p1 = _destination(_state[0], ACTIONS[_a])
        if _p1 not in _CELL_SET:
            continue
        for _b in range(len(ACTIONS)):
            _p2 = _destination(_state[1], ACTIONS[_b])
            if (_p2 not in _CELL_SET or _p1 == _p2
                    or (_p1 == _state[1] and _p2 == _state[0])):
                continue
            _next = (_p1, _p2)
            _options.append(((_a, _b), _next))
            _REVERSE[_next].append(_state)
    _EDGES[_state] = _options


@lru_cache(maxsize=512)
def _distance_to_pair(goals: tuple) -> dict[tuple, int]:
    if goals not in _REVERSE:
        return {}
    distances = {goals: 0}
    queue = deque([goals])
    while queue:
        state = queue.popleft()
        next_cost = distances[state] + 1
        for predecessor in _REVERSE[state]:
            if predecessor not in distances:
                distances[predecessor] = next_cost
                queue.append(predecessor)
    return distances


def _distinct_goals(positions: tuple, goals: list, batteries: list,
                    carried: list) -> tuple:
    target = tuple(goal if goal is not None else positions[role]
                   for role, goal in enumerate(goals))
    if target[0] != target[1]:
        return target
    common = target[0]
    priority = (0 if carried[0] is not None and carried[1] is None else
                1 if carried[1] is not None and carried[0] is None else
                0 if batteries[0] <= batteries[1] else 1)
    secondary = 1 - priority
    best = None
    for staging in _CELLS:
        if staging == common:
            continue
        candidate = [None, None]
        candidate[priority] = common
        candidate[secondary] = staging
        pair = tuple(candidate)
        distance = _distance_to_pair(pair).get(positions)
        if distance is None:
            continue
        score = (distance, _distance(staging, common),
                 _distance(positions[secondary], staging), staging)
        if best is None or score < best[0]:
            best = score, pair
    return best[1] if best is not None else target


def public_lookahead_teacher(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Choose a legal first move of a shortest collision-free joint route."""
    data, legal = np.asarray(obs, dtype=np.float32), np.asarray(masks, dtype=np.float32)
    if (data.shape != (2, OBSERVATION_DIM) or legal.shape != (2, len(ACTIONS))
            or not np.isfinite(data).all() or not np.isfinite(legal).all()):
        raise ValueError("invalid public Warehouse BC input")
    positions = tuple(_position(data[index], "self") for index in (0, 1))
    batteries = [float(data[index, _INDEX["self_battery_fraction"]] * 100.0)
                 for index in (0, 1)]
    orders = [_order(data[0], index) for index in (0, 1)]
    carried = [next((task for task in (0, 1)
                     if data[index, _INDEX[f"self_carrying_task_{task}"]] > 0.5), None)
               for index in (0, 1)]
    goals = [orders[carried[role]]["delivery"] if carried[role] is not None else None
             for role in (0, 1)]
    free = [role for role in (0, 1) if carried[role] is None]
    available = [task for task in (0, 1) if orders[task]["available"]]
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
    forced_wait = [False, False]
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
            forced_wait[role] = positions[role] == _CHARGER and batteries[role] < 80
        elif goals[role] is not None and positions[role] == goals[role]:
            forced_wait[role] = True
    target = _distinct_goals(positions, goals, batteries, carried)
    distances = _distance_to_pair(target)
    best = None
    for (first, second), next_state in _EDGES.get(positions, []):
        if (legal[0, first] <= 0.5 or legal[1, second] <= 0.5
                or forced_wait[0] and ACTIONS[first] != "WAIT"
                or forced_wait[1] and ACTIONS[second] != "WAIT"):
            continue
        remaining = distances.get(next_state)
        if remaining is None:
            continue
        progress = sum(_distance(next_state[role], target[role])
                       for role in (0, 1))
        score = (remaining, progress, int(ACTIONS[first] == "WAIT") +
                 int(ACTIONS[second] == "WAIT"), first, second)
        if best is None or score < best[0]:
            best = (score, (first, second))
    if best is None:
        return public_bc_teacher(data, legal)
    return np.asarray(best[1], dtype=np.int64)
