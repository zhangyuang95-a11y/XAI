"""Pure map navigation and static action legality rules."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from .domain import AgentState, WarehouseState
from .layouts import DEFAULT_MAP_LAYOUT, get_map_layout


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
MOVE_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
ROWS = DEFAULT_MAP_LAYOUT.rows
COLS = DEFAULT_MAP_LAYOUT.cols
CHARGER_POSITION = DEFAULT_MAP_LAYOUT.charger_position
WAITING_POSITIONS = DEFAULT_MAP_LAYOUT.robot_start_positions
WAITING_ZONE = tuple(
    position
    for position in DEFAULT_MAP_LAYOUT.passable_positions
    if position[0] == ROWS - 1 and position != CHARGER_POSITION
)
SHELF_POSITIONS = DEFAULT_MAP_LAYOUT.blocked_positions

# Compatibility views for older analysis code.  Runtime geometry always uses
# ``MapLayout.blocked_positions`` rather than their Cartesian product.
SHELF_ROWS = tuple(sorted({row for row, _ in SHELF_POSITIONS}))
SHELF_COLUMNS = tuple(sorted({column for _, column in SHELF_POSITIONS}))


def assigned_waiting_positions(
    num_agents: int = 2,
) -> tuple[tuple[int, int], ...]:
    if num_agents != 2:
        raise ValueError("The collaborative environment has exactly two robots.")
    return WAITING_POSITIONS


def in_bounds(
    position: tuple[int, int],
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> bool:
    return get_map_layout(layout_id).in_bounds(position)


def is_shelf(
    position: tuple[int, int],
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> bool:
    return get_map_layout(layout_id).is_blocked(position)


def is_passable(
    position: tuple[int, int],
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> bool:
    return get_map_layout(layout_id).is_passable(position)


def all_passable_positions(
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> tuple[tuple[int, int], ...]:
    return get_map_layout(layout_id).passable_positions


def pickup_pairs(
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    layout = get_map_layout(layout_id)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for shelf in layout.blocked_positions:
        row, column = shelf
        for access in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if layout.is_passable(access):
                pairs.append((shelf, access))
    return tuple(dict.fromkeys(pairs))


@lru_cache(maxsize=None)
def shortest_path_distance(
    start: tuple[int, int],
    goal: tuple[int, int],
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> int:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        for delta in MOVE_DELTAS.values():
            candidate = (position[0] + delta[0], position[1] + delta[1])
            if candidate in visited or not is_passable(candidate, layout_id):
                continue
            if candidate == goal:
                return distance + 1
            visited.add(candidate)
            queue.append((candidate, distance + 1))
    layout = get_map_layout(layout_id)
    return layout.rows * layout.cols


def legal_action_mask(
    state: WarehouseState,
    agent: AgentState,
    layout_id: str = DEFAULT_MAP_LAYOUT.layout_id,
) -> tuple[float, ...]:
    """Return the static action-space mask for one robot.

    A teammate's current cell is deliberately *not* masked here.  Both robots
    choose simultaneously, so that cell may be vacated in the same joint step.
    Same-target, swap, and stationary-occupant conflicts are resolved only by
    :meth:`WarehouseMultiAgentEnv._resolve_motion`; the policy must be able to
    propose (and learn) coordinated hand-offs without a pre-action occupancy
    rule changing its action space.
    """

    if not agent.active:
        return tuple(float(action == "WAIT") for action in ACTIONS)
    mask: list[float] = []
    for action in ACTIONS:
        if action == "WAIT":
            mask.append(1.0)
            continue
        delta = MOVE_DELTAS[action]
        target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
        mask.append(float(is_passable(target, layout_id)))
    return tuple(mask)
