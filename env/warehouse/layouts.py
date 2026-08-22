"""Immutable warehouse layouts shared by simulation, policy, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapLayout:
    """One fully specified warehouse topology.

    Geometry lives here so environment, observations, explanations, and the
    browser cannot silently drift onto different hard-coded maps.
    """

    layout_id: str
    tiles: tuple[str, ...]
    robot_start_positions: tuple[tuple[int, int], tuple[int, int]]
    charger_position: tuple[int, int]
    task_endpoint_exclusions: tuple[tuple[int, int], ...] = ()
    pickup_endpoint_exclusions: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.tiles or not self.tiles[0]:
            raise ValueError("A warehouse layout cannot be empty.")
        width = len(self.tiles[0])
        if any(len(row) != width for row in self.tiles):
            raise ValueError("Warehouse layout rows must have equal width.")
        if any(symbol not in {".", "#"} for row in self.tiles for symbol in row):
            raise ValueError("Warehouse layouts use only '.' and '#'.")
        required = {
            *self.robot_start_positions,
            self.charger_position,
        }
        if any(not self.is_passable(position) for position in required):
            raise ValueError("Robot starts and the charger must be passable.")
        if len(set(self.robot_start_positions)) != 2:
            raise ValueError("The two robot starts must be distinct.")
        if self.charger_position in self.robot_start_positions:
            raise ValueError("The charger cannot overlap a robot start.")
        if any(
            not self.is_passable(position)
            for position in (
                *self.task_endpoint_exclusions,
                *self.pickup_endpoint_exclusions,
            )
        ):
            raise ValueError("Task endpoint exclusions must be passable cells.")

    @property
    def rows(self) -> int:
        return len(self.tiles)

    @property
    def cols(self) -> int:
        return len(self.tiles[0])

    @property
    def passable_positions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (row, column)
            for row, line in enumerate(self.tiles)
            for column, symbol in enumerate(line)
            if symbol == "."
        )

    @property
    def blocked_positions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (row, column)
            for row, line in enumerate(self.tiles)
            for column, symbol in enumerate(line)
            if symbol == "#"
        )

    @property
    def dead_end_positions(self) -> tuple[tuple[int, int], ...]:
        """Topology-derived side pockets, with no special study semantics."""

        return tuple(
            position
            for position in self.passable_positions
            if sum(
                self.is_passable(
                    (position[0] + row_delta, position[1] + column_delta)
                )
                for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            == 1
        )

    @property
    def four_way_intersections(self) -> tuple[tuple[int, int], ...]:
        """Passable cells connected in all four cardinal directions."""

        return tuple(
            position
            for position in self.passable_positions
            if sum(
                self.is_passable(
                    (position[0] + row_delta, position[1] + column_delta)
                )
                for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            == 4
        )

    def in_bounds(self, position: tuple[int, int]) -> bool:
        row, column = position
        return 0 <= row < self.rows and 0 <= column < self.cols

    def is_passable(self, position: tuple[int, int]) -> bool:
        return self.in_bounds(position) and self.tiles[position[0]][position[1]] == "."

    def is_blocked(self, position: tuple[int, int]) -> bool:
        return self.in_bounds(position) and not self.is_passable(position)


STAGGERED_AISLES_LAYOUT = MapLayout(
    layout_id="warehouse_alternating_shelves_10x11_v6_open_charger_approach",
    tiles=(
        "#####.#####",
        "......#####",
        "#####......",
        "......#####",
        "#####......",
        "......#####",
        "#####......",
        "......#####",
        "####.......",
        "####...####",
    ),
    robot_start_positions=((9, 4), (9, 6)),
    charger_position=(9, 5),
    # Keep the three-cell charger approach free of task endpoints.  These are
    # ordinary aisle cells, not special "yield bays".  Opening (8, 4) gives
    # both robots another route around a robot using the charger; the central
    # cell at (8, 5) is consequently an intentional four-way junction.
    task_endpoint_exclusions=((8, 4), (8, 5), (8, 6)),
    # An A point claims automatically when an empty robot enters it.  The
    # central spine is the only route to the charger and between alternating
    # aisles, so placing A there can force an unrelated charging robot to steal
    # a teammate's task.  Keep pickup points on shelf aisles while delivery B
    # remains free to use ordinary passable cells.
    pickup_endpoint_exclusions=tuple((row, 5) for row in range(10)),
)

CORRIDOR_CHARGER_APRON_LAYOUT = STAGGERED_AISLES_LAYOUT
CORRIDOR_SHELF_LAYOUT = STAGGERED_AISLES_LAYOUT
MAP_LAYOUTS = {
    STAGGERED_AISLES_LAYOUT.layout_id: STAGGERED_AISLES_LAYOUT
}
DEFAULT_MAP_LAYOUT = STAGGERED_AISLES_LAYOUT


def get_map_layout(layout_id: str) -> MapLayout:
    try:
        return MAP_LAYOUTS[str(layout_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown warehouse map layout: {layout_id!r}") from exc
