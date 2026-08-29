from __future__ import annotations

from env.warehouse.layouts import (
    COMPACT_STAGGERED_8X9_LAYOUT,
    COMPACT_STAGGERED_9X9_LAYOUT,
    STUDY_MAP_LAYOUT,
)
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.environment import WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.navigation import shortest_path_distance
from env.warehouse.scenarios import apply_empty_delivery_clearance_scenario
from ui.warehouse_view import warehouse_map_payload


def _assert_connected(layout) -> None:
    origin = layout.robot_start_positions[0]
    assert all(
        shortest_path_distance(origin, position, layout.layout_id)
        < layout.rows * layout.cols
        for position in layout.passable_positions
    )


def test_required_8x9_candidate_is_registered_as_real_topology() -> None:
    layout = COMPACT_STAGGERED_8X9_LAYOUT
    assert STUDY_MAP_LAYOUT is layout
    assert layout.tiles == (
        "####.####",
        ".....####",
        "####.....",
        ".....####",
        "####.....",
        ".....####",
        "###......",
        "###...###",
    )
    assert layout.robot_start_positions == ((7, 3), (7, 5))
    assert layout.charger_position == (7, 4)
    assert layout.robot_exit_positions == ((6, 3), (6, 4), (6, 5))
    assert all(layout.is_passable(position) for position in layout.robot_exit_positions)
    _assert_connected(layout)


def test_selected_9x9_layout_preserves_staggered_aisles_and_three_cell_exit() -> None:
    layout = COMPACT_STAGGERED_9X9_LAYOUT
    assert layout.tiles == (
        "#########",
        "#.......#",
        "##.###.##",
        "#.......#",
        "..#####..",
        "#.......#",
        "##..#..##",
        "###...###",
        "###...###",
    )
    # Each horizontal work aisle is joined to the next one at a different
    # column.  That creates alternate loops without aligned four-way crosses.
    assert tuple(column for column in range(9) if layout.is_passable((2, column))) == (2, 6)
    assert tuple(column for column in range(9) if layout.is_passable((4, column))) == (0, 1, 7, 8)
    assert tuple(column for column in range(9) if layout.is_passable((6, column))) == (2, 3, 5, 6)
    assert layout.robot_exit_positions == ((7, 3), (7, 4), (7, 5))
    assert layout.four_way_intersections == ()
    _assert_connected(layout)

    # A connected graph with more edges than a tree has real bypass cycles.
    edge_count = sum(
        layout.is_passable((row + dr, column + dc))
        for row, column in layout.passable_positions
        for dr, dc in ((1, 0), (0, 1))
    )
    assert edge_count - len(layout.passable_positions) + 1 >= 3


def test_ui_payload_is_derived_from_selected_map_layout() -> None:
    layout = STUDY_MAP_LAYOUT
    payload = warehouse_map_payload(layout)
    assert payload["layout_id"] == layout.layout_id
    assert (payload["rows"], payload["cols"]) == (layout.rows, layout.cols)
    assert tuple(map(tuple, payload["robot_exit_positions"])) == (
        layout.robot_exit_positions
    )
    assert {tuple(position) for position in payload["shelves"]} == set(
        layout.blocked_positions
    )


def test_conservative_9x9_candidate_supports_clearance_curriculum() -> None:
    layout = COMPACT_STAGGERED_9X9_LAYOUT
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(
            rows=layout.rows,
            cols=layout.cols,
            map_layout_id=layout.layout_id,
            horizon=20,
        )
    )
    for variant in range(16):
        environment.reset(seed=31_000 + variant)
        apply_empty_delivery_clearance_scenario(
            environment,
            variant=variant,
        )
        state = environment.get_state()
        actions = stable_coordination_actions(environment)
        _, _, invalid, collision, _, _ = environment._resolve_motion(
            state,
            actions,
        )
        assert not environment.validate_state(state)
        assert not invalid
        assert not collision
        assert tuple(actions.values()).count("WAIT") == 1
