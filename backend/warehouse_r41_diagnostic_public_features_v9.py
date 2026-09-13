"""Public-only relation features for the r4.1 diagnostic v9 program.

The frozen Actor observes 197 public values.  V9 keeps the complete v8
registry and adds deterministic relations which make absolute position,
public map topology, active task endpoints, and intervention-critical context
explicit to an axis-aligned tree.  No scene identifier, physical hash, Actor
parameter, logit, hidden activation, action label, or intervention label is
accepted by this transformer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import deque
from typing import Any

import numpy as np

from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.training.warehouse_native_common import digest


VERSION = "warehouse-r41-diagnostic-public-relations.v9"
ROWS = 6
COLS = 7
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
EPSILON = np.float32(1e-7)


class R41DiagnosticPublicRelationsV9:
    """Expand public observations without accepting any protected evidence."""

    def __init__(self, base_feature_names: Sequence[str]):
        self.v8 = R41DiagnosticPublicRelationsV8(base_feature_names)
        self.base_feature_names = self.v8.base_feature_names
        self._index = {name: index for index, name in enumerate(self.base_feature_names)}
        self._topology_cache: dict[tuple[bytes, int, int, int, int], tuple[float, ...]] = {}
        required = self._required_names()
        missing = sorted(required - set(self.base_feature_names))
        if missing:
            raise ValueError("Public base feature registry is missing: " + ", ".join(missing))
        probe = np.zeros((1, len(self.base_feature_names)), dtype=np.float32)
        _, names = self._append_v9(probe)
        self.v9_derived_feature_names = tuple(names)
        self.derived_feature_names = (
            *self.v8.derived_feature_names,
            *self.v9_derived_feature_names,
        )
        self.feature_names = (*self.base_feature_names, *self.derived_feature_names)
        if len(self.feature_names) != len(set(self.feature_names)):
            raise RuntimeError("Derived public feature names are not unique")

    @staticmethod
    def _required_names() -> set[str]:
        names = set(R41DiagnosticPublicRelationsV8._required_names())
        names.update({
            "role.robot_1", "role.robot_2", "self.row", "self.column",
            "other.row", "other.column", "self.battery", "other.battery",
            "self.carrying", "other.carrying", "other.path_distance",
            "history.valid", "history.self.move_canceled",
            "history.other.move_canceled", "history.joint.consecutive_collision",
        })
        names.update(f"map.{row}.{column}.passable"
                     for row in range(ROWS) for column in range(COLS))
        for who in ("self", "other"):
            for action in (*ACTIONS, "WAIT"):
                names.add(f"history.{who}.submitted.{action}")
            for action in ACTIONS:
                names.add(f"{who}.neighbor.{action}.passable")
            names.add(f"charger.{who}.path_distance")
            for action in ACTIONS:
                names.add(f"charger.{who}.neighbor_distance.{action}")
        for task in range(2):
            names.update((
                f"task.{task}.exists", f"task.{task}.available",
                f"task.{task}.carried_self", f"task.{task}.carried_other",
                f"task.{task}.pickup_row", f"task.{task}.pickup_column",
                f"task.{task}.delivery_row", f"task.{task}.delivery_column",
            ))
            for endpoint in ("pickup", "delivery"):
                for who in ("self", "other"):
                    names.add(f"task.{task}.{endpoint}.{who}.path_distance")
                    for action in ACTIONS:
                        names.add(
                            f"task.{task}.{endpoint}.{who}.neighbor_distance.{action}")
        return names

    def _column(self, values: np.ndarray, name: str) -> np.ndarray:
        return values[:, self._index[name]]

    @staticmethod
    def _coordinate(values: np.ndarray, size: int) -> np.ndarray:
        return np.clip(np.rint(values * np.float32(size - 1)), 0, size - 1).astype(np.int16)

    def _append_v9(self, values: np.ndarray) -> tuple[np.ndarray, list[str]]:
        columns: list[np.ndarray] = []
        names: list[str] = []

        def add(value: Any, name: str) -> None:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 1:
                array = array[:, None]
                local_names = [name]
            elif array.ndim == 2:
                local_names = [f"{name}.{index}" for index in range(array.shape[1])]
            else:
                raise ValueError("Derived public feature has invalid rank")
            if len(array) != len(values):
                raise ValueError("Derived public feature row count differs")
            columns.append(array)
            names.extend(local_names)

        # Recompute the three published critical predicates exactly as v8 does,
        # then expose their complete Boolean interaction.  This avoids asking a
        # shallow tree to rediscover conjunctions while preserving public input.
        critical = self.v8.critical_masks(values)
        bits = np.zeros(len(values), dtype=np.uint8)
        for index, group in enumerate(("narrow_passage", "shared_pickup", "shared_charger")):
            bits |= critical[group].astype(np.uint8) << index
        for value in range(8):
            add(bits == value, f"derived.v9.critical_bits.exact_{value}")

        self_row = self._coordinate(self._column(values, "self.row"), ROWS)
        self_column = self._coordinate(self._column(values, "self.column"), COLS)
        other_row = self._coordinate(self._column(values, "other.row"), ROWS)
        other_column = self._coordinate(self._column(values, "other.column"), COLS)
        positions = np.arange(ROWS * COLS, dtype=np.int16)
        add((self_row * COLS + self_column)[:, None] == positions[None, :],
            "derived.v9.self.position")
        add((other_row * COLS + other_column)[:, None] == positions[None, :],
            "derived.v9.other.position")
        add(self_row == other_row, "derived.v9.agents.same_row")
        add(self_column == other_column, "derived.v9.agents.same_column")
        add(np.abs(self_row - other_row), "derived.v9.agents.row_gap")
        add(np.abs(self_column - other_column), "derived.v9.agents.column_gap")

        grid = np.stack([
            self._column(values, f"map.{row}.{column}.passable")
            for row in range(ROWS) for column in range(COLS)
        ], axis=1).reshape(len(values), ROWS, COLS) > .5
        degree = np.zeros(grid.shape, dtype=np.float32)
        degree[:, 1:, :] += grid[:, :-1, :]
        degree[:, :-1, :] += grid[:, 1:, :]
        degree[:, :, 1:] += grid[:, :, :-1]
        degree[:, :, :-1] += grid[:, :, 1:]
        degree *= grid
        add(grid.reshape(len(values), -1).sum(1) / np.float32(ROWS * COLS),
            "derived.v9.map.passable_fraction")
        for value in range(5):
            add(((degree == value) & grid).reshape(len(values), -1).sum(1)
                / np.float32(ROWS * COLS), f"derived.v9.map.degree_fraction_{value}")
        row_index = np.arange(len(values))
        self_degree = degree[row_index, self_row, self_column]
        other_degree = degree[row_index, other_row, other_column]
        add(self_degree, "derived.v9.self.cell_degree")
        add(other_degree, "derived.v9.other.cell_degree")
        add((self_degree <= 2) & grid[row_index, self_row, self_column],
            "derived.v9.self.cell_narrow")
        add((other_degree <= 2) & grid[row_index, other_row, other_column],
            "derived.v9.other.cell_narrow")

        # Active endpoint coordinates and action improvements.  The selection
        # rule uses only the public carried/available flags and mirrors the
        # already-published v8 active-goal relation.
        active_distance: dict[str, list[np.ndarray]] = {"self": [], "other": []}
        active_improvement: dict[str, list[np.ndarray]] = {"self": [], "other": []}
        active_rows: dict[str, list[np.ndarray]] = {"self": [], "other": []}
        active_columns: dict[str, list[np.ndarray]] = {"self": [], "other": []}
        for task in range(2):
            exists = self._column(values, f"task.{task}.exists") > .5
            available = self._column(values, f"task.{task}.available") > .5
            pickup_row = self._coordinate(
                self._column(values, f"task.{task}.pickup_row"), ROWS)
            pickup_column = self._coordinate(
                self._column(values, f"task.{task}.pickup_column"), COLS)
            delivery_row = self._coordinate(
                self._column(values, f"task.{task}.delivery_row"), ROWS)
            delivery_column = self._coordinate(
                self._column(values, f"task.{task}.delivery_column"), COLS)
            add(pickup_row[:, None] == np.arange(ROWS)[None, :],
                f"derived.v9.task.{task}.pickup.row")
            add(pickup_column[:, None] == np.arange(COLS)[None, :],
                f"derived.v9.task.{task}.pickup.column")
            add(delivery_row[:, None] == np.arange(ROWS)[None, :],
                f"derived.v9.task.{task}.delivery.row")
            add(delivery_column[:, None] == np.arange(COLS)[None, :],
                f"derived.v9.task.{task}.delivery.column")
            for endpoint, endpoint_row, endpoint_column in (
                ("pickup", pickup_row, pickup_column),
                ("delivery", delivery_row, delivery_column),
            ):
                endpoint_degree = degree[row_index, endpoint_row, endpoint_column]
                add(endpoint_degree, f"derived.v9.task.{task}.{endpoint}.cell_degree")
                add((endpoint_degree <= 2) & exists,
                    f"derived.v9.task.{task}.{endpoint}.cell_narrow")
            for who in ("self", "other"):
                carried = self._column(values, f"task.{task}.carried_{who}") > .5
                row = np.where(carried, delivery_row, pickup_row)
                column = np.where(carried, delivery_column, pickup_column)
                valid = exists & (carried | available)
                current = np.where(
                    carried,
                    self._column(values, f"task.{task}.delivery.{who}.path_distance"),
                    np.where(
                        available,
                        self._column(values, f"task.{task}.pickup.{who}.path_distance"),
                        np.float32(2.0),
                    ),
                )
                neighbor = np.stack([
                    np.where(
                        carried,
                        self._column(values, f"task.{task}.delivery.{who}.neighbor_distance.{action}"),
                        np.where(
                            available,
                            self._column(values, f"task.{task}.pickup.{who}.neighbor_distance.{action}"),
                            np.float32(2.0),
                        ),
                    ) for action in ACTIONS
                ], axis=1)
                improvement = current[:, None] - neighbor
                active_distance[who].append(current)
                active_improvement[who].append(improvement)
                active_rows[who].append(np.where(valid, row, -1))
                active_columns[who].append(np.where(valid, column, -1))
                add(valid, f"derived.v9.task.{task}.{who}.active")
                add(valid & (row == (self_row if who == "self" else other_row)),
                    f"derived.v9.task.{task}.{who}.goal_same_row")
                add(valid & (column == (self_column if who == "self" else other_column)),
                    f"derived.v9.task.{task}.{who}.goal_same_column")
                add(improvement, f"derived.v9.task.{task}.{who}.active_improvement")

        for who in ("self", "other"):
            distances = np.stack(active_distance[who], axis=1)
            improvements = np.stack(active_improvement[who], axis=1)
            nearest = np.argmin(distances, axis=1)
            no_goal = np.all(distances >= np.float32(1.5), axis=1)
            add((nearest == 0) & ~no_goal, f"derived.v9.{who}.nearest_task_0")
            add((nearest == 1) & ~no_goal, f"derived.v9.{who}.nearest_task_1")
            add(no_goal, f"derived.v9.{who}.no_active_goal")
            selected_rows = np.choose(nearest, active_rows[who])
            selected_columns = np.choose(nearest, active_columns[who])
            own_row = self_row if who == "self" else other_row
            own_column = self_column if who == "self" else other_column
            add(np.where(no_goal, 0, selected_rows - own_row),
                f"derived.v9.{who}.nearest_goal.row_delta")
            add(np.where(no_goal, 0, selected_columns - own_column),
                f"derived.v9.{who}.nearest_goal.column_delta")
            chosen = improvements[row_index, nearest]
            add(chosen, f"derived.v9.{who}.nearest_goal.improvement")

        topology = np.empty((len(values), 9), dtype=np.float32)
        task_coordinates = []
        for task in range(2):
            task_coordinates.append((
                self._coordinate(self._column(values, f"task.{task}.pickup_row"), ROWS),
                self._coordinate(self._column(values, f"task.{task}.pickup_column"), COLS),
                self._coordinate(self._column(values, f"task.{task}.delivery_row"), ROWS),
                self._coordinate(self._column(values, f"task.{task}.delivery_column"), COLS),
            ))
        packed_maps = np.packbits(grid.reshape(len(values), -1), axis=1, bitorder="little")
        for index in range(len(values)):
            p0 = int(task_coordinates[0][0][index]) * COLS + int(task_coordinates[0][1][index])
            d0 = int(task_coordinates[0][2][index]) * COLS + int(task_coordinates[0][3][index])
            p1 = int(task_coordinates[1][0][index]) * COLS + int(task_coordinates[1][1][index])
            d1 = int(task_coordinates[1][2][index]) * COLS + int(task_coordinates[1][3][index])
            key = (packed_maps[index].tobytes(), p0, d0, p1, d1)
            cached = self._topology_cache.get(key)
            if cached is None:
                cached = self._task_pair_topology(grid[index], p0, d0, p1, d1)
                self._topology_cache[key] = cached
            topology[index] = cached
        topology_names = (
            "shared_edge_ratio_min", "shared_edge_ratio_max",
            "shared_bridge_fraction", "shared_crossing_fraction",
            "same_direction_fraction", "opposing_direction_fraction",
            "task_0_route_fraction", "task_1_route_fraction",
            "shared_crossing_at_4_3",
        )
        for index, name in enumerate(topology_names):
            add(topology[:, index], "derived.v9.task_pair." + name)
        ratio = topology[:, 0]
        for index, selected in enumerate((
            ratio < .37,
            (ratio >= .37) & (ratio < .42),
            (ratio >= .42) & (ratio < .49),
            ratio >= .49,
        )):
            add(selected, f"derived.v9.task_pair.overlap_band_{index}")
        add(topology[:, 2] * np.float32(ROWS * COLS - 1) >= 2,
            "derived.v9.task_pair.multiple_shared_bridges")
        add(topology[:, 3] > 0,
            "derived.v9.task_pair.has_shared_crossing")

        # Collision recovery and action-history interactions are public.  They
        # are especially useful for the ordinary WAIT endpoint of an isolated
        # counterfactual pair, which v8's shallow specialists underfit.
        collision = (
            (self._column(values, "history.self.move_canceled") > .5)
            | (self._column(values, "history.other.move_canceled") > .5)
            | (self._column(values, "history.joint.consecutive_collision") > 0)
        )
        add(collision, "derived.v9.history.any_collision")
        add(collision & critical["narrow_passage"],
            "derived.v9.history.collision_in_narrow")
        for who in ("self", "other"):
            for action in (*ACTIONS, "WAIT"):
                submitted = self._column(values, f"history.{who}.submitted.{action}") > .5
                add(submitted & collision,
                    f"derived.v9.history.{who}.{action}.collision")
                for bit in range(8):
                    add(submitted & (bits == bit),
                        f"derived.v9.history.{who}.{action}.critical_bits_{bit}")

        result = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Derived public features are non-finite")
        return result, names

    @staticmethod
    def _task_pair_topology(
        grid: np.ndarray, pickup0: int, delivery0: int,
        pickup1: int, delivery1: int,
    ) -> tuple[float, ...]:
        """Return public graph relations for the two visible task routes."""
        cells = tuple((row, column) for row in range(ROWS) for column in range(COLS)
                      if bool(grid[row, column]))
        passable = set(cells)

        def point(value: int) -> tuple[int, int]:
            return divmod(value, COLS)

        def neighbors(node: tuple[int, int]) -> tuple[tuple[int, int], ...]:
            row, column = node
            return tuple(sorted(candidate for candidate in (
                (row - 1, column), (row + 1, column),
                (row, column - 1), (row, column + 1),
            ) if candidate in passable))

        def paths(start: tuple[int, int], goal: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
            if start not in passable or goal not in passable:
                return ()
            queue = deque((start,))
            distances = {start: 0}
            while queue:
                current = queue.popleft()
                for target in neighbors(current):
                    if target not in distances:
                        distances[target] = distances[current] + 1
                        queue.append(target)
            if goal not in distances:
                return ()
            result: list[tuple[tuple[int, int], ...]] = []

            def visit(current: tuple[int, int], route: tuple[tuple[int, int], ...]) -> None:
                if current == goal:
                    result.append(route)
                    return
                for target in neighbors(current):
                    if distances.get(target) == distances[current] + 1 \
                            and distances[target] <= distances[goal]:
                        visit(target, route + (target,))
            visit(start, (start,))
            return tuple(result)

        def undirected(route: Sequence[tuple[int, int]]) -> frozenset[frozenset[tuple[int, int]]]:
            return frozenset(frozenset(edge) for edge in zip(route, route[1:]))

        def directed(route: Sequence[tuple[int, int]]) -> frozenset[tuple[tuple[int, int], tuple[int, int]]]:
            return frozenset(zip(route, route[1:]))

        def bridges() -> frozenset[frozenset[tuple[int, int]]]:
            timer = 0
            seen: set[tuple[int, int]] = set()
            tin: dict[tuple[int, int], int] = {}
            low: dict[tuple[int, int], int] = {}
            result: set[frozenset[tuple[int, int]]] = set()

            def search(node: tuple[int, int], parent: tuple[int, int] | None) -> None:
                nonlocal timer
                seen.add(node)
                tin[node] = low[node] = timer
                timer += 1
                for target in neighbors(node):
                    if target == parent:
                        continue
                    if target in seen:
                        low[node] = min(low[node], tin[target])
                    else:
                        search(target, node)
                        low[node] = min(low[node], low[target])
                        if low[target] > tin[node]:
                            result.add(frozenset((node, target)))
            for node in cells:
                if node not in seen:
                    search(node, None)
            return frozenset(result)

        endpoints = ((point(pickup0), point(delivery0)),
                     (point(pickup1), point(delivery1)))
        routes = tuple(paths(*pair) for pair in endpoints)
        if any(not item for item in routes):
            return (0.0,) * 9
        edge_sets = tuple(tuple(undirected(route) for route in item) for item in routes)
        ratios: list[float] = []
        shared_edges: set[frozenset[tuple[int, int]]] = set()
        shared_nodes: set[tuple[int, int]] = set()
        for left_index, left in enumerate(routes[0]):
            for right_index, right in enumerate(routes[1]):
                shared = edge_sets[0][left_index] & edge_sets[1][right_index]
                ratios.append(len(shared) / max(1, min(
                    len(edge_sets[0][left_index]), len(edge_sets[1][right_index]))))
                shared_edges.update(shared)
                shared_nodes.update(set(left) & set(right))
        graph_bridges = bridges()
        crossings = {node for node in shared_nodes if len(neighbors(node)) >= 3}

        # Current public task routes, independent of robot assignment, expose
        # same/opposing aisle demand directly.
        same = opposing = 0
        for left in routes[0]:
            left_edges = directed(left)
            for right in routes[1]:
                right_edges = directed(right)
                same = max(same, len(left_edges & right_edges))
                opposing = max(opposing, len(
                    left_edges & frozenset((after, before) for before, after in right_edges)))
        scale = float(max(1, ROWS * COLS - 1))
        return (
            float(min(ratios)), float(max(ratios)),
            len(shared_edges & graph_bridges) / scale,
            len(crossings) / scale,
            same / scale, opposing / scale,
            (len(routes[0][0]) - 1) / scale,
            (len(routes[1][0]) - 1) / scale,
            float((4, 3) in crossings),
        )

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != len(self.base_feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Public observation batch differs")
        v8_values = self.v8.transform_batch(values)
        extra, names = self._append_v9(values)
        if tuple(names) != self.v9_derived_feature_names:
            raise RuntimeError("Derived public feature registry changed")
        return np.concatenate((v8_values, extra), axis=1).astype(np.float32, copy=False)

    def transform_mapping(self, observation: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(observation, Mapping):
            raise ValueError("Public observation must be a mapping")
        try:
            vector = np.asarray([
                float(observation[name]) for name in self.base_feature_names
            ], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every public base feature is required") from exc
        if vector.shape != (len(self.base_feature_names),) or not np.isfinite(vector).all():
            raise ValueError("Public observation vector differs")
        expanded = self.transform_batch(vector[None, :])[0]
        return {name: float(value) for name, value in zip(self.feature_names, expanded)}

    def critical_masks(self, observations: np.ndarray) -> dict[str, np.ndarray]:
        return self.v8.critical_masks(observations)

    def contract(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "base_feature_names": list(self.base_feature_names),
            "derived_feature_names": list(self.derived_feature_names),
            "v8_contract_sha256": digest(self.v8.contract()),
            "map_shape": [ROWS, COLS],
            "prediction_inputs": ["public_observation"],
            "scene_identifier_input": False,
            "physical_hash_input": False,
            "actor_parameters_input": False,
            "actor_logits_input": False,
            "actor_hidden_state_input": False,
            "intervention_metadata_input": False,
            "action_label_input": False,
        }


__all__ = ["VERSION", "R41DiagnosticPublicRelationsV9"]
