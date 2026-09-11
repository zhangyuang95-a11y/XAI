"""Fail-closed high-conflict task dynamics for warehouse r4.1.

The r4.1 environment keeps the native warehouse physics intact while replacing
only task creation.  Every pair of active jobs is an edge in one immutable
conflict graph.  Replacement jobs are sampled from the graph's safe two-core;
there is deliberately no call to the ordinary warehouse task sampler.

The mixin is usable with both :class:`NativeWarehouseEnv` and the observed197
online environment.  It predicts the exact delivery set from the same
pre-action state and native motion resolver, then supplies that explicit
remaining-task context while the native transition creates replacements.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import permutations
import json
import math
from typing import Any, Mapping, Sequence

from env.warehouse.domain import DeliveryTask, collaborative_study_config
from env.warehouse.navigation import (
    MOVE_DELTAS,
    all_passable_positions,
    pickup_pairs,
    shortest_path_distance,
)
from env.warehouse_native.environment import NativeWarehouseEnv


CONTRACT_VERSION = "warehouse-r41-active-task-conflict.v1"
SNAPSHOT_VERSION = "warehouse-r41-conflict-snapshot.v1"
SAMPLER_VERSION = "warehouse-r41-conditional-core-sampler.v1"
CONTRACT = {
    "version": CONTRACT_VERSION,
    "minimum_shared_edge_ratio": 0.30,
    "maximum_shared_edge_ratio": 0.55,
    "all_shortest_route_pairs_must_be_in_band": True,
    "require_shared_single_cell_bottleneck_or_crossing": True,
    "require_opposing_and_merge_opportunities": True,
    "merge_definition": (
        "the two minimum-work robot missions contain at least one shared "
        "directed edge, so they can merge into the same aisle flow"
    ),
    "active_task_count": 2,
    "replacement_policy": "sample only a compatible safe-two-core node; never fall back",
    "simultaneous_replacement_policy": (
        "sample a safe-two-core node with an available compatible successor, "
        "then sample that successor"
    ),
    "replacement_pickup_occupancy": (
        "prefer a pickup not occupied after motion; if every strict conflict "
        "successor pickup is occupied, keep the strict successor and expose it "
        "for pickup no earlier than the next confirmed step"
    ),
    "replacement_delivery_occupancy": (
        "a new delivery endpoint on a robot is recorded but does not claim or "
        "deliver a task in the creation frame"
    ),
    "immediate_recreation": "a job delivered in this frame cannot be recreated in the same frame",
    "creation_frame_claim_semantics": (
        "task creation follows pickup resolution, so a new pickup under a robot "
        "cannot be claimed until a later confirmed step"
    ),
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


CONTRACT_SHA256 = digest(CONTRACT)


class R41ConflictSamplingError(RuntimeError):
    """Raised instead of silently using an ordinary low-conflict task."""


def _position(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2 or any(type(part) is not int for part in value):
        raise ValueError("Task endpoints must be integer grid coordinates")
    return int(value[0]), int(value[1])


def task_endpoints(task: DeliveryTask | Mapping[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(task, Mapping):
        return _position(task["pickup_position"]), _position(task["delivery_position"])
    if isinstance(task, (tuple, list)) and len(task) == 2:
        return _position(task[0]), _position(task[1])
    return tuple(task.pickup_position), tuple(task.delivery_position)


def task_node_id(task: DeliveryTask | Mapping[str, Any]) -> str:
    pickup, delivery = task_endpoints(task)
    return "job_" + digest({"pickup": pickup, "delivery": delivery})[:16]


def _neighbors(layout, position: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (position[0] + delta[0], position[1] + delta[1])
        for delta in MOVE_DELTAS.values()
        if layout.is_passable((position[0] + delta[0], position[1] + delta[1]))
    )


def all_shortest_paths(
    layout, start: tuple[int, int], goal: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Enumerate all shortest paths in a stable coordinate order."""
    queue = deque((start,))
    distances = {start: 0}
    while queue:
        current = queue.popleft()
        for target in sorted(_neighbors(layout, current)):
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
        for target in sorted(_neighbors(layout, current)):
            if distances.get(target) == distances[current] + 1 and distances[target] <= distances[goal]:
                visit(target, route + (target,))

    visit(start, (start,))
    return tuple(result)


def _directed_edges(route: Sequence[tuple[int, int]]) -> frozenset[tuple[tuple[int, int], tuple[int, int]]]:
    return frozenset(zip(route, route[1:]))


def _undirected_edges(route: Sequence[tuple[int, int]]) -> frozenset[frozenset[tuple[int, int]]]:
    return frozenset(frozenset(edge) for edge in zip(route, route[1:]))


def _graph_bridges(layout) -> frozenset[frozenset[tuple[int, int]]]:
    timer = 0
    seen: set[tuple[int, int]] = set()
    tin: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    bridges: set[frozenset[tuple[int, int]]] = set()

    def search(node: tuple[int, int], parent: tuple[int, int] | None) -> None:
        nonlocal timer
        seen.add(node)
        tin[node] = low[node] = timer
        timer += 1
        for target in _neighbors(layout, node):
            if target == parent:
                continue
            if target in seen:
                low[node] = min(low[node], tin[target])
                continue
            search(target, node)
            low[node] = min(low[node], low[target])
            if low[target] > tin[node]:
                bridges.add(frozenset((node, target)))

    for node in layout.passable_positions:
        if node not in seen:
            search(node, None)
    return frozenset(bridges)


def _mission_routes(layout, start: tuple[int, int], endpoints) -> tuple[tuple[tuple[int, int], ...], ...]:
    pickup, delivery = task_endpoints(endpoints)
    routes = []
    for prefix in all_shortest_paths(layout, start, pickup):
        for suffix in all_shortest_paths(layout, pickup, delivery):
            routes.append(prefix + suffix[1:])
    return tuple(routes)


def conflict_metrics(left, right, *, config=None) -> dict[str, Any]:
    """Return the immutable public-geometry conflict contract for two jobs."""
    config = config or collaborative_study_config()
    probe = NativeWarehouseEnv(config)
    layout = probe.layout
    endpoints = (task_endpoints(left), task_endpoints(right))
    if len({position for task in endpoints for position in task}) != 4:
        return {
            "passed": False,
            "checks": {
                "distinct_endpoints": False,
                "route_overlap_band": False,
                "shared_bottleneck_or_crossing": False,
                "opposing_and_merge_opportunities": False,
            },
            "task_node_ids": [task_node_id(left), task_node_id(right)],
        }
    task_routes = [all_shortest_paths(layout, pickup, delivery) for pickup, delivery in endpoints]
    if any(not routes for routes in task_routes):
        raise ValueError("A conflict task is unreachable")
    ratios: list[float] = []
    route_pairs: list[tuple[Any, Any, frozenset[frozenset[tuple[int, int]]]]] = []
    for left_route in task_routes[0]:
        left_edges = _undirected_edges(left_route)
        for right_route in task_routes[1]:
            right_edges = _undirected_edges(right_route)
            shared = left_edges & right_edges
            ratios.append(len(shared) / max(1, min(len(left_edges), len(right_edges))))
            route_pairs.append((left_route, right_route, shared))

    bridges = _graph_bridges(layout)
    shared_bridge_edges: set[frozenset[tuple[int, int]]] = set()
    shared_crossings: set[tuple[int, int]] = set()
    for left_route, right_route, shared in route_pairs:
        shared_bridge_edges.update(shared & bridges)
        shared_crossings.update(
            node
            for node in set(left_route) & set(right_route)
            if len(_neighbors(layout, node)) >= 3
        )

    best_witness: dict[str, Any] | None = None
    minimum_work = math.inf
    for assignment in permutations(range(2)):
        candidates = [
            _mission_routes(layout, layout.robot_start_positions[role], endpoints[assignment[role]])
            for role in range(2)
        ]
        work = sum(min(len(route) - 1 for route in routes) for routes in candidates)
        if work < minimum_work:
            minimum_work, best_witness = work, None
        if work != minimum_work:
            continue
        for first_route in candidates[0]:
            first_edges = _directed_edges(first_route)
            for second_route in candidates[1]:
                second_edges = _directed_edges(second_route)
                same = first_edges & second_edges
                opposing = first_edges & frozenset((after, before) for before, after in second_edges)
                witness = {
                    "assignment": list(assignment),
                    "same_direction_edges": len(same),
                    "opposing_edges": len(opposing),
                    "routes": [list(first_route), list(second_route)],
                }
                witness_rank = (
                    witness["same_direction_edges"] > 0 and witness["opposing_edges"] > 0,
                    witness["same_direction_edges"] + witness["opposing_edges"],
                )
                current_rank = (-1, -1) if best_witness is None else (
                    best_witness["same_direction_edges"] > 0 and best_witness["opposing_edges"] > 0,
                    best_witness["same_direction_edges"] + best_witness["opposing_edges"],
                )
                if witness_rank > current_rank:
                    best_witness = witness
    if best_witness is None:
        raise ValueError("No two-robot mission route exists")

    checks = {
        "distinct_endpoints": True,
        "route_overlap_band": (
            min(ratios) >= CONTRACT["minimum_shared_edge_ratio"]
            and max(ratios) <= CONTRACT["maximum_shared_edge_ratio"]
        ),
        "shared_bottleneck_or_crossing": bool(shared_bridge_edges or shared_crossings),
        "opposing_and_merge_opportunities": (
            best_witness["same_direction_edges"] > 0 and best_witness["opposing_edges"] > 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_node_ids": [task_node_id(left), task_node_id(right)],
        "shared_edge_ratio_min": min(ratios),
        "shared_edge_ratio_max": max(ratios),
        "shared_bridge_edges": [
            sorted(edge) for edge in sorted(shared_bridge_edges, key=lambda value: sorted(value))
        ],
        "shared_crossings": sorted(shared_crossings),
        "same_direction_mission_edges": best_witness["same_direction_edges"],
        "opposing_mission_edges": best_witness["opposing_edges"],
        "task_route_steps": [min(len(route) - 1 for route in routes) for routes in task_routes],
        "initial_joint_work_steps": int(minimum_work),
        "witness": best_witness,
    }


def _eligible_jobs(config=None) -> tuple[dict[str, Any], ...]:
    config = config or collaborative_study_config()
    env = NativeWarehouseEnv(config)
    layout = env.layout
    excluded = {
        layout.charger_position,
        *layout.robot_start_positions,
        *layout.task_endpoint_exclusions,
    }
    pickups = sorted(
        {
            access
            for _, access in pickup_pairs(config.map_layout_id)
            if access not in excluded
            and access not in layout.pickup_endpoint_exclusions
            and access in layout.dead_end_positions
        }
    )
    deliveries = sorted(
        position
        for position in all_passable_positions(config.map_layout_id)
        if position not in excluded
        and position
        not in {
            layout.charger_position,
            *layout.robot_start_positions,
            *layout.dead_end_positions,
        }
    )
    result = []
    for pickup in pickups:
        for delivery in deliveries:
            if shortest_path_distance(pickup, delivery, config.map_layout_id) < config.minimum_task_distance:
                continue
            record = {"pickup_position": pickup, "delivery_position": delivery}
            result.append({"node_id": task_node_id(record), **record})
    return tuple(result)


def build_conflict_graph(config=None) -> dict[str, Any]:
    """Build and verify the complete task graph for the fixed public layout."""
    config = config or collaborative_study_config()
    nodes = _eligible_jobs(config)
    by_id = {node["node_id"]: node for node in nodes}
    edges = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            metrics = conflict_metrics(left, right, config=config)
            if not metrics["passed"]:
                continue
            pair = sorted((left["node_id"], right["node_id"]))
            edge = {"edge_id": "edge_" + digest(pair)[:16], "nodes": pair, "metrics": metrics}
            edges.append(edge)
            adjacency[pair[0]].add(pair[1])
            adjacency[pair[1]].add(pair[0])

    core = set(adjacency)
    while True:
        removed = {node for node in core if len(adjacency[node] & core) < 2}
        if not removed:
            break
        core -= removed
    core_edges = [edge for edge in edges if set(edge["nodes"]) <= core]
    graph = {
        "version": "warehouse-r41-task-conflict-graph.v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "map_layout_id": config.map_layout_id,
        "configuration_sha256": digest(asdict(config)),
        "nodes": sorted(nodes, key=lambda row: row["node_id"]),
        "edges": sorted(edges, key=lambda row: row["edge_id"]),
        "safe_two_core_nodes": sorted(core),
        "safe_two_core_edges": sorted(edge["edge_id"] for edge in core_edges),
        "transition_table": {
            node: sorted(adjacency[node] & core) for node in sorted(core)
        },
    }
    # This assertion makes topology/config drift an explicit versioning event.
    if config == collaborative_study_config() and (
        len(nodes) != 20 or len(edges) != 14 or len(core) != 6 or len(core_edges) != 6
    ):
        raise RuntimeError("The r4.1 warehouse task graph changed unexpectedly")
    return graph


CONFLICT_GRAPH = build_conflict_graph()
CONFLICT_GRAPH_SHA256 = digest(CONFLICT_GRAPH)
_NODE_BY_ID = {node["node_id"]: node for node in CONFLICT_GRAPH["nodes"]}
_CORE = frozenset(CONFLICT_GRAPH["safe_two_core_nodes"])
_TRANSITIONS = {
    node: tuple(neighbors) for node, neighbors in CONFLICT_GRAPH["transition_table"].items()
}


def validate_active_conflict(tasks: Sequence[DeliveryTask | Mapping[str, Any]], *, config=None) -> dict[str, Any]:
    if len(tasks) != 2:
        raise ValueError("r4.1 requires exactly two active tasks")
    node_ids = [task_node_id(task) for task in tasks]
    if any(node not in _CORE for node in node_ids):
        raise ValueError("An active r4.1 task is outside the safe conflict core")
    if node_ids[1] not in _TRANSITIONS[node_ids[0]]:
        raise ValueError("Active r4.1 tasks are not a conflict-graph edge")
    metrics = conflict_metrics(tasks[0], tasks[1], config=config)
    if not metrics["passed"]:
        raise ValueError("Active r4.1 tasks violate the conflict contract")
    return metrics


class R41ConflictMixin:
    """Task-creation mixin; place before a native environment in the MRO."""

    def _initialize_r41_conflict(self) -> None:
        if self.config != collaborative_study_config():
            raise ValueError("r4.1 conflict dynamics require the frozen study configuration")
        self._r41_sampler_draws = 0
        self._r41_creation_history: list[dict[str, Any]] = []
        self._r41_active_task_nodes: dict[str, str] = {}
        self._r41_sampling_context: dict[str, Any] | None = None

    def _begin_sampling_batch(
        self,
        remaining_node_ids: Sequence[str],
        expected_count: int,
        *,
        post_motion_agent_positions: Sequence[tuple[int, int]] = (),
        delivered_node_ids: Sequence[str] = (),
    ) -> None:
        if self._r41_sampling_context is not None:
            raise RuntimeError("Nested r4.1 replacement sampling is forbidden")
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("Invalid r4.1 replacement count")
        remaining = list(remaining_node_ids)
        delivered = list(delivered_node_ids)
        if len(set(remaining)) != len(remaining) or any(node not in _CORE for node in remaining):
            raise ValueError("Invalid remaining conflict-task context")
        if len(set(delivered)) != len(delivered) or any(node not in _CORE for node in delivered):
            raise ValueError("Invalid delivered conflict-task context")
        if set(remaining) & set(delivered):
            raise ValueError("A task cannot be both remaining and delivered")
        self._r41_sampling_context = {
            "expected_count": expected_count,
            "created": [],
            "active_nodes": remaining,
            "post_motion_agent_positions": [tuple(value) for value in post_motion_agent_positions],
            "delivered_nodes": delivered,
        }

    def _finish_sampling_batch(self) -> list[dict[str, Any]]:
        context = self._r41_sampling_context
        self._r41_sampling_context = None
        if context is None:
            raise RuntimeError("Missing r4.1 sampling context")
        if len(context["created"]) != context["expected_count"]:
            raise RuntimeError("Native transition created an unexpected number of r4.1 tasks")
        return deepcopy(context["created"])

    def _available_node_ids(
        self,
        excluded_positions: set[tuple[int, int]],
        active_nodes: Sequence[str],
        forbidden_node_ids: Sequence[str] = (),
    ) -> list[str]:
        active = tuple(active_nodes)
        forbidden = frozenset(forbidden_node_ids)
        candidates = []
        for node_id in sorted(_CORE):
            if node_id in forbidden:
                continue
            node = _NODE_BY_ID[node_id]
            endpoints = {tuple(node["pickup_position"]), tuple(node["delivery_position"])}
            if endpoints & excluded_positions:
                continue
            if active and any(node_id not in _TRANSITIONS[other] for other in active):
                continue
            if not active:
                # First half of a simultaneous replacement must leave at least
                # one valid second half under the same physical exclusions.
                if not any(
                    neighbor != node_id
                    and neighbor not in forbidden
                    and not (
                        {
                            tuple(_NODE_BY_ID[neighbor]["pickup_position"]),
                            tuple(_NODE_BY_ID[neighbor]["delivery_position"]),
                        }
                        & (excluded_positions | endpoints)
                    )
                    for neighbor in _TRANSITIONS[node_id]
                ):
                    continue
            candidates.append(node_id)
        return candidates

    def _ranked_node_ids(
        self,
        excluded_positions: set[tuple[int, int]],
        active_nodes: Sequence[str],
        forbidden_node_ids: Sequence[str],
        agent_positions: Sequence[tuple[int, int]],
    ) -> tuple[list[str], str]:
        candidates = self._available_node_ids(
            excluded_positions, active_nodes, forbidden_node_ids
        )
        occupied = {tuple(position) for position in agent_positions}
        preferred = [
            node_id
            for node_id in candidates
            if tuple(_NODE_BY_ID[node_id]["pickup_position"]) not in occupied
        ]
        if preferred:
            return preferred, "preferred_clear_pickup"
        return candidates, "strict_conflict_underfoot_pickup"

    def _sample_delivery_job(self, *, task_index, created_frame, excluded_positions):
        context = self._r41_sampling_context
        if context is None:
            raise R41ConflictSamplingError(
                "r4.1 task creation requires an explicit remaining-task context"
            )
        if len(context["created"]) >= context["expected_count"]:
            raise R41ConflictSamplingError("r4.1 task sampler was called too many times")
        # Native task creation happens after this frame's pickup checks.  r4.1
        # additionally excludes every job completed in this creation batch.
        # Clear pickup cells are the first tier; if all strict successors have
        # an occupied pickup, the graph-valid successor remains usable because
        # creation follows this frame's pickup phase.  The sampler never loosens
        # the conflict contract or delegates to the ordinary sampler.
        excluded = {
            self.layout.charger_position,
            *self.layout.robot_start_positions,
            *self.layout.task_endpoint_exclusions,
        }
        for node_id in context["active_nodes"]:
            node = _NODE_BY_ID[node_id]
            excluded.update(
                (tuple(node["pickup_position"]), tuple(node["delivery_position"]))
            )
        candidates, selection_tier = self._ranked_node_ids(
            excluded,
            context["active_nodes"],
            context["delivered_nodes"],
            context["post_motion_agent_positions"],
        )
        if not candidates:
            raise R41ConflictSamplingError(
                "No safe-core successor satisfies the r4.1 conflict contract; no fallback is allowed"
            )
        # Shuffle a stable list with the environment RNG.  This is conditional,
        # deterministic under snapshot restore, and never retries a looser gate.
        self._rng.shuffle(candidates)
        selected = candidates[0]
        node = _NODE_BY_ID[selected]
        task = DeliveryTask(
            task_id=f"task_{int(task_index)}",
            pickup_position=tuple(node["pickup_position"]),
            delivery_position=tuple(node["delivery_position"]),
            created_frame=int(created_frame),
        )
        spawned_on = sorted(
            set((task.pickup_position, task.delivery_position))
            & set(context["post_motion_agent_positions"])
        )
        new_pickup_on_agent = task.pickup_position in set(context["post_motion_agent_positions"])
        new_delivery_on_agent = task.delivery_position in set(context["post_motion_agent_positions"])
        immediate_recreation = selected in set(context["delivered_nodes"])
        if immediate_recreation:
            raise RuntimeError("r4.1 strict replacement filtering was bypassed")
        record = {
            "draw_index": self._r41_sampler_draws,
            "task_id": task.task_id,
            "task_index": int(task_index),
            "created_frame": int(created_frame),
            "node_id": selected,
            "condition_node_ids": list(context["active_nodes"]),
            "delivered_node_ids": list(context["delivered_nodes"]),
            "eligible_node_ids_sha256": digest(sorted(candidates)),
            "eligible_count": len(candidates),
            "selection_tier": selection_tier,
            "spawned_on_agent_endpoint": bool(spawned_on),
            "occupied_spawn_endpoints": spawned_on,
            "new_endpoint_on_agent": bool(spawned_on),
            "new_pickup_on_agent": new_pickup_on_agent,
            "new_delivery_on_agent": new_delivery_on_agent,
            "immediate_task_recreation": immediate_recreation,
        }
        self._r41_sampler_draws += 1
        self._r41_creation_history.append(deepcopy(record))
        context["created"].append(deepcopy(record))
        context["active_nodes"].append(selected)
        return task

    def reset(self, *, seed=None):
        self._r41_sampler_draws = 0
        self._r41_creation_history = []
        self._r41_active_task_nodes = {}
        self._begin_sampling_batch((), int(self.config.active_task_count))
        try:
            result = super().reset(seed=seed)
            created = self._finish_sampling_batch()
        except Exception:
            self._r41_sampling_context = None
            raise
        self._r41_active_task_nodes = {
            task.task_id: task_node_id(task) for task in self.state.tasks
        }
        if {row["task_id"] for row in created} != set(self._r41_active_task_nodes):
            raise RuntimeError("r4.1 reset task provenance differs from native state")
        validate_active_conflict(self.state.tasks, config=self.config)
        return result

    def _predicted_transition_context(
        self, actions: Mapping[str, Any]
    ) -> tuple[list[str], list[tuple[int, int]]]:
        if self.done:
            return [], []
        if any(key not in self.agent_ids for key in actions):
            return [], []
        raw = {key: str(actions.get(key, "WAIT")) for key in self.agent_ids}
        targets = self._resolve_motion(self.state, raw)[0]
        result = []
        for agent in self.state.agents:
            if agent.carrying_task_id is None:
                continue
            task = self.state.task_by_id(agent.carrying_task_id)
            if targets[agent.agent_id] == task.delivery_position:
                result.append(task.task_id)
        return result, [tuple(targets[key]) for key in self.agent_ids]

    def step(self, actions, *, decision_metadata=None):
        predicted, post_motion_positions = self._predicted_transition_context(actions)
        remaining = [
            self._r41_active_task_nodes[task.task_id]
            for task in self.state.tasks
            if task.task_id not in set(predicted)
        ] if self.state is not None else []
        self._begin_sampling_batch(
            remaining,
            len(predicted),
            post_motion_agent_positions=post_motion_positions,
            delivered_node_ids=[self._r41_active_task_nodes[task_id] for task_id in predicted],
        )
        try:
            result = super().step(actions, decision_metadata=decision_metadata)
            created = self._finish_sampling_batch()
        except Exception:
            self._r41_sampling_context = None
            raise
        info = result[-1]
        actual_delivered = sorted(
            event["task_id"] for event in info["events"] if event["event"] == "delivery"
        )
        if actual_delivered != sorted(predicted):
            raise RuntimeError("r4.1 delivery prediction diverged from native physics")
        actual_created = sorted(
            event["task_id"] for event in info["events"] if event["event"] == "task_created"
        )
        if actual_created != sorted(row["task_id"] for row in created):
            raise RuntimeError("r4.1 replacement provenance diverged from native physics")
        self._r41_active_task_nodes = {
            task.task_id: task_node_id(task) for task in self.state.tasks
        }
        metrics = validate_active_conflict(self.state.tasks, config=self.config)
        info["r41_conflict"] = {
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
            "created": created,
            "active_task_node_ids": sorted(self._r41_active_task_nodes.values()),
            "active_pair_metrics_sha256": digest(metrics),
        }
        return result

    def snapshot(self):
        payload = super().snapshot()
        payload["r41_conflict"] = {
            "version": SNAPSHOT_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
            "sampler_draws": self._r41_sampler_draws,
            "active_task_nodes": deepcopy(self._r41_active_task_nodes),
            "creation_history": deepcopy(self._r41_creation_history),
            "successor_state_sha256": digest({
                "rng": payload["rng"],
                "sampler_draws": self._r41_sampler_draws,
                "active_task_nodes": self._r41_active_task_nodes,
                "creation_history": self._r41_creation_history,
            }),
        }
        return payload

    def restore(self, payload, **kwargs):
        if not isinstance(payload, Mapping):
            raise ValueError("r4.1 snapshot must be an object")
        metadata = deepcopy(payload.get("r41_conflict"))
        expected_static = {
            "version": SNAPSHOT_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        }
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected_static.items()):
            raise ValueError("r4.1 snapshot conflict contract mismatch")
        if set(metadata) != {
            *expected_static,
            "sampler_draws",
            "active_task_nodes",
            "creation_history",
            "successor_state_sha256",
        }:
            raise ValueError("r4.1 snapshot metadata schema mismatch")
        base = deepcopy(dict(payload))
        del base["r41_conflict"]
        super().restore(base, **kwargs)
        draws = metadata["sampler_draws"]
        active = metadata["active_task_nodes"]
        history = metadata["creation_history"]
        if type(draws) is not int or draws < 2 or not isinstance(active, dict) or not isinstance(history, list):
            raise ValueError("Invalid r4.1 successor state")
        if draws != len(history):
            raise ValueError("r4.1 sampler draw count differs from creation history")
        if sorted(row.get("draw_index") for row in history) != list(range(draws)):
            raise ValueError("r4.1 creation history is not contiguous")
        expected_active = {task.task_id: task_node_id(task) for task in self.state.tasks}
        if active != expected_active:
            raise ValueError("r4.1 active task-node mapping differs from physical state")
        known_tasks = {task.task_id: task for task in (*self.state.tasks, *self.state.completed_tasks)}
        for row in history:
            if not isinstance(row, dict) or set(row) != {
                "draw_index", "task_id", "task_index", "created_frame", "node_id",
                "condition_node_ids", "eligible_node_ids_sha256", "eligible_count",
                "spawned_on_agent_endpoint", "occupied_spawn_endpoints",
                "delivered_node_ids", "new_endpoint_on_agent", "immediate_task_recreation",
                "new_pickup_on_agent", "new_delivery_on_agent",
                "selection_tier",
            }:
                raise ValueError("r4.1 creation history schema mismatch")
            task = known_tasks.get(row["task_id"])
            if task is None or row["node_id"] != task_node_id(task) or row["created_frame"] != task.created_frame:
                raise ValueError("r4.1 creation history differs from physical tasks")
        successor = digest({
            "rng": base["rng"],
            "sampler_draws": draws,
            "active_task_nodes": active,
            "creation_history": history,
        })
        if metadata["successor_state_sha256"] != successor:
            raise ValueError("r4.1 successor state hash mismatch")
        self._r41_sampler_draws = draws
        self._r41_active_task_nodes = deepcopy(active)
        self._r41_creation_history = deepcopy(history)
        self._r41_sampling_context = None
        validate_active_conflict(self.state.tasks, config=self.config)


class R41NativeWarehouseEnv(R41ConflictMixin, NativeWarehouseEnv):
    """Native177 r4.1 environment, mainly for deterministic dynamics audits."""

    def __init__(self, config=None):
        super().__init__(config)
        self._initialize_r41_conflict()


def r41_scene_fingerprint(env: R41ConflictMixin) -> str:
    """Bind initial physical state, RNG successor state, and conflict graph."""
    snapshot = env.snapshot()
    return digest({
        "version": "warehouse-r41-scene-fingerprint.v1",
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "snapshot": snapshot,
    })


def reset_r41_scenario(env: R41ConflictMixin, entry: Mapping[str, Any]):
    """Restore one manifest scene and verify every r4.1 binding."""
    if not isinstance(entry, Mapping) or not isinstance(entry.get("snapshot"), Mapping):
        raise ValueError("A complete r4.1 scenario entry is required")
    if entry.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Scenario conflict-contract binding differs")
    if entry.get("conflict_graph_sha256") != CONFLICT_GRAPH_SHA256:
        raise ValueError("Scenario task-graph binding differs")
    env.restore(deepcopy(entry["snapshot"]))
    actual = r41_scene_fingerprint(env)
    if entry.get("fingerprint") != actual:
        raise ValueError("Scenario fingerprint differs after r4.1 restore")
    validate_active_conflict(env.state.tasks, config=env.config)
    return env.observations(), env._info()
