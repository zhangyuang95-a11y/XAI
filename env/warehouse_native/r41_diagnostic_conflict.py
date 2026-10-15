"""Robust strict-conflict dynamics for the warehouse r4.1 diagnostic.

The historical r4.1 graph and diagnostic-v1 artifacts remain immutable.  This
v2 contract builds a separate, finite task graph on the same public map and
keeps only the greatest fixed point for which a strict successor always exists.
A replacement task is never created with either endpoint underneath either
robot, including after simultaneous deliveries.  There is no ordinary sampler
fallback and no just-delivered task can be recreated in the same frame.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
from typing import Any, Mapping, Sequence

from env.warehouse.domain import DeliveryTask, collaborative_study_config
from env.warehouse.navigation import (
    all_passable_positions,
    pickup_pairs,
    shortest_path_distance,
)
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH_SHA256 as BASE_CONFLICT_GRAPH_SHA256,
    CONTRACT_SHA256 as R41_CONTRACT_SHA256,
    R41ConflictMixin,
    R41ConflictSamplingError,
    canonical,
    conflict_metrics,
    digest,
    task_node_id,
)


DIAGNOSTIC_CONTRACT_VERSION = "warehouse-r41-diagnostic-conflict.v2"
DIAGNOSTIC_SNAPSHOT_VERSION = "warehouse-r41-diagnostic-conflict-snapshot.v2"
DIAGNOSTIC_SAMPLER_VERSION = "warehouse-r41-diagnostic-robust-endpoint-sampler.v2"
_DIAGNOSTIC_CONFLICT_SNAPSHOT_VERSION = (
    "warehouse-r41-diagnostic-conflict-core-snapshot.v2"
)
_EXTRA_PICKUP_POSITIONS = ((1, 1), (2, 5))


def _candidate_endpoints() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return the minimal robust endpoint expansion on the unchanged map."""
    config = collaborative_study_config()
    probe = NativeWarehouseEnv(config)
    layout = probe.layout
    visible_access = {
        tuple(access) for _, access in pickup_pairs(config.map_layout_id)
    }
    pickups = tuple(sorted({*layout.dead_end_positions, *_EXTRA_PICKUP_POSITIONS}))
    deliveries = tuple(sorted(
        position
        for position in layout.passable_positions
        if position not in {
            layout.charger_position,
            *layout.robot_start_positions,
            *layout.dead_end_positions,
        }
    ))
    if (
        any(position not in visible_access for position in pickups)
        or any(not layout.is_passable(position) for position in (*pickups, *deliveries))
        or any(position in layout.robot_start_positions for position in (*pickups, *deliveries))
        or layout.charger_position in {*pickups, *deliveries}
        or not set(layout.task_endpoint_exclusions) <= set(deliveries)
    ):
        raise RuntimeError("Diagnostic v2 endpoints are not visible passable map cells")
    return pickups, deliveries


def _eligible_jobs() -> tuple[dict[str, Any], ...]:
    config = collaborative_study_config()
    pickups, deliveries = _candidate_endpoints()
    rows = []
    for pickup in pickups:
        for delivery in deliveries:
            if (
                delivery == pickup
                or shortest_path_distance(
                    pickup, delivery, config.map_layout_id
                ) < config.minimum_task_distance
            ):
                continue
            item = {
                "pickup_position": tuple(pickup),
                "delivery_position": tuple(delivery),
            }
            rows.append({"node_id": task_node_id(item), **item})
    return tuple(sorted(rows, key=lambda row: row["node_id"]))


def _edge_id(left: str, right: str) -> str:
    return "edge_" + digest(sorted((left, right)))[:16]


def _endpoints(row: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    return frozenset((
        tuple(row["pickup_position"]), tuple(row["delivery_position"]),
    ))


def _greatest_robust_fixed_point(
    edges: set[frozenset[str]],
    nodes: Mapping[str, Mapping[str, Any]],
    positions: Sequence[tuple[int, int]],
) -> tuple[set[frozenset[str]], list[dict[str, int]]]:
    """Prune until every possible delivery has an endpoint-clear successor."""
    viable = set(edges)
    rounds: list[dict[str, int]] = []
    while True:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in viable:
            left, right = tuple(edge)
            adjacency[left].add(right)
            adjacency[right].add(left)

        def clear(node_id: str, occupied: set[tuple[int, int]]) -> bool:
            return not (_endpoints(nodes[node_id]) & occupied)

        def single_safe(remaining: str, delivered: str) -> bool:
            delivery_position = tuple(nodes[delivered]["delivery_position"])
            for other_position in positions:
                if other_position == delivery_position:
                    continue
                occupied = {delivery_position, tuple(other_position)}
                if not any(
                    candidate != delivered
                    and frozenset((remaining, candidate)) in viable
                    and clear(candidate, occupied)
                    for candidate in adjacency[remaining]
                ):
                    return False
            return True

        def simultaneous_safe(left: str, right: str) -> bool:
            occupied = {
                tuple(nodes[left]["delivery_position"]),
                tuple(nodes[right]["delivery_position"]),
            }
            return any(
                not ({left, right} & set(candidate_edge))
                and all(clear(node_id, occupied) for node_id in candidate_edge)
                for candidate_edge in viable
            )

        kept = set()
        single_failures = simultaneous_failures = 0
        for edge in viable:
            left, right = tuple(edge)
            if not (
                single_safe(left, right) and single_safe(right, left)
            ):
                single_failures += 1
                continue
            if not simultaneous_safe(left, right):
                simultaneous_failures += 1
                continue
            kept.add(edge)
        rounds.append({
            "input_edges": len(viable),
            "kept_edges": len(kept),
            "single_delivery_failures": single_failures,
            "simultaneous_delivery_failures": simultaneous_failures,
        })
        if kept == viable:
            return viable, rounds
        viable = kept


def _family_id(metrics: Mapping[str, Any]) -> str:
    """Group every robust edge into one of six public geometry classes."""
    ratio = float(metrics["shared_edge_ratio_min"])
    if ratio < 0.37:
        index = 1
    elif ratio < 0.42:
        index = 2
    elif ratio < 0.49:
        index = 3
    elif len(metrics["shared_bridge_edges"]) >= 2:
        index = 5 if (4, 3) in metrics["shared_crossings"] else 4
    else:
        index = 6
    return f"conflict_family_{index:02d}"


def _build_diagnostic_graph() -> dict[str, Any]:
    config = collaborative_study_config()
    candidates = _eligible_jobs()
    nodes = {row["node_id"]: row for row in candidates}
    candidate_edges: set[frozenset[str]] = set()
    metrics_by_edge: dict[frozenset[str], dict[str, Any]] = {}
    for left, right in combinations(candidates, 2):
        metrics = conflict_metrics(left, right, config=config)
        if not metrics["passed"]:
            continue
        edge = frozenset((left["node_id"], right["node_id"]))
        candidate_edges.add(edge)
        metrics_by_edge[edge] = metrics
    robust, rounds = _greatest_robust_fixed_point(
        candidate_edges,
        nodes,
        tuple(sorted(all_passable_positions(config.map_layout_id))),
    )
    robust_nodes = sorted(set().union(*(set(edge) for edge in robust))) if robust else []
    edges = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in sorted(robust, key=lambda value: sorted(value)):
        left, right = sorted(edge)
        metrics = metrics_by_edge[edge]
        family = _family_id(metrics)
        edges.append({
            "edge_id": _edge_id(left, right),
            "nodes": [left, right],
            "family_id": family,
            "metrics": metrics,
        })
        adjacency[left].add(right)
        adjacency[right].add(left)
    graph = {
        "version": "warehouse-r41-diagnostic-robust-conflict-graph.v2",
        "base_r41_contract_sha256": R41_CONTRACT_SHA256,
        "base_conflict_graph_sha256": BASE_CONFLICT_GRAPH_SHA256,
        "configuration_sha256": digest(asdict(config)),
        "pickup_positions": [list(value) for value in _candidate_endpoints()[0]],
        "delivery_positions": [list(value) for value in _candidate_endpoints()[1]],
        "candidate_node_count": len(candidates),
        "candidate_edge_count": len(candidate_edges),
        "robust_nodes": robust_nodes,
        "robust_edges": edges,
        "transition_table": {
            node: sorted(adjacency[node]) for node in robust_nodes
        },
        "fixed_point_rounds": rounds,
        "invariant": {
            "single_delivery_contexts": len(edges) * 2 * (
                len(tuple(all_passable_positions(config.map_layout_id))) - 1
            ),
            "simultaneous_delivery_contexts": len(edges),
            "checks_both_successor_endpoints": True,
            "forbids_immediate_recreation": True,
            "ordinary_sampler_fallback": False,
        },
    }
    family_counts = defaultdict(int)
    for edge in edges:
        family_counts[edge["family_id"]] += 1
    if (
        len(candidates) != 44
        or len(candidate_edges) != 79
        or len(robust_nodes) != 13
        or len(edges) != 31
        or set(family_counts) != {
            f"conflict_family_{index:02d}" for index in range(1, 7)
        }
        or any(not edge["metrics"]["passed"] for edge in edges)
        or rounds[-1]["input_edges"] != rounds[-1]["kept_edges"]
    ):
        raise RuntimeError(
            "Diagnostic v2 robust conflict graph changed unexpectedly: "
            f"candidates={len(candidates)}, candidate_edges={len(candidate_edges)}, "
            f"nodes={len(robust_nodes)}, edges={len(edges)}, "
            f"families={dict(family_counts)}, rounds={rounds}"
        )
    graph["family_edge_counts"] = dict(sorted(family_counts.items()))
    return graph


DIAGNOSTIC_CONFLICT_GRAPH = _build_diagnostic_graph()
DIAGNOSTIC_CONFLICT_GRAPH_SHA256 = digest(DIAGNOSTIC_CONFLICT_GRAPH)
NODE_BY_ID = {
    row["node_id"]: row
    for row in _eligible_jobs()
    if row["node_id"] in set(DIAGNOSTIC_CONFLICT_GRAPH["robust_nodes"])
}
EDGE_BY_NODES = {
    tuple(row["nodes"]): row for row in DIAGNOSTIC_CONFLICT_GRAPH["robust_edges"]
}
_TRANSITIONS = {
    node: tuple(neighbors)
    for node, neighbors in DIAGNOSTIC_CONFLICT_GRAPH["transition_table"].items()
}
_CORE = frozenset(_TRANSITIONS)


def _family_rows() -> tuple[dict[str, Any], ...]:
    descriptions = (
        "one-third shared-route conflict",
        "two-fifths shared-route conflict",
        "three-sevenths shared-route conflict",
        "half-route shared two-bridge conflict",
        "half-route shared exit-crossing conflict",
        "half-route shared single-bridge conflict",
    )
    result = []
    for index, description in enumerate(descriptions, 1):
        family_id = f"conflict_family_{index:02d}"
        edge_ids = sorted(
            row["edge_id"]
            for row in DIAGNOSTIC_CONFLICT_GRAPH["robust_edges"]
            if row["family_id"] == family_id
        )
        result.append({
            "family_id": family_id,
            "geometry": description,
            "edge_ids": edge_ids,
            "edge_count": len(edge_ids),
            "edges_sha256": digest(edge_ids),
        })
    return tuple(result)


CONFLICT_FAMILIES = _family_rows()
CONFLICT_FAMILIES_SHA256 = digest(CONFLICT_FAMILIES)
FAMILY_BY_EDGE = {
    row["edge_id"]: row["family_id"]
    for row in DIAGNOSTIC_CONFLICT_GRAPH["robust_edges"]
}


def diagnostic_graph_invariant_audit() -> dict[str, Any]:
    """Exhaustively recheck the endpoint-clear closure used by the sampler."""
    positions = tuple(sorted(all_passable_positions(
        collaborative_study_config().map_layout_id
    )))
    single_counts = []
    simultaneous_first_counts = []
    simultaneous_second_counts = []
    for edge_row in DIAGNOSTIC_CONFLICT_GRAPH["robust_edges"]:
        left, right = edge_row["nodes"]
        for remaining, delivered in ((left, right), (right, left)):
            delivery = tuple(NODE_BY_ID[delivered]["delivery_position"])
            for other in positions:
                if other == delivery:
                    continue
                occupied = {delivery, other}
                candidates = [
                    node_id for node_id in _TRANSITIONS[remaining]
                    if node_id != delivered
                    and not (_endpoints(NODE_BY_ID[node_id]) & occupied)
                ]
                if not candidates:
                    raise RuntimeError("Diagnostic single-delivery closure is empty")
                single_counts.append(len(candidates))
        occupied = {
            tuple(NODE_BY_ID[left]["delivery_position"]),
            tuple(NODE_BY_ID[right]["delivery_position"]),
        }
        first = []
        second_counts = []
        for node_id in sorted(_CORE - {left, right}):
            endpoints = _endpoints(NODE_BY_ID[node_id])
            if endpoints & occupied:
                continue
            seconds = [
                neighbor for neighbor in _TRANSITIONS[node_id]
                if neighbor not in {left, right}
                and not (
                    _endpoints(NODE_BY_ID[neighbor])
                    & (occupied | set(endpoints))
                )
            ]
            if seconds:
                first.append(node_id)
                second_counts.append(len(seconds))
        if not first:
            raise RuntimeError("Diagnostic simultaneous-delivery closure is empty")
        simultaneous_first_counts.append(len(first))
        simultaneous_second_counts.extend(second_counts)
    result = {
        "version": "warehouse-r41-diagnostic-robust-invariant-audit.v2",
        "edge_count": len(DIAGNOSTIC_CONFLICT_GRAPH["robust_edges"]),
        "single_delivery_contexts": len(single_counts),
        "single_successor_count_min": min(single_counts),
        "single_successor_count_max": max(single_counts),
        "simultaneous_delivery_contexts": len(simultaneous_first_counts),
        "simultaneous_first_count_min": min(simultaneous_first_counts),
        "simultaneous_first_count_max": max(simultaneous_first_counts),
        "simultaneous_second_count_min": min(simultaneous_second_counts),
        "simultaneous_second_count_max": max(simultaneous_second_counts),
        "both_endpoints_clear": True,
        "immediate_recreation_forbidden": True,
        "ordinary_sampler_fallback": False,
        "passed": True,
    }
    result["audit_sha256"] = digest(result)
    return result

DIAGNOSTIC_CONTRACT = {
    "version": DIAGNOSTIC_CONTRACT_VERSION,
    "base_r41_contract_sha256": R41_CONTRACT_SHA256,
    "base_conflict_graph_sha256": BASE_CONFLICT_GRAPH_SHA256,
    "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    "conflict_family_definition": "six public route-geometry classes over the robust edge closure",
    "conflict_family_count": 6,
    "replacement_policy": "sample only a robust-closure successor with both endpoints clear of both robots",
    "occupied_endpoint_policy": "fail closed; never create either endpoint underneath a robot",
    "resampling": "deterministic environment-RNG shuffle over the complete strict robust candidate set",
    "single_delivery_robustness": "exhaustive over both delivery sides and every distinct passable other-agent position",
    "simultaneous_delivery_robustness": "every allowed active edge has a disjoint endpoint-clear robust replacement edge",
    "immediate_task_recreation": False,
    "ordinary_sampler_fallback": False,
    "post_policy_action_override": False,
}
DIAGNOSTIC_CONTRACT_SHA256 = digest(DIAGNOSTIC_CONTRACT)


def active_edge_id(tasks: Sequence[Any]) -> str:
    nodes = tuple(sorted(task_node_id(task) for task in tasks))
    try:
        return EDGE_BY_NODES[nodes]["edge_id"]
    except KeyError as error:
        raise ValueError("Active tasks are outside the diagnostic robust edge closure") from error


def conflict_family_id(tasks: Sequence[Any]) -> str:
    return FAMILY_BY_EDGE[active_edge_id(tasks)]


def validate_diagnostic_active_conflict(
    tasks: Sequence[Any], *, config=None
) -> dict[str, Any]:
    if len(tasks) != 2:
        raise ValueError("Diagnostic v2 requires exactly two active tasks")
    node_ids = [task_node_id(task) for task in tasks]
    if any(node not in _CORE for node in node_ids):
        raise ValueError("An active task is outside the diagnostic robust core")
    edge_id = active_edge_id(tasks)
    metrics = conflict_metrics(tasks[0], tasks[1], config=config)
    if not metrics["passed"]:
        raise ValueError("Active diagnostic tasks violate the public conflict gates")
    return {
        **metrics,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_family_id": FAMILY_BY_EDGE[edge_id],
        "initial_edge_id": edge_id,
    }


class R41DiagnosticConflictMixin(R41ConflictMixin):
    """Use the robust v2 graph without modifying historical r4.1 dynamics."""

    def _begin_sampling_batch(
        self,
        remaining_node_ids: Sequence[str],
        expected_count: int,
        *,
        post_motion_agent_positions: Sequence[tuple[int, int]] = (),
        delivered_node_ids: Sequence[str] = (),
    ) -> None:
        if self._r41_sampling_context is not None:
            raise RuntimeError("Nested diagnostic replacement sampling is forbidden")
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("Invalid diagnostic replacement count")
        remaining = list(remaining_node_ids)
        delivered = list(delivered_node_ids)
        if (
            len(set(remaining)) != len(remaining)
            or any(node not in _CORE for node in remaining)
            or len(set(delivered)) != len(delivered)
            or any(node not in _CORE for node in delivered)
            or set(remaining) & set(delivered)
        ):
            raise ValueError("Invalid diagnostic robust-task sampling context")
        self._r41_sampling_context = {
            "expected_count": expected_count,
            "created": [],
            "active_nodes": remaining,
            "post_motion_agent_positions": [
                tuple(value) for value in post_motion_agent_positions
            ],
            "delivered_nodes": delivered,
        }

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
            endpoints = _endpoints(NODE_BY_ID[node_id])
            if endpoints & excluded_positions:
                continue
            if active and any(node_id not in _TRANSITIONS[other] for other in active):
                continue
            if not active and not any(
                neighbor not in forbidden
                and not (_endpoints(NODE_BY_ID[neighbor]) & (excluded_positions | set(endpoints)))
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
        occupied = {tuple(value) for value in agent_positions}
        return (
            self._available_node_ids(
                excluded_positions | occupied, active_nodes, forbidden_node_ids
            ),
            "diagnostic_robust_clear_endpoints",
        )

    def _sample_delivery_job(self, *, task_index, created_frame, excluded_positions):
        context = self._r41_sampling_context
        if context is None:
            raise R41ConflictSamplingError(
                "Diagnostic task creation requires an explicit robust context"
            )
        if len(context["created"]) >= context["expected_count"]:
            raise R41ConflictSamplingError(
                "Diagnostic robust sampler was called too many times"
            )
        excluded = {
            self.layout.charger_position,
            *self.layout.robot_start_positions,
        }
        for node_id in context["active_nodes"]:
            excluded.update(_endpoints(NODE_BY_ID[node_id]))
        candidates, selection_tier = self._ranked_node_ids(
            excluded,
            context["active_nodes"],
            context["delivered_nodes"],
            context["post_motion_agent_positions"],
        )
        if not candidates:
            raise R41ConflictSamplingError(
                "No endpoint-clear robust successor exists; no fallback is allowed"
            )
        self._rng.shuffle(candidates)
        selected = candidates[0]
        node = NODE_BY_ID[selected]
        task = DeliveryTask(
            task_id=f"task_{int(task_index)}",
            pickup_position=tuple(node["pickup_position"]),
            delivery_position=tuple(node["delivery_position"]),
            created_frame=int(created_frame),
        )
        occupied = set(context["post_motion_agent_positions"])
        spawned_on = sorted(_endpoints(node) & occupied)
        immediate_recreation = selected in set(context["delivered_nodes"])
        if spawned_on or immediate_recreation:
            raise RuntimeError("Diagnostic robust successor filtering was bypassed")
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
            "spawned_on_agent_endpoint": False,
            "occupied_spawn_endpoints": [],
            "new_endpoint_on_agent": False,
            "new_pickup_on_agent": False,
            "new_delivery_on_agent": False,
            "immediate_task_recreation": False,
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
        self._r41_sampling_context = None
        self._begin_sampling_batch((), int(self.config.active_task_count))
        try:
            result = super(R41ConflictMixin, self).reset(seed=seed)
            created = self._finish_sampling_batch()
        except Exception:
            self._r41_sampling_context = None
            raise
        self._r41_active_task_nodes = {
            task.task_id: task_node_id(task) for task in self.state.tasks
        }
        if {row["task_id"] for row in created} != set(self._r41_active_task_nodes):
            raise RuntimeError("Diagnostic reset provenance differs from physical state")
        validate_diagnostic_active_conflict(self.state.tasks, config=self.config)
        return result

    def step(self, actions, *, decision_metadata=None):
        predicted, post_motion_positions = self._predicted_transition_context(actions)
        predicted_set = set(predicted)
        remaining = [
            self._r41_active_task_nodes[task.task_id]
            for task in self.state.tasks
            if task.task_id not in predicted_set
        ] if self.state is not None else []
        self._begin_sampling_batch(
            remaining,
            len(predicted),
            post_motion_agent_positions=post_motion_positions,
            delivered_node_ids=[
                self._r41_active_task_nodes[task_id] for task_id in predicted
            ],
        )
        try:
            result = super(R41ConflictMixin, self).step(
                actions, decision_metadata=decision_metadata
            )
            created = self._finish_sampling_batch()
        except Exception:
            self._r41_sampling_context = None
            raise
        info = result[-1]
        actual_delivered = sorted(
            event["task_id"] for event in info["events"]
            if event["event"] == "delivery"
        )
        if actual_delivered != sorted(predicted):
            raise RuntimeError("Diagnostic delivery prediction diverged from physics")
        actual_created = sorted(
            event["task_id"] for event in info["events"]
            if event["event"] == "task_created"
        )
        if actual_created != sorted(row["task_id"] for row in created):
            raise RuntimeError("Diagnostic replacement provenance diverged from physics")
        if any(
            row["new_endpoint_on_agent"] or row["immediate_task_recreation"]
            for row in created
        ):
            raise RuntimeError("Diagnostic endpoint/recreation contract was bypassed")
        self._r41_active_task_nodes = {
            task.task_id: task_node_id(task) for task in self.state.tasks
        }
        metrics = validate_diagnostic_active_conflict(
            self.state.tasks, config=self.config
        )
        info["r41_conflict"] = {
            "contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "created": deepcopy(created),
            "active_task_node_ids": sorted(self._r41_active_task_nodes.values()),
            "active_pair_metrics_sha256": digest(metrics),
        }
        info["r41_diagnostic_conflict"] = {
            "contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
            "conflict_family_id": metrics["conflict_family_id"],
            "active_pair_metrics_sha256": digest(metrics),
            "created": deepcopy(created),
            "ordinary_sampler_fallback": False,
        }
        return result

    def snapshot(self):
        payload = super(R41ConflictMixin, self).snapshot()
        successor_state = {
            "rng": payload["rng"],
            "sampler_draws": self._r41_sampler_draws,
            "active_task_nodes": self._r41_active_task_nodes,
            "creation_history": self._r41_creation_history,
        }
        payload["r41_conflict"] = {
            "version": _DIAGNOSTIC_CONFLICT_SNAPSHOT_VERSION,
            "sampler_version": DIAGNOSTIC_SAMPLER_VERSION,
            "contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "sampler_draws": self._r41_sampler_draws,
            "active_task_nodes": deepcopy(self._r41_active_task_nodes),
            "creation_history": deepcopy(self._r41_creation_history),
            "successor_state_sha256": digest(successor_state),
        }
        base_successor_sha = payload["r41_conflict"]["successor_state_sha256"]
        payload["r41_diagnostic_conflict"] = {
            "version": DIAGNOSTIC_SNAPSHOT_VERSION,
            "sampler_version": DIAGNOSTIC_SAMPLER_VERSION,
            "contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "base_contract_sha256": R41_CONTRACT_SHA256,
            "base_conflict_graph_sha256": BASE_CONFLICT_GRAPH_SHA256,
            "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
            "base_successor_state_sha256": base_successor_sha,
            "binding_sha256": digest({
                "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
                "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
                "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
                "base_successor_state_sha256": base_successor_sha,
            }),
        }
        return payload

    def restore(self, payload, **kwargs):
        if not isinstance(payload, Mapping):
            raise ValueError("Diagnostic conflict snapshot must be an object")
        diagnostic = deepcopy(payload.get("r41_diagnostic_conflict"))
        conflict = deepcopy(payload.get("r41_conflict"))
        expected_diagnostic = {
            "version": DIAGNOSTIC_SNAPSHOT_VERSION,
            "sampler_version": DIAGNOSTIC_SAMPLER_VERSION,
            "contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "base_contract_sha256": R41_CONTRACT_SHA256,
            "base_conflict_graph_sha256": BASE_CONFLICT_GRAPH_SHA256,
            "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        }
        expected_conflict = {
            "version": _DIAGNOSTIC_CONFLICT_SNAPSHOT_VERSION,
            "sampler_version": DIAGNOSTIC_SAMPLER_VERSION,
            "contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        }
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != {
                *expected_diagnostic,
                "base_successor_state_sha256", "binding_sha256",
            }
            or any(diagnostic.get(key) != value for key, value in expected_diagnostic.items())
            or not isinstance(conflict, dict)
            or set(conflict) != {
                *expected_conflict,
                "sampler_draws", "active_task_nodes", "creation_history",
                "successor_state_sha256",
            }
            or any(conflict.get(key) != value for key, value in expected_conflict.items())
        ):
            raise ValueError("Diagnostic v2 snapshot binding differs")
        expected_binding = digest({
            "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
            "base_successor_state_sha256": conflict["successor_state_sha256"],
        })
        if (
            diagnostic["base_successor_state_sha256"]
            != conflict["successor_state_sha256"]
            or diagnostic["binding_sha256"] != expected_binding
        ):
            raise ValueError("Diagnostic v2 snapshot hash differs")
        base = deepcopy(dict(payload))
        del base["r41_conflict"]
        del base["r41_diagnostic_conflict"]
        super(R41ConflictMixin, self).restore(base, **kwargs)
        draws = conflict["sampler_draws"]
        active = conflict["active_task_nodes"]
        history = conflict["creation_history"]
        if (
            type(draws) is not int or draws < 2
            or not isinstance(active, dict) or not isinstance(history, list)
            or draws != len(history)
            or sorted(row.get("draw_index") for row in history) != list(range(draws))
        ):
            raise ValueError("Invalid diagnostic v2 successor state")
        expected_active = {
            task.task_id: task_node_id(task) for task in self.state.tasks
        }
        if active != expected_active or any(node not in _CORE for node in active.values()):
            raise ValueError("Diagnostic active task mapping differs")
        known_tasks = {
            task.task_id: task
            for task in (*self.state.tasks, *self.state.completed_tasks)
        }
        expected_history_fields = {
            "draw_index", "task_id", "task_index", "created_frame", "node_id",
            "condition_node_ids", "eligible_node_ids_sha256", "eligible_count",
            "selection_tier", "spawned_on_agent_endpoint",
            "occupied_spawn_endpoints", "delivered_node_ids",
            "new_endpoint_on_agent", "new_pickup_on_agent",
            "new_delivery_on_agent", "immediate_task_recreation",
        }
        seen_task_ids = set()
        for row in history:
            task = known_tasks.get(row.get("task_id")) if isinstance(row, dict) else None
            condition_nodes = row.get("condition_node_ids") if isinstance(row, dict) else None
            delivered_nodes = row.get("delivered_node_ids") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict) or set(row) != expected_history_fields
                or task is None or row["node_id"] != task_node_id(task)
                or row["created_frame"] != task.created_frame
                or row["task_id"] in seen_task_ids
                or row["selection_tier"] != "diagnostic_robust_clear_endpoints"
                or row["spawned_on_agent_endpoint"] is not False
                or row["occupied_spawn_endpoints"] != []
                or row["new_endpoint_on_agent"] is not False
                or row["new_pickup_on_agent"] is not False
                or row["new_delivery_on_agent"] is not False
                or row["immediate_task_recreation"] is not False
                or not isinstance(condition_nodes, list)
                or len(condition_nodes) != len(set(condition_nodes))
                or any(node not in _CORE for node in condition_nodes)
                or not isinstance(delivered_nodes, list)
                or len(delivered_nodes) != len(set(delivered_nodes))
                or any(node not in _CORE for node in delivered_nodes)
                or set(condition_nodes) & set(delivered_nodes)
                or row["node_id"] in {*condition_nodes, *delivered_nodes}
                or any(
                    row["node_id"] not in _TRANSITIONS[node]
                    for node in condition_nodes
                )
                or type(row["eligible_count"]) is not int
                or row["eligible_count"] < 1
                or not isinstance(row["eligible_node_ids_sha256"], str)
                or len(row["eligible_node_ids_sha256"]) != 64
            ):
                raise ValueError("Diagnostic creation history differs from physical tasks")
            seen_task_ids.add(row["task_id"])
        successor = digest({
            "rng": base["rng"],
            "sampler_draws": draws,
            "active_task_nodes": active,
            "creation_history": history,
        })
        if conflict["successor_state_sha256"] != successor:
            raise ValueError("Diagnostic v2 successor state hash mismatch")
        self._r41_sampler_draws = draws
        self._r41_active_task_nodes = deepcopy(active)
        self._r41_creation_history = deepcopy(history)
        self._r41_sampling_context = None
        validate_diagnostic_active_conflict(self.state.tasks, config=self.config)


class R41DiagnosticNativeWarehouseEnv(R41DiagnosticConflictMixin, NativeWarehouseEnv):
    """Native environment implementing the diagnostic v2 robust graph."""

    def __init__(self, config=None):
        super().__init__(config)
        self._initialize_r41_conflict()


def diagnostic_scene_fingerprint(env: R41DiagnosticConflictMixin) -> str:
    snapshot = env.snapshot()
    return digest({
        "version": "warehouse-r41-diagnostic-scene-fingerprint.v2",
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "snapshot": snapshot,
    })


def reset_diagnostic_scenario(
    env: R41DiagnosticConflictMixin, entry: Mapping[str, Any]
):
    if not isinstance(entry, Mapping) or not isinstance(entry.get("snapshot"), Mapping):
        raise ValueError("A complete diagnostic conflict scene is required")
    if entry.get("diagnostic_contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256:
        raise ValueError("Diagnostic scene contract binding differs")
    if (
        entry.get("diagnostic_conflict_graph_sha256")
        != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
    ):
        raise ValueError("Diagnostic scene robust graph binding differs")
    if entry.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256:
        raise ValueError("Diagnostic scene family binding differs")
    env.restore(deepcopy(entry["snapshot"]))
    if entry.get("fingerprint") != diagnostic_scene_fingerprint(env):
        raise ValueError("Diagnostic scene fingerprint differs after restore")
    metrics = validate_diagnostic_active_conflict(env.state.tasks, config=env.config)
    if entry.get("family_id") != metrics["conflict_family_id"]:
        raise ValueError("Diagnostic scene conflict family differs")
    return env.observations(), env._info()


def diagnostic_contract_receipt() -> dict[str, Any]:
    return {
        "version": DIAGNOSTIC_CONTRACT_VERSION,
        "contract": deepcopy(DIAGNOSTIC_CONTRACT),
        "contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "base_contract_sha256": R41_CONTRACT_SHA256,
        "base_conflict_graph_sha256": BASE_CONFLICT_GRAPH_SHA256,
        "conflict_graph": deepcopy(DIAGNOSTIC_CONFLICT_GRAPH),
        "conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "robustness_audit": diagnostic_graph_invariant_audit(),
        "conflict_families": deepcopy(CONFLICT_FAMILIES),
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "configuration_sha256": digest(asdict(collaborative_study_config())),
        "canonical_sha256": digest(canonical(DIAGNOSTIC_CONTRACT)),
    }
