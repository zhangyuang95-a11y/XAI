"""Deterministic scene production and admission for warehouse r4.1.

The producer creates disjoint training, validation, final-test, tutorial and
play pools.  Every scene uses the same fail-closed conflict-task environment,
and every saved scene is replayed through repeated task replenishment before it
can enter a manifest.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from backend.warehouse_r41_online_runtime import (
    DEFAULT_REWARD_CONFIG,
    R41ConflictWarehouseEnv,
)
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH,
    CONFLICT_GRAPH_SHA256,
    CONTRACT,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    canonical,
    conflict_metrics,
    digest,
    r41_scene_fingerprint,
    reset_r41_scenario,
    task_node_id,
    validate_active_conflict,
)


MANIFEST_VERSION = "warehouse-r41-conflict-scene-manifest.v1"
SELECTION_VERSION = "warehouse-r41-conflict-play-selection.v1"
FORBIDDEN_LEGACY_SEEDS = tuple(range(260_908, 260_920))
SPLIT_SEED_STARTS = {
    "train": 4_110_000,
    "conflict_validation": 4_210_000,
    "final_test": 4_310_000,
    "tutorial": 4_410_000,
    "play_candidates": 4_510_000,
}
DEFAULT_COUNTS = {
    "train": 128,
    "conflict_validation": 64,
    "final_test": 64,
    "tutorial": 1,
    "play_candidates": 60,
}
OPERATIONAL_SPLITS = ("train", "conflict_validation", "final_test", "tutorial", "play")
SOURCE_ROOT = Path(__file__).resolve().parents[2]

# Stable training-facing name; the lower-level name remains available for
# runtime code that emphasizes snapshot restoration.
reset_conflict_scenario = reset_r41_scenario


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def task_geometry_signature(tasks: Sequence[Mapping[str, Any] | Any]) -> str:
    endpoints = sorted(
        (
            tuple(task["pickup_position"] if isinstance(task, Mapping) else task.pickup_position),
            tuple(task["delivery_position"] if isinstance(task, Mapping) else task.delivery_position),
        )
        for task in tasks
    )
    return digest({"map_layout_id": collaborative_study_config().map_layout_id, "tasks": endpoints})


def _scene_entries_from_old_manifest(value: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    splits = value.get("splits")
    if isinstance(splits, Mapping):
        for scene in splits.get("play", []):
            if isinstance(scene, Mapping):
                yield scene
    play = value.get("play")
    if isinstance(play, list):
        for scene in play:
            if isinstance(scene, Mapping):
                yield scene
    selected = value.get("selected_play")
    if isinstance(selected, Mapping):
        for key in ("X", "Y"):
            for scene in selected.get(key, []):
                if isinstance(scene, Mapping):
                    yield scene


def old_geometry_receipts(paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], set[str], set[int]]:
    if not paths:
        raise ValueError("At least one immutable old scene manifest is required")
    receipts: list[dict[str, Any]] = []
    geometries: set[str] = set()
    seeds: set[int] = set()
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        raw = path.read_bytes()
        value = json.loads(raw)
        entries = list(_scene_entries_from_old_manifest(value))
        for scene in entries:
            snapshot = scene.get("snapshot")
            tasks = snapshot.get("state", {}).get("tasks") if isinstance(snapshot, Mapping) else None
            if not isinstance(tasks, list) or len(tasks) != 2:
                raise ValueError(f"Old play scene in {path} lacks two task geometries")
            geometries.add(task_geometry_signature(tasks))
            if type(scene.get("seed")) is int:
                seeds.add(int(scene["seed"]))
        receipts.append(
            {
                "path": str(path),
                "sha256": sha256(raw).hexdigest(),
                "play_scene_count": len(entries),
                "play_geometry_sha256": digest(sorted(
                    task_geometry_signature(scene["snapshot"]["state"]["tasks"])
                    for scene in entries
                )),
            }
        )
    return receipts, geometries, seeds


def _force_delivery_state(env: R41ConflictWarehouseEnv, delivered_indices: Sequence[int]) -> None:
    state = deepcopy(env.state)
    selected = set(delivered_indices)
    for task in state.tasks:
        task.status = "available"
        task.carrier_agent_id = None
        task.claimed_frame = None
        task.delivered_frame = None
    for index, agent in enumerate(state.agents):
        agent.position = env.layout.robot_start_positions[index]
        agent.battery = 100.0
        agent.active = True
        agent.carrying_task_id = None
        agent.last_action = "WAIT"
        agent.last_executed_action = "WAIT"
    for agent_index, task_index in enumerate(sorted(selected)):
        task = state.tasks[task_index]
        agent = state.agents[agent_index]
        task.status = "carried"
        task.carrier_agent_id = agent.agent_id
        task.claimed_frame = state.frame
        agent.position = task.delivery_position
        agent.carrying_task_id = task.task_id
    state.terminated = False
    state.truncated = False
    state.terminal_reason = None
    env.set_state(state)


def successor_replay_audit(
    snapshot: Mapping[str, Any], *, cycles: int = 12
) -> dict[str, Any]:
    """Exercise actual native ``step`` replenishment and exact branch replay."""
    if type(cycles) is not int or cycles < 3:
        raise ValueError("Successor replay needs at least three cycles")
    env = R41ConflictWarehouseEnv()
    env.restore(deepcopy(snapshot))
    transcript = []
    patterns = ((0,), (1,), (0, 1))
    replacements = 0
    spawned_on_agent_endpoint = 0
    immediate_task_recreation = 0
    new_endpoint_on_agent = 0
    new_pickup_on_agent = 0
    new_delivery_on_agent = 0
    for cycle in range(cycles):
        pattern = patterns[cycle % len(patterns)]
        before_nodes = [task_node_id(task) for task in env.state.tasks]
        _force_delivery_state(env, pattern)
        forced = env.snapshot()
        first = R41ConflictWarehouseEnv()
        second = R41ConflictWarehouseEnv()
        first.restore(deepcopy(forced))
        second.restore(deepcopy(forced))
        first_result = first.step({"robot_1": "WAIT", "robot_2": "WAIT"})
        second_result = second.step({"robot_1": "WAIT", "robot_2": "WAIT"})
        replay_equal = first.snapshot() == second.snapshot()
        after_metrics = validate_active_conflict(first.state.tasks)
        created = [
            event["task_id"]
            for event in first_result[-1]["events"]
            if event["event"] == "task_created"
        ]
        creation_records = first_result[-1]["r41_conflict"]["created"]
        replacements += len(created)
        spawned_on_agent_endpoint += sum(
            bool(row["spawned_on_agent_endpoint"]) for row in creation_records
        )
        immediate_task_recreation += sum(
            bool(row["immediate_task_recreation"]) for row in creation_records
        )
        new_endpoint_on_agent += sum(
            bool(row["new_endpoint_on_agent"]) for row in creation_records
        )
        new_pickup_on_agent += sum(
            bool(row["new_pickup_on_agent"]) for row in creation_records
        )
        new_delivery_on_agent += sum(
            bool(row["new_delivery_on_agent"]) for row in creation_records
        )
        transcript.append(
            {
                "cycle": cycle,
                "delivery_indices": list(pattern),
                "before_node_ids": before_nodes,
                "after_node_ids": [task_node_id(task) for task in first.state.tasks],
                "created_task_ids": created,
                "spawned_on_agent_endpoint": sum(
                    bool(row["spawned_on_agent_endpoint"]) for row in creation_records
                ),
                "immediate_task_recreation": sum(
                    bool(row["immediate_task_recreation"]) for row in creation_records
                ),
                "new_endpoint_on_agent": sum(
                    bool(row["new_endpoint_on_agent"]) for row in creation_records
                ),
                "new_pickup_on_agent": sum(
                    bool(row["new_pickup_on_agent"]) for row in creation_records
                ),
                "new_delivery_on_agent": sum(
                    bool(row["new_delivery_on_agent"]) for row in creation_records
                ),
                "replay_equal": replay_equal,
                "conflict_passed": after_metrics["passed"],
                "conflict_metrics_sha256": digest(after_metrics),
                "after_snapshot_sha256": digest(first.snapshot()),
            }
        )
        if not replay_equal or not after_metrics["passed"]:
            raise ValueError("r4.1 successor replay failed")
        env = first
    return {
        "version": "warehouse-r41-successor-replay.v1",
        "cycles": cycles,
        "replacement_tasks": replacements,
        "spawned_on_agent_endpoint": spawned_on_agent_endpoint,
        "immediate_task_recreation": immediate_task_recreation,
        "new_endpoint_on_agent": new_endpoint_on_agent,
        "new_pickup_on_agent": new_pickup_on_agent,
        "new_delivery_on_agent": new_delivery_on_agent,
        "patterns": [list(pattern) for pattern in patterns],
        "all_replays_equal": all(row["replay_equal"] for row in transcript),
        "all_active_pairs_passed": all(row["conflict_passed"] for row in transcript),
        "trace_sha256": digest(transcript),
        "active_pair_sequence_sha256": digest([row["after_node_ids"] for row in transcript]),
    }


def transition_closure_audit() -> dict[str, Any]:
    """Prove every edge/delivery case survives every two-robot occupancy."""
    nodes = {row["node_id"]: row for row in CONFLICT_GRAPH["nodes"]}
    core = set(CONFLICT_GRAPH["safe_two_core_nodes"])
    edges = [row for row in CONFLICT_GRAPH["edges"] if set(row["nodes"]) <= core]
    env = R41ConflictWarehouseEnv()
    env.reset(seed=SPLIT_SEED_STARTS["train"])
    rows = []
    position_case_count = 0
    underfoot_tier_count = 0
    static = {
        env.layout.charger_position,
        *env.layout.robot_start_positions,
        *env.layout.task_endpoint_exclusions,
    }
    for edge in edges:
        for delivered in ((0,), (1,), (0, 1)):
            delivered_nodes = [edge["nodes"][index] for index in delivered]
            remaining = [node for index, node in enumerate(edge["nodes"]) if index not in delivered]
            protected = set(static)
            for node_id in remaining:
                protected.update((
                    tuple(nodes[node_id]["pickup_position"]),
                    tuple(nodes[node_id]["delivery_position"]),
                ))
            local_cases = 0
            local_underfoot = 0
            for first_position, second_position in permutations(env.layout.passable_positions, 2):
                agent_positions = (first_position, second_position)
                first, tier = env._ranked_node_ids(
                    protected, remaining, delivered_nodes, agent_positions
                )
                if not first:
                    raise ValueError("A conflict-graph transition has no successor")
                local_underfoot += int(tier == "strict_conflict_underfoot_pickup")
                if len(delivered) == 2:
                    for node_id in first:
                        node = nodes[node_id]
                        second_protected = protected | {
                            tuple(node["pickup_position"]), tuple(node["delivery_position"])
                        }
                        second, _ = env._ranked_node_ids(
                            second_protected, [node_id], delivered_nodes, agent_positions
                        )
                        if not second:
                            raise ValueError("A simultaneous replacement has no second successor")
                local_cases += 1
            position_case_count += local_cases
            underfoot_tier_count += local_underfoot
            rows.append({
                "edge_id": edge["edge_id"],
                "delivered_indices": list(delivered),
                "robot_position_cases": local_cases,
                "underfoot_tier_cases": local_underfoot,
            })
    return {
        "version": "warehouse-r41-transition-closure.v1",
        "passed": True,
        "core_edges": len(edges),
        "delivery_cases": len(rows),
        "robot_position_cases": position_case_count,
        "underfoot_tier_cases": underfoot_tier_count,
        "cases_sha256": digest(rows),
    }


def _edge_id(tasks: Sequence[Any]) -> str:
    nodes = sorted(task_node_id(task) for task in tasks)
    for edge in CONFLICT_GRAPH["edges"]:
        if edge["nodes"] == nodes:
            return edge["edge_id"]
    raise ValueError("Scene tasks are absent from the conflict graph")


def make_scene(scene_id: str, split: str, seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed in FORBIDDEN_LEGACY_SEEDS:
        raise ValueError("Legacy 260908-260919 seeds are forbidden")
    env = R41ConflictWarehouseEnv()
    env.reset(seed=seed)
    snapshot = json.loads(canonical(env.snapshot()))
    initial = validate_active_conflict(env.state.tasks)
    replay = successor_replay_audit(snapshot)
    return {
        "id": scene_id,
        "split": split,
        "seed": seed,
        "fingerprint": r41_scene_fingerprint(env),
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "task_geometry_signature": task_geometry_signature(env.state.tasks),
        "initial_edge_id": _edge_id(env.state.tasks),
        "initial_conflict": initial,
        "successor_replay": replay,
        "snapshot": snapshot,
    }


def _generate_pool(
    split: str,
    count: int,
    *,
    seed_start: int,
    forbidden_geometries: set[str],
    forbidden_seeds: set[int],
    used_fingerprints: set[str],
    balanced_edges: bool = False,
    maximum_draws: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if type(count) is not int or count < 1:
        raise ValueError("Every r4.1 generated pool must be non-empty")
    if balanced_edges and count < 60:
        raise ValueError("The play candidate pool requires at least 60 scenes")
    core_edge_ids = sorted(CONFLICT_GRAPH["safe_two_core_edges"])
    quota = {edge: count // len(core_edge_ids) for edge in core_edge_ids}
    for edge in core_edge_ids[: count % len(core_edge_ids)]:
        quota[edge] += 1
    edge_counts = {edge: 0 for edge in core_edge_ids}
    accepted = []
    exclusions: dict[str, int] = {}
    seed = seed_start
    draws = 0
    while len(accepted) < count and draws < maximum_draws:
        current = seed
        seed += 1
        draws += 1
        if current in FORBIDDEN_LEGACY_SEEDS or current in forbidden_seeds:
            exclusions["forbidden_seed"] = exclusions.get("forbidden_seed", 0) + 1
            continue
        scene = make_scene(f"{split}_{len(accepted):04d}", split, current)
        if scene["task_geometry_signature"] in forbidden_geometries:
            exclusions["old_geometry"] = exclusions.get("old_geometry", 0) + 1
            continue
        if scene["fingerprint"] in used_fingerprints:
            exclusions["duplicate_fingerprint"] = exclusions.get("duplicate_fingerprint", 0) + 1
            continue
        if balanced_edges and edge_counts[scene["initial_edge_id"]] >= quota[scene["initial_edge_id"]]:
            exclusions["edge_quota_filled"] = exclusions.get("edge_quota_filled", 0) + 1
            continue
        used_fingerprints.add(scene["fingerprint"])
        forbidden_seeds.add(current)
        edge_counts[scene["initial_edge_id"]] += 1
        accepted.append(scene)
    if len(accepted) != count:
        raise RuntimeError(f"Could not generate the required {split} pool without fallback")
    if balanced_edges and any(edge_counts[edge] != quota[edge] for edge in core_edge_ids):
        raise RuntimeError("Candidate edge quotas were not filled exactly")
    return accepted, {"draws": draws, "accepted": len(accepted), **exclusions,
                      **{f"edge_{key}": value for key, value in edge_counts.items()}}


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(1e-12, (abs(left) + abs(right)) / 2.0)


def select_play_scenes(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(candidates) < 60:
        raise ValueError("At least 60 pre-registered candidates are required")
    by_edge: dict[str, list[Mapping[str, Any]]] = {}
    for scene in candidates:
        by_edge.setdefault(scene["initial_edge_id"], []).append(scene)
    expected_edges = set(CONFLICT_GRAPH["safe_two_core_edges"])
    if set(by_edge) != expected_edges or any(not rows for rows in by_edge.values()):
        raise ValueError("Candidate pool does not cover every safe-core conflict edge")
    chosen = [min(by_edge[edge], key=lambda row: (row["seed"], row["fingerprint"])) for edge in sorted(by_edge)]

    def workload(scene):
        return float(scene["initial_conflict"]["initial_joint_work_steps"])

    def overlap(scene):
        return float(scene["initial_conflict"]["shared_edge_ratio_min"])

    best = None
    # Six distinct geometries are small enough to enumerate exact X/Y pairing
    # and orientation; no greedy fallback can silently change the design.
    for order in permutations(range(6)):
        if order[0] != 0:
            continue
        pairs = ((order[0], order[1]), (order[2], order[3]), (order[4], order[5]))
        if any(left > right for left, right in pairs):
            continue
        if len({index for pair in pairs for index in pair}) != 6:
            continue
        for orientation in range(8):
            x_rows, y_rows = [], []
            for pair_index, pair in enumerate(pairs):
                left, right = pair
                if orientation & (1 << pair_index):
                    left, right = right, left
                x_rows.append(chosen[left])
                y_rows.append(chosen[right])
            pair_work = [abs(workload(x_rows[i]) - workload(y_rows[i])) for i in range(3)]
            pair_overlap = [abs(overlap(x_rows[i]) - overlap(y_rows[i])) for i in range(3)]
            wx, wy = mean(map(workload, x_rows)), mean(map(workload, y_rows))
            ox, oy = mean(map(overlap, x_rows)), mean(map(overlap, y_rows))
            if max(pair_work) > 1.0 or max(pair_overlap) > 0.10:
                continue
            if _relative_difference(wx, wy) > 0.05 or _relative_difference(ox, oy) > 0.10:
                continue
            score = (sum(pair_work) + 10 * sum(pair_overlap)
                     + _relative_difference(wx, wy) + _relative_difference(ox, oy))
            tie = tuple(row["seed"] for row in (*x_rows, *y_rows))
            candidate = (score, tie, x_rows, y_rows)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        raise RuntimeError("No exact balanced X/Y pairing exists; selection does not fall back")
    score, _, x_rows, y_rows = best
    pairs = [[x_rows[index]["id"], y_rows[index]["id"]] for index in range(3)]
    return {
        "version": SELECTION_VERSION,
        "candidate_count": len(candidates),
        "candidate_pool_sha256": digest(candidates),
        "X": [deepcopy(row) for row in x_rows],
        "Y": [deepcopy(row) for row in y_rows],
        "pairs": pairs,
        "balance": {
            "X_mean_initial_work": mean(map(workload, x_rows)),
            "Y_mean_initial_work": mean(map(workload, y_rows)),
            "work_relative_difference": _relative_difference(
                mean(map(workload, x_rows)), mean(map(workload, y_rows))
            ),
            "X_mean_overlap": mean(map(overlap, x_rows)),
            "Y_mean_overlap": mean(map(overlap, y_rows)),
            "overlap_relative_difference": _relative_difference(
                mean(map(overlap, x_rows)), mean(map(overlap, y_rows))
            ),
            "selection_score": score,
        },
    }


def build_manifest(
    old_manifests: Sequence[str | Path],
    *,
    counts: Mapping[str, int] | None = None,
    seed_starts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    counts = {**DEFAULT_COUNTS, **dict(counts or {})}
    seed_starts = {**SPLIT_SEED_STARTS, **dict(seed_starts or {})}
    if set(counts) != set(DEFAULT_COUNTS) or set(seed_starts) != set(SPLIT_SEED_STARTS):
        raise ValueError("r4.1 split schema is fixed")
    if counts["play_candidates"] < 60:
        raise ValueError("r4.1 requires at least 60 play candidates")
    receipts, old_geometries, old_seeds = old_geometry_receipts(old_manifests)
    used_fingerprints: set[str] = set()
    used_seeds = set(old_seeds) | set(FORBIDDEN_LEGACY_SEEDS)
    pools: dict[str, list[dict[str, Any]]] = {}
    scans = {}
    for split in ("train", "conflict_validation", "final_test", "tutorial", "play_candidates"):
        pool, scan = _generate_pool(
            split,
            counts[split],
            seed_start=seed_starts[split],
            forbidden_geometries=old_geometries,
            forbidden_seeds=used_seeds,
            used_fingerprints=used_fingerprints,
            balanced_edges=split == "play_candidates",
        )
        pools[split], scans[split] = pool, scan
    selection = select_play_scenes(pools["play_candidates"])
    play = []
    for index, source in enumerate((*selection["X"], *selection["Y"])):
        row = deepcopy(source)
        row["candidate_source_id"] = row["id"]
        row["id"] = f"play_{index:04d}"
        row["split"] = "play"
        play.append(row)
    selected = {
        **{key: deepcopy(selection[key]) for key in ("version", "candidate_count", "candidate_pool_sha256", "pairs", "balance")},
        "X": [row["id"] for row in play[:3]],
        "Y": [row["id"] for row in play[3:]],
        "candidate_pairs": deepcopy(selection["pairs"]),
    }
    source_paths = (
        Path(__file__),
        SOURCE_ROOT / "env/warehouse_native/r41_conflict.py",
        SOURCE_ROOT / "backend/warehouse_r41_online_runtime.py",
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "contract": deepcopy(CONTRACT),
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph": deepcopy(CONFLICT_GRAPH),
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "configuration": asdict(collaborative_study_config()),
        "reward_config": deepcopy(DEFAULT_REWARD_CONFIG),
        "forbidden_legacy_seeds": list(FORBIDDEN_LEGACY_SEEDS),
        "old_scene_receipts": receipts,
        "old_geometry_signatures": sorted(old_geometries),
        "split_seed_starts": seed_starts,
        "counts": counts,
        "scans": scans,
        "transition_closure": transition_closure_audit(),
        "splits": {
            "train": pools["train"],
            "conflict_validation": pools["conflict_validation"],
            "final_test": pools["final_test"],
            "tutorial": pools["tutorial"],
            "play": play,
        },
        "candidate_pool": pools["play_candidates"],
        "selected_play": selected,
        "source_sha256": {
            str(path.relative_to(SOURCE_ROOT)): file_hash(path) for path in source_paths
        },
    }
    manifest["content_sha256"] = digest(manifest)
    return manifest


def _validate_scene(scene: Mapping[str, Any], expected_split: str, *, replay: bool) -> None:
    required = {
        "id", "split", "seed", "fingerprint", "contract_sha256",
        "conflict_graph_sha256", "task_geometry_signature", "initial_edge_id",
        "initial_conflict", "successor_replay", "snapshot",
    }
    allowed = required | ({"candidate_source_id"} if expected_split == "play" else set())
    if not isinstance(scene, Mapping) or set(scene) != allowed:
        raise ValueError("r4.1 scene schema mismatch")
    if scene["split"] != expected_split or scene["seed"] in FORBIDDEN_LEGACY_SEEDS:
        raise ValueError("r4.1 scene split/seed mismatch")
    if scene["contract_sha256"] != CONTRACT_SHA256 or scene["conflict_graph_sha256"] != CONFLICT_GRAPH_SHA256:
        raise ValueError("r4.1 scene contract binding mismatch")
    env = R41ConflictWarehouseEnv()
    reset_r41_scenario(env, scene)
    metrics = validate_active_conflict(env.state.tasks)
    if canonical(metrics) != canonical(scene["initial_conflict"]):
        raise ValueError("r4.1 initial conflict evidence differs")
    if task_geometry_signature(env.state.tasks) != scene["task_geometry_signature"]:
        raise ValueError("r4.1 task geometry signature differs")
    if _edge_id(env.state.tasks) != scene["initial_edge_id"]:
        raise ValueError("r4.1 graph edge binding differs")
    if replay and successor_replay_audit(scene["snapshot"]) != scene["successor_replay"]:
        raise ValueError("r4.1 successor replay evidence differs")


def validate_conflict_manifest(manifest: Mapping[str, Any], *, replay: bool = True) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("r4.1 manifest must be an object")
    content = deepcopy(dict(manifest))
    claimed = content.pop("content_sha256", None)
    if claimed != digest(content):
        raise ValueError("r4.1 manifest content hash mismatch")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("r4.1 manifest version mismatch")
    if (canonical(manifest.get("contract")) != canonical(CONTRACT)
            or manifest.get("contract_sha256") != CONTRACT_SHA256):
        raise ValueError("r4.1 manifest conflict contract mismatch")
    if (canonical(manifest.get("conflict_graph")) != canonical(CONFLICT_GRAPH)
            or manifest.get("conflict_graph_sha256") != CONFLICT_GRAPH_SHA256):
        raise ValueError("r4.1 manifest task conflict graph mismatch")
    if canonical(manifest.get("configuration")) != canonical(asdict(collaborative_study_config())):
        raise ValueError("r4.1 manifest environment configuration mismatch")
    if manifest.get("reward_config") != DEFAULT_REWARD_CONFIG:
        raise ValueError("r4.1 manifest reward configuration mismatch")
    if manifest.get("forbidden_legacy_seeds") != list(FORBIDDEN_LEGACY_SEEDS):
        raise ValueError("r4.1 legacy seed exclusion mismatch")
    if manifest.get("transition_closure") != transition_closure_audit():
        raise ValueError("r4.1 transition closure evidence mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(OPERATIONAL_SPLITS):
        raise ValueError("r4.1 operational split schema mismatch")
    candidates = manifest.get("candidate_pool")
    if not isinstance(candidates, list) or len(candidates) < 60:
        raise ValueError("r4.1 candidate pool contains fewer than 60 scenes")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(DEFAULT_COUNTS):
        raise ValueError("r4.1 manifest split counts are incomplete")
    expected_counts = {
        "train": len(splits["train"]),
        "conflict_validation": len(splits["conflict_validation"]),
        "final_test": len(splits["final_test"]),
        "tutorial": len(splits["tutorial"]),
        "play_candidates": len(candidates),
    }
    if dict(counts) != expected_counts:
        raise ValueError("r4.1 declared split counts differ from saved scenes")

    source_paths = (
        Path(__file__),
        SOURCE_ROOT / "env/warehouse_native/r41_conflict.py",
        SOURCE_ROOT / "backend/warehouse_r41_online_runtime.py",
    )
    expected_sources = {
        str(path.relative_to(SOURCE_ROOT)): file_hash(path) for path in source_paths
    }
    if manifest.get("source_sha256") != expected_sources:
        raise ValueError("r4.1 scene producer source closure changed")
    receipts = manifest.get("old_scene_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("r4.1 old-scene exclusion receipts are missing")
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "path", "sha256", "play_scene_count", "play_geometry_sha256"
        }:
            raise ValueError("r4.1 old-scene receipt schema mismatch")
        path = Path(receipt["path"])
        if not path.is_file() or file_hash(path) != receipt["sha256"]:
            raise ValueError("r4.1 old-scene source changed or disappeared")

    all_operational = []
    for split in OPERATIONAL_SPLITS:
        rows = splits[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"r4.1 split {split} is empty")
        for row in rows:
            _validate_scene(row, split, replay=replay)
            all_operational.append(row)
    for row in candidates:
        _validate_scene(row, "play_candidates", replay=replay)

    seeds = [row["seed"] for row in all_operational]
    fingerprints = [row["fingerprint"] for row in all_operational]
    successor_states = [
        row["snapshot"]["r41_conflict"]["successor_state_sha256"]
        for row in all_operational
    ]
    # Play rows intentionally originate in the candidate pool, but every
    # operational split is otherwise strictly disjoint.
    if len(seeds) != len(set(seeds)) or len(fingerprints) != len(set(fingerprints)):
        raise ValueError("r4.1 operational splits are not seed/fingerprint disjoint")
    if len(successor_states) != len(set(successor_states)):
        raise ValueError("r4.1 operational splits share a successor sampler state")
    non_play = [row for split in OPERATIONAL_SPLITS if split != "play" for row in splits[split]]
    candidate_seeds = {row["seed"] for row in candidates}
    candidate_fingerprints = {row["fingerprint"] for row in candidates}
    if any(row["seed"] in candidate_seeds or row["fingerprint"] in candidate_fingerprints for row in non_play):
        raise ValueError("r4.1 candidates leak into training/validation/final/tutorial")
    old_geometries = set(manifest.get("old_geometry_signatures", []))
    if any(row["task_geometry_signature"] in old_geometries for row in (*all_operational, *candidates)):
        raise ValueError("r4.1 reused an old play geometry")
    if any(row["seed"] in FORBIDDEN_LEGACY_SEEDS for row in (*all_operational, *candidates)):
        raise ValueError("r4.1 reused a forbidden 260908-260919 seed")

    selected = manifest.get("selected_play")
    if not isinstance(selected, Mapping) or len(selected.get("X", [])) != 3 or len(selected.get("Y", [])) != 3:
        raise ValueError("r4.1 selected play must contain X/Y three scenes each")
    play_by_id = {row["id"]: row for row in splits["play"]}
    if set((*selected["X"], *selected["Y"])) != set(play_by_id):
        raise ValueError("r4.1 selected play IDs differ from play split")
    if len({row["initial_edge_id"] for row in splits["play"]}) != 6:
        raise ValueError("r4.1 play scenes must cover six distinct conflict geometries")
    candidate_by_id = {row["id"]: row for row in candidates}
    for row in splits["play"]:
        source = candidate_by_id.get(row["candidate_source_id"])
        comparable = deepcopy(dict(row))
        comparable["id"] = comparable.pop("candidate_source_id")
        comparable["split"] = "play_candidates"
        if source != comparable:
            raise ValueError("r4.1 play scene differs from its registered candidate")
    regenerated = select_play_scenes(candidates)
    expected_selected = {
        **{key: deepcopy(regenerated[key]) for key in (
            "version", "candidate_count", "candidate_pool_sha256", "pairs", "balance"
        )},
        "X": [row["id"] for row in splits["play"][:3]],
        "Y": [row["id"] for row in splits["play"][3:]],
        "candidate_pairs": deepcopy(regenerated["pairs"]),
    }
    if selected != expected_selected:
        raise ValueError("r4.1 saved X/Y selection is not the deterministic selection")

    spawn_counts = {
        split: sum(
            int(row["successor_replay"]["spawned_on_agent_endpoint"])
            for row in splits[split]
        )
        for split in OPERATIONAL_SPLITS
    }
    spawn_counts["play_candidates"] = sum(
        int(row["successor_replay"]["spawned_on_agent_endpoint"])
        for row in candidates
    )
    recreation_counts = {
        split: sum(
            int(row["successor_replay"]["immediate_task_recreation"])
            for row in splits[split]
        )
        for split in OPERATIONAL_SPLITS
    }
    recreation_counts["play_candidates"] = sum(
        int(row["successor_replay"]["immediate_task_recreation"])
        for row in candidates
    )
    endpoint_counts = {
        split: sum(
            int(row["successor_replay"]["new_endpoint_on_agent"])
            for row in splits[split]
        )
        for split in OPERATIONAL_SPLITS
    }
    endpoint_counts["play_candidates"] = sum(
        int(row["successor_replay"]["new_endpoint_on_agent"])
        for row in candidates
    )
    pickup_counts = {
        split: sum(
            int(row["successor_replay"]["new_pickup_on_agent"])
            for row in splits[split]
        )
        for split in OPERATIONAL_SPLITS
    }
    pickup_counts["play_candidates"] = sum(
        int(row["successor_replay"]["new_pickup_on_agent"])
        for row in candidates
    )
    delivery_counts = {
        split: sum(
            int(row["successor_replay"]["new_delivery_on_agent"])
            for row in splits[split]
        )
        for split in OPERATIONAL_SPLITS
    }
    delivery_counts["play_candidates"] = sum(
        int(row["successor_replay"]["new_delivery_on_agent"])
        for row in candidates
    )
    if sum(recreation_counts.values()) != 0 or sum(pickup_counts.values()) != 0:
        raise ValueError("r4.1 successor replay recreated a task or spawned a pickup on a robot")

    return {
        "version": "warehouse-r41-conflict-manifest-validation.v1",
        "passed": True,
        "manifest_content_sha256": claimed,
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "candidate_count": len(candidates),
        "operational_counts": {split: len(splits[split]) for split in OPERATIONAL_SPLITS},
        "successor_replay_enabled": replay,
        "split_seed_disjoint": True,
        "split_fingerprint_disjoint": True,
        "split_successor_state_disjoint": True,
        "old_geometry_reused": 0,
        "forbidden_seed_reused": 0,
        "spawned_on_agent_endpoint": spawn_counts,
        "spawned_on_agent_endpoint_total": sum(spawn_counts.values()),
        "immediate_task_recreation": recreation_counts,
        "immediate_task_recreation_total": sum(recreation_counts.values()),
        "new_endpoint_on_agent": endpoint_counts,
        "new_endpoint_on_agent_total": sum(endpoint_counts.values()),
        "new_pickup_on_agent": pickup_counts,
        "new_pickup_on_agent_total": sum(pickup_counts.values()),
        "new_delivery_on_agent": delivery_counts,
        "new_delivery_on_agent_total": sum(delivery_counts.values()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--old-manifest", action="append", required=True)
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--report", required=True)
    validate.add_argument("--skip-successor-replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        manifest = build_manifest(args.old_manifest)
        write_json(output / "task_conflict_graph.json", CONFLICT_GRAPH)
        write_json(output / "manifest.json", manifest)
        report = validate_conflict_manifest(manifest, replay=True)
        report["manifest_file_sha256"] = file_hash(output / "manifest.json")
        report["task_conflict_graph_file_sha256"] = file_hash(output / "task_conflict_graph.json")
        write_json(output / "validation.json", report)
        print(canonical(report))
        return 0
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = validate_conflict_manifest(manifest, replay=not args.skip_successor_replay)
    report["manifest_file_sha256"] = file_hash(args.manifest)
    write_json(args.report, report)
    print(canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
