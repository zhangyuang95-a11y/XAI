"""Versioned strict-conflict scene generation for the r4.1 diagnostic pilot.

Historical r4.1 and diagnostic-v1 manifests remain immutable.  This producer
draws fresh non-play scenes and admits each one only after replaying the exact
Actor/partner/branch workload that will consume its split.  It also creates up
to three new candidate batches, each with exactly 120 deterministic states for
each of the six conflict-edge families.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from itertools import permutations
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
)
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
    FROZEN_ACTOR_SHA256,
    VERSION as WORKLOAD_SCREEN_VERSION,
    contract as workload_contract,
    load_frozen_actor,
    replay_and_compare as replay_workload_and_compare,
    screen_scene,
    source_sha256 as workload_source_sha256,
    validate_receipt as validate_workload_receipt,
)
from env.warehouse.domain import DeliveryTask, collaborative_study_config
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES,
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
    active_edge_id,
    conflict_family_id,
    diagnostic_contract_receipt,
    diagnostic_graph_invariant_audit,
    diagnostic_scene_fingerprint,
    digest,
    reset_diagnostic_scenario,
    validate_diagnostic_active_conflict,
)
from env.warehouse_native.r41_conflict import canonical, task_node_id


VERSION = "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
VALIDATION_VERSION = "warehouse-r41-diagnostic-conflict-scene-validation.v3"
BATCH_SEED_STARTS = (48_100_000, 48_200_000, 48_300_000)
PER_FAMILY_PER_BATCH = 120
MAXIMUM_BATCHES = 3
SOURCE_SPLITS = ("train", "conflict_validation", "final_test", "tutorial")
BASE_SPLITS = (*SOURCE_SPLITS, "question_bank")
EXPECTED_BASE_COUNTS = {
    "train": 128,
    "conflict_validation": 64,
    "final_test": 64,
    "tutorial": 1,
    "question_bank": 36,
}
SPLIT_SEED_STARTS = {
    "train": 49_100_000,
    "conflict_validation": 49_200_000,
    "final_test": 49_300_000,
    "tutorial": 49_400_000,
    "question_bank": 49_500_000,
}
MAXIMUM_SPLIT_DRAWS = 100_000
FAMILY_IDS = tuple(row["family_id"] for row in CONFLICT_FAMILIES)
ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def producer_sources() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
    )
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Diagnostic scene producer source is missing")
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return dict(sorted(result.items()))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _edge_geometry_signature(tasks: Sequence[Any]) -> str:
    return digest(
        sorted(
            (
                tuple(task.pickup_position),
                tuple(task.delivery_position),
            )
            for task in tasks
        )
    )


def _initial_public_joint_work(env: R41DiagnosticConflictWarehouseEnv) -> int:
    agents = tuple(env.state.agents)
    tasks = tuple(env.state.tasks)
    return int(
        min(
            sum(
                shortest_path_distance(
                    agents[agent_index].position,
                    tasks[task_index].pickup_position,
                    env.config.map_layout_id,
                )
                + shortest_path_distance(
                    tasks[task_index].pickup_position,
                    tasks[task_index].delivery_position,
                    env.config.map_layout_id,
                )
                for agent_index, task_index in enumerate(assignment)
            )
            for assignment in permutations(range(2))
        )
    )


def _successor_stream_seed(seed: int, batch_index: int, family_id: str) -> int:
    return int(
        digest(
            {
                "version": VERSION,
                "seed": seed,
                "batch_index": batch_index,
                "family_id": family_id,
            }
        )[:16],
        16,
    )


def _candidate_start_state(
    env: R41DiagnosticConflictWarehouseEnv,
    *,
    seed: int,
    batch_index: int,
    family_id: str,
) -> None:
    """Create deterministic public-state diversity without changing tasks."""
    rng = random.Random(
        int(
            digest(
                {
                    "version": VERSION,
                    "purpose": "candidate-public-start",
                    "seed": seed,
                    "batch_index": batch_index,
                    "family_id": family_id,
                }
            )[:16],
            16,
        )
    )
    state = env.get_state()
    endpoints = {
        tuple(position)
        for task in state.tasks
        for position in (task.pickup_position, task.delivery_position)
    }
    allowed = sorted(
        position
        for position in env.layout.passable_positions
        if position not in endpoints
        and position != env.layout.charger_position
        and position not in env.layout.task_endpoint_exclusions
    )
    if len(allowed) < 2:
        raise RuntimeError("Diagnostic scene has fewer than two clear robot starts")
    starts = rng.sample(allowed, 2)
    headings = tuple(ACTIONS[:-1])
    for index, agent in enumerate(state.agents):
        agent.position = starts[index]
        agent.heading = rng.choice(headings)
        agent.battery = 100.0
        agent.active = True
        agent.carrying_task_id = None
        agent.last_action = "WAIT"
        agent.last_executed_action = "WAIT"
        agent.recent_positions = ()
        agent.recent_goal_types = ()
        agent.navigation_goal_position = agent.position
    for task in state.tasks:
        task.status = "available"
        task.carrier_agent_id = None
        task.claimed_frame = None
        task.claimed_battery = None
        task.delivered_frame = None
    state.frame = 0
    state.total_deliveries = 0
    state.shutdown_count = 0
    state.user_score = 0.0
    state.terminated = False
    state.truncated = False
    state.terminal_reason = None
    env.set_state(state)
    stream_seed = _successor_stream_seed(seed, batch_index, family_id)
    env.set_rng_state(random.Random(stream_seed).getstate())


def _force_delivery(
    env: R41DiagnosticConflictWarehouseEnv, delivered_indices: Sequence[int]
) -> None:
    state = env.get_state()
    for task in state.tasks:
        task.status = "available"
        task.carrier_agent_id = None
        task.claimed_frame = None
        task.claimed_battery = None
        task.delivered_frame = None
    reserved = set()
    for index, agent in enumerate(state.agents):
        agent.battery = 100.0
        agent.active = True
        agent.carrying_task_id = None
        agent.last_action = "WAIT"
        agent.last_executed_action = "WAIT"
        if index < len(delivered_indices):
            task = state.tasks[int(delivered_indices[index])]
            task.status = "carried"
            task.carrier_agent_id = agent.agent_id
            task.claimed_frame = state.frame
            task.claimed_battery = 100.0
            agent.position = task.delivery_position
            agent.carrying_task_id = task.task_id
            reserved.add(agent.position)
        else:
            choices = sorted(
                position
                for position in env.layout.passable_positions
                if position not in reserved
                and position not in {task.pickup_position for task in state.tasks}
            )
            agent.position = choices[-1]
            reserved.add(agent.position)
    state.terminated = False
    state.truncated = False
    state.terminal_reason = None
    env.set_state(state)


def successor_replay_audit(
    snapshot: Mapping[str, Any], *, cycles: int = 6
) -> dict[str, Any]:
    if type(cycles) is not int or cycles < 3:
        raise ValueError("Diagnostic successor replay requires at least three cycles")
    env = R41DiagnosticConflictWarehouseEnv()
    env.restore(deepcopy(snapshot))
    transcript = []
    patterns = ((0,), (1,), (0, 1))
    for cycle in range(cycles):
        _force_delivery(env, patterns[cycle % len(patterns)])
        forced = env.snapshot()
        first = R41DiagnosticConflictWarehouseEnv()
        second = R41DiagnosticConflictWarehouseEnv()
        first.restore(deepcopy(forced))
        second.restore(deepcopy(forced))
        one = first.step({"robot_1": "WAIT", "robot_2": "WAIT"})[-1]
        two = second.step({"robot_1": "WAIT", "robot_2": "WAIT"})[-1]
        created = one["r41_diagnostic_conflict"]["created"]
        row = {
            "cycle": cycle,
            "delivered_indices": list(patterns[cycle % len(patterns)]),
            "replay_equal": first.snapshot() == second.snapshot(),
            "active_conflict_passed": validate_diagnostic_active_conflict(
                first.state.tasks
            )["passed"],
            "new_pickup_on_agent": sum(bool(item["new_pickup_on_agent"]) for item in created),
            "new_delivery_on_agent": sum(bool(item["new_delivery_on_agent"]) for item in created),
            "new_endpoint_on_agent": sum(bool(item["new_endpoint_on_agent"]) for item in created),
            "immediate_task_recreation": sum(bool(item["immediate_task_recreation"]) for item in created),
            "ordinary_sampler_fallback": one["r41_diagnostic_conflict"][
                "ordinary_sampler_fallback"
            ],
            "after_family_id": conflict_family_id(first.state.tasks),
            "after_snapshot_sha256": digest(first.snapshot()),
        }
        if one["requested_actions"] != two["requested_actions"]:
            row["replay_equal"] = False
        transcript.append(row)
        env = first
    return {
        "version": "warehouse-r41-diagnostic-successor-replay.v1",
        "cycles": cycles,
        "passed": all(
            row["replay_equal"]
            and row["active_conflict_passed"]
            and row["new_pickup_on_agent"] == 0
            and row["new_delivery_on_agent"] == 0
            and row["new_endpoint_on_agent"] == 0
            and row["immediate_task_recreation"] == 0
            and row["ordinary_sampler_fallback"] is False
            for row in transcript
        ),
        "new_pickup_on_agent": sum(row["new_pickup_on_agent"] for row in transcript),
        "new_delivery_on_agent": sum(row["new_delivery_on_agent"] for row in transcript),
        "new_endpoint_on_agent": sum(row["new_endpoint_on_agent"] for row in transcript),
        "immediate_task_recreation": sum(row["immediate_task_recreation"] for row in transcript),
        "transcript_sha256": digest(transcript),
    }


def _scene_from_environment(
    env: R41DiagnosticConflictWarehouseEnv,
    *,
    scene_id: str,
    split: str,
    seed: int,
    batch_index: int | None,
) -> dict[str, Any]:
    snapshot = json.loads(canonical(env.snapshot()))
    metrics = validate_diagnostic_active_conflict(env.state.tasks)
    family_id = metrics["conflict_family_id"]
    result = {
        "id": scene_id,
        "split": split,
        "seed": seed,
        "fingerprint": diagnostic_scene_fingerprint(env),
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "family_id": family_id,
        "initial_edge_id": active_edge_id(env.state.tasks),
        "task_geometry_signature": _edge_geometry_signature(env.state.tasks),
        "initial_conflict": metrics,
        "initial_public_joint_work_steps": _initial_public_joint_work(env),
        "initial_robot_positions": [list(agent.position) for agent in env.state.agents],
        "successor_stream_seed": _successor_stream_seed(
            seed, -1 if batch_index is None else batch_index, family_id
        ),
        "snapshot": snapshot,
    }
    if batch_index is not None:
        result["batch_index"] = batch_index
    return result


def make_scene(
    scene_id: str,
    split: str,
    seed: int,
    *,
    batch_index: int | None = None,
    candidate_start: bool = False,
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("Diagnostic scene seed must be a non-negative integer")
    env = R41DiagnosticConflictWarehouseEnv()
    env.reset(seed=seed)
    family_id = conflict_family_id(env.state.tasks)
    if candidate_start:
        if batch_index is None:
            raise ValueError("Candidate starts require a batch index")
        _candidate_start_state(
            env,
            seed=seed,
            batch_index=batch_index,
            family_id=family_id,
        )
    return _scene_from_environment(
        env,
        scene_id=scene_id,
        split=split,
        seed=seed,
        batch_index=batch_index,
    )


def generate_candidate_batch(
    batch_index: int,
    *,
    per_family: int = PER_FAMILY_PER_BATCH,
    maximum_draws: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(batch_index) is not int or not 0 <= batch_index < MAXIMUM_BATCHES:
        raise ValueError("Diagnostic candidate batch index must be zero, one, or two")
    if type(per_family) is not int or per_family < 1:
        raise ValueError("Diagnostic family quota must be positive")
    counts = Counter()
    candidates = []
    seen_fingerprints = set()
    seed = BATCH_SEED_STARTS[batch_index]
    draws = 0
    while any(counts[family] < per_family for family in FAMILY_IDS):
        if draws >= maximum_draws:
            raise RuntimeError("Could not fill every diagnostic conflict-family quota")
        current = seed
        seed += 1
        draws += 1
        env = R41DiagnosticConflictWarehouseEnv()
        env.reset(seed=current)
        family_id = conflict_family_id(env.state.tasks)
        if counts[family_id] >= per_family:
            continue
        _candidate_start_state(
            env,
            seed=current,
            batch_index=batch_index,
            family_id=family_id,
        )
        scene = _scene_from_environment(
            env,
            scene_id=f"diagnostic_b{batch_index}_{family_id}_{counts[family_id]:03d}",
            split="play_candidates",
            seed=current,
            batch_index=batch_index,
        )
        if scene["fingerprint"] in seen_fingerprints:
            continue
        seen_fingerprints.add(scene["fingerprint"])
        counts[family_id] += 1
        candidates.append(scene)
    candidates.sort(key=lambda row: (row["family_id"], row["seed"]))
    report = {
        "batch_index": batch_index,
        "seed_start": BATCH_SEED_STARTS[batch_index],
        "draws": draws,
        "candidate_count": len(candidates),
        "per_family": dict(sorted(counts.items())),
        "candidate_fingerprints_sha256": digest(
            [row["fingerprint"] for row in candidates]
        ),
    }
    return candidates, report


def _family_quotas(count: int) -> dict[str, int]:
    if type(count) is not int or count < len(FAMILY_IDS):
        raise ValueError("A balanced diagnostic split requires at least six scenes")
    quotient, remainder = divmod(count, len(FAMILY_IDS))
    return {
        family: quotient + int(index < remainder)
        for index, family in enumerate(FAMILY_IDS)
    }


def _generate_screened_split(
    split: str,
    *,
    seed_start: int,
    count: int,
    actor,
    maximum_draws: int = MAXIMUM_SPLIT_DRAWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in BASE_SPLITS:
        raise ValueError("Unknown diagnostic base split")
    quotas = ({family: 1_000_000 for family in FAMILY_IDS}
              if split == "tutorial" else _family_quotas(count))
    accepted_by_family = Counter()
    rejected_by_family = Counter()
    failure_types = Counter()
    failure_examples = []
    rows: list[dict[str, Any]] = []
    seed = seed_start
    draws = 0
    started = time.perf_counter()
    while len(rows) < count:
        if draws >= maximum_draws:
            raise RuntimeError(
                f"Could not fill workload-safe {split} split after {draws} draws"
            )
        current = seed
        seed += 1
        draws += 1
        scene = make_scene(
            f"diagnostic_{split}_{len(rows):04d}", split, current
        )
        family = scene["family_id"]
        if accepted_by_family[family] >= quotas[family]:
            continue
        receipt = screen_scene(
            scene,
            split=split,
            scene_index=len(rows),
            actor=(None if split == "tutorial" else actor),
            train_count=EXPECTED_BASE_COUNTS["train"],
        )
        if receipt["passed"] is not True:
            rejected_by_family[family] += 1
            failure = receipt["failure"]
            failure_types[(failure["workload"], failure["partner"])] += 1
            if len(failure_examples) < 12:
                failure_examples.append(
                    {
                        "seed": current,
                        "family_id": family,
                        "scene_index": len(rows),
                        "failure": deepcopy(failure),
                        "failed_receipt_sha256": receipt["receipt_sha256"],
                    }
                )
            continue
        scene["workload_screen"] = receipt
        validate_workload_receipt(
            receipt,
            scene=scene,
            split=split,
            scene_index=len(rows),
        )
        accepted_by_family[family] += 1
        rows.append(scene)
        if len(rows) == count or len(rows) % 8 == 0:
            elapsed = time.perf_counter() - started
            print(
                canonical(
                    {
                        "event": "diagnostic_workload_screen_progress",
                        "split": split,
                        "accepted": len(rows),
                        "target": count,
                        "rejected_workload_unsafe": sum(
                            rejected_by_family.values()
                        ),
                        "draws": draws,
                        "elapsed_seconds": elapsed,
                        "accepted_per_second": len(rows) / max(elapsed, 1e-9),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    report = {
        "split": split,
        "seed_start": seed_start,
        "last_seed_examined": seed - 1,
        "draws": draws,
        "accepted": len(rows),
        "accepted_by_family": dict(sorted(accepted_by_family.items())),
        "rejected_workload_unsafe": sum(rejected_by_family.values()),
        "rejected_by_family": dict(sorted(rejected_by_family.items())),
        "failure_types": {
            f"{workload}:{partner}": value
            for (workload, partner), value in sorted(failure_types.items())
        },
        "failure_examples": failure_examples,
        "accepted_receipts_sha256": digest(
            [row["workload_screen"]["receipt_sha256"] for row in rows]
        ),
    }
    return rows, report


def build_manifest(
    source_manifest_path: str | Path,
    *,
    actor_path: str | Path,
    batch_count: int = 1,
    per_family: int = PER_FAMILY_PER_BATCH,
) -> dict[str, Any]:
    if type(batch_count) is not int or not 1 <= batch_count <= MAXIMUM_BATCHES:
        raise ValueError("Diagnostic generation supports one to three batches")
    source_path = Path(source_manifest_path).expanduser().resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source.get("splits"), Mapping):
        raise ValueError("Immutable r4.1 source manifest has no split map")
    actor_path = Path(actor_path).expanduser().resolve()
    actor = load_frozen_actor(actor_path)
    splits: dict[str, list[dict[str, Any]]] = {}
    workload_generation_reports = []
    # Order is part of the RCPD RNG contract: validation indices are offset by
    # the final accepted train count, never by rejected draws.
    for split in BASE_SPLITS:
        rows, report = _generate_screened_split(
            split,
            seed_start=SPLIT_SEED_STARTS[split],
            count=EXPECTED_BASE_COUNTS[split],
            actor=actor,
        )
        splits[split] = rows
        workload_generation_reports.append(report)
    batches = []
    batch_reports = []
    for batch_index in range(batch_count):
        rows, report = generate_candidate_batch(
            batch_index, per_family=per_family
        )
        batches.append(rows)
        batch_reports.append(report)
    graph_invariant = diagnostic_graph_invariant_audit()
    manifest: dict[str, Any] = {
        "version": VERSION,
        "configuration": asdict(collaborative_study_config()),
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "diagnostic_graph_invariant_audit": graph_invariant,
        "diagnostic_graph_invariant_audit_sha256": digest(graph_invariant),
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "conflict_families": deepcopy(CONFLICT_FAMILIES),
        "frozen_actor": {
            "sha256": actor.artifact_sha256,
            "actor_parameters_sha256": actor.metadata.get(
                "actor_parameters_sha256"
            ),
        },
        "workload_screen": {
            "version": WORKLOAD_SCREEN_VERSION,
            "contract": workload_contract(),
            "contract_sha256": WORKLOAD_CONTRACT_SHA256,
            "source_sha256": workload_source_sha256(),
        },
        "producer_sources": producer_sources(),
        "producer_sources_sha256": digest(producer_sources()),
        "source_r41_manifest": {
            "path": str(source_path),
            "file_sha256": sha256(source_bytes).hexdigest(),
            "content_sha256": source.get("content_sha256"),
            "version": source.get("version"),
        },
        "protocol": {
            "per_family_per_batch": per_family,
            "batch_count": batch_count,
            "maximum_batches": MAXIMUM_BATCHES,
            "candidate_public_start_positions": "deterministic clear passable cells",
            "candidate_initial_battery": [100.0, 100.0],
            "candidate_successor_stream": "unique deterministic stream bound to scene seed, batch, and family",
            "successor_graph": "diagnostic v2 robust fixed-point closure",
            "successor_endpoint_occupancy": "pickup and delivery both clear of both post-motion robots",
            "immediate_task_recreation": False,
            "ordinary_task_fallback": False,
            "participant_data_read": False,
            "final_test_used_for_selection": False,
            "base_scene_policy": "fresh deterministic draws admitted only after exact split workload replay",
            "base_split_seed_starts": deepcopy(SPLIT_SEED_STARTS),
            "workload_unsafe_scene_policy": "reject and advance seed; strict sampler remains fail closed",
        },
        "splits": splits,
        "workload_generation_reports": workload_generation_reports,
        "candidate_batches": batches,
        "batch_reports": batch_reports,
    }
    manifest["content_sha256"] = digest(manifest)
    return manifest


def _validate_scene(
    scene: Mapping[str, Any], *, expected_split: str, expected_batch: int | None,
    expected_index: int | None = None,
) -> None:
    required = {
        "id",
        "split",
        "seed",
        "fingerprint",
        "diagnostic_contract_sha256",
        "diagnostic_conflict_graph_sha256",
        "conflict_families_sha256",
        "family_id",
        "initial_edge_id",
        "task_geometry_signature",
        "initial_conflict",
        "initial_public_joint_work_steps",
        "initial_robot_positions",
        "successor_stream_seed",
        "snapshot",
    }
    optional = {"batch_index"} if expected_batch is not None else {"workload_screen"}
    if not required <= set(scene) or set(scene) - required - optional:
        raise ValueError("Diagnostic scene schema differs")
    if scene["split"] != expected_split:
        raise ValueError("Diagnostic scene split differs")
    if scene["diagnostic_conflict_graph_sha256"] != DIAGNOSTIC_CONFLICT_GRAPH_SHA256:
        raise ValueError("Diagnostic scene robust graph differs")
    if expected_batch is not None and scene.get("batch_index") != expected_batch:
        raise ValueError("Diagnostic candidate batch differs")
    if expected_batch is None:
        if expected_index is None:
            raise ValueError("Diagnostic base scene requires its registered index")
        validate_workload_receipt(
            scene.get("workload_screen"),
            scene=scene,
            split=expected_split,
            scene_index=expected_index,
        )
    env = R41DiagnosticConflictWarehouseEnv()
    reset_diagnostic_scenario(env, scene)
    metrics = validate_diagnostic_active_conflict(env.state.tasks)
    if (
        scene["initial_edge_id"] != metrics["initial_edge_id"]
        or scene["family_id"] != metrics["conflict_family_id"]
        or scene["task_geometry_signature"]
        != _edge_geometry_signature(env.state.tasks)
        or canonical(scene["initial_conflict"]) != canonical(metrics)
        or scene["initial_public_joint_work_steps"] != _initial_public_joint_work(env)
        or scene["initial_robot_positions"]
        != [list(agent.position) for agent in env.state.agents]
    ):
        raise ValueError("Diagnostic scene geometry binding differs")


def validate_diagnostic_manifest(
    manifest: Mapping[str, Any], *, replay: bool = True,
    workload_actor_path: str | Path | None = None,
    replay_workloads: bool = False,
) -> dict[str, Any]:
    if manifest.get("version") != VERSION:
        raise ValueError("Diagnostic conflict manifest version differs")
    content = deepcopy(dict(manifest))
    claimed = content.pop("content_sha256", None)
    if claimed != digest(content):
        raise ValueError("Diagnostic conflict manifest content hash differs")
    graph_invariant = diagnostic_graph_invariant_audit()
    if (
        manifest.get("diagnostic_contract_version") != DIAGNOSTIC_CONTRACT_VERSION
        or manifest.get("diagnostic_contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
        or manifest.get("diagnostic_conflict_graph_sha256")
        != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
        or canonical(manifest.get("diagnostic_graph_invariant_audit"))
        != canonical(graph_invariant)
        or manifest.get("diagnostic_graph_invariant_audit_sha256")
        != digest(graph_invariant)
        or graph_invariant.get("passed") is not True
        or manifest.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256
        or canonical(manifest.get("conflict_families"))
        != canonical(CONFLICT_FAMILIES)
    ):
        raise ValueError("Diagnostic conflict manifest contract differs")
    actor_binding = manifest.get("frozen_actor")
    workload_binding = manifest.get("workload_screen")
    sources = producer_sources()
    if (
        not isinstance(actor_binding, Mapping)
        or actor_binding.get("sha256") != FROZEN_ACTOR_SHA256
        or not isinstance(actor_binding.get("actor_parameters_sha256"), str)
        or not isinstance(workload_binding, Mapping)
        or workload_binding.get("version") != WORKLOAD_SCREEN_VERSION
        or workload_binding.get("contract") != workload_contract()
        or workload_binding.get("contract_sha256") != WORKLOAD_CONTRACT_SHA256
        or workload_binding.get("source_sha256") != workload_source_sha256()
        or manifest.get("producer_sources") != sources
        or manifest.get("producer_sources_sha256") != digest(sources)
    ):
        raise ValueError("Diagnostic workload/Actor binding differs")
    if replay_workloads and workload_actor_path is None:
        raise ValueError("Exact Actor path is required to replay workload screens")
    workload_actor = (
        load_frozen_actor(workload_actor_path)
        if replay_workloads else None
    )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(BASE_SPLITS):
        raise ValueError("Diagnostic base split schema differs")
    seen_seeds = set()
    seen_fingerprints = set()
    for split in BASE_SPLITS:
        rows = splits[split]
        if len(rows) != EXPECTED_BASE_COUNTS[split]:
            raise ValueError("Diagnostic base split count differs")
        for scene_index, scene in enumerate(rows):
            _validate_scene(
                scene,
                expected_split=split,
                expected_batch=None,
                expected_index=scene_index,
            )
            if replay_workloads:
                replay_workload_and_compare(
                    scene,
                    split=split,
                    scene_index=scene_index,
                    actor=(None if split == "tutorial" else workload_actor),
                    train_count=EXPECTED_BASE_COUNTS["train"],
                )
            if scene["seed"] in seen_seeds or scene["fingerprint"] in seen_fingerprints:
                raise ValueError("Diagnostic base splits overlap")
            seen_seeds.add(scene["seed"])
            seen_fingerprints.add(scene["fingerprint"])
    batches = manifest.get("candidate_batches")
    reports = manifest.get("batch_reports")
    expected_batch_count = manifest.get("protocol", {}).get("batch_count")
    per_family = manifest.get("protocol", {}).get("per_family_per_batch")
    if (
        not isinstance(batches, list)
        or not isinstance(reports, list)
        or len(batches) != expected_batch_count
        or len(reports) != expected_batch_count
        or not 1 <= len(batches) <= MAXIMUM_BATCHES
    ):
        raise ValueError("Diagnostic candidate batch schema differs")
    replay_rows = []
    for batch_index, rows in enumerate(batches):
        counts = Counter(row.get("family_id") for row in rows)
        if (
            len(rows) != per_family * len(FAMILY_IDS)
            or counts != Counter({family: per_family for family in FAMILY_IDS})
        ):
            raise ValueError("Diagnostic candidate family quota differs")
        for scene in rows:
            _validate_scene(
                scene, expected_split="play_candidates", expected_batch=batch_index
            )
            if scene["seed"] in seen_seeds or scene["fingerprint"] in seen_fingerprints:
                raise ValueError("Diagnostic candidate pools overlap another split")
            seen_seeds.add(scene["seed"])
            seen_fingerprints.add(scene["fingerprint"])
        if replay:
            # Exhaustive dynamics are performed by the full selector.  This
            # bounded replay confirms each immutable batch's strict sampler.
            for family in FAMILY_IDS:
                scene = next(row for row in rows if row["family_id"] == family)
                replay_rows.append(
                    {
                        "scene_id": scene["id"],
                        "family_id": family,
                        **successor_replay_audit(scene["snapshot"]),
                    }
                )
    if replay and not all(row["passed"] for row in replay_rows):
        raise ValueError("Diagnostic strict successor replay failed")
    generation_reports = manifest.get("workload_generation_reports")
    if (
        not isinstance(generation_reports, list)
        or len(generation_reports) != len(BASE_SPLITS)
        or [row.get("split") for row in generation_reports] != list(BASE_SPLITS)
        or any(
            row.get("accepted") != EXPECTED_BASE_COUNTS[row.get("split")]
            or row.get("rejected_workload_unsafe", -1) < 0
            or row.get("draws", -1) < row.get("accepted", 0)
            or row.get("accepted_receipts_sha256")
            != digest(
                [
                    scene["workload_screen"]["receipt_sha256"]
                    for scene in splits[row["split"]]
                ]
            )
            for row in generation_reports
        )
    ):
        raise ValueError("Diagnostic workload generation evidence differs")
    question_metrics = [
        scene["workload_screen"]["metrics"] for scene in splits["question_bank"]
    ]
    next_actions = {
        action for metrics in question_metrics for action in metrics["next_actions"]
    }
    wait_displacements = {
        tuple(value)
        for metrics in question_metrics
        for value in metrics["wait_displacements"]
    }
    if len(next_actions) < 4 or len(wait_displacements) < 4:
        raise ValueError("Diagnostic question split lacks four outcome classes")
    return {
        "version": VALIDATION_VERSION,
        "passed": True,
        "manifest_content_sha256": claimed,
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "diagnostic_graph_invariant_audit": deepcopy(graph_invariant),
        "diagnostic_graph_invariant_audit_sha256": digest(graph_invariant),
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "producer_sources_sha256": digest(sources),
        "base_split_counts": {key: len(splits[key]) for key in BASE_SPLITS},
        "candidate_batch_count": len(batches),
        "candidate_count": sum(map(len, batches)),
        "per_family_per_batch": per_family,
        "unique_seed_count": len(seen_seeds),
        "unique_fingerprint_count": len(seen_fingerprints),
        "strict_successor_replays": replay_rows,
        "workload_screen": {
            "version": WORKLOAD_SCREEN_VERSION,
            "contract_sha256": WORKLOAD_CONTRACT_SHA256,
            "source_sha256": workload_source_sha256(),
            "frozen_actor_sha256": FROZEN_ACTOR_SHA256,
            "replayed": bool(replay_workloads),
            "split_protocols": deepcopy(workload_contract()["split_workloads"]),
            "generation": deepcopy(generation_reports),
            "accepted_scene_count": sum(EXPECTED_BASE_COUNTS.values()),
            "accepted_receipts_sha256": digest(
                [
                    scene["workload_screen"]["receipt_sha256"]
                    for split in BASE_SPLITS
                    for scene in splits[split]
                ]
            ),
            "question_next_actions": sorted(next_actions, key=ACTIONS.index),
            "question_wait_displacements": [
                list(value) for value in sorted(wait_displacements)
            ],
        },
        "new_pickup_on_agent": sum(
            row.get("new_pickup_on_agent", 0) for row in replay_rows
        ),
        "new_delivery_on_agent": sum(
            row.get("new_delivery_on_agent", 0) for row in replay_rows
        ),
        "new_endpoint_on_agent": sum(
            row.get("new_endpoint_on_agent", 0) for row in replay_rows
        ),
        "immediate_task_recreation": sum(
            row.get("immediate_task_recreation", 0) for row in replay_rows
        ),
    }


def build_to_directory(
    source_manifest_path: str | Path,
    output: str | Path,
    *,
    actor_path: str | Path,
    batch_count: int = 1,
    per_family: int = PER_FAMILY_PER_BATCH,
) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(
        source_manifest_path,
        actor_path=actor_path,
        batch_count=batch_count,
        per_family=per_family,
    )
    validation = validate_diagnostic_manifest(
        manifest,
        replay=True,
        workload_actor_path=actor_path,
        replay_workloads=True,
    )
    _write_json(output / "diagnostic_contract.json", diagnostic_contract_receipt())
    _write_json(output / "manifest.json", manifest)
    validation["artifacts"] = {
        "diagnostic_contract.json": file_hash(output / "diagnostic_contract.json"),
        "manifest.json": file_hash(output / "manifest.json"),
    }
    _write_json(output / "validation.json", validation)
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--per-family", type=int, default=PER_FAMILY_PER_BATCH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_to_directory(
        args.source_manifest,
        args.output,
        actor_path=args.actor,
        batch_count=args.batch_count,
        per_family=args.per_family,
    )
    print(canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
