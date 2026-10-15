"""Actor-conditioned six-family scene selection for r4.1 diagnostic.

The immutable final two-million-step Actor is evaluated without action masks,
re-selection, or post-policy overrides.  Every generated batch is cheaply
screened first; only the best states in each conflict family receive the full
five-baseline plus compatible-reference protocol.  Selection stops at the
first batch prefix that supplies one admitted scene from every family.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from statistics import mean
import tempfile
import time
from typing import Any, Mapping, Sequence

from backend.training.warehouse_r4_conflict_scene_selection import (
    BASELINES,
    CONTINUATION_SEEDS,
    DYNAMIC,
    PAIRING,
    _actor_action,
    _dangerous_player_actions,
    _legal_player_actions,
    compatible_reference_action,
    dynamic_metrics,
    simple_action,
    update_collision_recovery,
)
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS,
    MAXIMUM_BATCHES,
    PER_FAMILY_PER_BATCH,
    VERSION as MANIFEST_VERSION,
    file_hash,
    validate_diagnostic_manifest,
)
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    DEFAULT_REWARD_CONFIG,
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_conflict import R41ConflictSamplingError, canonical, digest
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
    diagnostic_scene_fingerprint,
    reset_diagnostic_scenario,
)


VERSION = "warehouse-r41-diagnostic-conflict-dynamic-selection.v3"
ROOT = Path(__file__).resolve().parents[2]
FROZEN_ACTOR_SHA256 = "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
PREFILTER_CONTINUATION_SEEDS = CONTINUATION_SEEDS[:2]
PREFILTER_PROFILES = (*BASELINES, "compatible_reference")
PREFILTER_HORIZON = 60
FULL_SHORTLIST_PER_FAMILY = 20
PROTOCOL = {
    "version": VERSION,
    "scope": "diagnostic play-scene selection only; no participant records and no Actor selection",
    "frozen_actor_sha256": FROZEN_ACTOR_SHA256,
    "manifest_version": MANIFEST_VERSION,
    "workload_screen_contract_sha256": WORKLOAD_CONTRACT_SHA256,
    "candidate_generation": {
        "per_family_per_batch": PER_FAMILY_PER_BATCH,
        "conflict_family_count": len(FAMILY_IDS),
        "maximum_batches": MAXIMUM_BATCHES,
        "batch_policy": "stop after first prefix admitting a balanced six-scene selection",
    },
    "cheap_prefilter": {
        "profiles": list(PREFILTER_PROFILES),
        "continuation_seeds": list(PREFILTER_CONTINUATION_SEEDS),
        "horizon": PREFILTER_HORIZON,
        "shortlist_per_family": FULL_SHORTLIST_PER_FAMILY,
        "purpose": "rank scenes for full evaluation; never changes a hard gate",
    },
    "full_dynamic": {
        "profiles": [*BASELINES, "compatible_reference"],
        "continuation_seeds": list(CONTINUATION_SEEDS),
        "runs_per_profile": len(CONTINUATION_SEEDS),
        "full_horizon": True,
        "gates": deepcopy(DYNAMIC),
    },
    "pairing_gates": {
        **deepcopy(PAIRING),
        "one_scene_per_each_of_six_conflict_families": True,
        "pair_reference_delivery_difference_max": 1.0,
        "pair_conflict_opportunity_difference_max": 0.03,
        "group_workload_relative_difference_max": 0.05,
        "group_conflict_relative_difference_max": 0.05,
    },
    "successor_rng": "sha256(scene successor_stream_seed, continuation seed)",
    "actor_role": "robot_2 deterministic raw argmax; zero action overrides",
    "strict_successors": "neither replacement endpoint on either robot; no immediate recreation; no ordinary fallback",
    "participant_data_read": False,
    "final_test_rollouts": 0,
}

_WORKER_ACTOR: NumPyNativeActor | None = None


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        ROOT / "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
        ROOT / "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
        ROOT / "backend/training/warehouse_r4_conflict_scene_selection.py",
    )


def _current_source_sha256() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_hash(path) for path in _source_paths()
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def actor_environment(
    actor: NumPyNativeActor, scene: Mapping[str, Any]
) -> R41DiagnosticConflictWarehouseEnv:
    if actor.obs_dim != 197:
        raise ValueError("Diagnostic play audit requires an observed197 Actor")
    env = R41DiagnosticConflictWarehouseEnv(
        reward_config=deepcopy(DEFAULT_REWARD_CONFIG)
    )
    reset_diagnostic_scenario(env, scene)
    if list(env.feature_names) != actor.metadata.get("feature_names"):
        raise ValueError("Actor feature contract differs from diagnostic environment")
    return env


def _derived_continuation_seed(
    scene: Mapping[str, Any], continuation_seed: int
) -> int:
    return int(
        digest(
            {
                "version": VERSION,
                "successor_stream_seed": int(scene["successor_stream_seed"]),
                "continuation_seed": int(continuation_seed),
            }
        )[:16],
        16,
    )


def run_actor_episode(
    scene: Mapping[str, Any],
    actor: NumPyNativeActor,
    profile: str,
    continuation_seed: int,
    *,
    maximum_steps: int | None = None,
) -> dict[str, Any]:
    if profile not in (*BASELINES, "compatible_reference"):
        raise ValueError("Unknown diagnostic dynamic-audit profile")
    if continuation_seed not in CONTINUATION_SEEDS:
        raise ValueError("Continuation seed was not pre-registered")
    env = actor_environment(actor, scene)
    derived_seed = _derived_continuation_seed(scene, continuation_seed)
    env.set_rng_state(random.Random(derived_seed).getstate())
    initial_fingerprint = diagnostic_scene_fingerprint(env)
    pending_collisions: list[dict[str, Any]] = []
    recovered = 0
    counts = Counter()
    risky_mass = 0.0
    no_progress_streak = longest_no_progress = 0
    consecutive_collisions = longest_collisions = 0
    successor_sampling_failure = None
    while not env.done and (
        maximum_steps is None or counts["steps"] < maximum_steps
    ):
        before_snapshot = env.snapshot()
        actor_action = _actor_action(actor, env)
        if env.snapshot() != before_snapshot:
            raise RuntimeError("Diagnostic Actor inference changed environment state")
        dangerous = _dangerous_player_actions(env, actor_action)
        legal = _legal_player_actions(env)
        player_action = (
            compatible_reference_action(env, actor_action)
            if profile == "compatible_reference"
            else simple_action(env, profile)
        )
        if env.snapshot() != before_snapshot:
            raise RuntimeError("Diagnostic partner decision changed environment state")
        before_potential = env.potential()
        before_deliveries = env.state.total_deliveries
        before_pickups = sum(task.status == "carried" for task in env.state.tasks)
        try:
            _, _, _, _, info = env.step(
                {"robot_1": player_action, "robot_2": actor_action},
                decision_metadata={
                    "policy_action": actor_action,
                    "submitted_action": actor_action,
                    "post_policy_overrides": 0,
                },
            )
        except R41ConflictSamplingError as error:
            successor_sampling_failure = {
                "frame": env.state.frame,
                "error": str(error),
            }
            break
        submitted = info["requested_actions"]["robot_2"]
        if submitted != actor_action:
            raise AssertionError("Diagnostic Actor action was not submitted unchanged")
        creation = info["r41_diagnostic_conflict"]["created"]
        counts["steps"] += 1
        counts["actor_submission_frames"] += 1
        counts["actor_action_override_frames"] += int(submitted != actor_action)
        counts["conflict_opportunity_frames"] += int(bool(dangerous))
        risky_mass += len(dangerous) / len(legal)
        counts["replacement_tasks"] += len(creation)
        for name in (
            "spawned_on_agent_endpoint",
            "immediate_task_recreation",
            "new_endpoint_on_agent",
            "new_pickup_on_agent",
            "new_delivery_on_agent",
        ):
            counts[name] += sum(bool(row[name]) for row in creation)
        counts["active_conflict_pair_checks"] += 1
        progress = bool(
            env.state.total_deliveries > before_deliveries
            or sum(task.status == "carried" for task in env.state.tasks)
            > before_pickups
            or env.potential() > before_potential + 1e-9
        )
        no_progress_streak = 0 if progress else no_progress_streak + 1
        longest_no_progress = max(longest_no_progress, no_progress_streak)
        collided = bool(info["robot_collision"])
        counts["collisions"] += int(collided)
        consecutive_collisions = consecutive_collisions + 1 if collided else 0
        longest_collisions = max(longest_collisions, consecutive_collisions)
        if collided:
            pending_collisions.append({"frame": env.state.frame, "resolved": False})
        recovered += update_collision_recovery(
            pending_collisions, env.state.frame, progress
        )
    return {
        "scene_id": scene["id"],
        "seed": int(scene["seed"]),
        "batch_index": int(scene["batch_index"]),
        "family_id": scene["family_id"],
        "initial_edge_id": scene["initial_edge_id"],
        "profile": profile,
        "continuation_seed": int(continuation_seed),
        "derived_continuation_seed": derived_seed,
        "initial_continuation_fingerprint": initial_fingerprint,
        "steps": counts["steps"],
        "deliveries": env.state.total_deliveries,
        "deliveries_by_robot": {
            agent.agent_id: agent.deliveries_completed for agent in env.state.agents
        },
        "collisions": counts["collisions"],
        "shutdowns": env.state.shutdown_count,
        "conflict_opportunity_frames": counts["conflict_opportunity_frames"],
        "player_risky_action_mass_sum": risky_mass,
        "recovered_collisions_within_10": recovered,
        "longest_consecutive_collisions": longest_collisions,
        "longest_no_progress_streak": longest_no_progress,
        "terminal_consecutive_collisions": consecutive_collisions,
        "terminal_no_progress_streak": no_progress_streak,
        "actor_submission_frames": counts["actor_submission_frames"],
        "actor_action_override_frames": counts["actor_action_override_frames"],
        "replacement_tasks": counts["replacement_tasks"],
        "spawned_on_agent_endpoint": counts["spawned_on_agent_endpoint"],
        "immediate_task_recreation": counts["immediate_task_recreation"],
        "new_endpoint_on_agent": counts["new_endpoint_on_agent"],
        "new_pickup_on_agent": counts["new_pickup_on_agent"],
        "new_delivery_on_agent": counts["new_delivery_on_agent"],
        "active_conflict_pair_checks": counts["active_conflict_pair_checks"],
        "successor_sampling_failure": successor_sampling_failure,
        "terminal_reason": (
            "conflict_successor_unavailable"
            if successor_sampling_failure is not None
            else ("prefilter_horizon" if maximum_steps and not env.done else env.state.terminal_reason)
        ),
    }


def _strict_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = dynamic_metrics(episodes)
    for name in (
        "replacement_tasks",
        "spawned_on_agent_endpoint",
        "immediate_task_recreation",
        "new_endpoint_on_agent",
        "new_pickup_on_agent",
        "new_delivery_on_agent",
        "active_conflict_pair_checks",
    ):
        metrics[name] = sum(int(row[name]) for row in episodes)
    metrics["successor_sampling_failures"] = sum(
        row["successor_sampling_failure"] is not None for row in episodes
    )
    metrics["checks"].update(
        no_immediate_task_recreation=metrics["immediate_task_recreation"] == 0,
        no_new_pickup_on_agent=metrics["new_pickup_on_agent"] == 0,
        no_new_delivery_on_agent=metrics["new_delivery_on_agent"] == 0,
        no_new_endpoint_on_agent=metrics["new_endpoint_on_agent"] == 0,
        no_successor_sampling_failure=metrics["successor_sampling_failures"] == 0,
    )
    metrics["passed"] = all(metrics["checks"].values())
    return metrics


def _partial_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    simple = [row for row in episodes if row["profile"] in BASELINES]
    reference = [row for row in episodes if row["profile"] == "compatible_reference"]
    simple_steps = sum(row["steps"] for row in simple)
    if not simple_steps:
        raise ValueError("Diagnostic prefilter produced no simple-baseline steps")
    opportunity = sum(row["conflict_opportunity_frames"] for row in simple) / simple_steps
    risky = sum(row["player_risky_action_mass_sum"] for row in simple) / simple_steps
    cancellation = sum(row["collisions"] for row in simple) / simple_steps
    profile_deliveries = {
        profile: mean(row["deliveries"] for row in simple if row["profile"] == profile)
        for profile in BASELINES
    }
    reference_deliveries = mean(row["deliveries"] for row in reference)
    gap = reference_deliveries - max(profile_deliveries.values())
    invalid = bool(
        sum(row["actor_action_override_frames"] for row in episodes)
        or sum(row["new_endpoint_on_agent"] for row in episodes)
        or sum(row["immediate_task_recreation"] for row in episodes)
        or any(row["successor_sampling_failure"] is not None for row in episodes)
    )
    score = (
        abs(opportunity - 0.325) / 0.075
        + abs(risky - 0.15) / 0.05
        + abs(cancellation - 0.13) / 0.05
        - min(gap, 4.0) / 4.0
        + (1_000.0 if invalid else 0.0)
    )
    return {
        "score": score,
        "conflict_opportunity_fraction": opportunity,
        "player_risky_action_mass": risky,
        "collision_cancellation_fraction": cancellation,
        "compatible_reference_mean_deliveries": reference_deliveries,
        "best_simple_mean_deliveries": max(profile_deliveries.values()),
        "reference_absolute_delivery_gap": gap,
        "actor_action_override_frames": sum(
            row["actor_action_override_frames"] for row in episodes
        ),
        "new_pickup_on_agent": sum(row["new_pickup_on_agent"] for row in episodes),
        "new_delivery_on_agent": sum(row["new_delivery_on_agent"] for row in episodes),
        "new_endpoint_on_agent": sum(row["new_endpoint_on_agent"] for row in episodes),
        "immediate_task_recreation": sum(row["immediate_task_recreation"] for row in episodes),
        "successor_sampling_failures": sum(
            row["successor_sampling_failure"] is not None for row in episodes
        ),
    }


def _worker_initialize(actor_path: str) -> None:
    global _WORKER_ACTOR
    _WORKER_ACTOR = NumPyNativeActor(actor_path)


def _prefilter_worker(scene: Mapping[str, Any]):
    if _WORKER_ACTOR is None:
        raise RuntimeError("Diagnostic prefilter worker has no Actor")
    episodes = [
        run_actor_episode(
            scene,
            _WORKER_ACTOR,
            profile,
            continuation_seed,
            maximum_steps=PREFILTER_HORIZON,
        )
        for profile in PREFILTER_PROFILES
        for continuation_seed in PREFILTER_CONTINUATION_SEEDS
    ]
    return _partial_metrics(episodes)


def _full_worker(scene: Mapping[str, Any]):
    if _WORKER_ACTOR is None:
        raise RuntimeError("Diagnostic full-audit worker has no Actor")
    episodes = [
        run_actor_episode(scene, _WORKER_ACTOR, profile, continuation_seed)
        for profile in (*BASELINES, "compatible_reference")
        for continuation_seed in CONTINUATION_SEEDS
    ]
    return episodes, _strict_metrics(episodes)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(1e-12, (abs(left) + abs(right)) / 2.0)


def _quality(row: Mapping[str, Any]) -> tuple[float, int, str]:
    dynamic = row["dynamic"]
    return (
        abs(dynamic["conflict_opportunity_fraction"] - 0.325) / 0.075
        + abs(dynamic["player_risky_action_mass"] - 0.15) / 0.05
        + abs(dynamic["collision_cancellation_fraction"] - 0.13) / 0.05
        - min(dynamic["reference_absolute_delivery_gap"], 4.0) / 4.0,
        int(row["seed"]),
        str(row["id"]),
    )


def select_balanced_six(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    passed = sorted(
        (row for row in candidates if row["dynamic"]["passed"]), key=_quality
    )
    if set(row["family_id"] for row in passed) != set(FAMILY_IDS):
        return None

    # Limiting each family to its twenty best admitted candidates preserves all
    # pre-registered gates while keeping exact pairing search tractable.
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in passed:
        if len(by_family[row["family_id"]]) < 20:
            by_family[row["family_id"]].append(row)
    family_matchings = []

    def match(remaining: tuple[str, ...], pairs: tuple[tuple[str, str], ...]):
        if not remaining:
            family_matchings.append(pairs)
            return
        first = remaining[0]
        for other in remaining[1:]:
            match(
                tuple(value for value in remaining if value not in (first, other)),
                pairs + ((first, other),),
            )

    match(tuple(FAMILY_IDS), ())
    best = None
    pairing_attempts = 0
    for family_pairs in family_matchings:
        pair_options = []
        viable = True
        for left_family, right_family in family_pairs:
            options = []
            for left in by_family[left_family]:
                for right in by_family[right_family]:
                    if (
                        abs(
                            left["dynamic"]["compatible_reference_mean_deliveries"]
                            - right["dynamic"]["compatible_reference_mean_deliveries"]
                        )
                        <= 1.0
                        and abs(
                            left["dynamic"]["conflict_opportunity_fraction"]
                            - right["dynamic"]["conflict_opportunity_fraction"]
                        )
                        <= 0.03
                    ):
                        options.append((left, right))
            options.sort(
                key=lambda pair: (
                    abs(
                        pair[0]["dynamic"]["conflict_opportunity_fraction"]
                        - pair[1]["dynamic"]["conflict_opportunity_fraction"]
                    ),
                    _quality(pair[0]),
                    _quality(pair[1]),
                )
            )
            # Exact gates are unchanged; this deterministic beam only removes
            # lower-ranked alternatives with the same family pairing.
            options = options[:40]
            if not options:
                viable = False
                break
            pair_options.append(options)
        if not viable:
            continue
        for first in pair_options[0]:
            for second in pair_options[1]:
                for third in pair_options[2]:
                    pairing_attempts += 1
                    pairs = (first, second, third)
                    for orientation in range(8):
                        x_rows, y_rows = [], []
                        for index, pair in enumerate(pairs):
                            left, right = pair
                            if orientation & (1 << index):
                                left, right = right, left
                            x_rows.append(left)
                            y_rows.append(right)
                        workload_x = mean(
                            row["initial_public_joint_work_steps"]
                            for row in x_rows
                        )
                        workload_y = mean(
                            row["initial_public_joint_work_steps"]
                            for row in y_rows
                        )
                        conflict_x = mean(
                            row["dynamic"]["conflict_opportunity_fraction"]
                            for row in x_rows
                        )
                        conflict_y = mean(
                            row["dynamic"]["conflict_opportunity_fraction"]
                            for row in y_rows
                        )
                        workload_difference = _relative_difference(
                            workload_x, workload_y
                        )
                        conflict_difference = _relative_difference(
                            conflict_x, conflict_y
                        )
                        if workload_difference > 0.05 or conflict_difference > 0.05:
                            continue
                        score = (
                            workload_difference
                            + conflict_difference
                            + sum(
                                abs(
                                    pair[0]["dynamic"]["conflict_opportunity_fraction"]
                                    - pair[1]["dynamic"]["conflict_opportunity_fraction"]
                                )
                                for pair in pairs
                            )
                            + 0.001 * sum(_quality(row)[0] for row in (*x_rows, *y_rows))
                        )
                        tie = tuple(row["seed"] for row in (*x_rows, *y_rows))
                        value = (
                            score,
                            tie,
                            {
                                "score": score,
                                "X": [deepcopy(row) for row in x_rows],
                                "Y": [deepcopy(row) for row in y_rows],
                                "pairs": [
                                    [x_rows[index]["id"], y_rows[index]["id"]]
                                    for index in range(3)
                                ],
                                "balance": {
                                    "X_mean_initial_workload": workload_x,
                                    "Y_mean_initial_workload": workload_y,
                                    "workload_relative_difference": workload_difference,
                                    "X_mean_conflict_opportunity": conflict_x,
                                    "Y_mean_conflict_opportunity": conflict_y,
                                    "conflict_relative_difference": conflict_difference,
                                },
                            },
                        )
                        if best is None or value[:2] < best[:2]:
                            best = value
    if best is None:
        return None
    result = best[2]
    result["pairing_attempts"] = pairing_attempts
    result["dynamic_pass_count"] = len(passed)
    return result


def _public_scene(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "split",
        "seed",
        "batch_index",
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
    )
    return {key: deepcopy(row[key]) for key in keys}


def audit(
    manifest_path: str | Path,
    actor_path: str | Path,
    output: str | Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Diagnostic workers must be an integer from one to eight")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(manifest_path).expanduser().resolve()
    actor_path = Path(actor_path).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    validation = validate_diagnostic_manifest(manifest, replay=True)
    if not validation["passed"] or manifest["version"] != MANIFEST_VERSION:
        raise ValueError("Diagnostic source manifest did not pass validation")
    actor = NumPyNativeActor(actor_path)
    if actor.sha256 != FROZEN_ACTOR_SHA256:
        raise ValueError("Diagnostic selection requires the exact final 2M Actor")
    source_paths = _source_paths()
    provenance = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": deepcopy(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "source_manifest": str(manifest_path),
        "source_manifest_file_sha256": sha256(manifest_bytes).hexdigest(),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "actor": {
            "path": str(actor_path),
            "sha256": actor.sha256,
            "deterministic": True,
            "post_policy_overrides": 0,
        },
        "source_sha256": _current_source_sha256(),
        "participant_data_read": False,
        "final_test_rollouts": 0,
    }
    _write_json(output / "protocol.json", provenance)
    started = time.perf_counter()
    evaluated: list[dict[str, Any]] = []
    prefilter_records = []
    selection = None
    batches_considered = 0
    with (output / "prefilter.jsonl").open("x", encoding="utf-8") as prefilter_stream, (
        output / "episodes.jsonl"
    ).open("x", encoding="utf-8") as episode_stream:
        for batch_index, candidates in enumerate(manifest["candidate_batches"]):
            batches_considered += 1
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_initialize,
                initargs=(str(actor_path),),
            ) as pool:
                metrics_rows = list(pool.map(_prefilter_worker, candidates))
            ranked_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for scene, metrics in zip(candidates, metrics_rows):
                record = {
                    "scene_id": scene["id"],
                    "batch_index": batch_index,
                    "family_id": scene["family_id"],
                    "seed": scene["seed"],
                    **metrics,
                }
                prefilter_stream.write(canonical(record) + "\n")
                prefilter_records.append(record)
                ranked_by_family[scene["family_id"]].append(
                    {**deepcopy(scene), "prefilter": metrics}
                )
            prefilter_stream.flush()
            shortlist = []
            for family in FAMILY_IDS:
                ranked = sorted(
                    ranked_by_family[family],
                    key=lambda row: (
                        row["prefilter"]["score"], row["seed"], row["id"]
                    ),
                )
                shortlist.extend(ranked[:FULL_SHORTLIST_PER_FAMILY])
            shortlist.sort(key=lambda row: (row["family_id"], row["seed"]))
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_initialize,
                initargs=(str(actor_path),),
            ) as pool:
                outcomes = pool.map(_full_worker, shortlist)
                for index, (scene, outcome) in enumerate(zip(shortlist, outcomes), 1):
                    episodes, metrics = outcome
                    for episode in episodes:
                        episode_stream.write(canonical(episode) + "\n")
                    episode_stream.flush()
                    evaluated.append({**deepcopy(scene), "dynamic": metrics})
                    print(
                        canonical(
                            {
                                "batch": batch_index,
                                "completed": index,
                                "total": len(shortlist),
                                "scene_id": scene["id"],
                                "family_id": scene["family_id"],
                                "passed": metrics["passed"],
                            }
                        ),
                        flush=True,
                    )
            selection = select_balanced_six(evaluated)
            if selection is not None:
                break
    selected = None
    selected_physical_replay_episodes = 0
    if selection is not None:
        selected = {
            "version": VERSION,
            "status": "accepted_diagnostic_dynamic_selection",
            "release_eligible": True,
            "actor_sha256": actor.sha256,
            "source_manifest_file_sha256": sha256(manifest_bytes).hexdigest(),
            "source_manifest_content_sha256": manifest["content_sha256"],
            "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
            "protocol_sha256": digest(PROTOCOL),
            "tutorial": deepcopy(manifest["splits"]["tutorial"][0]),
            "X": [_public_scene(row) for row in selection["X"]],
            "Y": [_public_scene(row) for row in selection["Y"]],
            "pairs": deepcopy(selection["pairs"]),
            "balance": deepcopy(selection["balance"]),
            "selection_score": selection["score"],
            "six_distinct_conflict_families": True,
            "zero_action_overrides": True,
            "no_replacement_pickup_on_agent": True,
            "no_replacement_delivery_on_agent": True,
            "no_replacement_endpoint_on_agent": True,
            "ordinary_sampler_fallback": False,
        }
        _write_json(output / "selected_scenes.json", selected)
    report = {
        **provenance,
        "status": (
            "accepted_diagnostic_dynamic_selection"
            if selected is not None
            else "blocked_no_six_family_dynamic_selection"
        ),
        "release_eligible": selected is not None,
        "batches_available": len(manifest["candidate_batches"]),
        "batches_considered": batches_considered,
        "prefilter_evaluated": len(prefilter_records),
        "full_dynamic_evaluated": len(evaluated),
        "dynamic_passed": sum(row["dynamic"]["passed"] for row in evaluated),
        "dynamic_passed_by_family": dict(
            Counter(
                row["family_id"] for row in evaluated if row["dynamic"]["passed"]
            )
        ),
        "evaluated_scene_ids": [row["id"] for row in evaluated],
        "actor_submission_frames": sum(
            row["dynamic"]["actor_submission_frames"] for row in evaluated
        ),
        "actor_action_override_frames": sum(
            row["dynamic"]["actor_action_override_frames"] for row in evaluated
        ),
        "replacement_tasks": sum(
            row["dynamic"]["replacement_tasks"] for row in evaluated
        ),
        "new_pickup_on_agent": sum(
            row["dynamic"]["new_pickup_on_agent"] for row in evaluated
        ),
        "new_delivery_on_agent": sum(
            row["dynamic"]["new_delivery_on_agent"] for row in evaluated
        ),
        "new_endpoint_on_agent": sum(
            row["dynamic"]["new_endpoint_on_agent"] for row in evaluated
        ),
        "immediate_task_recreation": sum(
            row["dynamic"]["immediate_task_recreation"] for row in evaluated
        ),
        "successor_sampling_failures": sum(
            row["dynamic"]["successor_sampling_failures"] for row in evaluated
        ),
        "selection": (
            None
            if selection is None
            else {
                key: deepcopy(selection[key])
                for key in (
                    "pairs",
                    "balance",
                    "score",
                    "pairing_attempts",
                    "dynamic_pass_count",
                )
            }
        ),
        "scene_metrics": [
            {
                "id": row["id"],
                "seed": row["seed"],
                "batch_index": row["batch_index"],
                "family_id": row["family_id"],
                "initial_edge_id": row["initial_edge_id"],
                "prefilter": deepcopy(row["prefilter"]),
                "dynamic": deepcopy(row["dynamic"]),
            }
            for row in evaluated
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if report["actor_action_override_frames"] != 0:
        raise ValueError("Diagnostic dynamic audit observed an action override")
    if (
        report["new_pickup_on_agent"] != 0
        or report["new_delivery_on_agent"] != 0
        or report["new_endpoint_on_agent"] != 0
        or report["immediate_task_recreation"] != 0
    ):
        raise ValueError("Diagnostic audit observed an unsafe replacement task")
    if any(
        file_hash(path) != provenance["source_sha256"][str(path.relative_to(ROOT))]
        for path in source_paths
    ):
        raise RuntimeError("Diagnostic dynamic-audit source changed during evaluation")
    if manifest_path.read_bytes() != manifest_bytes or file_hash(actor_path) != actor.sha256:
        raise RuntimeError("Diagnostic frozen audit input changed during evaluation")
    report["evidence_artifacts"] = {
        "protocol.json": file_hash(output / "protocol.json"),
        "prefilter.jsonl": file_hash(output / "prefilter.jsonl"),
        "episodes.jsonl": file_hash(output / "episodes.jsonl"),
        **(
            {"selected_scenes.json": file_hash(output / "selected_scenes.json")}
            if selected is not None
            else {}
        ),
    }
    _write_json(output / "report.json", report)
    return report


def _strict_json(path: Path, *, maximum_bytes: int = 256 * 1024 * 1024) -> Any:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve() != path
        or path.stat().st_size > maximum_bytes
    ):
        raise ValueError("Diagnostic evidence file is missing or unsafe")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate field in diagnostic evidence")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite diagnostic evidence: " + token)
        ),
    )


def _strict_jsonl(
    path: Path, *, maximum_bytes: int = 512 * 1024 * 1024
) -> list[dict[str, Any]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve() != path
        or path.stat().st_size > maximum_bytes
    ):
        raise ValueError("Diagnostic evidence journal is missing or unsafe")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("Blank diagnostic evidence row")
            value = json.loads(
                line,
                object_pairs_hook=lambda items: _unique_pairs(items, line_number),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError("Non-finite diagnostic journal value: " + token)
                ),
            )
            if not isinstance(value, dict):
                raise ValueError("Diagnostic journal row must be an object")
            rows.append(value)
    return rows


def _unique_pairs(items, line_number: int) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(
                f"Duplicate field in diagnostic journal row {line_number}"
            )
        result[key] = value
    return result


_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREFILTER_FIELDS = frozenset((
    "scene_id", "batch_index", "family_id", "seed", "score",
    "conflict_opportunity_fraction", "player_risky_action_mass",
    "collision_cancellation_fraction", "compatible_reference_mean_deliveries",
    "best_simple_mean_deliveries", "reference_absolute_delivery_gap",
    "actor_action_override_frames", "new_pickup_on_agent",
    "new_delivery_on_agent", "new_endpoint_on_agent",
    "immediate_task_recreation", "successor_sampling_failures",
))
_EPISODE_FIELDS = frozenset((
    "scene_id", "seed", "batch_index", "family_id", "initial_edge_id",
    "profile", "continuation_seed", "derived_continuation_seed",
    "initial_continuation_fingerprint", "steps", "deliveries",
    "deliveries_by_robot", "collisions", "shutdowns",
    "conflict_opportunity_frames", "player_risky_action_mass_sum",
    "recovered_collisions_within_10", "longest_consecutive_collisions",
    "longest_no_progress_streak", "terminal_consecutive_collisions",
    "terminal_no_progress_streak", "actor_submission_frames",
    "actor_action_override_frames", "replacement_tasks",
    "spawned_on_agent_endpoint", "immediate_task_recreation",
    "new_endpoint_on_agent", "new_pickup_on_agent", "new_delivery_on_agent",
    "active_conflict_pair_checks", "successor_sampling_failure",
    "terminal_reason",
))
_REISSUE_VERSION = "warehouse-r41-diagnostic-selection-journal-reissue.v1"
_REISSUE_FIELDS = frozenset((
    "version", "source_output", "source_report_sha256",
    "source_evidence_artifacts", "source_producer_sha256",
    "source_status_not_trusted", "journals_copied_byte_exact",
    "canonical_journal_rows_validated", "complete_batch_prefix_recomputed",
    "all_scene_metrics_recomputed", "balanced_selection_recomputed",
    "selected_physical_replay_episodes",
))


def _exact_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _inside_repo_directory(value: str | Path, label: str) -> tuple[Path, str]:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    return path, relative


def _canonical_journal(path: Path, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    rows = _strict_jsonl(path)
    raw = path.read_bytes()
    expected = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
    if raw != expected:
        raise ValueError(label + " is not a canonical JSONL journal")
    return raw, rows


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _validate_prefilter_journal(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _PREFILTER_FIELDS:
            raise ValueError("Diagnostic prefilter journal schema differs")
        for name in (
            "batch_index", "seed", "actor_action_override_frames",
            "new_pickup_on_agent", "new_delivery_on_agent",
            "new_endpoint_on_agent", "immediate_task_recreation",
            "successor_sampling_failures",
        ):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError("Diagnostic prefilter integer metric differs")
        for name in (
            "conflict_opportunity_fraction", "player_risky_action_mass",
            "collision_cancellation_fraction",
        ):
            value = _finite_number(row[name], "diagnostic prefilter " + name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("Diagnostic prefilter fraction is outside [0,1]")
        reference = _finite_number(
            row["compatible_reference_mean_deliveries"],
            "diagnostic prefilter reference deliveries",
        )
        best_simple = _finite_number(
            row["best_simple_mean_deliveries"],
            "diagnostic prefilter simple deliveries",
        )
        gap = _finite_number(
            row["reference_absolute_delivery_gap"],
            "diagnostic prefilter delivery gap",
        )
        if min(reference, best_simple) < 0 or not math.isclose(
            gap, reference - best_simple, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Diagnostic prefilter delivery arithmetic differs")
        invalid = bool(
            row["actor_action_override_frames"]
            or row["new_endpoint_on_agent"]
            or row["immediate_task_recreation"]
            or row["successor_sampling_failures"]
        )
        expected_score = (
            abs(float(row["conflict_opportunity_fraction"]) - 0.325) / 0.075
            + abs(float(row["player_risky_action_mass"]) - 0.15) / 0.05
            + abs(float(row["collision_cancellation_fraction"]) - 0.13) / 0.05
            - min(gap, 4.0) / 4.0
            + (1_000.0 if invalid else 0.0)
        )
        score = _finite_number(row["score"], "diagnostic prefilter score")
        if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Diagnostic prefilter score was not recomputed")


def _validate_episode_journal(rows: Sequence[Mapping[str, Any]]) -> None:
    profiles = (*BASELINES, "compatible_reference")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _EPISODE_FIELDS:
            raise ValueError("Diagnostic full-episode journal schema differs")
        if (
            row["profile"] not in profiles
            or row["continuation_seed"] not in CONTINUATION_SEEDS
            or type(row["steps"]) is not int
            or row["steps"] <= 0
            or row["actor_submission_frames"] != row["steps"]
            or row["active_conflict_pair_checks"] != row["steps"]
            or row["actor_action_override_frames"] != 0
            or row["new_pickup_on_agent"] != 0
            or row["new_delivery_on_agent"] != 0
            or row["new_endpoint_on_agent"] != 0
            or row["immediate_task_recreation"] != 0
            or row["successor_sampling_failure"] is not None
            or not isinstance(row["terminal_reason"], str)
            or _HEX_SHA256.fullmatch(
                str(row["initial_continuation_fingerprint"])
            ) is None
        ):
            raise ValueError("Diagnostic full-episode protocol differs")
        deliveries = row["deliveries_by_robot"]
        if (
            not isinstance(deliveries, Mapping)
            or set(deliveries) != {"robot_1", "robot_2"}
            or any(type(value) is not int or value < 0 for value in deliveries.values())
            or sum(deliveries.values()) != row["deliveries"]
        ):
            raise ValueError("Diagnostic full-episode delivery accounting differs")
        for name in (
            "player_risky_action_mass_sum", "deliveries", "collisions",
            "shutdowns", "conflict_opportunity_frames",
            "recovered_collisions_within_10", "longest_consecutive_collisions",
            "longest_no_progress_streak", "terminal_consecutive_collisions",
            "terminal_no_progress_streak", "replacement_tasks",
            "spawned_on_agent_endpoint", "derived_continuation_seed",
        ):
            value = _finite_number(row[name], "diagnostic episode " + name)
            if value < 0:
                raise ValueError("Diagnostic full-episode metric is negative")


def _reissue_source_evidence(
    source_output: str | Path,
    *,
    expected_report_sha256: str,
    actor_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    source, source_relative = _inside_repo_directory(
        source_output, "diagnostic reissue source"
    )
    report_path = source / "report.json"
    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or report_path.resolve() != report_path
        or file_hash(report_path) != _exact_sha256(
            expected_report_sha256, "source diagnostic report"
        )
    ):
        raise ValueError("Source diagnostic report bytes differ")
    report = _strict_json(report_path)
    if not isinstance(report, Mapping):
        raise ValueError("Source diagnostic report must be an object")
    protocol_path = source / "protocol.json"
    selected_path = source / "selected_scenes.json"
    artifacts = report.get("evidence_artifacts")
    expected_names = {
        "protocol.json", "prefilter.jsonl", "episodes.jsonl",
        "selected_scenes.json",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        raise ValueError("Source diagnostic artifact registry differs")
    artifact_paths = {name: source / name for name in expected_names}
    for name, path in artifact_paths.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or file_hash(path) != _exact_sha256(
                artifacts[name], "source diagnostic " + name
            )
        ):
            raise ValueError("Source diagnostic artifact bytes differ: " + name)
    protocol = _strict_json(protocol_path)
    selected = _strict_json(selected_path)
    if not isinstance(protocol, Mapping) or not isinstance(selected, Mapping):
        raise ValueError("Source diagnostic protocol and selection must be objects")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _strict_json(manifest_path)
    validation = validate_diagnostic_manifest(manifest, replay=True)
    actor_path = Path(actor_path).expanduser().resolve()
    actor = NumPyNativeActor(actor_path)
    current_sources = _current_source_sha256()
    source_sources = report.get("source_sha256")
    selector_name = str(Path(__file__).relative_to(ROOT))
    if (
        not isinstance(report, Mapping)
        or report.get("version") != VERSION
        or report.get("protocol") != PROTOCOL
        or report.get("protocol_sha256") != digest(PROTOCOL)
        or report.get("source_manifest_file_sha256") != file_hash(manifest_path)
        or report.get("source_manifest_content_sha256")
        != manifest.get("content_sha256")
        or report.get("diagnostic_contract_version")
        != DIAGNOSTIC_CONTRACT_VERSION
        or report.get("diagnostic_contract_sha256")
        != DIAGNOSTIC_CONTRACT_SHA256
        or report.get("diagnostic_conflict_graph_sha256")
        != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
        or report.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256
        or report.get("actor", {}).get("sha256") != actor.sha256
        or actor.sha256 != FROZEN_ACTOR_SHA256
        or not validation.get("passed")
        or not isinstance(source_sources, Mapping)
        or set(source_sources) != set(current_sources)
        or _HEX_SHA256.fullmatch(str(source_sources.get(selector_name))) is None
        or source_sources.get(selector_name) == current_sources[selector_name]
        or any(
            source_sources.get(name) != value
            for name, value in current_sources.items()
            if name != selector_name
        )
        or protocol.get("version") != VERSION
        or protocol.get("protocol") != PROTOCOL
        or protocol.get("protocol_sha256") != digest(PROTOCOL)
        or protocol.get("source_sha256") != source_sources
        or protocol.get("actor", {}).get("sha256") != actor.sha256
        or protocol.get("source_manifest_file_sha256") != file_hash(manifest_path)
        or protocol.get("source_manifest_content_sha256")
        != manifest.get("content_sha256")
    ):
        raise ValueError("Source diagnostic evidence binding differs")
    prefilter_raw, prefilter_rows = _canonical_journal(
        artifact_paths["prefilter.jsonl"], "diagnostic prefilter"
    )
    episodes_raw, episode_rows = _canonical_journal(
        artifact_paths["episodes.jsonl"], "diagnostic full episodes"
    )
    _validate_prefilter_journal(prefilter_rows)
    _validate_episode_journal(episode_rows)
    return {
        "source": source,
        "source_relative": source_relative,
        "report": report,
        "protocol": protocol,
        "selected": selected,
        "manifest": manifest,
        "actor": actor,
        "artifact_sha256": dict(artifacts),
        "prefilter_raw": prefilter_raw,
        "episodes_raw": episodes_raw,
        "source_producer_sha256": dict(source_sources),
    }


def _validate_reissue_provenance(
    report: Mapping[str, Any], protocol: Mapping[str, Any], output: Path
) -> None:
    record = report.get("evidence_reissue")
    if record is None:
        return
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(record, Mapping)
        or set(record) != _REISSUE_FIELDS
        or protocol.get("evidence_reissue") != record
        or record.get("version") != _REISSUE_VERSION
        or record.get("source_status_not_trusted") is not True
        or record.get("journals_copied_byte_exact") is not True
        or record.get("canonical_journal_rows_validated") is not True
        or record.get("complete_batch_prefix_recomputed") is not True
        or record.get("all_scene_metrics_recomputed") is not True
        or record.get("balanced_selection_recomputed") is not True
        or record.get("selected_physical_replay_episodes") != 720
        or not isinstance(record.get("source_evidence_artifacts"), Mapping)
        or not isinstance(record.get("source_producer_sha256"), Mapping)
    ):
        raise ValueError("Diagnostic journal-reissue provenance differs")
    source = ROOT / str(record["source_output"])
    source, relative = _inside_repo_directory(source, "diagnostic reissue source")
    if relative != record["source_output"]:
        raise ValueError("Diagnostic journal-reissue source path differs")
    source_report = source / "report.json"
    if (
        source_report.is_symlink()
        or not source_report.is_file()
        or source_report.resolve() != source_report
        or file_hash(source_report) != _exact_sha256(
            record["source_report_sha256"], "reissue source report"
        )
    ):
        raise ValueError("Diagnostic journal-reissue source report changed")
    source_report_value = _strict_json(source_report)
    source_artifacts = record["source_evidence_artifacts"]
    if not isinstance(source_report_value, Mapping):
        raise ValueError("Diagnostic journal-reissue source report must be an object")
    if set(source_artifacts) != {
        "protocol.json", "prefilter.jsonl", "episodes.jsonl",
        "selected_scenes.json",
    } or source_report_value.get("evidence_artifacts") != source_artifacts:
        raise ValueError("Diagnostic journal-reissue artifact registry differs")
    if source_report_value.get("source_sha256") != record["source_producer_sha256"]:
        raise ValueError("Diagnostic journal-reissue producer binding differs")
    for name, expected in source_artifacts.items():
        path = source / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or file_hash(path) != _exact_sha256(
                expected, "reissue source " + name
            )
        ):
            raise ValueError("Diagnostic journal-reissue source artifact changed")
    evidence = report.get("evidence_artifacts", {})
    for name in ("prefilter.jsonl", "episodes.jsonl"):
        path = output / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or evidence.get(name) != source_artifacts.get(name)
            or file_hash(path) != source_artifacts.get(name)
        ):
            raise ValueError("Diagnostic reissued journal bytes differ")


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def read_saved_diagnostic_selection(
    output: str | Path,
    *,
    expected_report_sha256: str,
    actor_path: str | Path,
    manifest_path: str | Path,
    selected_scenes_path: str | Path | None,
    require_selected: bool = True,
) -> dict[str, Any]:
    selected_physical_replay_episodes = 0
    output = Path(output).expanduser().resolve()
    report_path = output / "report.json"
    if file_hash(report_path) != expected_report_sha256:
        raise ValueError("Bound diagnostic dynamic report is required")
    report = _strict_json(report_path)
    if not isinstance(report, dict):
        raise ValueError("Diagnostic dynamic report must be an object")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _strict_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Diagnostic source manifest must be an object")
    validate_diagnostic_manifest(manifest, replay=True)
    actor = NumPyNativeActor(actor_path)
    protocol_path = output / "protocol.json"
    prefilter_path = output / "prefilter.jsonl"
    episodes_path = output / "episodes.jsonl"
    artifacts = report.get("evidence_artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("protocol.json") != file_hash(protocol_path)
        or artifacts.get("prefilter.jsonl") != file_hash(prefilter_path)
        or artifacts.get("episodes.jsonl") != file_hash(episodes_path)
    ):
        raise ValueError("Diagnostic dynamic evidence hashes differ")
    protocol = _strict_json(protocol_path)
    _validate_reissue_provenance(report, protocol, output)
    expected_source_sha = _current_source_sha256()
    if (
        actor.sha256 != FROZEN_ACTOR_SHA256
        or report.get("actor", {}).get("sha256") != actor.sha256
        or report.get("source_manifest_file_sha256") != file_hash(manifest_path)
        or report.get("source_manifest_content_sha256") != manifest["content_sha256"]
        or report.get("diagnostic_contract_version") != DIAGNOSTIC_CONTRACT_VERSION
        or report.get("diagnostic_contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
        or report.get("diagnostic_conflict_graph_sha256")
        != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
        or report.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256
        or report.get("protocol_sha256") != digest(PROTOCOL)
        or report.get("actor_action_override_frames") != 0
        or report.get("new_pickup_on_agent") != 0
        or report.get("new_delivery_on_agent") != 0
        or report.get("new_endpoint_on_agent") != 0
        or report.get("immediate_task_recreation") != 0
        or not isinstance(protocol, dict)
        or protocol.get("version") != VERSION
        or protocol.get("protocol") != PROTOCOL
        or protocol.get("protocol_sha256") != digest(PROTOCOL)
        or protocol.get("source_sha256") != expected_source_sha
        or protocol.get("actor", {}).get("sha256") != actor.sha256
        or protocol.get("source_manifest_file_sha256") != file_hash(manifest_path)
        or protocol.get("source_manifest_content_sha256")
        != manifest["content_sha256"]
        or protocol.get("diagnostic_contract_version")
        != DIAGNOSTIC_CONTRACT_VERSION
        or protocol.get("diagnostic_contract_sha256")
        != DIAGNOSTIC_CONTRACT_SHA256
        or protocol.get("diagnostic_conflict_graph_sha256")
        != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
    ):
        raise ValueError("Diagnostic dynamic report binding differs")

    candidate_by_id = {
        row["id"]: row
        for batch in manifest["candidate_batches"]
        for row in batch
    }
    prefilter_rows = _strict_jsonl(prefilter_path)
    if len(prefilter_rows) != report.get("prefilter_evaluated"):
        raise ValueError("Diagnostic prefilter row count differs")
    prefilter_by_id = {}
    for row in prefilter_rows:
        scene_id = row.get("scene_id")
        scene = candidate_by_id.get(scene_id)
        if (
            scene is None
            or scene_id in prefilter_by_id
            or row.get("family_id") != scene["family_id"]
            or row.get("batch_index") != scene["batch_index"]
            or row.get("seed") != scene["seed"]
            or row.get("actor_action_override_frames") != 0
            or row.get("new_pickup_on_agent") != 0
            or row.get("new_delivery_on_agent") != 0
            or row.get("new_endpoint_on_agent") != 0
            or row.get("immediate_task_recreation") != 0
        ):
            raise ValueError("Diagnostic prefilter provenance differs")
        prefilter_by_id[scene_id] = row
    considered = int(report.get("batches_considered", -1))
    if not 1 <= considered <= len(manifest["candidate_batches"]):
        raise ValueError("Diagnostic considered-batch count differs")
    expected_prefilter_ids = {
        row["id"]
        for batch in manifest["candidate_batches"][:considered]
        for row in batch
    }
    if set(prefilter_by_id) != expected_prefilter_ids:
        raise ValueError("Diagnostic prefilter did not cover complete batch prefixes")

    expected_shortlist_ids = set()
    for batch_index in range(considered):
        for family in FAMILY_IDS:
            rows = [
                row
                for row in manifest["candidate_batches"][batch_index]
                if row["family_id"] == family
            ]
            rows.sort(
                key=lambda scene: (
                    prefilter_by_id[scene["id"]]["score"],
                    scene["seed"],
                    scene["id"],
                )
            )
            expected_shortlist_ids.update(
                row["id"] for row in rows[:FULL_SHORTLIST_PER_FAMILY]
            )

    episode_rows = _strict_jsonl(episodes_path)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        by_scene[row.get("scene_id")].append(row)
    evaluated_ids = report.get("evaluated_scene_ids")
    if (
        not isinstance(evaluated_ids, list)
        or len(evaluated_ids) != len(set(evaluated_ids))
        or set(evaluated_ids) != expected_shortlist_ids
        or set(by_scene) != expected_shortlist_ids
        or report.get("full_dynamic_evaluated") != len(evaluated_ids)
    ):
        raise ValueError("Diagnostic full-evaluation shortlist differs")
    profiles = (*BASELINES, "compatible_reference")
    evaluated = []
    metrics_by_id = {
        row.get("id"): row for row in report.get("scene_metrics", [])
    }
    if set(metrics_by_id) != expected_shortlist_ids:
        raise ValueError("Diagnostic scene-metric index differs")
    for scene_id in evaluated_ids:
        scene = candidate_by_id[scene_id]
        rows = by_scene[scene_id]
        expected_keys = {
            (profile, seed) for profile in profiles for seed in CONTINUATION_SEEDS
        }
        if (
            len(rows) != len(expected_keys)
            or {(row.get("profile"), row.get("continuation_seed")) for row in rows}
            != expected_keys
            or any(
                row.get("actor_action_override_frames") != 0
                or row.get("new_pickup_on_agent") != 0
                or row.get("new_delivery_on_agent") != 0
                or row.get("new_endpoint_on_agent") != 0
                or row.get("immediate_task_recreation") != 0
                or row.get("family_id") != scene["family_id"]
                or row.get("batch_index") != scene["batch_index"]
                or row.get("derived_continuation_seed")
                != _derived_continuation_seed(scene, row["continuation_seed"])
                for row in rows
            )
        ):
            raise ValueError("Diagnostic full episode protocol differs")
        recomputed = _strict_metrics(rows)
        saved_metrics = metrics_by_id[scene_id]
        if (
            saved_metrics.get("seed") != scene["seed"]
            or saved_metrics.get("batch_index") != scene["batch_index"]
            or saved_metrics.get("family_id") != scene["family_id"]
            or saved_metrics.get("initial_edge_id") != scene["initial_edge_id"]
            or canonical(saved_metrics.get("prefilter"))
            != canonical(
                {
                    key: value
                    for key, value in prefilter_by_id[scene_id].items()
                    if key not in {"scene_id", "batch_index", "family_id", "seed"}
                }
            )
            or canonical(saved_metrics.get("dynamic")) != canonical(recomputed)
        ):
            raise ValueError("Diagnostic dynamic metrics do not replay from episodes")
        evaluated.append(
            {
                **deepcopy(scene),
                "prefilter": {
                    key: deepcopy(value)
                    for key, value in prefilter_by_id[scene_id].items()
                    if key not in {"scene_id", "batch_index", "family_id", "seed"}
                },
                "dynamic": recomputed,
            }
        )
    recomputed_selection = select_balanced_six(evaluated)
    recomputed_counts = {
        "dynamic_passed": sum(row["dynamic"]["passed"] for row in evaluated),
        "actor_submission_frames": sum(
            row["dynamic"]["actor_submission_frames"] for row in evaluated
        ),
        "actor_action_override_frames": sum(
            row["dynamic"]["actor_action_override_frames"] for row in evaluated
        ),
        "replacement_tasks": sum(
            row["dynamic"]["replacement_tasks"] for row in evaluated
        ),
        "new_pickup_on_agent": sum(
            row["dynamic"]["new_pickup_on_agent"] for row in evaluated
        ),
        "new_delivery_on_agent": sum(
            row["dynamic"]["new_delivery_on_agent"] for row in evaluated
        ),
        "new_endpoint_on_agent": sum(
            row["dynamic"]["new_endpoint_on_agent"] for row in evaluated
        ),
        "immediate_task_recreation": sum(
            row["dynamic"]["immediate_task_recreation"] for row in evaluated
        ),
        "successor_sampling_failures": sum(
            row["dynamic"]["successor_sampling_failures"] for row in evaluated
        ),
    }
    if any(report.get(key) != value for key, value in recomputed_counts.items()):
        raise ValueError("Diagnostic aggregate dynamic metrics differ")
    recomputed_passed_by_family = dict(
        Counter(
            row["family_id"] for row in evaluated if row["dynamic"]["passed"]
        )
    )
    if report.get("dynamic_passed_by_family") != recomputed_passed_by_family:
        raise ValueError("Diagnostic per-family pass counts differ")
    expected_report_selection = (
        None
        if recomputed_selection is None
        else {
            key: deepcopy(recomputed_selection[key])
            for key in (
                "pairs",
                "balance",
                "score",
                "pairing_attempts",
                "dynamic_pass_count",
            )
        }
    )
    if (
        canonical(report.get("selection")) != canonical(expected_report_selection)
        or report.get("release_eligible") is not (recomputed_selection is not None)
        or report.get("status")
        != (
            "accepted_diagnostic_dynamic_selection"
            if recomputed_selection is not None
            else "blocked_no_six_family_dynamic_selection"
        )
    ):
        raise ValueError("Diagnostic report selection does not replay")
    selected = None
    if selected_scenes_path is not None:
        selected_path = Path(selected_scenes_path).expanduser().resolve()
        selected = _strict_json(selected_path)
        if (
            report.get("evidence_artifacts", {}).get("selected_scenes.json")
            != file_hash(selected_path)
            or selected.get("status") != "accepted_diagnostic_dynamic_selection"
            or selected.get("release_eligible") is not True
            or selected.get("actor_sha256") != actor.sha256
            or selected.get("source_manifest_file_sha256") != file_hash(manifest_path)
            or selected.get("source_manifest_content_sha256")
            != manifest["content_sha256"]
            or selected.get("protocol_sha256") != digest(PROTOCOL)
            or selected.get("six_distinct_conflict_families") is not True
            or selected.get("zero_action_overrides") is not True
            or selected.get("no_replacement_pickup_on_agent") is not True
            or selected.get("no_replacement_delivery_on_agent") is not True
            or selected.get("no_replacement_endpoint_on_agent") is not True
        ):
            raise ValueError("Diagnostic selected-scene evidence differs")
        rows = [*selected.get("X", []), *selected.get("Y", [])]
        if (
            len(selected.get("X", [])) != 3
            or len(selected.get("Y", [])) != 3
            or set(row.get("family_id") for row in rows) != set(FAMILY_IDS)
            or len({row.get("fingerprint") for row in rows}) != 6
        ):
            raise ValueError("Diagnostic selection does not cover six families")
        if recomputed_selection is None:
            raise ValueError("Saved diagnostic selection cannot be recomputed")
        expected_selected = {
            "version": VERSION,
            "status": "accepted_diagnostic_dynamic_selection",
            "release_eligible": True,
            "actor_sha256": actor.sha256,
            "source_manifest_file_sha256": file_hash(manifest_path),
            "source_manifest_content_sha256": manifest["content_sha256"],
            "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
            "protocol_sha256": digest(PROTOCOL),
            "tutorial": deepcopy(manifest["splits"]["tutorial"][0]),
            "X": [_public_scene(row) for row in recomputed_selection["X"]],
            "Y": [_public_scene(row) for row in recomputed_selection["Y"]],
            "pairs": deepcopy(recomputed_selection["pairs"]),
            "balance": deepcopy(recomputed_selection["balance"]),
            "selection_score": recomputed_selection["score"],
            "six_distinct_conflict_families": True,
            "zero_action_overrides": True,
            "no_replacement_pickup_on_agent": True,
            "no_replacement_delivery_on_agent": True,
            "no_replacement_endpoint_on_agent": True,
            "ordinary_sampler_fallback": False,
        }
        if canonical(selected) != canonical(expected_selected):
            raise ValueError("Diagnostic selected scenes differ from episode replay")
        selected_ids = {row["id"] for row in rows}
        for scene_id in selected_ids:
            dynamic = next(
                row["dynamic"] for row in evaluated if row["id"] == scene_id
            )
            if (
                dynamic["passed"] is not True
                or dynamic["checks"].get("collision_recovery_at_least_90_percent")
                is not True
                or dynamic["checks"].get("actor_actions_submitted_unchanged")
                is not True
                or dynamic["checks"].get("no_new_pickup_on_agent") is not True
                or dynamic["checks"].get("no_new_delivery_on_agent") is not True
                or dynamic["checks"].get("no_new_endpoint_on_agent") is not True
                or dynamic["checks"].get("no_successor_sampling_failure") is not True
            ):
                raise ValueError("A selected diagnostic scene fails a hard gate")
            scene = candidate_by_id[scene_id]
            saved_by_key = {
                (row["profile"], row["continuation_seed"]): row
                for row in by_scene[scene_id]
            }
            for profile in profiles:
                for continuation_seed in CONTINUATION_SEEDS:
                    replayed = run_actor_episode(
                        scene, actor, profile, continuation_seed
                    )
                    if canonical(replayed) != canonical(
                        saved_by_key[(profile, continuation_seed)]
                    ):
                        raise ValueError(
                            "Selected diagnostic physical episode did not replay"
                        )
                    selected_physical_replay_episodes += 1
        for row in rows:
            env = R41DiagnosticConflictWarehouseEnv()
            reset_diagnostic_scenario(env, row)
    if require_selected and selected is None:
        raise ValueError("A release-eligible diagnostic selection is required")
    return {
        "version": VERSION,
        "passed": bool(
            report.get("release_eligible") is True
            and selected is not None
            and recomputed_selection is not None
        ),
        "report_sha256": expected_report_sha256,
        "actor_sha256": actor.sha256,
        "manifest_file_sha256": file_hash(manifest_path),
        "selected_scenes_sha256": (
            None
            if selected_scenes_path is None
            else file_hash(selected_scenes_path)
        ),
        "selection": selected,
        "dynamic_passed": recomputed_counts["dynamic_passed"],
        "batches_considered": considered,
        "actor_submission_frames": recomputed_counts["actor_submission_frames"],
        "actor_action_override_frames": 0,
        "new_pickup_on_agent": 0,
        "new_delivery_on_agent": 0,
        "new_endpoint_on_agent": 0,
        "immediate_task_recreation": 0,
        "selected_physical_replay_passed": bool(
            selected is not None and selected_physical_replay_episodes == 720
        ),
        "selected_physical_replay_episodes": selected_physical_replay_episodes,
        "selected_family_ids": (
            []
            if selected is None
            else sorted(row["family_id"] for row in (*selected["X"], *selected["Y"]))
        ),
    }


def reissue_from_verified_journals(
    source_output: str | Path,
    *,
    expected_source_report_sha256: str,
    actor_path: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Reissue source-bound evidence without rerunning unselected episodes.

    The prior report only authenticates immutable journals.  Its accepted
    status, saved metrics, and selected rows are never used as a gate: the
    ordinary saved reader recomputes the complete batch prefix, every full
    scene metric, and the balanced selection with this source, then physically
    replays all 720 selected profile/continuation episodes before the fresh
    directory is published.
    """

    source = _reissue_source_evidence(
        source_output,
        expected_report_sha256=expected_source_report_sha256,
        actor_path=actor_path,
        manifest_path=manifest_path,
    )
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("Diagnostic reissue output must stay inside repository") from None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ValueError("Diagnostic reissue output parent is unsafe")

    manifest_path = Path(manifest_path).expanduser().resolve()
    actor_path = Path(actor_path).expanduser().resolve()
    manifest = source["manifest"]
    actor = source["actor"]
    current_sources = _current_source_sha256()
    created = datetime.now(timezone.utc).isoformat()
    record = {
        "version": _REISSUE_VERSION,
        "source_output": source["source_relative"],
        "source_report_sha256": expected_source_report_sha256,
        "source_evidence_artifacts": deepcopy(source["artifact_sha256"]),
        "source_producer_sha256": deepcopy(source["source_producer_sha256"]),
        "source_status_not_trusted": True,
        "journals_copied_byte_exact": True,
        "canonical_journal_rows_validated": True,
        "complete_batch_prefix_recomputed": True,
        "all_scene_metrics_recomputed": True,
        "balanced_selection_recomputed": True,
        "selected_physical_replay_episodes": 720,
    }
    protocol = deepcopy(source["protocol"])
    protocol.update({
        "created_utc": created,
        "source_manifest": str(manifest_path),
        "source_manifest_file_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "actor": {
            "path": str(actor_path),
            "sha256": actor.sha256,
            "deterministic": True,
            "post_policy_overrides": 0,
        },
        "source_sha256": current_sources,
        "evidence_reissue": record,
    })

    candidates = {
        row["id"]: row
        for batch in manifest["candidate_batches"]
        for row in batch
    }
    prior_selected = source["selected"]

    def rebuilt_group(name: str) -> list[dict[str, Any]]:
        rows = prior_selected.get(name)
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError("Source diagnostic selection group differs")
        ids = [row.get("id") if isinstance(row, Mapping) else None for row in rows]
        if len(set(ids)) != 3 or any(scene_id not in candidates for scene_id in ids):
            raise ValueError("Source diagnostic selected-scene identity differs")
        return [_public_scene(candidates[scene_id]) for scene_id in ids]

    selected = {
        "version": VERSION,
        "status": "accepted_diagnostic_dynamic_selection",
        "release_eligible": True,
        "actor_sha256": actor.sha256,
        "source_manifest_file_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "protocol_sha256": digest(PROTOCOL),
        "tutorial": deepcopy(manifest["splits"]["tutorial"][0]),
        "X": rebuilt_group("X"),
        "Y": rebuilt_group("Y"),
        "pairs": deepcopy(prior_selected.get("pairs")),
        "balance": deepcopy(prior_selected.get("balance")),
        "selection_score": prior_selected.get("selection_score"),
        "six_distinct_conflict_families": True,
        "zero_action_overrides": True,
        "no_replacement_pickup_on_agent": True,
        "no_replacement_delivery_on_agent": True,
        "no_replacement_endpoint_on_agent": True,
        "ordinary_sampler_fallback": False,
    }
    report = deepcopy(source["report"])
    report.update({
        "created_utc": created,
        "protocol": deepcopy(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "source_manifest": str(manifest_path),
        "source_manifest_file_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "actor": deepcopy(protocol["actor"]),
        "source_sha256": current_sources,
        "evidence_reissue": record,
    })

    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".reissue-", dir=destination.parent
    ))
    try:
        _write_exclusive(temporary / "prefilter.jsonl", source["prefilter_raw"])
        _write_exclusive(temporary / "episodes.jsonl", source["episodes_raw"])
        _write_exclusive(temporary / "protocol.json", _json_bytes(protocol))
        _write_exclusive(temporary / "selected_scenes.json", _json_bytes(selected))
        report["evidence_artifacts"] = {
            "protocol.json": file_hash(temporary / "protocol.json"),
            "prefilter.jsonl": file_hash(temporary / "prefilter.jsonl"),
            "episodes.jsonl": file_hash(temporary / "episodes.jsonl"),
            "selected_scenes.json": file_hash(temporary / "selected_scenes.json"),
        }
        _write_exclusive(temporary / "report.json", _json_bytes(report))
        report_sha256 = file_hash(temporary / "report.json")
        replay = read_saved_diagnostic_selection(
            temporary,
            expected_report_sha256=report_sha256,
            actor_path=actor_path,
            manifest_path=manifest_path,
            selected_scenes_path=temporary / "selected_scenes.json",
            require_selected=True,
        )
        if (
            replay.get("passed") is not True
            or replay.get("selected_physical_replay_passed") is not True
            or replay.get("selected_physical_replay_episodes") != 720
        ):
            raise ValueError("Diagnostic journal reissue did not pass physical replay")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        **report,
        "report_sha256": file_hash(destination / "report.json"),
        "selected_scenes_sha256": file_hash(destination / "selected_scenes.json"),
        "reissued_output": str(destination),
        "selected_physical_replay_passed": True,
        "selected_physical_replay_episodes": 720,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reissue-from")
    parser.add_argument("--expected-source-report-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.reissue_from) != bool(args.expected_source_report_sha256):
        raise ValueError(
            "--reissue-from and --expected-source-report-sha256 are required together"
        )
    report = (
        reissue_from_verified_journals(
            args.reissue_from,
            expected_source_report_sha256=args.expected_source_report_sha256,
            actor_path=args.actor,
            manifest_path=args.manifest,
            output=args.output,
        )
        if args.reissue_from
        else audit(args.manifest, args.actor, args.output, workers=args.workers)
    )
    print(canonical({
        "status": report["status"],
        "release_eligible": report["release_eligible"],
        "dynamic_passed": report["dynamic_passed"],
        "batches_considered": report["batches_considered"],
        "reissued": bool(args.reissue_from),
    }))
    return 0 if report["release_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
