"""Frozen-Actor dynamic audit and final play selection for warehouse r4.1.

The scene producer's six ``splits.play`` rows are static placeholders only.
This module evaluates all sixty pre-registered candidates with five simple
partners and one compatible reference, over twenty continuation RNG seeds.
Only a six-scene, six-geometry selection passing every dynamic and balance gate
is written as release-eligible output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import mean
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
from backend.training.warehouse_r41_conflict_scenarios import (
    MANIFEST_VERSION,
    file_hash,
    reset_conflict_scenario,
    validate_conflict_manifest,
)
from backend.warehouse_r41_online_runtime import (
    DEFAULT_REWARD_CONFIG,
    R41ConflictWarehouseEnv,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH_SHA256,
    CONTRACT_SHA256,
    R41ConflictSamplingError,
    canonical,
    digest,
    r41_scene_fingerprint,
)


VERSION = "warehouse-r41-conflict-dynamic-play-selection.v1"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "version": VERSION,
    "scope": "development scene selection only; never Actor selection or treatment-effect estimation",
    "source_static_play_status": "ignored_provisional_not_release_eligible",
    "candidate_count": 60,
    "profiles": [*BASELINES, "compatible_reference"],
    "continuation_seeds": list(CONTINUATION_SEEDS),
    "runs_per_profile": len(CONTINUATION_SEEDS),
    "full_horizon_replay": True,
    "dynamic_gates": deepcopy(DYNAMIC),
    "pairing_gates": deepcopy(PAIRING),
    "additional_pairing_gates": {
        "six_distinct_initial_conflict_edges": True,
        "pair_reference_delivery_difference_max": 1.0,
        "pair_conflict_opportunity_difference_max": 0.03,
        "group_workload_relative_difference_max": 0.05,
        "group_conflict_relative_difference_max": 0.05,
    },
    "actor_role": "robot_2 deterministic raw argmax; zero post-policy overrides",
    "task_successors": "r4.1 conditional conflict graph for the complete episode",
    "participant_data_read": False,
    "final_test_rollouts": 0,
}
_WORKER_ACTOR: NumPyNativeActor | None = None


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def actor_environment(actor: NumPyNativeActor, scene: Mapping[str, Any]) -> R41ConflictWarehouseEnv:
    if actor.obs_dim != 197:
        raise ValueError("r4.1 play audit requires the observed197 Actor")
    env = R41ConflictWarehouseEnv(reward_config=deepcopy(DEFAULT_REWARD_CONFIG))
    reset_conflict_scenario(env, scene)
    if list(env.feature_names) != actor.metadata.get("feature_names"):
        raise ValueError("Actor feature contract differs from the r4.1 environment")
    return env


def run_actor_episode(
    scene: Mapping[str, Any],
    actor: NumPyNativeActor,
    profile: str,
    continuation_seed: int,
) -> dict[str, Any]:
    if profile not in (*BASELINES, "compatible_reference"):
        raise ValueError("Unknown r4.1 dynamic-audit profile")
    if continuation_seed not in CONTINUATION_SEEDS:
        raise ValueError("Continuation seed was not pre-registered")
    env = actor_environment(actor, scene)
    # Initial state/headings remain frozen.  Only the future conditional task
    # sampler changes across the twenty pre-registered continuations.
    env.set_rng_state(random.Random(continuation_seed).getstate())
    initial_fingerprint = r41_scene_fingerprint(env)
    pending_collisions: list[dict[str, Any]] = []
    recovered = 0
    steps = opportunities = collisions = actor_submission_frames = actor_action_override_frames = 0
    risky_mass = 0.0
    no_progress_streak = longest_no_progress = consecutive_collisions = longest_collisions = 0
    spawned_on_agent_endpoint = replacement_tasks = 0
    immediate_task_recreation = new_endpoint_on_agent = 0
    new_pickup_on_agent = new_delivery_on_agent = 0
    conflict_pair_checks = 0
    successor_sampling_failure = None
    while not env.done:
        actor_action = _actor_action(actor, env)
        dangerous = _dangerous_player_actions(env, actor_action)
        legal = _legal_player_actions(env)
        player_action = (
            compatible_reference_action(env, actor_action)
            if profile == "compatible_reference"
            else simple_action(env, profile)
        )
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
        if info["requested_actions"]["robot_2"] != actor_action:
            raise AssertionError("r4.1 Actor action was not submitted unchanged")
        conflict_pair_checks += 1
        creation = info["r41_conflict"]["created"]
        replacement_tasks += len(creation)
        spawned_on_agent_endpoint += sum(
            bool(row["spawned_on_agent_endpoint"]) for row in creation
        )
        immediate_task_recreation += sum(
            bool(row["immediate_task_recreation"]) for row in creation
        )
        new_endpoint_on_agent += sum(
            bool(row["new_endpoint_on_agent"]) for row in creation
        )
        new_pickup_on_agent += sum(
            bool(row["new_pickup_on_agent"]) for row in creation
        )
        new_delivery_on_agent += sum(
            bool(row["new_delivery_on_agent"]) for row in creation
        )
        steps += 1
        actor_submission_frames += 1
        actor_action_override_frames += int(info["requested_actions"]["robot_2"] != actor_action)
        opportunities += int(bool(dangerous))
        risky_mass += len(dangerous) / len(legal)
        progress = bool(
            env.state.total_deliveries > before_deliveries
            or sum(task.status == "carried" for task in env.state.tasks) > before_pickups
            or env.potential() > before_potential + 1e-9
        )
        no_progress_streak = 0 if progress else no_progress_streak + 1
        longest_no_progress = max(longest_no_progress, no_progress_streak)
        collided = bool(info["robot_collision"])
        collisions += int(collided)
        consecutive_collisions = consecutive_collisions + 1 if collided else 0
        longest_collisions = max(longest_collisions, consecutive_collisions)
        if collided:
            pending_collisions.append({"frame": env.state.frame, "resolved": False})
        recovered += update_collision_recovery(
            pending_collisions, env.state.frame, progress
        )
    return {
        "scene_id": scene["id"],
        "seed": scene["seed"],
        "initial_edge_id": scene["initial_edge_id"],
        "profile": profile,
        "continuation_seed": continuation_seed,
        "initial_continuation_fingerprint": initial_fingerprint,
        "steps": steps,
        "deliveries": env.state.total_deliveries,
        "deliveries_by_robot": {
            agent.agent_id: agent.deliveries_completed for agent in env.state.agents
        },
        "collisions": collisions,
        "shutdowns": env.state.shutdown_count,
        "conflict_opportunity_frames": opportunities,
        "player_risky_action_mass_sum": risky_mass,
        "recovered_collisions_within_10": recovered,
        "longest_consecutive_collisions": longest_collisions,
        "longest_no_progress_streak": longest_no_progress,
        "terminal_consecutive_collisions": consecutive_collisions,
        "terminal_no_progress_streak": no_progress_streak,
        "actor_submission_frames": actor_submission_frames,
        "actor_action_override_frames": actor_action_override_frames,
        "replacement_tasks": replacement_tasks,
        "spawned_on_agent_endpoint": spawned_on_agent_endpoint,
        "immediate_task_recreation": immediate_task_recreation,
        "new_endpoint_on_agent": new_endpoint_on_agent,
        "new_pickup_on_agent": new_pickup_on_agent,
        "new_delivery_on_agent": new_delivery_on_agent,
        "active_conflict_pair_checks": conflict_pair_checks,
        "successor_sampling_failure": successor_sampling_failure,
        "terminal_reason": (
            "conflict_successor_unavailable"
            if successor_sampling_failure is not None
            else env.state.terminal_reason
        ),
    }


def _worker_initialize(actor_path: str) -> None:
    global _WORKER_ACTOR
    _WORKER_ACTOR = NumPyNativeActor(actor_path)


def _evaluate_worker(scene: Mapping[str, Any]):
    if _WORKER_ACTOR is None:
        raise RuntimeError("r4.1 play worker has no frozen Actor")
    episodes = [
        run_actor_episode(scene, _WORKER_ACTOR, profile, continuation_seed)
        for profile in (*BASELINES, "compatible_reference")
        for continuation_seed in CONTINUATION_SEEDS
    ]
    metrics = dynamic_metrics(episodes)
    metrics["replacement_tasks"] = sum(row["replacement_tasks"] for row in episodes)
    metrics["spawned_on_agent_endpoint"] = sum(
        row["spawned_on_agent_endpoint"] for row in episodes
    )
    metrics["immediate_task_recreation"] = sum(
        row["immediate_task_recreation"] for row in episodes
    )
    metrics["new_endpoint_on_agent"] = sum(
        row["new_endpoint_on_agent"] for row in episodes
    )
    metrics["new_pickup_on_agent"] = sum(
        row["new_pickup_on_agent"] for row in episodes
    )
    metrics["new_delivery_on_agent"] = sum(
        row["new_delivery_on_agent"] for row in episodes
    )
    metrics["checks"]["no_immediate_task_recreation"] = (
        metrics["immediate_task_recreation"] == 0
    )
    metrics["checks"]["no_new_pickup_on_agent"] = (
        metrics["new_pickup_on_agent"] == 0
    )
    metrics["successor_sampling_failures"] = sum(
        row["successor_sampling_failure"] is not None for row in episodes
    )
    metrics["checks"]["no_successor_sampling_failure"] = (
        metrics["successor_sampling_failures"] == 0
    )
    metrics["passed"] = all(metrics["checks"].values())
    metrics["active_conflict_pair_checks"] = sum(
        row["active_conflict_pair_checks"] for row in episodes
    )
    return episodes, metrics


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(1e-12, (abs(left) + abs(right)) / 2.0)


def _quality(row: Mapping[str, Any]) -> tuple[float, int, str]:
    dynamic = row["dynamic"]
    score = (
        abs(dynamic["conflict_opportunity_fraction"] - 0.325) / 0.075
        + abs(dynamic["player_risky_action_mass"] - 0.15) / 0.05
        + abs(dynamic["collision_cancellation_fraction"] - 0.13) / 0.05
        + abs(row["initial_conflict"]["shared_edge_ratio_min"] - 0.40) / 0.15
        - min(dynamic["reference_absolute_delivery_gap"], 4.0) / 4.0
    )
    return score, int(row["seed"]), str(row["id"])


def select_balanced_six(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select the first exact dynamic/balance solution without weakening gates."""
    passed = sorted(
        (row for row in candidates if row["dynamic"]["passed"]), key=_quality
    )
    if len(passed) < 6 or len({row["initial_edge_id"] for row in passed}) < 6:
        return None

    def pair_compatible(left: int, right: int) -> bool:
        a, b = passed[left], passed[right]
        return bool(
            a["fingerprint"] != b["fingerprint"]
            and a["initial_edge_id"] != b["initial_edge_id"]
            and abs(
                a["dynamic"]["compatible_reference_mean_deliveries"]
                - b["dynamic"]["compatible_reference_mean_deliveries"]
            ) <= 1.0
            and abs(
                a["dynamic"]["conflict_opportunity_fraction"]
                - b["dynamic"]["conflict_opportunity_fraction"]
            ) <= 0.03
        )

    attempts = 0

    def orient(matching: tuple[tuple[int, int], ...]):
        nonlocal attempts
        attempts += 1
        indices = tuple(index for pair in matching for index in pair)
        if len(set(indices)) != 6:
            return None
        rows = [passed[index] for index in indices]
        if len({row["fingerprint"] for row in rows}) != 6:
            return None
        if len({row["initial_edge_id"] for row in rows}) != 6:
            return None
        best = None
        for orientation in range(8):
            x_indices, y_indices = [], []
            for pair_index, pair in enumerate(matching):
                left, right = pair
                if orientation & (1 << pair_index):
                    left, right = right, left
                x_indices.append(left)
                y_indices.append(right)
            workload_x = mean(
                passed[index]["initial_conflict"]["initial_joint_work_steps"]
                for index in x_indices
            )
            workload_y = mean(
                passed[index]["initial_conflict"]["initial_joint_work_steps"]
                for index in y_indices
            )
            conflict_x = mean(
                passed[index]["dynamic"]["conflict_opportunity_fraction"]
                for index in x_indices
            )
            conflict_y = mean(
                passed[index]["dynamic"]["conflict_opportunity_fraction"]
                for index in y_indices
            )
            workload_difference = _relative_difference(workload_x, workload_y)
            conflict_difference = _relative_difference(conflict_x, conflict_y)
            if workload_difference > 0.05 or conflict_difference > 0.05:
                continue
            score = (
                workload_difference
                + conflict_difference
                + sum(
                    abs(
                        passed[a]["dynamic"]["conflict_opportunity_fraction"]
                        - passed[b]["dynamic"]["conflict_opportunity_fraction"]
                    )
                    for a, b in matching
                )
                + 0.001 * sum(_quality(passed[index])[0] for index in indices)
            )
            result = {
                "score": score,
                "X": [deepcopy(passed[index]) for index in x_indices],
                "Y": [deepcopy(passed[index]) for index in y_indices],
                "pairs": [
                    [passed[x_indices[i]]["id"], passed[y_indices[i]]["id"]]
                    for i in range(3)
                ],
                "balance": {
                    "X_mean_initial_workload": workload_x,
                    "Y_mean_initial_workload": workload_y,
                    "workload_relative_difference": workload_difference,
                    "X_mean_conflict_opportunity": conflict_x,
                    "Y_mean_conflict_opportunity": conflict_y,
                    "conflict_relative_difference": conflict_difference,
                },
            }
            tie = tuple(row["seed"] for row in (*result["X"], *result["Y"]))
            if best is None or (score, tie) < best[0]:
                best = ((score, tie), result)
        return None if best is None else best[1]

    def search(
        available: tuple[int, ...],
        matching: tuple[tuple[int, int], ...],
        used_edges: frozenset[str],
    ):
        if len(matching) == 3:
            return orient(matching)
        needed = 2 * (3 - len(matching))
        if len(available) < needed:
            return None
        first = available[0]
        first_edge = passed[first]["initial_edge_id"]
        if first_edge not in used_edges:
            partners = sorted(
                (
                    other
                    for other in available[1:]
                    if passed[other]["initial_edge_id"] not in used_edges | {first_edge}
                    and pair_compatible(first, other)
                ),
                key=lambda other: (
                    abs(
                        passed[first]["dynamic"]["conflict_opportunity_fraction"]
                        - passed[other]["dynamic"]["conflict_opportunity_fraction"]
                    ),
                    _quality(passed[other]),
                ),
            )
            for other in partners:
                remaining = tuple(
                    index for index in available if index not in (first, other)
                )
                result = search(
                    remaining,
                    matching + ((first, other),),
                    used_edges
                    | {first_edge, passed[other]["initial_edge_id"]},
                )
                if result is not None:
                    result["pairing_attempts"] = attempts
                    result["dynamic_pass_count"] = len(passed)
                    return result
        return search(available[1:], matching, used_edges)

    return search(tuple(range(len(passed))), (), frozenset())


def _public_scene(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "split", "seed", "fingerprint", "contract_sha256",
        "conflict_graph_sha256", "task_geometry_signature", "initial_edge_id",
        "initial_conflict", "successor_replay", "snapshot",
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
        raise ValueError("r4.1 dynamic workers must be an integer from one to eight")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(manifest_path).expanduser().resolve()
    actor_path = Path(actor_path).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    validation = validate_conflict_manifest(manifest, replay=True)
    if validation["passed"] is not True or manifest["version"] != MANIFEST_VERSION:
        raise ValueError("r4.1 source scene manifest did not pass admission")
    candidates = manifest["candidate_pool"]
    if len(candidates) != 60:
        raise ValueError("r4.1 dynamic protocol requires exactly sixty candidates")
    actor = NumPyNativeActor(actor_path)
    if actor.obs_dim != 197:
        raise ValueError("r4.1 dynamic protocol requires an observed197 Actor")
    source_paths = (
        Path(__file__),
        ROOT / "backend/training/warehouse_r41_conflict_scenarios.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
        ROOT / "backend/training/warehouse_r4_conflict_scene_selection.py",
    )
    provenance = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": deepcopy(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "source_manifest": str(manifest_path),
        "source_manifest_file_sha256": sha256(manifest_bytes).hexdigest(),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "source_static_play_status": "ignored_provisional_not_release_eligible",
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "actor": {
            "path": str(actor_path),
            "sha256": actor.sha256,
            "deterministic": True,
            "post_policy_overrides": 0,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): file_hash(path) for path in source_paths
        },
        "participant_data_read": False,
        "final_test_rollouts": 0,
    }
    write_json(output / "protocol.json", provenance)
    started = time.perf_counter()
    evaluated = []
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        if workers == 1:
            _worker_initialize(str(actor_path))
            iterator = map(_evaluate_worker, candidates)
            pool = None
        else:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_initialize,
                initargs=(str(actor_path),),
            )
            iterator = pool.map(_evaluate_worker, candidates)
        try:
            for index, (scene, outcome) in enumerate(zip(candidates, iterator)):
                episodes, metrics = outcome
                for episode in episodes:
                    stream.write(canonical(episode) + "\n")
                stream.flush()
                evaluated.append({**deepcopy(scene), "dynamic": metrics})
                print(canonical({
                    "completed": index + 1,
                    "total": len(candidates),
                    "scene_id": scene["id"],
                    "passed": metrics["passed"],
                }), flush=True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    selection = select_balanced_six(evaluated)
    selected = None
    if selection is not None:
        selected = {
            "version": VERSION,
            "status": "accepted_final_dynamic_selection",
            "release_eligible": True,
            "actor_sha256": actor.sha256,
            "source_manifest_file_sha256": sha256(manifest_bytes).hexdigest(),
            "source_manifest_content_sha256": manifest["content_sha256"],
            "contract_sha256": CONTRACT_SHA256,
            "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
            "tutorial": deepcopy(manifest["splits"]["tutorial"][0]),
            "X": [_public_scene(row) for row in selection["X"]],
            "Y": [_public_scene(row) for row in selection["Y"]],
            "pairs": deepcopy(selection["pairs"]),
            "balance": deepcopy(selection["balance"]),
            "selection_score": selection["score"],
            "six_distinct_initial_edges": True,
            "static_placeholder_play_ignored": True,
        }
        write_json(output / "selected_scenes.json", selected)
    report = {
        **provenance,
        "status": (
            "accepted_final_dynamic_selection"
            if selected is not None
            else "blocked_no_six_scene_dynamic_selection"
        ),
        "release_eligible": selected is not None,
        "candidates_evaluated": len(evaluated),
        "dynamic_passed": sum(row["dynamic"]["passed"] for row in evaluated),
        "actor_submission_frames": sum(
            row["dynamic"]["actor_submission_frames"] for row in evaluated
        ),
        "actor_action_override_frames": sum(
            row["dynamic"]["actor_action_override_frames"] for row in evaluated
        ),
        "replacement_tasks": sum(row["dynamic"]["replacement_tasks"] for row in evaluated),
        "spawned_on_agent_endpoint": sum(
            row["dynamic"]["spawned_on_agent_endpoint"] for row in evaluated
        ),
        "immediate_task_recreation": sum(
            row["dynamic"]["immediate_task_recreation"] for row in evaluated
        ),
        "new_endpoint_on_agent": sum(
            row["dynamic"]["new_endpoint_on_agent"] for row in evaluated
        ),
        "new_pickup_on_agent": sum(
            row["dynamic"]["new_pickup_on_agent"] for row in evaluated
        ),
        "new_delivery_on_agent": sum(
            row["dynamic"]["new_delivery_on_agent"] for row in evaluated
        ),
        "successor_sampling_failures": sum(
            row["dynamic"]["successor_sampling_failures"] for row in evaluated
        ),
        "active_conflict_pair_checks": sum(
            row["dynamic"]["active_conflict_pair_checks"] for row in evaluated
        ),
        "selection": None if selection is None else {
            key: deepcopy(selection[key])
            for key in ("pairs", "balance", "score", "pairing_attempts", "dynamic_pass_count")
        },
        "scene_metrics": [
            {
                "id": row["id"],
                "seed": row["seed"],
                "initial_edge_id": row["initial_edge_id"],
                "dynamic": deepcopy(row["dynamic"]),
            }
            for row in evaluated
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if report["actor_action_override_frames"] != 0:
        raise ValueError("r4.1 dynamic audit observed a post-policy action override")
    if (report["immediate_task_recreation"] != 0
            or report["new_pickup_on_agent"] != 0):
        raise ValueError("r4.1 dynamic audit observed an invalid replacement pickup")
    if any(file_hash(path) != provenance["source_sha256"][str(path.relative_to(ROOT))]
           for path in source_paths):
        raise RuntimeError("r4.1 dynamic-audit source changed during evaluation")
    if manifest_path.read_bytes() != manifest_bytes or file_hash(actor_path) != actor.sha256:
        raise RuntimeError("r4.1 frozen audit input changed during evaluation")
    report["evidence_artifacts"] = {
        "protocol.json": file_hash(output / "protocol.json"),
        "episodes.jsonl": file_hash(output / "episodes.jsonl"),
        **({"selected_scenes.json": file_hash(output / "selected_scenes.json")}
           if selected is not None else {}),
    }
    write_json(output / "report.json", report)
    return report


def _strict_json(path: Path) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > 128 * 1024 * 1024):
        raise ValueError("r4.1 dynamic evidence file is missing or unsafe")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate field in r4.1 dynamic evidence")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("Non-finite r4.1 dynamic evidence: " + value)))


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > 512 * 1024 * 1024):
        raise ValueError("r4.1 dynamic episode journal is missing or unsafe")
    rows = []
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate field in r4.1 dynamic episode")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("Blank r4.1 dynamic episode row")
            value = json.loads(line, object_pairs_hook=pairs,
                               parse_constant=lambda token: (_ for _ in ()).throw(
                                   ValueError("Non-finite dynamic row: " + token)))
            if not isinstance(value, dict):
                raise ValueError(f"Dynamic episode row {number} is not an object")
            rows.append(value)
    return rows


def _metrics_from_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = dynamic_metrics(episodes)
    for name in (
        "replacement_tasks", "spawned_on_agent_endpoint",
        "immediate_task_recreation", "new_endpoint_on_agent",
        "new_pickup_on_agent", "new_delivery_on_agent",
        "active_conflict_pair_checks",
    ):
        metrics[name] = sum(int(row[name]) for row in episodes)
    metrics["successor_sampling_failures"] = sum(
        row["successor_sampling_failure"] is not None for row in episodes
    )
    metrics["checks"]["no_immediate_task_recreation"] = (
        metrics["immediate_task_recreation"] == 0
    )
    metrics["checks"]["no_new_pickup_on_agent"] = (
        metrics["new_pickup_on_agent"] == 0
    )
    metrics["checks"]["no_successor_sampling_failure"] = (
        metrics["successor_sampling_failures"] == 0
    )
    metrics["passed"] = all(metrics["checks"].values())
    return metrics


def read_saved_report(
    output: str | Path, *, expected_report_sha256: str,
    actor_path: str | Path, manifest_path: str | Path,
    selected_scenes_path: str | Path | None, require_selected: bool = True,
) -> dict[str, Any]:
    """Recompute the final six-scene selection from immutable episode rows."""
    output = Path(output).expanduser().resolve()
    report_path = output / "report.json"
    protocol_path = output / "protocol.json"
    episodes_path = output / "episodes.jsonl"
    actor_path = Path(actor_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if (not output.is_dir() or output.is_symlink()
            or file_hash(report_path) != expected_report_sha256
            or not actor_path.is_file() or actor_path.is_symlink()
            or not manifest_path.is_file() or manifest_path.is_symlink()):
        raise ValueError("Bound r4.1 dynamic audit inputs are required")
    report = _strict_json(report_path)
    protocol = _strict_json(protocol_path)
    manifest = _strict_json(manifest_path)
    validate_conflict_manifest(manifest, replay=True)
    actor = NumPyNativeActor(actor_path)
    expected_sources = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in (
            Path(__file__),
            ROOT / "backend/training/warehouse_r41_conflict_scenarios.py",
            ROOT / "backend/warehouse_r41_online_runtime.py",
            ROOT / "env/warehouse_native/r41_conflict.py",
            ROOT / "backend/training/warehouse_r4_conflict_scene_selection.py",
        )
    }
    expected_protocol = {
        "version": VERSION,
        "created_utc": protocol.get("created_utc"),
        "protocol": deepcopy(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "source_manifest": str(manifest_path),
        "source_manifest_file_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "source_static_play_status": "ignored_provisional_not_release_eligible",
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "actor": {"path": str(actor_path), "sha256": actor.sha256,
                  "deterministic": True, "post_policy_overrides": 0},
        "source_sha256": expected_sources,
        "participant_data_read": False,
        "final_test_rollouts": 0,
    }
    if (not isinstance(protocol.get("created_utc"), str)
            or canonical(protocol) != canonical(expected_protocol)):
        raise ValueError("R4.1 dynamic protocol provenance differs")
    artifacts = report.get("evidence_artifacts")
    if (not isinstance(artifacts, dict)
            or artifacts.get("protocol.json") != file_hash(protocol_path)
            or artifacts.get("episodes.jsonl") != file_hash(episodes_path)):
        raise ValueError("R4.1 dynamic evidence hashes differ")
    rows = _strict_jsonl(episodes_path)
    profiles = (*BASELINES, "compatible_reference")
    expected_keys = {
        (scene["id"], profile, seed)
        for scene in manifest["candidate_pool"]
        for profile in profiles for seed in CONTINUATION_SEEDS
    }
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("scene_id"), row.get("profile"), row.get("continuation_seed"))
        if key in by_key:
            raise ValueError("Duplicate r4.1 dynamic episode")
        by_key[key] = row
    if set(by_key) != expected_keys or len(rows) != 60 * len(profiles) * len(CONTINUATION_SEEDS):
        raise ValueError("R4.1 dynamic episode matrix is incomplete")
    evaluated = []
    stored_rows = report.get("scene_metrics")
    if not isinstance(stored_rows, list) or len(stored_rows) != 60:
        raise ValueError("R4.1 dynamic scene matrix is incomplete")
    stored_by_id = {row.get("id"): row for row in stored_rows}
    if len(stored_by_id) != 60:
        raise ValueError("Duplicate r4.1 dynamic scene summary")
    for scene in manifest["candidate_pool"]:
        episode_rows = []
        for profile in profiles:
            for seed in CONTINUATION_SEEDS:
                saved_episode = by_key[(scene["id"], profile, seed)]
                replayed_episode = run_actor_episode(scene, actor, profile, seed)
                if canonical(saved_episode) != canonical(replayed_episode):
                    raise ValueError("R4.1 dynamic episode differs from physical replay")
                episode_rows.append(saved_episode)
        metrics = _metrics_from_episodes(episode_rows)
        expected = {"id": scene["id"], "seed": scene["seed"],
                    "initial_edge_id": scene["initial_edge_id"],
                    "dynamic": metrics}
        if canonical(stored_by_id.get(scene["id"])) != canonical(expected):
            raise ValueError("R4.1 dynamic metrics differ from episode rows")
        evaluated.append({**deepcopy(scene), "dynamic": metrics})
    selection = select_balanced_six(evaluated)
    if require_selected and selection is None:
        raise ValueError("R4.1 dynamic audit has no eligible six-scene selection")
    expected_selected = None
    selected_sha = None
    if selection is not None:
        if selected_scenes_path is None:
            raise ValueError("Passing dynamic audit requires selected_scenes.json")
        selected_path = Path(selected_scenes_path).expanduser().resolve()
        if (selected_path != output / "selected_scenes.json"
                or not selected_path.is_file() or selected_path.is_symlink()):
            raise ValueError("R4.1 selected scenes path differs")
        expected_selected = {
            "version": VERSION,
            "status": "accepted_final_dynamic_selection",
            "release_eligible": True,
            "actor_sha256": actor.sha256,
            "source_manifest_file_sha256": file_hash(manifest_path),
            "source_manifest_content_sha256": manifest["content_sha256"],
            "contract_sha256": CONTRACT_SHA256,
            "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
            "tutorial": deepcopy(manifest["splits"]["tutorial"][0]),
            "X": [_public_scene(row) for row in selection["X"]],
            "Y": [_public_scene(row) for row in selection["Y"]],
            "pairs": deepcopy(selection["pairs"]),
            "balance": deepcopy(selection["balance"]),
            "selection_score": selection["score"],
            "six_distinct_initial_edges": True,
            "static_placeholder_play_ignored": True,
        }
        saved = _strict_json(selected_path)
        if canonical(saved) != canonical(expected_selected):
            raise ValueError("R4.1 selected scene artifact differs from recomputation")
        selected_sha = file_hash(selected_path)
        if artifacts.get("selected_scenes.json") != selected_sha:
            raise ValueError("R4.1 selected scene hash differs")
    elif set(artifacts) != {"protocol.json", "episodes.jsonl"}:
        raise ValueError("Failed r4.1 selection cannot expose selected scenes")
    expected_selection_summary = None if selection is None else {
        key: deepcopy(selection[key])
        for key in ("pairs", "balance", "score", "pairing_attempts", "dynamic_pass_count")
    }
    aggregate_fields = {
        "candidates_evaluated": len(evaluated),
        "dynamic_passed": sum(row["dynamic"]["passed"] for row in evaluated),
        "actor_submission_frames": sum(row["dynamic"]["actor_submission_frames"] for row in evaluated),
        "actor_action_override_frames": sum(row["dynamic"]["actor_action_override_frames"] for row in evaluated),
        "replacement_tasks": sum(row["dynamic"]["replacement_tasks"] for row in evaluated),
        "spawned_on_agent_endpoint": sum(row["dynamic"]["spawned_on_agent_endpoint"] for row in evaluated),
        "immediate_task_recreation": sum(row["dynamic"]["immediate_task_recreation"] for row in evaluated),
        "new_endpoint_on_agent": sum(row["dynamic"]["new_endpoint_on_agent"] for row in evaluated),
        "new_pickup_on_agent": sum(row["dynamic"]["new_pickup_on_agent"] for row in evaluated),
        "new_delivery_on_agent": sum(row["dynamic"]["new_delivery_on_agent"] for row in evaluated),
        "successor_sampling_failures": sum(row["dynamic"]["successor_sampling_failures"] for row in evaluated),
        "active_conflict_pair_checks": sum(row["dynamic"]["active_conflict_pair_checks"] for row in evaluated),
    }
    if (report.get("version") != VERSION
            or report.get("status") != ("accepted_final_dynamic_selection" if selection else "blocked_no_six_scene_dynamic_selection")
            or report.get("release_eligible") is not (selection is not None)
            or report.get("selection") != expected_selection_summary
            or report.get("actor") != expected_protocol["actor"]
            or report.get("source_manifest_file_sha256") != file_hash(manifest_path)
            or report.get("source_manifest_content_sha256") != manifest["content_sha256"]
            or report.get("source_sha256") != expected_sources
            or any(report.get(key) != value for key, value in expected_protocol.items())
            or any(report.get(key) != value for key, value in aggregate_fields.items())
            or report.get("participant_data_read") is not False
            or report.get("final_test_rollouts") != 0
            or not isinstance(report.get("elapsed_seconds"), (int, float))
            or not math.isfinite(float(report["elapsed_seconds"]))
            or report["elapsed_seconds"] < 0):
        raise ValueError("R4.1 dynamic report differs from immutable evidence")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit(args.manifest, args.actor, args.output, workers=args.workers)
    print(canonical({
        "status": report["status"],
        "release_eligible": report["release_eligible"],
        "candidates_evaluated": report["candidates_evaluated"],
        "dynamic_passed": report["dynamic_passed"],
    }))
    return 0 if report["release_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
