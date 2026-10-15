"""Independent six-partner audit for the warehouse r4.1 release.

The frozen r4.1 runner inherited a historical evaluation bug: ``fixed_yield``
only yields when it controls ``robot_2``, while validation always assigns the
program partner to ``robot_1``.  Its ``skilled`` and ``fixed_yield`` rows are
therefore the same policy.  This additive post-training audit fixes that role
locally without changing any frozen trainer, checkpoint, or training receipt.

Every committed 50k Actor is physically evaluated on both validation suites.
The six program partners must also produce pairwise-distinguishable action
traces on those evaluation episodes.  Production may admit only the Actor that
is both the frozen ledger selection and the earliest boundary passing this
corrected audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from itertools import combinations, zip_longest
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_active_evaluation as active_evaluation
from backend.training import warehouse_r41_training_ledger as training_ledger
from backend.training import warehouse_r4_active_evaluation as original_evaluation
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment
from backend.training.warehouse_native_public_feedback_evaluation import REWARD
from backend.training.warehouse_r4_active_trainer import shaping_for_transition
from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.partners import _goals, partner_action
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.scenarios import reset_scenario
from backend.training.warehouse_r41_conflict_scenarios import (
    reset_conflict_scenario,
    validate_conflict_manifest,
)


VERSION = "warehouse-r41-corrected-six-partner-audit.v1"
PARTNERS = (
    "skilled", "assertive", "noisy", "fixed_yield", "fixed_region", "fixed_task",
)
EVALUATION_SEED = 260_911_900
DISTINCTION_GATE = {
    "minimum_different_episodes_per_pair": 10,
    "minimum_aligned_action_disagreements_per_pair": 20,
    "minimum_aligned_action_disagreement_rate_per_pair": .005,
}
PROTOCOL = {
    "version": VERSION,
    "purpose": "additive post-training correction; no PPO or checkpoint mutation",
    "partners": list(PARTNERS),
    "fixed_yield": (
        "robot_1 waits whenever public shortest-path distance to robot_2 is at most two "
        "unless the skilled public energy goal requires charging; otherwise it uses the "
        "skilled public-state action"
    ),
    "same_episode_seed_across_partner_profiles": True,
    "suites": ["original_validation", "conflict_validation"],
    "suite_scene_counts": {
        "original_validation": 50, "conflict_validation": 64,
    },
    "episodes_per_actor": 684,
    "distinction_required_overall_and_per_suite": True,
    "candidate_actor_shutdown_count_max": 0,
    "all_committed_boundaries_replayed": True,
    "distinction_gate": deepcopy(DISTINCTION_GATE),
    "runtime_action_override": False,
    "ppo_joint_steps": 0,
}


def producer_sources() -> dict[str, str]:
    """Hash the executable closure that defines this independent audit."""
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    return local_source_hashes((Path(__file__),))


def _strict_json(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(label + " must be a regular file")

    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("Duplicate JSON field in " + label)
            value[key] = child
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must contain a JSON object")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def corrected_partner_action(env, agent_id: str, profile: str, rng) -> str:
    """Return one public-state partner action under the corrected protocol."""
    if profile not in PARTNERS or agent_id != "robot_1":
        raise ValueError("Corrected audit requires a registered robot_1 partner")
    if profile != "fixed_yield":
        return partner_action(env, agent_id, profile, rng)
    agent = env.state.by_id(agent_id)
    other = next(value for value in env.state.agents if value.agent_id != agent_id)
    if not agent.active:
        return "WAIT"
    charger = tuple(get_map_layout(env.config.map_layout_id).charger_position)
    must_charge = tuple(_goals(env, "skilled")[agent_id]) == charger
    if (not must_charge and shortest_path_distance(
            agent.position, other.position, env.config.map_layout_id) <= 2):
        return "WAIT"
    return partner_action(env, agent_id, "skilled", rng)


def _environment(suite: str, scene: Mapping[str, Any]):
    if suite == "original_validation":
        env = PublicFeedbackEnvironment(
            collaborative_study_config(), REWARD, collision_cost=.05, mode="observed")
        reset_scenario(env, scene)
        return env
    if suite == "conflict_validation":
        env = R41ConflictWarehouseEnv(
            reward_config=deepcopy(REWARD), collision_cost=.05, mode="observed")
        reset_conflict_scenario(env, scene)
        return env
    raise ValueError("Unknown corrected-audit suite")


def _episode(actor: NumPyNativeActor, suite: str, scene: Mapping[str, Any],
             profile: str, seed: int) -> dict[str, Any]:
    env = _environment(suite, scene)
    rng = np.random.default_rng(seed)
    counts = Counter()
    trace: list[str] = []
    longest_collision = collision_streak = 0
    longest_no_progress = no_progress = 0
    first_productive_latency = None
    frame = 0
    while not env.done:
        before_snapshot = env.snapshot()
        before_hash = digest(before_snapshot)
        before = env.public_view()
        player_action = corrected_partner_action(env, "robot_1", profile, rng)
        if player_action not in ACTIONS or digest(env.snapshot()) != before_hash:
            raise RuntimeError("Corrected program partner mutated the environment")
        observations = env.observations()
        policy_actions = active_evaluation._actor_actions(actor, observations)
        ai_action = policy_actions["robot_2"]
        if digest(env.snapshot()) != before_hash:
            raise RuntimeError("Actor inference mutated the corrected-audit state")
        submitted = {"robot_1": player_action, "robot_2": ai_action}
        _, _, _, _, info = env.step(submitted)
        after_snapshot = env.snapshot()
        frame += 1
        trace.append(player_action)
        if info.get("requested_actions") != submitted:
            raise RuntimeError("Corrected audit did not submit exact Actor output")
        counts["policy_actions"] += 1
        counts["submitted_actions"] += 1
        counts["action_equal"] += int(info["requested_actions"]["robot_2"] == ai_action)
        active = bool(before["agents"][1]["active"])
        counts["active_frames"] += int(active)
        counts["active_non_wait"] += int(active and ai_action != "WAIT")
        counts[f"action.{ai_action}"] += int(active)
        charger = tuple(get_map_layout(env.config.map_layout_id).charger_position)
        full_noncharger = bool(
            active and float(before["agents"][1]["battery"]) >= 100.
            and tuple(before["agents"][1]["position"]) != charger
        )
        counts["full_noncharger_frames"] += int(full_noncharger)
        counts["full_noncharger_waits"] += int(full_noncharger and ai_action == "WAIT")
        _, flags = shaping_for_transition({
            "before": before_snapshot, "after": after_snapshot,
            "requested_actions": submitted,
            "executed_actions": info["executed_actions"], "events": info["events"],
        }, 1)
        productive = bool(active and flags["productive"])
        counts["productive_actions"] += int(productive)
        counts["charge_needed_frames"] += int(active and flags["charge_needed"])
        counts["charge_needed_waits"] += int(
            active and flags["charge_needed"] and ai_action == "WAIT")
        counts["charge_needed_nonproductive"] += int(
            active and flags["charge_needed"] and not productive)
        if productive and first_productive_latency is None:
            first_productive_latency = frame
        collision = bool(info["robot_collision"])
        collision_streak = collision_streak + 1 if collision else 0
        longest_collision = max(longest_collision, collision_streak)
        counts["collision_steps"] += int(collision)
        counts["collision_cancellations"] += int(
            active and ai_action != "WAIT"
            and info["executed_actions"]["robot_2"] == "WAIT" and collision)
        progress = any(event.get("event") in ("pickup", "delivery")
                       for event in info["events"])
        no_progress = 0 if progress else no_progress + 1
        longest_no_progress = max(longest_no_progress, no_progress)
        feature = (env.feature_names.index(f"self.neighbor.{ai_action}.passable")
                   if ai_action in MOVE_DELTAS else None)
        counts["static_wall_commands"] += int(
            active and feature is not None and observations["robot_2"][feature] < .5)
    state = env.state
    ai_deliveries = int(state.agents[1].deliveries_completed)
    team_deliveries = int(state.total_deliveries)
    return {
        "suite": suite, "scenario_id": scene["id"],
        "scenario_fingerprint": scene["fingerprint"], "partner": profile,
        "seed": int(seed), "steps": frame,
        "active_frames": counts["active_frames"],
        "active_non_wait": counts["active_non_wait"],
        "productive_actions": counts["productive_actions"],
        "action_counts": {action: counts[f"action.{action}"] for action in ACTIONS},
        "charge_needed_frames": counts["charge_needed_frames"],
        "charge_needed_waits": counts["charge_needed_waits"],
        "charge_needed_nonproductive": counts["charge_needed_nonproductive"],
        "full_noncharger_frames": counts["full_noncharger_frames"],
        "full_noncharger_waits": counts["full_noncharger_waits"],
        "first_productive_latency": (
            first_productive_latency if first_productive_latency is not None
            else env.config.horizon + 1),
        "ai_deliveries": ai_deliveries, "team_deliveries": team_deliveries,
        "ai_delivery_share": ai_deliveries / max(1, team_deliveries),
        "collision_steps": counts["collision_steps"],
        "collision_cancellations": counts["collision_cancellations"],
        "longest_collision_streak": longest_collision,
        "longest_no_task_progress_streak": longest_no_progress,
        "static_wall_commands": counts["static_wall_commands"],
        "shutdowns": int(state.shutdown_count),
        "ai_shutdown": int(not state.agents[1].active),
        "player_shutdown": int(not state.agents[0].active),
        "ai_active_end": bool(state.agents[1].active),
        "policy_actions": counts["policy_actions"],
        "submitted_actions": counts["submitted_actions"],
        "action_equal": counts["action_equal"],
        "action_overrides": counts["submitted_actions"] - counts["action_equal"],
        "terminal_reason": state.terminal_reason,
        "participant_action_trace": trace,
    }


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(row))
    result["participant_action_trace_sha256"] = digest(
        result.pop("participant_action_trace"))
    return result


def _distinction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["suite"], row["scenario_id"], row["partner"]): row
        for row in rows
    }
    expected = len({(row["suite"], row["scenario_id"]) for row in rows})
    comparisons = []
    for left, right in combinations(PARTNERS, 2):
        different_episodes = disagreements = aligned = 0
        for suite, scene_id in sorted({(row["suite"], row["scenario_id"])
                                      for row in rows}):
            left_trace = by_key[(suite, scene_id, left)]["participant_action_trace"]
            right_trace = by_key[(suite, scene_id, right)]["participant_action_trace"]
            different_episodes += int(left_trace != right_trace)
            for left_action, right_action in zip_longest(
                    left_trace, right_trace, fillvalue="<ended>"):
                aligned += 1
                disagreements += int(left_action != right_action)
        rate = disagreements / max(1, aligned)
        checks = {
            "different_episodes": different_episodes >= DISTINCTION_GATE[
                "minimum_different_episodes_per_pair"],
            "aligned_action_disagreements": disagreements >= DISTINCTION_GATE[
                "minimum_aligned_action_disagreements_per_pair"],
            "aligned_action_disagreement_rate": rate >= DISTINCTION_GATE[
                "minimum_aligned_action_disagreement_rate_per_pair"],
        }
        comparisons.append({
            "left": left, "right": right, "matched_episodes": expected,
            "different_episodes": different_episodes,
            "aligned_actions": aligned, "aligned_action_disagreements": disagreements,
            "aligned_action_disagreement_rate": rate,
            "checks": checks, "passed": all(checks.values()),
        })
    signatures = {
        profile: digest([
            row["participant_action_trace"]
            for row in rows if row["partner"] == profile
        ]) for profile in PARTNERS
    }
    unique = len(set(signatures.values())) == len(PARTNERS)
    return {
        "passed": unique and all(row["passed"] for row in comparisons),
        "profiles": list(PARTNERS), "profile_trace_sha256": signatures,
        "all_profile_trace_signatures_unique": unique,
        "gate": deepcopy(DISTINCTION_GATE), "pairwise": comparisons,
    }


def _distinction_bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_suite = {
        suite: _distinction([row for row in rows if row["suite"] == suite])
        for suite in ("original_validation", "conflict_validation")
    }
    overall = _distinction(rows)
    return {
        "passed": overall["passed"] and all(
            value["passed"] for value in by_suite.values()),
        "required_scopes": ["overall", "original_validation", "conflict_validation"],
        "overall": overall, "by_suite": by_suite,
    }


def _evaluate_actor(actor_path: str | Path, original_scenes, conflict_scenes,
                    *, seed: int = EVALUATION_SEED) -> dict[str, Any]:
    actor = NumPyNativeActor(actor_path)
    rows = []
    suites = (
        ("original_validation", original_scenes),
        ("conflict_validation", conflict_scenes),
    )
    for suite, scenes in suites:
        for profile in PARTNERS:
            for scene_index, scene in enumerate(scenes):
                rows.append(_episode(actor, suite, scene, profile, seed + scene_index))
    summaries = {
        suite: original_evaluation.summarize([
            _public_row(row) for row in rows if row["suite"] == suite
        ]) for suite, _ in suites
    }
    authority = all(
        row["action_overrides"] == 0
        and row["policy_actions"] == row["submitted_actions"] == row["action_equal"]
        for row in rows
    )
    actor_shutdown_count_by_suite = {
        suite: int(summary["ai_shutdown_count"])
        for suite, summary in summaries.items()
    }
    actor_shutdown_count = sum(actor_shutdown_count_by_suite.values())
    return {
        "actor_sha256": file_hash(actor_path), "suite_summaries": summaries,
        "partner_distinction": _distinction_bundle(rows),
        "action_authority_exact": authority,
        "actor_shutdown_count_by_suite": actor_shutdown_count_by_suite,
        "actor_shutdown_count": actor_shutdown_count,
        "actor_zero_shutdowns": actor_shutdown_count == 0,
        "episode_count": len(rows),
        "episode_evidence_sha256": digest([_public_row(row) for row in rows]),
    }


def _boundary_worker(arguments):
    step, actor_path, original_scenes, conflict_scenes = arguments
    return step, _evaluate_actor(actor_path, original_scenes, conflict_scenes)


def _candidate_eligible(baseline: Mapping[str, Any], candidate: Mapping[str, Any],
                        decisions: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(
        all(row["selected"] for row in decisions.values())
        and baseline["partner_distinction"]["passed"]
        and candidate["partner_distinction"]["passed"]
        and baseline["action_authority_exact"]
        and candidate["action_authority_exact"]
        and candidate["actor_shutdown_count"] == 0
    )


def audit(*, ledger_path: str | Path, expected_ledger_sha256: str,
          dual_evaluation_path: str | Path, conflict_manifest_path: str | Path,
          workers: int = 1) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be between one and eight")
    ledger = training_ledger.read_saved_ledger(
        ledger_path, expected_sha256=expected_ledger_sha256, require_selected=True)
    dual = _strict_json(dual_evaluation_path, "r4.1 frozen dual evaluation")
    manifest = _strict_json(conflict_manifest_path, "r4.1 conflict manifest")
    validate_conflict_manifest(manifest, replay=True)
    original_info = dual.get("manifests", {}).get("original", {})
    original_path = Path(original_info.get("path", "")).expanduser().resolve()
    original = _strict_json(original_path, "r4.1 original validation manifest")
    original_scenes = original.get("splits", {}).get("validation")
    conflict_scenes = manifest.get("splits", {}).get("conflict_validation")
    if (not isinstance(original_scenes, list) or len(original_scenes) != 50
            or not isinstance(conflict_scenes, list) or len(conflict_scenes) != 64
            or file_hash(original_path) != original_info.get("file_sha256")
            or digest(original_scenes) != original_info.get("validation_entries_sha256")
            or file_hash(conflict_manifest_path)
                != dual.get("manifests", {}).get("conflict", {}).get("file_sha256")
            or digest(conflict_scenes)
                != dual.get("manifests", {}).get("conflict", {}).get(
                    "validation_entries_sha256")):
        raise ValueError("Corrected audit suite binding differs")
    baseline_path = Path(
        dual.get("original", {}).get("baseline", {}).get("actor", {}).get("path", "")
    ).expanduser().resolve()
    if (not baseline_path.is_file()
            or file_hash(baseline_path) != ledger["source"]["actor_sha256"]):
        raise ValueError("Corrected audit baseline Actor differs from the ledger parent")
    baseline = _evaluate_actor(baseline_path, original_scenes, conflict_scenes)
    boundary_inputs = []
    for boundary in ledger["boundaries"]:
        actor_path = Path(boundary["actor_path"]).expanduser().resolve()
        if not actor_path.is_file() or file_hash(actor_path) != boundary["actor_sha256"]:
            raise ValueError("Corrected audit boundary Actor bytes differ")
        boundary_inputs.append((boundary["step"], actor_path,
                                original_scenes, conflict_scenes))
    if workers == 1:
        evaluated = [_boundary_worker(value) for value in boundary_inputs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_boundary_worker, boundary_inputs))
    by_step = dict(evaluated)
    boundaries = []
    for ledger_boundary in ledger["boundaries"]:
        candidate = by_step[ledger_boundary["step"]]
        decisions = {
            suite: active_evaluation._suite_decision(
                baseline["suite_summaries"][suite],
                candidate["suite_summaries"][suite],
            ) for suite in ("original_validation", "conflict_validation")
        }
        corrected_eligible = _candidate_eligible(baseline, candidate, decisions)
        boundaries.append({
            "step": ledger_boundary["step"],
            "actor_sha256": ledger_boundary["actor_sha256"],
            "frozen_runner_selected": ledger_boundary["selected"],
            "corrected_eligible": corrected_eligible,
            "actor_shutdown_count": candidate["actor_shutdown_count"],
            "suite_decisions": decisions,
            "evaluation": candidate,
        })
    eligible = [row for row in boundaries if row["corrected_eligible"]]
    earliest = eligible[0] if eligible else None
    frozen = ledger["selected"]
    passed = bool(
        earliest is not None
        and earliest["step"] == frozen["step"]
        and earliest["actor_sha256"] == frozen["actor_sha256"]
        and baseline["partner_distinction"]["passed"]
    )
    return {
        "version": VERSION, "status": "passed" if passed else "failed",
        "admission_eligible": passed, "protocol": deepcopy(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "producer_sources_sha256": digest(producer_sources()),
        "training_ledger_sha256": file_hash(ledger_path),
        "training_ledger_semantic_sha256": digest(ledger),
        "dual_evaluation_sha256": file_hash(dual_evaluation_path),
        "conflict_manifest_sha256": file_hash(conflict_manifest_path),
        "conflict_manifest_content_sha256": manifest["content_sha256"],
        "original_manifest_sha256": file_hash(original_path),
        "original_validation_entries_sha256": digest(original_scenes),
        "conflict_validation_entries_sha256": digest(conflict_scenes),
        "evaluated_boundary_count": len(boundaries),
        "evaluated_boundary_steps": [row["step"] for row in boundaries],
        "all_ledger_boundaries_replayed": len(boundaries) == len(ledger["boundaries"]),
        "baseline": baseline, "boundaries": boundaries,
        "corrected_earliest_passing_step": None if earliest is None else earliest["step"],
        "corrected_earliest_actor_sha256": (
            None if earliest is None else earliest["actor_sha256"]),
        "frozen_selected_step": frozen["step"],
        "frozen_selected_actor_sha256": frozen["actor_sha256"],
        "selected_actor_shutdown_count": (
            None if earliest is None else earliest["actor_shutdown_count"]),
        "selected_actor_matches_corrected_earliest": passed,
        "runtime_action_override": False, "ppo_joint_steps": 0,
    }


def _valid_distinction(value: Any, *, matched_episodes: int) -> bool:
    if (not isinstance(value, dict) or value.get("gate") != DISTINCTION_GATE
            or value.get("profiles") != list(PARTNERS)
            or not isinstance(value.get("profile_trace_sha256"), dict)
            or set(value["profile_trace_sha256"]) != set(PARTNERS)
            or any(not isinstance(item, str) or len(item) != 64
                   for item in value["profile_trace_sha256"].values())):
        return False
    unique = len(set(value["profile_trace_sha256"].values())) == len(PARTNERS)
    comparisons = value.get("pairwise")
    expected_pairs = {tuple(pair) for pair in combinations(PARTNERS, 2)}
    if not isinstance(comparisons, list) or len(comparisons) != len(expected_pairs):
        return False
    seen = set()
    all_passed = True
    for row in comparisons:
        if not isinstance(row, dict):
            return False
        pair = (row.get("left"), row.get("right"))
        if pair not in expected_pairs or pair in seen:
            return False
        seen.add(pair)
        values = (
            row.get("different_episodes"), row.get("aligned_actions"),
            row.get("aligned_action_disagreements"),
        )
        if (any(type(item) is not int or item < 0 for item in values)
                or row.get("matched_episodes") != matched_episodes
                or row["matched_episodes"] < row["different_episodes"]
                or row["aligned_actions"] < row["aligned_action_disagreements"]
                or type(row.get("aligned_action_disagreement_rate")) not in (int, float)):
            return False
        rate = row["aligned_action_disagreements"] / max(1, row["aligned_actions"])
        if abs(float(row["aligned_action_disagreement_rate"]) - rate) > 1e-12:
            return False
        checks = {
            "different_episodes": row["different_episodes"] >= DISTINCTION_GATE[
                "minimum_different_episodes_per_pair"],
            "aligned_action_disagreements": row["aligned_action_disagreements"]
                >= DISTINCTION_GATE[
                    "minimum_aligned_action_disagreements_per_pair"],
            "aligned_action_disagreement_rate": rate >= DISTINCTION_GATE[
                "minimum_aligned_action_disagreement_rate_per_pair"],
        }
        passed = all(checks.values())
        if row.get("checks") != checks or row.get("passed") is not passed:
            return False
        all_passed = all_passed and passed
    passed = unique and all_passed and seen == expected_pairs
    return (value.get("all_profile_trace_signatures_unique") is unique
            and value.get("passed") is passed)


def _valid_distinction_bundle(value: Any) -> bool:
    counts = PROTOCOL["suite_scene_counts"]
    if (not isinstance(value, dict)
            or value.get("required_scopes")
                != ["overall", "original_validation", "conflict_validation"]
            or not isinstance(value.get("by_suite"), dict)
            or set(value["by_suite"]) != set(counts)
            or not _valid_distinction(
                value.get("overall"), matched_episodes=sum(counts.values()))
            or any(not _valid_distinction(value["by_suite"].get(suite),
                                          matched_episodes=count)
                   for suite, count in counts.items())):
        return False
    passed = value["overall"]["passed"] and all(
        child["passed"] for child in value["by_suite"].values())
    return value.get("passed") is passed


def _valid_evaluation(value: Any) -> bool:
    if (not isinstance(value, dict)
            or not isinstance(value.get("actor_sha256"), str)
            or len(value["actor_sha256"]) != 64
            or not isinstance(value.get("suite_summaries"), dict)
            or set(value["suite_summaries"])
                != {"original_validation", "conflict_validation"}
            or type(value.get("episode_count")) is not int
            or value["episode_count"] != PROTOCOL["episodes_per_actor"]
            or not isinstance(value.get("episode_evidence_sha256"), str)
            or len(value["episode_evidence_sha256"]) != 64
            or not _valid_distinction_bundle(value.get("partner_distinction"))):
        return False
    authority = all(
        isinstance(summary, dict)
        and summary.get("action_override_count") == 0
        and summary.get("policy_action_equality_rate") == 1.
        for summary in value["suite_summaries"].values()
    )
    shutdowns = {
        suite: summary.get("ai_shutdown_count")
        for suite, summary in value["suite_summaries"].items()
    }
    if (any(type(count) is not int or count < 0 for count in shutdowns.values())
            or value.get("actor_shutdown_count_by_suite") != shutdowns
            or value.get("actor_shutdown_count") != sum(shutdowns.values())
            or value.get("actor_zero_shutdowns")
                is not (sum(shutdowns.values()) == 0)):
        return False
    return value.get("action_authority_exact") is authority


def read_saved_report(path: str | Path, *, expected_sha256: str,
                      expected_ledger_sha256: str, expected_actor_sha256: str,
                      require_passed: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if file_hash(path) != expected_sha256:
        raise ValueError("Corrected six-partner audit file SHA-256 differs")
    report = _strict_json(path, "corrected six-partner audit")
    if (report.get("version") != VERSION
            or report.get("protocol") != PROTOCOL
            or report.get("protocol_sha256") != digest(PROTOCOL)
            or report.get("producer_sources_sha256") != digest(producer_sources())
            or report.get("training_ledger_sha256") != expected_ledger_sha256
            or report.get("runtime_action_override") is not False
            or report.get("ppo_joint_steps") != 0
            or report.get("all_ledger_boundaries_replayed") is not True
            or report.get("evaluated_boundary_count")
                != len(report.get("boundaries", []))
            or report.get("frozen_selected_actor_sha256") != expected_actor_sha256
            or report.get("selected_actor_matches_corrected_earliest")
                is not report.get("admission_eligible")):
        raise ValueError("Corrected six-partner audit identity differs")
    baseline = report.get("baseline")
    boundaries = report.get("boundaries")
    steps = report.get("evaluated_boundary_steps")
    if (not isinstance(baseline, dict) or not isinstance(boundaries, list)
            or not boundaries or not isinstance(steps, list)
            or steps != [row.get("step") for row in boundaries]
            or steps != sorted(set(steps))
            or any(type(step) is not int or step <= 0 or step % 50_000
                   for step in steps)
            or not _valid_evaluation(baseline)):
        raise ValueError("Corrected six-partner audit evidence is malformed")
    recomputed_eligible = []
    frozen_selected = []
    for row in boundaries:
        evaluation = row.get("evaluation")
        if (not isinstance(row, dict) or not _valid_evaluation(evaluation)
                or evaluation.get("actor_sha256") != row.get("actor_sha256")):
            raise ValueError("Corrected six-partner boundary evidence is malformed")
        decisions = {
            suite: active_evaluation._suite_decision(
                baseline["suite_summaries"][suite],
                evaluation["suite_summaries"][suite],
            ) for suite in ("original_validation", "conflict_validation")
        }
        eligible = _candidate_eligible(baseline, evaluation, decisions)
        if row.get("suite_decisions") != decisions \
                or row.get("corrected_eligible") is not eligible \
                or row.get("actor_shutdown_count") \
                    != evaluation["actor_shutdown_count"]:
            raise ValueError("Corrected six-partner boundary decision differs")
        if eligible:
            recomputed_eligible.append(row)
        if row.get("frozen_runner_selected") is True:
            frozen_selected.append(row)
    earliest = recomputed_eligible[0] if recomputed_eligible else None
    if (len(frozen_selected) != 1
            or frozen_selected[0].get("step") != report.get("frozen_selected_step")
            or frozen_selected[0].get("actor_sha256")
                != report.get("frozen_selected_actor_sha256")
            or report.get("corrected_earliest_passing_step")
                != (None if earliest is None else earliest["step"])
            or report.get("corrected_earliest_actor_sha256")
                != (None if earliest is None else earliest["actor_sha256"])
            or report.get("selected_actor_shutdown_count")
                != (None if earliest is None else earliest["actor_shutdown_count"])):
        raise ValueError("Corrected six-partner selection evidence differs")
    selected_matches = bool(
        earliest is not None
        and earliest["step"] == frozen_selected[0]["step"]
        and earliest["actor_sha256"] == frozen_selected[0]["actor_sha256"]
    )
    if (report.get("selected_actor_matches_corrected_earliest") is not selected_matches
            or report.get("admission_eligible") is not selected_matches
            or report.get("status") != ("passed" if selected_matches else "failed")):
        raise ValueError("Corrected six-partner final decision differs")
    if require_passed and (report.get("status") != "passed"
                           or report.get("admission_eligible") is not True):
        raise ValueError("Corrected six-partner audit did not pass")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ledger", type=Path, required=True)
    parser.add_argument("--expected-training-ledger-sha256", required=True)
    parser.add_argument("--dual-evaluation", type=Path, required=True)
    parser.add_argument("--conflict-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args(argv)
    report = audit(
        ledger_path=args.training_ledger,
        expected_ledger_sha256=args.expected_training_ledger_sha256,
        dual_evaluation_path=args.dual_evaluation,
        conflict_manifest_path=args.conflict_manifest,
        workers=args.workers,
    )
    _write_new(args.output.expanduser().resolve(), report)
    print(canonical({
        "version": VERSION, "status": report["status"],
        "admission_eligible": report["admission_eligible"],
        "evaluated_boundary_count": report["evaluated_boundary_count"],
        "corrected_earliest_passing_step": report["corrected_earliest_passing_step"],
        "output": str(args.output.expanduser().resolve()),
        "output_sha256": file_hash(args.output.expanduser().resolve()),
    }))
    return 0 if report["admission_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "PARTNERS", "DISTINCTION_GATE", "PROTOCOL",
    "producer_sources", "corrected_partner_action", "audit", "read_saved_report",
]
