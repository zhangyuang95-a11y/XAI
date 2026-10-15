"""Dual-suite activity, safety, and action-authority audit for warehouse r4.1.

The original frozen validation suite intentionally keeps its historical task
sampler.  The conflict suite is stepped only through the shared r4.1 conflict
environment, which is also used by the online runtime.  An Actor is selectable
only when it passes the registered r4 gates on both suites and every neural
command reaches physics unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from backend.training.warehouse_native_common import digest, file_hash
from backend.training import warehouse_r4_active_evaluation as r4_evaluation
from backend.training.warehouse_r4_active_trainer import shaping_for_transition
from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-active-dual-evaluation.v1"
PARTNERS = r4_evaluation.PARTNERS
ABSOLUTE_GATE = deepcopy(r4_evaluation.ABSOLUTE_GATE)
RELATIVE_GATE = deepcopy(r4_evaluation.RELATIVE_GATE)
MINIMUM_CONFLICT_VALIDATION_SCENES = 50


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _conflict_api():
    # Kept local so source-only audits can inspect the trainer before a conflict
    # package is materialized.  Any real evaluation fails closed if it is absent.
    from types import SimpleNamespace
    from env.warehouse_native import r41_conflict as contract
    from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
    from backend.training import warehouse_r41_conflict_scenarios as scenarios

    if (not hasattr(contract, "CONTRACT_SHA256")
            or not hasattr(scenarios, "validate_conflict_manifest")
            or not hasattr(scenarios, "reset_conflict_scenario")):
        raise RuntimeError("The shared r4.1 conflict runtime contract is incomplete")
    return SimpleNamespace(
        R41ConflictWarehouseEnv=R41ConflictWarehouseEnv,
        reset_conflict_scenario=scenarios.reset_conflict_scenario,
        validate_conflict_manifest=scenarios.validate_conflict_manifest,
        CONTRACT_SHA256=contract.CONTRACT_SHA256,
    )


def _new_environment():
    api = _conflict_api()
    return api.R41ConflictWarehouseEnv(
        reward_config=deepcopy(r4_evaluation.REWARD), collision_cost=.05,
        mode="observed",
    )


def _actor_actions(actor, observations):
    actions, probabilities = actor.act(observations, deterministic=True)
    if set(actions) != set(observations) or set(probabilities) != set(observations):
        raise RuntimeError("Actor returned an incomplete joint distribution")
    for agent_id in observations:
        distribution = np.asarray(probabilities[agent_id], dtype=np.float64)
        if (distribution.shape != (len(ACTIONS),)
                or not np.isfinite(distribution).all()
                or abs(float(distribution.sum()) - 1.) > 1e-5
                or actions[agent_id] != ACTIONS[int(np.argmax(distribution))]):
            raise RuntimeError("Actor action differs from its deterministic output")
    return actions


def evaluate_conflict_episode(actor, scene, partner, *, seed):
    """Evaluate one genuine r4.1 successor-table episode."""
    api = _conflict_api()
    env = _new_environment()
    api.reset_conflict_scenario(env, scene)
    if getattr(env, "conflict_contract_sha256", api.CONTRACT_SHA256) != api.CONTRACT_SHA256:
        raise RuntimeError("Evaluation environment is not bound to the shared conflict contract")
    rng = np.random.default_rng(seed)
    counts = Counter()
    longest_collision = collision_streak = 0
    longest_no_progress = no_progress = 0
    first_productive_latency = None
    frame = 0
    while not env.done:
        before_snapshot = env.snapshot()
        before = env.public_view()
        before_hash = digest(before_snapshot)
        # The program partner is called before Actor inference and cannot read
        # the participant's current action from any side channel.
        player_action = partner_action(env, "robot_1", partner, rng)
        if player_action not in ACTIONS or digest(env.snapshot()) != before_hash:
            raise RuntimeError("Program partner changed the evaluation state")
        observations = env.observations()
        policy_actions = _actor_actions(actor, observations)
        ai_action = policy_actions["robot_2"]
        if digest(env.snapshot()) != before_hash:
            raise RuntimeError("Actor inference changed the evaluation state")
        submitted = {"robot_1": player_action, "robot_2": ai_action}
        _, _, _, _, info = env.step(submitted)
        after_snapshot = env.snapshot()
        after = env.public_view()
        frame += 1
        if info.get("requested_actions") != submitted:
            raise RuntimeError("Environment received a command other than Actor output")
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
            "executed_actions": info["executed_actions"],
            "events": info["events"],
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
    return {
        "scenario_id": scene["id"],
        "scenario_fingerprint": scene["fingerprint"],
        "successor_table_sha256": scene.get("successor_table_sha256"),
        "conflict_contract_sha256": api.CONTRACT_SHA256,
        "partner": partner, "seed": int(seed), "steps": frame,
        "active_frames": counts["active_frames"],
        "active_non_wait": counts["active_non_wait"],
        "productive_actions": counts["productive_actions"],
        "action_counts": {action: counts[f"action.{action}"] for action in ACTIONS},
        "charge_needed_frames": counts["charge_needed_frames"],
        "charge_needed_waits": counts["charge_needed_waits"],
        "charge_needed_nonproductive": counts["charge_needed_nonproductive"],
        "full_noncharger_frames": counts["full_noncharger_frames"],
        "full_noncharger_waits": counts["full_noncharger_waits"],
        "first_productive_latency": (first_productive_latency
                                      if first_productive_latency is not None
                                      else env.config.horizon + 1),
        "ai_deliveries": int(state.agents[1].deliveries_completed),
        "team_deliveries": int(state.total_deliveries),
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
    }


def evaluate_conflict_actor(actor_path, scenes, *, output_dir: Path,
                            seed=260_911_500):
    actor = NumPyNativeActor(actor_path)
    rows = []
    started = time.monotonic()
    for partner_index, partner in enumerate(PARTNERS):
        for scene_index, scene in enumerate(scenes):
            rows.append(evaluate_conflict_episode(
                actor, scene, partner,
                seed=seed + partner_index * 100_000 + scene_index,
            ))
    summary = r4_evaluation.summarize(rows)
    receipt = {
        "version": VERSION,
        "suite": "r41_conflict_validation",
        "actor": {
            "path": str(Path(actor_path).resolve()),
            "artifact_sha256": actor.artifact_sha256,
            "parameters_sha256": actor.metadata.get(
                "actor_parameters_sha256", actor.metadata.get("parameters_sha256")),
            "runtime_action_override": actor.metadata.get("runtime_action_override"),
        },
        "partners": list(PARTNERS),
        "validation_entries_sha256": digest(scenes),
        "conflict_contract_sha256": _conflict_api().CONTRACT_SHA256,
        "seed": seed, "summary": summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_jsonl(Path(output_dir) / "episodes.jsonl", rows)
    _write_json(Path(output_dir) / "report.json", receipt)
    return receipt


def _suite_decision(baseline, candidate):
    absolute = r4_evaluation._absolute_checks(candidate, baseline)
    relative = r4_evaluation._relative_checks(candidate, baseline)
    relative_passed = sum(relative.values())
    selected = all(absolute.values()) and relative_passed >= RELATIVE_GATE["minimum_checks_passed"]
    return {
        "selected": bool(selected), "absolute_checks": absolute,
        "relative_checks": relative, "relative_checks_passed": relative_passed,
    }


def prepare_dual_baseline(baseline_actor, original_scenarios_path,
                          conflict_manifest_path, output_root):
    """Evaluate frozen r3 once; every 50k candidate reuses this bound receipt."""
    output_root = Path(output_root)
    original_path = Path(original_scenarios_path).resolve()
    conflict_path = Path(conflict_manifest_path).resolve()
    original = json.loads(original_path.read_text())
    conflict = json.loads(conflict_path.read_text())
    api = _conflict_api()
    api.validate_conflict_manifest(conflict)
    old_scenes = original.get("splits", {}).get("validation", [])
    conflict_scenes = conflict.get("splits", {}).get("conflict_validation", [])
    if len(old_scenes) != 50:
        raise ValueError("The original r4.1 audit requires all 50 frozen scenes")
    if len(conflict_scenes) < MINIMUM_CONFLICT_VALIDATION_SCENES:
        raise ValueError("The conflict audit requires at least 50 held-out scenes")
    old = r4_evaluation.evaluate_actor(
        baseline_actor, old_scenes, output_dir=output_root / "original")
    conflict_report = evaluate_conflict_actor(
        baseline_actor, conflict_scenes, output_dir=output_root / "conflict")
    report = {
        "version": VERSION, "kind": "frozen_r3_dual_baseline",
        "baseline_actor_sha256": file_hash(baseline_actor),
        "action_authority_exact": all(
            item["summary"]["action_override_count"] == 0
            and item["summary"]["policy_action_equality_rate"] == 1.
            for item in (old, conflict_report)),
        "original": old, "conflict": conflict_report,
        "manifests": {
            "original": {"path": str(original_path), "file_sha256": file_hash(original_path),
                         "validation_entries_sha256": digest(old_scenes)},
            "conflict": {"path": str(conflict_path), "file_sha256": file_hash(conflict_path),
                         "semantic_sha256": digest(conflict),
                         "validation_entries_sha256": digest(conflict_scenes),
                         "contract_sha256": api.CONTRACT_SHA256},
        },
    }
    if not report["action_authority_exact"]:
        raise RuntimeError("Frozen r3 baseline did not preserve Actor action authority")
    _write_json(output_root / "baseline_report.json", report)
    return report


def read_dual_baseline(path, *, baseline_actor, original_scenarios_path,
                       conflict_manifest_path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    original_path = Path(original_scenarios_path).resolve()
    conflict_path = Path(conflict_manifest_path).resolve()
    conflict = json.loads(conflict_path.read_text())
    api = _conflict_api()
    api.validate_conflict_manifest(conflict)
    expected = {
        "baseline_actor_sha256": file_hash(baseline_actor),
        "original_file_sha256": file_hash(original_path),
        "conflict_file_sha256": file_hash(conflict_path),
        "conflict_semantic_sha256": digest(conflict),
        "contract_sha256": api.CONTRACT_SHA256,
    }
    if (report.get("version") != VERSION
            or report.get("kind") != "frozen_r3_dual_baseline"
            or report.get("baseline_actor_sha256") != expected["baseline_actor_sha256"]
            or report.get("action_authority_exact") is not True
            or report.get("manifests", {}).get("original", {}).get("file_sha256")
                != expected["original_file_sha256"]
            or report.get("manifests", {}).get("conflict", {}).get("file_sha256")
                != expected["conflict_file_sha256"]
            or report.get("manifests", {}).get("conflict", {}).get("semantic_sha256")
                != expected["conflict_semantic_sha256"]
            or report.get("manifests", {}).get("conflict", {}).get("contract_sha256")
                != expected["contract_sha256"]):
        raise ValueError("Frozen r3 dual-suite baseline binding differs")
    for suite in ("original", "conflict"):
        summary = report.get(suite, {}).get("summary", {})
        if (summary.get("action_override_count") != 0
                or summary.get("policy_action_equality_rate") != 1.):
            raise ValueError("Frozen baseline contains overwritten Actor actions")
    return report


def paired_dual_audit(baseline_actor, candidate_actor, original_scenarios_path,
                      conflict_manifest_path, output_root, *, baseline_report_path=None):
    """Require the earliest r4.1 candidate to pass both frozen suites."""
    output_root = Path(output_root)
    original_path = Path(original_scenarios_path).resolve()
    conflict_path = Path(conflict_manifest_path).resolve()
    original = json.loads(original_path.read_text())
    conflict = json.loads(conflict_path.read_text())
    api = _conflict_api()
    api.validate_conflict_manifest(conflict)
    old_scenes = original.get("splits", {}).get("validation", [])
    conflict_scenes = conflict.get("splits", {}).get("conflict_validation", [])
    if len(old_scenes) != 50:
        raise ValueError("The original r4.1 audit requires all 50 frozen scenes")
    if len(conflict_scenes) < MINIMUM_CONFLICT_VALIDATION_SCENES:
        raise ValueError("The conflict audit requires at least 50 held-out scenes")
    if baseline_report_path is None:
        baseline = prepare_dual_baseline(
            baseline_actor, original_path, conflict_path, output_root / "r3_baseline")
    else:
        baseline = read_dual_baseline(
            baseline_report_path, baseline_actor=baseline_actor,
            original_scenarios_path=original_path,
            conflict_manifest_path=conflict_path,
        )
    old_base = baseline["original"]
    old_candidate = r4_evaluation.evaluate_actor(
        candidate_actor, old_scenes, output_dir=output_root / "original" / "candidate")
    conflict_base = baseline["conflict"]
    conflict_candidate = evaluate_conflict_actor(
        candidate_actor, conflict_scenes,
        output_dir=output_root / "conflict" / "candidate")
    decisions = {
        "original_validation": _suite_decision(
            old_base["summary"], old_candidate["summary"]),
        "conflict_validation": _suite_decision(
            conflict_base["summary"], conflict_candidate["summary"]),
    }
    action_authority_exact = all(
        report["summary"]["action_override_count"] == 0
        and report["summary"]["policy_action_equality_rate"] == 1.
        for report in (old_base, old_candidate, conflict_base, conflict_candidate)
    )
    selected = all(item["selected"] for item in decisions.values()) and action_authority_exact
    report = {
        "version": VERSION,
        "status": "passed" if selected else "failed",
        "selected": bool(selected),
        "candidate_actor_sha256": file_hash(candidate_actor),
        "baseline_actor_sha256": file_hash(baseline_actor),
        "absolute_gate": deepcopy(ABSOLUTE_GATE),
        "relative_gate": deepcopy(RELATIVE_GATE),
        "suite_decisions": decisions,
        "action_authority_exact": action_authority_exact,
        "original": {"baseline": old_base, "candidate": old_candidate},
        "conflict": {"baseline": conflict_base, "candidate": conflict_candidate},
        "manifests": {
            "original": {"path": str(original_path), "file_sha256": file_hash(original_path),
                         "validation_entries_sha256": digest(old_scenes)},
            "conflict": {"path": str(conflict_path), "file_sha256": file_hash(conflict_path),
                         "semantic_sha256": digest(conflict),
                         "validation_entries_sha256": digest(conflict_scenes),
                         "contract_sha256": api.CONTRACT_SHA256},
        },
        "selection_rule": "both suites pass every absolute gate and at least four of five relative gates",
    }
    _write_json(output_root / "paired_dual_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-actor", required=True)
    parser.add_argument("--candidate-actor", required=True)
    parser.add_argument("--original-scenarios", required=True)
    parser.add_argument("--conflict-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = paired_dual_audit(
        args.baseline_actor, args.candidate_actor, args.original_scenarios,
        args.conflict_manifest, args.output,
    )
    print(json.dumps({"status": report["status"], "selected": report["selected"],
                      "report": str((Path(args.output) / "paired_dual_report.json").resolve())},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
