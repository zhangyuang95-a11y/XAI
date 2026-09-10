"""Paired r3/r4 activity and safety audit on the frozen validation split.

The program partner chooses from the public pre-action state.  The neural
Actor is then evaluated on that same state and its deterministic command is
submitted unchanged.  Environment collision and wall resolution are recorded
separately from policy authority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment
from backend.training.warehouse_native_public_feedback_evaluation import REWARD
from backend.training.warehouse_native_common import digest
from backend.training.warehouse_r4_active_trainer import shaping_for_transition
from env.warehouse.domain import collaborative_study_config
from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.scenarios import reset_scenario


VERSION = "warehouse-r4-active-evaluation.v1"
PARTNERS = ("skilled", "assertive", "noisy", "fixed_yield", "fixed_region", "fixed_task")
ABSOLUTE_GATE = {
    "active_non_wait_rate_min": .85,
    "productive_action_rate_min": .65,
    "full_battery_non_charger_wait_rate_max": .05,
    "first_productive_latency_median_max": 2.,
    "first_productive_latency_p90_max": 5.,
    "ai_delivery_share_min": .35,
    "no_task_progress_streak_p95_max": 18.,
    "mean_longest_collision_streak_max": 3.,
    "static_wall_command_rate_max": .01,
    "shutdown_count_max": 0,
    "action_override_count_max": 0,
}
RELATIVE_GATE = {
    "productive_action_rate_improvement_min": .08,
    "ai_deliveries_relative_improvement_min": .15,
    "ai_deliveries_absolute_improvement_min": 1.,
    "collision_cancellation_rate_increase_max": .02,
    "minimum_checks_passed": 4,
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.


def _actor_actions(actor, observations):
    actions, probabilities = actor.act(observations, deterministic=True)
    if set(actions) != set(observations) or set(probabilities) != set(observations):
        raise RuntimeError("Actor returned an incomplete joint distribution")
    for agent_id in observations:
        distribution = np.asarray(probabilities[agent_id], dtype=np.float64)
        if (distribution.shape != (len(ACTIONS),) or not np.isfinite(distribution).all()
                or abs(float(distribution.sum()) - 1.) > 1e-5
                or actions[agent_id] != ACTIONS[int(np.argmax(distribution))]):
            raise RuntimeError("Actor action differs from its deterministic output")
    return actions


def evaluate_episode(actor, scene, partner, *, seed):
    config = collaborative_study_config()
    env = PublicFeedbackEnvironment(config, REWARD, collision_cost=.05, mode="observed")
    reset_scenario(env, scene)
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
        # The partner cannot inspect the Actor distribution or the user's
        # current command, and is always called first.
        player_action = partner_action(env, "robot_1", partner, rng)
        if player_action not in ACTIONS or digest(env.snapshot()) != before_hash:
            raise RuntimeError("Program partner mutated the environment")
        observations = env.observations()
        policy_actions = _actor_actions(actor, observations)
        ai_action = policy_actions["robot_2"]
        if digest(env.snapshot()) != before_hash:
            raise RuntimeError("Actor inference mutated the environment")
        submitted = {"robot_1": player_action, "robot_2": ai_action}
        _, _, _, _, info = env.step(submitted)
        after = env.public_view()
        after_snapshot = env.snapshot()
        frame += 1
        if info["requested_actions"] != submitted:
            raise RuntimeError("Environment received a command other than the Actor output")
        counts["policy_actions"] += 1
        counts["submitted_actions"] += 1
        counts["action_equal"] += int(info["requested_actions"]["robot_2"] == ai_action)
        active = bool(before["agents"][1]["active"])
        counts["active_frames"] += int(active)
        counts["active_non_wait"] += int(active and ai_action != "WAIT")
        counts[f"action.{ai_action}"] += int(active)
        charger = tuple(get_map_layout(config.map_layout_id).charger_position)
        full_noncharger = bool(
            active and float(before["agents"][1]["battery"]) >= 100.
            and tuple(before["agents"][1]["position"]) != charger
        )
        counts["full_noncharger_frames"] += int(full_noncharger)
        counts["full_noncharger_waits"] += int(full_noncharger and ai_action == "WAIT")
        physical_record = {
            "before": before_snapshot,
            "after": after_snapshot,
            "requested_actions": submitted,
            "executed_actions": info["executed_actions"],
            "events": info["events"],
        }
        _, shaping = shaping_for_transition(physical_record, 1)
        productive = bool(active and shaping["productive"])
        counts["productive_actions"] += int(productive)
        counts["charge_needed_frames"] += int(active and shaping["charge_needed"])
        counts["charge_needed_waits"] += int(
            active and shaping["charge_needed"] and ai_action == "WAIT")
        counts["charge_needed_nonproductive"] += int(
            active and shaping["charge_needed"] and not productive)
        if productive and first_productive_latency is None:
            first_productive_latency = frame
        collision = bool(info["robot_collision"])
        collision_streak = collision_streak + 1 if collision else 0
        longest_collision = max(longest_collision, collision_streak)
        counts["collision_steps"] += int(collision)
        collision_cancellation = bool(
            active and ai_action != "WAIT"
            and info["executed_actions"]["robot_2"] == "WAIT"
            and collision
        )
        counts["collision_cancellations"] += int(collision_cancellation)
        progress = any(event.get("event") in ("pickup", "delivery") for event in info["events"])
        no_progress = 0 if progress else no_progress + 1
        longest_no_progress = max(longest_no_progress, no_progress)
        passable_index = env.feature_names.index(f"self.neighbor.{ai_action}.passable") if ai_action in MOVE_DELTAS else None
        static_wall = bool(active and passable_index is not None and observations["robot_2"][passable_index] < .5)
        counts["static_wall_commands"] += int(static_wall)
    state = env.state
    ai_deliveries = int(state.agents[1].deliveries_completed)
    team_deliveries = int(state.total_deliveries)
    return {
        "scenario_id": scene["id"],
        "scenario_fingerprint": scene["fingerprint"],
        "partner": partner,
        "seed": int(seed),
        "steps": frame,
        "active_frames": counts["active_frames"],
        "active_non_wait": counts["active_non_wait"],
        "productive_actions": counts["productive_actions"],
        "action_counts": {action: counts[f"action.{action}"] for action in ACTIONS},
        "charge_needed_frames": counts["charge_needed_frames"],
        "charge_needed_waits": counts["charge_needed_waits"],
        "charge_needed_nonproductive": counts["charge_needed_nonproductive"],
        "full_noncharger_frames": counts["full_noncharger_frames"],
        "full_noncharger_waits": counts["full_noncharger_waits"],
        "first_productive_latency": first_productive_latency if first_productive_latency is not None else config.horizon + 1,
        "ai_deliveries": ai_deliveries,
        "team_deliveries": team_deliveries,
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
    }


def summarize(rows):
    active = sum(row["active_frames"] for row in rows)
    active_moves = sum(row["active_non_wait"] for row in rows)
    productive = sum(row["productive_actions"] for row in rows)
    full = sum(row["full_noncharger_frames"] for row in rows)
    full_waits = sum(row["full_noncharger_waits"] for row in rows)
    team_deliveries = sum(row["team_deliveries"] for row in rows)
    ai_deliveries = sum(row["ai_deliveries"] for row in rows)
    commands = sum(row["submitted_actions"] for row in rows)
    collision_cancellations = sum(row["collision_cancellations"] for row in rows)
    return {
        "episodes": len(rows),
        "environment_steps": sum(row["steps"] for row in rows),
        "active_frames": active,
        "active_non_wait_rate": active_moves / max(1, active),
        "productive_action_rate": productive / max(1, active),
        "action_distribution": {
            action: sum(row.get("action_counts", {}).get(action, 0) for row in rows) / max(1, active)
            for action in ACTIONS
        },
        "charge_needed_frames": sum(row.get("charge_needed_frames", 0) for row in rows),
        "charge_needed_wait_rate": (
            sum(row.get("charge_needed_waits", 0) for row in rows)
            / max(1, sum(row.get("charge_needed_frames", 0) for row in rows))
        ),
        "charge_needed_nonproductive_rate": (
            sum(row.get("charge_needed_nonproductive", 0) for row in rows)
            / max(1, sum(row.get("charge_needed_frames", 0) for row in rows))
        ),
        "full_battery_non_charger_wait_rate": full_waits / max(1, full),
        "first_productive_latency_median": _percentile([row["first_productive_latency"] for row in rows], 50),
        "first_productive_latency_p90": _percentile([row["first_productive_latency"] for row in rows], 90),
        "mean_ai_deliveries": ai_deliveries / max(1, len(rows)),
        "mean_team_deliveries": team_deliveries / max(1, len(rows)),
        "ai_delivery_share": ai_deliveries / max(1, team_deliveries),
        "collision_cancellation_rate": collision_cancellations / max(1, active_moves),
        "collision_steps_rate": sum(row["collision_steps"] for row in rows) / max(1, commands),
        "mean_longest_collision_streak": float(np.mean([row["longest_collision_streak"] for row in rows])),
        "no_task_progress_streak_p95": _percentile([row["longest_no_task_progress_streak"] for row in rows], 95),
        "static_wall_command_rate": sum(row["static_wall_commands"] for row in rows) / max(1, active),
        "shutdown_count": sum(row["shutdowns"] for row in rows),
        "shutdown_episode_rate": sum(row["shutdowns"] > 0 for row in rows) / max(1, len(rows)),
        "ai_shutdown_count": sum(row["ai_shutdown"] for row in rows),
        "ai_shutdown_episode_rate": sum(row["ai_shutdown"] > 0 for row in rows) / max(1, len(rows)),
        "player_shutdown_count": sum(row["player_shutdown"] for row in rows),
        "action_override_count": sum(row["action_overrides"] for row in rows),
        "policy_action_equality_rate": sum(row["action_equal"] for row in rows) / max(1, commands),
    }


def _absolute_checks(candidate, baseline):
    checks = {
        "active_non_wait": candidate["active_non_wait_rate"] >= ABSOLUTE_GATE["active_non_wait_rate_min"],
        "productive_action": candidate["productive_action_rate"] >= ABSOLUTE_GATE["productive_action_rate_min"],
        "full_battery_non_charger_wait": candidate["full_battery_non_charger_wait_rate"] <= ABSOLUTE_GATE["full_battery_non_charger_wait_rate_max"],
        "first_productive_median": candidate["first_productive_latency_median"] <= ABSOLUTE_GATE["first_productive_latency_median_max"],
        "first_productive_p90": candidate["first_productive_latency_p90"] <= ABSOLUTE_GATE["first_productive_latency_p90_max"],
        "ai_delivery_share": candidate["ai_delivery_share"] >= ABSOLUTE_GATE["ai_delivery_share_min"],
        "no_task_progress_streak_p95": candidate["no_task_progress_streak_p95"] <= ABSOLUTE_GATE["no_task_progress_streak_p95_max"],
        "collision_increase": candidate["collision_cancellation_rate"] <= baseline["collision_cancellation_rate"] + RELATIVE_GATE["collision_cancellation_rate_increase_max"],
        "mean_longest_collision_streak": candidate["mean_longest_collision_streak"] <= ABSOLUTE_GATE["mean_longest_collision_streak_max"],
        "static_wall_command": candidate["static_wall_command_rate"] <= ABSOLUTE_GATE["static_wall_command_rate_max"],
        "shutdowns": candidate["ai_shutdown_count"] <= ABSOLUTE_GATE["shutdown_count_max"],
        "action_authority": candidate["action_override_count"] <= ABSOLUTE_GATE["action_override_count_max"],
    }
    return checks


def _relative_checks(candidate, baseline):
    return {
        "active_non_wait_improved": candidate["active_non_wait_rate"] > baseline["active_non_wait_rate"],
        "productive_action_improved_8pp": candidate["productive_action_rate"] >= baseline["productive_action_rate"] + RELATIVE_GATE["productive_action_rate_improvement_min"],
        "ai_deliveries_improved_15pct": candidate["mean_ai_deliveries"] >= baseline["mean_ai_deliveries"] * (1. + RELATIVE_GATE["ai_deliveries_relative_improvement_min"]),
        "ai_deliveries_improved_one": candidate["mean_ai_deliveries"] >= baseline["mean_ai_deliveries"] + RELATIVE_GATE["ai_deliveries_absolute_improvement_min"],
        "no_progress_p95_improved": candidate["no_task_progress_streak_p95"] < baseline["no_task_progress_streak_p95"],
    }


def evaluate_actor(actor_path, scenarios, *, output_dir: Path, seed=260_910_500):
    actor = NumPyNativeActor(actor_path)
    rows = []
    started = time.monotonic()
    for partner_index, partner in enumerate(PARTNERS):
        for scene_index, scene in enumerate(scenarios):
            episode_seed = seed + partner_index * 10_000 + scene_index
            rows.append(evaluate_episode(actor, scene, partner, seed=episode_seed))
    summary = summarize(rows)
    receipt = {
        "version": VERSION,
        "actor": {
            "path": str(Path(actor_path).resolve()),
            "artifact_sha256": actor.artifact_sha256,
            "parameters_sha256": actor.metadata.get("actor_parameters_sha256", actor.metadata.get("parameters_sha256")),
            "runtime_action_override": actor.metadata.get("runtime_action_override"),
        },
        "partners": list(PARTNERS),
        "validation_entries_sha256": digest(scenarios),
        "seed": seed,
        "summary": summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_jsonl(output_dir / "episodes.jsonl", rows)
    _write_json(output_dir / "report.json", receipt)
    return receipt


def paired_audit(baseline_actor, candidate_actor, scenarios_path, output_root):
    output_root = Path(output_root)
    manifest = json.loads(Path(scenarios_path).read_text())
    scenarios = manifest["splits"]["validation"]
    if len(scenarios) != 50:
        raise ValueError("R4 audit requires all 50 frozen validation scenes")
    baseline = evaluate_actor(baseline_actor, scenarios, output_dir=output_root / "r3")
    candidate = evaluate_actor(candidate_actor, scenarios, output_dir=output_root / "candidate")
    absolute = _absolute_checks(candidate["summary"], baseline["summary"])
    relative = _relative_checks(candidate["summary"], baseline["summary"])
    relative_passed = sum(relative.values())
    report = {
        "version": VERSION,
        "status": "passed" if all(absolute.values()) and relative_passed >= RELATIVE_GATE["minimum_checks_passed"] else "failed",
        "selected": bool(all(absolute.values()) and relative_passed >= RELATIVE_GATE["minimum_checks_passed"]),
        "absolute_gate": deepcopy(ABSOLUTE_GATE),
        "relative_gate": deepcopy(RELATIVE_GATE),
        "absolute_checks": absolute,
        "relative_checks": relative,
        "relative_checks_passed": relative_passed,
        "baseline": baseline,
        "candidate": candidate,
        "scenario_manifest": {
            "path": str(Path(scenarios_path).resolve()),
            "sha256": sha256(Path(scenarios_path).read_bytes()).hexdigest(),
            "validation_entries_sha256": digest(scenarios),
        },
        "definitions": {
            "productive_action": "A safe-energy pickup/delivery, or a move that decreases the public task/charge-goal distance used by r4 shaping.",
            "collision_cancellation_rate": "AI non-WAIT commands canceled to WAIT on a robot-collision step divided by active AI non-WAIT commands.",
            "no_task_progress_streak": "Consecutive environment steps with no pickup or delivery by either robot.",
            "first_productive_latency": "One-based frame of the AI's first productive action; horizon+1 if absent.",
        },
    }
    _write_json(output_root / "paired_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-actor", required=True)
    parser.add_argument("--candidate-actor", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = paired_audit(args.baseline_actor, args.candidate_actor, args.scenarios, args.output)
    print(json.dumps({
        "status": report["status"],
        "selected": report["selected"],
        "absolute_checks": report["absolute_checks"],
        "relative_checks": report["relative_checks"],
        "baseline": report["baseline"]["summary"],
        "candidate": report["candidate"]["summary"],
        "report": str((Path(args.output) / "paired_report.json").resolve()),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
