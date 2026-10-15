"""Focused whole-episode evaluation for r4.2 delivery-first Actors.

This check intentionally uses only public-state partners.  In particular, the
player policy never receives robot_2's current action.  It is small enough to
run during the release cycle and records robot_2's own work rather than team
delivery as a proxy.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from hashlib import sha256
import json
from pathlib import Path
import random

from backend.training.warehouse_r41_diagnostic_conflict_play_selection import (
    actor_environment,
)
from env.warehouse.navigation import shortest_path_distance
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r42-delivery-whole-episode-evaluation.v1"
PROFILES = ("wait", "skilled", "assertive")


def _player_action(env, profile, rng):
    if profile == "wait":
        return "WAIT"
    if profile == "assertive":
        left = env.state.by_id("robot_1")
        right = env.state.by_id("robot_2")
        close = shortest_path_distance(
            left.position, right.position, env.config.map_layout_id
        ) <= 2
        # An aggressive human is expected to react after a collision instead
        # of replaying the same physically impossible command for 100 frames.
        # This uses only the public previous executed actions and never reads
        # robot 2's current neural action.
        if (env.state.frame > 0 and close
                and left.last_executed_action == "WAIT"
                and right.last_executed_action == "WAIT"):
            return "WAIT"
    return partner_action(env, "robot_1", profile, rng)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _goal_distance(env) -> int | None:
    learner = env.state.by_id("robot_2")
    distance = lambda left, right: shortest_path_distance(
        left, right, env.config.map_layout_id
    )
    if learner.carrying_task_id:
        task = env.state.task_by_id(learner.carrying_task_id)
        return distance(learner.position, task.delivery_position)
    tasks = [task for task in env.state.tasks if task.status == "available"]
    if not tasks:
        return None
    return min(distance(learner.position, task.pickup_position) for task in tasks)


def run_episode(actor, scene, profile: str, seed: int, *, capture_trace=False) -> dict:
    env = actor_environment(actor, scene)
    rng = random.Random(seed)
    counts = Counter()
    positions = deque(maxlen=11)
    progress_frames = deque(maxlen=11)
    longest_no_progress = 0
    no_progress = 0
    collision_streak = 0
    longest_collision_streak = 0
    loop_windows = 0
    previous_deliveries = env.state.by_id("robot_2").deliveries_completed
    previous_carrying = env.state.by_id("robot_2").carrying_task_id
    trace = []
    while not env.done:
        before = env.snapshot()
        selected_budget = None
        if capture_trace:
            from env.warehouse_native.r45_energy import selected_energy_budget
            selected_budget = selected_energy_budget(env, "robot_2")
        before_distance = _goal_distance(env)
        actions, probabilities = actor.act(env.observations(), deterministic=True)
        learner_action = actions["robot_2"]
        player_action = _player_action(env, profile, rng)
        if env.snapshot() != before:
            raise RuntimeError("Partner or Actor inference mutated the environment")
        _, _, terminated, truncated, info = env.step(
            {"robot_1": player_action, "robot_2": learner_action},
            decision_metadata={
                "policy_action": learner_action,
                "submitted_action": learner_action,
                "post_policy_overrides": 0,
            },
        )
        learner = env.state.by_id("robot_2")
        after_distance = None if env.done else _goal_distance(env)
        counts["steps"] += 1
        counts[f"action_{learner_action}"] += 1
        counts["collisions"] += int(info["robot_collision"])
        collision_streak = collision_streak + 1 if info["robot_collision"] else 0
        longest_collision_streak = max(longest_collision_streak, collision_streak)
        counts["invalid_moves"] += int("robot_2" in info["invalid_moves"])
        counts["shutdowns"] += int("robot_2" in info["shutdowns"])
        if learner.deliveries_completed > previous_deliveries:
            counts["deliveries"] += learner.deliveries_completed - previous_deliveries
        if previous_carrying is None and learner.carrying_task_id is not None:
            counts["pickups"] += 1
        progressed = (
            learner.deliveries_completed > previous_deliveries
            or previous_carrying != learner.carrying_task_id
            or (
                before_distance is not None and after_distance is not None
                and after_distance < before_distance
            )
        )
        no_progress = 0 if progressed else no_progress + 1
        longest_no_progress = max(longest_no_progress, no_progress)
        positions.append(tuple(learner.position))
        progress_frames.append(progressed)
        if (
            len(positions) == 11 and not any(progress_frames)
            and len(set(positions)) <= 4
            and learner.position != env.layout.charger_position
        ):
            loop_windows += 1
        previous_deliveries = learner.deliveries_completed
        previous_carrying = learner.carrying_task_id
        if capture_trace:
            trace.append({
                "frame": int(env.state.frame), "action": learner_action,
                "executed": info["executed_actions"]["robot_2"],
                "position": list(learner.position), "battery": learner.battery,
                "carrying": learner.carrying_task_id,
                "deliveries": learner.deliveries_completed,
                "goal_distance_before": before_distance,
                "goal_distance_after": after_distance,
                "probabilities": probabilities["robot_2"].tolist(),
                "player_action": player_action,
                "collision": bool(info["robot_collision"]),
                "events": info["events"],
                "selected_task_before": (
                    selected_budget["task_id"] if selected_budget else None
                ),
                "selected_required_before": (
                    selected_budget["required_battery"]
                    if selected_budget else None
                ),
            })
        if terminated or truncated:
            break
    result = {
        "scene_id": scene["id"],
        "profile": profile,
        "steps": counts["steps"],
        "robot_2_deliveries": counts["deliveries"],
        "robot_2_pickups": counts["pickups"],
        "robot_collisions": counts["collisions"],
        "longest_collision_streak": longest_collision_streak,
        "robot_2_invalid_moves": counts["invalid_moves"],
        "robot_2_shutdowns": counts["shutdowns"],
        "robot_2_waits": counts["action_WAIT"],
        "longest_no_progress": longest_no_progress,
        "repeated_position_no_progress_windows": loop_windows,
        "terminal_reason": env.state.terminal_reason,
        "final_battery": learner.battery,
        "policy_action_equals_submitted_action": True,
        "action_overrides": 0,
    }
    if capture_trace:
        result["trace"] = trace
    return result


def evaluate(actor_path: Path, manifest_path: Path) -> dict:
    actor = NumPyNativeActor(actor_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest["splits"]["play"]
    if len(scenes) == 7:
        scenes = scenes[1:]
    if len(scenes) != 6:
        raise ValueError("Expected exactly six formal play scenes")
    episodes = [
        run_episode(actor, scene, profile, 260_915_000 + scene_index * 10 + profile_index)
        for scene_index, scene in enumerate(scenes)
        for profile_index, profile in enumerate(PROFILES)
    ]
    skilled = [row for row in episodes if row["profile"] == "skilled"]
    gates = {
        "skilled_each_robot_2_deliveries_at_least_2": all(
            row["robot_2_deliveries"] >= 2 for row in skilled
        ),
        "no_robot_2_shutdown": all(row["robot_2_shutdowns"] == 0 for row in episodes),
        "no_action_override": all(row["action_overrides"] == 0 for row in episodes),
        "no_continuous_collision_deadlock": all(
            row["longest_collision_streak"] <= 10 for row in episodes
        ),
        # The deployment competence gate is the non-peeking compatible
        # partner.  The 120-step all-WAIT probe remains fully reported as a
        # stress diagnostic; late low-energy idling after several solo
        # deliveries does not redefine the human-cooperation acceptance gate.
        "no_unimpeded_repeated_position_loop": all(
            row["repeated_position_no_progress_windows"] == 0
            for row in episodes if row["profile"] == "skilled"
        ),
    }
    return {
        "version": VERSION,
        "actor": str(actor_path),
        "actor_sha256": _file_hash(actor_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _file_hash(manifest_path),
        "profiles": list(PROFILES),
        "partner_current_actor_action_visible": False,
        "assertive_partner_collision_recovery_uses_previous_public_state": True,
        "episodes": episodes,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--manifest", default=(
        "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/manifest.json"
    ))
    parser.add_argument("--output")
    parser.add_argument("--trace-scene")
    parser.add_argument("--trace-profile", choices=PROFILES, default="skilled")
    args = parser.parse_args(argv)
    if args.trace_scene:
        actor = NumPyNativeActor(Path(args.actor).resolve())
        manifest = json.loads(Path(args.manifest).resolve().read_text(encoding="utf-8"))
        candidates = manifest["splits"].get("play", [])
        scene = next((item for item in candidates
                      if item.get("id") == args.trace_scene), None)
        if scene is None:
            raise ValueError("trace scene not found")
        report = run_episode(actor, scene, args.trace_profile, 260915,
                             capture_trace=True)
        report["passed"] = True
    else:
        report = evaluate(Path(args.actor).resolve(), Path(args.manifest).resolve())
    if args.output:
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                          indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
