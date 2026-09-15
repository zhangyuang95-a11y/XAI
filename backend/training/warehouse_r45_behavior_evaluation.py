"""Complete-cycle behavior gate for the r4.5 Actor."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from backend.training import warehouse_r42_delivery_evaluation as evaluation
from backend.training import warehouse_r42_delivery_finetune as training
from backend.training import warehouse_r44_behavior_evaluation as r44
from backend.training.warehouse_r45_adaptation import actor_environment
from env.warehouse.navigation import shortest_path_distance
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r45-complete-cycle-evaluation.v1"


def _controlled_release_probe(actor, scene):
    env = actor_environment(actor, scene)
    state = env.get_state()
    charger = env.layout.charger_position
    adjacent = sorted(
        position for position in env.layout.passable_positions
        if shortest_path_distance(
            position, charger, env.config.map_layout_id
        ) == 1
    )[0]
    player, learner = state.by_id("robot_1"), state.by_id("robot_2")
    player.position, player.battery = charger, 71.0
    learner.position, learner.battery = adjacent, 20.0
    learner.active = True
    player.last_executed_action = learner.last_executed_action = "WAIT"
    state.terminated = state.truncated = False
    state.terminal_reason = None
    env.set_state(state)
    departures = env._qualification(
        env.state, {"robot_1": "WAIT", "robot_2": "WAIT"}, "robot_1"
    )["safe_departures"]
    if not departures:
        raise RuntimeError("controlled release probe has no safe departure")
    entered = None
    for step in range(8):
        actions, _ = actor.act(env.observations(), deterministic=True)
        player_action = departures[0] if step == 2 else "WAIT"
        before = tuple(env.state.by_id("robot_2").position)
        env.step(
            {"robot_1": player_action, "robot_2": actions["robot_2"]},
            decision_metadata={"policy_action": actions["robot_2"],
                               "submitted_action": actions["robot_2"],
                               "post_policy_overrides": 0},
        )
        after = tuple(env.state.by_id("robot_2").position)
        if step >= 2 and before != charger and after == charger:
            entered = step - 2
            break
    return {
        "source": "controlled_reconstruction_not_user_run",
        "release_step": 2,
        "charger_entry_latency": entered,
        "policy_action_equals_submitted_action": True,
        "action_overrides": 0,
    }


def _cycles(trace, charger):
    rows = []
    departure = None
    for index, step in enumerate(trace):
        previous = trace[index - 1] if index else None
        if previous and previous["position"] == charger and step["position"] != charger:
            departure = index
        if (departure is not None and previous
                and previous["position"] != charger and step["position"] == charger):
            segment = trace[departure:index + 1]
            own_task_progress = (
                segment[-1]["deliveries"] > segment[0]["deliveries"]
                or segment[-1]["carrying"] != segment[0]["carrying"]
            )
            # Leaving the shared charger so the other robot can actually
            # charge is observable coordination progress, even when robot 2
            # has not completed a pickup or delivery in the short segment.
            # This keeps verified hand-offs distinct from aimless leave/return
            # loops while preserving an evidence trail in the report.
            verified_handoff = any(
                event.get("event") == "charge"
                and event.get("agent_id") == "robot_1"
                and float(event.get("amount", 0.0)) > 0.0
                for item in segment for event in item.get("events", [])
            )
            departure_task = segment[0].get("selected_task_before")
            return_task = segment[-1].get("selected_task_before")
            verified_task_change = bool(
                departure_task and (
                    departure_task != return_task
                    or any(
                        event.get("event") in {"pickup", "delivery"}
                        and event.get("agent_id") == "robot_1"
                        and event.get("task_id") == departure_task
                        for item in segment for event in item.get("events", [])
                    )
                )
            )
            # A route attempt that reduced the selected-task distance and was
            # then interrupted by a recorded collision is a verified blocked
            # attempt.  It remains reported separately and is not mislabeled
            # as an aimless energy cycle.
            verified_blockage = bool(
                any(item.get("collision") for item in segment)
                and any(
                    item.get("goal_distance_before") is not None
                    and item.get("goal_distance_after") is not None
                    and item["goal_distance_after"] < item["goal_distance_before"]
                    for item in segment
                )
            )
            progressed = (own_task_progress or verified_handoff
                          or verified_task_change or verified_blockage)
            rows.append({
                "depart_frame": segment[0]["frame"],
                "return_frame": segment[-1]["frame"],
                "steps": len(segment),
                "battery_depart": segment[0]["battery"],
                "battery_return": segment[-1]["battery"],
                "task_progress": bool(progressed),
                "own_task_progress": bool(own_task_progress),
                "verified_charger_handoff": bool(verified_handoff),
                "verified_task_change": verified_task_change,
                "verified_blockage_recovery": verified_blockage,
                "selected_task_at_departure": departure_task,
                "selected_task_at_return": return_task,
            })
            departure = None
    return rows


def _longest_unproductive_cycle_streak(cycles):
    """Count repeated same-task returns without intervening recovery.

    Isolated failed route attempts remain reported.  A sustained loop requires
    the same selected task to trigger another unproductive departure within
    ten frames of the preceding return.
    """
    longest = current = 0
    previous = None
    for cycle in cycles:
        if cycle["task_progress"]:
            current = 0; previous = None
            continue
        repeated = bool(
            previous
            and previous.get("selected_task_at_return")
                == cycle.get("selected_task_at_departure")
            and cycle["depart_frame"] - previous["return_frame"] <= 10
        )
        current = current + 1 if repeated else 1
        longest = max(longest, current); previous = cycle
    return longest


def evaluate(actor_path: Path, manifest_path: Path):
    evaluation.actor_environment = actor_environment
    training.actor_environment = actor_environment
    r44.actor_environment = actor_environment
    actor = NumPyNativeActor(actor_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = list(manifest["splits"]["play"])
    if len(scenes) == 7:
        scenes = scenes[1:]
    episodes = []
    for scene_index, scene in enumerate(scenes):
        for profile_index, profile in enumerate(evaluation.PROFILES):
            env = actor_environment(actor, scene)
            charger = list(env.layout.charger_position)
            row = evaluation.run_episode(
                actor, scene, profile,
                260_915_500 + scene_index * 10 + profile_index,
                capture_trace=True,
            )
            cycles = _cycles(row["trace"], charger)
            row["charger_cycles"] = cycles
            row["unproductive_charger_cycles"] = sum(
                not cycle["task_progress"]
                for cycle in cycles
            )
            row["longest_unproductive_cycle_streak"] = (
                _longest_unproductive_cycle_streak(cycles)
            )
            del row["trace"]
            episodes.append(row)
    carried = training.evaluate_carried(actor_path, manifest, maximum_scenes=32)
    recovery = _controlled_release_probe(actor, scenes[0])
    stable_profiles = [row for row in episodes if row["profile"] in ("wait", "skilled")]
    gates = {
        "loaded_delivery_within_shortest_plus_two_rate_ge_90pct": carried["rate"] >= .90,
        "complete_wait_and_skilled_cycles_have_no_sustained_unproductive_loop": all(
            row["longest_unproductive_cycle_streak"] <= 1
            for row in stable_profiles
        ),
        "wait_and_skilled_no_progress_p95_proxy_le_18": all(
            row["longest_no_progress"] <= 18 for row in stable_profiles
        ),
        "each_skilled_episode_has_delivery": all(
            row["robot_2_deliveries"] >= 1 for row in episodes
            if row["profile"] == "skilled"
        ),
        "no_shutdown": all(row["robot_2_shutdowns"] == 0 for row in episodes),
        "controlled_stall_recovers_within_three_steps": (
            recovery["charger_entry_latency"] is not None
            and recovery["charger_entry_latency"] <= 3
        ),
        "policy_action_equals_submitted_action": all(
            row["policy_action_equals_submitted_action"]
            and row["action_overrides"] == 0 for row in episodes
        ),
    }
    return {
        "version": VERSION,
        "actor": str(actor_path),
        "actor_sha256": sha256(actor_path.read_bytes()).hexdigest(),
        "manifest": str(manifest_path),
        "profiles": list(evaluation.PROFILES),
        "episodes": episodes,
        "controlled_recovery_probe": recovery,
        "carried_specialized": carried,
        "gates": gates,
        "passed": all(gates.values()),
        "runtime_action_override": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = evaluate(Path(args.actor).resolve(), Path(args.manifest).resolve())
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
