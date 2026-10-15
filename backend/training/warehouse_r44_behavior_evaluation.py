"""Focused r4.4 behavior and neural-action-authority report."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from backend.training import warehouse_r42_delivery_evaluation as evaluation
from backend.training import warehouse_r42_delivery_finetune as training
from backend.training.warehouse_r44_adaptation import actor_environment
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r44-focused-behavior-evaluation.v1"


def _controlled_recovery_probe(actor, scene):
    env = actor_environment(actor, scene)
    state = env.get_state()
    state.frame = 50
    player = state.by_id("robot_1")
    learner = state.by_id("robot_2")
    player.position = env.layout.charger_position
    player.battery = 71.0
    learner.position = (3, 0)
    learner.battery = 34.0
    learner.carrying_task_id = None
    learner.active = True
    for task, endpoints in zip(
        state.tasks, (((2, 5), (4, 4)), ((1, 1), (4, 2)))
    ):
        task.pickup_position, task.delivery_position = endpoints
        task.status = "available"
        task.carrier_agent_id = None
    state.completed_tasks = []
    state.total_deliveries = 0
    state.terminated = state.truncated = False
    state.terminal_reason = None
    env.set_state(state)
    release_step = 5
    entered_after_release = None
    positions, deliveries = [], []
    for step in range(50):
        actions, _ = actor.act(env.observations(), deterministic=True)
        before_position = tuple(env.state.by_id("robot_2").position)
        player_action = "UP" if step == release_step else "WAIT"
        env.step({"robot_1": player_action,
                  "robot_2": actions["robot_2"]},
                 decision_metadata={"policy_action": actions["robot_2"],
                                    "submitted_action": actions["robot_2"],
                                    "post_policy_overrides": 0})
        after = env.state.by_id("robot_2")
        positions.append(tuple(after.position))
        deliveries.append(int(after.deliveries_completed))
        if (step >= release_step and entered_after_release is None
                and before_position != env.layout.charger_position
                and after.position == env.layout.charger_position):
            entered_after_release = step - release_step
        if env.done:
            break
    immediate_returns = sum(
        index >= 2
        and positions[index] == env.layout.charger_position
        and positions[index - 1] != env.layout.charger_position
        and positions[index - 2] == env.layout.charger_position
        and deliveries[index] == deliveries[index - 2]
        for index in range(len(positions))
    )
    return {
        "source": "controlled_reconstruction_not_user_run",
        "release_step": release_step,
        "charger_entry_latency": entered_after_release,
        "robot_2_deliveries": deliveries[-1] if deliveries else 0,
        "immediate_unproductive_charger_returns": immediate_returns,
        "policy_action_equals_submitted_action": True,
        "action_overrides": 0,
    }


def evaluate(actor_path: Path, manifest_path: Path) -> dict:
    evaluation.actor_environment = actor_environment
    training.actor_environment = actor_environment
    actor = NumPyNativeActor(actor_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = list(manifest["splits"]["play"])
    if len(scenes) == 7:
        scenes = scenes[1:]
    episodes = []
    for scene_index, scene in enumerate(scenes):
        for profile_index, profile in enumerate(evaluation.PROFILES):
            charger = list(actor_environment(actor, scene).layout.charger_position)
            row = evaluation.run_episode(
                actor, scene, profile,
                260_915_440 + scene_index * 10 + profile_index,
                capture_trace=True,
            )
            row["charger_occupancy_penalties"] = sum(
                event.get("event") == "charger_occupancy_penalty"
                for step in row["trace"] for event in step["events"]
            )
            row["unproductive_charger_returns"] = sum(
                index >= 2
                and step["position"] == charger
                and row["trace"][index - 2]["position"]
                    == charger
                and row["trace"][index - 1]["position"]
                    != charger
                and step["deliveries"]
                    == row["trace"][index - 2]["deliveries"]
                for index, step in enumerate(row["trace"])
            )
            del row["trace"]
            episodes.append(row)
    skilled = [row for row in episodes if row["profile"] == "skilled"]
    carried = training.evaluate_carried(actor_path, manifest, maximum_scenes=32)
    recovery = _controlled_recovery_probe(actor, scenes[0])
    gates = {
        "loaded_delivery_within_shortest_plus_two_rate_ge_90pct": (
            carried["rate"] >= .90
        ),
        "skilled_partner_robot_2_delivers": all(
            row["robot_2_deliveries"] >= 1 for row in skilled
        ),
        "no_robot_2_shutdown": all(
            row["robot_2_shutdowns"] == 0 for row in episodes
        ),
        "controlled_stall_recovers_within_three_steps": (
            recovery["charger_entry_latency"] is not None
            and recovery["charger_entry_latency"] <= 3
        ),
        "controlled_cycle_completes_two_deliveries": (
            recovery["robot_2_deliveries"] >= 2
        ),
        "controlled_cycle_has_no_immediate_charger_return": (
            recovery["immediate_unproductive_charger_returns"] == 0
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
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "profiles": list(evaluation.PROFILES),
        "partner_current_actor_action_visible": False,
        "episodes": episodes,
        "controlled_recovery_probe": recovery,
        "carried_specialized": carried,
        "gates": gates,
        "passed": all(gates.values()),
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
