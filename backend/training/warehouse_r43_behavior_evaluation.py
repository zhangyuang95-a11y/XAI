"""Focused 18-run behavior report for the r4.3 internal pilot."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from backend.training import warehouse_r42_delivery_evaluation as base
from backend.training import warehouse_r42_delivery_finetune as training
from backend.training.warehouse_r43_adaptation import actor_environment
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r43-focused-behavior-evaluation.v1"


def evaluate(actor_path: Path, manifest_path: Path):
    base.actor_environment = actor_environment
    training.actor_environment = actor_environment
    actor = NumPyNativeActor(actor_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest["splits"]["play"]
    if len(scenes) == 7:
        scenes = scenes[1:]
    episodes = []
    for scene_index, scene in enumerate(scenes):
        for profile_index, profile in enumerate(base.PROFILES):
            row = base.run_episode(
                actor, scene, profile,
                260_915_300 + scene_index * 10 + profile_index,
                capture_trace=True,
            )
            row["charger_occupancy_penalties"] = sum(
                event.get("event") == "charger_occupancy_penalty"
                for step in row["trace"] for event in step["events"]
            )
            del row["trace"]
            episodes.append(row)
    compatible = [row for row in episodes if row["profile"] == "skilled"]
    carried = training.evaluate_carried(actor_path, manifest, maximum_scenes=32)
    gates = {
        "loaded_delivery_within_shortest_plus_two_rate_ge_90pct": carried["rate"] >= .90,
        "compatible_partner_each_scene_robot_2_deliveries_ge_2": all(
            row["robot_2_deliveries"] >= 2 for row in compatible),
        "no_compatible_loaded_non_delivery_episode": all(
            row["robot_2_deliveries"] >= 1 for row in compatible),
        "policy_action_equals_submitted_action": all(
            row["policy_action_equals_submitted_action"]
            and row["action_overrides"] == 0 for row in episodes),
    }
    return {
        "version": VERSION,
        "actor": str(actor_path),
        "actor_sha256": sha256(actor_path.read_bytes()).hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "profiles": list(base.PROFILES),
        "partner_current_actor_action_visible": False,
        "episodes": episodes,
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
    destination.write_text(json.dumps(report, ensure_ascii=False,
                                      sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
