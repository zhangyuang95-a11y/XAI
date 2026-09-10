"""Fast fixed-subset diagnostic before an expensive 300-episode r4 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.training.warehouse_r4_active_evaluation import evaluate_actor, _write_json


VERSION = "warehouse-r4-active-screen.v1"


def screen(candidate_actor, baseline_actor, scenarios_path, output_dir, count=10):
    output_dir = Path(output_dir)
    scenarios = json.loads(Path(scenarios_path).read_text())["splits"]["validation"][:count]
    baseline = evaluate_actor(baseline_actor, scenarios, output_dir=output_dir / "r3")["summary"]
    candidate = evaluate_actor(candidate_actor, scenarios, output_dir=output_dir / "candidate")["summary"]
    checks = {
        "active_non_wait_at_least_85pct": candidate["active_non_wait_rate"] >= .85,
        "productive_gain_at_least_4pp": candidate["productive_action_rate"] >= baseline["productive_action_rate"] + .04,
        "ai_delivery_gain_at_least_half": candidate["mean_ai_deliveries"] >= baseline["mean_ai_deliveries"] + .5,
        "collision_not_materially_worse": candidate["collision_cancellation_rate"] <= baseline["collision_cancellation_rate"] + .02,
        "wall_commands_below_one_pct": candidate["static_wall_command_rate"] <= .01,
        "ai_shutdown_rate_not_worse": candidate["ai_shutdown_episode_rate"] <= baseline["ai_shutdown_episode_rate"],
        "no_progress_p95_improved": candidate["no_task_progress_streak_p95"] < baseline["no_task_progress_streak_p95"],
        "action_authority_exact": candidate["action_override_count"] == 0,
    }
    efficacy = checks["productive_gain_at_least_4pp"] or checks["ai_delivery_gain_at_least_half"]
    safety = all(checks[key] for key in (
        "collision_not_materially_worse", "wall_commands_below_one_pct",
        "ai_shutdown_rate_not_worse", "action_authority_exact"))
    result = {
        "version": VERSION,
        "status": "promising_for_full_audit" if efficacy and safety else "diagnose_before_full_audit",
        "official_selection_gate": False,
        "fixed_validation_scene_count": count,
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "efficacy_signal": efficacy,
        "safety_screen": safety,
    }
    _write_json(output_dir / "screen.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-actor", required=True)
    parser.add_argument("--baseline-actor", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    result = screen(args.candidate_actor, args.baseline_actor, args.scenarios,
                    args.output, args.count)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
