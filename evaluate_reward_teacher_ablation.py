"""Run the four-way reward/teacher MAPPO ablation in isolated directories.

The default profiles reproduce the old team reward/strong teacher settings and
the new individual-credit/corrected teacher settings.  ``--quick-proxy`` only
scales BC epochs for a local resource check; reports label that mode clearly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


LEGACY_REWARD = {
    "individual_credit_enabled": 0,
    "progress_scale": 2.0,
    "coordination_clearance_cost": 16.0,
    "coordination_progress_cap": 0.04,
    "task_age_priority_scale": 0.5,
    "task_age_priority_horizon": 40,
    "counterfactual_regret_cost": 0.02,
    "avoidable_wait_streak_cost": 0.01,
    "avoidable_wait_streak_cap": 4,
    "avoidable_wait_cost": 0.01,
    "mission_regression_scale": 1.0,
}

VARIANTS = {
    "current_baseline": (False, False),
    "reward_only": (True, False),
    "teacher_only": (False, True),
    "combined": (True, True),
}


def _run_paths(root: Path, variant: str, seed: int) -> dict[str, Path]:
    run = root / variant / f"seed_{seed}"
    return {
        "root": run,
        "model": run / "warehouse_mappo.pt",
        "checkpoint": run / "training_checkpoint.pt",
        "metrics": run / "training_metrics.csv",
        "summary": run / "training_summary.json",
        "trajectory": run / "training_trajectory.json",
        "stdout": run / "run.log",
    }


def _command(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    paths: dict[str, Path],
) -> list[str]:
    reward_enabled, teacher_enabled = VARIANTS[variant]
    legacy_epochs, corrected_epochs = (10, 3) if args.quick_proxy else (100, 30)
    command = [
        sys.executable,
        "-m",
        "backend.training.warehouse",
        "--episodes",
        str(args.episodes),
        "--horizon",
        str(args.horizon),
        "--hidden-dim",
        str(args.hidden_dim),
        "--intent-dim",
        str(args.intent_dim),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--eval-episodes",
        str(args.eval_episodes),
        "--periodic-eval-every",
        "0",
        "--behavior-cloning-samples",
        str(args.behavior_samples),
        "--behavior-cloning-epochs",
        str(corrected_epochs if teacher_enabled else legacy_epochs),
        "--skill-retention-samples",
        str(args.retention_samples),
        "--skill-retention-weight",
        "1.0" if teacher_enabled else "5.0",
        "--skill-retention-fade-end",
        "0.50",
        "--checkpoint-every",
        "0",
        "--skip-seed-calibration",
        "--skip-reference-calibration",
        "--no-plot",
        "--output",
        str(paths["model"]),
        "--checkpoint-output",
        str(paths["checkpoint"]),
        "--metrics-output",
        str(paths["metrics"]),
        "--summary-output",
        str(paths["summary"]),
        "--trajectory-output",
        str(paths["trajectory"]),
    ]
    if not reward_enabled:
        command.extend(("--reward-overrides", json.dumps(LEGACY_REWARD)))
    if not teacher_enabled:
        command.extend(("--legacy-teacher-supervision", "--skill-retention-fixed"))
    return command


def _extract(summary: dict[str, Any]) -> dict[str, float]:
    neural = summary["evaluation"]["neural_policy"]
    efficiency = neural.get("per_agent_efficiency", {})
    agents = tuple(efficiency.values())
    episodes = max(1.0, float(neural.get("episodes", 1.0)))
    total_agent_steps = max(
        1.0,
        2.0 * episodes * float(neural.get("mean_episode_steps", 0.0)),
    )
    return {
        "mean_deliveries": float(neural["mean_deliveries"]),
        "mean_user_score": float(neural["mean_user_score"]),
        "collision_episode_rate": float(neural["collision_episode_rate"]),
        "shutdown_episode_rate": float(neural["shutdown_episode_rate"]),
        "avoidable_wait_rate": (
            sum(float(item.get("avoidable_wait_count", 0)) for item in agents)
            / total_agent_steps
        ),
        "loaded_detour_count": sum(
            float(item.get("loaded_detour_count", 0)) for item in agents
        ),
        "detour_count": sum(float(item.get("detour_count", 0)) for item in agents),
        "path_efficiency_actual_over_shortest_safe": float(
            neural["path_efficiency_actual_over_shortest_safe"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--intent-dim", type=int, default=32)
    parser.add_argument("--behavior-samples", type=int, default=2048)
    parser.add_argument("--retention-samples", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        default="output/collaborative/safe_mission_v28_individual_credit/"
        "reward_teacher_ablation",
    )
    parsed = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in parsed.seeds.split(","))
    if len(seeds) < 3:
        parser.error("the ablation requires at least three training seeds")
    root = Path(parsed.output_root)
    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in seeds:
            paths = _run_paths(root, variant, seed)
            command = _command(parsed, variant, seed, paths)
            run: dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "command": command,
                "summary": str(paths["summary"]),
            }
            if not parsed.dry_run:
                existing = [
                    paths[name]
                    for name in ("model", "metrics", "summary", "trajectory")
                    if paths[name].exists()
                ]
                if existing:
                    raise FileExistsError(
                        "Refusing to overwrite an ablation run; choose a new "
                        f"--output-root. Existing: {existing}"
                    )
                paths["root"].mkdir(parents=True, exist_ok=True)
                with paths["stdout"].open("w", encoding="utf-8") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                run["metrics"] = _extract(
                    json.loads(paths["summary"].read_text(encoding="utf-8"))
                )
            runs.append(run)
    report: dict[str, Any] = {
        "kind": "reward_teacher_four_way_ablation",
        "quick_proxy": parsed.quick_proxy,
        "seeds": seeds,
        "episodes": parsed.episodes,
        "runs": runs,
    }
    if not parsed.dry_run:
        report["variant_means"] = {
            variant: {
                metric: sum(
                    run["metrics"][metric]
                    for run in runs
                    if run["variant"] == variant
                )
                / len(seeds)
                for metric in next(
                    run["metrics"] for run in runs if run["variant"] == variant
                )
            }
            for variant in VARIANTS
        }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
