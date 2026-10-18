#!/usr/bin/env python3
"""Matched, unsealed, NN-only historical baseline diagnostics (not admission)."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.warehouse.admission import NativeBackend, _exclusive_json, _summary
from experiments.warehouse import protocol as p
from experiments.warehouse.quality_replay import replay_series
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def diagnostic_schedule(base_seeds, *, streams=(0, 1, 2)) -> list[dict]:
    """The same physical scene and draw stream are reused across frozen Actors."""
    seeds = list(base_seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("distinct nonnegative base seeds required")
    rows = []
    for task in p.TASKS:
        for seed in seeds:
            scenario = p.task_scenario_seed(seed, task)
            if is_sealed_scenario_seed(seed) or is_sealed_scenario_seed(scenario):
                raise ValueError("diagnostic schedule overlaps sealed seeds")
            for stream in streams:
                data = f"warehouse-v618-matched-diagnostic|{seed}|{task}|{stream}"
                action = int.from_bytes(sha256(data.encode()).digest()[:8], "big") & 0x7fff_ffff
                rows.append({"task": task, "base_seed": seed,
                             "scenario_seed": scenario, "stream": stream,
                             "action_seed": action})
    return rows


def cluster_summary(rows: list[dict]) -> dict:
    """Average tasks/streams within base scenes before estimating dispersion."""
    import numpy as np
    bases = sorted({row["base_seed"] for row in rows})
    clusters = [_summary([row for row in rows if row["base_seed"] == seed]) for seed in bases]
    rng = np.random.default_rng(618_20261018)
    result = {}
    for name in ("mean_deliveries", "mean_collisions", "shutdown_episode_rate"):
        values = np.array([row[name] for row in clusters])
        means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        result[name] = {"mean": float(values.mean()),
                        "base_scene_bootstrap_95_interval": [float(x) for x in np.quantile(means, [.025, .975])]}
    return {"base_scene_count": len(bases), "bootstrap_resamples": 5000,
            "scope": "within_frozen_model_scene_uncertainty_not_training_seed_generalization",
            "metrics": result}


def paired_scene_changes(reference: list[dict], candidate: list[dict]) -> dict:
    import numpy as np
    keys = ("task", "base_seed", "scenario_seed", "stream", "action_seed")
    if [[row[key] for key in keys] for row in reference] != [[row[key] for key in keys] for row in candidate]:
        raise ValueError("paired diagnostic scene/draw schedules differ")
    bases = sorted({row["base_seed"] for row in reference})
    ref = [_summary([row for row in reference if row["base_seed"] == seed]) for seed in bases]
    new = [_summary([row for row in candidate if row["base_seed"] == seed]) for seed in bases]
    rng = np.random.default_rng(618_20261018)
    metrics = {}
    for key in ("mean_deliveries", "mean_collisions", "shutdown_episode_rate"):
        delta = np.array([right[key]-left[key] for left, right in zip(ref, new, strict=True)])
        boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
        metrics[key] = {"candidate_minus_reference": float(delta.mean()),
                        "paired_base_scene_bootstrap_95_interval": [float(x) for x in np.quantile(boot, [.025, .975])]}
    return {"base_scene_count": len(bases), "metrics": metrics,
            "scope": "frozen_policy_comparison_not_causal_training_method_effect"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors-json", required=True,
                        help="JSON list of {label,path,sha256,role_sha256} frozen artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-seed", type=int, default=263_100)
    parser.add_argument("--base-count", type=int, default=24)
    args = parser.parse_args(argv)
    if not 1 <= args.base_count <= 256:
        parser.error("base-count must be between 1 and 256")
    actors = json.loads(Path(args.actors_json).read_bytes())
    if (not isinstance(actors, list) or not actors
            or len({item.get("label") for item in actors}) != len(actors)):
        parser.error("actors-json requires distinct labeled artifacts")
    for item in actors:
        if (not isinstance(item.get("label"), str) or not item["label"]
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in item["label"])):
            parser.error("Actor labels must be simple directory names")
    backend = NativeBackend()
    closure = backend.source_closure_sha256()
    schedule = diagnostic_schedule(range(args.base_seed, args.base_seed + args.base_count))
    output = Path(args.output_dir).resolve()
    # No resume with changed code, Actor hashes or scene/draw inventory.
    plan = {"schema": "warehouse-v618-matched-baseline-plan.v1",
            "purpose": "spent_development_scene_diagnostic_not_admission",
            "publishable": False, "sealed_evidence": False,
            "source_closure_sha256": closure,
            "diagnostic_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "actors": actors, "schedule": schedule, "schedule_sha256": digest(schedule)}
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != plan:
            raise RuntimeError("diagnostic immutable plan changed")
    else:
        _exclusive_json(plan_path, plan)
    results, all_rows = [], {}
    for artifact in actors:
        if backend.source_closure_sha256() != closure:
            raise RuntimeError("source closure changed during baseline diagnostic")
        actor = backend.load_actor(artifact)
        actor_dir = output / artifact["label"]
        rows = []
        for index, scheduled in enumerate(schedule):
            checkpoint = actor_dir / "episodes" / f"{index:04d}.json"
            if checkpoint.is_file():
                row = json.loads(checkpoint.read_bytes())
                if any(row.get(key) != value for key, value in scheduled.items()):
                    raise RuntimeError("diagnostic saved scene schedule changed")
                if row["quality_trace"]["actor_sha256"] != actor.artifact_sha256:
                    raise RuntimeError("diagnostic saved Actor binding changed")
            else:
                result = backend.evaluate(
                    actor, [scheduled["scenario_seed"]], trace=True,
                    action_seed_offset=scheduled["action_seed"]-scheduled["scenario_seed"])
                row = {**scheduled, **result["episodes"][0]}
                _exclusive_json(checkpoint, row)
            rows.append(row)
            if (index+1) % 24 == 0 or index+1 == len(schedule):
                print(json.dumps({"label": artifact["label"], "episodes": index+1,
                                  "of": len(schedule), "summary_so_far": _summary(rows)}), flush=True)
        if backend.source_closure_sha256() != closure:
            raise RuntimeError("source closure changed during baseline diagnostic")
        replay = replay_series(
            actor_path=artifact["path"], actor_sha256=artifact["sha256"],
            role_sha256=actor.role_sha256, rows=rows,
            seeds=[(row["scenario_seed"], row["action_seed"]) for row in rows])
        if backend.source_closure_sha256() != closure:
            raise RuntimeError("source closure changed during independent replay")
        report = {"schema": "warehouse-v618-matched-baseline-result.v1",
                  "purpose": plan["purpose"], "publishable": False,
                  "label": artifact["label"], "actor": artifact,
                  "source_closure_sha256": closure, "schedule_sha256": digest(schedule),
                  "episode_count": len(rows), "summary": _summary(rows),
                  "per_task": {str(task): _summary([row for row in rows if row["task"] == task]) for task in p.TASKS},
                  "clustered_scene_summary": cluster_summary(rows),
                  "independent_replay": replay, "episodes": rows}
        target = actor_dir / "result.json"
        if target.exists():
            if json.loads(target.read_bytes()) != report:
                raise RuntimeError("diagnostic immutable result changed")
        else:
            _exclusive_json(target, report)
        results.append({"label": artifact["label"], "result_path": str(target),
                        "result_sha256": sha256(target.read_bytes()).hexdigest(),
                        "summary": report["summary"]})
        all_rows[artifact["label"]] = rows
        print(json.dumps({"completed": artifact["label"], "summary": report["summary"]}), flush=True)
    combined = {"schema": "warehouse-v618-matched-baseline-comparison.v1",
                "purpose": plan["purpose"], "publishable": False,
                "source_closure_sha256": closure, "plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
                "results": results,
                "reference_label": actors[0]["label"],
                "paired_changes_vs_reference": {
                    item["label"]: paired_scene_changes(
                        all_rows[actors[0]["label"]], all_rows[item["label"]])
                    for item in actors[1:]}}
    _exclusive_json(output / "comparison.json", combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
