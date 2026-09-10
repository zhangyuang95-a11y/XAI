"""Fixed-play task matching audit; public programs only, never model selection.

This entry point intentionally does not import the neural runtime, training,
evaluation or participant service. It reads only play[0:7] from the frozen
scenario manifest and writes a new, separate audit directory. The protocol is
written before the first environment step and is never tuned from results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean
import sys
import time

from env.warehouse_native.environment import ENVIRONMENT_VERSION, NativeWarehouseEnv
from env.warehouse_native.partners import partner_action
from env.warehouse_native.scenarios import reset_scenario


VERSION = "warehouse-native-fixed-play-engineering-match-v1"
ROOT = Path(__file__).resolve().parents[2]
PERTURBATION_SEEDS = tuple(range(914300, 914310))
PROTOCOL = {
    "version": VERSION,
    "purpose": "fixed research-map development audit; not neural capability or statistical equivalence",
    "scene_indices": {"practice": [0], "X": [1, 2, 3], "Y": [4, 5, 6]},
    "pair_indices": [[1, 4], [2, 5], [3, 6]],
    "primary_matching": {
        "reference_group_mean_relative_difference_max": 0.15,
        "relative_difference_definition": "abs(mean_X - mean_Y) / ((mean_X + mean_Y) / 2)",
        "reference_pair_absolute_delivery_difference_max": 2,
        "both_reference_group_means_must_be_positive": True,
        "practice_excluded_from_matching": True,
        "all_conditions_required": True,
    },
    "reference": {"episodes_per_scene": 1, "robot_1": "skilled", "robot_2": "skilled"},
    "perturbed": {
        "episodes_per_scene": 10, "robot_1": "skilled", "robot_2": "noisy",
        "rng_seeds": list(PERTURBATION_SEEDS),
        "noise": "frozen noisy partner: independently replace robot_2 command with a uniform public action with probability 0.10",
        "paired_rng": "same ten partner RNG seeds reused for every scene; environment RNG restored from that scene",
        "matching_role": "descriptive robustness diagnostic only; no additional post-hoc pass threshold",
    },
    "synchrony": "both program actions are computed from the unchanged pre-action state",
    "stopping": "physical termination or configured horizon; no early discretionary end",
    "scope": "same fixed topology, different frozen initial task configurations and task-sampler RNG states",
    "prohibited_inputs": ["neural models", "participant records", "final_test rollouts"],
    "on_failure": "retain maps and thresholds; report failed/candidate; do not change study configuration",
    "training_joint_steps": 0,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def selected_scenes(manifest):
    """Do not iterate or evaluate any training/held-out/participant split."""
    play = manifest.get("splits", {}).get("play", [])
    if len(play) < 7:
        raise ValueError("The fixed protocol requires play[0:7]; no substitute scenes are permitted")
    selected = deepcopy(play[:7])
    if len({scene["id"] for scene in selected}) != 7 or len({scene["fingerprint"] for scene in selected}) != 7:
        raise ValueError("Fixed play scenes must have seven distinct IDs and physical fingerprints")
    return selected


def episode_metrics(env, step_calls):
    state = env.state
    return {
        "deliveries": int(state.total_deliveries),
        "native_score": float(env.native_score), "legacy_score": None,
        "collisions": int(state.robot_collision_events),
        "shutdowns": int(state.shutdown_count), "invalid_moves": int(state.invalid_move_count),
        "steps": int(state.frame), "environment_step_calls": step_calls,
        "deliveries_by_robot": {agent.agent_id: int(agent.deliveries_completed) for agent in state.agents},
        "remaining_battery": {agent.agent_id: float(agent.battery) for agent in state.agents},
        "terminated": bool(state.terminated), "truncated": bool(state.truncated),
        "terminal_reason": state.terminal_reason,
    }


def run_episode(scene, play_index, partner_kind, rng_seed, trace):
    env = NativeWarehouseEnv()
    reset_scenario(env, scene)
    initial_snapshot = env.snapshot()
    episode_id = f"play_{play_index:02d}_{partner_kind}_{rng_seed if rng_seed is not None else 'deterministic'}"
    rng = random.Random(rng_seed if rng_seed is not None else 0)
    role = "practice" if play_index == 0 else "X" if play_index <= 3 else "Y"
    base = {"episode_id": episode_id, "scene_id": scene["id"], "play_index": play_index,
            "task_set": role, "partner_kind": partner_kind, "partner_rng_seed": rng_seed}
    trace.write(canonical({"record": "episode_start", **base, "initial_snapshot": initial_snapshot,
                           "initial_snapshot_sha256": digest(initial_snapshot), "scene_fingerprint": scene["fingerprint"]}) + "\n")
    step_calls = 0
    event_counts = Counter()
    requested_counts = {key: Counter() for key in env.agent_ids}
    executed_counts = {key: Counter() for key in env.agent_ids}
    while not env.done:
        before_snapshot_hash = digest(env.snapshot())
        actions = {
            "robot_1": partner_action(env, "robot_1", "skilled", rng),
            "robot_2": partner_action(env, "robot_2", "skilled" if partner_kind == "reference" else "noisy", rng),
        }
        if digest(env.snapshot()) != before_snapshot_hash:
            raise AssertionError("Program decision mutated the pre-action state or environment RNG")
        _, _, _, _, info = env.step(actions)
        step_calls += 1
        if step_calls > env.config.horizon:
            raise AssertionError("Episode exceeded its frozen physical horizon")
        event_counts.update(event["event"] for event in info["events"])
        for key in env.agent_ids:
            requested_counts[key][actions[key]] += 1
            executed_counts[key][info["executed_actions"][key]] += 1
        trace.write(canonical({"record": "transition", "episode_id": episode_id, "frame": env.state.frame,
            "before_snapshot_sha256": before_snapshot_hash, "after_snapshot_sha256": digest(env.snapshot()),
            "requested_actions": actions, "executed_actions": info["executed_actions"],
            "invalid_moves": info["invalid_moves"], "events": info["events"], "public_state": env.public_view()}) + "\n")
    result = {**base, "scene_fingerprint": scene["fingerprint"], "initial_snapshot_sha256": digest(initial_snapshot),
              "final_snapshot_sha256": digest(env.snapshot()), "metrics": episode_metrics(env, step_calls),
              "event_counts": dict(event_counts), "requested_action_counts": requested_counts,
              "executed_action_counts": executed_counts}
    trace.write(canonical({"record": "episode_end", **result}) + "\n")
    return result


def distribution(values):
    return {"count": len(values), "mean": mean(values), "min": min(values), "max": max(values)}


def summarize(results):
    expected = {(index, "reference", None) for index in range(7)} | {
        (index, "perturbed", seed) for index in range(7) for seed in PERTURBATION_SEEDS}
    actual = [(row["play_index"], row["partner_kind"], row["partner_rng_seed"]) for row in results]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicate or extra fixed-protocol episode")
    references = {row["play_index"]: row["metrics"]["deliveries"] for row in results if row["partner_kind"] == "reference"}
    if set(references) != set(range(7)):
        raise ValueError("Incomplete or substituted fixed reference scenarios")
    x_mean, y_mean = mean(references[index] for index in (1, 2, 3)), mean(references[index] for index in (4, 5, 6))
    informative = x_mean > 0 and y_mean > 0
    relative = abs(x_mean - y_mean) / ((x_mean + y_mean) / 2) if x_mean + y_mean > 0 else None
    pairs = [{"X_play_index": left, "Y_play_index": right, "X_deliveries": references[left],
              "Y_deliveries": references[right], "absolute_difference": abs(references[left] - references[right]),
              "passed": abs(references[left] - references[right]) <= 2} for left, right in PROTOCOL["pair_indices"]]
    checks = {"nonzero_reference_group_means": informative,
              "reference_group_difference_within_15_percent": relative is not None and relative <= .15,
              "all_reference_pair_differences_at_most_two": all(pair["passed"] for pair in pairs)}
    scene_results = []
    for index in range(7):
        local = [row for row in results if row["play_index"] == index]
        reference = next(row for row in local if row["partner_kind"] == "reference")
        noisy = [row for row in local if row["partner_kind"] == "perturbed"]
        if len(noisy) != 10 or {row["partner_rng_seed"] for row in noisy} != set(PERTURBATION_SEEDS):
            raise ValueError("Incomplete fixed perturbation seeds")
        scene_results.append({"play_index": index, "scene_id": reference["scene_id"], "task_set": reference["task_set"],
            "scene_fingerprint": reference["scene_fingerprint"], "reference_metrics": reference["metrics"],
            "perturbed": {metric: distribution([row["metrics"][metric] for row in noisy])
                          for metric in ("deliveries", "native_score", "collisions", "shutdowns", "invalid_moves", "steps")}})
    perturbed_groups = {group: distribution([row["metrics"]["deliveries"] for row in results
                                           if row["task_set"] == group and row["partner_kind"] == "perturbed"])
                        for group in ("X", "Y")}
    return {"matching_status": "passed" if all(checks.values()) else "failed", "release_status": "candidate",
            "formal_ready": False, "checks": checks, "reference_X_mean_deliveries": x_mean,
            "reference_Y_mean_deliveries": y_mean, "reference_group_relative_difference": relative,
            "pairs": pairs, "scenes": scene_results, "perturbed_group_deliveries": perturbed_groups,
            "environment_audit_joint_steps": sum(row["metrics"]["environment_step_calls"] for row in results),
            "episodes": len(results), "training_joint_steps": 0,
            "limits": ["Engineering matching thresholds are not a statistical equivalence test.",
                "This uses public program pairs; it does not measure human difficulty, neural-team capability or explanation effects.",
                "The seven scenes share one map topology; this evaluates frozen initial task and subsequent task-stream difficulty.",
                "The reference is a program baseline, not an optimality proof; zero performance cannot establish matched usable tasks.",
                "Ten paired noise seeds provide a descriptive robustness check, not an independent participant sample.",
                "No maps, thresholds, runtime release gates or model selection were changed from these results.",
                "Wait counts and direction changes are recorded without being labeled coordination errors."]}


def assert_no_neural_imports():
    prohibited = [name for name in sys.modules if name == "torch" or name.startswith("torch.")
                  or name in {"env.warehouse_native.policy", "env.warehouse_native.runtime"}]
    if prohibited:
        raise RuntimeError("This audit must not load a neural model or PyTorch")


def audit(scenarios_path, output):
    assert_no_neural_imports()
    started = time.perf_counter()
    source_bytes = Path(scenarios_path).read_bytes()
    manifest = json.loads(source_bytes)
    scenes = selected_scenes(manifest)
    source_paths = sorted({Path(__file__).resolve(), * (ROOT / "env/warehouse_native").glob("*.py"),
                           * (ROOT / "env/warehouse").glob("*.py")})
    # Provenance hashes source files; it never imports model code or loads weights.
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths}
    provenance = {"namespace": "development_task_calibration", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL), "environment_version": ENVIRONMENT_VERSION,
        "scenario_file": str(Path(scenarios_path).resolve()), "scenario_file_sha256": sha256(source_bytes).hexdigest(),
        "source_sha256": source_hashes, "python_version": sys.version, "model_loaded": False,
        "evaluated_splits": ["play"], "final_test_rollouts": 0, "participant_data_read": False,
        "selected_scenes": [{"play_index": index, "id": scene["id"], "fingerprint": scene["fingerprint"],
                              "snapshot_sha256": digest(scene["snapshot"])} for index, scene in enumerate(scenes)]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", provenance)
    results = []
    with (output / "transitions.jsonl").open("x", encoding="utf-8") as trace, (output / "episodes.jsonl").open("x", encoding="utf-8") as episodes:
        for index, scene in enumerate(scenes):
            for kind, seed in [("reference", None), *(("perturbed", seed) for seed in PERTURBATION_SEEDS)]:
                result = run_episode(scene, index, kind, seed, trace)
                results.append(result)
                episodes.write(canonical(result) + "\n")
                episodes.flush()
            print(canonical({"completed_scenes": index + 1, "total_scenes": 7,
                             "environment_audit_joint_steps": sum(row["metrics"]["environment_step_calls"] for row in results)}), flush=True)
    assert_no_neural_imports()
    if any(file_hash(ROOT / path) != value for path, value in source_hashes.items()):
        raise RuntimeError("Audited source changed during rollout; no completed report produced")
    if Path(scenarios_path).read_bytes() != source_bytes:
        raise RuntimeError("Frozen scenario manifest changed during rollout")
    report = {**provenance, **summarize(results), "elapsed_seconds": time.perf_counter() - started,
              "artifacts": {name: file_hash(output / name) for name in ("protocol.json", "episodes.jsonl", "transitions.jsonl")}}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New audit directory; existing output is never overwritten")
    args = parser.parse_args()
    report = audit(args.scenarios, args.output)
    print(canonical({key: report[key] for key in ("matching_status", "release_status", "episodes", "environment_audit_joint_steps", "training_joint_steps", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
