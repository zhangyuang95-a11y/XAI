#!/usr/bin/env python3
"""Source-locked fixed-NN development validation, never PPO or publication."""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, role_sha256
from backend.training.warehouse_v6_program import audit_frozen_program, fit_program
from env.warehouse.navigation import ACTIONS
from env.warehouse.reset_audit_v614 import capture_stage1_resets, stage1_reset_phase
from experiments.warehouse import v619_protocol as v
from experiments.warehouse.protocol import quality_pass
from experiments.warehouse.quality_replay import (
    TRACE_SCHEMA, ACTION_ENCODING, replay_series,
)
from scripts.diagnose_warehouse_v618_program_feasibility import metric_gates
from scripts.diagnose_warehouse_v618_baseline import cluster_summary


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def checked_actor():
    precheck = ROOT / v.BASELINE_RUN / "final-bc-internal-precheck.json"
    artifact = json.loads(precheck.read_bytes())["final_bc_actor"]
    if artifact["sha256"] != v.ACTOR_SHA256:
        raise RuntimeError("v6.19 fixed baseline Actor identity changed")
    actor = FrozenWarehouseV6Actor.load(artifact["path"],
        expected_sha256=v.ACTOR_SHA256, expected_role_sha256=artifact["role_sha256"])
    return actor, artifact


def actor_identity(actor):
    return {role: role_sha256(actor.roles[role].weights,
                              model_version=actor.roles[role].model_version)
            for role in v.ROLES}


def summary(episodes):
    if not episodes:
        raise ValueError("empty fixed NN quality collection")
    return {"mean_deliveries": float(np.mean([r["deliveries"] for r in episodes])),
            "mean_collisions": float(np.mean([r["collisions"] for r in episodes])),
            "shutdown_episode_rate": float(np.mean([r["shutdown_episode"] for r in episodes]))}


def quality_row_from_recorded_nn(rows, outcome, actor):
    """Build an integrity trace from the same recorded rollout, not a new trial."""
    first, second = rows[v.ROLES[0]], rows[v.ROLES[1]]
    if not first or len(first) != len(second) or len(first) != outcome["frame_count"]:
        raise RuntimeError("fixed NN role/outcome frame counts differ")
    scenario, action_seed = outcome["seed"], outcome["action_seed"]
    chain = sha256(json_bytes({"schema": TRACE_SCHEMA,
        "actor_sha256": actor.artifact_sha256, "scenario_seed": scenario,
        "action_seed": action_seed, "episode_id": first[0]["episode_id"]})).digest()
    action_bytes = bytearray()
    for index, (a, b) in enumerate(zip(first, second, strict=True)):
        for key in ("frame", "base_seed", "episode_id", "pre_state_public_sha256", "post_state_public_sha256"):
            if a[key] != b[key]:
                raise RuntimeError("fixed NN role trajectories are not aligned")
        if (a["frame"] != index or a["base_seed"] != scenario
                or a["actor_sha256"] != actor.artifact_sha256
                or b["actor_sha256"] != actor.artifact_sha256):
            raise RuntimeError("fixed NN rollout identity differs")
        x, y = ACTIONS.index(a["sampled_action"]), ACTIONS.index(b["sampled_action"])
        action_bytes.append(5*x+y)
        chain = sha256(chain+json_bytes([index, a["pre_state_public_sha256"],
            x, y, a["post_state_public_sha256"]])).digest()
    trace = {"schema": TRACE_SCHEMA, "actor_sha256": actor.artifact_sha256,
        "actor_role_sha256": actor.role_sha256, "scenario_seed": scenario,
        "action_seed": action_seed, "episode_id": first[0]["episode_id"],
        "frame_count": len(first), "action_encoding": ACTION_ENCODING,
        "actions_b64": base64.b64encode(action_bytes).decode(),
        "initial_pre_state_sha256": first[0]["pre_state_public_sha256"],
        "final_post_state_sha256": first[-1]["post_state_public_sha256"],
        "chain_sha256": chain.hex()}
    return {"seed": scenario, "deliveries": outcome["deliveries"],
            "collisions": outcome["collisions"],
            "shutdown_episode": outcome["shutdown_episode"], "quality_trace": trace}


def prepare(ledger_path):
    ledger_path = Path(ledger_path).resolve(strict=True)
    ledger = json.loads(ledger_path.read_bytes())
    seeds = v.seed_plan()
    if (ledger.get("approved_for_fixed_validation") is not True
            or ledger.get("fixed_validation_seed_plan_sha256") != v.digest(seeds)):
        raise RuntimeError("v6.19 fixed validation requires a matching cleared seed ledger")
    actor, artifact = checked_actor()
    recipe_sha = v.validate_recipe(ROOT)
    directory = (ROOT / v.OUTPUT_ROOT / "fixed-validation-01").resolve()
    if directory.exists():
        raise RuntimeError("v6.19 fixed validation was already prepared or attempted")
    directory.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "warehouse-v619-fixed-validation-plan.v1",
        "candidate_id": v.CANDIDATE_ID, "purpose": "fresh_fixed_nn_program_development_validation",
        "primary_arm": "P2", "method": v.METHOD, "actor": artifact,
        "actor_role_sha256": actor_identity(actor), "seed_plan": seeds,
        "source_closure_sha256": runner.source_closure_sha256(),
        "recipe_sha256": recipe_sha, "script_sha256": file_sha(__file__),
        "ledger_path": str(ledger_path), "ledger_sha256": file_sha(ledger_path),
        "output_directory": str(directory), "publishable": False, "PPO_joint_steps": 0,
        "validation_can_select_another_arm": False,
        "independent_collection_requires_internal_validation_pass": True}
    write_new(directory / "plan.json", plan)
    return directory / "plan.json"


def execute(plan_path):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_bytes())
    actor, artifact = checked_actor()
    directory = plan_path.parent
    expected_directory = (ROOT / v.OUTPUT_ROOT / "fixed-validation-01").resolve()
    if (directory != expected_directory or plan.get("output_directory") != str(directory)
            or plan.get("candidate_id") != v.CANDIDATE_ID or plan.get("primary_arm") != "P2"
            or plan.get("method") != v.METHOD or plan.get("actor") != artifact
            or plan.get("seed_plan") != v.seed_plan() or plan.get("publishable") is not False
            or type(plan.get("PPO_joint_steps")) is not int or plan["PPO_joint_steps"] != 0
            or plan.get("validation_can_select_another_arm") is not False
            or plan.get("independent_collection_requires_internal_validation_pass") is not True):
        raise RuntimeError("v6.19 fixed validation immutable identity drifted")
    identity = actor_identity(actor)

    def guard():
        if (runner.source_closure_sha256() != plan["source_closure_sha256"]
                or v.validate_recipe(ROOT) != plan["recipe_sha256"]
                or file_sha(__file__) != plan["script_sha256"]
                or actor_identity(actor) != identity or identity != plan["actor_role_sha256"]
                or file_sha(artifact["path"]) != v.ACTOR_SHA256
                or file_sha(plan["ledger_path"]) != plan["ledger_sha256"]):
            raise RuntimeError("v6.19 fixed validation source, Actor or ledger changed")
    guard()
    ledger = json.loads(Path(plan["ledger_path"]).read_bytes())
    if (ledger.get("approved_for_fixed_validation") is not True
            or ledger.get("fixed_validation_seed_plan_sha256") != v.digest(v.seed_plan())):
        raise RuntimeError("v6.19 fixed validation seed clearance is invalid")
    write_new(directory / "started.json", {"plan_sha256": file_sha(plan_path),
        "source_closure_sha256": plan["source_closure_sha256"], "publishable": False})
    programs, internal = {}, {}
    reset_records, collection_artifacts, parts = [], {}, {}
    try:
        for split, cases in plan["seed_plan"]["splits"].items():
            merged = {role: [] for role in v.ROLES}
            authorizations = []
            for case in cases:
                guard()
                seed = case["scenario_seed"]
                rows, evidence = runner.collect_authorized_program_rows(actor, [seed],
                    deterministic=case["sampling_mode"] == "argmax",
                    include_episode_outcomes=True)
                reset_records.extend(evidence["reset_audit"]["resets"])
                authorizations.append(evidence)
                for role in v.ROLES:
                    merged[role].extend(rows[role])
            trace = runner._write_final_fit_trace(merged, split=split,
                seeds=[r["scenario_seed"] for r in cases],
                trace_root=directory / "collection" / "traces",
                actor_sha256=actor.artifact_sha256)
            receipt = {"trace": trace, "sampling_modes": cases,
                       "authorizations": authorizations, "excluded_frames": 0}
            target = directory / "collection" / f"{split}.json"
            write_new(target, receipt)
            collection_artifacts[split] = {"path": str(target), "sha256": file_sha(target)}
            parts[split] = merged
            print(json.dumps({"collected": split, "episodes": len(cases),
                "physical_frames": len(merged[v.ROLES[0]])}), flush=True)
        for role in v.ROLES:
            guard()
            program = fit_program(role, actor, *(parts[s][role] for s in v.SPLITS),
                                  method=v.METHOD)
            target = directory / "programs" / f"{role}.program.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            program.save(target)
            internal[role] = {"program_path": str(target), "program_sha256": file_sha(target),
                "gates": metric_gates(program.report["validation"]["feedback"],
                                      program.report["validation"]["explanation"])}
            programs[role] = type(program).load(target,
                expected_sha256=file_sha(target), actor_role_sha256=identity[role])
            write_new(directory / f"{role}.internal.json", internal[role])
            print(json.dumps({"role": role, "bc_precheck": internal[role]["gates"]["bc_precheck_35_30"]}), flush=True)
        internal_passed = all(r["gates"]["bc_precheck_35_30"]["passed"] for r in internal.values())
        write_new(directory / "internal-validation.json", {"roles": internal,
            "passed": internal_passed, "programs_frozen_before_independent": True})
        independent = None
        if internal_passed:
            frozen_programs = {role: programs[role].sha256() for role in v.ROLES}
            write_new(directory / "independent.started.json", {
                "actor_sha256": v.ACTOR_SHA256, "program_sha256": frozen_programs,
                "schedule_sha256": v.digest(plan["seed_plan"]["independent"])})
            minimal = {role: [] for role in v.ROLES}
            episodes, episode_paths = [], []
            for i, case in enumerate(plan["seed_plan"]["independent"]):
                guard()
                seed, action = case["scenario_seed"], case["action_seed"]
                rows, evidence = runner.collect_authorized_program_rows(actor, [seed],
                    action_seed_offset=action-seed, include_episode_outcomes=True)
                reset_records.extend(evidence["reset_audit"]["resets"])
                quality = quality_row_from_recorded_nn(rows, evidence["episode_outcomes"][0], actor)
                episodes.append({**quality, "task": case["task"], "base": case["base"],
                                 "stream": case["stream"], "action_seed": action})
                trace = runner._write_final_fit_trace(rows, split="validation", seeds=[seed],
                    trace_root=directory / "independent" / f"episode-{i:03d}",
                    actor_sha256=actor.artifact_sha256)
                episode_paths.append(trace)
                for role in v.ROLES:
                    for row in rows[role]:
                        minimal[role].append({k: row[k] for k in (
                            "obs", "mask", "base_seed", "episode_id", "frame", "sampled_action")}
                            | {"episode": f"{seed}:{action}"})
                if (i+1) % 24 == 0:
                    print(json.dumps({"independent_episodes": i+1, "of": 216}), flush=True)
            audits = {role: audit_frozen_program(programs[role], actor, minimal[role]) for role in v.ROLES}
            metrics = {role: metric_gates(audits[role]["feedback"], audits[role]["explanation"])
                       for role in v.ROLES}
            quality_summary = summary(episodes)
            passed = (quality_pass(quality_summary, development=True) and all(
                r["development_30_realized"]["passed"] for r in metrics.values()))
            write_new(directory / "independent-episodes.json", {"episodes": episodes,
                "joint_traces": episode_paths, "metrics": metrics, "quality": quality_summary})
            with capture_stage1_resets(directory / "independent-replay.resets.jsonl") as replay_resets:
                with stage1_reset_phase("quality"):
                    replay = replay_series(actor_path=artifact["path"], actor_sha256=v.ACTOR_SHA256,
                        role_sha256=identity, rows=episodes,
                        seeds=[(r["seed"], r["action_seed"]) for r in episodes])
            if (len(replay_resets) != 216 or
                    {role: programs[role].sha256() for role in v.ROLES} != frozen_programs):
                raise RuntimeError("independent frozen programs or integrity replay changed")
            independent = {"passed": passed, "quality": quality_summary,
                "quality_by_task": {str(task): summary([r for r in episodes if r["task"] == task])
                                    for task in v.TASKS},
                "clustered_scene_summary": cluster_summary([
                    {**row, "base_seed": row["base"]} for row in episodes]),
                "roles": metrics, "quality_integrity_replay": replay,
                "integrity_replay_resets": len(replay_resets),
                "integrity_replay_is_not_additional_evaluation": True}
        guard()
        expected = [r["scenario_seed"] for cases in plan["seed_plan"]["splits"].values() for r in cases]
        if internal_passed:
            expected += [r["scenario_seed"] for r in plan["seed_plan"]["independent"]]
        if [r["seed"] for r in reset_records] != expected:
            raise RuntimeError("fixed validation actual reset sequence differs from plan")
        result = {"schema": "warehouse-v619-fixed-validation-result.v1",
            "candidate_id": v.CANDIDATE_ID, "publishable": False, "PPO_joint_steps": 0,
            "passed": bool(internal_passed and independent and independent["passed"]),
            "internal_passed": internal_passed, "roles": internal, "independent": independent,
            "plan_sha256": file_sha(plan_path), "source_closure_sha256": plan["source_closure_sha256"],
            "actor_sha256": v.ACTOR_SHA256, "actor_role_sha256": identity,
            "actor_weights_unchanged": actor_identity(actor) == identity,
            "collections": collection_artifacts, "actual_collection_reset_count": len(reset_records),
            "actual_collection_resets_sha256": v.digest(reset_records),
            "scope": "fixed_actor_feasibility_not_paired_training_or_release_acceptance"}
        write_new(directory / "result.json", result)
        print(json.dumps({"completed": True, "passed": result["passed"],
            "PPO_joint_steps": 0, "result": str(directory / "result.json")}), flush=True)
        return 0 if result["passed"] else 1
    except BaseException as exc:
        interrupted_audit = getattr(exc, "warehouse_reset_audit", None)
        if interrupted_audit is not None:
            reset_records.extend(interrupted_audit["resets"])
        write_new(directory / "failure.json", {"publishable": False, "PPO_joint_steps": 0,
            "error": f"{type(exc).__name__}: {exc}", "plan_sha256": file_sha(plan_path),
            "actual_collection_reset_count": len(reset_records),
            "actual_collection_resets": reset_records,
            "interrupted_operation_reset_audit": interrupted_audit,
            "source_closure_sha256": plan["source_closure_sha256"]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--ledger")
    parser.add_argument("--plan")
    args = parser.parse_args()
    if args.prepare:
        if not args.ledger or args.plan:
            parser.error("--prepare requires --ledger and no --plan")
        print(json.dumps({"plan": str(prepare(args.ledger)), "rollouts_started": False}))
        return 0
    if not args.plan or args.ledger:
        parser.error("execution requires --plan only")
    return execute(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
