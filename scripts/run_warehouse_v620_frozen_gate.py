#!/usr/bin/env python3
"""Frozen Warehouse development precheck/216 confirmation; no PPO/publish authority.

Never derive execution permission from a proposed seed block. Execution needs
an external source/model/method-bound clearance and an exact single-use plan.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys

import numpy as np

ROLES = ("robot_1", "robot_2")
SPLITS = ("fit", "calibration", "eligibility", "validation")
METHODS = ("queued_path_constrained_tree_v1", "episode_robust_explanation_tree_v1")
PURPOSE = "warehouse-v620-frozen-precheck-and-independent-confirmation"
LOGGER = logging.getLogger("warehouse.frozen_gate")


def threshold_contract():
    return {"internal": {"feedback_and_explanation_coverage": .35,
        "expected_feedback_coverage": .30, "feedback_agreement": .93,
        "explanation_agreement": .95, "maximum_kl": .25, "whole_raw_illegal_and_uncertified": 0},
        "independent": {"feedback_and_explanation_coverage": .30,
        "feedback_and_explanation_actual_actionable_coverage": .30,
        "feedback_agreement": .93, "explanation_agreement": .95, "maximum_kl": .25,
        "whole_raw_illegal_and_uncertified": 0},
        "quality": {"minimum_deliveries": 5.5, "maximum_collisions": 6., "maximum_shutdown_rate": .05}}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump({**value, "development_only": True}, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def read_bound(pointer):
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise RuntimeError("artifact pointer requires explicit path and SHA")
    raw = Path(pointer["path"]).resolve(strict=True).read_bytes()
    if sha256(raw).hexdigest() != pointer["sha256"]:
        raise RuntimeError("bound artifact bytes changed")
    return json.loads(raw)


def pointer(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha(path)}


def validate_seed_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {"splits", "independent"} or set(plan["splits"]) != set(SPLITS):
        raise RuntimeError("exact split and independent seed schedules required")
    counts = {"fit": 100, "calibration": 25, "eligibility": 75, "validation": 75}
    all_cases = []
    for split in SPLITS:
        cases = plan["splits"][split]
        if not isinstance(cases, list) or len(cases) != counts[split]:
            raise RuntimeError("frozen precheck requires 100/25/75/75 exact episodes")
        modes = [r.get("sampling_mode") for r in cases]
        if (split == "fit" and (modes.count("sampled") != 50 or modes.count("argmax") != 50)
                or split != "fit" and modes != ["sampled"]*counts[split]):
            raise RuntimeError("only 50 fit episodes may use argmax")
        all_cases += cases
    independent = plan["independent"]
    if not isinstance(independent, list) or len(independent) != 216:
        raise RuntimeError("independent confirmation requires 216 exact draws")
    for row in all_cases+independent:
        if (any(type(row.get(k)) is not int or row[k] < 0 for k in ("base", "task", "scenario_seed", "action_seed"))
                or row["task"] not in (1, 2, 3) or row["scenario_seed"] != row["base"]*10+row["task"]
                or row["action_seed"] < row["scenario_seed"]):
            raise RuntimeError("invalid explicit scene/action identity or unsupported negative action offset")
    if (len({r["base"] for r in all_cases}) != 275
            or len({r["scenario_seed"] for r in all_cases}) != 275
            or len({r["action_seed"] for r in all_cases+independent}) != 491
            or {r["base"] for r in all_cases} & {r["base"] for r in independent}):
        raise RuntimeError("fit/calibration/eligibility/validation/independent seed leakage")
    if {r["action_seed"] for r in all_cases+independent} & {r["scenario_seed"] for r in all_cases+independent}:
        raise RuntimeError("environment and action arguments must occupy disjoint cleared namespaces")
    bases = {r["base"] for r in independent}
    if len(bases) != 24:
        raise RuntimeError("independent draws need exactly 24 base scene clusters")
    expected = {(base, task, stream) for base in bases for task in (1, 2, 3) for stream in (0, 1, 2)}
    keys = {(r["base"], r["task"], r.get("stream")) for r in independent}
    if (keys != expected or any(type(r.get("stream")) is not int or r["sampling_mode"] != "sampled" for r in independent)):
        raise RuntimeError("independent schedule must be 24 bases x 3 tasks x 3 draws")
    return plan


def reset_contract(seeds):
    return {"collection": [r["scenario_seed"] for split in SPLITS for r in seeds["splits"][split]]
            + [r["scenario_seed"] for r in seeds["independent"]],
            "physical_integrity_replay": [r["scenario_seed"] for r in seeds["independent"]],
            "independent_requires_internal_pass": True,
            "maximum_collection_resets": 491, "maximum_integrity_replay_resets": 216}


def gate(metrics, *, independent=False):
    checks = {}
    contract = threshold_contract()["independent" if independent else "internal"]
    for purpose in ("feedback", "explanation"):
        minimum_agreement = contract[purpose+"_agreement"]
        row = metrics[purpose]
        minimum = contract["feedback_and_explanation_coverage"]
        checks[purpose+"_coverage"] = row["coverage"] >= minimum
        checks[purpose+"_agreement"] = row.get("agreement") is not None and row["agreement"] >= minimum_agreement
        checks[purpose+"_kl"] = row.get("kl") is not None and row["kl"] <= contract["maximum_kl"]
        checks[purpose+"_whole_raw_legal"] = row["whole_tree_raw_illegal_count"] == 0
        checks[purpose+"_whole_path_proven"] = row["whole_tree_path_uncertified_count"] == 0
        if independent:
            checks[purpose+"_actual_actionable_coverage"] = row["actionable_coverage"] >= contract["feedback_and_explanation_actual_actionable_coverage"]
    if not independent:
        checks["expected_feedback_coverage"] = metrics["feedback"]["expected_actionable_coverage"] >= contract["expected_feedback_coverage"]
    return {"passed": all(checks.values()), "checks": checks}


def quality_summary(episodes):
    if not episodes:
        raise RuntimeError("quality requires actual same-trajectory outcomes")
    return {"mean_deliveries": float(np.mean([r["deliveries"] for r in episodes])),
        "mean_collisions": float(np.mean([r["collisions"] for r in episodes])),
        "shutdown_episode_rate": float(np.mean([r["shutdown_episode"] for r in episodes]))}


def quality_gate(quality):
    contract = threshold_contract()["quality"]
    return (quality["mean_deliveries"] >= contract["minimum_deliveries"]
            and quality["mean_collisions"] <= contract["maximum_collisions"]
            and quality["shutdown_episode_rate"] <= contract["maximum_shutdown_rate"])


def program_cluster_intervals(by_base, *, resamples=5000):
    if len(by_base) != 24:
        raise RuntimeError("program CI requires 24 whole base clusters")
    rng = np.random.default_rng(620_20261019)
    draws = rng.integers(0, 24, size=(resamples, 24))
    result = {}
    for purpose in ("feedback", "explanation"):
        rows = [by_base[base][purpose] for base in sorted(by_base)]
        frames = np.array([r["sample_count"] for r in rows], float)
        qualified = np.array([r["rows"] for r in rows], float)
        numerator = {
            "coverage": qualified,
            "actual_actionable_coverage": np.array([r["actionable_rows"] for r in rows], float),
            "expected_actionable_coverage": np.array([r["expected_actionable_rows"] for r in rows], float),
            "agreement": np.array([(r["agreement"] or 0)*r["rows"] for r in rows]),
            "kl": np.array([(r["kl"] or 0)*r["rows"] for r in rows])}
        result[purpose] = {}
        for metric, values in numerator.items():
            denominator = qualified if metric in ("agreement", "kl") else frames
            num, den = values[draws].sum(1), denominator[draws].sum(1)
            valid = den > 0
            estimates = num[valid]/den[valid]
            result[purpose][metric] = {"base_scene_bootstrap_95_interval":
                [float(x) for x in np.quantile(estimates, [.025, .975])] if valid.any() else None,
                "valid_resamples": int(valid.sum())}
    return {"base_scene_count": 24, "bootstrap_resamples": resamples,
        "weighting": "resample_whole_bases_then_sum_frame_numerators_and_denominators",
        "scope": "within_frozen_model_scene_uncertainty_not_training_seed_generalization", "metrics": result}


class Runtime:
    """Existing production helpers; lazy import keeps synthetic tests isolated."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        sys.path.insert(0, str(self.root))
        from backend.training import warehouse_v6_runner as runner
        from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
        from backend.training.warehouse_v6_program import fit_program, audit_frozen_program, WarehouseV6Program
        from scripts.run_warehouse_v619_fixed_validation import quality_row_from_recorded_nn, actor_identity
        from scripts.run_warehouse_v619_cached_comparison import program_metrics
        from scripts.diagnose_warehouse_v618_baseline import cluster_summary
        from experiments.warehouse.quality_replay import replay_series
        from env.warehouse.reset_audit_v614 import capture_stage1_resets, stage1_reset_phase
        self.runner, self.Actor, self.Program = runner, FrozenWarehouseV6Actor, WarehouseV6Program
        self.fit_fn, self.audit_fn = fit_program, audit_frozen_program
        self.quality_fn, self.identity_fn, self.cluster_fn = quality_row_from_recorded_nn, actor_identity, cluster_summary
        self.internal_metrics_fn = program_metrics
        self.replay_fn, self.capture, self.phase = replay_series, capture_stage1_resets, stage1_reset_phase

    def source(self):
        return self.runner.source_closure_sha256()

    def load(self, artifact):
        return self.Actor.load(artifact["path"], expected_sha256=artifact["sha256"],
                               expected_role_sha256=artifact["role_sha256"])

    def identity(self, actor):
        return self.identity_fn(actor)

    def collect(self, actor, case, journal):
        # Requires the accompanying compatibility patch to the one native
        # wrapper. This entry never manipulates private execution markers.
        return self.runner.collect_authorized_program_rows(actor, [case["scenario_seed"]],
            deterministic=case["sampling_mode"] == "argmax",
            action_seed_offset=case["action_seed"]-case["scenario_seed"],
            include_episode_outcomes=True, journal_path=journal)

    def trace(self, rows, split, cases, directory, actor):
        return self.runner._write_final_fit_trace(rows, split=split,
            seeds=[r["scenario_seed"] for r in cases], trace_root=directory,
            actor_sha256=actor.artifact_sha256)

    def fit(self, role, actor, parts, method):
        return self.fit_fn(role, actor, *(parts[name][role] for name in SPLITS), method=method)

    def save(self, program, path, role_sha):
        path.parent.mkdir(parents=True, exist_ok=True)
        program.save(path)
        return self.Program.load(path, expected_sha256=file_sha(path), actor_role_sha256=role_sha)

    def metrics(self, program, actor, rows):
        return self.audit_fn(program, actor, rows)

    def internal_metrics(self, program, actor, rows):
        return self.internal_metrics_fn(program, actor, rows)

    def quality(self, rows, outcome, actor):
        return self.quality_fn(rows, outcome, actor)

    def cluster_quality(self, episodes):
        return self.cluster_fn([{**r, "base_seed": r["base"]} for r in episodes])

    def replay(self, artifact, episodes, directory, reset_sink):
        # Keep this list alive in the caller even if replay raises/interruption
        # happens after a successful physical reset.
        with self.capture(directory/"physical-replay.resets.jsonl") as records:
            try:
                with self.phase("quality"):
                    result = self.replay_fn(actor_path=artifact["path"], actor_sha256=artifact["sha256"],
                        role_sha256=artifact["role_sha256"], rows=episodes,
                        seeds=[(r["seed"], r["action_seed"]) for r in episodes])
                return result
            finally:
                reset_sink.extend(records)


def validate_bindings(model_lock, seed_plan, ledger, runtime, output):
    model = read_bound(model_lock)
    seeds, ledger = read_bound(seed_plan), read_bound(ledger)
    validate_seed_plan(seeds)
    if (model.get("schema") != "warehouse-v620-model-method-lock.v1" or model.get("locked") is not True
            or model.get("program_method") not in METHODS or not isinstance(model.get("candidate_id"), str)
            or model.get("source_closure_sha256") != runtime.source()):
        raise RuntimeError("explicit current-source locked model/method receipt required")
    artifact = model["actor"]
    if (not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "role_sha256"}
            or set(artifact["role_sha256"]) != set(ROLES) or file_sha(artifact["path"]) != artifact["sha256"]):
        raise RuntimeError("frozen dual Actor binding is incomplete or changed")
    recipe, selection = read_bound(model["recipe"]), read_bound(model["method_selection_receipt"])
    if (selection.get("locked") is not True or selection.get("program_method") != model["program_method"]
            or selection.get("actor_sha256") != artifact["sha256"]):
        raise RuntimeError("method selection receipt must bind the same exact frozen Actor")
    if (ledger.get("schema") != "warehouse-frozen-seed-clearance.v1"
            or ledger.get("approved_for_frozen_validation") is not True or ledger.get("purpose") != PURPOSE
            or ledger.get("candidate_id") != model["candidate_id"]
            or ledger.get("seed_plan_sha256") != digest(seeds)
            or ledger.get("authorized_reset_schedule_sha256") != digest(reset_contract(seeds))
            or ledger.get("model_method_lock_sha256") != model_lock["sha256"]
            or ledger.get("source_closure_sha256") != runtime.source()
            or ledger.get("recipe_sha256") != model["recipe"]["sha256"]
            or ledger.get("output_directory") != str(Path(output).resolve())
            or ledger.get("unresolved") != [] or ledger.get("remaining_conflicts") != []
            or ledger.get("PPO_joint_steps") != 0 or type(ledger.get("PPO_joint_steps")) is not int
            or ledger.get("publishable") is not False):
        raise RuntimeError("proposals/audit drafts cannot replace exact source/model-bound seed clearance")
    actor = runtime.load(artifact)
    if runtime.identity(actor) != artifact["role_sha256"]:
        raise RuntimeError("loaded frozen Actor role hashes differ")
    return model, seeds, actor


def prepare(model_path, seeds_path, ledger_path, output, runtime):
    refs = {"model_lock": pointer(model_path), "seed_plan": pointer(seeds_path), "ledger": pointer(ledger_path)}
    output = Path(output).resolve()
    model, seeds, actor = validate_bindings(**refs, runtime=runtime, output=output)
    plan = {"schema": "warehouse-v620-frozen-gate-plan.v1", "purpose": PURPOSE,
        "candidate_id": model["candidate_id"], "output_directory": str(output), "bindings": refs,
        "source_closure_sha256": runtime.source(), "script_sha256": file_sha(__file__),
        "actor": model["actor"], "program_method": model["program_method"],
        "seed_plan_sha256": digest(seeds), "PPO_joint_steps": 0, "actor_updates": 0,
        "acceptance_thresholds": threshold_contract(),
        "publishable": False, "admission_evidence": False, "pilot_opened": False,
        "independent_requires_internal_pass": True, "scientific_retry_permitted": False}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"plan.json", plan)
    write_new(output/"plan.claim.json", {"plan_sha256": file_sha(output/"plan.json"), "purpose": PURPOSE,
        "source_closure_sha256": runtime.source(), "status": "claimed_single_use_even_if_interrupted"})
    return output/"plan.json"


def execute(plan_path, runtime):
    path = Path(plan_path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    claim = json.loads((path.parent/"plan.claim.json").read_bytes())
    plan_sha256 = file_sha(path)
    if (claim["plan_sha256"] != plan_sha256 or plan.get("schema") != "warehouse-v620-frozen-gate-plan.v1"
            or plan.get("purpose") != PURPOSE or plan.get("output_directory") != str(path.parent)
            or plan.get("script_sha256") != file_sha(__file__)
            or plan.get("source_closure_sha256") != runtime.source()
            or plan.get("acceptance_thresholds") != threshold_contract()
            or any(plan.get(k) is not False for k in ("publishable", "admission_evidence", "pilot_opened", "scientific_retry_permitted"))
            or any(type(plan.get(k)) is not int or plan[k] != 0 for k in ("PPO_joint_steps", "actor_updates"))
            or plan.get("independent_requires_internal_pass") is not True):
        raise RuntimeError("frozen gate immutable single-use plan changed")
    model, seeds, actor = validate_bindings(**plan["bindings"], runtime=runtime, output=path.parent)
    if plan["actor"] != model["actor"] or plan["program_method"] != model["program_method"] or plan["seed_plan_sha256"] != digest(seeds):
        raise RuntimeError("frozen gate model/method/seed plan drifted")
    directory = path.parent
    write_new(directory/"started.json", {"plan_sha256": file_sha(path), "publishable": False})
    collection_resets, replay_resets, parts, programs, program_pointers = [], [], {}, {}, {}

    def guard():
        if (runtime.source() != plan["source_closure_sha256"] or file_sha(__file__) != plan["script_sha256"]
                or file_sha(path) != claim["plan_sha256"] or file_sha(plan["actor"]["path"]) != plan["actor"]["sha256"]
                or runtime.identity(actor) != plan["actor"]["role_sha256"]):
            raise RuntimeError("source, script, plan or frozen Actor changed during gate")
        for value in plan["bindings"].values():
            read_bound(value)
        for value in (model["recipe"], model["method_selection_receipt"], *program_pointers.values()):
            if file_sha(value["path"]) != value["sha256"]:
                raise RuntimeError("recipe, selection or frozen program bytes changed")

    def collect(case):
        guard()
        rows, evidence = runtime.collect(actor, case,
            directory/"reset-journals"/f"collection-{len(collection_resets):03d}.jsonl")
        records = evidence["reset_audit"]["resets"]
        collection_resets.extend(records)
        if len(records) != 1 or records[0]["seed"] != case["scenario_seed"]:
            raise RuntimeError("one collected episode must match the declared reset")
        return rows, evidence

    try:
        for split in SPLITS:
            merged = {role: [] for role in ROLES}
            for index, case in enumerate(seeds["splits"][split]):
                rows, evidence = collect(case)
                write_new(directory/"collection"/split/f"episode-{index:03d}.json", {
                    "case": case, "evidence": evidence, "all_physical_frames": True})
                for role in ROLES:
                    merged[role].extend(rows[role])
            trace = runtime.trace(merged, split, seeds["splits"][split], directory/"traces", actor)
            write_new(directory/"collection"/f"{split}.json", {"trace": trace,
                "episodes": len(seeds["splits"][split]), "physical_frames_per_role": len(merged[ROLES[0]]), "excluded_frames": 0})
            parts[split] = merged
            LOGGER.info("collected split=%s episodes=%s physical_frames_per_role=%s", split,
                        len(seeds["splits"][split]), len(merged[ROLES[0]]))
        internal = {}
        for role in ROLES:
            guard()
            LOGGER.info("program_fit_started role=%s method=%s", role, model["program_method"])
            program = runtime.fit(role, actor, parts, model["program_method"])
            target = directory/"programs"/f"{role}.json"
            programs[role] = runtime.save(program, target, model["actor"]["role_sha256"][role])
            program_pointers[role] = pointer(target)
            metric = runtime.internal_metrics(programs[role], actor, parts["validation"][role])
            internal[role] = {"metrics": metric, "gate": gate(metric), "program": program_pointers[role]}
            LOGGER.info("internal_gate role=%s passed=%s", role, internal[role]["gate"]["passed"])
        internal_passed = all(row["gate"]["passed"] for row in internal.values())
        write_new(directory/"internal.json", {"roles": internal, "passed": internal_passed,
            "programs_frozen_before_independent": True})
        independent = None
        if internal_passed:
            write_new(directory/"independent.started.json", {"programs": program_pointers,
                "schedule_sha256": digest(seeds["independent"])})
            minimal, episodes = {role: [] for role in ROLES}, []
            for index, case in enumerate(seeds["independent"]):
                rows, evidence = collect(case)
                outcome = evidence["episode_outcomes"][0]
                quality = runtime.quality(rows, outcome, actor)
                episodes.append({**quality, "task": case["task"], "base": case["base"],
                    "stream": case["stream"], "action_seed": case["action_seed"]})
                trace = runtime.trace(rows, "validation", [case], directory/"independent"/f"episode-{index:03d}", actor)
                write_new(directory/"independent"/f"episode-{index:03d}.json", {"case": case,
                    "evidence": evidence, "quality": quality, "joint_trace": trace})
                for role in ROLES:
                    minimal[role] += [{**{key: row[key] for key in ("obs", "mask", "base_seed", "episode_id", "frame", "sampled_action")},
                        "episode": f"{case['scenario_seed']}:{case['action_seed']}", "cluster_base": case["base"]} for row in rows[role]]
                if (index+1) % 24 == 0:
                    LOGGER.info("independent_collected episodes=%s of=216", index+1)
            audits = {role: runtime.metrics(programs[role], actor, minimal[role]) for role in ROLES}
            gates = {role: gate(metrics, independent=True) for role, metrics in audits.items()}
            quality = quality_summary(episodes)
            per_base = {role: {base: runtime.metrics(programs[role], actor,
                [r for r in minimal[role] if r["cluster_base"] == base])
                for base in sorted({r["base"] for r in episodes})} for role in ROLES}
            guard()
            replay = runtime.replay(model["actor"], episodes, directory, replay_resets)
            if ([r["seed"] for r in replay_resets] != [r["scenario_seed"] for r in seeds["independent"]]):
                raise RuntimeError("physical integrity replay reset sequence differs")
            independent = {"passed": quality_gate(quality) and all(r["passed"] for r in gates.values()),
                "quality": quality, "quality_by_task": {str(task): quality_summary([r for r in episodes if r["task"] == task]) for task in (1, 2, 3)},
                "roles": audits, "role_gates": gates, "clustered_scene_quality": runtime.cluster_quality(episodes),
                "clustered_scene_program": {role: program_cluster_intervals(per_base[role]) for role in ROLES},
                "integrity_replay": replay, "integrity_replay_is_not_additional_quality_evaluation": True}
        guard()
        expected = [r["scenario_seed"] for split in SPLITS for r in seeds["splits"][split]]
        if internal_passed:
            expected += [r["scenario_seed"] for r in seeds["independent"]]
        if [r["seed"] for r in collection_resets] != expected:
            raise RuntimeError("actual collection reset sequence differs from exact clearance")
        result = {"schema": "warehouse-v620-frozen-gate-result.v1", "plan_sha256": file_sha(path),
            "development_only": True,
            "candidate_id": plan["candidate_id"], "internal": internal, "internal_passed": internal_passed,
            "independent": independent, "passed": bool(internal_passed and independent and independent["passed"]),
            "collection_resets": collection_resets, "physical_replay_resets": replay_resets,
            "actor_weights_unchanged": True, "PPO_joint_steps": 0, "actor_updates": 0,
            "publishable": False, "admission_evidence": False, "pilot_opened": False}
        write_new(directory/"result.json", result)
        return result
    except BaseException as error:
        interrupted = getattr(error, "warehouse_reset_audit", None)
        if interrupted is not None:
            collection_resets.extend(interrupted["resets"])
        write_new(directory/"failure.json", {"plan_sha256": plan_sha256,
            "error_type": type(error).__name__, "error": str(error),
            "collection_resets": collection_resets, "physical_replay_resets": replay_resets,
            "interrupted_operation_reset_audit": interrupted, "scientific_retry_permitted": False,
            "PPO_joint_steps": 0, "publishable": False, "pilot_opened": False})
        raise


def main(argv=None):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    runtime = Runtime(args.repo_root)
    if args.prepare:
        if args.plan or not all((args.model_lock, args.seeds, args.ledger, args.output_dir)):
            parser.error("prepare requires model-lock, seeds, ledger and output-dir")
        print(json.dumps({"plan": str(prepare(args.model_lock, args.seeds, args.ledger, args.output_dir, runtime)),
            "development_only": True, "rollouts_started": False}))
        return 0
    if not args.plan or any((args.model_lock, args.seeds, args.ledger, args.output_dir)):
        parser.error("execute requires only plan and repo-root")
    result = execute(args.plan, runtime)
    print(json.dumps({"passed": result["passed"], "PPO_joint_steps": 0, "publishable": False, "development_only": True}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
