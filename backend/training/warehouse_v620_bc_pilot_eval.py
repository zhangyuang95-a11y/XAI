"""Conditional shared-BC pilot comparison; spent development scenes only.

Importing does not authorize rollout, fitting or publication.
The saved CLI uses a source-bound runtime; engineering tests inject only stubs.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Mapping

SCHEMA = "warehouse-v620-bc-pilot-comparison-spec.v1"
PLAN_SCHEMA = "warehouse-v620-bc-pilot-comparison-plan.v1"
RECIPE_PATH = "configs/warehouse_v620_bc_pilot_comparison_recipe.json"
CANDIDATE_ID = "warehouse-v620-01"
GAMMAS = {"gamma-001": .01, "gamma-003": .03}
ROLES = ("robot_1", "robot_2")
SPLITS = ("fit", "calibration", "eligibility", "validation")
BC_STAGE_COUNTS = (
    ("initial_public_teacher", 100),
    ("initial_nn_dagger_1", 40), ("initial_nn_dagger_2", 40),
    ("joint_augmentation_sampled", 25), ("joint_augmentation_argmax", 25),
    ("risk_public_teacher", 40), ("risk_nn_visitation", 80),
    ("large_public_teacher", 200), ("large_nn_visitation", 120),
    ("deep_public_teacher", 200),
    *((f"deep_{i}_{kind}", 50) for i in range(1, 5) for kind in ("sampled", "argmax")),
)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path)}


def read_pointer(root, item, *, json_object=True):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("pilot evidence pointer malformed")
    path = (Path(root) / item["path"]).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()):
        raise RuntimeError("pilot evidence escaped the registered repository")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != item["sha256"]:
        raise RuntimeError("pilot evidence hash mismatch")
    value = json.loads(raw) if json_object else None
    if json_object and not isinstance(value, dict):
        raise ValueError("pilot evidence requires a JSON object")
    return path, value


def recipe():
    return {
        "schema": "warehouse-v620-bc-pilot-comparison-recipe.v1",
        "candidate_id": CANDIDATE_ID, "gammas": GAMMAS,
        "prerequisite": "complete_cv02_native_failure_chain_and_two_completed_paired_bc_runs",
        "method_selection": "max_full_cv_M1_M2_selection_score_then_M1_on_exact_tie",
        "method_lock_before_any_rollout": True,
        "scenes": "exact_already_spent_v619_124_scenes_and_sampling_modes",
        "episode_counts": {"fit": 50, "calibration": 25, "eligibility": 25, "validation": 24},
        "fit_sampling": {"sampled": 25, "argmax": 25},
        "action_base_seed": "scenario_seed", "excluded_physical_frames": 0,
        "quality_episodes": "same_24_sampled_validation_trajectories_not_additional_samples",
        "quality_physical_replay_required": True,
        "quality_only_hard_eligibility": {"deliveries_min": 5.5, "collisions_max": 6., "shutdown_rate_max": .05},
        "program_consistency_preference": {"both_roles": True, "feedback_agreement_min": .93,
            "explanation_agreement_min": .95, "maximum_qualified_kl": .25,
            "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0},
        "normal_ranking": ["program_consistency_pass", "minimum_role_explanation_coverage",
            "minimum_role_expected_feedback", "negative_maximum_qualified_kl",
            "quality_score", "negative_gamma"],
        "all_quality_eligible_consistency_failed_ranking": ["minimum_normalized_consistency_margin",
            "minimum_role_explanation_coverage", "minimum_role_expected_feedback",
            "negative_maximum_qualified_kl", "quality_score", "negative_gamma"],
        "consistency_margin": "min_over_both_roles(F_agreement/.93,E_agreement/.95,.25/max(F_KL,1e-12),.25/max(E_KL,1e-12))",
        "unavailable_qualified_metric": {"margin": 0., "kl_ordering": "worst"},
        "pilot_coverage_hard_gate": None,
        "quality_score": "deliveries-.25*collisions-5*shutdown_rate",
        "future_fresh_275_refits_new_programs": True,
        "pilot_program_reuse_for_admission": False,
        "fresh_precheck_failure_may_switch_gamma": False,
        "selected_model_method_lock_schema": "warehouse-v620-model-method-lock.v1",
        "native_selection_receipt": "selected-method.receipt.json_after_full_chain_revalidation",
        "selected_lock_grants_fresh_seed_permission": False,
        "actor_updates": 0, "program_gradient_updates": 0, "PPO_joint_steps": 0,
        "fresh_collection_authority": False, "publishable": False,
        "admission_evidence": False, "fresh_validation_evidence": False,
    }


def _score_component(value):
    if value is None:
        return -math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("nonfinite CV selection score")
    return float(value)


def choose_method(summary):
    """Development method only; absence of a passing arm does not authorize NN changes."""
    if (summary.get("chosen_method_arm") is not None or
            summary.get("next_stage") != "common_bc_structure_learning_before_fresh_validation"):
        raise RuntimeError("pilot requires the complete failed frozen-NN extraction route")
    scores = {}
    for arm in ("M1", "M2"):
        score = summary["arms"][arm]["selection_score"]
        if not isinstance(score, list) or len(score) != 4:
            raise ValueError("full CV selection score must contain four preregistered terms")
        scores[arm] = tuple(_score_component(value) for value in score)
    # An explicit final term avoids depending on dict iteration order.
    return max(("M1", "M2"), key=lambda arm: (scores[arm], arm == "M1"))


def quality_summary(episodes):
    if len(episodes) != 24:
        raise RuntimeError("pilot quality must use exactly the same 24 validation trajectories")
    for row in episodes:
        if (type(row.get("deliveries")) is not int or row["deliveries"] < 0
                or type(row.get("collisions")) is not int or row["collisions"] < 0
                or type(row.get("shutdown_episode")) is not bool):
            raise ValueError("invalid real episode quality outcome")
    value = {"mean_deliveries": sum(r["deliveries"] for r in episodes) / 24,
        "mean_collisions": sum(r["collisions"] for r in episodes) / 24,
        "shutdown_episode_rate": sum(r["shutdown_episode"] for r in episodes) / 24}
    value["quality_score"] = (value["mean_deliveries"] - .25 * value["mean_collisions"]
                                - 5 * value["shutdown_episode_rate"])
    value["quality_passed"] = (value["mean_deliveries"] >= 5.5 and value["mean_collisions"] <= 6
                                and value["shutdown_episode_rate"] <= .05)
    return value


def _bounded_number(value, *, upper=None, nullable=False):
    if value is None and nullable:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or upper is not None and value > upper):
        raise ValueError("invalid pilot program metric")
    return float(value)


def program_profile(role_metrics):
    if set(role_metrics) != set(ROLES):
        raise ValueError("pilot requires both independent role program profiles")
    coverage, expected, kls, ratios, checks = [], [], [], [], []
    for role in ROLES:
        metrics = role_metrics[role]
        coverage.append(_bounded_number(metrics["explanation"]["coverage"], upper=1))
        expected.append(_bounded_number(metrics["feedback"]["expected_actionable_coverage"], upper=1))
        for purpose, minimum in (("feedback", .93), ("explanation", .95)):
            item = metrics[purpose]
            agreement = _bounded_number(item.get("agreement"), upper=1, nullable=True)
            kl = _bounded_number(item.get("kl"), nullable=True)
            counts = [item.get(name) for name in
                      ("whole_tree_raw_illegal_count", "whole_tree_path_uncertified_count")]
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError("whole-tree raw legality/path counts are required")
            checks.append(agreement is not None and agreement >= minimum and kl is not None
                          and kl <= .25 and counts == [0, 0])
            ratios.extend([0. if agreement is None else agreement / minimum,
                           0. if kl is None else .25 / max(kl, 1e-12)])
            kls.append(math.inf if kl is None else kl)
    maximum_kl = max(kls)
    return {"program_consistency_pass": all(checks),
        "minimum_normalized_consistency_margin": min(ratios),
        "minimum_role_explanation_coverage": min(coverage),
        "minimum_role_expected_feedback": min(expected),
        "maximum_qualified_kl": None if math.isinf(maximum_kl) else maximum_kl}


def choose_gamma(results):
    """The sole hard eligibility is quality; consistency is a fixed preference."""
    if set(results) != set(GAMMAS):
        raise RuntimeError("both gamma comparisons must complete before selection")
    eligible = [name for name in GAMMAS if results[name]["quality"]["quality_passed"]]
    if not eligible:
        return None, "closed_both_quality_failed"
    all_inconsistent = not any(results[name]["program_profile"]["program_consistency_pass"] for name in eligible)

    def score(name):
        value, quality = results[name]["program_profile"], results[name]["quality"]
        prefix = (value["minimum_normalized_consistency_margin"],) if all_inconsistent else (
            int(value["program_consistency_pass"]),)
        return prefix + (value["minimum_role_explanation_coverage"], value["minimum_role_expected_feedback"],
            -math.inf if value["maximum_qualified_kl"] is None else -value["maximum_qualified_kl"],
            quality["quality_score"], -GAMMAS[name])

    return max(eligible, key=score), ("consistency_margin_fallback" if all_inconsistent else "consistency_then_coverage")


def _role_hashes(value):
    if not isinstance(value, dict) or set(value) != set(ROLES):
        raise ValueError("two role hashes required")
    if any(not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
           for x in value.values()):
        raise ValueError("malformed role hash")
    return value


def _bc_run(runtime, comparison_spec, gamma_name, cv_bound):
    """Recheck completed BC evidence without reopening its training capability."""
    root, contract = runtime.root, runtime.bc_contract
    if tuple(contract.STAGE_COUNTS) != BC_STAGE_COUNTS:
        raise RuntimeError("shared BC stage contract changed")
    pointers = comparison_spec["bc_runs"][gamma_name]
    if set(pointers) != {"spec", "authorization", "receipt", "report"}:
        raise ValueError("completed BC needs spec, authorization, receipt and report")
    loaded, files = {}, []
    for name in ("spec", "authorization", "receipt", "report"):
        path, data = read_pointer(root, pointers[name])
        loaded[name] = (path, data)
        files.append((str(path), file_sha(path)))
    spec_path, spec = loaded["spec"]
    authorization_path, authorization = loaded["authorization"]
    receipt_path, receipt = loaded["receipt"]
    report_path, report = loaded["report"]
    gamma = GAMMAS[gamma_name]
    run_id = "pair-6160-" + gamma_name
    bc_directory = (root / "output/warehouse_v620_bc_pilot" / contract.CANDIDATE_ID / run_id).resolve()
    if (spec.get("schema") != contract.SCHEMA or spec.get("candidate_id") != contract.CANDIDATE_ID
            or spec.get("pair_seed") != 6160 or spec.get("gamma") != gamma
            or spec.get("run_id") != run_id or spec.get("output_directory") != str(bc_directory)
            or spec.get("source_closure_sha256") != comparison_spec["bc_source_closure_sha256"]
            or spec.get("cv_summary") != comparison_spec["cv_summary"]
            or receipt_path != bc_directory / "bc-pilot.receipt.json"
            or report_path != bc_directory / "bc-pilot.training.json"):
        raise RuntimeError("completed BC pilot identity/source/path mismatch")
    schedule = spec["bc_seed_schedule"]
    if set(schedule) != {name for name, _ in BC_STAGE_COUNTS}:
        raise RuntimeError("completed BC stage purposes differ")
    expected, data_schedule = [], {}
    for name, count in BC_STAGE_COUNTS:
        cases = schedule[name]
        if (len(cases) != count or any(set(case) != {"scenario_seed", "action_seed"}
                or any(type(case[k]) is not int or case[k] < 0 for k in case) for case in cases)):
            raise ValueError("completed BC scenario/action schedule malformed")
        data_schedule[name] = [case["scenario_seed"] for case in cases]
        expected.extend({"phase": "formal_bc", "seed": scene} for scene in data_schedule[name])
        if name == "joint_augmentation_argmax":
            expected.extend({"phase": "formal_bc", "seed": spec["critic_seed"]} for _ in range(2))
    if len(expected) != 1272 or len({row["seed"] for row in expected}) != 1271:
        raise RuntimeError("completed BC data/critic reset budget or scene separation mismatch")
    recipe_path, bc_recipe = read_pointer(root, spec["recipe"])
    paired_path, paired = read_pointer(root, spec["paired_seed_plan"])
    clearance_path, clearance = read_pointer(root, spec["seed_clearance"])
    if bc_recipe != contract.recipe(gamma):
        raise RuntimeError("completed BC recipe differs from registered shared recipe")
    binding = contract.seed_binding(schedule, spec["critic_seed"])
    if (paired != {"schema": "warehouse-v620-paired-bc-seeds.v1",
            "candidate_id": contract.CANDIDATE_ID, "pair_seed": 6160,
            "gamma_candidates": [.01, .03], "seed_binding": binding}
            or clearance.get("approved_for_shared_bc") is not True
            or clearance.get("seed_binding_sha256") != digest(binding)
            or clearance.get("conflict_count") != 0):
        raise RuntimeError("completed BC paired schedule/clearance mismatch")
    initial_roles = _role_hashes(runtime.initial_role_hashes(6160))
    main_path = (root / "scripts/run_warehouse_v620_shared_bc_pilot.py").resolve(strict=True)
    bc_bound = tuple((str(path), file_sha(path)) for path in
                     (main_path, recipe_path, paired_path, clearance_path)) + tuple(cv_bound)
    required = {
        "authorization_kind": "warehouse-v620-shared-bc-pilot-only-v1",
        "candidate_id": contract.CANDIDATE_ID, "pair_seed": 6160,
        "gamma": gamma, "run_id": run_id, "original_spec_sha256": digest(spec),
        "source_closure_sha256": comparison_spec["bc_source_closure_sha256"],
        "recipe_file_sha256": file_sha(recipe_path), "bound_files_sha256": digest(bc_bound),
        "initial_role_sha256": initial_roles, "expected_reset_sequence_sha256": digest(expected),
        "output_directory": str(bc_directory), "action_physics_config_sha256": runtime.physics_sha256,
        "PPO_joint_steps": 0, "publishable": False, "fresh_validation_episodes": 0,
    }
    if authorization != required or any(receipt.get(k) != v for k, v in required.items()):
        raise RuntimeError("completed BC exact authorization identity mismatch")
    journal_path = bc_directory / "bc-pilot.resets.jsonl"
    journal_raw = journal_path.read_bytes()
    journal_rows = [json.loads(line) for line in journal_raw.splitlines()]
    projection = [{"phase": row["phase"], "seed": row["seed"]} for row in journal_rows]
    if (receipt.get("status") != "bc_completed_not_admitted" or receipt.get("error") is not None
            or receipt.get("authorization_unchanged") is not True
            or receipt.get("authorization_receipt_sha256") != file_sha(authorization_path)
            or receipt.get("bc_report_sha256") != file_sha(report_path)
            or receipt.get("actual_reset_count") != 1272 or receipt.get("expected_reset_count") != 1272
            or receipt.get("reset_full_sequence_matches") is not True
            or receipt.get("reset_prefix_matches") is not True
            or receipt.get("journal_path") != str(journal_path)
            or receipt.get("journal_sha256") != sha256(journal_raw).hexdigest()
            or receipt.get("actual_resets") != journal_rows or projection != expected):
        raise RuntimeError("completed BC reset journal/receipt chain mismatch")
    final_roles = _role_hashes(report["pretrained_role_sha256"])
    artifact = report["final_bc_actor"]
    actor_path, _ = read_pointer(root, {"path": artifact["path"], "sha256": artifact["sha256"]}, json_object=False)
    if (actor_path != bc_directory / "final-bc.actor.npz" or artifact.get("role_sha256") != final_roles
            or report.get("status") != "bc_completed_not_admitted"
            or report.get("initial_role_sha256") != initial_roles
            or any(final_roles[role] == initial_roles[role] for role in ROLES)
            or report.get("source_closure_sha256") != comparison_spec["bc_source_closure_sha256"]
            or report.get("bc_recipe") != bc_recipe["alignment"]
            or report.get("executable_recipe_file_sha256") != file_sha(recipe_path)
            or report.get("PPO_joint_steps") != 0 or report.get("publishable") is not False
            or report.get("fresh_validation_episodes") != 0
            or report.get("mid_bc_extracted_tree_installed") is not False
            or report.get("external_program_feedback") != 0
            or report.get("bc_data_seed_blocks") != data_schedule
            or report.get("bc_data_seed_schedule_sha256") != digest(data_schedule)
            or report.get("expected_bc_training_episodes") != 1270
            or report.get("expected_critic_environment_resets") != 2
            or report.get("collision_critic", {}).get("validation_passed") is not True):
        raise RuntimeError("completed BC snapshot/shared training provenance mismatch")
    files.extend(bc_bound)
    files.extend([(str(journal_path), file_sha(journal_path)), (str(actor_path), file_sha(actor_path))])
    return {"gamma": gamma, "initial_role_sha256": initial_roles,
        "paired_seed_plan": pointer(paired_path), "seed_binding_sha256": digest(binding),
        "actor": {"path": str(actor_path), "sha256": artifact["sha256"], "role_sha256": final_roles},
        "bc_report": pointer(report_path), "bc_receipt": pointer(receipt_path),
        "bc_reset_sequence_sha256": digest(expected)}, files


def verify_inputs(spec, runtime):
    keys = {"schema", "candidate_id", "source_closure_sha256", "bc_source_closure_sha256",
            "cv_summary", "bc_runs", "output_directory"}
    if (not isinstance(spec, dict) or set(spec) != keys or spec["schema"] != SCHEMA
            or spec["candidate_id"] != CANDIDATE_ID or set(spec["bc_runs"]) != set(GAMMAS)
            or runtime.source_sha256() != spec["source_closure_sha256"]):
        raise RuntimeError("pilot comparison spec/source identity mismatch")
    output = (runtime.root / "output/warehouse_v620_bc_pilot/comparison-01").resolve()
    if Path(spec["output_directory"]).resolve() != output:
        raise RuntimeError("pilot comparison output is not the registered exclusive directory")
    proof = runtime.native_cv_proof(spec["cv_summary"])
    method_arm = choose_method(proof["summary"])
    bound = list(proof["bound_files"])
    bc = {}
    for name in GAMMAS:
        bc[name], files = _bc_run(runtime, spec, name, proof["bound_files"])
        bound.extend(files)
    if (bc["gamma-001"]["initial_role_sha256"] != bc["gamma-003"]["initial_role_sha256"]
            or bc["gamma-001"]["paired_seed_plan"] != bc["gamma-003"]["paired_seed_plan"]
            or bc["gamma-001"]["seed_binding_sha256"] != bc["gamma-003"]["seed_binding_sha256"]
            or bc["gamma-001"]["bc_reset_sequence_sha256"] != bc["gamma-003"]["bc_reset_sequence_sha256"]):
        raise RuntimeError("gamma recipes do not share initialization and paired training plan")
    script_path = runtime.script_path.resolve(strict=True)
    if script_path != (runtime.root / "scripts/run_warehouse_v620_bc_pilot_comparison.py").resolve():
        raise RuntimeError("pilot comparison requires its registered saved entry script")
    bound.append((str(script_path), file_sha(script_path)))
    recipe_path = (runtime.root / RECIPE_PATH).resolve(strict=True)
    if json.loads(recipe_path.read_bytes()) != recipe():
        raise RuntimeError("pilot executable recipe differs from its canonical source contract")
    bound.append((str(recipe_path), file_sha(recipe_path)))
    cases = runtime.spent_cases()
    if ({split: len(cases[split]) for split in SPLITS} !=
            {"fit": 50, "calibration": 25, "eligibility": 25, "validation": 24}
            or sum(c["sampling_mode"] == "argmax" for c in cases["fit"]) != 25
            or any(c["sampling_mode"] not in ("sampled", "argmax") for c in cases["fit"])
            or any(c["sampling_mode"] != "sampled" for split in SPLITS[1:] for c in cases[split])
            or len({c["scenario_seed"] for split in SPLITS for c in cases[split]}) != 124):
        raise RuntimeError("pilot must exactly reuse the spent v6.19 scene/mode contract")
    return {"method_arm": method_arm, "method": runtime.methods[method_arm], "bc_runs": bc,
        "bound_files": sorted(set(bound)), "cases": cases,
        "cv_selection_scores": {arm: proof["summary"]["arms"][arm]["selection_score"] for arm in ("M1", "M2")}}, output


def _method_lock(spec, inputs):
    return {"schema": "warehouse-v620-bc-pilot-method-lock.v1", "candidate_id": CANDIDATE_ID,
        "original_spec_sha256": digest(spec), "source_closure_sha256": spec["source_closure_sha256"],
        "cv_summary": spec["cv_summary"], "cv_selection_scores": inputs["cv_selection_scores"],
        "arm": inputs["method_arm"], "method": inputs["method"],
        "development_method_only": True, "method_locked_before_any_pilot_rollout": True,
        "tie_rule": "M1", "scene_contract_sha256": digest(inputs["cases"]),
        "bc_runs": inputs["bc_runs"], "publishable": False, "fresh_collection_authority": False}


def prepare(spec, runtime):
    inputs, output = verify_inputs(spec, runtime)
    output.mkdir(parents=True, exist_ok=False)
    lock_path = output / "method.lock.json"
    write_new(lock_path, _method_lock(spec, inputs))
    plan = {"schema": PLAN_SCHEMA, "candidate_id": CANDIDATE_ID,
        "original_spec": spec, "recipe": recipe(), "inputs": inputs,
        "method_lock": pointer(lock_path), "output_directory": str(output),
        "publishable": False, "fresh_collection_authority": False,
        "fresh_validation_evidence": False, "admission_evidence": False,
        "PPO_joint_steps": 0, "actor_updates": 0}
    write_new(output / "plan.json", plan)
    return output / "plan.json"


def checked_plan(plan_path, runtime):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_bytes())
    inputs, output = verify_inputs(plan["original_spec"], runtime)
    lock_path, lock = read_pointer(runtime.root, plan["method_lock"])
    if (plan_path != output / "plan.json" or lock_path != output / "method.lock.json"
            or plan.get("schema") != PLAN_SCHEMA or plan.get("inputs") != json.loads(json.dumps(inputs))
            or plan.get("recipe") != recipe() or lock != _method_lock(plan["original_spec"], inputs)
            or any(plan.get(k) is not False for k in
                ("publishable", "fresh_collection_authority", "fresh_validation_evidence", "admission_evidence"))
            or plan.get("PPO_joint_steps") != 0 or plan.get("actor_updates") != 0):
        raise RuntimeError("pilot immutable plan/method lock mismatch")
    return plan_path, plan, output


def _reset_audit(audit, path, seeds, *, phase):
    if (not isinstance(audit, dict) or audit.get("journal_path") != str(path.resolve())
            or audit.get("journal_sha256") != file_sha(path)
            or audit.get("actual_reset_count") != len(seeds)):
        raise RuntimeError("persistent pilot reset journal identity mismatch")
    raw_rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if (raw_rows != audit.get("resets")
            or [{"phase": row["phase"], "seed": row["seed"]} for row in raw_rows]
                != [{"phase": phase, "seed": seed} for seed in seeds]):
        raise RuntimeError("persistent pilot reset sequence differs from locked cases")
    return pointer(path)


def _check_episode(rows, outcome, case, artifact):
    if set(rows) != set(ROLES) or len(rows[ROLES[0]]) != len(rows[ROLES[1]]) or not rows[ROLES[0]]:
        raise RuntimeError("pilot omitted a role or an entire physical trajectory")
    count = len(rows[ROLES[0]])
    if (outcome.get("seed") != case["scenario_seed"] or outcome.get("action_seed") != case["scenario_seed"]
            or outcome.get("sampling_mode") != case["sampling_mode"] or outcome.get("frame_count") != count):
        raise RuntimeError("pilot sampled outcome differs from locked scene/action/mode")
    for role in ROLES:
        for frame, row in enumerate(rows[role]):
            if (row["frame"] != frame or row["base_seed"] != case["scenario_seed"]
                    or row["actor_sha256"] != artifact["sha256"]
                    or row["actor_role_sha256"] != artifact["role_sha256"][role]
                    or row.get("nn_sampled_action") != row["sampled_action"]):
                raise RuntimeError("pilot physical frame/action/Actor binding mismatch")


def run_gamma(plan_path, gamma_name, runtime):
    if gamma_name not in GAMMAS:
        raise ValueError("only the two preregistered gamma candidates are permitted")
    plan_path, plan, output = checked_plan(plan_path, runtime)
    directory = output / gamma_name
    directory.mkdir(exist_ok=False)
    artifact = plan["inputs"]["bc_runs"][gamma_name]["actor"]
    method = plan["inputs"]["method"]
    plan_sha, lock_sha = file_sha(plan_path), plan["method_lock"]["sha256"]
    try:
        actor = runtime.load_actor(artifact)
        identity = runtime.actor_identity(actor)
        if identity != artifact["role_sha256"]:
            raise RuntimeError("frozen pilot Actor role identity mismatch")
    except BaseException as exc:
        write_new(directory / "failure.json", {
            "schema": "warehouse-v620-bc-pilot-gamma-failure.v1",
            "status": "failed_or_interrupted", "error_type": type(exc).__name__, "error": str(exc),
            "plan_sha256": plan_sha, "gamma_name": gamma_name,
            "completed_collection_episodes": 0, "attempted_persistent_journals": [],
            "collector_failure_reset_audit": None, "scientific_retry_permitted": False,
            "publishable": False, "fresh_collection_authority": False, "PPO_joint_steps": 0})
        raise
    write_new(directory / "started.json", {"schema": "warehouse-v620-bc-pilot-gamma-start.v1",
        "plan_sha256": plan_sha, "method_lock_sha256": lock_sha, "gamma": GAMMAS[gamma_name],
        "actor": artifact, "source_closure_sha256": plan["original_spec"]["source_closure_sha256"],
        "scene_reuse": "matched_spent_v619_development_not_fresh_evidence",
        "fresh_collection_authority": False, "publishable": False})
    collection, attempted_journals, quality_episodes, role_metrics = [], [], [], {}
    parts = {split: {role: [] for role in ROLES} for split in SPLITS}

    def guard(*, all_bound=False):
        if (runtime.source_sha256() != plan["original_spec"]["source_closure_sha256"]
                or file_sha(plan_path) != plan_sha
                or file_sha(plan["method_lock"]["path"]) != lock_sha
                or file_sha(artifact["path"]) != artifact["sha256"]
                or runtime.actor_identity(actor) != identity):
            raise RuntimeError("pilot source, method lock, plan or frozen NN changed")
        if all_bound:
            for path, expected in plan["inputs"]["bound_files"]:
                if file_sha(path) != expected:
                    raise RuntimeError("pilot prerequisite evidence changed")

    try:
        guard(all_bound=True)
        for split in SPLITS:
            for index, case in enumerate(plan["inputs"]["cases"][split]):
                guard()
                case_dir = directory / "collection" / split / f"episode-{index:03d}"
                case_dir.mkdir(parents=True, exist_ok=False)
                journal = (case_dir / "resets.jsonl").resolve()
                attempted_journals.append(str(journal))
                rows, evidence = runtime.collect(actor, case, journal_path=journal)
                audit_ptr = _reset_audit(evidence["reset_audit"], journal,
                                         [case["scenario_seed"]], phase="program")
                outcomes = evidence.get("episode_outcomes")
                if not isinstance(outcomes, list) or len(outcomes) != 1:
                    raise RuntimeError("pilot collector omitted its actual episode outcome")
                _check_episode(rows, outcomes[0], case, artifact)
                trace = runtime.write_trace(rows, split, [case["scenario_seed"]], case_dir / "trace", artifact)
                if trace.get("row_count") != len(rows[ROLES[0]]):
                    raise RuntimeError("pilot full-frame trace inventory differs")
                read_pointer(runtime.root, {"path": trace["path"], "sha256": trace["sha256"]}, json_object=False)
                record = {"split": split, "case": case, "trace": trace, "reset_journal": audit_ptr,
                          "outcome": outcomes[0], "excluded_frames": 0, "authorization": evidence}
                write_new(case_dir / "collection.json", record)
                collection.append(record)
                for role in ROLES:
                    parts[split][role].extend(rows[role])
                if split == "validation":
                    quality_episodes.append(runtime.quality_row(rows, outcomes[0], actor))
                guard()
        quality = quality_summary(quality_episodes)
        write_new(directory / "quality-episodes.json", quality_episodes)
        replay_journal = (directory / "quality-replay.resets.jsonl").resolve()
        attempted_journals.append(str(replay_journal))
        replay = runtime.replay(artifact, quality_episodes,
            [(case["scenario_seed"], case["scenario_seed"]) for case in plan["inputs"]["cases"]["validation"]],
            journal_path=replay_journal)
        _reset_audit(replay["reset_audit"], replay_journal,
            [case["scenario_seed"] for case in plan["inputs"]["cases"]["validation"]], phase="quality")
        if (replay["receipt"].get("passed") is not True
                or replay["receipt"].get("episode_count") != 24
                or replay["receipt"].get("actor_sha256") != artifact["sha256"]
                or replay["receipt"].get("actor_role_sha256") != identity
                or any(replay["receipt"].get("measured", {}).get(key) != quality[key] for key in
                       ("mean_deliveries", "mean_collisions", "shutdown_episode_rate"))):
            raise RuntimeError("pilot same-trajectory quality physical replay disagrees")
        write_new(directory / "quality-replay.json", replay)
        programs = {}
        for role in ROLES:
            guard()
            program_path = directory / "programs" / f"{role}.program.json"
            fitted = runtime.fit_program(role, actor, {s: parts[s][role] for s in SPLITS},
                                         method, plan["inputs"]["method_arm"], program_path)
            if fitted.get("search_budget_audit", {}).get("passed") is not True:
                raise RuntimeError("pilot external fit violated its locked search budget")
            program_pointer = fitted["program"]
            read_pointer(runtime.root, program_pointer, json_object=False)
            if Path(program_pointer["path"]).resolve() != program_path.resolve():
                raise RuntimeError("pilot fitted program escaped its role output")
            runtime.validate_program_record(artifact, role, fitted, method, plan["inputs"]["method_arm"])
            role_metrics[role] = fitted["metrics"]
            programs[role] = fitted
            guard()
        profile = program_profile(role_metrics)
        guard(all_bound=True)
        result = {"schema": "warehouse-v620-bc-pilot-gamma-result.v1", "gamma": GAMMAS[gamma_name],
            "gamma_name": gamma_name, "plan_sha256": plan_sha, "method_lock_sha256": lock_sha,
            "method": method, "actor": artifact, "actor_role_sha256_unchanged": identity,
            "source_closure_sha256": plan["original_spec"]["source_closure_sha256"],
            "bc_source_closure_sha256": plan["original_spec"]["bc_source_closure_sha256"],
            "quality": quality, "quality_episodes": pointer(directory / "quality-episodes.json"),
            "quality_replay": pointer(directory / "quality-replay.json"),
            "programs": programs, "role_metrics": role_metrics, "program_profile": profile,
            "collection": collection, "collection_episode_count": 124,
            "quality_statistical_episode_count": 24, "replay_episode_count": 24,
            "environment_reset_count": 148, "excluded_frames": 0,
            "scene_reuse": "matched_spent_v619_development_not_fresh_evidence",
            "admission_evidence": False, "fresh_validation_evidence": False,
            "fresh_collection_authority": False, "publishable": False,
            "actor_updates": 0, "program_gradient_updates": 0, "PPO_joint_steps": 0}
        write_new(directory / "result.json", result)
        write_new(directory / "completed.json", {"result": pointer(directory / "result.json"),
            "plan_sha256": plan_sha, "method_lock_sha256": lock_sha, "status": "pilot_development_completed"})
        return pointer(directory / "result.json")
    except BaseException as exc:
        write_new(directory / "failure.json", {"schema": "warehouse-v620-bc-pilot-gamma-failure.v1",
            "status": "failed_or_interrupted", "error_type": type(exc).__name__, "error": str(exc),
            "plan_sha256": plan_sha, "gamma_name": gamma_name,
            "completed_collection_episodes": len(collection),
            "attempted_persistent_journals": [{"path": path,
                "sha256": file_sha(path) if Path(path).is_file() else None} for path in attempted_journals],
            "collector_failure_reset_audit": getattr(exc, "warehouse_reset_audit", None),
            "scientific_retry_permitted": False, "publishable": False,
            "fresh_collection_authority": False, "PPO_joint_steps": 0})
        raise


def select(plan_path, runtime):
    plan_path, plan, output = checked_plan(plan_path, runtime)
    if (output / "gamma.lock.json").exists() or (output / "closed.json").exists():
        raise RuntimeError("pilot selection was already locked or candidate closed")
    results, references = {}, {}
    for name in GAMMAS:
        completed_path = output / name / "completed.json"
        completed = json.loads(completed_path.read_bytes())
        result_path, result = read_pointer(runtime.root, completed["result"])
        if (result_path != output / name / "result.json" or (output / name / "failure.json").exists()
                or completed.get("status") != "pilot_development_completed"
                or completed.get("plan_sha256") != file_sha(plan_path)
                or completed.get("method_lock_sha256") != plan["method_lock"]["sha256"]
                or result.get("gamma") != GAMMAS[name] or result.get("gamma_name") != name
                or result.get("plan_sha256") != file_sha(plan_path)
                or result.get("method_lock_sha256") != plan["method_lock"]["sha256"]
                or result.get("actor") != plan["inputs"]["bc_runs"][name]["actor"]
                or result.get("method") != plan["inputs"]["method"]
                or result.get("source_closure_sha256") != plan["original_spec"]["source_closure_sha256"]
                or result.get("bc_source_closure_sha256") != plan["original_spec"]["bc_source_closure_sha256"]
                or result.get("actor_role_sha256_unchanged") != result["actor"]["role_sha256"]
                or result.get("collection_episode_count") != 124 or result.get("environment_reset_count") != 148
                or result.get("quality_statistical_episode_count") != 24 or result.get("replay_episode_count") != 24
                or result.get("excluded_frames") != 0
                or result.get("scene_reuse") != "matched_spent_v619_development_not_fresh_evidence"
                or any(result.get(k) is not False for k in
                    ("admission_evidence", "fresh_validation_evidence", "fresh_collection_authority", "publishable"))
                or any(result.get(k) != 0 for k in ("actor_updates", "program_gradient_updates", "PPO_joint_steps"))):
            raise RuntimeError("completed gamma result identity/provenance differs")
        quality_path, _ = read_pointer(runtime.root, result["quality_episodes"], json_object=False)
        episodes = json.loads(quality_path.read_bytes())
        replay_path, replay = read_pointer(runtime.root, result["quality_replay"])
        quality, profile = quality_summary(episodes), program_profile(result["role_metrics"])
        validation_cases = plan["inputs"]["cases"]["validation"]
        seeds = [(case["scenario_seed"], case["scenario_seed"]) for case in validation_cases]
        runtime.validate_replay_claim(result["actor"], episodes, seeds, replay["receipt"])
        _reset_audit(replay["reset_audit"], output / name / "quality-replay.resets.jsonl",
            [case["scenario_seed"] for case in validation_cases], phase="quality")
        if (quality != result["quality"] or profile != result["program_profile"]
                or replay["receipt"].get("passed") is not True
                or replay["receipt"].get("episode_count") != 24
                or any(replay["receipt"].get("measured", {}).get(k) != quality[k] for k in
                       ("mean_deliveries", "mean_collisions", "shutdown_episode_rate"))):
            raise RuntimeError("saved pilot quality/profile evidence does not recompute")
        expected_cases = [(split, index, case) for split in SPLITS
                          for index, case in enumerate(plan["inputs"]["cases"][split])]
        if len(result["collection"]) != len(expected_cases):
            raise RuntimeError("pilot collection inventory is incomplete")
        for record, (split, index, case) in zip(result["collection"], expected_cases, strict=True):
            case_dir = output / name / "collection" / split / f"episode-{index:03d}"
            if (record.get("split") != split or record.get("case") != case
                    or record.get("excluded_frames") != 0
                    or record["outcome"].get("seed") != case["scenario_seed"]
                    or record["outcome"].get("action_seed") != case["scenario_seed"]
                    or record["outcome"].get("sampling_mode") != case["sampling_mode"]):
                raise RuntimeError("pilot reuse/physical-frame inventory differs from locked cases")
            _reset_audit(record["authorization"]["reset_audit"], case_dir / "resets.jsonl",
                         [case["scenario_seed"]], phase="program")
            trace_path, _ = read_pointer(runtime.root, {k: record["trace"][k] for k in ("path", "sha256")},
                                         json_object=False)
            if (not trace_path.is_relative_to(case_dir / "trace")
                    or len(trace_path.read_bytes().splitlines()) != record["trace"]["row_count"]
                    or record["trace"]["row_count"] != record["outcome"]["frame_count"]):
                raise RuntimeError("pilot saved full-frame trace is incomplete")
        for row, record in zip(episodes, [r for r in result["collection"] if r["split"] == "validation"], strict=True):
            if any(row[k] != record["outcome"][k] for k in ("deliveries", "collisions", "shutdown_episode")):
                raise RuntimeError("pilot quality used outcomes outside the same validation trajectories")
        if set(result["programs"]) != set(ROLES):
            raise RuntimeError("saved pilot omitted an independent role program")
        for role, fitted in result["programs"].items():
            read_pointer(runtime.root, fitted["program"], json_object=False)
            if Path(fitted["program"]["path"]).resolve() != output / name / "programs" / f"{role}.program.json":
                raise RuntimeError("saved pilot program escaped its locked role output")
            runtime.validate_program_record(result["actor"], role, fitted, plan["inputs"]["method"],
                                            plan["inputs"]["method_arm"])
            if fitted["metrics"] != result["role_metrics"][role]:
                raise RuntimeError("pilot role metrics differ from their saved fitted-program record")
        results[name] = result
        references[name] = pointer(result_path)
    # Selection can emit downstream evidence only after its entire saved chain
    # and current source are checked again. This performs no new rollout/fit.
    checked_plan(plan_path, runtime)
    for reference in references.values():
        read_pointer(runtime.root, reference)
    winner, mode = choose_gamma(results)
    common = {"schema": "warehouse-v620-bc-pilot-selection.v1", "plan": pointer(plan_path),
        "method_lock": plan["method_lock"], "method": plan["inputs"]["method"],
        "method_arm": plan["inputs"]["method_arm"], "gamma_results": references,
        "ranking_mode": mode, "recipe": recipe(), "pilot_development_only": True,
        "scene_reuse": "matched_spent_v619_development_not_fresh_evidence",
        "publishable": False, "admission_evidence": False, "fresh_validation_evidence": False,
        "fresh_collection_authority": False, "PPO_joint_steps": 0,
        "pilot_program_reuse_for_admission": False, "fresh_precheck_failure_may_switch_gamma": False}
    if winner is None:
        target = output / "closed.json"
        write_new(target, {**common, "status": "candidate_closed_both_gamma_quality_failed", "selected_gamma": None})
    else:
        target = output / "gamma.lock.json"
        write_new(target, {**common, "status": "gamma_and_frozen_actor_locked_for_new_fresh_precheck",
            "selected_gamma_name": winner, "selected_gamma": GAMMAS[winner],
            "actor": results[winner]["actor"], "actor_role_sha256": results[winner]["actor"]["role_sha256"],
            "fresh_precheck_requires_new_275_and_refitted_programs": True})
        recipe_pointer = pointer(runtime.root / RECIPE_PATH)
        _, chosen_bc_spec = read_pointer(runtime.root, plan["original_spec"]["bc_runs"][winner]["spec"])
        receipt_path = output / "selected-method.receipt.json"
        selected_receipt = {
            "schema": "warehouse-v620-selected-method-receipt.v1",
            "candidate_id": CANDIDATE_ID, "locked": True,
            "program_method": plan["inputs"]["method"], "method_arm": plan["inputs"]["method_arm"],
            "actor_sha256": results[winner]["actor"]["sha256"],
            "actor_role_sha256": results[winner]["actor"]["role_sha256"],
            "gamma_lock": pointer(target), "selected_gamma": GAMMAS[winner],
            "source_closure_sha256": plan["original_spec"]["source_closure_sha256"],
            "bc_source_closure_sha256": plan["original_spec"]["bc_source_closure_sha256"],
            "recipe": recipe_pointer, "bc_recipe": chosen_bc_spec["recipe"],
            "cv_summary": plan["original_spec"]["cv_summary"], "method_lock": plan["method_lock"],
            "comparison_plan": pointer(plan_path), "selected_gamma_result": references[winner],
            "pilot_development_only": True, "admission_evidence": False,
            "fresh_collection_authority": False, "publishable": False, "PPO_joint_steps": 0}
        write_new(receipt_path, selected_receipt)
        write_new(output / "model-method.lock.json", {
            "schema": "warehouse-v620-model-method-lock.v1",
            "candidate_id": CANDIDATE_ID, "locked": True,
            "source_closure_sha256": plan["original_spec"]["source_closure_sha256"],
            "bc_source_closure_sha256": plan["original_spec"]["bc_source_closure_sha256"],
            "program_method": plan["inputs"]["method"], "actor": results[winner]["actor"],
            "actor_role_sha256": results[winner]["actor"]["role_sha256"],
            "recipe": recipe_pointer, "bc_recipe": chosen_bc_spec["recipe"],
            "method_selection_receipt": pointer(receipt_path), "gamma_lock": pointer(target),
            "pilot_development_only": True, "admission_evidence": False,
            "fresh_collection_authority": False, "publishable": False, "PPO_joint_steps": 0})
    return pointer(target)
