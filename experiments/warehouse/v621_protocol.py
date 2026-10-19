"""Source-locked diagnostics on the already spent v6.20 275-episode cohort.

This module never creates an environment, collects data, updates an Actor, or
opens a downstream stage. Its strict byte bindings distinguish reuse for
method development from independent admission evidence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

CANDIDATE_ID = "warehouse-v621-shared-structure-01"
DATA_ROOT = "output/warehouse_v620_frozen/warehouse-v620-01-gamma003-precheck-01"
ACTOR_PATH = "output/warehouse_v620_bc_pilot/warehouse-v620-01-shared-bc-pilot/pair-6160-gamma-003/final-bc.actor.npz"
ACTOR_SHA256 = "1b6310f6d93309fb17e64516519a71680ad3c30e73f222be27df4dff10da81ec"
ROLE_SHA256 = {
    "robot_1": "7aee44859397b8cbe1bb9cd3b8541e82f2a080d884c83404fe1e4fd2a998eaf0",
    "robot_2": "17f2f75877421854c680fab07557f031389800a9503b14f060db5ff4f862fcf0",
}
LEGACY_SOURCE_SHA256 = "4fba3c172feeaefa11e4c0c2b435e21457f4f27c60c11e5b2cd0c24c618f229c"
OUTPUT_ROOT = "output/warehouse_v621"
BATCH_ID = "diagnostics-01"
RECIPE_PATH = "configs/warehouse_v621_recipe.json"
ROLES = ("robot_1", "robot_2")
SPLITS = ("fit", "calibration", "eligibility", "validation")
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
METHODS = {
    "A": "v620_m2_same_fold_refit_v1",
    "B": "actor_topology_compilation_distribution_refit_v1",
    "C": "path_constrained_alternating_tree_optimization_v1",
}
FOLD_SEED = "warehouse-v621-spent-v620-whole-base-cv.v1"
GATES = {
    "minimum_feedback_coverage": .45, "minimum_explanation_coverage": .45,
    "minimum_expected_feedback_coverage": .40,
    "minimum_feedback_agreement": .95, "minimum_explanation_agreement": .97,
    "maximum_kl": .20, "whole_tree_raw_illegal_count": 0,
    "whole_tree_path_uncertified_count": 0,
}
PINNED_INPUTS = {
    "plan.json": "48da68437e67765853bf0b93768405abaea3eb5ccc67de4a746cfe754cba3a70",
    "result.json": "4ee14be8dbfd14b697fb939140b9815d95146c9f730e5b658438dcfed5c35f4f",
    "internal.json": "481ed99b477b553688b9edee951fb1e6283debaf34129cb0b5f8d7814308c9a3",
    "collection/fit.json": "bbec01b3ef034e8d7a9a200e381af8098f32af0ba431647831429d8fa27f8bbc",
    "collection/calibration.json": "ff7a99538f08fa1b195f4d50d70e1a2299d205120bddee6cb3c264b8f9df3b45",
    "collection/eligibility.json": "8a79cb734b348cf39bbda62acd1663d8d6df44b3816f58fd38367c21a1b943b3",
    "collection/validation.json": "578a498efb450744ecc5b3d3a69fcd608c06bc5d01f632d15e1926b6cf48f679",
}
TRACE_SHA256 = {
    "fit": "d8afdc71b2eab6df06e56dc24a695da5b244e45e617507445f8187a8b3a58772",
    "calibration": "20f019ee8fd50aeb1aa6cc7c4cabe8064c5f8fe69e5538126a6aad9ddca0a884",
    "eligibility": "b20c0b2720cc3f14e579ccae4d999213d0e5bb9484a92c1a36a594b0981f4ebc",
    "validation": "53506c5fff6aea6b4fdb3897a07762bbeaf1b03b863173909f9cf555b9b912a9",
}
SOURCE_FRAME_COUNTS = {"fit": 11988, "calibration": 2942,
                       "eligibility": 8874, "validation": 8978}
EPISODE_RECEIPT_MANIFEST_SHA256 = "0431a6ddb7c9ea18ab5ea4bdc410309300efccbb5157a8f563a0584869219cb7"
NEW_SOURCE_FILES = (
    "experiments/warehouse/v621_protocol.py",
    "backend/training/warehouse_v621_compile.py",
    "backend/training/warehouse_v621_search.py",
    "configs/warehouse_v621_recipe.json",
    "scripts/run_warehouse_v621_diagnostics.py",
)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def read_pinned(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"pinned v6.20 development input changed: {path}")
    return raw


def source_cases():
    """Exact consumed seed arguments; these are not new seed reservations."""
    result = []
    for index in range(275):
        split = ("fit" if index < 100 else "calibration" if index < 125
                 else "eligibility" if index < 200 else "validation")
        base, task = 100270000+index, index % 3+1
        result.append({"base": base, "task": task, "scenario_seed": base*10+task,
            "sampling_mode": "argmax" if 50 <= index < 100 else "sampled",
            "action_seed": 1710000000+index, "original_split": split})
    return result


def _ordered(cases, purpose):
    return sorted(cases, key=lambda row: digest([FOLD_SEED, purpose, row["base"]]))


def grouped_folds(cases=None):
    """Three task-balanced folds; group whole bases and reserve argmax for fit."""
    cases = [dict(row) for row in (source_cases() if cases is None else cases)]
    if (len(cases) != 275 or len({r["base"] for r in cases}) != 275
            or len({r["scenario_seed"] for r in cases}) != 275
            or any(type(r["base"]) is not int or type(r["scenario_seed"]) is not int
                   or type(r["task"]) is not int or r["task"] not in (1, 2, 3)
                   or r["sampling_mode"] not in ("sampled", "argmax") for r in cases)):
        raise RuntimeError("diagnostics require 275 distinct valid whole-base episodes")
    natural = [r for r in cases if r["sampling_mode"] == "sampled"]
    argmax = [r for r in cases if r["sampling_mode"] == "argmax"]
    if (len(natural) != 225 or len(argmax) != 50
            or [sum(r["task"] == t for r in natural) for t in (1, 2, 3)] != [75, 76, 74]):
        raise RuntimeError("diagnostics require the pinned 225 natural task strata and 50 fit-only argmax episodes")
    held = [[] for _ in range(3)]
    offset = 0
    for task in (1, 2, 3):
        group = _ordered([r for r in natural if r["task"] == task], f"held-task-{task}")
        for index, row in enumerate(group):
            held[(offset+index) % 3].append(row)
        offset = (offset+len(group)) % 3
    result = []
    for fold in range(3):
        validation = held[fold]
        excluded = {r["base"] for r in validation}
        train = [r for r in natural if r["base"] not in excluded]
        assignments = {name: [] for name in SPLITS}
        for task in (1, 2, 3):
            group = _ordered([r for r in train if r["task"] == task], f"fold-{fold}-task-{task}")
            cal_count = 8+int(task == fold+1)
            assignments["fit"] += group[:25]
            assignments["calibration"] += group[25:25+cal_count]
            assignments["eligibility"] += group[25+cal_count:]
        assignments["fit"] += argmax
        assignments["validation"] = validation
        assignments = {name: sorted(rows, key=lambda r: r["base"])
                       for name, rows in assignments.items()}
        if {name: len(rows) for name, rows in assignments.items()} != {
                "fit": 125, "calibration": 25, "eligibility": 50, "validation": 75}:
            raise RuntimeError("v6.21 grouped split count changed")
        groups = [{r["base"] for r in assignments[name]} for name in SPLITS]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            raise RuntimeError("whole-base development split leakage")
        result.append({"fold": fold, "splits": assignments,
                       "assignment_sha256": digest(assignments)})
    if sorted(r["base"] for f in result for r in f["splits"]["validation"]) != sorted(r["base"] for r in natural):
        raise RuntimeError("natural episodes must be held out exactly once")
    return result


def jobs():
    return [{"job_id": f"fold-{fold}-{role}-{arm}", "fold": fold,
             "role": role, "arm": arm, "method": method}
            for fold in range(3) for role in ROLES for arm, method in METHODS.items()]


def round_budget_contract():
    return {arm: {"scope": "per_role_fit_across_all_initializations_including_final_refit",
        "maximum_initializations": 8, "maximum_total_rounds": 60,
        "maximum_leaf_api_calls_per_role_fit": 20000,
        "reserved_final_fixed_topology_leaf_calls": 16,
        "guards_and_empty_branches_count_towards_complexity": True}
        for arm in METHODS}


def gate_metrics(metrics):
    """One role/fold must pass every finite fidelity and whole-tree condition."""
    import math
    def minimum(value, limit):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= limit
    def maximum(value, limit):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value <= limit
    f, e = metrics.get("feedback", {}), metrics.get("explanation", {})
    checks = {
        "feedback_coverage": minimum(f.get("coverage"), GATES["minimum_feedback_coverage"]),
        "explanation_coverage": minimum(e.get("coverage"), GATES["minimum_explanation_coverage"]),
        "expected_feedback_coverage": minimum(f.get("expected_actionable_coverage"), GATES["minimum_expected_feedback_coverage"]),
        "feedback_agreement": minimum(f.get("agreement"), GATES["minimum_feedback_agreement"]),
        "explanation_agreement": minimum(e.get("agreement"), GATES["minimum_explanation_agreement"]),
        "feedback_kl": maximum(f.get("kl"), GATES["maximum_kl"]),
        "explanation_kl": maximum(e.get("kl"), GATES["maximum_kl"]),
    }
    for name, section in (("feedback", f), ("explanation", e)):
        for metric in ("whole_tree_raw_illegal_count", "whole_tree_path_uncertified_count"):
            checks[f"{name}_{metric}"] = type(section.get(metric)) is int and section[metric] == 0
    return {"passed": all(checks.values()), "checks": checks}


def restore_episodes(traces, cases, *, actor_sha256=ACTOR_SHA256, role_shas=None):
    """Retain every connected physical frame and bind both roles to one base."""
    import numpy as np
    role_shas = ROLE_SHA256 if role_shas is None else role_shas
    by_scene = {r["scenario_seed"]: r for r in cases}
    if len(by_scene) != len(cases) or len({r["base"] for r in cases}) != len(cases):
        raise RuntimeError("duplicate whole-base input")
    episodes, prior_states = {}, {}
    for row in traces:
        scene, frame = row["base_seed"], row["frame"]
        if scene not in by_scene or type(frame) is not int or frame < 0:
            raise RuntimeError("unregistered scene or physical frame")
        case, base = by_scene[scene], by_scene[scene]["base"]
        if row["actor_sha256"] != actor_sha256 or set(row["roles"]) != set(ROLES):
            raise RuntimeError("trace Actor/role binding differs")
        for prefix in ("pre", "post"):
            if digest(row[f"{prefix}_state_public"]) != row[f"{prefix}_state_public_sha256"]:
                raise RuntimeError("trace public state hash differs")
        if base not in episodes:
            episodes[base] = {role: [] for role in ROLES}
        if (frame != len(episodes[base][ROLES[0]])
                or (frame and prior_states[base] != row["pre_state_public_sha256"])):
            raise RuntimeError("omitted, duplicated, reordered or disconnected physical frame")
        prior_states[base] = row["post_state_public_sha256"]
        for role in ROLES:
            recorded = row["roles"][role]
            obs = np.asarray(recorded["obs"], dtype=np.float32)
            mask = np.asarray(recorded["mask"], dtype=np.float32)
            target = np.asarray(recorded["nn_probabilities"], dtype=np.float32)
            action = recorded["sampled_action"]
            stream = int.from_bytes(sha256(f'{case["action_seed"]}:{row["episode_id"]}:{frame}:{role}'.encode()).digest()[:8], "big")
            if (obs.ndim != 1 or not np.isfinite(obs).all()
                    or mask.shape != (5,) or np.any((mask != 0) & (mask != 1)) or not mask.any()
                    or target.shape != (5,) or not np.isfinite(target).all() or np.any(target < 0)
                    or abs(float(target.sum())-1) > 1e-5 or np.any(target[mask == 0] != 0)
                    or action not in ACTIONS or mask[ACTIONS.index(action)] != 1
                    or recorded["actor_role_sha256"] != role_shas[role]
                    or type(recorded["rng_seed"]) is not int or recorded["rng_seed"] != stream
                    or (case["sampling_mode"] == "argmax" and ACTIONS.index(action) != target.argmax())):
                raise RuntimeError("invalid physical mask, full NN distribution, action, role or RNG binding")
            episodes[base][role].append({"obs": obs, "mask": mask,
                "episode": f"spent-v620-base-{base}", "base": base, "task": case["task"],
                "sampling_mode": case["sampling_mode"], "action_seed": case["action_seed"],
                "base_seed": scene, "frame": frame, "episode_id": row["episode_id"],
                "sampled_action": action, "nn_probabilities": target,
                "rng_seed": recorded["rng_seed"], "actor_sha256": actor_sha256,
                "actor_role_sha256": role_shas[role],
                **{f"{prefix}_state_public{suffix}": row[f"{prefix}_state_public{suffix}"]
                   for prefix in ("pre", "post") for suffix in ("", "_sha256")}})
    if set(episodes) != {r["base"] for r in cases}:
        raise RuntimeError("development input omitted an entire episode")
    return episodes


def checked_inputs(root):
    root = Path(root).resolve()
    directory = root/DATA_ROOT
    blobs = {name: read_pinned(directory/name, expected) for name, expected in PINNED_INPUTS.items()}
    files = {str((directory/name).resolve()): expected for name, expected in PINNED_INPUTS.items()}
    plan, result = json.loads(blobs["plan.json"]), json.loads(blobs["result.json"])
    if (plan["source_closure_sha256"] != LEGACY_SOURCE_SHA256
            or plan["actor"]["sha256"] != ACTOR_SHA256 or plan["actor"]["role_sha256"] != ROLE_SHA256
            or result.get("internal_passed") is not False or result.get("passed") is not False
            or result.get("independent") is not None or result.get("actor_weights_unchanged") is not True
            or result.get("PPO_joint_steps") != 0 or result.get("actor_updates") != 0
            or result.get("pilot_opened") is not False):
        raise RuntimeError("source cohort must remain the closed v6.20 BC precheck")
    cases, traces, manifest, episode_receipts = source_cases(), [], [], {}
    for split in SPLITS:
        split_cases = [r for r in cases if r["original_split"] == split]
        receipt = json.loads(blobs[f"collection/{split}.json"])
        trace_sha = TRACE_SHA256[split]
        trace_path = (directory/"traces"/trace_sha/f"{split}.jsonl").resolve()
        if (receipt["trace"]["sha256"] != trace_sha
                or Path(receipt["trace"]["path"]).resolve() != trace_path
                or receipt.get("excluded_frames") != 0 or receipt["episodes"] != len(split_cases)
                or receipt["trace"]["seeds"] != [r["scenario_seed"] for r in split_cases]
                or receipt["physical_frames_per_role"] != SOURCE_FRAME_COUNTS[split]
                or receipt["trace"]["row_count"] != SOURCE_FRAME_COUNTS[split]):
            raise RuntimeError("spent cohort receipt/trace identity changed")
        raw = read_pinned(trace_path, trace_sha)
        files[str(trace_path)] = trace_sha
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != SOURCE_FRAME_COUNTS[split]:
            raise RuntimeError("spent cohort physical frames changed")
        counts = {}
        for row in rows:
            counts[row["base_seed"]] = counts.get(row["base_seed"], 0)+1
        for index, case in enumerate(split_cases):
            relative = f"collection/{split}/episode-{index:03d}.json"
            raw_episode = (directory/relative).read_bytes()
            episode_sha = sha256(raw_episode).hexdigest()
            episode_receipts[relative] = episode_sha
            files[str((directory/relative).resolve())] = episode_sha
            episode = json.loads(raw_episode)
            evidence = episode["evidence"]
            outcome = evidence["episode_outcomes"]
            if (episode.get("case") != {k: v for k, v in case.items() if k != "original_split"}
                    or episode.get("all_physical_frames") is not True
                    or episode.get("development_only") is not True
                    or evidence["actor_sha256"] != ACTOR_SHA256
                    or evidence["actor_role_sha256"] != ROLE_SHA256
                    or evidence["expected_source_closure_sha256"] != LEGACY_SOURCE_SHA256
                    or len(outcome) != 1 or outcome[0]["seed"] != case["scenario_seed"]
                    or outcome[0]["action_seed"] != case["action_seed"]
                    or outcome[0]["sampling_mode"] != case["sampling_mode"]
                    or counts.get(case["scenario_seed"]) != outcome[0]["frame_count"]):
                raise RuntimeError("spent episode identity or complete frame count changed")
        if set(counts) != {r["scenario_seed"] for r in split_cases}:
            raise RuntimeError("spent trace scene set changed")
        traces += rows
        manifest.append({"original_split": split, "episodes": len(split_cases),
            "physical_frames_per_role": len(rows), "excluded_frames": 0,
            "trace_sha256": trace_sha})
    if digest(episode_receipts) != EPISODE_RECEIPT_MANIFEST_SHA256:
        raise RuntimeError("spent episode receipt bytes changed")
    episodes = restore_episodes(traces, cases)
    if len(traces) != 32782 or len(episodes) != 275:
        raise RuntimeError("not the complete 275-episode/32782-frame cohort")
    return episodes, {"files_sha256": files, "source_episodes": manifest,
        "episode_counts": {str(base): len(rows[ROLES[0]]) for base, rows in sorted(episodes.items())},
        "physical_frames_per_role": len(traces), "excluded_frames": 0,
        "source_episode_receipts_sha256": digest(episode_receipts),
        "source_cases_sha256": digest(cases), "natural_episodes": 225,
        "fit_only_argmax_episodes": 50,
        "purpose": "already_spent_v620_development_data_not_independent_admission"}


def checked_actor(root):
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, role_sha256
    path = (Path(root).resolve()/ACTOR_PATH).resolve()
    actor = FrozenWarehouseV6Actor.load(path, expected_sha256=ACTOR_SHA256,
                                       expected_role_sha256=ROLE_SHA256)
    actual = {role: role_sha256(actor.roles[role].weights,
        model_version=actor.roles[role].model_version) for role in ROLES}
    if actual != ROLE_SHA256:
        raise RuntimeError("v6.20 frozen dual Actor roles changed")
    return actor, path


def target_identity(actor, episodes):
    import numpy as np
    from backend.training.warehouse_v6_program import _masked_probabilities
    from scripts.run_warehouse_v619_cached_comparison import array_digest
    ordered = [episodes[base] for base in sorted(episodes)]
    obs = np.asarray([[rows[role][i]["obs"] for role in ROLES]
        for rows in ordered for i in range(len(rows[ROLES[0]]))], dtype=np.float32)
    masks = np.asarray([[rows[role][i]["mask"] for role in ROLES]
        for rows in ordered for i in range(len(rows[ROLES[0]]))], dtype=np.float32)
    stored = np.asarray([[rows[role][i]["nn_probabilities"] for role in ROLES]
        for rows in ordered for i in range(len(rows[ROLES[0]]))], dtype=np.float32)
    target = _masked_probabilities(actor.logits(obs, masks), masks)
    if not np.allclose(target, stored, atol=2e-6, rtol=2e-5):
        raise RuntimeError("recorded full masked NN distributions differ from frozen Actor")
    return {"recomputed_full_masked_nn_target_sha256": array_digest(target),
        "recorded_full_masked_nn_target_sha256": array_digest(stored),
        "full_observation_sha256": array_digest(obs), "physical_mask_sha256": array_digest(masks),
        "physical_frames_per_role": len(obs),
        "ordering": "ascending_base_then_physical_frame_then_role"}


def source_closure_files(root):
    from backend.training import warehouse_v6_runner as legacy
    root = Path(root).resolve()
    # Hash relative to the requested checkout, not an accidentally imported peer.
    files = {relative: sha256((root/relative).read_bytes()).hexdigest()
             for relative in sorted(legacy.source_closure_files_sha256())}
    if len(files) != 180 or digest(files) != LEGACY_SOURCE_SHA256:
        raise RuntimeError("closed v6.20 180-file source closure changed")
    files.update({relative: sha256((root/relative).read_bytes()).hexdigest()
                  for relative in NEW_SOURCE_FILES})
    return dict(sorted(files.items()))


def source_closure_sha256(root):
    return digest(source_closure_files(root))


def executable_recipe():
    return {"schema": "warehouse-v621-spent-diagnostics-recipe.v1", "candidate_id": CANDIDATE_ID,
        "batch_id": BATCH_ID, "methods": dict(METHODS), "frozen_actor_sha256": ACTOR_SHA256,
        "frozen_actor_role_sha256": dict(ROLE_SHA256), "legacy_source_sha256": LEGACY_SOURCE_SHA256,
        "whole_tree_limits": {"maximum_depth": 5, "maximum_leaves": 16,
            "maximum_source_features_per_node": 3, "guards_and_empty_branches_count": True},
        "cv": {"fold_seed": FOLD_SEED, "folds": 3, "source_episodes": 275,
            "natural_episodes": 225, "fit_only_argmax_episodes": 50,
            "physical_frames_per_role": 32782, "all_physical_frames": True,
            "split_episode_counts": {"fit": 125, "calibration": 25, "eligibility": 50, "validation": 75},
            "fit_sampled_episodes": 75, "fit_argmax_episodes": 50,
            "whole_base_grouping": True, "argmax_only_in_fit": True,
            "fold_assignments_sha256": digest(grouped_folds()), "jobs": 18},
        "search": round_budget_contract(),
        "leaf_fit": {"epsilon": .001, "regularization": .0001, "maxiter": 200, "tolerance": .000001},
        "calibration_temperatures": [.8, .9, 1., 1.1, 1.2],
        "gates": dict(GATES),
        "selection": {"requires_all_eighteen_completed_or_scientific_infeasible": True,
            "eligible_method_requires_both_roles_all_three_folds": True,
            "ordering": ["minimum_explanation_coverage", "minimum_expected_feedback_coverage",
                         "negative_maximum_qualified_kl", "negative_total_leaf_count", "arm"],
            "infeasible_compiled_topology_is_scientific_failure": True,
            "none_pass_route": "shared_structure_actor_implementation",
            "any_pass_route": "three_initializer_bc_development"},
        "interpretation": {"current_topology_action_agreement_upper_bound_is_not_class_impossibility": True,
            "no_equal_compute_efficiency_claim_without_measured_costs": True,
            "spent_cohort_never_independent_admission": True},
        "future_stages_closed": ["new_cohort", "BC", "PPO", "formal_training", "sealed_audit", "holdout", "pilot"],
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "publishable": False, "admission_evidence": False, "fresh_validation_evidence": False}


def validate_recipe(root):
    raw = (Path(root)/RECIPE_PATH).read_bytes()
    if json.loads(raw) != executable_recipe():
        raise RuntimeError("v6.21 executable diagnostic recipe drifted")
    return sha256(raw).hexdigest()
