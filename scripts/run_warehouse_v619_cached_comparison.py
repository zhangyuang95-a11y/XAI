#!/usr/bin/env python3
"""P0/P1/P2 on the inspected 6150 cache; no rollout, training or admission."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLES = ("robot_1", "robot_2")
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
SPLITS = ("fit", "calibration", "eligibility", "validation")
COUNTS = {"fit": 6000, "calibration": 3000, "eligibility": 3000, "validation": 2880}
BASELINE = ROOT / "output/warehouse_v618_preflight/warehouse-v618-quality-selective-01-bc-preflight/pair_6150_steps_200000"
INPUT_SHA256 = {
    "final-bc.actor.npz": "76acb00d8d24c5540379513a89f7c1024f5849427858fe945a9c39cd67a3fd88",
    "final-bc-program-rows.npz": "c27e0645f01b691707dd885c9432b90b892e3be47143217b1f60e0f8876439dc",
    "final-bc-robot_1.program.json": "5f7b48108209ad4f7b37b23955ddcbba5ab1c5bcc0bb862ec0398195ebc708fc",
    "final-bc-robot_2.program.json": "17e358fb888661f2b6eea6e3b20a3e3a425651d037623e971682374c4fdb15ff",
    "final-bc-internal-precheck.json": "3c22a08fedabc892f40e287542a3d6ed34fa8fc2b5fb5dfe76bffb3d9b4286ed",
}
ROLE_SHA256 = {
    "robot_1": "014b487d6d5ebaae20f63cf399eeafa776319ab148cf6d72376922a05df235c2",
    "robot_2": "b16f5a9e20d136e19bf5061bd397ebdd016d288b79011ecd600785c7cf1dfbc3",
}
P0_EXPECTED = {
    "robot_1": {"rows": 736, "coverage": .25555555555555554,
        "expected_actionable_coverage": .2519976496696472,
        "agreement": .9918478260869565, "kl": .05971962958574295,
        "whole_tree_raw_illegal_count": 538, "whole_tree_path_uncertified_count": 1760},
    "robot_2": {"rows": 756, "coverage": .2625,
        "expected_actionable_coverage": .2611517310142517,
        "agreement": .9973544973544973, "kl": .01998724415898323,
        "whole_tree_raw_illegal_count": 194, "whole_tree_path_uncertified_count": 1464},
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def array_digest(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return sha256(header + b"\n" + array.tobytes()).hexdigest()


def check_inputs(directory):
    actual = {name: sha256((directory / name).read_bytes()).hexdigest() for name in INPUT_SHA256}
    if actual != INPUT_SHA256:
        raise RuntimeError("6150 input asset/cache/program bytes differ from the pinned manifest")
    return actual


def structure_only(program):
    """P1 contains topology only: old leaf logits are never optimization targets."""
    return {"root_ref": int(program.root_ref),
        "split_weights": np.array(program.split_weights, dtype=np.float32, copy=True),
        "split_bias": np.array(program.split_bias, dtype=np.float32, copy=True),
        "left_refs": np.array(program.left_refs, dtype=np.int64, copy=True),
        "right_refs": np.array(program.right_refs, dtype=np.int64, copy=True),
        "leaf_count": int(len(program.leaf_logits))}


def restore_cache(cache, *, counts=COUNTS):
    """Preserve every physical frame and split; labels are deliberately unused."""
    parts, manifests, seen_episodes, seen_scenarios = {}, {}, set(), set()
    for split in SPLITS:
        fields = {name: np.asarray(cache[f"{split}.{name}"]) for name in (
            "obs", "mask", "episode_ids", "scenario_seeds", "frames",
            "environment_episode_ids", "sampled_actions")}
        n = len(fields["obs"])
        obs, masks, sampled = fields["obs"], fields["mask"], fields["sampled_actions"]
        if (n != counts[split] or obs.ndim != 3 or obs.shape[1] != 2
                or masks.shape != (n, 2, 5) or sampled.shape != (n, 2)
                or not np.isfinite(obs).all() or not np.isfinite(masks).all()
                or np.any((masks != 0) & (masks != 1)) or np.any(masks.sum(-1) < 1)
                or not np.issubdtype(sampled.dtype, np.integer)
                or np.any(sampled < 0) or np.any(sampled >= 5)
                or np.any(np.take_along_axis(masks, sampled[..., None], axis=-1) <= .5)
                or any(len(value) != n for value in fields.values())):
            raise RuntimeError("cached physical frames, masks or sampled NN actions are invalid")
        episodes = set(map(str, fields["episode_ids"]))
        scenarios = set(map(int, fields["scenario_seeds"]))
        if episodes & seen_episodes or scenarios & seen_scenarios:
            raise RuntimeError("cached episode/scenario purposes overlap")
        seen_episodes.update(episodes); seen_scenarios.update(scenarios)
        for episode in episodes:
            take = fields["episode_ids"].astype(str) == episode
            frames = fields["frames"][take]
            if (not np.issubdtype(frames.dtype, np.integer)
                    or not np.array_equal(frames, np.arange(len(frames)))
                    or len(set(map(int, fields["scenario_seeds"][take]))) != 1):
                raise RuntimeError("cached episode contains omitted, duplicated or reordered frames")
        parts[split] = {role: [{
            "obs": obs[index, role_index], "mask": masks[index, role_index],
            "episode": str(fields["episode_ids"][index]),
            "base_seed": int(fields["scenario_seeds"][index]),
            "frame": int(fields["frames"][index]),
            "episode_id": int(fields["environment_episode_ids"][index]),
            "sampled_action": ACTIONS[int(sampled[index, role_index])],
        } for index in range(n)] for role_index, role in enumerate(ROLES)}
        arrays = {name: array_digest(value) for name, value in fields.items()}
        manifests[split] = {"physical_frames_per_role": n, "episode_count": len(episodes),
            "scenario_count": len(scenarios), "excluded_frames": 0,
            "episode_ids_sha256": digest(sorted(episodes)),
            "scenario_seeds_sha256": digest(sorted(scenarios)),
            "array_sha256": arrays, "full_frame_sha256": digest(arrays)}
    return parts, manifests


def actor_identity(actor):
    from backend.training.warehouse_v6_actor import role_sha256
    return {role: role_sha256(actor.roles[role].weights,
                              model_version=actor.roles[role].model_version) for role in ROLES}


def target_identity(actor, cache):
    from backend.training.warehouse_v6_program import _masked_probabilities
    return {split: array_digest(_masked_probabilities(actor.logits(
        cache[f"{split}.obs"], cache[f"{split}.mask"]), cache[f"{split}.mask"]))
        for split in SPLITS}


def ensure_identity(expected, actual):
    if actual != expected:
        raise RuntimeError("frozen source/Actor/cache/full-frame/NN-target identity changed")


def check_p0(role, metric):
    for key, expected in P0_EXPECTED[role].items():
        actual = metric.get(key)
        if (type(expected) is int and (type(actual) is not int or actual != expected)
                or type(expected) is float and (actual is None or abs(actual - expected) > 1e-6)):
            raise RuntimeError(f"P0 cached profile differs for {role}.{key}: {actual!r}")


def program_metrics(program, actor, rows):
    from backend.training.warehouse_v6_program import (_metric_context, _metrics,
        _masked_probabilities, certified_program_actions)
    role_actor = actor.roles[program.role]
    context = _metric_context(program, role_actor, rows)
    feedback = _metrics(program, role_actor, rows, set(program.feedback_leaves),
                        "feedback", context=context)
    explanation = _metrics(program, role_actor, rows, set(program.explanation_leaves),
                           "explanation", context=context)
    # Report all five NN argmax strata even when no leaf qualifies.
    probabilities, masks, leaves = context["p"], context["masks"], context["leaves"]
    raw_logits = program.leaf_logits[leaves]
    q = _masked_probabilities(raw_logits, masks)
    nn_top, raw_top, program_top = probabilities.argmax(-1), raw_logits.argmax(-1), q.argmax(-1)
    legal = masks[np.arange(len(rows)), raw_top] > .5
    certificates = certified_program_actions(program)
    proven = np.asarray([int(action) in certificates[int(leaf)]
                         for action, leaf in zip(raw_top, leaves, strict=True)])
    row_kl = np.sum(probabilities * (np.log(np.maximum(probabilities, 1e-8))
                    - np.log(np.maximum(q, 1e-8))), axis=-1)
    strata = {}
    for index, action in enumerate(ACTIONS):
        take = nn_top == index
        strata[action] = {"physical_frames": int(take.sum()),
            "whole_tree_raw_illegal_count": int((~legal & take).sum()),
            "whole_tree_path_uncertified_count": int((~proven & take).sum()),
            "whole_tree_agreement": float((program_top[take] == nn_top[take]).mean()) if take.any() else None,
            "whole_tree_mean_kl": float(row_kl[take].mean()) if take.any() else None,
            "feedback_qualified_frames": int((take & np.isin(leaves, tuple(program.feedback_leaves))).sum()),
            "explanation_qualified_frames": int((take & np.isin(leaves, tuple(program.explanation_leaves))).sum())}
    return {"feedback": feedback, "explanation": explanation,
        "whole_tree_mean_kl": float(row_kl.mean()), "nn_five_action_strata": strata,
        "illegal_count_scope": "whole_tree_raw_before_masking_and_qualification",
        "qualified_illegal_count_scope": "qualified_region_before_per_frame_action_matching",
        "all_physical_frames": len(rows), "excluded_frames": 0}


def solver_counts(report):
    fits = report.get("constrained_leaf_fits", {})
    candidates = [row for item in fits.values() for row in item.get("candidate_reports", [])]
    return {"final_leaf_count": len(fits), "final_leaf_action_solves": len(candidates),
        "final_leaf_unsuccessful_action_solves": sum(row.get("success") is not True for row in candidates),
        "topology_search": {key: value for key, value in report.get("fit_tree", {}).items()
            if key in ("actual_convex_solver_calls", "leaf_optimization_calls", "leaf_cache_hits",
                       "rejected_solver_candidates", "solver_failure_examples", "search_budget", "search_runs")},
        "solver_count_scope": "final_leaf_solves_plus_separately_reported_topology_search"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--execute", action="store_true", help="Fit P1/P2 only after source freeze")
    args = parser.parse_args(argv)
    from backend.training import warehouse_v6_runner as runner
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v6_program import (WarehouseV6Program,
        PATH_CONSTRAINED_METHOD, fit_program, program_structure_sha256)
    from experiments.warehouse import v619_protocol
    source_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if (output == source_dir or output.is_relative_to(source_dir) or source_dir.is_relative_to(output)):
        raise RuntimeError("comparison output must be isolated from the preserved source run")
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    initial_source = runner.source_closure_sha256()
    if initial_source != args.expected_source_sha256:
        raise RuntimeError("comparison source differs from the authorized frozen closure")
    input_identity = check_inputs(source_dir)
    actor = FrozenWarehouseV6Actor.load(source_dir / "final-bc.actor.npz",
        expected_sha256=INPUT_SHA256["final-bc.actor.npz"], expected_role_sha256=ROLE_SHA256)
    with np.load(source_dir / "final-bc-program-rows.npz", allow_pickle=False) as archive:
        cache = {key: np.array(archive[key], copy=True) for key in archive.files}
    for value in cache.values():
        value.setflags(write=False)
    parts, frame_manifest = restore_cache(cache)
    originals = {role: WarehouseV6Program.load(source_dir / f"final-bc-{role}.program.json",
        expected_sha256=INPUT_SHA256[f"final-bc-{role}.program.json"],
        actor_role_sha256=ROLE_SHA256[role]) for role in ROLES}
    structures = {role: structure_only(originals[role]) for role in ROLES}
    before = {"source_closure_sha256": initial_source,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_sha256": input_identity, "actor_role_sha256": actor_identity(actor),
        "masked_full_nn_target_sha256": target_identity(actor, cache),
        "full_frame_sha256": {name: row["full_frame_sha256"] for name, row in frame_manifest.items()}}
    if before["actor_role_sha256"] != ROLE_SHA256:
        raise RuntimeError("loaded frozen dual-Actor role hashes differ")

    def snapshot():
        _, frames = restore_cache(cache)
        return {"source_closure_sha256": runner.source_closure_sha256(),
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs_sha256": check_inputs(source_dir), "actor_role_sha256": actor_identity(actor),
            "masked_full_nn_target_sha256": target_identity(actor, cache),
            "full_frame_sha256": {name: row["full_frame_sha256"] for name, row in frames.items()}}

    plan = {"schema": "warehouse-v619-cached-comparison-plan.v1",
        "candidate_id": v619_protocol.CANDIDATE_ID, "purpose": "inspected_6150_cache_exploration_only",
        "publishable": False, "admission_evidence": False, "fresh_validation_evidence": False,
        "environment_resets": 0, "PPO_joint_steps": 0, "actor_updates": 0,
        "input_directory": str(source_dir), "identity": before, "frame_manifest": frame_manifest,
        "method": PATH_CONSTRAINED_METHOD, "executable_recipe_sha256": v619_protocol.validate_recipe(ROOT),
        "arms": {"P0": "original_v618_program_without_refit",
                 "P1": "path_constrained_leaf_fit_on_fixed_P0_topology",
                 "P2": "predeclared_fit_only_path_constrained_new_tree"},
        "P1_structure_sha256": {role: program_structure_sha256(value) for role, value in structures.items()},
        "P0_expected_feedback_profile": P0_EXPECTED,
        "optimization_target": "complete_frozen_nn_masked_five_action_distribution",
        "teacher_labels_and_old_leaf_logits_used_as_targets": False,
        "validation_cannot_select_another_actor_or_be_admission_evidence": True}
    write_exclusive(output / "plan.json", plan)
    if not args.execute:
        ensure_identity(before, snapshot())
        print(json.dumps({"plan_path": str(output / "plan.json"), "scientific_fits_started": False}))
        return 0
    results, errors = {}, []
    try:
        for role in ROLES:
            results[role] = {}
            for arm in ("P0", "P1", "P2"):
                ensure_identity(before, snapshot())
                arm_start = perf_counter()
                try:
                    kwargs = ({"fixed_structure": structures[role],
                        "fixed_structure_sha256": plan["P1_structure_sha256"][role]} if arm == "P1" else {})
                    program = (originals[role] if arm == "P0" else fit_program(role, actor,
                        *(parts[name][role] for name in SPLITS), method=PATH_CONSTRAINED_METHOD, **kwargs))
                    fitted_seconds = perf_counter() - arm_start
                    metrics = program_metrics(program, actor, parts["validation"][role])
                    if arm == "P0":
                        check_p0(role, metrics["feedback"])
                    if arm == "P1" and program_structure_sha256(structure_only(program)) != plan["P1_structure_sha256"][role]:
                        raise RuntimeError("P1 changed the pinned original topology")
                    path = output / f"{role}.{arm}.program.json"
                    program_sha = program.save(path)
                    restored = WarehouseV6Program.load(path, expected_sha256=program_sha,
                        actor_role_sha256=ROLE_SHA256[role])
                    if restored.sha256() != program.sha256():
                        raise RuntimeError("program serialization changed its contents")
                    item = {"status": "completed", "fit_or_load_seconds": fitted_seconds,
                        "total_seconds": perf_counter()-arm_start, "metrics": metrics,
                        "solver_counts": solver_counts(program.report), "report": program.report,
                        "program": {"path": str(path), "sha256": program_sha}}
                except Exception as error:
                    item = {"status": "error", "total_seconds": perf_counter()-arm_start,
                        "error": f"{type(error).__name__}: {error}",
                        "solver_error_report": getattr(error, "report", None)}
                    errors.append({"role": role, "arm": arm, "error": item["error"]})
                ensure_identity(before, snapshot())
                results[role][arm] = item
                write_exclusive(output / f"{role}.{arm}.result.json", item)
                print(json.dumps({"role": role, "arm": arm, "status": item["status"],
                                  "seconds": item["total_seconds"], "publishable": False}), flush=True)
                if arm == "P0" and item["status"] != "completed":
                    raise RuntimeError("original cached profile could not be reproduced; comparison stopped")
        after = snapshot()
        ensure_identity(before, after)
        result = {"schema": "warehouse-v619-cached-comparison-result.v1",
            "purpose": plan["purpose"], "publishable": False, "admission_evidence": False,
            "fresh_validation_evidence": False, "environment_resets": 0, "PPO_joint_steps": 0,
            "actor_updates": 0, "plan_sha256": sha256((output / "plan.json").read_bytes()).hexdigest(),
            "identity_before": before, "identity_after": after,
            "total_seconds": perf_counter()-start, "roles": results, "errors": errors,
            "completed_without_errors": not errors}
        write_exclusive(output / "result.json", result)
        return 0 if not errors else 1
    except Exception as error:
        write_exclusive(output / "failure.json", {"publishable": False, "admission_evidence": False,
            "environment_resets": 0, "PPO_joint_steps": 0, "identity_before": before,
            "error": f"{type(error).__name__}: {error}", "completed_arms": results})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
