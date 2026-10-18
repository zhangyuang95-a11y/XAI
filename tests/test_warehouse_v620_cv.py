from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.warehouse import v620_protocol as v
from scripts import run_warehouse_development_cv as c


def test_three_whole_base_folds_keep_all_episodes_roles_and_frames_together():
    folds = v.grouped_folds()
    assert folds == v.grouped_folds()
    held, fit_argmax = [], []
    for fold in folds:
        parts = fold["splits"]
        assert {k: len(rows) for k, rows in parts.items()} == {
            "fit": 55, "calibration": 18, "eligibility": 18, "validation": 33}
        bases = {k: {row["base"] for row in rows} for k, rows in parts.items()}
        for i, name in enumerate(v.SPLITS):
            assert all(not (bases[name] & bases[other]) for other in v.SPLITS[:i])
        assert sum(r["sampling_mode"] == "argmax" for r in parts["fit"]) == 25
        assert all(r["sampling_mode"] == "sampled" for name in v.SPLITS[1:] for r in parts[name])
        assert all(row["scenario_seed"] == row["base"]*10+row["task"]
                   for rows in parts.values() for row in rows)
        held += [row["base"] for row in parts["validation"]]
        fit_argmax.append({row["base"] for row in parts["fit"] if row["sampling_mode"] == "argmax"})
        assert fold["assignment_sha256"] == v.digest(parts)
    assert len(held) == len(set(held)) == 99
    assert fit_argmax[0] == fit_argmax[1] == fit_argmax[2]


def test_cv_rejects_related_argmax_and_sampled_scene_base():
    cases = [r for rows in v.seed_plan()["splits"].values() for r in rows]
    cases = deepcopy(cases)
    cases[25]["base"] = cases[0]["base"]
    with pytest.raises(RuntimeError, match="whole-base"):
        v.grouped_folds(cases)


def toy_trace():
    cases = [{"base": 8, "task": 1, "scenario_seed": 81, "sampling_mode": "sampled"},
             {"base": 9, "task": 2, "scenario_seed": 92, "sampling_mode": "argmax"}]
    traces = []
    for case in cases:
        for frame in range(3):
            pre, post = {"frame": frame}, {"frame": frame+1}
            traces.append({"base_seed": case["scenario_seed"], "frame": frame, "episode_id": 1,
                "actor_sha256": v.ACTOR_SHA256,
                "pre_state_public": pre, "pre_state_public_sha256": v.digest(pre),
                "post_state_public": post, "post_state_public_sha256": v.digest(post),
                "roles": {role: {"obs": [float(frame), 0], "mask": [1, 1, 1, 1, 1],
                    "nn_probabilities": [.6, .1, .1, .1, .1], "sampled_action": "UP",
                    "rng_seed": frame, "actor_role_sha256": c.ROLE_SHA256[role]}
                    for role in v.ROLES}})
    return traces, cases


def test_restore_retains_every_frame_action_distribution_and_same_base_roles():
    traces, cases = toy_trace()
    episodes = c.restore_episodes(traces, cases)
    assert set(episodes) == {8, 9}
    for base, roles in episodes.items():
        assert len(roles["robot_1"]) == len(roles["robot_2"]) == 3
        for role in v.ROLES:
            assert [row["frame"] for row in roles[role]] == [0, 1, 2]
            assert all(row["episode"] == f"spent-v619-base-{base}" for row in roles[role])
            assert all("label" not in row for row in roles[role])
            np.testing.assert_allclose(roles[role][0]["nn_probabilities"], [.6, .1, .1, .1, .1])


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reorder", "state", "mask", "target", "argmax", "actor", "role", "episode"])
def test_restore_rejects_corruption_without_rollout(kind):
    traces, cases = toy_trace()
    if kind == "missing":
        traces.pop(1)
    elif kind == "duplicate":
        traces.insert(1, deepcopy(traces[0]))
    elif kind == "reorder":
        traces[0], traces[1] = traces[1], traces[0]
    elif kind == "state":
        traces[1]["pre_state_public"]["frame"] = 4
    elif kind == "mask":
        traces[0]["roles"]["robot_1"]["mask"][1] = .5
    elif kind == "target":
        traces[0]["roles"]["robot_1"]["nn_probabilities"][1] = -.1
    elif kind == "argmax":
        traces[-1]["roles"]["robot_1"]["sampled_action"] = "WAIT"
    elif kind == "actor":
        traces[0]["actor_sha256"] = "wrong"
    elif kind == "role":
        traces[0]["roles"]["robot_1"]["actor_role_sha256"] = "wrong"
    else:
        traces = traces[:3]
    with pytest.raises(RuntimeError):
        c.restore_episodes(traces, cases)


def test_atomic_pinned_read_and_exclusive_receipts(tmp_path):
    source = tmp_path/"source"
    source.write_bytes(b"original")
    digest = c.file_sha(source)
    assert c.read_pinned(source, digest) == b"original"
    source.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="changed"):
        c.read_pinned(source, digest)
    target = tmp_path/"receipt.json"
    c.write_new(target, {"PPO_joint_steps": 0})
    with pytest.raises(FileExistsError):
        c.write_new(target, {"PPO_joint_steps": 100})


def metrics(coverage=.44, expected=.39, agreement=.98, kl=.10, illegal=0, path=0):
    common = {"coverage": coverage, "expected_actionable_coverage": expected,
        "agreement": agreement, "kl": kl, "whole_tree_raw_illegal_count": illegal,
        "whole_tree_path_uncertified_count": path}
    return {"feedback": deepcopy(common), "explanation": deepcopy(common)}


@pytest.mark.parametrize("kwargs", [{"coverage": .399}, {"expected": .349}, {"agreement": .969},
                                    {"kl": .201}, {"illegal": 1}, {"path": 1}])
def test_development_gate_never_substitutes_mean_for_worst_role_fold(kwargs):
    assert c.development_gate(metrics())["passed"]
    assert not c.development_gate(metrics(**kwargs))["passed"]


def results(coverages=None):
    coverages = coverages or {"M0": .9, "M1": .42, "M2": .46}
    out = []
    for job in v.jobs():
        row = metrics(coverage=coverages[job["arm"]])
        out.append({"job": job, "metrics": {"validation": row},
                    "development_gate": c.development_gate(row), "leaf_count": 16})
    return out


def test_complete_paired_summary_selects_only_new_methods_by_worst_coverage():
    rows = results()
    result = c.summarize_results(rows)
    assert result["chosen_method_arm"] == "M2"  # M0 is deliberately better, but ineligible.
    assert result["chosen_method"] == v.METHODS["M2"]
    assert result["environment_resets"] == result["actor_updates"] == result["PPO_joint_steps"] == 0
    assert result["fresh_validation_evidence"] is result["admission_evidence"] is False
    with pytest.raises(RuntimeError, match="eighteen"):
        c.summarize_results(rows[:-1])
    with pytest.raises(RuntimeError, match="eighteen"):
        c.summarize_results(rows[:-1]+[rows[0]])


def test_failed_new_methods_route_to_common_bc_without_new_data_authority():
    result = c.summarize_results(results({"M0": .9, "M1": .39, "M2": .38}))
    assert result["chosen_method_arm"] is None
    assert result["next_stage"] == "common_bc_structure_learning_before_fresh_validation"
    assert result["arms"]["M0"]["all_role_fold_gates_passed"]


def test_single_failed_role_fold_invalidates_arm_even_when_mean_exceeds_45_percent():
    rows = results({"M0": .5, "M1": .44, "M2": .60})
    row = next(row for row in rows if row["job"]["arm"] == "M2")
    row["metrics"]["validation"] = metrics(coverage=.39)
    row["development_gate"] = c.development_gate(row["metrics"]["validation"])
    result = c.summarize_results(rows)
    assert result["arms"]["M2"]["mean_45_percent_target_reached"]
    assert not result["arms"]["M2"]["all_role_fold_gates_passed"]
    assert result["chosen_method_arm"] == "M1"


def test_recipe_keeps_future_data_and_training_gated_and_raw_fair():
    recipe = v.executable_recipe()
    assert recipe["search"]["beam_width"] == 4
    assert recipe["search"]["max_nonimproving_steps"] == 2
    assert recipe["future_common_bc"]["raw_ppo_external_program_feedback"] == 0
    assert recipe["future_common_bc"]["residual_soft_penalty_candidates"] == [.01, .03]
    assert recipe["future_fresh"]["requires_method_lock_and_new_seed_clearance"]
    assert recipe["future_fresh"]["budget_restarts_from_same_bc"]
    assert recipe["environment_resets"] == recipe["actor_updates"] == recipe["PPO_joint_steps"] == 0


@pytest.mark.parametrize("mutation", ["fold", "job", "publishable", "steps", "schema"])
def test_parent_identity_drift_is_rejected_before_actor_or_fitting(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    directory = tmp_path/v.OUTPUT_ROOT/"cv-test"
    directory.mkdir(parents=True)
    plan = {"schema": "warehouse-development-cv-plan.v1", "output_directory": str(directory),
        "candidate_id": v.CANDIDATE_ID, "folds": v.grouped_folds(),
        "fold_assignment_sha256": v.digest(v.grouped_folds()), "jobs": v.jobs(),
        "gates": v.GATES, "selection_arms": ["M1", "M2"], "publishable": False,
        "admission_evidence": False, "fresh_validation_evidence": False,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0}
    if mutation == "fold":
        plan["folds"][0]["splits"]["validation"].pop()
    elif mutation == "job":
        plan["jobs"][0]["method"] = "fake"
    elif mutation == "publishable":
        plan["publishable"] = True
    elif mutation == "steps":
        plan["PPO_joint_steps"] = False  # bool must not pass as integer zero.
    else:
        plan["schema"] = "unknown"
    path = directory/"plan.json"
    c.write_new(path, plan)
    # The plan intentionally has no Actor identity: reaching that stage would
    # raise KeyError, demonstrating immutable checks must run first.
    with pytest.raises(RuntimeError, match="immutable parent"):
        c.load_plan(path)


def test_exclusive_job_claim_prevents_resume_or_scientific_retry(tmp_path, monkeypatch):
    plan_path = tmp_path/"plan.json"
    job = v.jobs()[0]
    job_dir = tmp_path/"jobs"/job["job_id"]
    job_dir.mkdir(parents=True)
    (job_dir/"failed.json").write_text('{"scientific_retry_permitted":false}')
    monkeypatch.setattr(c, "load_plan", lambda _: (plan_path, {"jobs": [job]}))
    with pytest.raises(FileExistsError):
        c.execute_job(plan_path, job["job_id"])
    assert not (job_dir/"started.json").exists()
