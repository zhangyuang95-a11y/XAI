from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from experiments.warehouse import v621_protocol as v


def test_three_folds_group_every_base_and_hold_natural_once_without_argmax_leakage():
    folds = v.grouped_folds()
    assert folds == v.grouped_folds(list(reversed(v.source_cases())))
    held = []
    source_argmax = {r["base"] for r in v.source_cases() if r["sampling_mode"] == "argmax"}
    for fold in folds:
        parts = fold["splits"]
        assert {k: len(rows) for k, rows in parts.items()} == {
            "fit": 125, "calibration": 25, "eligibility": 50, "validation": 75}
        groups = {k: {r["base"] for r in rows} for k, rows in parts.items()}
        for index, name in enumerate(v.SPLITS):
            assert all(not (groups[name] & groups[other]) for other in v.SPLITS[:index])
        assert sum(r["sampling_mode"] == "sampled" for r in parts["fit"]) == 75
        assert {r["base"] for r in parts["fit"] if r["sampling_mode"] == "argmax"} == source_argmax
        assert all(r["sampling_mode"] == "sampled" for name in v.SPLITS[1:] for r in parts[name])
        assert set.union(*groups.values()) == {r["base"] for r in v.source_cases()}
        assert [sum(r["task"] == t and r["sampling_mode"] == "sampled" for r in parts["fit"])
                for t in (1, 2, 3)] == [25, 25, 25]
        assert all(abs(sum(r["task"] == t for r in parts["validation"])-25) <= 1 for t in (1, 2, 3))
        assert fold["assignment_sha256"] == v.digest(parts)
        held += [r["base"] for r in parts["validation"]]
    assert len(held) == len(set(held)) == 225
    assert set(held).isdisjoint(source_argmax)


@pytest.mark.parametrize("mutation", ["base", "scene", "argmax", "task", "sampling", "omit"])
def test_fold_contract_rejects_related_groups_and_wrong_cohort(mutation):
    cases = v.source_cases()
    if mutation == "base":
        cases[1]["base"] = cases[0]["base"]
    elif mutation == "scene":
        cases[1]["scenario_seed"] = cases[0]["scenario_seed"]
    elif mutation == "argmax":
        cases[0]["sampling_mode"] = "argmax"
    elif mutation == "task":
        cases[0]["task"] = 2
    elif mutation == "sampling":
        cases[0]["sampling_mode"] = "teacher"
    else:
        cases.pop()
    with pytest.raises(RuntimeError):
        v.grouped_folds(cases)


def toy_trace():
    cases = [{"base": 8, "task": 1, "scenario_seed": 81, "sampling_mode": "sampled", "action_seed": 800},
             {"base": 9, "task": 2, "scenario_seed": 92, "sampling_mode": "argmax", "action_seed": 900}]
    traces = []
    for case in cases:
        for frame in range(3):
            pre, post = {"frame": frame}, {"frame": frame+1}
            roles = {}
            for role in v.ROLES:
                stream = int.from_bytes(sha256(f'{case["action_seed"]}:1:{frame}:{role}'.encode()).digest()[:8], "big")
                roles[role] = {"obs": [float(frame), 0], "mask": [1]*5,
                    "nn_probabilities": [.6, .1, .1, .1, .1], "sampled_action": "UP",
                    "rng_seed": stream, "actor_role_sha256": v.ROLE_SHA256[role]}
            traces.append({"base_seed": case["scenario_seed"], "frame": frame, "episode_id": 1,
                "actor_sha256": v.ACTOR_SHA256,
                "pre_state_public": pre, "pre_state_public_sha256": v.digest(pre),
                "post_state_public": post, "post_state_public_sha256": v.digest(post), "roles": roles})
    return traces, cases


def test_restore_preserves_all_joint_frames_distributions_and_whole_base_role_pairs():
    traces, cases = toy_trace()
    episodes = v.restore_episodes(traces, cases)
    assert set(episodes) == {8, 9}
    for base, roles in episodes.items():
        assert len(roles["robot_1"]) == len(roles["robot_2"]) == 3
        for role in v.ROLES:
            assert [r["frame"] for r in roles[role]] == [0, 1, 2]
            assert all(r["episode"] == f"spent-v620-base-{base}" for r in roles[role])
            assert all(r["task"] == base-7 for r in roles[role])
            assert all("label" not in r for r in roles[role])
            np.testing.assert_allclose(roles[role][0]["nn_probabilities"], [.6, .1, .1, .1, .1])


@pytest.mark.parametrize("kind", ["missing", "duplicate", "order", "state", "disconnected", "mask",
                                  "target", "argmax", "actor", "role", "rng", "episode"])
def test_restore_rejects_corrupt_frames_targets_bindings_without_rollouts(kind):
    traces, cases = toy_trace()
    if kind == "missing":
        traces.pop(1)
    elif kind == "duplicate":
        traces.insert(1, deepcopy(traces[0]))
    elif kind == "order":
        traces[0], traces[1] = traces[1], traces[0]
    elif kind == "state":
        traces[1]["pre_state_public"]["frame"] = 100
    elif kind == "disconnected":
        traces[1]["pre_state_public"] = {"frame": 100}
        traces[1]["pre_state_public_sha256"] = v.digest({"frame": 100})
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
    elif kind == "rng":
        traces[0]["roles"]["robot_1"]["rng_seed"] += 1
    else:
        traces = traces[:3]
    with pytest.raises(RuntimeError):
        v.restore_episodes(traces, cases)


def metrics():
    row = {"coverage": .45, "expected_actionable_coverage": .40, "agreement": .97,
           "kl": .20, "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0}
    return {"feedback": deepcopy(row), "explanation": deepcopy(row)}


@pytest.mark.parametrize("section,field,value", [
    ("feedback", "coverage", .449), ("explanation", "coverage", .449),
    ("feedback", "expected_actionable_coverage", .399), ("feedback", "agreement", .949),
    ("explanation", "agreement", .969), ("feedback", "kl", .201), ("explanation", "kl", .201),
    ("feedback", "whole_tree_raw_illegal_count", 1), ("explanation", "whole_tree_path_uncertified_count", 1),
    ("explanation", "agreement", None), ("feedback", "coverage", float("nan")),
    ("feedback", "kl", float("inf")), ("feedback", "whole_tree_raw_illegal_count", False),
])
def test_role_fold_gate_requires_every_real_finite_threshold(section, field, value):
    assert v.gate_metrics(metrics())["passed"]
    row = metrics()
    row[section][field] = value
    assert not v.gate_metrics(row)["passed"]


def test_diagnostics_protocol_has_eighteen_unique_jobs_global_budgets_and_no_downstream_authority():
    jobs = v.jobs()
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 18
    assert {j["arm"] for j in jobs} == {"A", "B", "C"}
    for arm, contract in v.round_budget_contract().items():
        assert contract["maximum_total_rounds"] == 60
        assert contract["maximum_initializations"] == 8
        assert contract["maximum_leaf_api_calls_per_role_fit"] == 20000
        assert contract["reserved_final_fixed_topology_leaf_calls"] == 16
        assert contract["scope"] == "per_role_fit_across_all_initializations_including_final_refit"
    recipe = v.executable_recipe()
    assert recipe["environment_resets"] == recipe["actor_updates"] == recipe["PPO_joint_steps"] == 0
    assert recipe["publishable"] is recipe["admission_evidence"] is recipe["fresh_validation_evidence"] is False
    assert set(recipe["future_stages_closed"]) == {"new_cohort", "BC", "PPO", "formal_training", "sealed_audit", "holdout", "pilot"}
    assert recipe["selection"]["infeasible_compiled_topology_is_scientific_failure"]
    assert recipe["selection"]["requires_all_eighteen_completed_or_scientific_infeasible"]


def test_pinned_input_and_recipe_reject_byte_or_semantic_tampering(tmp_path):
    file = tmp_path/"input.json"
    file.write_bytes(b"original")
    expected = sha256(file.read_bytes()).hexdigest()
    assert v.read_pinned(file, expected) == b"original"
    file.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="changed"):
        v.read_pinned(file, expected)
    path = tmp_path/v.RECIPE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(v.executable_recipe()))
    assert v.validate_recipe(tmp_path) == sha256(path.read_bytes()).hexdigest()
    bad = v.executable_recipe()
    bad["gates"]["minimum_feedback_coverage"] = .20
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="recipe drifted"):
        v.validate_recipe(tmp_path)
    assert v.GATES["minimum_feedback_coverage"] == .45
    bad = v.executable_recipe()
    bad["PPO_joint_steps"] = 1
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="recipe drifted"):
        v.validate_recipe(tmp_path)


def test_full_frozen_nn_targets_are_recomputed_and_mismatch_rejected():
    traces, cases = toy_trace()
    episodes = v.restore_episodes(traces, cases)
    class Actor:
        def logits(self, obs, masks):
            return np.broadcast_to(np.log(np.asarray([.6, .1, .1, .1, .1], dtype=np.float32)), masks.shape)
    result = v.target_identity(Actor(), episodes)
    assert result["physical_frames_per_role"] == 6
    assert result["ordering"] == "ascending_base_then_physical_frame_then_role"
    episodes[8]["robot_1"][0]["nn_probabilities"] = np.asarray([.5, .2, .1, .1, .1], np.float32)
    with pytest.raises(RuntimeError, match="full masked NN"):
        v.target_identity(Actor(), episodes)
