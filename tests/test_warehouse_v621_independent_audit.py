"""Synthetic-only audit tests; no saved weights, fit, reset, or training."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_warehouse_v621_diagnostics as a
from experiments.warehouse import v621_protocol as p

NAMES = ("same_target_collision_streak_capped", "robots_shortest_distance",
         "self_charger_return_margin", "public_hard_risk_context",
         *(f"legal_{action.lower()}" for action in a.ACTIONS))


def toy():
    m = np.asarray([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 1]], np.float32)
    x = np.ones((6, len(NAMES)), np.float32)
    x[:, 0] = 0
    x[2, 0] = .4
    x[4, 1] = 0
    x[0, 2] = 0
    x[:, 4:] = m
    target = np.asarray([[0, 0, 0, 0, 1], [.8, 0, 0, 0, .2], [.4, 0, 0, 0, .6],
                         [.7, 0, 0, 0, .3], [0, 0, 0, 0, 1], [.9, 0, 0, 0, .1]], np.float32)
    sampled = np.asarray([4, 0, 0, 4, 4, 0])
    episodes = np.asarray(["e0", "e1", "e1", "e2", "e2", "e3"])
    scenes = np.asarray(["10", "20", "20", "30", "30", "40"])
    w = np.zeros((1, len(NAMES)), np.float32)
    w[0, 4] = 1
    raw = {"root_ref": 0, "split_weights": w.tolist(), "split_bias": [-.5],
           "left_refs": [-1], "right_refs": [-2],
           "leaf_logits": [[0, 0, 0, 0, 4], [4, 0, 0, 0, 0]],
           "feedback_leaves": [1], "explanation_leaves": [1]}
    return raw, (x, m, target, sampled, episodes, scenes)


def test_independent_counting_keeps_all_frames_and_distinguishes_feedback_from_explanation():
    raw, data = toy()
    result, c = a.metrics(raw, data, NAMES)
    f, e = result["feedback"], result["explanation"]
    assert result["all_physical_frames"] == f["sample_count"] == e["sample_count"] == 6
    assert f["rows"] == e["rows"] == 4
    assert f["coverage"] == e["coverage"] == 4/6
    assert f["agreement"] == e["agreement"] == .75
    assert f["actionable_rows"] == 2
    assert e["actionable_rows"] == 3
    assert f["actionable_coverage"] == 2/6
    assert e["actionable_coverage"] == 3/6
    assert f["expected_actionable_coverage"] == pytest.approx(.4)
    assert e["expected_actionable_coverage"] == pytest.approx(2.8/6)
    assert f["whole_tree_raw_illegal_count"] == f["whole_tree_path_uncertified_count"] == 0
    assert f["distinct_episode_count"] == f["distinct_scenario_seed_count"] == 3
    assert f["max_episode_fraction"] == .5
    assert f["per_nn_action"]["UP"]["physical_frames"] == 3
    assert f["per_nn_action"]["WAIT"]["physical_frames"] == 3
    assert f["public_strata"]["conflict"]["physical_frames"] == 1
    assert f["public_strata"]["near_robots"]["physical_frames"] == 1
    assert f["public_strata"]["low_return_margin"]["physical_frames"] == 1
    assert f["masked_illegal_count"] == 0
    q = np.asarray([np.exp(4)/(np.exp(4)+1), 1/(np.exp(4)+1)])
    expected = np.mean([sum(v*(np.log(v)-np.log(q[j])) for j, v in enumerate(pair))
                        for pair in ((.8, .2), (.4, .6), (.7, .3), (.9, .1))])
    assert f["kl"] == pytest.approx(expected, abs=1.e-6)
    np.testing.assert_array_equal(c["leaves"], [0, 1, 1, 1, 0, 1])


def test_whole_raw_illegality_precedes_physical_mask_and_qualification():
    raw, data = toy()
    raw["leaf_logits"][0] = [4, 0, 0, 0, 0]
    result, _ = a.metrics(raw, data, NAMES)
    f = result["feedback"]
    assert f["whole_tree_raw_illegal_count"] == 2
    assert f["whole_tree_path_uncertified_count"] == 2
    assert f["illegal_count"] == f["path_uncertified_count"] == f["masked_illegal_count"] == 0
    assert result["nn_five_action_strata"]["WAIT"]["whole_tree_raw_illegal_count"] == 2


def test_empty_qualification_does_not_hide_whole_tree_illegal_frames_or_invent_kl():
    raw, data = toy()
    raw["feedback_leaves"] = raw["explanation_leaves"] = []
    raw["leaf_logits"][0] = [4, 0, 0, 0, 0]
    result, _ = a.metrics(raw, data, NAMES)
    for purpose in ("feedback", "explanation"):
        row = result[purpose]
        assert row["rows"] == row["actionable_rows"] == 0
        assert row["coverage"] == row["expected_actionable_coverage"] == 0.
        assert row["kl"] is None
        assert row["whole_tree_raw_illegal_count"] == 2
        assert row["per_nn_action"] == row["public_strata"] == {}


def test_independent_sparse_route_uses_float32_coefficients_float64_dot_and_left_ties():
    raw, data = toy()
    tree, _ = a.structure(dict(raw, leaf_count=2), NAMES)
    x = np.zeros((3, len(NAMES)), np.float32)
    x[:, 4] = [.5, np.nextafter(np.float32(.5), np.float32(0)), np.nextafter(np.float32(.5), np.float32(1))]
    np.testing.assert_array_equal(a.route(tree, x), [0, 0, 1])
    assert a.certificates(tree, NAMES) == {0: frozenset({4}), 1: frozenset({0, 4})}


@pytest.mark.parametrize("kind", ["cycle", "unreachable", "fractional", "source_cost", "depth"])
def test_structure_audit_rejects_graph_and_capacity_defects(kind):
    raw, _ = toy()
    raw["leaf_count"] = 2
    if kind == "cycle":
        raw["left_refs"] = [0]
    elif kind == "unreachable":
        raw["right_refs"] = [-1]
    elif kind == "fractional":
        raw["right_refs"] = [-1.5]
    elif kind == "source_cost":
        raw["split_weights"][0][0] = raw["split_weights"][0][1] = raw["split_weights"][0][3] = 1
    else:
        raw = {"root_ref": 0, "split_weights": [raw["split_weights"][0]]*6,
               "split_bias": [-.5]*6, "left_refs": [-1, -2, -3, -4, -5, -6],
               "right_refs": [1, 2, 3, 4, 5, -7], "leaf_count": 7}
    with pytest.raises(RuntimeError):
        a.structure(raw, NAMES)


def test_fixed_partition_oracle_is_observed_row_bound_and_tracks_empty_leaf():
    raw, data = toy()
    tree = dict(raw, leaf_count=2)
    result = a.oracle(tree, data, NAMES)
    assert result["rows"] == 6
    assert result["global_tree_optimality_claimed"] is False
    assert result["empirical_legality_is_not_a_path_certificate"] is True
    assert result["bounds"]["path_certified"]["matched_rows"] == 5
    assert result["bounds"]["path_certified"]["maximum_constant_action_agreement"] == 5/6
    assert result["leaves"][1]["nn_argmax_counts"] == [3, 0, 0, 0, 1]
    assert result["leaves"][1]["path_certified"]["best_constant_action"] == "UP"


def test_count_and_structure_comparison_is_exact_but_float_reduction_tolerance_is_bounded():
    a.assert_equal({"n": 6, "v": .123456}, {"n": 6, "v": .123457})
    with pytest.raises(RuntimeError):
        a.assert_equal({"n": 6}, {"n": 6.})
    with pytest.raises(RuntimeError):
        a.assert_equal({"n": False}, {"n": 0})
    with pytest.raises(RuntimeError):
        a.assert_equal({"v": .12}, {"v": .13})
    with pytest.raises(RuntimeError):
        a.assert_equal({"v": float("nan")}, {"v": float("nan")})


def test_batch_audit_refuses_partial_results_before_loading_weights_or_fitting(tmp_path):
    path = tmp_path/"plan.json"
    path.write_text(json.dumps({"jobs": p.jobs()}))
    with pytest.raises(RuntimeError, match="all eighteen"):
        a.require_complete(path)
    for job in p.jobs():
        directory = tmp_path/"jobs"/job["job_id"]
        directory.mkdir(parents=True)
        for name in ("started.json", "result.json", "completed.json"):
            (directory/name).write_text("{}")
    with pytest.raises(RuntimeError, match="summary"):
        a.require_complete(path)
    (tmp_path/"summary.json").write_text("{}")
    assert a.require_complete(path)[0] == path
    (tmp_path/"jobs"/p.jobs()[0]["job_id"]/"failed.json").write_text("{}")
    with pytest.raises(RuntimeError, match="all eighteen"):
        a.require_complete(path)


def test_auditor_source_never_calls_native_metrics_qualification_routing_or_fit():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(a.__file__).read_text())
    forbidden = {"program_metrics", "_metrics", "_metric_context", "fit_program", "fit_structure",
                 "leaf_indices", "fixed_topology_action_oracle", "_eligible", "compile_actor_structure"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            assert name not in forbidden


def test_program_interval_proof_and_oracle_explicit_crossing_have_distinct_contracts():
    raw, _ = toy()
    raw["split_bias"] = [.5]
    tree, _ = a.structure(dict(raw, leaf_count=2), NAMES)
    # Left branch is unreachable for either legal-bit value. The program path
    # interval proof is vacuous, while the compiler/oracle requires a real test.
    assert a.certificates(tree, NAMES)[0] == frozenset({0, 4})
    assert a.certificates(tree, NAMES, explicit_crossing=True)[0] == frozenset({4})


def test_compiler_audit_reconstructs_guards_from_actor_weights_and_rejects_changed_guard():
    empty = np.empty((0, len(NAMES)), np.float32)
    actor = SimpleNamespace(feature_names=NAMES, weights={
        "tree.node_count": np.asarray([0], np.float32), "tree.leaf_count": np.asarray([1], np.float32),
        "tree.split.weight": empty, "tree.split.bias": np.empty(0, np.float32),
        "tree.left_refs": np.empty(0, np.float32), "tree.right_refs": np.empty(0, np.float32),
        "tree.leaf_logits": np.asarray([[4, 0, 0, 0, 0]], np.float32)})
    original = {"root_ref": -1, "split_weights": empty, "split_bias": np.empty(0, np.float32),
                "left_refs": np.empty(0, np.int64), "right_refs": np.empty(0, np.int64),
                "leaf_count": 1, "feature_dim": len(NAMES)}
    raw, _ = toy()
    guarded = {k: raw[k] for k in ("root_ref", "split_weights", "split_bias", "left_refs", "right_refs")}
    guarded.update(leaf_count=2, feature_dim=len(NAMES))
    report = {
        "original_structure": a.plain(original), "original_structure_sha256": a.digest(original),
        "original_capacity": {"node_count": 0, "leaf_count": 1, "depth": 0,
            "maximum_source_features_per_node": 0, "source_features_per_node": [], "leaf_paths": {"0": []}},
        "required_guarded_structure": guarded, "required_guarded_structure_sha256": a.digest(guarded),
        "required_capacity": {"node_count": 1, "leaf_count": 2, "depth": 1,
            "maximum_source_features_per_node": 1, "source_features_per_node": [1],
            "leaf_paths": {"0": [{"node": 0, "branch": "left"}], "1": [{"node": 0, "branch": "right"}]}},
        "compiled_leaf_origins": [
            {"compiled_leaf": 0, "original_leaf": 0, "frozen_preferred_action": "WAIT",
             "guard_branch": "illegal_false_wait", "path_legal_actions": ["WAIT"],
             "compiled_path": [{"node": 0, "branch": "left"}]},
            {"compiled_leaf": 1, "original_leaf": 0, "frozen_preferred_action": "UP",
             "guard_branch": "legal_true", "path_legal_actions": ["UP", "WAIT"],
             "compiled_path": [{"node": 0, "branch": "right"}]}],
        "guard_count": 1, "infeasibility_reasons": [], "feasible": True,
        "guard_policy_uses_observations": False, "all_guards_and_empty_children_counted": True,
        "input_weights_modified": False}
    result, _ = a.compiler_review(report, actor)
    assert result["passed"] and result["feasible"]
    bad = deepcopy(report)
    bad["required_guarded_structure"]["split_weights"][0][4] = .5
    with pytest.raises(RuntimeError):
        a.compiler_review(bad, actor)


def test_budget_audit_counts_all_starts_and_final_refits_in_global_ceiling():
    search = {"search_runs": [{"rounds": [0, 1], "leaf_optimization_calls": 1,
               "actual_convex_solver_calls": 2} for _ in range(8)],
              "leaf_optimization_calls": 8, "global_search_rounds": 16,
              "actual_convex_solver_calls": 16}
    report = {"leaf_optimization_calls_outside_search": 2, "leaf_count": 2,
              "constrained_leaf_fits": {"0": {"enumerated_action_solves": 1}, "1": {"enumerated_action_solves": 2}},
              "convex_solver_calls_outside_search": 3}
    native = {"passed": True, "checks": {"test": True}, "actual_total_rounds": 16,
              "search_leaf_api_calls": 8, "final_refit_leaf_api_calls": 2,
              "end_to_end_leaf_api_calls": 10, "actual_search_convex_solver_calls": 16,
              "actual_final_convex_solver_calls": 3}
    assert a.budget_review(search, report, "C", native)["end_to_end_leaf_api_calls"] == 10
    search["search_runs"][0]["leaf_optimization_calls"] += 1
    with pytest.raises(RuntimeError, match="history sum"):
        a.budget_review(search, report, "C", native)


def test_eligibility_uses_only_supported_cross_episode_leaf_metrics_and_wilson():
    row = {"rows": 100, "distinct_episode_count": 4, "distinct_scenario_seed_count": 4,
           "max_episode_fraction": .4, "agreement": .98, "agreement_wilson_lower_90": .94,
           "kl": .1, "illegal_count": 0, "path_uncertified_count": 0}
    assert a.eligible(row, .93) and a.eligible(row, .95)
    for field, value in (("rows", 19), ("distinct_episode_count", 2), ("distinct_scenario_seed_count", 2),
                         ("max_episode_fraction", .51), ("agreement_wilson_lower_90", .84),
                         ("kl", .251), ("illegal_count", 1), ("path_uncertified_count", 1)):
        assert not a.eligible(dict(row, **{field: value}), .93)
