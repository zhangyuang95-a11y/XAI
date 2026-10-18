"""Offline synthetic contracts for constrained distribution-tree search."""
from dataclasses import replace

import numpy as np
import pytest

from backend.training import warehouse_v619_tree as search
from backend.training.warehouse_v619_leaf_fit import float32_preference_proof
from backend.training.warehouse_v6_tree_routing import (
    certified_leaf_actions, leaf_indices,
)


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
NAMES = ("self_row", "self_column", "public_hard_risk_context",
         *(f"legal_{action.lower()}" for action in ACTIONS))


def _data(n=240, *, split=False, variable_masks=False):
    x = np.zeros((n, len(NAMES)), np.float32)
    x[:, 0] = np.tile(np.arange(24)/24, n//24)
    mask = np.ones((n, 5), np.float32)
    preferred = np.where(x[:, 0] < .5, 2, 3) if split else np.full(n, 4)
    if variable_masks:
        mask[x[:, 0] >= .5, 2] = 0.
        mask[x[:, 0] < .5, 3] = 0.
    x[:, 3:] = mask
    logits = np.zeros((n, 5), np.float32)
    logits[np.arange(n), preferred] = 5.
    target = search.masked_probabilities(logits, mask).astype(np.float32)
    return x, mask, target, [f"fit:{index//24}" for index in range(n)]


def _config(**changes):
    return replace(search.PRODUCTION_SEARCH_CONFIG, initializations=1,
                   max_rounds=3, max_leaf_optimization_calls=300,
                   proposals_per_round=24, axis_features=2,
                   semantic_candidates_per_node=2, **changes)


def _fit(data, config=None):
    return search.fit_legal_structure(*data, feature_names=NAMES,
        actor_role_sha256="a"*64, actions=ACTIONS, config=config)


def _proof(structure, x, mask):
    count = len(structure["split_weights"])
    proof = certified_leaf_actions(structure["split_weights"], structure["split_bias"],
        structure["left_refs"], structure["right_refs"], count,
        structure["leaf_count"], tuple(range(3, 8)), explicit=True, wait_action=4)
    z = structure["initial_leaf_logits"]
    for index, legal in enumerate(proof):
        assert float32_preference_proof(z[index], int(z[index].argmax()), legal)["passed"]
    routes = leaf_indices(x, structure["split_weights"], structure["split_bias"],
        structure["left_refs"], structure["right_refs"], count, 5)
    chosen = z[routes].argmax(-1)
    assert np.all(mask[np.arange(len(mask)), chosen] == 1)
    return routes, chosen


def test_full_soft_distribution_and_eight_production_initializations():
    data = list(_data())
    data[0][:, :3] = 0.  # No information-bearing split is available.
    target = np.array([.08, .07, .12, .13, .60], np.float32)
    data[2] = np.tile(target, (len(data[0]), 1))
    structure, report = _fit(data)
    assert structure["leaf_count"] == 1
    q = search.masked_probabilities(structure["initial_leaf_logits"], np.ones((1, 5)))
    np.testing.assert_allclose(q[0], target, atol=.002)
    assert report["production_configuration"]
    assert len(report["search_runs"]) == 8
    assert report["leaf_optimization_calls"] == 1
    assert report["actual_convex_solver_calls"] == 1
    assert report["leaf_optimization_calls_remaining"] == 19999
    assert report["fit_expected_feedback_coverage"] == pytest.approx(.6)
    assert report["fit_masked_kl"] < .001
    assert not report["calibration_eligibility_validation_used"]
    assert "bounded_heuristic" in report["search_completeness"]
    _proof(structure, data[0], data[1])


def test_joint_semantic_guards_count_empty_leaves_and_fit_both_movements():
    data = _data(split=True)
    structure, report = _fit(data, _config(max_depth=2, max_leaves=4))
    routes, chosen = _proof(structure, data[0], data[1])
    assert np.array_equal(chosen, data[2].argmax(-1))
    assert structure["leaf_count"] == 4  # Includes two empty physical branches.
    assert len(structure["split_weights"]) == 3
    assert len(np.unique(routes)) == 2
    assert report["fit_qualified_leaf_coverage"] == 1.
    assert report["fit_expected_feedback_coverage"] > .95
    assert report["semantic_guard_joint_proposals"]
    assert report["fit_whole_tree_raw_illegal_count"] == 0
    assert all(leaf["raw_action"] in leaf["path_legal_actions"]
               for leaf in report["fit_leaf_reports"])
    assert all(not leaf["fit_qualified"] for leaf in report["fit_leaf_reports"]
               if leaf["rows"] == 0)


def test_path_proofs_hold_for_unseen_physical_mask_combinations():
    data = _data(split=True, variable_masks=True)
    structure, report = _fit(data, _config(max_depth=3, max_leaves=6))
    patterns = np.array([[int(bool(bits & (1 << a))) for a in range(4)]+[1]
                         for bits in range(16)], np.float32)
    mask = np.repeat(patterns, 3, axis=0)
    x = np.zeros((len(mask), len(NAMES)), np.float32)
    x[:, 0] = np.tile([0., .5, 1.], len(patterns))
    x[:, 3:] = mask
    _proof(structure, x, mask)
    assert report["fit_whole_tree_path_uncertified_count"] == 0


def test_constant_movement_requires_a_real_guard_even_with_all_fit_masks_legal():
    data = list(_data())
    data[0][:, :3] = 0.
    data[2] = np.tile([.01, .01, .01, .96, .01], (len(data[0]), 1)).astype(np.float32)
    structure, report = _fit(data, _config(max_depth=1, max_leaves=2))
    assert structure["leaf_count"] == 2
    np.testing.assert_array_equal(np.flatnonzero(structure["split_weights"][0]), [6])
    routes, chosen = _proof(structure, data[0], data[1])
    assert np.all(chosen == ACTIONS.index("RIGHT"))
    assert len(np.unique(routes)) == 1  # The second leaf is charged, not omitted.
    assert report["fit_qualified_leaf_coverage"] == 1.
    assert report["fit_expected_feedback_coverage"] == pytest.approx(.96)


def test_full_tree_still_evaluates_refit_prune_and_budget_reallocation():
    structure, report = _fit(_data(split=True), _config(max_depth=3, max_leaves=4))
    assert structure["leaf_count"] <= 4
    operations = report["operation_complete_evaluations"]
    assert operations["grow"] > 0
    assert operations["refit"] > 0
    assert operations["prune"] > 0
    assert operations["reallocate"] > 0
    for run in report["search_runs"]:
        for update in run["rounds"]:
            if update["accepted"]:
                assert search._improves(tuple(update["score"]), tuple(update["previous_score"]))


def test_budget_interruption_keeps_complete_prior_tree_and_counts_all_calls(monkeypatch):
    real = search.fit_constrained_leaf_distribution
    observed = []

    def counted(target, masks, legal_actions, **kwargs):
        observed.append(tuple(legal_actions))
        return real(target, masks, legal_actions, **kwargs)

    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", counted)
    config = replace(_config(), max_leaf_optimization_calls=2)
    data = _data(split=True)
    structure, report = _fit(data, config)
    assert report["leaf_optimization_calls"] == len(observed) == 2
    assert report["actual_convex_solver_calls"] == sum(map(len, observed))
    assert report["leaf_optimization_calls_remaining"] == 0
    assert report["search_runs"][0]["stop_reason"] == "leaf_optimization_call_budget"
    assert structure["leaf_count"] == 1
    _proof(structure, data[0], data[1])
    assert report["fit_qualified_leaf_coverage"] == 0.


def test_solver_failure_is_rejected_and_actual_attempts_are_counted(monkeypatch):
    real = search.fit_constrained_leaf_distribution
    errors = []

    def fail_movement(target, masks, legal_actions, **kwargs):
        if len(legal_actions) > 1:
            error = RuntimeError("synthetic converged-solution failure")
            error.enumerated_action_solves = len(legal_actions)
            errors.append(len(legal_actions))
            raise error
        return real(target, masks, legal_actions, **kwargs)

    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", fail_movement)
    data = _data(split=True)
    structure, report = _fit(data, replace(_config(), max_leaf_optimization_calls=30))
    assert report["rejected_solver_candidates"] > 0
    assert report["actual_convex_solver_calls"] >= sum(errors)
    assert report["solver_failure_examples"]
    assert report["fit_qualified_leaf_coverage"] == 0.
    _proof(structure, data[0], data[1])


def test_deterministic_replay_and_input_arrays_unchanged():
    data = _data(split=True)
    before = [item.copy() for item in data[:3]]
    config = _config(max_depth=2, max_leaves=4)
    first, first_report = _fit(data, config)
    second, second_report = _fit(data, config)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first_report == second_report
    for item, saved in zip(data[:3], before, strict=True):
        np.testing.assert_array_equal(item, saved)


def test_source_feature_cost_includes_compound_risk_and_guards():
    weight = np.zeros(len(NAMES), np.float32)
    weight[[0, 2]] = 1.
    assert search.source_feature_cost(weight, NAMES) == 3
    weight[3] = 1.
    assert search.source_feature_cost(weight, NAMES) == 4
    structure, report = _fit(_data(split=True), _config())
    assert report["maximum_source_features_per_node"] <= 3
    assert all(search.source_feature_cost(weight, NAMES) <= 3
               for weight in structure["split_weights"])


@pytest.mark.parametrize("change", [{"initializations": 9}, {"max_rounds": 31},
    {"max_leaf_optimization_calls": 20001}, {"max_depth": 6}, {"max_leaves": 17},
    {"max_rounds": True}])
def test_diagnostic_config_cannot_expand_registered_production_budget(change):
    with pytest.raises(ValueError, match="invalid constrained tree budget"):
        replace(search.PRODUCTION_SEARCH_CONFIG, **change)


def test_inconsistent_masks_or_unmasked_targets_are_rejected():
    data = list(_data(split=True, variable_masks=True))
    data[0] = data[0].copy()
    data[0][0, 3] = 0.
    with pytest.raises(ValueError, match="complete masked NN"):
        _fit(data, _config())
    data = list(_data(split=True, variable_masks=True))
    data[2] = data[2].copy()
    data[2][0] = [.2]*5
    with pytest.raises(ValueError, match="complete masked NN"):
        _fit(data, _config())
