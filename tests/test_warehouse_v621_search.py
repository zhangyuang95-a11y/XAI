"""Offline synthetic checks for the separate v6.21 A/C search protocol."""
from dataclasses import replace
from hashlib import sha256
import inspect

import numpy as np
import pytest

from backend.training import warehouse_v621_search as search
from backend.training import warehouse_program_search as legacy
from backend.training.warehouse_v619_leaf_fit import float32_preference_proof
from backend.training.warehouse_v6_tree_routing import certified_leaf_actions, leaf_indices

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
NAMES = ("position", "context", "public_hard_risk_context",
         *(f"legal_{action.lower()}" for action in ACTIONS))


def rows(n=480, *, split=False, varying_masks=False):
    x = np.zeros((n, len(NAMES)), np.float32)
    x[:, 0] = np.tile(np.arange(24)/24, n//24)
    m = np.ones((n, 5), np.float32)
    if varying_masks:
        m[x[:, 0] < .5, 3] = 0.
        m[x[:, 0] >= .5, 2] = 0.
    x[:, 3:] = m
    chosen = np.where(x[:, 0] < .5, 2, 3) if split else np.full(n, 4)
    z = np.zeros((n, 5), np.float32)
    z[np.arange(n), chosen] = 5.
    p = legacy.masked_probabilities(z, m).astype(np.float32)
    return x, m, p, [f"fit:{i//24}" for i in range(n)]


def cfg(**changes):
    settings = dict(initializations=1, max_rounds=3,
        max_leaf_optimization_calls=250, proposals_per_round=12,
        axis_features=2, semantic_candidates_per_node=2)
    settings.update(changes)
    return replace(search.PRODUCTION_SEARCH_CONFIG, **settings)


def fit(arm="C", data=None, config=None):
    return search.fit_structure(arm, *(rows() if data is None else data),
        feature_names=NAMES, actor_role_sha256="a"*64,
        config=cfg() if config is None else config)


def prove(structure, x, masks):
    w = structure["split_weights"]
    proofs = certified_leaf_actions(w, structure["split_bias"], structure["left_refs"],
        structure["right_refs"], len(w), structure["leaf_count"], tuple(range(3, 8)),
        explicit=True, wait_action=4)
    for z, legal in zip(structure["initial_leaf_logits"], proofs, strict=True):
        assert float32_preference_proof(z, int(z.argmax()), legal)["passed"]
    route = leaf_indices(x, w, structure["split_bias"], structure["left_refs"],
                         structure["right_refs"], len(w), 5)
    choice = structure["initial_leaf_logits"][route].argmax(-1)
    assert np.all(masks[np.arange(len(x)), choice] == 1)
    assert all(search.source_feature_cost(a, NAMES) <= 3 for a in w)
    return choice


def test_new_configuration_explicitly_extends_budget_without_mutating_legacy():
    assert search.PRODUCTION_SEARCH_CONFIG.max_rounds == 60
    assert search.PRODUCTION_SEARCH_CONFIG.max_leaf_optimization_calls == 19984
    assert legacy.PRODUCTION_SEARCH_CONFIG.max_rounds == 30
    assert legacy.PRODUCTION_SEARCH_CONFIG.max_leaf_optimization_calls == 20000
    with pytest.raises(ValueError):
        legacy.SearchConfig(max_rounds=60)
    for change in ({"max_rounds": 61}, {"max_leaf_optimization_calls": 19985},
                   {"initializations": 9}, {"max_depth": 6}, {"max_leaves": 17}):
        with pytest.raises(ValueError):
            cfg(**change)


@pytest.mark.parametrize("arm", ["A", "C"])
def test_global_round_and_end_to_end_leaf_budget_reserve(arm):
    structure, report = fit(arm, config=cfg(initializations=8, max_rounds=7,
                                          max_leaf_optimization_calls=160))
    used = sum(len(run["rounds"]) for run in report["search_runs"])
    assert len(report["search_runs"]) == 8
    assert used == report["global_search_rounds"] <= 7
    assert report["leaf_optimization_calls"] <= 160
    assert report["final_refit_reserved_calls"] == 16
    assert report["maximum_end_to_end_calls"] == 176 <= 20000
    assert report["end_to_end_leaf_api_ceiling"] == 20000
    assert not report["calibration_eligibility_validation_used"]
    assert report["topology_uses_fit_only"]
    prove(structure, *rows()[:2])


def test_A_really_delegates_unchanged_M2_with_new_explicit_configuration():
    data, config = rows(), cfg(max_rounds=1)
    original_before = sha256(inspect.getsource(legacy).encode()).hexdigest()
    structure, report = fit("A", data, config)
    expected, original = legacy.fit_legal_structure(*data, feature_names=NAMES,
        actor_role_sha256="a"*64, method=legacy.ROBUST_METHOD, config=config)
    for key in structure:
        np.testing.assert_array_equal(structure[key], expected[key])
    assert report["legacy_method"] == legacy.ROBUST_METHOD
    assert report["leaf_optimization_calls"] == original["leaf_optimization_calls"]
    assert original_before == sha256(inspect.getsource(legacy).encode()).hexdigest()


def test_C_fits_full_soft_distributions_and_charges_empty_guard_branches():
    data = rows(split=True)
    structure, report = fit("C", data, cfg(max_depth=2, max_leaves=4))
    choices = prove(structure, *data[:2])
    np.testing.assert_array_equal(choices, data[2].argmax(-1))
    assert structure["leaf_count"] == 4
    assert any(leaf["rows"] == 0 for leaf in report["fit_leaf_reports"])
    assert report["fit_stable_explanation_coverage"] == 1.
    assert report["fit_expected_feedback_coverage"] > .95
    assert report["node_updates_use_counterfactual_subtree_costs"]
    assert report["operation_complete_evaluations"]["grow"] > 0
    assert report["operation_complete_evaluations"]["cost_sensitive_node_update"] > 0


def test_counterfactual_costs_distinguish_identical_argmax_full_distributions():
    # Both target rows choose UP, yet their preferred complete distributions
    # point to opposite subtrees. Hard labels could not produce this signal.
    target = np.array([[.51, .45, .01, .01, .02], [.51, .01, .45, .01, .02]])
    masks = np.ones_like(target)
    left = np.broadcast_to(np.log(target[0]), target.shape)
    right = np.broadcast_to(np.log(target[1]), target.shape)
    lc = search.complete_distribution_losses(target, masks, left)
    rc = search.complete_distribution_losses(target, masks, right)
    assert np.all(target.argmax(-1) == 0)
    assert lc[0] < rc[0] and rc[1] < lc[1]
    expected = np.sum(target*(np.log(target)-np.log(legacy.masked_probabilities(left, masks))), axis=-1)
    np.testing.assert_allclose(lc, expected, atol=1.e-14)
    x = np.zeros((80, len(NAMES)), np.float32)
    x[:, 0] = np.repeat([0., 1.], 40)
    candidates = search._cost_sensitive_predicates(x, NAMES, np.arange(3),
        np.repeat(lc, 40), np.repeat(rc, 40), cfg(), (.5,))
    assert candidates
    cost, _, w, bias = candidates[0]
    assert cost < .001
    assert search.source_feature_cost(w, NAMES) <= 3
    right_route = x@w+bias > search.HARD_SPLIT_TIE_EPSILON
    np.testing.assert_array_equal(right_route, x[:, 0] == 1.)


def test_C_certificates_hold_for_unseen_physical_mask_patterns():
    structure, report = fit("C", rows(split=True, varying_masks=True),
                            cfg(max_depth=3, max_leaves=6))
    masks = np.array([[int(bool(bits & (1 << a))) for a in range(4)]+[1]
                      for bits in range(16)], np.float32)
    masks = np.repeat(masks, 3, axis=0)
    x = np.zeros((len(masks), len(NAMES)), np.float32)
    x[:, 0] = np.tile([0., .5, 1.], 16)
    x[:, 3:] = masks
    prove(structure, x, masks)
    assert report["fit_whole_tree_path_uncertified_count"] == 0
    assert report["guard_predicate_updates"] == "immutable"


def test_C_failed_leaf_attempts_are_spent_and_recorded(monkeypatch):
    original, observed = search.fit_constrained_leaf_distribution, []
    def solver(target, masks, legal, **kwargs):
        observed.append(tuple(legal))
        if len(legal) > 1:
            error = RuntimeError("synthetic rejection")
            error.enumerated_action_solves = len(legal)
            raise error
        return original(target, masks, legal, **kwargs)
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", solver)
    structure, report = fit("C", rows(split=True), cfg(max_leaf_optimization_calls=12))
    assert report["leaf_optimization_calls"] == len(observed)
    assert report["actual_convex_solver_calls"] == sum(map(len, observed))
    assert len(report["leaf_api_attempts"]) == len(observed)
    assert report["rejected_solver_candidates"] > 0
    assert any(not item["success"] for item in report["leaf_api_attempts"])
    prove(structure, *rows(split=True)[:2])


def test_C_partial_candidate_cannot_replace_completed_safe_root(monkeypatch):
    original, observed = search.fit_constrained_leaf_distribution, []
    def solver(*args, **kwargs):
        observed.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", solver)
    data = rows(split=True)
    structure, report = fit("C", data, cfg(max_leaf_optimization_calls=2))
    assert report["leaf_optimization_calls"] == len(observed) == 2
    assert structure["leaf_count"] == 1
    prove(structure, *data[:2])


@pytest.mark.parametrize("arm", ["A", "C"])
def test_reject_unmasked_or_hard_label_inputs_and_bind_full_distribution(arm):
    data = list(rows())
    with pytest.raises(ValueError):
        fit(arm, [*data[:2], data[2].argmax(-1), data[3]])
    data[1][:, 0] = 0
    data[0][:, 3] = 0
    with pytest.raises(ValueError):
        fit(arm, data)
    _, report = fit(arm)
    assert report["fit_target_sha256"] == sha256(np.ascontiguousarray(rows()[2], dtype="<f4").tobytes()).hexdigest()


def test_no_environment_or_actor_loader_imports_in_new_search():
    source = inspect.getsource(search)
    assert "import warehouse_env" not in source
    assert "WarehouseEnv(" not in source
    assert "load_actor" not in source
    assert not search.PRODUCTION_SEARCH_CONFIG.max_leaf_optimization_calls+16 > 20000


def test_C_post_solver_float32_rejection_keeps_actual_solver_count(monkeypatch):
    original, observed = search.fit_constrained_leaf_distribution, []
    def solver(target, masks, legal, **kwargs):
        observed.append(tuple(legal))
        z, report = original(target, masks, legal, **kwargs)
        if len(legal) > 1:
            z = np.zeros(5, np.float32)  # Deliberate invalid strict preference.
        return z, report
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", solver)
    structure, report = fit("C", rows(split=True), cfg(max_leaf_optimization_calls=12))
    assert report["actual_convex_solver_calls"] == sum(map(len, observed))
    assert sum(event["actual_convex_solver_calls"] for event in report["leaf_api_attempts"]) == report["actual_convex_solver_calls"]
    assert report["leaf_api_cache_misses"] == len(observed)
    assert report["failed_leaf_api_attempts"] > 0
    prove(structure, *rows(split=True)[:2])


def test_C_growth_can_separate_soft_modes_with_identical_NN_argmax():
    data = list(rows())
    data[2][data[0][:, 0] < .5] = [.51, .45, .01, .01, .02]
    data[2][data[0][:, 0] >= .5] = [.51, .01, .45, .01, .02]
    assert np.all(data[2].argmax(-1) == 0)
    structure, report = fit("C", data, cfg(max_depth=2, max_leaves=4))
    prove(structure, *data[:2])
    assert structure["leaf_count"] == 4
    assert report["fit_stable_explanation_coverage"] == 1.
    assert report["fit_masked_kl"] < .01
    assert any("soft_distribution_principal_modes" in item["label"]
        for run in report["search_runs"] for update in run["rounds"]
        for item in update["accepted_updates"])
