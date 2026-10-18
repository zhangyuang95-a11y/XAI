"""Synthetic offline tests for the fit-only v6.20 search engine."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_program_search as search
from backend.training.warehouse_v619_leaf_fit import float32_preference_proof
from backend.training.warehouse_v6_tree_routing import certified_leaf_actions, leaf_indices

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
NAMES = ("self_row", "self_column", "public_hard_risk_context",
         *(f"legal_{action.lower()}" for action in ACTIONS))


def data(n=240, *, split=False, masks_vary=False):
    x = np.zeros((n, len(NAMES)), np.float32)
    x[:, 0] = np.tile(np.arange(24)/24, n//24)
    masks = np.ones((n, 5), np.float32)
    actions = np.where(x[:, 0] < .5, 2, 3) if split else np.full(n, 4)
    if masks_vary:
        masks[x[:, 0] < .5, 3] = 0.
        masks[x[:, 0] >= .5, 2] = 0.
    x[:, 3:] = masks
    logits = np.zeros((n, 5), np.float32)
    logits[np.arange(n), actions] = 5.
    probs = search.masked_probabilities(logits, masks).astype(np.float32)
    return x, masks, probs, [f"fit:{index//24}" for index in range(n)]


def config(**changes):
    values = dict(initializations=1, max_rounds=2,
        max_leaf_optimization_calls=300, proposals_per_round=12,
        axis_features=2, semantic_candidates_per_node=2)
    values.update(changes)
    return replace(search.PRODUCTION_SEARCH_CONFIG, **values)


def fit(rows, cfg=None, method=search.QUEUED_METHOD):
    return search.fit_legal_structure(*rows, feature_names=NAMES,
        actor_role_sha256="a"*64, actions=ACTIONS, method=method, config=cfg)


def proof(structure, x, mask):
    weights = structure["split_weights"]
    certificates = certified_leaf_actions(weights, structure["split_bias"],
        structure["left_refs"], structure["right_refs"], len(weights),
        structure["leaf_count"], tuple(range(3, 8)), explicit=True, wait_action=4)
    for logits, legal in zip(structure["initial_leaf_logits"], certificates, strict=True):
        assert float32_preference_proof(logits, int(logits.argmax()), legal)["passed"]
    routes = leaf_indices(x, weights, structure["split_bias"],
        structure["left_refs"], structure["right_refs"], len(weights), 5)
    chosen = structure["initial_leaf_logits"][routes].argmax(-1)
    assert np.all(mask[np.arange(len(mask)), chosen] == 1.)
    return routes, chosen


def test_no_gain_first_page_does_not_discard_remaining_candidates():
    # Every split makes this already-perfect one-leaf WAIT model larger.
    # A page is one proposal; the entire queue must still be evaluated.
    structure, report = fit(data(), config(proposals_per_round=1, max_rounds=1))
    update = report["search_runs"][0]["rounds"][0]
    assert not update["accepted"]
    assert update["complete_evaluations"] > 1
    assert update["pages"] == update["proposals"]
    assert update["complete_evaluations"] == update["proposals"]
    assert update["remaining_queue"] == 0
    assert not update["budget_interrupted"]
    assert structure["leaf_count"] == 1


def test_beam_reserves_non_greedy_transition_and_preserves_best():
    best = SimpleNamespace(key=b"best", score=(1., 0., -1))
    better = SimpleNamespace(key=b"elite", score=(.99, 0., -2))
    transition = SimpleNamespace(key=b"bridge", score=(.2, 0., -4))
    entries = [search._BeamEntry(better, 0), search._BeamEntry(transition, 2)]
    selected = search._choose_beam(entries, best, 2)
    assert [entry.state.key for entry in selected] == [b"best", b"bridge"]
    assert selected[1].nonimproving_steps == 2
    _, report = fit(data(split=True), config(max_rounds=3))
    assert report["historical_best_preserved"]
    assert any(item["nonimproving_steps"] > 0 for run in report["search_runs"]
               for update in run["rounds"] for item in update["beam"])
    assert all(item["nonimproving_steps"] <= 2 for run in report["search_runs"]
               for update in run["rounds"] for item in update["beam"])


def test_eight_requested_initializations_are_different_fit_only_and_budget_reserved():
    rows = list(data())
    rows[0][:, :3] = 0.
    structure, report = fit(rows, config(initializations=8, max_rounds=1,
                                        max_leaf_optimization_calls=160))
    assert len(report["search_runs"]) == 8
    assert report["distinct_requested_initializations"] == 8
    assert len({run["requested_initial_tree_sha256"] for run in report["search_runs"]}) == 8
    assert [run["reserved_cumulative_leaf_call_limit"] for run in report["search_runs"]] == list(range(20, 161, 20))
    assert [run["reserved_cumulative_round_limit"] for run in report["search_runs"]] == [0]*7+[1]
    assert all(not run["rounds"] for run in report["search_runs"][:7])
    assert report["global_search_rounds"] <= 1
    assert not report["calibration_eligibility_validation_used"]
    assert report["topology_uses_fit_only"]
    proof(structure, rows[0], rows[1])


@pytest.mark.parametrize("method", search.METHODS)
@pytest.mark.parametrize("round_budget", [1, 3, 7, 30])
def test_round_budget_is_global_across_eight_starts_including_small_budgets(method, round_budget):
    rows = list(data())
    rows[0][:, :3] = 0.
    _, report = fit(rows, config(initializations=8, max_rounds=round_budget,
                                 max_leaf_optimization_calls=800), method)
    runs = report["search_runs"]
    used = sum(len(run["rounds"]) for run in runs)
    assert len(runs) == 8
    assert report["distinct_requested_initializations"] == 8
    assert report["round_budget_scope"] == "global_per_role_fit_across_all_initializations"
    assert used == report["global_search_rounds"] <= round_budget
    assert report["global_search_rounds_remaining"] == round_budget-used
    assert [run["reserved_cumulative_round_limit"] for run in runs] == [
        (index+1)*round_budget//8 for index in range(8)]
    cumulative = 0
    for run in runs:
        assert run["global_rounds_before_start"] == cumulative
        assert run["round_budget_available_at_start"] == run["reserved_cumulative_round_limit"]-cumulative
        assert len(run["rounds"]) <= run["round_budget_available_at_start"]
        for update in run["rounds"]:
            cumulative += 1
            assert update["global_round"] == cumulative
        assert run["global_rounds_after_start"] == cumulative
        assert cumulative <= run["reserved_cumulative_round_limit"]


def test_early_stop_carries_unused_round_share_and_logs_progress_without_configuring_logging(caplog):
    import logging

    rows = list(data())
    rows[0][:, :3] = 0.
    logger = logging.getLogger(search.__name__)
    handlers = list(logger.handlers)
    with caplog.at_level(logging.INFO, logger=search.__name__):
        _, report = fit(rows, config(initializations=8, max_rounds=30,
                                     max_leaf_optimization_calls=800))
    assert logger.handlers == handlers
    first, second = report["search_runs"][:2]
    assert first["reserved_cumulative_round_limit"] == 3
    assert first["rounds"] == []  # Constant inputs leave no root proposal.
    assert second["global_rounds_before_start"] == 0
    assert second["round_budget_available_at_start"] == 7  # Includes unused first share.
    starts = [record for record in caplog.records if "start_complete" in record.getMessage()]
    rounds = [record for record in caplog.records if "round_complete" in record.getMessage()]
    assert len(starts) == 8
    assert len(rounds) == report["global_search_rounds"]
    assert all(all(field in record.getMessage() for field in (
        "api_calls=", "convex_calls=", "fit_score=", "remaining_queue="))
        for record in starts+rounds)


def test_full_distribution_fits_and_joint_guards_include_empty_branches():
    rows = data(split=True)
    structure, report = fit(rows, config(max_depth=2, max_leaves=4), search.ROBUST_METHOD)
    _, chosen = proof(structure, rows[0], rows[1])
    np.testing.assert_array_equal(chosen, rows[2].argmax(-1))
    assert structure["leaf_count"] == 4
    assert report["fit_stable_explanation_coverage"] == 1.
    assert report["fit_expected_feedback_coverage"] > .95
    assert report["fit_whole_tree_raw_illegal_count"] == 0
    assert all(not leaf["fit_stable_explanation_qualified"] for leaf in report["fit_leaf_reports"]
               if leaf["rows"] == 0)
    assert all(leaf["optimizer"]["success"] for leaf in report["fit_leaf_reports"])


def test_unseen_masks_have_certified_raw_actions_and_source_budget():
    structure, report = fit(data(split=True, masks_vary=True), config(max_depth=3, max_leaves=6))
    patterns = np.array([[int(bool(bits & (1 << action))) for action in range(4)]+[1]
                         for bits in range(16)], np.float32)
    mask = np.repeat(patterns, 3, axis=0)
    x = np.zeros((len(mask), len(NAMES)), np.float32)
    x[:, 0] = np.tile([0., .5, 1.], len(patterns))
    x[:, 3:] = mask
    proof(structure, x, mask)
    weight = np.zeros(len(NAMES), np.float32)
    weight[[0, 2]] = 1.
    assert search.source_feature_cost(weight, NAMES) == 3
    weight[3] = 1.
    assert search.source_feature_cost(weight, NAMES) == 4
    assert all(search.source_feature_cost(weight, NAMES) <= 3
               for weight in structure["split_weights"])
    assert report["fit_whole_tree_path_uncertified_count"] == 0


def test_partial_candidate_is_never_returned_and_spent_leaf_calls_count(monkeypatch):
    real, observed = search.fit_constrained_leaf_distribution, []

    def counted(target, masks, legal, **kwargs):
        observed.append(tuple(legal))
        return real(target, masks, legal, **kwargs)

    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", counted)
    rows = data(split=True)
    structure, report = fit(rows, config(max_leaf_optimization_calls=2))
    assert report["leaf_optimization_calls"] == len(observed) == 2
    assert report["actual_convex_solver_calls"] == sum(map(len, observed))
    assert report["search_runs"][0]["stop_reason"] == "leaf_optimization_call_budget"
    assert report["remaining_proposal_queue"] > 0
    assert structure["leaf_count"] == 1
    proof(structure, rows[0], rows[1])


def test_failed_solver_counts_api_and_actual_enumerations(monkeypatch):
    real, failures = search.fit_constrained_leaf_distribution, []

    def failed(target, masks, legal, **kwargs):
        if len(legal) > 1:
            error = RuntimeError("synthetic solver failure")
            error.enumerated_action_solves = len(legal)
            failures.append(len(legal))
            raise error
        return real(target, masks, legal, **kwargs)

    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", failed)
    rows = data(split=True)
    structure, report = fit(rows, config(max_leaf_optimization_calls=20))
    assert report["rejected_solver_candidates"] > 0
    assert report["leaf_optimization_calls"] <= 20
    assert report["actual_convex_solver_calls"] >= sum(failures)
    assert report["solver_failure_examples"]
    proof(structure, rows[0], rows[1])


def test_episode_stability_penalizes_a_concentrated_failure_even_when_pooled_passes():
    rows = list(data(n=480))
    rows[0][:, :3] = 0.
    # 96% global WAIT agreement is enough for M1's 95% proxy, but is below
    # M2's 97% margin and concentrated in one complete fit episode group.
    rows[2][:] = [.01, .01, .01, .01, .96]
    rows[2][:19] = [.96, .01, .01, .01, .01]
    _, base = fit(rows, config(max_leaves=1))
    _, robust = fit(rows, config(max_leaves=1), search.ROBUST_METHOD)
    assert base["fit_qualified_leaf_coverage"] == 1.
    assert robust["fit_stable_explanation_coverage"] == 0.
    assert robust["fit_leaf_reports"][0]["episode_stability_groups"]
    assert robust["fit_explanation_margin"]["minimum_agreement"] == .97


def test_repeat_is_deterministic_and_input_arrays_unchanged():
    rows = data(split=True)
    snapshots = [array.copy() for array in rows[:3]]
    first, report = fit(rows, config(max_depth=2, max_leaves=4), search.ROBUST_METHOD)
    second, again = fit(rows, config(max_depth=2, max_leaves=4), search.ROBUST_METHOD)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert report == again
    for array, snapshot in zip(rows[:3], snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)
    assert report["fit_target_sha256"]


@pytest.mark.parametrize("changes", [{"initializations": 9}, {"max_rounds": 31},
    {"max_leaf_optimization_calls": 20001}, {"beam_width": 5},
    {"max_nonimproving_steps": 3}, {"max_depth": 6}, {"max_leaves": 17},
    {"beam_width": True}])
def test_config_cannot_expand_registered_budget(changes):
    with pytest.raises(ValueError, match="invalid queued tree budget"):
        config(**changes)


def test_unknown_method_or_corrupt_mask_rejected_before_search():
    with pytest.raises(ValueError, match="unknown queued tree method"):
        fit(data(), config(), "unregistered_method")
    rows = list(data())
    rows[0][0, 3] = 0.
    with pytest.raises(ValueError, match="complete masked NN"):
        fit(rows, config())
