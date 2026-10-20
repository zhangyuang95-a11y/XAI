"""Synthetic behavior checks for the new search, never scientific rollouts."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backend.training.warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES
from backend.training.warehouse_v6_tree_routing import (
    certified_leaf_actions, certified_feature_bounds, validate_certification_domain,
    leaf_indices,
)
from backend.training.warehouse_v619_tree import _Decision, _State, _tree_key, _replace, _size
from backend.training.warehouse_v622_program import normalize_structure, program_structure_sha256
from backend.training.warehouse_v622_search import _Clock, _leaves_score, _FitContext
from backend.training import warehouse_v624_search as search
from backend.training.warehouse_v623_search import fit_structure as s1_search, ContinuationBudget


INDICES = tuple(FEATURE_NAMES.index("legal_"+a) for a in ("up", "down", "left", "right", "wait"))


def data():
    x = np.zeros((320, FEATURE_DIM), np.float32)
    x[:, FEATURE_NAMES.index("self_row")] = np.tile(np.linspace(0, 1, 20), 16)
    masks = np.ones((len(x), 5), np.float32)
    targets = np.full(masks.shape, .01, np.float32)
    targets[:, 0] = .96
    right = x[:, FEATURE_NAMES.index("self_row")] >= .5
    targets[right, 0] = .01; targets[right, 1] = .96
    illegal_up = np.arange(len(x)) % 20 == 2
    masks[illegal_up, 0] = 0
    targets[illegal_up, 0] = 0; targets[illegal_up, 4] = .97
    x[:, INDICES] = masks
    episodes = np.asarray(["synthetic-"+str(i//20) for i in range(len(x))])
    validate_certification_domain(x, INDICES, certified_feature_bounds(FEATURE_NAMES))
    return x, masks, targets, episodes


def initial():
    w = np.zeros((3, FEATURE_DIM), np.float32)
    w[0, FEATURE_NAMES.index("self_row")] = 1
    w[1, INDICES[0]] = 1; w[2, INDICES[1]] = 1
    return {"root_ref": 0, "leaf_count": 4, "feature_dim": FEATURE_DIM,
        "split_weights": w, "split_bias": np.asarray([-.75, -.5, -.5], np.float32),
        "left_refs": np.asarray([1, -1, -3]), "right_refs": np.asarray([2, -2, -4])}


def run(mode="P1", *, budget=None, checkpoint=None, resume=False, rows=None):
    return search.fit_structure(mode, *(data() if rows is None else rows),
        feature_names=FEATURE_NAMES, actor_role_sha256="a"*64,
        initial_structure=initial(), initial_structure_sha256=program_structure_sha256(initial()),
        budget=budget or search.UpgradeBudget(maximum_new_leaf_calls=128,
            wall_seconds=660, maximum_rounds=1), checkpoint_path=checkpoint, resume=resume)


def assert_structure_equal(left, right):
    a, _ = normalize_structure(left); b, _ = normalize_structure(right)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(left["initial_leaf_logits"], right["initial_leaf_logits"])


def certified(structure, rows):
    tree, cap = normalize_structure(structure)
    proof = certified_leaf_actions(tree["split_weights"], tree["split_bias"], tree["left_refs"],
        tree["right_refs"], len(tree["split_weights"]), tree["leaf_count"], INDICES,
        explicit=True, wait_action=4)
    logits = np.asarray(structure["initial_leaf_logits"], np.float32)
    assert cap["depth"] <= 6 and cap["leaf_count"] <= 32
    assert cap["maximum_source_features_per_node"] <= 3
    for leaf, action in enumerate(logits.argmax(-1)):
        assert int(action) in proof[leaf]
    routes = leaf_indices(rows[0], tree["split_weights"], tree["split_bias"], tree["left_refs"],
        tree["right_refs"], len(tree["split_weights"]), cap["depth"])
    actions = logits[routes].argmax(-1)
    assert np.all(rows[1][np.arange(len(rows[0])), actions] == 1)


def test_p0_matches_s1_candidate_mechanism_exactly():
    rows = data(); kwargs = {"feature_names": FEATURE_NAMES, "actor_role_sha256": "a"*64,
        "initial_structure": initial(), "initial_structure_sha256": program_structure_sha256(initial())}
    legacy, old = s1_search("S1", *rows, **kwargs,
        budget=ContinuationBudget(maximum_new_leaf_calls=128, wall_seconds=660, maximum_rounds=1))
    current, new = run("P0", rows=rows)
    assert_structure_equal(legacy, current)
    assert old["new_leaf_calls"] == new["new_leaf_calls"]
    assert old["best_score"] == new["best_score"]
    assert new["production_configuration"] is False
    assert new["node_fit_calls"] == 0


def test_p1_uses_full_cost_node_updates_and_preserves_inputs_and_certificates():
    rows = data(); before = tuple(value.copy() for value in rows)
    structure, report = run(rows=rows)
    for original, supplied in zip(before, rows, strict=True):
        np.testing.assert_array_equal(original, supplied)
    certified(structure, rows)
    assert tuple(report["best_score"]) >= tuple(report["initial_score"])
    assert report["best_score"][0] > report["initial_score"][0]
    assert report["node_fit_calls"] > 0 and report["node_linear_solves"] > 0
    assert report["operation_counts"].get("cost_node", 0) > 0
    assert report["search_uses_fit_only"] and not report["calibration_eligibility_validation_used"]
    ledger = report["leaf_api_attempts"]
    assert len(ledger) == report["new_leaf_calls"] <= 96
    assert report["actual_convex_solver_calls"] == sum(e["actual_convex_solver_calls"] for e in ledger)
    assert report["full_leaf_statistic_cache_hits"] > 0
    for event in report["node_fit_events"]:
        assert len(event["fit_rows_sha256"]) == 64 and event["fit_only"]
        assert event["duration_seconds"] >= 0 and event["success"]
        for attempt in event["support_fits"]:
            assert attempt["source_feature_cost"] <= 3
            assert len(attempt["support"]) == len(attempt["source_features"]) <= 3
            assert attempt["duration_seconds"] >= 0


def test_safe_resume_preserves_candidate_order_fits_beam_and_best(tmp_path):
    budget = search.UpgradeBudget(maximum_new_leaf_calls=128, wall_seconds=660, maximum_rounds=1)
    complete, whole = run(budget=budget)
    path = tmp_path/"checkpoint.json"
    _, paused = run(budget=replace(budget, pause_after_candidates=3), checkpoint=path)
    assert paused["stop_reason"] == "paused_safe_boundary"
    assert not Path(str(path)+".dirty").exists()
    continued, resumed = run(budget=budget, checkpoint=path, resume=True)
    assert_structure_equal(complete, continued)
    for key in ("best_score", "new_leaf_calls", "actual_convex_solver_calls", "leaf_api_attempts",
                "candidate_order_sha256", "evaluated_candidates", "operation_counts", "rounds"):
        assert resumed[key] == whole[key]
    assert resumed["resume_count"] == 1
    assert resumed["checkpoint_resume_equivalent"]
    assert resumed["elapsed_seconds"] >= paused["elapsed_seconds"]


def test_unsafe_checkpoint_and_wrong_inputs_fail_closed(tmp_path):
    path = tmp_path/"checkpoint.json"
    budget = search.UpgradeBudget(maximum_new_leaf_calls=128, wall_seconds=660,
                                  maximum_rounds=1, pause_after_candidates=1)
    run(budget=budget, checkpoint=path)
    dirty = Path(str(path)+".dirty"); dirty.write_text("interrupted work")
    with pytest.raises(RuntimeError, match="unsafe work"):
        run(budget=budget, checkpoint=path, resume=True)
    dirty.unlink()
    rows = data(); rows[2][0] = np.asarray([.95, .02, .01, .01, .01], np.float32)
    with pytest.raises(RuntimeError, match="source, fit input or configuration mismatch"):
        run(budget=budget, checkpoint=path, resume=True, rows=rows)


def test_incremental_routes_and_full_leaf_statistics_equal_full_recalculation():
    rows = data()
    ctx = search._ExactContext(*rows, FEATURE_NAMES, INDICES, _Clock(60), 96)
    old_ctx = _FitContext(*rows, FEATURE_NAMES, INDICES, _Clock(60), 96)
    tree = search._tree(initial())
    regions = list(ctx.regions(tree)); leaves = [ctx.fit(r) for r in regions if r.tree is None]
    parent = _State(tree, leaves, regions, _leaves_score(leaves), _tree_key(tree))
    changed = _Decision(tree.weight, -.5, tree.left, tree.right)
    incremental = list(search._regions(ctx, changed, parent))
    full = list(old_ctx.regions(changed))
    assert len(incremental) == len(full)
    for actual, expected in zip(incremental, full, strict=True):
        assert actual.path == expected.path and actual.legal == expected.legal
        np.testing.assert_array_equal(actual.rows, expected.rows)
        if actual.tree is None:
            a = ctx.fit(actual); b = old_ctx.fit(expected)
            np.testing.assert_array_equal(a.logits, b.logits)
            assert (a.qualified, a.expected_mass, a.kl_sum, a.agreement, a.robust) == (
                b.qualified, b.expected_mass, b.kl_sum, b.agreement, b.robust)
    # Repeating identical work must reuse the complete statistics without API calls.
    previous = ctx.calls
    for region in incremental:
        if region.tree is None:
            ctx.fit(region)
    assert ctx.calls == previous and ctx.full_cache_hits > 0


def test_fixed_production_budget_and_bad_initial_sha():
    assert search.UpgradeBudget().production_configuration
    with pytest.raises(ValueError):
        search.UpgradeBudget(maximum_new_leaf_calls=20001)
    with pytest.raises(ValueError):
        search.fit_structure("P1", *data(), feature_names=FEATURE_NAMES,
            actor_role_sha256="a"*64, initial_structure=initial(), initial_structure_sha256="0"*64)


def test_full_leaf_cache_fifo_eviction_counts_refitting(monkeypatch):
    monkeypatch.setattr(search, "FULL_LEAF_CACHE_ENTRIES", 2)
    rows = data()
    ctx = search._ExactContext(*rows, FEATURE_NAMES, INDICES, _Clock(60), 10)
    regions = [search._Region((), None, np.arange(i*20, (i+1)*20), 0, frozenset({4})) for i in range(3)]
    for region in regions:
        ctx.fit(region)
    assert ctx.calls == 3 and ctx.full_cache_evictions == 1
    assert len(ctx.full_cache) == len(ctx.cache) == 2
    ctx.fit(regions[0])
    assert ctx.calls == 4 and ctx.full_cache_evictions == 2
    assert len(ctx.events) == 4


def test_initial_incomplete_is_scientific_but_route_fault_is_engineering(monkeypatch):
    with pytest.raises(search.SearchIncompleteError) as error:
        run(budget=search.UpgradeBudget(maximum_new_leaf_calls=33, wall_seconds=660, maximum_rounds=1))
    report = error.value.report
    assert report["failure_kind"] == "initial_fit_budget_exhausted"
    assert report["new_leaf_calls"] == len(report["leaf_api_attempts"]) == 1
    assert report["scientific_failure"] and len(report["fit_target_sha256"]) == 64
    assert report["configuration"] == search.search_configuration()
    def faulty_route(*args, **kwargs):
        raise RuntimeError("path proof disagrees with physical masks")
    monkeypatch.setattr(search._ExactContext, "fit", faulty_route)
    with pytest.raises(RuntimeError, match="path proof") as error:
        run()
    assert not isinstance(error.value, search.SearchIncompleteError)


def test_depth_six_bad_leaves_have_real_ancestor_repair_and_atomic_budget_moves():
    feature = FEATURE_NAMES.index("self_row")
    weight = np.zeros(FEATURE_DIM, np.float32); weight[feature] = 1
    def full(depth, low=0., high=1.):
        if not depth:
            return None
        split = (low+high)/2
        return _Decision(tuple(weight), -split, full(depth-1, low, split), full(depth-1, split, high))
    tree = full(5)
    tree = _replace(tree, (0, 0, 0, 0, 0), _Decision(tuple(weight), -1/64))
    tree = _replace(tree, (1, 1, 1, 1), None)
    assert _size(tree) == (32, 6)
    values = np.asarray([.007]*20+[.023]*20+list(np.linspace(.1, 1, 40)), np.float32)
    x = np.zeros((16*80, FEATURE_DIM), np.float32)
    x[:, feature] = np.tile(values, 16)
    masks = np.ones((len(x), 5), np.float32); x[:, INDICES] = masks
    p = np.full(masks.shape, .01, np.float32); p[:, 0] = .96
    second = (x[:, feature] > .015625) & (x[:, feature] < .03125)
    p[second, 0] = .01; p[second, 1] = .96
    episodes = np.asarray(["depth-synthetic-"+str(i//80) for i in range(len(x))])
    rows = x, masks, p, episodes
    ctx = search._ExactContext(*rows, FEATURE_NAMES, INDICES, _Clock(60), 128)
    regions = list(ctx.regions(tree)); leaves = [ctx.fit(r) for r in regions if r.tree is None]
    state = _State(tree, leaves, regions, _leaves_score(leaves), _tree_key(tree))
    offers = search._p1_proposals(ctx, state, search.SEARCH_INITIALIZATIONS[0], search.SearchConfig())
    ancestor = [r for r in offers if r[1] == "ancestor_bad_region_repair" and r[2].startswith("(0, 0, 0, 0)")]
    atomic = [r for r in offers if r[1] == "atomic_reallocate"]
    assert ancestor and atomic
    assert any(_size(r[0])[0] == 32 for r in atomic)
    for candidate, _, _, _ in ancestor[:2]+atomic[:2]:
        count, depth = _size(candidate)
        assert count <= 32 and depth <= 6
        candidate_regions = list(search._regions(ctx, candidate, state))
        fitted = [ctx.fit(r) for r in candidate_regions if r.tree is None]
        exported = search._export(candidate, fitted, FEATURE_DIM)
        exported["feature_dim"] = FEATURE_DIM
        certified(exported, rows)
