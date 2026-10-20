"""Synthetic search/ledger equivalence and immutable NN-target contracts."""
from dataclasses import replace
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_v625_search as search
from backend.training import warehouse_v625_program as program
from backend.training.warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES
from backend.training.warehouse_v622_program import normalize_structure, program_structure_sha256
from backend.training.warehouse_v619_tree import _Decision, _tree_key
from backend.training.warehouse_program_search import _BeamEntry, _choose_beam


INDICES = tuple(FEATURE_NAMES.index("legal_"+a) for a in ("up", "down", "left", "right", "wait"))


def data():
    x = np.zeros((320, FEATURE_DIM), np.float32)
    row = FEATURE_NAMES.index("self_row")
    x[:, row] = np.tile(np.linspace(0, 1, 20), 16)
    masks = np.ones((len(x), 5), np.float32)
    # Nonzero, finite variable preferences. Some physical rows explicitly
    # reject UP, exercising 0*log(0) and physical-mask certificate boundaries.
    z = np.full((len(x), 5), -5., np.float32)
    z[:, 0] = 3-6*x[:, row]
    z[:, 1] = -3+6*x[:, row]
    z[:, 4] = -3
    masks[np.arange(len(x)) % 20 == 2, 0] = 0
    x[:, INDICES] = masks
    targets = program.masked_probabilities(z, masks).astype(np.float32)
    episodes = np.asarray(["synthetic-"+str(i//20) for i in range(len(x))])
    return x, masks, targets, episodes


def initial():
    weights = np.zeros((3, FEATURE_DIM), np.float32)
    weights[0, FEATURE_NAMES.index("self_row")] = 1
    weights[1, INDICES[0]] = weights[2, INDICES[1]] = 1
    return {"root_ref": 0, "leaf_count": 4, "feature_dim": FEATURE_DIM,
        "split_weights": weights, "split_bias": np.asarray([-.75, -.5, -.5], np.float32),
        "left_refs": np.asarray([1, -1, -3]), "right_refs": np.asarray([2, -2, -4])}


def run(mode="P1", *, budget=None, checkpoint=None, resume=False, rows=None):
    return search.fit_structure(mode, *(data() if rows is None else rows),
        feature_names=FEATURE_NAMES, actor_role_sha256="a"*64,
        initial_structure=initial(), initial_structure_sha256=program_structure_sha256(initial()),
        budget=budget or search.UpgradeBudget(maximum_new_leaf_calls=96,
            wall_seconds=660, maximum_rounds=1), checkpoint_path=checkpoint, resume=resume)


def comparable_calls(report):
    events = report["leaf_api_attempts"]
    return [(r["event"], r["data"]["call"], r["data"].get("metadata"),
             r["data"].get("success"), r["data"].get("detail", {}).get("support"))
            for r in events if r["event"] in ("call_started", "call_completed")]


def test_real_leaf_search_keeps_nn_targets_immutable_and_counts_every_support(tmp_path):
    rows = data()
    before = [value.copy() for value in rows]
    structure, report = run(checkpoint=tmp_path/"checkpoint.json", rows=rows)
    for old, new in zip(before, rows, strict=True):
        np.testing.assert_array_equal(old, new)
    assert all(np.isfinite(report[key]).all() for key in ("best_score", "initial_score"))
    assert tuple(report["best_score"]) >= tuple(report["initial_score"])
    topology, capacity = normalize_structure(structure)
    assert capacity["leaf_count"] <= 32 and capacity["depth"] <= 6
    assert len(structure["initial_leaf_models"]) == capacity["leaf_count"]
    assert report["new_leaf_calls"] <= 64
    assert report["actual_convex_solver_calls"] == report["new_leaf_calls"]
    assert sum(r["event"] == "call_started" for r in report["leaf_api_attempts"]) == report["new_leaf_calls"]
    assert sum(r["event"] == "call_completed" for r in report["leaf_api_attempts"]) == report["new_leaf_calls"]
    assert report["search_uses_fit_only"] and not report["calibration_eligibility_validation_used"]
    assert report["zero_actor_updates"] and report["zero_environment_resets"]
    assert not report["execution_action_override"]
    for leaf in structure["initial_leaf_models"]:
        assert leaf["certificate"]["passed"]
        assert len(leaf["support"]) <= 3
        assert leaf["report"]["full_fit_calls"] <= 8


def test_both_arms_have_the_same_conditional_candidate_menu(tmp_path):
    rows = data()
    sessions = []
    try:
        menus = []
        for mode in ("P0", "P1"):
            from backend.training.warehouse_v625_checkpoint import CheckpointSession
            state = CheckpointSession(tmp_path/mode/"checkpoint.json", {"mode": mode}, max_calls=96, wall_seconds=60)
            sessions.append(state)
            ctx = search._Context(*rows, FEATURE_NAMES, INDICES, mode, state)
            tree = search._tree(initial())
            # Proposal construction reads NN fit rows and the current topology,
            # never which leaf language will subsequently fit a candidate.
            fitted = SimpleNamespace(tree=tree, key=_tree_key(tree), regions=list(ctx.regions(tree)))
            menus.append([(search._tree_key(t).hex(), k, label, mass)
                          for t, k, label, mass in search._proposal_trees(ctx, fitted,
                              search.SEARCH_INITIALIZATIONS[0], "S1", search.SearchConfig(), INDICES)])
        assert menus[0] == menus[1] and menus[0]
    finally:
        for state in sessions:
            state.close()


def test_queue_decode_reuses_shared_subtrees_and_has_bounded_fifo(tmp_path, monkeypatch):
    from backend.training.warehouse_v625_checkpoint import CheckpointSession
    state = CheckpointSession(tmp_path/"checkpoint.json", {"engineering": True},
                              max_calls=96, wall_seconds=60)
    ctx = search._Context(*data(), FEATURE_NAMES, INDICES, "P1", state)
    encoded = search._Codec(ctx)
    common = search._tree(initial()).left
    roots = [_Decision(tuple(np.eye(1, FEATURE_DIM, FEATURE_NAMES.index("self_row"), dtype=np.float32)[0]),
                       -float(i)/20., common, None) for i in range(6)]
    refs = [encoded.tree_ref(tree) for tree in roots]
    decoded = search._Codec(ctx)
    calls = Counter()
    actual_get = state.store.get_json
    def counted(ref):
        calls[ref["sha256"]] += 1
        return actual_get(ref)
    monkeypatch.setattr(state.store, "get_json", counted)
    result = [decoded.tree(ref) for ref in refs]
    assert all(value.left is result[0].left for value in result)
    assert all(search._tree_key(a) == search._tree_key(b) for a, b in zip(roots, result, strict=True))
    assert all(count == 1 for count in calls.values())
    assert len(decoded.decoded_trees) <= search.CODEC_TREE_CACHE_ENTRIES
    # A reduced diagnostic cap exercises eviction without thousands of files.
    monkeypatch.setattr(search, "CODEC_TREE_CACHE_ENTRIES", 3)
    bounded = search._Codec(ctx)
    bounded.tree(refs[0])
    oldest = next(iter(bounded.decoded_trees))
    for ref in refs[1:]:
        bounded.tree(ref)
        assert len(bounded.decoded_trees) <= 3
    assert oldest not in bounded.decoded_trees
    state.close()


def test_exact_queue_patches_preserve_every_candidate_and_share_parent_snapshot(tmp_path):
    from backend.training.warehouse_v625_checkpoint import CheckpointSession
    from backend.training.warehouse_v619_tree import _replace
    state = CheckpointSession(tmp_path/"checkpoint.json", {"engineering": True}, max_calls=96, wall_seconds=60)
    ctx = search._Context(*data(), FEATURE_NAMES, INDICES, "P0", state)
    regions = list(ctx.regions(search._tree(initial())))
    leaves = [ctx.fit(region) for region in regions if region.tree is None]
    tree = search._tree(initial())
    parent = search._State(tree, leaves, regions, search._score(leaves), search._tree_key(tree))
    codec = search._Codec(ctx)
    candidates = [tree, None, _replace(tree, (0,), None), _replace(tree, (1,), tree.left),
                  _Decision(tree.weight, -.25, tree.left, tree.right),
                  _Decision(tree.weight, tree.bias, None, tree.left)]
    queue = [(t, "synthetic", str(i), _BeamEntry(parent, i % 2)) for i, t in enumerate(candidates)]
    ref = codec.queue_ref(queue)
    packed = state.store.get_json(ref)
    assert packed["kind"] == "ordered_exact_parent_tree_patches.v1"
    assert len(packed["parents"]) == 2  # same fitted state, two debts
    assert len(packed["entries"]) == len(candidates)
    restored = search._Codec(ctx).queue(ref)
    assert [(kind, label, p.nonimproving_steps) for _, kind, label, p in restored] == [
        (kind, label, p.nonimproving_steps) for _, kind, label, p in queue]
    assert [search._tree_key(t) for t, *_ in restored] == [search._tree_key(t) for t in candidates]
    assert restored[0][0] is restored[0][3].state.tree
    state.close()


def test_patch_split_memo_uses_finite_coefficient_values_and_bounded_immutable_tuples(tmp_path, monkeypatch):
    from backend.training.warehouse_v625_checkpoint import CheckpointSession
    state = CheckpointSession(tmp_path/"checkpoint.json", {"engineering": True}, max_calls=96, wall_seconds=60)
    ctx = search._Context(*data(), FEATURE_NAMES, INDICES, "P0", state)
    codec = search._Codec(ctx)
    index = FEATURE_NAMES.index("self_row")
    split = [0., [[index, 1.]]]
    one = codec._apply_delta(None, [split, None, None])
    split[0] = -.5
    two = codec._apply_delta(None, [split, None, None])
    assert one.weight is two.weight and one.bias == 0. and two.bias == -.5
    assert isinstance(one.weight, tuple)
    split[1][0][1] = 2.
    changed = codec._apply_delta(None, [split, None, None])
    assert changed.weight[index] == 2. and one.weight[index] == 1.
    assert changed.weight is not one.weight
    monkeypatch.setattr(search, "CODEC_TREE_CACHE_ENTRIES", 3)
    for i in range(8):
        codec._apply_delta(None, [[0., [[index, float(i)]]], None, None])
        assert len(codec.patch_splits) <= 3
    with pytest.raises(RuntimeError, match="finite"):
        codec._apply_delta(None, [[0., [[index, float("nan")]]], None, None])
    state.close()


def test_explicit_pause_resume_preserves_fits_candidates_cache_and_best(tmp_path):
    budget = search.UpgradeBudget(maximum_new_leaf_calls=96, wall_seconds=660, maximum_rounds=1)
    whole, expected = run(budget=budget)
    path = tmp_path/"checkpoint.json"
    _, paused = run(budget=replace(budget, pause_after_candidates=2), checkpoint=path)
    assert paused["stop_reason"] == "paused_safe_boundary"
    resumed, actual = run(budget=budget, checkpoint=path, resume=True)
    a, _ = normalize_structure(whole); b, _ = normalize_structure(resumed)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert whole["initial_leaf_models"] == resumed["initial_leaf_models"] or [
        (l["support"], l["witness"], l["bias"], l["weights"]) for l in whole["initial_leaf_models"]] == [
        (l["support"], l["witness"], l["bias"], l["weights"]) for l in resumed["initial_leaf_models"]]
    for key in ("best_score", "initial_score", "new_leaf_calls", "leaf_cache_hits", "leaf_cache_evictions",
                "candidate_order_sha256", "evaluated_candidates", "operation_counts", "rounds"):
        assert expected[key] == actual[key]
    assert comparable_calls(expected) == comparable_calls(actual)
    assert actual["resume_count"] == 1
    assert actual["elapsed_seconds"] >= paused["elapsed_seconds"]
    assert expected["checkpoint_fraction_scope"] == "fresh_measured_segment"
    assert expected["checkpoint_fraction_denominator_seconds"] == expected["runner_segment_elapsed_seconds"]
    assert actual["checkpoint_fraction_scope"] == "resumed_cumulative_io_upper_over_actual_lower_bound"
    assert actual["checkpoint_fraction_numerator_seconds"] == actual["checkpoint_timing"]["checkpoint_total_io_upper_bound_seconds"]
    assert actual["checkpoint_fraction_denominator_seconds"] == actual["checkpoint_timing"]["actual_elapsed_lower_bound_seconds"]
    assert actual["checkpoint_fraction"] == actual["checkpoint_fraction_numerator_seconds"]/actual["checkpoint_fraction_denominator_seconds"]


def test_wrong_input_or_crash_cannot_resume_or_overwrite(tmp_path):
    path = tmp_path/"checkpoint.json"
    budget = search.UpgradeBudget(maximum_new_leaf_calls=96, wall_seconds=660,
                                  maximum_rounds=1, pause_after_candidates=1)
    run(budget=budget, checkpoint=path)
    rows = data(); rows[2][0] = np.asarray([.95, .02, .01, .01, .01], np.float32)
    with pytest.raises(RuntimeError, match="identical"):
        run(budget=budget, checkpoint=path, resume=True, rows=rows)
    with pytest.raises(RuntimeError, match="overwrite"):
        run(budget=budget, checkpoint=path)
    Path(str(path)+".dirty").write_text("crash")
    with pytest.raises(RuntimeError, match="dirty/crashed"):
        run(budget=budget, checkpoint=path, resume=True)


def test_bounded_transition_retention_matches_full_beam_even_when_best_changes():
    rng = np.random.default_rng(731)
    entries, compact = [], []
    best = None
    for i in range(1000):
        # Ties, mixed exploration debts, and an improving incumbent are all
        # represented. Candidate state keys determine stable tie breaking.
        score = tuple(map(float, rng.integers(0, 15, size=4)))
        item = SimpleNamespace(score=score, key=sha256(str(i).encode()).digest())
        entry = _BeamEntry(item, int(rng.integers(0, 3)))
        entries.append(entry)
        compact = search._bounded_transitions(compact, entry)
        if best is None or score > best.score or (score == best.score and item.key < best.key):
            best = item
        full_beam = _choose_beam(entries, best, 4)
        small_beam = _choose_beam(compact, best, 4)
        assert [(e.state.key, e.nonimproving_steps) for e in full_beam] == [
            (e.state.key, e.nonimproving_steps) for e in small_beam]
        assert len(compact) <= search.TRANSITION_ENTRIES


def test_initial_budget_failure_exposes_charged_ledger_and_not_a_fake_incumbent(tmp_path, monkeypatch):
    def consume_all(x, masks, targets, names, legal, mode="P1", budget=None, **kwargs):
        while True:
            token = budget.consume()
            budget.finish(token, success=False, detail={"support": []})
    monkeypatch.setattr(program, "fit_leaf", consume_all)
    budget = search.UpgradeBudget(maximum_new_leaf_calls=33, wall_seconds=660, maximum_rounds=1)
    with pytest.raises(search.SearchIncompleteError) as captured:
        run(budget=budget, checkpoint=tmp_path/"checkpoint.json")
    report = captured.value.report
    assert report["new_leaf_calls"] == 1 and not report["complete_incumbent"]
    assert report["failure_kind"] == "initial_leaf_api_exhausted"
    assert report["configuration"] == search.search_configuration()
    assert report["elapsed_seconds"] >= 0 and report["leaf_api_attempts"]
    assert report["checkpoint_timing"]["pending_calls"] == 0
    assert Path(str(tmp_path/"checkpoint.json")+".dirty").exists()
