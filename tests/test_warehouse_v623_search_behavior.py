"""Small synthetic search checks; no scientific cohort or Actor is loaded."""
from __future__ import annotations

import numpy as np
import pytest

from backend.training import warehouse_v622_search as leaf_api
from backend.training.warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES
from backend.training.warehouse_v6_tree_routing import (
    certified_feature_bounds,
    certified_leaf_actions,
    leaf_indices,
    validate_certification_domain,
)
from backend.training.warehouse_v619_tree import SEARCH_INITIALIZATIONS, _size
from backend.training.warehouse_v622_program import (
    normalize_structure,
    program_structure_sha256,
)
from backend.training.warehouse_v623_search import (
    ContinuationBudget,
    _candidate_subtrees,
    fit_structure,
)


MASKS = tuple(FEATURE_NAMES.index("legal_" + name) for name in
              ("up", "down", "left", "right", "wait"))


def synthetic_rows(*, semantic=False):
    """Sixteen whole episodes, each containing both observable states."""
    x = np.zeros((320, FEATURE_DIM), dtype=np.float32)
    masks = np.ones((len(x), 5), dtype=np.float32)
    second = np.arange(len(x)) % 2 == 1
    probabilities = np.full(masks.shape, .01, dtype=np.float32)
    probabilities[:, 0] = .96
    if semantic:
        x[second, FEATURE_NAMES.index("self_row")] = 1
        probabilities[second, 0] = .01
        probabilities[second, 1] = .96
    else:
        masks[second, 0] = 0
        probabilities[second, 0] = 0
        probabilities[second, 4] = .97
    x[:, MASKS] = masks
    episodes = np.asarray(["synthetic-episode-" + str(i // 20) for i in range(len(x))])
    validate_certification_domain(x, MASKS, certified_feature_bounds(FEATURE_NAMES))
    np.testing.assert_allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-6)
    assert FEATURE_DIM == 173
    return x, masks, probabilities, episodes


def initial_leaf():
    return {"root_ref": -1, "leaf_count": 1, "feature_dim": FEATURE_DIM,
            "split_weights": np.zeros((0, FEATURE_DIM), dtype=np.float32),
            "split_bias": np.zeros(0, dtype=np.float32),
            "left_refs": np.zeros(0, dtype=np.int64),
            "right_refs": np.zeros(0, dtype=np.int64)}


def invoke(mode, rows, *, new_calls=64):
    initial = initial_leaf()
    return fit_structure(mode, *rows, feature_names=FEATURE_NAMES,
        actor_role_sha256="a" * 64, initial_structure=initial,
        initial_structure_sha256=program_structure_sha256(initial),
        budget=ContinuationBudget(maximum_new_leaf_calls=new_calls,
            wall_seconds=630, maximum_rounds=2))


def assert_certified(structure, rows):
    x, masks, probabilities, _ = rows
    tree, capacity = normalize_structure(structure)
    assert capacity["depth"] <= 6 and capacity["leaf_count"] <= 32
    assert capacity["maximum_source_features_per_node"] <= 3
    proofs = certified_leaf_actions(tree["split_weights"], tree["split_bias"],
        tree["left_refs"], tree["right_refs"], len(tree["split_weights"]),
        tree["leaf_count"], MASKS, explicit=True, wait_action=4)
    logits = np.asarray(structure["initial_leaf_logits"], dtype=np.float32)
    for leaf, action in enumerate(logits.argmax(-1)):
        assert int(action) in proofs[leaf]
    routes = leaf_indices(x, tree["split_weights"], tree["split_bias"],
        tree["left_refs"], tree["right_refs"], len(tree["split_weights"]), capacity["depth"])
    actions = logits[routes].argmax(-1)
    assert np.all(masks[np.arange(len(x)), actions] == 1)
    return actions, probabilities.argmax(-1)


@pytest.mark.parametrize("mode", ["S1", "S2"])
def test_real_search_learns_guard_without_mutating_inputs_and_counts_work(mode, monkeypatch):
    rows = synthetic_rows()
    before = tuple(value.copy() for value in rows)
    calls = []
    original = leaf_api.fit_constrained_leaf_distribution

    def counted_api(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(leaf_api, "fit_constrained_leaf_distribution", counted_api)
    structure, report = invoke(mode, rows)
    for supplied, snapshot in zip(rows, before, strict=True):
        np.testing.assert_array_equal(supplied, snapshot)
    actions, nn_actions = assert_certified(structure, rows)
    # Both a legal movement and its physically illegal counterpart are tested.
    np.testing.assert_array_equal(actions, nn_actions)
    assert set(actions) == {0, 4}
    assert structure["leaf_count"] >= 2
    assert tuple(report["best_score"]) >= tuple(report["initial_score"])
    assert report["best_score"][0] > report["initial_score"][0]
    assert report["search_uses_fit_only"] is True
    assert report["calibration_eligibility_validation_used"] is False
    assert report["zero_actor_updates"] is True
    assert report["zero_environment_resets"] is True
    assert report["checkpoint_resume_equivalent"] is False
    ledger = report["leaf_api_attempts"]
    assert len(calls) == report["new_leaf_calls"] == len(ledger)
    assert 0 < len(calls) <= 32  # 64 total minus the untouched 32 final reserve.
    assert [item["call"] for item in ledger] == list(range(1, len(calls) + 1))
    assert [item["rows"] for item in ledger] == calls
    assert all(item["success"] for item in ledger)
    assert report["actual_convex_solver_calls"] == sum(
        item["actual_convex_solver_calls"] for item in ledger)


def test_exhausted_new_api_budget_keeps_the_complete_incumbent(monkeypatch):
    calls = []
    original = leaf_api.fit_constrained_leaf_distribution

    def counted_api(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(leaf_api, "fit_constrained_leaf_distribution", counted_api)
    rows = synthetic_rows()
    structure, report = invoke("S1", rows, new_calls=33)
    assert report["stop_reason"] == "leaf_call_budget"
    assert len(calls) == report["new_leaf_calls"] == 1
    assert report["reserved_final_leaf_calls"] == 32
    assert report["complete_incumbent"] is True
    assert report["best_score"] == report["initial_score"]
    assert structure["leaf_count"] == 1
    actions, _ = assert_certified(structure, rows)
    assert np.all(actions == 4)


def test_initial_topology_hash_mismatch_fails_before_leaf_work(monkeypatch):
    def forbidden_api(*args, **kwargs):
        pytest.fail("a bad incumbent SHA must fail before fitting")

    monkeypatch.setattr(leaf_api, "fit_constrained_leaf_distribution", forbidden_api)
    with pytest.raises(ValueError, match="topology SHA mismatch"):
        fit_structure("S1", *synthetic_rows(), feature_names=FEATURE_NAMES,
            actor_role_sha256="a" * 64, initial_structure=initial_leaf(),
            initial_structure_sha256="0" * 64,
            budget=ContinuationBudget(maximum_new_leaf_calls=64,
                                      wall_seconds=630, maximum_rounds=1))


def test_s2_really_proposes_two_level_semantic_subtrees():
    rows = synthetic_rows(semantic=True)
    ctx = leaf_api._FitContext(*rows, FEATURE_NAMES, MASKS, leaf_api._Clock(30), 32)
    region = next(ctx.regions(None))
    cfg = leaf_api.SearchConfig()
    simple = _candidate_subtrees(ctx, region, SEARCH_INITIALIZATIONS[0], cfg, "S1", MASKS)
    refined = _candidate_subtrees(ctx, region, SEARCH_INITIALIZATIONS[0], cfg, "S2", MASKS)
    assert not any(label.startswith("subtree:") for _, label in simple)
    two_level = [tree for tree, label in refined if label.startswith("subtree:")]
    assert two_level
    assert all(_size(tree) == (3, 2) for tree in two_level)
    assert ctx.calls == 0  # Proposal construction cannot consume leaf-fit work.
