"""Train-only sparse core construction; independent rows never refit it."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import backend.training.warehouse_v6_core_tree as core_tree_module
from backend.training import warehouse_v6_runner
from backend.training.warehouse_v6_core_tree import (
    CoreTreeConfig, CoreTreeResult, _insert_legality_splits, _leaf_metrics,
    evaluate_core_tree, fit_core_tree,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_observation_v6 import FEATURE_DIM


def _synthetic_rows(*, seed: int = 3, count: int = 240, prefix: str = "train"):
    rng = np.random.default_rng(seed)
    encoded = np.zeros((count, FEATURE_DIM), dtype=np.float32)
    encoded[:, 0] = rng.random(count)
    masks = np.ones((count, len(ACTIONS)), dtype=np.float32)
    masks[:, 0] = (rng.random(count) < 0.8).astype(np.float32)
    encoded[:, -len(ACTIONS):] = masks
    labels = np.where((encoded[:, 0] < 0.5) & (masks[:, 0] > 0.5), 0, 1)
    targets = np.eye(len(ACTIONS), dtype=np.float32)[labels]
    episodes = [f"{prefix}-{index // 24}" for index in range(count)]
    return encoded, masks, targets, episodes


_SMALL = CoreTreeConfig(
    min_leaf_rows=10, min_leaf_episodes=3,
    min_split_rows=(10,), tree_shapes=((2, 4), (4, 8)),
    include_sparse_projections=False,
)


def test_core_tree_implementation_is_in_training_source_closure():
    source = Path(core_tree_module.__file__).resolve()
    relative = "backend/training/warehouse_v6_core_tree.py"
    assert warehouse_v6_runner.source_closure_files_sha256()[relative] == (
        sha256(source.read_bytes()).hexdigest())


def test_core_tree_uses_public_sparse_nodes_and_structural_legality():
    x, mask, target, episodes = _synthetic_rows()
    result = fit_core_tree(x, mask, target, episodes,
                           target_provenance="public_teacher_one_hot", config=_SMALL)

    assert result.report["train_episode_count"] == 10
    assert result.report["target_provenance"] == "public_teacher_one_hot"
    assert result.report["program_qualification_episodes_used"] is False
    assert result.report["execution_action_selection"] is False
    assert result.node_count <= 15 and result.leaf_count <= 16
    assert np.count_nonzero(result.split_weights, axis=1).max() <= 3
    assert result.report["chosen_candidate"]["core_training_coverage"] >= 0.30
    assert result.report["chosen_candidate"]["core_raw_illegal_actions"] == 0
    for detail in result.report["leaf_diagnostics"]:
        if detail["core_eligible"]:
            assert detail["raw_action_legal_by_tree_path"]
            assert detail["raw_illegal_actions"] == 0
            assert detail["tree_path_depth"] <= 5


def test_independent_evaluation_cannot_select_or_modify_core_leaves():
    x, mask, target, episodes = _synthetic_rows()
    result = fit_core_tree(x, mask, target, episodes,
                           target_provenance="public_teacher_one_hot", config=_SMALL)
    fingerprint = sha256(b"".join((result.split_weights.tobytes(),
                                    result.split_bias.tobytes(),
                                    result.leaf_logits.tobytes(),
                                    result.core_mask.tobytes()))).hexdigest()
    independent = [value.replace("train", "independent") for value in episodes]
    first = evaluate_core_tree(result, x, mask, target, independent)
    assert first["core_coverage"] >= 0.30
    altered = np.zeros_like(target)
    altered[:, ACTIONS.index("WAIT")] = 1.0
    second = evaluate_core_tree(result, x, mask, altered, independent)
    assert second["core_agreement"] <= first["core_agreement"]
    assert second["selection_changed"] is False
    assert sha256(b"".join((result.split_weights.tobytes(),
                             result.split_bias.tobytes(),
                             result.leaf_logits.tobytes(),
                             result.core_mask.tobytes()))).hexdigest() == fingerprint
    with pytest.raises(ValueError, match="overlaps fitting"):
        evaluate_core_tree(result, x, mask, target, episodes)


def test_sparse_three_feature_projection_candidates_are_training_only():
    x, mask, target, episodes = _synthetic_rows(count=240)
    x[:, 2] = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    config = CoreTreeConfig(
        min_leaf_rows=10, min_leaf_episodes=3,
        min_split_rows=(12,), tree_shapes=((3, 6),),
        include_sparse_projections=True,
    )
    result = fit_core_tree(x, mask, target, episodes,
                           target_provenance="public_teacher_one_hot", config=config)
    assert any(item["input_kind"] == "sparse3"
               for item in result.report["candidate_diagnostics"])
    assert np.all(np.count_nonzero(result.split_weights, axis=1) <= 3)


def test_illegal_or_unnormalized_training_targets_are_rejected():
    x, mask, target, episodes = _synthetic_rows()
    bad = target.copy()
    forbidden = int(np.flatnonzero(mask[:, 0] == 0.0)[0])
    bad[forbidden] = np.eye(len(ACTIONS), dtype=np.float32)[0]
    with pytest.raises(ValueError, match="masked training distributions"):
        fit_core_tree(x, mask, bad, episodes,
                      target_provenance="public_teacher_one_hot", config=_SMALL)
    with pytest.raises(ValueError, match="provenance"):
        fit_core_tree(x, mask, target, episodes,
                      target_provenance="unknown", config=_SMALL)
    altered_features = x.copy()
    altered_features[0, -len(ACTIONS):] = 1.0 - mask[0]
    with pytest.raises(ValueError, match="masked training distributions"):
        fit_core_tree(altered_features, mask, target, episodes,
                      target_provenance="public_teacher_one_hot", config=_SMALL)


def _empty_tree(node_count: int, leaf_count: int, left: np.ndarray,
                right: np.ndarray, *, core_action: str | None = None,
                top_action: str | None = None) -> CoreTreeResult:
    weights = np.zeros((15, FEATURE_DIM), dtype=np.float32)
    bias = np.zeros(15, dtype=np.float32)
    weights[:node_count, 0] = 4.0
    bias[:node_count] = -2.0
    logits = np.zeros((16, len(ACTIONS)), dtype=np.float32)
    core = np.zeros(16, dtype=np.bool_)
    if top_action is not None:
        logits[0, ACTIONS.index(top_action)] = 3.0
    if core_action is not None:
        logits[0, ACTIONS.index(core_action)] = 3.0
        core[0] = True
    return CoreTreeResult(weights, bias, left, right, node_count,
                          leaf_count, logits, core, {}, frozenset())


def test_empty_root_can_insert_legal_bit_and_unproven_core_is_rejected():
    left = np.zeros(15, dtype=np.int64)
    right = np.zeros(15, dtype=np.int64)
    with pytest.raises(ValueError, match="structural action legality proof"):
        _empty_tree(0, 1, left, right, core_action="UP")
    # WAIT is a physical environment invariant: it remains legal even for an
    # inactive robot, so it needs no path gate.
    assert _empty_tree(0, 1, left, right, core_action="WAIT").core_mask[0]

    x, mask, _, episodes = _synthetic_rows()
    labels = np.where(mask[:, 0] > 0.5, 0, ACTIONS.index("WAIT"))
    target = np.eye(len(ACTIONS), dtype=np.float32)[labels]
    starter = _empty_tree(0, 1, left, right, top_action="UP")
    grown = _insert_legality_splits(starter, x, mask, target,
                                    np.asarray(episodes), _SMALL)
    assert grown.node_count >= 1 and grown.leaf_count >= 2
    assert grown.split_weights[0, FEATURE_DIM - len(ACTIONS)] == 4.0
    assert grown.split_bias[0] == -2.0
    details, protected, _ = _leaf_metrics(grown, x, mask, target,
                                          np.asarray(episodes), _SMALL)
    assert protected.any()
    assert all(detail["raw_action_legal_by_tree_path"]
               for detail in details if detail["core_eligible"])


def test_fitted_arrays_cannot_be_mutated_after_content_hash():
    x, mask, target, episodes = _synthetic_rows()
    result = fit_core_tree(x, mask, target, episodes,
                           target_provenance="public_teacher_one_hot", config=_SMALL)
    original = result.content_sha256()
    assert original == result.report["tree_content_sha256"]
    with pytest.raises(ValueError):
        result.leaf_logits[0, 0] += 1.0
    with pytest.raises(ValueError):
        result.core_mask[0] = not result.core_mask[0]
    with pytest.raises(ValueError):
        result.split_weights.flags.writeable = True
    assert result.content_sha256() == original


def test_one_float32_route_mismatch_rejects_candidate_without_aborting_search(monkeypatch):
    x, mask, target, episodes = _synthetic_rows()
    original = core_tree_module._convert_tree
    calls = 0

    def first_candidate_has_drift(*args, **kwargs):
        nonlocal calls
        candidate = original(*args, **kwargs)
        calls += 1
        if calls != 1 or not candidate.node_count:
            return candidate
        shifted = candidate.split_bias.copy()
        shifted[0] += 1_000.0
        return CoreTreeResult(candidate.split_weights, shifted,
                              candidate.left_refs, candidate.right_refs,
                              candidate.node_count, candidate.leaf_count,
                              candidate.leaf_logits, candidate.core_mask,
                              candidate.report, frozenset())

    monkeypatch.setattr(core_tree_module, "_convert_tree", first_candidate_has_drift)
    result = fit_core_tree(x, mask, target, episodes,
                           target_provenance="public_teacher_one_hot", config=_SMALL)
    assert calls > 1
    assert any(item["reason"] == "float32_export_route_mismatch"
               for item in result.report["rejected_candidate_diagnostics"])
    assert result.report["chosen_candidate"]["core_training_coverage"] >= 0.30


def test_depth_five_and_full_sixteen_leaf_budgets_are_enforced():
    # Five-node chain: its rightmost branch is at the allowed depth five.
    left = np.zeros(15, dtype=np.int64)
    right = np.zeros(15, dtype=np.int64)
    for node in range(5):
        left[node] = -1 - node
        right[node] = node + 1 if node < 4 else -1 - 5
    depth_five = _empty_tree(5, 6, left, right)
    assert depth_five.node_count == 5
    left[5] = -1 - 5
    right[4] = 5
    right[5] = -1 - 6
    with pytest.raises(ValueError, match="depth budget"):
        _empty_tree(6, 7, left, right)

    # Complete balanced tree fills 15 nodes and 16 leaves, so the legality
    # insertion routine must not exceed either budget.
    full_left = np.zeros(15, dtype=np.int64)
    full_right = np.zeros(15, dtype=np.int64)
    for node in range(15):
        lhs, rhs = 2 * node + 1, 2 * node + 2
        full_left[node] = lhs if lhs < 15 else -1 - (lhs - 15)
        full_right[node] = rhs if rhs < 15 else -1 - (rhs - 15)
    full = _empty_tree(15, 16, full_left, full_right)
    x, mask, target, episodes = _synthetic_rows()
    unchanged = _insert_legality_splits(full, x, mask, target,
                                        np.asarray(episodes), _SMALL)
    assert unchanged.node_count == 15 and unchanged.leaf_count == 16
    assert unchanged.content_sha256() == full.content_sha256()
    bad_weights = full.split_weights.copy()
    bad_weights[0, :4] = 1.0
    with pytest.raises(ValueError, match="invalid Warehouse sparse core tree"):
        CoreTreeResult(bad_weights, full.split_bias, full.left_refs,
                       full.right_refs, full.node_count, full.leaf_count,
                       full.leaf_logits, full.core_mask, {}, frozenset())
