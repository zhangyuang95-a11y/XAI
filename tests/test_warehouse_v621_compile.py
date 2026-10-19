"""Synthetic contracts; no Warehouse reset, weight file or training invocation."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training.warehouse_v621_compile import (
    compile_actor_structure, fixed_topology_action_oracle,
)
from backend.training.warehouse_v6_program import program_structure_sha256
from backend.training.warehouse_v6_tree_routing import (
    certified_leaf_actions, leaf_indices,
)


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
NAMES = ("self_row", "self_column", "public_hard_risk_context",
         *(f"legal_{a.lower()}" for a in ACTIONS))


def _role(*, depth=0, raw_action=3, legal_split=False):
    nodes = 2**depth-1
    leaves = 2**depth
    w = np.zeros((nodes, len(NAMES)), np.float32)
    b = np.full(nodes, -.5, np.float32)
    left, right = [], []
    next_node, next_leaf = 0, 0

    def build(level):
        nonlocal next_node, next_leaf
        if level == depth:
            ref = -1-next_leaf
            next_leaf += 1
            return ref
        node = next_node
        next_node += 1
        left.append(0)
        right.append(0)
        w[node, 3+raw_action if legal_split else 0] = 1.
        left[node], right[node] = build(level+1), build(level+1)
        return node

    build(0)
    logits = np.zeros((leaves, 5), np.float32)
    logits[:, raw_action] = 4.
    return SimpleNamespace(feature_names=NAMES, feature_dim=len(NAMES), role_sha256="a"*64,
                           weights={"tree.node_count": np.asarray([nodes], np.float32),
                                    "tree.leaf_count": np.asarray([leaves], np.float32),
                                    "tree.split.weight": w, "tree.split.bias": b,
                                    "tree.left_refs": np.asarray(left, np.float32),
                                    "tree.right_refs": np.asarray(right, np.float32),
                                    "tree.leaf_logits": logits})


def _encoded(patterns=None):
    mask = np.asarray(patterns if patterns is not None else [
        [int(bool(bits & (1 << a))) for a in range(4)]+[1] for bits in range(16)], np.float32)
    x = np.zeros((len(mask), len(NAMES)), np.float32)
    x[:, 3:] = mask
    return x, mask


def _route(tree, x, depth=5):
    return leaf_indices(x, tree["split_weights"], tree["split_bias"], tree["left_refs"],
                        tree["right_refs"], len(tree["split_weights"]), depth)


def test_constant_movement_gets_real_guard_and_counts_false_child():
    actor = _role()
    before = {k: v.copy() for k, v in actor.weights.items()}
    tree, report = compile_actor_structure(actor)
    assert report["feasible"]
    assert report["guard_count"] == 1
    assert report["required_capacity"]["depth"] == 1
    assert tree["leaf_count"] == 2
    assert not report["guard_policy_uses_observations"]
    assert not report["full_actor_policy_equivalence_claimed"]
    assert report["refit_contract"]["new_leaf_preferred_action_requires_new_path_proof"]
    x, mask = _encoded()
    routes = _route(tree, x)
    preferred = np.asarray([ACTIONS.index(r["frozen_preferred_action"])
                            for r in report["compiled_leaf_origins"]])
    chosen = preferred[routes]
    np.testing.assert_array_equal(chosen, np.where(mask[:, 3], 3, 4))
    assert np.all(mask[np.arange(len(mask)), chosen] == 1)
    for k in actor.weights:
        np.testing.assert_array_equal(actor.weights[k], before[k])


def test_already_certified_path_gets_no_duplicate_guard():
    actor = _role(depth=1, legal_split=True)
    actor.weights["tree.leaf_logits"][0] = [0, 0, 0, 0, 4]
    tree, report = compile_actor_structure(actor)
    assert tree["leaf_count"] == 2
    assert report["guard_count"] == 0
    assert report["original_unguarded_hard_route_parity"]
    x, _ = _encoded()
    np.testing.assert_array_equal(_route(tree, x), _route(report["original_structure"], x))


def test_wait_leaf_needs_no_guard_and_uses_float32_first_argmax_tie():
    actor = _role(raw_action=4)
    tree, report = compile_actor_structure(actor)
    assert tree["leaf_count"] == 1
    assert report["guard_count"] == 0
    actor.weights["tree.leaf_logits"][0] = [1, 1, 1, 1, 1]
    tree, report = compile_actor_structure(actor)
    assert report["leaf_guard_decisions"][0]["frozen_preferred_action"] == "UP"
    assert tree["leaf_count"] == 2


def test_capacity_failure_keeps_all_guards_and_original_partition_for_oracle():
    tree, report = compile_actor_structure(_role(depth=4))
    assert tree is None
    assert report["status"] == "structurally_infeasible"
    assert report["required_capacity"]["leaf_count"] == 32
    assert report["required_capacity"]["depth"] == 5
    assert report["guard_count"] == 16
    assert report["original_capacity"]["leaf_count"] == 16
    assert report["infeasibility_reasons"] == ["maximum_leaf_count_exceeded"]
    assert len(report["compiled_leaf_origins"]) == 32
    assert len(report["required_guarded_structure"]["split_weights"]) == 31


def test_depth_failure_is_reported_even_when_guards_are_structurally_empty():
    actor = _role(depth=5)
    tree, report = compile_actor_structure(actor)
    assert tree is None
    assert report["required_capacity"]["depth"] == 6
    assert "maximum_depth_exceeded" in report["infeasibility_reasons"]
    assert report["required_capacity"]["leaf_count"] == 64


def test_source_cost_counts_compound_public_feature_not_only_three_coefficients():
    actor = _role(depth=1, raw_action=4)
    actor.weights["tree.split.weight"][0, [0, 1, 2]] = 1.
    tree, report = compile_actor_structure(actor)
    assert tree is None
    assert report["required_capacity"]["maximum_source_features_per_node"] == 4
    assert report["infeasibility_reasons"] == ["maximum_source_feature_cost_exceeded"]


def test_non_three_sparse_input_cannot_claim_original_route_parity():
    actor = _role(depth=1, raw_action=4)
    actor.weights["tree.split.weight"][0, :4] = 1.
    tree, report = compile_actor_structure(actor)
    assert tree is None
    assert not report["original_unguarded_hard_route_parity"]
    assert "original_actor_top_three_route_not_identical" in report["infeasibility_reasons"]


def test_compiler_deterministic_and_report_json_serializable():
    import json
    actor = _role(depth=2)
    a, ar = compile_actor_structure(actor)
    b, br = compile_actor_structure(actor)
    assert json.dumps(ar, sort_keys=True) == json.dumps(br, sort_keys=True)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_compiled_structure_compatible_with_native_fixed_structure_digest():
    from backend.training.warehouse_v6_actor import FEATURE_NAMES
    actor = _role(raw_action=4)
    actor.feature_names = FEATURE_NAMES
    actor.feature_dim = len(FEATURE_NAMES)
    actor.weights["tree.split.weight"] = np.zeros((0, len(FEATURE_NAMES)), np.float32)
    tree, report = compile_actor_structure(actor)
    assert report["required_guarded_structure_sha256"] == program_structure_sha256(tree)


def test_original_hard_routes_and_paths_stable_on_threshold_boundaries():
    actor = _role(depth=2, raw_action=4)
    tree, report = compile_actor_structure(actor)
    x, _ = _encoded()
    x[:, 0] = np.resize([0., .5, np.nextafter(np.float32(.5), np.float32(1.)), 1.], len(x))
    expected = leaf_indices(x, actor.weights["tree.split.weight"], actor.weights["tree.split.bias"],
                            actor.weights["tree.left_refs"], actor.weights["tree.right_refs"], 3, 5)
    original = report["original_structure"]
    np.testing.assert_array_equal(_route(original, x), expected)
    np.testing.assert_array_equal(_route(tree, x), expected)
    assert len(report["original_capacity"]["leaf_paths"]) == 4


def _oracle_data():
    actor = _role(depth=1, raw_action=4)
    tree, _ = compile_actor_structure(actor)
    x, mask = _encoded(np.ones((200, 5)))
    x[100:, 0] = 1.
    # Left 96% RIGHT, right 98% LEFT: 95%-pure coverage=100%, 97%=50%.
    top = np.r_[np.full(96, 3), np.full(4, 4), np.full(98, 2), np.full(2, 4)]
    p = np.full((200, 5), .01, np.float32)
    p[np.arange(200), top] = .96
    return tree, x, mask, p, [f"e:{i//20}" for i in range(200)]


def test_fixed_oracle_exposes_certified_and_unrestricted_bounds_separately():
    data = _oracle_data()
    before = [a.copy() for a in data[1:4]]
    report = fixed_topology_action_oracle(*data, NAMES)
    assert report["rows"] == 200
    assert report["excluded_frames"] == 0
    assert report["bounds"]["unrestricted"]["pure_leaf_coverage_95"] == 1.
    assert report["bounds"]["unrestricted"]["pure_leaf_coverage_97"] == .5
    assert report["bounds"]["unrestricted"]["maximum_constant_action_agreement"] == .97
    assert report["bounds"]["path_certified"]["maximum_constant_action_agreement"] == .03
    assert report["bounds"]["path_certified"]["pure_leaf_coverage_95"] == 0.
    assert report["bounds"]["empirically_always_legal"]["pure_leaf_coverage_95"] == 1.
    assert report["leaves"][0]["nn_argmax_counts"] == [0, 0, 0, 96, 4]
    assert report["leaves"][0]["physically_legal_row_counts"] == [100]*5
    assert report["leaves"][0]["distinct_episode_count"] == 5
    assert not report["global_tree_optimality_claimed"]
    assert report["bounds"]["unrestricted"]["maximum_expected_feedback_coverage"] == pytest.approx(.97*.96)
    for a, saved in zip(data[1:4], before, strict=True):
        np.testing.assert_array_equal(a, saved)


def test_oracle_certified_guard_allows_movement_on_only_proven_child():
    tree, _ = compile_actor_structure(_role())
    x, mask = _encoded()
    top = np.where(mask[:, 3], 3, 4)
    p = np.zeros_like(mask)
    p[np.arange(len(p)), top] = 1.
    result = fixed_topology_action_oracle(tree, x, mask, p, ["e"]*len(p), NAMES)
    assert result["bounds"]["path_certified"]["pure_leaf_coverage_97"] == 1.
    assert result["bounds"]["path_certified"]["maximum_constant_action_agreement"] == 1.
    proof = certified_leaf_actions(tree["split_weights"], tree["split_bias"], tree["left_refs"],
                                   tree["right_refs"], 1, 2, tuple(range(3, 8)), explicit=True, wait_action=4)
    assert proof == (frozenset({4}), frozenset({3, 4}))


def test_oracle_empty_leaf_does_not_inflate_pure_coverage_or_support():
    tree, x, m, p, episodes = _oracle_data()
    x[:, 0] = 0.
    result = fixed_topology_action_oracle(tree, x, m, p, episodes, NAMES)
    empty = result["leaves"][1]
    assert empty["rows"] == empty["distinct_episode_count"] == 0
    assert empty["unrestricted"]["best_constant_action"] is None
    assert empty["empirically_always_legal_actions"] == []


@pytest.mark.parametrize("change", ["mismatch_mask", "nonbinary", "illegal_target", "not_normalized", "nan", "wait_illegal", "episodes"])
def test_oracle_rejects_incomplete_or_inconsistent_masked_targets(change):
    tree, x, m, p, episodes = _oracle_data()
    if change == "mismatch_mask":
        x[0, 3] = 0
    elif change == "nonbinary":
        m[0, 0] = x[0, 3] = .5
    elif change == "illegal_target":
        m[0, 0] = x[0, 3] = 0
    elif change == "not_normalized":
        p[0] *= .9
    elif change == "nan":
        p[0, 0] = np.nan
    elif change == "wait_illegal":
        m[0, 4] = x[0, 7] = 0
        p[0, 3] += p[0, 4]
        p[0, 4] = 0
    else:
        episodes = episodes[:-1]
    with pytest.raises(ValueError, match="complete masked NN"):
        fixed_topology_action_oracle(tree, x, m, p, episodes, NAMES)


def test_oracle_counts_95_and_97_thresholds_exactly_with_float32_targets():
    actor = _role(raw_action=4)
    tree, _ = compile_actor_structure(actor)
    x, m = _encoded(np.ones((100, 5)))
    p = np.zeros_like(m)
    p[:95, 4] = 1
    p[95:, 3] = 1
    result = fixed_topology_action_oracle(tree, x, m, p, ["e"]*100, NAMES)
    assert result["bounds"]["path_certified"]["pure_leaf_coverage_95"] == 1
    assert result["bounds"]["path_certified"]["pure_leaf_coverage_97"] == 0


@pytest.mark.parametrize("bad", ["duplicate_leaf", "cycle", "unreachable", "fractional_ref"])
def test_compiler_rejects_malformed_tree_instead_of_pruning(bad):
    actor = _role(depth=1)
    if bad == "duplicate_leaf":
        actor.weights["tree.right_refs"][0] = -1
    elif bad == "cycle":
        actor.weights["tree.left_refs"][0] = 0
    elif bad == "unreachable":
        actor.weights["tree.node_count"][0] = 0
    else:
        actor.weights["tree.left_refs"][0] = -.5
    with pytest.raises(ValueError):
        compile_actor_structure(actor)
