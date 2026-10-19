"""Compile a frozen Actor tree and diagnose its *fixed* partition.

This development-only module neither changes an Actor nor chooses environment
actions.  Guards depend only on frozen tree logits and structural path proofs;
no fit, calibration, eligibility or validation row is used by the compiler.
The resulting topology still requires independent full-NN leaf fitting through
``fit_program(..., method=PATH_CONSTRAINED_METHOD, fixed_structure=...)``.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from env.warehouse.navigation import ACTIONS
from .warehouse_v619_tree import source_feature_cost
from .warehouse_v6_program import _actor_sparse_topology
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, certified_feature_bounds, certified_leaf_actions,
    leaf_indices, validate_certification_domain,
)


MAX_DEPTH = 5
MAX_LEAVES = 16
MAX_SOURCE_FEATURES = 3
COMPILER_VERSION = "warehouse-v621-frozen-leaf-legal-guard.v1"
ORACLE_VERSION = "warehouse-v621-fixed-topology-action-oracle.v1"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _structure_sha256(structure: Mapping) -> str:
    return sha256(json.dumps(_plain(structure), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _schema(feature_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names = tuple(feature_names)
    if not names or len(names) != len(set(names)) or any(not isinstance(n, str) for n in names):
        raise ValueError("invalid Warehouse compiler feature schema")
    try:
        masks = tuple(names.index(f"legal_{action.lower()}") for action in ACTIONS)
    except ValueError as error:
        raise ValueError("Warehouse compiler requires explicit legal_ACTION features") from error
    return names, masks


def _validated_structure(structure: Mapping, feature_names: Sequence[str]) -> tuple[dict, dict]:
    """Validate all references without silently truncating an oversized tree."""
    names, _ = _schema(feature_names)
    if not isinstance(structure, Mapping):
        raise ValueError("Warehouse compiler structure must be a mapping")
    w = np.asarray(structure["split_weights"], np.float32)
    b = np.asarray(structure["split_bias"], np.float32)
    raw_left = np.asarray(structure["left_refs"])
    raw_right = np.asarray(structure["right_refs"])
    root, count = structure["root_ref"], structure["leaf_count"]
    if (type(root) not in (int, np.int32, np.int64)
            or type(count) not in (int, np.int32, np.int64)
            or int(count) < 1 or w.ndim != 2 or w.shape[1] != len(names)
            or b.shape != (len(w),) or raw_left.shape != b.shape or raw_right.shape != b.shape
            or any(not np.isfinite(a).all() for a in (w, b, raw_left, raw_right))
            or np.any(raw_left != np.floor(raw_left)) or np.any(raw_right != np.floor(raw_right))):
        raise ValueError("invalid Warehouse compiler structure tensors")
    left, right = raw_left.astype(np.int64), raw_right.astype(np.int64)
    n, count, root = len(w), int(count), int(root)
    # Shared routing/certification APIs intentionally have this single-root form.
    if root != (0 if n else -1) or n != count - 1:
        raise ValueError("invalid Warehouse compiler rooted binary tree")
    nodes, leaves, paths = set(), set(), {}
    actual_depth = 0

    def walk(ref, path):
        nonlocal actual_depth
        if ref < 0:
            leaf = -1-ref
            if leaf < 0 or leaf >= count or leaf in leaves:
                raise ValueError("invalid Warehouse compiler leaf reference")
            leaves.add(leaf)
            paths[leaf] = list(path)
            actual_depth = max(actual_depth, len(path))
            return
        if ref >= n or ref in nodes:
            raise ValueError("invalid Warehouse compiler node reference")
        nodes.add(ref)
        walk(int(left[ref]), (*path, {"node": ref, "branch": "left"}))
        walk(int(right[ref]), (*path, {"node": ref, "branch": "right"}))

    walk(root, ())
    if nodes != set(range(n)) or leaves != set(range(count)):
        raise ValueError("unreachable Warehouse compiler structure tensors")
    costs = [source_feature_cost(row, names) for row in w]
    normalized = {"root_ref": root, "split_weights": w.copy(), "split_bias": b.copy(),
                  "left_refs": left.copy(), "right_refs": right.copy(),
                  "leaf_count": count, "feature_dim": len(names)}
    return normalized, {"node_count": n, "leaf_count": count, "depth": actual_depth,
                        "maximum_source_features_per_node": max(costs, default=0),
                        "source_features_per_node": costs, "leaf_paths": paths}


def _certificates(structure: Mapping, mask_indices: tuple[int, ...]):
    return certified_leaf_actions(
        structure["split_weights"], structure["split_bias"], structure["left_refs"],
        structure["right_refs"], len(structure["split_weights"]), structure["leaf_count"],
        mask_indices, explicit=True, wait_action=ACTIONS.index("WAIT"))


def compile_actor_structure(role_actor) -> tuple[dict | None, dict]:
    """Copy internal hard topology and append deterministic physical guards.

    At each original leaf, freeze its float32 *unmasked* preferred action. If
    that action is not already path-proven, replace the leaf with the explicit
    predicate ``legal_ACTION - .5 > 1e-10``. The true child retains that action;
    the false child has WAIT as its preferred action. Both children count,
    including branches empty in all observed data. No rows are consulted.

    A guard does not preserve the Actor's masked fallback ranking or its full
    policy. All output leaf distributions must be refitted, and their *new*
    preferred actions must be proved legal again by the constrained fitter.
    """
    names, mask_indices = _schema(role_actor.feature_names)
    original, origin = _actor_sparse_topology(role_actor)
    original, original_capacity = _validated_structure(original, names)
    frozen_logits = np.asarray(role_actor.weights["tree.leaf_logits"], np.float32)
    if (frozen_logits.ndim != 2 or frozen_logits.shape[1] != len(ACTIONS)
            or len(frozen_logits) < original["leaf_count"] or not np.isfinite(frozen_logits).all()):
        raise ValueError("invalid frozen Actor tree leaf logits")
    original_proof = _certificates(original, mask_indices)
    preferred = frozen_logits[:original["leaf_count"]].argmax(-1)
    weights, bias, left, right, leaf_origins, records = [], [], [], [], [], []

    def new_leaf(source, action, branch):
        leaf = len(leaf_origins)
        leaf_origins.append({"compiled_leaf": leaf, "original_leaf": source,
                             "guard_branch": branch, "frozen_preferred_action": ACTIONS[action]})
        return -1-leaf

    def copy(ref):
        if ref < 0:
            source = -1-ref
            action = int(preferred[source])
            guarded = action not in original_proof[source]
            record = {"original_leaf": source, "frozen_preferred_action": ACTIONS[action],
                      "original_path": original_capacity["leaf_paths"][source],
                      "original_path_legal_actions": [ACTIONS[a] for a in sorted(original_proof[source])],
                      "guard_added": guarded}
            records.append(record)
            if not guarded:
                return new_leaf(source, action, "unchanged")
            node = len(weights)
            row = np.zeros(len(names), np.float32)
            row[mask_indices[action]] = 1.
            weights.append(row)
            bias.append(np.float32(-.5))
            left.append(new_leaf(source, ACTIONS.index("WAIT"), "illegal_false_wait"))
            right.append(new_leaf(source, action, "legal_true"))
            record.update({"guard_node": node, "guard_feature": names[mask_indices[action]],
                           "guard_coefficient": 1., "guard_bias": -.5})
            return node
        node = len(weights)
        weights.append(original["split_weights"][ref].copy())
        bias.append(original["split_bias"][ref])
        left.append(0)
        right.append(0)
        left[node] = copy(int(original["left_refs"][ref]))
        right[node] = copy(int(original["right_refs"][ref]))
        return node

    root = copy(original["root_ref"])
    guarded = {"root_ref": root,
               "split_weights": np.asarray(weights, np.float32).reshape(-1, len(names)),
               "split_bias": np.asarray(bias, np.float32),
               "left_refs": np.asarray(left, np.int64), "right_refs": np.asarray(right, np.int64),
               "leaf_count": len(leaf_origins), "feature_dim": len(names)}
    guarded, capacity = _validated_structure(guarded, names)
    final_proof = _certificates(guarded, mask_indices)
    for leaf, entry in enumerate(leaf_origins):
        action = ACTIONS.index(entry["frozen_preferred_action"])
        if action not in final_proof[leaf]:
            raise RuntimeError("Warehouse compiler failed its own preferred-action path proof")
        entry.update({"path_legal_actions": [ACTIONS[a] for a in sorted(final_proof[leaf])],
                      "compiled_path": capacity["leaf_paths"][leaf]})
    reasons = []
    if capacity["depth"] > MAX_DEPTH:
        reasons.append("maximum_depth_exceeded")
    if capacity["leaf_count"] > MAX_LEAVES:
        reasons.append("maximum_leaf_count_exceeded")
    if capacity["maximum_source_features_per_node"] > MAX_SOURCE_FEATURES:
        reasons.append("maximum_source_feature_cost_exceeded")
    # Frozen Actor routing applies a stable top-three sparsifier. Exact copied
    # hard-route parity is structural only when all source rows already fit it.
    sparse_parity = max(np.count_nonzero(original["split_weights"], axis=-1), default=0) <= 3
    if not sparse_parity:
        reasons.append("original_actor_top_three_route_not_identical")
    report = {
        "schema": COMPILER_VERSION, "status": "feasible" if not reasons else "structurally_infeasible",
        "feasible": not reasons, "infeasibility_reasons": reasons,
        "actor_role_sha256": role_actor.role_sha256, "actor_topology_origin": origin,
        "guard_policy": "frozen_float32_raw_leaf_argmax_then_explicit_legal_guard_else_wait",
        "guard_policy_uses_observations": False, "guard_count": sum(r["guard_added"] for r in records),
        "limits": {"depth": MAX_DEPTH, "leaves": MAX_LEAVES, "source_features_per_node": MAX_SOURCE_FEATURES},
        "original_capacity": _plain(original_capacity), "required_capacity": _plain(capacity),
        "original_structure": _plain(original), "required_guarded_structure": _plain(guarded),
        "original_structure_sha256": _structure_sha256(original),
        "required_guarded_structure_sha256": _structure_sha256(guarded),
        "original_unguarded_hard_route_parity": bool(sparse_parity),
        "routing_contract": HARD_SPLIT_ROUTING_VERSION,
        "leaf_guard_decisions": records, "compiled_leaf_origins": leaf_origins,
        "all_guards_and_empty_children_counted": True, "input_weights_modified": False,
        "masked_fallback_ranking_preserved": False, "full_actor_policy_equivalence_claimed": False,
        "leaf_outputs_reused_as_fitted_program": False,
        "refit_contract": {"method": "path_constrained_distribution_tree_v1",
                           "full_masked_nn_target_required": True,
                           "new_leaf_preferred_action_requires_new_path_proof": True},
    }
    return (None if reasons else guarded), report


def fixed_topology_action_oracle(structure: Mapping, encoded: np.ndarray,
                                 masks: np.ndarray, target_probs: np.ndarray,
                                 episodes: Sequence[str],
                                 feature_names: Sequence[str]) -> dict:
    """Exact observed-row bounds for a fixed partition and constant leaf actions.

    These diagnostics maximize over five constant actions per existing leaf.
    They ignore KL, support, Wilson bounds and unseen-state generalization.
    They prove neither optimality over other trees nor capacity sufficiency.
    """
    names, mask_indices = _schema(feature_names)
    tree, capacity = _validated_structure(structure, names)
    x, m, p = (np.asarray(item, np.float32) for item in (encoded, masks, target_probs))
    episode = np.asarray([str(value) for value in episodes])
    wait = ACTIONS.index("WAIT")
    if (x.ndim != 2 or x.shape[1] != len(names) or not len(x)
            or m.shape != (len(x), 5) or p.shape != m.shape or episode.shape != (len(x),)
            or any(not value for value in episode)
            or any(not np.isfinite(item).all() for item in (x, m, p))
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)
            or np.any(m[:, wait] != 1) or not np.array_equal(x[:, mask_indices], m)
            or np.any(p < 0) or np.any(p[m == 0] > 1e-8)
            or not np.allclose(p.sum(-1), 1., atol=1e-6, rtol=0)):
        raise ValueError("fixed topology oracle requires complete masked NN distributions")
    validate_certification_domain(x, mask_indices, certified_feature_bounds(names))
    routes = leaf_indices(x, tree["split_weights"], tree["split_bias"], tree["left_refs"],
                          tree["right_refs"], capacity["node_count"], capacity["depth"])
    proof = _certificates(tree, mask_indices)
    top = p.argmax(-1)
    reports = []
    totals = {name: {"matched_rows": 0, "expected_feedback_rows": 0.,
                     "pure_leaf_rows_95": 0, "pure_leaf_rows_97": 0}
              for name in ("path_certified", "empirically_always_legal", "unrestricted")}
    for leaf in range(tree["leaf_count"]):
        keep = routes == leaf
        count = int(keep.sum())
        counts = np.bincount(top[keep], minlength=5)
        legal_counts = m[keep].sum(0).astype(np.int64)
        always_legal = set(np.flatnonzero(legal_counts == count)) if count else set()
        expected = np.asarray([p[keep, action][top[keep] == action].sum(dtype=np.float64)
                               for action in range(5)])
        row = {"leaf": leaf, "rows": count, "distinct_episode_count": len(set(episode[keep])),
               "path": capacity["leaf_paths"][leaf], "nn_argmax_counts": counts.tolist(),
               "physically_legal_row_counts": legal_counts.tolist(),
               "path_certified_actions": [ACTIONS[a] for a in sorted(proof[leaf])],
               "empirically_always_legal_actions": [ACTIONS[a] for a in sorted(always_legal)]}
        for name, choices in (("path_certified", proof[leaf]),
                              ("empirically_always_legal", always_legal),
                              ("unrestricted", range(5))):
            choices = sorted(choices)
            action = min(choices, key=lambda a: (-counts[a], a)) if choices and count else None
            matched = int(counts[action]) if action is not None else 0
            fraction = matched/count if count else 0.
            exp_action = min(choices, key=lambda a: (-expected[a], a)) if choices and count else None
            exp_rows = float(expected[exp_action]) if exp_action is not None else 0.
            row[name] = {"best_constant_action": ACTIONS[action] if action is not None else None,
                         "maximum_matching_rows": matched, "maximum_agreement": fraction,
                         "best_expected_feedback_action": ACTIONS[exp_action] if exp_action is not None else None,
                         "maximum_expected_feedback_rows": exp_rows}
            totals[name]["matched_rows"] += matched
            totals[name]["expected_feedback_rows"] += exp_rows
            for threshold in (95, 97):
                if count and fraction >= threshold/100-1e-12:
                    totals[name][f"pure_leaf_rows_{threshold}"] += count
        reports.append(row)
    for total in totals.values():
        total.update({"maximum_constant_action_agreement": total["matched_rows"]/len(x),
                      "maximum_expected_feedback_coverage": total["expected_feedback_rows"]/len(x),
                      "pure_leaf_coverage_95": total["pure_leaf_rows_95"]/len(x),
                      "pure_leaf_coverage_97": total["pure_leaf_rows_97"]/len(x)})
    return {"schema": ORACLE_VERSION, "program_structure_sha256": _structure_sha256(tree),
            "rows": len(x), "distinct_episode_count": len(set(episode)), "excluded_frames": 0,
            "actions": list(ACTIONS), "capacity": _plain(capacity), "leaves": reports, "bounds": totals,
            "routing_contract": HARD_SPLIT_ROUTING_VERSION, "input_arrays_modified": False,
            "scope": "observed_rows_fixed_partition_constant_leaf_actions_only",
            "ignores": ["KL", "support", "Wilson_bounds", "unseen_states"],
            "empirical_legality_is_not_a_path_certificate": True,
            "unrestricted_bound_ignores_path_and_whole_tree_legality": True,
            "global_tree_optimality_claimed": False}
