#!/usr/bin/env python3
"""Independent, zero-environment audit of the complete v6.21 diagnostic batch.

The audit parses raw program JSON and implements routing and every statistic
itself. It never calls the production program metric, routing, qualification,
oracle, fitting, environment, training, or summary functions. Only the frozen
NN forward pass, public feature encoder, immutable input loader and hashes are
reused. Development evidence remains development evidence after this audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.warehouse import v621_protocol as protocol

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
TIE_EPSILON = 1.e-10
FLOAT_ATOL = 2.e-6
FLOAT_RTOL = 2.e-7
ZERO_FIELDS = ("environment_resets", "actor_updates", "PPO_joint_steps")
FALSE_FIELDS = ("publishable", "admission_evidence", "fresh_validation_evidence")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(child) for child in value]
    return value


def digest(value):
    return sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"audit byte binding failed: {path}")
    return json.loads(raw)


def assert_equal(actual, expected, path="root"):
    """Exact structures/counts; tolerance only for finite floating arithmetic."""
    actual, expected = plain(actual), plain(expected)
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"audit keys differ at {path}")
        for key in expected:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"audit sequence differs at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_equal(left, right, f"{path}[{index}]")
    elif type(expected) is float:
        if (type(actual) is not float or not np.isfinite(actual) or not np.isfinite(expected)
                or not np.isclose(actual, expected, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)):
            raise RuntimeError(f"audit float differs at {path}: {actual!r} != {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(f"audit value differs at {path}: {actual!r} != {expected!r}")


def softmax(logits, masks):
    selected = np.where(masks > .5, np.asarray(logits, np.float32), -1.e9)
    values = np.exp(selected-selected.max(axis=-1, keepdims=True))
    return values/values.sum(axis=-1, keepdims=True)


def wilson(fraction, count):
    if count == 0:
        return 0.
    z2 = 1.645**2
    return float(max(0., (fraction+z2/(2*count)-1.645*np.sqrt(
        (fraction*(1-fraction)+z2/(4*count))/count))/(1+z2/count)))


def structure(raw, feature_names, *, enforce_limits=True):
    """Validate a real binary tree, including every empty leaf and guard."""
    names = tuple(feature_names)
    w = np.asarray(raw["split_weights"], np.float32)
    if w.size == 0:
        w = np.empty((0, len(names)), np.float32)
    b = np.asarray(raw["split_bias"], np.float32)
    left0, right0 = np.asarray(raw["left_refs"]), np.asarray(raw["right_refs"])
    root, count = raw["root_ref"], raw["leaf_count"]
    if (type(root) is not int or type(count) is not int or not 1 <= count <= 128
            or w.ndim != 2 or w.shape[1] != len(names) or b.shape != (len(w),)
            or left0.shape != b.shape or right0.shape != b.shape
            or any(not np.isfinite(a).all() for a in (w, b, left0, right0))
            or np.any(left0 != np.floor(left0)) or np.any(right0 != np.floor(right0))
            or root != (0 if len(w) else -1) or len(w) != count-1):
        raise RuntimeError("audit invalid rooted tree tensors")
    left, right = left0.astype(np.int64), right0.astype(np.int64)
    nodes, leaves, paths = set(), set(), {}
    depth = 0
    def walk(ref, path):
        nonlocal depth
        if ref < 0:
            leaf = -1-ref
            if leaf >= count or leaf in leaves:
                raise RuntimeError("audit invalid leaf graph")
            leaves.add(leaf)
            paths[leaf] = list(path)
            depth = max(depth, len(path))
            return
        if ref >= len(w) or ref in nodes:
            raise RuntimeError("audit invalid split graph")
        nodes.add(ref)
        walk(int(left[ref]), [*path, {"node": ref, "branch": "left"}])
        walk(int(right[ref]), [*path, {"node": ref, "branch": "right"}])
    walk(root, [])
    if nodes != set(range(len(w))) or leaves != set(range(count)):
        raise RuntimeError("audit unreachable tree tensors")
    costs = np.count_nonzero(w, axis=1)
    if "public_hard_risk_context" in names:
        costs += w[:, names.index("public_hard_risk_context")] != 0
    if enforce_limits and (depth > 5 or count > 16 or np.any(costs > 3)):
        raise RuntimeError("audit tree exceeds registered capacity including guards")
    tree = {"root_ref": root, "split_weights": w, "split_bias": b,
        "left_refs": left, "right_refs": right, "leaf_count": count, "feature_dim": len(names)}
    capacity = {"node_count": len(w), "leaf_count": count, "depth": depth,
        "maximum_source_features_per_node": int(max(costs, default=0)),
        "source_features_per_node": costs.tolist(), "leaf_paths": paths}
    return tree, capacity


def route(tree, encoded):
    """Scalar sparse dot products in float64 from stored float32 inputs."""
    x = np.asarray(encoded, np.float32)
    w, b = tree["split_weights"], tree["split_bias"]
    leaves = np.empty(len(x), np.int64)
    for index, row in enumerate(x):
        node, visits = tree["root_ref"], 0
        while node >= 0:
            visits += 1
            if visits > len(w):
                raise RuntimeError("audit cyclic hard route")
            value = 0.
            for feature in np.flatnonzero(w[node]):
                value += float(row[feature])*float(w[node, feature])
            value += float(b[node])
            node = int(tree["right_refs"][node] if value > TIE_EPSILON else tree["left_refs"][node])
        leaves[index] = -1-node
    return leaves


def certificates(tree, feature_names, *, explicit_crossing=False):
    """A pure legal-bit ancestor proves a movement; WAIT is globally legal."""
    mask_indices = tuple(feature_names.index(f"legal_{a.lower()}") for a in ACTIONS)
    result = {}
    def walk(ref, proven):
        if ref < 0:
            result[-1-ref] = frozenset(proven)
            return
        used = np.flatnonzero(tree["split_weights"][ref])
        left, right = set(), set()
        if len(used) == 1 and int(used[0]) in mask_indices:
            # Setting this bit to zero fixes the score at the stored bias.
            # An impossible branch may be vacuously legal but still counts.
            action = mask_indices.index(int(used[0]))
            zero = float(tree["split_bias"][ref])
            one = zero+float(tree["split_weights"][ref, used[0]])
            if explicit_crossing:
                if zero <= TIE_EPSILON < one:
                    right.add(action)
                if one <= TIE_EPSILON < zero:
                    left.add(action)
            else:
                (right if zero <= TIE_EPSILON else left).add(action)
        walk(int(tree["left_refs"][ref]), proven | left)
        walk(int(tree["right_refs"][ref]), proven | right)
    walk(tree["root_ref"], {4})
    return result


def context(raw, data, names):
    tree, capacity = structure(dict(raw, leaf_count=len(raw["leaf_logits"])), names)
    x, masks, p, sampled, episode, scenario = data
    leaves = route(tree, x)
    logits = np.asarray(raw["leaf_logits"], np.float32)
    if logits.shape != (tree["leaf_count"], 5) or not np.isfinite(logits).all():
        raise RuntimeError("audit invalid five-action leaf distribution")
    proof = certificates(tree, names)
    raw_logits = logits[leaves]
    q = softmax(raw_logits, masks)
    raw_action, top, nn_top = raw_logits.argmax(-1), q.argmax(-1), p.argmax(-1)
    raw_legal = masks[np.arange(len(x)), raw_action] > .5
    proven = np.asarray([int(a) in proof[int(leaf)] for a, leaf in zip(raw_action, leaves, strict=True)])
    row_kl = np.sum(p*(np.log(np.maximum(p, 1.e-8))-np.log(np.maximum(q, 1.e-8))), axis=1)
    return {"tree": tree, "capacity": capacity, "proof": proof, "x": x, "m": masks, "p": p,
        "sampled": sampled, "episode": episode, "scenario": scenario, "leaves": leaves,
        "q": q, "raw_action": raw_action, "top": top, "nn_top": nn_top,
        "raw_legal": raw_legal, "proven": proven, "row_kl": row_kl, "names": tuple(names)}


def section(c, selected_leaves, purpose):
    selected = np.isin(c["leaves"], sorted(selected_leaves))
    n, count = len(selected), int(selected.sum())
    result = {"sample_count": n, "rows": count, "coverage": count/n,
        "actionable_rows": 0, "actionable_coverage": 0., "agreement": 0.,
        "agreement_wilson_lower_90": 0., "kl": None, "illegal_count": 0,
        "masked_illegal_count": 0, "path_uncertified_count": 0, "certified_rows": 0,
        "distinct_episode_count": 0, "distinct_scenario_seed_count": 0, "max_episode_fraction": 0.,
        "physical_frame_count": n, "excluded_frames": 0,
        "expected_actionable_coverage": 0., "expected_actionable_rows": 0.,
        "per_nn_action": {}, "public_strata": {},
        "whole_tree_raw_illegal_count": int((~c["raw_legal"]).sum()),
        "whole_tree_path_uncertified_count": int((~c["proven"]).sum())}
    if not count:
        return result
    m, p, q = c["m"][selected], c["p"][selected], c["q"][selected]
    actions, top, nn_top = c["sampled"][selected], c["top"][selected], c["nn_top"][selected]
    legal, proven = c["raw_legal"][selected], c["proven"][selected]
    valid = legal & proven
    matched = valid & (m[np.arange(count), actions] > .5) & (top == actions)
    if purpose == "feedback":
        matched &= actions == nn_top
    expected = valid*p[np.arange(count), top]
    if purpose == "feedback":
        expected *= top == nn_top
    agreement = float((top == nn_top).mean())
    frequencies = Counter(c["episode"][selected])
    kl = c["row_kl"][selected]
    result.update({"actionable_rows": int(matched.sum()), "actionable_coverage": float(matched.sum()/n),
        "agreement": agreement, "agreement_wilson_lower_90": wilson(agreement, count),
        "kl": float(kl.mean()), "illegal_count": int((~legal).sum()),
        "masked_illegal_count": int((m[np.arange(count), top] <= .5).sum()),
        "path_uncertified_count": int((~proven).sum()), "distinct_episode_count": len(frequencies),
        "distinct_scenario_seed_count": len(set(c["scenario"][selected])),
        "max_episode_fraction": max(frequencies.values())/count,
        "expected_actionable_rows": float(expected.sum()), "expected_actionable_coverage": float(expected.sum()/n)})
    for action, label in enumerate(ACTIONS):
        chosen = nn_top == action
        denominator = int((c["nn_top"] == action).sum())
        result["per_nn_action"][label] = {"rows": int(chosen.sum()), "physical_frames": denominator,
            "qualified_coverage": float(chosen.sum()/denominator) if denominator else None,
            "agreement": float((top[chosen] == nn_top[chosen]).mean()) if chosen.any() else None,
            "actual_actionable_rows": int(matched[chosen].sum()), "expected_actionable_rows": float(expected[chosen].sum())}
    names, x = c["names"], c["x"]
    conflict = x[:, names.index("same_target_collision_streak_capped")] > 0
    near = x[:, names.index("robots_shortest_distance")]*42 <= 2
    low = x[:, names.index("self_charger_return_margin")]*50 <= 4
    for label, all_rows in {"ordinary": ~(conflict | near | low), "conflict": conflict,
                           "near_robots": near, "low_return_margin": low}.items():
        chosen, denominator = all_rows[selected], int(all_rows.sum())
        result["public_strata"][label] = {"rows": int(chosen.sum()), "physical_frames": denominator,
            "qualified_coverage": float(chosen.sum()/denominator) if denominator else None,
            "agreement": float((top[chosen] == nn_top[chosen]).mean()) if chosen.any() else None,
            "kl": float(kl[chosen].mean()) if chosen.any() else None,
            "illegal_count": int((~legal[chosen]).sum()), "actual_actionable_rows": int(matched[chosen].sum()),
            "expected_actionable_rows": float(expected[chosen].sum())}
    return result


def metrics(raw, data, names):
    c = context(raw, data, names)
    feedback = section(c, raw["feedback_leaves"], "feedback")
    explanation = section(c, raw["explanation_leaves"], "explanation")
    strata = {}
    for index, label in enumerate(ACTIONS):
        take = c["nn_top"] == index
        strata[label] = {"physical_frames": int(take.sum()),
            "whole_tree_raw_illegal_count": int((~c["raw_legal"] & take).sum()),
            "whole_tree_path_uncertified_count": int((~c["proven"] & take).sum()),
            "whole_tree_agreement": float((c["top"][take] == c["nn_top"][take]).mean()) if take.any() else None,
            "whole_tree_mean_kl": float(c["row_kl"][take].mean()) if take.any() else None,
            "feedback_qualified_frames": int((take & np.isin(c["leaves"], raw["feedback_leaves"])).sum()),
            "explanation_qualified_frames": int((take & np.isin(c["leaves"], raw["explanation_leaves"])).sum())}
    return {"feedback": feedback, "explanation": explanation,
        "whole_tree_mean_kl": float(c["row_kl"].mean()), "nn_five_action_strata": strata,
        "illegal_count_scope": "whole_tree_raw_before_masking_and_qualification",
        "qualified_illegal_count_scope": "qualified_region_before_per_frame_action_matching",
        "all_physical_frames": len(data[0]), "excluded_frames": 0}, c


def eligible(value, agreement):
    return (value["rows"] >= 20 and value["distinct_episode_count"] >= 3
        and value["distinct_scenario_seed_count"] >= 3 and value["max_episode_fraction"] <= .5
        and value["agreement"] >= agreement and value["agreement_wilson_lower_90"] >= agreement-.08
        and value["kl"] is not None and value["kl"] <= .25 and value["illegal_count"] == 0
        and value["path_uncertified_count"] == 0)


def oracle(raw_tree, data, names):
    tree, capacity = structure(raw_tree, names, enforce_limits=False)
    x, m, p, _, episodes, _ = data
    routes, proof, top = route(tree, x), certificates(tree, names, explicit_crossing=True), p.argmax(-1)
    totals = {name: {"matched_rows": 0, "expected_feedback_rows": 0.,
        "pure_leaf_rows_95": 0, "pure_leaf_rows_97": 0}
        for name in ("path_certified", "empirically_always_legal", "unrestricted")}
    reports = []
    for leaf in range(tree["leaf_count"]):
        take = routes == leaf
        n = int(take.sum())
        counts = np.bincount(top[take], minlength=5)
        legal_counts = m[take].sum(0).astype(np.int64)
        always = set(np.flatnonzero(legal_counts == n)) if n else set()
        expected = np.asarray([p[take, a][top[take] == a].sum(dtype=np.float64) for a in range(5)])
        row = {"leaf": leaf, "rows": n, "distinct_episode_count": len(set(episodes[take])),
            "path": capacity["leaf_paths"][leaf], "nn_argmax_counts": counts.tolist(),
            "physically_legal_row_counts": legal_counts.tolist(),
            "path_certified_actions": [ACTIONS[a] for a in sorted(proof[leaf])],
            "empirically_always_legal_actions": [ACTIONS[a] for a in sorted(always)]}
        for name, choices in (("path_certified", proof[leaf]), ("empirically_always_legal", always), ("unrestricted", range(5))):
            choices = sorted(choices)
            action = min(choices, key=lambda a: (-counts[a], a)) if choices and n else None
            exp_action = min(choices, key=lambda a: (-expected[a], a)) if choices and n else None
            matched = int(counts[action]) if action is not None else 0
            fraction = matched/n if n else 0.
            mass = float(expected[exp_action]) if exp_action is not None else 0.
            row[name] = {"best_constant_action": ACTIONS[action] if action is not None else None,
                "maximum_matching_rows": matched, "maximum_agreement": fraction,
                "best_expected_feedback_action": ACTIONS[exp_action] if exp_action is not None else None,
                "maximum_expected_feedback_rows": mass}
            totals[name]["matched_rows"] += matched
            totals[name]["expected_feedback_rows"] += mass
            for threshold in (95, 97):
                if n and fraction >= threshold/100-1.e-12:
                    totals[name][f"pure_leaf_rows_{threshold}"] += n
        reports.append(row)
    for total in totals.values():
        total.update({"maximum_constant_action_agreement": total["matched_rows"]/len(x),
            "maximum_expected_feedback_coverage": total["expected_feedback_rows"]/len(x),
            "pure_leaf_coverage_95": total["pure_leaf_rows_95"]/len(x),
            "pure_leaf_coverage_97": total["pure_leaf_rows_97"]/len(x)})
    return {"schema": "warehouse-v621-fixed-topology-action-oracle.v1", "program_structure_sha256": digest(tree),
        "rows": len(x), "distinct_episode_count": len(set(episodes)), "excluded_frames": 0,
        "actions": list(ACTIONS), "capacity": capacity, "leaves": reports, "bounds": totals,
        "routing_contract": "fp64-shared-left-tie-1e-10.v2",
        "input_arrays_modified": False, "scope": "observed_rows_fixed_partition_constant_leaf_actions_only",
        "ignores": ["KL", "support", "Wilson_bounds", "unseen_states"],
        "empirical_legality_is_not_a_path_certificate": True,
        "unrestricted_bound_ignores_path_and_whole_tree_legality": True, "global_tree_optimality_claimed": False}


def require_complete(plan_path):
    path = Path(plan_path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    if plan.get("jobs") != protocol.jobs():
        raise RuntimeError("audit parent must contain the exact eighteen jobs")
    for job in plan["jobs"]:
        directory = path.parent/"jobs"/job["job_id"]
        if (directory/"failed.json").exists() or any(not (directory/name).is_file()
                for name in ("started.json", "result.json", "completed.json")):
            raise RuntimeError("all eighteen jobs must complete before independent batch audit")
    if not (path.parent/"summary.json").is_file():
        raise RuntimeError("complete native summary required before independent batch audit")
    return path, plan


def compiler_review(report, role_actor):
    """Reconstruct every guard from frozen weights without the compiler API."""
    names, weights = tuple(role_actor.feature_names), role_actor.weights
    n = int(weights["tree.node_count"][0])
    original = {"root_ref": 0 if n else -1,
        "split_weights": weights["tree.split.weight"][:n], "split_bias": weights["tree.split.bias"][:n],
        "left_refs": weights["tree.left_refs"][:n].astype(np.int64),
        "right_refs": weights["tree.right_refs"][:n].astype(np.int64),
        "leaf_count": int(weights["tree.leaf_count"][0])}
    original, capacity = structure(plain(original), names, enforce_limits=False)
    assert_equal(report["original_structure"], original, "compiler.original_structure")
    assert_equal(report["original_capacity"], capacity, "compiler.original_capacity")
    if report["original_structure_sha256"] != digest(original):
        raise RuntimeError("audit original compiler topology SHA differs")
    proof = certificates(original, names, explicit_crossing=True)
    preferred = weights["tree.leaf_logits"][:original["leaf_count"]].argmax(-1)
    w, b, left, right, origins = [], [], [], [], []
    guards = 0
    def leaf(source, action, branch):
        origins.append({"original_leaf": source, "frozen_preferred_action": ACTIONS[action], "guard_branch": branch})
        return -len(origins)
    def copy(ref):
        nonlocal guards
        if ref < 0:
            source, action = -1-ref, int(preferred[-1-ref])
            if action in proof[source]:
                return leaf(source, action, "unchanged")
            guards += 1
            node = len(w)
            row = np.zeros(len(names), np.float32)
            row[names.index(f"legal_{ACTIONS[action].lower()}")] = 1.
            w.append(row)
            b.append(-.5)
            left.append(leaf(source, 4, "illegal_false_wait"))
            right.append(leaf(source, action, "legal_true"))
            return node
        node = len(w)
        w.append(original["split_weights"][ref])
        b.append(float(original["split_bias"][ref]))
        left.append(0)
        right.append(0)
        left[node], right[node] = copy(int(original["left_refs"][ref])), copy(int(original["right_refs"][ref]))
        return node
    root = copy(original["root_ref"])
    guarded, required = structure({"root_ref": root, "split_weights": plain(w), "split_bias": b,
        "left_refs": left, "right_refs": right, "leaf_count": len(origins)}, names, enforce_limits=False)
    assert_equal(report["required_guarded_structure"], guarded, "compiler.required_structure")
    assert_equal(report["required_capacity"], required, "compiler.required_capacity")
    final_proof = certificates(guarded, names, explicit_crossing=True)
    for index, origin in enumerate(origins):
        row = report["compiled_leaf_origins"][index]
        expected = dict(origin, compiled_leaf=index,
            path_legal_actions=[ACTIONS[a] for a in sorted(final_proof[index])],
            compiled_path=required["leaf_paths"][index])
        assert_equal(row, expected, f"compiler.compiled_leaf_{index}")
        if ACTIONS.index(origin["frozen_preferred_action"]) not in final_proof[index]:
            raise RuntimeError("audit compiled preferred action lacks full path proof")
    sparse = max(np.count_nonzero(original["split_weights"], axis=1), default=0) <= 3
    reasons = []
    if required["depth"] > 5:
        reasons.append("maximum_depth_exceeded")
    if required["leaf_count"] > 16:
        reasons.append("maximum_leaf_count_exceeded")
    if required["maximum_source_features_per_node"] > 3:
        reasons.append("maximum_source_feature_cost_exceeded")
    if not sparse:
        reasons.append("original_actor_top_three_route_not_identical")
    if (report["guard_count"] != guards or report["infeasibility_reasons"] != reasons
            or report["feasible"] is not (not reasons)
            or report["required_guarded_structure_sha256"] != digest(guarded)
            or report["guard_policy_uses_observations"] is not False
            or report["all_guards_and_empty_children_counted"] is not True
            or report["input_weights_modified"] is not False):
        raise RuntimeError("audit compiler capacity, empty-branch or authority declarations differ")
    return {"passed": True, "feasible": not reasons, "guard_count": guards,
        "required_depth": required["depth"], "required_leaves": required["leaf_count"],
        "infeasibility_reasons": reasons}, original


def budget_review(search, report, arm, native):
    runs, calls = search.get("search_runs", []), search.get("leaf_optimization_calls", 0)
    outside = report.get("leaf_optimization_calls_outside_search", 0)
    if (type(calls) is not int or type(outside) is not int or calls < 0 or outside < 0
            or calls > 19984 or outside > 16 or calls+outside > 20000
            or not isinstance(runs, list) or sum(len(r["rounds"]) for r in runs) > 60):
        raise RuntimeError("audit global leaf/round budget exceeded")
    rounds = sum(len(r["rounds"]) for r in runs)
    if arm in ("A", "C"):
        if len(runs) != 8 or search.get("global_search_rounds") != rounds:
            raise RuntimeError("audit search initialization/round accounting differs")
        if sum(r["leaf_optimization_calls"] for r in runs) != calls:
            raise RuntimeError("audit search API history sum differs")
        if sum(r["actual_convex_solver_calls"] for r in runs) != search["actual_convex_solver_calls"]:
            raise RuntimeError("audit search solver history sum differs")
    elif arm != "B" or runs or calls:
        raise RuntimeError("audit compiled topology unexpectedly searched")
    solves = sum(r["enumerated_action_solves"] for r in report.get("constrained_leaf_fits", {}).values())
    if report and (outside != report["leaf_count"]
            or set(report["constrained_leaf_fits"]) != {str(i) for i in range(outside)}
            or report.get("convex_solver_calls_outside_search") != solves):
        raise RuntimeError("audit final fixed-tree refit accounting differs")
    derived = {"actual_total_rounds": rounds, "search_leaf_api_calls": calls,
        "final_refit_leaf_api_calls": outside, "end_to_end_leaf_api_calls": calls+outside,
        "actual_search_convex_solver_calls": search.get("actual_convex_solver_calls", 0),
        "actual_final_convex_solver_calls": solves}
    for key, value in derived.items():
        assert_equal(native[key], value, f"budget.{key}")
    if native.get("passed") is not True or not all(native["checks"].values()):
        raise RuntimeError("audit native budget result does not pass")
    if "leaf_api_attempts" in search and len(search["leaf_api_attempts"]) != calls:
        raise RuntimeError("audit explicit leaf attempt ledger count differs")
    return dict(derived, passed=True)


def gate(value):
    f, e = value["feedback"], value["explanation"]
    return bool(f["coverage"] >= .45 and e["coverage"] >= .45 and f["expected_actionable_coverage"] >= .40
        and f["agreement"] >= .95 and e["agreement"] >= .97
        and f["kl"] is not None and e["kl"] is not None and f["kl"] <= .20 and e["kl"] <= .20
        and f["whole_tree_raw_illegal_count"] == e["whole_tree_raw_illegal_count"] == 0
        and f["whole_tree_path_uncertified_count"] == e["whole_tree_path_uncertified_count"] == 0)


def assert_authority_closed(row, label):
    if any(type(row.get(k)) is not int or row[k] != 0 for k in ZERO_FIELDS) or any(
            row.get(k) is not False for k in FALSE_FIELDS):
        raise RuntimeError(f"audit zero-update/reset or closed-publication declaration differs: {label}")


def audit(plan_path, *, root=ROOT):
    path, plan = require_complete(plan_path)
    root = Path(root).resolve()
    plan_sha = file_sha(path)
    claim = json.loads((path.parent/"plan.claim.json").read_bytes())
    if (claim.get("plan_sha256") != plan_sha or claim.get("single_use") is not True
            or plan.get("candidate_id") != protocol.CANDIDATE_ID
            or plan.get("folds") != protocol.grouped_folds()
            or plan.get("round_budget_contract") != protocol.round_budget_contract()
            or plan.get("gates") != protocol.GATES or plan.get("scientific_retry_permitted") is not False):
        raise RuntimeError("audit immutable parent protocol or claim differs")
    assert_authority_closed(plan, "parent")
    identity = plan["identity"]
    files = protocol.source_closure_files(root)
    if (identity["source_files_sha256"] != files or identity["source_closure_sha256"] != digest(files)
            or identity["recipe_sha256"] != protocol.validate_recipe(root)
            or identity["actor_sha256"] != protocol.ACTOR_SHA256
            or identity["actor_role_sha256"] != protocol.ROLE_SHA256):
        raise RuntimeError("audit source, recipe or dual Actor identity differs")
    episodes, manifest = protocol.checked_inputs(root)
    if manifest != identity["inputs"]:
        raise RuntimeError("audit spent input manifest differs")
    actor, actor_path = protocol.checked_actor(root)
    if protocol.target_identity(actor, episodes) != identity["full_nn_targets"]:
        raise RuntimeError("audit complete NN target binding differs")
    from env.warehouse.public_action_features_v617 import encode_actor_tree_features, ACTION_PHYSICS_CONFIG_SHA256
    from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256
    arrays, completed, reports = {}, [], []
    for fold in plan["folds"]:
        for role in protocol.ROLES:
            for split in protocol.SPLITS:
                rows = [row for case in fold["splits"][split] for row in episodes[case["base"]][role]]
                obs = np.asarray([r["obs"] for r in rows], np.float32)
                m = np.asarray([r["mask"] for r in rows], np.float32)
                x = encode_actor_tree_features(obs, m)
                p = softmax(actor.roles[role].logits_encoded(x), m)
                stored = np.asarray([r["nn_probabilities"] for r in rows], np.float32)
                if not np.allclose(p, stored, atol=2.e-6, rtol=2.e-5):
                    raise RuntimeError("audit independently recomputed role NN five-probability target differs")
                arrays[(fold["fold"], role, split)] = (x, m, p,
                    np.asarray([ACTIONS.index(r["sampled_action"]) for r in rows], np.int64),
                    np.asarray([str(r["episode"]) for r in rows]), np.asarray([str(r["base_seed"]) for r in rows]))
    for job in plan["jobs"]:
        directory = path.parent/"jobs"/job["job_id"]
        done = json.loads((directory/"completed.json").read_bytes())
        result = read_bound(directory/"result.json", done["result_sha256"])
        started = json.loads((directory/"started.json").read_bytes())
        assert_authority_closed(result, job["job_id"])
        assert_authority_closed(started, job["job_id"]+".start")
        if (started.get("job") != job or result.get("job") != job
                or result.get("plan_sha256") != plan_sha or started.get("plan_sha256") != plan_sha
                or done.get("plan_sha256") != plan_sha or done.get("status") != result.get("status")
                or result.get("actor_weights_unchanged") is not True or result.get("excluded_frames") != 0
                or result.get("scientific_retry_permitted") is not False
                or result.get("fold_assignment_sha256") != plan["folds"][job["fold"]]["assignment_sha256"]):
            raise RuntimeError("audit child/start/completion binding differs")
        data = {name: arrays[(job["fold"], job["role"], name)] for name in protocol.SPLITS}
        assert_equal(result["physical_frames_per_split"], {k: len(a[0]) for k, a in data.items()}, "frame_counts")
        search = read_bound(result["search_report"]["path"], result["search_report"]["sha256"])
        program_report, independently_recomputed, compiler = {}, None, None
        names = tuple(actor.roles[job["role"]].feature_names)
        if result["compiler_report"] is not None:
            comp = read_bound(result["compiler_report"]["path"], result["compiler_report"]["sha256"])
            compiler, original = compiler_review(comp, actor.roles[job["role"]])
            observed = json.loads((directory/"original-topology-oracle.json").read_bytes())
            assert_equal({name: oracle(original, a, names) for name, a in data.items()}, observed, "original_oracle")
        if result["status"] == "structurally_infeasible":
            if (job["arm"] != "B" or compiler is None or compiler["feasible"]
                    or result["program"] is not None or result["metrics"] is not None
                    or result["development_gate"]["passed"] is not False):
                raise RuntimeError("audit compiler scientific infeasibility differs")
        elif result["status"] == "completed":
            raw = read_bound(result["program"]["path"], result["program"]["sha256"])
            program_report = raw["report"]
            topology = read_bound(result["structure"]["path"], result["structure"]["sha256"])
            tree, capacity = structure(topology, names)
            embedded, _ = structure(dict(raw, leaf_count=len(raw["leaf_logits"])), names)
            if (raw["role"] != job["role"] or raw["actor_role_sha256"] != protocol.ROLE_SHA256[job["role"]]
                    or raw["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256
                    or raw["action_physics_config_sha256"] != ACTION_PHYSICS_CONFIG_SHA256
                    or program_report["fit_method"] != "path_constrained_distribution_tree_v1"
                    or result["leaf_count"] != len(raw["leaf_logits"])
                    or program_report["fixed_structure_sha256"] != digest(tree)
                    or program_report["program_structure_sha256"] != digest(tree)):
                raise RuntimeError("audit program/schema/fixed topology identity differs")
            assert_equal(tree, embedded, "embedded_program_topology")
            fold = plan["folds"][job["fold"]]
            assert_equal(program_report["split_episode_ids"], {name: sorted(
                f'spent-v620-base-{r["base"]}' for r in fold["splits"][name]) for name in protocol.SPLITS}, "split_episode_ids")
            assert_equal(program_report["split_scenario_seed_ids"], {name: sorted(
                str(r["scenario_seed"]) for r in fold["splits"][name]) for name in protocol.SPLITS}, "split_scenarios")
            if program_report["fit_tree"]["calibration_eligibility_validation_used"] is not False:
                raise RuntimeError("audit topology declares use of non-fit partitions")
            independently_recomputed, contexts = {}, {}
            for name in protocol.SPLITS:
                independently_recomputed[name], contexts[name] = metrics(raw, data[name], names)
                assert_equal(independently_recomputed[name], result["metrics"][name], f'metrics.{job["job_id"]}.{name}')
            proof = contexts["fit"]["proof"]
            fit_counts = np.bincount(contexts["fit"]["leaves"], minlength=tree["leaf_count"])
            assert_equal(fit_counts.tolist(), program_report["fit_leaf_row_counts"], "fit_leaf_row_counts")
            chosen = np.asarray(raw["leaf_logits"], np.float32).argmax(-1)
            expected_feedback, expected_explanation = [], []
            for leaf in range(tree["leaf_count"]):
                f = section(contexts["eligibility"], {leaf}, "feedback")
                e = section(contexts["eligibility"], {leaf}, "explanation")
                assert_equal({"feedback": f, "explanation": e}, program_report["leaf_eligibility"][str(leaf)], f"leaf_eligibility.{leaf}")
                if int(chosen[leaf]) not in proof[leaf]:
                    raise RuntimeError("audit final calibrated raw action lacks path proof")
                if fit_counts[leaf] and eligible(f, .93):
                    expected_feedback.append(leaf)
                if fit_counts[leaf] and eligible(e, .95):
                    expected_explanation.append(leaf)
            assert_equal(raw["feedback_leaves"], expected_feedback, "feedback_leaf_selection")
            assert_equal(raw["explanation_leaves"], expected_explanation, "explanation_leaf_selection")
            passed = gate(independently_recomputed["validation"])
            if result["development_gate"]["passed"] is not passed or done["passed"] is not passed:
                raise RuntimeError("audit independently recomputed gate differs")
            observed = json.loads((directory/"fixed-topology-oracle.json").read_bytes())
            assert_equal({name: oracle(tree, a, names) for name, a in data.items()}, observed, "fixed_oracle")
            if job["arm"] in ("A", "C"):
                if (search["actor_role_sha256"] != protocol.ROLE_SHA256[job["role"]]
                        or search["fit_rows"] != len(data["fit"][0])
                        or search["fit_episode_count"] != len(set(data["fit"][4]))
                        or search["fit_target_sha256"] != sha256(np.ascontiguousarray(data["fit"][2], dtype="<f4").tobytes()).hexdigest()
                        or search["topology_uses_fit_only"] is not True
                        or search["calibration_eligibility_validation_used"] is not False):
                    raise RuntimeError("audit search fit-only/complete NN target binding differs")
        else:
            raise RuntimeError("audit unknown completed scientific status")
        budget = budget_review(search, program_report, job["arm"], result["budget_audit"])
        reports.append({"job": job, "passed": True, "status": result["status"],
            "result_sha256": done["result_sha256"], "compiler": compiler,
            "budget": budget, "validation": independently_recomputed["validation"] if independently_recomputed else None})
        completed.append(result)
    summary = json.loads((path.parent/"summary.json").read_bytes())
    arms = {}
    for arm in ("A", "B", "C"):
        rows = [r for r in reports if r["job"]["arm"] == arm]
        feasible = [r for r in rows if r["status"] == "completed"]
        e = [r["validation"]["explanation"] for r in feasible]
        f = [r["validation"]["feedback"] for r in feasible]
        max_kl = max((r["kl"] for r in e+f if r["kl"] is not None), default=None)
        score = [min((r["coverage"] for r in e), default=0.),
            min((r["expected_actionable_coverage"] for r in f), default=0.),
            -max_kl if max_kl is not None else None,
            -sum(r["leaf_count"] for r in completed if r["job"]["arm"] == arm and r["status"] == "completed")]
        arms[arm] = {"all_role_fold_gates_passed": len(feasible) == 6 and all(gate(r["validation"]) for r in feasible),
            "structurally_infeasible_units": len(rows)-len(feasible), "minimum_explanation_coverage": score[0],
            "mean_explanation_coverage": float(np.mean([r["coverage"] for r in e])) if e else None,
            "minimum_expected_feedback_coverage": score[1], "maximum_qualified_kl": max_kl, "selection_score": score}
    assert_equal(arms, summary["arms"], "summary.arms")
    allowed = [arm for arm in ("B", "C", "A") if arms[arm]["all_role_fold_gates_passed"]]
    selected = max(allowed, key=lambda arm: tuple(arms[arm]["selection_score"])) if allowed else None
    if summary["selected_arm"] != selected or summary["plan_sha256"] != plan_sha:
        raise RuntimeError("audit complete-batch route selection differs")
    if (protocol.source_closure_files(root) != files or file_sha(actor_path) != protocol.ACTOR_SHA256
            or file_sha(path) != plan_sha):
        raise RuntimeError("audit inputs/source/Actor changed during recomputation")
    return {"schema": "warehouse-v621-independent-diagnostics-audit.v1", "candidate_id": protocol.CANDIDATE_ID,
        "passed": True, "jobs_audited": len(reports), "jobs": reports, "arms": arms, "selected_arm": selected,
        "plan_sha256": plan_sha, "summary_sha256": file_sha(path.parent/"summary.json"),
        "auditor_sha256": file_sha(__file__), "source_closure_sha256": identity["source_closure_sha256"],
        "input_file_count": len(manifest["files_sha256"]), "physical_frames_per_role": 32782,
        "float_tolerance": {"absolute": FLOAT_ATOL, "relative": FLOAT_RTOL, "counts_and_structure": "exact"},
        "production_metric_routing_qualification_oracle_calls": 0,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "publishable": False, "admission_evidence": False, "fresh_validation_evidence": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError("independent audit output already exists")
    result = audit(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plain(result), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "jobs_audited": result["jobs_audited"],
        "selected_arm": result["selected_arm"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
