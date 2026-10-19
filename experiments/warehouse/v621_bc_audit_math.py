"""Independent, read-only arithmetic for completed v6.21 BC evidence.

Only arrays and a program JSON mapping enter this module.  It imports no
environment, Actor, program implementation, solver or execution capability.
The caller must establish completion, source/asset identity and frame integrity.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math

import numpy as np


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
TIE_EPSILON = 1e-10


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(label + " must be an integer")
    return int(value)


def _float32(value, label):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, np.float32)
    if not np.isfinite(result).all():
        raise ValueError(label + " contains a nonfinite or overflowing float32 value")
    return result


def _schema(feature_names, feature_bounds):
    names = tuple(feature_names)
    if not names or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("feature names must be unique strings")
    try:
        mask_indices = tuple(names.index("legal_" + a.lower()) for a in ACTIONS)
    except ValueError as error:
        raise ValueError("all five physical legal-action features are required") from error
    if not isinstance(feature_bounds, Mapping):
        raise ValueError("certification bounds must be an explicit mapping")
    bounds = {}
    for key, values in feature_bounds.items():
        index = names.index(key) if isinstance(key, str) and key in names else _integer(key, "bound index")
        if not 0 <= index < len(names) or index in bounds:
            raise ValueError("duplicate or out-of-range bound index")
        interval = np.asarray(values, np.float64)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError("feature interval must be finite and ordered")
        bounds[index] = tuple(map(float, interval))
    return names, mask_indices, bounds


def _topology(raw, names):
    tree = raw.get("structure", raw)
    if not isinstance(tree, Mapping):
        raise ValueError("program structure must be a mapping")
    weights = _float32(tree["split_weights"], "split weights")
    if weights.shape == (0,):
        weights = np.empty((0, len(names)), np.float32)
    bias = _float32(tree["split_bias"], "split bias")
    nodes = len(weights)
    logits = _float32(raw["leaf_logits"], "leaf logits")
    count = _integer(tree.get("leaf_count", len(logits)), "leaf count")
    root = _integer(tree.get("root_ref", 0 if nodes else -1), "root reference")
    left_raw, right_raw = np.asarray(tree["left_refs"]), np.asarray(tree["right_refs"])
    if (weights.shape != (nodes, len(names)) or bias.shape != (nodes,)
            or left_raw.shape != (nodes,) or right_raw.shape != (nodes,)
            or logits.shape != (count, 5) or not 1 <= count <= 16 or nodes != count - 1
            or root != (0 if nodes else -1)):
        raise ValueError("invalid rooted binary program dimensions or leaf capacity")
    left = np.asarray([_integer(x, "left reference") for x in left_raw], np.int64)
    right = np.asarray([_integer(x, "right reference") for x in right_raw], np.int64)
    costs = [sum(2 if names[i] == "public_hard_risk_context" else 1
                 for i in np.flatnonzero(row)) for row in weights]
    if max(costs, default=0) > 3:
        raise ValueError("program predicate exceeds three source features")
    seen_nodes, paths = set(), {}

    def visit(ref, path):
        if len(path) > 5:
            raise ValueError("program exceeds depth five")
        if ref < 0:
            leaf = -1-ref
            if not 0 <= leaf < count or leaf in paths:
                raise ValueError("invalid or multiply referenced program leaf")
            paths[leaf] = tuple(path)
            return
        if ref >= nodes or ref in seen_nodes:
            raise ValueError("invalid, cyclic or multiply referenced program node")
        seen_nodes.add(ref)
        visit(int(left[ref]), (*path, (ref, False)))
        visit(int(right[ref]), (*path, (ref, True)))

    visit(root, ())
    if seen_nodes != set(range(nodes)) or set(paths) != set(range(count)):
        raise ValueError("program contains unreachable structure")
    selected = {}
    for purpose in ("feedback", "explanation"):
        values = [_integer(v, purpose + " leaf") for v in raw[purpose + "_leaves"]]
        if len(set(values)) != len(values) or not set(values) <= set(range(count)):
            raise ValueError("invalid selected program leaves")
        selected[purpose] = frozenset(values)
    return weights, bias, left, right, logits, paths, selected, costs


def _branch_proofs(weights, bias, mask_indices, bounds):
    """Independent interval proof with mask[a]=0; no sampled-row inference."""
    used = set(np.flatnonzero(weights).tolist())
    left, right = set(), set()
    for action, index in enumerate(mask_indices):
        others = sorted(used - {index})
        if index not in used or any(i not in bounds for i in others):
            continue
        low = high = float(bias)
        for i in others:
            a, b = float(weights[i])*bounds[i][0], float(weights[i])*bounds[i][1]
            low += min(a, b)
            high += max(a, b)
        if high <= TIE_EPSILON:
            right.add(action)
        if low > TIE_EPSILON:
            left.add(action)
    return left, right


def _certificates(weights, bias, mask_indices, bounds, paths, *, shared):
    explicit, interval = [], []
    for row, threshold in zip(weights, bias, strict=True):
        lo, hi = _branch_proofs(row, threshold, mask_indices, bounds)
        interval.append((lo, hi))
        used = np.flatnonzero(row)
        if len(used) != 1 or int(used[0]) not in mask_indices:
            explicit.append((set(), set()))
            continue
        if shared:
            # Shared programs require the zero/one physical bit to straddle
            # the actual hard-routing boundary, not an interval-only proof.
            action = mask_indices.index(int(used[0]))
            z, o = float(threshold), float(threshold)+float(row[used[0]])
            explicit.append(({action} if o <= TIE_EPSILON < z else set(),
                             {action} if z <= TIE_EPSILON < o else set()))
        else:
            # The v6.20 ROBUST control permits interval proofs only at pure
            # physical mask predicates (its historical path contract).
            explicit.append((lo, hi))
    def collect(branches):
        return tuple(frozenset({4}).union(*(branches[node][int(right)] for node, right in paths[leaf]))
                     for leaf in range(len(paths)))
    return collect(explicit), collect(interval)


def _wilson(agreement, count):
    if not count:
        return 0.
    square = 1.645**2
    centre = agreement + square/(2*count)
    spread = 1.645*math.sqrt((agreement*(1-agreement)+square/(4*count))/count)
    return max(0., (centre-spread)/(1+square/count))


def _probabilities(logits, masks):
    z = np.asarray(logits, np.float64)
    legal = masks.astype(bool)
    maximum = np.where(legal, z, -np.inf).max(-1, keepdims=True)
    centered = np.full_like(z, -np.inf)
    np.subtract(z, maximum, out=centered, where=legal)
    with np.errstate(under="ignore"):
        exp = np.exp(centered)
    return exp/exp.sum(-1, keepdims=True)


def audit_program(raw_program, encoded, masks, nn_p, sampled_indices, episodes,
                  scenarios, feature_names, feature_bounds, shared: bool):
    """Recompute metrics without production routing, certification or metrics.

    ``raw_program`` accepts native shared JSON or normalized legacy JSON with
    the topology at either ``structure`` or the top level.  Context arrays
    retain every input row.  The caller supplies independently reconstructed
    NN probabilities; this function does not attest how they were generated.
    Shared KL uses 1e-300 and control KL uses its historical 1e-8 floor.
    """
    if type(shared) is not bool or not isinstance(raw_program, Mapping):
        raise ValueError("explicit shared flag and program mapping are required")
    names, mask_indices, bounds = _schema(feature_names, feature_bounds)
    x = _float32(encoded, "encoded features")
    m, p = np.asarray(masks, np.float64), np.asarray(nn_p, np.float64)
    actions = np.asarray(sampled_indices)
    n = len(x)
    if (not n or x.shape != (n, len(names)) or m.shape != (n, 5) or p.shape != (n, 5)
            or actions.shape != (n,) or not np.issubdtype(actions.dtype, np.integer)
            or actions.dtype == np.bool_ or np.any((actions < 0) | (actions >= 5))
            or not np.isfinite(m).all() or not np.isfinite(p).all()
            or np.any((m != 0) & (m != 1)) or np.any(m[:, 4] != 1)
            or not np.array_equal(x[:, mask_indices], m) or np.any(p < 0)
            or np.any(p[m == 0] != 0) or not np.allclose(p.sum(-1), 1., atol=1e-6, rtol=0)
            or np.any(m[np.arange(n), actions] != 1)):
        raise ValueError("invalid complete physical frames, masks, distributions or sampled actions")
    ids = []
    for value, label in ((episodes, "episode"), (scenarios, "scenario")):
        values = np.asarray(value)
        if values.shape != (n,):
            raise ValueError(label + " identities must cover every physical frame")
        ids.append(np.asarray([str(v) for v in values]))
    for i, (low, high) in bounds.items():
        if np.any((x[:, i] < low) | (x[:, i] > high)):
            raise ValueError("encoded physical feature is outside its certification domain")
    w, b, left, right, leaf_logits, paths, sets, costs = _topology(raw_program, names)
    certs, interval_certs = _certificates(w, b, mask_indices, bounds, paths, shared=shared)
    # Accumulate sparse products in the original feature-index order and in
    # float64 from stored float32 inputs, independently of batch shape/BLAS.
    x64 = x.astype(np.float64)
    scores = np.zeros((n, len(w)), np.float64)
    for node in range(len(w)):
        for index in np.flatnonzero(w[node]):
            scores[:, node] += x64[:, index]*float(w[node, index])
        scores[:, node] += float(b[node])
    refs = np.full(n, 0 if len(w) else -1, np.int64)
    for _ in range(5):
        active = np.flatnonzero(refs >= 0)
        if not len(active):
            break
        nodes = refs[active]
        refs[active] = np.where(scores[active, nodes] > TIE_EPSILON, right[nodes], left[nodes])
    if np.any(refs >= 0):
        raise ValueError("program route exceeded the validated depth")
    leaves = -1-refs
    q = _probabilities(leaf_logits[leaves], m)
    raw_top, top, nn = leaf_logits[leaves].argmax(-1), q.argmax(-1), p.argmax(-1)
    raw_legal = m[np.arange(n), raw_top] > .5
    path_proven = np.asarray([int(a) in certs[int(l)] for a, l in zip(raw_top, leaves, strict=True)])
    valid = raw_legal & path_proven
    epsilon = 1e-300 if shared else 1e-8
    with np.errstate(under="ignore"):
        kl = np.sum(p*(np.log(np.maximum(p, epsilon))-np.log(np.maximum(q, epsilon))), axis=-1)
    if shared:
        core = np.asarray(raw_program["core_mask"])
        residual_bound = raw_program["residual_bound"]
        if (core.shape != (len(paths),) or np.any((core != 0) & (core != 1))
                or type(residual_bound) not in (int, float)
                or residual_bound not in (.35, float(np.float32(.35)), .50)):
            raise ValueError("invalid shared static core or registered residual bound")
        core_rows = core[leaves] > .5
        bound = (float(residual_bound)+.05)**2/2
    else:
        core_rows, bound = np.zeros(n, bool), None
    results = {}
    for purpose in ("feedback", "explanation"):
        chosen = np.isin(leaves, tuple(sets[purpose]))
        count = int(chosen.sum())
        matched = valid & (top == actions)
        expected = valid*p[np.arange(n), top]
        if purpose == "feedback":
            matched &= actions == nn
            expected *= top == nn
        agreement = float((top[chosen] == nn[chosen]).mean()) if count else 0.
        frequencies = Counter(ids[0][chosen])
        metric = {"sample_count": n, "physical_frame_count": n, "excluded_frames": 0,
            "rows": count, "coverage": count/n, "actionable_rows": int((matched & chosen).sum()),
            "actionable_coverage": float((matched & chosen).sum()/n),
            "expected_actionable_rows": float(expected[chosen].sum()),
            "expected_actionable_coverage": float(expected[chosen].sum()/n),
            "agreement": agreement, "agreement_wilson_lower_90": _wilson(agreement, count),
            "kl": float(kl[chosen].mean()) if count else None,
            "illegal_count": int((~raw_legal & chosen).sum()),
            "masked_illegal_count": int((m[np.arange(n), top][chosen] == 0).sum()),
            "path_uncertified_count": int((~path_proven & chosen).sum()),
            "whole_tree_raw_illegal_count": int((~raw_legal).sum()),
            "whole_tree_path_uncertified_count": int((~path_proven).sum()),
            "certified_rows": int((valid & chosen).sum()) if shared else 0,
            "distinct_episode_count": len(frequencies),
            "distinct_scenario_seed_count": len(set(ids[1][chosen])),
            "max_episode_fraction": max(frequencies.values())/count if count else 0.,
            "whole_tree_agreement": float((top == nn).mean()), "whole_tree_kl": float(kl.mean()),
            "per_nn_action": {a: {"physical_frames": int((nn == i).sum()),
                "rows": int(((nn == i) & chosen).sum()),
                "qualified_coverage": float(((nn == i) & chosen).sum()/(nn == i).sum()) if np.any(nn == i) else None,
                "actual_actionable_rows": int(((nn == i) & chosen & matched).sum())}
                for i, a in enumerate(ACTIONS)}}
        if shared:
            metric.update({"core_KL_upper_bound": bound,
                "core_actual_max_kl": float(kl[core_rows].max()) if core_rows.any() else None,
                "core_KL_bound_violations": int((core_rows & (kl > bound+5e-6)).sum()),
                "core_rows": int(core_rows.sum())})
        else:
            for i, action in enumerate(ACTIONS):
                take = (nn == i) & chosen
                metric["per_nn_action"][action].update({
                    "agreement": float((top[take] == nn[take]).mean()) if take.any() else None,
                    "expected_actionable_rows": float(expected[take].sum())})
            public_names = ("same_target_collision_streak_capped", "robots_shortest_distance",
                            "self_charger_return_margin")
            if all(name in names for name in public_names):
                conflict = x[:, names.index(public_names[0])] > 0
                near = x[:, names.index(public_names[1])]*np.float32(42) <= 2
                low = x[:, names.index(public_names[2])]*np.float32(50) <= 4
                metric["public_strata"] = {}
                for label, take in {"ordinary": ~(conflict | near | low), "conflict": conflict,
                                    "near_robots": near, "low_return_margin": low}.items():
                    selected = take & chosen
                    denominator, support = int(take.sum()), int(selected.sum())
                    metric["public_strata"][label] = {"rows": support, "physical_frames": denominator,
                        "qualified_coverage": support/denominator if denominator else None,
                        "agreement": float((top[selected] == nn[selected]).mean()) if support else None,
                        "kl": float(kl[selected].mean()) if support else None,
                        "illegal_count": int((~raw_legal & selected).sum()),
                        "actual_actionable_rows": int((matched & selected).sum()),
                        "expected_actionable_rows": float(expected[selected].sum())}
        results[purpose] = metric
    results["context"] = {"leaf_indices": leaves, "program_probabilities": q,
        "raw_top": raw_top, "program_top": top, "nn_top": nn,
        "raw_legal": raw_legal, "path_certified": path_proven,
        "core_rows": core_rows, "row_kl": kl,
        "path_certificates": tuple(tuple(sorted(c)) for c in certs),
        "interval_path_certificates": tuple(tuple(sorted(c)) for c in interval_certs),
        "leaf_paths": paths, "source_features_per_node": costs,
        "control_probability_arithmetic": "independent_float64_softmax_legacy_KL_floor_1e-8" if not shared else None}
    return results
