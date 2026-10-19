"""Fit-only v6.21 topology comparison; no Actor or environment operations.

A is the unchanged v6.20 M2 algorithm with an explicit, versioned 60-round
configuration. C alternates full-distribution leaf fits with cost-sensitive
node coordinates. Its node costs are losses of the *complete existing child
subtrees*, rather than majority labels or a renamed M2 proposal heuristic.
Both are bounded heuristics, never a tree-capacity/optimality certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from . import warehouse_program_search as legacy
from .warehouse_v619_leaf_fit import (
    LEAF_EPSILON, LEAF_MAXITER, LEAF_REGULARIZATION, LEAF_TOLERANCE,
    fit_constrained_leaf_distribution, float32_preference_proof,
)
from .warehouse_v619_tree import (
    FIT_MIN_ROWS, FIT_MIN_EPISODES, FIT_MAX_EPISODE_FRACTION,
    FIT_MIN_AGREEMENT, FIT_MAX_KL, SEARCH_INITIALIZATIONS,
    _Decision, _Region, _Leaf, _State, _BudgetExhausted,
    _size, _tree_key, _replace, _export, source_feature_cost,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, HARD_SPLIT_TIE_EPSILON,
    certified_leaf_actions, explicit_mask_branches,
    leaf_indices, split_scores,
)

M2_ARM = "m2_global60_v621"
ALTERNATING_ARM = "guard_aware_whole_tree_alternating_v621"
ARMS = (M2_ARM, ALTERNATING_ARM)
FINAL_REFIT_RESERVED_CALLS = 16
END_TO_END_LEAF_API_CEILING = 20000
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class V621SearchConfig(legacy.SearchConfig):
    """Explicit new protocol; does not mutate or loosen legacy configuration."""
    max_rounds: int = 60
    max_leaf_optimization_calls: int = 19984

    def __post_init__(self):
        upper = {"initializations": 8, "max_rounds": 60,
            "max_leaf_optimization_calls": 19984, "proposals_per_round": 48,
            "beam_width": 4, "max_nonimproving_steps": 2, "axis_features": 12,
            "semantic_candidates_per_node": 6, "max_depth": 5, "max_leaves": 16}
        for name, limit in upper.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"invalid v6.21 topology search limit: {name}")


SearchConfig = V621SearchConfig
PRODUCTION_SEARCH_CONFIG = V621SearchConfig()


def _config(config):
    if config is None:
        return PRODUCTION_SEARCH_CONFIG
    if isinstance(config, V621SearchConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("v6.21 requires V621SearchConfig or a configuration mapping")
    return V621SearchConfig(**dict(config))


def _validate(encoded, masks, targets, episode_ids, names, actor_sha):
    x, m, p = [np.asarray(value, np.float32) for value in (encoded, masks, targets)]
    names = tuple(names)
    episodes = np.asarray([str(value) for value in episode_ids])
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
    if len(names) != len(set(names)):
        raise ValueError("duplicate program feature names")
    try:
        indices = tuple(names.index(f"legal_{a.lower()}") for a in actions)
    except ValueError as error:
        raise ValueError("explicit legal_ACTION features are required") from error
    if (x.ndim != 2 or x.shape[1] != len(names) or len(x) < 2
            or m.shape != (len(x), 5) or p.shape != m.shape
            or episodes.shape != (len(x),) or any(not value for value in episodes)
            or not isinstance(actor_sha, str) or len(actor_sha) != 64
            or any(value not in "0123456789abcdef" for value in actor_sha)
            or any(not np.isfinite(value).all() for value in (x, m, p))
            or np.any((m != 0) & (m != 1)) or np.any(m[:, 4] != 1)
            or not np.array_equal(x[:, indices], m) or np.any(p < 0)
            or np.any(p[m == 0] != 0)
            or not np.allclose(p.sum(-1), 1., atol=1.e-6, rtol=0)):
        raise ValueError("fit needs complete, finite, physically masked NN distributions")
    return x, m, p, episodes, names, indices


def complete_distribution_losses(targets, masks, logits):
    """Per-row KL of all supplied NN probabilities; no argmax reduction."""
    p, m, z = (np.asarray(a, np.float64) for a in (targets, masks, logits))
    if p.shape != m.shape or z.shape != p.shape or p.ndim != 2:
        raise ValueError("counterfactual subtree loss shapes differ")
    logq = np.where(m > .5, z, -np.inf)
    logq -= logsumexp(logq, axis=-1, keepdims=True)
    return np.sum(p*(np.log(np.maximum(p, 1.e-300))
                     - np.where(m > .5, logq, 0.)), axis=-1)


def _cost_sensitive_predicates(x, names, nonmask, left_cost, right_cost, cfg, quantiles):
    """Learn sparse weighted left/right separators from complete subtree costs.

    Direction >0 means right has smaller KL. Feature selection pays the
    declared public source cost, then a weighted logistic fit learns the
    selected coefficients. Threshold proposals are ranked by actual routed
    child-subtree KL, never by hard NN action accuracy.
    """
    if len(x) < 2 or not len(nonmask):
        return []
    advantage = np.asarray(left_cost)-np.asarray(right_cost)
    weight = np.abs(advantage)
    if float(weight.sum()) <= 1.e-12:
        return []
    values = np.asarray(x[:, nonmask], np.float64)
    mean, std = values.mean(0), np.maximum(values.std(0), 1.e-8)
    standardized = (values-mean)/std
    centered = advantage-advantage.mean()
    strength = np.abs(standardized.T@centered)
    order = np.argsort(-strength, kind="stable")
    support, source_cost = [], 0
    for index in order:
        price = 2 if names[nonmask[index]] == "public_hard_risk_context" else 1
        if source_cost+price <= 3 and strength[index] > 1.e-12:
            support.append(int(index))
            source_cost += price
        if source_cost == 3:
            break
    basis = []
    if support:
        features = standardized[:, support]
        y = (advantage > 0).astype(np.float64)
        normalized_weight = weight/weight.sum()
        def objective(theta):
            score = features@theta[:-1]+theta[-1]
            loss = np.sum(normalized_weight*(np.logaddexp(0., score)-y*score))
            gradient = normalized_weight*(expit(score)-y)
            return float(loss+1.e-4*np.sum(theta[:-1]**2)), np.r_[
                features.T@gradient+2.e-4*theta[:-1], gradient.sum()]
        result = minimize(objective, np.zeros(len(support)+1), method="L-BFGS-B",
            jac=True, options={"maxiter": 50, "ftol": 1.e-10})
        if np.isfinite(result.x).all():
            w = np.zeros(x.shape[1], np.float32)
            w[np.asarray(nonmask)[support]] = result.x[:-1]/std[support]
            bias = float(result.x[-1]-np.sum(result.x[:-1]*mean[support]/std[support]))
            if np.linalg.norm(w) > 1.e-10:
                scale = float(np.linalg.norm(w))
                basis.append((w/scale, bias/scale, "cost_sensitive_sparse_logistic"))
    for index in order[:cfg.axis_features]:
        if std[index] <= 1.e-8:
            continue
        w = np.zeros(x.shape[1], np.float32)
        w[nonmask[index]] = 1.
        basis.append((w, 0., f"cost_sensitive_axis:{names[nonmask[index]]}"))
    candidates, seen = [], set()
    for w, learned_bias, label in basis:
        values = split_scores(x, w[None], np.zeros(1, np.float32))[:, 0]
        thresholds = [-learned_bias, *[float(np.quantile(values, q)) for q in quantiles]]
        for threshold in thresholds:
            for sign in (1., -1.):
                ww, bias = np.asarray(sign*w, np.float32), float(np.float32(-sign*threshold))
                key = (ww.tobytes(), np.float32(bias).tobytes())
                if key in seen or source_feature_cost(ww, names) > 3:
                    continue
                seen.add(key)
                right = split_scores(x, ww[None], np.asarray([bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
                if min(int(right.sum()), int((~right).sum())) < 20:
                    continue
                cost = float(np.mean(np.where(right, right_cost, left_cost)))
                candidates.append((cost, label, ww, bias))
    candidates.sort(key=lambda item: (item[0], item[1], item[2].tobytes(), item[3]))
    return candidates[:cfg.semantic_candidates_per_node]


def _growth_prototypes(targets):
    """Fit-only soft prototypes, including modes with the same hard action.

    A leading full-five-probability principal direction creates two soft
    prototype groups. The hard-action means are an additional initialization,
    never the sole source of distributional growth proposals.
    """
    target = np.asarray(targets, np.float64)
    centered = target-target.mean(0)
    covariance = np.einsum("na,nb->ab", centered, centered, optimize=False)
    values, vectors = np.linalg.eigh(covariance)
    pairs = []
    if values[-1] > 1.e-12:
        direction = vectors[:, -1]
        first = np.flatnonzero(np.abs(direction) > 1.e-12)
        if len(first) and direction[first[0]] < 0:
            direction = -direction
        projection = np.einsum("na,a->n", centered, direction, optimize=False)
        order = np.argsort(projection, kind="stable")
        cut = len(order)//2
        if cut and cut < len(order):
            pairs.append((target[order[:cut]].mean(0),
                          target[order[cut:]].mean(0), "soft_distribution_principal_modes"))
    top = target.argmax(-1)
    actions = sorted(set(top.tolist()), key=lambda a: (-int(np.sum(top == a)), a))
    if len(actions) >= 2:
        pairs.append((target[top == actions[0]].mean(0),
                      target[top == actions[1]].mean(0), "hard_action_group_soft_means"))
    return pairs


def fit_structure(arm, encoded, masks, targets, episode_ids, *, feature_names,
                  actor_role_sha256, config=None):
    """Return legacy-compatible fixed topology plus fit-only protocol evidence."""
    arm = {"A": M2_ARM, "C": ALTERNATING_ARM}.get(arm, arm)
    if arm not in ARMS:
        raise ValueError("unknown v6.21 fit-only search arm")
    cfg = _config(config)
    # A receives the same validated complete-distribution contract as C.
    _validate(encoded, masks, targets, episode_ids, feature_names, actor_role_sha256)
    if arm == M2_ARM:
        structure, report = legacy.fit_legal_structure(encoded, masks, targets, episode_ids,
            feature_names=feature_names, actor_role_sha256=actor_role_sha256,
            method=legacy.ROBUST_METHOD, config=cfg)
        report["legacy_method"] = report["method"]
        report["method"] = M2_ARM
        report["legacy_source_unchanged"] = True
    else:
        structure, report = _fit_alternating(encoded, masks, targets, episode_ids,
            feature_names, actor_role_sha256, cfg)
    report.update({"v621_search_arm": arm,
        "search_protocol": "warehouse-v621-fit-only-global60-reserve16.v1",
        "search_budget": asdict(cfg),
        "leaf_api_cache_misses": report["leaf_optimization_calls"],
        "leaf_api_cache_hits": report["leaf_cache_hits"],
        "final_refit_reserved_calls": FINAL_REFIT_RESERVED_CALLS,
        "end_to_end_leaf_api_ceiling": END_TO_END_LEAF_API_CEILING,
        "end_to_end_accounting": "search_actual_api_including_failures_plus_final_fixed_topology_actual_leaf_calls",
        "maximum_end_to_end_calls": cfg.max_leaf_optimization_calls+FINAL_REFIT_RESERVED_CALLS,
        "production_configuration": cfg == PRODUCTION_SEARCH_CONFIG})
    return structure, report


def _fit_alternating(encoded, masks, targets, episode_ids, feature_names, actor_sha, cfg):
    x, m, p, ep, names, mask_indices = _validate(
        encoded, masks, targets, episode_ids, feature_names, actor_sha)
    nonmask = np.asarray([i for i in range(len(names)) if i not in mask_indices], int)
    top = p.argmax(-1)
    episode_order = sorted(set(ep), key=lambda value: sha256(value.encode()).digest())
    groups = {value: index % 3 for index, value in enumerate(episode_order)}
    group = np.asarray([groups[value] for value in ep], np.int8)
    cache, state_cache = {}, {}
    calls = solves = hits = failures = rounds = candidate_attempts = complete = 0
    call_limit = cfg.max_leaf_optimization_calls
    call_events, failure_examples, runs = [], [], []
    operation_counts = {"grow": 0, "prune": 0, "cost_sensitive_node_update": 0}

    def regions(tree, rows=None, depth=0, legal=None, path=()):
        rows = np.arange(len(x)) if rows is None else rows
        legal = frozenset({4}) if legal is None else legal
        yield _Region(path, tree, rows, depth, legal)
        if tree is not None:
            w = np.asarray(tree.weight, np.float32)
            right = split_scores(x[rows], w[None], np.asarray([tree.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            ll, rl = explicit_mask_branches(w, tree.bias, mask_indices)
            yield from regions(tree.left, rows[~right], depth+1, legal | ll, path+(0,))
            yield from regions(tree.right, rows[right], depth+1, legal | rl, path+(1,))

    def fit_leaf(region):
        nonlocal calls, solves, hits, failures
        rows, legal = region.rows, region.legal
        key = (sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest(), tuple(sorted(legal)))
        if key in cache:
            z, optimizer = cache[key]
            hits += 1
        else:
            if calls >= call_limit or calls >= cfg.max_leaf_optimization_calls:
                raise _BudgetExhausted()
            calls += 1
            event = {"call": calls, "rows_sha256": key[0], "rows": len(rows),
                     "path_legal_actions": list(key[1]), "cache_miss": True}
            call_events.append(event)
            try:
                z, optimizer = fit_constrained_leaf_distribution(p[rows], m[rows], key[1],
                    epsilon=LEAF_EPSILON, regularization=LEAF_REGULARIZATION,
                    maxiter=LEAF_MAXITER, tolerance=LEAF_TOLERANCE)
                enumerated = int(optimizer.get("enumerated_action_solves", len(legal)))
                solves += enumerated
                event["actual_convex_solver_calls"] = enumerated
                if (not optimizer.get("success") or not float32_preference_proof(
                        z, int(z.argmax()), legal)["passed"]):
                    raise RuntimeError("leaf optimizer returned an unproved float32 preference")
                event.update(success=True)
                cache[key] = (z, optimizer)
            except RuntimeError as error:
                # Solves from rejected post-fit proofs already entered above;
                # solver-raised failures expose their actual enumerations.
                n = int(getattr(error, "enumerated_action_solves", 0))
                solves += n
                failures += 1
                event.update(success=False,
                    actual_convex_solver_calls=event.get("actual_convex_solver_calls", 0)+n,
                    error=str(error)[:500])
                if len(failure_examples) < 20:
                    failure_examples.append(dict(event))
                raise
        if len(rows) and np.any(m[rows, int(z.argmax())] != 1):
            raise RuntimeError("path certificate conflicts with actual fit masks")
        kl = complete_distribution_losses(p[rows], m[rows], np.broadcast_to(z, (len(rows), 5)))
        matching = top[rows] == int(z.argmax())
        agreement = float(matching.mean()) if len(rows) else 0.
        _, counts = np.unique(ep[rows], return_counts=True)
        qualified = bool(len(rows) >= FIT_MIN_ROWS and len(counts) >= FIT_MIN_EPISODES
            and counts.max()/len(rows) <= FIT_MAX_EPISODE_FRACTION
            and agreement >= FIT_MIN_AGREEMENT and kl.mean() <= FIT_MAX_KL)
        parts = []
        for index in range(3):
            select = group[rows] == index
            parts.append({"group": index, "rows": int(select.sum()),
                "episodes": len(set(ep[rows[select]])),
                "agreement": float(matching[select].mean()) if select.any() else None,
                "masked_kl": float(kl[select].mean()) if select.any() else None})
        robust = bool(qualified and agreement >= .97 and kl.mean() <= .20 and all(
            part["episodes"] >= 2 and part["agreement"] >= .97
            and part["masked_kl"] <= .20 for part in parts))
        leaf = _Leaf(region, z, optimizer, qualified,
            float(np.sum(p[rows, int(z.argmax())]*matching)) if qualified else 0.,
            float(kl.sum()), agreement)
        leaf.robust = {"qualified": robust, "groups": parts}
        return leaf

    def evaluate(tree):
        key = _tree_key(tree)
        if key in state_cache:
            return state_cache[key]
        n, d = _size(tree)
        if n > cfg.max_leaves or d > cfg.max_depth:
            raise ValueError("alternating topology exceeds charged guard capacity")
        parts = list(regions(tree))
        leaves = [fit_leaf(part) for part in parts if part.tree is None]
        feedback = sum(leaf.expected_mass for leaf in leaves)/len(x)
        coverage = sum(len(leaf.region.rows) for leaf in leaves if leaf.robust["qualified"])/len(x)
        kl = sum(leaf.kl_sum for leaf in leaves)/len(x)
        state = _State(tree, leaves, parts, (coverage, feedback, -kl, -n), key)
        state_cache[key] = state
        if len(state_cache) > 128:
            del state_cache[next(iter(state_cache))]
        return state

    def guard(action, right=None):
        weight = np.zeros(len(names), np.float32)
        weight[mask_indices[action]] = 1.
        return _Decision(tuple(weight), -.5, None, right)

    def predict_subtree(tree, path, rows, leaves_by_path):
        if tree is None:
            return np.broadcast_to(leaves_by_path[path].logits, (len(rows), 5))
        right = split_scores(x[rows], np.asarray(tree.weight, np.float32)[None],
            np.asarray([tree.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
        out = np.empty((len(rows), 5), np.float32)
        out[~right] = predict_subtree(tree.left, path+(0,), rows[~right], leaves_by_path)
        out[right] = predict_subtree(tree.right, path+(1,), rows[right], leaves_by_path)
        return out

    def proposals(state, initialization):
        by_path = {leaf.region.path: leaf for leaf in state.leaves}
        offered = {}
        def add(tree, operation, label, cost=None):
            size, depth = _size(tree)
            if size > cfg.max_leaves or depth > cfg.max_depth:
                return
            key = _tree_key(tree)
            if key != state.key:
                offered.setdefault(key, (tree, operation, label, cost))
        # Complete existing child-subtree losses define each node's
        # counterfactual left/right labels and cost weights.
        for region in state.regions:
            if region.tree is None:
                continue
            w = np.asarray(region.tree.weight, np.float32)
            ll, rl = explicit_mask_branches(w, region.tree.bias, mask_indices)
            # A legal guard predicate never changes in a node update.
            if not ll and not rl and len(region.rows) >= 40:
                left_z = predict_subtree(region.tree.left, region.path+(0,), region.rows, by_path)
                right_z = predict_subtree(region.tree.right, region.path+(1,), region.rows, by_path)
                left_cost = complete_distribution_losses(p[region.rows], m[region.rows], left_z)
                right_cost = complete_distribution_losses(p[region.rows], m[region.rows], right_z)
                candidates = _cost_sensitive_predicates(x[region.rows], names, nonmask,
                    left_cost, right_cost, cfg, initialization["quantiles"])
                for cost, label, weight, bias in candidates:
                    node = _Decision(tuple(weight), bias, region.tree.left, region.tree.right)
                    add(_replace(state.tree, region.path, node), "cost_sensitive_node_update",
                        f"{region.path}:{label}", cost)
            # Pruning removes the *whole* subtree. evaluate() recomputes its
            # ancestor certificate and refits the new leaf, including guards.
            add(_replace(state.tree, region.path, None), "prune", str(region.path))
        # Structural growth complements the fixed-topology alternating pass.
        for region in state.regions:
            if region.tree is not None or region.depth >= cfg.max_depth or not len(region.rows):
                continue
            rows = region.rows
            actions = sorted((a for a in range(4) if a not in region.legal and np.any(top[rows] == a)),
                key=lambda a: (-int(np.sum(top[rows] == a)), a))
            for action in actions:
                add(_replace(state.tree, region.path, guard(action)), "grow", f"{region.path}:legal:{action}")
            if len(rows) < 40:
                continue
            # Complete soft target modes can differ even when every frame
            # has the same NN argmax. Growth must preserve that information.
            for left_p, right_p, prototype_label in _growth_prototypes(p[rows]):
                a, b = int(left_p.argmax()), int(right_p.argmax())
                pseudo = [complete_distribution_losses(p[rows], m[rows],
                    np.broadcast_to(np.log(np.maximum(prototype, 1.e-5)), (len(rows), 5)))
                    for prototype in (left_p, right_p)]
                for cost, label, weight, bias in _cost_sensitive_predicates(x[rows], names, nonmask,
                        pseudo[0], pseudo[1], cfg, initialization["quantiles"]):
                    left = guard(a) if a not in region.legal else None
                    right = guard(b) if b not in region.legal else None
                    node = _Decision(tuple(weight), bias, left, right)
                    add(_replace(state.tree, region.path, node), "grow",
                        f"{region.path}:semantic:{prototype_label}:{label}", cost)
        return sorted(offered.values(), key=lambda item: (
            {"cost_sensitive_node_update": 0, "grow": 1, "prune": 2}[item[1]],
            np.inf if item[3] is None else item[3], item[2], _tree_key(item[0])))

    movement = sorted(range(4), key=lambda a: (-int(np.sum(top == a)), a))
    pool = [None, *[guard(a) for a in movement],
            *[guard(a, guard(b)) for a in movement for b in movement if a != b]]
    starts, seen = [], set()
    for tree in pool:
        n, d = _size(tree)
        if n <= cfg.max_leaves and d <= cfg.max_depth and _tree_key(tree) not in seen:
            starts.append(tree)
            seen.add(_tree_key(tree))
    starts = [starts[i % len(starts)] for i in range(cfg.initializations)]
    best = None
    for index, initialization in enumerate(SEARCH_INITIALIZATIONS[:cfg.initializations]):
        before_calls, before_solves, before_rounds = calls, solves, rounds
        call_limit = max(1, (index+1)*cfg.max_leaf_optimization_calls//cfg.initializations)
        round_limit = (index+1)*cfg.max_rounds//cfg.initializations
        state = evaluate(None)
        initial_failure = None
        try:
            state = evaluate(starts[index])
        except (_BudgetExhausted, RuntimeError) as error:
            initial_failure = type(error).__name__+": "+str(error)[:500]
        start_best, history = state, []
        stop = "global_round_share_limit"
        for _ in range(round_limit-rounds):
            candidates = proposals(state, initialization)
            if not candidates:
                stop = "no_feasible_coordinates"
                break
            previous = state
            accepted, failed, evaluated, interrupted = [], 0, 0, False
            for tree, operation, label, counterfactual_cost in candidates:
                candidate_attempts += 1
                try:
                    candidate = evaluate(tree)
                except _BudgetExhausted:
                    interrupted = True
                    break
                except RuntimeError:
                    failed += 1
                    continue
                evaluated += 1
                complete += 1
                operation_counts[operation] += 1
                if legacy._improves(candidate.score, state.score) or (
                        candidate.score == state.score and candidate.key < state.key):
                    state = candidate
                    accepted.append({"operation": operation, "label": label,
                        "counterfactual_complete_subtree_kl": counterfactual_cost,
                        "score": list(state.score), "tree_sha256": state.key.hex()})
                if legacy._improves(candidate.score, start_best.score) or (
                        candidate.score == start_best.score and candidate.key < start_best.key):
                    start_best = candidate
            rounds += 1
            history.append({"round": len(history)+1, "global_round": rounds,
                "proposals": len(candidates), "complete_evaluations": evaluated,
                "failed_evaluations": failed, "accepted_updates": accepted,
                "previous_score": list(previous.score), "score": list(state.score),
                "leaf_optimization_calls_cumulative": calls, "budget_interrupted": interrupted})
            _LOG.info("v621 alternating round_complete start=%d global_round=%d/%d api_calls=%d convex_calls=%d score=%s complete=%d failed=%d",
                index+1, rounds, cfg.max_rounds, calls, solves, state.score, evaluated, failed)
            if interrupted:
                stop = "leaf_optimization_call_budget"
                break
            if state.key == previous.key:
                stop = "complete_coordinate_pass_no_improvement"
                break
        if best is None or legacy._improves(start_best.score, best.score) or (
                start_best.score == best.score and start_best.key < best.key):
            best = start_best
        runs.append({"initialization": dict(initialization), "rounds": history,
            "stop_reason": stop, "requested_initial_tree_sha256": _tree_key(starts[index]).hex(),
            "initial_tree_failure": initial_failure, "fit_score": list(start_best.score),
            "leaf_optimization_calls": calls-before_calls,
            "actual_convex_solver_calls": solves-before_solves,
            "reserved_cumulative_leaf_call_limit": call_limit,
            "reserved_cumulative_round_limit": round_limit,
            "global_rounds_before_start": before_rounds, "global_rounds_after_start": rounds})
    assert best is not None
    structure = _export(best.tree, best.leaves, len(names))
    weights = structure["split_weights"]
    proof = certified_leaf_actions(weights, structure["split_bias"], structure["left_refs"],
        structure["right_refs"], len(weights), structure["leaf_count"], mask_indices,
        explicit=True, wait_action=4)
    routes = leaf_indices(x, weights, structure["split_bias"], structure["left_refs"],
                          structure["right_refs"], len(weights), cfg.max_depth)
    for i, leaf in enumerate(best.leaves):
        if (not np.array_equal(np.flatnonzero(routes == i), leaf.region.rows)
                or proof[i] != leaf.region.legal
                or not float32_preference_proof(leaf.logits, int(leaf.logits.argmax()), proof[i])["passed"]):
            raise RuntimeError("alternating export changed hard routing/path proof")
    illegal = int(np.sum(m[np.arange(len(x)), structure["initial_leaf_logits"][routes].argmax(-1)] != 1))
    if illegal or any(source_feature_cost(w, names) > 3 for w in weights):
        raise RuntimeError("alternating export violates legal/source feature bounds")
    if not (calls <= cfg.max_leaf_optimization_calls and 0 <= solves <= 5*calls
            and rounds <= cfg.max_rounds
            and rounds == sum(len(run["rounds"]) for run in runs)
            and len(call_events) == calls):
        raise RuntimeError("alternating search accounting failed")
    report = {"method": ALTERNATING_ARM,
        "topology_origin": "fit_only_guard_aware_whole_tree_cost_sensitive_alternation",
        "target_provenance": "complete_frozen_nn_masked_distribution",
        "actor_role_sha256": actor_sha, "fit_rows": len(x), "fit_episode_count": len(set(ep)),
        "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
        "search_runs": runs, "leaf_optimization_calls": calls,
        "leaf_optimization_calls_remaining": cfg.max_leaf_optimization_calls-calls,
        "actual_convex_solver_calls": solves, "leaf_cache_hits": hits,
        "leaf_api_attempts": call_events, "rejected_solver_candidates": failures,
        "failed_leaf_api_attempts": failures,
        "solver_failure_examples": failure_examples, "candidate_evaluation_attempts": candidate_attempts,
        "complete_candidate_evaluations": complete, "operation_complete_evaluations": operation_counts,
        "global_search_rounds": rounds, "global_search_rounds_remaining": cfg.max_rounds-rounds,
        "round_budget_scope": "global_per_role_fit_across_all_initializations",
        "distinct_requested_initializations": len({_tree_key(t) for t in starts}),
        "fit_stable_explanation_coverage": best.score[0],
        "fit_expected_feedback_coverage": best.score[1], "fit_masked_kl": -best.score[2],
        "fit_qualified_leaf_coverage": sum(len(l.region.rows) for l in best.leaves if l.qualified)/len(x),
        "fit_whole_tree_raw_illegal_count": illegal, "fit_whole_tree_path_uncertified_count": 0,
        "fit_leaf_reports": [{"leaf": i, "rows": len(l.region.rows), "depth": l.region.depth,
            "path_legal_actions": sorted(l.region.legal), "raw_action": int(l.logits.argmax()),
            "fit_qualified": l.qualified, "fit_stable_explanation_qualified": l.robust["qualified"],
            "episode_stability_groups": l.robust["groups"], "agreement": l.agreement,
            "masked_kl": l.kl_sum/len(l.region.rows) if len(l.region.rows) else None,
            "expected_feedback_mass": l.expected_mass, "optimizer": l.optimizer}
            for i, l in enumerate(best.leaves)],
        "node_coordinate_objective": "complete_existing_left_right_subtree_all_five_action_masked_KL",
        "node_updates_use_counterfactual_subtree_costs": True,
        "node_direction_optimizer": "fit_only_sparse_weighted_logistic_LBFGS_50_iterations",
        "guard_predicate_updates": "immutable",
        "subtree_pruning": "remove_entire_subtree_then_reprove_ancestor_legal_set_and_refit",
        "leaf_optimizer": {"epsilon": LEAF_EPSILON, "regularization": LEAF_REGULARIZATION,
            "maxiter": LEAF_MAXITER, "tolerance": LEAF_TOLERANCE},
        "fit_proxy_qualification": {"minimum_rows": FIT_MIN_ROWS,
            "minimum_episodes": FIT_MIN_EPISODES,
            "maximum_episode_fraction": FIT_MAX_EPISODE_FRACTION,
            "minimum_agreement": FIT_MIN_AGREEMENT, "maximum_kl": FIT_MAX_KL},
        "fit_explanation_margin": {"minimum_agreement": .97, "maximum_kl": .20,
            "episode_groups": 3, "minimum_episodes_per_group": 2,
            "group_assignment": "sha256_sorted_complete_fit_episodes_round_robin"},
        "candidate_score": "fit_episode_stable_explanation_coverage_then_expected_feedback_then_kl_then_leaf_count.v1",
        "historical_best_preserved": True, "hard_routing_version": HARD_SPLIT_ROUTING_VERSION,
        "guard_capacity_included_in_search": True, "whole_tree_raw_action_path_proof": True,
        "maximum_source_features_per_node": max((source_feature_cost(w, names) for w in weights), default=0),
        "topology_uses_fit_only": True, "calibration_eligibility_validation_used": False,
        "fit_only_not_independent_certificate": True,
        "search_completeness": "bounded_heuristic_not_a_capacity_or_global_optimality_proof",
        "execution_action_override": False,
        "split_sha256": sha256(json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in structure.items() if key != "initial_leaf_logits"},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return structure, report
