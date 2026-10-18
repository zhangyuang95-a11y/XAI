"""Fit-only search for an independent, path-certified distribution tree.

No environment, Actor mutation, calibration or eligibility data enter this
module. All search variants are deterministic and all final leaf distributions
are fitted against all five probabilities of the complete masked NN.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .warehouse_v6_tree_routing import (
    HARD_SPLIT_TIE_EPSILON, split_scores, leaf_indices as route_leaf_indices,
)

MAX_DEPTH = 5
MAX_LEAVES = 16
MAX_SOURCE_FEATURES = 3
SEARCH_VARIANTS = (
    {"initialization": 0, "quantiles": (.25, .50, .75)},
    {"initialization": 1, "quantiles": (.15, .35, .65, .85)},
    {"initialization": 2, "quantiles": (.10, .30, .50, .70, .90)},
)
SEARCH_BUDGET = {"beam_width": 2, "exact_candidates_per_round": 20,
                 "axis_features": 18, "candidates_per_leaf": 10}
_EPS = 1.e-9
_SEARCH_GAIN_EPSILON = 1.e-7


def source_feature_cost(weights: np.ndarray, feature_names: Sequence[str]) -> int:
    """The derived risk flag spends its two registered source measurements."""
    used = np.flatnonzero(weights)
    return sum(2 if feature_names[index] == "public_hard_risk_context" else 1
               for index in used)


def masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    z = np.where(masks > .5, logits, -1.e9)
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def fit_leaf_distribution(target: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Convex five-logit fit; physical masks participate in the objective."""
    if not len(target):
        result = np.full(target.shape[-1], -8., dtype=np.float32)
        result[-1] = 0.  # Empty leaves can never become qualified evidence.
        return result
    prior = np.log((target.sum(0) + .01) / (masks.sum(0) + .05))
    prior -= prior.mean()

    def loss(z):
        q = masked_probabilities(z[None], masks)
        logq = np.log(np.maximum(q, _EPS))
        delta = z - prior
        objective = -(target * logq).sum(-1).mean() + .0001 * (delta**2).mean()
        gradient = (q-target).mean(0) + .0002 * delta / len(z)
        return float(objective), gradient.astype(np.float64)

    result = minimize(loss, prior.astype(np.float64), method="L-BFGS-B", jac=True,
                      options={"maxiter": 60, "ftol": 1.e-10})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("independent masked distribution fit failed")
    return np.asarray(result.x - result.x.mean(), dtype=np.float32)


@dataclass
class _Leaf:
    rows: np.ndarray
    depth: int
    legal: frozenset[int]
    logits: np.ndarray
    qualified_mass: float
    potential: float
    kl_sum: float
    raw_action: int
    nonlegality_qualified: bool


@dataclass
class _State:
    weights: list[np.ndarray]
    bias: list[float]
    left: list[int]
    right: list[int]
    leaves: list[_Leaf]

    def score(self, *, final=False):
        mass = sum(leaf.qualified_mass for leaf in self.leaves)
        potential = sum(leaf.potential for leaf in self.leaves)
        kl = sum(leaf.kl_sum for leaf in self.leaves)
        return (mass if final else mass + .20*potential, -kl, -len(self.leaves))


def _quick_partition_score(rows, legal, target):
    if not len(rows):
        return 0.
    raw = int(np.argmax(target[rows].mean(0)))
    purity = float(np.mean(target[rows].argmax(-1) == raw))
    return len(rows)*purity**3*(1. if raw in legal else .65)


def _quick_split_gain(parent_rows, left_rows, right_rows,
                      parent_legal, right_legal, target):
    """Comparable split benefit; pure parent mass is not an improvement."""
    return (_quick_partition_score(left_rows, parent_legal, target)
            + _quick_partition_score(right_rows, right_legal, target)
            - _quick_partition_score(parent_rows, parent_legal, target))


def _state_key(state):
    digest = sha256()
    for weight in state.weights:
        digest.update(np.ascontiguousarray(weight, dtype="<f4").tobytes())
    for values, dtype in ((state.bias, "<f4"), (state.left, "<i8"),
                          (state.right, "<i8")):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.digest()


def fit_independent_structure(
        encoded: np.ndarray, masks: np.ndarray, target_probs: np.ndarray,
        episode_ids: Sequence[str], *, feature_names: Sequence[str],
        actor_role_sha256: str, actions: Sequence[str] =
        ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")) -> tuple[dict, dict]:
    """Search three deterministic fit-only initializations under the full budget.

    The returned topology is independent of the Actor tree. Its leaf logits
    are a fit initialization, not an independent qualification certificate.
    Dedicated legal-bit gates are searched with their real depth/leaf cost,
    including a zero-support residual child for an all-legal fit partition.
    """
    x = np.asarray(encoded, np.float32)
    m = np.asarray(masks, np.float32)
    p = np.asarray(target_probs, np.float32)
    names = tuple(feature_names)
    actions = tuple(actions)
    episode = np.asarray([str(value) for value in episode_ids])
    mask_indices = tuple(names.index(f"legal_{action.lower()}") for action in actions)
    if (x.ndim != 2 or x.shape[1] != len(names) or len(x) < 2
            or m.shape != (len(x), len(actions)) or p.shape != m.shape
            or episode.shape != (len(x),) or any(not value for value in episode)
            or len(actor_role_sha256) != 64
            or any(value not in "0123456789abcdef" for value in actor_role_sha256)
            or any(not np.isfinite(a).all() for a in (x, m, p))
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)
            or np.any(m[:, actions.index("WAIT")] != 1)
            or not np.array_equal(x[:, mask_indices], m)
            or np.any(p < 0) or np.any(p*(1-m) > 1.e-6)
            or not np.allclose(p.sum(-1), 1., atol=1.e-5)):
        raise ValueError("independent tree requires complete masked NN fit distributions")
    wait = actions.index("WAIT")
    top = p.argmax(-1)
    cache: dict[bytes, np.ndarray] = {}

    def leaf(rows, depth, legal):
        key = rows.astype("<i8", copy=False).tobytes()
        if key not in cache:
            cache[key] = fit_leaf_distribution(p[rows], m[rows])
        z = cache[key]
        if not len(rows):
            return _Leaf(rows, depth, legal, z, 0., 0., 0., int(z.argmax()), False)
        q = masked_probabilities(z[None], m[rows])
        qtop = q.argmax(-1)
        raw = int(z.argmax())
        matching = qtop == top[rows]
        kl = np.sum(p[rows]*(np.log(np.maximum(p[rows], _EPS))
                             - np.log(np.maximum(q, _EPS))), axis=-1)
        _, counts = np.unique(episode[rows], return_counts=True)
        supported = (len(rows) >= 80 and len(counts) >= 8
                     and counts.max()/len(rows) <= .20)
        nonlegality_qualified = bool(supported and matching.mean() >= .95
                                    and kl.mean() <= .25)
        qualified = (nonlegality_qualified and raw in legal and m[rows, raw].all())
        expected = p[rows, raw] * matching
        # Accumulate float32 probabilities in float64 so the same covered
        # rows have identical mass before/after a purely distributional split.
        # Otherwise regrouping roundoff can outrank a real KL improvement.
        mass = float(np.sum(expected, dtype=np.float64)) if qualified else 0.
        purity = np.bincount(top[rows], minlength=len(actions)).max()/len(rows)
        can_gate = raw in legal or depth < MAX_DEPTH
        potential = (0. if qualified or not can_gate else
                     float(len(rows)*purity**3*np.exp(-max(0., kl.mean()))))
        return _Leaf(rows, depth, legal, z, mass, potential, float(np.sum(kl, dtype=np.float64)),
                     raw, nonlegality_qualified)

    def candidates(part, variant):
        rows = part.rows
        if part.depth >= MAX_DEPTH or len(rows) < 80:
            return []
        result, seen = [], set()

        def add(weight, threshold, label, action=None):
            if source_feature_cost(weight, names) > MAX_SOURCE_FEATURES:
                return
            b = np.float32(-threshold)
            right = split_scores(x[rows], weight[None], np.asarray([b]))[:, 0] > HARD_SPLIT_TIE_EPSILON
            a, c = rows[~right], rows[right]
            if action is None and min(len(a), len(c)) < 20:
                return
            if action is not None and len(c) < 80:
                return
            if action is not None and not len(a):
                # Empty physical-mask fallbacks are sometimes necessary to
                # prove a movement already faithful on the fit cohort. They
                # are not extra partitions of a qualified WAIT-only region.
                if (part.qualified_mass > 0 or action != part.raw_action
                        or action in part.legal or not part.nonlegality_qualified
                        or not np.all(m[rows, action] > .5)):
                    return
            key = (right.tobytes(), action)
            if key in seen:
                return
            seen.add(key)
            legal = part.legal | (frozenset({action}) if action is not None else frozenset())
            score = _quick_split_gain(rows, a, c, part.legal, legal, p)
            result.append((score, label, weight, float(b), a, c, legal, action))

        for action, index in enumerate(mask_indices):
            if action in part.legal or not np.any(m[rows, action]):
                continue
            weight = np.zeros(len(names), np.float32)
            weight[index] = 1.
            add(weight, .5, f"legal:{actions[action]}", action)
        nonmask = np.asarray([i for i in range(len(names)) if i not in mask_indices])
        features = x[np.ix_(rows, nonmask)].astype(np.float64)
        response = (np.log(np.maximum(p[rows], 1.e-5))
                    if variant["initialization"] == 1 else p[rows]).astype(np.float64)
        centered = features-features.mean(0)
        response -= response.mean(0)
        covariance = np.einsum("nf,na->fa", centered, response, optimize=False)
        scale = np.maximum(centered.std(0), 1.e-8)
        covariance /= len(rows)*scale[:, None]
        strength = np.linalg.norm(covariance, axis=1)
        order = np.argsort(-strength, kind="stable")[:SEARCH_BUDGET["axis_features"]]
        basis = []
        for index in nonmask[order]:
            weight = np.zeros(len(names), np.float32)
            weight[index] = 1.
            basis.append((f"axis:{names[index]}", weight))
        for action in range(len(actions)):
            direction = covariance[:, action]
            if variant["initialization"] == 2:
                direction = direction-covariance[:, (action+1) % len(actions)]
            support, cost = [], 0
            for index in np.argsort(-np.abs(direction), kind="stable"):
                c = 2 if names[nonmask[index]] == "public_hard_risk_context" else 1
                if cost+c <= MAX_SOURCE_FEATURES and abs(direction[index]) > 1.e-10:
                    support.append(index)
                    cost += c
                if cost == MAX_SOURCE_FEATURES:
                    break
            if len(support) < 2:
                continue
            weight = np.zeros(len(names), np.float32)
            weight[nonmask[support]] = direction[support]
            weight /= max(float(np.linalg.norm(weight)), 1.e-8)
            basis.append((f"sparse:{actions[action]}", weight))
        for label, weight in basis:
            values = split_scores(x[rows], weight[None], np.zeros(1, np.float32))[:, 0]
            for quantile in variant["quantiles"]:
                add(weight, float(np.quantile(values, quantile)), f"{label}:{quantile}")
        gates = [c for c in result if c[-1] is not None]
        ordinary = sorted((c for c in result if c[-1] is None), key=lambda c: (-c[0], c[1]))
        return gates + ordinary[:SEARCH_BUDGET["candidates_per_leaf"]]

    runs = []
    best = None
    for variant in SEARCH_VARIANTS:
        root = _State([], [], [], [], [leaf(np.arange(len(x)), 0, frozenset({wait}))])
        beam = [root]
        run_best = root
        history = []
        explored = 1
        attempted = set()
        seen_states = {_state_key(root)}
        for round_index in range(MAX_LEAVES-1):
            proposals = []
            for state_index, state in enumerate(beam):
                state_key = _state_key(state)
                if len(state.leaves) >= MAX_LEAVES:
                    continue
                for index, part in enumerate(state.leaves):
                    for candidate in candidates(part, variant):
                        transition = (state_key, index, candidate[2].tobytes(),
                                      np.float32(candidate[3]).tobytes())
                        if transition not in attempted:
                            proposals.append((candidate[0], state_index, index,
                                              candidate, state, transition))
            if not proposals:
                break
            # Reserve half the exact-fit budget for informative state splits;
            # many cheap guard proposals must not starve topology learning.
            budget = SEARCH_BUDGET["exact_candidates_per_round"]
            gates = sorted((c for c in proposals if c[3][-1] is not None),
                           key=lambda c: (-c[0], c[1], c[2], c[3][1]))
            other = sorted((c for c in proposals if c[3][-1] is None),
                           key=lambda c: (-c[0], c[1], c[2], c[3][1]))
            chosen = gates[:budget//2] + other[:budget//2]
            remaining = sorted(gates[budget//2:] + other[budget//2:],
                               key=lambda c: (-c[0], c[1], c[2], c[3][1]))
            chosen += remaining[:budget-len(chosen)]
            next_states = []
            for _, _, index, c, state, transition in chosen:
                attempted.add(transition)
                _, _, weight, bias, a, b, legal, _ = c
                old = state.leaves[index]
                weights, biases = state.weights.copy(), state.bias.copy()
                left, right, parts = state.left.copy(), state.right.copy(), state.leaves.copy()
                node, new_leaf = len(weights), len(parts)
                if node:
                    found = [(j, refs) for refs in (left, right)
                             for j, ref in enumerate(refs) if ref == -1-index]
                    if len(found) != 1:
                        raise RuntimeError("independent tree lost leaf parent")
                    j, refs = found[0]
                    refs[j] = node
                weights.append(weight)
                biases.append(bias)
                left.append(-1-index)
                right.append(-1-new_leaf)
                parts[index] = leaf(a, old.depth+1, old.legal)
                parts.append(leaf(b, old.depth+1, legal))
                # Qualified regions can still be refined when their full
                # distribution or actionable mass improves. Roundoff or a
                # redundant guard alone cannot consume more of the budget.
                if old.qualified_mass > 0:
                    tolerance = max(1.e-6, len(old.rows)*_SEARCH_GAIN_EPSILON)
                    mass_gain = parts[index].qualified_mass + parts[-1].qualified_mass-old.qualified_mass
                    kl_gain = old.kl_sum-parts[index].kl_sum-parts[-1].kl_sum
                    if mass_gain <= tolerance and kl_gain <= tolerance:
                        continue
                expanded = _State(weights, biases, left, right, parts)
                key = _state_key(expanded)
                if key in seen_states:
                    continue
                seen_states.add(key)
                explored += 1
                next_states.append(expanded)
                if expanded.score(final=True) > run_best.score(final=True):
                    run_best = expanded
            # Stopping is a genuine search choice. Retain existing candidates
            # alongside growth; attempted transitions prevent re-evaluating a
            # retained parent's same split every round. Intermediate semantic
            # splits may still bridge to a later legal guard.
            pool = {_state_key(state): state for state in (*beam, *next_states, run_best)}
            previous = {_state_key(state) for state in beam}
            beam = sorted(pool.values(), key=lambda s: s.score(), reverse=True)[:SEARCH_BUDGET["beam_width"]]
            history.append({"round": round_index+1, "proposals": len(proposals),
                            "exact_fits": len(chosen), "accepted_expansions": len(next_states),
                            "retained_stopping_candidates": sum(_state_key(s) in previous for s in beam),
                            "best_fit_expected_feedback_mass": run_best.score(final=True)[0]/len(x)})
        if best is None or run_best.score(final=True) > best.score(final=True):
            best = run_best
        runs.append({"initialization": variant["initialization"],
                     "quantiles": list(variant["quantiles"]), "rounds": history,
                     "explored_topologies": explored,
                     "exact_transition_evaluations": len(attempted),
                     "fit_expected_feedback_mass": run_best.score(final=True)[0]/len(x),
                     "fit_masked_kl": -run_best.score(final=True)[1]/len(x)})
    assert best is not None
    weights = (np.stack(best.weights).astype(np.float32) if best.weights
               else np.zeros((0, len(names)), np.float32))
    structure = {"root_ref": 0 if best.weights else -1,
                 "split_weights": weights, "split_bias": np.asarray(best.bias, np.float32),
                 "left_refs": np.asarray(best.left, np.int64),
                 "right_refs": np.asarray(best.right, np.int64),
                 "leaf_count": len(best.leaves),
                 "initial_leaf_logits": np.stack([part.logits for part in best.leaves])}
    report = {"topology_origin": "independent_fit_only_sparse_search",
              "target_provenance": "complete_frozen_nn_masked_distribution",
              "actor_role_sha256": actor_role_sha256,
              "fit_rows": len(x), "fit_episode_count": len(set(episode)),
              "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
              "search_budget": dict(SEARCH_BUDGET), "search_runs": runs,
              "candidate_score": "child_proxy_minus_parent_proxy.v2",
              "empty_guard_rule": "unqualified_faithful_movement_missing_only_its_path_proof.v2",
              "qualified_leaf_growth_rule": "strict_positive_actionable_mass_or_distribution_gain.v2",
              "stopping_candidates_retained": True,
              "topology_uses_fit_only": True,
              "calibration_eligibility_validation_used": False,
              "fit_only_not_independent_certificate": True,
              "search_completeness": "bounded_heuristic_not_a_capacity_proof",
              "guard_capacity_included_in_search": True,
              "maximum_source_features_per_node": max(
                  (source_feature_cost(w, names) for w in weights), default=0),
              "execution_action_override": False}
    routed = route_leaf_indices(x, structure["split_weights"],
        structure["split_bias"], structure["left_refs"], structure["right_refs"],
        len(weights), MAX_DEPTH)
    if any(not np.array_equal(np.flatnonzero(routed == index), part.rows)
           for index, part in enumerate(best.leaves)):
        raise RuntimeError("independent sparse tree export changed fit routing")
    report["export_fit_route_mismatches"] = 0
    report["split_sha256"] = sha256(json.dumps(
        {key: value.tolist() if isinstance(value, np.ndarray) else value
         for key, value in structure.items() if key != "initial_leaf_logits"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return structure, report
