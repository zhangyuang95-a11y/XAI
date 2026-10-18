"""Fit-only queued beam search for path-certified Warehouse programs.

The legacy module supplies immutable tree representations and export helpers.
This engine changes search traversal and fit-only scoring; it does not change
Actor inference or read calibration, eligibility, or validation observations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from .warehouse_v619_leaf_fit import (
    LEAF_EPSILON, LEAF_MAXITER, LEAF_REGULARIZATION, LEAF_TOLERANCE,
    fit_constrained_leaf_distribution, float32_preference_proof,
)
from .warehouse_v619_tree import (
    MAX_DEPTH, MAX_LEAVES, MAX_SOURCE_FEATURES, FIT_MIN_ROWS, FIT_MIN_EPISODES,
    FIT_MAX_EPISODE_FRACTION, FIT_MIN_AGREEMENT, FIT_MAX_KL,
    SEARCH_INITIALIZATIONS, PROPOSAL_QUOTAS, source_feature_cost,
    masked_probabilities, _Decision, _Region, _Leaf, _State, _Proposal,
    _BudgetExhausted, _size, _tree_key, _replace, _export,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, HARD_SPLIT_TIE_EPSILON,
    certified_leaf_actions, explicit_mask_branches,
    leaf_indices as route_leaf_indices, split_scores,
)

QUEUED_METHOD = "queued_path_constrained_tree_v1"
ROBUST_METHOD = "episode_robust_explanation_tree_v1"
METHODS = (QUEUED_METHOD, ROBUST_METHOD)
ROBUST_MIN_AGREEMENT = .97
ROBUST_MAX_KL = .20
STABILITY_GROUPS = 3
_GAIN_TOLERANCE = 1.e-10
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchConfig:
    """Production limits can only be reduced for offline unit diagnostics."""

    initializations: int = 8
    max_rounds: int = 30  # Global sum of completed/partial search rounds per role fit.
    max_leaf_optimization_calls: int = 20000
    proposals_per_round: int = 48  # A page size, never a round truncation.
    beam_width: int = 4
    max_nonimproving_steps: int = 2
    axis_features: int = 12
    semantic_candidates_per_node: int = 6
    max_depth: int = MAX_DEPTH
    max_leaves: int = MAX_LEAVES

    def __post_init__(self):
        maximum = {"initializations": 8, "max_rounds": 30,
            "max_leaf_optimization_calls": 20000, "proposals_per_round": 48,
            "beam_width": 4, "max_nonimproving_steps": 2, "axis_features": 12,
            "semantic_candidates_per_node": 6, "max_depth": MAX_DEPTH,
            "max_leaves": MAX_LEAVES}
        for name, limit in maximum.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"invalid queued tree budget: {name}")


PRODUCTION_SEARCH_CONFIG = SearchConfig()


def _improves(new, old):
    for a, b in zip(new, old, strict=True):
        if a > b + _GAIN_TOLERANCE:
            return True
        if abs(a-b) > _GAIN_TOLERANCE:
            return False
    return False


@dataclass
class _BeamEntry:
    state: _State
    nonimproving_steps: int = 0


def _rank_state(state):
    return (*(-float(value) for value in state.score), state.key)


def _choose_beam(entries, best, width):
    """Protect the best and reserve one slot for a legal non-greedy transition."""
    unique = {}
    for entry in entries:
        old = unique.get(entry.state.key)
        if old is None or entry.nonimproving_steps < old.nonimproving_steps:
            unique[entry.state.key] = entry
    unique.pop(best.key, None)
    ordered = sorted(unique.values(), key=lambda entry: (
        _rank_state(entry.state), entry.nonimproving_steps))
    selected = [_BeamEntry(best, 0)]
    if width == 1:
        return selected
    transition = next((entry for entry in ordered if entry.nonimproving_steps > 0), None)
    slots = width - 1 - int(transition is not None)
    selected.extend(ordered[:slots])
    if transition is not None and all(entry.state.key != transition.state.key for entry in selected):
        selected.append(transition)
    if len(selected) < width:
        selected.extend(entry for entry in ordered
            if all(entry.state.key != chosen.state.key for chosen in selected))
    return selected[:width]


def fit_legal_structure(
        encoded: np.ndarray, masks: np.ndarray, target_probs: np.ndarray,
        episode_ids: Sequence[str], *, feature_names: Sequence[str],
        actor_role_sha256: str, actions: Sequence[str] =
        ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"),
        method: str, config: SearchConfig | Mapping | None = None) -> tuple[dict, dict]:
    """Return a standalone sparse topology and fit-only diagnostic report.

    The hard 20,000 budget counts actual leaf API calls (cache misses). Each
    API may enumerate up to five certified actions; actual solver calls are
    reported separately. Cache keys bind rows and legal sets, inside this
    immutable target/Actor invocation. A budget interruption never returns a
    partially optimized candidate.
    """
    if method not in METHODS:
        raise ValueError("unknown queued tree method")
    cfg = (PRODUCTION_SEARCH_CONFIG if config is None else config
           if isinstance(config, SearchConfig) else SearchConfig(**dict(config)))
    x, m, p = (np.asarray(value, np.float32) for value in
               (encoded, masks, target_probs))
    names, actions = tuple(feature_names), tuple(actions)
    episode = np.asarray([str(value) for value in episode_ids])
    if (len(actions) != 5 or len(set(actions)) != 5 or "WAIT" not in actions
            or len(names) != len(set(names))):
        raise ValueError("invalid constrained tree feature/action schema")
    try:
        mask_indices = tuple(names.index(f"legal_{action.lower()}") for action in actions)
    except ValueError as error:
        raise ValueError("constrained tree requires explicit legal_ACTION features") from error
    wait = actions.index("WAIT")
    if (x.ndim != 2 or x.shape[1] != len(names) or len(x) < 2
            or m.shape != (len(x), 5) or p.shape != m.shape
            or episode.shape != (len(x),) or any(not value for value in episode)
            or not isinstance(actor_role_sha256, str) or len(actor_role_sha256) != 64
            or any(value not in "0123456789abcdef" for value in actor_role_sha256)
            or any(not np.isfinite(value).all() for value in (x, m, p))
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)
            or np.any(m[:, wait] != 1) or not np.array_equal(x[:, mask_indices], m)
            or np.any(p < 0) or np.any(p[m == 0] > 1.e-8)
            or not np.allclose(p.sum(-1), 1., atol=1.e-6, rtol=0)):
        raise ValueError("constrained tree requires complete masked NN fit distributions")
    top = p.argmax(-1)
    nonmask = np.asarray([i for i in range(len(names)) if i not in mask_indices], int)
    # Group complete episodes, rather than frames.  This is a fit-side
    # stability surrogate, not an independent qualification certificate.
    episode_order = sorted(set(episode), key=lambda value: sha256(value.encode()).digest())
    episode_groups = {value: index % STABILITY_GROUPS
                      for index, value in enumerate(episode_order)}
    group = np.asarray([episode_groups[value] for value in episode], np.int8)
    fit_cache, quick_cache, state_cache = {}, {}, {}
    robustness = {}
    calls = solves = cache_hits = rejected_solver_candidates = 0
    failure_examples = []
    current_call_limit = cfg.max_leaf_optimization_calls

    def regions(tree, rows=None, depth=0, legal=None, path=()):
        rows = np.arange(len(x)) if rows is None else rows
        legal = frozenset({wait}) if legal is None else legal
        region = _Region(path, tree, rows, depth, legal)
        yield region
        if tree is None:
            return
        weight = np.asarray(tree.weight, np.float32)
        go_right = split_scores(x[rows], weight[None],
                                np.asarray([tree.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
        left_legal, right_legal = explicit_mask_branches(weight, tree.bias, mask_indices)
        yield from regions(tree.left, rows[~go_right], depth+1,
                           legal | left_legal, path+(0,))
        yield from regions(tree.right, rows[go_right], depth+1,
                           legal | right_legal, path+(1,))

    def fit_leaf(region):
        nonlocal calls, solves, cache_hits
        rows, legal = region.rows, region.legal
        key = (sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).digest(),
               tuple(sorted(legal)))
        if key in fit_cache:
            z, optimizer = fit_cache[key]
            cache_hits += 1
        else:
            if calls >= current_call_limit or calls >= cfg.max_leaf_optimization_calls:
                raise _BudgetExhausted()
            calls += 1
            try:
                z, optimizer = fit_constrained_leaf_distribution(
                    p[rows], m[rows], tuple(sorted(legal)), epsilon=LEAF_EPSILON,
                    regularization=LEAF_REGULARIZATION, maxiter=LEAF_MAXITER,
                    tolerance=LEAF_TOLERANCE)
            except RuntimeError as error:
                solves += int(getattr(error, "enumerated_action_solves", 0))
                # Failed and interrupted solver attempts consume the same API
                # budget as successful attempts.  A failed tree is never used.
                raise
            solves += int(optimizer.get("enumerated_action_solves", len(legal)))
            chosen = int(np.asarray(z).argmax())
            if (not optimizer.get("success")
                    or not float32_preference_proof(z, chosen, legal,
                                                   epsilon=LEAF_EPSILON)["passed"]):
                raise RuntimeError("constrained leaf returned an uncertified float32 preference")
            fit_cache[key] = (z, optimizer)
        if not len(rows):
            robustness[region.path] = {"qualified": False, "groups": []}
            return _Leaf(region, z, optimizer, False, 0., 0., 0.)
        chosen = int(z.argmax())
        if np.any(m[rows, chosen] != 1):
            raise RuntimeError("constrained tree path proof contradicts fit physical mask")
        # Stable log probabilities preserve every supplied nonzero target,
        # without clipping the full-NN tail at the proposal-response floor.
        masked_z = np.where(m[rows] > .5, z.astype(np.float64)[None], -np.inf)
        logq = masked_z-logsumexp(masked_z, axis=-1, keepdims=True)
        target = p[rows].astype(np.float64)
        kl = np.sum(target*(np.log(np.maximum(target, 1.e-300))
                            - np.where(m[rows] > .5, logq, 0.)), axis=-1)
        matching = logq.argmax(-1) == top[rows]
        _, counts = np.unique(episode[rows], return_counts=True)
        agreement = float(np.mean(matching))
        qualified = bool(len(rows) >= FIT_MIN_ROWS and len(counts) >= FIT_MIN_EPISODES
                         and counts.max()/len(rows) <= FIT_MAX_EPISODE_FRACTION
                         and agreement >= FIT_MIN_AGREEMENT and kl.mean() <= FIT_MAX_KL)
        groups = []
        for index in range(STABILITY_GROUPS):
            selected = group[rows] == index
            groups.append({"group": index, "rows": int(selected.sum()),
                "episodes": len(set(episode[rows[selected]])),
                "agreement": float(matching[selected].mean()) if selected.any() else None,
                "masked_kl": float(kl[selected].mean()) if selected.any() else None})
        robust = bool(qualified and agreement >= ROBUST_MIN_AGREEMENT
            and kl.mean() <= ROBUST_MAX_KL and all(
                part["episodes"] >= 2 and part["agreement"] >= ROBUST_MIN_AGREEMENT
                and part["masked_kl"] <= ROBUST_MAX_KL for part in groups))
        robustness[region.path] = {"qualified": robust, "groups": groups}
        mass = float(np.sum(p[rows, chosen]*matching, dtype=np.float64)) if qualified else 0.
        return _Leaf(region, z, optimizer, qualified, mass,
                     float(np.sum(kl, dtype=np.float64)), agreement)

    def evaluate(tree):
        key = _tree_key(tree)
        if key in state_cache:
            return state_cache[key]
        size, depth = _size(tree)
        if size > cfg.max_leaves or depth > cfg.max_depth:
            raise ValueError("constrained proposal exceeds tree capacity")
        parts = list(regions(tree))
        leaves = [fit_leaf(part) for part in parts if part.tree is None]
        feedback = sum(leaf.expected_mass for leaf in leaves)/len(x)
        kl = sum(leaf.kl_sum for leaf in leaves)/len(x)
        explanation = sum(len(leaf.region.rows) for leaf in leaves
                          if robustness[leaf.region.path]["qualified"])/len(x)
        score = ((feedback, -kl, -size) if method == QUEUED_METHOD
                 else (explanation, feedback, -kl, -size))
        state = _State(tree, leaves, parts, score, key)
        # Store stability evidence on the fitted state, not a mutable path map:
        # the same path can refer to different regions in different trees.
        state.robustness = {leaf.region.path: robustness[leaf.region.path]
                            for leaf in leaves}
        state_cache[key] = state
        # Tree states contain row-index partitions. Bound their memoization;
        # the compact leaf-result cache retains the expensive optimizer work.
        if len(state_cache) > 256:
            del state_cache[next(iter(state_cache))]
        return state

    def quick(tree, rows, legal):
        key = (_tree_key(tree),
               sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).digest(),
               tuple(sorted(legal)))
        if key not in quick_cache:
            if tree is None:
                n = len(rows)
                masses = [float(np.sum(p[rows, a]*(top[rows] == a), dtype=np.float64))
                          *float(np.mean(top[rows] == a))**3 if n else 0.
                          for a in sorted(legal)]
                quick_cache[key] = max(masses, default=0.)
            else:
                weight = np.asarray(tree.weight, np.float32)
                right = split_scores(x[rows], weight[None],
                    np.asarray([tree.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
                ll, rl = explicit_mask_branches(weight, tree.bias, mask_indices)
                quick_cache[key] = (quick(tree.left, rows[~right], legal | ll)
                                    + quick(tree.right, rows[right], legal | rl))
        return quick_cache[key]

    def guard(action, child=None):
        weight = np.zeros(len(names), np.float32)
        weight[mask_indices[action]] = 1.
        return _Decision(tuple(weight), -.5, None, child)

    def preferred_guard(rows, legal):
        choices = [a for a in range(5) if a not in legal
                   and np.any(m[rows, a] > .5) and np.any(top[rows] == a)]
        return (sorted(choices, key=lambda a: (-int(np.sum(top[rows] == a)), a))[0]
                if choices else None)

    def predicates(region, initialization):
        rows = region.rows
        if len(rows) < 2:
            return []
        out, seen = [], set()

        def add(weight, threshold, label, is_guard=False):
            weight = np.asarray(weight, np.float32)
            if source_feature_cost(weight, names) > MAX_SOURCE_FEATURES:
                return
            bias = float(np.float32(-threshold))
            right = split_scores(x[rows], weight[None],
                np.asarray([bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            if not is_guard and min(int(right.sum()), int((~right).sum())) < 20:
                return
            key = (weight.tobytes(), np.float32(bias).tobytes())
            if key not in seen:
                seen.add(key)
                out.append((weight, bias, label, is_guard, right))

        for action, index in enumerate(mask_indices):
            if action != wait and np.any(m[rows, action]) and np.any(top[rows] == action):
                weight = np.zeros(len(names), np.float32)
                weight[index] = 1.
                add(weight, .5, f"legal:{actions[action]}", True)
        if not len(nonmask):
            return out
        features = x[np.ix_(rows, nonmask)].astype(np.float64)
        centered = features-features.mean(0)
        response = (np.log(np.maximum(p[rows], 1.e-5))
                    if initialization["response"] == "log_probability" else p[rows]).astype(np.float64)
        response -= response.mean(0)
        covariance = np.einsum("nf,na->fa", centered, response, optimize=False)
        covariance /= len(rows)*np.maximum(centered.std(0), 1.e-8)[:, None]
        strength = np.linalg.norm(covariance, axis=1)
        order = np.argsort(-strength, kind="stable")[:cfg.axis_features]
        basis = []
        for index in nonmask[order]:
            weight = np.zeros(len(names), np.float32)
            weight[index] = 1.
            basis.append((f"axis:{names[index]}", weight))
        for action in range(5):
            direction = covariance[:, action].copy()
            if initialization["contrast"]:
                direction -= covariance[:, (action+initialization["contrast"]) % 5]
            support, cost = [], 0
            for index in np.argsort(-np.abs(direction), kind="stable"):
                price = 2 if names[nonmask[index]] == "public_hard_risk_context" else 1
                if cost+price <= MAX_SOURCE_FEATURES and abs(direction[index]) > 1.e-10:
                    support.append(index)
                    cost += price
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
            for quantile in initialization["quantiles"]:
                add(weight, float(np.quantile(values, quantile)), f"{label}:{quantile}")
        gates = [candidate for candidate in out if candidate[3]]
        ordinary = [candidate for candidate in out if not candidate[3]]
        # Rank semantic splits with guards already attached. This prevents
        # unproven movement partitions being incorrectly valued as WAIT-only.
        def joint_gain(candidate):
            w, b, _, _, right = candidate
            la = preferred_guard(rows[~right], region.legal)
            ra = preferred_guard(rows[right], region.legal)
            child = _Decision(tuple(w), b,
                              guard(la) if la is not None else None,
                              guard(ra) if ra is not None else None)
            return quick(child, rows, region.legal)-quick(None, rows, region.legal)
        ordinary.sort(key=lambda candidate: (-joint_gain(candidate), candidate[2],
                                             candidate[0].tobytes()))
        return gates+ordinary[:cfg.semantic_candidates_per_node]

    def recipes(region, initialization):
        result = []
        if region.depth >= cfg.max_depth:
            return result
        for weight, bias, label, is_guard, right in predicates(region, initialization):
            base = _Decision(tuple(weight), bias)
            result.append((base, label))
            if is_guard or region.depth+2 > cfg.max_depth:
                continue
            la = preferred_guard(region.rows[~right], region.legal)
            ra = preferred_guard(region.rows[right], region.legal)
            if la is not None or ra is not None:
                joint = _Decision(tuple(weight), bias,
                    guard(la) if la is not None else None,
                    guard(ra) if ra is not None else None)
                result.append((joint, f"joint:{label}"))
                if la is not None and ra is not None:
                    result.append((_Decision(tuple(weight), bias, guard(la), None),
                                   f"joint_left:{label}"))
                    result.append((_Decision(tuple(weight), bias, None, guard(ra)),
                                   f"joint_right:{label}"))
            # A shared guard before a semantic split uses two nodes, rather
            # than separately guarding both children with the same movement.
            for action in sorted({value for value in (la, ra) if value is not None}):
                result.append((guard(action, base), f"guard_then_semantic:{actions[action]}:{label}"))
        return [(tree, label) for tree, label in result
                if region.depth+_size(tree)[1] <= cfg.max_depth]

    def proposals(state, initialization, attempted):
        offered = {}

        def add(tree, kind, label, gain):
            count, depth = _size(tree)
            if count > cfg.max_leaves or depth > cfg.max_depth:
                return
            key = _tree_key(tree)
            if key == state.key or key in attempted:
                return
            proposal = _Proposal(tree, kind, label, float(gain), key)
            if key not in offered or (kind, label) < (offered[key].kind, offered[key].label):
                offered[key] = proposal

        leaves = [region for region in state.regions if region.tree is None]
        branches = [region for region in state.regions if region.tree is not None]
        recipe_cache = {}

        def get_recipes(region):
            if region.path not in recipe_cache:
                recipe_cache[region.path] = recipes(region, initialization)
            return recipe_cache[region.path]

        for region in leaves:
            before = quick(None, region.rows, region.legal)
            for replacement, label in get_recipes(region):
                add(_replace(state.tree, region.path, replacement), "grow", label,
                    quick(replacement, region.rows, region.legal)-before)
        for region in branches:
            before = quick(region.tree, region.rows, region.legal)
            add(_replace(state.tree, region.path, None), "prune",
                f"prune:{region.path}", quick(None, region.rows, region.legal)-before)
            for weight, bias, label, _, _ in predicates(region, initialization):
                replacement = _Decision(tuple(weight), bias, region.tree.left, region.tree.right)
                add(_replace(state.tree, region.path, replacement), "refit", label,
                    quick(replacement, region.rows, region.legal)-before)
            for replacement, label in get_recipes(region):
                add(_replace(state.tree, region.path, replacement), "replace", label,
                    quick(replacement, region.rows, region.legal)-before)
        # Reclaim a low-contribution subtree and grow a different existing
        # branch in one proposal, so strict monotonicity does not forbid the
        # temporary capacity loss of a standalone pruning step.
        reclaim = sorted(branches, key=lambda region: (
            quick(region.tree, region.rows, region.legal)/max(1, _size(region.tree)[0]-1),
            region.path))[:4]
        for source in reclaim:
            pruned = _replace(state.tree, source.path, None)
            released_gain = (quick(None, source.rows, source.legal)
                             - quick(source.tree, source.rows, source.legal))
            for destination in leaves:
                if destination.path[:len(source.path)] == source.path:
                    continue
                before = quick(None, destination.rows, destination.legal)
                for replacement, label in get_recipes(destination):
                    add(_replace(pruned, destination.path, replacement), "reallocate",
                        f"prune:{source.path}:grow:{destination.path}:{label}",
                        released_gain+quick(replacement, destination.rows, destination.legal)-before)
        groups = {kind: [] for kind in PROPOSAL_QUOTAS}
        for candidate in offered.values():
            groups[candidate.kind].append(candidate)
        rank = lambda candidate: (-candidate.gain, candidate.kind, candidate.label, candidate.key)
        chosen = []
        for kind, quota in PROPOSAL_QUOTAS.items():
            groups[kind].sort(key=rank)
            chosen.extend(groups[kind][:quota])
        # Fair first-page scheduling does not discard the rest of the menu.
        # The returned queue is drained in pages, with only a hard API budget
        # permitting an unevaluated suffix.
        chosen.sort(key=rank)
        chosen_keys = {candidate.key for candidate in chosen}
        remainder = sorted((candidate for candidate in offered.values()
                            if candidate.key not in chosen_keys), key=rank)
        chosen.extend(remainder)
        return chosen, len(offered)

    def initial_trees():
        root = next(regions(None))
        movements = sorted((action for action in range(5) if action != wait),
            key=lambda action: (-int(np.sum(top == action)), action))
        chains = [guard(a, guard(b)) for a in movements for b in movements if a != b]
        pool = [None]
        pool.extend(guard(action) for action in movements)
        pool.extend(chains[:2])
        # A data-derived semantic/guard seed participates as well as the legal
        # guard chains.  Every candidate is capacity checked before selection.
        semantic = [(tree, label) for tree, label in recipes(root, SEARCH_INITIALIZATIONS[3])
                    if "axis:" in label or "sparse:" in label]
        semantic.sort(key=lambda item: (-quick(item[0], root.rows, root.legal), item[1]))
        if semantic:
            pool.append(semantic[0][0])
        pool.extend(chains[2:])
        pool.extend(tree for tree, _ in semantic[1:])
        unique, seen = [], set()
        for tree in pool:
            key = _tree_key(tree)
            leaves, depth = _size(tree)
            if leaves <= cfg.max_leaves and depth <= cfg.max_depth and key not in seen:
                unique.append(tree)
                seen.add(key)
        # Under a one-leaf diagnostic limit, eight distinct structures cannot
        # exist.  Production's registered limits always permit all eight.
        return [unique[index % len(unique)] for index in range(cfg.initializations)]

    starts = initial_trees()
    best, runs = None, []
    global_rounds = 0
    operation_counts = {kind: 0 for kind in PROPOSAL_QUOTAS}
    for start, initialization in enumerate(SEARCH_INITIALIZATIONS[:cfg.initializations]):
        before_calls, before_solves = calls, solves
        # Deterministic cumulative shares let later starts use earlier unused
        # calls, while never exceeding the one per-role API-call hard bound.
        current_call_limit = max(1, ((start+1)*cfg.max_leaf_optimization_calls)//cfg.initializations)
        # Both limits are cumulative shares. An early stop passes its unused
        # rounds to later starts; even when rounds < starts, the zero-share
        # starts cannot spend a round reserved for a later initialization.
        current_round_limit = ((start+1)*cfg.max_rounds)//cfg.initializations
        before_rounds = global_rounds
        available_rounds = current_round_limit-global_rounds
        root_fallback = evaluate(None)
        initial_failure = None
        try:
            state = evaluate(starts[start])
            initial_interrupted = False
        except _BudgetExhausted:
            # A diagnostic budget may not finish the requested initial tree.
            # Preserve the completely evaluated safe root, and account for
            # every leaf call already spent on the rejected partial tree.
            state, initial_interrupted = root_fallback, True
            initial_failure = "leaf_optimization_call_budget"
        except RuntimeError as error:
            rejected_solver_candidates += 1
            initial_failure = str(error)[:1200]
            if len(failure_examples) < 20:
                failure_examples.append({"initialization": start, "round": 0,
                    "operation": "initial_topology", "error": initial_failure})
            state, initial_interrupted = root_fallback, True
        initial_state = state
        start_best = state
        beam = [_BeamEntry(state)]
        attempted, history = {state.key}, []
        expanded = set()
        stop = "global_round_share_limit"
        remaining_queue = 0
        for round_index in range(available_rounds):
            offered = {}
            parent_entries = []
            for entry in beam:
                if entry.state.key in expanded:
                    continue
                expanded.add(entry.state.key)
                parent_entries.append(entry)
                candidates, _ = proposals(entry.state, initialization, attempted)
                for candidate in candidates:
                    old = offered.get(candidate.key)
                    # Use the best predecessor for a duplicate topology so its
                    # transition debt is conservative and deterministic.
                    value = (candidate, entry)
                    if old is None or (_rank_state(entry.state), entry.nonimproving_steps) < (
                            _rank_state(old[1].state), old[1].nonimproving_steps):
                        offered[candidate.key] = value
            # Dict insertion preserves each parent's operation-balanced first
            # page followed by its remaining gain-ranked proposals. Replacing
            # a duplicate's predecessor does not change that queue position.
            queue = deque(offered.values())
            proposal_count = len(queue)
            if not queue:
                stop = "no_new_feasible_proposals"
                break
            evaluated, failed, interrupted, pages = 0, 0, False, 0
            transitions = []
            nonimproving_evaluations = 0
            previous_score = start_best.score
            while queue:
                pages += 1
                page = [queue.popleft() for _ in range(min(cfg.proposals_per_round, len(queue)))]
                for offset, (candidate, parent) in enumerate(page):
                    attempted.add(candidate.key)
                    try:
                        fitted = evaluate(candidate.tree)
                    except _BudgetExhausted:
                        # Put the incomplete candidate back too; it has not
                        # acquired a complete score and must not enter a beam.
                        queue.extendleft(reversed(page[offset:]))
                        interrupted = True
                        break
                    except RuntimeError as error:
                        failed += 1
                        rejected_solver_candidates += 1
                        if len(failure_examples) < 20:
                            failure_examples.append({"initialization": start,
                                "round": round_index+1, "operation": candidate.kind,
                                "error": str(error)[:1200]})
                        continue
                    evaluated += 1
                    operation_counts[candidate.kind] += 1
                    improvement = _improves(fitted.score, parent.state.score)
                    debt = 0 if improvement else parent.nonimproving_steps+1
                    nonimproving_evaluations += int(not improvement)
                    if debt <= cfg.max_nonimproving_steps:
                        transitions.append(_BeamEntry(fitted, debt))
                    if _improves(fitted.score, start_best.score) or (
                            fitted.score == start_best.score and fitted.key < start_best.key):
                        start_best = fitted
                if interrupted:
                    break
            beam = _choose_beam(transitions, start_best, cfg.beam_width)
            remaining_queue = len(queue)
            changed = _improves(start_best.score, previous_score)
            global_rounds += 1
            history.append({"round": round_index+1, "global_round": global_rounds,
                "proposals": proposal_count,
                "scheduled_proposals": proposal_count, "pages": pages,
                "complete_evaluations": evaluated, "failed_evaluations": failed,
                "remaining_queue": remaining_queue,
                "accepted": changed, "previous_score": list(previous_score),
                "score": list(start_best.score),
                "expanded_beam_states": len(parent_entries),
                "nonimproving_complete_evaluations": nonimproving_evaluations,
                "beam": [{"tree_sha256": entry.state.key.hex(),
                    "score": list(entry.state.score),
                    "nonimproving_steps": entry.nonimproving_steps}
                    for entry in beam],
                "leaf_optimization_calls_cumulative": calls,
                "budget_interrupted": interrupted})
            _LOG.info(
                "Warehouse program search round_complete method=%s start=%d "
                "round=%d global_rounds=%d/%d api_calls=%d convex_calls=%d "
                "fit_score=%s proposals=%d complete=%d failed=%d remaining_queue=%d",
                method, start+1, round_index+1, global_rounds, cfg.max_rounds,
                calls, solves, list(start_best.score), proposal_count, evaluated,
                failed, remaining_queue)
            if interrupted:
                stop = "leaf_optimization_call_budget"
                break
            if not any(entry.state.key not in expanded for entry in beam):
                stop = "complete_queue_no_unexpanded_beam_state"
                break
        if best is None or _improves(start_best.score, best.score) or (
                start_best.score == best.score and start_best.key < best.key):
            best = start_best
        runs.append({"initialization": dict(initialization), "rounds": history,
            "stop_reason": stop, "leaf_optimization_calls": calls-before_calls,
            "actual_convex_solver_calls": solves-before_solves,
            "fit_score": list(start_best.score), "tree_sha256": start_best.key.hex(),
            "requested_initial_tree_sha256": _tree_key(starts[start]).hex(),
            "initial_tree_sha256": initial_state.key.hex(),
            "initial_tree_leaf_count": _size(initial_state.tree)[0],
            "initial_tree_incomplete": initial_interrupted,
            "initial_tree_failure": initial_failure,
            "remaining_queue": remaining_queue,
            "global_rounds_before_start": before_rounds,
            "global_rounds_after_start": global_rounds,
            "round_budget_available_at_start": available_rounds,
            "reserved_cumulative_round_limit": current_round_limit,
            "reserved_cumulative_leaf_call_limit": current_call_limit})
        _LOG.info(
            "Warehouse program search start_complete method=%s start=%d/%d "
            "rounds=%d global_rounds=%d/%d reserved_round_limit=%d api_calls=%d "
            "convex_calls=%d fit_score=%s remaining_queue=%d stop=%s",
            method, start+1, cfg.initializations, len(history), global_rounds,
            cfg.max_rounds, current_round_limit, calls, solves,
            list(start_best.score), remaining_queue, stop)
    assert best is not None
    if not (calls <= cfg.max_leaf_optimization_calls and 0 <= solves <= 5*calls
            and global_rounds == sum(len(run["rounds"]) for run in runs)
            and 0 <= global_rounds <= cfg.max_rounds):
        raise RuntimeError("constrained tree global round/leaf/solver accounting is invalid")
    structure = _export(best.tree, best.leaves, len(names))
    weights = structure["split_weights"]
    if any(source_feature_cost(weight, names) > MAX_SOURCE_FEATURES for weight in weights):
        raise RuntimeError("constrained tree export exceeds source-feature capacity")
    routes = route_leaf_indices(x, weights, structure["split_bias"],
        structure["left_refs"], structure["right_refs"], len(weights), cfg.max_depth)
    proof = certified_leaf_actions(weights, structure["split_bias"],
        structure["left_refs"], structure["right_refs"], len(weights),
        structure["leaf_count"], mask_indices, explicit=True, wait_action=wait)
    if any(not np.array_equal(np.flatnonzero(routes == index), leaf.region.rows)
           for index, leaf in enumerate(best.leaves)):
        raise RuntimeError("constrained tree export changed hard routing")
    for index, leaf in enumerate(best.leaves):
        if (proof[index] != leaf.region.legal
                or not float32_preference_proof(leaf.logits, int(leaf.logits.argmax()),
                                               proof[index])["passed"]):
            raise RuntimeError("constrained tree export changed path preference proof")
    raw = structure["initial_leaf_logits"][routes].argmax(-1)
    whole_illegal = int(np.sum(m[np.arange(len(x)), raw] != 1))
    if whole_illegal:
        raise RuntimeError("path-constrained tree contains a raw illegal fit action")
    feedback_mass = sum(leaf.expected_mass for leaf in best.leaves)/len(x)
    all_kl = sum(leaf.kl_sum for leaf in best.leaves)/len(x)
    robust_coverage = sum(len(leaf.region.rows) for leaf in best.leaves
        if best.robustness[leaf.region.path]["qualified"])/len(x)
    report = {"topology_origin": "independent_fit_only_queued_beam_whole_tree_search",
        "method": method,
        "target_provenance": "complete_frozen_nn_masked_distribution",
        "actor_role_sha256": actor_role_sha256, "fit_rows": len(x),
        "fit_episode_count": len(set(episode)),
        "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
        "search_budget": asdict(cfg), "production_configuration": cfg == PRODUCTION_SEARCH_CONFIG,
        "round_budget_scope": "global_per_role_fit_across_all_initializations",
        "global_search_rounds": global_rounds,
        "global_search_rounds_remaining": cfg.max_rounds-global_rounds,
        "round_share_rule": "floor((start_index_plus_one)*max_rounds/initializations)_cumulative_with_unused_carry",
        "initial_structure": "eight_distinct_fit_ranked_certified_guard_topologies",
        "initialization_scope": "deterministic_fit_only_guard_topologies_and_predicate_grids",
        "distinct_requested_initializations": len({_tree_key(tree) for tree in starts}),
        "search_runs": runs, "leaf_optimization_calls": calls,
        "leaf_optimization_calls_remaining": cfg.max_leaf_optimization_calls-calls,
        "actual_convex_solver_calls": solves, "leaf_cache_hits": cache_hits,
        "rejected_solver_candidates": rejected_solver_candidates,
        "solver_failure_examples": failure_examples,
        "operation_complete_evaluations": operation_counts,
        "proposal_quotas": dict(PROPOSAL_QUOTAS),
        "proposal_queue_order": "operation_balanced_front_page_then_complete_gain_ranked_suffix",
        "fit_expected_feedback_coverage": feedback_mass, "fit_masked_kl": all_kl,
        "fit_stable_explanation_coverage": robust_coverage,
        "fit_whole_tree_raw_illegal_count": whole_illegal,
        "fit_whole_tree_path_uncertified_count": 0,
        "fit_qualified_leaf_coverage": sum(len(leaf.region.rows) for leaf in best.leaves if leaf.qualified)/len(x),
        "fit_leaf_reports": [{"leaf": index, "rows": len(leaf.region.rows),
            "depth": leaf.region.depth, "path_legal_actions": sorted(leaf.region.legal),
            "raw_action": int(leaf.logits.argmax()), "fit_qualified": leaf.qualified,
            "fit_stable_explanation_qualified": best.robustness[leaf.region.path]["qualified"],
            "episode_stability_groups": best.robustness[leaf.region.path]["groups"],
            "agreement": leaf.agreement, "masked_kl": leaf.kl_sum/len(leaf.region.rows) if len(leaf.region.rows) else None,
            "expected_feedback_mass": leaf.expected_mass, "optimizer": leaf.optimizer}
            for index, leaf in enumerate(best.leaves)],
        "fit_proxy_qualification": {"minimum_rows": FIT_MIN_ROWS,
            "minimum_episodes": FIT_MIN_EPISODES, "maximum_episode_fraction": FIT_MAX_EPISODE_FRACTION,
            "minimum_agreement": FIT_MIN_AGREEMENT, "maximum_kl": FIT_MAX_KL},
        "fit_explanation_margin": {"minimum_agreement": ROBUST_MIN_AGREEMENT,
            "maximum_kl": ROBUST_MAX_KL, "episode_groups": STABILITY_GROUPS,
            "minimum_episodes_per_group": 2,
            "group_assignment": "sha256_sorted_complete_fit_episodes_round_robin"},
        "candidate_score": ("fit_qualified_expected_feedback_then_all_frame_kl_then_leaf_count.v1"
            if method == QUEUED_METHOD else
            "fit_episode_stable_explanation_coverage_then_expected_feedback_then_kl_then_leaf_count.v1"),
        "candidate_ranking": "subtree_quick_gain_with_joint_guard_capacity.v1",
        "accepted_updates_strictly_monotonic": False,
        "historical_best_preserved": True,
        "proposal_queue_drained_before_no_gain_stop": True,
        "remaining_proposal_queue": sum(run["remaining_queue"] for run in runs),
        "proposal_page_size": cfg.proposals_per_round,
        "beam_width": cfg.beam_width,
        "maximum_nonimproving_transition_steps": cfg.max_nonimproving_steps,
        "initialization_tie_rule": "fit_score_then_lexicographic_tree_sha256",
        "hard_routing_version": HARD_SPLIT_ROUTING_VERSION,
        "leaf_optimizer": {"epsilon": LEAF_EPSILON, "regularization": LEAF_REGULARIZATION,
                           "maxiter": LEAF_MAXITER, "tolerance": LEAF_TOLERANCE},
        "maximum_source_features_per_node": max((source_feature_cost(w, names) for w in weights), default=0),
        "guard_capacity_included_in_search": True,
        "whole_tree_raw_action_path_proof": True,
        "semantic_guard_joint_proposals": True,
        "topology_uses_fit_only": True,
        "calibration_eligibility_validation_used": False,
        "fit_only_not_independent_certificate": True,
        "search_completeness": "bounded_heuristic_not_a_capacity_or_global_optimality_proof",
        "execution_action_override": False,
        "split_sha256": sha256(json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in structure.items() if key != "initial_leaf_logits"},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return structure, report
