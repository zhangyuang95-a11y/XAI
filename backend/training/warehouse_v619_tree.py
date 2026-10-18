"""Fit-only, path-constrained whole-tree search for Warehouse v6.19.

Every evaluated leaf, including an empty child, is optimized over actions
certified by its actual ancestor legal-bit guards.  Search never changes an
Actor and never sees calibration, eligibility or validation episodes.

This is a bounded heuristic, not a TAO or mixed-integer optimality guarantee.
Atomic subtree replacements and prune/grow reallocations can improve an
already full tree; useful semantic partitions and their guards are proposed
together, with both depth and empty children charged to the capacity budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from .warehouse_v619_leaf_fit import (
    LEAF_EPSILON, LEAF_MAXITER, LEAF_REGULARIZATION, LEAF_TOLERANCE,
    fit_constrained_leaf_distribution, float32_preference_proof,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, HARD_SPLIT_TIE_EPSILON,
    certified_leaf_actions, explicit_mask_branches,
    leaf_indices as route_leaf_indices, split_scores,
)


MAX_DEPTH = 5
MAX_LEAVES = 16
MAX_SOURCE_FEATURES = 3
FIT_MIN_ROWS = 80
FIT_MIN_EPISODES = 8
FIT_MAX_EPISODE_FRACTION = .20
FIT_MIN_AGREEMENT = .95
FIT_MAX_KL = .25
_GAIN_TOLERANCE = 1.e-10


@dataclass(frozen=True)
class SearchConfig:
    """Only smaller diagnostic budgets are accepted outside production."""

    initializations: int = 8
    max_rounds: int = 30
    max_leaf_optimization_calls: int = 20000
    proposals_per_round: int = 48
    axis_features: int = 12
    semantic_candidates_per_node: int = 6
    max_depth: int = MAX_DEPTH
    max_leaves: int = MAX_LEAVES

    def __post_init__(self):
        upper = {"initializations": 8, "max_rounds": 30,
                 "max_leaf_optimization_calls": 20000,
                 "proposals_per_round": 48, "axis_features": 12,
                 "semantic_candidates_per_node": 6,
                 "max_depth": MAX_DEPTH, "max_leaves": MAX_LEAVES}
        for name, maximum in upper.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid constrained tree budget: {name}")


PRODUCTION_SEARCH_CONFIG = SearchConfig()
# Fixed fit-only initializations. No validation-dependent seed or restart.
SEARCH_INITIALIZATIONS = (
    {"id": 0, "response": "probability", "quantiles": (.25, .50, .75), "contrast": 0},
    {"id": 1, "response": "log_probability", "quantiles": (.20, .40, .60, .80), "contrast": 0},
    {"id": 2, "response": "probability", "quantiles": (.15, .35, .65, .85), "contrast": 1},
    {"id": 3, "response": "log_probability", "quantiles": (.10, .30, .50, .70, .90), "contrast": 1},
    {"id": 4, "response": "probability", "quantiles": (.20, .50, .80), "contrast": 2},
    {"id": 5, "response": "log_probability", "quantiles": (.15, .40, .60, .85), "contrast": 2},
    {"id": 6, "response": "probability", "quantiles": (.10, .35, .65, .90), "contrast": 3},
    {"id": 7, "response": "log_probability", "quantiles": (.25, .50, .75), "contrast": 3},
)
# Per-round fair scheduling prevents cheap guards from starving node refits or
# reallocations. Remaining slots are filled in deterministic global gain order.
PROPOSAL_QUOTAS = {"grow": 12, "refit": 12, "replace": 8,
                   "prune": 4, "reallocate": 12}


def source_feature_cost(weights: np.ndarray, feature_names: Sequence[str]) -> int:
    return sum(2 if feature_names[index] == "public_hard_risk_context" else 1
               for index in np.flatnonzero(weights))


def masked_probabilities(logits, masks):
    z = np.where(np.asarray(masks) > .5, np.asarray(logits, np.float64), -np.inf)
    return np.exp(z-logsumexp(z, axis=-1, keepdims=True))


@dataclass(frozen=True)
class _Decision:
    weight: tuple[float, ...]
    bias: float
    left: _Decision | None = None
    right: _Decision | None = None


@dataclass
class _Region:
    path: tuple[int, ...]
    tree: _Decision | None
    rows: np.ndarray
    depth: int
    legal: frozenset[int]


@dataclass
class _Leaf:
    region: _Region
    logits: np.ndarray
    optimizer: dict
    qualified: bool
    expected_mass: float
    kl_sum: float
    agreement: float


@dataclass
class _State:
    tree: _Decision | None
    leaves: list[_Leaf]
    regions: list[_Region]
    score: tuple[float, float, int]
    key: bytes


@dataclass
class _Proposal:
    tree: _Decision | None
    kind: str
    label: str
    gain: float
    key: bytes


class _BudgetExhausted(RuntimeError):
    pass


def _size(tree):
    if tree is None:
        return 1, 0
    left, ld = _size(tree.left)
    right, rd = _size(tree.right)
    return left+right, 1+max(ld, rd)


def _tree_key(tree):
    digest = sha256()

    def visit(node):
        if node is None:
            digest.update(b"L")
            return
        digest.update(b"N")
        digest.update(np.asarray(node.weight, dtype="<f4").tobytes())
        digest.update(np.asarray([node.bias], dtype="<f4").tobytes())
        visit(node.left)
        visit(node.right)

    visit(tree)
    return digest.digest()


def _replace(tree, path, replacement):
    if not path:
        return replacement
    if tree is None:
        raise RuntimeError("constrained tree replacement path is not reachable")
    if path[0] == 0:
        return _Decision(tree.weight, tree.bias,
                         _replace(tree.left, path[1:], replacement), tree.right)
    return _Decision(tree.weight, tree.bias, tree.left,
                     _replace(tree.right, path[1:], replacement))


def _improves(new, old):
    if new[0] > old[0]+_GAIN_TOLERANCE:
        return True
    if abs(new[0]-old[0]) > _GAIN_TOLERANCE:
        return False
    if new[1] > old[1]+_GAIN_TOLERANCE:
        return True
    return abs(new[1]-old[1]) <= _GAIN_TOLERANCE and new[2] > old[2]


def _export(tree, leaves, feature_count):
    weights, biases, left, right = [], [], [], []
    leaf_cursor = 0

    def visit(node):
        nonlocal leaf_cursor
        if node is None:
            ref = -1-leaf_cursor
            leaf_cursor += 1
            return ref
        ref = len(weights)
        weights.append(node.weight)
        biases.append(node.bias)
        left.append(0)
        right.append(0)
        left[ref] = visit(node.left)
        right[ref] = visit(node.right)
        return ref

    root = visit(tree)
    return {"root_ref": root,
            "split_weights": np.asarray(weights, np.float32).reshape(-1, feature_count),
            "split_bias": np.asarray(biases, np.float32),
            "left_refs": np.asarray(left, np.int64),
            "right_refs": np.asarray(right, np.int64),
            "leaf_count": leaf_cursor,
            "initial_leaf_logits": np.stack([leaf.logits for leaf in leaves])}


def fit_legal_structure(
        encoded: np.ndarray, masks: np.ndarray, target_probs: np.ndarray,
        episode_ids: Sequence[str], *, feature_names: Sequence[str],
        actor_role_sha256: str, actions: Sequence[str] =
        ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"),
        config: SearchConfig | Mapping | None = None) -> tuple[dict, dict]:
    """Return a standalone sparse topology and fit-only diagnostic report.

    The hard 20,000 budget counts actual leaf API calls (cache misses). Each
    API may enumerate up to five certified actions; actual solver calls are
    reported separately. Cache keys bind rows and legal sets, inside this
    immutable target/Actor invocation. A budget interruption never returns a
    partially optimized candidate.
    """
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
    fit_cache, quick_cache = {}, {}
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
                raise
            solves += int(optimizer.get("enumerated_action_solves", len(legal)))
            chosen = int(np.asarray(z).argmax())
            if (not optimizer.get("success")
                    or not float32_preference_proof(z, chosen, legal,
                                                   epsilon=LEAF_EPSILON)["passed"]):
                raise RuntimeError("constrained leaf returned an uncertified float32 preference")
            fit_cache[key] = (z, optimizer)
        if not len(rows):
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
        mass = float(np.sum(p[rows, chosen]*matching, dtype=np.float64)) if qualified else 0.
        return _Leaf(region, z, optimizer, qualified, mass,
                     float(np.sum(kl, dtype=np.float64)), agreement)

    def evaluate(tree):
        size, depth = _size(tree)
        if size > cfg.max_leaves or depth > cfg.max_depth:
            raise ValueError("constrained proposal exceeds tree capacity")
        parts = list(regions(tree))
        leaves = [fit_leaf(part) for part in parts if part.tree is None]
        score = (sum(leaf.expected_mass for leaf in leaves)/len(x),
                 -sum(leaf.kl_sum for leaf in leaves)/len(x), -size)
        return _State(tree, leaves, parts, score, _tree_key(tree))

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
        chosen.sort(key=rank)
        chosen = chosen[:cfg.proposals_per_round]
        chosen_keys = {candidate.key for candidate in chosen}
        remainder = sorted((candidate for candidate in offered.values()
                            if candidate.key not in chosen_keys), key=rank)
        chosen.extend(remainder[:cfg.proposals_per_round-len(chosen)])
        return chosen, len(offered)

    best, runs = None, []
    operation_counts = {kind: 0 for kind in PROPOSAL_QUOTAS}
    for start, initialization in enumerate(SEARCH_INITIALIZATIONS[:cfg.initializations]):
        before_calls, before_solves = calls, solves
        # Deterministic cumulative shares let later starts use earlier unused
        # calls, while never exceeding the one per-role API-call hard bound.
        current_call_limit = max(1, ((start+1)*cfg.max_leaf_optimization_calls)//cfg.initializations)
        state = evaluate(None)
        attempted, history = set(), []
        stop = "round_limit"
        for round_index in range(cfg.max_rounds):
            candidates, proposal_count = proposals(state, initialization, attempted)
            if not candidates:
                stop = "no_new_feasible_proposals"
                break
            improved = state
            accepted_operation = None
            evaluated, interrupted = 0, False
            for candidate in candidates:
                attempted.add(candidate.key)
                try:
                    fitted = evaluate(candidate.tree)
                except _BudgetExhausted:
                    interrupted = True
                    break
                except RuntimeError as error:
                    rejected_solver_candidates += 1
                    if len(failure_examples) < 20:
                        failure_examples.append({"initialization": start,
                            "round": round_index+1, "operation": candidate.kind,
                            "error": str(error)[:1200]})
                    continue
                evaluated += 1
                operation_counts[candidate.kind] += 1
                if (_improves(fitted.score, improved.score)
                        or (fitted.score == improved.score and fitted.key < improved.key
                            and _improves(fitted.score, state.score))):
                    improved, accepted_operation = fitted, candidate.kind
            changed = _improves(improved.score, state.score)
            history.append({"round": round_index+1, "proposals": proposal_count,
                "scheduled_proposals": len(candidates), "complete_evaluations": evaluated,
                "accepted": changed, "accepted_operation": accepted_operation,
                "previous_score": list(state.score), "score": list(improved.score),
                "leaf_optimization_calls_cumulative": calls,
                "budget_interrupted": interrupted})
            state = improved
            if interrupted:
                stop = "leaf_optimization_call_budget"
                break
            if not changed:
                stop = "no_strict_improvement_among_scheduled_proposals"
                break
        if best is None or _improves(state.score, best.score) or (
                state.score == best.score and state.key < best.key):
            best = state
        runs.append({"initialization": dict(initialization), "rounds": history,
            "stop_reason": stop, "leaf_optimization_calls": calls-before_calls,
            "actual_convex_solver_calls": solves-before_solves,
            "fit_score": list(state.score), "tree_sha256": state.key.hex()})
    assert best is not None
    if not (calls <= cfg.max_leaf_optimization_calls and 0 <= solves <= 5*calls):
        raise RuntimeError("constrained tree leaf/solver call accounting is invalid")
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
    report = {"topology_origin": "independent_fit_only_path_constrained_whole_tree_search",
        "method": "path_constrained_distribution_tree_v1",
        "target_provenance": "complete_frozen_nn_masked_distribution",
        "actor_role_sha256": actor_role_sha256, "fit_rows": len(x),
        "fit_episode_count": len(set(episode)),
        "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
        "search_budget": asdict(cfg), "production_configuration": cfg == PRODUCTION_SEARCH_CONFIG,
        "initial_structure": "same_single_path_certified_root_leaf_for_all_starts",
        "initialization_scope": "deterministic_fit_only_predicate_grids_not_independent_random_restarts",
        "search_runs": runs, "leaf_optimization_calls": calls,
        "leaf_optimization_calls_remaining": cfg.max_leaf_optimization_calls-calls,
        "actual_convex_solver_calls": solves, "leaf_cache_hits": cache_hits,
        "rejected_solver_candidates": rejected_solver_candidates,
        "solver_failure_examples": failure_examples,
        "operation_complete_evaluations": operation_counts,
        "proposal_quotas": dict(PROPOSAL_QUOTAS),
        "fit_expected_feedback_coverage": best.score[0], "fit_masked_kl": -best.score[1],
        "fit_whole_tree_raw_illegal_count": whole_illegal,
        "fit_whole_tree_path_uncertified_count": 0,
        "fit_qualified_leaf_coverage": sum(len(leaf.region.rows) for leaf in best.leaves if leaf.qualified)/len(x),
        "fit_leaf_reports": [{"leaf": index, "rows": len(leaf.region.rows),
            "depth": leaf.region.depth, "path_legal_actions": sorted(leaf.region.legal),
            "raw_action": int(leaf.logits.argmax()), "fit_qualified": leaf.qualified,
            "agreement": leaf.agreement, "masked_kl": leaf.kl_sum/len(leaf.region.rows) if len(leaf.region.rows) else None,
            "expected_feedback_mass": leaf.expected_mass, "optimizer": leaf.optimizer}
            for index, leaf in enumerate(best.leaves)],
        "fit_proxy_qualification": {"minimum_rows": FIT_MIN_ROWS,
            "minimum_episodes": FIT_MIN_EPISODES, "maximum_episode_fraction": FIT_MAX_EPISODE_FRACTION,
            "minimum_agreement": FIT_MIN_AGREEMENT, "maximum_kl": FIT_MAX_KL},
        "candidate_score": "fit_qualified_expected_feedback_then_all_frame_kl_then_leaf_count.v1",
        "candidate_ranking": "subtree_quick_gain_with_joint_guard_capacity.v1",
        "accepted_updates_strictly_monotonic": True,
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
