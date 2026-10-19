"""Fit-only v6.22 A32 and finite-template D32 program searches.

A32 versions the M2 engine without changing any legacy module's constants.
D32 is exact only over the completed frozen terminal tables of its declared
templates. Neither search is a global sparse-tree capacity certificate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import logging
import math
from time import monotonic
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from .warehouse_v619_leaf_fit import (
    LEAF_EPSILON, LEAF_MAXITER, LEAF_REGULARIZATION, LEAF_TOLERANCE, ConstrainedLeafFitError,
    float32_preference_proof,
)
from .warehouse_v619_tree import (
    FIT_MIN_ROWS, FIT_MIN_EPISODES, FIT_MAX_EPISODE_FRACTION,
    FIT_MIN_AGREEMENT, FIT_MAX_KL, SEARCH_INITIALIZATIONS, PROPOSAL_QUOTAS,
    source_feature_cost, masked_probabilities, _Decision, _Region, _Leaf,
    _State, _Proposal, _BudgetExhausted, _size, _tree_key, _replace, _export,
)
from .warehouse_program_search import _BeamEntry, _choose_beam, _improves, _rank_state
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, HARD_SPLIT_TIE_EPSILON,
    certified_leaf_actions, explicit_mask_branches,
    leaf_indices as route_leaf_indices, split_scores,
)

A32 = "m2_depth6_leaves32_v622"
D32 = "four_frozen_templates_leaf_budget_dp_v622"
ARMS = (A32, D32)
ROBUST_METHOD = "episode_robust_explanation_tree_v1"
QUEUED_METHOD = "queued_path_constrained_tree_v1"
METHODS = (ROBUST_METHOD,)
MAX_SOURCE_FEATURES = 3
ROBUST_MIN_AGREEMENT, ROBUST_MAX_KL, STABILITY_GROUPS = .97, .20, 3
FINAL_REFIT_RESERVED_CALLS = 32
A32_END_TO_END_LEAF_API_CEILING = 20000
D32_GENERATION_LEAF_API_CEILING = 20000
D32_ADDITIONAL_LEAF_API_CEILING = 540
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchConfig:
    initializations: int = 8
    max_rounds: int = 60
    max_leaf_optimization_calls: int = 19968
    proposals_per_round: int = 48
    beam_width: int = 4
    max_nonimproving_steps: int = 2
    axis_features: int = 12
    semantic_candidates_per_node: int = 6
    max_depth: int = 6
    max_leaves: int = 32
    wall_time_seconds: float = 5400.
    template_count: int = 4
    template_max_leaf_optimization_calls: int = 20000
    table_max_leaf_optimization_calls: int = 508
    template_candidates_per_growth: int = 16

    def __post_init__(self):
        limits = {"initializations": 8, "max_rounds": 60,
            "max_leaf_optimization_calls": 19968, "proposals_per_round": 48,
            "beam_width": 4, "max_nonimproving_steps": 2, "axis_features": 12,
            "semantic_candidates_per_node": 6, "max_depth": 6, "max_leaves": 32,
            "template_count": 4, "template_max_leaf_optimization_calls": 20000,
            "table_max_leaf_optimization_calls": 508, "template_candidates_per_growth": 16}
        for key, maximum in limits.items():
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= maximum:
                raise ValueError("invalid v6.22 search limit: " + key)
        if (type(self.wall_time_seconds) not in (int, float) or not math.isfinite(self.wall_time_seconds)
                or not 0 < self.wall_time_seconds <= 5400):
            raise ValueError("v6.22 role/fold deadline must be positive and at most 90 minutes")


PRODUCTION_SEARCH_CONFIG = SearchConfig()


class SearchIncompleteError(RuntimeError):
    """No completely fitted and certified incumbent exists; never invent one."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class _DeadlineExhausted(_BudgetExhausted):
    pass


class _Clock:
    def __init__(self, seconds):
        self.start, self.limit = monotonic(), float(seconds)
        self.timed_out = False

    def check(self):
        if self.timed_out or monotonic()-self.start >= self.limit:
            self.timed_out = True
            raise _DeadlineExhausted("role/fold monotonic wall-clock budget exhausted")

    def report(self):
        elapsed = monotonic()-self.start
        return {"wall_time_limit_seconds": self.limit, "elapsed_seconds": elapsed,
            "wall_time_budget_exhausted": self.timed_out or elapsed >= self.limit,
            "deadline_enforcement": "monotonic_checks_before_solver_each_SLSQP_objective_callback_and_between_proposals",
            "inflight_leaf_API_cannot_be_preempted": False,
            "inflight_solver_maxiter": LEAF_MAXITER}


def _configuration(config):
    return PRODUCTION_SEARCH_CONFIG if config is None else config if isinstance(config, SearchConfig) else SearchConfig(**dict(config))


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Mapping): return {str(k): _plain(x) for k, x in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(x) for x in value]
    if isinstance(value, np.generic): return value.item()
    return value


def _digest_plain(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _input_identity(x, m, p, episodes, names, actor):
    return {"actor_role_sha256": actor, "feature_names": list(names),
        "fit_encoded_sha256": sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest(),
        "fit_masks_sha256": sha256(np.ascontiguousarray(m, dtype="<f4").tobytes()).hexdigest(),
        "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
        "fit_episode_ids_sha256": _digest_plain(list(map(str, episodes)))}


def _validate(encoded, masks, targets, episode_ids, feature_names, actor_role_sha256):
    x, m, p = (np.asarray(a, np.float32) for a in (encoded, masks, targets))
    names = tuple(feature_names)
    episodes = np.asarray([str(e) for e in episode_ids])
    try:
        indices = tuple(names.index("legal_"+a.lower()) for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))
    except ValueError as error:
        raise ValueError("explicit five-action physical mask feature schema required") from error
    if (len(names) != len(set(names)) or any(not isinstance(n, str) for n in names)
            or x.ndim != 2 or len(x) < 2 or x.shape[1] != len(names)
            or m.shape != (len(x), 5) or p.shape != m.shape or episodes.shape != (len(x),)
            or any(not e for e in episodes) or any(not np.isfinite(a).all() for a in (x, m, p))
            or np.any((m != 0) & (m != 1)) or np.any(m[:, 4] != 1)
            or not np.array_equal(x[:, indices], m) or np.any(p < 0) or np.any(p[m == 0] != 0)
            or not np.allclose(p.sum(-1), 1., atol=1e-6, rtol=0)
            or not isinstance(actor_role_sha256, str) or len(actor_role_sha256) != 64
            or any(c not in "0123456789abcdef" for c in actor_role_sha256)):
        raise ValueError("fit requires complete physically masked five-action NN distributions and fit episode identities")
    return x, m, p, episodes, names, indices


def fit_structure(arm, encoded, masks, targets, episode_ids, *, feature_names,
                  actor_role_sha256, config=None):
    cfg = _configuration(config)
    clock = _Clock(cfg.wall_time_seconds)
    x, m, p, episodes, names, _ = _validate(encoded, masks, targets, episode_ids, feature_names, actor_role_sha256)
    if arm in ("A", "A32", A32):
        structure, report = _fit_m2(x, m, p, episodes, feature_names=names,
            actor_role_sha256=actor_role_sha256, method=ROBUST_METHOD, config=cfg, clock=clock)
        report.update({"arm": "A32", "method": A32, "legacy_algorithm": ROBUST_METHOD,
            "final_refit_reserved_calls": 32, "end_to_end_leaf_api_ceiling": 20000,
            "maximum_end_to_end_calls": report["leaf_optimization_calls"]+32,
            "global_tree_optimality_gap": None})
    elif arm in ("D", "D32", D32):
        structure, report = _fit_d32(x, m, p, episodes, names, actor_role_sha256, cfg, clock)
    else:
        raise ValueError("unknown v6.22 search arm")
    structure["feature_dim"] = len(names)
    report.update({"input_identity": _input_identity(x, m, p, episodes, names, actor_role_sha256),
        "search_budget": asdict(cfg), "complete_incumbent": True,
        "search_uses_fit_only": True, "calibration_eligibility_validation_used": False,
        "execution_action_override": False, **clock.report()})
    report["structure_provenance_sha256"] = _digest_plain(structure)
    report = _plain(report)
    report["report_sha256_scope"] = "canonical_JSON_without_report_sha256"
    report["report_sha256"] = _digest_plain(report)
    return structure, report


def _add_scores(left, right):
    return tuple(float(a)+float(b) for a, b in zip(left, right, strict=True))


class _FitContext:
    """Immutable fit-side rows with phase-specific, audited solver ledgers."""
    def __init__(self, x, masks, p, episodes, names, indices, clock, max_calls):
        self.x, self.m, self.p, self.episodes, self.names = x, masks, p, episodes, names
        self.indices, self.clock, self.max_calls = indices, clock, max_calls
        self.calls = self.solves = self.cache_hits = 0
        self.events, self.cache = [], {}
        self.current_limit = max_calls
        self.top = p.argmax(-1)
        order = sorted(set(episodes), key=lambda value: sha256(value.encode()).digest())
        groups = {episode: i % 3 for i, episode in enumerate(order)}
        self.group = np.asarray([groups[e] for e in episodes])

    def regions(self, tree, rows=None, path=(), legal=frozenset({4})):
        rows = np.arange(len(self.x)) if rows is None else rows
        region = _Region(path, tree, rows, len(path), legal)
        yield region
        if tree is not None:
            w = np.asarray(tree.weight, np.float32)
            right = split_scores(self.x[rows], w[None], np.asarray([tree.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            ll, rl = explicit_mask_branches(w, tree.bias, self.indices)
            yield from self.regions(tree.left, rows[~right], path+(0,), legal | ll)
            yield from self.regions(tree.right, rows[right], path+(1,), legal | rl)

    def fit(self, region, *, fresh=False):
        self.clock.check()
        rows, legal = region.rows, tuple(sorted(region.legal))
        key = (sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest(), legal)
        if not fresh and key in self.cache:
            self.cache_hits += 1
            logits, optimizer = self.cache[key]
        else:
            self.clock.check()
            if self.calls >= min(self.current_limit, self.max_calls):
                raise _BudgetExhausted("phase leaf API budget exhausted")
            self.calls += 1
            event = {"call": self.calls, "rows_sha256": key[0], "rows": len(rows),
                "path_legal_actions": list(legal), "cache_miss": True, "success": False}
            self.events.append(event)
            try:
                logits, optimizer = fit_constrained_leaf_distribution(self.p[rows], self.m[rows], legal,
                    epsilon=LEAF_EPSILON, regularization=LEAF_REGULARIZATION,
                    maxiter=LEAF_MAXITER, tolerance=LEAF_TOLERANCE, clock=self.clock)
            except RuntimeError as error:
                event.update(actual_convex_solver_calls=int(getattr(error, "enumerated_action_solves", 0)),
                             error=str(error)[:500])
                self.solves += event["actual_convex_solver_calls"]
                raise
            event["actual_convex_solver_calls"] = int(optimizer.get("enumerated_action_solves", len(legal)))
            self.solves += event["actual_convex_solver_calls"]
            if (not optimizer.get("success") or not float32_preference_proof(
                    logits, int(np.asarray(logits).argmax()), legal)["passed"]):
                event["error"] = "uncertified float32 preference"
                raise RuntimeError(event["error"])
            logits = np.asarray(logits, np.float32).copy()
            logits.setflags(write=False)
            event["success"] = True
            if not fresh:
                self.cache[key] = logits, optimizer
        chosen = int(logits.argmax())
        if len(rows) and np.any(self.m[rows, chosen] != 1):
            raise RuntimeError("path proof disagrees with actual physical masks")
        z = np.where(self.m[rows] > .5, logits.astype(np.float64), -np.inf)
        logq = z-logsumexp(z, axis=-1, keepdims=True)
        p = self.p[rows].astype(np.float64)
        kl = np.sum(p*(np.log(np.maximum(p, 1e-300))-np.where(self.m[rows] > .5, logq, 0.)), axis=-1)
        matches = self.top[rows] == chosen
        _, counts = np.unique(self.episodes[rows], return_counts=True)
        agreement = float(matches.mean()) if len(rows) else 0.
        qualified = bool(len(rows) >= FIT_MIN_ROWS and len(counts) >= FIT_MIN_EPISODES
            and counts.max()/len(rows) <= FIT_MAX_EPISODE_FRACTION
            and agreement >= FIT_MIN_AGREEMENT and kl.mean() <= FIT_MAX_KL)
        groups = []
        for group in range(3):
            take = self.group[rows] == group
            groups.append({"group": group, "rows": int(take.sum()),
                "episodes": len(set(self.episodes[rows[take]])),
                "agreement": float(matches[take].mean()) if take.any() else None,
                "masked_kl": float(kl[take].mean()) if take.any() else None})
        robust = bool(qualified and agreement >= .97 and kl.mean() <= .20
            and all(g["episodes"] >= 2 and g["agreement"] >= .97 and g["masked_kl"] <= .20 for g in groups))
        expected = float(np.sum(p[:, chosen]*matches)) if qualified else 0.
        leaf = _Leaf(region, logits, dict(optimizer), qualified, expected, float(kl.sum()), agreement)
        leaf.robust = {"qualified": robust, "groups": groups}
        return leaf

    def report(self):
        return {"leaf_optimization_calls": self.calls, "actual_convex_solver_calls": self.solves,
            "leaf_cache_hits": self.cache_hits, "leaf_api_attempts": self.events,
            "failed_leaf_api_attempts": sum(not e["success"] for e in self.events)}


def _leaf_score(leaf):
    return (float(len(leaf.region.rows)) if leaf.robust["qualified"] else 0.,
            leaf.expected_mass, -leaf.kl_sum, -1.)


def _leaves_score(leaves):
    score = (0., 0., 0., 0.)
    for leaf in leaves:
        score = _add_scores(score, _leaf_score(leaf))
    return score


def _template_predicates(ctx, region, recipe, cfg):
    """Finite M2-style full-distribution candidate menu, using fit rows only."""
    ctx.clock.check()
    rows = region.rows
    if not len(rows):
        return []
    result, seen = [], set()
    def add(w, threshold, label, guard=False):
        w = np.asarray(w, np.float32)
        b = float(np.float32(-threshold))
        if source_feature_cost(w, ctx.names) > 3:
            return
        right = split_scores(ctx.x[rows], w[None], np.asarray([b], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
        if not guard and min(int(right.sum()), int((~right).sum())) < 20:
            return
        key = (w.tobytes(), np.float32(b).tobytes())
        if key in seen:
            return
        seen.add(key)
        # Ranking uses all five supplied probabilities, not teacher hard labels.
        means = [ctx.p[rows[take]].mean(0) if take.any() else np.zeros(5) for take in (~right, right)]
        difference = float(np.square(means[0]-means[1]).sum())
        balance = min(int(right.sum()), int((~right).sum()))
        bonus = float(np.sum(ctx.p[rows, ctx.indices.index(int(np.flatnonzero(w)[0]))])) if guard else 0.
        result.append((w, b, label, guard, difference*balance+bonus))
    for action, index in enumerate(ctx.indices):
        if action != 4 and action not in region.legal and np.any(ctx.top[rows] == action):
            w = np.zeros(len(ctx.names), np.float32); w[index] = 1
            add(w, .5, "legal:"+str(action), True)
    nonmask = np.asarray([i for i in range(len(ctx.names)) if i not in ctx.indices], int)
    if len(nonmask) and len(rows) >= 40:
        values = ctx.x[np.ix_(rows, nonmask)].astype(np.float64)
        centered = values-values.mean(0)
        response = ctx.p[rows].astype(float)
        if recipe["response"] == "log_probability":
            response = np.log(np.maximum(response, 1e-5))
        response -= response.mean(0)
        covariance = np.einsum("nf,na->fa", centered, response, optimize=False)
        covariance /= len(rows)*np.maximum(centered.std(0), 1e-8)[:, None]
        order = np.argsort(-np.linalg.norm(covariance, axis=1), kind="stable")[:cfg.axis_features]
        basis = []
        for index in nonmask[order]:
            w = np.zeros(len(ctx.names), np.float32); w[index] = 1
            basis.append((w, "axis:"+ctx.names[index]))
        for action in range(5):
            direction = covariance[:, action].copy()
            if recipe["contrast"]:
                direction -= covariance[:, (action+recipe["contrast"]) % 5]
            support, cost = [], 0
            for index in np.argsort(-np.abs(direction), kind="stable"):
                price = 2 if ctx.names[nonmask[index]] == "public_hard_risk_context" else 1
                if cost+price <= 3 and abs(direction[index]) > 1e-10:
                    support.append(index); cost += price
            if len(support) >= 2:
                w = np.zeros(len(ctx.names), np.float32); w[nonmask[support]] = direction[support]
                w /= max(float(np.linalg.norm(w)), 1e-8)
                basis.append((w, "sparse:"+str(action)))
        for w, label in basis:
            scores = split_scores(ctx.x[rows], w[None], np.zeros(1, np.float32))[:, 0]
            for quantile in recipe["quantiles"]:
                add(w, float(np.quantile(scores, quantile)), label+":"+str(quantile))
    result.sort(key=lambda value: (-value[4], not value[3], value[2], value[0].tobytes(), value[1]))
    return result[:cfg.template_candidates_per_growth]


def _grow_template(ctx, recipe, cfg, observer):
    tree = None
    root = ctx.fit(next(ctx.regions(tree)))
    leaves, history, blocked = [root], [], set()
    observer(tree, leaves)
    termination = "maximal_under_fit_predicate_menu"
    while True:
        if _size(tree)[0] >= 64:
            termination = "maximum_template_leaves"; break
        frontier = [r for r in ctx.regions(tree) if r.tree is None and len(r.path) < cfg.max_depth
                    and len(r.rows) and r.path not in blocked]
        frontier.sort(key=lambda r: (-len(r.rows), r.path))
        if not frontier:
            break
        region = frontier[0]
        try:
            menu = _template_predicates(ctx, region, recipe, cfg)
        except _DeadlineExhausted:
            termination = "wall_time_budget"; break
        best = None
        for w, b, label, _, _ in menu:
            replacement = _Decision(tuple(w), b)
            candidate = _replace(tree, region.path, replacement)
            candidate_regions = [r for r in ctx.regions(candidate) if r.tree is None]
            try:
                candidate_leaves = [ctx.fit(r) for r in candidate_regions]
            except _BudgetExhausted:
                termination = "wall_time_budget" if ctx.clock.timed_out else "template_generation_leaf_call_budget"
                return tree, leaves, history, termination, False
            except RuntimeError:
                continue
            score, key = _leaves_score(candidate_leaves), _tree_key(candidate)
            if best is None or score > best[0] or score == best[0] and key < best[1]:
                best = score, key, candidate, candidate_leaves, label
        if best is None:
            blocked.add(region.path)
            continue
        _, _, tree, leaves, label = best
        history.append({"growth": len(history)+1, "path": list(region.path), "predicate": label,
            "leaf_count": _size(tree)[0], "depth": _size(tree)[1], "fit_score": list(_leaves_score(leaves)),
            "tree_sha256": _tree_key(tree).hex(), "generation_calls_cumulative": ctx.calls})
        observer(tree, leaves)
    return tree, leaves, history, termination, True


def _freeze_table(ctx, tree):
    records = []
    for region in ctx.regions(tree):
        leaf = ctx.fit(region, fresh=True)
        records.append({"node_id": len(records), "path": list(region.path), "rows": len(region.rows),
            "rows_sha256": sha256(np.ascontiguousarray(region.rows, dtype="<i8").tobytes()).hexdigest(),
            "path_legal_actions": sorted(region.legal), "logits": leaf.logits.tolist(),
            "score": list(_leaf_score(leaf)), "optimizer": leaf.optimizer,
            "fit_qualified": leaf.qualified, "fit_stable_explanation_qualified": leaf.robust["qualified"],
            "episode_stability_groups": leaf.robust["groups"], "agreement": leaf.agreement})
    return records


def _prune_frozen_tree(tree, table, max_leaves, clock=None):
    by_path = {tuple(row["path"]): row for row in table}
    choices = 0
    def solve(node, path):
        nonlocal choices
        if clock is not None:
            clock.check()
        record = by_path[path]
        states = {1: (tuple(record["score"]), None, (path,))}
        if node is not None:
            left, right = solve(node.left, path+(0,)), solve(node.right, path+(1,))
            for lk, lv in left.items():
                if clock is not None:
                    clock.check()
                for rk, rv in right.items():
                    if lk+rk > max_leaves:
                        continue
                    choices += 1
                    score = _add_scores(lv[0], rv[0])
                    candidate = _Decision(node.weight, node.bias, lv[1], rv[1])
                    value = score, candidate, lv[2]+rv[2]
                    old = states.get(lk+rk)
                    if old is None or score > old[0] or score == old[0] and _tree_key(candidate) < _tree_key(old[1]):
                        states[lk+rk] = value
        return states
    states = solve(tree, ())
    winner = min(states.values(), key=lambda value: (tuple(-v for v in value[0]), _tree_key(value[1])))
    return winner, {"objective_vector": list(winner[0]), "reachable_leaf_budgets": sorted(states),
        "complete_DP": True, "combinations_evaluated": choices,
        "selected_terminal_paths": [list(p) for p in winner[2]],
        "allowed_moves": "terminal_at_node_or_keep_both_children_no_child_promotion"}


def _export_table_winner(ctx, winner, table):
    by_path = {tuple(row["path"]): row for row in table}
    selected = []
    for path in winner[2]:
        record = by_path[path]
        region = next(r for r in ctx.regions(winner[1]) if r.path == path)
        z = np.asarray(record["logits"], np.float32)
        leaf = _Leaf(region, z, record["optimizer"], record["fit_qualified"],
            float(record["score"][1]), -float(record["score"][2]), record["agreement"])
        leaf.robust = {"qualified": record["fit_stable_explanation_qualified"],
                       "groups": record["episode_stability_groups"]}
        leaf.table_node_id = record["node_id"]
        leaf.table_path = list(path)
        selected.append(leaf)
    return selected


def _fit_d32(x, masks, p, episodes, names, actor, cfg, clock):
    indices = tuple(names.index("legal_"+a.lower()) for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))
    generation = _FitContext(x, masks, p, episodes, names, indices, clock, cfg.template_max_leaf_optimization_calls)
    terminal = _FitContext(x, masks, p, episodes, names, indices, clock, cfg.table_max_leaf_optimization_calls)
    incumbent, generation_incumbent = None, None
    def observe(tree, leaves):
        nonlocal generation_incumbent
        if _size(tree)[0] <= cfg.max_leaves:
            score, key = _leaves_score(leaves), _tree_key(tree)
            if (generation_incumbent is None or score > generation_incumbent[0]
                    or score == generation_incumbent[0] and key < generation_incumbent[1]):
                generation_incumbent = score, key, tree, leaves
    templates, completed_tables, generations = [], [], []
    for index in range(cfg.template_count):
        recipe = SEARCH_INITIALIZATIONS[(2*index) % len(SEARCH_INITIALIZATIONS)]
        generation.current_limit = max(1, ((index+1)*cfg.template_max_leaf_optimization_calls)//cfg.template_count)
        before = generation.calls
        try:
            tree, leaves, history, stop, maximal = _grow_template(generation, recipe, cfg, observe)
        except (_BudgetExhausted, RuntimeError) as error:
            generations.append({"template": index, "recipe": dict(recipe), "completed_structure": False,
                "generation_calls": generation.calls-before, "failure": str(error),
                "stop_reason": "wall_time_budget" if clock.timed_out else "initial_leaf_failure_or_budget"})
            if clock.timed_out:
                break
            continue
        structure = _export(tree, leaves, len(names)); structure["feature_dim"] = len(names)
        templates.append(structure)
        generation_report = {"template": index, "recipe": dict(recipe), "completed_structure": True,
            "generation_calls": generation.calls-before, "history": history, "stop_reason": stop,
            "maximal_under_declared_menu": maximal, "structure": structure,
            "structure_sha256": _digest_plain(structure), "leaf_count": _size(tree)[0], "depth": _size(tree)[1]}
        generations.append(generation_report)
        try:
            records = _freeze_table(terminal, tree)
        except (_BudgetExhausted, RuntimeError) as error:
            generation_report["frozen_table_completed"] = False
            generation_report["frozen_table_failure"] = str(error)
            if clock.timed_out:
                break
            continue
        generation_report["frozen_table_completed"] = True
        try:
            winner, certificate = _prune_frozen_tree(tree, records, cfg.max_leaves, clock=clock)
        except _DeadlineExhausted:
            generation_report.update(dp_completed=False,
                unscored_frozen_terminal_table_sha256=_digest_plain(records),
                stop_reason="wall_time_budget_during_DP")
            break
        selected = _export_table_winner(terminal, winner, records)
        completed_tables.append({"template": index, "template_structure_sha256": _digest_plain(structure),
            "records": records, "frozen_terminal_table_sha256": _digest_plain(records), "certificate": certificate})
        score, key = winner[0], _tree_key(winner[1])
        if incumbent is None or score > incumbent[0] or score == incumbent[0] and key < incumbent[1]:
            incumbent = score, key, winner[1], selected, index
        generation_report["frozen_table_completed"] = True
        generation_report["dp_completed"] = True
    if incumbent is None:
        raise SearchIncompleteError("D32 did not complete a certified incumbent", {
            "complete_incumbent": False, "generation": generation.report(), "terminal_table": terminal.report(),
            "complete_generation_incumbent_available_but_not_a_DP_certificate": generation_incumbent is not None,
            "generation_incumbent_score": list(generation_incumbent[0]) if generation_incumbent is not None else None,
            **clock.report()})
    score, _, tree, leaves, selected_template = incumbent
    structure = _export(tree, leaves, len(names)); structure["feature_dim"] = len(names)
    _verify_export(structure, x, masks, names, cfg.max_depth, cfg.max_leaves)
    if generation.calls > 20000 or terminal.calls > 508 or terminal.calls+32 > 540:
        raise RuntimeError("D32 phase budget accounting exceeds its declared ceiling")
    selected_table = next(t for t in completed_tables if t["template"] == selected_template)
    selected_table_sha = selected_table["frozen_terminal_table_sha256"]
    selected_paths = [leaf.table_path for leaf in leaves]
    selected_logits_sha = sha256(np.ascontiguousarray(structure["initial_leaf_logits"], dtype="<f4").tobytes()).hexdigest()
    structure["frozen_table_certificate"] = {"selected_table_sha256": selected_table_sha,
        "selected_terminal_paths": selected_paths, "selected_table_node_ids": [leaf.table_node_id for leaf in leaves],
        "selected_leaf_logits_sha256": selected_logits_sha, "restricted_table_gap": 0.,
        "global_function_class_optimality_gap": None}
    report = {"arm": "D32", "method": D32, "actor_role_sha256": actor,
        "target_provenance": "complete_frozen_nn_masked_distribution", "fit_rows": len(x),
        "fit_episode_count": len(set(episodes)), "generation": generation.report(),
        "terminal_table": terminal.report(), "template_generation_reports": generations,
        "frozen_terminal_tables": completed_tables, "templates_requested": cfg.template_count,
        "complete_templates": len(templates), "complete_frozen_tables": len(completed_tables),
        "unique_template_structures": len({_digest_plain(t) for t in templates}),
        "selected_template": selected_template, "selected_fit_score": list(score),
        "leaf_optimization_calls": generation.calls+terminal.calls,
        "actual_convex_solver_calls": generation.solves+terminal.solves,
        "template_generation_leaf_optimization_calls": generation.calls,
        "frozen_table_leaf_optimization_calls": terminal.calls,
        "final_refit_reserved_calls": 0, "unused_final_refit_allowance": 32,
        "final_leaf_outputs_frozen_from_selected_table": True,
        "maximum_end_to_end_calls": generation.calls+terminal.calls,
        "end_to_end_leaf_api_ceiling": 20540, "generation_leaf_api_ceiling": 20000,
        "additional_table_plus_final_leaf_api_ceiling": 540,
        "global_search_rounds": 0, "search_runs": [],
        "fit_stable_explanation_coverage": score[0]/len(x), "fit_expected_feedback_coverage": score[1]/len(x),
        "fit_masked_kl": -score[2]/len(x), "fit_whole_tree_raw_illegal_count": 0,
        "fit_whole_tree_path_uncertified_count": 0,
        "certificate": {"scope": "only_declared_completed_templates_and_frozen_terminal_outputs_before_final_refit_or_calibration",
            "selected_incumbent_has_completed_table_certificate": True,
            "all_requested_templates_completed_and_enumerated": len(completed_tables) == cfg.template_count,
            "completed_template_family_combination_gap": 0., "restricted_table_gap": 0.,
            "restricted_objective_lower_bound_vector": list(score),
            "restricted_objective_upper_bound_vector": list(score),
            "selected_table_sha256": selected_table_sha, "selected_terminal_paths": selected_paths,
            "selected_leaf_logits_sha256": selected_logits_sha,
            "global_function_class_optimality_gap": None,
            "global_sparse_tree_lower_bound": None, "global_tree_optimality_gap": None,
            "continuous_leaf_output_optimality_claimed": False,
            "final_refit_and_calibrated_program_covered_by_DP_certificate": False},
        "topology_uses_fit_only": True, "calibration_eligibility_validation_used": False,
        "fit_only_not_independent_certificate": True, "guard_capacity_included_in_search": True,
        "maximum_source_features_per_node": max((source_feature_cost(w, names) for w in structure["split_weights"]), default=0),
        "hard_routing_version": HARD_SPLIT_ROUTING_VERSION,
        "fit_leaf_reports": [{"leaf": i, "rows": len(leaf.region.rows), "depth": leaf.region.depth,
            "path_legal_actions": sorted(leaf.region.legal), "raw_action": int(leaf.logits.argmax()),
            "fit_qualified": leaf.qualified, "fit_stable_explanation_qualified": leaf.robust["qualified"],
            "table_node_id": leaf.table_node_id, "selected_terminal_path": leaf.table_path,
            "agreement": leaf.agreement, "expected_feedback_mass": leaf.expected_mass,
            "masked_kl": leaf.kl_sum/len(leaf.region.rows) if len(leaf.region.rows) else None,
            "optimizer": leaf.optimizer} for i, leaf in enumerate(leaves)]}
    structure["fit_leaf_reports"] = report["fit_leaf_reports"]
    structure["fit_target_sha256"] = sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest()
    report["fit_target_sha256"] = structure["fit_target_sha256"]
    return structure, report


def _verify_export(structure, x, masks, names, depth, leaves):
    w = np.asarray(structure["split_weights"], np.float32)
    if structure["leaf_count"] > leaves or any(source_feature_cost(row, names) > 3 for row in w):
        raise RuntimeError("export exceeds v6.22 charged capacity")
    proofs = certified_leaf_actions(w, structure["split_bias"], structure["left_refs"], structure["right_refs"],
        len(w), structure["leaf_count"], tuple(names.index("legal_"+a.lower())
        for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")), explicit=True, wait_action=4)
    logits = structure["initial_leaf_logits"]
    for z, proof in zip(logits, proofs, strict=True):
        if not float32_preference_proof(z, int(z.argmax()), proof)["passed"]:
            raise RuntimeError("export changed a terminal leaf path legality certificate")
    route = route_leaf_indices(x, w, structure["split_bias"], structure["left_refs"], structure["right_refs"], len(w), depth)
    actions = logits[route].argmax(-1)
    if np.any(masks[np.arange(len(x)), actions] != 1):
        raise RuntimeError("export contains an illegal raw fit action")


def _fit_m2(
        encoded: np.ndarray, masks: np.ndarray, target_probs: np.ndarray,
        episode_ids: Sequence[str], *, feature_names: Sequence[str],
        actor_role_sha256: str, actions: Sequence[str] =
        ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"),
        method: str, config: SearchConfig | Mapping | None = None, clock=None) -> tuple[dict, dict]:
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
    leaf_api_attempts = []
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
        clock.check()
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
            event = {"call": calls, "rows_sha256": key[0].hex(), "rows": len(rows),
                "path_legal_actions": list(key[1]), "cache_miss": True, "success": False}
            leaf_api_attempts.append(event)
            try:
                z, optimizer = fit_constrained_leaf_distribution(
                    p[rows], m[rows], tuple(sorted(legal)), epsilon=LEAF_EPSILON,
                    regularization=LEAF_REGULARIZATION, maxiter=LEAF_MAXITER,
                    tolerance=LEAF_TOLERANCE, clock=clock)
            except RuntimeError as error:
                event["actual_convex_solver_calls"] = int(getattr(error, "enumerated_action_solves", 0))
                event["error"] = str(error)[:500]
                solves += event["actual_convex_solver_calls"]
                # Failed and interrupted solver attempts consume the same API
                # budget as successful attempts.  A failed tree is never used.
                raise
            event["actual_convex_solver_calls"] = int(optimizer.get("enumerated_action_solves", len(legal)))
            solves += event["actual_convex_solver_calls"]
            chosen = int(np.asarray(z).argmax())
            if (not optimizer.get("success")
                    or not float32_preference_proof(z, chosen, legal,
                                                   epsilon=LEAF_EPSILON)["passed"]):
                event["error"] = "uncertified float32 preference"
                raise RuntimeError("constrained leaf returned an uncertified float32 preference")
            event["success"] = True
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
        clock.check()
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
        clock.check()
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

    try:
        safe_root = evaluate(None)
    except (RuntimeError, _BudgetExhausted) as error:
        raise SearchIncompleteError("A32 did not complete its first certified root", {
            "leaf_optimization_calls": calls, "actual_convex_solver_calls": solves,
            "leaf_api_attempts": leaf_api_attempts, "complete_incumbent": False,
            "failure": str(error), **clock.report()}) from error
    bootstrap_calls, bootstrap_solves = calls, solves
    initial_topologies_materialized = True
    try:
        starts = initial_trees()
    except _DeadlineExhausted:
        starts = [None]*cfg.initializations
        initial_topologies_materialized = False
    best, runs = safe_root, []
    global_rounds = 0
    operation_counts = {kind: 0 for kind in PROPOSAL_QUOTAS}
    for start, initialization in enumerate(SEARCH_INITIALIZATIONS[:cfg.initializations]):
        before_calls, before_solves = calls, solves
        try:
            clock.check()
        except _DeadlineExhausted:
            runs.append({"initialization": dict(initialization), "rounds": [],
                "stop_reason": "wall_time_budget", "initialization_started": False,
                "leaf_optimization_calls": 0, "actual_convex_solver_calls": 0,
                "remaining_queue": 0})
            continue
        # Deterministic cumulative shares let later starts use earlier unused
        # calls, while never exceeding the one per-role API-call hard bound.
        current_call_limit = max(1, ((start+1)*cfg.max_leaf_optimization_calls)//cfg.initializations)
        # Both limits are cumulative shares. An early stop passes its unused
        # rounds to later starts; even when rounds < starts, the zero-share
        # starts cannot spend a round reserved for a later initialization.
        current_round_limit = ((start+1)*cfg.max_rounds)//cfg.initializations
        before_rounds = global_rounds
        available_rounds = current_round_limit-global_rounds
        root_fallback = safe_root
        initial_failure = None
        try:
            state = evaluate(starts[start])
            initial_interrupted = False
        except _BudgetExhausted:
            # A diagnostic budget may not finish the requested initial tree.
            # Preserve the completely evaluated safe root, and account for
            # every leaf call already spent on the rejected partial tree.
            state, initial_interrupted = root_fallback, True
            initial_failure = "wall_time_budget" if clock.timed_out else "leaf_optimization_call_budget"
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
            try:
                clock.check()
            except _DeadlineExhausted:
                stop = "wall_time_budget"
                break
            offered = {}
            parent_entries = []
            for entry in beam:
                if entry.state.key in expanded:
                    continue
                expanded.add(entry.state.key)
                parent_entries.append(entry)
                try:
                    candidates, _ = proposals(entry.state, initialization, attempted)
                except _DeadlineExhausted:
                    break
                for candidate in candidates:
                    old = offered.get(candidate.key)
                    # Use the best predecessor for a duplicate topology so its
                    # transition debt is conservative and deterministic.
                    value = (candidate, entry)
                    if old is None or (_rank_state(entry.state), entry.nonimproving_steps) < (
                            _rank_state(old[1].state), old[1].nonimproving_steps):
                        offered[candidate.key] = value
            if clock.timed_out:
                global_rounds += 1
                history.append({"round": round_index+1, "global_round": global_rounds,
                    "budget_interrupted": True, "stop_phase": "proposal_generation",
                    "complete_evaluations": 0, "failed_evaluations": 0,
                    "leaf_optimization_calls_cumulative": calls})
                stop = "wall_time_budget"
                break
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
                stop = "wall_time_budget" if clock.timed_out else "leaf_optimization_call_budget"
                break
            if not any(entry.state.key not in expanded for entry in beam):
                stop = "complete_queue_no_unexpanded_beam_state"
                break
        if best is None or _improves(start_best.score, best.score) or (
                start_best.score == best.score and start_best.key < best.key):
            best = start_best
        runs.append({"initialization": dict(initialization), "initialization_started": True, "rounds": history,
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
            and 0 <= global_rounds <= cfg.max_rounds
            and calls == bootstrap_calls+sum(run["leaf_optimization_calls"] for run in runs)
            and solves == bootstrap_solves+sum(run["actual_convex_solver_calls"] for run in runs)):
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
        "bootstrap_leaf_optimization_calls": bootstrap_calls,
        "bootstrap_actual_convex_solver_calls": bootstrap_solves,
        "bootstrap_scope": "shared_complete_safe_root_not_an_additional_search_start",
        "initial_topologies_materialized": initial_topologies_materialized,
        "global_search_rounds": global_rounds,
        "global_search_rounds_remaining": cfg.max_rounds-global_rounds,
        "round_share_rule": "floor((start_index_plus_one)*max_rounds/initializations)_cumulative_with_unused_carry",
        "initial_structure": "eight_distinct_fit_ranked_certified_guard_topologies",
        "initialization_scope": "deterministic_fit_only_guard_topologies_and_predicate_grids",
        "distinct_requested_initializations": len({_tree_key(tree) for tree in starts}),
        "search_runs": runs, "leaf_optimization_calls": calls,
        "leaf_api_attempts": leaf_api_attempts,
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


# Same convex leaf objective as v6.19, with per-evaluation deadline enforcement.
def fit_constrained_leaf_distribution(target, masks, legal_actions, *,
        epsilon: float = LEAF_EPSILON,
        regularization: float = LEAF_REGULARIZATION,
        maxiter: int = LEAF_MAXITER,
        tolerance: float = LEAF_TOLERANCE, clock=None) -> tuple[np.ndarray, dict]:
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    p = np.asarray(target, np.float64)
    m = np.asarray(masks, np.float64)
    if (p.ndim != 2 or p.shape[1] != 5 or m.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(m).all()
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)
            or np.any(p < 0) or np.any(p[m == 0] > 0)
            or (len(p) and not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0))
            or not np.isfinite(epsilon) or epsilon <= 0
            or not np.isfinite(regularization) or regularization <= 0
            or type(maxiter) is not int or maxiter < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("invalid constrained leaf full-distribution inputs/config")
    raw_legal = tuple(legal_actions)
    if (not raw_legal or any(type(a) not in (int, np.int64, np.int32)
                            or not 0 <= a < 5 for a in raw_legal)):
        raise ValueError("constrained leaf requires path-certified action indices")
    legal = tuple(sorted(set(int(a) for a in raw_legal)))
    # A path certificate is inconsistent if an observed physical mask rejects
    # its action. Such a caller error is never repaired by another action.
    if len(m) and any(np.any(m[:, a] == 0) for a in legal):
        raise ValueError("path-certified action conflicts with observed physical mask")
    count = len(p)
    mean_target = p.mean(0) if count else np.zeros(5, np.float64)
    patterns, inverse, frequencies = np.unique(m, axis=0, return_inverse=True, return_counts=True)
    # For unit distributions this is count/n. Retaining the actual floating
    # row sums makes the grouped objective exactly equal to -mean(p*log q),
    # including float32 NN probabilities whose stored sum differs by one ULP.
    fractions = (np.bincount(inverse, weights=p.sum(-1), minlength=len(patterns))/count
                 if count else np.zeros(0, np.float64))
    prior = np.log((p.sum(0)+.2)/(m.sum(0)+1.))
    prior -= prior.mean()
    entropy_constant = float(np.sum(np.where(p > 0, p*np.log(np.maximum(p, 1e-300)), 0))/count) if count else 0.
    safety = 8*np.finfo(np.float32).eps*max(1., float(np.max(np.abs(prior))))
    constraint_margin = epsilon+safety

    def ce_gradient(z):
        ce, gradient = -float(mean_target@z), -mean_target.copy()
        for pattern, fraction in zip(patterns, fractions, strict=True):
            active = pattern > .5
            normalizer = logsumexp(z[active])
            ce += float(fraction*normalizer)
            gradient[active] += fraction*np.exp(z[active]-normalizer)
        return ce, gradient

    def objective(z):
        ce, gradient = ce_gradient(z)
        offset = z-prior
        return ce+regularization*float(offset@offset), gradient+2*regularization*offset

    candidates, records = [], []
    for action in legal:
        others = [a for a in range(5) if a != action]
        matrix = np.zeros((4, 5), np.float64)
        matrix[:, action] = 1.
        matrix[np.arange(4), others] = -1.
        initial = prior.copy()
        initial[action] = max(initial[action], float(initial[others].max())+constraint_margin)
        initial -= initial.mean()
        solver_started = False
        def checked_objective(z):
            if clock is not None:
                clock.check()
            return objective(z)
        try:
            if clock is not None:
                clock.check()
            solver_started = True
            result = minimize(checked_objective, initial, method="SLSQP", jac=True,
                callback=(lambda z: clock.check()) if clock is not None else None,
                constraints=(
                    {"type": "eq", "fun": lambda z: float(z.sum()),
                     "jac": lambda z: np.ones(5, np.float64)},
                    {"type": "ineq", "fun": lambda z, mat=matrix: mat@z-constraint_margin,
                     "jac": lambda z, mat=matrix: mat},
                ), options={"maxiter": maxiter, "ftol": tolerance, "disp": False})
        except _DeadlineExhausted as error:
            if solver_started:
                records.append({"action": action, "success": False, "solver_status": None,
                    "message": "wall_time_budget", "finite": False, "float32_feasible": False})
            error.enumerated_action_solves = len(records)
            error.report = {"success": False, "candidate_reports": records,
                "enumerated_action_solves": len(records), "wall_time_budget_exhausted": True}
            raise
        except Exception as exc:
            records.append({"action": action, "success": False, "solver_status": None,
                            "message": f"{type(exc).__name__}: {exc}",
                            "finite": False, "float32_feasible": False})
            continue
        z64 = np.asarray(result.x, np.float64)
        z32 = np.asarray(z64-z64.mean(), np.float32)
        proof = float32_preference_proof(z32, action, legal, epsilon=epsilon)
        feasible = (np.isfinite(z64).all() and abs(float(z64.sum())) <= tolerance
                    and float(np.min(matrix@z64-constraint_margin)) >= -tolerance
                    and proof["passed"])
        final_ce, _ = ce_gradient(z32.astype(np.float64))
        final_objective, _ = objective(z32.astype(np.float64))
        record = {"action": action, "success": bool(result.success),
                  "solver_status": int(result.status), "message": str(result.message),
                  "iterations": int(result.nit), "finite": bool(np.isfinite(z32).all()),
                  "float32_feasible": bool(feasible),
                  "objective": float(final_objective) if np.isfinite(final_objective) else None,
                  "masked_cross_entropy": float(final_ce) if np.isfinite(final_ce) else None,
                  "masked_kl": float(final_ce+entropy_constant) if np.isfinite(final_ce) else None,
                  "proof": proof}
        records.append(record)
        if result.success and feasible and np.isfinite(final_objective):
            candidates.append((float(final_objective), action, z32, record))
    if not candidates:
        raise ConstrainedLeafFitError(
            "No converged finite float32-feasible path-constrained leaf solution: "+repr(records),
            candidate_reports=records)
    _, action, logits, winner = min(candidates, key=lambda item: (item[0], item[1]))
    report = {"success": True, "row_count": count, "empty_leaf": not bool(count),
        "eligible_for_qualification": bool(count), "chosen_action": action,
        "objective": winner["objective"], "masked_cross_entropy": winner["masked_cross_entropy"],
        "masked_kl": winner["masked_kl"], "epsilon": float(epsilon),
        "effective_constraint_margin": float(constraint_margin),
        "float32_safety_margin": float(safety), "regularization": float(regularization),
        "solver": "SLSQP", "solver_status": winner["solver_status"],
        "iterations": winner["iterations"], "maxiter": maxiter, "tolerance": float(tolerance),
        "float32_feasible": True, "enumerated_action_solves": len(legal),
        "mask_pattern_count": len(patterns), "physical_fit_frame_count": count,
        "objective_scope": "exact_all_frame_masked_cross_entropy_plus_prior_l2",
        "prior_logits": prior.tolist(), "candidate_reports": records,
    }
    return logits, report

