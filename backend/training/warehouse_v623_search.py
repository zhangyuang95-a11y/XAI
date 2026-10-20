"""v6.23 fit-only continuation of a saved v6.22 A32 topology.

The frozen neural policy is only a target distribution. This code never chooses
an environment action. The v6.22 implementation and receipts remain untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from collections import deque

import numpy as np

from .warehouse_program_search import _BeamEntry, _choose_beam, _improves
from .warehouse_v619_tree import (
    SEARCH_INITIALIZATIONS, _Decision, _BudgetExhausted, _export, _replace,
    _size, _tree_key,
)
from .warehouse_v622_program import DEFAULT_LIMITS, normalize_structure
from .warehouse_v622_search import (
    _Clock, _DeadlineExhausted, _FitContext, _leaves_score,
    _template_predicates, _verify_export, SearchConfig,
)

METHODS = {"S1": "v623_a32_persistent_fit_search_v1",
           "S2": "v623_a32_subtree_refinement_v1"}


@dataclass(frozen=True)
class ContinuationBudget:
    maximum_new_leaf_calls: int = 20000
    final_leaf_reserve: int = 32
    wall_seconds: int = 5400
    final_wall_reserve: int = 600
    beam_width: int = 4
    stagnant_recipe_cycles: int = 3
    maximum_rounds: int = 512

    def __post_init__(self):
        values = vars(self)
        if any(type(v) is not int or v < 1 for v in values.values()):
            raise ValueError("v623 budgets must be positive integers")
        if (self.maximum_new_leaf_calls > 20000 or self.final_leaf_reserve != 32
                or self.wall_seconds > 5400 or self.final_wall_reserve != 600
                or self.beam_width > 4 or self.stagnant_recipe_cycles > 3
                or self.maximum_rounds > 512):
            raise ValueError("v623 budget exceeds registered ceilings")
        if self.maximum_new_leaf_calls <= self.final_leaf_reserve or self.wall_seconds <= self.final_wall_reserve:
            raise ValueError("v623 final refit reserve consumes search budget")


def _tree(structure):
    tree, capacity = normalize_structure(structure, limits=DEFAULT_LIMITS)
    def convert(ref):
        if ref < 0:
            return None
        return _Decision(tuple(float(x) for x in tree["split_weights"][ref]),
                         float(tree["split_bias"][ref]),
                         convert(int(tree["left_refs"][ref])),
                         convert(int(tree["right_refs"][ref])))
    result = convert(tree["root_ref"])
    if _size(result) != (capacity["leaf_count"], capacity["depth"]):
        raise ValueError("saved incumbent changed hard topology")
    return result


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    return value


def _digest(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _guard(feature_count, index):
    w = np.zeros(feature_count, np.float32); w[index] = 1
    return _Decision(tuple(w), -.5)


def _candidate_subtrees(ctx, region, recipe, cfg, mode, masks):
    """Finite, fit-only candidate menu. Every result is evaluated as a hard tree."""
    predicates = _template_predicates(ctx, region, recipe, cfg)
    rows = region.rows
    top = ctx.top[rows]
    def preferred(child_rows, certified):
        choices = [a for a in range(4) if a not in certified and np.any(top[child_rows] == a)]
        return min(choices, key=lambda a: (-int(np.sum(top[child_rows] == a)), a)) if choices else None
    out = []
    for w, b, label, is_guard, _ in predicates:
        base = _Decision(tuple(w), b)
        out.append((base, label))
        if is_guard or region.depth + 2 > cfg.max_depth:
            continue
        from .warehouse_v6_tree_routing import split_scores, HARD_SPLIT_TIE_EPSILON
        right = split_scores(ctx.x[rows], np.asarray(w)[None], np.asarray([b], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
        left_action = preferred(~right, region.legal)
        right_action = preferred(right, region.legal)
        if left_action is not None or right_action is not None:
            left = _guard(len(ctx.names), masks[left_action]) if left_action is not None else None
            child = _guard(len(ctx.names), masks[right_action]) if right_action is not None else None
            out.append((_Decision(tuple(w), b, left, child), "joint:" + label))
        if mode == "S2" and region.depth + 3 <= cfg.max_depth:
            # Replace two neighbouring levels together; certified guard and
            # empty children count in the same 32-leaf/depth-six limits.
            for branch, child_rows in ((0, rows[~right]), (1, rows[right])):
                if len(child_rows) < 40:
                    continue
                child_region = type(region)(region.path+(branch,), None, child_rows,
                                            region.depth+1, region.legal)
                child_pred = _template_predicates(ctx, child_region, recipe, cfg)[:3]
                for cw, cb, clabel, _, _ in child_pred:
                    sub = _Decision(tuple(cw), cb)
                    pair = _Decision(tuple(w), b, sub if branch == 0 else None,
                                     sub if branch == 1 else None)
                    out.append((pair, "subtree:" + label + ":" + clabel))
    return out


def _proposal_trees(ctx, state, recipe, mode, cfg, masks):
    offered = {}
    def add(tree, kind, label, mass):
        count, depth = _size(tree)
        if count > cfg.max_leaves or depth > cfg.max_depth:
            return
        key = _tree_key(tree)
        if key != state.key:
            offered.setdefault(key, (tree, kind, label, float(mass)))
    leaves = [r for r in state.regions if r.tree is None]
    branches = [r for r in state.regions if r.tree is not None]
    for r in leaves:
        for replacement, label in _candidate_subtrees(ctx, r, recipe, cfg, mode, masks):
            add(_replace(state.tree, r.path, replacement), "grow", label, len(r.rows))
    for r in branches:
        add(_replace(state.tree, r.path, None), "prune", str(r.path), len(r.rows))
        menu = _candidate_subtrees(ctx, r, recipe, cfg, mode, masks)
        for replacement, label in menu:
            add(_replace(state.tree, r.path, replacement), "replace", label, len(r.rows))
            if replacement is not None and r.tree is not None:
                node = _Decision(replacement.weight, replacement.bias,
                                 r.tree.left, r.tree.right)
                add(_replace(state.tree, r.path, node), "node", label, len(r.rows))
    # A paired prune/grow move crosses a temporary capacity barrier.
    for source in sorted(branches, key=lambda r: (len(r.rows), r.path))[:2]:
        reduced = _replace(state.tree, source.path, None)
        for destination in sorted(leaves, key=lambda r: (-len(r.rows), r.path))[:3]:
            if destination.path[:len(source.path)] == source.path:
                continue
            for replacement, label in _candidate_subtrees(ctx, destination, recipe, cfg, mode, masks)[:4]:
                add(_replace(reduced, destination.path, replacement), "reallocate", label,
                    len(destination.rows))
    order = {"grow": 0, "node": 1, "replace": 2, "reallocate": 3, "prune": 4}
    return sorted(offered.values(), key=lambda row: (order[row[1]], -row[3], row[2], _tree_key(row[0])))


def fit_structure(mode, encoded, masks, targets, episode_ids, *, feature_names,
                  actor_role_sha256, initial_structure, initial_structure_sha256,
                  budget=ContinuationBudget(), checkpoint_path=None):
    if mode not in METHODS or not isinstance(budget, ContinuationBudget):
        raise ValueError("unknown or unregistered v623 search")
    initial, capacity = normalize_structure(initial_structure, limits=DEFAULT_LIMITS)
    if _digest(initial) != initial_structure_sha256:
        raise ValueError("saved v622 topology SHA mismatch")
    x = np.asarray(encoded, np.float32); m = np.asarray(masks, np.float32)
    p = np.asarray(targets, np.float32); episodes = np.asarray(list(map(str, episode_ids)))
    names = tuple(feature_names)
    mask_indices = tuple(names.index("legal_"+a.lower()) for a in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"))
    if (x.ndim != 2 or len(x) < 2 or x.shape[1] != len(names) or m.shape != (len(x), 5)
            or p.shape != m.shape or episodes.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(m).all() or not np.isfinite(p).all()
            or not np.array_equal(x[:, mask_indices], m) or np.any((m != 0)&(m != 1))
            or np.any(p < 0) or np.any(p[m == 0] != 0) or not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0)
            or any(not e for e in episodes) or len(actor_role_sha256) != 64):
        raise ValueError("v623 requires complete masked fit-only NN distribution rows")
    cfg = SearchConfig()
    clock = _Clock(budget.wall_seconds-budget.final_wall_reserve)
    fit_input_sha256 = _digest({"encoded": x, "masks": m, "targets": p,
                                "episodes": episodes.tolist()})
    ctx = _FitContext(x, m, p, episodes, names, mask_indices, clock,
                      budget.maximum_new_leaf_calls-budget.final_leaf_reserve)
    states = {}
    def evaluate(tree):
        key = _tree_key(tree)
        if key in states: return states[key]
        n, depth = _size(tree)
        if n > cfg.max_leaves or depth > cfg.max_depth:
            raise ValueError("v623 candidate exceeds fixed complexity")
        regions = list(ctx.regions(tree))
        leaves = [ctx.fit(r) for r in regions if r.tree is None]
        score = _leaves_score(leaves)
        from .warehouse_v619_tree import _State
        state = _State(tree, leaves, regions, score, key)
        states[key] = state
        if len(states) > 128:
            states.pop(next(iter(states)))
        return state
    start = evaluate(_tree(initial))
    best = start
    beam = [_BeamEntry(start)]
    expanded = set()
    rounds = []
    no_gain = 0
    stop = "maximum_rounds"
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    for index in range(budget.maximum_rounds):
        recipe_id = index % len(SEARCH_INITIALIZATIONS)
        recipe = SEARCH_INITIALIZATIONS[recipe_id]
        try:
            clock.check()
            queue = {}
            for entry in beam:
                marker = (entry.state.key, recipe_id)
                if marker in expanded: continue
                expanded.add(marker)
                for tree, kind, label, mass in _proposal_trees(ctx, entry.state, recipe, mode, cfg, mask_indices):
                    key = _tree_key(tree)
                    queue.setdefault(key, (tree, kind, label, entry))
            if not queue:
                no_gain += 1
                rounds.append({"round": index+1, "recipe": recipe_id,
                               "evaluated": 0, "best_score": list(best.score), "new_leaf_calls": ctx.calls})
                if no_gain >= 8*budget.stagnant_recipe_cycles:
                    stop = "three_complete_recipe_cycles_without_improvement"; break
                continue
            transitions = []
            before = best.score
            evaluated = failed = 0
            for tree, kind, label, parent in queue.values():
                try:
                    fitted = evaluate(tree)
                except (_BudgetExhausted, _DeadlineExhausted):
                    stop = "wall_budget" if clock.timed_out else "leaf_call_budget"
                    raise
                except RuntimeError:
                    failed += 1; continue
                evaluated += 1
                improve = _improves(fitted.score, parent.state.score)
                debt = 0 if improve else parent.nonimproving_steps+1
                if debt <= cfg.max_nonimproving_steps:
                    transitions.append(_BeamEntry(fitted, debt))
                if _improves(fitted.score, best.score) or (fitted.score == best.score and fitted.key < best.key):
                    best = fitted
            beam = _choose_beam(transitions, best, budget.beam_width)
            changed = _improves(best.score, before)
            no_gain = 0 if changed else no_gain+1
            rounds.append({"round": index+1, "recipe": recipe_id, "evaluated": evaluated,
                           "failed": failed, "improved": changed,
                           "best_score": list(best.score), "new_leaf_calls": ctx.calls,
                           "beam": [e.state.key.hex() for e in beam]})
            if checkpoint is not None:
                # Durable diagnostic and incumbent snapshot. A process killed
                # during a round has uncommitted work; no automatic retry.
                payload = {"schema": "warehouse-v623-search-checkpoint.v1",
                           "round": index+1, "mode": mode, "actor_role_sha256": actor_role_sha256,
                           "fit_input_sha256": fit_input_sha256,
                           "initial_structure_sha256": initial_structure_sha256,
                           "best_structure": _plain(_export(best.tree, best.leaves, len(names))),
                           "best_score": list(best.score), "leaf_calls": ctx.calls,
                           "solver_calls": ctx.solves, "expanded_count": len(expanded),
                           "beam": [e.state.key.hex() for e in beam],
                           "resume_equivalent": False}
                temporary = checkpoint.with_suffix(checkpoint.suffix+".tmp")
                with temporary.open("x") as handle:
                    json.dump(payload, handle, sort_keys=True, allow_nan=False)
                    handle.flush()
                    import os
                    os.fsync(handle.fileno())
                temporary.replace(checkpoint)
            if no_gain >= 8*budget.stagnant_recipe_cycles:
                stop = "three_complete_recipe_cycles_without_improvement"; break
        except (_BudgetExhausted, _DeadlineExhausted):
            stop = "wall_budget" if clock.timed_out else "leaf_call_budget"
            break
    result = _export(best.tree, best.leaves, len(names))
    result["feature_dim"] = len(names)
    _verify_export(result, x, m, names, cfg.max_depth, cfg.max_leaves)
    report = {"schema": "warehouse-v623-fit-search.v1", "method": METHODS[mode],
        "mode": mode, "actor_role_sha256": actor_role_sha256,
        "initial_structure_sha256": initial_structure_sha256,
        "fit_input_sha256": fit_input_sha256,
        "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
        "search_uses_fit_only": True, "calibration_eligibility_validation_used": False,
        "execution_action_override": False, "complete_incumbent": True,
        "new_leaf_calls": ctx.calls, "actual_convex_solver_calls": ctx.solves,
        "leaf_api_attempts": ctx.events, "leaf_cache_hits": ctx.cache_hits,
        "failed_leaf_api_attempts": sum(not e["success"] for e in ctx.events),
        "maximum_new_leaf_calls_including_final_reserve": budget.maximum_new_leaf_calls,
        "reserved_final_leaf_calls": budget.final_leaf_reserve,
        "new_leaf_calls_remaining": budget.maximum_new_leaf_calls-budget.final_leaf_reserve-ctx.calls,
        "rounds": rounds, "stop_reason": stop, "best_score": list(best.score),
        "initial_score": list(start.score), "best_tree_sha256": best.key.hex(),
        "leaf_count": result["leaf_count"], "depth": _size(best.tree)[1],
        "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0,
        "zero_actor_updates": True, "zero_environment_resets": True,
        "elapsed_seconds": clock.report()["elapsed_seconds"],
        "heuristic_not_global_optimality_proof": True,
        "budget": vars(budget), "checkpoint_resume_equivalent": False}
    return result, report
