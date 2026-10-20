"""Bounded fit-only upgrades of frozen S1 trees; never selects an action.

P0 retains the S1 candidate menu. P1 adds cost-sensitive sparse node proposals,
two-level repairs and atomic leaf reallocations. Both use identical exact leaf
statistics, routing, certification, acceptance and budget accounting.

Checkpoints are trusted local training files, not public model assets. Only
explicitly committed candidate boundaries can resume. A dirty marker rejects
an interrupted computation rather than silently forgetting consumed work.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from hashlib import sha256
import gzip
import itertools
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
from scipy.special import logsumexp

from .warehouse_program_search import _BeamEntry, _choose_beam, _improves
from .warehouse_v619_tree import (
    SEARCH_INITIALIZATIONS, _Decision, _Leaf, _Region, _State, _BudgetExhausted,
    _export, _replace, _size, _tree_key, source_feature_cost,
)
from .warehouse_v619_leaf_fit import ConstrainedLeafFitError
from .warehouse_v622_program import normalize_structure, program_structure_sha256
from .warehouse_v622_search import (
    _Clock, _DeadlineExhausted, _FitContext, _leaves_score, _template_predicates,
    _validate, _verify_export, SearchConfig,
)
from .warehouse_v623_search import _tree, _proposal_trees as _s1_proposals, _guard
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_TIE_EPSILON, explicit_mask_branches, split_scores,
)


METHODS = {"P0": "v624_s1_cached_continuation_v1",
           "P1": "v624_downstream_cost_atomic_refinement_v1"}
NODE_FEATURE_MENU = 8
NODE_CANDIDATES = 8
NODES_PER_ROUND = 8
BAD_LEAVES_PER_ROUND = 3
REPAIR_PREDICATES = 4
REPAIR_CHILD_PREDICATES = 2
DONORS_PER_ROUND = 6
CHECKPOINT_CANDIDATES = 64
CHECKPOINT_SECONDS = 30.
FULL_LEAF_CACHE_ENTRIES = 8192
STATE_CACHE_ENTRIES = 128
NODE_CACHE_ENTRIES = 128
# Bulk snapshot I/O is measured directly. This bounded one-second charge
# covers the final small commit receipt, fsyncs and dirty-marker removal.
CHECKPOINT_COMMIT_CHARGE = 1.
CHECKPOINT_FINISH_RESERVE = 10.


class SearchIncompleteError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def search_configuration():
    return {"maximum_depth": 6, "maximum_leaves": 32, "maximum_source_feature_cost": 3,
        "guard_nodes_fixed_in_cost_optimization": True,
        "node_feature_menu": NODE_FEATURE_MENU, "node_candidates_per_node": NODE_CANDIDATES,
        "nodes_per_round": NODES_PER_ROUND, "node_ridge": 1e-5,
        "node_threshold_quantiles": [.1, .25, .5, .75, .9], "node_min_child_rows": 20,
        "bad_leaves_per_round": BAD_LEAVES_PER_ROUND,
        "deep_bad_leaf_repair_anchor": "nearest_ancestor_with_two_remaining_levels",
        "repair_parent_predicates": REPAIR_PREDICATES,
        "repair_child_predicates": REPAIR_CHILD_PREDICATES,
        "repair_min_child_rows": 40, "donors_per_round": DONORS_PER_ROUND,
        "donor_screening": "reserve_best_disjoint_donor_per_repair_region_then_global_rank",
        "move_family_page_size": 8, "s1_candidate_menu_retained": True,
        "full_leaf_cache_entries": FULL_LEAF_CACHE_ENTRIES,
        "tree_state_cache_entries": STATE_CACHE_ENTRIES,
        "node_candidate_cache_entries": NODE_CACHE_ENTRIES, "cache_eviction": "FIFO",
        "checkpoint_candidate_interval": CHECKPOINT_CANDIDATES,
        "checkpoint_seconds_interval": CHECKPOINT_SECONDS,
        "checkpoint_small_commit_charge_seconds": CHECKPOINT_COMMIT_CHARGE,
        "checkpoint_finish_reserve_seconds": CHECKPOINT_FINISH_RESERVE,
        "checkpoint_scope": "committed_candidate_boundary_only",
        "maximum_nonimproving_steps": 2,
        "acceptance": "lexicographic_fit_robust_coverage_expected_feedback_negative_KL_negative_leaves",
        "cost_objective": "full_five_action_physical_masked_cross_entropy",
        "cost_support_menu": "top8_weighted_covariance_all_1_2_3_source_cost_supports"}


@dataclass(frozen=True)
class UpgradeBudget:
    maximum_new_leaf_calls: int = 20000
    final_leaf_reserve: int = 32
    wall_seconds: int = 5400
    final_wall_reserve: int = 600
    beam_width: int = 4
    stagnant_recipe_cycles: int = 3
    maximum_rounds: int = 512
    pause_after_candidates: int = 0  # Explicit small synthetic diagnostics only.

    def __post_init__(self):
        ceilings = {"maximum_new_leaf_calls": 20000, "wall_seconds": 5400,
            "beam_width": 4, "stagnant_recipe_cycles": 3, "maximum_rounds": 512}
        for key, limit in ceilings.items():
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= limit:
                raise ValueError("invalid v624 budget: " + key)
        if (self.final_leaf_reserve != 32 or self.final_wall_reserve != 600
                or type(self.pause_after_candidates) is not int or self.pause_after_candidates < 0
                or self.maximum_new_leaf_calls <= 32 or self.wall_seconds <= 600):
            raise ValueError("v624 reserves 32 final calls and 600 final seconds")

    @property
    def production_configuration(self):
        return asdict(self) == asdict(type(self)())


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _source_identity():
    files = (Path(__file__), Path(__file__).with_name("warehouse_v623_search.py"),
             Path(__file__).with_name("warehouse_v622_search.py"))
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in files}


class _ExactContext(_FitContext):
    """Cache the complete immutable leaf result, not just optimizer logits."""
    def __init__(self, *args):
        super().__init__(*args)
        self.full_cache = OrderedDict()
        self.full_cache_hits = 0
        self.routing_rows = self.routing_reused_regions = 0
        self.node_cache = OrderedDict()
        self.node_fit_calls = self.node_linear_solves = 0
        self.node_fit_seconds = 0.
        self.full_cache_evictions = self.node_cache_hits = self.node_cache_evictions = 0
        self.node_fit_events = []
        self.phase = "initial"
        self.phase_cache_statistics = {}
        self.proposal_failed_solvers = 0
        self.checkpoint_count = 0
        self.checkpoint_io_seconds = self.checkpoint_commit_charge_seconds = 0.

    def fit(self, region, *, fresh=False):
        self.clock.check()
        phase = self.phase_cache_statistics.setdefault(self.phase, Counter())
        key = (sha256(np.ascontiguousarray(region.rows, dtype="<i8").tobytes()).hexdigest(),
               tuple(sorted(region.legal)))
        if not fresh and key in self.full_cache:
            self.cache_hits += 1
            self.full_cache_hits += 1
            phase["full_statistic_hits"] += 1
            values, robust = self.full_cache[key]
            leaf = _Leaf(region, *values)
            leaf.robust = robust
            return leaf
        phase["full_statistic_misses"] += 1
        before, before_solves = self.calls, self.solves
        try:
            leaf = super().fit(region, fresh=fresh)
        finally:
            phase["actual_leaf_API_calls"] += self.calls-before
            phase["actual_convex_solver_calls"] += self.solves-before_solves
            for event in self.events[before:]:
                event["phase"] = self.phase
        if not fresh:
            self.full_cache[key] = ((leaf.logits, leaf.optimizer, leaf.qualified,
                leaf.expected_mass, leaf.kl_sum, leaf.agreement), leaf.robust)
            if len(self.full_cache) > FULL_LEAF_CACHE_ENTRIES:
                old, _ = self.full_cache.popitem(last=False)
                self.cache.pop(old, None)
                self.full_cache_evictions += 1
                phase["full_statistic_evictions"] += 1
        return leaf


def _regions(ctx, tree, parent=None):
    """Reuse a subtree iff topology, incoming rows and ancestor proof agree."""
    prior = {r.path: r for r in parent.regions} if parent is not None else {}
    def walk(node, rows, path, legal):
        old = prior.get(path)
        if (old is not None and old.legal == legal
                and (node is old.tree or node == old.tree)
                and np.array_equal(rows, old.rows)):
            reused = [r for r in parent.regions if r.path[:len(path)] == path]
            ctx.routing_reused_regions += len(reused)
            yield from reused
            return
        region = _Region(path, node, rows, len(path), legal)
        yield region
        if node is not None:
            ctx.clock.check()
            w = np.asarray(node.weight, np.float32)
            right = split_scores(ctx.x[rows], w[None], np.asarray([node.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            ctx.routing_rows += len(rows)
            ll, rl = explicit_mask_branches(w, node.bias, ctx.indices)
            yield from walk(node.left, rows[~right], path+(0,), legal | ll)
            yield from walk(node.right, rows[right], path+(1,), legal | rl)
    yield from walk(tree, np.arange(len(ctx.x)), (), frozenset({4}))


def _subtree_cost(ctx, state, region, branch):
    """Exact full masked CE for hypothetical routing through an existing child."""
    leaves = {leaf.region.path: leaf for leaf in state.leaves}
    result = np.empty(len(region.rows), np.float64)
    def walk(node, offsets, path):
        if node is None:
            leaf = leaves[path]
            rows = region.rows[offsets]
            z = np.where(ctx.m[rows] > .5, leaf.logits.astype(np.float64), -np.inf)
            logq = z-logsumexp(z, axis=-1, keepdims=True)
            result[offsets] = -np.sum(ctx.p[rows].astype(np.float64)*np.where(ctx.m[rows] > .5, logq, 0.), axis=-1)
            invalid = ctx.m[rows, int(leaf.logits.argmax())] != 1
            result[offsets[invalid]] = np.inf
            return
        w = np.asarray(node.weight, np.float32)
        right = split_scores(ctx.x[region.rows[offsets]], w[None], np.asarray([node.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
        walk(node.left, offsets[~right], path+(0,))
        walk(node.right, offsets[right], path+(1,))
    child = region.tree.left if branch == 0 else region.tree.right
    walk(child, np.arange(len(region.rows)), region.path+(branch,))
    return result


def _cost_nodes_impl(ctx, state, region):
    key = (state.key, region.path)
    if key in ctx.node_cache:
        ctx.node_cache_hits += 1
        return ctx.node_cache[key]
    begin = time.monotonic()
    ctx.node_fit_calls += 1
    event = {"call": ctx.node_fit_calls, "state_sha256": state.key.hex(),
        "path": list(region.path), "fit_rows": len(region.rows),
        "fit_rows_sha256": sha256(np.ascontiguousarray(region.rows, dtype="<i8").tobytes()).hexdigest(),
        "fit_only": True, "support_fits": [], "success": False, "duration_seconds": None}
    ctx.node_fit_events.append(event)
    left, right = (_subtree_cost(ctx, state, region, b) for b in (0, 1))
    # A semantic node has no legal-bit proof. Existing guards are kept, so
    # both hypothetical children must remain physically certified for every row.
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("downstream cost encountered an unproved hypothetical action")
    weight = np.abs(left-right)
    result = []
    if weight.sum() > 1e-12:
        rows = region.rows
        y = np.where(right < left, 1., -1.)
        x = ctx.x[rows].astype(np.float64)
        sample_weight = weight/weight.sum()
        mean = np.sum(x*sample_weight[:, None], axis=0)
        centered = x-mean
        ymean = float(np.sum(sample_weight*y))
        cov = np.sum(centered*(sample_weight*(y-ymean))[:, None], axis=0)
        scale = np.sqrt(np.sum(centered**2*sample_weight[:, None], axis=0))
        ranked = sorted((i for i in range(len(ctx.names))
            if i not in ctx.indices and scale[i] > 1e-8),
            key=lambda i: (-abs(cov[i])/scale[i], i))[:NODE_FEATURE_MENU]
        candidates = {}
        for width in (1, 2, 3):
            for support in itertools.combinations(ranked, width):
                ctx.clock.check()
                if sum(2 if ctx.names[i] == "public_hard_risk_context" else 1 for i in support) > 3:
                    continue
                a = centered[:, support]/scale[list(support)]
                gram = np.einsum("ni,nj,n->ij", a, a, sample_weight, optimize=False)+np.eye(width)*1e-5
                rhs = np.einsum("ni,n->i", a, sample_weight*(y-ymean), optimize=False)
                ctx.node_linear_solves += 1
                linear_begin = time.monotonic()
                attempt = {"support": list(support), "source_features": [ctx.names[i] for i in support],
                    "source_feature_cost": sum(2 if ctx.names[i] == "public_hard_risk_context" else 1 for i in support),
                    "success": False}
                event["support_fits"].append(attempt)
                try:
                    coefficients = np.linalg.solve(gram, rhs)/scale[list(support)]
                    attempt["success"] = bool(np.isfinite(coefficients).all())
                except np.linalg.LinAlgError as exc:
                    attempt["error"] = str(exc)
                    continue
                finally:
                    attempt["duration_seconds"] = time.monotonic()-linear_begin
                if not attempt["success"]:
                    continue
                w = np.zeros(len(ctx.names), np.float32)
                w[list(support)] = coefficients
                norm = float(np.linalg.norm(w))
                if norm <= 1e-12:
                    continue
                w /= norm
                projection = split_scores(ctx.x[rows], w[None], np.zeros(1, np.float32))[:, 0]
                thresholds = [float(np.sum(mean*w)-ymean/norm)]
                thresholds.extend(float(np.quantile(projection, q)) for q in (.1, .25, .5, .75, .9))
                for threshold in thresholds:
                    b = float(np.float32(-threshold))
                    go = projection+b > HARD_SPLIT_TIE_EPSILON
                    if min(int(go.sum()), int((~go).sum())) < 20:
                        continue
                    cost = float(np.where(go, right, left).sum())
                    node = _Decision(tuple(w), b, region.tree.left, region.tree.right)
                    candidates.setdefault(_tree_key(node), (cost, node))
        result = [v[1] for _, v in sorted(candidates.items(), key=lambda item: (item[1][0], item[0]))[:NODE_CANDIDATES]]
    duration = time.monotonic()-begin
    ctx.node_fit_seconds += duration
    event.update(success=True, duration_seconds=duration,
        candidate_count=len(result),
        candidates=[{"support": np.flatnonzero(np.asarray(node.weight)).tolist(),
            "weights": [float(node.weight[i]) for i in np.flatnonzero(np.asarray(node.weight))],
            "bias": node.bias, "tree_sha256": _tree_key(node).hex()} for node in result])
    ctx.node_cache[key] = result
    if len(ctx.node_cache) > NODE_CACHE_ENTRIES:
        ctx.node_cache.popitem(last=False)
        ctx.node_cache_evictions += 1
    return result


def _cost_nodes(ctx, state, region):
    begin = time.monotonic()
    prior_events = len(ctx.node_fit_events)
    try:
        return _cost_nodes_impl(ctx, state, region)
    except BaseException as exc:
        if len(ctx.node_fit_events) > prior_events:
            event = ctx.node_fit_events[-1]
            if event["duration_seconds"] is None:
                duration = time.monotonic()-begin
                event.update(success=False, duration_seconds=duration,
                             error_type=type(exc).__name__, error=str(exc))
                ctx.node_fit_seconds += duration
        raise


def _repair_menu(ctx, region, recipe, cfg):
    """Finite menu: four predicates, two child predicates, all guards charged."""
    from .warehouse_v623_search import _candidate_subtrees
    result = list(_candidate_subtrees(ctx, region, recipe, cfg, "S1", ctx.indices))
    if region.depth+2 <= cfg.max_depth:
        for w, b, label, guard, _ in _template_predicates(ctx, region, recipe, cfg)[:REPAIR_PREDICATES]:
            if guard:
                continue
            go = split_scores(ctx.x[region.rows], np.asarray(w)[None], np.asarray([b], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            for branch, rows in ((0, region.rows[~go]), (1, region.rows[go])):
                if len(rows) < 40:
                    continue
                child = _Region(region.path+(branch,), None, rows, region.depth+1, region.legal)
                for cw, cb, clabel, _, _ in _template_predicates(ctx, child, recipe, cfg)[:REPAIR_CHILD_PREDICATES]:
                    node = _Decision(tuple(cw), cb)
                    result.append((_Decision(tuple(w), b, node if branch == 0 else None,
                        node if branch == 1 else None), "two-level:"+label+":"+clabel))
    return result


def _p1_proposals(ctx, state, recipe, cfg):
    offered = OrderedDict()
    def add(tree, kind, label):
        count, depth = _size(tree)
        key = _tree_key(tree)
        if key != state.key and count <= 32 and depth <= 6:
            offered.setdefault(key, (tree, kind, label, 0.))
    semantic = [r for r in state.regions if r.tree is not None
        and not any(explicit_mask_branches(np.asarray(r.tree.weight), r.tree.bias, ctx.indices))]
    semantic.sort(key=lambda r: (-len(r.rows), r.path))
    for region in semantic[:NODES_PER_ROUND]:
        for index, node in enumerate(_cost_nodes(ctx, state, region)):
            add(_replace(state.tree, region.path, node), "cost_node", str(region.path)+":"+str(index))
    bad = sorted((leaf for leaf in state.leaves if not leaf.robust["qualified"]),
        key=lambda leaf: (-len(leaf.region.rows), -leaf.kl_sum, leaf.region.path))[:BAD_LEAVES_PER_ROUND]
    menus, repair_regions = {}, OrderedDict()
    by_path = {region.path: region for region in state.regions}
    for leaf in bad:
        # A depth-six terminal has no room to grow. Rebuild its nearest
        # ancestor with two remaining levels, charging every replaced leaf
        # and proving the new paths after fitting; freeing leaves alone does
        # not free depth.
        path = leaf.region.path[:min(len(leaf.region.path), cfg.max_depth-2)]
        repair_regions.setdefault(path, by_path[path])
    for path, region in repair_regions.items():
        menus[path] = _repair_menu(ctx, region, recipe, cfg)
        for replacement, label in menus[path]:
            kind = "ancestor_bad_region_repair" if region.tree is not None else "bad_leaf_repair"
            add(_replace(state.tree, path, replacement), kind, str(path)+":"+label)
    # Exact collapsed-leaf fit supplies loss/reclaimed-leaf ranking. These
    # proposal-construction calls consume the same actual API ledger/budget.
    donors = []
    for region in state.regions:
        if region.tree is None or not region.path:
            continue
        count = _size(region.tree)[0]
        descendants = [leaf for leaf in state.leaves if leaf.region.path[:len(region.path)] == region.path]
        prior = _leaves_score(descendants)
        # Screen at most six donors by current covered mass per freed leaf.
        donors.append((prior[0]/(count-1), region.path, region, count-1, prior))
    ordered = sorted(donors)
    screened = OrderedDict()
    for destination in repair_regions:
        for item in ordered:
            path = item[1]
            if destination[:len(path)] != path and path[:len(destination)] != destination:
                screened.setdefault(path, item)
                break
    for item in ordered:
        if len(screened) >= DONORS_PER_ROUND:
            break
        screened.setdefault(item[1], item)
    ranked = []
    for _, path, region, freed, prior in screened.values():
        try:
            collapsed = ctx.fit(_Region(path, None, region.rows, region.depth, region.legal))
        except ConstrainedLeafFitError:
            ctx.proposal_failed_solvers += 1
            continue
        after = _leaves_score([collapsed])
        loss = tuple((a-b)/freed for a, b in zip(prior, after, strict=True))
        ranked.append((loss, path, region))
    for _, path, donor in sorted(ranked):
        reduced = _replace(state.tree, path, None)
        for destination, recipient in repair_regions.items():
            if destination[:len(path)] == path or path[:len(destination)] == destination:
                continue
            for replacement, label in menus[destination]:
                add(_replace(reduced, destination, replacement), "atomic_reallocate", str(path)+">"+str(destination)+":"+label)
    for tree, kind, label, mass in _s1_proposals(ctx, state, recipe, "S1", cfg, ctx.indices):
        add(tree, "s1_"+kind, label)
    # Deterministic round robin gives every move family a chance in a partial
    # wall-budget round. It is not a claim of exhaustive sparse-tree search.
    groups = OrderedDict()
    for row in offered.values():
        groups.setdefault(row[1], []).append(row)
    result = []
    while any(groups.values()):
        for values in groups.values():
            result.extend(values[:8]); del values[:8]
    return result


def _checkpoint_write(path, identity, state, ctx, clock):
    if path is None:
        return
    runtime = {key: value for key, value in vars(ctx).items()
        if key not in {"x", "m", "p", "episodes", "names", "indices", "clock", "top", "group"}}
    begin = time.monotonic()
    ctx.checkpoint_count += 1
    runtime["checkpoint_count"] = ctx.checkpoint_count
    payload = {"identity": identity, "state": state, "context": runtime,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    raw = gzip.compress(pickle.dumps(payload, protocol=5), compresslevel=1, mtime=0)
    data_path = Path(str(path)+".payload")
    data_temporary = Path(str(path)+".payload.tmp")
    with data_temporary.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    data_temporary.replace(data_path)
    bulk_seconds = time.monotonic()-begin
    ctx.checkpoint_io_seconds += bulk_seconds
    ctx.checkpoint_commit_charge_seconds += CHECKPOINT_COMMIT_CHARGE
    envelope = {"schema": "warehouse-v624-safe-local-checkpoint.v1",
        "safe_boundary": True, "payload_sha256": sha256(raw).hexdigest(),
        "elapsed": clock.report()["elapsed_seconds"]+CHECKPOINT_COMMIT_CHARGE,
        "checkpoint_io_seconds": ctx.checkpoint_io_seconds,
        "checkpoint_commit_charge_seconds": ctx.checkpoint_commit_charge_seconds,
        "payload_file": data_path.name}
    commit_begin = time.monotonic()
    temporary = Path(str(path)+".tmp")
    with temporary.open("x") as handle:
        json.dump(envelope, handle, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    Path(str(path)+".dirty").unlink(missing_ok=True)
    commit_seconds = time.monotonic()-commit_begin
    if commit_seconds > CHECKPOINT_COMMIT_CHARGE:
        _dirty(path)
        raise RuntimeError("checkpoint commit exceeded its conservatively billed second")
    clock.start -= CHECKPOINT_COMMIT_CHARGE-commit_seconds


def _checkpoint_read(path, identity):
    if path is None or Path(str(path)+".dirty").exists():
        raise RuntimeError("v624 resume rejects missing path or interrupted unsafe work")
    envelope = json.loads(path.read_bytes())
    if envelope.get("schema") != "warehouse-v624-safe-local-checkpoint.v1" or envelope.get("safe_boundary") is not True:
        raise RuntimeError("v624 checkpoint is not a committed safe boundary")
    data_path = Path(str(path)+".payload")
    if envelope.get("payload_file") != data_path.name:
        raise RuntimeError("v624 checkpoint payload path mismatch")
    raw = data_path.read_bytes()
    if sha256(raw).hexdigest() != envelope["payload_sha256"]:
        raise RuntimeError("v624 checkpoint payload hash mismatch")
    payload = pickle.loads(gzip.decompress(raw))
    if payload["identity"] != identity:
        raise RuntimeError("v624 checkpoint source, fit input or configuration mismatch")
    payload["elapsed"] = envelope["elapsed"]
    payload["context"]["checkpoint_io_seconds"] = envelope["checkpoint_io_seconds"]
    payload["context"]["checkpoint_commit_charge_seconds"] = envelope["checkpoint_commit_charge_seconds"]
    return payload


def _dirty(path):
    if path is not None and not Path(str(path)+".dirty").exists():
        with Path(str(path)+".dirty").open("x") as handle:
            handle.write("uncommitted work: hard interruption must fail closed\n")
            handle.flush(); os.fsync(handle.fileno())


def fit_structure(mode, encoded, masks, targets, episode_ids, *, feature_names,
                  actor_role_sha256, initial_structure, initial_structure_sha256,
                  budget=UpgradeBudget(), checkpoint_path=None, resume=False):
    if mode not in METHODS or not isinstance(budget, UpgradeBudget):
        raise ValueError("unknown v624 method or budget")
    initial, _ = normalize_structure(initial_structure)
    if program_structure_sha256(initial) != initial_structure_sha256:
        raise ValueError("v624 saved S1 topology SHA mismatch")
    x, m, p, episodes, names, indices = _validate(encoded, masks, targets, episode_ids, feature_names, actor_role_sha256)
    algorithm_budget = asdict(budget); algorithm_budget.pop("pause_after_candidates")
    identity = {"mode": mode, "source": _source_identity(), "budget": algorithm_budget,
        "configuration": search_configuration(),
        "actor_role_sha256": actor_role_sha256, "initial_structure_sha256": initial_structure_sha256,
        "feature_names": list(names), "episode_ids": list(episodes),
        **{key: sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
           for key, value in (("encoded", x), ("masks", m), ("targets", p))}}
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not resume and (checkpoint.exists() or Path(str(checkpoint)+".dirty").exists()):
            raise RuntimeError("v624 fresh run cannot overwrite an existing checkpoint")
    clock = _Clock(max(.01, budget.wall_seconds-budget.final_wall_reserve-CHECKPOINT_FINISH_RESERVE))
    ctx = _ExactContext(x, m, p, episodes, names, indices, clock,
                        budget.maximum_new_leaf_calls-budget.final_leaf_reserve)
    cfg = SearchConfig()
    state = {"states": OrderedDict(), "expanded": set(), "round": 0,
        "no_gain": 0, "queue": None, "cursor": 0, "transitions": [],
        "rounds": [], "evaluated": 0, "operation_counts": Counter(),
        "operation_seconds": Counter(), "stop": None, "resume_count": 0,
        "candidate_order_sha256": "0"*64}
    if resume:
        saved = _checkpoint_read(checkpoint, identity)
        state = saved["state"]
        vars(ctx).update(saved["context"])
        clock.start -= saved["elapsed"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        state["resume_count"] += 1
    def evaluate(tree, parent=None):
        key = _tree_key(tree)
        if key in state["states"]:
            return state["states"][key]
        count, depth = _size(tree)
        if count > 32 or depth > 6:
            raise RuntimeError("v624 candidate exceeds charged capacity")
        regions = list(_regions(ctx, tree, parent))
        leaves = [ctx.fit(r) for r in regions if r.tree is None]
        fitted = _State(tree, leaves, regions, _leaves_score(leaves), key)
        state["states"][key] = fitted
        if len(state["states"]) > STATE_CACHE_ENTRIES:
            state["states"].popitem(last=False)
        return fitted
    _dirty(checkpoint)
    if not resume:
        try:
            start = evaluate(_tree(initial))
        except (_BudgetExhausted, _DeadlineExhausted, ConstrainedLeafFitError) as exc:
            raise SearchIncompleteError("v624 initial topology has no fully fitted incumbent", {
                "schema": "warehouse-v624-search-incomplete.v1", "status": "search_no_complete_incumbent",
                "scientific_failure": True, "method": METHODS[mode], "mode": mode,
                "failure_kind": "initial_fit_budget_exhausted" if isinstance(exc, _BudgetExhausted) else "initial_leaf_solver_infeasible",
                "actor_role_sha256": actor_role_sha256,
                "initial_structure_sha256": initial_structure_sha256,
                "fit_target_sha256": identity["targets"], "fit_input_sha256": _digest(identity),
                "configuration": search_configuration(), "zero_actor_updates": True,
                "zero_environment_resets": True, "execution_action_override": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "new_leaf_calls": ctx.calls, "leaf_api_attempts": ctx.events,
                "actual_convex_solver_calls": ctx.solves,
                "failed_leaf_api_attempts": sum(not e["success"] for e in ctx.events),
                "reserved_final_leaf_calls": budget.final_leaf_reserve,
                "maximum_new_leaf_calls_including_final_reserve": budget.maximum_new_leaf_calls,
                "node_fit_events": ctx.node_fit_events, "node_fit_calls": ctx.node_fit_calls,
                "node_linear_solves": ctx.node_linear_solves, "node_fit_seconds": ctx.node_fit_seconds,
                "elapsed_seconds": clock.report()["elapsed_seconds"],
                "wall_seconds": budget.wall_seconds,
                "search_wall_seconds": budget.wall_seconds-budget.final_wall_reserve,
                "production_configuration": budget.production_configuration,
                "budget": asdict(budget)}) from exc
        state.update(start=start, best=start, beam=[_BeamEntry(start)])
    last_saved = time.monotonic()
    invocation_evaluated = 0
    while state["stop"] is None and state["round"] < budget.maximum_rounds:
        try:
            clock.check()
            recipe_id = state["round"] % len(SEARCH_INITIALIZATIONS)
            if state["queue"] is None:
                ctx.phase = "proposal"
                queue = OrderedDict()
                for entry in state["beam"]:
                    marker = (entry.state.key, recipe_id)
                    if marker in state["expanded"]:
                        continue
                    state["expanded"].add(marker)
                    offers = (_s1_proposals(ctx, entry.state, SEARCH_INITIALIZATIONS[recipe_id], "S1", cfg, indices)
                        if mode == "P0" else _p1_proposals(ctx, entry.state, SEARCH_INITIALIZATIONS[recipe_id], cfg))
                    for tree, kind, label, _ in offers:
                        queue.setdefault(_tree_key(tree), (tree, kind, label, entry))
                state.update(queue=list(queue.values()), cursor=0, transitions=[],
                    before=state["best"].score, round_evaluated=0, round_failed=0)
            while state["cursor"] < len(state["queue"]):
                clock.check()
                tree, kind, label, parent = state["queue"][state["cursor"]]
                begin = time.monotonic()
                ctx.phase = "candidate"
                try:
                    fitted = evaluate(tree, parent.state)
                except (_BudgetExhausted, _DeadlineExhausted):
                    raise
                except ConstrainedLeafFitError:
                    state["round_failed"] += 1
                else:
                    state["round_evaluated"] += 1
                    improving = _improves(fitted.score, parent.state.score)
                    debt = 0 if improving else parent.nonimproving_steps+1
                    if debt <= cfg.max_nonimproving_steps:
                        state["transitions"].append(_BeamEntry(fitted, debt))
                    best = state["best"]
                    if _improves(fitted.score, best.score) or (fitted.score == best.score and fitted.key < best.key):
                        state["best"] = fitted
                state["operation_counts"][kind] += 1
                state["operation_seconds"][kind] += time.monotonic()-begin
                state["candidate_order_sha256"] = _digest({"prior": state["candidate_order_sha256"],
                    "tree": _tree_key(tree).hex(), "kind": kind, "label": label,
                    "parent": parent.state.key.hex()})
                state["cursor"] += 1; state["evaluated"] += 1; invocation_evaluated += 1
                pause = budget.pause_after_candidates and invocation_evaluated >= budget.pause_after_candidates
                if pause or state["evaluated"] % CHECKPOINT_CANDIDATES == 0 or time.monotonic()-last_saved >= CHECKPOINT_SECONDS:
                    _checkpoint_write(checkpoint, identity, state, ctx, clock)
                    last_saved = time.monotonic()
                    if pause:
                        state["pause_return"] = True
                        break
                    _dirty(checkpoint)
            if state.pop("pause_return", False):
                break
            changed = _improves(state["best"].score, state["before"])
            state["no_gain"] = 0 if changed else state["no_gain"]+1
            state["beam"] = _choose_beam(state["transitions"], state["best"], budget.beam_width)
            state["rounds"].append({"round": state["round"]+1, "recipe": recipe_id,
                "evaluated": state["round_evaluated"], "failed": state["round_failed"],
                "best_score": list(state["best"].score), "new_leaf_calls": ctx.calls,
                "improved": changed})
            state["round"] += 1; state["queue"] = None
            _checkpoint_write(checkpoint, identity, state, ctx, clock)
            last_saved = time.monotonic(); _dirty(checkpoint)
            if state["no_gain"] >= 8*budget.stagnant_recipe_cycles:
                state["stop"] = "three_complete_recipe_cycles_without_improvement"
        except (_BudgetExhausted, _DeadlineExhausted):
            state["stop"] = "wall_budget" if clock.timed_out else "leaf_call_budget"
    paused = state["stop"] is None and state["queue"] is not None and budget.pause_after_candidates > 0
    if not paused and state["stop"] is None:
        state["stop"] = "maximum_rounds"
    result = _export(state["best"].tree, state["best"].leaves, len(names))
    result["feature_dim"] = len(names)
    _verify_export(result, x, m, names, 6, 32)
    _checkpoint_write(checkpoint, identity, state, ctx, clock)
    report = {"schema": "warehouse-v624-fit-search.v1", "method": METHODS[mode], "mode": mode,
        "actor_role_sha256": actor_role_sha256, "initial_structure_sha256": initial_structure_sha256,
        "fit_input_sha256": _digest(identity), "fit_target_sha256": identity["targets"],
        "search_uses_fit_only": True, "calibration_eligibility_validation_used": False,
        "execution_action_override": False, "complete_incumbent": True,
        "new_leaf_calls": ctx.calls, "actual_convex_solver_calls": ctx.solves,
        "leaf_api_attempts": ctx.events, "leaf_cache_hits": ctx.cache_hits,
        "full_leaf_statistic_cache_hits": ctx.full_cache_hits,
        "full_leaf_statistic_cache_entries": len(ctx.full_cache),
        "full_leaf_statistic_cache_evictions": ctx.full_cache_evictions,
        "cache_statistics_by_phase": {key: dict(value) for key, value in ctx.phase_cache_statistics.items()},
        "proposal_failed_leaf_solvers": ctx.proposal_failed_solvers,
        "incremental_routing_rows": ctx.routing_rows,
        "incremental_routing_reused_regions": ctx.routing_reused_regions,
        "failed_leaf_api_attempts": sum(not e["success"] for e in ctx.events),
        "maximum_new_leaf_calls_including_final_reserve": budget.maximum_new_leaf_calls,
        "reserved_final_leaf_calls": budget.final_leaf_reserve,
        "new_leaf_calls_remaining": budget.maximum_new_leaf_calls-budget.final_leaf_reserve-ctx.calls,
        "rounds": state["rounds"], "stop_reason": "paused_safe_boundary" if paused else state["stop"],
        "best_score": list(state["best"].score), "initial_score": list(state["start"].score),
        "best_tree_sha256": state["best"].key.hex(), "leaf_count": result["leaf_count"],
        "depth": _size(state["best"].tree)[1], "whole_tree_raw_illegal_count": 0,
        "whole_tree_path_uncertified_count": 0, "zero_actor_updates": True,
        "zero_environment_resets": True, "elapsed_seconds": clock.report()["elapsed_seconds"],
        "wall_seconds": budget.wall_seconds, "search_wall_seconds": budget.wall_seconds-budget.final_wall_reserve,
        "budget": asdict(budget), "production_configuration": budget.production_configuration,
        "configuration": search_configuration(),
        "heuristic_not_global_optimality_proof": True,
        "checkpoint_resume_equivalent": True, "checkpoint_equivalence_scope": "committed_candidate_boundaries_only",
        "unsafe_interruption_fails_closed": True, "resume_count": state["resume_count"],
        "operation_counts": dict(state["operation_counts"]), "operation_seconds": dict(state["operation_seconds"]),
        "evaluated_candidates": state["evaluated"],
        "candidate_order_sha256": state["candidate_order_sha256"],
        "node_fit_calls": ctx.node_fit_calls, "node_linear_solves": ctx.node_linear_solves,
        "node_fit_seconds": ctx.node_fit_seconds,
        "node_fit_events": ctx.node_fit_events,
        "node_candidate_cache_hits": ctx.node_cache_hits,
        "node_candidate_cache_evictions": ctx.node_cache_evictions,
        "checkpoint_count": ctx.checkpoint_count,
        "checkpoint_io_seconds": ctx.checkpoint_io_seconds,
        "checkpoint_commit_charge_seconds": ctx.checkpoint_commit_charge_seconds,
        "cache_limits": {"full_leaf_statistics": FULL_LEAF_CACHE_ENTRIES,
            "tree_states": STATE_CACHE_ENTRIES, "node_candidates": NODE_CACHE_ENTRIES},
        "finite_menu": {"node_feature_menu": NODE_FEATURE_MENU, "node_candidates_per_node": NODE_CANDIDATES,
            "nodes_per_round": NODES_PER_ROUND, "bad_leaves_per_round": BAD_LEAVES_PER_ROUND,
            "repair_parent_predicates": REPAIR_PREDICATES, "repair_child_predicates": REPAIR_CHILD_PREDICATES,
            "donors_per_round": DONORS_PER_ROUND, "s1_candidate_menu_retained": True}}
    return result, report
