"""Fit-only P0/P1 search with a shared finite topology menu.

P0 uses constant leaves; P1 uses bounded sparse logistic leaves. Both use the
same S1 topology proposals and ordering conditional on the current tree. Leaf
scores can select different beams, so actual later candidate sequences need
not agree. No environment, Actor update, or execution action is present here.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np

from .warehouse_program_search import _BeamEntry, _choose_beam, _improves
from .warehouse_v619_tree import (
    SEARCH_INITIALIZATIONS, _Decision, _Leaf, _Region, _State, _export,
    _size, _tree_key,
)
from .warehouse_v619_leaf_fit import ConstrainedLeafFitError
from .warehouse_v622_program import normalize_structure, program_structure_sha256
from .warehouse_v622_search import SearchConfig, _validate
from .warehouse_v623_search import _tree, _proposal_trees
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_TIE_EPSILON, explicit_mask_branches, split_scores,
)
from .warehouse_v625_checkpoint import (
    BudgetExhausted, CheckpointError, CheckpointSession, DeadlineExceeded,
    digest, plain,
)


METHODS = {"P0": "v625_shared_topology_constant_leaf_v1",
           "P1": "v625_shared_topology_sparse_logistic_leaf_v1"}
CHECKPOINT_CANDIDATES = 64
CHECKPOINT_SECONDS = 30.
SETTLEMENT_RESERVE_SECONDS = 10.
LEAF_CACHE_ENTRIES = 512
STATE_CACHE_ENTRIES = 128
TRANSITION_ENTRIES = 8
CODEC_TREE_CACHE_ENTRIES = 8192


@dataclass(frozen=True)
class UpgradeBudget:
    maximum_new_leaf_calls: int = 20000
    final_leaf_reserve: int = 32
    wall_seconds: int = 5400
    final_wall_reserve: int = 600
    beam_width: int = 4
    stagnant_recipe_cycles: int = 3
    maximum_rounds: int = 512
    pause_after_candidates: int = 0

    def __post_init__(self):
        limits = {"maximum_new_leaf_calls": 20000, "wall_seconds": 5400,
                  "beam_width": 4, "stagnant_recipe_cycles": 3, "maximum_rounds": 512}
        if any(type(getattr(self, k)) is not int or not 0 < getattr(self, k) <= v
               for k, v in limits.items()):
            raise ValueError("v625 budget exceeds registered ceilings")
        if (self.final_leaf_reserve != 32 or self.final_wall_reserve != 600
                or self.maximum_new_leaf_calls <= 32 or self.wall_seconds <= 611
                or type(self.pause_after_candidates) is not int or self.pause_after_candidates < 0):
            raise ValueError("v625 reserves exactly 32 final API calls and 600 seconds")

    @property
    def production_configuration(self):
        return asdict(self) == asdict(type(self)())


def search_configuration():
    return {"maximum_depth": 6, "maximum_leaves": 32,
        "maximum_public_sources_per_node": 3, "maximum_shared_public_sources_per_leaf": 3,
        "topology_menu": "unchanged_v623_S1_proposal_trees",
        "topology_recipe_count": len(SEARCH_INITIALIZATIONS),
        "candidate_order": "S1_family_mass_label_tree_hash_conditional_on_current_tree",
        "same_actual_sequence_required": False,
        "support_menu_features": 8, "maximum_full_support_fits_per_leaf": 8,
        "P0_leaf_language": "constant", "P1_leaf_language": "sparse_logistic",
        "score": "fit_robust_coverage_expected_feedback_negative_KL_negative_leaves",
        "qualification_stability_groups": 3,
        "leaf_cache_entries": LEAF_CACHE_ENTRIES,
        "state_cache_entries": STATE_CACHE_ENTRIES,
        "transition_entries": TRANSITION_ENTRIES,
        "codec_tree_reference_cache_entries": CODEC_TREE_CACHE_ENTRIES,
        "codec_decoded_tree_cache_entries": CODEC_TREE_CACHE_ENTRIES,
        "codec_patch_split_cache_entries": CODEC_TREE_CACHE_ENTRIES,
        "codec_state_reference_cache_entries": STATE_CACHE_ENTRIES,
        "cache_eviction": "FIFO",
        "checkpoint_candidate_interval": CHECKPOINT_CANDIDATES,
        "checkpoint_seconds_interval": CHECKPOINT_SECONDS,
        "settlement_reserve_seconds": SETTLEMENT_RESERVE_SECONDS,
        "checkpoint": "content_addressed_JSON_tree_deltas_and_predebit_ledger",
        "queue_checkpoint": "single_ordered_exact_sparse_patches_against_unique_parent_states",
        "object_publication": "batch_immutable_uint8_npy_CAS_packs_with_JSON_offset_index_and_registry",
        "resume": "explicit_pause_only_never_dirty_crash_or_unsettled_call",
        "search_calls": 19968, "search_seconds": 4800,
        "final_calls_reserved": 32, "final_seconds_reserved": 600,
        "no_actor_updates": True, "no_environment_resets": True,
        "global_optimality_claim": False}


class SearchIncompleteError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _source_identity():
    paths = (Path(__file__), Path(__file__).with_name("warehouse_v625_checkpoint.py"),
             Path(__file__).with_name("warehouse_v625_program.py"),
             Path(__file__).with_name("warehouse_v623_search.py"),
             Path(__file__).with_name("warehouse_v622_search.py"))
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in paths}


def _tuple_tree(value):
    if isinstance(value, list):
        return tuple(_tuple_tree(x) for x in value)
    return value


def _solver_calls(events):
    # A predebit can expire during its fsync, before scipy is entered. It
    # remains charged but must not be reported as an observed solver call.
    return sum(event["event"] == "call_completed"
        and (event["data"].get("detail", {}).get("solver") == "SLSQP"
             or "exception" in event["data"].get("detail", {}))
        for event in events)


class _Clock:
    def __init__(self, session):
        self.session = session

    def check(self):
        self.session.check(reserve_seconds=SETTLEMENT_RESERVE_SECONDS)


class _LeafBudget:
    """Program solver hook: durable predebit, settlement, callback deadline."""
    def __init__(self, context, key, region):
        self.context, self.key, self.region = context, key, region

    def check(self):
        self.context.clock.check()

    def consume(self):
        self.check()
        return self.context.session.consume({"leaf_key": self.key,
            "rows": len(self.region.rows), "path": list(self.region.path),
            "path_legal_actions": sorted(self.region.legal), "phase": self.context.phase})

    def finish(self, token=None, *, success=True, detail=None):
        return self.context.session.finish(token, success=success, detail=detail)


class _Context:
    def __init__(self, x, masks, p, episodes, names, indices, mode, session):
        self.x, self.m, self.p, self.episodes = x, masks, p, episodes
        self.names, self.indices, self.mode = names, indices, mode
        self.session, self.clock = session, _Clock(session)
        self.top = p.argmax(-1)
        order = sorted(set(episodes), key=lambda value: sha256(value.encode()).digest())
        groups = {episode: i % 3 for i, episode in enumerate(order)}
        self.group = np.asarray([groups[e] for e in episodes])
        self.cache, self.cache_refs = OrderedDict(), OrderedDict()
        self.cache_hits = self.cache_evictions = 0
        self.phase = "initial"

    def regions(self, tree, rows=None, path=(), legal=frozenset({4})):
        rows = np.arange(len(self.x)) if rows is None else rows
        yield _Region(path, tree, rows, len(path), legal)
        if tree is not None:
            self.clock.check()
            w = np.asarray(tree.weight, np.float32)
            go = split_scores(self.x[rows], w[None], np.asarray([tree.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            ll, rl = explicit_mask_branches(w, tree.bias, self.indices)
            yield from self.regions(tree.left, rows[~go], path+(0,), legal | ll)
            yield from self.regions(tree.right, rows[go], path+(1,), legal | rl)

    def fit(self, region):
        from .warehouse_v625_program import fit_leaf, evaluate_leaf, certify_leaf
        self.clock.check()
        rows = region.rows
        key = digest({"rows": sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest(),
                      "legal": sorted(region.legal), "mode": self.mode})
        if key in self.cache:
            self.cache_hits += 1
            data = self.cache[key]
        else:
            hook = _LeafBudget(self, key, region)
            try:
                model = fit_leaf(self.x[rows], self.m[rows], self.p[rows],
                    self.names, sorted(region.legal), mode=self.mode, budget=hook)
            except BaseException as error:
                # Solver implementations must settle each support. Retain a
                # failure record as a final safeguard; never refund its call.
                if self.session.pending:
                    self.session.finish(success=False, detail={"error_type": type(error).__name__,
                        "error": str(error)[:500], "settled_by_search_error_boundary": True})
                raise
            if self.session.pending:
                raise CheckpointError("program fit returned with an unsettled support solve")
            certificate = certify_leaf(model, self.names, sorted(region.legal))
            if certificate.get("passed") is not True:
                raise CheckpointError("program solver returned an uncertified leaf")
            logits, q = evaluate_leaf(model, self.x[rows], self.m[rows])
            logits, q = np.asarray(logits), np.asarray(q, dtype=np.float64)
            if logits.shape != (len(rows), 5) or q.shape != logits.shape:
                raise CheckpointError("leaf prediction shape differs")
            top = logits.argmax(-1)
            if len(rows) and (np.any(self.m[rows, top] != 1) or np.any(~np.isin(top, tuple(region.legal)))):
                raise CheckpointError("leaf path certificate disagrees with its physical rows")
            p = self.p[rows].astype(np.float64)
            kl = np.sum(p*(np.log(np.maximum(p, 1e-300))-np.log(np.maximum(q, 1e-300))), axis=-1)
            matches = self.top[rows] == top
            _, counts = np.unique(self.episodes[rows], return_counts=True)
            agreement = float(matches.mean()) if len(rows) else 0.
            qualified = bool(len(rows) >= 80 and len(counts) >= 8
                and counts.max()/len(rows) <= .20 and agreement >= .95 and kl.mean() <= .25)
            groups = []
            for group in range(3):
                take = self.group[rows] == group
                groups.append({"group": group, "rows": int(take.sum()),
                    "episodes": len(set(self.episodes[rows[take]])),
                    "agreement": float(matches[take].mean()) if take.any() else None,
                    "masked_kl": float(kl[take].mean()) if take.any() else None})
            robust = bool(qualified and agreement >= .97 and kl.mean() <= .20
                and all(g["episodes"] >= 2 and g["agreement"] >= .97
                        and g["masked_kl"] <= .20 for g in groups))
            expected = float(np.sum(p[np.arange(len(rows)), top]*matches)) if qualified else 0.
            data = {"model": plain(model), "qualified": qualified, "agreement": agreement,
                "expected_mass": expected, "kl_sum": float(kl.sum()),
                "robust": {"qualified": robust, "groups": groups}, "rows_sha256": key}
            ref = self.session.store.put_json(data)
            self.cache[key], self.cache_refs[key] = data, ref
            if len(self.cache) > LEAF_CACHE_ENTRIES:
                removed, _ = self.cache.popitem(last=False)
                self.cache_refs.pop(removed)
                self.cache_evictions += 1
        leaf = _Leaf(region, np.asarray(data["model"]["bias"], np.float32),
            data["model"].get("report", {}), data["qualified"], data["expected_mass"],
            data["kl_sum"], data["agreement"])
        leaf.robust, leaf.model, leaf.cache_key = data["robust"], data["model"], key
        leaf.cache_ref = self.cache_refs[key]
        return leaf


def _score(leaves):
    return (float(sum(len(l.region.rows) for l in leaves if l.robust["qualified"])),
            float(sum(l.expected_mass for l in leaves)),
            -float(sum(l.kl_sum for l in leaves)), -float(len(leaves)))


class _Codec:
    """Merkle topology/state references; large training rows never serialize."""
    def __init__(self, context):
        self.ctx, self.store = context, context.session.store
        self.tree_refs, self.state_refs = OrderedDict(), OrderedDict()
        self.decoded_trees = OrderedDict()
        self.patch_splits = OrderedDict()

    def tree_ref(self, tree):
        if tree is None:
            return None
        key = _tree_key(tree).hex()
        if key not in self.tree_refs:
            support = np.flatnonzero(np.asarray(tree.weight))
            self.tree_refs[key] = self.store.put_json({"kind": "sparse_tree_node",
                "support": support.tolist(), "weights": [float(tree.weight[i]) for i in support],
                "feature_dim": len(self.ctx.names), "bias": float(tree.bias),
                "left": self.tree_ref(tree.left), "right": self.tree_ref(tree.right)})
            if len(self.tree_refs) > CODEC_TREE_CACHE_ENTRIES:
                self.tree_refs.popitem(last=False)
        return self.tree_refs[key]

    def tree(self, ref):
        if ref is None:
            return None
        cache_key = (ref["sha256"], ref["kind"], ref["size"])
        if cache_key in self.decoded_trees:
            return self.decoded_trees[cache_key]
        node = self.store.get_json(ref)
        if node.get("kind") != "sparse_tree_node" or node.get("feature_dim") != len(self.ctx.names):
            raise CheckpointError("stored tree schema differs")
        weights = np.zeros(len(self.ctx.names), np.float32)
        weights[node["support"]] = node["weights"]
        result = _Decision(tuple(map(float, weights)), float(node["bias"]),
                           self.tree(node["left"]), self.tree(node["right"]))
        self.tree_refs[_tree_key(result).hex()] = ref
        if len(self.tree_refs) > CODEC_TREE_CACHE_ENTRIES:
            self.tree_refs.popitem(last=False)
        self.decoded_trees[cache_key] = result
        if len(self.decoded_trees) > CODEC_TREE_CACHE_ENTRIES:
            self.decoded_trees.popitem(last=False)
        return result

    def state_ref(self, state):
        key = state.key.hex()
        if key not in self.state_refs:
            self.state_refs[key] = self.store.put_json({"kind": "fitted_tree_state",
                "tree": self.tree_ref(state.tree), "tree_sha256": key,
                "leaves": [leaf.cache_ref for leaf in state.leaves],
                "score": list(state.score)})
            if len(self.state_refs) > STATE_CACHE_ENTRIES:
                self.state_refs.popitem(last=False)
        return self.state_refs[key]

    def state(self, ref):
        value = self.store.get_json(ref)
        tree = self.tree(value["tree"])
        regions = list(self.ctx.regions(tree))
        leaf_regions = [region for region in regions if region.tree is None]
        if value.get("kind") != "fitted_tree_state" or len(value["leaves"]) != len(leaf_regions):
            raise CheckpointError("stored fitted state has inconsistent leaves")
        leaves = []
        for region, leaf_ref in zip(leaf_regions, value["leaves"], strict=True):
            data = self.store.get_json(leaf_ref)
            leaf = _Leaf(region, np.asarray(data["model"]["bias"], np.float32),
                data["model"].get("report", {}), data["qualified"], data["expected_mass"],
                data["kl_sum"], data["agreement"])
            leaf.model, leaf.robust = data["model"], data["robust"]
            leaf.cache_key, leaf.cache_ref = data["rows_sha256"], leaf_ref
            leaves.append(leaf)
        key, score = _tree_key(tree), _score(leaves)
        if key.hex() != value["tree_sha256"] or tuple(value["score"]) != score:
            raise CheckpointError("stored fitted state identity/score differs")
        result = _State(tree, leaves, regions, score, key)
        self.state_refs[key.hex()] = ref
        if len(self.state_refs) > STATE_CACHE_ENTRIES:
            self.state_refs.popitem(last=False)
        return result

    def queue_ref(self, queue):
        if queue is None:
            return None
        parents, parent_indices, entries = [], {}, []
        for tree, kind, label, parent in queue:
            key = (parent.state.key.hex(), parent.nonimproving_steps)
            if key not in parent_indices:
                parent_indices[key] = len(parents)
                parents.append([self.state_ref(parent.state), parent.nonimproving_steps])
            entries.append([parent_indices[key], kind, label, self._tree_delta(parent.state.tree, tree)])
        return self.store.put_json({"kind": "ordered_exact_parent_tree_patches.v1",
            "feature_dim": len(self.ctx.names), "parents": parents, "entries": entries})

    @classmethod
    def _tree_delta(cls, base, tree):
        """Exact changed paths, without serializing common parent subtrees.

        None retains a subtree, integer zero makes a leaf, and a three-item
        list holds optional changed split / left patch / right patch. All
        coefficients are stored explicitly; there is no search regeneration.
        """
        if tree is base:
            return None
        if tree is None:
            return 0
        same_split = base is not None and tree.weight == base.weight and tree.bias == base.bias
        split = None if same_split else [float(tree.bias),
            [[i, float(value)] for i, value in enumerate(tree.weight) if value != 0]]
        left = cls._tree_delta(base.left if base is not None else None, tree.left)
        right = cls._tree_delta(base.right if base is not None else None, tree.right)
        return None if split is None and left is None and right is None else [split, left, right]

    def _apply_delta(self, base, patch):
        if patch is None:
            return base
        if type(patch) is int and patch == 0:
            return None
        if not isinstance(patch, list) or len(patch) != 3:
            raise CheckpointError("invalid exact topology patch")
        split, left, right = patch
        if split is None:
            if base is None:
                raise CheckpointError("missing split in a new patched tree node")
            weight, bias = base.weight, base.bias
        else:
            if not isinstance(split, list) or len(split) != 2:
                raise CheckpointError("invalid patched split")
            normalized, seen = [], set()
            for index, value in split[1]:
                if type(index) is not int or not 0 <= index < len(self.ctx.names) or index in seen:
                    raise CheckpointError("invalid patched support")
                seen.add(index)
                coefficient = np.float32(value)
                if not np.isfinite(coefficient):
                    raise CheckpointError("patched split coefficient is not finite")
                normalized.append((index, coefficient))
            if len(seen) > 3 or not math.isfinite(split[0]):
                raise CheckpointError("patched sparse split exceeds its numeric contract")
            normalized.sort(key=lambda pair: pair[0])
            key = (len(self.ctx.names), tuple((i, value.tobytes()) for i, value in normalized))
            weight = self.patch_splits.get(key)
            if weight is None:
                weights = np.zeros(len(self.ctx.names), np.float32)
                for index, value in normalized:
                    weights[index] = value
                weight = tuple(map(float, weights))
                self.patch_splits[key] = weight
                if len(self.patch_splits) > CODEC_TREE_CACHE_ENTRIES:
                    self.patch_splits.popitem(last=False)
            bias = float(split[0])
        return _Decision(weight, bias,
            self._apply_delta(base.left if base is not None else None, left),
            self._apply_delta(base.right if base is not None else None, right))

    def queue(self, ref):
        if ref is None:
            return None
        packed = self.store.get_json(ref)
        if (packed.get("kind") != "ordered_exact_parent_tree_patches.v1"
                or packed.get("feature_dim") != len(self.ctx.names)):
            raise CheckpointError("invalid packed queue schema")
        parents = [_BeamEntry(self.state(parent_ref), debt) for parent_ref, debt in packed["parents"]]
        result = []
        for parent_index, kind, label, patch in packed["entries"]:
            if type(parent_index) is not int or not 0 <= parent_index < len(parents):
                raise CheckpointError("queue patch parent index is invalid")
            parent = parents[parent_index]
            result.append((self._apply_delta(parent.state.tree, patch), kind, label, parent))
        return result


def _bounded_transitions(entries, candidate):
    by_key = {entry.state.key: entry for entry in entries}
    old = by_key.get(candidate.state.key)
    if old is None or candidate.nonimproving_steps < old.nonimproving_steps:
        by_key[candidate.state.key] = candidate
    ordered = sorted(by_key.values(), key=lambda e: (tuple(-v for v in e.state.score), e.state.key,
                                                      e.nonimproving_steps))
    keep = ordered[:4]
    for entry in (e for e in ordered if e.nonimproving_steps > 0):
        if all(entry.state.key != chosen.state.key for chosen in keep):
            keep.append(entry)
        if len(keep) >= TRANSITION_ENTRIES:
            break
    return keep


def fit_structure(mode, encoded, masks, targets, episode_ids, feature_names=None,
                  actor_role_sha256=None, initial_structure=None,
                  initial_structure_sha256=None, budget=UpgradeBudget(),
                  checkpoint_path=None, resume=False):
    if mode not in METHODS or not isinstance(budget, UpgradeBudget):
        raise ValueError("unknown v625 mode or unregistered budget")
    runner_started = time.monotonic()
    initial, _ = normalize_structure(initial_structure)
    if program_structure_sha256(initial) != initial_structure_sha256:
        raise ValueError("saved S1 topology SHA mismatch")
    x, m, p, episodes, names, indices = _validate(encoded, masks, targets, episode_ids,
        feature_names, actor_role_sha256)
    algorithm_budget = asdict(budget)
    algorithm_budget.pop("pause_after_candidates")
    identity = {"mode": mode, "sources": _source_identity(), "budget": algorithm_budget,
        "configuration": search_configuration(), "actor_role_sha256": actor_role_sha256,
        "initial_structure_sha256": initial_structure_sha256, "feature_names": list(names),
        "fit_episode_ids_sha256": digest(list(episodes)),
        **{name: sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
           for name, value in (("encoded", x), ("masks", m), ("targets", p))}}
    temporary = None
    if checkpoint_path is None:
        if resume:
            raise ValueError("resuming requires an explicit checkpoint path")
        temporary = tempfile.TemporaryDirectory(prefix="warehouse-v625-diagnostic-")
        checkpoint_path = Path(temporary.name)/"checkpoint.json"
    session = CheckpointSession(checkpoint_path, identity,
        max_calls=budget.maximum_new_leaf_calls-budget.final_leaf_reserve,
        wall_seconds=budget.wall_seconds-budget.final_wall_reserve, resume=resume,
        started_at=runner_started)
    ctx = _Context(x, m, p, episodes, names, indices, mode, session)
    codec, cfg = _Codec(ctx), SearchConfig()
    states = OrderedDict()
    runtime = {"round": 0, "no_gain": 0, "cursor": 0, "evaluated": 0,
        "expanded": [], "rounds": [], "operation_counts": {}, "stop": None,
        "resume_count": 0, "candidate_order_sha256": "0"*64,
        "queue_ref": None, "before": None, "round_evaluated": 0, "round_failed": 0}
    queue, transitions, beam, start, best = None, [], [], None, None

    def evaluate(tree):
        key = _tree_key(tree)
        if key in states:
            return states[key]
        count, depth = _size(tree)
        if count > 32 or depth > 6:
            raise ValueError("candidate exceeds depth/leaf budget")
        regions = list(ctx.regions(tree))
        leaves = [ctx.fit(region) for region in regions if region.tree is None]
        value = _State(tree, leaves, regions, _score(leaves), key)
        states[key] = value
        if len(states) > STATE_CACHE_ENTRIES:
            states.popitem(last=False)
        return value

    def snapshot_payload():
        return {"schema": "warehouse-v625-search-state.v1", "runtime": plain(runtime),
            "best": codec.state_ref(best), "start": codec.state_ref(start),
            "beam": [[codec.state_ref(entry.state), entry.nonimproving_steps] for entry in beam],
            "transitions": [[codec.state_ref(entry.state), entry.nonimproving_steps] for entry in transitions],
            "states": [codec.state_ref(value) for value in states.values()],
            "leaf_cache": [[key, ref] for key, ref in ctx.cache_refs.items()],
            "leaf_cache_hits": ctx.cache_hits, "leaf_cache_evictions": ctx.cache_evictions,
            "python_rng": plain(random.getstate()),
            "numpy_rng": plain(np.random.get_state())}

    paused = False
    invocation_evaluated = 0
    try:
        with session.hard_deadline():
            if resume:
                saved = session.state
                if saved.get("schema") != "warehouse-v625-search-state.v1":
                    raise CheckpointError("not a v625 search checkpoint")
                runtime = saved["runtime"]
                runtime["resume_count"] += 1
                for key, ref in saved["leaf_cache"]:
                    ctx.cache_refs[key] = ref
                    ctx.cache[key] = session.store.get_json(ref)
                ctx.cache_hits, ctx.cache_evictions = saved["leaf_cache_hits"], saved["leaf_cache_evictions"]
                for ref in saved["states"]:
                    item = codec.state(ref)
                    states[item.key] = item
                best, start = codec.state(saved["best"]), codec.state(saved["start"])
                beam = [_BeamEntry(codec.state(ref), debt) for ref, debt in saved["beam"]]
                transitions = [_BeamEntry(codec.state(ref), debt) for ref, debt in saved["transitions"]]
                queue = codec.queue(runtime["queue_ref"])
                random.setstate(_tuple_tree(saved["python_rng"]))
                n = saved["numpy_rng"]
                np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
            else:
                try:
                    start = best = evaluate(_tree(initial))
                except (BudgetExhausted, ConstrainedLeafFitError) as error:
                    failure_events = session.events()
                    raise SearchIncompleteError("initial tree did not finish within its registered budget",
                        {"schema": "warehouse-v625-search-incomplete.v1", "status": "search_no_complete_incumbent",
                         "mode": mode, "method": METHODS[mode], "new_leaf_calls": session.calls,
                         "error": str(error), "error_type": type(error).__name__,
                         "failure_kind": ("initial_deadline_exhausted" if isinstance(error, DeadlineExceeded)
                             else "initial_leaf_api_exhausted" if isinstance(error, BudgetExhausted)
                             else "initial_solver_infeasible"),
                         "actor_role_sha256": actor_role_sha256,
                         "initial_structure_sha256": initial_structure_sha256,
                         "fit_input_sha256": digest(identity), "fit_target_sha256": identity["targets"],
                         "input_identity": identity, "budget": asdict(budget),
                         "configuration": search_configuration(), "production_configuration": budget.production_configuration,
                         "elapsed_seconds": session.elapsed_seconds,
                         "runner_segment_elapsed_seconds": time.monotonic()-runner_started,
                         "leaf_api_attempts": failure_events, "checkpoint_timing": session.summary(),
                         "actual_convex_solver_calls": _solver_calls(failure_events),
                         "reserved_final_leaf_calls": 32, "reserved_final_wall_seconds": 600,
                         "complete_incumbent": False, "search_uses_fit_only": True,
                         "calibration_eligibility_validation_used": False,
                         "zero_environment_resets": True, "zero_actor_updates": True,
                         "execution_action_override": False}) from error
                beam = [_BeamEntry(start)]
            last_checkpoint = time.monotonic()
            while runtime["stop"] is None and runtime["round"] < budget.maximum_rounds:
                try:
                    ctx.clock.check()
                    recipe_id = runtime["round"] % len(SEARCH_INITIALIZATIONS)
                    if queue is None:
                        ctx.phase = "proposal"
                        offered = OrderedDict()
                        expanded = set(map(tuple, runtime["expanded"]))
                        for entry in beam:
                            marker = (entry.state.key.hex(), recipe_id)
                            if marker in expanded:
                                continue
                            expanded.add(marker)
                            for tree, kind, label, _ in _proposal_trees(ctx, entry.state,
                                    SEARCH_INITIALIZATIONS[recipe_id], "S1", cfg, indices):
                                offered.setdefault(_tree_key(tree), (tree, kind, label, entry))
                        queue = list(offered.values())
                        runtime.update(expanded=[list(marker) for marker in sorted(expanded)],
                            cursor=0, queue_ref=codec.queue_ref(queue), before=list(best.score),
                            round_evaluated=0, round_failed=0)
                        transitions = []
                    while runtime["cursor"] < len(queue):
                        ctx.clock.check()
                        tree, kind, label, parent = queue[runtime["cursor"]]
                        ctx.phase = "candidate"
                        try:
                            fitted = evaluate(tree)
                        except ConstrainedLeafFitError:
                            runtime["round_failed"] += 1
                        else:
                            runtime["round_evaluated"] += 1
                            debt = 0 if _improves(fitted.score, parent.state.score) else parent.nonimproving_steps+1
                            if debt <= cfg.max_nonimproving_steps:
                                transitions = _bounded_transitions(transitions, _BeamEntry(fitted, debt))
                            if (_improves(fitted.score, best.score)
                                    or (fitted.score == best.score and fitted.key < best.key)):
                                best = fitted
                        counts = runtime["operation_counts"]
                        counts[kind] = counts.get(kind, 0)+1
                        runtime["candidate_order_sha256"] = digest({"prior": runtime["candidate_order_sha256"],
                            "tree": _tree_key(tree).hex(), "kind": kind, "label": label,
                            "parent": parent.state.key.hex()})
                        runtime["cursor"] += 1
                        runtime["evaluated"] += 1
                        invocation_evaluated += 1
                        paused = bool(budget.pause_after_candidates
                                      and invocation_evaluated >= budget.pause_after_candidates)
                        if paused:
                            break
                        if (runtime["evaluated"] % CHECKPOINT_CANDIDATES == 0
                                or time.monotonic()-last_checkpoint >= CHECKPOINT_SECONDS):
                            session.snapshot(snapshot_payload())
                            last_checkpoint = time.monotonic()
                    if paused:
                        break
                    improved = _improves(best.score, runtime["before"])
                    runtime["no_gain"] = 0 if improved else runtime["no_gain"]+1
                    beam = _choose_beam(transitions, best, budget.beam_width)
                    runtime["rounds"].append({"round": runtime["round"]+1, "recipe": recipe_id,
                        "evaluated": runtime["round_evaluated"], "failed": runtime["round_failed"],
                        "improved": improved, "new_leaf_calls": session.calls, "best_score": list(best.score)})
                    runtime["round"] += 1
                    runtime["queue_ref"] = None
                    queue = None
                    if runtime["no_gain"] >= len(SEARCH_INITIALIZATIONS)*budget.stagnant_recipe_cycles:
                        runtime["stop"] = "three_complete_recipe_cycles_without_improvement"
                except BudgetExhausted as error:
                    if session.pending:
                        session.finish(success=False, detail={"error_type": type(error).__name__, "error": str(error)})
                    runtime["stop"] = "wall_budget" if isinstance(error, DeadlineExceeded) else "leaf_call_budget"
            if not paused and runtime["stop"] is None:
                runtime["stop"] = "maximum_rounds"
            result = _export(best.tree, best.leaves, len(names))
            result["feature_dim"] = len(names)
            result["initial_leaf_models"] = [plain(leaf.model) for leaf in best.leaves]
            # This field remains a bias-only compatibility view; programs must
            # use initial_leaf_models for conditional predictions and final fits.
            result["initial_leaf_logits_scope"] = "leaf_bias_only_not_complete_P1_prediction"
            session.commit(snapshot_payload(), status="paused" if paused else "completed")
        events = session.events()
        summary = session.summary()
        runner_elapsed = time.monotonic()-runner_started
        fraction_numerator = summary["checkpoint_total_io_upper_bound_seconds"]
        fraction_denominator = summary["actual_elapsed_lower_bound_seconds"] if resume else runner_elapsed
        fraction = fraction_numerator/fraction_denominator
        report = {"schema": "warehouse-v625-fit-search.v1", "mode": mode, "method": METHODS[mode],
            "actor_role_sha256": actor_role_sha256, "initial_structure_sha256": initial_structure_sha256,
            "fit_input_sha256": digest(identity), "fit_target_sha256": identity["targets"],
            "configuration": search_configuration(), "menu_configuration_sha256": digest(search_configuration()),
            "production_configuration": budget.production_configuration, "budget": asdict(budget),
            "new_leaf_calls": session.calls, "maximum_new_leaf_calls_including_final_reserve": budget.maximum_new_leaf_calls,
            "actual_convex_solver_calls": _solver_calls(events),
            "reserved_final_leaf_calls": 32, "reserved_final_wall_seconds": 600,
            "elapsed_seconds": summary["elapsed_upper_bound_seconds"],
            "runner_segment_elapsed_seconds": runner_elapsed,
            "checkpoint_io_seconds": fraction_numerator,
            "checkpoint_fraction_numerator_seconds": fraction_numerator,
            "checkpoint_fraction_denominator_seconds": fraction_denominator,
            "checkpoint_fraction_scope": ("resumed_cumulative_io_upper_over_actual_lower_bound" if resume
                                          else "fresh_measured_segment"),
            "checkpoint_fraction": fraction,
            "checkpoint_timing": summary,
            "checkpoint_overhead_at_most_ten_percent": fraction <= .10,
            "leaf_api_attempts": events, "leaf_cache_hits": ctx.cache_hits,
            "leaf_cache_evictions": ctx.cache_evictions, "leaf_cache_entries": len(ctx.cache),
            "stop_reason": "paused_safe_boundary" if paused else runtime["stop"],
            "best_score": list(best.score), "initial_score": list(start.score),
            "best_tree_sha256": best.key.hex(), "leaf_count": result["leaf_count"],
            "depth": _size(best.tree)[1], "complete_incumbent": True,
            "search_uses_fit_only": True, "calibration_eligibility_validation_used": False,
            "execution_action_override": False, "zero_actor_updates": True, "zero_environment_resets": True,
            "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0,
            "checkpoint_resume_equivalent": True, "checkpoint_equivalence_scope": "explicit_safe_pause_only",
            "unsafe_interruption_fails_closed": True, "resume_count": runtime["resume_count"],
            "operation_counts": runtime["operation_counts"], "evaluated_candidates": runtime["evaluated"],
            "candidate_order_sha256": runtime["candidate_order_sha256"], "rounds": runtime["rounds"],
            "heuristic_not_global_optimality_proof": True}
        return result, plain(report)
    finally:
        if temporary is not None:
            temporary.cleanup()
