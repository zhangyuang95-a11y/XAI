#!/usr/bin/env python3
"""Replay saved S1 leaf logits to check exact caches/routes without optimization.

This is an engineering check on already spent fit rows. It loads no validation
rows and performs no convex solve, environment reset or Actor update. Replayed
solutions must never be imported into a scientific search cache.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def statistics(leaf):
    return {"logits": leaf.logits.tolist(), "qualified": leaf.qualified,
            "expected_mass": leaf.expected_mass, "kl_sum": leaf.kl_sum,
            "agreement": leaf.agreement, "robust": leaf.robust}


def region_identity(regions):
    return [(r.path, r.legal, sha256(np.asarray(r.rows, dtype="<i8").tobytes()).hexdigest())
            for r in regions]


def verify(*, repeats=10):
    from backend.training import warehouse_v622_search as legacy
    from backend.training import warehouse_v624_search as upgrade
    from backend.training.warehouse_v619_tree import _Decision, _State, _replace, _tree_key
    from backend.training.warehouse_v6_tree_routing import explicit_mask_branches
    from experiments.warehouse import v624_protocol as v
    from experiments.warehouse import v622_recovery_protocol as old
    from scripts.run_warehouse_v622_capacity import arrays

    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("repeats must be between 1 and 100")
    sources = v.source_files()
    _, initials = v.checked_baseline()
    actor, _ = old.checked_actor(ROOT)
    episodes, manifest = old.checked_inputs(ROOT)
    rows = []
    for fold in range(3):
        for role in v.ROLES:
            job_id = f"fold-{fold}-{role}"
            parts = old.grouped_folds()[fold]["splits"]["fit"]
            fit_rows = [row for case in parts for row in episodes[case["base"]][role]]
            x, masks, p, ep = arrays(actor, role, fit_rows)
            ep = np.asarray(ep)
            names = tuple(actor.roles[role].feature_names)
            indices = tuple(names.index("legal_"+a) for a in ("up", "down", "left", "right", "wait"))
            structure = v.bound(initials[job_id]["structure"])
            tree = upgrade._tree(structure)
            contexts = [klass(x, masks, p, ep, names, indices, legacy._Clock(120), 100)
                        for klass in (legacy._FitContext, upgrade._ExactContext)]
            regions = [list(ctx.regions(tree)) for ctx in contexts]
            if region_identity(regions[0]) != region_identity(regions[1]):
                raise RuntimeError("initial routing differs")
            leaves = []
            replay_calls = 0
            for leaf_index, (a, b) in enumerate(zip(
                    (r for r in regions[0] if r.tree is None),
                    (r for r in regions[1] if r.tree is None), strict=True)):
                logits = np.asarray(structure["initial_leaf_logits"][leaf_index], np.float32)
                def replay_solution(*args, chosen=logits, **kwargs):
                    nonlocal replay_calls
                    replay_calls += 1
                    return chosen.copy(), {"success": True, "enumerated_action_solves": 0,
                                           "engineering_saved_solution_replay": True}
                with patch.object(legacy, "fit_constrained_leaf_distribution", replay_solution):
                    left, right = contexts[0].fit(a), contexts[1].fit(b)
                if statistics(left) != statistics(right):
                    raise RuntimeError("complete leaf statistics differ")
                leaves.append(right)
            timings = []
            for ctx, rr in zip(contexts, regions, strict=True):
                start = time.monotonic()
                for _ in range(repeats):
                    for r in rr:
                        if r.tree is None:
                            ctx.fit(r)
                timings.append(time.monotonic()-start)
            state = _State(tree, leaves, regions[1], legacy._leaves_score(leaves), _tree_key(tree))
            unchanged = list(upgrade._regions(contexts[1], tree, state))
            if region_identity(unchanged) != region_identity(regions[0]):
                raise RuntimeError("subtree reuse changed routing")
            semantic = next(r for r in regions[1] if r.tree is not None
                and not any(explicit_mask_branches(np.asarray(r.tree.weight), r.tree.bias, indices)))
            changed = _replace(tree, semantic.path, _Decision(semantic.tree.weight,
                float(np.float32(semantic.tree.bias+.01)), semantic.tree.left, semantic.tree.right))
            if region_identity(list(upgrade._regions(contexts[1], changed, state))) != region_identity(list(contexts[0].regions(changed))):
                raise RuntimeError("changed-subtree incremental routing differs")
            if contexts[0].solves != 0 or contexts[1].solves != 0:
                raise RuntimeError("engineering replay unexpectedly invoked a solver")
            rows.append({"job_id": job_id, "fit_frames": len(x), "leaves": len(leaves),
                "exact_statistics": True, "unchanged_and_changed_routing_exact": True,
                "replayed_API_wrappers": replay_calls, "actual_convex_solves": 0,
                "cache_hit_repetitions": repeats,
                "legacy_cache_hit_seconds": timings[0], "complete_cache_hit_seconds": timings[1],
                "complete_cache_hits": contexts[1].full_cache_hits,
                "routing_reused_regions": contexts[1].routing_reused_regions})
    if sources != v.source_files():
        raise RuntimeError("source changed during engineering check")
    return {"schema": "warehouse-v624-engineering-replay.v1", "passed": True,
        "candidate_id": v.CANDIDATE, "source_sha256": v.digest(sources),
        "verifier": v.pointer(__file__), "actor_sha256": old.ACTOR_SHA256,
        "input_manifest": manifest, "jobs": rows,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "actual_leaf_optimization_calls": 0, "actual_convex_solver_calls": 0,
        "replayed_solutions_only": True, "scientific_search_cache_exported": False,
        "scientific_precheck_passed": False, "pilot_opened": False,
        "timing_scope": "warm cache hits only; not a whole-search speedup estimate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "jobs": len(result["jobs"]),
                      "actual_leaf_optimization_calls": 0, "receipt": str(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
