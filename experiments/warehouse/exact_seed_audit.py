"""Exact, typed seed evidence. This module never authorizes an experiment.

It can project a completed broader metadata audit onto exact arguments only
when every freshness-sensitive value was in that audit's declared scope. Fixed
algorithm RNG/hash arguments are reported as declared controls, never as fresh
scenes. A final filename/stat delta is a separate required step.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from . import seed_metadata_audit as metadata

FRESH_KINDS = {"init", "env", "scenario", "action_rng", "critic", "sealed"}
FIXED_KINDS = {"torch_rng", "classifier_rng", "hash_argument"}
BC_STAGES = (
    ("initial_public_teacher", 100, "teacher"),
    ("initial_nn_dagger_1", 40, "sampled"), ("initial_nn_dagger_2", 40, "sampled"),
    ("joint_augmentation_sampled", 25, "sampled"), ("joint_augmentation_argmax", 25, "argmax"),
    ("risk_public_teacher", 40, "teacher"), ("risk_nn_visitation", 80, "sampled"),
    ("large_public_teacher", 200, "teacher"), ("large_nn_visitation", 120, "sampled"),
    ("deep_public_teacher", 200, "teacher"),
    *((f"deep_{i}_{mode}", 50, mode) for i in range(1, 5) for mode in ("sampled", "argmax")),
)


def read_exact_plan(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    plan = json.loads(raw)
    if (plan.get("schema") != "warehouse-exact-seed-plan.v1"
            or not isinstance(plan.get("candidate_id"), str) or not plan["candidate_id"]
            or not isinstance(plan.get("uses"), list) or not plan["uses"]
            or plan.get("authorization_granted") is not False):
        raise ValueError("invalid exact seed plan or nonzero authority")
    groups = set()
    for row in plan["uses"]:
        allowed = {"kind", "purpose", "argument", "value", "group", "reuse_tag", "consumed_as_random_draw"}
        if (not isinstance(row, dict) or set(row) - allowed
                or not {"kind", "purpose", "argument", "value", "group", "reuse_tag"} <= set(row)
                or row.get("kind") not in FRESH_KINDS | FIXED_KINDS
                or type(row.get("value")) is not int or not 0 <= row["value"] < 2**63
                or any(not isinstance(row.get(k), str) or not row[k] for k in ("purpose", "argument", "group", "reuse_tag"))
                or ("consumed_as_random_draw" in row and type(row["consumed_as_random_draw"]) is not bool)):
            raise ValueError("malformed typed exact seed use")
        key = (row["kind"], row["group"], row["purpose"], row["argument"])
        if key in groups:
            raise ValueError("duplicate exact argument call identity")
        groups.add(key)
    # Reusing the initialization argument to verify its role SHA and matched
    # gamma training is declared explicitly. No cross-purpose fresh scene or
    # action sharing can hide behind a generic reuse tag.
    used = defaultdict(list)
    for row in plan["uses"]:
        if row["kind"] in FRESH_KINDS:
            used[(row["kind"] == "init", row["value"])].append(row)
    for rows in used.values():
        if len(rows) < 2:
            continue
        if len({r["reuse_tag"] for r in rows}) != 1:
            raise ValueError("unmatched exact fresh argument reuse")
        if not all(r["kind"] == "init" for r in rows) and len({(r["kind"], r["purpose"]) for r in rows}) != 1:
            raise ValueError("fresh seed shared across purposes")
    return plan, {"path": str(path), "sha256": metadata.file_sha(path)}


def validate_bc_pilot_plan(plan, paired_plan):
    """Independently expand the registered 6160 pilot; no trainer import."""
    if (paired_plan.get("schema") != "warehouse-v620-paired-bc-seeds.v1"
            or paired_plan.get("candidate_id") != plan["candidate_id"]
            or paired_plan.get("pair_seed") != 6160
            or paired_plan.get("gamma_candidates") != [.01, .03]):
        raise ValueError("paired BC pilot identity differs")
    binding = paired_plan["seed_binding"]
    if (set(binding) != {"bc_seed_schedule", "critic_environment_reset_seeds", "critic_seed", "initialization_pair_seed", "training_rng_schedule"}
            or plan.get("seed_binding_sha256") != metadata.digest(binding)
            or binding["initialization_pair_seed"] != 6160
            or binding["critic_seed"] != 1810000000
            or binding["critic_environment_reset_seeds"] != [1810000000, 1810000000]):
        raise ValueError("paired BC binding/reset identity differs")
    expected_rng = {
        "actor_initialization_torch": 6160, "bc_global_torch_seed_reset": 6160,
        "initial_dagger_augmentation_torch_shuffle_shared_generator": 6191,
        "risk_torch_shuffle": 6361, "large_torch_shuffle": 6461,
        "deep_torch_shuffle": [6561, 6562, 6563, 6564], "final_torch_shuffle": 6659,
        "initial_public_tree_classifier_random_states": [19, 20],
        "critic_initialization_torch": 1810000000,
        "critic_minibatch_torch_shuffle": 1810000001,
        "critic_unordered_pair_split_hash_argument": 1810000000,
    }
    if binding["training_rng_schedule"] != expected_rng:
        raise ValueError("exact Torch/classifier/hash argument declaration differs")
    expected = []
    def item(kind, purpose, argument, value, draw=None):
        result = (kind, purpose, argument, value)
        expected.append(result + (draw,))
    item("init", "training_actor_initialization", "torch.manual_seed", 6160)
    item("init", "authorization_initial_role_verification", "torch.manual_seed", 6160)
    schedule, mapping = binding["bc_seed_schedule"], plan.get("stage_mapping")
    if set(schedule) != {name for name, _, _ in BC_STAGES} or not isinstance(mapping, list) or len(mapping) != len(BC_STAGES):
        raise ValueError("exact BC stage inventory differs")
    for index, (name, count, mode) in enumerate(BC_STAGES):
        scenes, actions = 616010000 + index*1000, 1740010000 + index*1000
        expected_cases = [{"scenario_seed": scenes+i, "action_seed": actions+i} for i in range(count)]
        if (schedule[name] != expected_cases or mapping[index] != {
                "index": index, "stage": name, "episode_count": count, "sampling": mode,
                "scenario_start": scenes, "scenario_end_exclusive": scenes+count,
                "action_start": actions, "action_end_exclusive": actions+count,
                "action_arguments_consumed_as_random_draws": mode == "sampled"}):
            raise ValueError("exact BC stage scenarios/actions/modes differ")
        for case in expected_cases:
            item("env", name, "reset(seed)", case["scenario_seed"])
            item("action_rng", name, "declared_action_base_seed", case["action_seed"], mode == "sampled")
    for _ in range(2):
        item("critic", "collision_critic_environment", "reset(seed)", 1810000000)
    for purpose, value in expected_rng.items():
        if purpose == "actor_initialization_torch":
            continue  # the real constructor and role verification above
        values = value if isinstance(value, list) else [value]
        for argument_value in values:
            kind = "classifier_rng" if "classifier" in purpose else "hash_argument" if "hash_argument" in purpose else "torch_rng"
            argument = ("classifier.random_state" if kind == "classifier_rng" else
                        "sha256_position_pair_partition_seed_argument" if kind == "hash_argument" else
                        "torch.manual_seed" if purpose in ("bc_global_torch_seed_reset", "critic_initialization_torch") else "torch.Generator.manual_seed")
            item(kind, purpose, argument, argument_value)
    actual = [(r["kind"], r["purpose"], r["argument"], r["value"], r.get("consumed_as_random_draw")) for r in plan["uses"]]
    if Counter(actual) != Counter(expected):
        raise ValueError("exact uses table does not equal the paired actual BC call/argument plan")
    return {"BC_scene_calls_per_gamma": 1270, "critic_scene_calls_per_gamma": 2,
        "total_environment_resets_per_gamma": 1272, "unique_physical_scenes": 1271,
        "declared_action_arguments": 1270,
        "stochastic_action_arguments": sum(count for _, count, mode in BC_STAGES if mode == "sampled"),
        "teacher_or_argmax_reserved_action_arguments": sum(count for _, count, mode in BC_STAGES if mode != "sampled"),
        "typed_uses": len(actual), "gamma_reuse": "matched_scene_action_stream_not_independent_evidence",
        "fixed_rng_or_hash_not_claimed_fresh": True}


def compile_families(plan):
    grouped = defaultdict(set)
    for use in plan["uses"]:
        if use["kind"] in FRESH_KINDS:
            # The actor constructor and initial role SHA verification consume
            # the same explicitly matched initialization argument.
            purpose = "actor_initialization" if use["kind"] == "init" else use["purpose"]
            grouped[(use["kind"], purpose, False)].add(use["value"])
        else:
            grouped[("init", "declared_fixed_rng_and_hash_arguments", True)].add(use["value"])
    families = []
    for (kind, purpose, query), values in sorted(grouped.items()):
        families.append({"name": kind + ":" + purpose, "kind": kind, "values": sorted(values),
            "intervals": metadata.value_intervals(values), "count": len(values), "query_only": query})
    return families


def project_inventory(inventory, families):
    """Lossless within original numeric scope; does not invent missing reads."""
    uncovered = []
    for family in families:
        for value in family["values"]:
            if not any((old["kind"] == "init") == (family["kind"] == "init")
                       and any(start <= value < end for start, end in old["intervals"])
                       for old in inventory["families"]):
                uncovered.append({"family": family["name"], "value": value, "query_only": family["query_only"]})
    fresh_uncovered = [r for r in uncovered if not r["query_only"]]
    if fresh_uncovered:
        raise RuntimeError("fresh exact values outside the bounded audit scope: " + json.dumps(fresh_uncovered[:10]))
    rows = []
    for original in inventory["inventory"]:
        row = deepcopy(original)
        detail = row.get("detail", {})
        # Historical draft ledgers discarded state-like keys heuristically.
        # Restore those exact numeric references before applying the stricter
        # source-backed context layer. Explicit hash/count metadata stays in
        # the ignored list and remains visible as evidence.
        legacy_state = [m for m in detail.get("ignored_internal_or_nonseed_matches", [])
                        if m["semantics"].startswith("internal_rng_")]
        detail["candidate_matches"] = detail.get("candidate_matches", []) + [
            {**m, "semantics": "ambiguous_internal_rng_state_requires_context_proof"} for m in legacy_state]
        detail["ignored_internal_or_nonseed_matches"] = [m for m in detail.get("ignored_internal_or_nonseed_matches", [])
                                                         if not m["semantics"].startswith("internal_rng_")]
        for key in ("candidate_matches", "ignored_internal_or_nonseed_matches"):
            matches, seen = [], set()
            for match in detail.get(key, []):
                names = metadata.matching_families(match["value"], families)
                identity = (tuple(match["path"]), match["value"], match["semantics"])
                if names and identity not in seen:
                    matches.append({**match, "families": names})
                    seen.add(identity)
            detail[key] = matches
        spans, seen = [], set()
        for span in detail.get("reservation_overlaps", []):
            start, end = span["historical_interval"]["start"], span["historical_interval"]["end_exclusive"]
            for family in families:
                if not metadata.compatible(family["kind"], span["semantics"]):
                    continue
                for lo, hi in family["intervals"]:
                    if max(lo, start) >= min(hi, end):
                        continue
                    current = {**span, "family": family["name"], "overlaps": [[max(lo, start), min(hi, end)]]}
                    identity = (tuple(current["path"]), current["family"], start, end, tuple(current["overlaps"][0]))
                    if identity not in seen:
                        spans.append(current)
                        seen.add(identity)
        detail["reservation_overlaps"] = spans
        row["detail"] = detail
        rows.append(row)
    projected = {**inventory, "families": families, "inventory": rows,
        "unscanned_fixed_argument_values_not_claimed_fresh": uncovered,
        "exact_projection_is_not_a_new_history_scan": True}
    projected["summary"] = metadata.summarize(rows, families, selection_stats=inventory.get("selection_stats"),
                                             missing_roots=inventory.get("missing_roots", ()))
    return projected


def apply_context(inventory, annotation):
    """Use exact original file SHA/path/value identities; no blanket key rules."""
    from . import seed_context
    if (annotation.get("input_inventory_digest") != metadata.digest(inventory)
            or annotation.get("context_producer_sha256") != metadata.file_sha(seed_context.__file__)):
        raise RuntimeError("context annotations differ from the exact inventory or approved producer")
    proven = {}
    for row in annotation["records"]:
        if row["context_status"].startswith("confirmed_"):
            key = (row["file"], row["file_sha256"], tuple(row["path"]), row["value"])
            if not row.get("context_proof", {}).get("source_keys"):
                raise ValueError("context exclusion missing producer source proof")
            for name in row["context_proof"]["source_keys"]:
                source = annotation["source_evidence"][name]
                if metadata.file_sha(source["path"]) != source["sha256"]:
                    raise RuntimeError("context producer source evidence changed")
            proven[key] = row
    result = deepcopy(inventory)
    excluded = []
    for row in result["inventory"]:
        detail = row.get("detail", {})
        kept = []
        for match in detail.get("candidate_matches", []):
            key = (row["path"], row.get("sha256"), tuple(match["path"]), match["value"])
            if key in proven:
                excluded.append(proven[key])
            else:
                kept.append(match)
        detail["candidate_matches"] = kept
    result["summary"] = metadata.summarize(result["inventory"], result["families"],
                                            selection_stats=result.get("selection_stats"),
                                            missing_roots=result.get("missing_roots", ()))
    result["contextually_excluded_original_matches"] = excluded
    return result


def delta_inventory(baseline, proposal_path, *, output_directory):
    """New/changed/removed metadata blocks final negative conclusions."""
    output = Path(output_directory)
    selection = metadata.inventory_only(proposal_path, [Path(r) for r in baseline["scan_roots"]], output)
    old = {r["path"]: r for r in baseline["inventory"] if r["classification"] != "source_template_environment_reservation"}
    now = {r["path"]: r for r in selection["candidates"]}
    changed = [path for path in old.keys() & now.keys()
        if (old[path]["bytes"], old[path].get("mtime_ns")) != (now[path]["bytes"], now[path]["mtime_ns"])]
    old_weights = {r["path"]: r for r in baseline.get("selection_stats", {}).get("excluded_weights_name_stat_only", [])}
    now_weights = {r["path"]: r for r in selection["selection_stats"].get("excluded_weights_name_stat_only", [])}
    changed_weights = [p for p in old_weights.keys() & now_weights.keys()
        if (old_weights[p]["bytes"], old_weights[p]["mtime_ns"]) != (now_weights[p]["bytes"], now_weights[p]["mtime_ns"])]
    delta = {"schema": "warehouse-seed-metadata-delta.v1", "authorization_granted": False,
        "new_paths": sorted(now.keys() - old.keys()), "removed_paths": sorted(old.keys() - now.keys()),
        "changed_paths": sorted(changed), "requires_new_content_audit": bool(changed or old.keys() != now.keys()),
        "new_opaque_weight_paths_name_stat_only": sorted(now_weights.keys() - old_weights.keys()),
        "removed_opaque_weight_paths_name_stat_only": sorted(old_weights.keys() - now_weights.keys()),
        "changed_opaque_weight_paths_name_stat_only": sorted(changed_weights),
        "requires_weight_path_receipt_review": bool(changed_weights or old_weights.keys() != now_weights.keys()),
        "selection": {"path": str((output/"selection.json").resolve()), "sha256": metadata.file_sha(output/"selection.json")},
        "scope": "filename/stat only; no content negative proof and no model reads"}
    metadata.write_new(output/"delta.json", delta)
    return delta


def assess(plan_path, inventory_path, output, *, annotation_path=None, paired_plan_path=None,
           context_source_root=None):
    plan, identity = read_exact_plan(plan_path)
    raw = Path(inventory_path).read_bytes()
    inventory = json.loads(raw)
    if (inventory.get("selection_scope") != metadata.SELECTION_SCOPE
            or inventory.get("read_model_weight_files") is not False):
        raise ValueError("unrecognized or unsafe metadata inventory scope")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = project_inventory(inventory, compile_families(plan))
    if context_source_root is not None:
        if annotation_path is not None:
            raise ValueError("choose source-backed context generation or an existing annotation, not both")
        from .seed_context import annotations
        generated = annotations(result, source_root=context_source_root, output_directory=output/"context")
        result = apply_context(result, generated)
        result["context_annotations"] = {"path": str((output/"context/annotations.json").resolve()),
                                           "sha256": metadata.file_sha(output/"context/annotations.json")}
    if paired_plan_path:
        paired = json.loads(Path(paired_plan_path).read_bytes())
        result["exact_bc_plan_validation"] = validate_bc_pilot_plan(plan, paired)
        result["paired_plan"] = {"path": str(Path(paired_plan_path).resolve()), "sha256": metadata.file_sha(paired_plan_path)}
    if annotation_path:
        annotation = json.loads(Path(annotation_path).read_bytes())
        result = apply_context(result, annotation)
        result["context_annotations"] = {"path": str(Path(annotation_path).resolve()), "sha256": metadata.file_sha(annotation_path)}
    result.update({"schema": "warehouse-exact-seed-evidence.v1", "exact_plan": identity,
        "bounded_or_triaged_inventory": {"path": str(Path(inventory_path).resolve()), "sha256": metadata.file_sha(inventory_path)},
        "exact_auditor_sha256": metadata.file_sha(__file__), "authorization_granted": False,
        "environment_resets": 0, "fit_calls": 0, "preliminary_only": True,
        "requires_final_delta_inventory": True,
        "declared_fixed_rng_and_hash_uses": [r for r in plan["uses"] if r["kind"] in FIXED_KINDS]})
    if metadata.file_sha(plan_path) != identity["sha256"] or Path(inventory_path).read_bytes() != raw:
        raise RuntimeError("exact seed evidence inputs changed")
    metadata.write_new(output/"result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--paired-plan", type=Path)
    parser.add_argument("--context-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assess(args.plan, args.inventory, args.output_dir,
                    annotation_path=args.annotations, paired_plan_path=args.paired_plan,
                    context_source_root=args.context_source_root)
    print(json.dumps({"result": str(args.output_dir/"result.json"), "authorization_granted": False,
        "unresolved": len(result["summary"]["unresolved_paths"]),
        "fixed_values_outside_prior_scan_scope": len(result["unscanned_fixed_argument_values_not_claimed_fresh"])}))
