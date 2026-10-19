#!/usr/bin/env python3
"""Exact v6.21 BC seed metadata evidence; never grants reset authority.

This is a new generic expander for the registered three-initializer recipes;
it does not invoke the old 6160-only expansion. Large JSON, YAML and inert
opaque metadata are triaged completely by the existing read-only tools. Only
verified producer-backed MT state words/counters may be contextual exclusions.
A final metadata/weight-name delta and human-readable limited-scope review are
still required before a separate source-bound capability clearance is issued.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.warehouse import v621_bc_protocol as protocol
from experiments.warehouse import v621_protocol as diagnostics
from experiments.warehouse import seed_metadata_audit as metadata
from experiments.warehouse import exact_seed_audit as exact


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    metadata.write_new(Path(path), value)


def exact_plan():
    """Typed fresh training arguments plus separately declared spent reuse."""
    seeds = protocol.seed_plan()
    uses = []
    def item(kind, purpose, argument, value, group, tag, consumed=None):
        row = {"kind": kind, "purpose": purpose, "argument": argument, "value": value,
               "group": group, "reuse_tag": tag}
        if consumed is not None:
            row["consumed_as_random_draw"] = consumed
        uses.append(row)
    for initialization in protocol.INITIALIZATIONS:
        registered = seeds["initializations"][str(initialization)]
        prefix = f"init-{initialization}"
        for recipe in protocol.RECIPES:
            # The shared architecture factory first constructs the matched
            # additive primary and then its own tree/residual model.
            for constructor in range(1 if recipe == "additive_control" else 2):
                item("init", "actor_initialization", "torch.manual_seed", initialization,
                     f"{prefix}/{recipe}/constructor-{constructor}", f"matched-actor-init-{initialization}")
        for index, (stage, count) in enumerate(protocol.STAGES):
            teacher = stage in protocol.TEACHER_STAGES
            recipes = ("shared_cache",) if teacher else protocol.RECIPES
            sampled = not teacher and not stage.endswith("argmax")
            for recipe in recipes:
                for episode_index, case in enumerate(registered["stages"][stage]):
                    group = f"{prefix}/{recipe}/{stage}/{episode_index}"
                    tag = f"matched-{initialization}-{stage}-{episode_index}"
                    item("env", stage, "reset(seed)", case["scenario_seed"], group+"/reset", tag)
                    item("action_rng", stage, "declared_action_base_seed", case["action_seed"],
                         group+"/action", tag, sampled)
        critic = registered["critic_seed"]
        for index in range(2):
            item("critic", "shared_collision_critic_environment", "reset(seed)", critic,
                 f"{prefix}/shared_cache/critic-reset-{index}", f"matched-shared-critic-{initialization}")
        for name, value in registered["rng"].items():
            if name == "shared_tree_search_initialization":
                continue
            values = value if isinstance(value, list) else [value]
            scopes = (protocol.RECIPES if name == "critic_cache_load_torch" else
                ("shared_cache",) if name.startswith("critic_") else protocol.RECIPES)
            for recipe in scopes:
                for index, scalar in enumerate(values):
                    kind = "classifier_rng" if "classifier" in name else "hash_argument" if "hash_argument" in name else "torch_rng"
                    argument = ("DecisionTreeClassifier.random_state" if kind == "classifier_rng" else
                        "sha256_position_pair_partition_seed_argument" if kind == "hash_argument" else
                        "torch.manual_seed" if name in ("critic_torch_initialization", "critic_cache_load_torch") else "torch.Generator.manual_seed")
                    item(kind, name, argument, scalar, f"{prefix}/{recipe}/{name}/{index}", f"declared-control-{name}-{scalar}")
    return {"schema": "warehouse-exact-seed-plan.v1", "candidate_id": diagnostics.CANDIDATE_ID,
        "batch_id": protocol.BATCH, "uses": uses, "authorization_granted": False,
        "registered_seed_plan_sha256": metadata.digest(seeds),
        "registered_recipe_sha256": metadata.digest(protocol.recipe()),
        "spent_development_reuse": {"freshness_claimed": False, "cases": seeds["development_evaluation"]["cases"],
            "repeated_per_initialization_recipe": list(protocol.RECIPES),
            "initializations": list(protocol.INITIALIZATIONS),
            "purpose": "already_consumed_v620_275_scene_and_action_arguments_development_only",
            "excluded_from_fresh_absence_claim": True},
        "backup_conditional_not_automatically_authorized": True,
        "new_physical_training_arguments_include_teacher_reserved_action_bases": True,
        "fixed_rng_or_hash_arguments_not_claimed_fresh": True,
        "environment_resets": 0, "fit_calls": 0, "PPO_joint_steps": 0, "publishable": False}


def validate_exact(path):
    plan, identity = exact.read_exact_plan(path)
    if plan != exact_plan():
        raise RuntimeError("v6.21 exact uses differ from generic registered protocol expansion")
    counts = Counter(row["kind"] for row in plan["uses"])
    return {"exact_plan": identity, "typed_use_counts": dict(counts),
        "new_training_environment_calls_maximum": counts["env"]+counts["critic"],
        "shared_cache_resets_per_initialization": 542,
        "model_training_visits_per_recipe_initialization": 730,
        "spent_development_resets_per_recipe_initialization": 275,
        "conditional_maximum_all_scopes_resets": 10671,
        "authorization_granted": False, "PPO_joint_steps": 0}


def _registered_traces(root, previous_inventory=None):
    result = set()
    if previous_inventory:
        old = json.loads(Path(previous_inventory).read_bytes())
        result.update(old.get("registered_trace_paths", []))
    # Explicitly register all four content-addressed traces in the closed
    # cohort. No arbitrary per-frame payload folder is unconditionally opened.
    directory = root/diagnostics.DATA_ROOT
    for split in diagnostics.SPLITS:
        receipt = json.loads((directory/f"collection/{split}.json").read_bytes())
        path = Path(receipt["trace"]["path"]).resolve(strict=True)
        if file_sha(path) != receipt["trace"]["sha256"]:
            raise RuntimeError("closed development trace changed before seed inventory")
        result.add(str(path))
    return sorted(result)


def prepare(root, output, *, previous_inventory=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = exact_plan()
    exact_path = output/"exact-uses.json"
    write(exact_path, plan)
    checked = validate_exact(exact_path)
    # Query fixed algorithm RNG/hash controls only when their numeric values
    # do not explode the index/frame query scope. Unscanned fixed 19/20 values
    # are explicitly reported by exact projection and never called fresh.
    expanded = exact.compile_families(plan)
    # Broad intervals are a strict superset for the metadata read only. Exact
    # typed projection subsequently restores every original purpose/value.
    # This avoids testing >100 tiny ranges for every scalar in multi-GB JSON.
    fresh = [row for row in plan["uses"] if row["kind"] in exact.FRESH_KINDS]
    families = []
    for kind in ("init", "env", "action_rng", "critic"):
        values = [row["value"] for row in fresh if row["kind"] == kind]
        families.append({"name": f"broad-read-{kind}", "kind": kind,
            "purpose": f"metadata_read_superset_then_exact_{kind}_projection",
            "start": min(values), "end_exclusive": max(values)+1, "query_only": False})
    fixed = [row["value"] for row in plan["uses"] if row["kind"] in exact.FIXED_KINDS and row["value"] > 1000]
    for label, values in (("low", [v for v in fixed if v < 1000000]),
                          ("high", [v for v in fixed if v >= 1000000])):
        if values:
            families.append({"name": "declared-fixed-control-query-"+label, "kind": "init",
                "purpose": "declared_fixed_rng_and_hash_arguments_not_claimed_fresh",
                "start": min(values), "end_exclusive": max(values)+1, "query_only": True})
    proposal = {"schema": "warehouse-seed-proposal.v1", "candidate_id": diagnostics.CANDIDATE_ID,
        "families": families, "registered_trace_paths": _registered_traces(root, previous_inventory),
        "source_proposal": {"path": str(exact_path), "sha256": file_sha(exact_path)},
        "exclude_paths": [str(output)], "authorization_granted": False,
        "limitations": ["Declared metadata roots and explicit program traces only; model weights are never read.",
            "19/20 classifier controls are declared but not globally searched for absence.",
            "Spent 275-scene development reuse is not included in any new-seed absence claim.",
            "Final delta and source/model/recipe-bound limited review remain required."]}
    write(output/"proposal.json", proposal)
    metadata.read_proposal(output/"proposal.json")
    write(output/"expansion-check.json", checked)
    return {"proposal": str(output/"proposal.json"), "exact_plan": str(exact_path), **checked}


def typed_model_lock_review(root, proposal_path, output):
    """Read one pinned known metadata schema despite its conservative name."""
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = root/diagnostics.DATA_ROOT/"plan.json"
    original = json.loads(diagnostics.read_pinned(plan_path, diagnostics.PINNED_INPUTS["plan.json"]))
    pointer = original["bindings"]["model_lock"]
    path = Path(pointer["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or file_sha(path) != pointer["sha256"]:
        raise RuntimeError("known model-method metadata binding changed")
    value = json.loads(path.read_bytes())
    if value.get("schema") != "warehouse-v620-model-method-lock.v1":
        raise RuntimeError("possible JSON weight is not the pinned known metadata schema")
    producer = root/"backend/training/warehouse_v620_bc_pilot_eval.py"
    text = producer.read_text()
    if '"warehouse-v620-model-method-lock.v1"' not in text or '"model-method.lock.json"' not in text:
        raise RuntimeError("known model-lock metadata producer proof is unavailable")
    _, families, _ = metadata.read_proposal(proposal_path)
    detail = metadata.analyze_literal(value, families)
    result = {"schema": "warehouse-seed-audit-known-metadata-schema.v1", "authorization_granted": False,
        "path": str(path), "sha256": pointer["sha256"], "parent_plan_sha256": file_sha(plan_path),
        "known_schema": value["schema"], "classification": "pinned_model_method_metadata_not_neural_weights",
        "producer": {"path": str(producer), "sha256": file_sha(producer),
            "lines": [text[:text.index('"warehouse-v620-model-method-lock.v1"')].count("\n")+1,
                      text[:text.index('"model-method.lock.json"')].count("\n")+1]},
        "detail": detail, "model_weight_content_reads": 0, "environment_resets": 0}
    write(output, result)
    return result


def scan(root, prepared, roots, output):
    """Full bounded inventory + complete triage + strong contextual projection."""
    from experiments.warehouse.seed_metadata_triage import triage
    workflow_sha = file_sha(__file__)
    root, prepared, output = Path(root).resolve(), Path(prepared).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    exact_path, proposal_path = prepared/"exact-uses.json", prepared/"proposal.json"
    validate_exact(exact_path)
    # All roots, including missing ones, are retained in the evidence; missing
    # roots block a final negative conclusion rather than silently shrinking it.
    inventory = metadata.audit(proposal_path, [Path(r).resolve() for r in roots], output/"bounded")
    typed_model_lock_review(root, proposal_path, output/"known-model-metadata.json")
    triaged = triage(output/"bounded/inventory.json", proposal_path, output/"triaged")
    result = exact.assess(exact_path, output/"triaged/result.json", output/"exact",
                          context_source_root=root)
    report = {"schema": "warehouse-v621-bc-seed-evidence-review.v1", "authorization_granted": False,
        "expansion": validate_exact(exact_path), "scan_roots": inventory["scan_roots"],
        "workflow_driver_sha256": workflow_sha,
        "known_model_metadata_review": {"path": str(output/"known-model-metadata.json"), "sha256": file_sha(output/"known-model-metadata.json")},
        "bounded_inventory": {"path": str(output/"bounded/inventory.json"), "sha256": file_sha(output/"bounded/inventory.json")},
        "triaged_inventory": {"path": str(output/"triaged/result.json"), "sha256": file_sha(output/"triaged/result.json")},
        "exact_evidence": {"path": str(output/"exact/result.json"), "sha256": file_sha(output/"exact/result.json")},
        "summary": result["summary"], "requires_final_delta_inventory": True,
        "source_freeze_and_model_binding_still_required": True,
        "possible_json_weight_metadata_needs_explicit_schema_review": True,
        "environment_resets": 0, "fit_calls": 0, "PPO_joint_steps": 0, "publishable": False}
    if file_sha(__file__) != workflow_sha:
        raise RuntimeError("seed workflow source changed during read-only evidence generation")
    write(output/"review.json", report)
    return report


def delta(inventory_path, prepared, output):
    report = json.loads(Path(inventory_path).read_bytes())
    validate_exact(Path(prepared)/"exact-uses.json")
    return exact.delta_inventory(report, Path(prepared)/"proposal.json", output_directory=output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "scan", "delta"))
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--scan-root", type=Path, action="append")
    parser.add_argument("--previous-inventory", type=Path)
    parser.add_argument("--inventory", type=Path)
    options = parser.parse_args(argv)
    if options.operation == "prepare":
        result = prepare(options.source_root, options.output_dir, previous_inventory=options.previous_inventory)
    elif options.operation == "scan":
        if not options.prepared or not options.scan_root:
            parser.error("scan requires --prepared and explicit --scan-root declarations")
        result = scan(options.source_root, options.prepared, options.scan_root, options.output_dir)
    else:
        if not options.prepared or not options.inventory:
            parser.error("delta requires --prepared and --inventory")
        result = delta(options.inventory, options.prepared, options.output_dir)
    print(json.dumps({"authorization_granted": False, "operation": options.operation,
        "output_directory": str(options.output_dir.resolve()),
        "requires_final_review": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
