#!/usr/bin/env python3
"""Parameterized, bounded, read-only Warehouse metadata inventory.

This tool reports evidence and conflicts. It never grants execution authority.
Use seed_metadata_triage.py for skipped JSON/gzip/YAML metadata. Both drivers
bind an external proposal's bytes; neither imports policy/environment code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tomllib
from .seed_interval_index_v642 import SeedIntervalIndex
from . import seed_interval_index_v642 as _interval_index_source

KINDS = {"init", "env", "scenario", "action_rng", "critic", "sealed"}
PRUNE = {".git", "node_modules", ".venv", "__pycache__", ".next", "dist", "build"}
EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".toml"}
MAX_BYTES = 12*1024*1024
METADATA_NAME = re.compile(r"auth|claim|reset|seed|schedule|receipt|acceptance|training|report|progress|ledger|plan|manifest|config|evaluation|result|predeclared|lock", re.I)
WEIGHT_NAME = re.compile(r"(^|[_.-])(actor|weights?|model|policy)([_.-]|\d|$)", re.I)
WEIGHT_METADATA = re.compile(r"training|report|receipt|acceptance|manifest|config|evaluation|metadata|parity|audit|diagnos|summary|schedule|binding|proof|check", re.I)
OPAQUE_WEIGHT_SUFFIXES = {".npz", ".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".bin"}
OPAQUE_WEIGHT_DIRECTORIES = {"actors", "checkpoints", "weights", "models"}
SELECTION_SCOPE = "metadata_basename_whitelist_plus_explicit_registered_traces_weights_excluded.v2"
PATH_SEED_HINT = re.compile(r"(?:^|[/_.-])(pair|seed)[_-]?(\d+)(?=$|[/_.-])", re.I)
INTERNAL_STATE_KEYS = {"rng_state", "random_state", "numpy_rng_state", "np_rng_state",
    "torch_rng_state", "cuda_rng_state", "mt19937", "mt_state", "generator_state"}
DECLARATION_NAMES = re.compile(r"plan|recipe|config|schedule|authorization|manifest|predeclared|lock", re.I)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def value_intervals(values):
    """Losslessly compress exact consecutive values, preserving every gap."""
    result = []
    for value in sorted(values):
        if result and result[-1][1] == value:
            result[-1] = (result[-1][0], value + 1)
        else:
            result.append((value, value + 1))
    return tuple(result)


def file_sha(path):
    result = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8*1024*1024):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def read_proposal(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    proposal = json.loads(raw)
    if (proposal.get("schema") != "warehouse-seed-proposal.v1"
            or not isinstance(proposal.get("candidate_id"), str)
            or not proposal["candidate_id"] or not isinstance(proposal.get("families"), list)):
        raise ValueError("invalid external proposal schema")
    names, families = set(), []
    for row in proposal["families"]:
        if (not isinstance(row, dict) or row.get("kind") not in KINDS
                or not isinstance(row.get("name"), str) or not row["name"]
                or row["name"] in names or row.get("query_only", False) not in (True, False)):
            raise ValueError("invalid family identity/kind")
        names.add(row["name"])
        if "values" in row:
            if set(row) - {"name", "kind", "values", "query_only", "purpose"}:
                raise ValueError("unknown explicit family fields")
            values = row["values"]
            if (not isinstance(values, list) or not values
                    or any(type(value) is not int or not 0 <= value < 2**63 for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError("invalid explicit seed values")
            intervals = value_intervals(values)
        else:
            if set(row) - {"name", "kind", "start", "end_exclusive", "query_only", "purpose"}:
                raise ValueError("unknown interval family fields")
            start, end = row.get("start"), row.get("end_exclusive")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= 2**63:
                raise ValueError("invalid seed interval")
            intervals = ((start, end),)
        families.append({**row, "intervals": intervals,
            "count": sum(end-start for start, end in intervals)})
    if not families:
        raise ValueError("empty seed proposal")
    # Distinct intended non-query purposes must not share initialization
    # arguments. Numeric init-vs-env equality is separately namespaced.
    for i, first in enumerate(families):
        for second in families[:i]:
            if first.get("query_only") or second.get("query_only"):
                continue
            if (first["kind"] == "init") != (second["kind"] == "init"):
                continue
            if any(max(a, c) < min(b, d) for a, b in first["intervals"] for c, d in second["intervals"]):
                raise ValueError(f"proposal purposes overlap: {first['name']}, {second['name']}")
    return proposal, families, {"path": str(path), "sha256": sha256(raw).hexdigest()}


def matching_families(value, families):
    if type(value) is not int:
        return []
    return [row["name"] for row in families if any(start <= value < end for start, end in row["intervals"])]


def seed_semantics(keypath):
    keys = [str(key).lower() for key in keypath if isinstance(key, str)]
    # State words are not initial RNG arguments. Constructor arguments inside
    # inert pickle records and scalar random_state=N remain real/ambiguous
    # initialization arguments rather than being discarded by their parent.
    for index, key in enumerate(keypath):
        if str(key).lower() in INTERNAL_STATE_KEYS:
            descendants = keypath[index+1:]
            if "constructor_args" in descendants or "constructor_kwargs" in descendants:
                return "ambiguous_seed_argument"
            numeric_indices = sum(isinstance(k, int) for k in descendants)
            word_keys = {str(k).lower() for k in descendants if isinstance(k, str)}
            if numeric_indices >= 2 or (numeric_indices and word_keys & {"key", "state", "mt", "words"}):
                # A suggestive key and nested indices are insufficient to
                # establish generator state. Keep the match until a complete
                # state shape and its producer are independently established.
                return "ambiguous_internal_rng_state_requires_context_proof"
    key = keys[-1] if keys else ""
    if any(token in key for token in ("sha", "hash", "count", "offset", "schema", "percentage")):
        return "nonseed_metadata"
    if key == "random_state":
        return "ambiguous_seed_argument"
    if key in ("base", "scenario_base", "base_seeds") and any("seed" in k for k in keys[:-1]):
        return "scenario"
    if any(token in key for token in ("pair_seed", "initialization_seed", "model_seed", "actor_seed", "torch_seed", "numpy_seed")):
        return "init"
    if any(token in key for token in ("action_seed", "rng_seed", "sampling_seed", "action_rng")):
        return "action_rng"
    if "critic" in key and "seed" in key:
        return "critic"
    if any(token in key for token in ("scenario_seed", "environment_seed", "reset_seed", "episode_seed", "scene_seed", "base_seed")):
        return "env"
    if "seed" in key:
        return "ambiguous_seed_argument"
    if any("seed" in k for k in keys):
        return "ambiguous_seed_context"
    return "numeric_coincidence_needs_context_review"


def compatible(kind, semantics):
    if semantics.startswith("internal_rng_") or semantics == "nonseed_metadata":
        return False
    if kind == "init":
        return (semantics == "init" or semantics.startswith("ambiguous_")
                or semantics == "numeric_coincidence_needs_context_review")
    return semantics != "init"


def file_classification(path, value=None):
    name = Path(path).name.lower()
    if "claim" in str(path).lower():
        return "claim_spent_even_if_failed_or_interrupted"
    if "reset" in name and Path(path).suffix == ".jsonl":
        return "actual_environment_reset_journal"
    if isinstance(value, dict):
        schema = str(value.get("schema", "")).lower()
        if any(token in schema for token in ("seed-ledger", "seed-audit", "seed-triage", "seed-clearance")):
            return "audit_reference_report_not_original_execution"
        if "frame" in value and "roles" in value and "base_seed" in value:
            return "recorded_physical_nn_trajectory"
    if DECLARATION_NAMES.search(name):
        return "declaration_or_reservation_not_execution_proof"
    return "receipt_or_metadata_report_not_independently_replayed"


def analyze_events(events, families):
    lookup = SeedIntervalIndex.compile(families).matches
    matches, ignored, ranges, range_pairs = [], [], {}, {}
    for path, value in events:
        if type(value) is not int:
            continue
        if path and path[-1] in ("start", "end_exclusive", "count"):
            ranges.setdefault(tuple(path[:-1]), {})[path[-1]] = value
        if (path and isinstance(path[-1], int)
                and any("seed_range" in str(key).lower() for key in path)):
            range_pairs.setdefault(tuple(path[:-1]), {})[path[-1]] = value
        names = lookup(value)
        if names:
            semantics = seed_semantics(path)
            row = {"path": list(path), "value": value, "semantics": semantics,
                   "families": names}
            if semantics.startswith("internal_rng_") or semantics == "nonseed_metadata":
                ignored.append(row)
            else:
                matches.append(row)
    spans = []
    normalized_ranges = []
    for path, value in ranges.items():
        if "start" in value and "end_exclusive" in value and value["start"] < value["end_exclusive"]:
            normalized_ranges.append((path, value["start"], value["end_exclusive"], "explicit_end_exclusive"))
        elif ("start" in value and "count" in value and value["count"] > 0
                and any("seed" in str(key).lower() or "reserv" in str(key).lower() for key in path)):
            normalized_ranges.append((path, value["start"], value["start"]+value["count"], "explicit_start_count"))
    for path, pair in range_pairs.items():
        if set(pair) == {0, 1} and pair[0] >= 0 and pair[1] > 0:
            # Historical evaluation ledgers use [start,count]. Cover both
            # conventions conservatively until context adjudicates a hit.
            end = max(pair[0]+pair[1], pair[1])
            normalized_ranges.append((path, pair[0], end,
                                      "conservative_union_start_count_or_start_end"))
    for path, start, end, convention in normalized_ranges:
        if start < end:
            for family in families:
                # Historical reserved intervals refer to environment seed
                # families, not model parameter initialization arguments.
                range_semantics = "init" if seed_semantics(path) == "init" else "environment_reservation"
                if (family["kind"] == "init") != (range_semantics == "init"):
                    continue
                overlaps = [[max(start, a), min(end, b)]
                            for a, b in family["intervals"]
                            if max(start, a) < min(end, b)]
                if overlaps:
                    spans.append({"path": list(path), "family": family["name"],
                        "historical_interval": {"start": start, "end_exclusive": end},
                        "range_interpretation": convention, "overlaps": overlaps,
                        "semantics": range_semantics})
    return {"candidate_matches": matches, "ignored_internal_or_nonseed_matches": ignored,
            "reservation_overlaps": spans}


def literal_events(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from literal_events(child, path+(str(key),))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from literal_events(child, path+(index,))
    else:
        yield path, value


def analyze_literal(value, families):
    return analyze_events(literal_events(value), families)


def candidates(roots, excludes, *, registered_trace_paths=(), selection_stats=None):
    """Filename-only filtering precedes every content read or content hash."""
    registered = {Path(value).resolve() for value in registered_trace_paths}
    registered_names = {str(value) for value in registered}
    stats = selection_stats if selection_stats is not None else {}
    stats.setdefault("excluded_counts", {})
    stats.setdefault("excluded_weights_name_stat_only", [])
    def skip(reason):
        stats["excluded_counts"][reason] = stats["excluded_counts"].get(reason, 0)+1
    for root in roots:
        for directory, dirs, names in os.walk(root, followlinks=False):
            directory = Path(directory)
            dirs[:] = sorted(name for name in dirs if name not in PRUNE
                and not any((directory/name).resolve().is_relative_to(excluded) for excluded in excludes))
            for name in sorted(names):
                path = directory/name
                if not ("warehouse" in str(path).lower() or re.search(r"seed|claim|authoriz", name, re.I)):
                    skip("outside_warehouse_seed_metadata_scope"); continue
                weight_directory = any(part.lower() in OPAQUE_WEIGHT_DIRECTORIES for part in path.parts)
                weight_file = (path.suffix.lower() in OPAQUE_WEIGHT_SUFFIXES or
                    (path.suffix.lower() == ".json" and WEIGHT_NAME.search(name)
                     and not WEIGHT_METADATA.search(name)))
                # Cheap basename rejection precedes filesystem calls for the
                # hundreds of thousands of historical per-step JSON files.
                is_registered = str(path) in registered_names
                # Claims can use opaque/hash basenames, and durable start
                # guards can be named merely started.json. Keep their record
                # context without reopening arbitrary per-step JSON payloads.
                claim_context = any("claim" in part.lower() for part in directory.parts)
                v6_start_guard = (("warehouse_v619" in str(path).lower() or
                                   "warehouse_v620" in str(path).lower())
                                  and (name == "started.json" or name.endswith(".started.json")))
                whitelisted = bool(METADATA_NAME.search(name) or claim_context or v6_start_guard)
                if not weight_directory and not weight_file and (
                        path.suffix.lower() not in EXTENSIONS or
                        (not whitelisted and not is_registered)):
                    skip("basename_not_whitelist_or_registered_trace"); continue
                if path.is_symlink():
                    skip("symlink_not_followed"); continue
                if any(path.resolve().is_relative_to(excluded) for excluded in excludes):
                    skip("explicit_exclusion"); continue
                if weight_directory or weight_file:
                    skip("opaque_or_possible_json_weight_not_read")
                    info = path.stat()
                    stats["excluded_weights_name_stat_only"].append({"path":str(path.resolve()),
                        "bytes":info.st_size,"mtime_ns":info.st_mtime_ns,
                        "reason":"opaque_model_directory_or_weight_filename_not_read"})
                    continue
                if path.suffix.lower() not in EXTENSIONS:
                    skip("nonmetadata_extension"); continue
                if path.resolve() in registered:
                    if (path.suffix.lower() != ".jsonl" or path.name not in
                            {"fit.jsonl","calibration.jsonl","eligibility.jsonl","validation.jsonl"}):
                        raise ValueError("registered trace is not an explicit program-purpose JSONL")
                elif not whitelisted:
                    skip("basename_not_metadata_whitelist_or_registered_trace"); continue
                yield path


def weight_path_hints(selection_stats, family):
    """Names/stat may reveal earlier initialization; never open weight bytes."""
    result = []
    for row in (selection_stats or {}).get("excluded_weights_name_stat_only", []):
        for match in PATH_SEED_HINT.finditer(row["path"]):
            value = int(match.group(2))
            if not any(start <= value < end for start, end in family["intervals"]):
                continue
            if match.group(1).lower() == "pair" and family["kind"] != "init":
                continue
            result.append({**row, "value": value, "token": match.group(0),
                "semantics": "model_initialization_path_hint_requires_receipt_review"
                    if match.group(1).lower() == "pair" else "ambiguous_seed_path_hint_requires_receipt_review",
                "content_read": False})
    return result


def summarize(inventory, families, *, selection_stats=None, missing_roots=()):
    summaries = {}
    unresolved = [row["path"] for row in inventory if row.get("resolved") is not True] + list(missing_roots)
    for family in families:
        matches, spans, ignored, own_declarations, audit_references = [], [], [], [], []
        for row in inventory:
            detail = row.get("detail", {})
            for match in detail.get("candidate_matches", []):
                if family["name"] not in match["families"] or not compatible(family["kind"], match["semantics"]):
                    continue
                current = {"file": row["path"], "file_sha256": row.get("sha256"),
                           "classification": row["classification"], **match}
                if row.get("same_candidate_declaration"):
                    own_declarations.append(current)
                elif row["classification"] == "audit_reference_report_not_original_execution":
                    audit_references.append(current)
                else:
                    matches.append(current)
            for span in detail.get("reservation_overlaps", []):
                if span["family"] == family["name"]:
                    if row.get("same_candidate_declaration"):
                        own_declarations.append({"file": row["path"], **span})
                    else:
                        spans.append({"file": row["path"], **span})
            ignored += [{"file": row["path"], **match} for match in detail.get("ignored_internal_or_nonseed_matches", [])
                        if family["name"] in match["families"]]
        path_hints = weight_path_hints(selection_stats, family)
        summaries[family["name"]] = {"kind": family["kind"], "query_only": family.get("query_only", False),
            "count": family["count"], "references_or_reservations": matches,
            "reservation_overlaps": spans, "ignored_mt_or_nonseed": ignored,
            "own_candidate_declarations_not_use": own_declarations,
            "audit_references_not_original_execution": audit_references,
            "excluded_weight_path_seed_hints": path_hints,
            "status": "blocked_by_recorded_reference_or_reservation" if matches or spans or path_hints else
                "audit_references_require_original_use_context_review" if audit_references else
                "missing_roots_prevent_negative_conclusion" if missing_roots else
                "unresolved_metadata_prevents_negative_conclusion" if unresolved else
                "no_known_conflict_within_declared_metadata_scope_not_authorization"}
    return {"families": summaries, "unresolved_paths": unresolved,
        "authorization_granted": False, "environment_resets": 0, "fit_calls": 0,
        "limitations": ["Recorded whitelisted metadata and explicit traces only; no whole-disk, excluded-model, external-system or unrecorded-run negative proof.",
            "Opaque model weights and non-whitelisted metadata are not read; no global freshness inference is permitted.",
            "Active CV output can gain files after inventory; a final delta inventory is required before any separate clearance.",
            "All claims count as spent regardless of failed/interrupted status.",
            "Opaque, unparsed or skipped metadata blocks a negative conclusion.",
            "An MT19937 state word is not an RNG initialization argument.",
            "Numeric coincidences without seed semantics require explicit contextual adjudication.",
            "Older ledger/audit references are not original execution proof, but remain unresolved until their use context is reviewed.",
            "Model initialization arguments are typed separately from environment reserved intervals."]}


def inventory_only(proposal_path, roots, output, max_bytes=MAX_BYTES):
    started = datetime.now(timezone.utc).isoformat()
    proposal, families, pointer = read_proposal(proposal_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    excludes = [Path(pointer["path"]), output]
    excludes += [Path(path).resolve() for path in proposal.get("exclude_paths", [])]
    stats, selected = {}, []
    for path in candidates(roots, excludes,
            registered_trace_paths=proposal.get("registered_trace_paths", []), selection_stats=stats):
        info = path.stat()
        selected.append({"path":str(path.resolve()),"bytes":info.st_size,
            "mtime_ns":info.st_mtime_ns,"large_deferred":info.st_size > max_bytes})
    report = {"schema":"warehouse-seed-selection-stat-only.v2", "proposal":pointer,
        "source_proposal":proposal.get("source_proposal"),
        "auditor_sha256":file_sha(__file__),"scan_roots":list(map(str,roots)),
        "selection_scope":SELECTION_SCOPE,"scope_exclusions":list(map(str,excludes)),
        "registered_trace_paths":proposal.get("registered_trace_paths", []),
        "selected_metadata_files":len(selected),"selected_metadata_bytes":sum(row["bytes"] for row in selected),
        "bounded_direct_bytes":sum(row["bytes"] for row in selected if not row["large_deferred"]),
        "large_deferred_files":sum(row["large_deferred"] for row in selected),
        "large_deferred_bytes":sum(row["bytes"] for row in selected if row["large_deferred"]),
        "metadata_content_read_count":0,"model_weight_read_count":0,
        "started_at_utc":started,"completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "selection_stats":stats,"candidates":selected,"authorization_granted":False,
        "preliminary_only":True,"requires_final_delta_inventory":True}
    write_new(output/"selection.json",report)
    return report


def audit(proposal_path, roots, output, max_bytes=MAX_BYTES):
    driver_hash = file_sha(__file__)
    proposal, families, pointer = read_proposal(proposal_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    excludes = [Path(pointer["path"]), output]
    excludes += [Path(path).resolve() for path in proposal.get("exclude_paths", [])]
    inventory = []
    selection_stats = {}
    registered_trace_paths = proposal.get("registered_trace_paths", [])
    if not isinstance(registered_trace_paths,list) or any(not isinstance(p,str) for p in registered_trace_paths):
        raise ValueError("registered trace path declaration malformed")
    missing = [str(root) for root in roots if not Path(root).is_dir()]
    for path in candidates(roots, excludes, registered_trace_paths=registered_trace_paths,
                           selection_stats=selection_stats):
        stat = path.stat()
        row = {"path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "classification": file_classification(path),
            "resolved": False}
        if stat.st_size > max_bytes:
            row["needs_triage"] = "large_metadata_stream_parse"
        else:
            raw = path.read_bytes()
            row["sha256"] = sha256(raw).hexdigest()
            try:
                if raw[:2] == b"\x1f\x8b":
                    raise ValueError("gzip_metadata_requires_inert_triage")
                if not raw.strip():
                    value = None
                elif path.suffix == ".jsonl":
                    value = [json.loads(line) for line in raw.splitlines() if line.strip()]
                elif path.suffix == ".toml":
                    value = tomllib.loads(raw.decode())
                elif path.suffix in (".yaml", ".yml"):
                    raise ValueError("yaml_requires_safe_triage")
                else:
                    value = json.loads(raw)
                sample = value[0] if isinstance(value, list) and value else value
                row["classification"] = file_classification(path, sample)
                row["same_candidate_declaration"] = bool(isinstance(sample, dict)
                    and sample.get("candidate_id") == proposal["candidate_id"]
                    and row["classification"] == "declaration_or_reservation_not_execution_proof")
                row["detail"] = analyze_literal(value, families)
                row["resolved"] = True
            except (ValueError, UnicodeError) as error:
                row["needs_triage"] = str(error)[:160]
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"metadata changed during inventory: {path}")
        inventory.append(row)
    for root in roots:
        template = Path(root)/"backend/training/seed_ledger.py"
        if template.is_file():
            raw = template.read_bytes()
            if b"500_004" in raw and b"10_500_204" in raw:
                inventory.append({"path": str(template), "bytes": len(raw),
                    "sha256": sha256(raw).hexdigest(), "resolved": True,
                    "classification": "source_template_environment_reservation",
                    "detail": analyze_literal({"reserved_intervals": [
                        {"start": 500004, "end_exclusive": 10500204}]}, families)})
    report = {"schema": "warehouse-seed-metadata-audit.v2", "proposal": pointer,
        "proposal_candidate_id": proposal["candidate_id"], "families": families,
        "scan_roots": list(map(str, roots)), "missing_roots": missing,
        "scope_exclusions": list(map(str, excludes)),
        "maximum_direct_metadata_bytes": max_bytes, "read_model_weight_files": False,
        "auditor_sha256": driver_hash, "interval_index_source_sha256": file_sha(_interval_index_source.__file__), "inventory": inventory,
        "selection_scope": SELECTION_SCOPE, "selection_stats": selection_stats,
        "registered_trace_paths": registered_trace_paths,
        "source_proposal": proposal.get("source_proposal"),
        "preliminary_only": True, "requires_final_delta_inventory": True,
        "summary": summarize(inventory, families, selection_stats=selection_stats, missing_roots=missing)}
    if file_sha(pointer["path"]) != pointer["sha256"] or file_sha(__file__) != driver_hash:
        raise RuntimeError("proposal or auditor source changed during inventory")
    write_new(output/"inventory.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-metadata-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_metadata_bytes < 1:
        parser.error("positive metadata byte limit required")
    if args.inventory_only:
        report = inventory_only(args.proposal, [p.resolve() for p in args.root],
                                args.output_dir, args.max_metadata_bytes)
        print(json.dumps({key:report[key] for key in (
            "selected_metadata_files","selected_metadata_bytes","bounded_direct_bytes",
            "large_deferred_files","large_deferred_bytes","metadata_content_read_count",
            "model_weight_read_count","authorization_granted")}))
        return
    report = audit(args.proposal, [p.resolve() for p in args.root], args.output_dir,
                   args.max_metadata_bytes)
    print(json.dumps({"inventory": str(args.output_dir/"inventory.json"),
        "files": len(report["inventory"]), "unresolved": len(report["summary"]["unresolved_paths"]),
        "authorization_granted": False}))


if __name__ == "__main__":
    main()
