#!/usr/bin/env python3
"""Resolve the bounded ledger's YAML/gzip/large JSON metadata, without rollouts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
import json
import io
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_warehouse_v619_seed_ledger import extract, digest

JQ_FILTER = r'''
select(length == 2)
| select(
    (.[0] | any(type == "string" and (ascii_downcase | contains("seed"))))
    or (.[0][-1] == "start" or .[0][-1] == "end_exclusive")
    or ((.[1] | type) == "number" and $targets[0][(.[1] | tostring)] == true)
  )
'''


def file_sha256(path):
    result = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


class InertRecord:
    """Serialization fields only; no original class or callable is imported."""
    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        instance.constructor_args = args
        instance.constructor_kwargs = kwargs
        return instance

    def __init__(self, *args, **kwargs):
        self.constructor_args = args
        self.constructor_kwargs = kwargs

    def __setstate__(self, state):
        self.literal_state = state


class InertUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Even a REDUCE/NEWOBJ global resolves to this inert local placeholder.
        # No original environment, imported function, or reducer can execute.
        return InertRecord

    def persistent_load(self, pid):
        raise ValueError("external pickle references are forbidden")


def inert_pickle_fields(raw):
    stream = io.BytesIO(raw)
    value = InertUnpickler(stream).load()
    if stream.read():
        raise ValueError("unexpected trailing bytes in gzip pickle")
    visited = set()

    def literals(item):
        if isinstance(item, (dict, list, tuple, set, InertRecord)):
            if id(item) in visited:
                return None
            visited.add(id(item))
        if isinstance(item, InertRecord):
            return {key: literals(value) for key, value in vars(item).items()}
        if isinstance(item, dict):
            return {str(key): literals(value) for key, value in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [literals(value) for value in item]
        if isinstance(item, (bytes, bytearray)):
            return {"opaque_bytes_sha256": sha256(item).hexdigest(), "opaque_bytes_count": len(item)}
        return item
    return literals(value)


def stream_seeds(path, targets_path, error_path):
    seeds, intervals, objects = set(), set(), defaultdict(dict)
    target_numeric = set()
    event_count = 0
    # jq validates the complete JSON stream; filtered events never skip parsing bytes.
    with error_path.open("x") as errors:
        process = subprocess.Popen(["/usr/bin/jq", "--stream", "-c", "--slurpfile",
            "targets", str(targets_path), JQ_FILTER, str(path)],
            stdout=subprocess.PIPE, stderr=errors, text=True)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                keypath, value = json.loads(line)
                event_count += 1
                if type(value) is int:
                    if any("seed" in str(k).lower() and not any(x in str(k).lower()
                           for x in ("sha", "hash", "count", "offset", "schema")) for k in keypath):
                        seeds.add(value)
                    target_numeric.add(value)
                if keypath and keypath[-1] in ("start", "end_exclusive") and type(value) is int:
                    objects[tuple(keypath[:-1])][keypath[-1]] = value
            returncode = process.wait()
        except BaseException:
            process.kill(); process.wait()
            raise
    for row in objects.values():
        if set(row) == {"start", "end_exclusive"} and row["start"] < row["end_exclusive"]:
            intervals.add((row["start"], row["end_exclusive"]))
    if returncode:
        raise ValueError(f"jq streaming parse failed: {error_path.read_text()[:300]}")
    return seeds, intervals, target_numeric, event_count


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-output", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    from experiments.warehouse import v619_protocol as v
    import yaml
    inventory_raw = args.inventory.read_bytes()
    inventory = json.loads(inventory_raw)
    original_by_path = {row["path"]: row for row in inventory["inventory"]}
    plan = v.seed_plan()
    targets = set(range(v.FRESH_BASE, v.FRESH_BASE+124))
    targets.update(seed*10+task for seed in list(targets) for task in v.TASKS)
    targets.update(v.INDEPENDENT_BASES)
    targets.update(row["scenario_seed"] for row in plan["independent"])
    targets.update(row["action_seed"] for row in plan["independent"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets_path = output / "candidate-seeds.json"
    write(targets_path, {str(seed): True for seed in sorted(targets)})
    rows, content_cache, failures, references, spans = [], {}, [], [], []
    for previous_output in args.previous_output:
        for path in sorted(previous_output.glob("file-*.json")):
            row = json.loads(path.read_bytes())
            if row.get("resolved") is True:
                content_cache[row["sha256"]] = {key: value for key, value in row.items()
                    if key not in ("bytes", "original_skip_reason", "path", "resolved", "sha256")}
    def process_item(item):
        index, previous = item
        path = Path(previous["path"])
        before = file_sha256(path)
        if previous.get("sha256", before) != before:
            raise RuntimeError("historical metadata changed since bounded audit")
        row = {"path": str(path), "bytes": path.stat().st_size,
               "sha256": before, "original_skip_reason": previous["reason"]}
        try:
            if before in content_cache:
                detail = dict(content_cache[before])
                detail["same_content_as"] = detail["parsed_path"]
            elif path.stat().st_size == 0:
                detail = {"parser": "empty_file_no_records", "parsed_path": str(path),
                          "seed_count": 0, "reserved_intervals": [], "candidate_references": []}
            elif path.suffix in (".yaml", ".yml"):
                value = yaml.safe_load(path.read_text())
                seeds, intervals = extract(value)
                detail = {"parser": "yaml.safe_load", "parsed_path": str(path),
                    "seed_count": len(seeds), "reserved_intervals": sorted(map(list, intervals)),
                    "candidate_references": sorted(seeds & targets)}
            elif path.open("rb").read(2) == b"\x1f\x8b":
                raw = gzip.open(path, "rb").read()
                is_pickle = raw.startswith(b"\x80")
                value = inert_pickle_fields(raw) if is_pickle else json.loads(raw)
                seeds, intervals = extract(value)
                detail = {"parser": "gzip_pickle_inert_fields_only" if is_pickle else "gzip_json_complete", "parsed_path": str(path),
                    "seed_count": len(seeds), "reserved_intervals": sorted(map(list, intervals)),
                    "candidate_references": sorted(seeds & targets)}
            else:
                seeds, intervals, numeric, events = stream_seeds(
                    path, targets_path, output/f"jq-{index:03d}.stderr")
                detail = {"parser": "jq_1.7_stream_complete_json", "parsed_path": str(path),
                    "filtered_event_count": events, "seed_count": len(seeds),
                    "reserved_intervals": sorted(map(list, intervals)),
                    "candidate_references": sorted((seeds | numeric) & targets)}
            content_cache[before] = dict(detail)
            row.update(detail, resolved=True)
            if file_sha256(path) != before:
                raise RuntimeError("historical metadata changed while reading")
            if row["candidate_references"]:
                references.append({"path": str(path), "sha256": before, "seeds": row["candidate_references"]})
            for start, end in row["reserved_intervals"]:
                matching = sorted(seed for seed in targets if start <= seed < end)
                if matching:
                    spans.append({"path": str(path), "sha256": before, "start": start,
                                  "end_exclusive": end, "matching_seeds": matching})
        except Exception as error:
            row.update(resolved=False, error=f"{type(error).__name__}: {error}")
            failures.append(row)
        print(json.dumps({"index": index+1, "of": len(inventory["skipped"]),
                          "resolved": row["resolved"], "parser": row.get("parser"), "bytes": row["bytes"]}), flush=True)
        write(output/f"file-{index:03d}.json", row)
        return row
    if not 1 <= args.workers <= 4:
        raise ValueError("metadata triage allows one to four independent streaming readers")
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        rows = list(workers.map(process_item, enumerate(inventory["skipped"])))
    # Recheck prior parsed scope, whose base/expanded sets cover all task variants.
    prior_conflicts = []
    for split in ("fixed_fit", "fixed_calibration", "fixed_eligibility", "fixed_validation", "fixed_verification"):
        for family in ("base", "task_expanded"):
            item = inventory["proposals"][f"high/{split}:{family}"]
            if item["explicit_reference_count"] or item["reservation_overlaps"]:
                prior_conflicts.append(f"high/{split}:{family}")
    item = inventory["proposals"]["high/fixed_verification:rng_contiguous_backup"]
    if item["explicit_reference_count"] or item["reservation_overlaps"]:
        prior_conflicts.append("fixed_verification:rng_contiguous_backup")
    approved = not failures and not references and not spans and not prior_conflicts
    result = {"schema": "warehouse-v619-seed-metadata-triage.v1",
        "publishable": False, "admission_evidence": False,
        "approved_for_fixed_validation": approved,
        "approval_scope": "recorded_metadata_in_four_declared_roots_for_fixed_6150_actor_development_only",
        "fixed_actor_sha256": v.ACTOR_SHA256,
        "fixed_validation_seed_plan_sha256": v.digest(plan),
        "original_inventory": {"path": str(args.inventory.resolve()), "sha256": sha256(inventory_raw).hexdigest()},
        "triage_script_sha256": file_sha256(Path(__file__)),
        "candidate_seed_set_sha256": digest(sorted(targets)),
        "candidate_count": len(targets), "candidate_file_sha256": file_sha256(targets_path),
        "resolved_file_count": sum(row["resolved"] for row in rows), "total_file_count": len(rows),
        "parser_counts": dict(Counter(row.get("parser", "error") for row in rows)),
        "candidate_references": references, "reservation_overlaps": spans,
        "prior_parsed_scope_conflicts": prior_conflicts, "unresolved": failures,
        "claimed_or_spent_sealed_blocks": [],
        "scan_scope": {"roots": inventory["scan_roots"],
            "prior_read_metadata_files": inventory["inventory_count"],
            "triaged_files": len(rows), "read_weight_file_formats": [],
            "new_v619_declarations_excluded_from_prior_use_evidence": True},
        "limitations": ["No whole-disk or external-system claim; unrecorded executions cannot be disproved.",
            "Any later discovered sealed claim counts as spent, including failures and interruptions.",
            "This is not permission for formal paired training, sealed audit, heldout tests, or release.",
            "Old inspected 6150 Actor/cache results remain exploratory; approval only concerns the new fixed-Actor data plan."],
        "environment_resets": 0, "training_steps": 0, "scientific_fit_calls": 0,
        "files": rows}
    write(output/"result.json", result)
    print(json.dumps({"result": str(output/"result.json"), "sha256": file_sha256(output/"result.json"),
                      "approved_for_fixed_validation": approved, "unresolved": len(failures)}), flush=True)
    return 0 if approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
