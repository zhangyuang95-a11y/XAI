#!/usr/bin/env python3
"""v628 read-only triage acceleration, retaining exact seed classification.

The jq numeric prefilter uses one conservative envelope per family. Every
result is subsequently classified against the ORIGINAL exact intervals.
Seed/path/range metadata predicates are unchanged. The envelope is a query
superset, never a clearance or a new scientific seed allocation.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys

from .seed_metadata_audit import (
    analyze_events, analyze_literal, file_classification, file_sha, read_proposal,
    summarize, write_new,
)

JQ_FILTER = r'''
select(length == 2)
| .[1] as $value
| select(
    (.[0] | any(type == "string" and (ascii_downcase | contains("seed"))))
    or (.[0][-1] == "start" or .[0][-1] == "end_exclusive" or .[0][-1] == "count")
    or (.[0] == ["schema"] or .[0] == ["candidate_id"])
    or (($value | type) == "number" and
        any($ranges[]; $value >= .[0] and $value < .[1]))
  )
'''


class InertRecord:
    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        instance.constructor_args = args
        instance.constructor_kwargs = kwargs
        return instance

    def __init__(self, *args, **kwargs):
        self.constructor_args, self.constructor_kwargs = args, kwargs

    def __setstate__(self, state):
        self.literal_state = state


class InertUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return InertRecord

    def persistent_load(self, pid):
        raise ValueError("external pickle references are forbidden")


def inert_pickle_fields(raw):
    stream = io.BytesIO(raw)
    value = InertUnpickler(stream).load()
    if stream.read():
        raise ValueError("unexpected trailing bytes in gzip pickle")
    seen = set()

    def literal(item):
        if isinstance(item, (dict, list, tuple, set, InertRecord)):
            if id(item) in seen:
                return None
            seen.add(id(item))
        if isinstance(item, InertRecord):
            return {key: literal(child) for key, child in vars(item).items()}
        if isinstance(item, dict):
            return {str(key): literal(child) for key, child in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [literal(child) for child in item]
        if isinstance(item, (bytes, bytearray)):
            return {"opaque_bytes_sha256": sha256(item).hexdigest(),
                    "opaque_bytes_count": len(item)}
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise ValueError("nonliteral inert pickle field")
    return literal(value)


def opaque_pickle_fields(value, path=()):
    """Unknown binary scalar/array fields cannot prove absent seed arguments."""
    if isinstance(value, dict):
        if "opaque_bytes_sha256" in value:
            yield {"path": list(path), **value}
        else:
            for key, child in value.items():
                yield from opaque_pickle_fields(child, path+(key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from opaque_pickle_fields(child, path+(index,))


def stream_detail(path, families, error_path):
    intervals = [[min(span[0] for span in family["intervals"]),
                  max(span[1] for span in family["intervals"])] for family in families]
    # analyze_events below still receives the exact, unexpanded families.
    metadata = {}
    with Path(error_path).open("x") as errors:
        # JSONL traces contain one object per physical line. Preserve the line
        # in evidence paths; otherwise the same key in thousands of frames
        # would appear to identify one source field.
        query = JQ_FILTER
        is_jsonl = Path(path).suffix == ".jsonl"
        if is_jsonl:
            query += '\n| [([input_line_number - 1] + .[0]), .[1]]'
        process = subprocess.Popen(["/usr/bin/jq", "--stream", "-c", "--argjson",
            "ranges", json.dumps(intervals, separators=(",", ":")), query, str(path)],
            stdout=subprocess.PIPE, stderr=errors, text=True)
        assert process.stdout is not None
        count = 0

        def events():
            nonlocal count
            for line in process.stdout:
                keypath, value = json.loads(line)
                count += 1
                if keypath in (["schema"], ["candidate_id"]):
                    metadata[keypath[0]] = value
                elif is_jsonl and keypath in ([0, "schema"], [0, "candidate_id"]):
                    metadata[keypath[1]] = value
                yield tuple(keypath), value
        try:
            detail = analyze_events(events(), families)
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if returncode:
        raise ValueError(f"jq complete-stream parse failed: {Path(error_path).read_text()[:300]}")
    return detail, metadata, count


def triage(inventory_path, proposal_path, output_dir, max_expanded_bytes=256*1024*1024):
    driver_hash = file_sha(__file__)
    inventory_path = Path(inventory_path).resolve(strict=True)
    raw = inventory_path.read_bytes()
    report = json.loads(raw)
    proposal, families, pointer = read_proposal(proposal_path)
    if (report.get("schema") != "warehouse-seed-metadata-audit.v2"
            or report["proposal"] != pointer or report["families"] != json.loads(json.dumps(families))):
        raise RuntimeError("triage proposal differs from its bounded inventory")
    from . import seed_metadata_audit as warehouse_seed_audit
    if file_sha(warehouse_seed_audit.__file__) != report["auditor_sha256"]:
        raise RuntimeError("parameterized auditor source changed after bounded inventory")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(report)
    result.update({"schema": "warehouse-seed-metadata-triage.v2",
        "bounded_inventory": {"path": str(inventory_path), "sha256": sha256(raw).hexdigest()},
        "triage_driver_sha256": driver_hash, "inventory": []})
    content_cache = {}
    for index, previous in enumerate(report["inventory"]):
        row = dict(previous)
        path = Path(row["path"])
        if row["classification"] == "source_template_environment_reservation":
            if file_sha(path) != row["sha256"]:
                raise RuntimeError("source reservation changed after bounded audit")
            result["inventory"].append(row)
            continue
        before = path.stat()
        if before.st_size != row["bytes"] or before.st_mtime_ns != row["mtime_ns"]:
            raise RuntimeError(f"historical metadata changed since audit: {path}")
        before_hash = file_sha(path)
        if row.get("sha256", before_hash) != before_hash:
            raise RuntimeError("historical metadata hash changed since bounded audit")
        if row.get("resolved") is not True:
            row["sha256"] = before_hash
            try:
                with path.open("rb") as handle:
                    magic = handle.read(2)
                if before_hash in content_cache:
                    detail, sample, parser = content_cache[before_hash]
                    row["detail"], row["parser"] = detail, parser
                    row["same_content_reused_within_exact_proposal"] = True
                elif magic == b"\x1f\x8b":
                    with gzip.open(path, "rb") as handle:
                        expanded = handle.read(max_expanded_bytes+1)
                    if len(expanded) > max_expanded_bytes:
                        raise ValueError("gzip expanded metadata exceeds declared bound")
                    is_pickle = expanded.startswith(b"\x80")
                    value = inert_pickle_fields(expanded) if is_pickle else json.loads(expanded)
                    row["detail"] = analyze_literal(value, families)
                    row["parser"] = "gzip_pickle_inert_literal_fields" if is_pickle else "gzip_json_complete"
                    sample = value
                    if is_pickle:
                        row["opaque_pickle_fields"] = list(opaque_pickle_fields(value))
                        if row["opaque_pickle_fields"]:
                            raise ValueError("opaque pickle scalar/array bytes require producer-specific contextual review")
                elif path.suffix in (".yaml", ".yml"):
                    import yaml
                    value = yaml.safe_load(path.read_text())
                    row["detail"] = analyze_literal(value, families)
                    row["parser"] = "yaml.safe_load"
                    sample = value
                else:
                    detail, sample, count = stream_detail(path, families,
                        output/f"file-{index:05d}.jq.stderr")
                    row["detail"], row["stream_event_count"] = detail, count
                    row["parser"] = "jq_complete_json_stream"
                content_cache[before_hash] = (row["detail"], sample, row["parser"])
                row["classification"] = file_classification(path, sample)
                row["same_candidate_declaration"] = bool(isinstance(sample, dict)
                    and sample.get("candidate_id") == proposal["candidate_id"]
                    and row["classification"] == "declaration_or_reservation_not_execution_proof")
                row["resolved"] = True
                row.pop("needs_triage", None)
            except Exception as error:
                row["resolved"] = False
                row["triage_error"] = f"{type(error).__name__}: {error}"
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or file_sha(path) != before_hash:
            raise RuntimeError(f"metadata changed during triage: {path}")
        write_new(output/f"file-{index:05d}.json", row)
        result["inventory"].append(row)
    if file_sha(pointer["path"]) != pointer["sha256"]:
        raise RuntimeError("external proposal bytes changed during triage")
    if (file_sha(inventory_path) != sha256(raw).hexdigest() or file_sha(__file__) != driver_hash
            or file_sha(warehouse_seed_audit.__file__) != report["auditor_sha256"]):
        raise RuntimeError("inventory or parameterized source changed during triage")
    result["summary"] = summarize(result["inventory"], families,
                                  selection_stats=report.get("selection_stats"),
                                  missing_roots=report.get("missing_roots", ()))
    write_new(output/"result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gzip-expanded-bytes", type=int, default=256*1024*1024)
    args = parser.parse_args(argv)
    if args.max_gzip_expanded_bytes < 1:
        parser.error("positive gzip expansion bound required")
    report = triage(args.inventory, args.proposal, args.output_dir,
                    args.max_gzip_expanded_bytes)
    print(json.dumps({"result": str(args.output_dir/"result.json"),
        "unresolved": len(report["summary"]["unresolved_paths"]), "authorization_granted": False}))
    return 0 if not report["summary"]["unresolved_paths"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
