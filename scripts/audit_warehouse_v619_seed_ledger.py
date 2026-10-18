#!/usr/bin/env python3
"""Bounded, read-only metadata audit; never creates an environment or reads weights."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = tuple(ROOT.parent / name for name in (
    "XAI", "XAI-study-v3", "XAI-v6-integration",
    "XAI-v6-recovery-20261015T165508Z"))
NAME = re.compile(r"auth|claim|reset|seed|schedule|receipt|acceptance|training|report|progress|ledger|plan|manifest|config|evaluation|result|predeclared|lock", re.I)
PRUNE = {".git", "node_modules", ".venv", "__pycache__", ".next", "dist", "build"}
MAX_BYTES = 12 * 1024 * 1024


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def action_seed(namespace, base, task, stream):
    value = sha256(f"{namespace}|{base}|{task}|{stream}".encode()).digest()
    return int.from_bytes(value[:8], "big") & 0x7fffffff


def proposed_families(high=True):
    offset = 100_000_000 if high else 0
    families = {"development_pair": set(range(6153, 6156))}
    for name, start, count in (
        ("quality", 265100, 24), ("stress", 265200, 256),
        ("program_audit", 265500, 24), ("fixed_fit", 266000, 50),
        ("fixed_calibration", 266050, 25), ("fixed_eligibility", 266075, 25),
        ("fixed_validation", 266100, 24), ("fixed_verification", 266200, 24)):
        base = set(range(offset + start, offset + start + count))
        families[name + ":base"] = base
        # Check expansion for all blocks, even where execution uses direct seeds.
        families[name + ":task_expanded"] = {s * 10 + t for s in base for t in (1, 2, 3)}
        if name in ("stress", "fixed_verification"):
            namespace = ("warehouse-v6-stress" if name == "stress" else
                         "warehouse-v619-01|fixed-verification")
            families[name + ":rng_proposal"] = {
                action_seed(namespace, s, t, stream)
                for s in base for t in (1, 2, 3) for stream in range(3)}
    # Conservative superset of stage-derived BC scenes and critic probe seeds.
    families["development_bc_stage_superset"] = {
        pair * 100000 + stage * 1000 + index
        for pair in range(6153, 6156) for stage in range(10, 77)
        for index in range(256)}
    families["development_critic"] = {
        pair + 6200 + index for pair in range(6153, 6156) for index in range(2)}
    families["fixed_verification:rng_contiguous_backup"] = set(range(1700000000, 1700000216))
    return families


def classify(path):
    value = str(path).lower()
    if "claim" in path.name.lower() and any(s in value for s in ("sealed", "heldout", "audit", "claim")):
        return "claim_counts_as_spent_regardless_status"
    if "reset" in path.name.lower() and path.suffix == ".jsonl":
        return "reset_journal"
    if any(s in path.name.lower() for s in ("authorization", "auth", "lock", "seed", "schedule")):
        return "reservation_or_authorization"
    if any(s in path.name.lower() for s in ("config", "manifest", "plan")):
        return "declaration_not_execution_proof"
    return "receipt_or_report_not_independently_replayed"


def extract(value):
    """Only explicit seed keys/ranges count; ordinary measurements do not."""
    seeds, intervals = set(), set()

    def visit(obj, context=""):
        if isinstance(obj, dict):
            if (type(obj.get("start")) is int and
                    type(obj.get("end_exclusive")) is int and
                    obj["start"] < obj["end_exclusive"]):
                intervals.add((obj["start"], obj["end_exclusive"]))
            for key, item in obj.items():
                key = str(key).lower()
                if "seed" in key and not any(x in key for x in ("sha", "hash", "count", "offset", "schema")):
                    visit(item, key)
                elif isinstance(item, (dict, list)):
                    visit(item, context)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item, context)
        elif type(obj) is int and context:
            seeds.add(obj)
    visit(value)
    return seeds, intervals


def candidates(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in PRUNE)
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or "v619" in name.lower():
                continue  # New declarations cannot serve as evidence of historical use.
            if path.suffix not in (".json", ".jsonl", ".toml", ".yaml", ".yml") or not NAME.search(name):
                continue
            if "warehouse" not in str(path).lower() and not any(s in str(path).lower() for s in ("authoriz", "seed", "claim")):
                continue
            yield path


def audit(roots, *, max_bytes=MAX_BYTES):
    low, high = proposed_families(False), proposed_families(True)
    targets = {"low/" + k: v for k, v in low.items()}
    targets.update({"high/" + k: v for k, v in high.items()})
    target_union = set().union(*targets.values())
    inventory, skipped, references, intervals = [], [], defaultdict(list), defaultdict(list)
    observed = set()
    for root in roots:
        if not root.is_dir():
            skipped.append({"path": str(root), "reason": "missing_root"})
            continue
        for path in candidates(root):
            size = path.stat().st_size
            if size > max_bytes:
                skipped.append({"path": str(path), "bytes": size, "reason": "bounded_metadata_size_limit"})
                continue
            raw = path.read_bytes()
            item = {"path": str(path), "sha256": sha256(raw).hexdigest(),
                    "bytes": len(raw), "classification": classify(path)}
            seeds, file_intervals = set(), set()
            try:
                if path.suffix == ".jsonl":
                    for line in raw.splitlines():
                        if line.strip():
                            current, spans = extract(json.loads(line))
                            seeds.update(current); file_intervals.update(spans)
                else:
                    value = (tomllib.loads(raw.decode()) if path.suffix == ".toml" else json.loads(raw))
                    seeds, file_intervals = extract(value)
            except (ValueError, UnicodeError) as error:
                item["parse_error"] = type(error).__name__
                skipped.append({"path": str(path), "reason": "metadata_parse_error", "sha256": item["sha256"]})
            item["explicit_seed_count"] = len(seeds)
            item["reserved_intervals"] = [list(v) for v in sorted(file_intervals)]
            item["proposed_seed_references"] = sorted(seeds & target_union)
            inventory.append(item)
            observed.update(seeds)
            for seed in seeds & target_union:
                references[seed].append({"path": str(path), "classification": item["classification"], "sha256": item["sha256"]})
            for span in file_intervals:
                intervals[span].append({"path": str(path), "sha256": item["sha256"]})
    # The template is an additional reservation declaration, not an execution receipt.
    template = ROOT / "backend/training/seed_ledger.py"
    raw = template.read_bytes()
    if b"500_004" in raw and b"10_500_204" in raw:
        intervals[(500004, 10500204)].append({"path": str(template),
            "sha256": sha256(raw).hexdigest(), "classification": "source_template_reservation"})
    summary = {}
    for name, values in targets.items():
        hits = sorted(values & observed)
        spans = []
        for (start, end), sources in sorted(intervals.items()):
            covered = sorted(seed for seed in values if start <= seed < end)
            if covered:
                spans.append({"start": start, "end_exclusive": end,
                    "overlap_count": len(covered), "overlap_sample": covered[:8],
                    "source_count": len(sources), "source_examples": sources[:2]})
        summary[name] = {"count": len(values), "min": min(values), "max": max(values),
            "seed_set_sha256": digest(sorted(values)), "explicit_reference_count": len(hits),
            "explicit_reference_examples": [{"seed": seed, "sources": references[seed][:3]} for seed in hits[:5]],
            "reservation_overlaps": spans,
            "status": "blocked_by_historical_reference_or_reservation" if hits or spans else
                      "no_reference_in_bounded_scanned_metadata_not_global_freshness_proof"}
    return {"schema": "warehouse-v619-bounded-seed-ledger.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "publishable": False,
        "environment_resets": 0, "training_steps": 0, "read_model_weight_files": False,
        "scan_roots": list(map(str, roots)), "max_metadata_file_bytes": max_bytes,
        "inventory_count": len(inventory), "unique_content_count": len({x["sha256"] for x in inventory}),
        "read_metadata_bytes": sum(x["bytes"] for x in inventory),
        "classification_counts": dict(Counter(x["classification"] for x in inventory)),
        "skipped": skipped, "proposals": summary,
        "limitations": ["Filename-selected metadata only; no whole-disk negative claim.",
            "Opaque or skipped artifacts and unrecorded runs cannot establish freshness.",
            "Any discovered sealed claim is spent even when its status failed or interrupted.",
            "Configurations/reservations are not execution proof but conservatively block reuse.",
            "RNG proposals are namespace-bound suggestions, not registered schedules.",
            "The separate current_ui_source checkout and external systems are outside these four roots."],
        "inventory": inventory}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(tuple(p.resolve() for p in (args.root or DEFAULT_ROOTS)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(args.output), "sha256": sha256(args.output.read_bytes()).hexdigest(),
        "metadata_files": report["inventory_count"], "skipped": len(report["skipped"]),
        "proposals": {key: {"references": value["explicit_reference_count"],
            "interval_overlaps": sum(x["overlap_count"] for x in value["reservation_overlaps"])}
            for key, value in report["proposals"].items()}}))


if __name__ == "__main__":
    main()
