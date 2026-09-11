"""Build and replay the frozen r4.1 prediction question bank.

The public payload intentionally keeps the existing portable projection
schema.  Its private provenance is rebound to the frozen r4.1 Actor, conflict
manifest and conflict runtime; all eight answers are recomputed on load.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r4_question_bank as base
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_conflict_scenarios import validate_conflict_manifest
from backend.warehouse_r41_online_runtime import R41OnlineAlignmentRuntime


VERSION = "warehouse-r41-frozen-question-bank.v1"
PROJECTION_VERSION = base.PROJECTION_VERSION
VIEW_VERSION = base.VIEW_VERSION
PORTABLE_BANK_VERSION = base.PORTABLE_BANK_VERSION


def producer_sources() -> dict[str, str]:
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    return local_source_hashes((Path(__file__), Path(base.__file__)))


def _with_base(pool_manifest_sha256: str):
    replacements = {
        "VERSION": VERSION,
        "REGISTERED_POOL_MANIFEST_SHA256": pool_manifest_sha256,
        "_source_binding": producer_sources,
    }
    previous = {key: getattr(base, key) for key in replacements}
    return previous, replacements


def _call_base(name: str, pool_manifest_sha256: str, *args, **kwargs):
    previous, replacements = _with_base(pool_manifest_sha256)
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        return getattr(base, name)(*args, **kwargs)
    finally:
        for key, value in previous.items():
            setattr(base, key, value)


def _pool(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_conflict_manifest(manifest, replay=True)
    rows = manifest.get("splits", {}).get("final_test")
    if (not isinstance(rows, list) or len(rows) < 8
            or len({row.get("id") for row in rows}) != len(rows)):
        raise ValueError("R4.1 frozen final-test pool is incomplete")
    return {"test_fixture": False, "scenes": deepcopy(rows)}


def build(runtime: R41OnlineAlignmentRuntime, manifest: Mapping[str, Any], *,
          output: str | Path, manifest_file_sha256: str) -> dict[str, Any]:
    if type(runtime) is not R41OnlineAlignmentRuntime:
        raise ValueError("Exact r4.1 runtime required for question-bank generation")
    runtime.verify_binding()
    report = _call_base(
        "build", manifest_file_sha256, runtime, _pool(manifest),
        output=output, pool_manifest_sha256=manifest_file_sha256,
    )
    report["version"] = VERSION
    report["conflict_manifest_content_sha256"] = manifest["content_sha256"]
    report["conflict_contract_sha256"] = manifest["contract_sha256"]
    report["conflict_graph_sha256"] = manifest["conflict_graph_sha256"]
    # The base writer ran before these r4.1-only report bindings were added.
    # Rewrite only that newly-created report atomically via a sibling file.
    report_path = Path(output).expanduser().absolute() / "report.json"
    replacement = report_path.with_name("report.r41.tmp")
    base._write_private(replacement, report)
    replacement.replace(report_path)
    return report


def validate_payload(runtime: R41OnlineAlignmentRuntime,
                     payload: Mapping[str, Any]) -> dict[str, Any]:
    if type(runtime) is not R41OnlineAlignmentRuntime:
        raise ValueError("Exact r4.1 runtime required for question-bank replay")
    pool_sha = payload.get("pool_manifest_sha256") if isinstance(payload, Mapping) else None
    if not isinstance(pool_sha, str):
        raise ValueError("R4.1 question bank lacks its frozen manifest binding")
    result = _call_base("validate_payload", pool_sha, runtime, payload)
    if payload.get("runtime_family") != "alignment_feedback197":
        raise ValueError("R4.1 question bank runtime family differs")
    return result


public_items = base.public_items


def main(argv: Sequence[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--conflict-manifest", type=Path, required=True)
    parser.add_argument("--expected-conflict-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.conflict_manifest.expanduser().resolve()
    if (not manifest_path.is_file() or manifest_path.is_symlink()
            or file_hash(manifest_path) != args.expected_conflict_manifest_sha256):
        parser.error("conflict manifest bytes differ from expected SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    runtime = R41OnlineAlignmentRuntime(
        args.actor, protocol=protocol,
        expected_actor_sha256=file_hash(args.actor),
        expected_protocol_sha256=digest(protocol), allow_test_fixture=False,
    )
    report = build(runtime, manifest, output=args.output,
                   manifest_file_sha256=args.expected_conflict_manifest_sha256)
    print(canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "PROJECTION_VERSION", "build", "validate_payload",
    "producer_sources", "public_items", "main",
]
