"""Frozen eight-item question bank for the r4.1 diagnostic runtime."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r4_question_bank as base
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)


VERSION = "warehouse-r41-diagnostic-frozen-question-bank.v3"
SCENE_MANIFEST_VERSION = "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
WORKLOAD_SCREEN_VERSION = "warehouse-r41-diagnostic-workload-screen.v2"
PROJECTION_VERSION = base.PROJECTION_VERSION
VIEW_VERSION = base.VIEW_VERSION
PORTABLE_BANK_VERSION = base.PORTABLE_BANK_VERSION
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    # The base module imports the production-admission hasher only for its own
    # source receipt.  This diagnostic wrapper replaces that hook with the
    # function above, so release/UI dependencies are deliberately excluded.
    base_directory = Path(base.__file__).parent
    unused_dependencies = (
        base_directory / "warehouse_r4_production_admission.py",
        base_directory / "warehouse_family_question_pool.py",
    )
    return local_source_hashes(
        (Path(__file__), Path(base.__file__)),
        exclude_paths=unused_dependencies,
    )


def _patched(manifest_sha256: str):
    replacements = {
        "VERSION": VERSION,
        "REGISTERED_POOL_MANIFEST_SHA256": manifest_sha256,
        "_source_binding": producer_sources,
    }
    previous = {key: getattr(base, key) for key in replacements}
    return previous, replacements


def _call(name: str, manifest_sha256: str, *args, **kwargs):
    previous, replacements = _patched(manifest_sha256)
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        return getattr(base, name)(*args, **kwargs)
    finally:
        for key, value in previous.items():
            setattr(base, key, value)


def _pool(manifest: Mapping[str, Any]) -> dict[str, Any]:
    splits = manifest.get("splits")
    rows = splits.get("question_bank") if isinstance(splits, dict) else None
    if not isinstance(rows, list) or len(rows) != 36:
        raise ValueError("Exact independent 36-scene diagnostic question split required")
    question = {row.get("fingerprint") for row in rows}
    excluded = {row.get("fingerprint") for name, values in splits.items()
                if name != "question_bank" and isinstance(values, list) for row in values}
    if len(question) != len(rows) or question & excluded:
        raise ValueError("Diagnostic question scenes overlap another split")
    return {"test_fixture": False, "scenes": deepcopy(rows)}


def build(runtime: R41DiagnosticOnlineAlignmentRuntime,
          manifest: Mapping[str, Any], *, output: str | Path,
          manifest_file_sha256: str) -> dict[str, Any]:
    if type(runtime) is not R41DiagnosticOnlineAlignmentRuntime:
        raise ValueError("Exact diagnostic runtime required for question-bank generation")
    runtime.verify_binding()
    if (manifest.get("version") != SCENE_MANIFEST_VERSION
            or manifest.get("frozen_actor") != {
                "sha256": runtime.actor_sha256,
                "actor_parameters_sha256": runtime.actor.metadata.get(
                    "actor_parameters_sha256")}
            or not isinstance(manifest.get("workload_screen"), Mapping)
            or manifest["workload_screen"].get("version")
                != WORKLOAD_SCREEN_VERSION):
        raise ValueError("Exact workload-screened diagnostic manifest required")
    report = _call("build", manifest_file_sha256, runtime, _pool(manifest),
                   output=output, pool_manifest_sha256=manifest_file_sha256)
    report["version"] = VERSION
    report["diagnostic_manifest_content_sha256"] = manifest["content_sha256"]
    report["diagnostic_manifest_semantic_sha256"] = digest(manifest)
    report["source_full_manifest_bindings"] = deepcopy(
        runtime.source_full_manifest_bindings)
    path = Path(output).expanduser().absolute() / "report.json"
    replacement = path.with_name("report.diagnostic.tmp")
    base._write_private(replacement, report)
    replacement.replace(path)
    read_saved_report(output,
        expected_report_sha256=file_hash(path),
        expected_question_bank_sha256=file_hash(Path(output) / "question_bank.json"),
        runtime=runtime, manifest=manifest,
        manifest_file_sha256=manifest_file_sha256)
    return report


def validate_payload(runtime: R41DiagnosticOnlineAlignmentRuntime,
                     payload: Mapping[str, Any]) -> dict[str, Any]:
    if type(runtime) is not R41DiagnosticOnlineAlignmentRuntime:
        raise ValueError("Exact diagnostic runtime required for question-bank replay")
    manifest_sha = payload.get("pool_manifest_sha256") if isinstance(payload, Mapping) else None
    if not isinstance(manifest_sha, str):
        raise ValueError("Diagnostic question bank lacks its manifest binding")
    return _call("validate_payload", manifest_sha, runtime, payload)


public_items = base.public_items


def read_saved_report(output: str | Path, *, expected_report_sha256: str,
                      expected_question_bank_sha256: str,
                      runtime: R41DiagnosticOnlineAlignmentRuntime,
                      manifest: Mapping[str, Any],
                      manifest_file_sha256: str) -> dict[str, Any]:
    """Authenticate, regenerate, and replay a saved diagnostic question bank."""
    if type(runtime) is not R41DiagnosticOnlineAlignmentRuntime:
        raise ValueError("Exact diagnostic runtime required for question-bank evidence")
    runtime.verify_binding()
    output = Path(output).expanduser().absolute()
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output):
        raise ValueError("Diagnostic question-bank directory is missing or unsafe")
    report_path = output / "report.json"
    bank_path = output / "question_bank.json"
    if (_HEX.fullmatch(str(expected_report_sha256)) is None
            or _HEX.fullmatch(str(expected_question_bank_sha256)) is None
            or _HEX.fullmatch(str(manifest_file_sha256)) is None
            or report_path.is_symlink() or not report_path.is_file()
            or bank_path.is_symlink() or not bank_path.is_file()
            or file_hash(report_path) != expected_report_sha256
            or file_hash(bank_path) != expected_question_bank_sha256):
        raise ValueError("Diagnostic question-bank evidence hash differs")
    if (not isinstance(manifest, dict)
            or manifest.get("version") != SCENE_MANIFEST_VERSION
            or manifest.get("frozen_actor") != {
                "sha256": runtime.actor_sha256,
                "actor_parameters_sha256": runtime.actor.metadata.get(
                    "actor_parameters_sha256")}
            or manifest.get("content_sha256") is None
            or runtime.manifest_file_sha256 != manifest_file_sha256
            or runtime.manifest_content_sha256 != manifest.get("content_sha256")
            or runtime.manifest_semantic_sha256 != digest(manifest)):
        raise ValueError("Diagnostic question-bank manifest binding differs")
    content = deepcopy(dict(manifest)); claimed = content.pop("content_sha256")
    if claimed != digest(content):
        raise ValueError("Diagnostic question-bank manifest content differs")
    pool = _pool(manifest)
    payload = base._read_json(bank_path)
    replayed = validate_payload(runtime, payload)

    previous, replacements = _patched(manifest_file_sha256)
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        candidates = base._collect(runtime, pool["scenes"])
        selected_next, selected_wait = base._select(candidates)
        regenerated_items = base._items(runtime, selected_next, selected_wait)
    finally:
        for key, value in previous.items():
            setattr(base, key, value)
    if (digest(regenerated_items) != digest(payload.get("items"))
            or replayed.get("items") != payload.get("items")):
        raise ValueError("Diagnostic question selection differs from its frozen pool")
    report = base._read_json(report_path)
    expected = {
        "version": VERSION, "status": "candidate_ready",
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature,
        "pool_manifest_sha256": manifest_file_sha256,
        "candidate_frames": len(candidates), "selected_scenarios": 8,
        "checks": payload["checks"],
        "source_bank_signature": payload["source_bank_signature"],
        "payload_sha256": digest(payload), "formal_ready": False,
        "diagnostic_manifest_content_sha256": manifest["content_sha256"],
        "diagnostic_manifest_semantic_sha256": digest(manifest),
        "source_full_manifest_bindings": deepcopy(
            runtime.source_full_manifest_bindings),
    }
    if digest(report) != digest(expected):
        raise ValueError("Diagnostic question-bank report differs from replayed evidence")
    return report


def _runtime(actor: Path, protocol: Path, manifest_path: Path,
             manifest: Mapping[str, Any]) -> R41DiagnosticOnlineAlignmentRuntime:
    content = deepcopy(dict(manifest)); claimed = content.pop("content_sha256", None)
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    return R41DiagnosticOnlineAlignmentRuntime(
        actor, training_protocol_path=protocol, manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor),
        expected_training_protocol_file_sha256=file_hash(protocol),
        expected_training_protocol_content_sha256=digest(protocol_value),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(manifest),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.expanduser().absolute()
    if file_hash(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("Diagnostic question manifest bytes differ")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = _runtime(args.actor.absolute(), args.protocol.absolute(), manifest_path, manifest)
    report = build(runtime, manifest, output=args.output,
                   manifest_file_sha256=args.expected_manifest_sha256)
    print(canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "SCENE_MANIFEST_VERSION", "WORKLOAD_SCREEN_VERSION",
           "PROJECTION_VERSION", "VIEW_VERSION", "PORTABLE_BANK_VERSION",
           "producer_sources", "build", "validate_payload", "read_saved_report",
           "public_items", "main"]
