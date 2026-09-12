"""Freeze a program-blind development-validation supplement for r4.1.

The source candidate pool predates every RCPD fit.  This module chooses a new
64-scene development-validation split after excluding every registered,
participant, and retired-final scene.  Selection uses only the public workload
contract and the frozen Actor's ability to execute that workload; it never
loads an explanation program or final-audit labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS, VERSION as MANIFEST_VERSION, validate_diagnostic_manifest,
)
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
    VERSION as WORKLOAD_VERSION,
    screen_scene,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-development-supplement.v1"
TOTAL_SCENES = 64
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
ORDER_SALT = "r41-diagnostic-v7-development-validation-20260912"
ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 512 * 1024 * 1024


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_manifest_version": MANIFEST_VERSION,
        "workload_screen_version": WORKLOAD_VERSION,
        "workload_screen_contract_sha256": WORKLOAD_CONTRACT_SHA256,
        "purpose": "fresh development validation for RCPD v7",
        "selection_population": "source manifest play_candidates only",
        "selection_order": "ascending sha256(order salt, scene fingerprint)",
        "selection_rule": (
            "first conflict-validation-workload-safe scene in each family after "
            "excluding all registered, participant, and retired-final scenes"
        ),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "scene_count": TOTAL_SCENES,
        "program_access": False,
        "program_predictions_access": False,
        "participant_data_access": False,
        "final_audit_rows_access": False,
        "retired_final_use": "scene fingerprints for exclusion only",
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        ROOT / "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
    )
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def _read(path: Path) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError("Development-supplement JSON is missing, linked, or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Development-supplement JSON must be an object")
    return value


def _write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Development-supplement output path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _retired_scene_fingerprints(path: Path, expected_versions: set[str]) -> set[str]:
    value = _read(path)
    scenes = value.get("scenes")
    if (value.get("version") not in expected_versions
            or value.get("program_access") is not False
            or value.get("statistics", {}).get("accepted") != TOTAL_SCENES
            or value.get("content_sha256") != digest({
                key: item for key, item in value.items() if key != "content_sha256"
            })
            or not isinstance(scenes, list) or len(scenes) != TOTAL_SCENES):
        raise ValueError("Retired final registry identity differs")
    result = {str(row.get("fingerprint")) for row in scenes}
    if len(result) != TOTAL_SCENES:
        raise ValueError("Retired final scene registry contains duplicates")
    return result


def _ordered_candidates(manifest: Mapping[str, Any], family: str) -> list[dict]:
    rows = [deepcopy(row) for batch in manifest["candidate_batches"]
            for row in batch if row["family_id"] == family]
    rows.sort(key=lambda row: digest({"salt": ORDER_SALT,
                                     "fingerprint": row["fingerprint"]}))
    return rows


def build(*, actor_path: str | Path, manifest_path: str | Path,
          selected_scenes_path: str | Path,
          retired_holdout_paths: Sequence[str | Path],
          output: str | Path) -> dict[str, Any]:
    actor_path = Path(actor_path).expanduser().absolute()
    manifest_path = Path(manifest_path).expanduser().absolute()
    selected_path = Path(selected_scenes_path).expanduser().absolute()
    retired_paths = [Path(value).expanduser().absolute()
                     for value in retired_holdout_paths]
    inputs = [actor_path, manifest_path, selected_path, *retired_paths]
    if (len(retired_paths) != 2 or any(
            not path.is_file() or path.is_symlink() or path.resolve() != path
            for path in inputs)):
        raise ValueError("Exact canonical Actor, manifest, selection and two retired finals required")
    manifest = _read(manifest_path)
    validate_diagnostic_manifest(manifest, replay=False)
    selected = _read(selected_path)
    if (selected.get("release_eligible") is not True
            or selected.get("actor_sha256") != file_hash(actor_path)):
        raise ValueError("Development-supplement selected-scene binding differs")
    excluded = {row["fingerprint"] for rows in manifest["splits"].values()
                for row in rows}
    excluded.update(row["fingerprint"] for key in ("X", "Y")
                    for row in selected[key])
    excluded.update(_retired_scene_fingerprints(
        retired_paths[0], {"warehouse-r41-diagnostic-fresh-final-holdout.v1"}))
    excluded.update(_retired_scene_fingerprints(
        retired_paths[1], {"warehouse-r41-diagnostic-fresh-final-holdout.v2"}))
    actor = NumPyNativeActor(actor_path)
    accepted: list[dict] = []
    trace: list[dict] = []
    for family in FAMILY_IDS:
        count = 0
        for candidate in _ordered_candidates(manifest, family):
            if count >= FAMILY_QUOTAS[family]:
                break
            reason = None
            receipt = None
            if candidate["fingerprint"] in excluded:
                reason = "excluded_scene"
            else:
                receipt = screen_scene(candidate, split="conflict_validation",
                                       scene_index=len(accepted), actor=actor)
                if receipt.get("passed") is not True:
                    reason = "development_workload_failed"
            admitted = reason is None
            trace.append({
                "family_id": family, "fingerprint": candidate["fingerprint"],
                "accepted": admitted, "rejection_reason": reason,
                "workload_receipt": receipt,
            })
            if admitted:
                frozen = deepcopy(candidate)
                frozen["id"] = f"diagnostic_v7_development_validation_{len(accepted):04d}"
                frozen["split"] = "development_validation"
                frozen["workload_screen"] = receipt
                accepted.append(frozen)
                excluded.add(frozen["fingerprint"])
                count += 1
        if count != FAMILY_QUOTAS[family]:
            raise RuntimeError("Development family quota unavailable: " + family)
    statistics = {
        "accepted": len(accepted), "evaluated": len(trace),
        "families": dict(sorted(Counter(row["family_id"] for row in accepted).items())),
        "rejected": dict(sorted(Counter(row["rejection_reason"] for row in trace
                                        if row["rejection_reason"]).items())),
        "unique_fingerprints": len({row["fingerprint"] for row in accepted}),
    }
    bindings = {
        "actor_sha256": file_hash(actor_path),
        "source_manifest_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "selected_scenes_sha256": file_hash(selected_path),
        "retired_holdout_sha256": [file_hash(path) for path in retired_paths],
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(producer_sources()),
    }
    value = {"version": VERSION, "status": "passed", "contract": contract(),
             "bindings": bindings, "scenes": accepted, "selection_trace": trace,
             "statistics": statistics, "program_access": False,
             "final_audit_rows_access": False, "formal_ready": False}
    value["content_sha256"] = digest(value)
    output = Path(output).expanduser().absolute()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    _write(output / "development_supplement.json", value)
    report = {"version": VERSION, "status": "passed", "bindings": bindings,
              "supplement_file_sha256": file_hash(output / "development_supplement.json"),
              "supplement_content_sha256": value["content_sha256"],
              "statistics": statistics, "producer_sources": producer_sources(),
              "program_access": False, "final_audit_rows_access": False,
              "formal_ready": False}
    _write(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selected-scenes", required=True)
    parser.add_argument("--retired-holdout", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(actor_path=args.actor, manifest_path=args.manifest,
        selected_scenes_path=args.selected_scenes,
        retired_holdout_paths=args.retired_holdout, output=args.output)
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "TOTAL_SCENES", "FAMILY_QUOTAS", "contract",
           "producer_sources", "build", "main"]
