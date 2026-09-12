"""Freeze the program-blind development expansion for diagnostic RCPD v8.

The immutable conflict manifest contains a play-candidate population generated
before any explanation program was fitted.  This registry first freezes 64 new
development-validation scenes, then 128 disjoint fit-supplement scenes.  It
uses only scene identity, the public workload contract, and the frozen Actor.
No explanation program, program prediction, participant record, or final-audit
label is an input to this producer.

The validation split is selected first so later fit selection cannot influence
which scenes are held out.  Both split streams and all six family quotas are
fixed by this source.  Existing registered splits, formal X/Y scenes, the v1
development supplement, and the two retired fresh-final registries are used
only as exclusion registries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_designation as designation_api
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS,
    VERSION as MANIFEST_VERSION,
    validate_diagnostic_manifest,
)
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
    VERSION as WORKLOAD_VERSION,
    load_frozen_actor,
    screen_scene,
)


VERSION = "warehouse-r41-diagnostic-development-expansion.v8"
STATUS = "passed_program_blind_registry"
ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 512 * 1024 * 1024

# These offsets describe the eventual v8 row-collection order.  The registered
# 128 train and 64 conflict-validation scenes already occupy indexes 0..191.
BASE_DEVELOPMENT_FIT_SCENES = 192
FIT_SUPPLEMENT_SCENES = 128
TOTAL_DEVELOPMENT_FIT_SCENES = BASE_DEVELOPMENT_FIT_SCENES + FIT_SUPPLEMENT_SCENES
VALIDATION_SCENES = 64

FIT_FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (22, 22, 21, 21, 21, 21)))
VALIDATION_FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
VALIDATION_ORDER_SALT = (
    "r41-diagnostic-v8-program-blind-development-validation-20260912"
)
FIT_ORDER_SALT = "r41-diagnostic-v8-program-blind-fit-supplement-20260912"

EXPECTED_DESIGNATION_SHA256 = (
    "d80f2736c6d5e359764f6ae4c09ccfa277f26e98a8910c1f18d86d84cb99851f"
)
EXPECTED_MANIFEST_SHA256 = (
    "af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_manifest_version": MANIFEST_VERSION,
        "workload_screen_version": WORKLOAD_VERSION,
        "workload_screen_contract_sha256": WORKLOAD_CONTRACT_SHA256,
        "purpose": "program-blind expanded development registry for RCPD v8",
        "selection_population": "source manifest play_candidates only",
        "selection_sequence": ["development_validation", "fit_supplement"],
        "selection_order": {
            "development_validation": (
                "ascending sha256(validation salt, family, scene fingerprint)"
            ),
            "fit_supplement": (
                "ascending sha256(fit salt, family, scene fingerprint)"
            ),
        },
        "selection_rule": (
            "first exact-RCPD-workload-safe scene in each family after excluding "
            "all registered, formal X/Y, previous-development, retired-final, "
            "and already accepted expansion scenes"
        ),
        "fit_scene_index_offset": BASE_DEVELOPMENT_FIT_SCENES,
        "fit_supplement_scene_count": FIT_SUPPLEMENT_SCENES,
        "fit_family_quotas": deepcopy(FIT_FAMILY_QUOTAS),
        "validation_train_count": TOTAL_DEVELOPMENT_FIT_SCENES,
        "validation_scene_count": VALIDATION_SCENES,
        "validation_family_quotas": deepcopy(VALIDATION_FAMILY_QUOTAS),
        "validation_frozen_before_fit_selection": True,
        "program_access": False,
        "program_predictions_access": False,
        "actor_logits_access": False,
        "actor_hidden_state_access": False,
        "participant_data_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "retired_final_use": "registry identity and scene exclusion only",
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    """Return the explicit source set that defines registry selection.

    Keeping this closure explicit prevents unrelated deployment or explanation
    code from becoming an input to program-blind scene selection.
    """
    paths = (
        Path(__file__).resolve(),
        ROOT / "backend/training/warehouse_native_common.py",
        ROOT / "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
        ROOT / "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        Path(designation_api.__file__).resolve(),
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "env/warehouse_native/policy.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
    )
    result: dict[str, str] = {}
    for path in paths:
        path = path.resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError("Development-expansion producer source is unsafe")
        result[path.relative_to(ROOT.resolve()).as_posix()] = file_hash(path)
    return dict(sorted(result.items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _read(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (
        type(claimed) is str
        and _HEX.fullmatch(claimed) is not None
        and claimed == digest({key: item for key, item in value.items()
                               if key != "content_sha256"})
    )


def _scene_identities(rows: Any, *, expected_count: int, label: str) -> tuple[set[str], set[int]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(label + " scene count differs")
    fingerprints: set[str] = set()
    seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(label + " scene schema differs")
        fingerprint = row.get("fingerprint")
        seed = row.get("seed")
        if (_HEX.fullmatch(str(fingerprint)) is None
                or type(seed) is not int or seed < 0
                or fingerprint in fingerprints or seed in seeds):
            raise ValueError(label + " scene identity differs")
        fingerprints.add(str(fingerprint))
        seeds.add(seed)
    return fingerprints, seeds


def _validate_designation(value: Mapping[str, Any], *, actor_sha256: str) -> None:
    bindings = value.get("bindings")
    if (value.get("version") != designation_api.VERSION
            or value.get("status") != designation_api.STATUS
            or value.get("designated") is not True
            or value.get("release_class") != designation_api.RELEASE_CLASS
            or value.get("behavior_performance_gate_waived") is not True
            or value.get("waiver_scope") != ["behavior_performance"]
            or value.get("runtime_action_override") is not False
            or value.get("formal_ready") is not False
            or value.get("formal_sample_eligible") is not False
            or value.get("test_fixture") is not False
            or not isinstance(bindings, Mapping)
            or bindings.get("actor_sha256") != actor_sha256
            or bindings.get("actor_parameters_sha256")
                != designation_api.EXPECTED_ACTOR_PARAMETERS_SHA256):
        raise ValueError("Exact frozen diagnostic Actor designation required")


def _validate_selected(value: Mapping[str, Any], *, actor_sha256: str,
                       manifest_path: Path, manifest: Mapping[str, Any]) -> tuple[set[str], set[int]]:
    if (value.get("version") != "warehouse-r41-diagnostic-conflict-dynamic-selection.v3"
            or value.get("release_eligible") is not True
            or value.get("actor_sha256") != actor_sha256
            or value.get("source_manifest_file_sha256") != file_hash(manifest_path)
            or value.get("source_manifest_content_sha256") != manifest.get("content_sha256")
            or value.get("six_distinct_conflict_families") is not True
            or value.get("zero_action_overrides") is not True):
        raise ValueError("Formal X/Y selection binding differs")
    rows = [*value.get("X", []), *value.get("Y", [])]
    return _scene_identities(rows, expected_count=6, label="formal X/Y")


def _validate_previous_development(
    value: Mapping[str, Any], *, actor_sha256: str, manifest_path: Path,
    selected_path: Path,
) -> tuple[set[str], set[int]]:
    bindings = value.get("bindings")
    if (value.get("version") != "warehouse-r41-diagnostic-development-supplement.v1"
            or value.get("status") != "passed"
            or value.get("program_access") is not False
            or value.get("final_audit_rows_access") is not False
            or not _content_valid(value) or not isinstance(bindings, Mapping)
            or bindings.get("actor_sha256") != actor_sha256
            or bindings.get("source_manifest_sha256") != file_hash(manifest_path)
            or bindings.get("selected_scenes_sha256") != file_hash(selected_path)
            or value.get("statistics", {}).get("accepted") != 64):
        raise ValueError("Previous development supplement identity differs")
    return _scene_identities(
        value.get("scenes"), expected_count=64, label="previous development"
    )


def _validate_retired_holdout(
    value: Mapping[str, Any], *, expected_version: str,
) -> tuple[set[str], set[int]]:
    statistics = value.get("statistics")
    if (value.get("version") != expected_version
            or value.get("program_access") is not False
            or value.get("contract", {}).get("program_predictions_access") is not False
            or value.get("contract", {}).get("final_labels_used_for_selection") is not False
            or not _content_valid(value) or not isinstance(statistics, Mapping)
            or statistics.get("accepted") != 64
            or statistics.get("public_observation_overlap") != 0):
        raise ValueError("Retired fresh-final registry identity differs")
    return _scene_identities(
        value.get("scenes"), expected_count=64,
        label="retired " + expected_version,
    )


def _ordered_candidates(
    manifest: Mapping[str, Any], family: str, *, salt: str,
) -> list[dict[str, Any]]:
    rows = [
        deepcopy(row)
        for batch in manifest["candidate_batches"]
        for row in batch
        if row["family_id"] == family
    ]
    rows.sort(key=lambda row: digest({
        "salt": salt,
        "family_id": family,
        "fingerprint": row["fingerprint"],
    }))
    return rows


def _select_split(
    *, manifest: Mapping[str, Any], actor: Any, split_name: str,
    quotas: Mapping[str, int], salt: str,
    excluded_fingerprints: set[str], excluded_seeds: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_name not in {"development_validation", "fit_supplement"}:
        raise ValueError("Unknown development-expansion split")
    accepted: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        family_count = 0
        for rank, candidate in enumerate(
            _ordered_candidates(manifest, family, salt=salt)
        ):
            if family_count >= quotas[family]:
                break
            reason = None
            receipt = None
            fingerprint = candidate["fingerprint"]
            seed = candidate["seed"]
            if fingerprint in excluded_fingerprints or seed in excluded_seeds:
                reason = "excluded_scene"
            else:
                if split_name == "development_validation":
                    receipt = screen_scene(
                        candidate,
                        split="conflict_validation",
                        scene_index=len(accepted),
                        train_count=TOTAL_DEVELOPMENT_FIT_SCENES,
                        actor=actor,
                    )
                else:
                    receipt = screen_scene(
                        candidate,
                        split="train",
                        scene_index=BASE_DEVELOPMENT_FIT_SCENES + len(accepted),
                        actor=actor,
                    )
                if receipt.get("passed") is not True:
                    reason = "public_workload_failed"
            admitted = reason is None
            trace.append({
                "split": split_name,
                "family_id": family,
                "candidate_rank": rank,
                "fingerprint": fingerprint,
                "seed": seed,
                "accepted": admitted,
                "rejection_reason": reason,
                "workload_receipt": receipt,
            })
            if admitted:
                frozen = deepcopy(candidate)
                frozen["id"] = (
                    f"diagnostic_v8_{split_name}_{len(accepted):04d}"
                )
                frozen["split"] = split_name
                frozen["workload_screen"] = receipt
                accepted.append(frozen)
                excluded_fingerprints.add(fingerprint)
                excluded_seeds.add(seed)
                family_count += 1
        if family_count != quotas[family]:
            raise RuntimeError(
                "Development-expansion family quota unavailable: " + family
            )
    return accepted, trace


def _select(
    *, manifest: Mapping[str, Any], actor: Any,
    excluded_fingerprints: set[str], excluded_seeds: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    # Freeze validation first, before any fit-supplement choice is observed.
    validation, validation_trace = _select_split(
        manifest=manifest,
        actor=actor,
        split_name="development_validation",
        quotas=VALIDATION_FAMILY_QUOTAS,
        salt=VALIDATION_ORDER_SALT,
        excluded_fingerprints=excluded_fingerprints,
        excluded_seeds=excluded_seeds,
    )
    fit, fit_trace = _select_split(
        manifest=manifest,
        actor=actor,
        split_name="fit_supplement",
        quotas=FIT_FAMILY_QUOTAS,
        salt=FIT_ORDER_SALT,
        excluded_fingerprints=excluded_fingerprints,
        excluded_seeds=excluded_seeds,
    )
    return fit, validation, {
        "development_validation": validation_trace,
        "fit_supplement": fit_trace,
    }


def _split_statistics(
    scenes: Sequence[Mapping[str, Any]], trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "accepted": len(scenes),
        "evaluated": len(trace),
        "families": dict(sorted(Counter(
            str(row["family_id"]) for row in scenes
        ).items())),
        "rejected": dict(sorted(Counter(
            str(row["rejection_reason"]) for row in trace
            if row.get("rejection_reason") is not None
        ).items())),
        "unique_fingerprints": len({row["fingerprint"] for row in scenes}),
        "unique_seeds": len({row["seed"] for row in scenes}),
    }


def _write_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Development-expansion output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_output(output: Path, registry: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    parent = output.parent
    if (not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent
            or output.exists() or output.is_symlink()):
        raise ValueError("Development-expansion destination is unsafe or already exists")
    lock = parent / ("." + output.name + ".lock")
    lock_fd = None
    temporary: Path | None = None
    try:
        lock_fd = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        temporary = Path(tempfile.mkdtemp(
            prefix="." + output.name + ".tmp-", dir=parent
        )).absolute()
        os.chmod(temporary, 0o700)
        _write_exclusive(temporary / "development_expansion.json", registry)
        _write_exclusive(temporary / "report.json", report)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise ValueError("Development-expansion destination appeared during build")
        os.rename(temporary, output)
        temporary = None
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)


def _input_identity(
    *, actor_path: Path, manifest_path: Path, designation_path: Path,
    selected_path: Path, previous_development_path: Path,
    retired_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[str], set[int], dict[str, Any]]:
    actor_sha256 = file_hash(actor_path)
    if actor_sha256 != designation_api.EXPECTED_ACTOR_SHA256:
        raise ValueError("Development expansion requires the frozen diagnostic Actor")
    if file_hash(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Development expansion requires the current frozen manifest")
    if file_hash(designation_path) != EXPECTED_DESIGNATION_SHA256:
        raise ValueError("Development expansion requires the current Actor designation")

    manifest = _read(manifest_path, "diagnostic conflict manifest")
    validate_diagnostic_manifest(manifest, replay=False)
    if (manifest.get("version") != MANIFEST_VERSION
            or not _content_valid(manifest)
            or manifest.get("frozen_actor", {}).get("sha256") != actor_sha256
            or manifest.get("frozen_actor", {}).get("actor_parameters_sha256")
                != designation_api.EXPECTED_ACTOR_PARAMETERS_SHA256):
        raise ValueError("Diagnostic conflict manifest binding differs")

    designation = _read(designation_path, "diagnostic Actor designation")
    _validate_designation(designation, actor_sha256=actor_sha256)
    selected = _read(selected_path, "formal X/Y selection")
    fingerprints, seeds = _validate_selected(
        selected, actor_sha256=actor_sha256,
        manifest_path=manifest_path, manifest=manifest,
    )

    # Every manifest split is registered and unavailable to this expansion.
    registered_rows = [row for rows in manifest["splits"].values() for row in rows]
    registered_fp, registered_seeds = _scene_identities(
        registered_rows,
        expected_count=sum(len(rows) for rows in manifest["splits"].values()),
        label="registered manifest splits",
    )
    fingerprints.update(registered_fp)
    seeds.update(registered_seeds)

    previous_development = _read(
        previous_development_path, "previous development supplement"
    )
    previous_fp, previous_seeds = _validate_previous_development(
        previous_development,
        actor_sha256=actor_sha256,
        manifest_path=manifest_path,
        selected_path=selected_path,
    )
    fingerprints.update(previous_fp)
    seeds.update(previous_seeds)

    retired_values: dict[str, dict[str, Any]] = {}
    expected_versions = {
        "warehouse-r41-diagnostic-fresh-final-holdout.v1",
        "warehouse-r41-diagnostic-fresh-final-holdout.v2",
    }
    for path in retired_paths:
        value = _read(path, "retired fresh-final registry")
        version = value.get("version")
        if version not in expected_versions or version in retired_values:
            raise ValueError("Exactly one retired v1 and v2 holdout registry required")
        retired_fp, retired_seeds = _validate_retired_holdout(
            value, expected_version=str(version)
        )
        fingerprints.update(retired_fp)
        seeds.update(retired_seeds)
        retired_values[str(version)] = value
    if set(retired_values) != expected_versions:
        raise ValueError("Exactly one retired v1 and v2 holdout registry required")

    exclusion_summary = {
        "registered_manifest_scenes": len(registered_rows),
        "formal_xy_scenes": 6,
        "previous_development_scenes": 64,
        "retired_fresh_final_scenes": 128,
        "unique_excluded_fingerprints": len(fingerprints),
        "unique_excluded_seeds": len(seeds),
        "excluded_fingerprints_sha256": digest(sorted(fingerprints)),
        "excluded_seeds_sha256": digest(sorted(seeds)),
    }
    return manifest, designation, selected, fingerprints, seeds, exclusion_summary


def build(
    *, actor_path: str | Path, manifest_path: str | Path,
    designation_path: str | Path, selected_scenes_path: str | Path,
    previous_development_path: str | Path,
    retired_holdout_paths: Sequence[str | Path], output: str | Path,
) -> dict[str, Any]:
    actor_path = _regular(actor_path, "frozen diagnostic Actor")
    manifest_path = _regular(manifest_path, "diagnostic conflict manifest")
    designation_path = _regular(designation_path, "diagnostic Actor designation")
    selected_path = _regular(selected_scenes_path, "formal X/Y selection")
    previous_path = _regular(
        previous_development_path, "previous development supplement"
    )
    retired_paths = [
        _regular(path, "retired fresh-final registry")
        for path in retired_holdout_paths
    ]
    if len(retired_paths) != 2:
        raise ValueError("Exactly two retired fresh-final registries required")

    (manifest, designation, selected, excluded_fingerprints, excluded_seeds,
     exclusion_summary) = _input_identity(
        actor_path=actor_path,
        manifest_path=manifest_path,
        designation_path=designation_path,
        selected_path=selected_path,
        previous_development_path=previous_path,
        retired_paths=retired_paths,
    )
    actor = load_frozen_actor(actor_path)
    fit, validation, trace = _select(
        manifest=manifest,
        actor=actor,
        excluded_fingerprints=set(excluded_fingerprints),
        excluded_seeds=set(excluded_seeds),
    )
    fit_stats = _split_statistics(fit, trace["fit_supplement"])
    validation_stats = _split_statistics(
        validation, trace["development_validation"]
    )
    if (fit_stats["families"] != FIT_FAMILY_QUOTAS
            or validation_stats["families"] != VALIDATION_FAMILY_QUOTAS
            or {row["fingerprint"] for row in fit}
                & {row["fingerprint"] for row in validation}
            or {row["seed"] for row in fit} & {row["seed"] for row in validation}):
        raise RuntimeError("Development-expansion split invariant failed")

    sources = producer_sources()
    retired_binding = {
        _read(path, "retired fresh-final registry")["version"]: {
            "file_sha256": file_hash(path),
            "content_sha256": _read(
                path, "retired fresh-final registry"
            )["content_sha256"],
        }
        for path in retired_paths
    }
    bindings = {
        "actor_sha256": file_hash(actor_path),
        "actor_parameters_sha256": designation["bindings"]["actor_parameters_sha256"],
        "designation_file_sha256": file_hash(designation_path),
        "designation_semantic_sha256": digest(designation),
        "designation_protocol_file_sha256": designation["bindings"]["protocol_file_sha256"],
        "designation_protocol_content_sha256": designation["bindings"]["protocol_content_sha256"],
        "source_manifest_file_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "source_manifest_semantic_sha256": digest(manifest),
        "selected_scenes_file_sha256": file_hash(selected_path),
        "selected_scenes_semantic_sha256": digest(selected),
        "previous_development_file_sha256": file_hash(previous_path),
        "previous_development_content_sha256": _read(
            previous_path, "previous development supplement"
        )["content_sha256"],
        "retired_holdouts": dict(sorted(retired_binding.items())),
        "workload_contract_sha256": WORKLOAD_CONTRACT_SHA256,
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(sources),
    }
    statistics = {
        "fit_supplement": fit_stats,
        "development_validation": validation_stats,
        "total_accepted": len(fit) + len(validation),
        "cross_split_fingerprint_overlap": 0,
        "cross_split_seed_overlap": 0,
        "exclusions": exclusion_summary,
    }
    registry = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": bindings,
        "fit_supplement": fit,
        "development_validation": validation,
        "selection_trace": trace,
        "statistics": statistics,
        "program_access": False,
        "program_predictions_access": False,
        "participant_data_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    registry["content_sha256"] = digest(registry)
    report = {
        "version": VERSION,
        "status": STATUS,
        "bindings": bindings,
        "registry_content_sha256": registry["content_sha256"],
        "statistics": statistics,
        "producer_sources": sources,
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "formal_ready": False,
    }

    output_path = Path(output).expanduser().absolute()
    if not output_path.parent.exists():
        raise ValueError("Development-expansion output parent must already exist")
    # Compute the exact registry byte hash before the directory becomes visible.
    report["registry_file_sha256"] = digest_bytes(
        (canonical(registry) + "\n").encode("utf-8")
    )
    report["content_sha256"] = digest(report)
    _atomic_output(output_path, registry, report)
    if file_hash(output_path / "development_expansion.json") != report["registry_file_sha256"]:
        raise RuntimeError("Atomic development-expansion bytes differ")
    return deepcopy(report)


def digest_bytes(value: bytes) -> str:
    from hashlib import sha256
    return sha256(value).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--selected-scenes", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--retired-holdout", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(
        actor_path=args.actor,
        manifest_path=args.manifest,
        designation_path=args.designation,
        selected_scenes_path=args.selected_scenes,
        previous_development_path=args.previous_development,
        retired_holdout_paths=args.retired_holdout,
        output=args.output,
    )
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "BASE_DEVELOPMENT_FIT_SCENES",
    "FIT_SUPPLEMENT_SCENES", "TOTAL_DEVELOPMENT_FIT_SCENES",
    "VALIDATION_SCENES", "FIT_FAMILY_QUOTAS", "VALIDATION_FAMILY_QUOTAS",
    "contract", "producer_sources", "build", "main",
]
