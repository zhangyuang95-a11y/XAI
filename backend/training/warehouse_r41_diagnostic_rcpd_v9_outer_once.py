"""Irrevocably score one locked v9 program on one fresh development outer.

The label-blind ordered hash projection is authenticated first.  A permanent
``O_EXCL`` attempt anchor is then written *before* this module opens the raw
public observations, Actor probabilities, or action labels in the full outer
row archive.  The fresh outer is consequently consumed even if validation or
scoring later fails.  A different candidate cannot reuse the same projected
outer because the permanent attempt key depends only on the outer identity.

This module has no protected final/holdout input and never fits or modifies a
program.  It evaluates the already locked explicit public-tree program once.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v9 as projection_api
from backend.training import warehouse_r41_diagnostic_outer_collection_v9 as collection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_split as registry_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    R41DiagnosticPublicTreeProgramV9,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-rcpd-v9-outer-once.v1"
LOCK_SCHEMA = "warehouse_r41_diagnostic_rcpd_v9_candidate_lock_v1"
COLLECTION_SCHEMA = "warehouse_r41_diagnostic_rcpd_v9_outer_collection_v1"
LOCK_STATUS = "locked"
COLLECTION_STATUS = "collected_unscored"
STATUS_PASSED = "passed_fresh_development_outer_gates"
STATUS_FAILED = "failed_fresh_development_outer_gates"
STATUS_ABORTED = "failed_after_outer_attempt_claim"
ANCHOR_NAME = "attempt_anchor.json"
RESULT_NAME = "outer_result.json"
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024 * 1024
PREDICTION_BATCH_SIZE = 16_384
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_LOCK_BINDINGS = frozenset((
    "actor_sha256", "protocol_sha256", "runtime_manifest_sha256",
    "designation_sha256", "failed_outer_closeout_sha256",
    "fresh_outer_registry_sha256", "outer_hash_projection_sha256",
    "development_rows_sha256", "program_sha256",
    "selector_report_sha256", "source_closure_sha256",
))
_SAFE_ROW_FIELDS = frozenset((
    "observation_hashes", "scene_fingerprints", "episode_ids", "frames",
    "group_bits", "kinds", "anchor_ids", "branch_actions",
    "physical_hashes", "source_state_hashes", "trajectory_done",
    "split_validation",
))
_PRIVATE_ROW_FIELDS = frozenset((
    "observations", "probabilities", "action_indices", "weights",
    "submitted_equal",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "one irrevocable score of a locked program on fresh development outer",
        "attempt_key_scope": "outer identity, independent of candidate and private row payload",
        "attempt_anchor_before_private_outer_read": True,
        "preclaim_row_fields": sorted(_SAFE_ROW_FIELDS),
        "preclaim_forbidden_row_fields": sorted(_PRIVATE_ROW_FIELDS),
        "candidate_refit": False,
        "program_mutation": False,
        "runtime_action_override": False,
        "protected_final_access": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: child for key, child in value.items()
                                   if key != "content_sha256"}))


def _regular(value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a nonempty bounded canonical regular file")
    return path


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must contain one JSON object")
    return value


def _strict_json(
    value: str | Path, label: str, *, expected_sha256: str,
) -> tuple[Path, bytes, dict[str, Any]]:
    path = _regular(value, label)
    expected = _sha(expected_sha256, label + " SHA-256")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact " + label + " bytes required")
    parsed = _strict_json_bytes(raw, label)
    if file_hash(path) != expected:
        raise RuntimeError(label + " changed during read")
    return path, raw, parsed


def _authenticate_file(
    value: str | Path, label: str, *, expected_sha256: str,
    maximum: int = MAX_JSON_BYTES,
) -> Path:
    path = _regular(value, label, maximum=maximum)
    if file_hash(path) != _sha(expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _candidate_lock(
    path: str | Path, *, expected_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    lock_path, _, value = _strict_json(
        path, "v9 candidate lock", expected_sha256=expected_sha256)
    bindings = value.get("bindings")
    if (value.get("schema_version") != LOCK_SCHEMA
            or value.get("status") != LOCK_STATUS
            or value.get("formal_ready") is not False
            or not _content_valid(value)
            or not isinstance(bindings, Mapping)
            or not _LOCK_BINDINGS.issubset(bindings)
            or any(type(name) is not str or type(child) is not str
                   or _HEX.fullmatch(child) is None
                   for name, child in bindings.items())):
        raise ValueError("V9 candidate lock semantics differ")
    return lock_path, value, dict(bindings)


def _verify_lock_files(bindings: Mapping[str, str], **paths: str | Path) -> dict[str, Path]:
    expected_keys = {
        "actor": "actor_sha256", "protocol": "protocol_sha256",
        "runtime_manifest": "runtime_manifest_sha256",
        "designation": "designation_sha256",
        "failed_outer_closeout": "failed_outer_closeout_sha256",
        "fresh_outer_registry": "fresh_outer_registry_sha256",
        "outer_hash_projection": "outer_hash_projection_sha256",
        "development_rows": "development_rows_sha256",
        "program": "program_sha256", "selector_report": "selector_report_sha256",
    }
    if set(paths) != set(expected_keys):
        raise ValueError("Complete v9 candidate-lock file set required")
    result = {}
    for name, binding in expected_keys.items():
        maximum = MAX_NPZ_BYTES if name in {"actor", "development_rows"} else MAX_JSON_BYTES
        result[name] = _authenticate_file(
            paths[name], name.replace("_", " "), expected_sha256=bindings[binding],
            maximum=maximum)
    return result


def _validate_candidate_source_closure(
    lock: Mapping[str, Any], selector_report_path: Path,
    bindings: Mapping[str, str],
) -> None:
    selector = _strict_json_bytes(
        selector_report_path.read_bytes(), "locked v9 selector report")
    closure = None
    for candidate in (
        lock.get("source_closure"), selector.get("sources"),
        selector.get("producer_sources"),
    ):
        if isinstance(candidate, Mapping):
            closure = dict(candidate)
            break
    if closure is None:
        report_binding = selector.get("bindings", {}).get(
            "source_closure_sha256") if isinstance(
                selector.get("bindings"), Mapping) else None
        if report_binding != bindings["source_closure_sha256"]:
            raise ValueError("V9 candidate source closure is not authenticated")
        return
    if (not closure
            or any(type(name) is not str or type(value) is not str
                   or _HEX.fullmatch(value) is None
                   for name, value in closure.items())
            or digest(closure) != bindings["source_closure_sha256"]):
        raise ValueError("V9 candidate source closure binding differs")


def _registry(path: Path) -> dict[str, Any]:
    value = _strict_json_bytes(path.read_bytes(), "fresh v9 outer registry")
    scenes = value.get("development_outer")
    if (value.get("version") != registry_api.VERSION
            or value.get("status") != registry_api.STATUS
            or not _content_valid(value)
            or not isinstance(scenes, list)
            or len(scenes) != projection_api.SCENE_COUNT
            or value.get("program_access") is not False
            or value.get("program_predictions_access") is not False
            or value.get("action_labels_access") is not False
            or value.get("probabilities_access") is not False
            or value.get("final_audit_rows_access") is not False
            or value.get("formal_ready") is not False):
        raise ValueError("Fresh v9 outer registry semantics differ")
    fingerprints = [row.get("fingerprint") if isinstance(row, Mapping) else None
                    for row in scenes]
    if (len(set(fingerprints)) != projection_api.SCENE_COUNT
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in fingerprints)):
        raise ValueError("Fresh v9 outer registry identities differ")
    return value


def _program_payload(path: Path) -> dict[str, Any]:
    value = _strict_json_bytes(path.read_bytes(), "locked v9 program")
    # Instantiate the executable only after the irreversible attempt claim.
    # Pre-claim authentication is limited to exact bytes and strict JSON.
    if value.get("version") != "warehouse-r41-diagnostic-public-tree-program.v9":
        raise ValueError("Locked v9 program identity differs")
    return value


def _zip_member(archive: zipfile.ZipFile, name: str, *, label: str) -> np.ndarray:
    info = archive.getinfo(name + ".npy")
    if info.is_dir() or info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(label + " member is unsafe")
    try:
        return np.load(BytesIO(archive.read(info)), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(label + " member is not a safe NumPy array") from error


def _safe_row_projection(
    path: Path, *, expected_sha256: str, fields: frozenset[str], label: str,
) -> dict[str, np.ndarray]:
    expected = _sha(expected_sha256, label + " SHA-256")
    if file_hash(path) != expected:
        raise ValueError("Exact " + label + " bytes required")
    expected_members = {name + ".npy" for name in rows_api._FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError(label + " archive schema differs")
            result = {name: _zip_member(archive, name, label=label + " " + name)
                      for name in fields}
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(label + " cannot be safely projected") from error
    if file_hash(path) != expected:
        raise RuntimeError(label + " changed during projection")
    return result


def _decode(array: np.ndarray, label: str) -> list[str]:
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError(label + " must be a one-dimensional byte-string array")
    try:
        return list(map(str, np.char.decode(array, "ascii")))
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be ASCII") from error


def _row_identity_hashes(arrays: Mapping[str, np.ndarray]) -> list[str]:
    hashes = _decode(arrays["observation_hashes"], "outer observation hashes")
    count = len(hashes)
    text = {name: _decode(arrays[name], "outer " + name) for name in (
        "scene_fingerprints", "episode_ids", "kinds", "anchor_ids",
        "branch_actions", "physical_hashes", "source_state_hashes")}
    if any(len(values) != count for values in text.values()):
        raise ValueError("Outer ordered row identity length differs")
    for name in ("frames", "group_bits", "trajectory_done", "split_validation"):
        if arrays[name].shape != (count,):
            raise ValueError("Outer ordered row identity shape differs")
    return [digest({
        "scene_fingerprint": text["scene_fingerprints"][index],
        "episode_id": text["episode_ids"][index],
        "frame": int(arrays["frames"][index]),
        "group_bits": int(arrays["group_bits"][index]),
        "kind": text["kinds"][index],
        "anchor_id": text["anchor_ids"][index],
        "branch_action": text["branch_actions"][index],
        "physical_sha256": text["physical_hashes"][index],
        "source_state_sha256": text["source_state_hashes"][index],
        "trajectory_done": bool(arrays["trajectory_done"][index]),
        "split_validation": bool(arrays["split_validation"][index]),
    }) for index in range(count)]


def _preflight_outer_rows(
    *, rows_path: Path, rows_sha256: str, projection: Mapping[str, Any],
    development_rows_path: Path, development_rows_sha256: str,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    outer = _safe_row_projection(
        rows_path, expected_sha256=rows_sha256, fields=_SAFE_ROW_FIELDS,
        label="full unscored outer rows")
    ordered = _decode(outer["observation_hashes"], "outer observation hashes")
    expected = projection["projection"]
    row_ids = _row_identity_hashes(outer)
    scenes = _decode(outer["scene_fingerprints"], "outer scene fingerprints")
    registry_scenes = {str(row["fingerprint"])
                       for row in registry["development_outer"]}
    if (ordered != expected["ordered_observation_hashes"]
            or row_ids != expected["ordered_row_identity_hashes"]
            or digest(ordered) != expected["ordered_observation_hashes_sha256"]
            or digest(row_ids) != expected["ordered_row_identity_hashes_sha256"]
            or len(ordered) != expected["row_count"]
            or not np.all(outer["split_validation"])
            or set(scenes) != registry_scenes):
        raise ValueError("Full outer ordered hash projection differs")
    development = _safe_row_projection(
        development_rows_path, expected_sha256=development_rows_sha256,
        fields=frozenset(("observation_hashes",)), label="locked development rows")
    development_hashes = set(_decode(
        development["observation_hashes"], "development observation hashes"))
    overlap = set(ordered) & development_hashes
    if overlap:
        raise ValueError(
            "Fresh outer observation hashes overlap locked development rows: "
            + str(len(overlap)))
    return {
        "outer_row_count": len(ordered),
        "outer_unique_observation_count": len(set(ordered)),
        "outer_scene_count": len(set(scenes)),
        "development_unique_observation_count": len(development_hashes),
        "development_outer_observation_overlap": 0,
        "ordered_observation_hashes_sha256": digest(ordered),
        "ordered_row_identity_hashes_sha256": digest(row_ids),
        "private_row_members_opened": False,
    }


def _collection_report(
    path: str | Path, *, expected_sha256: str, lock_sha256: str,
    bindings: Mapping[str, str], rows_path: str | Path, rows_sha256: str,
    projection: Mapping[str, Any], projection_copy_path: str | Path,
    expected_projection_copy_sha256: str,
    projection_receipt_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    value = collection_api.authenticate_saved_collection(
        report_path=path, rows_path=rows_path,
        projection_copy_path=projection_copy_path,
        expected_report_sha256=expected_sha256,
        expected_rows_sha256=rows_sha256,
        expected_projection_copy_sha256=expected_projection_copy_sha256,
    )
    report_path = _authenticate_file(
        path, "unscored outer collection report", expected_sha256=expected_sha256)
    report_bindings = value.get("bindings")
    collection = value.get("collection")
    boundary = value.get("information_boundary")
    required = {
        "candidate_lock_sha256": lock_sha256,
        "actor_sha256": bindings["actor_sha256"],
        "protocol_sha256": bindings["protocol_sha256"],
        "runtime_manifest_sha256": bindings["runtime_manifest_sha256"],
        "designation_sha256": bindings["designation_sha256"],
        "failed_outer_closeout_sha256": bindings[
            "failed_outer_closeout_sha256"],
        "fresh_outer_registry_sha256": bindings["fresh_outer_registry_sha256"],
        "outer_hash_projection_sha256": bindings["outer_hash_projection_sha256"],
        "outer_hash_projection_receipt_sha256": projection_receipt_sha256,
        "development_rows_sha256": bindings["development_rows_sha256"],
        "program_sha256": bindings["program_sha256"],
        "selector_report_sha256": bindings["selector_report_sha256"],
        "rows_sha256": rows_sha256,
        "projection_copy_sha256": expected_projection_copy_sha256,
        "ordered_replay_sha256": projection["projection"]["ordered_replay_sha256"],
    }
    if (value.get("version") != collection_api.VERSION
            or value.get("schema_version") != COLLECTION_SCHEMA
            or value.get("status") != COLLECTION_STATUS
            or value.get("formal_ready") is not False
            or not _content_valid(value)
            or not isinstance(report_bindings, Mapping)
            or any(report_bindings.get(name) != expected
                   for name, expected in required.items())
            or not isinstance(collection, Mapping)
            or collection.get("row_count") != projection["projection"]["row_count"]
            or collection.get("scene_count") != projection_api.SCENE_COUNT
            or collection.get("scored") is not False
            or collection.get("all_submitted_actions_equal_policy_actions") is not True
            or collection.get("all_actor_probabilities_and_actions_exact") is not True
            or not isinstance(boundary, Mapping)
            or boundary.get("candidate_lock_authenticated_before_outer_replay") is not True
            or boundary.get("labels_and_probabilities_collected_only_after_candidate_lock")
                is not True
            or boundary.get("outer_metrics_or_candidate_score_computed") is not False
            or boundary.get("program_loaded_or_executed") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("runtime_action_override") is not False):
        raise ValueError("Unscored outer collection report semantics differ")
    return report_path, value


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _predict(program: R41DiagnosticPublicTreeProgramV9,
             observations: np.ndarray) -> np.ndarray:
    chunks = [program.predict_proba_batch(
        observations[start:start + PREDICTION_BATCH_SIZE])
        for start in range(0, len(observations), PREDICTION_BATCH_SIZE)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty(
        (0, len(metrics_api.ACTIONS)), dtype=np.float64)


def _load_private_rows(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray]:
    expected = _sha(expected_sha256, "full outer rows SHA-256")
    if file_hash(path) != expected:
        raise ValueError("Full outer rows changed before private read")
    arrays = collection_api.load_authenticated_rows(
        path, expected_rows_sha256=expected)
    if file_hash(path) != expected:
        raise RuntimeError("Full outer rows changed during private read")
    return arrays


def _attempt_identity(
    *, bindings: Mapping[str, str], projection: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    values = {
        "scheme": VERSION + ".outer-identity.v1",
        "fresh_outer_registry_sha256": bindings["fresh_outer_registry_sha256"],
        "fresh_outer_registry_content_sha256": registry["content_sha256"],
        "outer_hash_projection_sha256": bindings["outer_hash_projection_sha256"],
        "outer_hash_projection_content_sha256": projection["content_sha256"],
        "ordered_replay_sha256": projection["projection"]["ordered_replay_sha256"],
    }
    return digest(values), values


def _failure_result(
    *, attempt_key: str, anchor: Mapping[str, Any], error: BaseException,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS_ABORTED,
        "attempt_key": attempt_key,
        "attempt_anchor_content_sha256": anchor["content_sha256"],
        "failure": {
            "exception_type": type(error).__name__,
            "message": str(error)[:1000],
            "outer_consumed": True,
            "retry_permitted": False,
        },
        "producer_sources_sha256": digest(sources),
        "protected_final_access": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def build(
    *, candidate_lock_path: str | Path, expected_candidate_lock_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    runtime_manifest_path: str | Path, designation_path: str | Path,
    failed_outer_closeout_path: str | Path,
    fresh_outer_registry_path: str | Path,
    outer_hash_projection_path: str | Path,
    outer_hash_projection_receipt_path: str | Path,
    expected_outer_hash_projection_receipt_sha256: str,
    outer_hash_projection_copy_path: str | Path,
    expected_outer_hash_projection_copy_sha256: str,
    development_rows_path: str | Path, program_path: str | Path,
    selector_report_path: str | Path,
    outer_collection_report_path: str | Path,
    expected_outer_collection_report_sha256: str,
    outer_rows_path: str | Path, expected_outer_rows_sha256: str,
    permanent_registry: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Claim and score one outer.  No private outer field is read pre-claim."""
    sources = producer_sources()
    if any("fresh_final" in name or "final_once" in name for name in sources):
        raise RuntimeError("Protected final/holdout source entered outer scorer closure")
    lock_path, lock, bindings = _candidate_lock(
        candidate_lock_path, expected_sha256=expected_candidate_lock_sha256)
    paths = _verify_lock_files(
        bindings, actor=actor_path, protocol=protocol_path,
        runtime_manifest=runtime_manifest_path, designation=designation_path,
        failed_outer_closeout=failed_outer_closeout_path,
        fresh_outer_registry=fresh_outer_registry_path,
        outer_hash_projection=outer_hash_projection_path,
        development_rows=development_rows_path, program=program_path,
        selector_report=selector_report_path)
    _validate_candidate_source_closure(lock, paths["selector_report"], bindings)
    registry = _registry(paths["fresh_outer_registry"])
    program_payload = _program_payload(paths["program"])
    projection, projection_receipt = projection_api.read_saved_projection(
        projection_path=paths["outer_hash_projection"],
        receipt_path=outer_hash_projection_receipt_path,
        expected_projection_sha256=bindings["outer_hash_projection_sha256"],
        expected_receipt_sha256=expected_outer_hash_projection_receipt_sha256)
    if (projection["identity"]["registry_file_sha256"]
            != bindings["fresh_outer_registry_sha256"]
            or projection["identity"]["registry_content_sha256"]
                != registry["content_sha256"]
            or projection_receipt["bindings"]["actor_sha256"]
                != bindings["actor_sha256"]):
        raise ValueError("Candidate lock, registry, and projection identity differ")
    projection_copy_path = _authenticate_file(
        outer_hash_projection_copy_path, "copied outer hash projection",
        expected_sha256=expected_outer_hash_projection_copy_sha256)
    if (file_hash(projection_copy_path) != bindings["outer_hash_projection_sha256"]
            or projection_copy_path.read_bytes()
                != paths["outer_hash_projection"].read_bytes()):
        raise ValueError("Collected outer projection copy differs from locked projection")
    rows_path = _authenticate_file(
        outer_rows_path, "full unscored outer rows",
        expected_sha256=expected_outer_rows_sha256, maximum=MAX_NPZ_BYTES)
    report_path, collection_report = _collection_report(
        outer_collection_report_path,
        expected_sha256=expected_outer_collection_report_sha256,
        lock_sha256=file_hash(lock_path), bindings=bindings,
        rows_path=rows_path, rows_sha256=file_hash(rows_path), projection=projection,
        projection_copy_path=outer_hash_projection_copy_path,
        expected_projection_copy_sha256=(
            expected_outer_hash_projection_copy_sha256),
        projection_receipt_sha256=file_hash(
            Path(outer_hash_projection_receipt_path).expanduser().absolute()))
    preflight = _preflight_outer_rows(
        rows_path=rows_path, rows_sha256=file_hash(rows_path),
        projection=projection, development_rows_path=paths["development_rows"],
        development_rows_sha256=bindings["development_rows_sha256"],
        registry=registry)
    attempt_key, attempt_inputs = _attempt_identity(
        bindings=bindings, projection=projection, registry=registry)
    permanent = _directory(permanent_registry, "permanent outer-attempt registry")
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "outer result output parent")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Outer result output already exists")
    anchor: dict[str, Any] = {
        "version": VERSION + ".attempt-anchor.v1",
        "status": "fresh_outer_attempt_irrevocably_claimed",
        "attempt_key": attempt_key,
        "attempt_key_inputs": attempt_inputs,
        "bindings": {
            "candidate_lock_sha256": file_hash(lock_path),
            "candidate_lock_content_sha256": lock["content_sha256"],
            "program_sha256": file_hash(paths["program"]),
            "program_content_sha256": digest(program_payload),
            "outer_collection_report_sha256": file_hash(report_path),
            "outer_collection_report_content_sha256": collection_report[
                "content_sha256"],
            "outer_rows_sha256": file_hash(rows_path),
            "outer_hash_projection_receipt_sha256": file_hash(
                Path(outer_hash_projection_receipt_path).expanduser().absolute()),
            "source_closure_sha256": digest(sources),
        },
        "preflight": preflight,
        "private_outer_fields_read_before_claim": False,
        "retry_permitted": False,
        "protected_final_access": False,
        "formal_ready": False,
    }
    anchor["content_sha256"] = digest(anchor)
    campaign_directory = permanent / attempt_key
    try:
        os.mkdir(campaign_directory, 0o700)
    except FileExistsError:
        raise FileExistsError(
            "This fresh outer has already been claimed and cannot be scored again") from None
    if campaign_directory.is_symlink() or campaign_directory.resolve() != campaign_directory:
        raise RuntimeError("Permanent outer-attempt directory is unsafe")
    _write_exclusive(campaign_directory / ANCHOR_NAME, _json_bytes(anchor))
    _fsync_directory(campaign_directory)
    _fsync_directory(permanent)

    result: dict[str, Any] | None = None
    try:
        # This is intentionally the first operation that opens observations,
        # Actor probabilities, action labels, weights, or submitted_equal.
        arrays = _load_private_rows(rows_path, expected_sha256=file_hash(rows_path))
        program = R41DiagnosticPublicTreeProgramV9.from_dict(program_payload)
        if tuple(program.action_names) != tuple(metrics_api.ACTIONS):
            raise ValueError("Locked v9 program action registry differs")
        actor = NumPyNativeActor(paths["actor"])
        rows_api._validate_arrays(
            arrays, actor=actor, train_scenes=[],
            validation_scenes=registry["development_outer"])
        if (not np.all(arrays["split_validation"])
                or _decode(arrays["observation_hashes"], "outer observation hashes")
                    != projection["projection"]["ordered_observation_hashes"]
                or _row_identity_hashes(arrays)
                    != projection["projection"]["ordered_row_identity_hashes"]):
            raise ValueError("Private outer rows differ from claimed ordered projection")
        probabilities = _predict(program, arrays["observations"])
        mask = np.ones(len(arrays["observations"]), dtype=np.bool_)
        pairs = rows_api._effective_pairs(arrays, mask)
        pair_bits = metrics_api._pair_group_bits(arrays, pairs)
        metrics = metrics_api._metrics_from_probabilities(
            probabilities, arrays, mask, pairs=pairs,
            pair_group_bits=pair_bits)
        gate = metrics_api._gate(metrics)
        result = {
            "version": VERSION,
            "status": STATUS_PASSED if gate["passed"] else STATUS_FAILED,
            "attempt_key": attempt_key,
            "attempt_anchor_content_sha256": anchor["content_sha256"],
            "candidate_lock_sha256": file_hash(lock_path),
            "program_sha256": file_hash(paths["program"]),
            "program_content_sha256": digest(program_payload),
            "outer_rows_sha256": file_hash(rows_path),
            "outer_collection_report_sha256": file_hash(report_path),
            "metrics": metrics,
            "gate": gate,
            "row_accounting": {
                "rows": len(arrays["observations"]),
                "scenes": len(set(_decode(
                    arrays["scene_fingerprints"], "outer scene fingerprints"))),
                "effective_intervention_pairs": len(pairs),
                "all_submitted_actions_equal_policy_actions": bool(
                    np.all(arrays["submitted_equal"])),
                "runtime_action_overrides": 0,
            },
            "execution": {
                "candidate_refit": False,
                "program_mutated": False,
                "private_outer_fields_read_after_permanent_claim": True,
                "outer_consumed": True,
                "retry_permitted": False,
                "protected_final_access": False,
            },
            "producer_sources": sources,
            "producer_sources_sha256": digest(sources),
            "explanation_eligible": bool(gate["passed"]),
            "formal_ready": False,
        }
        result["content_sha256"] = digest(result)
        if (producer_sources() != sources
                or file_hash(lock_path) != expected_candidate_lock_sha256
                or file_hash(paths["program"]) != bindings["program_sha256"]
                or file_hash(rows_path) != expected_outer_rows_sha256
                or file_hash(report_path)
                    != expected_outer_collection_report_sha256):
            raise RuntimeError("Outer scorer inputs or source closure changed during scoring")
        _write_exclusive(campaign_directory / RESULT_NAME, _json_bytes(result))
        _fsync_directory(campaign_directory)
    except BaseException as error:
        failure = _failure_result(
            attempt_key=attempt_key, anchor=anchor, error=error, sources=sources)
        try:
            _write_exclusive(campaign_directory / RESULT_NAME, _json_bytes(failure))
            _fsync_directory(campaign_directory)
        finally:
            raise

    if producer_sources() != sources:
        raise RuntimeError("Outer scorer source closure changed after scoring")
    if result is None:
        raise RuntimeError("Outer scorer did not produce a result")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / ANCHOR_NAME, _json_bytes(anchor))
        _write_exclusive(temporary / RESULT_NAME, _json_bytes(result))
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        temporary = None
        _fsync_directory(parent)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(result)


def read_saved_result(
    path: str | Path, *, expected_result_sha256: str,
    permanent_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate a published or permanent result against its claim anchor."""
    result_path, raw, value = _strict_json(
        path, "v9 one-shot outer result",
        expected_sha256=expected_result_sha256)
    status = value.get("status")
    attempt_key = value.get("attempt_key")
    if (value.get("version") != VERSION
            or status not in {STATUS_PASSED, STATUS_FAILED, STATUS_ABORTED}
            or type(attempt_key) is not str or _HEX.fullmatch(attempt_key) is None
            or not _content_valid(value)
            or value.get("formal_ready") is not False
            or value.get("producer_sources_sha256") != digest(producer_sources())):
        raise ValueError("Saved one-shot outer result semantics differ")
    permanent = _directory(permanent_registry, "permanent outer-attempt registry")
    campaign = _directory(permanent / attempt_key, "permanent outer-attempt directory")
    anchor_path = _regular(campaign / ANCHOR_NAME, "permanent outer-attempt anchor")
    permanent_result = _regular(
        campaign / RESULT_NAME, "permanent one-shot outer result")
    anchor = _strict_json_bytes(
        anchor_path.read_bytes(), "permanent outer-attempt anchor")
    if (anchor.get("version") != VERSION + ".attempt-anchor.v1"
            or anchor.get("status") != "fresh_outer_attempt_irrevocably_claimed"
            or anchor.get("attempt_key") != attempt_key
            or not _content_valid(anchor)
            or anchor.get("retry_permitted") is not False
            or anchor.get("protected_final_access") is not False
            or anchor.get("formal_ready") is not False
            or value.get("attempt_anchor_content_sha256")
                != anchor["content_sha256"]
            or permanent_result.read_bytes() != raw
            or file_hash(permanent_result) != expected_result_sha256
            or result_path.read_bytes() != raw):
        raise ValueError("Permanent one-shot outer result or anchor differs")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", required=True)
    parser.add_argument("--expected-candidate-lock-sha256", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--failed-outer-closeout", required=True)
    parser.add_argument("--fresh-outer-registry", required=True)
    parser.add_argument("--outer-hash-projection", required=True)
    parser.add_argument("--outer-hash-projection-receipt", required=True)
    parser.add_argument("--expected-outer-hash-projection-receipt-sha256", required=True)
    parser.add_argument("--outer-hash-projection-copy", required=True)
    parser.add_argument("--expected-outer-hash-projection-copy-sha256", required=True)
    parser.add_argument("--development-rows", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--selector-report", required=True)
    parser.add_argument("--outer-collection-report", required=True)
    parser.add_argument("--expected-outer-collection-report-sha256", required=True)
    parser.add_argument("--outer-rows", required=True)
    parser.add_argument("--expected-outer-rows-sha256", required=True)
    parser.add_argument("--permanent-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(
        candidate_lock_path=args.candidate_lock,
        expected_candidate_lock_sha256=args.expected_candidate_lock_sha256,
        actor_path=args.actor, protocol_path=args.protocol,
        runtime_manifest_path=args.runtime_manifest,
        designation_path=args.designation,
        failed_outer_closeout_path=args.failed_outer_closeout,
        fresh_outer_registry_path=args.fresh_outer_registry,
        outer_hash_projection_path=args.outer_hash_projection,
        outer_hash_projection_receipt_path=args.outer_hash_projection_receipt,
        expected_outer_hash_projection_receipt_sha256=(
            args.expected_outer_hash_projection_receipt_sha256),
        outer_hash_projection_copy_path=args.outer_hash_projection_copy,
        expected_outer_hash_projection_copy_sha256=(
            args.expected_outer_hash_projection_copy_sha256),
        development_rows_path=args.development_rows,
        program_path=args.program, selector_report_path=args.selector_report,
        outer_collection_report_path=args.outer_collection_report,
        expected_outer_collection_report_sha256=(
            args.expected_outer_collection_report_sha256),
        outer_rows_path=args.outer_rows,
        expected_outer_rows_sha256=args.expected_outer_rows_sha256,
        permanent_registry=args.permanent_registry, output=args.output,
    )
    print(canonical({
        "status": result["status"], "attempt_key": result["attempt_key"],
        "gate_passed": result["gate"]["passed"],
        "output": str(Path(args.output).expanduser().absolute()),
    }))
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "LOCK_SCHEMA", "COLLECTION_SCHEMA", "STATUS_PASSED",
    "STATUS_FAILED", "STATUS_ABORTED", "ANCHOR_NAME", "RESULT_NAME",
    "contract", "producer_sources", "build", "read_saved_result", "main",
    "_safe_row_projection", "_preflight_outer_rows",
]
