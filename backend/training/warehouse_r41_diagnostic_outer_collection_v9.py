"""Collect the full fresh v9 outer rows only after candidate lock.

The companion hash-projection producer commits to the exact replay order while
withholding observations, actions, and probabilities.  This module first
authenticates an immutable candidate lock and every artifact named by that
lock.  Only then does it replay the same schedule, require row-for-row hash
parity, and publish the unscored strict row archive for the one-shot scorer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v9 as projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_v7
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import ImmutableInputSnapshot


VERSION = "warehouse-r41-diagnostic-outer-collection.v9"
SCHEMA_VERSION = "warehouse_r41_diagnostic_rcpd_v9_outer_collection_v1"
STATUS = "collected_unscored"
CANDIDATE_LOCK_VERSION = "warehouse_r41_diagnostic_rcpd_v9_candidate_lock_v1"
REPORT_NAME = "collection_report.json"
ROWS_NAME = "rows.npz"
PROJECTION_COPY_NAME = projection_api.PROJECTION_NAME
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_NPZ_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_LOCK_BINDINGS = frozenset((
    "actor_sha256", "protocol_sha256", "runtime_manifest_sha256",
    "designation_sha256", "failed_outer_closeout_sha256",
    "fresh_outer_registry_sha256", "outer_hash_projection_sha256",
    "development_rows_sha256", "program_sha256", "selector_report_sha256",
    "source_closure_sha256",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "population": "fresh v9 identity-frozen development outer only",
        "scene_count": projection_api.SCENE_COUNT,
        "scene_offset": projection_api.SCENE_OFFSET,
        "partners": list(rows_v7.PARTNERS),
        "candidate_lock_required_before_actor_output_collection": True,
        "ordered_projection_required_row_for_row": True,
        "row_schema": sorted(rows_v7._FIELDS),
        "score_computed": False,
        "program_inference_used": False,
        "protected_final_access": False,
        "runtime_action_override": False,
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


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
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
        raise ValueError(label + " must be one JSON object")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), label)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_static_rows(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != rows_v7._FIELDS:
        raise ValueError("V9 outer row array schema differs")
    count = len(arrays["observations"])
    expected_dtypes = {
        "action_indices": np.dtype(np.uint8),
        "weights": np.dtype(np.float32),
        "observation_hashes": np.dtype("S64"),
        "scene_fingerprints": np.dtype("S64"),
        "episode_ids": np.dtype("S180"),
        "frames": np.dtype(np.int16),
        "group_bits": np.dtype(np.uint8),
        "kinds": np.dtype("S16"),
        "anchor_ids": np.dtype("S240"),
        "branch_actions": np.dtype("S8"),
        "physical_hashes": np.dtype("S64"),
        "source_state_hashes": np.dtype("S64"),
        "submitted_equal": np.dtype(np.bool_),
        "trajectory_done": np.dtype(np.bool_),
        "split_validation": np.dtype(np.bool_),
    }
    if (count <= 0
            or arrays["observations"].shape != (count, 197)
            or arrays["observations"].dtype != np.dtype(np.float32)
            or arrays["probabilities"].shape != (count, 5)
            or arrays["probabilities"].dtype != np.dtype(np.float32)
            or any(arrays[name].shape != (count,) or arrays[name].dtype != dtype
                   for name, dtype in expected_dtypes.items())):
        raise ValueError("V9 outer row shapes or dtypes differ")
    observations = arrays["observations"]
    probabilities = arrays["probabilities"]
    labels = arrays["action_indices"]
    if (not np.isfinite(observations).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(1), 1.0, rtol=0.0, atol=2e-6)
            or np.any(labels >= 5)
            or not np.array_equal(labels, probabilities.argmax(1).astype(np.uint8))
            or not np.isfinite(arrays["weights"]).all()
            or np.any(arrays["weights"] <= 0.0)
            or not np.all(arrays["submitted_equal"])
            or not np.all(arrays["split_validation"])
            or np.any(arrays["frames"] < 0)
            or np.any(arrays["group_bits"] >= 8)):
        raise ValueError("V9 outer numeric or action-authority rows differ")
    expected_observation_hashes = np.asarray(
        [rows_v7.legacy._obs_hash(row) for row in observations], dtype="S64")
    if not np.array_equal(
            expected_observation_hashes, arrays["observation_hashes"]):
        raise ValueError("V9 outer public-observation hashes differ")
    for name in ("observation_hashes", "scene_fingerprints",
                 "source_state_hashes"):
        try:
            values = np.char.decode(arrays[name], "ascii").astype(str)
        except UnicodeDecodeError as error:
            raise ValueError("V9 outer row text is not ASCII") from error
        if any(_HEX.fullmatch(str(value)) is None for value in values):
            raise ValueError("V9 outer row SHA-256 field differs")


def load_authenticated_rows(
    rows_path: str | Path, *, expected_rows_sha256: str,
) -> dict[str, np.ndarray]:
    """Load one exact strict row archive without pickle or extra members."""
    path = _regular(rows_path, "v9 outer rows", maximum=MAX_NPZ_BYTES)
    if file_hash(path) != _sha(expected_rows_sha256, "v9 outer rows"):
        raise ValueError("Exact v9 outer row bytes required")
    expected_members = {name + ".npy" for name in rows_v7._FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (len(names) != len(set(names)) or set(names) != expected_members
                    or len(names) != len(expected_members)
                    or any(item.is_dir() or item.file_size <= 0
                           or item.compress_size < 0 or item.flag_bits & 0x1
                           for item in infos)
                    or sum(item.file_size for item in infos) > MAX_NPZ_EXPANDED_BYTES):
                raise ValueError("V9 outer row archive contents differ")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != rows_v7._FIELDS:
                raise ValueError("V9 outer row array schema differs")
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("V9 outer rows are not a safe strict NPZ") from error
    if file_hash(path) != expected_rows_sha256:
        raise RuntimeError("V9 outer rows changed during strict load")
    _validate_static_rows(arrays)
    return arrays


def _validate_candidate_lock_shape(value: Mapping[str, Any]) -> dict[str, str]:
    bindings = value.get("bindings")
    if (value.get("schema_version") != CANDIDATE_LOCK_VERSION
            or value.get("status") != "locked"
            or value.get("formal_ready") is not False
            or not _content_valid(value)
            or not isinstance(bindings, Mapping)
            or not _REQUIRED_LOCK_BINDINGS.issubset(bindings)
            or any(_HEX.fullmatch(str(bindings.get(name, ""))) is None
                   for name in _REQUIRED_LOCK_BINDINGS)):
        raise ValueError("Exact locked v9 candidate contract required")
    return {name: str(bindings[name]) for name in _REQUIRED_LOCK_BINDINGS}


def _selector_source_closure(
    lock: Mapping[str, Any], selector_report: Mapping[str, Any],
) -> Mapping[str, str] | None:
    for candidate in (
        lock.get("source_closure"), selector_report.get("sources"),
        selector_report.get("producer_sources"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _validate_candidate_bindings(
    *, lock: Mapping[str, Any], paths: Mapping[str, Path],
) -> dict[str, str]:
    bindings = _validate_candidate_lock_shape(lock)
    actual = {
        "actor_sha256": file_hash(paths["actor"]),
        "protocol_sha256": file_hash(paths["protocol"]),
        "runtime_manifest_sha256": file_hash(paths["manifest"]),
        "designation_sha256": file_hash(paths["designation"]),
        "failed_outer_closeout_sha256": file_hash(paths["failure_closeout"]),
        "fresh_outer_registry_sha256": file_hash(paths["registry"]),
        "outer_hash_projection_sha256": file_hash(paths["projection"]),
        "development_rows_sha256": file_hash(paths["development_rows"]),
        "program_sha256": file_hash(paths["program"]),
        "selector_report_sha256": file_hash(paths["selector_report"]),
    }
    if any(bindings[name] != value for name, value in actual.items()):
        raise ValueError("V9 candidate lock artifact binding differs")
    selector_report = _strict_json(paths["selector_report"], "v9 selector report")
    closure = _selector_source_closure(lock, selector_report)
    if closure is not None:
        normalized = dict(closure)
        if (not normalized
                or any(type(name) is not str or type(value) is not str
                       or _HEX.fullmatch(value) is None
                       for name, value in normalized.items())
                or digest(normalized) != bindings["source_closure_sha256"]):
            raise ValueError("V9 candidate source closure binding differs")
    else:
        report_binding = selector_report.get("bindings", {}).get(
            "source_closure_sha256")
        if report_binding != bindings["source_closure_sha256"]:
            raise ValueError("V9 candidate source closure is not authenticated")
    return bindings


def _resolve_snapshot(
    *, lock: Mapping[str, Any], actor_path: str | Path,
    protocol_path: str | Path, manifest_path: str | Path,
    designation_path: str | Path, registry_path: str | Path,
    registry_report_path: str | Path, projection_path: str | Path,
    projection_receipt_path: str | Path, expected_registry_sha256: str,
    expected_registry_report_sha256: str, expected_projection_sha256: str,
    expected_projection_receipt_sha256: str, candidate_lock_path: str | Path,
    expected_candidate_lock_sha256: str, failure_closeout_path: str | Path,
    development_rows_path: str | Path, program_path: str | Path,
    selector_report_path: str | Path,
) -> tuple[ImmutableInputSnapshot, dict[str, Path], dict[str, Path], Path]:
    bindings = _validate_candidate_lock_shape(lock)
    originals = {
        "actor": _regular(actor_path, "frozen Actor", maximum=MAX_NPZ_BYTES),
        "protocol": _regular(protocol_path, "frozen protocol"),
        "manifest": _regular(manifest_path, "runtime manifest"),
        "designation": _regular(designation_path, "Actor designation"),
        "registry": _regular(registry_path, "fresh outer registry"),
        "registry_report": _regular(registry_report_path, "fresh outer registry report"),
        "projection": _regular(projection_path, "outer hash projection"),
        "projection_receipt": _regular(
            projection_receipt_path, "outer hash projection receipt"),
        "candidate_lock": _regular(candidate_lock_path, "v9 candidate lock"),
        "failure_closeout": _regular(
            failure_closeout_path, "failed outer closeout"),
        "development_rows": _regular(
            development_rows_path, "locked development rows", maximum=MAX_NPZ_BYTES),
        "program": _regular(program_path, "locked v9 program"),
        "selector_report": _regular(selector_report_path, "locked selector report"),
    }
    components = designation_binding.resolve_bound_components(originals["designation"])
    if (components["actor"] != originals["actor"]
            or components["protocol"] != originals["protocol"]):
        raise ValueError("Explicit Actor/protocol must be designation components")
    originals["manifest_validation"] = _regular(
        originals["manifest"].parent / "validation.json", "manifest validation")
    expected = {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "registry": _sha(expected_registry_sha256, "fresh outer registry"),
        "registry_report": _sha(
            expected_registry_report_sha256, "fresh outer registry report"),
        "projection": _sha(expected_projection_sha256, "outer hash projection"),
        "projection_receipt": _sha(
            expected_projection_receipt_sha256, "outer projection receipt"),
        "candidate_lock": _sha(expected_candidate_lock_sha256, "candidate lock"),
        "failure_closeout": bindings["failed_outer_closeout_sha256"],
        "development_rows": bindings["development_rows_sha256"],
        "program": bindings["program_sha256"],
        "selector_report": bindings["selector_report_sha256"],
    }
    for name, path in components.items():
        key = "designation_" + name
        originals[key] = path
        expected[key] = {
            "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
            "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
            "training_ledger": designation_binding.designation.EXPECTED_LEDGER_SHA256,
            "dual_evaluation": designation_binding.designation.EXPECTED_DUAL_EVALUATION_SHA256,
            "failure_closeout": designation_binding.designation.EXPECTED_CLOSEOUT_SHA256,
        }[name]
    snapshot = ImmutableInputSnapshot(
        originals, expected_sha256=expected,
        relative_names={"manifest": "manifest/manifest.json",
                        "manifest_validation": "manifest/validation.json"},
        maximum_bytes={"actor": MAX_NPZ_BYTES, "designation_actor": MAX_NPZ_BYTES,
                       "development_rows": MAX_NPZ_BYTES},
        prefix="warehouse-r41-v9-outer-collection-inputs-",
    )
    return snapshot, snapshot.paths, components, originals["designation"]


def _projection_matches_arrays(
    *, projection: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    registry_file_sha256: str, registry_content_sha256: str,
    selected_identity_sha256: str, environment_steps: int,
) -> None:
    replay = projection_api.projection_from_arrays(
        arrays, registry_file_sha256=registry_file_sha256,
        registry_content_sha256=registry_content_sha256,
        selected_identity_sha256=selected_identity_sha256,
        environment_steps=environment_steps)
    # This comparison covers both ordered per-row vectors, their unique set,
    # every count, and every schedule/identity binding.
    if replay != projection:
        expected = projection.get("projection", {})
        actual = replay.get("projection", {})
        if expected.get("ordered_observation_hashes") != actual.get(
                "ordered_observation_hashes"):
            raise ValueError("Fresh outer observation hashes differ in replay order")
        if expected.get("ordered_row_identity_hashes") != actual.get(
                "ordered_row_identity_hashes"):
            raise ValueError("Fresh outer row identities differ in replay order")
        raise ValueError("Fresh outer hash projection binding differs")


def _report(
    *, paths: Mapping[str, Path], lock: Mapping[str, Any],
    projection: Mapping[str, Any], projection_receipt: Mapping[str, Any],
    registry: Mapping[str, Any], registry_report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray], accounting: Mapping[str, Any],
    environment_steps: int, rows_path: Path, projection_copy_path: Path,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    split = arrays["split_validation"]
    kinds = np.char.decode(arrays["kinds"], "ascii").astype(str)
    pair_count = len(rows_v7._effective_pairs(arrays, split))
    value: dict[str, Any] = {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "actor_sha256": file_hash(paths["actor"]),
            "protocol_sha256": file_hash(paths["protocol"]),
            "runtime_manifest_sha256": file_hash(paths["manifest"]),
            "designation_sha256": file_hash(paths["designation"]),
            "failed_outer_closeout_sha256": file_hash(paths["failure_closeout"]),
            "fresh_outer_registry_sha256": file_hash(paths["registry"]),
            "fresh_outer_registry_content_sha256": registry["content_sha256"],
            "fresh_outer_registry_report_sha256": file_hash(paths["registry_report"]),
            "fresh_outer_registry_report_content_sha256": registry_report[
                "content_sha256"],
            "outer_hash_projection_sha256": file_hash(paths["projection"]),
            "outer_hash_projection_content_sha256": projection["content_sha256"],
            "outer_hash_projection_receipt_sha256": file_hash(
                paths["projection_receipt"]),
            "outer_hash_projection_receipt_content_sha256": projection_receipt[
                "content_sha256"],
            "candidate_lock_sha256": file_hash(paths["candidate_lock"]),
            "candidate_lock_content_sha256": lock["content_sha256"],
            "development_rows_sha256": file_hash(paths["development_rows"]),
            "program_sha256": file_hash(paths["program"]),
            "selector_report_sha256": file_hash(paths["selector_report"]),
            "rows_sha256": file_hash(rows_path),
            "rows_semantic_sha256": _arrays_digest(arrays),
            "projection_copy_sha256": file_hash(projection_copy_path),
            "ordered_replay_sha256": projection["projection"][
                "ordered_replay_sha256"],
            "contract_sha256": digest(contract()),
            "source_closure_sha256": digest(sources),
        },
        "collection": {
            "scene_count": projection_api.SCENE_COUNT,
            "partner_count": len(rows_v7.PARTNERS),
            "row_count": len(arrays["observations"]),
            "ordinary_row_count": int(np.sum(kinds == "ordinary")),
            "intervention_row_count": int(np.sum(kinds == "intervention")),
            "effective_pair_count": pair_count,
            "environment_steps": int(environment_steps),
            "row_accounting": deepcopy(dict(accounting)),
            "ordered_projection_verified_row_for_row": True,
            "all_actor_probabilities_and_actions_exact": True,
            "all_submitted_actions_equal_policy_actions": True,
            "score_computed": False,
            "scored": False,
        },
        "artifacts": {
            ROWS_NAME: {
                "file_sha256": file_hash(rows_path),
                "semantic_sha256": _arrays_digest(arrays),
            },
            PROJECTION_COPY_NAME: {
                "file_sha256": file_hash(projection_copy_path),
                "content_sha256": projection["content_sha256"],
            },
        },
        "sources": deepcopy(dict(sources)),
        "information_boundary": {
            "candidate_lock_authenticated_before_outer_replay": True,
            "labels_and_probabilities_collected_only_after_candidate_lock": True,
            "projection_verified_before_rows_publication": True,
            "outer_metrics_or_candidate_score_computed": False,
            "program_loaded_or_executed": False,
            "protected_final_access": False,
            "participant_data_accessed": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    summary: dict[str, Any] = {}
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        summary[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": __import__("hashlib").sha256(
                memoryview(value).cast("B")).hexdigest(),
        }
    return digest(summary)


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    registry_path: str | Path, registry_report_path: str | Path,
    expected_registry_sha256: str, expected_registry_report_sha256: str,
    projection_path: str | Path, projection_receipt_path: str | Path,
    expected_projection_sha256: str, expected_projection_receipt_sha256: str,
    candidate_lock_path: str | Path, expected_candidate_lock_sha256: str,
    failure_closeout_path: str | Path, development_rows_path: str | Path,
    program_path: str | Path, selector_report_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    # Candidate lock existence, bytes, canonical content, and locked status are
    # authenticated before any runtime or outer Actor output is constructed.
    candidate_file = _regular(candidate_lock_path, "v9 candidate lock")
    if file_hash(candidate_file) != _sha(
            expected_candidate_lock_sha256, "v9 candidate lock"):
        raise ValueError("Exact v9 candidate lock bytes required")
    candidate_lock = _strict_json(candidate_file, "v9 candidate lock")
    _validate_candidate_lock_shape(candidate_lock)

    sources = producer_sources()
    snapshot, paths, components, designation_original = _resolve_snapshot(
        lock=candidate_lock, actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        registry_path=registry_path, registry_report_path=registry_report_path,
        projection_path=projection_path,
        projection_receipt_path=projection_receipt_path,
        expected_registry_sha256=expected_registry_sha256,
        expected_registry_report_sha256=expected_registry_report_sha256,
        expected_projection_sha256=expected_projection_sha256,
        expected_projection_receipt_sha256=expected_projection_receipt_sha256,
        candidate_lock_path=candidate_lock_path,
        expected_candidate_lock_sha256=expected_candidate_lock_sha256,
        failure_closeout_path=failure_closeout_path,
        development_rows_path=development_rows_path, program_path=program_path,
        selector_report_path=selector_report_path)
    destination = Path(output).expanduser().absolute()
    temporary: Path | None = None
    published = False
    lock_path: Path | None = None
    lock_fd: int | None = None
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink() or destination.parent.resolve() != destination.parent:
            raise ValueError("Collection output parent is unsafe")
        lock_path = destination.parent / ("." + destination.name + ".lock")
        lock_fd = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        locked = _strict_json(paths["candidate_lock"], "snapshotted v9 candidate lock")
        _validate_candidate_bindings(lock=locked, paths=paths)
        # Projection authentication still does not expose outer labels/probabilities.
        projection, projection_receipt = projection_api.read_saved_projection(
            projection_path=paths["projection"],
            receipt_path=paths["projection_receipt"],
            expected_projection_sha256=expected_projection_sha256,
            expected_receipt_sha256=expected_projection_receipt_sha256)
        actor, scenes, registry, registry_report = projection_api._validate_registry(
            paths=paths, component_originals=components,
            designation_original=designation_original)
        if (projection["identity"]["registry_file_sha256"]
                != file_hash(paths["registry"])
                or locked["bindings"].get("outer_hash_projection_sha256")
                    != file_hash(paths["projection"])):
            raise ValueError("Candidate lock and outer projection identity differ")

        arrays, accounting, environment_steps = projection_api._replay_outer(
            actor_path=paths["actor"], protocol_path=paths["protocol"],
            manifest_path=paths["manifest"], scenes=scenes,
            progress_label="fresh_v9_outer_full_collection")
        _projection_matches_arrays(
            projection=projection, arrays=arrays,
            registry_file_sha256=file_hash(paths["registry"]),
            registry_content_sha256=registry["content_sha256"],
            selected_identity_sha256=registry_report["selection"][
                "selected_identity_sha256"],
            environment_steps=environment_steps)
        # The frozen Actor is checked again after projection parity; no program
        # prediction is evaluated by this collection producer.
        rows_v7._validate_arrays(
            arrays, actor=actor, train_scenes=[], validation_scenes=scenes)

        temporary = Path(tempfile.mkdtemp(
            prefix=".warehouse-r41-v9-outer-collection-", dir=destination.parent))
        rows_path = temporary / ROWS_NAME
        _write_npz(rows_path, arrays)
        reopened = load_authenticated_rows(
            rows_path, expected_rows_sha256=file_hash(rows_path))
        if (set(reopened) != set(arrays)
                or any(not np.array_equal(reopened[name], arrays[name]) for name in arrays)):
            raise RuntimeError("Staged v9 outer row archive changed")
        projection_copy = temporary / PROJECTION_COPY_NAME
        _write_exclusive(projection_copy, paths["projection"].read_bytes())
        report = _report(
            paths=paths, lock=locked, projection=projection,
            projection_receipt=projection_receipt, registry=registry,
            registry_report=registry_report, arrays=arrays, accounting=accounting,
            environment_steps=environment_steps, rows_path=rows_path,
            projection_copy_path=projection_copy, sources=sources)
        _write_exclusive(temporary / REPORT_NAME, _json_bytes(report))
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Outer collection source closure changed before publication")
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Outer collection source closure changed at publication")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
        temporary = None
        published = True
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Outer collection source closure changed after publication")
        return deepcopy(report)
    except BaseException:
        if published:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)
        snapshot.__exit__(None, None, None)


def authenticate_saved_collection(
    *, report_path: str | Path, rows_path: str | Path,
    projection_copy_path: str | Path, expected_report_sha256: str,
    expected_rows_sha256: str, expected_projection_copy_sha256: str,
) -> dict[str, Any]:
    """Authenticate the receipt and opaque rows bytes without loading labels."""
    report_file = _regular(report_path, "v9 outer collection report")
    rows_file = _regular(rows_path, "v9 outer rows", maximum=MAX_NPZ_BYTES)
    projection_file = _regular(
        projection_copy_path, "v9 copied outer hash projection")
    expected = {
        "report": _sha(expected_report_sha256, "v9 outer collection report"),
        "rows": _sha(expected_rows_sha256, "v9 outer rows"),
        "projection": _sha(
            expected_projection_copy_sha256, "v9 copied hash projection"),
    }
    if (file_hash(report_file) != expected["report"]
            or file_hash(rows_file) != expected["rows"]
            or file_hash(projection_file) != expected["projection"]):
        raise ValueError("Exact v9 outer collection artifact set required")
    report = _strict_json(report_file, "v9 outer collection report")
    sources = producer_sources()
    bindings = report.get("bindings")
    artifacts = report.get("artifacts")
    boundary = report.get("information_boundary")
    if (report.get("version") != VERSION
            or report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != STATUS
            or not _content_valid(report) or report.get("contract") != contract()
            or not isinstance(bindings, Mapping) or not isinstance(artifacts, Mapping)
            or set(artifacts) != {ROWS_NAME, PROJECTION_COPY_NAME}
            or artifacts[ROWS_NAME].get("file_sha256") != expected["rows"]
            or artifacts[PROJECTION_COPY_NAME].get("file_sha256")
                != expected["projection"]
            or bindings.get("rows_sha256") != expected["rows"]
            or bindings.get("projection_copy_sha256") != expected["projection"]
            or bindings.get("source_closure_sha256") != digest(sources)
            or report.get("sources") != sources
            or not isinstance(boundary, Mapping)
            or boundary.get("candidate_lock_authenticated_before_outer_replay") is not True
            or boundary.get("labels_and_probabilities_collected_only_after_candidate_lock")
                is not True
            or boundary.get("projection_verified_before_rows_publication") is not True
            or boundary.get("outer_metrics_or_candidate_score_computed") is not False
            or boundary.get("program_loaded_or_executed") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("runtime_action_override") is not False
            or report.get("formal_ready") is not False):
        raise ValueError("V9 outer collection receipt differs")
    if file_hash(report_file) != expected["report"]:
        raise RuntimeError("V9 outer collection report changed during authentication")
    return deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-report", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-report-sha256", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--projection-receipt", required=True)
    parser.add_argument("--expected-projection-sha256", required=True)
    parser.add_argument("--expected-projection-receipt-sha256", required=True)
    parser.add_argument("--candidate-lock", required=True)
    parser.add_argument("--expected-candidate-lock-sha256", required=True)
    parser.add_argument("--failure-closeout", required=True)
    parser.add_argument("--development-rows", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--selector-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        registry_path=args.registry, registry_report_path=args.registry_report,
        expected_registry_sha256=args.expected_registry_sha256,
        expected_registry_report_sha256=args.expected_registry_report_sha256,
        projection_path=args.projection,
        projection_receipt_path=args.projection_receipt,
        expected_projection_sha256=args.expected_projection_sha256,
        expected_projection_receipt_sha256=args.expected_projection_receipt_sha256,
        candidate_lock_path=args.candidate_lock,
        expected_candidate_lock_sha256=args.expected_candidate_lock_sha256,
        failure_closeout_path=args.failure_closeout,
        development_rows_path=args.development_rows, program_path=args.program,
        selector_report_path=args.selector_report, output=args.output)
    print(canonical({"status": report["status"],
                     "rows": report["collection"]["row_count"],
                     "output": str(Path(args.output).expanduser().absolute())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "SCHEMA_VERSION", "STATUS", "CANDIDATE_LOCK_VERSION", "REPORT_NAME",
    "ROWS_NAME", "PROJECTION_COPY_NAME", "contract", "producer_sources",
    "load_authenticated_rows", "build", "authenticate_saved_collection", "main",
]
