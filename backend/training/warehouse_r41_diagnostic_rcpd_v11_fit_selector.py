"""Select and lock one public-tree candidate for a replacement v11 fresh outer.

The selector treats the failed v8 outer as development data only after its
permanent closeout is authenticated.  The consumed v9 outer is excluded from
the replacement registry, but its labels and probabilities are not used for
selection.  The replacement v11 outer contributes a label-blind ordered
observation-hash projection.  Those hashes win over the development archive
before any action or probability member is opened.

Candidate selection uses two independently salted, whole-scene three-fold
cross-validations.  A candidate must pass all nine aggregate v8 explanation
gates under both salts.  Among passing candidates, the selector maximises the
worst public scene-family by exact-critical-bit intervention cell and then
chooses the smallest observed explicit program.  No fresh-outer row archive
and no protected final/holdout source is accepted by this module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_outer_failure_closeout_v9 as closeout_api
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v11 as projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_v7
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_fit as fit_api
from backend.training import warehouse_r41_diagnostic_rcpd_v11_outer_split as registry_api
from backend.training import warehouse_r41_diagnostic_rows_v8 as rows_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import ImmutableInputSnapshot
from backend.warehouse_r41_diagnostic_public_features_v9 import R41DiagnosticPublicRelationsV9
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_diagnostic_conflict import conflict_family_id


VERSION = "warehouse-r41-diagnostic-rcpd-v11-fit-selector.v1"
STATUS = "locked_development_candidate_pending_fresh_outer"
LOCK_SCHEMA = "warehouse_r41_diagnostic_rcpd_v11_candidate_lock_v1"
FOLD_ASSIGNMENT_VERSION = (
    "warehouse-r41-diagnostic-rcpd-v11-family-fold-support-repair.v1")
LEGACY_FOLD_ASSIGNMENT_VERSION = (
    "warehouse-r41-diagnostic-rcpd-v11-family-only-folds-retired.v1")
OFFICIAL_FOLD_ASSIGNMENT_SHA256S = {
    "warehouse-r41-v9-blocked-cv-a-20260913": (
        "e06d3167af3783323cb943276ce4c83c0c5b6f3717115bbc8356a94d3dd2d218"),
    "warehouse-r41-v9-blocked-cv-b-20260913": (
        "99fb243bc9a834e4bd1963a3de17a1a7811252329e38429621ab0661ca30195e"),
}
GRID_VERSION = "warehouse-r41-diagnostic-rcpd-v9-candidate-grid.v1"
FROZEN_CANDIDATE_GRID_SHA256 = (
    "900578faaf4aadc4d4b0d25494d2468f654d644a3aa3d23f9b1b155dc149680e"
)
PROGRAM_NAME = "program.json"
REPORT_NAME = "selector_report.json"
LOCK_NAME = "candidate_lock.json"
CV_SALTS = (
    "warehouse-r41-v9-blocked-cv-a-20260913",
    "warehouse-r41-v9-blocked-cv-b-20260913",
)
FOLD_COUNT = 3
MAX_GRID_CONFIGS = 32
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_NPZ_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_EARLY_DEVELOPMENT_MEMBERS = (
    "observation_hashes", "scene_fingerprints", "observations",
)
_PRIOR_PROJECTION_COMPONENTS = frozenset((
    "consumed_v8", "consumed_v9", "consumed_v10",
))
_PRIOR_PROJECTION_FIELDS = frozenset((
    "version", "source_v8_closeout_content_sha256",
    "source_v8_projection_content_sha256",
    "source_v9_closeout_content_sha256",
    "source_v9_projection_content_sha256",
    "source_v10_final_closeout_content_sha256",
    "source_v10_projection_content_sha256", "outer_observation_hashes",
    "unique_outer_observation_hash_count", "outer_observation_hashes_sha256",
    "component_unique_counts", "component_hashes_sha256", "selector_rule",
    "raw_observations_included", "actions_included",
    "probabilities_included", "labels_included",
    "selection_used_this_projection", "formal_ready", "content_sha256",
))
_LOCK_BINDINGS = frozenset((
    "actor_sha256", "actor_feature_names_sha256", "protocol_sha256",
    "runtime_manifest_sha256", "public_feature_contract_sha256",
    "designation_sha256", "failed_outer_closeout_sha256",
    "fresh_outer_registry_sha256", "fresh_outer_registry_report_sha256",
    "prior_outer_hash_projection_sha256",
    "prior_outer_hash_projection_content_sha256",
    "outer_hash_projection_sha256", "outer_hash_projection_receipt_sha256",
    "development_rows_sha256", "candidate_grid_sha256", "program_sha256",
    "selector_report_sha256", "source_closure_sha256",
))


class NoEligibleCandidateError(RuntimeError):
    """The complete frozen grid was evaluated without an eligible candidate."""

    def __init__(self, candidate_reports: Sequence[Mapping[str, Any]]) -> None:
        failures = [{
            "config_sha256": row.get("config_sha256"),
            "evaluation": row.get("evaluation"),
            "both_salts_pass_all_aggregate_gates": row.get(
                "both_salts_pass_all_aggregate_gates"),
        } for row in candidate_reports]
        self.failure_summary_sha256 = digest(failures)
        self.candidate_count = len(candidate_reports)
        super().__init__(
            "No v11 candidate passed both salted aggregate gate suites; "
            "candidate_count=" + str(self.candidate_count)
            + "; deterministic_failure_summary_sha256="
            + self.failure_summary_sha256)


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "development_source": "permanently closed failed-v8 outer rows",
        "candidate_grid_sha256": FROZEN_CANDIDATE_GRID_SHA256,
        "consumed_v9_outer_input": (
            "identity exclusion is bound by the v11 registry; labels and "
            "probabilities are not selector inputs"
        ),
        "consumed_v10_outer_input": (
            "the registry-bound label-free observation-hash projection is "
            "included in validation-wins before development labels are read"
        ),
        "prior_outer_input": (
            "registry-bound union of consumed v8, v9, and v10 one-way "
            "observation hashes only"
        ),
        "prior_outer_projection_bindings": [
            "file_sha256", "content_sha256",
        ],
        "fresh_outer_input": "ordered v11 label-blind observation SHA-256 projection only",
        "validation_wins_before_private_member_read": True,
        "early_development_members": list(_EARLY_DEVELOPMENT_MEMBERS),
        "late_development_members": sorted(
            set(rows_v7._FIELDS) - set(_EARLY_DEVELOPMENT_MEMBERS)),
        "all_retained_failed_v8_rows_reclassified_as_development": True,
        "scene_family_source": "public active-task geometry",
        "cross_validation": {
            "salts": list(CV_SALTS),
            "folds": FOLD_COUNT,
            "unit": "whole scene family-blocked with minimal same-family support swaps",
            "assignment_version": FOLD_ASSIGNMENT_VERSION,
            "assignment_inputs": [
                "public scene fingerprint", "public scene family",
                "public replacement-route row count",
                "public replacement-route episode count",
            ],
            "assignment_uses_action_labels_or_probabilities": False,
            "replacement_support_registry_binding": (
                "development.replacement_support_registry_sha256"),
            "retired_family_only_failure_binding": (
                "development.legacy_family_only_split_failure"),
            "hard_gate_scope": "aggregate out-of-fold rows for each salt",
        },
        "selection": (
            "both salts pass all nine gates; maximise worst family-by-exact-bit "
            "intervention direction cell; then minimise observed explicit capacity"
        ),
        "optional_replacement_specialist": {
            "group": "shared_pickup",
            "route_input": "197-value public observation only",
            "fit_target_scope": "fit-only public-route rows",
            "held_labels_choose_output": False,
            "minimum_fit_and_validation_partition_support_required": True,
            "runtime_action_override": False,
        },
        "consumed_v9_outer_labels_or_probabilities_read": False,
        "consumed_v10_outer_labels_or_probabilities_read": False,
        "fresh_v11_outer_labels_or_probabilities_read": False,
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


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
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
        raise ValueError(label + " must contain one JSON object")
    return value


def _strict_json(path: Path, label: str, *, expected_sha256: str) -> dict[str, Any]:
    expected = _sha(expected_sha256, label + " SHA-256")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact " + label + " bytes required")
    value = _strict_json_bytes(raw, label)
    if file_hash(path) != expected:
        raise RuntimeError(label + " changed during strict read")
    return value


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


def read_candidate_grid(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    candidate = _regular(path, "v9 candidate grid")
    value = _strict_json(candidate, "v9 candidate grid",
                         expected_sha256=expected_sha256)
    if set(value) != {"version", "configs", "content_sha256"}:
        raise ValueError("V9 candidate grid schema differs")
    configs = value.get("configs")
    if (value.get("version") != GRID_VERSION or not _content_valid(value)
            or not isinstance(configs, list)
            or not 1 <= len(configs) <= MAX_GRID_CONFIGS):
        raise ValueError("V9 candidate grid semantics differ")
    normalized = [fit_api.normalize_config(config) for config in configs]
    hashes = [digest(config) for config in normalized]
    if len(hashes) != len(set(hashes)):
        raise ValueError("V9 candidate grid contains duplicate configurations")
    return {"version": GRID_VERSION, "configs": normalized,
            "content_sha256": value["content_sha256"]}


def read_prior_outer_hash_projection(
    path: str | Path, *, expected_sha256: str,
    expected_content_sha256: str,
) -> dict[str, Any]:
    """Authenticate the registry-bound v8/v9/v10 hash-only projection."""
    candidate = _regular(path, "prior outer observation-hash projection")
    value = _strict_json(
        candidate, "prior outer observation-hash projection",
        expected_sha256=expected_sha256)
    expected_content = _sha(
        expected_content_sha256, "prior outer projection content")
    hashes = value.get("outer_observation_hashes")
    counts = value.get("component_unique_counts")
    component_hashes = value.get("component_hashes_sha256")
    source_fields = (
        "source_v8_closeout_content_sha256",
        "source_v8_projection_content_sha256",
        "source_v9_closeout_content_sha256",
        "source_v9_projection_content_sha256",
        "source_v10_final_closeout_content_sha256",
        "source_v10_projection_content_sha256",
    )
    if (set(value) != _PRIOR_PROJECTION_FIELDS
            or value.get("version") != registry_api.VERSION
                + ".validation-wins-exclusion.v1"
            or not _content_valid(value)
            or value.get("content_sha256") != expected_content
            or not isinstance(hashes, list) or not hashes
            or hashes != sorted(set(hashes))
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in hashes)
            or value.get("unique_outer_observation_hash_count") != len(hashes)
            or value.get("outer_observation_hashes_sha256") != digest(hashes)
            or not isinstance(counts, Mapping)
            or set(counts) != _PRIOR_PROJECTION_COMPONENTS
            or any(type(counts[name]) is not int or counts[name] <= 0
                   for name in _PRIOR_PROJECTION_COMPONENTS)
            or not isinstance(component_hashes, Mapping)
            or set(component_hashes) != _PRIOR_PROJECTION_COMPONENTS
            or any(type(component_hashes[name]) is not str
                   or _HEX.fullmatch(component_hashes[name]) is None
                   for name in _PRIOR_PROJECTION_COMPONENTS)
            or any(type(value.get(name)) is not str
                   or _HEX.fullmatch(value[name]) is None
                   for name in source_fields)
            or value.get("selector_rule") != (
                "remove fit rows matching any prior outer observation hash "
                "before reading actions, probabilities, or raw observations")
            or value.get("raw_observations_included") is not False
            or value.get("actions_included") is not False
            or value.get("probabilities_included") is not False
            or value.get("labels_included") is not False
            or value.get("selection_used_this_projection") is not False
            or value.get("formal_ready") is not False
            or file_hash(candidate) != expected_sha256):
        raise ValueError("Prior outer observation-hash projection differs")
    return value


def validation_hash_union(
    prior_hashes: Sequence[str], fresh_hashes: Sequence[str],
) -> list[str]:
    """Return the sorted union used to freeze the development keep mask."""
    normalized: list[list[str]] = []
    for label, values in (("prior", prior_hashes), ("fresh", fresh_hashes)):
        if (not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))):
            raise ValueError(label + " observation hashes differ")
        rows = list(values)
        if (rows != sorted(set(rows))
                or any(type(item) is not str or _HEX.fullmatch(item) is None
                       for item in rows)):
            raise ValueError(label + " observation hashes differ")
        normalized.append(rows)
    return sorted(set(normalized[0]) | set(normalized[1]))


def _safe_npz_directory(path: Path) -> dict[str, zipfile.ZipInfo]:
    expected = {name + ".npy" for name in rows_v7._FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (len(names) != len(set(names)) or set(names) != expected
                    or len(names) != len(expected)
                    or any(info.is_dir() or info.file_size <= 0
                           or info.file_size > MAX_NPZ_MEMBER_BYTES
                           or info.compress_size < 0 or info.flag_bits & 0x1
                           or stat.S_IFMT(info.external_attr >> 16)
                               not in (0, stat.S_IFREG)
                           for info in infos)
                    or sum(info.file_size for info in infos)
                        > MAX_NPZ_EXPANDED_BYTES):
                raise ValueError("Development row archive contents differ")
            return {info.filename: info for info in infos}
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("Development rows are not a safe strict NPZ") from error


def _read_npy_member(path: Path, name: str, *, audit: list[str] | None = None) -> np.ndarray:
    member = name + ".npy"
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            if (info.is_dir() or info.file_size <= 0
                    or info.file_size > MAX_NPZ_MEMBER_BYTES
                    or info.flag_bits & 0x1
                    or stat.S_IFMT(info.external_attr >> 16)
                        not in (0, stat.S_IFREG)):
                raise ValueError("Unsafe development NPZ member: " + member)
            value = np.load(BytesIO(archive.read(info)), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Cannot read development NPZ member: " + member) from error
    if value.dtype.hasobject:
        raise ValueError("Object arrays are forbidden in development rows")
    if audit is not None:
        audit.append(name)
    return value


def read_public_development_projection(
    path: str | Path, *, expected_sha256: str, audit: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read exactly the three label-free members needed for validation-wins."""
    candidate = _regular(path, "failed-v8 development rows", maximum=MAX_NPZ_BYTES)
    expected = _sha(expected_sha256, "failed-v8 development rows")
    if file_hash(candidate) != expected:
        raise ValueError("Exact failed-v8 development row bytes required")
    _safe_npz_directory(candidate)
    values = {name: _read_npy_member(candidate, name, audit=audit)
              for name in _EARLY_DEVELOPMENT_MEMBERS}
    hashes = values["observation_hashes"]
    scenes = values["scene_fingerprints"]
    observations = values["observations"]
    count = len(observations) if observations.ndim == 2 else -1
    if (count <= 0 or observations.shape != (count, 197)
            or observations.dtype != np.dtype(np.float32)
            or hashes.shape != (count,) or hashes.dtype != np.dtype("S64")
            or scenes.shape != (count,) or scenes.dtype != np.dtype("S64")
            or not np.isfinite(observations).all()):
        raise ValueError("Public development projection shapes or dtypes differ")
    try:
        decoded_hashes = np.char.decode(hashes, "ascii").astype(str)
        decoded_scenes = np.char.decode(scenes, "ascii").astype(str)
    except UnicodeDecodeError as error:
        raise ValueError("Public development projection is not ASCII") from error
    if (any(_HEX.fullmatch(str(value)) is None for value in decoded_hashes)
            or any(_HEX.fullmatch(str(value)) is None for value in decoded_scenes)):
        raise ValueError("Public development projection hashes differ")
    expected_hashes = np.asarray(
        [rows_v7.legacy._obs_hash(row) for row in observations], dtype="S64")
    if not np.array_equal(expected_hashes, hashes):
        raise ValueError("Development observation hash projection differs")
    if file_hash(candidate) != expected:
        raise RuntimeError("Development rows changed during public projection")
    return values


def freeze_validation_wins(
    public: Mapping[str, np.ndarray], outer_observation_hashes: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    hashes = np.char.decode(
        np.asarray(public["observation_hashes"]), "ascii").astype(str)
    if (not isinstance(outer_observation_hashes, Sequence)
            or isinstance(outer_observation_hashes, (str, bytes))):
        raise ValueError("Fresh outer observation hash projection differs")
    outer = list(outer_observation_hashes)
    if (outer != sorted(set(outer))
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in outer)):
        raise ValueError("Fresh outer unique observation hashes differ")
    outer_array = np.asarray(outer, dtype="U64")
    keep = ~np.isin(hashes, outer_array)
    retained = hashes[keep]
    overlap = sorted(set(map(str, retained)) & set(outer))
    if overlap:
        raise RuntimeError("Validation-wins failed to remove fresh outer overlap")
    packed = np.ascontiguousarray(keep.astype(np.uint8))
    audit = {
        "source_rows": len(hashes),
        "retained_rows": int(np.sum(keep)),
        "removed_rows": int(np.sum(~keep)),
        "source_unique_observations": len(set(map(str, hashes))),
        "retained_unique_observations": len(set(map(str, retained))),
        "fresh_outer_unique_observations": len(outer),
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": sha256(memoryview(packed).cast("B")).hexdigest(),
        "retained_observation_hashes_sha256": digest(list(map(str, retained))),
        "private_development_members_read_before_mask_frozen": False,
    }
    audit["content_sha256"] = digest(audit)
    return keep.astype(np.bool_), audit


def _load_full_development_rows(path: Path) -> dict[str, np.ndarray]:
    return rows_api._load_npz(path, "failed-v8 development rows")


def prepare_development_rows(
    path: str | Path, *, expected_sha256: str,
    outer_observation_hashes: Sequence[str], actor: NumPyNativeActor,
    early_member_audit: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply overlap removal before opening any target/private NPZ member."""
    candidate = _regular(path, "failed-v8 development rows", maximum=MAX_NPZ_BYTES)
    public = read_public_development_projection(
        candidate, expected_sha256=expected_sha256, audit=early_member_audit)
    keep, overlap_audit = freeze_validation_wins(public, outer_observation_hashes)
    # This full load is deliberately below the frozen keep-mask and zero-overlap
    # assertion.  It is the first point action labels/probabilities are opened.
    full = _load_full_development_rows(candidate)
    if file_hash(candidate) != expected_sha256:
        raise RuntimeError("Development rows changed before private-member read")
    for name in _EARLY_DEVELOPMENT_MEMBERS:
        if not np.array_equal(full[name], public[name]):
            raise RuntimeError("Development public member changed between stages")
    rows_api._validate_base_shapes(full, actor, "failed-v8 development")
    retained = {name: np.asarray(value)[keep].copy()
                for name, value in full.items()}
    if not len(retained["observations"]):
        raise ValueError("Validation-wins removed every development row")
    retained["split_validation"] = np.zeros(
        len(retained["observations"]), dtype=np.bool_)
    decoded = np.char.decode(
        retained["observation_hashes"], "ascii").astype(str)
    if set(map(str, decoded)) & set(outer_observation_hashes):
        raise RuntimeError("Private development load reintroduced outer overlap")
    audit = {
        **overlap_audit,
        "retained_rows_semantic_sha256": rows_api.arrays_digest(retained),
        "all_retained_rows_marked_development": bool(
            not np.any(retained["split_validation"])),
        "private_members_read_only_after_mask_frozen": True,
        "source_archive_reauthenticated_after_private_read": True,
    }
    audit["content_sha256"] = digest({
        key: value for key, value in audit.items() if key != "content_sha256"})
    return retained, audit


def scene_families_from_public_geometry(
    manifest: Mapping[str, Any], *, required_fingerprints: Sequence[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Recompute families from tasks without opening any protected split."""
    batches = manifest.get("candidate_batches")
    splits = manifest.get("splits")
    if not isinstance(batches, list) or not isinstance(splits, Mapping):
        raise ValueError("Authenticated development scene projection differs")
    # Deliberately name only the two development splits.  Never enumerate or
    # copy any other split from the source mapping.
    sources: list[tuple[str, Mapping[str, Any]]] = []
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, list):
            raise ValueError("Development candidate batch differs")
        sources.extend((f"candidate_batch_{batch_index}", row) for row in batch)
    for split_name in ("train", "conflict_validation"):
        rows = splits.get(split_name)
        if not isinstance(rows, list):
            raise ValueError("Authenticated development split differs: " + split_name)
        sources.extend((split_name, row) for row in rows)
    mapping: dict[str, str] = {}
    source_counts: dict[str, int] = defaultdict(int)
    for source, row in sources:
        if not isinstance(row, Mapping):
            raise ValueError("Development scene row differs")
        fingerprint = row.get("fingerprint")
        snapshot = row.get("snapshot")
        state = snapshot.get("state") if isinstance(snapshot, Mapping) else None
        tasks = state.get("tasks") if isinstance(state, Mapping) else None
        if (type(fingerprint) is not str or _HEX.fullmatch(fingerprint) is None
                or not isinstance(tasks, list) or len(tasks) != 2):
            raise ValueError("Development public task geometry differs")
        family = conflict_family_id(tasks)
        if row.get("family_id") != family:
            raise ValueError("Stored scene family differs from public task geometry")
        previous = mapping.setdefault(fingerprint, family)
        if previous != family:
            raise ValueError("One scene fingerprint has conflicting public families")
        source_counts[source] += 1
    required = list(required_fingerprints)
    if (not required or any(type(value) is not str or _HEX.fullmatch(value) is None
                            for value in required)):
        raise ValueError("Required development scene fingerprints differ")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValueError("Development scene-family registry is incomplete")
    selected = {fingerprint: mapping[fingerprint]
                for fingerprint in sorted(set(required))}
    audit = {
        "scene_count": len(selected),
        "family_counts": {
            family: sum(value == family for value in selected.values())
            for family in sorted(set(selected.values()))
        },
        "public_geometry_recomputed": True,
        "stored_family_cross_checked": True,
        "source_counts": dict(sorted(source_counts.items())),
        "protected_split_enumerated": False,
        "protected_final_access": False,
        "mapping_sha256": digest(selected),
    }
    audit["content_sha256"] = digest(audit)
    return selected, audit


def assign_blocked_scene_folds(
    scene_families: Mapping[str, str], *, salt: str,
    replacement_support: Mapping[str, Mapping[str, int]] | None = None,
    minimum_support: Mapping[str, int] | None = None,
    fold_count: int = FOLD_COUNT,
) -> dict[str, int]:
    assignment, _audit = assign_blocked_scene_folds_with_audit(
        scene_families, salt=salt, replacement_support=replacement_support,
        minimum_support=minimum_support, fold_count=fold_count)
    return assignment


def assign_blocked_scene_folds_with_audit(
    scene_families: Mapping[str, str], *, salt: str,
    replacement_support: Mapping[str, Mapping[str, int]] | None = None,
    minimum_support: Mapping[str, int] | None = None,
    fold_count: int = FOLD_COUNT,
) -> tuple[dict[str, int], dict[str, Any]]:
    if type(salt) is not str or not salt or type(fold_count) is not int or fold_count < 2:
        raise ValueError("Blocked-fold parameters differ")
    by_family: dict[str, list[str]] = defaultdict(list)
    for fingerprint, family in scene_families.items():
        if (type(fingerprint) is not str or _HEX.fullmatch(fingerprint) is None
                or type(family) is not str or not family):
            raise ValueError("Blocked-fold scene registry differs")
        by_family[family].append(fingerprint)
    if not by_family or any(len(rows) < fold_count for rows in by_family.values()):
        raise ValueError("Each public family needs at least one scene per fold")
    result: dict[str, int] = {}
    for family, fingerprints in sorted(by_family.items()):
        ranked = sorted(fingerprints, key=lambda fingerprint: (digest({
            "salt": salt, "family": family, "fingerprint": fingerprint,
        }), fingerprint))
        for index, fingerprint in enumerate(ranked):
            result[fingerprint] = index % fold_count
    legacy_assignment_sha256 = digest(dict(sorted(result.items())))
    if replacement_support is None:
        audit = {
            "version": LEGACY_FOLD_ASSIGNMENT_VERSION,
            "legacy_assignment_sha256": legacy_assignment_sha256,
            "repaired_assignment_sha256": legacy_assignment_sha256,
            "repairs": [], "repairs_sha256": digest([]),
            "x_only_public_inputs": True,
            "action_labels_or_probabilities_used": False,
        }
        audit["content_sha256"] = digest(audit)
        return result, audit
    if (set(replacement_support) != set(scene_families)
            or any(not isinstance(value, Mapping)
                   or set(value) != {"rows", "episodes"}
                   or type(value.get("rows")) is not int
                   or type(value.get("episodes")) is not int
                   or value["rows"] < 0 or value["episodes"] < 0
                   or (value["rows"] == 0) != (value["episodes"] == 0)
                   for value in replacement_support.values())):
        raise ValueError("Public replacement-support scene registry differs")
    minimum = dict(minimum_support or {
        "rows": 2, "scenes": 2, "episodes": 2})
    if (set(minimum) != {"rows", "scenes", "episodes"}
            or any(type(value) is not int or value <= 0
                   for value in minimum.values())):
        raise ValueError("Replacement fold minimum support differs")

    def fold_support(assignment: Mapping[str, int]) -> dict[int, dict[str, int]]:
        return {fold: {
            "rows": sum(replacement_support[fingerprint]["rows"]
                        for fingerprint, assigned in assignment.items()
                        if assigned == fold),
            "scenes": sum(replacement_support[fingerprint]["rows"] > 0
                          for fingerprint, assigned in assignment.items()
                          if assigned == fold),
            "episodes": sum(replacement_support[fingerprint]["episodes"]
                            for fingerprint, assigned in assignment.items()
                            if assigned == fold),
        } for fold in range(fold_count)}

    before = fold_support(result)
    repairs: list[dict[str, Any]] = []
    while True:
        support = fold_support(result)
        deficient = [fold for fold in range(fold_count)
                     if any(support[fold][name] < minimum[name]
                            for name in minimum)]
        if not deficient:
            break
        target = deficient[0]
        candidates: list[tuple[int, int, str, str, str, int]] = []
        for route_scene, donor in result.items():
            route_support = replacement_support[route_scene]
            if donor == target or route_support["rows"] <= 0:
                continue
            if any(support[donor][name] - {
                    "rows": route_support["rows"], "scenes": 1,
                    "episodes": route_support["episodes"],
                    }[name] < minimum[name] for name in minimum):
                continue
            family = scene_families[route_scene]
            for zero_scene, assigned in result.items():
                if (assigned == target and scene_families[zero_scene] == family
                        and replacement_support[zero_scene]["rows"] == 0):
                    rank = digest({
                        "version": FOLD_ASSIGNMENT_VERSION, "salt": salt,
                        "target_fold": target, "donor_fold": donor,
                        "family": family, "route_scene": route_scene,
                        "zero_route_scene": zero_scene,
                    })
                    # Every admissible same-family swap changes exactly two
                    # scenes.  Minimise moved public route rows, then moved
                    # public route episodes, before the salted public tie-break.
                    candidates.append((
                        route_support["rows"], route_support["episodes"],
                        rank, route_scene, zero_scene, donor))
        if not candidates:
            raise fit_api.ReplacementSupportError(
                "V11 public same-family swap cannot satisfy frozen support")
        _rows, _episodes, _rank, route_scene, zero_scene, donor = min(candidates)
        family = scene_families[route_scene]
        result[route_scene] = target
        result[zero_scene] = donor
        repairs.append({
            "family": family,
            "route_scene": route_scene,
            "route_from_fold": donor,
            "route_to_fold": target,
            "zero_route_scene": zero_scene,
            "zero_route_from_fold": target,
            "zero_route_to_fold": donor,
        })
        if len(repairs) > len(scene_families):
            raise RuntimeError("Replacement fold repair did not converge")
    after = fold_support(result)
    if (set(result) != set(scene_families)
            or any(any(not any(
                assigned == fold and scene_families[fingerprint] == family
                for fingerprint, assigned in result.items())
                for fold in range(fold_count)) for family in by_family)
            or any(any(after[fold][name] < minimum[name] for name in minimum)
                   for fold in range(fold_count))):
        raise ValueError("Support-repaired whole-scene fold coverage differs")
    audit = {
        "version": FOLD_ASSIGNMENT_VERSION,
        "legacy_assignment_sha256": legacy_assignment_sha256,
        "repaired_assignment_sha256": digest(dict(sorted(result.items()))),
        "minimum_support": minimum,
        "support_before": {str(key): value for key, value in before.items()},
        "support_after": {str(key): value for key, value in after.items()},
        "repairs": repairs,
        "repairs_sha256": digest(repairs),
        "x_only_public_inputs": True,
        "action_labels_or_probabilities_used": False,
    }
    audit["content_sha256"] = digest(audit)
    return result, audit


def public_replacement_scene_support(
    arrays: Mapping[str, np.ndarray],
    relations: R41DiagnosticPublicRelationsV9,
) -> dict[str, dict[str, int]]:
    """Project only public route support needed for fold assignment."""
    scenes = np.char.decode(
        np.asarray(arrays["scene_fingerprints"]), "ascii").astype(str)
    episodes = np.char.decode(
        np.asarray(arrays["episode_ids"]), "ascii").astype(str)
    route = fit_api._replacement_route_mask(arrays["observations"], relations)
    if route.shape != scenes.shape or episodes.shape != scenes.shape:
        raise ValueError("Public replacement-support rows differ")
    result: dict[str, dict[str, int]] = {}
    for fingerprint in sorted(set(map(str, scenes))):
        selected = route & (scenes == fingerprint)
        result[fingerprint] = {
            "rows": int(np.sum(selected)),
            "episodes": len(set(map(str, episodes[selected]))),
        }
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _probability_metrics(
    probabilities: np.ndarray, arrays: Mapping[str, np.ndarray], mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    pairs = rows_v7._effective_pairs(arrays, mask)
    pair_bits = metrics_api._pair_group_bits(arrays, pairs)
    metrics = metrics_api._metrics_from_probabilities(
        probabilities, arrays, mask, pairs=pairs, pair_group_bits=pair_bits)
    return (_json_safe(metrics), _json_safe(metrics_api._gate(metrics)),
            pairs, pair_bits)


def family_exact_bit_direction_cells(
    probabilities: np.ndarray, arrays: Mapping[str, np.ndarray],
    scene_families: Mapping[str, str], *, mask: np.ndarray,
) -> dict[str, Any]:
    pairs = rows_v7._effective_pairs(arrays, mask)
    pair_bits = metrics_api._pair_group_bits(arrays, pairs)
    predictions = np.argmax(probabilities, axis=1).astype(np.uint8)
    correct = predictions == np.asarray(arrays["action_indices"])
    scenes = np.char.decode(
        np.asarray(arrays["scene_fingerprints"]), "ascii").astype(str)
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (wait_row, branch_row) in enumerate(pairs):
        left, right = str(scenes[wait_row]), str(scenes[branch_row])
        if left != right or left not in scene_families:
            raise ValueError("Intervention pair public scene-family binding differs")
        cells[(scene_families[left], int(pair_bits[index]))].append(index)
    rows: dict[str, Any] = {}
    pair_correct = (correct[pairs[:, 0]] & correct[pairs[:, 1]]) \
        if len(pairs) else np.empty(0, dtype=np.bool_)
    for (family, bits), indices in sorted(cells.items()):
        chosen = np.asarray(indices, dtype=np.int64)
        key = family + "|exact_bits=" + format(bits, f"0{len(metrics_api.GROUPS)}b")
        rows[key] = {
            "family_id": family,
            "exact_group_bits": bits,
            "pairs": len(indices),
            "scenes": len(set(map(str, scenes[pairs[chosen, 0]]))),
            "fidelity": float(np.mean(pair_correct[chosen])),
        }
    minimum = min((row["fidelity"] for row in rows.values()), default=0.0)
    return {"cells": rows, "cell_count": len(rows),
            "minimum_fidelity": float(minimum)}


def _fit_arrays(arrays: Mapping[str, np.ndarray], fit_mask: np.ndarray) -> dict[str, np.ndarray]:
    result = dict(arrays)
    # Explicit fold-local split state is passed even though fit_program also
    # receives fit_mask.  Downstream weight builders must never reuse an old
    # split or inspect a held-out fold's action balance.
    result["split_validation"] = (~fit_mask).astype(np.bool_)
    return result


def _replacement_support_audit(
    arrays: Mapping[str, np.ndarray], *, fit_mask: np.ndarray,
    relations: R41DiagnosticPublicRelationsV9, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the frozen replacement prerequisite without reading targets."""
    route = fit_api._replacement_route_mask(arrays["observations"], relations)
    scenes = np.char.decode(
        np.asarray(arrays["scene_fingerprints"]), "ascii").astype(str)
    episodes = np.char.decode(
        np.asarray(arrays["episode_ids"]), "ascii").astype(str)
    selected = np.asarray(fit_mask, dtype=np.bool_)
    if (route.shape != selected.shape or scenes.shape != route.shape
            or episodes.shape != route.shape):
        raise ValueError("V11 replacement support registry differs")
    replacement = config["shared_pickup_replacement"]
    minimum = {
        "rows": replacement["minimum_rows_per_partition"],
        "scenes": replacement["minimum_scenes_per_partition"],
        "episodes": replacement["minimum_episodes_per_partition"],
    }
    partitions = {
        "fit": fit_api._partition_support(route & selected, scenes, episodes),
        "validation": fit_api._partition_support(
            route & ~selected, scenes, episodes),
    }
    passed = (replacement["enabled"] is False or all(
        all(support[name] >= minimum[name] for name in minimum)
        for support in partitions.values()))
    return {
        "minimum_per_partition": minimum,
        "partitions": partitions,
        "passed": bool(passed),
        "support_uses_action_labels_or_probabilities": False,
    }


def legacy_family_only_support_failure(
    arrays: Mapping[str, np.ndarray], *, scene_families: Mapping[str, str],
    relations: R41DiagnosticPublicRelationsV9, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Record why the old public-family-only partition was retired."""
    scenes = np.char.decode(
        np.asarray(arrays["scene_fingerprints"]), "ascii").astype(str)
    for salt in CV_SALTS:
        assignment = assign_blocked_scene_folds(scene_families, salt=salt)
        row_folds = np.asarray(
            [assignment[str(scene)] for scene in scenes], dtype=np.int8)
        for fold in range(FOLD_COUNT):
            audit = _replacement_support_audit(
                arrays, fit_mask=row_folds != fold, relations=relations,
                config=config)
            if audit["passed"] is False:
                result = {
                    "assignment_version": LEGACY_FOLD_ASSIGNMENT_VERSION,
                    "salt": salt,
                    "fold": fold,
                    "fold_assignment_sha256": digest(
                        dict(sorted(assignment.items()))),
                    "support": audit,
                    "reason": "frozen_replacement_support_prerequisite_not_met",
                    "action_labels_or_probabilities_used": False,
                }
                result["content_sha256"] = digest(result)
                return result
    raise RuntimeError(
        "Retired family-only assignment no longer reproduces its public failure")


def _ineligible_support_report(
    normalized: Mapping[str, Any], *, salt: str, fold: int,
    assignment: Mapping[str, int], support_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if support_audit.get("passed") is not False:
        raise ValueError("Ineligible support report requires a failed public precheck")
    failure = {
        "type": "ReplacementSupportError",
        "code": "frozen_replacement_support_prerequisite_not_met",
        "salt": salt,
        "fold": fold,
        "fold_assignment_sha256": digest(dict(sorted(assignment.items()))),
        "support": _json_safe(support_audit),
        "action_labels_or_probabilities_used": False,
    }
    failure["content_sha256"] = digest(failure)
    return {
        "config": deepcopy(normalized),
        "config_sha256": digest(normalized),
        "evaluation": {
            "status": "ineligible_replacement_support",
            "failure": failure,
        },
        "salts": [],
        "both_salts_pass_all_aggregate_gates": False,
        "robust_minimum_family_exact_bit_direction_fidelity": 0.0,
        "observed_capacity": {
            "maximum_total_nodes": 0,
            "maximum_tree_depth": 0,
            "fold_program_count": 0,
        },
    }


def evaluate_candidate(
    arrays: Mapping[str, np.ndarray], *, scene_families: Mapping[str, str],
    relations: R41DiagnosticPublicRelationsV9, config: Mapping[str, Any],
    salts: Sequence[str] = CV_SALTS,
    fit_program: Callable[..., tuple[Any, Mapping[str, Any]]] = fit_api.fit_program,
    predict_program: Callable[..., np.ndarray] = fit_api.predict_in_batches,
) -> dict[str, Any]:
    if tuple(salts) != CV_SALTS:
        raise ValueError("Exactly the two frozen CV salts are required")
    normalized = fit_api.normalize_config(config)
    scenes = np.char.decode(
        np.asarray(arrays["scene_fingerprints"]), "ascii").astype(str)
    observations = np.asarray(arrays["observations"])
    replacement_support = public_replacement_scene_support(arrays, relations)
    all_mask = np.ones(len(observations), dtype=np.bool_)
    salt_reports = []
    observed_complexities = []
    minimum_support = {
        "rows": normalized["shared_pickup_replacement"][
            "minimum_rows_per_partition"],
        "scenes": normalized["shared_pickup_replacement"][
            "minimum_scenes_per_partition"],
        "episodes": normalized["shared_pickup_replacement"][
            "minimum_episodes_per_partition"],
    }
    for salt in salts:
        assignment, _assignment_audit = assign_blocked_scene_folds_with_audit(
            scene_families, salt=salt,
            replacement_support=replacement_support,
            minimum_support=minimum_support)
        row_folds = np.asarray([assignment[str(scene)] for scene in scenes], dtype=np.int8)
        oof = np.zeros((len(observations), len(fit_api.ACTIONS)), dtype=np.float64)
        fold_reports = []
        for fold in range(FOLD_COUNT):
            validation = row_folds == fold
            fit_mask = ~validation
            if not np.any(validation) or not np.any(fit_mask):
                raise ValueError("Blocked fold has an empty fit or validation partition")
            support_audit = _replacement_support_audit(
                arrays, fit_mask=fit_mask, relations=relations,
                config=normalized)
            if support_audit["passed"] is not True:
                return _ineligible_support_report(
                    normalized, salt=salt, fold=fold,
                    assignment=assignment, support_audit=support_audit)
            fold_arrays = _fit_arrays(arrays, fit_mask)
            binding = digest({
                "selector": VERSION, "config_sha256": digest(normalized),
                "salt": salt, "fold": fold,
                "fit_scene_fingerprints_sha256": digest(sorted(set(
                    map(str, scenes[fit_mask])))),
                "validation_scene_fingerprints_sha256": digest(sorted(set(
                    map(str, scenes[validation])))),
            })
            try:
                program, diagnostics = fit_program(
                    fold_arrays, fit_mask, relations=relations,
                    scene_families=scene_families, config=normalized,
                    binding_sha256=binding)
            except fit_api.ReplacementSupportError as error:
                # The same public prerequisite was just authenticated.  A
                # conflicting fit result is implementation drift, not an
                # ineligible data partition that may be silently recorded.
                raise RuntimeError(
                    "Replacement support precheck and fit implementation disagree"
                ) from error
            predicted = np.asarray(predict_program(
                program, observations[validation]), dtype=np.float64)
            if (predicted.shape != (int(np.sum(validation)), len(fit_api.ACTIONS))
                    or not np.isfinite(predicted).all()
                    or np.any(predicted < 0.0)
                    or not np.allclose(predicted.sum(1), 1.0, rtol=0.0, atol=2e-12)):
                raise ValueError("V11 fold prediction probabilities differ")
            oof[validation] = predicted
            # Metrics validate every row in the supplied probability matrix,
            # while the fold mask scores only held-out rows.  Give unscored
            # rows a fixed, exact distribution instead of copying historical
            # float32 Actor probabilities (whose harmless rounding error can
            # exceed the strict float64 normalisation tolerance).  This also
            # keeps non-fold probability targets outside candidate selection.
            complete = np.zeros_like(oof)
            complete[:, 0] = 1.0
            complete[validation] = predicted
            fold_metrics, fold_gate, _, _ = _probability_metrics(
                complete, arrays, validation)
            complexity = _json_safe(program.complexity())
            observed_complexities.append(complexity)
            fold_reports.append({
                "fold": fold,
                "fit_rows": int(np.sum(fit_mask)),
                "validation_rows": int(np.sum(validation)),
                "fit_scenes": len(set(map(str, scenes[fit_mask]))),
                "validation_scenes": len(set(map(str, scenes[validation]))),
                "fit_scene_fingerprints_sha256": digest(sorted(set(
                    map(str, scenes[fit_mask])))),
                "validation_scene_fingerprints_sha256": digest(sorted(set(
                    map(str, scenes[validation])))),
                "fit_diagnostics": _json_safe(diagnostics),
                "complexity": complexity,
                "metrics": fold_metrics,
                "informational_gate": fold_gate,
            })
        if not np.allclose(oof.sum(1), 1.0, rtol=0.0, atol=2e-12):
            raise RuntimeError("Out-of-fold probabilities do not cover every row exactly once")
        aggregate_metrics, aggregate_gate, _, _ = _probability_metrics(
            oof, arrays, all_mask)
        cells = family_exact_bit_direction_cells(
            oof, arrays, scene_families, mask=all_mask)
        salt_reports.append({
            "salt": salt,
            "fold_assignment_sha256": digest(dict(sorted(assignment.items()))),
            "folds": fold_reports,
            "aggregate_metrics": aggregate_metrics,
            "aggregate_gate": aggregate_gate,
            "family_exact_bit_direction": cells,
        })
    both_pass = (len(salt_reports) == len(CV_SALTS)
                 and all(row["aggregate_gate"]["passed"] is True
                         for row in salt_reports))
    max_nodes = max((int(row["total_nodes"]) for row in observed_complexities),
                    default=2**63 - 1)
    max_depth = max((int(row["maximum_tree_depth"])
                     for row in observed_complexities), default=2**31 - 1)
    robust_minimum = min((row["family_exact_bit_direction"]["minimum_fidelity"]
                          for row in salt_reports), default=0.0)
    return {
        "config": deepcopy(normalized),
        "config_sha256": digest(normalized),
        "evaluation": {"status": "completed", "failure": None},
        "salts": salt_reports,
        "both_salts_pass_all_aggregate_gates": both_pass,
        "robust_minimum_family_exact_bit_direction_fidelity": float(robust_minimum),
        "observed_capacity": {
            "maximum_total_nodes": max_nodes,
            "maximum_tree_depth": max_depth,
            "fold_program_count": len(observed_complexities),
        },
    }


def select_candidate(candidate_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [deepcopy(dict(row)) for row in candidate_reports
                if row.get("evaluation", {}).get("status") == "completed"
                and row.get("both_salts_pass_all_aggregate_gates") is True]
    if not eligible:
        raise NoEligibleCandidateError(candidate_reports)
    eligible.sort(key=lambda row: (
        -float(row["robust_minimum_family_exact_bit_direction_fidelity"]),
        int(row["observed_capacity"]["maximum_total_nodes"]),
        int(row["observed_capacity"]["maximum_tree_depth"]),
        str(row["config_sha256"]),
    ))
    selected = eligible[0]
    return {
        "selected_config": deepcopy(selected["config"]),
        "selected_config_sha256": selected["config_sha256"],
        "selection_key": {
            "robust_minimum_family_exact_bit_direction_fidelity": selected[
                "robust_minimum_family_exact_bit_direction_fidelity"],
            "maximum_total_nodes": selected["observed_capacity"]["maximum_total_nodes"],
            "maximum_tree_depth": selected["observed_capacity"]["maximum_tree_depth"],
        },
        "eligible_config_sha256s": [row["config_sha256"] for row in eligible],
        "eligible_count": len(eligible),
    }


def evaluate_grid(
    arrays: Mapping[str, np.ndarray], *, scene_families: Mapping[str, str],
    relations: R41DiagnosticPublicRelationsV9, configs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports = [evaluate_candidate(
        arrays, scene_families=scene_families, relations=relations, config=config)
        for config in configs]
    return reports, select_candidate(reports)


def make_candidate_lock(*, bindings: Mapping[str, str],
                        source_closure: Mapping[str, str],
                        selected_config_sha256: str) -> dict[str, Any]:
    if (not _LOCK_BINDINGS.issubset(bindings)
            or any(type(key) is not str or type(value) is not str
                   or _HEX.fullmatch(value) is None
                   for key, value in bindings.items())
            or not source_closure
            or any(type(key) is not str or not key or type(value) is not str
                   or _HEX.fullmatch(value) is None
                   for key, value in source_closure.items())
            or digest(dict(source_closure)) != bindings.get("source_closure_sha256")):
        raise ValueError("Candidate-lock bindings differ")
    value: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "status": "locked",
        "formal_ready": False,
        "bindings": dict(sorted(bindings.items())),
        "source_closure": dict(sorted(source_closure.items())),
        "selection": {
            "selected_config_sha256": _sha(
                selected_config_sha256, "selected v11 fit configuration"),
            "two_salt_three_fold_development_cv_passed": True,
            "fresh_outer_scored": False,
        },
        "information_boundary": {
            "candidate_locked_before_full_fresh_outer_collection": True,
            "fresh_outer_hashes_used_only_for_validation_wins": True,
            "prior_outer_hashes_used_for_validation_wins": True,
            "consumed_v9_outer_labels_or_probabilities_read": False,
            "consumed_v10_outer_labels_or_probabilities_read": False,
            "fresh_outer_actions_or_probabilities_read": False,
            "protected_final_access": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
    }
    value["content_sha256"] = digest(value)
    return value


def _resolve_snapshot(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    failure_closeout_path: str | Path, expected_failure_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
    registry_path: str | Path, registry_report_path: str | Path,
    expected_registry_sha256: str, expected_registry_report_sha256: str,
    prior_projection_path: str | Path,
    expected_prior_projection_sha256: str,
    projection_path: str | Path, projection_receipt_path: str | Path,
    expected_projection_sha256: str, expected_projection_receipt_sha256: str,
    development_rows_path: str | Path, candidate_grid_path: str | Path,
    expected_candidate_grid_sha256: str,
) -> tuple[ImmutableInputSnapshot, dict[str, Path], dict[str, Path], Path,
           dict[str, Any]]:
    candidate_grid_sha256 = _sha(
        expected_candidate_grid_sha256, "frozen candidate grid")
    if candidate_grid_sha256 != FROZEN_CANDIDATE_GRID_SHA256:
        raise ValueError("Exact pre-v9-outer candidate grid SHA-256 required")
    closeout_original = _regular(failure_closeout_path, "failed outer closeout")
    permanent = _directory(permanent_closeout_registry, "permanent closeout registry")
    closeout = closeout_api.read_saved_closeout(
        closeout_original,
        expected_closeout_sha256=expected_failure_closeout_sha256,
        permanent_registry=permanent)
    development_sha = _sha(
        closeout.get("source", {}).get("v8_combined_rows_sha256"),
        "closed failed-v8 development rows")
    originals = {
        "actor": _regular(actor_path, "frozen Actor", maximum=MAX_NPZ_BYTES),
        "protocol": _regular(protocol_path, "frozen protocol"),
        "manifest": _regular(manifest_path, "runtime manifest"),
        "designation": _regular(designation_path, "Actor designation"),
        "failure_closeout": closeout_original,
        "registry": _regular(registry_path, "fresh outer registry"),
        "registry_report": _regular(registry_report_path, "fresh outer registry report"),
        "prior_projection": _regular(
            prior_projection_path, "prior outer observation-hash projection"),
        "projection": _regular(projection_path, "fresh outer hash projection"),
        "projection_receipt": _regular(
            projection_receipt_path, "fresh outer projection receipt"),
        "development_rows": _regular(
            development_rows_path, "failed-v8 development rows", maximum=MAX_NPZ_BYTES),
        "candidate_grid": _regular(candidate_grid_path, "v9 candidate grid"),
    }
    components = designation_binding.resolve_bound_components(originals["designation"])
    if components["actor"] != originals["actor"] or components["protocol"] != originals["protocol"]:
        raise ValueError("Explicit Actor/protocol must be designation components")
    originals["manifest_validation"] = _regular(
        originals["manifest"].parent / "validation.json", "manifest validation")
    expected = {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "failure_closeout": _sha(
            expected_failure_closeout_sha256, "failed outer closeout"),
        "registry": _sha(expected_registry_sha256, "fresh outer registry"),
        "registry_report": _sha(
            expected_registry_report_sha256, "fresh outer registry report"),
        "prior_projection": _sha(
            expected_prior_projection_sha256,
            "prior outer observation-hash projection"),
        "projection": _sha(expected_projection_sha256, "fresh outer projection"),
        "projection_receipt": _sha(
            expected_projection_receipt_sha256, "fresh outer projection receipt"),
        "development_rows": development_sha,
        "candidate_grid": candidate_grid_sha256,
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
        prefix="warehouse-r41-v11-fit-selector-inputs-",
    )
    return snapshot, snapshot.paths, components, originals["designation"], closeout


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    failure_closeout_path: str | Path, expected_failure_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
    registry_path: str | Path, registry_report_path: str | Path,
    expected_registry_sha256: str, expected_registry_report_sha256: str,
    prior_projection_path: str | Path,
    expected_prior_projection_sha256: str,
    projection_path: str | Path, projection_receipt_path: str | Path,
    expected_projection_sha256: str, expected_projection_receipt_sha256: str,
    development_rows_path: str | Path, candidate_grid_path: str | Path,
    expected_candidate_grid_sha256: str, output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    snapshot, paths, components, designation_original, closeout = _resolve_snapshot(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        failure_closeout_path=failure_closeout_path,
        expected_failure_closeout_sha256=expected_failure_closeout_sha256,
        permanent_closeout_registry=permanent_closeout_registry,
        registry_path=registry_path, registry_report_path=registry_report_path,
        expected_registry_sha256=expected_registry_sha256,
        expected_registry_report_sha256=expected_registry_report_sha256,
        prior_projection_path=prior_projection_path,
        expected_prior_projection_sha256=expected_prior_projection_sha256,
        projection_path=projection_path, projection_receipt_path=projection_receipt_path,
        expected_projection_sha256=expected_projection_sha256,
        expected_projection_receipt_sha256=expected_projection_receipt_sha256,
        development_rows_path=development_rows_path,
        candidate_grid_path=candidate_grid_path,
        expected_candidate_grid_sha256=expected_candidate_grid_sha256)
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
            raise ValueError("Selector output parent is unsafe")
        lock_path = destination.parent / ("." + destination.name + ".lock")
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                          | getattr(os, "O_NOFOLLOW", 0), 0o600)

        designation = designation_binding.read_bound_designation_snapshot(
            paths["designation"], original_path=designation_original,
            components={name: paths["designation_" + name] for name in components},
            original_components=components,
            expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256)
        actor, _outer_scenes, registry, registry_report = projection_api._validate_registry(
            paths=paths, component_originals=components,
            designation_original=designation_original)
        manifest = manifest_binding.read_saved_manifest(
            paths["manifest"], actor_path=paths["actor"], replay_scope="development")
        projection, projection_receipt = projection_api.read_saved_projection(
            projection_path=paths["projection"], receipt_path=paths["projection_receipt"],
            expected_projection_sha256=expected_projection_sha256,
            expected_receipt_sha256=expected_projection_receipt_sha256)
        grid = read_candidate_grid(
            paths["candidate_grid"], expected_sha256=expected_candidate_grid_sha256)
        closed_rows_sha = closeout["source"]["v8_combined_rows_sha256"]
        receipt_bindings = projection_receipt.get("bindings", {})
        registry_bindings = registry.get("bindings", {})
        prior_projection_file_sha256 = _sha(
            registry_bindings.get(
                "outer_observation_hash_projection_file_sha256"),
            "registry-bound prior outer projection file")
        prior_projection_content_sha256 = _sha(
            registry_bindings.get(
                "outer_observation_hash_projection_content_sha256"),
            "registry-bound prior outer projection content")
        if (prior_projection_file_sha256
                != _sha(expected_prior_projection_sha256,
                        "prior outer projection")
                or file_hash(paths["prior_projection"])
                    != prior_projection_file_sha256):
            raise ValueError(
                "Prior outer projection does not match the v11 registry")
        prior_projection = read_prior_outer_hash_projection(
            paths["prior_projection"],
            expected_sha256=prior_projection_file_sha256,
            expected_content_sha256=prior_projection_content_sha256)
        if (designation.get("bindings", {}).get("actor_sha256") != actor.artifact_sha256
                or receipt_bindings.get("actor_sha256") != actor.artifact_sha256
                or receipt_bindings.get("protocol_sha256") != file_hash(paths["protocol"])
                or receipt_bindings.get("runtime_manifest_sha256") != file_hash(paths["manifest"])
                or receipt_bindings.get("designation_sha256") != file_hash(paths["designation"])
                or receipt_bindings.get("fresh_outer_registry_sha256")
                    != file_hash(paths["registry"])
                or receipt_bindings.get("fresh_outer_registry_report_sha256")
                    != file_hash(paths["registry_report"])
                or registry_bindings.get("failed_rows_file_sha256") != closed_rows_sha
                or registry_bindings.get("failure_closeout_file_sha256")
                    != file_hash(paths["failure_closeout"])
                or closeout.get("campaign_key_inputs", {}).get("actor_sha256")
                    != actor.artifact_sha256
                or closeout.get("campaign_key_inputs", {}).get("manifest_sha256")
                    != file_hash(paths["manifest"])):
            raise ValueError("V11 selector evidence bindings differ")

        prior_hashes = prior_projection["outer_observation_hashes"]
        fresh_hashes = projection["projection"]["unique_observation_hashes"]
        outer_hashes = validation_hash_union(prior_hashes, fresh_hashes)
        validation_hash_inputs = {
            "prior_outer_file_sha256": prior_projection_file_sha256,
            "prior_outer_content_sha256": prior_projection_content_sha256,
            "prior_outer_unique_observations": len(prior_hashes),
            "prior_outer_observation_hashes_sha256": digest(prior_hashes),
            "fresh_outer_file_sha256": file_hash(paths["projection"]),
            "fresh_outer_content_sha256": projection["content_sha256"],
            "fresh_outer_unique_observations": len(fresh_hashes),
            "fresh_outer_observation_hashes_sha256": digest(fresh_hashes),
            "prior_fresh_observation_overlap": len(
                set(prior_hashes) & set(fresh_hashes)),
            "combined_unique_observations": len(outer_hashes),
            "combined_observation_hashes_sha256": digest(outer_hashes),
            "labels_or_probabilities_included": False,
        }
        validation_hash_inputs["content_sha256"] = digest(
            validation_hash_inputs)
        arrays, overlap_audit = prepare_development_rows(
            paths["development_rows"], expected_sha256=closed_rows_sha,
            outer_observation_hashes=outer_hashes, actor=actor)
        scene_values = np.char.decode(
            arrays["scene_fingerprints"], "ascii").astype(str)
        scene_families, family_audit = scene_families_from_public_geometry(
            manifest, required_fingerprints=list(map(str, scene_values)))
        relations = R41DiagnosticPublicRelationsV9(actor.metadata["feature_names"])
        actor_feature_names = tuple(actor.metadata["feature_names"])
        if relations.base_feature_names != actor_feature_names:
            raise ValueError("V11 relation base features differ from the frozen Actor")
        actor_feature_names_sha256 = digest(list(actor_feature_names))
        replacement_support_registry = public_replacement_scene_support(
            arrays, relations)
        replacement_support_registry_sha256 = digest(
            replacement_support_registry)
        frozen_replacement = grid["configs"][0]["shared_pickup_replacement"]
        minimum_support = {
            "rows": frozen_replacement["minimum_rows_per_partition"],
            "scenes": frozen_replacement["minimum_scenes_per_partition"],
            "episodes": frozen_replacement["minimum_episodes_per_partition"],
        }
        fold_assignment_repairs = {}
        for salt in CV_SALTS:
            _assignment, assignment_audit = assign_blocked_scene_folds_with_audit(
                scene_families, salt=salt,
                replacement_support=replacement_support_registry,
                minimum_support=minimum_support)
            if (assignment_audit["repaired_assignment_sha256"]
                    != OFFICIAL_FOLD_ASSIGNMENT_SHA256S[salt]):
                raise RuntimeError("Official v11 fold-assignment regression differs")
            fold_assignment_repairs[salt] = assignment_audit
        legacy_support_failure = legacy_family_only_support_failure(
            arrays, scene_families=scene_families, relations=relations,
            config=grid["configs"][0])
        candidate_reports, selection = evaluate_grid(
            arrays, scene_families=scene_families, relations=relations,
            configs=grid["configs"])
        all_fit = np.ones(len(arrays["observations"]), dtype=np.bool_)
        final_arrays = _fit_arrays(arrays, all_fit)
        final_binding = digest({
            "selector": VERSION,
            "development_rows_sha256": closed_rows_sha,
            "retained_rows_semantic_sha256": overlap_audit[
                "retained_rows_semantic_sha256"],
            "selected_config_sha256": selection["selected_config_sha256"],
            "prior_outer_projection_sha256": prior_projection_file_sha256,
            "prior_outer_projection_content_sha256": (
                prior_projection_content_sha256),
            "fresh_outer_projection_sha256": file_hash(paths["projection"]),
            "combined_validation_observation_hashes_sha256": digest(
                outer_hashes),
        })
        program, final_diagnostics = fit_api.fit_program(
            final_arrays, all_fit, relations=relations,
            scene_families=scene_families,
            config=selection["selected_config"], binding_sha256=final_binding)
        program_payload = program.to_dict()
        # Round-trip through the exact runtime class before locking bytes.
        runtime_program = type(program).from_dict(program_payload)
        if (program.base_feature_names != actor_feature_names
                or runtime_program.base_feature_names != actor_feature_names
                or program.relations.contract() != relations.contract()
                or runtime_program.relations.contract() != relations.contract()):
            raise RuntimeError("Locked v11 program relation/Actor binding differs")
        final_probabilities = fit_api.predict_in_batches(
            program, arrays["observations"])
        runtime_probabilities = fit_api.predict_in_batches(
            runtime_program, arrays["observations"])
        final_metrics, final_gate, _, _ = _probability_metrics(
            final_probabilities, arrays, all_fit)
        roundtrip_max_error = float(np.max(np.abs(
            final_probabilities - runtime_probabilities)))
        if (not np.array_equal(
                np.argmax(final_probabilities, axis=1),
                np.argmax(runtime_probabilities, axis=1))
                or roundtrip_max_error != 0.0):
            raise RuntimeError("Locked v11 program round-trip action parity differs")
        if final_gate.get("passed") is not True:
            raise RuntimeError("Full-development v11 refit failed the nine hard gates")

        temporary = Path(tempfile.mkdtemp(
            prefix=".warehouse-r41-v11-fit-selector-", dir=destination.parent))
        program_path = temporary / PROGRAM_NAME
        _write_exclusive(program_path, _json_bytes(program_payload))
        common_bindings = {
            "actor_sha256": file_hash(paths["actor"]),
            "actor_feature_names_sha256": actor_feature_names_sha256,
            "public_feature_contract_sha256": digest(relations.contract()),
            "protocol_sha256": file_hash(paths["protocol"]),
            "runtime_manifest_sha256": file_hash(paths["manifest"]),
            "designation_sha256": file_hash(paths["designation"]),
            "failed_outer_closeout_sha256": file_hash(paths["failure_closeout"]),
            "fresh_outer_registry_sha256": file_hash(paths["registry"]),
            "fresh_outer_registry_report_sha256": file_hash(paths["registry_report"]),
            "prior_outer_hash_projection_sha256": (
                prior_projection_file_sha256),
            "prior_outer_hash_projection_content_sha256": (
                prior_projection_content_sha256),
            "outer_hash_projection_sha256": file_hash(paths["projection"]),
            "outer_hash_projection_receipt_sha256": file_hash(paths["projection_receipt"]),
            "development_rows_sha256": file_hash(paths["development_rows"]),
            "candidate_grid_sha256": file_hash(paths["candidate_grid"]),
            "program_sha256": file_hash(program_path),
            "source_closure_sha256": digest(sources),
        }
        report: dict[str, Any] = {
            "version": VERSION,
            "status": STATUS,
            "contract": contract(),
            "bindings": deepcopy(common_bindings),
            "development": {
                "failed_v8_outer_permanently_closed": True,
                "failed_v8_outer_reclassified_as_development": True,
                "validation_wins": overlap_audit,
                "validation_hash_inputs": validation_hash_inputs,
                "scene_families": family_audit,
                "fold_assignment_version": FOLD_ASSIGNMENT_VERSION,
                "replacement_support_registry_sha256": (
                    replacement_support_registry_sha256),
                "fold_assignment_repairs": fold_assignment_repairs,
                "legacy_family_only_split_failure": legacy_support_failure,
                "retained_scene_count": len(set(map(str, scene_values))),
                "retained_row_count": len(arrays["observations"]),
            },
            "candidate_grid": {
                "config_count": len(grid["configs"]),
                "config_sha256s": [digest(config) for config in grid["configs"]],
                "candidate_reports": candidate_reports,
            },
            "selection": selection,
            "final_fit": {
                "binding_sha256": final_binding,
                "diagnostics": _json_safe(final_diagnostics),
                "complexity": _json_safe(program.complexity()),
                "development_metrics": final_metrics,
                "development_gate": final_gate,
                "program_file_sha256": file_hash(program_path),
                "runtime_roundtrip_actions_equal": True,
                "runtime_roundtrip_max_probability_error": roundtrip_max_error,
                "runtime_roundtrip_rows": len(arrays["observations"]),
            },
            "sources": deepcopy(sources),
            "information_boundary": {
                "fresh_outer_projection_authenticated_before_development_targets": True,
                "prior_outer_projection_authenticated_before_development_targets": True,
                "validation_wins_keep_mask_frozen_before_action_or_probability_read": True,
                "consumed_v9_outer_labels_or_probabilities_read": False,
                "consumed_v10_outer_labels_or_probabilities_read": False,
                "fresh_outer_actions_or_probabilities_read": False,
                "fresh_outer_full_collection_access": False,
                "protected_final_access": False,
                "runtime_action_override": False,
                "formal_ready": False,
            },
            "formal_ready": False,
        }
        report["content_sha256"] = digest(report)
        report_path = temporary / REPORT_NAME
        _write_exclusive(report_path, _json_bytes(report))
        lock_bindings = {
            **common_bindings,
            "selector_report_sha256": file_hash(report_path),
        }
        candidate_lock = make_candidate_lock(
            bindings=lock_bindings, source_closure=sources,
            selected_config_sha256=selection["selected_config_sha256"])
        _write_exclusive(temporary / LOCK_NAME, _json_bytes(candidate_lock))

        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Selector source closure changed before publication")
        closeout_api.read_saved_closeout(
            failure_closeout_path,
            expected_closeout_sha256=expected_failure_closeout_sha256,
            permanent_registry=permanent_closeout_registry)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Selector source closure changed at publication")
        os.rename(temporary, destination)
        temporary = None
        published = True
        snapshot.verify()
        closeout_api.read_saved_closeout(
            failure_closeout_path,
            expected_closeout_sha256=expected_failure_closeout_sha256,
            permanent_registry=permanent_closeout_registry)
        if producer_sources() != sources:
            raise RuntimeError("Selector source closure changed after publication")
        return deepcopy(candidate_lock)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--failure-closeout", required=True)
    parser.add_argument("--expected-failure-closeout-sha256", required=True)
    parser.add_argument("--permanent-closeout-registry", required=True)
    parser.add_argument("--fresh-outer-registry", required=True)
    parser.add_argument("--expected-fresh-outer-registry-sha256", required=True)
    parser.add_argument("--fresh-outer-registry-report", required=True)
    parser.add_argument("--expected-fresh-outer-registry-report-sha256", required=True)
    parser.add_argument("--prior-outer-hash-projection", required=True)
    parser.add_argument(
        "--expected-prior-outer-hash-projection-sha256", required=True)
    parser.add_argument("--outer-hash-projection", required=True)
    parser.add_argument("--expected-outer-hash-projection-sha256", required=True)
    parser.add_argument("--outer-hash-projection-receipt", required=True)
    parser.add_argument("--expected-outer-hash-projection-receipt-sha256", required=True)
    parser.add_argument("--development-rows", required=True)
    parser.add_argument("--candidate-configs", required=True)
    parser.add_argument("--expected-candidate-configs-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    lock = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        failure_closeout_path=args.failure_closeout,
        expected_failure_closeout_sha256=args.expected_failure_closeout_sha256,
        permanent_closeout_registry=args.permanent_closeout_registry,
        registry_path=args.fresh_outer_registry,
        registry_report_path=args.fresh_outer_registry_report,
        expected_registry_sha256=args.expected_fresh_outer_registry_sha256,
        expected_registry_report_sha256=args.expected_fresh_outer_registry_report_sha256,
        prior_projection_path=args.prior_outer_hash_projection,
        expected_prior_projection_sha256=(
            args.expected_prior_outer_hash_projection_sha256),
        projection_path=args.outer_hash_projection,
        projection_receipt_path=args.outer_hash_projection_receipt,
        expected_projection_sha256=args.expected_outer_hash_projection_sha256,
        expected_projection_receipt_sha256=(
            args.expected_outer_hash_projection_receipt_sha256),
        development_rows_path=args.development_rows,
        candidate_grid_path=args.candidate_configs,
        expected_candidate_grid_sha256=args.expected_candidate_configs_sha256,
        output=args.output)
    print(canonical(lock))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "LOCK_SCHEMA", "GRID_VERSION",
    "FOLD_ASSIGNMENT_VERSION", "LEGACY_FOLD_ASSIGNMENT_VERSION",
    "OFFICIAL_FOLD_ASSIGNMENT_SHA256S",
    "FROZEN_CANDIDATE_GRID_SHA256", "CV_SALTS",
    "FOLD_COUNT", "contract", "producer_sources", "read_candidate_grid",
    "read_prior_outer_hash_projection", "validation_hash_union",
    "read_public_development_projection", "freeze_validation_wins",
    "prepare_development_rows", "scene_families_from_public_geometry",
    "assign_blocked_scene_folds", "public_replacement_scene_support",
    "legacy_family_only_support_failure", "family_exact_bit_direction_cells",
    "evaluate_candidate", "select_candidate", "evaluate_grid",
    "NoEligibleCandidateError", "make_candidate_lock", "build", "main",
]
