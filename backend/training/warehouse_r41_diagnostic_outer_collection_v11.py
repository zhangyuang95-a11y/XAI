"""Collect the full fresh v11 outer rows only after candidate lock.

The companion hash-projection producer commits to the exact replay order while
withholding observations, actions, and probabilities.  This module first
authenticates an immutable candidate lock and every artifact named by that
lock.  Only then does it replay the same schedule, require row-for-row hash
parity, and publish the unscored strict row archive for the one-shot scorer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
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
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v11 as projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_v7
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import ImmutableInputSnapshot


VERSION = "warehouse-r41-diagnostic-outer-collection.v11"
SCHEMA_VERSION = "warehouse_r41_diagnostic_rcpd_v11_outer_collection_v1"
STATUS = "collected_unscored"
CANDIDATE_LOCK_VERSION = "warehouse_r41_diagnostic_rcpd_v11_candidate_lock_v1"
SELECTOR_VERSION = "warehouse-r41-diagnostic-rcpd-v11-fit-selector.v1"
SELECTOR_STATUS = "locked_development_candidate_pending_fresh_outer"
SELECTOR_GRID_VERSION = "warehouse-r41-diagnostic-rcpd-v9-candidate-grid.v1"
CV_SALTS = (
    "warehouse-r41-v9-blocked-cv-a-20260913",
    "warehouse-r41-v9-blocked-cv-b-20260913",
)
CV_FOLDS = 3
REPORT_NAME = "collection_report.json"
ROWS_NAME = "rows.npz"
PROJECTION_COPY_NAME = projection_api.PROJECTION_NAME
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_NPZ_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_LOCK_BINDINGS = frozenset((
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
_LOCK_FIELDS = frozenset((
    "schema_version", "status", "formal_ready", "bindings",
    "source_closure", "selection", "information_boundary", "content_sha256",
))
_LOCK_SELECTION_FIELDS = frozenset((
    "selected_config_sha256", "two_salt_three_fold_development_cv_passed",
    "fresh_outer_scored",
))
_LOCK_BOUNDARY_FIELDS = frozenset((
    "candidate_locked_before_full_fresh_outer_collection",
    "fresh_outer_hashes_used_only_for_validation_wins",
    "prior_outer_hashes_used_for_validation_wins",
    "consumed_v9_outer_labels_or_probabilities_read",
    "consumed_v10_outer_labels_or_probabilities_read",
    "fresh_outer_actions_or_probabilities_read", "protected_final_access",
    "runtime_action_override", "formal_ready",
))
_SELECTOR_FIELDS = frozenset((
    "version", "status", "contract", "bindings", "development",
    "candidate_grid", "selection", "final_fit", "sources",
    "information_boundary", "formal_ready", "content_sha256",
))
_SELECTOR_BINDINGS = frozenset((
    "actor_sha256", "actor_feature_names_sha256", "protocol_sha256",
    "runtime_manifest_sha256", "public_feature_contract_sha256",
    "designation_sha256", "failed_outer_closeout_sha256",
    "fresh_outer_registry_sha256", "fresh_outer_registry_report_sha256",
    "prior_outer_hash_projection_sha256",
    "prior_outer_hash_projection_content_sha256",
    "outer_hash_projection_sha256", "outer_hash_projection_receipt_sha256",
    "development_rows_sha256", "candidate_grid_sha256", "program_sha256",
    "source_closure_sha256",
))
_CANDIDATE_REPORT_FIELDS = frozenset((
    "config", "config_sha256", "salts",
    "both_salts_pass_all_aggregate_gates",
    "robust_minimum_family_exact_bit_direction_fidelity", "observed_capacity",
))
_SALT_REPORT_FIELDS = frozenset((
    "salt", "fold_assignment_sha256", "folds", "aggregate_metrics",
    "aggregate_gate", "family_exact_bit_direction",
))
_FOLD_REPORT_FIELDS = frozenset((
    "fold", "fit_rows", "validation_rows", "fit_scenes", "validation_scenes",
    "fit_scene_fingerprints_sha256", "validation_scene_fingerprints_sha256",
    "fit_diagnostics", "complexity", "metrics", "informational_gate",
))
_FINAL_FIT_FIELDS = frozenset((
    "binding_sha256", "diagnostics", "complexity", "development_metrics",
    "development_gate", "program_file_sha256",
    "runtime_roundtrip_actions_equal", "runtime_roundtrip_max_probability_error",
    "runtime_roundtrip_rows",
))
_SELECTION_FIELDS = frozenset((
    "selected_config", "selected_config_sha256", "selection_key",
    "eligible_config_sha256s", "eligible_count",
))
_DEVELOPMENT_FIELDS = frozenset((
    "failed_v8_outer_permanently_closed",
    "failed_v8_outer_reclassified_as_development", "validation_wins",
    "validation_hash_inputs", "scene_families", "retained_scene_count",
    "retained_row_count",
))
_VALIDATION_WINS_FIELDS = frozenset((
    "source_rows", "retained_rows", "removed_rows",
    "source_unique_observations", "retained_unique_observations",
    "fresh_outer_unique_observations",
    "retained_fresh_outer_observation_overlap", "keep_mask_sha256",
    "retained_observation_hashes_sha256",
    "private_development_members_read_before_mask_frozen",
    "retained_rows_semantic_sha256", "all_retained_rows_marked_development",
    "private_members_read_only_after_mask_frozen",
    "source_archive_reauthenticated_after_private_read", "content_sha256",
))
_VALIDATION_HASH_INPUT_FIELDS = frozenset((
    "prior_outer_file_sha256", "prior_outer_content_sha256",
    "prior_outer_unique_observations",
    "prior_outer_observation_hashes_sha256",
    "fresh_outer_file_sha256", "fresh_outer_content_sha256",
    "fresh_outer_unique_observations",
    "fresh_outer_observation_hashes_sha256",
    "prior_fresh_observation_overlap", "combined_unique_observations",
    "combined_observation_hashes_sha256",
    "labels_or_probabilities_included", "content_sha256",
))
_SELECTOR_BOUNDARY_FIELDS = frozenset((
    "fresh_outer_projection_authenticated_before_development_targets",
    "prior_outer_projection_authenticated_before_development_targets",
    "validation_wins_keep_mask_frozen_before_action_or_probability_read",
    "consumed_v9_outer_labels_or_probabilities_read",
    "consumed_v10_outer_labels_or_probabilities_read",
    "fresh_outer_actions_or_probabilities_read", "fresh_outer_full_collection_access",
    "protected_final_access", "runtime_action_override", "formal_ready",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "population": "fresh v11 identity-frozen development outer only",
        "scene_count": projection_api.SCENE_COUNT,
        "scene_offset": projection_api.SCENE_OFFSET,
        "partners": list(rows_v7.PARTNERS),
        "candidate_lock_required_before_actor_output_collection": True,
        "selector_cv_and_full_development_refit_authenticated_before_collection": True,
        "prior_outer_hash_projection_authenticated_before_collection": True,
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
        raise ValueError("V11 outer row array schema differs")
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
        raise ValueError("V11 outer row shapes or dtypes differ")
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
        raise ValueError("V11 outer numeric or action-authority rows differ")
    expected_observation_hashes = np.asarray(
        [rows_v7.legacy._obs_hash(row) for row in observations], dtype="S64")
    if not np.array_equal(
            expected_observation_hashes, arrays["observation_hashes"]):
        raise ValueError("V11 outer public-observation hashes differ")
    for name in ("observation_hashes", "scene_fingerprints",
                 "source_state_hashes"):
        try:
            values = np.char.decode(arrays[name], "ascii").astype(str)
        except UnicodeDecodeError as error:
            raise ValueError("V11 outer row text is not ASCII") from error
        if any(_HEX.fullmatch(str(value)) is None for value in values):
            raise ValueError("V11 outer row SHA-256 field differs")


def load_authenticated_rows(
    rows_path: str | Path, *, expected_rows_sha256: str,
) -> dict[str, np.ndarray]:
    """Load one exact strict row archive without pickle or extra members."""
    path = _regular(rows_path, "v11 outer rows", maximum=MAX_NPZ_BYTES)
    if file_hash(path) != _sha(expected_rows_sha256, "v11 outer rows"):
        raise ValueError("Exact v11 outer row bytes required")
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
                raise ValueError("V11 outer row archive contents differ")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != rows_v7._FIELDS:
                raise ValueError("V11 outer row array schema differs")
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("V11 outer rows are not a safe strict NPZ") from error
    if file_hash(path) != expected_rows_sha256:
        raise RuntimeError("V11 outer rows changed during strict load")
    _validate_static_rows(arrays)
    return arrays


def _validate_candidate_lock_shape(value: Mapping[str, Any]) -> dict[str, str]:
    bindings = value.get("bindings")
    selection = value.get("selection")
    boundary = value.get("information_boundary")
    closure = value.get("source_closure")
    if (set(value) != _LOCK_FIELDS
            or value.get("schema_version") != CANDIDATE_LOCK_VERSION
            or value.get("status") != "locked"
            or value.get("formal_ready") is not False
            or not _content_valid(value)
            or not isinstance(bindings, Mapping)
            or not _REQUIRED_LOCK_BINDINGS.issubset(bindings)
            or any(type(name) is not str or type(child) is not str
                   or _HEX.fullmatch(child) is None
                   for name, child in bindings.items())
            or not isinstance(closure, Mapping) or not closure
            or any(type(name) is not str or not name or type(child) is not str
                   or _HEX.fullmatch(child) is None
                   for name, child in closure.items())
            or digest(dict(closure)) != bindings.get("source_closure_sha256")
            or not isinstance(selection, Mapping)
            or set(selection) != _LOCK_SELECTION_FIELDS
            or type(selection.get("selected_config_sha256")) is not str
            or _HEX.fullmatch(selection["selected_config_sha256"]) is None
            or selection.get("two_salt_three_fold_development_cv_passed") is not True
            or selection.get("fresh_outer_scored") is not False
            or not isinstance(boundary, Mapping)
            or set(boundary) != _LOCK_BOUNDARY_FIELDS
            or boundary.get("candidate_locked_before_full_fresh_outer_collection")
                is not True
            or boundary.get("fresh_outer_hashes_used_only_for_validation_wins") is not True
            or boundary.get("prior_outer_hashes_used_for_validation_wins") is not True
            or boundary.get(
                "consumed_v9_outer_labels_or_probabilities_read") is not False
            or boundary.get(
                "consumed_v10_outer_labels_or_probabilities_read") is not False
            or boundary.get("fresh_outer_actions_or_probabilities_read") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("runtime_action_override") is not False
            or boundary.get("formal_ready") is not False):
        raise ValueError("Exact locked v11 candidate contract required")
    return dict(bindings)


def _gate_matches(metrics: Any, gate: Any, *, label: str,
                  require_passed: bool) -> bool:
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError(label + " metrics or gate differs")
    try:
        recomputed = metrics_api._gate(metrics)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(label + " metrics cannot be gated") from error
    if dict(gate) != recomputed or (require_passed and recomputed.get("passed") is not True):
        raise ValueError(label + " gate differs")
    return recomputed.get("passed") is True


def _finite_unit(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or type(value) not in (int, float):
        raise ValueError(label + " differs")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(label + " differs")
    return result


def _validate_family_cells(value: Any, *, label: str) -> float:
    if not isinstance(value, Mapping) or set(value) != {
            "cells", "cell_count", "minimum_fidelity"}:
        raise ValueError(label + " cells differ")
    cells = value.get("cells")
    if not isinstance(cells, Mapping) or not cells:
        raise ValueError(label + " cells differ")
    fidelities = []
    for name, row in cells.items():
        if (type(name) is not str or not isinstance(row, Mapping)
                or set(row) != {"family_id", "exact_group_bits", "pairs",
                                "scenes", "fidelity"}
                or type(row.get("family_id")) is not str
                or type(row.get("exact_group_bits")) is not int
                or not 0 <= row["exact_group_bits"] < 8
                or type(row.get("pairs")) is not int or row["pairs"] <= 0
                or type(row.get("scenes")) is not int or row["scenes"] <= 0
                or row["scenes"] > row["pairs"]
                or name != row["family_id"] + "|exact_bits=" + format(
                    row["exact_group_bits"], f"0{len(metrics_api.GROUPS)}b")):
            raise ValueError(label + " cell row differs")
        fidelities.append(_finite_unit(row.get("fidelity"), label + " cell fidelity"))
    minimum = _finite_unit(value.get("minimum_fidelity"), label + " minimum")
    if value.get("cell_count") != len(cells) or minimum != min(fidelities):
        raise ValueError(label + " cell summary differs")
    return minimum


def _validate_validation_wins(
    value: Mapping[str, Any], *, retained_row_count: int,
) -> str:
    if (set(value) != _VALIDATION_WINS_FIELDS or not _content_valid(value)
            or type(value.get("source_rows")) is not int
            or type(value.get("retained_rows")) is not int
            or type(value.get("removed_rows")) is not int
            or value["source_rows"] <= 0 or value["retained_rows"] <= 0
            or value["removed_rows"] < 0
            or value["source_rows"]
                != value["retained_rows"] + value["removed_rows"]
            or value["retained_rows"] != retained_row_count
            or any(type(value.get(name)) is not int or value[name] <= 0
                   for name in (
                       "source_unique_observations",
                       "retained_unique_observations",
                       "fresh_outer_unique_observations"))
            or value["retained_unique_observations"]
                > value["source_unique_observations"]
            or value.get("retained_fresh_outer_observation_overlap") != 0
            or any(type(value.get(name)) is not str
                   or _HEX.fullmatch(value[name]) is None
                   for name in (
                       "keep_mask_sha256", "retained_observation_hashes_sha256",
                       "retained_rows_semantic_sha256"))
            or value.get(
                "private_development_members_read_before_mask_frozen") is not False
            or value.get("all_retained_rows_marked_development") is not True
            or value.get("private_members_read_only_after_mask_frozen") is not True
            or value.get(
                "source_archive_reauthenticated_after_private_read") is not True):
        raise ValueError("Locked v11 selector validation-wins trace differs")
    return value["retained_rows_semantic_sha256"]


def _validate_validation_hash_inputs(value: Mapping[str, Any]) -> None:
    if (set(value) != _VALIDATION_HASH_INPUT_FIELDS
            or not _content_valid(value)
            or any(type(value.get(name)) is not str
                   or _HEX.fullmatch(value[name]) is None
                   for name in (
                       "prior_outer_file_sha256",
                       "prior_outer_content_sha256",
                       "prior_outer_observation_hashes_sha256",
                       "fresh_outer_file_sha256",
                       "fresh_outer_content_sha256",
                       "fresh_outer_observation_hashes_sha256",
                       "combined_observation_hashes_sha256"))
            or any(type(value.get(name)) is not int or value[name] <= 0
                   for name in (
                       "prior_outer_unique_observations",
                       "fresh_outer_unique_observations",
                       "combined_unique_observations"))
            or type(value.get("prior_fresh_observation_overlap")) is not int
            or value["prior_fresh_observation_overlap"] < 0
            or value["prior_fresh_observation_overlap"] > min(
                value["prior_outer_unique_observations"],
                value["fresh_outer_unique_observations"])
            or value["combined_unique_observations"] != (
                value["prior_outer_unique_observations"]
                + value["fresh_outer_unique_observations"]
                - value["prior_fresh_observation_overlap"])
            or value.get("labels_or_probabilities_included") is not False):
        raise ValueError("Locked v11 selector validation hash inputs differ")


def _validate_selector_report(
    lock: Mapping[str, Any], report: Mapping[str, Any], *,
    expected_report_sha256: str, expected_program_sha256: str,
) -> None:
    """Authenticate the official selector result before fresh labels are read."""
    bindings = _validate_candidate_lock_shape(lock)
    report_bindings = report.get("bindings")
    sources = report.get("sources")
    if (set(report) != _SELECTOR_FIELDS
            or report.get("version") != SELECTOR_VERSION
            or report.get("status") != SELECTOR_STATUS
            or report.get("formal_ready") is not False
            or not _content_valid(report)
            or bindings.get("selector_report_sha256") != expected_report_sha256
            or bindings.get("program_sha256") != expected_program_sha256
            or not isinstance(report_bindings, Mapping)
            or set(report_bindings) != _SELECTOR_BINDINGS
            or any(bindings.get(name) != child
                   for name, child in report_bindings.items())
            or report_bindings.get("program_sha256") != expected_program_sha256
            or not isinstance(sources, Mapping)
            or dict(sources) != dict(lock["source_closure"])
            or digest(dict(sources)) != bindings["source_closure_sha256"]):
        raise ValueError("Locked v11 selector report identity or bindings differ")

    contract_value = report.get("contract")
    development = report.get("development")
    validation_wins = development.get("validation_wins") \
        if isinstance(development, Mapping) else None
    validation_hash_inputs = development.get("validation_hash_inputs") \
        if isinstance(development, Mapping) else None
    boundary = report.get("information_boundary")
    if (not isinstance(contract_value, Mapping)
            or contract_value.get("version") != SELECTOR_VERSION
            or contract_value.get(
                "consumed_v9_outer_labels_or_probabilities_read") is not False
            or contract_value.get(
                "consumed_v10_outer_labels_or_probabilities_read") is not False
            or contract_value.get(
                "fresh_v11_outer_labels_or_probabilities_read") is not False
            or contract_value.get("protected_final_access") is not False
            or contract_value.get("runtime_action_override") is not False
            or contract_value.get("formal_ready") is not False
            or not isinstance(development, Mapping)
            or set(development) != _DEVELOPMENT_FIELDS
            or development.get("failed_v8_outer_permanently_closed") is not True
            or development.get("failed_v8_outer_reclassified_as_development") is not True
            or not isinstance(validation_wins, Mapping)
            or not isinstance(validation_hash_inputs, Mapping)
            or validation_hash_inputs.get("prior_outer_file_sha256")
                != bindings.get("prior_outer_hash_projection_sha256")
            or validation_hash_inputs.get("prior_outer_content_sha256")
                != bindings.get("prior_outer_hash_projection_content_sha256")
            or validation_hash_inputs.get("fresh_outer_file_sha256")
                != bindings.get("outer_hash_projection_sha256")
            or validation_hash_inputs.get("labels_or_probabilities_included")
                is not False
            or validation_wins.get("fresh_outer_unique_observations")
                != validation_hash_inputs.get("combined_unique_observations")
            or validation_wins.get("retained_fresh_outer_observation_overlap") != 0
            or validation_wins.get("private_development_members_read_before_mask_frozen")
                is not False
            or validation_wins.get("private_members_read_only_after_mask_frozen") is not True
            or validation_wins.get("all_retained_rows_marked_development") is not True
            or not isinstance(boundary, Mapping)
            or set(boundary) != _SELECTOR_BOUNDARY_FIELDS
            or boundary.get("fresh_outer_projection_authenticated_before_development_targets")
                is not True
            or boundary.get("prior_outer_projection_authenticated_before_development_targets")
                is not True
            or boundary.get("validation_wins_keep_mask_frozen_before_action_or_probability_read")
                is not True
            or boundary.get(
                "consumed_v9_outer_labels_or_probabilities_read") is not False
            or boundary.get(
                "consumed_v10_outer_labels_or_probabilities_read") is not False
            or boundary.get("fresh_outer_actions_or_probabilities_read") is not False
            or boundary.get("fresh_outer_full_collection_access") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("runtime_action_override") is not False
            or boundary.get("formal_ready") is not False):
        raise ValueError("Locked v11 selector information boundary differs")
    _validate_validation_hash_inputs(validation_hash_inputs)
    if (type(development.get("retained_row_count")) is not int
            or development["retained_row_count"] <= 0):
        raise ValueError("Locked v11 selector development row count differs")
    retained_rows_semantic_sha256 = _validate_validation_wins(
        validation_wins, retained_row_count=development["retained_row_count"])

    grid = report.get("candidate_grid")
    selection = report.get("selection")
    if (not isinstance(grid, Mapping)
            or set(grid) != {"config_count", "config_sha256s", "candidate_reports"}
            or not isinstance(selection, Mapping)
            or set(selection) != _SELECTION_FIELDS):
        raise ValueError("Locked v11 selector selection differs")
    candidate_reports = grid.get("candidate_reports")
    config_hashes = grid.get("config_sha256s")
    if (not isinstance(candidate_reports, list) or not candidate_reports
            or not isinstance(config_hashes, list)
            or grid.get("config_count") != len(candidate_reports)
            or len(config_hashes) != len(candidate_reports)
            or len(set(config_hashes)) != len(config_hashes)):
        raise ValueError("Locked v11 selector candidate registry differs")

    normalized_candidates = []
    for candidate, expected_config_sha in zip(candidate_reports, config_hashes):
        if (not isinstance(candidate, Mapping)
                or set(candidate) != _CANDIDATE_REPORT_FIELDS):
            raise ValueError("Locked v11 selector candidate report differs")
        config = candidate.get("config")
        config_sha = candidate.get("config_sha256")
        salts = candidate.get("salts")
        capacity = candidate.get("observed_capacity")
        if (not isinstance(config, Mapping) or type(config_sha) is not str
                or _HEX.fullmatch(config_sha) is None
                or config_sha != digest(dict(config))
                or expected_config_sha != config_sha
                or not isinstance(salts, list) or len(salts) != len(CV_SALTS)
                or not isinstance(capacity, Mapping)):
            raise ValueError("Locked v11 selector candidate report differs")
        salt_passes = []
        salt_minima = []
        complexities = []
        assignment_hashes = []
        for salt_report, expected_salt in zip(salts, CV_SALTS):
            folds = salt_report.get("folds") if isinstance(salt_report, Mapping) else None
            if (not isinstance(salt_report, Mapping)
                    or set(salt_report) != _SALT_REPORT_FIELDS
                    or salt_report.get("salt") != expected_salt
                    or type(salt_report.get("fold_assignment_sha256")) is not str
                    or _HEX.fullmatch(salt_report["fold_assignment_sha256"]) is None
                    or not isinstance(folds, list) or len(folds) != CV_FOLDS
                    or [row.get("fold") if isinstance(row, Mapping) else None
                        for row in folds] != list(range(CV_FOLDS))):
                raise ValueError("Locked v11 selector two-salt CV trace differs")
            assignment_hashes.append(salt_report["fold_assignment_sha256"])
            validation_row_total = 0
            validation_scene_total = 0
            for fold in folds:
                complexity = fold.get("complexity")
                if (set(fold) != _FOLD_REPORT_FIELDS
                        or not isinstance(complexity, Mapping)
                        or type(complexity.get("total_nodes")) is not int
                        or complexity["total_nodes"] <= 0
                        or type(complexity.get("maximum_tree_depth")) is not int
                        or complexity["maximum_tree_depth"] < 0
                        or type(fold.get("fit_rows")) is not int
                        or fold["fit_rows"] <= 0
                        or type(fold.get("validation_rows")) is not int
                        or fold["validation_rows"] <= 0
                        or type(fold.get("fit_scenes")) is not int
                        or fold["fit_scenes"] <= 0
                        or type(fold.get("validation_scenes")) is not int
                        or fold["validation_scenes"] <= 0
                        or type(fold.get("fit_scene_fingerprints_sha256")) is not str
                        or _HEX.fullmatch(fold["fit_scene_fingerprints_sha256"]) is None
                        or type(fold.get("validation_scene_fingerprints_sha256"))
                            is not str
                        or _HEX.fullmatch(
                            fold["validation_scene_fingerprints_sha256"]) is None):
                    raise ValueError("Locked v11 selector fold trace differs")
                if (fold["fit_rows"] + fold["validation_rows"]
                        != development.get("retained_row_count")
                        or fold["fit_scenes"] + fold["validation_scenes"]
                        != development.get("retained_scene_count")
                        or fold["fit_scene_fingerprints_sha256"]
                            == fold["validation_scene_fingerprints_sha256"]
                        or not isinstance(fold.get("fit_diagnostics"), Mapping)):
                    raise ValueError("Locked v11 selector fold partition differs")
                validation_row_total += fold["validation_rows"]
                validation_scene_total += fold["validation_scenes"]
                _gate_matches(
                    fold.get("metrics"), fold.get("informational_gate"),
                    label="Locked v11 selector informational fold",
                    require_passed=False,
                )
                complexities.append((complexity["total_nodes"],
                                     complexity["maximum_tree_depth"]))
            if (validation_row_total != development.get("retained_row_count")
                    or validation_scene_total
                        != development.get("retained_scene_count")):
                raise ValueError("Locked v11 selector fold coverage differs")
            salt_passes.append(_gate_matches(
                salt_report.get("aggregate_metrics"),
                salt_report.get("aggregate_gate"),
                label="Locked v11 selector aggregate", require_passed=False))
            salt_minima.append(_validate_family_cells(
                salt_report.get("family_exact_bit_direction"),
                label="Locked v11 selector robust"))
        if len(set(assignment_hashes)) != len(CV_SALTS):
            raise ValueError("Locked v11 selector salted assignments differ")
        both_pass = all(salt_passes)
        robust = _finite_unit(
            candidate.get("robust_minimum_family_exact_bit_direction_fidelity"),
            "Locked v11 selector robust minimum")
        expected_capacity = {
            "maximum_total_nodes": max(row[0] for row in complexities),
            "maximum_tree_depth": max(row[1] for row in complexities),
            "fold_program_count": len(complexities),
        }
        if (candidate.get("both_salts_pass_all_aggregate_gates") is not both_pass
                or robust != min(salt_minima)
                or dict(capacity) != expected_capacity):
            raise ValueError("Locked v11 selector candidate summary differs")
        normalized_candidates.append(candidate)

    candidate_grid_payload: dict[str, Any] = {
        "version": SELECTOR_GRID_VERSION,
        "configs": [deepcopy(dict(row["config"])) for row in normalized_candidates],
    }
    candidate_grid_payload["content_sha256"] = digest(candidate_grid_payload)
    candidate_grid_file_sha256 = sha256(
        _json_bytes(candidate_grid_payload)).hexdigest()
    if candidate_grid_file_sha256 != bindings["candidate_grid_sha256"]:
        raise ValueError("Locked v11 selector candidate-grid binding differs")

    eligible = [candidate for candidate in normalized_candidates
                if candidate["both_salts_pass_all_aggregate_gates"] is True]
    eligible.sort(key=lambda candidate: (
        -float(candidate["robust_minimum_family_exact_bit_direction_fidelity"]),
        int(candidate["observed_capacity"]["maximum_total_nodes"]),
        int(candidate["observed_capacity"]["maximum_tree_depth"]),
        str(candidate["config_sha256"]),
    ))
    if not eligible:
        raise ValueError("Locked v11 selector has no eligible candidate")
    chosen = eligible[0]
    expected_selection = {
        "selected_config": chosen["config"],
        "selected_config_sha256": chosen["config_sha256"],
        "selection_key": {
            "robust_minimum_family_exact_bit_direction_fidelity": chosen[
                "robust_minimum_family_exact_bit_direction_fidelity"],
            "maximum_total_nodes": chosen["observed_capacity"]["maximum_total_nodes"],
            "maximum_tree_depth": chosen["observed_capacity"]["maximum_tree_depth"],
        },
        "eligible_config_sha256s": [row["config_sha256"] for row in eligible],
        "eligible_count": len(eligible),
    }
    if (dict(selection) != expected_selection
            or lock["selection"]["selected_config_sha256"]
                != chosen["config_sha256"]):
        raise ValueError("Locked v11 selector chosen candidate differs")

    final_fit = report.get("final_fit")
    if (not isinstance(final_fit, Mapping)
            or set(final_fit) != _FINAL_FIT_FIELDS):
        raise ValueError("Locked v11 selector full-development refit differs")
    _gate_matches(final_fit.get("development_metrics"),
                  final_fit.get("development_gate"),
                  label="Locked v11 selector full-development", require_passed=True)
    final_complexity = final_fit.get("complexity")
    expected_final_binding_sha256 = digest({
        "selector": SELECTOR_VERSION,
        "development_rows_sha256": bindings["development_rows_sha256"],
        "retained_rows_semantic_sha256": retained_rows_semantic_sha256,
        "selected_config_sha256": selection["selected_config_sha256"],
        "prior_outer_projection_sha256": bindings[
            "prior_outer_hash_projection_sha256"],
        "prior_outer_projection_content_sha256": bindings[
            "prior_outer_hash_projection_content_sha256"],
        "fresh_outer_projection_sha256": bindings[
            "outer_hash_projection_sha256"],
        "combined_validation_observation_hashes_sha256": (
            validation_hash_inputs["combined_observation_hashes_sha256"]),
    })
    if (final_fit.get("binding_sha256") != expected_final_binding_sha256
            or not isinstance(final_fit.get("diagnostics"), Mapping)
            or not isinstance(final_complexity, Mapping)
            or type(final_complexity.get("total_nodes")) is not int
            or final_complexity["total_nodes"] <= 0
            or type(final_complexity.get("maximum_tree_depth")) is not int
            or final_complexity["maximum_tree_depth"] < 0
            or final_fit.get("program_file_sha256") != expected_program_sha256
            or final_fit.get("runtime_roundtrip_actions_equal") is not True
            or final_fit.get("runtime_roundtrip_max_probability_error") != 0.0
            or type(final_fit.get("runtime_roundtrip_rows")) is not int
            or final_fit["runtime_roundtrip_rows"] <= 0
            or type(development.get("retained_row_count")) is not int
            or development["retained_row_count"] != final_fit["runtime_roundtrip_rows"]
            or type(development.get("retained_scene_count")) is not int
            or development["retained_scene_count"] <= 0):
        raise ValueError("Locked v11 selector full-development refit differs")


def authenticate_locked_candidate_selector(
    *, lock: Mapping[str, Any], selector_report_path: str | Path,
    expected_selector_report_sha256: str, expected_program_sha256: str,
) -> dict[str, Any]:
    path = _regular(selector_report_path, "locked v11 selector report")
    expected = _sha(expected_selector_report_sha256, "locked selector report")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact locked v11 selector report bytes required")
    report = _strict_json_bytes(raw, "locked v11 selector report")
    if file_hash(path) != expected:
        raise RuntimeError("Locked v11 selector report changed during authentication")
    _validate_selector_report(
        lock, report, expected_report_sha256=expected,
        expected_program_sha256=_sha(expected_program_sha256, "locked program"))
    return deepcopy(report)


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
        "fresh_outer_registry_report_sha256": file_hash(paths["registry_report"]),
        "prior_outer_hash_projection_sha256": file_hash(
            paths["prior_projection"]),
        "outer_hash_projection_sha256": file_hash(paths["projection"]),
        "outer_hash_projection_receipt_sha256": file_hash(
            paths["projection_receipt"]),
        "development_rows_sha256": file_hash(paths["development_rows"]),
        "program_sha256": file_hash(paths["program"]),
        "selector_report_sha256": file_hash(paths["selector_report"]),
    }
    if any(bindings[name] != value for name, value in actual.items()):
        raise ValueError("V11 candidate lock artifact binding differs")
    selector_report = authenticate_locked_candidate_selector(
        lock=lock, selector_report_path=paths["selector_report"],
        expected_selector_report_sha256=bindings["selector_report_sha256"],
        expected_program_sha256=bindings["program_sha256"],
    )
    closure = _selector_source_closure(lock, selector_report)
    if closure is not None:
        normalized = dict(closure)
        if (not normalized
                or any(type(name) is not str or type(value) is not str
                       or _HEX.fullmatch(value) is None
                       for name, value in normalized.items())
                or digest(normalized) != bindings["source_closure_sha256"]):
            raise ValueError("V11 candidate source closure binding differs")
    else:
        report_binding = selector_report.get("bindings", {}).get(
            "source_closure_sha256")
        if report_binding != bindings["source_closure_sha256"]:
            raise ValueError("V11 candidate source closure is not authenticated")
    return bindings


def _resolve_snapshot(
    *, lock: Mapping[str, Any], actor_path: str | Path,
    protocol_path: str | Path, manifest_path: str | Path,
    designation_path: str | Path, registry_path: str | Path,
    registry_report_path: str | Path, projection_path: str | Path,
    prior_projection_path: str | Path,
    projection_receipt_path: str | Path, expected_registry_sha256: str,
    expected_registry_report_sha256: str, expected_projection_sha256: str,
    expected_prior_projection_sha256: str,
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
        "prior_projection": _regular(
            prior_projection_path, "prior outer observation-hash projection"),
        "projection": _regular(projection_path, "outer hash projection"),
        "projection_receipt": _regular(
            projection_receipt_path, "outer hash projection receipt"),
        "candidate_lock": _regular(candidate_lock_path, "v11 candidate lock"),
        "failure_closeout": _regular(
            failure_closeout_path, "failed outer closeout"),
        "development_rows": _regular(
            development_rows_path, "locked development rows", maximum=MAX_NPZ_BYTES),
        "program": _regular(program_path, "locked v11 program"),
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
        "prior_projection": _sha(
            expected_prior_projection_sha256,
            "prior outer observation-hash projection"),
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
        prefix="warehouse-r41-v11-outer-collection-inputs-",
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
    prior_projection: Mapping[str, Any],
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
            "prior_outer_hash_projection_sha256": file_hash(
                paths["prior_projection"]),
            "prior_outer_hash_projection_content_sha256": prior_projection[
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
            "selector_report_and_passed_gates_authenticated_before_outer_replay": True,
            "prior_outer_hash_projection_authenticated_before_outer_replay": True,
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
    prior_projection_path: str | Path,
    expected_prior_projection_sha256: str,
    projection_path: str | Path, projection_receipt_path: str | Path,
    expected_projection_sha256: str, expected_projection_receipt_sha256: str,
    candidate_lock_path: str | Path, expected_candidate_lock_sha256: str,
    failure_closeout_path: str | Path, development_rows_path: str | Path,
    program_path: str | Path, selector_report_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    # Candidate lock existence, bytes, canonical content, and locked status are
    # authenticated before any runtime or outer Actor output is constructed.
    candidate_file = _regular(candidate_lock_path, "v11 candidate lock")
    if file_hash(candidate_file) != _sha(
            expected_candidate_lock_sha256, "v11 candidate lock"):
        raise ValueError("Exact v11 candidate lock bytes required")
    candidate_lock = _strict_json(candidate_file, "v11 candidate lock")
    _validate_candidate_lock_shape(candidate_lock)

    sources = producer_sources()
    snapshot, paths, components, designation_original = _resolve_snapshot(
        lock=candidate_lock, actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        registry_path=registry_path, registry_report_path=registry_report_path,
        prior_projection_path=prior_projection_path,
        projection_path=projection_path,
        projection_receipt_path=projection_receipt_path,
        expected_registry_sha256=expected_registry_sha256,
        expected_registry_report_sha256=expected_registry_report_sha256,
        expected_prior_projection_sha256=expected_prior_projection_sha256,
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
        locked = _strict_json(paths["candidate_lock"], "snapshotted v11 candidate lock")
        _validate_candidate_bindings(lock=locked, paths=paths)
        selector_report = _strict_json(
            paths["selector_report"], "locked v11 selector report")
        validation_hash_inputs = selector_report["development"][
            "validation_hash_inputs"]
        prior_projection = _strict_json(
            paths["prior_projection"], "prior outer observation-hash projection")
        prior_hashes = prior_projection.get("outer_observation_hashes")
        if (not _content_valid(prior_projection)
                or prior_projection.get("content_sha256") != locked[
                    "bindings"]["prior_outer_hash_projection_content_sha256"]
                or not isinstance(prior_hashes, list) or not prior_hashes
                or prior_hashes != sorted(set(prior_hashes))
                or any(type(item) is not str or _HEX.fullmatch(item) is None
                       for item in prior_hashes)):
            raise ValueError("Locked prior outer projection content differs")
        # Projection authentication still does not expose outer labels/probabilities.
        projection, projection_receipt = projection_api.read_saved_projection(
            projection_path=paths["projection"],
            receipt_path=paths["projection_receipt"],
            expected_projection_sha256=expected_projection_sha256,
            expected_receipt_sha256=expected_projection_receipt_sha256)
        fresh_hashes = projection["projection"]["unique_observation_hashes"]
        combined_hashes = sorted(set(prior_hashes) | set(fresh_hashes))
        actual_validation_hash_inputs = {
            "prior_outer_file_sha256": file_hash(paths["prior_projection"]),
            "prior_outer_content_sha256": prior_projection["content_sha256"],
            "prior_outer_unique_observations": len(prior_hashes),
            "prior_outer_observation_hashes_sha256": digest(prior_hashes),
            "fresh_outer_file_sha256": file_hash(paths["projection"]),
            "fresh_outer_content_sha256": projection["content_sha256"],
            "fresh_outer_unique_observations": len(fresh_hashes),
            "fresh_outer_observation_hashes_sha256": digest(fresh_hashes),
            "prior_fresh_observation_overlap": len(
                set(prior_hashes) & set(fresh_hashes)),
            "combined_unique_observations": len(combined_hashes),
            "combined_observation_hashes_sha256": digest(combined_hashes),
            "labels_or_probabilities_included": False,
        }
        actual_validation_hash_inputs["content_sha256"] = digest(
            actual_validation_hash_inputs)
        if validation_hash_inputs != actual_validation_hash_inputs:
            raise ValueError("Locked validation hash union differs")
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
            progress_label="fresh_v11_outer_full_collection")
        _projection_matches_arrays(
            projection=projection, arrays=arrays,
            registry_file_sha256=file_hash(paths["registry"]),
            registry_content_sha256=registry["content_sha256"],
            selected_identity_sha256=registry_report["selection"][
                "selected_identity_sha256"],
            environment_steps=environment_steps)
        # The frozen Actor is checked again after projection parity; no program
        # prediction is evaluated by this collection producer.
        # Fresh outer rows are validation-only and intentionally carry unit
        # placeholder weights.  The legacy development validator requires a
        # non-empty fit partition to derive class-balancing weights, so it is
        # not a valid contract for this archive.  Reuse the projection
        # validator that authenticates the same complete row schema, frozen
        # Actor outputs, episode matrix, intervention schedule, and unit
        # weights without consulting a fit split.
        projection_api._validate_projection_replay_arrays(
            arrays, actor=actor, scenes=scenes)

        temporary = Path(tempfile.mkdtemp(
            prefix=".warehouse-r41-v11-outer-collection-", dir=destination.parent))
        rows_path = temporary / ROWS_NAME
        _write_npz(rows_path, arrays)
        reopened = load_authenticated_rows(
            rows_path, expected_rows_sha256=file_hash(rows_path))
        if (set(reopened) != set(arrays)
                or any(not np.array_equal(reopened[name], arrays[name]) for name in arrays)):
            raise RuntimeError("Staged v11 outer row archive changed")
        projection_copy = temporary / PROJECTION_COPY_NAME
        _write_exclusive(projection_copy, paths["projection"].read_bytes())
        report = _report(
            paths=paths, lock=locked, projection=projection,
            projection_receipt=projection_receipt,
            prior_projection=prior_projection, registry=registry,
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
    report_file = _regular(report_path, "v11 outer collection report")
    rows_file = _regular(rows_path, "v11 outer rows", maximum=MAX_NPZ_BYTES)
    projection_file = _regular(
        projection_copy_path, "v11 copied outer hash projection")
    expected = {
        "report": _sha(expected_report_sha256, "v11 outer collection report"),
        "rows": _sha(expected_rows_sha256, "v11 outer rows"),
        "projection": _sha(
            expected_projection_copy_sha256, "v11 copied hash projection"),
    }
    if (file_hash(report_file) != expected["report"]
            or file_hash(rows_file) != expected["rows"]
            or file_hash(projection_file) != expected["projection"]):
        raise ValueError("Exact v11 outer collection artifact set required")
    report = _strict_json(report_file, "v11 outer collection report")
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
            or boundary.get(
                "selector_report_and_passed_gates_authenticated_before_outer_replay")
                is not True
            or boundary.get(
                "prior_outer_hash_projection_authenticated_before_outer_replay")
                is not True
            or boundary.get("labels_and_probabilities_collected_only_after_candidate_lock")
                is not True
            or boundary.get("projection_verified_before_rows_publication") is not True
            or boundary.get("outer_metrics_or_candidate_score_computed") is not False
            or boundary.get("program_loaded_or_executed") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("runtime_action_override") is not False
            or report.get("formal_ready") is not False):
        raise ValueError("V11 outer collection receipt differs")
    if file_hash(report_file) != expected["report"]:
        raise RuntimeError("V11 outer collection report changed during authentication")
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
    parser.add_argument("--prior-outer-hash-projection", required=True)
    parser.add_argument(
        "--expected-prior-outer-hash-projection-sha256", required=True)
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
        prior_projection_path=args.prior_outer_hash_projection,
        expected_prior_projection_sha256=(
            args.expected_prior_outer_hash_projection_sha256),
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
    "load_authenticated_rows", "authenticate_locked_candidate_selector",
    "build", "authenticate_saved_collection", "main",
]
