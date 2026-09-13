"""Irrevocably close the failed diagnostic RCPD v8 outer evaluation.

This producer authenticates the complete 25-artifact v8 evidence directory
through the strict v8 reader.  It records the failed gates, permanently marks
the 64 scored outer identities as development-exposed, and projects only the
already-authenticated outer public-observation hashes.  Raw observations,
actions, and probabilities are never published by this module.

The permanent campaign anchor is created with ``O_EXCL`` before the ordinary
output is published.  A repeated closeout of the same campaign therefore
fails even if a previous invocation stopped after writing the anchor.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from backend.training import warehouse_r41_diagnostic_rcpd_v8 as v8
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-outer-failure-closeout.v9"
STATUS = "failed_outer_irrevocably_closed"
ROOT = Path(__file__).resolve().parents[2]
ANCHOR_FILENAME = "failure_closeout.json"
OUTER_SCENE_COUNT = 64
EXPECTED_SELECTOR_OUTER_OVERLAP_UNIQUE_HASHES = 795
EXPECTED_SELECTOR_OUTER_OVERLAP_ROWS = 1002
EXPECTED_SELECTOR_OUTER_OVERLAP_EFFECTIVE_PAIRS = 68
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024 * 1024
EXPECTED_V8_ARTIFACTS = frozenset((
    "inputs.json",
    "prior_rows_reauthentication_report.json",
    "prior_v7_rows.npz",
    "source_v7_report.json",
    "expansion_rows_reauthentication_report.json",
    "expansion_rows.npz",
    "source_expansion_collection_report.json",
    "development_expansion.json",
    "development_expansion_report.json",
    "fit_config.json",
    "rows.npz",
    "pairs.npz",
    "weights_audit.json",
    "program.json",
    "candidate.json",
    "fit_selector_report.json",
    "fit_selector_source_v8_report.json",
    "fit_selector_source_v8_rows.npz",
    "fit_selector_fit_only_rows.npz",
    "fit_selector_scope.json",
    "fit_selector_config_registry.json",
    "fit_selector_inner_split_audit.json",
    "fit_selector_inner_selection.json",
    "fit_selector_inner_fit_program.json",
    "fit_selector_selected_config.json",
))
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    """Return the complete local Python source closure for the closeout."""
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_dir() or path.is_symlink() or path.resolve() != path):
        raise ValueError(label + " must be a canonical directory")
    return path


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size <= 0 or path.stat().st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(label + " must contain one JSON object")
    return parsed


def _strict_json(value: str | Path, label: str, *,
                 expected_sha256: str) -> tuple[Path, bytes, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    expected = _sha(expected_sha256, label + " SHA-256")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact " + label + " bytes required")
    parsed = _strict_json_bytes(raw, label)
    if file_hash(path) != expected:
        raise RuntimeError(label + " changed during read")
    return path, raw, parsed


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: child for key, child in value.items()
                                   if key != "content_sha256"}))


def _identity(row: Any, label: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(label + " must be an object")
    seed, fingerprint = row.get("seed"), row.get("fingerprint")
    family, batch = row.get("family_id"), row.get("batch_index")
    if (type(seed) is not int or seed < 0
            or type(fingerprint) is not str or _HEX.fullmatch(fingerprint) is None
            or type(family) is not str or not family
            or type(batch) is not int or batch < 0):
        raise ValueError(label + " identity differs")
    return {
        "batch_index": batch,
        "family_id": family,
        "seed": seed,
        "fingerprint": fingerprint,
    }


def _outer_identities(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (registry.get("version")
            != "warehouse-r41-diagnostic-rcpd-v8-fresh-outer-registry.v1"
            or registry.get("status")
            != "frozen_identity_only_pending_outer_collection"
            or not _content_valid(registry)
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or registry.get("final_labels_used_for_selection") is not False
            or registry.get("formal_ready") is not False):
        raise ValueError("Authenticated v8 outer registry semantics differ")
    selected = registry.get("selected_outer_identities")
    validation = registry.get("development_validation")
    if (not isinstance(selected, list) or len(selected) != OUTER_SCENE_COUNT
            or not isinstance(validation, list)
            or len(validation) != OUTER_SCENE_COUNT):
        raise ValueError("Authenticated v8 outer identity count differs")
    identities = [_identity(row, "selected v8 outer scene") for row in selected]
    materialised = [_identity(row, "materialised v8 outer scene") for row in validation]
    if (identities != materialised
            or len({row["seed"] for row in identities}) != OUTER_SCENE_COUNT
            or len({row["fingerprint"] for row in identities}) != OUTER_SCENE_COUNT):
        raise ValueError("Authenticated v8 outer identity registry differs")
    expected_counts = {"conflict_family_01": 11, "conflict_family_02": 11,
                       "conflict_family_03": 11, "conflict_family_04": 11,
                       "conflict_family_05": 10, "conflict_family_06": 10}
    if dict(sorted(Counter(row["family_id"] for row in identities).items())) != expected_counts:
        raise ValueError("Authenticated v8 outer family quota differs")
    return identities


def _npy_member(archive: zipfile.ZipFile, name: str, *, label: str) -> np.ndarray:
    info = archive.getinfo(name)
    if (info.is_dir() or info.file_size <= 0
            or info.file_size > MAX_MEMBER_BYTES):
        raise ValueError(label + " member is not bounded")
    try:
        return np.load(BytesIO(archive.read(info)), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(label + " member is not a safe NumPy array") from error


def _ascii(array: np.ndarray, label: str) -> np.ndarray:
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError(label + " must be a one-dimensional byte-string array")
    try:
        return np.char.decode(array, "ascii").astype(str)
    except UnicodeDecodeError as error:
        raise ValueError(label + " is not ASCII") from error


def _selector_outer_overlap_audit(
    combined_rows_path: str | Path, *, combined_rows_sha256: str,
    selector_fit_rows_path: str | Path, selector_fit_rows_sha256: str,
) -> dict[str, Any]:
    """Recompute the already-discovered selector/outer overlap exactly.

    Only stored hashes plus the minimal intervention-pair membership fields are
    opened.  Raw observations and probabilities are never deserialised, and no
    per-row label or other payload is emitted by the resulting numeric audit.
    """
    combined_path = _regular(
        combined_rows_path, "authenticated v8 combined rows",
        maximum=MAX_NPZ_BYTES)
    selector_path = _regular(
        selector_fit_rows_path, "authenticated v8 selector fit rows",
        maximum=MAX_NPZ_BYTES)
    combined_expected = _sha(
        combined_rows_sha256, "authenticated v8 combined rows SHA-256")
    selector_expected = _sha(
        selector_fit_rows_sha256, "authenticated v8 selector fit rows SHA-256")
    if (file_hash(combined_path) != combined_expected
            or file_hash(selector_path) != selector_expected):
        raise ValueError("Exact selector/outer overlap inputs required")
    expected_members = {name + ".npy" for name in v8._ROW_FIELDS}
    try:
        with zipfile.ZipFile(selector_path) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError("Selector fit row archive schema differs")
            selector_hashes = _npy_member(
                archive, "observation_hashes.npy",
                label="selector observation hash")
        with zipfile.ZipFile(combined_path) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError("Combined v8 row archive schema differs")
            arrays = {
                name: _npy_member(archive, name + ".npy", label=name)
                for name in (
                    "observation_hashes", "split_validation", "kinds",
                    "anchor_ids", "branch_actions", "physical_hashes",
                    "action_indices",
                )
            }
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Selector/outer overlap inputs cannot be projected") from error
    selector_decoded = _ascii(selector_hashes, "selector observation hashes")
    count = len(arrays["observation_hashes"])
    if (count <= 0
            or arrays["observation_hashes"].shape != (count,)
            or arrays["observation_hashes"].dtype != np.dtype("S64")
            or arrays["split_validation"].shape != (count,)
            or arrays["split_validation"].dtype != np.dtype(np.bool_)
            or arrays["action_indices"].shape != (count,)
            or arrays["action_indices"].dtype.kind not in "iu"
            or np.any(arrays["action_indices"] < 0)
            or np.any(arrays["action_indices"] >= len(v8.ACTIONS))
            or selector_hashes.dtype != np.dtype("S64")):
        raise ValueError("Selector/outer overlap array schema differs")
    decoded = {
        name: _ascii(arrays[name], "combined " + name)
        for name in ("observation_hashes", "kinds", "anchor_ids",
                     "branch_actions", "physical_hashes")
    }
    if (any(values.shape != (count,) for values in decoded.values())
            or any(_HEX.fullmatch(str(value)) is None
                   for value in decoded["observation_hashes"])
            or any(_HEX.fullmatch(str(value)) is None
                   for value in selector_decoded)):
        raise ValueError("Selector/outer overlap hash values differ")
    selector_set = set(map(str, selector_decoded))
    outer = arrays["split_validation"]
    matching = outer & np.asarray([
        str(value) in selector_set for value in decoded["observation_hashes"]
    ], dtype=np.bool_)

    grouped: dict[str, dict[str, int]] = {}
    for index in np.flatnonzero(outer & (decoded["kinds"] == "intervention")):
        grouped.setdefault(str(decoded["anchor_ids"][index]), {})[
            str(decoded["branch_actions"][index])] = int(index)
    effective_pairs: list[tuple[int, int]] = []
    for values in grouped.values():
        wait = values.get("WAIT")
        if wait is None:
            continue
        for action in v8.ACTIONS[:-1]:
            changed = values.get(action)
            if (changed is not None
                    and decoded["physical_hashes"][wait]
                        != decoded["physical_hashes"][changed]
                    and arrays["action_indices"][wait]
                        != arrays["action_indices"][changed]):
                effective_pairs.append((wait, changed))
    pair_array = np.asarray(effective_pairs, dtype=np.int64).reshape(-1, 2)
    overlapping_pairs = int(np.sum(matching[pair_array].all(axis=1)))
    result = {
        "version": VERSION + ".selector-outer-overlap-audit.v1",
        "selector_fit_rows_file_sha256": selector_expected,
        "failed_outer_rows_file_sha256": combined_expected,
        "unique_observation_hashes": len(set(map(
            str, decoded["observation_hashes"][matching]))),
        "outer_rows": int(np.sum(matching)),
        "effective_intervention_pairs": overlapping_pairs,
        "pair_overlap_rule": "both effective outer pair endpoints match selector fit hashes",
        "raw_observations_read": False,
        "probabilities_read": False,
        "row_payload_published": False,
        "formal_ready": False,
    }
    if (result["unique_observation_hashes"]
            != EXPECTED_SELECTOR_OUTER_OVERLAP_UNIQUE_HASHES
            or result["outer_rows"] != EXPECTED_SELECTOR_OUTER_OVERLAP_ROWS
            or result["effective_intervention_pairs"]
                != EXPECTED_SELECTOR_OUTER_OVERLAP_EFFECTIVE_PAIRS):
        raise ValueError("Selector/failed-outer overlap finding differs")
    result["content_sha256"] = digest(result)
    if (file_hash(combined_path) != combined_expected
            or file_hash(selector_path) != selector_expected):
        raise RuntimeError("Selector/outer overlap inputs changed during audit")
    return result


def _outer_observation_hash_projection(
    rows_path: str | Path, *, expected_sha256: str,
    expected_validation_rows: int, expected_validation_scenes: int,
) -> dict[str, Any]:
    """Project only authenticated outer observation and scene hashes.

    The archive is opened directly so NumPy never sees or deserialises the
    observation, action, probability, weight, intervention, or metadata
    members.  The stored hashes were recomputed from raw public observations
    by the strict v8 reader immediately before this projection is made.
    """
    path = _regular(rows_path, "authenticated v8 combined rows",
                    maximum=MAX_NPZ_BYTES)
    expected = _sha(expected_sha256, "authenticated v8 combined rows SHA-256")
    if file_hash(path) != expected:
        raise ValueError("Exact authenticated v8 combined row bytes required")
    expected_members = {name + ".npy" for name in v8._ROW_FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError("Authenticated v8 combined row archive schema differs")
            hashes = _npy_member(
                archive, "observation_hashes.npy", label="observation hash")
            split = _npy_member(
                archive, "split_validation.npy", label="validation split")
            scenes = _npy_member(
                archive, "scene_fingerprints.npy", label="scene fingerprint")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Authenticated v8 combined rows cannot be projected") from error
    count = len(hashes) if hashes.ndim == 1 else -1
    if (count <= 0 or hashes.dtype != np.dtype("S64")
            or split.shape != (count,) or split.dtype != np.dtype(np.bool_)
            or scenes.shape != (count,) or scenes.dtype != np.dtype("S64")):
        raise ValueError("Authenticated v8 projection arrays differ")
    try:
        outer_hashes = np.char.decode(hashes[split], "ascii").astype(str)
        outer_scenes = np.char.decode(scenes[split], "ascii").astype(str)
    except UnicodeDecodeError as error:
        raise ValueError("Authenticated v8 projection is not ASCII") from error
    if (len(outer_hashes) != expected_validation_rows
            or len(set(map(str, outer_scenes))) != expected_validation_scenes
            or any(_HEX.fullmatch(str(value)) is None for value in outer_hashes)
            or any(_HEX.fullmatch(str(value)) is None for value in outer_scenes)
            or file_hash(path) != expected):
        raise ValueError("Authenticated v8 outer projection semantics differ")
    unique_hashes = sorted(set(map(str, outer_hashes)))
    unique_scenes = sorted(set(map(str, outer_scenes)))
    projection: dict[str, Any] = {
        "version": VERSION + ".outer-observation-hash-projection.v1",
        "source_rows_file_sha256": expected,
        "source_field": "authenticated observation_hashes",
        "split_field": "authenticated split_validation",
        "outer_row_count": len(outer_hashes),
        "unique_outer_observation_hash_count": len(unique_hashes),
        "outer_scene_count": len(unique_scenes),
        "outer_observation_hashes": unique_hashes,
        "outer_scene_fingerprints_sha256": digest(unique_scenes),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
    }
    projection["content_sha256"] = digest(projection)
    return projection


def _strict_artifact_directory(
    directory: str | Path, report: Mapping[str, Any], *,
    expected_report_sha256: str,
) -> tuple[Path, dict[str, str]]:
    root = _canonical_directory(directory, "Diagnostic v8 evidence directory")
    expected_names = EXPECTED_V8_ARTIFACTS | {"report.json"}
    actual_entries = list(root.iterdir())
    if ({entry.name for entry in actual_entries} != expected_names
            or len(actual_entries) != len(expected_names)
            or any(not entry.is_file() or entry.is_symlink()
                   or entry.resolve() != entry for entry in actual_entries)):
        raise ValueError("Diagnostic v8 evidence directory must contain exactly 25 artifacts and report.json")
    artifacts = report.get("evidence_artifacts")
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != EXPECTED_V8_ARTIFACTS
            or len(artifacts) != 25):
        raise ValueError("Diagnostic v8 evidence artifact registry differs")
    result: dict[str, str] = {}
    for name in sorted(EXPECTED_V8_ARTIFACTS):
        expected = _sha(artifacts.get(name), "v8 artifact " + name)
        path = _regular(root / name, "v8 artifact " + name,
                        maximum=(MAX_NPZ_BYTES if name.endswith(".npz")
                                 else MAX_JSON_BYTES))
        if file_hash(path) != expected:
            raise ValueError("Diagnostic v8 artifact hash differs: " + name)
        result[name] = expected
    if file_hash(root / "report.json") != _sha(
            expected_report_sha256, "v8 report SHA-256"):
        raise ValueError("Diagnostic v8 report changed after strict authentication")
    return root, result


def _failed_gate_names(report: Mapping[str, Any]) -> list[str]:
    candidate = report.get("candidate")
    gate = candidate.get("gate") if isinstance(candidate, Mapping) else None
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    if (not isinstance(checks, Mapping) or gate.get("passed") is not False
            or not checks or any(type(value) is not bool for value in checks.values())):
        raise ValueError("Diagnostic v8 gate evidence differs")
    failed = sorted(str(name) for name, passed in checks.items() if not passed)
    if not failed:
        raise ValueError("Failed diagnostic v8 report has no failed gate")
    return failed


def _campaign_key_inputs(report: Mapping[str, Any], *, report_sha256: str,
                         artifacts: Mapping[str, str]) -> dict[str, str]:
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Diagnostic v8 bindings are missing")
    return {
        "scheme": VERSION + ".campaign-key.v1",
        "source_report_sha256": report_sha256,
        "actor_sha256": _sha(bindings.get("actor_file_sha256"), "v8 Actor"),
        "manifest_sha256": _sha(bindings.get("manifest_file_sha256"), "v8 manifest"),
        "outer_registry_sha256": artifacts["development_expansion.json"],
        "outer_registry_report_sha256": artifacts["development_expansion_report.json"],
        "selector_report_sha256": artifacts["fit_selector_report.json"],
    }


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source": "strictly authenticated failed warehouse-r41-diagnostic-rcpd.v8 outer evaluation",
        "evidence_artifact_count": 25,
        "permanent_anchor": "campaign-key directory and O_EXCL failure_closeout.json",
        "failed_outer_reclassified_as_development": True,
        "outer_claim_reuse_permitted": False,
        "selector_outer_overlap_recomputed": {
            "unique_observation_hashes": EXPECTED_SELECTOR_OUTER_OVERLAP_UNIQUE_HASHES,
            "outer_rows": EXPECTED_SELECTOR_OUTER_OVERLAP_ROWS,
            "effective_intervention_pairs": (
                EXPECTED_SELECTOR_OUTER_OVERLAP_EFFECTIVE_PAIRS),
        },
        "outer_observation_projection": "unique authenticated SHA-256 values only",
        "raw_observations_published": False,
        "action_labels_published": False,
        "probabilities_published": False,
        "protected_final_access": False,
        "formal_ready": False,
    }


def create_closeout(
    *, candidate_output: str | Path, expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    expected_prior_rows_report_sha256: str,
    previous_development_path: str | Path,
    expected_expansion_rows_report_sha256: str,
    expected_selector_report_sha256: str,
) -> tuple[dict[str, Any], str]:
    expected_report_sha256 = _sha(expected_report_sha256, "v8 report")
    sources = producer_sources()
    report = v8.read_saved_report(
        candidate_output, expected_report_sha256=expected_report_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expansion_registry_path=expansion_registry_path,
        expected_expansion_registry_sha256=expected_expansion_registry_sha256,
        expansion_report_path=expansion_report_path,
        expected_expansion_report_sha256=expected_expansion_report_sha256,
        expected_prior_rows_report_sha256=expected_prior_rows_report_sha256,
        previous_development_path=previous_development_path,
        expected_expansion_rows_report_sha256=expected_expansion_rows_report_sha256,
        expected_selector_report_sha256=expected_selector_report_sha256,
        require_passed=False, refit=False,
    )
    directory, artifacts = _strict_artifact_directory(
        candidate_output, report, expected_report_sha256=expected_report_sha256)
    if (report.get("version") != v8.VERSION
            or report.get("status") != v8.STATUS_FAILED
            or report.get("explanation_eligible") is not False
            or report.get("formal_ready") is not False
            or not isinstance(report.get("execution"), Mapping)
            or report["execution"].get("final_rows_accessed") is not False
            or report["execution"].get("final_labels_accessed") is not False):
        raise ValueError("Only the authenticated failed v8 outer result can be closed")
    failed_gates = _failed_gate_names(report)
    _, _, registry = _strict_json(
        directory / "development_expansion.json", "embedded v8 outer registry",
        expected_sha256=artifacts["development_expansion.json"])
    identities = _outer_identities(registry)
    if (file_hash(Path(expansion_registry_path).expanduser().absolute())
            != artifacts["development_expansion.json"]
            or file_hash(Path(expansion_report_path).expanduser().absolute())
            != artifacts["development_expansion_report.json"]):
        raise ValueError("External v8 outer registry pair differs from embedded evidence")
    row_accounting = report.get("row_accounting")
    if not isinstance(row_accounting, Mapping):
        raise ValueError("Diagnostic v8 row accounting is missing")
    validation_rows = row_accounting.get("retained_validation_rows")
    validation_scenes = report.get("validation_scene_count")
    if (type(validation_rows) is not int or validation_rows <= 0
            or type(validation_scenes) is not int
            or validation_scenes != OUTER_SCENE_COUNT):
        raise ValueError("Diagnostic v8 validation-row accounting differs")
    projection = _outer_observation_hash_projection(
        directory / "rows.npz", expected_sha256=artifacts["rows.npz"],
        expected_validation_rows=validation_rows,
        expected_validation_scenes=validation_scenes,
    )
    # The boolean-and expression above must not admit bools as counts.
    if projection["outer_row_count"] != row_accounting.get("retained_validation_rows"):
        raise ValueError("Diagnostic v8 validation-row accounting differs")
    overlap_audit = _selector_outer_overlap_audit(
        directory / "rows.npz",
        combined_rows_sha256=artifacts["rows.npz"],
        selector_fit_rows_path=directory / "fit_selector_fit_only_rows.npz",
        selector_fit_rows_sha256=artifacts["fit_selector_fit_only_rows.npz"],
    )
    candidate = report.get("candidate")
    metrics = candidate.get("metrics") if isinstance(candidate, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValueError("Diagnostic v8 metric evidence is missing")
    campaign_key_inputs = _campaign_key_inputs(
        report, report_sha256=expected_report_sha256, artifacts=artifacts)
    campaign_key = digest(campaign_key_inputs)
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "campaign_key": campaign_key,
        "campaign_key_inputs": campaign_key_inputs,
        "contract": contract(),
        "source": {
            "v8_report_sha256": expected_report_sha256,
            "v8_candidate_sha256": artifacts["candidate.json"],
            "v8_combined_rows_sha256": artifacts["rows.npz"],
            "v8_program_sha256": artifacts["program.json"],
            "v8_outer_registry_sha256": artifacts["development_expansion.json"],
            "v8_outer_registry_report_sha256": artifacts["development_expansion_report.json"],
            "v8_selector_report_sha256": artifacts["fit_selector_report.json"],
            "evidence_artifacts": dict(sorted(artifacts.items())),
            "evidence_artifact_count": len(artifacts),
            "evidence_artifacts_sha256": digest(dict(sorted(artifacts.items()))),
        },
        "failure": {
            "source_status": report["status"],
            "candidate_selection_status": candidate.get("selection_status"),
            "failed_gate_names": failed_gates,
            "gate": deepcopy(candidate.get("gate")),
            "metrics": deepcopy(metrics),
            "selector_outer_overlap": overlap_audit,
            "interpretation": (
                "The failed outer is permanently development-exposed and "
                "cannot support any later outer claim."
            ),
        },
        "consumed_outer": {
            "scene_count": len(identities),
            "identities": identities,
            "identities_sha256": digest(identities),
            "observation_hash_projection": projection,
        },
        "disposition": {
            "eligible_for_development_use": True,
            "eligible_for_outer_claim": False,
            "outer_identity_reuse_permitted": False,
            "outer_observation_hashes_must_win_over_future_fit_rows": True,
            "formal_ready": False,
        },
        "information_boundary": {
            "source_outer_labels_already_exposed": True,
            "overlap_audit_fields_read": [
                "observation_hashes", "split_validation", "kinds",
                "anchor_ids", "branch_actions", "physical_hashes",
                "action_indices",
            ],
            "raw_observations_published": False,
            "action_labels_published": False,
            "probabilities_published": False,
            "selector_may_read_projection_hashes_only": True,
            "protected_final_access": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    if producer_sources() != sources:
        raise RuntimeError("Closeout source closure changed during authentication")
    if (file_hash(directory / "report.json") != expected_report_sha256
            or any(file_hash(directory / name) != expected
                   for name, expected in artifacts.items())):
        raise RuntimeError("Diagnostic v8 evidence changed before closeout")
    return receipt, campaign_key


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(*, permanent_registry: str | Path, output: str | Path,
          **kwargs: Any) -> dict[str, Any]:
    receipt, campaign_key = create_closeout(**kwargs)
    permanent = _canonical_directory(permanent_registry, "Permanent closeout registry")
    destination = Path(output).expanduser().absolute()
    parent = _canonical_directory(destination.parent, "Closeout output parent")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Closeout output destination already exists")
    raw = (canonical(receipt) + "\n").encode("utf-8")

    campaign_directory = permanent / campaign_key
    try:
        os.mkdir(campaign_directory, 0o700)
    except FileExistsError:
        raise FileExistsError(
            "This failed outer campaign already has a permanent closeout anchor") from None
    if (campaign_directory.is_symlink()
            or campaign_directory.resolve() != campaign_directory):
        raise RuntimeError("Permanent closeout campaign directory is unsafe")
    try:
        _write_exclusive(campaign_directory / ANCHOR_FILENAME, raw)
        _fsync_directory(campaign_directory)
        _fsync_directory(permanent)
    except BaseException:
        # Never remove a successfully-created campaign directory.  Its presence
        # permanently prevents this campaign from being silently replayed.
        raise

    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / ANCHOR_FILENAME, raw)
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        temporary = None
        _fsync_directory(parent)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return receipt


def _validate_saved_receipt(value: Mapping[str, Any]) -> None:
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or not _content_valid(value) or value.get("formal_ready") is not False
            or value.get("producer_sources_sha256")
                != digest(value.get("producer_sources"))):
        raise ValueError("Saved failed-outer closeout semantics differ")
    source = value.get("source")
    disposition = value.get("disposition")
    boundary = value.get("information_boundary")
    consumed = value.get("consumed_outer")
    failure = value.get("failure")
    gate = failure.get("gate") if isinstance(failure, Mapping) else None
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    overlap = (failure.get("selector_outer_overlap")
               if isinstance(failure, Mapping) else None)
    failed_from_checks = (sorted(
        str(name) for name, passed in checks.items() if passed is False)
        if isinstance(checks, Mapping)
        and all(type(passed) is bool for passed in checks.values()) else None)
    if (not isinstance(source, Mapping)
            or source.get("evidence_artifact_count") != 25
            or not isinstance(source.get("evidence_artifacts"), Mapping)
            or set(source["evidence_artifacts"]) != EXPECTED_V8_ARTIFACTS
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in source["evidence_artifacts"].values())
            or source.get("evidence_artifacts_sha256")
                != digest(dict(sorted(source["evidence_artifacts"].items())))
            or source.get("v8_candidate_sha256")
                != source["evidence_artifacts"].get("candidate.json")
            or source.get("v8_combined_rows_sha256")
                != source["evidence_artifacts"].get("rows.npz")
            or source.get("v8_program_sha256")
                != source["evidence_artifacts"].get("program.json")
            or source.get("v8_outer_registry_sha256")
                != source["evidence_artifacts"].get("development_expansion.json")
            or source.get("v8_outer_registry_report_sha256")
                != source["evidence_artifacts"].get("development_expansion_report.json")
            or source.get("v8_selector_report_sha256")
                != source["evidence_artifacts"].get("fit_selector_report.json")
            or not isinstance(failure, Mapping)
            or failure.get("source_status") != v8.STATUS_FAILED
            or not isinstance(failure.get("failed_gate_names"), list)
            or not failure["failed_gate_names"]
            or failure["failed_gate_names"]
                != sorted(set(failure["failed_gate_names"]))
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or failed_from_checks is None
            or failure["failed_gate_names"] != failed_from_checks
            or not isinstance(overlap, Mapping)
            or not _content_valid(overlap)
            or overlap.get("unique_observation_hashes")
                != EXPECTED_SELECTOR_OUTER_OVERLAP_UNIQUE_HASHES
            or overlap.get("outer_rows")
                != EXPECTED_SELECTOR_OUTER_OVERLAP_ROWS
            or overlap.get("effective_intervention_pairs")
                != EXPECTED_SELECTOR_OUTER_OVERLAP_EFFECTIVE_PAIRS
            or overlap.get("selector_fit_rows_file_sha256")
                != source["evidence_artifacts"].get(
                    "fit_selector_fit_only_rows.npz")
            or overlap.get("failed_outer_rows_file_sha256")
                != source.get("v8_combined_rows_sha256")
            or overlap.get("raw_observations_read") is not False
            or overlap.get("probabilities_read") is not False
            or overlap.get("row_payload_published") is not False
            or not isinstance(disposition, Mapping)
            or disposition.get("eligible_for_development_use") is not True
            or disposition.get("eligible_for_outer_claim") is not False
            or disposition.get("outer_identity_reuse_permitted") is not False
            or disposition.get("outer_observation_hashes_must_win_over_future_fit_rows") is not True
            or disposition.get("formal_ready") is not False
            or not isinstance(boundary, Mapping)
            or boundary.get("raw_observations_published") is not False
            or boundary.get("action_labels_published") is not False
            or boundary.get("probabilities_published") is not False
            or boundary.get("protected_final_access") is not False
            or not isinstance(consumed, Mapping)
            or consumed.get("scene_count") != OUTER_SCENE_COUNT
            or not isinstance(consumed.get("identities"), list)
            or consumed.get("identities_sha256") != digest(consumed["identities"])):
        raise ValueError("Saved failed-outer closeout disposition differs")
    identities = [_identity(row, "saved consumed outer scene")
                  for row in consumed["identities"]]
    if (len({row["seed"] for row in identities}) != OUTER_SCENE_COUNT
            or len({row["fingerprint"] for row in identities})
                != OUTER_SCENE_COUNT):
        raise ValueError("Saved consumed outer identities differ")
    projection = consumed.get("observation_hash_projection")
    if (not isinstance(projection, Mapping) or not _content_valid(projection)
            or projection.get("raw_observations_included") is not False
            or projection.get("actions_included") is not False
            or projection.get("probabilities_included") is not False
            or projection.get("labels_included") is not False
            or projection.get("source_rows_file_sha256")
                != source.get("v8_combined_rows_sha256")
            or projection.get("outer_scene_count") != OUTER_SCENE_COUNT
            or type(projection.get("outer_row_count")) is not int
            or projection.get("outer_row_count") <= 0
            or not isinstance(projection.get("outer_observation_hashes"), list)
            or projection.get("unique_outer_observation_hash_count")
                != len(projection["outer_observation_hashes"])
            or projection["outer_observation_hashes"]
                != sorted(set(projection["outer_observation_hashes"]))
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in projection["outer_observation_hashes"])):
        raise ValueError("Saved outer observation-hash projection differs")
    source_key = value.get("campaign_key_inputs")
    if (not isinstance(source_key, Mapping)
            or set(source_key) != {
                "scheme", "source_report_sha256", "actor_sha256",
                "manifest_sha256", "outer_registry_sha256",
                "outer_registry_report_sha256", "selector_report_sha256",
            }
            or source_key.get("scheme") != VERSION + ".campaign-key.v1"
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for key, item in source_key.items() if key != "scheme")
            or source_key.get("source_report_sha256") != source.get("v8_report_sha256")
            or source_key.get("outer_registry_sha256")
                != source["evidence_artifacts"].get("development_expansion.json")
            or source_key.get("outer_registry_report_sha256")
                != source["evidence_artifacts"].get("development_expansion_report.json")
            or source_key.get("selector_report_sha256")
                != source["evidence_artifacts"].get("fit_selector_report.json")):
        raise ValueError("Saved failed-outer campaign key inputs differ")
    expected_campaign_key = digest(dict(source_key))
    if value.get("campaign_key") != expected_campaign_key:
        raise ValueError("Saved failed-outer campaign key differs")


def read_saved_closeout(
    path: str | Path, *, expected_closeout_sha256: str,
    permanent_registry: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    closeout_path, raw, receipt = _strict_json(
        path, "failed outer closeout",
        expected_sha256=expected_closeout_sha256)
    if closeout_path.name != ANCHOR_FILENAME:
        raise ValueError("Canonical failure_closeout.json filename required")
    _validate_saved_receipt(receipt)
    if receipt.get("producer_sources") != sources:
        raise ValueError("Failed-outer closeout source closure differs")
    permanent = _canonical_directory(permanent_registry, "Permanent closeout registry")
    campaign_key = _sha(receipt.get("campaign_key"), "failed outer campaign key")
    campaign_directory = _canonical_directory(
        permanent / campaign_key, "Permanent closeout campaign directory")
    entries = list(campaign_directory.iterdir())
    if (len(entries) != 1 or entries[0].name != ANCHOR_FILENAME
            or not entries[0].is_file() or entries[0].is_symlink()
            or entries[0].read_bytes() != raw
            or file_hash(entries[0]) != _sha(
                expected_closeout_sha256, "failed outer closeout SHA-256")):
        raise ValueError("Permanent failed-outer closeout anchor differs")
    if (producer_sources() != sources
            or closeout_path.read_bytes() != raw):
        raise RuntimeError("Failed-outer closeout changed during read")
    return deepcopy(receipt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--expansion-registry", required=True)
    parser.add_argument("--expected-expansion-registry-sha256", required=True)
    parser.add_argument("--expansion-report", required=True)
    parser.add_argument("--expected-expansion-report-sha256", required=True)
    parser.add_argument("--expected-prior-rows-report-sha256", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--expected-expansion-rows-report-sha256", required=True)
    parser.add_argument("--expected-selector-report-sha256", required=True)
    parser.add_argument("--permanent-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build(
        candidate_output=args.candidate_output,
        expected_report_sha256=args.expected_report_sha256,
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        expansion_registry_path=args.expansion_registry,
        expected_expansion_registry_sha256=args.expected_expansion_registry_sha256,
        expansion_report_path=args.expansion_report,
        expected_expansion_report_sha256=args.expected_expansion_report_sha256,
        expected_prior_rows_report_sha256=args.expected_prior_rows_report_sha256,
        previous_development_path=args.previous_development,
        expected_expansion_rows_report_sha256=args.expected_expansion_rows_report_sha256,
        expected_selector_report_sha256=args.expected_selector_report_sha256,
        permanent_registry=args.permanent_registry, output=args.output,
    )
    print(canonical({
        "status": receipt["status"],
        "campaign_key": receipt["campaign_key"],
        "closeout": str(Path(args.output).resolve() / ANCHOR_FILENAME),
        "failed_gate_names": receipt["failure"]["failed_gate_names"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "ANCHOR_FILENAME", "EXPECTED_V8_ARTIFACTS",
    "contract", "producer_sources", "create_closeout", "build",
    "read_saved_closeout", "main", "_outer_observation_hash_projection",
]
