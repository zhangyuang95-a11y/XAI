"""Freeze the next fresh development outer registry after the failed v8 outer.

The failed v8 outer is first authenticated through its permanent v9 closeout.
Every identity exposed by that campaign and every earlier registered source is
excluded from the fixed 2,160-scene candidate population.  Selection is made
only from seed, scene fingerprint, family, and exclusion membership using a
new fixed salt.

The only v8 row payload opened here is ``scene_fingerprints.npy``.  A separate
artifact copies the closeout's one-way outer observation-hash projection so a
future selector can apply validation-wins before it reads development labels
or probabilities.  No raw observation, action, probability, program output,
metric, Actor inference, or protected test source is available to selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from backend.training import warehouse_r41_diagnostic_conflict_scenarios as scenes_api
from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as source_expansion_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_outer_failure_closeout_v9 as closeout_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as old_outer_api
from backend.training import warehouse_r41_diagnostic_retired_identity_projection_v8 as retired_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticConflictWarehouseEnv
from env.warehouse_native.r41_diagnostic_conflict import (
    conflict_family_id,
    diagnostic_scene_fingerprint,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v9-fresh-outer-registry.v1"
STATUS = "frozen_identity_only_pending_outer_collection"
ROOT = Path(__file__).resolve().parents[2]
SELECTION_SALT = "warehouse-r41-v9-fresh-development-outer-20260913-v1"
FAMILY_IDS = tuple(scenes_api.FAMILY_IDS)
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
FRESH_OUTER_SCENE_COUNT = 64
SOURCE_ROW_SCENE_COUNT = 448
FIXED_CANDIDATE_SCENE_COUNT = 2160
EXPECTED_REMAINING_SCENE_COUNT = 1695
EXPECTED_REMAINING_FAMILY_COUNTS = {
    "conflict_family_01": 279,
    "conflict_family_02": 280,
    "conflict_family_03": 282,
    "conflict_family_04": 279,
    "conflict_family_05": 287,
    "conflict_family_06": 288,
}
ORIGINAL_FIT_SCENE_COUNT = 128
ORIGINAL_OUTER_SCENE_COUNT = 64
CURRENT_FIT_SCENE_COUNT = 128
CURRENT_OUTER_SCENE_COUNT = 64
ORIGINAL_TRACE_IDENTITY_COUNT = 209
FORMAL_SCENE_COUNT = 6
PREVIOUS_DEVELOPMENT_SCENE_COUNT = 64
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024

EXPECTED_MANIFEST_SHA256 = manifest_binding.EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_VALIDATION_SHA256 = manifest_binding.EXPECTED_VALIDATION_SHA256
EXPECTED_FAILED_ROWS_SHA256 = (
    "cf631ef9ddd16f333d0cd43121650633eb58a337e473bb8279d9ab12ace3024f"
)
EXPECTED_ORIGINAL_EXPANSION_SHA256 = (
    "a687fd3fd4b145ed432af77f3ce26726d4e328df4875e4d69fc3ed351ad98748"
)
EXPECTED_CURRENT_OUTER_REGISTRY_SHA256 = (
    "a598f78b0b8054bfe9e3cb0befa9c1007f0bcf6e0e599ac2565523d1f0e62332"
)
EXPECTED_CURRENT_OUTER_REPORT_SHA256 = (
    "bf21daa21e3799c701c641a34b72130d95bdb64294a908d78d901938623e3115"
)
EXPECTED_FORMAL_SELECTION_SHA256 = (
    "30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8"
)
EXPECTED_PREVIOUS_DEVELOPMENT_SHA256 = (
    "8931b74940f1c41940d9fbb73a52f940f527b8f9c67dfd6b94a4a4d1afd977fe"
)
EXPECTED_RETIRED_PROJECTION_SHA256 = (
    "cdd17b1a8d46b54dfeec70f74fb3acb43dc1b575c54f58cf9be982ad89d5f111"
)
EXPECTED_RETIRED_REPORT_SHA256 = (
    "4188b293f5c0e02cc80ece7fa1c0841134369c740fffea2aecef90653e077db5"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size <= 0 or path.stat().st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json(value: str | Path, label: str, *,
                 expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    expected = _sha(expected_sha256, label + " SHA-256")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact " + label + " bytes required")

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
    if not isinstance(parsed, dict) or file_hash(path) != expected:
        raise ValueError(label + " changed during read")
    return path, parsed


def _authenticate_opaque(value: str | Path, label: str, *,
                         expected_sha256: str) -> Path:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    if file_hash(path) != _sha(expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: child for key, child in value.items()
                                   if key != "content_sha256"}))


def _identity(row: Any, label: str) -> tuple[int, str]:
    if not isinstance(row, Mapping):
        raise ValueError(label + " must be an object")
    seed, fingerprint = row.get("seed"), row.get("fingerprint")
    if (type(seed) is not int or seed < 0 or type(fingerprint) is not str
            or _HEX.fullmatch(fingerprint) is None):
        raise ValueError(label + " identity differs")
    return seed, fingerprint


def _unique_identities(rows: Any, *, expected_count: int,
                       label: str) -> tuple[set[int], set[str]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(label + " scene count differs")
    seeds: set[int] = set()
    fingerprints: set[str] = set()
    for row in rows:
        seed, fingerprint = _identity(row, label)
        if seed in seeds or fingerprint in fingerprints:
            raise ValueError(label + " contains a duplicate identity")
        seeds.add(seed)
        fingerprints.add(fingerprint)
    return seeds, fingerprints


def _scene_fingerprints_only(value: str | Path, *,
                             expected_sha256: str) -> tuple[Path, set[str]]:
    """Open only the scene-fingerprint member of authenticated v8 rows."""
    path = _regular(value, "failed v8 combined rows", maximum=MAX_NPZ_BYTES)
    expected = _sha(expected_sha256, "failed v8 combined rows SHA-256")
    if file_hash(path) != expected:
        raise ValueError("Exact failed v8 combined row bytes required")
    expected_members = {name + ".npy" for name in closeout_api.v8._ROW_FIELDS}
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError("Failed v8 combined row archive schema differs")
            member = archive.getinfo("scene_fingerprints.npy")
            if (member.is_dir() or member.file_size <= 0
                    or member.file_size > MAX_MEMBER_BYTES):
                raise ValueError("Failed v8 scene identity member is invalid")
            array = np.load(BytesIO(archive.read(member)), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Failed v8 scene identities cannot be read") from error
    if (array.ndim != 1 or array.dtype != np.dtype("S64")
            or file_hash(path) != expected):
        raise ValueError("Failed v8 scene identity array differs")
    try:
        decoded = np.char.decode(array, "ascii").astype(str)
    except UnicodeDecodeError as error:
        raise ValueError("Failed v8 scene identity is not ASCII") from error
    fingerprints = set(map(str, decoded))
    if (len(fingerprints) != SOURCE_ROW_SCENE_COUNT
            or any(_HEX.fullmatch(item) is None for item in fingerprints)):
        raise ValueError("Failed v8 scene population differs")
    return path, fingerprints


def _source_expansion(value: Mapping[str, Any]) -> tuple[
        set[int], set[str], set[int], set[str]]:
    if (value.get("version") != source_expansion_api.VERSION
            or value.get("status") != source_expansion_api.STATUS
            or not _content_valid(value)
            or value.get("program_access") is not False
            or value.get("program_predictions_access") is not False
            or value.get("final_audit_rows_access") is not False
            or value.get("final_labels_used_for_selection") is not False):
        raise ValueError("Exact original development expansion required")
    fit_seeds, fit_fingerprints = _unique_identities(
        value.get("fit_supplement"), expected_count=ORIGINAL_FIT_SCENE_COUNT,
        label="original fit supplement")
    outer_seeds, outer_fingerprints = _unique_identities(
        value.get("development_validation"),
        expected_count=ORIGINAL_OUTER_SCENE_COUNT,
        label="original exposed outer")
    trace = value.get("selection_trace")
    if not isinstance(trace, Mapping) or set(trace) != {
            "fit_supplement", "development_validation"}:
        raise ValueError("Original expansion selection trace differs")
    trace_seeds: set[int] = set()
    trace_fingerprints: set[str] = set()
    for split in ("fit_supplement", "development_validation"):
        rows = trace.get(split)
        if not isinstance(rows, list):
            raise ValueError("Original expansion selection trace is missing")
        for row in rows:
            seed, fingerprint = _identity(row, "original expansion trace")
            trace_seeds.add(seed)
            trace_fingerprints.add(fingerprint)
    if (len(trace_fingerprints) != ORIGINAL_TRACE_IDENTITY_COUNT
            or not fit_fingerprints.issubset(trace_fingerprints)
            or not outer_fingerprints.issubset(trace_fingerprints)
            or fit_seeds & outer_seeds or fit_fingerprints & outer_fingerprints):
        raise ValueError("Original expansion trace population differs")
    return trace_seeds, trace_fingerprints, outer_seeds, outer_fingerprints


def _current_outer(value: Mapping[str, Any], report: Mapping[str, Any], *,
                   registry_sha256: str) -> tuple[
                       list[dict[str, Any]], set[int], set[str], set[str]]:
    if (value.get("version") != old_outer_api.VERSION
            or value.get("status") != old_outer_api.STATUS
            or not _content_valid(value)
            or value.get("program_access") is not False
            or value.get("program_predictions_access") is not False
            or value.get("final_audit_rows_access") is not False
            or value.get("final_labels_used_for_selection") is not False
            or value.get("formal_ready") is not False):
        raise ValueError("Exact consumed v8 outer registry required")
    fit_seeds, fit_fingerprints = _unique_identities(
        value.get("fit_supplement"), expected_count=CURRENT_FIT_SCENE_COUNT,
        label="consumed v8 fit supplement")
    outer_seeds, outer_fingerprints = _unique_identities(
        value.get("development_validation"), expected_count=CURRENT_OUTER_SCENE_COUNT,
        label="consumed v8 outer")
    selected = value.get("selected_outer_identities")
    selected_seeds, selected_fingerprints = _unique_identities(
        selected, expected_count=CURRENT_OUTER_SCENE_COUNT,
        label="consumed v8 selected outer")
    if (selected_seeds != outer_seeds or selected_fingerprints != outer_fingerprints
            or fit_seeds & outer_seeds or fit_fingerprints & outer_fingerprints):
        raise ValueError("Consumed v8 selected outer differs from materialised outer")
    old_outer = value.get("previously_exposed_validation_scene_fingerprints")
    if (not isinstance(old_outer, list)
            or len(old_outer) != ORIGINAL_OUTER_SCENE_COUNT
            or old_outer != sorted(set(old_outer))
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in old_outer)):
        raise ValueError("Consumed v8 prior outer fingerprint registry differs")
    if (report.get("version") != old_outer_api.REPORT_VERSION
            or report.get("status") != old_outer_api.STATUS
            or not _content_valid(report)
            or report.get("registry_file_sha256") != registry_sha256
            or report.get("registry_content_sha256") != value.get("content_sha256")
            or report.get("formal_ready") is not False
            or report.get("selection", {}).get("selected_identity_sha256")
                != digest([{key: row[key] for key in (
                    "batch_index", "family_id", "seed", "fingerprint")}
                           for row in selected])):
        raise ValueError("Consumed v8 outer report differs")
    identities = [{key: row[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")} for row in selected]
    return identities, fit_seeds | outer_seeds, fit_fingerprints | outer_fingerprints, set(old_outer)


def _formal_identities(value: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    if (value.get("version")
            != "warehouse-r41-diagnostic-conflict-dynamic-selection.v3"
            or value.get("release_eligible") is not True
            or value.get("source_manifest_file_sha256") != EXPECTED_MANIFEST_SHA256
            or value.get("zero_action_overrides") is not True):
        raise ValueError("Exact formal X/Y selection required")
    return _unique_identities(
        [*value.get("X", []), *value.get("Y", [])],
        expected_count=FORMAL_SCENE_COUNT, label="formal X/Y")


def _previous_identities(value: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    if (value.get("version")
            != "warehouse-r41-diagnostic-development-supplement.v1"
            or value.get("status") != "passed"
            or value.get("program_access") is not False
            or value.get("final_audit_rows_access") is not False
            or not _content_valid(value)):
        raise ValueError("Exact previous development supplement required")
    return _unique_identities(
        value.get("scenes"), expected_count=PREVIOUS_DEVELOPMENT_SCENE_COUNT,
        label="previous development supplement")


def _candidate_identity_population() -> list[dict[str, Any]]:
    batches = retired_api._regenerate_development_candidate_identity_batches()
    per_batch = scenes_api.PER_FAMILY_PER_BATCH * len(FAMILY_IDS)
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    seen_fingerprints: set[str] = set()
    for batch_index, batch in enumerate(batches):
        if len(batch) != per_batch:
            raise ValueError("Fixed candidate identity batch size differs")
        for index, identity in enumerate(batch):
            seed, fingerprint = _identity(identity, "fixed candidate")
            family_index, family_offset = divmod(
                index, scenes_api.PER_FAMILY_PER_BATCH)
            if seed in seen_seeds or fingerprint in seen_fingerprints:
                raise ValueError("Fixed candidate population overlaps")
            seen_seeds.add(seed)
            seen_fingerprints.add(fingerprint)
            rows.append({
                "batch_index": batch_index,
                "family_offset": family_offset,
                "family_id": FAMILY_IDS[family_index],
                "seed": seed,
                "fingerprint": fingerprint,
            })
    if len(rows) != FIXED_CANDIDATE_SCENE_COUNT:
        raise ValueError("Fixed candidate identity population differs")
    return rows


def _rank(row: Mapping[str, Any]) -> str:
    return digest({
        "salt": SELECTION_SALT,
        "family_id": row["family_id"],
        "fingerprint": row["fingerprint"],
    })


def _remaining_and_selected(
    candidates: Sequence[Mapping[str, Any]], *, excluded_seeds: set[int],
    excluded_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if len(candidates) != FIXED_CANDIDATE_SCENE_COUNT:
        raise ValueError("Fixed candidate population count differs")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        seed, fingerprint = _identity(raw, "fixed candidate population")
        family = raw.get("family_id")
        batch_index, family_offset = raw.get("batch_index"), raw.get("family_offset")
        if (family not in FAMILY_IDS or type(batch_index) is not int
                or type(family_offset) is not int):
            raise ValueError("Fixed candidate family identity differs")
        if seed not in excluded_seeds and fingerprint not in excluded_fingerprints:
            by_family[str(family)].append(deepcopy(dict(raw)))
    remaining = [row for family in FAMILY_IDS for row in by_family[family]]
    counts = dict(sorted(Counter(row["family_id"] for row in remaining).items()))
    if (len(remaining) != EXPECTED_REMAINING_SCENE_COUNT
            or counts != EXPECTED_REMAINING_FAMILY_COUNTS):
        raise ValueError("Fixed candidate remaining population differs")
    selected: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ordered = sorted(by_family[family], key=lambda row: (
            _rank(row), row["fingerprint"]))
        quota = FAMILY_QUOTAS[family]
        if len(ordered) < quota:
            raise ValueError("Insufficient fresh candidate identities for " + family)
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: (FAMILY_IDS.index(row["family_id"]), _rank(row)))
    if (len(selected) != FRESH_OUTER_SCENE_COUNT
            or Counter(row["family_id"] for row in selected) != Counter(FAMILY_QUOTAS)):
        raise RuntimeError("Fresh v9 outer family quota differs")
    return remaining, selected, counts


def _materialize_scene(identity: Mapping[str, Any], index: int) -> dict[str, Any]:
    seed, expected_fingerprint = _identity(identity, "selected fresh v9 outer")
    batch_index, family = identity.get("batch_index"), identity.get("family_id")
    if type(batch_index) is not int or family not in FAMILY_IDS:
        raise ValueError("Selected fresh v9 outer batch identity differs")
    env = R41DiagnosticConflictWarehouseEnv()
    retired_api._identity_only_reset(env, seed=seed)
    reset_family = conflict_family_id(env.state.tasks)
    if reset_family != family:
        raise ValueError("Selected fresh v9 outer family differs at reset")
    scenes_api._candidate_start_state(
        env, seed=seed, batch_index=batch_index, family_id=reset_family)
    if diagnostic_scene_fingerprint(env) != expected_fingerprint:
        raise ValueError("Selected fresh v9 outer fingerprint differs")
    scene = scenes_api._scene_from_environment(
        env, scene_id=f"diagnostic_v9_fresh_outer_{index:04d}",
        split="development_outer", seed=seed, batch_index=batch_index)
    if (scene.get("fingerprint") != expected_fingerprint
            or scene.get("family_id") != family):
        raise ValueError("Materialised fresh v9 outer identity differs")
    return scene


def _projection_artifact(closeout: Mapping[str, Any]) -> dict[str, Any]:
    source = closeout["consumed_outer"]["observation_hash_projection"]
    values = source.get("outer_observation_hashes")
    if not isinstance(values, list):
        raise ValueError("Failed closeout observation-hash projection is missing")
    projection: dict[str, Any] = {
        "version": VERSION + ".validation-wins-exclusion.v1",
        "source_closeout_content_sha256": closeout["content_sha256"],
        "source_projection_content_sha256": source["content_sha256"],
        "outer_observation_hashes": deepcopy(values),
        "unique_outer_observation_hash_count": len(values),
        "selector_rule": (
            "remove development-fit rows with a matching hash before reading "
            "their actions, probabilities, or raw observations"
        ),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
        "formal_ready": False,
    }
    projection["content_sha256"] = digest(projection)
    return projection


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "fresh unfit v9 development outer registry",
        "selection_salt": SELECTION_SALT,
        "fixed_candidate_population": FIXED_CANDIDATE_SCENE_COUNT,
        "required_remaining_population": EXPECTED_REMAINING_SCENE_COUNT,
        "required_remaining_family_counts": deepcopy(EXPECTED_REMAINING_FAMILY_COUNTS),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "selection_inputs": ["seed", "scene fingerprint", "family", "exclusion membership"],
        "all_exposed_identity_sources_excluded": True,
        "failed_outer_observation_hashes": "one-way projection copied for future validation-wins",
        "selection_uses_outer_observation_hashes": False,
        "selection_uses_actor": False,
        "selection_uses_observations": False,
        "selection_uses_actions_probabilities_program_or_metrics": False,
        "protected_final_access": False,
        "formal_ready": False,
    }


def create_registry(
    *, manifest_path: str | Path, failed_rows_path: str | Path,
    original_expansion_path: str | Path,
    current_outer_registry_path: str | Path,
    current_outer_report_path: str | Path,
    failure_closeout_path: str | Path,
    expected_failure_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
    formal_selection_path: str | Path,
    previous_development_path: str | Path,
    retired_projection_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = producer_sources()
    manifest = _authenticate_opaque(
        manifest_path, "frozen diagnostic manifest",
        expected_sha256=EXPECTED_MANIFEST_SHA256)
    manifest_validation = _authenticate_opaque(
        manifest.parent / "validation.json", "manifest validation receipt",
        expected_sha256=EXPECTED_MANIFEST_VALIDATION_SHA256)
    rows_file, row_fingerprints = _scene_fingerprints_only(
        failed_rows_path, expected_sha256=EXPECTED_FAILED_ROWS_SHA256)
    original_file, original = _strict_json(
        original_expansion_path, "original development expansion",
        expected_sha256=EXPECTED_ORIGINAL_EXPANSION_SHA256)
    trace_seeds, trace_fingerprints, old_outer_seeds, old_outer_fingerprints = (
        _source_expansion(original))
    current_file, current = _strict_json(
        current_outer_registry_path, "consumed v8 outer registry",
        expected_sha256=EXPECTED_CURRENT_OUTER_REGISTRY_SHA256)
    current_report_file, current_report = _strict_json(
        current_outer_report_path, "consumed v8 outer report",
        expected_sha256=EXPECTED_CURRENT_OUTER_REPORT_SHA256)
    current_identities, current_seeds, current_fingerprints, registered_old_outer = (
        _current_outer(current, current_report,
                       registry_sha256=EXPECTED_CURRENT_OUTER_REGISTRY_SHA256))
    if registered_old_outer != old_outer_fingerprints:
        raise ValueError("Consumed v8 registry does not bind the original exposed outer")
    closeout = closeout_api.read_saved_closeout(
        failure_closeout_path,
        expected_closeout_sha256=expected_failure_closeout_sha256,
        permanent_registry=permanent_closeout_registry)
    closeout_source = closeout.get("source", {})
    closeout_identities = closeout.get("consumed_outer", {}).get("identities")
    if (closeout_source.get("v8_combined_rows_sha256") != EXPECTED_FAILED_ROWS_SHA256
            or closeout_source.get("v8_outer_registry_sha256")
                != EXPECTED_CURRENT_OUTER_REGISTRY_SHA256
            or closeout_source.get("v8_outer_registry_report_sha256")
                != EXPECTED_CURRENT_OUTER_REPORT_SHA256
            or closeout_identities != current_identities
            or closeout.get("disposition", {}).get("eligible_for_outer_claim") is not False):
        raise ValueError("Failed v8 closeout does not bind the consumed outer inputs")
    formal_file, formal = _strict_json(
        formal_selection_path, "formal X/Y selection",
        expected_sha256=EXPECTED_FORMAL_SELECTION_SHA256)
    formal_seeds, formal_fingerprints = _formal_identities(formal)
    previous_file, previous = _strict_json(
        previous_development_path, "previous development supplement",
        expected_sha256=EXPECTED_PREVIOUS_DEVELOPMENT_SHA256)
    previous_seeds, previous_fingerprints = _previous_identities(previous)
    retired_file = _regular(
        retired_projection_path, "retired identity projection",
        maximum=MAX_JSON_BYTES)
    retired_report_file = _regular(
        retired_file.parent / "report.json", "retired projection report",
        maximum=MAX_JSON_BYTES)
    if (file_hash(retired_file) != EXPECTED_RETIRED_PROJECTION_SHA256
            or file_hash(retired_report_file) != EXPECTED_RETIRED_REPORT_SHA256):
        raise ValueError("Exact retired identity projection pair required")
    retired = retired_api.read_saved_projection(
        retired_file,
        expected_projection_sha256=EXPECTED_RETIRED_PROJECTION_SHA256,
        expected_report_sha256=EXPECTED_RETIRED_REPORT_SHA256)
    retired_seeds, retired_fingerprints = _unique_identities(
        retired.get("exposed_identities"),
        expected_count=retired_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
        label="retired exposed projection")

    candidates = _candidate_identity_population()
    by_fingerprint = {row["fingerprint"]: row for row in candidates}
    if len(by_fingerprint) != len(candidates):
        raise ValueError("Fixed candidate fingerprint map differs")
    row_candidate_fingerprints = row_fingerprints & set(by_fingerprint)
    row_candidate_seeds = {
        int(by_fingerprint[fingerprint]["seed"])
        for fingerprint in row_candidate_fingerprints
    }
    excluded_seeds = (
        trace_seeds | old_outer_seeds | current_seeds | formal_seeds
        | previous_seeds | retired_seeds | row_candidate_seeds
        | {int(row["seed"]) for row in closeout_identities}
    )
    excluded_fingerprints = (
        trace_fingerprints | old_outer_fingerprints | current_fingerprints
        | registered_old_outer | formal_fingerprints | previous_fingerprints
        | retired_fingerprints | row_fingerprints
        | {str(row["fingerprint"]) for row in closeout_identities}
    )
    remaining, selected, remaining_counts = _remaining_and_selected(
        candidates, excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    selected_public = [{key: row[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")} for row in selected]
    if (any(row["seed"] in excluded_seeds
            or row["fingerprint"] in excluded_fingerprints for row in selected)
            or len({row["seed"] for row in selected}) != FRESH_OUTER_SCENE_COUNT
            or len({row["fingerprint"] for row in selected}) != FRESH_OUTER_SCENE_COUNT):
        raise RuntimeError("Fresh v9 outer overlaps an exposed identity")

    # Selection is now frozen.  Any materialisation mismatch aborts the whole
    # registry; it may never trigger replacement from the ranked list.
    scenes = [_materialize_scene(row, index) for index, row in enumerate(selected)]
    if [(row["seed"], row["fingerprint"]) for row in scenes] != [
            (row["seed"], row["fingerprint"]) for row in selected]:
        raise RuntimeError("Fresh v9 outer materialisation changed the frozen set")
    projection = _projection_artifact(closeout)
    projection_file_sha256 = sha256(_json_bytes(projection)).hexdigest()
    bindings = {
        "actor_sha256": manifest_binding.FROZEN_ACTOR_SHA256,
        "actor_parameters_sha256": manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256,
        "manifest_file_sha256": file_hash(manifest),
        "manifest_validation_file_sha256": file_hash(manifest_validation),
        "original_expansion_file_sha256": file_hash(original_file),
        "failed_rows_file_sha256": file_hash(rows_file),
        "consumed_outer_registry_file_sha256": file_hash(current_file),
        "consumed_outer_report_file_sha256": file_hash(current_report_file),
        "failure_closeout_file_sha256": _sha(
            expected_failure_closeout_sha256, "failed outer closeout"),
        "failure_closeout_content_sha256": closeout["content_sha256"],
        "formal_selection_file_sha256": file_hash(formal_file),
        "previous_development_file_sha256": file_hash(previous_file),
        "retired_projection_file_sha256": file_hash(retired_file),
        "retired_projection_report_file_sha256": file_hash(retired_report_file),
        "outer_observation_hash_projection_file_sha256": projection_file_sha256,
        "outer_observation_hash_projection_content_sha256": projection["content_sha256"],
        "contract_sha256": digest(contract()),
    }
    exclusion_counts = {
        "source_row_scene_fingerprints": len(row_fingerprints),
        "source_rows_in_fixed_candidate_population": len(row_candidate_fingerprints),
        "original_expansion_trace_identities": len(trace_fingerprints),
        "consumed_outer_identities": len(current_identities),
        "formal_xy_identities": len(formal_fingerprints),
        "previous_development_identities": len(previous_fingerprints),
        "retired_exposed_identities": len(retired_fingerprints),
        "union_candidate_identities_excluded": len(candidates) - len(remaining),
    }
    exclusion_digests = {
        "excluded_seeds_sha256": digest(sorted(excluded_seeds)),
        "excluded_scene_fingerprints_sha256": digest(sorted(excluded_fingerprints)),
        "source_row_scene_fingerprints_sha256": digest(sorted(row_fingerprints)),
        "original_expansion_trace_fingerprints_sha256": digest(sorted(trace_fingerprints)),
        "consumed_outer_identities_sha256": digest(current_identities),
        "retired_exposed_fingerprints_sha256": digest(sorted(retired_fingerprints)),
    }
    statistics = {
        "fixed_candidate_scene_count": len(candidates),
        "remaining_candidate_scene_count": len(remaining),
        "remaining_family_counts": remaining_counts,
        "selected_outer_scene_count": len(scenes),
        "selected_outer_family_counts": dict(sorted(Counter(
            row["family_id"] for row in scenes).items())),
        "selected_exposed_seed_overlap": 0,
        "selected_exposed_fingerprint_overlap": 0,
        **exclusion_counts,
    }
    boundary = {
        "failed_outer_permanently_closed_before_selection": True,
        "all_previously_exposed_identities_excluded": True,
        "identity_selection_frozen_before_materialisation": True,
        "failed_rows_fields_read": ["scene_fingerprints"],
        "failed_rows_observations_read": False,
        "failed_rows_actions_read": False,
        "failed_rows_probabilities_read": False,
        "observation_hash_projection_used_for_identity_selection": False,
        "observation_hash_projection_published_for_future_validation_wins": True,
        "actor_loaded_or_inferred": False,
        "observations_generated": False,
        "program_predictions_or_metrics_read_for_selection": False,
        "protected_final_access": False,
        "formal_ready": False,
    }
    registry: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": bindings,
        "development_outer": scenes,
        "selected_outer_identities": selected_public,
        "exclusion_counts": exclusion_counts,
        "exclusion_digests": exclusion_digests,
        "statistics": statistics,
        "information_boundary": boundary,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "formal_ready": False,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
    }
    registry["content_sha256"] = digest(registry)
    report: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": deepcopy(bindings),
        "selection": {
            "salt": SELECTION_SALT,
            "family_quotas": deepcopy(FAMILY_QUOTAS),
            "selected_identity_sha256": digest(selected_public),
            "fixed_remaining_identity_sha256": digest([{key: row[key] for key in (
                "batch_index", "family_id", "seed", "fingerprint")}
                for row in sorted(remaining, key=lambda item: (
                    item["family_id"], item["batch_index"], item["family_offset"]))]),
            "exclusion_digests": deepcopy(exclusion_digests),
        },
        "statistics": deepcopy(statistics),
        "information_boundary": deepcopy(boundary),
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    if producer_sources() != sources:
        raise RuntimeError("Fresh v9 outer source closure changed during selection")
    return registry, report, projection


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = _json_bytes(value)
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


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    registry, report, projection = create_registry(**kwargs)
    destination = Path(output).expanduser().absolute()
    parent = _canonical_directory(destination.parent, "Fresh v9 outer output parent")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Fresh v9 outer output destination already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / "outer_observation_hashes.json", projection)
        if (file_hash(temporary / "outer_observation_hashes.json")
                != report["bindings"]["outer_observation_hash_projection_file_sha256"]):
            raise RuntimeError("Outer observation-hash projection bytes differ")
        _write_exclusive(temporary / "development_outer.json", registry)
        report["registry_file_sha256"] = file_hash(
            temporary / "development_outer.json")
        report["content_sha256"] = digest({
            key: child for key, child in report.items() if key != "content_sha256"})
        _write_exclusive(temporary / "report.json", report)
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--failed-rows", required=True)
    parser.add_argument("--original-expansion", required=True)
    parser.add_argument("--current-outer-registry", required=True)
    parser.add_argument("--current-outer-report", required=True)
    parser.add_argument("--failure-closeout", required=True)
    parser.add_argument("--expected-failure-closeout-sha256", required=True)
    parser.add_argument("--permanent-closeout-registry", required=True)
    parser.add_argument("--formal-selection", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--retired-projection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(
        manifest_path=args.manifest,
        failed_rows_path=args.failed_rows,
        original_expansion_path=args.original_expansion,
        current_outer_registry_path=args.current_outer_registry,
        current_outer_report_path=args.current_outer_report,
        failure_closeout_path=args.failure_closeout,
        expected_failure_closeout_sha256=args.expected_failure_closeout_sha256,
        permanent_closeout_registry=args.permanent_closeout_registry,
        formal_selection_path=args.formal_selection,
        previous_development_path=args.previous_development,
        retired_projection_path=args.retired_projection,
        output=args.output,
    )
    print(canonical({
        "status": report["status"],
        "registry": str(Path(args.output).resolve() / "development_outer.json"),
        "selected_identity_sha256": report["selection"]["selected_identity_sha256"],
        "remaining_candidate_scene_count": report["statistics"]["remaining_candidate_scene_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "SELECTION_SALT", "FAMILY_IDS", "FAMILY_QUOTAS",
    "EXPECTED_REMAINING_SCENE_COUNT", "EXPECTED_REMAINING_FAMILY_COUNTS",
    "contract", "producer_sources", "create_registry", "build", "main",
    "_scene_fingerprints_only", "_candidate_identity_population",
    "_remaining_and_selected", "_materialize_scene",
]
