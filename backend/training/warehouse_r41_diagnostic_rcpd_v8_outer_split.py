"""Freeze a genuinely fresh development outer registry for diagnostic RCPD v8.

Retired v8 candidates used every scene in their row archive for fit or
validation.  None may be renamed as a new outer fold.  This producer selects
64 replacement identities from the fixed manifest candidate population and
proves they are absent from every candidate-v3 row.

Selection is identity-only: no Actor load/inference, observation generation,
workload/replay, label, probability, program, or metric is available.  After
the identities are frozen, only those selected public scenes are materialised
through the audited no-observation reset so the existing collector can consume
them.  The old 128 fit-supplement scenes are retained unchanged; the scored
64-scene outer fold remains explicitly exposed and excluded.
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
from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as expansion_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_retired_identity_projection_v8 as retired_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticConflictWarehouseEnv
from env.warehouse_native.r41_diagnostic_conflict import (
    conflict_family_id,
    diagnostic_scene_fingerprint,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v8-fresh-outer-registry.v1"
REPORT_VERSION = VERSION
STATUS = "frozen_identity_only_pending_outer_collection"
ROOT = Path(__file__).resolve().parents[2]
SELECTION_SALT = "warehouse-r41-v8-fresh-development-outer-20260913-v1"
FAMILY_IDS = tuple(scenes_api.FAMILY_IDS)
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
FIT_SCENE_COUNT = 128
OLD_OUTER_SCENE_COUNT = 64
FRESH_OUTER_SCENE_COUNT = 64
# Compatibility names used by the existing v8 row/collector pipeline after
# its expansion_api import is moved to this new producer.
FIT_SUPPLEMENT_SCENES = FIT_SCENE_COUNT
VALIDATION_SCENES = FRESH_OUTER_SCENE_COUNT
SOURCE_ROW_SCENE_COUNT = 448
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
MAX_SCENE_MEMBER_BYTES = 64 * 1024 * 1024

EXPECTED_MANIFEST_SHA256 = manifest_binding.EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_VALIDATION_SHA256 = manifest_binding.EXPECTED_VALIDATION_SHA256
EXPECTED_SOURCE_EXPANSION_SHA256 = (
    "a687fd3fd4b145ed432af77f3ce26726d4e328df4875e4d69fc3ed351ad98748"
)
EXPECTED_SOURCE_ROWS_SHA256 = (
    "d3e8d505aefa5573825a454fc118f7a3b801bbc239498d7416fef744bd2535c0"
)
EXPECTED_FORMAL_SELECTION_SHA256 = expansion_api.EXPECTED_SELECTED_SCENES_SHA256
EXPECTED_PREVIOUS_DEVELOPMENT_SHA256 = expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256
EXPECTED_RETIRED_PROJECTION_SHA256 = expansion_api.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256
EXPECTED_RETIRED_REPORT_SHA256 = expansion_api.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((
        Path(__file__).resolve(),
        ROOT / "scripts/build_warehouse_r41_diagnostic_rcpd_v8_outer_split.py",
    )).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    expected = _sha(expected_sha256, label + " SHA-256")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact " + label + " bytes required")

    def pairs(rows):
        result = {}
        for key, child in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(parsed, dict) or file_hash(path) != expected:
        raise ValueError(label + " changed during read")
    return path, parsed


def _authenticate_opaque(value: str | Path, label: str, *, expected_sha256: str) -> Path:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    if file_hash(path) != _sha(expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _scene_fingerprints_only(value: str | Path, *, expected_sha256: str) -> tuple[Path, set[str]]:
    """Read only ``scene_fingerprints.npy`` from candidate-v3 rows."""
    path = _regular(value, "candidate-v3 rows", maximum=MAX_NPZ_BYTES)
    expected = _sha(expected_sha256, "candidate-v3 rows SHA-256")
    if file_hash(path) != expected:
        raise ValueError("Exact candidate-v3 row bytes required")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [row.filename for row in members]
            if len(names) != len(set(names)) or "scene_fingerprints.npy" not in names:
                raise ValueError("candidate-v3 row archive schema differs")
            member = archive.getinfo("scene_fingerprints.npy")
            if (member.is_dir() or member.file_size <= 0
                    or member.file_size > MAX_SCENE_MEMBER_BYTES):
                raise ValueError("candidate-v3 scene identity member is invalid")
            raw = archive.read(member)
        array = np.load(BytesIO(raw), allow_pickle=False)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("candidate-v3 scene identities cannot be read") from error
    if (array.ndim != 1 or array.dtype != np.dtype("S64")
            or file_hash(path) != expected):
        raise ValueError("candidate-v3 scene identity array differs")
    try:
        decoded = np.char.decode(array, "ascii").astype(str)
    except UnicodeDecodeError as error:
        raise ValueError("candidate-v3 scene identity is not ASCII") from error
    fingerprints = set(map(str, decoded))
    if (len(fingerprints) != SOURCE_ROW_SCENE_COUNT
            or any(_HEX.fullmatch(row) is None for row in fingerprints)):
        raise ValueError("candidate-v3 scene population differs")
    return path, fingerprints


def _identity(row: Mapping[str, Any], label: str) -> tuple[int, str]:
    if not isinstance(row, Mapping):
        raise ValueError(label + " scene must be an object")
    seed, fingerprint = row.get("seed"), row.get("fingerprint")
    if (type(seed) is not int or seed < 0 or type(fingerprint) is not str
            or _HEX.fullmatch(fingerprint) is None):
        raise ValueError(label + " scene identity differs")
    return seed, fingerprint


def _unique_identities(rows: Any, *, expected_count: int, label: str) -> tuple[set[int], set[str]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(label + " scene count differs")
    seeds: set[int] = set()
    fingerprints: set[str] = set()
    for row in rows:
        seed, fingerprint = _identity(row, label)
        if seed in seeds or fingerprint in fingerprints:
            raise ValueError(label + " contains a duplicate identity")
        seeds.add(seed); fingerprints.add(fingerprint)
    return seeds, fingerprints


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: child for key, child in value.items()
                                   if key != "content_sha256"}))


def _source_expansion(value: Mapping[str, Any]) -> tuple[list[dict[str, Any]], set[int], set[str], set[str]]:
    if (value.get("version") != expansion_api.VERSION
            or value.get("status") != expansion_api.STATUS
            or not _content_valid(value)
            or value.get("program_access") is not False
            or value.get("program_predictions_access") is not False
            or value.get("final_audit_rows_access") is not False
            or value.get("final_labels_used_for_selection") is not False):
        raise ValueError("Exact prior development expansion required")
    fit, old_outer = value.get("fit_supplement"), value.get("development_validation")
    fit_seeds, fit_fingerprints = _unique_identities(
        fit, expected_count=FIT_SCENE_COUNT, label="source fit supplement")
    outer_seeds, outer_fingerprints = _unique_identities(
        old_outer, expected_count=OLD_OUTER_SCENE_COUNT, label="exposed source outer validation")
    if fit_seeds & outer_seeds or fit_fingerprints & outer_fingerprints:
        raise ValueError("Prior development expansion splits overlap")
    if any(row.get("split") != "fit_supplement" for row in fit):
        raise ValueError("Source fit split label differs")
    trace = value.get("selection_trace")
    if not isinstance(trace, Mapping) or set(trace) != {"fit_supplement", "development_validation"}:
        raise ValueError("Prior expansion selection trace differs")
    trace_seeds: set[int] = set(); trace_fingerprints: set[str] = set()
    for split in ("development_validation", "fit_supplement"):
        rows = trace.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Prior expansion selection trace is missing")
        for row in rows:
            seed, fingerprint = _identity(row, "prior expansion trace")
            trace_seeds.add(seed); trace_fingerprints.add(fingerprint)
    if (not fit_fingerprints.issubset(trace_fingerprints)
            or not outer_fingerprints.issubset(trace_fingerprints)):
        raise ValueError("Prior accepted expansion identity is absent from trace")
    return deepcopy(fit), trace_seeds, trace_fingerprints, outer_fingerprints


def _source_bindings(value: Mapping[str, Any]) -> dict[str, str]:
    bindings = value.get("bindings")
    required = (
        "actor_sha256", "actor_parameters_sha256",
        "source_manifest_file_sha256", "designation_file_sha256",
        "previous_development_file_sha256",
    )
    if not isinstance(bindings, Mapping):
        raise ValueError("Prior expansion bindings are missing")
    result = {key: _sha(bindings.get(key), "source binding " + key)
              for key in required}
    if (result["actor_sha256"] != manifest_binding.FROZEN_ACTOR_SHA256
            or result["actor_parameters_sha256"]
                != manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256
            or result["source_manifest_file_sha256"]
                != EXPECTED_MANIFEST_SHA256
            or result["designation_file_sha256"]
                != expansion_api.EXPECTED_DESIGNATION_SHA256
            or result["previous_development_file_sha256"]
                != EXPECTED_PREVIOUS_DEVELOPMENT_SHA256):
        raise ValueError("Prior expansion component binding differs")
    return result


def _formal_identities(value: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    if (value.get("version") != "warehouse-r41-diagnostic-conflict-dynamic-selection.v3"
            or value.get("release_eligible") is not True
            or value.get("source_manifest_file_sha256") != EXPECTED_MANIFEST_SHA256
            or value.get("zero_action_overrides") is not True):
        raise ValueError("Exact formal X/Y selection required")
    return _unique_identities([*value.get("X", []), *value.get("Y", [])],
                              expected_count=6, label="formal X/Y")


def _previous_identities(value: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    if (value.get("version") != "warehouse-r41-diagnostic-development-supplement.v1"
            or value.get("status") != "passed" or value.get("program_access") is not False
            or value.get("final_audit_rows_access") is not False or not _content_valid(value)):
        raise ValueError("Exact previous development supplement required")
    return _unique_identities(value.get("scenes"), expected_count=64,
                              label="previous development supplement")


def _candidate_identity_population() -> list[dict[str, Any]]:
    """Authenticate all fixed candidate identities without scene payloads."""
    batches = retired_api._regenerate_development_candidate_identity_batches()
    per_batch = scenes_api.PER_FAMILY_PER_BATCH * len(FAMILY_IDS)
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set(); seen_fingerprints: set[str] = set()
    for batch_index, batch in enumerate(batches):
        if len(batch) != per_batch:
            raise ValueError("Fixed candidate identity batch size differs")
        for index, identity in enumerate(batch):
            seed, fingerprint = _identity(identity, "fixed candidate")
            family_index, family_offset = divmod(index, scenes_api.PER_FAMILY_PER_BATCH)
            if seed in seen_seeds or fingerprint in seen_fingerprints:
                raise ValueError("Fixed candidate population overlaps")
            seen_seeds.add(seed); seen_fingerprints.add(fingerprint)
            rows.append({"batch_index": batch_index, "family_offset": family_offset,
                         "family_id": FAMILY_IDS[family_index], "seed": seed,
                         "fingerprint": fingerprint})
    if len(rows) != per_batch * scenes_api.MAXIMUM_BATCHES:
        raise ValueError("Fixed candidate identity population differs")
    return rows


def _rank(row: Mapping[str, Any]) -> str:
    return digest({"salt": SELECTION_SALT, "family_id": row["family_id"],
                   "fingerprint": row["fingerprint"]})


def _select_fresh_identities(candidates: Sequence[Mapping[str, Any]], *,
                             excluded_seeds: set[int],
                             excluded_fingerprints: set[str]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        seed, fingerprint = _identity(row, "candidate population")
        family, batch_index = row.get("family_id"), row.get("batch_index")
        family_offset = row.get("family_offset")
        if (family not in FAMILY_IDS or type(batch_index) is not int
                or type(family_offset) is not int):
            raise ValueError("Candidate population family identity differs")
        if seed not in excluded_seeds and fingerprint not in excluded_fingerprints:
            by_family[str(family)].append(deepcopy(dict(row)))
    selected: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ordered = sorted(by_family[family], key=lambda row: (_rank(row), row["fingerprint"]))
        quota = FAMILY_QUOTAS[family]
        if len(ordered) < quota:
            raise ValueError("Insufficient fresh candidate identities for " + family)
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: (FAMILY_IDS.index(row["family_id"]), _rank(row)))
    if len(selected) != FRESH_OUTER_SCENE_COUNT or Counter(
            row["family_id"] for row in selected) != Counter(FAMILY_QUOTAS):
        raise RuntimeError("Fresh outer family quota differs")
    return selected


def _materialize_scene(identity: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Materialise a frozen identity without invoking observations."""
    seed, expected_fingerprint = _identity(identity, "selected fresh outer")
    batch_index, family = identity.get("batch_index"), identity.get("family_id")
    if type(batch_index) is not int or family not in FAMILY_IDS:
        raise ValueError("Selected fresh outer batch identity differs")
    env = R41DiagnosticConflictWarehouseEnv()
    retired_api._identity_only_reset(env, seed=seed)
    reset_family = conflict_family_id(env.state.tasks)
    if reset_family != family:
        raise ValueError("Selected fresh outer family differs at reset")
    scenes_api._candidate_start_state(env, seed=seed, batch_index=batch_index,
                                      family_id=reset_family)
    if diagnostic_scene_fingerprint(env) != expected_fingerprint:
        raise ValueError("Selected fresh outer fingerprint differs at materialisation")
    scene = scenes_api._scene_from_environment(
        env, scene_id=f"diagnostic_v8_fresh_outer_{index:04d}",
        split="development_validation", seed=seed, batch_index=batch_index)
    if scene.get("fingerprint") != expected_fingerprint or scene.get("family_id") != family:
        raise ValueError("Materialised fresh outer identity differs")
    return scene


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "fresh unfit development outer registry for RCPD v8",
        "candidate_population": "authenticated fixed manifest candidate batches",
        "manifest_registered_scene_policy": (
            "candidate-pool disjointness authenticated by the fixed manifest receipt"
        ),
        "selection_salt": SELECTION_SALT,
        "selection_inputs": ["seed", "fingerprint", "family_id", "exclusion membership"],
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "fit_supplement_policy": "copy exact prior 128 scenes unchanged",
        "old_outer_policy": "permanently exposed and excluded",
        "candidate_v3_rows_policy": "all 448 scene identities excluded",
        "prior_expansion_trace_policy": "all evaluated identities excluded",
        "selection_uses_observations": False,
        "selection_uses_actor": False,
        "selection_uses_workload_or_replay": False,
        "selection_uses_labels_probabilities_program_or_metrics": False,
        "full_manifest_json_parsed": False,
        "protected_final_access": False,
        "formal_ready": False,
    }


def create_registry(*, manifest_path: str | Path, source_rows_path: str | Path,
                    source_expansion_path: str | Path, formal_selection_path: str | Path,
                    previous_development_path: str | Path,
                    retired_projection_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _authenticate_opaque(manifest_path, "frozen diagnostic manifest",
                                    expected_sha256=EXPECTED_MANIFEST_SHA256)
    manifest_validation = _authenticate_opaque(
        manifest.parent / "validation.json", "manifest validation receipt",
        expected_sha256=EXPECTED_MANIFEST_VALIDATION_SHA256)
    rows_file, source_row_fingerprints = _scene_fingerprints_only(
        source_rows_path, expected_sha256=EXPECTED_SOURCE_ROWS_SHA256)
    expansion_file, source_expansion = _strict_json(
        source_expansion_path, "source development expansion",
        expected_sha256=EXPECTED_SOURCE_EXPANSION_SHA256)
    fit, trace_seeds, trace_fingerprints, old_outer_fingerprints = _source_expansion(source_expansion)
    source_bindings = _source_bindings(source_expansion)
    formal_file, formal = _strict_json(formal_selection_path, "formal X/Y selection",
                                       expected_sha256=EXPECTED_FORMAL_SELECTION_SHA256)
    formal_seeds, formal_fingerprints = _formal_identities(formal)
    previous_file, previous = _strict_json(
        previous_development_path, "previous development supplement",
        expected_sha256=EXPECTED_PREVIOUS_DEVELOPMENT_SHA256)
    previous_seeds, previous_fingerprints = _previous_identities(previous)
    retired_file = _regular(retired_projection_path, "retired identity projection",
                            maximum=MAX_JSON_BYTES)
    retired_report = _regular(retired_file.parent / "report.json", "retired projection audit",
                              maximum=MAX_JSON_BYTES)
    if (file_hash(retired_file) != EXPECTED_RETIRED_PROJECTION_SHA256
            or file_hash(retired_report) != EXPECTED_RETIRED_REPORT_SHA256):
        raise ValueError("Exact retired identity projection pair required")
    retired = retired_api.read_saved_projection(
        retired_file, expected_projection_sha256=EXPECTED_RETIRED_PROJECTION_SHA256,
        expected_report_sha256=EXPECTED_RETIRED_REPORT_SHA256)
    retired_seeds, retired_fingerprints = _unique_identities(
        retired.get("exposed_identities"),
        expected_count=retired_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
        label="retired exposed projection")

    excluded_seeds = trace_seeds | formal_seeds | previous_seeds | retired_seeds
    excluded_fingerprints = (trace_fingerprints | formal_fingerprints
                             | previous_fingerprints | retired_fingerprints
                             | source_row_fingerprints)
    identities = _select_fresh_identities(
        _candidate_identity_population(), excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    selected_fingerprints = {row["fingerprint"] for row in identities}
    selected_seeds = {row["seed"] for row in identities}
    if (selected_fingerprints & source_row_fingerprints
            or selected_fingerprints & old_outer_fingerprints
            or selected_fingerprints & trace_fingerprints
            or selected_seeds & excluded_seeds):
        raise RuntimeError("Fresh outer identity overlaps an exposed source")

    # No mismatch may cause replacement: the complete identity set is frozen.
    validation = [_materialize_scene(row, index) for index, row in enumerate(identities)]
    if {row["fingerprint"] for row in validation} != selected_fingerprints:
        raise RuntimeError("Fresh outer materialisation changed the frozen set")

    sources = producer_sources()
    bindings = {
        **source_bindings,
        "manifest_file_sha256": file_hash(manifest),
        "manifest_validation_file_sha256": file_hash(manifest_validation),
        "manifest_development_projection_sha256": manifest_binding.EXPECTED_DEVELOPMENT_PROJECTION_SHA256,
        "candidate_batches_identity_sha256": manifest_binding.EXPECTED_CANDIDATE_BATCHES_SHA256,
        "candidate_batch_reports_sha256": manifest_binding.EXPECTED_CANDIDATE_REPORTS_SHA256,
        "source_rows_file_sha256": file_hash(rows_file),
        "source_expansion_file_sha256": file_hash(expansion_file),
        "formal_selection_file_sha256": file_hash(formal_file),
        "previous_development_file_sha256": file_hash(previous_file),
        "retired_projection_file_sha256": file_hash(retired_file),
        "retired_projection_report_file_sha256": file_hash(retired_report),
        "contract_sha256": digest(contract()),
    }
    identities_public = [{key: row[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")} for row in identities]
    statistics = {
        "source_candidate_v3_scenes": len(source_row_fingerprints),
        "source_fit_supplement_scenes": len(fit),
        "previously_exposed_validation_scenes": len(old_outer_fingerprints),
        "prior_expansion_trace_identities_excluded": len(trace_fingerprints),
        "formal_xy_identities_excluded": len(formal_fingerprints),
        "previous_development_identities_excluded": len(previous_fingerprints),
        "retired_trace_touched_identities_excluded": len(retired_fingerprints),
        "fresh_outer_scenes": len(validation),
        "fresh_outer_family_counts": dict(sorted(Counter(
            row["family_id"] for row in validation).items())),
        "fresh_outer_source_rows_scene_overlap": 0,
        "fresh_outer_old_outer_scene_overlap": 0,
        "fresh_outer_prior_expansion_trace_overlap": 0,
        "fresh_outer_excluded_seed_overlap": 0,
        "fresh_outer_manifest_registered_scene_overlap": 0,
    }
    boundary = {
        "candidate_identity_selection_frozen_before_materialisation": True,
        "source_rows_fields_read": ["scene_fingerprints"],
        "source_rows_observations_read": False,
        "source_rows_action_labels_read": False,
        "source_rows_probabilities_read": False,
        "source_fit_payload_copied": True,
        "source_fit_payload_used_for_selection": False,
        "actor_loaded_or_inferred": False,
        "observations_generated": False,
        "workload_screen_or_replay_run": False,
        "candidate_program_or_metrics_read": False,
        "full_manifest_json_parsed": False,
        "candidate_population_disjoint_from_all_manifest_base_splits_authenticated": True,
        "protected_final_access": False,
        "outer_actor_rows_collected": False,
        "outer_candidate_scored": False,
        "old_outer_previously_exposed": True,
        "formal_ready": False,
    }
    registry: dict[str, Any] = {
        "version": VERSION, "status": STATUS, "contract": contract(),
        "bindings": bindings, "fit_supplement": fit,
        "development_validation": validation,
        "selected_outer_identities": identities_public,
        "previously_exposed_validation_scene_fingerprints": sorted(old_outer_fingerprints),
        "statistics": statistics, "information_boundary": boundary,
        "program_access": False, "program_predictions_access": False,
        "final_audit_rows_access": False, "final_labels_used_for_selection": False,
        "runtime_action_override": False, "producer_sources": sources,
        "producer_sources_sha256": digest(sources), "formal_ready": False,
    }
    registry["content_sha256"] = digest(registry)
    report: dict[str, Any] = {
        "version": REPORT_VERSION, "status": STATUS,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": deepcopy(bindings),
        "selection": {
            "salt": SELECTION_SALT, "family_quotas": deepcopy(FAMILY_QUOTAS),
            "selected_identity_sha256": digest(identities_public),
            "source_rows_scene_fingerprints_sha256": digest(sorted(source_row_fingerprints)),
            "previously_exposed_outer_fingerprints_sha256": digest(sorted(old_outer_fingerprints)),
            "prior_expansion_trace_fingerprints_sha256": digest(sorted(trace_fingerprints)),
        },
        "statistics": deepcopy(statistics), "information_boundary": deepcopy(boundary),
        "producer_sources": sources, "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    return registry, report


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def build(*, manifest_path: str | Path, source_rows_path: str | Path,
          source_expansion_path: str | Path, formal_selection_path: str | Path,
          previous_development_path: str | Path, retired_projection_path: str | Path,
          output: str | Path) -> dict[str, Any]:
    registry, report = create_registry(
        manifest_path=manifest_path, source_rows_path=source_rows_path,
        source_expansion_path=source_expansion_path,
        formal_selection_path=formal_selection_path,
        previous_development_path=previous_development_path,
        retired_projection_path=retired_projection_path)
    destination = Path(output).expanduser().absolute(); parent = destination.parent
    if (not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Fresh outer output destination is unsafe")
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / "development_expansion.json", registry)
        report["registry_file_sha256"] = file_hash(temporary / "development_expansion.json")
        report["content_sha256"] = digest({key: child for key, child in report.items()
                                           if key != "content_sha256"})
        _write_exclusive(temporary / "report.json", report)
        os.rename(temporary, destination); temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--source-expansion", required=True)
    parser.add_argument("--formal-selection", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--retired-projection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(
        manifest_path=args.manifest, source_rows_path=args.source_rows,
        source_expansion_path=args.source_expansion,
        formal_selection_path=args.formal_selection,
        previous_development_path=args.previous_development,
        retired_projection_path=args.retired_projection, output=args.output)
    print(canonical(report)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "REPORT_VERSION", "STATUS", "FAMILY_IDS", "FAMILY_QUOTAS",
           "contract", "create_registry", "build", "main"]
