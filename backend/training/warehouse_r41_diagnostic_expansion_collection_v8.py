"""Collect the fixed program-blind development expansion for diagnostic v8.

This producer is deliberately separate from the RCPD fitter and from every
final-test component.  It consumes only the frozen Actor/protocol/designation,
the development replay projection of the fixed manifest, and the exact
program-blind expansion registry.  Its collection schedule is fixed here and
cannot be selected by a caller.
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

import numpy as np

from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as expansion_api
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training import warehouse_r41_diagnostic_rows_v8 as rows_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
    read_authenticated_bytes,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-expansion-collection.v8"
STATUS = "passed_fresh_program_blind_expansion_collection"
ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 512 * 1024 * 1024
FIT_SCENE_OFFSET = 192
FIT_SCENE_COUNT = 128
VALIDATION_SCENE_OFFSET = 320
VALIDATION_SCENE_COUNT = 64
EXPECTED_EXPANSION_REGISTRY_SHA256 = (
    "a687fd3fd4b145ed432af77f3ce26726d4e328df4875e4d69fc3ed351ad98748"
)
EXPECTED_EXPANSION_REPORT_SHA256 = (
    "be0c009847ea227710ad7ff26afbfc8c414b0e74f78505bbd2805ef0a6311d2a"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REPORT_FIELDS = {
    "version", "status", "contract", "bindings", "collection",
    "sources", "artifacts", "content_sha256", "final_test_accessed",
    "final_identity_commitment_used_for_overlap_exclusion",
    "final_scene_geometry_access", "final_trajectories_access",
    "final_actor_outputs_or_labels_access",
    "final_used_for_fit_selection_metrics", "participant_data_accessed",
    "program_accessed", "runtime_action_override", "formal_ready",
}
_BINDING_FIELDS = {
    "actor_file_sha256", "actor_parameters_sha256", "protocol_file_sha256",
    "protocol_content_sha256", "manifest_file_sha256",
    "manifest_content_sha256", "manifest_validation_file_sha256",
    "designation_file_sha256", "expansion_registry_file_sha256",
    "expansion_registry_content_sha256", "expansion_report_file_sha256",
    "expansion_report_semantic_sha256", "rows_file_sha256",
    "rows_semantic_sha256", "contract_sha256", "producer_sources_sha256",
}
_COLLECTION_FIELDS = {
    "source_layout", "fit_scene_offset", "fit_scene_count",
    "validation_scene_offset", "validation_scene_count", "partner_count",
    "rows", "fit_rows", "validation_rows", "fit_scene_population",
    "validation_scene_population", "fit_environment_steps",
    "validation_environment_steps", "environment_steps", "row_accounting",
    "all_actor_probabilities_and_actions_exact",
    "all_submitted_actions_equal_policy_actions",
    "validation_wins_observation_deduplication",
}
_ACCOUNTING_FIELDS = {
    "raw_train_rows", "raw_validation_rows", "raw_train_ordinary_rows",
    "raw_validation_ordinary_rows", "raw_train_anchor_count",
    "raw_validation_anchor_count", "raw_train_environment_steps",
    "raw_validation_environment_steps", "raw_collection_environment_steps",
    "train_rows_removed_for_exact_validation_overlap",
    "train_anchors_affected_by_duplicate_row_removal",
    "effective_train_pair_endpoint_rows",
}


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "fresh program-blind expansion-only public-observation rows",
        "input_population": "fixed development expansion registry only",
        "fit_collection": {
            "scene_offset": FIT_SCENE_OFFSET,
            "scene_count": FIT_SCENE_COUNT,
            "dense_critical": True,
        },
        "validation_collection": {
            "scene_offset": VALIDATION_SCENE_OFFSET,
            "scene_count": VALIDATION_SCENE_COUNT,
            "dense_critical": False,
            "critical_anchor_period": 5,
        },
        "partners": list(v7.PARTNERS),
        "source_layout": "expansion_only",
        "validation_wins_observation_deduplication": True,
        "actor_probabilities_and_actions_recomputed": True,
        "runtime_action_override": False,
        "program_access": False,
        "prior_rows_access": False,
        "retired_holdout_access": False,
        "historical_final_access": False,
        "participant_data_access": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    sources = dict(local_source_hashes((Path(__file__).resolve(),)))
    for path, value in designation_binding.designation.source_closure().items():
        if path in sources and sources[path] != value:
            raise RuntimeError("Designation/source closure hash disagreement: " + path)
        sources[path] = value
    return dict(sorted(sources.items()))


def _regular(
    value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES,
) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items):
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
        raise ValueError(label + " cannot be parsed") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_JSON_BYTES:
                raise ValueError(label + " exceeds its size limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return _strict_json_bytes(b"".join(chunks), label)


def _manifest_validation_path(manifest_path: Path) -> Path:
    path = _regular(
        manifest_path.parent / "validation.json", "manifest validation")
    if (path.parent != manifest_path.parent
            or file_hash(path) != manifest_binding.EXPECTED_VALIDATION_SHA256):
        raise ValueError("Exact canonical manifest validation bytes required")
    return path


def _resolve_components(designation_path: Path) -> dict[str, Path]:
    raw = read_authenticated_bytes(
        designation_path, label="diagnostic Actor designation",
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
        maximum=MAX_JSON_BYTES,
    )
    return designation_binding.resolve_bound_components_from_bytes(
        raw, original_path=designation_path,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )


def _expected_hashes(
    components: Mapping[str, Path], *, saved_report_sha256: str | None = None,
    saved_rows_sha256: str | None = None,
) -> dict[str, str]:
    result = {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "expansion_registry": EXPECTED_EXPANSION_REGISTRY_SHA256,
        "expansion_report": EXPECTED_EXPANSION_REPORT_SHA256,
        "designation_actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "designation_protocol": (
            designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256),
        "designation_training_ledger": (
            designation_binding.designation.EXPECTED_LEDGER_SHA256),
        "designation_dual_evaluation": (
            designation_binding.designation.EXPECTED_DUAL_EVALUATION_SHA256),
        "designation_failure_closeout": (
            designation_binding.designation.EXPECTED_CLOSEOUT_SHA256),
    }
    if set(components) != set(designation_binding.designation.ARTIFACT_NAMES):
        raise ValueError("Complete designation component set required")
    if saved_report_sha256 is not None or saved_rows_sha256 is not None:
        if (_HEX.fullmatch(str(saved_report_sha256)) is None
                or _HEX.fullmatch(str(saved_rows_sha256)) is None):
            raise ValueError("Exact saved collection hashes required")
        result["saved_report"] = str(saved_report_sha256)
        result["saved_rows"] = str(saved_rows_sha256)
    return result


def _validate_registry_and_inputs(
    *, paths: Mapping[str, Path], designation_original: Path,
    component_originals: Mapping[str, Path], sources: Mapping[str, str],
) -> tuple[NumPyNativeActor, list[dict[str, Any]], list[dict[str, Any]],
           dict[str, Any], dict[str, Any], dict[str, Any]]:
    if producer_sources() != dict(sources):
        raise RuntimeError("Expansion collector sources changed before validation")
    designation = designation_binding.read_bound_designation_snapshot(
        paths["designation"], original_path=designation_original,
        components={name: paths["designation_" + name]
                    for name in component_originals},
        original_components=component_originals,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    if (component_originals["actor"] != paths["actor"]
            and file_hash(component_originals["actor"]) != file_hash(paths["actor"])):
        raise ValueError("Explicit Actor differs from designation")
    actor = NumPyNativeActor(paths["actor"])
    protocol = _strict_json(paths["protocol"], "training protocol")
    manifest = manifest_binding.read_saved_manifest(
        paths["manifest"], actor_path=paths["actor"],
        replay_scope="development")
    expansion = _strict_json(
        paths["expansion_registry"], "development expansion registry")
    expansion_report = _strict_json(
        paths["expansion_report"], "development expansion report")
    if (file_hash(paths["expansion_registry"])
            != EXPECTED_EXPANSION_REGISTRY_SHA256
            or file_hash(paths["expansion_report"])
                != EXPECTED_EXPANSION_REPORT_SHA256
            or expansion_report.get("version") != expansion_api.VERSION
            or expansion_report.get("status") != expansion_api.STATUS
            or expansion_report.get("content_sha256") != digest({
                key: value for key, value in expansion_report.items()
                if key != "content_sha256"
            })
            or expansion_report.get("registry_file_sha256")
                != EXPECTED_EXPANSION_REGISTRY_SHA256
            or expansion_report.get("registry_content_sha256")
                != expansion.get("content_sha256")
            or expansion_report.get("bindings") != expansion.get("bindings")
            or expansion_report.get("producer_sources")
                != expansion_api.producer_sources()
            or expansion.get("bindings", {}).get(
                "previous_development_file_sha256")
                != expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256):
        raise ValueError("Exact program-blind development expansion required")
    fit_scenes, validation_scenes = rows_api._validate_expansion_registry(
        expansion, registry_path=paths["expansion_registry"], actor=actor,
        manifest_path=paths["manifest"], designation_path=paths["designation"],
        prior_report={"bindings": {
            "development_supplement_file_sha256": (
                expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256),
        }},
    )
    if (len(fit_scenes) != FIT_SCENE_COUNT
            or len(validation_scenes) != VALIDATION_SCENE_COUNT):
        raise ValueError("Fixed expansion scene count differs")
    if (designation.get("bindings", {}).get("actor_sha256")
            != actor.artifact_sha256
            or designation.get("bindings", {}).get("protocol_file_sha256")
                != file_hash(paths["protocol"])
            or manifest.get("frozen_actor", {}).get("sha256")
                != actor.artifact_sha256):
        raise ValueError("Expansion collection input identity differs")
    return (actor, fit_scenes, validation_scenes, protocol, manifest,
            expansion_report)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
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


def _report(
    *, paths: Mapping[str, Path], actor: NumPyNativeActor,
    protocol: Mapping[str, Any], manifest: Mapping[str, Any],
    expansion_report: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    accounting: Mapping[str, Any], fit_steps: int, validation_steps: int,
    sources: Mapping[str, str], rows_path: Path,
) -> dict[str, Any]:
    scenes = rows_api._decode(
        arrays["scene_fingerprints"], "collected scene fingerprints")
    split = arrays["split_validation"]
    rows_sha = file_hash(rows_path)
    semantic_sha = rows_api.arrays_digest(arrays)
    value = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "actor_file_sha256": file_hash(paths["actor"]),
            "actor_parameters_sha256": actor.metadata[
                "actor_parameters_sha256"],
            "protocol_file_sha256": file_hash(paths["protocol"]),
            "protocol_content_sha256": digest(protocol),
            "manifest_file_sha256": file_hash(paths["manifest"]),
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_validation_file_sha256": file_hash(
                paths["manifest_validation"]),
            "designation_file_sha256": file_hash(paths["designation"]),
            "expansion_registry_file_sha256": file_hash(
                paths["expansion_registry"]),
            "expansion_registry_content_sha256": expansion_report[
                "registry_content_sha256"],
            "expansion_report_file_sha256": file_hash(
                paths["expansion_report"]),
            "expansion_report_semantic_sha256": digest(expansion_report),
            "rows_file_sha256": rows_sha,
            "rows_semantic_sha256": semantic_sha,
            "contract_sha256": digest(contract()),
            "producer_sources_sha256": digest(sources),
        },
        "collection": {
            "source_layout": "expansion_only",
            "fit_scene_offset": FIT_SCENE_OFFSET,
            "fit_scene_count": FIT_SCENE_COUNT,
            "validation_scene_offset": VALIDATION_SCENE_OFFSET,
            "validation_scene_count": VALIDATION_SCENE_COUNT,
            "partner_count": len(v7.PARTNERS),
            "rows": len(arrays["observations"]),
            "fit_rows": int(np.sum(~split)),
            "validation_rows": int(np.sum(split)),
            "fit_scene_population": len(set(map(str, scenes[~split]))),
            "validation_scene_population": len(set(map(str, scenes[split]))),
            "fit_environment_steps": fit_steps,
            "validation_environment_steps": validation_steps,
            "environment_steps": fit_steps + validation_steps,
            "row_accounting": deepcopy(dict(accounting)),
            "all_actor_probabilities_and_actions_exact": True,
            "all_submitted_actions_equal_policy_actions": True,
            "validation_wins_observation_deduplication": True,
        },
        "sources": deepcopy(dict(sources)),
        "artifacts": {
            "expansion_rows.npz": {
                "file_sha256": rows_sha,
                "semantic_sha256": semantic_sha,
            },
        },
        "final_test_accessed": False,
        "final_identity_commitment_used_for_overlap_exclusion": False,
        "final_scene_geometry_access": False,
        "final_trajectories_access": False,
        "final_actor_outputs_or_labels_access": False,
        "final_used_for_fit_selection_metrics": False,
        "participant_data_accessed": False,
        "program_accessed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _validate_saved_snapshot(
    *, paths: Mapping[str, Path], designation_original: Path,
    component_originals: Mapping[str, Path], sources: Mapping[str, str],
    expected_arrays: Mapping[str, np.ndarray] | None = None,
    expected_accounting: Mapping[str, Any] | None = None,
    expected_fit_steps: int | None = None,
    expected_validation_steps: int | None = None,
) -> dict[str, Any]:
    actor, fit_scenes, validation_scenes, protocol, manifest, expansion_report = (
        _validate_registry_and_inputs(
            paths=paths, designation_original=designation_original,
            component_originals=component_originals, sources=sources))
    report = _strict_json(paths["saved_report"], "expansion collection report")
    if (set(report) != _REPORT_FIELDS
            or report.get("version") != VERSION
            or report.get("status") != STATUS
            or report.get("contract") != contract()
            or report.get("content_sha256") != digest({
                key: value for key, value in report.items()
                if key != "content_sha256"
            })
            or report.get("sources") != dict(sources)
            or report.get("final_test_accessed") is not False
            or report.get("final_identity_commitment_used_for_overlap_exclusion")
                is not False
            or report.get("final_scene_geometry_access") is not False
            or report.get("final_trajectories_access") is not False
            or report.get("final_actor_outputs_or_labels_access") is not False
            or report.get("final_used_for_fit_selection_metrics") is not False
            or report.get("participant_data_accessed") is not False
            or report.get("program_accessed") is not False
            or report.get("runtime_action_override") is not False
            or report.get("formal_ready") is not False):
        raise ValueError("Expansion collection report schema differs")
    arrays = rows_api._load_npz(
        paths["saved_rows"], "fresh development expansion rows")
    relations = R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])
    layout = rows_api._validate_expansion_rows(
        arrays, actor=actor, fit_scenes=fit_scenes,
        validation_scenes=validation_scenes, prior_arrays=None,
        relations=relations)
    if layout != "expansion_only":
        raise ValueError("Fresh expansion collection layout differs")
    if expected_arrays is None:
        runtime = manifest_binding.build_runtime(
            actor_path=paths["actor"], protocol_path=paths["protocol"],
            manifest_path=paths["manifest"])
        fit_rows, replay_fit_steps = v7._collect(
            runtime, fit_scenes, scene_offset=FIT_SCENE_OFFSET,
            dense_critical=True, progress_label=None)
        validation_rows, replay_validation_steps = v7._collect(
            runtime, validation_scenes, scene_offset=VALIDATION_SCENE_OFFSET,
            dense_critical=False, progress_label=None)
        replay_arrays, replay_accounting = v7._rows_to_arrays(
            fit_rows, validation_rows)
    else:
        if (expected_accounting is None or expected_fit_steps is None
                or expected_validation_steps is None):
            raise ValueError("Complete in-memory collection expectation required")
        replay_arrays = {name: np.asarray(value).copy()
                         for name, value in expected_arrays.items()}
        replay_accounting = deepcopy(dict(expected_accounting))
        replay_fit_steps = expected_fit_steps
        replay_validation_steps = expected_validation_steps
    if (set(arrays) != set(replay_arrays)
            or any(not np.array_equal(arrays[name], replay_arrays[name])
                   for name in arrays)):
        raise ValueError("Expansion rows differ from fixed collection replay")
    rows_sha = file_hash(paths["saved_rows"])
    semantic_sha = rows_api.arrays_digest(arrays)
    bindings = report.get("bindings")
    artifact = report.get("artifacts", {}).get("expansion_rows.npz")
    collection = report.get("collection")
    scenes = rows_api._decode(
        arrays["scene_fingerprints"], "collected scene fingerprints")
    split = arrays["split_validation"]
    accounting = (collection.get("row_accounting")
                  if isinstance(collection, Mapping) else None)
    if (not isinstance(bindings, Mapping)
            or set(bindings) != _BINDING_FIELDS
            or not isinstance(artifact, Mapping)
            or set(report.get("artifacts", {})) != {"expansion_rows.npz"}
            or set(artifact) != {"file_sha256", "semantic_sha256"}
            or artifact.get("file_sha256") != rows_sha
            or artifact.get("semantic_sha256") != semantic_sha
            or bindings.get("rows_file_sha256") != rows_sha
            or bindings.get("rows_semantic_sha256") != semantic_sha
            or bindings.get("actor_file_sha256") != file_hash(paths["actor"])
            or bindings.get("actor_parameters_sha256")
                != actor.metadata["actor_parameters_sha256"]
            or bindings.get("protocol_file_sha256")
                != file_hash(paths["protocol"])
            or bindings.get("protocol_content_sha256") != digest(protocol)
            or bindings.get("manifest_file_sha256")
                != file_hash(paths["manifest"])
            or bindings.get("manifest_content_sha256") != manifest["content_sha256"]
            or bindings.get("manifest_validation_file_sha256")
                != file_hash(paths["manifest_validation"])
            or bindings.get("designation_file_sha256")
                != file_hash(paths["designation"])
            or bindings.get("expansion_registry_file_sha256")
                != file_hash(paths["expansion_registry"])
            or bindings.get("expansion_registry_content_sha256")
                != expansion_report["registry_content_sha256"]
            or bindings.get("expansion_report_file_sha256")
                != file_hash(paths["expansion_report"])
            or bindings.get("expansion_report_semantic_sha256")
                != digest(expansion_report)
            or bindings.get("contract_sha256") != digest(contract())
            or bindings.get("producer_sources_sha256") != digest(sources)
            or not isinstance(collection, Mapping)
            or set(collection) != _COLLECTION_FIELDS
            or not isinstance(accounting, Mapping)
            or set(accounting) != _ACCOUNTING_FIELDS
            or any(type(value) is not int or value < 0
                   for value in accounting.values())
            or collection.get("source_layout") != "expansion_only"
            or collection.get("fit_scene_offset") != FIT_SCENE_OFFSET
            or collection.get("fit_scene_count") != FIT_SCENE_COUNT
            or collection.get("validation_scene_offset")
                != VALIDATION_SCENE_OFFSET
            or collection.get("validation_scene_count")
                != VALIDATION_SCENE_COUNT
            or collection.get("partner_count") != len(v7.PARTNERS)
            or collection.get("rows") != len(arrays["observations"])
            or collection.get("fit_rows") != int(np.sum(~split))
            or collection.get("validation_rows") != int(np.sum(split))
            or collection.get("fit_scene_population")
                != len(set(map(str, scenes[~split])))
            or collection.get("validation_scene_population")
                != len(set(map(str, scenes[split])))
            or collection.get("environment_steps")
                != collection.get("fit_environment_steps", -1)
                    + collection.get("validation_environment_steps", -2)
            or collection.get("fit_environment_steps") != replay_fit_steps
            or collection.get("validation_environment_steps")
                != replay_validation_steps
            or dict(accounting) != dict(replay_accounting)
            or collection.get("fit_environment_steps")
                != accounting.get("raw_train_environment_steps")
            or collection.get("validation_environment_steps")
                != accounting.get("raw_validation_environment_steps")
            or collection.get("environment_steps")
                != accounting.get("raw_collection_environment_steps")
            or collection.get("rows")
                != accounting.get("raw_train_rows")
                    + accounting.get("raw_validation_rows")
                    - accounting.get(
                        "train_rows_removed_for_exact_validation_overlap")
            or collection.get("all_actor_probabilities_and_actions_exact")
                is not True
            or collection.get("all_submitted_actions_equal_policy_actions")
                is not True
            or collection.get("validation_wins_observation_deduplication")
                is not True):
        raise ValueError("Expansion collection report binding differs")
    return deepcopy(report)


def _snapshot_inputs(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path, expansion_report_path: str | Path,
    saved_report_path: str | Path | None = None,
    saved_rows_path: str | Path | None = None,
    expected_saved_report_sha256: str | None = None,
    expected_saved_rows_sha256: str | None = None,
) -> tuple[dict[str, Path], dict[str, Path], Path, ImmutableInputSnapshot]:
    designation = _regular(designation_path, "designation")
    components = _resolve_components(designation)
    actor = _regular(
        actor_path, "Actor", maximum=rows_api.MAX_NPZ_COMPRESSED_BYTES)
    protocol = _regular(protocol_path, "protocol")
    if components["actor"] != actor or components["protocol"] != protocol:
        raise ValueError("Explicit Actor/protocol must be designation components")
    manifest = _regular(manifest_path, "manifest")
    originals: dict[str, Path] = {
        "actor": actor,
        "protocol": protocol,
        "manifest": manifest,
        "manifest_validation": _manifest_validation_path(manifest),
        "designation": designation,
        "expansion_registry": _regular(
            expansion_registry_path, "expansion registry"),
        "expansion_report": _regular(
            expansion_report_path, "expansion report"),
    }
    originals.update({"designation_" + name: path
                      for name, path in components.items()})
    if saved_report_path is not None or saved_rows_path is not None:
        if (saved_report_path is None or saved_rows_path is None
                or expected_saved_report_sha256 is None
                or expected_saved_rows_sha256 is None):
            raise ValueError("Complete saved collection artifact pair required")
        originals["saved_report"] = _regular(
            saved_report_path, "saved expansion collection report")
        originals["saved_rows"] = _regular(
            saved_rows_path, "saved expansion collection rows",
            maximum=rows_api.MAX_NPZ_COMPRESSED_BYTES)
    snapshot = ImmutableInputSnapshot(
        originals,
        expected_sha256=_expected_hashes(
            components, saved_report_sha256=expected_saved_report_sha256,
            saved_rows_sha256=expected_saved_rows_sha256),
        relative_names={
            "manifest": "manifest/manifest.json",
            "manifest_validation": "manifest/validation.json",
        },
        maximum_bytes={
            "actor": rows_api.MAX_NPZ_COMPRESSED_BYTES,
            "designation_actor": rows_api.MAX_NPZ_COMPRESSED_BYTES,
            **({"saved_rows": rows_api.MAX_NPZ_COMPRESSED_BYTES}
               if saved_rows_path is not None else {}),
        },
        prefix="warehouse-r41-expansion-collection-inputs-",
    )
    return snapshot.paths, components, designation, snapshot


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path, expansion_report_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    paths, components, designation_original, snapshot = _snapshot_inputs(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expansion_registry_path=expansion_registry_path,
        expansion_report_path=expansion_report_path)
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        snapshot.__exit__(None, None, None)
        raise FileExistsError(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve() != parent:
        snapshot.__exit__(None, None, None)
        raise ValueError("Expansion collection output parent is unsafe")
    temporary: Path | None = None
    lock = parent / ("." + destination.name + ".lock")
    lock_fd: int | None = None
    published = False
    try:
        lock_fd = os.open(
            lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        actor, fit_scenes, validation_scenes, protocol, manifest, expansion_report = (
            _validate_registry_and_inputs(
                paths=paths, designation_original=designation_original,
                component_originals=components, sources=sources))
        runtime = manifest_binding.build_runtime(
            actor_path=paths["actor"], protocol_path=paths["protocol"],
            manifest_path=paths["manifest"])
        fit_rows, fit_steps = v7._collect(
            runtime, fit_scenes, scene_offset=FIT_SCENE_OFFSET,
            dense_critical=True, progress_label="fit_supplement")
        validation_rows, validation_steps = v7._collect(
            runtime, validation_scenes, scene_offset=VALIDATION_SCENE_OFFSET,
            dense_critical=False, progress_label="development_validation")
        arrays, accounting = v7._rows_to_arrays(fit_rows, validation_rows)
        relations = R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])
        layout = rows_api._validate_expansion_rows(
            arrays, actor=actor, fit_scenes=fit_scenes,
            validation_scenes=validation_scenes, prior_arrays=None,
            relations=relations)
        if layout != "expansion_only":
            raise RuntimeError("Fresh collection did not produce expansion-only rows")
        temporary = Path(tempfile.mkdtemp(
            prefix=".warehouse-r41-expansion-collection-", dir=parent))
        os.chmod(temporary, 0o700)
        rows_path = temporary / "expansion_rows.npz"
        _write_npz(rows_path, arrays)
        report = _report(
            paths=paths, actor=actor, protocol=protocol, manifest=manifest,
            expansion_report=expansion_report, arrays=arrays,
            accounting=accounting, fit_steps=fit_steps,
            validation_steps=validation_steps, sources=sources,
            rows_path=rows_path)
        _write_json(temporary / "report.json", report)
        # Validate the complete staged pair before publication.
        staged_paths = dict(paths)
        staged_paths.update({
            "saved_report": temporary / "report.json",
            "saved_rows": rows_path,
        })
        checked = _validate_saved_snapshot(
            paths=staged_paths, designation_original=designation_original,
            component_originals=components, sources=sources,
            expected_arrays=arrays, expected_accounting=accounting,
            expected_fit_steps=fit_steps,
            expected_validation_steps=validation_steps)
        if checked != report:
            raise RuntimeError("Staged expansion collection changed")
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Expansion collector sources changed before publish")
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Expansion collector sources changed at publish")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
        published = True
        temporary = None
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Expansion collector sources changed after publish")
        published_paths = dict(paths)
        published_paths.update({
            "saved_report": destination / "report.json",
            "saved_rows": destination / "expansion_rows.npz",
        })
        checked = _validate_saved_snapshot(
            paths=published_paths, designation_original=designation_original,
            component_originals=components, sources=sources,
            expected_arrays=arrays, expected_accounting=accounting,
            expected_fit_steps=fit_steps,
            expected_validation_steps=validation_steps)
        if checked != report:
            raise RuntimeError("Published expansion collection changed")
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Expansion collector sources changed at return")
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
            lock.unlink(missing_ok=True)
        snapshot.__exit__(None, None, None)


def read_saved_artifacts(
    *, report_path: str | Path, rows_path: str | Path,
    expected_report_sha256: str, expected_rows_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path, expansion_report_path: str | Path,
) -> dict[str, Any]:
    if (_HEX.fullmatch(str(expected_report_sha256)) is None
            or _HEX.fullmatch(str(expected_rows_sha256)) is None):
        raise ValueError("Exact saved expansion collection hashes required")
    sources = producer_sources()
    paths, components, designation_original, snapshot = _snapshot_inputs(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expansion_registry_path=expansion_registry_path,
        expansion_report_path=expansion_report_path,
        saved_report_path=report_path, saved_rows_path=rows_path,
        expected_saved_report_sha256=expected_report_sha256,
        expected_saved_rows_sha256=expected_rows_sha256)
    try:
        report = _validate_saved_snapshot(
            paths=paths, designation_original=designation_original,
            component_originals=components, sources=sources)
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Expansion collector sources changed during read")
        return deepcopy(report)
    finally:
        snapshot.__exit__(None, None, None)


def read_saved_report(
    output: str | Path, *, expected_report_sha256: str,
    expected_rows_sha256: str, actor_path: str | Path,
    protocol_path: str | Path, manifest_path: str | Path,
    designation_path: str | Path, expansion_registry_path: str | Path,
    expansion_report_path: str | Path,
) -> dict[str, Any]:
    directory = Path(output).expanduser().absolute()
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        raise ValueError("Expansion collection directory is unsafe")
    return read_saved_artifacts(
        report_path=directory / "report.json",
        rows_path=directory / "expansion_rows.npz",
        expected_report_sha256=expected_report_sha256,
        expected_rows_sha256=expected_rows_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expansion_registry_path=expansion_registry_path,
        expansion_report_path=expansion_report_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--expansion-registry", required=True)
    parser.add_argument("--expansion-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        expansion_registry_path=args.expansion_registry,
        expansion_report_path=args.expansion_report, output=args.output)
    print(canonical({
        "status": report["status"],
        "rows": report["collection"]["rows"],
        "output": str(Path(args.output).expanduser().absolute()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "FIT_SCENE_OFFSET", "FIT_SCENE_COUNT",
    "VALIDATION_SCENE_OFFSET", "VALIDATION_SCENE_COUNT",
    "EXPECTED_EXPANSION_REGISTRY_SHA256", "EXPECTED_EXPANSION_REPORT_SHA256",
    "contract", "producer_sources", "build", "read_saved_artifacts",
    "read_saved_report", "main",
]
