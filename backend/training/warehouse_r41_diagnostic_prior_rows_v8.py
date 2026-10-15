"""Strictly reauthenticate the v7 row evidence consumed by RCPD v8.

RCPD v8 never consumes the v7 program or its failed candidate.  This producer
therefore reauthenticates only the exact historical v7 report and row archive,
validates every row against the unchanged Actor and frozen scene registries,
and binds that row evidence to designation v2.  It performs no model fit and
has no final-test input.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_development_supplement as supplement_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
    read_authenticated_bytes,
)
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training import warehouse_r41_diagnostic_rows_v8 as rows_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-prior-v7-rows.v8-designation-v2"
STATUS = "passed_exact_v7_row_reauthentication"
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_REPORT_SHA256 = (
    "cde770032730db0a663e1b4363e5a5b4df93e97cbd1b0b3aa7db88cddbaeb99d"
)
EXPECTED_SOURCE_ROWS_SHA256 = (
    "a21bd052a8bed48bb103a14e7da6a1f21291af7cac475928686d38693bd52686"
)
EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256 = (
    "8931b74940f1c41940d9fbb73a52f940f527b8f9c67dfd6b94a4a4d1afd977fe"
)


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "consumed_by": "warehouse-r41-diagnostic-rcpd.v8",
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_rows_sha256": EXPECTED_SOURCE_ROWS_SHA256,
        "scope": "v7 rows consumed by v8 only",
        "v7_program_refit": False,
        "v7_candidate_reselected": False,
        "row_validation": [
            "schema dtype shape and finite values",
            "exact public observation hashes",
            "fixed Actor probabilities and deterministic actions",
            "all submitted actions equal policy actions",
            "complete scene episode and intervention population",
            "development validation split and zero observation overlap",
        ],
        "historical_v7_report_role": "fixed provenance only",
        "historical_final_overlap_check": (
            "deferred to the irrevocable claim-bound exclusion phase"
        ),
        "final_identity_commitment_used_for_overlap_exclusion": False,
        "final_scene_geometry_access": False,
        "final_trajectories_access": False,
        "final_actor_outputs_or_labels_access": False,
        "final_used_for_fit_selection_metrics": False,
        "participant_data_accessed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    sources = local_source_hashes((Path(__file__).resolve(),))
    for path, sha256 in designation_binding.designation.source_closure().items():
        if path in sources and sources[path] != sha256:
            raise RuntimeError("Designation/source closure hash disagreement: " + path)
        sources[path] = sha256
    return dict(sorted(sources.items()))


def _fixed_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash every canonical input used by one authentication transaction."""
    result: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if (not path.is_file() or path.is_symlink() or path.resolve() != path):
            raise RuntimeError(
                "Prior-row fixed input changed during authentication: " + name)
        result[name] = file_hash(path)
    return result


def _integrity_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Freeze the complete producer closure and fixed input bytes pre-auth."""
    return {
        "producer_sources": producer_sources(),
        "fixed_input_sha256": _fixed_input_hashes(paths),
    }


def _verify_integrity(
    snapshot: Mapping[str, Any], paths: Mapping[str, Path], *, phase: str,
) -> None:
    try:
        current_sources = producer_sources()
        current_inputs = _fixed_input_hashes(paths)
    except BaseException as error:
        raise RuntimeError(
            "Prior-row sources or fixed inputs changed during " + phase
        ) from error
    if (current_sources != snapshot.get("producer_sources")
            or current_inputs != snapshot.get("fixed_input_sha256")):
        raise RuntimeError(
            "Prior-row sources or fixed inputs changed during " + phase)


def _regular(value: str | Path, label: str, maximum: int | None = None) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or (maximum is not None and path.stat().st_size > maximum)):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _manifest_validation_path(manifest_path: Path) -> Path:
    validation = _regular(
        manifest_path.parent / "validation.json", "manifest validation")
    if (validation.parent != manifest_path.parent
            or file_hash(validation)
                != manifest_binding.EXPECTED_VALIDATION_SHA256):
        raise ValueError("Exact canonical manifest validation bytes required")
    return validation


def _read(path: Path, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _designation_component_paths(designation_path: Path) -> dict[str, Path]:
    """Resolve all five fixed designation inputs through the shared adapter."""
    return {
        "designation_" + name: _regular(path, "designation " + name)
        for name, path in designation_binding.resolve_bound_components(
            designation_path).items()
    }


def _resolved_designation_components(designation_path: Path) -> dict[str, Path]:
    """Resolve component paths from the same authenticated designation bytes."""
    raw = read_authenticated_bytes(
        designation_path, label="diagnostic Actor designation",
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    return designation_binding.resolve_bound_components_from_bytes(
        raw, original_path=designation_path,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )


def _expected_snapshot_hashes(
    *, expected_report_sha256: str | None = None,
) -> dict[str, str]:
    result = {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "supplement": EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_rows": EXPECTED_SOURCE_ROWS_SHA256,
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
    if expected_report_sha256 is not None:
        result.update({
            "saved_report": expected_report_sha256,
            "saved_rows": EXPECTED_SOURCE_ROWS_SHA256,
            "embedded_source_report": EXPECTED_SOURCE_REPORT_SHA256,
        })
    return result


def _validate_source_report(
    report: Mapping[str, Any], *, actor_path: Path, protocol_path: Path,
    manifest_path: Path, development_supplement_path: Path,
    rows_path: Path,
) -> None:
    bindings = report.get("bindings")
    artifacts = report.get("evidence_artifacts")
    accounting = report.get("row_accounting")
    execution = report.get("execution")
    if (report.get("version") != v7.VERSION
            or report.get("status") != "failed"
            or report.get("explanation_eligible") is not False
            or report.get("formal_ready") is not False
            or not isinstance(bindings, Mapping)
            or bindings.get("actor_file_sha256") != file_hash(actor_path)
            or bindings.get("protocol_file_sha256") != file_hash(protocol_path)
            or bindings.get("manifest_file_sha256") != file_hash(manifest_path)
            or bindings.get("development_supplement_file_sha256")
                != file_hash(development_supplement_path)
            or bindings.get("designation_sha256")
                != designation_binding.designation.SUPERSEDES_DESIGNATION_SHA256
            or not isinstance(artifacts, Mapping)
            or artifacts.get("rows.npz") != file_hash(rows_path)
            or not isinstance(accounting, Mapping)
            or not isinstance(execution, Mapping)
            or execution.get("final_test_accessed") is not False
            or execution.get("runtime_action_overrides") != 0
            or execution.get("actor_changed") is not False
            or execution.get("environment_steps")
                != accounting.get("raw_collection_environment_steps")
            or execution.get("derived_environment_steps")
                != accounting.get("raw_collection_environment_steps")):
        raise ValueError("Exact historical v7 row report differs")


def _authenticate(
    *, actor_path: Path, protocol_path: Path, manifest_path: Path,
    designation_path: Path, development_supplement_path: Path,
    source_report_path: Path, rows_path: Path,
    designation_original_path: Path | None = None,
    designation_snapshot_components: Mapping[str, Path] | None = None,
    designation_original_components: Mapping[str, Path] | None = None,
    sources: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if file_hash(source_report_path) != EXPECTED_SOURCE_REPORT_SHA256:
        raise ValueError("Exact historical v7 report bytes required")
    if file_hash(rows_path) != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError("Exact historical v7 row bytes required")
    if (file_hash(development_supplement_path)
            != EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256):
        raise ValueError("Exact historical development supplement bytes required")
    if (designation_original_path is None
            or designation_snapshot_components is None
            or designation_original_components is None
            or sources is None):
        raise ValueError("Immutable prior-row authentication snapshot required")
    if producer_sources() != dict(sources):
        raise RuntimeError("Prior-row sources changed before authentication")
    actor = NumPyNativeActor(actor_path)
    designation = designation_binding.read_bound_designation_snapshot(
        designation_path,
        original_path=designation_original_path,
        components=designation_snapshot_components,
        original_components=designation_original_components,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    protocol = v7._read(protocol_path, "training protocol")
    if (designation["bindings"]["actor_sha256"] != file_hash(actor_path)
            or designation["bindings"]["actor_parameters_sha256"]
                != actor.metadata["actor_parameters_sha256"]
            or designation["bindings"]["protocol_file_sha256"]
                != file_hash(protocol_path)
            or designation["bindings"]["protocol_content_sha256"]
                != digest(protocol)):
        raise ValueError("Prior-row designation v2 binding differs")
    manifest = manifest_binding.read_saved_manifest(
        manifest_path, actor_path=actor_path, replay_scope="development")
    frozen_actor = manifest.get("frozen_actor")
    if frozen_actor != {
        "sha256": actor.artifact_sha256,
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
    }:
        raise ValueError("Prior-row manifest/Actor identity differs")
    supplement = _read(development_supplement_path, "development supplement")
    supplement_bindings = supplement.get("bindings")
    if (
        supplement.get("version") != supplement_api.VERSION
        or supplement.get("status") != "passed"
        or canonical(supplement.get("contract"))
        != canonical(supplement_api.contract())
        or supplement.get("program_access") is not False
        or supplement.get("final_audit_rows_access") is not False
        or supplement.get("formal_ready") is not False
        or supplement.get("content_sha256") != digest({
            key: value for key, value in supplement.items()
            if key != "content_sha256"
        })
        or not isinstance(supplement_bindings, Mapping)
        or supplement_bindings.get("actor_sha256") != file_hash(actor_path)
        or supplement_bindings.get("source_manifest_sha256")
            != file_hash(manifest_path)
        or supplement_bindings.get("source_manifest_content_sha256")
            != manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256
        or supplement_bindings.get("contract_sha256")
            != digest(supplement_api.contract())
        or supplement_bindings.get("producer_sources_sha256")
            != digest(supplement_api.producer_sources())
    ):
        raise ValueError("Exact development supplement differs")
    train_scenes = deepcopy([
        *manifest["splits"]["train"],
        *manifest["splits"]["conflict_validation"],
    ])
    validation_scenes = deepcopy(supplement.get("scenes", []))
    if len(validation_scenes) != supplement_api.TOTAL_SCENES:
        raise ValueError("Exact development validation scene count differs")
    protected_development = {
        row["fingerprint"]
        for name in ("tutorial", "question_bank")
        for row in manifest["splits"][name]
    }
    train_fingerprints = {row.get("fingerprint") for row in train_scenes}
    validation_fingerprints = {
        row.get("fingerprint") for row in validation_scenes
    }
    if (
        len(train_fingerprints) != len(train_scenes)
        or len(validation_fingerprints) != len(validation_scenes)
        or train_fingerprints & validation_fingerprints
        or (train_fingerprints | validation_fingerprints)
            & protected_development
    ):
        raise ValueError("Prior development scene registries overlap protected scenes")
    source_report = _read(source_report_path, "historical v7 report")
    _validate_source_report(
        source_report, actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path,
        development_supplement_path=development_supplement_path,
        rows_path=rows_path)
    arrays = v7._load_npz(rows_path)
    v7._validate_arrays(
        arrays, actor=actor, train_scenes=train_scenes,
        validation_scenes=validation_scenes)
    accounting = source_report["row_accounting"]
    if (source_report.get("rows") != len(arrays["observations"])
            or source_report.get("train_rows")
                != int(np.sum(~arrays["split_validation"]))
            or source_report.get("validation_rows")
                != int(np.sum(arrays["split_validation"]))
            or source_report.get("train_scene_count") != len(train_scenes)
            or source_report.get("validation_scene_count")
                != len(validation_scenes)
            or accounting.get("raw_train_rows", 0)
                - accounting.get("train_rows_removed_for_exact_validation_overlap", 0)
                != int(np.sum(~arrays["split_validation"]))
            or accounting.get("raw_validation_rows")
                != int(np.sum(arrays["split_validation"]))):
        raise ValueError("Historical v7 row accounting differs")
    sources = deepcopy(dict(sources))
    report = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "actor_file_sha256": file_hash(actor_path),
            "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
            "protocol_file_sha256": file_hash(protocol_path),
            "protocol_content_sha256": digest(protocol),
            "manifest_file_sha256": file_hash(manifest_path),
            "manifest_content_sha256": manifest["content_sha256"],
            "designation_file_sha256": file_hash(designation_path),
            "development_supplement_file_sha256": file_hash(
                development_supplement_path),
            "source_v7_report_file_sha256": file_hash(source_report_path),
            "source_v7_report_semantic_sha256": digest(source_report),
            "source_v7_rows_file_sha256": file_hash(rows_path),
            "source_v7_rows_semantic_sha256": rows_api.arrays_digest(arrays),
            "runtime_sources_sha256": manifest_binding.runtime_sources()[
                "sources_sha256"],
            "producer_sources_sha256": digest(sources),
            "protected_final_identity_sha256": (
                manifest_binding.EXPECTED_FINAL_IDENTITY_SHA256),
        },
        "rows": len(arrays["observations"]),
        "ordinary_rows": int(np.sum(v7._decode(arrays["kinds"]) == "ordinary")),
        "intervention_rows": int(np.sum(
            v7._decode(arrays["kinds"]) == "intervention")),
        "train_rows": int(np.sum(~arrays["split_validation"])),
        "validation_rows": int(np.sum(arrays["split_validation"])),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "row_accounting": deepcopy(accounting),
        "evidence_artifacts": {
            "rows.npz": file_hash(rows_path),
            "source_v7_report.json": file_hash(source_report_path),
        },
        "validation": {
            "all_actor_probabilities_and_actions_exact": True,
            "all_submitted_actions_equal_policy_actions": True,
            "scene_episode_intervention_split_complete": True,
            "v7_program_refit": False,
            "historical_final_overlap_check_deferred_to_claim": True,
            "final_identity_commitment_used_for_overlap_exclusion": False,
            "final_scene_geometry_access": False,
            "final_trajectories_access": False,
            "final_actor_outputs_or_labels_access": False,
            "final_used_for_fit_selection_metrics": False,
        },
        "sources": sources,
        "final_identity_commitment_used_for_overlap_exclusion": False,
        "final_scene_geometry_access": False,
        "final_trajectories_access": False,
        "final_actor_outputs_or_labels_access": False,
        "final_used_for_fit_selection_metrics": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    return report, arrays


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _copy(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(source_fd, "rb") as incoming, os.fdopen(
                destination_fd, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    development_supplement_path: str | Path,
    source_report_path: str | Path, source_rows_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    originals = {
        "actor": _regular(actor_path, "Actor"),
        "protocol": _regular(protocol_path, "protocol"),
        "manifest": _regular(manifest_path, "manifest"),
        "designation": _regular(designation_path, "designation"),
        "supplement": _regular(development_supplement_path, "development supplement"),
        "source_report": _regular(source_report_path, "historical v7 report"),
        "source_rows": _regular(source_rows_path, "historical v7 rows", 512 * 1024 * 1024),
    }
    originals["manifest_validation"] = _manifest_validation_path(
        originals["manifest"])
    components = _resolved_designation_components(originals["designation"])
    originals.update({"designation_" + name: path
                      for name, path in components.items()})
    if (components["actor"] != originals["actor"]
            or components["protocol"] != originals["protocol"]):
        raise ValueError("Explicit Actor/protocol differ from designation registry")
    destination = Path(output).expanduser().absolute()
    parent = destination.parent
    if (not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Prior-row destination is unsafe or already exists")
    lock = parent / ("." + destination.name + ".lock")
    lock_fd: int | None = None
    temporary: Path | None = None
    try:
        lock_fd = os.open(
            lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with ImmutableInputSnapshot(
            originals,
            expected_sha256=_expected_snapshot_hashes(),
            relative_names={
                "manifest": "manifest/manifest.json",
                "manifest_validation": "manifest/validation.json",
            },
            maximum_bytes={"source_rows": 512 * 1024 * 1024},
            prefix="warehouse-r41-prior-rows-inputs-",
        ) as frozen:
            paths = frozen.paths
            snapshot_components = {
                name: paths["designation_" + name] for name in components
            }
            report, _ = _authenticate(
                actor_path=paths["actor"], protocol_path=paths["protocol"],
                manifest_path=paths["manifest"],
                designation_path=paths["designation"],
                development_supplement_path=paths["supplement"],
                source_report_path=paths["source_report"],
                rows_path=paths["source_rows"],
                designation_original_path=originals["designation"],
                designation_snapshot_components=snapshot_components,
                designation_original_components=components,
                sources=sources,
            )
            frozen.verify()
            if producer_sources() != sources:
                raise RuntimeError(
                    "Prior-row sources changed during semantic authentication")
            temporary = Path(tempfile.mkdtemp(
                prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
            os.chmod(temporary, 0o700)
            _copy(paths["source_rows"], temporary / "rows.npz")
            _copy(paths["source_report"], temporary / "source_v7_report.json")
            _write_json(temporary / "report.json", report)
            report_file_sha256 = sha256(
                (canonical(report) + "\n").encode("utf-8")).hexdigest()
            if (file_hash(temporary / "rows.npz")
                    != frozen.sha256["source_rows"]
                    or file_hash(temporary / "source_v7_report.json")
                        != frozen.sha256["source_report"]):
                raise RuntimeError(
                    "Prior-row staged evidence differs from fixed inputs")
            directory_fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            frozen.verify()
            if producer_sources() != sources:
                raise RuntimeError("Prior-row sources changed during publication")
            if destination.exists() or destination.is_symlink():
                raise ValueError("Prior-row destination appeared during build")
            os.rename(temporary, destination)
            temporary = None
            try:
                frozen.verify()
                if producer_sources() != sources:
                    raise RuntimeError(
                        "Prior-row sources changed during publication")
                if (file_hash(destination / "rows.npz")
                        != frozen.sha256["source_rows"]
                        or file_hash(destination / "source_v7_report.json")
                            != frozen.sha256["source_report"]
                        or file_hash(destination / "report.json")
                            != report_file_sha256):
                    raise RuntimeError("Published prior-row evidence differs")
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise
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
    return deepcopy(report)


def read_saved_artifacts(
    *, report_path: str | Path, rows_path: str | Path,
    embedded_source_report_path: str | Path, expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    development_supplement_path: str | Path,
    source_report_path: str | Path, source_rows_path: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    try:
        read_authenticated_bytes(
            report_path, label="prior-row report",
            expected_sha256=expected_report_sha256)
    except ValueError:
        raise ValueError("Prior-row report hash differs") from None
    originals = {
        "saved_report": _regular(report_path, "prior-row report"),
        "saved_rows": _regular(rows_path, "prior-row rows"),
        "embedded_source_report": _regular(
            embedded_source_report_path, "embedded historical v7 report"),
        "actor": _regular(actor_path, "Actor"),
        "protocol": _regular(protocol_path, "protocol"),
        "manifest": _regular(manifest_path, "manifest"),
        "designation": _regular(designation_path, "designation"),
        "supplement": _regular(
            development_supplement_path, "development supplement"),
        "source_report": _regular(source_report_path, "historical v7 report"),
        "source_rows": _regular(source_rows_path, "historical v7 rows"),
    }
    originals["manifest_validation"] = _manifest_validation_path(
        originals["manifest"])
    components = _resolved_designation_components(originals["designation"])
    originals.update({"designation_" + name: path
                      for name, path in components.items()})
    if (components["actor"] != originals["actor"]
            or components["protocol"] != originals["protocol"]):
        raise ValueError("Explicit Actor/protocol differ from designation registry")
    with ImmutableInputSnapshot(
        originals,
        expected_sha256=_expected_snapshot_hashes(
            expected_report_sha256=expected_report_sha256),
        relative_names={
            "manifest": "manifest/manifest.json",
            "manifest_validation": "manifest/validation.json",
        },
        maximum_bytes={
            "saved_rows": 512 * 1024 * 1024,
            "source_rows": 512 * 1024 * 1024,
        },
        prefix="warehouse-r41-prior-rows-reader-inputs-",
    ) as frozen:
        paths = frozen.paths
        snapshot_components = {
            name: paths["designation_" + name] for name in components
        }
        saved = _read_saved_artifacts_snapshot(
            paths=paths,
            designation_original_path=originals["designation"],
            designation_snapshot_components=snapshot_components,
            designation_original_components=components,
            sources=sources,
        )
        frozen.verify()
        if producer_sources() != sources:
            raise RuntimeError(
                "Prior-row sources changed during saved-report return")
        return deepcopy(saved)


def _read_saved_artifacts_snapshot(
    *, paths: Mapping[str, Path], designation_original_path: Path,
    designation_snapshot_components: Mapping[str, Path],
    designation_original_components: Mapping[str, Path],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Authenticate already immutable paths; the owning transaction guards them."""
    required = {
        "saved_report", "saved_rows", "embedded_source_report", "actor",
        "protocol", "manifest", "manifest_validation", "designation",
        "supplement", "source_report", "source_rows",
    }
    if not required.issubset(paths):
        raise ValueError("Complete immutable prior-row artifact set required")
    expected, _ = _authenticate(
        actor_path=paths["actor"], protocol_path=paths["protocol"],
        manifest_path=paths["manifest"],
        designation_path=paths["designation"],
        development_supplement_path=paths["supplement"],
        source_report_path=paths["source_report"],
        rows_path=paths["source_rows"],
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
        sources=sources,
    )
    saved = _read(paths["saved_report"], "prior-row report")
    if saved.get("evidence_artifacts") != {
        "rows.npz": EXPECTED_SOURCE_ROWS_SHA256,
        "source_v7_report.json": EXPECTED_SOURCE_REPORT_SHA256,
    }:
        raise ValueError("Prior-row embedded artifact registry differs")
    if canonical(saved) != canonical(expected):
        raise ValueError("Prior-row report differs from live reauthentication")
    return deepcopy(saved)


def read_saved_report(
    output: str | Path, *, expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    development_supplement_path: str | Path,
    source_report_path: str | Path, source_rows_path: str | Path,
) -> dict[str, Any]:
    directory = Path(output).expanduser().absolute()
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        raise ValueError("Prior-row evidence directory is unsafe")
    return read_saved_artifacts(
        report_path=directory / "report.json",
        rows_path=directory / "rows.npz",
        embedded_source_report_path=directory / "source_v7_report.json",
        expected_report_sha256=expected_report_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        development_supplement_path=development_supplement_path,
        source_report_path=source_report_path, source_rows_path=source_rows_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--development-supplement", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        development_supplement_path=args.development_supplement,
        source_report_path=args.source_report, source_rows_path=args.source_rows,
        output=args.output)
    print(canonical({"status": report["status"], "rows": report["rows"],
                     "output": str(Path(args.output).expanduser().absolute())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "EXPECTED_SOURCE_REPORT_SHA256",
    "EXPECTED_SOURCE_ROWS_SHA256", "EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256",
    "contract", "producer_sources", "build",
    "read_saved_artifacts", "read_saved_report", "main",
]
