"""Reauthenticate retained fit rows and the fresh diagnostic v8 outer rows.

The source collection is rebuilt from the current identity-frozen development
registry. This producer authenticates that collector receipt, replays its
fixed collection schedule, checks disjointness from the exact prior rows, and
freezes the exact pair for the RCPD fitter. It never reads final-test data.
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

from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as expansion_api
from backend.training import warehouse_r41_diagnostic_expansion_collection_v8 as collection_api
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
    read_authenticated_bytes,
)
from backend.training import warehouse_r41_diagnostic_rows_v8 as rows_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-expansion-rows.v8-fresh-collection"
STATUS = "passed_fresh_expansion_rows_reauthentication"
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_COLLECTION_REPORT_SHA256 = (
    "4272f3b1a0c6b975787aad933270559456d2a802c34f91c5580456fbb80a7994"
)
EXPECTED_SOURCE_ROWS_SHA256 = (
    "7c80beb23c840dcdc71ccf340cbdad87aaced4388c5fe174179202524d2b6bba"
)
EXPECTED_PRIOR_ROWS_SHA256 = (
    "a21bd052a8bed48bb103a14e7da6a1f21291af7cac475928686d38693bd52686"
)


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_collection_report_sha256": (
            EXPECTED_SOURCE_COLLECTION_REPORT_SHA256),
        "source_rows_sha256": EXPECTED_SOURCE_ROWS_SHA256,
        "prior_rows_sha256": EXPECTED_PRIOR_ROWS_SHA256,
        "source_layout": "expansion_only",
        "validation": [
            "fresh collector receipt and fixed schedule replay",
            "exact row archive schema and observation hashes",
            "all probabilities and deterministic actions from fixed Actor",
            "exact expansion scene and episode population",
            "complete validation trajectories and intervention matrices",
            "scene-identity disjointness from exact authenticated prior rows",
            "public critical-group derivation and action authority",
        ],
        "historical_final_overlap_check": (
            "deferred to the irrevocable claim-bound exclusion phase"
        ),
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
                "Expansion-row fixed input changed during authentication: " + name)
        result[name] = file_hash(path)
    return result


def _integrity_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
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
            "Expansion-row sources or fixed inputs changed during " + phase
        ) from error
    if (current_sources != snapshot.get("producer_sources")
            or current_inputs != snapshot.get("fixed_input_sha256")):
        raise RuntimeError(
            "Expansion-row sources or fixed inputs changed during " + phase)


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

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " cannot be read") from error
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
    raw = read_authenticated_bytes(
        designation_path, label="diagnostic Actor designation",
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    return designation_binding.resolve_bound_components_from_bytes(
        raw, original_path=designation_path,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )


def _expected_snapshot_hashes(
    *, expected_expansion_registry_sha256: str,
    expected_expansion_report_sha256: str,
    expected_saved_report_sha256: str | None = None,
) -> dict[str, str]:
    result = {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "expansion_registry": expected_expansion_registry_sha256,
        "expansion_report": expected_expansion_report_sha256,
        "previous": expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256,
        "source_collection_report": (
            EXPECTED_SOURCE_COLLECTION_REPORT_SHA256),
        "source_rows": EXPECTED_SOURCE_ROWS_SHA256,
        "prior_rows": EXPECTED_PRIOR_ROWS_SHA256,
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
    if expected_saved_report_sha256 is not None:
        result.update({
            "saved_report": expected_saved_report_sha256,
            "embedded_collection_report": (
                EXPECTED_SOURCE_COLLECTION_REPORT_SHA256),
            "embedded_rows": EXPECTED_SOURCE_ROWS_SHA256,
        })
    return result


def _copy_exclusive(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(source_fd, "rb") as incoming, os.fdopen(
                destination_fd, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _validate(
    *, actor_path: Path, protocol_path: Path, manifest_path: Path,
    designation_path: Path, expansion_registry_path: Path,
    expected_expansion_registry_sha256: str, expansion_report_path: Path,
    expected_expansion_report_sha256: str, previous_development_path: Path,
    source_collection_report_path: Path, source_rows_path: Path,
    prior_rows_path: Path,
    designation_original_path: Path | None = None,
    designation_snapshot_components: Mapping[str, Path] | None = None,
    designation_original_components: Mapping[str, Path] | None = None,
    sources: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if (file_hash(source_collection_report_path)
            != EXPECTED_SOURCE_COLLECTION_REPORT_SHA256):
        raise ValueError("Exact fresh expansion collection report required")
    if file_hash(source_rows_path) != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError("Exact fresh expansion row bytes required")
    if file_hash(prior_rows_path) != EXPECTED_PRIOR_ROWS_SHA256:
        raise ValueError("Exact historical prior row bytes required")
    if file_hash(expansion_registry_path) != expected_expansion_registry_sha256:
        raise ValueError("Development expansion registry hash differs")
    if file_hash(expansion_report_path) != expected_expansion_report_sha256:
        raise ValueError("Development expansion report hash differs")
    if (designation_original_path is None
            or designation_snapshot_components is None
            or designation_original_components is None
            or sources is None):
        raise ValueError("Immutable expansion-row authentication snapshot required")
    if producer_sources() != dict(sources):
        raise RuntimeError("Expansion-row sources changed before authentication")
    actor = NumPyNativeActor(actor_path)
    if actor.artifact_sha256 != designation_binding.designation.EXPECTED_ACTOR_SHA256:
        raise ValueError("Expansion rows require the fixed diagnostic Actor")
    designation = designation_binding.read_bound_designation_snapshot(
        designation_path,
        original_path=designation_original_path,
        components=designation_snapshot_components,
        original_components=designation_original_components,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    if (designation["bindings"]["actor_sha256"] != file_hash(actor_path)
            or designation["bindings"]["protocol_file_sha256"]
                != file_hash(protocol_path)):
        raise ValueError("Expansion-row designation binding differs")
    manifest = manifest_binding.read_saved_manifest(
        manifest_path, actor_path=actor_path, replay_scope="development")
    previous = _read(previous_development_path, "previous development supplement")
    expansion = _read(expansion_registry_path, "development expansion registry")
    expansion_report = _read(expansion_report_path, "development expansion report")
    if (expansion_report.get("version") != expansion_api.VERSION
            or expansion_report.get("status") != expansion_api.STATUS
            or expansion_report.get("content_sha256") != digest({
                key: value for key, value in expansion_report.items()
                if key != "content_sha256"
            })
            or expansion.get("content_sha256") != digest({
                key: value for key, value in expansion.items()
                if key != "content_sha256"
            })
            or expansion_report.get("registry_file_sha256")
                != file_hash(expansion_registry_path)
            or expansion_report.get("registry_content_sha256")
                != expansion.get("content_sha256")
            or expansion_report.get("producer_sources")
                != expansion_api.producer_sources()
            or expansion_report.get("bindings") != expansion.get("bindings")
            or expansion.get("bindings", {}).get("designation_file_sha256")
                != file_hash(designation_path)
            or expansion.get("bindings", {}).get("source_manifest_file_sha256")
                != file_hash(manifest_path)):
        raise ValueError("Development expansion provenance differs")
    fit_scenes, validation_scenes = rows_api._validate_expansion_registry(
        expansion, registry_path=expansion_registry_path, actor=actor,
        manifest_path=manifest_path, designation_path=designation_path,
        prior_report={"bindings": {
            "development_supplement_file_sha256": file_hash(
                previous_development_path),
        }},
    )
    source_arrays = rows_api._load_npz(
        source_rows_path, "fresh development expansion rows")
    prior_arrays = rows_api._load_npz(
        prior_rows_path, "historical prior development rows")
    rows_api._validate_base_shapes(prior_arrays, actor, "Prior development rows")
    relations = R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])
    collection_paths = {
        "saved_report": source_collection_report_path,
        "saved_rows": source_rows_path,
        "actor": actor_path,
        "protocol": protocol_path,
        "manifest": manifest_path,
        "manifest_validation": manifest_path.parent / "validation.json",
        "designation": designation_path,
        "expansion_registry": expansion_registry_path,
        "expansion_report": expansion_report_path,
        **{"designation_" + name: path
           for name, path in designation_snapshot_components.items()},
    }
    collection_report = collection_api._validate_saved_snapshot(
        paths=collection_paths,
        designation_original=designation_original_path,
        component_originals=designation_original_components,
        sources=collection_api.producer_sources(),
    )
    layout = rows_api._validate_expansion_rows(
        source_arrays, actor=actor, fit_scenes=fit_scenes,
        validation_scenes=validation_scenes, prior_arrays=prior_arrays,
        relations=relations,
    )
    if layout != "expansion_only":
        raise ValueError("Fresh expansion rows must use expansion-only layout")
    source_fingerprints = set(map(str, rows_api._decode(
        source_arrays["scene_fingerprints"], "Expansion scenes")))
    prior_fingerprints = set(map(str, rows_api._decode(
        prior_arrays["scene_fingerprints"], "Prior scenes")))
    if source_fingerprints & prior_fingerprints:
        raise ValueError("Fresh expansion and prior scene identities overlap")
    sources = deepcopy(dict(sources))
    receipt = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "actor_file_sha256": file_hash(actor_path),
            "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
            "protocol_file_sha256": file_hash(protocol_path),
            "manifest_file_sha256": file_hash(manifest_path),
            "manifest_content_sha256": manifest["content_sha256"],
            "designation_file_sha256": file_hash(designation_path),
            "expansion_registry_file_sha256": file_hash(expansion_registry_path),
            "expansion_registry_content_sha256": expansion["content_sha256"],
            "expansion_report_file_sha256": file_hash(expansion_report_path),
            "expansion_report_semantic_sha256": digest(expansion_report),
            "previous_development_file_sha256": file_hash(
                previous_development_path),
            "previous_development_semantic_sha256": digest(previous),
            "source_collection_report_file_sha256": file_hash(
                source_collection_report_path),
            "source_collection_report_semantic_sha256": digest(
                collection_report),
            "source_rows_file_sha256": file_hash(source_rows_path),
            "source_rows_semantic_sha256": rows_api.arrays_digest(source_arrays),
            "prior_rows_file_sha256": file_hash(prior_rows_path),
            "prior_rows_semantic_sha256": rows_api.arrays_digest(prior_arrays),
            "runtime_sources_sha256": manifest_binding.runtime_sources()[
                "sources_sha256"],
            "producer_sources_sha256": digest(sources),
        },
        "validation": {
            "layout": layout,
            "rows": len(source_arrays["observations"]),
            "fit_rows": int(np.sum(~source_arrays["split_validation"])),
            "validation_rows": int(np.sum(source_arrays["split_validation"])),
            "scene_count": len(set(map(
                bytes, source_arrays["scene_fingerprints"]))),
            "all_actor_probabilities_and_actions_exact": True,
            "all_submitted_actions_equal_policy_actions": True,
            "fresh_collection_schedule_replayed": True,
            "prior_scene_identity_disjointness_verified": True,
            "historical_final_overlap_check_deferred_to_claim": True,
        },
        "sources": sources,
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
    receipt["content_sha256"] = digest(receipt)
    return receipt, source_arrays


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    previous_development_path: str | Path,
    source_collection_report_path: str | Path,
    source_rows_path: str | Path,
    prior_rows_path: str | Path, output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    originals = {
        "actor": _regular(actor_path, "Actor"),
        "protocol": _regular(protocol_path, "protocol"),
        "manifest": _regular(manifest_path, "manifest"),
        "designation": _regular(designation_path, "designation"),
        "expansion_registry": _regular(
            expansion_registry_path, "expansion registry"),
        "expansion_report": _regular(expansion_report_path, "expansion report"),
        "previous": _regular(previous_development_path, "previous development"),
        "source_collection_report": _regular(
            source_collection_report_path, "source expansion collection report"),
        "source_rows": _regular(
            source_rows_path, "source expansion rows",
            rows_api.MAX_NPZ_COMPRESSED_BYTES),
        "prior_rows": _regular(
            prior_rows_path, "prior rows", rows_api.MAX_NPZ_COMPRESSED_BYTES),
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
        raise ValueError("Expansion-row destination is unsafe or already exists")
    lock = parent / ("." + destination.name + ".lock")
    lock_fd: int | None = None
    temporary: Path | None = None
    try:
        lock_fd = os.open(
            lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with ImmutableInputSnapshot(
            originals,
            expected_sha256=_expected_snapshot_hashes(
                expected_expansion_registry_sha256=(
                    expected_expansion_registry_sha256),
                expected_expansion_report_sha256=(
                    expected_expansion_report_sha256),
            ),
            relative_names={
                "manifest": "manifest/manifest.json",
                "manifest_validation": "manifest/validation.json",
            },
            maximum_bytes={
                "source_rows": rows_api.MAX_NPZ_COMPRESSED_BYTES,
                "prior_rows": rows_api.MAX_NPZ_COMPRESSED_BYTES,
            },
            prefix="warehouse-r41-expansion-rows-inputs-",
        ) as frozen:
            paths = frozen.paths
            snapshot_components = {
                name: paths["designation_" + name] for name in components
            }
            receipt, _ = _validate(
                actor_path=paths["actor"], protocol_path=paths["protocol"],
                manifest_path=paths["manifest"],
                designation_path=paths["designation"],
                expansion_registry_path=paths["expansion_registry"],
                expected_expansion_registry_sha256=(
                    expected_expansion_registry_sha256),
                expansion_report_path=paths["expansion_report"],
                expected_expansion_report_sha256=(
                    expected_expansion_report_sha256),
                previous_development_path=paths["previous"],
                source_collection_report_path=paths[
                    "source_collection_report"],
                source_rows_path=paths["source_rows"],
                prior_rows_path=paths["prior_rows"],
                designation_original_path=originals["designation"],
                designation_snapshot_components=snapshot_components,
                designation_original_components=components,
                sources=sources,
            )
            frozen.verify()
            if producer_sources() != sources:
                raise RuntimeError(
                    "Expansion-row sources changed during semantic authentication")
            temporary = Path(tempfile.mkdtemp(
                prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
            os.chmod(temporary, 0o700)
            _copy_exclusive(
                paths["source_collection_report"],
                temporary / "source_collection_report.json")
            _copy_exclusive(paths["source_rows"], temporary / "expansion_rows.npz")
            _write_json(temporary / "report.json", receipt)
            report_file_sha256 = sha256(
                (canonical(receipt) + "\n").encode("utf-8")).hexdigest()
            if (file_hash(temporary / "source_collection_report.json")
                    != frozen.sha256["source_collection_report"]
                    or file_hash(temporary / "expansion_rows.npz")
                        != frozen.sha256["source_rows"]):
                raise RuntimeError(
                    "Staged expansion row evidence differs from fixed inputs")
            directory_fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            frozen.verify()
            if producer_sources() != sources:
                raise RuntimeError("Expansion-row sources changed during publication")
            if destination.exists() or destination.is_symlink():
                raise ValueError("Expansion-row destination appeared during build")
            os.rename(temporary, destination)
            temporary = None
            try:
                frozen.verify()
                if producer_sources() != sources:
                    raise RuntimeError(
                        "Expansion-row sources changed during publication")
                if (file_hash(destination / "source_collection_report.json")
                        != frozen.sha256["source_collection_report"]
                        or file_hash(destination / "expansion_rows.npz")
                            != frozen.sha256["source_rows"]
                        or file_hash(destination / "report.json")
                            != report_file_sha256):
                    raise RuntimeError("Published expansion-row evidence differs")
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
    if (file_hash(destination / "source_collection_report.json")
            != EXPECTED_SOURCE_COLLECTION_REPORT_SHA256
            or file_hash(destination / "expansion_rows.npz")
                != EXPECTED_SOURCE_ROWS_SHA256):
        raise RuntimeError("Published expansion row bytes differ")
    return deepcopy(receipt)


def read_saved_artifacts(
    *, report_path: str | Path, rows_path: str | Path,
    collection_report_path: str | Path,
    expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    previous_development_path: str | Path,
    source_collection_report_path: str | Path,
    source_rows_path: str | Path, prior_rows_path: str | Path,
) -> dict[str, Any]:
    """Reauthenticate an exact receipt and embedded fresh collection pair."""
    sources = producer_sources()
    try:
        read_authenticated_bytes(
            report_path, label="expansion-row report",
            expected_sha256=expected_report_sha256)
    except ValueError:
        raise ValueError("Expansion-row report hash differs") from None
    originals = {
        "saved_report": _regular(report_path, "expansion-row report"),
        "embedded_rows": _regular(
            rows_path, "embedded expansion rows",
            rows_api.MAX_NPZ_COMPRESSED_BYTES),
        "embedded_collection_report": _regular(
            collection_report_path, "embedded expansion collection report"),
        "actor": _regular(actor_path, "Actor"),
        "protocol": _regular(protocol_path, "protocol"),
        "manifest": _regular(manifest_path, "manifest"),
        "designation": _regular(designation_path, "designation"),
        "expansion_registry": _regular(
            expansion_registry_path, "expansion registry"),
        "expansion_report": _regular(
            expansion_report_path, "expansion report"),
        "previous": _regular(previous_development_path, "previous development"),
        "source_collection_report": _regular(
            source_collection_report_path, "source expansion collection report"),
        "source_rows": _regular(
            source_rows_path, "source expansion rows",
            rows_api.MAX_NPZ_COMPRESSED_BYTES),
        "prior_rows": _regular(
            prior_rows_path, "prior rows", rows_api.MAX_NPZ_COMPRESSED_BYTES),
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
            expected_expansion_registry_sha256=(
                expected_expansion_registry_sha256),
            expected_expansion_report_sha256=(
                expected_expansion_report_sha256),
            expected_saved_report_sha256=expected_report_sha256,
        ),
        relative_names={
            "manifest": "manifest/manifest.json",
            "manifest_validation": "manifest/validation.json",
        },
        maximum_bytes={
            "embedded_rows": rows_api.MAX_NPZ_COMPRESSED_BYTES,
            "source_rows": rows_api.MAX_NPZ_COMPRESSED_BYTES,
            "prior_rows": rows_api.MAX_NPZ_COMPRESSED_BYTES,
        },
        prefix="warehouse-r41-expansion-rows-reader-inputs-",
    ) as frozen:
        paths = frozen.paths
        snapshot_components = {
            name: paths["designation_" + name] for name in components
        }
        saved = _read_saved_artifacts_snapshot(
            paths=paths,
            expected_expansion_registry_sha256=(
                expected_expansion_registry_sha256),
            expected_expansion_report_sha256=(
                expected_expansion_report_sha256),
            designation_original_path=originals["designation"],
            designation_snapshot_components=snapshot_components,
            designation_original_components=components,
            sources=sources,
        )
        frozen.verify()
        if producer_sources() != sources:
            raise RuntimeError(
                "Expansion-row sources changed during saved-report return")
        return deepcopy(saved)


def _read_saved_artifacts_snapshot(
    *, paths: Mapping[str, Path],
    expected_expansion_registry_sha256: str,
    expected_expansion_report_sha256: str,
    designation_original_path: Path,
    designation_snapshot_components: Mapping[str, Path],
    designation_original_components: Mapping[str, Path],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Authenticate already immutable paths; the owning transaction guards them."""
    required = {
        "saved_report", "embedded_rows", "actor", "protocol", "manifest",
        "manifest_validation", "designation", "expansion_registry",
        "expansion_report", "previous", "embedded_collection_report",
        "source_collection_report", "source_rows", "prior_rows",
    }
    if not required.issubset(paths):
        raise ValueError("Complete immutable expansion-row artifact set required")
    expected, _ = _validate(
        actor_path=paths["actor"], protocol_path=paths["protocol"],
        manifest_path=paths["manifest"],
        designation_path=paths["designation"],
        expansion_registry_path=paths["expansion_registry"],
        expected_expansion_registry_sha256=expected_expansion_registry_sha256,
        expansion_report_path=paths["expansion_report"],
        expected_expansion_report_sha256=expected_expansion_report_sha256,
        previous_development_path=paths["previous"],
        source_collection_report_path=paths["source_collection_report"],
        source_rows_path=paths["source_rows"],
        prior_rows_path=paths["prior_rows"],
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
        sources=sources,
    )
    saved = _read(paths["saved_report"], "expansion-row report")
    if (file_hash(paths["embedded_collection_report"])
            != EXPECTED_SOURCE_COLLECTION_REPORT_SHA256
            or file_hash(paths["embedded_rows"]) != EXPECTED_SOURCE_ROWS_SHA256):
        raise ValueError("Embedded fresh expansion collection differs")
    if canonical(saved) != canonical(expected):
        raise ValueError("Expansion-row report differs from live reauthentication")
    return deepcopy(saved)


def read_saved_report(
    output: str | Path, *, expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    previous_development_path: str | Path,
    source_collection_report_path: str | Path,
    source_rows_path: str | Path, prior_rows_path: str | Path,
) -> dict[str, Any]:
    directory = Path(output).expanduser().absolute()
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        raise ValueError("Expansion-row evidence directory is unsafe")
    return read_saved_artifacts(
        report_path=directory / "report.json",
        rows_path=directory / "expansion_rows.npz",
        collection_report_path=directory / "source_collection_report.json",
        expected_report_sha256=expected_report_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expansion_registry_path=expansion_registry_path,
        expected_expansion_registry_sha256=expected_expansion_registry_sha256,
        expansion_report_path=expansion_report_path,
        expected_expansion_report_sha256=expected_expansion_report_sha256,
        previous_development_path=previous_development_path,
        source_collection_report_path=source_collection_report_path,
        source_rows_path=source_rows_path, prior_rows_path=prior_rows_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--expansion-registry", required=True)
    parser.add_argument("--expected-expansion-registry-sha256", required=True)
    parser.add_argument("--expansion-report", required=True)
    parser.add_argument("--expected-expansion-report-sha256", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--source-collection-report", required=True)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--prior-rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        expansion_registry_path=args.expansion_registry,
        expected_expansion_registry_sha256=(
            args.expected_expansion_registry_sha256),
        expansion_report_path=args.expansion_report,
        expected_expansion_report_sha256=args.expected_expansion_report_sha256,
        previous_development_path=args.previous_development,
        source_collection_report_path=args.source_collection_report,
        source_rows_path=args.source_rows, prior_rows_path=args.prior_rows,
        output=args.output,
    )
    print(canonical({
        "status": result["status"],
        "rows": result["validation"]["rows"],
        "output": str(Path(args.output).expanduser().absolute()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "EXPECTED_SOURCE_COLLECTION_REPORT_SHA256",
    "EXPECTED_SOURCE_ROWS_SHA256",
    "EXPECTED_PRIOR_ROWS_SHA256", "contract", "producer_sources", "build",
    "read_saved_artifacts", "read_saved_report", "main",
]
