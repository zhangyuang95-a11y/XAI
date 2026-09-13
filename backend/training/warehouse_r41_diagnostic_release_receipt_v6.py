"""Immutable receipt for one built r4.1 diagnostic Secret File."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable, Mapping
import zipfile

from backend.training import warehouse_r41_diagnostic_admission_v6 as admission_api
from backend.training import warehouse_r41_diagnostic_designation_v2 as designation_api
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_input_snapshot_v8 as input_snapshot_api
from backend.training.warehouse_native_common import canonical, digest, file_hash
from ui import warehouse_alignment_r41_diagnostic_release_v8 as release


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-release-receipt.v6"
STATUS = "completed_internal_diagnostic_release"
EXPECTED_ROLLBACK_VERSION = "warehouse-r41-diagnostic-r3-rollback.v1"
EXPECTED_ROLLBACK_STATUS = "saved_before_diagnostic_deployment"
EXPECTED_ROLLBACK_RECEIPT_SHA256 = (
    "491ccd2eb13f7e50bde492195425272c5128a0b0076f4bdf87ba28021fcfd126"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
CANDIDATE_LINEAGE_FIELDS = frozenset((
    "prior_rows_reauthentication_report_sha256",
    "expansion_rows_reauthentication_report_sha256",
    "expansion_source_collection_report_sha256",
    "development_expansion_registry_sha256",
    "development_expansion_report_sha256",
    "retired_identity_projection_sha256",
    "retired_identity_projection_content_sha256",
    "retired_identity_projection_report_sha256",
    "prior_v7_source_report_sha256", "prior_v7_rows_sha256",
    "expansion_rows_sha256", "v8_fit_config_sha256",
    "v8_fit_selector_report_sha256",
    "v8_fit_selector_fit_only_rows_sha256",
    "v8_fit_selector_scope_sha256",
    "v8_fit_selector_config_registry_sha256",
    "v8_fit_selector_inner_split_audit_sha256",
    "v8_fit_selector_inner_selection_sha256",
    "v8_fit_selector_selected_config_sha256",
    "v8_fit_selector_inner_fit_program_sha256",
    "v8_fit_selector_source_v8_report_sha256",
    "v8_fit_selector_source_v8_rows_sha256",
    "final_rcpd_inputs_sha256", "final_rcpd_report_sha256",
    "final_rcpd_rows_sha256", "final_rcpd_pairs_sha256",
    "final_rcpd_weights_audit_sha256", "final_rcpd_candidate_sha256",
))
FIELDS = frozenset((
    "version", "status", "release_version", "pilot_class",
    "behavior_performance_gate_passed", "behavior_performance_gate_waived",
    "waiver_scope", "formal_ready", "formal_sample_eligible",
    "human_explanation_effect_validated", "data_persistent",
    "runtime_action_override", "actor_sha256", "designation_sha256",
    "admission_sha256", "package_sha256", "manifest_sha256",
    "base64_sha256", "runtime_manifest_signature",
    "portable_runtime_manifest_sha256", "question_bank_sha256",
    "tutorial_sha256",
    "release_sources_sha256", "package_contract_sha256",
    "dynamic_selection_protocol_sha256", "dynamic_selection_prefilter_sha256",
    "dynamic_selection_episodes_sha256",
    "diagnostic_publication_authentication_sha256",
    *CANDIDATE_LINEAGE_FIELDS,
    "program_sha256", "program_content_sha256", "program_identity_sha256",
    "public_feature_contract_sha256", "public_feature_registry_sha256",
    "program_complexity_sha256", "final_once_identity_sha256",
    "final_once_campaign_key", "final_once_permanent_anchor_sha256",
    "final_once_attempt_started_sha256", "final_once_attempt_completed_sha256",
    "final_once_candidate_authenticated_sha256",
    "final_once_holdout_started_sha256",
    "final_once_historical_exclusion_started_sha256",
    "final_once_historical_exclusion_completed_sha256",
    "final_once_holdout_completed_sha256",
    "final_once_audit_started_sha256", "final_once_audit_completed_sha256",
    "fresh_final_v3_exclusion_sha256",
    "fresh_final_v3_exclusion_content_sha256",
    "explanation_audit_inputs_sha256", "explanation_audit_evidence_sha256",
    "explanation_audit_sha256", "physical_replay_sha256",
    "selected_scene_fingerprints", "rollback_receipt_path",
    "rollback_receipt_sha256", "self_path",
))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _file(value: str | Path, label: str) -> tuple[Path, str]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    return path, relative


def _json(path: Path, label: str) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, child in rows:
            if key in value:
                raise ValueError("Duplicate JSON field in " + label)
            value[key] = child
        return value
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _exact_json(path: Path, label: str, expected_sha256: str) -> dict[str, Any]:
    """Parse only the exact bytes named by an authenticated registry hash."""
    expected = _sha(expected_sha256, label)
    canonical_path = _canonical_regular(path, label)
    descriptor = os.open(
        canonical_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(label + " bytes differ")

    def pairs(rows):
        value = {}
        for key, child in rows:
            if key in value:
                raise ValueError("Duplicate JSON field in " + label)
            value[key] = child
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def source_closure() -> dict[str, str]:
    """Return the release-receipt producer closure bound by admission v6."""
    return admission_api.source_closure()


def _canonical_regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    return path


def _release_input_paths(
    *, admission_path: Path, designation_path: Path, package_path: Path,
    base64_path: Path, rollback_path: Path,
    expected_admission_sha256: str, expected_rollback_sha256: str,
) -> dict[str, Path]:
    """Resolve all direct and transitive files consumed while making a receipt."""
    raw_admission = _exact_json(
        admission_path, "diagnostic admission", expected_admission_sha256)
    components = _component_paths_from_admission(raw_admission)
    raw_rollback = _exact_json(
        rollback_path, "r3 rollback receipt", expected_rollback_sha256)
    rollback_files = raw_rollback.get("files")
    if not isinstance(rollback_files, Mapping):
        raise ValueError("R3 rollback artifact registry differs")
    result = {
        "admission": admission_path,
        "designation": designation_path,
        "package": package_path,
        "base64": base64_path,
        "rollback_receipt": rollback_path,
        "permanent_final_anchor": _canonical_regular(
            admission_api.final_once_api._anchor_path(),
            "permanent final-once anchor"),
    }
    result.update({
        "admission_component:" + name: path
        for name, path in sorted(components.items())
    })
    result.update({
        "rollback_artifact:" + name: _canonical_regular(
            rollback_path.parent / name, "r3 rollback artifact " + name)
        for name in sorted(rollback_files)
    })
    if (file_hash(admission_path) != expected_admission_sha256
            or file_hash(rollback_path) != expected_rollback_sha256):
        raise RuntimeError(
            "Diagnostic admission or rollback receipt changed during resolution")
    return result


def _release_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    if not paths:
        raise ValueError("Diagnostic receipt input registry is empty")
    return {
        name: file_hash(_canonical_regular(path, "release input " + name))
        for name, path in sorted(paths.items())
    }


def _guard_release_inputs(
    paths: Mapping[str, Path], *, expected_hashes: Mapping[str, str],
    expected_sources: Mapping[str, str], phase: str,
    resolve_paths: Callable[[], Mapping[str, Path]],
) -> None:
    if source_closure() != dict(expected_sources):
        raise RuntimeError("Diagnostic receipt source changed during " + phase)
    try:
        current_paths = dict(resolve_paths())
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "Diagnostic receipt input registry changed during " + phase
        ) from error
    if (current_paths != dict(paths)
            or _release_input_hashes(paths) != dict(expected_hashes)
            or source_closure() != dict(expected_sources)):
        raise RuntimeError(
            "Diagnostic receipt input or source changed during " + phase)


def _publish_new_receipt(
    path: Path, value: Mapping[str, Any], *, before_publish: Callable[[], None],
) -> None:
    """Fsync a private sibling and atomically publish it after the last guard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError("Diagnostic release receipt parent is unsafe")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    staging = path.with_name(
        "." + path.name + "." + os.urandom(8).hex() + ".partial")
    linked = False
    try:
        descriptor = os.open(
            staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((canonical(value) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        before_publish()
        os.link(staging, path, follow_symlinks=False)
        linked = True
        staging.unlink()
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        staging.unlink(missing_ok=True)
        if linked:
            path.unlink(missing_ok=True)
        raise


def _manifest_sha(package: Path) -> str:
    try:
        with zipfile.ZipFile(package, "r") as archive:
            infos = [row for row in archive.infolist()
                     if row.filename == release.MANIFEST_NAME]
            if (len(infos) != 1 or infos[0].compress_type
                    != release.ARCHIVE_COMPRESSION
                    or not 0 < infos[0].file_size <= release.MAX_MANIFEST_BYTES):
                raise ValueError("Diagnostic package manifest is unsafe")
            with archive.open(infos[0]) as stream:
                raw = stream.read(release.MAX_MANIFEST_BYTES + 1)
            if len(raw) != infos[0].file_size:
                raise ValueError("Diagnostic package manifest is truncated")
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError("Diagnostic package has no valid manifest") from error
    return sha256(raw).hexdigest()


def _validate_rollback(path: Path, expected_sha256: str) -> None:
    if (_sha(expected_sha256, "r3 rollback receipt")
            != EXPECTED_ROLLBACK_RECEIPT_SHA256):
        raise ValueError("Exact saved r3 rollback receipt SHA-256 required")
    receipt = _exact_json(path, "r3 rollback receipt", expected_sha256)
    if (receipt.get("version") != EXPECTED_ROLLBACK_VERSION
            or receipt.get("status") != EXPECTED_ROLLBACK_STATUS
            or receipt.get("public_url")
                != "https://policylens-warehouse-study.onrender.com"):
        raise ValueError("Exact pre-diagnostic r3 rollback receipt required")
    parent = path.parent
    files = receipt.get("files")
    if not isinstance(files, Mapping) or set(files) != {
            "render.r3.yaml", "warehouse_alignment_online.zip",
            "warehouse_alignment_release.b64"}:
        raise ValueError("R3 rollback artifact registry differs")
    for name, record in files.items():
        target = parent / name
        if (target.is_symlink() or not target.is_file() or target.parent != parent
                or file_hash(target) != record.get("sha256")
                or target.stat().st_size != record.get("size")
                or stat.S_IMODE(target.stat().st_mode) & 0o077):
            raise ValueError("R3 rollback artifact changed: " + name)


def _validate_fixed_final_once_ledger(admission: Mapping[str, Any]) -> None:
    """Re-authenticate the fixed account ledger referenced by admission.

    The external ledger path is deliberately derived from final-once itself;
    neither ``HOME`` nor a receipt argument may redirect this check.
    """
    bindings = admission.get("bindings")
    artifacts = admission.get("artifacts")
    if not isinstance(bindings, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("Diagnostic admission final-once registry is missing")
    final_api = admission_api.final_once_api
    identity = final_api._campaign_identity()
    key = final_api.campaign_key()
    if (bindings.get("final_once_campaign_key") != key
            or bindings.get("final_once_identity_sha256") != digest(identity)):
        raise ValueError("Diagnostic admission final-once campaign differs")
    registry = final_api._ledger_root() / key
    for artifact_name, filename in admission_api.FINAL_ONCE_LEDGER_ARTIFACTS.items():
        record = artifacts.get(artifact_name)
        binding_name = artifact_name + "_sha256"
        expected_path = "external/final_once_ledger/" + filename
        target = registry / filename
        if (not isinstance(record, Mapping)
                or record.get("path") != expected_path
                or record.get("sha256") != bindings.get(binding_name)
                or target.is_symlink() or not target.is_file()
                or target.resolve() != target
                or file_hash(target) != bindings.get(binding_name)):
            raise ValueError("Diagnostic admission external ledger differs: " + filename)
    completion = final_api.read_completion(
        registry,
        expected_completion_sha256=bindings[
            "final_once_attempt_completed_sha256"],
        expected_identity=identity,
    )
    if (completion.get("permanent_anchor_sha256")
            != bindings.get("final_once_permanent_anchor_sha256")):
        raise ValueError("Diagnostic admission permanent final anchor differs")


def _component_paths_from_admission(
        admission: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every admission artifact from its authenticated registry.

    Repository artifacts are resolved below the fixed repository root.  The
    final-once phase files are the sole exception and are resolved from the
    non-caller-selectable account ledger used by final-once itself.
    """
    artifacts = admission.get("artifacts")
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != set(admission_api.ARTIFACT_NAMES)):
        raise ValueError("Diagnostic admission artifact registry differs")
    final_api = admission_api.final_once_api
    registry = final_api._ledger_root() / final_api.campaign_key()
    components: dict[str, Path] = {}
    for name in admission_api.ARTIFACT_NAMES:
        record = artifacts.get(name)
        if (not isinstance(record, Mapping)
                or set(record) != {"path", "sha256"}):
            raise ValueError("Diagnostic admission artifact record differs: " + name)
        expected_sha256 = _sha(record.get("sha256"), "admission artifact " + name)
        ledger_name = admission_api.FINAL_ONCE_LEDGER_ARTIFACTS.get(name)
        if ledger_name is not None:
            expected_relative = "external/final_once_ledger/" + ledger_name
            if record.get("path") != expected_relative:
                raise ValueError(
                    "Diagnostic admission external ledger path differs: " + name)
            target = (registry / ledger_name).absolute()
        else:
            relative = record.get("path")
            candidate = PurePosixPath(relative) if isinstance(relative, str) else None
            if (candidate is None or candidate.is_absolute()
                    or candidate.as_posix() != relative
                    or any(part in ("", ".", "..") for part in candidate.parts)):
                raise ValueError("Diagnostic admission artifact path is unsafe: " + name)
            target = (ROOT / Path(*candidate.parts)).absolute()
            try:
                target.relative_to(ROOT.resolve())
            except ValueError:
                raise ValueError(
                    "Diagnostic admission artifact escaped repository: " + name
                ) from None
        if (target.is_symlink() or not target.is_file()
                or target.resolve() != target
                or file_hash(target) != expected_sha256):
            raise ValueError("Diagnostic admission artifact bytes differ: " + name)
        components[name] = target
    return components


def read_strict_admission_anchor(
        path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """Re-authenticate the complete admission and all live evidence files."""
    admission_path, _ = _file(path, "diagnostic admission")
    expected = _sha(expected_sha256, "diagnostic admission")
    raw = _exact_json(admission_path, "diagnostic admission", expected)
    components = _component_paths_from_admission(raw)
    authenticated = admission_api.read_saved_admission(
        admission_path, expected_sha256=expected, components=components)
    if canonical(authenticated) != canonical(raw):
        raise ValueError("Diagnostic admission differs from strict live validation")
    return authenticated


def _validate_inputs(*, admission_path: Path, designation_path: Path,
                     package_path: Path, base64_path: Path,
                     rollback_path: Path, expected_admission_sha256: str,
                     expected_designation_sha256: str,
                     expected_rollback_sha256: str) -> dict[str, Any]:
    expected_admission_sha256 = _sha(expected_admission_sha256, "admission")
    if (_sha(expected_designation_sha256, "designation")
            != designation_binding.EXPECTED_DESIGNATION_SHA256
            or file_hash(designation_path) != expected_designation_sha256):
        raise ValueError("Diagnostic designation bytes differ")
    admission = read_strict_admission_anchor(
        admission_path, expected_sha256=expected_admission_sha256)
    designation = _exact_json(
        designation_path, "diagnostic designation",
        expected_designation_sha256)
    if (admission.get("version") != admission_api.VERSION
            or admission.get("status") != admission_api.STATUS
            or admission.get("behavior_performance_gate_passed") is not False
            or admission.get("behavior_performance_gate_waived") is not True
            or admission.get("waiver_scope") != ["behavior_performance"]
            or admission.get("formal_sample_eligible") is not False
            or admission.get("data_persistent") is not False
            or admission.get("bindings", {}).get("diagnostic_designation_sha256")
                != expected_designation_sha256
            or admission.get("bindings", {}).get("actor_sha256")
                != admission_api.FIXED_ACTOR_SHA256
            or designation.get("version") != designation_api.VERSION
            or designation.get("status") != designation_api.STATUS
            or designation.get("waiver_scope") != ["behavior_performance"]):
        raise ValueError("Exact internal diagnostic admission/designation required")
    decoded = release._package_bytes(base64_path=base64_path)
    package_bytes = release._package_bytes(package_path=package_path)
    if decoded != package_bytes:
        raise ValueError("Diagnostic Base64 does not encode the package")
    package_sha = sha256(package_bytes).hexdigest()
    manifest_sha = _manifest_sha(package_path)
    manifest = release.inspect_online_release(
        package_path=package_path,
        expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha,
    )
    identities, parent = manifest["identities"], manifest["parent"]
    bindings = admission.get("bindings", {})
    _validate_fixed_final_once_ledger(admission)
    expected_parent = {
        "version": admission["version"], "status": admission["status"],
        "diagnostic_admission_sha256": expected_admission_sha256,
        **{name: bindings[name] for name in release._PARENT_FIELDS
           if name not in {"version", "status", "diagnostic_admission_sha256"}},
    }
    identity_bindings = {
        "actor_sha256": "actor_sha256",
        "actor_parameters_sha256": "actor_parameters_sha256",
        "protocol_file_sha256": "protocol_file_sha256",
        "protocol_content_sha256": "protocol_content_sha256",
        "program_sha256": "program_sha256",
        "program_content_sha256": "program_content_sha256",
        "program_identity_sha256": "program_identity_sha256",
        "public_feature_contract_sha256": "public_feature_contract_sha256",
        "public_feature_registry_sha256": "public_feature_registry_sha256",
        "program_complexity_sha256": "program_complexity_sha256",
        "parent_runtime_signature": "runtime_signature",
        "parent_explainer_signature": "explainer_signature",
        "parent_question_bank_signature": "question_bank_signature",
        "tutorial_signature": "tutorial_signature",
        "tutorial_scene_id": "tutorial_scene_id",
        "tutorial_scene_fingerprint": "tutorial_scene_fingerprint",
        "tutorial_successor_state_sha256": "tutorial_successor_state_sha256",
        "tutorial_snapshot_sha256": "tutorial_snapshot_sha256",
        "diagnostic_contract_sha256": "diagnostic_contract_sha256",
        "conflict_families_sha256": "conflict_families_sha256",
    }
    if (manifest.get("release") != release._release_projection()
            or parent != expected_parent
            or parent.get("diagnostic_designation_sha256")
                != expected_designation_sha256
            or any(identities.get(identity) != bindings.get(binding)
                   for identity, binding in identity_bindings.items())):
        raise ValueError("Diagnostic package differs from its admission")
    loaded = release.load_online_release(
        package_path=package_path,
        expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha,
    )
    try:
        play = loaded.scenarios.get("splits", {}).get("play", [])
        if (loaded.release != release._release_projection()
                or loaded.runtime.actor_sha256 != identities["actor_sha256"]
                or loaded.provenance.get("version") != release.VERSION
                or len(play) != 7):
            raise ValueError("Reloaded diagnostic package identity differs")
    finally:
        loaded.close()
    _validate_rollback(rollback_path, expected_rollback_sha256)
    return {
        "admission": admission, "designation": designation,
        "manifest": manifest, "package_sha256": package_sha,
        "manifest_sha256": manifest_sha,
        "base64_sha256": file_hash(base64_path),
        "scene_fingerprints": [
            row["fingerprint"]
            for row in _json_from_archive(package_path)["runtime_scenes"]
        ],
    }


def _json_from_archive(package: Path) -> dict[str, Any]:
    with zipfile.ZipFile(package, "r") as archive:
        runtime = json.loads(archive.read(release.ARTIFACT_PATHS["runtime_manifest"]))
    return {"runtime_scenes": runtime["splits"]["play"]}


def build_receipt(*, admission_path: str | Path,
                  expected_admission_sha256: str,
                  designation_path: str | Path,
                  expected_designation_sha256: str,
                  package_path: str | Path, base64_path: str | Path,
                  rollback_receipt_path: str | Path,
                  expected_rollback_receipt_sha256: str,
                  output: str | Path) -> dict[str, Any]:
    admission_file, _ = _file(admission_path, "diagnostic admission")
    designation_file, _ = _file(designation_path, "diagnostic designation")
    package_file, _ = _file(package_path, "diagnostic package")
    base64_file, _ = _file(base64_path, "diagnostic Base64")
    rollback_file, rollback_relative = _file(
        rollback_receipt_path, "r3 rollback receipt")
    frozen_sources = dict(source_closure())

    def resolve_inputs() -> dict[str, Path]:
        return _release_input_paths(
            admission_path=admission_file, designation_path=designation_file,
            package_path=package_file, base64_path=base64_file,
            rollback_path=rollback_file,
            expected_admission_sha256=expected_admission_sha256,
            expected_rollback_sha256=expected_rollback_receipt_sha256,
        )

    frozen_paths = resolve_inputs()
    frozen_hashes = _release_input_hashes(frozen_paths)
    _guard_release_inputs(
        frozen_paths, expected_hashes=frozen_hashes,
        expected_sources=frozen_sources, phase="entry snapshot",
        resolve_paths=resolve_inputs)
    # Package inspection and rollback validation reopen their inputs through
    # several library layers (Base64, zipfile, runtime loading).  Freeze those
    # bytes once and make every semantic consumer use only the private copies;
    # the live admission is separately authenticated by its strict reader.
    snapshot_originals = {
        "designation": designation_file,
        "package": package_file,
        "base64": base64_file,
        "rollback_receipt": rollback_file,
    }
    rollback_snapshot_keys: dict[str, str] = {}
    for index, name in enumerate(sorted(
            key for key in frozen_paths
            if key.startswith("rollback_artifact:"))):
        snapshot_name = f"rollback_artifact_{index:03d}"
        snapshot_originals[snapshot_name] = frozen_paths[name]
        rollback_snapshot_keys[name] = snapshot_name
    expected_by_original = {
        path: frozen_hashes[name] for name, path in frozen_paths.items()
    }
    snapshot_expected = {
        name: expected_by_original[path]
        for name, path in snapshot_originals.items()
    }
    snapshot_relative = {
        "designation": "release/designation.json",
        "package": "release/package.zip",
        "base64": "release/package.b64",
        "rollback_receipt": "rollback/receipt.json",
        **{
            snapshot_name: "rollback/" + frozen_paths[registry_name].name
            for registry_name, snapshot_name in rollback_snapshot_keys.items()
        },
    }
    with input_snapshot_api.ImmutableInputSnapshot(
            snapshot_originals, expected_sha256=snapshot_expected,
            relative_names=snapshot_relative,
            prefix="warehouse-r41-release-receipt-") as immutable:
        checked = _validate_inputs(
            admission_path=admission_file,
            designation_path=immutable.paths["designation"],
            package_path=immutable.paths["package"],
            base64_path=immutable.paths["base64"],
            rollback_path=immutable.paths["rollback_receipt"],
            expected_admission_sha256=expected_admission_sha256,
            expected_designation_sha256=expected_designation_sha256,
            expected_rollback_sha256=expected_rollback_receipt_sha256,
        )
        try:
            immutable.verify()
        except RuntimeError as error:
            raise RuntimeError(
                "Diagnostic receipt input or source changed during "
                "semantic validation") from error
    _guard_release_inputs(
        frozen_paths, expected_hashes=frozen_hashes,
        expected_sources=frozen_sources, phase="semantic validation",
        resolve_paths=resolve_inputs)
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.absolute()
    try:
        self_path = output_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError("Diagnostic release receipt must stay inside repository") from None
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    manifest = checked["manifest"]
    value = {
        "version": VERSION, "status": STATUS,
        "release_version": release.PUBLIC_RELEASE_VERSION,
        "pilot_class": release.PILOT_CLASS,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False, "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "data_persistent": False, "runtime_action_override": False,
        "actor_sha256": manifest["identities"]["actor_sha256"],
        "designation_sha256": expected_designation_sha256,
        "admission_sha256": expected_admission_sha256,
        "package_sha256": checked["package_sha256"],
        "manifest_sha256": checked["manifest_sha256"],
        "base64_sha256": checked["base64_sha256"],
        "runtime_manifest_signature": manifest["identities"]
            ["runtime_manifest_signature"],
        "portable_runtime_manifest_sha256": manifest["parent"]
            ["portable_runtime_manifest_sha256"],
        "question_bank_sha256": manifest["parent"]["question_bank_sha256"],
        "tutorial_sha256": manifest["parent"]["tutorial_sha256"],
        "release_sources_sha256": manifest["parent"]["release_sources_sha256"],
        "package_contract_sha256": manifest["parent"]["package_contract_sha256"],
        "dynamic_selection_protocol_sha256": manifest["parent"]
            ["dynamic_selection_protocol_sha256"],
        "dynamic_selection_prefilter_sha256": manifest["parent"]
            ["dynamic_selection_prefilter_sha256"],
        "dynamic_selection_episodes_sha256": manifest["parent"]
            ["dynamic_selection_episodes_sha256"],
        "diagnostic_publication_authentication_sha256": manifest["parent"]
            ["diagnostic_publication_authentication_sha256"],
        **{
            name: manifest["parent"][name]
            for name in CANDIDATE_LINEAGE_FIELDS
        },
        "program_sha256": manifest["identities"]["program_sha256"],
        "program_content_sha256": manifest["identities"]["program_content_sha256"],
        "program_identity_sha256": manifest["identities"]["program_identity_sha256"],
        "public_feature_contract_sha256": manifest["identities"]
            ["public_feature_contract_sha256"],
        "public_feature_registry_sha256": manifest["identities"]
            ["public_feature_registry_sha256"],
        "program_complexity_sha256": manifest["identities"]
            ["program_complexity_sha256"],
        "final_once_identity_sha256": manifest["parent"]
            ["final_once_identity_sha256"],
        "final_once_campaign_key": manifest["parent"]
            ["final_once_campaign_key"],
        "final_once_permanent_anchor_sha256": manifest["parent"]
            ["final_once_permanent_anchor_sha256"],
        "final_once_attempt_started_sha256": manifest["parent"]
            ["final_once_attempt_started_sha256"],
        "final_once_candidate_authenticated_sha256": manifest["parent"]
            ["final_once_candidate_authenticated_sha256"],
        "final_once_holdout_started_sha256": manifest["parent"]
            ["final_once_holdout_started_sha256"],
        "final_once_historical_exclusion_started_sha256": manifest["parent"]
            ["final_once_historical_exclusion_started_sha256"],
        "final_once_historical_exclusion_completed_sha256": manifest["parent"]
            ["final_once_historical_exclusion_completed_sha256"],
        "final_once_holdout_completed_sha256": manifest["parent"]
            ["final_once_holdout_completed_sha256"],
        "final_once_audit_started_sha256": manifest["parent"]
            ["final_once_audit_started_sha256"],
        "final_once_audit_completed_sha256": manifest["parent"]
            ["final_once_audit_completed_sha256"],
        "final_once_attempt_completed_sha256": manifest["parent"]
            ["final_once_attempt_completed_sha256"],
        "explanation_audit_inputs_sha256": manifest["parent"]
            ["explanation_audit_inputs_sha256"],
        "explanation_audit_evidence_sha256": manifest["parent"]
            ["explanation_audit_evidence_sha256"],
        "explanation_audit_sha256": manifest["parent"]["explanation_audit_sha256"],
        "physical_replay_sha256": manifest["parent"]["physical_replay_sha256"],
        "fresh_final_v3_exclusion_sha256": manifest["parent"]
            ["fresh_final_v3_exclusion_sha256"],
        "fresh_final_v3_exclusion_content_sha256": manifest["parent"]
            ["fresh_final_v3_exclusion_content_sha256"],
        "selected_scene_fingerprints": checked["scene_fingerprints"],
        "rollback_receipt_path": rollback_relative,
        "rollback_receipt_sha256": expected_rollback_receipt_sha256,
        "self_path": self_path,
    }
    _guard_release_inputs(
        frozen_paths, expected_hashes=frozen_hashes,
        expected_sources=frozen_sources, phase="receipt assembly",
        resolve_paths=resolve_inputs)
    published = False
    try:
        _publish_new_receipt(
            output_path, value,
            before_publish=lambda: _guard_release_inputs(
                frozen_paths, expected_hashes=frozen_hashes,
                expected_sources=frozen_sources, phase="receipt publication",
                resolve_paths=resolve_inputs),
        )
        published = True
        _guard_release_inputs(
            frozen_paths, expected_hashes=frozen_hashes,
            expected_sources=frozen_sources,
            phase="post-publication validation", resolve_paths=resolve_inputs)
        read_saved_receipt(output_path, expected_sha256=file_hash(output_path))
    except BaseException:
        if published:
            output_path.unlink(missing_ok=True)
        raise
    return value


def read_saved_receipt(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    frozen_sources = dict(source_closure())
    saved_path, relative = _file(path, "diagnostic release receipt")
    expected_receipt = _sha(expected_sha256, "release receipt")
    value = _exact_json(
        saved_path, "diagnostic release receipt", expected_receipt)
    if (set(value) != FIELDS or value.get("version") != VERSION
            or value.get("status") != STATUS
            or value.get("release_version") != release.PUBLIC_RELEASE_VERSION
            or value.get("pilot_class") != release.PILOT_CLASS
            or value.get("behavior_performance_gate_passed") is not False
            or value.get("behavior_performance_gate_waived") is not True
            or value.get("waiver_scope") != ["behavior_performance"]
            or value.get("formal_ready") is not False
            or value.get("formal_sample_eligible") is not False
            or value.get("human_explanation_effect_validated") is not False
            or value.get("data_persistent") is not False
            or value.get("runtime_action_override") is not False
            or value.get("actor_sha256") != admission_api.FIXED_ACTOR_SHA256
            or value.get("designation_sha256")
                != designation_binding.EXPECTED_DESIGNATION_SHA256
            or value.get("self_path") != relative
            or not isinstance(value.get("selected_scene_fingerprints"), list)
            or len(value["selected_scene_fingerprints"]) != 7
            or len(set(value["selected_scene_fingerprints"])) != 7):
        raise ValueError("Exact non-formal diagnostic release receipt required")
    for name in (
        "actor_sha256", "designation_sha256", "admission_sha256",
        "package_sha256", "manifest_sha256", "base64_sha256",
        "runtime_manifest_signature", "rollback_receipt_sha256",
        "portable_runtime_manifest_sha256", "question_bank_sha256",
        "tutorial_sha256",
        "release_sources_sha256", "package_contract_sha256",
        "dynamic_selection_protocol_sha256", "dynamic_selection_prefilter_sha256",
        "dynamic_selection_episodes_sha256",
        "diagnostic_publication_authentication_sha256",
        *CANDIDATE_LINEAGE_FIELDS,
        "program_sha256", "program_content_sha256", "program_identity_sha256",
        "public_feature_contract_sha256", "public_feature_registry_sha256",
        "program_complexity_sha256", "final_once_identity_sha256",
        "final_once_campaign_key", "final_once_permanent_anchor_sha256",
        "final_once_attempt_started_sha256", "final_once_attempt_completed_sha256",
        "final_once_candidate_authenticated_sha256",
        "final_once_holdout_started_sha256",
        "final_once_historical_exclusion_started_sha256",
        "final_once_historical_exclusion_completed_sha256",
        "final_once_holdout_completed_sha256",
        "final_once_audit_started_sha256", "final_once_audit_completed_sha256",
        "fresh_final_v3_exclusion_sha256",
        "fresh_final_v3_exclusion_content_sha256",
        "explanation_audit_inputs_sha256", "explanation_audit_evidence_sha256",
        "explanation_audit_sha256", "physical_replay_sha256",
    ):
        _sha(value[name], "diagnostic release receipt " + name)
    for fingerprint in value["selected_scene_fingerprints"]:
        _sha(fingerprint, "diagnostic release scene fingerprint")
    rollback, rollback_relative = _file(
        value["rollback_receipt_path"], "r3 rollback receipt")
    if rollback_relative != value["rollback_receipt_path"]:
        raise ValueError("R3 rollback receipt path differs")
    rollback_value = _exact_json(
        rollback, "r3 rollback receipt", value["rollback_receipt_sha256"])
    rollback_files = rollback_value.get("files")
    if not isinstance(rollback_files, Mapping):
        raise ValueError("R3 rollback artifact registry differs")
    snapshot_paths: dict[str, Path] = {"receipt": rollback}
    snapshot_hashes = {"receipt": value["rollback_receipt_sha256"]}
    snapshot_relatives = {"receipt": "rollback/receipt.json"}
    for index, name in enumerate(sorted(rollback_files)):
        record = rollback_files[name]
        target = _canonical_regular(
            rollback.parent / name, "r3 rollback artifact " + name)
        if (not isinstance(record, Mapping)
                or target.stat().st_size != record.get("size")
                or stat.S_IMODE(target.stat().st_mode) & 0o077):
            raise ValueError("R3 rollback artifact changed: " + name)
        key = f"artifact_{index:03d}"
        snapshot_paths[key] = target
        snapshot_hashes[key] = _sha(
            record.get("sha256"), "r3 rollback artifact " + name)
        snapshot_relatives[key] = "rollback/" + name
    with input_snapshot_api.ImmutableInputSnapshot(
            snapshot_paths, expected_sha256=snapshot_hashes,
            relative_names=snapshot_relatives,
            prefix="warehouse-r41-saved-release-receipt-") as immutable:
        _validate_rollback(
            immutable.paths["receipt"], value["rollback_receipt_sha256"])
        immutable.verify()
    if source_closure() != frozen_sources:
        raise RuntimeError("Diagnostic release receipt source changed during read")
    _exact_json(saved_path, "diagnostic release receipt", expected_receipt)
    return deepcopy(value)


__all__ = [
    "VERSION", "STATUS", "EXPECTED_ROLLBACK_RECEIPT_SHA256",
    "CANDIDATE_LINEAGE_FIELDS", "FIELDS",
    "source_closure", "read_strict_admission_anchor", "build_receipt",
    "read_saved_receipt",
]
