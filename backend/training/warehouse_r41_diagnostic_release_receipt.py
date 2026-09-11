"""Immutable receipt for one built r4.1 diagnostic Secret File."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import zipfile

from backend.training import warehouse_r41_diagnostic_admission as admission_api
from backend.training import warehouse_r41_diagnostic_designation as designation_api
from backend.training.warehouse_native_common import canonical, file_hash
from ui import warehouse_alignment_r41_diagnostic_release as release


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-release-receipt.v1"
STATUS = "completed_internal_diagnostic_release"
EXPECTED_ROLLBACK_VERSION = "warehouse-r41-diagnostic-r3-rollback.v1"
EXPECTED_ROLLBACK_STATUS = "saved_before_diagnostic_deployment"
EXPECTED_ROLLBACK_RECEIPT_SHA256 = (
    "491ccd2eb13f7e50bde492195425272c5128a0b0076f4bdf87ba28021fcfd126"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
FIELDS = frozenset((
    "version", "status", "release_version", "pilot_class",
    "behavior_performance_gate_passed", "behavior_performance_gate_waived",
    "waiver_scope", "formal_ready", "formal_sample_eligible",
    "human_explanation_effect_validated", "data_persistent",
    "runtime_action_override", "actor_sha256", "designation_sha256",
    "admission_sha256", "package_sha256", "manifest_sha256",
    "base64_sha256", "runtime_manifest_signature",
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _manifest_sha(package: Path) -> str:
    try:
        with zipfile.ZipFile(package, "r") as archive:
            raw = archive.read(release.MANIFEST_NAME)
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError("Diagnostic package has no valid manifest") from error
    return sha256(raw).hexdigest()


def _validate_rollback(path: Path, expected_sha256: str) -> None:
    if (_sha(expected_sha256, "r3 rollback receipt")
            != EXPECTED_ROLLBACK_RECEIPT_SHA256):
        raise ValueError("Exact saved r3 rollback receipt SHA-256 required")
    if file_hash(path) != expected_sha256:
        raise ValueError("R3 rollback receipt bytes differ")
    receipt = _json(path, "r3 rollback receipt")
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


def _validate_inputs(*, admission_path: Path, designation_path: Path,
                     package_path: Path, base64_path: Path,
                     rollback_path: Path, expected_admission_sha256: str,
                     expected_designation_sha256: str,
                     expected_rollback_sha256: str) -> dict[str, Any]:
    if file_hash(admission_path) != _sha(expected_admission_sha256, "admission"):
        raise ValueError("Diagnostic admission bytes differ")
    if file_hash(designation_path) != _sha(
            expected_designation_sha256, "designation"):
        raise ValueError("Diagnostic designation bytes differ")
    admission = _json(admission_path, "diagnostic admission")
    designation = _json(designation_path, "diagnostic designation")
    if (admission.get("version") != admission_api.VERSION
            or admission.get("status") != admission_api.STATUS
            or admission.get("behavior_performance_gate_passed") is not False
            or admission.get("behavior_performance_gate_waived") is not True
            or admission.get("waiver_scope") != ["behavior_performance"]
            or admission.get("formal_sample_eligible") is not False
            or admission.get("data_persistent") is not False
            or admission.get("bindings", {}).get("diagnostic_designation_sha256")
                != expected_designation_sha256
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
    if (manifest.get("release") != release._release_projection()
            or parent.get("diagnostic_admission_sha256")
                != expected_admission_sha256
            or parent.get("diagnostic_designation_sha256")
                != expected_designation_sha256
            or identities.get("actor_sha256")
                != admission.get("bindings", {}).get("actor_sha256")):
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
    checked = _validate_inputs(
        admission_path=admission_file, designation_path=designation_file,
        package_path=package_file, base64_path=base64_file,
        rollback_path=rollback_file,
        expected_admission_sha256=expected_admission_sha256,
        expected_designation_sha256=expected_designation_sha256,
        expected_rollback_sha256=expected_rollback_receipt_sha256,
    )
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
        "selected_scene_fingerprints": checked["scene_fingerprints"],
        "rollback_receipt_path": rollback_relative,
        "rollback_receipt_sha256": expected_rollback_receipt_sha256,
        "self_path": self_path,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink() or output_path.parent.resolve() != output_path.parent:
        raise ValueError("Diagnostic release receipt parent is unsafe")
    descriptor = os.open(
        output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush(); os.fsync(stream.fileno())
    try:
        read_saved_receipt(output_path, expected_sha256=file_hash(output_path))
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return value


def read_saved_receipt(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    saved_path, relative = _file(path, "diagnostic release receipt")
    if file_hash(saved_path) != _sha(expected_sha256, "release receipt"):
        raise ValueError("Diagnostic release receipt bytes differ")
    value = _json(saved_path, "diagnostic release receipt")
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
            or value.get("self_path") != relative
            or not isinstance(value.get("selected_scene_fingerprints"), list)
            or len(value["selected_scene_fingerprints"]) != 7
            or len(set(value["selected_scene_fingerprints"])) != 7):
        raise ValueError("Exact non-formal diagnostic release receipt required")
    for name in (
        "actor_sha256", "designation_sha256", "admission_sha256",
        "package_sha256", "manifest_sha256", "base64_sha256",
        "runtime_manifest_signature", "rollback_receipt_sha256",
    ):
        _sha(value[name], "diagnostic release receipt " + name)
    for fingerprint in value["selected_scene_fingerprints"]:
        _sha(fingerprint, "diagnostic release scene fingerprint")
    rollback, rollback_relative = _file(
        value["rollback_receipt_path"], "r3 rollback receipt")
    if rollback_relative != value["rollback_receipt_path"]:
        raise ValueError("R3 rollback receipt path differs")
    _validate_rollback(rollback, value["rollback_receipt_sha256"])
    return deepcopy(value)


__all__ = [
    "VERSION", "STATUS", "EXPECTED_ROLLBACK_RECEIPT_SHA256", "FIELDS",
    "build_receipt", "read_saved_receipt",
]
