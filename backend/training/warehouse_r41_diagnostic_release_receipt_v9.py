"""Append-only receipt for one admitted r4.1 diagnostic v9 package."""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import zipfile

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from ui import warehouse_alignment_r41_diagnostic_release_v9 as release

VERSION = "warehouse-r41-diagnostic-release-receipt.v9"
STATUS = "authenticated_internal_diagnostic_release_v9"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset((
    "version", "status", "release_version", "pilot_class",
    "admission_sha256", "admission_content_sha256", "package_sha256",
    "package_size", "base64_sha256", "base64_size", "manifest_sha256",
    "actor_sha256", "program_sha256", "compact_program_sha256",
    "designation_sha256", "candidate_lock_sha256", "outer_result_sha256",
    "final_audit_sha256", "play_scene_fingerprints_sha256",
    "release_sources_sha256", "behavior_performance_gate_passed",
    "behavior_performance_gate_waived", "formal_ready",
    "formal_sample_eligible", "data_persistent", "runtime_action_override",
    "producer_sources", "producer_sources_sha256", "content_sha256",
))


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat(follow_symlinks=False).st_size <= maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, child in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               ValueError("Non-finite JSON in " + label)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must contain one object")
    return value


def _manifest_sha(package: Path) -> str:
    try:
        with zipfile.ZipFile(package, "r") as archive:
            matches = [row for row in archive.infolist()
                       if row.filename == release.MANIFEST_NAME]
            if len(matches) != 1:
                raise ValueError("V9 package must contain one manifest")
            raw = archive.read(matches[0])
    except zipfile.BadZipFile as error:
        raise ValueError("V9 package is not a ZIP") from error
    return sha256(raw).hexdigest()


def validate_inputs(*, admission_path: str | Path,
                    expected_admission_sha256: str,
                    components: Mapping[str, str | Path],
                    outer_permanent_registry: str | Path,
                    final_permanent_registry: str | Path,
                    package_path: str | Path,
                    base64_path: str | Path) -> dict[str, Any]:
    from backend.training import warehouse_r41_diagnostic_admission_v9 as admission_api
    admission_file = _regular(admission_path, "v9 admission", 128 * 1024 * 1024)
    package = _regular(package_path, "v9 package", release.MAX_PACKAGE_BYTES)
    encoded_file = _regular(base64_path, "v9 Base64 secret", release.MAX_BASE64_BYTES)
    admission_sha = _sha(expected_admission_sha256, "v9 admission")
    if file_hash(admission_file) != admission_sha:
        raise ValueError("V9 admission bytes differ")
    admission = admission_api.read_saved_admission(
        admission_file, expected_sha256=admission_sha, components=components,
        outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry)
    encoded_raw = encoded_file.read_bytes()
    if len(encoded_raw) > release.MAX_BASE64_BYTES:
        raise ValueError("V9 Base64 Secret File exceeds 960,000 bytes")
    try:
        decoded = base64.b64decode(b"".join(encoded_raw.split()), validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("V9 Base64 Secret File is invalid") from error
    package_raw = package.read_bytes()
    if decoded != package_raw:
        raise ValueError("V9 Base64 Secret File does not encode the package")
    package_sha = sha256(package_raw).hexdigest()
    manifest_sha = _manifest_sha(package)
    manifest = release.inspect_online_release(
        package_path=package, expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha)
    if (manifest.get("parent", {}).get("admission_sha256") != admission_sha
            or manifest.get("parent", {}).get("admission_content_sha256")
                != admission["content_sha256"]
            or manifest.get("parent", {}).get("bindings_sha256")
                != digest(admission["bindings"])
            or manifest.get("play_scenes") != admission["play_scenes"]
            or manifest.get("sources", {}).get("release")
                != release.release_sources()):
        raise ValueError("V9 package does not derive from the admission")
    identities = manifest["identities"]
    bindings = admission["bindings"]
    for name in identities:
        if name in bindings and identities[name] != bindings[name]:
            raise ValueError("V9 package identity differs from admission: " + name)
    return {
        "version": VERSION, "status": STATUS,
        "release_version": release.PUBLIC_RELEASE_VERSION,
        "pilot_class": release.PILOT_CLASS,
        "admission_sha256": admission_sha,
        "admission_content_sha256": admission["content_sha256"],
        "package_sha256": package_sha, "package_size": len(package_raw),
        "base64_sha256": file_hash(encoded_file),
        "base64_size": len(encoded_raw), "manifest_sha256": manifest_sha,
        "actor_sha256": bindings["actor_sha256"],
        "program_sha256": bindings["program_sha256"],
        "compact_program_sha256": bindings["compact_program_sha256"],
        "designation_sha256": bindings["designation_sha256"],
        "candidate_lock_sha256": bindings["candidate_lock_sha256"],
        "outer_result_sha256": bindings["outer_result_sha256"],
        "final_audit_sha256": bindings["final_audit_sha256"],
        "play_scene_fingerprints_sha256": bindings[
            "play_scene_fingerprints_sha256"],
        "release_sources_sha256": bindings["release_sources_sha256"],
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False, "runtime_action_override": False,
        "producer_sources": producer_sources(),
        "producer_sources_sha256": digest(producer_sources()),
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush(); os.fsync(stream.fileno())


def build_receipt(*, output: str | Path, **inputs: Any) -> dict[str, Any]:
    value = validate_inputs(**inputs)
    value["content_sha256"] = digest(value)
    target = Path(output).expanduser().absolute()
    _write_new(target, value)
    expected = file_hash(target)
    try:
        read_saved_receipt(target, expected_sha256=expected, **inputs)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return deepcopy(value)


def read_saved_receipt(path: str | Path, *, expected_sha256: str,
                       **inputs: Any) -> dict[str, Any]:
    receipt = _regular(path, "v9 release receipt", 8 * 1024 * 1024)
    if file_hash(receipt) != _sha(expected_sha256, "v9 release receipt"):
        raise ValueError("V9 release receipt bytes differ")
    value = _strict_json(receipt, "v9 release receipt")
    if (set(value) != _FIELDS or value.get("version") != VERSION
            or value.get("status") != STATUS
            or value.get("content_sha256") != digest({
                key: child for key, child in value.items()
                if key != "content_sha256"})):
        raise ValueError("Exact v9 release receipt required")
    current = validate_inputs(**inputs)
    for name, child in current.items():
        if value.get(name) != child:
            raise ValueError("V9 receipt differs from live input: " + name)
    return deepcopy(value)


__all__ = ["VERSION", "STATUS", "producer_sources", "validate_inputs", "build_receipt",
           "read_saved_receipt"]
