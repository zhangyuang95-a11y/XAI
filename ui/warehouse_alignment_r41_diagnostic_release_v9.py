"""Strict portable release for the r4.1 diagnostic v9 study.

The archive is a deployment derivative of a passing v9 admission.  It contains
only the six files needed by the participant service.  Protected outer/final
evidence stays outside the archive and is represented only by hashes in the
admission-derived manifest.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tempfile
from typing import Any, Mapping
import zipfile

from backend import warehouse_r41_diagnostic_compact_public_tree_v9 as compact_api
from backend import warehouse_r41_diagnostic_online_explanation_v9 as explanation_api
from backend.warehouse_r41_diagnostic_online_runtime_portable_v1 import (
    PORTABLE_RUNTIME_MANIFEST_VERSION,
    R41DiagnosticOnlineAlignmentRuntime,
    diagnostic_runtime_sources,
)
from backend.training.warehouse_native_common import canonical, digest, file_hash
from ui import warehouse_alignment_online_release as portable
from ui import warehouse_alignment_r41_diagnostic_release_v8 as v8_release
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r41-diagnostic-online-release.v9"
STATUS = "r41_diagnostic_online_portable_internal_experiment_v9"
PUBLIC_RELEASE_VERSION = "r4.1-diagnostic"
PILOT_CLASS = "internal_diagnostic"
ANIMATION_DURATION_MS = 380
MANIFEST_NAME = "manifest.json"
ARTIFACT_PATHS = {
    "actor": "artifacts/actor.npz",
    "protocol": "artifacts/training_protocol.json",
    "runtime_manifest": "artifacts/runtime_manifest.json",
    "program": "artifacts/program.ctree.xz",
    "question_bank": "artifacts/question_bank.json",
    "tutorial": "artifacts/tutorial.json",
}
ARCHIVE_WHITELIST = frozenset((MANIFEST_NAME, *ARTIFACT_PATHS.values()))
MAX_BASE64_BYTES = 1_000_000
MAX_PACKAGE_BYTES = 749_997
ARCHIVE_COMPRESSION = zipfile.ZIP_BZIP2
ARCHIVE_COMPRESSLEVEL = 9
MAX_MANIFEST_BYTES = 512_000
MAX_ARTIFACT_BYTES = {
    "actor": 2_000_000,
    "protocol": 2_000_000,
    "runtime_manifest": 8_000_000,
    "program": compact_api.MAX_COMPRESSED_BYTES,
    "question_bank": 16_000_000,
    "tutorial": 2_000_000,
}
MAX_UNCOMPRESSED_BYTES = MAX_MANIFEST_BYTES + sum(MAX_ARTIFACT_BYTES.values())
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|"
    r"database[_-]?url|authorization|bearer[_-]?token|"
    r"participant[_-]?(?:id|key|record|data))"
    r"(?:$|[_-])", re.I)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|"
    r"\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://|"
    r"(?:^|[/\\])Users[/\\])", re.I)
_MANIFEST_FIELDS = frozenset((
    "version", "status", "test_fixture", "pilot_class", "formal_ready",
    "formal_sample_eligible", "data_persistent", "parent", "artifacts",
    "identities", "play_scenes", "sources", "analysis", "release",
))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    return portable._parse_json(raw, label)


def _reject_sensitive(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                raise ValueError("Sensitive field in " + label)
            _reject_sensitive(child, label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_sensitive(child, label)
    elif isinstance(value, str):
        stripped = value.strip()
        if (_SENSITIVE_VALUE.search(value) or Path(stripped).is_absolute()
                or PureWindowsPath(stripped).is_absolute()):
            raise ValueError("Sensitive or absolute-local value in " + label)


def release_sources() -> dict[str, str]:
    """Hash every source file used by the deployed v9 participant service."""

    paths = (
        Path(__file__),
        ROOT / "ui/warehouse_alignment_online_release.py",
        ROOT / "ui/warehouse_alignment_r41_diagnostic_release_v8.py",
        ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "backend/warehouse_r41_diagnostic_compact_public_tree_v9.py",
        ROOT / "ui/warehouse_alignment_r41_diagnostic_tutorial.py",
        ROOT / "ui/warehouse_alignment_r41_tutorial.py",
        ROOT / "ui/warehouse_family_feedback_research/index.html",
        ROOT / "ui/warehouse_family_feedback_research/app.js",
        ROOT / "ui/warehouse_family_feedback_research/styles.css",
        ROOT / "ui/warehouse_family_feedback_research/favicon.svg",
        ROOT / "requirements-render.txt",
    )
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("v9 release source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = file_hash(path)
    for values in (v8_release.release_sources(), diagnostic_runtime_sources(),
                   explanation_api.explanation_sources(),
                   tutorial_api.producer_sources()):
        for name, value in values.items():
            if name in result and result[name] != value:
                raise ValueError("v9 release source hash conflict: " + name)
            result[name] = value
    return dict(sorted(result.items()))


def release_projection(*, ready: bool = False) -> dict[str, Any]:
    return {
        "release_version": PUBLIC_RELEASE_VERSION,
        "status": ("internal_diagnostic_technically_verified_v9" if ready
                   else "r41_diagnostic_v9_wiring_ready_artifacts_unbound"),
        "namespace": "internal_diagnostic",
        "pilot_class": PILOT_CLASS,
        "formal_ready": False,
        "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "data_persistent": False,
        "test_fixture": False,
        "online_portable": True,
        "participant_enabled": ready,
        "model_ready": ready,
        "study_ready": ready,
        "explanation_ready": ready,
        "runtime_action_override": False,
        "animation_duration_ms": ANIMATION_DURATION_MS,
        "message": {
            "zh": "内部诊断实验：数据仅用于探索，服务重启后可能丢失，不纳入正式研究样本。",
            "en": "Internal diagnostic study: data are exploratory, may be lost after a service restart, and are excluded from the formal sample.",
        },
    }


def _artifact_records(artifacts: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_PATHS):
        raise ValueError("Exact v9 portable artifact set required")
    result = {}
    for name, relative in ARTIFACT_PATHS.items():
        raw = artifacts[name]
        if type(raw) is not bytes or not raw or len(raw) > MAX_ARTIFACT_BYTES[name]:
            raise ValueError("V9 portable artifact size differs: " + name)
        result[name] = {"path": relative, "size": len(raw),
                        "sha256": sha256(raw).hexdigest()}
    return result


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS
            or value.get("version") != VERSION or value.get("status") != STATUS
            or value.get("test_fixture") is not False
            or value.get("pilot_class") != PILOT_CLASS
            or value.get("formal_ready") is not False
            or value.get("formal_sample_eligible") is not False
            or value.get("data_persistent") is not False
            or value.get("release") != release_projection(ready=True)
            or value.get("analysis") != portable._analysis_protocol()):
        raise ValueError("Exact v9 diagnostic release manifest required")
    parent = value.get("parent")
    if (not isinstance(parent, Mapping)
            or set(parent) != {"version", "status", "admission_sha256",
                              "admission_content_sha256", "bindings_sha256",
                              "gates_sha256"}
            or (parent.get("version"), parent.get("status")) not in {
                ("warehouse-r41-diagnostic-admission.v9",
                 "admitted_internal_diagnostic_v9"),
                ("warehouse-r41-diagnostic-admission.v10",
                 "admitted_internal_diagnostic_v10"),
                ("warehouse-r41-diagnostic-admission.v11",
                 "admitted_internal_diagnostic_v11"),
            }):
        raise ValueError("V9 diagnostic admission parent differs")
    for name in set(parent) - {"version", "status"}:
        _sha(parent[name], "v9 parent " + name)
    identities = value.get("identities")
    required_identities = {
        "actor_sha256", "protocol_file_sha256", "protocol_content_sha256",
        "runtime_manifest_file_sha256", "runtime_manifest_content_sha256",
        "runtime_manifest_semantic_sha256", "runtime_signature",
        "runtime_manifest_signature", "candidate_lock_sha256",
        "program_sha256", "program_content_sha256", "compact_program_sha256",
        "runtime_program_sha256", "runtime_program_content_sha256",
        "actor_feature_names_sha256", "public_feature_contract_sha256",
        "question_bank_sha256", "question_bank_private_items_sha256",
        "question_bank_public_items_sha256", "question_bank_signature",
        "tutorial_sha256", "tutorial_signature", "release_sources_sha256",
        "designation_sha256", "outer_result_sha256", "final_audit_sha256",
    }
    if (not isinstance(identities, Mapping)
            or set(identities) != required_identities):
        raise ValueError("Exact v9 portable identities required")
    for name, child in identities.items():
        _sha(child, "v9 identity " + name)
    records = value.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != set(ARTIFACT_PATHS):
        raise ValueError("Exact v9 archive record set required")
    expected_hashes = {
        "actor": identities["actor_sha256"],
        "protocol": identities["protocol_file_sha256"],
        "runtime_manifest": identities["runtime_manifest_file_sha256"],
        "program": identities["compact_program_sha256"],
        "question_bank": identities["question_bank_sha256"],
        "tutorial": identities["tutorial_sha256"],
    }
    for name, relative in ARTIFACT_PATHS.items():
        record = records.get(name)
        if (not isinstance(record, Mapping)
                or set(record) != {"path", "size", "sha256"}
                or record.get("path") != relative
                or type(record.get("size")) is not int
                or not 0 < record["size"] <= MAX_ARTIFACT_BYTES[name]
                or record.get("sha256") != expected_hashes[name]):
            raise ValueError("V9 archived artifact record differs: " + name)
    scenes = value.get("play_scenes")
    if (not isinstance(scenes, list) or len(scenes) != 7
            or len({row.get("id") for row in scenes
                    if isinstance(row, Mapping)}) != 7
            or len({row.get("fingerprint") for row in scenes
                    if isinstance(row, Mapping)}) != 7
            or any(not isinstance(row, Mapping)
                   or set(row) != {"id", "family_id", "fingerprint", "scene_sha256"}
                   or _HEX.fullmatch(str(row.get("fingerprint", ""))) is None
                   or _HEX.fullmatch(str(row.get("scene_sha256", ""))) is None
                   for row in scenes)):
        raise ValueError("Exact tutorial plus six play-scene identities required")
    sources = value.get("sources")
    if (not isinstance(sources, Mapping) or set(sources) != {"release"}
            or portable._validate_source_map(
                sources["release"], "v9 diagnostic release") != release_sources()
            or digest(sources["release"]) != identities["release_sources_sha256"]):
        raise ValueError("V9 release source closure differs")
    _reject_sensitive(value, "v9 release manifest")
    return deepcopy(dict(value))


def _archive_bytes(manifest: Mapping[str, Any], artifacts: Mapping[str, bytes]) -> bytes:
    _validate_manifest(manifest)
    if _artifact_records(artifacts) != manifest["artifacts"]:
        raise ValueError("V9 manifest and artifact bytes differ")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=ARCHIVE_COMPRESSION,
                         compresslevel=ARCHIVE_COMPRESSLEVEL) as archive:
        members = ((MANIFEST_NAME, (canonical(manifest) + "\n").encode("utf-8")),
                   *((ARTIFACT_PATHS[name], artifacts[name])
                     for name in ARTIFACT_PATHS))
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = ARCHIVE_COMPRESSION
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, raw, compress_type=ARCHIVE_COMPRESSION,
                             compresslevel=ARCHIVE_COMPRESSLEVEL)
    raw = stream.getvalue()
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("V9 ZIP exceeds the 1 MB Secret File boundary")
    return raw


def _read_archive(raw: bytes, *, expected_package_sha256: str,
                  expected_manifest_sha256: str) -> tuple[dict, dict[str, bytes]]:
    if (type(raw) is not bytes or not raw or len(raw) > MAX_PACKAGE_BYTES
            or sha256(raw).hexdigest() != _sha(
                expected_package_sha256, "v9 package")):
        raise ValueError("V9 package bytes differ")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (archive.comment or len(names) != len(set(names))
                    or set(names) != ARCHIVE_WHITELIST
                    or sum(item.file_size for item in infos)
                        > MAX_UNCOMPRESSED_BYTES):
                raise ValueError("V9 archive member set or size differs")
            by_path = {value: name for name, value in ARTIFACT_PATHS.items()}
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (pure.is_absolute() or ".." in pure.parts
                        or pure.as_posix() != info.filename or info.is_dir()
                        or info.create_system != 3
                        or stat.S_IFMT(mode) != stat.S_IFREG
                        or stat.S_IMODE(mode) != 0o600
                        or info.date_time != (1980, 1, 1, 0, 0, 0)
                        or info.extra or info.comment
                        or info.compress_type != ARCHIVE_COMPRESSION
                        or info.flag_bits & 0x1):
                    raise ValueError("V9 archive member is unsafe")
                limit = (MAX_MANIFEST_BYTES if info.filename == MANIFEST_NAME
                         else MAX_ARTIFACT_BYTES[by_path[info.filename]])
                if not 0 < info.file_size <= limit:
                    raise ValueError("V9 archive member exceeds its limit")
            manifest_raw = archive.read(MANIFEST_NAME)
            if (len(manifest_raw) > MAX_MANIFEST_BYTES
                    or sha256(manifest_raw).hexdigest()
                        != _sha(expected_manifest_sha256, "v9 manifest")):
                raise ValueError("V9 manifest bytes differ")
            manifest = _validate_manifest(_parse_json(
                manifest_raw, "v9 archive manifest"))
            artifacts: dict[str, bytes] = {}
            for name, relative in ARTIFACT_PATHS.items():
                record = manifest["artifacts"][name]
                item = archive.read(relative)
                if (len(item) != record["size"]
                        or sha256(item).hexdigest() != record["sha256"]):
                    raise ValueError("V9 archived artifact differs: " + name)
                artifacts[name] = item
    except zipfile.BadZipFile as error:
        raise ValueError("V9 package is not a valid ZIP") from error
    return manifest, artifacts


def _package_bytes(*, package_path=None, base64_path=None) -> bytes:
    if (package_path is None) == (base64_path is None):
        raise ValueError("Choose exactly one v9 ZIP or Base64 Secret File")
    if package_path is not None:
        return portable._read_bounded(package_path, MAX_PACKAGE_BYTES,
                                      "v9 package")
    encoded = b"".join(portable._read_bounded(
        base64_path, MAX_BASE64_BYTES, "v9 Base64 secret").split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Invalid v9 Base64 Secret File") from error
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Decoded v9 package exceeds its safe limit")
    return raw


def inspect_online_release(*, expected_package_sha256: str,
                           expected_manifest_sha256: str,
                           package_path=None, base64_path=None) -> dict[str, Any]:
    raw = _package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, _ = _read_archive(
        raw, expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256)
    return manifest


def _write_material(root: Path, artifacts: Mapping[str, bytes]) -> dict[str, Path]:
    paths = {}
    for name, relative in ARTIFACT_PATHS.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(artifacts[name]); stream.flush(); os.fsync(stream.fileno())
        paths[name] = target
    return paths


@dataclass
class R41DiagnosticOnlineReleaseContext:
    runtime: object
    scenarios: dict
    explainer: object
    question_bank: object
    tutorial: dict
    tutorial_signature: str
    evidence: dict
    source_binding: dict
    manifest_sha256: str
    signature: str
    release: dict
    provenance: dict
    root: Path
    package_sha256: str
    _temporary: object = field(repr=False)
    closed: bool = False

    def close(self):
        if not self.closed:
            self.closed = True
            self._temporary.cleanup()


def load_online_release(*, expected_package_sha256: str,
                        expected_manifest_sha256: str,
                        package_path=None, base64_path=None):
    raw = _package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, artifacts = _read_archive(
        raw, expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256)
    for name in ("protocol", "runtime_manifest", "question_bank", "tutorial"):
        _reject_sensitive(_parse_json(artifacts[name], "v9 " + name),
                          "v9 " + name)
    temporary = tempfile.TemporaryDirectory(prefix="warehouse-r41-diagnostic-v9-")
    root = Path(temporary.name)
    try:
        paths = _write_material(root, artifacts)
        identities = manifest["identities"]
        runtime_manifest = _parse_json(
            artifacts["runtime_manifest"], "v9 runtime manifest")
        content = deepcopy(runtime_manifest); claimed = content.pop("content_sha256", None)
        if runtime_manifest.get("version") != PORTABLE_RUNTIME_MANIFEST_VERSION:
            raise ValueError("Exact portable v9 runtime manifest required")
        runtime = R41DiagnosticOnlineAlignmentRuntime(
            paths["actor"], training_protocol_path=paths["protocol"],
            manifest_path=paths["runtime_manifest"],
            expected_actor_sha256=identities["actor_sha256"],
            expected_training_protocol_file_sha256=identities[
                "protocol_file_sha256"],
            expected_training_protocol_content_sha256=identities[
                "protocol_content_sha256"],
            expected_manifest_file_sha256=identities[
                "runtime_manifest_file_sha256"],
            expected_manifest_content_sha256=claimed,
            expected_manifest_semantic_sha256=digest(runtime_manifest))
        runtime.verify_binding()
        if (runtime.signature != identities["runtime_signature"]
                or runtime.runtime_manifest_signature
                    != identities["runtime_manifest_signature"]):
            raise ValueError("V9 serving runtime identity differs")
        program, header = compact_api.decode_program(
            artifacts["program"],
            expected_compact_sha256=identities["compact_program_sha256"],
            expected_source_program_sha256=identities["program_sha256"],
            expected_actor_feature_names_sha256=identities[
                "actor_feature_names_sha256"],
            expected_public_feature_contract_sha256=identities[
                "public_feature_contract_sha256"])
        program_raw = compact_api.program_json_bytes(program)
        if (sha256(program_raw).hexdigest() != identities["runtime_program_sha256"]
                or digest(program.to_dict())
                    != identities["runtime_program_content_sha256"]
                or header["source"]["content_sha256"]
                    != identities["program_content_sha256"]):
            raise ValueError("V9 compact program identity differs")
        program_path = root / "runtime-program.json"
        descriptor = os.open(program_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(program_raw); stream.flush(); os.fsync(stream.fileno())
        artifact_binding = explanation_api.make_artifact_binding(
            actor_sha256=runtime.actor_sha256,
            program_sha256=identities["runtime_program_sha256"],
            runtime_signature=runtime.signature,
            runtime_manifest_sha256=runtime.manifest_file_sha256,
            candidate_lock_sha256=identities["candidate_lock_sha256"],
            public_feature_contract_sha256=identities[
                "public_feature_contract_sha256"])
        explainer = explanation_api.R41DiagnosticOnlineAlignmentExplainerV9(
            program_path,
            expected_program_sha256=identities["runtime_program_sha256"],
            runtime=runtime, artifact_binding=artifact_binding)
        question = _parse_json(artifacts["question_bank"], "v9 question bank")
        bank_identities = {
            "parent_runtime_signature": identities["runtime_signature"],
            "question_bank_private_items_sha256": identities[
                "question_bank_private_items_sha256"],
            "question_bank_public_items_sha256": identities[
                "question_bank_public_items_sha256"],
            "parent_question_bank_signature": identities[
                "question_bank_signature"],
        }
        bank = v8_release._diagnostic_portable_bank(
            question, runtime=runtime, identities=bank_identities,
            raw=artifacts["question_bank"])
        tutorial = _parse_json(artifacts["tutorial"], "v9 tutorial")
        play = runtime_manifest.get("splits", {}).get("play")
        if not isinstance(play, list) or len(play) != 7:
            raise ValueError("V9 play scenes differ")
        tutorial_scene = play[0]
        expected_tutorial = {
            "scene_manifest_version": runtime_manifest[
                "source_full_manifest_version"],
            "scene_manifest_file_sha256": runtime_manifest[
                "source_full_manifest_file_sha256"],
            "scene_manifest_content_sha256": runtime_manifest[
                "source_full_manifest_content_sha256"],
            "scene_manifest_semantic_sha256": runtime_manifest[
                "source_full_manifest_semantic_sha256"],
            "tutorial_scene_fingerprint": tutorial_scene["fingerprint"],
            "tutorial_successor_state_sha256": tutorial_scene["snapshot"]
                ["r41_diagnostic_conflict"]["binding_sha256"],
            "tutorial_snapshot_sha256": digest(tutorial_scene["snapshot"]),
            "diagnostic_contract_sha256": runtime_manifest[
                "diagnostic_contract_sha256"],
            "diagnostic_contract_version": runtime_manifest[
                "diagnostic_contract_version"],
            "diagnostic_conflict_graph_sha256": runtime_manifest[
                "diagnostic_conflict_graph_sha256"],
            "conflict_families_sha256": runtime_manifest[
                "conflict_families_sha256"],
            "producer_sources_sha256": digest(tutorial_api.producer_sources()),
        }
        tutorial_replay = tutorial_api.validate_neutral_tutorial(
            tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
            expected_bindings=expected_tutorial)
        if (digest(tutorial) != identities["tutorial_signature"]
                or tutorial_replay["tutorial_signature"]
                    != identities["tutorial_signature"]):
            raise ValueError("V9 tutorial identity differs")
        source_binding = deepcopy(manifest["sources"]["release"])
        evidence = {
            "version": VERSION,
            "parent": deepcopy(manifest["parent"]),
            "identities": deepcopy(identities),
            "tutorial_replay": tutorial_replay,
            "runtime_action_override": False,
            "formal_ready": False,
        }
        provenance = {
            "version": VERSION,
            "release_version": PUBLIC_RELEASE_VERSION,
            "namespace": "internal_diagnostic",
            "pilot_class": PILOT_CLASS,
            "manifest_sha256": expected_manifest_sha256,
            "package_sha256": expected_package_sha256,
            "diagnostic_admission_sha256": manifest["parent"]["admission_sha256"],
            "actor_sha256": runtime.actor_sha256,
            "runtime_signature": runtime.signature,
            "runtime_manifest_signature": runtime.runtime_manifest_signature,
            "program_sha256": identities["program_sha256"],
            "runtime_program_sha256": explainer.program_sha256,
            "question_bank_signature": bank.signature,
            "tutorial_signature": identities["tutorial_signature"],
            "source_binding": source_binding,
            "release": deepcopy(manifest["release"]),
            "online_portable": True,
            "formal_ready": False,
            "formal_sample_eligible": False,
            "data_persistent": False,
        }
        signature = digest({
            "version": VERSION, "manifest_sha256": expected_manifest_sha256,
            "package_sha256": expected_package_sha256,
            "runtime_signature": runtime.signature,
            "program_sha256": explainer.program_sha256,
            "question_bank_signature": bank.signature,
            "tutorial_signature": identities["tutorial_signature"],
            "sources": source_binding,
        })
        context = R41DiagnosticOnlineReleaseContext(
            runtime, deepcopy(runtime_manifest), explainer, bank,
            deepcopy(tutorial), identities["tutorial_signature"], evidence,
            source_binding, expected_manifest_sha256, signature,
            deepcopy(manifest["release"]), provenance, root,
            expected_package_sha256, temporary)
        return validate_bound_context(
            context, expected_package_sha256=expected_package_sha256,
            expected_manifest_sha256=expected_manifest_sha256)
    except BaseException:
        temporary.cleanup()
        raise


def validate_bound_context(context: Any, *, expected_package_sha256: str,
                           expected_manifest_sha256: str) -> Any:
    if (_HEX.fullmatch(str(expected_package_sha256)) is None
            or _HEX.fullmatch(str(expected_manifest_sha256)) is None):
        raise ValueError("Exact v9 package and manifest hashes required")
    release = getattr(context, "release", None)
    provenance = getattr(context, "provenance", None)
    explainer = getattr(context, "explainer", None)
    if (not isinstance(release, Mapping) or not isinstance(provenance, Mapping)
            or provenance.get("version") != VERSION
            or provenance.get("package_sha256") != expected_package_sha256
            or provenance.get("manifest_sha256") != expected_manifest_sha256
            or release != release_projection(ready=True)
            or type(explainer).__module__
                != "backend.warehouse_r41_diagnostic_online_explanation_v9"
            or type(explainer).__name__
                != "R41DiagnosticOnlineAlignmentExplainerV9"
            or not callable(getattr(explainer, "answer_study", None))
            or not isinstance(getattr(explainer, "artifact_binding", None), Mapping)
            or not callable(getattr(context, "close", None))):
        raise ValueError("Admitted v9 online context differs from its release boundary")
    return context


def _assemble_admitted(*, admission: Mapping[str, Any],
        admission_gate_names: tuple[str, ...],
        admission_label: str,
        expected_diagnostic_admission_sha256: str,
        components: Mapping[str, str | Path],
        output_package: str | Path,
        output_base64: str | Path | None = None) -> dict[str, Any]:
    """Package one already re-authenticated diagnostic admission."""
    if (admission.get("admitted") is not True
            or admission.get("gates")
                != {name: True for name in admission_gate_names}):
        raise ValueError("Passing " + admission_label + " admission required")
    paths = {name: Path(value).expanduser().absolute()
             for name, value in components.items()}
    packaged = {
        "actor": paths["actor"].read_bytes(),
        "protocol": paths.get("runtime_protocol", paths["protocol"]).read_bytes(),
        "runtime_manifest": paths["runtime_manifest"].read_bytes(),
        "program": paths["compact_program"].read_bytes(),
        "question_bank": paths["question_bank"].read_bytes(),
        "tutorial": paths["tutorial"].read_bytes(),
    }
    bindings = admission["bindings"]
    identity_names = {
        "actor_sha256", "protocol_file_sha256", "protocol_content_sha256",
        "runtime_manifest_file_sha256", "runtime_manifest_content_sha256",
        "runtime_manifest_semantic_sha256", "runtime_signature",
        "runtime_manifest_signature", "candidate_lock_sha256",
        "program_sha256", "program_content_sha256", "compact_program_sha256",
        "runtime_program_sha256", "runtime_program_content_sha256",
        "actor_feature_names_sha256", "public_feature_contract_sha256",
        "question_bank_sha256", "question_bank_private_items_sha256",
        "question_bank_public_items_sha256", "question_bank_signature",
        "tutorial_sha256", "tutorial_signature", "release_sources_sha256",
        "designation_sha256", "outer_result_sha256", "final_audit_sha256",
    }
    identities = {name: bindings[name] for name in identity_names}
    parent = {
        "version": admission["version"], "status": admission["status"],
        "admission_sha256": expected_diagnostic_admission_sha256,
        "admission_content_sha256": admission["content_sha256"],
        "bindings_sha256": digest(bindings),
        "gates_sha256": digest(admission["gates"]),
    }
    manifest = {
        "version": VERSION, "status": STATUS, "test_fixture": False,
        "pilot_class": PILOT_CLASS, "formal_ready": False,
        "formal_sample_eligible": False, "data_persistent": False,
        "parent": parent, "artifacts": _artifact_records(packaged),
        "identities": identities,
        "play_scenes": deepcopy(admission["play_scenes"]),
        "sources": {"release": release_sources()},
        "analysis": portable._analysis_protocol(),
        "release": release_projection(ready=True),
    }
    _reject_sensitive(manifest, "v9 release manifest")
    raw = _archive_bytes(manifest, packaged)
    encoded = base64.b64encode(raw) + b"\n"
    if len(encoded) > MAX_BASE64_BYTES:
        raise ValueError("V9 Base64 Secret File exceeds 1,000,000 bytes")
    package = portable._write_new(output_package, raw)
    encoded_path = None
    try:
        if output_base64 is not None:
            encoded_path = portable._write_new(output_base64, encoded)
        manifest_sha = sha256((canonical(manifest) + "\n").encode("utf-8")).hexdigest()
        package_sha = sha256(raw).hexdigest()
        loaded = load_online_release(
            package_path=package, expected_package_sha256=package_sha,
            expected_manifest_sha256=manifest_sha)
        loaded.close()
    except BaseException:
        package.unlink(missing_ok=True)
        if encoded_path is not None:
            encoded_path.unlink(missing_ok=True)
        raise
    return {
        "version": VERSION,
        "status": ("built_from_" + admission_label
                   + "_admission_and_independently_reloaded"),
        "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": PILOT_CLASS,
        "package": str(package), "package_size": len(raw),
        "package_sha256": package_sha,
        "base64": str(encoded_path) if encoded_path else None,
        "base64_size": len(encoded), "manifest_sha256": manifest_sha,
        "diagnostic_admission_sha256": expected_diagnostic_admission_sha256,
        "actor_sha256": identities["actor_sha256"],
        "program_sha256": identities["program_sha256"],
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False,
    }


def assemble_from_admitted_components(*,
        diagnostic_admission_path: str | Path,
        expected_diagnostic_admission_sha256: str,
        components: Mapping[str, str | Path],
        outer_permanent_registry: str | Path,
        final_permanent_registry: str | Path,
        promoted_v11_permanent_registry: str | Path,
        output_package: str | Path,
        output_base64: str | Path | None = None) -> dict[str, Any]:
    """Build only after the complete immutable v9 admission rereads cleanly."""

    from backend.training import warehouse_r41_diagnostic_admission_v9 as admission_api
    admission = admission_api.read_saved_admission(
        diagnostic_admission_path,
        expected_sha256=_sha(expected_diagnostic_admission_sha256,
                             "v9 admission"),
        components=components,
        outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry,
        promoted_v11_permanent_registry=promoted_v11_permanent_registry)
    return _assemble_admitted(
        admission=admission, admission_gate_names=admission_api.GATE_NAMES,
        admission_label="v9",
        expected_diagnostic_admission_sha256=(
            expected_diagnostic_admission_sha256),
        components=components, output_package=output_package,
        output_base64=output_base64)


def assemble_from_v10_admitted_components(*,
        diagnostic_admission_path: str | Path,
        expected_diagnostic_admission_sha256: str,
        components: Mapping[str, str | Path],
        outer_permanent_registry: str | Path,
        final_permanent_registry: str | Path,
        permanent_promotion_closeout_registry: str | Path,
        output_package: str | Path,
        output_base64: str | Path | None = None) -> dict[str, Any]:
    """Build the v9 runtime package from a passed v13-evidence admission."""

    from backend.training import warehouse_r41_diagnostic_admission_v10 as admission_api
    admission = admission_api.read_saved_admission(
        diagnostic_admission_path,
        expected_sha256=_sha(expected_diagnostic_admission_sha256,
                             "v10 admission"),
        components=components,
        outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry,
        permanent_promotion_closeout_registry=(
            permanent_promotion_closeout_registry))
    return _assemble_admitted(
        admission=admission, admission_gate_names=admission_api.GATE_NAMES,
        admission_label="v10",
        expected_diagnostic_admission_sha256=(
            expected_diagnostic_admission_sha256),
        components=components, output_package=output_package,
        output_base64=output_base64)


def assemble_from_v11_admitted_components(*,
        diagnostic_admission_path: str | Path,
        expected_diagnostic_admission_sha256: str,
        components: Mapping[str, str | Path],
        outer_permanent_registry: str | Path,
        final_permanent_registry: str | Path,
        permanent_promotion_closeout_registry: str | Path,
        output_package: str | Path,
        output_base64: str | Path | None = None) -> dict[str, Any]:
    """Build the v9 runtime package from a passed v14-evidence admission."""

    from backend.training import warehouse_r41_diagnostic_admission_v11 as admission_api
    admission = admission_api.read_saved_admission(
        diagnostic_admission_path,
        expected_sha256=_sha(expected_diagnostic_admission_sha256,
                             "v11 admission"),
        components=components,
        outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry,
        permanent_promotion_closeout_registry=(
            permanent_promotion_closeout_registry))
    return _assemble_admitted(
        admission=admission, admission_gate_names=admission_api.GATE_NAMES,
        admission_label="v11",
        expected_diagnostic_admission_sha256=(
            expected_diagnostic_admission_sha256),
        components=components, output_package=output_package,
        output_base64=output_base64)


__all__ = [
    "VERSION", "STATUS", "PUBLIC_RELEASE_VERSION", "PILOT_CLASS",
    "ANIMATION_DURATION_MS", "MANIFEST_NAME", "ARTIFACT_PATHS",
    "ARCHIVE_WHITELIST", "MAX_PACKAGE_BYTES", "MAX_BASE64_BYTES",
    "ARCHIVE_COMPRESSION", "ARCHIVE_COMPRESSLEVEL",
    "R41DiagnosticOnlineReleaseContext", "release_sources",
    "release_projection", "validate_bound_context", "_archive_bytes",
    "_read_archive", "inspect_online_release", "load_online_release",
    "assemble_from_admitted_components", "assemble_from_v10_admitted_components",
    "assemble_from_v11_admitted_components",
]
