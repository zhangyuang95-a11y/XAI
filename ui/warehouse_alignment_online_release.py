"""Portable, hash-bound online release for the qualified warehouse pilot.

The local release deliberately keeps the complete capability, explanation and
question-bank evidence beside the service.  A hosted service does not need to
replay those completed audits on every cold start.  This module therefore
exports a small *projection* from an already fully admitted release context and
reconstructs only the objects needed by the lightweight online study service:

* the frozen NumPy Actor and its complete protocol;
* the twelve frozen participant play scenes;
* the qualified executable RCPD program; and
* the eight privately gradeable questionnaire items.

The archive is an immutable deployment derivative, not a new qualification.
Its manifest binds the parent release, every current source file used by the
parent service, and every byte in a fixed archive whitelist.  Loading never
reads the parent's absolute evidence paths or its large validation journals.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


VERSION = "warehouse-alignment-online-release.v1"
STATUS = "online_portable_local_pilot"
MANIFEST_NAME = "manifest.json"
ARTIFACT_PATHS = {
    "actor": "artifacts/actor.npz",
    "protocol": "artifacts/protocol.json",
    "play_scenarios": "artifacts/play_scenarios.json",
    "program": "artifacts/program.json",
    "question_bank": "artifacts/question_bank.json",
}
ARCHIVE_WHITELIST = frozenset((MANIFEST_NAME, *ARTIFACT_PATHS.values()))

# A Render Secret File is limited to 1 MiB.  Keeping the encoded derivative at
# or below one million bytes also leaves room for a trailing newline.
MAX_PACKAGE_BYTES = 750_000
MAX_BASE64_BYTES = 1_000_000
MAX_MANIFEST_BYTES = 512_000
MAX_ARTIFACT_BYTES = {
    "actor": 512_000,
    "protocol": 4_000_000,
    "play_scenarios": 2_000_000,
    "program": 1_000_000,
    "question_bank": 5_000_000,
}
MAX_UNCOMPRESSED_BYTES = MAX_MANIFEST_BYTES + sum(MAX_ARTIFACT_BYTES.values())

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = frozenset((
    "version", "status", "test_fixture", "formal_ready", "parent",
    "artifacts", "identities", "sources", "analysis", "release",
))
_PARENT_FIELDS = frozenset((
    "version", "status", "manifest_sha256", "context_signature",
    "source_binding_sha256", "scenario_manifest_sha256",
))
_SOURCE_FIELDS = frozenset(("parent", "portable"))
_IDENTITY_FIELDS = frozenset((
    "actor_sha256", "protocol_sha256", "program_sha256",
    "parent_runtime_signature", "parent_explainer_signature",
    "parent_question_bank_signature",
    "question_bank_private_items_sha256", "question_bank_public_items_sha256",
    "portable_scenarios_sha256", "play_scene_count", "play_scene_ids",
))
_RELEASE_FIELDS = frozenset((
    "status", "namespace", "model_ready", "explanation_ready", "study_ready",
    "participant_enabled", "explanation_qualified", "release_ready",
    "web_integration_completed", "formal_ready", "test_fixture",
    "qualification_evaluated", "online_portable", "message",
))
_QUESTION_FIELDS = frozenset((
    "version", "test_fixture", "source_bank_signature", "anchors",
    "bank_sha256", "replay_receipt_sha256", "pool_manifest_sha256",
    "actor_sha256", "runtime_signature", "protocol_sha256", "runtime_family",
    "sources", "checks", "items", "private_items_sha256",
    "public_items_sha256", "summary_sha256",
))
_ITEM_FIELDS = frozenset((
    "id", "kind", "scenario_id", "frame", "snapshot", "snapshot_sha256",
    "preview", "answer", "diversity_key", "evidence", "options", "prompt",
))


def _sha(value, label="SHA256"):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(f"Explicit lowercase SHA256 required: {label}")
    return value


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def _parse_json(raw, label):
    if type(raw) is not bytes:
        raise ValueError(label + " bytes required")

    def reject(value):
        raise ValueError("Non-finite JSON value in " + label + ": " + value)

    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid JSON: " + label) from error
    if type(value) is not dict:
        raise ValueError("Top-level JSON object required: " + label)
    return value


def _portable_sources():
    paths = (
        Path(__file__),
        ROOT / "scripts/build_warehouse_alignment_online_release.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "backend/warehouse_alignment_online_explanation.py",
        ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "requirements-render.txt",
    )
    result = {}
    for path in paths:
        if not path.is_file():
            raise ValueError("Portable release source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _validate_source_map(values, label):
    if type(values) is not dict or not values:
        raise ValueError(label + " source binding is required")
    clean = {}
    for name, expected in values.items():
        part = PurePosixPath(name) if type(name) is str else None
        if (part is None or part.is_absolute() or ".." in part.parts
                or part.as_posix() != name):
            raise ValueError("Unsafe source path in " + label)
        _sha(expected, label + " source")
        path = ROOT.joinpath(*part.parts)
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError("Missing regular source in " + label + ": " + name)
        if file_hash(path) != expected:
            raise ValueError("Source changed in " + label + ": " + name)
        clean[name] = expected
    return dict(sorted(clean.items()))


def _analysis_protocol():
    """Frozen A/B projection, repeated here to keep online imports lightweight."""
    return {
        "version": "warehouse-native-local-ab-analysis.v1",
        "primary": {
            "metric": "task2_mean_deliveries",
            "aggregation": "arithmetic_mean_of_three_unique_ended_rounds",
            "missing_rounds": "do_not_impute",
            "explicit_early_end": "retain_record",
        },
        "rounds_per_task": {"task1": 3, "task2": 3},
        "task_order": ["XY", "YX"],
        "condition_access": {
            group: {
                "task1_explanation": group == "A",
                "task1_review_explanation": group == "A",
                "task2_explanation": False,
                "task2_old_answers": False,
            }
            for group in ("A", "B")
        },
        "same_actor_ab": True,
        "stage_difference": "descriptive_only",
        "detour_metrics": False,
        "namespace": "local_pilot",
        "formal_ready": False,
        "questionnaire": {"next_action": 4, "wait_three": 4},
    }


def _artifact_records(artifacts):
    if type(artifacts) is not dict or set(artifacts) != set(ARTIFACT_PATHS):
        raise ValueError("Exact portable artifact set required")
    records = {}
    for name, path in ARTIFACT_PATHS.items():
        raw = artifacts[name]
        if type(raw) is not bytes or not raw or len(raw) > MAX_ARTIFACT_BYTES[name]:
            raise ValueError("Portable artifact size is invalid: " + name)
        records[name] = {
            "path": path,
            "size": len(raw),
            "sha256": sha256(raw).hexdigest(),
        }
    return records


def _release_projection(message=None):
    return {
        "status": "local_pilot_technically_verified",
        "namespace": "local_pilot",
        "model_ready": True,
        "explanation_ready": True,
        "study_ready": True,
        "participant_enabled": True,
        "explanation_qualified": True,
        "release_ready": True,
        "web_integration_completed": True,
        "formal_ready": False,
        "test_fixture": False,
        "qualification_evaluated": True,
        "online_portable": True,
        "message": deepcopy(message) if type(message) is dict else {
            "zh": "在线预实验：模型、解释系统与问卷证据已通过技术验收；正式研究结论仍待人类实验。",
            "en": "Online pilot: model, explanation system, and questionnaire evidence passed technical acceptance; formal conclusions remain pending.",
        },
    }


def _question_projection(bank):
    from ui.warehouse_alignment_metadata_bank_view import FrozenAlignmentMetadataBank
    if type(bank) is not FrozenAlignmentMetadataBank:
        raise ValueError("A fully admitted frozen alignment bank is required")
    bank.verify_binding()
    items = bank.items
    public_items = bank.public_items()
    checks = bank.checks
    anchors = deepcopy(getattr(bank, "_anchors", None))
    if type(anchors) is not dict or not anchors:
        raise ValueError("Frozen bank anchors are unavailable")
    return {
        "version": "warehouse-alignment-portable-question-projection.v1",
        "test_fixture": False,
        "source_bank_signature": bank.signature,
        "anchors": anchors,
        "bank_sha256": bank.bank_sha256,
        "replay_receipt_sha256": bank.replay_receipt_sha256,
        "pool_manifest_sha256": bank.pool_manifest_sha256,
        "actor_sha256": bank.actor_sha256,
        "runtime_signature": bank.runtime_signature,
        "protocol_sha256": bank.protocol_sha256,
        "runtime_family": bank.summary()["runtime_family"],
        "sources": bank.sources,
        "checks": checks,
        "items": items,
        "private_items_sha256": digest(items),
        "public_items_sha256": digest(public_items),
        "summary_sha256": digest(bank.summary()),
    }


def _validate_question_projection(value, identities):
    if (type(value) is not dict or set(value) != _QUESTION_FIELDS
            or value.get("version") != "warehouse-alignment-portable-question-projection.v1"
            or value.get("test_fixture") is not False):
        raise ValueError("Exact portable question projection required")
    for name in ("source_bank_signature", "bank_sha256", "replay_receipt_sha256",
                 "pool_manifest_sha256", "actor_sha256", "runtime_signature",
                 "protocol_sha256", "private_items_sha256", "public_items_sha256",
                 "summary_sha256"):
        _sha(value.get(name), "question " + name)
    if (value["source_bank_signature"] != identities["parent_question_bank_signature"]
            or value["actor_sha256"] != identities["actor_sha256"]
            or value["runtime_signature"] != identities["parent_runtime_signature"]
            or value["protocol_sha256"] != identities["protocol_sha256"]):
        raise ValueError("Question projection belongs to another runtime or bank")
    sources = _validate_source_map(value.get("sources"), "question bank")
    anchors = value.get("anchors")
    if type(anchors) is not dict or not anchors:
        raise ValueError("Question projection anchors are missing")
    for name, expected in anchors.items():
        if type(name) is not str or not name:
            raise ValueError("Invalid question anchor name")
        _sha(expected, "question anchor " + name)
    items = value.get("items")
    required_ids = {
        *(f"prediction_next_action_{index}" for index in range(1, 5)),
        *(f"prediction_wait_three_{index}" for index in range(1, 5)),
    }
    if (type(items) is not list or len(items) != 8
            or any(type(item) is not dict or set(item) != _ITEM_FIELDS for item in items)
            or {item.get("id") for item in items} != required_ids):
        raise ValueError("Exactly eight complete projected questions are required")
    for item in items:
        kind, frame = item.get("kind"), item.get("frame")
        options = item.get("options")
        if (kind not in ("next_action", "wait_three") or type(frame) is not int or frame < 1
                or type(item.get("snapshot")) is not dict
                or digest(item["snapshot"]) != item.get("snapshot_sha256")
                or item["snapshot"].get("state", {}).get("frame") != frame
                or type(item.get("prompt")) is not dict
                or set(item["prompt"]) != {"zh", "en"}
                or type(options) is not list
                or len(options) != (5 if kind == "next_action" else 4)
                or any(type(option) is not dict
                       or set(option) != ({"value", "label"} if kind == "next_action"
                                          else {"value", "label", "marker", "position"})
                       or type(option["value"]) is not str
                       or type(option["label"]) is not dict
                       or set(option["label"]) != {"zh", "en"}
                       or (kind == "wait_three" and (
                           type(option["marker"]) is not str
                           or type(option["position"]) is not list
                           or len(option["position"]) != 2
                           or any(type(coordinate) is not int
                                  for coordinate in option["position"])))
                       for option in options)):
            raise ValueError("Projected question content is invalid")
        values = [option["value"] for option in options]
        if len(set(values)) != len(values) or item.get("answer") not in values:
            raise ValueError("Projected answer or options are invalid")
    if (digest(items) != value["private_items_sha256"]
            or digest(items) != identities["question_bank_private_items_sha256"]):
        raise ValueError("Projected private question items changed")
    return value, sources


def _archive_bytes(manifest, artifacts):
    records = _artifact_records(artifacts)
    if manifest.get("artifacts") != records:
        raise ValueError("Manifest artifact records differ")
    members = {MANIFEST_NAME: (canonical(manifest) + "\n").encode()}
    members.update({ARTIFACT_PATHS[name]: raw for name, raw in artifacts.items()})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, strict_timestamps=True) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_STORED if name == ARTIFACT_PATHS["actor"] else zipfile.ZIP_DEFLATED
            archive.writestr(info, members[name], compress_type=info.compress_type,
                             compresslevel=9 if info.compress_type == zipfile.ZIP_DEFLATED else None)
    raw = output.getvalue()
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Portable package exceeds the safe deployment limit")
    return raw


def _safe_output(path):
    value = Path(path).expanduser().absolute()
    if (value.resolve() != value or value.exists() or value.parent.resolve() != value.parent
            or not value.parent.is_dir() or value.parent.is_symlink()):
        raise ValueError("Use a new canonical output file")
    return value


def _write_new(path, raw, mode=0o600):
    path = _safe_output(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return path


def build_online_release(context, *, output_package, output_base64=None):
    """Export a portable archive from an already fully loaded release context."""
    if getattr(context, "closed", False):
        raise ValueError("Parent release context is closed")
    if (type(getattr(context, "manifest_sha256", None)) is not str
            or type(getattr(context, "source_binding", None)) is not dict
            or type(getattr(context, "provenance", None)) is not dict):
        raise ValueError("A fully loaded parent release context is required")
    parent_manifest = _sha(context.manifest_sha256, "parent manifest")
    release = context.release
    if (type(release) is not dict
            or release.get("status") != "local_pilot_technically_verified"
            or release.get("test_fixture") is not False
            or release.get("formal_ready") is not False
            or any(release.get(name) is not True for name in (
                "model_ready", "explanation_ready", "study_ready",
                "participant_enabled", "explanation_qualified", "release_ready",
            ))):
        raise ValueError("Qualified non-formal parent release is required")
    from ui import warehouse_alignment_server as server_api
    identity, bank_identity, required = server_api._components(
        context.runtime, context.explainer, context.scenarios,
        expected_scenarios=context.provenance.get("scenario_manifest_sha256"),
        bank=context.question_bank, required_bank=True,
    )
    if (identity.get("runtime_signature") != context.runtime.signature
            or bank_identity.get("signature") != context.question_bank.signature):
        raise ValueError("Parent component identities changed during export")
    parent_sources = server_api.previous._sources(context.source_binding, required)
    portable_sources = _portable_sources()
    if set(parent_sources) & set(portable_sources):
        raise ValueError("Portable and parent source bindings overlap")

    actor_raw = Path(context.runtime._actor_path).read_bytes()
    protocol_raw = (canonical(context.runtime.protocol) + "\n").encode()
    program_raw = Path(context.explainer.program_path).read_bytes()
    play = deepcopy(context.scenarios.get("splits", {}).get("play"))
    if (type(play) is not list or len(play) < 7 or len({scene.get("id") for scene in play}) != len(play)):
        raise ValueError("Unique parent play scenes are required")
    portable_scenarios = {
        "version": "warehouse-alignment-portable-play-scenes.v1",
        "test_fixture": False,
        "source_scenario_manifest_sha256": context.provenance["scenario_manifest_sha256"],
        "splits": {"play": play},
    }
    scenario_raw = (canonical(portable_scenarios) + "\n").encode()
    questions = _question_projection(context.question_bank)
    question_raw = (canonical(questions) + "\n").encode()
    artifacts = {
        "actor": actor_raw,
        "protocol": protocol_raw,
        "play_scenarios": scenario_raw,
        "program": program_raw,
        "question_bank": question_raw,
    }
    records = _artifact_records(artifacts)
    public_items = context.question_bank.public_items()
    identities = {
        "actor_sha256": context.runtime.actor_sha256,
        "protocol_sha256": context.runtime.protocol_sha256,
        "program_sha256": context.explainer.program_sha256,
        "parent_runtime_signature": context.runtime.signature,
        "parent_explainer_signature": context.explainer.signature,
        "parent_question_bank_signature": context.question_bank.signature,
        "question_bank_private_items_sha256": digest(context.question_bank.items),
        "question_bank_public_items_sha256": digest(public_items),
        "portable_scenarios_sha256": digest(portable_scenarios),
        "play_scene_count": len(play),
        "play_scene_ids": [scene["id"] for scene in play],
    }
    manifest = {
        "version": VERSION,
        "status": STATUS,
        "test_fixture": False,
        "formal_ready": False,
        "parent": {
            "version": context.provenance.get("version"),
            "status": release["status"],
            "manifest_sha256": parent_manifest,
            "context_signature": _sha(context.signature, "parent context"),
            "source_binding_sha256": digest(parent_sources),
            "scenario_manifest_sha256": context.provenance["scenario_manifest_sha256"],
        },
        "artifacts": records,
        "identities": identities,
        "sources": {"parent": parent_sources, "portable": portable_sources},
        "analysis": deepcopy(context.provenance.get("analysis")),
        "release": _release_projection(release.get("message")),
    }
    _validate_manifest(manifest)
    package_raw = _archive_bytes(manifest, artifacts)
    manifest_raw = (canonical(manifest) + "\n").encode()
    target = _write_new(output_package, package_raw)
    encoded_target = None
    encoded = base64.b64encode(package_raw) + b"\n"
    if len(encoded) > MAX_BASE64_BYTES:
        target.unlink(missing_ok=True)
        raise ValueError("Base64 Secret File exceeds the deployment limit")
    if output_base64 is not None:
        try:
            encoded_target = _write_new(output_base64, encoded)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return {
        "version": VERSION,
        "status": STATUS,
        "package": str(target),
        "package_size": len(package_raw),
        "package_sha256": sha256(package_raw).hexdigest(),
        "base64": str(encoded_target) if encoded_target else None,
        "base64_size": len(encoded),
        "manifest_sha256": sha256(manifest_raw).hexdigest(),
        "parent_manifest_sha256": parent_manifest,
        "play_scene_count": len(play),
        "question_count": len(public_items),
        "formal_ready": False,
    }


def _validate_manifest(manifest):
    if (type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS
            or manifest.get("version") != VERSION or manifest.get("status") != STATUS
            or manifest.get("test_fixture") is not False
            or manifest.get("formal_ready") is not False):
        raise ValueError("Exact non-formal portable manifest required")
    parent = manifest.get("parent")
    if (type(parent) is not dict or set(parent) != _PARENT_FIELDS
            or parent.get("status") != "local_pilot_technically_verified"):
        raise ValueError("Portable manifest parent binding differs")
    for name in ("manifest_sha256", "context_signature", "source_binding_sha256",
                 "scenario_manifest_sha256"):
        _sha(parent.get(name), "parent " + name)
    identities = manifest.get("identities")
    if type(identities) is not dict or set(identities) != _IDENTITY_FIELDS:
        raise ValueError("Portable identity binding differs")
    for name in _IDENTITY_FIELDS - {"play_scene_count", "play_scene_ids"}:
        _sha(identities.get(name), "identity " + name)
    ids = identities.get("play_scene_ids")
    if (type(identities.get("play_scene_count")) is not int
            or identities["play_scene_count"] < 7
            or type(ids) is not list or len(ids) != identities["play_scene_count"]
            or len(set(ids)) != len(ids)
            or any(type(value) is not str or not value for value in ids)):
        raise ValueError("Portable play-scene identity differs")
    records = manifest.get("artifacts")
    if type(records) is not dict or set(records) != set(ARTIFACT_PATHS):
        raise ValueError("Portable artifact manifest differs")
    for name, expected_path in ARTIFACT_PATHS.items():
        record = records[name]
        if (type(record) is not dict or set(record) != {"path", "size", "sha256"}
                or record.get("path") != expected_path
                or type(record.get("size")) is not int
                or not 0 < record["size"] <= MAX_ARTIFACT_BYTES[name]):
            raise ValueError("Portable artifact record differs: " + name)
        _sha(record.get("sha256"), "artifact " + name)
    sources = manifest.get("sources")
    if type(sources) is not dict or set(sources) != _SOURCE_FIELDS:
        raise ValueError("Portable source groups differ")
    parent_sources = _validate_source_map(sources["parent"], "parent")
    portable_sources = _validate_source_map(sources["portable"], "portable")
    if set(parent_sources) & set(portable_sources):
        raise ValueError("Portable source groups overlap")
    if digest(parent_sources) != parent["source_binding_sha256"]:
        raise ValueError("Parent source binding changed")
    if canonical(manifest.get("analysis")) != canonical(_analysis_protocol()):
        raise ValueError("Frozen A/B analysis contract changed")
    release = manifest.get("release")
    if (type(release) is not dict or set(release) != _RELEASE_FIELDS
            or release.get("status") != "local_pilot_technically_verified"
            or release.get("namespace") != "local_pilot"
            or release.get("formal_ready") is not False
            or release.get("test_fixture") is not False
            or release.get("online_portable") is not True
            or release.get("qualification_evaluated") is not True
            or any(release.get(name) is not True for name in (
                "model_ready", "explanation_ready", "study_ready",
                "participant_enabled", "explanation_qualified", "release_ready",
                "web_integration_completed",
            )) or type(release.get("message")) is not dict
            or set(release["message"]) != {"zh", "en"}):
        raise ValueError("Portable release flags differ")
    return manifest


def _read_bounded(path, limit, label):
    value = Path(path).expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(value, flags)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("Readable " + label + " file required") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Regular " + label + " file required")
        size = before.st_size
        if size <= 0 or size > limit:
            raise ValueError(label + " file exceeds the safe limit")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                  value.st_mtime_ns)
        if identity(before) != identity(after) or len(raw) != size or len(raw) > limit:
            raise ValueError(label + " changed or exceeds the safe limit")
        return raw
    finally:
        os.close(descriptor)


def _package_bytes(*, package_path=None, base64_path=None):
    if (package_path is None) == (base64_path is None):
        raise ValueError("Choose exactly one local package or Base64 Secret File")
    if package_path is not None:
        return _read_bounded(package_path, MAX_PACKAGE_BYTES, "package")
    encoded = b"".join(_read_bounded(base64_path, MAX_BASE64_BYTES, "Base64 secret").split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Invalid Base64 Secret File") from error
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Decoded package exceeds the safe limit")
    return raw


def _read_archive(raw, *, expected_package_sha256, expected_manifest_sha256):
    if type(raw) is not bytes or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Bound portable package bytes required")
    if sha256(raw).hexdigest() != _sha(expected_package_sha256, "package"):
        raise ValueError("Portable package bytes differ")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (len(names) != len(ARCHIVE_WHITELIST) or len(set(names)) != len(names)
                    or set(names) != ARCHIVE_WHITELIST):
                raise ValueError("Archive member whitelist differs")
            total = 0
            for info in infos:
                part = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (part.is_absolute() or ".." in part.parts or part.as_posix() != info.filename
                        or info.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or info.file_size < 0):
                    raise ValueError("Unsafe archive member")
                limit = MAX_MANIFEST_BYTES if info.filename == MANIFEST_NAME else next(
                    MAX_ARTIFACT_BYTES[name] for name, path in ARTIFACT_PATHS.items()
                    if path == info.filename)
                if info.file_size > limit:
                    raise ValueError("Archive member exceeds its safe limit")
                total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Archive expands beyond the safe limit")
            manifest_raw = archive.read(MANIFEST_NAME)
            if sha256(manifest_raw).hexdigest() != _sha(expected_manifest_sha256, "manifest"):
                raise ValueError("Portable manifest bytes differ")
            manifest = _validate_manifest(_parse_json(manifest_raw, "portable manifest"))
            artifacts = {}
            for name, record in manifest["artifacts"].items():
                info = archive.getinfo(record["path"])
                if info.file_size != record["size"]:
                    raise ValueError("Portable artifact size changed: " + name)
                with archive.open(info, "r") as stream:
                    data = stream.read(record["size"] + 1)
                if len(data) != record["size"] or sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("Portable artifact bytes changed: " + name)
                artifacts[name] = data
    except zipfile.BadZipFile as error:
        raise ValueError("Invalid portable ZIP package") from error
    return manifest, artifacts


def inspect_online_release(*, expected_package_sha256, expected_manifest_sha256,
                           package_path=None, base64_path=None):
    """Validate archive framing, paths, sizes, hashes and current source bytes."""
    raw = _package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, _ = _read_archive(raw, expected_package_sha256=expected_package_sha256,
                                expected_manifest_sha256=expected_manifest_sha256)
    return deepcopy(manifest)


def _write_material(root, artifacts):
    artifact_root = root / "artifacts"
    artifact_root.mkdir(mode=0o700)
    paths = {}
    for name, relative in ARTIFACT_PATHS.items():
        target = root / relative
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(artifacts[name])
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        paths[name] = target
    return paths


class PortableQuestionBank:
    """Immutable eight-item projection; no journal or physics replay on load."""

    _VERSION = "warehouse-alignment-frozen-metadata-bank-view.v1"
    _PORTABLE_VERSION = "warehouse-alignment-portable-question-bank.v1"

    def __init__(self, payload, *, runtime, identities, raw):
        payload, sources = _validate_question_projection(payload, identities)
        if (payload["actor_sha256"] != runtime.actor_sha256
                or payload["runtime_signature"] != identities["parent_runtime_signature"]
                or payload["protocol_sha256"] != runtime.protocol_sha256):
            raise ValueError("Portable bank runtime binding differs")
        self._payload = deepcopy(payload)
        self._items = deepcopy(payload["items"])
        self._checks = deepcopy(payload["checks"])
        self._sources = deepcopy(sources)
        self.test_fixture = False
        self.actor_sha256 = payload["actor_sha256"]
        self.source_runtime_signature = payload["runtime_signature"]
        self.runtime_signature = runtime.signature
        self.protocol_sha256 = payload["protocol_sha256"]
        self.pool_manifest_sha256 = payload["pool_manifest_sha256"]
        self.bank_sha256 = payload["bank_sha256"]
        self.replay_receipt_sha256 = payload["replay_receipt_sha256"]
        self.source_bank_signature = payload["source_bank_signature"]
        self._raw_sha256 = sha256(raw).hexdigest()
        self.signature = digest({
            "version": self._PORTABLE_VERSION,
            "source_bank_signature": self.source_bank_signature,
            "runtime_signature": self.runtime_signature,
            "projection_sha256": self._raw_sha256,
        })
        self._memory_sha256 = digest(self._memory_state())
        self.verify_binding()
        if (digest(self.public_items()) != payload["public_items_sha256"]
                or digest(self.public_items()) != identities["question_bank_public_items_sha256"]
                or digest(self.summary()) != payload["summary_sha256"]
                or self.source_bank_signature
                    != identities["parent_question_bank_signature"]):
            raise ValueError("Portable bank projection changed")

    def _memory_state(self):
        return {
            "payload": self._payload,
            "items": self._items,
            "checks": self._checks,
            "sources": self._sources,
            "identity": [
                self.test_fixture, self.actor_sha256, self.runtime_signature,
                self.protocol_sha256, self.pool_manifest_sha256,
                self.bank_sha256, self.replay_receipt_sha256, self.signature,
                self.source_bank_signature, self.source_runtime_signature,
                self._raw_sha256,
            ],
        }

    def verify_binding(self):
        if (digest(self._memory_state()) != self._memory_sha256
                or digest({
                    "version": self._VERSION,
                    "anchors": self._payload["anchors"],
                    "bank_sha256": self.bank_sha256,
                    "sources": self._sources,
                }) != self.source_bank_signature
                or self.signature != digest({
                    "version": self._PORTABLE_VERSION,
                    "source_bank_signature": self.source_bank_signature,
                    "runtime_signature": self.runtime_signature,
                    "projection_sha256": self._raw_sha256,
                })):
            raise ValueError("Portable question bank changed")
        _validate_source_map(self._sources, "question bank")
        return self.signature

    @property
    def content_eligible(self):
        return True

    @property
    def eligible(self):
        return False

    @property
    def participant_enabled(self):
        return False

    @property
    def formal_ready(self):
        return False

    @property
    def release_ready(self):
        return False

    @property
    def checks(self):
        return deepcopy(self._checks)

    @property
    def items(self):
        return deepcopy(self._items)

    @property
    def sources(self):
        return deepcopy(self._sources)

    def public_items(self):
        return [{
            "id": item["id"],
            "type": "choice",
            "prediction_kind": item["kind"],
            "required": True,
            "prompt": deepcopy(item["prompt"]),
            "options": [{"value": option["value"], "label": deepcopy(option["label"])}
                        for option in item["options"]],
            "preview": deepcopy(item["preview"]),
            "source_frame": item["frame"],
            "source_scenario": item["scenario_id"],
        } for item in self._items]

    def summary(self):
        return {
            "status": "candidate_ready",
            "formal_ready": False,
            "release_ready": False,
            "model_capability_evaluated": False,
            "available": True,
            "item_count": len(self.public_items()),
            "test_fixture": False,
            "preview_history_available_to_both_conditions": True,
            "version": self._VERSION,
            "eligible": False,
            "participant_enabled": False,
            "independent_replay_previously_verified": True,
            "physics_replay_on_load": False,
            "runtime_family": self._payload["runtime_family"],
            "bank_sha256": self.bank_sha256,
            "replay_receipt_sha256": self.replay_receipt_sha256,
        }

    def grade(self, answers):
        if type(answers) is not dict:
            raise ValueError("Prediction answers must be an object")
        result = {"bank_signature": self.signature, "candidate_only": True,
                  "formal_ready": False}
        for kind in ("next_action", "wait_three"):
            items = [item for item in self._items if item["kind"] == kind]
            if any(answers.get(item["id"]) not in {
                    option["value"] for option in item["options"]} for item in items):
                raise ValueError("Incomplete or invalid prediction answers")
            correct = sum(answers[item["id"]] == item["answer"] for item in items)
            result[kind] = {"correct": correct, "total": len(items),
                            "accuracy": correct / len(items)}
        return result


def _portable_bank(payload, *, runtime, identities, raw):
    return PortableQuestionBank(payload, runtime=runtime, identities=identities, raw=raw)


@dataclass
class OnlineReleaseContext:
    runtime: object
    scenarios: dict
    explainer: object
    question_bank: object
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


def load_online_release(*, expected_package_sha256, expected_manifest_sha256,
                        package_path=None, base64_path=None):
    """Load the portable release without opening any parent evidence path."""
    raw = _package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, artifacts = _read_archive(
        raw, expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    temporary = tempfile.TemporaryDirectory(prefix="warehouse-alignment-online-")
    root = Path(temporary.name)
    try:
        # These deployment components are deliberately imported only after the
        # package and manifest have passed their external hash checks.  Both are
        # NumPy-only and have clean-import tests that forbid torch/sklearn.
        from backend import warehouse_alignment_online_runtime as runtime_api
        from backend import warehouse_alignment_online_explanation as explanation_api

        paths = _write_material(root, artifacts)
        protocol = _parse_json(artifacts["protocol"], "protocol")
        scenarios = _parse_json(artifacts["play_scenarios"], "play scenarios")
        identities = manifest["identities"]
        if (scenarios.get("version") != "warehouse-alignment-portable-play-scenes.v1"
                or scenarios.get("test_fixture") is not False
                or scenarios.get("source_scenario_manifest_sha256")
                    != manifest["parent"]["scenario_manifest_sha256"]
                or digest(scenarios) != identities["portable_scenarios_sha256"]
                or type(scenarios.get("splits", {}).get("play")) is not list
                or len(scenarios["splits"]["play"]) != identities["play_scene_count"]
                or [scene.get("id") for scene in scenarios["splits"]["play"]]
                    != identities["play_scene_ids"]):
            raise ValueError("Portable play scenarios changed")
        runtime = runtime_api.OnlineAlignmentRuntime(
            paths["actor"], protocol=protocol,
            expected_actor_sha256=manifest["artifacts"]["actor"]["sha256"],
            expected_protocol_sha256=digest(protocol), allow_test_fixture=False,
        )
        if (runtime.actor_sha256 != identities["actor_sha256"]
                or runtime.protocol_sha256 != identities["protocol_sha256"]):
            raise ValueError("Portable runtime identity differs")
        runtime.verify_binding()
        explainer = explanation_api.OnlineAlignmentExplainer(
            paths["program"],
            expected_program_sha256=manifest["artifacts"]["program"]["sha256"],
            runtime=runtime, allow_test_fixture=False,
        )
        if explainer.program_sha256 != identities["program_sha256"]:
            raise ValueError("Portable explainer identity differs")
        explainer._assert_current(runtime)
        question_payload = _parse_json(artifacts["question_bank"], "question bank")
        bank = _portable_bank(question_payload, runtime=runtime, identities=identities,
                              raw=artifacts["question_bank"])
        source_binding = {
            **manifest["sources"]["parent"],
            **manifest["sources"]["portable"],
        }
        if len(source_binding) != (len(manifest["sources"]["parent"])
                                   + len(manifest["sources"]["portable"])):
            raise ValueError("Portable source groups overlap")
        _validate_source_map(source_binding, "complete portable release")
        release = deepcopy(manifest["release"])
        provenance = {
            "version": VERSION,
            "namespace": "local_pilot",
            "manifest_sha256": expected_manifest_sha256,
            "parent_manifest_sha256": manifest["parent"]["manifest_sha256"],
            "package_sha256": expected_package_sha256,
            "actor_sha256": runtime.actor_sha256,
            "runtime_signature": runtime.signature,
            "parent_runtime_signature": identities["parent_runtime_signature"],
            "protocol_sha256": runtime.protocol_sha256,
            "scenario_manifest_sha256": digest(scenarios),
            "parent_scenario_manifest_sha256": manifest["parent"]["scenario_manifest_sha256"],
            "program_sha256": explainer.program_sha256,
            "explainer_signature": explainer.signature,
            "parent_explainer_signature": identities["parent_explainer_signature"],
            "question_bank_signature": bank.signature,
            "parent_question_bank_signature": identities["parent_question_bank_signature"],
            "analysis": deepcopy(manifest["analysis"]),
            "source_binding": deepcopy(source_binding),
            "release": deepcopy(release),
            "online_portable": True,
        }
        evidence = {
            "version": VERSION,
            "parent": deepcopy(manifest["parent"]),
            "identities": deepcopy(identities),
            "artifact_sha256": {
                name: record["sha256"] for name, record in manifest["artifacts"].items()
            },
            "large_parent_evidence_loaded": False,
            "formal_ready": False,
        }
        signature = digest({
            "version": VERSION,
            "manifest_sha256": expected_manifest_sha256,
            "package_sha256": expected_package_sha256,
            "runtime_signature": runtime.signature,
            "program_sha256": explainer.program_sha256,
            "bank_signature": bank.signature,
            "scenario_manifest_sha256": digest(scenarios),
            "sources": source_binding,
        })
        return OnlineReleaseContext(
            runtime, deepcopy(scenarios), explainer, bank, evidence,
            source_binding, expected_manifest_sha256, signature, release,
            provenance, root, expected_package_sha256, temporary,
        )
    except BaseException:
        temporary.cleanup()
        raise


__all__ = [
    "VERSION", "STATUS", "MAX_PACKAGE_BYTES", "MAX_BASE64_BYTES",
    "OnlineReleaseContext", "build_online_release", "inspect_online_release",
    "load_online_release",
]
