"""Independent, hash-bound portable release for warehouse r4.1.

This format is additive to the historical online and r4 archives.  It accepts
only a fully revalidated r4.1 production admission, including the independently
extracted final explanation program, dynamically selected conflict scenes, and
a replay-verified neutral tutorial.  Loading is NumPy-only and reconstructs
:class:`R41OnlineAlignmentRuntime`; no r4 fallback or historical environment is
accepted.
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
from typing import Any, Mapping
import zipfile

from backend.warehouse_r41_online_explanation import R41OnlineAlignmentExplainer
from backend.warehouse_r41_online_runtime import R41OnlineAlignmentRuntime
from ui import warehouse_alignment_online_release as portable
from ui.warehouse_alignment_r41_tutorial import (
    producer_sources,
    validate_neutral_tutorial,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r41-online-release.v1"
STATUS = "r41_online_portable_internal_pilot"
MANIFEST_NAME = "manifest.json"
ARTIFACT_PATHS = {
    "actor": "artifacts/actor.npz",
    "protocol": "artifacts/protocol.json",
    "play_scenarios": "artifacts/play_scenarios.json",
    "program": "artifacts/program.json",
    "question_bank": "artifacts/question_bank.json",
    "tutorial": "artifacts/tutorial.json",
}
ARCHIVE_WHITELIST = frozenset((MANIFEST_NAME, *ARTIFACT_PATHS.values()))
MAX_PACKAGE_BYTES = 750_000
MAX_BASE64_BYTES = 1_000_000
MAX_MANIFEST_BYTES = 512_000
MAX_ARTIFACT_BYTES = {
    **portable.MAX_ARTIFACT_BYTES,
    "tutorial": 1_000_000,
}
MAX_UNCOMPRESSED_BYTES = MAX_MANIFEST_BYTES + sum(MAX_ARTIFACT_BYTES.values())

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = frozenset((
    "version", "status", "test_fixture", "formal_ready", "parent",
    "artifacts", "identities", "sources", "analysis", "release",
))
_PARENT_FIELDS = frozenset((
    "version", "status", "production_admission_sha256",
    "training_ledger_sha256", "dual_evaluation_sha256",
    "corrected_six_partner_audit_sha256",
    "conflict_manifest_file_sha256", "conflict_manifest_content_sha256",
    "conflict_validation_sha256", "dynamic_selection_report_sha256",
    "selected_scenes_file_sha256", "final_rcpd_report_sha256",
    "explanation_audit_sha256", "question_bank_report_sha256",
    "tutorial_sha256",
))
_IDENTITY_FIELDS = frozenset((
    "actor_sha256", "protocol_sha256", "program_sha256",
    "parent_runtime_signature", "parent_explainer_signature",
    "parent_question_bank_signature", "question_bank_private_items_sha256",
    "question_bank_public_items_sha256", "portable_scenarios_sha256",
    "play_scene_count", "play_scene_ids", "tutorial_signature",
    "tutorial_scene_id", "tutorial_scene_fingerprint",
    "tutorial_successor_state_sha256", "tutorial_snapshot_sha256",
    "conflict_contract_sha256", "conflict_graph_sha256",
    "scene_manifest_content_sha256", "uses_final_actor",
))
_SCENARIO_FIELDS = frozenset((
    "version", "test_fixture", "source_conflict_manifest_file_sha256",
    "source_conflict_manifest_content_sha256", "conflict_contract_sha256",
    "conflict_graph_sha256", "splits",
))
_SOURCE_FIELDS = frozenset(("release",))
_VALIDATION_FIELDS = frozenset((
    "version", "passed", "manifest_content_sha256", "manifest_file_sha256",
    "contract_sha256", "conflict_graph_sha256", "candidate_count",
    "operational_counts", "successor_replay_enabled", "split_seed_disjoint",
    "split_fingerprint_disjoint", "split_successor_state_disjoint",
    "old_geometry_reused", "forbidden_seed_reused",
    "spawned_on_agent_endpoint", "spawned_on_agent_endpoint_total",
    "task_conflict_graph_file_sha256",
))
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|"
    r"database[_-]?url|authorization|bearer[_-]?token)(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|"
    r"\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://)",
    re.IGNORECASE,
)


def canonical(value: Any) -> str:
    return portable.canonical(value)


def digest(value: Any) -> str:
    return portable.digest(value)


def file_hash(path: str | Path) -> str:
    return portable.file_hash(path)


def _sha(value: Any, label: str) -> str:
    return portable._sha(value, label)


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    return portable._parse_json(raw, label)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    return _parse_json(path.read_bytes(), label)


def _component_path(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().absolute()
    if value.resolve() != value or value.is_symlink() or not value.is_file():
        raise ValueError(label + " must be a canonical regular file")
    return value


def _reject_secret_material(value: Any, label: str) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if _SENSITIVE_KEY.search(str(key)):
                    raise ValueError("Credential-like field in " + label)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and _SENSITIVE_VALUE.search(item):
            raise ValueError("Credential-like value in " + label)
    visit(value)


def release_sources() -> dict[str, str]:
    """Exact runtime, web, build and dependency closure selected by r4.1."""
    paths = (
        Path(__file__),
        ROOT / "scripts/build_warehouse_r41_online_release.py",
        ROOT / "scripts/build_warehouse_r41_neutral_tutorial.py",
        ROOT / "backend/__init__.py",
        ROOT / "ui/warehouse_alignment_online_release.py",
        ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "ui/warehouse_alignment_r41_tutorial.py",
        ROOT / "ui/__init__.py",
        ROOT / "ui/warehouse_family_feedback_research/app.js",
        ROOT / "ui/warehouse_family_feedback_research/index.html",
        ROOT / "ui/warehouse_family_feedback_research/styles.css",
        ROOT / "ui/warehouse_family_feedback_research/favicon.svg",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_explanation.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "backend/warehouse_alignment_online_explanation.py",
        ROOT / "core/policy_contracts.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
        ROOT / "env/warehouse_native/__init__.py",
        ROOT / "env/warehouse_native/observations.py",
        ROOT / "env/warehouse_native/scenarios.py",
        ROOT / "core/program.py",
        ROOT / "env/__init__.py",
        ROOT / "env/warehouse/__init__.py",
        ROOT / "env/warehouse/contracts.py",
        ROOT / "env/warehouse/coordination_plan.py",
        ROOT / "env/warehouse/coordination_priority.py",
        ROOT / "env/warehouse/credit_assignment.py",
        ROOT / "env/warehouse/decision_protocol.py",
        ROOT / "env/warehouse/domain.py",
        ROOT / "env/warehouse/energy_management.py",
        ROOT / "env/warehouse/environment.py",
        ROOT / "env/warehouse/frozen_missions.py",
        ROOT / "env/warehouse/goal_management.py",
        ROOT / "env/warehouse/layouts.py",
        ROOT / "env/warehouse/navigation.py",
        ROOT / "env/warehouse/rewards.py",
        ROOT / "env/warehouse/route_goals.py",
        ROOT / "env/warehouse/state_support.py",
        ROOT / "env/warehouse/temporal_audit.py",
        ROOT / "env/warehouse/transition_audit.py",
        ROOT / "env/warehouse/transition_outcome.py",
        ROOT / "requirements-render.txt",
    )
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("r4.1 release source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = file_hash(path)
    # The producer has an independently checked minimal physics/projection
    # closure.  Its members must agree with the broader release binding.
    for name, expected in producer_sources().items():
        if name in result and result[name] != expected:
            raise ValueError("r4.1 tutorial source binding differs")
        result[name] = expected
    return dict(sorted(result.items()))


def _artifact_records(artifacts: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_PATHS):
        raise ValueError("Exact r4.1 portable artifact set required")
    records = {}
    for name, relative in ARTIFACT_PATHS.items():
        raw = artifacts[name]
        if type(raw) is not bytes or not raw or len(raw) > MAX_ARTIFACT_BYTES[name]:
            raise ValueError("r4.1 portable artifact size is invalid: " + name)
        records[name] = {
            "path": relative,
            "size": len(raw),
            "sha256": sha256(raw).hexdigest(),
        }
    return records


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS
            or manifest.get("version") != VERSION or manifest.get("status") != STATUS
            or manifest.get("test_fixture") is not False
            or manifest.get("formal_ready") is not False):
        raise ValueError("Exact non-formal r4.1 portable manifest required")
    parent = manifest.get("parent")
    if (not isinstance(parent, Mapping) or set(parent) != _PARENT_FIELDS
            or parent.get("version") != "warehouse-r41-production-admission.v1"
            or parent.get("status") != "admitted_internal_pilot"):
        raise ValueError("r4.1 portable parent binding differs")
    for name in _PARENT_FIELDS - {"version", "status"}:
        _sha(parent.get(name), "r4.1 parent " + name)
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != _IDENTITY_FIELDS:
        raise ValueError("r4.1 portable identity binding differs")
    for name in _IDENTITY_FIELDS - {
            "play_scene_count", "play_scene_ids", "uses_final_actor",
            "tutorial_scene_id"}:
        _sha(identities.get(name), "r4.1 identity " + name)
    if (not isinstance(identities.get("tutorial_scene_id"), str)
            or not identities["tutorial_scene_id"]):
        raise ValueError("r4.1 tutorial scene ID differs")
    if identities.get("uses_final_actor") is not False:
        raise ValueError("r4.1 tutorial cannot use the final Actor")
    ids = identities.get("play_scene_ids")
    if (type(identities.get("play_scene_count")) is not int
            or identities["play_scene_count"] != 7
            or not isinstance(ids, list) or len(ids) != 7
            or len(set(ids)) != 7
            or any(not isinstance(value, str) or not value for value in ids)):
        raise ValueError("r4.1 portable play-scene identity differs")
    if (identities["scene_manifest_content_sha256"]
            != parent["conflict_manifest_content_sha256"]):
        raise ValueError("r4.1 scene manifest identity differs")
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != set(ARTIFACT_PATHS):
        raise ValueError("r4.1 portable artifact manifest differs")
    for name, relative in ARTIFACT_PATHS.items():
        record = records[name]
        if (not isinstance(record, Mapping)
                or set(record) != {"path", "size", "sha256"}
                or record.get("path") != relative
                or type(record.get("size")) is not int
                or not 0 < record["size"] <= MAX_ARTIFACT_BYTES[name]):
            raise ValueError("r4.1 portable artifact record differs: " + name)
        _sha(record.get("sha256"), "r4.1 artifact " + name)
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != _SOURCE_FIELDS:
        raise ValueError("r4.1 portable source groups differ")
    current = portable._validate_source_map(sources["release"], "r4.1 release")
    if current != release_sources():
        raise ValueError("r4.1 release source set differs")
    if canonical(manifest.get("analysis")) != canonical(portable._analysis_protocol()):
        raise ValueError("Frozen A/B analysis contract changed")
    release = manifest.get("release")
    # Reuse the old public projection schema, but bind its internal-pilot flags
    # here rather than allowing a package to assert formal readiness.
    if (not isinstance(release, Mapping)
            or set(release) != portable._RELEASE_FIELDS
            or release.get("status") != "local_pilot_technically_verified"
            or release.get("namespace") != "local_pilot"
            or release.get("formal_ready") is not False
            or release.get("test_fixture") is not False
            or release.get("online_portable") is not True
            or any(release.get(name) is not True for name in (
                "model_ready", "explanation_ready", "study_ready",
                "participant_enabled", "explanation_qualified", "release_ready",
                "web_integration_completed", "qualification_evaluated",
            ))):
        raise ValueError("r4.1 portable release flags differ")
    return deepcopy(dict(manifest))


def _archive_bytes(manifest: Mapping[str, Any], artifacts: Mapping[str, bytes]) -> bytes:
    records = _artifact_records(artifacts)
    if manifest.get("artifacts") != records:
        raise ValueError("r4.1 manifest artifact records differ")
    _validate_manifest(manifest)
    members = {MANIFEST_NAME: (canonical(manifest) + "\n").encode("utf-8")}
    members.update({ARTIFACT_PATHS[name]: raw for name, raw in artifacts.items()})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, strict_timestamps=True) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = (zipfile.ZIP_STORED
                                  if name == ARTIFACT_PATHS["actor"]
                                  else zipfile.ZIP_DEFLATED)
            archive.writestr(info, members[name], compress_type=info.compress_type,
                             compresslevel=9 if info.compress_type
                             == zipfile.ZIP_DEFLATED else None)
    raw = output.getvalue()
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("r4.1 portable package exceeds the safe deployment limit")
    return raw


def _read_archive(raw: bytes, *, expected_package_sha256: str,
                  expected_manifest_sha256: str):
    if type(raw) is not bytes or not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Bound r4.1 package bytes required")
    if sha256(raw).hexdigest() != _sha(expected_package_sha256, "r4.1 package"):
        raise ValueError("r4.1 portable package bytes differ")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (len(names) != len(ARCHIVE_WHITELIST)
                    or len(set(names)) != len(names)
                    or set(names) != ARCHIVE_WHITELIST):
                raise ValueError("r4.1 archive member whitelist differs")
            total = 0
            for info in infos:
                part = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (part.is_absolute() or ".." in part.parts
                        or part.as_posix() != info.filename or info.is_dir()
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                        or info.compress_type not in (
                            zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or info.file_size < 0):
                    raise ValueError("Unsafe r4.1 archive member")
                limit = MAX_MANIFEST_BYTES if info.filename == MANIFEST_NAME else next(
                    MAX_ARTIFACT_BYTES[name] for name, path in ARTIFACT_PATHS.items()
                    if path == info.filename)
                if info.file_size > limit:
                    raise ValueError("r4.1 archive member exceeds its safe limit")
                total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("r4.1 archive expands beyond the safe limit")
            manifest_raw = archive.read(MANIFEST_NAME)
            if sha256(manifest_raw).hexdigest() != _sha(
                    expected_manifest_sha256, "r4.1 manifest"):
                raise ValueError("r4.1 portable manifest bytes differ")
            manifest = _validate_manifest(
                _parse_json(manifest_raw, "r4.1 portable manifest"))
            artifacts = {}
            for name, record in manifest["artifacts"].items():
                info = archive.getinfo(record["path"])
                if info.file_size != record["size"]:
                    raise ValueError("r4.1 artifact size changed: " + name)
                with archive.open(info, "r") as stream:
                    data = stream.read(record["size"] + 1)
                if (len(data) != record["size"]
                        or sha256(data).hexdigest() != record["sha256"]):
                    raise ValueError("r4.1 artifact bytes changed: " + name)
                artifacts[name] = data
    except zipfile.BadZipFile as error:
        raise ValueError("Invalid r4.1 portable ZIP package") from error
    return manifest, artifacts


def inspect_online_release(*, expected_package_sha256: str,
                           expected_manifest_sha256: str,
                           package_path=None, base64_path=None):
    raw = portable._package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, _ = _read_archive(
        raw, expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return deepcopy(manifest)


def _write_material(root: Path, artifacts: Mapping[str, bytes]) -> dict[str, Path]:
    (root / "artifacts").mkdir(mode=0o700)
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


def _tutorial_binding_from_identities(identities: Mapping[str, Any]) -> dict[str, str]:
    return {
        "scene_manifest_version": "warehouse-r41-conflict-scene-manifest.v1",
        "scene_manifest_content_sha256": identities["scene_manifest_content_sha256"],
        "tutorial_scene_fingerprint": identities["tutorial_scene_fingerprint"],
        "tutorial_successor_state_sha256": identities["tutorial_successor_state_sha256"],
        "tutorial_snapshot_sha256": identities["tutorial_snapshot_sha256"],
        "conflict_contract_sha256": identities["conflict_contract_sha256"],
        "conflict_graph_sha256": identities["conflict_graph_sha256"],
        "producer_sources_sha256": digest(producer_sources()),
    }


def _validate_scenarios(payload: Mapping[str, Any], identities: Mapping[str, Any],
                        parent: Mapping[str, Any]):
    if (not isinstance(payload, Mapping) or set(payload) != _SCENARIO_FIELDS
            or payload.get("version") != "warehouse-r41-portable-scenes.v1"
            or payload.get("test_fixture") is not False
            or payload.get("source_conflict_manifest_file_sha256")
                != parent["conflict_manifest_file_sha256"]
            or payload.get("source_conflict_manifest_content_sha256")
                != parent["conflict_manifest_content_sha256"]
            or payload.get("conflict_contract_sha256")
                != identities["conflict_contract_sha256"]
            or payload.get("conflict_graph_sha256")
                != identities["conflict_graph_sha256"]
            or digest(payload) != identities["portable_scenarios_sha256"]):
        raise ValueError("r4.1 portable scenarios changed")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"play", "tutorial"}:
        raise ValueError("r4.1 portable scenario splits differ")
    play, tutorial = splits["play"], splits["tutorial"]
    if (not isinstance(play, list) or len(play) != 7
            or not isinstance(tutorial, list) or len(tutorial) != 1
            or play[0] != tutorial[0]
            or [row.get("id") for row in play] != identities["play_scene_ids"]
            or tutorial[0].get("id") != identities["tutorial_scene_id"]
            or tutorial[0].get("fingerprint")
                != identities["tutorial_scene_fingerprint"]
            or tutorial[0].get("snapshot", {}).get("r41_conflict", {}).get(
                "successor_state_sha256")
                != identities["tutorial_successor_state_sha256"]
            or digest(tutorial[0].get("snapshot"))
                != identities["tutorial_snapshot_sha256"]):
        raise ValueError("r4.1 play/tutorial scene binding differs")
    return play, tutorial[0]


@dataclass
class R41OnlineReleaseContext:
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
    """Load only the independent r4.1 package and replay its tutorial."""
    raw = portable._package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, artifacts = _read_archive(
        raw, expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    temporary = tempfile.TemporaryDirectory(prefix="warehouse-r41-online-")
    root = Path(temporary.name)
    try:
        paths = _write_material(root, artifacts)
        protocol = _parse_json(artifacts["protocol"], "r4.1 protocol")
        identities = manifest["identities"]
        runtime = R41OnlineAlignmentRuntime(
            paths["actor"], protocol=protocol,
            expected_actor_sha256=manifest["artifacts"]["actor"]["sha256"],
            expected_protocol_sha256=digest(protocol), allow_test_fixture=False,
        )
        if (runtime.actor_sha256 != identities["actor_sha256"]
                or runtime.protocol_sha256 != identities["protocol_sha256"]
                or runtime.signature != identities["parent_runtime_signature"]):
            raise ValueError("r4.1 portable runtime identity differs")
        runtime.verify_binding()
        explainer = R41OnlineAlignmentExplainer(
            paths["program"],
            expected_program_sha256=manifest["artifacts"]["program"]["sha256"],
            runtime=runtime, allow_test_fixture=False,
        )
        if (explainer.program_sha256 != identities["program_sha256"]
                or explainer.signature != identities["parent_explainer_signature"]):
            raise ValueError("r4.1 portable explainer identity differs")
        explainer._assert_current(runtime)
        scenarios = _parse_json(artifacts["play_scenarios"], "r4.1 scenarios")
        play, tutorial_scene = _validate_scenarios(
            scenarios, identities, manifest["parent"])
        for scene in play:
            environment = runtime.environment(deepcopy(scene))
            if environment.state.frame != 0:
                raise ValueError("r4.1 portable play scene does not start at frame zero")
        question_payload = _parse_json(artifacts["question_bank"],
                                       "r4.1 question bank")
        bank = portable._portable_bank(
            question_payload, runtime=runtime, identities=identities,
            raw=artifacts["question_bank"],
        )
        tutorial = _parse_json(artifacts["tutorial"], "r4.1 tutorial")
        if (digest(tutorial) != identities["tutorial_signature"]
                or tutorial.get("uses_final_actor") is not False):
            raise ValueError("r4.1 tutorial signature or Actor boundary differs")
        tutorial_report = validate_neutral_tutorial(
            tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
            expected_bindings=_tutorial_binding_from_identities(identities),
        )
        if tutorial_report["tutorial_signature"] != identities["tutorial_signature"]:
            raise ValueError("r4.1 tutorial replay identity differs")
        source_binding = deepcopy(manifest["sources"]["release"])
        if portable._validate_source_map(source_binding, "r4.1 release") \
                != release_sources():
            raise ValueError("r4.1 portable source binding changed")
        release = deepcopy(manifest["release"])
        provenance = {
            "version": VERSION,
            "namespace": "local_pilot",
            "manifest_sha256": expected_manifest_sha256,
            "package_sha256": expected_package_sha256,
            "production_admission_sha256": manifest["parent"][
                "production_admission_sha256"],
            "training_ledger_sha256": manifest["parent"]["training_ledger_sha256"],
            "actor_sha256": runtime.actor_sha256,
            "runtime_signature": runtime.signature,
            "protocol_sha256": runtime.protocol_sha256,
            "scenario_manifest_sha256": digest(scenarios),
            "parent_scenario_manifest_sha256": manifest["parent"][
                "conflict_manifest_content_sha256"],
            "program_sha256": explainer.program_sha256,
            "explainer_signature": explainer.signature,
            "question_bank_signature": bank.signature,
            "parent_question_bank_signature": identities[
                "parent_question_bank_signature"],
            "tutorial_signature": identities["tutorial_signature"],
            "tutorial_scene_fingerprint": identities["tutorial_scene_fingerprint"],
            "tutorial_uses_final_actor": False,
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
                name: record["sha256"]
                for name, record in manifest["artifacts"].items()
            },
            "tutorial_replay": tutorial_report,
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
            "tutorial_signature": identities["tutorial_signature"],
            "sources": source_binding,
        })
        return R41OnlineReleaseContext(
            runtime, deepcopy(scenarios), explainer, bank, deepcopy(tutorial),
            identities["tutorial_signature"], evidence, source_binding,
            expected_manifest_sha256, signature, release, provenance, root,
            expected_package_sha256, temporary,
        )
    except BaseException:
        temporary.cleanup()
        raise


def _scenarios_from_admission(selection: Mapping[str, Any],
                              bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the admitted tutorial and dynamically selected X/Y tasks."""
    if (not isinstance(selection, Mapping)
            or selection.get("version")
                != "warehouse-r41-conflict-dynamic-play-selection.v1"
            or selection.get("status") != "accepted_final_dynamic_selection"
            or selection.get("release_eligible") is not True
            or selection.get("actor_sha256") != bindings["actor_sha256"]
            or selection.get("source_manifest_file_sha256")
                != bindings["conflict_manifest_file_sha256"]
            or selection.get("source_manifest_content_sha256")
                != bindings["conflict_manifest_content_sha256"]
            or selection.get("contract_sha256")
                != bindings["conflict_contract_sha256"]
            or selection.get("conflict_graph_sha256")
                != bindings["conflict_graph_sha256"]):
        raise ValueError("Admitted r4.1 dynamic scene selection differs")
    tutorial = selection.get("tutorial")
    x_scenes, y_scenes = selection.get("X"), selection.get("Y")
    if (not isinstance(tutorial, Mapping)
            or not isinstance(x_scenes, list) or len(x_scenes) != 3
            or not isinstance(y_scenes, list) or len(y_scenes) != 3):
        raise ValueError("Admitted r4.1 tutorial/X/Y scene sets are incomplete")
    play = [deepcopy(dict(tutorial)), *deepcopy(x_scenes), *deepcopy(y_scenes)]
    ids = [row.get("id") for row in play]
    fingerprints = [row.get("fingerprint") for row in play]
    if (any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != 7
            or any(not isinstance(value, str) or _HEX.fullmatch(value) is None
                   for value in fingerprints)
            or len(set(fingerprints)) != 7):
        raise ValueError("Admitted r4.1 scenes are not seven distinct states")
    return {
        "version": "warehouse-r41-portable-scenes.v1",
        "test_fixture": False,
        "source_conflict_manifest_file_sha256": bindings[
            "conflict_manifest_file_sha256"],
        "source_conflict_manifest_content_sha256": bindings[
            "conflict_manifest_content_sha256"],
        "conflict_contract_sha256": bindings["conflict_contract_sha256"],
        "conflict_graph_sha256": bindings["conflict_graph_sha256"],
        "splits": {"play": play, "tutorial": [deepcopy(dict(tutorial))]},
    }


def assemble_from_admitted_components(*,
          production_admission_path: str | Path,
          expected_production_admission_sha256: str,
          components: Mapping[str, str | Path],
          output_package: str | Path,
          output_base64: str | Path | None = None) -> dict[str, Any]:
    """Build only from a fully revalidated r4.1 production admission.

    ``components`` must contain the exact artifact set registered by
    ``warehouse_r41_production_admission``.  In particular,
    ``final_rcpd_program`` is the independently extracted post-freeze program;
    no training-boundary program is accepted or consulted here.
    """
    # Delayed import keeps production deployment imports NumPy-only and avoids
    # a module cycle through production_admission.package_contract().
    from backend.training import warehouse_r41_production_admission as production

    if not isinstance(components, Mapping) or set(components) != set(
            production.ARTIFACT_NAMES):
        raise ValueError("Exact admitted r4.1 component set required")
    paths = {
        name: _component_path(components[name], "r4.1 " + name)
        for name in production.ARTIFACT_NAMES
    }
    admission_path = _component_path(
        production_admission_path, "r4.1 production admission")
    admission_sha = _sha(
        expected_production_admission_sha256, "r4.1 production admission")
    admission = production.read_saved_admission(
        admission_path, expected_sha256=admission_sha, components=paths)
    if (admission.get("version") != production.VERSION
            or admission.get("status") != production.STATUS
            or admission.get("admitted") is not True
            or admission.get("formal_ready") is not False
            or admission.get("internal_pilot_only") is not True
            or admission.get("human_explanation_effect_validated") is not False
            or admission.get("package_contract") != production.package_contract()):
        raise ValueError("Exact non-formal r4.1 production admission required")
    bindings = admission["bindings"]
    if set(bindings) != production.BINDING_FIELDS:
        raise ValueError("R4.1 production admission binding schema differs")

    protocol_raw = paths["protocol"].read_bytes()
    program_raw = paths["final_rcpd_program"].read_bytes()
    question_raw = paths["question_bank"].read_bytes()
    tutorial_raw = paths["tutorial"].read_bytes()
    protocol = _parse_json(protocol_raw, "r4.1 protocol")
    question = _parse_json(question_raw, "r4.1 question bank")
    tutorial = _parse_json(tutorial_raw, "r4.1 tutorial")
    selection = _read_json(paths["selected_scenes"], "r4.1 selected scenes")
    if (file_hash(paths["selected_scenes"])
            != bindings["selected_scenes_file_sha256"]
            or digest(selection) != bindings["selected_scenes_sha256"]):
        raise ValueError("R4.1 admitted selected-scene bytes differ")
    scenarios = _scenarios_from_admission(selection, bindings)
    tutorial_scene = scenarios["splits"]["tutorial"][0]

    runtime = R41OnlineAlignmentRuntime(
        paths["actor"], protocol=protocol,
        expected_actor_sha256=bindings["actor_sha256"],
        expected_protocol_sha256=bindings["protocol_sha256"],
        allow_test_fixture=False,
    )
    if (runtime.signature != bindings["runtime_signature"]
            or runtime.protocol_sha256 != bindings["protocol_sha256"]):
        raise ValueError("R4.1 admitted runtime identity differs")
    runtime.verify_binding()
    explainer = R41OnlineAlignmentExplainer(
        paths["final_rcpd_program"],
        expected_program_sha256=bindings["program_sha256"],
        runtime=runtime, allow_test_fixture=False,
    )
    if explainer.signature != bindings["explainer_signature"]:
        raise ValueError("R4.1 admitted final explainer identity differs")
    explainer._assert_current(runtime)
    tutorial_report = validate_neutral_tutorial(
        tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
        expected_bindings={
            "scene_manifest_version":
                "warehouse-r41-conflict-scene-manifest.v1",
            "scene_manifest_content_sha256":
                bindings["conflict_manifest_content_sha256"],
            "tutorial_scene_fingerprint":
                bindings["tutorial_scene_fingerprint"],
            "tutorial_successor_state_sha256":
                bindings["tutorial_successor_state_sha256"],
            "tutorial_snapshot_sha256": bindings["tutorial_snapshot_sha256"],
            "conflict_contract_sha256": bindings["conflict_contract_sha256"],
            "conflict_graph_sha256": bindings["conflict_graph_sha256"],
            "producer_sources_sha256": digest(producer_sources()),
        },
    )
    if (digest(tutorial) != bindings["tutorial_signature"]
            or tutorial_report.get("tutorial_signature")
                != bindings["tutorial_signature"]
            or file_hash(paths["tutorial"]) != bindings["tutorial_sha256"]
            or tutorial.get("uses_final_actor") is not False):
        raise ValueError("R4.1 admitted neutral tutorial identity differs")

    identities = {
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "program_sha256": explainer.program_sha256,
        "parent_runtime_signature": runtime.signature,
        "parent_explainer_signature": explainer.signature,
        "parent_question_bank_signature": question.get(
            "source_bank_signature"),
        "question_bank_private_items_sha256": question.get(
            "private_items_sha256"),
        "question_bank_public_items_sha256": question.get(
            "public_items_sha256"),
        "portable_scenarios_sha256": digest(scenarios),
        "play_scene_count": len(scenarios["splits"]["play"]),
        "play_scene_ids": [row["id"] for row in scenarios["splits"]["play"]],
        "tutorial_signature": bindings["tutorial_signature"],
        "tutorial_scene_id": bindings["tutorial_scene_id"],
        "tutorial_scene_fingerprint": bindings[
            "tutorial_scene_fingerprint"],
        "tutorial_successor_state_sha256": bindings[
            "tutorial_successor_state_sha256"],
        "tutorial_snapshot_sha256": bindings["tutorial_snapshot_sha256"],
        "conflict_contract_sha256": bindings["conflict_contract_sha256"],
        "conflict_graph_sha256": bindings["conflict_graph_sha256"],
        "scene_manifest_content_sha256": bindings[
            "conflict_manifest_content_sha256"],
        "uses_final_actor": False,
    }
    if (identities["parent_question_bank_signature"]
            != bindings["question_bank_signature"]):
        raise ValueError("R4.1 admitted question-bank identity differs")
    portable._validate_question_projection(question, identities)
    bank = portable._portable_bank(
        question, runtime=runtime, identities=identities, raw=question_raw)
    bank.verify_binding()
    if bank.source_bank_signature != bindings["question_bank_signature"]:
        raise ValueError("R4.1 admitted portable question bank differs")

    for label, value in (
            ("protocol", protocol), ("question bank", question),
            ("selected scenes", selection), ("tutorial", tutorial),
            ("production admission", admission)):
        _reject_secret_material(value, "r4.1 " + label)
    artifacts = {
        "actor": paths["actor"].read_bytes(),
        "protocol": protocol_raw,
        "play_scenarios": (canonical(scenarios) + "\n").encode("utf-8"),
        "program": program_raw,
        "question_bank": question_raw,
        "tutorial": tutorial_raw,
    }
    message = {
        "zh": "r4.1 内部预实验：主动型神经队友、冲突任务、解释、问卷与中性教学演示均已绑定并通过技术验收；正式结论仍待人类实验。",
        "en": "R4.1 internal pilot: the active neural teammate, conflict tasks, explanations, questionnaire, and neutral tutorial are bound and technically verified; formal conclusions still require a human study.",
    }
    manifest = {
        "version": VERSION,
        "status": STATUS,
        "test_fixture": False,
        "formal_ready": False,
        "parent": {
            "version": admission["version"],
            "status": admission["status"],
            "production_admission_sha256": admission_sha,
            "training_ledger_sha256": bindings["training_ledger_sha256"],
            "dual_evaluation_sha256": bindings["dual_evaluation_sha256"],
            "corrected_six_partner_audit_sha256": bindings[
                "corrected_six_partner_audit_sha256"],
            "conflict_manifest_file_sha256": bindings[
                "conflict_manifest_file_sha256"],
            "conflict_manifest_content_sha256": bindings[
                "conflict_manifest_content_sha256"],
            "conflict_validation_sha256": bindings[
                "conflict_validation_sha256"],
            "dynamic_selection_report_sha256": bindings[
                "dynamic_selection_report_sha256"],
            "selected_scenes_file_sha256": bindings[
                "selected_scenes_file_sha256"],
            "final_rcpd_report_sha256": bindings["final_rcpd_report_sha256"],
            "explanation_audit_sha256": bindings["explanation_audit_sha256"],
            "question_bank_report_sha256": bindings[
                "question_bank_report_sha256"],
            "tutorial_sha256": bindings["tutorial_sha256"],
        },
        "artifacts": _artifact_records(artifacts),
        "identities": identities,
        "sources": {"release": release_sources()},
        "analysis": portable._analysis_protocol(),
        "release": portable._release_projection(message),
    }
    _reject_secret_material(manifest, "r4.1 archive manifest")
    raw = _archive_bytes(manifest, artifacts)
    package = portable._write_new(output_package, raw)
    encoded_path = None
    encoded = base64.b64encode(raw) + b"\n"
    if len(encoded) > MAX_BASE64_BYTES:
        package.unlink(missing_ok=True)
        raise ValueError("r4.1 Base64 Secret File exceeds the deployment limit")
    if output_base64 is not None:
        try:
            encoded_path = portable._write_new(output_base64, encoded)
        except BaseException:
            package.unlink(missing_ok=True)
            raise
    manifest_sha = sha256(
        (canonical(manifest) + "\n").encode("utf-8")).hexdigest()
    package_sha = sha256(raw).hexdigest()
    try:
        loaded = load_online_release(
            package_path=package, expected_package_sha256=package_sha,
            expected_manifest_sha256=manifest_sha,
        )
        try:
            if (type(loaded.runtime) is not R41OnlineAlignmentRuntime
                    or loaded.runtime.actor_sha256 != runtime.actor_sha256
                    or loaded.runtime.signature != runtime.signature
                    or loaded.explainer.program_sha256 != bindings["program_sha256"]
                    or loaded.tutorial_signature
                        != identities["tutorial_signature"]
                    or loaded.tutorial.get("uses_final_actor") is not False):
                raise ValueError("Independent r4.1 package reload differs")
        finally:
            loaded.close()
    except BaseException:
        package.unlink(missing_ok=True)
        if encoded_path is not None:
            encoded_path.unlink(missing_ok=True)
        raise
    return {
        "version": VERSION,
        "status": "built_from_production_admission_and_independently_reloaded",
        "package": str(package),
        "package_size": len(raw),
        "package_sha256": package_sha,
        "base64": str(encoded_path) if encoded_path else None,
        "base64_size": len(encoded),
        "manifest_sha256": manifest_sha,
        "production_admission_sha256": admission_sha,
        "actor_sha256": runtime.actor_sha256,
        "program_sha256": explainer.program_sha256,
        "tutorial_signature": identities["tutorial_signature"],
        "tutorial_uses_final_actor": False,
        "independent_final_program": True,
        "formal_ready": False,
    }


__all__ = [
    "VERSION", "STATUS", "ARTIFACT_PATHS", "ARCHIVE_WHITELIST",
    "MAX_PACKAGE_BYTES", "MAX_BASE64_BYTES", "R41OnlineReleaseContext",
    "assemble_from_admitted_components", "inspect_online_release",
    "load_online_release", "release_sources",
]
