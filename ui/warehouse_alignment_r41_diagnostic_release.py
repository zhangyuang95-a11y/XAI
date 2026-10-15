"""Hash-bound portable release for the r4.1 internal diagnostic.

This release is deliberately separate from r4.1 production admission.  It
packages the user-designated terminal Actor only after scene, explanation,
questionnaire and tutorial gates pass.  Behavioral performance is recorded as
failed and waived; formal-study and formal-sample eligibility remain false.
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

from backend.warehouse_r41_diagnostic_online_runtime import (
    CONFLICT_VALIDATION_VERSION,
    FULL_SCENE_MANIFEST_VERSION,
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_online_explanation import (
    R41DiagnosticOnlineAlignmentExplainer,
)
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
)
from ui import warehouse_alignment_online_release as portable
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-r41-diagnostic-online-release.v3"
STATUS = "r41_diagnostic_online_portable_internal_experiment"
PUBLIC_RELEASE_VERSION = "r4.1-diagnostic"
PILOT_CLASS = "internal_diagnostic"
MANIFEST_NAME = "manifest.json"
RUNTIME_MANIFEST_VERSION = "warehouse-r41-diagnostic-portable-runtime-manifest.v1"
DYNAMIC_SELECTION_VERSION = "warehouse-r41-diagnostic-conflict-dynamic-selection.v3"
ARTIFACT_PATHS = {
    "actor": "artifacts/actor.npz",
    "protocol": "artifacts/training_protocol.json",
    "runtime_manifest": "artifacts/runtime_manifest.json",
    "program": "artifacts/program.json",
    "question_bank": "artifacts/question_bank.json",
    "tutorial": "artifacts/tutorial.json",
}
ARCHIVE_WHITELIST = frozenset((MANIFEST_NAME, *ARTIFACT_PATHS.values()))
MAX_PACKAGE_BYTES = 750_000
MAX_BASE64_BYTES = 1_000_000
MAX_MANIFEST_BYTES = 512_000
MAX_ARTIFACT_BYTES = {
    "actor": 2_000_000,
    "protocol": 2_000_000,
    "runtime_manifest": 8_000_000,
    "program": 4_000_000,
    "question_bank": 16_000_000,
    "tutorial": 2_000_000,
}
MAX_UNCOMPRESSED_BYTES = MAX_MANIFEST_BYTES + sum(MAX_ARTIFACT_BYTES.values())

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = frozenset((
    "version", "status", "test_fixture", "pilot_class", "formal_ready",
    "formal_sample_eligible", "data_persistent", "parent", "artifacts",
    "identities", "sources", "analysis", "release",
))
_PARENT_FIELDS = frozenset((
    "version", "status", "diagnostic_admission_sha256",
    "diagnostic_designation_sha256", "training_ledger_sha256",
    "dual_evaluation_sha256", "failure_closeout_sha256",
    "conflict_manifest_file_sha256", "conflict_manifest_content_sha256",
    "conflict_manifest_semantic_sha256", "conflict_validation_sha256",
    "dynamic_selection_report_sha256", "selected_scenes_file_sha256",
    "selected_scenes_semantic_sha256", "final_rcpd_report_sha256",
    "explanation_audit_sha256", "question_bank_report_sha256",
    "tutorial_sha256",
))
_IDENTITY_FIELDS = frozenset((
    "actor_sha256", "actor_parameters_sha256", "protocol_file_sha256",
    "protocol_content_sha256", "program_sha256", "parent_runtime_signature",
    "runtime_manifest_signature", "parent_explainer_signature",
    "parent_question_bank_signature", "question_bank_private_items_sha256",
    "question_bank_public_items_sha256", "runtime_manifest_file_sha256",
    "runtime_manifest_content_sha256", "runtime_manifest_semantic_sha256",
    "source_full_manifest_file_sha256", "source_full_manifest_content_sha256",
    "source_full_manifest_semantic_sha256", "source_full_manifest_version",
    "source_conflict_validation_version", "source_conflict_validation_sha256",
    "diagnostic_contract_version", "diagnostic_conflict_graph_sha256",
    "play_scene_count",
    "play_scene_ids", "play_scene_fingerprints", "tutorial_signature",
    "tutorial_scene_id", "tutorial_scene_fingerprint",
    "tutorial_successor_state_sha256", "tutorial_snapshot_sha256",
    "diagnostic_contract_sha256", "conflict_families_sha256",
    "uses_terminal_designated_actor", "action_override_count",
))
_RUNTIME_MANIFEST_FIELDS = frozenset((
    "version", "source_full_manifest_version",
    "source_conflict_validation_version", "source_conflict_validation_sha256",
    "diagnostic_contract_version", "diagnostic_contract_sha256",
    "diagnostic_conflict_graph_sha256", "conflict_families_sha256",
    "source_full_manifest_file_sha256", "source_full_manifest_content_sha256",
    "source_full_manifest_semantic_sha256",
    "source_dynamic_selection_report_sha256",
    "source_selected_scenes_file_sha256",
    "source_selected_scenes_semantic_sha256", "splits", "content_sha256",
))
_RELEASE_FIELDS = frozenset((
    "release_version", "status", "namespace", "pilot_class", "formal_ready",
    "formal_sample_eligible", "human_explanation_effect_validated",
    "behavior_performance_gate_passed", "behavior_performance_gate_waived",
    "data_persistent", "test_fixture", "online_portable", "participant_enabled",
    "model_ready", "study_ready", "explanation_ready", "runtime_action_override",
    "message",
))
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|"
    r"database[_-]?url|authorization|bearer[_-]?token)(?:$|[_-])", re.I,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|"
    r"\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://)", re.I,
)
_QUESTION_PUBLIC_FIELDS = frozenset((
    "id", "type", "prediction_kind", "required", "prompt", "options",
    "preview", "source_frame", "source_scenario",
))


canonical = portable.canonical
digest = portable.digest
file_hash = portable.file_hash


def _participant_question_projection(items):
    """Convert authenticated private marker names to the canvas schema.

    The portable bank authenticates its original ``{marker, position}``
    projection before this adapter is constructed.  This presentation-only
    adapter accepts exactly that frozen shape and emits the existing server
    contract ``{label, position}``; ambiguous or malformed markers fail
    closed.
    """

    if not isinstance(items, list):
        raise ValueError("Diagnostic question projection must be a list")
    result = deepcopy(items)
    for item in result:
        if (not isinstance(item, dict) or set(item) != _QUESTION_PUBLIC_FIELDS
                or item.get("type") != "choice"
                or item.get("required") is not True
                or item.get("prediction_kind") not in ("next_action", "wait_three")
                or not isinstance(item.get("preview"), dict)):
            raise ValueError("Diagnostic question public schema differs")
        if any(name in item for name in
               ("answer", "snapshot", "snapshot_sha256", "evidence",
                "diversity_key")):
            raise ValueError("Diagnostic question public projection leaked evidence")
        preview = item["preview"]
        markers = preview.get("question_markers")
        if item["prediction_kind"] == "next_action":
            if markers is not None:
                raise ValueError("Diagnostic next-action question carries markers")
            continue
        if not isinstance(markers, list) or len(markers) != 4:
            raise ValueError("Diagnostic wait-three markers differ")
        rows = preview.get("map", {}).get("rows") if isinstance(
            preview.get("map"), Mapping) else None
        cols = preview.get("map", {}).get("cols") if isinstance(
            preview.get("map"), Mapping) else None
        if type(rows) is not int or rows <= 0 or type(cols) is not int or cols <= 0:
            raise ValueError("Diagnostic marker map bounds differ")
        labels: set[str] = set()
        positions: set[tuple[int, int]] = set()
        normalized = []
        for marker in markers:
            if (not isinstance(marker, dict)
                    or set(marker) != {"marker", "position"}
                    or type(marker.get("marker")) is not str
                    or marker["marker"] not in "ABCD"
                    or not isinstance(marker.get("position"), list)
                    or len(marker["position"]) != 2
                    or any(type(value) is not int for value in marker["position"])):
                raise ValueError("Diagnostic wait-three marker schema differs")
            label = marker["marker"]
            position = tuple(marker["position"])
            if (label in labels or position in positions
                    or not 0 <= position[0] < rows
                    or not 0 <= position[1] < cols):
                raise ValueError("Diagnostic wait-three marker value differs")
            labels.add(label); positions.add(position)
            normalized.append({"label": label, "position": list(position)})
        if labels != set("ABCD"):
            raise ValueError("Diagnostic wait-three marker labels differ")
        option_values = item.get("options")
        if (not isinstance(option_values, list) or len(option_values) != 4
                or {option.get("value") for option in option_values
                    if isinstance(option, Mapping)}
                    != {f"{row},{col}" for row, col in positions}):
            raise ValueError("Diagnostic wait-three marker options differ")
        preview["question_markers"] = normalized
    return result


class R41DiagnosticParticipantQuestionBank:
    """Integrity-preserving presentation proxy for the diagnostic server."""

    def __init__(self, bank):
        if type(bank) is not portable.PortableQuestionBank:
            raise ValueError("Exact authenticated portable question bank required")
        bank.verify_binding()
        self._bank = bank
        self._source_public_sha256 = digest(bank.public_items())
        self._participant_public_sha256 = digest(
            _participant_question_projection(bank.public_items()))
        self.verify_binding()

    def verify_binding(self):
        signature = self._bank.verify_binding()
        source = self._bank.public_items()
        if (digest(source) != self._source_public_sha256
                or digest(_participant_question_projection(source))
                    != self._participant_public_sha256):
            raise ValueError("Diagnostic participant question projection changed")
        return signature

    @property
    def signature(self):
        return self._bank.signature

    @property
    def test_fixture(self):
        return self._bank.test_fixture

    @property
    def actor_sha256(self):
        return self._bank.actor_sha256

    @property
    def runtime_signature(self):
        return self._bank.runtime_signature

    @property
    def protocol_sha256(self):
        return self._bank.protocol_sha256

    @property
    def source_bank_signature(self):
        return self._bank.source_bank_signature

    @property
    def content_eligible(self):
        return self._bank.content_eligible

    @property
    def eligible(self):
        return self._bank.eligible

    @property
    def participant_enabled(self):
        return self._bank.participant_enabled

    @property
    def formal_ready(self):
        return self._bank.formal_ready

    @property
    def release_ready(self):
        return self._bank.release_ready

    @property
    def items(self):
        return self._bank.items

    @property
    def checks(self):
        return self._bank.checks

    @property
    def sources(self):
        return self._bank.sources

    def public_items(self):
        self.verify_binding()
        return _participant_question_projection(self._bank.public_items())

    def summary(self):
        self.verify_binding()
        return self._bank.summary()

    def grade(self, answers):
        self.verify_binding()
        return self._bank.grade(answers)


def _diagnostic_portable_bank(payload, *, runtime, identities, raw):
    bank = portable._portable_bank(
        payload, runtime=runtime, identities=identities, raw=raw,
    )
    return R41DiagnosticParticipantQuestionBank(bank)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    return portable._parse_json(raw, label)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    value = Path(path).expanduser().absolute()
    if value.is_symlink() or not value.is_file() or value.resolve() != value:
        raise ValueError(label + " must be a canonical regular file")
    return _parse_json(value.read_bytes(), label)


def _component_path(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().absolute()
    if value.is_symlink() or not value.is_file() or value.resolve() != value:
        raise ValueError(label + " must be a canonical regular file")
    return value


def _reject_sensitive(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                raise ValueError("Credential-like field in " + label)
            _reject_sensitive(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive(child, label)
    elif isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise ValueError("Credential-like value in " + label)


def release_sources() -> dict[str, str]:
    """Exact dependency-light serving and UI source set."""
    paths = (
        Path(__file__),
        ROOT / "scripts/build_warehouse_r41_diagnostic_online_release.py",
        ROOT / "scripts/preflight_warehouse_r41_diagnostic_render.py",
        ROOT / "backend/__init__.py",
        ROOT / "ui/__init__.py",
        ROOT / "ui/warehouse_alignment_online_release.py",
        ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "ui/warehouse_alignment_r41_diagnostic_tutorial.py",
        ROOT / "ui/warehouse_alignment_r41_tutorial.py",
        ROOT / "ui/warehouse_family_feedback_research/app.js",
        ROOT / "ui/warehouse_family_feedback_research/index.html",
        ROOT / "ui/warehouse_family_feedback_research/styles.css",
        ROOT / "ui/warehouse_family_feedback_research/favicon.svg",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_explanation.py",
        ROOT / "backend/warehouse_r41_diagnostic_model_tree.py",
        ROOT / "backend/warehouse_alignment_online_explanation.py",
        ROOT / "core/policy_contracts.py",
        ROOT / "core/program.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
        ROOT / "env/warehouse_native/__init__.py",
        ROOT / "env/warehouse_native/observations.py",
        ROOT / "env/warehouse_native/scenarios.py",
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
            raise ValueError("Diagnostic release source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = file_hash(path)
    for name, expected in tutorial_api.producer_sources().items():
        if name in result and result[name] != expected:
            raise ValueError("Diagnostic tutorial source binding differs")
        result[name] = expected
    return dict(sorted(result.items()))


def _release_projection() -> dict[str, Any]:
    return {
        "release_version": PUBLIC_RELEASE_VERSION,
        "status": "internal_diagnostic_technically_verified",
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
        "participant_enabled": True,
        "model_ready": True,
        "study_ready": True,
        "explanation_ready": True,
        "runtime_action_override": False,
        "message": {
            "zh": "内部诊断实验：数据仅用于探索，服务重启后可能丢失，不纳入正式研究样本。",
            "en": "Internal diagnostic study: data are exploratory, may be lost after a service restart, and are excluded from the formal sample.",
        },
    }


def _artifact_records(artifacts: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_PATHS):
        raise ValueError("Exact diagnostic portable artifact set required")
    records = {}
    for name, relative in ARTIFACT_PATHS.items():
        raw = artifacts[name]
        if type(raw) is not bytes or not raw or len(raw) > MAX_ARTIFACT_BYTES[name]:
            raise ValueError("Diagnostic portable artifact size is invalid: " + name)
        records[name] = {
            "path": relative, "size": len(raw), "sha256": sha256(raw).hexdigest(),
        }
    return records


def _runtime_manifest(selection: Mapping[str, Any], full_manifest: Mapping[str, Any],
                      bindings: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(selection, Mapping)
            or selection.get("version")
                != DYNAMIC_SELECTION_VERSION
            or selection.get("status")
                != "accepted_diagnostic_dynamic_selection"
            or selection.get("release_eligible") is not True
            or selection.get("actor_sha256") != bindings["actor_sha256"]
            or selection.get("source_manifest_file_sha256")
                != bindings["conflict_manifest_file_sha256"]
            or selection.get("source_manifest_content_sha256")
                != bindings["conflict_manifest_content_sha256"]
            or selection.get("diagnostic_contract_sha256")
                != bindings["diagnostic_contract_sha256"]
            or selection.get("diagnostic_contract_version")
                != DIAGNOSTIC_CONTRACT_VERSION
            or selection.get("diagnostic_conflict_graph_sha256")
                != bindings["conflict_graph_sha256"]
            or selection.get("diagnostic_conflict_graph_sha256")
                != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
            or selection.get("conflict_families_sha256")
                != bindings["conflict_families_sha256"]
            or selection.get("zero_action_overrides") is not True
            or selection.get("no_replacement_pickup_on_agent") is not True
            or selection.get("no_replacement_delivery_on_agent") is not True
            or selection.get("no_replacement_endpoint_on_agent") is not True
            or selection.get("ordinary_sampler_fallback") is not False
            or digest(selection) != bindings["selected_scenes_semantic_sha256"]):
        raise ValueError("Diagnostic selected scenes differ from admission")
    if (not isinstance(full_manifest, Mapping)
            or full_manifest.get("version") != FULL_SCENE_MANIFEST_VERSION
            or full_manifest.get("diagnostic_contract_version")
                != DIAGNOSTIC_CONTRACT_VERSION
            or full_manifest.get("diagnostic_contract_sha256")
                != bindings["diagnostic_contract_sha256"]
            or full_manifest.get("diagnostic_contract_sha256")
                != DIAGNOSTIC_CONTRACT_SHA256
            or full_manifest.get("diagnostic_conflict_graph_sha256")
                != bindings["conflict_graph_sha256"]
            or full_manifest.get("diagnostic_conflict_graph_sha256")
                != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
            or full_manifest.get("conflict_families_sha256")
                != bindings["conflict_families_sha256"]
            or full_manifest.get("conflict_families_sha256")
                != CONFLICT_FAMILIES_SHA256):
        raise ValueError("Diagnostic full manifest source contract differs")
    tutorial = selection.get("tutorial")
    x_rows, y_rows = selection.get("X"), selection.get("Y")
    if (not isinstance(tutorial, Mapping) or not isinstance(x_rows, list)
            or not isinstance(y_rows, list) or len(x_rows) != 3 or len(y_rows) != 3):
        raise ValueError("Diagnostic tutorial/X/Y scene sets are incomplete")
    formal = [*x_rows, *y_rows]
    if (selection.get("six_distinct_conflict_families") is not True
            or len({row.get("family_id") for row in formal}) != 6):
        raise ValueError("Diagnostic release needs six distinct conflict families")
    play = [deepcopy(dict(tutorial)), *deepcopy(x_rows), *deepcopy(y_rows)]
    if (len({row.get("id") for row in play}) != 7
            or len({row.get("fingerprint") for row in play}) != 7):
        raise ValueError("Diagnostic release needs seven distinct scenes")
    value = {
        "version": RUNTIME_MANIFEST_VERSION,
        "source_full_manifest_version": FULL_SCENE_MANIFEST_VERSION,
        "source_conflict_validation_version": CONFLICT_VALIDATION_VERSION,
        "source_conflict_validation_sha256": bindings[
            "conflict_validation_sha256"],
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": bindings["diagnostic_contract_sha256"],
        "diagnostic_conflict_graph_sha256": bindings["conflict_graph_sha256"],
        "conflict_families_sha256": bindings["conflict_families_sha256"],
        "source_full_manifest_file_sha256": bindings[
            "conflict_manifest_file_sha256"],
        "source_full_manifest_content_sha256": bindings[
            "conflict_manifest_content_sha256"],
        "source_full_manifest_semantic_sha256": bindings[
            "conflict_manifest_semantic_sha256"],
        "source_dynamic_selection_report_sha256": bindings[
            "dynamic_selection_report_sha256"],
        "source_selected_scenes_file_sha256": bindings[
            "selected_scenes_file_sha256"],
        "source_selected_scenes_semantic_sha256": bindings[
            "selected_scenes_semantic_sha256"],
        "splits": {"play": play, "tutorial": [deepcopy(dict(tutorial))]},
    }
    value["content_sha256"] = digest(value)
    if (full_manifest.get("content_sha256")
            != value["source_full_manifest_content_sha256"]
            or digest(full_manifest) != value["source_full_manifest_semantic_sha256"]):
        raise ValueError("Diagnostic portable manifest source binding differs")
    return value


def _validate_runtime_manifest(value: Mapping[str, Any],
                               parent: Mapping[str, Any]) -> tuple[list, dict]:
    if (not isinstance(value, Mapping) or set(value) != _RUNTIME_MANIFEST_FIELDS
            or value.get("version") != RUNTIME_MANIFEST_VERSION):
        raise ValueError("Exact diagnostic portable runtime manifest required")
    content = deepcopy(dict(value)); claimed = content.pop("content_sha256", None)
    if claimed != digest(content):
        raise ValueError("Diagnostic portable runtime manifest content differs")
    expected = {
        "source_full_manifest_version": FULL_SCENE_MANIFEST_VERSION,
        "source_conflict_validation_version": CONFLICT_VALIDATION_VERSION,
        "source_conflict_validation_sha256": parent[
            "conflict_validation_sha256"],
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "source_full_manifest_file_sha256": parent["conflict_manifest_file_sha256"],
        "source_full_manifest_content_sha256": parent["conflict_manifest_content_sha256"],
        "source_full_manifest_semantic_sha256": parent["conflict_manifest_semantic_sha256"],
        "source_dynamic_selection_report_sha256": parent["dynamic_selection_report_sha256"],
        "source_selected_scenes_file_sha256": parent["selected_scenes_file_sha256"],
        "source_selected_scenes_semantic_sha256": parent["selected_scenes_semantic_sha256"],
    }
    if (value.get("diagnostic_contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
            or value.get("diagnostic_conflict_graph_sha256")
                != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
            or value.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256
            or any(value.get(key) != expected_value for key, expected_value in expected.items())):
        raise ValueError("Diagnostic runtime manifest source identity differs")
    splits = value.get("splits")
    play = splits.get("play") if isinstance(splits, Mapping) else None
    tutorial = splits.get("tutorial") if isinstance(splits, Mapping) else None
    if (set(splits or {}) != {"play", "tutorial"} or not isinstance(play, list)
            or len(play) != 7 or not isinstance(tutorial, list) or len(tutorial) != 1
            or play[0] != tutorial[0]
            or len({row.get("id") for row in play}) != 7
            or len({row.get("fingerprint") for row in play}) != 7
            or len({row.get("family_id") for row in play[1:]}) != 6):
        raise ValueError("Diagnostic runtime scene projection differs")
    return play, tutorial[0]


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS
            or manifest.get("version") != VERSION or manifest.get("status") != STATUS
            or manifest.get("test_fixture") is not False
            or manifest.get("pilot_class") != PILOT_CLASS
            or manifest.get("formal_ready") is not False
            or manifest.get("formal_sample_eligible") is not False
            or manifest.get("data_persistent") is not False):
        raise ValueError("Exact non-formal diagnostic portable manifest required")
    parent = manifest.get("parent")
    if (not isinstance(parent, Mapping) or set(parent) != _PARENT_FIELDS
            or parent.get("version") != "warehouse-r41-diagnostic-admission.v1"
            or parent.get("status") != "admitted_internal_diagnostic"):
        raise ValueError("Diagnostic portable parent binding differs")
    for name in _PARENT_FIELDS - {"version", "status"}:
        _sha(parent.get(name), "parent " + name)
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != _IDENTITY_FIELDS:
        raise ValueError("Diagnostic portable identity schema differs")
    scalar_exceptions = {
        "play_scene_count", "play_scene_ids", "play_scene_fingerprints",
        "uses_terminal_designated_actor", "action_override_count",
        "source_full_manifest_version", "source_conflict_validation_version",
        "diagnostic_contract_version",
    }
    for name in _IDENTITY_FIELDS - scalar_exceptions:
        _sha(identities.get(name), "identity " + name)
    if (identities.get("source_full_manifest_version")
                != FULL_SCENE_MANIFEST_VERSION
            or identities.get("source_conflict_validation_version")
                != CONFLICT_VALIDATION_VERSION
            or identities.get("diagnostic_contract_version")
                != DIAGNOSTIC_CONTRACT_VERSION
            or identities.get("diagnostic_conflict_graph_sha256")
                != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
            or identities.get("diagnostic_contract_sha256")
                != DIAGNOSTIC_CONTRACT_SHA256
            or identities.get("conflict_families_sha256")
                != CONFLICT_FAMILIES_SHA256
            or identities.get("play_scene_count") != 7
            or not isinstance(identities.get("play_scene_ids"), list)
            or len(identities["play_scene_ids"]) != 7
            or len(set(identities["play_scene_ids"])) != 7
            or not isinstance(identities.get("play_scene_fingerprints"), list)
            or len(identities["play_scene_fingerprints"]) != 7
            or len(set(identities["play_scene_fingerprints"])) != 7
            or any(_HEX.fullmatch(value) is None
                   for value in identities["play_scene_fingerprints"])
            or identities.get("uses_terminal_designated_actor") is not True
            or identities.get("action_override_count") != 0):
        raise ValueError("Diagnostic portable play/authority identity differs")
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != set(ARTIFACT_PATHS):
        raise ValueError("Diagnostic portable artifact manifest differs")
    for name, relative in ARTIFACT_PATHS.items():
        record = records[name]
        if (not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}
                or record.get("path") != relative or type(record.get("size")) is not int
                or not 0 < record["size"] <= MAX_ARTIFACT_BYTES[name]):
            raise ValueError("Diagnostic artifact record differs: " + name)
        _sha(record.get("sha256"), "artifact " + name)
    sources = manifest.get("sources")
    if (not isinstance(sources, Mapping) or set(sources) != {"release"}
            or portable._validate_source_map(sources["release"], "diagnostic release")
                != release_sources()):
        raise ValueError("Diagnostic release source binding differs")
    if canonical(manifest.get("analysis")) != canonical(portable._analysis_protocol()):
        raise ValueError("Diagnostic A/B analysis contract differs")
    if manifest.get("release") != _release_projection():
        raise ValueError("Diagnostic public release declaration differs")
    return deepcopy(dict(manifest))


def _archive_bytes(manifest: Mapping[str, Any],
                   artifacts: Mapping[str, bytes]) -> bytes:
    _validate_manifest(manifest)
    records = _artifact_records(artifacts)
    if records != manifest["artifacts"]:
        raise ValueError("Diagnostic manifest artifact bytes differ")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, raw in ((MANIFEST_NAME, (canonical(manifest) + "\n").encode()),
                          *((ARTIFACT_PATHS[key], artifacts[key])
                            for key in ARTIFACT_PATHS)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, raw)
    raw = stream.getvalue()
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Diagnostic ZIP exceeds the Secret File package limit")
    return raw


def _read_archive(raw: bytes, *, expected_package_sha256: str,
                  expected_manifest_sha256: str) -> tuple[dict, dict[str, bytes]]:
    if (type(raw) is not bytes or not raw or len(raw) > MAX_PACKAGE_BYTES
            or sha256(raw).hexdigest() != _sha(
                expected_package_sha256, "diagnostic package")):
        raise ValueError("Diagnostic package bytes differ")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (len(names) != len(set(names)) or set(names) != ARCHIVE_WHITELIST
                    or sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES):
                raise ValueError("Diagnostic archive member set or size differs")
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (pure.is_absolute() or ".." in pure.parts
                        or pure.as_posix() != info.filename or info.is_dir()
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or info.flag_bits & 0x1 or info.file_size < 0):
                    raise ValueError("Diagnostic archive member is unsafe")
                limit = (MAX_MANIFEST_BYTES if info.filename == MANIFEST_NAME
                         else next(MAX_ARTIFACT_BYTES[name]
                                   for name, path in ARTIFACT_PATHS.items()
                                   if path == info.filename))
                if info.file_size > limit:
                    raise ValueError("Diagnostic archive member exceeds its safe limit")
            manifest_info = archive.getinfo(MANIFEST_NAME)
            with archive.open(manifest_info, "r") as stream:
                manifest_raw = stream.read(MAX_MANIFEST_BYTES + 1)
            if (not manifest_raw or len(manifest_raw) > MAX_MANIFEST_BYTES
                    or sha256(manifest_raw).hexdigest() != _sha(
                        expected_manifest_sha256, "diagnostic manifest")):
                raise ValueError("Diagnostic archive manifest bytes differ")
            manifest = _parse_json(manifest_raw, "diagnostic archive manifest")
            _validate_manifest(manifest)
            artifacts = {}
            for name, relative in ARTIFACT_PATHS.items():
                info = archive.getinfo(relative)
                record = manifest["artifacts"][name]
                if info.file_size != record["size"]:
                    raise ValueError("Diagnostic archived artifact size differs: " + name)
                with archive.open(info, "r") as stream:
                    raw_item = stream.read(record["size"] + 1)
                if (len(raw_item) != record["size"]
                        or sha256(raw_item).hexdigest() != record["sha256"]):
                    raise ValueError("Diagnostic archived artifact differs: " + name)
                artifacts[name] = raw_item
    except zipfile.BadZipFile as error:
        raise ValueError("Diagnostic package is not a valid ZIP") from error
    return manifest, artifacts


def _package_bytes(*, package_path=None, base64_path=None) -> bytes:
    """Read one bounded local ZIP or Render Secret File."""
    if (package_path is None) == (base64_path is None):
        raise ValueError("Choose exactly one diagnostic ZIP or Base64 Secret File")
    if package_path is not None:
        return portable._read_bounded(
            package_path, MAX_PACKAGE_BYTES, "diagnostic package")
    encoded = b"".join(portable._read_bounded(
        base64_path, MAX_BASE64_BYTES, "diagnostic Base64 secret").split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Invalid diagnostic Base64 Secret File") from error
    if not raw or len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("Decoded diagnostic package exceeds the safe limit")
    return raw


def inspect_online_release(*, expected_package_sha256: str,
                           expected_manifest_sha256: str,
                           package_path=None, base64_path=None) -> dict[str, Any]:
    raw = _package_bytes(package_path=package_path, base64_path=base64_path)
    manifest, _ = _read_archive(
        raw, expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return manifest


def _write_material(root: Path, artifacts: Mapping[str, bytes]) -> dict[str, Path]:
    paths = {}
    for name, relative in ARTIFACT_PATHS.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(artifacts[name]); stream.flush(); os.fsync(stream.fileno())
        paths[name] = target
    return paths


def _runtime(paths: Mapping[str, Path],
             identities: Mapping[str, Any]) -> R41DiagnosticOnlineAlignmentRuntime:
    runtime_manifest = _parse_json(
        paths["runtime_manifest"].read_bytes(), "diagnostic runtime manifest")
    content = deepcopy(runtime_manifest); claimed = content.pop("content_sha256", None)
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        paths["actor"], training_protocol_path=paths["protocol"],
        manifest_path=paths["runtime_manifest"],
        expected_actor_sha256=identities["actor_sha256"],
        expected_training_protocol_file_sha256=identities["protocol_file_sha256"],
        expected_training_protocol_content_sha256=identities["protocol_content_sha256"],
        expected_manifest_file_sha256=identities["runtime_manifest_file_sha256"],
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(runtime_manifest),
    )
    if (runtime.signature != identities["parent_runtime_signature"]
            or runtime.runtime_manifest_signature
                != identities["runtime_manifest_signature"]):
        raise ValueError("Diagnostic serving runtime identity differs")
    runtime.verify_binding()
    return runtime


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
        expected_manifest_sha256=expected_manifest_sha256,
    )
    temporary = tempfile.TemporaryDirectory(prefix="warehouse-r41-diagnostic-online-")
    root = Path(temporary.name)
    try:
        paths = _write_material(root, artifacts)
        identities, parent = manifest["identities"], manifest["parent"]
        runtime_manifest = _parse_json(
            artifacts["runtime_manifest"], "diagnostic runtime manifest")
        play, tutorial_scene = _validate_runtime_manifest(runtime_manifest, parent)
        if (file_hash(paths["runtime_manifest"])
                != identities["runtime_manifest_file_sha256"]
                or runtime_manifest["content_sha256"]
                    != identities["runtime_manifest_content_sha256"]
                or digest(runtime_manifest)
                    != identities["runtime_manifest_semantic_sha256"]
                or runtime_manifest["source_full_manifest_version"]
                    != identities["source_full_manifest_version"]
                or runtime_manifest["source_conflict_validation_version"]
                    != identities["source_conflict_validation_version"]
                or runtime_manifest["source_conflict_validation_sha256"]
                    != identities["source_conflict_validation_sha256"]
                or runtime_manifest["diagnostic_contract_version"]
                    != identities["diagnostic_contract_version"]
                or runtime_manifest["diagnostic_contract_sha256"]
                    != identities["diagnostic_contract_sha256"]
                or runtime_manifest["diagnostic_conflict_graph_sha256"]
                    != identities["diagnostic_conflict_graph_sha256"]
                or runtime_manifest["conflict_families_sha256"]
                    != identities["conflict_families_sha256"]
                or [row["id"] for row in play] != identities["play_scene_ids"]
                or [row["fingerprint"] for row in play]
                    != identities["play_scene_fingerprints"]):
            raise ValueError("Diagnostic portable scene identity differs")
        runtime = _runtime(paths, identities)
        for scene in play:
            environment = runtime.environment(deepcopy(scene))
            if environment.state.frame != 0:
                raise ValueError("Diagnostic play scene does not start at frame zero")
        explainer = R41DiagnosticOnlineAlignmentExplainer(
            paths["program"],
            expected_program_sha256=identities["program_sha256"],
            runtime=runtime,
        )
        if explainer.signature != identities["parent_explainer_signature"]:
            raise ValueError("Diagnostic explainer identity differs")
        explainer._assert_current(runtime)
        question_payload = _parse_json(
            artifacts["question_bank"], "diagnostic question bank")
        bank = _diagnostic_portable_bank(
            question_payload, runtime=runtime, identities=identities,
            raw=artifacts["question_bank"],
        )
        tutorial = _parse_json(artifacts["tutorial"], "diagnostic tutorial")
        expected_tutorial_bindings = {
            "scene_manifest_version":
                FULL_SCENE_MANIFEST_VERSION,
            "scene_manifest_file_sha256": parent["conflict_manifest_file_sha256"],
            "scene_manifest_content_sha256": parent["conflict_manifest_content_sha256"],
            "scene_manifest_semantic_sha256": parent["conflict_manifest_semantic_sha256"],
            "tutorial_scene_fingerprint": identities["tutorial_scene_fingerprint"],
            "tutorial_successor_state_sha256": identities[
                "tutorial_successor_state_sha256"],
            "tutorial_snapshot_sha256": identities["tutorial_snapshot_sha256"],
            "diagnostic_contract_sha256": identities["diagnostic_contract_sha256"],
            "diagnostic_contract_version": identities[
                "diagnostic_contract_version"],
            "diagnostic_conflict_graph_sha256": identities[
                "diagnostic_conflict_graph_sha256"],
            "conflict_families_sha256": identities["conflict_families_sha256"],
            "producer_sources_sha256": digest(tutorial_api.producer_sources()),
        }
        tutorial_replay = tutorial_api.validate_neutral_tutorial(
            tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
            expected_bindings=expected_tutorial_bindings,
        )
        if (digest(tutorial) != identities["tutorial_signature"]
                or tutorial_replay["tutorial_signature"]
                    != identities["tutorial_signature"]):
            raise ValueError("Diagnostic tutorial identity differs")
        source_binding = deepcopy(manifest["sources"]["release"])
        if source_binding != release_sources():
            raise ValueError("Diagnostic serving sources changed")
        scenarios = deepcopy(runtime_manifest)
        provenance = {
            "version": VERSION,
            "release_version": PUBLIC_RELEASE_VERSION,
            "namespace": "internal_diagnostic",
            "pilot_class": PILOT_CLASS,
            "manifest_sha256": expected_manifest_sha256,
            "package_sha256": expected_package_sha256,
            "diagnostic_admission_sha256": parent["diagnostic_admission_sha256"],
            "diagnostic_designation_sha256": parent[
                "diagnostic_designation_sha256"],
            "actor_sha256": runtime.actor_sha256,
            "runtime_signature": runtime.signature,
            "runtime_manifest_signature": runtime.runtime_manifest_signature,
            "protocol_sha256": runtime.protocol_sha256,
            "scenario_manifest_sha256": digest(runtime_manifest),
            "program_sha256": explainer.program_sha256,
            "explainer_signature": explainer.signature,
            "question_bank_signature": bank.signature,
            "tutorial_signature": identities["tutorial_signature"],
            "source_binding": deepcopy(source_binding),
            "release": deepcopy(manifest["release"]),
            "online_portable": True,
            "formal_ready": False,
            "formal_sample_eligible": False,
            "data_persistent": False,
        }
        evidence = {
            "version": VERSION,
            "parent": deepcopy(parent),
            "identities": deepcopy(identities),
            "artifact_sha256": {
                name: record["sha256"] for name, record in manifest["artifacts"].items()
            },
            "tutorial_replay": tutorial_replay,
            "behavior_performance_gate_passed": False,
            "behavior_performance_gate_waived": True,
            "runtime_action_override": False,
            "formal_ready": False,
            "formal_sample_eligible": False,
        }
        signature = digest({
            "version": VERSION,
            "manifest_sha256": expected_manifest_sha256,
            "package_sha256": expected_package_sha256,
            "runtime_signature": runtime.signature,
            "runtime_manifest_signature": runtime.runtime_manifest_signature,
            "program_sha256": explainer.program_sha256,
            "question_bank_signature": bank.signature,
            "tutorial_signature": identities["tutorial_signature"],
            "sources": source_binding,
        })
        return R41DiagnosticOnlineReleaseContext(
            runtime, scenarios, explainer, bank, deepcopy(tutorial),
            identities["tutorial_signature"], evidence, source_binding,
            expected_manifest_sha256, signature, deepcopy(manifest["release"]),
            provenance, root, expected_package_sha256, temporary,
        )
    except BaseException:
        temporary.cleanup()
        raise


def assemble_from_admitted_components(*,
          diagnostic_admission_path: str | Path,
          expected_diagnostic_admission_sha256: str,
          components: Mapping[str, str | Path],
          output_package: str | Path,
          output_base64: str | Path | None = None) -> dict[str, Any]:
    from backend.training import warehouse_r41_diagnostic_admission as admission_api

    if not isinstance(components, Mapping) or set(components) != set(
            admission_api.ARTIFACT_NAMES):
        raise ValueError("Exact admitted diagnostic component set required")
    paths = {
        name: _component_path(components[name], "diagnostic " + name)
        for name in admission_api.ARTIFACT_NAMES
    }
    admission_path = _component_path(
        diagnostic_admission_path, "diagnostic admission")
    admission_sha = _sha(
        expected_diagnostic_admission_sha256, "diagnostic admission")
    admission = admission_api.read_saved_admission(
        admission_path, expected_sha256=admission_sha, components=paths,
    )
    if (admission.get("status") != "admitted_internal_diagnostic"
            or admission.get("admitted") is not True
            or admission.get("behavior_performance_gate_passed") is not False
            or admission.get("behavior_performance_gate_waived") is not True
            or admission.get("waiver_scope") != ["behavior_performance"]
            or admission.get("formal_ready") is not False
            or admission.get("formal_sample_eligible") is not False
            or admission.get("data_persistent") is not False):
        raise ValueError("Exact internal diagnostic admission required")
    bindings = admission["bindings"]
    full_manifest = _read_json(paths["conflict_manifest"], "diagnostic full manifest")
    selection = _read_json(paths["selected_scenes"], "diagnostic selected scenes")
    runtime_manifest = _runtime_manifest(selection, full_manifest, bindings)
    runtime_manifest_raw = (canonical(runtime_manifest) + "\n").encode()
    protocol_raw = paths["protocol"].read_bytes()
    question_raw = paths["question_bank"].read_bytes()
    tutorial_raw = paths["tutorial"].read_bytes()
    program_raw = paths["final_rcpd_program"].read_bytes()
    artifacts = {
        "actor": paths["actor"].read_bytes(),
        "protocol": protocol_raw,
        "runtime_manifest": runtime_manifest_raw,
        "program": program_raw,
        "question_bank": question_raw,
        "tutorial": tutorial_raw,
    }
    with tempfile.TemporaryDirectory(prefix="warehouse-r41-diagnostic-assemble-") as tmp:
        material = _write_material(Path(tmp), artifacts)
        # Runtime construction precedes final identities because its distinct
        # serving-manifest signature is itself part of the package identity.
        content = deepcopy(runtime_manifest); content_sha = content.pop("content_sha256")
        runtime = R41DiagnosticOnlineAlignmentRuntime(
            material["actor"], training_protocol_path=material["protocol"],
            manifest_path=material["runtime_manifest"],
            expected_actor_sha256=bindings["actor_sha256"],
            expected_training_protocol_file_sha256=bindings["protocol_file_sha256"],
            expected_training_protocol_content_sha256=bindings["protocol_content_sha256"],
            expected_manifest_file_sha256=file_hash(material["runtime_manifest"]),
            expected_manifest_content_sha256=content_sha,
            expected_manifest_semantic_sha256=digest(runtime_manifest),
        )
        runtime.verify_binding()
        if runtime.signature != bindings["runtime_signature"]:
            raise ValueError("Diagnostic full and portable runtime identities differ")
        explainer = R41DiagnosticOnlineAlignmentExplainer(
            material["program"], expected_program_sha256=bindings["program_sha256"],
            runtime=runtime,
        )
        explainer._assert_current(runtime)
        question = _parse_json(question_raw, "diagnostic question bank")
        tutorial = _parse_json(tutorial_raw, "diagnostic tutorial")
        play, tutorial_scene = _validate_runtime_manifest(runtime_manifest, {
            "conflict_manifest_file_sha256": bindings["conflict_manifest_file_sha256"],
            "conflict_manifest_content_sha256": bindings["conflict_manifest_content_sha256"],
            "conflict_manifest_semantic_sha256": bindings["conflict_manifest_semantic_sha256"],
            "conflict_validation_sha256": bindings["conflict_validation_sha256"],
            "dynamic_selection_report_sha256": bindings["dynamic_selection_report_sha256"],
            "selected_scenes_file_sha256": bindings["selected_scenes_file_sha256"],
            "selected_scenes_semantic_sha256": bindings["selected_scenes_semantic_sha256"],
        })
        identities = {
            "actor_sha256": bindings["actor_sha256"],
            "actor_parameters_sha256": bindings["actor_parameters_sha256"],
            "protocol_file_sha256": bindings["protocol_file_sha256"],
            "protocol_content_sha256": bindings["protocol_content_sha256"],
            "program_sha256": bindings["program_sha256"],
            "parent_runtime_signature": runtime.signature,
            "runtime_manifest_signature": runtime.runtime_manifest_signature,
            "parent_explainer_signature": explainer.signature,
            "parent_question_bank_signature": question["source_bank_signature"],
            "question_bank_private_items_sha256": question[
                "private_items_sha256"],
            "question_bank_public_items_sha256": question[
                "public_items_sha256"],
            "runtime_manifest_file_sha256": file_hash(material["runtime_manifest"]),
            "runtime_manifest_content_sha256": runtime_manifest["content_sha256"],
            "runtime_manifest_semantic_sha256": digest(runtime_manifest),
            "source_full_manifest_file_sha256": bindings[
                "conflict_manifest_file_sha256"],
            "source_full_manifest_content_sha256": bindings[
                "conflict_manifest_content_sha256"],
            "source_full_manifest_semantic_sha256": bindings[
                "conflict_manifest_semantic_sha256"],
            "source_full_manifest_version": FULL_SCENE_MANIFEST_VERSION,
            "source_conflict_validation_version": CONFLICT_VALIDATION_VERSION,
            "source_conflict_validation_sha256": bindings[
                "conflict_validation_sha256"],
            "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
            "diagnostic_conflict_graph_sha256": bindings[
                "conflict_graph_sha256"],
            "play_scene_count": 7,
            "play_scene_ids": [row["id"] for row in play],
            "play_scene_fingerprints": [row["fingerprint"] for row in play],
            "tutorial_signature": digest(tutorial),
            "tutorial_scene_id": tutorial_scene["id"],
            "tutorial_scene_fingerprint": tutorial_scene["fingerprint"],
            "tutorial_successor_state_sha256": tutorial_scene["snapshot"]
                ["r41_diagnostic_conflict"]["binding_sha256"],
            "tutorial_snapshot_sha256": digest(tutorial_scene["snapshot"]),
            "diagnostic_contract_sha256": bindings["diagnostic_contract_sha256"],
            "conflict_families_sha256": bindings["conflict_families_sha256"],
            "uses_terminal_designated_actor": True,
            "action_override_count": 0,
        }
        if (identities["parent_explainer_signature"]
                != bindings["explainer_signature"]
                or identities["parent_question_bank_signature"]
                    != bindings["question_bank_signature"]
                or identities["tutorial_signature"] != bindings["tutorial_signature"]):
            raise ValueError("Diagnostic admitted explanation/question/tutorial differs")
        bank = _diagnostic_portable_bank(
            question, runtime=runtime, identities=identities, raw=question_raw,
        )
        bank.verify_binding()
        expected_tutorial = tutorial_api.bindings(
            full_manifest, tutorial_scene,
            manifest_file_sha256=bindings["conflict_manifest_file_sha256"],
        )
        tutorial_result = tutorial_api.validate_neutral_tutorial(
            tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
            expected_bindings=expected_tutorial,
        )
        if tutorial_result["tutorial_signature"] != identities["tutorial_signature"]:
            raise ValueError("Diagnostic tutorial replay identity differs")

    parent = {
        "version": admission["version"],
        "status": admission["status"],
        "diagnostic_admission_sha256": admission_sha,
        "diagnostic_designation_sha256": bindings["diagnostic_designation_sha256"],
        "training_ledger_sha256": bindings["training_ledger_sha256"],
        "dual_evaluation_sha256": bindings["dual_evaluation_sha256"],
        "failure_closeout_sha256": bindings["failure_closeout_sha256"],
        "conflict_manifest_file_sha256": bindings["conflict_manifest_file_sha256"],
        "conflict_manifest_content_sha256": bindings["conflict_manifest_content_sha256"],
        "conflict_manifest_semantic_sha256": bindings["conflict_manifest_semantic_sha256"],
        "conflict_validation_sha256": bindings["conflict_validation_sha256"],
        "dynamic_selection_report_sha256": bindings["dynamic_selection_report_sha256"],
        "selected_scenes_file_sha256": bindings["selected_scenes_file_sha256"],
        "selected_scenes_semantic_sha256": bindings["selected_scenes_semantic_sha256"],
        "final_rcpd_report_sha256": bindings["final_rcpd_report_sha256"],
        "explanation_audit_sha256": bindings["explanation_audit_sha256"],
        "question_bank_report_sha256": bindings["question_bank_report_sha256"],
        "tutorial_sha256": bindings["tutorial_sha256"],
    }
    for label, value in (("protocol", _parse_json(protocol_raw, "protocol")),
                         ("runtime manifest", runtime_manifest),
                         ("question bank", question), ("tutorial", tutorial),
                         ("admission", admission)):
        _reject_sensitive(value, "diagnostic " + label)
    manifest = {
        "version": VERSION,
        "status": STATUS,
        "test_fixture": False,
        "pilot_class": PILOT_CLASS,
        "formal_ready": False,
        "formal_sample_eligible": False,
        "data_persistent": False,
        "parent": parent,
        "artifacts": _artifact_records(artifacts),
        "identities": identities,
        "sources": {"release": release_sources()},
        "analysis": portable._analysis_protocol(),
        "release": _release_projection(),
    }
    raw = _archive_bytes(manifest, artifacts)
    package = portable._write_new(output_package, raw)
    encoded_path = None
    encoded = base64.b64encode(raw) + b"\n"
    if len(encoded) > MAX_BASE64_BYTES:
        package.unlink(missing_ok=True)
        raise ValueError("Diagnostic Base64 Secret File exceeds 1 MB")
    if output_base64 is not None:
        try:
            encoded_path = portable._write_new(output_base64, encoded)
        except BaseException:
            package.unlink(missing_ok=True)
            raise
    manifest_sha = sha256((canonical(manifest) + "\n").encode()).hexdigest()
    package_sha = sha256(raw).hexdigest()
    try:
        loaded = load_online_release(
            package_path=package, expected_package_sha256=package_sha,
            expected_manifest_sha256=manifest_sha,
        )
        loaded.close()
    except BaseException:
        package.unlink(missing_ok=True)
        if encoded_path is not None:
            encoded_path.unlink(missing_ok=True)
        raise
    return {
        "version": VERSION,
        "status": "built_from_diagnostic_admission_and_independently_reloaded",
        "release_version": PUBLIC_RELEASE_VERSION,
        "pilot_class": PILOT_CLASS,
        "package": str(package),
        "package_size": len(raw),
        "package_sha256": package_sha,
        "base64": str(encoded_path) if encoded_path else None,
        "base64_size": len(encoded),
        "manifest_sha256": manifest_sha,
        "diagnostic_admission_sha256": admission_sha,
        "actor_sha256": identities["actor_sha256"],
        "program_sha256": identities["program_sha256"],
        "runtime_manifest_signature": identities["runtime_manifest_signature"],
        "formal_ready": False,
        "formal_sample_eligible": False,
        "data_persistent": False,
    }


__all__ = [
    "VERSION", "STATUS", "PUBLIC_RELEASE_VERSION", "PILOT_CLASS",
    "MANIFEST_NAME", "RUNTIME_MANIFEST_VERSION", "DYNAMIC_SELECTION_VERSION",
    "ARTIFACT_PATHS",
    "ARCHIVE_WHITELIST", "MAX_PACKAGE_BYTES", "MAX_BASE64_BYTES",
    "R41DiagnosticParticipantQuestionBank", "R41DiagnosticOnlineReleaseContext",
    "assemble_from_admitted_components",
    "inspect_online_release", "load_online_release", "release_sources",
]
