from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from backend.training.warehouse_native_common import canonical, digest
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256, DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256, DIAGNOSTIC_CONTRACT_VERSION,
)
from ui import warehouse_alignment_online_release as portable
from ui import warehouse_alignment_r41_diagnostic_release as release


ZERO = "0" * 64


def _artifacts(question=None):
    return {
        "actor": b"actor", "protocol": b"{}\n",
        "runtime_manifest": b"{}\n", "program": b"{}\n",
        "question_bank": question or b"{}\n", "tutorial": b"{}\n",
    }


def _identities():
    value = {name: ZERO for name in release._IDENTITY_FIELDS}
    value.update({
        "source_full_manifest_version": release.FULL_SCENE_MANIFEST_VERSION,
        "source_conflict_validation_version": release.CONFLICT_VALIDATION_VERSION,
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "play_scene_count": 7,
        "play_scene_ids": [f"scene-{index}" for index in range(7)],
        "play_scene_fingerprints": [f"{index + 1:064x}" for index in range(7)],
        "uses_terminal_designated_actor": True,
        "action_override_count": 0,
    })
    return value


def _manifest(monkeypatch, artifacts):
    source_name = "ui/warehouse_alignment_r41_diagnostic_release.py"
    sources = {source_name: release.file_hash(release.ROOT / source_name)}
    monkeypatch.setattr(release, "release_sources", lambda: sources)
    parent = {name: ZERO for name in release._PARENT_FIELDS}
    parent.update({
        "version": "warehouse-r41-diagnostic-admission.v1",
        "status": "admitted_internal_diagnostic",
    })
    return {
        "version": release.VERSION, "status": release.STATUS,
        "test_fixture": False, "pilot_class": "internal_diagnostic",
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False, "parent": parent,
        "artifacts": release._artifact_records(artifacts),
        "identities": _identities(), "sources": {"release": sources},
        "analysis": portable._analysis_protocol(),
        "release": release._release_projection(),
    }


def test_diagnostic_archive_is_deterministic_closed_and_nonformal(monkeypatch, tmp_path):
    artifacts = _artifacts()
    manifest = _manifest(monkeypatch, artifacts)
    first = release._archive_bytes(manifest, artifacts)
    assert first == release._archive_bytes(manifest, artifacts)
    package = tmp_path / "diagnostic.zip"; package.write_bytes(first)
    manifest_sha = sha256((canonical(manifest) + "\n").encode()).hexdigest()
    inspected = release.inspect_online_release(
        package_path=package, expected_package_sha256=sha256(first).hexdigest(),
        expected_manifest_sha256=manifest_sha,
    )
    assert inspected["release"]["release_version"] == "r4.1-diagnostic"
    assert inspected["release"]["formal_ready"] is False
    assert inspected["release"]["formal_sample_eligible"] is False
    assert inspected["release"]["data_persistent"] is False
    assert inspected["release"]["behavior_performance_gate_waived"] is True
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == release.ARCHIVE_WHITELIST
    assert len(base64.b64encode(first) + b"\n") <= release.MAX_BASE64_BYTES


def test_diagnostic_archive_rejects_extra_member(monkeypatch, tmp_path):
    artifacts = _artifacts()
    manifest = _manifest(monkeypatch, artifacts)
    raw = release._archive_bytes(manifest, artifacts)
    changed = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(changed, "w") as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr("artifacts/unbound.json", b"{}\n")
    bad = changed.getvalue()
    with pytest.raises(ValueError, match="member set"):
        release._read_archive(
            bad, expected_package_sha256=sha256(bad).hexdigest(),
            expected_manifest_sha256=sha256(
                (canonical(manifest) + "\n").encode()).hexdigest(),
        )


def test_diagnostic_archive_rejects_symlink_member(monkeypatch):
    artifacts = _artifacts()
    manifest = _manifest(monkeypatch, artifacts)
    raw = release._archive_bytes(manifest, artifacts)
    changed = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(
            changed, "w") as target:
        for original in source.infolist():
            info = zipfile.ZipInfo(original.filename)
            info.compress_type = original.compress_type
            info.external_attr = original.external_attr
            if original.filename == release.ARTIFACT_PATHS["actor"]:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            target.writestr(info, source.read(original.filename))
    bad = changed.getvalue()
    with pytest.raises(ValueError, match="member is unsafe"):
        release._read_archive(
            bad, expected_package_sha256=sha256(bad).hexdigest(),
            expected_manifest_sha256=sha256(
                (canonical(manifest) + "\n").encode()).hexdigest(),
        )


def test_diagnostic_archive_rejects_traversal_member(monkeypatch):
    artifacts = _artifacts()
    manifest = _manifest(monkeypatch, artifacts)
    raw = release._archive_bytes(manifest, artifacts)
    changed = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(
            changed, "w") as target:
        for info in source.infolist():
            name = ("../actor.npz" if info.filename
                    == release.ARTIFACT_PATHS["actor"] else info.filename)
            target.writestr(name, source.read(info.filename))
    bad = changed.getvalue()
    with pytest.raises(ValueError, match="member set"):
        release._read_archive(
            bad, expected_package_sha256=sha256(bad).hexdigest(),
            expected_manifest_sha256=sha256(
                (canonical(manifest) + "\n").encode()).hexdigest(),
        )


def test_diagnostic_secret_file_reader_enforces_size_and_regular_file(tmp_path):
    oversized = tmp_path / "oversized.b64"
    oversized.write_bytes(b"A" * (release.MAX_BASE64_BYTES + 1))
    with pytest.raises(ValueError, match="safe limit"):
        release._package_bytes(base64_path=oversized)
    directory = tmp_path / "directory.b64"
    directory.mkdir()
    with pytest.raises(ValueError, match="Regular diagnostic Base64"):
        release._package_bytes(base64_path=directory)


def test_diagnostic_archive_rejects_changed_parent_version(monkeypatch):
    artifacts = _artifacts()
    manifest = _manifest(monkeypatch, artifacts)
    manifest["parent"]["version"] = "warehouse-r41-production-admission.v1"
    with pytest.raises(ValueError, match="parent binding"):
        release._archive_bytes(manifest, artifacts)


def test_loader_rejects_rehashed_archive_with_forged_parent_version(monkeypatch):
    artifacts = _artifacts()
    manifest = _manifest(monkeypatch, artifacts)
    original = release._archive_bytes(manifest, artifacts)
    forged = json.loads(json.dumps(manifest))
    forged["parent"]["version"] = "warehouse-r41-diagnostic-admission.forged"
    forged_raw = (canonical(forged) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            raw = (forged_raw if info.filename == release.MANIFEST_NAME
                   else source.read(info.filename))
            target.writestr(info, raw)
    package = stream.getvalue()
    with pytest.raises(ValueError, match="parent binding"):
        release._read_archive(
            package, expected_package_sha256=sha256(package).hexdigest(),
            expected_manifest_sha256=sha256(forged_raw).hexdigest(),
        )


def test_compact_manifest_binds_full_evidence_and_six_families():
    full = {
        "version": "warehouse-r41-diagnostic-conflict-scene-manifest.v3",
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "splits": {},
    }
    full["content_sha256"] = digest(full)
    scenes = [{
        "id": f"scene-{index}", "fingerprint": f"{index + 1:064x}",
        "family_id": "tutorial" if index == 0 else f"family-{index}",
    } for index in range(7)]
    selection = {
        "version": "warehouse-r41-diagnostic-conflict-dynamic-selection.v3",
        "status": "accepted_diagnostic_dynamic_selection",
        "release_eligible": True, "actor_sha256": "a" * 64,
        "source_manifest_file_sha256": "b" * 64,
        "source_manifest_content_sha256": full["content_sha256"],
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "protocol_sha256": "c" * 64,
        "tutorial": scenes[0], "X": scenes[1:4], "Y": scenes[4:],
        "pairs": [], "balance": {}, "selection_score": 0,
        "six_distinct_conflict_families": True,
        "zero_action_overrides": True,
        "no_replacement_pickup_on_agent": True,
        "no_replacement_delivery_on_agent": True,
        "no_replacement_endpoint_on_agent": True,
        "ordinary_sampler_fallback": False,
    }
    bindings = {
        "actor_sha256": "a" * 64,
        "conflict_manifest_file_sha256": "b" * 64,
        "conflict_manifest_content_sha256": full["content_sha256"],
        "conflict_manifest_semantic_sha256": digest(full),
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "dynamic_selection_report_sha256": "d" * 64,
        "conflict_validation_sha256": "f" * 64,
        "selected_scenes_file_sha256": "e" * 64,
        "selected_scenes_semantic_sha256": digest(selection),
    }
    compact = release._runtime_manifest(selection, full, bindings)
    parent = {
        "conflict_manifest_file_sha256": bindings["conflict_manifest_file_sha256"],
        "conflict_manifest_content_sha256": bindings["conflict_manifest_content_sha256"],
        "conflict_manifest_semantic_sha256": bindings["conflict_manifest_semantic_sha256"],
        "conflict_validation_sha256": bindings["conflict_validation_sha256"],
        "dynamic_selection_report_sha256": bindings["dynamic_selection_report_sha256"],
        "selected_scenes_file_sha256": bindings["selected_scenes_file_sha256"],
        "selected_scenes_semantic_sha256": bindings["selected_scenes_semantic_sha256"],
    }
    play, tutorial = release._validate_runtime_manifest(compact, parent)
    assert len(play) == 7 and tutorial == scenes[0]
    assert len({row["family_id"] for row in play[1:]}) == 6
    assert compact["source_full_manifest_file_sha256"] == "b" * 64
    assert compact["source_full_manifest_version"] == (
        "warehouse-r41-diagnostic-conflict-scene-manifest.v3")
    assert compact["source_conflict_validation_version"] == (
        "warehouse-r41-diagnostic-conflict-scene-validation.v3")
    assert compact["diagnostic_conflict_graph_sha256"] == (
        DIAGNOSTIC_CONFLICT_GRAPH_SHA256)

    for field, legacy in (
        ("source_full_manifest_version",
         "warehouse-r41-diagnostic-conflict-scene-manifest.v2"),
        ("source_conflict_validation_version",
         "warehouse-r41-diagnostic-conflict-scene-validation.v2"),
        ("diagnostic_contract_version",
         "warehouse-r41-diagnostic-conflict.v1"),
        ("diagnostic_conflict_graph_sha256", "0" * 64),
    ):
        changed = deepcopy(compact)
        changed.pop("content_sha256")
        changed[field] = legacy
        changed["content_sha256"] = digest(changed)
        with pytest.raises(ValueError, match="source identity"):
            release._validate_runtime_manifest(changed, parent)

    legacy_selection = deepcopy(selection)
    legacy_selection["version"] = (
        "warehouse-r41-diagnostic-conflict-dynamic-selection.v2")
    legacy_bindings = deepcopy(bindings)
    legacy_bindings["selected_scenes_semantic_sha256"] = digest(legacy_selection)
    with pytest.raises(ValueError, match="selected scenes"):
        release._runtime_manifest(legacy_selection, full, legacy_bindings)

    legacy_full = deepcopy(full)
    legacy_full["version"] = (
        "warehouse-r41-diagnostic-conflict-scene-manifest.v2")
    legacy_full.pop("content_sha256")
    legacy_full["content_sha256"] = digest(legacy_full)
    legacy_full_bindings = deepcopy(bindings)
    legacy_full_bindings.update(
        conflict_manifest_content_sha256=legacy_full["content_sha256"],
        conflict_manifest_semantic_sha256=digest(legacy_full),
        selected_scenes_semantic_sha256=digest(selection),
    )
    legacy_selection_for_full = deepcopy(selection)
    legacy_selection_for_full["source_manifest_content_sha256"] = (
        legacy_full["content_sha256"])
    legacy_full_bindings["selected_scenes_semantic_sha256"] = digest(
        legacy_selection_for_full)
    with pytest.raises(ValueError, match="full manifest source contract"):
        release._runtime_manifest(
            legacy_selection_for_full, legacy_full, legacy_full_bindings)


def test_packaged_diagnostic_question_projection_reaches_participant_schema(
        monkeypatch, tmp_path):
    from ui.warehouse_alignment_online_server import (
        _participant_questionnaire_items,
    )

    runtime = SimpleNamespace(actor_sha256="a" * 64,
                              protocol_sha256="b" * 64,
                              signature="c" * 64)
    source_name = "backend/training/warehouse_r41_diagnostic_question_bank.py"
    sources = {source_name: release.file_hash(release.ROOT / source_name)}
    anchors = {}
    bank_sha = "e" * 64
    source_signature = digest({
        "version": portable.PortableQuestionBank._VERSION,
        "anchors": anchors, "bank_sha256": bank_sha, "sources": sources,
    })
    positions = ((2, 2), (2, 3), (3, 2), (3, 3))
    options = [{
        "value": f"{row},{col}", "marker": marker, "position": [row, col],
        "label": {"zh": marker, "en": marker},
    } for marker, (row, col) in zip("ABCD", positions)]
    private_item = {
        "id": "prediction_wait_three_1", "kind": "wait_three",
        "scenario_id": "question_scene_1", "frame": 7,
        "snapshot": {"private": "snapshot-secret"},
        "snapshot_sha256": "2" * 64, "answer": "3,2",
        "diversity_key": "1,0", "evidence": {"private": "evidence-secret"},
        "prompt": {"zh": "三步后在哪里？", "en": "Where after three steps?"},
        "options": options,
        "preview": {
            "map": {"rows": 6, "cols": 7, "shelves": [],
                    "charger_position": [5, 3],
                    "robot_exit_positions": [[4, 2], [4, 3], [4, 4]],
                    "waiting_zone": [[5, 2], [5, 4]],
                    "robot_start_positions": [[5, 2], [5, 4]],
                    "shared_delivery_tasks": True},
            "state": {"frame": 7, "tasks": [], "total_deliveries": 0,
                      "collision_count": 0, "shutdown_count": 0,
                      "terminated": False, "truncated": False,
                      "terminal_reason": None,
                      "agents": [
                          {"agent_id": "robot_1", "position": [5, 2],
                           "battery": 90.0, "carrying_task_id": None},
                          {"agent_id": "robot_2", "position": [4, 2],
                           "battery": 88.0, "carrying_task_id": None},
                      ]},
            "public_feedback": {"valid": False},
            "question_markers": [
                {"marker": marker, "position": [row, col]}
                for marker, (row, col) in zip("ABCD", positions)
            ],
        },
    }
    raw_public = [{
        "id": private_item["id"], "type": "choice",
        "prediction_kind": private_item["kind"], "required": True,
        "prompt": deepcopy(private_item["prompt"]),
        "options": [{"value": option["value"],
                     "label": deepcopy(option["label"])} for option in options],
        "preview": deepcopy(private_item["preview"]),
        "source_frame": private_item["frame"],
        "source_scenario": private_item["scenario_id"],
    }]
    payload = {
        "version": "warehouse-alignment-portable-question-projection.v1",
        "test_fixture": False, "source_bank_signature": source_signature,
        "anchors": anchors, "bank_sha256": bank_sha,
        "replay_receipt_sha256": "f" * 64,
        "pool_manifest_sha256": "1" * 64,
        "actor_sha256": runtime.actor_sha256,
        "runtime_signature": runtime.signature,
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_family": "alignment_feedback197", "sources": sources,
        "checks": {}, "items": [private_item],
        "private_items_sha256": digest([private_item]),
        "public_items_sha256": digest(raw_public), "summary_sha256": "0" * 64,
    }
    summary = {
        "status": "candidate_ready", "formal_ready": False,
        "release_ready": False, "model_capability_evaluated": False,
        "available": True, "item_count": 1, "test_fixture": False,
        "preview_history_available_to_both_conditions": True,
        "version": portable.PortableQuestionBank._VERSION,
        "eligible": False, "participant_enabled": False,
        "independent_replay_previously_verified": True,
        "physics_replay_on_load": False,
        "runtime_family": "alignment_feedback197", "bank_sha256": bank_sha,
        "replay_receipt_sha256": "f" * 64,
    }
    payload["summary_sha256"] = digest(summary)
    question_raw = (canonical(payload) + "\n").encode()
    artifacts = _artifacts(question_raw)
    manifest = _manifest(monkeypatch, artifacts)
    identities = manifest["identities"]
    identities.update({
        "actor_sha256": runtime.actor_sha256,
        "protocol_content_sha256": runtime.protocol_sha256,
        "parent_runtime_signature": runtime.signature,
        "parent_question_bank_signature": source_signature,
        "question_bank_private_items_sha256": digest([private_item]),
        "question_bank_public_items_sha256": digest(raw_public),
    })
    manifest["artifacts"] = release._artifact_records(artifacts)
    raw = release._archive_bytes(manifest, artifacts)
    package = tmp_path / "diagnostic.zip"; package.write_bytes(raw)
    unpacked, archived = release._read_archive(
        raw, expected_package_sha256=sha256(raw).hexdigest(),
        expected_manifest_sha256=sha256(
            (canonical(manifest) + "\n").encode()).hexdigest(),
    )
    monkeypatch.setattr(portable, "_validate_question_projection",
                        lambda value, ids: (value, sources))
    bank = release._diagnostic_portable_bank(
        json.loads(archived["question_bank"]), runtime=runtime,
        identities=unpacked["identities"], raw=archived["question_bank"],
    )
    assert type(bank) is release.R41DiagnosticParticipantQuestionBank
    assert bank.signature and bank.verify_binding() == bank.signature
    projected = bank.public_items()
    assert projected[0]["preview"]["question_markers"] == [
        {"label": marker, "position": [row, col]}
        for marker, (row, col) in zip("ABCD", positions)
    ]
    participant = _participant_questionnaire_items(projected)
    assert participant[0]["preview"]["question_markers"] ==         projected[0]["preview"]["question_markers"]
    serialized = canonical(participant)
    assert "snapshot-secret" not in serialized and "evidence-secret" not in serialized
    assert '\"answer\"' not in serialized and '\"marker\"' not in serialized


@pytest.mark.parametrize("mutation", (
    lambda markers: markers[0].update({"label": "A"}),
    lambda markers: markers[0].pop("marker"),
    lambda markers: markers[1].update({"marker": "A"}),
    lambda markers: markers[1].update({"position": [2, 2]}),
    lambda markers: markers[1].update({"position": [99, 99]}),
))
def test_diagnostic_participant_marker_projection_fails_closed(mutation):
    positions = ((2, 2), (2, 3), (3, 2), (3, 3))
    markers = [{"marker": marker, "position": [row, col]}
               for marker, (row, col) in zip("ABCD", positions)]
    item = {
        "id": "prediction_wait_three_1", "type": "choice",
        "prediction_kind": "wait_three", "required": True,
        "prompt": {"zh": "q", "en": "q"},
        "options": [{"value": f"{row},{col}", "label": {"zh": marker, "en": marker}}
                    for marker, (row, col) in zip("ABCD", positions)],
        "preview": {"map": {"rows": 6, "cols": 7},
                    "question_markers": markers},
        "source_frame": 7, "source_scenario": "question_scene_1",
    }
    mutation(markers)
    with pytest.raises(ValueError, match="marker"):
        release._participant_question_projection([item])


def test_diagnostic_release_loader_import_is_numpy_only():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import ui.warehouse_alignment_r41_diagnostic_release; "
        "assert 'torch' not in sys.modules",
    ], cwd=Path(__file__).resolve().parents[1], text=True,
       capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_diagnostic_release_source_scope_binds_model_tree_runtime():
    sources = release.release_sources()
    assert "backend/warehouse_r41_diagnostic_model_tree.py" in sources
