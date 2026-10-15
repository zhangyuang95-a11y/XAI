"""Public protocol checks for the r4.1 diagnostic online classification."""
from __future__ import annotations

import http.client
from hashlib import sha256
import json
from pathlib import Path
import threading
from uuid import uuid4

import pytest

from tests.test_warehouse_alignment_online_server import FakeContext
from ui import warehouse_alignment_online_server as online


def diagnostic_context(root: str | Path) -> FakeContext:
    context = FakeContext(root)
    context.provenance.update(
        version=online.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSION,
        package_sha256="8" * 64,
        manifest_sha256="9" * 64,
        diagnostic_actor_designation_sha256="7" * 64,
        behavior_performance_gate_waived=True,
    )
    context.release.update(
        release_version="r4.1-diagnostic",
        pilot_class="internal_diagnostic",
        formal_sample_eligible=False,
        human_explanation_effect_validated=False,
        behavior_performance_gate_passed=False,
        behavior_performance_gate_waived=True,
        data_persistent=False,
    )
    for index, scene in enumerate(context.scenarios["splits"]["play"]):
        scene["fingerprint"] = format(index + 1, "064x")
        scene["seed"] = 910_000 + index
    context.tutorial["version"] = (
        "warehouse-alignment-diagnostic-neutral-tutorial.v3")
    context.tutorial["bindings"] = {
        "scene_manifest_version":
            "warehouse-r41-diagnostic-conflict-scene-manifest.v3",
        "scene_manifest_file_sha256": "0" * 64,
        "scene_manifest_content_sha256": "1" * 64,
        "scene_manifest_semantic_sha256": "2" * 64,
        "tutorial_scene_fingerprint": "3" * 64,
        "tutorial_successor_state_sha256": "4" * 64,
        "tutorial_snapshot_sha256": "5" * 64,
        "diagnostic_contract_sha256": "6" * 64,
        "diagnostic_contract_version": "warehouse-r41-diagnostic-conflict.v2",
        "diagnostic_conflict_graph_sha256": "b" * 64,
        "conflict_families_sha256": "7" * 64,
        "producer_sources_sha256": "a" * 64,
    }
    context.tutorial_signature = online._digest(context.tutorial)
    context.provenance["tutorial_signature"] = context.tutorial_signature
    return context


def test_diagnostic_store_exposes_only_safe_classification(tmp_path):
    context = diagnostic_context(tmp_path)
    store = online.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic.sqlite3",
        storage_mode="ephemeral")
    try:
        view = store.view(store.session())
        assert view["release"]["release_version"] == "r4.1-diagnostic"
        assert view["release"]["pilot_class"] == "internal_diagnostic"
        assert view["release"]["formal_ready"] is False
        assert view["release"]["formal_sample_eligible"] is False
        assert view["release"]["data_persistent"] is False
        assert view["enrollment"]["mode"] == "internal_diagnostic"
        assert view["enrollment"]["formal_sample_eligible"] is False
        assert "内部探索" in view["flow"]["consent_text"]["zh"]
        assert "重启可能丢失" in view["flow"]["consent_text"]["zh"]
        assert store.namespace == "online_diagnostic"
        assert store.cookie_name == online.DIAGNOSTIC_COOKIE
        raw = json.dumps(view, ensure_ascii=False, sort_keys=True)
        assert "behavior_performance_gate_waived" not in raw
        assert "diagnostic_actor_designation_sha256" not in raw
        assert "7" * 64 not in raw
        assert "8" * 64 not in raw
        assert "9" * 64 not in raw
        assert "namespace" not in view["release"]
        for scene in context.scenarios["splits"]["play"]:
            assert scene["fingerprint"] not in raw

        identity = store.deployment_identity()
        assert identity["event"] == "warehouse_r41_diagnostic_deployment_identity"
        assert identity["server_version"] == online.DIAGNOSTIC_SERVER_VERSION
        assert identity["release_version"] == "r4.1-diagnostic"
        assert identity["actor_sha256"] == "a" * 64
        assert identity["package_sha256"] == "8" * 64
        assert identity["manifest_sha256"] == "9" * 64
    finally:
        store.close()


def test_active_diagnostic_view_and_history_do_not_expose_scene_identity(tmp_path):
    context = diagnostic_context(tmp_path)
    private_fingerprints = [
        row["fingerprint"] for row in context.scenarios["splits"]["play"]]
    private_seeds = [str(row["seed"])
                     for row in context.scenarios["splits"]["play"]]
    store = online.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic.sqlite3",
        storage_mode="ephemeral")
    try:
        sid = store.session()
        view = store.view(sid)
        view = store.command(sid, {
            "operation_id": uuid4().hex, "expected_version": view["version"],
            "kind": "start", "mode": "study", "participant_id": "user_01",
            "consent": True,
        })
        view = store.command(sid, {
            "operation_id": uuid4().hex, "expected_version": view["version"],
            "kind": "begin_task1",
        })
        history = store.history(sid)
        serialized = json.dumps(
            {"view": view, "history": history}, ensure_ascii=False,
            sort_keys=True).lower()
        for key in ("fingerprint", "scenario_sha256", "actor_sha256",
                    "manifest_sha256", '"seed"'):
            assert key not in serialized
        for private in (*private_fingerprints, *private_seeds):
            assert private not in serialized
    finally:
        store.close()


@pytest.mark.parametrize("field,value", [
    ("release_version", "r4.1"),
    ("pilot_class", "internal_pilot"),
    ("formal_sample_eligible", True),
    ("human_explanation_effect_validated", True),
    ("behavior_performance_gate_passed", True),
    ("behavior_performance_gate_waived", False),
    ("data_persistent", True),
])
def test_diagnostic_store_rejects_misclassified_release(tmp_path, field, value):
    context = diagnostic_context(tmp_path)
    context.release[field] = value
    with pytest.raises(ValueError, match="classification is incomplete"):
        online.OnlineAlignmentStudyStore(
            context, database=tmp_path / f"bad-{field}.sqlite3",
            storage_mode="ephemeral")
    context.close()


def test_diagnostic_release_label_requires_exact_context_version(tmp_path):
    context = diagnostic_context(tmp_path)
    context.provenance["version"] = "warehouse-r41-diagnostic-online-release.forged"
    with pytest.raises(ValueError, match="exact context version"):
        online.OnlineAlignmentStudyStore(
            context, database=tmp_path / "wrong-context.sqlite3",
            storage_mode="ephemeral")
    context.close()


def test_diagnostic_server_rejects_legacy_tutorial_contract(tmp_path):
    context = diagnostic_context(tmp_path)
    context.tutorial["bindings"]["diagnostic_contract_version"] = (
        "warehouse-r41-diagnostic-conflict.v1")
    context.tutorial_signature = online._digest(context.tutorial)
    context.provenance["tutorial_signature"] = context.tutorial_signature
    with pytest.raises(ValueError, match="hash-bound neutral"):
        online.OnlineAlignmentStudyStore(
            context, database=tmp_path / "legacy-contract.sqlite3",
            storage_mode="ephemeral")
    context.close()


def test_diagnostic_assets_and_storage_are_explicit(tmp_path):
    context = diagnostic_context(tmp_path)
    store = online.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic.sqlite3",
        storage_mode="ephemeral")
    try:
        javascript = store.web_assets["app.js"].decode("utf-8")
        assert "内部诊断实验 · r4.1-diagnostic" in javascript
        assert "Internal diagnostic · r4.1-diagnostic" in javascript
        assert "内部探索" in javascript
        assert "380" in javascript
        assert 'FRONTEND_VERSION="warehouse-alignment-online.r4.1-diagnostic"' in javascript
        assert "warehouse-alignment-online.r4_1_diagnostic.pending" in javascript
        assert store.web_assets["index.html"].count(b"data-question=") == 6
    finally:
        store.close()

    context = diagnostic_context(tmp_path)
    with pytest.raises(ValueError, match="explicitly ephemeral"):
        online.OnlineAlignmentStudyStore(
            context, database=tmp_path / "auto.sqlite3", storage_mode="auto")
    context.close()


def test_diagnostic_module_is_allowed_and_loaded(monkeypatch):
    expected = object()

    class Module:
        @staticmethod
        def load_online_release(**kwargs):
            assert kwargs["expected_package_sha256"] == "1" * 64
            assert kwargs["expected_manifest_sha256"] == "2" * 64
            return expected

    monkeypatch.setattr(online.importlib, "import_module", lambda name: (
        Module if name == online.R41_DIAGNOSTIC_RELEASE_MODULE else None))
    assert online.load_online_context(
        expected_package_sha256="1" * 64,
        expected_manifest_sha256="2" * 64,
        package_path=Path("release.zip"),
        release_module=online.R41_DIAGNOSTIC_RELEASE_MODULE,
    ) is expected


def test_startup_identity_hashes_secret_without_exposing_it(tmp_path):
    context = diagnostic_context(tmp_path)
    store = online.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic.sqlite3",
        storage_mode="ephemeral")
    secret = tmp_path / "warehouse_alignment_release.b64"
    secret.write_bytes(b"diagnostic-secret-file-bytes\n")
    try:
        identity = online._startup_identity(store, base64_path=secret)
        assert identity["secret_file_sha256"] == sha256(secret.read_bytes()).hexdigest()
        assert "behavior_performance_gate_waived" not in identity
    finally:
        store.close()


def test_startup_identity_rejects_oversized_or_nonregular_secret(tmp_path):
    context = diagnostic_context(tmp_path)
    store = online.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic.sqlite3",
        storage_mode="ephemeral")
    try:
        oversized = tmp_path / "oversized.b64"
        oversized.write_bytes(b"A" * (online.MAX_SECRET_FILE_BYTES + 1))
        with pytest.raises(ValueError, match="safe limit"):
            online._startup_identity(store, base64_path=oversized)
        directory = tmp_path / "secret-directory"
        directory.mkdir()
        with pytest.raises(ValueError, match="safe limit"):
            online._startup_identity(store, base64_path=directory)
    finally:
        store.close()


def test_diagnostic_health_exposes_classification_without_operator_identity(
        tmp_path):
    context = diagnostic_context(tmp_path)
    store = online.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic.sqlite3",
        storage_mode="ephemeral")
    server = online.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        online.handler_class(store, public_origin="https://study.test"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/health")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload == {
            "status": "ok", "service": online.SERVICE_FAMILY,
            "version": online.DIAGNOSTIC_SERVER_VERSION,
            "release_version": "r4.1-diagnostic",
            "pilot_class": "internal_diagnostic",
            "data_persistent": False, "formal_ready": False,
            "formal_sample_eligible": False,
        }
        serialized = json.dumps(payload, sort_keys=True)
        for private in ("actor_sha256", "package_sha256", "manifest_sha256",
                        "scene_fingerprints", "seed"):
            assert private not in serialized
        connection.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(2); store.close()


def test_participant_answer_strips_operator_scene_identity_lines():
    row = {
        "id": "q1", "status": "complete", "frame": 4,
        "question": "为什么？", "answer": "机器人2在靠近取货点。",
        "run_id": "run1", "focus": "executed",
        "evidence_detail": (
            "第4帧动作与物理回放一致。\n"
            "seed: 260991\n"
            "scene fingerprint: " + "a" * 64),
    }
    answer = online._participant_answer(row)
    assert answer["text"] == "机器人2在靠近取货点。"
    assert answer["evidence_detail"] == "第4帧动作与物理回放一致。"


def test_participant_answer_fails_closed_on_scene_identity_in_main_text():
    row = {
        "id": "q1", "status": "complete", "frame": 4,
        "question": "为什么？", "answer": "The seed was 260991.",
        "run_id": "run1", "focus": "executed", "evidence_detail": "",
    }
    answer = online._participant_answer(row)
    assert answer["status"] == "failed"
    assert answer["text"] == ""
