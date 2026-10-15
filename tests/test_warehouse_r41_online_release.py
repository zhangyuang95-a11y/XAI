from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from ui import warehouse_alignment_online_release as old_release
from ui import warehouse_alignment_online_server as server
from ui import warehouse_alignment_r41_online_release as release


ZERO = "0" * 64


def _artifacts():
    return {
        "actor": b"actor",
        "protocol": b"{}\n",
        "play_scenarios": b"{}\n",
        "program": b"{}\n",
        "question_bank": b"{}\n",
        "tutorial": b"{}\n",
    }


def _identities():
    ids = ["tutorial_0000", *(f"play_{index:04d}" for index in range(6))]
    return {
        "actor_sha256": ZERO, "protocol_sha256": ZERO,
        "program_sha256": ZERO, "parent_runtime_signature": ZERO,
        "parent_explainer_signature": ZERO,
        "parent_question_bank_signature": ZERO,
        "question_bank_private_items_sha256": ZERO,
        "question_bank_public_items_sha256": ZERO,
        "portable_scenarios_sha256": ZERO,
        "play_scene_count": 7, "play_scene_ids": ids,
        "tutorial_signature": ZERO, "tutorial_scene_id": ids[0],
        "tutorial_scene_fingerprint": ZERO,
        "tutorial_successor_state_sha256": ZERO,
        "tutorial_snapshot_sha256": ZERO,
        "conflict_contract_sha256": ZERO,
        "conflict_graph_sha256": ZERO,
        "scene_manifest_content_sha256": ZERO,
        "uses_final_actor": False,
    }


def _manifest(artifacts=None):
    artifacts = artifacts or _artifacts()
    return {
        "version": release.VERSION, "status": release.STATUS,
        "test_fixture": False, "formal_ready": False,
        "parent": {
            "version": "warehouse-r41-production-admission.v1",
            "status": "admitted_internal_pilot",
            "production_admission_sha256": ZERO,
            "training_ledger_sha256": ZERO,
            "dual_evaluation_sha256": ZERO,
            "corrected_six_partner_audit_sha256": ZERO,
            "conflict_manifest_file_sha256": ZERO,
            "conflict_manifest_content_sha256": ZERO,
            "conflict_validation_sha256": ZERO,
            "dynamic_selection_report_sha256": ZERO,
            "selected_scenes_file_sha256": ZERO,
            "final_rcpd_report_sha256": ZERO,
            "explanation_audit_sha256": ZERO,
            "question_bank_report_sha256": ZERO,
            "tutorial_sha256": ZERO,
        },
        "artifacts": release._artifact_records(artifacts),
        "identities": _identities(),
        "sources": {"release": release.release_sources()},
        "analysis": old_release._analysis_protocol(),
        "release": old_release._release_projection(),
    }


def _package(tmp_path, manifest, artifacts):
    raw = release._archive_bytes(manifest, artifacts)
    path = tmp_path / "r41.zip"
    path.write_bytes(raw)
    manifest_sha = sha256((release.canonical(manifest) + "\n").encode()).hexdigest()
    return path, sha256(raw).hexdigest(), manifest_sha


def test_r41_archive_has_exact_tutorial_member_hash_size_and_source_binding(tmp_path):
    artifacts = _artifacts()
    manifest = _manifest(artifacts)
    path, package_sha, manifest_sha = _package(tmp_path, manifest, artifacts)
    inspected = release.inspect_online_release(
        package_path=path, expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha,
    )
    assert set(inspected["artifacts"]) == set(release.ARTIFACT_PATHS)
    assert inspected["artifacts"]["tutorial"]["path"] == "artifacts/tutorial.json"
    assert inspected["artifacts"]["tutorial"]["size"] == len(artifacts["tutorial"])
    assert inspected["sources"]["release"] == release.release_sources()
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == release.ARCHIVE_WHITELIST


def test_r41_archive_rejects_missing_tutorial_and_extra_members(tmp_path):
    artifacts = _artifacts()
    missing = dict(artifacts)
    missing.pop("tutorial")
    with pytest.raises(ValueError, match="Exact r4.1 portable artifact"):
        release._artifact_records(missing)

    manifest = _manifest(artifacts)
    path, _, manifest_sha = _package(tmp_path, manifest, artifacts)
    output = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(output, "w") as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr("artifacts/unbound.json", b"{}")
    bad = tmp_path / "extra.zip"
    bad.write_bytes(output.getvalue())
    with pytest.raises(ValueError, match="whitelist"):
        release.inspect_online_release(
            package_path=bad,
            expected_package_sha256=sha256(output.getvalue()).hexdigest(),
            expected_manifest_sha256=manifest_sha,
        )


def test_portable_scenes_use_only_admitted_tutorial_and_dynamic_x_y():
    bindings = {
        "actor_sha256": "a" * 64,
        "conflict_manifest_file_sha256": "b" * 64,
        "conflict_manifest_content_sha256": "c" * 64,
        "conflict_contract_sha256": "d" * 64,
        "conflict_graph_sha256": "e" * 64,
    }
    scenes = [
        {"id": f"dynamic-{index}", "fingerprint": f"{index + 1:x}" * 64}
        for index in range(7)
    ]
    selection = {
        "version": "warehouse-r41-conflict-dynamic-play-selection.v1",
        "status": "accepted_final_dynamic_selection",
        "release_eligible": True,
        "actor_sha256": bindings["actor_sha256"],
        "source_manifest_file_sha256": bindings[
            "conflict_manifest_file_sha256"],
        "source_manifest_content_sha256": bindings[
            "conflict_manifest_content_sha256"],
        "contract_sha256": bindings["conflict_contract_sha256"],
        "conflict_graph_sha256": bindings["conflict_graph_sha256"],
        "tutorial": scenes[0], "X": scenes[1:4], "Y": scenes[4:],
    }
    projected = release._scenarios_from_admission(selection, bindings)
    assert [row["id"] for row in projected["splits"]["play"]] == [
        row["id"] for row in scenes]
    assert projected["splits"]["tutorial"] == [scenes[0]]
    changed = deepcopy(selection)
    changed["version"] = "warehouse-r41-conflict-scene-manifest.v1"
    with pytest.raises(ValueError, match="dynamic scene selection"):
        release._scenarios_from_admission(changed, bindings)


def test_loader_constructs_r41_runtime_and_exposes_bound_tutorial(monkeypatch, tmp_path):
    artifacts = {
        "actor": b"actor", "protocol": b"{}", "play_scenarios": b"{}",
        "program": b"program", "question_bank": b"{}",
        "tutorial": b'{"uses_final_actor":false}',
    }
    manifest = _manifest(artifacts)
    identities = manifest["identities"]
    identities["actor_sha256"] = sha256(artifacts["actor"]).hexdigest()
    identities["program_sha256"] = sha256(artifacts["program"]).hexdigest()
    identities["protocol_sha256"] = release.digest({})
    identities["parent_runtime_signature"] = "1" * 64
    identities["parent_explainer_signature"] = "2" * 64
    identities["tutorial_signature"] = release.digest({"uses_final_actor": False})
    manifest["artifacts"] = release._artifact_records(artifacts)

    class Runtime:
        def __init__(self, *args, **kwargs):
            assert kwargs["allow_test_fixture"] is False
            self.actor_sha256 = identities["actor_sha256"]
            self.protocol_sha256 = identities["protocol_sha256"]
            self.signature = identities["parent_runtime_signature"]
        def verify_binding(self): return self.signature
        def environment(self, scene):
            return SimpleNamespace(state=SimpleNamespace(frame=0))

    class Explainer:
        def __init__(self, *args, **kwargs):
            assert type(kwargs["runtime"]) is Runtime
            self.program_sha256 = identities["program_sha256"]
            self.signature = identities["parent_explainer_signature"]
        def _assert_current(self, runtime): assert type(runtime) is Runtime

    bank = SimpleNamespace(signature="bank", public_items=lambda: [])
    scene = {"id": identities["tutorial_scene_id"]}
    monkeypatch.setattr(release, "R41OnlineAlignmentRuntime", Runtime)
    monkeypatch.setattr(release, "R41OnlineAlignmentExplainer", Explainer)
    monkeypatch.setattr(release, "_read_archive", lambda *args, **kwargs: (manifest, artifacts))
    monkeypatch.setattr(old_release, "_package_bytes", lambda **kwargs: b"package")
    monkeypatch.setattr(release, "_validate_scenarios",
                        lambda *args: ([scene] * 7, scene))
    monkeypatch.setattr(old_release, "_portable_bank", lambda *args, **kwargs: bank)
    monkeypatch.setattr(release, "validate_neutral_tutorial",
        lambda *args, **kwargs: {"tutorial_signature": identities["tutorial_signature"]})
    monkeypatch.setattr(old_release, "_validate_source_map",
                        lambda values, label: dict(values))
    monkeypatch.setattr(release, "release_sources",
                        lambda: dict(manifest["sources"]["release"]))

    context = release.load_online_release(
        package_path=tmp_path / "ignored",
        expected_package_sha256=ZERO, expected_manifest_sha256=ZERO,
    )
    try:
        assert type(context.runtime) is Runtime
        assert context.tutorial == {"uses_final_actor": False}
        assert context.tutorial_signature == identities["tutorial_signature"]
        assert context.provenance["tutorial_uses_final_actor"] is False
    finally:
        context.close()


def test_build_fails_closed_without_exact_admitted_component_set(tmp_path):
    package, encoded = tmp_path / "release.zip", tmp_path / "release.b64"
    with pytest.raises(ValueError, match="Exact admitted"):
        release.assemble_from_admitted_components(
            production_admission_path=tmp_path / "missing-admission.json",
            expected_production_admission_sha256=ZERO,
            components={},
            output_package=package, output_base64=encoded)
    assert not package.exists() and not encoded.exists()


def test_online_server_explicitly_selects_r41_and_keeps_historical_loader(monkeypatch):
    calls = []
    module = SimpleNamespace(load_online_release=lambda **kwargs: calls.append(kwargs) or "ok")
    monkeypatch.setattr(server.importlib, "import_module",
                        lambda name: calls.append(name) or module)
    assert server.load_online_context(
        expected_package_sha256=ZERO, expected_manifest_sha256=ZERO,
        package_path="release.zip") == "ok"
    assert calls[0] == server.DEFAULT_RELEASE_MODULE
    calls.clear()
    assert server.load_online_context(
        expected_package_sha256=ZERO, expected_manifest_sha256=ZERO,
        package_path="release.zip", release_module=server.R41_RELEASE_MODULE) == "ok"
    assert calls[0] == server.R41_RELEASE_MODULE
    with pytest.raises(ValueError, match="unsupported"):
        server.load_online_context(
            expected_package_sha256=ZERO, expected_manifest_sha256=ZERO,
            package_path="release.zip", release_module="ui.arbitrary")


def test_render_style_cli_passes_explicit_r41_loader_before_store(monkeypatch, tmp_path):
    calls = []

    class StopAfterLoad(Exception):
        pass

    def load(**kwargs):
        calls.append(kwargs)
        raise StopAfterLoad

    monkeypatch.setattr(server, "load_online_context", load)
    monkeypatch.setattr(
        server, "OnlineAlignmentStudyStore",
        lambda *args, **kwargs: pytest.fail("store constructed after failed load"),
    )
    with pytest.raises(StopAfterLoad):
        server.main([
            "--base64", str(tmp_path / "mounted-secret.b64"),
            "--expected-package-sha256", "a" * 64,
            "--expected-manifest-sha256", "b" * 64,
            "--release-module", server.R41_RELEASE_MODULE,
            "--public-origin", "https://policylens-warehouse-study.onrender.com",
        ])
    assert calls == [{
        "expected_package_sha256": "a" * 64,
        "expected_manifest_sha256": "b" * 64,
        "package_path": None,
        "base64_path": tmp_path / "mounted-secret.b64",
        "release_module": server.R41_RELEASE_MODULE,
    }]


def test_r41_release_import_stays_torch_and_training_free():
    code = """
import sys
import ui.warehouse_alignment_r41_online_release
assert 'torch' not in sys.modules
assert not any(name == 'backend.training' or name.startswith('backend.training.')
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=release.ROOT, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
