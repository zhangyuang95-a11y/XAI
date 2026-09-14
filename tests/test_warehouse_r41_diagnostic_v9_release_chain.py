"""Synthetic v9 release-chain tests; no real outer/final artifact is read."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import zipfile

import pytest

from backend.training import warehouse_r41_diagnostic_admission_v9 as admission
from backend.training import warehouse_r41_diagnostic_release_receipt_v9 as receipt
from backend.training.warehouse_native_common import canonical, digest, file_hash
from scripts import preflight_warehouse_r41_diagnostic_render_v9 as preflight
from ui import warehouse_alignment_online_release as portable
from ui import warehouse_alignment_r41_diagnostic_release_v9 as release


def test_admission_authenticates_v12_outer_result():
    assert admission.outer_api.VERSION.startswith(
        "warehouse-r41-diagnostic-rcpd-v12-")


def test_admission_authenticates_promoted_v11_closeout_and_rows(
        tmp_path, monkeypatch):
    rows = tmp_path / "promoted-v11-rows.npz"
    closeout_path = tmp_path / "promoted-v11-closeout.json"
    rows.write_bytes(b"promoted rows")
    closeout_path.write_bytes(b"promoted closeout")
    hashes = {
        "promoted_v11_rows": file_hash(rows),
        "promoted_v11_closeout": file_hash(closeout_path),
    }
    lock_bindings = {
        "promoted_v11_rows_sha256": hashes["promoted_v11_rows"],
        "promoted_v11_closeout_sha256": hashes["promoted_v11_closeout"],
    }
    closeout = {
        "content_sha256": "1" * 64,
        "consumed_outer": {
            "rows_sha256": hashes["promoted_v11_rows"],
            "rows_semantic_sha256": "2" * 64,
        },
    }
    calls = []
    monkeypatch.setattr(
        admission.promoted_closeout_api, "read_saved_closeout",
        lambda path, **kwargs: calls.append((Path(path), kwargs)) or closeout)
    authenticated, semantic = admission._authenticate_promoted_v11(
        {"promoted_v11_rows": rows,
         "promoted_v11_closeout": closeout_path},
        hashes, lock_bindings, permanent_registry=tmp_path)
    assert authenticated is closeout
    assert semantic == "2" * 64
    assert calls == [(closeout_path, {
        "expected_closeout_sha256": hashes["promoted_v11_closeout"],
        "permanent_attempt_registry": tmp_path,
    })]

    changed = dict(lock_bindings)
    changed["promoted_v11_rows_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="candidate lock"):
        admission._authenticate_promoted_v11(
            {"promoted_v11_rows": rows,
             "promoted_v11_closeout": closeout_path},
            hashes, changed, permanent_registry=tmp_path)


def _artifacts():
    return {
        "actor": b"actor",
        "protocol": b"{}\n",
        "runtime_manifest": b"{}\n",
        "program": b"compact",
        "question_bank": b"{}\n",
        "tutorial": b"{}\n",
    }


def _manifest(artifacts=None):
    artifacts = artifacts or _artifacts()
    h = lambda raw: sha256(raw).hexdigest()
    identities = {name: (str(index + 1) * 64)[:64] for index, name in enumerate((
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
    ))}
    identities.update({
        "actor_sha256": h(artifacts["actor"]),
        "protocol_file_sha256": h(artifacts["protocol"]),
        "runtime_manifest_file_sha256": h(artifacts["runtime_manifest"]),
        "compact_program_sha256": h(artifacts["program"]),
        "question_bank_sha256": h(artifacts["question_bank"]),
        "tutorial_sha256": h(artifacts["tutorial"]),
        "release_sources_sha256": digest(release.release_sources()),
    })
    scenes = [{
        "id": "tutorial" if index == 0 else f"scene-{index}",
        "family_id": "tutorial" if index == 0 else f"family-{index}",
        "fingerprint": f"{index + 1:064x}",
        "scene_sha256": f"{index + 10:064x}",
    } for index in range(7)]
    return {
        "version": release.VERSION, "status": release.STATUS,
        "test_fixture": False, "pilot_class": release.PILOT_CLASS,
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False,
        "parent": {
            "version": admission.VERSION, "status": admission.STATUS,
            "admission_sha256": "a" * 64,
            "admission_content_sha256": "b" * 64,
            "bindings_sha256": "c" * 64, "gates_sha256": "d" * 64,
        },
        "artifacts": release._artifact_records(artifacts),
        "identities": identities, "play_scenes": scenes,
        "sources": {"release": release.release_sources()},
        "analysis": portable._analysis_protocol(),
        "release": release.release_projection(ready=True),
    }


def _raw_archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=release.ARCHIVE_COMPRESSION,
                         compresslevel=release.ARCHIVE_COMPRESSLEVEL) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = release.ARCHIVE_COMPRESSION
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, raw, compress_type=release.ARCHIVE_COMPRESSION,
                             compresslevel=release.ARCHIVE_COMPRESSLEVEL)
    return stream.getvalue()


def test_admission_separates_full_and_portable_manifests_and_all_hard_gates():
    assert "source_manifest" in admission.ARTIFACT_NAMES
    assert "source_manifest_validation" in admission.ARTIFACT_NAMES
    assert "runtime_manifest" in admission.ARTIFACT_NAMES
    assert "selected_scenes" in admission.ARTIFACT_NAMES
    assert "question_bank_report" in admission.ARTIFACT_NAMES
    assert {"promoted_v11_rows", "promoted_v11_closeout"}.issubset(
        admission.ARTIFACT_NAMES)
    assert {"final_anchor", "final_material", "final_rows", "final_audit"}.issubset(
        admission.ARTIFACT_NAMES)
    assert admission.GATE_NAMES == (
        "actor_designation", "runtime_action_authority", "portable_manifest",
        "six_high_conflict_scenes", "locked_program",
        "promoted_v11_development_binding", "fresh_outer",
        "protected_final_audit", "compact_program_parity", "question_bank",
        "neutral_tutorial", "participant_ui_source_closure")
    contract = admission.package_contract()
    assert contract["maximum_base64_bytes"] == 960_000
    assert contract["protected_outer_or_final_artifacts_packaged"] is False
    assert set(contract["archive_whitelist"]) == set(release.ARCHIVE_WHITELIST)


def test_admission_rejects_nonofficial_final_materializer_producer_binding():
    _source, official_sources = admission.final_api._official_materializer_binding()
    anchor = {
        "attempt_key": "a" * 64,
        "content_sha256": "b" * 64,
        "bindings": {
            "final_materializer_source_closure_sha256": digest(official_sources),
        },
    }
    scenes = [{
        "id": f"final-{index}", "seed": 900_000 + index,
        "family_id": f"family-{index % 6}",
        "fingerprint": sha256(f"final-{index}".encode()).hexdigest(),
    } for index in range(admission.audit_api.FINAL_SCENE_COUNT)]
    material = {
        "version": admission.final_api.MATERIAL_VERSION,
        "status": admission.final_api.MATERIAL_STATUS,
        "claim": {
            "attempt_key": anchor["attempt_key"],
            "attempt_anchor_content_sha256": anchor["content_sha256"],
        },
        "scenes": scenes,
        "selection": {
            "whole_scene_selection": True,
            "scene_count": admission.audit_api.FINAL_SCENE_COUNT,
            "program_access": False,
            "program_predictions_access": False,
            "actor_outputs_access": False,
            "action_labels_access": False,
            "salt_access_after_permanent_claim": True,
            "candidate_adaptation": False,
            "runtime_action_override": False,
        },
        "producer_sources": dict(official_sources),
        "producer_sources_sha256": digest(official_sources),
        "formal_ready": False,
    }
    material["content_sha256"] = digest(material)
    authenticated = admission._authenticate_official_final_materializer(
        anchor, material)
    assert authenticated["source_closure_sha256"] == digest(official_sources)

    copied_sources = {"copied/final_materializer.py": "c" * 64}
    changed = deepcopy(material)
    changed["producer_sources"] = copied_sources
    changed["producer_sources_sha256"] = digest(copied_sources)
    changed["content_sha256"] = digest({
        key: value for key, value in changed.items() if key != "content_sha256"
    })
    changed_anchor = deepcopy(anchor)
    changed_anchor["bindings"][
        "final_materializer_source_closure_sha256"] = digest(copied_sources)
    with pytest.raises(ValueError, match="official materializer"):
        admission._authenticate_official_final_materializer(
            changed_anchor, changed)


def test_archive_roundtrip_is_deterministic_strict_and_under_960kb():
    artifacts = _artifacts(); manifest = _manifest(artifacts)
    first = release._archive_bytes(manifest, artifacts)
    second = release._archive_bytes(manifest, artifacts)
    assert first == second
    assert len(first) <= release.MAX_PACKAGE_BYTES
    assert 4 * ((len(first) + 2) // 3) + 1 <= release.MAX_BASE64_BYTES
    manifest_raw = (canonical(manifest) + "\n").encode()
    loaded, rows = release._read_archive(
        first, expected_package_sha256=sha256(first).hexdigest(),
        expected_manifest_sha256=sha256(manifest_raw).hexdigest())
    assert loaded == manifest and rows == artifacts


def test_archive_rejects_traversal_duplicate_and_nonwhitelist_members():
    artifacts = _artifacts(); manifest = _manifest(artifacts)
    manifest_raw = (canonical(manifest) + "\n").encode()
    normal = [(release.MANIFEST_NAME, manifest_raw), *[
        (release.ARTIFACT_PATHS[name], raw) for name, raw in artifacts.items()]]
    for members in (
        [("../manifest.json", manifest_raw), *normal[1:]],
        [*normal, (release.MANIFEST_NAME, manifest_raw)],
        [*normal, ("artifacts/private.sqlite3", b"private")],
    ):
        raw = _raw_archive(members)
        with pytest.raises(ValueError, match="member set|unsafe"):
            release._read_archive(
                raw, expected_package_sha256=sha256(raw).hexdigest(),
                expected_manifest_sha256=sha256(manifest_raw).hexdigest())


def test_release_rejects_credentials_participant_data_and_absolute_paths():
    with pytest.raises(ValueError, match="Sensitive"):
        release._reject_sensitive({"deepseek_api_key": "sk-" + "x" * 24}, "x")
    with pytest.raises(ValueError, match="Sensitive"):
        release._reject_sensitive({"participant_id": "person_01"}, "x")
    with pytest.raises(ValueError, match="absolute"):
        release._reject_sensitive({"path": "/Users/research/private.json"}, "x")


def test_base64_reader_enforces_960000_byte_ceiling(tmp_path):
    path = tmp_path / "too-large.b64"
    path.write_bytes(b"A" * (release.MAX_BASE64_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        release._package_bytes(base64_path=path)


def test_assembler_and_receipt_stop_before_writing_when_admission_fails(
        tmp_path, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("outer gate failed")
    monkeypatch.setattr(admission, "read_saved_admission", fail)
    package = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="outer gate failed"):
        release.assemble_from_admitted_components(
            diagnostic_admission_path=tmp_path / "admission.json",
            expected_diagnostic_admission_sha256="a" * 64,
            components={}, outer_permanent_registry=tmp_path,
            final_permanent_registry=tmp_path,
            promoted_v11_permanent_registry=tmp_path,
            output_package=package)
    assert not package.exists()
    admission_path = tmp_path / "admission.json"
    package_path = tmp_path / "candidate.zip"
    base64_path = tmp_path / "candidate.b64"
    for path, raw in ((admission_path, b"{}\n"), (package_path, b"zip"),
                      (base64_path, b"emlw\n")):
        path.write_bytes(raw); path.chmod(0o600)
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="outer gate failed"):
        receipt.build_receipt(
            admission_path=admission_path,
            expected_admission_sha256=sha256(admission_path.read_bytes()).hexdigest(),
            components={}, outer_permanent_registry=tmp_path,
            final_permanent_registry=tmp_path,
            promoted_v11_permanent_registry=tmp_path,
            package_path=package_path,
            base64_path=base64_path, output=receipt_path)
    assert not receipt_path.exists()
    assert len(calls) == 2
    assert all(call["promoted_v11_permanent_registry"] == tmp_path
               for call in calls)


def test_preflight_propagates_promoted_v11_permanent_registry(
        tmp_path, monkeypatch):
    package = tmp_path / "candidate.zip"
    encoded = tmp_path / "candidate.b64"
    receipt_path = tmp_path / "receipt.json"
    for path, raw in ((package, b"zip"), (encoded, b"emlw\n"),
                      (receipt_path, b"receipt")):
        path.write_bytes(raw)
        path.chmod(0o600)
    seen = []

    def stop(path, **kwargs):
        seen.append((Path(path), kwargs))
        raise ValueError("stop after receipt authentication boundary")

    monkeypatch.setattr(preflight.receipt_api, "read_saved_receipt", stop)
    with pytest.raises(ValueError, match="receipt authentication boundary"):
        preflight.preflight(
            admission_path=tmp_path / "admission.json",
            expected_admission_sha256="a" * 64, components={},
            outer_permanent_registry=tmp_path,
            final_permanent_registry=tmp_path,
            promoted_v11_permanent_registry=tmp_path,
            package_path=package, base64_path=encoded,
            receipt_path=receipt_path,
            expected_receipt_sha256=file_hash(receipt_path),
            render_yaml=tmp_path / "render.yaml", check_clean_checkout=False)
    assert seen[0][1]["promoted_v11_permanent_registry"] == tmp_path


def _render_yaml(package="a" * 64, manifest="b" * 64):
    return f"""services:
  - type: web
    name: {preflight.SERVICE_NAME}
    runtime: python
    plan: free
    numInstances: 1
    startCommand: {preflight.START_COMMAND}
    healthCheckPath: /health
    autoDeployTrigger: off
    envVars:
      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256
        value: {package}
      - key: WAREHOUSE_RELEASE_MANIFEST_SHA256
        value: {manifest}
      - key: WAREHOUSE_ONLINE_DATABASE
        value: /tmp/warehouse_r41_diagnostic_v9.sqlite3
      - key: WAREHOUSE_STORAGE_MODE
        value: ephemeral
      - key: WAREHOUSE_PUBLIC_ORIGIN
        value: {preflight.PUBLIC_ORIGIN}
"""


def test_render_preflight_is_v9_manual_free_ephemeral_and_hash_bound(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text(_render_yaml(), encoding="utf-8")
    result = preflight.check_render_yaml(
        path, package_sha256="a" * 64, manifest_sha256="b" * 64)
    assert result["release_module"] == release.__name__
    assert result["secret_path"] == preflight.SECRET_PATH
    assert result["auto_deploy"] is False
    assert result["data_persistent"] is False
    path.write_text(_render_yaml(package="c" * 64), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        preflight.check_render_yaml(
            path, package_sha256="a" * 64, manifest_sha256="b" * 64)


def test_public_metadata_never_claims_formal_or_persistent_readiness():
    for ready in (False, True):
        view = release.release_projection(ready=ready)
        assert view["release_version"] == "r4.1-diagnostic"
        assert view["pilot_class"] == "internal_diagnostic"
        assert view["formal_ready"] is False
        assert view["formal_sample_eligible"] is False
        assert view["data_persistent"] is False
        assert view["runtime_action_override"] is False
        assert view["animation_duration_ms"] == 380
