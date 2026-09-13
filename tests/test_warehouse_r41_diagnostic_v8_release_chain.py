from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from backend.training import warehouse_r41_diagnostic_admission_v6 as admission
from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as audit
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as frozen_manifest
from backend.training import warehouse_r41_diagnostic_frozen_publication_v2 as frozen_publication
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as holdout
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as weights
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as rcpd
from backend.training import warehouse_r41_diagnostic_release_receipt_v6 as receipt
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS, MODEL_KIND, VERSION as BOOSTED_VERSION,
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    GROUPS, assemble_public_tree_program_v8,
)
from ui import warehouse_alignment_r41_diagnostic_release_v8 as release
from scripts import preflight_warehouse_r41_diagnostic_render_v6 as preflight
from ui import warehouse_alignment_online_server as server


def test_active_release_chain_requires_cycle_free_designation_v2():
    expected = "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
    assert admission.designation_api.VERSION == (
        "warehouse-r41-diagnostic-actor-designation.v2")
    assert admission.designation_binding.EXPECTED_DESIGNATION_SHA256 == expected
    assert receipt.designation_api.VERSION == admission.designation_api.VERSION
    assert receipt.designation_binding.EXPECTED_DESIGNATION_SHA256 == expected
    assert preflight.designation_api.VERSION == admission.designation_api.VERSION
    assert preflight.designation_binding.EXPECTED_DESIGNATION_SHA256 == expected
    assert holdout.EXPECTED_DESIGNATION_SHA256 == expected
    assert release.FIXED_DESIGNATION_SHA256 == expected
    assert admission.publication_api is frozen_publication
    assert frozen_publication.manifest_reader is frozen_manifest
    assert frozen_publication.final_once is admission.final_once_api
    assert frozen_manifest.EXPECTED_MANIFEST_SHA256 == (
        "af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c"
    )
    assert set(admission.RCPD_CANDIDATE_ARTIFACTS) == set(
        admission.final_once_api.CANDIDATE_ARTIFACT_KEYS)
    assert set(admission.RCPD_CANDIDATE_ARTIFACTS.values()) <= set(
        admission.ARTIFACT_NAMES)
    assert admission.RCPD_CANDIDATE_ARTIFACTS[
        "development_expansion.json"] == (
            "final_rcpd_development_expansion_registry")
    assert admission.RCPD_CANDIDATE_ARTIFACTS[
        "development_expansion_report.json"] == (
            "final_rcpd_development_expansion_report")


def test_admission_requires_full_replay_publication_authentication(
        tmp_path, monkeypatch):
    candidate_inputs = admission.RCPD_CANDIDATE_ARTIFACTS
    files = {
        name: tmp_path / name
        for name in ({
            "conflict_manifest", "conflict_validation",
            "diagnostic_contract", "actor",
            *candidate_inputs.values(),
            *admission.FINAL_ONCE_LEDGER_ARTIFACTS,
        })
    }
    hashes = {
        "conflict_manifest": frozen_manifest.EXPECTED_MANIFEST_SHA256,
        "conflict_validation": frozen_publication.EXPECTED_VALIDATION_SHA256,
        "diagnostic_contract": frozen_publication.EXPECTED_CONTRACT_SHA256,
        "actor": admission.FIXED_ACTOR_SHA256,
    }
    for index, name in enumerate(candidate_inputs.values(), 1):
        hashes[name] = format(index, "x") * 64
    for index, name in enumerate(
            admission.FINAL_ONCE_LEDGER_ARTIFACTS, 8):
        hashes[name] = format(index, "x") * 64
    manifest = {
        "content_sha256": frozen_manifest.EXPECTED_MANIFEST_CONTENT_SHA256,
        "splits": {"final_test": [{"id": "final-1"}]},
    }
    validation = {"passed": True}
    contract = {"contract_sha256": "c" * 64}
    runtime = {
        "version": frozen_manifest.VERSION,
        "sources": {"runtime.py": "d" * 64},
        "sources_sha256": "e" * 64,
    }
    phase_receipts = {
        relative: hashes[name]
        for name, relative in admission.FINAL_ONCE_LEDGER_ARTIFACTS.items()
        if name not in {
            "final_once_attempt_started", "final_once_attempt_completed"
        }
    }
    candidate_artifacts = {
        name: hashes[candidate_inputs[name]] for name in sorted(candidate_inputs)
    }
    successful_final = {
        "version": frozen_publication.SUCCESSFUL_FINAL_VERSION,
        "status": "authenticated_completed_passed",
        "final_once_version": admission.final_once_api.VERSION,
        "campaign_key": admission.final_once_api.campaign_key(),
        "campaign_identity": admission.final_once_api._campaign_identity(),
        "campaign_identity_sha256": digest(
            admission.final_once_api._campaign_identity()),
        "candidate_identity_sha256": digest(
            admission.final_once_api._campaign_identity()),
        "attempt_started_sha256": hashes["final_once_attempt_started"],
        "attempt_completed_sha256": hashes["final_once_attempt_completed"],
        "permanent_anchor_sha256": "f" * 64,
        "candidate_authenticated_sha256": hashes[
            "final_once_candidate_authenticated"],
        "candidate_artifacts": candidate_artifacts,
        "candidate_artifacts_sha256": digest(candidate_artifacts),
        "phase_receipts": phase_receipts,
        "phase_receipts_sha256": digest(phase_receipts),
        "actor_file_sha256": hashes["actor"],
        "manifest_file_sha256": hashes["conflict_manifest"],
        "program_file_sha256": hashes["final_rcpd_program"],
        "retry_allowed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    authentication = {
        "version": frozen_publication.AUTHENTICATION_VERSION,
        "status": "authenticated_exact_publication_with_full_workload_replay",
        "replay_scope": "all",
        "final_test_replayed_for_authentication": True,
        "final_test_rows_used_for_fit_or_selection": False,
        "final_labels_used": False,
        "final_test_scene_count": 1,
        "manifest_file_sha256": hashes["conflict_manifest"],
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_semantic_sha256": digest(manifest),
        "validation_file_sha256": hashes["conflict_validation"],
        "validation_semantic_sha256": digest(validation),
        "contract_file_sha256": hashes["diagnostic_contract"],
        "contract_semantic_sha256": digest(contract),
        "current_runtime_sources_sha256": runtime["sources_sha256"],
        "successful_final_capability_sha256": digest(successful_final),
        "final_once_version": successful_final["final_once_version"],
        "final_once_campaign_key": successful_final["campaign_key"],
        "final_once_campaign_identity_sha256": successful_final[
            "campaign_identity_sha256"],
        "final_once_candidate_identity_sha256": successful_final[
            "candidate_identity_sha256"],
        "final_once_attempt_started_sha256": successful_final[
            "attempt_started_sha256"],
        "final_once_attempt_completed_sha256": successful_final[
            "attempt_completed_sha256"],
        "final_once_permanent_anchor_sha256": successful_final[
            "permanent_anchor_sha256"],
        "final_once_candidate_authenticated_sha256": successful_final[
            "candidate_authenticated_sha256"],
        "final_once_candidate_artifacts_sha256": successful_final[
            "candidate_artifacts_sha256"],
        "final_once_phase_receipts_sha256": successful_final[
            "phase_receipts_sha256"],
        "actor_file_sha256": successful_final["actor_file_sha256"],
        "program_file_sha256": successful_final["program_file_sha256"],
    }
    publication = {
        "manifest": manifest,
        "validation": validation,
        "contract": contract,
        "current_runtime": runtime,
        "authentication_receipt": authentication,
        "authentication_sha256": digest(authentication),
        "successful_final": successful_final,
        "publication_identity_sha256": digest({
            "manifest": hashes["conflict_manifest"],
            "validation": hashes["conflict_validation"],
            "contract": hashes["diagnostic_contract"],
            "current_runtime_sources_sha256": runtime["sources_sha256"],
        }),
    }
    calls = []

    def successful_capability(**kwargs):
        calls.append(("capability", kwargs))
        return deepcopy(successful_final)

    def read_saved_publication(**kwargs):
        calls.append(("publication", kwargs))
        return deepcopy(publication)

    monkeypatch.setattr(
        admission.publication_api, "_successful_final_capability",
        successful_capability,
    )
    monkeypatch.setattr(
        admission.publication_api, "read_saved_publication",
        read_saved_publication,
    )
    monkeypatch.setattr(
        admission.manifest_api, "runtime_sources", lambda: deepcopy(runtime),
    )
    checked = admission._validate_publication(files, hashes)
    assert checked["authentication_sha256"] == digest(authentication)
    common = {
        "manifest_path": files["conflict_manifest"],
        "actor_path": files["actor"],
        "program_path": files["final_rcpd_program"],
    }
    assert calls == [
        ("capability", common),
        ("publication", {
            **common,
            "validation_path": files["conflict_validation"],
            "contract_path": files["diagnostic_contract"],
        }),
    ]

    publication["authentication_receipt"]["replay_scope"] = "development"
    publication["authentication_sha256"] = digest(
        publication["authentication_receipt"])
    with pytest.raises(ValueError, match="publication differs"):
        admission._validate_publication(files, hashes)


def test_release_receipt_rejects_any_non_v2_designation_bytes(tmp_path):
    admission_path = tmp_path / "admission.json"
    designation_path = tmp_path / "designation.json"
    admission_path.write_text("{}\n", encoding="utf-8")
    designation_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="designation bytes differ"):
        receipt._validate_inputs(
            admission_path=admission_path,
            designation_path=designation_path,
            package_path=tmp_path / "package.zip",
            base64_path=tmp_path / "package.b64",
            rollback_path=tmp_path / "rollback.json",
            expected_admission_sha256=file_hash(admission_path),
            expected_designation_sha256=file_hash(designation_path),
            expected_rollback_sha256="0" * 64,
        )


def _receipt_admission_registry_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt, "ROOT", tmp_path)
    final_api = admission.final_once_api
    campaign_key = "c" * 64
    ledger_root = tmp_path / "account-ledger"
    registry = ledger_root / campaign_key
    registry.mkdir(parents=True)
    monkeypatch.setattr(final_api, "campaign_key", lambda: campaign_key)
    monkeypatch.setattr(final_api, "_ledger_root", lambda: ledger_root)
    artifacts = {}
    components = {}
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for index, name in enumerate(admission.ARTIFACT_NAMES):
        ledger_name = admission.FINAL_ONCE_LEDGER_ARTIFACTS.get(name)
        if ledger_name is None:
            path = evidence / (name + ".bin")
            relative = path.relative_to(tmp_path).as_posix()
        else:
            path = registry / ledger_name
            relative = "external/final_once_ledger/" + ledger_name
        path.write_bytes((str(index) + ":" + name + "\n").encode("utf-8"))
        components[name] = path
        artifacts[name] = {"path": relative, "sha256": file_hash(path)}
    value = {
        "artifacts": artifacts,
        # This intentionally forged non-runtime claim must never be trusted by
        # receipt/package self-consistency alone.
        "bindings": {"explanation_audit_sha256": "f" * 64},
    }
    path = tmp_path / "diagnostic_admission.json"
    path.write_text(canonical(value) + "\n", encoding="utf-8")
    return path, value, components


def test_release_receipt_reauthenticates_every_admission_component(
        tmp_path, monkeypatch):
    path, value, expected_components = _receipt_admission_registry_fixture(
        tmp_path, monkeypatch)
    calls = []

    def strict_reader(saved_path, *, expected_sha256, components):
        calls.append((saved_path, expected_sha256, components))
        return deepcopy(value)

    monkeypatch.setattr(admission, "read_saved_admission", strict_reader)
    result = receipt.read_strict_admission_anchor(
        path, expected_sha256=file_hash(path))
    assert result == value
    assert calls == [(path, file_hash(path), expected_components)]


def test_release_receipt_strict_anchor_rejects_forged_nonruntime_lineage(
        tmp_path, monkeypatch):
    path, registry, expected_components = _receipt_admission_registry_fixture(
        tmp_path, monkeypatch)
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    bindings = {
        name: sha256(("binding:" + name).encode("utf-8")).hexdigest()
        for name in admission.BINDING_FIELDS
        if name != "tutorial_scene_id"
    }
    bindings["tutorial_scene_id"] = "tutorial"
    valid = {
        "version": admission.VERSION,
        "status": admission.STATUS,
        "admitted": True,
        "namespace": admission.NAMESPACE,
        "pilot_class": admission.NAMESPACE,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False,
        "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "data_persistent": False,
        "runtime_action_override": False,
        "internal_diagnostic_only": True,
        "test_fixture": False,
        "bindings": bindings,
        "artifacts": registry["artifacts"],
        "gates": {name: True for name in admission.GATE_NAMES},
        "sources": {},
        "package_contract": {},
        "self_path": path.relative_to(tmp_path).as_posix(),
    }
    checked = {
        key: deepcopy(valid[key])
        for key in ("bindings", "artifacts", "gates", "sources",
                    "package_contract")
    }
    forged = deepcopy(valid)
    forged["bindings"]["explanation_audit_sha256"] = "0" * 64
    assert forged["bindings"] != checked["bindings"]
    path.write_text(canonical(forged) + "\n", encoding="utf-8")
    monkeypatch.setattr(admission, "source_closure", lambda: {})
    monkeypatch.setattr(
        admission, "_canonical_component_paths",
        lambda components: dict(expected_components))
    monkeypatch.setattr(
        admission, "_validate_component_transaction",
        lambda *args, **kwargs: deepcopy(checked))
    with pytest.raises(ValueError, match="bindings differs from live evidence"):
        receipt.read_strict_admission_anchor(
            path, expected_sha256=file_hash(path))


def test_release_receipt_rejects_forged_lineage_before_reading_package(
        tmp_path, monkeypatch):
    path, _, _ = _receipt_admission_registry_fixture(tmp_path, monkeypatch)
    designation = tmp_path / "designation.json"
    designation.write_text("{}\n", encoding="utf-8")
    designation_sha256 = file_hash(designation)
    monkeypatch.setattr(
        receipt.designation_binding, "EXPECTED_DESIGNATION_SHA256",
        designation_sha256)
    monkeypatch.setattr(
        admission, "read_saved_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("strict live admission lineage differs")))
    monkeypatch.setattr(
        receipt.release, "_package_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("package must not be read before admission revalidation")))
    with pytest.raises(ValueError, match="strict live admission lineage differs"):
        receipt._validate_inputs(
            admission_path=path, designation_path=designation,
            package_path=tmp_path / "forged.zip",
            base64_path=tmp_path / "forged.b64",
            rollback_path=tmp_path / "rollback.json",
            expected_admission_sha256=file_hash(path),
            expected_designation_sha256=designation_sha256,
            expected_rollback_sha256="0" * 64,
        )


def _receipt_build_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt, "ROOT", tmp_path)
    inputs = {}
    for name in ("admission", "designation", "package", "base64", "rollback"):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode("ascii"))
        inputs[name] = path
    parent = {
        name: sha256(name.encode("utf-8")).hexdigest()
        for name in release._PARENT_FIELDS
        if name not in {"version", "status", "tutorial_scene_id"}
    }
    parent.update({
        "version": admission.VERSION,
        "status": admission.STATUS,
        "tutorial_scene_id": "tutorial",
    })
    identities = {
        name: sha256(("identity:" + name).encode("utf-8")).hexdigest()
        for name in (
            "actor_sha256", "runtime_manifest_signature", "program_sha256",
            "program_content_sha256", "program_identity_sha256",
            "public_feature_contract_sha256",
            "public_feature_registry_sha256", "program_complexity_sha256",
        )
    }
    checked = {
        "manifest": {"parent": parent, "identities": identities},
        "package_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "base64_sha256": "3" * 64,
        "scene_fingerprints": [str(index) * 64 for index in range(1, 8)],
    }
    monkeypatch.setattr(receipt, "source_closure", lambda: {})
    monkeypatch.setattr(
        receipt, "_release_input_paths",
        lambda **kwargs: {name: path for name, path in inputs.items()},
    )
    return inputs, parent, checked


def test_release_receipt_copies_every_candidate_lineage_field(
        tmp_path, monkeypatch):
    inputs, parent, checked = _receipt_build_fixture(tmp_path, monkeypatch)
    semantic_paths = {}

    def validate_snapshot(**kwargs):
        argument_names = {
            "designation": "designation_path", "package": "package_path",
            "base64": "base64_path", "rollback": "rollback_path",
        }
        for name, argument in argument_names.items():
            semantic_paths[name] = Path(kwargs[argument])
            assert semantic_paths[name] != inputs[name]
            assert semantic_paths[name].read_bytes() == inputs[name].read_bytes()
        return checked

    monkeypatch.setattr(receipt, "_validate_inputs", validate_snapshot)
    monkeypatch.setattr(
        receipt, "read_saved_receipt", lambda *args, **kwargs: {})
    value = receipt.build_receipt(
        admission_path=inputs["admission"],
        expected_admission_sha256="a" * 64,
        designation_path=inputs["designation"],
        expected_designation_sha256="b" * 64,
        package_path=inputs["package"], base64_path=inputs["base64"],
        rollback_receipt_path=inputs["rollback"],
        expected_rollback_receipt_sha256="c" * 64,
        output=tmp_path / "receipt.json",
    )
    assert set(value) == receipt.FIELDS
    assert {
        name: value[name] for name in receipt.CANDIDATE_LINEAGE_FIELDS
    } == {
        name: parent[name] for name in receipt.CANDIDATE_LINEAGE_FIELDS
    }
    assert set(semantic_paths) == {"designation", "package", "base64", "rollback"}


def test_release_receipt_input_drift_after_validation_publishes_nothing(
        tmp_path, monkeypatch):
    inputs, _, checked = _receipt_build_fixture(tmp_path, monkeypatch)

    def validate_then_mutate(**kwargs):
        inputs["package"].write_bytes(b"changed after validation\n")
        return checked

    monkeypatch.setattr(receipt, "_validate_inputs", validate_then_mutate)
    output = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="input or source changed"):
        receipt.build_receipt(
            admission_path=inputs["admission"],
            expected_admission_sha256="a" * 64,
            designation_path=inputs["designation"],
            expected_designation_sha256="b" * 64,
            package_path=inputs["package"], base64_path=inputs["base64"],
            rollback_receipt_path=inputs["rollback"],
            expected_rollback_receipt_sha256="c" * 64,
            output=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".receipt.json.*.partial"))


def test_release_receipt_registry_swap_after_resolution_publishes_nothing(
        tmp_path, monkeypatch):
    admission_path, admission_value, _ = _receipt_admission_registry_fixture(
        tmp_path, monkeypatch)
    expected_admission_sha256 = file_hash(admission_path)
    anchor = tmp_path / "permanent-anchor.json"
    anchor.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        admission.final_once_api, "_anchor_path", lambda: anchor)

    rollback_asset = tmp_path / "rollback-asset.bin"
    rollback_asset.write_bytes(b"rollback asset\n")
    rollback_path = tmp_path / "rollback.json"
    rollback_path.write_text(canonical({
        "files": {
            rollback_asset.name: {
                "sha256": file_hash(rollback_asset),
                "size": rollback_asset.stat().st_size,
            },
        },
    }) + "\n", encoding="utf-8")
    expected_rollback_sha256 = file_hash(rollback_path)
    direct = {}
    for name in ("designation", "package", "base64"):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode("ascii"))
        direct[name] = path

    replacement_value = deepcopy(admission_value)
    replace_name = next(
        name for name in admission.ARTIFACT_NAMES
        if name not in admission.FINAL_ONCE_LEDGER_ARTIFACTS)
    replacement_component = tmp_path / "replacement-component.bin"
    replacement_component.write_bytes(b"different registry component\n")
    replacement_value["artifacts"][replace_name] = {
        "path": replacement_component.relative_to(tmp_path).as_posix(),
        "sha256": file_hash(replacement_component),
    }
    replacement = tmp_path / "replacement-admission.json"
    replacement.write_text(canonical(replacement_value) + "\n", encoding="utf-8")

    original_resolver = receipt._release_input_paths
    calls = 0

    def resolve_then_swap(**kwargs):
        nonlocal calls
        result = original_resolver(**kwargs)
        calls += 1
        if calls == 1:
            os.replace(replacement, admission_path)
        return result

    monkeypatch.setattr(receipt, "source_closure", lambda: {})
    monkeypatch.setattr(receipt, "_release_input_paths", resolve_then_swap)
    monkeypatch.setattr(
        receipt, "_validate_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("semantic validation must not run after registry drift")),
    )
    output = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="input registry changed"):
        receipt.build_receipt(
            admission_path=admission_path,
            expected_admission_sha256=expected_admission_sha256,
            designation_path=direct["designation"],
            expected_designation_sha256="b" * 64,
            package_path=direct["package"], base64_path=direct["base64"],
            rollback_receipt_path=rollback_path,
            expected_rollback_receipt_sha256=expected_rollback_sha256,
            output=output,
        )
    assert calls == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".receipt.json.*.partial"))


def test_render_preflight_reauthenticates_full_admission_before_package(
        tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    release_root = tmp_path / "release"
    release_root.mkdir()
    paths = {}
    for name, relative in preflight.FILES.items():
        path = release_root / relative
        path.write_bytes((name + "\n").encode("ascii"))
        path.chmod(0o600)
        paths[name] = path
    expected_receipt_sha256 = file_hash(paths["receipt"])
    mocked_receipt = {
        "admission_sha256": file_hash(paths["admission"]),
        "designation_sha256": file_hash(paths["designation"]),
        "package_sha256": file_hash(paths["package"]),
        "base64_sha256": file_hash(paths["base64"]),
    }
    monkeypatch.setattr(
        preflight.receipt_api, "read_saved_receipt",
        lambda *args, **kwargs: mocked_receipt)
    monkeypatch.setattr(
        preflight.receipt_api, "read_strict_admission_anchor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("preflight strict admission lineage differs")))
    monkeypatch.setattr(
        preflight.release, "_package_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("package must not be read before admission revalidation")))
    with pytest.raises(ValueError, match="preflight strict admission lineage differs"):
        preflight.preflight(
            release_root=release_root,
            expected_receipt_sha256=expected_receipt_sha256,
            render_yaml=tmp_path / "unused-render.yaml",
        )


def test_admission_strictly_consumes_both_row_reauthentication_reports(
        tmp_path, monkeypatch):
    prior_dir = tmp_path / "prior-reauth"
    expansion_dir = tmp_path / "expansion-reauth"
    prior_dir.mkdir(); expansion_dir.mkdir()
    prior_rows = tmp_path / "prior_v7_rows.npz"
    expansion_rows = tmp_path / "expansion_rows.npz"
    prior_rows.write_bytes(b"prior rows")
    expansion_rows.write_bytes(b"expansion rows")
    (prior_dir / "rows.npz").write_bytes(prior_rows.read_bytes())
    (expansion_dir / "expansion_rows.npz").write_bytes(
        expansion_rows.read_bytes())

    prior_contract = {"version": "prior-contract", "final_test": False}
    prior_receipt = {
        "version": admission.prior_rows_api.VERSION,
        "status": admission.prior_rows_api.STATUS,
        "contract": prior_contract,
        "validation": {
            "final_scene_geometry_access": False,
            "final_trajectories_access": False,
            "final_actor_outputs_or_labels_access": False,
            "final_used_for_fit_selection_metrics": False,
        },
        "runtime_action_override": False,
        "formal_ready": False,
    }
    prior_receipt["content_sha256"] = digest(prior_receipt)
    (prior_dir / "report.json").write_text(
        canonical(prior_receipt) + "\n", encoding="utf-8")

    expansion_contract = {
        "version": "expansion-contract", "final_test_accessed": False}
    expansion_receipt = {
        "version": admission.expansion_rows_api.VERSION,
        "status": admission.expansion_rows_api.STATUS,
        "contract": expansion_contract,
        "final_test_accessed": False,
        "participant_data_accessed": False,
        "program_accessed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    expansion_receipt["content_sha256"] = digest(expansion_receipt)
    (expansion_dir / "report.json").write_text(
        canonical(expansion_receipt) + "\n", encoding="utf-8")

    names = (
        "actor", "protocol", "conflict_manifest", "diagnostic_designation",
        "development_supplement", "prior_v7_source_report",
        "development_expansion_registry", "development_expansion_report",
        "expansion_source_collection_report",
    )
    files = {name: tmp_path / name for name in names}
    for path in files.values():
        path.write_text("x", encoding="utf-8")
    files.update({
        "prior_rows_reauthentication_report": prior_dir / "report.json",
        "expansion_rows_reauthentication_report": expansion_dir / "report.json",
        "prior_v7_rows": prior_rows,
        "expansion_rows": expansion_rows,
    })
    hashes = {name: file_hash(path) for name, path in files.items()}
    monkeypatch.setattr(
        admission.prior_rows_api, "EXPECTED_SOURCE_REPORT_SHA256",
        hashes["prior_v7_source_report"])
    monkeypatch.setattr(
        admission.prior_rows_api, "EXPECTED_SOURCE_ROWS_SHA256",
        hashes["prior_v7_rows"])
    monkeypatch.setattr(
        admission.expansion_rows_api, "EXPECTED_SOURCE_ROWS_SHA256",
        hashes["expansion_rows"])
    monkeypatch.setattr(
        admission.expansion_rows_api,
        "EXPECTED_SOURCE_COLLECTION_REPORT_SHA256",
        hashes["expansion_source_collection_report"])
    monkeypatch.setattr(
        admission.expansion_rows_api, "EXPECTED_PRIOR_ROWS_SHA256",
        hashes["prior_v7_rows"])
    monkeypatch.setattr(
        admission.prior_rows_api, "contract", lambda: prior_contract)
    monkeypatch.setattr(
        admission.expansion_rows_api, "contract", lambda: expansion_contract)
    calls = []

    def read_prior(*args, **kwargs):
        calls.append(("prior", args, kwargs))
        return deepcopy(prior_receipt)

    def read_expansion(*args, **kwargs):
        calls.append(("expansion", args, kwargs))
        return deepcopy(expansion_receipt)

    monkeypatch.setattr(
        admission.prior_rows_api, "read_saved_artifacts", read_prior)
    monkeypatch.setattr(
        admission.expansion_rows_api, "read_saved_artifacts", read_expansion)
    prior, expansion = admission._validate_row_reauthentication(files, hashes)
    assert prior == prior_receipt and expansion == expansion_receipt
    assert [row[0] for row in calls] == ["prior", "expansion"]
    assert calls[0][2]["report_path"] == files[
        "prior_rows_reauthentication_report"]
    assert calls[0][2]["rows_path"] == files["prior_v7_rows"]
    assert calls[0][2]["embedded_source_report_path"] == files[
        "prior_v7_source_report"]
    assert calls[0][2]["source_report_path"] == files["prior_v7_source_report"]
    assert calls[0][2]["source_rows_path"] == files["prior_v7_rows"]
    assert calls[1][2]["source_rows_path"] == files[
        "expansion_rows"]
    assert calls[1][2]["collection_report_path"] == files[
        "expansion_source_collection_report"]
    assert calls[1][2]["prior_rows_path"] == files["prior_v7_rows"]
    assert calls[1][2]["report_path"] == files[
        "expansion_rows_reauthentication_report"]
    assert calls[1][2]["rows_path"] == files["expansion_rows"]
    assert calls[1][2]["expected_report_sha256"] == hashes[
        "expansion_rows_reauthentication_report"]



def _features():
    required = sorted(R41DiagnosticPublicRelationsV8._required_names())
    return tuple((*required, *(f"public.filler.{i}"
                  for i in range(197 - len(required)))))


def _config():
    common = {"learning_rate": 0.1, "max_iter": 20,
              "max_leaf_nodes": 15, "min_samples_leaf": 5,
              "l2_regularization": 0.1, "max_depth": 8, "max_bins": 64,
              "random_state": 17}
    return {"version": rcpd.CONFIG_VERSION, "pair_pool_multiplier": 8.0,
            "use_action_factor": True,
            "models": {name: {**common, "random_state": 17 + index}
                       for index, name in enumerate(("base", *GROUPS))},
            "mix_weights": {"narrow_passage": 0.2,
                            "shared_pickup": 0.3,
                            "shared_charger": 0.4}}


def _component(relations, name, binding, model_config):
    trees = [{"iteration": 0, "output_index": output,
              "nodes": [{"kind": "leaf", "value": float(output == 0)}]}
             for output in range(5)]
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_VERSION, "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(range(5)),
        "action_names": ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"],
        "output_kind": "multiclass_logits", "baseline": [0.0] * 5,
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1, "trees": trees,
        "metadata": {"diagnostic_rcpd_version": rcpd.VERSION,
                     "diagnostic_rcpd_binding_sha256": binding,
                     "component": name, "fit_rows": 20,
                     "fit_config": model_config,
                     "prediction_input": "349 deterministic public features",
                     "validation_labels_used_for_fit": False,
                     "actor_logits_used_as_program_input": False,
                     "actor_hidden_states_used": False,
                     "runtime_action_override": False}})


def _fixture(tmp_path):
    names = _features(); relations = R41DiagnosticPublicRelationsV8(names)
    config = _config(); binding = "b" * 64
    actor_sha = admission.FIXED_ACTOR_SHA256
    actor_parameters = "a" * 64
    manifest_bindings = {"manifest_file_sha256": "c" * 64,
                         "manifest_content_sha256": "d" * 64}
    components = {name: _component(relations, name, binding,
                                   config["models"][name])
                  for name in ("base", *GROUPS)}
    program = assemble_public_tree_program_v8(
        names, components["base"], components["narrow_passage"],
        components["shared_pickup"], components["shared_charger"],
        mix_weights=config["mix_weights"], routes=rcpd._fixed_routes(),
        metadata={"native_source_actor_sha256": actor_sha,
                  "source_actor_parameters_sha256": actor_parameters,
                  "source_full_manifest_bindings": manifest_bindings,
                  "diagnostic_rcpd_version": rcpd.VERSION,
                  "diagnostic_rcpd_binding_sha256": binding,
                  "fit_config_sha256": digest(config),
                  "public_feature_contract_sha256": digest(relations.contract()),
                  "pair_weight_contract_sha256": digest(weights.contract()),
                  "actions": list(admission.EXACT_ACTION_NAMES),
                  "classes": list(admission.EXACT_CLASSES),
                  "runtime_controller": "native_neural_actor_only",
                  "runtime_action_override": False,
                  "program_feedback_into_actor": False,
                  "formal_ready": False})
    path = tmp_path / "program.json"
    path.write_text(canonical(program.to_dict()) + "\n", encoding="utf-8")
    report = {"diagnostic_rcpd_binding_sha256": binding,
              "program_file_sha256": file_hash(path),
              "program_content_sha256": digest(program.to_dict()),
              "candidate": {"complexity": program.complexity(),
                            "program_content_sha256": digest(program.to_dict())}}
    return path, names, actor_sha, actor_parameters, manifest_bindings, report, config


def _validate(fixture):
    path, names, actor_sha, parameters, manifest, report, config = fixture
    return admission.validate_v8_program_identity(
        path, actor_feature_names=names, actor_sha256=actor_sha,
        actor_parameters_sha256=parameters,
        source_full_manifest_bindings=manifest,
        rcpd_report=report, fit_config=config)


def _rewrite(fixture, mutate):
    path, names, actor_sha, parameters, manifest, report, config = fixture
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(canonical(payload) + "\n", encoding="utf-8")
    report["program_file_sha256"] = file_hash(path)
    report["program_content_sha256"] = digest(payload)
    report["candidate"]["program_content_sha256"] = digest(payload)
    try:
        report["candidate"]["complexity"] = (
            __import__(
                "backend.warehouse_r41_diagnostic_public_tree_program_v8",
                fromlist=["R41DiagnosticPublicTreeProgramV8"])
            .R41DiagnosticPublicTreeProgramV8.from_dict(payload).complexity())
    except ValueError:
        pass


def test_external_program_identity_accepts_exact_explicit_public_tree(tmp_path):
    result = _validate(_fixture(tmp_path))
    assert result["program_file_sha256"] == file_hash(tmp_path / "program.json")
    assert result["complexity"]["component_count"] == 4
    assert len(result["public_feature_registry_sha256"]) == 64


@pytest.mark.parametrize("mutation", [
    lambda p: p["action_names"].__setitem__(0, "NORTH"),
    lambda p: p["metadata"]["actions"].__setitem__(0, "NORTH"),
    lambda p: p["metadata"]["classes"].__setitem__(0, 9),
    lambda p: p["metadata"].update(unregistered_identity="unexpected"),
    lambda p: p["base"]["metadata"].update(unregistered_identity="unexpected"),
    lambda p: p["specialists"][0]["route"].update(threshold=0.6),
    lambda p: p["specialists"][1].update(mix_weight=0.9),
    lambda p: p["metadata"].update(runtime_controller="tree_controller"),
    lambda p: p["metadata"].update(program_feedback_into_actor=True),
])
def test_external_program_identity_fails_closed_on_registry_changes(tmp_path, mutation):
    fixture = _fixture(tmp_path); _rewrite(fixture, mutation)
    with pytest.raises(ValueError):
        _validate(fixture)


def test_external_program_identity_enforces_complexity_limit(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(admission, "MAX_TOTAL_NODES", 1)
    with pytest.raises(ValueError, match="complexity"):
        _validate(fixture)


def test_program_must_be_json_and_rejects_pickle_marker(tmp_path):
    fixture = _fixture(tmp_path)
    path = fixture[0]
    path.write_bytes(b"\x80pickle")
    with pytest.raises(ValueError, match="Pickle"):
        _validate(fixture)


def test_bilingual_question_contract_is_strict():
    option = {"value": "WAIT", "label": {"zh": "等待", "en": "Wait"}}
    payload = {"items": [{"prompt": {"zh": "问题", "en": "Question"},
                           "options": [deepcopy(option)]} for _ in range(8)]}
    admission._validate_bilingual_question_bank(payload)
    payload["items"][3]["prompt"].pop("en")
    with pytest.raises(ValueError, match="Chinese and English"):
        admission._validate_bilingual_question_bank(payload)


def test_release_assembly_freezes_exact_admitted_question_and_tutorial_bytes(
        tmp_path, monkeypatch):
    components = {}
    artifact_registry = {}
    for index, name in enumerate(admission.ARTIFACT_NAMES):
        path = tmp_path / (str(index).zfill(2) + "-" + name + ".bin")
        raw = ("component:" + name + "\n").encode("utf-8")
        path.write_bytes(raw)
        components[name] = path
        artifact_registry[name] = {
            "path": path.name,
            "sha256": sha256(raw).hexdigest(),
        }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text("{}\n", encoding="utf-8")
    admission_sha = file_hash(admission_path)
    admitted_release_sources = release.release_sources()
    admitted_package_contract = admission.package_contract()
    admitted = {
        "version": admission.VERSION,
        "status": admission.STATUS,
        "admitted": True,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False,
        "formal_sample_eligible": False,
        "data_persistent": False,
        "bindings": {
            "question_bank_sha256": artifact_registry["question_bank"]["sha256"],
            "tutorial_sha256": artifact_registry["tutorial"]["sha256"],
            "release_sources_sha256": release.digest(admitted_release_sources),
            "package_contract_sha256": release.digest(admitted_package_contract),
        },
        "artifacts": artifact_registry,
    }
    monkeypatch.setattr(
        admission, "read_saved_admission",
        lambda *args, **kwargs: deepcopy(admitted))

    original_question = components["question_bank"].read_bytes()

    def inspect_frozen(*, paths, frozen, **kwargs):
        assert paths["question_bank"] != components["question_bank"]
        assert paths["question_bank"].read_bytes() == original_question
        components["question_bank"].write_bytes(b"transient replacement\n")
        frozen.verify()
        raise AssertionError("input drift must be rejected")

    monkeypatch.setattr(
        release, "_assemble_from_frozen_admitted_components", inspect_frozen)
    output = tmp_path / "must-not-exist.zip"
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        release.assemble_from_admitted_components(
            diagnostic_admission_path=admission_path,
            expected_diagnostic_admission_sha256=admission_sha,
            components=components, output_package=output)
    assert not output.exists()


def test_portable_manifest_binds_every_packaged_artifact_to_admitted_identity():
    parent = {
        name: "a" * 64
        for name in release._PARENT_FIELDS - {"version", "status"}
    }
    parent.update({
        "version": admission.VERSION,
        "status": admission.STATUS,
        "diagnostic_designation_sha256": release.FIXED_DESIGNATION_SHA256,
        "release_sources_sha256": release.digest(release.release_sources()),
    })
    scalar = {
        "play_scene_count", "play_scene_ids", "play_scene_fingerprints",
        "uses_terminal_designated_actor", "action_override_count",
        "source_full_manifest_version", "source_conflict_validation_version",
        "diagnostic_contract_version", "program_action_names",
        "program_classes", "program_routes", "program_mix_weights",
        "program_aggregation",
    }
    identities = {
        name: "b" * 64 for name in release._IDENTITY_FIELDS - scalar
    }
    identities.update({
        "actor_sha256": release.FIXED_ACTOR_SHA256,
        "source_full_manifest_version": release.FULL_SCENE_MANIFEST_VERSION,
        "source_conflict_validation_version": release.CONFLICT_VALIDATION_VERSION,
        "diagnostic_contract_version": release.DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_conflict_graph_sha256": release.DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "diagnostic_contract_sha256": release.DIAGNOSTIC_CONTRACT_SHA256,
        "conflict_families_sha256": release.CONFLICT_FAMILIES_SHA256,
        "play_scene_count": 7,
        "play_scene_ids": ["scene-" + str(index) for index in range(7)],
        "play_scene_fingerprints": [format(index + 1, "x") * 64 for index in range(7)],
        "uses_terminal_designated_actor": True,
        "action_override_count": 0,
        "program_action_names": ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"],
        "program_classes": [0, 1, 2, 3, 4],
        "program_routes": [{
            "feature_name": "derived.critical." + group,
            "operator": ">", "threshold": 0.5,
        } for group in release.PROGRAM_GROUPS],
        "program_mix_weights": {group: 0.5 for group in release.PROGRAM_GROUPS},
        "program_aggregation": release.PROGRAM_AGGREGATION,
        "runtime_manifest_file_sha256": parent[
            "portable_runtime_manifest_sha256"],
    })
    admitted = {
        "actor": identities["actor_sha256"],
        "protocol": identities["protocol_file_sha256"],
        "runtime_manifest": identities["runtime_manifest_file_sha256"],
        "program": identities["program_sha256"],
        "question_bank": parent["question_bank_sha256"],
        "tutorial": parent["tutorial_sha256"],
    }
    records = {
        name: {"path": release.ARTIFACT_PATHS[name], "size": 1,
               "sha256": admitted[name]}
        for name in release.ARTIFACT_PATHS
    }
    manifest = {
        "version": release.VERSION, "status": release.STATUS,
        "test_fixture": False, "pilot_class": release.PILOT_CLASS,
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False, "parent": parent,
        "artifacts": records, "identities": identities,
        "sources": {"release": release.release_sources()},
        "analysis": release.portable._analysis_protocol(),
        "release": release._release_projection(),
    }
    release._validate_manifest(manifest)
    forged = deepcopy(manifest)
    forged["artifacts"]["question_bank"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="admitted exact bytes"):
        release._validate_manifest(forged)
    forged = deepcopy(manifest)
    forged["identities"]["runtime_manifest_file_sha256"] = "c" * 64
    forged["artifacts"]["runtime_manifest"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="runtime manifest binding"):
        release._validate_manifest(forged)
    forged = deepcopy(manifest)
    forged["artifacts"]["tutorial"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="admitted exact bytes"):
        release._validate_manifest(forged)


def test_release_assembly_source_guard_rejects_post_admission_drift(
        tmp_path, monkeypatch):
    admitted_sources = {"release.py": "a" * 64}
    monkeypatch.setattr(
        release, "release_sources", lambda: {"release.py": "b" * 64})
    with pytest.raises(RuntimeError, match="release sources changed"):
        release._assemble_from_frozen_admitted_components(
            admission={}, admission_sha="a" * 64,
            bindings={
                "release_sources_sha256": release.digest(admitted_sources),
            },
            paths={}, output_package=tmp_path / "must-not-exist.zip",
            output_base64=None,
            frozen=object(), admitted_release_sources=admitted_sources,
        )
    assert not (tmp_path / "must-not-exist.zip").exists()


def test_release_limits_and_exact_v8_source_closure(tmp_path):
    too_large_zip = tmp_path / "too-large.zip"
    too_large_zip.write_bytes(b"x" * (release.MAX_PACKAGE_BYTES + 1))
    with pytest.raises(ValueError):
        release._package_bytes(package_path=too_large_zip)
    too_large_b64 = tmp_path / "too-large.b64"
    too_large_b64.write_bytes(b"A" * (release.MAX_BASE64_BYTES + 1))
    with pytest.raises(ValueError):
        release._package_bytes(base64_path=too_large_b64)
    sources = release.release_sources()
    assert "backend/warehouse_r41_diagnostic_public_tree_program_v8.py" in sources
    assert "backend/warehouse_r41_diagnostic_boosted_tree.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_admission_v6.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_input_snapshot_v8.py" in sources
    admission_sources = admission.source_closure()
    assert "backend/training/warehouse_r41_diagnostic_designation_v2.py" in admission_sources
    assert "scripts/build_warehouse_r41_diagnostic_designation_v2.py" in admission_sources
    assert (
        "backend/training/warehouse_r41_diagnostic_designation_v2_binding.py"
        in admission_sources
    )
    assert (
        "backend/training/warehouse_r41_diagnostic_frozen_manifest_v2.py"
        in admission_sources
    )
    assert (
        "backend/training/warehouse_r41_diagnostic_frozen_publication_v2.py"
        in admission_sources
    )
    assert (
        "backend/training/warehouse_r41_diagnostic_designation.py"
        not in admission_sources
    )
    assert "backend/warehouse_r41_diagnostic_model_tree_ensemble_v2.py" not in sources
    contract = admission.package_contract()
    assert contract["maximum_package_bytes"] == 750_000
    assert contract["maximum_base64_bytes"] == 1_000_000
    assert contract["archive_compression"] == "ZIP_BZIP2"
    assert contract["archive_compresslevel"] == 9
    assert contract["pickle_allowed"] is False


def test_admission_designation_source_closure_disagreement_fails(monkeypatch):
    own = Path(admission.__file__).resolve().relative_to(admission.ROOT).as_posix()
    actual = file_hash(Path(admission.__file__).resolve())
    replacement = "f" * 64 if actual != "f" * 64 else "e" * 64
    monkeypatch.setattr(
        admission.designation_api, "source_closure",
        lambda: {own: replacement})
    with pytest.raises(ValueError, match="designation source closure differs"):
        admission.source_closure()


def _temporary_admission_source_tree(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    dependency = package / "dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    seed = tmp_path / "seed.py"
    seed.write_text("from pkg import dependency\n", encoding="utf-8")
    return seed, dependency


def test_admission_source_closure_rejects_transient_split_read(
        tmp_path, monkeypatch):
    seed, _ = _temporary_admission_source_tree(tmp_path)
    helper = admission.source_closure_api
    original_local_source_hashes = helper.local_source_hashes
    original_os_read = helper.os.read
    original = seed.read_text(encoding="utf-8")
    seed_inode = seed.stat().st_ino
    attacked = False

    monkeypatch.setattr(helper, "ROOT", tmp_path)

    def transient_replacement(descriptor, count):
        nonlocal attacked
        if not attacked and os.fstat(descriptor).st_ino == seed_inode:
            attacked = True
            preserved = tmp_path / "seed-preserved.py"
            replacement = tmp_path / "seed-replacement.py"
            preserved.write_text(original, encoding="utf-8")
            replacement.write_text("VALUE = 2\n", encoding="utf-8")
            os.replace(replacement, seed)
            raw = original_os_read(descriptor, count)
            os.replace(preserved, seed)
            return raw
        return original_os_read(descriptor, count)

    monkeypatch.setattr(helper.os, "read", transient_replacement)
    monkeypatch.setattr(
        helper, "local_source_hashes",
        lambda _seeds: original_local_source_hashes((seed,)),
    )

    with pytest.raises(RuntimeError, match="changed while it was read"):
        admission.source_closure()
    assert attacked is True
    assert seed.read_text(encoding="utf-8") == original


def test_admission_source_closure_rejects_first_pass_hidden_import(
        tmp_path, monkeypatch):
    seed, dependency = _temporary_admission_source_tree(tmp_path)
    helper = admission.source_closure_api
    original_local_source_hashes = helper.local_source_hashes
    original_module_files = helper._module_files
    hidden = False

    monkeypatch.setattr(helper, "ROOT", tmp_path)

    def hide_once(parts, *, root):
        nonlocal hidden
        if list(parts) == ["pkg", "dependency"] and not hidden:
            hidden = True
            return []
        return original_module_files(parts, root=root)

    monkeypatch.setattr(helper, "_module_files", hide_once)
    monkeypatch.setattr(
        helper, "local_source_hashes",
        lambda _seeds: original_local_source_hashes((seed,)),
    )

    with pytest.raises(RuntimeError, match="changed during discovery"):
        admission.source_closure()
    assert hidden is True
    assert dependency.is_file()


def _minimal_admission_checked(sources):
    return {
        "bindings": {}, "artifacts": {}, "gates": {},
        "sources": deepcopy(sources),
        "package_contract": {"release_sources_sha256": admission.digest({})},
    }


def test_admission_semantics_use_one_private_snapshot_during_transient_swap(
        tmp_path, monkeypatch):
    component = tmp_path / "component.bin"
    component.write_bytes(b"authenticated")
    files = {name: component for name in admission.ARTIFACT_NAMES}
    hashes = {name: file_hash(component) for name in admission.ARTIFACT_NAMES}
    relatives = {name: "evidence/" + name for name in admission.ARTIFACT_NAMES}
    sources = {"admission.py": "a" * 64}
    monkeypatch.setattr(admission, "source_closure", lambda: deepcopy(sources))
    from ui import warehouse_alignment_r41_diagnostic_release_v8 as release_api
    monkeypatch.setattr(release_api, "release_sources", lambda: {})

    def validate(originals, semantic, frozen_relatives, frozen_hashes):
        assert originals == files
        assert frozen_relatives == relatives
        assert frozen_hashes == hashes
        assert len(set(semantic.values())) == 1
        private = next(iter(semantic.values()))
        assert private != component
        component.write_bytes(b"transient replacement")
        try:
            assert private.read_bytes() == b"authenticated"
        finally:
            component.write_bytes(b"authenticated")
        return _minimal_admission_checked(sources)

    monkeypatch.setattr(admission, "_validate_components_snapshot", validate)
    checked = admission._validate_component_transaction(
        files, hashes=hashes, relatives=relatives,
        expected_sources=sources, phase="transient swap regression")
    assert checked == _minimal_admission_checked(sources)


def test_admission_component_drift_during_validation_leaves_no_output(
        tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    component = tmp_path / "component.bin"
    component.write_bytes(b"before")
    frozen_paths = {name: component for name in admission.ARTIFACT_NAMES}
    sources = {"admission.py": "a" * 64}
    monkeypatch.setattr(admission, "source_closure", lambda: deepcopy(sources))
    monkeypatch.setattr(
        admission, "_canonical_component_paths",
        lambda paths: dict(frozen_paths))

    def validate(*args, **kwargs):
        component.write_bytes(b"after")
        return _minimal_admission_checked(sources)

    monkeypatch.setattr(admission, "_validate_component_transaction", validate)
    output = tmp_path / "admission.json"
    with pytest.raises(RuntimeError, match="component or source changed"):
        admission.build_admission({}, output=output)
    assert not output.exists()
    assert not (tmp_path / ".admission.json.partial").exists()


def test_admission_source_drift_during_staged_write_leaves_no_output(
        tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    component = tmp_path / "component.bin"
    component.write_bytes(b"fixed")
    frozen_paths = {name: component for name in admission.ARTIFACT_NAMES}
    drifted = {"value": False}
    monkeypatch.setattr(
        admission, "source_closure",
        lambda: {"admission.py": ("b" if drifted["value"] else "a") * 64})
    monkeypatch.setattr(
        admission, "_canonical_component_paths",
        lambda paths: dict(frozen_paths))
    monkeypatch.setattr(
        admission, "_validate_component_transaction",
        lambda *args, **kwargs: _minimal_admission_checked(
            {"admission.py": "a" * 64}))
    original_write = admission._write_new

    def drift_after_fsync(path, value):
        original_write(path, value)
        drifted["value"] = True

    monkeypatch.setattr(admission, "_write_new", drift_after_fsync)
    output = tmp_path / "admission.json"
    with pytest.raises(RuntimeError, match="source changed"):
        admission.build_admission({}, output=output)
    assert not output.exists()
    assert not (tmp_path / ".admission.json.partial").exists()


def test_saved_admission_reader_detects_component_drift_before_return(
        tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    component = tmp_path / "component.bin"
    component.write_bytes(b"fixed")
    frozen_paths = {name: component for name in admission.ARTIFACT_NAMES}
    sources = {"admission.py": "a" * 64}
    bindings = {name: "a" * 64 for name in admission.BINDING_FIELDS}
    bindings["tutorial_scene_id"] = "tutorial"
    checked = {
        "bindings": bindings,
        "artifacts": {},
        "gates": {name: True for name in admission.GATE_NAMES},
        "sources": sources,
        "package_contract": {},
    }
    value = {
        "version": admission.VERSION, "status": admission.STATUS,
        "admitted": True, "namespace": admission.NAMESPACE,
        "pilot_class": admission.NAMESPACE,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"], "formal_ready": False,
        "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "data_persistent": False, "runtime_action_override": False,
        "internal_diagnostic_only": True, "test_fixture": False,
        **checked, "self_path": "admission.json",
    }
    saved = tmp_path / "admission.json"
    saved.write_text(canonical(value) + "\n", encoding="utf-8")
    monkeypatch.setattr(admission, "source_closure", lambda: deepcopy(sources))
    monkeypatch.setattr(
        admission, "_canonical_component_paths",
        lambda paths: dict(frozen_paths))

    def validate(*args, **kwargs):
        component.write_bytes(b"changed")
        return deepcopy(checked)

    monkeypatch.setattr(admission, "_validate_component_transaction", validate)
    with pytest.raises(RuntimeError, match="component or source changed"):
        admission.read_saved_admission(
            saved, expected_sha256=file_hash(saved), components={})


def test_release_archive_uses_only_bzip2_and_rejects_deflate(monkeypatch):
    artifacts = {name: (name.encode("ascii") + b"\n") * 4
                 for name in release.ARTIFACT_PATHS}
    manifest = {"artifacts": release._artifact_records(artifacts)}
    monkeypatch.setattr(release, "_validate_manifest", lambda value: value)
    raw = release._archive_bytes(manifest, artifacts)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert {info.compress_type for info in archive.infolist()} == {
            zipfile.ZIP_BZIP2}
        assert all(info.create_system == 3 for info in archive.infolist())
        assert all((info.external_attr >> 16) & 0o777 == 0o600
                   for info in archive.infolist())

    tampered = dict(artifacts)
    tampered["program"] = b"X" + tampered["program"][1:]
    stream = io.BytesIO()
    manifest_raw = (release.canonical(manifest) + "\n").encode("utf-8")
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_BZIP2,
                         compresslevel=9) as archive:
        for name, item in ((release.MANIFEST_NAME, manifest_raw),
                           *((release.ARTIFACT_PATHS[key], tampered[key])
                             for key in release.ARTIFACT_PATHS)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100000 | 0o600) << 16
            archive.writestr(info, item, compress_type=zipfile.ZIP_BZIP2,
                             compresslevel=9)
    tampered_raw = stream.getvalue()
    with pytest.raises(ValueError, match="archived artifact differs"):
        release._read_archive(
            tampered_raw,
            expected_package_sha256=sha256(tampered_raw).hexdigest(),
            expected_manifest_sha256=sha256(manifest_raw).hexdigest(),
        )

    deflated = io.BytesIO()
    with zipfile.ZipFile(deflated, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(release.MANIFEST_NAME, manifest_raw)
        for name, path in release.ARTIFACT_PATHS.items():
            archive.writestr(path, artifacts[name])
    deflated_raw = deflated.getvalue()
    with pytest.raises(ValueError, match="unsafe"):
        release._read_archive(
            deflated_raw,
            expected_package_sha256=sha256(deflated_raw).hexdigest(),
            expected_manifest_sha256=sha256(manifest_raw).hexdigest(),
        )


def test_admission_rejects_credentials_participant_data_and_absolute_local_paths():
    for value in ({"api_key": "placeholder"},
                  {"participant": {"database_url": "placeholder"}},
                  {"path": str(Path("/") / "private" / "example.json")},
                  {"path": r"C:\\Users\\example\\private.json"},
                  {"token": "Bearer abcdefghijklmnop"}):
        with pytest.raises(ValueError):
            admission._reject_sensitive(value, "fixture")


def test_release_rejects_absolute_local_paths_but_accepts_public_urls():
    for value in ({"path": "/tmp/private.json"},
                  {"path": r"D:\\private\\evidence.json"}):
        with pytest.raises(ValueError, match="absolute-local"):
            release._reject_sensitive(value, "fixture")
    release._reject_sensitive(
        {"origin": "https://policylens-warehouse-study.onrender.com"},
        "fixture",
    )


def test_clean_checkout_source_closure_requires_tracked_hash_exact_files(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                   cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"],
                   cwd=checkout, check=True)
    source = checkout / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    sources = {"source.py": release.file_hash(source)}
    result = preflight.check_clean_checkout_source_closure(checkout, sources)
    assert result["clean"] is True
    assert result["tracked"] == 1

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        preflight.check_clean_checkout_source_closure(checkout, sources)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    untracked = checkout / "untracked.py"
    untracked.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="untracked"):
        preflight.check_clean_checkout_source_closure(
            checkout, {**sources, "untracked.py": release.file_hash(untracked)})


def test_clean_checkout_load_uses_only_authenticated_committed_sources(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                   cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"],
                   cwd=checkout, check=True)
    module = checkout / "fixture_release.py"
    module.write_text(
        """import base64
from hashlib import sha256
from types import SimpleNamespace

class Context:
    def __init__(self):
        self.provenance = {
            "version": "fixture.v1", "release_version": "fixture-release",
            "formal_sample_eligible": False, "data_persistent": False,
        }
        self.runtime = SimpleNamespace(actor_sha256="a" * 64)
        self.release = {"formal_ready": False}
        self.scenarios = {"splits": {"play": [
            {"fingerprint": str(index) * 64, "family_id": "f" + str(index)}
            for index in range(1, 8)
        ]}}
    def close(self):
        pass

def load_online_release(*, base64_path, expected_package_sha256,
                        expected_manifest_sha256):
    raw = base64.b64decode(open(base64_path, "rb").read(), validate=True)
    assert sha256(raw).hexdigest() == expected_package_sha256
    assert len(expected_manifest_sha256) == 64
    return Context()
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "fixture_release.py"], cwd=checkout,
                   check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout,
                   check=True)
    secret = tmp_path / "fixture.b64"
    package = b"synthetic-package"
    secret.write_bytes(base64.b64encode(package))
    secret.chmod(0o600)
    sources = {"fixture_release.py": release.file_hash(module)}
    result = preflight.check_clean_checkout_load(
        checkout, sources, base64_path=secret,
        package_sha256=sha256(package).hexdigest(),
        manifest_sha256="b" * 64, release_module="fixture_release",
    )
    assert result["version"] == "fixture.v1"
    assert result["actor_sha256"] == "a" * 64
    assert result["source_closure"]["tracked"] == 1

    dependency = checkout / "missing_dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    incomplete = checkout / "incomplete_release.py"
    incomplete.write_text(
        module.read_text(encoding="utf-8")
        .replace("import base64\n", "import base64\nimport missing_dependency\n"),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "missing_dependency.py", "incomplete_release.py"],
        cwd=checkout, check=True,
    )
    subprocess.run(["git", "commit", "-qm", "incomplete"], cwd=checkout,
                   check=True)
    with pytest.raises(ValueError, match="Clean-checkout"):
        preflight.check_clean_checkout_load(
            checkout,
            {"incomplete_release.py": release.file_hash(incomplete)},
            base64_path=secret, package_sha256=sha256(package).hexdigest(),
            manifest_sha256="b" * 64, release_module="incomplete_release",
        )


def test_release_schema_binds_final_once_and_v8_program_identity():
    candidate_lineage = {
        "prior_rows_reauthentication_report_sha256",
        "prior_v7_source_report_sha256", "prior_v7_rows_sha256",
        "expansion_rows_reauthentication_report_sha256",
        "expansion_source_collection_report_sha256",
        "expansion_rows_sha256", "development_expansion_registry_sha256",
        "development_expansion_report_sha256", "v8_fit_config_sha256",
        "retired_identity_projection_sha256",
        "retired_identity_projection_content_sha256",
        "retired_identity_projection_report_sha256",
        "final_rcpd_inputs_sha256", "final_rcpd_report_sha256",
        "final_rcpd_rows_sha256", "final_rcpd_pairs_sha256",
        "final_rcpd_weights_audit_sha256", "final_rcpd_candidate_sha256",
    }
    retired_lineage_aliases = {
        "prior_rows_reauthentication_receipt_sha256",
        "expansion_rows_reauthentication_receipt_sha256",
        "prior_v7_rcpd_report_sha256", "prior_v7_rcpd_rows_sha256",
        "development_expansion_rows_sha256",
    }
    assert admission.FIXED_ACTOR_SHA256 == release.FIXED_ACTOR_SHA256
    assert candidate_lineage <= admission.BINDING_FIELDS
    assert candidate_lineage <= release._PARENT_FIELDS
    assert candidate_lineage <= receipt.FIELDS
    assert candidate_lineage == receipt.CANDIDATE_LINEAGE_FIELDS
    assert not retired_lineage_aliases & admission.BINDING_FIELDS
    assert not retired_lineage_aliases & release._PARENT_FIELDS
    assert not retired_lineage_aliases & receipt.FIELDS
    assert {"final_once_identity_sha256", "final_once_attempt_completed_sha256",
            "final_once_campaign_key", "final_once_permanent_anchor_sha256",
            "final_once_candidate_authenticated_sha256",
            "final_once_holdout_started_sha256",
            "final_once_historical_exclusion_started_sha256",
            "final_once_historical_exclusion_completed_sha256",
            "final_once_holdout_completed_sha256",
            "final_once_audit_started_sha256",
            "final_once_audit_completed_sha256",
            "fresh_final_v3_exclusion_sha256",
            "fresh_final_v3_exclusion_content_sha256",
            "diagnostic_publication_authentication_sha256",
            "prior_rows_reauthentication_report_sha256",
            "expansion_rows_reauthentication_report_sha256",
            "expansion_source_collection_report_sha256",
            "prior_v7_source_report_sha256", "prior_v7_rows_sha256",
            "expansion_rows_sha256", "final_rcpd_inputs_sha256",
            "final_rcpd_pairs_sha256", "final_rcpd_weights_audit_sha256",
            "final_rcpd_candidate_sha256",
            "explanation_audit_evidence_sha256", "physical_replay_sha256",
            "program_identity_sha256", "public_feature_contract_sha256",
            "program_complexity_sha256"} <= admission.BINDING_FIELDS
    assert "retired_fresh_final_holdout_v3" not in admission.ARTIFACT_NAMES
    assert "retired_fresh_final_holdout_v1" not in admission.ARTIFACT_NAMES
    assert "retired_fresh_final_holdout_v2" not in admission.ARTIFACT_NAMES
    assert {"retired_identity_projection",
            "retired_identity_projection_report"} <= set(admission.ARTIFACT_NAMES)
    assert "fresh_final_v3_exclusion" in admission.ARTIFACT_NAMES
    assert "diagnostic_publication_authentication_sha256" in release._PARENT_FIELDS
    assert {"question_bank_sha256", "tutorial_sha256",
            "portable_runtime_manifest_sha256", "release_sources_sha256",
            "package_contract_sha256"} <= release._PARENT_FIELDS
    assert {
        "prior_rows_reauthentication_report_sha256",
        "expansion_rows_reauthentication_report_sha256",
        "expansion_source_collection_report_sha256",
        "prior_v7_source_report_sha256", "prior_v7_rows_sha256",
        "expansion_rows_sha256", "final_rcpd_inputs_sha256",
        "final_rcpd_pairs_sha256", "final_rcpd_weights_audit_sha256",
        "final_rcpd_candidate_sha256",
    } <= release._PARENT_FIELDS
    assert {"program_action_names", "program_classes", "program_routes",
            "program_mix_weights", "program_aggregation"} <= release._IDENTITY_FIELDS
    assert release.ARCHIVE_WHITELIST == {
        "manifest.json", "artifacts/actor.npz", "artifacts/training_protocol.json",
        "artifacts/runtime_manifest.json", "artifacts/program.json",
        "artifacts/question_bank.json", "artifacts/tutorial.json"}
    assert {"public_feature_contract_sha256", "public_feature_registry_sha256",
            "program_complexity_sha256", "final_once_identity_sha256",
            "final_once_attempt_started_sha256",
            "final_once_attempt_completed_sha256",
            "final_once_campaign_key", "final_once_permanent_anchor_sha256",
            "final_once_candidate_authenticated_sha256",
            "final_once_holdout_started_sha256",
            "final_once_historical_exclusion_started_sha256",
            "final_once_historical_exclusion_completed_sha256",
            "final_once_holdout_completed_sha256",
            "final_once_audit_started_sha256",
            "final_once_audit_completed_sha256",
            "fresh_final_v3_exclusion_sha256",
            "fresh_final_v3_exclusion_content_sha256",
            "diagnostic_publication_authentication_sha256",
            "prior_rows_reauthentication_report_sha256",
            "expansion_rows_reauthentication_report_sha256",
            "expansion_source_collection_report_sha256",
            "development_expansion_registry_sha256",
            "development_expansion_report_sha256",
            "prior_v7_source_report_sha256",
            "prior_v7_rows_sha256",
            "expansion_rows_sha256",
            "v8_fit_config_sha256", "final_rcpd_report_sha256",
            "final_rcpd_inputs_sha256", "final_rcpd_rows_sha256",
            "final_rcpd_pairs_sha256", "final_rcpd_weights_audit_sha256",
            "final_rcpd_candidate_sha256",
            "explanation_audit_inputs_sha256",
            "explanation_audit_evidence_sha256", "explanation_audit_sha256",
            "physical_replay_sha256", "release_sources_sha256",
            "package_contract_sha256"} <= receipt.FIELDS


def test_final_once_ledger_artifact_paths_are_external_and_redacted(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    registry = tmp_path / "ledger" / ("a" * 64)
    registry.mkdir(parents=True)
    started = registry / "attempt_started.json"
    started.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(admission, "ROOT", repo)

    path, stored = admission._artifact_path(
        started, "final-once attempt start",
        external_name="attempt_started.json",
    )
    assert path == started
    assert stored == "external/final_once_ledger/attempt_started.json"
    assert str(tmp_path) not in stored
    assert admission.FINAL_ONCE_LEDGER_ARTIFACTS == {
        "final_once_attempt_started": "attempt_started.json",
        "final_once_candidate_authenticated": "candidate_authenticated.json",
        "final_once_holdout_started": "holdout_started.json",
        "final_once_historical_exclusion_started": (
            "historical_exclusion_started.json"),
        "final_once_historical_exclusion_completed": (
            "historical_exclusion_completed.json"),
        "final_once_holdout_completed": "holdout_completed.json",
        "final_once_audit_started": "audit_started.json",
        "final_once_audit_completed": "audit_completed.json",
        "final_once_attempt_completed": "attempt_completed.json",
    }
    assert set(admission.FINAL_ONCE_LEDGER_ARTIFACTS.values()) == {
        "attempt_started.json", "attempt_completed.json",
        *admission.final_once_api.PHASE_RECEIPT_NAMES,
    }

    inside = repo / "attempt_started.json"
    inside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external final-once ledger"):
        admission._artifact_path(
            inside, "final-once attempt start",
            external_name="attempt_started.json",
        )


def test_admission_consumes_holdout_v4_four_field_v3_claim_interface(
        monkeypatch):
    candidates = {}
    for family_index, family in enumerate(holdout.FAMILY_IDS):
        candidates[family] = [
            {"fingerprint": digest({"family": family, "index": index}),
             "seed": 1000 * family_index + index, "family_id": family}
            for index in range(holdout.FAMILY_QUOTAS[family])
        ]
    monkeypatch.setattr(
        holdout.legacy_v3, "_ordered_candidates",
        lambda manifest, family: candidates[family])
    monkeypatch.setattr(holdout, "screen_scene",
                        lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(
        holdout, "_exact_final_workload_observations",
        lambda runtime, scene, index: {digest({"observation": index})})
    expected = admission._final_claim_binding(
        campaign_key="a" * 64,
        attempt_started_sha256="b" * 64,
        candidate_identity_sha256="c" * 64,
        candidate_authenticated_sha256="d" * 64,
    )
    tombstone = holdout._build_v3_tombstone(
        runtime=object(), actor=object(),
        manifest={"splits": {"train": []}, "candidate_batches": []},
        selected={"X": [], "Y": []}, legacy_development_hashes=set(),
        projected_retired_fingerprints=set(), projected_retired_seeds=set(),
        claim_binding=expected,
    )
    admission._require_v3_claim_binding(
        tombstone["claim_binding"], expected)
    assert set(tombstone["claim_binding"]) == {
        "campaign_key", "attempt_started_sha256",
        "candidate_identity_sha256", "candidate_authenticated_sha256",
    }

    missing_authentication = dict(tombstone["claim_binding"])
    missing_authentication.pop("candidate_authenticated_sha256")
    with pytest.raises(ValueError, match="claim binding"):
        admission._require_v3_claim_binding(
            missing_authentication, expected)


@pytest.mark.parametrize("changed", [
    "candidate_authenticated_sha256",
    "candidate_artifacts",
    "candidate_artifacts_sha256",
    "source_full_manifest_bindings",
    "source_full_manifest_bindings_sha256",
    "runtime_sources",
    "runtime_sources_sha256",
])
def test_admission_consumes_complete_audit_v8_candidate_and_source_interface(
        monkeypatch, changed):
    full_manifest = {"manifest.py": "a" * 64}
    runtime_sources = {"runtime.py": "b" * 64}
    candidate_artifacts = {
        filename: sha256(filename.encode("utf-8")).hexdigest()
        for filename in admission.final_once_api.CANDIDATE_ARTIFACT_KEYS
    }
    monkeypatch.setattr(
        audit, "diagnostic_runtime_sources", lambda: dict(runtime_sources))
    runtime = SimpleNamespace(
        source_full_manifest_bindings=dict(full_manifest))

    # Construct the producer side from audit-v8's runtime-source interface,
    # independently of admission's consumer helper.
    produced = {
        "candidate_authenticated_sha256": "c" * 64,
        "candidate_artifacts": dict(sorted(candidate_artifacts.items())),
        "candidate_artifacts_sha256": audit.digest(
            dict(sorted(candidate_artifacts.items()))),
        "source_full_manifest_bindings": dict(full_manifest),
        "source_full_manifest_bindings_sha256": audit.digest(full_manifest),
        "runtime_sources": audit.diagnostic_runtime_sources(),
        "runtime_sources_sha256": audit.digest(
            audit.diagnostic_runtime_sources()),
    }
    consumed = {
        **admission._audit_candidate_bindings(
            candidate_authenticated_sha256="c" * 64,
            candidate_artifacts=candidate_artifacts,
        ),
        **admission._audit_runtime_source_bindings(runtime),
    }
    assert consumed == produced
    expected = {"actor_sha256": admission.FIXED_ACTOR_SHA256, **consumed}
    admission._require_exact_audit_bindings(dict(expected), expected)

    mismatched = deepcopy(expected)
    mismatched.pop(changed)
    with pytest.raises(ValueError, match="external bindings"):
        admission._require_exact_audit_bindings(mismatched, expected)


@pytest.mark.parametrize("producer", [holdout, audit])
def test_admission_requires_complete_recursive_producer_source_closure(
        producer):
    sources = producer.producer_sources()
    own_path = Path(producer.__file__).resolve().relative_to(admission.ROOT).as_posix()
    assert sources[own_path] == file_hash(Path(producer.__file__).resolve())
    assert len(sources) > 1
    admission._require_exact_producer_source_closure(
        dict(sources), sources, label="fixture")

    changed = dict(sources)
    first = next(iter(changed))
    changed[first] = "f" * 64 if changed[first] != "f" * 64 else "e" * 64
    with pytest.raises(ValueError, match="producer source closure"):
        admission._require_exact_producer_source_closure(
            changed, sources, label="fixture")
    missing = dict(sources)
    missing.pop(first)
    with pytest.raises(ValueError, match="producer source closure"):
        admission._require_exact_producer_source_closure(
            missing, sources, label="fixture")


def test_final_once_fixed_paths_ignore_home_and_ledger_environment(
        tmp_path, monkeypatch):
    final_api = admission.final_once_api
    expected_ledger = final_api.REGISTRY_ROOT.absolute()
    expected_anchor = final_api.PERMANENT_ANCHOR.absolute()
    monkeypatch.setenv("HOME", str(tmp_path / "substitute-home"))
    monkeypatch.setenv(final_api.LEDGER_ENV, str(tmp_path / "substitute-ledger"))
    monkeypatch.setenv(
        final_api.PERMANENT_ANCHOR_ENV, str(tmp_path / "substitute-anchor"))
    assert final_api._ledger_root() == expected_ledger
    assert final_api._anchor_path() == expected_anchor
    assert holdout._ledger_root() == expected_ledger
    assert holdout._anchor_path() == expected_anchor

    campaign_key = "a" * 64
    admission._require_fixed_final_once_registry(
        expected_ledger / campaign_key, campaign_key)
    with pytest.raises(ValueError, match="permanent registry layout"):
        admission._require_fixed_final_once_registry(
            tmp_path / "substitute-ledger" / campaign_key, campaign_key)


def test_release_receipt_reauthenticates_same_fixed_external_ledger(
        tmp_path, monkeypatch):
    final_api = admission.final_once_api
    identity = {"version": "synthetic-final-campaign.v1"}
    key = digest(identity)
    ledger_root = tmp_path / "account-ledger"
    registry = ledger_root / key
    registry.mkdir(parents=True)
    bindings = {
        "final_once_campaign_key": key,
        "final_once_identity_sha256": digest(identity),
        "final_once_permanent_anchor_sha256": "f" * 64,
    }
    artifacts = {}
    for index, (artifact_name, filename) in enumerate(
            admission.FINAL_ONCE_LEDGER_ARTIFACTS.items()):
        path = registry / filename
        path.write_text(canonical({"index": index}) + "\n", encoding="utf-8")
        value = file_hash(path)
        bindings[artifact_name + "_sha256"] = value
        artifacts[artifact_name] = {
            "path": "external/final_once_ledger/" + filename,
            "sha256": value,
        }
    monkeypatch.setattr(final_api, "_campaign_identity", lambda: dict(identity))
    monkeypatch.setattr(final_api, "campaign_key", lambda: key)
    monkeypatch.setattr(final_api, "_ledger_root", lambda: ledger_root)

    def read_completion(path, *, expected_completion_sha256,
                        expected_identity):
        assert path == registry
        assert expected_completion_sha256 == bindings[
            "final_once_attempt_completed_sha256"]
        assert expected_identity == identity
        return {"permanent_anchor_sha256": "f" * 64}

    monkeypatch.setattr(final_api, "read_completion", read_completion)
    monkeypatch.setenv("HOME", str(tmp_path / "substitute-home"))
    value = {"bindings": bindings, "artifacts": artifacts}
    receipt._validate_fixed_final_once_ledger(value)

    redirected = deepcopy(value)
    redirected["artifacts"]["final_once_attempt_started"]["path"] = (
        str(tmp_path / "substitute-home" / "attempt_started.json"))
    with pytest.raises(ValueError, match="external ledger differs"):
        receipt._validate_fixed_final_once_ledger(redirected)


def test_admission_pins_identity_projection_and_report_to_expansion(
        tmp_path, monkeypatch):
    projection = tmp_path / "retired_identity_projection.json"
    report = tmp_path / "report.json"
    projection.write_text("{}\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")
    files = {
        "retired_identity_projection": projection,
        "retired_identity_projection_report": report,
    }
    hashes = {name: file_hash(path) for name, path in files.items()}
    binding = {
        "file_sha256": hashes["retired_identity_projection"],
        "audit_file_sha256": hashes["retired_identity_projection_report"],
        "content_sha256": "a" * 64, "identity_count": 139,
    }
    monkeypatch.setattr(
        admission.holdout_api, "EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256",
        hashes["retired_identity_projection"])
    monkeypatch.setattr(
        admission.holdout_api,
        "EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256",
        hashes["retired_identity_projection_report"])
    monkeypatch.setattr(
        admission.holdout_api, "_retired_identity_projection",
        lambda path: ([{"seed": index, "fingerprint": digest(index)}
                       for index in range(139)], binding))
    monkeypatch.setattr(
        admission.holdout_api, "_validate_retired_expansion_binding",
        lambda actual, expansion: None if actual == binding and expansion[
            "bindings"]["retired_identity_projection"] == binding else
        (_ for _ in ()).throw(ValueError("projection differs")))
    expansion = {"bindings": {"retired_identity_projection": binding}}
    assert admission._validate_retired_identity_projection(
        files, hashes, expansion) == binding

    replacement_hashes = dict(hashes)
    replacement_hashes["retired_identity_projection"] = "f" * 64
    with pytest.raises(ValueError, match="identity-only projection"):
        admission._validate_retired_identity_projection(
            files, replacement_hashes, expansion)


def test_admission_binds_final_candidate_to_strict_refit_marker_bytes(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    by_filename = {}
    for filename in admission.final_once_api.CANDIDATE_ARTIFACT_KEYS:
        path = candidate / filename
        path.write_bytes((filename + "\n").encode("ascii"))
        by_filename[filename] = path
    simple = {}
    for name in ("actor", "protocol", "manifest", "designation",
                 "selected_scenes"):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode("ascii"))
        simple[name] = path
    singles = dict(simple)
    for filename, key in admission.final_once_api.CANDIDATE_ARTIFACT_KEYS.items():
        if key != "development_rows":
            singles[key] = by_filename[filename]
    row_paths = [by_filename["rows.npz"]]
    development_paths = []
    for index, version in enumerate((
            holdout.DEVELOPMENT_SUPPLEMENT_VERSION,
            holdout.DEVELOPMENT_EXPANSION_VERSION)):
        path = tmp_path / f"development-{index}.json"
        path.write_text(canonical({"version": version}) + "\n", encoding="utf-8")
        development_paths.append(path)
    artifacts = admission.final_once_api._candidate_artifact_hashes(
        singles, row_paths)
    campaign_key = "a" * 64
    identity_sha = "b" * 64
    start_sha = "c" * 64
    marker = {
        "version": admission.final_once_api.VERSION
            + ".candidate-authentication.v1",
        "status": "passed_strict_reader_and_refit",
        "campaign_key": campaign_key,
        "candidate_identity_sha256": identity_sha,
        "attempt_started_sha256": start_sha,
        "candidate_artifacts": artifacts,
        "candidate_artifacts_sha256": digest(artifacts),
        "rcpd_report_file_sha256": artifacts["report.json"],
        "program_file_sha256": artifacts["program.json"],
        "rows_file_sha256": artifacts["rows.npz"],
        "prior_rows_reauthentication_report_file_sha256": artifacts[
            "prior_rows_reauthentication_report.json"],
        "prior_v7_source_report_file_sha256": artifacts[
            "source_v7_report.json"],
        "prior_v7_rows_file_sha256": artifacts["prior_v7_rows.npz"],
        "expansion_rows_reauthentication_report_file_sha256": artifacts[
            "expansion_rows_reauthentication_report.json"],
        "expansion_source_collection_report_file_sha256": artifacts[
            "source_expansion_collection_report.json"],
        "expansion_rows_file_sha256": artifacts["expansion_rows.npz"],
        "development_expansion_registry_file_sha256": artifacts[
            "development_expansion.json"],
        "development_expansion_report_file_sha256": artifacts[
            "development_expansion_report.json"],
        "fit_config_file_sha256": artifacts["fit_config.json"],
        "actor_file_sha256": file_hash(simple["actor"]),
        "protocol_file_sha256": file_hash(simple["protocol"]),
        "manifest_file_sha256": file_hash(simple["manifest"]),
        "designation_file_sha256": file_hash(simple["designation"]),
        "selected_scenes_file_sha256": file_hash(simple["selected_scenes"]),
        "development_registries": {
            json.loads(path.read_text())["version"]: file_hash(path)
            for path in development_paths
        },
        "require_passed": True, "refit": True, "formal_ready": False,
    }
    marker_path = tmp_path / "candidate_authenticated.json"
    marker_path.write_text(canonical(marker) + "\n", encoding="utf-8")
    files = {"final_once_candidate_authenticated": marker_path}
    hashes = {
        "final_once_candidate_authenticated": file_hash(marker_path),
        "final_once_attempt_started": start_sha,
        "actor": file_hash(simple["actor"]),
        "protocol": file_hash(simple["protocol"]),
        "conflict_manifest": file_hash(simple["manifest"]),
        "diagnostic_designation": file_hash(simple["designation"]),
        "selected_scenes": file_hash(simple["selected_scenes"]),
    }
    assert admission._validate_candidate_authentication_marker(
        marker, files=files, hashes=hashes, singles=singles,
        row_paths=row_paths, development_paths=development_paths,
        campaign_key=campaign_key,
        candidate_identity_sha256=identity_sha,
    ) == artifacts

    by_filename["program.json"].write_bytes(b"post-claim replacement\n")
    with pytest.raises(ValueError, match="candidate artifacts changed"):
        admission._validate_candidate_authentication_marker(
            marker, files=files, hashes=hashes, singles=singles,
            row_paths=row_paths, development_paths=development_paths,
            campaign_key=campaign_key,
            candidate_identity_sha256=identity_sha,
        )


def test_online_server_recognizes_and_dispatches_v8(monkeypatch):
    assert release.VERSION == server.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSION_V8
    assert release.VERSION in server.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSIONS
    assert (server.R41_DIAGNOSTIC_RELEASE_MODULE_V8
            == "ui.warehouse_alignment_r41_diagnostic_release_v8")
    assert server.R41_DIAGNOSTIC_RELEASE_MODULE_V8 in server.RELEASE_MODULES

    calls = []

    class Module:
        @staticmethod
        def load_online_release(**kwargs):
            calls.append(kwargs)
            return "v8-context"

    monkeypatch.setattr(server.importlib, "import_module", lambda name: (
        Module if name == server.R41_DIAGNOSTIC_RELEASE_MODULE_V8 else None))
    result = server.load_online_context(
        expected_package_sha256="a" * 64,
        expected_manifest_sha256="b" * 64,
        base64_path="private.b64",
        release_module=server.R41_DIAGNOSTIC_RELEASE_MODULE_V8,
    )
    assert result == "v8-context"
    assert calls == [{
        "expected_package_sha256": "a" * 64,
        "expected_manifest_sha256": "b" * 64,
        "package_path": None,
        "base64_path": "private.b64",
    }]


def test_online_store_classifies_v8_context_as_diagnostic(tmp_path):
    from tests.test_warehouse_r41_diagnostic_online_server import (
        diagnostic_context,
    )

    context = diagnostic_context(tmp_path)
    context.provenance["version"] = release.VERSION
    store = server.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic-v8.sqlite3",
        storage_mode="ephemeral",
    )
    try:
        assert store.is_diagnostic is True
        assert store.public_release_version == release.PUBLIC_RELEASE_VERSION
    finally:
        store.close()


def test_online_server_cli_lists_v8_release_module():
    completed = subprocess.run(
        ["python", "-m", "ui.warehouse_alignment_online_server", "--help"],
        cwd=release.ROOT, check=True, capture_output=True, text=True,
    )
    assert server.R41_DIAGNOSTIC_RELEASE_MODULE_V8 in completed.stdout
