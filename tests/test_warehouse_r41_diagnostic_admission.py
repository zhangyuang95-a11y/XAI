from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_admission as admission
from backend.training.warehouse_native_common import canonical, file_hash


ZERO = "0" * 64


def _checked():
    bindings = {name: ZERO for name in admission.BINDING_FIELDS}
    bindings["tutorial_scene_id"] = "tutorial-diagnostic-001"
    return {
        "bindings": bindings,
        "artifacts": {
            name: {"path": f"output/{name}", "sha256": ZERO}
            for name in admission.ARTIFACT_NAMES
        },
        "gates": {name: True for name in admission.GATE_NAMES},
        "sources": {"source.py": ZERO},
        "package_contract": {
            "behavior_performance_waiver_scope": ["behavior_performance"],
            "formal_ready": False, "formal_sample_eligible": False,
            "data_persistent": False,
        },
    }


def test_diagnostic_admission_waives_only_behavior_and_never_formal(
        monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "ROOT", tmp_path.resolve())
    monkeypatch.setattr(admission, "validate_components", lambda paths: _checked())
    output = tmp_path / "release" / "diagnostic_admission.json"
    components = {name: tmp_path / name for name in admission.ARTIFACT_NAMES}
    value = admission.build_admission(components, output=output)
    assert value["status"] == "admitted_internal_diagnostic"
    assert value["behavior_performance_gate_passed"] is False
    assert value["behavior_performance_gate_waived"] is True
    assert value["waiver_scope"] == ["behavior_performance"]
    assert value["formal_ready"] is False
    assert value["formal_sample_eligible"] is False
    assert value["human_explanation_effect_validated"] is False
    assert value["data_persistent"] is False
    assert "behavior_performance" not in admission.GATE_NAMES
    assert admission.read_saved_admission(
        output, expected_sha256=file_hash(output), components=components,
    ) == value


def test_diagnostic_admission_rejects_broader_or_missing_waiver(
        monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "ROOT", tmp_path.resolve())
    monkeypatch.setattr(admission, "validate_components", lambda paths: _checked())
    output = tmp_path / "diagnostic_admission.json"
    components = {name: tmp_path / name for name in admission.ARTIFACT_NAMES}
    value = admission.build_admission(components, output=output)
    changed = dict(value)
    changed["waiver_scope"] = ["behavior_performance", "explanation"]
    output.write_text(canonical(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Exact non-formal diagnostic admission"):
        admission.read_saved_admission(
            output, expected_sha256=file_hash(output), components=components,
        )


def test_diagnostic_admission_component_and_binding_sets_are_closed():
    assert set(admission.ARTIFACT_NAMES) == {
        "diagnostic_designation", "actor", "protocol", "training_ledger",
        "dual_evaluation", "failure_closeout", "diagnostic_contract",
        "conflict_manifest", "conflict_validation", "dynamic_selection_report",
        "selected_scenes", "final_rcpd_report", "final_rcpd_program",
        "explanation_audit_report", "question_bank", "question_bank_report",
        "tutorial",
    }
    assert {
        "runtime_audit_sha256", "dynamic_selection_report_sha256",
        "explanation_audit_sha256", "question_bank_report_sha256",
        "tutorial_sha256", "diagnostic_designation_sha256",
    } <= admission.BINDING_FIELDS
    assert admission._RCPD_SELECTED_FIELDS == (
        "profile", "metrics", "gate", "complexity", "fit_diagnostics",
        "program_content_sha256",
    )


def test_publication_validation_replays_workloads_with_designated_actor(
        monkeypatch, tmp_path):
    contract = admission.diagnostic_contract_receipt()
    contract_path = (tmp_path / "diagnostic_contract.json").resolve()
    contract_path.write_text(canonical(contract) + "\n", encoding="utf-8")
    actor_path = (tmp_path / "actor.npz").resolve()
    actor_path.write_bytes(b"designated actor fixture")

    replayed = {
        "version": "fixture-validation",
        "passed": True,
        "workload_screen": {"replayed": True},
    }
    validation_path = (tmp_path / "validation.json").resolve()
    saved = {
        **replayed,
        "artifacts": {
            "diagnostic_contract.json": "contract-sha",
            "manifest.json": "manifest-sha",
        },
    }
    validation_path.write_text(canonical(saved) + "\n", encoding="utf-8")

    calls = []

    def validate(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return replayed

    monkeypatch.setattr(
        admission.scenes_api, "validate_diagnostic_manifest", validate,
    )
    files = {
        "diagnostic_contract": contract_path,
        "conflict_validation": validation_path,
        "actor": actor_path,
    }
    hashes = {
        "diagnostic_contract": "contract-sha",
        "conflict_manifest": "manifest-sha",
    }

    actual = admission._validate_publication(
        files, hashes, {"scene": "fixture"},
    )
    assert canonical(actual) == canonical(contract)
    assert calls == [
        (
            {"scene": "fixture"},
            {
                "replay": True,
                "workload_actor_path": actor_path,
                "replay_workloads": True,
            },
        )
    ]
