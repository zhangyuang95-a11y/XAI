"""Fail-closed candidate runtime routing, without promoting the active v5 config."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest
import torch

from backend.training.warehouse_structured_v5 import (
    STRUCTURED_FEATURE_SCHEMA_SHA256, STRUCTURED_MODEL_VERSION,
    StructuredDualNN, export_structured_actor,
)
from domains.warehouse import neural_turnbased as warehouse
from study_v3 import policies


RAW_ID = "warehouse-structured-raw-v54"
REGULARIZED_ID = "warehouse-structured-regularized-v54"


def _candidate_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(warehouse, "ROOT", tmp_path)
    monkeypatch.setattr(policies, "ROOT", tmp_path)
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    torch.manual_seed(7)
    model = StructuredDualNN()
    artifact = export_structured_actor(
        model, tmp_path / "structured_raw.npz", pair_seed=6099, variant="raw")
    report = tmp_path / "structured_raw_training.json"
    report.write_text("{}")
    report_hash = sha256(report.read_bytes()).hexdigest()
    config = {
        "version": STRUCTURED_MODEL_VERSION,
        "model_version": STRUCTURED_MODEL_VERSION,
        "scenario_version": "warehouse-structured-dual-nn-three-tasks.v5.4",
        "observation_version": warehouse.OBSERVATION_VERSION,
        "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
        "demo_seed": 40786,
        "variants": {
            "raw": {"policy_id": RAW_ID, "actor_path": "structured_raw.npz",
                    "actor_sha256": artifact["sha256"],
                    "role_sha256": artifact["role_sha256"],
                    "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
                    "program_paths": [],
                    "training_report_path": report.name,
                    "acceptance_path": "structured_acceptance.json"},
            "regularized": {"policy_id": REGULARIZED_ID,
                            "actor_path": "structured_regularized.npz",
                            "actor_sha256": None,
                            "role_sha256": {role: None for role in artifact["role_sha256"]},
                            "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
                            "program_paths": ["robot_1_program.json",
                                              "robot_2_program.json"],
                            "training_report_path": "structured_regularized_training.json",
                            "acceptance_path": "structured_acceptance.json"},
        },
    }
    config_path = tmp_path / warehouse.STRUCTURED_CONFIG_NAME
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    acceptance = {
        "release": policies.RELEASE_ID,
        "passed": True, "heldout_evaluated": True,
        "actor_model_version": STRUCTURED_MODEL_VERSION,
        "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
        "program_schema_version": "warehouse-sparse-oblique-v5.0",
        "max_features_per_node": 3,
        "artifact_sha256": {"raw": artifact["sha256"]},
        "role_sha256": {"raw": artifact["role_sha256"]},
        "training_report_sha256": {"raw": report_hash},
        "program_sha256": {"regularized": {}},
        "pair_seeds": [6101, 6102, 6103],
        "selected_pair_seed": 6101,
        "joint_steps": {"raw": 200_000, "regularized": 200_000},
        "raw_program_feedback_rows": {"robot_1": 0, "robot_2": 0},
        "regularized_program_feedback_rows": {"robot_1": 1, "robot_2": 1},
        "post_policy_override_count": 0,
    }
    acceptance_path = tmp_path / "structured_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance))
    return config_path, acceptance_path, acceptance, artifact


def _promote_in_memory(config_path, monkeypatch):
    """Exercise the config-only promotion path without writing active config."""
    promoted = json.loads(config_path.read_text())
    promoted["variants"]["raw"]["policy_id"] = "warehouse-raw-v5"
    promoted["variants"]["regularized"]["policy_id"] = "warehouse-regularized-v5"
    monkeypatch.setattr(warehouse, "V5_CONFIG", promoted)
    return promoted








def test_promoted_standard_id_rejects_structured_schema_mismatch(
        tmp_path, monkeypatch):
    config_path, _, _, _ = _candidate_assets(tmp_path, monkeypatch)
    promoted = _promote_in_memory(config_path, monkeypatch)
    promoted["variants"]["raw"]["feature_schema_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="schema mismatch"):
        warehouse.policy_artifacts()
    with pytest.raises(RuntimeError, match="schema mismatch"):
        warehouse.initial_state(103, 1, policy_id="warehouse-raw-v5")






def test_structured_release_gate_requires_model_and_feature_schema_hashes(
        tmp_path, monkeypatch):
    _, acceptance_path, acceptance, _ = _candidate_assets(tmp_path, monkeypatch)
    entry = warehouse.structured_policy_artifacts()["raw"]
    binding = policies._binding_from_entry("warehouse", "raw", entry, RAW_ID)
    monkeypatch.setattr(policies, "_candidate_training_valid", lambda *_: True)
    monkeypatch.setattr(policies, "_selected_feedback_matches", lambda *_: True)
    monkeypatch.setattr(policies, "_development_and_audit_valid", lambda *_: True)
    monkeypatch.setattr(policies, "_warehouse_once_receipts_valid", lambda *_: True)
    monkeypatch.setattr(policies, "_heldout_quality_valid", lambda *_: True)
    assert policies._accepted(entry, binding)
    acceptance["feature_schema_sha256"] = "0" * 64
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance["feature_schema_sha256"] = STRUCTURED_FEATURE_SCHEMA_SHA256
    acceptance["actor_model_version"] = "different-model"
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance["actor_model_version"] = STRUCTURED_MODEL_VERSION
    acceptance["program_schema_version"] = "different-program"
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)


def test_promoted_standard_id_uses_structured_release_and_receipt_contract(
        tmp_path, monkeypatch):
    config_path, acceptance_path, acceptance, _ = _candidate_assets(tmp_path, monkeypatch)
    _promote_in_memory(config_path, monkeypatch)
    entry = warehouse.policy_artifacts()["raw"]
    binding = policies._binding_from_entry(
        "warehouse", "raw", entry, "warehouse-raw-v5")
    monkeypatch.setattr(policies, "_candidate_training_valid", lambda *_: True)
    monkeypatch.setattr(policies, "_selected_feedback_matches", lambda *_: True)
    monkeypatch.setattr(policies, "_development_and_audit_valid", lambda *_: True)
    monkeypatch.setattr(policies, "_warehouse_once_receipts_valid", lambda *_: True)
    monkeypatch.setattr(policies, "_heldout_quality_valid", lambda *_: True)
    assert policies._accepted(entry, binding)
    acceptance["feature_schema_sha256"] = "0" * 64
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance["feature_schema_sha256"] = STRUCTURED_FEATURE_SCHEMA_SHA256
    acceptance["actor_model_version"] = "other"
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance["actor_model_version"] = STRUCTURED_MODEL_VERSION
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted({**entry, "model_version": "other"}, binding)


@pytest.mark.parametrize("promoted", [False, True])
def test_structured_once_only_receipts_bind_model_and_feature_schema(
        tmp_path, monkeypatch, promoted):
    config_path, _, acceptance, artifact = _candidate_assets(tmp_path, monkeypatch)
    if promoted:
        _promote_in_memory(config_path, monkeypatch)
    entry = (warehouse.policy_artifacts() if promoted else
             warehouse.structured_policy_artifacts())["raw"]
    acceptance.update({
        "candidate_training": [{
            "pair_seed": 6101,
            "variants": {variant: {"scenario_schedule_sha256": "a" * 64}
                         for variant in ("raw", "regularized")},
        }],
        "role_sha256": {"raw": artifact["role_sha256"],
                        "regularized": {"robot_1": "1" * 64,
                                        "robot_2": "2" * 64}},
        "program_sha256": {"regularized": {"robot_1": "3" * 64,
                                           "robot_2": "4" * 64}},
        "heldout_scenario_seed_rule": "enrollment_seed_times_10_plus_task",
        "heldout_seed_ranges": {str(task): [250000, 250099]
                                for task in (1, 2, 3)},
        "all_candidate_regularized_feedback_passed": True,
        "final_program_audit": {"robot_1": {}, "robot_2": {}},
        "heldout": {"raw": {}, "regularized": {}},
    })
    selection = {
        "release": policies.RELEASE_ID,
        "actor_model_version": STRUCTURED_MODEL_VERSION,
        "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
        "selected_pair_seed": 6101,
        "selected_role_sha256": acceptance["role_sha256"],
        "scenario_schedule_sha256": {"raw": "a" * 64, "regularized": "a" * 64},
        "selected_program_sha256": acceptance["program_sha256"]["regularized"],
        "heldout_scenario_seed_rule": "enrollment_seed_times_10_plus_task",
        "heldout_seed_ranges": acceptance["heldout_seed_ranges"],
    }
    prefix = tmp_path / "structured"
    for suffix, payload, selected in (
        ("final_program_audit_receipt.json", acceptance["final_program_audit"],
         {**selection, "final_program_audit_seeds": list(range(240100, 240124))}),
        ("heldout_receipt.json", acceptance["heldout"], selection),
    ):
        key = "audit" if suffix.startswith("final") else "heldout"
        (tmp_path / f"{prefix.name}_{suffix}").write_text(json.dumps({
            "status": "completed", "selection": selected, key: payload,
            key + "_sha256": sha256(json.dumps(
                payload, sort_keys=True).encode()).hexdigest(),
        }))
    assert policies._warehouse_once_receipts_valid(acceptance, entry)
    acceptance["feature_schema_sha256"] = "0" * 64
    assert not policies._warehouse_once_receipts_valid(acceptance, entry)
