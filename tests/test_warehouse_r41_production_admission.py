from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_explanation_audit as explanation
from backend.training import warehouse_r41_final_rcpd as final_rcpd
from backend.training import warehouse_r41_production_admission as admission
from backend.training import warehouse_r41_question_bank as question
from backend.training import warehouse_r4_question_bank as old_question
from backend.training.warehouse_native_common import canonical
from backend.training.warehouse_native_public_feedback import (
    HISTORY_FEATURE_NAMES,
    VERSION as PUBLIC_FEEDBACK_VERSION,
)
from backend.training.warehouse_native_public_feedback_evaluation import REWARD
from backend.warehouse_alignment_online_runtime import digest
from backend.warehouse_r41_online_explanation import R41OnlineAlignmentExplainer
from backend.warehouse_r41_online_runtime import R41OnlineAlignmentRuntime
from core.program import ExecutableProgram, ProgramNode
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION
from env.warehouse_native.scenarios import generate_manifest


ZERO = "0" * 64


def _synthetic_actor(root: Path):
    """Build the minimal valid r4.1 runtime fixture without historical test imports."""
    config = collaborative_study_config()
    scenes = generate_manifest(counts={"play": 1}, seed=91631)
    source = {
        "trainer_version": "warehouse-native-delivery-credit-trainer.v1",
        "branch": "own_credit",
        "checkpoint_sha256": "a" * 64,
        "state_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "source_sha256": "d" * 64,
        "joint_steps": 16,
        "cumulative_joint_steps": 48,
    }
    protocol = {
        "version": "warehouse-native-continuation-protocol.v1",
        "test_fixture": True,
        "public_feedback_mode": "observed",
        "public_feedback_version": PUBLIC_FEEDBACK_VERSION,
        "reward": deepcopy(REWARD),
        "collision_training_cost": .05,
        "source": source,
        "source_lineage": [source],
        "cycle_id": "temporary-runtime",
        "branch": "own_credit",
        "delivery_credit_alpha": .5,
    }
    feature_names = list(observation_names(config)) + list(HISTORY_FEATURE_NAMES)
    metadata = {
        "format": NATIVE_ACTOR_FORMAT,
        "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": len(feature_names),
        "state_dim": 354,
        "hidden": 128,
        "actions": list(ACTIONS),
        "architecture": "two_hidden_layer_tanh",
        "action_masks": False,
        "runtime_action_override": False,
        "feature_names": feature_names,
        "experiment_version": "warehouse-native-continuation-trainer.v1",
        "protocol_sha256": digest(protocol),
        "source_sha256": source["source_sha256"],
        "scenario_manifest_sha256": digest(scenes),
        "initialization_sha256": source["state_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "public_feedback_mode": "observed",
        "public_feedback_version": PUBLIC_FEEDBACK_VERSION,
        "cycle_id": protocol["cycle_id"],
        "branch": "own_credit",
        "delivery_credit_alpha": .5,
        "source_lineage": [source],
        "source_counters": {"joint_steps": 48},
        "joint_steps": 16,
        "candidate": True,
        "test_fixture": True,
    }
    rng = np.random.default_rng(56)
    obs_dim = len(feature_names)
    weights = {
        "0.weight": rng.normal(0, .015, (128, obs_dim)).astype(np.float32),
        "0.bias": np.zeros(128, np.float32),
        "2.weight": rng.normal(0, .015, (128, 128)).astype(np.float32),
        "2.bias": np.zeros(128, np.float32),
        "4.weight": rng.normal(0, .015, (len(ACTIONS), 128)).astype(np.float32),
        "4.bias": np.asarray([1, 0, 0, 0, 0], np.float32),
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "synthetic.npz"
    np.savez(path, metadata_json=np.asarray(canonical(metadata)), **weights)
    return path, protocol


def _components(root: Path):
    result = {}
    for name in admission.ARTIFACT_NAMES:
        path = root / (name + (".npz" if name == "actor" else ".json"))
        path.write_bytes((name + "\n").encode())
        result[name] = path
    return result


def _checked(root: Path, components):
    bindings = {name: ZERO for name in admission.BINDING_FIELDS}
    bindings["tutorial_scene_id"] = "tutorial_0000"
    artifacts = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in components.items()
    }
    package = {
        "release_version": "warehouse-r41-online-release.v1",
        "release_status": "r41_online_portable_internal_pilot",
        "admission_version": admission.VERSION,
        "artifact_paths": {}, "archive_whitelist": [],
        "maximum_package_bytes": 1, "maximum_base64_bytes": 1,
        "release_sources_sha256": ZERO,
        "requires_external_admission_sha256": True,
        "independent_final_program": True, "formal_ready": False,
    }
    return {
        "bindings": bindings, "artifacts": artifacts,
        "gates": {name: True for name in admission.GATE_NAMES},
        "sources": {"source.py": ZERO}, "package_contract": package,
        "runtime_audit": [],
    }


def test_admission_is_internal_pilot_only_and_revalidates_components(monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    components = _components(tmp_path)
    checked = _checked(tmp_path, components)
    calls = []
    monkeypatch.setattr(admission, "validate_components",
                        lambda values: calls.append(dict(values)) or deepcopy(checked))
    output = tmp_path / "admission.json"
    built = admission.build_admission(components, output=output)
    expected_sha = sha256(output.read_bytes()).hexdigest()
    saved = admission.read_saved_admission(
        output, expected_sha256=expected_sha, components=components)
    assert built == saved
    assert saved["version"] == "warehouse-r41-production-admission.v1"
    assert saved["status"] == "admitted_internal_pilot"
    assert saved["admitted"] is True
    assert saved["formal_ready"] is False
    assert saved["internal_pilot_only"] is True
    assert saved["human_explanation_effect_validated"] is False
    assert len(calls) == 3  # build, build's reload, explicit reload


def test_admission_fails_before_writing_when_any_component_gate_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    components = _components(tmp_path)
    monkeypatch.setattr(admission, "validate_components",
                        lambda values: (_ for _ in ()).throw(ValueError("dual eval failed")))
    output = tmp_path / "admission.json"
    with pytest.raises(ValueError, match="dual eval failed"):
        admission.build_admission(components, output=output)
    assert not output.exists()


def test_saved_admission_rejects_formal_claim_external_hash_and_component_drift(
        monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    components = _components(tmp_path)
    checked = _checked(tmp_path, components)
    monkeypatch.setattr(admission, "validate_components", lambda values: deepcopy(checked))
    output = tmp_path / "admission.json"
    admission.build_admission(components, output=output)
    good_sha = sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="bytes differ"):
        admission.read_saved_admission(output, expected_sha256="f" * 64,
                                        components=components)

    payload = json.loads(output.read_text())
    payload["formal_ready"] = True
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    changed_sha = sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Exact non-formal"):
        admission.read_saved_admission(output, expected_sha256=changed_sha,
                                        components=components)

    payload["formal_ready"] = False
    payload["bindings"]["actor_sha256"] = "a" * 64
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    changed_sha = sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="bindings differs"):
        admission.read_saved_admission(output, expected_sha256=changed_sha,
                                        components=components)


def test_component_set_and_binding_schema_are_exact(tmp_path):
    values = _components(tmp_path)
    values.pop("tutorial")
    with pytest.raises(ValueError, match="Exact r4.1 production artifact set"):
        admission.validate_components(values)
    assert len(admission.ARTIFACT_NAMES) == 16
    assert "final_rcpd_program" in admission.ARTIFACT_NAMES
    assert "corrected_six_partner_audit_report" in admission.ARTIFACT_NAMES
    assert "corrected_six_partner_audit_sha256" in admission.BINDING_FIELDS
    assert "final_rcpd_binding_sha256" in admission.BINDING_FIELDS
    assert "package_contract_sha256" in admission.BINDING_FIELDS


def test_final_rcpd_contract_is_zero_ppo_disjoint_and_has_required_grid():
    value = final_rcpd.contract()
    assert value["source_split"] == {
        "train": "train:first_70",
        "validation": "conflict_validation:first_30",
    }
    assert {4, 6, 8}.issubset(value["candidate_depths"])
    assert {16, 32, 64}.issubset(value["candidate_leaves"])
    assert value["minimum_overall_fidelity"] == .90
    assert value["minimum_critical_fidelity"] == .85
    assert value["ppo_joint_steps"] == 0
    assert value["optimizer_updates"] == 0
    assert value["program_feedback_into_actor"] is False


def test_explanation_contract_uses_frozen_final_test_and_nonformal_status():
    value = explanation.contract()
    assert value["version"] == "warehouse-r41-online-explanation-audit.v1"
    assert value["split"] == "final_test"
    assert value["scenes"] == 64
    assert value["ordinary_fidelity_min"] == .90
    assert value["effective_direction_min"] == .85
    assert value["tree_controls_runtime"] is False
    assert value["formal_ready"] is False


def test_explanation_audit_constructs_exact_r41_runtime_and_explainer(tmp_path):
    """Catch accidental reuse of the historical exact-type explainer.

    The production explanation audit constructs an r4.1 runtime directly.  Its
    explainer must therefore be the r4.1-bound wrapper; the historical wrapper
    deliberately rejects subclasses and would make every real audit fail before
    the held-out trajectories are evaluated.
    """
    actor, protocol = _synthetic_actor(tmp_path / "synthetic_r41_actor")
    runtime = explanation.OnlineAlignmentRuntime(
        actor,
        protocol=protocol,
        expected_actor_sha256=sha256(actor.read_bytes()).hexdigest(),
        expected_protocol_sha256=digest(protocol),
        allow_test_fixture=True,
    )
    assert type(runtime) is R41OnlineAlignmentRuntime
    assert explanation.OnlineAlignmentExplainer is R41OnlineAlignmentExplainer

    program = ExecutableProgram(
        ACTIONS,
        tuple(runtime.actor.metadata["feature_names"]),
        ProgramNode(probabilities=(1.0, 0.0, 0.0, 0.0, 0.0)),
        {"native_source_actor_sha256": runtime.actor_sha256},
    )
    program_path = program.save_json(tmp_path / "r41_program.json")
    explainer = explanation.OnlineAlignmentExplainer(
        program_path,
        expected_program_sha256=sha256(program_path.read_bytes()).hexdigest(),
        runtime=runtime,
        allow_test_fixture=True,
    )
    assert type(explainer) is R41OnlineAlignmentExplainer
    explainer._assert_current(runtime)


def test_admission_source_and_package_closures_include_executable_boundaries():
    sources = admission.source_closure()
    for name in (
        "core/__init__.py", "env/__init__.py",
        "backend/training/warehouse_r41_production_admission.py",
        "backend/training/warehouse_r41_final_rcpd.py",
        "backend/training/warehouse_r41_explanation_audit.py",
        "backend/training/warehouse_r41_question_bank.py",
        "backend/training/warehouse_r41_conflict_play_selection.py",
        "backend/training/warehouse_r41_corrected_partner_audit.py",
        "scripts/run_warehouse_r41_postfreeze_release.py",
        "ui/warehouse_alignment_r41_online_release.py",
    ):
        assert name in sources
        assert len(sources[name]) == 64
    package = admission.package_contract()
    assert package["admission_version"] == admission.VERSION
    assert package["requires_external_admission_sha256"] is True
    assert package["independent_final_program"] is True
    assert package["formal_ready"] is False
    assert len(package["release_sources_sha256"]) == 64


def test_admission_requires_corrected_audit_for_every_ledger_boundary(
        monkeypatch, tmp_path):
    report_path = tmp_path / "corrected.json"
    report_path.write_text("{}\n", encoding="utf-8")
    ledger = {
        "boundaries": [
            {"step": 50_000, "actor_sha256": "1" * 64},
            {"step": 100_000, "actor_sha256": "2" * 64},
        ],
    }
    passed = {
        "dual_evaluation_sha256": "d" * 64,
        "conflict_manifest_sha256": "m" * 64,
        "conflict_manifest_content_sha256": "c" * 64,
        "evaluated_boundary_steps": [50_000, 100_000],
        "evaluated_boundary_count": 2,
        "all_ledger_boundaries_replayed": True,
        "selected_actor_shutdown_count": 0,
        "boundaries": [
            {"actor_sha256": "1" * 64}, {"actor_sha256": "2" * 64},
        ],
    }
    monkeypatch.setattr(
        admission.corrected_partner_audit, "read_saved_report",
        lambda *args, **kwargs: deepcopy(passed),
    )
    result = admission._revalidate_corrected_partner_audit(
        report_path, expected_sha256="a" * 64, ledger=ledger,
        ledger_sha256="l" * 64, actor_sha256="2" * 64,
        dual_evaluation_sha256="d" * 64,
        manifest={"content_sha256": "c" * 64}, manifest_sha256="m" * 64,
    )
    assert result == passed

    passed["boundaries"].pop()
    with pytest.raises(ValueError, match="every ledger boundary"):
        admission._revalidate_corrected_partner_audit(
            report_path, expected_sha256="a" * 64, ledger=ledger,
            ledger_sha256="l" * 64, actor_sha256="2" * 64,
            dual_evaluation_sha256="d" * 64,
            manifest={"content_sha256": "c" * 64},
            manifest_sha256="m" * 64,
        )

    passed["boundaries"].append({"actor_sha256": "2" * 64})
    passed["selected_actor_shutdown_count"] = 1
    with pytest.raises(ValueError, match="every ledger boundary"):
        admission._revalidate_corrected_partner_audit(
            report_path, expected_sha256="a" * 64, ledger=ledger,
            ledger_sha256="l" * 64, actor_sha256="2" * 64,
            dual_evaluation_sha256="d" * 64,
            manifest={"content_sha256": "c" * 64},
            manifest_sha256="m" * 64,
        )


def test_question_validation_restores_shared_base_globals(monkeypatch):
    original_version = old_question.VERSION
    original_source = old_question._source_binding
    seen = {}

    def validate(runtime, payload):
        seen["version"] = old_question.VERSION
        seen["source"] = old_question._source_binding
        return {"ok": True}

    monkeypatch.setattr(old_question, "validate_payload", validate)
    runtime = object.__new__(__import__(
        "backend.warehouse_r41_online_runtime", fromlist=["R41OnlineAlignmentRuntime"]
    ).R41OnlineAlignmentRuntime)
    payload = {"pool_manifest_sha256": ZERO, "runtime_family": "alignment_feedback197"}
    assert question.validate_payload(runtime, payload) == {"ok": True}
    assert seen["version"] == question.VERSION
    assert seen["source"] is question.producer_sources
    assert old_question.VERSION == original_version
    assert old_question._source_binding is original_source
