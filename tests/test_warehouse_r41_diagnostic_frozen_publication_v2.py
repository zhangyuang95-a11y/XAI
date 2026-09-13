from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_frozen_publication_v2 as subject
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_reader
from backend.training.warehouse_native_common import digest, file_hash


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912"
MANIFEST = DIRECTORY / "manifest.json"
VALIDATION = DIRECTORY / "validation.json"
CONTRACT = DIRECTORY / "diagnostic_contract.json"
ACTOR = ROOT / (
    "output/warehouse_native/r41_active_2m_20260911/boundaries/"
    "step_2000000/actor.npz"
)
PROGRAM = ROOT / (
    "output/warehouse_native/r41_diagnostic_rcpd_v6_release_candidate_20260912/"
    "program.json"
)


def _successful_final() -> dict:
    phase_receipts = {
        name: str(index + 1) * 64
        for index, name in enumerate(subject.final_once.PHASE_RECEIPT_NAMES)
    }
    candidate_artifacts = {
        name: (file_hash(PROGRAM) if name == "program.json" else "a" * 64)
        for name in subject.final_once.CANDIDATE_ARTIFACT_KEYS
    }
    return {
        "version": subject.SUCCESSFUL_FINAL_VERSION,
        "status": "authenticated_completed_passed",
        "final_once_version": subject.final_once.VERSION,
        "campaign_key": subject.final_once.campaign_key(),
        "campaign_identity": subject.final_once._campaign_identity(),
        "campaign_identity_sha256": digest(
            subject.final_once._campaign_identity()),
        "candidate_identity_sha256": digest(
            subject.final_once._campaign_identity()),
        "attempt_started_sha256": "d" * 64,
        "attempt_completed_sha256": "b" * 64,
        "permanent_anchor_sha256": "c" * 64,
        "candidate_authenticated_sha256": phase_receipts[
            "candidate_authenticated.json"],
        "candidate_artifacts": dict(sorted(candidate_artifacts.items())),
        "candidate_artifacts_sha256": digest(candidate_artifacts),
        "phase_receipts": dict(sorted(phase_receipts.items())),
        "phase_receipts_sha256": digest(dict(sorted(phase_receipts.items()))),
        "actor_file_sha256": file_hash(ACTOR),
        "manifest_file_sha256": file_hash(MANIFEST),
        "program_file_sha256": file_hash(PROGRAM),
        "retry_allowed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def test_public_reader_rejects_generic_all_without_post_success_capability():
    assert not hasattr(subject, "_read_full_manifest_after_success")
    assert not hasattr(
        manifest_reader, "_read_saved_manifest_for_post_success")
    assert not hasattr(
        manifest_reader, "_read_saved_manifest_for_authorized_full_replay")
    with pytest.raises(ValueError, match="post-success publication authorization"):
        manifest_reader.read_saved_manifest(
            MANIFEST, actor_path=ACTOR, replay_scope="all")


def test_publication_pins_exact_historical_bytes_and_live_runtime(monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runtime = manifest_reader.runtime_sources()
    capability = _successful_final()
    calls = []

    def strict_manifest(raw):
        calls.append(("parse", raw == MANIFEST.read_bytes()))
        return manifest

    def strict_replay(value, *, workload_actor, replay_scope):
        calls.append(("replay", replay_scope))
        assert value == manifest

    monkeypatch.setattr(
        subject, "_successful_final_capability",
        lambda **kwargs: deepcopy(capability))
    monkeypatch.setattr(
        subject.manifest_reader, "_read_json_object",
        strict_manifest)
    monkeypatch.setattr(subject.manifest_reader, "_validate_rows", strict_replay)
    monkeypatch.setattr(subject.manifest_reader, "runtime_sources", lambda: runtime)
    result = subject.read_saved_publication(
        manifest_path=MANIFEST, validation_path=VALIDATION,
        contract_path=CONTRACT, actor_path=ACTOR, program_path=PROGRAM)

    assert file_hash(VALIDATION) == subject.EXPECTED_VALIDATION_SHA256
    assert file_hash(CONTRACT) == subject.EXPECTED_CONTRACT_SHA256
    assert calls == [("parse", True), ("replay", "all")]
    assert result["validation"]["passed"] is True
    assert result["current_runtime"] == runtime
    assert result["successful_final"] == capability
    authentication = result["authentication_receipt"]
    assert authentication == {
        "version": subject.AUTHENTICATION_VERSION,
        "status": "authenticated_exact_publication_with_full_workload_replay",
        "replay_scope": "all",
        "final_test_replayed_for_authentication": True,
        "final_test_rows_used_for_fit_or_selection": False,
        "final_labels_used": False,
        "final_test_scene_count": len(manifest["splits"]["final_test"]),
        "manifest_file_sha256": file_hash(MANIFEST),
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_semantic_sha256": digest(manifest),
        "validation_file_sha256": file_hash(VALIDATION),
        "validation_semantic_sha256": digest(result["validation"]),
        "contract_file_sha256": file_hash(CONTRACT),
        "contract_semantic_sha256": digest(result["contract"]),
        "current_runtime_sources_sha256": runtime["sources_sha256"],
        "successful_final_capability_sha256": digest(capability),
        "final_once_version": capability["final_once_version"],
        "final_once_campaign_key": capability["campaign_key"],
        "final_once_campaign_identity_sha256": capability[
            "campaign_identity_sha256"],
        "final_once_candidate_identity_sha256": capability[
            "candidate_identity_sha256"],
        "final_once_attempt_started_sha256": capability[
            "attempt_started_sha256"],
        "final_once_attempt_completed_sha256": capability[
            "attempt_completed_sha256"],
        "final_once_permanent_anchor_sha256": capability[
            "permanent_anchor_sha256"],
        "final_once_candidate_authenticated_sha256": capability[
            "candidate_authenticated_sha256"],
        "final_once_candidate_artifacts_sha256": capability[
            "candidate_artifacts_sha256"],
        "final_once_phase_receipts_sha256": capability[
            "phase_receipts_sha256"],
        "actor_file_sha256": file_hash(ACTOR),
        "program_file_sha256": file_hash(PROGRAM),
    }
    assert result["authentication_sha256"] == digest(authentication)


def test_publication_rejects_rewritten_validation_before_manifest_replay(
        tmp_path, monkeypatch):
    changed = tmp_path / "validation.json"
    changed.write_bytes(VALIDATION.read_bytes() + b"\n")
    monkeypatch.setattr(
        subject, "_successful_final_capability", lambda **kwargs: _successful_final())
    monkeypatch.setattr(
        subject.manifest_reader, "_read_json_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manifest replay must occur after byte identity")),
    )
    with pytest.raises(ValueError, match="validation bytes differ"):
        subject.read_saved_publication(
            manifest_path=MANIFEST, validation_path=changed,
            contract_path=CONTRACT, actor_path=ACTOR, program_path=PROGRAM)


def _install_completion_fixture(tmp_path, monkeypatch, completion):
    account_home = tmp_path / "account-home"
    registry_root = account_home / ".local/state/policylens/final-once"
    permanent_anchor = account_home / ".local/state/policylens/final.anchor"
    monkeypatch.setattr(subject.final_once, "_ACCOUNT_HOME", account_home)
    monkeypatch.setattr(subject.final_once, "REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(subject.final_once, "PERMANENT_ANCHOR", permanent_anchor)
    monkeypatch.setattr(
        subject.final_once.holdout_api, "DEFAULT_LEDGER_ROOT", registry_root)
    monkeypatch.setattr(
        subject.final_once.holdout_api, "DEFAULT_PERMANENT_ANCHOR",
        permanent_anchor)
    registry = registry_root / subject.final_once.campaign_key()
    registry.mkdir(parents=True)
    (registry / "attempt_completed.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        subject.final_once, "read_completion",
        lambda *args, **kwargs: deepcopy(completion),
    )


def _write_successful_ledger(tmp_path, monkeypatch):
    account_home = tmp_path / "account-home"
    registry_root = account_home / ".local/state/policylens/final-once"
    permanent_anchor = account_home / ".local/state/policylens/final.anchor"
    monkeypatch.setattr(subject.final_once, "_ACCOUNT_HOME", account_home)
    monkeypatch.setattr(subject.final_once, "REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(subject.final_once, "PERMANENT_ANCHOR", permanent_anchor)
    monkeypatch.setattr(
        subject.final_once.holdout_api, "DEFAULT_LEDGER_ROOT", registry_root)
    monkeypatch.setattr(
        subject.final_once.holdout_api, "DEFAULT_PERMANENT_ANCHOR",
        permanent_anchor)
    identity = subject.final_once._campaign_identity()
    key, registry, _ = subject.final_once._claim(
        identity, output=tmp_path / "final-output")
    started_sha = file_hash(registry / "attempt_started.json")
    candidate_artifacts = {
        name: (file_hash(PROGRAM) if name == "program.json" else "a" * 64)
        for name in subject.final_once.CANDIDATE_ARTIFACT_KEYS
    }
    marker = {
        "version": subject.final_once.VERSION + ".candidate-authentication.v1",
        "status": "passed_strict_reader_and_refit",
        "campaign_key": key,
        "candidate_identity_sha256": digest(identity),
        "attempt_started_sha256": started_sha,
        "candidate_artifacts": candidate_artifacts,
        "candidate_artifacts_sha256": digest(candidate_artifacts),
        "rcpd_report_file_sha256": candidate_artifacts["report.json"],
        "program_file_sha256": candidate_artifacts["program.json"],
        "rows_file_sha256": candidate_artifacts["rows.npz"],
        "prior_rows_reauthentication_report_file_sha256": (
            candidate_artifacts["prior_rows_reauthentication_report.json"]),
        "prior_v7_source_report_file_sha256": candidate_artifacts[
            "source_v7_report.json"],
        "prior_v7_rows_file_sha256": candidate_artifacts["prior_v7_rows.npz"],
        "expansion_rows_reauthentication_report_file_sha256": (
            candidate_artifacts[
                "expansion_rows_reauthentication_report.json"]),
        "expansion_rows_file_sha256": candidate_artifacts["expansion_rows.npz"],
        "development_expansion_registry_file_sha256": candidate_artifacts[
            "development_expansion.json"],
        "development_expansion_report_file_sha256": candidate_artifacts[
            "development_expansion_report.json"],
        "fit_config_file_sha256": candidate_artifacts["fit_config.json"],
        "actor_file_sha256": subject.final_once.EXPECTED_ACTOR_SHA256,
        "protocol_file_sha256": subject.final_once.EXPECTED_PROTOCOL_SHA256,
        "manifest_file_sha256": subject.final_once.EXPECTED_MANIFEST_SHA256,
        "designation_file_sha256": subject.final_once.EXPECTED_DESIGNATION_SHA256,
        "selected_scenes_file_sha256": (
            subject.final_once.EXPECTED_SELECTED_SCENES_SHA256),
        "development_registries": {
            subject.final_once.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION:
                "d" * 64,
            subject.final_once.holdout_api.DEVELOPMENT_EXPANSION_VERSION:
                subject.final_once.EXPECTED_EXPANSION_REGISTRY_SHA256,
        },
        "require_passed": True,
        "refit": True,
        "formal_ready": False,
    }
    (registry / "candidate_authenticated.json").write_text(
        subject.canonical(marker) + "\n", encoding="utf-8")
    candidate_sha = file_hash(registry / "candidate_authenticated.json")
    common = {
        "campaign_key": key,
        "candidate_identity_sha256": digest(identity),
        "attempt_started_sha256": started_sha,
        "candidate_authenticated_sha256": candidate_sha,
    }
    (registry / "holdout_started.json").write_text(
        subject.canonical({**common, "status": "started_no_retry"}) + "\n",
        encoding="utf-8")
    historical_common = {
        **common,
        "holdout_started_sha256": file_hash(registry / "holdout_started.json"),
    }
    row_hashes = {
        "merged_rows.npz": candidate_artifacts["rows.npz"],
        "prior_rows.npz": candidate_artifacts["prior_v7_rows.npz"],
        "expansion_rows.npz": candidate_artifacts["expansion_rows.npz"],
    }
    row_artifact_evidence = {
        name: {
            "file_sha256": value,
            "row_count": 1,
            "unique_scene_fingerprint_count": 1,
            "scene_fingerprints_sha256": "e" * 64,
            "unique_public_observation_count": 1,
            "public_observations_sha256": "f" * 64,
        }
        for name, value in row_hashes.items()
    }
    (registry / "historical_exclusion_started.json").write_text(
        subject.canonical({
            **historical_common,
            "status": "started_irrevocable_no_retry",
            "manifest_file_sha256": subject.final_once.EXPECTED_MANIFEST_SHA256,
            "historical_final_access_refunds_attempt": False,
            "formal_ready": False,
            "row_artifact_evidence": row_artifact_evidence,
            "row_artifact_evidence_sha256": digest(row_artifact_evidence),
        }) + "\n", encoding="utf-8")
    historical_completed = {
        **historical_common,
        "status": "completed_observation_hash_exclusion",
        "historical_exclusion_started_sha256": file_hash(
            registry / "historical_exclusion_started.json"),
        "row_artifact_evidence": {
            name: {
                **value,
                "historical_scene_fingerprint_overlap": 0,
                "historical_public_observation_overlap": 0,
                "zero_historical_overlap": True,
            }
            for name, value in row_artifact_evidence.items()
        },
        "historical_final_scenes_returned": False,
        "fresh_selection_conditioned_on_historical_final": False,
        "fresh_selection_private_overlap_fallback": False,
        "fresh_selection_historical_scene_fingerprint_overlap": 0,
        "fresh_selection_historical_seed_overlap": 0,
        "fresh_selection_historical_public_observation_overlap": 0,
        "historical_final_actor_outputs_exposed": False,
        "historical_final_actor_outputs_persisted": False,
        "historical_final_labels_used": False,
        "historical_final_saved_metrics_replayed_for_authentication": False,
        "historical_final_metrics_exposed": False,
        "historical_final_metrics_persisted": False,
        "historical_final_metrics_used_for_fit_or_program_selection": False,
        "historical_final_used_for_fit_or_program_selection": False,
        "historical_final_replayed_for_exclusion_only": True,
        "historical_final_actor_executed_for_observation_exclusion_only": True,
        "retry_allowed": False,
        "formal_ready": False,
        "manifest_file_sha256": subject.final_once.EXPECTED_MANIFEST_SHA256,
        "historical_final_identity_sha256": (
            subject.final_once.holdout_api.manifest_binding
            .EXPECTED_FINAL_IDENTITY_SHA256),
        "historical_final_scene_count": subject.final_once.holdout_api.TOTAL_SCENES,
        "historical_final_public_observation_count": 1,
        "historical_final_public_observations_sha256": "d" * 64,
        "combined_excluded_scene_fingerprint_count": (
            subject.final_once.holdout_api.TOTAL_SCENES + 1),
        "combined_excluded_scene_fingerprints_sha256": "a" * 64,
        "combined_excluded_seed_count": (
            subject.final_once.holdout_api.TOTAL_SCENES + 1),
        "combined_excluded_seeds_sha256": "b" * 64,
        "combined_forbidden_public_observation_count": 1,
        "combined_forbidden_public_observations_sha256": "c" * 64,
        "combined_replayed_excluded_public_observation_count": 1,
        "combined_replayed_excluded_public_observations_sha256": "e" * 64,
        "development_row_scene_overlap": 0,
        "preexisting_scene_identity_overlap": 0,
        "preexisting_public_observation_overlap": 0,
    }
    historical_completed["row_artifact_evidence_sha256"] = digest(
        historical_completed["row_artifact_evidence"])
    historical_completed["receipt_sha256"] = digest(historical_completed)
    (registry / "historical_exclusion_completed.json").write_text(
        subject.canonical(historical_completed) + "\n", encoding="utf-8")
    (registry / "holdout_completed.json").write_text(
        subject.canonical({
            **common,
            "status": "completed_program_blind",
            "historical_exclusion_completed_sha256": file_hash(
                registry / "historical_exclusion_completed.json"),
        }) + "\n", encoding="utf-8")
    audit_common = {
        **common,
        "holdout_completed_sha256": file_hash(
            registry / "holdout_completed.json"),
        "candidate_artifacts_sha256": digest(candidate_artifacts),
    }
    (registry / "audit_started.json").write_text(
        subject.canonical({**audit_common, "status": "started_no_retry"})
        + "\n", encoding="utf-8")
    (registry / "audit_completed.json").write_text(
        subject.canonical({**audit_common, "status": "passed"}) + "\n",
        encoding="utf-8")
    phase_receipts = {
        name: file_hash(registry / name)
        for name in subject.final_once.PHASE_RECEIPT_NAMES
    }
    completion = {
        "version": subject.final_once.VERSION,
        "status": "completed_passed",
        "key": key,
        "campaign_key": key,
        "candidate_identity_sha256": digest(identity),
        "identity": identity,
        "attempt_started_sha256": started_sha,
        "permanent_anchor_sha256": file_hash(permanent_anchor),
        "phase_receipts": phase_receipts,
        "candidate_authenticated_sha256": candidate_sha,
        "candidate_artifacts": candidate_artifacts,
        "candidate_artifacts_sha256": digest(candidate_artifacts),
        "output_created": True,
        "reason": None,
        "automatic_retry": False,
        "retry_allowed": False,
        "holdout_status": "passed_program_blind_registry",
        "audit_status": "passed",
        "physical_replay_status": "passed",
        "candidate_authentication_status": "passed_strict_reader_and_refit",
        "development_authentication_refit": True,
        "program_fits": 0,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
        "producer_sources": subject.final_once.producer_sources(),
    }
    (registry / "attempt_completed.json").write_text(
        subject.canonical(completion) + "\n", encoding="utf-8")
    return completion, registry, permanent_anchor


def test_successful_capability_uses_real_completion_reader_and_anchor(
        tmp_path, monkeypatch):
    completion, registry, anchor = _write_successful_ledger(
        tmp_path, monkeypatch)
    result = subject._successful_final_capability(
        actor_path=ACTOR, manifest_path=MANIFEST, program_path=PROGRAM)
    assert result["status"] == "authenticated_completed_passed"
    assert result["attempt_completed_sha256"] == file_hash(
        registry / "attempt_completed.json")
    assert result["permanent_anchor_sha256"] == file_hash(anchor)
    assert result["candidate_artifacts"] == completion["candidate_artifacts"]
    assert result["phase_receipts"] == completion["phase_receipts"]


def _completion() -> dict:
    capability = _successful_final()
    return {
        "status": "completed_passed",
        "key": capability["campaign_key"],
        "campaign_key": capability["campaign_key"],
        "candidate_identity_sha256": capability["candidate_identity_sha256"],
        "output_created": True,
        "reason": None,
        "automatic_retry": False,
        "retry_allowed": False,
        "holdout_status": "passed_program_blind_registry",
        "audit_status": "passed",
        "physical_replay_status": "passed",
        "candidate_authentication_status": "passed_strict_reader_and_refit",
        "development_authentication_refit": True,
        "program_fits": 0,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
        "permanent_anchor_sha256": capability["permanent_anchor_sha256"],
        "candidate_authenticated_sha256": capability[
            "candidate_authenticated_sha256"],
        "candidate_artifacts": capability["candidate_artifacts"],
        "candidate_artifacts_sha256": capability["candidate_artifacts_sha256"],
        "phase_receipts": capability["phase_receipts"],
    }


def _assert_proof_failure_precedes_full_manifest(
        tmp_path, monkeypatch, *, completion=None, expected):
    if completion is not None:
        _install_completion_fixture(tmp_path, monkeypatch, completion)
    else:
        account_home = tmp_path / "account-home"
        registry_root = account_home / ".local/state/policylens/final-once"
        permanent_anchor = account_home / ".local/state/policylens/final.anchor"
        monkeypatch.setattr(subject.final_once, "_ACCOUNT_HOME", account_home)
        monkeypatch.setattr(subject.final_once, "REGISTRY_ROOT", registry_root)
        monkeypatch.setattr(subject.final_once, "PERMANENT_ANCHOR", permanent_anchor)
        monkeypatch.setattr(
            subject.final_once.holdout_api, "DEFAULT_LEDGER_ROOT", registry_root)
        monkeypatch.setattr(
            subject.final_once.holdout_api, "DEFAULT_PERMANENT_ANCHOR",
            permanent_anchor)
    monkeypatch.setattr(
        subject.manifest_reader, "_read_json_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full manifest must remain unread")),
    )
    with pytest.raises(ValueError, match=expected):
        subject.read_saved_publication(
            manifest_path=MANIFEST, validation_path=VALIDATION,
            contract_path=CONTRACT, actor_path=ACTOR, program_path=PROGRAM)


def test_missing_completion_fails_before_full_manifest(tmp_path, monkeypatch):
    _assert_proof_failure_precedes_full_manifest(
        tmp_path, monkeypatch, expected="Successful final completion")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(status="burned_failed"),
         "Exact successful final chain"),
        (lambda value: value.update(campaign_key="0" * 64),
         "Exact successful final chain"),
        (lambda value: value.update(candidate_identity_sha256="0" * 64),
         "Exact successful final chain"),
        (lambda value: value["phase_receipts"].pop("audit_completed.json"),
         "Exact successful final chain"),
        (lambda value: value["candidate_artifacts"].update(
            {"report.json": "0" * 64}),
         "Exact successful final chain"),
        (lambda value: value["candidate_artifacts"].update(
            {"program.json": "0" * 64}),
         "Exact successful final chain"),
    ],
)
def test_failed_or_mismatched_completion_fails_before_full_manifest(
        tmp_path, monkeypatch, mutation, expected):
    completion = _completion()
    mutation(completion)
    _assert_proof_failure_precedes_full_manifest(
        tmp_path, monkeypatch, completion=completion, expected=expected)


def test_anchor_mismatch_fails_before_full_manifest(tmp_path, monkeypatch):
    completion = _completion()
    _install_completion_fixture(tmp_path, monkeypatch, completion)
    monkeypatch.setattr(
        subject.final_once, "read_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Final-once permanent record differs")),
    )
    monkeypatch.setattr(
        subject.manifest_reader, "_read_json_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full manifest must remain unread")),
    )
    with pytest.raises(ValueError, match="permanent record differs"):
        subject.read_saved_publication(
            manifest_path=MANIFEST, validation_path=VALIDATION,
            contract_path=CONTRACT, actor_path=ACTOR, program_path=PROGRAM)


@pytest.mark.parametrize("changed_input", ["actor", "manifest", "program"])
def test_fixed_candidate_inputs_fail_before_full_manifest(
        tmp_path, monkeypatch, changed_input):
    completion = _completion()
    _install_completion_fixture(tmp_path, monkeypatch, completion)
    changed = tmp_path / (changed_input + ".bin")
    changed.write_bytes(b"replacement candidate input\n")
    paths = {"actor": ACTOR, "manifest": MANIFEST, "program": PROGRAM}
    paths[changed_input] = changed
    monkeypatch.setattr(
        subject.manifest_reader, "_read_json_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full manifest must remain unread")),
    )
    with pytest.raises(ValueError, match="Exact successful final chain"):
        subject.read_saved_publication(
            manifest_path=paths["manifest"], validation_path=VALIDATION,
            contract_path=CONTRACT, actor_path=paths["actor"],
            program_path=paths["program"])
