import ast
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as subject
from backend import warehouse_r41_diagnostic_online_runtime as runtime_api


def _exception_module_material(error, module_names):
    """Collect scalar material reachable through protected tracebacks."""
    material = []
    seen = set()

    def visit(value):
        identity = id(value)
        if identity in seen:
            return
        if isinstance(value, (BaseException, dict, list, tuple, set, frozenset)):
            seen.add(identity)
        if isinstance(value, bytes):
            material.extend((repr(value), value.hex()))
        elif isinstance(value, (str, int)):
            material.append(str(value))
        elif isinstance(value, BaseException):
            material.append(str(value))
            visit(value.args)
            visit(value.__dict__)
            traceback = value.__traceback__
            while traceback is not None:
                frame = traceback.tb_frame
                if frame.f_globals.get("__name__") in module_names:
                    visit(dict(frame.f_locals))
                traceback = traceback.tb_next
            if value.__cause__ is not None:
                visit(value.__cause__)
            if value.__context__ is not None:
                visit(value.__context__)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    visit(error)
    return "\n".join(material)


def _assert_private_material_absent(error, fragments):
    rendered = _exception_module_material(error, {subject.__name__})
    assert all(fragment not in rendered for fragment in fragments), rendered


class _Program:
    base_feature_names = tuple(f"f{i}" for i in range(197))

    def predict_batch(self, observations):
        values = np.asarray(observations)
        return (values[:, 0] != 0).astype(np.int64)

    def trace(self, features):
        assert tuple(features) == self.base_feature_names
        index = int(features["f0"] != 0)
        return {"prediction": ("UP", "DOWN")[index], "prediction_index": index,
                "public_sum": float(sum(features.values()))}


def _passing_arrays():
    ordinary = []
    pairs = []
    observation = np.zeros(197, dtype=np.float32)
    observation_hash = sha256(observation.astype("<f4").tobytes()).hexdigest()
    trace_hash = subject.digest(_Program().trace(dict(zip(
        _Program.base_feature_names, map(float, observation)))))
    for partner in range(3):
        for scene in range(64):
            fingerprint = sha256(f"scene:{scene}".encode()).hexdigest()
            ordinary.append({
                "observation": observation, "observation_hash": observation_hash,
                "trace_hash": trace_hash, "fingerprint": fingerprint,
                "scene_index": scene, "partner_index": partner, "frame": 0,
                "group_bits": 7, "player_action": 4, "actor_action": 0,
                "submitted_actor_action": 0, "executed_actor_action": 0,
                "program_action": 0, "after_physical_hash": "1" * 64,
                "done": True, "source_unchanged": True,
                "zero_overrides": True,
            })
            for action in range(4):
                changed = observation.copy(); changed[0] = action + 1
                changed_hash = sha256(changed.astype("<f4").tobytes()).hexdigest()
                changed_trace = subject.digest(_Program().trace(dict(zip(
                    _Program.base_feature_names, map(float, changed)))))
                pairs.append({
                    "fingerprint": fingerprint, "scene_index": scene,
                    "partner_index": partner, "frame": 0, "group_bits": 7,
                    "player_action": action, "active": True,
                    "physical_effect": True, "actor_changed": True,
                    "correct": True,
                    "wait_valid": True, "wait_observation": observation,
                    "wait_observation_hash": observation_hash,
                    "wait_trace_hash": trace_hash, "wait_actor_action": 0,
                    "wait_program_action": 0, "wait_physical_hash": "2" * 64,
                    "wait_submitted_equal": True,
                    "changed_valid": True, "changed_observation": changed,
                    "changed_observation_hash": changed_hash,
                    "changed_trace_hash": changed_trace, "changed_actor_action": 0,
                    "changed_program_action": 0,
                    "changed_physical_hash": "3" * 64,
                    "changed_submitted_equal": True,
                    "source_unchanged": True,
                })
    arrays = subject._as_arrays(ordinary, pairs)
    arrays["pair_changed_actor_actions"][:] = 1
    arrays["pair_changed_program_actions"][:] = 1
    arrays["pair_actor_changed"][:] = True
    return arrays


def test_audit_contract_keeps_all_hard_gates_and_raw_observations():
    contract = subject.contract()
    assert contract["ordinary_fidelity_min"] == 0.90
    assert contract["ordinary_nonwait_min"] == 0.90
    assert contract["critical_fidelity_min"] == 0.85
    assert contract["effective_direction_min"] == 0.85
    assert contract["raw_public_observations_persisted"] is True
    assert contract["reader_recomputes_predictions"] is True
    assert contract["reader_recomputes_structured_trace_hashes"] is True
    assert contract["tree_controls_runtime"] is False


def test_audit_source_has_no_training_pickle_cli_or_public_writer():
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(module == "torch" or module.startswith("torch.")
                   or module == "sklearn" or module.startswith("sklearn.")
                   or module == "pickle" for module in modules)
    assert not hasattr(subject, "main")
    assert "audit" not in subject.__all__
    sources = subject.producer_sources()
    assert {
        "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        "backend/training/warehouse_r41_diagnostic_fresh_final_holdout_v3.py",
        "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
        "backend/warehouse_r41_diagnostic_public_tree_program_v8.py",
        "scripts/build_warehouse_r41_diagnostic_designation_v2.py",
        "env/warehouse/transition_outcome.py",
        "env/warehouse_native/environment.py",
    }.issubset(sources)
    assert "core/__init__.py" in sources
    assert not any("admission" in path or "release" in path for path in sources)


def test_metrics_pass_complete_matrix_and_fail_any_direction_group():
    arrays = _passing_arrays()
    metrics = subject._metrics(arrays, development_hashes={"f" * 64})
    assert metrics["passed"] is True
    assert metrics["episodes"] == 64 * 3
    assert metrics["intervention_direction"]["overall"]["fidelity"] == 1.0
    arrays["pair_correct"][0] = False
    with pytest.raises(ValueError, match="derivations"):
        subject._metrics(arrays, development_hashes=set())


def test_raw_npz_roundtrip_and_observation_or_trace_tamper_fail_closed(tmp_path):
    arrays = _passing_arrays()
    path = tmp_path / "evidence.npz"
    subject._write_npz(path, arrays)
    restored = subject._load_npz(path)
    assert set(restored) == subject.ARRAY_KEYS
    subject._recompute_program_evidence(restored, _Program(), _Program.base_feature_names)

    restored["ordinary_observations"][0, 0] = 9
    with pytest.raises(ValueError, match="differ"):
        subject._recompute_program_evidence(
            restored, _Program(), _Program.base_feature_names)


def test_metrics_reject_development_overlap_and_bad_anchor_matrix():
    arrays = _passing_arrays()
    overlap = arrays["ordinary_observation_hashes"][0].decode("ascii")
    metrics = subject._metrics(arrays, development_hashes={overlap})
    assert metrics["passed"] is False
    assert metrics["checks"]["exact_development_observation_separation"] is False

    keep = np.ones(len(arrays["pair_frames"]), dtype=bool)
    keep[0] = False
    broken = {key: (value if key.startswith("ordinary_") else value[keep])
              for key, value in arrays.items()}
    with pytest.raises(ValueError, match="four directions"):
        subject._metrics(broken, development_hashes=set())


def test_direct_audit_rejects_before_final_or_program_access(monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append("claim")
        raise ValueError("claim rejected")

    monkeypatch.setattr(subject, "_claim_receipt", reject)
    monkeypatch.setattr(
        subject, "_regular",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an input was opened before claim")))
    with pytest.raises(ValueError, match="claim rejected"):
        subject.audit(
            actor_path="actor", protocol_path="protocol", program_path="program",
            rcpd_report_path="report", manifest_path="manifest",
            designation_path="designation", development_expansion_path="expansion",
            fresh_holdout_path="holdout", fresh_holdout_report_path="holdout-report",
            development_rows_paths=["rows"], output="output",
            claim_receipt_path="claim", expected_claim_sha256="a" * 64,
            expected_campaign_key="b" * 64,
            expected_candidate_identity_sha256="c" * 64,
            expected_holdout_completion_sha256="d" * 64,
        )
    assert calls == ["claim"]


def _audit_drift_setup(tmp_path, monkeypatch):
    claim_dir = tmp_path / "claim"
    claim_dir.mkdir()
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    candidate_paths = {}
    for name in subject.CANDIDATE_ARTIFACT_NAMES:
        path = candidate_dir / name
        path.write_bytes(("candidate:" + name).encode("utf-8"))
        candidate_paths[name] = path
    candidate_hashes = {
        name: subject.file_hash(path)
        for name, path in sorted(candidate_paths.items())
    }
    candidate_marker = {
        "candidate_artifacts": candidate_hashes,
        "candidate_artifacts_sha256": subject.digest(candidate_hashes),
    }
    (claim_dir / "candidate_authenticated.json").write_text(
        subject.canonical(candidate_marker) + "\n", encoding="utf-8")
    (claim_dir / "attempt_started.json").write_text("{}\n", encoding="utf-8")
    (claim_dir / "holdout_started.json").write_text("{}\n", encoding="utf-8")
    (claim_dir / "historical_exclusion_started.json").write_text(
        "{}\n", encoding="utf-8")
    permanent_anchor = tmp_path / "permanent-anchor.json"
    permanent_anchor.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(subject.holdout_api, "_anchor_path", lambda: permanent_anchor)
    (claim_dir / "historical_exclusion_completed.json").write_text(
        "{}\n", encoding="utf-8")

    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    holdout_path = holdout_dir / "holdout.json"
    holdout_report_path = holdout_dir / "report.json"
    exclusion_path = holdout_dir / "v3_exclusion.json"
    holdout_path.write_text("{}\n", encoding="utf-8")
    holdout_report_path.write_text("{}\n", encoding="utf-8")
    exclusion_path.write_text("{}\n", encoding="utf-8")
    holdout_phase = {
        "holdout_file_sha256": subject.file_hash(holdout_path),
        "report_file_sha256": subject.file_hash(holdout_report_path),
        "v3_exclusion_file_sha256": subject.file_hash(exclusion_path),
    }
    (claim_dir / "holdout_completed.json").write_text(
        subject.canonical(holdout_phase) + "\n", encoding="utf-8")

    direct = {}
    for name in (
            "actor", "protocol", "manifest", "designation",
            "development_expansion"):
        path = tmp_path / (name + ".json")
        path.write_bytes(name.encode("utf-8"))
        direct[name] = path
    implicit = tmp_path / "implicit-validation.json"
    implicit.write_bytes(b"implicit validation")
    monkeypatch.setattr(subject, "_claim_receipt", lambda *args, **kwargs: (
        claim_dir, {}))
    monkeypatch.setattr(
        subject, "_validate_completed_claim_chain",
        lambda *args, **kwargs: {
            "candidate_authenticated.json": candidate_marker,
            "holdout_completed.json": holdout_phase,
        })
    monkeypatch.setattr(
        subject, "_implicit_audit_input_paths",
        lambda paths: {"manifest_validation": implicit})
    args = {
        "actor_path": direct["actor"],
        "protocol_path": direct["protocol"],
        "program_path": candidate_paths["program.json"],
        "rcpd_report_path": candidate_paths["report.json"],
        "manifest_path": direct["manifest"],
        "designation_path": direct["designation"],
        "development_expansion_path": direct["development_expansion"],
        "fresh_holdout_path": holdout_path,
        "fresh_holdout_report_path": holdout_report_path,
        "development_rows_paths": [candidate_paths["rows.npz"]],
        "output": tmp_path / "audit-output",
        "claim_receipt_path": claim_dir / "attempt_started.json",
        "expected_claim_sha256": "a" * 64,
        "expected_campaign_key": "b" * 64,
        "expected_candidate_identity_sha256": "c" * 64,
        "expected_holdout_completion_sha256": "d" * 64,
    }
    return args, implicit, claim_dir


def _strict_claim_chain_fixture(tmp_path, monkeypatch):
    claim_dir = tmp_path / "strict-claim"
    claim_dir.mkdir()
    anchor_path = tmp_path / "strict-anchor.json"
    monkeypatch.setattr(subject.holdout_api, "_anchor_path", lambda: anchor_path)
    identity = subject.holdout_api._campaign_identity()
    key = subject.digest(identity)
    identity_sha256 = subject.digest(identity)
    artifact_hashes = {
        name: sha256(("artifact:" + name).encode()).hexdigest()
        for name in subject.CANDIDATE_ARTIFACT_NAMES
    }

    def write(name, value, *, path=None):
        target = (claim_dir / name) if path is None else Path(path)
        target.write_text(subject.canonical(value) + "\n", encoding="utf-8")
        return target

    anchor = {
        "version": subject.FINAL_ONCE_VERSION,
        "campaign_key": key,
        "candidate_identity_sha256": identity_sha256,
        "identity": identity,
        "status": "claimed_irrevocable_no_retry",
        "output_requested": str(tmp_path / "output"),
        "created_at": "2026-09-13T00:00:00+00:00",
        "automatic_retry": False,
        "output_removal_refunds_attempt": False,
        "formal_ready": False,
    }
    write("permanent_anchor.json", anchor, path=anchor_path)
    attempt = {
        "version": subject.FINAL_ONCE_VERSION,
        "key": key,
        "campaign_key": key,
        "candidate_identity_sha256": identity_sha256,
        "status": "started_irrevocable_no_retry",
        "identity": identity,
        "permanent_anchor_path": str(anchor_path),
        "permanent_anchor_sha256": subject.file_hash(anchor_path),
        "output_requested": str(tmp_path / "output"),
        "created_at": "2026-09-13T00:00:00+00:00",
        "automatic_retry": False,
        "output_removal_refunds_attempt": False,
        "program_evaluation_started": False,
        "formal_ready": False,
    }
    attempt_path = write("attempt_started.json", attempt)
    attempt_sha = subject.file_hash(attempt_path)
    candidate = {
        "version": subject.FINAL_ONCE_VERSION + ".candidate-authentication.v1",
        "status": "passed_strict_reader_and_refit",
        "campaign_key": key,
        "candidate_identity_sha256": identity_sha256,
        "attempt_started_sha256": attempt_sha,
        "candidate_artifacts": artifact_hashes,
        "candidate_artifacts_sha256": subject.digest(artifact_hashes),
        "rcpd_report_file_sha256": artifact_hashes["report.json"],
        "program_file_sha256": artifact_hashes["program.json"],
        "rows_file_sha256": artifact_hashes["rows.npz"],
        "prior_rows_reauthentication_report_file_sha256": artifact_hashes[
            "prior_rows_reauthentication_report.json"],
        "prior_v7_source_report_file_sha256": artifact_hashes[
            "source_v7_report.json"],
        "prior_v7_rows_file_sha256": artifact_hashes["prior_v7_rows.npz"],
        "expansion_rows_reauthentication_report_file_sha256": artifact_hashes[
            "expansion_rows_reauthentication_report.json"],
        "expansion_source_collection_report_file_sha256": artifact_hashes[
            "source_expansion_collection_report.json"],
        "expansion_rows_file_sha256": artifact_hashes["expansion_rows.npz"],
        "development_expansion_registry_file_sha256": artifact_hashes[
            "development_expansion.json"],
        "development_expansion_report_file_sha256": artifact_hashes[
            "development_expansion_report.json"],
        "fit_config_file_sha256": artifact_hashes["fit_config.json"],
        "actor_file_sha256": subject.holdout_api.EXPECTED_ACTOR_SHA256,
        "protocol_file_sha256": subject.holdout_api.EXPECTED_PROTOCOL_SHA256,
        "manifest_file_sha256": subject.holdout_api.EXPECTED_MANIFEST_SHA256,
        "designation_file_sha256": subject.holdout_api.EXPECTED_DESIGNATION_SHA256,
        "selected_scenes_file_sha256": (
            subject.holdout_api.EXPECTED_SELECTED_SCENES_SHA256),
        "development_registries": {
            subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION: "7" * 64,
            subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION: (
                subject.holdout_api.EXPECTED_EXPANSION_REGISTRY_SHA256),
        },
        "require_passed": True,
        "refit": True,
        "formal_ready": False,
    }
    candidate_path = write("candidate_authenticated.json", candidate)
    candidate_sha = subject.file_hash(candidate_path)
    common = {
        "campaign_key": key,
        "candidate_identity_sha256": identity_sha256,
        "attempt_started_sha256": attempt_sha,
        "candidate_authenticated_sha256": candidate_sha,
    }
    holdout_started = {
        "version": subject.FRESH_HOLDOUT_VERSION,
        "status": "started_no_retry",
        **common,
        "selection_salt_commitment": subject.holdout_api.HOLDOUT_SALT_COMMITMENT,
    }
    holdout_started_path = write("holdout_started.json", holdout_started)
    rows = {
        name: {
            "file_sha256": file_sha,
            "row_count": index + 1,
            "unique_scene_fingerprint_count": index + 2,
            "scene_fingerprints_sha256": str(index + 1) * 64,
            "unique_public_observation_count": index + 3,
            "public_observations_sha256": str(index + 4) * 64,
        }
        for index, (name, file_sha) in enumerate(sorted({
            "merged_rows.npz": artifact_hashes["rows.npz"],
            "prior_rows.npz": artifact_hashes["prior_v7_rows.npz"],
            "expansion_rows.npz": artifact_hashes["expansion_rows.npz"],
        }.items()))
    }
    exclusion = {
        "retired_actor_executed": False,
        "retired_observations_derived": False,
    }
    historical_started = {
        "version": subject.FRESH_HOLDOUT_VERSION + ".historical-exclusion.v1",
        "status": "started_irrevocable_no_retry",
        **common,
        "holdout_started_sha256": subject.file_hash(holdout_started_path),
        "manifest_file_sha256": subject.holdout_api.EXPECTED_MANIFEST_SHA256,
        "row_artifact_evidence": rows,
        "row_artifact_evidence_sha256": subject.digest(rows),
        "public_exclusion_commitment": exclusion,
        "public_exclusion_commitment_sha256": subject.digest(exclusion),
        "historical_final_access_refunds_attempt": False,
        "formal_ready": False,
    }
    historical_started_path = write(
        "historical_exclusion_started.json", historical_started)
    completed_rows = {
        name: {
            **value,
            "historical_scene_fingerprint_overlap": 0,
            "historical_public_observation_overlap": 0,
            "zero_historical_overlap": True,
        }
        for name, value in rows.items()
    }
    historical = {
        "version": subject.FRESH_HOLDOUT_VERSION + ".historical-exclusion.v1",
        "status": "completed_observation_hash_exclusion",
        **common,
        "holdout_started_sha256": subject.file_hash(holdout_started_path),
        "historical_exclusion_started_sha256": subject.file_hash(
            historical_started_path),
        "manifest_file_sha256": subject.holdout_api.EXPECTED_MANIFEST_SHA256,
        "historical_final_identity_sha256": (
            subject.holdout_api.manifest_binding.EXPECTED_FINAL_IDENTITY_SHA256),
        "historical_final_scene_count": subject.holdout_api.TOTAL_SCENES,
        "historical_final_public_observation_count": 1,
        "historical_final_public_observations_sha256": "8" * 64,
        "combined_excluded_scene_fingerprint_count": (
            subject.holdout_api.TOTAL_SCENES + 1),
        "combined_excluded_scene_fingerprints_sha256": "9" * 64,
        "combined_excluded_seed_count": subject.holdout_api.TOTAL_SCENES + 1,
        "combined_excluded_seeds_sha256": "a" * 64,
        "combined_forbidden_public_observation_count": 2,
        "combined_forbidden_public_observations_sha256": "b" * 64,
        "combined_replayed_excluded_public_observation_count": 2,
        "combined_replayed_excluded_public_observations_sha256": "c" * 64,
        "development_row_scene_overlap": 0,
        "preexisting_scene_identity_overlap": 0,
        "preexisting_public_observation_overlap": 0,
        "row_artifact_evidence": completed_rows,
        "row_artifact_evidence_sha256": subject.digest(completed_rows),
        "historical_final_replayed_for_exclusion_only": True,
        "historical_final_actor_executed_for_observation_exclusion_only": True,
        "historical_final_actor_outputs_exposed": False,
        "historical_final_actor_outputs_persisted": False,
        "historical_final_labels_used": False,
        "historical_final_saved_metrics_replayed_for_authentication": False,
        "historical_final_metrics_exposed": False,
        "historical_final_metrics_persisted": False,
        "historical_final_metrics_used_for_fit_or_program_selection": False,
        "historical_final_used_for_fit_or_program_selection": False,
        "historical_final_scenes_returned": False,
        "fresh_selection_conditioned_on_historical_final": False,
        "fresh_selection_private_overlap_fallback": False,
        "fresh_selection_historical_scene_fingerprint_overlap": 0,
        "fresh_selection_historical_seed_overlap": 0,
        "fresh_selection_historical_public_observation_overlap": 0,
        "retry_allowed": False,
        "formal_ready": False,
    }
    historical["receipt_sha256"] = subject.digest(historical)
    historical_path = write("historical_exclusion_completed.json", historical)
    v3_path = write("v3_exclusion.json", {"version": "v3"})
    holdout = {
        "version": subject.FRESH_HOLDOUT_VERSION,
        "status": "completed_program_blind",
        **common,
        "historical_exclusion_completed_sha256": subject.file_hash(
            historical_path),
        "holdout_file_sha256": "d" * 64,
        "report_file_sha256": "e" * 64,
        "v3_exclusion_file_sha256": subject.file_hash(v3_path),
    }
    holdout_path = write("holdout_completed.json", holdout)
    paths = {
        "attempt_started.json": attempt_path,
        "permanent_anchor.json": anchor_path,
        "candidate_authenticated.json": candidate_path,
        "holdout_started.json": holdout_started_path,
        "historical_exclusion_started.json": historical_started_path,
        "historical_exclusion_completed.json": historical_path,
        "holdout_completed.json": holdout_path,
        "v3_exclusion.json": v3_path,
    }
    expected = {
        "expected_claim_sha256": attempt_sha,
        "expected_campaign_key": key,
        "expected_candidate_identity_sha256": identity_sha256,
        "expected_holdout_completion_sha256": subject.file_hash(holdout_path),
    }
    return paths, expected


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("holdout_started.json", "status", "forged"),
        ("historical_exclusion_started.json", "status", "forged"),
        ("historical_exclusion_completed.json",
         "fresh_selection_conditioned_on_historical_final", True),
        ("historical_exclusion_completed.json",
         "fresh_selection_private_overlap_fallback", True),
        ("historical_exclusion_completed.json", "historical_final_labels_used", True),
        ("historical_exclusion_completed.json", "retry_allowed", True),
    ],
)
def test_snapshot_claim_chain_rejects_forged_phase_or_private_use_claim(
        tmp_path, monkeypatch, name, field, value):
    paths, expected = _strict_claim_chain_fixture(tmp_path, monkeypatch)
    assert subject._validate_completed_claim_chain(paths, **expected)
    forged = subject._read_json(paths[name], "phase")
    forged[field] = value
    if name == "historical_exclusion_completed.json":
        forged["receipt_sha256"] = subject.digest({
            key: item for key, item in forged.items() if key != "receipt_sha256"
        })
    paths[name].write_text(subject.canonical(forged) + "\n", encoding="utf-8")
    historical_started = subject._read_json(
        paths["historical_exclusion_started.json"], "historical start")
    historical_started["holdout_started_sha256"] = subject.file_hash(
        paths["holdout_started.json"])
    paths["historical_exclusion_started.json"].write_text(
        subject.canonical(historical_started) + "\n", encoding="utf-8")
    historical = subject._read_json(
        paths["historical_exclusion_completed.json"], "historical")
    historical["holdout_started_sha256"] = subject.file_hash(
        paths["holdout_started.json"])
    historical["historical_exclusion_started_sha256"] = subject.file_hash(
        paths["historical_exclusion_started.json"])
    historical["receipt_sha256"] = subject.digest({
        key: item for key, item in historical.items()
        if key != "receipt_sha256"
    })
    historical_path = paths["historical_exclusion_completed.json"]
    historical_path.write_text(
        subject.canonical(historical) + "\n", encoding="utf-8")
    holdout = subject._read_json(paths["holdout_completed.json"], "holdout")
    holdout["historical_exclusion_completed_sha256"] = subject.file_hash(
        historical_path)
    paths["holdout_completed.json"].write_text(
        subject.canonical(holdout) + "\n", encoding="utf-8")
    expected["expected_holdout_completion_sha256"] = subject.file_hash(
        paths["holdout_completed.json"])
    with pytest.raises(ValueError):
        subject._validate_completed_claim_chain(paths, **expected)


def test_audit_implicit_input_drift_fails_before_output(tmp_path, monkeypatch):
    args, implicit, claim_dir = _audit_drift_setup(tmp_path, monkeypatch)

    def drift_while_authenticating(**kwargs):
        implicit.write_bytes(b"changed implicit validation")
        return {}, {}, [], {}, {}

    monkeypatch.setattr(subject, "_validate_inputs", drift_while_authenticating)
    with pytest.raises(
            subject._ExplanationAuditPrivatePhaseError,
            match="^explanation_audit_private_phase_failed$"):
        subject.audit(**args)
    assert not Path(args["output"]).exists()
    assert (claim_dir / "audit_started.json").is_file()
    assert not (claim_dir / "audit_completed.json").exists()


def test_audit_permanent_anchor_drift_is_in_claim_chain_guard(
        tmp_path, monkeypatch):
    args, _, claim_dir = _audit_drift_setup(tmp_path, monkeypatch)
    anchor = subject.holdout_api._anchor_path()

    def drift_while_authenticating(**kwargs):
        anchor.write_text('{"changed":true}\n', encoding="utf-8")
        return {}, {}, [], {}, {}

    monkeypatch.setattr(subject, "_validate_inputs", drift_while_authenticating)
    with pytest.raises(
            subject._ExplanationAuditPrivatePhaseError,
            match="^explanation_audit_private_phase_failed$"):
        subject.audit(**args)
    assert not Path(args["output"]).exists()
    assert (claim_dir / "audit_started.json").is_file()
    assert not (claim_dir / "audit_completed.json").exists()


def test_audit_source_drift_during_claim_authentication_precedes_input_access(
        tmp_path, monkeypatch):
    changed = {"value": False}
    monkeypatch.setattr(
        subject, "producer_sources",
        lambda: {"audit.py": ("b" if changed["value"] else "a") * 64})

    def claim(*args, **kwargs):
        changed["value"] = True
        return tmp_path, {}

    monkeypatch.setattr(subject, "_claim_receipt", claim)
    monkeypatch.setattr(
        subject, "_regular",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("input opened after source drift")))
    with pytest.raises(RuntimeError, match="claim authentication"):
        subject.audit(
            actor_path="actor", protocol_path="protocol", program_path="program",
            rcpd_report_path="report", manifest_path="manifest",
            designation_path="designation", development_expansion_path="expansion",
            fresh_holdout_path="holdout", fresh_holdout_report_path="holdout-report",
            development_rows_paths=["rows"], output=tmp_path / "output",
            claim_receipt_path="claim", expected_claim_sha256="a" * 64,
            expected_campaign_key="b" * 64,
            expected_candidate_identity_sha256="c" * 64,
            expected_holdout_completion_sha256="d" * 64,
        )
    assert not (tmp_path / "output").exists()


def test_audit_source_drift_during_evidence_collection_leaves_no_output(
        tmp_path, monkeypatch):
    args, _, claim_dir = _audit_drift_setup(tmp_path, monkeypatch)
    state = {"drifted": False}
    frozen_sources = {"audit.py": "a" * 64}
    changed_sources = {"audit.py": "b" * 64}
    monkeypatch.setattr(
        subject, "producer_sources",
        lambda: changed_sources if state["drifted"] else frozen_sources)
    payload = {"program": "payload"}
    scenes = [{"fingerprint": f"{index:064x}"} for index in range(64)]
    monkeypatch.setattr(
        subject, "_validate_inputs",
        lambda **kwargs: ({}, {}, scenes, {
            "program_content_sha256": subject.digest(payload)},
            {"content_sha256": "e" * 64}))
    monkeypatch.setattr(
        subject, "_load_program", lambda path: (_Program(), payload))
    runtime = SimpleNamespace(
        config=SimpleNamespace(horizon=120),
        actor=SimpleNamespace(metadata={"actor_parameters_sha256": "f" * 64}),
        signature="runtime", runtime_manifest_signature="runtime-manifest",
        source_full_manifest_bindings={"source.py": "1" * 64},
    )
    monkeypatch.setattr(subject, "_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        subject, "diagnostic_runtime_sources",
        lambda: {"runtime.py": "2" * 64})

    def collect(*args, **kwargs):
        state["drifted"] = True
        return {}, {}

    monkeypatch.setattr(subject, "_collect_evidence", collect)
    with pytest.raises(
            subject._ExplanationAuditPrivatePhaseError,
            match="^explanation_audit_private_phase_failed$"):
        subject.audit(**args)
    assert not Path(args["output"]).exists()
    assert (claim_dir / "audit_started.json").is_file()
    assert not (claim_dir / "audit_completed.json").exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_direct_audit_private_failure_has_no_fresh_final_traceback_material(
        tmp_path, monkeypatch, interrupted):
    args, _, claim_dir = _audit_drift_setup(tmp_path, monkeypatch)
    private_fragments = (
        "fresh-final-seed-918273", "fresh-final-fingerprint-7f4d",
        "fresh-final-observation-c0ffee",
    )
    scenes = [{
        "seed": private_fragments[0],
        "fingerprint": private_fragments[1],
        "observation": private_fragments[2],
    }] + [{"fingerprint": f"{index:064x}"} for index in range(1, 64)]
    payload = {"program": "payload"}
    monkeypatch.setattr(
        subject, "_validate_inputs",
        lambda **kwargs: ({}, {}, scenes, {
            "program_content_sha256": subject.digest(payload)},
            {"content_sha256": "e" * 64}))
    monkeypatch.setattr(subject, "_load_program", lambda path: (_Program(), payload))
    runtime = SimpleNamespace(
        config=SimpleNamespace(horizon=120),
        actor=SimpleNamespace(metadata={"actor_parameters_sha256": "f" * 64}),
        signature="runtime", runtime_manifest_signature="runtime-manifest",
        source_full_manifest_bindings={"source.py": "1" * 64},
    )
    monkeypatch.setattr(subject, "_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        subject, "diagnostic_runtime_sources",
        lambda: {"runtime.py": "2" * 64})

    class PrivateInterrupt(BaseException):
        pass

    failure = (
        PrivateInterrupt(" | ".join(private_fragments)) if interrupted
        else RuntimeError(" | ".join(private_fragments))
    )
    monkeypatch.setattr(
        subject, "_collect_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    expected = (
        subject._ExplanationAuditPrivatePhaseInterrupt if interrupted
        else subject._ExplanationAuditPrivatePhaseError
    )
    message = (
        "explanation_audit_private_phase_interrupted" if interrupted
        else "explanation_audit_private_phase_failed"
    )
    with pytest.raises(expected, match="^" + message + "$") as raised:
        subject.audit(**args)
    _assert_private_material_absent(raised.value, private_fragments)
    assert not hasattr(subject, "_audit_sensitive")
    assert not Path(args["output"]).exists()
    assert (claim_dir / "audit_started.json").is_file()
    assert not (claim_dir / "audit_completed.json").exists()


def test_audit_claim_authentication_reuses_permanent_anchor_validation(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("permanent anchor rejected")

    monkeypatch.setattr(subject.holdout_api, "_claim_receipt", reject)
    with pytest.raises(ValueError, match="permanent anchor rejected"):
        subject._claim_receipt(
            "forged", expected_claim_sha256="a" * 64,
            expected_campaign_key="b" * 64,
            expected_candidate_identity_sha256="c" * 64,
            expected_holdout_completion_sha256="d" * 64)


def test_claim_phase_partial_payload_never_gets_final_name(tmp_path, monkeypatch):
    marker = tmp_path / "audit_started.json"

    def partial_then_fail(stream, raw):
        stream.write(raw[: max(1, len(raw) // 2)])
        stream.flush()
        raise OSError("synthetic phase payload failure")

    monkeypatch.setattr(subject, "_write_phase_payload", partial_then_fail)
    with pytest.raises(OSError, match="payload failure"):
        subject._claim_marker(tmp_path, marker.name, {
            "status": "started_no_retry", "campaign_key": "a" * 64,
        })
    assert not marker.exists()
    assert not (tmp_path / ".audit_started.json.partial").exists()


def test_claim_phase_rechecks_frozen_inputs_after_staging(tmp_path):
    marker = tmp_path / "audit_completed.json"

    def drift():
        raise RuntimeError("synthetic staged source drift")

    with pytest.raises(RuntimeError, match="staged source drift"):
        subject._claim_marker(
            tmp_path, marker.name,
            {"status": "passed", "campaign_key": "a" * 64},
            before_publish=drift)
    assert not marker.exists()
    assert not (tmp_path / ".audit_completed.json.partial").exists()


def test_runtime_source_closure_contains_transitive_physics_dependencies():
    sources = runtime_api.diagnostic_runtime_sources()
    assert {
        "env/warehouse_native/environment.py",
        "env/warehouse/transition_outcome.py",
        "env/warehouse/layouts.py",
        "env/warehouse/state_support.py",
    }.issubset(sources)
    assert "core/__init__.py" in sources
    assert not any("admission" in path or "release" in path for path in sources)


def test_candidate_artifact_binding_rejects_embedded_file_replacement(
        tmp_path):
    candidate = tmp_path / "candidate"; candidate.mkdir()
    paths = {}
    for name in subject.CANDIDATE_ARTIFACT_NAMES:
        path = candidate / name
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    frozen = {name: subject.file_hash(path) for name, path in sorted(paths.items())}
    marker = {
        "candidate_artifacts": frozen,
        "candidate_artifacts_sha256": subject.digest(frozen),
    }
    assert subject._verify_candidate_artifacts(
        marker, program_path=paths["program.json"],
        rcpd_report_path=paths["report.json"],
        row_paths=[paths["rows.npz"]]) == frozen
    paths["source_v7_report.json"].write_bytes(b"self-consistent replacement")
    with pytest.raises(ValueError, match="candidate artifacts changed"):
        subject._verify_candidate_artifacts(
            marker, program_path=paths["program.json"],
            rcpd_report_path=paths["report.json"],
            row_paths=[paths["rows.npz"]])


def test_physical_replay_rejects_runtime_source_identity_mismatch(tmp_path, monkeypatch):
    output = tmp_path / "audit"
    output.mkdir()
    evidence = output / "evidence.npz"; evidence.write_bytes(b"evidence")
    paths = {}
    for name in ("actor", "protocol", "manifest", "holdout"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    validation = tmp_path / "validation.json"
    validation.write_bytes(b"validation")
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_VALIDATION_SHA256",
        subject.file_hash(validation))
    candidate = tmp_path / "candidate"; candidate.mkdir()
    for name in subject.CANDIDATE_ARTIFACT_NAMES:
        path = candidate / name
        path.write_bytes(name.encode())
        paths[name] = path
    paths["program"] = paths["program.json"]
    holdout = {
        "version": subject.FRESH_HOLDOUT_VERSION,
        "scenes": [{"fingerprint": f"{index:064x}"} for index in range(64)],
    }
    monkeypatch.setattr(
        subject, "_read_json",
        lambda value, label: holdout if label == "fresh final holdout" else {})
    runtime = SimpleNamespace(
        signature="runtime", runtime_manifest_signature="manifest-runtime",
        actor=SimpleNamespace(metadata={"actor_parameters_sha256": "p" * 64}),
        source_full_manifest_bindings={"source.py": "s" * 64},
    )
    monkeypatch.setattr(subject, "_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        subject.manifest_binding, "read_saved_manifest",
        lambda *args, **kwargs: {"authentication": {"replay_scope": "development"}},
    )
    monkeypatch.setattr(subject, "diagnostic_runtime_sources",
                        lambda: {"runtime.py": "r" * 64})
    bindings = {
        "actor_sha256": subject.file_hash(paths["actor"]),
        "protocol_file_sha256": subject.file_hash(paths["protocol"]),
        "program_file_sha256": subject.file_hash(paths["program"]),
        "manifest_file_sha256": subject.file_hash(paths["manifest"]),
        "fresh_holdout_file_sha256": subject.file_hash(paths["holdout"]),
        "runtime_signature": "runtime",
        "runtime_manifest_signature": "manifest-runtime",
        "actor_parameters_sha256": "p" * 64,
        "source_full_manifest_bindings": {"source.py": "different"},
        "source_full_manifest_bindings_sha256": "0" * 64,
        "runtime_sources": {"runtime.py": "r" * 64},
        "runtime_sources_sha256": subject.digest({"runtime.py": "r" * 64}),
        "producer_sources_sha256": subject.digest(subject.producer_sources()),
        "candidate_authenticated_sha256": "c" * 64,
        "candidate_artifacts": {
            name: subject.file_hash(paths[name])
            for name in subject.CANDIDATE_ARTIFACT_NAMES
        },
    }
    bindings["candidate_artifacts_sha256"] = subject.digest(
        bindings["candidate_artifacts"])
    with pytest.raises(ValueError, match="runtime/source"):
        subject.replay_saved_audit(
            output, actor_path=paths["actor"], protocol_path=paths["protocol"],
            program_path=paths["program"], manifest_path=paths["manifest"],
            fresh_holdout_path=paths["holdout"],
            expected_evidence_sha256=subject.file_hash(evidence),
            expected_bindings=bindings,
        )


def test_runtime_verify_binding_rejects_changed_transitive_physics_source(
        monkeypatch):
    root = Path(runtime_api.__file__).resolve().parents[1]
    actor_path = root / (
        "output/warehouse_native/r41_active_2m_20260911/"
        "boundaries/step_2000000/actor.npz")
    protocol_path = root / "output/warehouse_native/r41_active_2m_20260911/protocol.json"
    manifest_path = root / (
        "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/"
        "manifest.json")
    if not all(path.is_file() for path in (actor_path, protocol_path, manifest_path)):
        pytest.skip("diagnostic runtime development artifacts are not present")
    protocol = subject._read_json(protocol_path, "training protocol")
    manifest = subject._read_json(manifest_path, "conflict manifest")
    content = dict(manifest); content_sha = content.pop("content_sha256")
    runtime = runtime_api.R41DiagnosticOnlineAlignmentRuntime(
        actor_path, training_protocol_path=protocol_path, manifest_path=manifest_path,
        expected_actor_sha256=subject.file_hash(actor_path),
        expected_training_protocol_file_sha256=subject.file_hash(protocol_path),
        expected_training_protocol_content_sha256=subject.digest(protocol),
        expected_manifest_file_sha256=subject.file_hash(manifest_path),
        expected_manifest_content_sha256=content_sha,
        expected_manifest_semantic_sha256=subject.digest(manifest),
    )
    original = runtime_api.diagnostic_runtime_sources()
    calls = []
    monkeypatch.setattr(
        runtime_api, "diagnostic_runtime_sources",
        lambda: calls.append("full-hash") or dict(original))
    started = time.perf_counter()
    for _ in range(20):
        runtime.verify_binding()
    assert calls == []
    assert time.perf_counter() - started < 1.0

    changed = dict(original)
    changed["env/warehouse/transition_outcome.py"] = "0" * 64
    monkeypatch.setattr(runtime_api, "diagnostic_runtime_sources", lambda: changed)
    identities = dict(runtime._source_file_identities)
    identities["env/warehouse/transition_outcome.py"] = (0, 0, 0, 0, 0)
    monkeypatch.setattr(runtime_api, "_runtime_source_identities", lambda _: identities)
    with pytest.raises(ValueError, match="runtime binding changed"):
        runtime.verify_binding()
