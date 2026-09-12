import ast
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as subject
from backend import warehouse_r41_diagnostic_online_runtime as runtime_api


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
    paths["prior_v7_report.json"].write_bytes(b"self-consistent replacement")
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
        lambda value, label: holdout if Path(value) == paths["holdout"] else {})
    runtime = SimpleNamespace(
        signature="runtime", runtime_manifest_signature="manifest-runtime",
        actor=SimpleNamespace(metadata={"actor_parameters_sha256": "p" * 64}),
        source_full_manifest_bindings={"source.py": "s" * 64},
    )
    monkeypatch.setattr(subject, "_runtime", lambda *args, **kwargs: runtime)
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
