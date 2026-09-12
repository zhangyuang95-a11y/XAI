import ast
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as subject


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


def test_physical_replay_rejects_runtime_source_identity_mismatch(tmp_path, monkeypatch):
    output = tmp_path / "audit"
    output.mkdir()
    evidence = output / "evidence.npz"; evidence.write_bytes(b"evidence")
    paths = {}
    for name in ("actor", "protocol", "program", "manifest", "holdout"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
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
    }
    with pytest.raises(ValueError, match="runtime/source"):
        subject.replay_saved_audit(
            output, actor_path=paths["actor"], protocol_path=paths["protocol"],
            program_path=paths["program"], manifest_path=paths["manifest"],
            fresh_holdout_path=paths["holdout"],
            expected_evidence_sha256=subject.file_hash(evidence),
            expected_bindings=bindings,
        )
