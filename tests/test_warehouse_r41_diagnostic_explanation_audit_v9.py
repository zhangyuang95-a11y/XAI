from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_explanation_audit_v9 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def test_final_audit_uses_v11_row_and_projection_contracts():
    assert subject.collection_api.VERSION == (
        "warehouse-r41-diagnostic-outer-collection.v11")
    assert subject.projection_api.VERSION == (
        "warehouse-r41-diagnostic-outer-hash-projection.v11")


class _Actor:
    def __init__(self, _path):
        self.metadata = {"feature_names": [f"feature-{i}" for i in range(197)]}


class _Program:
    action_names = tuple(subject.metrics_api.ACTIONS)
    base_feature_names = tuple(f"feature-{i}" for i in range(197))

    @classmethod
    def from_dict(cls, _payload):
        return cls()

    def predict_proba_batch(self, observations):
        result = np.zeros((len(observations), 5), dtype=np.float64)
        result[np.arange(len(observations)), observations[:, 0].astype(int)] = 1.0
        return result


def _fixture(tmp_path: Path, monkeypatch):
    actor = tmp_path / "actor.npz"
    actor.write_bytes(b"actor")
    program = tmp_path / "program.json"
    program.write_text(canonical({"program": "synthetic"}) + "\n", encoding="utf-8")
    scenes = [{
        "id": f"final-{index}", "seed": 90_000 + index,
        "family_id": f"family-{index % 6}", "fingerprint": _fp(f"scene-{index}"),
    } for index in range(subject.FINAL_SCENE_COUNT)]
    rows = []
    for scene_index, scene in enumerate(scenes):
        for partner in subject.rows_api.PARTNERS:
            rows.append((scene, f"{scene['id']}:{partner}", 0, "ordinary", "", "",
                         scene_index % 5, 0, "", _fp(f"ordinary-state-{scene_index}-{partner}")))
        group = scene_index % 3
        anchor = f"{scene['id']}:anchor"
        episode = f"{scene['id']}:{subject.rows_api.PARTNERS[0]}"
        rows.append((scene, episode, 1, "intervention", anchor, "WAIT", 4,
                     1 << group, _fp(f"physical-wait-{scene_index}"),
                     _fp(f"state-wait-{scene_index}")))
        rows.append((scene, episode, 1, "intervention", anchor, "UP", 0,
                     1 << group, _fp(f"physical-up-{scene_index}"),
                     _fp(f"state-up-{scene_index}")))
    count = len(rows)
    observations = np.zeros((count, 197), dtype=np.float32)
    labels = np.asarray([row[6] for row in rows], dtype=np.uint8)
    observations[:, 0] = labels
    probabilities = np.zeros((count, 5), dtype=np.float32)
    probabilities[np.arange(count), labels] = 1.0
    arrays = {
        "observations": observations,
        "probabilities": probabilities,
        "action_indices": labels,
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": np.asarray([
            subject.rows_api.legacy._obs_hash(row) for row in observations
        ], dtype="S64"),
        "scene_fingerprints": np.asarray(
            [row[0]["fingerprint"] for row in rows], dtype="S64"),
        "episode_ids": np.asarray([row[1] for row in rows], dtype="S180"),
        "frames": np.asarray([row[2] for row in rows], dtype=np.int16),
        "group_bits": np.asarray([row[7] for row in rows], dtype=np.uint8),
        "kinds": np.asarray([row[3] for row in rows], dtype="S16"),
        "anchor_ids": np.asarray([row[4] for row in rows], dtype="S240"),
        "branch_actions": np.asarray([row[5] for row in rows], dtype="S8"),
        "physical_hashes": np.asarray([row[8] for row in rows], dtype="S64"),
        "source_state_hashes": np.asarray([row[9] for row in rows], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.asarray(
            [row[3] == "ordinary" for row in rows], dtype=np.bool_),
        "split_validation": np.ones(count, dtype=np.bool_),
    }
    # Repeated labels intentionally produce repeated observations inside final;
    # the hard boundary is zero overlap across splits, not uniqueness within an episode.
    monkeypatch.setattr(subject, "NumPyNativeActor", _Actor)
    monkeypatch.setattr(subject, "R41DiagnosticPublicTreeProgramV9", _Program)
    monkeypatch.setattr(
        subject.projection_api, "_validate_projection_replay_arrays",
        lambda *a, **k: None)
    monkeypatch.setattr(subject, "producer_sources", lambda: {"audit.py": _fp("audit")})
    bindings = {
        "actor_sha256": file_hash(actor), "program_sha256": file_hash(program),
    }
    return actor, program, scenes, arrays, bindings


def _run(tmp_path, monkeypatch, **overrides):
    actor, program, scenes, arrays, bindings = _fixture(tmp_path, monkeypatch)
    values = {
        "actor_path": actor, "program_path": program,
        "program_payload": {"program": "synthetic"},
        "program_sha256": bindings["program_sha256"], "arrays": arrays,
        "replay_arrays": deepcopy(arrays), "scenes": scenes,
        "development_observation_hashes": {_fp("development-observation")},
        "outer_observation_hashes": {_fp("outer-observation")},
        "development_scene_fingerprints": {_fp("development-scene")},
        "outer_scene_fingerprints": {_fp("outer-scene")},
        "bindings": bindings, "environment_steps": 100,
        "replay_environment_steps": 100,
    }
    values.update(overrides)
    return subject.audit_rows(**values), arrays, bindings


def test_passes_exact_nine_gates_and_physical_replay(tmp_path, monkeypatch):
    report, _arrays, bindings = _run(tmp_path, monkeypatch)
    assert report["status"] == subject.STATUS_PASSED
    assert report["nine_gate"]["passed"] is True
    assert len(report["nine_gate"]["checks"]) == 9
    assert report["physical_counterfactual_audit"]["independent_replay_exact"] is True
    assert report["action_authority"]["program_controls_runtime_actions"] is False
    assert subject.validate_report(
        report, expected_bindings=bindings, require_passed=True) == report


def test_rejects_nonidentical_physical_replay(tmp_path, monkeypatch):
    actor, program, scenes, arrays, bindings = _fixture(tmp_path, monkeypatch)
    replay = deepcopy(arrays)
    replay["frames"][0] = 2
    with pytest.raises(ValueError, match="Physical replay differs"):
        subject.audit_rows(
            actor_path=actor, program_path=program,
            program_payload={"program": "synthetic"},
            program_sha256=bindings["program_sha256"], arrays=arrays,
            replay_arrays=replay, scenes=scenes,
            development_observation_hashes=set(), outer_observation_hashes=set(),
            development_scene_fingerprints=set(), outer_scene_fingerprints=set(),
            bindings=bindings, environment_steps=1, replay_environment_steps=1)


def test_rejects_any_cross_split_observation_overlap(tmp_path, monkeypatch):
    actor, program, scenes, arrays, bindings = _fixture(tmp_path, monkeypatch)
    overlap = arrays["observation_hashes"][0].decode("ascii")
    with pytest.raises(ValueError, match="overlap prior splits"):
        subject.audit_rows(
            actor_path=actor, program_path=program,
            program_payload={"program": "synthetic"},
            program_sha256=bindings["program_sha256"], arrays=arrays,
            replay_arrays=deepcopy(arrays), scenes=scenes,
            development_observation_hashes={overlap}, outer_observation_hashes=set(),
            development_scene_fingerprints=set(), outer_scene_fingerprints=set(),
            bindings=bindings, environment_steps=1, replay_environment_steps=1)


def test_rejects_action_authority_tampering(tmp_path, monkeypatch):
    actor, program, scenes, arrays, bindings = _fixture(tmp_path, monkeypatch)
    arrays["submitted_equal"][0] = False
    with pytest.raises(ValueError, match="action-authority"):
        subject.audit_rows(
            actor_path=actor, program_path=program,
            program_payload={"program": "synthetic"},
            program_sha256=bindings["program_sha256"], arrays=arrays,
            replay_arrays=deepcopy(arrays), scenes=scenes,
            development_observation_hashes=set(), outer_observation_hashes=set(),
            development_scene_fingerprints=set(), outer_scene_fingerprints=set(),
            bindings=bindings, environment_steps=1, replay_environment_steps=1)


def test_strict_reader_rejects_metric_tampering(tmp_path, monkeypatch):
    report, _arrays, bindings = _run(tmp_path, monkeypatch)
    report["metrics"]["overall"]["fidelity"] = 0.0
    report["content_sha256"] = digest({
        key: value for key, value in report.items() if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="nine-gate"):
        subject.validate_report(report, expected_bindings=bindings, require_passed=True)
