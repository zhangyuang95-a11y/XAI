"""The Warehouse v6 adapter rejects incomplete release bindings directly."""
from __future__ import annotations

import pytest
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np

from domains.warehouse import neural_turnbased as warehouse


def test_successor_policy_uses_its_own_release_and_stays_pending():
    release_id = "policylens-three-domain-20261016.v6.1"
    assert warehouse._v6_identity("warehouse-regularized-v6.1") == (
        "regularized", release_id)
    assert warehouse._config_for_policy("warehouse-raw-v6.1")[
        "release_id"] == release_id
    with pytest.raises(RuntimeError, match="release_id mismatch"):
        warehouse.initial_state(1000, 1, "warehouse-raw-v6.1",
                                release_id=warehouse.V6_RELEASE_ID)
    with pytest.raises(RuntimeError, match="unavailable|accepted"):
        warehouse.initial_state(1000, 1, "warehouse-raw-v6.1",
                                release_id=release_id)
    assert warehouse.policy_artifacts_v6(release_id)["regularized"][
        "readiness"] is False


def test_legacy_pending_config_remains_readable_without_promoting_old_actor(
        monkeypatch, tmp_path):
    current = json.loads(warehouse.V6_CONFIG_PATH.read_text())
    assert current["model_version"] == warehouse.V6_LEGACY_MODEL_VERSION
    assert warehouse._v6_config()["model_version"] == current["model_version"]

    successor = {**current, "version": warehouse.V6_MODEL_VERSION,
                 "model_version": warehouse.V6_MODEL_VERSION,
                 "feature_schema_sha256": warehouse.V6_FEATURE_SCHEMA_SHA256}
    successor_path = tmp_path / "v616.json"
    successor_path.write_text(json.dumps(successor))
    monkeypatch.setattr(warehouse, "V6_CONFIG_PATH", successor_path)
    assert warehouse._v6_config()["model_version"] == warehouse.V6_MODEL_VERSION

    successor["version"] = warehouse.V6_LEGACY_MODEL_VERSION
    successor_path.write_text(json.dumps(successor))
    with pytest.raises(RuntimeError, match="configuration schema mismatch"):
        warehouse._v6_config()


def test_v6_runtime_rejects_artifact_model_version_drift(monkeypatch):
    policy_id = "warehouse-raw-v6"
    entry = {"readiness": True, "actor_path": "unused.npz",
             "actor_sha256": "a" * 64,
             "role_sha256": {"human": "b" * 64, "ai": "c" * 64},
             "model_version": warehouse.V6_MODEL_VERSION,
                 "feature_schema_sha256": warehouse.V6_FEATURE_SCHEMA_SHA256}
    monkeypatch.setattr(warehouse, "policy_artifacts_v6",
                        lambda _: {"raw": entry})
    monkeypatch.setattr(warehouse.FrozenWarehouseV6Actor, "load",
                        lambda *_args, **_kwargs: SimpleNamespace(
                            model_version=warehouse.V6_LEGACY_MODEL_VERSION))
    warehouse._actor.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="model version differs from release"):
            warehouse._actor(policy_id)
    finally:
        warehouse._actor.cache_clear()
from experiments.warehouse import protocol
from study_v3 import release_catalog


def test_pending_v6_catalog_closes_direct_entrypoints(monkeypatch):
    def pending(*args, **kwargs):
        raise release_catalog.ReleaseAssetUnavailable("warehouse_release_pending")

    monkeypatch.setattr(release_catalog, "warehouse_policy_artifacts", pending)
    inventory = warehouse.policy_artifacts_v6()
    assert inventory["raw"]["readiness"] is False
    assert inventory["regularized"]["readiness"] is False
    for policy_id in ("warehouse-raw-v6", "warehouse-regularized-v6"):
        with pytest.raises(RuntimeError, match="accepted policy binding is unavailable"):
            warehouse.initial_state(1000, 1, policy_id,
                                    release_id=warehouse.V6_RELEASE_ID)
        with pytest.raises(RuntimeError, match="accepted policy binding is unavailable"):
            warehouse.demonstration(policy_id, release_id=warehouse.V6_RELEASE_ID)


def test_v6_release_identity_is_required_for_v6_policy():
    with pytest.raises(RuntimeError, match="release_id mismatch"):
        warehouse.initial_state(1000, 1, "warehouse-raw-v6",
                                release_id="another-release")
    with pytest.raises(RuntimeError, match="requires a v6 policy_id"):
        warehouse.initial_state(1000, 1, "warehouse-raw-v5",
                                release_id=warehouse.V6_RELEASE_ID)
    with pytest.raises(RuntimeError, match="release_id mismatch"):
        warehouse.demonstration("warehouse-regularized-v6",
                                release_id="another-release")


def test_explanation_audit_uses_state_release_identity_and_sealed_block(monkeypatch):
    later = "policylens-three-domain-20261016.v6.1"
    expected = release_catalog.warehouse_release_spec(later)
    inventory = warehouse.policy_artifacts(later)
    assert inventory["regularized"]["policy_id"] == expected.policy_ids["regularized"]
    assert inventory["regularized"]["readiness"] is False
    calls = []
    real_seeds = protocol.final_audit_seeds

    def seeds(release_id):
        calls.append(("audit", release_id))
        return real_seeds(release_id)

    def pending(release_id):
        calls.append(("catalog", release_id))
        return {"regularized": {"readiness": False}}

    monkeypatch.setattr(protocol, "final_audit_seeds", seeds)
    monkeypatch.setattr(warehouse, "policy_artifacts_v6", pending)
    state = {"terminal": False, "release_id": later,
             "policy_id": expected.policy_ids["regularized"]}
    assert warehouse._audited_v6_program_evidence(state) is None
    assert calls == [("audit", later), ("catalog", later)]
    assert real_seeds(later) != real_seeds(warehouse.V6_RELEASE_ID)

    calls.clear()
    state["policy_id"] = "warehouse-regularized-v6"
    assert warehouse._audited_v6_program_evidence(state) is None
    assert calls == [("audit", later)]


def test_audited_explanation_supplies_the_bound_frozen_ai_role(monkeypatch, tmp_path):
    """An accepted program must inspect the same Actor role used for execution."""
    release_id = warehouse.V6_RELEASE_ID
    policy_id = "warehouse-regularized-v6"
    episode_ids = [str(seed) for seed in protocol.final_audit_seeds(release_id)]
    program_path = tmp_path / "program.json"
    program_path.write_text("{}")
    program_sha = sha256(program_path.read_bytes()).hexdigest()
    receipt = {
        "passed": True, "heldout_evaluated": True, "release_id": release_id,
        "artifact_sha256": {"regularized": "actor-sha"},
        "role_sha256": {"regularized": {"robot_2": "ai-sha"}},
        "program_sha256": {"regularized": {"robot_2": program_sha}},
        "final_program_audit": {"robot_2": {
            "actor_role_sha256": "ai-sha", "program_sha256": program_sha,
            "feature_schema_sha256": warehouse.V6_FEATURE_SCHEMA_SHA256,
            "episode_ids": episode_ids, "sample_count": 1200,
            "explanation_eligible": True, "max_depth": 5,
            "leaf_count": 16, "max_features_per_node": 3,
            "explanation": {"coverage": .31, "actionable_coverage": .30,
                            "agreement": .96, "kl": .10, "illegal_count": 0},
        }},
    }
    report = {"selected_role_sha256": {"robot_2": "ai-sha"}}
    (tmp_path / "acceptance.json").write_text(json.dumps(receipt))
    (tmp_path / "report.json").write_text(json.dumps(report))
    entry = {
        "readiness": True, "acceptance_path": "acceptance.json",
        "training_report_path": "report.json", "program_paths": ["unused", "program.json"],
        "program_sha256": {"robot_2": program_sha}, "actor_sha256": "actor-sha",
        "role_sha256": {"ai": "ai-sha"},
    }
    role = object()
    actor = SimpleNamespace(artifact_sha256="actor-sha",
                            role_sha256={"robot_2": "ai-sha"},
                            roles={"robot_2": role})
    seen = []

    class Program:
        @staticmethod
        def load(*args, **kwargs):
            return Program()

        def covered(self, sample, *, actor_action, purpose, actor_role):
            seen.append((actor_action, purpose, actor_role))
            return {"path": [], "leaf_id": 0}

    monkeypatch.setattr(warehouse, "ROOT", tmp_path)
    monkeypatch.setattr(warehouse, "policy_artifacts_v6",
                        lambda _: {"regularized": entry})
    monkeypatch.setattr(warehouse, "WarehouseV6Program", Program)
    monkeypatch.setattr(warehouse, "_restore", lambda _:
                        SimpleNamespace(state=object(), config=SimpleNamespace(map_layout_id="map")))
    monkeypatch.setattr(warehouse, "v6_role_observation", lambda *_: np.zeros(1))
    monkeypatch.setattr(warehouse, "action_masks", lambda *_: np.ones((2, 5)))
    monkeypatch.setattr(warehouse, "_selection", lambda *_: ("WAIT", {}, {}))
    monkeypatch.setattr(warehouse, "_actor", lambda _: actor)
    state = {"terminal": False, "release_id": release_id,
             "policy_id": policy_id, "scenario_seed": 7}
    evidence = warehouse._audited_v6_program_evidence(state)
    assert evidence is not None and evidence["nn_action"] == "wait"
    assert seen == [("WAIT", "explanation", role)]

    actor.role_sha256["robot_2"] = "different-sha"
    assert warehouse._audited_v6_program_evidence(state) is None
    assert seen == [("WAIT", "explanation", role)]
