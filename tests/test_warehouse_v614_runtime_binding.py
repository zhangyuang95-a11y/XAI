"""The v6.14 public-action encoder is bound to the deployed physics."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from domains.warehouse import neural_turnbased as warehouse
from env.warehouse.domain import participant_study_config
from env.warehouse.public_action_features_v614 import (
    ACTION_PHYSICS_CONFIG_SHA256, validate_study_action_physics,
)


def test_v614_configuration_and_saved_state_reject_changed_physics(
        monkeypatch, tmp_path):
    config = json.loads(warehouse.V6_CONFIG_PATH.read_text())
    assert config["action_physics_config_sha256"] == ACTION_PHYSICS_CONFIG_SHA256
    assert warehouse._v6_config()["action_physics_config_sha256"] == (
        ACTION_PHYSICS_CONFIG_SHA256)

    changed = tmp_path / "changed-physics.json"
    changed.write_text(json.dumps({**config,
        "action_physics_config_sha256": "0" * 64}))
    monkeypatch.setattr(warehouse, "V6_CONFIG_PATH", changed)
    with pytest.raises(RuntimeError, match="configuration schema mismatch"):
        warehouse._v6_config()

    monkeypatch.setattr(warehouse, "_verify_v5_artifact", lambda _: None)
    state = {"policy_id": "warehouse-raw-v6",
             "release_id": warehouse.V6_RELEASE_ID,
             "feature_schema_sha256": warehouse.V6_FEATURE_SCHEMA_SHA256,
             "action_physics_config_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="state release or feature binding mismatch"):
        warehouse._restore(state)


def test_v614_task_and_demonstration_reject_physics_drift_before_reset(
        monkeypatch):
    monkeypatch.setattr(warehouse, "_verify_v5_artifact", lambda _: None)
    monkeypatch.setattr(warehouse, "_actor", lambda _: SimpleNamespace(
        artifact_sha256="frozen-actor-hash"))
    monkeypatch.setattr(warehouse, "participant_study_config", lambda:
                        participant_study_config(move_battery_cost=3.0))
    with pytest.raises(ValueError, match="action physics differs"):
        validate_study_action_physics(participant_study_config(
            charge_per_wait=8.0))
    with pytest.raises(ValueError, match="action physics differs"):
        warehouse.initial_state(1000, 1, "warehouse-raw-v6")
    warehouse.demonstration.cache_clear()
    with pytest.raises(ValueError, match="action physics differs"):
        warehouse.demonstration("warehouse-regularized-v6")
