"""Direct Warehouse v5 entrypoints must honor the full release gate."""

from copy import deepcopy

import pytest
import torch

from backend.training.warehouse_dual_v5 import DualNN, export_actor
from backend.training.warehouse_pure_nn import PureNN, export_actor as export_legacy_actor
from domains.warehouse import neural_turnbased as warehouse
from study_v3 import policies, registry


def test_direct_v5_calls_recheck_central_acceptance_on_existing_state(
        tmp_path, monkeypatch):
    torch.manual_seed(334)
    model = DualNN(hidden=32)
    artifact = export_actor(model, tmp_path / "pair.npz", pair_seed=334,
                            variant="raw")
    config = deepcopy(warehouse.V5_CONFIG)
    config["variants"]["raw"].update(
        actor_path=artifact["path"], actor_sha256=artifact["sha256"],
        role_sha256=artifact["role_sha256"])
    monkeypatch.setattr(warehouse, "V5_CONFIG", config)
    inventory = {"readiness": True, "policy_id": "warehouse-raw-v5"}
    monkeypatch.setattr(warehouse, "policy_artifacts", lambda: {"raw": inventory})
    accepted = {"value": True}
    calls = []
    def binding_for_group(domain, group):
        calls.append((domain, group))
        return {"policy_id": "warehouse-raw-v5"}
    monkeypatch.setattr(policies, "binding_for_group", binding_for_group)
    monkeypatch.setattr(policies, "_accepted",
                        lambda entry, binding: accepted["value"] and
                        entry is inventory and binding["policy_id"] == "warehouse-raw-v5")
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    try:
        state = warehouse.initial_state(77, 1, policy_id="warehouse-raw-v5")
        assert calls and set(calls) == {("warehouse", "C")}
        assert warehouse.demonstration("warehouse-raw-v5")["frames"]
        accepted["value"] = False  # e.g. missing final-audit or held-out receipt
        terminal_state = {**state, "terminal": True}
        for operation in (
            lambda: warehouse.initial_state(78, 1, policy_id="warehouse-raw-v5"),
            lambda: warehouse.decide(state),
            lambda: warehouse.decide(terminal_state),
            lambda: warehouse.step(state, "wait"),
            lambda: warehouse.demonstration("warehouse-raw-v5"),
        ):
            with pytest.raises(RuntimeError, match="accepted policy binding is unavailable"):
                operation()
        accepted["value"] = True
        inventory["readiness"] = False  # missing/mismatched receipt, report or program
        with pytest.raises(RuntimeError, match="accepted policy binding is unavailable"):
            warehouse.decide(state)
    finally:
        warehouse._actor.cache_clear()
        warehouse.demonstration.cache_clear()


def test_v4_legacy_entrypoint_does_not_require_v5_release_receipt(tmp_path, monkeypatch):
    checkpoint = tmp_path / "legacy-checkpoint.bin"
    checkpoint.write_bytes(b"test-only-checkpoint")
    torch.manual_seed(884)
    artifact = export_legacy_actor(
        PureNN(hidden=8), tmp_path / "legacy-actor.npz",
        training_seed=884, training_episodes=0, checkpoint_path=checkpoint)
    config = deepcopy(warehouse.CONFIG)
    config.update(actor_path="legacy-actor.npz", actor_sha256=artifact["sha256"])
    monkeypatch.setattr(warehouse, "ROOT", tmp_path)
    monkeypatch.setattr(warehouse, "CONFIG", config)
    monkeypatch.setattr(policies, "_accepted", lambda *_: (_ for _ in ()).throw(
        AssertionError("v4 should not enter v5 release gate")))
    warehouse._actor.cache_clear()
    try:
        state = warehouse.initial_state(77, 1)
        assert state.get("policy_id") is None
        assert warehouse.decide(state)["action"] in warehouse.legal_actions(state)
    finally:
        warehouse._actor.cache_clear()


@pytest.mark.parametrize("failure", ("removed_receipt", "actor_hash_mismatch"))
def test_registry_cached_v5_demo_rechecks_binding_before_serving_frames(
        monkeypatch, failure):
    """A cached demonstration cannot survive lost or rebound release evidence."""
    policy_id = "warehouse-raw-v5"
    entry = {"policy_id": policy_id, "actor_sha256": "a" * 64}
    binding = {"policy_id": policy_id, "actor_sha256": "a" * 64}
    receipt = {"present": True}
    calls = []

    class FakeEngine:
        def demonstration(self, requested_policy_id, *, release_id=None):
            calls.append(requested_policy_id)
            return {"frames": [{"rollout": len(calls)}]}

    monkeypatch.setattr(registry, "engine", lambda domain: FakeEngine())
    monkeypatch.setattr(policies, "_catalog_entry", lambda domain, variant: entry)
    monkeypatch.setattr(policies, "binding_for_group", lambda domain, group: binding)
    monkeypatch.setattr(
        policies, "_accepted",
        lambda current_entry, current_binding: (
            receipt["present"]
            and current_entry["actor_sha256"] == current_binding["actor_sha256"]),
    )
    registry._accepted_demonstration.cache_clear()
    try:
        assert registry.demonstration("warehouse", policy_id)["frames"] == [{"rollout": 1}]
        assert registry.demonstration("warehouse", policy_id)["frames"] == [{"rollout": 1}]
        assert calls == [policy_id]
        if failure == "removed_receipt":
            receipt["present"] = False
        else:
            entry["actor_sha256"] = "b" * 64
        with pytest.raises(policies.PolicyUnavailable, match="policy_not_accepted"):
            registry.demonstration("warehouse", policy_id)
        assert calls == [policy_id]
    finally:
        registry._accepted_demonstration.cache_clear()


def test_registry_cached_v5_demo_keys_rollout_by_verified_binding(monkeypatch):
    policy_id = "warehouse-raw-v5"
    entry = {"policy_id": policy_id, "actor_sha256": "a" * 64}
    binding = {"policy_id": policy_id, "actor_sha256": "a" * 64}
    calls = []

    class FakeEngine:
        def demonstration(self, requested_policy_id, *, release_id=None):
            calls.append(requested_policy_id)
            return {"frames": [{"rollout": len(calls)}]}

    monkeypatch.setattr(registry, "engine", lambda domain: FakeEngine())
    monkeypatch.setattr(policies, "_catalog_entry", lambda domain, variant: entry)
    monkeypatch.setattr(policies, "binding_for_group", lambda domain, group: binding)
    monkeypatch.setattr(
        policies, "_accepted",
        lambda current_entry, current_binding: (
            current_entry["actor_sha256"] == current_binding["actor_sha256"]),
    )
    registry._accepted_demonstration.cache_clear()
    try:
        assert registry.demonstration("warehouse", policy_id)["frames"] == [{"rollout": 1}]
        entry["actor_sha256"] = binding["actor_sha256"] = "b" * 64
        assert registry.demonstration("warehouse", policy_id)["frames"] == [{"rollout": 2}]
        assert calls == [policy_id, policy_id]
    finally:
        registry._accepted_demonstration.cache_clear()
