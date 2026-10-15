"""The Warehouse v6 adapter rejects incomplete release bindings directly."""
from __future__ import annotations

import pytest

from domains.warehouse import neural_turnbased as warehouse
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
