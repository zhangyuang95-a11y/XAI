"""The Warehouse v6 adapter rejects incomplete release bindings directly."""
from __future__ import annotations

import pytest

from domains.warehouse import neural_turnbased as warehouse
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
