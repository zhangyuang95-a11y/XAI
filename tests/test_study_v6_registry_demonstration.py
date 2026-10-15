"""The public AI-AI demo binds each Warehouse candidate to its own release."""
from __future__ import annotations

import pytest

from study_v3 import RELEASE_ID, policies, registry


FUTURE_RELEASE = "policylens-three-domain-20261016.v6.1"


def test_candidate_demo_uses_exact_release_and_rechecks_cached_acceptance(monkeypatch):
    entries = {
        RELEASE_ID: {variant: {"policy_id": f"warehouse-{variant}-v6",
                              "actor_sha256": "a" * 64}
                     for variant in ("raw", "regularized")},
        FUTURE_RELEASE: {variant: {"policy_id": f"warehouse-{variant}-v6.1",
                                  "actor_sha256": "b" * 64}
                         for variant in ("raw", "regularized")},
    }
    available = {RELEASE_ID: True, FUTURE_RELEASE: True}
    audits, rollouts = [], []

    class FakeEngine:
        def demonstration(self, policy_id, *, release_id):
            rollouts.append((policy_id, release_id))
            return {"frames": [{"release_id": release_id,
                                "policy_id": policy_id}]}

    def catalog_entry(domain, variant, release_id=RELEASE_ID):
        assert domain == "warehouse"
        return entries[release_id][variant]

    def binding(domain, group, release_id=RELEASE_ID):
        variant = "raw" if group == "C" else "regularized"
        entry = entries[release_id][variant]
        return {"policy_id": entry["policy_id"], "domain": domain,
                "variant": variant, "actor_sha256": entry["actor_sha256"]}

    def accepted(entry, bound, release_id=RELEASE_ID):
        audits.append((entry["policy_id"], release_id))
        return (available[release_id]
                and entry["policy_id"] == bound["policy_id"]
                and entry["actor_sha256"] == bound["actor_sha256"])

    monkeypatch.setattr(registry, "engine", lambda domain: FakeEngine())
    monkeypatch.setattr(policies, "_catalog_entry", catalog_entry)
    monkeypatch.setattr(policies, "binding_for_group", binding)
    monkeypatch.setattr(policies, "_accepted", accepted)
    registry._accepted_demonstration.cache_clear()
    try:
        old_id, new_id = "warehouse-regularized-v6", "warehouse-regularized-v6.1"
        old = registry.demonstration("warehouse", old_id)
        new = registry.demonstration("warehouse", new_id)
        assert old["frames"][0]["release_id"] == RELEASE_ID
        assert new["frames"][0]["release_id"] == FUTURE_RELEASE
        assert registry.demonstration("warehouse", new_id) == new
        assert rollouts == [(old_id, RELEASE_ID), (new_id, FUTURE_RELEASE)]
        assert audits[-1] == (new_id, FUTURE_RELEASE)

        available[FUTURE_RELEASE] = False
        with pytest.raises(policies.PolicyUnavailable, match="policy_not_accepted"):
            registry.demonstration("warehouse", new_id)
        assert len(rollouts) == 2
    finally:
        registry._accepted_demonstration.cache_clear()


@pytest.mark.parametrize("policy_id", (
    "warehouse-raw-v6.00", "warehouse-regularized-v6.01",
    "warehouse-raw-v6.-1"))
def test_noncanonical_candidate_demo_id_is_rejected(policy_id, monkeypatch):
    called = []

    def catalog_entry(domain, variant):
        called.append((domain, variant))
        return {"policy_id": f"warehouse-{variant}-v6"}

    monkeypatch.setattr(policies, "_catalog_entry", catalog_entry)
    with pytest.raises(ValueError, match="invalid_policy_id"):
        registry.demonstration("warehouse", policy_id)
    assert called
