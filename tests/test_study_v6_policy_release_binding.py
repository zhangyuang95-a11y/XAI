"""Candidate Warehouse releases cannot reuse another version's policy gate."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256, MODEL_VERSION
from env.warehouse.public_action_features_v614 import ACTION_PHYSICS_CONFIG_SHA256
from study_v3 import policies
from study_v3 import release_catalog as catalog


FUTURE_RELEASE = "policylens-three-domain-20261016.v6.1"


def _candidate_catalog(root, *, status: str, forged_actor_hash: bool = False):
    spec = catalog.warehouse_release_spec(FUTURE_RELEASE)
    folder = root / "releases" / FUTURE_RELEASE
    folder.mkdir(parents=True)
    entries = {}
    for variant in ("raw", "regularized"):
        actor = f"{variant}-actor".encode()
        actual = sha256(actor).hexdigest()
        claimed = "0" * 64 if forged_actor_hash else actual
        actor_path = f"releases/{FUTURE_RELEASE}/assets/{claimed}/{variant}.npz"
        path = root / actor_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(actor)
        entries[variant] = {
            "policy_id": spec.policy_ids[variant],
            "actor_path": actor_path, "actor_sha256": claimed,
            "model_version": MODEL_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
            "role_sha256": {"human": "1" * 64, "ai": "2" * 64},
            "program_paths": [], "program_sha256": {},
            "training_report_path": actor_path,
            "training_report_sha256": claimed,
            "acceptance_path": actor_path, "acceptance_sha256": claimed,
        }
    (folder / "catalog.json").write_text(json.dumps({
        "format": "policylens-release-catalog.v1",
        "release_id": FUTURE_RELEASE,
        "upstream_release_id": catalog.UPSTREAM_RELEASE_ID,
        "warehouse": {"status": status, "policy_ids": spec.policy_ids},
        "policies": {"warehouse": entries},
    }))


def test_v61_pending_pair_cannot_bind(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    _candidate_catalog(tmp_path, status="pending")
    with pytest.raises(policies.PolicyUnavailable, match="policy_catalog_unavailable"):
        policies.binding_for_group("warehouse", "A", release_id=FUTURE_RELEASE)
    with pytest.raises(policies.PolicyUnavailable, match="policy_catalog_unavailable"):
        policies.policy_id("warehouse", "C", release_id=FUTURE_RELEASE)


def test_v61_forged_actor_hash_cannot_bind(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    _candidate_catalog(tmp_path, status="accepted", forged_actor_hash=True)
    with pytest.raises(policies.PolicyUnavailable, match="policy_catalog_unavailable"):
        policies.binding_for_group("warehouse", "C", release_id=FUTURE_RELEASE)


def test_v61_binding_and_acceptance_use_exact_release(tmp_path, monkeypatch):
    raw = tmp_path / "raw.npz"
    raw.write_bytes(b"candidate actor")
    actor_sha = sha256(raw.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text("{}")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"source_closure_sha256": "f" * 64}))
    entry = {
        "policy_id": catalog.warehouse_release_spec(FUTURE_RELEASE).policy_ids["raw"],
        "actor_path": raw.name, "actor_sha256": actor_sha,
        "model_version": MODEL_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
        "role_sha256": {"human": "1" * 64, "ai": "2" * 64},
        "program_paths": [], "program_sha256": {},
        "training_report_path": report.name,
        "training_report_sha256": sha256(report.read_bytes()).hexdigest(),
        "acceptance_path": acceptance.name,
        "acceptance_sha256": sha256(acceptance.read_bytes()).hexdigest(),
    }
    calls = []

    def artifacts(release_id=catalog.RELEASE_ID):
        calls.append(("artifacts", release_id))
        assert release_id == FUTURE_RELEASE
        return {"raw": entry, "regularized": {}}

    def snapshot(release_id=catalog.RELEASE_ID):
        calls.append(("snapshot", release_id))
        assert release_id == FUTURE_RELEASE
        return {}, ("revision-v6.1",)

    def revision(release_id=catalog.RELEASE_ID):
        calls.append(("revision", release_id))
        assert release_id == FUTURE_RELEASE
        return ("revision-v6.1",)

    monkeypatch.setattr(policies, "ROOT", tmp_path)
    monkeypatch.setattr(policies, "warehouse_policy_artifacts", artifacts)
    monkeypatch.setattr(policies, "warehouse_evidence_snapshot", snapshot)
    monkeypatch.setattr(policies, "warehouse_evidence_revision", revision)
    monkeypatch.setattr(policies, "warehouse_runtime_source_closure_sha256",
                        lambda: "f" * 64)
    monkeypatch.setattr(policies, "_validated_v6_actor", lambda *args: None)
    monkeypatch.setattr(policies, "_WAREHOUSE_ACCEPTANCE_CACHE", set())

    from experiments.warehouse import acceptance as verifier

    def validate(_acceptance, *, release_id, entries, evidence, repository_root):
        calls.append(("validate", release_id))
        assert release_id == FUTURE_RELEASE
        assert entries["raw"] == entry and evidence == {}
        assert repository_root == tmp_path
        return True

    monkeypatch.setattr(verifier, "validate_acceptance", validate)
    binding = policies.binding_for_group("warehouse", "C",
                                         release_id=FUTURE_RELEASE)
    assert binding["policy_id"] == "warehouse-raw-v6.1"
    assert policies._accepted(entry, binding, release_id=FUTURE_RELEASE)
    assert policies._accepted(entry, binding, release_id=FUTURE_RELEASE)
    assert calls.count(("validate", FUTURE_RELEASE)) == 1
    assert not policies._accepted(entry, binding, release_id=catalog.RELEASE_ID)
    assert policies.verify_binding(binding, release_id=FUTURE_RELEASE) == binding

    raw.write_bytes(b"tampered")
    with pytest.raises(policies.PolicyUnavailable, match="actor_hash_mismatch"):
        policies.verify_binding(binding, release_id=FUTURE_RELEASE)
