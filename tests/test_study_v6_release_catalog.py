"""v6 packaging preserves accepted v5 models without reusing their held-out seeds."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from domains.kitchen import engine_v5_frozen as kitchen_physics
from domains.kitchen import neural_domain as kitchen
from domains.pong import turnbased as pong
from study_v3.config import Settings
from study_v3.release_catalog import (RELEASE_ID, UPSTREAM_RELEASE_ID,
                                      ReleaseAssetUnavailable, policy_artifacts,
                                      runtime_config, source_provenance,
                                      warehouse_policy_artifacts,
                                      warehouse_release_spec,
                                      warehouse_runtime_source_closure_sha256)
from study_v3.store import Store, StudyError


@pytest.mark.parametrize("domain", ("pong", "kitchen"))
@pytest.mark.parametrize("variant", ("raw", "regularized"))
def test_accepted_v5_assets_are_byte_identical_in_v6_package(domain, variant):
    entries = policy_artifacts(domain)
    entry = entries[variant]
    assert entry["policy_id"] == f"{domain}-{variant}-v5"
    assert entry["upstream_release_id"] == UPSTREAM_RELEASE_ID
    index = source_provenance()
    for source_key, package_key in (("source_actor_path", "actor_path"),
                                    ("source_training_report_path", "training_report_path"),
                                    ("source_acceptance_path", "acceptance_path")):
        source = entry[source_key]
        assert index[source]["bundle_path"] == entry[package_key]
        assert index[source]["sha256"] == sha256(
            (kitchen.ROOT / entry[package_key]).read_bytes()).hexdigest()
    receipt = json.loads((kitchen.ROOT / entry["acceptance_path"]).read_text())
    assert receipt.get("release_id", receipt.get("release")) == UPSTREAM_RELEASE_ID
    assert receipt["passed"] is True


def test_kitchen_five_dish_physics_config_is_packaged_and_rollout_identical(monkeypatch):
    import study_v3.release_catalog as catalog_module

    source = kitchen.ROOT / "configs/study_v3_kitchen_frozen_v5.json"
    packaged = runtime_config("kitchen")
    assert packaged.read_bytes() == source.read_bytes()

    def rollout():
        state = kitchen.initial_state(1000, 1, "kitchen-regularized-v5",
                                      release_id=RELEASE_ID)
        frames = []
        for _ in range(12):
            frames.append((kitchen.public_state(state), kitchen.decide(state)))
            if state["terminal"]:
                break
            state = kitchen.step(state, kitchen.legal_actions(state)[0])
        return frames

    with monkeypatch.context() as context:
        context.setattr(catalog_module, "runtime_config",
                        lambda domain, release_id=RELEASE_ID: source)
        kitchen_physics._configuration.cache_clear()
        source_frames = rollout()
    kitchen_physics._configuration.cache_clear()
    try:
        assert rollout() == source_frames
    finally:
        kitchen_physics._configuration.cache_clear()


def test_kitchen_physics_rejects_package_hash_mismatch(monkeypatch):
    import study_v3.release_catalog as catalog_module

    bound_state = kitchen.initial_state(1000, 1, "kitchen-regularized-v5",
                                        release_id=RELEASE_ID)
    packaged = runtime_config("kitchen")
    relative = str(packaged.relative_to(kitchen.ROOT))
    monkeypatch.setattr(catalog_module, "_catalog", lambda release_id=RELEASE_ID: {
        "runtime_configs": {"kitchen": {"path": relative, "sha256": "0" * 64}}})
    kitchen_physics._configuration.cache_clear()
    try:
        with pytest.raises(ReleaseAssetUnavailable, match="release_asset_hash_mismatch"):
            kitchen_physics.initial_state(1000, 1)
        with pytest.raises(ReleaseAssetUnavailable, match="release_asset_hash_mismatch"):
            kitchen.initial_state(1000, 1, "kitchen-regularized-v5",
                                  release_id=RELEASE_ID)
        with pytest.raises(ReleaseAssetUnavailable, match="release_asset_hash_mismatch"):
            kitchen.decide(bound_state)
    finally:
        kitchen_physics._configuration.cache_clear()


@pytest.mark.parametrize("adapter,seed", ((pong, 730100), (kitchen, 1000)))
def test_packaged_runtime_matches_legacy_v5_actions_on_development_seed(adapter, seed):
    ident = f"{adapter.DOMAIN}-regularized-v5"
    legacy = adapter.initial_state(seed, 1, ident)
    bundled = adapter.initial_state(seed, 1, ident, release_id=RELEASE_ID)
    for _ in range(20):
        assert {k: v for k, v in bundled.items() if k != "release_id"} == legacy
        if legacy["terminal"]:
            break
        assert adapter.decide(bundled) == adapter.decide(legacy)
        human = (adapter.human_advisor(legacy) if adapter.DOMAIN == "kitchen"
                 else adapter.legal_actions(legacy)[0])
        legacy = adapter.step(legacy, human)
        bundled = adapter.step(bundled, human)


def test_warehouse_v6_binding_metadata_is_required_on_restore(tmp_path, monkeypatch):
    import study_v3.store as store_module

    store = Store(Settings(database=str(tmp_path / "study.sqlite3")))
    instance = {"id": "v6-instance", "domain": "warehouse", "group_code": "A",
                "mode": "preview"}
    monkeypatch.setattr(store_module, "policy_id", lambda domain, group:
                        "warehouse-regularized-v6")
    monkeypatch.setattr(store_module, "verify_binding", lambda binding: binding)
    try:
        with store.db.transaction() as db:
            db.execute("INSERT INTO pl3_policy_bindings VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (instance["id"], "warehouse-regularized-v6", "warehouse",
                        "regularized", "releases/v6/actor.npz", "a" * 64,
                        "b" * 64, "c" * 64, "{}", 1.0))
        with store.db.transaction(read_only=True) as db:
            with pytest.raises(StudyError, match="policy_binding_metadata_missing"):
                store._binding(db, instance)
        with store.db.transaction() as db:
            db.execute("INSERT INTO pl3_policy_binding_metadata VALUES(?,?,?,?)",
                       (instance["id"], "warehouse-dual-actor.v6.1", "d" * 64, 1.0))
        with store.db.transaction(read_only=True) as db:
            restored = store._binding(db, instance)
        assert restored["actor_model_version"] == "warehouse-dual-actor.v6.1"
        assert restored["feature_schema_sha256"] == "d" * 64
    finally:
        store.db.close()


def test_warehouse_catalog_rejects_development_paths_even_with_correct_hash(monkeypatch):
    import study_v3.release_catalog as catalog_module

    source = "configs/study_v3_pong.json"
    digest = sha256((kitchen.ROOT / source).read_bytes()).hexdigest()
    entry = {"policy_id": "warehouse-raw-v6",
             "model_version": "warehouse-dual-actor.v6.1",
             "feature_schema_sha256":
             "1ba9702c1d70b3375eb12f4c0569393ed480995265a4c4ca3aaec37167ff1346",
             "actor_path": source, "actor_sha256": digest,
             "training_report_path": source, "training_report_sha256": digest,
             "acceptance_path": source, "acceptance_sha256": digest,
             "role_sha256": {"human": "a" * 64, "ai": "b" * 64},
             "program_paths": [], "program_sha256": {}}
    monkeypatch.setattr(catalog_module, "_catalog", lambda release_id=RELEASE_ID: {
        "warehouse": {"status": "accepted", "policy_ids": {
            "raw": "warehouse-raw-v6", "regularized": "warehouse-regularized-v6"}},
        "policies": {"warehouse": {"raw": entry,
            "regularized": {**entry, "policy_id": "warehouse-regularized-v6"}}}})
    with pytest.raises(ReleaseAssetUnavailable, match="release_asset_hash_mismatch"):
        warehouse_policy_artifacts()


def test_lightweight_warehouse_source_closure_matches_training_contract():
    from backend.training.warehouse_v6_runner import source_closure_sha256

    assert warehouse_runtime_source_closure_sha256() == source_closure_sha256()


def test_future_warehouse_release_addresses_separate_package_without_activation():
    current = warehouse_release_spec()
    future = warehouse_release_spec("policylens-three-domain-20261016.v6.1")
    assert current.policy_ids == {"raw": "warehouse-raw-v6",
                                  "regularized": "warehouse-regularized-v6"}
    assert future.policy_ids == {"raw": "warehouse-raw-v6.1",
                                 "regularized": "warehouse-regularized-v6.1"}
    assert future.catalog_path != current.catalog_path
    with pytest.raises(ReleaseAssetUnavailable, match="release_catalog_unavailable"):
        warehouse_policy_artifacts(future.release_id)
    with pytest.raises(ReleaseAssetUnavailable, match="warehouse_release_id_invalid"):
        warehouse_release_spec("policylens-three-domain-20261016.v6.00")


def test_packager_rejects_changed_source_before_copy(tmp_path):
    from scripts.package_warehouse_v6_release import package

    receipt_path = (tmp_path / "runs" / "warehouse-v6-synthetic" / "admission" /
                    "warehouse_v6_acceptance.json")
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({
        "candidate_id": "warehouse-v6-synthetic", "release": RELEASE_ID,
        "release_id": RELEASE_ID,
        "policy_id": warehouse_release_spec().policy_ids,
        "passed": True, "heldout_evaluated": True,
        "source_closure_sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="source closure differs"):
        package(receipt_path)


def test_successor_release_starts_pending_with_copied_v5_bytes(tmp_path):
    from scripts.start_warehouse_v6_candidate_release import stage

    current = tmp_path / "releases" / RELEASE_ID
    asset_bytes = b"accepted-v5-actor"
    asset_sha = sha256(asset_bytes).hexdigest()
    relative = f"releases/{RELEASE_ID}/assets/{asset_sha}/actor.bin"
    asset = tmp_path / relative
    asset.parent.mkdir(parents=True)
    asset.write_bytes(asset_bytes)
    current_catalog = {
        "format": "policylens-release-catalog.v1", "release_id": RELEASE_ID,
        "upstream_release_id": UPSTREAM_RELEASE_ID,
        "warehouse": {"status": "pending", "policy_ids": warehouse_release_spec().policy_ids},
        "policies": {"pong": {"raw": {"actor_path": relative,
                                            "program_sha256": {relative: asset_sha}}},
                     "kitchen": {}},
        "source_asset_index": {"upstream/actor.bin": {
            "bundle_path": relative, "sha256": asset_sha}},
    }
    (current / "catalog.json").write_text(json.dumps(current_catalog))
    closure = tmp_path / "candidate.closed.json"
    closure.write_text(json.dumps({"release_id": RELEASE_ID,
                                   "candidate_id": "failed-v6-0", "phase": "heldout",
                                   "passed": False}))
    future_id = "policylens-three-domain-20261016.v6.1"
    result = stage(future_id, closure, repository_root=tmp_path)
    future = json.loads((tmp_path / "releases" / future_id / "catalog.json").read_text())
    future_path = future["policies"]["pong"]["raw"]["actor_path"]
    assert result["warehouse_status"] == future["warehouse"]["status"] == "pending"
    assert future["warehouse"]["policy_ids"]["raw"] == "warehouse-raw-v6.1"
    assert future_path.startswith(f"releases/{future_id}/assets/")
    assert (tmp_path / future_path).read_bytes() == asset_bytes
    assert future["policies"]["pong"]["raw"]["program_sha256"] == {
        future_path: asset_sha}
    with pytest.raises(FileExistsError):
        stage(future_id, closure, repository_root=tmp_path)
