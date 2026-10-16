"""A validated release stays fast while every asset change fails closed."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from copy import deepcopy

import pytest

from backend.training.warehouse_v6_actor import MODEL_VERSION, FEATURE_SCHEMA_SHA256
from experiments.warehouse import protocol as p
from study_v3 import release_catalog as catalog


def _synthetic_package(root: Path) -> dict:
    release = catalog.RELEASE_ID
    folder = root / "releases" / release

    def asset(name: str, payload: bytes) -> dict:
        digest = sha256(payload).hexdigest()
        relative = f"releases/{release}/assets/{digest}/{name}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return {"path": relative, "sha256": digest}

    def training(seed: int) -> dict:
        return {variant: {
            "actor": asset(f"{seed}.{variant}.actor.npz",
                           f"actor:{seed}:{variant}".encode()),
            "training_report": asset(f"{seed}.{variant}.training.json",
                                     f"report:{seed}:{variant}".encode()),
            "programs": ({role: asset(f"{seed}.{role}.program.json",
                                      f"program:{seed}:{role}".encode())
                         for role in p.ROLES} if variant == "regularized" else {}),
        } for variant in p.VARIANTS}

    training_assets = {str(seed): training(seed) for seed in p.FORMAL_PAIR_SEEDS}
    development_assets = {str(seed): training(seed)
                          for seed in p.DEVELOPMENT_PAIR_SEEDS}
    evidence = {
        "phase_receipts": {name: asset(name, name.encode()) for name in (
            "budget.lock.json", "formal.selection.json",
            *(f"formal.training.{seed}.json" for seed in p.FORMAL_PAIR_SEEDS),
            "program.preaudit.json", "final_program_audit.json", "heldout.json")},
        "one_use_claims": {phase: asset(f"{phase}.claim.json", phase.encode())
                           for phase in ("final_program_audit", "heldout")},
        "pair_reports": {str(seed): asset("pair.training.json",
                                          f"formal:{seed}".encode())
                         for seed in p.FORMAL_PAIR_SEEDS},
        "development_receipts": {"200000": {
            str(seed): asset(f"development.200000.{seed}.json",
                             f"development:{seed}".encode())
            for seed in p.DEVELOPMENT_PAIR_SEEDS}},
        "development_pair_reports": {"200000": {
            str(seed): asset("pair.training.json", f"dev-pair:{seed}".encode())
            for seed in p.DEVELOPMENT_PAIR_SEEDS}},
        "training_assets": training_assets,
        "development_training_assets": {"200000": development_assets},
        "program_traces": {
            "development": {"200000": {
                str(seed): {} for seed in p.DEVELOPMENT_PAIR_SEEDS}},
            "preaudit": {role: asset(f"preaudit.{role}.trace.json",
                                      f"preaudit:{role}".encode())
                         for role in p.ROLES},
            "final_audit": {role: asset(f"final.{role}.trace.json",
                                         f"final:{role}".encode())
                            for role in p.ROLES},
        },
        "final_fit_traces": {
            "formal": {str(seed): {split: asset(
                f"{split}.jsonl", f"fit:{seed}:{split}".encode())
                for split in p.FINAL_PROGRAM_SPLIT_COUNTS}
                for seed in p.FORMAL_PAIR_SEEDS},
            "development": {"200000": {str(seed): {split: asset(
                f"{split}.jsonl", f"dev-fit:{seed}:{split}".encode())
                for split in p.FINAL_PROGRAM_SPLIT_COUNTS}
                for seed in p.DEVELOPMENT_PAIR_SEEDS}},
        },
        "final_fit_replays": {
            "formal": {str(seed): asset(
                "final_fit_replay.receipt.json", f"fit-replay:{seed}".encode())
                for seed in p.FORMAL_PAIR_SEEDS},
            "development": {"200000": {str(seed): asset(
                "final_fit_replay.receipt.json", f"dev-fit-replay:{seed}".encode())
                for seed in p.DEVELOPMENT_PAIR_SEEDS}},
        },
        "quality_replays": {
            "formal": {str(seed): {variant: asset(
                f"formal.{seed}.{variant}.quality-replay.json",
                f"quality:formal:{seed}:{variant}".encode())
                for variant in p.VARIANTS}
                for seed in p.FORMAL_PAIR_SEEDS},
            "development": {"200000": {str(seed): {} for seed in
                                      p.DEVELOPMENT_PAIR_SEEDS}},
            "heldout": {variant: {str(task): asset(
                f"heldout.{variant}.task{task}.quality-replay.json",
                f"quality:heldout:{variant}:{task}".encode())
                for task in p.TASKS} for variant in p.VARIANTS},
        },
    }
    acceptance = asset("warehouse_v6_acceptance.json", json.dumps({
        "source_closure_sha256": "7" * 64}).encode())
    chosen = training_assets[str(p.FORMAL_PAIR_SEEDS[0])]
    entries = {}
    for variant in p.VARIANTS:
        row = chosen[variant]
        programs = row["programs"]
        entries[variant] = {
            "policy_id": f"warehouse-{variant}-v6",
            "model_version": MODEL_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "actor_path": row["actor"]["path"],
            "actor_sha256": row["actor"]["sha256"],
            "training_report_path": row["training_report"]["path"],
            "training_report_sha256": row["training_report"]["sha256"],
            "acceptance_path": acceptance["path"],
            "acceptance_sha256": acceptance["sha256"],
            "role_sha256": {"human": "a" * 64, "ai": "b" * 64},
            "program_paths": [programs[role]["path"] for role in p.ROLES
                              if role in programs],
            "program_sha256": {role: programs[role]["sha256"]
                               for role in programs},
        }
    (folder / "catalog.json").write_text(json.dumps({
        "format": "policylens-release-catalog.v1", "release_id": release,
        "upstream_release_id": catalog.UPSTREAM_RELEASE_ID,
        "warehouse": {"status": "accepted", "locked_budget": 200000,
                      "policy_ids": catalog.warehouse_release_spec().policy_ids,
                      "evidence": evidence},
        "policies": {"warehouse": entries},
    }))
    return {"entries": entries, "evidence": evidence}


def test_cached_evidence_skips_rehash_but_rejects_changed_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(catalog, "CATALOG_PATH",
                        tmp_path / "releases" / catalog.RELEASE_ID / "catalog.json")
    monkeypatch.setattr(catalog, "_WAREHOUSE_EVIDENCE_CACHE", {})
    package = _synthetic_package(tmp_path)
    assert catalog.warehouse_policy_artifacts() == package["entries"]

    original_asset = catalog._asset
    verified = []

    def counted(path, expected, release_id=catalog.RELEASE_ID):
        verified.append(path)
        return original_asset(path, expected, release_id)

    monkeypatch.setattr(catalog, "_asset", counted)
    assert catalog.warehouse_evidence() == package["evidence"]
    assert verified == []  # file identity snapshot reused; no full package SHA pass

    selected = package["entries"]["regularized"]
    for relative in (selected["actor_path"], selected["program_paths"][0],
                     selected["acceptance_path"]):
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(b"tampered")
        with pytest.raises(catalog.ReleaseAssetUnavailable):
            catalog.warehouse_policy_artifacts()
        path.write_bytes(original)
        assert catalog.warehouse_policy_artifacts() == package["entries"]

    nonselected = package["evidence"]["training_assets"][
        str(p.FORMAL_PAIR_SEEDS[-1])]["raw"]["actor"]
    (tmp_path / nonselected["path"]).unlink()
    with pytest.raises(catalog.ReleaseAssetUnavailable):
        catalog.warehouse_policy_artifacts()


def test_acceptance_proof_is_reused_only_for_unchanged_evidence(tmp_path, monkeypatch):
    from experiments.warehouse import acceptance as verifier
    from study_v3 import policies

    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(catalog, "CATALOG_PATH",
                        tmp_path / "releases" / catalog.RELEASE_ID / "catalog.json")
    monkeypatch.setattr(catalog, "_WAREHOUSE_EVIDENCE_CACHE", {})
    monkeypatch.setattr(policies, "ROOT", tmp_path)
    monkeypatch.setattr(policies, "_WAREHOUSE_ACCEPTANCE_CACHE", set())
    monkeypatch.setattr(policies, "warehouse_runtime_source_closure_sha256",
                        lambda: "7" * 64)
    package = _synthetic_package(tmp_path)
    entry = package["entries"]["regularized"]
    binding = {"domain": "warehouse", "variant": "regularized",
               "policy_id": entry["policy_id"],
               "actor_path": entry["actor_path"],
               "actor_sha256": entry["actor_sha256"],
               "human_actor_sha256": entry["role_sha256"]["human"],
               "ai_actor_sha256": entry["role_sha256"]["ai"],
               "program_hashes": {path: entry["program_sha256"][role]
                                  for role, path in zip(p.ROLES, entry["program_paths"])},
               "actor_model_version": entry["model_version"],
               "feature_schema_sha256": entry["feature_schema_sha256"]}
    calls = []
    monkeypatch.setattr(verifier, "validate_acceptance",
                        lambda *args, **kwargs: calls.append(True) or True)
    assert policies._accepted_warehouse_v6(entry, binding)
    assert policies._accepted_warehouse_v6(entry, binding)
    assert len(calls) == 1

    wrong_path = {**binding, "actor_path": "releases/other/actor.npz"}
    assert not policies._accepted_warehouse_v6(entry, wrong_path)
    wrong_program = {**binding, "program_hashes": {
        **binding["program_hashes"], entry["program_paths"][0]: "0" * 64}}
    assert not policies._accepted_warehouse_v6(entry, wrong_program)
    assert len(calls) == 1

    missing = package["evidence"]["pair_reports"][str(p.FORMAL_PAIR_SEEDS[-1])]
    (tmp_path / missing["path"]).unlink()
    assert not policies._accepted_warehouse_v6(entry, binding)
    assert len(calls) == 1


def test_catalog_replacement_during_acceptance_cannot_cache_old_proof(
        tmp_path, monkeypatch):
    from experiments.warehouse import acceptance as verifier
    from study_v3 import policies

    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    catalog_path = (tmp_path / "releases" / catalog.RELEASE_ID / "catalog.json")
    monkeypatch.setattr(catalog, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(catalog, "_WAREHOUSE_EVIDENCE_CACHE", {})
    monkeypatch.setattr(policies, "ROOT", tmp_path)
    monkeypatch.setattr(policies, "_WAREHOUSE_ACCEPTANCE_CACHE", set())
    monkeypatch.setattr(policies, "warehouse_runtime_source_closure_sha256",
                        lambda: "7" * 64)
    package = _synthetic_package(tmp_path)
    entry = package["entries"]["regularized"]
    binding = {"domain": "warehouse", "variant": "regularized",
               "policy_id": entry["policy_id"],
               "actor_path": entry["actor_path"],
               "actor_sha256": entry["actor_sha256"],
               "human_actor_sha256": entry["role_sha256"]["human"],
               "ai_actor_sha256": entry["role_sha256"]["ai"],
               "program_hashes": {path: entry["program_sha256"][role]
                                  for role, path in zip(p.ROLES, entry["program_paths"])},
               "actor_model_version": entry["model_version"],
               "feature_schema_sha256": entry["feature_schema_sha256"]}

    changed = deepcopy(json.loads(catalog_path.read_text()))
    payload = b"semantically-invalid-new-budget-receipt"
    digest = sha256(payload).hexdigest()
    relative = (f"releases/{catalog.RELEASE_ID}/assets/{digest}/"
                "budget.lock.json")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    changed["warehouse"]["evidence"]["phase_receipts"]["budget.lock.json"] = {
        "path": relative, "sha256": digest}
    old_hash = package["evidence"]["phase_receipts"]["budget.lock.json"]["sha256"]
    seen = []

    def validating(_acceptance, *, evidence, **_kwargs):
        observed = evidence["phase_receipts"]["budget.lock.json"]["sha256"]
        seen.append(observed)
        if observed == old_hash:
            catalog_path.write_text(json.dumps(changed))
            return True
        return False

    monkeypatch.setattr(verifier, "validate_acceptance", validating)
    assert not policies._accepted_warehouse_v6(entry, binding)
    assert not policies._accepted_warehouse_v6(entry, binding)
    assert seen == [old_hash, digest]
    assert policies._WAREHOUSE_ACCEPTANCE_CACHE == set()


def test_catalog_replacement_during_evidence_snapshot_fails_closed(
        tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    catalog_path = (tmp_path / "releases" / catalog.RELEASE_ID / "catalog.json")
    monkeypatch.setattr(catalog, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(catalog, "_WAREHOUSE_EVIDENCE_CACHE", {})
    _synthetic_package(tmp_path)
    original_revision = catalog._warehouse_evidence_revision

    def replace_before_revision(release_id, evidence):
        changed = json.loads(catalog_path.read_text())
        changed["warehouse"]["candidate_id"] = "different-candidate"
        catalog_path.write_text(json.dumps(changed))
        return original_revision(release_id, evidence)

    monkeypatch.setattr(catalog, "_warehouse_evidence_revision",
                        replace_before_revision)
    with pytest.raises(catalog.ReleaseAssetUnavailable,
                       match="warehouse_catalog_changed_during_validation"):
        catalog.warehouse_evidence_snapshot()
    assert catalog._WAREHOUSE_EVIDENCE_CACHE == {}


def test_catalog_replacement_between_policy_entries_and_evidence_fails_closed(
        tmp_path, monkeypatch):
    from experiments.warehouse import acceptance as verifier
    from study_v3 import policies

    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    catalog_path = (tmp_path / "releases" / catalog.RELEASE_ID / "catalog.json")
    monkeypatch.setattr(catalog, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(catalog, "_WAREHOUSE_EVIDENCE_CACHE", {})
    monkeypatch.setattr(policies, "ROOT", tmp_path)
    monkeypatch.setattr(policies, "_WAREHOUSE_ACCEPTANCE_CACHE", set())
    monkeypatch.setattr(policies, "warehouse_runtime_source_closure_sha256",
                        lambda: "7" * 64)
    package = _synthetic_package(tmp_path)
    entry = package["entries"]["regularized"]
    binding = {"domain": "warehouse", "variant": "regularized",
               "policy_id": entry["policy_id"],
               "actor_path": entry["actor_path"],
               "actor_sha256": entry["actor_sha256"],
               "human_actor_sha256": entry["role_sha256"]["human"],
               "ai_actor_sha256": entry["role_sha256"]["ai"],
               "program_hashes": {path: entry["program_sha256"][role]
                                  for role, path in zip(p.ROLES, entry["program_paths"])},
               "actor_model_version": entry["model_version"],
               "feature_schema_sha256": entry["feature_schema_sha256"]}
    changed = json.loads(catalog_path.read_text())
    changed["policies"]["warehouse"]["regularized"]["role_sha256"]["ai"] = "c" * 64
    original_snapshot = policies.warehouse_evidence_snapshot

    def replace_then_snapshot():
        catalog_path.write_text(json.dumps(changed))
        return original_snapshot()

    monkeypatch.setattr(policies, "warehouse_evidence_snapshot", replace_then_snapshot)
    monkeypatch.setattr(verifier, "validate_acceptance",
                        lambda *_args, **_kwargs: True)
    assert not policies._accepted_warehouse_v6(entry, binding)
    assert policies._WAREHOUSE_ACCEPTANCE_CACHE == set()
