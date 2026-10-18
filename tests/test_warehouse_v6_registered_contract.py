"""Registered v6 refuses silent BC and program-method substitutions."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import (
    FEATURE_SCHEMA_SHA256, MODEL_VERSION, WarehouseV6Actor, role_hashes,
)
from backend.training.warehouse_v6_runner import _checked_spec
from experiments.warehouse import protocol as p
from scripts import package_warehouse_v6_release as packaging
from tests.warehouse_v6_contract_fixture import bc_receipt, topology_freeze_evidence, topology_diagnostic


def _registered_mapping(spec: p.PairRunSpec) -> dict:
    pair_dir = (Path(spec.output_dir) / spec.candidate_id /
                f"pair_{spec.pair_seed}_steps_{spec.joint_steps}")
    return {**spec.to_mapping(),
            "warmup_actor_path": str(pair_dir / "warmup.actor.npz"),
            "final_bc_actor_path": str(pair_dir / "final-bc.actor.npz"),
            "final_bc_precheck_receipt_path": str(pair_dir / "precheck.json")}


def test_registered_bc_schedule_and_method_are_inseparable_from_run_spec(tmp_path):
    spec = p.PairRunSpec("registered-contract", p.DEVELOPMENT_PAIR_SEEDS[0],
                         200_000, str(tmp_path),
                         p.PROGRAM_FIT_METHOD_CANDIDATES[0],
                         checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    schedule = p.bc_data_seed_schedule(spec.pair_seed)
    assert len(schedule) == 23
    assert sum(map(len, schedule.values())) == 1_394
    assert len(set(seed for block in schedule.values() for seed in block)) == 1_394
    assert spec.bc_data_seed_schedule_sha256 == p._hash_json(schedule)
    critic_resets = p.bc_critic_reset_schedule(spec.pair_seed, spec.release_id)
    assert critic_resets == {"seed": spec.pair_seed + 6_200,
                             "environment_reset_count": 2}
    assert critic_resets["seed"] not in {
        seed for block in schedule.values() for seed in block}
    assert spec.bc_critic_reset_schedule_sha256 == p._hash_json(critic_resets)
    run_spec = _registered_mapping(spec)
    assert _checked_spec(run_spec)["program_fit_method"] == \
        p.PROGRAM_FIT_METHOD_CANDIDATES[0]
    for key, bad in (
            ("program_fit_method", "axis_tree_v1"),
            ("bc_recipe_sha256", "0" * 64),
            ("risk_dual_config_sha256", "0" * 64),
            ("bc_data_seed_schedule_sha256", "0" * 64),
            ("bc_critic_reset_schedule_sha256", "0" * 64)):
        with pytest.raises(ValueError):
            _checked_spec({**run_spec, key: bad})
        missing = dict(run_spec)
        missing.pop(key)
        with pytest.raises(ValueError):
            _checked_spec(missing)
    with pytest.raises(ValueError, match="critic reset schedule"):
        _checked_spec({**run_spec,
                       "bc_critic_reset_schedule": {"seed": spec.pair_seed,
                                                     "environment_reset_count": 2}})


def test_bc_receipt_cannot_hide_removed_stage_or_critic_drift():
    seed, closure = p.DEVELOPMENT_PAIR_SEEDS[0], "a" * 64
    report = bc_receipt(seed, closure)
    assert p.bc_report_valid(report, seed, closure)
    removed = deepcopy(report)
    removed["bc_stages"].pop(10)
    assert not p.bc_report_valid(removed, seed, closure)
    drift = deepcopy(report)
    drift["collision_critic"]["epochs"] += 1
    assert not p.bc_report_valid(drift, seed, closure)
    drift = deepcopy(report)
    drift["collision_critic"]["learning_rate"] *= 2
    assert not p.bc_report_valid(drift, seed, closure)
    drift = deepcopy(report)
    next(row for row in drift["bc_stages"] if row["stage"] ==
         "final_standard_margin_calibration")["fit"]["margin_weight"] = 0.03
    assert not p.bc_report_valid(drift, seed, closure)
    drift = deepcopy(report)
    next(row for row in drift["bc_stages"] if row["stage"] ==
         "post_refit_public_bc")["warmup_masked_kl_weight"] = 1.0
    assert not p.bc_report_valid(drift, seed, closure)
    moved = deepcopy(report)
    moved["bc_stages"][0]["episode_seeds"][0] += 1
    assert not p.bc_report_valid(moved, seed, closure)
    drift = deepcopy(report)
    drift["collision_critic"]["environment_reset_count"] = 1
    assert not p.bc_report_valid(drift, seed, closure)


def test_packaging_rejects_axis_program_even_if_file_hashes_match(
        tmp_path, monkeypatch):
    candidate = "registered-package"
    seed = p.FORMAL_PAIR_SEEDS[0]
    source_closure = "a" * 64
    bc = bc_receipt(seed, source_closure)
    final_actor_path = tmp_path / "final-bc.actor.npz"
    final_actor_path.write_bytes(b"synthetic-final-bc-actor")
    final_actor = bc["final_bc_internal_precheck"]["final_bc_actor"]
    final_actor["path"] = str(final_actor_path)
    final_actor["sha256"] = sha256(final_actor_path.read_bytes()).hexdigest()
    independent_path = tmp_path / "final-bc-precheck.json"
    independent_path.write_text(json.dumps(
        bc["final_bc_internal_precheck"], sort_keys=True, allow_nan=False) + "\n")
    bc["final_bc_internal_precheck_receipt"] = {
        "path": str(independent_path),
        "sha256": sha256(independent_path.read_bytes()).hexdigest(),
    }
    pair = {"candidate_id": candidate, "pair_seed": seed,
            "joint_steps": 200_000,
            "topology_refresh_steps": list(p.topology_refresh_steps(200_000)),
            "topology_frozen_at_step": p.topology_frozen_at_step(200_000),
            "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                "action_physics_config_sha256"],
            "program_fit_method": p.PROGRAM_FIT_METHOD_CANDIDATES[0],
            "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
            "risk_dual_config_sha256": p.REGISTERED_RISK_DUAL_CONFIG_SHA256,
            "source_closure_sha256": source_closure,
            "bc": bc,
            "bc_data_seed_schedule_sha256": p._hash_json(
                p.bc_data_seed_schedule(seed)),
            "variants": {}}
    assets = {}
    for variant in p.VARIANTS:
        role_hashes = {role: str(index + 1) * 64
                       for index, role in enumerate(p.ROLES)}
        program_items = ({role: {"sha256": str(index + 4) * 64}
                          for index, role in enumerate(p.ROLES)}
                         if variant == "regularized" else {})
        actor_sha, report_sha = "1" * 64, "2" * 64
        pair["variants"][variant] = {
            "program_fit_method": p.PROGRAM_FIT_METHOD_CANDIDATES[0],
            "topology_refresh_steps": pair["topology_refresh_steps"],
            "topology_frozen_at_step": pair["topology_frozen_at_step"],
            "topology_refresh_history": [topology_diagnostic(joint_step=step)
                                         for step in pair["topology_refresh_steps"]],
            **topology_freeze_evidence(200_000),
            "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                "action_physics_config_sha256"],
            "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
            "risk_dual_config_sha256": p.REGISTERED_RISK_DUAL_CONFIG_SHA256,
            "actor_artifact": {"sha256": actor_sha},
            "selected_role_sha256": role_hashes,
            "training_report_sha256": report_sha,
            "program_artifacts": program_items,
        }
        assets[variant] = {
            "actor": {"sha256": actor_sha, "path": "unused"},
            "training_report": {"sha256": report_sha, "path": "unused"},
            "programs": {role: {"sha256": item["sha256"], "path": "unused"}
                         for role, item in program_items.items()},
        }
    relative = "pair.training.json"
    (tmp_path / relative).write_text(json.dumps(pair))
    monkeypatch.setattr(packaging, "ROOT", tmp_path)
    from backend.training import warehouse_v6_actor, warehouse_v6_program
    monkeypatch.setattr(warehouse_v6_actor.FrozenWarehouseV6Actor, "load",
                        lambda *_args, **kwargs: type("Actor", (), {
                            "role_sha256": kwargs["expected_role_sha256"]})())
    monkeypatch.setattr(warehouse_v6_program.WarehouseV6Program, "load",
                        lambda *_args, **kwargs: type("Program", (), {
                            "role": p.ROLES[0],
                            "report": {"fit_method": "axis_tree_v1"}})())
    receipt = {"candidate_id": candidate, "locked_budget": 200_000,
               "program_fit_method": p.PROGRAM_FIT_METHOD_CANDIDATES[0]}
    evidence = {"pair_reports": {str(seed): {"path": relative}},
                "training_assets": {str(seed): assets},
                "development_pair_reports": {}}
    with pytest.raises(ValueError, match="axis program"):
        packaging._verify_training_package(receipt, evidence)



def test_direct_training_entrypoints_reject_unscoped_resume(tmp_path):
    """v6.14 admits only a new, SHA-locked pair; old resume calls stay closed."""
    registered = p.PairRunSpec(
        "resume-closed", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    run_spec = _registered_mapping(registered)
    spec = _checked_spec(run_spec)
    model = WarehouseV6Actor()
    pair_dir = (tmp_path / registered.candidate_id /
                f"pair_{registered.pair_seed}_steps_{registered.joint_steps}")
    pair_dir.mkdir(parents=True)
    (pair_dir / "paired_bc.checkpoint.pt").write_bytes(b"old checkpoint")
    with pytest.raises(RuntimeError, match="review-only"):
        runner.train_pair(run_spec)
    with pytest.raises(RuntimeError, match="review-only"):
        runner._train_variant(model.state_dict(), variant="raw",
                              spec=spec, directory=pair_dir)


def test_locked_formal_pair_refuses_preexisting_checkpoint_directory(tmp_path):
    registered = p.PairRunSpec(
        "fresh-only", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    run_spec = _registered_mapping(registered)
    pair_dir = (tmp_path / registered.candidate_id /
                f"pair_{registered.pair_seed}_steps_{registered.joint_steps}")
    pair_dir.mkdir(parents=True)
    (pair_dir / "paired_bc.checkpoint.pt").write_bytes(b"old checkpoint")
    with pytest.raises(RuntimeError, match="output is not isolated and new"):
        runner.run_locked_formal_pair(
            run_spec, authorization_receipt_path=tmp_path / "missing-auth.json")
