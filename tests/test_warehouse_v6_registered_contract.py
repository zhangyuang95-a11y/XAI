"""Registered v6 refuses silent BC and program-method substitutions."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import (
    FEATURE_SCHEMA_SHA256, MODEL_VERSION, WarehouseV6Actor, role_hashes,
)
from backend.training.warehouse_v6_runner import _checked_spec
from experiments.warehouse import protocol as p
from scripts import package_warehouse_v6_release as packaging
from tests.warehouse_v6_contract_fixture import bc_receipt


def test_registered_bc_schedule_and_method_are_inseparable_from_run_spec(tmp_path):
    spec = p.PairRunSpec("registered-contract", p.DEVELOPMENT_PAIR_SEEDS[0],
                         200_000, str(tmp_path),
                         p.PROGRAM_FIT_METHOD_CANDIDATES[0],
                         checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    schedule = p.bc_data_seed_schedule(spec.pair_seed)
    assert len(schedule) == 16
    assert sum(map(len, schedule.values())) == 1_220
    assert len(set(seed for block in schedule.values() for seed in block)) == 1_220
    assert spec.bc_data_seed_schedule_sha256 == p._hash_json(schedule)
    critic_resets = p.bc_critic_reset_schedule(spec.pair_seed, spec.release_id)
    assert critic_resets == {"seed": spec.pair_seed + 6_200,
                             "environment_reset_count": 2}
    assert critic_resets["seed"] not in {
        seed for block in schedule.values() for seed in block}
    assert spec.bc_critic_reset_schedule_sha256 == p._hash_json(critic_resets)
    assert _checked_spec(spec.to_mapping())["program_fit_method"] == \
        p.PROGRAM_FIT_METHOD_CANDIDATES[0]
    for key, bad in (
            ("program_fit_method", "axis_tree_v1"),
            ("bc_recipe_sha256", "0" * 64),
            ("risk_dual_config_sha256", "0" * 64),
            ("bc_data_seed_schedule_sha256", "0" * 64),
            ("bc_critic_reset_schedule_sha256", "0" * 64)):
        with pytest.raises(ValueError):
            _checked_spec({**spec.to_mapping(), key: bad})
        missing = spec.to_mapping()
        missing.pop(key)
        with pytest.raises(ValueError):
            _checked_spec(missing)
    with pytest.raises(ValueError, match="critic reset schedule"):
        _checked_spec({**spec.to_mapping(),
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
    drift["bc_stages"][-1]["fit"]["margin_weight"] = 0.0
    assert not p.bc_report_valid(drift, seed, closure)
    drift = deepcopy(report)
    drift["bc_stages"][1]["gate_loss_weight"] = 0.0
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
    pair = {"candidate_id": candidate, "pair_seed": seed,
            "joint_steps": 200_000,
            "program_fit_method": p.PROGRAM_FIT_METHOD_CANDIDATES[0],
            "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
            "risk_dual_config_sha256": p.REGISTERED_RISK_DUAL_CONFIG_SHA256,
            "source_closure_sha256": source_closure,
            "bc": bc_receipt(seed, source_closure),
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


def test_registered_variant_resume_rejects_method_change_before_training(tmp_path):
    spec = _checked_spec(p.PairRunSpec(
        "resume-method", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS).to_mapping())
    model = WarehouseV6Actor()
    payload = {
        "training_config_sha256": runner._training_config_sha256(spec),
        "source_closure_sha256": spec["source_closure_sha256"],
        "variant": "raw", "initial_role_sha256": role_hashes(model),
        "model_version": MODEL_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "program_fit_method": "axis_tree_v1",
        "bc_recipe_sha256": spec["bc_recipe_sha256"],
        "bc_data_seed_schedule_sha256": spec["bc_data_seed_schedule_sha256"],
        "risk_dual_config_sha256": spec["risk_dual_config_sha256"],
    }
    torch.save(payload, tmp_path / "raw.checkpoint.pt")
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        runner._train_variant(model.state_dict(), variant="raw",
                              spec=spec, directory=tmp_path)


def test_registered_bc_resume_rejects_method_change_before_training(tmp_path):
    run_spec = p.PairRunSpec(
        "resume-bc", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS).to_mapping()
    spec = _checked_spec(run_spec)
    directory = tmp_path / spec["candidate_id"] / (
        f"pair_{spec['pair_seed']}_steps_{spec['joint_steps']}")
    directory.mkdir(parents=True)
    torch.save({
        "training_config_sha256": runner._training_config_sha256(spec),
        "source_closure_sha256": spec["source_closure_sha256"],
        "program_fit_method": "axis_tree_v1",
        "bc_recipe_sha256": spec["bc_recipe_sha256"],
        "bc_data_seed_schedule_sha256": spec["bc_data_seed_schedule_sha256"],
    }, directory / "paired_bc.checkpoint.pt")
    with pytest.raises(RuntimeError, match="BC checkpoint run spec mismatch"):
        runner.train_pair(run_spec)


@pytest.mark.parametrize("tamper", ("state", "report", "joint_steps"))
def test_completed_resume_rechecks_selected_state_report_and_steps(tmp_path, tamper):
    spec = _checked_spec(p.PairRunSpec(
        "resume-complete", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS).to_mapping())
    model = WarehouseV6Actor()
    state = deepcopy(model.state_dict())
    report = {
        "run_spec_sha256": runner._scientific_run_spec_sha256(spec),
        "training_config_sha256": runner._training_config_sha256(spec),
        "source_closure_sha256": spec["source_closure_sha256"],
        "pair_seed": spec["pair_seed"], "variant": "raw",
        "joint_steps": spec["joint_steps"],
        "actor_decisions": 2 * spec["joint_steps"],
        "program_fit_method": spec["program_fit_method"],
        "bc_recipe_sha256": spec["bc_recipe_sha256"],
        "bc_data_seed_schedule_sha256": spec["bc_data_seed_schedule_sha256"],
        "risk_dual_config_sha256": spec["risk_dual_config_sha256"],
        "actual_episode_resets": [],
        "actual_episode_resets_sha256": runner._digest_json([]),
        "selected_role_sha256": role_hashes(model),
    }
    payload = {
        "training_config_sha256": runner._training_config_sha256(spec),
        "source_closure_sha256": spec["source_closure_sha256"],
        "variant": "raw", "initial_role_sha256": role_hashes(model),
        "model_version": MODEL_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "run_spec_sha256": runner._scientific_run_spec_sha256(spec),
        "pair_seed": spec["pair_seed"], "joint_steps": spec["joint_steps"],
        "program_fit_method": spec["program_fit_method"],
        "bc_recipe_sha256": spec["bc_recipe_sha256"],
        "bc_data_seed_schedule_sha256": spec["bc_data_seed_schedule_sha256"],
        "risk_dual_config_sha256": spec["risk_dual_config_sha256"],
        "complete": True, "selected_state_dict": state,
        "selected_programs": {}, "report": report,
        "selected_state_sha256": runner._state_dict_sha256(state),
        "selected_programs_sha256": runner._digest_json({}),
        "report_sha256": runner._digest_json(report),
    }
    if tamper == "state":
        first = next(iter(state))
        state[first] = state[first].clone() + 1
    elif tamper == "report":
        report["bc_recipe_sha256"] = "0" * 64
    else:
        report["joint_steps"] = 120
        payload["report_sha256"] = runner._digest_json(report)
    torch.save(payload, tmp_path / "raw.checkpoint.pt")
    with pytest.raises(RuntimeError, match="completed checkpoint"):
        runner._train_variant(model.state_dict(), variant="raw",
                              spec=spec, directory=tmp_path)


def test_bc_resume_rechecks_state_and_receipt_digest(tmp_path):
    run_spec = p.PairRunSpec(
        "resume-bc-hash", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS).to_mapping()
    spec = _checked_spec(run_spec)
    directory = tmp_path / spec["candidate_id"] / (
        f"pair_{spec['pair_seed']}_steps_{spec['joint_steps']}")
    directory.mkdir(parents=True)
    torch.manual_seed(spec["pair_seed"])
    base = WarehouseV6Actor()
    state = deepcopy(base.state_dict())
    report = bc_receipt(spec["pair_seed"], spec["source_closure_sha256"])
    report["initial_role_sha256"] = role_hashes(base)
    report["pretrained_role_sha256"] = role_hashes(base)
    payload = {
        "training_config_sha256": runner._training_config_sha256(spec),
        "run_spec_sha256": runner._scientific_run_spec_sha256(spec),
        "pair_seed": spec["pair_seed"], "joint_steps": spec["joint_steps"],
        "source_closure_sha256": spec["source_closure_sha256"],
        "model_version": MODEL_VERSION,
        "program_fit_method": spec["program_fit_method"],
        "bc_recipe_sha256": spec["bc_recipe_sha256"],
        "bc_data_seed_schedule_sha256": spec["bc_data_seed_schedule_sha256"],
        "risk_dual_config_sha256": spec["risk_dual_config_sha256"],
        "state_dict": state, "report": report,
        "state_sha256": runner._state_dict_sha256(state),
        "report_sha256": runner._digest_json(report),
    }
    first = next(iter(state))
    state[first] = state[first].clone() + 1
    torch.save(payload, directory / "paired_bc.checkpoint.pt")
    with pytest.raises(RuntimeError, match="BC checkpoint evidence changed"):
        runner.train_pair(run_spec)


def test_partial_resume_rechecks_progress_before_loading_optimizer(tmp_path):
    spec = _checked_spec(p.PairRunSpec(
        "resume-partial", p.DEVELOPMENT_PAIR_SEEDS[0], 200_000,
        str(tmp_path), p.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS).to_mapping())
    model = WarehouseV6Actor()
    state = deepcopy(model.state_dict())
    payload = {
        "training_config_sha256": runner._training_config_sha256(spec),
        "source_closure_sha256": spec["source_closure_sha256"],
        "variant": "raw", "initial_role_sha256": role_hashes(model),
        "model_version": MODEL_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "run_spec_sha256": runner._scientific_run_spec_sha256(spec),
        "pair_seed": spec["pair_seed"], "joint_steps": spec["joint_steps"],
        "program_fit_method": spec["program_fit_method"],
        "bc_recipe_sha256": spec["bc_recipe_sha256"],
        "bc_data_seed_schedule_sha256": spec["bc_data_seed_schedule_sha256"],
        "risk_dual_config_sha256": spec["risk_dual_config_sha256"],
        "complete": False, "state_dict": state,
        "used_steps": 120, "episode_number": 1,
        "histories": [{"joint_steps": 120, "start_joint_step": 0,
                       "episode_index": 0}],
        "updates": [{"joint_steps": 120}], "development": [],
        "next_eval": 20_000, "next_extraction_index": 0,
        "last_extraction_step": 0, "risk_dual_snapshot": {},
        "extractions": {}, "best": None,
        "state_sha256": runner._state_dict_sha256(state),
        "best_state_sha256": None,
    }
    payload["resume_progress_sha256"] = runner._resume_progress_sha256(payload)
    payload["used_steps"] = 121
    torch.save(payload, tmp_path / "raw.checkpoint.pt")
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        runner._train_variant(model.state_dict(), variant="raw",
                              spec=spec, directory=tmp_path)
