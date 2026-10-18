"""Compact frozen NN outcome traces and canonical paired-run identity."""
from __future__ import annotations

import base64

import pytest
import torch

from backend.training.warehouse_v6_actor import WarehouseV6Actor, export_actor, FrozenWarehouseV6Actor
from backend.training.warehouse_v6_runner import (
    _checked_spec, _scientific_run_spec_sha256, _training_config_sha256, evaluate,
)
from experiments.warehouse.admission import _quality_trace_valid, _validated_episodes
from experiments.warehouse.acceptance import _digest, _paired_schedule_evidence
from experiments.warehouse.protocol import (
    DEVELOPMENT_SELECTION_SEEDS, FINAL_PROGRAM_SPLIT_COUNTS, PairRunSpec,
    REGISTERED_BC_SPEC, REGISTERED_PROGRAM_FIT_METHOD, ROLES,
    SCENARIO_BLOCK_STEPS,
    scenario_schedule_sha256, scenario_seed,
)
from tests.warehouse_v6_contract_fixture import bc_receipt, topology_freeze_evidence
from tests.warehouse_v614_test_scopes import isolated_evaluation_scope


def test_frozen_quality_trace_keeps_both_nn_actions_and_binding(tmp_path):
    torch.manual_seed(42)
    artifact = export_actor(
        WarehouseV6Actor(), tmp_path / "actor.npz", pair_seed=42,
        variant="raw", candidate_id="trace-smoke")
    frozen = FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    with isolated_evaluation_scope():
        measured = evaluate(frozen, [1013001], action_seed_offset=4, trace=True)
    row = _validated_episodes(
        measured, [1013001], require_trace=True,
        actor_sha256=artifact["sha256"], action_seed_offset=4)[0]
    evidence = row["quality_trace"]
    actions = base64.b64decode(evidence["actions_b64"], validate=True)
    assert len(actions) == evidence["frame_count"]
    assert evidence["actor_sha256"] == artifact["sha256"]
    assert evidence["actor_role_sha256"] == artifact["role_sha256"]
    assert evidence["action_seed"] == 1013005
    assert all(0 <= value < 25 for value in actions)
    tampered = dict(evidence, actions_b64=base64.b64encode(actions + b"\0").decode())
    assert not _quality_trace_valid(tampered, 1013001, 1013005,
                                    artifact["sha256"])


def test_preregistered_spec_hash_is_identical_in_runner_and_acceptance(tmp_path):
    spec = PairRunSpec(
        candidate_id="hash-contract", pair_seed=6140, joint_steps=200_000,
        output_dir=str(tmp_path),
        program_fit_method=REGISTERED_PROGRAM_FIT_METHOD,
        checkpoint_seeds=DEVELOPMENT_SELECTION_SEEDS)
    run_spec = {
        **spec.to_mapping(),
        "warmup_actor_path": str(tmp_path / "warmup.actor.npz"),
        "final_bc_actor_path": str(tmp_path / "final-bc.actor.npz"),
        "final_bc_precheck_receipt_path": str(tmp_path / "precheck.json"),
    }
    checked = _checked_spec(run_spec)
    assert checked["architecture_mode"] == spec.architecture_mode
    assert checked["program_fit_method"] == spec.program_fit_method
    assert checked["bc_recipe_version"] == spec.bc_recipe_version
    assert checked["online_program_extraction_steps"] == list(
        spec.online_program_extraction_steps)
    scientific = _scientific_run_spec_sha256(checked)
    assert scientific == spec.sha256()
    assert scientific != _training_config_sha256(checked)
    assert _training_config_sha256(checked) == spec.training_config_sha256(
        checked["source_closure_sha256"])
    for field, value, message in (
            ("architecture_mode", "unregistered", "registered architecture"),
            ("program_fit_method", "axis_tree_v1", "not registered"),
            ("bc_recipe_version", "unregistered", "registered architecture"),
            ("online_program_extraction_steps", [150_000, 170_000, 186_000],
             "registered architecture")):
        with pytest.raises(ValueError, match=message):
            _checked_spec({**run_spec, field: value})

    resets = []
    for index, start in enumerate(range(0, spec.joint_steps, SCENARIO_BLOCK_STEPS)):
        assigned = scenario_seed(spec.scenario_namespace, spec.pair_seed, start)
        resets.append({
            "seed": assigned, "episode_index": index,
            "episode_key": f"{assigned}:{index}", "start_joint_step": start,
            "joint_steps": min(SCENARIO_BLOCK_STEPS, spec.joint_steps - start),
            "complete": True, "shutdown_episode": False,
        })
    schedule_sha = scenario_schedule_sha256(
        spec.scenario_namespace, spec.pair_seed, spec.joint_steps)
    bc_schedule = spec.to_mapping()["bc_data_seed_schedule"]
    refresh_history = [{
        "joint_step": step,
        "role_tree_sha256": {role: "c" * 64 for role in ROLES},
        "route_parity_passed": True, "path_legality_passed": True,
        "core_qualification_recomputed": True,
    } for step in spec.topology_refresh_steps]
    variants = {variant: {
        "variant": variant, "pair_seed": spec.pair_seed,
        "joint_steps": spec.joint_steps,
        "architecture_mode": spec.architecture_mode,
        "program_fit_method": spec.program_fit_method,
        "bc_recipe_version": spec.bc_recipe_version,
        "bc_recipe_sha256": spec.bc_recipe_sha256,
        "action_physics_config_sha256": REGISTERED_BC_SPEC[
            "action_physics_config_sha256"],
        "bc_data_seed_schedule_sha256": spec.bc_data_seed_schedule_sha256,
        "risk_dual_config_sha256": spec.risk_dual_config_sha256,
        "online_program_extraction_steps": list(
            spec.online_program_extraction_steps),
        "topology_refresh_steps": list(spec.topology_refresh_steps),
        "topology_refresh_history": refresh_history,
        **topology_freeze_evidence(spec.joint_steps),
        "online_program_split_block_cycle": FINAL_PROGRAM_SPLIT_COUNTS,
        "online_program_cohorts_disjoint": True,
        "run_spec_sha256": scientific,
        "training_config_sha256": _training_config_sha256(checked),
        "scenario_schedule_sha256": schedule_sha,
        "assigned_scenario_schedule_kind": "fixed_120_joint_step_blocks",
        "actual_episode_resets": resets,
        "actual_episode_resets_sha256": _digest(resets),
    } for variant in ("raw", "regularized")}
    pair = {
        "candidate_id": spec.candidate_id, "release_id": spec.release_id,
        "pair_seed": spec.pair_seed, "joint_steps": spec.joint_steps,
        "architecture_mode": spec.architecture_mode,
        "program_fit_method": spec.program_fit_method,
        "online_program_extraction_steps": list(
            spec.online_program_extraction_steps),
        "topology_refresh_steps": list(spec.topology_refresh_steps),
        "topology_frozen_at_step": spec.topology_frozen_at_step,
        "run_spec_sha256": scientific,
        "training_config_sha256": _training_config_sha256(checked),
        "source_closure_sha256": checked["source_closure_sha256"],
        "bc_recipe_version": spec.bc_recipe_version,
        "bc_recipe_sha256": spec.bc_recipe_sha256,
        "action_physics_config_sha256": REGISTERED_BC_SPEC[
            "action_physics_config_sha256"],
        "bc_data_seed_schedule": bc_schedule,
        "bc_data_seed_schedule_sha256": spec.bc_data_seed_schedule_sha256,
        "risk_dual_config_sha256": spec.risk_dual_config_sha256,
        "bc": bc_receipt(spec.pair_seed, checked["source_closure_sha256"]),
        "assigned_scenario_schedule_sha256": schedule_sha,
        "checkpoint_seeds": list(spec.checkpoint_seeds),
        "variants": variants,
        "actual_episode_reset_count": {variant: len(resets) for variant in variants},
        "actual_episode_resets_equal": True,
    }
    assert _paired_schedule_evidence(pair)
    pair["variants"]["raw"]["run_spec_sha256"] = _training_config_sha256(checked)
    assert not _paired_schedule_evidence(pair)
    pair["variants"]["raw"]["run_spec_sha256"] = scientific
    pair["training_config_sha256"] = "0" * 64
    assert not _paired_schedule_evidence(pair)
