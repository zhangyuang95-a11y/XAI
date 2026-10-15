"""Compact frozen NN outcome traces and canonical paired-run identity."""
from __future__ import annotations

import base64

import torch

from backend.training.warehouse_v6_actor import WarehouseV6Actor, export_actor, FrozenWarehouseV6Actor
from backend.training.warehouse_v6_runner import (
    _checked_spec, _scientific_run_spec_sha256, _training_config_sha256, evaluate,
)
from experiments.warehouse.admission import _quality_trace_valid, _validated_episodes
from experiments.warehouse.acceptance import _digest, _paired_schedule_evidence
from experiments.warehouse.protocol import (
    DEVELOPMENT_SELECTION_SEEDS, PairRunSpec, SCENARIO_BLOCK_STEPS,
    scenario_schedule_sha256, scenario_seed,
)


def test_frozen_quality_trace_keeps_both_nn_actions_and_binding(tmp_path):
    torch.manual_seed(42)
    artifact = export_actor(
        WarehouseV6Actor(), tmp_path / "actor.npz", pair_seed=42,
        variant="raw", candidate_id="trace-smoke")
    frozen = FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
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
        candidate_id="hash-contract", pair_seed=6120, joint_steps=200_000,
        output_dir=str(tmp_path), checkpoint_seeds=DEVELOPMENT_SELECTION_SEEDS)
    checked = _checked_spec(spec.to_mapping())
    scientific = _scientific_run_spec_sha256(checked)
    assert scientific == spec.sha256()
    assert scientific != _training_config_sha256(checked)
    assert _training_config_sha256(checked) == spec.training_config_sha256(
        checked["source_closure_sha256"])

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
    variants = {variant: {
        "variant": variant, "pair_seed": spec.pair_seed,
        "joint_steps": spec.joint_steps,
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
        "run_spec_sha256": scientific,
        "training_config_sha256": _training_config_sha256(checked),
        "source_closure_sha256": checked["source_closure_sha256"],
        "bc_recipe_version": "warehouse-public-dual-bc.v6.2",
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
