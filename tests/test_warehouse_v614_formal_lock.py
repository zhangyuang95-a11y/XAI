"""Formal authorization and reset-sequence contracts; no environment reset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.training import warehouse_v6_runner as runner
from env.warehouse.reset_audit_v614 import (
    capture_stage1_resets, record_successful_reset, stage1_reset_phase,
)
from experiments.warehouse import protocol


def test_exploratory_successor_bc_and_critic_seeds_are_fresh():
    old = {seed for rows in runner.exploratory_stage1_bc_seed_schedule(12469).values()
           for seed in rows}
    next_pair = {seed for rows in runner.exploratory_stage1_bc_seed_schedule(
        12470).values() for seed in rows}
    old_critic = protocol.bc_critic_reset_schedule(12469,
                                                   protocol.INITIAL_RELEASE_ID)["seed"]
    next_critic = protocol.bc_critic_reset_schedule(12470,
                                                    protocol.INITIAL_RELEASE_ID)["seed"]
    assert len(old) == len(next_pair) == 1_394
    assert not (old | {12469, old_critic}) & (
        next_pair | {12470, next_critic})


def _checked_spec(output_dir: Path) -> tuple[dict, protocol.PairRunSpec]:
    pair = protocol.PairRunSpec(
        candidate_id="v614-static-formal-lock", pair_seed=protocol.DEVELOPMENT_PAIR_SEEDS[0],
        joint_steps=protocol.BUDGETS[0], output_dir=str(output_dir),
        program_fit_method=protocol.PROGRAM_FIT_METHOD_CANDIDATES[0],
        checkpoint_seeds=protocol.DEVELOPMENT_SELECTION_SEEDS)
    raw = pair.to_mapping()
    directory = (output_dir / pair.candidate_id /
                 f"pair_{pair.pair_seed}_steps_{pair.joint_steps}")
    raw.update(warmup_actor_path=str(directory / "warmup.actor.npz"),
               final_bc_actor_path=str(directory / "final-bc.actor.npz"),
               final_bc_precheck_receipt_path=str(directory / "final-bc-precheck.json"))
    return runner._checked_spec(raw), pair


def _fake_pair_receipt(spec: dict, pair: protocol.PairRunSpec) -> dict:
    rows = []
    cursor = 0
    while cursor < spec["joint_steps"]:
        length = min(protocol.SCENARIO_BLOCK_STEPS, spec["joint_steps"] - cursor)
        seed = protocol.scenario_seed(spec["scenario_namespace"],
                                      spec["pair_seed"], cursor)
        index = len(rows)
        rows.append({"seed": seed, "episode_index": index,
                     "episode_key": f"{seed}:{index}",
                     "start_joint_step": cursor, "joint_steps": length,
                     "complete": False, "shutdown_episode": False})
        cursor += length
    start = {"robot_1": "a" * 64, "robot_2": "b" * 64}
    variant = {"actual_episode_resets": rows,
               "actual_episode_resets_sha256": runner._digest_json(rows),
               "scenario_schedule_sha256": protocol.scenario_schedule_sha256(
                   spec["scenario_namespace"], spec["pair_seed"], spec["joint_steps"]),
               "checkpoint_seeds": spec["checkpoint_seeds"],
               "development": [{"joint_steps": spec["joint_steps"]}],
               "initial_role_sha256": start}
    return {"pair_seed": spec["pair_seed"], "joint_steps": spec["joint_steps"],
            "source_closure_sha256": spec["source_closure_sha256"],
            "run_spec_sha256": pair.sha256(),
            "training_config_sha256": runner._training_config_sha256(spec),
            "program_fit_method": spec["program_fit_method"],
            "bc": {"bc_data_seed_blocks": spec["bc_data_seed_schedule"],
                   "pretrained_role_sha256": start},
            "variants": {"raw": dict(variant), "regularized": dict(variant)},
            "actual_episode_resets_equal": True}


def test_formal_spec_hashes_and_full_dynamic_reset_reconciliation(tmp_path):
    spec, pair = _checked_spec(tmp_path)
    assert runner._scientific_run_spec_sha256(spec) == pair.sha256()
    assert runner._training_config_sha256(spec) == pair.training_config_sha256(
        spec["source_closure_sha256"])
    expected = runner._formal_expected_reset_sequence(
        spec, _fake_pair_receipt(spec, pair))
    assert len(expected) == 4_902
    assert [row["phase"] for row in expected[:1_396]] == ["formal_bc"] * 1_396
    assert runner._formal_partial_reset_prefix(spec, expected)
    assert runner._formal_partial_reset_prefix(spec, expected[:1_397])
    changed = [dict(row) for row in expected]
    changed[1_396]["seed"] += 1
    assert not runner._formal_partial_reset_prefix(spec, changed)
    changed = [dict(row) for row in expected]
    changed[-1]["seed"] += 1
    assert not runner._formal_partial_reset_prefix(spec, changed)


def test_formal_authorization_rejects_unpinned_script_before_environment(tmp_path):
    spec, pair = _checked_spec(tmp_path)
    raw_spec = {**pair.to_mapping(), **{key: spec[key] for key in (
        "warmup_actor_path", "final_bc_actor_path",
        "final_bc_precheck_receipt_path")}}
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "authorization_kind": "unlocked",
        "reset_audit_path": str(tmp_path / "audit.json"),
        "reset_audit_journal_path": str(tmp_path / "resets.jsonl"),
    }))
    with pytest.raises(RuntimeError, match="authorization identity drifted"):
        runner.run_locked_formal_pair(raw_spec,
            authorization_receipt_path=authorization)
    assert not (tmp_path / spec["candidate_id"]).exists()
    assert not (tmp_path / "resets.jsonl").exists()


def test_formal_reset_journal_binds_rollout_step_and_episode(tmp_path):
    journal = tmp_path / "resets.jsonl"
    with capture_stage1_resets(journal) as rows:
        with pytest.raises(RuntimeError, match="metadata"):
            with stage1_reset_phase("formal_raw_rollout"):
                pass
        with stage1_reset_phase("formal_raw_rollout", start_joint_step=120,
                                episode_index=2):
            record_successful_reset(900, 1, "study-layout", 2.0, 10.0)
    assert rows == [json.loads(journal.read_text().strip())]
    assert rows[0]["start_joint_step"] == 120
    assert rows[0]["episode_index"] == 2
