"""Reject invalid development identities before collection; replay same NN frames."""
from copy import deepcopy
import json

import pytest

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import WarehouseV6Actor, FrozenWarehouseV6Actor, export_actor
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
from experiments.warehouse import v619_protocol as v
from experiments.warehouse.quality_replay import replay_episode
from scripts import run_warehouse_v619_fixed_validation as script


@pytest.fixture
def frozen_actor(tmp_path):
    asset = export_actor(WarehouseV6Actor(), tmp_path / "unit.actor.npz",
                         pair_seed=6150, variant="raw", candidate_id="unit-test")
    return FrozenWarehouseV6Actor.load(asset["path"], expected_sha256=asset["sha256"],
                                     expected_role_sha256=asset["role_sha256"])


def test_registered_splits_and_draws_are_disjoint_and_not_sealed():
    plan = v.seed_plan()
    assert [len(plan["splits"][s]) for s in v.SPLITS] == [50, 25, 25, 24]
    fit = plan["splits"]["fit"]
    assert [r["sampling_mode"] for r in fit] == ["sampled"]*25 + ["argmax"]*25
    fitting = [r["scenario_seed"] for rows in plan["splits"].values() for r in rows]
    independent = plan["independent"]
    assert len(set(fitting)) == 124
    assert not set(fitting) & {r["scenario_seed"] for r in independent}
    assert len(independent) == 216
    assert len({(r["base"], r["task"], r["stream"]) for r in independent}) == 216
    assert {r["action_seed"] for r in independent} == set(range(1700000000, 1700000216))
    assert not any(is_sealed_scenario_seed(s) for s in fitting + [r["scenario_seed"] for r in independent])


@pytest.mark.parametrize("approved,digest", [(False, None), (True, "0"*64)])
def test_uncleared_or_changed_seed_ledger_rejects_before_actor_load(tmp_path, monkeypatch, approved, digest):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"approved_for_fixed_validation": approved,
                               "fixed_validation_seed_plan_sha256": digest}))
    monkeypatch.setattr(script, "checked_actor", lambda: pytest.fail("loaded before ledger check"))
    with pytest.raises(RuntimeError, match="cleared seed ledger"):
        script.prepare(path)


def test_alternative_arm_rejects_before_collection(tmp_path, frozen_actor, monkeypatch):
    artifact = {"path": "unused", "sha256": frozen_actor.artifact_sha256}
    monkeypatch.setattr(script, "checked_actor", lambda: (frozen_actor, artifact))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"primary_arm": "P1"}))
    monkeypatch.setattr(runner, "collect_authorized_program_rows", lambda *a, **kw: pytest.fail("collection started"))
    with pytest.raises(RuntimeError, match="immutable identity"):
        script.execute(path)


def test_leaf_tree_and_protocol_changes_are_inside_source_guard(tmp_path, frozen_actor, monkeypatch):
    for source in ("backend/training/warehouse_v619_leaf_fit.py",
                   "backend/training/warehouse_v619_tree.py",
                   "experiments/warehouse/v619_protocol.py"):
        assert source in runner._SOURCE_FILES
    artifact = {"path": "unused", "sha256": frozen_actor.artifact_sha256}
    monkeypatch.setattr(script, "checked_actor", lambda: (frozen_actor, artifact))
    monkeypatch.setattr(script, "ROOT", tmp_path)
    directory = tmp_path / v.OUTPUT_ROOT / "fixed-validation-01"
    directory.mkdir(parents=True)
    plan = {"output_directory": str(directory), "candidate_id": v.CANDIDATE_ID,
        "primary_arm": "P2", "method": v.METHOD, "actor": artifact,
        "seed_plan": v.seed_plan(), "publishable": False, "PPO_joint_steps": 0,
        "validation_can_select_another_arm": False,
        "independent_collection_requires_internal_validation_pass": True,
        "source_closure_sha256": "0"*64}
    path = directory / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, "source_closure_sha256", lambda: "1"*64)
    monkeypatch.setattr(runner, "collect_authorized_program_rows", lambda *a, **kw: pytest.fail("collection started"))
    with pytest.raises(RuntimeError, match="source, Actor or ledger changed"):
        script.execute(path)
    assert not (directory / "started.json").exists()


def test_same_collected_sampled_frames_replay_without_new_quality_evaluation(frozen_actor):
    rows, evidence = runner.collect_authorized_program_rows(frozen_actor, [263101],
        action_seed_offset=500, include_episode_outcomes=True)
    assert evidence["sampling_mode"] == "sampled"
    outcome = evidence["episode_outcomes"][0]
    assert outcome["action_seed"] == 263601
    quality = script.quality_row_from_recorded_nn(rows, outcome, frozen_actor)
    replayed = replay_episode(frozen_actor, quality, scenario_seed=263101, action_seed=263601)
    assert replayed["frame_count"] == outcome["frame_count"]
    changed = deepcopy(rows)
    changed["robot_2"][0]["post_state_public_sha256"] = "0"*64
    with pytest.raises(RuntimeError, match="not aligned"):
        script.quality_row_from_recorded_nn(changed, outcome, frozen_actor)


def test_argmax_visitation_and_invalid_options(frozen_actor):
    rows, evidence = runner.collect_authorized_program_rows(frozen_actor, [263102],
        deterministic=True, include_episode_outcomes=True)
    assert evidence["sampling_mode"] == "argmax"
    assert all(r["sampled_action"] == r["nn_argmax_action"] for role in v.ROLES for r in rows[role])
    for options in ({"deterministic": 1}, {"action_seed_offset": True},
                    {"action_seed_offset": -999999}, {"include_episode_outcomes": 1}):
        with pytest.raises(ValueError, match="visitation mode"):
            runner.collect_authorized_program_rows(frozen_actor, [263102], **options)


def test_default_collection_retains_legacy_call_signature(frozen_actor, monkeypatch):
    calls = []
    def legacy(actor, seeds):
        calls.append(list(seeds))
        return {role: [] for role in v.ROLES}
    monkeypatch.setattr(runner, "collect_program_audit_rows", legacy)
    monkeypatch.setattr(runner, "_authorized_frozen_operation", lambda actor, seeds, auth, operation, **kw: (operation(), {}))
    rows, evidence = runner.collect_authorized_program_rows(frozen_actor, [263100])
    assert calls == [[263100]] and evidence == {} and not rows["robot_1"]


def test_interrupted_collection_preserves_consumed_reset_before_temporary_journal_disappears(frozen_actor, monkeypatch):
    def interrupted(actor, seeds):
        environment = runner._environment(training_reward=False)
        environment.reset(seed=seeds[0])
        raise RuntimeError("interrupt after actual reset")
    monkeypatch.setattr(runner, "collect_program_audit_rows", interrupted)
    with pytest.raises(RuntimeError, match="interrupt after actual reset") as captured:
        runner.collect_authorized_program_rows(frozen_actor, [263103])
    audit = captured.value.warehouse_reset_audit
    assert audit["actual_reset_count"] == 1
    assert audit["resets"][0]["seed"] == 263103
    assert len(audit["journal_sha256"]) == 64
    assert runner._ACTIVE_LOCKED_EXECUTION.get() is None
