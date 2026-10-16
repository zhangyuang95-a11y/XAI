"""The v6 Warehouse runtime uses the bound frozen pair on every action path.

These random but valid Actors exercise execution mechanics only. Scientific
admission is covered by the separate immutable receipt and package tests.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import pytest
import torch
import numpy as np

from backend.training.warehouse_v6_actor import WarehouseV6Actor, export_actor
from domains.warehouse import neural_turnbased as warehouse
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.domain import participant_study_config
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import action_masks
from env.warehouse.public_observation_v6 import (
    encode_public_observation, rebuild_public_pre_action_state_from_frames,
)
from study_v3 import policies, registry
from study_v3 import store as store_module
from study_v3.config import Settings
from study_v3.qa import simulate
from study_v3.store import Store, StudyError


@pytest.fixture
def bound_v6(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    entries = {}
    for index, variant in enumerate(("raw", "regularized")):
        torch.manual_seed(6010 + index)
        path = tmp_path / f"warehouse-{variant}.npz"
        artifact = export_actor(WarehouseV6Actor(), path, pair_seed=6101,
                                variant=variant, candidate_id="frame-test")
        program_paths = []
        program_hashes = {}
        if variant == "regularized":
            for role in ("robot_1", "robot_2"):
                program = tmp_path / f"{role}.program.json"
                program.write_text(json.dumps({"role": role}))
                program_paths.append(program.name)
                program_hashes[role] = sha256(program.read_bytes()).hexdigest()
        entries[variant] = {
            "policy_id": f"warehouse-{variant}-v6",
            "model_version": warehouse.V6_MODEL_VERSION,
            "feature_schema_sha256": warehouse.V6_FEATURE_SCHEMA_SHA256,
            "action_physics_config_sha256": warehouse.ACTION_PHYSICS_CONFIG_SHA256,
            "actor_path": path.name, "actor_sha256": artifact["sha256"],
            "role_sha256": {"human": artifact["role_sha256"]["robot_1"],
                            "ai": artifact["role_sha256"]["robot_2"]},
            "program_paths": program_paths, "program_sha256": program_hashes,
            "readiness": True,
        }

    monkeypatch.setattr(warehouse, "ROOT", tmp_path)
    monkeypatch.setattr(policies, "ROOT", tmp_path)
    monkeypatch.setattr(warehouse, "policy_artifacts_v6", lambda release_id=warehouse.V6_RELEASE_ID: entries)
    original_catalog_entry = policies._catalog_entry
    monkeypatch.setattr(policies, "_catalog_entry", lambda domain, variant,
                        release_id=warehouse.V6_RELEASE_ID:
                        entries[variant] if domain == "warehouse"
                        else original_catalog_entry(domain, variant,
                                                    release_id=release_id))

    def verify_fixture(policy_id):
        if policy_id not in warehouse._V6_POLICY_IDS:
            raise RuntimeError("fixture requires v6 Warehouse policy")
        item = entries[warehouse._V6_POLICY_IDS[policy_id]]
        actor = tmp_path / item["actor_path"]
        if not actor.is_file() or sha256(actor.read_bytes()).hexdigest() != item["actor_sha256"]:
            raise RuntimeError("Warehouse v6 accepted policy binding is unavailable")

    monkeypatch.setattr(warehouse, "_verify_v5_artifact", verify_fixture)
    monkeypatch.setattr(registry, "demonstration", lambda domain, policy_id:
                        registry.engine(domain).demonstration(policy_id))
    monkeypatch.setattr(store_module, "demonstration", registry.demonstration)
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    policies._validated_v6_actor.cache_clear()
    yield entries, tmp_path
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    policies._validated_v6_actor.cache_clear()


def _expected(state):
    env = warehouse._restore(state)
    actions, _ = warehouse._actor(state["policy_id"]).act(
        env.state, base_seed=state["scenario_seed"])
    masks = action_masks(env.state, env.config.map_layout_id)
    for index, role in enumerate(("robot_1", "robot_2")):
        assert masks[index, ACTIONS.index(actions[role])] == 1
    return actions


def test_historical_warehouse_public_frame_omits_v6_history_fields():
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=1802)
    state = warehouse._state(env, 1, 1802,
                             policy_id=warehouse.LEGACY_POLICY_ID)
    public = warehouse.public_state(state)
    for key in ("episode_id", "previous_joint_actions",
                "previous_collision_kind", "same_target_collision_streak"):
        assert key not in public
    historical = deepcopy(state)
    unpacked = warehouse.legacy._unpack(historical["snapshot"]["state"])
    unpacked.pop("same_target_collision_streak")
    historical["snapshot"]["state"] = warehouse.legacy._pack(unpacked)
    assert warehouse.public_state(historical) == public


def test_v6_pair_binding_and_direct_human_takeover(bound_v6):
    entries, tmp_path = bound_v6
    a = policies.binding_for_group("warehouse", "A")
    b = policies.binding_for_group("warehouse", "B")
    c = policies.binding_for_group("warehouse", "C")
    assert a == b
    assert a["actor_sha256"] != c["actor_sha256"]
    assert a["human_actor_sha256"] != c["human_actor_sha256"]
    assert a["ai_actor_sha256"] != c["ai_actor_sha256"]
    assert len(a["program_hashes"]) == 2 and c["program_hashes"] == {}
    for policy_id in (a["policy_id"], c["policy_id"]):
        state = warehouse.initial_state(1800, 1, policy_id,
                                        release_id=warehouse.V6_RELEASE_ID)
        public_frames = [warehouse.public_state(state)]
        assert public_frames[0]["previous_joint_actions"] is None
        for _ in range(4):
            chosen = _expected(state)
            decision = warehouse.decide(state)
            assert decision["action"] == chosen["robot_2"].lower()
            assert decision["controller_trace"]["selection_changed_policy"] is False
            assert decision["controller_trace"]["selected_actions"]["robot_2"] == chosen["robot_2"]
            human = next(action for action in warehouse.legal_actions(state)
                         if action != chosen["robot_1"].lower())
            next_state = warehouse.step(state, human, decision)
            assert next_state["last_actions"] == {"human": human,
                                                   "ai": chosen["robot_2"].lower()}
            assert next_state["actor_sha256"] == state["actor_sha256"]
            assert next_state["role_sha256"] == state["role_sha256"]
            assert next_state["release_id"] == warehouse.V6_RELEASE_ID
            visible = warehouse.public_state(next_state)
            public_frames.append(visible)
            assert visible["previous_joint_actions"]["human"]["proposed"] == human
            assert visible["previous_joint_actions"]["ai"]["proposed"] == chosen["robot_2"].lower()
            frame_rebuilt = rebuild_public_pre_action_state_from_frames(public_frames)
            np.testing.assert_array_equal(
                encode_public_observation(frame_rebuilt),
                encode_public_observation(warehouse._restore(next_state).state))
            assert simulate(warehouse, state, decision, [human], horizon=1)["trace"][0][
                "ai_action"] == decision["action"]
            forged = deepcopy(decision)
            forged["action"] = "left" if decision["action"] != "left" else "right"
            with pytest.raises(ValueError, match="decision does not match"):
                warehouse.step(state, human, forged)
            state = next_state
        actor_path = tmp_path / entries[warehouse._V6_POLICY_IDS[policy_id]]["actor_path"]
        original = actor_path.read_bytes()
        actor_path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match="binding is unavailable"):
            warehouse.decide(state)
        actor_path.write_bytes(original)


def test_v6_human_illegal_physical_move_is_proposed_but_not_executed(bound_v6):
    state = warehouse.initial_state(1801, 1, "warehouse-raw-v6",
                                    release_id=warehouse.V6_RELEASE_ID)
    env = warehouse._restore(state)
    altered = env.get_state()
    altered.by_id("robot_1").position = (1, 1)
    env.set_state(altered)
    state = warehouse._state(env, state["task"], state["seed"],
                             policy_id=state["policy_id"])
    state["scenario_seed"] = 18011
    assert warehouse.legal_actions(state, "human") == [item.lower() for item in ACTIONS]
    next_state = warehouse.step(state, "up")
    prior = warehouse.public_state(next_state)["previous_joint_actions"]
    assert prior["human"] == {"proposed": "up", "executed": "wait"}
    assert next_state["last_actions"]["human"] == "up"
    with pytest.raises(ValueError, match="public history differs"):
        forged = deepcopy(next_state)
        forged["previous_joint_actions"]["human"]["proposed"] = "wait"
        warehouse._restore(forged)


def test_v6_demo_is_two_nn_and_session_replay_is_pinned(bound_v6, monkeypatch):
    _, tmp_path = bound_v6
    policy_id = "warehouse-regularized-v6"
    actor = warehouse._actor(policy_id)
    original_step = WarehouseMultiAgentEnv.step
    seen = []
    demo_seed = warehouse._v6_config()["demo_seed"]

    def check_demo(self, actions, *args, **kwargs):
        expected, _ = actor.act(self.state, base_seed=demo_seed)
        assert actions == expected
        seen.append(deepcopy(actions))
        return original_step(self, actions, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(WarehouseMultiAgentEnv, "step", check_demo)
        demo = warehouse.demonstration(policy_id, release_id=warehouse.V6_RELEASE_ID)
    assert seen and len(seen) == len(demo["frames"]) - 1
    assert demo["provenance"]["actor_sha256"] == actor.artifact_sha256
    for action, frame in zip(seen, demo["frames"][1:], strict=True):
        assert frame["demo_actions"] == {
            "human": action["robot_1"].lower(), "ai": action["robot_2"].lower()}

    database = tmp_path / "study.sqlite3"
    study = Store(Settings(database=str(database)))
    try:
        token, view = study.create({"consent": True, "domain": "warehouse",
                                    "mode": "test", "group": "A",
                                    "participant_id": "v6-frame-test"}, admin=True)
        assert view["policy_id"] == policy_id
        assert view["actor_sha256"] == actor.artifact_sha256
        view = study.command(token, "demo_skip", {"instance_id": view["instance_id"],
            "revision": view["revision"], "command_id": uuid4().hex})
        assert view["stage"] == "task1"
        run_id = view["run_id"]
        for turn in range(3):
            with study.db.transaction(read_only=True) as db:
                row = db.one("SELECT * FROM pl3_frames WHERE run_id=? AND turn=?",
                             (run_id, turn))
            state = json.loads(row["state_json"])
            decision = json.loads(row["decision_json"])
            assert decision["action"] == _expected(state)["robot_2"].lower()
            assert study.frame(token, view["instance_id"], run_id, turn)["state"] == \
                warehouse.public_state(state)
            human = next(action for action in warehouse.legal_actions(state)
                         if action != _expected(state)["robot_1"].lower())
            view = study.command(token, "action", {"instance_id": view["instance_id"],
                "revision": view["revision"], "command_id": uuid4().hex,
                "run_id": run_id, "turn": turn, "action": human})
        restored = Store(Settings(database=str(database)))
        try:
            recovered = restored.recover_view(token, "warehouse")
            assert recovered["actor_sha256"] == actor.artifact_sha256
            assert restored.frame(token, view["instance_id"], run_id, 3)["state"] == view["state"]
        finally:
            restored.db.close()
        with study.db.transaction() as db:
            row = db.one("SELECT public_json FROM pl3_frames WHERE run_id=? AND turn=1",
                         (run_id,))
            original_public = row["public_json"]
            forged_public = json.loads(original_public)
            forged_public["previous_joint_actions"]["human"]["proposed"] = "forged"
            db.execute("UPDATE pl3_frames SET public_json=? WHERE run_id=? AND turn=1",
                       (json.dumps(forged_public), run_id))
        with pytest.raises(StudyError, match="recorded_frame_mismatch"):
            study.frame(token, view["instance_id"], run_id, 1)
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_frames SET public_json=? WHERE run_id=? AND turn=1",
                       (original_public, run_id))
        with study.db.transaction() as db:
            row = db.one("SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=1",
                         (run_id,))
            forged = json.loads(row["decision_json"])
            forged["action"] = "forged"
            db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=1",
                       (json.dumps(forged), run_id))
        with pytest.raises(StudyError, match="recorded_decision_mismatch"):
            study.frame(token, view["instance_id"], run_id, 1)
    finally:
        study.db.close()
