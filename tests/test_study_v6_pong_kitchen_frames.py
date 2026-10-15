"""Accepted Pong/Kitchen v5 pairs remain exact frozen-NN runtimes in v6."""
from __future__ import annotations

from copy import deepcopy
import json
from uuid import uuid4

import pytest

from domains.kitchen import neural_domain as kitchen
from domains.kitchen import neural_policy as kitchen_policy
from domains.pong import nn_policy as pong_policy
from domains.pong import turnbased as pong
from study_v3 import RELEASE_ID, policies, registry
from study_v3.config import Settings
from study_v3.qa import simulate
from study_v3.store import Store, StudyError


@pytest.mark.parametrize("domain,engine", (("pong", pong), ("kitchen", kitchen)))
def test_each_task_frame_and_counterfactual_uses_accepted_frozen_ai(domain, engine):
    regularized = policies.binding_for_group(domain, "A")
    assert regularized == policies.binding_for_group(domain, "B")
    raw = policies.binding_for_group(domain, "C")
    assert regularized["actor_sha256"] != raw["actor_sha256"]
    assert regularized["human_actor_sha256"] != raw["human_actor_sha256"]
    assert regularized["ai_actor_sha256"] != raw["ai_actor_sha256"]
    for binding in (regularized, raw):
        policy_id = binding["policy_id"]
        assert policies._accepted(policies._catalog_entry(domain, binding["variant"]), binding)
        state = engine.initial_state(1234, 1, policy_id, release_id=RELEASE_ID)
        for _ in range(4):
            before = deepcopy(state)
            decision = engine.decide(state)
            if domain == "pong":
                expected_ai = pong_policy.choose(state, "ai", policy_id)
                expected_human = pong_policy.choose(state, "human", policy_id)
                assert decision["action"] == expected_ai
                assert decision["human_action"] == expected_human
            else:
                expected_ai = kitchen._actor(policy_id).choose(state, "ai")[0]
                expected_human = kitchen._actor(policy_id).choose(state, "human")[0]
                assert decision["action_id"] == expected_ai
                assert decision["legal_mask"][kitchen_policy.AI_ACTIONS.index(expected_ai)]
            assert decision["policy_id"] == policy_id
            human = next(action for action in engine.legal_actions(state)
                         if action != expected_human)
            counterfactual = simulate(engine, state, decision, [human], horizon=1)
            assert counterfactual["trace"][0]["ai_action"] == expected_ai
            assert state == before
            state = engine.step(state, human, decision)
            assert state["policy_id"] == policy_id
            assert state.get("actor_sha256", state.get("policy_sha256")) == \
                binding["actor_sha256"]


@pytest.mark.parametrize("domain,engine", (("pong", pong), ("kitchen", kitchen)))
def test_ai_ai_demo_uses_the_same_accepted_pair(domain, engine):
    for group in ("A", "C"):
        binding = policies.binding_for_group(domain, group)
        demo = registry.demonstration(domain, binding["policy_id"])
        assert len(demo["frames"]) > 1
        if domain == "pong":
            assert all(frame["actor_sha256"] == binding["actor_sha256"]
                       for frame in demo["frames"])
            for first, second in zip(demo["frames"], demo["frames"][1:]):
                for role in ("human", "ai"):
                    choice = pong_policy.choose(first, role, binding["policy_id"])
                    assert second[role]["x"] == first[role]["x"] + engine._DELTAS[choice]
        else:
            assert demo["actor_sha256"] == binding["actor_sha256"]
            assert all(frame["ai"]["policy"]["policy_id"] == binding["policy_id"]
                       for frame in demo["frames"])


@pytest.mark.parametrize("domain,engine", (("pong", pong), ("kitchen", kitchen)))
def test_saved_frames_and_recovery_recheck_the_bound_nn(domain, engine, tmp_path):
    database = tmp_path / "study.sqlite3"
    study = Store(Settings(database=str(database)))
    try:
        token, view = study.create({"consent": True, "domain": domain,
                                    "mode": "test", "group": "A",
                                    "participant_id": f"v6-{domain}-frame"}, admin=True)
        binding = policies.binding_for_group(domain, "A")
        assert view["actor_sha256"] == binding["actor_sha256"]

        def command(current, kind, **extra):
            return study.command(token, kind, {"instance_id": current["instance_id"],
                "revision": current["revision"], "command_id": uuid4().hex,
                **extra})

        view = command(view, "demo_skip")
        if domain == "kitchen":
            assert view["stage"] == "practice"
            view = command(view, "practice_skip")
        assert view["stage"] == "task1"
        run_id = view["run_id"]
        for turn in range(3):
            with study.db.transaction(read_only=True) as db:
                frame = db.one("SELECT state_json,decision_json FROM pl3_frames "
                               "WHERE run_id=? AND turn=?", (run_id, turn))
            state = json.loads(frame["state_json"])
            decision = json.loads(frame["decision_json"])
            expected = (pong_policy.choose(state, "ai", binding["policy_id"])
                        if domain == "pong" else
                        kitchen._actor(binding["policy_id"]).choose(state, "ai")[0])
            assert decision["action" if domain == "pong" else "action_id"] == expected
            assert study.frame(token, view["instance_id"], run_id, turn)["state"] == \
                engine.public_state(state)
            view = command(view, "action", run_id=run_id, turn=turn,
                           action=engine.legal_actions(state)[0])
        restored = Store(Settings(database=str(database)))
        try:
            recovered = restored.recover_view(token, domain)
            assert recovered["actor_sha256"] == binding["actor_sha256"]
            assert restored.frame(token, view["instance_id"], run_id, 3)["state"] == \
                view["state"]
        finally:
            restored.db.close()
        with study.db.transaction() as db:
            row = db.one("SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=1",
                         (run_id,))
            forged = json.loads(row["decision_json"])
            forged["action" if domain == "pong" else "action_id"] = "forged"
            db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=1",
                       (json.dumps(forged), run_id))
        with pytest.raises(StudyError, match="recorded_decision_mismatch"):
            study.frame(token, view["instance_id"], run_id, 1)
    finally:
        study.db.close()
