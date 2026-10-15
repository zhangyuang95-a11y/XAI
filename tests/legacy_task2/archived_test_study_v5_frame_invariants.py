"""End-to-end v5 frame checks with temporary frozen NN pairs, no training run.

The test data have local identity receipts but lack full release evidence.
Preview sessions exercise execution while the central pilot gate stays closed.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from backend.training.warehouse_dual_v5 import DualNN, export_actor
from env.warehouse.environment import WarehouseMultiAgentEnv
from domains.kitchen import engine as kitchen_physics
from domains.kitchen import neural_domain as kitchen
from domains.kitchen import neural_policy as kitchen_policy
from domains.pong import nn_policy as pong_policy
from domains.pong import turnbased as pong
from domains.warehouse import neural_turnbased as warehouse
from study_v3 import policies, registry
from study_v3.config import Settings
from study_v3.qa import Explainer, _digest, simulate
from study_v3.store import Store, StudyError
from study_v3 import store as store_module


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _warehouse_pair(tmp_path, monkeypatch):
    config = deepcopy(warehouse.V5_CONFIG)
    for index, variant in enumerate(("raw", "regularized")):
        torch.manual_seed(5100 + index)
        artifact = export_actor(DualNN(hidden=32), tmp_path / f"warehouse-{variant}.npz",
                                pair_seed=5100 + index, variant=variant)
        config["variants"][variant].update(
            actor_path=Path(artifact["path"]).name,
            actor_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"])
        paths = []
        if variant == "regularized":
            for role in ("robot_1", "robot_2"):
                path = tmp_path / f"warehouse-{role}.program.json"
                path.write_text(json.dumps({"role": role, "fixture": True}))
                paths.append(path.name)
        config["variants"][variant]["program_paths"] = paths
    monkeypatch.setattr(warehouse, "ROOT", tmp_path)
    monkeypatch.setattr(warehouse, "V5_CONFIG", config)
    # These deliberately unaccepted preview fixtures test action/replay
    # mechanics. The separate direct-entrypoint tests cover the release gate.
    def fixture_actor_hash_only(policy_id):
        if policy_id == warehouse.LEGACY_POLICY_ID:
            return
        record = config["variants"][warehouse._V5_POLICY_IDS[policy_id]]
        path = tmp_path / record["actor_path"]
        try:
            digest = _hash(path)
        except FileNotFoundError as exc:
            raise RuntimeError("Warehouse dual NN artifact missing") from exc
        if digest != record["actor_sha256"]:
            raise RuntimeError("Warehouse dual NN artifact hash mismatch")
    monkeypatch.setattr(warehouse, "_verify_v5_artifact", fixture_actor_hash_only)
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    return warehouse


def _pong_pair(tmp_path, monkeypatch):
    source = json.loads(pong_policy.MODEL_PATH.read_text())
    manifest = {"format": "pong-paired-policies-v5", "variants": {}}
    for index, variant in enumerate(("raw", "regularized")):
        payload = deepcopy(source)
        for role, offset in (("human", 0.03), ("ai", 0.05)):
            payload["actors"][role][-1]["bias"][index + 1] += offset
        payload["actors_sha256"] = sha256(json.dumps(
            payload["actors"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        actor = tmp_path / f"pong-{variant}.json"
        actor.write_text(json.dumps(payload))
        report = tmp_path / f"pong-{variant}.training.json"
        report.write_text("{}")
        paths, program_hashes = [], {}
        if variant == "regularized":
            for role in ("human", "ai"):
                program = tmp_path / f"pong-{role}.program.json"
                program.write_text(json.dumps({"role": role, "fixture": True}))
                paths.append(program.name)
                program_hashes[program.name] = _hash(program)
        manifest["variants"][variant] = {
            "policy_id": f"pong-{variant}-v5", "actor_path": actor.name,
            "actor_sha256": _hash(actor),
            "role_sha256": {role: pong_policy._role_digest(payload["actors"][role])
                            for role in ("human", "ai")},
            "program_paths": paths, "program_sha256": program_hashes,
            "training_report_path": report.name, "training_report_sha256": _hash(report),
            "acceptance_path": f"pong-{variant}.acceptance.json"}
        (tmp_path / f"pong-{variant}.acceptance.json").write_text(json.dumps({
            "release": pong_policy.V5_RELEASE_ID,
            "passed": True, "heldout_evaluated": True,
            "artifact_sha256": {variant: _hash(actor)},
            "training_report_sha256": {variant: _hash(report)},
            "program_sha256": {"regularized": {
                role: program_hashes[path]
                for role, path in zip(("human", "ai"), paths)}},
        }))
    manifest_path = tmp_path / "pong.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(pong_policy, "ROOT", tmp_path)
    monkeypatch.setattr(pong_policy, "V5_MANIFEST_PATH", manifest_path)
    pong_policy._cached_bound_payload.cache_clear()
    return pong


def _kitchen_pair(tmp_path, monkeypatch):
    observation_dim = len(kitchen_policy.features(kitchen_physics.initial_state(1000, 1)))
    hidden = 16
    metadata = {"model_version": kitchen_policy.MODEL_VERSION,
                "feature_version": kitchen_policy.FEATURE_VERSION,
                "observation_dim": observation_dim, "hidden": hidden,
                "human_actions": list(kitchen_policy.HUMAN_ACTIONS),
                "ai_actions": list(kitchen_policy.AI_ACTIONS)}
    entries = {}
    for index, variant in enumerate(("raw", "regularized")):
        rng = np.random.default_rng(5500 + index)
        weights = {"metadata_json": np.asarray(json.dumps(metadata))}
        for role, actions in (("human", kitchen_policy.HUMAN_ACTIONS),
                              ("ai", kitchen_policy.AI_ACTIONS)):
            dimensions = (observation_dim, hidden, hidden, len(actions))
            for layer in range(3):
                weights[f"{role}.weight{layer}"] = rng.normal(
                    0, .02, size=(dimensions[layer + 1], dimensions[layer])).astype(np.float32)
                weights[f"{role}.bias{layer}"] = rng.normal(
                    0, .02, size=dimensions[layer + 1]).astype(np.float32)
        actor_path = tmp_path / f"kitchen-{variant}.npz"
        np.savez(actor_path, **weights)
        frozen = kitchen_policy.FrozenActors(actor_path)
        report = tmp_path / f"kitchen-{variant}.training.json"
        report.write_text("{}")
        paths, program_hashes = [], {}
        if variant == "regularized":
            for role in ("human", "ai"):
                program = tmp_path / f"kitchen-{role}.program.json"
                program.write_text(json.dumps({"role": role, "fixture": True}))
                paths.append(program.name)
                program_hashes[role] = _hash(program)
        entries[variant] = {
            "policy_id": f"kitchen-{variant}-v5", "actor_path": actor_path.name,
            "actor_sha256": frozen.sha256, "role_sha256": frozen.role_sha256,
            "program_paths": paths, "program_sha256": program_hashes,
            "training_report_path": report.name,
            "training_report_sha256": _hash(report),
            "acceptance_path": f"kitchen-{variant}.acceptance.json"}
        (tmp_path / f"kitchen-{variant}.acceptance.json").write_text(json.dumps({
            "release_id": kitchen.V5_RELEASE_ID,
            "passed": True, "heldout_evaluated": True,
            "artifact_sha256": {variant: frozen.sha256},
            "training_report_sha256": {variant: _hash(report)},
            "program_sha256": {"regularized": program_hashes},
        }))
    monkeypatch.setattr(kitchen, "ROOT", tmp_path)
    monkeypatch.setattr(kitchen, "policy_artifacts", lambda: entries)
    kitchen_policy.load_frozen.cache_clear()
    return kitchen


@pytest.fixture(params=("warehouse", "pong", "kitchen"))
def bound_domain(request, tmp_path, monkeypatch):
    """Install two distinct, internally valid Actor pairs under temporary paths."""
    torch.set_num_threads(1)
    monkeypatch.setattr(policies, "ROOT", tmp_path)
    constructors = {"warehouse": _warehouse_pair, "pong": _pong_pair,
                    "kitchen": _kitchen_pair}
    domain = request.param
    eng = constructors[domain](tmp_path, monkeypatch)
    # These temporary preview Actors deliberately lack release acceptance.
    # Exercise their rollouts directly; the registry admission/cache contract
    # is covered separately with accepted and revoked bindings.
    def preview_demonstration(requested_domain, policy_id):
        return registry.engine(requested_domain).demonstration(policy_id)
    monkeypatch.setattr(registry, "demonstration", preview_demonstration)
    monkeypatch.setattr(store_module, "demonstration", preview_demonstration)
    registry._accepted_demonstration.cache_clear()
    yield domain, eng, tmp_path
    registry._accepted_demonstration.cache_clear()
    warehouse._actor.cache_clear()
    warehouse.demonstration.cache_clear()
    pong_policy._cached_bound_payload.cache_clear()
    kitchen_policy.load_frozen.cache_clear()


def _independent_choice(domain, eng, state, role):
    if domain == "warehouse":
        selected, _ = eng._actor(state["policy_id"]).act(
            eng._restore(state).state, base_seed=state["scenario_seed"])
        return selected["robot_1" if role == "human" else "robot_2"].lower()
    if domain == "pong":
        return pong_policy.choose(state, role, state["policy_id"])
    return kitchen._actor(state["policy_id"]).choose(state, role)[0]


def _assert_decision_matches_actor(domain, eng, state, decision):
    expected = _independent_choice(domain, eng, state, "ai")
    assert decision["action_id" if domain == "kitchen" else "action"] == expected
    assert decision.get("policy_id") == state["policy_id"]
    assert decision.get("actor_sha256", decision.get("model_sha256")) == (
        state.get("actor_sha256", state.get("policy_sha256")))
    if domain == "warehouse":
        assert decision["controller_trace"]["selection_changed_policy"] is False
        assert decision["controller_trace"]["selected_actions"]["robot_2"].lower() == expected
    if domain == "pong":
        assert decision["human_action"] == _independent_choice(domain, eng, state, "human")
    if domain == "kitchen":
        assert decision["legal_mask"][kitchen_policy.AI_ACTIONS.index(expected)]


def _different_human_action(domain, eng, state):
    neural = _independent_choice(domain, eng, state, "human")
    return next(action for action in eng.legal_actions(state) if action != neural)


def _command(study, token, view, kind, **payload):
    return study.command(token, kind, {"instance_id": view["instance_id"],
                  "revision": view["revision"], "command_id": uuid4().hex, **payload})


def test_every_bound_frame_uses_the_same_masked_nn_and_human_replaces_one_role(bound_domain):
    domain, eng, tmp_path = bound_domain
    a = policies.binding_for_group(domain, "A")
    b = policies.binding_for_group(domain, "B")
    c = policies.binding_for_group(domain, "C")
    assert a == b
    assert a["actor_sha256"] != c["actor_sha256"]
    assert a["human_actor_sha256"] != c["human_actor_sha256"]
    assert a["ai_actor_sha256"] != c["ai_actor_sha256"]
    assert a["program_hashes"] and not c["program_hashes"]
    assert policies.policy_manifest()[domain]["raw"]["accepted"] is False
    assert policies.policy_manifest()[domain]["regularized"]["accepted"] is False

    database = tmp_path / "study.sqlite3"
    study = Store(Settings(database=str(database)))
    try:
        token, view = study.create({"consent": True, "domain": domain, "mode": "test",
                                   "participant_id": f"v5-{domain}-frames", "group": "A"}, admin=True)
        assert view["policy_id"] == a["policy_id"]
        assert view["actor_sha256"] == a["actor_sha256"]
        assert view["demo"]["frames"]
        view = _command(study, token, view, "demo_skip")
        if domain == "kitchen":
            assert view["stage"] == "practice"
            view = _command(study, token, view, "practice_skip")
        assert view["stage"] == "task1"
        run_id = view["run_id"]
        # A forged but internally consistent turn-zero frame must not replace
        # the seeded initial state to which this instance was assigned.
        with study.db.transaction() as db:
            recorded_run = db.one("SELECT seed,state_json FROM pl3_runs WHERE id=?", (run_id,))
            original_run = recorded_run["state_json"]
            original_frame = db.one("SELECT * FROM pl3_frames WHERE run_id=? AND turn=0", (run_id,))
            forged_start = eng.initial_state(recorded_run["seed"] + 1, 1, a["policy_id"])
            assert forged_start != json.loads(original_frame["state_json"])
            db.execute("UPDATE pl3_runs SET state_json=? WHERE id=?",
                       (json.dumps(forged_start), run_id))
            db.execute("UPDATE pl3_frames SET state_json=?,public_json=?,decision_json=? "
                       "WHERE run_id=? AND turn=0",
                       (json.dumps(forged_start), json.dumps(eng.public_state(forged_start)),
                        json.dumps(eng.decide(forged_start)), run_id))
        with pytest.raises(StudyError, match="recorded_initial_mismatch"):
            study.view(token, view["instance_id"])
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_runs SET state_json=? WHERE id=?", (original_run, run_id))
            db.execute("UPDATE pl3_frames SET state_json=?,public_json=?,decision_json=? "
                       "WHERE run_id=? AND turn=0",
                       (original_frame["state_json"], original_frame["public_json"],
                        original_frame["decision_json"], run_id))
        for turn in range(3):
            with study.db.transaction(read_only=True) as db:
                row = db.one("SELECT * FROM pl3_frames WHERE run_id=? AND turn=?", (run_id, turn))
            state = json.loads(row["state_json"])
            decision = json.loads(row["decision_json"])
            _assert_decision_matches_actor(domain, eng, state, decision)
            assert study.frame(token, view["instance_id"], run_id, turn)["state"] == eng.public_state(state)
            human_action = _different_human_action(domain, eng, state)
            expected = eng.step(deepcopy(state), human_action, decision)
            view = _command(study, token, view, "action", run_id=run_id,
                            turn=turn, action=human_action)
            with study.db.transaction(read_only=True) as db:
                previous = db.one("SELECT * FROM pl3_frames WHERE run_id=? AND turn=?", (run_id, turn))
                next_row = db.one("SELECT * FROM pl3_frames WHERE run_id=? AND turn=?", (run_id, turn + 1))
            assert previous["human_action"] == human_action
            assert json.loads(next_row["state_json"]) == expected
            assert view["state"] == eng.public_state(expected)
            if domain == "warehouse":
                assert expected["last_actions"] == {"human": human_action,
                                                     "ai": decision["action"]}
            assert study.frame(token, view["instance_id"], run_id, turn + 1)["state"] == view["state"]

        restored = Store(Settings(database=str(database)))
        try:
            recovered = restored.recover_view(token, domain)
            assert recovered["instance_id"] == view["instance_id"]
            assert recovered["actor_sha256"] == a["actor_sha256"]
            assert recovered["ai_actor_sha256"] == a["ai_actor_sha256"]
            assert recovered["state"] == view["state"]
            assert restored.frame(token, view["instance_id"], run_id, 3)["state"] == view["state"]
        finally:
            restored.db.close()

        with study.db.transaction() as db:
            row = db.one("SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=1", (run_id,))
            saved_decision = row["decision_json"]
            forged = json.loads(saved_decision)
            forged["action"] = "forged-action"
            db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=1",
                       (json.dumps(forged), run_id))
        with pytest.raises(StudyError, match="recorded_decision_mismatch"):
            study.frame(token, view["instance_id"], run_id, 1)
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=1",
                       (saved_decision, run_id))
            original_start_decision = db.one(
                "SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=0",
                (run_id,))["decision_json"]
            forged_start_decision = json.loads(original_start_decision)
            forged_start_decision["action"] = "forged-action"
            db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=0",
                       (json.dumps(forged_start_decision), run_id))
        with pytest.raises(StudyError, match="recorded_decision_mismatch"):
            study.frame(token, view["instance_id"], run_id, 3)
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=0",
                       (original_start_decision, run_id))
            row = db.one("SELECT human_action FROM pl3_frames WHERE run_id=? AND turn=2", (run_id,))
            saved_human = row["human_action"]
            db.execute("UPDATE pl3_frames SET human_action=? WHERE run_id=? AND turn=2",
                       ("forged-action", run_id))
        with pytest.raises(StudyError, match="recorded_transition_mismatch"):
            study.frame(token, view["instance_id"], run_id, 3)
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_frames SET human_action=? WHERE run_id=? AND turn=2",
                       (saved_human, run_id))

        # Prior releases are historical evidence. They can be read and exported,
        # but can never resume control through the current frozen policy.
        old_release = "policylens-three-domain-20260927.v4.0"
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_instances SET release_id=? WHERE id=?",
                       (old_release, view["instance_id"]))
        archived = study.view(token, view["instance_id"])
        assert archived["stage"] == "archived" and archived["actions"] == []
        assert archived["state"] == view["state"]
        assert study.frame(token, view["instance_id"], run_id, 3)["state"] == view["state"]
        assert len(study.export(release_id=old_release)["frames"]) == 4
        with pytest.raises(StudyError, match="release_changed"):
            _command(study, token, view, "action", run_id=run_id, turn=3, action="wait")
        assert study.recover_view(token, domain)["archived_instances"][0]["id"] == view["instance_id"]
        with study.db.transaction() as db:
            db.execute("UPDATE pl3_instances SET release_id=? WHERE id=?",
                       (view["release_id"], view["instance_id"]))

        actor = tmp_path / a["actor_path"]
        actor.write_bytes(actor.read_bytes() + b"tampered")
        with pytest.raises(StudyError, match="policy_unavailable"):
            study.frame(token, view["instance_id"], run_id, 0)
        with pytest.raises(StudyError, match="policy_unavailable"):
            study.recover_view(token, domain)
    finally:
        study.db.close()


def test_qa_counterfactual_keeps_the_bound_first_nn_action(bound_domain):
    domain, eng, _ = bound_domain
    state = eng.initial_state(8341, 2, policy_id=f"{domain}-raw-v5")
    decision = eng.decide(state)
    original = deepcopy(state)
    action = _different_human_action(domain, eng, state)
    result = simulate(eng, state, decision, [action], horizon=1)
    assert result["input_state_hash"] == _digest(original)
    assert result["steps_completed"] == 1
    assert result["trace"][0]["human_action"] == action
    assert result["trace"][0]["ai_action"] == decision["action"]
    assert state == original
    _assert_decision_matches_actor(domain, eng, state, decision)
    explainer = Explainer(Settings(database=":memory:"))
    plan = {"language": "en", "binding": {"task": state["task"], "turn": state["turn"]},
            "premise": "supported", "clarification": None,
            "intents": [{"kind": "counterfactual", "subject": "human",
                         "purpose": "comparison", "temporal_scope": "current",
                         "evidence_ids": [], "actions": [action], "horizon": 1}]}
    explainer._request_plan = lambda payload: (deepcopy(plan), {"reported_model": "fixture"})
    answer = explainer.answer(eng, state, decision, "What if I chose this action?",
                              public_history=[eng.public_state(state)])
    assert answer["status"] == "answered"
    assert answer["audit"]["simulations"][0]["trace"][0]["ai_action"] == decision["action"]
    assert state == original


def test_bound_demonstration_steps_use_both_frozen_actors(bound_domain, monkeypatch):
    domain, eng, _ = bound_domain
    ident = f"{domain}-raw-v5"
    checked = []
    if domain == "warehouse":
        original_step = WarehouseMultiAgentEnv.step

        def observe(self, actions, *args, **kwargs):
            expected, _ = eng._actor(ident).act(self.state, base_seed=eng.V5_CONFIG["demo_seed"])
            assert actions == expected
            checked.append(deepcopy(actions))
            return original_step(self, actions, *args, **kwargs)

        monkeypatch.setattr(WarehouseMultiAgentEnv, "step", observe)
        demo = eng.demonstration(ident)
        assert len(checked) == len(demo["frames"]) - 1
        for action, frame in zip(checked, demo["frames"][1:]):
            assert frame["demo_actions"] == {"human": action["robot_1"].lower(),
                                             "ai": action["robot_2"].lower()}
        assert demo["provenance"]["actor_sha256"] == policies.binding_for_group(domain, "C")["actor_sha256"]
    elif domain == "kitchen":
        original_step = eng.step

        def observe(state, human_action, decision=None):
            assert human_action == _independent_choice(domain, eng, state, "human")
            _assert_decision_matches_actor(domain, eng, state, decision)
            checked.append((human_action, decision["action_id"]))
            return original_step(state, human_action, decision)

        monkeypatch.setattr(eng, "step", observe)
        demo = eng.demonstration(ident)
        assert len(checked) == len(demo["frames"]) - 1
        assert demo["actor_sha256"] == policies.binding_for_group(domain, "C")["actor_sha256"]
    else:
        demo = eng.demonstration(ident)
        for first, second in zip(demo["frames"], demo["frames"][1:]):
            for role in ("human", "ai"):
                expected = pong_policy.choose(first, role, ident)
                assert second[role]["x"] == first[role]["x"] + eng._DELTAS[expected]
                checked.append(expected)
    assert checked
    if domain == "kitchen":
        assert all(frame["ai"]["policy"]["policy_id"] == ident for frame in demo["frames"])
    else:
        assert all(frame["policy_id"] == ident for frame in demo["frames"])


def test_missing_pinned_actor_blocks_bound_state_and_does_not_rebind(bound_domain):
    domain, eng, tmp_path = bound_domain
    bound = policies.binding_for_group(domain, "C")
    state = eng.initial_state(8342, 1, policy_id=bound["policy_id"])
    actor = tmp_path / bound["actor_path"]
    actor.unlink()
    with pytest.raises(policies.PolicyUnavailable, match="policy_artifact_missing"):
        policies.verify_binding(bound)
    with pytest.raises((RuntimeError, OSError)):
        eng.decide(state)


def test_illegal_human_command_and_forged_ai_decision_are_rejected(bound_domain):
    domain, eng, _ = bound_domain
    state = eng.initial_state(8343, 1, policy_id=f"{domain}-raw-v5")
    decision = eng.decide(state)
    with pytest.raises(ValueError):
        eng.step(state, "teleport", decision)
    forged = deepcopy(decision)
    if domain == "kitchen":
        forged["action_id"] = next(action for action in kitchen_policy.AI_ACTIONS
                                   if action != decision["action_id"])
    else:
        forged["action"] = next(action for action in ("wait", "left", "right")
                                if action != decision["action"])
    with pytest.raises(ValueError):
        eng.step(state, "wait", forged)
