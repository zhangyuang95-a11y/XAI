"""Three-task Kitchen execution contracts on the known development fixture 1000.

These bounded synthetic/development traces are engineering tests, not fresh
quality, heldout, admission, or participant evidence.  No heldout scene is run.
Temporary SQLite runs below are deliberately initialized at each task so that
replay/recovery can be exercised without rerunning preceding scientific tasks.
"""
from copy import deepcopy
import json
from uuid import uuid4

import numpy as np
import pytest

from domains.kitchen import neural_domain as kitchen
from domains.kitchen import neural_policy as policy
from study_v3 import RELEASE_ID, policies
from study_v3.config import Settings
from study_v3.qa import _catalog, simulate
from study_v3.store import Store, StudyError, encode


DEVELOPMENT_SEED = 1000  # Existing development fixture; never an acceptance cohort.
TASK_VARIANTS = [(task, variant) for task in (1, 2, 3)
                 for variant in ("raw", "regularized")]


def _binding(variant):
    entry = policies._catalog_entry("kitchen", variant)
    receipt = json.loads((kitchen.ROOT / entry["acceptance_path"]).read_bytes())
    # Check the stored cohort metadata before any fixture environment is made.
    # Kitchen keeps 100 seeds in each of three tasks, for NN and offline proxy.
    for outcome in receipt["heldout"].values():
        for partner in ("nn", "proxy"):
            assert DEVELOPMENT_SEED not in outcome[partner]["seeds"]
    binding = policies.binding_for_group("kitchen", "C" if variant == "raw" else "A")
    assert policies._accepted(entry, binding)
    return binding


def _initial(task, binding):
    return kitchen.initial_state(DEVELOPMENT_SEED, task, binding["policy_id"],
                                 release_id=RELEASE_ID)


def _neural_decision(state, binding):
    """Compute masked argmax independently of the adapter's choose/decide."""
    actor = kitchen._actor(binding["policy_id"], RELEASE_ID)
    logits = actor.logits(state, "ai")
    mask = policy.legal_mask(state, "ai")
    index = int(np.argmax(np.where(mask, logits, -np.inf)))
    expected = policy.AI_ACTIONS[index]
    decision = kitchen.decide(state)
    assert decision["action_id"] == expected
    assert decision["legal_mask"] == mask.tolist()
    assert mask[index]
    assert decision["model_sha256"] == binding["actor_sha256"]
    assert decision["role_sha256"] == actor.role_sha256 == {
        "human": binding["human_actor_sha256"], "ai": binding["ai_actor_sha256"]}
    assert decision["policy_id"] == binding["policy_id"]
    return actor, decision


def _human_takeover(state, actor):
    suggestion = actor.choose(state, "human")[0]
    return next(action for action in kitchen.legal_actions(state)
                if action != suggestion)


def _assert_physical_nn_transition(before, after, decision):
    x, y = before["ai"]["x"], before["ai"]["y"]
    if decision["action"] in kitchen.physics.MOVES:
        dx, dy = kitchen.physics.MOVES[decision["action"]]
        expected = (x+dx, y+dy) if (x+dx, y+dy) in kitchen.physics.FLOOR else (x, y)
    else:
        expected = (x, y)
    assert (after["ai"]["x"], after["ai"]["y"]) == expected


@pytest.mark.parametrize("task,variant", TASK_VARIANTS)
def test_three_tasks_nn_mask_and_human_takeover_match_physics_each_frame(task, variant, monkeypatch):
    binding = _binding(variant)
    state = _initial(task, binding)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a legacy controller selected a formal NN action")

    # The adapter has to pass its NN decision to physical stepping. A fallback
    # to the old controller/advisor is an implementation failure in this test.
    for name in ("decide", "human_advisor", "simulation_partner"):
        if hasattr(kitchen.physics, name):
            monkeypatch.setattr(kitchen.physics, name, forbidden)
    for turn in range(16):
        before = deepcopy(state)
        actor, decision = _neural_decision(state, binding)
        human = actor.choose(state, "human")[0] if turn < 4 else _human_takeover(state, actor)
        expected = kitchen.physics.step(state, human, decision)
        state = kitchen.step(state, human, decision)
        assert state == expected
        assert before["turn"] + 1 == state["turn"]
        assert before["policy_sha256"] == state["policy_sha256"] == binding["actor_sha256"]
        assert state["role_sha256"] == before["role_sha256"]
        _assert_physical_nn_transition(before, state, decision)
    # A synthetic boundary fixture exercises attempted movement into blocked
    # terrain. Kitchen's mask permits movement attempts; physics may block the
    # displacement. This must not be mistaken for a different NN action.
    edge = _initial(task, binding)
    edge["human"].update(x=0, y=0, facing="up")
    actor, decision = _neural_decision(edge, binding)
    assert "up" in kitchen.legal_actions(edge)
    nxt = kitchen.step(edge, "up", decision)
    assert (nxt["human"]["x"], nxt["human"]["y"]) == (0, 0)
    _assert_physical_nn_transition(edge, nxt, decision)


@pytest.mark.parametrize("task,variant", TASK_VARIANTS)
def test_three_tasks_source_and_packaged_physics_keep_identical_nn_traces(task, variant, monkeypatch):
    import study_v3.release_catalog as catalog

    binding = _binding(variant)
    source = kitchen.ROOT / "configs/study_v3_kitchen_frozen_v5.json"
    packaged = catalog.runtime_config("kitchen")
    assert packaged.read_bytes() == source.read_bytes()

    def trace():
        state = _initial(task, binding)
        rows = []
        for _ in range(12):
            actor, decision = _neural_decision(state, binding)
            human = _human_takeover(state, actor)
            rows.append((deepcopy(state), deepcopy(decision), human))
            state = kitchen.step(state, human, decision)
        return rows, state

    with monkeypatch.context() as context:
        context.setattr(catalog, "runtime_config", lambda domain, release_id=RELEASE_ID: source)
        kitchen.physics._configuration.cache_clear()
        source_trace = trace()
    kitchen.physics._configuration.cache_clear()
    try:
        assert trace() == source_trace
    finally:
        kitchen.physics._configuration.cache_clear()


@pytest.mark.parametrize("task,variant", TASK_VARIANTS)
def test_three_tasks_counterfactual_recomputes_nn_and_binds_program_evidence(task, variant):
    binding = _binding(variant)
    state = _initial(task, binding)
    actor, decision = _neural_decision(state, binding)
    before = deepcopy(state)
    actions = [_human_takeover(state, actor), "wait", "left"]
    result = simulate(kitchen, state, decision, actions, horizon=3)
    assert state == before
    branch = deepcopy(state)
    for index, record in enumerate(result["trace"]):
        _, branch_decision = _neural_decision(branch, binding)
        assert record["ai_action"] == branch_decision["action"]
        branch = kitchen.step(branch, actions[index], branch_decision)
        view = kitchen.public_state(branch)
        assert record["human"] == view["human"]
        assert record["ai"] == view["ai"]
    assert result["steps_completed"] == 3
    evidence = kitchen.audited_program_evidence(state)
    catalog = _catalog(kitchen, state, decision, [])
    if variant == "regularized":
        audited = kitchen._audited_ai_program(binding["policy_id"], RELEASE_ID)
        assert audited is not None  # Global role/model audit identity is valid.
        program, _, program_sha = audited
        agrees = program.execute(policy.program_features(state, "ai")).action == decision["action_id"]
        assert (evidence is not None) == agrees
        assert program_sha in binding["program_hashes"].values()
    else:
        assert evidence is None
    if evidence is None:
        assert "system:ai_program" not in catalog
    else:
        assert variant == "regularized"
        assert evidence["eligible"]
        assert evidence["policy_id"] == binding["policy_id"]
        assert evidence["model_sha256"] == binding["actor_sha256"]
        assert evidence["action_id"] == decision["action_id"]
        assert evidence["program_sha256"] in binding["program_hashes"].values()
        assert catalog["system:ai_program"]["program_sha256"] == evidence["program_sha256"]
        assert catalog["system:ai_program"]["policy_id"] == binding["policy_id"]
    forged = dict(decision, model_sha256="0"*64)
    with pytest.raises(ValueError, match="does not match"):
        simulate(kitchen, state, forged, [actions[0]], horizon=1)
    rebound = deepcopy(state)
    rebound["role_sha256"]["ai"] = "0"*64
    with pytest.raises(RuntimeError, match="binding"):
        simulate(kitchen, rebound, decision, [actions[0]], horizon=1)


@pytest.mark.parametrize("task,variant", TASK_VARIANTS)
def test_three_tasks_saved_frame_replay_recovery_and_corruption_rejection(task, variant, tmp_path, monkeypatch):
    import study_v3.store as store_module

    binding = _binding(variant)
    original_config = store_module.scenario_config
    def development_config(domain):
        config = deepcopy(original_config(domain))
        if domain == "kitchen":
            config["pilot_scenario_seeds"] = [DEVELOPMENT_SEED]
        return config
    monkeypatch.setattr(store_module, "scenario_config", development_config)
    # Enrollment scaffolding is synthetic. AI-AI demo execution is already
    # covered elsewhere; no additional rollout is needed to create a session.
    monkeypatch.setattr(store_module, "demonstration", lambda domain, policy_id: {
        "frames": [], "captions": []})
    database = tmp_path / "study.sqlite3"
    study = Store(Settings(database=str(database)))
    try:
        token, view = study.create({"consent": True, "domain": "kitchen", "mode": "test",
            "group": "C" if variant == "raw" else "A",
            "participant_id": f"v620-kitchen-{task}-{variant}"}, admin=True)
        state = _initial(task, binding)
        _, decision = _neural_decision(state, binding)
        run_id = uuid4().hex
        # Construct only a valid task-specific starting run in an isolated test
        # DB. No stage skipping or fixture mutation touches production data.
        with study.db.transaction() as db:
            db.execute("INSERT INTO pl3_runs VALUES(?,?,?,?,?,?,?,?,?)", (
                run_id, view["instance_id"], task, DEVELOPMENT_SEED, encode(state),
                "active", encode(kitchen.score(state)), 1., None))
            db.execute("INSERT INTO pl3_frames VALUES(?,?,?,?,?,?,?)", (
                run_id, 0, encode(state), encode(kitchen.public_state(state)),
                encode(decision), None, 1.))
            db.execute("UPDATE pl3_instances SET stage=?,current_run=? WHERE id=?", (
                f"task{task}", run_id, view["instance_id"]))
        view = study.view(token, view["instance_id"])
        assert view["state"]["task"] == task
        for turn in range(3):
            with study.db.transaction(read_only=True) as db:
                row = db.one("SELECT state_json,decision_json FROM pl3_frames WHERE run_id=? AND turn=?", (run_id, turn))
            saved = json.loads(row["state_json"])
            actor, expected = _neural_decision(saved, binding)
            assert json.loads(row["decision_json"]) == expected
            assert study.frame(token, view["instance_id"], run_id, turn)["state"] == kitchen.public_state(saved)
            view = study.command(token, "action", {"instance_id": view["instance_id"],
                "revision": view["revision"], "command_id": uuid4().hex,
                "run_id": run_id, "turn": turn, "action": _human_takeover(saved, actor)})
        restored = Store(Settings(database=str(database)))
        try:
            recovered = restored.recover_view(token, "kitchen")
            assert recovered["stage"] == f"task{task}"
            assert recovered["actor_sha256"] == binding["actor_sha256"]
            assert recovered["ai_actor_sha256"] == binding["ai_actor_sha256"]
            assert recovered["human_actor_sha256"] == binding["human_actor_sha256"]
            assert recovered["state"] == view["state"]
            assert restored.frame(token, view["instance_id"], run_id, 3)["state"] == view["state"]
            with restored.db.transaction() as db:
                row = db.one("SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=1", (run_id,))
                forged = json.loads(row["decision_json"])
                forged["model_sha256"] = "0"*64
                db.execute("UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=1", (encode(forged), run_id))
            with pytest.raises(StudyError, match="recorded_decision_mismatch"):
                restored.frame(token, view["instance_id"], run_id, 3)
            with pytest.raises(StudyError, match="recorded_decision_mismatch"):
                restored.recover_view(token, "kitchen")
        finally:
            restored.db.close()
    finally:
        study.db.close()
