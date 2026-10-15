"""Kitchen v5 binds two frozen role Actors and never routes actions by rule."""
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from domains.kitchen import engine as physics
from domains.kitchen import neural_domain as domain
from domains.kitchen import neural_policy as policy
from domains.kitchen import v5_training
from domains.kitchen.training import DualKitchenNetwork
from domains.kitchen.v5_training import (_collect, _extract, _regularizers, _update,
                                         _heldout_already_used, scenario_for_step,
                                         scenario_schedule_digest)


ROOT = Path(__file__).resolve().parents[1]


def test_v5_manifest_exposes_two_pinned_role_pairs():
    entries = domain.policy_artifacts()
    assert set(entries) == {"raw", "regularized"}
    for variant, entry in entries.items():
        assert entry["policy_id"] == f"kitchen-{variant}-v5"
        assert set(entry["role_sha256"]) == {"human", "ai"}
        assert entry["actor_path"].endswith(".npz")
        assert entry["acceptance_path"].endswith(".json")
    assert len(entries["regularized"]["program_paths"]) == 2
    assert entries["raw"]["program_paths"] == []


def test_explicit_v5_binding_checks_both_role_hashes_and_replay(monkeypatch):
    path = ROOT / "output/deployment/kitchen_dual_nn_v1.npz"
    actor = policy.FrozenActors(path)
    assert actor.role_sha256["human"] != actor.role_sha256["ai"]
    deployment = {"actor_path": str(path), "actor_sha256": actor.sha256,
                  "role_sha256": actor.role_sha256}
    monkeypatch.setattr(domain, "_v5_deployment", lambda _: deployment)
    state = domain.initial_state(1000, 1, policy_id="kitchen-regularized-v5")
    decision = domain.decide(state)
    assert decision["policy_id"] == "kitchen-regularized-v5"
    assert decision["model_sha256"] == actor.sha256
    assert decision["role_sha256"] == actor.role_sha256
    assert domain.audited_program_evidence(state) is None
    assert domain.step(state, "wait", decision) == domain.step(deepcopy(state), "wait", decision)
    corrupted = deepcopy(state)
    corrupted["role_sha256"]["human"] = "0" * 64
    with pytest.raises(RuntimeError, match="saved policy binding"):
        domain.decide(corrupted)
    deployment["role_sha256"] = {"human": "0" * 64, "ai": actor.role_sha256["ai"]}
    with pytest.raises(RuntimeError, match="role Actor checksum"):
        domain.initial_state(1000, 1, policy_id="kitchen-regularized-v5")


def test_v5_demonstration_uses_frozen_pair_without_teacher(monkeypatch):
    path = ROOT / "output/deployment/kitchen_dual_nn_v1.npz"
    actor = policy.FrozenActors(path)
    deployment = {"actor_path": str(path), "actor_sha256": actor.sha256,
                  "role_sha256": actor.role_sha256}
    monkeypatch.setattr(domain, "_v5_deployment", lambda _: deployment)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy rule controller selected a demo action")
    for name in ("decide", "simulation_partner", "human_advisor"):
        monkeypatch.setattr(physics, name, forbidden)
    demo = domain.demonstration("kitchen-regularized-v5")
    assert demo["policy_id"] == "kitchen-regularized-v5"
    assert demo["actor_sha256"] == actor.sha256
    assert len(demo["frames"]) > 2
    assert demo["frames"][-1]["ai"]["policy"]["policy_id"] == "kitchen-regularized-v5"


def test_v5_actor_tamper_is_detected_after_cache_warm(tmp_path, monkeypatch):
    path = tmp_path / "kitchen.npz"
    path.write_bytes((ROOT / "output/deployment/kitchen_dual_nn_v1.npz").read_bytes())
    actor = policy.FrozenActors(path)
    monkeypatch.setattr(domain, "_v5_deployment", lambda _: {
        "actor_path": str(path), "actor_sha256": actor.sha256,
        "role_sha256": actor.role_sha256})
    state = domain.initial_state(1000, 1, policy_id="kitchen-regularized-v5")
    assert domain.decide(state)["model_sha256"] == actor.sha256
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        domain.decide(state)


def test_program_evidence_cannot_change_the_neural_action(monkeypatch):
    path = ROOT / "output/deployment/kitchen_dual_nn_v1.npz"
    actor = policy.FrozenActors(path)
    monkeypatch.setattr(domain, "_v5_deployment", lambda _: {
        "actor_path": str(path), "actor_sha256": actor.sha256,
        "role_sha256": actor.role_sha256})
    state = domain.initial_state(1000, 1, policy_id="kitchen-regularized-v5")
    neural_action = domain.decide(state)["action_id"]
    def fake_program(action):
        return SimpleNamespace(execute=lambda _features: SimpleNamespace(
            action=action, trace=SimpleNamespace(tree_steps=[])))
    monkeypatch.setattr(domain, "_audited_ai_program", lambda _: (
        fake_program("wait" if neural_action != "wait" else "left"),
        {"action_fidelity": .91}, "a" * 64))
    assert domain.audited_program_evidence(state) is None
    assert domain.decide(state)["action_id"] == neural_action
    monkeypatch.setattr(domain, "_audited_ai_program", lambda _: (
        fake_program(neural_action), {"action_fidelity": .91}, "a" * 64))
    evidence = domain.audited_program_evidence(state)
    assert evidence["eligible"] and evidence["action_id"] == neural_action
    assert evidence["program_sha256"] == "a" * 64


def test_v5_facts_exclude_legacy_controller_intent_and_synthetic_plan(monkeypatch):
    path = ROOT / "output/deployment/kitchen_dual_nn_v1.npz"
    actor = policy.FrozenActors(path)
    monkeypatch.setattr(domain, "_v5_deployment", lambda _: {
        "actor_path": str(path), "actor_sha256": actor.sha256,
        "role_sha256": actor.role_sha256})
    state = domain.initial_state(1000, 1, policy_id="kitchen-regularized-v5")
    assert any(job["location"] == "plan" for job in physics.public_state(state)["ai"]["current_cooking"])
    assert domain.public_state(state)["ai"]["current_cooking"] == []
    monkeypatch.setattr(physics, "human_advisor", lambda *_: (_ for _ in ()).throw(
        AssertionError("legacy advisor was called while building v5 evidence")))
    suggestion = domain.human_advisor(state)
    assert suggestion == actor.choose(state, "human")[0]
    assert suggestion in domain.legal_actions(state, "human")
    rows = {row["id"]: row for row in domain.facts(state)}
    assert "public_rule10" not in rows
    assert "handoff_recovery" not in rows
    assert "human_available_option" not in rows
    assert rows["ai_current_dishes"]["en"].endswith("none.")
    assert "I cook" not in rows["recipe_sequence_pepper_meat"]["en"]
    assert "environment sequence" in rows["recipe_sequence_pepper_meat"]["en"]
    from study_v3.qa import _catalog
    catalog = _catalog(domain, state, domain.decide(state), [])
    assert catalog["system:human_advice"]["action"] == suggestion
    assert "frozen human-role Actor" in catalog["system:human_advice"]["en"]
    assert all("third still-blocked" not in row["en"] for row in catalog.values())
    forged = dict(domain.decide(state), action_id="wait")
    with pytest.raises(ValueError, match="evidence decision"):
        domain.facts(state, forged)


def test_audit_evidence_uses_both_final_role_hash_bound_development_results(tmp_path, monkeypatch):
    paths = {role: tmp_path / f"{role}.json" for role in ("human", "ai")}
    for role, path in paths.items():
        path.write_text(role)
    report = tmp_path / "training.json"
    report.write_text("{}")
    acceptance_path = tmp_path / "acceptance.json"
    actor_digest = "a" * 64
    role_digests = {"human": "b" * 64, "ai": "c" * 64}
    program_digests = {role: sha256(path.read_bytes()).hexdigest() for role, path in paths.items()}
    audits = {role: {"role": role, "actor_sha256": actor_digest,
                     "role_sha256": role_digests[role], "program_sha256": program_digests[role],
                     "development_seeds": list(range(41_000, 41_020)), "samples": 100,
                     "passed": True, "action_fidelity": .91, "mean_kl_divergence": .20}
              for role in ("human", "ai")}
    acceptance = {"passed": True, "verification_errors": [],
                  "artifact_sha256": {"regularized": actor_digest},
                  "training_report_sha256": {"regularized": sha256(report.read_bytes()).hexdigest()},
                  "program_sha256": {"regularized": program_digests},
                  "final_role_program_development_audit": audits,
                  "final_ai_program_development_audit": audits["ai"]}
    def write_acceptance():
        acceptance_path.write_text(json.dumps(acceptance))
    write_acceptance()
    monkeypatch.setattr(domain, "_v5_deployment", lambda _: {
        "actor_sha256": actor_digest, "role_sha256": role_digests,
        "program_paths": [str(paths[role]) for role in ("human", "ai")],
        "program_sha256": program_digests,
        "training_report_path": str(report),
        "training_report_sha256": sha256(report.read_bytes()).hexdigest(),
        "acceptance_path": str(acceptance_path)})
    monkeypatch.setattr(domain.ExecutableProgram, "load_json", lambda _: SimpleNamespace(
        metadata={"role": "ai", "runtime_control_allowed": False,
                  "program_phase": "final_frozen_actor_training_only_reextraction",
                  "actor_sha256": actor_digest, "role_sha256": role_digests["ai"],
                  "metrics": {"feedback_eligible": True, "explanation_eligible": False,
                              "action_fidelity": .20}}))
    result = domain._audited_ai_program("kitchen-regularized-v5")
    assert result[1]["action_fidelity"] == .91
    acceptance["final_role_program_development_audit"]["human"]["program_sha256"] = "0" * 64
    write_acceptance()
    assert domain._audited_ai_program("kitchen-regularized-v5") is None


def test_v5_joint_collection_and_raw_update_never_use_teacher(monkeypatch):
    torch.manual_seed(23)
    model = DualKitchenNetwork(16)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("teacher supplied a self-play action")
    for name in ("decide", "simulation_partner", "human_advisor"):
        monkeypatch.setattr(physics, name, forbidden)
    history = {role: deque(maxlen=100) for role in ("human", "ai")}
    trajectories, steps, episodes = _collect(model, 23, 0, 1, 9, history)
    assert (steps, episodes) == (9, 1)
    assert trajectories[-1]["bootstrap"] != 0.0
    before = {role: getattr(model, role)[0].weight.detach().clone() for role in ("human", "ai")}
    result = _update(model, trajectories, torch.optim.Adam(model.parameters(), lr=1e-4),
                     None, .86, epochs=1, batch_size=9)
    assert result == {"steps": 9, "feedback_rows": {"human": 0, "ai": 0}}
    assert all(not torch.equal(before[role], getattr(model, role)[0].weight)
               for role in ("human", "ai"))


def test_v5_programs_are_role_specific_and_bounded():
    torch.manual_seed(47)
    model = DualKitchenNetwork(16)
    history = {role: deque(maxlen=800) for role in ("human", "ai")}
    _collect(model, 47, 0, 2, 720, history)
    programs = _regularizers()
    _extract(model, programs, history, 10_000)
    for role in ("human", "ai"):
        assert programs[role].program is not None
        assert programs[role].program.root.depth() <= 5
        assert programs[role].program.root.leaf_count() <= 16
        assert programs[role].program.metadata["role"] == role


def test_v5_global_step_scenario_schedule_survives_different_early_terminals(monkeypatch):
    """An early finish cannot shift one policy onto a different scene seed."""
    torch.manual_seed(48)
    model = DualKitchenNetwork(16)
    original = v5_training._transition
    digests = []
    episode_counts = []
    for terminal_turn in (2, 3):
        def finish_early(state, human, ai):
            next_state = original(state, human, ai)
            if next_state["turn"] >= terminal_turn:
                next_state["terminal"] = True
            return next_state
        monkeypatch.setattr(v5_training, "_transition", finish_early)
        history = {role: deque(maxlen=100) for role in ("human", "ai")}
        actual = sha256()
        _, steps, episodes = _collect(model, 48, 0, 2, 370, history, actual)
        assert steps == 370
        digests.append(actual.hexdigest())
        episode_counts.append(episodes)
    assert episode_counts[0] != episode_counts[1]
    assert digests == [scenario_schedule_digest(48, 370)] * 2
    assert scenario_for_step(48, 359) == (17_000 + 48 * 1_000_000, 1)
    assert scenario_for_step(48, 360) == (17_001 + 48 * 1_000_000, 2)


def test_v5_heldout_receipt_prevents_a_second_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(v5_training, "ROOT", tmp_path)
    config_path = tmp_path / "configs/study_v3_kitchen_v5.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"policies": {"raw": {"acceptance_path": "acceptance.json"}}}))
    assert not _heldout_already_used(tmp_path)
    (tmp_path / "kitchen_dual_nn_v5_heldout_receipt.json").write_text(
        json.dumps({"status": "started"}))
    assert _heldout_already_used(tmp_path)
