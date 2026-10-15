"""Kitchen's participant AI acts through the frozen Actor and explicit physics."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from domains.kitchen import engine as physics
from domains.kitchen import neural_domain as domain
from domains.kitchen import neural_policy as policy
from domains.kitchen.build_neural_qa_cases import build as build_neural_qa_cases
from domains.kitchen.training import DualKitchenNetwork, collect_joint, export, ppo_update
from study_v3.qa import _catalog, simulate
from study_v3.config import Settings
from study_v3.store import Store, StudyError
from study_v3.policies import PolicyUnavailable


def prepared_egg():
    return {"id": "test-egg", "ingredient": "egg", "ingredients": ["egg"],
            "recipe": "egg_tomato", "stage": "prepared", "prepare_progress": 4,
            "components": ["test-egg"], "order_id": None, "pot_id": None,
            "container": None, "was_buffered": False, "acquired_turn": 0,
            "prepared_turn": 0, "freshness_started_turn": 0, "fresh_until": 20,
            "expires_turn": 20, "freshness_basis": "prepared"}


def test_ai_slot_is_explicit_and_first_free_slot_is_not_forced():
    state = physics.initial_state(1000, 1)
    state["ai"].update(x=6, y=3, facing="right", holding=prepared_egg())
    mask = policy.legal_mask(state, "ai")
    assert not mask[policy.AI_ACTIONS.index("interact")]
    assert mask[policy.AI_ACTIONS.index("interact_slot0")]
    assert mask[policy.AI_ACTIONS.index("interact_slot1")]
    after = physics.step(state, "wait", {"action": "interact", "slot": 1,
                                         "policy_origin": "kitchen_nn", "memory": {}})
    assert after["buffers"]["ai_raw"][0] is None
    assert after["buffers"]["ai_raw"][1]["id"] == "test-egg"


def test_actor_observation_excludes_seed_future_orders_and_legacy_plan():
    state = physics.initial_state(1000, 1)
    changed = deepcopy(state)
    changed["seed"] = 999999
    changed["_future_orders"] = [{"id": "hidden", "recipe": "egg_tomato"}]
    changed["policy_memory"] = {"goal": "rescue_pot", "rescue_pot": "pot2"}
    assert np.array_equal(policy.features(state), policy.features(changed))


def test_public_rules_do_not_promise_legacy_rescue_route():
    assert "handoff_grace_turns" not in domain.rule_metadata()
    assert all("two full turns" not in line for line in domain.rules("en"))
    state = physics.initial_state(1000, 1)
    state["version"] = domain.VERSION
    assert "handoff_grace" not in domain.public_state(state)


def test_ai_chooses_duplicate_recipe_order_at_load(monkeypatch):
    state = physics.initial_state(1000, 1)
    for order in state["orders"][:2]:
        order["recipe"] = "egg_tomato"
    state["ai"].update(x=6, y=1, facing="right", holding=prepared_egg())
    mask = policy.legal_mask(state, "ai")
    assert mask[policy.AI_ACTIONS.index("interact_order1")]
    assert mask[policy.AI_ACTIONS.index("interact_order2")]
    monkeypatch.setattr(physics, "_assign", lambda *_: (_ for _ in ()).throw(AssertionError("legacy assignment used")))
    after = physics.step(state, "wait", {"action": "interact", "order_id": "order2",
                                         "policy_origin": "kitchen_nn", "memory": {}})
    assert after["pots"][0]["order_id"] == "order2"
    assert after["pots"][0]["item"]["order_id"] == "order2"


class FakeActor:
    sha256 = "test-checksum"

    def __init__(self, selected):
        self.selected = selected

    def choose(self, state, role):
        assert role == "ai", "formal Kitchen must leave the human role to the participant"
        mask = policy.legal_mask(state, role)
        scores = np.zeros(len(policy.AI_ACTIONS), dtype=np.float32)
        scores[policy.AI_ACTIONS.index(self.selected)] = 10
        return self.selected, scores, mask


def test_formal_execution_uses_actor_choice_and_rejects_override(monkeypatch):
    state = physics.initial_state(1000, 1)
    state["version"] = domain.VERSION
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A hand-written controller selected the formal AI action")
    for name in ("decide", "_plan", "_rescue_after_work", "_holding_plan", "_empty_plan"):
        monkeypatch.setattr(physics, name, forbidden)
    monkeypatch.setattr(domain, "_actor", lambda: FakeActor("up"))
    up = domain.decide(state)
    assert up["action"] == "up" and up["policy_origin"] == "kitchen_nn"
    assert up["model_sha256"] == "test-checksum"
    assert domain.step(state, "wait", up)["ai"]["y"] == 2
    monkeypatch.setattr(domain, "_actor", lambda: FakeActor("left"))
    left = domain.decide(state)
    assert left["action"] == "left"
    assert domain.step(state, "wait", left)["ai"]["x"] == 5
    with pytest.raises(ValueError, match="does not match"):
        domain.step(state, "wait", up)
    assert state == deepcopy(state)


def test_frozen_actor_handoff_conflict_keeps_physical_simultaneity(monkeypatch):
    state = physics.initial_state(1000, 1)
    state["version"] = domain.VERSION
    state["human"].update(x=3, y=3, facing="right")
    state["ai"].update(x=5, y=3, facing="left")
    state["handoff"] = prepared_egg()
    monkeypatch.setattr(domain, "_actor", lambda: FakeActor("interact"))
    before = deepcopy(state["handoff"])
    after = domain.step(state, "interact", domain.decide(state))
    assert after["handoff"] == before
    assert after["human"]["holding"] is None and after["ai"]["holding"] is None
    assert after["metrics"]["handoff_conflicts"] == 1


def test_missing_model_blocks_formal_task(monkeypatch):
    monkeypatch.setattr(domain, "_actor", lambda: (_ for _ in ()).throw(RuntimeError("missing model")))
    with pytest.raises(RuntimeError, match="missing model"):
        domain.initial_state(1000, 1)


def test_mismatched_pinned_artifact_blocks_formal_task(tmp_path, monkeypatch):
    path = tmp_path / "actor.npz"
    export(DualKitchenNetwork(hidden=16), path)
    monkeypatch.setattr(domain, "_deployment", lambda: {"path": str(path), "sha256": "wrong"})
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        domain.initial_state(1000, 1)


def test_counterfactual_reuses_frozen_actor_without_mutating_state(monkeypatch):
    state = physics.initial_state(1000, 1)
    state["version"] = domain.VERSION
    monkeypatch.setattr(domain, "_actor", lambda: FakeActor("wait"))
    original = deepcopy(state)
    result = simulate(domain, state, domain.decide(state), ["wait"], 2)
    assert result["steps_completed"] == 2
    assert [step["ai_action"] for step in result["trace"]] == ["wait", "wait"]
    assert state == original


def test_frozen_actor_decisions_replay_identically_from_saved_state():
    state = domain.initial_state(1000, 2)
    original = deepcopy(state)
    actions = []
    decisions = []
    for _ in range(80):
        human = physics.simulation_partner(state)
        decision = domain.decide(state)
        actions.append(human)
        decisions.append(decision)
        state = domain.step(state, human, decision)
    replay = json.loads(json.dumps(original))
    for human, decision in zip(actions, decisions):
        assert domain.decide(replay) == decision
        replay = domain.step(replay, human, decision)
    assert replay == state


def test_live_qa_reason_is_grounded_in_recorded_nn_selection(monkeypatch):
    state = physics.initial_state(1000, 1)
    state["version"] = domain.VERSION
    monkeypatch.setattr(domain, "_actor", lambda: FakeActor("left"))
    decision = domain.decide(state)
    facts = {row["id"]: row for row in domain.facts(state, decision)}
    assert "frozen Kitchen neural network" in facts["ai_reason"]["en"]
    assert "move or face left" in facts["ai_next_action"]["en"].lower()
    assert "next_input_ingredient" not in facts
    catalog = _catalog(domain, state, decision, [])
    assert catalog["system:ai_reason"]["en"] == facts["ai_reason"]["en"]


@pytest.mark.parametrize("group", ("A", "B", "C"))
def test_formal_store_refuses_unpinned_v5_policy(tmp_path, monkeypatch, group):
    def unavailable(*_args):
        raise PolicyUnavailable("missing_v5_artifact")
    monkeypatch.setattr("study_v3.store.binding_for_group", unavailable)
    store = Store(Settings(database=str(tmp_path / f"kitchen-{group}.sqlite3")))
    with pytest.raises(StudyError, match="policy_unavailable"):
        store.create({"participant_id": f"kitchen-{group.lower()}-unready",
                      "domain": "kitchen", "mode": "test", "group": group,
                      "consent": True}, admin=True)
    with store.db.transaction(read_only=True) as db:
        assert db.one("SELECT COUNT(*) AS n FROM pl3_instances")["n"] == 0
        assert db.one("SELECT COUNT(*) AS n FROM pl3_runs")["n"] == 0
    store.db.close()


def test_legacy_v4_direct_kitchen_state_remains_replayable():
    original = domain.initial_state(1000, 1)
    assert original["version"] == domain.VERSION
    assert "policy_id" not in original
    state = deepcopy(original)
    recorded = []
    for _ in range(20):
        decision = domain.decide(state)
        recorded.append(decision)
        state = domain.step(state, "wait", decision)
    replay = deepcopy(original)
    for decision in recorded:
        assert domain.decide(replay) == decision
        replay = domain.step(replay, "wait", decision)
    assert replay == state


def test_exported_numpy_actor_matches_training_network(tmp_path):
    import torch

    model = DualKitchenNetwork(hidden=16)
    path = tmp_path / "kitchen.npz"
    digest = export(model, path)
    frozen = policy.FrozenActors(path, digest)
    state = physics.initial_state(1000, 1)
    x = torch.as_tensor(policy.features(state)[None, :])
    for role in ("human", "ai"):
        with torch.no_grad():
            expected = getattr(model, role)(x)[0].numpy()
        assert np.allclose(frozen.logits(state, role), expected, atol=1e-5)
    with pytest.raises(RuntimeError, match="checksum"):
        policy.FrozenActors(path, "bad-checksum")


def test_joint_update_trains_both_actors_from_nn_nn_actions(monkeypatch):
    import torch

    torch.manual_seed(41)
    model = DualKitchenNetwork(hidden=16)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A rule controller supplied a joint-training action")
    monkeypatch.setattr(physics, "decide", forbidden)
    monkeypatch.setattr(physics, "simulation_partner", forbidden)
    monkeypatch.setattr(physics, "human_advisor", forbidden)
    trajectories = collect_joint(model, 19000, episodes=1)
    before_h = model.human[0].weight.detach().clone()
    before_a = model.ai[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    assert ppo_update(model, trajectories, optimizer, epochs=1, batch_size=128) > 0
    assert not torch.equal(before_h, model.human[0].weight)
    assert not torch.equal(before_a, model.ai[0].weight)


def test_current_nn_qa_cases_match_frozen_trace_and_release_manifest():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "domains/kitchen/neural_qa_cases.json").read_text())
    assert cases == build_neural_qa_cases()
    assert len(cases) == 12
    assert all(case["state"]["version"] == domain.VERSION for case in cases)
    manifest = json.loads((root / "configs/study_v3_qa_cases.json").read_text())
    assert "domains/kitchen/neural_qa_cases.json" in manifest["external_case_files"]
