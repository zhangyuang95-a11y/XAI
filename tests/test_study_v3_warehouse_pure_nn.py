"""Control-source and version regressions for the new Warehouse release."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from backend.training.warehouse_pure_nn import PureNN, collect_episode
from domains.warehouse import neural_turnbased as warehouse
from env.warehouse.pure_nn import FrozenPureNN, action_masks, observations
from study_v3 import RELEASE_ID
from study_v3.qa import simulate
from study_v3.store import compatible_instance


def test_actor_uses_only_raw_public_fields_and_static_physical_mask():
    state = warehouse.initial_state(100, 2)
    env = warehouse._restore(state)
    original = observations(env.state)
    altered = deepcopy(env.state)
    altered.active_coordination_plan = {"joint_actions": {"robot_2": "WAIT"}}
    altered.agents[1].navigation_goal_position = (0, 0)
    altered.agents[1].navigation_goal_kind = "charge"
    altered.agents[1].route_commitment_task_id = "a-fabricated-task"
    np.testing.assert_array_equal(observations(altered), original)
    np.testing.assert_array_equal(action_masks(altered),
                                  np.asarray([env.action_masks()[agent_id]
                                              for agent_id in ("robot_1", "robot_2")]))


def test_frozen_numpy_logits_match_trained_actor():
    checkpoint = torch.load(warehouse.ROOT / "output/deployment/warehouse_pure_nn_v1.pt",
                            map_location="cpu", weights_only=False)
    model = PureNN(hidden=checkpoint["state_dict"]["actor.0.bias"].numel())
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    state = warehouse.initial_state(100, 1)
    features = observations(warehouse._restore(state).state)
    with torch.no_grad():
        expected = model.actor(torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(warehouse._actor().logits(features), expected,
                               rtol=1e-5, atol=1e-5)
    assert checkpoint["proxy_human_overrides"] == 0
    assert checkpoint["post_policy_interventions"] == 0
    assert checkpoint["selected_episode"] > 0


def test_missing_or_mismatched_neural_artifact_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="artifact missing"):
        FrozenPureNN.load(tmp_path / "absent.npz")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        FrozenPureNN.load(warehouse.ROOT / warehouse.CONFIG["actor_path"],
                          expected_sha256="0" * 64)
    with np.load(warehouse.ROOT / warehouse.CONFIG["actor_path"], allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(payload["metadata_json"].item())
    metadata["model_version"] = "historical-v68"
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    wrong_version = tmp_path / "wrong_version.npz"
    np.savez_compressed(wrong_version, **payload)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        FrozenPureNN.load(wrong_version)


def test_human_task_and_counterfactual_keep_the_exact_frozen_ai():
    state = warehouse.initial_state(100, 2)
    decision = warehouse.decide(state)
    before = deepcopy(state)
    env = warehouse._restore(state)
    actions, _ = warehouse._actor().act(env.state, base_seed=state["scenario_seed"])
    assert decision["action"] == actions["robot_2"].lower()
    assert decision["controller_trace"]["selection_changed_policy"] is False
    result = warehouse.step(state, "wait", decision)
    assert result["last_actions"]["ai"] == actions["robot_2"].lower()
    assert state == before
    branch = simulate(warehouse, state, decision, ["wait"], 1)
    assert branch["steps_completed"] == 1
    assert state == before


@pytest.mark.parametrize("task", (1, 2, 3))
def test_every_human_task_step_restore_replay_and_what_if(task):
    state = warehouse.initial_state(701, task)
    while not state["terminal"]:
        restored = warehouse._restore(json.loads(json.dumps(state)))
        neural, _ = warehouse._actor().act(restored.state, base_seed=state["scenario_seed"])
        decision = warehouse.decide(state)
        assert decision["action"] == neural["robot_2"].lower()
        human_action = warehouse.human_advisor(state)
        branch = simulate(warehouse, state, decision, [human_action], 1)
        after = warehouse.step(state, human_action, decision)
        public = warehouse.public_state(after)
        assert after["last_actions"]["ai"] == neural["robot_2"].lower()
        assert branch["trace"][0]["ai_action"] == after["last_actions"]["ai"]
        assert branch["human"] == public["human"]
        assert branch["ai"] == public["ai"]
        assert branch["raw_score_delta"] == warehouse.score(after)["raw_score"] - warehouse.score(state)["raw_score"]
        assert warehouse.step(json.loads(json.dumps(state)), human_action, decision) == after
        state = after
    assert state["turn"] <= state["max_turns"]


def test_ai_ai_demo_uses_both_nn_outputs():
    demo = warehouse.demonstration()
    assert demo["provenance"]["kind"] == "pure_neural_ai_ai_rollout"
    assert len(demo["frames"]) >= 2
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=demo["provenance"]["seed"])
    for prior, following in zip(demo["frames"], demo["frames"][1:]):
        assert following["turn"] == prior["turn"] + 1
        actions, _ = warehouse._actor().act(env.state, base_seed=demo["provenance"]["seed"])
        assert following["demo_actions"] == {
            "human": actions["robot_1"].lower(), "ai": actions["robot_2"].lower()}
        env.step(actions)
        assert following["human"]["x"] == env.state.by_id("robot_1").position[1]
        assert following["human"]["y"] == env.state.by_id("robot_1").position[0]
        assert following["ai"]["x"] == env.state.by_id("robot_2").position[1]
        assert following["ai"]["y"] == env.state.by_id("robot_2").position[0]
    assert env.state.terminated or env.state.truncated


def test_warehouse_old_releases_are_archived_not_reinterpreted():
    current = {"domain": "warehouse", "release_id": RELEASE_ID}
    old = {"domain": "warehouse", "release_id": "policylens-three-domain-20260922.v3.9"}
    assert compatible_instance(current)
    assert not compatible_instance(old)
    assert warehouse.VERSION != "warehouse-legacy-restored.v3.3"


def test_neural_question_bank_uses_current_state_and_reason():
    from domains.warehouse.build_neural_qa_cases import build
    recorded = json.loads((warehouse.ROOT / "domains/warehouse/neural_qa_cases.json").read_text())
    assert recorded == build()
    assert all(case["state"]["version"] == warehouse.VERSION for case in recorded)
    assert all(case["decision"]["reason_code"] == "neural_choice" for case in recorded)


def test_formal_self_play_never_overrides_either_actor():
    model = PureNN()
    rows, audit = collect_episode(model, seed=997, generator=torch.Generator().manual_seed(997))
    assert rows
    assert audit["actor_decisions"] == 2 * audit["steps"]
    assert audit["proxy_human_overrides"] == 0
    assert audit["post_policy_interventions"] == 0
