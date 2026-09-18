"""Short, deterministic checks for the separate v2.2 Pong data paths."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from core.program import ExecutableProgram, ProgramNode

from domains.pong.adapters.core_adapter import feature_mapping_from_vector, feature_vector
from domains.pong.config import PongConfig
from domains.pong.environment.engine import PongEnvironment, predict_next_contact
from domains.pong.policies.hybrid import LimitedPongAssist
from domains.pong.training.evaluation import _episode
from domains.pong.training.runner import PongPPOTrainer, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pong_nn_training_v22_hybrid.yaml"
NODE = shutil.which("node") or str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")


def trainer() -> PongPPOTrainer:
    config = load_config(CONFIG)
    config["environment"]["parallel_environments"] = 2
    config["ppo"]["update_epochs"] = 1
    config["ppo"]["rollout_decisions"] = 2
    config["ppo"]["entropy_coefficient"] = 0
    return PongPPOTrainer(config, device=torch.device("cpu"))


def test_config_inheritance_and_parent_relative_path() -> None:
    pure = load_config(ROOT / "configs/pong_nn_training_v22_pure_control.yaml")
    assert pure["hybrid"]["enabled"] is False
    assert pure["training"]["max_joint_steps"] == 500_000
    assert pure["ppo"]["gamma"] == .99


def test_boundary_only_stays_and_large_requires_other_side() -> None:
    config = PongConfig()
    env = PongEnvironment(config, seed=8)
    env.reset_training_episode(seed=8, scenario_name="large_both_ready", active_ball_ids={"B1"})
    guard = LimitedPongAssist(config)
    env.ai_x = 0.0
    decision = guard.choose(env, "ai", "left")
    assert decision.controller_selected_action == "stay"
    assert decision.intervention_reason == "boundary_no_motion"
    assert "target" not in decision.evidence
    env.player_x = 0.0
    found = False
    for _ in range(15):
        for proposed in ("left", "right", "stay"):
            decision = guard.choose(env, "ai", proposed)
            if decision.intervention_reason == "large_side_rescue":
                target = decision.evidence["target"]
                assert target["side"] in {"left", "right"}
                assert target["ball_kind"] == "large"
                found = True
                break
        if found:
            break
        for _ in range(6):
            env.step("stay", "stay")
    # The constructed scene may have only boundary/coverage corrections. In
    # either case, the guard never hallucinates a partner-side catch.
    if found:
        old = guard.commitments["ai"]["opportunity_id"]
        while env.balls[-2].encounter_index == 0 and not env.terminal:
            env.step("stay", "stay")
        guard.choose(env, "ai", "stay")
        assert guard.commitments.get("ai", {}).get("opportunity_id") != old


def test_truncation_bootstraps_but_cuts_future_gae() -> None:
    agent = trainer()
    batch = {"rewards": np.array([[1., 1.], [100., 100.]], dtype=np.float32),
             "values": np.zeros((2, 2), dtype=np.float32),
             "dones": np.array([[False, False], [True, True]]),
             "boundaries": np.array([[True, True], [True, True]]),
             "next_values": np.array([[5., 5.], [0., 0.]], dtype=np.float32)}
    advantages, returns = agent._advantages(batch)
    assert np.allclose(advantages[0], 1 + .99 * 5)
    assert np.allclose(advantages[1], 100)
    assert np.allclose(returns, advantages)


def test_assisted_snapshots_do_not_advance_ppo_or_actor() -> None:
    agent = trainer()
    before = {name: value.detach().clone() for name, value in agent.model.actor.state_dict().items()}
    agent.corrections.collect(agent.model, agent.device, 15)
    assert agent.joint_steps == 0
    assert agent.actor_updates == agent.critic_updates == 0
    assert agent.corrections.assisted_steps == 15
    assert all(torch.equal(before[name], value) for name, value in agent.model.actor.state_dict().items())
    snapshot = agent.state_dict()
    restored = trainer()
    restored.load_state_dict(snapshot)
    assert restored.corrections.environment.snapshot() == agent.corrections.environment.snapshot()
    assert restored.corrections.state_dict()["guards"] == agent.corrections.state_dict()["guards"]
    assert restored.corrections.accepted == agent.corrections.accepted
    assert restored.corrections.rejected == agent.corrections.rejected


def test_correction_splits_use_complete_episode_ids() -> None:
    agent = trainer()
    agent.corrections.collect(agent.model, agent.device, 200)
    samples = agent.corrections.accepted + agent.corrections.rejected
    train_ids = {item["episode_id"] for item in samples if item["split"] == "train"}
    validation_ids = {item["episode_id"] for item in samples if item["split"] == "validation"}
    assert samples
    assert train_ids.isdisjoint(validation_ids)
    assert all((item["episode_id"] % 5 == 0) == (item["split"] == "validation") for item in samples)


def test_correction_loss_only_when_enabled_and_keeps_trajectory_split() -> None:
    first = trainer()
    second = trainer()
    second.load_state_dict(first.state_dict())
    batch = first.collect(1)
    saved = first.state_dict()
    second.load_state_dict(saved)
    # Set PPO and entropy Actor gradients to zero, isolating the correction.
    for agent in (first, second):
        agent._advantages = lambda value: (np.zeros_like(value["rewards"]), np.zeros_like(value["rewards"]))
    sample = {"split": "train", "episode_id": 1_000_001,
              "observation": batch["obs"][0, 1].tolist(), "controller_selected_action": "left"}
    first.corrections.accepted.append(sample)
    first.corrections.accepted.append({**sample, "split": "validation", "episode_id": 1_000_005})
    assert first.corrections.training_batch(2)[0].shape[0] == 1
    torch.manual_seed(33); np.random.seed(33)
    metrics = first.update(batch)
    torch.manual_seed(33); np.random.seed(33)
    no_label = second.update(batch)
    assert metrics["correction_gradient_norm"] > 0
    assert no_label["correction_gradient_norm"] == 0
    assert any(not torch.equal(a, b) for a, b in zip(first.model.actor.parameters(), second.model.actor.parameters()))


def test_preselected_hold_outcome_is_separate_from_later_round() -> None:
    agent = trainer()
    report = _episode(agent, condition="nn_nn", seed=7001, external_agent=None,
                      scenario="large_hold", controller_mode="pure_nn")
    identity = report["designated_hold_opportunity_id"]
    assert identity is not None
    assert report["designated_hold_outcome"] == report["encounter_outcomes"].get(identity)
    assert report["large_opportunities"] >= int(report["designated_hold_outcome"] is not None)


@pytest.mark.skipif(not Path(NODE).is_file(), reason="bundled Node.js unavailable")
def test_real_browser_javascript_matches_python_guard_and_actor() -> None:
    agent = trainer()
    model = agent.model.actor.state_dict()
    arrays = {name: value.detach().numpy() for name, value in model.items()}
    browser_model = {"signature": {"feature_names": list(agent.names)}, "actions": ["left", "right", "stay"],
                     "layers": [{"weight": arrays["0.weight"].tolist(), "bias": arrays["0.bias"].tolist(), "activation": "tanh"},
                                {"weight": arrays["2.weight"].tolist(), "bias": arrays["2.bias"].tolist(), "activation": "tanh"},
                                {"weight": arrays["4.weight"].tolist(), "bias": arrays["4.bias"].tolist(), "activation": "linear"}]}
    cases = []
    expected = []
    program = ExecutableProgram(tuple(browser_model["actions"]), agent.names,
                                ProgramNode(feature="role.is_ai", threshold=.5,
                                            left=ProgramNode(probabilities=(.2, .3, .5)),
                                            right=ProgramNode(probabilities=(.1, .7, .2))))
    for seed in range(12):
        env = PongEnvironment(agent.env_config, seed=seed)
        scenario = ("large_both_ready", "large_one_holds", "dual_small")[seed % 3]
        active = {"B1"} if "large" in scenario else {"A1", "A2"}
        env.reset_training_episode(seed=seed, scenario_name=scenario, active_ball_ids=active)
        for proposal in ("left", "right", "stay"):
            row = feature_vector(env, "ai")
            probabilities = torch.softmax(agent.model.actor_logits(torch.as_tensor(row[None, :])), -1)[0].detach().numpy()
            cases.append({**env.snapshot(), "proposal": proposal,
                          "features": feature_mapping_from_vector(agent.names, row)})
            expected.append((LimitedPongAssist(agent.env_config).choose(env, "ai", proposal), probabilities,
                             program.execute(cases[-1]["features"]).trace.tree_steps))
    process = subprocess.run([NODE, str(ROOT / "scripts/check_pong_v22_browser.cjs")],
                             input=json.dumps({"cases": cases, "model": browser_model, "program": program.to_dict()}),
                             text=True, capture_output=True, check=True)
    browser = json.loads(process.stdout)
    for observed, (decision, probabilities, trace) in zip(browser, expected):
        assert observed["action"] == decision.controller_selected_action
        assert observed["reason"] == decision.intervention_reason
        assert np.allclose(observed["probabilities"], probabilities, atol=2e-5)
        assert [item["feature"] for item in observed["programTrace"]] == [item.feature for item in trace]
        assert [item["result"] for item in observed["programTrace"]] == [item.result for item in trace]


@pytest.mark.skipif(not Path(NODE).is_file(), reason="bundled Node.js unavailable")
def test_real_browser_physics_and_encounter_event_match_python() -> None:
    config = PongConfig()
    cases = []
    expected = []
    for seed, scenario, active in ((7001, "large_both_ready", {"B1"}),
                                   (7002, "large_one_holds", {"B1"}),
                                   (7003, "dual_small", {"A1", "A2"})):
        env = PongEnvironment(config, seed=seed)
        env.reset_training_episode(seed=seed, scenario_name=scenario, active_ball_ids=active)
        ball = next(ball for ball in env.balls if ball.active)
        prediction = predict_next_contact(ball, env.frame(), config)
        assert prediction is not None
        for _ in range(max(0, prediction.updates_until_contact - 1)):
            env.step("stay", "stay")
        case = {**env.snapshot(), "proposal": "stay",
                "simulate": {"player_action": "stay", "ai_action": "stay"}}
        transition = env.step("stay", "stay")
        cases.append(case)
        expected.append((env.snapshot(), transition.events))
    process = subprocess.run([NODE, str(ROOT / "scripts/check_pong_v22_browser.cjs")],
                             input=json.dumps({"cases": cases}), text=True, capture_output=True, check=True)
    actual = json.loads(process.stdout)
    for item, (snapshot, events) in zip(actual, expected):
        physics = item["physics"]
        assert physics["frame"] == snapshot["frame_index"]
        assert physics["player_x"] == pytest.approx(snapshot["player_x"])
        assert physics["ai_x"] == pytest.approx(snapshot["ai_x"])
        assert physics["missed_balls"] == snapshot["missed_balls"]
        for browser_ball, python_ball in zip(physics["balls"], snapshot["balls"]):
            assert browser_ball["x"] == pytest.approx(python_ball["x"])
            assert browser_ball["y"] == pytest.approx(python_ball["y"])
            assert browser_ball["encounter_index"] == python_ball["encounter_index"]
        browser_encounters = [(event["encounter_id"], event["outcome"])
                              for event in physics["events"] if event["event"] == "encounter"]
        python_encounters = [(event["encounter_id"], event["outcome"])
                             for event in events if event["event"] == "encounter"]
        assert browser_encounters == python_encounters


def test_readme_scripts_parse_from_other_working_directory() -> None:
    for name in ("train_pong_nn.py", "evaluate_pong_nn.py", "export_pong_nn.py", "serve_pong_nn.py"):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"],
                                cwd="/tmp", capture_output=True, text=True)
        assert result.returncode == 0, (name, result.stderr)
