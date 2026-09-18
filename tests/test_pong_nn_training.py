from __future__ import annotations

import copy

import numpy as np
import torch

from core.rcpd import RCPD
from core.rcpd_config import OracleOutput, RCPDConfig
from domains.pong.adapters.core_adapter import feature_names, feature_mapping_from_vector, feature_vector
from domains.pong.environment.engine import PongEnvironment
from domains.pong.environment.engine import paddle_covers_cell
from domains.pong.environment.model import BallKind
from domains.pong.policies.rule_demo import RuleDemoController
from domains.pong.config import PongConfig
from domains.pong.study import PongStudySession
from domains.pong.training.networks import PongActorCritic
from domains.pong.training.curriculum import PongCurriculum
from domains.pong.training.browser_compat import browser_style_probabilities, verify_browser_export
from domains.pong.training.evaluation import evaluate_trainer
from domains.pong.training.partners import PartnerManager
from domains.pong.training.reward import RewardLedger
from domains.pong.training.runner import PongPPOTrainer, _selection_is_better, load_config


def test_observation_has_stable_named_vector() -> None:
    env = PongEnvironment()
    names = feature_names(env)
    row = feature_vector(env, "ai")
    assert len(names) == len(row)
    assert feature_mapping_from_vector(names, row)["role.is_ai"] == 1.0
    assert feature_mapping_from_vector(names, feature_vector(env, "player"))["role.is_ai"] == 0.0


def test_ppo_update_changes_actor_and_excludes_external_partner() -> None:
    config = load_config("configs/pong_nn_training.yaml")
    config = copy.deepcopy(config)
    config["environment"]["parallel_environments"] = 2
    config["ppo"].update({"rollout_decisions": 2, "update_epochs": 1, "minibatch_size": 8})
    config["training"].update({"self_play_fraction": 0.0, "rule_partner_fraction": 1.0, "hesitant_partner_fraction": 0.0})
    trainer = PongPPOTrainer(config, device=torch.device("cpu"))
    before = {key: value.detach().clone() for key, value in trainer.model.actor.state_dict().items()}
    batch = trainer.collect(2)
    assert batch["neural"].sum() == 4  # only Robot 2 rows are neural
    trainer.update(batch)
    assert any(not torch.equal(before[key], value) for key, value in trainer.model.actor.state_dict().items())


def test_program_labels_are_current_neural_distribution() -> None:
    env = PongEnvironment(); names = feature_names(env); model = PongActorCritic(len(names))
    states = []
    for index in range(48):
        row = feature_mapping_from_vector(names, feature_vector(env, "ai")); row["self.paddle_x"] = index / 47
        states.append(row)
    def oracle(state):
        with torch.no_grad():
            values = torch.tensor([[state[name] for name in names]], dtype=torch.float32)
            probabilities = torch.softmax(model.actor_logits(values), -1)[0].numpy()
        return OracleOutput(dict(zip(("left", "right", "stay"), map(float, probabilities))))
    rcpd = RCPD(RCPDConfig(extraction_interval=1, minimum_extraction_samples=32, max_depth=3, max_leaf_nodes=8, min_samples_leaf=4))
    assert rcpd.maybe_extract(1, states, oracle, lambda state: state) is not None
    assert rcpd.program is not None


def test_study_requires_explicit_policy_for_frozen_nn() -> None:
    config = PongConfig(control_mode="frozen_nn")
    session = PongStudySession(config=config, nn_policy=lambda _features: {"left": 1.0, "right": 0.0, "stay": 0.0})
    assert session.tick()["frame"]["ai_x"] < 15.28


def test_checkpoint_restores_model_and_environment_state() -> None:
    config = load_config("configs/pong_nn_training.yaml")
    config = copy.deepcopy(config); config["environment"]["parallel_environments"] = 2
    original = PongPPOTrainer(config, device=torch.device("cpu")); original.collect(1)
    saved = original.state_dict(); restored = PongPPOTrainer(config, device=torch.device("cpu")); restored.load_state_dict(saved)
    assert restored.joint_steps == original.joint_steps
    assert restored.envs[0].snapshot() == original.envs[0].snapshot()
    for key, value in original.model.state_dict().items(): assert torch.equal(value, restored.model.state_dict()[key])


def v2_config() -> dict:
    config = load_config("configs/pong_nn_training_v2.yaml")
    config = copy.deepcopy(config)
    config["environment"].update({"parallel_environments": 2, "episode_seconds": 0.2})
    config["ppo"].update({"rollout_decisions": 2, "update_epochs": 1, "minibatch_size": 8})
    config["training"]["evaluation_episodes"] = 1
    config["evaluation"]["validation_seeds"] = [6101]
    config["evaluation"]["final_test_seeds"] = [9101]
    return config


def test_v2_partner_type_is_episode_stable_and_balances_external_roles() -> None:
    config = v2_config()
    settings = dict(config["training"], self_play_fraction=0.0, rule_partner_fraction=1.0, perturb_partner_fraction=0.0)
    partners = PartnerManager(PongConfig(), settings, 4, seed=3)
    first = [plan.kind for plan in partners.plans]
    assert first == ["rule"] * 4
    assert [plan.external_agent for plan in partners.plans] == [0, 1, 0, 1]
    environment = PongEnvironment()
    for _ in range(4):
        partners.action_pair(environment, 0, 0, 1)
        assert partners.plans[0].kind == "rule"
    assert partners.joint_steps["rule"] == 4


def test_rule_partner_uses_its_own_robot_position_and_restores_commitment() -> None:
    environment = PongEnvironment(PongConfig(duration_seconds=3.0), seed=4)
    for ball in environment.balls:
        ball.active = ball.ball_id == "A1"
    target = next(ball for ball in environment.balls if ball.ball_id == "A1")
    target.x, target.y, target.vx, target.vy = 2.0, 8.0, 0.0, 2.0
    environment.player_x, environment.ai_x = 7.0, 18.0
    player_controller = RuleDemoController(environment.config, controlled_agent="player")
    ai_controller = RuleDemoController(environment.config, controlled_agent="ai")
    assert player_controller.choose(environment.frame()).action == "left"
    assert ai_controller.choose(environment.frame()).action == "stay"

    large = next(ball for ball in environment.balls if ball.ball_id == "B1")
    large.active, target.active = True, False
    large.x, large.y, large.vx, large.vy = 10.0, 8.0, 0.0, 2.0
    environment.player_x, environment.ai_x = 7.0, 11.0
    committed = RuleDemoController(environment.config)
    assert committed.choose(environment.frame()).target_ball_id == "B1"
    restored = RuleDemoController(environment.config)
    restored.restore_state(committed.state_dict())
    assert restored.choose(environment.frame()).target_ball_id == "B1"


def test_curriculum_keeps_slots_but_inactive_balls_are_inert_and_explicit() -> None:
    environment = PongEnvironment(seed=4)
    environment.reset_training_episode(seed=5, scenario_name="single_small", active_ball_ids={"A1"})
    assert [ball.active for ball in environment.balls] == [True, False, False, False, False]
    features = feature_mapping_from_vector(feature_names(environment), feature_vector(environment, "ai"))
    assert features["ball.A1.active"] == 1.0
    assert features["ball.B1.active"] == 0.0
    before = environment.missed_balls
    for _ in range(240):
        environment.step("stay", "stay")
    assert environment.missed_balls >= before
    assert all(not ball.active or ball.ball_id == "A1" for ball in environment.balls)


def test_reward_timing_is_single_source_and_does_not_double_count() -> None:
    config = PongConfig(duration_seconds=4.0)
    environment = PongEnvironment(config, seed=4)
    for ball in environment.balls:
        ball.active = ball.ball_id == "B1"
    large = next(ball for ball in environment.balls if ball.ball_id == "B1")
    large.x, large.y, large.vx, large.vy = 10.0, config.paddle_y - large.height_cells - .01, 0.0, 2.0
    environment.player_x, environment.ai_x = 0.0, 18.0
    score_ledger = RewardLedger("score_at_bottom")
    early_ledger = RewardLedger("failure_at_contact")
    score_total = early_total = 0.0
    for _ in range(240):
        transition = environment.step("stay", "stay")
        score_total += score_ledger.reward(environment, transition)[0]
        early_total += early_ledger.reward(environment, transition)[0]
    assert score_total == early_total == -3.0
    assert len(score_ledger.entries) == len(early_ledger.entries) == 1


def test_v2_checkpoint_restores_curriculum_partner_and_reward_state() -> None:
    config = v2_config()
    original = PongPPOTrainer(config, device=torch.device("cpu"))
    original.collect(2)
    payload = original.state_dict()
    restored = PongPPOTrainer(config, device=torch.device("cpu"))
    restored.load_state_dict(payload)
    assert restored.partners.state_dict() == original.partners.state_dict()
    assert restored.curriculum.state_dict() == original.curriculum.state_dict()
    assert [item.state_dict() for item in restored.reward_ledgers] == [item.state_dict() for item in original.reward_ledgers]
    assert restored.episode_metadata == original.episode_metadata


def test_completed_trajectory_keeps_the_pre_reset_episode_identity() -> None:
    trainer = PongPPOTrainer(v2_config(), device=torch.device("cpu"))
    trainer.collect(2)  # 2 × 6 frames is the 0.2-second test episode
    assert not any(trainer.active_trajectory_observations)
    assert {item["episode_id"] for item in trainer.completed_trajectories} == {0, 1}
    assert {item["stage"] for item in trainer.completed_trajectories} == {"foundation"}
    assert all(item["observations"].shape[0] == 4 for item in trainer.completed_trajectories)
    assert trainer._maybe_extract()["feedback_gate"] == "not_in_full_curriculum_stage"


def test_terminal_gae_does_not_bootstrap_from_the_next_episode() -> None:
    trainer = PongPPOTrainer(v2_config(), device=torch.device("cpu"))
    for parameter in trainer.model.critic.parameters():
        parameter.data.zero_()
    batch = {
        "rewards": np.asarray([[1.0] * 4, [100.0] * 4], dtype=np.float32),
        "values": np.zeros((2, 4), dtype=np.float32),
        "dones": np.asarray([[True] * 4, [False] * 4]),
    }
    advantages, _ = trainer._advantages(batch)
    assert np.allclose(advantages[0], [1.0] * 4)


def test_each_ppo_epoch_iterates_one_complete_neural_permutation(monkeypatch) -> None:
    config = v2_config()
    config["training"].update({"self_play_fraction": 1.0, "rule_partner_fraction": 0.0, "perturb_partner_fraction": 0.0})
    config["ppo"].update({"update_epochs": 2, "minibatch_size": 3})
    trainer = PongPPOTrainer(config, device=torch.device("cpu"))
    batch = trainer.collect(2)
    seen: list[np.ndarray] = []
    original = np.random.permutation
    monkeypatch.setattr(np.random, "permutation", lambda values: (seen.append(np.asarray(values).copy()) or original(values)))
    trainer.update(batch)
    expected = np.flatnonzero(np.asarray(batch["neural"]).reshape(-1))
    assert len(seen) == 2
    assert all(np.array_equal(np.sort(order), expected) for order in seen)


def test_browser_float32_export_math_matches_python_actor() -> None:
    environment = PongEnvironment(seed=9)
    names = feature_names(environment)
    actor = PongActorCritic(len(names)).actor
    arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in actor.state_dict().items()}
    exported = {
        "signature": {"feature_names": list(names)},
        "layers": [
            {"weight": arrays["0.weight"].tolist(), "bias": arrays["0.bias"].tolist(), "activation": "tanh"},
            {"weight": arrays["2.weight"].tolist(), "bias": arrays["2.bias"].tolist(), "activation": "tanh"},
            {"weight": arrays["4.weight"].tolist(), "bias": arrays["4.bias"].tolist(), "activation": "linear"},
        ],
    }
    row = feature_vector(environment, "ai")
    features = feature_mapping_from_vector(names, row)
    with torch.no_grad():
        python = torch.softmax(actor(torch.as_tensor(row[None, :], dtype=torch.float32)), -1)[0].numpy()
    browser = browser_style_probabilities(exported, features)
    result = verify_browser_export(exported, features, python)
    assert np.allclose(browser, python, atol=1e-5)
    assert result["deterministic_action_matches"] is True


def test_selection_prefers_lower_misses_then_large_success_then_earlier() -> None:
    best = {"selection": {"mean_weighted_misses": 10.0, "mean_large_success_rate": .4}}
    assert _selection_is_better({"selection": {"mean_weighted_misses": 9.0, "mean_large_success_rate": .1}}, best)
    assert _selection_is_better({"selection": {"mean_weighted_misses": 10.0, "mean_large_success_rate": .5}}, best)
    assert not _selection_is_better({"selection": {"mean_weighted_misses": 10.0, "mean_large_success_rate": .4}}, best)


def test_fixed_seed_evaluation_reports_all_conditions_and_weighted_formula() -> None:
    trainer = PongPPOTrainer(v2_config(), device=torch.device("cpu"))
    progress: list[dict] = []
    report = evaluate_trainer(trainer, split="validation", partner="all", progress=progress.append)
    assert set(report["conditions"]) == {"nn_nn", "nn_rule", "nn_perturb", "idle_idle", "rule_rule"}
    assert progress[0]["event"] == "evaluation_started"
    assert progress[-1]["event"] == "evaluation_complete"
    for condition in report["conditions"].values():
        for episode in condition["episodes_detail"]:
            assert episode["weighted_misses"] == episode["small_misses"] + 3 * episode["large_misses"]
