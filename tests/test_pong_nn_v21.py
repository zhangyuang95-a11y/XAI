"""Targeted safeguards for the v2.1 Pong continuation protocol."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

from domains.pong.environment.engine import PongEnvironment, paddle_covers_cell, predict_next_contact
from domains.pong.training.evaluation import _summarize
from domains.pong.training.runner import PongPPOTrainer, load_config, run
from domains.pong.training.reward import RewardLedger


def v21_config() -> dict:
    config = copy.deepcopy(load_config("configs/pong_nn_training_v21.yaml"))
    config["environment"].update({"parallel_environments": 2, "episode_seconds": 1.5})
    config["training"].update({"max_joint_steps": 100, "checkpoint_every_joint_steps": 25, "evaluation_episodes": 1})
    config["ppo"].update({"rollout_decisions": 1, "update_epochs": 1, "minibatch_size": 4})
    config["curriculum"].update({"foundation_until_joint_steps": 50, "coordination_until_joint_steps": 75})
    config["evaluation"].update({"validation_seeds": [5001], "final_test_seeds": [9001], "controlled_hold_seeds": [7001]})
    return config


def test_v21_ready_scenario_has_real_two_paddle_coverage() -> None:
    trainer = PongPPOTrainer(v21_config(), device=torch.device("cpu"))
    environment = PongEnvironment(trainer.env_config, seed=44)
    environment.reset_training_episode(seed=44, scenario_name="large_both_ready", active_ball_ids={"B1"})
    ball = next(item for item in environment.balls if item.active)
    prediction = predict_next_contact(ball, environment.frame(), environment.config)
    assert prediction is not None
    left, right = prediction.contact_cells
    covered = ((paddle_covers_cell(environment.player_x, left, environment.config)
                and paddle_covers_cell(environment.ai_x, right, environment.config)) or
               (paddle_covers_cell(environment.player_x, right, environment.config)
                and paddle_covers_cell(environment.ai_x, left, environment.config)))
    assert covered
    assert .4 <= prediction.time_until_contact <= 1.2


def test_v21_full_scenario_is_the_normal_environment_reset() -> None:
    trainer = PongPPOTrainer(v21_config(), device=torch.device("cpu"))
    environment = PongEnvironment(trainer.env_config, seed=19)
    normal = environment.snapshot()
    environment.reset_training_episode(seed=19, scenario_name="full", active_ball_ids=set(environment.config.ball_ids))
    assert environment.snapshot()["balls"] == normal["balls"]
    assert environment.training_scenario_metadata["source"] == "normal_reset"


def test_v21_success_uses_encounter_outcome_not_bottom_score() -> None:
    summary = _summarize("nn_nn", [{
        "weighted_misses": 0, "small_misses": 0, "large_misses": 0,
        "small_opportunities": 1, "large_opportunities": 1,
        "small_catches": 0, "large_catches": 0,
        "small_success_rate": 0.0, "large_success_rate": 0.0,
        "unscored_confirmed_misses_at_terminal": 2, "diagnostics": {},
        "external_agent": None, "action_chain_valid": True,
    }])
    assert summary["aggregate_small_success_rate"] == 0.0
    assert summary["aggregate_large_success_rate"] == 0.0
    assert summary["mean_weighted_misses"] == 0.0
    assert summary["unscored_confirmed_misses_at_terminal"] == 2


def test_early_reward_links_failure_and_later_score_without_duplicate_penalty() -> None:
    from domains.pong.config import PongConfig
    config = PongConfig(duration_seconds=4.0)
    environment = PongEnvironment(config, seed=6)
    for ball in environment.balls:
        ball.active = ball.ball_id == "A1"
    ball = next(ball for ball in environment.balls if ball.active)
    ball.x, ball.y, ball.vx, ball.vy = 0.0, config.paddle_y - ball.height_cells - .01, 0.0, 2.0
    environment.player_x, environment.ai_x = 10.0, 18.0
    ledger = RewardLedger("failure_at_contact")
    total = 0.0
    for _ in range(240):
        total += ledger.reward(environment, environment.step("stay", "stay"))[0]
    assert total == -1.0
    assert len(ledger.entries) == 1
    assert ledger.entries[0]["failure_frame"] is not None
    assert ledger.entries[0]["score_frame"] is not None


def test_v21_parent_loads_weights_without_old_optimizer_or_counts() -> None:
    source = PongPPOTrainer(v21_config(), device=torch.device("cpu"))
    source.collect(1)
    source.update(source.collect(1))
    payload = source.state_dict()
    child = PongPPOTrainer(v21_config(), device=torch.device("cpu"))
    child.initialize_from_parent(payload, {"checkpoint": "synthetic"})
    assert child.joint_steps == 0
    assert child.actor_updates == 0
    assert child.actor_optimizer.state == {}
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, child.model.state_dict()[name])


def test_v21_final_checkpoint_is_evaluated_and_can_be_selected(tmp_path: Path, monkeypatch) -> None:
    config = v21_config()
    config["environment"]["parallel_environments"] = 2
    config["training"].update({"max_joint_steps": 2, "checkpoint_every_joint_steps": 25})
    parent = PongPPOTrainer(config, device=torch.device("cpu"))
    parent_path = tmp_path / "parent.pt"
    torch.save(parent.state_dict(), parent_path)
    config["initialization"]["parent_candidates"] = [str(parent_path), str(parent_path)]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def report(*_args, **_kwargs):
        return {"selection": {"mean_weighted_misses": 1.0, "mean_large_success_rate": .5,
                                "weighted_miss_reduction_vs_idle": .2},
                "conditions": {"nn_nn": {"mean_weighted_misses": 1.0, "aggregate_large_success_rate": .5},
                               "idle_idle": {"mean_weighted_misses": 2.0}},
                "action_chain_valid": True, "controlled_checks": {}}

    monkeypatch.setattr("domains.pong.training.evaluation.evaluate_trainer", report)
    output = tmp_path / "run"
    result = run(config_path, output, device="cpu")
    assert result["last_evaluation"]["joint_steps"] == 2
    assert (output / "best_candidate.pt").is_file()
    assert (output / "parent_comparison.json").is_file()


def test_v21_feedback_gate_reads_nn_nn_only() -> None:
    config = v21_config(); config["feedback_gate"]["minimum_joint_steps"] = 10
    trainer = PongPPOTrainer(config, device=torch.device("cpu"))
    trainer.joint_steps = 130
    trainer.validation_history = [{
        "stage": "full", "action_chain_valid": True,
        "conditions": {"nn_nn": {"mean_weighted_misses": 12, "aggregate_large_success_rate": .55},
                       "idle_idle": {"mean_weighted_misses": 20}},
        "selection": {"mean_weighted_misses": 1, "mean_large_success_rate": 1},
    }] * 3
    assert trainer._feedback_gate() == (True, "passed")
