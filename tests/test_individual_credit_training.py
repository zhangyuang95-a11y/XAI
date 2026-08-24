from __future__ import annotations

from argparse import Namespace

import pytest

from backend.training.warehouse_options import skill_retention_weight
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.domain import WarehouseConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.rewards import RewardConfig


def test_skill_retention_decays_linearly_to_zero_by_half_training() -> None:
    args = Namespace(
        episodes=100,
        skill_retention_weight=1.0,
        skill_retention_fade_end=0.5,
        skill_retention_fixed=False,
    )
    assert skill_retention_weight(args, 0) == pytest.approx(1.0)
    assert skill_retention_weight(args, 25) == pytest.approx(0.5)
    assert skill_retention_weight(args, 50) == pytest.approx(0.0)
    assert skill_retention_weight(args, 100) == pytest.approx(0.0)


def test_legacy_retention_is_available_only_for_explicit_ablation() -> None:
    args = Namespace(
        episodes=100,
        skill_retention_weight=5.0,
        skill_retention_fade_end=0.5,
        skill_retention_fixed=True,
    )
    assert skill_retention_weight(args, 0) == pytest.approx(5.0)
    assert skill_retention_weight(args, 100) == pytest.approx(5.0)


def test_old_reward_version_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="Unsupported reward version"):
        RewardConfig(version="warehouse_safe_mission_reward_v18_efficiency_penalties")


def test_legacy_team_credit_requires_an_explicit_ablation_flag() -> None:
    legacy = RewardConfig(
        individual_credit_enabled=False,
        progress_scale=2.0,
        coordination_clearance_cost=16.0,
        avoidable_wait_cost=0.01,
        mission_regression_scale=1.0,
    )
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8, reward=legacy))
    environment.reset(seed=7)
    _, rewards, _, _, info = environment.step(
        {"robot_1": "UP", "robot_2": "WAIT"}
    )
    assert rewards["robot_1"] == pytest.approx(rewards["robot_2"])
    assert info["training_reward"] == pytest.approx(rewards["robot_1"])


def test_teacher_prioritizes_the_more_urgent_empty_charger_entry() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=15038)
    while environment.get_state().frame < 90:
        actions = stable_coordination_actions(environment)
        _, _, terminated, truncated, info = environment.step(actions)
        assert not info.get("shutdowns")
        assert not terminated
        assert not truncated
    while True:
        actions = stable_coordination_actions(environment)
        _, _, terminated, truncated, info = environment.step(actions)
        assert not info.get("shutdowns")
        if terminated or truncated:
            break
    assert environment.get_state().total_deliveries >= 1
