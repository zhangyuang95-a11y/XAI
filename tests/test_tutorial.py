from __future__ import annotations

import pytest
import torch

from backend.adapters.warehouse import WarehouseAdapter
from env.warehouse.environment import WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from ui.tutorial import (
    TUTORIAL_SEED,
    _first_transition,
    _validate_continuous_mission,
    validate_tutorial_seed_isolated,
)


def _policy(*, horizon: int) -> MAPPOPolicy:
    return MAPPOPolicy(
        WarehouseConfig(horizon=horizon),
        MAPPOConfig(hidden_dim=16, seed=9),
    )


def test_tutorial_validator_accepts_one_continuous_scored_round() -> None:
    policy = _policy(horizon=1)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=TUTORIAL_SEED)
    rollout = WarehouseAdapter(environment).rollout(
        policy,
        horizon=1,
        deterministic=True,
    )

    _validate_continuous_mission(
        rollout.frames,
        terminal_reason=rollout.terminal_reason,
    )


def test_tutorial_transition_detection_uses_shared_task_ownership() -> None:
    policy = _policy(horizon=4)
    with torch.no_grad():
        final_layer = policy.network.actor[-1]
        final_layer.weight.zero_()
        final_layer.bias.zero_()
        final_layer.bias[-1] = 10.0
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=12)
    state = environment.get_state()
    state.by_id("robot_2").position = state.tasks[0].pickup_position
    environment.set_state(state)
    rollout = WarehouseAdapter(environment).rollout(
        policy,
        horizon=1,
        deterministic=True,
    )

    assert _first_transition(rollout.frames, "pickup") == (0, "robot_2")


def test_tutorial_seed_is_disjoint_from_all_scored_seed_forms() -> None:
    validate_tutorial_seed_isolated(
        TUTORIAL_SEED,
        task1_seeds=(51_000, 52_000, 53_000, 54_000),
        task2_seeds=(51_500, 52_500, 53_500, 54_500),
    )


def test_tutorial_seed_overlap_is_a_hard_error() -> None:
    with pytest.raises(RuntimeError, match="overlaps"):
        validate_tutorial_seed_isolated(
            TUTORIAL_SEED,
            task1_seeds=(TUTORIAL_SEED,),
            task2_seeds=(51_500,),
        )
