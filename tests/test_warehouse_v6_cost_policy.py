"""Shutdown cost uses actual complete episodes and both NN probabilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from backend.training.warehouse_v6_actor import WarehouseV6Actor
from backend.training.warehouse_v6_runner import _ppo_rows, ppo_update
from backend.training.warehouse_v6_shutdown_cost import (
    ShutdownRiskDual, episode_balanced_cost_critic_loss,
    joint_shutdown_surrogate, shutdown_cost_gae,
)
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.pure_nn import action_masks
from env.warehouse.public_observation_v6 import encode_public_observation


def _preaction(frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Real study-map observation and physical masks without reset."""
    state = WarehouseState(
        episode_id=100, frame=frame,
        agents=[AgentState("robot_1", STUDY_MAP_LAYOUT.robot_start_positions[0],
                           70.0),
                AgentState("robot_2", STUDY_MAP_LAYOUT.robot_start_positions[1],
                           70.0)],
        tasks=[DeliveryTask("task_0", (0, 3), (3, 2)),
               DeliveryTask("task_1", (1, 0), (3, 1))], next_task_index=2)
    return encode_public_observation(state), action_masks(state)


def _episode(*, cost: float, complete: bool, baseline: float = 0.2):
    records = []
    for frame in range(2):
        obs, mask = _preaction(frame)
        records.append({
            "obs": obs, "mask": mask,
            "action": np.zeros(2, dtype=np.int64),
            "logp": np.zeros(2, dtype=np.float32),
            "reward": np.zeros(2, dtype=np.float32),
            "reward_value": np.zeros(2, dtype=np.float32),
            "cost_value": baseline,
            "shutdown_cost": cost if frame == 1 else 0.0,
        })
    return {"records": records, "reward_bootstrap": np.zeros(2, dtype=np.float32),
            "cost_bootstrap": 0.0, "complete": complete,
            "shutdown_episode": bool(cost)}


def test_complete_episode_cost_is_undiscounted_and_partial_has_no_label():
    safe, safe_target = shutdown_cost_gae([0, 0], [0.2, 0.1], complete=True)
    failed, failed_target = shutdown_cost_gae([0, 1], [0.2, 0.1], complete=True)
    torch.testing.assert_close(safe, torch.tensor([-0.2, -0.1]))
    torch.testing.assert_close(failed, torch.tensor([0.8, 0.9]))
    torch.testing.assert_close(safe_target, torch.zeros(2))
    torch.testing.assert_close(failed_target, torch.ones(2))
    with pytest.raises(ValueError, match="complete"):
        shutdown_cost_gae([0], [0.2], complete=False, bootstrap=0.3)


def test_mixed_ppo_batch_excludes_truncated_fragment_from_cost_targets():
    data = _ppo_rows([_episode(cost=0, complete=True),
                      _episode(cost=1, complete=True),
                      _episode(cost=0, complete=False)])
    assert data["complete_episode_outcomes"] == {0: False, 1: True}
    torch.testing.assert_close(data["cost_adv"],
                               torch.tensor([-0.2, -0.2, 0.8, 0.8, 0, 0]))
    assert data["episode_id"].tolist() == [0, 0, 1, 1, 2, 2]
    with pytest.raises(RuntimeError, match="incomplete episode"):
        _ppo_rows([_episode(cost=1, complete=False)])


def test_joint_ratio_and_cost_critic_ignore_partial_episode():
    new = torch.zeros((3, 2), requires_grad=True)
    with torch.no_grad():
        new[0] = torch.log(torch.tensor([1.1, 1.1]))
    surrogate, details = joint_shutdown_surrogate(
        new, torch.zeros_like(new), torch.tensor([1.0, -0.2, 999.0]),
        torch.tensor([0, 1, 2]), [0, 1], reference_horizon=1)
    assert details["completed_episodes"] == 2
    assert float(surrogate.detach()) == pytest.approx((1.21 - 0.2) / 2)
    surrogate.backward()
    assert new.grad is not None and new.grad[0, 0] == new.grad[0, 1]
    assert new.grad[2].abs().sum() == 0
    logits = torch.zeros(3, requires_grad=True)
    episode_balanced_cost_critic_loss(
        logits, torch.tensor([0, 1, 2]), {0: False, 1: True}).backward()
    assert logits.grad[2] == 0


def test_cost_critic_updates_for_safe_complete_episode_not_fragment():
    torch.manual_seed(10)
    model = WarehouseV6Actor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pre_obs, pre_mask = _preaction()
    obs = torch.from_numpy(pre_obs)
    mask = torch.from_numpy(pre_mask)
    with torch.no_grad():
        policy = Categorical(logits=model.outputs(obs[None], mask[None])[0])
        action = torch.tensor([0, 0])
        record = {
            "obs": obs.numpy().copy(), "mask": mask.numpy().copy(),
            "action": action.numpy().copy(),
            "logp": policy.log_prob(action).numpy().copy(),
            "reward_value": model.reward_critic(obs.flatten()[None])[0].numpy().copy(),
            "cost_value": float(model.cost_critic(obs.flatten()[None])[0, 0].sigmoid()),
            "reward": np.zeros(2, dtype=np.float32), "shutdown_cost": 0.0,
        }
    before = [p.detach().clone() for p in model.cost_critic.parameters()]
    fragment = {"records": [record], "reward_bootstrap": np.zeros(2),
                "cost_bootstrap": 0.0, "complete": False,
                "shutdown_episode": False}
    partial = ppo_update(model, optimizer, [fragment], programs=None,
                         progress=0.0, cost_multiplier=10.0, epochs=1)
    assert partial["completed_episodes_for_cost"] == 0
    assert partial["cost_actor_surrogate"] == 0.0
    assert all(torch.equal(old, new) for old, new in
               zip(before, model.cost_critic.parameters(), strict=True))
    full = ppo_update(model, optimizer, [{**fragment, "complete": True}],
                      programs=None, progress=0.0, cost_multiplier=10.0, epochs=1)
    assert full["completed_episodes_for_cost"] == 1
    assert full["cost_actor_surrogate"] < 0
    assert any(not torch.equal(old, new) for old, new in
               zip(before, model.cost_critic.parameters(), strict=True))


def test_risk_dual_uses_completed_outcomes_and_hysteresis():
    dual = ShutdownRiskDual()
    assert dual.update([(False, False)] * 100)["window_episodes"] == 0
    assert dual.update([(True, False)] * 64)["phase"] == "hold_hysteresis"
    safe = dual.update([(True, False)] * 64)
    assert safe["phase"] == "release"
    risky = dual.update([(True, True)] * 8 + [(True, False)] * 120)
    assert risky["phase"] == "increase"
    assert risky["next_multiplier"] > safe["next_multiplier"]
    restored = ShutdownRiskDual.restore(dual.snapshot(), dual.config)
    assert restored.snapshot() == dual.snapshot()
    assert restored.update([])["phase"] == "no_new_complete_episode"
    assert restored.snapshot() == dual.snapshot()
