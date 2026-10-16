"""The v6.9 hard tree must actually learn in BC and self-play PPO."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from backend.training.warehouse_v6_actor import WarehouseV6Actor
from backend.training.warehouse_v6_runner import (
    _bc_epochs, _initialize_public_trees, ppo_update,
)
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.pure_nn import AGENTS, action_masks


def _public_rows(count: int = 128):
    # The action-geodesic encoder now checks the public map, coordinates, and
    # physical mask. Build two legal pre-action states without an env reset.
    positions = (((1, 0), (5, 2)), ((5, 2), (1, 0)))
    observations, masks, labels = [], [], []
    for index in range(count):
        own, partner = positions[index % 2]
        state = WarehouseState(
            episode_id=690117, frame=2,
            agents=[AgentState("robot_1", own, 70.0),
                    AgentState("robot_2", partner, 70.0)],
            tasks=[DeliveryTask("task_0", (0, 3), (3, 2)),
                   DeliveryTask("task_1", (1, 0), (3, 1))],
            next_task_index=2)
        observations.append(encode_public_observation(state))
        masks.append(action_masks(state))
        # RIGHT at (1, 0); UP at (5, 2). Both are physically legal.
        labels.append((3, 0) if index % 2 == 0 else (0, 3))
    obs = torch.from_numpy(np.stack(observations))
    mask = torch.from_numpy(np.stack(masks))
    targets = torch.tensor(labels, dtype=torch.long)
    assert torch.all(mask.gather(-1, targets[..., None]) > 0)
    return obs, mask, targets


def _split_weights(model):
    return {role: model.actors[role].tree.split.weight.detach().clone()
            for role in AGENTS}


def test_both_hard_role_splits_receive_bc_and_ppo_gradients():
    torch.manual_seed(690117)
    model = WarehouseV6Actor()
    obs, mask, labels = _public_rows()
    initialization = _initialize_public_trees(model, obs, mask, labels)
    assert all(not initialization[role]["split_frozen_during_bc_and_ppo"]
               for role in AGENTS)
    assert all(model.actors[role].tree.split.weight.requires_grad
               for role in AGENTS)
    initial = _split_weights(model)
    _bc_epochs(model, torch.optim.Adam(model.actors.parameters(), lr=3e-4),
               obs, mask, labels, epochs=1,
               generator=torch.Generator().manual_seed(690117))
    after_bc = _split_weights(model)
    assert all(not torch.equal(initial[role], after_bc[role]) for role in AGENTS)

    records = []
    with torch.no_grad():
        for frame in range(24):
            frame_obs, frame_mask = obs[frame], mask[frame]
            policy = Categorical(logits=model.outputs(
                frame_obs[None], frame_mask[None])[0])
            action = labels[frame]
            flat = frame_obs.flatten()[None]
            records.append({
                "obs": frame_obs.numpy().copy(),
                "mask": frame_mask.numpy().copy(),
                "action": action.numpy().copy(),
                "logp": policy.log_prob(action).numpy().copy(),
                "reward_value": model.reward_critic(flat)[0].numpy().copy(),
                "cost_value": float(model.cost_critic(flat)[0, 0].sigmoid()),
                "reward": np.asarray([float(labels[frame, 0]),
                                      float(labels[frame, 1])], dtype=np.float32),
                "shutdown_cost": 0.0,
            })
    update = ppo_update(
        model, torch.optim.Adam(model.parameters(), lr=3e-4),
        [{"records": records, "reward_bootstrap": np.zeros(2, dtype=np.float32),
          "cost_bootstrap": 0.0, "complete": True, "shutdown_episode": False}],
        programs=None, progress=0.0, cost_multiplier=1.0, epochs=1)
    assert all(update["split_gradient_norm"][role] > 0 for role in AGENTS)
    assert all(not torch.equal(after_bc[role], model.actors[role].tree.split.weight)
               for role in AGENTS)
