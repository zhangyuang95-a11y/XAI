"""Real shutdown event costs cannot invent Actor risk in event-free PPO batches."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from backend.training.warehouse_v6_actor import WarehouseV6Actor
from backend.training.warehouse_v6_runner import _cost_actor_advantage, ppo_update
from env.warehouse.public_observation_v6 import OBSERVATION_DIM


def test_cost_actor_advantage_keeps_real_event_scale_and_zeros_event_free_batch():
    data = {"cost_adv": torch.tensor([0.007, -0.004])}
    no_event = [{"records": [{"shutdown_cost": 0.0}, {"shutdown_cost": 0.0}]}]
    advantage, count, raw_std = _cost_actor_advantage(data, no_event)
    assert count == 0 and torch.count_nonzero(advantage) == 0
    assert raw_std > 0

    one_event = [{"records": [{"shutdown_cost": 0.0}, {"shutdown_cost": 1.0}]}]
    advantage, count, raw_std = _cost_actor_advantage(data, one_event)
    assert count == 1
    torch.testing.assert_close(advantage, data["cost_adv"])
    assert raw_std < 0.01  # Unlike the old implementation, not rescaled to one.


def test_cost_critic_still_learns_from_event_free_returns():
    torch.manual_seed(10)
    model = WarehouseV6Actor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
    masks = torch.ones((2, 5), dtype=torch.float32)
    with torch.no_grad():
        policy = Categorical(logits=model.outputs(obs[None], masks[None])[0])
        actions = torch.tensor([0, 0])
        logp = policy.log_prob(actions).numpy()
        reward_value = model.reward_critic(obs.flatten()[None])[0].numpy()
        cost_value = float(model.cost_critic(obs.flatten()[None])[0, 0])
    before = [parameter.detach().clone() for parameter in model.cost_critic.parameters()]
    episode = {"records": [{
        "obs": obs.numpy().copy(), "mask": masks.numpy().copy(),
        "action": actions.numpy().copy(), "logp": logp.copy(),
        "reward_value": reward_value.copy(), "cost_value": cost_value,
        "reward": np.zeros(2, dtype=np.float32), "shutdown_cost": 0.0,
    }], "reward_bootstrap": np.zeros(2, dtype=np.float32),
        "cost_bootstrap": 0.0}
    diagnostic = ppo_update(model, optimizer, [episode], programs=None,
                            progress=0.0, cost_multiplier=10.0, epochs=1)
    assert diagnostic["cost_event_rows"] == 0
    assert diagnostic["cost_actor_surrogate"] == 0.0
    assert any(not torch.equal(old, new) for old, new in
               zip(before, model.cost_critic.parameters(), strict=True))
