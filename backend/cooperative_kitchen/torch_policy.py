"""Shared Actor and centralized two-head Critic for kitchen MAPPO."""
import numpy as np
import torch
from torch import nn


class SharedActor(nn.Module):
    def __init__(self, observation_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_size, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 6))
        self._initialize(.01)

    def _initialize(self, output_gain):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.network[-1].weight, output_gain)

    def forward(self, observation):
        return self.network(observation)


class CentralCritic(SharedActor):
    def __init__(self, state_size):
        nn.Module.__init__(self)
        self.network = nn.Sequential(nn.Linear(state_size, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self._initialize(1.0)


def numpy_layers(module):
    return [(layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy())
            for layer in module.network if isinstance(layer, nn.Linear)]
