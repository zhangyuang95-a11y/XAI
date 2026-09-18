"""Small shared Pong Actor and centralized Critic."""
from __future__ import annotations

import torch
from torch import nn


class PongActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int = 3, hidden_size: int = 128) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(self.observation_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_size * 2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1),
        )

    def actor_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def values(self, joint_observations: torch.Tensor) -> torch.Tensor:
        return self.critic(joint_observations).squeeze(-1)
