import torch
from torch import nn

class ActorCritic(nn.Module):
    def __init__(self, spec, hidden=128):
        super().__init__()
        self.spec, self.hidden = spec, hidden
        def mlp(inputs, outputs):
            return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))
        self.actor = mlp(spec.observation_size + spec.agents, len(spec.action_names))
        self.critic = mlp(spec.state_size + spec.agents, 1)

    def role(self, roles):
        return torch.nn.functional.one_hot(roles.long(), self.spec.agents).float()

    def logits(self, observations, roles):
        return self.actor(torch.cat((observations, self.role(roles)), -1))

    def values(self, states, roles):
        return self.critic(torch.cat((states, self.role(roles)), -1)).squeeze(-1)

    def probabilities(self, observations, roles):
        return self.logits(observations, roles).softmax(-1)


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA unavailable")
    if name == "mps" and not torch.backends.mps.is_available(): raise ValueError("MPS unavailable")
    return torch.device(name)
