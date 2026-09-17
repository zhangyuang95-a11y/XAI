"""Synchronous fixed-agent collection. Truncations bootstrap, but stop GAE chains."""
import numpy as np
import torch
from core.interfaces import validate_frame, encode_features


def advantages(rewards, values, next_values, terminated, truncated, gamma, lam):
    result = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        result[t] = carry
    return result, result + values


class Collector:
    def __init__(self, env, model, device, seed):
        self.env, self.model, self.device = env, model, device
        self.frame = env.reset(seed=seed)
        self.samples = 0

    def collect(self, steps, gamma=.99, lam=.95):
        spec = self.env.spec
        roles = torch.arange(spec.agents, device=self.device)
        records = {k: [] for k in ['obs','states','roles','actions','logp','values','next_values',
                                   'rewards','terminated','truncated','mask','features']}
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        for _ in range(steps):
            frame = self.frame
            validate_frame(frame, spec)
            features = [encode_features(self.env, frame, i) for i in range(spec.agents)]
            obs, state = tensor(frame.observations), tensor(np.repeat(frame.state[None,:], spec.agents, axis=0))
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=self.model.logits(obs, roles))
                actions = dist.sample()
                logp = dist.log_prob(actions)
                values = self.model.values(state, roles)
            tr = self.env.step(actions.cpu().numpy().copy())
            validate_frame(tr.frame, spec)
            rewards = np.asarray(tr.rewards, dtype=np.float32)
            term, trunc, mask = [np.asarray(v, dtype=bool) for v in (tr.terminated,tr.truncated,tr.actor_mask)]
            if any(v.shape != (spec.agents,) for v in (rewards,term,trunc,mask,np.asarray(tr.submitted_actions))):
                raise ValueError("Transition vectors must match agent count")
            if not np.isfinite(rewards).all(): raise ValueError("Non-finite reward")
            if np.any(mask & (tr.submitted_actions != actions.cpu().numpy())):
                raise ValueError("External action must have actor_mask=False")
            ended = term | trunc
            if ended.any() and not ended.all():
                raise ValueError("This framework requires synchronous episode boundaries")
            with torch.no_grad():
                nxt = self.model.values(tensor(np.repeat(tr.frame.state[None,:],spec.agents,axis=0)),roles)
            row = dict(obs=obs.cpu().numpy(), states=state.cpu().numpy(), roles=roles.cpu().numpy(),
                       actions=actions.cpu().numpy(),logp=logp.cpu().numpy(),values=values.cpu().numpy(),
                       next_values=nxt.cpu().numpy(), rewards=rewards,terminated=term,truncated=trunc,
                       mask=mask,features=features)
            for k,v in row.items(): records[k].append(v)
            self.samples += int(mask.sum())
            self.frame = self.env.reset() if ended.all() else tr.frame
        data = {k: np.asarray(v) for k,v in records.items() if k != 'features'}
        adv, returns = advantages(data['rewards'],data['values'],data['next_values'],data['terminated'],data['truncated'],gamma,lam)
        data.update(advantages=adv, returns=returns)
        data = {k: v.reshape((-1,) + v.shape[2:]) for k,v in data.items()}
        data['features'] = [f for step in records['features'] for f in step]
        return data
