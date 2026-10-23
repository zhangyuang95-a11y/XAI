"""One continuation PPO numerical engine for variable-action independent Actors.

Collectors provide registered public frames, never teacher actions. This core
has no domain-name dispatch and no authority to open a pilot or a sealed set.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from torch.nn import functional as F
from .contracts import MethodSpec
from .losses import masked_kl, reference_loss, next_beta, program_loss, risk_config
from .model import tensor_hash
from .storage import save_checkpoint


def reward_gae(rewards, values, next_values, terminated, truncated):
    method = MethodSpec()
    r, v, nv = (np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values))
    term, trunc = np.asarray(terminated, bool), np.asarray(truncated, bool)
    if r.shape != v.shape or r.shape != nv.shape or r.shape[:-1] != term.shape or term.shape != trunc.shape or np.any(term & trunc):
        raise ValueError('invalid rollout or end flags')
    advantage = np.zeros_like(r); following = np.zeros_like(r[0])
    for i in range(len(r) - 1, -1, -1):
        delta = r[i] + method.gamma * nv[i] * (~term[i])[..., None] - v[i]
        following = delta + method.gamma * method.gae_lambda * following * (~(term[i] | trunc[i]))[..., None]
        advantage[i] = following
    return advantage.astype(np.float32), (advantage + v).astype(np.float32)


def _tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


class PPO:
    def __init__(self, model, *, optimizer_seed, reference_rows):
        from backend.training.warehouse_v6_shutdown_cost import ShutdownRiskDual
        self.model = model; self.reference = deepcopy(model).eval()
        for p in self.reference.parameters(): p.requires_grad_(False)
        self.method = MethodSpec(); self.reference_rows = reference_rows
        self.actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=self.method.actor_lr)
        self.critic_optimizer = torch.optim.Adam([*model.reward_critic.parameters(), *model.cost_critic.parameters()], lr=self.method.critic_lr)
        self.generator = torch.Generator().manual_seed(optimizer_seed)
        self.risk = ShutdownRiskDual(risk_config()); self.beta = {r.name: 1. for r in model.domain.roles}
        self.actor_updates = 0; self.critic_updates = 0; self.joint_steps = 0
        self.program_updates = {r.name: 0 for r in model.domain.roles}
        self.program_norm = {r.name: 0. for r in model.domain.roles}
        self.program_feedback_frames = {r.name: 0 for r in model.domain.roles}
        self.program_evaluated_frames = {r.name: 0 for r in model.domain.roles}
        self.accepted_actor_change = {r.name: 0. for r in model.domain.roles}
        self.reference_buffer_sha256 = {n: tensor_hash({k: torch.from_numpy(np.array(v, copy=True))
            for k, v in values.items()}) for n, values in reference_rows.items()}

    def update(self, rows, complete_episodes, *, variant, budget, program_provider):
        """One fixed rollout; provider must audit each current epoch independently.

        complete_episodes contain complete physical trajectories and true 0/1
        outcomes, including rows held across rollout boundaries. No unfinished
        episode is labelled safe. Provider receives no optimizer.
        """
        if variant not in ('raw', 'regularized') or budget not in (20000, *self.method.budgets):
            raise ValueError('unknown registered treatment or budget')
        if not rows: raise ValueError('empty PPO rollout')
        roles = self.model.domain.roles
        self.risk.update([(True, bool(e['failed'])) for e in complete_episodes])
        cost_rows = [dict(row, cost_outcome=int(e['failed'])) for e in complete_episodes for row in e['rows']]
        advantage = np.array([r['advantage'] for r in rows])
        advantage = (advantage - advantage.mean(0)) / (advantage.std(0) + 1e-8)
        for row, value in zip(rows, advantage, strict=True): row['normalized_advantage'] = value
        actor_active = self.joint_steps > self.method.critic_warmup
        for epoch in range(self.method.ppo_epochs):
            # A changed Actor invalidates the previous epoch's qualifications.
            envelope = program_provider(self.model, epoch) if (variant == 'regularized' and actor_active
                and self.method.coefficient(self.joint_steps, budget) > 0) else None
            for batch_number, selected in enumerate(torch.randperm(len(rows), generator=self.generator).split(self.method.minibatch)):
                batch = [rows[i] for i in selected.tolist()]
                self.actor_optimizer.zero_grad(set_to_none=True); self.critic_optimizer.zero_grad(set_to_none=True)
                joint_obs = _tensor([np.concatenate([r['observations'][role.name]['x'] for role in roles]) for r in batch])
                returns = _tensor([r['returns'] for r in batch])
                critic_loss = self.method.critic_weight * F.mse_loss(self.model.reward_critic(joint_obs), returns)
                loss = critic_loss; actor_loss = critic_loss * 0.
                reported_kl = {}; pg = {}
                for role_index, role in enumerate(roles):
                    n = role.name; x = _tensor([r['observations'][n]['x'] for r in batch])
                    af = _tensor([r['observations'][n]['action_features'] for r in batch])
                    mask = _tensor([r['observations'][n]['mask'] for r in batch], torch.bool)
                    sampled = _tensor([r['indices'][n] for r in batch], torch.long)
                    logits = self.model.actors[n](x, af, mask)
                    logp = logits.log_softmax(-1).gather(-1, sampled[:, None])[:, 0]
                    old = _tensor([r['old_logp'][n] for r in batch])
                    ratio = (logp - old).exp(); adv = _tensor([r['normalized_advantage'][role_index] for r in batch])
                    gain = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
                    safe_logp = logits.log_softmax(-1).masked_fill(~mask, 0.)
                    entropy = -(logits.softmax(-1) * safe_logp).sum(-1).mean()
                    # Reference/current states are exactly 50/50 train-only rows.
                    half = len(batch) // 2
                    if not half: continue
                    reference_ids = torch.randint(len(self.reference_rows[n]['x']), (half,), generator=self.generator).numpy()
                    buffer = self.reference_rows[n]
                    rx = torch.cat((_tensor(buffer['x'][reference_ids]), x[:half]), 0)
                    raf = torch.cat((_tensor(buffer['action_features'][reference_ids]), af[:half]), 0)
                    rm = torch.cat((_tensor(buffer['mask'][reference_ids], torch.bool), mask[:half]), 0)
                    current = self.model.actors[n](rx, raf, rm)
                    with torch.no_grad(): anchor = self.reference.actors[n](rx, raf, rm)
                    ref_loss = reference_loss(current, anchor, rm, ['reference_train'] * half + ['current_train'] * half, self.beta[n])
                    reported_kl[n] = float(masked_kl(anchor, current, rm).mean().detach())
                    role_loss = -gain - self.method.entropy * entropy + ref_loss
                    coefficient = self.method.coefficient(self.joint_steps, budget)
                    if variant == 'regularized' and coefficient > 0 and envelope is not None:
                        # The provider binds the immutable epoch NN; recheck it
                        # after each accepted update, not just at epoch start.
                        targets = envelope.targets(n, x.detach().numpy(), mask.numpy(), sampled.numpy(),
                                                   current_actor=self.model.actors[n])
                        q, qualified, raw_top = targets
                        ploss, used = program_loss(logits.masked_fill(~mask, 0.), _tensor(q), mask, sampled,
                            _tensor(qualified, torch.bool), _tensor(raw_top, torch.long), variant=variant, coefficient=coefficient)
                        self.program_evaluated_frames[n] += len(used)
                        self.program_feedback_frames[n] += int(used.sum())
                        if used.any() and ploss.requires_grad:
                            gradients = torch.autograd.grad(ploss, tuple(self.model.actors[n].parameters()), retain_graph=True, allow_unused=True)
                            norm = sum(float(g.square().sum()) for g in gradients if g is not None) ** .5
                            if norm > 0: pg[n] = norm
                        role_loss = role_loss + ploss
                    actor_loss = actor_loss + role_loss
                if cost_rows:
                    # Full-episode cost rows are balanced by episode, not length.
                    from backend.training.warehouse_v6_shutdown_cost import episode_balanced_cost_critic_loss, joint_shutdown_surrogate
                    episode_ids = []; cost_old = []; cost_adv = []; outcomes = {}; flattened_cost_rows = []
                    all_cost_obs = []
                    selected_episode_ids = torch.randperm(len(complete_episodes), generator=self.generator)[:self.method.minibatch].tolist()
                    samples_per_episode = max(1, self.method.minibatch // len(selected_episode_ids))
                    for identifier in selected_episode_ids:
                        episode = complete_episodes[identifier]
                        outcomes[identifier] = bool(episode['failed'])
                        choices = torch.randint(len(episode['rows']), (samples_per_episode,), generator=self.generator).tolist()
                        for choice in choices:
                            row = episode['rows'][choice]
                            flattened_cost_rows.append(row)
                            all_cost_obs.append(np.concatenate([row['observations'][r.name]['x'] for r in roles]))
                            episode_ids.append(identifier); cost_old.append([row['old_logp'][r.name] for r in roles])
                            # An unbiased episode-sum estimator; short failures
                            # are not diluted and sampling is never a fake label.
                            cost_adv.append((float(episode['failed']) - row['cost_value'])
                                            * len(episode['rows']) / samples_per_episode)
                    cost_logp = []
                    for role in roles:
                        name = role.name
                        z = self.model.actors[name](_tensor([r['observations'][name]['x'] for r in flattened_cost_rows]),
                            _tensor([r['observations'][name]['action_features'] for r in flattened_cost_rows]),
                            _tensor([r['observations'][name]['mask'] for r in flattened_cost_rows], torch.bool))
                        actions = _tensor([r['indices'][name] for r in flattened_cost_rows], torch.long)
                        cost_logp.append(z.log_softmax(-1).gather(-1, actions[:, None])[:, 0])
                    # Avoid differentiating the cost actor during critic warmup.
                    cost_logits = self.model.cost_critic(_tensor(all_cost_obs))[:, 0]
                    cost_value_loss = episode_balanced_cost_critic_loss(cost_logits, _tensor(episode_ids, torch.long), outcomes)
                    loss = loss + self.method.critic_weight * cost_value_loss
                    surrogate, _ = joint_shutdown_surrogate(torch.stack(cost_logp, dim=-1), _tensor(cost_old), _tensor(cost_adv),
                        _tensor(episode_ids, torch.long), list(outcomes), reference_horizon=self.model.domain.horizon)
                    actor_loss = actor_loss + self.risk.multiplier * surrogate
                if actor_active: loss = loss + actor_loss
                if not torch.isfinite(loss): raise ValueError('nonfinite common PPO loss')
                before = {r.name: [p.detach().clone() for p in self.model.actors[r.name].parameters()] for r in roles}
                loss.backward()
                if actor_active:
                    torch.nn.utils.clip_grad_norm_(self.model.actors.parameters(), .5, error_if_nonfinite=True)
                    self.actor_optimizer.step(); self.actor_updates += 1
                    for n in self.beta:
                        if n in reported_kl: self.beta[n] = next_beta(self.beta[n], reported_kl[n])
                    for r in roles:
                        change = sum(float((p.detach() - old).square().sum()) for p, old in zip(self.model.actors[r.name].parameters(), before[r.name], strict=True)) ** .5
                        self.accepted_actor_change[r.name] += change
                        if r.name in pg and change > 0:
                            self.program_updates[r.name] += 1; self.program_norm[r.name] += pg[r.name]
                torch.nn.utils.clip_grad_norm_([*self.model.reward_critic.parameters(), *self.model.cost_critic.parameters()], .5, error_if_nonfinite=True)
                self.critic_optimizer.step(); self.critic_updates += 1
                # Provider invalidates its epoch proof after weights change.
                if envelope is not None: envelope.invalidate_after_update(self.model.role_hashes())

    def checkpoint(self, path, *, collector_state, programs, plan_sha256, sources):
        return save_checkpoint(path, dict(version='three-domain-unified-ppo-checkpoint.v1',
            method_sha256=self.method.sha256, domain_sha256=self.model.domain.sha256, plan_sha256=plan_sha256,
            source_files=sources, model=self.model.state_dict(), reference=self.reference.state_dict(),
            reference_buffer_sha256=self.reference_buffer_sha256,
            reference_role_sha256=self.reference.role_hashes(), model_role_sha256=self.model.role_hashes(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            generator=self.generator.get_state(), torch_rng=torch.get_rng_state(), beta=self.beta,
            risk=self.risk.snapshot(), joint_steps=self.joint_steps, actor_updates=self.actor_updates,
            critic_updates=self.critic_updates, program_updates=self.program_updates, program_norm=self.program_norm,
            program_feedback_frames=self.program_feedback_frames, program_evaluated_frames=self.program_evaluated_frames,
            accepted_actor_change=self.accepted_actor_change, collector=collector_state, programs=programs))

    def restore_checkpoint(self, path, expected_sha256, *, root, sources, plan_sha256):
        from .contracts import file_sha
        from .storage import verify_sources
        from backend.training.warehouse_v6_shutdown_cost import ShutdownRiskDual
        if file_sha(path) != expected_sha256: raise ValueError('checkpoint file hash differs')
        verify_sources(root, sources)
        # Only local, hash-bound checkpoints produced by this owner are read.
        value = torch.load(path, map_location='cpu', weights_only=False)
        if (value['version'] != 'three-domain-unified-ppo-checkpoint.v1'
                or value['method_sha256'] != self.method.sha256
                or value['domain_sha256'] != self.model.domain.sha256
                or value['plan_sha256'] != plan_sha256 or value['source_files'] != sources
                or value['reference_role_sha256'] != self.reference.role_hashes()
                or value['reference_buffer_sha256'] != self.reference_buffer_sha256):
            raise ValueError('checkpoint reference, source, method or experiment differs')
        if any(not torch.isfinite(v).all() for group in ('model', 'reference') for v in value[group].values()):
            raise ValueError('checkpoint has nonfinite model state')
        self.model.load_state_dict(value['model'], strict=True)
        self.reference.load_state_dict(value['reference'], strict=True)
        self.model.train()
        for parameter in self.model.parameters(): parameter.requires_grad_(True)
        self.reference.eval()
        for parameter in self.reference.parameters(): parameter.requires_grad_(False)
        if self.model.role_hashes() != value['model_role_sha256'] or self.reference.role_hashes() != value['reference_role_sha256']:
            raise ValueError('checkpoint tensor hashes differ')
        self.actor_optimizer.load_state_dict(value['actor_optimizer'])
        self.critic_optimizer.load_state_dict(value['critic_optimizer'])
        self.generator.set_state(value['generator']); torch.set_rng_state(value['torch_rng'])
        self.beta = value['beta']; self.risk = ShutdownRiskDual.restore(value['risk'], risk_config())
        if set(self.beta) != set(self.model.actors) or any(not .001 <= b <= 100 for b in self.beta.values()):
            raise ValueError('checkpoint reference dual state differs')
        for field in ('joint_steps', 'actor_updates', 'critic_updates', 'program_updates', 'program_norm',
                      'program_feedback_frames', 'program_evaluated_frames', 'accepted_actor_change'):
            setattr(self, field, value[field])
        return value['collector'], value['programs']
