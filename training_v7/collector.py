"""Eight finite physical environments with policy-independent scenario blocks."""
from copy import deepcopy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .bc import choose
from .contracts import MethodSpec, digest, write_new
from .ppo import reward_gae
from .storage import append, export_actor


class Collector:
    def __init__(self, adapter, model, run, *, action_seed, budget):
        self.adapter, self.model, self.run, self.budget = adapter, model, run, budget
        self.generator = torch.Generator().manual_seed(action_seed)
        self.joint_steps = 0; self.states = [None] * 8; self.initials = [None] * 8
        self.pending = [[] for _ in range(8)]; self.current_block = [-1] * 8
        self.cases = {(c['slot'], c['block']): c for c in run.plan['cases'] if c['purpose'] == 'ppo'}

    def _reset(self, slot):
        block = (self.joint_steps // 8) // self.adapter.spec.horizon
        case = self.cases[(slot, block)]
        self.states[slot] = self.run.reset(self.adapter, case, purpose='ppo')
        self.initials[slot] = self.adapter.snapshot(self.states[slot]); self.current_block[slot] = block

    def collect(self):
        method = MethodSpec(); rows_by_time, completed = [], []
        boundaries = [self.budget, method.critic_warmup, *(self.budget * p // 100 for p in method.extraction_percentages)]
        future = [b for b in boundaries if b > self.joint_steps]
        if not future: return [], []
        vector_steps = min(128, (min(future) - self.joint_steps) // 8)
        if vector_steps <= 0: raise ValueError('milestone is not aligned with eight joint frames')
        actor_hashes = self.model.role_hashes()
        for _ in range(vector_steps):
            vector = []
            for slot in range(8):
                block = (self.joint_steps // 8) // self.adapter.spec.horizon
                if self.states[slot] is None or self.adapter.terminal(self.states[slot]): self._reset(slot)
                elif self.current_block[slot] != block:
                    raise ValueError('unrecorded scenario block boundary')
                state = self.states[slot]; public = self.adapter.public(state)
                observations = self.adapter.observe_public(public)
                indices, probabilities = choose(self.model, observations, self.generator)
                joint = np.concatenate([observations[r.name]['x'] for r in self.adapter.spec.roles])
                with torch.no_grad():
                    value = self.model.reward_critic(torch.from_numpy(joint[None]))[0].numpy()
                    cost_value = float(self.model.cost_critic(torch.from_numpy(joint[None])).sigmoid()[0, 0])
                before = self.adapter.snapshot(state)
                append(self.run.journal, dict(event='NN_joint_frame_reserved', joint_step=self.joint_steps+slot,
                    slot=slot, case_id=self.cases[(slot, block)]['case_id'], indices=indices,
                    actor_role_sha256=actor_hashes, public_sha256=digest(public)))
                state, reward = self.adapter.step(state, indices); self.states[slot] = state
                terminated, truncated = self.adapter.ending(state)
                training_cut = (((self.joint_steps // 8 + 1) % self.adapter.spec.horizon == 0)
                                and not self.adapter.terminal(state))
                with torch.no_grad():
                    next_value = self.model.reward_critic(torch.from_numpy(
                        self.adapter.critic_vector(self.adapter.public(state))[None]))[0].numpy()
                row = dict(observations=observations, indices=indices,
                    old_logp={n: float(np.log(probabilities[n][indices[n]])) for n in indices},
                    reward=reward, value=value, next_value=next_value, terminated=terminated, truncated=truncated or training_cut,
                    native_truncated=truncated, training_cut=training_cut,
                    cost_value=cost_value, public=public, before=before, after=self.adapter.snapshot(state),
                    physical_action_overrides=0)
                self.pending[slot].append(row); vector.append(row)
                if self.adapter.terminal(state):
                    outcome = self.adapter.outcome(state, self.adapter.restore(self.initials[slot]))
                    completed.append(dict(rows=list(self.pending[slot]), failed=outcome['failed'], outcome=outcome))
                    self.pending[slot] = []
                elif training_cut:
                    append(self.run.journal, dict(event='incomplete_training_block_cut', slot=slot,
                        frames=len(self.pending[slot]), outcome=None, counted_as_complete=False))
                    self.pending[slot] = []; self.states[slot] = None
                append(self.run.journal, dict(event='NN_joint_frame', joint_step=self.joint_steps+slot,
                    slot=slot, case_id=self.cases[(slot, block)]['case_id'], indices=indices,
                    actor_role_sha256=actor_hashes, public_sha256=digest(public),
                    physical_action_overrides=0, terminated=terminated, truncated=truncated, training_cut=training_cut))
            self.joint_steps += 8; rows_by_time.append(vector)
        rewards = [[r['reward'] for r in v] for v in rows_by_time]
        values = [[r['value'] for r in v] for v in rows_by_time]
        next_values = [[r['next_value'] for r in v] for v in rows_by_time]
        advantage, returns = reward_gae(rewards, values, next_values,
            [[r['terminated'] for r in v] for v in rows_by_time], [[r['truncated'] for r in v] for v in rows_by_time])
        for t, vector in enumerate(rows_by_time):
            for slot, row in enumerate(vector): row.update(advantage=advantage[t, slot], returns=returns[t, slot])
        return [r for v in rows_by_time for r in v], completed

    def snapshot(self):
        return dict(joint_steps=self.joint_steps, states=[self.adapter.snapshot(s) if s is not None else None for s in self.states],
                    initials=self.initials, pending=self.pending, current_block=self.current_block,
                    generator=self.generator.get_state(), used_cases=sorted(self.run.used))

    def restore(self, value):
        if (type(value['joint_steps']) is not int or value['joint_steps'] % 8 or not 0 <= value['joint_steps'] <= self.budget
                or any(len(value[k]) != 8 for k in ('states', 'initials', 'pending', 'current_block'))):
            raise ValueError('invalid finite collector checkpoint')
        self.joint_steps = value['joint_steps']
        self.states = [self.adapter.restore(s) if s is not None else None for s in value['states']]
        self.initials = deepcopy(value['initials']); self.pending = deepcopy(value['pending'])
        self.current_block = list(value['current_block']); self.generator.set_state(value['generator'])
        if set(value['used_cases']) != self.run.used:
            raise ValueError('collector checkpoint and actual reset journal differ')


def train_pair_variant(adapter, engine, run, provider, *, variant, budget, action_seed, sources):
    started = time.perf_counter(); checkpoint_seconds = 0.; collection_seconds = 0.; update_seconds = 0.
    folder = run.folder / variant; folder.mkdir(parents=True, exist_ok=False)
    collector = Collector(adapter, engine.model, run, action_seed=action_seed, budget=budget)
    milestones = {4000, budget, *(budget * p // 100 for p in (75, 84, 93))}
    while collector.joint_steps < budget:
        run.ensure_open()
        collect_actor = export_actor(engine.model, folder/f'collect-actor-{collector.joint_steps}.npz')
        append(run.journal, dict(event='rollout_actor_binding', joint_steps=collector.joint_steps,
                                actor=collect_actor, action_rng=collector.generator.get_state().tolist()))
        before = time.perf_counter()
        rows, episodes = collector.collect(); engine.joint_steps = collector.joint_steps
        collection_seconds += time.perf_counter() - before
        before = time.perf_counter()
        engine.update(rows, episodes, variant=variant, budget=budget, program_provider=provider)
        update_seconds += time.perf_counter() - before
        append(run.journal, dict(event='ppo_rollout_completed', variant=variant, joint_steps=engine.joint_steps,
            actor_updates=engine.actor_updates, critic_updates=engine.critic_updates,
            completed_episodes=len(episodes), risk=engine.risk.snapshot(), program_updates=engine.program_updates,
            program_gradient_norm_sum=engine.program_norm))
        if collector.joint_steps in milestones:
            step = collector.joint_steps
            if step in {budget*p//100 for p in (75, 84, 93)}:
                provider.refit(engine.model, step=step, folder=folder/f'program-step-{step}')
            before = time.perf_counter()
            checkpoint = engine.checkpoint(folder/f'checkpoint-{step}.pt', collector_state=collector.snapshot(),
                programs={n: p.as_dict() for n, p in provider.programs.items()}, plan_sha256=run.plan_hash, sources=sources)
            export_actor(engine.model, folder/f'actor-{step}.npz')
            checkpoint_seconds += time.perf_counter() - before
            append(run.journal, dict(event='checkpoint_quiescent', checkpoint=checkpoint,
                                    plan_sha256=run.plan_hash, joint_steps=step))
    final_programs = provider.refit(engine.model, step=budget, folder=folder/'final-programs')
    result = dict(variant=variant, joint_steps=engine.joint_steps, actor_role_sha256=engine.model.role_hashes(),
        reference_role_sha256=engine.reference.role_hashes(), actor_updates=engine.actor_updates,
        critic_updates=engine.critic_updates, program_updates=engine.program_updates,
        program_gradient_norm_sum=engine.program_norm, accepted_actor_change=engine.accepted_actor_change,
        program_feedback_frames=engine.program_feedback_frames, program_evaluated_frames=engine.program_evaluated_frames,
        final_program_metrics=final_programs, quality_evaluated=False, model_acceptance_passed=False, pilot_opened=False)
    result['training_gates'] = dict(raw_zero_feedback=variant != 'raw' or not (
        any(engine.program_updates.values()) or any(engine.program_norm.values()) or any(engine.program_feedback_frames.values())),
        regularized_both_positive=variant != 'regularized' or all(engine.program_updates.values()),
        both_Actors_changed=all(v > 0 for v in engine.accepted_actor_change.values()))
    elapsed = time.perf_counter() - started
    core_fraction = checkpoint_seconds/(checkpoint_seconds+collection_seconds+update_seconds)
    result['timing'] = dict(total_seconds=elapsed, collection_seconds=collection_seconds, update_seconds=update_seconds,
        checkpoint_seconds=checkpoint_seconds, checkpoint_fraction=checkpoint_seconds/elapsed,
        checkpoint_fraction_excluding_refits=core_fraction,
        checkpoint_scope='full_PPO_checkpoint_and_milestone_actor_export; rollout_actor_provenance_is_in_total',
        engineering_checkpoint_passed=core_fraction <= .10 and checkpoint_seconds/elapsed <= .10,
        scientific_model_gate_unchanged=True)
    write_new(folder/'result.json', result)
    run.finish()
    return result
