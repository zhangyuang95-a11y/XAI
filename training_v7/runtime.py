"""Explicit model-bound NN execution for all three domain physics adapters.

This is preview/test infrastructure until a separately verified v7 release
exists. It does not claim that a model has passed scientific admission.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import json
import numpy as np
from .contracts import CANDIDATE, RELEASE, MethodSpec, digest, file_sha
from .storage import load_frozen


@dataclass(frozen=True)
class Binding:
    domain_sha256: str
    policy_id: str
    variant: str
    actor_path: str
    actor_sha256: str
    role_sha256: tuple
    method_sha256: str = MethodSpec().sha256
    release_id: str = RELEASE

    def __post_init__(self):
        if (self.variant not in ('raw', 'regularized') or self.method_sha256 != MethodSpec().sha256
                or self.release_id != RELEASE or len(self.actor_sha256) != 64
                or len(self.role_sha256) != 2 or len({r for r, _ in self.role_sha256}) != 2
                or not self.policy_id):
            raise ValueError('invalid explicit policy binding')


class Runtime:
    def __init__(self, adapter, binding, *, mode='test'):
        if mode not in ('test', 'preview'):
            raise ValueError('v7 pilot remains closed until new three-domain release admission is implemented and passes')
        self.adapter, self.binding = adapter, binding
        if binding.domain_sha256 != adapter.spec.sha256: raise ValueError('bound domain or physics differs')
        self.actors = load_frozen(binding.actor_path, binding.actor_sha256, adapter.spec, dict(binding.role_sha256))

    def _check(self):
        if file_sha(self.binding.actor_path) != self.binding.actor_sha256: raise ValueError('model asset missing or changed')

    def wrap(self, native, *, seed, task, action_seed=None):
        self._check()
        state = dict(version='three-domain-unified-state.v1', binding=json.loads(json.dumps(asdict(self.binding))), seed=seed, task=task,
                     turn=self.adapter.frame(native),
                     action_seed=seed if action_seed is None else action_seed,
                     snapshot=self.adapter.snapshot(native))
        state = json.loads(json.dumps(state, allow_nan=False))
        state['state_sha256'] = digest(state)
        return state

    def initial_state(self, seed, task, binding):
        if binding != self.binding: raise ValueError('explicit binding differs')
        return self.wrap(self.adapter.initial_state(seed, task), seed=seed, task=task)

    def restore(self, state, binding):
        self._check()
        if (binding != self.binding or state['binding'] != json.loads(json.dumps(asdict(binding)))
                or digest({k: v for k, v in state.items() if k != 'state_sha256'}) != state['state_sha256']):
            raise ValueError('saved state, model or integrity binding differs')
        native = self.adapter.restore(state['snapshot'])
        if (type(state['action_seed']) is not int or not 0 <= state['action_seed'] < 2**63
                or state['turn'] != self.adapter.frame(native) or state['task'] not in (1, 2, 3)):
            raise ValueError('saved physical frame or action random stream differs')
        visible_task = self.adapter.public(native).get('task')
        if visible_task is not None and visible_task != state['task']: raise ValueError('saved task and physical task differ')
        return native

    def decide(self, state, binding, *, action_seed=None):
        native = self.restore(state, binding)
        if self.adapter.terminal(native): raise ValueError('terminal state has no executable decision')
        obs = self.adapter.observe_public(self.adapter.public(native))
        if action_seed is not None and action_seed != state['action_seed']:
            raise ValueError('decision cannot replace the saved action random stream')
        action_seed = state['action_seed']
        rng = np.random.default_rng(action_seed)
        rng.bit_generator.advance(state['turn'] * 2)
        uniforms = rng.random(2)
        indices, distributions = {}, {}
        for ordinal, role in enumerate(self.adapter.spec.roles):
            o = obs[role.name]
            z = self.actors[role.name].logits(o['x'], o['action_features'], o['mask']).astype(float)
            p = np.exp(z - np.max(z)); p /= p.sum()
            cdf = np.cumsum(p); cdf /= cdf[-1]
            indices[role.name] = int(np.searchsorted(cdf, uniforms[ordinal], side='right'))
            distributions[role.name] = p.tolist()
        return dict(state_sha256=state['state_sha256'], indices=indices, probabilities=distributions, action_seed=action_seed)

    def step(self, state, human_command, binding, *, decision=None):
        native = self.restore(state, binding)
        expected = self.decide(state, binding, action_seed=decision['action_seed'] if decision else None)
        if decision is not None and decision != expected: raise ValueError('submitted NN decision differs')
        indices = dict(expected['indices'])
        if human_command is not None:
            role = self.adapter.spec.roles[0]
            if type(human_command) is not int or not 0 <= human_command < len(role.actions): raise ValueError('invalid human command')
            if not self.adapter.observe_public(self.adapter.public(native))[role.name]['mask'][human_command]:
                raise ValueError('illegal human command')
            indices[role.name] = human_command
        after, reward = self.adapter.step(native, indices)
        return self.wrap(after, seed=state['seed'], task=state['task'], action_seed=state['action_seed'])

    def demonstration(self, binding, *, state, steps=12):
        if not 1 <= steps <= 1200: raise ValueError('finite demonstration length required')
        current = deepcopy(state); records = []
        for _ in range(steps):
            native = self.restore(current, binding)
            if self.adapter.terminal(native): break
            decision = self.decide(current, binding)
            after = self.step(current, None, binding, decision=decision)
            records.append(dict(before=current, decision=decision, human=None, after=after)); current = after
        return dict(initial=deepcopy(state), records=records)

    def replay(self, bundle, binding):
        current = deepcopy(bundle['initial'])
        for row in bundle['records']:
            if row['before'] != current: raise ValueError('replay frame chain differs')
            current = self.step(current, row['human'], binding, decision=row['decision'])
            if current != row['after']: raise ValueError('replay physical result differs')
        return current

    def counterfactual(self, state, human_commands, binding):
        if not 1 <= len(human_commands) <= 12: raise ValueError('finite counterfactual commands required')
        current = deepcopy(state); records = []
        for command in human_commands:
            decision = self.decide(current, binding)
            after = self.step(current, command, binding, decision=decision)
            records.append(dict(before=current, decision=decision, human=command, after=after)); current = after
        return dict(initial=deepcopy(state), records=records)

    def explanation(self, state, binding):
        self.restore(state, binding)
        return dict(available=False, reason='new_v7_independent_program_audit_not_available',
                    text='当前缺少新模型的独立程序审计证据，无法推断其动机。')
