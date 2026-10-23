"""Explicit DI for accepted Pong/Kitchen v5 weights over unchanged physics.

Only NN forward and physical transitions are reused. Old policy registries,
deployment lookup, rule planners and implicit constructor scenes are unused.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from core.program import ExecutableProgram
from domains.pong import nn_policy as pong_nn, turnbased as pong_physics
from domains.kitchen import neural_policy as kitchen_nn, engine_v5_frozen as kitchen_physics

from .catalog import PolicyCatalog, decode
from .contracts import PolicyBinding, GROUP_VARIANTS, digest, require
from .registry import BoundDomain

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'runtime-v642-carried-domain-state.v1'
REPLAY_VERSION = 'runtime-v642-carried-domain-replay.v1'
PROGRAM_SCHEMAS = {d: f'runtime-v642-carried-{d}-program-evidence.v1' for d in ('pong', 'kitchen')}
CONFIG_SHA = dict(pong='1ade6a67b51884113594884ee27215af70d4a85528c91f03839e4e6961af7d42',
    kitchen='b0aeeea1772221f047e294c44c6632a98ff0b92f5e5e645678142d62c7fd7b6d')
ORIGINAL_ACCEPTANCE_SHA = dict(pong='ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958',
    kitchen='012af827e8416f293bec97670bb3aec1b752de58dc0155051e98d33c56eabc9a')
METHOD_SCHEMA = 'carried-v5-paired-PPO-frozen-NN.v1'
STATE_FIELDS = frozenset(('version', 'domain', 'binding', 'binding_identity_sha256', 'task', 'turn',
    'max_turns', 'terminal', 'snapshot', 'snapshot_sha256', 'provenance', 'state_sha256'))


def domain_contract(domain):
    require(domain in CONFIG_SHA, 'unknown carried NN domain')
    if domain == 'pong':
        feature = digest(dict(model=pong_nn.MODEL_VERSION, names=pong_nn.FEATURE_NAMES,
            actions=pong_nn.ACTIONS, mask_features=['legal_left', 'legal_right']))
        return dict(model_schema=pong_nn.MODEL_VERSION, method_schema=METHOD_SCHEMA,
            feature_schema_sha256=feature, action_physics_config_sha256=CONFIG_SHA[domain])
    feature = digest(dict(version=kitchen_nn.FEATURE_VERSION, human_actions=kitchen_nn.HUMAN_ACTIONS,
        ai_actions=kitchen_nn.AI_ACTIONS, encoder_sha256=sha256(Path(kitchen_nn.__file__).read_bytes()).hexdigest()))
    return dict(model_schema=kitchen_nn.MODEL_VERSION, method_schema=METHOD_SCHEMA,
        feature_schema_sha256=feature, action_physics_config_sha256=CONFIG_SHA[domain])


def load_kitchen_actor(path, expected_sha256):
    """Same exact frozen implementation, without its implicit initial_state call."""
    path = Path(path); require(sha256(path.read_bytes()).hexdigest() == expected_sha256, 'Kitchen NPZ hash differs')
    actor = kitchen_nn.FrozenActors.__new__(kitchen_nn.FrozenActors)
    actor.sha256 = expected_sha256
    with np.load(path, allow_pickle=False) as archive:
        actor.metadata = json.loads(str(archive['metadata_json'].item()))
        actor.role_sha256 = {role: kitchen_nn.role_weight_sha256(archive, role) for role in ('human', 'ai')}
        actor.weights = {key: archive[key].astype(np.float64).copy() for key in archive.files if key != 'metadata_json'}
    m = actor.metadata; hidden, observation = m.get('hidden'), m.get('observation_dim')
    require(m.get('model_version') == kitchen_nn.MODEL_VERSION and m.get('feature_version') == kitchen_nn.FEATURE_VERSION
        and type(hidden) is int and hidden > 0 and type(observation) is int and observation > 0
        and tuple(m.get('human_actions', ())) == kitchen_nn.HUMAN_ACTIONS
        and tuple(m.get('ai_actions', ())) == kitchen_nn.AI_ACTIONS, 'Kitchen NPZ model contract differs')
    expected = set()
    for role, actions in (('human', kitchen_nn.HUMAN_ACTIONS), ('ai', kitchen_nn.AI_ACTIONS)):
        dims = (observation, hidden, hidden, len(actions))
        for layer in range(3):
            for kind, shape in (('weight', (dims[layer + 1], dims[layer])), ('bias', (dims[layer + 1],))):
                name = f'{role}.{kind}{layer}'; expected.add(name); value = actor.weights.get(name)
                require(value is not None and value.shape == shape and np.isfinite(value).all(), 'Kitchen NPZ tensor contract differs')
                value.flags.writeable = False
    require(set(actor.weights) == expected, 'Kitchen NPZ has undeclared tensors')
    return actor


class FrozenLegacyCore:
    """Pure numerical/physics core, supplied exact resolver/binding/config."""
    def __init__(self, resolver, binding, runtime_config_sha256):
        require(type(binding) is PolicyBinding and binding.domain in CONFIG_SHA
            and runtime_config_sha256 == CONFIG_SHA[binding.domain], 'unknown carried-domain/config identity')
        contract = domain_contract(binding.domain)
        require(all(getattr(binding, k) == v for k, v in contract.items()), 'carried NN feature/physics/model contract differs')
        self.assets, self.binding, self.domain, self.config_sha = resolver, binding, binding.domain, runtime_config_sha256
        self.config = decode(resolver.path(runtime_config_sha256).read_bytes())
        path = resolver.path(binding.actor_sha256)
        if self.domain == 'pong':
            require(self.config == pong_physics.CONFIG, 'Pong physical constants differ from pinned config')
            self.actor = pong_nn._validate_payload(decode(path.read_bytes()))
            roles = {role: pong_nn._role_digest(self.actor['actors'][role]) for role in ('human', 'ai')}
        else:
            require(self.config.get('rules_version') == kitchen_physics.VERSION
                and self.config.get('menu_version', kitchen_physics.MENU_VERSION) == kitchen_physics.MENU_VERSION,
                'Kitchen physical config differs')
            self.actor = load_kitchen_actor(path, binding.actor_sha256)
            roles = self.actor.role_sha256
        require(roles == dict(human=binding.human_actor_sha256, ai=binding.ai_actor_sha256), 'actual dual Actor role hashes differ')

    def verify(self):
        self.assets.path(self.binding.actor_sha256); self.assets.path(self.config_sha)

    def validate_native(self, state):
        self.verify()
        require(type(state) is dict and state.get('domain') == self.domain and type(state.get('task')) is int
            and state['task'] in (1, 2, 3) and type(state.get('turn')) is int
            and type(state.get('max_turns')) is int and 0 <= state['turn'] <= state['max_turns']
            and type(state.get('terminal')) is bool and type(state.get('policy_memory')) is dict,
            'invalid carried physical snapshot')
        if state.get('policy_id') is not None:
            require(state['policy_id'] == self.binding.policy_id
                and state.get('role_sha256') == dict(human=self.binding.human_actor_sha256, ai=self.binding.ai_actor_sha256)
                and state.get('actor_sha256', state.get('policy_sha256')) == self.binding.actor_sha256,
                'underlying legacy policy binding differs')
        require(state.get('release_id') in (None, self.binding.release_id), 'underlying release identity differs')
        if self.domain == 'pong':
            require(state.get('version') in (pong_physics.VERSION, pong_physics.RELEASE_VERSION)
                and state.get('scenario_version') == self.config['scenario_version']
                and state.get('lanes') == pong_physics.LANES and state.get('height') == pong_physics.HEIGHT
                and type(state.get('_schedule')) is list and type(state.get('balls')) is list
                and len(state['balls']) <= pong_nn.MAX_BALLS
                and all(type(state[r].get('x')) is int and 0 <= state[r]['x'] < pong_physics.LANES for r in ('human', 'ai')),
                'Pong snapshot schema/geometry differs')
        else:
            require(state.get('version') in (kitchen_physics.VERSION, 'kitchen-v7.0.0-neural', 'kitchen-v8.0.0-dual-actor-paired')
                and state.get('scenario_version') == kitchen_physics.SCENARIO_VERSION
                and state.get('menu_version') == kitchen_physics.MENU_VERSION
                and all((state[r]['x'], state[r]['y']) in kitchen_physics.FLOOR for r in ('human', 'ai'))
                and len(kitchen_nn.features(state)) == self.actor.metadata['observation_dim'], 'Kitchen snapshot schema/geometry differs')
        digest(state)  # rejects nonfinite/opaque state before NN or physics
        return state

    def choose(self, state):
        self.validate_native(state); require(not state['terminal'], 'no NN action after terminal frame')
        choices, scores, masks = {}, {}, {}
        for role in ('human', 'ai'):
            if self.domain == 'pong':
                scores[role] = pong_nn.logits_from_payload(state, role, self.actor)
                masks[role] = list(pong_nn.legal_mask(state, role))
                choices[role] = pong_nn.choose_from_payload(state, role, self.actor)
            else:
                choice, raw, mask = self.actor.choose(state, role)
                choices[role], scores[role], masks[role] = choice, raw.tolist(), mask.tolist()
        return choices, scores, masks

    def transition(self, state, choices):
        self.validate_native(state)
        if self.domain == 'pong':
            return pong_physics.step_pair(state, choices['human'], choices['ai'])
        spec = kitchen_nn.action_spec(choices['ai'])
        return kitchen_physics.step(state, choices['human'], {**spec, 'action_id': choices['ai'],
            'policy_origin': 'kitchen_nn', 'reason_code': 'neural_selection', 'memory': {}})


class LegacyProgramEvidence:
    """Historical development evidence, never represented as a new audit."""
    def __init__(self, catalog, binding, role):
        from .legacy_acceptance import verify_original_assets
        original = verify_original_assets(catalog.assets, tuple(catalog.policies[binding.domain].values()))
        self.catalog, self.binding, self.role = catalog, binding, role
        self.p = next(p for p in binding.programs if p.role == role)
        require(self.p.reader_schema == PROGRAM_SCHEMAS[binding.domain]
            and self.p.audit_scope == 'carried_selected_actor_development_audit'
            and self.p.archive_sha256 == self.p.source_program_sha256
            and self.p.independent_audit_sha256 == ORIGINAL_ACCEPTANCE_SHA[binding.domain], 'legacy program provenance differs')
        self.payload = decode(catalog.assets.path(self.p.source_program_sha256).read_bytes())
        self.program = ExecutableProgram.from_dict(self.payload['program'] if binding.domain == 'pong' else self.payload)
        self.metrics = deepcopy(original['selected_actor_development_program_audit'][role] if binding.domain == 'pong'
            else original['final_role_program_development_audit'][role])
        require(self.program.root.depth() <= 5 and self.program.root.leaf_count() <= 16, 'legacy bounded program complexity differs')

    def evidence(self, native, executed_NN_action, *, current_actor_role_sha256):
        require(self.binding == self.catalog.binding(self.binding.domain, self.binding.variant)
            and current_actor_role_sha256 == self.binding.role_hash(self.role), 'current legacy evidence target differs')
        self.catalog.assets.path(self.p.source_program_sha256); self.catalog.assets.path(self.p.independent_audit_sha256)
        features = (pong_nn.semantic_features(native, self.role) if self.binding.domain == 'pong'
                    else kitchen_nn.program_features(native, self.role))
        result = self.program.execute(features)
        if result.action != executed_NN_action: return None
        return dict(eligible=True, program_sha256=self.p.source_program_sha256,
            target_actor_role_sha256=current_actor_role_sha256, nn_action=executed_NN_action,
            audited_metrics=deepcopy(self.metrics), path=[dict(feature=s.feature, operator=s.operator,
                threshold=s.threshold, observed_value=s.observed_value, result=s.result) for s in result.trace.tree_steps],
            historical_acceptance_sha256=self.p.independent_audit_sha256,
            evidence_scope='existing_v5_selected_actor_development_audit', new_independent_audit_claimed=False,
            unique_causal_motive_established=False, external_program_selects_execution_action=False)


class LegacyDomainAdapter:
    def __init__(self, catalog, *, domain, runtime_config_sha256):
        require(type(catalog) is PolicyCatalog and domain in CONFIG_SHA and runtime_config_sha256 == CONFIG_SHA[domain],
            'explicit carried catalog/domain/config required')
        self.catalog, self.DOMAIN, self.config_sha, self._cores = catalog, domain, runtime_config_sha256, {}

    def _core(self, binding):
        from .legacy_acceptance import verify_original_assets
        require(type(binding) is PolicyBinding and binding.domain == self.DOMAIN
            and binding == self.catalog.binding(self.DOMAIN, binding.variant), 'unregistered carried-domain binding')
        verify_original_assets(self.catalog.assets, tuple(self.catalog.policies[self.DOMAIN].values()))
        if binding.identity_sha256 not in self._cores:
            self._cores[binding.identity_sha256] = FrozenLegacyCore(self.catalog.assets, binding, self.config_sha)
        core = self._cores[binding.identity_sha256]; core.verify(); return core

    def _wrap(self, native, binding, provenance):
        self._core(binding).validate_native(native)
        state = dict(version=VERSION, domain=self.DOMAIN, binding=binding.to_dict(),
            binding_identity_sha256=binding.identity_sha256, task=native['task'], turn=native['turn'],
            max_turns=native['max_turns'], terminal=native['terminal'], snapshot=deepcopy(native),
            snapshot_sha256=digest(native), provenance=deepcopy(provenance))
        state['state_sha256'] = digest(state); return state

    def recorded_state(self, native_snapshot, binding, *, provenance):
        require(type(provenance) is dict and provenance, 'complete recorded source provenance required')
        return self._wrap(native_snapshot, binding, provenance)

    def initial_state(self, seed, task, binding):
        core = self._core(binding)
        require(type(seed) is int and 0 <= seed < 2**63 and type(task) is int and task in (1, 2, 3), 'invalid explicit runtime scene')
        if self.DOMAIN == 'pong':
            human, ai = (6, 2) if task == 3 else (2, 6)
            native = pong_physics._new_state(pong_physics._schedule(seed, task, core.config), seed=seed,
                task=task, human=human, ai=ai, config=core.config)
        else:
            # Only a fully saved scene in the pinned config is accepted. No
            # fallback through the old module-global configuration/generator.
            scene = core.config.get('scenarios', {}).get(str(seed), {}).get(str(task))
            require(type(scene) is dict and scene.get('menu_version') == kitchen_physics.MENU_VERSION,
                'Kitchen scene is not present in the pinned scenario package')
            native = _saved_kitchen_scene(seed, task, scene)
        return self._wrap(native, binding, dict(kind='explicit_pinned_runtime_scene'))

    def restore(self, state, binding):
        core = self._core(binding)
        require(type(state) is dict and set(state) == STATE_FIELDS and state['version'] == VERSION
            and state['domain'] == self.DOMAIN and state['binding'] == binding.to_dict()
            and state['binding_identity_sha256'] == binding.identity_sha256
            and digest({k: v for k, v in state.items() if k != 'state_sha256'}) == state['state_sha256']
            and digest(state['snapshot']) == state['snapshot_sha256'], 'carried saved binding or state integrity differs')
        native = core.validate_native(deepcopy(state['snapshot']))
        require(all(state[k] == native[k] for k in ('task', 'turn', 'max_turns', 'terminal')), 'carried snapshot/frame boundary differs')
        return native

    def decide(self, state, binding):
        native = self.restore(state, binding); choices, scores, masks = self._core(binding).choose(native)
        return dict(action=choices['ai'], human_action=choices['human'], policy_actions=choices,
            scores=scores, action_masks=masks, state_sha256=state['state_sha256'],
            binding_identity_sha256=binding.identity_sha256, ai_action_overrides=0,
            execution_source='bound_frozen_NN_with_physical_mask', unique_causal_motive_established=False)

    def _advance(self, state, binding, human_action, decision=None):
        actual = self.decide(state, binding)
        require(decision is None or decision == actual, 'carried decision differs from bound NN frame')
        native = self.restore(state, binding); actions = dict(actual['policy_actions'])
        if human_action is not None:
            require(type(human_action) is str, 'explicit human command required')
            legal = pong_physics.legal_actions(native) if self.DOMAIN == 'pong' else kitchen_physics.legal_actions(native)
            require(human_action in legal, 'illegal carried human command'); actions['human'] = human_action
        nxt = self._core(binding).transition(native, actions)
        require(actions['ai'] == actual['action'], 'AI action changed before physics')
        wrapped = self._wrap(nxt, binding, state['provenance'])
        return wrapped, dict(human_action=human_action, decision=actual, state=wrapped,
            selected_actions=actions, ai_action_overrides=0)

    def step(self, state, human_action, binding, *, decision=None):
        require(human_action is not None, 'participant step requires an explicit human replacement')
        return self._advance(state, binding, human_action, decision)[0]

    def demonstration(self, binding, *, state=None, seed=None, task=1, steps=120):
        require((state is None) != (seed is None) and type(steps) is int and 0 < steps <= 360, 'invalid explicit NN-NN demonstration')
        current = self.initial_state(seed, task, binding) if state is None else deepcopy(state)
        self.restore(current, binding); initial = deepcopy(current); records = []
        for _ in range(steps):
            if current['terminal']: break
            current, record = self._advance(current, binding, None); records.append(record)
        return dict(version=REPLAY_VERSION, binding=binding.to_dict(), initial_state=initial,
            records=records, final_state=current, two_frozen_NN=True)

    def replay(self, bundle, binding):
        require(type(bundle) is dict and bundle.get('version') == REPLAY_VERSION and bundle.get('binding') == binding.to_dict()
            and type(bundle.get('records')) is list and type(bundle.get('two_frozen_NN')) is bool, 'unbound carried replay')
        current = deepcopy(bundle['initial_state']); self.restore(current, binding)
        for record in bundle['records']:
            require(not bundle['two_frozen_NN'] or record['human_action'] is None, 'NN demo contains a human replacement')
            current, actual = self._advance(current, binding, record['human_action'], record['decision'])
            require(actual == record, 'carried NN/physics replay differs')
        require(current == bundle['final_state'], 'carried replay final frame differs'); return current

    def counterfactual(self, state, human_actions, binding):
        require(type(human_actions) in (list, tuple) and 0 < len(human_actions) <= 12, 'counterfactual needs 1 to 12 commands')
        current = deepcopy(state); self.restore(current, binding); records = []
        for command in human_actions:
            if current['terminal']: break
            require(command is not None, 'counterfactual only replaces human')
            current, record = self._advance(current, binding, command); records.append(record)
        return dict(version=REPLAY_VERSION, binding=binding.to_dict(), initial_state=deepcopy(state),
            records=records, final_state=current, two_frozen_NN=False)

    def explanation(self, state, bound):
        require(type(bound) is BoundDomain and bound.adapter is self and bound.catalog is self.catalog
            and bound.group in GROUP_VARIANTS and GROUP_VARIANTS[bound.group] == bound.binding.variant
            and bound.mode in ('preview', 'test', 'pilot'), 'invalid explicit explanation handle')
        binding = bound.verify(); native = self.restore(state, binding)
        if state['terminal']: return None
        reader = bound.explanation_program(task=state['task'], role='ai')
        return None if reader is None else reader.evidence(native, self.decide(state, binding)['action'],
            current_actor_role_sha256=binding.ai_actor_sha256)

    def public_state(self, state, binding):
        native = self.restore(state, binding)
        public = pong_physics.public_state(native) if self.DOMAIN == 'pong' else kitchen_physics.public_state(native)
        if self.DOMAIN == 'kitchen':
            public.pop('handoff_grace', None)
            public['rule_metadata'].pop('handoff_grace_turns', None)
            public['ai']['current_cooking'] = [r for r in public['ai']['current_cooking'] if r.get('location') != 'plan']
            public['ai']['policy'] = dict(kind='frozen_neural_actor', version=kitchen_nn.MODEL_VERSION,
                policy_id=binding.policy_id, actor_sha256=binding.actor_sha256)
        public.update(binding=binding.to_dict(), binding_identity_sha256=binding.identity_sha256,
            state_sha256=state['state_sha256'], policy_execution='bound_frozen_NN')
        return public

    @staticmethod
    def neutral_practice_metadata():
        return dict(kind='neutral_kitchen_teaching_practice', model_evaluation=False, scientific_model_evidence=False,
            labelled_exception=True, frozen_teammate_experiment=False)


def _saved_kitchen_scene(seed, task, scene):
    """Data-only initialization from a pinned saved menu, without RNG/lookup."""
    e = kitchen_physics
    return dict(domain='kitchen', version=e.VERSION, scenario_version=e.SCENARIO_VERSION, menu_version=e.MENU_VERSION,
        rule_metadata=e.rule_metadata(), seed=seed, task=task, turn=0, max_turns=scene['max_turns'], terminal=False,
        termination_reason=None, events=[], policy_memory={},
        human=dict(x=scene['human_start'][0], y=scene['human_start'][1], facing='right', holding=None),
        ai=dict(x=scene['ai_start'][0], y=scene['ai_start'][1], facing='right', holding=None),
        handoff=None, buffers=dict(human=None, ai_raw=[None, None], protein=dict(pot1=None, pot2=None)),
        pots=[dict(id=name, x=e.STATION_BY_ID[name]['x'], y=e.STATION_BY_ID[name]['y'], status='empty', phase='idle',
            recipe=None, order_id=None, item=None, remaining=0, ready_age=0) for name in ('pot1', 'pot2')],
        orders=[dict(order, status='pending', served_turn=None) for order in scene['orders'] if order['arrival'] == 0],
        _future_orders=[deepcopy(o) for o in scene['orders'] if o['arrival'] > 0], total_orders=len(scene['orders']),
        next_item_id=1, raw_score=0, metrics=dict(completed_orders=0, burnt=0, expired=0, waste=0, spoiled=0,
            discarded_ingredients=0, discarded_dishes=0, step_penalty=0, discard_penalty=0, blocked_waits=0,
            human_waits=0, handoff_conflicts=0, parallel_cooking_turns=0, parallel_recipe_turns=0, wrong_order_buffered=0))


class PongAdapter(LegacyDomainAdapter):
    DOMAIN = 'pong'
    def __init__(self, catalog, *, runtime_config_sha256=CONFIG_SHA['pong']):
        super().__init__(catalog, domain=self.DOMAIN, runtime_config_sha256=runtime_config_sha256)


class KitchenAdapter(LegacyDomainAdapter):
    DOMAIN = 'kitchen'
    def __init__(self, catalog, *, runtime_config_sha256=CONFIG_SHA['kitchen']):
        super().__init__(catalog, domain=self.DOMAIN, runtime_config_sha256=runtime_config_sha256)
