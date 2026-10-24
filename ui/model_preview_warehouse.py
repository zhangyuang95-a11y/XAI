# Carried runtime_v642.warehouse methods are unchanged; only imports are made inference-only.
"""Explicit Warehouse dependencies, with no policy selection or action controller.

This module is a software adapter, not a scientific acceptance receipt. Fresh
resets occur only in initial_state; restoration and all replay paths allocate
from a complete recorded physics/RNG snapshot without calling reset.
"""
from copy import deepcopy

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, MODEL_VERSION, FEATURE_SCHEMA_SHA256,
)
from .model_preview_snapshot import capture_environment, restore_environment
from . import model_preview_views as views
from env.warehouse.decision_protocol import distribution_decision_metadata, serialized_action_distributions
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v617 import (
    ACTION_PHYSICS_CONFIG_SHA256, encode_actor_tree_features, validate_study_action_physics,
)
from env.warehouse.public_observation_v6 import encode_public_observation, public_pre_action_snapshot
from env.warehouse.pure_nn import AGENTS, action_masks

from runtime_v642.catalog import PolicyCatalog
from runtime_v642.contracts import GROUP_VARIANTS, PolicyBinding, digest, require
from runtime_v642.registry import BoundDomain

VERSION = 'runtime-v642-warehouse-state.v1'
DECISION_VERSION = 'runtime-v642-warehouse-NN-decision.v1'
REPLAY_VERSION = 'runtime-v642-warehouse-recorded-transitions.v1'
STATE_FIELDS = frozenset(('version', 'domain', 'binding', 'binding_identity_sha256',
    'task', 'seed', 'scenario_seed', 'action_seed', 'turn', 'max_turns', 'terminal',
    'snapshot', 'public_history', 'events', 'charge_events', 'provenance', 'state_sha256'))


def _seed(value):
    require(type(value) is int and 0 <= value < 2**63, 'invalid explicit Warehouse seed')
    return value


def _history(s):
    previous = None if s.frame == 0 else {
        role: {'proposed': s.by_id(agent).last_action.lower(),
               'executed': s.by_id(agent).last_executed_action.lower()}
        for role, agent in (('human', 'robot_1'), ('ai', 'robot_2'))}
    return dict(previous_joint_actions=previous,
        previous_collision_kind=s.last_robot_collision_kind if s.frame else None,
        same_target_collision_streak=s.same_target_collision_streak)


class WarehouseAdapter:
    DOMAIN = 'warehouse'

    def __init__(self, catalog):
        require(type(catalog) is PolicyCatalog, 'explicit immutable PolicyCatalog required')
        self.catalog = catalog
        self._actors = {}

    def _binding(self, binding):
        require(type(binding) is PolicyBinding and binding.domain == self.DOMAIN
            and binding == self.catalog.binding(self.DOMAIN, binding.variant), 'unregistered Warehouse binding')
        require(binding.model_schema == MODEL_VERSION and binding.feature_schema_sha256 == FEATURE_SCHEMA_SHA256
            and binding.action_physics_config_sha256 == ACTION_PHYSICS_CONFIG_SHA256,
            'Warehouse model, public feature or physical contract differs')
        return binding

    def _actor(self, binding):
        self._binding(binding)
        # Rehash the package before any reuse of an in-memory frozen Actor.
        path = self.catalog.assets.path(binding.actor_sha256)
        key = (binding.actor_sha256, binding.human_actor_sha256, binding.ai_actor_sha256)
        if key not in self._actors:
            actor = FrozenWarehouseV6Actor.load(path, expected_sha256=binding.actor_sha256,
                expected_role_sha256=dict(robot_1=binding.human_actor_sha256, robot_2=binding.ai_actor_sha256))
            require(actor.model_version == binding.model_schema, 'loaded Actor schema differs')
            for role in actor.roles.values():
                for value in role.weights.values():
                    value.flags.writeable = False
            self._actors[key] = actor
        return self._actors[key]

    def _state(self, env, binding, *, task, seed, scenario_seed, action_seed, events=(), charge_events=0, provenance):
        self._actor(binding)
        require(task in (1, 2, 3) and type(task) is int, 'task must be 1, 2 or 3')
        for value in (seed, scenario_seed, action_seed): _seed(value)
        require(env.state.participant_controlled_agent_id == 'robot_1'
            and validate_study_action_physics(env.config) == binding.action_physics_config_sha256,
            'physical environment must use direct-NN participant mode')
        result = dict(version=VERSION, domain=self.DOMAIN, binding=binding.to_dict(),
            binding_identity_sha256=binding.identity_sha256, task=task, seed=seed,
            scenario_seed=scenario_seed, action_seed=action_seed, turn=env.state.frame,
            max_turns=env.config.horizon, terminal=bool(env.state.terminated or env.state.truncated),
            snapshot=capture_environment(env), public_history=_history(env.state),
            events=views._plain(list(events)), charge_events=charge_events, provenance=deepcopy(provenance))
        result['state_sha256'] = digest(result)
        return result

    def initial_state(self, seed, task, binding):
        """Future explicit runtime reset; never invoked by recorded-frame tests."""
        self._actor(binding)
        _seed(seed)
        require(type(task) is int and task in (1, 2, 3), 'task must be 1, 2 or 3')
        scenario_seed = _seed(seed * 10 + task)
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=scenario_seed)
        # This environment flag disables its legacy action-selection controller.
        env.state.participant_controlled_agent_id = 'robot_1'
        return self._state(env, binding, task=task, seed=seed, scenario_seed=scenario_seed,
            action_seed=scenario_seed, provenance=dict(kind='explicit_runtime_reset'))

    def recorded_state(self, snapshot, binding, *, task, scenario_seed, action_seed, provenance):
        """Import a complete spent snapshot, retaining its original action stream.

        Callers must supply their source provenance. This is restoration only,
        and neither establishes scientific seed authorization nor grants pilot.
        """
        require(type(provenance) is dict and bool(provenance), 'recorded source provenance required')
        env = restore_environment(deepcopy(snapshot))
        return self._state(env, binding, task=task, seed=scenario_seed, scenario_seed=scenario_seed,
            action_seed=action_seed, provenance=dict(kind='recorded_snapshot', source=deepcopy(provenance)))

    def restore(self, state, binding):
        self._actor(binding)
        require(type(state) is dict and set(state) == STATE_FIELDS and state['version'] == VERSION
            and state['domain'] == self.DOMAIN and state['binding'] == binding.to_dict()
            and state['binding_identity_sha256'] == binding.identity_sha256,
            'Warehouse saved state and immutable binding differ')
        require(digest({k: v for k, v in state.items() if k != 'state_sha256'}) == state['state_sha256'],
            'Warehouse saved state integrity differs')
        require(type(state['task']) is int and state['task'] in (1, 2, 3)
            and type(state['turn']) is int and type(state['terminal']) is bool
            and type(state['charge_events']) is int and state['charge_events'] >= 0
            and type(state['events']) is list and type(state['provenance']) is dict,
            'Warehouse saved state boundary types differ')
        for name in ('seed', 'scenario_seed', 'action_seed'): _seed(state[name])
        env = restore_environment(deepcopy(state['snapshot']))
        require(env.state.participant_controlled_agent_id == 'robot_1'
            and validate_study_action_physics(env.config) == binding.action_physics_config_sha256
            and env.state.frame == state['turn'] and env.config.horizon == state['max_turns']
            and bool(env.state.terminated or env.state.truncated) == state['terminal']
            and _history(env.state) == state['public_history'], 'Warehouse snapshot, RNG, frame or public history differs')
        return env

    def _selection(self, env, state, binding):
        actions, distributions = self._actor(binding).act(env.state, base_seed=state['action_seed'])
        masks = action_masks(env.state, env.config.map_layout_id)
        require(set(actions) == set(AGENTS) and all(actions[agent] in ACTIONS
            and masks[i, ACTIONS.index(actions[agent])] == 1 for i, agent in enumerate(AGENTS)),
            'bound NN selected an action outside physical mask')
        return actions, distributions

    def decide(self, state, binding):
        env = self.restore(state, binding)
        require(not state['terminal'], 'cannot decide after terminal frame')
        actions, distributions = self._selection(env, state, binding)
        return dict(version=DECISION_VERSION, binding_identity_sha256=binding.identity_sha256,
            accepted_receipt_sha256=binding.accepted_receipt_sha256, state_sha256=state['state_sha256'],
            action=actions['robot_2'].lower(), policy_actions=dict(actions),
            distributions=serialized_action_distributions(distributions),
            ai_action_overrides=0, execution_source='bound_frozen_NN_with_physical_mask',
            goal='not_inferred', unique_causal_motive_established=False)

    def _advance(self, state, binding, human_action, decision=None):
        actual = self.decide(state, binding)
        require(decision is None or actual == decision, 'decision does not match exact bound pre-action frame')
        require(human_action is None or type(human_action) is str and human_action in tuple(a.lower() for a in ACTIONS),
            'unknown human physical command')
        env = self.restore(state, binding)
        before = env.get_state()
        proposed, distributions = self._selection(env, state, binding)
        require(proposed == actual['policy_actions'], 'NN output changed between decide and step')
        selected = dict(proposed)
        if human_action is not None: selected['robot_1'] = human_action.upper()
        runtime = dict(binding_identity_sha256=binding.identity_sha256, policy_actions=proposed,
            selected_actions=selected, ai_action_overrides=0, selection_changed_policy=False,
            human_role_replacement=human_action is not None)
        _, rewards, terminated, truncated, info = env.step(selected, decision_metadata=distribution_decision_metadata(
            distributions, decision_source='bound_participant_plus_NN' if human_action is not None else 'bound_dual_NN_demo',
            participant_overrides={} if human_action is None else {'robot_1': selected['robot_1']},
            policy_actions=proposed, selected_actions=selected, runtime_decision=runtime))
        # Environment provenance counts the authorized human replacement as an
        # override of robot_1's unused NN proposal. It must never include AI.
        allowed_overrides = ['robot_1'] if selected['robot_1'] != proposed['robot_1'] else []
        require(info['proposed_actions'] == selected and info['controller_selected_actions'] == selected
            and list(info['action_override_agents']) == allowed_overrides,
            'an execution controller rewrote a proposed NN/participant action')
        events = views._events(before, env.state, info)
        next_state = self._state(env, binding, task=state['task'], seed=state['seed'],
            scenario_seed=state['scenario_seed'], action_seed=state['action_seed'], events=events,
            charge_events=state['charge_events'] + sum(event['type'] == 'charge' for event in events),
            provenance=state['provenance'])
        record = dict(human_action=human_action, decision=actual, state=next_state,
            proposed_actions=selected, nn_policy_actions=proposed, rewards=rewards,
            terminated=terminated, truncated=truncated, shutdowns=sorted(info['shutdowns']),
            action_override_agents=list(info['action_override_agents']),
            post_state_public=public_pre_action_snapshot(env.state))
        return next_state, record

    def step(self, state, human_action, binding, *, decision=None):
        require(human_action is not None, 'participant step must replace only robot_1 with an explicit human command')
        return self._advance(state, binding, human_action, decision)[0]

    def demonstration(self, binding, *, state=None, seed=None, task=1, steps=120):
        require(type(steps) is int and 0 < steps <= 120, 'demonstration step budget is invalid')
        require((state is None) != (seed is None), 'demo needs exactly one explicit seed or recorded state')
        current = self.initial_state(seed, task, binding) if state is None else deepcopy(state)
        self.restore(current, binding)
        initial = deepcopy(current); records = []
        for _ in range(steps):
            if current['terminal']: break
            current, record = self._advance(current, binding, None)
            records.append(record)
        return dict(version=REPLAY_VERSION, binding=binding.to_dict(), initial_state=initial,
            records=records, final_state=current, two_frozen_NN=True)

    def replay(self, replay, binding):
        require(type(replay) is dict and replay.get('version') == REPLAY_VERSION
            and replay.get('binding') == binding.to_dict() and type(replay.get('records')) is list
            and type(replay.get('two_frozen_NN')) is bool, 'unbound or malformed replay')
        current = deepcopy(replay['initial_state'])
        self.restore(current, binding)
        for record in replay['records']:
            require(type(record) is dict and (not replay['two_frozen_NN'] or record.get('human_action') is None),
                'NN-NN replay includes an unrecorded human takeover')
            current, actual = self._advance(current, binding, record.get('human_action'), record.get('decision'))
            require(actual == record, 'replay NN action, physical transition or RNG differs')
        require(current == replay['final_state'], 'replay final state differs')
        return current

    def counterfactual(self, state, human_actions, binding):
        require(type(human_actions) in (list, tuple) and 0 < len(human_actions) <= 12,
            'counterfactual requires 1 to 12 explicit human commands')
        current = deepcopy(state); self.restore(current, binding); records = []
        for command in human_actions:
            if current['terminal']: break
            require(command is not None, 'counterfactual replaces only the human role')
            current, record = self._advance(current, binding, command)
            records.append(record)
        return dict(version=REPLAY_VERSION, binding=binding.to_dict(), initial_state=deepcopy(state),
            records=records, final_state=current, two_frozen_NN=False)

    def explanation(self, state, bound):
        require(type(bound) is BoundDomain and bound.adapter is self and bound.catalog is self.catalog,
            'explanation requires this exact registry handle')
        require(bound.group in GROUP_VARIANTS and GROUP_VARIANTS[bound.group] == bound.binding.variant
            and bound.mode in ('test', 'preview', 'pilot'), 'invalid registry group or execution mode')
        binding = bound.verify(); env = self.restore(state, binding)
        if state['terminal']: return None
        reader = bound.explanation_program(task=state['task'], role='ai')
        if reader is None: return None
        decision = self.decide(state, binding)
        masks = action_masks(env.state, env.config.map_layout_id)
        features = encode_actor_tree_features(encode_public_observation(env.state), masks)
        return reader.evidence(features[1], masks[1], decision['action'],
            current_actor_role_sha256=binding.ai_actor_sha256)

    def public_state(self, state, binding):
        env = self.restore(state, binding); s = env.state
        return dict(domain=self.DOMAIN, version=VERSION, task=state['task'], turn=state['turn'],
            max_turns=state['max_turns'], terminal=state['terminal'], binding=binding.to_dict(),
            binding_identity_sha256=binding.identity_sha256, state_sha256=state['state_sha256'],
            human=views._actor_view(s.by_id('robot_1')), ai=views._actor_view(s.by_id('robot_2')),
            orders=views._orders(env), events=deepcopy(state['events']), **_history(s),
            deliveries=s.total_deliveries, collision_events=s.robot_collision_events,
            shutdown_events=s.shutdown_count, charge_events=state['charge_events'],
            width=STUDY_MAP_LAYOUT.cols, height=STUDY_MAP_LAYOUT.rows,
            map_id=STUDY_MAP_LAYOUT.layout_id,
            score=dict(task_score=float(s.user_score), breakdown=dict(s.score_breakdown)))
