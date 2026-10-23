"""Physics, public observations and offline labels. No learned-policy decisions.

Only the factory dispatches on domain. The training/model/program algorithms do
not. Teacher calls are isolated in label_public(), never in step().
"""
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
from .contracts import DomainSpec, Feature, RoleSpec, digest, file_sha


def _source_hash(paths, config):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return digest(dict(files={p: file_sha(root / p) for p in paths}, config=config))


def validate_observation(role, x, mask, action_features):
    x, mask, af = np.asarray(x, np.float32), np.asarray(mask, bool), np.asarray(action_features, np.float32)
    if x.shape != (len(role.features),) or mask.shape != (len(role.actions),) or af.shape != (len(role.actions), role.action_feature_dim):
        raise ValueError('domain public feature shape differs')
    lo = np.array([f.lower for f in role.features], np.float32)
    hi = np.array([f.upper for f in role.features], np.float32)
    if (not np.isfinite(x).all() or not np.isfinite(af).all() or (x < lo).any() or (x > hi).any()
            or not mask.any() or not mask[role.wait_index]
            or not np.array_equal(x[list(role.legal_indices)], mask.astype(np.float32))):
        raise ValueError('public feature outside proved physical domain or mask differs')
    return dict(x=x, mask=mask, action_features=af)


class WarehouseDomain:
    def frame(self, state): return int(state.state.frame)
    def __init__(self):
        from backend.training.warehouse_v626_domain import domain_table
        from env.warehouse.domain import participant_study_config
        from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
        from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
        from env.warehouse.public_action_features_v617 import ACTION_FEATURE_DIM
        rows = domain_table()['features']
        features = tuple(Feature(r['name'], 1. if r['name'] == 'legal_wait' else r['lower'],
                        1. if r['name'] == 'legal_wait' else r['upper'],
                        ('same_target_collision_streak_capped', 'self_charger_return_margin')
                        if r['name'] == 'public_hard_risk_context' else (r['name'],),
                        'nonterminal_wait_always_legal' if r['name'] == 'legal_wait' else r['reason']) for r in rows)
        names = [f.name for f in features]
        roles = tuple(RoleSpec(name, tuple(ACTIONS), features,
            tuple(names.index('legal_' + a.lower()) for a in ACTIONS),
            tuple(names.index(n) for n in ('self_row', 'self_column', 'partner_row', 'partner_column')),
            tuple((MOVE_DELTAS.get(a, (0, 0))[0] / (layout.rows - 1),
                   MOVE_DELTAS.get(a, (0, 0))[1] / (layout.cols - 1)) for a in ACTIONS),
            ACTION_FEATURE_DIM, ACTIONS.index('WAIT')) for name in ('robot_1', 'robot_2'))
        self.config = participant_study_config()
        self.spec = DomainSpec('warehouse', roles, _source_hash([
            'env/warehouse/environment.py', 'env/warehouse/domain.py', 'env/warehouse/rewards.py',
            'env/warehouse/navigation.py', 'env/warehouse/public_observation_v6.py',
            'env/warehouse/public_action_features_v614.py', 'env/warehouse/public_action_features_v617.py'], asdict(self.config)),
            'complete_episode_shutdown_count_gt_zero', 'original_environment_per_role_reward', self.config.horizon)

    def initial_state(self, seed, task):
        from env.warehouse.environment import WarehouseMultiAgentEnv
        env = WarehouseMultiAgentEnv(self.config); env.reset(seed=seed)
        env.state.participant_controlled_agent_id = 'robot_1'
        return env

    def public(self, env):
        from env.warehouse.public_observation_v6 import public_pre_action_snapshot
        return public_pre_action_snapshot(env.state)

    def observe_public(self, snapshot):
        from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state, encode_public_observation
        from env.warehouse.pure_nn import action_masks
        from env.warehouse.public_action_features_v617 import encode_actor_tree_features, action_features_from_tree_encoded
        state = rebuild_public_pre_action_state(snapshot)
        masks = action_masks(state)
        x = encode_actor_tree_features(encode_public_observation(state), masks)
        af = action_features_from_tree_encoded(x)
        return {r.name: validate_observation(r, x[i], masks[i], af[i]) for i, r in enumerate(self.spec.roles)}

    def label_public(self, snapshot):
        from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state, encode_public_observation
        from env.warehouse.pure_nn import action_masks
        from backend.training.warehouse_v6_teacher_lookahead import public_lookahead_teacher
        state = rebuild_public_pre_action_state(snapshot)
        labels = public_lookahead_teacher(encode_public_observation(state), action_masks(state))
        return {r.name: int(labels[i]) for i, r in enumerate(self.spec.roles)}

    def critic_vector(self, snapshot):
        obs = self.observe_public(snapshot)
        return np.concatenate([obs[r.name]['x'] for r in self.spec.roles])

    def step(self, env, indices):
        actions = {r.name: r.actions[indices[r.name]] for r in self.spec.roles}
        _, reward, _, _, info = env.step(actions)
        if info['proposed_actions'] != actions or info['action_override_agents']:
            raise RuntimeError('Warehouse controller overrode NN actions')
        return env, np.array([reward[r.name] for r in self.spec.roles], np.float32)

    def terminal(self, env):
        return bool(env.state.terminated or env.state.truncated)

    def ending(self, env):
        return bool(env.state.terminated), bool(env.state.truncated)

    def outcome(self, env, initial=None):
        if not self.terminal(env):
            raise ValueError('partial episode has no binary failure label')
        return dict(deliveries=int(env.state.total_deliveries), collisions=int(env.state.robot_collision_events),
                    failed=bool(env.state.shutdown_count), frames=int(env.state.frame))

    def snapshot(self, env):
        from backend.training.warehouse_v629_environment_replay import capture_environment
        return capture_environment(env)

    def restore(self, snapshot):
        from backend.training.warehouse_v629_environment_replay import restore_environment
        env = restore_environment(deepcopy(snapshot))
        if env.config != self.config or env.state.participant_controlled_agent_id != 'robot_1':
            raise ValueError('Warehouse snapshot physical mode differs')
        return env


class PongDomain:
    def frame(self, state): return int(state['turn'])
    def __init__(self):
        from domains.pong import nn_policy as p, turnbased as e
        self.p, self.e = p, e
        features = []
        for name in p.FEATURE_NAMES:
            lower, upper, reason = 0., 1., 'public_lane_or_indicator'
            if name.endswith('_gap'):
                lower, upper, reason = -1., 1., 'difference_of_two_lanes_divided_by_8'
            elif name.endswith('_slack'):
                lower, upper, reason = -8 / 12, 1., 'arrival_0_12_minus_lane_distance_0_8'
            features.append(Feature(name, float(np.nextafter(np.float32(lower), np.float32(-np.inf))),
                                    float(np.nextafter(np.float32(upper), np.float32(np.inf))), (name,), reason))
        features += [Feature('legal_' + a, 1. if a == 'wait' else 0., 1., ('legal_' + a,),
                            'nonterminal_wait_always_legal' if a == 'wait' else 'physical_action_mask') for a in p.ACTIONS]
        features.append(Feature('vertical_coordinate', 0., 0., ('vertical_coordinate',), 'one_dimensional_paddle'))
        roles = tuple(RoleSpec(n, p.ACTIONS, tuple(features), tuple(range(p.FEATURE_SIZE, p.FEATURE_SIZE + 3)),
                               (0, p.FEATURE_SIZE + 3, 1, p.FEATURE_SIZE + 3),
                               ((0., 0.), (-1 / 8, 0.), (1 / 8, 0.)), 0, 0) for n in ('human', 'ai'))
        # A constant vertical coordinate makes the same two-dimensional geometry
        # calculation applicable to a one-dimensional physical paddle.
        self.spec = DomainSpec('pong', roles, _source_hash(['domains/pong/turnbased.py', 'domains/pong/nn_policy.py'], e.CONFIG),
            'complete_capacity_lt_0.80_or_AI_wait_gt_0.20', 'original_raw_score_delta',
            int(max(e.CONFIG['task_turns'].values())))

    def initial_state(self, seed, task):
        return self.e.initial_state(seed, task)

    def public(self, state):
        # Deliberately exclude hidden future schedule, seed and policy memory.
        return {k: deepcopy(state[k]) for k in ('human', 'ai', 'balls', 'turn', 'max_turns', 'task', 'terminal', 'raw_score')}

    def observe_public(self, state):
        result = {}
        for role in self.spec.roles:
            mask = self.p.legal_mask(state, role.name)
            result[role.name] = validate_observation(role, [*self.p.features(state, role.name), *mask, 0.],
                                                   mask, np.empty((3, 0), np.float32))
        return result

    def label_public(self, state):
        visible = {**deepcopy(state), 'policy_memory': {}}
        decision = self.e._decision_from_observation(self.e._observation(visible))
        return {'human': self.p.ACTIONS.index(decision['human_action']), 'ai': self.p.ACTIONS.index(decision['action'])}

    def critic_vector(self, snapshot):
        obs = self.observe_public(snapshot)
        return np.concatenate([obs[r.name]['x'] for r in self.spec.roles])

    def step(self, state, indices):
        result = self.e.step_pair(state, self.p.ACTIONS[indices['human']], self.p.ACTIONS[indices['ai']])
        return result, np.repeat(np.float32(result['raw_score'] - state['raw_score']), 2)

    def terminal(self, state):
        return state['terminal']

    def ending(self, state):
        # The native engine ends only when the task's turn budget expires.
        # Keep this distinct from an absorbing physical terminal for GAE.
        return False, bool(state['terminal'])

    def outcome(self, state, initial):
        if not self.terminal(state):
            raise ValueError('partial episode has no binary failure label')
        from scripts.train_rolling_pong_nn import contact_capacity
        capacity = contact_capacity(initial)
        if capacity <= 0 or state['turn'] <= 0:
            raise ValueError('undefined contact capacity or zero-turn episode')
        ratio, wait = state['raw_score'] / capacity, state['metrics']['waiting_turns'] / state['turn']
        return dict(capacity=ratio, wait=wait, capacity_numerator=state['raw_score'], capacity_denominator=capacity,
                    wait_numerator=state['metrics']['waiting_turns'], wait_denominator=state['turn'],
                    failed=5*state['raw_score'] < 4*capacity or 5*state['metrics']['waiting_turns'] > state['turn'], frames=state['turn'])

    def snapshot(self, state):
        return deepcopy(state)

    def restore(self, snapshot):
        return deepcopy(snapshot)


def _kitchen_features(p, horizon):
    features = []
    def add(name, lo=0., hi=1., reason='public_indicator_or_fraction'):
        features.append(Feature(name, float(np.nextafter(np.float32(lo), np.float32(-np.inf))),
                         float(np.nextafter(np.float32(hi), np.float32(np.inf))), (name,), reason))
    add('turn', hi=horizon / 360, reason='physical_horizon_divided_by_360')
    add('max_turns', hi=horizon / 360, reason='physical_horizon_divided_by_360')
    add('score', -horizon * 41 / 500, 1., 'at_most_two_discards_plus_one_step_cost_per_turn_and_five_serves')
    for task in (1, 2, 3): add(f'task_{task}')
    for order in range(1, 6):
        for field in ('present', *('recipe_' + r for r in p.e.RECIPES), 'pending', 'completed', 'expired', 'deadline'):
            add(f'order{order}_{field}')
    for role in ('human', 'ai'):
        add(role + '_x'); add(role + '_y')
        for facing in p.FACES: add(role + '_facing_' + facing)
    locations = ('human_item', 'ai_item', 'handoff_item', 'human_buffer', 'ai_raw0', 'ai_raw1',
                 'protein1', 'protein2', 'pot1_item', 'pot2_item')
    for location in locations:
        names = ('present', *('ingredient_' + n for n in p.e.LABELS), *('recipe_' + n for n in p.e.RECIPES),
                 *('stage_' + n for n in p.STAGES), *('order_' + str(i) for i in range(1, 6)),
                 'pot1', 'pot2', *('container_' + n for n in p.CONTAINERS), 'prepare_progress',
                 'freshness', 'storage_age', 'was_buffered', 'component_count', 'acquired_age')
        if len(names) != 33: raise RuntimeError('Kitchen item schema differs')
        for name in names:
            add(location + '_' + name, -1. if name == 'freshness' else 0.)
    for pot in ('pot1', 'pot2'):
        for name in (*('status_' + n for n in p.STATUSES), *('phase_' + n for n in p.PHASES),
                     *('order_' + str(i) for i in range(1, 6)), 'remaining', 'ready_age'):
            add(pot + '_' + name)
    return features


class KitchenDomain:
    def frame(self, state): return int(state['turn'])
    def __init__(self):
        from domains.kitchen import neural_policy as p, engine_v5_frozen as e
        self.p, self.e = p, e
        config = e._configuration(); horizon = max([360, *config.get('task_budgets', {}).values()])
        base = _kitchen_features(p, horizon); names = [f.name for f in base]
        positions = tuple(names.index(n) for n in ('human_x', 'human_y', 'ai_x', 'ai_y'))
        roles = []
        for role, actions in (('human', p.HUMAN_ACTIONS), ('ai', p.AI_ACTIONS)):
            features = (*base, *(Feature('legal_' + a, 1. if a == 'wait' else 0., 1., ('legal_' + a,),
                'nonterminal_wait_always_legal' if a == 'wait' else 'physical_action_mask') for a in actions))
            pos = positions if role == 'human' else (*positions[2:], *positions[:2])
            roles.append(RoleSpec(role, actions, features, tuple(range(len(base), len(features))), pos,
                tuple((e.MOVES.get(a, (0, 0))[0] / 8., e.MOVES.get(a, (0, 0))[1] / 6.) for a in actions), 0, actions.index('wait')))
        self.spec = DomainSpec('kitchen', tuple(roles), _source_hash([
            'domains/kitchen/engine_v5_frozen.py', 'domains/kitchen/neural_policy.py'], config),
            'complete_episode_completed_orders_ne_5', 'original_raw_score_delta', horizon)

    def initial_state(self, seed, task):
        return self.e.initial_state(seed, task)

    def public(self, state):
        # Explicit whitelist: no future orders, seed, teacher commitment or memory.
        keys = ('human', 'ai', 'orders', 'task', 'turn', 'max_turns', 'raw_score', 'terminal',
                'handoff', 'buffers', 'pots')
        return {k: deepcopy(state[k]) for k in keys}

    def observe_public(self, state):
        vector = self.p.features(state)
        result = {}
        for role in self.spec.roles:
            mask = self.p.legal_mask(state, role.name)
            result[role.name] = validate_observation(role, np.concatenate((vector, mask)), mask,
                                                   np.empty((len(role.actions), 0), np.float32))
        return result

    def label_public(self, snapshot):
        # Teachers receive a reconstructed visible world, never the native
        # hidden future order schedule or their previous commitment.
        visible = self.teacher_world(snapshot)
        decision = self.e.decide(visible)
        from domains.kitchen.training import _teacher_ai_action
        human = self.e.simulation_partner(visible)
        ai = _teacher_ai_action(visible, decision)
        result = {'human': self.p.HUMAN_ACTIONS.index(human), 'ai': self.p.AI_ACTIONS.index(ai)}
        for role in self.spec.roles:
            if not self.p.legal_mask(visible, role.name)[result[role.name]]:
                raise ValueError('public Kitchen teacher selected an illegal action')
        return result

    def critic_vector(self, snapshot):
        vector = self.p.features(snapshot)
        return np.concatenate([np.concatenate((vector, self.p.legal_mask(snapshot, r.name))).astype(np.float32)
                               for r in self.spec.roles])

    def proxy(self, state):
        visible = self.teacher_world(self.public(state))
        return self.p.HUMAN_ACTIONS.index(self.e.simulation_partner(visible))

    def teacher_world(self, snapshot):
        visible = deepcopy(snapshot)
        counters = ('completed_orders', 'burnt', 'expired', 'waste', 'spoiled', 'discarded_ingredients',
                    'discarded_dishes', 'step_penalty', 'discard_penalty', 'blocked_waits', 'human_waits',
                    'handoff_conflicts', 'parallel_cooking_turns', 'parallel_recipe_turns', 'wrong_order_buffered')
        # Simulation bookkeeping is reconstructed, not read from private state.
        ids = [int(i['id'][4:]) for i in self.e._all_items(visible)
               if i and str(i.get('id', '')).startswith('item') and str(i['id'])[4:].isdigit()]
        visible.update(policy_memory={}, _future_orders=[], events=[], metrics=dict.fromkeys(counters, 0),
                       termination_reason=None, next_item_id=max(ids, default=0) + 1, total_orders=5)
        return visible

    def step(self, state, indices):
        human, ai = self.p.HUMAN_ACTIONS[indices['human']], self.p.AI_ACTIONS[indices['ai']]
        decision = {**self.p.action_spec(ai), 'policy_origin': 'kitchen_nn', 'memory': {}, 'reason_code': 'neural_v7'}
        result = self.e.step(state, human, decision)
        return result, np.repeat(np.float32(result['raw_score'] - state['raw_score']), 2)

    def terminal(self, state):
        return state['terminal']

    def ending(self, state):
        return (bool(state['terminal'] and state['termination_reason'] != 'turn_budget'),
                bool(state['terminal'] and state['termination_reason'] == 'turn_budget'))

    def outcome(self, state, initial=None):
        if not self.terminal(state):
            raise ValueError('partial episode has no binary failure label')
        return dict(completed=int(state['metrics']['completed_orders']),
                    failed=state['metrics']['completed_orders'] != 5, frames=state['turn'])

    def snapshot(self, state):
        return deepcopy(state)

    def restore(self, snapshot):
        return deepcopy(snapshot)


def domain(name):
    return {'warehouse': WarehouseDomain, 'pong': PongDomain, 'kitchen': KitchenDomain}[name]()
