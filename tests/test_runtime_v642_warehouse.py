"""Software replay of spent frames; no new reset, scenario or scientific gate."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import shutil

import pytest

from backend.training.warehouse_v6_actor import MODEL_VERSION, FEATURE_SCHEMA_SHA256
from backend.training.warehouse_v629_environment_replay import capture_environment
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v617 import ACTION_PHYSICS_CONFIG_SHA256
from env.warehouse.public_observation_v6 import public_pre_action_snapshot
from env.warehouse.pure_nn import action_masks
from runtime_v642.catalog import AssetResolver, DOMAINS, PolicyCatalog, VERSION as CATALOG_VERSION
from runtime_v642.contracts import PolicyBinding, ProgramBinding, RuntimeUnavailable, digest
from runtime_v642.registry import BoundDomain
from runtime_v642.warehouse import WarehouseAdapter

ROOT = Path(__file__).resolve().parents[1]
SPENT = ROOT / 'output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration'


def _json(path): return json.loads(Path(path).read_text())


@pytest.fixture(autouse=True)
def forbid_fresh_environments(monkeypatch):
    def forbidden(*a, **kw): raise AssertionError('software test must not create/reset any environment')
    monkeypatch.setattr(WarehouseMultiAgentEnv, '__init__', forbidden)
    monkeypatch.setattr(WarehouseMultiAgentEnv, 'reset', forbidden)


@pytest.fixture(scope='module')
def spent():
    paths = {v: SPENT / v / 'rollouts/017824/trace.json' for v in ('raw', 'regularized')}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip('local already-spent full physics fixtures unavailable; never generate replacements')
    out = {}
    for variant, path in paths.items():
        trace = _json(path)
        assert sha256(Path(trace['actor']['path']).read_bytes()).hexdigest() == trace['actor']['sha256']
        case = next(e['case'] for e in trace['events'] if e['case']['episode_id'] in trace['physics']['resets'])
        ref = trace['physics']['resets'][case['episode_id']]
        assert sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
        events = [e for e in trace['events'] if e['case'] == case]
        assert events[0]['pre_state_public']['frame'] == 0 and len(events) == 65
        out[variant] = dict(trace=trace, trace_ref=dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest()),
            case=case, initial_ref=ref, snapshot=_json(ref['path']), events=events)
    return out


def package(root, spent, *, audits=False, wrong_role=False):
    records = {}
    def add(value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        key = sha256(raw).hexdigest(); path = root / 'assets' / key / 'asset.json'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o444)
        records[key] = dict(path=path.relative_to(root).as_posix(), dependencies=[])
        return key
    manifest = dict(version=CATALOG_VERSION, release_id='spent-fixture-software-test-not-accepted', policies={})
    for domain in DOMAINS:
        pair = {}
        receipt = add(dict(version='unit-fixture-unaccepted-receipt', domain=domain, passed=False))
        for variant in ('raw', 'regularized'):
            if domain == 'warehouse':
                trace = spent[variant]['trace']; actor = trace['actor']['sha256']
                path = root / 'assets' / actor / 'actor.npz'; path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(trace['actor']['path'], path); path.chmod(0o444)
                records[actor] = dict(path=path.relative_to(root).as_posix(), dependencies=[])
                roles = dict(trace['rollout_state']['actor_role_sha256'])
                if wrong_role and variant == 'raw': roles['robot_2'] = 'f' * 64
                schema, feature, physics = MODEL_VERSION, FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256
            else:
                actor = add(dict(unit_fake_never_loaded_actor=True, domain=domain, variant=variant))
                roles = {r: digest(dict(domain=domain, variant=variant, role=r)) for r in ('robot_1', 'robot_2')}
                schema, feature, physics = 'unit-fake-model', digest(domain + '-feature'), digest(domain + '-physics')
            programs = []
            if variant == 'regularized':
                for role, native in (('human', 'robot_1'), ('ai', 'robot_2')):
                    source = add(dict(unit_fake_program=True, domain=domain, role=role))
                    archive = add(dict(unit_fake_archive=True, domain=domain, role=role))
                    audit = add(dict(unit_fake_audit_not_scientific=True, domain=domain, role=role)) if audits else None
                    programs.append(ProgramBinding(role, native, 'warehouse-role-program.v6.42-explicit-depth-capacity',
                        archive, source, audit))
            pair[variant] = PolicyBinding(manifest['release_id'], f'{domain}-{variant}-spent-fixture', domain, variant,
                schema, 'unit-software-fixture-not-accepted', actor, roles['robot_1'], roles['robot_2'],
                feature, physics, tuple(programs), receipt).to_dict()
        manifest['policies'][domain] = pair
    manifest['assets'] = records
    return PolicyCatalog(manifest, AssetResolver(root, records))


@pytest.fixture
def adapter(tmp_path, spent): return WarehouseAdapter(package(tmp_path, spent))


def start(adapter, spent, variant='regularized'):
    data = spent[variant]; case = data['case']; binding = adapter.catalog.binding('warehouse', variant)
    state = adapter.recorded_state(data['snapshot'], binding, task=case['task'],
        scenario_seed=case['scenario_seed'], action_seed=case['action_seed'],
        provenance=dict(trace=data['trace_ref'], initial_snapshot=data['initial_ref']))
    return state, binding, data


@pytest.mark.parametrize('variant', ['raw', 'regularized'])
def test_all_65_spent_frames_exact_dual_NN_actions_and_physics(adapter, spent, variant):
    state, binding, data = start(adapter, spent, variant)
    assert any(event['post_state_public']['last_robot_collision_event'] for event in data['events'])
    for old in data['events']:
        assert public_pre_action_snapshot(adapter.restore(state, binding).state) == old['pre_state_public']
        decision = adapter.decide(state, binding)
        assert decision['policy_actions'] == old['proposed_actions']
        masks = action_masks(adapter.restore(state, binding).state)
        for i, role in enumerate(('robot_1', 'robot_2')):
            dist = decision['distributions'][role]
            assert dist['action_mask'] == masks[i].tolist()
            assert masks[i, ACTIONS.index(decision['policy_actions'][role])] == 1
            assert all(p == 0 for p, allowed in zip(dist['probabilities'], masks[i]) if not allowed)
        state, record = adapter._advance(state, binding, None, decision)
        assert record['proposed_actions'] == old['proposed_actions']
        assert record['post_state_public'] == old['post_state_public']
        for field in ('rewards', 'terminated', 'truncated', 'shutdowns', 'action_override_agents'):
            assert record[field] == old[field]


def test_demo_restore_replay_counterfactual_same_frozen_NN_and_RNG(adapter, spent):
    state, binding, _ = start(adapter, spent)
    demo = adapter.demonstration(binding, state=state, steps=8)
    assert demo['two_frozen_NN'] and adapter.replay(demo, binding) == demo['final_state']
    assert capture_environment(adapter.restore(state, binding)) == state['snapshot']
    commands = [r['nn_policy_actions']['robot_1'].lower() for r in demo['records']]
    counter = adapter.counterfactual(state, commands, binding)
    assert adapter.replay(counter, binding) == counter['final_state'] == demo['final_state']
    assert adapter.decide(state, binding) == adapter.decide(deepcopy(state), binding)
    assert state == demo['initial_state'] == counter['initial_state']


def test_human_replaces_only_robot1_and_no_rule_changes_AI(adapter, spent):
    state, binding, _ = start(adapter, spent)
    decision = adapter.decide(state, binding)
    command = next(a.lower() for a in ACTIONS if a != decision['policy_actions']['robot_1'])
    nxt = adapter.step(state, command, binding, decision=decision)
    assert nxt['public_history']['previous_joint_actions']['human']['proposed'] == command
    assert nxt['public_history']['previous_joint_actions']['ai']['proposed'] == decision['action']
    assert state['snapshot'] == start(adapter, spent)[0]['snapshot']


def test_human_wall_command_keeps_environment_physical_filter(adapter, spent):
    state, binding, _ = start(adapter, spent)
    env = adapter.restore(state, binding)
    illegal = [a.lower() for a, bit in zip(ACTIONS, action_masks(env.state)[0]) if bit == 0]
    assert illegal
    decision = adapter.decide(state, binding)
    nxt = adapter.step(state, illegal[0], binding)
    history = nxt['public_history']['previous_joint_actions']
    assert history['human'] == dict(proposed=illegal[0], executed='wait')
    assert history['ai']['proposed'] == decision['action']


@pytest.mark.parametrize('fault', ['binding', 'receipt', 'frame', 'rng', 'history', 'physics', 'action_seed', 'policy_switch'])
def test_state_restore_refuses_binding_snapshot_frame_and_sampling_tamper(adapter, spent, fault):
    state, binding, _ = start(adapter, spent)
    if fault == 'binding': state['binding']['actor_sha256'] = 'f' * 64
    elif fault == 'receipt': state['binding']['accepted_receipt_sha256'] = 'f' * 64
    elif fault == 'frame': state['turn'] += 1
    elif fault == 'rng': state['snapshot']['payload']['episode_counter'] += 1
    elif fault == 'history': state['public_history']['same_target_collision_streak'] += 1
    elif fault == 'physics': state['binding']['action_physics_config_sha256'] = 'f' * 64
    elif fault == 'action_seed': state['action_seed'] += 1
    else: binding = adapter.catalog.binding('warehouse', 'raw')
    with pytest.raises((RuntimeUnavailable, ValueError)):
        adapter.restore(state, binding)


def test_snapshot_integrity_cannot_bypass_history_or_frame_consistency(adapter, spent):
    state, binding, _ = start(adapter, spent)
    state['turn'] += 1
    state['state_sha256'] = digest({k: v for k, v in state.items() if k != 'state_sha256'})
    with pytest.raises(RuntimeUnavailable, match='snapshot, RNG, frame'):
        adapter.restore(state, binding)


def test_decision_and_replay_tampering_rejected(adapter, spent):
    state, binding, _ = start(adapter, spent)
    decision = adapter.decide(state, binding); decision['policy_actions']['robot_2'] = 'WAIT'
    decision['action'] = 'changed'
    with pytest.raises(RuntimeUnavailable, match='decision'):
        adapter.step(state, 'wait', binding, decision=decision)
    demo = adapter.demonstration(binding, state=state, steps=2)
    demo['records'][0]['rewards']['robot_2'] += 1
    with pytest.raises(RuntimeUnavailable, match='replay NN action'):
        adapter.replay(demo, binding)


@pytest.mark.parametrize('fault', ['deleted', 'tampered'])
def test_cached_actor_still_rehashes_assets_before_each_call(adapter, spent, fault):
    state, binding, _ = start(adapter, spent)
    path = adapter.catalog.assets.path(binding.actor_sha256)
    if fault == 'deleted': path.unlink()
    else: path.chmod(0o644); path.write_bytes(path.read_bytes() + b'tamper'); path.chmod(0o444)
    with pytest.raises(RuntimeUnavailable): adapter.decide(state, binding)


def test_npz_actual_role_hash_checked_not_just_manifest(tmp_path, spent):
    adapter = WarehouseAdapter(package(tmp_path, spent, wrong_role=True))
    with pytest.raises(RuntimeError, match='role hash'):
        start(adapter, spent, 'raw')


def test_A_task2_only_evidence_and_actual_NN_action_forwarded(tmp_path, spent, monkeypatch):
    adapter = WarehouseAdapter(package(tmp_path, spent, audits=True))
    state, binding, _ = start(adapter, spent)
    calls = []
    class ExplicitUnitFakeReader:
        def evidence(self, x, mask, action, *, current_actor_role_sha256):
            calls.append((x.copy(), mask.copy(), action, current_actor_role_sha256))
            return dict(unit_fake_not_actual_audited_evidence=True, nn_action=action)
    monkeypatch.setattr(adapter.catalog, 'program', lambda b, r: ExplicitUnitFakeReader())
    for group in ('B', 'C'):
        b = adapter.catalog.binding('warehouse', 'raw' if group == 'C' else 'regularized')
        other, _, _ = start(adapter, spent, b.variant)
        bound = BoundDomain(b, adapter, adapter.catalog, group, 'test')
        assert adapter.explanation(other, bound) is None
    bound = BoundDomain(binding, adapter, adapter.catalog, 'A', 'test')
    evidence = adapter.explanation(state, bound)
    assert len(calls) == 1 and calls[0][0].shape == (173,) and calls[0][1].shape == (5,)
    assert calls[0][2] == evidence['nn_action'] == adapter.decide(state, binding)['action']
    assert calls[0][3] == binding.ai_actor_sha256
    state['task'] = 1; state['state_sha256'] = digest({k: v for k, v in state.items() if k != 'state_sha256'})
    assert adapter.explanation(state, bound) is None and len(calls) == 1


def test_missing_audit_gives_no_explanation_and_preview_never_grants_pilot(adapter, spent):
    state, binding, _ = start(adapter, spent)
    bound = BoundDomain(binding, adapter, adapter.catalog, 'A', 'preview')
    assert adapter.explanation(state, bound) is None
    with pytest.raises(RuntimeUnavailable, match='pilot remains closed'):
        adapter.catalog.require_pilot()
    public = adapter.public_state(state, binding)
    assert public['binding'] == binding.to_dict() and 'snapshot' not in public and 'action_seed' not in public


def test_forged_group_variant_or_pilot_handle_cannot_explain(adapter, spent):
    state, binding, _ = start(adapter, spent)
    for group, mode in (('C', 'test'), ('A', 'unknown'), ('A', 'pilot')):
        handle = BoundDomain(binding, adapter, adapter.catalog, group, mode)
        with pytest.raises(RuntimeUnavailable): adapter.explanation(state, handle)


def test_nonphysical_human_command_and_empty_or_unbounded_counterfactual_rejected(adapter, spent):
    state, binding, _ = start(adapter, spent)
    for command in (None, 'rescue', 'UP', {}, 0):
        with pytest.raises(RuntimeUnavailable): adapter.step(state, command, binding)
    for commands in ([], ['wait'] * 13, 'wait', [None]):
        with pytest.raises(RuntimeUnavailable): adapter.counterfactual(state, commands, binding)
