"""Temporary DB/fake adapter software tests; no model or human-study admission."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import importlib.util
import json
import sqlite3

import pytest

from runtime_v642 import RuntimeRegistry, RuntimeUnavailable
from runtime_v642.contracts import digest, require
from runtime_v642.store import Store, StudyError, ReadOnlyLegacyReader


def package_at(path):
    spec = importlib.util.spec_from_file_location('core_store_synthetic_catalog_fixture',
        Path(__file__).with_name('test_runtime_v642_registry.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    package, manifest = helper.package.__wrapped__(path)
    # Deliberately NO fake verifier: pilot remains closed in every test.
    return package, manifest, package.catalog(manifest)


class FakeAdapter:
    """Deterministic toy transitions, explicitly not a neural/model evaluation."""
    def __init__(self, domain):
        self.DOMAIN, self.calls, self.evidence_mode = domain, [], 'none'

    def note(self, method, binding):
        value = binding.to_dict()
        assert 'group' not in value and 'group_code' not in value
        self.calls.append((method, value))

    def wrap(self, seed, task, turn, binding, total=0):
        value = dict(domain=self.DOMAIN, task=task, seed=seed, turn=turn, max_turns=2,
            terminal=turn == 2, binding=binding.to_dict(), binding_identity_sha256=binding.identity_sha256,
            synthetic_counter=total)
        value['state_sha256'] = digest(value)
        return value

    def initial_state(self, seed, task, binding):
        self.note('initial_state', binding)
        return self.wrap(seed, task, 0, binding)

    def restore(self, state, binding):
        self.note('restore', binding)
        require(state['binding'] == binding.to_dict() and state['binding_identity_sha256'] == binding.identity_sha256
                and digest({k: v for k, v in state.items() if k != 'state_sha256'}) == state['state_sha256'],
                'synthetic state corrupt')
        return deepcopy(state)

    def decide(self, state, binding):
        self.note('decide', binding)
        self.restore(state, binding)
        require(not state['terminal'], 'synthetic terminal')
        return dict(action='toy_'+binding.ai_actor_sha256[:8]+'_'+str(state['turn']),
            policy_actions=dict(human='wait', ai='toy_'+binding.ai_actor_sha256[:8]+'_'+str(state['turn'])),
            state_sha256=state['state_sha256'], binding_identity_sha256=binding.identity_sha256,
            ai_action_overrides=0, execution_source='bound_frozen_NN_with_physical_mask')

    def advance(self, state, action, binding, decision=None):
        actual = self.decide(state, binding)
        require(decision is None or actual == decision, 'synthetic decision changed')
        require(action is None or action in ('wait', 'up'), 'synthetic illegal command')
        nxt = self.wrap(state['seed'], state['task'], state['turn']+1, binding, state['synthetic_counter']+1)
        return nxt, dict(human_action=action, decision=actual, state=nxt)

    def step(self, state, human_action, binding, *, decision=None):
        self.note('step', binding)
        require(human_action is not None, 'synthetic human command missing')
        return self.advance(state, human_action, binding, decision)[0]

    def bundle(self, state, actions, binding, *, both=False):
        initial = deepcopy(state)
        records = []
        for action in actions:
            if state['terminal']:
                break
            state, record = self.advance(state, action, binding)
            records.append(record)
        return dict(version='synthetic-core-test-bundle.v1', binding=binding.to_dict(), initial_state=initial,
            records=records, final_state=state, two_frozen_NN=both)

    def demonstration(self, binding, *, seed, task, steps):
        self.note('demonstration', binding)
        return self.bundle(self.initial_state(seed, task, binding), [None]*min(steps, 2), binding, both=True)

    def counterfactual(self, state, human_actions, binding):
        self.note('counterfactual', binding)
        return self.bundle(state, human_actions, binding)

    def replay(self, bundle, binding):
        self.note('replay', binding)
        require(bundle['binding'] == binding.to_dict(), 'synthetic replay binding')
        state = self.restore(bundle['initial_state'], binding)
        for record in bundle['records']:
            state, actual = self.advance(state, record['human_action'], binding, record['decision'])
            require(actual == record and (not bundle['two_frozen_NN'] or record['human_action'] is None), 'synthetic replay changed')
        require(state == bundle['final_state'], 'synthetic final state changed')
        return state

    def public_state(self, state, binding):
        self.note('public_state', binding)
        self.restore(state, binding)
        return dict(binding=binding.to_dict(), state_sha256=state['state_sha256'],
            turn=state['turn'], terminal=state['terminal'], synthetic_not_a_model_evaluation=True)

    def explanation(self, state, bound):
        bound.verify()
        self.note('explanation', bound.binding)
        assert bound.group == 'A' and state['task'] == 2
        if self.evidence_mode == 'none':
            return None
        p = next(p for p in bound.binding.programs if p.role == 'ai')
        result = dict(eligible=True, target_actor_role_sha256=bound.binding.ai_actor_sha256,
            nn_action=self.decide(state, bound.binding)['action'], source_program_sha256=p.source_program_sha256,
            independent_audit_sha256=p.independent_audit_sha256, external_program_selects_execution_action=False,
            unique_causal_motive_established=False, synthetic_unit_test_evidence=True)
        if self.evidence_mode == 'wrong_action':
            result['nn_action'] = 'wrong'
        return result


@pytest.fixture
def configured(tmp_path):
    package, manifest, catalog = package_at(tmp_path/'package')
    adapters = {d: FakeAdapter(d) for d in ('warehouse', 'pong', 'kitchen')}
    registry = RuntimeRegistry(catalog, adapters=adapters)
    pools = {d: (11, 12) for d in adapters}
    demos = {d: 10 for d in adapters}
    store = Store(tmp_path/'new.sqlite', registry, scenario_pools=pools, demonstration_seeds=demos,
                  mode='test', choose=lambda values: values[0])
    yield store, adapters, registry, package, manifest
    store.close()


def enrollment(store, domain='warehouse', group='A', participant='synthetic-1'):
    result = store.enroll(domain=domain, participant_reference=participant, group=group, admin=True)
    return result['session_token'], result['instance']['id'], result['instance']


def demo_and_start(store, token, instance_id):
    demo = store.demonstration(token, instance_id)
    assert demo['two_frozen_NN']
    assert store.demonstration(token, instance_id) == demo
    store.acknowledge_demonstration(token, instance_id)
    return store.start_task(token, instance_id)


def finish(store, token, instance_id, task):
    for turn in (0, 1):
        result = store.action(token, instance_id, command_id=f'{task}-{turn}', expected_turn=turn, human_action='wait')
    return result


@pytest.mark.parametrize('domain', ['warehouse', 'pong', 'kitchen'])
@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_complete_persistent_core_three_tasks_demo_resume_replay_QA_export(configured, domain, group):
    store, adapters, registry, _, _ = configured
    token, iid, identity = enrollment(store, domain, group)
    with pytest.raises(StudyError, match='previous_stage'):
        store.start_task(token, iid)
    view = demo_and_start(store, token, iid)
    assert view['current']['state']['task'] == 1
    finish(store, token, iid, 1)
    store.start_task(token, iid)
    if group == 'A':
        answer = store.ask(token, iid, question='为什么？', turn=0, human_actions=['up'])
        assert answer['status'] == 'insufficient_evidence' and '无法推断动机' in answer['answer']
        assert answer['counterfactual']['binding'] == identity['binding']
        assert not answer['counterfactual']['two_frozen_NN']
        adapters[domain].evidence_mode = 'valid'
        assert store.ask(token, iid, question='What action?', turn=0, language='en')['status'] == 'grounded_pattern'
    else:
        with pytest.raises(StudyError, match='only_A'):
            store.ask(token, iid, question='why?', turn=0)
    finish(store, token, iid, 2)
    store.start_task(token, iid)
    with pytest.raises(StudyError, match='only_A'):
        store.ask(token, iid, question='why?', turn=0)
    finish(store, token, iid, 3)
    assert store.resume(token, iid)['instance']['stage'] == 'complete'
    replay = store.replay(token, iid, task=2)
    assert len(replay['frames']) == 3 and replay['all_AI_actions_recomputed']
    assert store.frame(token, iid, task=2, turn=1) == replay['frames'][1]
    exported = store.export_instance(token, iid, admin=True)
    assert len(exported['records']['frames']) == 9
    assert all(f['binding'] == identity['binding'] for f in exported['records']['frames'])
    assert 'token_hash' not in json.dumps(exported)
    initial_count = sum(m == 'initial_state' for m, _ in adapters[domain].calls)
    store.resume(token, iid)
    assert sum(m == 'initial_state' for m, _ in adapters[domain].calls) == initial_count
    # Real disk reopen, same immutable binding and state, no implicit reset.
    store.close()
    store.db = Store(store.database, registry, scenario_pools=store.scenarios,
        demonstration_seeds=store.demo_seeds, mode='test').db
    assert store.resume(token, iid)['instance']['binding'] == identity['binding']


def test_A_B_same_pair_C_raw_and_nonadmin_overrides_forbidden(configured):
    store, _, _, _, _ = configured
    rows = [enrollment(store, group=group, participant=group)[2] for group in ('A', 'B', 'C')]
    assert rows[0]['binding'] == rows[1]['binding']
    assert rows[0]['binding']['actor_sha256'] != rows[2]['binding']['actor_sha256']
    assert rows[0]['binding']['ai_actor_sha256'] != rows[2]['binding']['ai_actor_sha256']
    with pytest.raises(StudyError, match='admin_required'):
        store.enroll(domain='warehouse', participant_reference='bad', group='C')
    with pytest.raises(StudyError, match='admin_required'):
        store.enroll(domain='warehouse', participant_reference='bad', scenario_seed=11)


def test_balanced_assignment_per_domain_and_scene_is_atomic(configured):
    store, _, _, _, _ = configured
    for domain in ('warehouse', 'pong', 'kitchen'):
        results = [store.enroll(domain=domain, participant_reference=f'person-{i}')['instance'] for i in range(18)]
        for seed in (11, 12):
            assert [sum(r['scenario_seed'] == seed and r['group'] == g for r in results) for g in ('A', 'B', 'C')] == [3]*3
    with pytest.raises(StudyError, match='already_enrolled'):
        store.enroll(domain='warehouse', participant_reference='person-1')


def test_pilot_unknown_formal_reader_blocks_every_entry_before_execution(configured, tmp_path):
    _, adapters, registry, _, _ = configured
    pilot = Store(tmp_path/'pilot.sqlite', registry, scenario_pools={d: (11,) for d in adapters},
                  demonstration_seeds={d: 10 for d in adapters})
    try:
        calls = [lambda: pilot.status(), lambda: pilot.enroll(domain='warehouse', participant_reference='x', group='A'),
            lambda: pilot.view('nonexistent-token-0000', 'none'), lambda: pilot.demonstration('nonexistent-token-0000', 'none'),
            lambda: pilot.start_task('nonexistent-token-0000', 'none'),
            lambda: pilot.action('nonexistent-token-0000', 'none', command_id='x', expected_turn=0, human_action='wait'),
            lambda: pilot.frame('nonexistent-token-0000', 'none', task=1, turn=0),
            lambda: pilot.replay('nonexistent-token-0000', 'none', task=1),
            lambda: pilot.ask('nonexistent-token-0000', 'none', question='why?', turn=0),
            lambda: pilot.export_instance('nonexistent-token-0000', 'none', admin=True)]
        for call in calls:
            with pytest.raises(RuntimeUnavailable, match='pilot remains closed'):
                call()
        assert pilot.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0] == 0
        assert all(not adapter.calls for adapter in adapters.values())
    finally:
        pilot.close()


def test_commands_idempotent_stale_turn_and_illegal_action_fail_closed(configured):
    store, _, _, _, _ = configured
    token, iid, _ = enrollment(store)
    demo_and_start(store, token, iid)
    first = store.action(token, iid, command_id='same', expected_turn=0, human_action='wait')
    assert store.action(token, iid, command_id='same', expected_turn=0, human_action='wait') == first
    with pytest.raises(StudyError, match='reused'):
        store.action(token, iid, command_id='same', expected_turn=0, human_action='up')
    with pytest.raises(StudyError, match='stale'):
        store.action(token, iid, command_id='other', expected_turn=0, human_action='wait')
    with pytest.raises(RuntimeUnavailable, match='illegal'):
        store.action(token, iid, command_id='bad', expected_turn=1, human_action='rescue_controller')
    assert store.resume(token, iid)['current']['turn'] == 1


def test_command_id_cannot_be_reused_for_a_new_task_same_turn_and_action(configured):
    store, _, _, _, _ = configured
    token, iid, _ = enrollment(store)
    demo_and_start(store, token, iid)
    first = store.action(token, iid, command_id='cross-task-id', expected_turn=0, human_action='wait')
    assert store.action(token, iid, command_id='cross-task-id', expected_turn=0, human_action='wait') == first
    store.action(token, iid, command_id='finish-first-task', expected_turn=1, human_action='wait')
    store.start_task(token, iid)
    with pytest.raises(StudyError, match='command_id_reused'):
        store.action(token, iid, command_id='cross-task-id', expected_turn=0, human_action='wait')
    assert store.resume(token, iid)['current']['state']['task'] == 2
    assert store.resume(token, iid)['current']['turn'] == 0


def test_public_projection_hides_private_state_distribution_and_research_export_needs_admin(configured):
    store, _, _, _, _ = configured
    token, iid, _ = enrollment(store)
    demo = store.demonstration(token, iid)
    projected = store.participant_view(token, iid, demo)
    assert not projected['contains_restorable_private_snapshot']
    assert projected['records'][0]['NN_decision']['NN_pair_actions']
    assert 'initial_state' not in projected and 'state' not in projected['records'][0]
    store.acknowledge_demonstration(token, iid)
    full = store.start_task(token, iid)
    public = store.participant_view(token, iid, full)
    assert 'state' not in public['current'] and 'transition' not in public['current']
    assert set(public['current']['decision']) == {'action', 'binding_identity_sha256', 'state_sha256'}
    with pytest.raises(StudyError, match='admin_required'):
        store.research_export(iid)
    index = store.research_index(admin=True)
    assert index['instances'][0]['id'] == iid
    assert store.research_export(iid, admin=True)['all_AI_frames_recomputed']


def test_binding_and_frames_immutable_and_corrupt_decision_rejected(configured):
    store, _, _, _, _ = configured
    token, iid, _ = enrollment(store)
    demo_and_start(store, token, iid)
    for query in ("UPDATE v642_instances SET group_code='C'", "UPDATE v642_instances SET binding_json='{}'",
                  "UPDATE v642_frames SET decision_json='{}'"):
        with pytest.raises(sqlite3.IntegrityError):
            store.db.execute(query)
    # Simulate a corrupt offline DB beyond application write permissions.
    store.db.execute('DROP TRIGGER v642_frames_update_immutable')
    store.db.execute("UPDATE v642_frames SET decision_json='{}'")
    with pytest.raises(RuntimeUnavailable, match='hash chain'):
        store.resume(token, iid)


def test_missing_model_refuses_resume_demo_replay_export(configured):
    store, _, registry, package, _ = configured
    token, iid, identity = enrollment(store)
    demo_and_start(store, token, iid)
    path = package.root/package.records[identity['binding']['actor_sha256']]['path']
    path.unlink()
    for call in (lambda: store.resume(token, iid), lambda: store.demonstration(token, iid),
                 lambda: store.replay(token, iid, task=1), lambda: store.export_instance(token, iid, admin=True)):
        with pytest.raises(RuntimeUnavailable):
            call()


def test_bad_QA_evidence_rejected_and_kitchen_exception_labelled(configured):
    store, adapters, _, _, _ = configured
    token, iid, _ = enrollment(store, domain='kitchen')
    label = store.kitchen_practice_metadata(token, iid)
    assert label['labelled_exception'] and not label['model_evaluation'] and not label['interactive_practice_available']
    demo_and_start(store, token, iid)
    finish(store, token, iid, 1)
    store.start_task(token, iid)
    adapters['kitchen'].evidence_mode = 'wrong_action'
    with pytest.raises(RuntimeUnavailable, match='matching the current NN'):
        store.ask(token, iid, question='why?', turn=0)


def test_session_authority_cannot_read_another_instance(configured):
    store, _, _, _, _ = configured
    one, iid, _ = enrollment(store, participant='one')
    other, _, _ = enrollment(store, participant='two')
    with pytest.raises(StudyError, match='not_found'):
        store.resume(other, iid)


def test_readonly_legacy_snapshot_read_export_no_writes_or_model_resume(tmp_path):
    path = tmp_path/'legacy.sqlite'
    db = sqlite3.connect(path)
    db.executescript('''CREATE TABLE pl3_instances(id TEXT,release_id TEXT,participant_id TEXT);
        CREATE TABLE pl3_runs(id TEXT,instance_id TEXT);CREATE TABLE pl3_frames(run_id TEXT,turn INTEGER);
        INSERT INTO pl3_instances VALUES('old','v5','p');INSERT INTO pl3_runs VALUES('r','old');
        INSERT INTO pl3_frames VALUES('r',0);''')
    db.commit(); db.close()
    before = sha256(path.read_bytes()).hexdigest()
    reader = ReadOnlyLegacyReader(path)
    assert reader.read_instance('old')['resumable_with_v642'] is False
    assert reader.export_instance('old')['records']['frames'] == [dict(run_id='r', turn=0)]
    assert before == sha256(path.read_bytes()).hexdigest()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['legacy.sqlite']
    with reader._read() as ro:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("UPDATE pl3_instances SET release_id='new'")
    Path(str(path)+'-wal').write_bytes(b'pending')
    with pytest.raises(RuntimeUnavailable, match='quiescent'):
        reader.read_instance('old')
