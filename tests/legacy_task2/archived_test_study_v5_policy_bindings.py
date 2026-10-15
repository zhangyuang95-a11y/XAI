"""V5 treatment assignment and immutable frozen-policy provenance."""
from copy import deepcopy
from hashlib import sha256
import json
import uuid

import pytest

from study_v3 import policies
from study_v3.config import Settings
from study_v3.store import Store, StudyError
from study_v3 import store as store_module
from study_v3 import server as server_module


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def test_policy_binding_verifies_actor_and_programs(tmp_path, monkeypatch):
    monkeypatch.setattr(policies, 'ROOT', tmp_path)
    entries = {}
    for variant in ('regularized', 'raw'):
        actor = tmp_path / f'{variant}.npz'
        actor.write_bytes(variant.encode())
        programs = []
        if variant == 'regularized':
            for role in ('human', 'ai'):
                program = tmp_path / f'{variant}-{role}.json'
                program.write_text(role)
                programs.append(program.name)
        entries[variant] = {'policy_id': f'pong-{variant}-v5',
                            'actor_path': actor.name, 'actor_sha256': _digest(actor),
                            'role_sha256': {'human': 'a'*64, 'ai': 'b'*64},
                            'program_paths': programs}
    monkeypatch.setattr(policies, '_catalog_entry', lambda domain, variant: entries[variant])
    a = policies.binding_for_group('pong', 'A')
    b = policies.binding_for_group('pong', 'B')
    c = policies.binding_for_group('pong', 'C')
    assert a == b
    assert a['actor_sha256'] != c['actor_sha256']
    assert len(a['program_hashes']) == 2 and not c['program_hashes']
    (tmp_path / 'regularized.npz').write_bytes(b'tampered')
    with pytest.raises(policies.PolicyUnavailable, match='actor_hash_mismatch'):
        policies.verify_binding(a)


def test_release_gate_requires_real_feedback_and_pinned_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(policies, 'ROOT', tmp_path)
    actor = tmp_path / 'actor.npz'
    report = tmp_path / 'train.json'
    programs = [tmp_path / f'{role}.json' for role in ('human', 'ai')]
    for path in (actor, report, *programs):
        path.write_text(path.name)
    binding = {'domain': 'pong', 'variant': 'regularized', 'actor_sha256': _digest(actor),
               'program_hashes': {path.name: _digest(path) for path in programs}}
    candidates = []
    for seed in (11, 23, 37):
        common = {'joint_steps': 200000, 'bc_checkpoint_sha256': 'b'*64,
                  'initial_actor_sha256': {'human': '1'*64, 'ai': '2'*64},
                  'seed_schedule_sha256': str(seed).zfill(64),
                  'actor_weight_change_l2': {'human': 1.0, 'ai': 1.0},
                  'post_policy_override_count': 0}
        candidates.append({'pair_seed': seed,
                           'raw': {**common, 'program_feedback_rows': {'human': 0, 'ai': 0}},
                           'regularized': {**common, 'program_feedback_rows': {'human': 20, 'ai': 15}}})
    heldout = {variant: {'rows': [{'task': task, 'seed': seed,
                                  'capacity_percent': 95, 'ai_wait_rate': .05}
                                    for task in (1, 2, 3) for seed in range(890000, 890100)],
                         'summaries': [{'task': task, 'mean_capacity_percent': 95,
                                        'min_capacity_percent': 95,
                                        'mean_ai_wait_rate': .05,
                                        'max_ai_wait_rate': .05} for task in (1, 2, 3)]}
               for variant in ('raw', 'regularized')}
    acceptance = {'release': policies.RELEASE_ID,
                  'passed': True, 'heldout_evaluated': True,
                  'pair_seeds': [11, 23, 37], 'selected_pair_seed': 23,
                  'candidate_training': candidates, 'heldout': heldout,
                  'development': {variant: {'passed': True}
                                  for variant in ('raw', 'regularized')},
                  'runtime_parity': {'raw': True, 'regularized': True},
                  'selected_actor_development_program_audit': {
                      'human': {'passed': True, 'purpose': 'training_feedback',
                                'sample_count': 1920, 'action_fidelity': .85,
                                'mean_kl_divergence': .37},
                      'ai': {'passed': True, 'purpose': 'explanation',
                             'sample_count': 1920, 'action_fidelity': .98,
                             'mean_kl_divergence': .09}},
                  'joint_steps': {'raw': 200000, 'regularized': 200000},
                  'artifact_sha256': {'regularized': _digest(actor)},
                  'training_report_sha256': {'regularized': _digest(report)},
                  'program_sha256': {'regularized': {'human': _digest(programs[0]),
                                                      'ai': _digest(programs[1])}},
                  'raw_program_feedback_rows': {'human': 0, 'ai': 0},
                  'regularized_program_feedback_rows': {'human': 20, 'ai': 15},
                  'post_policy_override_count': 0}
    acceptance_path = tmp_path / 'acceptance.json'
    acceptance_path.write_text(json.dumps(acceptance))
    entry = {'training_report_path': report.name, 'acceptance_path': acceptance_path.name,
             'program_paths': [path.name for path in programs]}
    assert policies._accepted(entry, binding)
    acceptance['program_sha256']['regularized'] = {
        'human': _digest(programs[1]), 'ai': _digest(programs[0])}
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['program_sha256']['regularized'] = {
        'human': _digest(programs[0]), 'ai': _digest(programs[1])}
    acceptance['regularized_program_feedback_rows']['ai'] = 0
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['regularized_program_feedback_rows']['ai'] = 15
    acceptance['heldout']['raw']['rows'][0]['seed'] = 890999
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['heldout']['raw']['rows'][0]['seed'] = 890000
    acceptance['heldout']['raw']['rows'][0]['capacity_percent'] = 79
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['heldout']['raw']['rows'][0]['capacity_percent'] = 95
    acceptance['selected_actor_development_program_audit']['ai']['passed'] = False
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['selected_actor_development_program_audit']['ai']['passed'] = True
    acceptance['selected_actor_development_program_audit']['ai']['action_fidelity'] = .94
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['selected_actor_development_program_audit']['ai']['action_fidelity'] = .98
    acceptance['runtime_parity']['regularized'] = False
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)
    acceptance['runtime_parity']['regularized'] = True
    acceptance['candidate_training'][0]['regularized']['program_feedback_rows']['ai'] = 0
    acceptance_path.write_text(json.dumps(acceptance))
    assert not policies._accepted(entry, binding)


def test_release_requires_distinct_role_weights_not_just_variant_metadata(monkeypatch):
    common = {'accepted': True, 'human_actor_sha256': '1'*64,
              'ai_actor_sha256': '2'*64}
    monkeypatch.setattr(policies, 'policy_manifest', lambda: {
        domain: {'raw': {**common, 'actor_sha256': 'a'*64},
                 'regularized': {**common, 'actor_sha256': 'b'*64}}
        for domain in ('warehouse', 'pong', 'kitchen')})
    assert not policies.all_policies_accepted()


def test_warehouse_selected_checkpoint_must_have_both_program_gradients():
    candidate = {'pair_seed': 6001, 'variants': {
        'raw': {'feedback_rows': {'robot_1': 0, 'robot_2': 0}},
        'regularized': {'feedback_rows': {'robot_1': 120, 'robot_2': 80},
                        'selected_feedback_rows': {'robot_1': 0, 'robot_2': 80}}}}
    receipt = {'candidate_training': [candidate], 'selected_pair_seed': 6001,
               'raw_program_feedback_rows': candidate['variants']['raw']['feedback_rows'],
               'regularized_program_feedback_rows': candidate['variants']['regularized']['feedback_rows'],
               'selected_regularized_feedback_rows': candidate['variants']['regularized']['selected_feedback_rows']}
    assert not policies._selected_feedback_matches('warehouse', receipt)
    candidate['variants']['regularized']['selected_feedback_rows']['robot_1'] = 50
    assert policies._selected_feedback_matches('warehouse', receipt)


def test_warehouse_candidates_share_bc_start_and_account_for_every_feedback_window_row():
    initial = {'robot_1': '1' * 64, 'robot_2': '2' * 64}
    def variant(regularized):
        feedback = {'robot_1': 90, 'robot_2': 80} if regularized else {
            'robot_1': 0, 'robot_2': 0}
        exposure = {'robot_1': 100, 'robot_2': 100} if regularized else {
            'robot_1': 0, 'robot_2': 0}
        return {'joint_steps': 200000, 'actor_decisions': 400000,
                'scenario_schedule_sha256': 'a' * 64,
                'initial_role_sha256': initial,
                'selected_role_sha256': {'robot_1': ('5' if regularized else '3') * 64,
                                         'robot_2': ('6' if regularized else '4') * 64},
                'actor_weight_delta_l2': {'robot_1': .2, 'robot_2': .3},
                'actor_weight_changed_parameter_count': {'robot_1': 12, 'robot_2': 13},
                'both_roles_changed': True, 'feedback_rows': feedback,
                'selected_feedback_rows': {'robot_1': 40, 'robot_2': 30} if regularized else {
                    'robot_1': 0, 'robot_2': 0},
                'feedback_window_actor_rows': exposure,
                'zero_action_coverage_rows': {role: exposure[role] - feedback[role]
                                               for role in initial},
                'physical_no_legal_action_rows': {'robot_1': 0, 'robot_2': 0},
                'post_policy_interventions': 0}
    receipt = {'candidate_training': [
        {'pair_seed': seed, 'initial_role_sha256': initial,
         'variants': {'raw': variant(False), 'regularized': variant(True)}}
        for seed in (11, 23, 37)]}
    assert policies._candidate_training_valid('warehouse', receipt, [11, 23, 37])
    mutations = (
        ('raw', 'initial_role_sha256', {'robot_1': '0' * 64, 'robot_2': '2' * 64}),
        ('regularized', 'actor_weight_delta_l2', {'robot_1': 0, 'robot_2': .3}),
        ('regularized', 'actor_weight_changed_parameter_count', {'robot_1': 0, 'robot_2': 13}),
        ('regularized', 'feedback_window_actor_rows', {'robot_1': 89, 'robot_2': 100}),
        ('regularized', 'zero_action_coverage_rows', {'robot_1': 0, 'robot_2': 20}),
        ('regularized', 'physical_no_legal_action_rows', {'robot_1': 1, 'robot_2': 0}),
    )
    for variant_name, field, invalid in mutations:
        changed = deepcopy(receipt)
        changed['candidate_training'][0]['variants'][variant_name][field] = invalid
        assert not policies._candidate_training_valid('warehouse', changed, [11, 23, 37]), field


def test_kitchen_heldout_aggregates_must_match_failure_rows():
    result = {'episodes': 300, 'successes': 299, 'failures': [{'seed': 50000}],
              'completion_rate': 299 / 300, 'partner': None,
              'seeds': list(range(50000, 50100))}
    heldout = {variant: {partner: {**result, 'partner': partner}
                         for partner in ('nn', 'proxy')}
               for variant in ('raw', 'regularized')}
    assert policies._heldout_quality_valid('kitchen', {'heldout': heldout})
    changed = deepcopy(heldout)
    changed['raw']['nn']['failures'] = []
    assert not policies._heldout_quality_valid('kitchen', {'heldout': changed})
    changed = deepcopy(heldout)
    changed['regularized']['proxy']['completion_rate'] = 1.0
    assert not policies._heldout_quality_valid('kitchen', {'heldout': changed})


def test_warehouse_final_selective_audit_checks_measured_coverage_and_fidelity():
    def metric(rows, fidelity):
        return {'rows': rows, 'coverage': rows / 2400, 'agreement': fidelity,
                'kl': .20, 'illegal_count': 0}

    receipt = {'role_sha256': {'regularized': {'robot_1': '1' * 64,
                                               'robot_2': '2' * 64}},
               'program_sha256': {'regularized': {'robot_1': 'a' * 64,
                                                   'robot_2': 'b' * 64}}}
    base = {'audit_source': 'final_independent_development_rollouts',
            'development_seeds': list(range(240100, 240124)),
            'sample_count': 2400, 'program_depth': 5, 'program_leaf_count': 16,
            'feedback_eligible': True, 'explanation_eligible': True,
            'validation': {'global': metric(2400, .70),
                           'feedback': metric(600, .88),
                           'explanation': metric(500, .92)}}
    ai = {**base, 'role': 'robot_2', 'actor_role_sha256': '2' * 64,
          'program_sha256': 'b' * 64}
    assert policies._warehouse_selective_audit_valid(ai, 'robot_2', receipt,
                                                       explanation=True)
    low_coverage = json.loads(json.dumps(ai))
    low_coverage['validation']['explanation'] = metric(400, .99)
    assert not policies._warehouse_selective_audit_valid(low_coverage, 'robot_2',
                                                          receipt, explanation=True)
    low_fidelity = json.loads(json.dumps(ai))
    low_fidelity['validation']['explanation']['agreement'] = .899
    assert not policies._warehouse_selective_audit_valid(low_fidelity, 'robot_2',
                                                          receipt, explanation=True)
    wrong_actor = {**ai, 'actor_role_sha256': '3' * 64}
    assert not policies._warehouse_selective_audit_valid(wrong_actor, 'robot_2',
                                                          receipt, explanation=True)
    human = {**base, 'role': 'robot_1', 'actor_role_sha256': '1' * 64,
             'program_sha256': 'a' * 64, 'explanation_eligible': False}
    assert policies._warehouse_selective_audit_valid(human, 'robot_1', receipt,
                                                       explanation=False)


def test_warehouse_release_requires_matching_one_use_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(policies, 'ROOT', tmp_path)
    folder = tmp_path / 'output' / 'deployment'
    folder.mkdir(parents=True)
    acceptance = {
        'selected_pair_seed': 6001,
        'candidate_training': [{'pair_seed': 6001, 'variants': {
            variant: {'scenario_schedule_sha256': variant[0] * 64}
            for variant in ('raw', 'regularized')}}],
        'role_sha256': {variant: {'robot_1': '1' * 64, 'robot_2': '2' * 64}
                        for variant in ('raw', 'regularized')},
        'program_sha256': {'regularized': {'robot_1': 'a' * 64,
                                            'robot_2': 'b' * 64}},
        'heldout_scenario_seed_rule': 'enrollment_seed_times_10_plus_task',
        'heldout_seed_ranges': {str(task): [250000, 250099] for task in (1, 2, 3)},
        'all_candidate_regularized_feedback_passed': True,
        'final_program_audit': {'robot_1': {'passed': True},
                                'robot_2': {'passed': True}},
        'heldout': {'raw': {'episodes': []}, 'regularized': {'episodes': []}},
    }
    selection = {
        'release': policies.RELEASE_ID, 'selected_pair_seed': 6001,
        'selected_role_sha256': acceptance['role_sha256'],
        'scenario_schedule_sha256': {'raw': 'r' * 64, 'regularized': 'r' * 64},
        'selected_program_sha256': acceptance['program_sha256']['regularized'],
        'heldout_scenario_seed_rule': acceptance['heldout_scenario_seed_rule'],
        'heldout_seed_ranges': acceptance['heldout_seed_ranges'],
    }
    def write_receipt(name, selected, key, evidence):
        path = folder / name
        path.write_text(json.dumps({'status': 'completed', 'selection': selected,
             key: evidence,
             key + '_sha256': sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()}))
        return path
    program = write_receipt('warehouse_dual_v5_final_program_audit_receipt.json',
                            {**selection, 'final_program_audit_seeds':
                                list(range(240100, 240124))},
                            'audit', acceptance['final_program_audit'])
    heldout = write_receipt('warehouse_dual_v5_heldout_receipt.json',
                            selection, 'heldout', acceptance['heldout'])
    assert policies._warehouse_once_receipts_valid(acceptance)
    payload = json.loads(heldout.read_text())
    payload['selection']['selected_pair_seed'] = 6002
    heldout.write_text(json.dumps(payload))
    assert not policies._warehouse_once_receipts_valid(acceptance)
    heldout.unlink()
    assert not policies._warehouse_once_receipts_valid(acceptance)
    assert program.is_file()


class _FakeEngine:
    @staticmethod
    def rules(_language):
        return []


def test_three_groups_balance_per_domain_and_bind_once(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, 'engine', lambda domain: _FakeEngine())
    monkeypatch.setattr(store_module, 'demonstration', lambda domain, policy_id: {
        'captions': [{'index': 0, 'en': 'Two AIs', 'zh': '双 AI'}], 'frames': [{}]})
    monkeypatch.setattr(store_module, 'scenario_config', lambda domain: {
        'heldout_seeds': [700, 701, 702], 'version': 'fixture'})
    def fake_binding(domain, group):
        variant = 'raw' if group == 'C' else 'regularized'
        return {'policy_id': f'{domain}-{variant}-v5', 'domain': domain,
                'variant': variant, 'actor_path': f'{variant}.npz',
                'actor_sha256': ('c' if variant == 'raw' else 'a')*64,
                'human_actor_sha256': '1'*64, 'ai_actor_sha256': '2'*64,
                'program_hashes': {}}
    monkeypatch.setattr(store_module, 'binding_for_group', fake_binding)
    monkeypatch.setattr(store_module, 'verify_binding', lambda binding: binding)
    study = Store(Settings(database=str(tmp_path / 'study.sqlite3')))
    try:
        first_token, first_view = study.create({'consent': True, 'domain': 'pong', 'mode': 'test',
                 'participant_id': 'v5-user-0'}, admin=True)
        views = [first_view] + [study.create({'consent': True, 'domain': 'pong', 'mode': 'test',
                 'participant_id': f'v5-user-{i}'}, admin=True)[1] for i in range(1, 9)]
        assert {group: sum(view['group'] == group for view in views)
                for group in ('A', 'B', 'C')} == {'A': 3, 'B': 3, 'C': 3}
        with study.db.transaction(read_only=True) as db:
            assignments = db.all('SELECT scenario_seed,group_code FROM pl3_instances')
        for seed in (700, 701, 702):
            assert {group: sum(row['group_code'] == group and row['scenario_seed'] == seed
                               for row in assignments)
                    for group in ('A', 'B', 'C')} == {'A': 1, 'B': 1, 'C': 1}
        assert {view['actor_sha256'] for view in views if view['group'] in ('A', 'B')} == {'a'*64}
        assert {view['actor_sha256'] for view in views if view['group'] == 'C'} == {'c'*64}
        with study.db.transaction(read_only=True) as db:
            rows = db.all('SELECT policy_id,actor_sha256 FROM pl3_policy_bindings')
        assert len(rows) == 9
        assert {row['policy_id'] for row in rows} == {'pong-regularized-v5', 'pong-raw-v5'}
        exported = study.export()
        assert len(exported['policy_bindings']) == 9
        different = next(group for group in ('A', 'B', 'C') if group != first_view['group'])
        with pytest.raises(StudyError, match='group_binding_immutable'):
            study.create({'consent': True, 'domain': 'pong', 'mode': 'test',
                          'participant_id': 'v5-user-0', 'group': different},
                         token=first_token, admin=True)
        with study.db.transaction() as db:
            db.execute('UPDATE pl3_policy_bindings SET variant=? WHERE instance_id=?',
                       ('raw' if first_view['group'] in ('A', 'B') else 'regularized',
                        first_view['id']))
        with pytest.raises(StudyError, match='policy_binding_mismatch'):
            study.view(first_token, first_view['id'])
        # Exercise pilot allocation under a synthetic accepted release. The
        # separate release-gate test keeps real admission closed.
        monkeypatch.setattr(Store, 'ready', property(lambda self: True))
        monkeypatch.setattr(store_module, 'all_policies_accepted', lambda: True)
        monkeypatch.setattr(Store, '_release_policies_match', staticmethod(lambda db: True))
        pilot_views = [study.create({'consent': True, 'domain': 'pong', 'mode': 'pilot',
                        'participant_id': f'pilot-user-{i}'})[1] for i in range(9)]
        assert {group: sum(view['group'] == group for view in pilot_views)
                for group in ('A', 'B', 'C')} == {'A': 3, 'B': 3, 'C': 3}
        with study.db.transaction(read_only=True) as db:
            pilot_assignments = db.all("SELECT scenario_seed,group_code FROM pl3_instances WHERE mode='pilot'")
            sources = db.all("SELECT assignment_source FROM pl3_enrollments e JOIN pl3_instances i ON i.id=e.instance_id WHERE i.mode='pilot'")
        assert all(row['assignment_source'] == 'randomized_balanced' for row in sources)
        for seed in (700, 701, 702):
            assert {group: sum(row['group_code'] == group and row['scenario_seed'] == seed
                               for row in pilot_assignments)
                    for group in ('A', 'B', 'C')} == {'A': 1, 'B': 1, 'C': 1}
    finally:
        study.db.close()


def test_pilot_rejects_self_selected_group(tmp_path, monkeypatch):
    monkeypatch.setattr(Store, 'ready', property(lambda self: True))
    study = Store(Settings(database=str(tmp_path / 'study.sqlite3')))
    try:
        with pytest.raises(StudyError, match='group_override_forbidden'):
            study.create({'consent': True, 'domain': 'pong', 'mode': 'pilot',
                          'participant_id': 'test-pilot', 'group': 'A'})
    finally:
        study.db.close()


def test_pilot_waits_for_persisted_release_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, 'all_policies_accepted', lambda: True)
    current_policies = {'pong': {'raw': {'actor_sha256': 'a'*64}}}
    monkeypatch.setattr(store_module, 'policy_manifest', lambda: current_policies)
    settings = Settings(database=str(tmp_path / 'study.sqlite3'), persistent=True,
                        admin_token='researcher', mode='pilot', llm_base_url='https://example.invalid',
                        llm_model='fixture', verified=True)
    study = Store(settings)
    try:
        assert not study.ready
        with study.db.transaction() as db:
            db.execute('INSERT INTO pl3_releases VALUES(?,?,?)',
                       (store_module.RELEASE_ID, json.dumps({
                           'policy_assets_accepted': True, 'policies': current_policies}), 1.0))
        assert study.ready
        current_policies['pong']['raw']['actor_sha256'] = 'b'*64
        assert not study.ready
    finally:
        study.db.close()


def test_server_prewarms_each_bound_ai_pair(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(server_module, 'all_policies_accepted', lambda: True)
    monkeypatch.setattr(server_module, 'demonstration',
                        lambda domain, ident: calls.append((domain, ident)))
    service = server_module.make_server(
        Settings(database=str(tmp_path / 'study.sqlite3')), port=0, explainer=object())
    try:
        assert set(calls) == {(domain, f'{domain}-{variant}-v5')
                              for domain in ('warehouse', 'pong', 'kitchen')
                              for variant in ('raw', 'regularized')}
    finally:
        service.server_close()


def test_kitchen_demo_precedes_neutral_practice_and_formal_takeover(tmp_path, monkeypatch):
    class FakeKitchen(_FakeEngine):
        def __init__(self):
            self.started_with = None

        def rule_metadata(self, policy_id):
            return {'policy_id': policy_id}

        def legal_actions(self, state):
            return ['wait']

        def initial_state(self, seed, task, policy_id):
            self.started_with = policy_id
            return {'domain': 'kitchen', 'policy_id': policy_id,
                    'policy_sha256': 'a'*64,
                    'role_sha256': {'human': '1'*64, 'ai': '2'*64},
                    'turn': 0, 'terminal': False, 'max_turns': 1}

        def public_state(self, state):
            return {**state, 'score': {'task_score': 0}}

        def decide(self, state):
            return {'action': 'wait', 'policy_id': state['policy_id'], 'model_sha256': 'a'*64}

        def step(self, state, human_action, decision):
            assert human_action == 'wait' and decision == self.decide(state)
            return {**state, 'turn': state['turn'] + 1, 'terminal': True}

        def score(self, state):
            return {'task_score': 0}

    fake = FakeKitchen()
    monkeypatch.setattr(store_module, 'engine', lambda domain: fake)
    monkeypatch.setattr(store_module, 'demonstration', lambda domain, policy_id: {
        'captions': [{'index': 0, 'en': 'NN pair', 'zh': '神经网络组合'}],
        'frames': [{'domain': 'kitchen', 'turn': 0, 'score': {'task_score': 0}}]})
    monkeypatch.setattr(store_module, 'scenario_config', lambda domain: {
        'heldout_seeds': [700], 'version': 'fixture'})
    binding = {'policy_id': 'kitchen-regularized-v5', 'domain': 'kitchen',
               'variant': 'regularized', 'actor_path': 'fixture.npz',
               'actor_sha256': 'a'*64, 'human_actor_sha256': '1'*64,
               'ai_actor_sha256': '2'*64, 'program_hashes': {}}
    monkeypatch.setattr(store_module, 'binding_for_group', lambda domain, group: binding)
    monkeypatch.setattr(store_module, 'verify_binding', lambda saved: saved)
    study = Store(Settings(database=str(tmp_path / 'study.sqlite3')))
    try:
        token, view = study.create({'consent': True, 'domain': 'kitchen', 'mode': 'test',
                                    'participant_id': 'v5-kitchen-flow', 'group': 'A'}, admin=True)
        assert view['stage'] == 'demo' and view['demo']['captions'][0]['en'] == 'NN pair'
        def command(kind, current, **extra):
            return study.command(token, kind, {'instance_id': current['instance_id'],
                 'revision': current['revision'], 'command_id': uuid.uuid4().hex, **extra})
        view = command('demo_skip', view)
        assert view['stage'] == 'practice' and view['tutorial']
        view = command('practice_skip', view)
        assert view['stage'] == 'task1'
        assert fake.started_with == 'kitchen-regularized-v5'
        assert view['state']['policy_id'] == fake.started_with
        recorded = study.frame(token, view['instance_id'], view['run_id'], 0)
        assert recorded['state']['policy_id'] == fake.started_with
        with study.db.transaction() as db:
            original = db.one('SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=0',
                              (view['run_id'],))['decision_json']
            db.execute('UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=0',
                (json.dumps({'action': 'left', 'policy_id': fake.started_with,
                             'model_sha256': 'a'*64}), view['run_id']))
        with pytest.raises(StudyError, match='recorded_decision_mismatch'):
            study.frame(token, view['instance_id'], view['run_id'], 0)
        with study.db.transaction() as db:
            db.execute('UPDATE pl3_frames SET decision_json=? WHERE run_id=? AND turn=0',
                       (original, view['run_id']))
        view = command('action', view, run_id=view['run_id'], turn=0, action='wait')
        assert view['run_status'] == 'completed'
        with study.db.transaction() as db:
            row = db.one('SELECT state_json FROM pl3_frames WHERE run_id=? AND turn=1',
                         (view['run_id'],))
            corrupt = json.loads(row['state_json'])
            corrupt['max_turns'] = 2
            db.execute('UPDATE pl3_frames SET state_json=?,public_json=? WHERE run_id=? AND turn=1',
                       (json.dumps(corrupt), json.dumps(fake.public_state(corrupt)), view['run_id']))
        with pytest.raises(StudyError, match='recorded_transition_mismatch'):
            study.frame(token, view['instance_id'], view['run_id'], 1)
    finally:
        study.db.close()
