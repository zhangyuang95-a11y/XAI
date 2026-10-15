"""Still-supported v5 binding and historical data compatibility checks.

Obsolete v5 active-release assertions remain in
legacy_task2/archived_test_study_v5_policy_bindings.py.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from study_v3 import policies
from study_v3.config import Settings
from study_v3.store import Store, StudyError
from study_v3 import store as store_module


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
