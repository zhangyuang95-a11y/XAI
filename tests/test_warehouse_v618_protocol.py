"""Receipts distinguish immutable diagnostics from program qualifications."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from experiments.warehouse import protocol as p
from tests.warehouse_v6_contract_fixture import bc_receipt


def _diagnostic(stage, *, support_locked=True):
    actor = {role: sha256((role+':actor').encode()).hexdigest() for role in p.ROLES}
    tree = {role: sha256((role+':tree').encode()).hexdigest() for role in p.ROLES}
    return {
        'stage': stage, 'role_tree_sha256': tree,
        'role_support_sha256': {role: sha256((role+':support').encode()).hexdigest()
                                for role in p.ROLES},
        'actor_role_sha256_before': actor,
        'actor_role_sha256_after': dict(actor),
        'route_parity_passed': True, 'guard_invariants_passed': True,
        'core_qualification_recomputed': False, 'actor_rebuilt': False,
        'auxiliary_ce_updates': 0, 'support_locked': support_locked,
        'roles': {role: {
            'topology_sha256': tree[role], 'training_scene_count': 100,
            'parity': {'routing_mismatches': 0, 'maximum_logit_abs_difference': 0.,
                       'role_sha256': actor[role]},
        } for role in p.ROLES},
    }


def _bc():
    seed, source = p.DEVELOPMENT_PAIR_SEEDS[0], 'f'*64
    result = bc_receipt(seed, source)
    result['bc_topology_refresh_history'] = [
        _diagnostic(stage, support_locked=index == len(p.bc_topology_refresh_stages())-1)
        for index, stage in enumerate(p.bc_topology_refresh_stages())]
    final = result['final_bc_internal_precheck']
    for role in p.ROLES:
        item = final['roles'][role]
        program = sha256((role+':program').encode()).hexdigest()
        item.update({
            'program_sha256': program, 'qualification_changes_actor': False,
            'qualified_validation_mass': .4, 'expected_feedback_actionable_mass': .35,
            'program_qualification': {
                'method': p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD,
                'validation_passed': True, 'program_sha256': program,
                'frozen_role_sha256': result['pretrained_role_sha256'][role],
            },
        })
        for key in ('actor_core_qualification', 'installed_core_validation_mass',
                    'installed_core_expected_actionable_mass'):
            item.pop(key, None)
    _resign(result)
    return seed, source, result


def _resign(report):
    report['final_bc_internal_precheck_receipt']['sha256'] = sha256(
        (json.dumps(report['final_bc_internal_precheck'], sort_keys=True,
                    allow_nan=False)+'\n').encode()).hexdigest()


def test_bc_diagnostics_do_not_require_forward_certificates_and_final_lock_is_required():
    stages = p.bc_topology_refresh_stages()
    history = [_diagnostic(stage, support_locked=index == len(stages)-1)
               for index, stage in enumerate(stages)]
    assert p.bc_topology_refresh_history_valid(history)
    history[-1]['support_locked'] = False
    assert not p.bc_topology_refresh_history_valid(history)


@pytest.mark.parametrize('path,value', [
    (('actor_role_sha256_after', 'robot_1'), '1'*64),
    (('roles', 'robot_1', 'parity', 'role_sha256'), '1'*64),
    (('roles', 'robot_2', 'parity', 'maximum_logit_abs_difference'), .001),
    (('guard_invariants_passed',), False),
    (('core_qualification_recomputed',), True),
    (('actor_rebuilt',), True),
    (('auxiliary_ce_updates',), 1),
    (('support_locked',), False),
])
def test_ppo_diagnostic_rejects_actor_mutation_and_hidden_supervised_updates(path, value):
    history = [{**_diagnostic(f'ppo_{step}'), 'joint_step': step}
               for step in p.topology_refresh_steps(200_000)]
    assert p.topology_refresh_history_valid(history, joint_steps=200_000)
    target = history[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not p.topology_refresh_history_valid(history, joint_steps=200_000)


def test_topology_freeze_allows_leaf_learning_but_rejects_support_drift():
    step = p.topology_frozen_at_step(200_000)
    diagnostic = _diagnostic(f'ppo_freeze_{step}')
    support = {role: sha256((role+':support').encode()).hexdigest() for role in p.ROLES}
    report = {
        'topology_frozen_at_step': step,
        'frozen_topology_role_sha256': diagnostic['role_tree_sha256'],
        'final_topology_role_sha256': dict(diagnostic['role_tree_sha256']),
        'frozen_support_role_sha256': support, 'final_support_role_sha256': dict(support),
        'frozen_actor_role_sha256': diagnostic['actor_role_sha256_after'],
        'topology_freeze_qualification': diagnostic, 'topology_updates_after_freeze': 0,
        'core_leaf_freezing_used': False, 'leaf_outputs_remain_trainable': True,
        # Historical fields do not impose leaf-output freezing on v6.18.
        'frozen_core_leaf_logits_sha256': {'robot_1': 'a'*64, 'robot_2': 'b'*64},
        'final_core_leaf_logits_sha256': {'robot_1': 'c'*64, 'robot_2': 'd'*64},
    }
    assert p.frozen_topology_valid(report)
    report['final_support_role_sha256']['robot_2'] = 'a'*64
    assert not p.frozen_topology_valid(report)


def test_bc_program_evidence_is_bound_and_never_installed_into_actor():
    seed, source, report = _bc()
    assert p.bc_report_valid(report, seed, source)
    for key, value in (
            ('qualified_validation_mass', .34),
            ('expected_feedback_actionable_mass', .29),
            ('qualification_changes_actor', True),
            ('program_sha256', 'a'*64)):
        changed = deepcopy(report)
        changed['final_bc_internal_precheck']['roles']['robot_1'][key] = value
        _resign(changed)
        assert not p.bc_report_valid(changed, seed, source)
    changed = deepcopy(report)
    changed.pop('quality_screen_seeds')
    changed['final_bc_internal_precheck'].pop('quality_screen_seeds')
    _resign(changed)
    assert not p.bc_report_valid(changed, seed, source)


def test_program_illegal_action_count_cannot_be_rounded_down():
    metrics = {'coverage': .5, 'actionable_coverage': .4, 'agreement': .99,
               'kl': .01, 'rows': 100, 'actionable_rows': 40, 'illegal_count': 0,
               'whole_tree_raw_illegal_count': 0, 'whole_tree_path_uncertified_count': 0,
               'path_uncertified_count': 0}
    assert p.program_pass(metrics, purpose='feedback', development=True)
    assert not p.program_pass({**metrics, 'illegal_count': .1},
                              purpose='feedback', development=True)
