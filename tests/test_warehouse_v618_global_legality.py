"""A selective legal region cannot hide illegal raw outputs elsewhere."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from backend.training.warehouse_v6_actor import FEATURE_DIM, FrozenWarehouseV6Actor
from backend.training.warehouse_v6_program import (
    AXIS_TREE_METHOD, FEEDBACK_AGREEMENT, INDEPENDENT_SELECTIVE_METHOD,
    WarehouseV6Program, _eligible, _metrics, _passes,
    audit_frozen_program, current_actor_fidelity,
)
from experiments.warehouse import protocol as p
from scripts.diagnose_warehouse_v618_program_feasibility import metric_gates
from tests.test_warehouse_v618_actor import _rows
from tests.test_warehouse_v618_training import _wait_model
from env.warehouse.public_action_features_v617 import TREE_FEATURE_NAMES


def _example(outside_action=1, *, method=INDEPENDENT_SELECTIVE_METHOD):
    actor = FrozenWarehouseV6Actor.from_model(_wait_model())
    obs, masks, _ = _rows()
    rows = [{'obs': obs[index%3, 0], 'mask': masks[index%3, 0],
             'episode': str(986000+index), 'base_seed': 986000+index,
             'sampled_action': 'WAIT'} for index in range(120)]
    weights = np.zeros((1, FEATURE_DIM), np.float32)
    weights[0, TREE_FEATURE_NAMES.index('self_row')] = 1.
    logits = np.zeros((2, 5), np.float32)
    logits[0, 4] = 5.
    logits[1, outside_action] = 5.
    scope = {'whole_tree_raw_illegal_count': 0,
             'whole_tree_path_uncertified_count': 0, 'path_uncertified_count': 0}
    report = {'fit_method': method, 'max_depth': 1, 'leaf_count': 2,
              'split_episode_ids': {part: [] for part in ('fit','calibration','eligibility','validation')},
              'split_scenario_seed_ids': {part: [] for part in ('fit','calibration','eligibility','validation')},
              'validation': {'feedback_passed': True, 'explanation_passed': True,
                             'feedback': dict(scope), 'explanation': dict(scope)}}
    program = WarehouseV6Program('robot_1', actor.role_sha256['robot_1'], 0,
        weights, np.array([-.5], np.float32), np.array([-1], np.int64),
        np.array([-2], np.int64), logits, frozenset({0}), frozenset({0}), report)
    return actor, program, rows


def test_local_eligibility_stays_diagnostic_but_global_raw_illegal_fails_every_gate():
    actor, program, rows = _example(outside_action=1)  # DOWN is illegal in the other leaf.
    metric = _metrics(program, actor.roles['robot_1'], rows, {0})
    assert metric['coverage'] == pytest.approx(2/3)
    assert metric['illegal_count'] == 0 and metric['path_uncertified_count'] == 0
    assert metric['whole_tree_raw_illegal_count'] == 40
    assert metric['whole_tree_path_uncertified_count'] == 40
    assert _eligible(metric, FEEDBACK_AGREEMENT)  # Never let a global failure erase local evidence.
    assert not _passes(metric, FEEDBACK_AGREEMENT, development=True, require_whole_tree_zero=True)
    assert not p.program_pass(metric, purpose='feedback', development=True)
    assert not current_actor_fidelity(program, actor, rows)['passed']
    audit = audit_frozen_program(program, actor, rows)
    assert not audit['passed'] and audit['feedback']['whole_tree_raw_illegal_count'] == 40
    gates = metric_gates(metric, metric)
    assert not gates['bc_precheck_35_30']['passed']
    assert not gates['development_30_realized']['passed']


def test_zero_qualified_coverage_still_counts_all_raw_tree_failures():
    actor, program, rows = _example()
    metric = _metrics(program, actor.roles['robot_1'], rows, set())
    assert metric['rows'] == 0 and metric['illegal_count'] == 0
    assert metric['whole_tree_raw_illegal_count'] == 40
    assert metric['whole_tree_path_uncertified_count'] == 40


@pytest.mark.parametrize('outside_action,expected_unproved', [(4, 0), (0, 40)])
def test_global_legal_tree_passes_without_requiring_unqualified_path_proofs(
        outside_action, expected_unproved):
    # WAIT is universally proven; UP in the unqualified final leaf is legal
    # on these public states but lacks a legal_UP path. That extra proof was
    # never a global gate in the accepted plan.
    actor, program, rows = _example(outside_action=outside_action)
    metric = _metrics(program, actor.roles['robot_1'], rows, {0})
    assert metric['whole_tree_raw_illegal_count'] == 0
    assert metric['whole_tree_path_uncertified_count'] == expected_unproved
    assert _passes(metric, FEEDBACK_AGREEMENT, development=True, require_whole_tree_zero=True)
    assert p.program_pass(metric, purpose='feedback', development=True)
    assert current_actor_fidelity(program, actor, rows)['passed']
    assert audit_frozen_program(program, actor, rows)['passed']


def test_qualified_movement_without_path_proof_still_fails():
    actor, program, rows = _example(outside_action=0)
    metric = _metrics(program, actor.roles['robot_1'], rows, {0, 1})
    assert metric['whole_tree_raw_illegal_count'] == 0
    assert metric['path_uncertified_count'] == 40
    assert not _passes(metric, FEEDBACK_AGREEMENT, require_whole_tree_zero=True)
    assert not p.program_pass(metric, purpose='feedback', development=False)


def test_missing_scope_fields_cannot_inherit_old_true_flags_but_legacy_remains_readable():
    actor, program, rows = _example(outside_action=4)
    metric = _metrics(program, actor.roles['robot_1'], rows, {0})
    for field in ('whole_tree_raw_illegal_count', 'whole_tree_path_uncertified_count'):
        incomplete = {key: value for key, value in metric.items() if key != field}
        assert not p.program_pass(incomplete, purpose='feedback', development=True)
        assert not _passes(incomplete, FEEDBACK_AGREEMENT, require_whole_tree_zero=True)
    program.report['validation']['feedback'].pop('whole_tree_raw_illegal_count')
    assert not current_actor_fidelity(program, actor, rows)['passed']
    assert program.covered(rows[0], actor_action='WAIT', purpose='feedback') is None
    # The new counters do not rewrite historical selected-region receipts.
    _, legacy, old_rows = _example(outside_action=1, method=AXIS_TREE_METHOD)
    old = _metrics(legacy, actor.roles['robot_1'], old_rows, {0})
    assert 'whole_tree_raw_illegal_count' not in old
    assert _passes(old, FEEDBACK_AGREEMENT, development=True)
    assert p.program_pass(old, purpose='feedback', development=True, method=AXIS_TREE_METHOD)
