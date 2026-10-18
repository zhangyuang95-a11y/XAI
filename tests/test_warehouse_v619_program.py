"""New constrained program API; old program semantics and NN bytes are retained."""
from copy import deepcopy

import numpy as np
import pytest

from backend.training.warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES
from backend.training.warehouse_v6_program import (
    INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD,
    PATH_CONSTRAINED_PROGRAM_VERSION, WarehouseV6Program, certified_program_actions,
    current_actor_fidelity, fit_program, program_structure_sha256,
)
from tests.test_warehouse_v618_program import _soft_actor, _parts


def _fixed(empty=False):
    n = int(empty)
    weights = np.zeros((n, FEATURE_DIM), np.float32)
    if empty:
        weights[0, FEATURE_NAMES.index('self_row')] = 1
    return {'root_ref': 0 if empty else -1, 'split_weights': weights,
            'split_bias': np.array([-1.5] if empty else [], np.float32),
            'left_refs': np.array([-1] if empty else [], np.int64),
            'right_refs': np.array([-2] if empty else [], np.int64), 'leaf_count': n+1}


def test_fixed_structure_full_distribution_fit_calibrate_roundtrip_and_actor_side_effects():
    actor, target = _soft_actor()
    before = {key: value.copy() for key,value in actor.roles['robot_1'].weights.items()}
    structure = _fixed(empty=True)
    digest = program_structure_sha256(structure)
    program = fit_program('robot_1',actor,*_parts(),method=PATH_CONSTRAINED_METHOD,
                          fixed_structure=structure,fixed_structure_sha256=digest)
    assert program.report['validation']['development_feedback_passed']
    assert program.report['program_structure_sha256'] == digest
    assert program.report['fixed_structure_sha256'] == digest
    assert program.report['fit_leaf_row_counts'] == [240,0]
    assert 1 not in program.feedback_leaves | program.explanation_leaves
    assert program.report['constrained_leaf_fits']['1']['empty_leaf']
    assert program.report['constrained_leaf_calibration']['1']['temperature'] == 1
    assert program.report['validation']['feedback']['whole_tree_raw_illegal_count'] == 0
    assert program.report['validation']['feedback']['whole_tree_path_uncertified_count'] == 0
    for leaf, proof in certified_program_actions(program).items():
        assert program.leaf_logits[leaf].argmax() in proof
    np.testing.assert_allclose(list(program.lookup(_parts()[-1][0])['probabilities'].values()),target,atol=.003)
    restored = WarehouseV6Program.from_dict(program.to_dict(),actor_role_sha256=program.actor_role_sha256)
    assert restored.sha256() == program.sha256()
    assert restored.to_dict()['version'] == PATH_CONSTRAINED_PROGRAM_VERSION
    for key,value in before.items():
        np.testing.assert_array_equal(actor.roles['robot_1'].weights[key], value)


def test_unqualified_unvisited_leaf_cannot_lose_final_path_certificate_on_load():
    actor, _ = _soft_actor()
    structure = _fixed(empty=True)
    program = fit_program('robot_1',actor,*_parts(),method=PATH_CONSTRAINED_METHOD,
                          fixed_structure=structure,fixed_structure_sha256=program_structure_sha256(structure))
    data = program.to_dict()
    assert 1 not in data['feedback_leaves']+data['explanation_leaves']
    data['leaf_logits'][1] = [10,0,0,0,0]
    with pytest.raises(RuntimeError, match='certificate'):
        WarehouseV6Program.from_dict(data,actor_role_sha256=program.actor_role_sha256)


def test_fixed_structure_hash_and_version_mismatch_fail_before_modifying_actor():
    actor, _ = _soft_actor()
    with pytest.raises(ValueError, match='SHA mismatch'):
        fit_program('robot_1',actor,*_parts(),method=PATH_CONSTRAINED_METHOD,
                    fixed_structure=_fixed(),fixed_structure_sha256='0'*64)
    with pytest.raises(ValueError, match='require the new'):
        fit_program('robot_1',actor,*_parts(),method=INDEPENDENT_SELECTIVE_METHOD,
                    fixed_structure=_fixed(),fixed_structure_sha256=program_structure_sha256(_fixed()))


def test_p2_api_uses_new_tree_and_validation_never_selects_new_topology(monkeypatch):
    from backend.training import warehouse_v619_tree as tree
    real = tree.fit_legal_structure
    monkeypatch.setattr(tree,'fit_legal_structure',lambda *args,**kwargs:
        real(*args,**kwargs,config=tree.SearchConfig(initializations=1,max_rounds=1,
            max_leaf_optimization_calls=16,proposals_per_round=2,axis_features=1,
            semantic_candidates_per_node=1)))
    actor,_ = _soft_actor()
    parts = _parts()
    first = fit_program('robot_1',actor,*parts,method=PATH_CONSTRAINED_METHOD)
    changed = [{**row,'sampled_action':'RIGHT'} for row in parts[-1]]
    second = fit_program('robot_1',actor,*parts[:-1],changed,method=PATH_CONSTRAINED_METHOD)
    assert first.report['leaf_optimization_calls_outside_search'] == 0
    assert first.report['convex_solver_calls_outside_search'] == 0
    assert first.report['fit_tree'] == second.report['fit_tree']
    np.testing.assert_array_equal(first.leaf_logits,second.leaf_logits)
    assert first.feedback_leaves == second.feedback_leaves
    assert not second.report['validation']['feedback_passed']
