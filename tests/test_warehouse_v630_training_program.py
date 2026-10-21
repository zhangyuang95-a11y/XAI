"""Numerical surrogate tests only; no actual NN training or new scenes."""
from copy import deepcopy
from pathlib import Path
import importlib.util

import numpy as np
import pytest

from backend.training import warehouse_v630_training_program as t
from backend.training.warehouse_v630_program import from_previous,split_identical
from backend.training.warehouse_v628_program import protected_indices as exact_path_indices
from backend.training.warehouse_v6_actor import FEATURE_NAMES


@pytest.fixture
def refined(tmp_path):
    spec=importlib.util.spec_from_file_location('v630_refit_synthetic_helper',
                Path(__file__).with_name('test_warehouse_v629.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source,parts=helper.fixture(tmp_path)
    source.maximum_leaves=256
    base=from_previous(source)
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_column')]=1.
    program,_=split_identical(base,0,w,-.5)
    for a in parts.values():
        a['p']=np.tile(np.array([.05,.05,.05,.05,.8],np.float32),(len(a['x']),1))
    return program,parts


class NoSolves:
    def check(self):pass
    def solve(self,*args,**kwargs):pytest.fail('qualification only may not fit or prove via solver')


def test_protected_region_children_are_not_lost_by_exact_path_matching(refined):
    p,parts=refined
    assert len(t.protected_indices(p))==2 and exact_path_indices(p)==frozenset()
    assert t.limits(p)==dict(leaf=16,node=0,proof=104)
    before=deepcopy(p.to_dict())
    out,m=t.complete_program(p,parts,NoSolves(),refit=False)
    assert p.to_dict()==before
    assert out.report['protected_refit_and_calibration_skipped']==[0,1]
    assert out.report['qualification_only_no_leaf_fitting'] is True
    assert all(np.array_equal(a,b) for a,b in zip(p.distributions(parts['validation']['x'],parts['validation']['m']),
                    out.distributions(parts['validation']['x'],parts['validation']['m']),strict=True))


def test_protected_child_keeps_output_but_cannot_inherit_current_NN_qualification(refined):
    p,parts=refined
    a=parts['eligibility'];child=next(iter(t.protected_indices(p)))
    take=p.leaf_indices(a['x'])==child
    assert take.any()
    a['p'][take]=[.8,.05,.05,.05,.05]
    out,_=t.complete_program(p,parts,NoSolves(),refit=False)
    assert child not in out.feedback_leaves and child not in out.explanation_leaves
    assert t.language_lock(p)==t.language_lock(out)


def test_actual_fixed_support_solve_and_calibration_skip_all_protected_children(refined,tmp_path):
    from backend.training.warehouse_v626_budget import Budget
    from scripts.audit_warehouse_v626 import independent_dual
    p,parts=refined;before=deepcopy(p.to_dict());protected=t.protected_indices(p)
    budget=Budget(tmp_path/'checkpoint.json',dict(synthetic_surrogate_refit_only=True),t.limits(p),30)
    budget.phase='final'
    try:
        out,m=t.complete_program(p,parts,budget)
        assert budget.counts['leaf']>0 and budget.counts['node']==0
        assert t.language_lock(out)==t.language_lock(p) and p.to_dict()==before
        assert out.report['qualification_only_no_leaf_fitting'] is False
        assert all(t.payload(out.leaves[i])==t.payload(p.leaves[i]) for i in protected)
        assert any(t.payload(out.leaves[i])!=t.payload(p.leaves[i]) for i in set(range(len(p.leaves)))-protected)
        for i,leaf in enumerate(out.leaves):
            independent_dual(leaf,t.region(out.structure,i,256),out.certificates()[i])
        assert not budget.pending
    finally:budget.session.close()


@pytest.mark.parametrize('fault',['scene_overlap','invalid_probability','illegal_sample'])
def test_bad_training_partitions_fail_before_any_optimization(refined,fault):
    p,parts=refined
    if fault=='scene_overlap':parts['validation']['rows'][0]['base_seed']=parts['fit']['rows'][0]['base_seed']
    elif fault=='invalid_probability':parts['fit']['p'][0,0]=np.nan
    else:parts['fit']['sampled'][0]=5
    with pytest.raises((RuntimeError,ValueError)):t.complete_program(p,parts,NoSolves())
