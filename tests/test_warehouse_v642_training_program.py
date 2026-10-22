"""New grammar numerical refit only; no candidate NN updates or new scenes."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import pytest
from backend.training import warehouse_v642_training_program as t,warehouse_v642_program as language

class NoSolves:
    def check(self):pass
    def solve(self,*args,**kwargs):pytest.fail('qualification-only must not fit or solve LP')

@pytest.fixture
def refined(tmp_path):
    spec=importlib.util.spec_from_file_location('v642_training_synthetic_legacy',Path(__file__).with_name('test_warehouse_v630_training_program.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old,parts=helper.refined.__wrapped__(tmp_path)
    return language.import_previous(old,12),parts


def test_new_target_binding_requalifies_without_mutating_anchor_or_path(refined):
    p,parts=refined;before=deepcopy(p.to_dict());target='b'*64
    out,m=t.complete_program(p,parts,NoSolves(),refit=False,target_actor_role_sha256=target)
    assert p.to_dict()==before and t.language_lock(p)==t.language_lock(out)
    assert out.target_actor_role_sha256==out.report['target_actor_role_sha256']==target
    assert out.actor_role_sha256==p.actor_role_sha256!=target and out.maximum_depth==12
    assert out.report['qualification_only_no_leaf_fitting'] and t.protected_indices(out)==frozenset({0,1})
    assert language.Program.from_dict(out.to_dict(),target).to_dict()==out.to_dict()
    with pytest.raises(RuntimeError):language.Program.from_dict(out.to_dict(),p.actor_role_sha256)


def test_protected_payload_stays_fixed_but_current_NN_qualification_does_not(refined):
    p,parts=refined;child=next(iter(t.protected_indices(p)));a=parts['eligibility'];take=p.leaf_indices(a['x'])==child
    a['p'][take]=[.8,.05,.05,.05,.05]
    out,_=t.complete_program(p,parts,NoSolves(),refit=False,target_actor_role_sha256='c'*64)
    assert child not in out.feedback_leaves and child not in out.explanation_leaves
    assert t.payload(out.leaves[child])==t.payload(p.leaves[child]) and t.language_lock(p)==t.language_lock(out)


def test_fixed_support_fit_calibration_and_new_target_proofs(refined,tmp_path):
    from backend.training.warehouse_v626_budget import Budget
    from scripts.audit_warehouse_v626 import independent_dual
    p,parts=refined;before=deepcopy(p.to_dict());budget=Budget(tmp_path/'checkpoint.json',dict(synthetic_numerical_only=True),t.limits(p),30);budget.phase='final'
    try:
        out,_=t.complete_program(p,parts,budget,refit=True,target_actor_role_sha256='d'*64)
        assert p.to_dict()==before and t.language_lock(p)==t.language_lock(out)
        assert budget.counts['leaf']>0 and budget.counts['node']==0 and not budget.pending
        assert all(t.payload(out.leaves[i])==t.payload(p.leaves[i]) for i in t.protected_indices(p))
        assert any(t.payload(out.leaves[i])!=t.payload(p.leaves[i]) for i in set(range(len(p.leaves)))-t.protected_indices(p))
        for i,leaf in enumerate(out.leaves):independent_dual(leaf,t.region(out.structure,i,256,12),out.certificates()[i])
    finally:budget.session.close()


@pytest.mark.parametrize('fault',['scene_overlap','bad_probability','illegal_sample'])
def test_invalid_parts_stop_before_fitting(refined,fault):
    p,parts=refined
    if fault=='scene_overlap':parts['validation']['rows'][0]['base_seed']=parts['fit']['rows'][0]['base_seed']
    elif fault=='bad_probability':parts['fit']['p'][0,0]=float('nan')
    else:parts['fit']['sampled'][0]=5
    with pytest.raises((RuntimeError,ValueError)):t.complete_program(p,parts,NoSolves())


def test_real_locked_depth11_program_fixed_support_refit_and_roundtrip(tmp_path):
    """Spent development arrays only: integration evidence, not a new trial."""
    from experiments.warehouse import v630_protocol as v,v628_data as data
    from backend.training.warehouse_v626_budget import Budget
    from scripts.audit_warehouse_v642_program import audited_program
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    ref=v.pointer(v.ROOT/'output/warehouse_v642/warehouse-v642-depth9-depth12-01/depth-12-fold-0-robot_2/program.json')
    encoded=v.bound(ref);p=language.Program.from_dict(encoded,encoded['target_actor_role_sha256']);before=deepcopy(p.to_dict())
    assert language.structure_info(p.structure,256,12)[1]['depth']==11
    parts=data.load_parts(dp,0,'robot_2');budget=Budget(tmp_path/'checkpoint.json',dict(spent_data_numerical_refit_only=True,source=ref),t.limits(p),1200);budget.phase='final'
    try:
        out,measurements=t.complete_program(p,parts,budget,refit=True,target_actor_role_sha256=p.target_actor_role_sha256)
        assert t.language_lock(out)==t.language_lock(p) and p.to_dict()==before
        assert all(t.payload(out.leaves[i])==t.payload(p.leaves[i]) for i in t.protected_indices(p))
        assert budget.counts['node']==0 and budget.counts['leaf']>0 and not budget.pending
        assert language.Program.from_dict(out.to_dict(),out.target_actor_role_sha256).to_dict()==out.to_dict()
        audited_program(out,parts,{s:{k:measurements[s][k] for k in ('feedback','explanation')} for s in parts})
    finally:budget.session.close()


def test_refit_requires_explicit_fit_NN_identity_before_any_solve(refined):
    p,parts=refined
    with pytest.raises(RuntimeError,match='explicit complete-NN'):
        t.complete_program(p,parts,NoSolves(),refit=True)
