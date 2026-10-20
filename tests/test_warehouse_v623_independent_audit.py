"""Reject false scientific receipts without loading cohorts or running fits."""
from copy import deepcopy

import pytest

from scripts import audit_warehouse_v623_refinement as audit

ROLE='robot_1'
INITIAL='a'*64
TARGET='b'*64


def search_receipt():
    return {'schema':'warehouse-v623-fit-search.v1',
        'actor_role_sha256':audit.old.ROLE_SHA256[ROLE],
        'initial_structure_sha256':INITIAL,'fit_target_sha256':TARGET,
        'budget':dict(audit.BUDGET),'search_uses_fit_only':True,
        'calibration_eligibility_validation_used':False,'complete_incumbent':True,
        'execution_action_override':False,'zero_actor_updates':True,'zero_environment_resets':True,
        'new_leaf_calls':1,'leaf_api_attempts':[{'call':1,'cache_miss':True,
            'path_legal_actions':[1,4],'actual_convex_solver_calls':2,
            'rows':20,'rows_sha256':'c'*64,'success':True}],
        'actual_convex_solver_calls':2,'failed_leaf_api_attempts':0,
        'reserved_final_leaf_calls':32,'maximum_new_leaf_calls_including_final_reserve':20000,
        'new_leaf_calls_remaining':19967,'elapsed_seconds':4800.04,
        'rounds':[],'stop_reason':'wall_budget'}


def test_actual_leaf_and_solver_ledger_is_reconciled():
    search=search_receipt()
    result=audit.search_ledger(search,ROLE,INITIAL,TARGET)
    assert result['new_leaf_calls']==1
    assert result['convex_solver_calls']==2
    for mutate in (
        lambda r:r.update(new_leaf_calls=2),
        lambda r:r.update(actual_convex_solver_calls=3),
        lambda r:r['leaf_api_attempts'][0].update(call=2),
        lambda r:r['leaf_api_attempts'][0].update(actual_convex_solver_calls=1),
        lambda r:r.update(new_leaf_calls_remaining=0),
        lambda r:r.update(elapsed_seconds=4805.01),
    ):
        bad=deepcopy(search);mutate(bad)
        with pytest.raises(RuntimeError):
            audit.search_ledger(bad,ROLE,INITIAL,TARGET)


def audited_rows():
    return [{'job_id':j['job_id'],'status':'completed','passed':False,
        'explanation_coverage':.35,'expected_feedback_coverage':.33,
        'result':{'path':'/evidence/'+j['job_id'],'sha256':'d'*64}}
        for j in audit.v.jobs()]


def test_summary_recomputed_from_six_real_gates_rejects_self_asserted_pass():
    rows=audited_rows();expected=audit.reconstruct_summary('S1','e'*64,rows)
    assert expected['all_six_pass'] is False and expected['next_stage']=='S2'
    audit.verify_summary(deepcopy(expected),expected)
    corruptions=[]
    fake=deepcopy(expected);fake.update(all_six_pass=True,next_stage='fresh_confirmation');corruptions.append(fake)
    fake=deepcopy(expected);fake['role_folds']=[];corruptions.append(fake)
    fake=deepcopy(expected);fake['role_folds'][0]['result']['sha256']='f'*64;corruptions.append(fake)
    for fake in corruptions:
        with pytest.raises(RuntimeError):audit.verify_summary(fake,expected)
    with pytest.raises(RuntimeError):audit.reconstruct_summary('S1','e'*64,[])
    duplicate=deepcopy(rows);duplicate[1]=deepcopy(duplicate[0])
    with pytest.raises(RuntimeError):audit.reconstruct_summary('S1','e'*64,duplicate)


def test_all_six_gates_are_required_and_failed_s2_closes_candidate():
    rows=audited_rows()
    for r in rows:r['passed']=True
    assert audit.reconstruct_summary('S1','e'*64,rows)['next_stage']=='fresh_confirmation'
    rows[-1].update(status='scientific_budget_exhausted',passed=False)
    assert audit.reconstruct_summary('S2','e'*64,rows)['next_stage']=='closed_failed_program_precheck'
    rows[-1]['passed']=True
    with pytest.raises(RuntimeError):audit.reconstruct_summary('S2','e'*64,rows)


def timeout_receipt():
    return {'status':'scientific_budget_exhausted','gate':{'passed':False},'elapsed_seconds':5400.1,
        'failure':{'status':'budget_exhausted','scientific_failure':True,
            'actor_role_sha256':audit.old.ROLE_SHA256[ROLE],'failure_stage':'calibration',
            'leaf_optimization_calls':2,'convex_solver_calls':4,
            'fit_budget':{'prior_leaf_optimization_calls':1,
                'actual_final_leaf_optimization_calls':2,'actual_total_leaf_optimization_calls':3}}}


def test_timeout_requires_evidence_and_cannot_carry_usable_program():
    result=timeout_receipt();search=search_receipt()
    assert audit.validate_timeout_failure(result,search,ROLE)['new_leaf_calls']==1
    for mutate in (
        lambda r:r.pop('failure'),
        lambda r:r.update(elapsed_seconds=5399.),
        lambda r:r['failure'].update(scientific_failure=False),
        lambda r:r['failure'].update(actor_role_sha256='f'*64),
        lambda r:r['failure'].update(failure_stage=''),
        lambda r:r['failure']['fit_budget'].update(actual_total_leaf_optimization_calls=2),
        lambda r:r.update(metrics={}),
        lambda r:r.update(program={'path':'/unsafe/program'}),
        lambda r:r.update(gate={'passed':True}),
    ):
        bad=deepcopy(result);mutate(bad)
        with pytest.raises(RuntimeError):audit.validate_timeout_failure(bad,search,ROLE)
