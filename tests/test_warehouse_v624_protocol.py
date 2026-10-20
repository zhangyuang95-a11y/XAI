"""Paired comparison fairness, real baseline pinning and fail-closed gates."""
from copy import deepcopy
import math
from pathlib import Path
import pytest
from experiments.warehouse import v624_protocol as v


def metrics(coverage=.5,expected=.43,kl=.1):
    row={'coverage':coverage,'expected_actionable_coverage':expected,'agreement':.99,'kl':kl,
         'whole_tree_raw_illegal_count':0,'whole_tree_path_uncertified_count':0,'illegal_count':0,'path_uncertified_count':0}
    return {'feedback':deepcopy(row),'explanation':deepcopy(row)}


def results():
    rows=[]
    for spec in v.jobs():
        m=metrics();rows.append({'job':spec,'status':'completed','leaf_count':32,
            'metrics':{'validation':m},'gate':v.program_gate(m)})
    return rows


def test_balanced_twelve_jobs_have_identical_role_fold_start_keys():
    jobs=v.jobs();assert len(jobs)==len({j['job_id'] for j in jobs})==12
    assert sum(jobs[i]['arm']=='P0' for i in range(0,12,2))==3
    for i in range(0,12,2):
        a,b=jobs[i:i+2]
        assert (a['fold'],a['role'])==(b['fold'],b['role']) and {a['arm'],b['arm']}=={'P0','P1'}


def test_pinned_complete_s1_and_closed_receipt_are_real_and_unchanged():
    baseline,initial=v.checked_baseline()
    assert baseline['finalization_receipt']['sha256']==v.PREVIOUS_RECEIPT_SHA
    assert baseline['evidence_manifest']['sha256']==v.PREVIOUS_MANIFEST_SHA
    assert len(initial)==6
    for refs in initial.values():
        result=v.bound(refs['result']);assert result['status']=='completed'
        assert result['structure']==refs['structure']
    assert len(v.recipe())==64


def test_missing_or_changed_asset_fails_closed(tmp_path):
    path=tmp_path/'model.json'
    with pytest.raises(RuntimeError):v.bound({'path':str(path),'sha256':'f'*64})
    path.write_text('{}');ref=v.pointer(path);assert v.bound(ref)=={}
    path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError):v.bound(ref)


def test_gate_handles_empty_qualification_and_nonfinite_values_as_scientific_failure():
    assert v.program_gate(metrics())['passed']
    empty=metrics(0,0,None);empty['feedback']['agreement']=0;empty['explanation']['agreement']=0
    assert not v.program_gate(empty)['passed']
    for value in (None,float('nan'),float('inf'),-.1,1.1):
        for purpose in ('feedback','explanation'):
            for key in ('coverage','agreement'):
                bad=metrics();bad[purpose][key]=value;assert not v.program_gate(bad)['passed']
    for value in (None,float('nan'),float('inf'),-.1):
        bad=metrics();bad['feedback']['kl']=value;assert not v.program_gate(bad)['passed']
    bad=metrics();bad['feedback']['whole_tree_raw_illegal_count']=1
    assert not v.program_gate(bad)['passed']


def test_only_complete_twelve_and_six_passing_units_can_select():
    rows=results();assert v.comparison(rows)['selected_arm']=='P0'
    for bad in (rows[:-1],rows+[deepcopy(rows[0])],[deepcopy(rows[0]),deepcopy(rows[0]),*deepcopy(rows[2:])]):
        with pytest.raises(RuntimeError):v.comparison(bad)
    for r in rows:
        if r['job']['arm']=='P1':r['metrics']['validation']=metrics(.51,.44);r['gate']=v.program_gate(r['metrics']['validation'])
    assert v.comparison(rows)['selected_arm']=='P1'
    next(r for r in rows if r['job']['arm']=='P1')['gate']['passed']=False
    with pytest.raises(RuntimeError):v.comparison(rows)


def test_ranking_uses_worst_unit_and_never_pools_failures():
    rows=results()
    for r in rows:
        m=metrics(.8,.7) if r['job']['arm']=='P1' else metrics()
        r['metrics']['validation']=m;r['gate']=v.program_gate(m)
    last=next(r for r in rows if r['job']['arm']=='P1');last['metrics']['validation']=metrics(.44,.43);last['gate']=v.program_gate(last['metrics']['validation'])
    assert v.comparison(rows)['selected_arm']=='P0'
    for r in rows:
        r['metrics']['validation']=metrics(0,0,None);r['gate']=v.program_gate(r['metrics']['validation'])
    out=v.comparison(rows)
    assert out['selected_arm'] is None and out['next_stage']=='closed_failed_program_precheck'
    assert all(a['selection_score'] is None for a in out['arms'].values())
    rows[0]={'job':rows[0]['job'],'status':'scientific_budget_exhausted','gate':{'passed':False}}
    assert v.comparison(rows)['selected_arm'] is None
    rows[0]['status']='engineering_failed'
    with pytest.raises(RuntimeError):v.comparison(rows)


def test_registered_budgets_and_future_ci_cannot_silently_change(tmp_path):
    import json
    original=json.loads((v.ROOT/v.RECIPE).read_bytes())
    for field,value in (('maximum_new_leaf_calls',20001),('search_wall_seconds',4801),('final_leaf_reserve',31)):
        bad=deepcopy(original);bad[field]=value
        path=tmp_path/v.RECIPE;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):v.recipe(tmp_path)
    bad=deepcopy(original);bad['future_confirmation']['individual_one_sided_lower_confidence_level']=.95
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):v.recipe(tmp_path)


def test_rank_tiebreaks_expected_feedback_then_kl_then_leaves():
    for field,value in (('expected',.44),('kl',.05),('leaf_count',16)):
        rows=results()
        for r in rows:
            if r['job']['arm']=='P1':
                if field=='leaf_count':r[field]=value
                else:
                    kwargs={'coverage':.5,'expected':.43,'kl':.1};kwargs[field]=value
                    r['metrics']['validation']=metrics(**kwargs);r['gate']=v.program_gate(r['metrics']['validation'])
        assert v.comparison(rows)['selected_arm']=='P1'
