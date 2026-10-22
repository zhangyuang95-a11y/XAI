"""Post-training finalization cannot run early or select another extraction."""
import numpy as np
from types import SimpleNamespace
from pathlib import Path
import json
import pytest
from experiments.warehouse import v642_final_programs as f


def test_incomplete_calibration_refuses_before_source_pool_replay_or_fit(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('missing completion reached replay, fitting or randomness')
    monkeypatch.setattr(f.pools,'audit_observed_pool',forbidden)
    monkeypatch.setattr(f,'complete_program',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='missing/symlink scientific asset'):
        f.run(completed_assets_ref=dict(path=str(tmp_path/'missing.json'),sha256='missing'),
            reference_directory=None,run_id=None)
    assert not list(tmp_path.iterdir())


def test_missing_final_program_report_has_no_fallback():
    with pytest.raises(RuntimeError,match='completed final-NN program report is required'):
        f.read_completed(None,{})


def fixture(monkeypatch):
    owner={'path':'owner','sha256':'owner-sha'};source={'path':'source','sha256':'source-sha'}
    binding={'path':'binding','sha256':'binding-sha'};extraction={'path':'extraction','sha256':'extraction-sha'}
    programs={r:{'program':dict(path=r,sha256=r)} for r in f.ROLES}
    b=dict(schema='warehouse-v642-observed-refit-source-binding.v1',owner=owner,observed_source=source,
        source_files={'source':'hash'},extraction=extraction)
    result=dict(identity=dict(owner=owner,variant='raw',step=18600),roles=programs)
    monkeypatch.setattr(f.pools,'source_files',lambda:{'source':'hash'})
    monkeypatch.setattr(f,'bound',lambda ref:b if ref==binding else result)
    events=[dict(kind='scheduled_refit_bound',variant='raw',step=s,binding=binding) for s in (15000,16800,18600)]
    return owner,source,b,result,events


def test_only_last_of_three_exact_scheduled_extractions_is_used(monkeypatch):
    owner,source,b,r,events=fixture(monkeypatch)
    programs,refs=f.last_programs(events,variant='raw',owner_ref=owner,source_ref=source)
    assert programs=={role:r['roles'][role]['program'] for role in f.ROLES}
    assert refs==[events[-1]['binding'],b['extraction']]


@pytest.mark.parametrize('change',['missing_schedule','earlier_selected','other_owner','other_pool','other_variant','role_missing'])
def test_wrong_provenance_or_cherry_picked_extraction_is_rejected(monkeypatch,change):
    owner,source,b,r,events=fixture(monkeypatch)
    if change=='missing_schedule':events.pop(0)
    elif change=='earlier_selected':r['identity']['step']=16800
    elif change=='other_owner':b['owner']={}
    elif change=='other_pool':b['observed_source']={}
    elif change=='other_variant':r['identity']['variant']='regularized'
    else:r['roles'].pop('robot_2')
    with pytest.raises(RuntimeError):f.last_programs(events,variant='raw',owner_ref=owner,source_ref=source)


@pytest.fixture
def completed_schema_fixture(tmp_path,monkeypatch):
    """Typed metadata component fixture only, with no NN/program fitting."""
    monkeypatch.setattr(f,'OUTPUT',tmp_path/'finals');folder=f.OUTPUT/'synthetic';folder.mkdir(parents=True)
    claim=tmp_path/'synthetic-finalization-claim.json';monkeypatch.setattr(f,'claim_path',lambda:claim)
    assets=tmp_path/'synthetic-assets.json';assets.write_text('{}');assets_ref=f.pointer(assets)
    sources=f.source_files()
    for name in sources:
        p=folder/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((f.q.ROOT/name).read_bytes());p.chmod(0o444)
    roles={v:{r:str(i*2+j+1)*64 for j,r in enumerate(f.ROLES)} for i,v in enumerate(('raw','regularized'))}
    fixed=dict(actors={'fixture':'no real NN'},actor_role_sha256=roles)
    monkeypatch.setattr(f.q,'assets',lambda ref:fixed)
    def read_program(value,expected):
        if value['target_actor_role_sha256']!=expected:raise ValueError('NN role mismatch')
        return SimpleNamespace(role=value['role'],target_actor_role_sha256=expected,
            feedback_leaves=frozenset({0}),explanation_leaves=frozenset({0}))
    monkeypatch.setattr(f.Program,'from_dict',read_program)
    monkeypatch.setattr(f,'limits',lambda p:dict(leaf=1,node=0,proof=1))
    metric=dict(coverage=.75,expected_actionable_coverage=.70,agreement=.99,agreement_wilson_lower_90=.95,
        rows=100,distinct_episode_count=20,distinct_scenario_seed_count=20,max_episode_fraction=.05,kl=.01,
        illegal_count=0,path_uncertified_count=0,whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0)
    reports={}
    for variant in ('raw','regularized'):
        reports[variant]={}
        for role in f.ROLES:
            p=folder/variant/role;p.mkdir(parents=True)
            (p/'program.json').write_text(json.dumps(dict(role=role,target_actor_role_sha256=roles[variant][role])))
            (p/'checkpoint.json').write_text('{}');(p/'ledger.jsonl').write_text('synthetic metadata only\n')
            val={purpose:dict(metric) for purpose in ('feedback','explanation')}
            reports[variant][role]=dict(status='completed',calibration_program_precheck_passed=True,
                target_actor_role_sha256=roles[variant][role],program=f.pointer(p/'program.json'),
                checkpoint=f.pointer(p/'checkpoint.json'),ledger=f.pointer(p/'ledger.jsonl'),
                current=dict(validation=val,gate=f.gate(val),feedback_leaves=[0],explanation_leaves=[0]),
                metrics=dict(validation=val),solver=dict(limits=dict(leaf=1,node=0,proof=1),counts=dict(leaf=0,node=0,proof=0)))
    plan=dict(version=f.VERSION,completed_assets=assets_ref,source_files=sources,source_sha256=f.digest(sources))
    (folder/'plan.json').write_text(json.dumps(plan));plan_ref=f.pointer(folder/'plan.json')
    claim.write_text(json.dumps(dict(plan=plan_ref,output=str(folder))))
    r=dict(version=f.VERSION,plan=plan_ref,claim=f.pointer(claim),source_files=sources,completed_assets=assets_ref,
        actors=fixed['actors'],actor_role_sha256=roles,roles=reports,input_asset_refs=[assets_ref],
        calibration_program_precheck_passed=True,new_NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    def seal(value,exit_code=0):
        (folder/'result.json').write_text(json.dumps(value));ref=f.pointer(folder/'result.json')
        (folder/'completed.json').write_text(json.dumps(dict(result=ref)))
        (folder/'supervisor-exit.json').write_text(json.dumps(dict(result=ref,exit_code=exit_code,observed_by='exec_tool_terminal_result')))
        return ref
    return r,assets_ref,seal


def test_complete_schema_reader_binds_all_four_programs_and_observed_exit(completed_schema_fixture):
    r,assets,seal=completed_schema_fixture;ref=seal(r)
    result=f.read_completed(ref,assets)
    assert result['report']==ref and set(result['programs'])=={'raw','regularized'}
    assert all(set(programs)==set(f.ROLES) for programs in result['programs'].values())


@pytest.mark.parametrize('change',['role_target','gate','exit','budget'])
def test_resealed_false_metadata_cannot_open_final_program_gate(completed_schema_fixture,change):
    r,assets,seal=completed_schema_fixture;item=r['roles']['regularized']['robot_2']
    if change=='role_target':item['target_actor_role_sha256']='wrong'
    elif change=='gate':item['current']['validation']['explanation']['agreement']=.8
    elif change=='budget':item['solver']['counts']['node']=1
    ref=seal(r,exit_code=1 if change=='exit' else 0)
    with pytest.raises((RuntimeError,ValueError)):f.read_completed(ref,assets)
