"""Synthetic guard/isolation tests; these are not model acceptance evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from experiments.warehouse import v642_quality_verified_cache as c


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_text(json.dumps(value));path.chmod(0o444)
    return c.pointer(path)


@pytest.fixture
def guard(tmp_path):
    folder=tmp_path/'plan';source=folder/'source'/'one.py';source.parent.mkdir(parents=True)
    source.write_text('frozen\n');source.chmod(0o444)
    actor=save(tmp_path/'actor.json',{'weights':'synthetic'})
    program=save(tmp_path/'program.json',{'qualification':'fixed'})
    proof=save(tmp_path/'proof.json',{'proof':'complete'})
    exact=save(folder/'exact.json',{'seeds':'synthetic'})
    sources={'one.py':c.file_sha(source)}
    plan=dict(source_files=sources,exact_plan=exact,
        fixed=dict(actors={'raw':actor},inputs=[proof]),
        final_programs=dict(programs={'raw':{'robot_1':program}}))
    ref=save(folder/'plan.json',plan)
    calls=[]
    def full_check(r):
        calls.append(r)
        for a in (r,actor,program,proof,exact):
            c.require(c.pointer(a['path'])==a,'full byte check failed')
        return deepcopy(plan),folder
    cache=c.VerifiedPrerequisites(full_check,lambda:dict(sources))
    return cache,ref,plan,folder,proof,calls


def test_native_module_is_not_patched_and_scientific_code_is_reused():
    before=c.native.checked
    api,cache=c.isolated_api()
    assert c.native.checked is before
    assert cache.full_check.__code__ is before.__code__
    for name in ('collect','audit_seeds','clearance','prepare','verify_frame','replay_public_episode'):
        assert getattr(api,name).__code__ is getattr(c.native,name).__code__
    assert api.quality is c.native.quality
    assert api.VERSION==c.VERSION!=c.native.VERSION and api.OUTPUT!=c.native.OUTPUT
    assert api.claim_path()!=c.native.claim_path()


def test_source_profile_extends_frozen_inputs_without_changing_them():
    sources=c.source_files()
    assert all(sources[k]==v for k,v in c.native.source_files().items())
    assert all(name in sources for name in c.FILES)


def test_first_check_full_then_fast_and_final_check_full(guard):
    cache,ref,plan,folder,_,calls=guard
    assert cache.checked(ref)==(plan,folder)
    assert cache.checked(ref)==(plan,folder)
    assert cache.full(ref)==(plan,folder)
    assert len(calls)==cache.full_checks==2 and cache.fast_checks==1


@pytest.mark.parametrize('kind',['actor','program','exact','plan','source_snapshot'])
def test_critical_changes_reject_before_next_reset(guard,kind):
    cache,ref,plan,folder,_,_=guard;cache.checked(ref)
    path={'actor':plan['fixed']['actors']['raw']['path'],
          'program':plan['final_programs']['programs']['raw']['robot_1']['path'],
          'exact':plan['exact_plan']['path'],'plan':ref['path'],
          'source_snapshot':folder/'source/one.py'}[kind]
    path=Path(path);path.chmod(0o644);path.write_text('changed')
    with pytest.raises(RuntimeError):cache.checked(ref)


def test_nonexecution_evidence_change_cannot_pass_end_bracket(guard):
    cache,ref,_,_,proof,_=guard;cache.checked(ref)
    save(Path(proof['path']),{'proof':'tampered'})
    with pytest.raises(RuntimeError,match='full byte'):cache.full(ref)


def test_cached_return_does_not_allow_mutating_saved_plan(guard):
    cache,ref,_,_,_,_=guard
    p,_=cache.checked(ref);p['fixed']['actors'].clear()
    assert cache.checked(ref)[0]['fixed']['actors']


def test_cache_cannot_transfer_authority_to_another_plan(guard):
    cache,ref,_,_,_,_=guard;cache.checked(ref)
    with pytest.raises(RuntimeError,match='plan changed'):
        cache.checked(dict(ref,sha256='a'*64))


def test_actual_source_change_rejects_cached_authority(guard):
    cache,ref,_,_,_,_=guard;cache.checked(ref)
    cache.sources=lambda:{'one.py':'changed-source-sha'}
    with pytest.raises(RuntimeError,match='source changed'):cache.checked(ref)


def test_final_full_guard_failure_seals_collection_failed(guard,monkeypatch):
    cache,ref,_,folder,proof,_=guard
    out=folder/'collection';out.mkdir()
    def collect(r):
        cache.checked(r)
        result=save(out/'result.json',{'synthetic':True})
        save(Path(proof['path']),{'proof':'tampered during synthetic collection'})
        return result
    monkeypatch.setattr(c,'require_prior_terminal',lambda:None)
    monkeypatch.setattr(c,'isolated_api',lambda:(SimpleNamespace(collect=collect),cache))
    with pytest.raises(RuntimeError,match='full byte'):c.collect(ref)
    failure=json.loads((out/'failed.json').read_text())
    assert failure['single_use_claim_retained'] and not failure['pilot_opened']
    assert not (out/'verification-bracket.json').exists()


@pytest.mark.parametrize('exit_code',[0,None,'-15'])
def test_original_active_success_or_unknown_exit_cannot_authorize_new_attempt(tmp_path,monkeypatch,exit_code):
    ref=save(tmp_path/'old/plan.json',{})
    claim=save(tmp_path/'old.claim.json',{'plan':ref})
    save(tmp_path/'old/collection/supervisor-exit.json',
         dict(observed_by='source_bound_subprocess_wait',exit_code=exit_code))
    monkeypatch.setattr(c.native,'claim_path',lambda:Path(claim['path']))
    with pytest.raises(RuntimeError):c.require_prior_terminal()


def test_explicit_nonzero_terminal_is_preserved_as_prior_evidence(tmp_path,monkeypatch):
    from experiments.warehouse import v642_stage_supervisor as s
    ref=save(tmp_path/'old/plan.json',{})
    claim=save(tmp_path/'old.claim.json',{'plan':ref})
    folder=tmp_path/'old/collection';supervision=folder.with_name('collection.supervision')
    command=['synthetic-interpreter','synthetic-script']
    started=save(supervision/'started.json',dict(version=s.VERSION,stage='calibration-quality',
        arguments={},output=str(folder),source_files=c.native.source_files(),command=command,pilot_opened=False))
    log=save(supervision/'run.log',{'synthetic':'no scientific execution'})
    value=dict(version=s.VERSION,stage='calibration-quality',started=started,log=log,command=command,
        observed_by='source_bound_subprocess_wait',exit_code=-15,pilot_opened=False)
    terminal=save(folder/'supervisor-exit.json',value);save(supervision/'exit.json',value)
    monkeypatch.setattr(c.native,'claim_path',lambda:Path(claim['path']))
    monkeypatch.setattr(s,'invocation',lambda *args:(command,folder))
    assert c.require_prior_terminal()==terminal
    save(Path(log['path']),{'synthetic':'tampered'})
    with pytest.raises(RuntimeError):c.require_prior_terminal()
