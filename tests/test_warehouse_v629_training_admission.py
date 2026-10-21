"""Synthetic prerequisite records; no scientific data allocation or updates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pytest
from experiments.warehouse import v629_training_admission as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return a.v.pointer(path)


def sealed_plan(path,**values):
    snapshot=path.parent/'source'/'synthetic-source-set'/'reader.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_text('# synthetic source snapshot\n');snapshot.chmod(0o444)
    plan=dict(source_files={'reader.py':a.v.file_sha(snapshot)},source_sha256='synthetic-source-set',**values)
    ref=write(path,plan);write(path.with_name('plan.claim.json'),dict(sha256=ref['sha256'],single_use=True))
    return plan


def test_missing_real_winner_stops_before_reference_rng_initialization_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(a.v,'OUTPUT',tmp_path/'candidate-not-created')
    def forbidden(*args,**kwargs):raise AssertionError('closed A must not inspect reference or draw RNG')
    monkeypatch.setattr(a.reference,'source_bundle',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):a.load_readiness(tmp_path/'also-not-created')
    assert not list(tmp_path.iterdir())


def test_source_closure_includes_actual_ppo_and_full_buffer_reader():
    files=a.source_files()
    for name in (*a.FILES,'backend/training/warehouse_v625_actor_training.py'):
        assert files[name]==a.v.file_sha(a.v.ROOT/name)


@pytest.mark.parametrize('leaves',[64,128,256])
def test_training_structure_hash_accepts_new_capacity_without_old_32_leaf_limit(leaves):
    from backend.training.warehouse_v626_search import Node,encode
    from backend.training.warehouse_v6_actor import FEATURE_NAMES
    from backend.training.warehouse_v622_program import program_structure_sha256
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_row')]=1
    def balanced(depth):return None if depth==0 else Node(tuple(map(float,w)),-.5,None if depth==1 else balanced(depth-1),None if depth==1 else balanced(depth-1))
    tree=encode(balanced(int(np.log2(leaves))))[0]
    with pytest.raises(ValueError):program_structure_sha256(tree)
    assert len(a.structure_sha256(tree,leaves))==64
    with pytest.raises(ValueError):a.structure_sha256(tree,leaves//2)


def test_language_lock_preserves_protection_and_shared_sparse_support(tmp_path):
    spec=importlib.util.spec_from_file_location('training_admission_program_helpers',Path(__file__).with_name('test_warehouse_v629.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    program,_=helper.fixture(tmp_path);lock=a.language_lock(program)
    assert lock['anchor_actor_role_sha256']==program.actor_role_sha256
    assert lock['protection_sha256']==a.v.digest(program.protection)
    assert lock['supports']==[l['support'] for l in program.leaves] and lock['witnesses']==[l['witness'] for l in program.leaves]
    assert a.PROGRAM_FOLD==a.recipe()['program_pair_fold']==0


@pytest.fixture
def confirmation(tmp_path,monkeypatch):
    folder=tmp_path/'fixed-confirmation';folder.mkdir();path=folder/'plan.json'
    actor=write(tmp_path/'actor.json',dict(synthetic=True))
    winner=SimpleNamespace(actor=actor,provenance=dict(lock=write(tmp_path/'lock.json',dict(fixed=True))))
    allocation=write(tmp_path/'proposal/allocation.json',dict(cases=[]))
    write(tmp_path/'proposal/allocation.claim.json',dict(sha256=allocation['sha256']))
    prerequisite=write(tmp_path/'prerequisite.json',dict(synthetic=True))
    plan=sealed_plan(path,cases=[],allocation=allocation,prerequisites={prerequisite['path']:prerequisite['sha256']})
    collection_ref=write(folder/'collection.json',dict(synthetic=True))
    exact_inputs={name:write(folder/'seed-ledger'/(name+'.json'),dict(synthetic=True))
        for name in ('exact_plan','bounded_or_triaged_inventory','context_annotations')}
    exact=write(folder/'seed-ledger/exact/result.json',exact_inputs)
    authority=dict(claim=write(folder/'claim.json',dict(synthetic=True)),
        clearance=write(folder/'seed-ledger/clearance.json',dict(exact=exact)),exact_evidence=exact)
    monkeypatch.setattr(a.c,'INVENTORY_REF',write(tmp_path/'inventory.json',dict(registered_trace_paths=[])))
    audit=dict(collection=collection_ref,integrity_passed=True,**authority)
    write(folder/'collection-audit.json',audit)
    metrics=dict(metrics_passed=True,all_qualification_and_temperatures_unchanged=True)
    stored={**deepcopy(metrics),**dict(plan=a.v.pointer(path),collection_audit=a.v.pointer(folder/'collection-audit.json'),collection=collection_ref)}
    write(folder/'metrics.json',stored);summary=a.expected_summary(path,True);write(folder/'summary.json',summary)
    monkeypatch.setattr(a.v,'OUTPUT',tmp_path/'candidate')
    write(a.v.OUTPUT/'confirmation.passed.json',dict(summary=a.v.pointer(folder/'summary.json'),confirmation_passed=True))
    calls=[]
    monkeypatch.setattr(a.c,'checked_plan',lambda p:(path,plan,winner))
    monkeypatch.setattr(a.confirmation_audit,'load_datasets',lambda p,plan:('audited-synthetic-arrays',dict(outcomes=[],cases={}),audit))
    monkeypatch.setattr(a.c,'checked_clearance',lambda p,plan,execution:dict(exact=authority['exact_evidence']))
    def rescore(*args):calls.append(args);return deepcopy(metrics)
    monkeypatch.setattr(a.c.metrics,'evaluate_fixed',rescore)
    return folder,path,winner,metrics,stored,summary,calls


def test_confirmation_is_recomputed_not_inferred_from_passed_marker(confirmation):
    folder,path,winner,*_,calls=confirmation
    value,refs=a.confirmation_evidence(winner)
    assert len(calls)==1 and calls[0][1]=='audited-synthetic-arrays'
    assert value['summary']==a.v.pointer(folder/'summary.json')
    names={Path(ref['path']).name for ref in refs}
    assert {'allocation.json','allocation.claim.json','plan.claim.json','reader.py','prerequisite.json',
        'exact_plan.json','bounded_or_triaged_inventory.json','context_annotations.json','inventory.json'}<=names


@pytest.mark.parametrize('change',['current_failure','stored_tamper','summary_authority','next_stage','failed_marker','other_winner','other_summary'])
def test_confirmation_booleans_and_changed_binding_cannot_arm_training(confirmation,change):
    folder,path,winner,metrics,stored,summary,calls=confirmation
    if change=='current_failure':metrics['metrics_passed']=False
    if change=='stored_tamper':stored['all_qualification_and_temperatures_unchanged']=False;write(folder/'metrics.json',stored)
    if change=='summary_authority':summary['pilot_opened']=True
    if change=='next_stage':summary['next_stage']='publish_without_training'
    if change in ('summary_authority','next_stage'):
        write(folder/'summary.json',summary)
        write(a.v.OUTPUT/'confirmation.passed.json',dict(summary=a.v.pointer(folder/'summary.json'),confirmation_passed=True))
    if change=='failed_marker':write(a.v.OUTPUT/'confirmation.failed.json',dict(confirmation_passed=False))
    if change=='other_winner':winner=SimpleNamespace(actor=winner.actor,provenance=dict(other='run'))
    if change=='other_summary':
        other=write(folder/'other-summary.json',summary);write(a.v.OUTPUT/'confirmation.passed.json',dict(summary=other,confirmation_passed=True))
    with pytest.raises(RuntimeError):a.confirmation_evidence(winner)


@pytest.fixture
def reference_package(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('training_admission_reference_helpers',Path(__file__).with_name('test_warehouse_v629_reference_buffer.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    before=helper.torch.get_num_threads();helper.torch.set_num_threads(1)
    try:
        actor=helper.actor.__wrapped__(tmp_path)
        plan=helper.selected_plan();cases=helper.b.select_cases(plan);receipt=dict(cases={});outcomes={}
        for case in cases:
            refs,outcome=helper.case_files(tmp_path/'spent-data',actor,case)
            receipt['cases'][case['case_id']]=refs;outcomes[case['case_id']]=outcome
        plan=sealed_plan(tmp_path/'original/plan.json',**plan,actor=a.v.pointer(tmp_path/'actor.npz'),actor_role_sha256=actor.role_sha256)
        source=write(tmp_path/'original/collection/result.json',dict(complete=True))
        write(tmp_path/'original/collection/completed.json',dict(result_sha256=source['sha256']))
        monkeypatch.setattr(helper.b.v,'ROOT',tmp_path)
        source_file=tmp_path/'synthetic-source.py';source_file.write_text('# synthetic preparation fixture only\n')
        monkeypatch.setattr(helper.b,'source_files',lambda:{'synthetic-source.py':a.v.file_sha(source_file)})
        monkeypatch.setattr(helper.b,'source_bundle',lambda:(tmp_path/'original/plan.json',plan,cases,source,receipt,outcomes,actor))
        output=tmp_path/'output/warehouse_v629/training-preparation/test-only';result=helper.b.prepare(output)
        helper.b.audit(output)
        monkeypatch.setattr(a.v,'ROOT',tmp_path)
        yield output,result,plan,receipt,SimpleNamespace(actor=plan['actor'])
    finally:helper.torch.set_num_threads(before)


def test_existing_audited_reference_package_is_checked_against_common_fit_sources(reference_package):
    output,result,plan,receipt,winner=reference_package
    value,refs=a.reference_evidence(winner,output)
    assert value['buffer']==result['buffer'] and value['complete_episodes']==300 and value['joint_frames']==600
    assert len(refs)==612  # Headers, source snapshot, Actor and all 300 NPZ+trace pairs.


@pytest.mark.parametrize('change',['Actor','missing_audit','changed_buffer','other_case'])
def test_reference_package_cannot_be_relabelled_or_unbound(reference_package,change):
    output,result,plan,receipt,winner=reference_package
    if change=='Actor':winner.actor=write(output/'foreign-Actor.json',dict(other=True))
    if change=='missing_audit':(output/'independent-audit.json').unlink()
    if change=='changed_buffer':
        path=Path(result['buffer']['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ');path.chmod(0o444)
    if change=='other_case':
        result=deepcopy(result)  # Keep the independently reloaded source plan unchanged.
        result['case_provenance'][0]['case']['sampling_mode']='sampled'
        result_ref=write(output/'result.json',result);write(output/'completed.json',dict(result_sha256=result_ref['sha256']))
        audit=json.loads((output/'independent-audit.json').read_bytes());audit['result']=result_ref;write(output/'independent-audit.json',audit)
    with pytest.raises((RuntimeError,FileNotFoundError,ValueError)):a.reference_evidence(winner,output)


def test_readiness_does_not_grant_environment_or_optimizer_authority(tmp_path,monkeypatch):
    ref=write(tmp_path/'authority.json',dict(synthetic=True));files={'source':ref['sha256']}
    value=dict(source_files=files,asset_manifest_sha256=a.v.digest((ref,)),recipe=a.recipe(),environment_authority=False,optimizer_authority=False)
    monkeypatch.setattr(a,'source_files',lambda:deepcopy(files))
    ticket=a.TrainingReadiness(value,(deepcopy(ref),),a.v.digest(value));assert ticket.verify() is ticket
    ticket.refs[0]['sha256']='0'*64
    with pytest.raises(RuntimeError,match='references or source changed'):ticket.verify()


@pytest.mark.parametrize('name',['exact_plan.json','bounded_or_triaged_inventory.json','context_annotations.json',
    'inventory.json','allocation.claim.json','prerequisite.json','reader.py'])
def test_actual_consumed_confirmation_dependencies_stay_bound_after_metric_replay(confirmation,monkeypatch,name):
    _,_,winner,*_=confirmation
    _,refs=a.confirmation_evidence(winner)
    value=dict(source_files={},asset_manifest_sha256=a.v.digest(tuple(refs)),recipe=a.recipe())
    monkeypatch.setattr(a,'source_files',lambda:{})
    ticket=a.TrainingReadiness(value,tuple(refs),a.v.digest(value));ticket.verify()
    target=next(Path(ref['path']) for ref in refs if Path(ref['path']).name==name)
    target.chmod(0o644);target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(RuntimeError,match='asset missing or changed'):ticket.verify()


def test_winner_binds_all_settled_jobs_not_only_the_six_selected_programs(tmp_path,monkeypatch):
    folder=tmp_path/'candidate';monkeypatch.setattr(a.v,'OUTPUT',folder)
    jobs=[dict(job_id=f'job-{i}') for i in range(24)]
    plan=sealed_plan(folder/'plan.json',jobs=jobs)
    winner=SimpleNamespace(provenance=dict(plan=a.v.pointer(folder/'plan.json')),actor={},units=[])
    monkeypatch.setattr(a.v,'checked_plan',lambda:(folder/'plan.json',plan))
    monkeypatch.setattr(a.c.metrics,'load_locked_bundle',lambda:winner)
    for i,job in enumerate(jobs):
        result=dict(job=job,status='completed' if i<6 else 'scientific_budget_exhausted')
        if i<6:result['search_report']=write(folder/'jobs'/job['job_id']/'search-report.json',dict(synthetic=True))
        ref=write(folder/'jobs'/job['job_id']/'result.json',result)
        write(Path(ref['path']).with_name('completed.json'),dict(result_sha256=ref['sha256']))
    refs=a.winner_evidence(winner)
    assert sum(Path(ref['path']).name=='result.json' for ref in refs)==24
    assert sum(Path(ref['path']).name=='completed.json' for ref in refs)==24
    assert sum(Path(ref['path']).name=='search-report.json' for ref in refs)==6
    files={};monkeypatch.setattr(a,'source_files',lambda:files)
    value=dict(source_files=files,asset_manifest_sha256=a.v.digest(tuple(refs)),recipe=a.recipe())
    ticket=a.TrainingReadiness(value,tuple(refs),a.v.digest(value));ticket.verify()
    target=folder/'jobs/job-23/completed.json';target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(RuntimeError,match='asset missing or changed'):ticket.verify()


def test_failed_confirmation_cannot_read_or_initialize_reference_buffer(tmp_path,monkeypatch):
    monkeypatch.setattr(a.c.metrics,'load_locked_bundle',lambda:object())
    monkeypatch.setattr(a,'source_files',lambda:{})
    monkeypatch.setattr(a,'winner_evidence',lambda winner:[])
    def failed(*args):raise RuntimeError('B confirmation did not pass')
    def forbidden(*args):raise AssertionError('failed B cannot initialize a training reference')
    monkeypatch.setattr(a,'confirmation_evidence',failed);monkeypatch.setattr(a,'reference_evidence',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='B confirmation did not pass'):a.load_readiness(tmp_path/'unused')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change',['none','source_during_replay','asset_during_replay'])
def test_complete_prerequisite_composition_rechecks_sources_and_assets_before_return(tmp_path,monkeypatch,change):
    actor=write(tmp_path/'actor.json',dict(synthetic=True));header=write(tmp_path/'prerequisite.json',dict(synthetic=True))
    winner=SimpleNamespace(actor=actor,provenance=dict(header=header),units=[])
    source=tmp_path/'source.py';source.write_text('# initial source\n')
    monkeypatch.setattr(a.c.metrics,'load_locked_bundle',lambda:winner)
    monkeypatch.setattr(a,'source_files',lambda:{'source.py':a.v.file_sha(source)})
    monkeypatch.setattr(a,'winner_evidence',lambda w:[header])
    monkeypatch.setattr(a,'confirmation_evidence',lambda w:({'fixed_confirmation':header},[header]))
    def reference(*args):
        if change=='source_during_replay':source.write_text('# source changed during replay\n')
        if change=='asset_during_replay':Path(header['path']).write_text('{}')
        return {'buffer':actor},[actor]
    monkeypatch.setattr(a,'reference_evidence',reference)
    monkeypatch.setattr(a,'program_pair',lambda w:{'synthetic_pair_only':True})
    if change!='none':
        with pytest.raises(RuntimeError,match='source changed during replay|asset missing or changed'):a.load_readiness(tmp_path)
    else:
        ticket=a.load_readiness(tmp_path);ticket.verify()
        assert ticket.value['prerequisites_passed'] and ticket.value['new_environment_resets']==ticket.value['Actor_updates']==ticket.value['PPO_joint_steps']==0
        assert all(ticket.value[k] is False for k in ('environment_authority','optimizer_authority','seed_allocation_authority','publishable','pilot_opened'))
