"""Synthetic owner graphs/tensors only; no scientific claim, reset or training."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
import torch

from experiments.warehouse import v643_program_namespace as n
from experiments.warehouse import v643_program_provider as p
from backend.training import warehouse_v629_training_feedback as feedback


def module(name):
    spec=importlib.util.spec_from_file_location('v643_program_test_'+name,Path(__file__).with_name(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));path.chmod(0o444);return n.pointer(path)


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(before)


@pytest.fixture
def sample(tmp_path,monkeypatch):
    folder=tmp_path/'tensors';folder.mkdir()
    state,x,m,_,legacy,epoch=module('test_warehouse_v642_online_evidence').paired.__wrapped__(folder)
    sources={'new-v643-synthetic-numerical-source.py':'a'*64}
    monkeypatch.setattr(n,'source_files',lambda:dict(sources));monkeypatch.setattr(p,'source_files',lambda:dict(sources))
    root=tmp_path/'new-v643-candidate';monkeypatch.setattr(n,'OWNER_ROOT',root)
    monkeypatch.setattr(n,'REFERENCE_SHA256',state.reference_sha256);monkeypatch.setattr(p,'REFERENCE_SHA256',state.reference_sha256)
    role_parts={role:p.numerical.legacy.load_pool(e.pool_ref,e.target_actor_ref,state.reference,role)[0] for role,e in legacy.items()}
    cases=[];parts={};counts={}
    for si,split in enumerate(p.numerical.legacy.SPLITS):
        old=role_parts['robot_1'][split];rows=[]
        for i in range(len(old['rows'])):
            ci=i//10;base=1000000+si*1000+ci;action=5000000+si*1000+ci;task=ci%3+1
            episode=n.CANDIDATE+'-'+split+'-'+str(ci)
            case=dict(episode=episode,base_seed=base,scenario_seed=10*base+task,task=task,action_seed=action,split=split)
            if i%10==0:cases.append(case)
            rows.append(dict(episode=episode,base_seed=str(base),task=task,frame=i%10,action_seed=action,episode_id=1))
        parts[split]=dict(x=np.stack([role_parts[r][split]['x'] for r in p.ROLES],axis=1),
            m=np.stack([role_parts[r][split]['m'] for r in p.ROLES],axis=1),
            behavior_p=np.stack([role_parts[r][split]['p'] for r in p.ROLES],axis=1),rows=rows)
        parts[split]['executed']=np.array([[np.random.default_rng(p.independent_agent_seed(base_seed=r['action_seed'],
            episode_id=1,frame=r['frame'],agent_id=role)).choice(5,p=parts[split]['behavior_p'][i,j])
            for j,role in enumerate(p.ROLES)] for i,r in enumerate(rows)],np.int64)
        for key in ('x','m','behavior_p','executed'):parts[split][key].flags.writeable=False
        counts[split]=len({r['episode'] for r in rows})
    monkeypatch.setattr(n,'COUNTS',counts)
    inventory=dict(schema=n.INVENTORY_VERSION,scope=n.SCOPE,candidate_id=n.CANDIDATE,expected_counts=counts,cases=cases)
    inventory_ref=write(root/'program-inventory.json',inventory)
    training=write(root/'train-inventory.json',dict(cases=[dict(episode_id=n.CANDIDATE+'-train0',scenario_seed=90000001,action_seed=9500000)]))
    quality=write(root/'quality-inventory.json',dict(cases=[dict(case_id=n.CANDIDATE+'-quality0',scenario_seed=80000001,action_seed=8500000)]))
    templates={role:write(root/(role+'-template.json'),legacy[role].program.to_dict()) for role in p.ROLES}
    # Only test tensors are sealed here; no physical/audit claim is fabricated.
    for path in folder.rglob('*'):
        if path.is_file():path.chmod(0o444)
    fields=dict(train_case_inventory=training,program_evidence_inventory=inventory_ref,quality_case_inventory=quality,
        program_templates=templates)
    partition=write(root/'partitions/test/partition.json',dict(version=n.PARTITION_VERSION,candidate_id=n.CANDIDATE,
        reference_actor=n.pointer(state.reference_path),**fields))
    plan=write(root/'seed-plans/test/plan.json',dict(version=n.PLAN_VERSION,candidate_id=n.CANDIDATE,
        partition=partition,source_files=sources,source_sha256=n.digest(sources)))
    descriptor=write(root/'seed-plans/test/calibration/descriptor.json',dict(candidate_id=n.CANDIDATE,plan=plan,
        partition=partition,source_files=sources,source_sha256=n.digest(sources),collector_integration_enabled=False,**fields))
    clearance=write(root/'synthetic-clearance.json',dict(candidate_id=n.CANDIDATE,plan=plan,
        synthetic_software_graph_only=True))
    claim=write(root/'single-use-20k.claim.json',dict(schema=n.CLAIM_VERSION,candidate_id=n.CANDIDATE,
        plan=plan,partition=partition,descriptor=descriptor,source_sha256=n.digest(sources),clearance=clearance,
        single_use_even_if_interrupted=True))
    def owner(variant):
        return write(root/'seed-plans/test/calibration'/variant/'owner.json',dict(schema=n.OWNER_VERSION,
            candidate_id=n.CANDIDATE,variant=variant,pair_seed=0,claim=claim,plan=plan,descriptor=descriptor,
            source_files=sources,source_sha256=n.digest(sources),**fields))
    owner_ref=owner('regularized');namespace=n.ProgramNamespace.from_owner(owner_ref)
    source=write(root/'synthetic-tensor-source.json',dict(synthetic_software_tensors_only=True))
    refs=tuple([source,*namespace.value['refs']])
    audit=dict(source=source,owner=owner_ref,observed_data_sha256=n.data_sha(parts),asset_manifest_sha256=n.digest(refs),
        synthetic_software_tensors_only=True,training_authority=False)
    observed=n.ObservedPool(namespace,source,parts,refs,audit,n.digest(audit))
    # Deliberate private test seam. The public auditor would refuse this tensor
    # source as a physical trace; no positive physical-source receipt is made.
    object.__setattr__(observed,'_replayed_seal',n._REPLAYED);observed.verify()
    current=p.export_current_actor(state,root/'new-current.npz',pair_seed=0,variant='regularized',candidate_id=n.CANDIDATE)
    Path(current['path']).chmod(0o444);current=n.pointer(current['path'])
    return dict(state=state,x=x,m=m,namespace=namespace,observed=observed,templates=templates,inventory=inventory,
        current=current,root=root,raw_owner=lambda:owner('raw'))


def install(sample, step=15000, *, provider=None, program_refs=None):
    state=sample['state'];state.joint_steps=step;nspace=sample['namespace'];observed=sample['observed']
    targets=p.relabel_current_NN(observed,sample['current'],nspace.output_root/f'test-targets-{step}')
    refit=p.refit_pair_at_boundary(nspace,state,observed,targets,program_refs or sample['templates'],
        nspace.output_root/f'test-refit-{step}',epoch_id='synthetic-refit')
    provider=provider or p.CurrentNNEvidenceProvider(nspace,output=nspace.output_root/'provider')
    provider.install_extraction(refit,observed,state)
    return provider,refit


def test_old_owner_refused_before_output_export_or_rng(tmp_path,monkeypatch):
    old=write(tmp_path/'old-owner.json',dict(schema='warehouse-v642-variant-owner.v1',candidate_id='warehouse-v642-depth9-depth12-01'))
    def forbidden(*args,**kwargs):pytest.fail('old claim touched NN or RNG')
    monkeypatch.setattr(p,'export_current_actor',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='own claimed variant owner'):n.ProgramNamespace.from_owner(old)
    assert list(tmp_path.iterdir())==[Path(old['path'])]


def test_candidate_namespace_is_new_and_confers_no_training_authority(sample):
    space=sample['namespace'];space.verify()
    assert space.value['training_authority'] is False
    assert space.value['seed_history_provenance_recomputed'] is False
    assert space.output_root.is_relative_to(sample['root'])
    with pytest.raises(RuntimeError,match='isolated'):space.output(sample['root'].parent/'warehouse_v642/provider')
    raw=n.ProgramNamespace.from_owner(sample['raw_owner']())
    with pytest.raises(RuntimeError,match='raw must not'):p.CurrentNNEvidenceProvider(raw,output=raw.output_root/'provider')
    assert not (raw.output_root/'provider').exists()


@pytest.mark.parametrize('change',['old_schema','old_candidate','old_id','overlap_scene','cross_split','wrong_count'])
def test_new_program_inventory_rejects_reinterpretation_or_partition_overlap(sample,change):
    value=deepcopy(sample['inventory'])
    if change=='old_schema':value['schema']='warehouse-v642-program-evidence-case-inventory.v1'
    elif change=='old_candidate':value['candidate_id']='warehouse-v642-depth9-depth12-01'
    elif change=='old_id':value['cases'][0]['episode']='old-unbound-episode'
    elif change=='overlap_scene':value['cases'][1].update({k:value['cases'][0][k] for k in ('base_seed','scenario_seed','task')})
    elif change=='cross_split':value['cases'][0]['split']='validation'
    else:value['expected_counts']['fit']+=1
    with pytest.raises(RuntimeError):n.validate_inventory(value)


def test_plain_pool_or_tensor_source_cannot_claim_physical_replay(sample):
    pool=sample['observed'];unsealed=n.ObservedPool(pool.namespace,pool.source_ref,pool.parts,pool.refs,pool.audit,pool.audit_sha256)
    with pytest.raises(RuntimeError,match='actual physical trace replay'):unsealed.verify()
    with pytest.raises(RuntimeError,match='another candidate'):n.audit_observed_pool(sample['namespace'],pool.source_ref)


def test_current_targets_require_new_candidate_metadata_exact_role_hashes_and_splits(sample):
    namespace=sample['namespace'];observed=sample['observed']
    with pytest.raises(RuntimeError,match='old or differently owned'):
        p.relabel_current_NN(observed,n.pointer(sample['state'].reference_path),namespace.output_root/'old-target')
    target=p.relabel_current_NN(observed,sample['current'],namespace.output_root/'targets')
    value,parts,_=p.checked_targets(observed,target)
    assert value['target_actor_role_sha256']==p.role_hashes(sample['state'].model)
    assert value['counterfactual_targets_not_executed'] and value['actual_source_actions_unchanged']
    assert value['current_policy_task_quality_estimate'] is None
    assert all(parts[r]['fit']['rows']==observed.parts['fit']['rows'] for r in p.ROLES)
    assert value['NN_updates']==value['environment_resets']==0


def test_v642_numerical_refit_is_reused_and_new_epoch_evidence_has_actual_gradients(sample):
    from backend.training.warehouse_v642_training_program import complete_program
    assert p.complete_program is complete_program
    provider,refit=install(sample);state=sample['state'];state.joint_steps=16500
    before=p.training_fingerprint(state);evidence=provider(state,'new-v643-epoch-0')
    assert p.training_fingerprint(state)==before
    assert all(e.report['gate']['passed'] and e.payload()['version']==p.EVIDENCE_VERSION for e in evidence.values())
    parts=p.checked_targets(sample['observed'],n.bound(refit)['target_package'])[1]
    sampled=torch.tensor(np.stack([parts[r]['fit']['sampled'] for r in p.ROLES],axis=1))
    result=feedback.feedback_loss(state.model,sample['x'],sample['m'],sampled,joint_steps=16500,total_steps=20000,
        variant='regularized',epoch_id='new-v643-epoch-0',evidence=evidence)
    assert all(v>0 for v in result.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    result.program_loss.loss.backward();state.actor_optimizer.step()
    with pytest.raises(RuntimeError,match='expired after Actor update'):
        evidence['robot_1'].verify(current_role_sha256=p.role_hashes(state.model)['robot_1'],epoch_id='new-v643-epoch-0')
    next_evidence=provider(state,'new-v643-epoch-1')
    assert all(next_evidence[r].target_role_sha256==p.role_hashes(state.model)[r] for r in p.ROLES)


def test_raw_never_reads_provider_or_receives_external_gradients(sample):
    class Unreadable:
        def __getattribute__(self,key):raise AssertionError('raw read an external program')
    result=feedback.feedback_loss(sample['state'].model,sample['x'],sample['m'],None,joint_steps=18000,total_steps=20000,
        variant='raw',epoch_id='raw-v643',evidence=Unreadable())
    assert result.program_loss.loss.item()==0 and not result.evidence
    assert all(v==0 for v in result.diagnostics['selected_frames'].values())
    assert all(v==0 for v in result.diagnostics['role_program_gradient_norm_before_shared_clip'].values())


def test_duplicate_epoch_fails_and_scheduled_refit_is_once_only(sample):
    provider,_=install(sample);sample['state'].joint_steps=16500
    provider(sample['state'],'once')
    with pytest.raises(RuntimeError,match='fresh in-window'):provider(sample['state'],'once')
    assert provider.failed
    sample['state'].joint_steps=15000
    target=p.relabel_current_NN(sample['observed'],sample['current'],sample['namespace'].output_root/'repeat-target')
    with pytest.raises(RuntimeError,match='once-only'):
        p.refit_pair_at_boundary(sample['namespace'],sample['state'],sample['observed'],target,sample['templates'],
            sample['namespace'].output_root/'duplicate-refit',epoch_id='not-authorized-retry')


def test_failure_latch_and_exact_checkpoint_restore(sample,monkeypatch):
    provider,_=install(sample);state=sample['state'];state.joint_steps=16500
    original=p.numerical.legacy.current_report
    def failed(program,parts):
        value=original(program,parts);value['validation']['feedback']['agreement']=.94
        value['gate']=p.numerical.legacy.gate(value['validation']);return value
    monkeypatch.setattr(p.numerical.legacy,'current_report',failed)
    first=provider(state,'fidelity-failed');assert provider.latch
    assert all(not e.feedback_enabled for e in first.values())
    monkeypatch.setattr(p.numerical.legacy,'current_report',original)
    second=provider(state,'fidelity-recovered-but-still-latched')
    assert all(e.report['gate']['passed'] and not e.feedback_enabled for e in second.values())
    snapshot=provider.snapshot(state);state.resumed_training_state=True
    restored=p.CurrentNNEvidenceProvider.restore(snapshot,state,namespace=sample['namespace'],observed=sample['observed'])
    assert restored.latch and restored.used_epochs==provider.used_epochs
    bad=deepcopy(snapshot);bad['payload']['latch']=False;bad['payload_sha256']=n.digest(bad['payload'])
    with pytest.raises(RuntimeError,match='in-memory state'):
        p.CurrentNNEvidenceProvider.restore(bad,state,namespace=sample['namespace'],observed=sample['observed'])
    provider(state,'later-work')
    with pytest.raises(RuntimeError,match='post-snapshot'):
        p.CurrentNNEvidenceProvider.restore(snapshot,state,namespace=sample['namespace'],observed=sample['observed'])
    install(sample,16800,provider=provider,program_refs=provider.programs)
    assert provider.latch is False
    state.joint_steps=16801
    reset=provider(state,'after-next-scheduled-refit')
    assert all(e.feedback_enabled for e in reset.values())


def test_pre_extraction_snapshot_restores_no_inherited_program_or_pool(sample):
    provider=p.CurrentNNEvidenceProvider(sample['namespace'],output=sample['namespace'].output_root/'early-provider')
    state=sample['state'];snapshot=provider.snapshot(state);state.resumed_training_state=True
    restored=p.CurrentNNEvidenceProvider.restore(snapshot,state,namespace=sample['namespace'],observed=None)
    assert restored.programs is restored.observed is None and not restored.installed and not restored.used_epochs
    (provider.output/'calls'/'000000').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='incomplete or post-snapshot'):
        p.CurrentNNEvidenceProvider.restore(snapshot,state,namespace=sample['namespace'],observed=None)


@pytest.fixture(scope='module')
def recorded_physics(tmp_path_factory):
    generator=module('test_warehouse_v629_observed_pool').recording.__wrapped__(tmp_path_factory)
    try:yield next(generator)
    finally:generator.close()


@pytest.mark.parametrize('tamper',[False,True])
def test_new_source_reader_replays_recorded_physics_without_resets(recorded_physics,tmp_path,monkeypatch,tamper):
    """Exercise the actual replay port independently of finite admission tests."""
    original=n.bound(recorded_physics['source']);old_inventory=n.bound(original['inventory'])
    cases=[];records={}
    for ci,old in enumerate(old_inventory['cases']):
        case={**old,'episode':n.CANDIDATE+'-recorded-synthetic-'+str(ci)};cases.append(case)
        item=deepcopy(original['cases'][old['episode']]);trace=n.bound(item['trace'])
        trace.update(schema=n.EPISODE_VERSION,case=case)
        if tamper and ci==0:trace['events'][0]['rewards']['robot_1']+=1
        item['trace']=write(tmp_path/(str(ci)+'-trace.json'),trace);records[case['episode']]=item
    inventory=write(tmp_path/'new-inventory.json',dict(schema=n.INVENTORY_VERSION,scope=n.SCOPE,
        candidate_id=n.CANDIDATE,expected_counts={s:4 for s in n.online.SPLITS},cases=cases))
    owner=write(tmp_path/'synthetic-physics-owner.json',dict(synthetic_software_only=True))
    value=dict(inventory=dict(cases=cases),program_inventory=inventory,refs=[owner,inventory])
    namespace=n.ProgramNamespace(owner,value,n.digest(value))
    # Namespace graph validation has independent tests above. This private test
    # seam avoids fabricating a real finite claim merely to check trace replay.
    monkeypatch.setattr(n.ProgramNamespace,'verify',lambda self:self)
    source=write(tmp_path/'source.json',dict(schema=n.VERSION,scope=n.SCOPE,candidate_id=n.CANDIDATE,
        owner=owner,inventory=inventory,cases=records))
    def forbidden(*args,**kwargs):pytest.fail('read-only replay constructed or reset an environment')
    monkeypatch.setattr(n.physics.WarehouseMultiAgentEnv,'__init__',forbidden)
    monkeypatch.setattr(n.physics.WarehouseMultiAgentEnv,'reset',forbidden)
    if tamper:
        with pytest.raises(RuntimeError,match='outcomes/physics differ'):n.audit_observed_pool(namespace,source)
    else:
        pool=n.audit_observed_pool(namespace,source)
        assert pool.audit['complete_source_episodes']==16 and pool.audit['joint_physical_frames']==1920
        assert pool.audit['environment_resets']==0 and pool.audit['training_authority'] is False
        assert all(np.all(part['executed']==4) for part in pool.parts.values())
