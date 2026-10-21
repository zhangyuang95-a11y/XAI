"""Synthetic reference states only; no new environments, sampling or training."""
from copy import deepcopy
from pathlib import Path
import gzip,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_reference_buffer as b
from backend.training.warehouse_v6_actor_training import WarehouseV6Actor,export_actor
from env.warehouse.domain import WarehouseState,AgentState,DeliveryTask
from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
from env.warehouse.public_observation_v6 import public_pre_action_snapshot


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(before)


def selected_plan():
    cases=[dict(case_id=f'fit-{i}',task=i%3+1,sampling_mode='argmax',base=100000+i,
        scenario_seed=10*(100000+i)+i%3+1,action_seed=300000+i,draw=0) for i in range(300)]
    cases.append(dict(case_id='sampled',task=1,sampling_mode='sampled'))
    groups=dict(fit=[c['case_id'] for c in cases],calibration=['cal'],eligibility=['qual'],validation=['val'])
    return dict(cases=cases,folds=[dict(fold=i,inner=deepcopy(groups),outer=deepcopy(groups)) for i in range(3)])


def test_all_common_fit_argmax_cases_selected_without_quality_filter():
    plan=selected_plan();cases=b.select_cases(plan)
    assert len(cases)==300 and all(c['sampling_mode']=='argmax' for c in cases)
    assert [c['case_id'] for c in cases]==[f'fit-{i}' for i in range(300)]


def test_preparation_source_closure_includes_actual_buffer_loader_and_writers():
    sources=b.source_files()
    for name in ('backend/training/warehouse_v625_actor_training.py','backend/training/warehouse_v629_reference_buffer.py',
                 'scripts/prepare_warehouse_v629_reference_buffer.py','scripts/measure_warehouse_v628_collection.py'):
        assert sources[name]==b.v.file_sha(b.v.ROOT/name)


@pytest.mark.parametrize('change',['calibration','eligibility','validation','fit_missing','partial','imbalance','missing_fold'])
def test_reference_selection_rejects_any_noncommon_fit_or_incomplete_inventory(change):
    plan=selected_plan()
    if change in ('calibration','eligibility','validation'):plan['folds'][1]['inner'][change].append('fit-0')
    if change=='fit_missing':plan['folds'][2]['outer']['fit'].remove('fit-0')
    if change=='partial':plan['cases'].pop(0)
    if change=='imbalance':plan['cases'][0]['task']=2
    if change=='missing_fold':plan['folds'].pop()
    with pytest.raises(RuntimeError):b.select_cases(plan)


@pytest.fixture
def actor(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    with torch.no_grad():
        for role in b.AGENTS:
            a=model.actors[role]
            for parameter in a.parameters():parameter.zero_()
            a.tree.node_count[0]=0;a.tree.leaf_count[0]=1
            a.tree.left_refs.zero_();a.tree.right_refs.zero_();a.tree.leaf_logits[0,4]=1000.;a.gate[-1].bias.fill_(1.)
    ref=export_actor(model,tmp_path/'actor.npz',pair_seed=0,variant='raw',candidate_id='synthetic-reference-buffer-only')
    return b.FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'])


def case_files(folder,actor,case,*,tamper=None):
    points=layout.passable_positions;events=[];xs=[];ms=[];ps=[];sampled=[]
    snapshots=[]
    for frame in range(3):
        state=WarehouseState(episode_id=1,frame=frame,
            agents=[AgentState(role,layout.robot_start_positions[i],battery=60.,active=not(frame==2 and i==0),
                heading='WAIT',recent_positions=(layout.robot_start_positions[i],)) for i,role in enumerate(b.AGENTS)],
            tasks=[DeliveryTask('first',points[1],points[-1]),DeliveryTask('second',points[2],points[-2])])
        snapshots.append(public_pre_action_snapshot(state))
    for i in range(2):
        state=b.rebuild_public_pre_action_state(snapshots[i]);obs=b.encode_public_observation(state);m=b.action_masks(state)
        x=b.encode_actor_tree_features(obs,m);p=b.masked_probabilities(actor.logits(obs,m),m)
        events.append(dict(case=case,frame=i,pre_state_public=snapshots[i],post_state_public=snapshots[i+1],
            proposed_actions={role:'WAIT' for role in b.AGENTS},actor_sha256=actor.artifact_sha256))
        xs.append(x);ms.append(m);ps.append(p);sampled.append(p.argmax(-1))
    if tamper=='p':ps[0]=np.ones((2,5),np.float32)/5
    if tamper=='x':xs[0][0,0]+=1
    if tamper=='action':sampled[0][0]=0
    if tamper=='post_action':events[0]['post_state_public']['agents'][0]['last_action']='UP'
    if tamper=='partial':events[-1]['post_state_public']['agents'][0]['active']=True
    if tamper=='case':events[0]['case']={**case,'sampling_mode':'sampled'}
    folder.mkdir(parents=True,exist_ok=True);file=folder/(case['case_id']+'.npz');trace=file.with_suffix('.jsonl.gz')
    np.savez_compressed(file,x=np.asarray(xs,np.float32),m=np.asarray(ms,np.float32),p=np.asarray(ps,np.float32),sampled=np.asarray(sampled,np.int64))
    with gzip.open(trace,'xt') as handle:
        for event in events:handle.write(json.dumps(event)+'\n')
    file.chmod(0o444);trace.chmod(0o444)
    refs=dict(arrays=b.v.pointer(file),trace=b.v.pointer(trace))
    if tamper=='writable':file.chmod(0o644)
    outcome=dict(case=case,deliveries=0,robot_collisions=0,shutdown=True,physical_frames=2)
    return refs,outcome


def test_reference_case_rebuilds_full_nn_distribution_and_public_terminal_counter(actor,tmp_path):
    case=selected_plan()['cases'][0];refs,outcome=case_files(tmp_path/'data',actor,case)
    x,m,actual=b.audit_case(actor,case,refs,outcome)
    assert len(x)==2 and m.shape==(2,2,5) and actual==outcome


@pytest.mark.parametrize('tamper',['p','x','action','post_action','partial','case','writable'])
def test_bad_reference_frames_or_missing_complete_episode_rejected(actor,tmp_path,tamper):
    case=selected_plan()['cases'][0];refs,outcome=case_files(tmp_path/'data',actor,case,tamper=tamper)
    with pytest.raises((RuntimeError,ValueError)):b.audit_case(actor,case,refs,outcome)


@pytest.fixture
def prepared_output(actor,tmp_path,monkeypatch):
    plan=selected_plan();cases=b.select_cases(plan);receipt=dict(cases={});outcomes={}
    for case in cases:
        refs,outcome=case_files(tmp_path/'spent-data',actor,case);receipt['cases'][case['case_id']]=refs;outcomes[case['case_id']]=outcome
    source=tmp_path/'original-collection.json';source.write_text(json.dumps(dict(complete=True)))
    plan_path=tmp_path/'original-plan.json';plan_path.write_text(json.dumps(plan))
    plan.update(actor=b.v.pointer(tmp_path/'actor.npz'),actor_role_sha256=actor.role_sha256)
    monkeypatch.setattr(b.v,'ROOT',tmp_path)
    (tmp_path/'synthetic-source.py').write_text('# synthetic preparation fixture only\n')
    monkeypatch.setattr(b,'source_files',lambda:{'synthetic-source.py':b.v.file_sha(tmp_path/'synthetic-source.py')})
    monkeypatch.setattr(b,'source_bundle',lambda:(plan_path,plan,cases,b.v.pointer(source),receipt,outcomes,actor))
    output=tmp_path/'output/warehouse_v629/training-preparation/test-only'
    result=b.prepare(output)
    return output,result,plan,receipt,source


def test_complete_buffer_roundtrip_and_audit_uses_original_frames_not_result_booleans(prepared_output):
    output,result,plan,receipt,source=prepared_output;audit=b.audit(output)
    assert result['complete_episodes']==audit['complete_episodes']==300
    assert result['joint_frames']==audit['joint_frames']==600 and audit['NN_action_checks']==1200
    assert audit['independent_replay_passed'] and not result['training_authority']
    assert not result['used_qualification_or_validation'] and result['raw_and_regularized_same_buffer']
    target=Path(result['buffer']['path']);target.chmod(0o644)
    with np.load(target,allow_pickle=False) as z:changed={key:z[key].copy() for key in z.files}
    changed['encoded'][0,0,0]+=1
    np.savez_compressed(target,**changed);target.chmod(0o444)
    result['buffer']=b.v.pointer(target);(output/'result.json').write_text(json.dumps(result))
    (output/'completed.json').write_text(json.dumps(dict(result_sha256=b.v.file_sha(output/'result.json'))))
    with pytest.raises(RuntimeError,match='stored reference block differs'):b.audit(output)


@pytest.mark.parametrize('kind',['source','actor','collection','result','buffer','episode_array'])
def test_asset_change_after_first_episode_replay_cannot_obtain_audit_receipt(prepared_output,monkeypatch,kind):
    output,result,plan,receipt,source=prepared_output;original=b.audit_case;changed=False
    def modified(*args):
        nonlocal changed
        returned=original(*args)
        if not changed:
            files=dict(source=b.v.ROOT/'synthetic-source.py',actor=Path(plan['actor']['path']),collection=source,
                result=output/'result.json',buffer=Path(result['buffer']['path']),
                episode_array=Path(next(iter(receipt['cases'].values()))['arrays']['path']))
            target=files[kind];target.chmod(0o644);target.write_bytes(target.read_bytes()+b' ');target.chmod(0o444);changed=True
        return returned
    monkeypatch.setattr(b,'audit_case',modified)
    with pytest.raises(RuntimeError,match='changed (during|after)'):b.audit(output)
    assert changed and not (output/'independent-audit.json').exists()
