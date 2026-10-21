"""Prepare shared reference-KL states from already spent common-fit episodes.

No environment, sampling, fitting, optimization or training capability. The
300 predetermined argmax episodes were fit-only in every v628 inner/outer
fold; no sampled, qualification, validation or confirmation episode is reused.
This is an initial frozen-NN state buffer, not a new BC reproduction.
"""
from pathlib import Path
from datetime import datetime,timezone
import gzip,json,os
import numpy as np
from experiments.warehouse import v628_protocol as v
from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA,ReferenceTrainingBuffer
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,FEATURE_DIM,encode_actor_tree_features
from backend.training.warehouse_v622_program import masked_probabilities
from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state,encode_public_observation
from env.warehouse.pure_nn import AGENTS,action_masks
from env.warehouse.navigation import ACTIONS
from scripts.measure_warehouse_v628_collection import episode as public_episode
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v629-shared-reference-buffer.v1'


def source_files():
    sources=v.source_files()
    for relative in ('backend/training/warehouse_v629_reference_buffer.py','scripts/prepare_warehouse_v629_reference_buffer.py',
                     'scripts/measure_warehouse_v628_collection.py','backend/training/warehouse_v625_actor_training.py'):
        sources[relative]=v.file_sha(v.ROOT/relative)
    return dict(sorted(sources.items()))


def select_cases(plan):
    """Fixed common-fit intersection, never choose cases by observed quality."""
    cases=[c for c in plan['cases'] if c['sampling_mode']=='argmax']
    v.require(len(cases)==300 and len({c['case_id'] for c in cases})==300
        and all(sum(c['task']==task for c in cases)==100 for task in (1,2,3)),'all 300 balanced predetermined argmax fit cases required')
    ids={c['case_id'] for c in cases}
    v.require(len(plan['folds'])==3,'all three original folds required')
    for fold in plan['folds']:
        for scope in ('inner','outer'):
            groups=fold[scope]
            v.require(set(groups)=={'fit','calibration','eligibility','validation'}
                and ids<=set(groups['fit']) and not any(ids&set(groups[key]) for key in groups if key!='fit'),
                'reference buffer would consume held-out program data')
    return cases


def audit_case(actor,case,refs,outcome):
    """Rebuild exact joint inputs/distributions/actions from complete public trace."""
    for ref in refs.values():
        path=Path(ref['path'])
        v.require(v.pointer(path)==ref and not path.is_symlink() and not path.stat().st_mode&0o222,'source reference episode missing, changed or writable')
    with np.load(refs['arrays']['path'],allow_pickle=False) as z:
        v.require(set(z.files)=={'x','m','p','sampled'},'unknown original joint trajectory arrays')
        arrays={key:z[key].copy() for key in z.files}
    n=len(arrays['x'])
    v.require(arrays['x'].shape==(n,2,FEATURE_DIM) and arrays['m'].shape==arrays['p'].shape==(n,2,5)
        and arrays['sampled'].shape==(n,2) and arrays['sampled'].dtype==np.int64
        and all(arrays[key].dtype==np.float32 and np.isfinite(arrays[key]).all() for key in ('x','m','p')),
        'invalid original full-NN target arrays')
    with gzip.open(refs['trace']['path'],'rt') as handle:events=[json.loads(line) for line in handle]
    derived=public_episode(case,events,n,actor.artifact_sha256)
    v.require(derived==outcome and case['sampling_mode']=='argmax','source complete episode counter or argmax identity differs')
    for i,event in enumerate(events):
        state=rebuild_public_pre_action_state(event['pre_state_public'])
        obs=encode_public_observation(state);mask=action_masks(state);x=encode_actor_tree_features(obs,mask)
        p=masked_probabilities(actor.logits(obs,mask),mask);chosen=p.argmax(-1)
        proposed={role:ACTIONS[int(chosen[j])] for j,role in enumerate(AGENTS)}
        v.require(state.frame==i and np.array_equal(arrays['x'][i],x) and np.array_equal(arrays['m'][i],mask)
            and np.array_equal(arrays['p'][i],p) and np.array_equal(arrays['sampled'][i],chosen)
            and event['proposed_actions']==proposed and {a['agent_id']:a['last_action'] for a in event['post_state_public']['agents']}==proposed,
            'reference frame inputs, full-NN target or submitted argmax action differs')
    for ref in refs.values():v.require(v.pointer(ref['path'])==ref,'reference episode changed during replay')
    return arrays['x'],arrays['m'],derived


def source_bundle():
    path,plan=v.checked_plan();selected=select_cases(plan);collection_ref=v.pointer(v.OUTPUT/'collection/result.json')
    receipt=v.bound(collection_ref)
    v.require(json.loads((v.OUTPUT/'collection/completed.json').read_bytes())==dict(result_sha256=collection_ref['sha256'])
        and receipt['plan_sha256']==v.file_sha(path) and receipt['actor']==plan['actor']
        and receipt['actual_reset_count']==1650 and receipt['post_policy_interventions']==0
        and receipt['actor_updates']==receipt['PPO_joint_steps']==0,'original shared collection not complete or bound')
    outcomes={row['case']['case_id']:row for row in receipt['outcomes']}
    v.require(len(outcomes)==1650 and len(receipt['outcomes'])==1650,'original collection duplicate or missing episodes')
    actor=FrozenWarehouseV6Actor.load(plan['actor']['path'],expected_sha256=plan['actor']['sha256'],expected_role_sha256=plan['actor_role_sha256'])
    return path,plan,selected,collection_ref,receipt,outcomes,actor


def buffer_metadata(path,plan,collection_ref,sources):
    return dict(schema=BUFFER_SCHEMA,split='train',reference_sha256=plan['actor']['sha256'],
        provenance='spent_frozen_v620_argmax_fit_only_in_all_folds',source_plan=v.pointer(path),shared_collection=collection_ref,
        source_files_sha256=v.digest(sources),new_BC_reproduction=False,new_environment_resets=0,authorizes_training=False)


def verify_original_refs(rows):
    for row in rows:
        for ref in row['source'].values():
            file=Path(ref['path'])
            v.require(v.pointer(file)==ref and not file.is_symlink() and not file.stat().st_mode&0o222,
                'original reference episode changed after its frame replay')


def prepare(output):
    path,plan,cases,collection_ref,receipt,outcomes,actor=source_bundle()
    output=Path(output).resolve();base=v.ROOT/'output/warehouse_v629/training-preparation'
    v.require(output.is_relative_to(base) and output!=base,'reference preparation must use a new isolated output folder')
    output.mkdir(parents=True,exist_ok=False);sources=source_files()
    write_new(output/'started.json',dict(version=VERSION,source_files=sources,source_plan=v.pointer(path),
        shared_collection=collection_ref,case_ids=[c['case_id'] for c in cases],training_authority=False))
    try:
        for name,sha in sources.items():
            source=v.ROOT/name;target=output/'source'/v.digest(sources)/name
            v.require(v.file_sha(source)==sha,'reference preparation source changed before snapshot')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());target.chmod(0o444)
        encoded=[];masks=[];episode_ids=[];rows=[];offset=0
        for i,case in enumerate(cases):
            refs=receipt['cases'][case['case_id']];x,m,counter=audit_case(actor,case,refs,outcomes[case['case_id']]);n=len(x)
            encoded.append(x);masks.append(m);episode_ids.extend([case['case_id']]*n)
            rows.append(dict(case=case,source=refs,offset=offset,physical_frames=n,public_counter=counter));offset+=n
            if (i+1)%25==0:print(json.dumps(dict(reference_episodes_replayed=i+1,total=300,joint_frames=offset)),flush=True)
        file=output/'reference-training.npz'
        metadata=buffer_metadata(path,plan,collection_ref,sources)
        with file.open('xb') as handle:
            np.savez_compressed(handle,encoded=np.concatenate(encoded),masks=np.concatenate(masks),
                episode_ids=np.asarray(episode_ids),metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
            handle.flush();os.fsync(handle.fileno())
        file.chmod(0o444);ref=v.pointer(file);loaded=ReferenceTrainingBuffer.load(file,ref['sha256'],plan['actor']['sha256'])
        v.require(len(loaded.encoded)==offset and len(set(loaded.episode_ids))==300,'stored reference rows or episode IDs differ')
        verify_original_refs(rows)
        v.require(source_files()==sources and v.pointer(collection_ref['path'])==collection_ref and v.pointer(plan['actor']['path'])==plan['actor'],
            'source, collection or Actor changed while preparing reference buffer')
        result=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),buffer=ref,source_plan=v.pointer(path),
            actor=plan['actor'],actor_role_sha256=plan['actor_role_sha256'],shared_collection=collection_ref,source_files=sources,
            case_provenance=rows,complete_episodes=300,joint_frames=offset,NN_action_checks=2*offset,
            purpose='shared_reference_KL_states_after_independent_training_admission',not_BC_reproduction=True,
            used_qualification_or_validation=False,raw_and_regularized_same_buffer=True,
            new_environment_resets=0,new_action_random_streams=0,fit_calls=0,Actor_updates=0,PPO_joint_steps=0,
            training_authority=False,pilot_opened=False)
        write_new(output/'result.json',result);write_new(output/'completed.json',dict(result_sha256=v.file_sha(output/'result.json')))
        return result
    except BaseException as error:
        write_new(output/'failed.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False,training_authority=False));raise


def audit(output):
    output=Path(output).resolve();result_ref=v.pointer(output/'result.json');result=v.bound(result_ref)
    v.require(json.loads((output/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(output/'result.json')),'reference preparation seal changed')
    path,plan,cases,collection_ref,receipt,outcomes,actor=source_bundle()
    v.require(result['version']==VERSION and result['source_files']==source_files()
        and result['source_plan']==v.pointer(path) and result['shared_collection']==collection_ref
        and result['actor']==plan['actor'] and result['actor_role_sha256']==plan['actor_role_sha256']
        and [r['case'] for r in result['case_provenance']]==cases,'reference sources or complete common-fit membership differ')
    for name,sha in result['source_files'].items():
        snapshot=output/'source'/v.digest(result['source_files'])/name
        v.require(v.file_sha(snapshot)==sha and not snapshot.stat().st_mode&0o222,'reference source snapshot changed or writable')
    target=output/'reference-training.npz'
    v.require(result['buffer']==v.pointer(target) and not target.is_symlink() and not target.stat().st_mode&0o222,
        'reference buffer must be the immutable local output asset')
    with np.load(target,allow_pickle=False) as archive:
        v.require(json.loads(str(archive['metadata_json'].item()))==buffer_metadata(path,plan,collection_ref,result['source_files']),
            'reference buffer metadata/provenance differs')
    buffer=ReferenceTrainingBuffer.load(result['buffer']['path'],result['buffer']['sha256'],actor.artifact_sha256);offset=0
    for row in result['case_provenance']:
        case=row['case'];refs=receipt['cases'][case['case_id']]
        x,m,counter=audit_case(actor,case,refs,outcomes[case['case_id']]);n=len(x)
        v.require(row==dict(case=case,source=refs,offset=offset,physical_frames=n,public_counter=counter)
            and np.array_equal(buffer.encoded[offset:offset+n],x) and np.array_equal(buffer.masks[offset:offset+n],m)
            and buffer.episode_ids[offset:offset+n]==(case['case_id'],)*n,'independent stored reference block differs')
        offset+=n
    v.require(offset==result['joint_frames']==len(buffer.encoded) and result['NN_action_checks']==2*offset
        and result['complete_episodes']==300 and result['used_qualification_or_validation'] is False
        and result['raw_and_regularized_same_buffer'] is True and result['training_authority'] is False
        and all(result[key]==0 for key in ('new_environment_resets','new_action_random_streams','fit_calls','Actor_updates','PPO_joint_steps')),
        'reference state count or authority declaration differs')
    verify_original_refs(result['case_provenance'])
    v.require(source_files()==result['source_files'] and v.pointer(collection_ref['path'])==collection_ref
        and v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(path)==result['source_plan']
        and v.pointer(result['buffer']['path'])==result['buffer'] and v.pointer(output/'result.json')==result_ref
        and json.loads((output/'completed.json').read_bytes())==dict(result_sha256=result_ref['sha256']),
        'reference source, Actor, collection or output changed during independent replay')
    value=dict(version=VERSION,independent_replay_passed=True,result=result_ref,buffer=result['buffer'],
        source_files=result['source_files'],source_plan=v.pointer(path),shared_collection=collection_ref,
        complete_episodes=300,joint_frames=offset,NN_action_checks=2*offset,
        qualification_validation_or_confirmation_frames=0,environment_resets=0,Actor_updates=0,PPO_joint_steps=0,
        training_authority=False,pilot_opened=False)
    write_new(output/'independent-audit.json',value);return value
