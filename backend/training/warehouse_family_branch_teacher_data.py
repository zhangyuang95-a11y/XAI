"""Versioned, durable two-role development queries for branch-aware RCPD.

Only genuine registered NumPy actors supply labels. Public training starts are
split by physical fingerprint before queries. No held-out explanation labels,
PPO samples, optimizer updates, program predictions or participant data enter.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import argparse
import fcntl
import gzip
import json
import os
import time

import numpy as np

from .warehouse_native_common import ROOT, canonical, digest, file_hash, atomic_json
from .warehouse_family_branch_feedback_samples import collect_branch_pairs
from . import warehouse_family_feedback_cycle_candidate as candidates
from .warehouse_native_evaluation import critical_groups
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS
from backend.warehouse_family_explanation import actor_parameter_sha256

VERSION = 'warehouse-family-dual-role-branch-teacher-data.v1'
PROFILES = ('selfplay', 'skilled', 'assertive', 'noisy')
ROLES = ('robot_1', 'robot_2')
COUNTERS = ('base_steps_attempted','base_steps','branch_steps_attempted','branch_steps',
            'base_queries_attempted','base_queries','branch_queries_attempted','branch_queries')


def _put(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        raw = (canonical(value)+'\n').encode()
        stream.write(gzip.compress(raw,compresslevel=1,mtime=0) if path.suffix=='.gz' else raw)
        stream.flush(); os.fsync(stream.fileno())


def _load(path):
    path=Path(path); raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def _empty():
    return {key:[] for key in ('observations','probabilities','actions','episode_ids',
        'scene_fingerprints','groups','kind','roles','row_sources','pairs')}


def _row(data, observation, probabilities, role, context, groups, frame, phase,
         *, other_action=None, anchor_id=None):
    values=np.asarray(observation,dtype=np.float32); probs=np.asarray(probabilities,dtype=np.float32)
    if values.shape!=(197,) or probs.shape!=(5,) or not np.isfinite(values).all() or not np.isfinite(probs).all():
        raise ValueError('Genuine observed197 and five-action labels required')
    if (probs<0).any() or not np.isclose(probs.sum(),1.): raise ValueError('Invalid NN distribution')
    source=dict(context_id=context['id'],scene_id=context['scene_id'],frame=frame,
        target_role=role,phase=phase,other_action=other_action,anchor_id=anchor_id,
        role_semantics='executed_NN_command' if phase=='base' else 'isolated_next_NN_query')
    row=dict(observations=values.tolist(),probabilities=probs.tolist(),actions=ACTIONS[int(probs.argmax())],
        episode_ids=context['id'],scene_fingerprints=context['fingerprint'],groups=list(groups),
        kind=phase,roles=role,row_sources=source)
    index=len(data['observations'])
    for key,value in row.items(): data[key].append(value)
    return index


def _plan(context, candidate, manifest_sha, *, horizon=120):
    scenes=context.scenarios['splits']['train'][:48]
    if len(scenes)!=48: raise ValueError('48 original physical training starts required')
    fingerprints={scene['fingerprint'] for scene in scenes}
    forbidden={scene['fingerprint'] for pool,entries in context.scenarios['splits'].items()
               if pool!='train' for scene in entries}
    if len(fingerprints)!=48 or fingerprints & forbidden: raise ValueError('Physical pool isolation failed')
    contexts=[]
    for index,scene in enumerate(scenes):
        pool='train' if index<32 else 'selection'
        for profile_index,profile in enumerate(PROFILES):
            contexts.append(dict(id=f'{pool}:{index:03d}:{profile}',pool=pool,scene_index=index,
                scene_id=scene['id'],fingerprint=scene['fingerprint'],profile=profile,
                program_role=-1 if profile=='selfplay' else (index+profile_index)%2,
                rng_seed=26091030+index*10+profile_index))
    runtime=context.runtime
    bindings=dict(actor_sha256=runtime.actor_sha256,
        actor_parameters_sha256=actor_parameter_sha256(runtime.actor),
        protocol_sha256=runtime.protocol_sha256,source_sha256=runtime.actor.metadata['source_sha256'],
        runtime_signature=runtime.signature)
    return dict(version=VERSION,candidate=str(candidate),candidate_manifest_sha256=manifest_sha,
        actor_bindings=bindings,feature_names=list(runtime.actor.metadata['feature_names']),
        scenarios_sha256=digest(context.scenarios),contexts=contexts,
        horizon=horizon,anchor_frames=list(range(0,horizon,10)),
        training_scene_count=32,selection_scene_count=16,source_split='original_neural_train',
        selection_is_independent_of_teacher_training=True,selection_is_independent_of_prior_NN_training=False,
        maximum_base_steps=len(contexts)*horizon,
        maximum_branch_steps=len(contexts)*len(range(0,horizon,10))*10,
        maximum_branch_queries=len(contexts)*len(range(0,horizon,10))*11,
        source_files={str(Path(__file__).relative_to(ROOT)):file_hash(__file__),
            'backend/training/warehouse_family_branch_feedback_samples.py':file_hash(ROOT/'backend/training/warehouse_family_branch_feedback_samples.py')},
        all_nonterminal_pairs_retained=True,groups_from_source_anchor=True,
        all_submitted_neural_roles_retained_including_inactive=True,
        PPO_steps=0,optimizer_updates=0,tree_fits=0,qualification_evaluated=False)


def collect(output, *, candidate, manifest_sha, stop_after_contexts=None):
    """Continue only at committed episode boundaries; never resample a pending episode."""
    output=Path(output).expanduser().resolve();candidate=Path(candidate).expanduser().resolve()
    if output==candidate or candidate in output.parents or output in candidate.parents:
        raise ValueError('Development output must be independent of frozen candidate')
    if stop_after_contexts is not None and (type(stop_after_contexts) is not int or stop_after_contexts<1):
        raise ValueError('Explicit positive context boundary required')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        context=candidates.load_candidate(candidate,expected_manifest_sha256=manifest_sha)
        plan=_plan(context,candidate,manifest_sha);plan_hash=digest(plan)
        if (output/'plan.json').exists():
            if _load(output/'plan.json')!=plan:raise ValueError('Source or fixed collection matrix changed')
            state=_load(output/'state.json')
        else:
            _put(output/'plan.json',plan)
            state=dict(version=VERSION,plan_sha256=plan_hash,status='prepared',completed=[],pending=None,
                counts=dict.fromkeys(COUNTERS,0),elapsed_seconds=0.)
            atomic_json(output/'state.json',state)
        if state['plan_sha256']!=plan_hash or state['pending'] is not None:
            raise ValueError('Pending/unacknowledged collection requires evidence recovery, not resampling')
        if state['status']=='completed':return state
        runtime=context.runtime;scenes=context.scenarios['splits']['train'][:48]
        count_at_start=len(state['completed']);started=time.monotonic()
        for index,ctx in enumerate(plan['contexts']):
            if index<len(state['completed']):
                entry=state['completed'][index]
                if entry['context']!=ctx or file_hash(output/entry['path'])!=entry['sha256']:
                    raise ValueError('Previously confirmed context changed')
                continue
            if stop_after_contexts is not None and len(state['completed'])-count_at_start>=stop_after_contexts:break
            directory=output/'episodes'/f'{index:04d}';directory.mkdir(parents=True,exist_ok=False)
            state.update(status='running',pending=dict(index=index,context=ctx))
            atomic_json(output/'state.json',state)
            before_counts=deepcopy(state['counts']);data=_empty()
            env=runtime.environment(scenes[ctx['scene_index']]);rng=np.random.default_rng(ctx['rng_seed'])
            events=gzip.open(directory/'events.jsonl.gz','xb',compresslevel=1)
            event_id=0
            def record(event):
                nonlocal event_id
                key={'before_step':'branch_steps_attempted','after_step':'branch_steps',
                     'before_query':'branch_queries_attempted','after_query':'branch_queries',
                     'before_base_step':'base_steps_attempted','after_base_step':'base_steps',
                     'before_base_query':'base_queries_attempted','after_base_query':'base_queries'}.get(event['kind'])
                if key is not None:state['counts'][key]+=1
                if state['counts']['base_steps_attempted']>plan['maximum_base_steps'] or state['counts']['branch_steps_attempted']>plan['maximum_branch_steps']:
                    raise ValueError('Declared auxiliary environment budget exceeded')
                if state['counts']['branch_queries_attempted']>plan['maximum_branch_queries']:
                    raise ValueError('Declared branch query budget exceeded')
                row=dict(context_id=ctx['id'],operation_id=f'{index:04d}:{event_id:06d}',event=event)
                events.write((canonical(row)+'\n').encode());events.flush();os.fsync(events.fileno())
                event_id+=1;return True
            try:
                while not env.done and env.state.frame<plan['horizon']:
                    frame=env.state.frame;before=env.snapshot();before_hash=digest(before)
                    groups={role:critical_groups(env,role) for role in ROLES}
                    # The program acts from S_t before any current neural output exists.
                    program_role=ctx['program_role']
                    program=None if program_role<0 else partner_action(env,ROLES[program_role],ctx['profile'],rng)
                    if digest(env.snapshot())!=before_hash:raise ValueError('Program changed the live environment')
                    if frame in plan['anchor_frames']:
                        result=collect_branch_pairs(runtime,env,record_event=record)
                        decision=result['base_decision'];observations=result['base_observations']
                        proposals=decision['policy_actions'];anchor_id=f'{ctx["id"]}:{frame}'
                        indices={}
                        for bi,branch in enumerate(result['branches']):
                            if branch['done']:continue
                            role=branch['target_role']
                            indices[bi]=_row(data,branch['observations'][role],branch['next_decision']['probabilities'][role],
                                role,ctx,groups[role],frame+1,'counterfactual',
                                other_action=branch['other_action'],anchor_id=anchor_id)
                        for pair in result['pairs']:
                            a,b=pair['branch_indices'];role=pair['target_role']
                            data['pairs'].append(dict(baseline_index=indices[a],changed_index=indices[b],
                                target_role=role,scene_fingerprint=ctx['fingerprint'],groups=groups[role],
                                physical_effect=pair['physical_effect'],nn_changed=pair['nn_changed'],anchor_id=anchor_id,
                                other_action=pair['other_action']))
                    else:
                        record(dict(kind='before_base_query',snapshot=before))
                        proposals,decision=runtime.decision(env)
                        observations={role:env.observations()[role].tolist() for role in ROLES}
                        record(dict(kind='after_base_query',decision=decision,observations=observations))
                    if decision['post_policy_overrides']!=0 or decision['masks'] is not False:raise ValueError('NN actions altered')
                    submitted=dict(proposals)
                    if program_role>=0:submitted[ROLES[program_role]]=program
                    for role_index,role in enumerate(ROLES):
                        if role_index!=program_role:
                            if submitted[role]!=ACTIONS[int(np.argmax(decision['probabilities'][role]))]:
                                raise ValueError('Executed command differs from frozen NN')
                            _row(data,observations[role],decision['probabilities'][role],role,ctx,groups[role],frame,'base')
                    record(dict(kind='before_base_step',before=before,decision=decision,submitted=submitted))
                    _,reward,terminated,truncated,info=env.step(submitted)
                    if info['requested_actions']!=submitted:raise ValueError('Submitted physical commands differ')
                    record(dict(kind='after_base_step',after=env.snapshot(),info=info,rewards=reward,
                        done=bool(terminated or truncated)))
                events.close()
                _put(directory/'data.json.gz',data)
                entry=dict(context=ctx,path=str((directory/'data.json.gz').relative_to(output)),
                    sha256=file_hash(directory/'data.json.gz'),events_sha256=file_hash(directory/'events.jsonl.gz'),
                    rows=len(data['observations']),pairs=len(data['pairs']),final_frame=env.state.frame,
                    counts={k:state['counts'][k]-before_counts[k] for k in COUNTERS})
                _put(directory/'receipt.json',entry);state['completed'].append(entry)
                state.update(pending=None,status='ready',elapsed_seconds=state['elapsed_seconds']+time.monotonic()-started)
                started=time.monotonic();atomic_json(output/'state.json',state)
                print(json.dumps(dict(completed_contexts=len(state['completed']),total_contexts=len(plan['contexts']),
                    rows=entry['rows'],pairs=entry['pairs'],counts=state['counts'])),flush=True)
            except BaseException as error:
                events.close();state.update(status='failed',failure=repr(error))
                atomic_json(output/'state.json',state);raise
        if len(state['completed'])==len(plan['contexts']):
            state.update(status='completed');atomic_json(output/'state.json',state)
            _put(output/'manifest.json',dict(**state,plan_file_sha256=file_hash(output/'plan.json'),
                actor_bindings=plan['actor_bindings'],qualification_evaluated=False))
        return state


def read_data(output):
    """Read complete labelled data without running NN, physics or a tree fit."""
    output=Path(output);plan=_load(output/'plan.json');manifest=_load(output/'manifest.json')
    if manifest['status']!='completed' or manifest['pending'] is not None or manifest['plan_sha256']!=digest(plan):
        raise ValueError('Only completely acknowledged data may be fitted')
    if manifest['plan_file_sha256']!=file_hash(output/'plan.json') or len(manifest['completed'])!=len(plan['contexts']):
        raise ValueError('Plan or context matrix changed')
    pools={pool:_empty() for pool in ('train','selection')}
    for expected,entry in zip(plan['contexts'],manifest['completed']):
        if entry['context']!=expected or file_hash(output/entry['path'])!=entry['sha256']:
            raise ValueError('Context or row file changed')
        data=_load(output/entry['path']);target=pools[expected['pool']];offset=len(target['observations'])
        if len(data['observations'])!=entry['rows'] or len(data['pairs'])!=entry['pairs']:raise ValueError('Row counts changed')
        for pair in data['pairs']:
            target['pairs'].append({**pair,'baseline_index':pair['baseline_index']+offset,
                                   'changed_index':pair['changed_index']+offset})
        for key in data:
            if key!='pairs':target[key].extend(data[key])
    if set(pools['train']['scene_fingerprints']) & set(pools['selection']['scene_fingerprints']):
        raise ValueError('Physical teacher train/selection overlap')
    for pool,data in pools.items():
        for key in ('observations','probabilities'):data[key]=np.asarray(data[key],dtype=np.float32)
    return pools,plan,manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--candidate',required=True);parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--stop-after-contexts',type=int)
    args=parser.parse_args();collect(args.output,candidate=args.candidate,manifest_sha=args.manifest_sha,
        stop_after_contexts=args.stop_after_contexts)


if __name__=='__main__':main()
