"""Durable complete current-Actor branch collection through its genuine runtime.

This new producer preserves the original192-context matrix and row schema.
It never routes a branch export through an older candidate loader or registry.
Only raw original training scenarios are queried; trees and PPO are external.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import gzip
import json
import os
import time

import numpy as np

from .warehouse_native_common import ROOT, canonical, digest, file_hash, atomic_json
from . import warehouse_family_branch_runtime_samples as samples
from . import warehouse_family_branch_teacher_data as data_io
from .warehouse_family_branch_teacher_data import _put, _load, _empty, _row, PROFILES, ROLES, COUNTERS
from .warehouse_native_evaluation import critical_groups
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS
from backend import warehouse_branch_runtime as runtime_api
from backend.warehouse_family_explanation import actor_parameter_sha256

VERSION = 'warehouse-family-current-branch-teacher-collection.v2'
DATA_VERSION = data_io.VERSION
collect_branch_pairs = samples.collect_branch_pairs


def _context(actor, protocol, actor_sha, protocol_sha, scenarios):
    paths={key:Path(value).expanduser().resolve() for key,value in
        (('actor',actor),('protocol',protocol),('scenarios',scenarios))}
    inputs={key:dict(path=str(path),sha256=file_hash(path)) for key,path in paths.items()}
    if inputs['actor']['sha256']!=actor_sha:
        raise ValueError('Current Actor bytes differ from their external SHA256')
    declared=_load(paths['protocol'])
    if digest(declared)!=protocol_sha:raise ValueError('Current branch protocol semantic hash differs')
    scenes=_load(paths['scenarios'])
    runtime=runtime_api.BranchRuntime(paths['actor'],protocol=declared,expected_actor_sha256=actor_sha,
        expected_protocol_sha256=protocol_sha)
    if digest(scenes)!=runtime.actor.metadata['scenario_manifest_sha256']:
        raise ValueError('Original scenario manifest differs from current Actor provenance')
    runtime_api.verify(runtime)
    if any(file_hash(Path(item['path']))!=item['sha256'] for item in inputs.values()):
        raise ValueError('Collection inputs changed while loading')
    return SimpleNamespace(runtime=runtime,scenarios=scenes),inputs


def context_matrix(scenario_manifest):
    """Pure fixed physical-fingerprint split; constructing it executes nothing."""
    scenes=scenario_manifest['splits']['train'][:48]
    if len(scenes)!=48:raise ValueError('Original first48 training starts required')
    fingerprints={scene['fingerprint'] for scene in scenes}
    forbidden={scene['fingerprint'] for pool,entries in scenario_manifest['splits'].items()
        if pool!='train' for scene in entries}
    if len(fingerprints)!=48 or fingerprints & forbidden:
        raise ValueError('Physical original train/selection scenario isolation failed')
    contexts=[]
    for index,scene in enumerate(scenes):
        if scene.get('split','train')!='train' or scene.get('test_fixture',False) is not False:
            raise ValueError('Only actual original training starts are permitted')
        pool='train' if index<32 else 'selection'
        for profile_index,profile in enumerate(PROFILES):
            contexts.append(dict(id=f'{pool}:{index:03d}:{profile}',pool=pool,scene_index=index,
                scene_id=scene['id'],fingerprint=scene['fingerprint'],profile=profile,
                program_role=-1 if profile=='selfplay' else (index+profile_index)%2,
                rng_seed=26091030+index*10+profile_index))
    return contexts


def _plan(runtime, scenario_manifest, inputs):
    identity=runtime_api.verify(runtime)
    contexts=context_matrix(scenario_manifest)
    bindings=dict(actor_sha256=runtime.actor_sha256,
        actor_parameters_sha256=actor_parameter_sha256(runtime.actor),
        protocol_sha256=runtime.protocol_sha256,source_sha256=runtime.actor.metadata['source_sha256'],
        runtime_signature=runtime.signature)
    sources={**runtime.sources,str(Path(__file__).relative_to(ROOT)):file_hash(__file__),
        str(Path(samples.__file__).relative_to(ROOT)):file_hash(samples.__file__),
        str(Path(data_io.__file__).relative_to(ROOT)):file_hash(data_io.__file__)}
    return dict(version=VERSION,data_version=DATA_VERSION,producer=VERSION,runtime_identity=identity,
        input_files=deepcopy(inputs),actor_bindings=bindings,feature_names=list(runtime.actor.metadata['feature_names']),
        actor_training_clock=runtime.actor.metadata['source_counters']['joint_steps']+runtime.actor.metadata['joint_steps'],
        scenarios_sha256=digest(scenario_manifest),contexts=contexts,horizon=120,anchor_frames=list(range(0,120,10)),
        training_scene_count=32,selection_scene_count=16,source_split='original_neural_train',
        selection_is_independent_of_teacher_training=True,selection_is_independent_of_prior_NN_training=False,
        maximum_base_steps=192*120,maximum_branch_steps=192*12*10,maximum_branch_queries=192*12*11,
        source_files=sources,all_nonterminal_pairs_retained=True,groups_from_source_anchor=True,
        all_submitted_neural_roles_retained_including_inactive=True,
        runtime_verification='full_content_hashes_at_context_entry_and_exit_before_ACK; object/read-only-array checks within',
        PPO_steps=0,optimizer_updates=0,tree_fits=0,qualification_evaluated=False)


def collect(output, *, actor, protocol, actor_sha, protocol_sha, scenarios, stop_after_contexts=None):
    """Continue only at committed episode boundaries; never resample a pending episode."""
    output=Path(output).expanduser().resolve()
    context, inputs = _context(actor, protocol, actor_sha, protocol_sha, scenarios)
    for path in inputs.values():
        source_path=Path(path['path'])
        if output==source_path or output in source_path.parents:
            raise ValueError('New collection output cannot contain a frozen input')
    if stop_after_contexts is not None and (type(stop_after_contexts) is not int or stop_after_contexts<1):
        raise ValueError('Explicit positive context boundary required')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=_plan(context.runtime,context.scenarios,inputs);plan_hash=digest(plan)
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
                with runtime.verified_context(env):
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
    """Read the unchanged row codec, with this producer's genuine source identity."""
    output=Path(output).expanduser().resolve()
    plan=_load(output/'plan.json')
    if (plan.get('version')!=VERSION or plan.get('producer')!=VERSION
            or plan.get('data_version')!=DATA_VERSION
            or plan.get('runtime_identity',{}).get('runtime_version')!=runtime_api.RUNTIME_VERSION
            or plan.get('runtime_identity',{}).get('family')!=runtime_api.FAMILY):
        raise ValueError('Only the genuine current branch-runtime collection is supported')
    for relative,expected in plan['source_files'].items():
        if file_hash(ROOT/relative)!=expected:raise ValueError('Branch collection source changed')
    for item in plan['input_files'].values():
        if file_hash(Path(item['path']))!=item['sha256']:raise ValueError('Collection input bytes changed')
    manifest=_load(output/'manifest.json')
    for entry in manifest.get('completed',[]):
        directory=(output/entry['path']).parent
        if (file_hash(directory/'events.jsonl.gz')!=entry['events_sha256']
                or _load(directory/'receipt.json')!=entry):
            raise ValueError('Acknowledged context event stream or receipt changed')
    return data_io.read_data(output)


def main():
    parser=argparse.ArgumentParser()
    for name in ('output','actor','protocol','actor-sha','protocol-sha','scenarios'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--stop-after-contexts',type=int)
    args=parser.parse_args()
    collect(args.output,actor=args.actor,protocol=args.protocol,actor_sha=args.actor_sha,
        protocol_sha=args.protocol_sha,scenarios=args.scenarios,stop_after_contexts=args.stop_after_contexts)


if __name__=='__main__':main()
