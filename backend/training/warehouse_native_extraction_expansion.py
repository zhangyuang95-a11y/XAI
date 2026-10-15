"""Finite additional on-policy evidence from the same qualified frozen Actor.

Adds registered train initial states 64..191 to the original 64. The original
32 selection initial states and all earlier failures remain untouched. This
collector does no PPO, tree fitting, model selection or explanation release.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import time

from . import warehouse_native_shutdown_initial_rcpd_run as initial
from . import warehouse_native_shutdown_rcpd as extraction
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import reserve_sampling
from env.warehouse.domain import collaborative_study_config

VERSION = 'warehouse-native-extraction-train-expansion.v1'
BASE_SHA = '762dc0077860f696b8c9b8fc31e1b55c95153f7cde0ad5f16ad24edebb901c9f'
START, COUNT, CAP = 64, 128, 61440


def sources():
    value=initial.sources();value[str(Path(__file__).relative_to(ROOT))]=file_hash(Path(__file__))
    return value


def extra_pools(original, scenes):
    train=scenes['splits']['train']
    if len(original['train'])!=START or digest(train[:START])!=digest(original['train']):
        raise ValueError('Original fixed training prefix differs')
    extra=deepcopy(train[START:START+COUNT])
    seen={row['fingerprint'] for row in extra}
    excluded={row['fingerprint'] for row in original['train']}
    for split,rows in scenes['splits'].items():
        if split!='train': excluded.update(row['fingerprint'] for row in rows)
    if len(extra)!=COUNT or len(seen)!=COUNT or seen & excluded:
        raise ValueError('Additional registered training states are missing, duplicated or overlap another pool')
    if any(not row['id'].startswith('train_') or row.get('split','train')!='train' for row in extra):
        raise ValueError('Only registered training initial states may be added')
    return {'train':extra,'selection':[]}


def check_manifest(output, plan, manifest, *, fixture=False):
    if (manifest.get('version')!=VERSION or manifest.get('plan_sha256')!=digest(plan)
            or manifest.get('ppo_steps')!=0 or plan.get('test_fixture') is not fixture
            or len(manifest['episodes'])>len(plan['contexts'])):
        raise ValueError('Expansion manifest or explicit fixture scope differs')
    actual=reserved=0
    for index,entry in enumerate(manifest['episodes']):
        if entry.get('status')!='completed':raise ValueError('Unconfirmed expansion episode cannot be retried automatically')
        if entry['context']!=plan['contexts'][index]:raise ValueError('Expansion episode order changed')
        data=initial.load_episode(output,entry,fixture=fixture)
        if data['actor_bindings']!=plan['actor_bindings']:raise ValueError('Expansion Actor differs')
        if not 0<data['joint_transitions']<=entry['context']['horizon']:raise ValueError('Expansion reservation exceeded')
        actual+=data['joint_transitions'];reserved+=entry['context']['horizon']
    budget=initial.read_json(Path(output)/'auxiliary_budget.json')
    if (budget['cap']!=plan['maximum_auxiliary_steps'] or budget['reserved_joint_steps']!=reserved
            or manifest['actual_auxiliary_steps']!=actual or manifest['reserved_auxiliary_steps']!=reserved
            or reserved>plan['maximum_auxiliary_steps']):raise ValueError('Expansion accounting differs')
    if manifest['status']=='completed' and len(manifest['episodes'])!=len(plan['contexts']):
        raise ValueError('Incomplete expansion cannot be marked completed')
    return manifest


def collect_dataset(output, plan, pools, actor, protocol, *, fixture=False, config=None, until_episodes=None):
    """Internal fixture-capable collector; the CLI authenticates production first."""
    output=Path(output);manifest=check_manifest(output,plan,initial.read_json(output/'manifest.json'),fixture=fixture)
    limit=len(plan['contexts']) if until_episodes is None else until_episodes
    if type(limit) is not int or not len(manifest['episodes'])<=limit<=len(plan['contexts']):
        raise ValueError('Invalid finite expansion endpoint')
    start=time.monotonic();start_steps=manifest['actual_auxiliary_steps']
    for context in plan['contexts'][len(manifest['episodes']):limit]:
        reserved=reserve_sampling(output/'auxiliary_budget.json',context['horizon'],cap=plan['maximum_auxiliary_steps'])
        pending=deepcopy(manifest);pending['episodes'].append({'status':'pending','context':deepcopy(context)})
        pending.update(status='sampling',reserved_auxiliary_steps=reserved)
        initial.write_json(output/'manifest.json',pending,replace=True);manifest=pending
        data=None
        try:
            data=extraction.collect_episode(actor,pools['train'][context['scene_index']],
                **{k:context[k] for k in ('pool','profile','program_role','seed','episode_id','sampling_mode')},
                expected_bindings=plan['actor_bindings'],protocol=protocol,allow_test_fixture=fixture,config=config)
            if not 0<data['joint_transitions']<=context['horizon']:raise ValueError('Expansion episode exceeded reservation')
            entry=initial.save_episode(output,context,data)
            confirmed=deepcopy(manifest);confirmed['episodes'][-1]=entry
            confirmed['actual_auxiliary_steps']+=data['joint_transitions']
            initial.write_json(output/'manifest.json',confirmed,replace=True)
        except BaseException as error:
            failure={'error':repr(error),'context':context,'automatic_retry':False,
                'known_steps':getattr(error,'actual_joint_steps',data['joint_transitions'] if data is not None else None)}
            try:initial.write_json(output/'sampling_failure.json',failure)
            except BaseException as journal_error:error.expansion_journal_error=str(journal_error)
            raise
        manifest=confirmed
        if len(manifest['episodes'])%16==0 or len(manifest['episodes'])==limit:
            elapsed=time.monotonic()-start;new_steps=manifest['actual_auxiliary_steps']-start_steps
            print(json.dumps({'event':'expansion_episodes_ack','episodes':len(manifest['episodes']),
                'total_episodes':len(plan['contexts']),'actual_auxiliary_steps':manifest['actual_auxiliary_steps'],
                'sampling_and_persistence_steps_per_second':new_steps/max(elapsed,1e-9)}),flush=True)
    if limit==len(plan['contexts']):
        manifest['status']='completed';initial.write_json(output/'manifest.json',manifest,replace=True)
    return manifest


def run(base,output,*,until_episodes=None):
    base,output=Path(base).expanduser().resolve(),Path(output).expanduser().resolve()
    if base==output or base in output.parents or output in base.parents:
        raise ValueError('Use a separate expansion directory')
    bundle=initial.read_bundle(base,expected_manifest_sha256=BASE_SHA,require_reliable=False)
    if bundle['fit_result']['reliable'] is not False:raise ValueError('This expansion follows the registered failed first tree')
    actor,protocol=bundle['actor'],bundle['protocol']
    scene_path=Path(bundle['plan']['source_descriptor']['run'])/'scenarios.json'
    scenes=initial.read_json(scene_path)
    if digest(scenes)!=actor.metadata['scenario_manifest_sha256']:raise ValueError('Source registered scenarios differ')
    pools=extra_pools(bundle['pools'],scenes)
    contexts=initial.make_contexts(pools,120,seed_base=6260908)
    if len(contexts)!=512 or sum(c['horizon'] for c in contexts)!=CAP:raise ValueError('Fixed 128-state expansion cap differs')
    plan={'version':VERSION,'base':str(base),'base_manifest_sha256':BASE_SHA,
        'base_plan_sha256':digest(bundle['plan']),'actor_bindings':deepcopy(bundle['plan']['actor_bindings']),
        'protocol_sha256':digest(protocol),'scene_pools_sha256':digest(pools),'contexts':contexts,
        'training_indices':[START,START+COUNT],'maximum_auxiliary_steps':CAP,'ppo_steps':0,'tree_fits':0,
        'runtime_sources':sources(),'test_fixture':False,'selection_pool_unchanged':True,
        'release_ready':False,'explanation_qualified':False,'sampling_replay_allowed':False}
    if not output.exists():
        output.mkdir(parents=True,exist_ok=False)
        for name,value in [('plan.json',plan),('protocol.json',protocol),('scene_pools.json',pools)]:initial.write_json(output/name,value)
        initial.write_bytes(output/'actor.npz',actor.path.read_bytes())
        for name in plan['runtime_sources']:initial.write_bytes(output/'source_snapshot'/name,(ROOT/name).read_bytes())
        reserve_sampling(output/'auxiliary_budget.json',0,cap=CAP)
        initial.write_json(output/'manifest.json',{'version':VERSION,'plan_sha256':digest(plan),'status':'prepared',
            'episodes':[],'actual_auxiliary_steps':0,'reserved_auxiliary_steps':0,'ppo_steps':0})
    with initial.lease(output):
        if (initial.read_json(output/'plan.json')!=json.loads(json.dumps(plan))
                or digest(initial.read_json(output/'protocol.json'))!=digest(protocol)
                or initial.read_json(output/'scene_pools.json')!=pools
                or file_hash(output/'actor.npz')!=actor.artifact_sha256):raise ValueError('Expansion copied inputs changed')
        for name,sha in plan['runtime_sources'].items():
            if file_hash(output/'source_snapshot'/name)!=sha:raise ValueError('Expansion archived source differs')
        result=collect_dataset(output,plan,pools,actor,protocol,config=collaborative_study_config(),until_episodes=until_episodes)
        if sources()!=plan['runtime_sources']:raise ValueError('Expansion execution sources changed')
    return {'status':result['status'],'episodes':len(result['episodes']),
        'actual_auxiliary_steps':result['actual_auxiliary_steps'],'reserved_auxiliary_steps':result['reserved_auxiliary_steps'],
        'ppo_steps':0,'tree_fits':0,'explanation_qualified':False,'release_ready':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--until-episodes',type=int)
    args=parser.parse_args();print(json.dumps(run(args.base,args.output,until_episodes=args.until_episodes)),flush=True)


if __name__=='__main__':main()
