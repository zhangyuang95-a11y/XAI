"""Post-hoc extraction for the fixed control endpoint of a completed pair.

This new driver authenticates a saved, externally anchored comparison. It uses
the unchanged paired runner's *actual* collection/fit primitives in a private
subdirectory; their transport version stays attached to those operations.
No learner is restored, no program is attached to a policy, and no final-test
state is used. Pending collection or fitting is never silently replayed.
"""
from copy import deepcopy
from dataclasses import asdict
import argparse
import json
import re
from pathlib import Path

from . import warehouse_native_shutdown_feedback_result as comparison
from . import warehouse_native_shutdown_feedback_run as collector
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_shutdown_result import _Inputs, _json_bytes, _same
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.policy import NumPyNativeActor

VERSION = 'warehouse-native-shutdown-terminal-control-tree.v1'


def sources():
    result = collector.sources()
    for module in (comparison,):
        p = Path(module.__file__); result[str(p.relative_to(ROOT))] = file_hash(p)
    p = Path(__file__); result[str(p.relative_to(ROOT))] = file_hash(p)
    return result


def _linked(read, result, path):
    path = Path(path).resolve()
    bound = result['input_bindings'].get(str(path))
    if not bound or set(bound) != {'sha256', 'size'}:
        raise ValueError('Endpoint file was not verified by the anchored comparison')
    return read.raw(path.parent, path.name, bound['sha256'], bound['size'])


def source_material(pair, report, expected_sha256, *, fixture=False):
    """Read the already verified endpoint; never decode another checkpoint."""
    if type(expected_sha256) is not str or re.fullmatch(r'[a-f0-9]{64}', expected_sha256) is None:
        raise ValueError('Explicit completed comparison SHA256 required')
    pair, report = Path(pair).resolve(), Path(report).resolve()
    read = _Inputs(); result = read.json(report.parent, report.name, expected_sha256)
    if (result.get('version') != comparison.VERSION or result.get('status') != 'fixed_endpoint_complete'
            or result.get('result_source_sha256') != file_hash(Path(comparison.__file__))
            or result.get('test_fixture') is not fixture or result.get('source_selection') is not None
            or result.get('formal_ready') is not False or result.get('explanation_qualified') is not False
            or result.get('control_terminal_tree_extracted') is not False):
        raise ValueError('A complete fixed, unqualified paired result is required')
    endpoint = result['primary_endpoint']
    if type(endpoint) is not int or endpoint <= 0 or (not fixture and endpoint != 250000):
        raise ValueError('Only the registered fixed endpoint may supply this extraction')
    audits = [x for x in result['checkpoint_audits']
              if x.get('kind') == 'final_effective_boundary' and x.get('branch') == 'control']
    if (len(audits) != 1 or audits[0].get('actor_export_equal') is not True
            or audits[0].get('dual_adam_steps_verified') is not True):
        raise ValueError('Actual checkpoint-to-export verification is required first')
    completion_path = pair / f'completion_{endpoint:07d}.json'
    if file_hash(completion_path) != result['completion_sha256']:
        raise ValueError('Original fixed completion changed')
    prepared = _json_bytes(_linked(read, result, pair/'prepared.json'))
    if prepared['runtime_sources'] != collector.sources():
        raise ValueError('Frozen paired source changed')
    protocol = _json_bytes(_linked(read, result, pair/'protocol.json'))
    scenes = _json_bytes(_linked(read, result, pair/'scenarios.json'))
    pools = _json_bytes(_linked(read, result, pair/'refresh_pools.json'))
    actor_path = pair/f'branches/control/actors/actor_{endpoint:07d}.npz'
    actor_bytes = _linked(read, result, actor_path)
    validation = _json_bytes(_linked(read, result, pair/f'branches/control/validation/step_{endpoint:07d}/report.json'))
    bindings = validation['actor_bindings']
    if (bindings['feedback_branch'] != 'control' or bindings['joint_steps'] != endpoint
            or bindings['actor_parameters_sha256'] != audits[0]['actor_parameters_sha256']
            or bindings['actor_sha256'] != file_hash(actor_path)):
        raise ValueError('The actual control endpoint differs from the retained weights')
    actor = NumPyNativeActor(actor_path)
    config = collaborative_study_config(horizon=scenes['configuration']['horizon'])
    admitted = collector.extraction.validate_actor(actor, expected_bindings=bindings,
        protocol=protocol, allow_test_fixture=fixture, config=config)
    _same(asdict(config), scenes['configuration'], 'Public physics configuration differs')
    _same(pools['train'], scenes['splits']['train'][:len(pools['train'])], 'Registered training prefix differs')
    _same(pools['selection'], scenes['splits']['extraction'][:len(pools['selection'])], 'Original selection prefix differs')
    if not fixture and (len(pools['train']), len(pools['selection'])) != (192, 32):
        raise ValueError('The same 192/32 extraction allocation is required')
    seen = {}
    for pool, split in (('train', 'train'), ('selection', 'extraction')):
        fps = [collector._physical_fingerprint(e, scenes['configuration']) for e in pools[pool]]
        if len(set(fps)) != len(fps): raise ValueError('Repeated physical extraction initial state')
        seen[pool] = set(fps)
        for name, entries in scenes['splits'].items():
            if name != split and seen[pool] & {e['fingerprint'] for e in entries}:
                raise ValueError('Extraction initial states overlap a different registered pool')
    if seen['train'] & seen['selection']: raise ValueError('Extraction pools overlap')
    read.unchanged()
    return {'comparison_sha256':expected_sha256, 'completion_sha256':result['completion_sha256'],
        'pair_identity_sha256':digest(prepared['identity']), 'endpoint':endpoint,
        'protocol':protocol, 'scenes':scenes, 'pools':pools, 'bindings':bindings,
        'actor_bytes':actor_bytes, 'actor':actor, 'admitted':admitted, 'inputs':read.records}


def prepare(pair, report, expected_sha256, output, *, allow_test_fixture=False):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Terminal extraction requires a new output directory')
    collector._separate(output, Path(pair).resolve(), Path(report).resolve().parent)
    material = source_material(pair, report, expected_sha256, fixture=allow_test_fixture)
    current = sources(); contexts = collector._contexts(material['pools'], material['scenes']['configuration']['horizon'])
    cap = sum(c['horizon'] for c in contexts)
    if not allow_test_fixture and (len(contexts) != 896 or cap != 107520):
        raise ValueError('Fixed terminal collection allocation differs')
    plan = {'version':VERSION, 'purpose':'posthoc_control_fixed_endpoint', 'branch':'control',
        'pair':str(Path(pair).resolve()), 'comparison':str(Path(report).resolve()),
        'comparison_sha256':expected_sha256, 'completion_sha256':material['completion_sha256'],
        'pair_identity_sha256':material['pair_identity_sha256'], 'endpoint':material['endpoint'],
        'actor_bindings':material['bindings'], 'protocol_sha256':digest(material['protocol']),
        'scenarios_sha256':digest(material['scenes']), 'pools_sha256':digest(material['pools']),
        'runtime_sources':current, 'input_bindings':material['inputs'],
        'maximum_auxiliary_steps':cap, 'maximum_rcpd_calls':13, 'maximum_sklearn_fits':104,
        'ppo_steps':0, 'checkpoint_decodes':0, 'test_fixture':allow_test_fixture,
        'release_ready':False, 'explanation_qualified':False, 'automatic_retry':False}
    # This plan describes collector.collect_dataset/fit_collected, which are
    # invoked unchanged. It is not presented as a feedback-training boundary.
    operation = {'version':collector.VERSION, 'purpose':'terminal_posthoc_control_collection',
        'parent_driver':VERSION, 'parent_plan_sha256':digest(plan),
        'pair_identity_sha256':material['pair_identity_sha256'], 'actor_bindings':material['bindings'],
        'contexts':contexts, 'maximum_auxiliary_steps':cap,
        'cumulative_fit_step':material['admitted']['actor_training_clock'],
        'ppo_steps':0, 'test_fixture':allow_test_fixture}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('plan',plan), ('protocol',material['protocol']), ('scenarios',material['scenes']), ('pools',material['pools'])]:
        collector.write_json(output/(name+'.json'), value)
    collector.write_bytes(output/'actor.npz', material['actor_bytes'])
    for name in current: collector.write_bytes(output/'source_snapshot'/name, (ROOT/name).read_bytes())
    collector.write_json(output/'collection/plan.json', operation)
    collector.reserve_sampling(output/'collection/auxiliary_budget.json', 0, cap=cap)
    collector.write_json(output/'collection/manifest.json', {'version':collector.VERSION,
        'plan_sha256':digest(operation), 'status':'prepared', 'episodes':[],
        'actual_auxiliary_steps':0, 'reserved_auxiliary_steps':0, 'ppo_steps':0})
    collector.write_json(output/'manifest.json', {'version':VERSION, 'status':'prepared',
        'plan_sha256':digest(plan), 'explanation_qualified':False, 'release_ready':False})
    return plan


def advance(output, *, allow_test_fixture=False, until_episodes=None):
    output = Path(output).resolve(); plan = collector._json(output/'plan.json')
    manifest = collector._json(output/'manifest.json')
    if (plan.get('version') != VERSION or plan.get('test_fixture') is not allow_test_fixture
            or plan.get('runtime_sources') != sources() or manifest.get('plan_sha256') != digest(plan)
            or manifest.get('version') != VERSION): raise ValueError('Terminal extraction source binding differs')
    if manifest['status'] not in ('prepared','sampling','completed'): raise ValueError('Interrupted fit cannot be replayed')
    for name, h in plan['runtime_sources'].items():
        if file_hash(output/'source_snapshot'/name) != h: raise ValueError('Archived execution source differs')
    values = {name:collector._json(output/(name+'.json')) for name in ('protocol','scenarios','pools')}
    for name, value in values.items():
        if digest(value) != plan[name+'_sha256']: raise ValueError('Saved '+name+' changed')
    if file_hash(output/'actor.npz') != plan['actor_bindings']['actor_sha256']: raise ValueError('Frozen control Actor changed')
    config = collaborative_study_config(horizon=values['scenarios']['configuration']['horizon'])
    folder=output/'collection'; operation=collector._json(folder/'plan.json')
    if (operation['parent_driver'] != VERSION or operation['parent_plan_sha256'] != digest(plan)
            or operation['actor_bindings'] != plan['actor_bindings']
            or operation['contexts'] != collector._contexts(values['pools'],config.horizon)
            or operation['maximum_auxiliary_steps'] != plan['maximum_auxiliary_steps']):
        raise ValueError('Private collector operation differs')
    if manifest['status']=='completed':
        # Cached completion still authenticates every consumed artifact; it
        # cannot keep claiming reliability after source or program replacement.
        report=_json_bytes(collector.initial_tree_reader.bound_bytes(output,manifest['report']))
        sampled=collector.check_manifest(folder,operation,collector._json(folder/'manifest.json'),fixture=allow_test_fixture)
        if sampled.get('status')!='completed' or len(sampled['episodes'])!=len(operation['contexts']):
            raise ValueError('Completed extraction lacks the full confirmed collection')
        result=_json_bytes(collector.initial_tree_reader.bound_bytes(output,report['fit_result']))
        request=_json_bytes(collector.initial_tree_reader.bound_bytes(folder,sampled['fit_request']))
        _same(result,_json_bytes(collector.initial_tree_reader.bound_bytes(folder,sampled['fit_result'])),'Saved fit bindings differ')
        if (request['plan_sha256']!=digest(operation) or request['acknowledged_episodes_sha256']!=digest(sampled['episodes'])
                or request['actor_sha256']!=plan['actor_bindings']['actor_sha256']):
            raise ValueError('Completed fit request differs from acknowledged data')
        _same(_json_bytes(collector.initial_tree_reader.bound_bytes(output,report['program'])),result.get('program'),'Saved standalone program differs')
        if (report['version']!=VERSION or report['status']!='completed' or report['branch']!='control'
                or report['endpoint']!=plan['endpoint'] or report['actor_bindings']!=plan['actor_bindings']
                or report['reliable'] is not result['reliable']
                or report['actual_auxiliary_steps']!=sampled['actual_auxiliary_steps']
                or report['reserved_auxiliary_steps']!=sampled['reserved_auxiliary_steps']
                or report['release_ready'] is not False or report['explanation_qualified'] is not False):
            raise ValueError('Completed terminal result differs')
        return report
    actor = NumPyNativeActor(output/'actor.npz')
    collector.extraction.validate_actor(actor, expected_bindings=plan['actor_bindings'],
        protocol=values['protocol'], allow_test_fixture=allow_test_fixture, config=config)
    manifest['status']='sampling'; collector.write_json(output/'manifest.json',manifest,replace=True)
    sampled=collector.collect_dataset(folder,operation,values['pools'],actor,values['protocol'],
        fixture=allow_test_fixture,config=config,until_episodes=until_episodes)
    if len(sampled['episodes']) != len(operation['contexts']):
        return {'status':'sampling','acknowledged_episodes':len(sampled['episodes']),
            'actual_auxiliary_steps':sampled['actual_auxiliary_steps'],'release_ready':False}
    manifest['status']='fitting';collector.write_json(output/'manifest.json',manifest,replace=True)
    result=collector.fit_collected(folder,sampled,operation,actor,prior_manager_state=None,
        feedback_config=collector.expanded.feedback_config(),fixture=allow_test_fixture)
    audit=None
    if result.get('manager_state') is not None:
        datasets={pool:collector.merge_saved(folder,sampled['episodes'],pool,fixture=allow_test_fixture)
                  for pool in ('train','selection')}
        audit=collector.expanded_reader.verify_result(result, datasets['train'],datasets['selection'],
            require_reliable=False,fixture=allow_test_fixture)
    collector.write_json(output/'program.json',result.get('program'))
    report={'version':VERSION,'status':'completed','branch':'control','endpoint':plan['endpoint'],
        'actor_bindings':plan['actor_bindings'],'reliable':result['reliable'],'verification':audit,
        'fit_result':collector.binding(output,folder/'fit_result.json'),
        'program':collector.binding(output,output/'program.json'),
        'actual_auxiliary_steps':sampled['actual_auxiliary_steps'],
        'reserved_auxiliary_steps':sampled['reserved_auxiliary_steps'],
        'ppo_steps':0,'checkpoint_decodes':0,'release_ready':False,'explanation_qualified':False,
        'scope':'Same-current-Actor train/selection extraction; no independent explanation or ability qualification'}
    collector.write_json(output/'report.json',report)
    manifest.update(status='completed',report=collector.binding(output,output/'report.json'))
    collector.write_json(output/'manifest.json',manifest,replace=True)
    return report


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--pair',required=True);p.add_argument('--report',required=True)
    p.add_argument('--report-sha256',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('advance');p.add_argument('--output',required=True)
    args=parser.parse_args()
    result=prepare(args.pair,args.report,args.report_sha256,args.output) if args.command=='prepare' else advance(args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('version','status','reliable','actual_auxiliary_steps','maximum_auxiliary_steps')},sort_keys=True))


if __name__=='__main__': main()
