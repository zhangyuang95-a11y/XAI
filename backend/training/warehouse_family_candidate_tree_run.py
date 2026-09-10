"""Same-Actor post-hoc extraction for the completed finite energy candidate.

The source's original validation rows are recomputed before sampling. Collection
and fitting invoke the frozen stage collector through its real durable wrapper;
its operation VERSION is retained without relabeling the native Actor. A tree
is never installed into runtime actions, and this driver grants no release or
independent explanation qualification. Pending sampling/fits are not retried.
"""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import stat

from . import warehouse_family_energy_continuation_run as source_run
from . import warehouse_native_shutdown_feedback_run as collector
from .warehouse_native_common import ROOT, canonical, digest, file_hash
from .warehouse_native_shutdown_result import _Inputs, _json_bytes
from backend.warehouse_family_explanation import actor_parameter_sha256
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.policy import NumPyNativeActor

VERSION = 'warehouse-family-native-candidate-posthoc-tree.v2'


def source_reader(kind):
    if kind == 'energy': return source_run
    if kind == 'native_cycle':
        from . import warehouse_family_native_cycle_run
        return warehouse_family_native_cycle_run
    raise ValueError('Unknown completed native source kind')


def sources(source_kind='energy'):
    value = {**collector.sources(), **source_reader(source_kind).sources()}
    for name in ('backend/training/warehouse_family_candidate_tree_run.py',
                 'backend/warehouse_family_explanation.py'):
        value[name] = file_hash(ROOT/name)
    return value


def _same(left, right, reason):
    if canonical(left) != canonical(right): raise ValueError(reason)


def _sha(value):
    return source_run._sha(value)


def _material(source, anchors, pool_pair_root, expected_pool_prepared_sha256, fixture, source_kind='energy'):
    """Zero-step admission: saved fixed validation plus actual export identity."""
    if type(fixture) is not bool: raise ValueError('Explicit fixture scope required')
    for value in (*anchors.values(), expected_pool_prepared_sha256): _sha(value)
    source, pair = Path(source).resolve(), Path(pool_pair_root).resolve()
    # The producer's explicit completed reader does not invoke its learner loader.
    reader = source_reader(source_kind)
    saved = reader.read_completed(source, expected_completion_sha256=anchors['completion_sha256'],
                                     allow_test_fixture=fixture)
    completion, report = saved['completion'], saved['report']
    if (completion.get('status') != 'both_gates_ready'
            or any(report.get(key, {}).get('eligible') is not True for key in ('capability','warmup_capability'))
            or completion.get('feedback_disabled_after_source') is not True
            or completion.get('feedback_lambda') != 0. or completion.get('used_final_test') is not False):
        raise ValueError('Both original capability and warmup gates must pass before extraction')
    prepared, protocol, scenes = saved['prepared'], saved['protocol'], saved['scenarios']
    endpoint = prepared['primary_endpoint']
    if (type(endpoint) is not int or endpoint <= 0 or (not fixture and source_kind == 'energy' and endpoint != 50000)
            or completion['until'] != endpoint or prepared['validation_endpoints'] != [endpoint]):
        raise ValueError('Only the registered fixed candidate endpoint may supply a tree')
    read = _Inputs()
    path = Path(saved['actor_path']).resolve()
    raw_actor = read.raw(path.parent,path.name,anchors['actor_sha256'])
    report_path = source/f'branches/beta1/validation/step_{endpoint:07d}/report.json'
    _same(read.json(report_path.parent,report_path.name,anchors['validation_report_sha256']), report,
          'Recomputed validation differs from its external byte anchor')
    _same(read.json(source,f'completion_{endpoint:07d}.json',anchors['completion_sha256']),completion,'Completed source changed')
    bindings = deepcopy(saved['actor_bindings'])
    if (bindings['actor_sha256'] != anchors['actor_sha256'] or bindings['joint_steps'] != endpoint
            or bindings['experiment_version'] != source_run.native.VERSION
            or bindings['shutdown_arm'] != 'beta1' or bindings['own_shutdown_beta'] != 1.
            or bindings['branch'] != 'own_credit' or 'feedback_branch' in bindings):
        raise ValueError('Only the genuine beta1 pure-PPO native candidate is accepted')
    # This anchor is separately supplied by the caller, never invented from a
    # missing metadata field. The producer binds its actual learner/export proof.
    proof = saved['actor_parameter_receipt']
    if (proof.get('actor_sha256') != anchors['actor_sha256']
            or proof.get('actor_parameters_sha256') != anchors['actor_parameters_sha256']
            or proof.get('checkpoint_sha256') != completion['checkpoint']['sha256']
            or proof.get('all_six_arrays_equal') is not True):
        raise ValueError('The current endpoint requires its actual six-array export proof')
    actor = NumPyNativeActor(path)
    if actor_parameter_sha256(actor) != anchors['actor_parameters_sha256']:
        raise ValueError('Actual NPZ parameter semantics differ from the external endpoint anchor')
    bindings['actor_parameters_sha256'] = anchors['actor_parameters_sha256']
    config = collaborative_study_config(horizon=scenes['configuration']['horizon'])
    _same(asdict(config),scenes['configuration'],'Public physics differs')
    admitted = collector.extraction.validate_actor(actor,expected_bindings=bindings,protocol=protocol,
        config=config,allow_test_fixture=fixture)
    old = read.json(pair,'prepared.json',expected_pool_prepared_sha256)
    if (old.get('version') != collector.VERSION or old.get('test_fixture') is not fixture
            or old.get('runtime_sources') != collector.sources()):
        raise ValueError('Extraction must retain the original frozen paired pool preparation')
    pools = read.json(pair,'refresh_pools.json')
    if digest(pools) != old['refresh_pools_sha256'] or digest(scenes) != old['scenario_manifest_sha256']:
        raise ValueError('Original paired pools or common scene manifest differ')
    if set(pools) != {'train','selection'} or any(not isinstance(x,list) or not x for x in pools.values()):
        raise ValueError('Two nonempty original collection pools required')
    if not fixture and (len(pools['train']),len(pools['selection'])) != (192,32):
        raise ValueError('The original fixed 192/32 collection allocation is required')
    fingerprints = {}
    for name, split in (('train','train'),('selection','extraction')):
        _same(pools[name],scenes['splits'][split][:len(pools[name])],'Original pool prefix or ordering changed')
        values = [collector._physical_fingerprint(entry,scenes['configuration']) for entry in pools[name]]
        if len(set(values)) != len(values): raise ValueError('Repeated physical collection state')
        fingerprints[name] = set(values)
        for other, entries in scenes['splits'].items():
            if other != split and fingerprints[name] & {e['fingerprint'] for e in entries}:
                raise ValueError('Collection pool overlaps another registered initial-state pool')
    if fingerprints['train'] & fingerprints['selection']: raise ValueError('Collection pools overlap')
    inputs = deepcopy(saved['input_bindings'])
    for name, value in read.records.items():
        if name in inputs: _same(inputs[name],value,'Source byte bindings disagree')
        inputs[name] = value
    for name, value in inputs.items():
        p = Path(name); read.raw(p.parent,p.name,value['sha256'],value['size'])
    read.unchanged()
    return {'protocol':protocol,'scenes':scenes,'pools':pools,'actor':actor,'actor_bytes':raw_actor,
        'bindings':bindings,'admitted':admitted,'endpoint':endpoint,'input_bindings':inputs,
        'capability':deepcopy(report['capability']),'warmup_capability':deepcopy(report['warmup_capability']),
        'actor_parameter_receipt':deepcopy(proof),'pool_pair_identity_sha256':digest(old['identity'])}


def prepare(source, *, expected_completion_sha256, expected_actor_sha256,
            expected_actor_parameters_sha256, expected_validation_report_sha256,
            pool_pair_root, expected_pool_prepared_sha256, output, allow_test_fixture=False, source_kind='energy'):
    output = Path(output).resolve()
    collector._separate(output,Path(source).resolve(),Path(pool_pair_root).resolve())
    anchors = {'completion_sha256':expected_completion_sha256,'actor_sha256':expected_actor_sha256,
        'actor_parameters_sha256':expected_actor_parameters_sha256,'validation_report_sha256':expected_validation_report_sha256}
    material = _material(source,anchors,pool_pair_root,expected_pool_prepared_sha256,allow_test_fixture,source_kind)
    current = sources(source_kind); contexts = collector._contexts(material['pools'],material['scenes']['configuration']['horizon'])
    cap = sum(c['horizon'] for c in contexts)
    if not allow_test_fixture and (len(contexts),cap) != (896,107520):
        raise ValueError('Fixed 896-episode auxiliary reservation differs')
    plan = {'version':VERSION,'purpose':'same_native_candidate_posthoc','source':str(Path(source).resolve()),
        'source_kind':source_kind,'source_version':source_reader(source_kind).VERSION,
        'source_anchors':anchors,'pool_pair_root':str(Path(pool_pair_root).resolve()),
        'pool_prepared_sha256':expected_pool_prepared_sha256,'endpoint':material['endpoint'],
        'actor_bindings':material['bindings'],'protocol_sha256':digest(material['protocol']),
        'scenarios_sha256':digest(material['scenes']),'pools_sha256':digest(material['pools']),
        'input_bindings':material['input_bindings'],'runtime_sources':current,
        'capability':material['capability'],'warmup_capability':material['warmup_capability'],
        'actor_parameter_receipt':material['actor_parameter_receipt'],
        'pool_pair_identity_sha256':material['pool_pair_identity_sha256'],
        'cumulative_fit_step':material['admitted']['actor_training_clock'],
        'extraction_contract':collector.expanded.contract(),'maximum_auxiliary_steps':cap,
        'maximum_rcpd_calls':13,'maximum_sklearn_fits':104,'ppo_steps':0,'checkpoint_decodes':0,
        'automatic_retry':False,'test_fixture':allow_test_fixture,'explanation_qualified':False,'release_ready':False}
    operation = {'version':collector.VERSION,'purpose':'same_native_candidate_posthoc_collection','parent_driver':VERSION,
        'parent_plan_sha256':digest(plan),'actor_bindings':plan['actor_bindings'],'contexts':contexts,
        'maximum_auxiliary_steps':cap,'cumulative_fit_step':plan['cumulative_fit_step'],
        'ppo_steps':0,'test_fixture':allow_test_fixture}
    output.mkdir(parents=True,exist_ok=False)
    collector.write_bytes(output/'candidate_tree.lock',b'')
    for name,value in (('plan',plan),('protocol',material['protocol']),('scenarios',material['scenes']),('pools',material['pools'])):
        collector.write_json(output/(name+'.json'),value)
    collector.write_bytes(output/'actor.npz',material['actor_bytes'])
    for name in current: collector.write_bytes(output/'source_snapshot'/name,(ROOT/name).read_bytes())
    collector.write_json(output/'collection/plan.json',operation)
    collector.reserve_sampling(output/'collection/auxiliary_budget.json',0,cap=cap)
    collector.write_json(output/'collection/manifest.json',{'version':collector.VERSION,'plan_sha256':digest(operation),
        'status':'prepared','episodes':[],'actual_auxiliary_steps':0,'reserved_auxiliary_steps':0,'ppo_steps':0})
    collector.write_json(output/'manifest.json',{'version':VERSION,'status':'prepared','plan_sha256':digest(plan),
        'explanation_qualified':False,'release_ready':False})
    return plan


def _load(output, fixture):
    output = Path(output).resolve(); plan = collector._json(output/'plan.json'); manifest = collector._json(output/'manifest.json')
    if (type(fixture) is not bool or plan.get('version') != VERSION or plan.get('test_fixture') is not fixture
            or plan.get('runtime_sources') != sources(plan.get('source_kind'))
            or plan.get('source_version') != source_reader(plan.get('source_kind')).VERSION or manifest.get('version') != VERSION
            or manifest.get('plan_sha256') != digest(plan) or manifest.get('status') not in ('prepared','sampling','completed')
            or plan.get('ppo_steps') != 0 or plan.get('checkpoint_decodes') != 0 or plan.get('automatic_retry') is not False
            or any(plan.get(k) is not False or manifest.get(k) is not False for k in ('explanation_qualified','release_ready'))
            or any(plan.get(k,{}).get('eligible') is not True for k in ('capability','warmup_capability'))):
        raise ValueError('Candidate extraction identity differs or an interrupted fit cannot be replayed')
    _same(plan['extraction_contract'],collector.expanded.contract(),'Fixed thirteen-candidate contract changed')
    if (plan['maximum_rcpd_calls'],plan['maximum_sklearn_fits']) != (13,104): raise ValueError('Finite fit cap differs')
    for name,h in plan['runtime_sources'].items():
        if file_hash(output/'source_snapshot'/name) != h: raise ValueError('Archived execution source differs')
    read = _Inputs()
    for name,b in plan['input_bindings'].items():
        p=Path(name);read.raw(p.parent,p.name,b['sha256'],b['size'])
    values={name:collector._json(output/(name+'.json')) for name in ('protocol','scenarios','pools')}
    for name,value in values.items():
        if digest(value) != plan[name+'_sha256']: raise ValueError('Saved '+name+' changed')
    if file_hash(output/'actor.npz') != plan['actor_bindings']['actor_sha256']: raise ValueError('Frozen candidate Actor changed')
    config=collaborative_study_config(horizon=values['scenarios']['configuration']['horizon'])
    contexts=collector._contexts(values['pools'],config.horizon)
    operation=collector._json(output/'collection/plan.json')
    if (operation.get('version') != collector.VERSION or operation.get('parent_driver') != VERSION
            or operation.get('parent_plan_sha256') != digest(plan) or operation.get('actor_bindings') != plan['actor_bindings']
            or operation.get('contexts') != contexts or operation.get('cumulative_fit_step') != plan['cumulative_fit_step']
            or operation.get('maximum_auxiliary_steps') != plan['maximum_auxiliary_steps']
            or sum(c['horizon'] for c in contexts) != plan['maximum_auxiliary_steps']
            or (not fixture and (len(contexts),plan['maximum_auxiliary_steps']) != (896,107520))):
        raise ValueError('Private actual collector operation differs')
    read.unchanged()
    return output,plan,manifest,values,config,operation


def _completed(output,plan,manifest,operation,fixture):
    folder=output/'collection'
    report=_json_bytes(collector.initial_tree_reader.bound_bytes(output,manifest['report']))
    sampled=collector.check_manifest(folder,operation,collector._json(folder/'manifest.json'),fixture=fixture)
    if sampled.get('status') != 'completed' or len(sampled['episodes']) != len(operation['contexts']):
        raise ValueError('Complete confirmed collection is required')
    result=_json_bytes(collector.initial_tree_reader.bound_bytes(folder,sampled['fit_result']))
    _same(result,_json_bytes(collector.initial_tree_reader.bound_bytes(output,report['fit_result'])),'Fit bindings differ')
    request=_json_bytes(collector.initial_tree_reader.bound_bytes(folder,sampled['fit_request']))
    if (request['plan_sha256'] != digest(operation) or request['acknowledged_episodes_sha256'] != digest(sampled['episodes'])
            or request['actor_sha256'] != plan['actor_bindings']['actor_sha256']
            or request['cumulative_fit_step'] != plan['cumulative_fit_step']
            or request['prior_manager_state_sha256'] != digest(None)
            or request.get('version') != collector.VERSION or request.get('retry_allowed') is not False
            or request.get('maximum_candidate_fits') != 13 or request.get('maximum_sklearn_fits') != 104
            or request.get('feedback_config_sha256') != digest(asdict(collector.expanded.feedback_config()))
            or request['extraction_contract_sha256'] != digest(collector.expanded.contract())):
        raise ValueError('Acknowledged fit request differs')
    _same(_json_bytes(collector.initial_tree_reader.bound_bytes(output,report['program'])),result.get('program'),'Standalone program differs')
    datasets={p:collector.merge_saved(folder,sampled['episodes'],p,fixture=fixture) for p in ('train','selection')}
    _same(request['dataset_receipts'],{p:d['collector_receipt_sha256'] for p,d in datasets.items()},'Fit dataset receipts differ')
    audit=None
    if result.get('manager_state') is not None:
        audit=collector.expanded_reader.verify_result(result,datasets['train'],datasets['selection'],require_reliable=False,fixture=fixture)
    elif (result.get('reliable') is not False or result.get('program') is not None
            or result.get('input_rejection',{}).get('request_sha256') != digest(request)
            or result.get('version') != collector.expanded.VERSION):
        raise ValueError('Failed data admission cannot claim a fitted program')
    expected={'version':VERSION,'status':'completed','endpoint':plan['endpoint'],'actor_bindings':plan['actor_bindings'],
        'reliable':result['reliable'],'verification':audit,'fit_result':collector.binding(output,folder/'fit_result.json'),
        'program':collector.binding(output,output/'program.json'),'actual_auxiliary_steps':sampled['actual_auxiliary_steps'],
        'reserved_auxiliary_steps':sampled['reserved_auxiliary_steps'],'ppo_steps':0,'checkpoint_decodes':0,
        'release_ready':False,'explanation_qualified':False,
        'scope':'Same-current-native-Actor posthoc train/selection evidence; no independent explanation or release qualification'}
    _same(report,expected,'Completed candidate report differs from confirmed data and recomputed tree metrics')
    return report


@contextmanager
def _lease(output):
    # The reused collector expects its caller to serialize budget mutations.
    # A small per-run lease replaces the paired trainer's outer budget lease.
    lock=Path(output).resolve()/'candidate_tree.lock'
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        held=os.fstat(fd)
        if not stat.S_ISREG(held.st_mode) or held.st_nlink != 1: raise ValueError('Invalid collection lease')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=os.stat(lock,follow_symlinks=False)
        if (current.st_dev,current.st_ino)!=(held.st_dev,held.st_ino): raise ValueError('Collection lease changed')
        yield
    finally: os.close(fd)


def advance(output, *, allow_test_fixture=False, until_episodes=None):
    with _lease(output):
        return _advance(output,allow_test_fixture=allow_test_fixture,until_episodes=until_episodes)


def _advance(output, *, allow_test_fixture=False, until_episodes=None):
    output,plan,manifest,values,config,operation=_load(output,allow_test_fixture)
    if manifest['status']=='completed': return _completed(output,plan,manifest,operation,allow_test_fixture)
    actor=NumPyNativeActor(output/'actor.npz')
    if actor_parameter_sha256(actor) != plan['actor_bindings']['actor_parameters_sha256']:
        raise ValueError('Actual current parameters differ')
    collector.extraction.validate_actor(actor,expected_bindings=plan['actor_bindings'],protocol=values['protocol'],config=config,allow_test_fixture=allow_test_fixture)
    manifest['status']='sampling';collector.write_json(output/'manifest.json',manifest,replace=True)
    folder=output/'collection'
    sampled=collector.collect_dataset(folder,operation,values['pools'],actor,values['protocol'],fixture=allow_test_fixture,
        config=config,until_episodes=until_episodes)
    if len(sampled['episodes']) != len(operation['contexts']):
        return {'status':'sampling','acknowledged_episodes':len(sampled['episodes']),
            'actual_auxiliary_steps':sampled['actual_auxiliary_steps'],'release_ready':False}
    manifest['status']='fitting';collector.write_json(output/'manifest.json',manifest,replace=True)
    result=collector.fit_collected(folder,sampled,operation,actor,prior_manager_state=None,
        feedback_config=collector.expanded.feedback_config(),fixture=allow_test_fixture)
    audit=None
    if result.get('manager_state') is not None:
        datasets={p:collector.merge_saved(folder,sampled['episodes'],p,fixture=allow_test_fixture) for p in ('train','selection')}
        audit=collector.expanded_reader.verify_result(result,datasets['train'],datasets['selection'],require_reliable=False,fixture=allow_test_fixture)
    collector.write_json(output/'program.json',result.get('program'))
    report={'version':VERSION,'status':'completed','endpoint':plan['endpoint'],'actor_bindings':plan['actor_bindings'],
        'reliable':result['reliable'],'verification':audit,'fit_result':collector.binding(output,folder/'fit_result.json'),
        'program':collector.binding(output,output/'program.json'),'actual_auxiliary_steps':sampled['actual_auxiliary_steps'],
        'reserved_auxiliary_steps':sampled['reserved_auxiliary_steps'],'ppo_steps':0,'checkpoint_decodes':0,
        'release_ready':False,'explanation_qualified':False,
        'scope':'Same-current-native-Actor posthoc train/selection evidence; no independent explanation or release qualification'}
    collector.write_json(output/'report.json',report)
    manifest.update(status='completed',report=collector.binding(output,output/'report.json'))
    collector.write_json(output/'manifest.json',manifest,replace=True)
    return report


def read_completed(output, *, expected_manifest_sha256, require_reliable=True, allow_test_fixture=False):
    """Pure saved-record/tree metric reconciliation; never resumes collection."""
    if type(require_reliable) is not bool: raise ValueError('Explicit reliability requirement needed')
    _sha(expected_manifest_sha256)
    if file_hash(Path(output)/'manifest.json') != expected_manifest_sha256: raise ValueError('External completed tree manifest differs')
    output,plan,manifest,values,config,operation=_load(output,allow_test_fixture)
    if manifest['status'] != 'completed': raise ValueError('Only a completed tree record can be read')
    report=_completed(output,plan,manifest,operation,allow_test_fixture)
    if require_reliable and report['reliable'] is not True: raise ValueError('Candidate tree did not pass its unchanged reliability gates')
    if file_hash(output/'manifest.json') != expected_manifest_sha256: raise ValueError('Completed manifest changed while reading')
    return {'plan':plan,'report':report,'manifest_sha256':expected_manifest_sha256,
        'program':_json_bytes(collector.initial_tree_reader.bound_bytes(output,report['program'])),
        'fit_result':_json_bytes(collector.initial_tree_reader.bound_bytes(output,report['fit_result'])),
        'release_ready':False,'explanation_qualified':False,'new_environment_steps':0,'new_fit_calls':0}
