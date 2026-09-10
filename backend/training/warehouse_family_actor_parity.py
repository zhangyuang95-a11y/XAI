"""Calibration-only numeric parity for exact registered observed197 runtimes.

Both implementations execute the same six exported NPZ tensors. This is not
an independent checkpoint-export proof, capability test or qualification.
Boundary states must be reached by predeclared public joint actions from an
actual calibration initial state; no position/battery/timer edits are made.
These are audit-action witnesses, not autonomous neural-policy trajectories.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend import warehouse_runtime_family as registry
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_partner_mix_run import write_json, write_bytes
from .warehouse_native_public_feedback_initialization import initialization_sha256 as parameter_digest
from .warehouse_native_revision_provenance import _absolute, _open_dir, _bytes, _json
from env.warehouse_native.scenarios import SCENARIO_VERSION, scenario_fingerprint
from env.warehouse_native.policy import ACTIONS

VERSION='warehouse-family-actor-numeric-parity.v1'
WITNESS_VERSION='warehouse-calibration-public-action-witness.v1'
REQUIRED_COVERAGE=('normal','low_battery','zero_battery','carrying','charging','collision','timer_boundary')
EXPECTED_KEYS={'runtime_family','runtime_version','runtime_signature','actor_sha256','protocol_sha256',
    'scenario_manifest_sha256','actor_parameters_sha256','feature_names_sha256'}
SHAPES={'0.weight':(128,197),'0.bias':(128,),'2.weight':(128,128),'2.bias':(128,),'4.weight':(5,128),'4.bias':(5,)}


def execution_sources():
    result=registry.execution_sources()
    for path in (Path(__file__),Path(__file__).with_name('warehouse_native_partner_mix_run.py'),
            Path(__file__).with_name('warehouse_native_public_feedback_initialization.py'),
            Path(__file__).with_name('warehouse_native_revision_provenance.py'),
            Path(__file__).with_name('warehouse_native_common.py')):
        result[str(path.relative_to(ROOT))]=file_hash(path)
    return result


def _same(left,right,reason):
    if digest(left)!=digest(right):raise ValueError(reason)


def _tensors(runtime,expected):
    """Typed CPU copies of the exact exported six tensors; never torch.load/model construction."""
    raw=_bytes(runtime._actor_path)
    if sha256(raw).hexdigest()!=expected['actor_sha256']:raise ValueError('Actual Actor bytes differ')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!=set(SHAPES)|{'metadata_json'}:raise ValueError('Unexpected Actor arrays')
        tensors={}
        for name,shape in SHAPES.items():
            value=archive[name]
            if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Expected exact finite float32 Actor tensors')
            if not np.array_equal(value,runtime.actor.weights[name]):raise ValueError('Runtime cached Actor differs from original export')
            tensors[name]=torch.from_numpy(value.copy())
    actual=parameter_digest(tensors)
    if actual!=expected['actor_parameters_sha256']:raise ValueError('Exported Actor parameters differ from external semantic anchor')
    recorded=runtime.actor.metadata.get('actor_parameters_sha256')
    if recorded is not None and recorded!=actual:raise ValueError('Recorded Actor parameter hash differs')
    return tensors


def _inputs(runtime,scenarios,witnesses,expected,fixture):
    if type(fixture) is not bool or type(expected) is not dict or set(expected)!=EXPECTED_KEYS:
        raise ValueError('Complete external runtime, feature and Actor parameter bindings required')
    identity=registry.verify(runtime,allow_test_fixture=fixture,expected_family=expected['runtime_family'],
        expected_signature=expected['runtime_signature'])
    for key in ('runtime_version','actor_sha256','protocol_sha256'):_same(identity[key],expected[key],'Runtime input binding differs: '+key)
    for key in EXPECTED_KEYS-{'runtime_family','runtime_version'}:
        if type(expected[key]) is not str or len(expected[key])!=64 or any(x not in '0123456789abcdef' for x in expected[key]):
            raise ValueError('Malformed external digest: '+key)
    if digest(runtime.actor.metadata['feature_names'])!=expected['feature_names_sha256'] or len(runtime.actor.metadata['feature_names'])!=197:
        raise ValueError('Ordered 197 input feature contract differs')
    if scenarios.get('version')!=SCENARIO_VERSION or scenarios.get('test_fixture',False) is not fixture:
        raise ValueError('The original physical scenario manifest is required')
    if digest(scenarios)!=expected['scenario_manifest_sha256'] or runtime.actor.metadata['scenario_manifest_sha256']!=digest(scenarios):
        raise ValueError('Actor and calibration scenario manifest differ')
    _same(scenarios['configuration'],asdict(runtime.config),'Calibration physical configuration differs')
    if 'split' in scenarios:_same(scenarios['split'],scenarios['splits'],'Conflicting split aliases')
    if (type(witnesses) is not dict or set(witnesses)!={'version','test_fixture','calibration_scene_count','maximum_environment_steps','segments'}
            or witnesses['version']!=WITNESS_VERSION or witnesses['test_fixture'] is not fixture):
        raise ValueError('An explicit fixed public-action witness plan is required')
    count=witnesses['calibration_scene_count'];horizon=runtime.config.horizon
    if type(count) is not int or not 1<=count<=12 or (not fixture and count!=12):raise ValueError('Use the first twelve fixed calibration scenes')
    scenes=scenarios['splits']['calibration'][:count]
    if len(scenes)!=count:raise ValueError('Missing fixed calibration initial states')
    excluded={s['fingerprint'] for name,pool in scenarios['splits'].items() if name!='calibration' for s in pool}
    physical=set()
    for index,scene in enumerate(scenes):
        if (scene['id']!=f'calibration_{index:04d}' or scene.get('split','calibration')!='calibration'
                or scene.get('test_fixture',False) is not fixture or any(k.startswith('public_feedback') for k in scene['snapshot'])):
            raise ValueError('Only original calibration starts with unknown history are accepted')
        env=runtime.environment(scene)
        if env.state.frame!=0 or env.done or scenario_fingerprint(env)!=scene['fingerprint']:
            raise ValueError('Calibration physical fingerprint or initial frame differs')
        if scene['fingerprint'] in physical|excluded:raise ValueError('Calibration initial state overlaps another registered state')
        physical.add(scene['fingerprint'])
    segments=witnesses['segments'];used=set();ids=set();amounts={s['id']:0 for s in scenes}
    by_id={s['id']:s for s in scenes}
    if type(segments) is not list or not segments or len(segments)>count*horizon:
        raise ValueError('Finite witness segments are required')
    for segment in segments:
        if type(segment) is not dict or set(segment)!={'id','scenario_id','initial_fingerprint','actions'}:
            raise ValueError('Witnesses contain only a calibration identity and public joint actions')
        name=segment['id'];scene=by_id.get(segment['scenario_id']);actions=segment['actions']
        if (type(name) is not str or not name or name in ids or scene is None or segment['initial_fingerprint']!=scene['fingerprint']
                or type(actions) is not list or not 1<=len(actions)<=horizon):raise ValueError('Invalid witness identity, fingerprint or horizon')
        for action in actions:
            if type(action) is not dict or set(action)!={'robot_1','robot_2'} or any(type(a) is not str or a not in ACTIONS for a in action.values()):
                raise ValueError('Witness contains a nonpublic joint action')
        ids.add(name);used.add(scene['id']);amounts[scene['id']]+=len(actions)
    maximum=sum(amounts.values())
    if used!=set(by_id) or any(n>horizon for n in amounts.values()) or type(witnesses['maximum_environment_steps']) is not int or witnesses['maximum_environment_steps']!=maximum:
        raise ValueError('Witness consumption exceeds its frozen per-scene or total cap')
    if not fixture and maximum>12*120:raise ValueError('Production calibration cap exceeded')
    tensors=_tensors(runtime,expected)
    return identity,scenes,tensors,maximum


def _categories(env,role,info):
    agent=env.state.agents[role];values=['normal']
    if 0<agent.battery<=8:values.append('low_battery')
    if agent.battery==0:values.append('zero_battery')
    if agent.carrying_task_id is not None:values.append('carrying')
    if tuple(agent.position)==tuple(env.layout.charger_position) and agent.last_battery_delta>0:values.append('charging')
    if info is not None and info.get('robot_collision') is True:values.append('collision')
    if env.state.frame>=env.config.horizon-1:values.append('timer_boundary')
    return values


def compare_logits(actual,expected):
    """Compare numeric outputs and actual native float32-softmax actions.

    The same NumPy float32 probability transform as NumPyNativeActor.act is
    applied to both backends. Raw-logit argmax is only a separate diagnostic:
    float32 exponentiation/normalization can collapse nearby logits to a tie.
    """
    actual=np.asarray(actual);expected=np.asarray(expected)
    if (actual.dtype!=np.float32 or expected.dtype!=np.float32 or actual.ndim!=2
            or actual.shape!=expected.shape or actual.shape[1]!=5 or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError('Identical finite float32 Nx5 Actor outputs are required')
    probabilities=[]
    for values in (actual,expected):
        result=np.exp(values-values.max(axis=-1,keepdims=True))
        result/=result.sum(axis=-1,keepdims=True)
        probabilities.append(result)
    actual_prob,expected_prob=probabilities
    actual_actions=actual_prob.argmax(-1);expected_actions=expected_prob.argmax(-1)
    agreement=actual_actions==expected_actions
    raw_agreement=actual.argmax(-1)==expected.argmax(-1)
    error=np.abs(actual-expected)
    return {'numpy_logits':actual,'torch_logits':expected,
        'numpy_probabilities':actual_prob,'torch_probabilities':expected_prob,
        'numpy_action_indices':actual_actions,'torch_action_indices':expected_actions,
        'maximum_absolute_error':float(error.max()),
        'maximum_probability_error':float(np.abs(actual_prob-expected_prob).max()),
        'argmax_semantics':'native_float32_exp_normalize_probabilities',
        'argmax_mismatches':int((~agreement).sum()),'all_argmax_equal':bool(agreement.all()),
        'raw_logit_argmax_mismatches':int((~raw_agreement).sum()),
        'raw_logit_argmax_all_equal':bool(raw_agreement.all()),
        'passed':bool(error.max()<=1e-4 and agreement.all())}


def compare_rows(runtime,observations,tensors):
    """One NumPy batch versus a CPU functional pass using the same NPZ tensors."""
    rows=np.asarray(observations)
    if rows.dtype!=np.float32 or rows.ndim!=2 or rows.shape[1]!=197 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('Identical finite float32 Nx197 observations are required')
    with torch.inference_mode():
        value=torch.from_numpy(rows.copy())
        for layer in (0,2,4):
            value=F.linear(value,tensors[f'{layer}.weight'],tensors[f'{layer}.bias'])
            if layer!=4:value=torch.tanh(value)
        expected=value.numpy().copy()
    actual=runtime.actor.logits(rows)
    return compare_logits(actual,expected)


def run(runtime,scenarios,witnesses,*,expected_bindings,output,allow_test_fixture=False):
    """One nonresumable audit directory; reserve -> actual step -> durable raw -> ack.

An interrupted/failed output is retained and can never be automatically adopted
or retried. Callers still own authorization and any cross-audit budget policy.
"""
    output=_absolute(output)
    if output.exists() or output.is_symlink():raise FileExistsError('Parity output must be new; failed attempts cannot be resumed')
    actor_parent=runtime._actor_path.parent if hasattr(runtime,'_actor_path') else None
    if actor_parent is not None and (output==actor_parent or output.is_relative_to(actor_parent) or actor_parent.is_relative_to(output)):
        raise ValueError('Parity output must be separate from original Actor storage')
    identity,scenes,tensors,maximum=_inputs(runtime,scenarios,witnesses,expected_bindings,allow_test_fixture)
    source=execution_sources();original_scenes=digest(scenarios);original_witness=digest(witnesses)
    binding={'version':VERSION,'runtime':identity,'expected_bindings':deepcopy(expected_bindings),
        'witnesses_sha256':original_witness,'source_sha256':digest(source),'maximum_environment_steps':maximum,
        'test_fixture':allow_test_fixture,'scope':'same_NPZ_tensors_numeric_execution_not_checkpoint_export_provenance'}
    fd=_open_dir(output.parent);os.close(fd);output.mkdir(mode=0o700,exist_ok=False)
    fd=os.open(output.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    write_json(output/'binding.json',binding);write_json(output/'witnesses.json',witnesses);write_json(output/'sources.json',source)
    journal={'version':VERSION,'binding_sha256':digest(binding),'cap':maximum,'reserved':0,'acknowledged':0,'operations':[],'status':'prepared'}
    write_json(output/'journal.json',journal)
    observations=[];origins=[];snapshots=[];step_returns=attempted=0;numeric_calls=0

    def current():
        checked=registry.verify(runtime,allow_test_fixture=allow_test_fixture,
            expected_family=identity['family'],expected_signature=identity['runtime_signature'])
        _same(checked,identity,'Runtime input identity changed during parity audit')
        if execution_sources()!=source or digest(scenarios)!=original_scenes or digest(witnesses)!=original_witness:
            raise ValueError('Parity sources or fixed inputs changed during audit')

    def add(env,segment,info=None):
        snapshot=env.snapshot();restored=runtime.from_snapshot(snapshot)
        original=env.observations();roundtrip=restored.observations()
        for role,key in enumerate(('robot_1','robot_2')):
            if not np.array_equal(original[key],roundtrip[key]):raise ValueError('Confirmed public history does not restore identical 197 observations')
            observations.append(original[key].copy());snapshots.append(snapshot)
            origins.append({'row':len(origins),'segment':segment['id'],'scenario_id':segment['scenario_id'],
                'initial_fingerprint':segment['initial_fingerprint'],'frame':env.state.frame,'agent_id':key,
                'snapshot_sha256':digest(snapshot),'categories':_categories(env,role,info),
                'origin':'calibration_initial' if env.state.frame==0 else 'actual_public_action_witness'})

    try:
        by_id={s['id']:s for s in scenes}
        for segment_index,segment in enumerate(witnesses['segments']):
            current();env=runtime.environment(by_id[segment['scenario_id']]);add(env,segment)
            for index,actions in enumerate(segment['actions']):
                if env.done:raise ValueError('Declared witness attempts a step after the real terminal state')
                current();before=env.snapshot()
                op={'segment':segment_index,'step':index,'before_sha256':digest(before),'actions':deepcopy(actions),
                    'status':'reserved','record':None}
                reserved=deepcopy(journal);reserved['reserved']+=1;reserved['operations'].append(op);reserved['status']='collecting'
                if reserved['reserved']>maximum:raise ValueError('Permanent parity reservation cap exceeded')
                write_json(output/'journal.json',reserved,replace=True);journal=reserved
                attempted+=1
                _,rewards,terminated,truncated,info=env.step(deepcopy(actions));step_returns+=1
                if info['requested_actions']!=actions:raise ValueError('Environment altered public audit input')
                record={'binding_sha256':digest(binding),'segment':deepcopy(segment['id']),'scenario_id':segment['scenario_id'],
                    'initial_fingerprint':segment['initial_fingerprint'],'before':before,'after':env.snapshot(),
                    'submitted_audit_actions':deepcopy(actions),'executed_actions':info['executed_actions'],
                    'rewards':rewards,'info':info,'terminated':terminated,'truncated':truncated,
                    'neural_policy_used_for_witness_actions':False,'state_interventions':[]}
                name=f'transitions/{segment_index:04d}_{index:04d}.json';write_json(output/name,record)
                confirmed=deepcopy(journal);confirmed['acknowledged']+=1
                confirmed['operations'][-1].update(status='acknowledged',record={'path':name,'sha256':file_hash(output/name),'size':(output/name).stat().st_size})
                write_json(output/'journal.json',confirmed,replace=True);journal=confirmed
                add(env,segment,info)
        current();journal=deepcopy(journal);journal['status']='collection_complete'
        write_json(output/'journal.json',journal,replace=True)
        matrix=np.asarray(observations,dtype=np.float32)
        raw=io.BytesIO();np.savez_compressed(raw,observations=matrix)
        write_bytes(output/'observations.npz',raw.getvalue());write_json(output/'row_provenance.json',{'rows':origins,'snapshots':snapshots})
        numeric_calls+=1;comparison=compare_rows(runtime,matrix,tensors)
        arrays={name:comparison.pop(name) for name in ('numpy_logits','torch_logits','numpy_probabilities','torch_probabilities',
            'numpy_action_indices','torch_action_indices')}
        raw=io.BytesIO();np.savez_compressed(raw,**arrays)
        write_bytes(output/'numeric_outputs.npz',raw.getvalue())
        coverage={name:sum(name in row['categories'] for row in origins) for name in REQUIRED_COVERAGE}
        missing=[name for name,n in coverage.items() if not n]
        current();_tensors(runtime,expected_bindings)
        report={'version':VERSION,'status':'numeric_parity_passed' if comparison['passed'] and not missing else 'numeric_parity_failed',
            'passed':comparison['passed'] and not missing,'numerical_result':comparison,'coverage':coverage,'missing_categories':missing,
            'rows':len(matrix),'binding':binding,'journal_sha256':file_hash(output/'journal.json'),
            'actual_environment_steps':step_returns,'acknowledged_environment_steps':journal['acknowledged'],'reserved_environment_steps':journal['reserved'],
            'numpy_batch_forward_calls':numeric_calls,'torch_functional_forward_passes':numeric_calls,'torch_functional_linear_calls':3*numeric_calls,
            'torch_nn_constructions':0,'checkpoint_decodes':0,'optimizer_steps':0,'training_steps':0,'state_mutations_outside_physics':0,
            'audit_witness_reachability_verified':True,'autonomous_neural_reachability_claimed':False,
            'same_exported_tensors_used_by_both_implementations':True,'independent_checkpoint_export_verification':False,
            'release_ready':False,'formal_ready':False,'explanation_qualified':False,'test_fixture':allow_test_fixture,
            'evaluated_splits':['calibration'],'final_test_execution':False,'explanation_test_execution':False}
        write_json(output/'report.json',report)
        return report
    except BaseException as error:
        durable=_json(_bytes(output/'journal.json'))
        failure={'version':VERSION,'status':'failed','reason':type(error).__name__+': '+str(error),
            'known_step_returns':step_returns,'attempted_environment_calls':attempted,
            'confirmed_environment_steps':durable['acknowledged'],'permanently_reserved_steps':durable['reserved'],
            'unconfirmed_actual_steps_unknown':attempted>step_returns,'numeric_passes_attempted':numeric_calls,
            'retry_allowed':False,'qualification_granted':False,'binding_sha256':digest(binding)}
        try:write_json(output/'failure.json',failure)
        except BaseException as log_error:error.audit_journal_error=str(log_error)
        raise
