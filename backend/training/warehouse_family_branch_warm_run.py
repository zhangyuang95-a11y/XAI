"""Finite 40k/arm orchestration with same-source capability initialization for the genuine branch-feedback learner.

The caller supplies externally anchored source/checkpoint, teacher and TRAIN-only
auxiliary material. No collection or fitting is performed here. Reservations are
permanent; an interrupted, unacknowledged operation is never sampled again.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import gzip
import math
import re
import time

import numpy as np

from . import warehouse_family_branch_trainer as learner
from . import warehouse_family_branch_evaluation as evaluation
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save
from .warehouse_native_cycle_budget import CycleBudget
from .warehouse_native_partner_mix_run import write_bytes, write_json, bound, decode, save_batch
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic

VERSION = 'warehouse-family-branch-warm-source-run.v1'
INPUT_VERSION = 'warehouse-family-branch-run-inputs.v1'
BRANCHES = ('control', 'feedback')
PPO_CAP, BATCH_STEPS = 40000, 2000
CHECKPOINTS = (10000, 20000, 30000, 40000)
EVALUATIONS = {'control': [40000], 'feedback': [10000, 30000, 40000]}
ARTIFACTS = ('source_checkpoint', 'source_actor', 'scenarios', 'source_report',
             'baselines', 'teacher', 'auxiliary', 'auxiliary_binding')


def sources():
    result = evaluation.execution_sources()
    for name in (Path(__file__).name, 'warehouse_native_cycle_budget.py',
                 'warehouse_native_partner_mix_run.py'):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _json(path):
    return json.loads(Path(path).read_bytes())


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('An explicit external SHA256 is required')
    return value


def _same(a, b, message):
    if digest(a) != digest(b): raise ValueError(message)


def _check(root, record):
    root = Path(root).resolve(); relative = Path(record['path'])
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Unsafe or missing run artifact')
    if file_hash(path) != _sha(record['sha256']) or ('size' in record and path.stat().st_size != record['size']):
        raise ValueError('Run artifact bytes changed: '+str(path))
    return path


def _external(record):
    if not isinstance(record, dict): raise ValueError('External artifact binding required')
    path = Path(record['path']).expanduser()
    if not path.is_absolute() or not path.is_file() or file_hash(path) != _sha(record['sha256']):
        raise ValueError('External input bytes differ: '+str(path))
    return path.resolve()


def _schedule(fixture, fixture_settings=None):
    if type(fixture) is not bool: raise ValueError('Explicit Boolean fixture scope required')
    if not fixture:
        if fixture_settings is not None: raise ValueError('Production cannot override its finite schedule')
        return {'ppo_cap': PPO_CAP, 'batch_steps': BATCH_STEPS,
            'checkpoints': list(CHECKPOINTS), 'evaluations': deepcopy(EVALUATIONS), 'episode_cap': 120,
            'evaluation_steps_per_boundary': evaluation.STEP_BUDGET}
    if not isinstance(fixture_settings, dict): raise ValueError('Explicit small fixture schedule required')
    value = deepcopy(fixture_settings)
    if set(value) != {'ppo_cap','batch_steps','checkpoints','evaluations','episode_cap','evaluation_steps_per_boundary'}:
        raise ValueError('Incomplete fixture schedule')
    for key in ('ppo_cap','batch_steps','episode_cap','evaluation_steps_per_boundary'):
        if type(value[key]) is not int or value[key] <= 0: raise ValueError('Positive fixture cap required')
    checkpoints = value['checkpoints']
    if (type(checkpoints) is not list or not checkpoints or checkpoints != sorted(set(checkpoints))
            or checkpoints[-1] != value['ppo_cap'] or any(type(x) is not int or x <= 0
                or x % value['batch_steps'] for x in checkpoints)
            or set(value['evaluations']) != set(BRANCHES)):
        raise ValueError('Invalid fixture checkpoints')
    for branch in BRANCHES:
        endpoints = value['evaluations'][branch]
        if (not endpoints or endpoints != sorted(set(endpoints)) or endpoints[-1] != value['ppo_cap']
                or any(x not in checkpoints for x in endpoints)):
            raise ValueError('Fixture evaluations must use registered endpoints')
    return value


def _read_inputs(path, expected, fixture):
    path = Path(path).expanduser().resolve()
    if file_hash(path) != _sha(expected): raise ValueError('Inputs file differs from its external anchor')
    value = _json(path)
    if value.get('version') != INPUT_VERSION or value.get('test_fixture') is not fixture:
        raise ValueError('Input producer or fixture scope differs')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,80}', value.get('cycle_id', '')):
        raise ValueError('An explicit finite cycle ID is required')
    for key in ARTIFACTS: _external(value[key])
    _sha(value['source_native_state_sha256'])
    if type(value.get('source_cumulative_step')) is not int or value['source_cumulative_step'] <= 0:
        raise ValueError('Actual cumulative source clock required')
    origin = value.get('auxiliary_origin', {})
    if origin.get('pool') != 'train': raise ValueError('Only original TRAIN pairs may supply gradients')
    for key in ('training_data_sha256','collection_manifest_sha256'): _sha(origin.get(key))
    return value


def _native_payload(payload):
    if payload.get('version') == learner.native.VERSION: return payload
    outer = payload.get('version'); state = payload.get('trainer', {})
    expected = {'warehouse-family-feedback-cycle-run.v1': learner.frozen.VERSION,
                VERSION: learner.VERSION}
    if outer not in expected or state.get('version') != expected[outer]:
        raise ValueError('Unsupported actual source checkpoint type')
    native = state.get('native_state')
    if not isinstance(native, dict) or native.get('version') != learner.native.VERSION:
        raise ValueError('A genuine complete native source state is required')
    return native


def _actor_equal(trainer, path):
    actual = trainer.model.actor.state_dict()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(actual) | {'metadata_json'}:
            raise ValueError('Actor must contain its exact six arrays and metadata')
        if any(not np.array_equal(value.detach().cpu().numpy(), archive[key]) for key, value in actual.items()):
            raise ValueError('Current actual Torch Actor and source/export NPZ differ')
        metadata = json.loads(str(archive['metadata_json'].item()))
    return metadata, learner.frozen.actor_parameter_sha256(actual)


def _restore_source(inputs, device, fixture):
    checkpoint = _external(inputs['source_checkpoint'])
    state = _native_payload(decode(checkpoint, inputs['source_checkpoint']['sha256']))
    scenes = _json(_external(inputs['scenarios']))
    source = learner.restore_shutdown_source(state, scenes,
        expected_state_sha256=inputs['source_native_state_sha256'], device=device, test_fixture=fixture)
    if source.source_counters['joint_steps'] + source.joint_steps != inputs['source_cumulative_step']:
        raise ValueError('Actual source cumulative clock differs')
    metadata, parameters = _actor_equal(source, _external(inputs['source_actor']))
    if (metadata.get('test_fixture') is not fixture or metadata.get('runtime_action_override') is not False
            or metadata.get('obs_dim') != 197 or metadata.get('state_dim') != 354
            or metadata.get('source_counters', {}).get('joint_steps', -1) + metadata.get('joint_steps', -1)
                != inputs['source_cumulative_step']):
        raise ValueError('Actual source Actor metadata differs')
    return source, scenes, parameters


def _material(inputs, source, scenes, parameters, schedule, fixture):
    baseline = _json(_external(inputs['baselines'])); report = _json(_external(inputs['source_report']))
    validation = scenes['splits']['validation']
    if not fixture:
        evaluation.physical._baseline(baseline['reference'], validation)
        evaluation.physical._baseline(baseline['random'], validation)
    fit = _json(_external(inputs['teacher'])); aux_binding = _json(_external(inputs['auxiliary_binding']))
    with np.load(_external(inputs['auxiliary']), allow_pickle=False) as archive:
        if set(archive.files) != {'observations', 'weights'}: raise ValueError('Only branch observations/weights are allowed')
        observations, weights = archive['observations'].copy(), archive['weights'].copy()
    if (observations.dtype != np.float32 or weights.dtype != np.float32 or not len(weights)
            or not np.all(weights == 1)):
        raise ValueError('All original train pairs require float32 observations and unit weights')
    expected_binding = learner.auxiliary_binding(observations, weights,
        source_actor_sha256=inputs['source_actor']['sha256'], source_actor_parameters_sha256=parameters,
        fit_step=inputs['source_cumulative_step'], evidence_sha256=inputs['auxiliary_origin']['collection_manifest_sha256'])
    _same(aux_binding, expected_binding, 'Actual auxiliary source/clock/arrays differ')
    binding = fit.get('fit_report', {}).get('observed197_bindings', {})
    if (fit.get('version') != learner.TEACHER_VERSION or fit.get('ordinary_reliable') is not True
            or fit.get('reliable') is not True or binding.get('actor_sha256') != inputs['source_actor']['sha256']
            or binding.get('actor_parameters_sha256') != parameters
            or binding.get('cumulative_fit_step') != inputs['source_cumulative_step']
            or binding.get('training_data_sha256') != inputs['auxiliary_origin']['training_data_sha256']):
        raise ValueError('New same-Actor teacher must pass its ordinary reliability gates; no old-tree fallback')
    protocol = learner.make_protocol(source, source_checkpoint_sha256=inputs['source_checkpoint']['sha256'],
        source_state_sha256=inputs['source_native_state_sha256'], ppo_cap=schedule['ppo_cap'], cycle_id=inputs['cycle_id'],
        source_report=report, team_reference=baseline['reference'],
        evaluation_checkpoints=sorted(set(sum(schedule['evaluations'].values(), []))), test_fixture=fixture)
    return protocol, fit, observations, weights, aux_binding


def _trainer(protocol, source, branch, inputs, device, fixture):
    return learner.BranchFeedbackTrainer(protocol, source, feedback_branch=branch,
        expected_source_state_sha256=inputs['source_native_state_sha256'],
        source_checkpoint_sha256=inputs['source_checkpoint']['sha256'], device=device, test_fixture=fixture)


def _initialize_source_gate(trainer, inputs, protocol, parameters, metadata):
    """Reuse only the externally anchored report of these exact source tensors.

    This establishes ramp_start at the inherited cumulative clock. It neither
    re-labels the report as a new evaluation nor advances native learning state.
    Subsequent failed reports supersede it permanently for this run.
    """
    report = _json(_external(inputs['source_report']))
    _same(report, protocol['source_report'], 'Initial report differs from the frozen protocol')
    bindings = report.get('actor_bindings', {})
    required = ('actor_sha256','actor_parameters_sha256','experiment_version','protocol_sha256',
        'source_sha256','source_checkpoint_sha256','initialization_sha256','scenario_manifest_sha256',
        'joint_steps','branch','cycle_id')
    if (any(k not in bindings for k in required)
            or bindings['actor_sha256'] != inputs['source_actor']['sha256']
            or bindings['actor_parameters_sha256'] != parameters
            or any(digest(v) != digest(metadata.get(k)) for k,v in bindings.items() if k != 'actor_sha256')
            or report.get('identity',{}).get('actor_bindings') != bindings
            or report['identity'].get('actor_metadata_sha256') != digest(metadata)):
        raise ValueError('Initial capability report is not bound to the exact source Actor/parameters/metadata')
    clock = inputs['source_cumulative_step']; inherited = metadata.get('source_counters',{}).get('joint_steps')
    if (type(inherited) is not int or inherited + metadata['joint_steps'] != clock
            or report.get('source_joint_steps') != inherited
            or report.get('additional_joint_steps') != metadata['joint_steps']
            or report.get('total_actor_training_joint_steps') != clock
            or trainer.joint_steps != 0 or trainer.feedback_clock != clock):
        raise ValueError('Initial capability report cumulative source clock differs')
    before = semantic(trainer.native.state_dict())
    gate = trainer.update_feedback_gate(report,capability_report_sha256=inputs['source_report']['sha256'])
    if (trainer.capability_evidence.get('capability_eligible') is not True
            or trainer.feedback.current_lambda != 0 or trainer.feedback.ramp_start != clock
            or semantic(trainer.native.state_dict()) != before):
        raise ValueError('Initial same-source gate must open the zero-lambda ramp without native mutation')
    return {'report':deepcopy(inputs['source_report']),'actor_sha256':inputs['source_actor']['sha256'],
        'actor_parameters_sha256':parameters,'source_cumulative_step':clock,'ramp_start':clock,
        'lambda':0.,'new_validation_steps':0,'scope':'same_frozen_source_Actor_validation_reuse',
        'gate':gate}


def _save_checkpoint(path, payload):
    if path.exists(): raise ValueError('Uncommitted checkpoint already exists; no overwrite or replay')
    atomic_torch_save(path, payload)


def prepare(output, inputs_path, *, expected_inputs_sha256, device='mps', allow_test_fixture=False,
            fixture_settings=None):
    schedule = _schedule(allow_test_fixture, fixture_settings)
    inputs = _read_inputs(inputs_path, expected_inputs_sha256, allow_test_fixture)
    output = Path(output).expanduser().resolve()
    if output.exists(): raise FileExistsError('A unique new output directory is required')
    if device not in ('cpu','mps') or (not allow_test_fixture and device != 'mps'):
        raise ValueError('Production uses the actual MPS source device')
    frozen_sources = sources()
    source, scenes, parameters = _restore_source(inputs, device, allow_test_fixture)
    if (schedule['batch_steps'] % source.cfg['environments'] or
            (not allow_test_fixture and source.cfg['environments'] != 16)):
        raise ValueError('Joint batch must preserve the actual environment count')
    protocol, fit, obs, weights, aux_binding = _material(inputs, source, scenes, parameters, schedule, allow_test_fixture)
    source_metadata, verified_parameters = _actor_equal(source, _external(inputs['source_actor']))
    if verified_parameters != parameters: raise ValueError('Source Actor parameters changed before initial gate')
    output.mkdir(parents=True)
    # Source and auxiliary originals remain externally anchored; local JSON is a
    # semantic copy and its own exact bytes are separately bound below.
    for name, value in (('inputs.json',inputs), ('protocol.json',protocol)):
        write_json(output/name, value)
    caps = {b: {'ppo': schedule['ppo_cap'], 'evaluation': len(schedule['evaluations'][b]) *
        schedule['evaluation_steps_per_boundary']} for b in BRANCHES}
    identity = {'version': VERSION, 'cycle_id': inputs['cycle_id'], 'protocol_sha256': digest(protocol),
        'runtime_sources': frozen_sources, 'inputs_sha256': expected_inputs_sha256, 'budget_caps': caps,
        'schedule': schedule, 'test_fixture': allow_test_fixture,
        'initial_gate_contract': {'source_report_sha256':inputs['source_report']['sha256'],
            'source_actor_sha256':inputs['source_actor']['sha256'], 'source_actor_parameters_sha256':parameters,
            'source_cumulative_step':inputs['source_cumulative_step'], 'new_validation_steps':0,
            'later_report_failure_supersedes_source':True}}
    prepared = {'version': VERSION, 'identity': identity, 'inputs': bound(output,output/'inputs.json'),
        'original_inputs': {'path': str(Path(inputs_path).resolve()), 'sha256': expected_inputs_sha256},
        'protocol': bound(output,output/'protocol.json'), 'device': device, 'schedule': schedule,
        'actor_parameters_sha256': parameters, 'test_fixture': allow_test_fixture,
        'new_auxiliary_sampling_steps': 0, 'automatic_teacher_refresh': False,
        'terminal_feedback_closed': True, 'qualification_granted': False}
    ledger = CycleBudget.create(output, identity)
    native_hashes = {}; initial_gate = None
    with ledger.lease():
        for branch in BRANCHES:
            trainer = _trainer(protocol, source, branch, inputs, device, allow_test_fixture)
            if branch == 'feedback':
                trainer.attach_feedback_state(fit['manager_state'], source_actor_sha256=inputs['source_actor']['sha256'],
                    source_actor_parameters_sha256=parameters, evidence_sha256=fit['evidence_sha256'])
                trainer.attach_auxiliary_pool(obs, weights, binding=aux_binding)
                if trainer.feedback.current_lambda != 0 or not trainer.feedback.reliable:
                    raise ValueError('Initial teacher must be ordinary-reliable with lambda zero')
                initial_gate = _initialize_source_gate(trainer,inputs,protocol,parameters,source_metadata)
            state = trainer.state_dict(); native_hashes[branch] = semantic(state['native_state'])
            path = output/'branches'/branch/'checkpoints'/'initial.pt'
            _save_checkpoint(path, {'version': VERSION, 'branch': branch, 'cycle_id': inputs['cycle_id'],
                'identity_sha256': digest(identity), 'trainer': state, 'actual_steps': 0, 'evidence': {}})
            for kind in ('ppo','evaluation'): ledger.initialize_head(kind,branch,str(path.relative_to(output)),file_hash(path))
        if len(set(native_hashes.values())) != 1: raise ValueError('Pair did not retain identical native initial states')
        if sources() != frozen_sources: raise ValueError('Execution sources changed during prepare')
        write_json(output/'initialization.json', {'version': VERSION, 'native_state_sha256_by_branch': native_hashes,
            'identical_native_learning_state': True, 'source_state_sha256': inputs['source_native_state_sha256'],
            'source_actor_parameters_sha256': parameters, 'initial_feedback_gate':initial_gate,
            'new_environment_steps': 0, 'new_optimizer_updates': 0,
            'note': 'Actual source decoding and environment restore/reset are not counted as environment steps.'})
        prepared['initialization'] = bound(output,output/'initialization.json')
        write_json(output/'prepared.json', prepared)
    return prepared


def read_prepared(output, *, allow_test_fixture=False):
    output = Path(output).resolve(); p = _json(output/'prepared.json')
    if p.get('version') != VERSION or p.get('test_fixture') is not allow_test_fixture:
        raise ValueError('Prepared run identity/fixture differs')
    _same(p['identity']['runtime_sources'], sources(), 'Prepared execution source bytes changed')
    _same(p['schedule'], _schedule(allow_test_fixture, p['schedule'] if allow_test_fixture else None), 'Schedule differs')
    original = _read_inputs(p['original_inputs']['path'], p['original_inputs']['sha256'], allow_test_fixture)
    inputs = _json(_check(output,p['inputs'])); _same(inputs, original, 'Copied input contract differs')
    protocol = _json(_check(output,p['protocol'])); _check(output,p['initialization'])
    if digest(protocol) != p['identity']['protocol_sha256'] or p['identity']['schedule'] != p['schedule']:
        raise ValueError('Prepared protocol/schedule identity differs')
    ledger = CycleBudget(output,p['identity'])
    return output,p,inputs,protocol,ledger


def _envelope(output, ledger, branch, payload, head):
    if (payload.get('version') != VERSION or payload.get('branch') != branch
            or payload.get('cycle_id') != ledger.identity['cycle_id']
            or payload.get('identity_sha256') != digest(ledger.identity)
            or payload['trainer']['native_state']['joint_steps'] != head['step']):
        raise ValueError('Checkpoint identity/current native step differs')
    for binding in payload.get('evidence',{}).values(): _check(output,binding)


def effective_checkpoint(output, ledger, branch):
    output = Path(output); head = ledger.head('ppo',branch)
    payload = decode(_check(output,head),head['sha256']); _envelope(output,ledger,branch,payload,head)
    if head['step']:
        op = ledger.read()['operations'].get(payload.get('operation_id'),{})
        if (op.get('status') != 'acknowledged' or op['completion']['checkpoint'] != head
                or payload.get('audit',{}).get('neural_overrides') != 0):
            raise ValueError('Missing actual PPO ACK or original zero-override audit')
    marker = output/'branches'/branch/'boundaries'/f"step_{head['step']:07d}.json"
    if marker.exists():
        record = _json(marker)
        if record.get('version') != VERSION or record.get('branch') != branch or record.get('predecessor_head') != head:
            raise ValueError('Boundary is not bound to this acknowledged learner')
        for evidence in record['evidence'].values(): _check(output,evidence)
        newer = decode(_check(output,record['checkpoint']),record['checkpoint']['sha256'])
        _envelope(output,ledger,branch,newer,head)
        if (newer.get('predecessor_head') != head or newer.get('gate') != record.get('gate')
                or newer.get('evidence') != record['evidence']
                or semantic(newer['trainer']['native_state']) != semantic(payload['trainer']['native_state'])):
            raise ValueError('A zero-step gate boundary changed native parameters/Adam/environment/RNG')
        payload = newer
    return head,payload


def _verify_evaluation_receipt(output, ledger, branch, operation, receipt):
    """Saved-record verification only; no Actor construction or physics replay."""
    step = receipt.get('step'); schedule = ledger.identity['schedule']
    if (receipt.get('version') != VERSION or receipt.get('branch') != branch
            or step not in schedule['evaluations'][branch] or ledger.head('ppo',branch)['step'] != step):
        raise ValueError('Pending evaluation receipt has another branch/endpoint')
    folder = output/'branches'/branch/'validation'/f'step_{step:07d}'
    actor = output/'branches'/branch/'actors'/f'actor_{step:07d}.npz'
    if receipt.get('actor_sha256') != file_hash(actor): raise ValueError('Pending evaluation Actor changed')
    callback = receipt['context']
    additions = {'actual_steps','row_path','row_sha256','trace_path','trace_sha256'}
    context = {k:v for k,v in callback.items() if k not in additions}
    manifest = _json(folder/'manifest.json'); identity = manifest['identity']
    unhashed = {k:v for k,v in context.items() if k != 'operation_id'}
    if (context['operation_id'] != operation['opid'] or context['feedback_branch'] != branch
            or context['cycle_id'] != ledger.identity['cycle_id']
            or context['maximum_environment_steps'] != operation['request']['steps']
            or context['actor_bindings']['actor_sha256'] != file_hash(actor)
            or identity.get('protocol_sha256') != ledger.identity['protocol_sha256']
            or identity.get('actor_bindings') != context['actor_bindings']
            or context['operation_id'] != digest({'evaluation':digest(identity),'episode':unhashed})):
        raise ValueError('Original pending evaluation reservation identity differs')
    entries = [entry for entry in manifest['episodes'] if entry['context'].get('operation_id') == operation['opid']]
    if len(entries) != 1: raise ValueError('Exactly one complete saved episode is required')
    entry = entries[0]; row = evaluation.compact._read_entry(folder,entry,context)
    row_path = _check(output,receipt['row']); trace_path = _check(output,receipt['trace'])
    if (row_path != (folder/entry['row']['path']).resolve() or trace_path != (folder/entry['trace']['path']).resolve()
            or callback['row_sha256'] != receipt['row']['sha256'] or callback['trace_sha256'] != receipt['trace']['sha256']
            or Path(callback['row_path']).resolve() != row_path or Path(callback['trace_path']).resolve() != trace_path
            or callback['actual_steps'] != row['steps']):
        raise ValueError('Pending full-record receipt differs from its original episode')
    evaluation.verify_saved_trace(trace_path.read_bytes(),context,row)
    return row['steps']


def recover_pending(output, ledger, branch, trainer):
    """Only commit already complete original artifacts; never execute a sample.

    The genuine trainer loader validates recovered full optimizer/environment/RNG
    state. Missing or partial evidence remains pending with no refund.
    """
    output = Path(output); data = ledger.read()
    for kind in ('ppo','evaluation'):
        opid = data['branches'][branch][kind]['pending']
        if opid is None: continue
        operation = data['operations'][opid]; request = operation['request']
        if kind == 'ppo':
            old_head = operation['old_head']; amount = request['steps']; step = old_head['step']+amount
            path = output/'branches'/branch/'checkpoints'/f'step_{step:07d}.pt'
            if not path.is_file(): raise ValueError('Pending PPO has no complete checkpoint; never resample or refund')
            sha = file_hash(path); payload = decode(path,sha)
            new_head = {'path':str(path.relative_to(output)),'sha256':sha,'step':step}
            _envelope(output,ledger,branch,payload,new_head)
            if (opid != f'ppo_{branch}_{step:07d}' or payload.get('operation_id') != opid
                    or payload.get('predecessor_head') != old_head or payload.get('actual_steps') != amount
                    or old_head != ledger.head('ppo',branch) or trainer.joint_steps != old_head['step']
                    or payload.get('before_state_sha256') != semantic(trainer.state_dict())
                    or payload.get('audit',{}).get('neural_overrides') != 0 or not _finite(payload.get('metrics'))
                    or set(payload.get('evidence',{})) != {'arrays','trace'}):
                raise ValueError('Pending PPO checkpoint does not match its original reservation/predecessor')
            rows = [json.loads(raw) for raw in gzip.decompress(_check(output,payload['evidence']['trace']).read_bytes()).splitlines()]
            with np.load(_check(output,payload['evidence']['arrays']),allow_pickle=False) as archive:
                if not archive.files or any(archive[k].dtype.hasobject for k in archive.files):
                    raise ValueError('Missing original numerical PPO batch')
            if (len(rows) != amount or any(r.get('version') != learner.VERSION or r.get('feedback_branch') != branch
                    or r.get('shutdown_arm') != trainer.shutdown_arm or r.get('own_shutdown_beta') != trainer.beta for r in rows)):
                raise ValueError('Original saved PPO trace rows disagree with their reserved amount')
            staged = deepcopy(trainer); staged.load_state_dict(payload['trainer'])
            if semantic(staged.state_dict()) != semantic(payload['trainer']):
                raise ValueError('Actual full loader did not preserve the recovered checkpoint')
            ledger.ack(opid,new_head['path'],sha,amount)
            trainer.load_state_dict(payload['trainer'])
        else:
            path = output/'branches'/branch/'evaluation_receipts'/(opid+'.json')
            if not path.is_file(): raise ValueError('Unconfirmed evaluation lacks a complete durable receipt; never resample')
            receipt = _json(path); actual = _verify_evaluation_receipt(output,ledger,branch,operation,receipt)
            ledger.ack(opid,str(path.relative_to(output)),file_hash(path),actual)
        data = ledger.read()
    return ledger.head('ppo',branch)


def _recover_boundary(output, ledger, branch, trainer):
    head = ledger.head('ppo',branch); folder = output/'branches'/branch/'boundaries'
    path = folder/f"step_{head['step']:07d}.pt"; marker = path.with_suffix('.json')
    if marker.exists() or not path.exists(): return
    payload = decode(path,file_hash(path)); _envelope(output,ledger,branch,payload,head)
    if (payload.get('predecessor_head') != head or payload.get('actual_steps') != 0
            or semantic(payload['trainer']['native_state']) != semantic(trainer.native.state_dict())):
        raise ValueError('Interrupted boundary changed the confirmed native learning state')
    staged = deepcopy(trainer); staged.load_state_dict(payload['trainer'])
    if semantic(staged.state_dict()) != semantic(payload['trainer']): raise ValueError('Incomplete boundary trainer state')
    write_json(marker,{'version':VERSION,'branch':branch,'step':head['step'],'new_training_steps':0,
        'predecessor_head':head,'checkpoint':bound(output,path),'evidence':payload['evidence'],'gate':payload['gate'],
        'feedback_closed_pending_refresh':staged.refresh_pending if branch=='feedback' else False})
    trainer.load_state_dict(payload['trainer'])


def _finite(value):
    if isinstance(value,dict): return all(_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(_finite(v) for v in value)
    if isinstance(value,(float,np.floating)): return math.isfinite(value)
    return True


def train_to(output, trainer, ledger, branch, until):
    """Requires one caller-held lease; committed operations never run again."""
    output = Path(output); schedule = ledger.identity['schedule']; batch_steps = schedule['batch_steps']
    if type(until) is not int or until % batch_steps or not trainer.joint_steps <= until <= schedule['ppo_cap']:
        raise ValueError('Target must be a finite registered joint-batch boundary')
    head,payload = effective_checkpoint(output,ledger,branch)
    if head['step'] != trainer.joint_steps or semantic(payload['trainer']) != semantic(trainer.state_dict()):
        raise ValueError('In-memory learner differs from its confirmed checkpoint')
    while trainer.joint_steps < until:
        data = ledger.read()
        if any(data['branches'][branch][k]['pending'] for k in ('ppo','evaluation')):
            raise ValueError('Pending operation is not replay permission; reservation remains consumed')
        if trainer.joint_steps in schedule['checkpoints'] and not (output/'branches'/branch/'boundaries'/f'step_{trainer.joint_steps:07d}.json').exists():
            raise ValueError('Registered checkpoint/export/evaluation boundary must complete before more PPO')
        amount = min(batch_steps,until-trainer.joint_steps)
        opid = f'ppo_{branch}_{trainer.joint_steps+amount:07d}'
        reservation = ledger.reserve('ppo',branch,amount,opid,expected_step=trainer.joint_steps)
        if not reservation['execution_permitted']: raise ValueError('Duplicate request is not permission to resample')
        before = semantic(trainer.state_dict()); previous = deepcopy(head)
        capability = trainer.capability_evidence
        if (branch == 'feedback' and capability is not None and trainer.feedback.reliable
                and not trainer.refresh_pending):
            trainer.update_feedback_gate(capability['report'], capability_report_sha256=capability['report_sha256'])
        started = time.monotonic(); batch,metrics = trainer.train_chunk(amount//trainer.cfg['environments'])
        if (trainer.joint_steps != previous['step']+amount or len(batch['transition_records']) != amount
                or batch['audit'].get('neural_overrides') != 0 or not _finite(metrics)
                or any(r.get('version') != learner.VERSION or r.get('feedback_branch') != branch
                    or r.get('shutdown_arm') != trainer.shutdown_arm or r.get('own_shutdown_beta') != trainer.beta
                    for r in batch['transition_records'])
                or (branch == 'control' and metrics.get('feedback_loss',0) != 0)):
            raise ValueError('Actual batch/metrics differ from the reserved neural-only contract')
        evidence = save_batch(output,branch,opid,batch)
        episodes = deepcopy(trainer.completed_episodes); trainer.completed_episodes.clear()
        payload = {'version': VERSION, 'branch': branch, 'cycle_id': ledger.identity['cycle_id'],
            'identity_sha256': digest(ledger.identity), 'operation_id': opid, 'predecessor_head': previous,
            'before_state_sha256': before, 'trainer': trainer.state_dict(), 'audit': batch['audit'],
            'metrics': metrics, 'evidence': evidence, 'actual_steps': amount, 'completed_episodes': episodes,
            'elapsed_seconds': time.monotonic()-started}
        path = output/'branches'/branch/'checkpoints'/f'step_{trainer.joint_steps:07d}.pt'
        _save_checkpoint(path,payload)
        ledger.ack(opid,str(path.relative_to(output)),file_hash(path),amount)
        head = ledger.head('ppo',branch)
        write_json(output/'progress.json',{'version':VERSION,'branch':branch,'joint_steps':trainer.joint_steps,
            'metrics':metrics,'totals':ledger.read()['totals'],'qualification_granted':False},replace=True)
        print(json.dumps({'event':'branch_ppo_ack','branch':branch,'step':trainer.joint_steps,
            'feedback_lambda':metrics.get('feedback_lambda'), 'auxiliary_rows':metrics.get('branch_update',{}).get('auxiliary_endpoints_used')}),flush=True)


def _evaluate(output, p, inputs, protocol, ledger, branch, step, actor):
    folder = output/'branches'/branch/'validation'/f'step_{step:07d}'
    operations = ledger.read()['operations']
    if ledger.read()['branches'][branch]['evaluation']['pending']:
        raise ValueError('Unconfirmed evaluation cannot be sampled again')
    confirmed = {key for key,op in operations.items() if op['request']['kind']=='evaluation'
        and op['request']['branch']==branch and op['status']=='acknowledged'}
    scenes = _json(_external(inputs['scenarios']))['splits']['validation']
    baselines = _json(_external(inputs['baselines']))
    actor_sha = file_hash(actor)
    def before(context):
        if (context['feedback_branch'] != branch or context['actor_bindings']['actor_sha256'] != actor_sha
                or context['maximum_environment_steps'] != p['schedule']['episode_cap']):
            raise ValueError('Evaluation request source or cap differs')
        reserved = ledger.reserve('evaluation',branch,context['maximum_environment_steps'],context['operation_id'])
        if not reserved['execution_permitted']: raise ValueError('Repeated evaluation is not executable')
    def after(context):
        op = ledger.read()['operations'][context['operation_id']]
        if op['status'] != 'pending' or op['request']['branch'] != branch:
            raise ValueError('Evaluation completion lacks its actual pending reservation')
        row = Path(context['row_path']).resolve(); trace = Path(context['trace_path']).resolve()
        if not row.is_relative_to(folder.resolve()) or not trace.is_relative_to(folder.resolve()):
            raise ValueError('Evaluation callback points outside its registered boundary')
        if file_hash(row) != context['row_sha256'] or file_hash(trace) != context['trace_sha256']:
            raise ValueError('Actual evaluation row/trace bytes differ before acknowledgment')
        receipt = output/'branches'/branch/'evaluation_receipts'/(context['operation_id']+'.json')
        write_json(receipt,{'version':VERSION,'branch':branch,'step':step,'actor_sha256':actor_sha,
            'context':context,'row':bound(output,row),'trace':bound(output,trace)})
        ledger.ack(context['operation_id'],str(receipt.relative_to(output)),file_hash(receipt),context['actual_steps'])
    kwargs = dict(expected_actor_sha256=actor_sha,reference_report=baselines['reference'],random_report=baselines['random'])
    if (folder/'report.json').exists():
        report = evaluation.read_existing(actor,scenes,protocol,folder,confirmed_operation_ids=confirmed,**kwargs)
    else:
        report = evaluation.evaluate(actor,scenes,protocol,folder,before_episode=before,on_episode=after,
            confirmed_operation_ids=confirmed,**kwargs)
    # Verify every external acknowledgment still covers its full original row/trace.
    for op in ledger.read()['operations'].values():
        if op['request']['kind'] != 'evaluation' or op['request']['branch'] != branch or op['status'] != 'acknowledged': continue
        checkpoint = op['completion']['checkpoint']; receipt = _json(_check(output,checkpoint))
        if receipt['step'] != step: continue
        if receipt['actor_sha256'] != actor_sha or receipt['context']['operation_id'] != op['opid']:
            raise ValueError('Saved validation acknowledgment source differs')
        _check(output,receipt['row']); _check(output,receipt['trace'])
    return report, bound(output,folder/'report.json')


def finish_boundary(output, p, inputs, protocol, trainer, ledger, branch):
    step = trainer.joint_steps; schedule = p['schedule']
    if step not in schedule['checkpoints']: raise ValueError('Unregistered export boundary')
    folder = output/'branches'/branch/'boundaries'; marker = folder/f'step_{step:07d}.json'
    if marker.exists():
        _,payload = effective_checkpoint(output,ledger,branch); trainer.load_state_dict(payload['trainer'])
        return _json(marker)
    head = ledger.head('ppo',branch); native_before = semantic(trainer.native.state_dict())
    actor = output/'branches'/branch/'actors'/f'actor_{step:07d}.npz'
    if not actor.exists(): trainer.export(actor)
    metadata, parameters = _actor_equal(trainer,actor)
    if (metadata.get('experiment_version') != learner.VERSION or metadata.get('joint_steps') != step
            or metadata.get('feedback_branch') != branch or metadata.get('protocol_sha256') != digest(protocol)):
        raise ValueError('Boundary Actor identity differs')
    evidence = {'actor':bound(output,actor)}; report = None; gate = None
    if step in schedule['evaluations'][branch]:
        report, report_binding = _evaluate(output,p,inputs,protocol,ledger,branch,step,actor)
        evidence['report'] = report_binding
        gate = learner.gate_values(report, protocol['source_report'],team_reference=protocol['team_reference'])
        if branch == 'feedback' and step < schedule['ppo_cap']:
            trainer.update_feedback_gate(report,capability_report_sha256=report_binding['sha256'])
    if branch == 'feedback' and step == schedule['ppo_cap'] and not trainer.refresh_pending:
        trainer.begin_feedback_refresh()
    if semantic(trainer.native.state_dict()) != native_before:
        raise ValueError('Export/validation/gate changed the actual native learning state')
    receipt = folder/f'step_{step:07d}_actor.json'
    if not receipt.exists(): write_json(receipt,{'version':VERSION,'branch':branch,'step':step,
        'actor_sha256':file_hash(actor),'actor_parameters_sha256':parameters,'all_six_arrays_equal':True,
        'checkpoint':head,'new_neural_forward_calls':0,'additional_checkpoint_decodes':0})
    receipt_value = _json(receipt)
    if (receipt_value.get('actor_sha256') != file_hash(actor) or receipt_value.get('actor_parameters_sha256') != parameters
            or receipt_value.get('checkpoint') != head): raise ValueError('Actor tensor receipt differs')
    evidence['actor_tensor_receipt'] = bound(output,receipt)
    payload = {'version':VERSION,'branch':branch,'cycle_id':ledger.identity['cycle_id'],
        'identity_sha256':digest(ledger.identity),'predecessor_head':head,'trainer':trainer.state_dict(),
        'actual_steps':0,'evidence':evidence,'gate':gate}
    checkpoint = folder/f'step_{step:07d}.pt'; _save_checkpoint(checkpoint,payload)
    record = {'version':VERSION,'branch':branch,'step':step,'new_training_steps':0,'predecessor_head':head,
        'checkpoint':bound(output,checkpoint),'evidence':evidence,'gate':gate,
        'feedback_closed_pending_refresh':trainer.refresh_pending if branch=='feedback' else False}
    write_json(marker,record)
    return record


def advance(output, *, until=None, allow_test_fixture=False):
    output,p,inputs,protocol,ledger = read_prepared(output,allow_test_fixture=allow_test_fixture)
    target = p['schedule']['ppo_cap'] if until is None else until
    if type(target) is not int or target not in p['schedule']['checkpoints']:
        raise ValueError('Advance stops only at registered export boundaries')
    with ledger.lease():
        source,scenes,parameters = _restore_source(inputs,p['device'],allow_test_fixture)
        _same(parameters,p['actor_parameters_sha256'],'Original source Actor parameters differ')
        results = {}
        # Sequential branches share neither learner objects nor RNG/optimizer state.
        for branch in BRANCHES:
            trainer = _trainer(protocol,source,branch,inputs,p['device'],allow_test_fixture)
            head,payload = effective_checkpoint(output,ledger,branch); trainer.load_state_dict(payload['trainer'])
            head = recover_pending(output,ledger,branch,trainer)
            _recover_boundary(output,ledger,branch,trainer)
            if head['step'] > target: raise ValueError('Cannot rewind confirmed PPO progress')
            for endpoint in p['schedule']['checkpoints']:
                if endpoint > target or endpoint < trainer.joint_steps: continue
                train_to(output,trainer,ledger,branch,endpoint)
                finish_boundary(output,p,inputs,protocol,trainer,ledger,branch)
            results[branch] = {'joint_steps':trainer.joint_steps,'usage':deepcopy(trainer.usage),
                'checkpoint':ledger.head('ppo',branch),
                'feedback_closed_pending_refresh':trainer.refresh_pending if branch=='feedback' else False}
        data = ledger.read()
        if any(cell[k]['pending'] for cell in data['branches'].values() for k in ('ppo','evaluation')):
            raise ValueError('Cannot complete with an unconfirmed operation')
        boundaries = {b:_json(output/'branches'/b/'boundaries'/f'step_{target:07d}.json') for b in BRANCHES}
        result = {'version':VERSION,'cycle_id':inputs['cycle_id'],'until':target,'status':'fixed_endpoint_completed'
            if target==p['schedule']['ppo_cap'] else 'registered_boundary_completed','branches':results,
            'boundaries':boundaries,'ledger':data,'new_auxiliary_sampling_steps':0,
            'auxiliary_training_scope':'frozen_original_train_pairs_only','automatic_teacher_refresh':False,
            'qualified':False,'explanation_qualified':False,'release_ready':False}
        path = output/f'completion_{target:07d}.json'
        if path.exists(): _same(_json(path),result,'Completed result changed; no rerun permitted')
        else: write_json(path,result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(); modes.add_argument('--prepare',action='store_true'); modes.add_argument('--run',action='store_true')
    parser.add_argument('--inputs'); parser.add_argument('--inputs-sha256'); parser.add_argument('--output'); parser.add_argument('--device',default='mps',choices=('mps','cpu'))
    parser.add_argument('--until',type=int)
    args=parser.parse_args(argv)
    if not args.prepare and not args.run:
        print(json.dumps({'version':VERSION,'plan_only':True,'schedule':_schedule(False),
            'ppo_steps':2*PPO_CAP,'validation_step_cap':72000,'auxiliary_sampling_steps':0,
            'initial_lambda':0,'first_feedback_gate_step':0,'ramp_clock':'inherited_cumulative_steps','automatic_teacher_refresh':False,
            'qualification_granted':False},ensure_ascii=False)); return
    if not args.output: parser.error('--output is required')
    if args.prepare:
        if not args.inputs or not args.inputs_sha256: parser.error('--inputs and --inputs-sha256 are required')
        value=prepare(args.output,args.inputs,expected_inputs_sha256=args.inputs_sha256,device=args.device)
    else: value=advance(args.output,until=args.until)
    print(json.dumps(value,ensure_ascii=False))


if __name__ == '__main__': main()
