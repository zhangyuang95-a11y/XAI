"""Finite unchanged-beta1 native cycles derived from completed development runs.

Each new cycle has an explicit PPO cap, a probe included in that cap, and one
fixed final validation. Genuine energy/native-cycle checkpoint loaders preserve
both Adam states, environments and RNG; no old payload is relabeled. Source gate
failure is retained as development evidence, never turned into qualification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

from . import warehouse_family_energy_continuation_run as energy
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save
from .warehouse_native_cycle_budget import CycleBudget
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_partner_mix_run import AUTHORIZATION, AUTHORIZATION_SHA, decode, save_batch, write_bytes, write_json
from .warehouse_native_shutdown_result import _Inputs, _json_bytes
from . import warehouse_native_shutdown_feedback_run as paired
from . import warehouse_native_shutdown_feedback_trainer as feedback
from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_native_shutdown_continuation_run as stage

VERSION = 'warehouse-family-native-cycle-run.v1'
BRANCH = 'beta1'
PROBE = 4096
MAX_PPO_CAP = 1000000
EVALUATION_CAP = 18000
SOURCE_RELATIONSHIP = 'derived_single_arm_not_original_pair'
_same, _sha, _json = energy._same, energy._sha, energy._json
_authorization, _trainer = energy._authorization, energy._trainer
_learning_checks, _ready, _actor_arrays = energy._learning_checks, energy._ready, energy._actor_arrays


def sources():
    result = energy.sources()
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _limits(cap, fixture, *, probe=None, environments=None):
    if type(fixture) is not bool or type(cap) is not int or not (1 if fixture else PROBE) <= cap <= MAX_PPO_CAP:
        raise ValueError('A finite integer PPO cap is required (production: 4096..1000000)')
    expected_probe = min(PROBE, cap) if fixture else PROBE
    if probe is None:
        probe = expected_probe
    if type(probe) is not int or not 0 < probe <= cap or (not fixture and probe != PROBE):
        raise ValueError('Invalid registered probe endpoint')
    if environments is not None and (type(environments) is not int or environments < 1
            or cap % environments or probe % environments):
        raise ValueError('Cap and probe require complete joint environment batches')
    return cap, probe


def source_records(source_run, expected_completion_sha256, *, fixture=False, _ancestors=()):
    """Verify closed source records before any checkpoint decode/model construction."""
    _sha(expected_completion_sha256)
    if type(fixture) is not bool:
        raise ValueError('Explicit fixture scope required')
    root = Path(source_run).expanduser().resolve()
    if str(root) in _ancestors:
        raise ValueError('Cyclic native continuation ancestry')
    raw = _json(root/'prepared.json')
    if raw.get('version') == energy.VERSION:
        saved = energy.read_completed(root, expected_completion_sha256=expected_completion_sha256, allow_test_fixture=fixture)
        kind = 'energy'
        original = deepcopy(saved['source_descriptor'])
    elif raw.get('version') == VERSION:
        saved = read_completed(root, expected_completion_sha256=expected_completion_sha256,
            allow_test_fixture=fixture, _ancestors=_ancestors)
        kind = 'native_cycle'
        original = deepcopy(saved['original_feedback_descriptor'])
    else:
        raise ValueError('Only a completed energy or genuine native-cycle runner is supported')
    p, completed, protocol = saved['prepared'], saved['completion'], saved['protocol']
    cap = p['primary_endpoint']
    if (type(cap) is not int or completed['until'] != cap or completed['counts']['ppo_steps'] != cap
            or p['test_fixture'] is not fixture or p['shutdown_arm'] != BRANCH or p['branch'] != 'own_credit'
            or p['own_shutdown_beta'] != 1. or p['feedback_disabled_after_source'] is not True
            or p['feedback_lambda'] != 0. or protocol['version'] != native.PROTOCOL_VERSION
            or protocol['evaluation']['checkpoints_ppo_steps'] != [cap]
            or any(completed.get(k) is not False for k in ('used_final_test','formal_ready','release_ready','explanation_qualified'))):
        raise ValueError('Source is not its completed fixed beta1 pure-PPO endpoint')
    binding = saved['actor_bindings']; receipt = saved['actor_parameter_receipt']
    if (binding['joint_steps'] != cap or receipt['checkpoint_sha256'] != saved['checkpoint']['sha256']
            or receipt['actor_sha256'] != binding['actor_sha256'] or receipt['all_six_arrays_equal'] is not True):
        raise ValueError('Completed source Actor/checkpoint/tensor receipt differs')
    actor = Path(saved['actor_path']).resolve()
    if not actor.is_relative_to(root):
        raise ValueError('Source Actor must belong to its own completed run')
    descriptor = {'run': str(root), 'run_version': p['version'], 'source_kind': kind,
        'source_step': cap, 'source_branch': 'own_credit', 'shutdown_arm': BRANCH, 'own_shutdown_beta': 1.,
        'completion_sha256': expected_completion_sha256, 'prepared_sha256': file_hash(root/'prepared.json'),
        'protocol_sha256': digest(protocol), 'scenario_manifest_sha256': digest(saved['scenarios']),
        'checkpoint': deepcopy(saved['checkpoint']), 'actor_path': str(actor.relative_to(root)),
        'actor_sha256': binding['actor_sha256'], 'actor_parameters_sha256': receipt['actor_parameters_sha256'],
        'validation_report_sha256': completed['report_sha256'],
        'original_capability': {'capability': deepcopy(saved['report']['capability']),
            'warmup_capability': deepcopy(saved['report']['warmup_capability'])},
        'input_bindings_sha256': digest(saved['input_bindings']),
        'original_feedback_descriptor': original, 'test_fixture': fixture}
    return descriptor, saved


def load_source(source_run, expected_completion_sha256, *, device, fixture=False, expected=None, _ancestors=()):
    _authorization(); execution = sources()
    root = Path(source_run).expanduser().resolve()
    descriptor, saved = source_records(root, expected_completion_sha256, fixture=fixture, _ancestors=_ancestors)
    if expected is not None:
        _same(descriptor, expected, 'Bound native-cycle source changed')
    if saved['prepared']['device'] != device:
        raise ValueError('Inherited source device cannot change')
    producer = energy if descriptor['source_kind'] == 'energy' else sys.modules[__name__]
    args = {'allow_test_fixture': fixture}
    if producer is not energy:
        args['_ancestors'] = _ancestors
    p, protocol, scenes, ancestor = producer.read_prepared(root, **args)
    _same(p, saved['prepared'], 'Source preparation changed during genuine restoration')
    _same(protocol, saved['protocol'], 'Source protocol changed during genuine restoration')
    _same(scenes, saved['scenarios'], 'Source scenes changed during genuine restoration')
    source = producer._trainer(p, protocol, ancestor)
    head = saved['checkpoint']; payload = decode(paired._check(root, head), head['sha256'])
    producer._envelope(payload, source)
    account = saved['completion']['ledger']; op = account['operations'].get(payload.get('operation_id'))
    if (not op or op['status'] != 'acknowledged' or op['completion']['checkpoint'] != head
            or payload.get('audit', {}).get('neural_overrides') != 0
            or payload.get('actual_steps') != op['completion']['actual_steps']):
        raise ValueError('Completed source checkpoint lacks its actual PPO acknowledgment')
    for item in payload['evidence'].values():
        paired._check(root, item)
    source.load_state_dict(payload['trainer'])
    if (type(source) is not native.ShutdownContinuationTrainer or source.joint_steps != descriptor['source_step']
            or source.feedback_enabled or source.beta != 1. or source.shutdown_arm != BRANCH):
        raise ValueError('Restored source is not the genuine fixed native beta1 learner')
    _same(_counts(source), saved['completion']['counts'], 'Actual source optimizer counters differ from completed records')
    if file_hash(root/descriptor['actor_path']) != descriptor['actor_sha256']:
        raise ValueError('Original source Actor bytes changed during restoration')
    arrays, parameter_sha, _ = _actor_arrays(root/descriptor['actor_path'])
    actual = source.model.actor.state_dict()
    if (set(actual) != set(arrays) or any(not np.array_equal(actual[k].detach().cpu().numpy(), v) for k,v in arrays.items())
            or parameter_sha != descriptor['actor_parameters_sha256']
            or feedback.actor_parameter_sha256(actual) != parameter_sha):
        raise ValueError('Actual source checkpoint six tensors differ from its original NPZ receipt')
    if sources() != execution:
        raise ValueError('Execution source changed during restoration')
    return source, descriptor, scenes



def prepare(output, *, source_run, expected_completion_sha256, ppo_cap, cycle_id=None, device='mps',
            allow_test_fixture=False, fixture_probe_endpoint=None):
    _sha(expected_completion_sha256); _authorization()
    if fixture_probe_endpoint is not None and not allow_test_fixture:
        raise ValueError('Fixture probe override cannot be used in production')
    cap, probe = _limits(ppo_cap, allow_test_fixture, probe=fixture_probe_endpoint)
    output = Path(output).expanduser().resolve()
    source_run = Path(source_run).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a new independent finite diagnostic directory')
    paired._separate(output, source_run)
    if type(allow_test_fixture) is not bool or device not in ('cpu', 'mps'):
        raise ValueError('Explicit device and fixture boundary required')
    source, descriptor, scenes = load_source(source_run, expected_completion_sha256, device=device, fixture=allow_test_fixture)
    if not allow_test_fixture and (len(scenes['splits']['validation']) != 50 or scenes['configuration']['horizon'] != 120):
        raise ValueError('The fixed 150-episode validation matrix differs')
    _limits(cap, allow_test_fixture, probe=probe, environments=len(source.envs))
    before = source.state_dict(); state_sha = initialization_sha256(before)
    cycle_id = cycle_id or output.name
    protocol = native.make_protocol(source, source_checkpoint_sha256=descriptor['checkpoint']['sha256'],
        source_state_sha256=state_sha, ppo_cap=cap, cycle_id=cycle_id,
        evaluation_checkpoints=[cap], test_fixture=allow_test_fixture)
    p = {'version': VERSION, 'cycle_id': cycle_id, 'device': device, 'branch': 'own_credit',
        'shutdown_arm': BRANCH, 'own_shutdown_beta': 1., 'source_descriptor': descriptor,
        'source_checkpoint_sha256': descriptor['checkpoint']['sha256'], 'source_state_sha256': state_sha,
        'protocol_sha256': digest(protocol), 'scenario_manifest_sha256': digest(scenes),
        'baselines_sha256': file_hash(source_run/'baselines.json'), 'runtime_sources': sources(),
        'authorization_record_sha256': AUTHORIZATION_SHA, 'budget_caps': {BRANCH: {'ppo': cap, 'evaluation': EVALUATION_CAP}},
        'primary_endpoint': cap, 'probe_endpoint': probe, 'validation_endpoints': [cap],
        'feedback_disabled_after_source': True, 'feedback_lambda': 0., 'stop_gate': 'full_and_warmup',
        'source_relationship': SOURCE_RELATIONSHIP, 'source_kind': descriptor['source_kind'],
        'original_feedback_descriptor': deepcopy(descriptor['original_feedback_descriptor']),
        'test_fixture': allow_test_fixture, 'formal_ready': False, 'created_unix': time.time()}
    p['identity'] = {k: deepcopy(v) for k, v in p.items() if k not in ('created_unix', 'formal_ready')}
    learner = _trainer(p, protocol, source); saved = learner.state_dict()
    matched = _learning_checks(before, saved, device)
    if learner.joint_steps != 0 or learner.feedback_enabled or sources() != p['runtime_sources']:
        raise ValueError('New diagnostic must start at zero with unchanged sources and no tree KL')
    if cycle_id == source.protocol.get('cycle_id'):
        raise ValueError('A new cycle requires a distinct cycle_id')
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('prepared', p), ('protocol', protocol), ('scenarios', scenes)):
        write_json(output/(name+'.json'), value)
    write_bytes(output/'baselines.json', (source_run/'baselines.json').read_bytes())
    write_bytes(output/'authorization.json', AUTHORIZATION.read_bytes())
    for name in p['runtime_sources']:
        write_bytes(output/'source_snapshot'/name, (ROOT/name).read_bytes())
    ledger = CycleBudget.create(output, p['identity'])
    with ledger.lease():
        checkpoint = output/'branches'/BRANCH/'checkpoints/initial.pt'
        atomic_torch_save(checkpoint, _payload(learner, operation_id=None))
        ledger.initialize_head('ppo', BRANCH, str(checkpoint.relative_to(output)), file_hash(checkpoint))
        reference = output/'branches'/BRANCH/'initial_evaluation_reference.json'
        write_json(reference, {'source_descriptor': descriptor, 'new_environment_steps': 0})
        ledger.initialize_head('evaluation', BRANCH, str(reference.relative_to(output)), file_hash(reference))
    write_json(output/'initialization_check.json', {'identical_learning_state': True, 'matched': matched,
        'source_state_sha256': state_sha, 'feedback_disabled_after_source': True, 'new_environment_steps': 0})
    return {'status': 'prepared', 'output': str(output), 'cycle_id': cycle_id, 'ppo_cap': cap,
        'evaluation_cap': EVALUATION_CAP, 'probe_endpoint': probe, 'feedback_disabled_after_source': True, 'formal_ready': False}


def _material(output, *, allow_test_fixture=False):
    output = Path(output).expanduser().resolve(); p = _json(output/'prepared.json')
    expected_identity = {k: v for k, v in p.items() if k not in ('identity', 'created_unix', 'formal_ready')}
    if (p.get('version') != VERSION or p.get('test_fixture') is not allow_test_fixture
            or p.get('runtime_sources') != sources() or p.get('shutdown_arm') != BRANCH
            or p.get('feedback_disabled_after_source') is not True or p.get('feedback_lambda') != 0.
            or p.get('stop_gate') != 'full_and_warmup' or p.get('branch') != 'own_credit'
            or p.get('own_shutdown_beta') != 1.):
        raise ValueError('Prepared native cycle identity changed')
    cap, probe = _limits(p.get('primary_endpoint'), allow_test_fixture, probe=p.get('probe_endpoint'))
    _same(p['identity'], expected_identity, 'Prepared identity changed')
    _same(p['budget_caps'], {BRANCH: {'ppo': cap, 'evaluation': EVALUATION_CAP}}, 'Fixed finite caps changed')
    if (p['primary_endpoint'], p['probe_endpoint'], p['validation_endpoints']) != (cap, probe, [cap]):
        raise ValueError('Registered diagnostic endpoints changed')
    _authorization()
    if p['authorization_record_sha256'] != AUTHORIZATION_SHA:
        raise ValueError('Authorization identity changed')
    for name, key in (('authorization.json', 'authorization_record_sha256'), ('baselines.json', 'baselines_sha256')):
        if file_hash(output/name) != p[key]:
            raise ValueError('Bound diagnostic input changed: ' + name)
    for name, sha in p['runtime_sources'].items():
        if file_hash(output/'source_snapshot'/name) != sha:
            raise ValueError('Archived execution source changed')
    protocol, scenes = _json(output/'protocol.json'), _json(output/'scenarios.json')
    _same(digest(protocol), p['protocol_sha256'], 'Protocol changed')
    _same(digest(scenes), p['scenario_manifest_sha256'], 'Scenarios changed')
    if (protocol.get('version') != native.PROTOCOL_VERSION or protocol.get('cycle_id') != p['cycle_id']
            or protocol.get('feedback_training_enabled') is not False
            or protocol.get('own_shutdown_beta') != 1. or protocol.get('shutdown_arm') != BRANCH
            or protocol.get('budget', {}).get('maximum_ppo_joint_steps') != cap
            or protocol.get('budget', {}).get('maximum_ppo_joint_steps_per_arm') != cap
            or protocol.get('evaluation', {}).get('checkpoints_ppo_steps') != [cap]
            or protocol.get('evaluation', {}).get('maximum_environment_steps') != EVALUATION_CAP
            or protocol.get('evaluation', {}).get('read_final_test') is not False
            or protocol.get('source', {}).get('checkpoint_sha256') != p['source_checkpoint_sha256']
            or protocol.get('source', {}).get('state_sha256') != p['source_state_sha256']):
        raise ValueError('Native protocol differs from this finite beta1 diagnostic')
    _limits(cap, allow_test_fixture, probe=probe, environments=protocol['training']['environments'])
    if (p.get('source_relationship') != SOURCE_RELATIONSHIP
            or p.get('source_kind') != p['source_descriptor']['source_kind']
            or p.get('original_feedback_descriptor') != p['source_descriptor']['original_feedback_descriptor']):
        raise ValueError('Original pair and direct native source identities differ')
    initial = _json(output/'initialization_check.json')
    if (initial.get('identical_learning_state') is not True or initial.get('new_environment_steps') != 0
            or initial.get('source_state_sha256') != p['source_state_sha256']
            or initial.get('feedback_disabled_after_source') is not True):
        raise ValueError('Missing complete-state inheritance check')
    return p, protocol, scenes


def read_prepared(output, *, allow_test_fixture=False, _ancestors=()):
    root = Path(output).expanduser().resolve()
    if str(root) in _ancestors:
        raise ValueError('Cyclic native continuation ancestry')
    p, protocol, scenes = _material(root, allow_test_fixture=allow_test_fixture)
    d = p['source_descriptor']
    source, actual, source_scenes = load_source(d['run'], d['completion_sha256'],
        device=p['device'], fixture=allow_test_fixture, expected=d, _ancestors=(*_ancestors, str(root)))
    _same(scenes, source_scenes, 'Cycle scenarios differ from its genuine source')
    _same(initialization_sha256(source.state_dict()), p['source_state_sha256'], 'Full native source state changed')
    expected = native.make_protocol(source, source_checkpoint_sha256=p['source_checkpoint_sha256'],
        source_state_sha256=p['source_state_sha256'], ppo_cap=p['primary_endpoint'], cycle_id=p['cycle_id'],
        evaluation_checkpoints=[p['primary_endpoint']], test_fixture=allow_test_fixture)
    _same(protocol, expected, 'Genuine native protocol differs from the finite cycle')
    return p, protocol, scenes, source


def _payload(trainer, *, operation_id, **fields):
    return {'version': VERSION, 'cycle_id': trainer.protocol['cycle_id'], 'branch': 'own_credit',
        'shutdown_arm': BRANCH, 'feedback_disabled_after_source': True, 'feedback_lambda': 0.,
        'operation_id': operation_id, 'trainer': trainer.state_dict(), **fields}


def _envelope(payload, trainer):
    if (payload.get('version') != VERSION or payload.get('branch') != 'own_credit'
            or payload.get('shutdown_arm') != BRANCH or payload.get('cycle_id') != trainer.protocol['cycle_id']
            or payload.get('feedback_disabled_after_source') is not True or payload.get('feedback_lambda') != 0.
            or payload.get('trainer', {}).get('version') != native.VERSION):
        raise ValueError('Checkpoint is not the genuine new single-arm diagnostic')


def _counts(trainer):
    return {'ppo_steps': trainer.joint_steps, 'optimizer_updates': trainer.optimizer_updates,
        'actor_optimizer_steps': trainer.actor_optimizer_steps, 'critic_optimizer_steps': trainer.critic_optimizer_steps}


def train_to(output, trainer, ledger, target):
    p = _json(output/'prepared.json')
    cap, probe = _limits(p['primary_endpoint'], p['test_fixture'], probe=p['probe_endpoint'], environments=trainer.cfg['environments'])
    if target not in (probe, cap) or trainer.shutdown_arm != BRANCH or trainer.feedback_enabled:
        raise ValueError('Only fixed beta1 pure PPO diagnostic boundaries are allowed')
    n = trainer.cfg['environments']
    while trainer.joint_steps < target:
        if any(op['status'] in ('pending', 'abandoned') for op in ledger.read()['operations'].values()):
            raise ValueError('Unconfirmed sampling requires diagnosis; no resampling or refund')
        head = ledger.head('ppo', BRANCH)
        if trainer.joint_steps != head['step']:
            raise ValueError('Actual learner differs from confirmed head')
        old = decode(output/head['path'], head['sha256']); _envelope(old, trainer)
        before = initialization_sha256(trainer.state_dict())
        _same(initialization_sha256(old['trainer']), before, 'In-memory state differs from its confirmed checkpoint')
        amount = min(n*trainer.cfg['rollout_steps'], target-trainer.joint_steps, ledger.remaining('ppo', BRANCH))
        amount -= amount % n
        if amount <= 0:
            raise ValueError('Finite reservation cannot reach the registered endpoint')
        operation_id = 'ppo_'+uuid.uuid4().hex
        reservation = ledger.reserve('ppo', BRANCH, amount, operation_id, expected_step=trainer.joint_steps)
        if not reservation['execution_permitted']:
            raise ValueError('A repeated reservation cannot execute again')
        before_counts = _counts(trainer)
        batch, metrics = trainer.train_chunk(amount//n)
        if (trainer.joint_steps != head['step']+amount or len(batch['transition_records']) != amount
                or batch['audit'].get('neural_overrides') != 0 or any(not np.isfinite(x) for x in metrics.values())
                or any(row.get('shutdown_arm') != BRANCH or row.get('own_shutdown_beta') != 1.
                    for row in batch['transition_records'])
                or metrics.get('feedback_lambda', 0.) != 0.):
            raise ValueError('Actual beta1 PPO step/count/NN contract differs')
        evidence = save_batch(output, BRANCH, operation_id, batch)
        episodes = deepcopy(trainer.completed_episodes); trainer.completed_episodes.clear()
        checkpoint = output/'branches'/BRANCH/'checkpoints'/f'step_{trainer.joint_steps:07d}_{operation_id}.pt'
        payload = _payload(trainer, operation_id=operation_id, audit=batch['audit'], metrics=metrics,
            evidence=evidence, actual_steps=amount, completed_episodes=episodes, before_state_sha256=before)
        atomic_torch_save(checkpoint, payload)
        write_json(checkpoint.with_suffix('.json'), {'version': VERSION, 'operation_id': operation_id,
            'cycle_id': trainer.protocol['cycle_id'], 'shutdown_arm': BRANCH,
            'before_counts': before_counts, 'after_counts': _counts(trainer),
            'predecessor_head': head, 'checkpoint': {'path': str(checkpoint.relative_to(output)),
                'sha256': file_hash(checkpoint), 'step': trainer.joint_steps},
            'actual_steps': amount, 'audit': batch['audit'], 'metrics': metrics, 'evidence': evidence,
            'feedback_disabled_after_source': True, 'feedback_lambda': 0.})
        ledger.ack(operation_id, str(checkpoint.relative_to(output)), file_hash(checkpoint), amount)
        write_json(output/'progress.json', {'status': 'training', 'steps': trainer.joint_steps,
            'target': target, 'ledger': ledger.read(), 'feedback_disabled_after_source': True}, replace=True)
        print(json.dumps({'event': 'ppo_ack', 'steps': trainer.joint_steps, 'neural_overrides': 0, 'feedback_lambda': 0.}), flush=True)


def _actor_receipt(output, learner, checkpoint, report):
    cap = learner.protocol['budget']['maximum_ppo_joint_steps']
    actor = output/'branches'/BRANCH/'actors'/f'actor_{cap:07d}.npz'
    arrays, parameter_sha, fields = _actor_arrays(actor)
    actual = learner.model.actor.state_dict()
    if set(actual) != set(arrays) or any(not np.array_equal(actual[key].detach().cpu().numpy(), value) for key, value in arrays.items()):
        raise ValueError('Actual terminal learner Actor differs from its frozen NPZ')
    if feedback.actor_parameter_sha256(actual) != parameter_sha:
        raise ValueError('Terminal Actor tensor encoding differs from the static NPZ encoding')
    if file_hash(actor) != report['actor_bindings']['actor_sha256']:
        raise ValueError('Actual terminal Actor differs from its validation report')
    value = {'version': VERSION, 'cycle_id': learner.protocol['cycle_id'], 'joint_steps': cap,
        'checkpoint_sha256': checkpoint['sha256'], 'checkpoint_path': checkpoint['path'],
        'actor_sha256': file_hash(actor), 'actor_parameters_sha256': parameter_sha,
        'all_six_arrays_equal': True, 'field_names': sorted(arrays), 'fields': fields,
        'new_neural_forward_calls': 0, 'additional_checkpoint_decodes': 0,
        'scope': 'Actual in-memory confirmed learner vs exact exported NPZ; no new forward or PT decode'}
    path = output/'actor_tensor_receipt.json'
    if path.exists():
        _same(_json(path), value, 'Original terminal tensor receipt changed')
    else:
        write_json(path, value)
    return {'path': path.name, 'sha256': file_hash(path), 'size': path.stat().st_size}


def read_completed(output, *, expected_completion_sha256, allow_test_fixture=False, _ancestors=()):
    """Read the externally anchored fixed endpoint, with no PT decode or sampling.

    A genuine static NumPy Actor is loaded by the saved-report reader. This is
    record verification, not new performance evaluation or model qualification.
    """
    _sha(expected_completion_sha256)
    output = Path(output).expanduser().resolve(); reads = _Inputs()
    if str(output) in _ancestors:
        raise ValueError('Cyclic native continuation ancestry')
    p, protocol, scenes = _material(output, allow_test_fixture=allow_test_fixture)
    cap = p['primary_endpoint']
    completed = reads.json(output, f'completion_{cap:07d}.json', expected_completion_sha256)
    for name in ('prepared.json', 'protocol.json', 'scenarios.json', 'baselines.json', 'authorization.json', 'initialization_check.json'):
        reads.raw(output, name)
    for name, sha in p['runtime_sources'].items():
        reads.raw(output, 'source_snapshot/'+name, sha)
    descriptor = p['source_descriptor']
    actual, _ = source_records(descriptor['run'], descriptor['completion_sha256'], fixture=allow_test_fixture,
        _ancestors=(*_ancestors, str(output)))
    _same(actual, descriptor, 'Original complete feedback provenance changed')
    if (completed.get('version') != VERSION or completed.get('until') != cap
            or completed.get('cycle_id') != p['cycle_id'] or completed.get('feedback_disabled_after_source') is not True
            or completed.get('feedback_lambda') != 0. or completed.get('used_final_test') is not False
            or any(completed.get(key) is not False for key in ('formal_ready', 'release_ready', 'explanation_qualified'))):
        raise ValueError('Not a completed fixed diagnostic endpoint')
    _same(completed['original_feedback_descriptor'], p['original_feedback_descriptor'], 'Original feedback ancestry changed')
    if completed.get('source_relationship') != SOURCE_RELATIONSHIP:
        raise ValueError('Cycle cannot extend or replace the original paired experiment')
    _same(completed['source_descriptor'], descriptor, 'Completed source descriptor differs')
    account = CycleBudget(output, p['identity']).read()
    _same(completed['ledger'], account, 'Completed ledger differs from acknowledged finite accounting')
    if any(op['status'] != 'acknowledged' for op in account['operations'].values()):
        raise ValueError('The diagnostic contains unconfirmed sampling')
    ppo = sorted((op for op in account['operations'].values() if op['request']['kind'] == 'ppo'), key=lambda op: op['sequence'])
    cell = account['branches'][BRANCH]['ppo']
    if cell['reserved'] != cap or cell['acknowledged'] != cap or completed['checkpoint'] != cell['head']:
        raise ValueError('The diagnostic does not have exactly its registered cap of confirmed PPO steps')
    expected_counts = {'ppo_steps': 0, 'optimizer_updates': 0, 'actor_optimizer_steps': 0, 'critic_optimizer_steps': 0}
    cfg = protocol['training']
    bindings = completed.get('training_records', {})
    if set(bindings) != {operation['opid'] for operation in ppo}:
        raise ValueError('Completed training sidecar inventory differs')
    for operation in ppo:
        amount = operation['completion']['actual_steps']
        if amount != operation['request']['steps'] or operation['request']['branch'] != BRANCH:
            raise ValueError('Recorded PPO amount or branch differs')
        item = operation['completion']['checkpoint']
        reads.raw(output, item['path'], item['sha256'])
        sidecar = bindings[operation['opid']]
        if sidecar['path'] != str(Path(item['path']).with_suffix('.json')):
            raise ValueError('Training sidecar belongs to another checkpoint')
        record = _json_bytes(reads.bound(output, sidecar))
        if (record.get('version') != VERSION or record.get('cycle_id') != p['cycle_id']
                or record.get('operation_id') != operation['opid'] or record.get('checkpoint') != item
                or record.get('predecessor_head') != operation['old_head'] or record.get('actual_steps') != amount
                or record.get('shutdown_arm') != BRANCH or record.get('feedback_disabled_after_source') is not True
                or record.get('feedback_lambda') != 0. or record.get('audit', {}).get('neural_overrides') != 0):
            raise ValueError('Original committed PPO record differs')
        _same(record['before_counts'], expected_counts, 'Committed training counter chain differs')
        after = record['after_counts']; minibatches = cfg['epochs']*((2*amount+cfg['minibatch']-1)//cfg['minibatch'])
        if (after['ppo_steps'] != expected_counts['ppo_steps']+amount
                or after['optimizer_updates'] != expected_counts['optimizer_updates']+1
                or after['critic_optimizer_steps'] != expected_counts['critic_optimizer_steps']+minibatches
                or not 0 <= after['actor_optimizer_steps']-expected_counts['actor_optimizer_steps'] <= minibatches):
            raise ValueError('Recorded actual optimizer counters differ from the fixed PPO batches')
        for evidence in record['evidence'].values():
            reads.raw(output, evidence['path'], evidence['sha256'], evidence.get('size'))
        expected_counts = deepcopy(after)
    _same(completed['counts'], expected_counts, 'Recorded PPO/update counts differ from fixed batches')
    report_path = output/'branches'/BRANCH/'validation'/f'step_{cap:07d}'/'report.json'
    reads.json(report_path.parent, report_path.name, completed['report_sha256'])
    manifest = reads.json(report_path.parent, 'manifest.json', _sha(completed.get('validation_manifest_sha256')))
    for episode in manifest['episodes']:
        for kind in ('row', 'trace'):
            reads.bound(report_path.parent, episode[kind])
    report = stage._report_record(output, BRANCH, cap, p, protocol, scenes, account, allow_test_fixture)
    receipt_binding = completed.get('actor_tensor_receipt', {})
    if receipt_binding.get('path') != 'actor_tensor_receipt.json':
        raise ValueError('Missing original current-checkpoint tensor receipt')
    tensor_receipt = _json_bytes(reads.bound(output, receipt_binding))
    actor_path = report_path.parent.parent.parent/'actors'/f'actor_{cap:07d}.npz'
    reads.raw(actor_path.parent, actor_path.name, report['actor_bindings']['actor_sha256'])
    reads.raw(actor_path.parent, actor_path.with_suffix('.json').name)
    arrays, parameter_sha, fields = _actor_arrays(actor_path)
    if (tensor_receipt.get('version') != VERSION or tensor_receipt.get('cycle_id') != p['cycle_id']
            or tensor_receipt.get('joint_steps') != cap or tensor_receipt.get('all_six_arrays_equal') is not True
            or tensor_receipt.get('checkpoint_sha256') != completed['checkpoint']['sha256']
            or tensor_receipt.get('checkpoint_path') != completed['checkpoint']['path']
            or tensor_receipt.get('actor_sha256') != report['actor_bindings']['actor_sha256']
            or tensor_receipt.get('actor_parameters_sha256') != parameter_sha
            or tensor_receipt.get('field_names') != sorted(arrays) or tensor_receipt.get('fields') != fields
            or tensor_receipt.get('new_neural_forward_calls') != 0 or tensor_receipt.get('additional_checkpoint_decodes') != 0):
        raise ValueError('Saved terminal tensor/NPZ/validation/checkpoint binding differs')
    _same(completed['capability'], report['capability'], 'Original capability report differs')
    _same(completed['warmup_capability'], report['warmup_capability'], 'Original warmup report differs')
    expected_status = 'both_gates_ready' if _ready(report) else 'fixed_endpoint_completed'
    if completed.get('status') != expected_status:
        raise ValueError('Completed status contradicts the unchanged two-gate rule')
    if (account['branches'][BRANCH]['evaluation']['reserved'] != EVALUATION_CAP
            or account['branches'][BRANCH]['evaluation']['acknowledged'] != report['environment_steps']):
        if not allow_test_fixture:
            raise ValueError('The complete fixed validation allocation differs')
        if account['branches'][BRANCH]['evaluation']['acknowledged'] != report['environment_steps']:
            raise ValueError('Fixture validation acknowledgments differ')
    reads.unchanged()
    return {'version': VERSION, 'completion_sha256': expected_completion_sha256, 'completion': completed,
        'prepared': p, 'protocol': protocol, 'scenarios': scenes, 'report': report,
        'actor_path': str(actor_path), 'actor_bindings': report['actor_bindings'], 'actor_parameter_receipt': tensor_receipt,
        'checkpoint': deepcopy(completed['checkpoint']), 'both_gates_ready': _ready(report),
        'input_bindings': reads.records,
        'source_descriptor': descriptor, 'original_feedback_descriptor': deepcopy(p['original_feedback_descriptor']),
        'source_kind': 'native_cycle', 'ppo_cap': cap, 'inspection_scope': 'saved records; static NumPy Actor load; no forward, PT decode or physics',
        'formal_ready': False, 'release_ready': False, 'explanation_qualified': False}


def advance(output, *, until=None, allow_test_fixture=False):
    output = Path(output).expanduser().resolve(); raw = _json(output/'prepared.json')
    cap, probe = _limits(raw['primary_endpoint'], allow_test_fixture, probe=raw['probe_endpoint'])
    until = cap if until is None else until
    if type(until) is not int or until not in (probe, cap):
        raise ValueError('Use only the registered probe or fixed endpoint of this cycle')
    if raw.get('version') != VERSION or raw.get('test_fixture') is not allow_test_fixture:
        raise ValueError('Diagnostic version or fixture boundary differs')
    ledger = CycleBudget(output, raw['identity'])
    with ledger.lease():
        account = ledger.read()
        if any(op['status'] in ('pending', 'abandoned') for op in account['operations'].values()):
            raise ValueError('Unconfirmed sampling requires diagnosis; never resample')
        p, protocol, scenes, source = read_prepared(output, allow_test_fixture=allow_test_fixture)
        learner = _trainer(p, protocol, source)
        head = ledger.head('ppo', BRANCH)
        payload = decode(output/head['path'], head['sha256']); _envelope(payload, learner)
        if head['step']:
            op = account['operations'].get(payload.get('operation_id'))
            if (not op or op['status'] != 'acknowledged' or op['completion']['checkpoint'] != head
                    or payload.get('audit', {}).get('neural_overrides') != 0):
                raise ValueError('Checkpoint has no original completed PPO acknowledgment')
            for binding in payload['evidence'].values():
                paired._check(output, binding)
        learner.load_state_dict(payload['trainer'])
        if learner.joint_steps != head['step'] or learner.joint_steps > until:
            raise ValueError('Restored progress differs from confirmed budget or requested boundary')
        report = None
        for target in sorted(set((probe, cap))):
            if target > until or target < learner.joint_steps:
                continue
            train_to(output, learner, ledger, target)
            if target == cap:
                folder = output/'branches'/BRANCH/'validation'/f'step_{cap:07d}'
                if not (folder/'report.json').exists():
                    stage.evaluate_boundary(output, learner, ledger, BRANCH, scenes)
                report = stage._report_record(output, BRANCH, cap, p, protocol, scenes, ledger.read(), allow_test_fixture)
        ready = report is not None and _ready(report)
        tensor_receipt = _actor_receipt(output, learner, ledger.head('ppo', BRANCH), report) if report else None
        training_records = {}
        for opid, operation in ledger.read()['operations'].items():
            if operation['request']['kind'] == 'ppo':
                sidecar = Path(operation['completion']['checkpoint']['path']).with_suffix('.json')
                training_records[opid] = {'path': str(sidecar), 'sha256': file_hash(output/sidecar), 'size': (output/sidecar).stat().st_size}
        result = {'version': VERSION, 'status': 'both_gates_ready' if ready else ('fixed_endpoint_completed' if learner.joint_steps == cap else 'probe_completed'),
            'until': learner.joint_steps, 'cycle_id': p['cycle_id'], 'checkpoint': ledger.head('ppo', BRANCH),
            'counts': _counts(learner), 'training_records': training_records, 'actor_tensor_receipt': tensor_receipt,
            'ledger': ledger.read(), 'feedback_disabled_after_source': True, 'feedback_lambda': 0.,
            'source_descriptor': p['source_descriptor'], 'original_pair_preserved': True,
            'original_feedback_descriptor': p['original_feedback_descriptor'], 'source_relationship': SOURCE_RELATIONSHIP,
            'capability': report['capability'] if report else None, 'warmup_capability': report['warmup_capability'] if report else None,
            'report_sha256': file_hash(output/'branches'/BRANCH/'validation'/f'step_{cap:07d}'/'report.json') if report else None,
            'validation_manifest_sha256': file_hash(output/'branches'/BRANCH/'validation'/f'step_{cap:07d}'/'manifest.json') if report else None,
            'used_final_test': False, 'formal_ready': False, 'release_ready': False, 'explanation_qualified': False}
        path = output/f'completion_{learner.joint_steps:07d}.json'
        if path.exists():
            _same(_json(path), result, 'Completed diagnostic record cannot change on resume')
        else:
            write_json(path, result)
        write_json(output/'progress.json', result, replace=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--source-run'); parser.add_argument('--expected-completion-sha256')
    parser.add_argument('--ppo-cap', type=int); parser.add_argument('--cycle-id')
    parser.add_argument('--device', choices=('cpu','mps'), default='mps'); parser.add_argument('--until', type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        if not args.source_run or not args.expected_completion_sha256 or args.ppo_cap is None:
            parser.error('Preparation requires the genuine completed source, external completion SHA and finite --ppo-cap')
        result = prepare(args.output, source_run=args.source_run, expected_completion_sha256=args.expected_completion_sha256,
            ppo_cap=args.ppo_cap, cycle_id=args.cycle_id, device=args.device)
    else:
        class Tee:
            def __init__(self, stream, log): self.stream, self.log = stream, log
            def write(self, value): self.stream.write(value); self.log.write(value); self.log.flush()
            def flush(self): self.stream.flush(); self.log.flush()
        with (Path(args.output)/'stdout.log').open('a', encoding='utf-8') as log:
            previous = sys.stdout; sys.stdout = Tee(previous, log)
            try: result = advance(args.output, until=args.until)
            finally: sys.stdout = previous
    print(json.dumps(result, ensure_ascii=False)); return result


if __name__ == '__main__':
    main()
