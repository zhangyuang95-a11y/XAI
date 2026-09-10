"""One 50k beta1 PPO diagnostic derived from the completed feedback endpoint.

The actual outer feedback checkpoint is restored by its own loader before its
genuine native learner is passed to ShutdownContinuationTrainer. The new cycle
retains learned weights, both Adam states, in-flight environments and RNG. Tree
KL is deliberately disabled after the source; this is not an extension of the
original paired experiment. No source record, trainer identity or gate changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save
from .warehouse_native_cycle_budget import CycleBudget
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_partner_mix_run import (
    AUTHORIZATION, AUTHORIZATION_SHA, decode, save_batch, write_bytes, write_json,
)
from .warehouse_native_shutdown_result import _Inputs, _json_bytes
from . import warehouse_native_shutdown_feedback_result as comparison
from . import warehouse_native_shutdown_feedback_run as paired
from . import warehouse_native_shutdown_feedback_trainer as feedback
from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_native_shutdown_continuation_run as stage
from backend import warehouse_family_explanation as family_explanation

VERSION = 'warehouse-family-energy-continuation-run.v1'
BRANCH = 'beta1'
SOURCE_STEP = 250000
PPO_CAP = 50000
PROBE = 4096
EVALUATION_CAP = 18000


def sources():
    result = stage.sources()
    result.update(paired.sources())
    for module in (comparison, feedback, native, family_explanation, sys.modules[__name__]):
        path = Path(module.__file__)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _same(a, b, reason):
    if digest(a) != digest(b):
        raise ValueError(reason)


def _sha(value):
    if type(value) is not str or re.fullmatch(r'[a-f0-9]{64}', value) is None:
        raise ValueError('An explicit external SHA256 is required')
    return value


def _json(path):
    return _json_bytes(Path(path).read_bytes())


def _authorization():
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError('Autonomous training authorization changed')


def _comparison_gate(result, completion_sha, fixture):
    if (result.get('version') != comparison.VERSION or result.get('status') != 'fixed_endpoint_complete'
            or result.get('test_fixture') is not fixture or result.get('primary_endpoint') != SOURCE_STEP
            or result.get('completion_sha256') != completion_sha
            or result.get('result_source_sha256') != file_hash(Path(comparison.__file__))
            or result.get('source_selection') is not None
            or any(result.get(k) is not False for k in ('formal_ready', 'release_ready', 'explanation_qualified'))):
        raise ValueError('A completed externally anchored fixed comparison is required')
    for arm in ('control', 'feedback'):
        record = result['training'][arm]
        if record['joint_steps'] != SOURCE_STEP or record['raw_neural_overrides'] != 0:
            raise ValueError('Original paired training or raw NN submission differs')
    if result['training']['control']['updates'] != result['training']['feedback']['updates']:
        raise ValueError('Original paired PPO update allocation differs')
    if result['training']['feedback'].get('positive_lambda_joint_steps', 0) <= 0:
        raise ValueError('The source lacks actual positive tree KL updates')
    audits = [x for x in result.get('checkpoint_audits', [])
        if x.get('kind') == 'final_effective_boundary' and x.get('branch') == 'feedback']
    if (len(audits) != 1 or audits[0].get('actor_export_equal') is not True
            or audits[0].get('dual_adam_steps_verified') is not True):
        raise ValueError('The actual completed checkpoint requires prior Actor/dual Adam inspection')
    positive = [x for x in result.get('checkpoint_audits', []) if x.get('kind') == 'first_positive_lambda_update']
    if (len(positive) != 1 or positive[0].get('branch') != 'feedback'
            or positive[0].get('nonzero_feedback_gradient_observed') is not True):
        raise ValueError('The source lacks its actual nonzero feedback-gradient evidence')
    # Development continuation is allowed when a capability gate failed. Its
    # original values are retained verbatim; no selection gate is bypassed.
    for key in ('full_capability', 'warmup_capability'):
        if type(result['primary']['feedback'][key].get('eligible')) is not bool:
            raise ValueError('Original development capability report is missing')
    return audits[0]


def source_records(source_run, comparison_path, expected_comparison_sha256,
                   expected_completion_sha256, *, fixture=False):
    """Pure saved-record/source check, before checkpoint decoding or model creation."""
    _sha(expected_comparison_sha256); _sha(expected_completion_sha256)
    if type(fixture) is not bool:
        raise ValueError('Explicit fixture scope required')
    root, report_path = Path(source_run).expanduser().resolve(), Path(comparison_path).expanduser().resolve()
    read = _Inputs()
    result = read.json(report_path.parent, report_path.name, expected_comparison_sha256)
    audited = _comparison_gate(result, expected_completion_sha256, fixture)
    completed = read.json(root, 'completion_0250000.json', expected_completion_sha256)
    p, protocol, scenes, _ = comparison._prepare(root, read, completed, fixture)
    if (p['primary_endpoint'] != SOURCE_STEP or p['shutdown_arm'] != BRANCH
            or p['own_shutdown_beta'] != 1. or p['source_branch'] != 'own_credit'):
        raise ValueError('Only the actual beta1 feedback 250k source is supported')
    if any(op.get('status') != 'acknowledged' for op in completed['ledger']['operations'].values()):
        raise ValueError('Completed paired source contains unconfirmed operations')
    _same(completed['counts']['control'], completed['counts']['feedback'], 'Completed pair is not equal PPO')
    if completed['counts']['feedback']['ppo_steps'] != SOURCE_STEP:
        raise ValueError('Source PPO endpoint is incomplete')
    prefix = 'branches/feedback/validation/step_0250000'
    report = read.json(root, prefix + '/report.json')
    marker, refresh = comparison._boundary(root, read, p, 'feedback', SOURCE_STEP, completed['ledger'], report)
    _same(marker['checkpoint'], audited['checkpoint'], 'Final marker differs from independently audited checkpoint')
    _same(refresh, result['refresh_status']['step_0250000'], 'Final refresh differs from completed comparison')
    _same(comparison._facts(report), result['primary']['feedback'], 'Fixed source report differs from comparison')
    binding = report['actor_bindings']
    if (binding['joint_steps'] != SOURCE_STEP or binding['feedback_branch'] != 'feedback'
            or binding['shutdown_arm'] != BRANCH or binding['own_shutdown_beta'] != 1.
            or binding['actor_parameters_sha256'] != audited['actor_parameters_sha256']):
        raise ValueError('Final report and actual checkpoint Actor identity differ')
    actor_relative = 'branches/feedback/actors/actor_0250000.npz'
    read.raw(root, actor_relative, binding['actor_sha256'])
    # Every source byte consumed here was already included in the external
    # completed comparison. In particular only the final marker is accepted;
    # effective_checkpoint's refresh_begin fallback is intentionally not used.
    for path, value in read.records.items():
        if path == str(report_path):
            continue
        _same(result['input_bindings'].get(path), value, 'Source byte is absent from the external comparison: ' + path)
    read.unchanged()
    descriptor = {'run': str(root), 'run_version': paired.VERSION, 'source_branch': 'feedback',
        'source_step': SOURCE_STEP, 'shutdown_arm': BRANCH, 'own_shutdown_beta': 1.,
        'comparison_path': str(report_path), 'comparison_sha256': expected_comparison_sha256,
        'completion_sha256': expected_completion_sha256, 'prepared_sha256': file_hash(root/'prepared.json'),
        'protocol_sha256': digest(protocol), 'scenario_manifest_sha256': digest(scenes),
        'boundary_marker_sha256': file_hash(root/'branches/feedback/boundaries/step_0250000.json'),
        'checkpoint': deepcopy(marker['checkpoint']), 'actor_sha256': binding['actor_sha256'],
        'actor_parameters_sha256': binding['actor_parameters_sha256'], 'actor_path': actor_relative,
        'validation_report_sha256': file_hash(root/prefix/'report.json'),
        'original_capability': deepcopy(result['primary']['feedback']),
        'feedback_disabled_after_source': True, 'source_feedback_trainer_version': feedback.VERSION,
        'test_fixture': fixture}
    return descriptor, p, protocol, scenes, marker


def load_source(source_run, comparison_path, expected_comparison_sha256, expected_completion_sha256,
                *, device, fixture=False, expected=None):
    _authorization()
    execution = sources()
    descriptor, p, protocol, scenes, marker = source_records(source_run, comparison_path,
        expected_comparison_sha256, expected_completion_sha256, fixture=fixture)
    if expected is not None:
        _same(descriptor, expected, 'Bound feedback source changed')
    if p['device'] != device:
        raise ValueError('The inherited training device cannot change')
    # Rebuild the actual parent class and ancestry. No class/version relabeling
    # or partial model-only checkpoint is accepted.
    actual, actual_protocol, actual_scenes, ancestor = paired.read_prepared(source_run, allow_test_fixture=fixture)
    _same(actual, p, 'Paired preparation changed during restoration')
    _same(actual_protocol, protocol, 'Paired protocol changed during restoration')
    _same(actual_scenes, scenes, 'Paired scenarios changed during restoration')
    wrapper = feedback.ShutdownFeedbackTrainer(protocol, ancestor, feedback_branch='feedback',
        expected_source_state_sha256=p['source_state_sha256'], source_checkpoint_sha256=p['source_checkpoint_sha256'],
        device=device, test_fixture=fixture)
    path = paired._check(Path(source_run).resolve(), marker['checkpoint'])
    payload = decode(path, marker['checkpoint']['sha256'])
    if (payload.get('version') != paired.VERSION or payload.get('branch') != 'feedback'
            or payload.get('cycle_id') != protocol['cycle_id'] or payload.get('new_training_steps') != 0
            or payload.get('predecessor_head') != marker['predecessor_head'] or payload.get('evidence') != marker['evidence']):
        raise ValueError('Source payload is not the complete acknowledged final boundary')
    wrapper.load_state_dict(payload['trainer'])
    if (type(wrapper.native) is not native.ShutdownContinuationTrainer or wrapper.joint_steps != SOURCE_STEP
            or wrapper.refresh_pending or wrapper.native.feedback_enabled or wrapper.native.beta != 1.):
        raise ValueError('Source native learner or completed refresh state differs')
    weights = {k.removeprefix('actor.'): v.detach().cpu().numpy()
        for k, v in wrapper.native.model.state_dict().items() if k.startswith('actor.')}
    actor_path = Path(source_run).resolve()/descriptor['actor_path']
    if file_hash(actor_path) != descriptor['actor_sha256']:
        raise ValueError('Bound source Actor changed')
    with np.load(io.BytesIO(actor_path.read_bytes()), allow_pickle=False) as archive:
        if set(archive.files) != {*weights, 'metadata_json'} or len(weights) != 6:
            raise ValueError('Actual source Actor inventory differs')
        if any(not np.array_equal(value, archive[key]) for key, value in weights.items()):
            raise ValueError('Actual complete source checkpoint and exported Actor differ')
    if feedback.actor_parameter_sha256(wrapper.native.model.actor.state_dict()) != descriptor['actor_parameters_sha256']:
        raise ValueError('Actual source Actor parameter semantic binding differs')
    if sources() != execution:
        raise ValueError('Execution sources changed during source loading')
    return wrapper.native, descriptor, scenes


def _trainer(prepared, protocol, source):
    return native.ShutdownContinuationTrainer(protocol, source,
        expected_source_state_sha256=prepared['source_state_sha256'],
        source_checkpoint_sha256=prepared['source_checkpoint_sha256'],
        device=prepared['device'], test_fixture=prepared['test_fixture'])


def _learning_checks(before, after, device):
    keys = ['model', 'optimizers', 'envs', 'rng', 'python_rng', 'numpy_rng', 'torch_rng',
        'partner_kinds', 'program_roles', 'scenario_ids', 'episode_context', 'episode_returns', 'episode_reward_components']
    if device == 'mps':
        keys.append('mps_rng')
    matched = {}
    for key in keys:
        matched[key] = initialization_sha256(after[key])
        _same(matched[key], initialization_sha256(before[key]), 'Continuation changed inherited ' + key)
    return matched


def prepare(output, *, source_run, comparison_path, expected_comparison_sha256,
            expected_completion_sha256, cycle_id=None, device='mps', allow_test_fixture=False):
    _sha(expected_comparison_sha256); _sha(expected_completion_sha256); _authorization()
    output = Path(output).expanduser().resolve()
    source_run, comparison_path = Path(source_run).expanduser().resolve(), Path(comparison_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a new independent finite diagnostic directory')
    paired._separate(output, source_run, comparison_path.parent)
    if type(allow_test_fixture) is not bool or device not in ('cpu', 'mps'):
        raise ValueError('Explicit device and fixture boundary required')
    source, descriptor, scenes = load_source(source_run, comparison_path, expected_comparison_sha256,
        expected_completion_sha256, device=device, fixture=allow_test_fixture)
    if (not allow_test_fixture and (len(scenes['splits']['validation']) != 50 or scenes['configuration']['horizon'] != 120)
            or PPO_CAP % len(source.envs) or PROBE % len(source.envs)):
        raise ValueError('Fixed 50k/probe and 150-episode validation matrix differ')
    before = source.state_dict(); state_sha = initialization_sha256(before)
    cycle_id = cycle_id or output.name
    protocol = native.make_protocol(source, source_checkpoint_sha256=descriptor['checkpoint']['sha256'],
        source_state_sha256=state_sha, ppo_cap=PPO_CAP, cycle_id=cycle_id,
        evaluation_checkpoints=[PPO_CAP], test_fixture=allow_test_fixture)
    p = {'version': VERSION, 'cycle_id': cycle_id, 'device': device, 'branch': 'own_credit',
        'shutdown_arm': BRANCH, 'own_shutdown_beta': 1., 'source_descriptor': descriptor,
        'source_checkpoint_sha256': descriptor['checkpoint']['sha256'], 'source_state_sha256': state_sha,
        'protocol_sha256': digest(protocol), 'scenario_manifest_sha256': digest(scenes),
        'baselines_sha256': file_hash(source_run/'baselines.json'), 'runtime_sources': sources(),
        'authorization_record_sha256': AUTHORIZATION_SHA, 'budget_caps': {BRANCH: {'ppo': PPO_CAP, 'evaluation': EVALUATION_CAP}},
        'primary_endpoint': PPO_CAP, 'probe_endpoint': PROBE, 'validation_endpoints': [PPO_CAP],
        'feedback_disabled_after_source': True, 'feedback_lambda': 0., 'stop_gate': 'full_and_warmup',
        'source_relationship': 'derived_single_arm_diagnostic_not_original_paired_extension',
        'test_fixture': allow_test_fixture, 'formal_ready': False, 'created_unix': time.time()}
    p['identity'] = {k: deepcopy(v) for k, v in p.items() if k not in ('created_unix', 'formal_ready')}
    learner = _trainer(p, protocol, source); saved = learner.state_dict()
    matched = _learning_checks(before, saved, device)
    if learner.joint_steps != 0 or learner.feedback_enabled or sources() != p['runtime_sources']:
        raise ValueError('New diagnostic must start at zero with unchanged sources and no tree KL')
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
    return {'status': 'prepared', 'output': str(output), 'cycle_id': cycle_id, 'ppo_cap': PPO_CAP,
        'evaluation_cap': EVALUATION_CAP, 'probe_endpoint': PROBE, 'feedback_disabled_after_source': True, 'formal_ready': False}


def _material(output, *, allow_test_fixture=False):
    output = Path(output).expanduser().resolve(); p = _json(output/'prepared.json')
    expected_identity = {k: v for k, v in p.items() if k not in ('identity', 'created_unix', 'formal_ready')}
    if (p.get('version') != VERSION or p.get('test_fixture') is not allow_test_fixture
            or p.get('runtime_sources') != sources() or p.get('shutdown_arm') != BRANCH
            or p.get('feedback_disabled_after_source') is not True or p.get('feedback_lambda') != 0.
            or p.get('stop_gate') != 'full_and_warmup' or p.get('branch') != 'own_credit'
            or p.get('own_shutdown_beta') != 1.):
        raise ValueError('Prepared energy diagnostic identity changed')
    _same(p['identity'], expected_identity, 'Prepared identity changed')
    _same(p['budget_caps'], {BRANCH: {'ppo': PPO_CAP, 'evaluation': EVALUATION_CAP}}, 'Fixed finite caps changed')
    if (p['primary_endpoint'], p['probe_endpoint'], p['validation_endpoints']) != (PPO_CAP, PROBE, [PPO_CAP]):
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
            or protocol.get('budget', {}).get('maximum_ppo_joint_steps') != PPO_CAP
            or protocol.get('budget', {}).get('maximum_ppo_joint_steps_per_arm') != PPO_CAP
            or protocol.get('evaluation', {}).get('checkpoints_ppo_steps') != [PPO_CAP]
            or protocol.get('evaluation', {}).get('maximum_environment_steps') != EVALUATION_CAP
            or protocol.get('evaluation', {}).get('read_final_test') is not False
            or protocol.get('source', {}).get('checkpoint_sha256') != p['source_checkpoint_sha256']
            or protocol.get('source', {}).get('state_sha256') != p['source_state_sha256']):
        raise ValueError('Native protocol differs from this finite beta1 diagnostic')
    initial = _json(output/'initialization_check.json')
    if (initial.get('identical_learning_state') is not True or initial.get('new_environment_steps') != 0
            or initial.get('source_state_sha256') != p['source_state_sha256']
            or initial.get('feedback_disabled_after_source') is not True):
        raise ValueError('Missing complete-state inheritance check')
    return p, protocol, scenes


def read_prepared(output, *, allow_test_fixture=False):
    p, protocol, scenes = _material(output, allow_test_fixture=allow_test_fixture)
    d = p['source_descriptor']
    source, actual, source_scenes = load_source(d['run'], d['comparison_path'], d['comparison_sha256'], d['completion_sha256'],
        device=p['device'], fixture=allow_test_fixture, expected=d)
    _same(scenes, source_scenes, 'Diagnostic scenarios differ from source')
    _same(initialization_sha256(source.state_dict()), p['source_state_sha256'], 'Full native source state changed')
    expected_protocol = native.make_protocol(source, source_checkpoint_sha256=p['source_checkpoint_sha256'],
        source_state_sha256=p['source_state_sha256'], ppo_cap=PPO_CAP, cycle_id=p['cycle_id'],
        evaluation_checkpoints=[PPO_CAP], test_fixture=allow_test_fixture)
    _same(protocol, expected_protocol, 'Genuine native protocol differs from fixed diagnostic')
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
    if target not in (PROBE, PPO_CAP) or trainer.shutdown_arm != BRANCH or trainer.feedback_enabled:
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


def _ready(report):
    return report['capability'].get('eligible') is True and report['warmup_capability'].get('eligible') is True


def _actor_arrays(path):
    with np.load(io.BytesIO(Path(path).read_bytes()), allow_pickle=False) as archive:
        names = {'0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias'}
        if set(archive.files) != names | {'metadata_json'}:
            raise ValueError('Unexpected terminal Actor inventory')
        arrays = {key: archive[key].copy() for key in names}
    semantic = family_explanation.actor_parameter_sha256(SimpleNamespace(weights=arrays))
    fields = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
        'semantic_sha256': initialization_sha256(value)} for key, value in sorted(arrays.items())}
    return arrays, semantic, fields


def _actor_receipt(output, learner, checkpoint, report):
    actor = output/'branches'/BRANCH/'actors'/f'actor_{PPO_CAP:07d}.npz'
    arrays, parameter_sha, fields = _actor_arrays(actor)
    actual = learner.model.actor.state_dict()
    if set(actual) != set(arrays) or any(not np.array_equal(actual[key].detach().cpu().numpy(), value) for key, value in arrays.items()):
        raise ValueError('Actual terminal learner Actor differs from its frozen NPZ')
    if feedback.actor_parameter_sha256(actual) != parameter_sha:
        raise ValueError('Terminal Actor tensor encoding differs from the static NPZ encoding')
    if file_hash(actor) != report['actor_bindings']['actor_sha256']:
        raise ValueError('Actual terminal Actor differs from its validation report')
    value = {'version': VERSION, 'cycle_id': learner.protocol['cycle_id'], 'joint_steps': PPO_CAP,
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


def read_completed(output, *, expected_completion_sha256, allow_test_fixture=False):
    """Read the externally anchored 50k endpoint, with no PT decode or sampling.

    A genuine static NumPy Actor is loaded by the saved-report reader. This is
    record verification, not new performance evaluation or model qualification.
    """
    _sha(expected_completion_sha256)
    output = Path(output).expanduser().resolve(); reads = _Inputs()
    completed = reads.json(output, 'completion_0050000.json', expected_completion_sha256)
    p, protocol, scenes = _material(output, allow_test_fixture=allow_test_fixture)
    for name in ('prepared.json', 'protocol.json', 'scenarios.json', 'baselines.json', 'authorization.json', 'initialization_check.json'):
        reads.raw(output, name)
    for name, sha in p['runtime_sources'].items():
        reads.raw(output, 'source_snapshot/'+name, sha)
    descriptor = p['source_descriptor']
    actual, _, _, _, _ = source_records(descriptor['run'], descriptor['comparison_path'],
        descriptor['comparison_sha256'], descriptor['completion_sha256'], fixture=allow_test_fixture)
    _same(actual, descriptor, 'Original complete feedback provenance changed')
    if (completed.get('version') != VERSION or completed.get('until') != PPO_CAP
            or completed.get('cycle_id') != p['cycle_id'] or completed.get('feedback_disabled_after_source') is not True
            or completed.get('feedback_lambda') != 0. or completed.get('used_final_test') is not False
            or any(completed.get(key) is not False for key in ('formal_ready', 'release_ready', 'explanation_qualified'))):
        raise ValueError('Not a completed fixed diagnostic endpoint')
    _same(completed['source_descriptor'], descriptor, 'Completed source descriptor differs')
    account = CycleBudget(output, p['identity']).read()
    _same(completed['ledger'], account, 'Completed ledger differs from acknowledged finite accounting')
    if any(op['status'] != 'acknowledged' for op in account['operations'].values()):
        raise ValueError('The diagnostic contains unconfirmed sampling')
    ppo = sorted((op for op in account['operations'].values() if op['request']['kind'] == 'ppo'), key=lambda op: op['sequence'])
    cell = account['branches'][BRANCH]['ppo']
    if cell['reserved'] != PPO_CAP or cell['acknowledged'] != PPO_CAP or completed['checkpoint'] != cell['head']:
        raise ValueError('The diagnostic does not have exactly 50000 confirmed PPO steps')
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
    report_path = output/'branches'/BRANCH/'validation/step_0050000/report.json'
    reads.json(report_path.parent, report_path.name, completed['report_sha256'])
    manifest = reads.json(report_path.parent, 'manifest.json', _sha(completed.get('validation_manifest_sha256')))
    for episode in manifest['episodes']:
        for kind in ('row', 'trace'):
            reads.bound(report_path.parent, episode[kind])
    report = stage._report_record(output, BRANCH, PPO_CAP, p, protocol, scenes, account, allow_test_fixture)
    receipt_binding = completed.get('actor_tensor_receipt', {})
    if receipt_binding.get('path') != 'actor_tensor_receipt.json':
        raise ValueError('Missing original current-checkpoint tensor receipt')
    tensor_receipt = _json_bytes(reads.bound(output, receipt_binding))
    actor_path = report_path.parent.parent.parent/'actors/actor_0050000.npz'
    reads.raw(actor_path.parent, actor_path.name, report['actor_bindings']['actor_sha256'])
    arrays, parameter_sha, fields = _actor_arrays(actor_path)
    if (tensor_receipt.get('version') != VERSION or tensor_receipt.get('cycle_id') != p['cycle_id']
            or tensor_receipt.get('joint_steps') != PPO_CAP or tensor_receipt.get('all_six_arrays_equal') is not True
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
        'source_descriptor': descriptor, 'inspection_scope': 'saved records; static NumPy Actor load; no forward, PT decode or physics',
        'formal_ready': False, 'release_ready': False, 'explanation_qualified': False}


def advance(output, *, until=None, allow_test_fixture=False):
    output = Path(output).expanduser().resolve(); raw = _json(output/'prepared.json')
    until = PPO_CAP if until is None else until
    if type(until) is not int or until not in (PROBE, PPO_CAP):
        raise ValueError('Use only the registered 4096 probe or 50000 endpoint')
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
        for target in (PROBE, PPO_CAP):
            if target > until or target < learner.joint_steps:
                continue
            train_to(output, learner, ledger, target)
            if target == PPO_CAP:
                folder = output/'branches'/BRANCH/'validation'/f'step_{PPO_CAP:07d}'
                if not (folder/'report.json').exists():
                    stage.evaluate_boundary(output, learner, ledger, BRANCH, scenes)
                report = stage._report_record(output, BRANCH, PPO_CAP, p, protocol, scenes, ledger.read(), allow_test_fixture)
        ready = report is not None and _ready(report)
        tensor_receipt = _actor_receipt(output, learner, ledger.head('ppo', BRANCH), report) if report else None
        training_records = {}
        for opid, operation in ledger.read()['operations'].items():
            if operation['request']['kind'] == 'ppo':
                sidecar = Path(operation['completion']['checkpoint']['path']).with_suffix('.json')
                training_records[opid] = {'path': str(sidecar), 'sha256': file_hash(output/sidecar), 'size': (output/sidecar).stat().st_size}
        result = {'version': VERSION, 'status': 'both_gates_ready' if ready else ('fixed_endpoint_completed' if learner.joint_steps == PPO_CAP else 'probe_completed'),
            'until': learner.joint_steps, 'cycle_id': p['cycle_id'], 'checkpoint': ledger.head('ppo', BRANCH),
            'counts': _counts(learner), 'training_records': training_records, 'actor_tensor_receipt': tensor_receipt,
            'ledger': ledger.read(), 'feedback_disabled_after_source': True, 'feedback_lambda': 0.,
            'source_descriptor': p['source_descriptor'], 'original_pair_preserved': True,
            'capability': report['capability'] if report else None, 'warmup_capability': report['warmup_capability'] if report else None,
            'report_sha256': file_hash(output/'branches'/BRANCH/'validation'/f'step_{PPO_CAP:07d}'/'report.json') if report else None,
            'validation_manifest_sha256': file_hash(output/'branches'/BRANCH/'validation'/f'step_{PPO_CAP:07d}'/'manifest.json') if report else None,
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
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--run', action='store_true')
    parser.add_argument('--source-run'); parser.add_argument('--comparison-path')
    parser.add_argument('--expected-comparison-sha256'); parser.add_argument('--expected-completion-sha256')
    parser.add_argument('--cycle-id'); parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--until', type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        if not all((args.source_run, args.comparison_path, args.expected_comparison_sha256, args.expected_completion_sha256)):
            parser.error('Preparation requires the actual source run, comparison path and both external SHA256 anchors')
        result = prepare(args.output, source_run=args.source_run, comparison_path=args.comparison_path,
            expected_comparison_sha256=args.expected_comparison_sha256,
            expected_completion_sha256=args.expected_completion_sha256, cycle_id=args.cycle_id, device=args.device)
    else:
        # The caller can also redirect stdout; this local log preserves each
        # acknowledged chunk across normal separate-process resumes.
        class Tee:
            def __init__(self, stream, log): self.stream, self.log = stream, log
            def write(self, value): self.stream.write(value); self.log.write(value); self.log.flush()
            def flush(self): self.stream.flush(); self.log.flush()
        with (Path(args.output)/'stdout.log').open('a', encoding='utf-8') as log:
            previous = sys.stdout; sys.stdout = Tee(previous, log)
            try: result = advance(args.output, until=args.until)
            finally: sys.stdout = previous
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
