"""Finite same-source PPO / RCPD-feedback cycle for the qualified 3.38m NN.

This driver has its own source admission, strict development guard, checkpoint
identity and accounts. The genuine retained-beta trainer still owns PPO/KL,
Adam, RNG and environment history. Frozen extraction/evaluation primitives keep
their original producer versions. No tree supplies a runtime action and neither
training reliability nor this experiment grants explanation/release status.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save, reserve_sampling
from .warehouse_native_partner_mix_run import (
    write_bytes, write_json, decode, save_batch, AUTHORIZATION, AUTHORIZATION_SHA)
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_native_cycle_budget import CycleBudget
from . import warehouse_family_native_cycle_run as native_cycle
from . import warehouse_family_feedback_cycle_source as source_api
from . import warehouse_native_shutdown_feedback_run as primitive
from . import warehouse_native_shutdown_stage_rcpd as extraction
from . import warehouse_native_expanded_rcpd as expanded
from env.warehouse.domain import collaborative_study_config

VERSION = 'warehouse-family-feedback-cycle-run.v1'
BRANCHES = ('control', 'feedback')
PPO_CAP = 500000
VALIDATION_INTERVAL = 50000
PROBE = 4096
GUARD = {'full_and_warmup': True, 'per_partner_nn_retention': .9,
    'nn_reference': 'fixed_starting_NN_validation',
    'team_reference': 'original_program_reference',
    'team_drop_guard': 'unchanged_ExactProgramManager',
    'roles': 'all_actual_neural_training_rows', 'lambda_max': .01,
    'ramp_steps': 100000, 'qualification_granted': False}


def sources():
    result = primitive.sources()
    for collection in (native_cycle.sources(), source_api.sources()):
        for name, sha in collection.items():
            if name in result and result[name] != sha:
                raise ValueError('Execution source bindings disagree')
            result[name] = sha
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


_json = primitive._json
_same = primitive._same
_bound = primitive._bound
_check = primitive._check
# These functions perform genuine fixed-matrix inference / saved-record checks;
# their own stage evaluator version remains visible in all produced records.
evaluate_boundary = primitive.evaluate_boundary
_report_record = primitive._report_record


def _admit(source, completion_sha, tree, manifest_sha, *, fixture):
    if fixture:
        raise ValueError('Synthetic admission needs explicit test transport; no production gate is bypassed')
    return source_api.admit(source, completion_sha, tree, manifest_sha)


def _restore_source(source, completion_sha, *, device, fixture):
    # load_source restores the supplied run's completed checkpoint. Its ancestor
    # is used only to construct the actual native learner before strict loading.
    return native_cycle.load_source(source, completion_sha, device=device, fixture=fixture)


def _trainer(prepared, protocol, source, branch):
    from .warehouse_native_shutdown_feedback_trainer import ShutdownFeedbackTrainer
    return ShutdownFeedbackTrainer(protocol, source, feedback_branch=branch,
        expected_source_state_sha256=prepared['source_state_sha256'],
        source_checkpoint_sha256=prepared['source_checkpoint_sha256'],
        device=prepared['device'], test_fixture=prepared['test_fixture'])


def _guard(report, source_report, reference):
    value = source_api.gate_values(report, source_report, team_reference=reference)
    # Diagnostics and unrelated eligibility flags are never passed to trainer.
    kwargs = {key: value['update_kwargs'][key] for key in
              ('validation_score', 'reference_score', 'capability_eligible')}
    return kwargs, value['diagnostics']


def _install(trainer, fit, bindings, report, report_sha, source_report, reference):
    if fit.get('reliable') is not True:
        raise ValueError('Installing an unreliable same-Actor tree is forbidden')
    trainer.attach_feedback_state(fit['manager_state'],
        source_actor_sha256=bindings['actor_sha256'],
        source_actor_parameters_sha256=bindings['actor_parameters_sha256'],
        evidence_sha256=fit['evidence_sha256'])
    kwargs, diagnostics = _guard(report, source_report, reference)
    gate = trainer.update_feedback_gate(**kwargs, capability_report_sha256=report_sha)
    return {'manager': gate, 'strict_guard': diagnostics,
            'capability_eligible': kwargs['capability_eligible']}


def _schedule(cap, interval, n, fixture, fixture_probe_endpoint=None):
    if (type(fixture) is not bool or type(cap) is not int or type(interval) is not int
            or n < 1 or cap < n or interval < n or cap % n or interval % n):
        raise ValueError('Finite boundaries must be positive complete environment batches')
    if not fixture and (cap != PPO_CAP or interval != VALIDATION_INTERVAL or n != 16
                        or fixture_probe_endpoint is not None):
        raise ValueError('Production fixes 16 environments, 500k per arm, 50k intervals and 4096 probe')
    endpoints = list(range(interval, cap, interval)) + [cap]
    probe = min(PROBE, cap) if fixture_probe_endpoint is None else fixture_probe_endpoint
    if type(probe) is not int or probe < n or probe % n or probe > min(cap, endpoints[0]):
        raise ValueError('The registered probe must fit before the first validation')
    return endpoints, probe


def _caps(endpoints, pools, scenes):
    horizon = scenes['configuration']['horizon']
    contexts = primitive._contexts(pools, horizon)
    one_aux = sum(x['horizon'] for x in contexts)
    one_eval = 3 * len(scenes['splits']['validation']) * horizon
    aux = {'control': {f'step_{endpoints[-1]:07d}': one_aux},
           'feedback': {f'step_{s:07d}': one_aux for s in endpoints}}
    return contexts, {b: {'ppo': endpoints[-1], 'evaluation': len(endpoints)*one_eval}
                      for b in BRANCHES}, aux


def _initial_binding(admitted):
    return {key: deepcopy(admitted[key]) for key in
            ('descriptor', 'actor_bindings', 'input_bindings', 'source_hashes', 'role_scope')}


def prepare(output, *, source_run, expected_source_completion_sha256, initial_tree,
            expected_initial_tree_manifest_sha256, ppo_cap=PPO_CAP,
            validation_interval=VALIDATION_INTERVAL, cycle_id=None, device='mps',
            allow_test_fixture=False, fixture_probe_endpoint=None):
    from .warehouse_native_shutdown_feedback_trainer import make_protocol
    output = Path(output).expanduser().resolve()
    source_run = Path(source_run).expanduser().resolve()
    initial_tree = Path(initial_tree).expanduser().resolve()
    primitive._separate(output, source_run, initial_tree)
    for sha in (expected_source_completion_sha256, expected_initial_tree_manifest_sha256):
        source_api._sha(sha)
    if (type(allow_test_fixture) is not bool or device not in ('cpu', 'mps')
            or (not allow_test_fixture and device != 'mps')
            or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA):
        raise ValueError('Explicit production MPS / fixture boundary and unchanged authorization are required')
    if (type(ppo_cap) is not int or ppo_cap < 1 or type(validation_interval) is not int
            or validation_interval < 1 or (fixture_probe_endpoint is not None
            and (type(fixture_probe_endpoint) is not int or fixture_probe_endpoint < 1))
            or (not allow_test_fixture and (ppo_cap != PPO_CAP or validation_interval != VALIDATION_INTERVAL
                                            or fixture_probe_endpoint is not None))):
        raise ValueError('Invalid or unregistered finite training schedule')
    runtime = sources()
    admitted = _admit(source_run, expected_source_completion_sha256, initial_tree,
                      expected_initial_tree_manifest_sha256, fixture=allow_test_fixture)
    source, descriptor, scenes = _restore_source(source_run, expected_source_completion_sha256,
                                                 device=device, fixture=allow_test_fixture)
    _same(scenes, admitted['scenes'], 'Restored source scenarios differ from admitted records')
    if source.shutdown_arm != 'beta1' or source.beta != 1. or source.branch != 'own_credit':
        raise ValueError('The actual retained beta1 / own_credit learner is required')
    state = source.state_dict(); state_sha = semantic(state)
    saved = admitted['saved_source']; fit = admitted['fit_result']
    _same(descriptor, admitted['descriptor'], 'Genuine restoration differs from admitted source identity')
    _same(descriptor['checkpoint'], saved['checkpoint'], 'Restoration did not load the admitted current endpoint')
    _same(descriptor['actor_sha256'], admitted['actor_bindings']['actor_sha256'], 'Source Actor export changed')
    if source.joint_steps != admitted['descriptor']['source_step']:
        raise ValueError('Restored source clock differs')
    n = source.cfg['environments']
    endpoints, probe = _schedule(ppo_cap, validation_interval, n, allow_test_fixture, fixture_probe_endpoint)
    protocol = make_protocol(source, source_checkpoint_sha256=descriptor['checkpoint']['sha256'],
        source_state_sha256=state_sha, ppo_cap=ppo_cap, cycle_id=cycle_id or output.name,
        evaluation_checkpoints=endpoints, feedback_config=expanded.feedback_config(), test_fixture=allow_test_fixture)
    pools = deepcopy(admitted['pools']); contexts, caps, aux = _caps(endpoints, pools, scenes)
    if not allow_test_fixture and (len(pools['train']), len(pools['selection']),
            len(contexts), sum(c['horizon'] for c in contexts)) != (192, 32, 896, 107520):
        raise ValueError('The original fixed 192/32 pools and four profiles are required')
    report_path = Path(admitted['source_report_path'])
    if report_path != source_run/'branches/beta1/validation'/f"step_{source.joint_steps:07d}"/'report.json':
        raise ValueError('Source report is not the actual current endpoint')
    if file_hash(report_path) != admitted['report_sha256']:
        raise ValueError('Original admitted source report bytes changed')
    _same(_json(report_path), admitted['report'], 'Original source report content differs')
    _same(_json(source_run/'baselines.json'), admitted['baselines'], 'Original baseline bytes differ')
    initial = _initial_binding(admitted)
    prepared = {'version': VERSION, 'cycle_id': protocol['cycle_id'], 'device': device,
        'source_run': str(source_run), 'source_completion_sha256': expected_source_completion_sha256,
        'source_descriptor': descriptor, 'source_branch': source.branch, 'shutdown_arm': source.shutdown_arm,
        'own_shutdown_beta': source.beta, 'source_checkpoint_sha256': descriptor['checkpoint']['sha256'],
        'source_state_sha256': state_sha, 'source_report_sha256': admitted['report_sha256'],
        'source_cumulative_steps': source.source_counters['joint_steps']+source.joint_steps,
        'initial_tree': {'root': str(initial_tree), 'manifest_sha256': expected_initial_tree_manifest_sha256,
                         'fit_sha256': digest(fit), 'binding_sha256': digest(initial)},
        'protocol_sha256': digest(protocol), 'scenario_manifest_sha256': digest(scenes),
        'runtime_sources': runtime, 'budget_caps': caps, 'auxiliary_caps': aux,
        'maximum_auxiliary_steps': sum(sum(x.values()) for x in aux.values()),
        'refresh_pools_sha256': digest(pools), 'refresh_contexts_sha256': digest(contexts),
        'extraction_contract': expanded.contract(), 'collection_primitive_version': primitive.VERSION,
        'maximum_rcpd_calls_per_extraction': 13, 'maximum_sklearn_fits_per_extraction': 104,
        'primary_endpoint': ppo_cap, 'validation_endpoints': endpoints,
        'validation_interval': validation_interval, 'probe_endpoint': probe,
        'fixture_probe_endpoint': fixture_probe_endpoint, 'environment_batch_size': n,
        'guard': deepcopy(GUARD), 'test_fixture': allow_test_fixture,
        'baselines_sha256': file_hash(source_run/'baselines.json'),
        'authorization_record_sha256': AUTHORIZATION_SHA,
        'formal_ready': False, 'explanation_qualified': False, 'created_unix': time.time()}
    prepared['identity'] = {k: deepcopy(v) for k, v in prepared.items()
                           if k not in ('created_unix', 'formal_ready', 'explanation_qualified')}
    trainers = {b: _trainer(prepared, protocol, source, b) for b in BRANCHES}
    initial_gate = _install(trainers['feedback'], fit, admitted['actor_bindings'], admitted['report'],
        admitted['report_sha256'], admitted['report'], admitted['baselines']['reference'])
    if not initial_gate['capability_eligible']:
        raise ValueError('The initial full AND warmup AND per-partner retention guard must pass')
    _same(semantic(trainers['control'].native.state_dict()), semantic(trainers['feedback'].native.state_dict()),
          'Initial arms differ in actual native learning state')
    native = trainers['control'].native.state_dict(); matched = {}
    for key in ('model', 'optimizers', 'envs', 'rng', 'python_rng', 'numpy_rng', 'torch_rng',
                'partner_kinds', 'program_roles', 'scenario_ids', 'episode_context',
                'episode_returns', 'episode_reward_components'):
        matched[key] = semantic(native[key])
        _same(matched[key], semantic(state[key]), 'Source learning state changed: '+key)
    if device == 'mps':
        _same(semantic(native['mps_rng']), semantic(state['mps_rng']), 'Source MPS RNG changed')
    if sources() != runtime:
        raise ValueError('Sources changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('prepared', prepared), ('protocol', protocol), ('scenarios', scenes),
                        ('refresh_pools', pools), ('refresh_contexts', contexts),
                        ('initial_fit', fit), ('initial_admission', initial)):
        write_json(output/(name+'.json'), value)
    for target, source_path in (('source_validation.json', report_path),
                               ('baselines.json', source_run/'baselines.json'), ('authorization.json', AUTHORIZATION)):
        write_bytes(output/target, source_path.read_bytes())
    for name in runtime:
        write_bytes(output/'source_snapshot'/name, (ROOT/name).read_bytes())
    ledger = CycleBudget.create(output, prepared['identity'])
    with ledger.lease():
        for branch, trainer in trainers.items():
            checkpoint = output/'branches'/branch/'checkpoints/initial.pt'
            atomic_torch_save(checkpoint, {'version': VERSION, 'cycle_id': protocol['cycle_id'], 'branch': branch,
                'source_branch': source.branch, 'shutdown_arm': source.shutdown_arm, 'operation_id': None,
                'trainer': trainer.state_dict(), 'new_environment_steps': 0})
            ledger.initialize_head('ppo', branch, str(checkpoint.relative_to(output)), file_hash(checkpoint))
            evidence = output/'branches'/branch/'initial_evaluation_reference.json'
            write_json(evidence, {'source_descriptor': descriptor, 'source_report_sha256': admitted['report_sha256'],
                                 'new_environment_steps': 0})
            ledger.initialize_head('evaluation', branch, str(evidence.relative_to(output)), file_hash(evidence))
    write_json(output/'initialization_check.json', {'version': VERSION, 'identical_native_learning_state': True,
        'matched': matched, 'source_state_sha256': state_sha, 'initial_gate': initial_gate,
        'new_environment_steps': 0, 'program_runtime_controller': False})
    return {'version': VERSION, 'status': 'prepared', 'output': str(output), 'budget_caps': caps,
            'maximum_auxiliary_steps': prepared['maximum_auxiliary_steps'], 'new_environment_steps': 0,
            'initial_tree_budget_is_separate': True, 'explanation_qualified': False}


def _check_inputs(bindings):
    for name, item in bindings.items():
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or path.stat().st_size != item['size'] or file_hash(path) != item['sha256']):
            raise ValueError('Originally admitted source input changed: '+name)


def read_prepared(output, *, allow_test_fixture=False):
    from .warehouse_native_shutdown_feedback_trainer import make_protocol
    output = Path(output).expanduser().resolve(); p = _json(output/'prepared.json')
    if (p.get('version') != VERSION or p.get('test_fixture') is not allow_test_fixture
            or p['runtime_sources'] != sources()):
        raise ValueError('Feedback cycle version, source closure or fixture differs')
    identity = {k: deepcopy(v) for k, v in p.items()
                if k not in ('identity', 'created_unix', 'formal_ready', 'explanation_qualified')}
    _same(identity, p['identity'], 'Cycle identity differs')
    _same(p['guard'], GUARD, 'Registered strict guard changed')
    if (p.get('formal_ready') is not False or p.get('explanation_qualified') is not False
            or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA):
        raise ValueError('This driver cannot grant release or change authorization')
    for name, key in (('authorization.json', 'authorization_record_sha256'),
                      ('baselines.json', 'baselines_sha256'), ('source_validation.json', 'source_report_sha256')):
        if file_hash(output/name) != p[key]:
            raise ValueError('Bound original input changed: '+name)
    for name, sha in p['runtime_sources'].items():
        if file_hash(output/'source_snapshot'/name) != sha:
            raise ValueError('Archived execution source changed')
    protocol, scenes, pools, contexts = (_json(output/(n+'.json')) for n in
                                        ('protocol', 'scenarios', 'refresh_pools', 'refresh_contexts'))
    for value, key in ((protocol, 'protocol_sha256'), (scenes, 'scenario_manifest_sha256'),
                       (pools, 'refresh_pools_sha256'), (contexts, 'refresh_contexts_sha256')):
        _same(digest(value), p[key], 'Bound configuration differs: '+key)
    initial = _json(output/'initial_admission.json'); fit = _json(output/'initial_fit.json')
    _same(digest(initial), p['initial_tree']['binding_sha256'], 'Original admitted tree/source bindings differ')
    _same(digest(fit), p['initial_tree']['fit_sha256'], 'Initial tree fit changed')
    _check_inputs(initial['input_bindings'])
    source, descriptor, source_scenes = _restore_source(p['source_run'], p['source_completion_sha256'],
                                                       device=p['device'], fixture=allow_test_fixture)
    _same(descriptor, p['source_descriptor'], 'Actual restored source descriptor changed')
    _same(source_scenes, scenes, 'Restored source scenarios differ')
    _same(semantic(source.state_dict()), p['source_state_sha256'], 'Restored complete source state changed')
    endpoints, probe = _schedule(p['primary_endpoint'], p['validation_interval'], source.cfg['environments'],
                                allow_test_fixture, p['fixture_probe_endpoint'])
    _same(endpoints, p['validation_endpoints'], 'Registered endpoints changed')
    _same(probe, p['probe_endpoint'], 'Registered probe changed')
    expected = make_protocol(source, source_checkpoint_sha256=p['source_checkpoint_sha256'],
        source_state_sha256=p['source_state_sha256'], ppo_cap=p['primary_endpoint'], cycle_id=p['cycle_id'],
        evaluation_checkpoints=endpoints, feedback_config=expanded.feedback_config(), test_fixture=allow_test_fixture)
    _same(protocol, expected, 'Actual retained-beta feedback protocol differs')
    expected_contexts, caps, aux = _caps(endpoints, pools, scenes)
    _same(contexts, expected_contexts, 'Extraction order or episode seeds changed')
    _same(p['budget_caps'], caps, 'PPO/validation caps changed')
    _same(p['auxiliary_caps'], aux, 'Auxiliary schedule changed')
    _same(p['maximum_auxiliary_steps'], sum(sum(v.values()) for v in aux.values()), 'Auxiliary cap changed')
    _same(p['extraction_contract'], expanded.contract(), 'Actual expanded extraction contract changed')
    if (p['collection_primitive_version'] != primitive.VERSION or p['maximum_rcpd_calls_per_extraction'] != 13
            or p['maximum_sklearn_fits_per_extraction'] != 104):
        raise ValueError('Actual extraction producer or finite fit counts changed')
    init = _json(output/'initialization_check.json')
    if (init.get('version') != VERSION or init.get('identical_native_learning_state') is not True
            or init['source_state_sha256'] != p['source_state_sha256']):
        raise ValueError('Preparation was not completed')
    kwargs, _ = _guard(_json(output/'source_validation.json'), _json(output/'source_validation.json'),
                        _json(output/'baselines.json')['reference'])
    if not kwargs['capability_eligible']:
        raise ValueError('Original admitted source gate no longer passes')
    return p, protocol, scenes, source

# Versioned checkpoint and sampling orchestration adapted from the frozen
# primitive. The actual learner.collect/update mathematics is not copied here.


def effective_checkpoint(output,ledger,branch):
    head=ledger.head('ppo',branch);path=_check(output,head);payload=decode(path,head['sha256'])
    if payload.get('version')!=VERSION or payload.get('branch')!=branch:raise ValueError('Wrong acknowledged feedback checkpoint')
    if head['step']:
        operation=ledger.read()['operations'].get(payload.get('operation_id'))
        if (not operation or operation['status']!='acknowledged' or operation['completion']['checkpoint']!=head
                or payload.get('audit',{}).get('neural_overrides')!=0):raise ValueError('Missing original PPO acknowledgment')
    for binding in payload.get('evidence',{}).values():_check(output,binding)
    marker=output/'branches'/branch/'boundaries'/f"step_{head['step']:07d}.json"
    if not marker.exists():
        marker=marker.with_name(marker.stem+'_refresh_begin.json')
    if marker.exists():
        rec=_json(marker);_same(rec['predecessor_head'],head,'Boundary refers to another acknowledged checkpoint')
        if (rec['branch']!=branch or rec['version']!=VERSION
                or rec.get('new_training_steps')!=0):raise ValueError('Boundary condition differs')
        for binding in rec['evidence'].values():_check(output,binding)
        bpath=_check(output,rec['checkpoint']);boundary=decode(bpath,rec['checkpoint']['sha256'])
        if (boundary.get('version')!=VERSION or boundary.get('branch')!=branch
                or boundary.get('predecessor_head')!=head or boundary.get('evidence')!=rec['evidence']):raise ValueError('Boundary checkpoint binding differs')
        for key in ('gate', 'terminal_tree'):
            if key in rec or key in boundary:
                _same(rec.get(key), boundary.get(key), 'Boundary decision evidence differs: '+key)
        _same(semantic(payload['trainer']['native_state']),semantic(boundary['trainer']['native_state']),'Boundary changed native learner state')
        payload=boundary
    return head,payload


def train_to(output,trainer,ledger,branch,target):
    if any(ledger.read()['branches'][branch][kind]['pending'] for kind in ('ppo','evaluation')):
        raise ValueError('Unconfirmed paired operation requires diagnosis; never resample')
    n=trainer.cfg['environments']
    while trainer.joint_steps<target:
        if trainer.joint_steps in trainer.protocol['evaluation']['checkpoints_ppo_steps']:
            marker=output/'branches'/branch/'boundaries'/f'step_{trainer.joint_steps:07d}.json'
            if not marker.exists():raise ValueError('Fixed boundary validation and refresh must finish before further PPO')
        if any(ledger.read()['branches'][branch][kind]['pending'] for kind in ('ppo','evaluation')):
            raise ValueError('Unconfirmed paired operation requires diagnosis; never resample')
        amount=min(n*trainer.cfg['rollout_steps'],target-trainer.joint_steps,ledger.remaining('ppo',branch));amount-=amount%n
        if amount<=0:raise ValueError('Reserved budget cannot reach this fixed paired endpoint')
        head,old=effective_checkpoint(output,ledger,branch);before=semantic(trainer.state_dict())
        if head['step']!=trainer.joint_steps or semantic(old['trainer'])!=before:raise ValueError('Learner differs from confirmed paired predecessor')
        opid='ppo_'+uuid.uuid4().hex;reservation=ledger.reserve('ppo',branch,amount,opid,expected_step=trainer.joint_steps)
        if not reservation['execution_permitted']:raise ValueError('Repeated reservation is not execution permission')
        # The latest completed validation remains the ramp/performance evidence
        # between fixed refreshes. Every current-state ramp change is checkpointed.
        capability=trainer.capability_evidence
        if trainer.feedback_enabled and capability is not None and trainer.feedback.reliable:
            gate=capability['gate'];trainer.update_feedback_gate(gate['validation_score'],
                capability_eligible=capability['capability_eligible'],reference_score=gate['reference_score'],
                capability_report_sha256=capability['report_sha256'])
        started=time.monotonic();batch,metrics=trainer.train_chunk(amount//n)
        if (trainer.joint_steps!=head['step']+amount or len(batch['transition_records'])!=amount
                or batch['audit']['neural_overrides']!=0 or any(not np.isfinite(v) for v in metrics.values())
                or any(r.get('feedback_branch')!=branch or r.get('shutdown_arm')!=trainer.shutdown_arm
                       or r.get('own_shutdown_beta')!=trainer.beta for r in batch['transition_records'])
                or (branch=='control' and metrics.get('feedback_loss',0)!=0)):
            raise ValueError('Actual paired training differs from reserved NN-only contract')
        evidence=save_batch(output,branch,opid,batch);episodes=deepcopy(trainer.completed_episodes);trainer.completed_episodes.clear()
        path=output/'branches'/branch/'checkpoints'/(f'step_{trainer.joint_steps:07d}_'+opid+'.pt')
        payload={'version':VERSION,'cycle_id':trainer.protocol['cycle_id'],'branch':branch,'source_branch':trainer.branch,'shutdown_arm':trainer.shutdown_arm,
            'operation_id':opid,'trainer':trainer.state_dict(),'audit':batch['audit'],'metrics':metrics,'evidence':evidence,
            'actual_steps':amount,'completed_episodes':episodes,'before_state_sha256':before,'elapsed_seconds':time.monotonic()-started}
        atomic_torch_save(path,payload);ledger.ack(opid,str(path.relative_to(output)),file_hash(path),amount)
        write_json(output/'progress.json',{'status':'training','branch':branch,'steps':trainer.joint_steps,
            'metrics':metrics,'ledger':ledger.read()},replace=True)
        print(json.dumps({'event':'paired_ppo_ack','branch':branch,'steps':trainer.joint_steps,
            'lambda':metrics.get('feedback_lambda',0),'neural_overrides':batch['audit']['neural_overrides']}),flush=True)


def auxiliary_accounting(output, prepared):
    """No refunded reservations: each registered branch/step has one ledger."""
    output = Path(output); total_reserved = total_actual = 0; result = {}
    for branch in BRANCHES:
        root = output/'branches'/branch/'refresh'; rows = {}
        if root.exists():
            for folder in root.iterdir():
                if not folder.is_dir() or folder.name not in prepared['auxiliary_caps'][branch]:
                    raise ValueError('Unregistered auxiliary directory')
                plan = _json(folder/'plan.json'); budget = _json(folder/'auxiliary_budget.json')
                manifest = _json(folder/'manifest.json')
                if (plan.get('parent_driver') != VERSION or plan.get('feedback_branch') != branch
                        or plan.get('pair_identity_sha256') != digest(prepared['identity'])
                        or plan.get('maximum_auxiliary_steps') != prepared['auxiliary_caps'][branch][folder.name]
                        or budget['cap'] != plan['maximum_auxiliary_steps']
                        or budget['reserved_joint_steps'] != manifest['reserved_auxiliary_steps']
                        or any(e['status'] != 'completed' for e in manifest['episodes'])):
                    raise ValueError('Unconfirmed or mismatched auxiliary execution; never resample')
                primitive.check_manifest(folder, plan, manifest, fixture=prepared['test_fixture'])
                if (manifest.get('status') == 'fitting' or (folder/'fit_failure.json').exists()
                        or (folder/'sampling_failure.json').exists()):
                    raise ValueError('Unconfirmed auxiliary fit/sampling requires diagnosis, not retry')
                reserved = budget['reserved_joint_steps']; actual = manifest['actual_auxiliary_steps']
                rows[folder.name] = {'reserved': reserved, 'acknowledged': actual,
                    'status': manifest['status'], 'reliable': manifest.get('reliable'),
                    'manifest': _bound(output, folder/'manifest.json')}
                total_reserved += reserved; total_actual += actual
        result[branch] = rows
    if total_reserved > prepared['maximum_auxiliary_steps']:
        raise ValueError('Whole-cycle auxiliary cap exceeded')
    return {'cap': prepared['maximum_auxiliary_steps'], 'reserved': total_reserved,
            'acknowledged': total_actual, 'branches': result}


def refresh(output, prepared, trainer, actor, prior, branch):
    """Use actual original collector/fit producer, explicitly nested in this run."""
    folder = output/'branches'/branch/'refresh'/f'step_{trainer.joint_steps:07d}'
    if folder.name not in prepared['auxiliary_caps'][branch]:
        raise ValueError('Unregistered extraction boundary')
    bindings = {key: actor.metadata[key] for key in extraction.required_bindings(trainer.protocol)
                if key != 'actor_sha256'}
    bindings.update(actor_sha256=actor.artifact_sha256, cycle_id=trainer.protocol['cycle_id'],
                    feedback_branch=branch)
    pools = _json(output/'refresh_pools.json'); contexts = _json(output/'refresh_contexts.json')
    plan = {'version': primitive.VERSION, 'pair_version': VERSION, 'parent_driver': VERSION,
        'feedback_branch': branch, 'pair_identity_sha256': digest(prepared['identity']),
        'actor_bindings': bindings, 'protocol_sha256': digest(trainer.protocol),
        'scene_pools_sha256': digest(pools), 'contexts': contexts,
        'maximum_auxiliary_steps': prepared['auxiliary_caps'][branch][folder.name],
        'cumulative_fit_step': trainer.feedback_clock, 'ppo_steps': 0, 'test_fixture': trainer.test_fixture,
        'prior_manager_sha256': digest(prior), 'extraction_config': extraction.extraction_config(),
        'expanded_fit_contract': expanded.contract(), 'maximum_rcpd_calls': 13, 'maximum_sklearn_fits': 104,
        'runtime_sources': prepared['runtime_sources'],
        'purpose': 'feedback_refresh' if branch == 'feedback' else 'control_terminal_tree',
        'explanation_qualified': False}
    if folder.exists():
        _same(_json(folder/'plan.json'), plan, 'Extraction plan or prior manager changed')
    else:
        folder.mkdir(parents=True, exist_ok=False); write_json(folder/'plan.json', plan)
        reserve_sampling(folder/'auxiliary_budget.json', 0, cap=plan['maximum_auxiliary_steps'])
        write_json(folder/'manifest.json', {'version': primitive.VERSION, 'parent_driver': VERSION,
            'plan_sha256': digest(plan), 'status': 'prepared', 'episodes': [],
            'actual_auxiliary_steps': 0, 'reserved_auxiliary_steps': 0, 'ppo_steps': 0})
    # No unconfirmed operation in any branch can be bypassed by switching arms.
    auxiliary_accounting(output, prepared)
    manifest = primitive.collect_dataset(folder, plan, pools, actor, trainer.protocol,
        fixture=trainer.test_fixture,
        config=collaborative_study_config(horizon=_json(output/'scenarios.json')['configuration']['horizon']))
    _same(trainer.protocol['feedback_config'], asdict(expanded.feedback_config()), 'Actual fit configuration differs')
    fit = primitive.fit_collected(folder, manifest, plan, actor, prior_manager_state=prior,
                                 feedback_config=expanded.feedback_config(), fixture=trainer.test_fixture)
    return fit, bindings, folder


def _boundary_payload(output, ledger, branch, marker):
    receipt = _json(marker); head = ledger.head('ppo', branch)
    if (receipt.get('version') != VERSION or receipt.get('branch') != branch
            or receipt.get('new_training_steps') != 0):
        raise ValueError('Boundary identity differs')
    _same(receipt['predecessor_head'], head, 'Boundary predecessor differs')
    for item in receipt['evidence'].values(): _check(output, item)
    payload = decode(_check(output, receipt['checkpoint']), receipt['checkpoint']['sha256'])
    if (payload.get('version') != VERSION or payload.get('branch') != branch
            or payload.get('predecessor_head') != head or payload.get('evidence') != receipt['evidence']):
        raise ValueError('Boundary checkpoint envelope differs')
    for key in ('gate', 'terminal_tree'):
        if key in receipt or key in payload:
            _same(receipt.get(key), payload.get(key), 'Boundary decision evidence differs: '+key)
    return receipt, payload


def finish_boundary(output, prepared, trainer, ledger, branch, report, actor):
    head = ledger.head('ppo', branch); step = trainer.joint_steps
    folder = output/'branches'/branch/'boundaries'; marker = folder/f'step_{step:07d}.json'
    if marker.exists():
        receipt, saved = _boundary_payload(output, ledger, branch, marker)
        _same(semantic(trainer.native.state_dict()), semantic(saved['trainer']['native_state']),
              'Completed boundary changed native state')
        trainer.load_state_dict(saved['trainer']); return receipt
    original = semantic(trainer.native.state_dict())
    validation = output/'branches'/branch/'validation'/f'step_{step:07d}'
    evidence = {'validation_report': _bound(output, validation/'report.json'),
                'validation_manifest': _bound(output, validation/'manifest.json')}
    source_report = _json(output/'source_validation.json'); reference = _json(output/'baselines.json')['reference']
    kwargs, diagnostics = _guard(report, source_report, reference)
    gate = {'manager': {'active': False, 'lambda': 0., 'reason': 'control'},
            'strict_guard': diagnostics, 'capability_eligible': kwargs['capability_eligible']}
    needs_tree = branch == 'feedback' or step == prepared['primary_endpoint']
    prior = None; tree_result = None
    if branch == 'feedback':
        begin = folder/f'step_{step:07d}_refresh_begin.json'
        if begin.exists():
            record, saved = _boundary_payload(output, ledger, branch, begin)
            _same(semantic(saved['trainer']['native_state']), original, 'Refresh start changed native learner')
            trainer.load_state_dict(saved['trainer']); prior = record['prior_manager_state']
            _same(prior, trainer.feedback.state_dict(), 'Saved closed-manager prior differs')
        else:
            prior = trainer.begin_feedback_refresh()
            start = folder/f'step_{step:07d}_refresh_begin.pt'
            if start.exists(): raise ValueError('Unacknowledged refresh-start checkpoint requires diagnosis')
            atomic_torch_save(start, {'version': VERSION, 'branch': branch, 'cycle_id': prepared['cycle_id'],
                'predecessor_head': head, 'trainer': trainer.state_dict(), 'evidence': deepcopy(evidence),
                'new_training_steps': 0})
            write_json(begin, {'version': VERSION, 'branch': branch, 'predecessor_head': head,
                'checkpoint': _bound(output, start), 'evidence': deepcopy(evidence),
                'prior_manager_state': prior, 'new_training_steps': 0})
    if needs_tree:
        fit, bindings, collected = refresh(output, prepared, trainer, actor, prior, branch)
        for name in ('fit_result.json', 'fit_request.json', 'manifest.json', 'plan.json', 'auxiliary_budget.json'):
            evidence['tree_'+name.replace('.', '_')] = _bound(output, collected/name)
        tree_result = {'reliable': fit.get('reliable') is True, 'actor_sha256': actor.artifact_sha256,
            'actor_parameters_sha256': bindings['actor_parameters_sha256'],
            'cumulative_fit_step': trainer.feedback_clock, 'fit_result': _bound(output, collected/'fit_result.json'),
            'explanation_qualified': False}
        if branch == 'feedback':
            if fit.get('reliable') is True:
                gate = _install(trainer, fit, bindings, report, evidence['validation_report']['sha256'],
                                source_report, reference)
            else:
                trainer.fail_feedback_refresh(fit['fit_report'])
                gate = {'manager': {'active': False, 'lambda': 0., 'reason': 'unreliable_refreshed_program'},
                        'strict_guard': diagnostics, 'capability_eligible': False}
    if semantic(trainer.native.state_dict()) != original:
        raise ValueError('Validation or extraction mutated the native learner')
    path = folder/f'step_{step:07d}.pt'
    if path.exists(): raise ValueError('Unacknowledged boundary checkpoint requires diagnosis')
    payload = {'version': VERSION, 'branch': branch, 'cycle_id': prepared['cycle_id'], 'predecessor_head': head,
        'trainer': trainer.state_dict(), 'evidence': evidence, 'gate': gate, 'terminal_tree': tree_result,
        'new_training_steps': 0}
    atomic_torch_save(path, payload)
    receipt = {'version': VERSION, 'branch': branch, 'predecessor_head': head, 'checkpoint': _bound(output, path),
        'evidence': evidence, 'gate': gate, 'terminal_tree': tree_result, 'new_training_steps': 0}
    write_json(marker, receipt)
    return receipt


def _old_boundaries(output, prepared, branch, head):
    for step in prepared['validation_endpoints']:
        if step >= head['step']: break
        marker = output/'branches'/branch/'boundaries'/f'step_{step:07d}.json'
        if not marker.exists(): raise ValueError('Earlier fixed boundary is incomplete')
        receipt = _json(marker)
        if (receipt.get('version') != VERSION or receipt.get('branch') != branch
                or receipt.get('new_training_steps') != 0 or receipt['predecessor_head']['step'] != step):
            raise ValueError('Earlier boundary identity differs')
        _check(output, receipt['checkpoint'])
        for item in receipt['evidence'].values(): _check(output, item)


def advance(output, *, until=None, allow_test_fixture=False):
    output = Path(output).expanduser().resolve(); raw = _json(output/'prepared.json')
    until = raw['primary_endpoint'] if until is None else until
    if type(until) is not int or until not in (raw['probe_endpoint'], *raw['validation_endpoints']):
        raise ValueError('Only a registered probe or validation endpoint can run')
    if raw.get('version') != VERSION or raw.get('test_fixture') is not allow_test_fixture:
        raise ValueError('Cycle identity or fixture differs')
    ledger = CycleBudget(output, raw['identity'])
    with ledger.lease():
        if any(op['status'] in ('pending', 'abandoned') for op in ledger.read()['operations'].values()):
            raise ValueError('Unconfirmed cycle operation requires diagnosis; no automatic replay')
        auxiliary_accounting(output, raw)
        p, protocol, scenes, source = read_prepared(output, allow_test_fixture=allow_test_fixture)
        trainers = {}; reports = {}; boundaries = {}
        for branch in BRANCHES:
            trainer = _trainer(p, protocol, source, branch)
            head, payload = effective_checkpoint(output, ledger, branch)
            _old_boundaries(output, p, branch, head)
            trainer.load_state_dict(payload['trainer'])
            if trainer.joint_steps != head['step'] or trainer.joint_steps > until:
                raise ValueError('Confirmed PPO counter or requested endpoint differs')
            trainers[branch] = trainer
        for target in sorted(set((p['probe_endpoint'], *p['validation_endpoints']))):
            if target > until: break
            for branch, trainer in trainers.items():
                if trainer.joint_steps > target: continue
                train_to(output, trainer, ledger, branch, target)
                if target in p['validation_endpoints']:
                    marker = output/'branches'/branch/'boundaries'/f'step_{target:07d}.json'
                    if marker.exists():
                        # Completed boundaries recheck saved raw records and ACKs;
                        # neither export/inference nor sampling is invoked again.
                        report = _report_record(output, branch, target, p, protocol, scenes,
                                                ledger.read(), allow_test_fixture)
                        receipt, saved = _boundary_payload(output, ledger, branch, marker)
                        _same(semantic(saved['trainer']), semantic(trainer.state_dict()), 'Completed boundary state differs')
                    else:
                        report, actor = evaluate_boundary(output, trainer, ledger, branch, scenes)
                        verified = _report_record(output, branch, target, p, protocol, scenes,
                                                  ledger.read(), allow_test_fixture)
                        _same(report, verified, 'Live and acknowledged validation reports differ')
                        receipt = finish_boundary(output, p, trainer, ledger, branch, report, actor)
                    reports[branch+':'+str(target)] = report; boundaries[branch+':'+str(target)] = receipt
                    print(json.dumps({'event': 'feedback_cycle_validation_complete', 'branch': branch, 'steps': target,
                        'primary_value': report['primary_value'], 'strict_guard': receipt['gate']['capability_eligible']}), flush=True)
        counts = {branch: {'ppo_steps': t.joint_steps, 'optimizer_updates': t.optimizer_updates,
            'actor_optimizer_steps': t.actor_optimizer_steps, 'critic_optimizer_steps': t.critic_optimizer_steps}
            for branch, t in trainers.items()}
        if counts['control'] != counts['feedback']:
            raise ValueError('Conditions did not receive equal PPO and neural update counts')
        terminal = {}
        if until == p['primary_endpoint']:
            for branch in BRANCHES:
                path = output/'branches'/branch/'boundaries'/f'step_{until:07d}.json'
                item = _json(path)['terminal_tree']
                if item is None: raise ValueError('Both terminal same-Actor tree attempts must complete')
                terminal[branch] = item
        result = {'version': VERSION, 'status': 'fixed_endpoint_completed' if until == p['primary_endpoint'] else 'boundary_completed',
            'until': until, 'cycle_id': p['cycle_id'], 'source_cumulative_steps': p['source_cumulative_steps'],
            'counts': counts, 'ledger': ledger.read(), 'auxiliary': auxiliary_accounting(output, p),
            'terminal_trees': terminal, 'formal_ready': False, 'explanation_qualified': False,
            'used_final_test': False, 'program_runtime_controller': False, 'total_interaction_equal': False,
            'initial_tree_budget_is_separate': True,
            'reports': {k: {x: y for x, y in v.items() if x not in ('rows', 'artifacts')} for k, v in reports.items()}}
        completion = output/f'completion_{until:07d}.json'
        if completion.exists():
            # Reading a previously completed run must not silently replace it.
            existing = _json(completion)
            for key in ('version', 'status', 'until', 'cycle_id', 'source_cumulative_steps', 'counts', 'ledger',
                        'auxiliary', 'terminal_trees', 'formal_ready', 'explanation_qualified', 'used_final_test'):
                _same(existing[key], result[key], 'Completed cycle evidence changed: '+key)
            return existing
        write_json(completion, result); write_json(output/'progress.json', result, replace=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--run', action='store_true')
    parser.add_argument('--source-run'); parser.add_argument('--expected-source-completion-sha256')
    parser.add_argument('--initial-tree'); parser.add_argument('--expected-initial-tree-manifest-sha256')
    parser.add_argument('--cycle-id'); parser.add_argument('--until', type=int)
    parser.add_argument('--device', choices=('mps',), default='mps')
    args = parser.parse_args(argv)
    if args.prepare:
        required = ('source_run', 'expected_source_completion_sha256', 'initial_tree', 'expected_initial_tree_manifest_sha256')
        if any(getattr(args, key) is None for key in required): parser.error('Explicit source and initial tree byte anchors required')
        result = prepare(args.output, source_run=args.source_run,
            expected_source_completion_sha256=args.expected_source_completion_sha256, initial_tree=args.initial_tree,
            expected_initial_tree_manifest_sha256=args.expected_initial_tree_manifest_sha256,
            cycle_id=args.cycle_id, device=args.device)
    else:
        import torch
        torch.set_num_threads(1)
        result = advance(args.output, until=args.until)
    print(json.dumps({k: v for k, v in result.items() if k not in ('ledger', 'reports')}), flush=True)


if __name__ == '__main__': main()
