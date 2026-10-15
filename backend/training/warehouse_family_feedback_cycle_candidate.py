"""Fixed 500k feedback candidate capsule, not a release or a new evaluator.

freeze authenticates every input of an externally anchored, completed result
which already contains its three registered CPU checkpoint audits. It does not
repeat those audits. read_frozen authenticates the capsule and current code;
it deliberately does not revisit the original trajectory/ancestor inventory.
load_candidate additionally uses the unchanged genuine runtime/renderer.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import numpy as np

from backend import warehouse_runtime_family as registry
from backend import warehouse_family_explanation as explanation
from backend.warehouse_shutdown_runtime import ShutdownRuntime, RUNTIME_VERSION, runtime_sources
from . import warehouse_family_feedback_cycle_result as result_api
from .warehouse_native_common import ROOT, canonical, digest, file_hash
from .warehouse_native_revision_provenance import _json

VERSION = 'warehouse-family-fixed-feedback-cycle-candidate.v1'
ENDPOINT = 500000
SOURCE_CLOCK = 3380000
ARTIFACTS = frozenset(('actor.npz', 'protocol.json', 'scenarios.json', 'program.json',
    'actor_bindings.json', 'selection.json', 'validation.json',
    'terminal_tree_verification.json', 'source_proof.json'))
FLAGS = dict(qualification_evaluated=False, release_ready=False, formal_ready=False,
             explanation_qualified=False, participant_enabled=False)
LEVELS = {'freeze': 'all externally anchored result input bytes, once; no producer rerun or PT decode',
    'read': 'externally anchored capsule and current runtime/renderer/adapter code only; no ancestor revalidation',
    'checkpoint': 'existing three registered CPU audit records; no new gradient calculation or state replay',
    'tree': 'saved candidate arithmetic and same-Actor bindings; no new fitting or independent explanation test'}


def _require(value, message):
    if not value: raise ValueError(message)


def _same(a, b, message):
    _require(canonical(a) == canonical(b), message)


def _sha(value):
    _require(type(value) is str and re.fullmatch('[a-f0-9]{64}', value), 'Explicit external SHA256 required')
    return value


def _path(path):
    path = Path(path).expanduser().absolute()
    _require(path.resolve() == path, 'Canonical nonsymlink path required')
    return path


def _raw(path, expected=None, size=None):
    path = _path(path)
    with path.open('rb') as stream: raw = stream.read()
    if expected is not None: _same(sha256(raw).hexdigest(), _sha(expected), 'Bound bytes changed')
    if size is not None: _same(len(raw), size, 'Bound byte size changed')
    return raw


def _binding(raw): return {'sha256': sha256(raw).hexdigest(), 'size': len(raw)}
def _encode(value): return (canonical(value)+'\n').encode()


def sources():
    records = {**result_api.sources(), **explanation.explanation_sources()}
    # These are code hashes, not the historical trajectory/checkpoint inventory.
    for path in (Path(__file__), Path(result_api.__file__), Path(result_api.source.__file__),
                 ROOT/'backend/training/warehouse_native_revision_provenance.py'):
        records[str(path.relative_to(ROOT))] = file_hash(path)
    return records


def _all_inputs(records):
    """Stream each bound original file once, retaining only stat fingerprints."""
    _require(type(records) is dict and bool(records), 'Complete result input inventory required')
    states = {}; total = 0
    for name, binding in records.items():
        path = _path(name); _require(path.is_absolute() and str(path) == name, 'Original absolute identity required')
        _require(set(binding) == {'sha256', 'size'} and type(binding['size']) is int and binding['size'] >= 0,
                 'Original input binding differs')
        _sha(binding['sha256']); hasher = sha256(); count = 0
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(1024*1024), b''): hasher.update(block); count += len(block)
            after = os.fstat(stream.fileno())
        _same((before.st_ino, before.st_size, before.st_mtime_ns),
              (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed while hashing')
        _same({'sha256': hasher.hexdigest(), 'size': count}, binding, 'Original result input changed')
        states[name] = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns); total += count
    return states, total


def _unchanged(states):
    for name, expected in states.items():
        s = _path(name).stat()
        _same((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns), expected, 'Input changed during freeze')


def _projection(result):
    keys = ('version', 'runner_version', 'status', 'cycle_id', 'primary_endpoint', 'source_cumulative_steps',
        'final_cumulative_steps_per_arm', 'primary', 'completed_learning_counts', 'budget',
        'registered_checkpoint_audits', 'checkpoint_audits', 'first_positive_feedback_operation',
        'same_source_initialization', 'completion_sha256', 'test_fixture', 'source_selection',
        'total_interaction_equal', 'initial_tree_budget_is_separate', 'release_ready', 'formal_ready',
        'explanation_qualified', 'used_final_test', 'inspection_counts')
    value = {k: deepcopy(result[k]) for k in keys}
    value['training'] = {arm: {k: result['training'][arm][k] for k in
        ('joint_steps', 'updates', 'raw_neural_overrides', 'positive_lambda_joint_steps')}
        for arm in result_api.BRANCHES}
    return value


def _gate(result, completion, prepared, markers, fixture):
    _require(type(fixture) is bool and result.get('test_fixture') is fixture, 'Explicit matching fixture scope required')
    for key, expected in {'version': result_api.VERSION, 'runner_version': result_api.runner.VERSION,
            'status': 'fixed_endpoint_complete', 'primary_endpoint': ENDPOINT,
            'source_cumulative_steps': SOURCE_CLOCK, 'final_cumulative_steps_per_arm': SOURCE_CLOCK+ENDPOINT,
            'source_selection': None, 'total_interaction_equal': False, 'initial_tree_budget_is_separate': True,
            'release_ready': False, 'formal_ready': False, 'explanation_qualified': False, 'used_final_test': False}.items():
        _same(result.get(key), expected, 'Fixed feedback result differs: '+key)
    _require(type(result['primary_endpoint']) is int, 'Integer fixed endpoint required')
    for value in (prepared, completion):
        _same(value['version'], result_api.runner.VERSION, 'Original runner producer differs')
        _same(value['cycle_id'], result['cycle_id'], 'Cycle identity differs')
        _same(value['source_cumulative_steps'], SOURCE_CLOCK, 'Original source clock differs')
    _same(prepared['primary_endpoint'], ENDPOINT, 'Only registered final endpoint can be selected')
    _same(prepared['validation_endpoints'], list(range(50000, ENDPOINT+1, 50000)), 'Registered validation endpoints differ')
    _same(prepared['test_fixture'], fixture, 'Prepared fixture scope differs')
    _same(completion['status'], 'fixed_endpoint_completed', 'Incomplete final boundary')
    _same(completion['until'], ENDPOINT, 'Completion is not final')
    _same(completion['counts'], result['completed_learning_counts'], 'Actual completed counts differ')
    _same(completion['counts']['control'], completion['counts']['feedback'], 'PPO/dual Adam counts differ')
    for arm in result_api.BRANCHES:
        training = result['training'][arm]
        _same(training['joint_steps'], ENDPOINT, 'Incomplete paired PPO')
        _same(training['raw_neural_overrides'], 0, 'NN actions were overridden')
        _same(training['updates'], completion['counts'][arm]['optimizer_updates'], 'PPO update count differs')
        _same(completion['counts'][arm]['ppo_steps'], ENDPOINT, 'Completed PPO cap differs')
        marker = markers[arm]
        _require(marker['version'] == result_api.runner.VERSION and marker['branch'] == arm
            and marker['new_training_steps'] == 0, 'Wrong final boundary producer')
        _same(marker['checkpoint']['path'], f'branches/{arm}/boundaries/step_{ENDPOINT:07d}.pt', 'Final effective checkpoint required')
        _same(marker['terminal_tree'], completion['terminal_trees'][arm], 'Final tree completion differs')
    _require(result['training']['feedback']['positive_lambda_joint_steps'] > 0, 'No actual positive feedback updates')
    primary = result['primary']['feedback']; guard = primary['strict_development_guard']
    _require(primary['full_capability']['eligible'] is True and primary['warmup_capability']['eligible'] is True
        and guard['capability_eligible'] is True, 'Fixed feedback endpoint failed development gates; no fallback')
    _require(guard['diagnostics']['full'] is True and guard['diagnostics']['warmup'] is True
        and all(x['passed'] is True for x in guard['diagnostics']['nn_retention'].values())
        and set(guard['diagnostics']['nn_retention']) == set(result_api.evaluation.PARTNERS), 'Source NN retention failed')
    _same(markers['feedback']['gate']['strict_guard'], guard['diagnostics'], 'Boundary guard differs')
    _require(markers['feedback']['gate']['capability_eligible'] is True
        and primary['terminal_tree']['reliable'] is True, 'Reliable final same-Actor tree required')
    _same(primary['terminal_tree']['actor_binding'], markers['feedback']['terminal_tree'], 'Terminal tree summary differs')
    registered = result['registered_checkpoint_audits']; audits = result['checkpoint_audits']
    _require(len(registered) == len(audits) == 3 and result['inspection_counts']['checkpoint_decodes'] == 3,
             'All three previously performed checkpoint audits are required')
    _require(len({x['checkpoint']['path'] for x in registered}) == 3, 'Checkpoint audit identities must be distinct')
    for arm in result_api.BRANCHES:
        matches = [a for a in audits if a['kind'] == 'final_effective_boundary' and a['branch'] == arm]
        _require(len(matches) == 1, 'Missing terminal checkpoint audit')
        audit = matches[0]
        _require(audit['all_six_export_arrays_equal'] is True and audit['complete_state_fields_present'] is True
            and set(audit['dual_adam_absolute_steps']) == {'actor', 'critic'}
            and all(type(v) is int and v > 0 for v in audit['dual_adam_absolute_steps'].values()), 'Actual terminal export/dual Adam audit required')
        _same(audit['checkpoint'], markers[arm]['checkpoint'], 'Terminal audit used another checkpoint')
    positive = [a for a in audits if a['kind'] == 'first_positive_lambda_update' and a['branch'] == 'feedback']
    _require(len(positive) == 1 and positive[0]['nonzero_feedback_gradient_observed'] is True,
             'Actual nonzero committed feedback gradient audit required')
    _same(positive[0]['checkpoint'], result['first_positive_feedback_operation']['checkpoint'], 'First positive audit binding differs')
    metric = result_api._positive_metrics(positive[0]['metrics'], result['first_positive_feedback_operation']['lambda'])
    _require(metric['nonzero_feedback_gradient_observed'], 'Saved gradient was zero')
    _same([{k: a[k] for k in ('branch', 'kind', 'checkpoint')} for a in audits], registered, 'Registered audits differ')
    initial = result['same_source_initialization']
    _require(initial['identical_native_learning_state'] is True and initial['new_environment_steps'] == 0
        and initial['program_runtime_controller'] is False, 'Same complete source state required')


def _actor(raw, protocol, bindings, scenes, fixture):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        fields = result_api.source.FIELDS
        _require(len(archive.files) == 7 and set(archive.files) == set(fields)|{'metadata_json'}, 'Exact six-array export required')
        metadata = _json(str(archive['metadata_json'].item()))
        weights = {name: archive[name] for name in fields}
    parameter_sha = explanation.actor_parameter_sha256(SimpleNamespace(weights=weights))
    for key, expected in bindings.items():
        actual = sha256(raw).hexdigest() if key == 'actor_sha256' else metadata.get(key)
        _same(actual, expected, 'Original Actor metadata binding differs: '+key)
    expected = {'experiment_version': result_api.trainer.VERSION, 'joint_steps': ENDPOINT, 'feedback_branch': 'feedback',
        'feedback_enabled': True, 'branch': 'own_credit', 'shutdown_arm': 'beta1', 'own_shutdown_beta': 1.,
        'obs_dim': 197, 'state_dim': 354, 'test_fixture': fixture, 'protocol_sha256': digest(protocol),
        'scenario_manifest_sha256': digest(scenes), 'actor_parameters_sha256': parameter_sha,
        'action_masks': False, 'runtime_action_override': False}
    for k, v in expected.items(): _same(metadata.get(k), v, 'Actual feedback Actor identity differs: '+k)
    _same(metadata['source_counters']['joint_steps'], SOURCE_CLOCK, 'Actor inherited clock differs')
    _same(metadata['source_lineage'], protocol['source_lineage'], 'Actor ancestry metadata differs')
    _require(protocol['version'] == result_api.trainer.PROTOCOL_VERSION
        and protocol['native_protocol']['version'] == result_api.trainer.native.PROTOCOL_VERSION, 'Original nested producer required')
    config = result_api._configuration(scenes, fixture)
    features = list(result_api.evaluation.physical.observation_names(config))+list(result_api.evaluation.HISTORY_FEATURE_NAMES)
    _same(metadata['feature_names'], features, 'Ordered observed197 features differ')
    signature = digest({'version': RUNTIME_VERSION, 'actor_sha256': sha256(raw).hexdigest(),
        'protocol_sha256': digest(protocol), 'actor_metadata_sha256': digest(metadata),
        'configuration': scenes['configuration'], 'sources': runtime_sources()})
    return metadata, parameter_sha, signature


def _sampling(receipt, row_counts, fixture):
    _require(receipt['version'] == result_api.runner.primitive.VERSION
        and receipt['parent_driver'] == result_api.runner.VERSION and receipt['status'] == 'completed',
        'Original terminal collection producer/completion differs')
    actual = reserved = 0; rows = {'train': 0, 'selection': 0}
    for entry in receipt['episode_acks']:
        _require(entry['status'] == 'completed' and type(entry['actual_joint_steps']) is int
            and type(entry['horizon']) is int and 0 < entry['actual_joint_steps'] <= entry['horizon'] <= 120,
            'Terminal collection contains an unconfirmed or invalid episode')
        _require(entry['pool'] in rows and type(entry['neural_rows']) is int and entry['neural_rows'] > 0,
                 'Terminal collected row counts differ')
        actual += entry['actual_joint_steps']; reserved += entry['horizon']; rows[entry['pool']] += entry['neural_rows']
    _same(rows, row_counts, 'Fit row totals differ from acknowledged collection')
    _same([reserved, receipt['reserved_auxiliary_steps'], receipt['allocation']['reserved_joint_steps'],
           receipt['allocation']['cap'], receipt['maximum_auxiliary_steps']], [reserved]*5, 'Terminal permanent auxiliary reservation differs')
    _same(actual, receipt['actual_auxiliary_steps'], 'Terminal actual auxiliary steps differ')
    if not fixture: _same((len(receipt['episode_acks']), reserved), (896, 107520), 'Original fixed terminal extraction matrix differs')
    return {'episodes': len(receipt['episode_acks']), 'actual_auxiliary_steps': actual,
        'reserved_auxiliary_steps': reserved, 'fit_result_sha256': receipt['fit_result_binding']['sha256']}


def _verify(documents, actor_raw, fixture):
    proof = documents['source_proof.json']; r = proof['result_projection']
    _require(proof['version'] == VERSION and proof['test_fixture'] is fixture, 'Candidate proof producer differs')
    _same(proof['verification_levels'], LEVELS, 'Source verification scope differs')
    verified = proof['source_inventory_verification']
    _require(verified['all_input_bytes_verified'] is True and type(verified['files']) is int
        and verified['files'] > 0 and type(verified['bytes']) is int and verified['bytes'] > 0,
        'Complete original source byte verification receipt required')
    for key in ('inventory_sha256', 'result_sources_sha256'): _sha(verified[key])
    for key in ('new_checkpoint_decodes', 'new_neural_forwards', 'new_environment_steps', 'new_fits'):
        _same(verified[key], 0, 'Freeze cannot create execution evidence')
    _gate(r, proof['completion_projection'], proof['prepared_projection'], proof['terminal_markers'], fixture)
    report = documents['validation.json']; bindings = documents['actor_bindings.json']; protocol = documents['protocol.json']
    _same(report['actor_bindings'], bindings, 'Validation and exported Actor bindings differ')
    _same({k: v for k, v in r['primary']['feedback'].items() if k not in ('strict_development_guard', 'terminal_tree')},
          result_api.previous._facts(report), 'Original fixed feedback report facts differ')
    guard = result_api.source.gate_values(report, proof['source_validation'], team_reference=proof['team_reference'])
    _same(guard, r['primary']['feedback']['strict_development_guard'], 'Recomputed strict retention guard differs')
    metadata, parameter_sha, signature = _actor(actor_raw, protocol, bindings, documents['scenarios.json'], fixture)
    _same(metadata['cycle_id'], r['cycle_id'], 'Actual Actor belongs to another cycle')
    source_checkpoint_sha = metadata.get('source_checkpoint_sha256')
    _sha(source_checkpoint_sha)
    _same(source_checkpoint_sha, proof['prepared_projection'].get('source_checkpoint_sha256'),
          'Actual Actor starting checkpoint differs from prepared source')
    _same(source_checkpoint_sha, protocol['native_protocol']['source']['checkpoint_sha256'],
          'Actual Actor starting checkpoint differs from native protocol')
    tree = documents['terminal_tree_verification.json']; fit = tree['fit']; plan = tree['fit_plan']
    _same(plan['actor_bindings'], bindings, 'Terminal collection Actor differs')
    _same(plan['cumulative_fit_step'], SOURCE_CLOCK+ENDPOINT, 'Terminal fit clock differs')
    saved = result_api._fit_record(fit, plan, metadata['feature_names'], tree['row_counts'])
    _require(saved['reliable'] is True, 'Final tree reliability failed')
    auxiliary = _sampling(tree['sampling_receipt'], tree['row_counts'], fixture)
    terminal_summary = r['primary']['feedback']['terminal_tree']
    _same({k: terminal_summary[k] for k in auxiliary}, auxiliary, 'Original terminal auxiliary/fit evidence differs')
    _same({k: v for k, v in terminal_summary.items() if k not in {*auxiliary, 'actor_binding'}}, saved,
          'Saved terminal tree arithmetic differs')
    _same(documents['program.json'], fit['program'], 'Program is not the final fit object')
    terminal = proof['terminal_markers']['feedback']['terminal_tree']
    _same(tree['sampling_receipt']['fit_result_binding'], terminal['fit_result'], 'Terminal fit byte receipt differs')
    _same(proof['program_origin']['sha256'], auxiliary['fit_result_sha256'], 'Program object came from another original fit')
    _same(terminal['actor_sha256'], bindings['actor_sha256'], 'Terminal tree used another Actor')
    _same(terminal['actor_parameters_sha256'], parameter_sha, 'Terminal tree used other six arrays')
    _same(terminal['cumulative_fit_step'], SOURCE_CLOCK+ENDPOINT, 'Terminal receipt clock differs')
    audit = next(a for a in r['checkpoint_audits'] if a['kind'] == 'final_effective_boundary' and a['branch'] == 'feedback')
    _same(audit['actor_parameters_sha256'], parameter_sha, 'Final CPU checkpoint parameters differ')
    chosen = {'version': VERSION, 'selection_rule': 'registered_feedback_500000_only_all_development_guards_and_terminal_tree',
        'selected_step': ENDPOINT, 'selected_cumulative_step': SOURCE_CLOCK+ENDPOINT,
        'selection_split': 'validation', 'final_test_used': False, 'frozen': True,
        'source_checkpoint_sha256': source_checkpoint_sha,
        'cycle_id': r['cycle_id'], 'branch': 'own_credit', 'feedback_branch': 'feedback', 'shutdown_arm': 'beta1',
        'own_shutdown_beta': 1., 'actor_sha256': bindings['actor_sha256'], 'actor_parameters_sha256': parameter_sha,
        'protocol_sha256': digest(protocol), 'scenario_manifest_sha256': digest(documents['scenarios.json']),
        'runtime_signature': signature, 'program_sha256': digest(fit['program']),
        'result_sha256': proof['result_binding']['sha256'], 'completion_sha256': r['completion_sha256'],
        'checkpoint': audit['checkpoint'], 'test_fixture': fixture, **FLAGS}
    if 'selection.json' in documents: _same(documents['selection.json'], chosen, 'Selection capsule differs')
    return chosen


def freeze(run_root, *, result_path, expected_result_sha256, expected_completion_sha256,
           output, allow_test_fixture=False):
    """Authenticate a previously audited result. Never call compare_fixed/loaders."""
    _sha(expected_result_sha256); _sha(expected_completion_sha256)
    root = _path(run_root); target = _path(output); result_path = _path(result_path)
    _require(not target.exists(), 'Output must be new')
    result_raw = _raw(result_path, expected_result_sha256); result = _json(result_raw)
    _same(result['completion_sha256'], expected_completion_sha256, 'Result/completion external anchors differ')
    _same(result['result_sources'], result_api.sources(), 'Original result verification code differs')
    inventory = result['input_bindings']; closure = sources()
    for name in inventory:
        path = _path(name)
        _require(target != path and target not in path.parents and path.parent not in target.parents,
                 'Output overlaps original source records')
    _require(root not in target.parents and target != root and target != result_path.parent,
             'Independent new candidate output required')
    def original(relative):
        path = root/relative; item = inventory.get(str(path))
        _require(item is not None, 'Required evidence missing from original result inventory: '+relative)
        return _raw(path, **{'expected': item['sha256'], 'size': item['size']})
    p = _json(original('prepared.json')); c = _json(original(f'completion_{ENDPOINT:07d}.json'))
    _same(sha256(original(f'completion_{ENDPOINT:07d}.json')).hexdigest(), expected_completion_sha256, 'Completion bytes differ')
    markers = {arm: _json(original(f'branches/{arm}/boundaries/step_{ENDPOINT:07d}.json')) for arm in result_api.BRANCHES}
    _gate(result, c, p, markers, allow_test_fixture)
    raw = {name: original(name) for name in ('protocol.json', 'scenarios.json')}
    raw['actor.npz'] = original(f'branches/feedback/actors/actor_{ENDPOINT:07d}.npz')
    raw['validation.json'] = original(f'branches/feedback/validation/step_{ENDPOINT:07d}/report.json')
    fit_prefix = f'branches/feedback/refresh/step_{ENDPOINT:07d}/'
    fit_raw = original(fit_prefix+'fit_result.json'); fit = _json(fit_raw); plan = _json(original(fit_prefix+'plan.json'))
    _same(_binding(fit_raw), {k: markers['feedback']['terminal_tree']['fit_result'][k] for k in ('sha256', 'size')}, 'Final fit byte binding differs')
    collection = _json(original(fit_prefix+'manifest.json')); allocation = _json(original(fit_prefix+'auxiliary_budget.json'))
    _same(collection['plan_sha256'], digest(plan), 'Terminal collection plan differs')
    _same(collection['fit_result'], {'path': 'fit_result.json', **_binding(fit_raw)}, 'Collection fit bytes differ')
    _same([entry['context'] for entry in collection['episodes']], plan['contexts'], 'Terminal collection order differs')
    sampling = {k: collection[k] for k in ('version', 'parent_driver', 'status', 'actual_auxiliary_steps', 'reserved_auxiliary_steps')}
    sampling.update(allocation=allocation, maximum_auxiliary_steps=plan['maximum_auxiliary_steps'],
        fit_result_binding=markers['feedback']['terminal_tree']['fit_result'],
        manifest_binding={'path': str(root/fit_prefix/'manifest.json'), **inventory[str(root/fit_prefix/'manifest.json')]},
        episode_acks=[{**{k: entry[k] for k in ('status', 'actual_joint_steps', 'neural_rows')},
            **{k: entry['context'][k] for k in ('pool', 'horizon')}} for entry in collection['episodes']])
    result_api._contract(p, _json(raw['protocol.json']), _json(raw['scenarios.json']), c, allow_test_fixture)
    proof = {'version': VERSION, 'test_fixture': allow_test_fixture, 'result_binding': {'path': str(result_path), **_binding(result_raw)},
        'result_projection': _projection(result), 'prepared_projection': {k: p[k] for k in
            ('version', 'cycle_id', 'source_cumulative_steps', 'source_checkpoint_sha256',
             'primary_endpoint', 'validation_endpoints', 'test_fixture')},
        'completion_projection': {k: c[k] for k in ('version', 'cycle_id', 'source_cumulative_steps', 'status', 'until', 'counts', 'terminal_trees')},
        'terminal_markers': markers, 'source_validation': _json(original('source_validation.json')),
        'team_reference': _json(original('baselines.json'))['reference'], 'verification_levels': LEVELS,
        'program_origin': {'input': str(root/fit_prefix/'fit_result.json'), **_binding(fit_raw), 'json_pointer': '/program',
            'encoding': 'canonical JSON object; original nested metadata and producer unchanged'}}
    raw['actor_bindings.json'] = _encode(_json(raw['validation.json'])['actor_bindings'])
    raw['program.json'] = canonical(fit['program']).encode()
    raw['terminal_tree_verification.json'] = _encode({'version': VERSION, 'fit': fit,
        'fit_plan': {k: plan[k] for k in ('test_fixture', 'actor_bindings', 'cumulative_fit_step', 'expanded_fit_contract')},
        'row_counts': {'train': fit['fit_report']['train_rows'], 'selection': fit['fit_report']['validation_rows']},
        'sampling_receipt': sampling})
    states, total = _all_inputs(inventory)
    proof['source_inventory_verification'] = {'files': len(states), 'bytes': total, 'inventory_sha256': digest(inventory),
        'result_sources_sha256': digest(result['result_sources']), 'all_input_bytes_verified': True,
        'new_checkpoint_decodes': 0, 'new_neural_forwards': 0, 'new_environment_steps': 0, 'new_fits': 0}
    raw['source_proof.json'] = _encode(proof)
    documents = {k: _json(v) for k, v in raw.items() if k.endswith('.json')}
    raw['selection.json'] = _encode(_verify(documents, raw['actor.npz'], allow_test_fixture))
    manifest = {'version': VERSION, 'status': 'frozen_for_independent_acceptance', 'test_fixture': allow_test_fixture,
        'source_hashes': closure, 'artifacts': {k: {'path': k, **_binding(v)} for k, v in raw.items()},
        'result_sha256': expected_result_sha256, 'completion_sha256': expected_completion_sha256,
        'verification_levels': LEVELS, **FLAGS}
    # Validate in private storage before the success marker is published.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.feedback-candidate-', dir=target.parent) as temporary:
        staging = Path(temporary).resolve()
        for name, value in raw.items(): result_api.runner.write_bytes(staging/name, value)
        result_api.runner.write_json(staging/'manifest.json', manifest)
        read_frozen(staging, expected_manifest_sha256=file_hash(staging/'manifest.json'), allow_test_fixture=allow_test_fixture)
        _unchanged(states); _same(sources(), closure, 'Candidate code changed during freeze')
        _raw(result_path, expected_result_sha256)
        target.mkdir(exist_ok=False)
        try:
            for name in sorted(raw): os.replace(staging/name, target/name)
            os.replace(staging/'manifest.json', target/'manifest.json')
            fd = os.open(target, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except BaseException:
            if (target/'manifest.json').exists(): (target/'manifest.json').unlink()
            raise
    return {'directory': str(target), 'manifest_sha256': file_hash(target/'manifest.json'), **manifest}


def read_frozen(root, *, expected_manifest_sha256, allow_test_fixture=False):
    """No ancestor reads, source admission, PT decode, runtime or Actor objects."""
    _sha(expected_manifest_sha256); root = _path(root)
    manifest = _json(_raw(root/'manifest.json', expected_manifest_sha256))
    _require(manifest['version'] == VERSION and manifest['status'] == 'frozen_for_independent_acceptance'
        and type(allow_test_fixture) is bool and manifest['test_fixture'] is allow_test_fixture, 'Candidate capsule version/scope differs')
    for key, value in FLAGS.items(): _same(manifest[key], value, 'Candidate cannot grant qualification')
    _same(manifest['source_hashes'], sources(), 'Current candidate/runtime code differs')
    _same(manifest['verification_levels'], LEVELS, 'Verification scope differs')
    _same(sorted(manifest['artifacts']), sorted(ARTIFACTS), 'Complete candidate artifact set required')
    raw = {}
    for name, entry in manifest['artifacts'].items():
        _same(entry['path'], name, 'Candidate artifact path differs')
        raw[name] = _raw(root/name, entry['sha256'], entry['size'])
    documents = {k: _json(v) for k, v in raw.items() if k.endswith('.json')}
    chosen = _verify(documents, raw['actor.npz'], allow_test_fixture)
    for k in ('result_sha256', 'completion_sha256'): _same(manifest[k], chosen[k], 'External evidence binding differs')
    return {'root': str(root), 'manifest': manifest, 'manifest_sha256': expected_manifest_sha256,
        'documents': documents, 'selection_raw': raw['selection.json'], 'selection': chosen,
        'verification_level': LEVELS['read'], **FLAGS}


@dataclass
class CandidateContext:
    runtime: object
    scenarios: dict
    explainer: object
    question_bank: object
    protocol: dict
    selection_raw: bytes
    candidate: dict
    closed: bool = False

    def close(self): self.closed = True


def load_candidate(root, *, expected_manifest_sha256, allow_test_fixture=False):
    saved = read_frozen(root, expected_manifest_sha256=expected_manifest_sha256, allow_test_fixture=allow_test_fixture)
    docs = saved['documents']; chosen = saved['selection']; root = Path(saved['root'])
    runtime = ShutdownRuntime(root/'actor.npz', protocol=docs['protocol.json'],
        expected_actor_sha256=chosen['actor_sha256'], expected_protocol_sha256=chosen['protocol_sha256'],
        expected_bindings=docs['actor_bindings.json'], allow_test_fixture=allow_test_fixture,
        config=result_api._configuration(docs['scenarios.json'], allow_test_fixture))
    registry.verify(runtime, allow_test_fixture=allow_test_fixture, expected_family='retained_beta197',
                    expected_signature=chosen['runtime_signature'])
    explainer = explanation.FamilyExplainer(root/'program.json', expected_program_sha256=chosen['program_sha256'],
        runtime=runtime, allow_test_fixture=allow_test_fixture)
    _require(explainer.eligible is False and explainer.participant_enabled is False, 'Component qualification changed')
    return CandidateContext(runtime, docs['scenarios.json'], explainer, None, docs['protocol.json'],
        saved['selection_raw'], {k: v for k, v in saved.items() if k not in ('documents', 'selection_raw')})
