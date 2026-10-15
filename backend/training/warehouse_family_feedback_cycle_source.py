"""Admit one actual native source and its reliable, dual-role trajectory tree.

The earlier held-out explanation failure remains a failure. This component
admits training evidence only; it never grants explanation or release status.
Completed readers load saved NumPy weights and recompute tree metrics, but do
not sample, fit, decode a checkpoint, or restore a learner. The caller owns the
subsequent genuine checkpoint load and the finite training budget.
"""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import re

import numpy as np

from . import warehouse_family_native_cycle_run as native_cycle
from . import warehouse_family_candidate_tree_run as candidate_tree
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_shutdown_result import _Inputs, _json_bytes, _same
from backend.warehouse_family_explanation import actor_parameter_sha256
from types import SimpleNamespace

VERSION = 'warehouse-family-feedback-cycle-source.v1'
SOURCE = ROOT/'output/warehouse_native/native_cycle_500k_20260909'
SOURCE_COMPLETION_SHA = 'baa1eedc1959e8ddf3400b9c4fb7066e35fb351e7fe98bc322f47f176190073a'
TREE = ROOT/'output/warehouse_native/native_cycle_500k_tree_20260909'
TREE_MANIFEST_SHA = '732fb570d0651161570aa626be2aa3b57f2a45201bce45d6bd1974f6904acdc6'
SOURCE_STEP = 500000
SOURCE_CLOCK = 3380000
PARTNERS = ('skilled', 'assertive', 'noisy')
FIELDS = ('0.bias', '0.weight', '2.bias', '2.weight', '4.bias', '4.weight')


def sources():
    result = candidate_tree.sources('native_cycle')
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Explicit lowercase external SHA256 required')
    return value


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Finite nonnegative recorded mean required')
    return float(value)


def _gate(report, name):
    value = report.get(name, {}); checks = value.get('checks', {})
    if (type(value.get('eligible')) is not bool or not checks or type(checks) is not dict
            or any(type(x) is not bool for x in checks.values())
            or value['eligible'] is not all(checks.values())):
        raise ValueError('Malformed or contradictory saved capability gate')
    return value['eligible']


def gate_values(report, source_report, *, team_reference):
    """Guard saved/recomputed reports; this function does not verify episodes.

    The NN retention reference is the fixed starting NN. The original program
    baseline separately supplies the manager's team reference; its existing
    ten-percent team-performance guard is not replaced here.
    """
    full, warmup = _gate(report, 'capability'), _gate(report, 'warmup_capability')
    retention = {}; current_team = []; reference_team = []
    for partner in PARTNERS:
        current = report['summary'][partner]; initial = source_report['summary'][partner]
        a = _number(current['mean_ai_deliveries']); b = _number(initial['mean_ai_deliveries'], positive=True)
        retention[partner] = {'current_nn_mean': a, 'source_nn_mean': b,
            'minimum_nn_mean': .9*b, 'ratio': a/b, 'passed': a >= .9*b}
        current_team.append(_number(current['mean_team_deliveries']))
        reference_team.append(_number(team_reference['summary'][partner]['mean_team_deliveries'], positive=True))
    values = {'validation_score': sum(current_team)/3, 'reference_score': sum(reference_team)/3,
        'capability_eligible': full and warmup and all(x['passed'] for x in retention.values())}
    return {**values, 'update_kwargs': dict(values), 'diagnostics': {'full': full, 'warmup': warmup,
        'nn_retention': retention, 'nn_reference': 'fixed_source_NN_validation_report',
        'team_reference': 'original_program_reference_report', 'team_drop_guard': 'existing_manager_unchanged'},
        'qualification_evaluated': False, 'explanation_qualified': False, 'release_ready': False}


def _role_scope(entries):
    counts = {p: {'robot_1': 0, 'robot_2': 0, 'episodes': 0} for p in ('train', 'selection')}
    for entry in entries:
        context = entry['context']; pool = context['pool']; role = context['program_role']
        steps = entry['actual_joint_steps']; rows = entry['neural_rows']
        if (pool not in counts or type(role) is not int or role not in (-1, 0, 1)
                or type(steps) is not int or not 0 < steps <= context['horizon']
                or type(rows) is not int or entry['status'] != 'completed'
                or rows != steps*(2 if role == -1 else 1)):
            raise ValueError('Actual acknowledged neural-role collection differs')
        counts[pool]['episodes'] += 1
        if role in (-1, 1): counts[pool]['robot_1'] += steps
        if role in (-1, 0): counts[pool]['robot_2'] += steps
    if any(v['robot_1'] <= 0 or v['robot_2'] <= 0 for v in counts.values()):
        raise ValueError('Both actual NN roles are required in both trajectory pools')
    return counts


def _records(saved, bundle, metadata, parameter_sha, role_scope):
    """Pure reciprocal identity checks after the genuine completed readers."""
    completion, report, prepared = saved['completion'], saved['report'], saved['prepared']
    plan, fit = bundle['plan'], bundle['fit_result']; binding = saved['actor_bindings']
    proof = saved['actor_parameter_receipt']; expanded = candidate_tree.collector.expanded
    if (saved.get('version') != native_cycle.VERSION or saved.get('source_kind') != 'native_cycle'
            or saved.get('completion_sha256') != SOURCE_COMPLETION_SHA
            or prepared.get('primary_endpoint') != SOURCE_STEP or prepared.get('validation_endpoints') != [SOURCE_STEP]
            or completion.get('until') != SOURCE_STEP or completion.get('status') != 'both_gates_ready'
            or completion.get('feedback_disabled_after_source') is not True or completion.get('feedback_lambda') != 0.
            or completion.get('used_final_test') is not False):
        raise ValueError('Only the actual completed fixed 500k native source is admitted')
    if not _gate(report, 'capability') or not _gate(report, 'warmup_capability'):
        raise ValueError('Source full AND warmup must pass')
    for name in ('capability', 'warmup_capability'):
        _same(completion[name], report[name], 'Completed source gate differs')
    if (binding.get('experiment_version') != native_cycle.native.VERSION or binding.get('joint_steps') != SOURCE_STEP
            or binding.get('branch') != 'own_credit' or binding.get('shutdown_arm') != 'beta1'
            or type(binding.get('own_shutdown_beta')) not in (int, float) or binding['own_shutdown_beta'] != 1.
            or 'feedback_branch' in binding or 'actor_parameters_sha256' in binding):
        raise ValueError('Actual native Actor metadata was relabeled')
    for key, value in binding.items():
        if key != 'actor_sha256': _same(metadata[key], value, 'Actual exported Actor binding differs')
    if (metadata.get('obs_dim') != 197 or metadata.get('state_dim') != 354
            or metadata.get('test_fixture') is not False or metadata.get('action_masks') is not False
            or metadata.get('runtime_action_override') is not False
            or len(metadata.get('feature_names', [])) != 197
            or metadata.get('actions') != ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']):
        raise ValueError('Actual observed197/354 five-action source differs')
    clock = metadata['source_counters']['joint_steps'] + metadata['joint_steps']
    if (clock != SOURCE_CLOCK or report.get('total_actor_training_joint_steps') != clock
            or plan.get('cumulative_fit_step') != clock or plan.get('endpoint') != SOURCE_STEP
            or plan.get('source') != str(SOURCE) or plan.get('source_kind') != 'native_cycle'
            or plan.get('source_version') != native_cycle.VERSION or plan.get('version') != candidate_tree.VERSION
            or bundle.get('manifest_sha256') != TREE_MANIFEST_SHA or plan.get('test_fixture') is not False):
        raise ValueError('Tree and actual source clock or original identity differ')
    expected = {**binding, 'actor_parameters_sha256': parameter_sha}
    _same(plan['actor_bindings'], expected, 'Tree is not from the same current Actor')
    _same(plan['source_anchors'], {'completion_sha256': SOURCE_COMPLETION_SHA, 'actor_sha256': binding['actor_sha256'],
        'actor_parameters_sha256': parameter_sha, 'validation_report_sha256': completion['report_sha256']}, 'Tree source anchors differ')
    _same(plan['actor_parameter_receipt'], proof, 'Tree copied another terminal tensor receipt')
    if (proof.get('all_six_arrays_equal') is not True or proof.get('actor_parameters_sha256') != parameter_sha
            or proof.get('actor_sha256') != binding['actor_sha256'] or proof.get('field_names') != list(FIELDS)
            or proof.get('checkpoint_sha256') != saved['checkpoint']['sha256']
            or proof.get('checkpoint_path') != saved['checkpoint']['path']):
        raise ValueError('Actual six-array export/checkpoint receipt differs')
    fr = fit['fit_report']; fb = fr['observed197_bindings']; manager = fit['manager_state']
    if (fit.get('version') != expanded.VERSION or fit.get('reliable') is not True
            or fit.get('test_fixture') is not False or fr.get('reliable') is not True
            or fr.get('intervention_direction_not_tested_here') is not True
            or fb.get('actor_sha256') != binding['actor_sha256'] or fb.get('actor_parameters_sha256') != parameter_sha
            or fb.get('cumulative_fit_step') != clock or fit.get('evidence_sha256') != digest({'binding': fb, 'fit_report': fr})
            or manager.get('reliable') is not True or manager.get('current_lambda') != 0
            or manager.get('last_step') != clock or manager.get('last_fit_step') != clock):
        raise ValueError('Only the original reliable trajectory fit may be installed')
    _same(manager['program'], fit['program'], 'Saved manager program differs')
    _same(bundle['program'], fit['program'], 'Standalone tree differs')
    _same(manager['feature_names'], metadata['feature_names'], 'Tree/NN feature order differs')
    verification = bundle['report']['verification']
    if (bundle['report'].get('reliable') is not True or verification.get('reliable') is not True
            or verification.get('verified_candidates') != 13 or fr.get('rows_removed') != 0):
        raise ValueError('Original complete thirteen-candidate reliability verification is required')
    _same(verification['selected'], fr['selected'], 'Recomputed selected training tree differs')
    by_role = fr['selection_metrics']['by_role']
    for pool, key in (('train', 'train_rows'), ('selection', 'validation_rows')):
        if sum(role_scope[pool][r] for r in ('robot_1', 'robot_2')) != fr[key]:
            raise ValueError('Actual dual-role collection counts differ from the fit')
    for key, role in (('0', 'robot_1'), ('1', 'robot_2')):
        _same(by_role[key]['rows'], role_scope['selection'][role], 'Saved selection role provenance differs')
    return expected


def admit(source, source_completion_sha, tree, tree_manifest_sha):
    """Production record admission for this fixed source; never loads a learner."""
    _sha(source_completion_sha); _sha(tree_manifest_sha)
    if (Path(source).expanduser().absolute() != SOURCE or Path(tree).expanduser().absolute() != TREE
            or source_completion_sha != SOURCE_COMPLETION_SHA or tree_manifest_sha != TREE_MANIFEST_SHA):
        raise ValueError('Only the registered 500k source and original trajectory tree are allowed')
    current_sources = sources()
    # source_records calls the actual completed reader once and returns the
    # exact descriptor consumed by native_cycle.load_source(expected=...).
    descriptor, saved = native_cycle.source_records(SOURCE, source_completion_sha, fixture=False)
    bundle = candidate_tree.read_completed(TREE, expected_manifest_sha256=tree_manifest_sha, require_reliable=True)
    reads = _Inputs(); plan = bundle['plan']
    manifest = reads.json(TREE, 'manifest.json', tree_manifest_sha)
    _same(reads.json(TREE, 'plan.json'), plan, 'Original tree plan changed')
    _same(reads.json(TREE, manifest['report']['path'], manifest['report']['sha256']), bundle['report'], 'Original tree report changed')
    for key in ('fit_result', 'program'):
        _same(_json_bytes(reads.bound(TREE, bundle['report'][key])), bundle[key], 'Original fitted artifact changed')
    values = {name: reads.json(TREE, name+'.json') for name in ('protocol', 'scenarios', 'pools')}
    for name, value in values.items(): _same(digest(value), plan[name+'_sha256'], 'Original tree input changed')
    _same(values['protocol'], saved['protocol'], 'Source/tree protocol differs')
    _same(values['scenarios'], saved['scenarios'], 'Source/tree scenarios differ')
    baselines = reads.json(SOURCE, 'baselines.json')
    raw = reads.raw(Path(saved['actor_path']).parent, Path(saved['actor_path']).name, saved['actor_bindings']['actor_sha256'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != 7 or set(archive.files) != set(FIELDS)|{'metadata_json'}:
            raise ValueError('Actual six-array Actor export required')
        metadata = json.loads(str(archive['metadata_json'].item()))
        parameter_sha = actor_parameter_sha256(SimpleNamespace(weights={k: archive[k].copy() for k in FIELDS}))
    collection = reads.json(TREE, 'collection/manifest.json')
    operation = reads.json(TREE, 'collection/plan.json')
    if (collection.get('status') != 'completed' or collection.get('plan_sha256') != digest(operation)
            or len(collection['episodes']) != len(operation['contexts'])):
        raise ValueError('Original collection is incomplete')
    role_scope = _role_scope(collection['episodes'])
    binding = _records(saved, bundle, metadata, parameter_sha, role_scope)
    # These original raw files were read/verified by the genuine completed tree
    # reader above. Carry their byte anchors forward without duplicating all raw
    # trajectory I/O or pretending to perform a second independent replay.
    inputs = deepcopy(saved['input_bindings'])
    for name, value in plan['input_bindings'].items():
        if name in inputs: _same(inputs[name], value, 'Source input anchors disagree')
        inputs[name] = deepcopy(value)
    for entry in collection['episodes']:
        for key in ('arrays', 'record'):
            item = entry[key]; path = Path(item['path'])
            if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe original episode path')
            _sha(item['sha256']); inputs[str(TREE/'collection'/path)] = {'sha256': item['sha256'], 'size': item['size']}
    for name, expected in plan['runtime_sources'].items():
        inputs[str(TREE/'source_snapshot'/name)] = {'sha256': expected, 'size': (TREE/'source_snapshot'/name).stat().st_size}
    reads.json(TREE, 'collection/auxiliary_budget.json')
    for key in ('fit_request', 'fit_result'): reads.bound(TREE/'collection', collection[key])
    for name, value in reads.records.items():
        if name in inputs: _same(inputs[name], value, 'Read source byte anchors disagree')
        inputs[name] = value
    _same(current_sources, sources(), 'Source adapter execution closure changed')
    reads.unchanged()
    gate = gate_values(saved['report'], saved['report'], team_reference=baselines['reference'])
    if not gate['capability_eligible']: raise ValueError('Fixed initial source gate failed')
    report_path = SOURCE/f'branches/beta1/validation/step_{SOURCE_STEP:07d}/report.json'
    report_binding = {'path': str(report_path), **saved['input_bindings'][str(report_path)]}
    _same(report_binding['sha256'], saved['completion']['report_sha256'], 'Source report byte anchor differs')
    return {'version': VERSION, 'saved_source': saved, 'tree_bundle': bundle, 'fit_result': bundle['fit_result'],
        'bindings': binding, 'actor_bindings': binding, 'protocol': saved['protocol'], 'scenes': saved['scenarios'],
        'pools': values['pools'], 'baselines': baselines, 'source_report': saved['report'], 'report': saved['report'],
        'report_sha256': saved['completion']['report_sha256'], 'source_report_path': str(report_path),
        'source_report_binding': report_binding, 'input_bindings': inputs, 'source_hashes': current_sources,
        'descriptor': descriptor, 'admission_binding': {'version': VERSION, 'source': str(SOURCE), 'source_completion_sha256': SOURCE_COMPLETION_SHA,
            'tree': str(TREE), 'tree_manifest_sha256': TREE_MANIFEST_SHA, 'source_step': SOURCE_STEP,
            'source_cumulative_steps': SOURCE_CLOCK, 'actor_bindings': binding,
            'training_tree_evidence_sha256': bundle['fit_result']['evidence_sha256'], 'test_fixture': False},
        'role_scope': role_scope, 'gate': gate, 'training_tree_reliable': True,
        'independent_explanation_passed': False, 'heldout_failure_preserved': True,
        'checkpoint_decodes': 0, 'new_environment_steps': 0, 'new_neural_forwards': 0, 'new_fits': 0,
        'scope': 'training-reliable dual-role trajectory evidence, not independent explanation qualification',
        'explanation_qualified': False, 'release_ready': False}
