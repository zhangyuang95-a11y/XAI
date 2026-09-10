"""Finite metadata-bank work using the frozen linear operation journal.

The workflow is new; LEDGER_VERSION/OP_VERSION and before/observe/after are the
actual unchanged generic linear record implementation. Only new question
scenes are restored by the genuine bank component. Phase entry/completion and
cache reads verify the full chain; each action checks only its new operation.
Neither a completed replay nor eight available questions grants qualification.
"""
from copy import deepcopy
from hashlib import sha256
import fcntl
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.training import warehouse_family_bank_linear_run as linear
from backend import warehouse_alignment_runtime as runtime_api
from backend.training import warehouse_family_question_pool as pools
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.warehouse_family_explanation import actor_parameter_sha256
from ui import warehouse_alignment_metadata_bank as bank_api
from ui.warehouse_family_bank_view import _items as validate_saved_items
from ui.warehouse_public_history_bank import PublicHistoryQuestionBank as PureBank

VERSION = 'warehouse-alignment-metadata-bank-workflow.v1'
PRODUCTION_REGISTRY_ROOT = ROOT/'output/warehouse_native/.warehouse_alignment_metadata_bank_consumption'
FIXTURE_REGISTRY_ROOT = None  # Tests must explicitly inject an isolated directory.
LEDGER_VERSION, OP_VERSION = linear.LEDGER_VERSION, linear.OP_VERSION
PHASES = linear.PHASES
INPUT_NAMES = ('actor.npz', 'protocol.json', 'actor_bindings.json', 'selection.json')
FLAGS = dict(eligible=False, release_ready=False, formal_ready=False,
             explanation_qualified=False, qualification_evaluated=False)
old_io = linear.old_io
_read, _json, _mkdir, put, _binding = linear._read, linear._json, linear._mkdir, linear.put, linear._binding
_absolute, _same = linear._absolute, linear._same
_initial_head, _reservation, _operation = linear._initial_head, linear._reservation, linear._operation
_observe = linear._observe
ACCOUNTING = {'module': 'backend/training/warehouse_family_bank_linear_run.py',
    'ledger_version': LEDGER_VERSION, 'operation_version': OP_VERSION,
    'reuse': 'original initial_head/reservation/operation/Budget before-observe-after; new workflow finish/content checks'}


def _sources(bank_sources):
    result = deepcopy(bank_sources)
    for path in (Path(__file__), Path(linear.__file__), Path(linear.original.__file__), Path(old_io.__file__),
                 ROOT/'backend/warehouse_family_explanation.py', ROOT/'ui/warehouse_family_bank_view.py',
                 ROOT/'ui/warehouse_family_bank.py',
                 ROOT/'backend/training/warehouse_native_revision_provenance.py'):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _current(records):
    for name, value in records.items():
        part = Path(name)
        if part.is_absolute() or '..' in part.parts or str(part) != name or file_hash(ROOT/part) != value:
            raise ValueError('Original execution source changed')


def _caps(count, horizon, limit, minimum, fixture):
    if (type(fixture) is not bool or any(type(x) is not int for x in (count, horizon, limit, minimum))
            or (not fixture and (count, horizon, limit, minimum) != (24, 120, 24, 1))
            or (fixture and (count, horizon, limit, minimum) != (4, 3, 1, 0))):
        raise ValueError('Only fixed production24/T24/min1 or explicit fixture4/h3/T1/min0 is supported')
    steps = count*(limit+3*(limit-minimum)); zero = count*(2*limit-minimum)
    return {phase: {'environment_steps': steps, 'zero_step_nn_queries': zero, 'nn_queries': steps+zero}
            for phase in PHASES}


def _actor_record(raw):
    """Six-array serialization check; no NumPy Actor or Torch object is built."""
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != 7 or len(set(archive.files)) != 7 or 'metadata_json' not in archive.files:
            raise ValueError('Exactly six Actor arrays and metadata are required')
        metadata = json.loads(str(archive['metadata_json'].item()))
        parameter = actor_parameter_sha256(SimpleNamespace(weights={k: archive[k] for k in archive.files if k != 'metadata_json'}))
    return metadata, parameter


def _contract(limit, minimum):
    return {'bank_content_version': bank_api.VERSION, 'trajectory_steps': limit, 'minimum_frame': minimum,
        'kinds': list(bank_api.KINDS), 'questions_per_kind': 4, 'independent_complete_replay': True,
        'controllers': 'same unmodified NN for both robots; human WAIT only within declared wait-three questions',
        'counterfactual_filter': bank_api.FILTER}


def _consumption(metadata, parameter, protocol, configuration, scenes, limit, minimum):
    return {'actor_parameters_sha256': parameter, 'feature_names_sha256': digest(metadata['feature_names']),
        'protocol_sha256': digest(protocol), 'configuration': configuration,
        'physical_question_initial_set_sha256': digest(sorted(s['fingerprint'] for s in scenes)),
        'bank_contract': _contract(limit, minimum)}


def _registry_root(fixture):
    production = _absolute(PRODUCTION_REGISTRY_ROOT)
    if not fixture: return production
    if FIXTURE_REGISTRY_ROOT is None: raise ValueError('Synthetic work requires an explicitly isolated claim registry')
    path = _absolute(FIXTURE_REGISTRY_ROOT)
    if path == production or path.is_relative_to(production) or production.is_relative_to(path):
        raise ValueError('Synthetic claims cannot use the production registry')
    return path


def _claim(root, p, output):
    """Exclusive directory is permanent consumption even if its write fails."""
    if not root.exists(): _mkdir(root)
    folder = root/p['identity']['consumption_key']; _mkdir(folder)
    value = {'version': VERSION, 'key': p['identity']['consumption_key'], 'identity_sha256': p['identity_sha256'],
        'output_identity': str(output), 'caps': p['identity']['caps'], 'test_fixture': p['identity']['test_fixture'],
        'permanent': True, 'refund_allowed': False, 'automatic_retry': False, **FLAGS}
    put(folder/'claim.json', value)
    return folder/'claim.json', _read(folder/'claim.json')


def _pool_binding(saved):
    return {'version': saved['version'], 'manifest_sha256': saved['manifest_sha256'],
        'pool_sha256': saved['pool_sha256'], 'source_scenario_manifest_sha256': saved['source_scenario_manifest_sha256'],
        'exclusion_binding': saved['exclusion_binding'], 'sources_sha256': digest(saved['sources']),
        'artifacts': {name: _binding(raw) for name, raw in saved['blobs'].items()}}


def prepare(output, *, runtime, frozen_selection, pool_root, expected_pool_manifest_sha256,
            trajectory_steps=24, minimum_frame=1, allow_test_fixture=False):
    output, pool_root = _absolute(output), _absolute(pool_root)
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    claim_root = _registry_root(allow_test_fixture)
    for protected in (pool_root, runtime._actor_path.parent, claim_root):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('New workflow must be separate from all inputs and the permanent registry')
    c = bank_api._context(runtime, pool_root, expected_pool_manifest_sha256, trajectory_steps, minimum_frame, allow_test_fixture)
    identity, saved = c['identity'], c['registered']
    if identity['family'] != runtime_api.FAMILY: raise ValueError('Genuine AlignmentRuntime required')
    old_io._selection(frozen_selection, runtime)
    for key, value in (('runtime_family', identity['family']), ('runtime_signature', runtime.signature),
                       ('shutdown_arm', runtime.actor.metadata['shutdown_arm']), ('own_shutdown_beta', runtime.actor.metadata['own_shutdown_beta'])):
        if key in frozen_selection: _same(frozen_selection[key], value, 'Frozen selection runtime differs')
    caps = _caps(len(c['scenes']), runtime.config.horizon, trajectory_steps, minimum_frame, allow_test_fixture)
    raw = _read(runtime._actor_path); metadata, parameter = _actor_record(raw)
    _same(metadata, runtime.actor.metadata, 'Actual exported metadata differs')
    _same(sha256(raw).hexdigest(), runtime.actor_sha256, 'Actual Actor bytes differ')
    _same(parameter, actor_parameter_sha256(runtime.actor), 'Actual six-array weights differ')
    raws = dict(zip(INPUT_NAMES, (raw, *(canonical(v).encode() for v in (runtime.protocol,
        {**metadata, 'actor_sha256': runtime.actor_sha256}, frozen_selection)))))
    consumption = _consumption(metadata, parameter, runtime.protocol, identity['configuration'], c['scenes'], trajectory_steps, minimum_frame)
    code = _sources(c['sources'])
    spec = {'version': VERSION, 'test_fixture': allow_test_fixture, 'runtime': identity,
        'inputs': {name: _binding(value) for name, value in raws.items()},
        'question_pool_binding': c['binding'], 'excluded_fingerprints_sha256': c['excluded_sha256'],
        'bank_sources': c['sources'], 'sources': code, 'sources_sha256': digest(code), 'accounting': ACCOUNTING,
        'trajectory_steps': trajectory_steps, 'minimum_frame': minimum_frame, 'caps': caps,
        'consumption': consumption, 'consumption_key': digest(consumption)}
    p = {'identity': spec, 'identity_sha256': digest(spec), 'status': 'prepared_candidate', **FLAGS,
         'prepare_environment_steps': 0, 'prepare_nn_queries': 0}
    # All real input checks precede permanent consumption. No new scene restore.
    _same(runtime_api.verify(runtime, allow_test_fixture=allow_test_fixture), identity, 'Runtime changed during preflight')
    _same(_sources(c['sources']), code, 'Source changed during preflight')
    _current(code)
    path, claim = _claim(claim_root, p, output)
    p['consumption_record'] = {'path': 'consumption.json', **_binding(claim), 'registry_path': str(path)}
    try:
        _mkdir(output); put(output/'consumption.json', claim)
        for name, value in raws.items(): put(output/'inputs'/name, value)
        for name, value in saved['blobs'].items(): put(output/'question_pool'/name, value)
        put(output/'prepared.json', p); put(output/'head.json', _initial_head(p)); put(output/'run.lock', b'')
    except BaseException:
        # The durable global claim remains consumed, including failed preparation.
        raise
    return {**deepcopy(p), 'prepared_sha256': file_hash(output/'prepared.json')}


def _inputs(output, expected, fixture):
    bank_api._sha(expected)
    output = _absolute(output); raw = _read(output/'prepared.json')
    _same(sha256(raw).hexdigest(), expected, 'External prepared bytes differ')
    p = json.loads(raw); s = p['identity']; identity = s['runtime']
    if (s['version'] != VERSION or type(fixture) is not bool or s['test_fixture'] is not fixture
            or identity['family'] != runtime_api.FAMILY or identity['test_fixture'] is not fixture
            or p['identity_sha256'] != digest(s) or set(s['inputs']) != set(INPUT_NAMES)
            or any(p[k] is not False for k in FLAGS)):
        raise ValueError('Explicit new workflow and fixture identity required')
    _same(s['accounting'], ACCOUNTING, 'Reused linear accounting provenance differs')
    _same(s['sources'], _sources(s['bank_sources']), 'Complete workflow source closure differs')
    _same(digest(s['sources']), s['sources_sha256'], 'Workflow source digest differs')
    _current(s['sources'])
    raws = {name: _read(output/'inputs'/name) for name in INPUT_NAMES}
    for name, value in raws.items(): _same(_binding(value), s['inputs'][name], 'Original input bytes differ')
    saved = pools.read_registered(output/'question_pool', expected_manifest_sha256=s['question_pool_binding']['manifest_sha256'], allow_test_fixture=fixture)
    _same(_pool_binding(saved), s['question_pool_binding'], 'Complete metadata/new-pool bytes differ')
    _same(saved['configuration'], identity['configuration'], 'Public configuration differs')
    _same(digest(sorted(saved['excluded_fingerprints'])), s['excluded_fingerprints_sha256'], 'Complete exclusions differ')
    metadata, parameter = _actor_record(raws['actor.npz']); protocol = json.loads(raws['protocol.json'])
    _same({**metadata, 'actor_sha256': sha256(raws['actor.npz']).hexdigest()}, json.loads(raws['actor_bindings.json']), 'Original Actor bindings differ')
    _same(sha256(raws['actor.npz']).hexdigest(), identity['actor_sha256'], 'Actual Actor bytes differ from runtime identity')
    _same(digest(metadata), identity['actor_metadata_sha256'], 'Original Actor metadata digest differs')
    _same(digest(protocol), identity['protocol_sha256'], 'Original protocol differs')
    _same(metadata['protocol_sha256'], identity['protocol_sha256'], 'Actual Actor protocol binding differs')
    actual_sources = runtime_api.runtime_sources()  # Hash-only genuine new source provider.
    _same(identity['version'], runtime_api.VERIFY_VERSION, 'Actual alignment verifier version differs')
    _same(identity['runtime_version'], runtime_api.RUNTIME_VERSION, 'Actual alignment runtime version differs')
    _same(identity['runtime_sources'], actual_sources, 'Actual runtime source closure differs')
    _same(identity['runtime_sources_sha256'], digest(actual_sources), 'Actual runtime source digest differs')
    _same(identity['actor_experiment_version'], metadata['experiment_version'], 'Actual Actor producer differs')
    _same(identity['verification_scope'], 'full_content_hashes', 'Preparation requires full content verification')
    if any(identity[k] is not False for k in ('qualification_evaluated', 'release_ready', 'explanation_qualified')):
        raise ValueError('Runtime identity cannot grant qualification')
    _same(identity['runtime_signature'], digest({'version': runtime_api.RUNTIME_VERSION,
        'actor_sha256': identity['actor_sha256'], 'protocol_sha256': identity['protocol_sha256'],
        'actor_metadata_sha256': digest(metadata), 'configuration': identity['configuration'], 'sources': actual_sources}),
        'Saved runtime signature differs from its actual inputs')
    # Actual admission was performed by prepare's exact AlignmentRuntime. Cache
    # reads bind those immutable bytes and independently check the new protocol;
    # no fake Actor/type or old-family metadata is constructed here.
    runtime_api.admission._validate_protocol(protocol, fixture=fixture)
    runtime_api.admission._alignment_bindings(metadata, protocol)
    _same(metadata['experiment_version'], runtime_api.admission.trainer.VERSION, 'Actual alignment producer differs')
    _same(metadata['actor_parameters_sha256'], parameter, 'Actual saved parameter binding differs')
    _same(metadata['scenario_manifest_sha256'], saved['source_scenario_manifest_sha256'], 'Original Actor source scenes differ')
    # This is the original pure record contract, not runtime-family admission.
    old_io._selection(json.loads(raws['selection.json']), SimpleNamespace(actor=SimpleNamespace(metadata=metadata),
        actor_sha256=identity['actor_sha256'], protocol_sha256=identity['protocol_sha256'], test_fixture=fixture))
    _same(_caps(len(saved['scenes']), identity['configuration']['horizon'], s['trajectory_steps'], s['minimum_frame'], fixture), s['caps'], 'Fixed caps differ')
    consumption = _consumption(metadata, parameter, protocol, identity['configuration'], saved['scenes'], s['trajectory_steps'], s['minimum_frame'])
    _same(s['consumption'], consumption, 'Physical/Actor consumption identity differs')
    _same(s['consumption_key'], digest(consumption), 'Permanent key differs')
    claim_raw = _read(output/'consumption.json'); claim = json.loads(claim_raw)
    _same(_binding(claim_raw), {k: p['consumption_record'][k] for k in ('sha256', 'size')}, 'Original claim bytes differ')
    expected_claim = {'version': VERSION, 'key': s['consumption_key'], 'identity_sha256': p['identity_sha256'],
        'output_identity': claim['output_identity'], 'caps': s['caps'], 'test_fixture': fixture,
        'permanent': True, 'refund_allowed': False, 'automatic_retry': False, **FLAGS}
    _same(claim, expected_claim, 'Permanent no-retry claim differs')
    _same(p['consumption_record']['path'], 'consumption.json', 'Saved claim path differs')
    path = _absolute(p['consumption_record']['registry_path']); _absolute(claim['output_identity'])
    if path.name != 'claim.json' or path.parent.name != s['consumption_key']:
        raise ValueError('Original permanent registry key/path differs')
    return output, p, saved


def _content(output, p, phase, report, binding, saved):
    bank = _json(output/'bank.private.json'); s = p['identity']; identity = s['runtime']
    if set(bank) != bank_api._FIELDS: raise ValueError('Actual metadata bank schema required')
    expected = {'version': bank_api.VERSION, 'status': 'candidate', 'formal_ready': False, 'release_ready': False,
        'test_fixture': s['test_fixture'], 'runtime_family': identity['family'], 'runtime_version': identity['runtime_version'],
        'runtime_sources_sha256': identity['runtime_sources_sha256'], 'actor_sha256': identity['actor_sha256'],
        'protocol_sha256': identity['protocol_sha256'], 'runtime_signature': identity['runtime_signature'],
        'sources': s['bank_sources'], 'sources_sha256': digest(s['bank_sources']),
        'question_pool_binding': s['question_pool_binding'], 'excluded_fingerprints_sha256': s['excluded_fingerprints_sha256'],
        'pool_namespace': bank_api.POOL_NAMESPACE, 'pool_scenes': saved['scenes'],
        'trajectory_steps': s['trajectory_steps'], 'minimum_frame': s['minimum_frame'],
        'counterfactual_filter': bank_api.FILTER, 'scope': bank_api.SCOPE}
    for name, value in expected.items(): _same(bank[name], value, 'Saved metadata bank identity differs: '+name)
    checks = bank_api._checks(bank['items']); _same(bank['checks'], checks, 'Actual question checks differ')
    if checks['passed']: validate_saved_items(bank, {scene['id']: scene for scene in saved['scenes']})
    public = PureBank.public_items(SimpleNamespace(_items=bank['items'], content_eligible=checks['passed'])) if phase == 'verification' else None
    fields = {'version': VERSION, 'phase': phase, 'status': 'candidate_generated' if phase == 'generation' else 'candidate_content_verified',
        'test_fixture': s['test_fixture'], 'runtime_family': identity['family'], 'runtime_signature': identity['runtime_signature'],
        'bank_binding': binding, 'independent_replay_completed': phase == 'verification',
        'content_checks': checks, 'content_eligible': checks['passed'], 'public_items': public,
        'question_pool_binding': s['question_pool_binding'], 'accounting': ACCOUNTING}
    _same({k: report.get(k) for k in fields}, fields, 'Report content or public whitelist differs')


def _full(output, p, head, saved):
    """Single complete pass; raw nesting/ACK arithmetic uses original _operation."""
    expected = _initial_head(p)
    for k in ('version', 'identity_sha256', 'automatic_resume', 'reservations_refunded'):
        _same(head[k], expected[k], 'Head identity or permanence differs')
    if type(head['next_sequence']) is not int or head['next_sequence'] < 0: raise ValueError('Invalid operation count')
    directory = output/'operations'; names = {f'{i:06d}' for i in range(head['next_sequence'])}
    if directory.exists() and (directory.is_symlink() or {x.name for x in directory.iterdir()} != names):
        raise ValueError('Uncommitted, missing or extra operation directory')
    roots = {phase: [] for phase in PHASES}; raw_counts = {phase: dict(decision=0, step=0, counterfactual=0) for phase in PHASES}
    seen = set(); previous = expected['last_ack_sha256']; pending = None
    for i in range(head['next_sequence']):
        folder = output/f'operations/{i:06d}'; raw = _read(folder/'reservation.json'); reservation = json.loads(raw)
        phase, context = reservation['phase'], reservation['context']
        if phase not in PHASES or context['operation_id'] in seen or pending is not None: raise ValueError('Duplicate operation or work after pending')
        if phase == 'generation' and roots['verification']: raise ValueError('Generation after verification')
        seen.add(context['operation_id']); _same(reservation, _reservation(p, i, phase, context, previous), 'Original reservation differs')
        binding = {'path': str((folder/'reservation.json').relative_to(output)), **_binding(raw)}
        ack_raw = _read(folder/'ack.json') if (folder/'ack.json').exists() else None
        ack = json.loads(ack_raw) if ack_raw is not None else None
        if {x.name for x in folder.iterdir()}-({'reservation.json', 'calls'}|({'ack.json'} if ack is not None else set())):
            raise ValueError('Unexpected operation artifact')
        value = _operation(output, p, reservation, binding, ack)
        for field, values in (('reserved', reservation['reserved']), ('confirmed', value['confirmed'])):
            for k, n in values.items(): expected['phases'][phase][field][k] += n
        for k, n in value['raw_counts'].items(): raw_counts[phase][k] += n
        if ack is None: pending = {'sequence': i, 'reservation': binding}
        else: previous = sha256(ack_raw).hexdigest(); roots[phase].append(value['root'])
    _same(head['pending'], pending, 'Pending reservation differs'); _same(head['last_ack_sha256'], previous, 'Last ACK differs')
    if set(head['phases']) != set(PHASES): raise ValueError('Unknown phase')
    for phase, value in head['phases'].items():
        for k in ('caps', 'reserved', 'confirmed'): _same(value[k], expected['phases'][phase][k], 'Phase counts or caps differ')
        if any(value['reserved'][k] > cap for k, cap in value['caps'].items()): raise ValueError('Finite phase exhausted')
        if value['status'] not in ('not_started', 'running', 'completed', 'failed'): raise ValueError('Unknown phase status')
        if value['status'] == 'not_started' and any(value['reserved'].values()): raise ValueError('Unstarted phase consumed work')
        if value['status'] == 'completed':
            if pending is not None: raise ValueError('Completed phase contains pending work')
            raw = _read(output/(phase+'_report.json')); report = json.loads(raw)
            _same(_binding(raw), value['report'], 'Original report bytes differ')
            _same(_binding(_read(output/'bank.private.json')), value['bank'], 'Original bank bytes differ')
            if (report['identity_sha256'] != p['identity_sha256'] or report['actor_sha256'] != p['identity']['runtime']['actor_sha256']
                    or report['confirmed_work'] != value['confirmed'] or report['actual_driver_calls'] != raw_counts[phase]
                    or report['original_operation_results_sha256'] != digest(roots[phase]) or any(report[k] is not False for k in FLAGS)):
                raise ValueError('Completed work or component scope differs')
            measured = {'completed_operations': len(roots[phase]), 'reserved_upper_bound': value['reserved']['environment_steps'],
                'actual_environment_steps': value['confirmed']['environment_steps'], 'automatic_resume': False, 'reservations_refunded': False}
            _same(report['bank_operation_audit'], measured, 'Original bank operation report differs from raw work')
            _same(_json(output/'bank.private.json')['generation_audit'], measured, 'Saved bank work differs from the full replay')
            _content(output, p, phase, report, value['bank'], saved)
    if head['phases']['verification']['status'] == 'completed':
        if head['phases']['generation']['status'] != 'completed' or roots['generation'] != roots['verification']:
            raise ValueError('Complete independent original results differ')
    return {'roots': roots, 'raw_counts': raw_counts, 'seen_ids': seen}


class _Budget(linear._Budget):
    def finish(self, report):
        value = deepcopy(self.head); value['phases'][self.phase].update(status='completed', bank=report['bank_binding'],
            report=_binding(_read(self.output/(self.phase+'_report.json'))))
        _full(self.output, self.p, value, self.saved_pool); self.commit(value)


def _receipt(output, p, head):
    return {'version': VERSION, 'accounting': ACCOUNTING, 'prepared_sha256': file_hash(output/'prepared.json'),
        'identity_sha256': p['identity_sha256'], 'head_binding': _binding(_read(output/'head.json')),
        'last_ack_sha256': head['last_ack_sha256'], 'operation_count': head['next_sequence'],
        'bank_binding': _binding(_read(output/'bank.private.json')),
        'reports': {phase: _binding(_read(output/(phase+'_report.json'))) for phase in PHASES},
        'runtime': p['identity']['runtime'], 'sources_sha256': p['identity']['sources_sha256'],
        'selection_binding': p['identity']['inputs']['selection.json'],
        'question_pool_binding': p['identity']['question_pool_binding'], 'consumption_record': p['consumption_record'],
        'physics_replay_on_cache_read': False, 'registry_reread_on_cache_read': False, **FLAGS}


def _static(output, expected, fixture):
    output, p, saved = _inputs(output, expected, fixture)
    if any((output/(phase+'_failure.json')).exists() for phase in PHASES): raise ValueError('Failed work cannot be automatically resumed')
    head = _json(output/'head.json'); audit = _full(output, p, head, saved)
    if head['phases']['verification']['status'] == 'completed':
        _same(_json(output/'completion_receipt.json'), _receipt(output, p, head), 'Original replay receipt differs')
    return output, p, saved, head, audit


def _runtime(output, p):
    s = p['identity']; identity = s['runtime']; folder = output/'inputs'
    runtime = runtime_api.AlignmentRuntime(folder/'actor.npz', protocol=_json(folder/'protocol.json'),
        expected_actor_sha256=identity['actor_sha256'], expected_protocol_sha256=identity['protocol_sha256'],
        expected_bindings=_json(folder/'actor_bindings.json'), allow_test_fixture=s['test_fixture'],
        config=old_io._configuration(identity['configuration']))
    c = bank_api._context(runtime, output/'question_pool', s['question_pool_binding']['manifest_sha256'],
        s['trajectory_steps'], s['minimum_frame'], s['test_fixture'])
    _same(c['identity'], identity, 'Private genuine runtime differs')
    _same(c['binding'], s['question_pool_binding'], 'Private bank pool differs')
    _same(c['sources'], s['bank_sources'], 'Actual bank sources differ')
    old_io._selection(_json(folder/'selection.json'), runtime)
    return runtime


def _work(output, phase, anchor, fixture):
    output = _absolute(output); fd = os.open(output/'run.lock', os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        output, p, saved, head, audit = _static(output, anchor, fixture)
        if head['phases'][phase]['status'] == 'completed': return deepcopy(_json(output/(phase+'_report.json')))
        if head['phases'][phase]['status'] != 'not_started' or head['pending'] is not None:
            raise ValueError('Failed/pending work cannot be resampled or refunded')
        if phase == 'verification' and head['phases']['generation']['status'] != 'completed': raise ValueError('Generation must complete first')
        claim = _json(output/'consumption.json')
        _same(claim['output_identity'], str(output), 'Unfinished workflows cannot move to another output')
        _same(p['consumption_record']['registry_path'], str(_registry_root(fixture)/p['identity']['consumption_key']/'claim.json'),
              'Live work must use the fixed permanent registry')
        if _read(p['consumption_record']['registry_path']) != _read(output/'consumption.json'):
            raise ValueError('Permanent global claim differs')
        runtime = _runtime(output, p); budget = _Budget(output, p, head, phase, audit); budget.saved_pool = saved
        s = p['identity']
        try:
            budget.start()
            with _observe(runtime, budget):
                if phase == 'generation':
                    bank = bank_api.generate_bank(runtime, output/'question_pool', expected_pool_manifest_sha256=s['question_pool_binding']['manifest_sha256'],
                        trajectory_steps=s['trajectory_steps'], minimum_frame=s['minimum_frame'], before_operation=budget.before,
                        after_operation=budget.after, allow_test_fixture=fixture)
                    put(output/'bank.private.json', bank); checks, measured, public = bank['checks'], bank['generation_audit'], None
                else:
                    bank = bank_api.AlignmentMetadataQuestionBank(output/'bank.private.json', runtime, output/'question_pool',
                        expected_bank_sha256=head['phases']['generation']['bank']['sha256'],
                        expected_pool_manifest_sha256=s['question_pool_binding']['manifest_sha256'],
                        before_operation=budget.before, after_operation=budget.after, allow_test_fixture=fixture)
                    checks, measured, public = bank.checks, bank.audit, bank.public_items()
            _same(runtime_api.verify(runtime, allow_test_fixture=fixture), s['runtime'], 'Runtime changed during work')
            _same(_sources(s['bank_sources']), s['sources'], 'Sources changed during work')
            _current(s['sources'])
            confirmed = budget.head['phases'][phase]['confirmed']
            if measured['actual_environment_steps'] != confirmed['environment_steps'] or measured['completed_operations'] != len(budget.roots):
                raise ValueError('Original component audit differs from durable actual work')
            report = {'version': VERSION, 'phase': phase, 'status': 'candidate_generated' if phase == 'generation' else 'candidate_content_verified',
                'test_fixture': fixture, 'identity_sha256': p['identity_sha256'], 'runtime_family': s['runtime']['family'],
                'actor_sha256': runtime.actor_sha256, 'runtime_signature': runtime.signature, 'content_checks': checks,
                'content_eligible': checks['passed'], 'public_items': public, 'independent_replay_completed': phase == 'verification',
                'bank_binding': _binding(_read(output/'bank.private.json')), 'question_pool_binding': s['question_pool_binding'],
                'confirmed_work': deepcopy(confirmed), 'bank_operation_audit': measured, 'accounting': ACCOUNTING,
                'original_operation_results_sha256': digest(budget.roots), 'actual_driver_calls': deepcopy(budget.actual_returns),
                'raw_collection_mechanism': 'unchanged private genuine-instance observation wrappers',
                'history_integrity_scope': 'full chain at entry/end/cache; each operation checks only its new records',
                'runtime_verification_scope': 'full content hashes at each scene entry/exit; immutable object checks during original operation ACKs', **FLAGS}
            put(output/(phase+'_report.json'), report); budget.finish(report)
            if phase == 'verification': put(output/'completion_receipt.json', _receipt(output, p, budget.head))
            return deepcopy(report)
        except BaseException as error:
            # Remove a success receipt before attempting failure writes. Counts
            # come from the durable head, never from uncommitted local totals.
            recovery = False
            try: (output/'completion_receipt.json').unlink(missing_ok=True)
            except OSError: recovery = True
            durable = None
            try: durable = _json(output/'head.json')
            except BaseException: pass
            try:
                put(output/(phase+'_failure.json'), {'version': VERSION, 'phase': phase, 'error': repr(error),
                    'durable_head': durable, 'actual_driver_returns': budget.actual_returns,
                    'partial_execution_may_be_unknown': durable is None or durable['pending'] is not None,
                    'publication_recovery_required': recovery, 'retry_allowed': False, **FLAGS})
                if durable is not None:
                    changed = deepcopy(durable); changed['phases'][phase].update(status='failed', error=repr(error))
                    changed['revision'] += 1; put(output/'head.json', changed, replace=True)
            except BaseException: pass
            raise


def generate(output, *, expected_prepared_sha256, allow_test_fixture=False):
    return _work(output, 'generation', expected_prepared_sha256, allow_test_fixture)


def verify(output, *, expected_prepared_sha256, allow_test_fixture=False):
    return _work(output, 'verification', expected_prepared_sha256, allow_test_fixture)


def read_completed(output, *, expected_prepared_sha256, allow_test_fixture=False):
    output = _absolute(output); fd = os.open(output/'run.lock', os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        output, p, saved, head, _ = _static(output, expected_prepared_sha256, allow_test_fixture)
        if head['phases']['verification']['status'] != 'completed': raise ValueError('Complete independent replay required')
        return {'version': VERSION, 'report': deepcopy(_json(output/'verification_report.json')),
            'prepared': deepcopy(p),
            'bank_private_data': deepcopy(_json(output/'bank.private.json')), 'bank_binding': deepcopy(head['phases']['verification']['bank']),
            'replay_receipt': deepcopy(_json(output/'completion_receipt.json')), 'replay_receipt_sha256': file_hash(output/'completion_receipt.json'),
            'question_pool_binding': deepcopy(p['identity']['question_pool_binding']), 'excluded_fingerprints': saved['excluded_fingerprints'],
            'public_items': deepcopy(_json(output/'verification_report.json')['public_items']), **FLAGS}
