"""Zero-execution readers of completed, externally byte-bound family evidence.

Stored numeric outputs and oracle records are reconciled, not reexecuted.  No
reader invokes Actor logits, environment construction/step, Torch operations,
answer generation, fitting, a resumable driver, or final-test evaluation.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re

import numpy as np

from backend import warehouse_runtime_family as registry
from backend.warehouse_family_explanation import actor_parameter_sha256
from backend.training import warehouse_family_actor_parity as parity
from backend.training import warehouse_family_answer_run as answer_driver
from backend.training import warehouse_native_public_feedback as history
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.layouts import get_map_layout
from env.warehouse_native.observations import public_observations
from env.warehouse_native.policy import ACTIONS
from ui import warehouse_family_answer_verification as answer

VERSION = 'warehouse-family-saved-evidence.v1'
PARITY_ANCHORS = frozenset(('binding.json', 'sources.json', 'witnesses.json', 'journal.json',
                          'row_provenance.json', 'observations.npz', 'numeric_outputs.npz'))
ZERO_EXECUTION = {'environment_constructions': 0, 'environment_steps': 0, 'environment_resets': 0,
                 'actor_loads': 0, 'NN_forwards': 0, 'Torch_operations': 0, 'checkpoint_decodes': 0,
                 'tree_fits': 0, 'training_updates': 0, 'answer_oracle_reexecutions': 0}


def sources():
    result = {**parity.execution_sources(), **answer_driver.sources()}
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _same(a, b, reason):
    if canonical(a) != canonical(b): raise ValueError(reason)


def _int(value, low=0):
    if type(value) is not int or value < low: raise ValueError('Invalid integer count')
    return value


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value): raise ValueError('External SHA256 required')
    return value


class _Files:
    def __init__(self, root):
        self.root = Path(root).expanduser().absolute(); self.records = {}
        if self.root.resolve() != self.root or not self.root.is_dir(): raise ValueError('Existing canonical evidence directory required')

    def raw(self, name, expected=None):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name: raise ValueError('Noncanonical artifact name')
        path = self.root / name
        if path.resolve() != path or not path.is_file(): raise ValueError('Missing or symlinked evidence')
        raw = path.read_bytes(); stat = path.stat(); item = {'sha256': sha256(raw).hexdigest(), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if expected is not None and item['sha256'] != _sha(expected): raise ValueError('External artifact bytes differ: ' + name)
        if name in self.records: _same(item, self.records[name], 'Evidence changed while reading')
        self.records[name] = item
        return raw

    def json(self, name, expected=None):
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value: raise ValueError('Duplicate JSON key')
                value[key] = item
            return value
        def invalid(x): raise ValueError('Nonfinite JSON')
        value = json.loads(self.raw(name, expected), object_pairs_hook=pairs, parse_constant=invalid); canonical(value)
        return value

    def npz(self, name, keys):
        with np.load(io.BytesIO(self.raw(name)), allow_pickle=False) as archive:
            if set(archive.files) != set(keys) or len(archive.files) != len(keys): raise ValueError('Array package names differ')
            return {k: archive[k].copy() for k in keys}

    def bound(self, item, name):
        if set(item) != {'path', 'sha256', 'size'} or item['path'] != name: raise ValueError('Artifact identity differs')
        raw = self.raw(name, item['sha256'])
        if len(raw) != item['size']: raise ValueError('Artifact size differs')
        return self.json(name)

    def unchanged(self):
        for name, item in tuple(self.records.items()): self.raw(name, item['sha256'])


def _stored_observations(snapshot, config):
    """Rebuild public features from inert dataclass values, never an environment."""
    _same(snapshot['configuration'], asdict(config), 'Snapshot physical configuration differs')
    if snapshot.get('public_feedback_version') != history.VERSION or snapshot.get('public_feedback_mode') != 'observed':
        raise ValueError('Complete genuine observed197 history required')
    raw = deepcopy(snapshot['state'])
    raw['agents'] = [AgentState(**{**a, 'position': tuple(a['position'])}) for a in raw['agents']]
    for key in ('tasks', 'completed_tasks'):
        raw[key] = [DeliveryTask(**{**t, 'pickup_position': tuple(t['pickup_position']), 'delivery_position': tuple(t['delivery_position'])}) for t in raw[key]]
    state = WarehouseState(**raw)
    saved = snapshot['public_feedback_history']
    history._validate_history(saved, state, ('robot_1', 'robot_2'), config.horizon)
    base = public_observations(state, config); result = {}
    for role, key in enumerate(('robot_1', 'robot_2')):
        values = np.zeros(20, dtype=np.float32)
        if saved['valid']:
            order = (key, ('robot_1', 'robot_2')[1-role]); extra = [1.]
            for person in order: extra.extend(float(saved['submitted_actions'][person] == a) for a in ACTIONS)
            extra.extend(float(saved['move_canceled'][person]) for person in order)
            extra.extend(float(saved['collision_kind'] == kind) for kind in history.COLLISION_KINDS)
            extra.extend(math.log1p(saved['consecutive_move_canceled'][person])/math.log1p(config.horizon) for person in order)
            extra.append(math.log1p(saved['consecutive_collision'])/math.log1p(config.horizon)); values[:] = extra
        result[key] = np.concatenate((base[key], values)).astype(np.float32)
    return state, result


def _numeric(arrays):
    a, b = arrays['numpy_logits'], arrays['torch_logits']
    if a.dtype != np.float32 or b.dtype != np.float32 or a.ndim != 2 or a.shape != b.shape or a.shape[1] != 5 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite float32 Nx5 numeric records required')
    ps = []
    for label, logits in (('numpy', a), ('torch', b)):
        p = np.exp(logits - logits.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True); ps.append(p)
        saved = arrays[label + '_probabilities']; actions = arrays[label + '_action_indices']
        if saved.dtype != np.float32 or not np.array_equal(p, saved) or actions.dtype != np.int64 or not np.array_equal(p.argmax(-1), actions):
            raise ValueError('Recorded probability/action differs from native float32 softmax')
    error = np.abs(a-b); same = ps[0].argmax(-1) == ps[1].argmax(-1); raw = a.argmax(-1) == b.argmax(-1)
    return {'maximum_absolute_error': float(error.max()), 'maximum_probability_error': float(np.abs(ps[0]-ps[1]).max()),
        'argmax_semantics': 'native_float32_exp_normalize_probabilities', 'argmax_mismatches': int((~same).sum()),
        'all_argmax_equal': bool(same.all()), 'raw_logit_argmax_mismatches': int((~raw).sum()),
        'raw_logit_argmax_all_equal': bool(raw.all()), 'passed': bool(error.max() <= 1e-4 and same.all())}


def _parity_rows(files, binding, witnesses, journal, origins, matrix, scenes, config):
    if journal['version'] != parity.VERSION or journal['status'] != 'collection_complete': raise ValueError('Parity collection incomplete')
    _same(journal['binding_sha256'], digest(binding), 'Journal binding differs')
    count = _int(witnesses['calibration_scene_count'], 1)
    if not binding['test_fixture'] and count != 12: raise ValueError('Production requires twelve calibration scenes')
    if count > 12 or len(scenes) != count: raise ValueError('Calibration scene count differs')
    if witnesses['version'] != parity.WITNESS_VERSION or witnesses['test_fixture'] is not binding['test_fixture']: raise ValueError('Witness version/scope differs')
    segments = witnesses['segments']; by_id = {s['id']: s for s in scenes}; used = {}; ids = set(); n = 0; all_obs = []; expected_origins = []; expected_snapshots = []
    if type(segments) is not list or not segments or len(segments) > count * config.horizon: raise ValueError('Invalid fixed witness segments')
    charger = tuple(get_map_layout(config.map_layout_id).charger_position)
    def add(snapshot, segment, info):
        state, rows = _stored_observations(snapshot, config)
        for role, key in enumerate(('robot_1', 'robot_2')):
            agent = state.agents[role]; categories = ['normal']
            if 0 < agent.battery <= 8: categories.append('low_battery')
            if agent.battery == 0: categories.append('zero_battery')
            if agent.carrying_task_id is not None: categories.append('carrying')
            if tuple(agent.position) == charger and agent.last_battery_delta > 0: categories.append('charging')
            if info is not None and info.get('robot_collision') is True: categories.append('collision')
            if state.frame >= config.horizon - 1: categories.append('timer_boundary')
            expected_origins.append({'row': len(all_obs), 'segment': segment['id'], 'scenario_id': segment['scenario_id'],
                'initial_fingerprint': segment['initial_fingerprint'], 'frame': state.frame, 'agent_id': key,
                'snapshot_sha256': digest(snapshot), 'categories': categories,
                'origin': 'calibration_initial' if state.frame == 0 else 'actual_public_action_witness'})
            all_obs.append(rows[key]); expected_snapshots.append(snapshot)
    for si, segment in enumerate(segments):
        if set(segment) != {'id', 'scenario_id', 'initial_fingerprint', 'actions'} or type(segment['id']) is not str or not segment['id'] or segment['id'] in ids: raise ValueError('Invalid witness identity')
        ids.add(segment['id']); scene = by_id.get(segment['scenario_id']); actions = segment['actions']
        if scene is None or segment['initial_fingerprint'] != scene['fingerprint'] or type(actions) is not list or not 1 <= len(actions) <= config.horizon: raise ValueError('Witness scene/horizon differs')
        used[scene['id']] = used.get(scene['id'], 0) + len(actions)
        previous = None
        for index, submitted in enumerate(actions):
            if type(submitted) is not dict or set(submitted) != {'robot_1', 'robot_2'} or any(a not in ACTIONS for a in submitted.values()): raise ValueError('Witness contains nonpublic actions')
            op = journal['operations'][n]; name = f'transitions/{si:04d}_{index:04d}.json'; row = files.bound(op['record'], name)
            if op['status'] != 'acknowledged' or op['segment'] != si or op['step'] != index: raise ValueError('Unconfirmed/reordered witness')
            _same(op['actions'], submitted, 'Journal submitted action differs'); _same(op['before_sha256'], digest(row['before']), 'Journal before hash differs')
            for key, value in {'binding_sha256': digest(binding), 'segment': segment['id'], 'scenario_id': scene['id'],
                'initial_fingerprint': scene['fingerprint'], 'submitted_audit_actions': submitted,
                'neural_policy_used_for_witness_actions': False, 'state_interventions': []}.items(): _same(row[key], value, 'Raw witness identity differs')
            if index == 0:
                for key, value in scene['snapshot'].items(): _same(row['before'][key], value, 'Witness initial raw state/RNG differs')
                if row['before']['state']['frame'] != 0 or row['before']['public_feedback_history']['valid'] is not False: raise ValueError('Initial witness history is not unknown')
                add(row['before'], segment, None)
            else: _same(row['before'], previous, 'Witness complete snapshot chain differs')
            before, after = row['before']['state'], row['after']['state']
            if before['terminated'] or before['truncated'] or before['frame'] != index or after['frame'] != index + 1: raise ValueError('Witness terminal/frame order differs')
            _same(row['info']['requested_actions'], submitted, 'Physical submitted actions differ')
            _same(row['executed_actions'], row['info']['executed_actions'], 'Physical execution fields differ')
            for flag in ('terminated', 'truncated'):
                if type(row[flag]) is not bool or row[flag] is not after[flag]: raise ValueError('Terminal records differ')
            _same(row['after']['public_feedback_history']['previous_counts'], {'valid': row['before']['public_feedback_history']['valid'],
                'frame': before['frame'], 'move_canceled': row['before']['public_feedback_history']['consecutive_move_canceled'],
                'collision': row['before']['public_feedback_history']['consecutive_collision']}, 'History counter chain differs')
            add(row['after'], segment, row['info']); previous = row['after']; n += 1
    maximum = sum(used.values())
    if set(used) != set(by_id) or any(v > config.horizon for v in used.values()): raise ValueError('Witness coverage/cap differs')
    for value in (journal['cap'], journal['reserved'], journal['acknowledged'], witnesses['maximum_environment_steps'], binding['maximum_environment_steps']):
        if _int(value) != maximum: raise ValueError('Parity consumption differs')
    if len(journal['operations']) != n or n != maximum: raise ValueError('Extra/unconfirmed parity operations')
    expected_paths = {f'{si:04d}_{j:04d}.json' for si, segment in enumerate(segments) for j in range(len(segment['actions']))}
    if {p.name for p in (files.root / 'transitions').iterdir()} != expected_paths: raise ValueError('Extra or missing parity raw record')
    _same(origins['rows'], expected_origins, 'Origin, categories or ordering differs'); _same(origins['snapshots'], expected_snapshots, 'Observation snapshot bytes differ')
    expected = np.asarray(all_obs, dtype=np.float32)
    if matrix.dtype != np.float32 or matrix.shape != expected.shape or not np.array_equal(matrix, expected): raise ValueError('Saved197 observations differ from complete snapshots')
    coverage = {name: sum(name in item['categories'] for item in expected_origins) for name in parity.REQUIRED_COVERAGE}
    return n, len(expected), coverage


def read_actor_parity(output, *, expected_report_sha256, expected_artifact_sha256,
                      runtime, scenarios, expected_bindings, allow_test_fixture=False):
    identity = registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    files = _Files(output)
    if (files.root / 'failure.json').exists(): raise ValueError('Failed parity output cannot be admitted')
    if type(expected_artifact_sha256) is not dict or set(expected_artifact_sha256) != PARITY_ANCHORS: raise ValueError('All seven original parity artifacts need external byte anchors')
    report = files.json('report.json', _sha(expected_report_sha256))
    for name, expected in expected_artifact_sha256.items(): files.raw(name, expected)
    binding = files.json('binding.json'); witness = files.json('witnesses.json'); journal = files.json('journal.json')
    if (binding['version'] != parity.VERSION or report['version'] != parity.VERSION or binding['test_fixture'] is not allow_test_fixture or report['test_fixture'] is not allow_test_fixture
            or set(expected_bindings) != parity.EXPECTED_KEYS): raise ValueError('Parity version/fixture/bindings differ')
    _same(binding['runtime'], identity, 'Actual runtime identity differs'); _same(binding['expected_bindings'], expected_bindings, 'External Actor bindings differ')
    for key in ('runtime_family', 'runtime_version', 'runtime_signature', 'actor_sha256', 'protocol_sha256'):
        _same(expected_bindings[key], identity['family' if key == 'runtime_family' else key], 'Actor/runtime binding differs')
    _same(expected_bindings['actor_parameters_sha256'], actor_parameter_sha256(runtime.actor), 'Actual NPZ tensor parameter hash differs')
    _same(expected_bindings['feature_names_sha256'], digest(runtime.actor.metadata['feature_names']), '197 feature ordering differs')
    _same(expected_bindings['scenario_manifest_sha256'], digest(scenarios), 'Scenario byte content differs')
    _same(runtime.actor.metadata['scenario_manifest_sha256'], digest(scenarios), 'Actor scenario binding differs')
    _same(scenarios['configuration'], asdict(runtime.config), 'Configuration differs')
    _same(report['binding'], binding, 'Report identity differs'); _same(files.records['journal.json']['sha256'], report['journal_sha256'], 'Report journal anchor differs')
    _same(files.json('sources.json'), parity.execution_sources(), 'Numeric producer sources changed')
    _same(binding['source_sha256'], digest(parity.execution_sources()), 'Numeric source digest differs')
    _same(binding['witnesses_sha256'], digest(witness), 'Witness input digest differs')
    count = witness['calibration_scene_count']; scenes = scenarios['splits']['calibration'][:count]
    if scenarios.get('test_fixture', False) is not allow_test_fixture: raise ValueError('Scenario fixture scope differs')
    if 'split' in scenarios: _same(scenarios['split'], scenarios['splits'], 'Scenario aliases differ')
    excluded = {s['fingerprint'] for k, pool in scenarios['splits'].items() if k != 'calibration' for s in pool}
    if len({s['fingerprint'] for s in scenes}) != len(scenes) or excluded.intersection(s['fingerprint'] for s in scenes): raise ValueError('Calibration pool overlap')
    for i, scene in enumerate(scenes):
        if scene['id'] != f'calibration_{i:04d}' or scene.get('test_fixture', False) is not allow_test_fixture or scene['snapshot']['state']['frame'] != 0: raise ValueError('Fixed initial calibration matrix differs')
    matrix = files.npz('observations.npz', ['observations'])['observations']; origins = files.json('row_provenance.json')
    n, rows, coverage = _parity_rows(files, binding, witness, journal, origins, matrix, scenes, runtime.config)
    arrays = files.npz('numeric_outputs.npz', ['numpy_logits', 'torch_logits', 'numpy_probabilities', 'torch_probabilities', 'numpy_action_indices', 'torch_action_indices'])
    if len(arrays['numpy_logits']) != rows: raise ValueError('Numeric row count differs')
    result = _numeric(arrays); _same(report['numerical_result'], result, 'Saved numeric statistics differ')
    missing = [k for k, v in coverage.items() if not v]; passed = result['passed'] and not missing
    expected = {'passed': passed, 'status': 'numeric_parity_passed' if passed else 'numeric_parity_failed', 'coverage': coverage, 'missing_categories': missing, 'rows': rows,
        'actual_environment_steps': n, 'acknowledged_environment_steps': n, 'reserved_environment_steps': n,
        'numpy_batch_forward_calls': 1, 'torch_functional_forward_passes': 1, 'torch_functional_linear_calls': 3,
        'torch_nn_constructions': 0, 'checkpoint_decodes': 0, 'optimizer_steps': 0, 'training_steps': 0, 'state_mutations_outside_physics': 0,
        'audit_witness_reachability_verified': True, 'autonomous_neural_reachability_claimed': False, 'same_exported_tensors_used_by_both_implementations': True,
        'independent_checkpoint_export_verification': False, 'release_ready': False, 'formal_ready': False, 'explanation_qualified': False,
        'evaluated_splits': ['calibration'], 'final_test_execution': False, 'explanation_test_execution': False}
    for key, value in expected.items(): _same(report.get(key), value, 'Report aggregate/scope differs: ' + key)
    files.unchanged(); _same(registry.verify(runtime, allow_test_fixture=allow_test_fixture), identity, 'Runtime changed during read')
    return {'version': VERSION, 'kind': 'actor_parity_saved_evidence', 'passed_saved_evidence': passed,
        'numerical_result': result, 'coverage': coverage, 'recorded_environment_steps': n, 'rows': rows,
        'externally_anchored_inputs': files.records, 'actual_reader_execution': deepcopy(ZERO_EXECUTION),
        'scope': 'Externally bound stored numeric outputs, pure197 features, witness chains and aggregates; not NN/Torch/physics replay or independent checkpoint export proof',
        'release_ready': False, 'formal_ready': False}


def _answer_transition(record, config, bindings):
    before, after = record['before'], record['after']; b = before['state']; a = after['state']
    _, observations = _stored_observations(before, config); _stored_observations(after, config)
    if b['terminated'] or b['truncated'] or a['frame'] != b['frame'] + 1: raise ValueError('Saved answer trajectory frame differs')
    if record['runtime_signature'] != bindings['runtime_signature']: raise ValueError('Answer trajectory runtime differs')
    submitted = record['submitted_actions']; decision = record['decision']
    if set(submitted) != {'robot_1', 'robot_2'} or record['participant_action'] != submitted['robot_1']: raise ValueError('Answer trajectory player binding differs')
    for key, value in {'actor_sha256': bindings['actor_sha256'], 'runtime_signature': bindings['runtime_signature'],
        'protocol_sha256': bindings['protocol_sha256'], 'frame': b['frame'], 'masks': False, 'post_policy_overrides': 0,
        'robot_1_policy_is_not_participant_input': True}.items(): _same(decision.get(key), value, 'Saved neural decision identity differs')
    for agent in ('robot_1', 'robot_2'):
        p = np.asarray(decision['probabilities'][agent], dtype=np.float32)
        if p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.): raise ValueError('Invalid saved NN distribution')
        _same(decision['observation_hashes'][agent], sha256(observations[agent].tobytes()).hexdigest(), 'Saved neural observation differs from197 state')
        _same(decision['policy_actions'][agent], ACTIONS[int(p.argmax())], 'Saved NN action differs from its probability distribution')
    _same(submitted['robot_2'], decision['policy_actions']['robot_2'], 'Saved NN command overwritten')
    return before, after


def _answer_cases(generated, verified, scenes, config, bindings):
    specs = answer.case_specs(); expected_ids = {s['id'] + '/' + c['id'] for s in scenes for c in specs}
    if len(specs) != 40 or len(generated['cases']) != len(expected_ids) or len(verified['cases']) != len(expected_ids): raise ValueError('Fixed forty-case matrix incomplete')
    g = {c['case_id']: c for c in generated['cases']}; v = {c['case_id']: c for c in verified['cases']}
    if set(g) != expected_ids or set(v) != expected_ids: raise ValueError('Case IDs duplicated, missing or substituted')
    if set(generated['trajectories']) != {s['id'] for s in scenes}: raise ValueError('Original scenario trajectory matrix differs')
    frames = {}; phase_prefix = []; coverage = set(); predicates = set(); passed_all = True; missing = 0; agreement = disagreement = 0
    for si, scene in enumerate(scenes):
        trajectory = generated['trajectories'][scene['id']]; transitions = trajectory['transitions']
        _same(trajectory['initial_fingerprint'], scene['fingerprint'], 'Answer scenario fingerprint differs')
        _same(trajectory['profile'], ('skilled', 'assertive', 'noisy')[si % 3], 'Fixed answer partner differs')
        _same(trajectory['seed_sequence'], [98131, si, 1], 'Fixed answer seed differs')
        if not transitions or len(transitions) > config.horizon: raise ValueError('Missing or overlong original answer trajectory')
        current = None; all_frames = {}; prefix = []
        for i, record in enumerate(transitions):
            before, after = _answer_transition(record, config, bindings)
            if i == 0:
                for key, value in scene['snapshot'].items(): _same(before[key], value, 'Answer initial physical/RNG state differs')
                if before['public_feedback_history']['valid'] is not False: raise ValueError('Answer frame zero must have unknown history')
                all_frames[0] = {'after': before, 'runtime_signature': bindings['runtime_signature']}
            else: _same(before, current, 'Answer trajectory full snapshot chain differs')
            if before['state']['frame'] != i: raise ValueError('Answer trace order differs')
            current = after; all_frames[i+1] = record; prefix.append(record['participant_action'])
            phase_prefix.append((scene['id'], record))
        _same(prefix, trajectory['player_actions'], 'Answer player prefix differs')
        if not (current['state']['terminated'] or current['state']['truncated']): raise ValueError('Answer required terminal anchor missing')
        frames[scene['id']] = all_frames
        for spec in specs:
            case_id = scene['id'] + '/' + spec['id']; supplied = g[case_id]; outcome = v[case_id]
            if type(outcome.get('passed')) is not bool: raise ValueError('Case verdict is not a boolean')
            target = {'initial': 0, 'first': 1, 'third': 3, 'terminal': len(prefix)}[spec['anchor']]
            if target not in all_frames:
                missing += 1; passed_all = False
                if supplied.get('missing_case') is not True or outcome['passed'] is not False: raise ValueError('Missing anchor was declared passed')
                continue
            record = all_frames[target]; request = answer.request_for_spec(spec, target)
            expected = {'case_id': case_id, 'scenario_id': scene['id'], 'initial_fingerprint': scene['fingerprint'],
                'player_actions': prefix[:target], 'request': request, 'expected': spec['expected'], 'language': spec['language'],
                'intent': spec['expected']['intent'], 'question': spec['question'], 'record_sha256': digest(record)}
            for key, value in expected.items(): _same(supplied.get(key), value, 'Raw bilingual case binding differs: ' + key)
            if type(supplied['answer']) is not str or sha256(supplied['answer'].encode()).hexdigest() != supplied['answer_sha256']: raise ValueError('Saved answer bytes differ')
            if not outcome['passed']:
                passed_all = False
                continue
            facts = outcome['independent_facts']; _same(outcome['independent_facts_sha256'], digest(facts), 'Original oracle facts hash differs')
            _same(outcome['selected_frame'], target, 'Original oracle selected frame differs')
            _same(outcome['expected_error'], spec['expected'].get('error'), 'Original rejection differs from fixed case')
            _same(supplied.get('error'), spec['expected'].get('error'), 'Generation rejection differs from fixed case')
            if 'error' not in spec['expected']: coverage.add((spec['language'], spec['expected']['intent']))
            branches = facts.get('transitions', [])
            _same(outcome['counterfactual_steps'], len(branches), 'Oracle branch count differs')
            if len(branches) > 3: raise ValueError('Counterfactual exceeds three steps')
            if spec['expected']['intent'] == 'counterfactual' and 'steps' in spec['expected']:
                planned = spec['expected']['player_actions'] + ['WAIT'] * (spec['expected']['steps'] - len(spec['expected']['player_actions']))
                _same(facts['assumed_player_actions'], planned, 'Counterfactual first-then-WAIT assumption differs')
                initial = record['before'] if spec['expected'].get('focus') == 'executed' else record['after']
                previous = initial
                for i, branch in enumerate(branches):
                    before, after = _answer_transition(branch, config, bindings)
                    _same(before, previous, 'Original oracle branch snapshot chain differs')
                    _same(branch['participant_action'], planned[i], 'Original oracle intervention action differs'); previous = after
            tree_same = facts['tree_action'] == facts['neural_action'] if 'tree_action' in facts else None
            _same(outcome['tree_agreement'], tree_same, 'Original tree-agreement aggregate differs')
            agreement += int(tree_same is True); disagreement += int(tree_same is False)
            predicates.update(t['feature'] for t in facts.get('tree_trace', []))
    required = {(language, intent) for language in ('zh', 'en') for intent in ('reason', 'alternative', 'counterfactual', 'failure', 'rules', 'clarify')}
    all_covered = required <= coverage; passed = passed_all and all_covered
    expected_summary = {'rows': len(expected_ids), 'coverage': [{'language': a, 'intent': b} for a, b in sorted(coverage)],
        'required_coverage_complete': all_covered, 'actual_tree_agreement_cases': agreement, 'actual_tree_disagreement_cases': disagreement,
        'actual_predicate_features': sorted(predicates), 'uncovered_history_predicates': sorted(set(history.HISTORY_FEATURE_NAMES) - predicates),
        'history_predicate_full_coverage': set(history.HISTORY_FEATURE_NAMES) <= predicates, 'passed': passed}
    for key, value in expected_summary.items(): _same(verified.get(key), value, 'Saved verification case aggregate differs: ' + key)
    _same(generated['missing_anchors'], missing, 'Generation missing anchor aggregate differs')
    return expected_summary, phase_prefix


def read_answer_evidence(output, *, expected_generation_report_sha256, expected_verification_report_sha256,
                         runtime, program_path, scenarios, expected_bindings, allow_test_fixture=False):
    identity = registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    files = _Files(output); root = files.root
    generated = files.json('generate/report.json', _sha(expected_generation_report_sha256))
    verified = files.json('verify/report.json', _sha(expected_verification_report_sha256))
    plan = files.json('plan.json')
    _same(answer.input_bindings(runtime, program_path, scenarios), expected_bindings, 'Actual answer input binding differs')
    actual_plan = answer_driver._plan(runtime, program_path, scenarios, expected_bindings, plan['execution_id'], allow_test_fixture)
    _same(plan, actual_plan, 'Original answer source/runtime/plan changed')
    manifest = files.json('manifest.json')
    if manifest['version'] != answer_driver.VERSION or manifest['plan_sha256'] != digest(plan): raise ValueError('Answer plan manifest differs')
    files.raw('program.json', plan['program_sha256']); files.bound(manifest['program'], 'program.json')
    _same(files.bound(manifest['scenarios'], 'scenarios.json'), scenarios, 'Frozen answer scenarios differ')
    states = {}; journal_prefix = {}
    for phase in answer_driver.PHASES:
        if (root / phase / 'failure.json').exists(): raise ValueError('Failed answer phase cannot be admitted')
        for path in (root / phase).rglob('*'):
            if path.is_file(): files.raw(path.relative_to(root).as_posix())
        state, original = answer_driver._journal_state(root, phase, plan)
        if state['status'] != 'completed' or original is None: raise ValueError('Both answer phases must already be completed; this reader never continues them')
        _same(original, generated if phase == 'generate' else verified, 'External report differs from completed phase')
        states[phase] = state; prefix = []
        for i in range(state['acknowledged_steps']):
            row = answer_driver._read(root / phase / 'operations' / f'{i:05d}.record.json.gz')
            if row['phase'] == 'fixed_prefix': prefix.append((row['scenario_id'], row))
        journal_prefix[phase] = prefix
    _same(verified.get('input_report_sha256'), digest(generated), 'Original verification consumed another generation report')
    scenes = scenarios['splits']['explanation_test'][:answer.SCENARIO_COUNT]
    if not scenes or (not allow_test_fixture and len(scenes) != answer.SCENARIO_COUNT): raise ValueError('Original twelve-scene answer matrix required')
    if scenarios.get('test_fixture', False) is not allow_test_fixture: raise ValueError('Answer scene scope differs')
    excluded = {s['fingerprint'] for key, pool in scenarios['splits'].items() if key != 'explanation_test' for s in pool}
    if len({s['id'] for s in scenes}) != len(scenes) or len({s['fingerprint'] for s in scenes}) != len(scenes) or excluded.intersection(s['fingerprint'] for s in scenes): raise ValueError('Answer scenario pool overlaps')
    expected_header = answer._header(runtime, program_path, scenarios, answer.answer_verification_sources(), len(scenes))
    for report in (generated, verified):
        for key, value in expected_header.items(): _same(report.get(key), value, 'Original answer immutable header differs')
    _same(generated['scenario_inputs'], scenes, 'Answer source scene list differs')
    for report in (generated, verified):
        if report.get('audit_runtime_only') is not True: raise ValueError('Original answer runtime audit scope missing')
    if verified.get('source_hashes_verified_after_execution') is not True: raise ValueError('Original verifier did not finish source checks')
    _same(verified.get('unverified'), ['independent_RCPD_action_fidelity', 'intervention_direction_fidelity', 'human_explanation_effect', 'release_qualification'], 'Original unverified scope differs')
    statistics, expected_prefix = _answer_cases(generated, verified, scenes, runtime.config, expected_bindings)
    for phase, rows in journal_prefix.items():
        if len(rows) != len(expected_prefix): raise ValueError('Answer fixed-prefix journal count differs')
        for (scene_id, raw), (expected_id, transition) in zip(rows, expected_prefix):
            _same(scene_id, expected_id, 'Journal scenario prefix differs')
            for key in ('before', 'after', 'submitted_actions', 'rewards', 'terminated', 'truncated'):
                _same(raw[key], transition[key] if key in transition else transition['after']['state'][key], 'Journal original prefix differs: ' + key)
    files.unchanged(); _same(registry.verify(runtime, allow_test_fixture=allow_test_fixture), identity, 'Runtime changed during saved answer read')
    return {'version': VERSION, 'kind': 'bilingual_answer_saved_evidence', 'passed_saved_evidence': statistics['passed'],
        'statistics': statistics, 'external_generation_report_sha256': expected_generation_report_sha256,
        'external_verification_report_sha256': expected_verification_report_sha256,
        'recorded_phase_steps': {p: states[p]['acknowledged_steps'] for p in states},
        'externally_anchored_reports_and_linked_records': files.records, 'actual_reader_execution': deepcopy(ZERO_EXECUTION),
        'scope': 'Original independently generated-and-replayed oracle result bytes, case binding, saved facts and durable step accounting; no new oracle, NN, physical or answer-text replay',
        'release_ready': False, 'formal_ready': False}
