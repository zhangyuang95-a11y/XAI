"""Pure saved-record reader for the explicit development intervention producer.

No Actor/environment is constructed or queried. All labels are saved actual NN
outputs. Counterfactual next decisions were queried, not executed. This module
provides development data, never explanation or release qualification.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from backend.training import warehouse_family_intervention_collection as producer
from backend.training.warehouse_native_common import ROOT, digest
from backend.training.warehouse_native_revision_provenance import _absolute, _bytes
from backend.training.warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from backend.warehouse_family_explanation import actor_parameter_sha256
from backend.warehouse_shutdown_runtime import RUNTIME_VERSION, runtime_sources
from env.warehouse.domain import WarehouseConfig
from env.warehouse.rewards import RewardConfig
from env.warehouse_native.observations import observation_names

VERSION = 'warehouse-family-development-intervention-dataset.v1'
PRODUCER_VERSION = 'warehouse-family-development-intervention-collection.v1'
ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
POOLS = ('train', 'selection')
ZERO = {'environment_constructions': 0, 'actor_constructions': 0, 'neural_forwards': 0,
        'environment_steps': 0, 'checkpoint_decodes': 0, 'tree_predictions': 0, 'tree_fits': 0}


def sources():
    result = producer.sources()
    result[str(Path(__file__).relative_to(ROOT))] = sha256(_bytes(Path(__file__))).hexdigest()
    return result


def _same(a, b, reason):
    if digest(a) != digest(b):
        raise ValueError(reason)


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Explicit lowercase external SHA256 required')
    return value


def _flags(value):
    for name in producer.FLAGS:
        if value.get(name) is not False:
            raise ValueError('Dataset evidence cannot grant qualification: ' + name)


class _Read:
    def __init__(self, root):
        self.root = root
        self.bindings = {}

    def raw(self, relative, expected=None):
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or str(p) != relative:
            raise ValueError('Noncanonical relative evidence path')
        return self.absolute(self.root / p, expected)

    def absolute(self, path, expected=None):
        path = _absolute(path)
        raw = _bytes(path); h = sha256(raw).hexdigest()
        if expected is not None and h != _sha(expected):
            raise ValueError('Original evidence SHA differs: ' + str(path))
        binding = {'sha256': h, 'size': len(raw)}
        previous = self.bindings.setdefault(str(path), binding)
        _same(previous, binding, 'Original evidence changed between reads')
        return raw

    def json(self, relative, expected=None):
        raw = self.raw(relative, expected)
        if relative.endswith('.gz'):
            import gzip
            raw = gzip.decompress(raw)
        return json.loads(raw)


def _candidate_artifacts(read, candidate, plan, protocol, scenes):
    """Only original byte/JSON identities; do not reopen a runtime or source loader."""
    if candidate.get('status') != 'frozen_for_independent_acceptance' or candidate.get('test_fixture') is not False:
        raise ValueError('Not the original genuine frozen candidate')
    for name, expected in (('actor.npz', plan['bindings']['actor_sha256']),
                           ('program.json', plan['bindings']['program_sha256'])):
        item = candidate['artifacts'][name]
        if item['path'] != name or item['sha256'] != expected or item['size'] != len(read.raw(name, expected)):
            raise ValueError('Copied Actor/program differs from frozen candidate artifact')
    for name, value in (('protocol.json', protocol), ('scenarios.json', scenes)):
        item = candidate['artifacts'][name]
        if item['path'] != name:
            raise ValueError('Candidate JSON artifact path differs')
        raw = read.absolute(producer.CANDIDATE / name, item['sha256'])
        _same(len(raw), item['size'], 'Candidate JSON artifact size differs')
        _same(json.loads(raw), value, 'Prepared protocol/scenarios differ from frozen candidate')


def _initial_inputs(read, plan, fixture):
    _same(plan['version'], PRODUCER_VERSION, 'Wrong development producer version')
    if producer.VERSION != PRODUCER_VERSION:
        raise ValueError('Reader requires its explicit producer version')
    _same(plan['test_fixture'], fixture, 'Explicit fixture opt-in differs')
    _flags(plan)
    _same(plan['contract'], producer.contract(), 'Development query contract differs')
    if not re.fullmatch('[A-Za-z0-9_-]{3,80}', plan['execution_id']):
        raise ValueError('Malformed collection execution identity')
    original_root = _absolute(plan['output_identity'])
    if str(original_root) != plan['output_identity']:
        raise ValueError('Original output identity differs')
    expected_sources = producer.sources()
    _same(plan['sources'], expected_sources, 'Producer source closure changed or incomplete')
    for name, expected in expected_sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Nonrepository producer source path')
        read.absolute(ROOT / name, expected)
    bindings = plan['bindings']
    for key, value in bindings.items():
        if key.endswith('sha256') or key == 'runtime_signature':
            _sha(value)
    _same(bindings['sources_sha256'], digest(plan['sources']), 'Producer sources digest differs')
    _same(bindings['contract_sha256'], digest(plan['contract']), 'Producer contract digest differs')
    _same(bindings['runtime_family'], 'retained_beta197', 'Wrong runtime family')
    _same(bindings['runtime_version'], RUNTIME_VERSION, 'Wrong runtime version')
    pools = read.json('pools.json'); scenes = read.json('scenarios.json'); protocol = read.json('protocol.json')
    _same(digest(pools), plan['pools_sha256'], 'Development pools changed')
    _same(digest(scenes), bindings['scenario_manifest_sha256'], 'Scenario manifest changed')
    _same(digest(protocol), bindings['protocol_sha256'], 'Protocol changed')
    _same(scenes['configuration'], plan['configuration'], 'Physical configuration differs')
    if scenes.get('test_fixture', False) is not fixture or set(pools) != set(POOLS):
        raise ValueError('Pool or scenario fixture identity differs')
    horizon = plan['configuration']['horizon']
    if type(horizon) is not int or (fixture and not 1 <= horizon <= 3) or (not fixture and horizon != 120):
        raise ValueError('Invalid collection horizon')
    sizes = {p: len(pools[p]) for p in POOLS}
    _same(sizes, {'train': 1, 'selection': 1} if fixture else producer.COUNTS, 'Fixed pool matrix differs')
    fingerprints = {}
    for pool, split in (('train', 'train'), ('selection', 'extraction')):
        _same(pools[pool], scenes['splits'][split][:sizes[pool]], 'Development pool prefix/order differs')
        values = [producer._physical_fingerprint(s, plan['configuration']) for s in pools[pool]]
        forbidden = {s['fingerprint'] for name, entries in scenes['splits'].items() if name != split for s in entries}
        if len(values) != len(set(values)) or set(values) & forbidden:
            raise ValueError('Initial states overlap or repeat across splits')
        if any(not s['id'].startswith(split + '_') or s.get('test_fixture', False) is not fixture for s in pools[pool]):
            raise ValueError('Not original train/extraction initial states')
        fingerprints[pool] = set(values)
    if fingerprints['train'] & fingerprints['selection']:
        raise ValueError('Train and selection initial states overlap')
    if fixture:
        _same(plan['source'], {'kind': 'synthetic_fixture_not_production'}, 'Synthetic source is not explicit')
        _same(plan['pool_source'], {'synthetic': True}, 'Synthetic pool source is not explicit')
    else:
        _same(plan['source'], {'kind': 'genuine_frozen_candidate', 'directory': str(producer.CANDIDATE),
            'manifest_sha256': producer.CANDIDATE_SHA256}, 'Different frozen development source')
        _same(plan['pool_source']['path'], str(producer.POOLS), 'Different original development pools')
        _same(plan['pool_source']['sha256'], producer.POOLS_SHA256, 'Different original pool anchor')
        candidate = json.loads(read.absolute(producer.CANDIDATE / 'manifest.json', producer.CANDIDATE_SHA256))
        _candidate_artifacts(read, candidate, plan, protocol, scenes)
        original = json.loads(read.absolute(producer.POOLS, producer.POOLS_SHA256))
        _same(read.bindings[str(producer.POOLS)]['size'], plan['pool_source']['size'], 'Original pool size differs')
        for pool in POOLS:
            _same(original[pool][:sizes[pool]], pools[pool], 'Frozen source pool prefix differs')
    contexts = producer._contexts(pools, horizon)
    _same(contexts, plan['contexts'], 'Fixed context ordering/roles/seeds differs')
    caps = {'base': len(contexts) * horizon, 'counterfactual': len(contexts) * ((horizon + 9)//10) * 5}
    _same(plan['caps'], caps, 'Fixed auxiliary budgets differ')
    _same(plan['maximum_environment_steps'], sum(caps.values()), 'Total auxiliary budget differs')
    configuration = {**plan['configuration'], 'reward': RewardConfig(**plan['configuration']['reward'])}
    names = tuple(observation_names(WarehouseConfig(**configuration))) + tuple(HISTORY_FEATURE_NAMES)
    if len(names) != 197:
        raise ValueError('Expected exact observed197 feature layout')
    _same(list(names), plan['feature_names'], 'Actual 197 feature order differs')
    with np.load(io.BytesIO(read.raw('actor.npz', bindings['actor_sha256'])), allow_pickle=False) as npz:
        expected = {'0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias', 'metadata_json'}
        if len(npz.files) != len(expected) or set(npz.files) != expected:
            raise ValueError('Six-array Actor export required')
        metadata = json.loads(str(npz['metadata_json'].item()))
        arrays = {name: npz[name].copy() for name in expected - {'metadata_json'}}
    _same(metadata, plan['actor_metadata'], 'Actual Actor metadata differs')
    _same(digest(metadata), bindings['actor_metadata_sha256'], 'Actor metadata digest differs')
    _same(metadata['feature_names'], list(names), 'Actor feature order differs')
    _same(metadata['actions'], list(ACTIONS), 'Actual five-action order differs')
    for key, expected in (('obs_dim', 197), ('state_dim', 354), ('test_fixture', fixture),
                          ('action_masks', False), ('runtime_action_override', False), ('public_feedback_mode', 'observed')):
        _same(metadata.get(key), expected, 'Actor shape/fixture/action metadata differs: ' + key)
    _same(actor_parameter_sha256(SimpleNamespace(weights=arrays)), bindings['actor_parameters_sha256'], 'Actual NPZ six-array semantic hash differs')
    _same(metadata['protocol_sha256'], bindings['protocol_sha256'], 'Actor protocol binding differs')
    # The producer's explicit tiny fixture remaps its synthetic scene IDs into
    # development pools; production retains the original trained scene hash.
    if not fixture:
        _same(metadata['scenario_manifest_sha256'], bindings['scenario_manifest_sha256'], 'Actor scene binding differs')
    _same(bindings['runtime_signature'], digest({'version': RUNTIME_VERSION, 'actor_sha256': bindings['actor_sha256'],
        'protocol_sha256': bindings['protocol_sha256'], 'actor_metadata_sha256': bindings['actor_metadata_sha256'],
        'configuration': plan['configuration'], 'sources': runtime_sources()}), 'Actual runtime signature differs')
    read.raw('program.json', bindings['program_sha256'])  # Byte binding only; tree never queried or used as a label.
    return pools, contexts, caps, original_root


def _decision(value, frame, bindings):
    if type(value) is not dict or value.get('role') != 'robot_2' or type(value.get('frame')) is not int or value['frame'] != frame:
        raise ValueError('NN decision role/frame differs')
    _same(value['actor_sha256'], bindings['actor_sha256'], 'Decision Actor differs')
    _same(value['runtime_signature'], bindings['runtime_signature'], 'Decision runtime differs')
    obs = np.asarray(value['observation'], dtype=np.float32)
    logits = np.asarray(value['logits'], dtype=np.float32)
    probabilities = np.asarray(value['probabilities'], dtype=np.float32)
    if (obs.shape != (197,) or logits.shape != (5,) or probabilities.shape != (5,)
            or not all(np.isfinite(x).all() for x in (obs, logits, probabilities))):
        raise ValueError('Nonfinite or malformed saved NN arrays')
    _same(digest(obs.tolist()), value['observation_sha256'], 'Observation SHA differs')
    expected = np.exp(logits - logits.max()); expected /= expected.sum()
    if not np.allclose(probabilities, expected, rtol=1e-6, atol=1e-7):
        raise ValueError('Saved NN probabilities differ from float32 softmax')
    if (value['neural_action'] != ACTIONS[int(expected.argmax())]
            or value['neural_action'] != ACTIONS[int(probabilities.argmax())]
            or value['logits_argmax_diagnostic'] != ACTIONS[int(logits.argmax())]):
        raise ValueError('Saved NN action differs from actual probability argmax')
    return obs, probabilities, value['neural_action']


def _empty():
    return {name: [] for name in ('observations', 'probabilities', 'actions', 'episode_ids',
        'scene_fingerprints', 'groups', 'kind', 'row_sources', 'pairs')}


def read_completed(root, *, expected_plan_sha256, expected_manifest_sha256, allow_test_fixture=False):
    """Read complete original development queries; external anchors and fixture opt-in are mandatory."""
    _sha(expected_plan_sha256); _sha(expected_manifest_sha256)
    if type(allow_test_fixture) is not bool:
        raise ValueError('Explicit boolean fixture opt-in required')
    read = _Read(_absolute(root)); reader_sources = sources()
    plan = read.json('plan.json', expected_plan_sha256)
    manifest = read.json('collection/manifest.json', expected_manifest_sha256)
    if manifest.get('version') != PRODUCER_VERSION or manifest.get('status') != 'completed':
        raise ValueError('Only complete explicit new-producer data are accepted')
    _same(manifest['plan_sha256'], expected_plan_sha256, 'Manifest binds another plan')
    _same(manifest['test_fixture'], allow_test_fixture, 'Manifest fixture identity differs')
    _flags(manifest)
    pools, contexts, caps, original_root = _initial_inputs(read, plan, allow_test_fixture)
    _same(manifest['bindings'], plan['bindings'], 'Manifest Actor/input bindings differ')
    _same(manifest['contexts'], contexts, 'Manifest context matrix differs')
    state = read.json('collection/state.json')
    for k, expected in (('version', PRODUCER_VERSION), ('status', 'completed'), ('pending', None),
                        ('plan_sha256', expected_plan_sha256), ('manifest_sha256', expected_manifest_sha256)):
        _same(state.get(k), expected, 'Incomplete or changed terminal ledger: ' + k)
    if any((read.root / ('collection/' + name)).exists() or (read.root / ('collection/' + name)).is_symlink()
           for name in ('pending.json', 'failure.json')):
        raise ValueError('Pending or failed execution cannot become a dataset')
    labels = read.json('collection/label_index.json', manifest['label_index_sha256'])
    report = read.json('collection/report.json', manifest['report_sha256']); _flags(report)
    _same(report['version'], PRODUCER_VERSION, 'Report producer differs')
    entries = manifest['entries']; episodes = manifest['episodes']
    if len(labels) != len(entries) or len(episodes) != len(contexts):
        raise ValueError('Missing label/episode matrix')
    for folder in ('steps', 'acks', 'reservations', 'confirmations'):
        suffix = '.json.gz' if folder == 'steps' else '.json'
        expected_names = {f'{i:06d}' + suffix for i in range(len(entries))}
        directory = read.root / 'collection' / folder
        if directory.is_symlink() or {p.name for p in directory.iterdir()} != expected_names:
            raise ValueError('Unregistered or missing original collection operation files')
    data = {pool: _empty() for pool in POOLS}; hashes = {pool: set() for pool in POOLS}
    anchors = {}; branches = {}; counts = Counter(); label_counts = Counter(); entry_index = 0
    for episode_index, (context, episode) in enumerate(zip(contexts, episodes)):
        for k, v in context.items(): _same(episode[k], v, 'Episode context differs')
        if episode['first_entry'] != entry_index or episode.get('terminal') is not True:
            raise ValueError('Episode matrix is incomplete or not terminal')
        end = episode['end_entry_exclusive']
        if type(end) is not int or not entry_index < end <= len(entries):
            raise ValueError('Invalid episode entry interval')
        pool = context['pool']; scene = pools[pool][context['pool_scene_index']]
        base_steps = 0; last_after = None; done = False
        episode_id = f"{plan['execution_id']}:{pool}:{context['partner']}:{context['pool_scene_index']:04d}"
        while entry_index < end:
            i = entry_index; entry_index += 1; entry = entries[i]
            relative = f'steps/{i:06d}.json.gz'
            _same(entry['path'], relative, 'Noncanonical raw entry path')
            d = read.json('collection/' + relative, entry['sha256'])
            _same(read.json(f'collection/acks/{i:06d}.json'), entry, 'Original ACK differs')
            phase = d['phase']; frame = d['frame']
            if phase not in caps or type(frame) is not int or d['producer_version'] != PRODUCER_VERSION:
                raise ValueError('Wrong raw phase/frame/producer')
            _same(entry, {'path': relative, 'sha256': entry['sha256'], 'phase': phase,
                'operation_id': f"{plan['execution_id']}:{phase}:{i:06d}", 'actual_steps': 1}, 'Entry operation differs')
            for k, v in context.items(): _same(d[k], v, 'Raw episode context differs')
            semantics = 'actual_frozen_NN_command_submitted' if phase == 'base' else 'next_frozen_NN_query_NOT_submitted_or_executed'
            _same(d['label_semantics'], semantics, 'NN query label semantics differ')
            _same(d['operation_id'], entry['operation_id'], 'Raw operation differs')
            before, after = d['before']['state'], d['after']['state']
            if (type(before['frame']) is not int or type(after['frame']) is not int or before['frame'] != frame
                    or after['frame'] != frame + 1 or before['terminated'] is not False or before['truncated'] is not False):
                raise ValueError('Raw transition frame/terminal state differs')
            _same(digest(d['before']), d['before_sha256'], 'Original prestate SHA differs')
            _same([d['terminated'], d['truncated']], [after['terminated'], after['truncated']], 'Raw terminal flags differ')
            current = _decision(d['decision'], frame, plan['bindings'])
            _same(d['submitted_actions']['robot_2'], current[2], 'NN command was overwritten')
            _same(d['submitted_actions'], d['info']['requested_actions'], 'Physical requested command differs')
            if set(d['submitted_actions']) != {'robot_1', 'robot_2'} or any(x not in ACTIONS for x in d['submitted_actions'].values()):
                raise ValueError('Unknown submitted role/action')
            _same(d['executed_actions'], d['info']['executed_actions'], 'Executed action evidence differs')
            _same(d['events'], d['info']['events'], 'Event evidence differs')
            for name, snapshot in (('before_view', before), ('after_view', after)):
                _same(producer.audit.physical_projection(d[name]), producer.audit.physical_projection(snapshot), 'Saved physical view differs')
            _same(d['groups'], producer.audit._groups(d['before']), 'Prestate critical groups differ')
            ctx = {**context, 'producer_version': PRODUCER_VERSION, 'phase': phase, 'frame': frame,
                'label_semantics': semantics, 'operation_id': entry['operation_id'], 'before_sha256': d['before_sha256'],
                'submitted_actions': d['submitted_actions'], 'reserved_steps': 1}
            if phase == 'counterfactual': ctx['intervention_action'] = d['intervention_action']
            reservation = {'version': PRODUCER_VERSION, 'context': ctx, 'permanent': True, 'refund_allowed': False}
            _same(read.json(f'collection/reservations/{i:06d}.json'), reservation, 'Permanent reservation differs')
            completion = {**ctx, 'actual_steps': 1, 'record_path': str(original_root / 'collection' / relative),
                'record_sha256': entry['sha256'], 'status': 'executed_unacknowledged'}
            _same(read.json(f'collection/confirmations/{i:06d}.json'),
                {'version': PRODUCER_VERSION, 'reservation': reservation, 'completion': completion}, 'Durable confirmation differs')
            counts[phase] += 1
            if counts[phase] > caps[phase]: raise ValueError('Permanent phase cap exceeded')
            anchor = (pool, context['partner'], context['pool_scene_index'], frame)
            if phase == 'base':
                if done or frame != base_steps or d['next_decision'] is not None:
                    raise ValueError('Base trajectory sequence differs')
                if base_steps:
                    _same(digest(d['before']), last_after, 'Base snapshot chain differs')
                else:
                    for field in ('state', 'rng', 'episode_counter'):
                        _same(d['before'][field], scene['snapshot'][field], 'Base initial snapshot differs')
                base_steps += 1; last_after = digest(d['after']); done = bool(after['terminated'] or after['truncated'])
                if not frame % 10 and d['groups']:
                    anchors[anchor] = {'before_sha256': d['before_sha256'], 'decision_sha256': digest(d['decision']),
                        'groups': d['groups'], 'scene_fingerprint': scene['fingerprint']}
                field = 'decision'; query = current
            else:
                if anchor not in anchors or d['intervention_action'] not in ACTIONS:
                    raise ValueError('Unregistered counterfactual anchor/action')
                _same(d['before_sha256'], anchors[anchor]['before_sha256'], 'Counterfactual prestate differs')
                _same(digest(d['decision']), anchors[anchor]['decision_sha256'], 'Counterfactual initial NN decision differs')
                _same(d['submitted_actions']['robot_1'], d['intervention_action'], 'Player intervention differs')
                if (d['next_decision'] is None) != bool(after['terminated'] or after['truncated']):
                    raise ValueError('Terminal counterfactual query differs')
                field = 'next_decision' if d['next_decision'] is not None else None
                query = _decision(d['next_decision'], after['frame'], plan['bindings']) if field else None
            expected_label = producer._label(entry, d)
            _same(labels[i], expected_label, 'Saved neural label index differs')
            groups = expected_label['groups']
            row_index = None
            if field:
                row_index = len(data[pool]['actions']); obs, probabilities, action = query
                for k, value in (('observations', obs), ('probabilities', probabilities), ('actions', action),
                    ('episode_ids', episode_id), ('scene_fingerprints', scene['fingerprint']), ('groups', groups), ('kind', phase),
                    ('row_sources', {'path': str(read.root / 'collection' / relative), 'sha256': entry['sha256'], 'field': field,
                        'executed': phase == 'base', 'producer_version': PRODUCER_VERSION, 'record_frame': frame,
                        'decision_frame': d[field]['frame'], 'role': 'robot_2'})):
                    data[pool][k].append(value)
                hashes[pool].add(sha256(obs.tobytes()).hexdigest()); label_counts[field] += 1
            else:
                label_counts['terminal_no_label'] += 1
            if phase == 'counterfactual':
                action = d['intervention_action']; branch = branches.setdefault(anchor, {})
                if action in branch: raise ValueError('Duplicate action at branch anchor')
                branch[action] = {'row_index': row_index, 'physical_sha256': digest(producer.audit.physical_projection(d['after_view'])),
                    'neural_action': None if query is None else query[2]}
        if not done or base_steps != episode['base_steps'] or base_steps > context['horizon']:
            raise ValueError('Missing complete base episode')
    if entry_index != len(entries) or set(branches) != set(anchors):
        raise ValueError('Incomplete original entry/branch matrix')
    pair_counts = Counter()
    for anchor, origin in anchors.items():
        branch = branches[anchor]
        if set(branch) != set(ACTIONS): raise ValueError('Five-action branch matrix incomplete')
        base = branch['WAIT']
        for action in ACTIONS[:-1]:
            changed = branch[action]; pair_counts['all_potential_pairs'] += 1
            if base['row_index'] is None or changed['row_index'] is None:
                pair_counts['terminal_endpoint_pairs_without_labels'] += 1
                continue
            pair = {'baseline_index': base['row_index'], 'changed_index': changed['row_index'],
                'physical_effect': base['physical_sha256'] != changed['physical_sha256'],
                'nn_changed': base['neural_action'] != changed['neural_action'], 'groups': deepcopy(origin['groups']),
                'scene_fingerprint': origin['scene_fingerprint'], 'partner': anchor[1], 'anchor': list(anchor), 'player_action': action}
            data[anchor[0]]['pairs'].append(pair); pair_counts['nonterminal_pairs'] += 1
            pair_counts['effective_pairs'] += pair['physical_effect'] and pair['nn_changed']
    if hashes['train'] & hashes['selection']:
        raise ValueError('Exact actual observed197 inputs overlap train and selection')
    expected_counts = {'base_steps': counts['base'], 'counterfactual_steps': counts['counterfactual'],
        'base_attempts': counts['base'], 'counterfactual_attempts': counts['counterfactual'],
        'acknowledged_steps': len(entries), 'numpy_logits_calls': label_counts['decision'] + label_counts['next_decision'],
        'numpy_logits_rows': 2 * (label_counts['decision'] + label_counts['next_decision']),
        'neural_updates': 0, 'tree_fits': 0, 'torch_loads': 0}
    execution = {'counts': expected_counts, 'pending_operation': None, 'automatic_retry': False, 'caps': caps, 'accounting_complete': True}
    _same(manifest['execution'], execution, 'Actual execution accounting differs')
    _same(report, {'version': PRODUCER_VERSION, 'scope': 'development_queries_not_independent_explanation_validation',
        'episodes': len(episodes), 'execution': execution, 'base_labels': label_counts['decision'],
        'counterfactual_query_labels': label_counts['next_decision'], 'terminal_branches_without_next_label': label_counts['terminal_no_label'],
        'program_teacher_labels': 0, 'nn_command_overrides': 0, **producer.FLAGS}, 'Raw labels/steps do not reconstruct original report')
    for field in ('reserved', 'acknowledged'):
        _same(state[field], {p: counts[p] for p in caps}, 'Terminal ledger accounting differs')
    _same(reader_sources, sources(), 'Reader or producer source changed during read')
    for pool in POOLS:
        data[pool]['observations'] = np.asarray(data[pool]['observations'], dtype=np.float32).reshape(-1, 197)
        data[pool]['probabilities'] = np.asarray(data[pool]['probabilities'], dtype=np.float32).reshape(-1, 5)
    return {'version': VERSION, 'plan': deepcopy(plan), 'manifest': deepcopy(manifest), 'data': data,
        'input_bindings': deepcopy(read.bindings), 'producer_sources': deepcopy(plan['sources']), 'reader_sources': reader_sources,
        'record_statistics': {'physical_steps': dict(counts), 'labels': dict(label_counts), 'pairs': dict(pair_counts),
            'exact_cross_pool_observation_overlap': 0, 'rows_removed': 0}, 'test_fixture': allow_test_fixture,
        'reader_execution_counts': dict(ZERO), 'scope': 'saved_actual_NN_development_queries_not_independent_replay',
        **producer.FLAGS}
