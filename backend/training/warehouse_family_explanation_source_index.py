"""Metadata-only source verification for one registered fresh explanation pool.

The external inventory anchors 54 original files. A single explicitly bound
expanded-corpus chain adds its historical 128-scene collection. Private snapshot
projection uses the frozen physical fingerprint function on plain data, never
an environment. No final snapshot, Actor, trajectory or performance is output.
This verifies exclusion provenance, not model/explanation/release capability.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

from .warehouse_native_common import ROOT, canonical, digest, file_hash
from .warehouse_native_shutdown_result import _json_bytes
from . import warehouse_family_fresh_explanation_run as fresh
from env.warehouse.domain import collaborative_study_config
from env.warehouse.layouts import get_map_layout
from env.warehouse_native.scenarios import SCENARIO_VERSION, SPLIT_NAMES, scenario_fingerprint

VERSION = 'warehouse-family-explanation-source-index.v1'
INVENTORY_VERSION = 'warehouse-family-fresh-source-inventory.readonly.v1'
INVENTORY_SHA256 = '7be69170e685afa74bc412153f4fec004f73aacb062e6deb7d11511a340b117d'
ORIGIN_ROOT = ROOT/'output/warehouse_native'
ORIGINAL = 'native_cycle_500k_candidate_20260909/scenarios.json'
PAIR = 'shutdown_feedback_pair_20260909_v2'
INITIAL = 'shutdown_initial_rcpd_beta1_20260909_r1'
EXPANDED = 'expanded_initial_rcpd_beta1_20260909'
TREE = 'native_cycle_500k_tree_20260909'
INTERVENTION = 'native_cycle_500k_intervention_development_20260909'
ACTIVE = 'feedback_cycle_500k_pair_20260909'
HELDOUT = 'native_cycle_500k_acceptance_20260909/heldout'
QUESTION = 'family_question_pool_preparation_20260909'
BANK = 'native_cycle_500k_acceptance_20260909/bank'
LEGACY_PAIR_VERSION = 'warehouse-native-own-shutdown-feedback-run.v2'
LEGACY_PAIR_SOURCE = 'backend/training/warehouse_native_shutdown_feedback_run.py'
LEGACY_COMPLETION_KEYS = frozenset(('auxiliary', 'control_terminal_tree_extracted', 'counts',
    'explanation_qualified', 'formal_ready', 'initial_tree_budget_is_separate', 'ledger', 'reports',
    'status', 'total_interaction_equal', 'until'))
CURRICULA = ('v2-foundation-seed260908', 'r1-foundation-seed260908')
LATER = ('public_history_pair_prepared_20260908', 'partner_mix_20260908', 'delivery_credit_20260908',
    'continuation_01_20260908', 'continuation_02_own_20260909', 'continuation_03_own_20260909',
    'native_cycle_100k_20260909', 'native_cycle_500k_20260909')
POOL_FILES = {
    INITIAL: ('scene_pools.json', ('plan.json', 'manifest.json')),
    PAIR: ('refresh_pools.json', ('prepared.json', 'completion_0250000.json')),
    'shutdown_control_terminal_tree_20260909': ('pools.json', ('plan.json', 'manifest.json', 'report.json')),
    TREE: ('pools.json', ('plan.json', 'manifest.json', 'report.json', 'collection/manifest.json')),
    INTERVENTION: ('pools.json', ('plan.json', 'collection/manifest.json')),
    ACTIVE: ('refresh_pools.json', ('prepared.json',)),
}


def required_origins():
    names = ['native_cycle_500k_candidate_20260909/actor_bindings.json',
        'v1-foundation-seed260908/scenarios.json', 'r1-foundation-seed260908/scenarios.json', ORIGINAL,
        QUESTION+'/exclusion_pools.json', BANK+'/inputs/exclusion_pools.json']
    names += [f'{r}/{n}' for r in CURRICULA for n in ('curriculum_bank.json', 'protocol.json', 'run.json', 'episodes.jsonl')]
    names += ['r1-foundation-seed260908/'+n for n in ('curriculum_binding.json', 'curriculum_quality.json')]
    for r, (pool, anchors) in POOL_FILES.items(): names += [r+'/'+pool, *(r+'/'+n for n in anchors)]
    names += [EXPANDED+'/'+n for n in ('plan.json', 'manifest.json')]
    names += [HELDOUT+'/audit/'+n for n in ('inputs.json', 'manifest.json')]
    names += [HELDOUT+'.receipt.json', HELDOUT+'/plan.json', QUESTION+'/preparation.json', QUESTION+'/pool_scenes.json',
        'native_cycle_500k_bank_retirement_20260909.json', BANK+'/prepared.json']
    names += [r+'/prepared.json' for r in LATER]
    return tuple(names)


def sources():
    paths = (Path(__file__), Path(fresh.__file__), ROOT/'env/warehouse_native/scenarios.py',
        ROOT/'env/warehouse/layouts.py', ROOT/'env/warehouse/domain.py',
        ROOT/'backend/training/warehouse_native_common.py', ROOT/'backend/training/warehouse_native_shutdown_result.py',
        ROOT/LEGACY_PAIR_SOURCE)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def _require(value, message):
    if not value: raise ValueError(message)


def _same(a, b, message):
    _require(canonical(a) == canonical(b), message)


def _sha(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'Explicit lowercase external SHA256 required')
    return value


def _safe(root, relative):
    p = Path(relative)
    _require(type(relative) is str and relative and not p.is_absolute() and str(p) == relative
        and all(x not in ('', '.', '..') for x in relative.split('/')) and '\\' not in relative,
        'Safe relative metadata path required')
    path = root/p
    _require(path.resolve() == path and path.is_file(), 'Linked, missing or nonregular metadata source')
    return path


def _bytes(path, expected, size=None):
    _sha(expected); path = Path(path)
    _require(path.resolve() == path and path.is_file(), 'Unlinked regular source required')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    _require(before.st_size == after.st_size == len(raw) and before.st_mtime_ns == after.st_mtime_ns,
        'Source changed during reading')
    _require(sha256(raw).hexdigest() == expected and (size is None or (type(size) is int and size == len(raw))), 'External source bytes differ')
    return raw


class _Origins:
    def __init__(self, inventory, fixture):
        self.root = Path(inventory.get('origin_root', ORIGIN_ROOT)).expanduser().absolute()
        _require(self.root.resolve() == self.root, 'Source root must not traverse aliases or symlinks')
        if fixture:
            _require(ROOT != self.root and ROOT not in self.root.parents, 'Synthetic origins must be outside production workspace')
        else: _same(str(self.root), str(ORIGIN_ROOT), 'Production origin root differs')
        entries = inventory['input_bindings']; expected = {str(self.root/n) for n in required_origins()}
        _require(len(expected) == 54 and set(entries) == expected, 'Complete 54-file direct source catalog required')
        self.entries = entries; self.records = {}; self.derived_records = {}
        # Every direct external byte binding is checked even if an artifact is
        # subsequently used only as a historical status/producer receipt.
        for name, binding in entries.items():
            _require(set(binding) >= {'sha256', 'size', 'mtime_ns'}, 'Incomplete original binding')
            self._read(Path(name), binding['sha256'], binding['size'], direct=True)

    def _read(self, path, anchor, size=None, direct=False):
        raw = _bytes(path, anchor, size); stat = path.stat()
        binding = {'path': str(path), 'sha256': sha256(raw).hexdigest(), 'size': len(raw), 'mtime_ns': stat.st_mtime_ns}
        previous = self.records.get(str(path))
        if previous is not None: _same(previous, binding, 'Previously checked source changed')
        self.records[str(path)] = binding
        if not direct and str(path) not in self.entries: self.derived_records[str(path)] = binding
        return raw

    def raw(self, relative):
        path = _safe(self.root, relative); binding = self.entries[str(path)]
        return self._read(path, binding['sha256'], binding['size'], direct=True)

    def json(self, relative): return _json_bytes(self.raw(relative))

    def derived(self, relative, anchor):
        return _json_bytes(self._read(_safe(self.root, relative), anchor))

    def binding(self, relative, **extra):
        value = deepcopy(self.records[str(self.root/relative)])
        return {**value, **extra}

    def unchanged(self):
        for value in self.records.values():
            _bytes(value['path'], value['sha256'], value['size'])
            _require(Path(value['path']).stat().st_mtime_ns == value['mtime_ns'], 'Verified source mtime changed')


def _configuration(value, fixture):
    horizon = value.get('horizon')
    _require(type(horizon) is int and (1 <= horizon <= 3 if fixture else horizon == 120), 'Explicit fixed public horizon required')
    _same(value, asdict(collaborative_study_config(horizon=horizon)), 'Original public configuration differs')


def _snapshot_fingerprint(snapshot, configuration, *, initial):
    """Private plain-data projection. Never return a snapshot or its fields."""
    _same(snapshot.get('configuration'), configuration, 'Private source configuration differs')
    state = snapshot['state']; frame = state.get('frame')
    _same(snapshot.get('version'), 'warehouse-native-physics-v1', 'Original native snapshot producer differs')
    _require(type(frame) is int and 0 <= frame <= configuration['horizon']
        and (not initial or frame == 0) and state.get('terminated') is False and state.get('truncated') is False,
        'Source start must be a live registered physical state')
    _require(not any(k.startswith('public_feedback') for k in snapshot), 'Original raw physical snapshot required')
    layout = get_map_layout(configuration['map_layout_id']); agents = state['agents']; tasks = state['tasks']
    _require(len(agents) == 2 and [a['agent_id'] for a in agents] == ['robot_1', 'robot_2'], 'Source robot identity differs')
    for agent in agents:
        position = agent['position']; battery = agent['battery']
        _require(isinstance(position, (list, tuple)) and len(position) == 2 and all(type(x) is int for x in position)
            and layout.is_passable(tuple(position)) and type(battery) in (int, float) and math.isfinite(battery)
            and 0 <= battery <= 100 and type(agent['active']) is bool, 'Invalid private physical robot fields')
    _require(len({tuple(a['position']) for a in agents}) == 2 and isinstance(tasks, list) and bool(tasks), 'Source physical state is malformed')
    proxy = SimpleNamespace(state=SimpleNamespace(agents=[SimpleNamespace(**a) for a in agents],
        tasks=[SimpleNamespace(**t) for t in tasks]), layout=layout, config=SimpleNamespace(**configuration))
    return scenario_fingerprint(proxy)


def _scene_fingerprints(entries, configuration, *, split=None):
    _require(isinstance(entries, list) and bool(entries), 'A nonempty source pool is required')
    values = []; ids = set()
    for i, scene in enumerate(entries):
        _require(type(scene.get('id')) is str and scene['id'] not in ids, 'Duplicate or missing source scene ID')
        if split is not None: _same(scene['id'], f'{split}_{i:04d}', 'Original ordered split identity differs')
        ids.add(scene['id']); actual = _snapshot_fingerprint(scene['snapshot'], configuration, initial=True)
        _same(actual, _sha(scene['fingerprint']), 'Declared private source fingerprint differs')
        values.append(actual)
    _require(len(set(values)) == len(values), 'Duplicate physical starts within a registered pool')
    return values


def _index(scope, pools, origins, scenario_sha, fixture, **metadata):
    _require(bool(origins), 'Original origin byte bindings required')
    return {'version': fresh.INDEX_VERSION, 'scope': scope, 'test_fixture': fixture,
        'source_scenario_manifest_sha256': scenario_sha, 'pools': pools,
        'counts': {name: len(values) for name, values in pools.items()}, 'origin_bindings': origins, **metadata}


def _verify_original(book, inventory, fixture):
    # All original (including final) state parsing remains inside this private
    # verification function. Only index data and ordinary public config escape.
    scenarios = book.json(ORIGINAL); configuration = scenarios['configuration']; _configuration(configuration, fixture)
    scenario_sha = digest(scenarios)
    _same(scenario_sha, inventory['source_scenario_manifest_sha256'], 'Original Actor scenario semantic SHA differs')
    actor = book.json('native_cycle_500k_candidate_20260909/actor_bindings.json')
    _same(actor['scenario_manifest_sha256'], scenario_sha, 'Actual Actor training identity differs')
    _same(scenarios['version'], SCENARIO_VERSION, 'Original scenario producer differs')
    splits = scenarios['splits']; _same(sorted(splits), sorted(SPLIT_NAMES), 'All original seven splits are required')
    if 'split' in scenarios: _same(scenarios['split'], splits, 'Original duplicate split alias differs')
    pools = {name: _scene_fingerprints(splits[name], configuration, split=name) for name in SPLIT_NAMES}
    counts = {name: len(values) for name, values in pools.items()}
    _same(scenarios['counts'], counts, 'Original registered counts differ')
    if not fixture: _same(counts, fresh.REGISTERED_COUNTS, 'Production seven-pool counts differ')
    _require(sum(counts.values()) == len(set().union(*(set(v) for v in pools.values()))), 'Original splits overlap physically')
    for relative in ('v1-foundation-seed260908/scenarios.json', 'r1-foundation-seed260908/scenarios.json'):
        _same(digest(book.json(relative)), scenario_sha, 'Historical original source manifest differs')
    for relative in (QUESTION+'/exclusion_pools.json', BANK+'/inputs/exclusion_pools.json'):
        _same(book.json(relative), splits, 'Original seven-pool exclusion copy differs')
    origins = {'scenarios': book.binding(ORIGINAL, semantic_sha256=scenario_sha),
        'v1_scenarios': book.binding('v1-foundation-seed260908/scenarios.json'),
        'r1_scenarios': book.binding('r1-foundation-seed260908/scenarios.json'),
        'actor_bindings': book.binding('native_cycle_500k_candidate_20260909/actor_bindings.json')}
    for name in LATER:
        prepared = book.json(name+'/prepared.json')
        _same(prepared['scenario_manifest_sha256'], scenario_sha, 'Later learning changed original scenario identity')
        _same(prepared['version'], inventory['later_training_source_bindings'][name]['version'], 'Historical learning producer differs')
        origins[name] = book.binding(name+'/prepared.json', producer_version=prepared['version'])
    index = _index('source_scenarios', pools, origins, scenario_sha, fixture,
        configuration=configuration, verification_scope='all declared physical fingerprints privately recomputed; no environment/final episode')
    # The returned source projections are only IDs/fingerprints, never states.
    train_ids = {scene['id']: scene['fingerprint'] for scene in splits['train']}
    return index, train_ids


def _verify_curriculum(book, inventory, root, original, train_ids, fixture):
    bank = book.json(root+'/curriculum_bank.json'); protocol = book.json(root+'/protocol.json'); run = book.json(root+'/run.json')
    _same(bank['scenario_manifest_sha256'], original['source_scenario_manifest_sha256'], 'Curriculum source scenario identity differs')
    _require(bank['source_split'] == 'train' and bank['teacher_action_labels'] is False and bank['manual_state_edits'] is False,
        'Curriculum must remain genuine training starts without teacher labels or manual states')
    expected = inventory['curricula'][root]; _same(bank['version'], expected['actual_producer_version'], 'Historical curriculum producer differs')
    _same(bank.get('selection_version'), expected['selection_version'], 'Independent curriculum selector identity differs')
    entries = bank['entries']; _require(len(entries) == (expected['entries'] if fixture else 512), 'Complete original curriculum bank required')
    _same(bank['counts'], dict(Counter(e['category'] for e in entries)), 'Complete category counts differ')
    if not fixture: _same(sorted(bank['counts'].values()), [128]*4, 'Original four curriculum categories differ')
    heldout = set().union(*(set(values) for name, values in original['pools'].items() if name != 'train'))
    values = []; by_id = {}; complete_ids = set()
    for entry in entries:
        _require(entry['source_split'] == 'train' and entry['source_id'] in train_ids and entry['id'] not in by_id,
            'Curriculum source or entry identity differs')
        _same(entry['source_initial_fingerprint'], train_ids[entry['source_id']], 'Curriculum original train fingerprint differs')
        actual = _snapshot_fingerprint(entry['snapshot'], original['configuration'], initial=False)
        _same(entry['frame'], entry['snapshot']['state']['frame'], 'Original curriculum frame differs')
        _same(actual, _sha(entry['initial_content_fingerprint']), 'Curriculum initial-content fingerprint differs')
        _require(actual not in heldout, 'Curriculum overlaps registered heldout initial content')
        _sha(entry['physical_fingerprint'])  # Preserve its separate identity; do not substitute it for initial content.
        by_id[entry['id']] = actual; values.append(actual)
    _require(len(set(values)) == len(values), 'Duplicate registered curriculum physical initial content')
    for line in book.raw(root+'/episodes.jsonl').splitlines():
        if not line.strip(): continue
        value = _json_bytes(line); key = value.get('curriculum_context', {}).get('curriculum_id')
        if key is not None:
            _require(key in by_id, 'Recorded training used an unregistered curriculum ID'); complete_ids.add(key)
    _same(sorted(values), expected['all_registered_initial_content_fingerprints'], 'Inventory omitted registered curriculum starts')
    _same(sorted(by_id[key] for key in complete_ids), expected['completed_episode_curriculum_initial_content_fingerprints'],
          'Original completed-episode curriculum projection differs')
    _same(len(complete_ids), expected['completed_episode_distinct_curriculum_ids'], 'Completed curriculum ID count differs')
    origins = {'plan': book.binding(root+'/protocol.json', producer_version=protocol.get('version')),
        'manifest': book.binding(root+'/curriculum_bank.json', producer_version=bank['version']),
        'run': book.binding(root+'/run.json', producer_version=run['version'], original_status=run.get('status')),
        'episodes': book.binding(root+'/episodes.jsonl')}
    if root.startswith('r1-'):
        binding = book.json(root+'/curriculum_binding.json'); quality = book.json(root+'/curriculum_quality.json')
        _same(binding['bank_sha256'], digest(bank), 'Original r1 bank completion binding differs')
        _require(quality['quality_passed'] is True, 'Original r1 curriculum quality receipt failed')
        origins.update(bank_binding=book.binding(root+'/curriculum_binding.json'), quality=book.binding(root+'/curriculum_quality.json'))
    return _index('development', {'curriculum_starts': values}, origins, original['source_scenario_manifest_sha256'], fixture,
        curriculum_ids=sorted(by_id), completed_episode_used_ids=sorted(complete_ids),
        actual_producer_version=bank['version'], selection_version=bank.get('selection_version'),
        scope_note='All registered starts excluded, including possible in-flight use; complete episode log is not all reset history',
        fingerprint_semantics='initial_content_fingerprint; not physical_fingerprint')


def _legacy_pair_completion(prepared, completion):
    """The frozen producer's lines742–748 omit version in this receipt.

    Its real version belongs to prepared.json and its bound runner source, not
    to a newly invented field on the historical completion.
    """
    _same(sorted(completion), sorted(LEGACY_COMPLETION_KEYS), 'Original pair completion key contract differs')
    _same(prepared['version'], LEGACY_PAIR_VERSION, 'Original enclosing pair producer differs')
    _same(prepared['runtime_sources'][LEGACY_PAIR_SOURCE], file_hash(ROOT/LEGACY_PAIR_SOURCE),
        'Original pair runner byte binding differs')
    _require(type(prepared['primary_endpoint']) is int and prepared['primary_endpoint'] == 250000
        and type(completion['until']) is int and completion['until'] == prepared['primary_endpoint']
        and completion['status'] == 'fixed_endpoint_completed', 'Original fixed pair endpoint is incomplete')
    return {'producer_version': None, 'producer_version_present': False,
        'enclosing_producer_version': prepared['version'], 'producer_source_path': LEGACY_PAIR_SOURCE,
        'producer_source_sha256': prepared['runtime_sources'][LEGACY_PAIR_SOURCE]}


def _verify_pools(book, inventory, original, fixture):
    configuration = original['configuration']; original_sets = original['pools']; checked = {}; origins = {}; statuses = {}
    for root, (pool_file, anchors) in POOL_FILES.items():
        raw_pools = book.json(root+'/'+pool_file)
        _same(sorted(raw_pools), ['selection', 'train'], 'Actual extraction pool roles differ')
        values = {k: _scene_fingerprints(v, configuration) for k, v in raw_pools.items()}
        sizes = (len(values['train']), len(values['selection']))
        if not fixture: _same(sizes, (64, 32) if root in (INITIAL, INTERVENTION) else (192, 32), 'Fixed actual extraction pool size differs')
        for pool, split in (('train', 'train'), ('selection', 'extraction')):
            _same(values[pool], original_sets[split][:len(values[pool])], 'Actual source pool is not the original fixed prefix')
        _same(values, inventory['development_pools'][root]['pools'], 'Inventory pool projection differs')
        checked[root] = values; parsed = {name: book.json(root+'/'+name) for name in anchors}
        for name, value in parsed.items(): origins[root+'/'+name] = book.binding(root+'/'+name, producer_version=value.get('version'), original_status=value.get('status'))
        origins[root+'/'+pool_file] = book.binding(root+'/'+pool_file)
        if root in (PAIR, ACTIVE):
            prepared = parsed['prepared.json']; _same(prepared['refresh_pools_sha256'], digest(raw_pools), 'Paired prepared pool binding differs')
            _same(prepared['scenario_manifest_sha256'], original['source_scenario_manifest_sha256'], 'Paired original scenarios differ')
            if root == ACTIVE: statuses[root] = 'frozen_registered_pools_only_not_sampling_completion'
            else:
                original_contract = _legacy_pair_completion(prepared, parsed['completion_0250000.json'])
                origins[root+'/completion_0250000.json'] = book.binding(root+'/completion_0250000.json',
                    **original_contract, enclosing_prepared_sha256=book.binding(root+'/prepared.json')['sha256'],
                    original_status=parsed['completion_0250000.json']['status'])
                statuses[root] = parsed['completion_0250000.json']['status']
        else:
            plan = parsed['plan.json']; manifest = parsed['collection/manifest.json' if root == INTERVENTION else 'manifest.json']
            _same(plan.get('scene_pools_sha256', plan.get('pools_sha256')), digest(raw_pools), 'Original pool plan binding differs')
            _same(manifest['plan_sha256'], book.binding(root+'/plan.json')['sha256'] if root == INTERVENTION else digest(plan),
                  'Original collection/driver manifest plan binding differs')
            _require(manifest['status'] == 'completed', 'Historical collection or tree driver was not completed')
            statuses[root] = manifest['status']
            if root == INTERVENTION:
                _same(plan['bindings']['scenario_manifest_sha256'], original['source_scenario_manifest_sha256'], 'Intervention source scenarios differ')
                _same(manifest['bindings'], plan['bindings'], 'Actual intervention source binding differs')
            if root == INITIAL:
                _require(len(manifest['episodes']) == len(plan['contexts']) and all(e['status'] == 'completed' for e in manifest['episodes']),
                    'Original initial collection acknowledgments are incomplete')
    for root, values in checked.items():
        for pool in values: _same(values[pool], checked[PAIR][pool][:len(values[pool])], 'Actual development pool changed its fixed prefix')
    origins['plan'] = origins[PAIR+'/prepared.json']; origins['manifest'] = origins[PAIR+'/completion_0250000.json']
    return checked, origins, statuses


def _verify_expansion(book, checked, original, fixture):
    plan = book.json(EXPANDED+'/plan.json'); manifest = book.json(EXPANDED+'/manifest.json')
    _same(manifest['plan_sha256'], digest(plan), 'Expanded fitted producer plan binding differs')
    _require(manifest['status'] == 'completed', 'Original expanded fitting did not complete')
    corpus_root = 'extraction_expanded_corpus_20260909'; extra_root = 'extraction_train_expansion_20260909'
    _same(plan['corpus'], str(book.root/corpus_root), 'Only the fixed derived original corpus is allowed')
    corpus = book.derived(corpus_root+'/manifest.json', plan['corpus_manifest_sha256'])
    _require(corpus['version'] == 'warehouse-native-expanded-extraction-corpus.v1'
        and corpus['status'] == 'verified_merged_corpus', 'Original expanded corpus identity differs')
    _same(corpus['base'], str(book.root/INITIAL), 'Corpus base differs')
    _same(corpus['base_manifest_sha256'], book.binding(INITIAL+'/manifest.json')['sha256'], 'Corpus original initial collection anchor differs')
    _same(corpus['expansion'], str(book.root/extra_root), 'Only the fixed original expansion is allowed')
    _same(sorted(corpus['expansion_anchors']), ['auxiliary_budget.json', 'manifest.json', 'plan.json'], 'Derived original anchors are incomplete')
    extra = {name: book.derived(extra_root+'/'+name, anchor) for name, anchor in corpus['expansion_anchors'].items()}
    ep, em = extra['plan.json'], extra['manifest.json']
    _require(ep['version'] == em['version'] == 'warehouse-native-extraction-train-expansion.v1'
        and em['status'] == 'completed' and ep['selection_pool_unchanged'] is True,
        'Real original train expansion producer/completion differs')
    _same(em['plan_sha256'], digest(ep), 'Expansion plan receipt differs')
    _same(ep['base_manifest_sha256'], corpus['base_manifest_sha256'], 'Expansion base receipt differs')
    raw_pools = book.derived(extra_root+'/scene_pools.json', _anchor_by_semantic(book, extra_root+'/scene_pools.json', ep['scene_pools_sha256']))
    _same(raw_pools['selection'], [], 'Original selection must not be expanded')
    fingerprints = _scene_fingerprints(raw_pools['train'], original['configuration'])
    initial_count = len(checked[INITIAL]['train'])
    _same(fingerprints, checked[PAIR]['train'][initial_count:], 'First64 plus actual additional128 must equal the original192 pool')
    if not fixture: _same((initial_count, len(fingerprints)), (64, 128), 'Fixed actual expansion counts differ')
    _require(len(em['episodes']) == len(ep['contexts']) and all(x['status'] == 'completed' for x in em['episodes']),
        'Derived collection original ACKs are incomplete')
    _same(extra['auxiliary_budget.json']['reserved_joint_steps'], em['reserved_auxiliary_steps'], 'Derived permanent collection budget differs')
    return {name: deepcopy(value) for name, value in book.derived_records.items()}, {
        'producer': ep['version'], 'status': em['status'], 'additional_train_count': len(fingerprints),
        'selected_tree_producer': plan['version'], 'corpus_producer': corpus['version']}


def _anchor_by_semantic(book, relative, expected):
    """One named JSON is authorized by its already byte-bound plan digest."""
    _sha(expected); path = _safe(book.root, relative); raw = path.read_bytes()
    _same(digest(_json_bytes(raw)), expected, 'Derived source semantic pool binding differs')
    return sha256(raw).hexdigest()


def _verify_exposed_and_question(book, inventory, original, fixture):
    header = book.json(HELDOUT+'/audit/inputs.json'); manifest = book.json(HELDOUT+'/audit/manifest.json')
    receipt = book.json(HELDOUT+'.receipt.json'); plan = book.json(HELDOUT+'/plan.json')
    _same(manifest['header_sha256'], book.binding(HELDOUT+'/audit/inputs.json')['sha256'], 'Original exposed header byte anchor differs')
    _same(receipt['evidence_anchor']['audit_manifest_sha256'], book.binding(HELDOUT+'/audit/manifest.json')['sha256'], 'Original exposed manifest byte anchor differs')
    _require(receipt['status'] == 'completed' and receipt['original_report_passed'] is False
        and receipt['final_test_execution'] is False, 'Historical failed explanation status must be preserved')
    _same(header['bindings']['scenario_manifest_sha256'], original['source_scenario_manifest_sha256'], 'Exposed original scenario identity differs')
    values = _scene_fingerprints(header['scenes'], original['configuration'], split='explanation_test')
    _same(values, original['pools']['explanation_test'], 'The complete old exposed explanation pool is required')
    _same(values, inventory['exposed_explanation']['fingerprints'], 'Inventory omitted exposed starts')
    exposed = _index('used_explanation_test', {'explanation_test': values},
        {'audit_inputs': book.binding(HELDOUT+'/audit/inputs.json', producer_version=header['version']),
         'audit_manifest': book.binding(HELDOUT+'/audit/manifest.json', producer_version=manifest['version']),
         'outer_receipt': book.binding(HELDOUT+'.receipt.json'), 'plan': book.binding(HELDOUT+'/plan.json', producer_version=plan['version'])},
        original['source_scenario_manifest_sha256'], fixture, original_report_passed=False, original_status='completed_failed_explanation')
    question = book.json(QUESTION+'/preparation.json'); entries = book.json(QUESTION+'/pool_scenes.json')
    retired = book.json('native_cycle_500k_bank_retirement_20260909.json'); prepared = book.json(BANK+'/prepared.json')
    values = _scene_fingerprints(entries, original['configuration'])
    if not fixture: _same(len(values), 24, 'Entire registered question development pool is required')
    _same(values, inventory['additional_question_development']['initial_fingerprints'], 'Question-development inventory differs')
    _same(question['pool_sha256'], book.binding(QUESTION+'/pool_scenes.json')['sha256'], 'Original question pool bytes differ')
    _same(question['exclusions_sha256'], book.binding(QUESTION+'/exclusion_pools.json')['sha256'], 'Original question exclusions bytes differ')
    _require(retired.get('completed') is False and retired.get('same_output_not_restarted') is True
        and retired['phases']['verification']['status'] == 'not_started', 'Retired bank may not masquerade as a completed verification')
    _same(retired['phases'], inventory['additional_question_development']['retired_phase_counts'], 'Original retirement scope differs')
    result = _index('development', {'registered_question_starts': values},
        {'plan': book.binding(QUESTION+'/preparation.json', original_status=question['status']),
         'manifest': book.binding('native_cycle_500k_bank_retirement_20260909.json', original_status='retired_not_completed'),
         'pool': book.binding(QUESTION+'/pool_scenes.json'), 'bank_prepared': book.binding(BANK+'/prepared.json', original_status=prepared['status'])},
        original['source_scenario_manifest_sha256'], fixture, original_bank_completed=False,
        scope_note='Conservative exclusion of all registered24 starts; partial queries do not imply complete bank verification')
    return exposed, result


def _no_snapshots(value):
    if isinstance(value, dict):
        _require(not {'snapshot', 'snapshots', 'state', 'agents', 'tasks', 'observations', 'weights', 'program'} & set(value),
            'Output must contain metadata only')
        for item in value.values(): _no_snapshots(item)
    elif isinstance(value, list):
        for item in value: _no_snapshots(item)


def _encoded(value):
    _no_snapshots(value); return (canonical(value)+'\n').encode()


def _binding(name, raw): return {'path': name, 'sha256': sha256(raw).hexdigest(), 'size': len(raw)}


def _write(root, name, raw):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def verify_and_write(inventory_path, *, expected_inventory_sha256, output, allow_test_fixture=False):
    """Verify actual sources privately, then publish only immutable metadata."""
    _require(type(allow_test_fixture) is bool, 'Explicit source fixture scope required')
    path = Path(inventory_path).expanduser().absolute(); out = Path(output).expanduser().absolute()
    _require(out.resolve() == out and not out.exists(), 'Unique unlinked output required')
    if not allow_test_fixture: _same(expected_inventory_sha256, INVENTORY_SHA256, 'Production inventory is fixed before projection')
    raw_inventory = _bytes(path, expected_inventory_sha256); inventory = _json_bytes(raw_inventory)
    _require(inventory.get('version') == INVENTORY_VERSION and inventory.get('test_fixture', False) is allow_test_fixture
        and inventory.get('status') == 'limited_inventory_not_source_acceptance', 'Explicit original inventory identity differs')
    code = sources()
    for name, binding in inventory['static_source_bindings'].items():
        _bytes(_safe(ROOT, name), binding['sha256'], binding['size'])
    book = _Origins(inventory, allow_test_fixture)
    source_directories = {book.root/name.split('/')[0] for name in required_origins() if '/' in name}
    _require(out != book.root and all(out != directory and directory not in out.parents for directory in source_directories),
        'Output must be independent from all source run directories')
    original, train_ids = _verify_original(book, inventory, allow_test_fixture)
    indices = {'indices/original.json': original}
    for root, label in zip(CURRICULA, ('curriculum_v2', 'curriculum_r1')):
        indices[f'indices/{label}.json'] = _verify_curriculum(book, inventory, root, original, train_ids, allow_test_fixture)
    pools, origins, statuses = _verify_pools(book, inventory, original, allow_test_fixture)
    derived, expansion = _verify_expansion(book, pools, original, allow_test_fixture)
    origins.update({'derived_'+str(i): value for i, value in enumerate(derived.values())})
    origins['expanded_plan'] = book.binding(EXPANDED+'/plan.json'); origins['expanded_manifest'] = book.binding(EXPANDED+'/manifest.json')
    indices['indices/development.json'] = _index('development', pools[PAIR], origins,
        original['source_scenario_manifest_sha256'], allow_test_fixture, original_statuses=statuses,
        expansion=expansion, current_cycle_scope='frozen_registered_pools_only_not_sampling_completion')
    indices['indices/exposed_explanation.json'], indices['indices/question_development.json'] = _verify_exposed_and_question(book, inventory, original, allow_test_fixture)
    blobs = {name: _encoded(value) for name, value in indices.items()}
    catalog = [{**_binding(name, blobs[name]), 'scope': value['scope'], 'pools_sha256': digest(value['pools']),
        'origin_bindings': value['origin_bindings']} for name, value in indices.items()]
    book.unchanged(); _same(sources(), code, 'Projection sources changed')
    _bytes(path, expected_inventory_sha256, len(raw_inventory))
    acceptance = {'version': fresh.INDEX_ACCEPTANCE_VERSION, 'producer_version': VERSION, 'status': 'source_indices_verified',
        'test_fixture': allow_test_fixture, 'source_scenario_manifest_sha256': original['source_scenario_manifest_sha256'],
        'configuration': original['configuration'], 'index_contract': deepcopy(fresh.INDEX_CONTRACT), 'registered_indices': catalog,
        'checks': dict.fromkeys(('original_source_bytes_and_index_sets', 'all_original_seven_pool_counts',
            'all_previously_exposed_explanation_pools', 'all_actual_training_tree_intervention_sources'), True),
        'verifier_sources': code, 'inventory_sha256': expected_inventory_sha256,
        'verified_direct_origins': {name: value for name, value in book.records.items() if name in book.entries},
        'derived_bound_origins': derived, 'direct_origin_count': 54, 'derived_origin_count': len(derived),
        'scope': 'Actual source metadata and private fingerprint derivation; no episode/NN/physics trajectory replay, no model selection',
        'current_cycle_scope': 'frozen_registered_pools_only_not_sampling_completion',
        'environment_steps': 0, 'environment_constructions': 0, 'neural_forwards': 0, 'checkpoint_decodes': 0,
        'fits': 0, 'registry_operations': 0, 'final_performance_tests': 0, 'release_ready': False, 'explanation_qualified': False}
    blobs['source_index_acceptance.json'] = _encoded(acceptance)
    exclusions = {'version': fresh.EXCLUSIONS_VERSION, 'test_fixture': allow_test_fixture,
        'source_scenario_manifest_sha256': original['source_scenario_manifest_sha256'], 'registered_indices': catalog,
        'source_index_acceptance': _binding('source_index_acceptance.json', blobs['source_index_acceptance.json'])}
    blobs['exclusions.json'] = _encoded(exclusions)
    manifest = {'version': VERSION, 'status': 'metadata_sources_complete', 'test_fixture': allow_test_fixture,
        'files': {name: _binding(name, raw) for name, raw in blobs.items()}, 'inventory_sha256': expected_inventory_sha256,
        'source_scenario_manifest_sha256': original['source_scenario_manifest_sha256'], 'qualification_granted': False}
    blobs['manifest.json'] = _encoded(manifest)
    # No public output exists before all source checks and output-shape checks.
    out.mkdir(parents=True, exist_ok=False)
    for name in (*indices, 'exclusions.json', 'source_index_acceptance.json', 'manifest.json'):
        _write(out, name, blobs[name])  # Verified acceptance and commit manifest are last.
    return {'version': VERSION, 'output': str(out), 'status': 'metadata_sources_complete',
        'exclusions_sha256': sha256(blobs['exclusions.json']).hexdigest(),
        'source_index_acceptance_sha256': sha256(blobs['source_index_acceptance.json']).hexdigest(),
        'manifest_sha256': sha256(blobs['manifest.json']).hexdigest(),
        'source_scenario_manifest_sha256': original['source_scenario_manifest_sha256'],
        'counts': {name: value['counts'] for name, value in indices.items()}, 'direct_origins': 54,
        'derived_bound_origins': len(derived), 'environment_steps': 0, 'neural_forwards': 0,
        'checkpoint_decodes': 0, 'fits': 0, 'registry_operations': 0, 'qualification_granted': False,
        'test_fixture': allow_test_fixture}


def read_registered(root, *, expected_exclusions_sha256, expected_source_index_acceptance_sha256, allow_test_fixture=False):
    """Read only committed metadata; never reopen original states or provenance readers."""
    _require(type(allow_test_fixture) is bool, 'Explicit fixture scope required')
    root = Path(root).expanduser().absolute(); _require(root.resolve() == root, 'Unlinked metadata root required')
    blobs = {name: _bytes(_safe(root, name), anchor) for name, anchor in (
        ('exclusions.json', expected_exclusions_sha256), ('source_index_acceptance.json', expected_source_index_acceptance_sha256))}
    excluded = _json_bytes(blobs['exclusions.json']); acceptance = _json_bytes(blobs['source_index_acceptance.json'])
    _require(excluded['version'] == fresh.EXCLUSIONS_VERSION and acceptance['version'] == fresh.INDEX_ACCEPTANCE_VERSION
        and acceptance.get('producer_version') == VERSION and acceptance.get('status') == 'source_indices_verified'
        and excluded.get('test_fixture') is allow_test_fixture and acceptance.get('test_fixture') is allow_test_fixture,
        'Registered metadata producer, completion or fixture differs')
    _same(acceptance['index_contract'], fresh.INDEX_CONTRACT, 'Exact source-index contract differs')
    _same(acceptance['verifier_sources'], sources(), 'Registered source verifier closure differs')
    _same(acceptance['registered_indices'], excluded['registered_indices'], 'Accepted exclusion catalog differs')
    _same(excluded['source_index_acceptance'], _binding('source_index_acceptance.json', blobs['source_index_acceptance.json']), 'Independent source acceptance binding differs')
    checks = ('original_source_bytes_and_index_sets', 'all_original_seven_pool_counts',
        'all_previously_exposed_explanation_pools', 'all_actual_training_tree_intervention_sources')
    _same(acceptance['checks'], dict.fromkeys(checks, True), 'Source-index checks did not pass')
    expected_index_paths = {'indices/'+name+'.json' for name in ('original', 'curriculum_v2', 'curriculum_r1',
        'development', 'exposed_explanation', 'question_development')}
    _require(len(excluded['registered_indices']) == 6
        and {e['path'] for e in excluded['registered_indices']} == expected_index_paths,
        'Complete six-index producer catalog required')
    _require(type(acceptance['direct_origin_count']) is int and acceptance['direct_origin_count'] == 54
        and len(acceptance['verified_direct_origins']) == 54
        and type(acceptance['derived_origin_count']) is int and acceptance['derived_origin_count'] == 5
        and len(acceptance['derived_bound_origins']) == 5, 'Complete direct and derived origin metadata required')
    for value in (*acceptance['verified_direct_origins'].values(), *acceptance['derived_bound_origins'].values()):
        _sha(value['sha256']); _require(type(value['size']) is int and value['size'] > 0, 'Original binding size differs')
    for key in ('environment_steps', 'environment_constructions', 'neural_forwards', 'checkpoint_decodes',
                'fits', 'registry_operations', 'final_performance_tests'):
        _require(type(acceptance[key]) is int and acceptance[key] == 0, 'Metadata verification must retain zero execution')
    _require(acceptance['release_ready'] is False and acceptance['explanation_qualified'] is False
        and acceptance['current_cycle_scope'] == 'frozen_registered_pools_only_not_sampling_completion',
        'Source index cannot grant model qualification or current-cycle completion')
    scenario_sha = acceptance['source_scenario_manifest_sha256']; _sha(scenario_sha)
    _same(excluded['source_scenario_manifest_sha256'], scenario_sha, 'Original scenario identity differs')
    _configuration(acceptance['configuration'], allow_test_fixture)
    scopes = {name: [] for name in ('source_scenarios', 'used_explanation_test', 'development')}; used = set()
    for entry in excluded['registered_indices']:
        name = entry['path']; _require(name not in blobs and entry['scope'] in scopes, 'Duplicate or invalid registered index')
        raw = _bytes(_safe(root, name), entry['sha256'], entry['size']); value = _json_bytes(raw); _no_snapshots(value)
        _require(value['version'] == fresh.INDEX_VERSION and value['test_fixture'] is allow_test_fixture
            and value['scope'] == entry['scope'] and value['source_scenario_manifest_sha256'] == scenario_sha, 'Source index identity differs')
        _same(value['origin_bindings'], entry['origin_bindings'], 'Registered origin anchors differ')
        required = {'source_scenarios': {'scenarios'}, 'used_explanation_test': {'audit_inputs', 'audit_manifest'}, 'development': {'plan', 'manifest'}}[entry['scope']]
        _require(required <= set(value['origin_bindings']), 'Registered source roles are incomplete')
        for item in value['origin_bindings'].values():
            _sha(item['sha256']); _require(type(item['size']) is int and item['size'] > 0, 'Original source byte count is invalid')
        pools = {k: fresh._fingerprints(v) for k, v in value['pools'].items()}
        _require(bool(pools), 'Registered index contains no pools')
        _same(value['counts'], {k: len(v) for k, v in pools.items()}, 'Registered index pool count differs')
        _same(entry['pools_sha256'], digest(value['pools']), 'Registered pool digest differs')
        scopes[entry['scope']].append(value); used.update(set().union(*pools.values())); blobs[name] = raw
    _require(len(scopes['source_scenarios']) == 1 and scopes['used_explanation_test'] and scopes['development'], 'A complete source family is missing')
    original = scopes['source_scenarios'][0]; pools = original['pools']
    _same(sorted(pools), sorted(SPLIT_NAMES), 'Original seven pools are incomplete')
    _same(original['origin_bindings']['scenarios']['semantic_sha256'], scenario_sha, 'Actual Actor scenario semantic binding is missing')
    _same(original['configuration'], acceptance['configuration'], 'Registered public configuration differs')
    _require(sum(len(v) for v in pools.values()) == len(set().union(*(set(v) for v in pools.values()))), 'Original registered pools overlap')
    if not allow_test_fixture:
        _same(original['counts'], fresh.REGISTERED_COUNTS, 'Production seven-pool counts differ')
        _require(all(len(v) == 100 for idx in scopes['used_explanation_test'] for v in idx['pools'].values()), 'Complete exposed100 pools required')
    exposed = set().union(*(set(v) for idx in scopes['used_explanation_test'] for v in idx['pools'].values()))
    _require(set(pools['explanation_test']) <= exposed, 'Previously exposed explanation starts omitted')
    manifest_raw = _safe(root, 'manifest.json').read_bytes(); manifest = _json_bytes(manifest_raw); _no_snapshots(manifest)
    _require(manifest['version'] == VERSION and manifest['status'] == 'metadata_sources_complete'
        and manifest['test_fixture'] is allow_test_fixture and manifest['qualification_granted'] is False,
        'Metadata publication did not complete')
    _same(manifest['files'], {name: _binding(name, raw) for name, raw in blobs.items()}, 'Complete committed metadata inventory differs')
    _same(manifest['inventory_sha256'], acceptance['inventory_sha256'], 'Original inventory commitment differs')
    _same(manifest['source_scenario_manifest_sha256'], scenario_sha, 'Committed original scenario identity differs')
    blobs['manifest.json'] = manifest_raw
    for raw in blobs.values(): _no_snapshots(_json_bytes(raw))
    return {'version': VERSION, 'configuration': deepcopy(acceptance['configuration']),
        'source_scenario_manifest_sha256': scenario_sha, 'excluded_fingerprints': sorted(used), 'blobs': blobs,
        'catalog': deepcopy(excluded['registered_indices']), 'test_fixture': allow_test_fixture,
        'source_index_acceptance_sha256': expected_source_index_acceptance_sha256,
        'exclusions_sha256': expected_exclusions_sha256, 'qualification_granted': False,
        'scope': 'Complete externally anchored metadata only; original54 sources are not reopened'}
