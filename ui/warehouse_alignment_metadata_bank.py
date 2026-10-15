"""Private same-Actor questions with verified metadata-only exclusion sources.

Only the new question-pool scenes enter a real runtime environment. Original
training/final/play exclusions and fresh explanation starts remain fingerprints
and external byte bindings. Generation and independent loading both execute the
complete frozen candidate/selection algorithm with genuine Alignment admission; neither grants study admission.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from backend import warehouse_alignment_runtime as runtime_api
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from env.warehouse_native.scenarios import scenario_fingerprint
from ui import warehouse_public_history_bank as primitives
from ui.warehouse_public_history_bank import PublicHistoryQuestionBank as _PureMethods
from ui.warehouse_public_history_bank import _candidate, _preview, _Operations, _select, _checks, KINDS, FILTER

VERSION = 'warehouse-alignment-metadata-prediction-bank.v1'
POOL_NAMESPACE = 'independent_question_bank_development_verified_metadata'
SCOPE = 'same_actor_question_content_and_complete_independent_replay_only'
_FIELDS = frozenset(('version', 'status', 'formal_ready', 'release_ready', 'test_fixture',
    'runtime_family', 'runtime_version', 'runtime_sources_sha256', 'actor_sha256', 'protocol_sha256',
    'runtime_signature', 'sources', 'sources_sha256', 'question_pool_binding',
    'excluded_fingerprints_sha256', 'pool_namespace', 'pool_scenes', 'trajectory_steps', 'minimum_frame',
    'trajectories', 'items', 'checks', 'generation_audit', 'counterfactual_filter', 'scope'))


def _same(a, b, reason):
    if canonical(a) != canonical(b): raise ValueError(reason)


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('An explicit lowercase external SHA256 is required')
    return value


def bank_sources(runtime, *, pool_sources):
    """Bind genuine runtime, reused primitives and the registered pool reader."""
    from backend.training import warehouse_family_question_pool as question_pool
    values = primitives.bank_sources(runtime)
    if not isinstance(pool_sources, dict) or not pool_sources:
        raise ValueError('Verified question-pool source closure is required')
    for name, anchor in pool_sources.items():
        _sha(anchor); path = Path(name)
        if (type(name) is not str or path.is_absolute() or '..' in path.parts or str(path) != name
                or (ROOT/path).resolve() != ROOT/path or file_hash(ROOT/path) != anchor):
            raise ValueError('Registered question-pool source changed')
        if name in values and values[name] != anchor: raise ValueError('Source closures disagree')
        values[name] = anchor
    for path in (Path(__file__), Path(question_pool.__file__)):
        name = str(path.relative_to(ROOT)); anchor = file_hash(path)
        if name in values and values[name] != anchor: raise ValueError('Question reader source differs')
        values[name] = anchor
    return values


def _context(runtime, pool_root, anchor, trajectory_steps, minimum_frame, fixture):
    """All exclusion admission here is metadata only, with no environment."""
    from backend.training import warehouse_family_question_pool as question_pool
    _sha(anchor)
    identity = runtime_api.verify(runtime, allow_test_fixture=fixture)
    registered = question_pool.read_registered(pool_root, expected_manifest_sha256=anchor, allow_test_fixture=fixture)
    if registered['test_fixture'] is not fixture or registered['manifest_sha256'] != anchor:
        raise ValueError('Registered question-pool external identity or fixture differs')
    _same(registered['configuration'], asdict(runtime.config), 'Question pool and actual runtime configuration differ')
    _same(registered['source_scenario_manifest_sha256'], runtime.actor.metadata['scenario_manifest_sha256'],
        'Question pool does not descend from the actual Actor scenario identity')
    scenes = deepcopy(registered['scenes']); horizon = runtime.config.horizon
    if (type(trajectory_steps) is not int or type(minimum_frame) is not int
            or not 0 <= minimum_frame < trajectory_steps <= horizon
            or not isinstance(scenes, list)
            or (not fixture and (len(scenes) != 24 or horizon != 120 or trajectory_steps != 24 or minimum_frame != 1))
            or (fixture and (len(scenes) != 4 or not 1 <= horizon <= 3 or not 1 <= trajectory_steps <= 3))):
        raise ValueError('Fixed production24/T24/min1 or explicit small fixture4 is required')
    excluded = registered['excluded_fingerprints']
    if (not isinstance(excluded, list) or not excluded or len(set(excluded)) != len(excluded)):
        raise ValueError('Complete unique verified exclusion fingerprints required')
    for value in excluded: _sha(value)
    artifacts = {}
    if not isinstance(registered['blobs'], dict) or not registered['blobs']:
        raise ValueError('Registered pool byte artifacts are required')
    for name, raw in registered['blobs'].items():
        path = Path(name)
        if (type(name) is not str or path.is_absolute() or '..' in path.parts or str(path) != name
                or type(raw) is not bytes): raise ValueError('Unsafe registered pool artifact')
        artifacts[name] = {'sha256': sha256(raw).hexdigest(), 'size': len(raw)}
    binding = {'version': registered['version'], 'manifest_sha256': anchor,
        'pool_sha256': _sha(registered['pool_sha256']),
        'source_scenario_manifest_sha256': _sha(registered['source_scenario_manifest_sha256']),
        'exclusion_binding': deepcopy(registered['exclusion_binding']),
        'sources_sha256': digest(registered['sources']), 'artifacts': artifacts}
    return {'identity': identity, 'registered': registered, 'scenes': scenes,
        'excluded': set(excluded), 'excluded_sha256': digest(sorted(excluded)), 'binding': binding,
        'sources': bank_sources(runtime, pool_sources=registered['sources'])}


def _new_scenes(runtime, context):
    """Only these new question scenes may be restored, never any exclusions."""
    scenes = {}; found = set()
    for scene in context['scenes']:
        if type(scene.get('id')) is not str or not scene['id'] or scene['id'] in scenes:
            raise ValueError('Duplicate or invalid question scene identity')
        _sha(scene.get('fingerprint'))
        env = runtime.environment(scene)
        actual = scenario_fingerprint(env)
        if (actual != scene['fingerprint'] or env.state.frame != 0 or env.done
                or actual in context['excluded'] or actual in found):
            raise ValueError('Actual new question start is not initial, unique and excluded-source disjoint')
        scenes[scene['id']] = scene; found.add(actual)
    return scenes


def _unchanged(runtime, pool_root, anchor, limit, minimum, fixture, before):
    after = _context(runtime, pool_root, anchor, limit, minimum, fixture)
    for key in ('identity', 'binding', 'sources', 'scenes', 'excluded_sha256'):
        _same(after[key], before[key], 'Actual runtime or registered pool changed during bank work')


def _make_candidate(runtime, env, scene_id, kind, operations):
    before = digest(env.snapshot())
    result = _candidate(runtime, env, scene_id, kind, operations)
    # The frozen primitive checks successful items itself; also check the
    # filtered/None path without altering that filtering or any action.
    if digest(env.snapshot()) != before: raise ValueError('Question branch changed its live source or RNG')
    return result


def generate_bank(runtime, pool_root, *, expected_pool_manifest_sha256, trajectory_steps,
                  minimum_frame=1, before_operation, after_operation, allow_test_fixture=False):
    operations = _Operations(before_operation, after_operation, 'generation')
    c = _context(runtime, pool_root, expected_pool_manifest_sha256, trajectory_steps, minimum_frame, allow_test_fixture)
    scenes = _new_scenes(runtime, c); candidates, trajectories = [], {}
    for scene_id, scene in scenes.items():
        env = runtime.environment(scene); rows = []
        with runtime.verified_context(env):
            for _ in range(trajectory_steps):
                if env.done: break
                if env.state.frame >= minimum_frame:
                    for kind in KINDS:
                        item = _make_candidate(runtime, env, scene_id, kind, operations)
                        if item is not None: candidates.append(item)
                actions, _ = operations.call(lambda: runtime.decision(env), kind='selfplay_decision',
                    scene=scene_id, frame=env.state.frame, maximum_steps=0)
                rows.append(operations.call(lambda: runtime.step(env, actions['robot_1']), kind='trajectory',
                    scene=scene_id, frame=env.state.frame, maximum_steps=1))
        trajectories[scene_id] = rows
    _unchanged(runtime, pool_root, expected_pool_manifest_sha256, trajectory_steps, minimum_frame, allow_test_fixture, c)
    items = _select(candidates); identity = c['identity']
    return {'version': VERSION, 'status': 'candidate', 'formal_ready': False, 'release_ready': False,
        'test_fixture': allow_test_fixture, 'runtime_family': identity['family'], 'runtime_version': identity['runtime_version'],
        'runtime_sources_sha256': identity['runtime_sources_sha256'], 'actor_sha256': runtime.actor_sha256,
        'protocol_sha256': runtime.protocol_sha256, 'runtime_signature': runtime.signature,
        'sources': c['sources'], 'sources_sha256': digest(c['sources']), 'question_pool_binding': c['binding'],
        'excluded_fingerprints_sha256': c['excluded_sha256'], 'pool_namespace': POOL_NAMESPACE,
        'pool_scenes': c['scenes'], 'trajectory_steps': trajectory_steps, 'minimum_frame': minimum_frame,
        'trajectories': trajectories, 'items': items, 'checks': _checks(items),
        'generation_audit': operations.report(), 'counterfactual_filter': FILTER, 'scope': SCOPE}


class AlignmentMetadataQuestionBank(_PureMethods):
    """Genuine independent replay; inherited public projection never exposes answers."""
    def __init__(self, path, runtime, pool_root, *, expected_bank_sha256, expected_pool_manifest_sha256,
                 before_operation, after_operation, allow_test_fixture=False):
        _sha(expected_bank_sha256); descriptor = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream: raw = stream.read()
        if sha256(raw).hexdigest() != expected_bank_sha256: raise ValueError('Private bank external byte anchor differs')
        b = json.loads(raw)
        if set(b) != _FIELDS: raise ValueError('Explicit metadata-bank private field schema differs')
        limit, minimum = b['trajectory_steps'], b['minimum_frame']
        operations = _Operations(before_operation, after_operation, 'load_reverification')
        c = _context(runtime, pool_root, expected_pool_manifest_sha256, limit, minimum, allow_test_fixture)
        identity = c['identity']
        expected = {'version': VERSION, 'status': 'candidate', 'formal_ready': False, 'release_ready': False,
            'test_fixture': allow_test_fixture, 'runtime_family': identity['family'], 'runtime_version': identity['runtime_version'],
            'runtime_sources_sha256': identity['runtime_sources_sha256'], 'actor_sha256': runtime.actor_sha256,
            'protocol_sha256': runtime.protocol_sha256, 'runtime_signature': runtime.signature,
            'sources': c['sources'], 'sources_sha256': digest(c['sources']), 'question_pool_binding': c['binding'],
            'excluded_fingerprints_sha256': c['excluded_sha256'], 'pool_namespace': POOL_NAMESPACE,
            'pool_scenes': c['scenes'], 'counterfactual_filter': FILTER, 'scope': SCOPE}
        for name, value in expected.items(): _same(b[name], value, 'Private question-bank source, schema or scope differs')
        if (not isinstance(b['items'], list) or len(b['items']) > 8
                or len({x['id'] for x in b['items']}) != len(b['items'])
                or set(b['trajectories']) != {s['id'] for s in c['scenes']}):
            raise ValueError('Question/trajectory source coverage differs')
        for item in b['items']:
            if (item['kind'] not in KINDS or item['scenario_id'] not in b['trajectories']
                    or item['id'] not in {f"prediction_{item['kind']}_{i}" for i in range(1, 5)}
                    or type(item['frame']) is not int or not minimum <= item['frame'] < limit):
                raise ValueError('Unknown or out-of-range question source')
        scenes = _new_scenes(runtime, c); snapshots, candidates = {}, []
        for scene_id, scene in scenes.items():
            env = runtime.environment(scene); rows = b['trajectories'][scene_id]
            if not isinstance(rows, list) or not 1 <= len(rows) <= limit: raise ValueError('Missing complete source trajectory')
            with runtime.verified_context(env):
                for row in rows:
                    if env.done: raise ValueError('Source trajectory continues past terminal')
                    snapshots[(scene_id, env.state.frame)] = env.snapshot()
                    if env.state.frame >= minimum:
                        for kind in KINDS:
                            item = _make_candidate(runtime, env, scene_id, kind, operations)
                            if item is not None: candidates.append(item)
                    actions, _ = operations.call(lambda: runtime.decision(env), kind='selfplay_decision',
                        scene=scene_id, frame=env.state.frame, maximum_steps=0)
                    actual = operations.call(lambda: runtime.step(env, actions['robot_1']), kind='trajectory',
                        scene=scene_id, frame=env.state.frame, maximum_steps=1)
                    _same(actual, row, 'Source is not the same-Actor legal full-history trajectory')
                if len(rows) != limit and not env.done: raise ValueError('Source trajectory was silently truncated')
        _same(_select(candidates), b['items'], 'Complete question selection, options or private answers differ')
        for item in b['items']:
            snapshot = snapshots.get((item['scenario_id'], item['frame']))
            if (snapshot is None or digest(snapshot) != item['snapshot_sha256']
                    or digest(item['snapshot']) != item['snapshot_sha256']):
                raise ValueError('Question snapshot differs from its actual source frame')
            runtime.from_snapshot(item['snapshot'])
        checks = _checks(b['items'])
        _same(checks, b['checks'], 'Saved content-check claims differ')
        _same(operations.report(), b['generation_audit'], 'Complete independent replay work differs')
        _unchanged(runtime, pool_root, expected_pool_manifest_sha256, limit, minimum, allow_test_fixture, c)
        self._bank, self._items = deepcopy(b), deepcopy(b['items'])
        self.checks, self.content_eligible = checks, checks['passed']
        self.test_fixture, self.actor_sha256, self.runtime_signature = allow_test_fixture, runtime.actor_sha256, runtime.signature
        self.question_pool_binding, self.excluded_fingerprints_sha256 = deepcopy(c['binding']), c['excluded_sha256']
        self.signature = digest({'version': VERSION, 'bank_sha256': expected_bank_sha256,
            'sources_sha256': digest(c['sources']), 'runtime_signature': runtime.signature,
            'question_pool_binding': c['binding']})
        self.audit = operations.report()

    @property
    def eligible(self): return False
    @property
    def participant_enabled(self): return False
    @property
    def release_ready(self): return False
    @property
    def formal_ready(self): return False

    def summary(self):
        return {**super().summary(), 'version': VERSION, 'eligible': False,
            'participant_enabled': False, 'independent_replay_completed': True, 'scope': SCOPE}

    def grade(self, answers):
        if type(answers) is not dict: raise ValueError('Prediction answers must be an object')
        return super().grade(answers)
