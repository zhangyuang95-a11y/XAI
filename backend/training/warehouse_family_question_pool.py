"""Independent question starts, excluding verified sources and the fresh audit pool.

Only newly drawn question states are constructed. Original training/final
snapshots are never loaded by this component. The completed fresh generation
is authenticated as registered input, not promoted to a passed explanation
audit. No Actor, model score, or human result influences sampling.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re

from backend.training import warehouse_family_explanation_source_index as index
from backend.training import warehouse_family_fresh_explanation_pool as fresh_generator
from backend.training import warehouse_family_fresh_explanation_run as fresh
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.scenarios import scenario_fingerprint

VERSION = 'warehouse-family-verified-metadata-question-pool.v1'
PURPOSE = 'independent_question_bank_development'
COUNT = 24
ATTEMPT_CAP = 10000
SELECTION = 'physical_uniqueness_and_complete_registered_exclusions_only'


def sources():
    return {**fresh_generator.sources(), str(Path(__file__).relative_to(ROOT)): file_hash(__file__)}


def _require(condition, message):
    if not condition: raise ValueError(message)


def _same(a, b, message):
    _require(canonical(a) == canonical(b), message)


def _path(value):
    value = Path(value).expanduser().absolute()
    _require(value.resolve() == value, 'Canonical unlinked path required')
    return value


def _encoded(value): return (canonical(value)+'\n').encode()


def _binding(raw): return {'sha256': sha256(raw).hexdigest(), 'size': len(raw)}


def _raw(root, name, binding):
    _require(type(binding) is dict and set(binding) == {'sha256', 'size'}, 'Exact external byte binding required')
    return index._bytes(index._safe(root, name), binding['sha256'], binding['size'])


def _write(root, name, raw):
    target = root/name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(target.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _abort(root, error, sampled, fixture):
    # Revoke this invocation's completion before attempting another write.
    # Failure details may be unwritable; an existing manifest must not alone
    # make that interrupted invocation acceptable to a later reader.
    try:
        (root/'manifest.json').unlink(missing_ok=True)
        descriptor = os.open(root, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    except OSError as persistence:
        error.add_note('Completion retraction needs filesystem recovery: '+repr(persistence))
    try:
        _write(root, 'failure.json', _encoded({'version': VERSION, 'reason': repr(error), 'actual_sampling': sampled,
            'completed_pool_published': False, 'test_fixture': fixture, 'release_ready': False}))
    except OSError as persistence:
        error.add_note('Failure persistence unavailable; preserve existing plan/report and do not retry: '+repr(persistence))


def _counts(attempts):
    return dict(environment_constructions=1, environment_resets=attempts,
        joint_environment_steps=0, neural_forwards=0, actor_loads=0,
        checkpoint_decodes=0, tree_fits=0)


def _attempts(plan, report, scenes, excluded):
    """Recompute saved selection/accounting without redrawing any initial state."""
    draws = report['attempts']; accepted = []; seen = set(); reasons = {'registered_exclusion': 0, 'duplicate_draw': 0}
    _require(type(draws) is list and plan['count'] <= len(draws) <= plan['attempt_cap'], 'Complete bounded draw history required')
    for offset, row in enumerate(draws):
        _require(set(row) == {'seed', 'fingerprint', 'result'}, 'Draw record contains unexpected fields')
        index._sha(row['fingerprint'])
        _require(type(row['seed']) is int and row['seed'] == plan['seed_start']+offset, 'Draw seed sequence differs')
        reason = ('registered_exclusion' if row['fingerprint'] in excluded else
                  'duplicate_draw' if row['fingerprint'] in seen else 'accepted')
        _same(row['result'], reason, 'Saved physical exclusion outcome differs')
        if reason == 'accepted':
            accepted.append((row['seed'], row['fingerprint'])); seen.add(row['fingerprint'])
            _require(len(accepted) <= plan['count'] and (len(accepted) < plan['count'] or offset == len(draws)-1),
                     'Generation continued after its fixed sample was complete')
        else: reasons[reason] += 1
    _same(accepted, [(s['seed'], s['fingerprint']) for s in scenes], 'Accepted draw order differs from registered scenes')
    _same(report['rejections'], reasons, 'Rejected draw count differs')
    _same(report['counts'], _counts(len(draws)), 'Sampling execution counts differ')


def _fresh_source(root, expected_manifest_sha256, fixture):
    """Read a completed physical generator and its original nine metadata blobs."""
    root = _path(root)
    raw = index._bytes(index._safe(root, 'generation_manifest.json'), expected_manifest_sha256)
    manifest = json.loads(raw)
    _require(manifest['version'] == fresh_generator.VERSION and manifest['status'] == 'complete'
        and manifest['test_fixture'] is fixture and manifest['qualification_evaluated'] is False,
        'A completed genuine fresh-generation producer is required')
    registered = index.read_registered(root, expected_exclusions_sha256=manifest['exclusions_sha256'],
        expected_source_index_acceptance_sha256=manifest['source_index_acceptance_sha256'], allow_test_fixture=fixture)
    required = set(registered['blobs']) | {'generation_plan.json', 'pool.json', 'generation_report.json'}
    _same(sorted(manifest['artifacts']), sorted(required), 'Complete fresh generator artifact inventory required')
    blobs = {}
    for name, binding in manifest['artifacts'].items():
        _same(binding['path'], name, 'Original fresh artifact path differs')
        blobs[name] = _raw(root, name, {k: binding[k] for k in ('sha256', 'size')})
    for name, value in registered['blobs'].items(): _same(_binding(blobs[name]), _binding(value), 'Metadata source copy differs')
    _require(not (root/'generation_failure.json').exists(), 'Failed fresh generation is not a complete source')
    parsed = fresh._pool(root, manifest['pool_sha256'], manifest['exclusions_sha256'],
                         manifest['source_index_acceptance_sha256'], fixture)
    pool = parsed['pool']; plan = json.loads(blobs['generation_plan.json']); report = json.loads(blobs['generation_report.json'])
    fresh_generator._request(plan['configuration'], plan['pool_id'], plan['seed_start'], plan['count'], plan['attempt_cap'], fixture)
    _same(plan['sources'], fresh_generator.sources(), 'Original fresh generation code differs')
    _require(plan['version'] == fresh_generator.VERSION and report['version'] == fresh_generator.VERSION
        and plan['test_fixture'] is fixture and report['test_fixture'] is fixture
        and report['status'] == 'physically_registered_only', 'Original physical generation identity differs')
    _same(plan['selection_rule'], fresh_generator.SELECTION, 'Fresh physical-only sampling rule differs')
    _same(plan['battery_values'], list(fresh_generator.BATTERIES), 'Initial battery distribution differs')
    _same(plan['configuration'], registered['configuration'], 'Public source configuration differs')
    for key in ('pool_id', 'configuration', 'source_scenario_manifest_sha256', 'exclusions_sha256'):
        _same(plan[key], pool[key], 'Original fresh generation binding differs: '+key)
    _same(pool['generation'], {'version': fresh_generator.VERSION, 'selection_rule': fresh_generator.SELECTION,
        'seed_start': plan['seed_start'], 'sources_sha256': digest(plan['sources']),
        'generation_plan_sha256': sha256(blobs['generation_plan.json']).hexdigest()}, 'Complete original fresh generation binding differs')
    _same(plan['source_index_acceptance_sha256'], manifest['source_index_acceptance_sha256'], 'Source acceptance binding differs')
    _same(plan['excluded_fingerprints_sha256'], digest(sorted(registered['excluded_fingerprints'])), 'Original excluded set differs')
    _same(plan['source_scenario_manifest_sha256'], registered['source_scenario_manifest_sha256'], 'Original source scenario identity differs')
    _require(plan['count'] == len(pool['scenes']) and report['accepted_initial_states'] == plan['count']
        and report['excluded_initial_states'] == len(registered['excluded_fingerprints']), 'Original source sample count differs')
    _same(report['initial_fingerprints_sha256'], pool['initial_fingerprints_sha256'], 'Source draw fingerprint summary differs')
    _require(report['all_source_exclusions_applied'] is True and report['nn_or_tree_based_selection'] is False
        and report['explanation_qualified'] is False and report['release_ready'] is False
        and report['final_performance_test_executed'] is False
        and plan['joint_environment_step_cap'] == plan['neural_query_cap'] == 0
        and plan['final_performance_test_executed'] is False and plan['qualification_evaluated'] is False,
        'Source generation cannot execute or claim model qualification')
    _attempts(plan, report, pool['scenes'], set(registered['excluded_fingerprints']))
    _same(manifest['counts'], report['counts'], 'Committed original sampling counts differ')
    blobs['generation_manifest.json'] = raw
    excluded = sorted(set(registered['excluded_fingerprints']) | set(parsed['fingerprints']))
    return {'configuration': registered['configuration'], 'source_scenario_manifest_sha256': registered['source_scenario_manifest_sha256'],
        'excluded_fingerprints': excluded, 'blobs': blobs,
        'binding': {'generation_manifest_sha256': expected_manifest_sha256, 'fresh_pool_sha256': manifest['pool_sha256'],
            'exclusions_sha256': manifest['exclusions_sha256'], 'source_index_acceptance_sha256': manifest['source_index_acceptance_sha256'],
            'registered_source_count': len(registered['excluded_fingerprints']), 'registered_fresh_count': len(parsed['fingerprints']),
            'combined_count': len(excluded), 'combined_fingerprints_sha256': digest(excluded),
            'fresh_audit_pass_claimed': False, 'scope': 'All registered source starts plus the complete registered fresh pool; not every visited state'}}


def _request(configuration, pool_id, seed_start, count, attempt_cap, fixture):
    _require(type(fixture) is bool and type(seed_start) is int and seed_start >= 0, 'Explicit fixture scope and nonnegative seed required')
    _require(type(pool_id) is str and re.fullmatch('[A-Za-z0-9_-]{3,80}', pool_id) is not None, 'Separate question-pool ID required')
    horizon = configuration.get('horizon')
    _require(type(count) is int and type(attempt_cap) is int and type(horizon) is int,
             'Integer bounded sampling parameters required')
    _require((fixture and count == 4 and 4 <= attempt_cap <= 12 and 1 <= horizon <= 3)
        or (not fixture and count == COUNT and attempt_cap == ATTEMPT_CAP and horizon == 120),
        'Production fixes 24 states, 10000 attempts and horizon 120; fixture fixes four short states')
    config = collaborative_study_config(horizon=horizon)
    _same(configuration, asdict(config), 'Original public physics and map configuration must be unchanged')
    return config


def _sample(configuration, excluded_fingerprints, pool_id, seed_start, count, attempt_cap, fixture):
    config = _request(configuration, pool_id, seed_start, count, attempt_cap, fixture)
    env = NativeWarehouseEnv(config); variation = random.Random(seed_start+918271)
    excluded = set(excluded_fingerprints); scenes = []; seen = set(); attempts = []
    for offset in range(attempt_cap):
        seed = seed_start+offset; env.reset(seed=seed)
        for agent in env.state.agents: agent.battery = float(variation.choice(fresh_generator.BATTERIES))
        env.state.episode_id = 1; env._episode_counter = 1
        fingerprint = scenario_fingerprint(env)
        reason = 'registered_exclusion' if fingerprint in excluded else 'duplicate_draw' if fingerprint in seen else 'accepted'
        attempts.append({'seed': seed, 'fingerprint': fingerprint, 'result': reason})
        if reason != 'accepted': continue
        seen.add(fingerprint)
        scenes.append({'id': f'question_{pool_id}_{len(scenes):04d}', 'seed': seed,
            'fingerprint': fingerprint, 'snapshot': deepcopy(env.snapshot())})
        if len(scenes) == count: break
    return {'scenes': scenes, 'attempts': attempts, 'counts': _counts(len(attempts)),
        'complete': len(scenes) == count,
        'rejections': {name: sum(row['result'] == name for row in attempts) for name in ('registered_exclusion', 'duplicate_draw')}}


def generate(fresh_pool_root, *, expected_fresh_generation_manifest_sha256, pool_id, seed_start,
             output, allow_test_fixture=False, count=COUNT, attempt_cap=ATTEMPT_CAP):
    """Register new question starts only; no question labels or NN are generated."""
    origin = _path(fresh_pool_root); root = _path(output)
    _require(not root.exists() and root != origin and root not in origin.parents and origin not in root.parents,
        'A new independent output directory is required')
    source = _fresh_source(origin, expected_fresh_generation_manifest_sha256, allow_test_fixture)
    _request(source['configuration'], pool_id, seed_start, count, attempt_cap, allow_test_fixture)
    code = sources()
    plan = {'version': VERSION, 'test_fixture': allow_test_fixture, 'purpose': PURPOSE, 'pool_id': pool_id,
        'seed_start': seed_start, 'count': count, 'attempt_cap': attempt_cap, 'configuration': source['configuration'],
        'source_scenario_manifest_sha256': source['source_scenario_manifest_sha256'], 'exclusion_binding': source['binding'],
        'selection_rule': SELECTION, 'battery_values': list(fresh_generator.BATTERIES), 'sources': code,
        'joint_environment_step_cap': 0, 'neural_query_cap': 0, 'qualification_evaluated': False}
    root.mkdir(parents=True, exist_ok=False); _write(root, 'plan.json', _encoded(plan)); sampled = None
    try:
        for name, raw in source['blobs'].items(): _write(root, 'sources/'+name, raw)
        sampled = _sample(source['configuration'], source['excluded_fingerprints'], pool_id, seed_start, count, attempt_cap, allow_test_fixture)
        _require(sampled['complete'], 'Question draw cap exhausted; no completed question pool')
        pool = {'version': VERSION, 'test_fixture': allow_test_fixture, 'purpose': PURPOSE, 'pool_id': pool_id,
            'configuration': source['configuration'], 'source_scenario_manifest_sha256': source['source_scenario_manifest_sha256'],
            'exclusion_binding': source['binding'], 'scenes': sampled['scenes'],
            'initial_fingerprints_sha256': digest([s['fingerprint'] for s in sampled['scenes']]),
            'selection_rule': SELECTION, 'plan_sha256': sha256(_encoded(plan)).hexdigest()}
        report = {'version': VERSION, 'test_fixture': allow_test_fixture, 'status': 'physical_question_starts_registered',
            'attempts': sampled['attempts'], 'rejections': sampled['rejections'], 'counts': sampled['counts'],
            'accepted_initial_states': count, 'explanation_qualified': False, 'release_ready': False,
            'question_answers_generated': False, 'final_performance_test_executed': False}
        _attempts(plan, report, pool['scenes'], set(source['excluded_fingerprints']))
        _same([fresh._fingerprint(s, pool['configuration']) for s in pool['scenes']],
              [s['fingerprint'] for s in pool['scenes']], 'New draw physics differs before publication')
        _write(root, 'pool.json', _encoded(pool)); _write(root, 'report.json', _encoded(report))
        _same(sources(), code, 'Question generator code changed')
        artifacts = {str(p.relative_to(root)): _binding(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}
        manifest = {'version': VERSION, 'test_fixture': allow_test_fixture, 'status': 'complete',
            'artifacts': artifacts, 'pool_sha256': artifacts['pool.json']['sha256'],
            'source_generation_manifest_sha256': expected_fresh_generation_manifest_sha256,
            'counts': sampled['counts'], 'qualification_evaluated': False}
        _write(root, 'manifest.json', _encoded(manifest))
        return {'root': str(root), 'manifest_sha256': sha256(_encoded(manifest)).hexdigest(),
            'pool_sha256': manifest['pool_sha256'], 'counts': sampled['counts'], 'qualification_granted': False}
    except BaseException as error:
        _abort(root, error, sampled, allow_test_fixture)
        raise


def read_registered(root, *, expected_manifest_sha256, allow_test_fixture=False):
    """Authenticate immutable metadata and new states; zero physical execution."""
    _require(type(allow_test_fixture) is bool, 'Explicit fixture scope required')
    root = _path(root); raw = index._bytes(index._safe(root, 'manifest.json'), expected_manifest_sha256); manifest = json.loads(raw)
    _require(manifest['version'] == VERSION and manifest['status'] == 'complete'
        and manifest['test_fixture'] is allow_test_fixture and manifest['qualification_evaluated'] is False
        and not (root/'failure.json').exists(), 'Complete question-pool publication required')
    blobs = {name: _raw(root, name, binding) for name, binding in manifest['artifacts'].items()}
    source = _fresh_source(root/'sources', manifest['source_generation_manifest_sha256'], allow_test_fixture)
    expected = {'plan.json', 'pool.json', 'report.json'} | {'sources/'+name for name in source['blobs']}
    _same(sorted(blobs), sorted(expected), 'Question-pool artifact inventory is incomplete')
    for name, value in source['blobs'].items(): _same(_binding(blobs['sources/'+name]), _binding(value), 'Preserved source bytes differ')
    plan, pool, report = (json.loads(blobs[name]) for name in ('plan.json', 'pool.json', 'report.json'))
    _request(plan['configuration'], plan['pool_id'], plan['seed_start'], plan['count'], plan['attempt_cap'], allow_test_fixture)
    _same(plan['sources'], sources(), 'Registered question generator sources differ')
    for item in (plan, pool, report):
        _require(item['version'] == VERSION and item['test_fixture'] is allow_test_fixture, 'Question fixture/producer differs')
    _require(plan['purpose'] == pool['purpose'] == PURPOSE and plan['selection_rule'] == pool['selection_rule'] == SELECTION
        and plan['joint_environment_step_cap'] == plan['neural_query_cap'] == 0 and plan['qualification_evaluated'] is False,
        'Question sampling purpose or no-inference contract differs')
    _same(plan['battery_values'], list(fresh_generator.BATTERIES), 'Question initial battery support differs')
    for key in ('configuration', 'source_scenario_manifest_sha256', 'exclusion_binding'):
        expected_value = source['binding'] if key == 'exclusion_binding' else source[key]
        _same(plan[key], expected_value, 'Question source plan binding differs: '+key)
        _same(pool[key], expected_value, 'Question source pool binding differs: '+key)
    _same(pool['pool_id'], plan['pool_id'], 'Question-pool name differs')
    _same(pool['plan_sha256'], sha256(blobs['plan.json']).hexdigest(), 'Question generation plan bytes differ')
    _same(manifest['pool_sha256'], sha256(blobs['pool.json']).hexdigest(), 'Question-pool external bytes differ')
    scenes = pool['scenes']; _require(len(scenes) == plan['count'], 'Complete new question sample required')
    _same([s['id'] for s in scenes], [f"question_{plan['pool_id']}_{i:04d}" for i in range(plan['count'])], 'Question scene order differs')
    fingerprints = [fresh._fingerprint(s, pool['configuration']) for s in scenes]
    _require(len(set(fingerprints)) == len(scenes) and not set(fingerprints) & set(source['excluded_fingerprints']),
        'Question starts overlap registered source/fresh pools or each other')
    _same(pool['initial_fingerprints_sha256'], digest(fingerprints), 'Question physical matrix differs')
    _require(all(a['battery'] in fresh_generator.BATTERIES for s in scenes for a in s['snapshot']['state']['agents']),
        'Question initial battery distribution differs')
    _attempts(plan, report, scenes, set(source['excluded_fingerprints']))
    _same(manifest['counts'], report['counts'], 'Committed question draw counts differ')
    _require(report['status'] == 'physical_question_starts_registered' and report['accepted_initial_states'] == plan['count']
        and report['explanation_qualified'] is False and report['release_ready'] is False
        and report['question_answers_generated'] is False and report['final_performance_test_executed'] is False,
        'Physical question registration cannot grant answer or model qualification')
    blobs['manifest.json'] = raw
    return {'version': VERSION, 'root': str(root), 'manifest_sha256': expected_manifest_sha256,
        'configuration': deepcopy(pool['configuration']), 'source_scenario_manifest_sha256': pool['source_scenario_manifest_sha256'],
        'scenes': deepcopy(scenes), 'excluded_fingerprints': source['excluded_fingerprints'],
        'pool_sha256': manifest['pool_sha256'], 'exclusion_binding': deepcopy(source['binding']), 'sources': plan['sources'],
        'test_fixture': allow_test_fixture, 'blobs': blobs, 'qualification_granted': False}
