"""Physically sampled, preregistered fresh explanation initial states.

This component draws no NN/tree predictions and executes no joint steps. It
uses a separate verified metadata-only exclusion catalog; excluded snapshots
are never supplied to the sampler. Production generation is scheduled only
after the new fixed Actor and its same-Actor tree have been frozen.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re

from backend.training import warehouse_family_fresh_explanation_run as audit
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.scenarios import scenario_fingerprint

VERSION = 'warehouse-family-physical-fresh-explanation-pool-generator.v1'
PRODUCTION_COUNT = 100
PRODUCTION_ATTEMPT_CAP = 10000
BATTERIES = (60, 70, 80, 90, 100)
SELECTION = 'physical_uniqueness_and_registered_exclusions_only'


def sources():
    from backend.training import warehouse_family_explanation_source_index as indices
    values = audit.sources()
    for path in (Path(__file__), Path(indices.__file__)):
        values[str(path.relative_to(ROOT))] = file_hash(path)
    return values


def _request(configuration, pool_id, seed_start, count, attempt_cap, fixture):
    if type(fixture) is not bool or type(seed_start) is not int or seed_start < 0:
        raise ValueError('Explicit fixture scope and nonnegative seed are required')
    if type(pool_id) is not str or not re.fullmatch('[A-Za-z0-9_-]{3,80}', pool_id):
        raise ValueError('A separate pool ID is required')
    if type(count) is not int or type(attempt_cap) is not int or not 0 < count <= attempt_cap:
        raise ValueError('Positive finite sample and attempt counts are required')
    horizon = configuration.get('horizon')
    if type(horizon) is not int:
        raise ValueError('An integer public horizon is required')
    if (not fixture and (count != PRODUCTION_COUNT or attempt_cap != PRODUCTION_ATTEMPT_CAP or horizon != 120)
            or fixture and (not 1 <= count <= 2 or not 1 <= attempt_cap <= 12 or not 1 <= horizon <= 3)):
        raise ValueError('Production fixes 100 states / 10000 attempts / horizon 120')
    config = collaborative_study_config(horizon=horizon)
    if configuration != asdict(config):
        raise ValueError('Original public physics and map configuration must be unchanged')
    return config


def _sample(configuration, *, excluded_fingerprints, pool_id, seed_start,
            count, attempt_cap, test_fixture):
    """One owned environment and RNG; select solely by physical fingerprints."""
    config = _request(configuration, pool_id, seed_start, count, attempt_cap, test_fixture)
    excluded = set(excluded_fingerprints)
    for value in excluded:
        audit._sha(value)
    env = NativeWarehouseEnv(config)
    variation = random.Random(seed_start + 918271)
    scenes, accepted, attempts = [], set(), []
    for offset in range(attempt_cap):
        current_seed = seed_start + offset
        env.reset(seed=current_seed)
        for agent in env.state.agents:
            agent.battery = float(variation.choice(BATTERIES))
        env.state.episode_id = 1
        env._episode_counter = 1
        fingerprint = scenario_fingerprint(env)
        reason = 'registered_exclusion' if fingerprint in excluded else 'duplicate_draw' if fingerprint in accepted else 'accepted'
        # This is a new draw, not an excluded source snapshot or model score.
        attempts.append({'seed': current_seed, 'fingerprint': fingerprint, 'result': reason})
        if reason != 'accepted':
            continue
        accepted.add(fingerprint)
        scenes.append({'id': f'explanation_test_{pool_id}_{len(scenes):04d}',
            'seed': current_seed, 'fingerprint': fingerprint,
            'snapshot': json.loads(json.dumps(env.snapshot()))})
        if len(scenes) == count:
            break
    counts = {'environment_constructions': 1, 'environment_resets': len(attempts),
        'joint_environment_steps': 0, 'neural_forwards': 0, 'actor_loads': 0,
        'checkpoint_decodes': 0, 'tree_fits': 0}
    return {'scenes': scenes, 'attempts': attempts, 'counts': counts,
        'complete': len(scenes) == count,
        'rejections': {name: sum(x['result'] == name for x in attempts)
                       for name in ('registered_exclusion', 'duplicate_draw')}}


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        file.write(raw); file.flush(); os.fsync(file.fileno())


def _json(path, value):
    _write(path, (canonical(value) + '\n').encode())


def _binding(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': file_hash(path), 'size': path.stat().st_size}


def generate(index_root, *, expected_exclusions_sha256, expected_source_index_acceptance_sha256,
             pool_id, seed_start, output, allow_test_fixture=False,
             count=PRODUCTION_COUNT, attempt_cap=PRODUCTION_ATTEMPT_CAP):
    """Publish a new pool; incomplete draws retain their plan and failure record."""
    from backend.training import warehouse_family_explanation_source_index as indices
    index_root = Path(index_root).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if (index_root.resolve() != index_root or output.resolve() != output or output.exists()
            or output == index_root or index_root in output.parents or output in index_root.parents):
        raise ValueError('Use separate canonical input and new output directories')
    registered = indices.read_registered(index_root,
        expected_exclusions_sha256=expected_exclusions_sha256,
        expected_source_index_acceptance_sha256=expected_source_index_acceptance_sha256,
        allow_test_fixture=allow_test_fixture)
    if registered['test_fixture'] is not allow_test_fixture:
        raise ValueError('Exclusion fixture scope differs')
    configuration = registered['configuration']
    _request(configuration, pool_id, seed_start, count, attempt_cap, allow_test_fixture)
    closure = sources()
    plan = {'version': VERSION, 'test_fixture': allow_test_fixture, 'pool_id': pool_id,
        'seed_start': seed_start, 'count': count, 'attempt_cap': attempt_cap,
        'configuration': configuration, 'source_scenario_manifest_sha256': registered['source_scenario_manifest_sha256'],
        'exclusions_sha256': expected_exclusions_sha256,
        'source_index_acceptance_sha256': expected_source_index_acceptance_sha256,
        'excluded_fingerprints_sha256': digest(sorted(registered['excluded_fingerprints'])),
        'selection_rule': SELECTION, 'battery_values': list(BATTERIES), 'sources': closure,
        'joint_environment_step_cap': 0, 'neural_query_cap': 0,
        'final_performance_test_executed': False, 'qualification_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    _json(output/'generation_plan.json', plan)
    result = None
    try:
        for relative, raw in registered['blobs'].items():
            part = Path(relative)
            if (part.is_absolute() or '..' in part.parts or str(part) != relative
                    or relative in ('pool.json', 'generation_plan.json', 'generation_report.json',
                                    'generation_manifest.json', 'generation_failure.json')):
                raise ValueError('Unsafe or reserved metadata-only index path')
            _write(output/part, raw)
        result = _sample(configuration, excluded_fingerprints=registered['excluded_fingerprints'],
            pool_id=pool_id, seed_start=seed_start, count=count, attempt_cap=attempt_cap,
            test_fixture=allow_test_fixture)
        if not result['complete']:
            raise ValueError('Attempt cap exhausted; no complete pool was published')
        pool = {'version': audit.POOL_VERSION, 'test_fixture': allow_test_fixture,
            'purpose': 'independent_explanation_test', 'pool_id': pool_id,
            'source_scenario_manifest_sha256': registered['source_scenario_manifest_sha256'],
            'configuration': configuration, 'scenes': result['scenes'],
            'initial_fingerprints_sha256': digest([x['fingerprint'] for x in result['scenes']]),
            'exclusions_sha256': expected_exclusions_sha256,
            'generation': {'version': VERSION, 'selection_rule': SELECTION,
                'seed_start': seed_start, 'sources_sha256': digest(closure),
                'generation_plan_sha256': file_hash(output/'generation_plan.json')}}
        _json(output/'pool.json', pool)
        # The real independent admission checks physical content and all three
        # external identities; no runtime, forward, restore or step is created.
        audit._pool(output, file_hash(output/'pool.json'), expected_exclusions_sha256,
                    expected_source_index_acceptance_sha256, allow_test_fixture)
        if sources() != closure:
            raise ValueError('Generator source changed during generation')
        report = {'version': VERSION, 'status': 'physically_registered_only',
            'test_fixture': allow_test_fixture, 'counts': result['counts'],
            'attempts': result['attempts'], 'rejections': result['rejections'],
            'accepted_initial_states': len(result['scenes']),
            'excluded_initial_states': len(registered['excluded_fingerprints']),
            'initial_fingerprints_sha256': pool['initial_fingerprints_sha256'],
            'all_source_exclusions_applied': True, 'nn_or_tree_based_selection': False,
            'generalization_scope': 'same_map_initial_states_not_all_visited_states_or_new_topology',
            'explanation_qualified': False, 'release_ready': False,
            'final_performance_test_executed': False}
        _json(output/'generation_report.json', report)
        files = sorted(p for p in output.rglob('*') if p.is_file())
        manifest = {'version': VERSION, 'status': 'complete', 'test_fixture': allow_test_fixture,
            'artifacts': {str(p.relative_to(output)): _binding(output, p) for p in files},
            'pool_sha256': file_hash(output/'pool.json'),
            'exclusions_sha256': expected_exclusions_sha256,
            'source_index_acceptance_sha256': expected_source_index_acceptance_sha256,
            'counts': result['counts'], 'qualification_evaluated': False}
        _json(output/'generation_manifest.json', manifest)
        return manifest
    except BaseException as error:
        _json(output/'generation_failure.json', {'version': VERSION, 'reason': repr(error),
            'test_fixture': allow_test_fixture, 'completed_pool_published': False,
            'actual_sampling': result, 'release_ready': False})
        raise
