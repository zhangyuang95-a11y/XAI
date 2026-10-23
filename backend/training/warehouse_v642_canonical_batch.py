"""Read-only numerical diagnostics for the unchanged frozen v6.20 forward.

This module is not an execution or training adapter.  An engineering candidate
is acceptable here only when every float32 bit equals the original physical-
frame calculation.  Diagnostic input-domain probes are not new trajectories,
and a numerical result grants no seed, training, audit or pilot authority.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import numpy as np

from . import warehouse_v6_actor as runtime
from .warehouse_v622_program import MASK_INDICES, masked_probabilities
from .warehouse_v626_domain import domain_table, validate_rows
from env.warehouse.public_action_features_v617 import append_context
from env.warehouse import public_action_features_v614 as legacy
from experiments.warehouse.v626_protocol import pointer, bound, require, file_sha, digest

VERSION = 'warehouse-v642-canonical-batch-numerical-diagnostic.v1'
METHODS = ('w_left_vectors', 'frame_matrices', 'frame_matrices_scalar_ufunc')
ROLES = ('robot_1', 'robot_2')
SPLITS = ('fit', 'calibration', 'eligibility', 'validation')


def source_files():
    root = Path(__file__).resolve().parents[2]
    from .warehouse_v642_observed_pool import source_files as original_sources
    names = ('backend/training/warehouse_v642_canonical_batch.py',
             'backend/training/warehouse_v6_actor.py',
             'backend/training/warehouse_v6_tree_routing.py',
             'backend/training/warehouse_v622_program.py',
             'backend/training/warehouse_v626_domain.py',
             'env/warehouse/public_action_features_v614.py',
             'env/warehouse/public_action_features_v617.py')
    files = original_sources()
    files.update({name: file_sha(root/name) for name in names})
    for name in ('tests/test_warehouse_v642_canonical_batch.py', 'docs/warehouse_v642_canonical_batch.md'):
        files[name] = file_sha(root/name)
    return dict(sorted(files.items()))


def bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def array_sha(a):
    a = np.asarray(a)
    return sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _input(role, x):
    x = validate_rows(x)
    require(x.ndim == 2 and len(x) and role.model_version == runtime.MODEL_VERSION
            and role.core_bound_mode == runtime.CORE_BOUND_MODE,
            'diagnostic only supports the unchanged additive v6.20 Actor')
    require(np.array_equal(append_context(x[:, :legacy.TREE_FEATURE_DIM]), x),
            'diagnostic context differs from its public sources')
    m = x[:, MASK_INDICES]
    require(np.isin(m, (0, 1)).all() and np.all(m[:, 4] == 1),
            'diagnostic masks must remain physical binary masks with legal wait')
    return x


def _affine(a, weight, bias, method, *, gate=False):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if method == 'w_left_vectors':
            prefix = (None,) * (a.ndim - 1)
            return np.matmul(weight[prefix], a[..., None]).squeeze(-1) + bias
        if gate and method in ('frame_matrices', 'frame_matrices_scalar_ufunc'):
            # Preserve the original inner M=1 gate matrix, instead of GEMM M=N.
            return np.matmul(a[:, None, :], weight.T)[:, 0, :] + bias
        return a @ weight.T + bias


def _per_frame(fn, a, method):
    if method == 'frame_matrices_scalar_ufunc':
        return np.concatenate([fn(row[None]) for row in a], axis=0)
    return fn(a)


def forward_trace(role, x, method):
    """Return each numerical layer; candidates remain diagnostic-only."""
    require(method in (*METHODS, 'original_shape'), 'unknown numerical candidate')
    x = _input(role, x)
    if method == 'original_shape':
        require(len(x) == 1, 'canonical tracing keeps exactly one physical frame')
    w = role.weights
    out = {'leaves': role.leaf(x)}
    tree = w['tree.leaf_logits'][out['leaves']]
    own = x[:, list(runtime._POSITION_COLUMNS[:2])]
    partner = x[:, list(runtime._POSITION_COLUMNS[2:])]
    relative = own - partner
    gate_input = np.concatenate((x, relative, np.abs(relative)), axis=-1)
    out['gate_input'] = gate_input
    gh = _affine(gate_input, w['gate.0.weight'], w['gate.0.bias'], method, gate=True)
    out['gate_affine_0'] = gh
    gh = np.maximum(gh, 0)
    out['gate_hidden'] = gh
    g = _affine(gh, w['gate.2.weight'], w['gate.2.bias'], method, gate=True)
    out['gate_affine_2'] = g
    # The original computes this sigmoid before replacing it with tanh.
    out['unused_gate_sigmoid'] = _per_frame(lambda z: 1.0/(1.0+np.exp(-z)), g, method)
    gate = _per_frame(np.tanh, g, method)
    out['gate_tanh'] = gate
    gap = own[:, None, :] + runtime._MOVE[None] - partner[:, None, :]
    geometry = np.concatenate((np.broadcast_to(relative[:, None, :], gap.shape),
                               gap, np.abs(gap)), axis=-1)
    relation = np.concatenate((
        np.broadcast_to(x[:, None, :], (len(x), 5, runtime.FEATURE_DIM)),
        geometry, np.broadcast_to(np.eye(5, dtype=np.float32)[None], (len(x), 5, 5)),
        runtime.action_features_from_tree_encoded(x)), axis=-1)
    out['relation'] = relation
    h0 = _affine(relation, w['action_residual.0.weight'], w['action_residual.0.bias'], method)
    out['residual_affine_0'] = h0
    h = np.maximum(h0, 0)
    out['residual_hidden_0'] = h
    h2 = _affine(h, w['action_residual.2.weight'], w['action_residual.2.bias'], method)
    out['residual_affine_2'] = h2
    h = np.maximum(h2, 0)
    out['residual_hidden_2'] = h
    score = _affine(h, w['action_residual.4.weight'], w['action_residual.4.bias'], method).squeeze(-1)
    out['residual_score'] = score
    tree = gate * (tree - tree.mean(axis=-1, keepdims=True))
    out['tree_contribution'] = tree
    out['logits'] = tree + score
    masks = x[:, MASK_INDICES]
    if method == 'frame_matrices_scalar_ufunc':
        out['probabilities'] = np.stack([masked_probabilities(z, m)
            for z, m in zip(out['logits'], masks, strict=True)])
    else:
        out['probabilities'] = masked_probabilities(out['logits'], masks)
    require(all(np.isfinite(a).all() for a in out.values()), 'nonfinite diagnostic forward')
    return out


def canonical_trace(role, x):
    """Trace, then independently cross-check against the original runtime."""
    x = _input(role, x)
    frames = []
    for row in x:
        trace = forward_trace(role, row[None], 'original_shape')
        z = role.logits_encoded(row)
        p = masked_probabilities(z, row[list(MASK_INDICES)])
        require(bits_equal(trace['logits'][0], z) and bits_equal(trace['probabilities'][0], p),
                'diagnostic transcription differs from canonical frozen forward')
        frames.append(trace)
    return {key: np.concatenate([frame[key] for frame in frames], axis=0)
            for key in frames[0]}


def differences(reference, candidate):
    require(set(reference) == set(candidate), 'diagnostic layers differ')
    result = {}
    for key in reference:
        a, b = reference[key], candidate[key]
        equal = bits_equal(a, b)
        require(a.shape == b.shape and a.dtype == b.dtype, 'diagnostic shape or dtype changed')
        # Bytewise equality includes signed zero. No tolerance is used.
        bytes_a = np.ascontiguousarray(a).view(np.uint8).reshape(*a.shape, a.dtype.itemsize)
        bytes_b = np.ascontiguousarray(b).view(np.uint8).reshape(*b.shape, b.dtype.itemsize)
        result[key] = dict(bitwise_equal=equal,
            differing_elements=int(np.any(bytes_a != bytes_b, axis=-1).sum()),
            maximum_absolute_difference=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
            reference_sha256=array_sha(a), candidate_sha256=array_sha(b))
    return result


def numerical_probes(x, *, count=512, control=642):
    """Bounded numeric perturbations of OLD rows, never new trajectories.

    Keep cached passable coordinates, categorical state, map and active bits;
    perturb public time/battery/history scalars and rebuild physical extras.
    These probes satisfy the runtime input contract, but are not claimed to be
    complete physically reachable environment states or independent samples.
    """
    require(type(count) is int and count >= 2 and type(control) is int,
            'explicit finite numerical-test controls required')
    x = validate_rows(x)
    rng = np.random.default_rng(control)  # Numerical unit-test control only.
    indices = rng.integers(0, len(x), count)
    obs = x[indices, :legacy.OBSERVATION_DIM].copy()
    masks = x[indices][:, MASK_INDICES].copy()
    table = domain_table()['features'][:legacy.OBSERVATION_DIM]
    for i, feature in enumerate(table):
        name = feature['name']
        if name == 'elapsed_fraction' or name == 'same_target_collision_streak_capped' or name.endswith(
                ('battery_fraction', 'last_battery_delta', 'steps_since_charging', 'charger_wait_streak')):
            lo, hi = np.float32(feature['lower']), np.float32(feature['upper'])
            obs[:, i] = lo+(hi-lo)*rng.random(count, dtype=np.float32)
            obs[0, i], obs[1, i] = lo, hi
    names = runtime.FEATURE_NAMES
    area = np.float32(domain_table()['map_area'])
    for role in ('self', 'partner'):
        capacity = obs[:, names.index(role+'_battery_fraction')]
        obs[:, names.index(role+'_feasible_move_steps')] = capacity
        distance = obs[:, names.index(role+'_charger_shortest_distance')]
        obs[:, names.index(role+'_charger_return_margin')] = capacity-distance*area/np.float32(50)
        for order in (0, 1):
            stem = f'order_{order}_{role}'
            distance = obs[:, names.index(stem+'_remaining_mission_distance')]
            obs[:, names.index(stem+'_charger_return_margin')] = capacity-distance*area/np.float32(50)
    out = append_context(legacy.encode_actor_tree_features(obs, masks))
    validate_rows(out)
    require(np.isin(out[:, MASK_INDICES], (0, 1)).all(), 'nonbinary numerical probe mask')
    return out


def compare(role, x, *, methods=METHODS):
    start = perf_counter()
    reference = canonical_trace(role, x)
    scalar_seconds = perf_counter()-start
    results = {}
    for method in methods:
        start = perf_counter()
        candidate = forward_trace(role, x, method)
        elapsed = perf_counter()-start
        diff = differences(reference, candidate)
        results[method] = dict(seconds=elapsed, all_layers_bitwise_equal=all(
            item['bitwise_equal'] for item in diff.values()), layers=diff)
    return dict(physical_input_rows=len(x), input_sha256=array_sha(x),
                original_trace_and_independent_runtime_seconds=scalar_seconds, candidates=results)


def canonical_probabilities(role, x):
    x = _input(role, x)
    return np.stack([masked_probabilities(role.logits_encoded(row), row[list(MASK_INDICES)]) for row in x])


def candidate_probabilities(role, x, *, chunk_size=256, method='frame_matrices_scalar_ufunc'):
    require(method in METHODS and type(chunk_size) is int and chunk_size > 0,
            'finite diagnostic chunk and known numerical method required')
    x = _input(role, x)
    return np.concatenate([forward_trace(role, x[start:start+chunk_size], method)['probabilities']
        for start in range(0, len(x), chunk_size)], axis=0)


def benchmark_probabilities(role, x, *, stored=None, chunk_sizes=(1, 7, 64, 256), method='frame_matrices_scalar_ufunc'):
    """Recompute all given OLD frames, without trusting cached targets."""
    start = perf_counter()
    expected = canonical_probabilities(role, x)
    scalar_seconds = perf_counter()-start
    if stored is not None:
        require(bits_equal(expected, stored), 'old cache differs from canonical full-NN forward')
    output = {}
    for chunk in chunk_sizes:
        start = perf_counter()
        found = candidate_probabilities(role, x, chunk_size=chunk, method=method)
        elapsed = perf_counter()-start
        output[str(chunk)] = dict(seconds=elapsed, bitwise_equal=bits_equal(expected, found),
            probability_sha256=array_sha(found), scalar_speedup=scalar_seconds/elapsed,
            differing_probabilities=int(np.any(np.ascontiguousarray(expected).view(np.uint8).reshape(-1, 5, 4)
                != np.ascontiguousarray(found).view(np.uint8).reshape(-1, 5, 4), axis=-1).sum()))
    return dict(frames=len(x), input_sha256=array_sha(x), canonical_seconds=scalar_seconds,
        canonical_probability_sha256=array_sha(expected), stored_target_bitwise_verified=stored is not None,
        method=method, chunks=output,
        all_chunks_bitwise_equal=all(item['bitwise_equal'] for item in output.values()))


def verify_all_layers(role, x, *, stored=None, chunk_size=256,
                      methods=('frame_matrices_scalar_ufunc',), candidate_chunk_sizes=None,
                      probability_observer=None, scalar_output_observer=None):
    """Streaming exhaustive layer/NN comparisons on the supplied old frames."""
    require(type(chunk_size) is int and chunk_size > 0 and len(set(methods)) == len(methods)
            and all(method in METHODS for method in methods), 'invalid diagnostic comparison budget')
    require(probability_observer is None or callable(probability_observer), 'invalid numerical observer')
    require(scalar_output_observer is None or callable(scalar_output_observer), 'invalid scalar numerical observer')
    if candidate_chunk_sizes is None:
        variants = [(method, method, chunk_size) for method in methods]
    else:
        require(candidate_chunk_sizes and len(set(candidate_chunk_sizes)) == len(candidate_chunk_sizes)
                and all(type(n) is int and n > 0 for n in candidate_chunk_sizes), 'invalid candidate chunk inventory')
        variants = [(f'{method}-chunk-{chunk}', method, chunk) for method in methods for chunk in candidate_chunk_sizes]
    keys = tuple(item[0] for item in variants)
    x = _input(role, x)
    times = dict(reference_trace_plus_independent_original_runtime=0., **{m: 0. for m in keys})
    counts, hashes = {m: {} for m in keys}, {m: {} for m in ('reference', *keys)}
    stored_checked = stored is not None
    if stored_checked:
        require(np.asarray(stored).shape == (len(x), 5), 'old probability cache shape differs')
    for start in range(0, len(x), chunk_size):
        rows = x[start:start+chunk_size]
        t = perf_counter()
        reference = canonical_trace(role, rows)
        times['reference_trace_plus_independent_original_runtime'] += perf_counter()-t
        if stored_checked:
            require(bits_equal(reference['probabilities'], stored[start:start+len(rows)]),
                    'old full target cache differs from canonical original runtime')
        if scalar_output_observer is not None:
            scalar_output_observer(reference['logits'], reference['probabilities'], start)
        for key, value in reference.items():
            hashes['reference'].setdefault(key, sha256()).update(value.tobytes())
        candidate_probabilities = {}
        for key_name, method, chunk in variants:
            t = perf_counter()
            traces = [forward_trace(role, rows[offset:offset+chunk], method) for offset in range(0, len(rows), chunk)]
            candidate = {key: np.concatenate([trace[key] for trace in traces], axis=0) for key in traces[0]}
            times[key_name] += perf_counter()-t
            diff = differences(reference, candidate)
            for key, value in candidate.items():
                hashes[key_name].setdefault(key, sha256()).update(value.tobytes())
                item = counts[key_name].setdefault(key, dict(differing_elements=0, maximum_absolute_difference=0.))
                item['differing_elements'] += diff[key]['differing_elements']
                item['maximum_absolute_difference'] = max(item['maximum_absolute_difference'],
                    diff[key]['maximum_absolute_difference'])
            candidate_probabilities[key_name] = candidate['probabilities']
        if probability_observer is not None:
            probability_observer(reference['probabilities'], candidate_probabilities, start)
    results = {}
    for method in keys:
        layers = {key: dict(**item, bitwise_equal=item['differing_elements'] == 0,
            reference_bytes_sha256=hashes['reference'][key].hexdigest(),
            candidate_bytes_sha256=hashes[method][key].hexdigest()) for key, item in counts[method].items()}
        results[method] = dict(all_layers_bitwise_equal=all(item['bitwise_equal'] for item in layers.values()),
            layers=layers)
    return dict(frames=len(x), input_sha256=array_sha(x), chunk_size=chunk_size,
        candidate_chunk_sizes=None if candidate_chunk_sizes is None else list(candidate_chunk_sizes),
        stored_canonical_targets_bitwise_verified=stored_checked, seconds=times, candidates=results,
        reference_timing_is_not_a_pure_runtime_speed_benchmark=True)


def profile_qualification(case):
    """Recompute the existing reports, without changing their qualification."""
    from .warehouse_v642_program import Program
    from .warehouse_v629_online_evidence import current_report
    out = {}
    for role in ROLES:
        ref = case['event']['programs'][role]
        value = bound(ref)
        start = perf_counter()
        program = Program.from_dict(value, value['target_actor_role_sha256'])
        loaded = perf_counter()-start
        start = perf_counter()
        report = current_report(program, case['parts'][role])
        elapsed = perf_counter()-start
        require(digest(report) == case['event']['reports'][role],
                'read-only qualification recomputation differs from the old committed event')
        out[role] = dict(program=ref, program_load_and_proof_seconds=loaded,
            current_report_seconds=elapsed, report_sha256=digest(report),
            original_event_report_identical=True)
    return out


def profile_source_hashes(case):
    """Time the original repeated source-ref loop and its distinct readset.

    This does not install a cache or waive any immutable source check.  It
    measures existing I/O only; no full physics/source admission is asserted.
    """
    source_ref = case['result']['observed_source']
    source = bound(source_ref)
    refs = [source_ref, source['owner'], source['inventory']]
    for item in source['cases'].values():
        refs.extend(item.values())
    out = {}
    unique = {(ref['path'], ref['sha256']): ref for ref in refs}
    for name, chosen in (('original_repeated_ref_loop', refs), ('distinct_refs', list(unique.values()))):
        byte_count = 0
        start = perf_counter()
        for ref in chosen:
            require(pointer(ref['path']) == ref and not Path(ref['path']).is_symlink(),
                    'old source changed while profiling')
            byte_count += Path(ref['path']).stat().st_size
        out[name] = dict(seconds=perf_counter()-start, pointer_calls=len(chosen), bytes_hashed=byte_count)
    out['source'] = source_ref
    out['full_physics_replay_performed'] = False
    out['optimization_installed'] = False
    return out


def load_old_target(result_ref, event_ref, *, splits=SPLITS):
    """Read only a completed native call; never complete partial/new calls."""
    require(splits and len(set(splits)) == len(splits) and set(splits) <= set(SPLITS),
            'unknown or empty old diagnostic partitions')
    a, event = bound(result_ref), bound(event_ref)
    require(a.get('schema') == 'warehouse-v642-counterfactual-NN-targets.v1'
        and a['environment_resets'] == a['NN_updates'] == a['PPO_joint_steps'] == 0
        and a['current_policy_trajectory'] is False and a['fresh_randomness_claimed'] is False
        and event['kind'] == 'validation' and event['target_actor'] == a['target_actor']
        and event['target_package'] == result_ref,
        'diagnostic requires an existing completed native target/event binding')
    refs = [result_ref, event_ref, a['target_actor'], a['observed_source'], *a['asset_refs']]
    for ref in refs:
        require(pointer(ref['path']) == ref and not Path(ref['path']).is_symlink(), 'old diagnostic input changed')
    actor = runtime.FrozenWarehouseV6Actor.load(a['target_actor']['path'], expected_sha256=a['target_actor']['sha256'])
    parts = {}
    for role in ROLES:
        pool = bound(a['pools'][role])
        require(pool['target_actor'] == a['target_actor'] and pool['role'] == role
                and set(pool['parts']) == set(SPLITS), 'old role target binding differs')
        parts[role] = {}
        for split in splits:
            item = pool['parts'][split]
            require(pointer(item['arrays']['path']) == item['arrays'], 'old target arrays changed')
            with np.load(item['arrays']['path'], allow_pickle=False) as archive:
                require(set(archive.files) == {'x', 'm', 'p', 'sampled'}, 'unknown native target array keys')
                value = {key: archive[key].copy() for key in archive.files}
            value['rows'] = bound(item['rows'])
            _input(actor.roles[role], value['x'])
            require(bits_equal(value['m'], value['x'][:, MASK_INDICES])
                    and len(value['rows']) == len(value['x']), 'old target masks or rows differ')
            parts[role][split] = value
    return dict(result=a, event=event, actor=actor, parts=parts, inputs=refs)


def run_finite_diagnostic(calls, *, output, trace_rows=128, numeric_rows=512,
                          benchmark_rows=1024, chunk_sizes=(1, 7, 64, 256),
                          profile_hashes=False):
    """Finite old-data-only report; full-pool numerical claims stay false.

    ``calls`` contains explicit already-complete result/event pointers.  No
    seed-plan, readiness or execution authorization is acquired or produced.
    This deliberately bounded diagnostic protects a concurrent real recovery.
    Full-pool ``verify_all_layers`` is a separate opt-in post-recovery task.
    """
    require(calls and all(type(n) is int and n > 0 for n in (trace_rows, numeric_rows, benchmark_rows)),
            'explicit finite diagnostic inputs required')
    folder = Path(output)
    require(not folder.exists() and not folder.is_symlink(), 'diagnostic output already exists')
    sources = source_files()
    folder.mkdir(parents=True)
    from scripts.run_warehouse_v625 import write_new
    import platform
    import os
    report = dict(version=VERSION, source_files=sources, source_sha256=digest(sources),
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        thread_environment={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        calls=[], environment_resets=0, new_environment_or_action_seeds=0, NN_updates=0,
        fitting_calls=0, new_scientific_samples=0, full_pool_numerical_verification=False,
        original_scalar_runtime_unchanged=True, optimized_runtime_installed=False,
        training_authorized=False, sealed_audit_authorized=False, pilot_opened=False)
    write_new(folder/'started.json', dict(version=VERSION, source_sha256=report['source_sha256'],
        calls=calls, trace_rows=trace_rows, numeric_rows=numeric_rows, benchmark_rows=benchmark_rows,
        full_pool_numerical_verification=False))
    for index, call in enumerate(calls):
        case = load_old_target(call['result'], call['event'], splits=('validation',))
        item = dict(result=call['result'], event=call['event'],
            actor=case['result']['target_actor'], actor_role_sha256=case['actor'].role_sha256,
            original_call=case['event']['sequence_call'], joint_step=case['event']['step'], roles={})
        for role in ROLES:
            part = case['parts'][role]['validation']
            indices = np.linspace(0, len(part['x'])-1, min(trace_rows, len(part['x'])), dtype=np.int64)
            cached = part['x'][indices]
            probes = numerical_probes(cached, count=numeric_rows)
            item['roles'][role] = dict(
                old_validation_rows=len(part['x']), selected_trace_indices_sha256=array_sha(indices),
                cached_layers=compare(case['actor'].roles[role], cached),
                numeric_input_probes=compare(case['actor'].roles[role], probes),
                numeric_probes_are_not_new_episodes=True,
                runtime_benchmark=benchmark_probabilities(case['actor'].roles[role],
                    part['x'][:benchmark_rows], stored=part['p'][:benchmark_rows], chunk_sizes=chunk_sizes))
        if profile_hashes and index == 0:
            item['source_pointer_hash_profile'] = profile_source_hashes(case)
        for ref in case['inputs']:
            require(pointer(ref['path']) == ref, 'old input changed during the numerical diagnostic')
        write_new(folder/f'call-{index:02d}.json', item)
        report['calls'].append(item)
    require(source_files() == sources, 'source closure changed during diagnostic')
    report['bounded_same_shape_checks_bitwise_equal'] = all(
        role[kind]['candidates']['frame_matrices_scalar_ufunc']['all_layers_bitwise_equal']
        and role['runtime_benchmark']['all_chunks_bitwise_equal']
        for item in report['calls'] for role in item['roles'].values()
        for kind in ('cached_layers', 'numeric_input_probes'))
    report['w_left_vectors_rejected'] = any(not role['cached_layers']['candidates']['w_left_vectors']['all_layers_bitwise_equal']
        for item in report['calls'] for role in item['roles'].values())
    write_new(folder/'result.json', report)
    return pointer(folder/'result.json')
