"""Complete OLD target-pool numerical check; no model/scientific admission.

Only explicit, already committed native calls are read.  Complete physical
rows are revisited numerically, not recollected.  No reset, fitting, optimizer,
new scene or new action-stream allocation exists in this module.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import os
import platform
import subprocess
import zipfile
import numpy as np

from backend.training import warehouse_v642_canonical_batch as numerical
from env.warehouse.decision_protocol import independent_agent_seed
from experiments.warehouse.v626_protocol import pointer, bound, require, digest, file_sha
from scripts.run_warehouse_v625 import write_new

VERSION = 'warehouse-v642-complete-old-canonical-numerical-check.v1'
METHOD = 'frame_matrices_scalar_ufunc'
CHUNKS = (64, 256)
SEGMENT_ROWS = 8192
REFERENCE_CHUNK = 256
EXPECTED_NATIVE_CALLS = (31, 47, 58)
ORIGINAL_SOURCE_SHA256 = '265fd2bacd027a31beae07b415a1ea12c9767f5f30e6c9b06bb11df2e0b3453f'


def source_files():
    root = Path(__file__).resolve().parents[2]
    out = numerical.source_files()
    for name in ('experiments/warehouse/v642_full_canonical_diagnostic.py',
                 'tests/test_warehouse_v642_full_canonical_diagnostic.py',
                 'docs/warehouse_v642_full_canonical_diagnostic.md'):
        out[name] = file_sha(root/name)
    return dict(sorted(out.items()))


def _shape(path, name):
    with zipfile.ZipFile(path) as archive:
        with archive.open(name+'.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    require(not fortran, 'old target arrays changed their native memory order')
    return tuple(shape), dtype.str


def inventory(calls):
    """Count existing NPY headers only; no forward or environment execution."""
    require(type(calls) is list and len(calls) == len(EXPECTED_NATIVE_CALLS), 'three explicit old completed native calls required')
    out, refs, observed, observed_data, original_owner = [], {}, None, None, None
    for expected, call in zip(EXPECTED_NATIVE_CALLS, calls, strict=True):
        require(set(call) == {'result', 'event'}, 'unknown numerical diagnostic call fields')
        result, event = bound(call['result']), bound(call['event'])
        require(event['kind'] == 'validation' and event['sequence_call'] == expected
            and event['target_package'] == call['result'] and event['target_actor'] == result['target_actor']
            and result['schema'] == 'warehouse-v642-counterfactual-NN-targets.v1'
            and result['NN_updates'] == result['PPO_joint_steps'] == result['environment_resets'] == 0
            and result['counterfactual_draws_not_executed'] is True
            and result['current_policy_trajectory'] is result['fresh_randomness_claimed'] is False,
            'partial, new, unbound or non-counterfactual old target package')
        if observed is None:
            observed = result['observed_source']
            observed_data = result['observed_data_sha256']
            original_owner = result['owner']
        require(observed['sha256'] == ORIGINAL_SOURCE_SHA256 and result['observed_data_sha256'] == observed_data
            and result['owner'] == original_owner and bound(original_owner)['variant'] == 'regularized',
            'diagnostic source is not the fixed original regularized observed pool')
        require(observed == result['observed_source'], 'old calls must share the same original regularized observed pool')
        for ref in (call['result'], call['event'], result['target_actor'], observed, *result['asset_refs']):
            require(pointer(ref['path']) == ref and not Path(ref['path']).is_symlink(), 'old target/event input changed')
            key = ref['path']
            require(key not in refs or refs[key] == ref, 'conflicting old target input hashes')
            refs[key] = deepcopy(ref)
        actor = numerical.runtime.FrozenWarehouseV6Actor.load(result['target_actor']['path'],
            expected_sha256=result['target_actor']['sha256'])
        for role in numerical.ROLES:
            pool_ref = result['pools'][role]
            pool = bound(pool_ref)
            require(pool['role'] == role and pool['target_actor'] == result['target_actor']
                and set(pool['parts']) == set(numerical.SPLITS), 'complete old role target pool differs')
            for split in numerical.SPLITS:
                part = pool['parts'][split]
                shape, dtype = _shape(part['arrays']['path'], 'x')
                n = shape[0]
                require(shape == (n, numerical.runtime.FEATURE_DIM) and dtype == np.dtype(np.float32).str and n > 0,
                        'old complete physical x shape/type differs')
                for name in ('m', 'p'):
                    require(_shape(part['arrays']['path'], name) == ((n, 5), np.dtype(np.float32).str), 'old physical mask/distribution shape differs')
                require(_shape(part['arrays']['path'], 'sampled') == ((n,), np.dtype(np.int64).str), 'old sampled action shape differs')
                out.append(dict(call=expected, role=role, split=split, rows=n, arrays=part['arrays'], rows_ref=part['rows'],
                    pool=pool_ref, actor=result['target_actor'], actor_role_sha256=actor.role_sha256,
                    role_sha256=actor.role_sha256[role], event=call['event'], result=call['result']))
    return dict(version=VERSION, partitions=out, input_refs=list(refs.values()), observed_source=observed,
        total_old_role_frames=sum(item['rows'] for item in out), partition_count=len(out),
        segment_rows=SEGMENT_ROWS, reference_chunk=REFERENCE_CHUNK, candidate_chunks=list(CHUNKS),
        original_runtime_replayed=True, recollected_episodes=0, new_scientific_samples=0,
        environment_resets=0, fitting_calls=0, NN_updates=0, training_authorized=False,
        full_future_NN_equivalence_claimed=False)


def _hex_bytes(array):
    return ''.join(float(value).hex()+'\n' for value in np.asarray(array).flat).encode()


class NumericalMismatch(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


class ScalarOutputs:
    """Retain every independently checked scalar logit/probability, ~48 MB total."""
    def __init__(self, rows, sampled):
        self.logits = np.empty((rows, 5), np.float32)
        self.probabilities = np.empty_like(self.logits)
        self.sampled, self.seen = sampled, 0

    def __call__(self, logits, probabilities, start):
        require(start == self.seen and logits.shape == probabilities.shape,
                'scalar outputs skipped/repeated physical rows')
        end = start+len(logits)
        self.logits[start:end], self.probabilities[start:end] = logits, probabilities
        self.seen = end

    def save(self, folder):
        require(self.seen == len(self.logits), 'incomplete original scalar numerical outputs')
        path = folder/'scalar-outputs.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                require(set(archive.files) == {'scalar_logits', 'scalar_probabilities', 'recorded_sampled'}
                    and numerical.bits_equal(archive['scalar_logits'], self.logits)
                    and numerical.bits_equal(archive['scalar_probabilities'], self.probabilities)
                    and numerical.bits_equal(archive['recorded_sampled'], self.sampled),
                    'uncommitted retained scalar outputs differ; never overwrite')
        else:
            with path.open('xb') as handle:
                np.savez(handle, scalar_logits=self.logits, scalar_probabilities=self.probabilities, recorded_sampled=self.sampled)
            path.chmod(0o444)
        return pointer(path)


class ProbabilityObserver:
    """Reconstruct only recorded counterfactual draws; never allocate streams."""
    def __init__(self, role, rows, stored_p, stored_actions):
        self.role, self.rows = role, rows
        self.stored_p, self.stored_actions = stored_p, stored_actions
        self.seen, self.seconds = 0, 0.
        self.hashes = {name: sha256() for name in ('stored', 'scalar', *(f'chunk-{n}' for n in CHUNKS))}
        self.actions = {name: sha256() for name in self.hashes}
        self.zero_counts = {name: dict(positive_zero=0, negative_zero=0) for name in self.hashes}

    def __call__(self, reference, candidates, start):
        t = perf_counter()
        require(start == self.seen, 'physical probability observer skipped/repeated rows')
        n = len(reference)
        versions = dict(stored=self.stored_p[start:start+n], scalar=reference,
            **{f'chunk-{chunk}': candidates[f'{METHOD}-chunk-{chunk}'] for chunk in CHUNKS})
        expected_hex = _hex_bytes(versions['stored'])
        for name, value in versions.items():
            require(numerical.bits_equal(versions['stored'], value), 'probability bytes/sign-zero differ from old target')
            encoded = expected_hex if name == 'stored' else _hex_bytes(value)
            require(encoded == expected_hex, 'float.hex differs from old target')
            self.hashes[name].update(encoded)
            zero = value == 0
            self.zero_counts[name]['negative_zero'] += int((zero & np.signbit(value)).sum())
            self.zero_counts[name]['positive_zero'] += int((zero & ~np.signbit(value)).sum())
        reconstructed = {name: [] for name in versions}
        for local, row in enumerate(self.rows[start:start+n]):
            require(type(row['action_seed']) is int and type(row['episode_id']) is int and type(row['frame']) is int,
                    'recorded draw provenance is missing')
            seed = independent_agent_seed(base_seed=row['action_seed'], episode_id=row['episode_id'],
                                          frame=row['frame'], agent_id=self.role)
            expected = int(self.stored_actions[start+local])
            for name, value in versions.items():
                actual = int(np.random.default_rng(seed).choice(5, p=value[local]))
                if actual != expected:
                    raise NumericalMismatch('recorded counterfactual sampled action differs', dict(
                        kind='recorded_sampling', variant=name, row_offset=start+local, row=deepcopy(row),
                        stored_action=expected, reconstructed_action=actual,
                        probability_float_hex=[float(v).hex() for v in value[local]],
                        probability_signbits=np.signbit(value[local]).tolist()))
                reconstructed[name].append(actual)
        for name, actions in reconstructed.items():
            self.actions[name].update(np.asarray(actions, np.int64).tobytes())
        self.seen += n
        self.seconds += perf_counter()-t

    def result(self):
        require(self.seen == len(self.rows), 'not all recorded counterfactual draws replayed')
        hexes = {name: h.hexdigest() for name, h in self.hashes.items()}
        actions = {name: h.hexdigest() for name, h in self.actions.items()}
        require(len(set(hexes.values())) == len(set(actions.values())) == 1, 'probability/draw reconstruction differs')
        return dict(rows=self.seen, seconds=self.seconds, probability_float_hex_sha256=hexes,
            sampled_action_bytes_sha256=actions, sign_zero_counts=self.zero_counts,
            all_probability_bits_hex_signzero_and_recorded_draws_equal=True,
            old_recorded_streams_replayed_only=True, new_action_seed_allocation=0,
            actual_environment_actions_executed=0)


def _counterexample(role, x, p, rows, folder):
    """Save the first failing original/candidate layer with exact float hex."""
    for start in range(0, len(x), REFERENCE_CHUNK):
        batch = x[start:start+REFERENCE_CHUNK]
        traces = [numerical.forward_trace(role, row[None], 'original_shape') for row in batch]
        reference = {key: np.concatenate([a[key] for a in traces], axis=0) for key in traces[0]}
        original_logits = np.stack([role.logits_encoded(row) for row in batch])
        original_p = numerical.canonical_probabilities(role, batch)
        candidates = {}
        for chunk in CHUNKS:
            traces = [numerical.forward_trace(role, batch[offset:offset+chunk], METHOD)
                for offset in range(0, len(batch), chunk)]
            candidates[chunk] = {key: np.concatenate([a[key] for a in traces], axis=0) for key in traces[0]}
        comparisons = [('trace_original_logits', original_logits, reference['logits']),
            ('trace_original_probability', original_p, reference['probabilities']),
            ('stored_probability', original_p, p[start:start+len(batch)])]
        comparisons.extend((f'chunk-{chunk}/{key}', reference[key], value[key])
            for chunk, value in candidates.items() for key in reference)
        for name, expected, found in comparisons:
            if numerical.bits_equal(expected, found):
                continue
            a = np.ascontiguousarray(expected).view(np.uint8).reshape(*expected.shape, expected.dtype.itemsize)
            b = np.ascontiguousarray(found).view(np.uint8).reshape(*found.shape, found.dtype.itemsize)
            index = tuple(map(int, np.argwhere(np.any(a != b, axis=-1))[0]))
            artifact = folder/'counterexample.npz'
            with artifact.open('xb') as handle:
                np.savez(handle, x=batch, stored_p=p[start:start+len(batch)], expected=expected, candidate=found)
            artifact.chmod(0o444)
            value = dict(layer=name, element_index=list(index), physical_row_offset=start+index[0],
                row=rows[start+index[0]], expected_float_hex=float(expected[index]).hex(),
                candidate_float_hex=float(found[index]).hex(), expected_signbit=bool(np.signbit(expected[index])),
                candidate_signbit=bool(np.signbit(found[index])), arrays=pointer(artifact), numerical_candidate_rejected=True)
            write_new(folder/'counterexample.json', value)
            return pointer(folder/'counterexample.json')
    return None


def _write_immutable(path, value):
    write_new(path, value)
    path.chmod(0o444)
    return pointer(path)


def _verify_sources(sources):
    require(source_files() == sources, 'diagnostic source closure changed')


def run(calls, *, output, resume=False):
    """Persist each completed finite segment; resume never reruns saved work."""
    start = perf_counter()
    sources = source_files()
    planned = inventory(calls)
    folder = Path(output)
    descriptor = dict(version=VERSION, inventory=planned, source_files=sources, source_sha256=digest(sources),
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        threads={name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        priority=os.getpriority(os.PRIO_PROCESS, 0), source_git_revision=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip())
    require(all(descriptor['threads'][name] == '1' for name in descriptor['threads'])
            and descriptor['priority'] >= 10, 'single-thread low-priority diagnostic required')
    if folder.exists():
        require(resume is True and not (folder/'result.json').exists() and not (folder/'failed.json').exists(),
                'old numerical run cannot be overwritten/retried after completion/failure')
        original_descriptor = bound(pointer(folder/'started.json'))
        descriptor['source_git_revision'] = original_descriptor['source_git_revision']
        require(original_descriptor == descriptor, 'resumed old diagnostic inputs or sources differ')
    else:
        require(resume is False, 'no started diagnostic to resume')
        folder.mkdir(parents=True)
        _write_immutable(folder/'started.json', descriptor)
        root = Path(__file__).resolve().parents[2]
        for name, expected in sources.items():
            path = folder/'source-snapshot'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((root/name).read_bytes())
            require(file_sha(path) == expected, 'source snapshot differs')
            path.chmod(0o444)
    completions = []
    actors = {}
    total = 0
    for part in planned['partitions']:
        actor_key = part['actor']['sha256']
        if actor_key not in actors:
            actors[actor_key] = numerical.runtime.FrozenWarehouseV6Actor.load(part['actor']['path'], expected_sha256=actor_key,
                expected_role_sha256=part['actor_role_sha256'])
        role = actors[actor_key].roles[part['role']]
        require(pointer(part['arrays']['path']) == part['arrays'], 'old input arrays changed before part')
        with np.load(part['arrays']['path'], allow_pickle=False) as archive:
            require(set(archive.files) == {'x', 'm', 'p', 'sampled'}, 'old native target fields differ')
            a = {key: archive[key].copy() for key in archive.files}
        rows = bound(part['rows_ref'])
        require(len(rows) == part['rows'] == len(a['x'])
            and numerical.bits_equal(a['m'], a['x'][:, numerical.MASK_INDICES]), 'old partition rows/physical masks differ')
        for lo in range(0, len(rows), SEGMENT_ROWS):
            hi = min(len(rows), lo+SEGMENT_ROWS)
            segment_folder = folder/'segments'/f"call-{part['call']:06d}"/part['role']/part['split']/f'{lo:06d}-{hi:06d}'
            result_path = segment_folder/'result.json'
            segment_binding = dict(part=part, start=lo, end=hi, source_sha256=descriptor['source_sha256'])
            if result_path.exists():
                value = bound(pointer(result_path))
                require(value['binding'] == segment_binding and value['all_layers_bitwise_equal'] is True
                    and value['probability_and_sampling']['rows'] == hi-lo, 'saved engineering segment differs')
                require(pointer(value['scalar_outputs']['path']) == value['scalar_outputs'], 'retained scalar outputs changed')
                completions.append(pointer(result_path)); total += hi-lo
                continue
            if segment_folder.exists():
                require(resume is True and {'started.json'} <= {p.name for p in segment_folder.iterdir()}
                    <= {'started.json', 'scalar-outputs.npz'}
                    and bound(pointer(segment_folder/'started.json')) == segment_binding,
                    'unfinished old diagnostic segment has unknown or conflicting files')
            else:
                segment_folder.mkdir(parents=True, exist_ok=False)
                _write_immutable(segment_folder/'started.json', segment_binding)
            observer = ProbabilityObserver(part['role'], rows[lo:hi], a['p'][lo:hi], a['sampled'][lo:hi])
            scalar = ScalarOutputs(hi-lo, a['sampled'][lo:hi])
            tick = perf_counter()
            try:
                layers = numerical.verify_all_layers(role, a['x'][lo:hi], stored=a['p'][lo:hi],
                    chunk_size=REFERENCE_CHUNK, methods=(METHOD,), candidate_chunk_sizes=CHUNKS,
                    probability_observer=observer, scalar_output_observer=scalar)
                observations = observer.result()
                equal = all(value['all_layers_bitwise_equal'] for value in layers['candidates'].values())
                require(equal, 'an intermediate layer differs; numerical candidate rejected')
                value = dict(version=VERSION, binding=segment_binding, layers=layers, probability_and_sampling=observations,
                    all_layers_bitwise_equal=equal, wall_seconds=perf_counter()-tick,
                    scalar_outputs=scalar.save(segment_folder),
                    actual_old_role_frames_recomputed=hi-lo, new_scientific_samples=0,
                    execution_actions_changed=False, future_NN_equivalence_claimed=False)
                ref = _write_immutable(result_path, value)
                completions.append(ref); total += hi-lo
                with (folder/'progress.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(completed_frames=total, expected_frames=planned['total_old_role_frames'],
                        segment=ref, wall_seconds=value['wall_seconds']), sort_keys=True)+'\n')
                    handle.flush(); os.fsync(handle.fileno())
                print(f"old-call={part['call']} role={part['role']} split={part['split']} rows={lo}:{hi} "
                      f"total={total}/{planned['total_old_role_frames']} seconds={value['wall_seconds']:.3f}", flush=True)
            except Exception as exc:
                if isinstance(exc, NumericalMismatch):
                    counterexample = _write_immutable(segment_folder/'sampling-counterexample.json', exc.evidence)
                else:
                    try:
                        counterexample = _counterexample(role, a['x'][lo:hi], a['p'][lo:hi], rows[lo:hi], segment_folder)
                    except Exception as diagnostic_error:
                        artifact = segment_folder/'failure-inputs.npz'
                        with artifact.open('xb') as handle:
                            np.savez(handle, x=a['x'][lo:hi], m=a['m'][lo:hi], p=a['p'][lo:hi], sampled=a['sampled'][lo:hi])
                        artifact.chmod(0o444)
                        counterexample = _write_immutable(segment_folder/'failure-inputs.json', dict(arrays=pointer(artifact),
                            secondary_diagnostic_exception=type(diagnostic_error).__name__, numerical_candidate_rejected=True))
                failed = dict(version=VERSION, binding=segment_binding, exception_type=type(exc).__name__,
                    reason=str(exc), counterexample=counterexample, completed_old_role_frames=total,
                    full_pool_numerical_verification=False, numerical_candidate_rejected=True,
                    future_training_integration_authorized=False, new_scientific_samples=0)
                _write_immutable(folder/'failed.json', failed)
                raise
        require(pointer(part['arrays']['path']) == part['arrays'] and pointer(part['rows_ref']['path']) == part['rows_ref'],
                'old arrays or row provenance changed during part')
    _verify_sources(sources)
    for ref in planned['input_refs']:
        require(pointer(ref['path']) == ref, 'old input changed during complete numerical replay')
    require(total == planned['total_old_role_frames'], 'complete old physical frame coverage is missing')
    result = dict(version=VERSION, descriptor=pointer(folder/'started.json'), source_sha256=descriptor['source_sha256'],
        full_pool_numerical_verification=True, old_role_frames_recomputed=total,
        expected_old_role_frames=planned['total_old_role_frames'], partition_count=planned['partition_count'],
        segments=completions, segment_count=len(completions), wall_seconds_this_process=perf_counter()-start,
        all_scalar_logits_probabilities_and_recorded_samples_retained=True,
        candidate_chunks=list(CHUNKS), all_layers_logits_probability_hex_signzero_and_recorded_samples_equal=True,
        new_scientific_samples=0, new_scene_or_action_stream_allocation=0, environment_resets=0, fitting_calls=0,
        NN_updates=0, actual_environment_actions_executed=0, model_effect_improvement_claimed=False,
        independent_audit_or_holdout_claimed=False, future_NN_equivalence_claimed=False,
        training_pipeline_modified=False, future_training_integration_authorized=False, pilot_opened=False)
    ref = _write_immutable(folder/'result.json', result)
    _write_immutable(folder/'completed.json', dict(result=ref, old_role_frames_recomputed=total))
    return ref
