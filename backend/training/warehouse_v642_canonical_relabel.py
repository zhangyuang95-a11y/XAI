"""New target producer; the native complete scalar audit remains mandatory.

Only the duplicate *production* forward is batched.  This module cannot mint a
replayed-target certificate, authorize training, or replace checked_targets.
The original gate M=1 and residual M=5 matrix shapes are retained, as are the
per-frame nonlinearities/softmax.  A fixed scalar witness is an extra guard;
it is not evidence that untested weights/platforms are generally equivalent.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import os, platform
import numpy as np

from . import warehouse_v642_observed_pool as native
from . import warehouse_v642_canonical_batch as batch
from experiments.warehouse.v626_protocol import require, pointer, bound, digest, file_sha
from scripts.run_warehouse_v625 import write_new

VERSION = 'warehouse-v642-canonical-target-producer.v1'
METHOD = 'frame_matrices_scalar_ufunc'
CHUNK = 256
WITNESSES = 64
ROOT = Path(__file__).resolve().parents[2]
FILES = ('backend/training/warehouse_v642_canonical_relabel.py',
         'tests/test_warehouse_v642_canonical_relabel.py',
         'docs/warehouse_v642_canonical_relabel.md')


def source_files():
    sources = batch.source_files()
    sources.update({name: file_sha(ROOT/name) for name in FILES})
    return dict(sorted(sources.items()))


def _immutable(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'explicit immutable Actor pointer required')
    path = Path(ref['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file()
        and all(not p.is_symlink() for p in (path, *path.parents))
        and not path.stat().st_mode & 0o222 and pointer(path) == ref,
        'canonical producer Actor/provenance is missing, writable, symlinked or changed')


def _guard(source_guard, observed, sources):
    require(callable(source_guard), 'explicit active owner/source guard required')
    value = source_guard()
    require(type(value) is dict and value.get('owner') == observed.owner_ref
        and type(value.get('source_files')) is dict
        and all(value['source_files'].get(name) == sha for name, sha in sources.items()),
        'canonical producer omitted its owner or complete source closure')
    digest(value)  # Reject non-JSON capabilities or implicit callable state.
    return deepcopy(value)


def witness_indices(n):
    require(type(n) is int and n > 0, 'nonempty complete physical partition required')
    # Integer arithmetic, no RNG or new stream. Endpoints and evenly spaced
    # old frames; schedule is independent of NN outputs and program results.
    return sorted({(i*(n-1))//(min(n, WITNESSES)-1) if min(n, WITNESSES)>1 else 0
                   for i in range(min(n, WITNESSES))})


def _float_contract(a):
    a = np.asarray(a)
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=batch.array_sha(a),
        float_hex=[float(v).hex() for v in a.flat],
        negative_zero=int(np.count_nonzero((a == 0) & np.signbit(a))))


def _part_inputs(observed):
    return {split: dict(rows_sha256=digest(part['rows']),
        **{name: batch.array_sha(part[name]) for name in ('x','m','behavior_p','executed')})
        for split, part in observed.parts.items()}


def produce_parts(observed, actor_ref, *, epoch_id, sequence, source_guard):
    """Return pure numeric parts and provenance, with no target certificate.

    The caller must subsequently materialize native targets and run the
    unchanged native checked_targets across every frame before using them.
    """
    require(isinstance(epoch_id, str) and 0 < len(epoch_id) <= 128
        and type(sequence) is int and sequence >= 0, 'explicit epoch and finite call sequence required')
    require(isinstance(observed, native.ObservedPool), 'audited observed states required')
    sources = source_files(); guard = _guard(source_guard, observed, sources)
    observed.verify(); _immutable(actor_ref)
    actor = native.FrozenWarehouseV6Actor.load(actor_ref['path'], expected_sha256=actor_ref['sha256'])
    require(actor.model_version == native.MODEL_VERSION, 'canonical producer NN architecture changed')
    started = perf_counter(); parts = {}; checks = {}; times = {}; inputs = _part_inputs(observed)
    for i, role in enumerate(native.AGENTS):
        parts[role] = {}; checks[role] = {}; times[role] = {}
        for split, part in observed.parts.items():
            x, m = part['x'][:,i], part['m'][:,i]
            require(batch.bits_equal(x[:,batch.MASK_INDICES], m), 'canonical producer physical mask differs')
            zparts = []; pparts = []; before = perf_counter()
            for start in range(0, len(x), CHUNK):
                trace = batch.forward_trace(actor.roles[role], x[start:start+CHUNK], METHOD)
                zparts.append(trace['logits']); pparts.append(trace['probabilities'])
            logits = np.concatenate(zparts); probabilities = np.concatenate(pparts)
            production_seconds = perf_counter()-before
            selected = witness_indices(len(x)); before = perf_counter()
            scalar_logits = np.stack([actor.roles[role].logits_encoded(x[j]) for j in selected])
            scalar_p = np.stack([batch.masked_probabilities(z, m[j])
                for z,j in zip(scalar_logits, selected, strict=True)])
            require(batch.bits_equal(logits[selected], scalar_logits)
                and batch.bits_equal(probabilities[selected], scalar_p),
                'canonical producer scalar witness differs bitwise; targets rejected')
            witness_seconds = perf_counter()-before
            checks[role][split] = dict(indices=selected, schedule='deterministic-inclusive-integer-spacing.v1',
                logits=_float_contract(scalar_logits), probabilities=_float_contract(scalar_p),
                all_witness_values_bitwise_equal=True, full_scalar_audit=False)
            before = perf_counter()
            sampled = np.array([np.random.default_rng(native.independent_agent_seed(
                base_seed=row['action_seed'], episode_id=row['episode_id'], frame=row['frame'], agent_id=role)).choice(5,p=p)
                for row,p in zip(part['rows'], probabilities, strict=True)], np.int64)
            times[role][split] = dict(producer_seconds=production_seconds, scalar_witness_seconds=witness_seconds,
                recorded_draw_reconstruction_seconds=perf_counter()-before, physical_frames=len(x))
            parts[role][split] = dict(x=x, m=m, p=probabilities, sampled=sampled, rows=part['rows'])
    observed.verify(); _immutable(actor_ref)
    require(_part_inputs(observed) == inputs and _guard(source_guard, observed, sources) == guard,
        'canonical producer source/owner/input changed during production')
    provenance = dict(version=VERSION, source_files=sources, source_sha256=digest(sources), source_guard=guard,
        observed_source=observed.source_ref, observed_audit_sha256=observed.audit_sha256,
        observed_data_sha256=observed.observed_data_sha256, owner=observed.owner_ref,
        target_actor=actor_ref, actor_role_sha256=actor.role_sha256, epoch_id=epoch_id, sequence=sequence,
        inputs=inputs, method=METHOD, chunk_size=CHUNK, gate_inner_M=1, residual_inner_M=5,
        scalar_ufunc_per_physical_frame=True, witnesses=checks, timing=times, total_seconds=perf_counter()-started,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
            threads={key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}),
        full_native_scalar_audit_required_before_feedback=True, native_scalar_audit_performed=False,
        replayed_target_certificate_minted=False, independent_validation=False, fresh_samples=0,
        environment_resets=0, NN_updates=0, PPO_joint_steps=0, training_authority=False, pilot_opened=False)
    return parts, provenance


def _put(path, value):
    write_new(path, value); path.chmod(0o444); return pointer(path)


def relabel_current_NN(observed, target_actor_ref, output, *, epoch_id, sequence, source_guard):
    """Write a NEW native-compatible package with a separate producer appendix."""
    # No output is created until active source guard and numerical guards pass.
    parts, provenance = produce_parts(observed, target_actor_ref, epoch_id=epoch_id,
        sequence=sequence, source_guard=source_guard)
    output = Path(output).resolve()
    require(output.is_relative_to(native.TARGET_ROOT.resolve()) and output != native.TARGET_ROOT.resolve(),
        'canonical targets need a new isolated native target directory')
    output.mkdir(parents=True, exist_ok=False)
    try:
        actual = {}; refs = []; pools = {}; differences = {}
        for split, part in observed.parts.items():
            path = output/(split+'-source-actual-actions.npz')
            with path.open('xb') as handle: np.savez_compressed(handle, executed=part['executed'])
            path.chmod(0o444); actual[split] = pointer(path); refs.append(actual[split])
        for i,role in enumerate(native.AGENTS):
            sources = {}; differences[role] = {}
            for split, part in parts[role].items():
                path = output/(role+'-'+split+'.npz')
                with path.open('xb') as handle: np.savez_compressed(handle, **{key:part[key] for key in ('x','m','p','sampled')})
                path.chmod(0o444); rows = _put(output/(role+'-'+split+'-rows.json'), part['rows'])
                sources[split] = dict(arrays=pointer(path), rows=rows); refs.extend(sources[split].values())
                differences[role][split] = int(np.count_nonzero(part['sampled'] != observed.parts[split]['executed'][:,i]))
            pools[role] = _put(output/(role+'-pool.json'), dict(schema=native.online.MANIFEST_VERSION,
                scope='online_program_training_evidence', target_actor=target_actor_ref, role=role, parts=sources))
            refs.append(pools[role])
        provenance['outputs'] = {role:{split:{key:batch.array_sha(p[key]) for key in ('x','m','p','sampled')}
            for split,p in values.items()} for role,values in parts.items()}
        appendix = _put(output/'canonical-producer.json', provenance)
        observed.verify(); _immutable(target_actor_ref)
        require(_guard(source_guard, observed, provenance['source_files']) == provenance['source_guard'],
            'canonical owner/source changed before target publication')
        result = dict(schema=native.TARGET_VERSION, observed_source=observed.source_ref,
            observed_audit_sha256=observed.audit_sha256, observed_data_sha256=observed.observed_data_sha256,
            owner=observed.owner_ref, target_actor=target_actor_ref, source_files=native.source_files(),
            pools=pools, actual_source_actions=actual, counterfactual_difference_counts=differences, asset_refs=refs,
            current_policy_trajectory=False, counterfactual_draws_not_executed=True, actual_actions_replaced=False,
            actual_PPO_feedback_counts_from_real_rollouts_only=True, current_policy_task_quality_estimate=None,
            uses_existing_recorded_seeds_for_counterfactual_numerical_replay=True, fresh_randomness_claimed=False,
            environment_resets=0, NN_updates=0, PPO_joint_steps=0, training_authority=False, pilot_opened=False,
            canonical_producer_provenance=appendix)
        ref = _put(output/'result.json', result)
        _put(output/'completed.json', dict(result_sha256=ref['sha256']))
        read_provenance(observed, ref, epoch_id=epoch_id, sequence=sequence, source_guard=source_guard)
        return ref
    except BaseException as error:
        _put(output/'failed.json', dict(error=str(error), automatic_retry=False, training_authority=False)); raise


def read_provenance(observed, result_ref, *, epoch_id, sequence, source_guard):
    """Verify producer binding only; this never substitutes for checked_targets."""
    observed.verify(); _immutable(result_ref); result = bound(result_ref)
    require('canonical_producer_provenance' in result, 'explicit canonical producer appendix required')
    ref = result['canonical_producer_provenance']; _immutable(ref); p = bound(ref); sources = source_files()
    require(p['version'] == VERSION and p['source_files'] == sources and p['source_sha256'] == digest(sources)
        and p['source_guard'] == _guard(source_guard, observed, sources) and p['owner'] == observed.owner_ref
        and p['target_actor'] == result['target_actor'] and p['epoch_id'] == epoch_id and p['sequence'] == sequence
        and p['observed_source'] == observed.source_ref and p['observed_audit_sha256'] == observed.audit_sha256
        and p['observed_data_sha256'] == observed.observed_data_sha256 and p['inputs'] == _part_inputs(observed)
        and p['method'] == METHOD and p['chunk_size'] == CHUNK and p['gate_inner_M'] == 1 and p['residual_inner_M'] == 5
        and p['scalar_ufunc_per_physical_frame'] is True
        and p['full_native_scalar_audit_required_before_feedback'] is True and p['native_scalar_audit_performed'] is False
        and p['replayed_target_certificate_minted'] is p['independent_validation'] is p['training_authority'] is p['pilot_opened'] is False
        and all(type(p[k]) is int and p[k] == 0 for k in ('fresh_samples','environment_resets','NN_updates','PPO_joint_steps')),
        'canonical producer epoch, owner, source, inputs or scope differs')
    _immutable(p['target_actor'])
    actor = native.FrozenWarehouseV6Actor.load(p['target_actor']['path'], expected_sha256=p['target_actor']['sha256'])
    require(actor.role_sha256 == p['actor_role_sha256'], 'canonical producer dual-role hash differs')
    for role in native.AGENTS:
        manifest = bound(result['pools'][role])
        for split in native.online.SPLITS:
            witness = p['witnesses'][role][split]
            require(witness['indices'] == witness_indices(len(observed.parts[split]['x']))
                and witness['all_witness_values_bitwise_equal'] is True and witness['full_scalar_audit'] is False,
                'canonical producer fixed scalar witness schedule/scope differs')
            ref_arrays = manifest['parts'][split]['arrays']; _immutable(ref_arrays)
            with np.load(ref_arrays['path'], allow_pickle=False) as arrays:
                require({key:batch.array_sha(arrays[key]) for key in ('x','m','p','sampled')} == p['outputs'][role][split],
                    'canonical produced numeric arrays differ from appendix')
                selected = witness['indices']
                require(_float_contract(arrays['p'][selected]) == witness['probabilities'],
                    'canonical producer stored scalar probability witness differs')
    return dict(provenance=p, provenance_ref=ref, native_scalar_audit_still_required=True)
