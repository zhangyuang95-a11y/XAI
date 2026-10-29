"""Clean same-claim recovery, exclusively on explicit synthetic physics states."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
import torch

from experiments.warehouse import v643_calibration_pipeline as c
from experiments.warehouse import v643_calibration_recovery as r


def helper():
    spec=importlib.util.spec_from_file_location('v643_recovery_pipeline_fixture',
        Path(__file__).with_name('test_warehouse_v643_calibration_pipeline.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def equal(left,right):
    if isinstance(left,torch.Tensor):assert isinstance(right,torch.Tensor) and torch.equal(left,right)
    elif isinstance(left,np.ndarray):assert isinstance(right,np.ndarray) and np.array_equal(left,right)
    elif type(left) is dict:
        assert type(right) is dict and set(left)==set(right)
        for key in left:equal(left[key],right[key])
    elif type(left) in (tuple,list):
        assert type(left) is type(right) and len(left)==len(right)
        for a,b in zip(left,right):equal(a,b)
    else:assert type(left) is type(right) and left==right


def short_fixture_batch(config):
    # This INSTANCE-only engineering prefix keeps all eight restored episodes
    # alive, so the test can forbid every constructor/reset. The registered
    # class/default recipe and finite seed plan are not changed.
    normal=config.next_collection_length
    object.__setattr__(config,'next_collection_length',lambda start:2 if start==1024 else normal(start))


def shadow_next_batch(original,folder):
    """Uninterrupted numerical/physics control, outside scientific outputs."""
    variant='raw';state=original.states[variant];core=original.cores[variant]
    short_fixture_batch(original.config)
    folder.mkdir();before=c.training_fingerprint(state)
    actor_ref,actor=c.frozen_export(state,folder/'actor.npz',variant)
    identity=dict(actor_role_sha256=c.role_hashes(state.model),
        critic_sha256=c.numerical.previous.critic_initialization_sha256(state.model),
        actor_updates=state.actor_updates,critic_updates=state.critic_updates)
    events=[];ends=[None]*8
    for index in range(16):
        slot=index%8;env=original.environments[variant][slot];case=original.active_cases[variant][slot]
        assert env is not None and not(env.state.terminated or env.state.truncated)
        row=c.FiniteCalibrationOwner._step(env,actor,case)
        events.append(dict(slot=slot,case=case,**{k:v for k,v in row.items() if k!='_case'}))
        state.advance_joint_steps(1)
        if index>=8:ends[slot]=c._write(folder/f'end-{slot}.json',c.physics.capture_environment(env))
    trace=dict(schema=c.TRACE_VERSION,partition='train',owner=original.owners[variant],config=c.asdict(original.config),
        start_joint_step=1024,end_joint_step=1040,actor=actor_ref,rollout_state=identity,
        physics=dict(schema=c.physics.PHYSICS_VERSION,resets={},end_snapshots=ends),events=events)
    trace_ref=c._write(folder/'trace.json',trace)
    batch,_=original.readers[variant].assemble(trace_ref,state)
    core.update_batch(batch,batch_sha256=trace_ref['sha256'])
    assert c.training_fingerprint(state)!=before
    return trace,core


def test_clean_pair_resume_restores_next_physical_batch_and_optimizer_bitwise(tmp_path,monkeypatch):
    prior_threads=torch.get_num_threads();torch.set_num_threads(1)
    h=helper();plan,fixture=h.engineering_plan.__wrapped__(tmp_path,monkeypatch)
    try:
        with pytest.raises(h.ProbeComplete):
            with c._claimed_owner(plan) as original:
                for v in ('raw','regularized'):original.initialize(v)
                h.controlled_physics(original,fixture,monkeypatch)
                for v in ('raw','regularized'):
                    original.collect_rollout(v);original.update(v);original.checkpoint(v)
                checkpoint=original.pair_checkpoint()
                saved_fingerprints={v:c.training_fingerprint(original.states[v]) for v in ('raw','regularized')}
                raise h.ProbeComplete()
        claim_before=c.pointer(c.identity.claim_path())
        pair,_,_,_=r.validate_clean_pair(plan,checkpoint)
        journal=Path(pair['execution_journal']['path']);pristine=journal.read_bytes()
        # Each rejected counterfactual is a temporary fixture edit, with no
        # reconstruction or new reset. Production recovery never erases one.
        for kind in ('rollout_predebit','optimizer_update_predebit','initialization_predebit','resume_attempt_predebit'):
            rows=r._execution_rows(pristine)
            extra=dict(version=c.VERSION,candidate_id=c.CANDIDATE,sequence=len(rows),previous_sha256=c.digest(rows[-1]),
                       claim=claim_before,kind=kind)
            journal.write_bytes(pristine+(json.dumps(extra,sort_keys=True,separators=(',',':'))+'\n').encode())
            with pytest.raises(RuntimeError,match='post-boundary'):
                r.validate_clean_pair(plan,checkpoint)
            journal.write_bytes(pristine)
        closed=original.folder/'calibration-closed.json';closed.write_text('{"synthetic_closed":true}')
        with pytest.raises(RuntimeError,match='scientifically failed'):
            r.validate_clean_pair(plan,checkpoint)
        closed.unlink()
        expected_trace,expected_core=shadow_next_batch(original,tmp_path/'uninterrupted-shadow')
        torch_before=torch.get_rng_state().clone();numpy_before=deepcopy(np.random.get_state())
        def forbidden(*_a,**_k):raise AssertionError('restore must not initialize or reset a new scientific stream/environment')
        with pytest.raises(h.ProbeComplete):
            # No initialize_from_reference/default_rng/constructor/reset can
            # run during reconstruction. PCG64 allocation uses only the
            # original registered argument then overwrites its saved state.
            with monkeypatch.context() as guard:
                guard.setattr(c,'initialize_from_reference',forbidden)
                guard.setattr(np.random,'default_rng',forbidden)
                guard.setattr(c.physics.WarehouseMultiAgentEnv,'__init__',forbidden)
                guard.setattr(c.physics.WarehouseMultiAgentEnv,'reset',forbidden)
                with r._resume_clean_pair(plan,checkpoint) as resumed:
                    assert torch.equal(torch_before,torch.get_rng_state())
                    equal(numpy_before,np.random.get_state())
                    for v in ('raw','regularized'):
                        assert c.training_fingerprint(resumed.states[v])==saved_fingerprints[v]
                        assert resumed.cores[v].risk_training.rng is resumed.cores[v].rng
                    # Undo ONLY this nested guard before normal hash-derived
                    # NN action sampling; constructors/resets remain forbidden
                    # by the outer fixture and explicit method guard below.
                    guard.undo()
                    for config in (resumed.config,resumed.cores['raw'].config,resumed.readers['raw'].config,
                                   resumed.readers['raw'].public.config):
                        if 'next_collection_length' not in vars(config):short_fixture_batch(config)
                    resumed.new_environment=forbidden
                    actual_ref=resumed.collect_rollout('raw');actual_trace=c.bound(actual_ref)
                    assert actual_trace['events']==expected_trace['events'] and actual_trace['physics']['resets']=={}
                    assert len(resumed.actual_resets)==len(pair['reset_records'])
                    resumed.update('raw');actual_core=resumed.cores['raw']
                    assert c.training_fingerprint(actual_core.state)==c.training_fingerprint(expected_core.state)
                    equal(actual_core.risk_training.optimizer.state_dict(),expected_core.risk_training.optimizer.state_dict())
                    equal(actual_core.rng.bit_generator.state,expected_core.rng.bit_generator.state)
                    a=actual_core.risk_training.snapshot();b=expected_core.risk_training.snapshot()
                    # Distinct shadow/owned trace paths give different receipt
                    # IDs, not different numerical or episode/RNG state.
                    a.pop('seen_batches');b.pop('seen_batches');equal(a,b)
                    assert actual_core.accepted_program_updates==dict(robot_1=0,robot_2=0)
                    raise h.ProbeComplete()
        assert c.pointer(c.identity.claim_path())==claim_before
        with pytest.raises(RuntimeError,match='post-boundary|owned output changed'):
            with r._resume_clean_pair(plan,checkpoint):raise AssertionError('same checkpoint was rearmed')
        rows=r._execution_rows(journal.read_bytes())
        assert sum(row['kind']=='resume_attempt_predebit' for row in rows)==1
        allocations=[row for row in rows if row['kind']=='restore_optimizer_rng_allocation_predebit']
        assert len(allocations)==2 and all(row['seed']==plan.partition.value['optimizer_seed']
            and row['allocation_only'] and row['draws_before_state_overwrite']==0 for row in allocations)
    finally:torch.set_num_threads(prior_threads)
