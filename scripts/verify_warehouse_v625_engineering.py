#!/usr/bin/env python3
"""Engineering-only checkpoint preflight; never fits or resets an environment.

The optional large replay reads one explicitly pinned, trusted local legacy
pickle. It cannot resume that scientific job. It maps its real queue, bounded
state cache and leaf statistics into the new JSON/array CAS, then replays the
same routing/distribution-statistic operations with and without checkpoints.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from backend.training.warehouse_v625_checkpoint import CheckpointSession,CheckpointError,DeadlineExceeded,digest,plain

FIXTURE=ROOT/'output/warehouse_v624/warehouse-v624-s1-upgrade-01/stage-A/jobs/fold-1-robot_2-P1/checkpoint.json.payload'
FIXTURE_SHA='f19bf327dbdd89ab1082a9760688fc0b3194b30d21d156236b1cd8d53e644eff'
ZERO={'environment_resets':0,'actor_updates':0,'PPO_joint_steps':0,'pilot_opened':False,
      'publishable':False,'sealed_audit_used':False,'heldout_used':False}


def file_sha(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def advance(self,amount):self.value+=amount


def must_reject(fn, error=(CheckpointError,DeadlineExceeded,OSError,RuntimeError)):
    try:fn()
    except error:return True
    raise AssertionError('unsafe checkpoint operation was accepted')


def fault_suite(directory):
    """Deterministic fault times, actual crash, corruption and resume checks."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    results={};identity={'engineering_only':True,'sequence':'fixed-queue-rng-v1'}
    path=directory/'equivalence.json';s=CheckpointSession(path,identity)
    rng=np.random.default_rng(125);queue=[int(v) for v in rng.integers(0,1000,24)]
    ref=s.store.put_array(np.arange(128,dtype=np.float32));cache=s.store.put_json({'array':ref,'fifo':[1,2,3]})
    token=s.consume();s.finish(token,success=True,detail={'engineering_only':True})
    state={'queue':queue,'cursor':7,'cache':cache,'rng':rng.bit_generator.state}
    s.commit(state,status='paused');r=CheckpointSession(path,identity,resume=True)
    restored=r.state;actual_rng=np.random.default_rng();actual_rng.bit_generator.state=restored['rng']
    assert restored['queue'][restored['cursor']:] == queue[7:]
    assert np.array_equal(actual_rng.integers(0,2**31,32),rng.integers(0,2**31,32))
    assert np.array_equal(r.store.get_array(r.store.get_json(restored['cache'])['array']),np.arange(128,dtype=np.float32))
    assert r.calls==1;r.commit(restored,status='completed')
    results['safe_pause_queue_cache_rng_and_call_equivalence']=True
    results['completed_job_cannot_resume']=must_reject(lambda:CheckpointSession(path,identity,resume=True))
    path=directory/'pending.json';s=CheckpointSession(path,identity);s.consume();s.close()
    results['inflight_call_no_refund_or_crash_resume']=must_reject(lambda:CheckpointSession(path,identity,resume=True))
    ledger=[json.loads(x) for x in Path(str(path)+'.ledger.jsonl').read_bytes().splitlines()]
    assert sum(r['event']=='call_started' for r in ledger)==1
    for hook in ('before_object_write','after_object_write','before_fsync','after_fsync',
                 'before_manifest_replace','after_manifest_replace','before_dirty_remove'):
        path=directory/('fault-'+hook+'.json');s=CheckpointSession(path,identity)
        armed={'value':True}
        def fail(event,context):
            if event==hook and armed['value']:
                armed['value']=False;raise OSError('injected '+hook)
        s.fault_hook=fail;s.store.fault_hook=fail
        must_reject(lambda:s.commit({'queue':[1,2,3]},status='paused'));s.close()
        results['fault_'+hook+'_fails_closed']=must_reject(lambda:CheckpointSession(path,identity,resume=True))
    for hook in ('before_object_write','before_fsync','before_manifest_replace','before_dirty_remove'):
        path=directory/('slow-'+hook+'.json');clock=Clock();s=CheckpointSession(path,identity,clock=clock,wall_seconds=5)
        def delay(event,context):
            if event==hook:clock.advance(6.)
        s.fault_hook=delay;s.store.fault_hook=delay
        results['slow_'+hook+'_deadline']=must_reject(lambda:s.commit({'actual_slow_io_simulated_clock':True},status='completed'))
        s.close();assert Path(str(path)+'.dirty').exists()
    for damage in ('missing_object','corrupt_object','ledger_tail','identity'):
        path=directory/('corrupt-'+damage+'.json');s=CheckpointSession(path,identity)
        ref=s.store.put_json({'real_payload':[1,2,3]});s.commit({'ref':ref},status='paused')
        obj=s.store.physical_path(ref)
        if damage=='missing_object':obj.unlink()
        elif damage=='corrupt_object':obj.write_bytes(b'{}')
        elif damage=='ledger_tail':
            with Path(str(path)+'.ledger.jsonl').open('ab') as f:f.write(b'{}\n')
        changed=dict(identity,changed=True) if damage=='identity' else identity
        results[damage+'_rejected']=must_reject(lambda:CheckpointSession(path,changed,resume=True))
    path=directory/'process-crash.json'
    code="from backend.training.warehouse_v625_checkpoint import CheckpointSession; import os; s=CheckpointSession("+repr(str(path))+", "+repr(identity)+"); s.consume(); os._exit(73)"
    process=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,check=False)
    assert process.returncode==73
    results['actual_process_crash_rejected']=must_reject(lambda:CheckpointSession(path,identity,resume=True))
    # A real POSIX timer interrupts CPU work; there is no sleep-based benchmark.
    path=directory/'hard-deadline.json';s=CheckpointSession(path,identity,wall_seconds=.03)
    def cpu_deadline():
        with s.hard_deadline():
            while True:sha256(b'bounded deadline exercise').digest()
    results['real_POSIX_hard_deadline']=must_reject(cpu_deadline);s.close()
    return {'passed':all(results.values()),'checks':results,
            'slow_io_scope':'deterministic fault clock; overhead measurement below uses actual monotonic time',
            'automatic_retry':False}


def _route_work(tree,x,m,p):
    """Actual candidate hard routes and full-distribution sufficient statistics."""
    from backend.training.warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
    output=[]
    def walk(node,rows):
        if node is None:
            targets=p[rows].astype(np.float64);masks=m[rows]
            bias=np.log((targets.sum(0)+.2)/(masks.sum(0)+1.));bias-=bias.mean()
            logits=np.where(masks>0,bias,-1e9);logits-=np.max(logits,axis=1,keepdims=True)
            q=np.exp(logits);q/=q.sum(1,keepdims=True)
            kl=np.sum(targets*(np.log(np.maximum(targets,1e-300))-np.log(np.maximum(q,1e-300))),axis=1)
            output.append([len(rows),float(kl.sum()),float(targets.sum()),sha256(rows.astype('<i8').tobytes()).hexdigest()]);return
        w=np.asarray(node.weight,np.float32)
        go=split_scores(x[rows],w[None],np.asarray([node.bias],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
        walk(node.left,rows[~go]);walk(node.right,rows[go])
    walk(tree,np.arange(len(x),dtype=np.int64));return digest(output)


def large_replay(directory,*,fixture=FIXTURE):
    """Real legacy state read once, then new bounded codec/queue replay only."""
    from experiments.warehouse import v622_recovery_protocol as old
    from backend.training.warehouse_v625_search import _Context,_Codec,_score,_tuple_tree
    from backend.training.warehouse_v619_tree import _State
    from scripts.run_warehouse_v622_capacity import arrays
    from scripts.audit_warehouse_v624_upgrade import Evidence
    fixture=Path(fixture).resolve();assert fixture==FIXTURE.resolve() and file_sha(fixture)==FIXTURE_SHA
    # Explicitly trusted, content-pinned local historical artifact. No other
    # arbitrary pickle path is accepted and none enters scientific caches.
    with gzip.open(fixture,'rb') as f:legacy=pickle.load(f)
    state=legacy['state'];actor,_=old.checked_actor(ROOT);episodes,manifest=old.checked_inputs(ROOT)
    fold=old.grouped_folds()[1];rows=[r for case in fold['splits']['fit'] for r in episodes[case['base']]['robot_2']]
    x,m,p,episode_ids=arrays(actor,'robot_2',rows);names=tuple(actor.roles['robot_2'].feature_names)
    from env.warehouse.navigation import ACTIONS
    indices=tuple(names.index('legal_'+a.lower()) for a in ACTIONS)
    queue=state['queue'];assert len(queue)>=128
    # Exactly two real checkpoint intervals; no repeated warmup or artificial
    # workloads are introduced to make the denominator larger.
    work_trees=[q[0] for q in queue[:128]]
    baseline_started=time.monotonic();expected=[_route_work(t,x,m,p) for t in work_trees]
    baseline_seconds=time.monotonic()-baseline_started
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);path=directory/'checkpoint.json'
    identity={'engineering_only':True,'fixture_sha256':FIXTURE_SHA,'input_manifest_sha256':digest(manifest),
              'candidate_count':128,'role':'robot_2','fold':1}
    measured_started=time.monotonic();s=CheckpointSession(path,identity)
    ctx=_Context(x,m,p,np.asarray(episode_ids),names,indices,'P0',s);codec=_Codec(ctx)
    memo={};leaf_refs=OrderedDict()
    def mapped(original):
        key=original.key
        if key in memo:return memo[key]
        for leaf in original.leaves:
            leaf.model={'engineering_only_legacy_constant_statistics':True,'bias':plain(leaf.logits),
                        'support':[],'weights':[[] for _ in range(5)],'report':plain(leaf.optimizer)}
            content={'model':leaf.model,'qualified':bool(leaf.qualified),'agreement':float(leaf.agreement),
                     'expected_mass':float(leaf.expected_mass),'kl_sum':float(leaf.kl_sum),
                     'robust':plain(leaf.robust),'rows_sha256':sha256(leaf.region.rows.astype('<i8').tobytes()).hexdigest()}
            leaf.cache_ref=s.store.put_json(content);leaf.cache_key=leaf.cache_ref['sha256']
            leaf_refs[leaf.cache_key]=leaf.cache_ref
            if len(leaf_refs)>512:leaf_refs.popitem(last=False)
        converted=_State(original.tree,original.leaves,original.regions,_score(original.leaves),key)
        memo[key]=converted;return converted
    actual_states=[mapped(z) for z in state['states'].values()]
    actual_best,actual_start=mapped(state['best']),mapped(state['start'])
    from backend.training.warehouse_program_search import _BeamEntry
    actual_queue=[(tree,kind,label,_BeamEntry(mapped(parent.state),parent.nonimproving_steps)) for tree,kind,label,parent in queue]
    actual_beam=[_BeamEntry(mapped(e.state),e.nonimproving_steps) for e in state['beam']]
    refs=[codec.state_ref(z) for z in actual_states];queue_ref=codec.queue_ref(actual_queue)
    payload={'schema':'engineering-v625-real-state-replay.v1','runtime':{'cursor':64,'queue_ref':queue_ref},
             'states':refs,'best':codec.state_ref(actual_best),'start':codec.state_ref(actual_start),
             'beam':[[codec.state_ref(e.state),e.nonimproving_steps] for e in actual_beam],
             'leaf_cache':list(leaf_refs.items()),'rng':legacy['numpy_rng'][1].tolist()}
    observed=[_route_work(t,x,m,p) for t in work_trees[:64]]
    s.commit(payload,status='paused');first=s.summary();restore_started=time.monotonic()
    restored=CheckpointSession(path,identity,resume=True);ctx2=_Context(x,m,p,np.asarray(episode_ids),names,indices,'P0',restored)
    codec2=_Codec(ctx2);restored_queue=codec2.queue(restored.state['runtime']['queue_ref'])
    restore_seconds=time.monotonic()-restore_started
    assert len(restored_queue)==len(queue)
    assert [str(z[1:3]) for z in restored_queue]==[str(z[1:3]) for z in queue]
    observed.extend(_route_work(z[0],x,m,p) for z in restored_queue[64:128])
    restored.commit(dict(restored.state,runtime={'cursor':128,'queue_ref':queue_ref}),status='completed')
    second=restored.summary();measured_seconds=time.monotonic()-measured_started
    # Include first object population, small commits, ledger work and restore.
    # Restored CAS read I/O is measured by the store as well as by this broad
    # interval; choose the broad interval without double counting its I/O.
    first_io=first['checkpoint_segment_io_seconds']
    second_io=second['checkpoint_segment_io_seconds']
    # More conservatively, all extra measured time relative to the identical
    # baseline counts as overhead, including Python serialization and codec.
    incremental_overhead=max(0.,measured_seconds-baseline_seconds)
    io_lower_bound=first_io+second_io
    overhead=max(incremental_overhead,io_lower_bound)
    fraction=overhead/measured_seconds
    assert observed==expected
    return {'passed':fraction<=.10,'fixture':{'path':str(fixture),'sha256':FIXTURE_SHA,'bytes':fixture.stat().st_size},
            'engineering_only':True,'legacy_scientific_resume':False,'legacy_cache_imported_for_science':False,
            'input_manifest_sha256':digest(manifest),'frozen_actor_sha256':old.ACTOR_SHA256,
            'actual_queue_entries':len(queue),'replayed_candidates':128,'new_bounded_state_entries':len(refs),
            'new_bounded_leaf_cache_entries':len(leaf_refs),'fit_physical_frames':len(x),
            'baseline_work_seconds':baseline_seconds,'checkpoint_replay_seconds':measured_seconds,
            'first_CAS_and_commit_io_seconds':first_io,'second_CAS_and_commit_io_seconds':second_io,
            'restore_inclusive_seconds':restore_seconds,'checkpoint_overhead_seconds':overhead,
            'cumulative_checkpoint_io_upper_bound_seconds':second['checkpoint_total_io_upper_bound_seconds'],
            'cumulative_actual_elapsed_lower_bound_seconds':second['actual_elapsed_lower_bound_seconds'],
            'checkpoint_fraction':fraction,'maximum_checkpoint_fraction':.10,
            'overhead_definition':'max(measured replay minus identical no-checkpoint replay, measured total CAS/ledger/manifest I/O); includes first population and restore, no sleeps',
            'safe_pause_replay_equal':True,'operation_output_sha256':digest(expected),
            'first_timing':first,'second_timing':second,
            'limitations':['No solver timing is added to this denominator; this is a conservative real-state routing/statistics replay, not proof of every full search job.',
                          'Each actual scientific job must independently meet measured checkpoint fraction <= 10%.',
                          'Legacy state is a trusted engineering fixture only; new bounded caches retain 128 states and 512 leaf entries.']}


def snapshot_sources(files, source_sha, directory, *, root=ROOT):
    """Copy and verify the measured source closure before any preflight work."""
    root=Path(root).resolve();directory=Path(directory).resolve()/source_sha
    if digest(files)!=source_sha:raise RuntimeError('source closure digest changed')
    if directory.is_symlink():raise RuntimeError('source snapshot cannot be a symlink')
    directory.mkdir(parents=True,exist_ok=True)
    total=0
    for relative,expected in sorted(files.items()):
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('source snapshot path escapes closure')
        source=root/rel;target=directory/rel
        if source.is_symlink() or not source.is_file():raise RuntimeError('missing/symlink source')
        raw=source.read_bytes()
        if sha256(raw).hexdigest()!=expected:raise RuntimeError('source changed before measurement')
        total+=len(raw)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or file_sha(target)!=expected:
                raise RuntimeError('existing source snapshot differs')
        else:
            with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            target.chmod(0o444)
        if target.stat().st_mode & 0o222:raise RuntimeError('source snapshot is not readonly')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=set(files):raise RuntimeError('source snapshot contains missing/extra files')
    return {'path':str(directory),'source_sha256':source_sha,'file_count':len(files),'bytes':total,
            'verified_before_measurement':True,'files_readonly':True}


def preflight(output,*,large=True):
    from experiments.warehouse import v625_protocol as v
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise RuntimeError('engineering receipts are single-use')
    source_files=v.source_files();source=digest(source_files)
    snapshot=snapshot_sources(source_files,source,output.parent/'source')
    if v.source_sha()!=source:raise RuntimeError('source changed while creating preflight snapshot')
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='warehouse-v625-engineering-') as tmp:
        fault=fault_suite(Path(tmp)/'faults')
        replay=large_replay(Path(tmp)/'large') if large else {'passed':False,'not_run':True}
    receipt=dict(ZERO,schema='warehouse-v625-engineering-preflight.v1',candidate_id=v.CANDIDATE,
        source_sha256=source,source_files=source_files,source_snapshot=snapshot,passed=fault['passed'] and replay['passed'],
        fault_checks=fault,real_state_replay=replay,scientific_fit_calls=0,
        fresh_data_used=False,automatic_retry=False,elapsed_seconds=time.monotonic()-started,
        at_utc=datetime.now(timezone.utc).isoformat())
    if v.source_sha()!=source:raise RuntimeError('source changed during engineering preflight')
    payload=json.dumps(plain(receipt),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    with output.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    content_path=output.with_name('preflight.'+sha256(payload).hexdigest()+'.json')
    os.link(output,content_path)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--faults-only',action='store_true')
    a=p.parse_args()
    if a.faults_only:
        with tempfile.TemporaryDirectory() as tmp:print(json.dumps(fault_suite(tmp),sort_keys=True))
        return
    from experiments.warehouse import v625_protocol as v
    output=a.output or v.OUTPUT.parent.parent/'engineering'/'preflight-03.json'
    result=preflight(output);print(json.dumps({'passed':result['passed'],'output':str(output),
        'checkpoint_fraction':result['real_state_replay']['checkpoint_fraction']},sort_keys=True))

if __name__=='__main__':main()
