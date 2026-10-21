"""Filesystem maintenance tests on private synthetic byte-packs only."""
from hashlib import sha256
from pathlib import Path
import json,os,platform,shutil
import numpy as np
import pytest
from scripts import compress_warehouse_checkpoint_storage as c


def pack(tmp_path):
    initial=tmp_path/'initial.npy';np.save(initial,np.tile(np.arange(256,dtype=np.uint8),8192),allow_pickle=False)
    path=tmp_path/(c.old.file_sha(initial)+'.npy');initial.rename(path)
    os.utime(path,ns=(1600000000123456789,1600000000123456789));path.chmod(0o444)
    return path


@pytest.mark.skipif(platform.system()!='Darwin',reason='real native compression requires APFS/HFS macOS')
def test_native_atomic_compression_preserves_every_logical_byte_and_reader(tmp_path):
    path=pack(tmp_path);before=c.metadata(path);expected=np.load(path,allow_pickle=False).copy()
    with (tmp_path/'journal.jsonl').open('x') as journal:result=c.compress_file(path,journal)
    assert result['status']=='compressed' and result['saved_allocated_bytes']>0
    after=c.metadata(path);assert c.logical_metadata(before)==c.logical_metadata(after)
    assert c.old.file_sha(path)==path.stem and np.array_equal(np.load(path,allow_pickle=False),expected)
    assert after['flags']==c.COMPRESSED
    rows=[json.loads(line) for line in (tmp_path/'journal.jsonl').read_text().splitlines()]
    assert [row['event'] for row in rows]==['prepared','verified_replacement_ready','committed']
    restored=tmp_path/'restored.npy';c.native_copy(path,restored,compressed=False)
    assert c.old.file_sha(restored)==path.stem and c.metadata(restored)['flags']==0
    assert c.logical_metadata(c.metadata(restored))==c.logical_metadata(before)
    with (tmp_path/'second-journal.jsonl').open('x') as journal:again=c.compress_file(path,journal)
    assert again['status']=='already_compressed' and c.metadata(path)==after


def test_failed_copy_never_replaces_original_and_retains_failure_journal(tmp_path):
    path=pack(tmp_path);before=c.metadata(path)
    def bad(source,target,**kw):target.write_bytes(b'wrong bytes')
    with (tmp_path/'journal.jsonl').open('x') as journal:
        with pytest.raises(RuntimeError,match='changed logical content'):c.compress_file(path,journal,copy_function=bad)
    assert c.metadata(path)==before and c.old.file_sha(path)==path.stem
    rows=[json.loads(line) for line in (tmp_path/'journal.jsonl').read_text().splitlines()]
    assert rows[-1]['event']=='failed' and Path(rows[-1]['retained_temporary']).exists()


def test_atomic_replacement_is_refused_if_original_changes_during_copy(tmp_path):
    path=pack(tmp_path)
    def stale(source,target,**kw):
        shutil.copy2(source,target);source.chmod(0o644);source.write_bytes(b'external concurrent change')
    with (tmp_path/'journal.jsonl').open('x') as journal:
        with pytest.raises(RuntimeError,match='original pack changed'):c.compress_file(path,journal,copy_function=stale)
    assert path.read_bytes()==b'external concurrent change'
    assert json.loads((tmp_path/'journal.jsonl').read_text().splitlines()[-1])['event']=='failed'


@pytest.mark.parametrize('kind',['symbolic','hardlink','wrong_hash','model_npz'])
def test_non_cas_or_shared_files_are_refused_before_copy(tmp_path,kind):
    path=pack(tmp_path);called=[]
    if kind=='symbolic':
        alias=tmp_path/('b'*64+'.npy');alias.symlink_to(path);path=alias
    if kind=='hardlink':os.link(path,tmp_path/'another-link')
    if kind=='wrong_hash':
        renamed=tmp_path/('b'*64+'.npy');path.rename(renamed);path=renamed
    if kind=='model_npz':
        renamed=tmp_path/'actor.npz';path.rename(renamed);path=renamed
    with (tmp_path/'journal.jsonl').open('x') as journal:
        with pytest.raises(RuntimeError):c.compress_file(path,journal,copy_function=lambda *a,**k:called.append(True))
    assert not called


def test_job_inventory_only_contains_npy_not_proofs_or_models(tmp_path):
    store=tmp_path/'checkpoint.json.objects';store.mkdir();path=pack(store)
    (store/'proof.json').write_text('{}');(tmp_path/'actor.npz').write_bytes(b'model')
    inventory=c.job_inventory(tmp_path)
    assert [row['path'] for row in inventory]==[str(path)]


def test_inventory_does_not_resolve_symlink_before_safety_check(tmp_path):
    outside=tmp_path/'live-job';outside.mkdir();source=pack(outside)
    closed=tmp_path/'closed-job/checkpoint.json.objects';closed.mkdir(parents=True)
    (closed/source.name).symlink_to(source)
    before=c.metadata(source)
    with pytest.raises(RuntimeError,match='symbolic or external pack'):
        c.job_inventory(closed.parent)
    assert c.metadata(source)==before and not c.metadata(source)['flags']


def test_low_scratch_space_refuses_before_copy_or_original_mutation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path=pack(tmp_path);before=c.metadata(path);calls=[]
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with (tmp_path/'journal.jsonl').open('x') as journal:
        with pytest.raises(RuntimeError,match='scratch headroom'):
            c.compress_file(path,journal,copy_function=lambda *a,**k:calls.append(True))
    assert not calls and c.metadata(path)==before and (tmp_path/'journal.jsonl').read_text()==''


def test_job_without_owned_worker_exit_never_becomes_eligible(tmp_path,monkeypatch):
    from types import SimpleNamespace
    root=tmp_path/'candidate';root.mkdir();plan=root/'plan.json';plan.write_text('{}')
    fake=SimpleNamespace(OUTPUT=root,CANDIDATE='synthetic',checked_plan=lambda:(plan,{'jobs':[{'job_id':'live'}]}))
    (root/'jobs/live').mkdir(parents=True);(root/'jobs/live/completed.json').write_text('{}')
    monkeypatch.setattr(c,'waited_jobs',lambda path:{})
    assert c.settled(fake)==[]


@pytest.mark.parametrize('tamper',[None,'start','checkpoint','live'])
def test_v629_exception_receipt_uses_original_start_and_checkpoint_bindings(tmp_path,monkeypatch,tamper):
    from types import SimpleNamespace
    from backend.training.warehouse_v625_checkpoint import digest
    root=tmp_path/'candidate';folder=root/'jobs/test';folder.mkdir(parents=True)
    (folder/'checkpoint.json.objects').mkdir();plan=root/'plan.json';plan.write_text('{}')
    job=dict(job_id='test');limits=dict(leaf=80000,node=4096,proof=400000)
    initial=dict(path='original',sha256='anchor')
    start=dict(job=job,plan_sha256=c.old.file_sha(plan),initial=initial)
    identity={**start,'solver_limits':limits,'budget_contract':'v629-cumulative-common-max256-reserve.v1'}
    checkpoint=dict(identity_sha256=digest(identity),budget=dict(maximum_calls=484096,wall_seconds=6000.))
    if tamper=='start':start['plan_sha256']='wrong'
    if tamper=='checkpoint':checkpoint['identity_sha256']='wrong'
    for name,data in [('started',start),('checkpoint',checkpoint),('result',dict(job=job,status='scientific_budget_exhausted'))]:
        (folder/(name+'.json')).write_text(json.dumps(data))
    (folder/'completed.json').write_text(json.dumps(dict(result_sha256=c.old.file_sha(folder/'result.json'))))
    fake=SimpleNamespace(OUTPUT=root,CANDIDATE='synthetic',checked_plan=lambda:(plan,dict(jobs=[job])),recipe=lambda:dict(limits=limits))
    monkeypatch.setattr(c,'new',fake)
    monkeypatch.setattr(c,'waited_jobs',lambda path:dict(test=dict(job=job,pid=123,waited=True)))
    monkeypatch.setattr(c,'process_identity',lambda pid:'python --job test' if tamper=='live' else None)
    if tamper:
        with pytest.raises(RuntimeError):c.settled(fake)
    else:
        row=c.settled(fake)[0]
        assert row[3]['start']==c.old.pointer(folder/'started.json')
        assert row[3]['checkpoint']==c.old.pointer(folder/'checkpoint.json')
        assert 'plan_sha256' not in json.loads((folder/'result.json').read_bytes())
