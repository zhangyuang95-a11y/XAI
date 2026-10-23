"""Temporary copied JSON bytes and real hardlinks; no scientific execution."""
from pathlib import Path
from types import SimpleNamespace
import errno,json,os,subprocess
import pytest
from experiments.warehouse import v642_rows_cas as r


OWNER='a'*64


def row(stage,name,data):
    path=stage.payload/name;path.write_bytes(data);path.chmod(0o444);return path


def test_direct_construction_or_old_path_does_not_grant_a_storage_capability(tmp_path):
    with pytest.raises(RuntimeError,match='exclusive new'):r.NewRowsStage()
    with pytest.raises(AttributeError):r.deduplicate_rows(tmp_path,names=['robot_1-fit-rows.json'])


def test_real_inode_nlink_byte_mode_and_baseline_preservation_across_two_new_stages(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'STORAGE_ROOT',tmp_path/'future-only')
    data=b'[{"episode":"synthetic-only","frame":0}]'*10000
    baseline=None
    for i in range(2):
        with r.new_stage(owner_sha256=OWNER,stage_id=f'call-{i}') as stage:
            first=row(stage,'robot_1-fit-rows.json',data);second=row(stage,'robot_2-fit-rows.json',data)
            before=first.stat();ref=r.deduplicate_rows(stage,names=[first.name,second.name])
            receipt=json.loads(Path(ref['path']).read_bytes());base=Path(receipt['rows'][0]['CAS'])
            assert first.stat().st_ino==second.stat().st_ino==base.stat().st_ino
            assert first.stat().st_dev==second.stat().st_dev==base.stat().st_dev
            assert base.stat().st_nlink==3+2*i
            assert first.read_bytes()==second.read_bytes()==base.read_bytes()==data
            assert first.stat().st_mode&0o777==second.stat().st_mode&0o777==base.stat().st_mode&0o777==0o444
            if baseline is not None:
                assert (base.read_bytes(),base.stat().st_mode,base.stat().st_mtime_ns)==baseline
            baseline=(base.read_bytes(),base.stat().st_mode,base.stat().st_mtime_ns)
            assert all(row['sha256']==r.digest_bytes(data) for row in receipt['rows'])
        assert stage.closed and (stage.folder/'storage-stage-closed.json').is_file()
        with pytest.raises(RuntimeError,match='not active'):r.deduplicate_rows(stage,names=[first.name])


@pytest.mark.parametrize('kind',['symlink','payload_symlink','hardlinked_input','writable','unknown_name','sealed','source_changed','CAS_symlink'])
def test_old_bound_foreign_or_mutable_paths_rejected_without_altering_original(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(r,'STORAGE_ROOT',tmp_path/'future-only')
    original=tmp_path/'old-20k-baseline.json';original.write_bytes(b'[]');original.chmod(0o444)
    identity=(original.read_bytes(),original.stat().st_mode,original.stat().st_mtime_ns)
    with pytest.raises(RuntimeError):
        with r.new_stage(owner_sha256=OWNER,stage_id='one') as stage:
            name='robot_1-fit-rows.json'
            if kind=='symlink':(stage.payload/name).symlink_to(original)
            elif kind=='payload_symlink':
                stage.payload.rmdir();stage.payload.symlink_to(tmp_path,target_is_directory=True)
            elif kind=='hardlinked_input':os.link(original,stage.payload/name)
            else:
                path=row(stage,name,b'[]')
                if kind=='writable':path.chmod(0o644)
                elif kind=='unknown_name':name='../old-20k-baseline.json'
                elif kind=='sealed':(stage.payload/'completed.json').write_text('{}')
                elif kind=='source_changed':monkeypatch.setattr(r,'source_files',lambda:{'changed':'invalid'})
                elif kind=='CAS_symlink':(stage.cas/(r.digest_bytes(b'[]')+'.rows.json')).symlink_to(original)
            r.deduplicate_rows(stage,names=[name])
    assert (original.read_bytes(),original.stat().st_mode,original.stat().st_mtime_ns)==identity


def test_exdev_is_refused_without_copy_fallback_or_baseline_modification(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'STORAGE_ROOT',tmp_path/'future-only')
    with pytest.raises(OSError,match='synthetic EXDEV'):
        with r.new_stage(owner_sha256=OWNER,stage_id='cross-fs') as stage:
            source=row(stage,'robot_1-fit-rows.json',b'[1]')
            def cross(*args,**kwargs):raise OSError(errno.EXDEV,'synthetic EXDEV')
            monkeypatch.setattr(r.os,'link',cross)
            r.deduplicate_rows(stage,names=[source.name])
    assert source.read_bytes()==b'[1]' and source.stat().st_nlink==1
    assert (stage.folder/'storage-stage-failed.json').is_file()


def test_stage_and_operation_are_single_use_even_after_interruption(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'STORAGE_ROOT',tmp_path/'future-only')
    with pytest.raises(RuntimeError,match='one finite'):
        with r.new_stage(owner_sha256=OWNER,stage_id='single') as stage:
            name='robot_1-fit-rows.json';row(stage,name,b'[1]')
            r.deduplicate_rows(stage,names=[name]);r.deduplicate_rows(stage,names=[name])
    with pytest.raises(RuntimeError,match='single-use'):
        with r.new_stage(owner_sha256=OWNER,stage_id='single'):pytest.fail('rearmed failed stage')


def test_actual_du_counts_shared_rows_once_including_CAS_and_receipt_overhead(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'STORAGE_ROOT',tmp_path/'future-only')
    data=(b'[{"synthetic":"rows","episode":1,"frame":10}]'*30000)
    def du():return int(subprocess.check_output(['du','-sk',str(r.STORAGE_ROOT)]).split()[0])*1024
    with r.new_stage(owner_sha256=OWNER,stage_id='space-proof') as stage:
        names=['robot_1-fit-rows.json','robot_2-fit-rows.json','robot_1-validation-rows.json','robot_2-validation-rows.json']
        for name in names:row(stage,name,data)
        before=du();r.deduplicate_rows(stage,names=names);after=du()
        assert before-after>2*len(data)
        assert len({(stage.payload/name).stat().st_ino for name in names})==1
        assert all((stage.payload/name).read_bytes()==data for name in names)


def test_probe_always_allocates_a_new_private_namespace():
    with r.new_probe(owner_sha256=OWNER,stage_id='only-probe') as stage:
        assert 'warehouse-v642-rows-cas-probe-' in str(stage.folder)
        assert stage.folder.is_absolute() and not stage.folder.is_relative_to(r.ROOT)
        assert json.loads(stage.header.read_bytes())['probe_only'] is True
