"""Lossless rows-only storage for NEW, unbound future-long target stages.

This is not installed in the existing producer. An admitted future producer
must call it before publishing its final target manifest/seal. It accepts no
old path, already bound package, NN/evaluation capability or execution action.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import fcntl,json,os,re,stat,tempfile
from types import SimpleNamespace

VERSION='warehouse-v642-new-long-rows-hardlink-cas.v1'
ROOT=Path(__file__).resolve().parents[2]
STORAGE_ROOT=ROOT/'output/warehouse_v642/future-long-row-storage'
FILES=('experiments/warehouse/v642_rows_cas.py','tests/test_warehouse_v642_rows_cas.py','docs/warehouse_v642_rows_cas.md')
ROW_NAMES=frozenset(f'{role}-{split}-rows.json' for role in ('robot_1','robot_2')
                    for split in ('fit','calibration','eligibility','validation'))
ACTIVE=ContextVar('warehouse_v642_new_long_rows_stage',default=None)
_TOKEN=object()


def require(value,message):
    if not value:raise RuntimeError(message)


def digest_bytes(value):return sha256(value).hexdigest()


def source_files():return {name:digest_bytes((ROOT/name).read_bytes()) for name in FILES}


def _new_json(path,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    with path.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    path.chmod(0o444);_sync(path.parent)
    return dict(path=str(path.resolve()),sha256=digest_bytes(raw))


def _sync(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _directory(path):
    require(path.is_absolute() and path.resolve()==path,'storage directory must be canonical and absolute')
    for parent in (path,*path.parents):
        require(not parent.is_symlink(),'storage ancestor is a symlink')
        if parent.exists():require(parent.is_dir(),'storage ancestor is not a directory')


def _identity(path):
    value=path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(),'rows/CAS source must be a regular nonsymlink file')
    require(stat.S_IMODE(value.st_mode)==0o444,'rows must already be read-only 0444 before deduplication')
    return value


def _signature(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,
            value.st_mode,value.st_uid,value.st_gid,value.st_nlink)


def _checked_sha(path):
    before=_identity(path)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as handle:
        require(_signature(os.fstat(handle.fileno()))==_signature(before),'rows changed while opening')
        result=sha256()
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
        require(_signature(os.fstat(handle.fileno()))==_signature(before),'rows changed while hashing')
    require(_signature(_identity(path))==_signature(before),'rows path changed while hashing')
    return result.hexdigest(),before


@dataclass(init=False)
class NewRowsStage:
    def __init__(self,*args,**kwargs):raise RuntimeError('create an exclusive new, unbound rows stage')

    def verify(self):
        require(ACTIVE.get() is self and self._token is _TOKEN and not self.closed and not self.failed,
            'rows stage is not active or has failed/closed')
        _identity(self.header)
        require(self.sources==source_files() and digest_bytes(self.header.read_bytes())==self.header_ref['sha256'],
            'new rows stage source/header changed')
        _directory(self.folder);_directory(self.payload);_directory(self.cas)
        for path in self.payload.rglob('*'):
            require(not path.is_symlink() and (path.is_file() or path.is_dir()),'new target stage contains a symlink or non-file')
            require(path.name not in ('completed.json','result.json','manifest.json','source-binding.json')
                and not path.name.endswith('.completed.json'),'bound/sealed target packages cannot be deduplicated')
        require(self.payload.stat().st_dev==self.cas.stat().st_dev,'rows CAS and stage must use the same filesystem')
        return self


@contextmanager
def _stage(root,*,owner_sha256,stage_id,token,probe=False):
    require(token is _TOKEN and re.fullmatch('[0-9a-f]{64}',owner_sha256) is not None,'explicit new owner SHA is required')
    require(type(stage_id) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',stage_id) is not None,'unsafe rows stage ID')
    require(ACTIVE.get() is None,'rows stage contexts cannot nest')
    _directory(root)
    owner=root/owner_sha256;folder=owner/'stages'/stage_id
    require(not folder.exists(),'rows stage is single-use, including after interruption')
    folder.mkdir(parents=True,exist_ok=False);payload=folder/'unbound-target';payload.mkdir()
    cas=owner/'rows';_directory(cas);cas.mkdir(exist_ok=True)
    stage=object.__new__(NewRowsStage)
    stage._token=_TOKEN;stage.folder=folder;stage.payload=payload;stage.cas=cas;stage.owner=owner
    stage.closed=False;stage.failed=False;stage.used=False;stage.sources=source_files();stage.operations=[]
    stage.header=folder/'stage.json'
    stage.header_ref=_new_json(stage.header,dict(version=VERSION,owner_sha256=owner_sha256,stage_id=stage_id,
        source_files=stage.sources,probe_only=probe,NN_authority=False,environment_authority=False,
        final_target_binding_authority=False,already_sealed_input_authorized=False))
    current=ACTIVE.set(stage)
    try:
        yield stage
        stage.verify()
        _new_json(folder/'storage-stage-closed.json',dict(version=VERSION,stage=stage.header_ref,
            operations=stage.operations,storage_only=True,NN_updates=0,environment_resets=0,pilot_opened=False))
    except BaseException as error:
        stage.failed=True
        _new_json(folder/'storage-stage-failed.json',dict(version=VERSION,stage=stage.header_ref,
            error_type=type(error).__name__,error=str(error),completed_operations=stage.operations,
            automatic_retry=False,historical_source_bytes_deleted=False,pilot_opened=False))
        raise
    finally:stage.closed=True;ACTIVE.reset(current)


def new_stage(*,owner_sha256,stage_id):
    """Allocate a fresh future-long stage, never import an existing package."""
    return _stage(STORAGE_ROOT,owner_sha256=owner_sha256,stage_id=stage_id,token=_TOKEN)


def new_probe(*,owner_sha256,stage_id):
    """Fresh /tmp copy-only namespace. Cannot adopt a supplied old directory."""
    root=Path(tempfile.mkdtemp(prefix='warehouse-v642-rows-cas-probe-')).resolve()
    return _stage(root,owner_sha256=owner_sha256,stage_id=stage_id,token=_TOKEN,probe=True)


@contextmanager
def probe_workspace(*,owner_sha256):
    """Several fresh copy-only stages share one newly allocated probe CAS."""
    root=Path(tempfile.mkdtemp(prefix='warehouse-v642-rows-cas-probe-')).resolve();alive=[True]
    def create(stage_id):
        require(alive[0],'probe workspace is closed')
        return _stage(root,owner_sha256=owner_sha256,stage_id=stage_id,token=_TOKEN,probe=True)
    try:yield SimpleNamespace(root=root,stage=create)
    finally:alive[0]=False


def _publish_object(stage,target,sha,before):
    base=stage.cas/(sha+'.rows.json')
    require(not base.is_symlink(),'CAS object cannot be a symlink')
    require(target.stat().st_dev==stage.cas.stat().st_dev,'cross-filesystem rows CAS is forbidden')
    if not base.exists():
        temporary=stage.cas/(sha+'.new')
        with os.fdopen(os.open(target,os.O_RDONLY|os.O_NOFOLLOW),'rb') as source,temporary.open('xb') as output:
            require(_signature(os.fstat(source.fileno()))==_signature(before),'new rows changed before CAS publication')
            observed=sha256()
            for block in iter(lambda:source.read(1024*1024),b''):output.write(block);observed.update(block)
            output.flush();os.fsync(output.fileno())
            require(observed.hexdigest()==sha and _signature(os.fstat(source.fileno()))==_signature(before),
                'new rows changed during CAS publication')
        temporary.chmod(0o444)
        require(_checked_sha(temporary)[0]==sha,'new CAS copy did not preserve exact logical bytes')
        os.link(temporary,base,follow_symlinks=False);temporary.unlink();_sync(stage.cas)
    actual,base_stat=_checked_sha(base)
    require(actual==sha and base_stat.st_dev==before.st_dev
        and (base_stat.st_uid,base_stat.st_gid)==(before.st_uid,before.st_gid),
        'CAS bytes, ownership or filesystem differ; base is never repaired or rewritten')
    return base,base_stat


def deduplicate_rows(stage,*,names):
    """Hardlink exact bytes once per fresh stage, before final binding.

    Input files must be newly produced regular single-link 0444 files. The
    CAS baseline's bytes, mode and mtime are unchanged. Linked inputs inherit
    the CAS mtime: pre-binding mtime differences are explicitly recorded.
    """
    stage.verify()
    require(not stage.used and type(names) in (list,tuple) and names and len(names)==len(set(names))
        and set(names)<=ROW_NAMES,'one finite, explicit rows-only operation per new stage')
    stage.used=True
    lock=stage.owner/'rows-cas.lock'
    require(not lock.is_symlink(),'CAS lock cannot be a symlink')
    try:
        with lock.open('a+b') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for name in names:
                stage.verify();target=stage.payload/name
                require(target.parent==stage.payload,'rows source escaped its new temporary target')
                sha,before=_checked_sha(target)
                require(before.st_nlink==1,'new rows input must be a private single-link file')
                base,base_before=_publish_object(stage,target,sha,before)
                require(_signature(_identity(target))==_signature(before),'new target changed before atomic deduplication')
                temporary=target.with_name(target.name+'.linking')
                os.link(base,temporary,follow_symlinks=False)
                require(_signature(_identity(target))==_signature(before),'new target changed before atomic replacement')
                os.replace(temporary,target);_sync(target.parent)
                actual,after=_checked_sha(target);base_after=_identity(base)
                require(actual==sha and (after.st_dev,after.st_ino)==(base_after.st_dev,base_after.st_ino)
                    and after.st_nlink>=2 and base_after.st_mode==base_before.st_mode
                    and base_after.st_mtime_ns==base_before.st_mtime_ns,'hardlink or baseline preservation check failed')
                row=dict(path=str(target),sha256=sha,bytes=before.st_size,CAS=str(base),
                    shared_device=after.st_dev,shared_inode=after.st_ino,nlink_after=after.st_nlink,
                    private_allocated_bytes_removed=before.st_blocks*512,
                    original_mtime_ns=before.st_mtime_ns,linked_mtime_ns=after.st_mtime_ns,
                    logical_path_bytes_mode_hash_preserved=True,baseline_bytes_mode_mtime_unchanged=True)
                stage.operations.append(row)
        ref=_new_json(stage.folder/'rows-deduplication.json',dict(version=VERSION,stage=stage.header_ref,rows=stage.operations,
            physical_hardlinks_verified=True,storage_only=True,NN_updates=0,environment_resets=0,pilot_opened=False))
        return ref
    except BaseException:stage.failed=True;raise
