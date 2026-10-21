#!/usr/bin/env python3
"""Lossless filesystem compression of waited, sealed CAS NPY packs only.

All original names, logical bytes, SHA256, modes and mtime remain unchanged.
No CAS object is pruned; JSON/ledgers, model NPZs and live jobs are untouched.
The compression is transparent to the existing, unmodified NumPy reader.
"""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import argparse,json,os,platform,re,shutil,stat,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as old,v629_protocol as new
from scripts.measure_warehouse_v628_terminal import waited_jobs
from scripts.run_warehouse_v629 import process_identity
from scripts.run_warehouse_v625 import write_new

COMPRESSED=32


def metadata(path):
    s=path.stat()
    return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,
        uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,inode=s.st_ino,allocated_bytes=s.st_blocks*512,
        flags=getattr(s,'st_flags',0))


def logical_metadata(value):return {k:value[k] for k in ('size','mode','mtime_ns','uid','gid','nlink')}


def event(journal,value):
    journal.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())


def fsync_directory(path):
    handle=os.open(path,os.O_RDONLY)
    try:os.fsync(handle)
    finally:os.close(handle)


def native_copy(source,target,*,compressed):
    args=['/usr/bin/ditto','--noclone']
    args+=['--hfsCompression'] if compressed else ['--nohfsCompression','--nopreserveHFSCompression']
    subprocess.run([*args,str(source),str(target)],check=True,timeout=60)


def compress_file(path,journal,*,copy_function=native_copy):
    """Prepared/verified/committed journal; only byte-identical atomic replace.

    A crash can leave a private .tmp, or an already compressed identical file.
    There is no scientific resume/retry and no silent deletion of old objects.
    """
    path=Path(path)
    old.require(not path.is_symlink() and path.is_file() and re.fullmatch(r'[0-9a-f]{64}\.npy',path.name),
        'only immutable content-addressed NPY pack files may be compressed')
    before=metadata(path)
    old.require(before['nlink']==1 and before['flags'] in (0,COMPRESSED),'shared links or unknown filesystem flags')
    expected=path.stem;old.require(old.file_sha(path)==expected,'CAS pack content SHA differs before compression')
    if before['flags']==COMPRESSED:return dict(path=str(path),sha256=expected,status='already_compressed',saved_allocated_bytes=0)
    old.require(shutil.disk_usage(path.parent).free>=max(4*1024**3,3*before['size']), 'insufficient compression scratch headroom')
    temporary=path.with_name('.native-compression-'+path.stem+'-'+str(os.getpid())+'.tmp')
    old.require(not temporary.exists() and not temporary.is_symlink(),'unresolved prior compression temporary; inspect journal')
    event(journal,dict(event='prepared',path=str(path),sha256=expected,before=before,temporary=str(temporary)))
    try:
        copy_function(path,temporary,compressed=True)
        copied=metadata(temporary)
        old.require(old.file_sha(temporary)==expected and logical_metadata(copied)==logical_metadata(before),
            'native compression changed logical content, timestamps or permissions')
        # NumPy must still read the exact byte-pack; no deserialization of code.
        import numpy as np
        values=np.load(temporary,allow_pickle=False)
        old.require(values.dtype==np.uint8 and values.ndim==1,'object is not a flat CAS byte-pack')
        del values
        old.require(metadata(path)==before and old.file_sha(path)==expected,'original pack changed while copying')
        if copied['flags']!=COMPRESSED or copied['allocated_bytes']>=before['allocated_bytes']:
            temporary.unlink();event(journal,dict(event='skipped_not_smaller',path=str(path),sha256=expected))
            return dict(path=str(path),sha256=expected,status='not_smaller_original_retained',saved_allocated_bytes=0)
        with temporary.open('rb') as handle:os.fsync(handle.fileno())
        event(journal,dict(event='verified_replacement_ready',path=str(path),sha256=expected,
            logical_metadata=logical_metadata(before),allocated_before=before['allocated_bytes'],allocated_after=copied['allocated_bytes']))
        os.replace(temporary,path);fsync_directory(path.parent)
        after=metadata(path)
        old.require(old.file_sha(path)==expected and logical_metadata(after)==logical_metadata(before)
            and after['flags']==COMPRESSED,'committed compressed file differs')
        result=dict(path=str(path),sha256=expected,status='compressed',before=before,after=after,
            saved_allocated_bytes=before['allocated_bytes']-after['allocated_bytes'])
        event(journal,dict(event='committed',**result));return result
    except BaseException as error:
        event(journal,dict(event='failed',path=str(path),sha256=expected,error_type=type(error).__name__,error=str(error),
            retained_temporary=str(temporary) if temporary.exists() else None))
        raise


def settled(protocol):
    plan_path,plan=protocol.checked_plan();exits=waited_jobs(protocol.OUTPUT/'execution/events.jsonl');result=[]
    for job in plan['jobs']:
        row=exits.get(job['job_id']);folder=protocol.OUTPUT/'jobs'/job['job_id']
        if row is None or not (folder/'completed.json').is_file():continue
        old.require(row['job']==job and row['waited'] is True,'worker exit binding differs')
        running=process_identity(row['pid'])
        old.require(running is None or '--job '+job['job_id'] not in running,'cannot touch a live job')
        record=old.bound(old.pointer(folder/'result.json'))
        old.require(record['job']==job
            and json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=old.file_sha(folder/'result.json')),
            'terminal result or completion seal changed')
        # A worker may fail before ever creating a CAS store; it has no packs to maintain.
        if not (folder/'checkpoint.json.objects').exists():continue
        extra={}
        if 'plan_sha256' in record:
            old.require(record['plan_sha256']==old.file_sha(plan_path),'terminal plan binding changed')
        else:
            # v629 exception receipts omit the plan hash. Bind the existing start
            # receipt and committed checkpoint identity instead; never amend receipts.
            old.require(protocol is new and record['status'] in ('failed','scientific_budget_exhausted'),
                'terminal result lacks a supported plan binding')
            start_ref=old.pointer(folder/'started.json');start=old.bound(start_ref)
            checkpoint_ref=old.pointer(folder/'checkpoint.json');checkpoint=old.bound(checkpoint_ref)
            from backend.training.warehouse_v625_checkpoint import digest
            identity=dict(plan_sha256=old.file_sha(plan_path),job=job,initial=start['initial'],
                solver_limits=new.recipe()['limits'],budget_contract='v629-cumulative-common-max256-reserve.v1')
            old.require(start['job']==job and start['plan_sha256']==old.file_sha(plan_path)
                and checkpoint['identity_sha256']==digest(identity)
                and checkpoint['budget']==dict(maximum_calls=484096,wall_seconds=6000.),
                'exception checkpoint or start plan binding changed')
            extra=dict(start=start_ref,checkpoint=checkpoint_ref)
        result.append((protocol.CANDIDATE,job,folder,dict(plan=old.pointer(plan_path),result=old.pointer(folder/'result.json'),worker_exit=row,**extra)))
    return result


def job_inventory(folder):
    store=folder/'checkpoint.json.objects'
    old.require(store.is_dir() and not store.is_symlink(),'missing original CAS object directory')
    files=sorted(store.glob('*.npy'))
    old.require(files,'completed job has no physical NPY packs')
    for path in files:
        old.require(not path.is_symlink() and path.is_file() and path.resolve().parent==store.resolve(),
            'CAS inventory cannot follow a symbolic or external pack path')
    return [dict(path=str(path.resolve()),sha256=path.stem,before=metadata(path)) for path in files]


def apply_job(output,candidate,job,folder,binding):
    target=output/candidate/job['job_id'];target.mkdir(parents=True,exist_ok=False)
    inventory=job_inventory(folder)
    write_new(target/'inventory.json',dict(job=job,source_binding=binding,files=inventory,
        source_script_sha256=old.file_sha(__file__),logical_files_deleted=0))
    rows=[]
    with (target/'transactions.jsonl').open('x') as journal:
        for i,item in enumerate(inventory):
            old.require(metadata(Path(item['path']))==item['before'],'pack inventory changed before transaction')
            rows.append(compress_file(item['path'],journal))
            if (i+1)%100==0:print(json.dumps(dict(job=job['job_id'],compressed_or_checked=i+1,total=len(inventory))),flush=True)
    for key,ref in binding.items():
        if key!='worker_exit':old.require(old.pointer(ref['path'])==ref,'scientific source/receipt changed during maintenance')
    result=dict(schema='warehouse-transparent-CAS-compression.v1',job=job,source_binding=binding,files=rows,
        original_logical_bytes=sum(i['before']['size'] for i in inventory),saved_allocated_bytes=sum(r['saved_allocated_bytes'] for r in rows),
        logical_file_names_and_bytes_unchanged=True,logical_files_deleted=0,mode_and_mtime_preserved=True,
        scope='completed_physical_NPY_packs_only_no_JSON_ledger_model_or_source_changes',
        fit_calls=0,LP_calls=0,environment_resets=0,actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
    write_new(target/'result.json',result);write_new(target/'completed.json',dict(sha256=old.file_sha(target/'result.json')))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true',required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--watch',action='store_true')
    parser.add_argument('--driver-pid',type=int,action='append',default=[]);args=parser.parse_args(argv)
    old.require(platform.system()=='Darwin' and Path('/usr/bin/ditto').is_file(),'native APFS/HFS compression requires this macOS host')
    output=args.output_dir.resolve();base=ROOT/'output/warehouse_storage'
    old.require(output.is_relative_to(base) and output!=base,'maintenance journal must be isolated under warehouse_storage')
    identities={pid:process_identity(pid) for pid in args.driver_pid}
    if args.watch:old.require(identities and all(value is not None and any(name in value for name in (
        'continue_warehouse_v628_seed_audit_bounded.py','run_warehouse_v629.py --pipeline')) for value in identities.values()),
        'watch requires actual live registered driver identities')
    output.mkdir(parents=True,exist_ok=False);source_sha=old.file_sha(__file__)
    write_new(output/'started.json',dict(script_sha256=source_sha,started_utc=datetime.now(timezone.utc).isoformat(),
        watched_process_identities={str(k):v for k,v in identities.items()},scope='finite_two_registered_batches',logical_files_deleted=0))
    done={}
    try:
        while True:
            old.require(old.file_sha(__file__)==source_sha,'maintenance script changed while active')
            for protocol in (old,new):
                for candidate,job,folder,binding in settled(protocol):
                    key=candidate+'/'+job['job_id']
                    if key in done:continue
                    result=apply_job(output,candidate,job,folder,binding);done[key]=old.pointer(output/candidate/job['job_id']/'result.json')
                    print(json.dumps(dict(completed_storage_job=key,saved_allocated_bytes=result['saved_allocated_bytes'])),flush=True)
            if not args.watch or not any(process_identity(pid)==identity for pid,identity in identities.items()):break
            time.sleep(30)
        write_new(output/'completed.json',dict(jobs=done,saved_allocated_bytes=sum(old.bound(ref)['saved_allocated_bytes'] for ref in done.values()),
            logical_files_deleted=0,scientific_fit_or_training_started=False,pilot_opened=False))
    except BaseException as error:
        write_new(output/'failed.json',dict(error_type=type(error).__name__,error=str(error),completed_jobs=done,
            automatic_retry=False,logical_files_deleted=0));raise


if __name__=='__main__':main()
