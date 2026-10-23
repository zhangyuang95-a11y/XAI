"""Explicit external cutover of the pinned v2 worker; no training authority.

Default preparation/assertion is read-only with respect to the worker. Only an
explicit execute with ready successor source proof can signal the exact pinned
claim/PID/parent/argv. A stopped worker is resumed on any failed pre-stop check.
No partial physical suffix or newly committed optimizer result may be dropped.
"""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import io,json,os,re,signal,subprocess,time
import torch
from . import v642_calibration_recovery as v2
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from backend.training.warehouse_v622_actor_training import CHECKPOINT_SCHEMA
from backend.training.warehouse_v6_actor import MODEL_VERSION
from scripts.run_warehouse_v625 import write_new

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'output/warehouse_v642/checkpoint-cutovers'
VERSION='warehouse-v642-pinned-checkpoint-external-cutover.v1'
PREFIX_VERSION='warehouse-v642-v3-immutable-prefix-manifest.v1'
PINNED_PLAN_SHA256='addc2567fdf48c3ac008476ceee231b544469a08b02cda0e5a0a09a8aebbef2b'
PINNED_CLAIM_SHA256='904da532173558a061db5ebe8d849d988347d0e98c29087b6b88b54849c6a112'
PINNED_SUPERVISOR_STARTED_SHA256='50c61394520598decf7f348fe7d03b0e6b21b02e7e7a4780b00eeba2d08b222d'
MODES=('original-safe-replay','checkpoint-18600')
FILES=('experiments/warehouse/v642_checkpoint_cutover.py','scripts/cutover_warehouse_v642_checkpoint.py',
       'tests/test_warehouse_v642_checkpoint_cutover.py','docs/warehouse_v642_checkpoint_cutover.md')


def source_files():return {name:file_sha(ROOT/name) for name in FILES}


def _put(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    write_new(temporary,value);temporary.chmod(0o444);os.link(temporary,path);temporary.unlink()
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return pointer(path)


def _asset(ref,*,readonly=True):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'explicit exact SHA pointer required')
    path=Path(ref['path']);require(path.is_absolute() and path.resolve()==path,'noncanonical cutover input')
    for part in (path,*path.parents):require(not part.is_symlink(),'symlinked cutover input')
    require(path.is_file() and pointer(path)==ref and (not readonly or not path.stat().st_mode&0o222),
            'cutover input missing, writable or changed')
    return path


def _lines(path):
    raw=Path(path).read_bytes();require(not raw or raw.endswith(b'\n'),'partial journal line cannot authorize cutover')
    return [json.loads(row) for row in raw.splitlines()]


def _process(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,command='],capture_output=True,text=True,check=False)
    require(r.returncode==0 and r.stdout.strip(),'pinned worker is not observable')
    fields=r.stdout.strip().split(None,3)
    require(len(fields)==4 and int(fields[0])==pid,'ambiguous worker process identity')
    return dict(pid=int(fields[0]),ppid=int(fields[1]),status=fields[2],command=fields[3])


def _binding(recovery_plan_ref,supervisor_started_ref,worker_pid,target_checkpoint_path,mode,*,terminal=False):
    _asset(recovery_plan_ref)
    require(recovery_plan_ref['sha256']==PINNED_PLAN_SHA256,'only the pinned current v2 attempt may be cut over')
    p=bound(recovery_plan_ref);folder=Path(recovery_plan_ref['path']).parent
    require(p['version']==v2.VERSION and p['source_files']==v2.source_files() and p['source_files'].keys()==set(v2.NEW_SOURCE_NAMES),
        'v2 frozen source or plan version changed')
    require(mode in MODES and type(worker_pid) is int and worker_pid>1,'unknown cutover mode or PID')
    _asset(supervisor_started_ref)
    require(Path(supervisor_started_ref['path'])==folder/'supervisor-started.json','wrong source-bound supervisor launch')
    started=bound(supervisor_started_ref);claim_ref=pointer(v2.claim_path(p['original_plan']));_asset(claim_ref);claim=bound(claim_ref)
    require(claim_ref['sha256']==PINNED_CLAIM_SHA256 and supervisor_started_ref['sha256']==PINNED_SUPERVISOR_STARTED_SHA256
        and claim['version']==v2.VERSION and claim['recovery_plan']==recovery_plan_ref and claim['owner_pid']==worker_pid
        and claim['single_use_even_if_interrupted'] is True and claim['source_files']==p['source_files'],
        'PID is not the exact pinned single-use v2 claim owner')
    cli=dict(path=str(ROOT/'scripts/resume_warehouse_v642_calibration.py'),sha256=p['source_files']['scripts/resume_warehouse_v642_calibration.py'])
    _asset(cli,readonly=False)
    require(started['version']=='warehouse-v642-calibration-recovery-supervisor.v1'
        and started['observed_by']=='source_bound_subprocess_launch' and started['recovery_plan']==recovery_plan_ref
        and started['supervisor_source']==cli and started['command'][1:]==[cli['path'],'--run','--recovery-plan',recovery_plan_ref['path']]
        and type(started['supervisor_pid']) is int and started['supervisor_pid']>1,
        'supervisor source/argv/plan is not the registered v2 launch')
    for name,sha in p['source_files'].items():
        require(file_sha(ROOT/name)==sha,'frozen six recovery files changed')
        snap=folder/'source'/name;require(file_sha(snap)==sha and not snap.stat().st_mode&0o222,'v2 source snapshot changed')
    for name,sha in p['original_source_files'].items():require(file_sha(ROOT/name)==sha,'original 407 source changed')
    for key in ('original_plan','original_claim','original_descriptor','original_controller_contract'):_asset(p[key])
    expected=(Path(p['original_calibration'])/'regularized/checkpoints/016800/owner-state.json' if mode=='original-safe-replay'
        else folder/'regularized/checkpoints/018600/owner-state.json')
    require(Path(target_checkpoint_path)==expected,'only the exact registered safe checkpoint path is allowed')
    if not terminal:
        require(datetime.now(timezone.utc)<datetime.fromisoformat(p['expires_utc']),'original finite lease expired; no cutover renewal')
        require(not (folder/'supervisor-exit.json').exists(),'v2 already has an observed terminal exit')
    return dict(recovery_plan=recovery_plan_ref,supervisor_started=supervisor_started_ref,recovery_claim=claim_ref,
        worker_pid=worker_pid,supervisor_pid=started['supervisor_pid'],command=started['command'],target_checkpoint=str(expected),
        mode=mode,expires_utc=p['expires_utc'],frozen_v2_source_files=p['source_files'],original_source_files=p['original_source_files'])


def prepare(*,recovery_plan_ref,supervisor_started_ref,worker_pid,target_checkpoint_path,mode,cutover_id):
    binding=_binding(recovery_plan_ref,supervisor_started_ref,worker_pid,target_checkpoint_path,mode)
    require(type(cutover_id) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',cutover_id),'unsafe cutover ID')
    process=_process(worker_pid)
    require(process['ppid']==binding['supervisor_pid'] and process['command']==' '.join(binding['command']),
        'actual PID/parent/argv differs from v2 launch')
    folder=OUTPUT/cutover_id;folder.mkdir(parents=True,exist_ok=False);sources=source_files()
    for name,sha in sources.items():
        target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle:handle.write((ROOT/name).read_bytes());handle.flush();os.fsync(handle.fileno())
        target.chmod(0o444);require(file_sha(target)==sha,'cutover source snapshot changed')
    return _put(folder/'plan.json',dict(version=VERSION,source_files=sources,binding=binding,
        authorization=v2.AUTHORIZATION,default_action='assert_only_no_signals',training_authority=False,
        model_acceptance_passed=False,pilot_opened=False,automatic_execution=False))


def checked_plan(ref,*,terminal=False):
    _asset(ref);p=bound(ref)
    require(p['version']==VERSION and p['source_files']==source_files() and p['authorization']==v2.AUTHORIZATION
        and p['training_authority'] is p['model_acceptance_passed'] is p['pilot_opened'] is False,
        'cutover source, authorization or scope differs')
    for name,sha in p['source_files'].items():_asset(dict(path=str(Path(ref['path']).parent/'source'/name),sha256=sha))
    b=p['binding'];require(b==_binding(b['recovery_plan'],b['supervisor_started'],b['worker_pid'],b['target_checkpoint'],b['mode'],terminal=terminal),
        'cutover pinned identity changed')
    return p


def _saved_state(snapshot,*,step,provider_version,provider_calls,installs):
    ref=snapshot['training_state'];_asset(ref);raw=Path(ref['path']).read_bytes();state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(state['schema']==CHECKPOINT_SCHEMA and state['model_version']==MODEL_VERSION
        and state['joint_steps']==step and all(type(state[k]) is int and state[k]>=0 for k in ('actor_updates','critic_updates'))
        and isinstance(state['rng'],dict) and set(state['rng'])=={'python','numpy','torch_cpu'}
        and isinstance(state['extra_rng_states'],dict) and isinstance(state['requires_grad'],dict)
        and all(type(v) is bool for v in state['requires_grad'].values()),'incomplete full NN/optimizer/RNG checkpoint')
    # A read-only tensor carrier for the original bitwise hash, not a model,
    # TrainingState, RNG constructor or ObservedPool capability.
    carrier=SimpleNamespace(**{k:state[k] for k in ('reference_path','reference_sha256','initialization_seed','beta','joint_steps','actor_updates','critic_updates')})
    carrier.model=SimpleNamespace(state_dict=lambda:state['model'],named_parameters=lambda:[(k,SimpleNamespace(requires_grad=v)) for k,v in state['requires_grad'].items()])
    carrier.actor_optimizer=SimpleNamespace(state_dict=lambda:state['actor_optimizer'])
    carrier.critic_optimizer=SimpleNamespace(state_dict=lambda:state['critic_optimizer'])
    fingerprint=training_fingerprint(carrier);core=snapshot['core'];reader=snapshot['reader'];rp=reader['payload'];public=rp['public']['payload']
    require(core['processed_joint_steps']==snapshot['reserved_joint_steps']==state['joint_steps']==step
        and core['safe_batch_boundary'] is True and core['failed'] is False
        and core['actor_updates']==state['actor_updates'] and core['critic_updates']==state['critic_updates'] and core['beta']==state['beta']
        and isinstance(core['rng_state'],dict) and core['rng_state']['bit_generator']=='PCG64'
        and len(core['seen_batch_receipts'])==core['update_sequence'] and core['variant']=='regularized'
        and core['reference_sha256']==state['reference_sha256'] and isinstance(core['risk'],dict)
        and isinstance(core['seen_complete_episodes'],list) and isinstance(core['accepted_program_updates'],dict)
        and core['source_sha256']==file_sha(ROOT/'backend/training/warehouse_v629_ppo_update.py'),
        'unsafe or incomplete optimizer/core checkpoint')
    require(reader['payload_sha256']==digest(rp) and rp['public']['payload_sha256']==digest(public)
        and public['processed_joint_steps']==step and public['safe_assembly_boundary'] is True
        and public['full_training_state_sha256']==fingerprint and len(rp['environments'])==len(rp['active'])==8
        and public['owner']==rp['owner']==snapshot['owner']
        and len(snapshot['environments'])==len(snapshot['active_cases'])==len(snapshot['reset_cursors'])==8
        and snapshot['program_source_cursor']==1650 and snapshot['refits']==installs,
        'unsafe or incomplete physical reader/cursor checkpoint')
    for env,case in zip(snapshot['environments'],snapshot['active_cases'],strict=True):
        require((env is None)==(case is None),'owner environment/active-case pairing differs')
        if env is not None:require(env['payload_sha256']==digest(env['payload']),'owner private environment/RNG hash differs')
    provider=snapshot['provider'];pp=provider['payload']
    require(provider['payload_sha256']==digest(pp) and pp['version']==provider_version and pp['full_training_state_sha256']==fingerprint
        and pp['owner']==snapshot['owner']
        and pp['call_count']==provider_calls and [i['step'] for i in pp['installed']]==installs
        and pp['config']==core['config'] and pp['variant']=='regularized' and type(pp['latch']) is bool
        and len(pp['used_epochs'])==provider_calls and len(set(pp['used_epochs']))==provider_calls,
        'provider control/NN fingerprint does not match the full safe checkpoint')
    for key in ('events','installed','programs'):
        require(key in pp,'provider snapshot omitted durable control state')
    return dict(training_state=ref,full_training_state_sha256=fingerprint,step=step,core_rng_sha256=digest(core['rng_state']),
        provider_calls=provider_calls,provider_control_sha256=digest(pp),restores_full_RNG_without_new_draws=True)


def assert_safe(plan_ref,*,_terminal=False):
    p=checked_plan(plan_ref,terminal=_terminal);b=p['binding'];old=bound(b['recovery_plan']);folder=Path(b['recovery_plan']['path']).parent
    if not _terminal:
        process=_process(b['worker_pid']);require(process['ppid']==b['supervisor_pid'] and process['command']==' '.join(b['command']),
            'actual worker PID/parent/argv changed')
    journal=folder/'owner-journal.jsonl';resets=folder/'actual-resets.jsonl';events=_lines(journal);reset_rows=_lines(resets)
    require(events and not any((folder/name).exists() for name in ('failed.json','result.json','completed.json')),
        'v2 terminated or lacks an auditable owner journal')
    original_events=[row for ref in old['journals'] for row in _lines(ref['path'])]
    original_resets=[row for ref in old['reset_journals'] for row in _lines(ref['path'])]
    require([e['sequence'] for e in events]==list(range(len(original_events),len(original_events)+len(events)))
        and all(e['version']==v2.VERSION and e['claim']==old['original_claim'] and e['recovery_claim']==b['recovery_claim'] for e in events),
        'v2 journal does not extend the original immutable claim prefix')
    checkpoint=pointer(b['target_checkpoint']);_asset(checkpoint);snapshot=bound(checkpoint)
    checkpoint_sources=v2.old.source_files()
    require(snapshot['claim']==old['original_claim'] and snapshot['owner']==old['owners']['regularized']
        and snapshot['source_files']==checkpoint_sources
        and all(old['original_source_files'].get(k)==v for k,v in checkpoint_sources.items()),
        'checkpoint belongs to another owner/source')
    consumption=v2.check_consumption(original_events+events,original_resets+reset_rows,
        bound(bound(old['original_descriptor'])['train_case_inventory']),
        bound(bound(old['original_descriptor'])['program_evidence_inventory']))
    if b['mode']=='original-safe-replay':
        require(checkpoint==old['checkpoints']['regularized'],'original safe checkpoint SHA differs from its pinned plan')
        expected=['explicit_recovery_claimed','full_training_checkpoint_restored','full_training_checkpoint_restored',
            'cached_rollout_replayed','optimizer_update_predebit','optimizer_update_accepted','cached_rollout_replayed','optimizer_update_predebit']
        require([e['kind'] for e in events]==expected and reset_rows==[],'original-safe mode has fresh physical work or another committed journal suffix')
        require(consumption==old['consumption'],'original safe-replay cursor history differs from the pinned old reset inventory')
        accepted=events[5];cached=old['cached_rollouts']['regularized'];require(len(cached)==2,'unknown cached replay inventory')
        _asset(accepted['receipt']);_asset(cached[0]['update']);_asset(cached[1]['trace']);_asset(cached[1]['audit'])
        require(accepted['variant']=='regularized' and accepted['step']==17824 and accepted['replay_of_existing_commit'] is True
            and accepted['receipt']['path']==str(folder/'regularized/rollouts/016800/update.json')
            and v2.exact_value_equal(bound(accepted['receipt']),bound(cached[0]['update'])),
            'only the exact already committed 17824 replay may have been accepted')
        require(events[-1]['variant']=='regularized' and events[-1]['step']==18600 and events[-1]['trace']==cached[1]['trace']
            and events[-2]['trace']==cached[1]['trace'] and events[-2]['audit']==cached[1]['audit']
            and events[-2]['original_update'] is None and events[-2]['new_environment_resets']==0,
            'pending update is not the original cached 776 physical frames')
        trace=bound(cached[1]['trace']);require(trace['start_joint_step']==17824 and trace['end_joint_step']==18600 and len(trace['events'])==776,
            'cached pending physical prefix differs')
        require(not (folder/'regularized/rollouts/017824').exists()
            and not (folder/'regularized/checkpoints').exists() and not (folder/'regularized/refit-targets').exists()
            and not (folder/'raw').exists(),'new committed checkpoint/optimizer/refit/physical suffix must not be discarded')
        facts=_saved_state(snapshot,step=16800,provider_version='warehouse-v642-current-NN-provider.v1',provider_calls=32,installs=[15000,16800])
        require(all(a<=c for a,c in zip(snapshot['reset_cursors'],consumption['training_cursors']['regularized'],strict=True)),
            'original checkpoint cursor exceeds already recorded physical reset prefix')
        discarded='only_uncommitted_numerical_PPO_computation_no_new_physical_samples_or_commits'
    else:
        last=events[-1]
        require(last['kind']=='checkpoint' and last['variant']=='regularized' and last['snapshot']==checkpoint,
            'post-checkpoint event/predebit appeared; resume and refuse rather than discard a tail')
        require(snapshot['version']==v2.VERSION and snapshot['recovery_claim']==b['recovery_claim']
            and snapshot['recovery_source_files']==old['source_files'],'checkpoint lacks exact v2 continuation source/claim')
        facts=_saved_state(snapshot,step=18600,provider_version='warehouse-v642-explicit-evidence-recovery.v1',provider_calls=64,installs=[15000,16800,18600])
        require(snapshot['reset_cursors']==consumption['training_cursors']['regularized'],'checkpoint reset cursor differs from committed physical history')
        discarded='none'
    logs={name:pointer(folder/name) for name in ('owner-journal.jsonl','actual-resets.jsonl','run.log')}
    return dict(version=VERSION,plan=plan_ref,binding=b,checkpoint=checkpoint,checkpoint_facts=facts,
        logs=logs,journal_last_sequence=events[-1]['sequence'],reset_consumption=consumption,
        discarded_work=discarded,no_new_seed_or_environment_authority=True,
        model_acceptance_passed=False,pilot_opened=False)


def _successor(ref):
    _asset(ref);value=bound(ref)
    require(value['version']=='warehouse-v642-accelerated-recovery-source-contract.v1' and value['source_files']
        and value['training_authority'] is False
        and value['source_sha256']==digest(value['source_files']) and set(value['source_snapshots'])==set(value['source_files']),
        'successor requires a concrete complete source contract, not a ready/passed flag')
    for name,sha in value['source_files'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and file_sha(ROOT/name)==sha,'successor current source differs')
        snap=value['source_snapshots'][name];_asset(snap);require(snap['sha256']==sha,'successor source snapshot differs')
    return value


def execute(plan_ref,*,authorization,successor_source_contract_ref,worker_pid,target_checkpoint_path,supervisor_started_ref,poll_seconds=.025):
    require(authorization==v2.AUTHORIZATION,'explicit execute authorization is required; preparation never signals')
    p=checked_plan(plan_ref);b=p['binding']
    require(worker_pid==b['worker_pid'] and str(Path(target_checkpoint_path))==b['target_checkpoint']
        and supervisor_started_ref==b['supervisor_started'],'explicit execute PID/checkpoint/supervisor differs')
    require(type(poll_seconds) in (int,float) and .001<=poll_seconds<=1.,'bounded polling interval required')
    _successor(successor_source_contract_ref);out=Path(plan_ref['path']).parent
    _put(out/'execution-started.json',dict(version=VERSION,plan=plan_ref,authorization=authorization,
        successor_source_contract=successor_source_contract_ref,started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    stopped=False
    try:
        folder=Path(b['recovery_plan']['path']).parent
        while True:
            require(datetime.now(timezone.utc)<datetime.fromisoformat(b['expires_utc']),'original cutover deadline expired')
            if b['mode']=='original-safe-replay':break
            events=_lines(folder/'owner-journal.jsonl')
            if any(e['kind']=='checkpoint' and e.get('variant')=='regularized' and e.get('snapshot',{}).get('path')==b['target_checkpoint'] for e in events):break
            require(not (folder/'supervisor-exit.json').exists(),'worker exited before the target checkpoint')
            time.sleep(poll_seconds)
        process=_process(worker_pid)
        require(process['ppid']==b['supervisor_pid'] and process['command']==' '.join(b['command']),'worker identity changed before stop')
        os.kill(worker_pid,signal.SIGSTOP);stopped=True
        # SIGSTOP delivery can race with the last write. Require actual stop
        # before inspecting the entire immutable numerical boundary.
        until=time.monotonic()+5
        while 'T' not in _process(worker_pid)['status']:
            require(time.monotonic()<until,'worker did not enter stopped state');time.sleep(.005)
        safe=assert_safe(plan_ref);_successor(successor_source_contract_ref)
        safe['stopped_utc']=datetime.now(timezone.utc).isoformat();safe['successor_source_contract']=successor_source_contract_ref
        pre=_put(out/'prestop-verified.json',safe)
        # The pinned old executable installs no SIGTERM handler. The actual
        # source-bound OS wait(-15), not this assumption, is the terminal gate.
        os.kill(worker_pid,signal.SIGTERM);os.kill(worker_pid,signal.SIGCONT);stopped=False
        exit_path=folder/'supervisor-exit.json'
        while not exit_path.exists():
            require(datetime.now(timezone.utc)<datetime.fromisoformat(b['expires_utc']),'no observed supervisor exit before finite deadline')
            time.sleep(poll_seconds)
        exited=pointer(exit_path);_asset(exited);e=bound(exited);started=bound(b['supervisor_started'])
        require(e['version']==started['version'] and e['observed_by']=='source_bound_subprocess_wait'
            and type(e['exit_code']) is int and e['exit_code']==-signal.SIGTERM
            and e['started']==b['supervisor_started'] and e['recovery_plan']==b['recovery_plan']
            and e['worker_pid']==worker_pid and e['supervisor_pid']==b['supervisor_pid']
            and e['command']==b['command'] and e['supervisor_source']==started['supervisor_source']
            and 'result' not in e and 'completed' not in e,'cutover requires the original supervisor actual OS wait(-15)')
        require(e['log']==safe['logs']['run.log'] and all(pointer(ref['path'])==ref for ref in safe['logs'].values())
            and pointer(safe['checkpoint']['path'])==safe['checkpoint'],'worker wrote a suffix after the verified stopped boundary')
        require(datetime.fromisoformat(safe['stopped_utc'])<=datetime.fromisoformat(e['finished_utc'])<datetime.fromisoformat(b['expires_utc']),
            'actual stopped/wait chronology exceeds the original finite lease')
        # The dead worker cannot seal its two open journals on default TERM.
        # Freeze permissions only after OS wait and exact byte recheck. Old
        # content, claims, paths and all SHA identities remain unchanged.
        for ref in safe['logs'].values():Path(ref['path']).chmod(0o444)
        return _put(out/'cutover-completed.json',dict(version=VERSION,plan=plan_ref,prestop=pre,actual_supervisor_exit=exited,
            stopped_for_engineering_continuation=True,actual_exit_code=-signal.SIGTERM,
            stopped_logs_sealed_by_mode_only=True,
            original_files_preserved=True,original_claim_success_written=False,training_completed=False,
            scientific_success_claimed=False,scientific_failure_claimed=False,pilot_opened=False))
    except BaseException as error:
        if stopped:
            os.kill(worker_pid,signal.SIGCONT);stopped=False
        _put(out/'cutover-refused.json',dict(version=VERSION,plan=plan_ref,error_type=type(error).__name__,error=str(error),
            stopped_worker_resumed_if_alive=True,automatic_retry=False,training_completed=False,pilot_opened=False))
        raise


def read_completed(ref):
    """Offline source-bound handoff proof; never infers success from a dead PID."""
    _asset(ref);done=bound(ref)
    require(done['version']==VERSION and done['stopped_for_engineering_continuation'] is True
        and done['actual_exit_code']==-signal.SIGTERM and done['original_files_preserved'] is True
        and done['original_claim_success_written'] is done['training_completed'] is done['scientific_success_claimed']
            is done['scientific_failure_claimed'] is done['pilot_opened'] is False,
        'cutover receipt claims training/scientific success or lacks an actual engineering stop')
    _asset(done['plan']);p=checked_plan(done['plan'],terminal=True);b=p['binding'];folder=Path(done['plan']['path']).parent
    require(Path(ref['path'])==folder/'cutover-completed.json' and not (folder/'cutover-refused.json').exists(),
        'cutover receipt relocated or refused')
    _asset(done['prestop']);before=bound(done['prestop'])
    require(Path(done['prestop']['path'])==folder/'prestop-verified.json','prestop receipt relocated')
    replay=assert_safe(done['plan'],_terminal=True)
    require(v2.exact_value_equal({k:v for k,v in before.items() if k not in ('stopped_utc','successor_source_contract')},replay),
        'stopped numerical boundary/log prefix did not reproduce exactly')
    source_ref=before['successor_source_contract'];_successor(source_ref)
    _asset(done['actual_supervisor_exit']);exit_value=bound(done['actual_supervisor_exit']);launch=bound(b['supervisor_started'])
    require(Path(done['actual_supervisor_exit']['path'])==Path(b['recovery_plan']['path']).parent/'supervisor-exit.json'
        and exit_value['version']==launch['version'] and exit_value['observed_by']=='source_bound_subprocess_wait'
        and type(exit_value['exit_code']) is int and exit_value['exit_code']==-signal.SIGTERM
        and exit_value['started']==b['supervisor_started'] and exit_value['recovery_plan']==b['recovery_plan']
        and exit_value['worker_pid']==b['worker_pid'] and exit_value['supervisor_pid']==b['supervisor_pid']
        and exit_value['command']==b['command'] and exit_value['supervisor_source']==launch['supervisor_source']
        and exit_value['log']==before['logs']['run.log'] and 'result' not in exit_value and 'completed' not in exit_value,
        'original supervisor terminal evidence is not the actual source-bound wait(-15)')
    times=[datetime.fromisoformat(x) for x in (launch['started_utc'],before['stopped_utc'],exit_value['finished_utc'])]
    require(all(t.tzinfo for t in times) and times==sorted(times)
        and times[-1]<datetime.fromisoformat(b['expires_utc']),'cutover launch/stop/exit times exceed the original lease')
    inputs=[ref,done['plan'],done['prestop'],source_ref,b['recovery_plan'],b['recovery_claim'],b['supervisor_started'],
        done['actual_supervisor_exit'],before['checkpoint'],before['checkpoint_facts']['training_state'],*before['logs'].values()]
    for item in inputs:_asset(item)
    return dict(version=VERSION,parent_v2_plan=b['recovery_plan'],parent_v2_claim=b['recovery_claim'],
        actual_supervisor_exit=done['actual_supervisor_exit'],source_contract=source_ref,inputs=inputs,
        mode=b['mode'],checkpoint=before['checkpoint'],checkpoint_facts=before['checkpoint_facts'],
        discarded_work=before['discarded_work'],new_training_authority=False,model_acceptance_passed=False,pilot_opened=False)
