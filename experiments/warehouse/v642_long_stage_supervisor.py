"""Actual OS supervision for long-training final programs and quality stages.

Fixed entry points only. Child validation and claims remain authoritative;
supervision itself never authorizes training, scenes, acceptance or release.
"""
from datetime import datetime,timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import os,subprocess,sys
from .v626_protocol import require,pointer,bound,file_sha
from scripts.run_warehouse_v625 import write_new

ROOT=Path(__file__).resolve().parents[2]
VERSION='warehouse-v642-long-post-training-supervisor.v1'
STAGES={'long-final-programs':('v642_long_training_final_programs','finalize_warehouse_v642_long_programs.py'),
        'long-quality':('v642_long_training_quality','run_warehouse_v642_long_quality.py')}


def invocation(stage,arguments):
    require(stage in STAGES and type(arguments) is dict,'unknown long stage or arguments')
    name,script=STAGES[stage];module=import_module('experiments.warehouse.'+name)
    if stage=='long-final-programs':
        require(set(arguments)=={'completed_assets','reference_directory'},'unknown long final-program arguments')
        require(pointer(arguments['completed_assets']['path'])==arguments['completed_assets'], 'long assets changed')
        report=bound(arguments['completed_assets']);plan=bound(report['seed_plan'])
        output=module.OUTPUT/module.identity(SimpleNamespace(value=plan))
        reference=str(Path(arguments['reference_directory']).resolve())
        require(reference==arguments['reference_directory'],'reference directory must be absolute')
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--completed-assets',
                 arguments['completed_assets']['path'],'--reference-directory',reference]
    else:
        require(set(arguments)=={'plan'} and pointer(arguments['plan']['path'])==arguments['plan'], 'long quality plan changed')
        path=Path(arguments['plan']['path']).resolve()
        require(path.name=='plan.json' and path.parent.is_relative_to(module.OUTPUT.resolve()),
                'long quality plan is outside its finite stage root')
        output=path.parent/'collection'
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--collect','--plan',str(path)]
    return module,command,output.resolve()


def readonly(path,value):
    write_new(path,value);path.chmod(0o444)


def supervise(stage,arguments):
    module,command,output=invocation(stage,arguments)
    require(not output.exists(),'long stage already has output; no automatic retry')
    supervision=output.with_name(output.name+'.supervision');supervision.mkdir(parents=True,exist_ok=False)
    started=dict(version=VERSION,stage=stage,arguments=arguments,command=command,output=str(output),
        source_files=module.source_files(),supervisor_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        scientific_authority=False,pilot_opened=False,automatic_retry=False)
    readonly(supervision/'started.json',started);log=supervision/'run.log'
    env=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    receipt=None
    try:
        with log.open('xb') as handle:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env)
            code=child.wait()
        log.chmod(0o444)
        receipt=dict(version=VERSION,stage=stage,observed_by='source_bound_subprocess_wait',exit_code=code,
            worker_pid=child.pid,supervisor_pid=os.getpid(),started=pointer(supervision/'started.json'),log=pointer(log),
            command=command,finished_utc=datetime.now(timezone.utc).isoformat(),scientific_acceptance=False,pilot_opened=False)
        if code==0:
            result=pointer(output/'result.json')
            require(bound(pointer(output/'completed.json'))==dict(result=result)
                    and not (output/'failed.json').exists(),'long child exited zero without a completion seal')
            receipt.update(result=result,completed=pointer(output/'completed.json'))
        readonly(supervision/'exit.json',receipt)
        if output.is_dir():readonly(output/'supervisor-exit.json',receipt)
        return receipt
    except BaseException as error:
        # Preserve an actual observed code even when the worker's completion
        # metadata is missing. Such an exit cannot pass verify_execution.
        failure=dict(version=VERSION,started=pointer(supervision/'started.json'),error_type=type(error).__name__,
            error=str(error),automatic_retry=False,pilot_opened=False)
        if receipt is not None:failure['observed_child_exit']=receipt
        readonly(supervision/'supervisor-failed.json',failure)
        raise


def verify_execution(execution_ref,result_ref,*,stage):
    e=bound(execution_ref)
    require(e.get('version')==VERSION and e.get('stage')==stage
        and e.get('observed_by')=='source_bound_subprocess_wait'
        and type(e.get('exit_code')) is int and e['exit_code']==0 and e.get('result')==result_ref,
        'long post-training child did not actually complete successfully')
    started=bound(e['started']);module,command,output=invocation(stage,started['arguments'])
    supervision=output.with_name(output.name+'.supervision')
    require(started['version']==VERSION and started['stage']==stage and started['output']==str(output)
        and started['command']==e['command']==command and started['source_files']==module.source_files()
        and type(e['worker_pid']) is int and e['worker_pid']>0
        and type(e['supervisor_pid']) is int and e['supervisor_pid']>0 and e['supervisor_pid']==started['supervisor_pid']
        and started['scientific_authority'] is started['pilot_opened'] is started['automatic_retry'] is False
        and e['scientific_acceptance'] is e['pilot_opened'] is False
        and execution_ref==pointer(output/'supervisor-exit.json') and e['started']==pointer(supervision/'started.json')
        and e['log']==pointer(supervision/'run.log') and e['completed']==pointer(output/'completed.json')
        and result_ref==pointer(output/'result.json') and bound(e['completed'])==dict(result=result_ref)
        and bound(pointer(supervision/'exit.json'))==e
        and not (output/'failed.json').exists() and not (supervision/'supervisor-failed.json').exists(),
        'long post-training source, command, identity or completion differs')
    before=datetime.fromisoformat(started['started_utc']);after=datetime.fromisoformat(e['finished_utc'])
    require(before.tzinfo and after.tzinfo and before<=after,'long supervision timestamps differ')
    refs=[execution_ref,result_ref,e['started'],e['log'],e['completed'],pointer(supervision/'exit.json')]
    for ref in refs:
        path=Path(ref['path'])
        require(not path.is_symlink() and not path.stat().st_mode&0o222 and file_sha(path)==ref['sha256'],
                'long post-training evidence is missing, writable or changed')
    return refs
