"""Software-only supervision tests: no policy, scenes or scientific authority."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from experiments.warehouse import v642_stage_supervisor as s


@pytest.fixture
def child(tmp_path,monkeypatch):
    output=tmp_path/'stage'
    args={'synthetic':'no scientific operation'}
    command=['synthetic-fixed-child']
    monkeypatch.setattr(s,'invocation',lambda stage,arguments:(command,output))
    monkeypatch.setattr(s,'sources',lambda:{'synthetic-source':'fixture'})
    def spawn(argv,**kwargs):
        assert argv==command and kwargs['stdin']==s.subprocess.DEVNULL
        assert 'shell' not in kwargs and kwargs['env']['OMP_NUM_THREADS']=='1'
        output.mkdir()
        (output/'result.json').write_text('{"synthetic":true}')
        (output/'completed.json').write_text(json.dumps(dict(result=s.pointer(output/'result.json'))))
        (output/'result.json').chmod(0o444)
        (output/'completed.json').chmod(0o444)
        return SimpleNamespace(pid=987654,wait=lambda:0)
    monkeypatch.setattr(s.subprocess,'Popen',spawn)
    return output,args


def test_actual_wait_receipt_is_bound_to_fixed_child_and_can_be_read(child):
    output,args=child
    receipt=s.supervise('synthetic',args)
    assert receipt['observed_by']=='source_bound_subprocess_wait'
    assert receipt['exit_code']==0 and receipt['scientific_acceptance'] is False
    refs=s.verify_execution(s.pointer(output/'supervisor-exit.json'),s.pointer(output/'result.json'),stage='synthetic')
    assert len(refs)==5


def test_single_use_supervision_does_not_retry_existing_output(child):
    output,args=child;s.supervise('synthetic',args)
    with pytest.raises(RuntimeError,match='already exists'):s.supervise('synthetic',args)


def test_spawn_failure_does_not_fabricate_exit_or_success(child,monkeypatch):
    output,args=child
    def fail(*a,**kw):raise OSError('synthetic spawn failure')
    monkeypatch.setattr(s.subprocess,'Popen',fail)
    with pytest.raises(OSError):s.supervise('synthetic',args)
    assert not (output/'supervisor-exit.json').exists()
    supervision=output.with_name(output.name+'.supervision')
    assert (supervision/'supervisor-failed.json').exists() and not (supervision/'exit.json').exists()


def test_nonzero_wait_is_preserved_without_completed_success(child,monkeypatch):
    output,args=child
    def spawn(*a,**kw):return SimpleNamespace(pid=987654,wait=lambda:7)
    monkeypatch.setattr(s.subprocess,'Popen',spawn)
    result=s.supervise('synthetic',args)
    assert result['exit_code']==7 and 'result' not in result
    assert not output.exists()


@pytest.mark.parametrize('field,value',[
    ('exit_code',True),('exit_code',1),('observed_by','PID_disappeared'),
    ('version','unknown'),('worker_pid',0),('scientific_acceptance',True)])
def test_resealed_wrong_execution_metadata_fails_closed(child,field,value):
    output,args=child;receipt=s.supervise('synthetic',args)
    receipt[field]=value
    for p in (output/'supervisor-exit.json',output.with_name(output.name+'.supervision')/'exit.json'):
        p.chmod(0o644);p.write_text(json.dumps(receipt));p.chmod(0o444)
    with pytest.raises(RuntimeError):s.verify_execution(s.pointer(output/'supervisor-exit.json'),
        s.pointer(output/'result.json'),stage='synthetic')


def test_changed_log_or_writable_started_receipt_is_rejected(child):
    output,args=child;receipt=s.supervise('synthetic',args)
    p=Path(receipt['started']['path']);p.chmod(0o644)
    with pytest.raises(RuntimeError,match='writable'):s.verify_execution(s.pointer(output/'supervisor-exit.json'),
        s.pointer(output/'result.json'),stage='synthetic')


def test_arbitrary_command_stage_is_not_exposed():
    with pytest.raises(RuntimeError,match='unknown post-calibration stage'):s.invocation('shell',{'cmd':'anything'})
