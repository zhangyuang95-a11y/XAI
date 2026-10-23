"""Actual-wait protocol tests with fake subprocesses; no scientific execution."""
from types import SimpleNamespace
import json
import pytest
from experiments.warehouse import v642_long_stage_supervisor as s


@pytest.fixture
def child(tmp_path,monkeypatch):
    out=tmp_path/'synthetic';command=['fixed-synthetic-child']
    module=SimpleNamespace(source_files=lambda:{'test':'not model evidence'})
    monkeypatch.setattr(s,'invocation',lambda stage,args:(module,command,out))
    def spawn(argv,**kwargs):
        assert argv==command and 'shell' not in kwargs
        out.mkdir();(out/'result.json').write_text('{}')
        (out/'completed.json').write_text(json.dumps(dict(result=s.pointer(out/'result.json'))))
        for path in out.iterdir():path.chmod(0o444)
        return SimpleNamespace(pid=123456,wait=lambda:0)
    monkeypatch.setattr(s.subprocess,'Popen',spawn)
    return out


def test_actual_wait_and_immutable_source_receipt(child):
    result=s.supervise('synthetic',{})
    refs=s.verify_execution(s.pointer(child/'supervisor-exit.json'),s.pointer(child/'result.json'),stage='synthetic')
    assert result['exit_code']==0 and len(refs)==6 and result['scientific_acceptance'] is False
    with pytest.raises(RuntimeError,match='already has output'):s.supervise('synthetic',{})


def test_unknown_stage_cannot_provide_arbitrary_command():
    with pytest.raises(RuntimeError):s.invocation('arbitrary-shell',{})


def test_missing_completion_preserves_real_zero_exit_as_failure(child,monkeypatch):
    def spawn(*args,**kwargs):
        child.mkdir();return SimpleNamespace(pid=123456,wait=lambda:0)
    monkeypatch.setattr(s.subprocess,'Popen',spawn)
    with pytest.raises((RuntimeError,FileNotFoundError)):s.supervise('synthetic',{})
    failure=s.bound(s.pointer(child.with_name(child.name+'.supervision')/'supervisor-failed.json'))
    assert failure['observed_child_exit']['exit_code']==0
    assert not (child/'supervisor-exit.json').exists()


@pytest.mark.parametrize('field,value',[('exit_code',True),('worker_pid',0),('scientific_acceptance',True),
    ('observed_by','file_exists'),('stage','unknown')])
def test_resealed_wrong_metadata_rejected(child,field,value):
    receipt=s.supervise('synthetic',{});receipt[field]=value
    for path in (child/'supervisor-exit.json',child.with_name(child.name+'.supervision')/'exit.json'):
        path.chmod(0o644);path.write_text(json.dumps(receipt));path.chmod(0o444)
    with pytest.raises(RuntimeError):s.verify_execution(s.pointer(child/'supervisor-exit.json'),
        s.pointer(child/'result.json'),stage='synthetic')
