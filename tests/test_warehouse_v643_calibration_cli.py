"""CLI contract tests; no candidate plan, seed, model or environment is used."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest

from experiments.warehouse import v643_calibration_pipeline as p


def cli():
    spec=importlib.util.spec_from_file_location('v643_explicit_cli',
        Path(__file__).resolve().parents[1]/'scripts/run_warehouse_v643_calibration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('argv',[[],['--plan','unused'],
    ['--reference-directory','unused','--plan','unused','--plan-sha','bad'],
    ['--reference-directory','unused','--plan','unused','--plan-sha','a'*64,'--resume','unused']])
def test_malformed_or_implicit_cli_never_imports_scientific_pipeline(argv,monkeypatch):
    c=cli()
    def forbidden():raise AssertionError('invalid CLI must not load a scientific module')
    monkeypatch.setattr(c,'_pipeline',forbidden)
    with pytest.raises(SystemExit) as error:c.main(argv)
    assert error.value.code==2


@pytest.mark.parametrize('passed,code',[(True,0),(False,1),(None,1)])
def test_exit_status_requires_actual_measured_precheck_not_training_completion(tmp_path,monkeypatch,passed,code):
    c=cli();calls=[]
    pipeline=SimpleNamespace(run=lambda **kw:(calls.append(kw) or {'synthetic_result':True}),
        bound=lambda _:dict(training_completed=True,calibration_precheck_passed=passed,
            candidate_closed_after_failed_calibration=passed is False,model_acceptance_passed=False,pilot_opened=False))
    monkeypatch.setattr(c,'_pipeline',lambda:pipeline)
    assert c.main(['--reference-directory',str(tmp_path),'--plan',str(tmp_path/'plan.json'),'--plan-sha','a'*64])==code
    assert len(calls)==1 and calls[0]['resume_ref'] is None


def test_resume_passes_both_explicit_hashes_and_keeps_same_plan(tmp_path,monkeypatch):
    c=cli();calls=[]
    monkeypatch.setattr(c,'_pipeline',lambda:SimpleNamespace(
        run=lambda **kw:(calls.append(kw) or {}),bound=lambda _:dict(calibration_precheck_passed=False,
            candidate_closed_after_failed_calibration=True,model_acceptance_passed=False,pilot_opened=False)))
    assert c.main(['--reference-directory',str(tmp_path),'--plan',str(tmp_path/'plan.json'),'--plan-sha','a'*64,
                  '--resume',str(tmp_path/'boundary.json'),'--resume-sha','b'*64])==1
    assert calls[0]['plan_ref']['sha256']=='a'*64 and calls[0]['resume_ref']['sha256']=='b'*64


def test_numeric_threads_are_set_before_scientific_import(tmp_path,monkeypatch):
    import os
    c=cli();seen=[]
    for name in c.THREAD_VARIABLES:monkeypatch.setenv(name,'99')
    def load():
        seen.append({name:os.environ.get(name) for name in c.THREAD_VARIABLES})
        return SimpleNamespace(run=lambda **_: {},bound=lambda _:dict(calibration_precheck_passed=False,
            candidate_closed_after_failed_calibration=True,model_acceptance_passed=False,pilot_opened=False))
    monkeypatch.setattr(c,'_pipeline',load)
    assert c.main(['--reference-directory',str(tmp_path),'--plan',str(tmp_path/'plan.json'),'--plan-sha','a'*64])==1
    assert seen==[{name:'1' for name in c.THREAD_VARIABLES}]


def test_old_or_missing_plan_refuses_before_rng_reset_or_private_execution(tmp_path,monkeypatch):
    def forbidden(*_a,**_k):raise AssertionError('invalid public plan must not initialize anything')
    monkeypatch.setattr(p,'_execute_reviewed_plan',forbidden)
    monkeypatch.setattr(p,'_continue_reviewed_pair',forbidden)
    monkeypatch.setattr(p,'Environment',forbidden)
    for plan in (None,{},dict(path=str(tmp_path/'absent.json'),sha256='a'*64)):
        with pytest.raises((RuntimeError,KeyError,FileNotFoundError)):
            p.run(reference_directory=tmp_path,plan_ref=plan)
    assert not list(tmp_path.iterdir())
