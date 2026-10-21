"""One synthetic unit through real writer/ledger/auditor, no science claim."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import pytest
from experiments.warehouse import v630_protocol as v
from backend.training import warehouse_v630_partition_search as search
from scripts import run_warehouse_v630 as runner,audit_warehouse_v630 as audit


def fixture_data():
    spec=importlib.util.spec_from_file_location('pipeline_search_fixture',Path(__file__).with_name('test_warehouse_v630_partition_search.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.data.__wrapped__()


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    folder=tmp_path_factory.mktemp('synthetic-v630-unit-not-scientific');source,arrays,cases,*_=fixture_data()
    with pytest.MonkeyPatch.context() as mp:
        output=folder/'output';output.mkdir();mp.setattr(v,'OUTPUT',output)
        old_path,old=v.previous.checked_plan();spec=dict(job_id='P-fold-0-robot_1',name='P',scope='protected',fold=0,role='robot_1')
        data_plan=folder/'synthetic-data-plan.json';data_plan.write_text(json.dumps(dict(scope='synthetic-unit-only')))
        previous=folder/'previous.json';previous.write_text(json.dumps(dict(previous_plan=v.pointer(data_plan))))
        plan=dict(jobs=[spec],actor=old['actor'],previous_plan=v.pointer(previous))
        plan_path=output/'plan.json';plan_path.write_text(json.dumps(plan));source_file=folder/'source.json';source_file.write_text(json.dumps(source.to_dict()))
        binding=dict(program=v.pointer(source_file),synthetic_targets_not_real_NN=True)
        completion=dict(synthetic_dependency_only=True);(output/'dependency-cleared.json').write_text(json.dumps(dict(completion=completion,sources={'0-robot_1':binding})))
        mp.setattr(v,'checked_plan',lambda:(plan_path,plan));mp.setattr(v,'old_completion',lambda:completion)
        mp.setattr(v,'source_program',lambda plan,fold,role:(deepcopy(source),binding))
        mp.setattr(v,'fit_data',lambda plan,fold,role:(arrays,cases,[]))
        outer={}
        for index,name in enumerate(('fit','calibration','eligibility','validation')):
            take=slice(index*150,(index+1)*150)
            outer[name]={k:value[take] for k,value in arrays.items()}
        mp.setattr(runner.data,'load_parts',lambda *args,**kwargs:outer)
        result=runner.run_job(spec['job_id']);assert result['status']=='completed',result
        job_folder=output/'jobs'/spec['job_id'];report=v.bound(result['search_report'])
        yield dict(source=source,outer=outer,plan=plan,result=result,folder=job_folder,report=report)


def test_actual_writer_ledger_and_per_unit_independent_audit_are_connected(synthetic):
    x=synthetic;result=x['result'];counts=audit.ledger(x['folder'],result)
    assert counts==dict(leaf=0,node=1,proof=0)
    replay=audit.audit_search(x['plan'],result,x['source'],x['report'])
    assert replay['local_tables_replayed'] and replay['fixed_additive_allocations_replayed']
    program=runner.search.Program.from_dict(v.bound(result['program']),x['source'].actor_role_sha256)
    audit.audited_program(program,x['outer'],result['metrics'])
    assert result['gate']==runner.gate(result['metrics']['validation'],x['report'],'protected')
    assert result['gate']['passed'] and result['NN_updates']==result['PPO_joint_steps']==result['environment_resets']==0
    assert v.bound(v.pointer(x['folder']/'completed.json'))==dict(result_sha256=v.file_sha(x['folder']/'result.json'))


def test_global_verifier_rejects_single_unit_as_eighteen_job_completion(synthetic):
    with pytest.raises(RuntimeError,match='all18 ordered'):audit.audit(synthetic['plan'],[synthetic['result']])


def test_tampered_option_table_is_not_accepted_as_an_audited_search(synthetic,tmp_path):
    x=synthetic;report=deepcopy(x['report']);tables=deepcopy(v.bound(report['option_tables']))
    group=next(iter(tables['groups']));tables['groups'][group][0]['score'][0]+=1
    path=tmp_path/'false-table.json';path.write_text(json.dumps(tables));report['option_tables']=v.pointer(path)
    with pytest.raises(RuntimeError,match='independent local tables'):audit.audit_search(x['plan'],x['result'],x['source'],report)
