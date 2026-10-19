"""Report formatting/formulas with JSON-only synthetic native boundaries.

No model is constructed/read, no environment exists, and no fit/gradient or
optimizer is invoked. Scientific truth still requires the real native checker.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts import report_warehouse_v621_bc_metrics as report

ROOT = Path(__file__).resolve().parents[1]
CASES = report.protocol.diagnostics.source_cases()


def pointer(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return pointer(path)


def model_result(i, recipe, *, quality_failure=False):
    control = recipe == 'additive_control'
    outcomes = [{'case': deepcopy(case), 'deliveries': 6 if control else 4 if quality_failure else 7,
        'robot_collisions': 2 if control else 1, 'shutdown': False} for case in CASES]
    if not control and not quality_failure:
        next(r for r in outcomes if r['case']['original_split']=='validation')['shutdown']=True
    selected=[r for r in outcomes if r['case']['original_split']=='validation']
    q=report._quality(selected)
    quality={k:q[k] for k in ('mean_deliveries','mean_collisions','shutdown_episode_rate')}
    roles={}
    for role in report.ROLES:
        metrics={}
        for purpose in ('feedback','explanation'):
            coverage,actual,expected=(.2,.1,.15) if control else ((.5,.4,.45) if purpose=='feedback' else (.6,.5,.55))
            metrics[purpose]={'sample_count':100,'rows':int(100*coverage),'coverage':coverage,
                'actionable_rows':int(100*actual),'actionable_coverage':actual,'expected_actionable_coverage':expected,
                'agreement':.90 if control else .98,'kl':.30 if control else .10,
                'whole_tree_raw_illegal_count':0,'whole_tree_path_uncertified_count':0}
        probe={'applicable':False,'reason':'diagnostic_control_not_new_candidate'} if control else {
            'passed':True,'optimizer_steps':0,'actual_PPO_joint_steps':0,'actor_parameters_unchanged':True,
            'program_coefficient':.01,'clone_virtual_frozen_at_step':150000,
            'variants':{'raw':{'weighted_loss':0.,'maximum_gradient_norm':0.,'feedback_rows':0},
                'regularized':{'weighted_loss':.0001,'maximum_gradient_norm':.002,'feedback_rows':40}}}
        evidence={'metrics':{'validation':metrics},'development_gate':{'passed':not control},'gradient_probe':probe}
        if not control:
            def measured(n,core,f,a,e):
                return {'physical_frames':n,'core_frames':core,'feedback_qualified_frames':f,
                    'feedback_coverage':f/n,'explanation_coverage':.6,'actual_feedback_coverage':a/n,
                    'expected_feedback_coverage':e/n,'mean_nn_top_probability':.9}
            evidence['validation_strata']={'nn_top_action':{a:measured(20,10,10,8,9) for a in report.ACTIONS},
                'task':{'1':measured(34,17,17,14,15.3),'2':measured(33,16,16,13,14.85),'3':measured(33,17,17,13,14.85)},
                'wait_share_of_explanation_qualified':.2,'purpose':'all_frame_descriptive_checks_not_extra_selection_gates'}
        roles[role]=evidence
    quality_passed=not quality_failure
    result={'initialization':i,'recipe':recipe,'status':'completed','quality':quality,'quality_passed':quality_passed,
        'program_passed':not control,'gradient_probe_passed':not control,'passed':quality_passed and not control,
        'PPO_joint_steps':0,'publishable':False,'roles':roles}
    return result,{'cases':deepcopy(CASES),'outcomes':outcomes}


def fixture_data(tmp_path,monkeypatch,*,backup=False):
    manifest=json.loads((ROOT/'experiments/warehouse/evidence/warehouse-v621-bc-source-manifest-02.json').read_bytes())
    write(tmp_path/'experiments/warehouse/evidence/warehouse-v621-bc-source-manifest-02.json',manifest)
    monkeypatch.setattr(report.checked,'_verify_source_closure',lambda _:deepcopy(manifest))
    directory=tmp_path/'output/warehouse_v621'/report.protocol.BATCH
    spec={'schema':'warehouse-v621-bc-development-spec.v1','candidate_id':report.protocol.diagnostics.CANDIDATE_ID,
        'batch_id':report.protocol.BATCH,'source_closure_sha256':report.checked.SOURCE02_SHA256,'source_files_sha256':manifest,
        'model_implementation_sha256':report.protocol.digest({p:manifest[p] for p in report.checked.MODEL_CODE_FILES}),
        'output_directory':str(directory),'fresh_precheck_authorized':False,'PPO_joint_steps':0,'publishable':False,
        'recipe':write(tmp_path/'configs/warehouse_v621_bc_recipe.json',report.protocol.recipe()),
        'seed_plan':write(tmp_path/'configs/warehouse_v621_bc_seed_plan.json',report.protocol.seed_plan())}
    spec_path=tmp_path/'output/warehouse_v621/synthetic.spec.json';spec_ref=write(spec_path,spec)
    refs={};required=report.RECIPES if backup else report.RECIPES[:2]
    for recipe in required:
        refs[recipe]={}
        for i in report.protocol.INITIALIZATIONS:
            folder=directory/f'init-{i}'/recipe
            row,collection=model_result(i,recipe,quality_failure=backup and recipe=='shared_b035' and i==6171)
            row.update(development_collection=write(folder/'development.collection.json',collection),
                schema='warehouse-v621-bc-development-result.v1',source_closure_sha256=report.checked.SOURCE02_SHA256,
                training_report=write(folder/'bc.training.json',{'external_program_feedback':0,'external_program_actor_updates':0}),
                actor={'path':str(folder/'UNCREATED-SYNTHETIC.actor.npz'),'sha256':'a'*64,'role_sha256':{r:'b'*64 for r in report.ROLES}})
            for role in report.ROLES:row['roles'][role]['program']=write(folder/(role+'.program.json'),{'synthetic':True})
            refs[recipe][str(i)]=write(folder/'result.json',row)
            cap=write(folder/'capability.completed.json',{'schema':'warehouse-v621-bc-capability-completed.v1',
                'initialization':i,'recipe':recipe,'preauthorized_resets':1005})
            write(folder/'completed.json',{'result':refs[recipe][str(i)],'capability_completed':cap,
                'source_closure_sha256':report.checked.SOURCE02_SHA256})
    def summary(candidate,summary_refs,parent):
        return {'schema':'warehouse-v621-bc-development-summary.v1','specification':parent,'results':summary_refs,
            'source_closure_sha256':report.checked.SOURCE02_SHA256,'candidate_id':report.protocol.diagnostics.CANDIDATE_ID,
            'candidate_recipe':candidate,'all_three_candidate_initializations_passed':not backup or candidate=='shared_b050',
            'main_quality_failure':backup,'next_stage':'conditional_shared_b050_development' if backup and candidate=='shared_b035'
                else 'lock_common_bc_and_implement_fresh_precheck','fresh_precheck_authorized':False,'PPO_joint_steps':0,
            'publishable':False,'pilot_opened':False}
    main_refs={r:refs[r] for r in report.RECIPES[:2]}
    write(directory/'main.summary.json',summary('shared_b035',main_refs,spec_ref))
    if backup:
        backup_spec_ref=write(directory/'backup.spec.json',{**spec,'main_results':refs['shared_b035']})
        write(directory/'backup.summary.json',summary('shared_b050',refs,backup_spec_ref))
    calls=[]
    def native_boundary(root,spec,specification,recipe,bound):
        calls.append(recipe);results={};collections={};pointers={}
        for i in report.protocol.INITIALIZATIONS:
            folder=Path(spec['output_directory'])/f'init-{i}'/recipe
            ref=pointer(folder/'result.json');row=report.checked._read_pointer(root,ref)
            collection=report.checked._read_pointer(root,row['development_collection'])
            bound[ref['path']]=ref['sha256'];bound[row['development_collection']['path']]=row['development_collection']['sha256']
            results[i]=row;collections[i]=collection;pointers[str(i)]=ref
        return results,collections,pointers
    monkeypatch.setattr(report.checked,'_load_cohort',native_boundary)
    return {'root':tmp_path,'specification_path':spec_path,'directory':directory,'calls':calls}


def prepare(f):
    return report.prepare_report(**{k:f[k] for k in ('root','specification_path')})


def test_quality_coverage_differences_and_probe_wording(monkeypatch,tmp_path):
    f=fixture_data(tmp_path,monkeypatch);value=prepare(f)
    assert value['model_count']==6 and len(value['paired_descriptive_differences'])==3
    delta=value['paired_descriptive_differences'][0]
    assert delta['quality']['mean_deliveries']==1 and delta['quality']['mean_collisions']==-1
    assert delta['quality']['shutdown_rate_percentage_points']==pytest.approx(100/75)
    m=delta['roles']['robot_1']['feedback']
    assert m['qualified_coverage_percentage_points']==pytest.approx(30)
    assert m['actual_actionable_coverage_percentage_points']==pytest.approx(30)
    assert m['expected_actionable_coverage_percentage_points']==pytest.approx(30)
    assert m['kl_difference']==pytest.approx(-.2)
    text=report.render_markdown(value)
    assert '不是新的独立验证' in text and '尚无实际 PPO 反馈' in text
    assert '对照未做该探针时不回填零' in text and '0.002' in text
    assert '不适用：诊断对照' in text
    assert not value['model_selection_performed'] and value['PPO_joint_steps']==0
    control=value['models'][0]['roles']['robot_1'];main=value['models'][1]['roles']['robot_1']
    assert 'recorded_validation_strata' not in control
    assert main['recorded_validation_strata']['overall_core_occupancy']['fraction']==.5
    assert set(main['recorded_validation_strata']['nn_top_action'])==set(report.ACTIONS)
    assert sum(q['episodes'] for q in value['models'][0]['quality_by_task'].values())==75


def test_conditional_complete_backup_reports_nine_models_without_selection(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch,backup=True);value=prepare(f)
    assert value['model_count']==9 and len(value['paired_descriptive_differences'])==9
    assert set(value['native_summaries'])=={'main','backup'}
    assert 'selected_initialization' not in value and not value['model_selection_performed']


@pytest.mark.parametrize('missing',('completed.json','capability.completed.json','result.json'))
def test_any_incomplete_job_refuses_before_native_model_checker(tmp_path,monkeypatch,missing):
    f=fixture_data(tmp_path,monkeypatch)
    (f['directory']/'init-6173'/'shared_b035'/missing).unlink()
    with pytest.raises(FileNotFoundError):prepare(f)
    assert f['calls']==[]


def test_backup_branch_waits_for_all_nine_before_reading_models(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch,backup=True)
    (f['directory']/'init-6173'/'shared_b050'/'completed.json').unlink()
    with pytest.raises(FileNotFoundError):prepare(f)
    assert f['calls']==[]


def test_failed_execution_refuses_before_native_model_checker(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch)
    write(f['directory']/'init-6173'/'shared_b035'/'capability.failed.json',{})
    with pytest.raises(RuntimeError,match='interrupted/failed'):prepare(f)
    assert f['calls']==[]


def test_changed_completion_sha_refuses_before_any_model_checker(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch)
    (f['directory']/'init-6173'/'shared_b035'/'capability.completed.json').write_text('{}')
    with pytest.raises(RuntimeError,match='SHA changed'):prepare(f)
    assert f['calls']==[]


def test_changed_bound_sha_is_rejected(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch)
    (f['directory']/'init-6171'/'shared_b035'/'bc.training.json').write_text('{}')
    with pytest.raises(RuntimeError,match='SHA changed'):prepare(f)


def test_wrong_source_refuses_before_native_model_checker(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch)
    spec=json.loads(f['specification_path'].read_bytes());spec['source_closure_sha256']='f'*64;write(f['specification_path'],spec)
    with pytest.raises(RuntimeError,match='source-bound'):prepare(f)
    assert f['calls']==[]


def test_missing_native_expected_metric_is_not_filled_with_zero():
    a,c=model_result(6171,'additive_control');b,d=model_result(6171,'shared_b035')
    for role in report.ROLES:del a['roles'][role]['metrics']['validation']['feedback']['expected_actionable_coverage']
    diff=report.compare_models(report.summarize_model(b,d),report.summarize_model(a,c))
    assert diff['roles']['robot_1']['feedback']['expected_actionable_coverage_percentage_points'] is None


def test_inconsistent_denominator_and_nonfinite_measurements_refuse():
    row,collection=model_result(6171,'shared_b035')
    row['roles']['robot_1']['metrics']['validation']['feedback']['rows']=51
    with pytest.raises(RuntimeError,match='frame counts'):report.summarize_model(row,collection)
    row,collection=model_result(6171,'shared_b035');row['roles']['robot_1']['metrics']['validation']['feedback']['kl']=float('nan')
    with pytest.raises(RuntimeError,match='finite'):report.summarize_model(row,collection)


def test_nonoverwriting_json_markdown_with_bound_pointers(tmp_path,monkeypatch):
    f=fixture_data(tmp_path,monkeypatch);output=tmp_path/'output/warehouse_v621/bc-metrics-reports/synthetic'
    refs=report.write_report(**{k:f[k] for k in ('root','specification_path')},output_directory=output)
    assert set(refs)=={'json','markdown'}
    for ref in refs.values():assert pointer(Path(ref['path']))==ref
    value=json.loads(Path(refs['json']['path']).read_bytes())
    assert value['models'][0]['evidence']['result']['path'].endswith('result.json')
    assert value['models'][0]['evidence']['collection']['path'].endswith('development.collection.json')
    with pytest.raises(RuntimeError,match='new isolated'):report.write_report(
        **{k:f[k] for k in ('root','specification_path')},output_directory=output)
