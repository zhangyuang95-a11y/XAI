from copy import deepcopy
import json
import pytest
from experiments.warehouse import v625_protocol as v


def metric(coverage=.55):
    row=dict(coverage=coverage,expected_actionable_coverage=.5,agreement=.99,kl=.1,
        whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0,
        illegal_count=0,path_uncertified_count=0)
    return dict(feedback=deepcopy(row),explanation=deepcopy(row))


def results():
    rows=[]
    for job in v.jobs():
        m=metric()
        rows.append(dict(job=job,status='completed',leaf_count=32,
                         metrics={'validation':m},gate=v.program_gate(m)))
    return rows


def test_P0_cannot_replace_failed_upgrade_and_worst_unit_is_not_pooled():
    rows=results()
    assert v.comparison(rows)['selected_arm']=='P1'
    bad=next(r for r in rows if r['job']['arm']=='P1')
    bad['metrics']['validation']=metric(.44)
    bad['gate']=v.program_gate(bad['metrics']['validation'])
    summary=v.comparison(rows)
    assert summary['arms']['P0']['all_six_pass']
    assert summary['selected_arm'] is None
    assert summary['next_stage']=='closed_failed_program_precheck'
    for sample in (rows[:-1],rows+[deepcopy(rows[0])]):
        with pytest.raises(RuntimeError):v.comparison(sample)


def test_source_closure_extends_but_cannot_rewrite_historical_bytes():
    historical=v.previous.source_files()
    assert v.digest(historical)==v.HISTORICAL_SOURCE
    current=v.source_files()
    assert all(current[k]==value for k,value in historical.items())
    assert set(v.NEW_FILES)<=set(current)


def test_registered_recipe_has_no_future_environment_or_training_authority(tmp_path):
    registered=v.expected_recipe()
    assert registered['environment_resets']==registered['actor_updates']==registered['PPO_joint_steps']==0
    assert registered['future_stages_authorized'] is False
    assert registered['future_training']['authorized_by_stage_A_recipe'] is False
    assert registered['future_confirmation']['pressure_gate']=='strict_development'
    assert registered['maximum_leaf_coefficients']==20
    assert registered['leaf_fit']['tolerance']==1e-6
    path=tmp_path/v.RECIPE;path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registered));assert len(v.recipe(tmp_path))==64
    for key,value in [('maximum_leaves',33),('eligible_upgrade_arm','P0'),('environment_resets',1)]:
        bad=deepcopy(registered);bad[key]=value;path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):v.recipe(tmp_path)


def test_future_phase_refuses_missing_complete_scientific_pass(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'OUTPUT',tmp_path)
    with pytest.raises(RuntimeError):v.require_stage_A_pass()
