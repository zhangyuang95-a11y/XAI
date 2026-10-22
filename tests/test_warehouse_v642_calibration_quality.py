"""Paired retention gates cannot pool failures or accept unchecked actions."""
from copy import deepcopy
import pytest
from experiments.warehouse import v642_calibration_quality as q


def fixture():
    cases=q.cases(90100000,9010000000)
    actors={v:dict(path=v,sha256=v) for v in q.VARIANTS}
    rows=[dict(variant=v,case=c,actor=actors[v],complete=True,physical_frames=120,NN_action_checks=240,
        action_overrides=0,deliveries=15,robot_collisions=3,shutdown=False) for v in q.VARIANTS for c in cases]
    return cases,actors,rows


def test_pair_passes_and_reports_task_strata_without_accepting_model():
    cases,actors,rows=fixture();r=q.evaluate(rows,cases,actors)
    assert r['task_retention_passed'] and not r['model_acceptance_passed']
    assert r['gates']['raw']['minimum_deliveries']==14.25 and r['gates']['raw']['maximum_collisions']==4.
    assert r['by_task']['regularized']['3']['episodes']==25


@pytest.mark.parametrize('change',['deliveries','collisions','shutdown','unchecked','override','missing','duplicate','hash'])
def test_one_failed_variant_or_invalid_scene_is_not_hidden_by_the_other(tmp_path,change):
    cases,actors,rows=fixture()
    reg=[r for r in rows if r['variant']=='regularized']
    if change=='deliveries':
        for r in reg:r['deliveries']=14
    elif change=='collisions':
        for r in reg:r['robot_collisions']=5
    elif change=='shutdown':
        for r in reg[:4]:r['shutdown']=True
    elif change=='unchecked':reg[0]['NN_action_checks']-=1
    elif change=='override':reg[0]['action_overrides']=1
    elif change=='missing':rows.pop()
    elif change=='duplicate':rows.append(deepcopy(rows[-1]))
    elif change=='hash':reg[0]['actor']=actors['raw']
    if change in ('deliveries','collisions','shutdown'):
        r=q.evaluate(rows,cases,actors)
        assert r['gates']['raw']['passed'] and not r['gates']['regularized']['passed']
        assert not r['task_retention_passed']
    else:
        with pytest.raises(RuntimeError):q.evaluate(rows,cases,actors)


def test_sealed_seed_or_task_imbalance_is_rejected_before_allocation():
    with pytest.raises(RuntimeError):q.cases(250000,10000000)
    cases,_,_=fixture();cases[-1]['task']=1
    with pytest.raises(RuntimeError):q.validate_cases(cases)
