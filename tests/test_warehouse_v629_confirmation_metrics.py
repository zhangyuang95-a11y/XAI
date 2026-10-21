"""Synthetic scoring tests; no collection authority, RNG scene use or solver."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.warehouse import v629_confirmation_metrics as c


def schedule():
    rows=[]
    for split,scenes,draws in (('validation',300,1),('pressure',72,3)):
        for i in range(scenes):
            base=1000000+(0 if split=='validation' else 300)+i
            for draw in range(draws):
                rows.append(dict(case_id=f'{split}-{i}-{draw}',split=split,base=base,scenario_seed=base*10+i%3+1,
                    task=i%3+1,action_seed=100000000+len(rows),sampling_mode='sampled',draw=draw))
    return rows


def records(cases, delta=(.1,.05,.2,.15)):
    result=[]
    for case in cases:
        for role_index,role in enumerate(c.ROLES):
            for fold in range(3):
                result.append(dict(case_id=case['case_id'],base=case['base'],task=case['task'],role=role,fold=fold,
                    program_explanation=.5+delta[2*role_index],reference_explanation=.5,
                    program_expected_feedback=.5+delta[2*role_index+1],reference_expected_feedback=.5))
    return result


def test_exact_confirmation_pressure_scene_schedule_is_not_a_reset_authorization():
    cases=schedule();assert c.validate_cases(cases) is cases
    assert len(cases)==516 and len({r['base'] for r in cases})==372
    assert all(row['sampling_mode']=='sampled' for row in cases)


@pytest.mark.parametrize('change',[
    lambda rows:rows.pop(),
    lambda rows:rows[0].update(sampling_mode='argmax'),
    lambda rows:rows[0].update(case_id='../escape'),
    lambda rows:rows[0].update(action_seed=rows[1]['action_seed']),
    lambda rows:rows[-1].update(draw=0),
    lambda rows:rows[-1].update(base=rows[0]['base'],scenario_seed=rows[0]['scenario_seed']),
])
def test_bad_case_plan_cannot_reach_scoring(change):
    cases=schedule();change(cases)
    with pytest.raises(RuntimeError):c.validate_cases(cases)


def test_four_paired_endpoints_fold_average_and_pressure_clusters():
    cases=schedule();r=records(cases)
    primary=c.paired_bootstrap(r,cases,'validation',iterations=1000)
    pressure=c.paired_bootstrap(r,cases,'pressure',iterations=1000)
    assert primary['independent_scenes']==300 and primary['trajectory_episodes']==300
    assert pressure['independent_scenes']==72 and pressure['trajectory_episodes']==216
    assert primary['passed'] and pressure['passed']
    assert [v['mean_difference'] for v in primary['endpoints'].values()]==pytest.approx([.1,.05,.2,.15])
    assert [v['lower'] for v in primary['endpoints'].values()]==pytest.approx([.1,.05,.2,.15])


def test_unchanged_program_cannot_claim_significant_improvement():
    cases=schedule();r=records(cases,(0,0,0,0))
    result=c.paired_bootstrap(r,cases,'validation',iterations=1000)
    assert not result['passed'] and all(v['raw_lower']==0 and v['lower']<=0 for v in result['endpoints'].values())


def test_long_episodes_and_three_folds_do_not_inflate_effect_or_sample_count():
    cases=[r for r in schedule() if r['split']=='validation'][:6];r=records(cases)
    for row in r:
        case=next(case for case in cases if case['case_id']==row['case_id'])
        delta=.3 if case['base']<1000003 else -.1
        row['program_explanation']=row['program_expected_feedback']=.5+delta
        row['physical_frames']=1 if delta>0 else 1000000  # deliberately extreme
    result=c.paired_bootstrap(r,cases,'validation',iterations=1000)
    assert result['independent_scenes']==6
    assert [v['mean_difference'] for v in result['endpoints'].values()]==pytest.approx([.1]*4)
    assert not result['passed']  # uncertainty from independent scenes remains


def test_pressure_draws_are_averaged_within_scene_before_resampling():
    cases=schedule();r=records(cases)
    byid={case['case_id']:case for case in cases}
    for row in r:
        case=byid[row['case_id']]
        if case['split']=='pressure':
            delta=(.3,0,-.3)[case['draw']]
            row['program_explanation']=row['program_expected_feedback']=.5+delta
    result=c.paired_bootstrap(r,cases,'pressure',iterations=1000)
    assert result['independent_scenes']==72 and not result['passed']
    assert [v['mean_difference'] for v in result['endpoints'].values()]==pytest.approx([0]*4,abs=1e-14)


@pytest.mark.parametrize('change',[
    lambda rows:rows.pop(0),lambda rows:rows.append(deepcopy(rows[0])),
    lambda rows:rows[0].update(base=-1),lambda rows:rows[0].update(case_id='unknown-case'),
])
def test_pairing_corruption_cannot_produce_a_bootstrap_pass(change):
    cases=schedule();r=records(cases);change(r)
    with pytest.raises(RuntimeError):c.paired_bootstrap(r,cases,'validation',iterations=1000)


def test_partial_episode_or_pooled_quality_cannot_hide_failed_pressure():
    cases=schedule();outcomes=[dict(case=row,complete=True,deliveries=10,robot_collisions=2,shutdown=False) for row in cases]
    assert all(q['passed'] for q in c.quality(outcomes,cases).values())
    for row in outcomes:
        if row['case']['split']=='pressure':row['shutdown']=True
    result=c.quality(outcomes,cases)
    assert result['validation']['passed'] and not result['pressure']['passed']
    outcomes[0]['complete']=False
    with pytest.raises(RuntimeError):c.quality(outcomes,cases)


def test_missing_winner_remains_closed_before_any_environment_access(tmp_path,monkeypatch):
    monkeypatch.setattr(c.v,'OUTPUT',tmp_path)
    with pytest.raises(RuntimeError,match='no locked audited winner'):c.load_locked_bundle()


def test_fixed_scoring_keeps_qualification_and_requires_real_improvement(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('v629_synthetic_helpers',Path(__file__).with_name('test_warehouse_v629.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    from backend.training.warehouse_v628_program import qualify
    from backend.training.warehouse_v622_program import digest
    p,training=helpers.fixture(tmp_path);qualify(p,training)
    actor=tmp_path/'synthetic-actor.bin';actor.write_bytes(b'never-loaded-as-a-real-NN')
    source=tmp_path/'synthetic-source.json';source.write_text('{}')
    units=[]
    for role in c.ROLES:
        model=deepcopy(p);model.role=role
        file=tmp_path/(role+'.json');file.write_text(json.dumps(model.to_dict()))
        for fold in range(3):
            units.append(c.Unit(role,fold,model,deepcopy(model),c.v.pointer(file),c.v.pointer(file),
                digest(model.to_dict()),digest(model.to_dict())))
    bundle=c.Bundle(tuple(units),64,c.v.pointer(actor),dict(synthetic_source=c.v.pointer(source)))
    cases=schedule();datasets={}
    for split in c.SPLITS:
        selected=[row for row in cases if row['split']==split];n=len(selected)
        x=np.tile(training['fit']['x'][0],(n,1));m=np.ones((n,5),np.float32)
        q=p.distributions(x,m)[2]
        rows=[dict(episode=row['case_id'],base_seed=str(row['base']),task=row['task']) for row in selected]
        a=dict(x=x,m=m,p=q,sampled=np.full(n,4,dtype=np.int64),rows=rows)
        datasets[split]={role:deepcopy(a) for role in c.ROLES}
    outcomes=[dict(case=row,complete=True,deliveries=10,robot_collisions=2,shutdown=False) for row in cases]
    def forbidden(*args,**kwargs):raise AssertionError('confirmation must not requalify')
    import backend.training.warehouse_v628_program as kernel
    monkeypatch.setattr(kernel,'qualify',forbidden)
    before=[digest(unit.program.to_dict()) for unit in bundle.units]
    result=c.evaluate_fixed(bundle,datasets,cases,outcomes)
    assert all(g['passed'] for unit in result['units'] for g in unit['gates'].values())
    assert not result['metrics_passed'] and not result['bootstrap']['validation']['passed']
    assert not result['full_confirmation_admission']
    assert result['fit_calls']==result['LP_calls']==result['scoring_environment_resets']==0
    assert before==[digest(unit.program.to_dict()) for unit in bundle.units]
    # Bound bytes, not just filenames, govern subsequent use.
    source.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='provenance changed'):bundle.verify()
