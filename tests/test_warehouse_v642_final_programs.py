"""Post-training finalization cannot run early or select another extraction."""
import numpy as np
import pytest
from experiments.warehouse import v642_final_programs as f


def test_incomplete_calibration_refuses_before_source_pool_replay_or_fit(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('missing completion reached replay, fitting or randomness')
    monkeypatch.setattr(f.pools,'audit_observed_pool',forbidden)
    monkeypatch.setattr(f,'complete_program',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='missing/symlink scientific asset'):
        f.run(completed_assets_ref=dict(path=str(tmp_path/'missing.json'),sha256='missing'),
            reference_directory=None,run_id=None)
    assert not list(tmp_path.iterdir())


def fixture(monkeypatch):
    owner={'path':'owner','sha256':'owner-sha'};source={'path':'source','sha256':'source-sha'}
    binding={'path':'binding','sha256':'binding-sha'};extraction={'path':'extraction','sha256':'extraction-sha'}
    programs={r:{'program':dict(path=r,sha256=r)} for r in f.ROLES}
    b=dict(schema='warehouse-v642-observed-refit-source-binding.v1',owner=owner,observed_source=source,
        source_files={'source':'hash'},extraction=extraction)
    result=dict(identity=dict(owner=owner,variant='raw',step=18600),roles=programs)
    monkeypatch.setattr(f.pools,'source_files',lambda:{'source':'hash'})
    monkeypatch.setattr(f,'bound',lambda ref:b if ref==binding else result)
    events=[dict(kind='scheduled_refit_bound',variant='raw',step=s,binding=binding) for s in (15000,16800,18600)]
    return owner,source,b,result,events


def test_only_last_of_three_exact_scheduled_extractions_is_used(monkeypatch):
    owner,source,b,r,events=fixture(monkeypatch)
    programs,refs=f.last_programs(events,variant='raw',owner_ref=owner,source_ref=source)
    assert programs=={role:r['roles'][role]['program'] for role in f.ROLES}
    assert refs==[events[-1]['binding'],b['extraction']]


@pytest.mark.parametrize('change',['missing_schedule','earlier_selected','other_owner','other_pool','other_variant','role_missing'])
def test_wrong_provenance_or_cherry_picked_extraction_is_rejected(monkeypatch,change):
    owner,source,b,r,events=fixture(monkeypatch)
    if change=='missing_schedule':events.pop(0)
    elif change=='earlier_selected':r['identity']['step']=16800
    elif change=='other_owner':b['owner']={}
    elif change=='other_pool':b['observed_source']={}
    elif change=='other_variant':r['identity']['variant']='regularized'
    else:r['roles'].pop('robot_2')
    with pytest.raises(RuntimeError):f.last_programs(events,variant='raw',owner_ref=owner,source_ref=source)
