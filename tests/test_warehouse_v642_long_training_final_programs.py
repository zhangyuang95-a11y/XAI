"""Long-program provenance component tests, without fitting or any NN run."""
from types import SimpleNamespace
import pytest
from experiments.warehouse import v642_long_training_final_programs as f


@pytest.mark.parametrize('budget',[200000,300000,400000])
def test_last_registered_budget_refit_only(monkeypatch,budget):
    config=f.shared.q.training.owner.PPOConfig(joint_step_budget=budget)
    owner={'owner':'synthetic'};source={'source':'synthetic'};binding={'binding':'synthetic'}
    extraction={'extraction':'synthetic'}
    b=dict(schema='warehouse-v642-observed-refit-source-binding.v1',owner=owner,observed_source=source,
           source_files={'synthetic':'source'},extraction=extraction)
    r=dict(identity=dict(owner=owner,variant='raw',step=config.extraction_steps[-1]),
           roles={role:{'program':{'synthetic-program':role}} for role in f.ROLES})
    monkeypatch.setattr(f,'bound',lambda ref:b if ref==binding else r)
    monkeypatch.setattr(f.shared.pools,'source_files',lambda:{'synthetic':'source'})
    events=[dict(kind='scheduled_refit_bound',variant='raw',step=n,binding=binding) for n in config.extraction_steps]
    result,refs=f.last_programs(events,variant='raw',owner_ref=owner,source_ref=source,config=config)
    assert result=={role:r['roles'][role]['program'] for role in f.ROLES}
    assert refs==[binding,extraction]
    events[0]['step']=15000
    with pytest.raises(RuntimeError,match='registered long-budget refits'):
        f.last_programs(events,variant='raw',owner_ref=owner,source_ref=source,config=config)


@pytest.mark.parametrize('phase,budget',[('formal',200000),('development',20000)])
def test_calibration_or_unlocked_formal_cannot_use_long_final_claim(phase,budget):
    plan=SimpleNamespace(value=dict(phase=phase,stream_id='development-0',joint_step_budget=budget))
    with pytest.raises(RuntimeError):f.identity(plan)


def test_budget_and_stream_have_separate_finalization_identity():
    ids={f.identity(SimpleNamespace(value=dict(phase='development',stream_id=f'development-{s}',joint_step_budget=b)))
         for s in range(3) for b in (200000,300000,400000)}
    assert len(ids)==9
