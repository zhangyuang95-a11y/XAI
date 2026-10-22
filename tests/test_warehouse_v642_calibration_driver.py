"""Declared quantities and closed-gate order, no actual NN initialization."""
from types import SimpleNamespace
import pytest
from scripts import run_warehouse_v642_calibration_pipeline as r


def args(tmp_path):
    return SimpleNamespace(reference_directory=tmp_path/'reference',run_id='synthetic-driver',initialization_seed=0,optimizer_seed=1,
        train_base=17000000,train_action_base=7700000000,evidence_base=18000000,evidence_action_base=7800000000)


def test_closed_actual_prerequisites_precede_recipes_inventories_and_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(r.partitions.admission.v,'OUTPUT',tmp_path/'missing')
    def forbidden(*args,**kwargs):pytest.fail('closed A/B touched new inventories or metadata')
    monkeypatch.setattr(r,'inventories',forbidden)
    with pytest.raises(RuntimeError,match='complete reaped'):r.run(args(tmp_path))
    assert not list(tmp_path.iterdir())


def test_fixed_full_evidence_counts_and_eight_finite_training_streams(tmp_path):
    train,evidence=r.inventories(args(tmp_path))
    assert len(train['cases'])==1024 and evidence['expected_counts']==dict(fit=900,calibration=150,eligibility=300,validation=300)
    assert len(evidence['cases'])==1650
    assert all(sum(c['slot']==slot for c in train['cases'])==128 for slot in range(8))
    assert len({c['base_seed'] for c in evidence['cases']})==1650
    assert r.recipe()['config']['joint_step_budget']==20000 and not r.recipe()['pilot_opened']
