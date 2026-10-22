"""Depth-aware fixed scoring, full-cohort admission and immutable old readers."""
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.warehouse import v642_confirmation_metrics as m,v642_protocol as v
from backend.training import warehouse_v642_program as language
from backend.training.warehouse_v626_search import Node,encode
from backend.training.warehouse_v622_program import plain,MASK_INDICES
from backend.training.warehouse_v626_domain import domain_table,domain_sha


def test_partial_development_rejects_before_any_confirmation_side_effect(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'OUTPUT',tmp_path)
    with pytest.raises(RuntimeError,match='complete reaped'):m.load_locked_bundle()
    assert not list(tmp_path.iterdir())


def test_depth12_scoring_does_not_fall_through_historical_depth9_router():
    w=np.zeros(173,np.float32);w[MASK_INDICES[0]]=1.;tree=None
    for _ in range(12):tree=Node(tuple(w),-.5,None,tree)
    structure=plain(encode(tree)[0]);leaf=dict(version='warehouse-role-program.v6.26-domain-complete',domain_sha256=domain_sha(),
        normalized=True,support=[],bias=[-5.,-5.,-5.,-5.,5.],weights=[[],[],[],[],[]],witness=4)
    p=language.Program('robot_1','a'*64,structure,[dict(leaf) for _ in range(13)],'C3',256,maximum_depth=12)
    x=np.array([[d['center'] for d in domain_table()['features']]],np.float32);x[:,MASK_INDICES]=1.;mask=x[:,MASK_INDICES]
    assert m.forward(p,x,mask)[0].tolist()==[12]
    assert all(np.array_equal(a,b) for a,b in zip(m.forward(p,x,mask),p.distributions(x,mask),strict=True))
    with pytest.raises(RuntimeError,match='exceeded depth'):m.arithmetic.forward(p,x,mask)


def test_complete_plan_has_equal_budgets_and_canonical_snapshot_paths():
    _,p=v.checked_plan();files=v.source_files()
    assert len(p['jobs'])==12 and all(not key.startswith('/') and '..' not in key.split('/') for key in files)
