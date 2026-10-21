"""Synthetic read-only checkpoint measurement and evidence-scope checks."""
import importlib.util
from pathlib import Path
import pytest
from backend.training.warehouse_v628_search import Budget,Context
from backend.training.warehouse_v628_program import qualify

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_committed_checkpoint_is_measured_without_resuming_or_mutating(tmp_path):
    fixtures=module('tests/test_warehouse_v628.py','fixture628')
    observer=module('scripts/measure_warehouse_v628_intermediate.py','observer628')
    _,p,parts=fixtures.fixture(tmp_path)
    job=dict(job_id='C64-fold-0-robot_1',capacity=64,fold=0,role='robot_1');plan_sha='b'*64
    folder=tmp_path/'job';folder.mkdir()
    budget=Budget(folder/'checkpoint.json',dict(plan_sha256=plan_sha,job=job))
    try:
        ctx=Context(parts,p,budget);model,score,_=ctx.evaluate(fixtures.decode(p.structure),preserve_parent=True)
        ctx.checkpoint({},[(model,score)],(model,score))
        before={str(f):f.read_bytes() for f in folder.rglob('*') if f.is_file()}
        snap=observer.load_snapshot(folder,job,plan_sha,p.actor_role_sha256)
        qualify(p,parts);result=observer.calculate(snap['program'],parts,p)
        after={str(f):f.read_bytes() for f in folder.rglob('*') if f.is_file()}
        assert before==after
        assert result['outer_validation_used'] is False and result['fit_calls']==0
        assert result['delta_explanation_percentage_points']==0
        assert result['metrics']['explanation']['coverage']==1
        assert result['metrics']['feedback']['physical_frame_count']==120
        assert result['scope']=='fit_internal_180_episode_score_not_outer_validation_or_admission'
        with pytest.raises(ValueError,match='identity mismatch'):
            observer.load_snapshot(folder,job,'c'*64,p.actor_role_sha256)
    finally:budget.session.close()
