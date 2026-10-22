"""Closed calibration and altered source/seed bindings cannot collect quality."""
from pathlib import Path
import json
import numpy as np
import pytest
from experiments.warehouse import v642_quality_collection as c


def test_missing_completed_training_refuses_before_seeds_or_environment(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('missing calibration reached seed generation or environment')
    monkeypatch.setattr(c.quality,'cases',forbidden)
    monkeypatch.setattr(c.WarehouseMultiAgentEnv,'__init__',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='missing/symlink scientific asset'):
        c.prepare(dict(path=str(tmp_path/'missing-report.json'),sha256='missing'),run_id='must-not-create',base_start=None,action_start=None)
    assert not list(tmp_path.iterdir())


def test_exact_parser_accepts_only_declared_matched_reuse(tmp_path):
    rows=c.quality.cases(90100000,9010000000)
    value=c.declaration(rows);p=tmp_path/'exact-plan.json';p.write_text(json.dumps(value))
    parsed,_=c.exact.read_exact_plan(p)
    assert parsed==value and len(value['uses'])==900
    for variant in c.quality.VARIANTS:
        assert sum(r['group'].startswith(variant+':') for r in value['uses'])==300
    assert len({r['value'] for r in value['uses'] if r['kind']=='env'})==75


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'plans')
    path=tmp_path/'synthetic-completed-report.json';path.write_text('{}');ref=c.pointer(path)
    fixed=dict(report=ref,actors={},actor_role_sha256={},inputs=[ref])
    monkeypatch.setattr(c,'assets',lambda r:fixed)
    monkeypatch.setattr(c,'claim_path',lambda:tmp_path/'quality-claim.json')
    plan=c.prepare(ref,run_id='synthetic-plan-only',base_start=90100000,action_start=9010000000)
    return plan


def test_prepared_plan_has_exact_inventory_source_and_no_reset_authority(prepared):
    p,folder=c.checked(prepared)
    assert p['recipe']['maximum_environment_resets']==225 and len(p['cases'])==75
    assert not p['training_authority'] and not p['sealed_authority'] and not c.claim_path().exists()


@pytest.mark.parametrize('change',['snapshot','seed','recipe','prerequisite'])
def test_altered_bindings_are_rejected_before_any_collection(prepared,change):
    p,folder=c.checked(prepared)
    if change=='snapshot':
        name=next(iter(p['source_files']));target=folder/'source'/name;target.chmod(0o644);target.write_bytes(b'tampered')
    elif change=='seed':
        target=Path(p['exact_plan']['path']);target.chmod(0o644);value=c.bound(p['exact_plan']);value['uses'][0]['value']+=1
        target.write_text(json.dumps(value))
    elif change=='recipe':
        target=Path(prepared['path']);target.chmod(0o644);p['recipe']['minimum_reference_delivery_fraction']=.9;target.write_text(json.dumps(p))
    else:
        target=Path(p['fixed']['inputs'][0]['path']);target.write_text('changed')
    with pytest.raises((RuntimeError,ValueError)):
        c.checked(c.pointer(folder/'plan.json'))
    assert not c.claim_path().exists()


def empty_history(tmp_path,monkeypatch):
    roots=[tmp_path/f'synthetic-history-{i}' for i in range(4)]
    for root in roots:root.mkdir()
    monkeypatch.setattr(c,'history_roots',lambda:roots)
    registry=tmp_path/'synthetic-trace-registry.json';registry.write_text('{"registered_trace_paths":[]}')
    monkeypatch.setattr(c,'REGISTRY_REF',c.pointer(registry))
    return roots


def test_indexed_four_root_exact_audit_and_delta_keep_authority_closed(prepared,tmp_path,monkeypatch):
    roots=empty_history(tmp_path,monkeypatch);ref=c.audit_seeds(prepared)
    assert c.clearance(prepared,execution=True)[0]==ref
    assert c.bound(ref)['environment_reset_ceiling']==225 and not c.claim_path().exists()
    (roots[0]/'new-warehouse-receipt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='historical metadata changed'):
        c.clearance(prepared,execution=True)


def test_registered_prior_seed_blocks_quality_before_any_reset(prepared,tmp_path,monkeypatch):
    roots=empty_history(tmp_path,monkeypatch)
    (roots[0]/'prior-warehouse-receipt.json').write_text('{"scenario_seed":901000001}')
    with pytest.raises(RuntimeError,match='historical seed conflict|unresolved metadata'):
        c.audit_seeds(prepared)
    assert not c.claim_path().exists()
