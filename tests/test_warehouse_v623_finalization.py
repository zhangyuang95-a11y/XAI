"""Synthetic guards for scientific closure; no real episodes or fitting."""
from copy import deepcopy
import json

import pytest

from scripts import finalize_warehouse_v623_refinement as final


def jobs(passes=False):
    return [{'job':spec,'status':'completed','gate_passed':passes,'ledger':{}}
            for spec in final.v.jobs()]


def test_six_unique_bound_role_folds_are_required():
    good=jobs();assert final.audited_stage_decision('S1',good)['all_six_pass'] is False
    for bad in (good[:5],good+[deepcopy(good[0])],
                [deepcopy(good[0]),deepcopy(good[0]),*deepcopy(good[2:])]):
        with pytest.raises(RuntimeError):final.audited_stage_decision('S1',bad)
    bad=deepcopy(good);bad[-1]['status']='engineering_failure'
    with pytest.raises(RuntimeError):final.audited_stage_decision('S1',bad)
    bad=deepcopy(good);bad[-1].update(status='scientific_budget_exhausted',gate_passed=True)
    with pytest.raises(RuntimeError):final.audited_stage_decision('S1',bad)
    bad[-1]['gate_passed']=False
    assert final.audited_stage_decision('S1',bad)['all_six_pass'] is False


def stage(name,passes=False):return final.audited_stage_decision(name,jobs(passes))


def test_stage_order_cannot_skip_failure_or_run_s2_after_success():
    s1,s2=stage('S1'),stage('S2')
    for bad in ([],[s2],[s2,s1],[s1],[s1,s1],[stage('S1',True),s2]):
        with pytest.raises(RuntimeError):final.candidate_decision(bad)
    closed=final.candidate_decision([s1,s2])
    assert closed=={'status':'closed_failed_program_precheck','selected_method':None,'candidate_closed':True}
    for good in ([stage('S1',True)],[s1,stage('S2',True)]):
        outcome=final.candidate_decision(good)
        assert outcome['status']=='pending_fresh_confirmation' and outcome['candidate_closed'] is False
        assert 'pilot_opened' not in outcome and 'publishable' not in outcome


def test_exact_asset_hash_and_path_binding(tmp_path):
    path=tmp_path/'result.json';path.write_text(json.dumps({'passed':False}))
    ref=final.pointer(path);assert final.read_bound(ref,path)=={'passed':False}
    bad=deepcopy(ref);bad['sha256']='0'*64
    with pytest.raises(RuntimeError):final.read_bound(bad,path)
    with pytest.raises(RuntimeError):final.read_bound(ref,tmp_path/'other.json')
    path.write_text(json.dumps({'passed':True}))
    with pytest.raises(RuntimeError):final.read_bound(ref,path)
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(RuntimeError):final.pointer(link)


def audit_evidence():
    summary={'path':'/fixed/S1/summary.json','sha256':'e'*64}
    live={'jobs':jobs(),'all_six_scientific_gates_pass':False,
          'independently_reconstructed_next_stage':'S2','plan_sha256':'f'*64}
    native=dict(final.independent.ZERO,schema='warehouse-v623-refinement-audit.v1',stage='S1',passed=True,
        source_sha256=final.SOURCE,summary_sha256=summary['sha256'],
        checks={j['job_id']:True for j in final.v.jobs()})
    supplemental=dict(final.independent.ZERO,schema='warehouse-v623-supplemental-audit.v1',stage='S1',
        candidate_id=final.CANDIDATE,passed=True,summary_verified=True,source_sha256=final.SOURCE,
        evidence_sha256={summary['path']:summary['sha256']},
        audit_script=final.pointer(final.ROOT/'scripts/audit_warehouse_v623_refinement.py'),**deepcopy(live))
    return summary,native,supplemental,live


def test_native_and_independent_receipts_must_bind_same_summary_source_and_jobs():
    summary,native,supplemental,live=audit_evidence()
    assert final.check_saved_audits('S1',summary,native,supplemental,live)['all_six_pass'] is False
    native_mutations=[lambda r:r.update(summary_sha256='0'*64),
        lambda r:r.update(source_sha256='0'*64),lambda r:r.update(stage='S2'),
        lambda r:r.update(passed=False),lambda r:r['checks'].pop('fold-2-robot_2')]
    for mutate in native_mutations:
        bad=deepcopy(native);mutate(bad)
        with pytest.raises(RuntimeError):final.check_saved_audits('S1',summary,bad,supplemental,live)
    supplemental_mutations=[lambda r:r.update(summary_verified=False),
        lambda r:r['evidence_sha256'].update({summary['path']:'0'*64}),
        lambda r:r.update(jobs=[]),lambda r:r['audit_script'].update(sha256='0'*64),
        lambda r:r.update(independently_reconstructed_next_stage='fresh_confirmation')]
    for mutate in supplemental_mutations:
        bad=deepcopy(supplemental);mutate(bad)
        with pytest.raises(RuntimeError):final.check_saved_audits('S1',summary,native,bad,live)
