"""v630 contracts/guards in temporary output; never launch real new search."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from experiments.warehouse import v630_protocol as v
from scripts import run_warehouse_v630 as runner


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return v.pointer(path)


def test_recipe_preserves_gates_and_frozen_NN_no_retries():
    r=v.recipe();assert r['gate']==v.previous.recipe()['gate'] and r['maximum_leaves']==256 and r['maximum_depth']==9
    assert r['limits']==dict(leaf=0,node=256,proof=0) and not r['automatic_retry']
    assert r['all_18_completed_before_selection'] and r['all_source_surrogate_distributions_frozen']
    assert r['actor_updates']==r['PPO_joint_steps']==r['environment_resets']==0 and not r['training_authority']
    jobs=v.jobs();assert len(jobs)==18 and len({j['job_id'] for j in jobs})==18
    assert {j['name'] for j in jobs}=={'C0','P','A'}


def test_wrong_live_dependency_is_rejected_before_any_output(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'OUTPUT',tmp_path/'candidate');monkeypatch.setattr(v,'process_identity',lambda pid:'synthetic unrelated process')
    with pytest.raises(RuntimeError,match='original v629 supervisor'):v.prepare(99)
    assert not (tmp_path/'candidate').exists()


def test_prepare_snapshots_complete_source_and_is_single_use(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'OUTPUT',tmp_path/'candidate')
    monkeypatch.setattr(v,'process_identity',lambda pid:'synthetic python run_warehouse_v629.py --pipeline')
    plan=v.prepare(99);_,checked=v.checked_plan()
    assert checked==plan and plan['source_files']==v.source_files()
    assert not plan['training_authority'] and not plan['sealed_authority'] and plan['dependency_process']['pid']==99
    with pytest.raises(FileExistsError):v.prepare(99)


def test_mutated_plan_or_source_snapshot_refuses_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'OUTPUT',tmp_path/'candidate')
    monkeypatch.setattr(v,'process_identity',lambda pid:'synthetic python run_warehouse_v629.py --pipeline')
    plan=v.prepare(99);snapshot=v.OUTPUT/'source'/plan['source_sha256']/'scripts/run_warehouse_v630.py'
    snapshot.chmod(0o644);snapshot.write_text('tampered')
    with pytest.raises(RuntimeError,match='source snapshot changed'):v.checked_plan()


def upstream(tmp_path,monkeypatch,*,passed=False):
    output=tmp_path/'previous';monkeypatch.setattr(v.previous,'OUTPUT',output);monkeypatch.setattr(v,'process_identity',lambda pid:None)
    from scripts.run_warehouse_v629 import gate as old_gate
    jobs=v.previous.jobs();plan_path=output/'synthetic-plan.json';write(plan_path,dict(jobs=jobs))
    monkeypatch.setattr(v.previous,'checked_plan',lambda:(plan_path,dict(jobs=jobs)))
    checked=[]
    for job in jobs:
        folder=output/'jobs'/job['job_id'];coverage=.5 if passed and job['name']=='G64' else .44
        metrics={k:dict(coverage=coverage,agreement=.99,kl=.1,expected_actionable_coverage=.44,
            illegal_count=0,path_uncertified_count=0,whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0)
            for k in ('feedback','explanation')}
        search_ref=write(folder/'search-report.json',dict(mandatory_complete=True))
        program=write(folder/'program.json',dict(synthetic=True,job=job))
        result_ref=write(folder/'result.json',dict(job=job,status='completed',program=program,search_report=search_ref,
            metrics=dict(validation=metrics),gate=old_gate(metrics,v.bound(search_ref))))
        write(folder/'completed.json',dict(result_sha256=result_ref['sha256']))
        checked.append(dict(job=job,status='completed',program=program,protected_exact_forward=True,
            independent_masks_and_duals=True,independent_reference_metrics=True))
    audit=write(output/'independent-audit.json',dict(integrity_passed=True,jobs=checked))
    summary=write(output/'summary.json',dict(all_24_settled=True,scientific_pass=passed,selected_capacity=64 if passed else None,independent_audit=audit))
    write(output/'execution/completed.json',dict(jobs=24,all_owned_workers_waited=True,
        exits=[dict(job=job,pid=999000+i,waited=True,returncode=0) for i,job in enumerate(jobs)]))
    if not passed:write(output/'candidate.closed.json',dict(status='closed_failed_program_precheck',summary=summary,
        scientific_pass=False,fresh_confirmation_used=False,pilot_opened=False))
    return output


def test_successful_upstream_winner_skips_new_method(tmp_path,monkeypatch):
    upstream(tmp_path,monkeypatch,passed=True);value=v.old_completion()
    assert value['skip_new_candidate'] and value['reason']=='original_v629_winner_has_priority'


def test_interrupted_or_unreaped_upstream_cannot_clear_dependency(tmp_path,monkeypatch):
    output=upstream(tmp_path,monkeypatch);path=output/'execution/completed.json';value=json.loads(path.read_text())
    value['exits'][0]['waited']=False;write(path,value)
    with pytest.raises(RuntimeError,match='reap all 24'):v.old_completion()


def test_failed_upstream_requires_exact_failure_seal(tmp_path,monkeypatch):
    output=upstream(tmp_path,monkeypatch);assert not v.old_completion()['skip_new_candidate']
    value=json.loads((output/'candidate.closed.json').read_text());value['fresh_confirmation_used']=True;write(output/'candidate.closed.json',value)
    with pytest.raises(RuntimeError,match='failed-candidate seal'):v.old_completion()


def test_actual_frozen_quality_reproduces_from_complete_episode_outcomes():
    _,old=v.previous.checked_plan();plan=dict(actor=old['actor'],collection=old['collection'],
        common_quality=v.pointer(v.data_protocol.OUTPUT/'common-NN-quality.json'))
    quality=v.common_quality(plan)
    assert quality['sampled']['episodes']==1350 and quality['argmax']['episodes']==300
    assert quality['sampled']['deliveries']>14 and quality['sampled']['shutdown_episodes']==28


def test_job_entry_without_registered_plan_cannot_create_work(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'OUTPUT',tmp_path/'absent')
    with pytest.raises(RuntimeError,match='missing/symlink scientific asset'):runner.run_job('A-fold-0-robot_1')
    assert not list(tmp_path.iterdir())


def test_gate_checks_each_purpose_and_reliable_proxy_without_average():
    m={k:dict(coverage=.46,agreement=.99,kl=.1,expected_actionable_coverage=.44,illegal_count=0,path_uncertified_count=0,
        whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0) for k in ('feedback','explanation')}
    report=dict(mandatory_complete=True,reliable_selection_passed=True)
    assert runner.gate(m,report,'all')['passed']
    bad=deepcopy(m);bad['explanation']['coverage']=.449
    assert not runner.gate(bad,report,'all')['passed']
    assert not runner.gate(m,dict(mandatory_complete=True,reliable_selection_passed=False),'all')['passed']
