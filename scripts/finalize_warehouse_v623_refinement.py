#!/usr/bin/env python3
"""Seal a single-use, spent-development conclusion for the fixed v6.23 run.

Never opens admission, performs fitting, creates fresh trajectories, or changes
existing experiment evidence. A failed scientific candidate and an engineering
interruption have different meanings: interruptions cannot justify closure.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse import v623_protocol as v
from experiments.warehouse import v622_recovery_protocol as old
from scripts import audit_warehouse_v623_refinement as independent

SOURCE='03cd01e2be51792c67e51c3b3a3f1cd0c9110c8e69630cb6a46ef973f5bb393c'
CANDIDATE='warehouse-v623-a32-refine-01'
DESTINATION=v.OUTPUT/'finalizations'/'completed-development-01'
AUTHORITY=dict(independent.ZERO,fresh_development_episodes=0,
               fresh_pressure_trajectories=0,formal_training_authorized=False)


def require(condition,message):
    if not condition:raise RuntimeError(message)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pointer(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(),'missing or symlink evidence '+str(p))
    return {'path':str(p.resolve()),'sha256':sha256(p.read_bytes()).hexdigest()}


def read_bound(ref,expected_path=None):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'malformed evidence pointer')
    if expected_path is not None:
        require(ref['path']==str(Path(expected_path).resolve()),'evidence path differs')
    actual=pointer(ref['path']);require(actual==ref,'evidence bytes/SHA differ '+ref['path'])
    return json.loads(Path(ref['path']).read_bytes())


def verify_all_pointers(value):
    """Re-read bound evidence immediately before writing the final manifest."""
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            require(pointer(value['path'])==value,'bound evidence changed before final write')
        else:
            for item in value.values():verify_all_pointers(item)
    elif isinstance(value,list):
        for item in value:verify_all_pointers(item)


def audited_stage_decision(stage,jobs):
    """Require each exact job once; only independently evidenced states count."""
    require(stage in ('S1','S2'),'unregistered stage')
    specs=v.jobs()
    require(isinstance(jobs,list) and len(jobs)==6,'scientific stage is partial')
    require([j.get('job') for j in jobs]==specs,'scientific stage jobs duplicate/misordered/mismatched')
    for j in jobs:
        require(j.get('status') in ('completed','scientific_budget_exhausted'),'engineering interruption cannot be scientifically finalized')
        require(type(j.get('gate_passed')) is bool,'missing independently audited gate')
        require(j['status']=='completed' or j['gate_passed'] is False,'budget failure claims scientific pass')
    passed=all(j['status']=='completed' and j['gate_passed'] for j in jobs)
    return {'stage':stage,'all_six_pass':passed,'jobs':jobs}


def candidate_decision(stages):
    """Enforce ordered complete stages and never grant training/publication."""
    require(isinstance(stages,list) and len(stages) in (1,2),'finalization needs one or two complete stages')
    require(stages[0].get('stage')=='S1','S1 must be first')
    for s in stages:require(type(s.get('all_six_pass')) is bool,'stage gate must be independently reconstructed')
    if stages[0]['all_six_pass']:
        require(len(stages)==1,'S2 is unauthorized after S1 success')
        return {'status':'pending_fresh_confirmation','selected_method':'S1','candidate_closed':False}
    require(len(stages)==2 and stages[1].get('stage')=='S2','complete audited S1 failure requires complete S2 before finalization')
    if stages[1]['all_six_pass']:
        return {'status':'pending_fresh_confirmation','selected_method':'S2','candidate_closed':False}
    return {'status':'closed_failed_program_precheck','selected_method':None,'candidate_closed':True}


def check_saved_audits(stage,summary_ref,native,supplemental,live):
    """Bind both saved auditors to exact summary/source and live re-verification."""
    require(native.get('schema')=='warehouse-v623-refinement-audit.v1'
            and native.get('stage')==stage and native.get('passed') is True
            and native.get('source_sha256')==SOURCE
            and native.get('summary_sha256')==summary_ref['sha256'],'native audit source/summary/stage differs')
    require(native.get('checks')=={j['job_id']:True for j in v.jobs()},'native audit is partial or failed')
    require(supplemental.get('schema')=='warehouse-v623-supplemental-audit.v1'
            and supplemental.get('stage')==stage and supplemental.get('candidate_id')==CANDIDATE
            and supplemental.get('passed') is True and supplemental.get('summary_verified') is True
            and supplemental.get('source_sha256')==SOURCE
            and supplemental.get('evidence_sha256',{}).get(summary_ref['path'])==summary_ref['sha256'],
            'supplemental audit source/summary/stage differs')
    require(supplemental.get('audit_script')==pointer(ROOT/'scripts/audit_warehouse_v623_refinement.py'),
            'saved supplemental auditor bytes differ')
    for row in (native,supplemental):independent.zero(row)
    for key in ('jobs','all_six_scientific_gates_pass','independently_reconstructed_next_stage','plan_sha256'):
        require(supplemental.get(key)==live.get(key),'saved/live independent scientific conclusion differs '+key)
    return audited_stage_decision(stage,live['jobs'])


def verify_stage(stage,supplemental_path):
    folder=v.folder(stage)
    summary_ref=pointer(folder/'summary.json');summary=read_bound(summary_ref)
    native_ref=pointer(folder/'audit.json');native=read_bound(native_ref)
    supplemental_ref=pointer(supplemental_path);saved=read_bound(supplemental_ref)
    # This calls only independent route/qualification/metric math and byte reads;
    # it writes nothing, fits nothing, and executes no environment trajectories.
    live=independent.audit(stage,True)
    decision=check_saved_audits(stage,summary_ref,native,saved,live)
    require(summary.get('all_six_pass')==decision['all_six_pass'],'native summary scientific pass differs')
    plan_ref=pointer(folder/'plan.json');plan=read_bound(plan_ref)
    assets={'summary':summary_ref,'native_audit':native_ref,'supplemental_audit':supplemental_ref,
            'plan':plan_ref,'plan_claim':pointer(folder/'plan.claim.json'),'jobs':{},
            'source_snapshot_root':str((folder/'source'/SOURCE).resolve()),
            'source_files_sha256':plan['source_files']}
    totals={'search_leaf_api_calls':0,'final_leaf_api_calls':0,'total_new_leaf_api_calls':0,
            'search_convex_solver_calls':0,'final_convex_solver_calls':0}
    for spec,ind in zip(v.jobs(),live['jobs'],strict=True):
        p=folder/'jobs'/spec['job_id'];resultref=pointer(p/'result.json');r=read_bound(resultref)
        require(r['job']==spec and r['plan_sha256']==plan_ref['sha256'],'bound result/plan mismatch')
        completionref=pointer(p/'completed.json');completion=read_bound(completionref)
        require(completion=={'plan_sha256':plan_ref['sha256'],'result_sha256':resultref['sha256'],
                            'passed':r['gate']['passed']},'bound result/completion mismatch')
        refs={'result':resultref,'started':pointer(p/'started.json'),'completion':completionref}
        for key,filename in (('program','program.json'),('structure','structure.json'),('search_report','search-report.json')):
            if key in r:
                read_bound(r[key],p/filename);refs[key]=r[key]
            elif (p/filename).is_file():refs[key+'_partial_evidence']=pointer(p/filename)
        assets['jobs'][spec['job_id']]=refs
        ledger=ind['ledger'];sc=ledger['new_leaf_calls'];ss=ledger['convex_solver_calls']
        if r['status']=='completed':
            fc=ledger['final_leaf_calls'];fs=ledger['final_convex_solver_calls']
        else:
            fc=r['failure']['leaf_optimization_calls'];fs=r['failure']['convex_solver_calls']
        totals['search_leaf_api_calls']+=sc;totals['final_leaf_api_calls']+=fc
        totals['total_new_leaf_api_calls']+=sc+fc
        totals['search_convex_solver_calls']+=ss;totals['final_convex_solver_calls']+=fs
    return decision,assets,totals,live


def engineering_interruption(plan):
    parent=plan['engineering_parent'];failed=read_bound(parent['failed']);oldplan=read_bound(parent['plan'])
    require(failed.get('error_type')=='RuntimeError' and failed.get('error')=='job failed without completion: fold-0-robot_1',
            'unexpected engineering parent')
    require(oldplan.get('candidate_id')==CANDIDATE and parent.get('reuse_old_jobs') is False
            and parent.get('old_complete_results')==0,'engineering parent imported scientific results')
    root=Path(parent['plan']['path']).parent
    require(not (root/'summary.json').exists() and not (root/'jobs').exists(),
            'engineering interruption contains scientific jobs; explicit accounting required')
    return {'classification':'engineering_interruption_before_job_creation',
        'parent_plan':parent['plan'],'failure':parent['failed'],
        'source_sha256':oldplan['source_sha256'],'source_snapshot_root':str((root/'source'/oldplan['source_sha256']).resolve()),
        'completed_scientific_jobs':0,'observed_leaf_api_calls':0,
        'counts_in_current_stage_totals':False,'scientific_failure_or_efficacy_evidence':False}


def finalize(supplemental_s1,supplemental_s2):
    own=pointer(__file__)
    require(v.CANDIDATE==CANDIDATE and v.source_sha()==SOURCE,'candidate/frozen source changed')
    require(not DESTINATION.exists(),'single-use development finalization already exists')
    first,assets1,total1,live1=verify_stage('S1',supplemental_s1)
    stages=[first];assets={'S1':assets1};totals={'S1':total1};reverify={'S1':digest(live1)}
    if first['all_six_pass']:
        require(not v.folder('S2').exists(),'S2 execution after S1 success is unauthorized')
    else:
        second,assets2,total2,live2=verify_stage('S2',supplemental_s2)
        stages.append(second);assets['S2']=assets2;totals['S2']=total2;reverify['S2']=digest(live2)
        plan2=read_bound(assets2['plan']);dep=plan2.get('stage_dependency')
        require(dep=={'summary':assets1['summary'],'audit':assets1['native_audit']},'S2 prior failed-stage authority differs')
    outcome=candidate_decision(stages)
    plan=read_bound(assets1['plan']);actor,actorpath=old.checked_actor(ROOT)
    require(plan['actor']==pointer(actorpath) and plan['actor']['sha256']==old.ACTOR_SHA256
            and plan['actor_role_sha256']==old.ROLE_SHA256,'Actor binding differs')
    require(plan['prior_A32']=={j['job_id']:v.checked_prior(j) for j in v.jobs()},'old A32 binding differs')
    manifest=dict(AUTHORITY,schema='warehouse-v623-development-evidence-manifest.v1',candidate_id=CANDIDATE,
        source_sha256=SOURCE,finalizer=own,supplemental_auditor=pointer(ROOT/'scripts/audit_warehouse_v623_refinement.py'),
        actor=plan['actor'],actor_role_sha256=old.ROLE_SHA256,prior_A32=plan['prior_A32'],
        current_scientific_stages=assets,actual_compute_totals_by_stage=totals,
        original_engineering_interruption=engineering_interruption(plan),
        independent_reverification_sha256=reverify,
        scope='already-spent development only; no pooling failed units; no task-quality or human-effect inference')
    receipt=dict(AUTHORITY,schema='warehouse-v623-development-finalization.v1',candidate_id=CANDIDATE,
        source_sha256=SOURCE,finalizer=own,**outcome,
        all_stage_gates=[{'stage':s['stage'],'all_six_pass':s['all_six_pass'],
                          'passed_role_folds':sum(j['gate_passed'] for j in s['jobs']),
                          'required_role_folds':6} for s in stages],
        at_utc=datetime.now(timezone.utc).isoformat(),single_use=True,
        release_id='policylens-three-domain-20261016.v6.0',release_generated=False)
    require(v.source_sha()==SOURCE and pointer(__file__)==own,'source/finalizer changed during finalization')
    verify_all_pointers(manifest)
    for stage_assets in assets.values():
        for relative,expected in stage_assets['source_files_sha256'].items():
            require(pointer(Path(stage_assets['source_snapshot_root'])/relative)['sha256']==expected,
                    'source snapshot changed before final write')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    mpath=DESTINATION/'evidence-manifest.json'
    with mpath.open('x') as f:json.dump(manifest,f,sort_keys=True,separators=(',',':'),allow_nan=False)
    receipt['evidence_manifest']=pointer(mpath)
    with (DESTINATION/'receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True,separators=(',',':'),allow_nan=False)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s1-supplemental',type=Path,default=v.folder('S1')/'tracked-postsummary-supplemental-audit.json')
    p.add_argument('--s2-supplemental',type=Path,default=v.folder('S2')/'tracked-postsummary-supplemental-audit.json')
    args=p.parse_args();r=finalize(args.s1_supplemental,args.s2_supplemental)
    print(json.dumps({'status':r['status'],'candidate_closed':r['candidate_closed'],
                      'receipt':str(DESTINATION/'receipt.json')},sort_keys=True))

if __name__=='__main__':main()
