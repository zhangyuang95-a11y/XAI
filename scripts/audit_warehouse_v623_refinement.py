#!/usr/bin/env python3
"""Read-only v6.23 scientific ledger/metric audit, outside frozen source closure.

Writes only a new audit receipt. Never fits, resumes, resets, finalizes or changes
candidate results. With --summary also independently reconstructs that summary.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v623_protocol as v
from experiments.warehouse import v622_recovery_protocol as old
from scripts import audit_warehouse_v621_diagnostics as math
from scripts.audit_warehouse_v622_capacity import context, encoded_data, section, independent_gate
from scripts.run_warehouse_v621_diagnostics import runtime_identity
from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256

EXPECTED_SOURCE='03cd01e2be51792c67e51c3b3a3f1cd0c9110c8e69630cb6a46ef973f5bb393c'
ZERO={'environment_resets':0,'actor_updates':0,'PPO_joint_steps':0,'pilot_opened':False,
      'publishable':False,'sealed_audit_used':False,'heldout_used':False}
BUDGET={'beam_width':4,'final_leaf_reserve':32,'final_wall_reserve':600,
        'maximum_new_leaf_calls':20000,'maximum_rounds':512,
        'stagnant_recipe_cycles':3,'wall_seconds':5400}
WALL_OVERSHOOT_SECONDS=5.
READ={}

def require(condition,message):
    if not condition: raise RuntimeError(message)

def raw(path,expected=None):
    p=Path(path).resolve(strict=True)
    require(not Path(path).is_symlink(),'symlink evidence '+str(path))
    b=p.read_bytes();digest=sha256(b).hexdigest()
    if expected is not None:require(digest==expected,'hash mismatch '+str(p))
    require(str(p) not in READ or READ[str(p)]==digest,'evidence changed '+str(p))
    READ[str(p)]=digest
    return b

def obj(path,expected=None):return json.loads(raw(path,expected))
def ref(path):
    p=Path(path).resolve(strict=True);raw(p)
    return {'path':str(p),'sha256':READ[str(p)]}

def bound(pointer,expected_path):
    require(pointer.get('path')==str(Path(expected_path).resolve()),'asset path mismatch '+str(expected_path))
    return obj(expected_path,pointer.get('sha256'))

def integer(x,low=0,high=None):return type(x) is int and x>=low and (high is None or x<=high)
def finite(x):return type(x) in (int,float) and np.isfinite(x)
def zero(row):
    for k,val in ZERO.items(): require(type(row.get(k)) is type(val) and row[k]==val,'closed authority differs '+k)

def search_ledger(s,role,initial_sha,target_sha):
    require(s.get('schema')=='warehouse-v623-fit-search.v1','search schema')
    require(s.get('actor_role_sha256')==old.ROLE_SHA256[role],'search role SHA')
    require(s.get('initial_structure_sha256')==initial_sha and s.get('fit_target_sha256')==target_sha,'search source/target SHA')
    require(s.get('budget')==BUDGET,'search budget contract')
    require(s.get('search_uses_fit_only') is True and s.get('calibration_eligibility_validation_used') is False,'search leakage')
    require(s.get('complete_incumbent') is True and s.get('execution_action_override') is False,'search completeness/action authority')
    require(s.get('zero_actor_updates') is True and s.get('zero_environment_resets') is True,'search update/reset')
    n=s.get('new_leaf_calls'); attempts=s.get('leaf_api_attempts')
    require(integer(n,1,19968) and isinstance(attempts,list) and len(attempts)==n,'search actual API ledger length')
    solves=failed=0
    for i,e in enumerate(attempts,1):
        legal=e.get('path_legal_actions'); count=e.get('actual_convex_solver_calls')
        require(e.get('call')==i and e.get('cache_miss') is True,'search API sequence/cache ledger')
        require(isinstance(legal,list) and legal==sorted(set(legal)) and 4 in legal
                and all(integer(a,0,4) for a in legal),'search path legal-action set')
        require(integer(count,0,len(legal)) and type(e.get('success')) is bool,'search solver attempts')
        require(integer(e.get('rows')) and isinstance(e.get('rows_sha256'),str) and len(e['rows_sha256'])==64,'search rows ledger')
        if e['success']:require(count==len(legal),'successful API did not enumerate all legal actions')
        solves+=count;failed+=not e['success']
    require(solves==s.get('actual_convex_solver_calls') and failed==s.get('failed_leaf_api_attempts'),'search aggregate solver ledger')
    require(s.get('reserved_final_leaf_calls')==32 and s.get('maximum_new_leaf_calls_including_final_reserve')==20000
            and s.get('new_leaf_calls_remaining')==19968-n,'search reserve ledger')
    require(finite(s.get('elapsed_seconds')) and 0<=s['elapsed_seconds']<=4800+WALL_OVERSHOOT_SECONDS,'search wall exceeds bound')
    rounds=s.get('rounds');require(isinstance(rounds,list) and len(rounds)<=512,'search round ceiling')
    previous=0
    for i,r in enumerate(rounds,1):
        require(r.get('round')==i and r.get('recipe')==(i-1)%8 and integer(r.get('new_leaf_calls'),previous,n),'search round/recipe/call sequence')
        previous=r['new_leaf_calls']
    require(s.get('stop_reason') in ('wall_budget','leaf_call_budget','three_complete_recipe_cycles_without_improvement','maximum_rounds'),'unknown search stop')
    return {'new_leaf_calls':n,'convex_solver_calls':solves,'failed_leaf_calls':failed,
            'search_elapsed_seconds':s['elapsed_seconds'],'stop_reason':s['stop_reason']}


def validate_timeout_failure(result,search,role):
    """Require actual, bounded timeout evidence before treating it as failure."""
    require(result.get('status')=='scientific_budget_exhausted','timeout status')
    f=result.get('failure',{})
    require(result.get('gate')=={'passed':False} and f.get('status')=='budget_exhausted'
            and f.get('scientific_failure') is True and f.get('actor_role_sha256')==old.ROLE_SHA256[role],
            'timeout failure identity/evidence')
    wall=result.get('elapsed_seconds')
    require(finite(wall) and 5400<=wall<=5400+WALL_OVERSHOOT_SECONDS
            and isinstance(f.get('failure_stage'),str) and bool(f['failure_stage']),
            'timeout actual wall/stage evidence')
    led=search_ledger(search,role,search['initial_structure_sha256'],search['fit_target_sha256'])
    b=f.get('fit_budget',{});n=f.get('leaf_optimization_calls');sol=f.get('convex_solver_calls')
    require(integer(n,0,32) and integer(sol,0,5*n) and b.get('prior_leaf_optimization_calls')==led['new_leaf_calls']
            and b.get('actual_final_leaf_optimization_calls')==n
            and b.get('actual_total_leaf_optimization_calls')==led['new_leaf_calls']+n<=20000,
            'timeout API evidence')
    require('metrics' not in result and 'program' not in result,'timeout claims usable evidence')
    return led


def reconstruct_summary(stage,plansha,rows):
    """Build native summary solely from the six audited scientific results."""
    require(stage in ('S1','S2'),'summary stage')
    expected_names=[j['job_id'] for j in v.jobs()]
    require(isinstance(rows,list) and [r.get('job_id') for r in rows]==expected_names,
            'summary must contain exactly six unique ordered jobs')
    for row in rows:
        require(type(row.get('passed')) is bool and row.get('status') in ('completed','scientific_budget_exhausted'),
                'summary scientific status/gate')
        require(row['status']=='completed' or row['passed'] is False,'incomplete summary claims pass')
    allpass=all(r['status']=='completed' and r['passed'] for r in rows)
    nextstage='fresh_confirmation' if allpass else 'S2' if stage=='S1' else 'closed_failed_program_precheck'
    return dict(ZERO,schema='warehouse-v623-refinement-summary.v1',candidate_id=v.CANDIDATE,
        stage=stage,plan_sha256=plansha,source_sha256=EXPECTED_SOURCE,
        all_six_pass=allpass,next_stage=nextstage,role_folds=rows)


def verify_summary(actual,expected):
    """Do not trust the summary's own pass flags or embedded evidence paths."""
    math.assert_equal(expected,actual,'independently_reconstructed_summary')


def audit(stage,with_summary):
    begin=time.monotonic();own=ref(__file__)
    require(ROOT==v.ROOT,'script root resolution')
    require(v.source_sha()==EXPECTED_SOURCE,'frozen source changed')
    folder=v.folder(stage);planpath=folder/'plan.json';plan=obj(planpath);plansha=READ[str(planpath.resolve())]
    require(obj(folder/'plan.claim.json')=={'sha256':plansha,'single_use':True},'plan claim')
    require(plan.get('schema')=='warehouse-v623-refinement-plan.v1' and plan.get('stage')==stage
            and plan.get('candidate_id')==v.CANDIDATE,'plan identity')
    require(plan.get('source_sha256')==EXPECTED_SOURCE and plan.get('source_files')==v.source_files()
            and plan.get('recipe_sha256')==v.recipe() and plan.get('runtime')==runtime_identity(),'plan source/recipe/runtime')
    require(plan.get('jobs')==v.jobs() and plan.get('folds')==old.grouped_folds()
            and plan.get('scientific_retry') is False,'job/split/retry contract')
    zero(plan)
    for relative,digest in plan['source_files'].items():
        raw(ROOT/relative,digest);raw(folder/'source'/EXPECTED_SOURCE/relative,digest)
    actor,actorpath=old.checked_actor(ROOT)
    require(plan['actor']==ref(actorpath) and plan['actor']['sha256']==old.ACTOR_SHA256
            and plan.get('actor_role_sha256')==old.ROLE_SHA256,'pinned actor binding')
    episodes,manifest=old.checked_inputs(ROOT)
    require(manifest==plan['inputs'] and old.target_identity(actor,episodes)==plan['targets'],'input/NN target binding')
    for p,h in manifest['files_sha256'].items():raw(p,h)
    require(plan['prior_A32']=={j['job_id']:v.checked_prior(j) for j in v.jobs()},'prior A32 binding')
    for key in ('failed','plan'):
        pointer=plan['engineering_parent'][key];obj(pointer['path'],pointer['sha256'])
    rows=[];jobs=[]
    for spec in v.jobs():
        name=spec['job_id'];p=folder/'jobs'/name
        require(not (p/'failed.json').exists(),'engineering failure '+name)
        result=obj(p/'result.json');completion=obj(p/'completed.json');started=obj(p/'started.json')
        zero(result);zero(started)
        require(started==dict(ZERO,job=spec,stage=stage,plan_sha256=plansha),'started binding '+name)
        require(result.get('schema')=='warehouse-v623-refinement-result.v1'
                and result.get('job')==spec and result.get('stage')==stage
                and result.get('plan_sha256')==plansha,'result binding '+name)
        require(completion=={'plan_sha256':plansha,'result_sha256':READ[str((p/'result.json').resolve())],
                            'passed':result['gate']['passed']},'completion binding '+name)
        require(finite(result.get('elapsed_seconds')) and 0<=result['elapsed_seconds']<=5400+WALL_OVERSHOOT_SECONDS,'whole-job elapsed exceeds audit envelope '+name)
        if result.get('status')=='scientific_budget_exhausted':
            # Strong evidence is mandatory. Insufficient timeout evidence blocks
            # transition; it cannot silently become a valid scientific failure.
            s=obj(p/'search-report.json')
            led=validate_timeout_failure(result,s,spec['role'])
            passed=False;validation=None
            jobs.append({'job':spec,'status':result['status'],'gate_passed':False,'ledger':led})
        else:
            require(result.get('status')=='completed','unknown scientific status '+name)
            require(result.get('actor')==plan['actor'],'result actor '+name)
            prior=plan['prior_A32'][name]['structure'];initial=bound(result['initial_structure'],prior['path'])
            require(result['initial_structure']==prior,'initial structure pointer '+name)
            rawprogram=bound(result['program'],p/'program.json')
            structure=bound(result['structure'],p/'structure.json')
            s=bound(result['search_report'],p/'search-report.json')
            require(s.get('mode')==stage and s.get('method')=={'S1':'v623_a32_persistent_fit_search_v1','S2':'v623_a32_subtree_refinement_v1'}[stage],'search method '+name)
            names=tuple(actor.roles[spec['role']].feature_names)
            tree,capacity=math.structure(structure,names,enforce_limits=False)
            initial_tree,_=math.structure(initial,names,enforce_limits=False)
            canonical_initial=sha256(json.dumps(math.plain(initial_tree),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            # Canonical scalar tree representation is shared with the original
            # normalization contract; identity must equal pinned original search.
            require(result['initial_structure_sha256']==s['initial_structure_sha256']==canonical_initial,'initial normalized SHA '+name)
            priorsearch=obj(plan['prior_A32'][name]['search_report']['path'],plan['prior_A32'][name]['search_report']['sha256'])
            require(result['initial_structure_sha256']==priorsearch.get('structure_sha256',result['initial_structure_sha256']), 'prior normalized topology SHA '+name)
            require(capacity['depth']<=6 and capacity['leaf_count']<=32 and capacity['maximum_source_features_per_node']<=3,'complexity '+name)
            require(rawprogram.get('version')=='warehouse-role-program.v6.22-depth6-leaves32'
                    and rawprogram.get('method')=='v620_m2_expanded_distribution_refit_v622'
                    and rawprogram.get('role')==spec['role'] and rawprogram.get('actor_role_sha256')==old.ROLE_SHA256[spec['role']]
                    and rawprogram.get('feature_schema_sha256')==FEATURE_SCHEMA_SHA256
                    and rawprogram.get('action_physics_config_sha256')==ACTION_PHYSICS_CONFIG_SHA256,'program identity '+name)
            contexts={};fold=plan['folds'][spec['fold']]
            for split in v.SPLITS:
                rr=[r for case in fold['splits'][split] for r in episodes[case['base']][spec['role']]]
                c=context(rawprogram,encoded_data(actor,spec['role'],rr),names);contexts[split]=c
                for purpose in ('feedback','explanation'):
                    math.assert_equal(section(c,rawprogram[purpose+'_leaves'],purpose),result['metrics'][split][purpose],name+'.'+split+'.'+purpose)
            embedded,_=math.structure(rawprogram['structure'],names,enforce_limits=False)
            math.assert_equal(tree,embedded,name+'.embedded_structure')
            r=rawprogram['report'];target=sha256(np.ascontiguousarray(contexts['fit']['p'],dtype='<f4').tobytes()).hexdigest()
            led=search_ledger(s,spec['role'],result['initial_structure_sha256'],target)
            require(r.get('fit_target_sha256')==target and r.get('full_NN_distribution_targets') is True
                    and r.get('external_program_selects_execution_action') is False and r.get('excluded_frames')==0,'full target report '+name)
            for purpose,threshold in (('feedback',.93),('explanation',.95)):
                expected=[i for i in range(capacity['leaf_count']) if (contexts['fit']['leaves']==i).any()
                          and math.eligible(section(contexts['eligibility'],[i],purpose),threshold)]
                math.assert_equal(expected,rawprogram[purpose+'_leaves'],name+'.qualified.'+purpose)
            fits=r.get('leaf_fits');n=capacity['leaf_count'];require(isinstance(fits,dict) and set(fits)=={str(i) for i in range(n)},'final leaf fit coverage '+name)
            finalsolves=0
            for i in range(n):
                fit=fits[str(i)];proof=contexts['fit']['proof'][i];chosen=int(np.asarray(rawprogram['leaf_logits'][i],np.float32).argmax())
                require(chosen in proof,'refitted action path proof '+name)
                candidates=fit.get('candidate_reports');require(isinstance(candidates,list) and len(candidates)==fit.get('enumerated_action_solves')==len(proof),'final legal action solver enumeration '+name)
                require(sorted(z['action'] for z in candidates)==sorted(proof),'final solver action set '+name)
                require(fit.get('success') is True and fit.get('float32_feasible') is True,'final solve failure '+name)
                actual_rows=int((contexts['fit']['leaves']==i).sum())
                require(fit.get('row_count')==actual_rows and fit.get('physical_fit_frame_count')==actual_rows,'final leaf row ledger '+name)
                finalsolves+=len(candidates)
            b=r['fit_budget'];searchcalls=led['new_leaf_calls']
            require(r.get('leaf_optimization_calls')==r.get('leaf_optimization_calls_outside_search')==n
                    and r.get('convex_solver_calls')==finalsolves,'final API/solver aggregate '+name)
            require(b.get('prior_leaf_optimization_calls')==searchcalls
                    and b.get('actual_final_leaf_optimization_calls')==n
                    and b.get('actual_total_leaf_optimization_calls')==searchcalls+n<=20000
                    and b.get('max_leaf_optimization_calls')==32 and b.get('max_total_leaf_optimization_calls')==20000
                    and b.get('provenance_sha256')==result['search_report']['sha256'],'combined API ledger '+name)
            require(result.get('search_calls')==searchcalls and result.get('final_calls')==n,'result actual calls '+name)
            require(result.get('budget_checks')==dict.fromkeys(('all_final_leaves_refitted','final_calls_at_most_32','no_actor_change',
                    'search_calls_with_reserved_final','total_new_calls_at_most_20000','wall_at_most_90_minutes'),True),'production budget checks '+name)
            validation={purpose:section(contexts['validation'],rawprogram[purpose+'_leaves'],purpose) for purpose in ('feedback','explanation')}
            passed=independent_gate(validation);require(type(result['gate']['passed']) is bool and result['gate']['passed']==passed,'recomputed gate '+name)
            jobs.append({'job':spec,'status':'completed','gate_passed':passed,'ledger':dict(led,final_leaf_calls=n,final_convex_solver_calls=finalsolves,total_new_leaf_calls=searchcalls+n),
                         'validation':{q:{k:validation[q][k] for k in ('coverage','expected_actionable_coverage','agreement','kl','whole_tree_raw_illegal_count','whole_tree_path_uncertified_count')} for q in ('feedback','explanation')}})
        rows.append({'job_id':name,'status':result['status'],'passed':passed,
            'explanation_coverage':validation['explanation']['coverage'] if validation else None,
            'expected_feedback_coverage':validation['feedback']['expected_actionable_coverage'] if validation else None,
            'result':ref(p/'result.json')})
    expected_summary=reconstruct_summary(stage,plansha,rows)
    allpass=expected_summary['all_six_pass'];nextstage=expected_summary['next_stage']
    if with_summary:verify_summary(obj(folder/'summary.json'),expected_summary)
    require(v.source_sha()==EXPECTED_SOURCE,'source changed during supplementary audit')
    for p,h in list(READ.items()):require(sha256(Path(p).read_bytes()).hexdigest()==h,'evidence changed during supplementary audit '+p)
    return dict(ZERO,schema='warehouse-v623-supplemental-audit.v1',stage=stage,candidate_id=v.CANDIDATE,
        audit_script=own,source_sha256=EXPECTED_SOURCE,plan_sha256=plansha,passed=True,
        all_six_scientific_gates_pass=allpass,independently_reconstructed_next_stage=nextstage,
        summary_verified=with_summary,wall_overshoot_tolerance_seconds=WALL_OVERSHOOT_SECONDS,
        elapsed_seconds=time.monotonic()-begin,at_utc=datetime.now(timezone.utc).isoformat(),
        jobs=jobs,evidence_sha256=dict(sorted(READ.items())),
        scope='spent-development-only; independent metrics, qualifications, paths, actual API/solver ledger, plan/completion/source binding; no fitting or acceptance evidence')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('S1','S2'),required=True)
    parser.add_argument('--summary',action='store_true');parser.add_argument('--receipt');args=parser.parse_args()
    target=Path(args.receipt) if args.receipt else v.folder(args.stage)/('postsummary-supplemental-audit.json' if args.summary else 'pretransition-audit.json')
    try:receipt=audit(args.stage,args.summary)
    except Exception as e:
        receipt=dict(ZERO,schema='warehouse-v623-supplemental-audit.v1',stage=args.stage,passed=False,
                     audit_script=ref(__file__),error_type=type(e).__name__,error=str(e),at_utc=datetime.now(timezone.utc).isoformat())
    with target.open('x') as f:json.dump(receipt,f,sort_keys=True,separators=(',',':'),allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('passed','stage','all_six_scientific_gates_pass','independently_reconstructed_next_stage','error') if k in receipt},sort_keys=True),flush=True)
    if not receipt['passed']:raise SystemExit(1)
if __name__=='__main__':main()
