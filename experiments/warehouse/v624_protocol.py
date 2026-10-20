"""Immutable paired S1-topology search comparison; spent development only."""
from hashlib import sha256
import json
import math
from pathlib import Path

from experiments.warehouse import v623_protocol as previous
from experiments.warehouse import v622_recovery_protocol as old

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE='warehouse-v624-s1-upgrade-01'
OUTPUT=ROOT/'output/warehouse_v624'/CANDIDATE/'stage-A'
RECIPE='configs/warehouse_v624_recipe.json'
PREVIOUS_SOURCE='03cd01e2be51792c67e51c3b3a3f1cd0c9110c8e69630cb6a46ef973f5bb393c'
PREVIOUS_ROOT='output/warehouse_v623/warehouse-v623-a32-refine-01/recovery-02'
PREVIOUS_RECEIPT_SHA='881aab18714c0667bad47bb6d3b9c8b246390ed78cc5bc05d5090f58dc32ac8a'
PREVIOUS_MANIFEST_SHA='56952ca25d2335a3c6282e27317dcc5d33b35597cf98e7e54333358cab4d1ad3'
S1_SHA={'summary':'c71c9296c8f9695d0872ad4cc3f5f39b4ab490769f814d2983646e739f826aa9',
        'native_audit':'5dc1ab93f72e4a49a3e45840624462809b3fe3bf22040a54418cea2e313a65fc',
        'supplemental_audit':'6a2a433d7a430992ae10159ba0796fe4bcc3a5f91c9dbd8bbebf49a61a380e77'}
NEW_FILES=(RECIPE,'experiments/warehouse/v624_protocol.py',
    'backend/training/warehouse_v624_search.py','scripts/run_warehouse_v624_upgrade.py',
    'scripts/audit_warehouse_v624_upgrade.py','scripts/finalize_warehouse_v624_upgrade.py',
    'scripts/audit_warehouse_v623_refinement.py','scripts/finalize_warehouse_v623_refinement.py',
    'scripts/verify_warehouse_v624_engineering.py')
SPLITS=old.SPLITS
ROLES=old.ROLES
ARMS=('P0','P1')
ZERO={'environment_resets':0,'actor_updates':0,'PPO_joint_steps':0,
      'pilot_opened':False,'publishable':False,'sealed_audit_used':False,'heldout_used':False}
BUDGET={'maximum_new_leaf_calls':20000,'final_leaf_reserve':32,
        'search_wall_seconds':4800,'final_wall_seconds_reserved':600}
GATES={'minimum_feedback_coverage':.45,'minimum_explanation_coverage':.45,
       'minimum_expected_feedback_coverage':.40,'minimum_feedback_agreement':.95,
       'minimum_explanation_agreement':.97,'maximum_qualified_KL':.20,
       'maximum_raw_illegal_actions':0,'maximum_uncertified_paths':0,
       'all_six_role_folds_required':True}


def require(condition,message):
    if not condition:raise RuntimeError(message)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'missing/symlink scientific asset '+str(path))
    return sha256(path.read_bytes()).hexdigest()


def pointer(path):return {'path':str(Path(path).resolve()),'sha256':file_sha(path)}


def bound(ref,expected_path=None):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'invalid asset pointer')
    if expected_path is not None:require(ref['path']==str(Path(expected_path).resolve()),'asset path mismatch')
    require(file_sha(ref['path'])==ref['sha256'],'asset SHA mismatch '+ref['path'])
    return json.loads(Path(ref['path']).read_bytes())


def recipe(root=ROOT):
    from dataclasses import asdict
    from backend.training.warehouse_v624_search import search_configuration,UpgradeBudget
    raw=(Path(root)/RECIPE).read_bytes();r=json.loads(raw)
    require(r.get('search_configuration')==search_configuration()
        and r.get('production_budget')==asdict(UpgradeBudget()),'search algorithm configuration changed')
    require(r.get('schema')=='warehouse-v624-s1-upgrade-recipe.v1' and r.get('candidate_id')==CANDIDATE
        and r.get('stage')=='A' and r.get('arms')==list(ARMS) and r.get('precheck')==GATES
        and r.get('roles')==list(ROLES) and r.get('folds')==3
        and r.get('initialization')=='paired_fresh_search_from_identical_pinned_S1_topology'
        and r.get('checkpoint')=='complete_safe_boundary_state_only_explicit_resume_no_inflight_retry'
        and r.get('selection_order')==['worst_explanation_coverage','worst_expected_feedback_coverage','negative_worst_qualified_KL','negative_total_leaves','P0_tie_preference']
        and r.get('spent_development_episodes')==275 and r.get('maximum_depth')==6
        and r.get('maximum_leaves')==32 and r.get('maximum_public_source_features_per_node')==3
        and r.get('all_guards_and_empty_branches_count') is True
        and r.get('all_twelve_complete_before_selection') is True
        and r.get('automatic_retry') is False and r.get('future_confirmation',{}).get('authorized_by_stage_A_recipe') is False,
        'v624 recipe scientific contract mismatch')
    require(all(r.get(k)==val and type(r.get(k)) is type(val) for k,val in dict(ZERO,**BUDGET).items()),'v624 recipe budget/authority mismatch')
    future=r['future_confirmation']
    require(r.get('maximum_workers')==3 and r.get('BLAS_threads_per_worker')==1
        and future.get('individual_one_sided_lower_confidence_level')==.9875
        and future.get('simultaneous_confidence_level')==.95
        and future.get('primary_comparisons')==['robot_1_explanation_coverage','robot_2_explanation_coverage','robot_1_expected_feedback_coverage','robot_2_expected_feedback_coverage']
        and future.get('paired_bootstrap_unit')=='complete_scenario_stratified_by_task'
        and future.get('fresh_episodes')=={'fit':100,'calibration':25,'eligibility':75,'validation':75}
        and future.get('fit_argmax_episodes')==50 and future.get('pressure_scenarios')==72
        and future.get('pressure_trajectories_per_scenario')==3,'future confirmation registration changed')
    return sha256(raw).hexdigest()


def source_files(root=ROOT):
    root=Path(root).resolve();files=previous.source_files(root)
    require(digest(files)==PREVIOUS_SOURCE,'v623 frozen source changed')
    files.update({relative:file_sha(root/relative) for relative in NEW_FILES})
    return dict(sorted(files.items()))


def source_sha(root=ROOT):return digest(source_files(root))


def jobs():
    rows=[]
    for fold in range(3):
        for role_index,role in enumerate(ROLES):
            arms=ARMS if (fold+role_index)%2==0 else tuple(reversed(ARMS))
            for arm in arms:rows.append({'job_id':f'fold-{fold}-{role}-{arm}','arm':arm,'fold':fold,'role':role})
    return rows


def checked_baseline(root=ROOT):
    root=Path(root);base=root/PREVIOUS_ROOT
    receipt_ref={'path':str((base/'finalizations/completed-development-01/receipt.json').resolve()),'sha256':PREVIOUS_RECEIPT_SHA}
    manifest_ref={'path':str((base/'finalizations/completed-development-01/evidence-manifest.json').resolve()),'sha256':PREVIOUS_MANIFEST_SHA}
    receipt=bound(receipt_ref);manifest=bound(manifest_ref)
    require(receipt.get('candidate_id')==previous.CANDIDATE and receipt.get('status')=='closed_failed_program_precheck'
        and receipt.get('candidate_closed') is True and receipt.get('source_sha256')==PREVIOUS_SOURCE
        and receipt.get('evidence_manifest')==manifest_ref and manifest.get('source_sha256')==PREVIOUS_SOURCE,
        'v623 final closure differs')
    require(manifest.get('actor',{}).get('sha256')==old.ACTOR_SHA256
        and manifest.get('actor_role_sha256')==old.ROLE_SHA256,'baseline Actor differs')
    refs={};paths={'summary':'summary.json','native_audit':'audit.json','supplemental_audit':'tracked-postsummary-supplemental-audit.json'}
    for key,filename in paths.items():
        refs[key]={'path':str((base/'S1'/filename).resolve()),'sha256':S1_SHA[key]};value=bound(refs[key])
        require(manifest['current_scientific_stages']['S1'][key]==refs[key],'S1 baseline absent from final manifest')
        if key=='summary':require(value.get('all_six_pass') is False,'baseline S1 unexpectedly qualified')
        else:require(value.get('passed') is True and value.get('source_sha256')==PREVIOUS_SOURCE,'baseline S1 audit failed')
    initial={}
    for spec in previous.jobs():
        name=spec['job_id'];assets=manifest['current_scientific_stages']['S1']['jobs'][name]
        r=bound(assets['result'],base/'S1/jobs'/name/'result.json')
        completion=bound(assets['completion'],base/'S1/jobs'/name/'completed.json')
        require(r.get('job')==spec and r.get('status')=='completed' and r.get('actor')==manifest['actor']
            and completion.get('result_sha256')==assets['result']['sha256'],'incomplete baseline S1 result')
        refs_initial={'result':assets['result']}
        for key,filename in (('structure','structure.json'),('program','program.json'),('search_report','search-report.json')):
            bound(r[key],base/'S1/jobs'/name/filename);require(r[key]==assets[key],'S1 asset differs from final manifest')
            refs_initial[key]=r[key]
        initial[name]=refs_initial
    return {'finalization_receipt':receipt_ref,'evidence_manifest':manifest_ref,**refs},initial


def _probability(value):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1


def _kl(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def program_gate(metrics):
    f,e=metrics.get('feedback',{}),metrics.get('explanation',{})
    def atleast(row,key,threshold):
        value=row.get(key);return _probability(value) and value>=threshold
    checks={'feedback_coverage':atleast(f,'coverage',.45),'explanation_coverage':atleast(e,'coverage',.45),
        'expected_feedback_coverage':atleast(f,'expected_actionable_coverage',.40),
        'feedback_agreement':atleast(f,'agreement',.95),'explanation_agreement':atleast(e,'agreement',.97),
        'feedback_kl':_kl(f.get('kl')) and f['kl']<=.20,
        'explanation_kl':_kl(e.get('kl')) and e['kl']<=.20}
    for purpose,m in (('feedback',f),('explanation',e)):
        for key in ('whole_tree_raw_illegal_count','whole_tree_path_uncertified_count','illegal_count','path_uncertified_count'):
            checks[purpose+'_'+key]=type(m.get(key)) is int and m[key]==0
    return {'passed':all(checks.values()),'checks':checks}


def comparison(results):
    """Complete twelve-job comparison; only six-pass arms can be selected."""
    require(isinstance(results,list) and len(results)==12 and [r.get('job') for r in results]==jobs(),
            'comparison requires exact twelve unique ordered bound jobs')
    arms={}
    for arm in ARMS:
        rr=[r for r in results if r['job']['arm']==arm];score=[];role_folds=[]
        for r in rr:
            status=r.get('status');require(status in ('completed','scientific_budget_exhausted'),'engineering failure cannot enter scientific comparison')
            if status=='completed':
                gate=program_gate(r['metrics']['validation']);require(r.get('gate')==gate,'job scientific gate differs')
                f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
                leaves=r['leaf_count'];require(type(leaves) is int and 1<=leaves<=32,'invalid final leaf count')
                if (_probability(e.get('coverage')) and _probability(f.get('expected_actionable_coverage'))
                        and _kl(f.get('kl')) and _kl(e.get('kl'))):
                    score.append((e['coverage'],f['expected_actionable_coverage'],max(f['kl'],e['kl']),leaves))
            else:
                require(r.get('gate')=={'passed':False} and 'metrics' not in r and 'program' not in r,'budget failure claims usable program')
            role_folds.append({'job_id':r['job']['job_id'],'status':status,'passed':r['gate']['passed'],
                'explanation_coverage':r.get('metrics',{}).get('validation',{}).get('explanation',{}).get('coverage'),
                'expected_feedback_coverage':r.get('metrics',{}).get('validation',{}).get('feedback',{}).get('expected_actionable_coverage')})
        passed=all(r['status']=='completed' and r['gate']['passed'] for r in rr)
        rank=[min(s[0] for s in score),min(s[1] for s in score),-max(s[2] for s in score),-sum(s[3] for s in score)] if len(score)==6 else None
        arms[arm]={'all_six_pass':passed,'role_folds':role_folds,'selection_score':rank}
    eligible=[arm for arm in ARMS if arms[arm]['all_six_pass']]
    selected=max(eligible,key=lambda arm:(*arms[arm]['selection_score'],arm=='P0')) if eligible else None
    return {'arms':arms,'selected_arm':selected,
        'next_stage':'fresh_confirmation' if selected else 'closed_failed_program_precheck'}
