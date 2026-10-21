"""Conditional frozen-output confidence-partition study, no NN authority."""
from pathlib import Path
import json,subprocess
from . import v629_protocol as previous,v628_protocol as data_protocol
from .v626_protocol import require,pointer,bound,digest,file_sha

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE='warehouse-v630-stable-partitions-01'
OUTPUT=ROOT/'output/warehouse_v630'/CANDIDATE
FILES=('backend/training/warehouse_v630_certificate.py','backend/training/warehouse_v630_program.py',
    'backend/training/warehouse_v630_partition_proposals.py','backend/training/warehouse_v630_partition_search.py',
    'experiments/warehouse/v630_semantic_protection.py','experiments/warehouse/v630_protocol.py',
    'scripts/run_warehouse_v630.py','scripts/audit_warehouse_v630.py','configs/warehouse_v630_recipe.json',
    'docs/warehouse_v630_plan.md','tests/test_warehouse_v630_program.py','tests/test_warehouse_v630_partition_proposals.py',
    'tests/test_warehouse_v630_partition_search.py','tests/test_warehouse_v630_protocol.py','tests/test_warehouse_v630_pipeline.py')


def recipe():
    return dict(schema='warehouse-v630-stable-partition-recipe.v1',candidate_id=CANDIDATE,
        method='fit_only_stable_confidence_partitions_frozen_distributions_v1',
        arms=[dict(name='C0',scope='control'),dict(name='P',scope='protected'),dict(name='A',scope='all')],
        initial='fixed_G128_v629_inner_search_best_for_every_role_fold_after_all_24_settled',
        outer_refitted_source_forbidden=True,maximum_leaves=256,maximum_depth=9,source_features_per_node=3,
        source_features_per_leaf=3,maximum_leaf_coefficients=20,all_source_surrogate_distributions_frozen=True,
        descendant_qualification_inherited=False,additional_classifier_depth=3,classifier='axis_aligned_entropy_CART',
        classifier_min_samples_leaf=40,classifier_error_class_weight=4.,classifier_random_state=0,
        classifier_random_state_is_fixed_engineering_control=True,
        data='same_spent_1650_v628_development_scenes_not_new_confirmation',
        confidence_fit_sampled_scenes=300,risk_sampled_scenes=150,selection_sampled_scenes=150,
        lane_rule='sorted_outer_fit_sampled_case_ids_per_task_100_50_50',argmax_confidence_frames=0,
        reliability=dict(subsets=30,scenes_per_task=30,minimum_passes=27,
            subset_rule='SHA256_of_method_repeat_and_original_episode_ID_no_new_RNG',
            feedback_leaf_threshold=.93,explanation_leaf_threshold=.95,
            selection_feedback_agreement=.95,selection_explanation_agreement=.97,selection_KL=.20),
        pruning='stop_or_keep_both_children_no_promotion_all_empty_leaves_count',
        allocation=dict(weights=[0.,.5,1.,2.],maximum_leaves=256,
            additive_primary='E/(.45*N)+weight*(F/(.45*N)+expected_F/(.40*N))',
            secondary=['E','expected_F','F'],scope='only_fixed_options_and_frozen_additive_tables',
            nonlinear_global_optimality_gap='unknown',candidate_selection='reliable_proxy_pass_then_same_v629_strict_selection_score'),
        limits=dict(leaf=0,node=256,proof=0),search_seconds=1200,final_seconds=600,total_seconds=1800,
        final_refit=False,final_temperature_calibration=False,final_qualification='original_outer_fit900_cal150_elig300_valid300',
        gate=previous.recipe()['gate'],maximum_workers=3,BLAS_threads=1,automatic_retry=False,
        all_18_settled_before_selection=True,all_18_completed_before_selection=True,
        selection='first_P_then_A_with_all_six_strict_gates_and_reliable_proxy',
        conditional_start='only_after_v629_all_24_reaped_integrity_audited_and_closed_without_winner',
        if_v629_passes='skip_new_candidate_and_continue_original_winner',
        fresh_confirmation_required=True,independent_final_audit_used=False,heldout_used=False,
        actor_updates=0,PPO_joint_steps=0,environment_resets=0,training_authority=False,pilot_opened=False,
        performance=previous.recipe()['performance'],release_id=previous.recipe()['release_id'])


def source_files():
    files=previous.source_files()
    require(digest(files)=='4cd4de16e45d76026ee76dc22ef2c3d42c8f54c8652ad17dce5f757a7d04723f','immutable v629 source changed')
    for name in FILES:files[name]=file_sha(ROOT/name)
    return dict(sorted(files.items()))


def jobs():
    return [dict(job_id=f'{arm["name"]}-fold-{fold}-{role}',fold=fold,role=role,**arm)
        for fold in range(3) for role in ('robot_1','robot_2') for arm in recipe()['arms']]


def process_identity(pid):
    if type(pid) is not int or pid<=0:return None
    result=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None


def runtime():
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    import sklearn
    return {**runtime_identity(),'sklearn':sklearn.__version__}


def prepare(wait_pid=None):
    path,old=previous.checked_plan();identity=process_identity(wait_pid)
    if identity is None:require((previous.OUTPUT/'summary.json').is_file(),'need the verified live v629 driver or its terminal summary')
    else:require('run_warehouse_v629.py --pipeline' in identity,'dependency is not the original v629 supervisor')
    require(json.loads((ROOT/'configs/warehouse_v630_recipe.json').read_bytes())==recipe(),'v630 recipe file differs')
    files=source_files();sha=digest(files);OUTPUT.mkdir(parents=True,exist_ok=False)
    for name,expected in files.items():
        target=OUTPUT/'source'/sha/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==expected,'source changed before v630 snapshot')
        target.write_bytes(raw);target.chmod(0o444)
    plan=dict(schema='warehouse-v630-plan.v1',candidate_id=CANDIDATE,recipe=recipe(),jobs=jobs(),source_files=files,source_sha256=sha,
        previous_plan=pointer(path),collection=old['collection'],actor=old['actor'],actor_role_sha256=old['actor_role_sha256'],
        common_quality=pointer(data_protocol.OUTPUT/'common-NN-quality.json'),
        dependency_process=None if identity is None else dict(pid=wait_pid,identity=identity),runtime=runtime(),
        actor_updates=0,PPO_joint_steps=0,environment_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    from scripts.run_warehouse_v625 import write_new
    write_new(OUTPUT/'plan.json',plan);(OUTPUT/'plan.json').chmod(0o444)
    write_new(OUTPUT/'plan.claim.json',dict(sha256=file_sha(OUTPUT/'plan.json'),single_use=True));(OUTPUT/'plan.claim.json').chmod(0o444)
    return checked_plan()[1]


def checked_plan():
    path=OUTPUT/'plan.json';plan=bound(pointer(path))
    require(bound(pointer(OUTPUT/'plan.claim.json'))==dict(sha256=file_sha(path),single_use=True),'v630 plan seal differs')
    old_path,old=previous.checked_plan()
    require(plan['candidate_id']==CANDIDATE and plan['source_files']==source_files()
        and plan['source_sha256']==digest(source_files()) and plan['recipe']==recipe() and plan['jobs']==jobs()
        and plan['previous_plan']==pointer(old_path) and plan['collection']==old['collection'] and plan['actor']==old['actor']
        and plan['actor_role_sha256']==old['actor_role_sha256'] and plan['runtime']==runtime(),'v630 source/config/inputs changed')
    for name,sha in plan['source_files'].items():
        snapshot=OUTPUT/'source'/plan['source_sha256']/name
        require(file_sha(snapshot)==sha and not snapshot.stat().st_mode&0o222,'v630 source snapshot changed')
    for key in ('actor','collection','common_quality'):require(pointer(plan[key]['path'])==plan[key],'v630 bound asset changed')
    return path,plan


def old_completion():
    from scripts.run_warehouse_v629 import gate as old_gate
    _,old_plan=previous.checked_plan()
    summary_ref=pointer(previous.OUTPUT/'summary.json');summary=bound(summary_ref)
    execution_ref=pointer(previous.OUTPUT/'execution/completed.json');execution=bound(execution_ref)
    require(summary['all_24_settled'] and execution['all_owned_workers_waited'] and execution['jobs']==24
        and len(execution['exits'])==24 and all(row['waited'] for row in execution['exits']),'v629 did not settle and reap all 24 workers')
    expected={j['job_id'] for j in previous.jobs()}
    require({r['job']['job_id'] for r in execution['exits']}==expected,'v629 completion job identities differ')
    integrity=bound(summary['independent_audit']);require(integrity['integrity_passed'] is True,'upstream deterministic audit failed')
    require([r['job'] for r in integrity['jobs']]==old_plan['jobs'],'upstream audit did not cover every ordered job')
    results=[];asset_refs=[]
    for job,checked in zip(old_plan['jobs'],integrity['jobs'],strict=True):
        folder=previous.OUTPUT/'jobs'/job['job_id'];ref=pointer(folder/'result.json');result=bound(ref)
        seal=pointer(folder/'completed.json')
        require(bound(seal)==dict(result_sha256=ref['sha256']) and result['job']==job and checked['status']==result['status'],
            'upstream terminal result or audit coverage changed')
        asset_refs.extend([ref,seal]);results.append(result)
        if result['status']=='completed':
            require(pointer(result['program']['path'])==result['program'] and checked['program']==result['program']
                and all(checked.get(k) is True for k in ('protected_exact_forward','independent_masks_and_duals','independent_reference_metrics')),
                'upstream audited program changed or lost evidence')
            require(result['gate']==old_gate(result['metrics']['validation'],bound(result['search_report'])),
                'upstream verdict does not reproduce from its stored metrics')
            asset_refs.extend([result['program'],result['search_report']])
    passing=[n for n in (64,128,256) if all(r['status']=='completed' and r['gate']['passed']
        for r in results if r['job']['name']==f'G{n}')]
    expected=min(passing) if passing else None
    require(summary['scientific_pass']==(expected is not None) and summary['selected_capacity']==expected,
        'upstream winner/closure is not the derived all-six verdict')
    for row in execution['exits']:
        identity=process_identity(row['pid'])
        require(identity is None or 'run_warehouse_v629.py --job' not in identity,'old v629 optimizer still live')
    refs=dict(summary=summary_ref,execution=execution_ref,independent_audit=summary['independent_audit'],terminal_assets=asset_refs)
    if summary['scientific_pass']:
        require(summary['selected_capacity'] in (64,128,256),'passing upstream summary lacks a winner')
        return dict(skip_new_candidate=True,reason='original_v629_winner_has_priority',**refs)
    closed=pointer(previous.OUTPUT/'candidate.closed.json')
    require(bound(closed)==dict(status='closed_failed_program_precheck',summary=summary_ref,scientific_pass=False,
        fresh_confirmation_used=False,pilot_opened=False) and summary['selected_capacity'] is None,'upstream failed-candidate seal differs')
    return dict(skip_new_candidate=False,closed=closed,**refs)


def source_program(plan,fold,role):
    from backend.training.warehouse_v629_program import Program
    folder=previous.OUTPUT/'jobs'/f'G128-fold-{fold}-{role}'
    result_ref=pointer(folder/'result.json');result=bound(result_ref)
    require(bound(pointer(folder/'completed.json'))==dict(result_sha256=result_ref['sha256'])
        and result['job']==next(j for j in previous.jobs() if j['job_id']==folder.name),'G128 source terminal seal differs')
    ref=pointer(folder/'search-best.json')
    if result['status']=='completed':require(result['search_best']==ref,'source is not the sealed inner-search output')
    p=Program.from_dict(bound(ref),plan['actor_role_sha256'][role])
    old=bound(plan['previous_plan']);data_plan=bound(old['previous_plan']);expected=data_plan['folds'][fold]['inner']
    by_id={c['case_id']:c for c in data_plan['cases']}
    require(p.role==role and p.maximum_leaves==128 and p.report is not None
        and p.report['split_episode_ids']=={s:sorted(ids) for s,ids in expected.items()}
        and p.report['split_scenario_ids']=={s:sorted(str(by_id[i]['base']) for i in ids) for s,ids in expected.items()},
        'G128 source contains outer qualification or a different role/capacity')
    return p,dict(program=ref,terminal=result_ref,completed=pointer(folder/'completed.json'))


def fit_data(plan,fold,role):
    import numpy as np
    old=bound(plan['previous_plan']);data_plan=bound(old['previous_plan']);receipt=bound(plan['collection'])
    by_id={c['case_id']:c for c in data_plan['cases']}
    cases=[by_id[i] for i in data_plan['folds'][fold]['outer']['fit'] if by_id[i]['sampling_mode']=='sampled']
    require(len(cases)==600,'incorrect frozen confidence pool')
    index=('robot_1','robot_2').index(role);arrays={k:[] for k in ('x','m','p','sampled')};rows=[];refs=[]
    for case in cases:
        ref=receipt['cases'][case['case_id']]['arrays'];require(pointer(ref['path'])==ref,'fit target arrays changed');refs.append(ref)
        with np.load(ref['path'],allow_pickle=False) as z:
            for key in arrays:arrays[key].append(z[key][:,index].copy())
            rows.extend(dict(episode=case['case_id'],base_seed=str(case['base']),task=case['task']) for _ in range(len(z['x'])))
    result={k:np.concatenate(a) for k,a in arrays.items()};result['rows']=rows
    return result,cases,refs


def common_quality(plan):
    receipt=bound(plan['collection']);stored=bound(plan['common_quality']);quality={}
    require(receipt['actor']==plan['actor'] and stored['collection']==plan['collection']
        and receipt['actual_reset_count']==1650 and len(receipt['outcomes'])==1650
        and receipt['post_policy_interventions']==receipt['actor_updates']==receipt['PPO_joint_steps']==0,
        'quality source is not the pure frozen paired NN collection')
    for mode,count in [('argmax',300),('sampled',1350)]:
        rows=[r for r in receipt['outcomes'] if r['case']['sampling_mode']==mode];require(len(rows)==count,'NN quality cohort size differs')
        shutdown=sum(r['shutdown'] for r in rows)
        quality[mode]=dict(episodes=count,deliveries=sum(r['deliveries'] for r in rows)/count,
            collisions=sum(r['robot_collisions'] for r in rows)/count,shutdown_episodes=shutdown,shutdown_rate=shutdown/count)
    require(stored['quality']==quality,'NN quality summary does not reproduce from actual complete episode outcomes')
    return quality
