#!/usr/bin/env python3
"""Twelve matched depth9/depth12 frozen-NN development experiments."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v642_search as search,warehouse_v642_program as language
from backend.training import warehouse_v635_program as previous_language
from scripts.run_warehouse_v636 import Budget
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.audit_warehouse_v642_program import audited_program
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new
from scripts.verify_warehouse_v625_native import real_deadline

VERSION='warehouse-v642-margin-context-rebuild-development.v1'
PARENT=ROOT/'output/warehouse_v640/warehouse-v640-alternating-regions-01'
FILES=('backend/training/warehouse_v642_certificate.py','backend/training/warehouse_v642_guards.py',
    'backend/training/warehouse_v642_program.py','backend/training/warehouse_v642_subtrees.py',
    'backend/training/warehouse_v642_contexts.py','backend/training/warehouse_v642_search.py',
    'scripts/run_warehouse_v642.py','scripts/audit_warehouse_v642_program.py',
    'tests/test_warehouse_v642_capacity.py','docs/warehouse_v642_plan.md')


def sources():
    receipt=v.bound(v.pointer(ROOT/'experiments/warehouse/evidence/warehouse-v640-closed-diagnostic-20261022-01.json'))
    v.require(receipt['status']=='closed_failed_development_precheck' and receipt['independent_replay_integrity_passed'],'need complete audited prior failure')
    files=dict(v.bound(receipt['references']['plan'])['source_files'])
    for name in FILES:files[str(ROOT/name)]=v.file_sha(ROOT/name)
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        p=Path(raw)
        if not p.is_absolute() and not p.is_file():continue
        p=p.resolve()
        if p.suffix=='.py' and p.is_relative_to(ROOT):files[str(p)]=v.file_sha(p)
    for name,sha in files.items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'immutable upstream source changed')
    return dict(sorted(files.items()))


def snapshot_source(source,target,expected_sha):
    v.require(v.file_sha(source)==expected_sha,'source changed before snapshot')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():v.require(v.file_sha(target)==expected_sha,'source alias collides with a different snapshot')
    else:target.write_bytes(source.read_bytes());target.chmod(0o444)


def work(output,spec):
    folder=Path(output)/spec['id'];folder.mkdir(exist_ok=False);planref=v.pointer(Path(output)/'plan.json')
    budget=Budget(folder/'ledger.jsonl',spec['limits']);write_new(folder/'started.json',dict(spec=spec,plan=planref,pid=os.getpid()))
    try:
        _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
        sr=v.bound(spec['source_result']);source=previous_language.Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
        with real_deadline(budget.started,4200):
            outer=data.load_parts(dp,spec['fold'],spec['role']);baseline={s:language.measure(source,a) for s,a in outer.items()}
            v.require(baseline==sr['metrics'],'actual uniform source metrics differ')
            cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=search.old.metrics.lane_parts(a,cases)
            program,report=search.run(source,outer,lanes,budget,maximum_depth=spec['maximum_depth'])
            v.require(report['frozen_leaves']==spec['frozen_leaves'] and [r['root'] for r in report['root_records']]==spec['structural_roots'],'prepared frozen/root inventory changed')
            write_new(folder/'search-report.json',report);budget.final=True
            language.qualify(program,outer);metrics={s:language.measure(program,a) for s,a in outer.items()}
            independent=audited_program(program,outer,metrics)
            take=np.isin(source.leaf_indices(outer['validation']['x']),spec['frozen_leaves'])
            before=source.distributions(outer['validation']['x'],outer['validation']['m']);after=independent['validation']
            v.require(np.array_equal(before[1][take],after[1][take]) and np.array_equal(before[2][take],after[2][take]),'original protected forward changed')
            task_metrics={};a=outer['validation']
            for task in (1,2,3):
                take=np.array([row['task']==task for row in a['rows']],bool);part=search.old.subset(a,take)
                task_metrics[str(task)]=dict(baseline=language.measure(source,part),candidate=language.measure(program,part),
                    old_scenarios=len({row['base_seed'] for row in part['rows']}))
        write_new(folder/'program.json',program.to_dict())
        r=dict(status='completed',spec=spec,plan=planref,program=v.pointer(folder/'program.json'),search_report=v.pointer(folder/'search-report.json'),
            baseline_metrics=baseline,metrics=metrics,task_metrics=task_metrics,numerical_passed=strict(metrics['validation']),
            reliable_fit_only_passed=report['reliable_selection_passed'],development_passed=strict(metrics['validation']) and report['reliable_selection_passed'],
            counts=budget.counts,elapsed_seconds=time.monotonic()-budget.started,independent_program_frames=sum(len(a['x']) for a in outer.values()),
            independent_leaf_certificates=len(program.leaves),original_protected_outputs_unchanged=True,structure_unchanged=False,
            NN_updates=0,PPO_joint_steps=0,new_environment_resets=0,program_training_gradients=0,model_acceptance_passed=False,pilot_opened=False)
    except Exception as error:
        r=dict(status='failed',spec=spec,plan=planref,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            counts=budget.counts,elapsed_seconds=time.monotonic()-budget.started,numerical_passed=False,development_passed=False,pilot_opened=False)
    finally:budget.close()
    for p in folder.iterdir():
        if p.is_file():p.chmod(0o444)
    write_new(folder/'result.json',r);(folder/'result.json').chmod(0o444);return v.pointer(folder/'result.json')


def main(output):
    v.require(not output.exists(),'single-use run already exists');_,old=v.checked_plan();parent=v.bound(v.pointer(PARENT/'result.json'))
    v.require(parent['all_six_settled'] and v.bound(v.pointer(PARENT/'independent-replay.json'))['integrity_passed'],'need completed replayed uniform parent')
    jobs=[]
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            ref=v.pointer(PARENT/f'fold-{fold}-{role}'/'result.json');r=v.bound(ref)
            program=language.import_previous(previous_language.Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][role]),9)
            prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
            cases,a,_=sampled_fit(dp,v.bound(prior['collection']),fold,role);lanes=search.old.metrics.lane_parts(a,cases)
            fixed,roots,paths,_=search.scan(program,lanes)
            maximum_models=sum(2**min(12-len(root),d+1) for root in roots if len(root)<12 for d in (2,3))
            maximum_nodes=sum(2**min(12-len(root)-1,d)-1 for root in roots if len(root)<12 for d in (2,3))
            for maximum_depth in (9,12):
                jobs.append(dict(id=f'depth-{maximum_depth}-fold-{fold}-{role}',fold=fold,role=role,source_result=ref,
                    maximum_depth=maximum_depth,frozen_leaves=sorted(fixed),structural_roots=[list(root) for root in roots],
                    limits=dict(node=maximum_nodes,leaf=72*maximum_models,proof=320*maximum_models)))
    plan=dict(schema=VERSION,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_files=sources(),
        parent_summary=v.pointer(PARENT/'result.json'),parent_audit=v.pointer(PARENT/'independent-replay.json'),actor=old['actor'],actor_role_sha256=old['actor_role_sha256'],
        source_method='uniform_v640_all_six_no_rolewise_selection',depth_arms=[9,12],maximum_leaves=256,equal_call_limits_per_pair=True,
        user_depth12_authorized=True,depth9_historical_assets_unchanged=True,
        selection_rule='one_uniform_depth_arm_all_six_pass_prefer_depth9_no_rolewise_mixing',maximum_node_and_leaf_public_sources=3,
        maximum_leaf_coefficients=20,context_depth_options=[2,3],minimum_context_child_frames=160,
        borrowed_public_directions='all10_logit_pairs_of_eight_old_models_ranked_by_fit_training_exposure',maximum_borrowed_directions_per_root=80,
        topology_target='complete_masked_NN_soft_distribution_entropy_old_fit900_internal80percent_only',
        leaves='unchanged_full_distribution_C3_fit_internal80_20_then_fullfit900',temperature='old_cal150_only',
        terminal_physical_guards_explicit_and_counted=True,original_core_and_current_fit_reliable_E_paths_outputs_frozen=True,
        old_subtree_remains_finite_option=True,finite_choice='whole_program_budget256_four_registered_allocations',
        reliability='unchanged_27of30_scene_subsets_150risk150select_all_gates_unchanged',fit_only_proxy_is_not_independent_new_data=True,
        outer_qualification=300,outer_validation=300,search_seconds=3600,total_seconds=4200,maximum_workers=3,BLAS_threads=1,automatic_retry=False,
        fresh_confirmation_required=True,NN_updates=0,PPO_joint_steps=0,new_environment_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    output.mkdir(parents=True)
    for name,sha in plan['source_files'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name;v.require(v.file_sha(p)==sha,'source changed before snapshot')
        snapshot_source(p,output/'source'/p.relative_to(ROOT),sha)
    write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444);refs=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(work,str(output),spec) for spec in jobs]):
            ref=f.result();refs.append(ref);r=v.bound(ref)
            print(json.dumps(dict(job=r['spec']['id'],status=r['status'],E=None if r['status']!='completed' else r['metrics']['validation']['explanation']['coverage'],
                numerical=r['numerical_passed'],reliable=r.get('reliable_fit_only_passed'),error=r.get('error'))),flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'frozen NN changed');rows=[v.bound(ref) for ref in refs]
    arms={str(d):dict(numerical_pass_count=sum(r['numerical_passed'] for r in rows if r['spec']['maximum_depth']==d),
        development_pass_count=sum(r['development_passed'] for r in rows if r['spec']['maximum_depth']==d),
        all_six_passed=sum(r['development_passed'] for r in rows if r['spec']['maximum_depth']==d)==6) for d in (9,12)}
    passed=any(a['all_six_passed'] for a in arms.values())
    summary=dict(schema=VERSION,plan=v.pointer(output/'plan.json'),units=sorted(refs,key=lambda r:r['path']),all_twelve_settled=len(rows)==12,arms=arms,
        numerical_pass_count=sum(r['numerical_passed'] for r in rows),development_pass_count=sum(r['development_passed'] for r in rows),scientific_development_passed=passed,
        method_locked=False,independent_replay_required=True,fresh_confirmation_required=True,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',summary);(output/'result.json').chmod(0o444)
    lines=['# v6.42深9/深12配对容量对照：十二单元开发结果','','| 单元 | 状态 | 来源E | 新E | 新F | 预期F | F/E一致 | F/E KL | 非法/未证 | 数值/可靠 |',
        '|---|---|---:|---:|---:|---:|---|---|---|---|']
    for r in sorted(rows,key=lambda r:r['spec']['id']):
        if r['status']!='completed':lines.append(f"| {r['spec']['id']} | failed | — | — | — | — | — | — | — | False/False |");continue
        b=r['baseline_metrics']['validation']['explanation'];f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
        lines.append(f"| {r['spec']['id']} | completed | {b['coverage']:.5%} | {e['coverage']:.5%} | {f['coverage']:.5%} | {f['expected_actionable_coverage']:.5%} | {f['agreement']:.5%}/{e['agreement']:.5%} | {f['kl']:.6f}/{e['kl']:.6f} | {e['whole_tree_raw_illegal_count']}/{e['whole_tree_path_uncertified_count']} | {r['numerical_passed']}/{r['reliable_fit_only_passed']} |")
    lines+=['','## 内部可靠选择覆盖（分母为同一旧fit-only选择帧）','','| 单元 | 可靠E | 可靠F | 可靠预期F | 距45% E差距 | 深度/叶数 | 调用node/leaf/proof |','|---|---:|---:|---:|---:|---|---|']
    for r in sorted(rows,key=lambda r:r['spec']['id']):
        if r['status']!='completed':continue
        report=v.bound(r['search_report']);chosen=report['candidates'][report['selected']];counts=chosen['reliable_counts']
        n=report['selection_frames'];program=v.bound(r['program']);_,info=language.structure_info(program['structure'],256,r['spec']['maximum_depth'])
        lines.append(f"| {r['spec']['id']} | {counts['E']/n:.5%} | {counts['F']/n:.5%} | {counts['expected_F']/n:.5%} | {max(0.,.45-counts['E']/n):.5%} | {info['depth']}/{info['leaf_count']} | {r['counts']['node']}/{r['counts']['leaf']}/{r['counts']['proof']} |")
    lines+=['',f'存在完整六项通过的容量方案：{passed}。独立重放和全新确认仍为必要后续门槛。','同一冻结NN、旧开发数据；未训练Actor，pilot关闭。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
