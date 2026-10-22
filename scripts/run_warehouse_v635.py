#!/usr/bin/env python3
"""Six uniform action-qualified root-rebuild experiments, NN stays frozen."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v635_search as search,warehouse_v635_program as language
from backend.training.warehouse_v630_program import Program as Previous
from backend.training.warehouse_v626_search import encode,decode
from backend.training.warehouse_v628_program import measure
from scripts.diagnose_warehouse_v634_free_roots import free_roots
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.audit_warehouse_v630 import audited_program
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new
from scripts.verify_warehouse_v625_native import real_deadline

VERSION='warehouse-v635-action-qualified-root-rebuild.v1'
PARENT=ROOT/'output/warehouse_v633/warehouse-v633-support-refresh-01'
FILES=('backend/training/warehouse_v635_guards.py','backend/training/warehouse_v635_program.py',
       'backend/training/warehouse_v635_subtrees.py','backend/training/warehouse_v635_search.py',
       'scripts/run_warehouse_v635.py','tests/test_warehouse_v635_guards.py','tests/test_warehouse_v635_subtrees.py',
       'tests/test_warehouse_v635_empty_regions.py','docs/warehouse_v635_plan.md')


class Budget:
    def __init__(self,path,limits,spent=None):
        spent=spent or dict(counts={k:0 for k in limits},seconds=0.,source=None)
        self.path=path;self.h=path.open('x');self.started=time.monotonic()-spent['seconds'];self.final=False;self.limits=limits
        self.counts=dict(spent['counts']);self.seq=0;self.tail='0'*64
        self.event('engineering_prefix_debit',spent)
    def check(self):v.require(time.monotonic()-self.started<(4200 if self.final else 3600),'registered root-rebuild phase wall budget exhausted')
    def solve(self,kind,metadata,callback):
        self.check();v.require(not self.final and kind in self.limits and self.counts[kind]<self.limits[kind],'root-rebuild call budget exhausted')
        self.counts[kind]+=1;number=self.counts[kind];self.event('call_started',dict(kind=kind,call=number,metadata=metadata))
        try:
            result=callback();self.event('call_completed',dict(kind=kind,call=number,success=bool(result.success)));return result
        except BaseException as error:
            self.event('call_completed',dict(kind=kind,call=number,success=False,error=type(error).__name__));raise
    def event(self,name,data):
        self.seq+=1;row=dict(sequence=self.seq,previous_sha256=self.tail,event=name,data=data)
        self.tail=v.digest(row);row['record_sha256']=self.tail
        self.h.write(json.dumps(row,separators=(',',':'))+'\n');self.h.flush();os.fsync(self.h.fileno())
    def checkpoint(self,data):self.event('progress',data)
    def close(self):self.h.close();self.path.chmod(0o444)


def sources():
    previous=v.bound(v.pointer(ROOT/'experiments/warehouse/evidence/warehouse-v634-closed-diagnostic-20261022-01.json'))
    v.require(previous['status']=='closed_failed_development_precheck' and previous['independent_replay_integrity_passed'],
              'need complete failed prior experiment before root rebuild')
    files=dict(v.bound(previous['references']['plan'])['source_files'])
    for name in FILES:files[str(ROOT/name)]=v.file_sha(ROOT/name)
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        p=Path(raw)
        if not p.is_absolute() and not p.is_file():continue
        p=p.resolve()
        if p.suffix=='.py' and p.is_relative_to(ROOT):files[str(p)]=v.file_sha(p)
    for name,sha in files.items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'old immutable source changed')
    return dict(sorted(files.items()))


def work(output,spec):
    folder=Path(output)/spec['id'];folder.mkdir(exist_ok=False);planref=v.pointer(Path(output)/'plan.json')
    budget=Budget(folder/'ledger.jsonl',spec['limits'],spec.get('engineering_spent'));write_new(folder/'started.json',dict(spec=spec,plan=planref,pid=os.getpid(),started_monotonic=budget.started))
    try:
        _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
        sr=v.bound(spec['source_result']);source=Previous.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
        with real_deadline(budget.started,4200):
            outer=data.load_parts(dp,spec['fold'],spec['role'])
            baseline={s:measure(source,a) for s,a in outer.items()};v.require(baseline==sr['metrics'],'baseline metric dictionaries differ')
            cases,a,refs=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=search.metrics.lane_parts(a,cases)
            program,report=search.run(source,outer,lanes,budget)
            write_new(folder/'search-report.json',report);budget.final=True
            language.qualify(program,outer);metrics={s:language.measure(program,a) for s,a in outer.items()}
            independent=audited_program(program,outer,metrics)
            protected=language.protection(program)['protected_descendants'];before=source.distributions(outer['validation']['x'],outer['validation']['m'])
            actual=independent['validation'];take=np.isin(actual[0],protected)
            v.require(np.array_equal(before[1][take],actual[1][take]) and np.array_equal(before[2][take],actual[2][take]),'original protected outputs changed')
        write_new(folder/'program.json',program.to_dict())
        r=dict(status='completed',spec=spec,plan=planref,program=v.pointer(folder/'program.json'),search_report=v.pointer(folder/'search-report.json'),
            baseline_metrics=baseline,metrics=metrics,numerical_passed=strict(metrics['validation']),reliable_fit_only_passed=report['reliable_selection_passed'],
            development_passed=strict(metrics['validation']) and report['reliable_selection_passed'],
            counts=budget.counts,elapsed_seconds=time.monotonic()-budget.started,independent_program_frames=sum(len(a['x']) for a in outer.values()),
            independent_leaf_certificates=len(program.leaves),original_protected_outputs_unchanged=True,
            NN_updates=0,PPO_joint_steps=0,new_environment_resets=0,program_training_gradients=0,model_acceptance_passed=False,pilot_opened=False)
    except Exception as error:
        r=dict(status='failed',spec=spec,plan=planref,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            counts=budget.counts,elapsed_seconds=time.monotonic()-budget.started,numerical_passed=False,development_passed=False,pilot_opened=False)
    finally:budget.close()
    for p in folder.iterdir():
        if p.is_file():p.chmod(0o444)
    write_new(folder/'result.json',r);(folder/'result.json').chmod(0o444);return v.pointer(folder/'result.json')


def main(output,engineering_parent=None):
    v.require(not output.exists(),'single-use run already exists');_,old=v.checked_plan()
    jobs=[]
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            ref=v.pointer(PARENT/f'fold-{fold}-{role}'/'result.json');r=v.bound(ref);source=Previous.from_dict(v.bound(r['program']),old['actor_role_sha256'][role])
            roots,_=free_roots(source);slots=[min(4,9-len(root)) for root in roots if len(root)<9]
            maximum_new_models=2*sum(s+1 for s in slots)
            limits=dict(node=2*sum(slots)*4*7,leaf=72*maximum_new_models,proof=320*maximum_new_models)
            jobs.append(dict(id=f'fold-{fold}-{role}',fold=fold,role=role,source_result=ref,limits=limits,structural_roots=[list(x) for x in roots]))
    if engineering_parent is not None:
        previous=v.bound(v.pointer(engineering_parent/'result.json'))
        v.require(previous['all_six_settled'] and not previous['scientific_development_passed'],'engineering parent is not terminal')
        byid={v.bound(ref)['spec']['id']:(ref,v.bound(ref)) for ref in previous['units']}
        for job in jobs:
            ref,failed=byid[job['id']]
            v.require(failed['status']=='failed' and failed['spec']['source_result']==job['source_result'] and failed['spec']['limits']==job['limits'] and
                failed['error'] in ("ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",'invalid Warehouse hard-tree split inputs'),
                'unrecognized repair or changed scientific settings')
            job['engineering_spent']=dict(counts=failed['counts'],seconds=failed['elapsed_seconds'],source=ref,
                ledger=v.pointer(engineering_parent/job['id']/'ledger.jsonl'))
    plan=dict(schema=VERSION,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_files=sources(),
        source_method='uniform_v633_final_all_six_no_rolewise_selection',actor=old['actor'],actor_role_sha256=old['actor_role_sha256'],
        maximum_depth=9,maximum_leaves=256,maximum_node_and_leaf_public_sources=3,maximum_leaf_coefficients=20,
        structure='greedy_action_qualified_decision_lists_inside_all_maximal_protected_free_roots',maximum_rules_per_root=4,
        training_precision_targets=[.98,.99],right_leaf_is_complete_NN_distribution_not_fixed_action=True,
        unknown_action_guard='physical_mask_bit_AND_learned_context_at_most_two_public_source_units',
        already_proven_action_context_sources=3,proof='exact_rational_public_zero_mask_upper_plus_FP64_error',
        root_choice='structural_only_no_validation_labels',condition_fit='old_outer_fit900_internal80percent_whole_scenes',
        leaf_support_selection='old_outer_fit900_internal80_20_complete_NN_distribution',final_leaf_fit='all_old_fit900',
        temperature='old_cal150_only',option_selection='unchanged_fit_lanes300_150_150_risk27of30_and_four_allocations',
        baseline_old_fit_already_used_these_lanes=True,fit_only_proxy_is_not_independent_new_data=True,
        source_original_protected_outputs_immutable=True,unprotected_program_outputs_may_change=True,NN_execution_unmodified=True,
        risk_and_final_gates_unchanged=True,search_seconds=3600,total_seconds=4200,maximum_workers=3,BLAS_threads=1,automatic_retry=False,
        fresh_confirmation_required=True,NN_updates=0,PPO_joint_steps=0,new_environment_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    plan['engineering_parent']=None if engineering_parent is None else v.pointer(engineering_parent/'result.json')
    plan['engineering_repair']='empty_bool_dtype_and_normalized_stump_router_only_cumulative_budgets'
    output.mkdir(parents=True)
    for name,sha in plan['source_files'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name;v.require(v.file_sha(p)==sha,'source changed before snapshot')
        target=output/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(p.read_bytes());target.chmod(0o444)
    write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444)
    refs=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(work,str(output),spec) for spec in jobs]):
            ref=f.result();refs.append(ref);r=v.bound(ref)
            print(json.dumps(dict(job=r['spec']['id'],status=r['status'],E=None if r['status']!='completed' else r['metrics']['validation']['explanation']['coverage'],
                numerical=r['numerical_passed'],reliable=r.get('reliable_fit_only_passed'),error=r.get('error'))),flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'frozen NN changed')
    rows=[v.bound(ref) for ref in refs];passed=len(rows)==6 and all(r['development_passed'] for r in rows)
    summary=dict(schema=VERSION,plan=v.pointer(output/'plan.json'),units=sorted(refs,key=lambda r:r['path']),all_six_settled=len(rows)==6,
        numerical_pass_count=sum(r['numerical_passed'] for r in rows),development_pass_count=sum(r['development_passed'] for r in rows),
        scientific_development_passed=passed,method_locked=False,independent_replay_required=True,fresh_confirmation_required=True,
        NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',summary);(output/'result.json').chmod(0o444)
    lines=['# v6.35动作合法条件子树：完整六单元','',
        '| 单元 | 状态 | 来源E | 新E | 新F | 预期F | F/E一致 | F/E KL | 非法/未证 | 数值/可靠 |',
        '|---|---|---:|---:|---:|---:|---|---|---|---|']
    for r in sorted(rows,key=lambda r:r['spec']['id']):
        if r['status']!='completed':
            lines.append(f"| {r['spec']['id']} | failed | — | — | — | — | — | — | — | False/False |");continue
        b=r['baseline_metrics']['validation']['explanation'];f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
        lines.append(f"| {r['spec']['id']} | completed | {b['coverage']:.5%} | {e['coverage']:.5%} | {f['coverage']:.5%} | {f['expected_actionable_coverage']:.5%} | {f['agreement']:.5%}/{e['agreement']:.5%} | {f['kl']:.6f}/{e['kl']:.6f} | {e['whole_tree_raw_illegal_count']}/{e['whole_tree_path_uncertified_count']} | {r['numerical_passed']}/{r['reliable_fit_only_passed']} |")
    lines+=['',f'六项科学开发门槛全过：{passed}。完整独立重放与全新确认仍为后续必要门槛。',
        'NN冻结，执行规则不改；程序只作训练反馈和解释。旧开发数据，绝非封存／留出或真人效果。pilot关闭。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering-parent',type=Path)
    a=p.parse_args();main(a.output.resolve(),None if a.engineering_parent is None else a.engineering_parent.resolve())
