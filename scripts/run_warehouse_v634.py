#!/usr/bin/env python3
"""Twelve preregistered entropy/precision-union old-development comparisons."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v634_search as search
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_program import measure
from scripts.run_warehouse_v632 import strict
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v625 import write_new
from scripts.verify_warehouse_v625_native import real_deadline

VERSION='warehouse-v634-precision-union-development.v1'
PARENT=ROOT/'output/warehouse_v633/warehouse-v633-support-refresh-01'
FILES=('backend/training/warehouse_v634_partitions.py','backend/training/warehouse_v634_search.py',
       'scripts/run_warehouse_v634.py','tests/test_warehouse_v634_partitions.py','docs/warehouse_v634_plan.md')


class Budget:
    def __init__(self,path):
        self.path=path;self.h=path.open('x');self.started=time.monotonic();self.final=False
        self.counts=dict(node=0,leaf=0,proof=0);self.seq=0;self.tail='0'*64
    def check(self):v.require(time.monotonic()-self.started<(2100 if self.final else 1800),'registered phase wall budget exhausted')
    def solve(self,kind,metadata,callback):
        self.check();v.require(not self.final and kind=='node' and self.counts[kind]<37632,'registered solver budget exhausted')
        self.counts[kind]+=1;number=self.counts[kind];self.event('call_started',dict(kind=kind,call=number,metadata=metadata))
        try:
            result=callback();self.event('call_completed',dict(kind=kind,call=number,success=bool(result.success)));return result
        except BaseException as error:
            self.event('call_completed',dict(kind=kind,call=number,success=False,error=type(error).__name__));raise
    def event(self,name,data):
        self.seq+=1;row=dict(sequence=self.seq,previous_sha256=self.tail,event=name,data=data)
        self.tail=v.digest(row);row['record_sha256']=self.tail
        self.h.write(json.dumps(row,separators=(',',':'))+'\n');self.h.flush();os.fsync(self.h.fileno())
    def close(self):self.h.close();self.path.chmod(0o444)


def sources():
    receipt=v.bound(v.pointer(ROOT/'experiments/warehouse/evidence/warehouse-v633-closed-diagnostic-20261022-01.json'))
    v.require(receipt['status']=='closed_failed_development_precheck' and receipt['independent_replay_integrity_passed'],
              'require complete failed prior closure')
    files=dict(v.bound(receipt['references']['plan'])['source_files'])
    for name in FILES:files[str(ROOT/name)]=v.file_sha(ROOT/name)
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        p=Path(raw)
        if not p.is_absolute() and not p.is_file():continue
        p=p.resolve()
        if p.suffix=='.py' and p.is_relative_to(ROOT):files[str(p)]=v.file_sha(p)
    for name,sha in files.items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'immutable source changed')
    return dict(sorted(files.items()))


def work(output,spec):
    folder=Path(output)/spec['id'];folder.mkdir(exist_ok=False);planref=v.pointer(Path(output)/'plan.json')
    budget=Budget(folder/'ledger.jsonl');write_new(folder/'started.json',dict(spec=spec,plan=planref,pid=os.getpid()))
    try:
        _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
        sr=v.bound(spec['source_result']);source=Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
        with real_deadline(budget.started,2100):
            cases,a,arrayrefs=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role'])
            parts=search.old.lane_parts(a,cases)
            p,report=search.run(source,parts,budget,selective=spec['selective'])
            write_new(folder/'fit-report.json',report);budget.final=True
            outer=data.load_parts(dp,spec['fold'],spec['role']);baseline={s:measure(source,a) for s,a in outer.items()}
            v.require(baseline==sr['metrics'],'actual source metric dictionaries differ')
            search.old.qualify_current(p,outer);metrics={s:measure(p,a) for s,a in outer.items()}
            independent=audited_program(p,outer,metrics)
            for split,a in outer.items():
                before=source.distributions(a['x'],a['m'])
                v.require(all(np.array_equal(x,y) for x,y in zip(before[1:],independent[split][1:])),'full q changed')
        write_new(folder/'program.json',p.to_dict())
        r=dict(status='completed',spec=spec,plan=planref,program=v.pointer(folder/'program.json'),fit_report=v.pointer(folder/'fit-report.json'),
            baseline_metrics=baseline,metrics=metrics,numerical_passed=strict(metrics['validation']),
            reliable_fit_only_passed=report['reliable_selection_passed'],development_passed=strict(metrics['validation']) and report['reliable_selection_passed'],
            counts=budget.counts,elapsed_seconds=time.monotonic()-budget.started,independent_program_frames=sum(len(a['x']) for a in outer.values()),
            independent_leaf_certificates=len(p.leaves),complete_source_q_preserved=True,NN_updates=0,PPO_joint_steps=0,
            new_environment_resets=0,program_training_gradients=0,model_acceptance_passed=False,pilot_opened=False)
    except Exception as error:
        r=dict(status='failed',spec=spec,plan=planref,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            counts=budget.counts,elapsed_seconds=time.monotonic()-budget.started,numerical_passed=False,development_passed=False,pilot_opened=False)
    finally:budget.close()
    for f in folder.iterdir():
        if f.is_file():f.chmod(0o444)
    write_new(folder/'result.json',r);(folder/'result.json').chmod(0o444);return v.pointer(folder/'result.json')


def main(output):
    v.require(not output.exists(),'single-use run cannot restart');_,old=v.checked_plan()
    parentref=v.pointer(PARENT/'result.json');parent=v.bound(parentref)
    v.require(v.bound(v.pointer(PARENT/'independent-replay.json'))['integrity_passed'] and parent['all_six_settled'] and not parent['scientific_development_passed'],
              'require all six prior units settled and independently audited without winner')
    jobs=[dict(id=f'{arm}-fold-{fold}-{role}',arm=arm,selective=arm=='precision_union',fold=fold,role=role,
               source_result=v.pointer(PARENT/f'fold-{fold}-{role}'/'result.json'))
          for fold in range(3) for role in ('robot_1','robot_2') for arm in ('entropy','precision_union')]
    plan=dict(schema=VERSION,registered_utc=datetime.now(timezone.utc).isoformat(),source_files=sources(),jobs=jobs,
        source_method='same_six_v633_final_programs_for_both_arms_no_rolewise_selection',parent_summary=parentref,
        actor=old['actor'],actor_role_sha256=old['actor_role_sha256'],maximum_depth=9,maximum_leaves=256,maximum_public_sources=3,
        precision_training_targets=list(search.precision.PRECISIONS),minimum_child_frames=40,templates_per_entropy_group=1,templates_per_union_group=3,
        same_maximum_optimizer_and_wall_budget=True,actual_compute_may_differ=True,
        causal_scope='joint_change_of_training_threshold_objective_and_finite_template_pool_not_compute_matched_objective_effect',
        classifier_kernel='immutable_v632_error_weight4_sparse_logistic_max7_support_fits_per_node',
        maximum_node_calls=37632,maximum_leaf_calls=0,maximum_proof_LP_calls=0,search_seconds=1800,total_seconds=2100,
        data='same_spent_1650_scene_collection',lanes='600_sampled_outer_fit_into300_train150_risk150_select',
        risk_repeats=30,risk_scenes_per_repeat=90,minimum_risk_passes=27,
        gates=dict(F_E_coverage=.45,expected_F=.40,F_E_agreement=[.95,.97],KL=.20,illegal=0,unproved=0),
        source_q_frozen=True,original_subtree_always_candidate=True,selection='same_four_fit_only_allocations_all_six_pass_then_entropy_first',
        fresh_confirmation_required=True,maximum_workers=3,BLAS_threads=1,automatic_retry=False,NN_updates=0,PPO_joint_steps=0,
        new_environment_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    output.mkdir(parents=True)
    for name,sha in plan['source_files'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name;v.require(v.file_sha(p)==sha,'source changed before snapshot')
        target=output/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(p.read_bytes());target.chmod(0o444)
    write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444)
    refs=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(work,str(output),s) for s in jobs]):
            ref=f.result();refs.append(ref);r=v.bound(ref)
            print(json.dumps(dict(job=r['spec']['id'],status=r['status'],E=None if r['status']!='completed' else r['metrics']['validation']['explanation']['coverage'],
                numerical=r['numerical_passed'],reliable=r.get('reliable_fit_only_passed'),error=r.get('error'))),flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'frozen Actor changed')
    rows=[v.bound(r) for r in refs];methods={arm:dict(completed=sum(r['status']=='completed' for r in rows if r['spec']['arm']==arm),
        numerical_passes=sum(r['numerical_passed'] for r in rows if r['spec']['arm']==arm),
        development_passes=sum(r['development_passed'] for r in rows if r['spec']['arm']==arm)) for arm in ('entropy','precision_union')}
    selected=next((a for a in ('entropy','precision_union') if methods[a]['development_passes']==6),None)
    summary=dict(schema=VERSION,plan=v.pointer(output/'plan.json'),units=sorted(refs,key=lambda r:r['path']),methods=methods,
        all_twelve_settled=len(refs)==12,selected_method=selected,scientific_development_passed=selected is not None,
        method_locked=False,independent_replay_required=True,fresh_confirmation_required=True,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',summary);(output/'result.json').chmod(0o444)
    lines=['# v6.34高精度覆盖条件：完整12项开发对照','',
        '| 方法 | 折/角色 | 状态 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 | 数值/可靠 |',
        '|---|---|---|---:|---:|---:|---|---|---|---|']
    for r in sorted(rows,key=lambda r:r['spec']['id']):
        if r['status']!='completed':
            lines.append(f"| {r['spec']['arm']} | {r['spec']['fold']}/{r['spec']['role']} | failed | — | — | — | — | — | — | False/False |");continue
        f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
        lines.append(f"| {r['spec']['arm']} | {r['spec']['fold']}/{r['spec']['role']} | completed | {f['coverage']:.5%} | {e['coverage']:.5%} | {f['expected_actionable_coverage']:.5%} | {f['agreement']:.5%}/{e['agreement']:.5%} | {f['kl']:.6f}/{e['kl']:.6f} | {e['whole_tree_raw_illegal_count']}/{e['whole_tree_path_uncertified_count']} | {r['numerical_passed']}/{r['reliable_fit_only_passed']} |")
    lines+=['',f'选中方法：{selected}。NN及完整程序输出不变，仅改变外部置信分区。',
        '同一旧开发数据；不是新确认、PPO、封存、留出或真人效果。pilot关闭。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
