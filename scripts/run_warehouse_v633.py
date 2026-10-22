#!/usr/bin/env python3
"""Six uniform support-refresh + confidence-partition development jobs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, json, os, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v, v628_data as data
from backend.training import warehouse_v633_support_refresh as support
from backend.training import warehouse_v632_search as confidence
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_program import measure, payload
from scripts.run_warehouse_v632 import strict
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v625 import write_new
from scripts.verify_warehouse_v625_native import real_deadline

VERSION='warehouse-v633-support-refresh-development.v1'
PARENT=ROOT/'output/warehouse_v632/warehouse-v632-oblique-confidence-01/outer-evaluation-01'
FILES=('backend/training/warehouse_v633_support_refresh.py','scripts/run_warehouse_v633.py',
       'tests/test_warehouse_v633_support_refresh.py','docs/warehouse_v633_plan.md')


class Budget:
    def __init__(self,path,limits):
        self.started=time.monotonic();self.phase='support';self.counts={k:0 for k in limits};self.limits=limits
        self.h=path.open('x');self.path=path;self.tail='0'*64;self.sequence=0
    def check(self):
        limit=dict(support=900,confidence=1800,final=2100)[self.phase]
        v.require(time.monotonic()-self.started<limit,'registered '+self.phase+' phase deadline exhausted')
    def solve(self,kind,metadata,callback):
        self.check();v.require(kind in self.limits and self.counts[kind]<self.limits[kind],'registered optimizer budget exhausted')
        self.counts[kind]+=1;number=self.counts[kind]
        self.event('call_started',dict(kind=kind,call=number,phase=self.phase,metadata=metadata))
        try:
            result=callback();self.event('call_completed',dict(kind=kind,call=number,phase=self.phase,success=bool(result.success)))
            return result
        except BaseException as error:
            self.event('call_completed',dict(kind=kind,call=number,phase=self.phase,success=False,error=type(error).__name__));raise
    def event(self,name,data):
        self.sequence+=1;row=dict(sequence=self.sequence,previous_sha256=self.tail,event=name,data=data)
        self.tail=v.digest(row);row['record_sha256']=self.tail
        self.h.write(json.dumps(row,separators=(',',':'))+'\n');self.h.flush();os.fsync(self.h.fileno())
    def close(self):self.h.close();self.path.chmod(0o444)


def source_files():
    receipt=v.bound(v.pointer(ROOT/'experiments/warehouse/evidence/warehouse-v632-closed-diagnostic-20261022-01.json'))
    v.require(receipt['status']=='closed_failed_development_precheck' and receipt['independent_replay_integrity_passed'],
              'need the completed failed v632 development closure')
    files=dict(v.bound(receipt['references']['report_plan'])['source_files'])
    for name in FILES:files[str(ROOT/name)]=v.file_sha(ROOT/name)
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        p=Path(raw)
        if not p.is_absolute() and not p.is_file():continue
        p=p.resolve()
        if p.suffix=='.py' and p.is_relative_to(ROOT):files[str(p)]=v.file_sha(p)
    for name,sha in files.items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'prior immutable source changed')
    return dict(sorted(files.items()))


def work(output,spec):
    folder=Path(output)/spec['id'];folder.mkdir(exist_ok=False)
    planref=v.pointer(Path(output)/'plan.json');plan=v.bound(planref)
    budget=Budget(folder/'ledger.jsonl',spec['limits']);write_new(folder/'started.json',dict(job=spec,plan=planref,pid=os.getpid()))
    try:
        _,old=v.checked_plan();dp=v.bound(v.bound(old['previous_plan'])['previous_plan'])
        sr=v.bound(spec['source_result']);source=Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
        with real_deadline(budget.started,2100):
            parts=data.load_parts(dp,spec['fold'],spec['role'])
            baseline={s:measure(source,a) for s,a in parts.items()}
            v.require(baseline==sr['metrics'],'actual baseline metrics differ')
            refreshed,refresh_report=support.refresh(source,parts,budget)
            write_new(folder/'refreshed-unqualified.json',refreshed.to_dict())
            write_new(folder/'support-report.json',refresh_report)
            support_counts=dict(budget.counts);budget.phase='confidence'
            previous=v.bound(old['previous_plan']);collection=v.bound(previous['collection'])
            cases,a,arrayrefs=sampled_fit(dp,collection,spec['fold'],spec['role'])
            lanes=confidence.old.lane_parts(a,cases)
            program,report=confidence.run(refreshed,lanes,budget,oblique=True)
            write_new(folder/'confidence-report.json',report)
            budget.phase='final'
            # Neither outer result is consulted until the complete fit-only
            # support/structure selection has already finished.
            confidence.old.qualify_current(refreshed,parts);confidence.old.qualify_current(program,parts)
            controls={s:measure(refreshed,a) for s,a in parts.items()}
            metrics={s:measure(program,a) for s,a in parts.items()}
            control_predictions=audited_program(refreshed,parts,controls)
            predictions=audited_program(program,parts,metrics)
            protected=support.protected_indices(source);old_routes=source.leaf_indices(parts['validation']['x'])
            a=parts['validation'];before=source.distributions(a['x'],a['m']);keep=np.isin(old_routes,list(protected))
            v.require(np.array_equal(before[1][keep],predictions['validation'][1][keep]) and
                      np.array_equal(before[2][keep],predictions['validation'][2][keep]),'original protected arithmetic changed')
            for split,a in parts.items():
                v.require(all(np.array_equal(x,y) for x,y in zip(control_predictions[split][1:],predictions[split][1:])),
                          'confidence partition changed refreshed full output')
        write_new(folder/'support-only-program.json',refreshed.to_dict());write_new(folder/'program.json',program.to_dict())
        r=dict(status='completed',spec=spec,plan=planref,baseline_metrics=baseline,support_only_metrics=controls,metrics=metrics,
            support_only_program=v.pointer(folder/'support-only-program.json'),program=v.pointer(folder/'program.json'),
            support_report=v.pointer(folder/'support-report.json'),confidence_report=v.pointer(folder/'confidence-report.json'),
            numerical_passed=strict(metrics['validation']),reliable_fit_only_passed=report['reliable_selection_passed'],
            development_passed=strict(metrics['validation']) and report['reliable_selection_passed'],
            source_protected_outputs_unchanged=True,refreshed_and_confidence_outputs_identical=True,
            support_counts=support_counts,counts=dict(budget.counts),elapsed_seconds=time.monotonic()-budget.started,
            independent_program_frames=2*sum(len(a['x']) for a in parts.values()),
            independent_leaf_certificates=len(refreshed.leaves)+len(program.leaves),NN_updates=0,new_environment_resets=0,
            program_training_gradients=0,model_acceptance_passed=False,pilot_opened=False)
    except Exception as error:
        r=dict(status='failed',spec=spec,plan=planref,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            counts=dict(budget.counts),phase=budget.phase,elapsed_seconds=time.monotonic()-budget.started,
            numerical_passed=False,development_passed=False,model_acceptance_passed=False,pilot_opened=False)
    finally:budget.close()
    for p in folder.iterdir():
        if p.is_file():p.chmod(0o444)
    write_new(folder/'result.json',r);(folder/'result.json').chmod(0o444)
    return v.pointer(folder/'result.json')


def main(output):
    v.require(not output.exists(),'single-use candidate directory already exists')
    _,old=v.checked_plan();parent_ref=v.pointer(PARENT/'result.json');parent=v.bound(parent_ref)
    audit=v.bound(v.pointer(PARENT/'independent-replay.json'))
    v.require(audit['integrity_passed'] and parent['selected_method'] is None,'need all v632 comparisons audited and closed without winner')
    jobs=[]
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            ref=v.pointer(PARENT/f'oblique-fold-{fold}-{role}'/'result.json');r=v.bound(ref)
            p=Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][role])
            jobs.append(dict(id=f'fold-{fold}-{role}',fold=fold,role=role,source_result=ref,limits=support.limits(p)))
    plan=dict(schema=VERSION,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_files=source_files(),
        parent_summary=parent_ref,parent_audit=v.pointer(PARENT/'independent-replay.json'),actor=old['actor'],
        actor_role_sha256=old['actor_role_sha256'],source_method='v632_oblique_for_all_six_no_rolewise_selection',
        same_frozen_NN=True,maximum_depth=9,maximum_leaves=256,maximum_public_sources=3,
        support_trials_per_leaf=8,proxy='unchanged_v626_proxy_then_fewer_sources',
        support_parameter_fit='old_outer_fit900_whole_scene_internal80_20_then_selected_fullfit900',
        temperature_calibration='old_outer_cal150_only_no_support_or_structure_choice',
        confidence='unchanged_v632_oblique_300_fit_150_risk_150_select_within_spent_outer_fit',
        old_source_and_refitted_outputs_use_fit900=True,fit_only_proxy_is_not_independent_new_data=True,
        qualification='old_outer_eligibility300',validation='old_outer_validation300',
        original_protected_payloads_immutable=True,unprotected_payloads_may_change=True,
        support_only_is_ablation_not_an_alternative_winner=True,all_six_must_pass_before_method_lock=True,
        per_unit_phase_deadlines=[900,1800,2100],maximum_workers=3,BLAS_threads=1,automatic_retry=False,
        data_scope='spent_development_only',fresh_confirmation_required=True,NN_updates=0,PPO_joint_steps=0,
        new_environment_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    output.mkdir(parents=True)
    for name,sha in plan['source_files'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name;v.require(v.file_sha(p)==sha,'registered source changed')
        target=output/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(p.read_bytes());target.chmod(0o444)
    write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444)
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(work,str(output),spec) for spec in jobs]
        for f in as_completed(fs):
            ref=f.result();results.append(ref);r=v.bound(ref)
            print(json.dumps(dict(job=r['spec']['id'],status=r['status'],E=None if r['status']!='completed' else
                r['metrics']['validation']['explanation']['coverage'],numerical=r['numerical_passed'],
                reliable=r.get('reliable_fit_only_passed'),error=r.get('error'))),flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'frozen Actor changed')
    rows=[v.bound(ref) for ref in results];passed=len(rows)==6 and all(r['development_passed'] for r in rows)
    summary=dict(schema=VERSION,plan=v.pointer(output/'plan.json'),units=sorted(results,key=lambda r:r['path']),
        all_six_settled=len(rows)==6,numerical_pass_count=sum(r['numerical_passed'] for r in rows),
        development_pass_count=sum(r['development_passed'] for r in rows),scientific_development_passed=passed,
        method_locked=False,independent_full_replay_required=True,fresh_confirmation_required=True,
        NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',summary);(output/'result.json').chmod(0o444)
    lines=['# v6.33支持刷新＋斜向分区：完整六单元','',
        '| 折/角色 | 状态 | 原E | 仅刷新E | 最终E | 最终F | 预期F | F/E一致 | F/E KL | 非法/未证 | 数值/可靠 |',
        '|---|---|---:|---:|---:|---:|---:|---|---|---|---|']
    for r in sorted(rows,key=lambda r:r['spec']['id']):
        if r['status']!='completed':
            lines.append(f"| {r['spec']['id']} | failed | — | — | — | — | — | — | — | — | False/False |");continue
        b=r['baseline_metrics']['validation']['explanation'];c=r['support_only_metrics']['validation']['explanation']
        f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
        lines.append(f"| {r['spec']['id']} | completed | {b['coverage']:.5%} | {c['coverage']:.5%} | {e['coverage']:.5%} | {f['coverage']:.5%} | {f['expected_actionable_coverage']:.5%} | {f['agreement']:.5%}/{e['agreement']:.5%} | {f['kl']:.6f}/{e['kl']:.6f} | {e['whole_tree_raw_illegal_count']}/{e['whole_tree_path_uncertified_count']} | {r['numerical_passed']}/{r['reliable_fit_only_passed']} |")
    lines+=['',f'六项科学开发门槛全部通过：{passed}。独立固定表及账本复核仍须完成，随后还需全新确认。',
        'NN及执行动作未改变；程序完整分布在未保护区域允许改变，原保护输出保持。旧开发数据，不是留出或真人效果。pilot关闭。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
