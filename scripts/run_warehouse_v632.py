#!/usr/bin/env python3
"""Single-use old-development axis/oblique experiment, no NN training owner."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json, os, sys, time, traceback
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as old, v628_data as data
from backend.training import warehouse_v632_search as search
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_program import measure
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v625 import write_new
from scripts.verify_warehouse_v625_native import real_deadline
from scripts.audit_warehouse_v630 import audited_program

VERSION = 'warehouse-v632-frozen-output-oblique-diagnostic.v1'
SOURCE = ROOT/'output/warehouse_v631/diagnostics/full-fixed-refit-matrix-20261022-03'
FILES = ('backend/training/warehouse_v632_partitions.py', 'backend/training/warehouse_v632_search.py',
         'scripts/run_warehouse_v632.py', 'tests/test_warehouse_v632_partitions.py', 'docs/warehouse_v632_plan.md')


class Budget:
    def __init__(self, path):
        self.started = time.monotonic(); self.counts = dict(node=0, leaf=0, proof=0)
        self.seq = 0; self.tail = '0'*64; self.handle = path.open('x'); self.path = path
    def check(self):
        old.require(time.monotonic()-self.started < 900, 'registered confidence search wall budget exhausted')
    def solve(self, kind, metadata, callback):
        self.check()
        old.require(kind == 'node' and self.counts[kind] < 12544, 'registered sparse confidence fit budget exhausted')
        self.counts[kind] += 1; count = self.counts[kind]
        self.event('call_started', dict(kind=kind, call=count, metadata=metadata))
        try:
            result = callback()
            self.event('call_completed', dict(kind=kind, call=count, success=bool(result.success)))
            return result
        except BaseException as error:
            self.event('call_completed', dict(kind=kind, call=count, success=False, error=type(error).__name__))
            raise
    def event(self, name, data):
        self.seq += 1; row = dict(sequence=self.seq, previous_sha256=self.tail, event=name, data=data)
        self.tail = old.digest(row); row['record_sha256'] = self.tail
        self.handle.write(json.dumps(row, separators=(',',':'))+'\n'); self.handle.flush(); os.fsync(self.handle.fileno())
    def close(self): self.handle.close(); self.path.chmod(0o444)


def sources():
    files = dict(old.source_files())
    files.update(old.bound(old.pointer(SOURCE/'plan.json'))['source_files'])
    for name in FILES: files[str(ROOT/name)] = old.file_sha(ROOT/name)
    for module in list(sys.modules.values()):
        raw = getattr(module, '__file__', None)
        if not raw: continue
        p = Path(raw)
        if not p.is_absolute() and not p.exists(): continue
        p = p.resolve()
        if p.suffix == '.py' and p.is_relative_to(ROOT): files[str(p)] = old.file_sha(p)
    return dict(sorted(files.items()))


def strict(m):
    f, e = m['feedback'], m['explanation']
    return all((f['coverage'] >= .45, e['coverage'] >= .45, f['expected_actionable_coverage'] >= .40,
        f['agreement'] >= .95, e['agreement'] >= .97, f['kl'] is not None and f['kl'] <= .20,
        e['kl'] is not None and e['kl'] <= .20, all(a[k] == 0 for a in (f,e) for k in
        ('illegal_count','path_uncertified_count','whole_tree_raw_illegal_count','whole_tree_path_uncertified_count'))))


def job(output, spec):
    folder = Path(output)/spec['id']; folder.mkdir(exist_ok=False)
    plan_ref = old.pointer(Path(output)/'plan.json'); plan = old.bound(plan_ref)
    _, previous = old.checked_plan(); previous_plan = old.bound(previous['previous_plan'])
    data_plan = old.bound(previous_plan['previous_plan']); collection = old.bound(previous_plan['collection'])
    source_result = old.bound(spec['source_result'])
    source = Program.from_dict(old.bound(source_result['program']), previous['actor_role_sha256'][spec['role']])
    budget = Budget(folder/'ledger.jsonl')
    write_new(folder/'started.json', dict(plan=plan_ref, job=spec, pid=os.getpid(), started_monotonic=budget.started))
    try:
        with real_deadline(budget.started, 900):
            cases, a, array_refs = sampled_fit(data_plan, collection, spec['fold'], spec['role'])
            parts = search.old.lane_parts(a, cases)
            program, report = search.run(source, parts, budget, oblique=spec['oblique'])
            write_new(folder/'fit-report.json', report)
            del a, parts
            outer = data.load_parts(data_plan, spec['fold'], spec['role'])
            before = {s:measure(source,a) for s,a in outer.items()}
            old.require(before == source_result['metrics'], 'registered fixed-refit baseline differs')
            search.old.qualify_current(program, outer)
            metrics = {s:measure(program,a) for s,a in outer.items()}
            frames = 0
            # Independently route and calculate full probabilities, per-leaf
            # eligibility and all exact rational dual proofs before reporting.
            independent = audited_program(program, outer, metrics)
            for split, a in outer.items():
                original = source.distributions(a['x'], a['m'])
                old.require(all(np.array_equal(x,y) for x,y in zip(original[1:], independent[split][1:])),
                            'fixed complete surrogate outputs changed on outer frames')
                frames += len(a['x'])
        write_new(folder/'program.json', program.to_dict())
        result = dict(status='completed', spec=spec, plan=plan_ref, program=old.pointer(folder/'program.json'),
            fit_report=old.pointer(folder/'fit-report.json'), baseline_metrics=before, metrics=metrics,
            strict_numeric_passed=strict(metrics['validation']),
            reliable_fit_only_passed=report['reliable_selection_passed'],
            development_passed=strict(metrics['validation']) and report['reliable_selection_passed'],
            independent_program_frames=frames, independent_leaf_certificates=len(program.leaves),
            distributions_preserved=True, counts=budget.counts, elapsed_seconds=time.monotonic()-budget.started,
            NN_updates=0, new_environment_resets=0, model_acceptance_passed=False, pilot_opened=False)
    except Exception as error:
        result = dict(status='failed', spec=spec, plan=plan_ref, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), counts=budget.counts, elapsed_seconds=time.monotonic()-budget.started,
            development_passed=False, strict_numeric_passed=False, model_acceptance_passed=False, pilot_opened=False)
    finally: budget.close()
    for p in folder.iterdir():
        if p.is_file(): p.chmod(0o444)
    write_new(folder/'result.json', result); (folder/'result.json').chmod(0o444)
    return old.pointer(folder/'result.json')


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    _, p = old.checked_plan()
    old.require(old.bound(old.pointer(SOURCE/'independent-replay.json'))['numerical_pass_count'] == 4,
                'need complete independently replayed six-unit refit before new comparison')
    config = dict(schema=VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
        source_files=sources(), source_matrix=old.pointer(SOURCE/'result.json'), actor=p['actor'],
        same_frozen_NN=True, source_method='all_six_v631_fixed_refits_no_rolewise_selection',
        arms=['axis','oblique'], maximum_depth=9, maximum_leaves=256, source_features_per_node=3,
        maximum_confidence_depth=3, minimum_fit_child_frames=40, error_class_weight=4,
        per_node_logistic_fits=7, maximum_node_fits=12544, per_unit_seconds=900,
        confidence_lanes='outer_fit_sampled_only_300_fit_150_risk_150_selection_whole_scenes',
        risk_repeats=30, minimum_passes=27, feedback_leaf_agreement=.93, explanation_leaf_agreement=.95,
        selection_feedback_agreement=.95, selection_explanation_agreement=.97, selection_KL=.20,
        selection='same_four_v630_finite_additive_allocations_then_fit_only_strict_score',
        original_subtrees_included=True, complete_surrogate_distributions_preserved=True,
        calibration_or_qualification_or_validation_parameter_fitting=False,
        data_scope='spent_development_not_fresh_confirmation_or_sealed_data', maximum_workers=3,
        BLAS_threads=1, NN_updates=0, new_environment_resets=0, training_authority=False, pilot_opened=False)
    jobs = []
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            ref=old.pointer(SOURCE/f'fold-{fold}-{role}'/'result.json')
            for arm in ('axis','oblique'):
                jobs.append(dict(id=f'{arm}-fold-{fold}-{role}', fold=fold, role=role, arm=arm,
                                 oblique=arm=='oblique', source_result=ref))
    config['jobs'] = jobs
    # Snapshot every registered repository file before the first optimizer.
    for name, sha in config['source_files'].items():
        source=Path(name) if Path(name).is_absolute() else ROOT/name
        old.require(old.file_sha(source)==sha, 'source closure changed before registration')
        target=output/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.write_bytes(source.read_bytes());target.chmod(0o444)
    write_new(output/'plan.json', config); (output/'plan.json').chmod(0o444)
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(job, str(output), spec):spec for spec in jobs}
        for future in as_completed(futures):
            ref=future.result();results.append(ref);r=old.bound(ref)
            print(json.dumps(dict(job=r['spec']['id'],status=r['status'],strict_numeric=r['strict_numeric_passed'],
                reliable=r.get('reliable_fit_only_passed'),E=None if r['status']!='completed' else
                r['metrics']['validation']['explanation']['coverage'],error=r.get('error')),ensure_ascii=False),flush=True)
    for name,sha in config['source_files'].items():old.require(old.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    old.require(old.pointer(p['actor']['path'])==p['actor'],'frozen NN bytes changed')
    rows=sorted((old.bound(ref) for ref in results),key=lambda r:r['spec']['id'])
    methods={arm:dict(completed=sum(r['status']=='completed' for r in rows if r['spec']['arm']==arm),
        numeric_passes=sum(r['strict_numeric_passed'] for r in rows if r['spec']['arm']==arm),
        development_passes=sum(r['development_passed'] for r in rows if r['spec']['arm']==arm)) for arm in ('axis','oblique')}
    selected=next((arm for arm in ('axis','oblique') if methods[arm]['development_passes']==6),None)
    out=dict(schema=VERSION,plan=old.pointer(output/'plan.json'),units=sorted(results,key=lambda r:r['path']),
        methods=methods,selected_method=selected,all_twelve_settled=len(results)==12,
        old_development_precheck_only=True,fresh_confirmation_required=True,
        NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',out);(output/'result.json').chmod(0o444)
    lines=['# v6.32 单特征／三特征置信划分开发对照','',
        '| 方法 | 折/角色 | 状态 | 反馈覆盖 | 解释覆盖 | 预期反馈 | F/E一致 | F/E KL | 非法/未证 | 数值/可靠 |',
        '|---|---|---|---:|---:|---:|---|---|---|---|']
    for r in rows:
        if r['status']!='completed':
            lines.append(f"| {r['spec']['arm']} | {r['spec']['fold']}/{r['spec']['role']} | failed | — | — | — | — | — | — | 否 |")
            continue
        f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
        lines.append(f"| {r['spec']['arm']} | {r['spec']['fold']}/{r['spec']['role']} | completed | {f['coverage']:.5%} | {e['coverage']:.5%} | {f['expected_actionable_coverage']:.5%} | {f['agreement']:.5%}/{e['agreement']:.5%} | {f['kl']:.6f}/{e['kl']:.6f} | {e['whole_tree_raw_illegal_count']}/{e['whole_tree_path_uncertified_count']} | {r['strict_numeric_passed']}/{r['reliable_fit_only_passed']} |")
    lines+=['',f'选中方法：{selected}。同一冻结NN，完整程序动作分布不变；只改变外部程序置信分区。',
            '旧开发数据；不是新确认、PPO、封存、留出或真人效果。pilot关闭。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(old.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
