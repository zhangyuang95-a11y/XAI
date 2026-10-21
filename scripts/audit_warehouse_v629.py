#!/usr/bin/env python3
"""Exact proof, independent forward and metric replay. No optimization/reset."""
from pathlib import Path
import csv, io, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.warehouse import v629_protocol as v, v628_data as data
from backend.training.warehouse_v629_program import Program
from backend.training.warehouse_v622_program import digest as tensor_digest
from backend.training.warehouse_v628_program import ParentProgram, payload
from backend.training.warehouse_v628_certificate import region, structure_info
from scripts.audit_warehouse_v628 import forward, ledger
from scripts.audit_warehouse_v626 import basic, eligible, independent_dual, same_metrics
from scripts.run_warehouse_v625 import write_new


def audit(plan, results):
    from scripts.run_warehouse_v629 import gate, initial_program
    v.require([r['job'] for r in results] == plan['jobs'], 'all24 ordered units required')
    oldplan = v.bound(plan['previous_plan']); checked = []
    for result in results:
        job = result['job']; folder = v.OUTPUT/'jobs'/job['job_id']
        row = dict(job=job, status=result['status']); checked.append(row)
        if result['status'] == 'completed':v.require(bool(result.get('solver')), 'completed result missing solver ledger')
        if result.get('solver'): row['counts'] = ledger(folder, result)
        if result['status'] != 'completed': continue
        v.require(result['plan_sha256'] == v.file_sha(v.OUTPUT/'plan.json'), 'result plan binding differs')
        v.require(v.pointer(result['program']['path']) == result['program'], 'program changed')
        p = Program.from_dict(v.bound(result['program']), plan['actor_role_sha256'][job['role']])
        _, info = structure_info(p.structure, job['capacity'])
        v.require(p.role == job['role'] and p.maximum_leaves == job['capacity']
            and result['leaf_count'] == len(p.leaves) and result['depth'] == info['depth'], 'result role/capacity/size differs')
        proofs = p.certificates(); parts = data.load_parts(oldplan, job['fold'], job['role'])
        pred = {s: forward(p, a['x'], a['m']) for s, a in parts.items()}
        for split, a in parts.items():
            actual = p.distributions(a['x'], a['m'])
            v.require(all(np.array_equal(x, y) for x, y in zip(pred[split], actual, strict=True)), 'independent forward differs')
            for purpose, selected in (('feedback',p.feedback_leaves),('explanation',p.explanation_leaves)):
                same_metrics(basic(p, a, pred[split], selected, purpose), result['metrics'][split][purpose])
        for purpose, selected, threshold in (('feedback',p.feedback_leaves,.93),('explanation',p.explanation_leaves,.95)):
            recomputed = set()
            for i in range(len(p.leaves)):
                value = basic(p, parts['eligibility'], pred['eligibility'], {i}, purpose)
                same_metrics(value, p.report['eligibility'][str(i)][purpose])
                if p.report['fit_leaf_counts'][i] and eligible(value,threshold): recomputed.add(i)
            v.require(set(selected) == recomputed, 'new eligibility differs')
        for i, leaf in enumerate(p.leaves): independent_dual(leaf, region(p.structure,i,job['capacity']), proofs[i])
        initial, initial_ref, _ = initial_program(plan, job)
        v.require(result['initial'] == initial_ref, 'baseline source provenance differs')
        reference = Program.from_dict(v.bound(result['reference']),p.actor_role_sha256)
        v.require(reference.role == job['role'] and reference.maximum_leaves == job['capacity']
            and tensor_digest(initial.structure) == tensor_digest(reference.structure)
            and [payload(leaf) for leaf in initial.leaves] == [payload(leaf) for leaf in reference.leaves],
            'reference is not the unchanged initial program')
        for split, a in parts.items():
            refpred = forward(reference,a['x'],a['m'])
            v.require(all(np.array_equal(x,y) for x,y in zip(refpred,reference.distributions(a['x'],a['m']),strict=True)),
                'reference independent forward differs')
            for purpose,selected in (('feedback',reference.feedback_leaves),('explanation',reference.explanation_leaves)):
                same_metrics(basic(reference,a,refpred,selected,purpose),result['reference_metrics'][split][purpose])
        ref_eligibility = forward(reference,parts['eligibility']['x'],parts['eligibility']['m'])
        for purpose,selected,threshold in (('feedback',reference.feedback_leaves,.93),('explanation',reference.explanation_leaves,.95)):
            recomputed = set()
            for i in range(len(reference.leaves)):
                metric = basic(reference,parts['eligibility'],ref_eligibility,{i},purpose)
                same_metrics(metric,reference.report['eligibility'][str(i)][purpose])
                if reference.report['fit_leaf_counts'][i] and eligible(metric,threshold):recomputed.add(i)
            v.require(set(selected) == recomputed, 'reference qualification differs')
        parent = ParentProgram.from_dict(v.bound(p.protection['parent']), p.actor_role_sha256)
        a = parts['validation']; original = parent.distributions(a['x'],a['m'])
        take = np.isin(original[0],list(parent.feedback_leaves|parent.explanation_leaves))
        v.require(np.array_equal(original[1][take],pred['validation'][1][take])
            and np.array_equal(original[2][take],pred['validation'][2][take]), 'protected forward differs')
        v.require(gate(result['metrics']['validation'],v.bound(result['search_report'])) == result['gate'], 'gate differs')
        row.update(program=result['program'], protected_exact_forward=True, independent_masks_and_duals=True,
            independent_reference_metrics=True)
    return dict(schema='warehouse-v629-arithmetic-audit.v1', integrity_passed=True, jobs=checked,
        scope='independent_deterministic_replay_not_external_review_or_sealed_audit', actor_updates=0, PPO_joint_steps=0)


def report(results, summary):
    rows = []
    for result in results:
        metrics = result.get('metrics',{}).get('validation',{})
        f, e = metrics.get('feedback',{}), metrics.get('explanation',{})
        ref = result.get('reference_metrics',{}).get('validation',{}).get('explanation',{}).get('coverage')
        rows.append(dict(**result['job'], status=result['status'], leaves=result.get('leaf_count'), depth=result.get('depth'),
            feedback_coverage=f.get('coverage'), explanation_coverage=e.get('coverage'),
            expected_feedback=f.get('expected_actionable_coverage'), actual_feedback=f.get('actionable_coverage'),
            feedback_agreement=f.get('agreement'), explanation_agreement=e.get('agreement'),
            feedback_KL=f.get('kl'), explanation_KL=e.get('kl'), illegal=f.get('whole_tree_raw_illegal_count'),
            unproved=f.get('whole_tree_path_uncertified_count'), v628_reference_explanation=ref,
            delta_explanation=e['coverage']-ref if e.get('coverage') is not None and ref is not None else None,
            passed=result['gate']['passed']))
    out=io.StringIO(); writer=csv.DictWriter(out,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (v.OUTPUT/'metrics-24.csv').write_text(out.getvalue())
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    lines=['# Warehouse v6.29 程序开发对照', '',
        '| 臂 | 折 | 角色 | 叶/深 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 起点E | 通过 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| '+' | '.join([r['name'],str(r['fold']),r['role'],str(r['leaves'])+'/'+str(r['depth']),
            pct(r['feedback_coverage']),pct(r['explanation_coverage']),pct(r['expected_feedback']),
            pct(r['feedback_agreement'])+'/'+pct(r['explanation_agreement']),str(r['feedback_KL'])+'/'+str(r['explanation_KL']),
            pct(r['v628_reference_explanation']),'是' if r['passed'] else '否'])+' |')
    lines += ['',f"选定容量：{summary['selected_capacity']}。",'',
        '共同v6.28开发集；非新的独立确认、封存审计、留出或真人实验。NN、执行动作及任务质量不因静态程序变化而改变。',
        '工程开销单独验收；模型训练与完整发布门槛尚未由本表替代。']
    (v.OUTPUT/'results-24.md').write_text('\n'.join(lines)+'\n')
    target=ROOT/'docs/warehouse_v629_results.md';v.require(not target.exists(),'report overwrite prohibited')
    target.write_text('\n'.join(lines)+'\n')
