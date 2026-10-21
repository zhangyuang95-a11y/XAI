#!/usr/bin/env python3
"""Independent deterministic replay, no fitting or environment construction."""
from pathlib import Path
import csv,io,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v,v628_data as data
from backend.training.warehouse_v628_program import Program,protected_indices,path_key,ParentProgram
from backend.training.warehouse_v628_certificate import region
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v6_tree_routing import leaf_indices
from backend.training.warehouse_v622_program import masked_probabilities
from scripts.audit_warehouse_v626 import basic,eligible,independent_dual,same_metrics
from scripts.run_warehouse_v625 import write_new


def forward(p,x,m):
    t=p.structure;n=len(t['split_bias']);d=domain_table()['features']
    route=leaf_indices(x,t['split_weights'],t['split_bias'],t['left_refs'],t['right_refs'],n,9)
    logits=np.empty((len(x),5),np.float32)
    for i,leaf in enumerate(p.leaves):
        take=route==i;z=np.broadcast_to(np.array(leaf['bias'],np.float32),(int(take.sum()),5)).copy()
        weights=np.array(leaf['weights'],np.float32)
        for j,index in enumerate(leaf['support']):
            column=np.multiply(np.subtract(x[take,index],np.float32(d[index]['center']),dtype=np.float32),np.float32(d[index]['scale']),dtype=np.float32)
            z=np.add(z,np.multiply(column[:,None],weights[None,:,j],dtype=np.float32),dtype=np.float32)
        logits[take]=z
    return route,logits,masked_probabilities(logits,m)


def ledger(folder,result):
    counts=dict(leaf=0,node=0,proof=0);tail='0'*64;sequence=0;pending=set()
    for line in (folder/'checkpoint.json.ledger.jsonl').read_bytes().splitlines():
        row=json.loads(line);sha=row.pop('record_sha256')
        v.require(row['sequence']==sequence+1 and row['previous_sha256']==tail and v.digest(row)==sha,'ledger hash chain differs')
        sequence+=1;tail=sha
        if row['event']=='call_started':
            v.require(not pending,'overlapping solver calls');pending.add(row['data']['call'])
            meta=row['data']['metadata'];counts[meta['solver_kind']]+=1
            if meta['phase']!='final':v.require(counts[meta['solver_kind']]<=v.recipe()['limits'][meta['solver_kind']]-v.recipe()['common_final_reserve'][meta['solver_kind']],'search consumed final reserve')
        elif row['event']=='call_completed':
            call=row['data']['call'];v.require(call in pending,'settlement without debit');pending.remove(call)
    v.require(not pending and counts==result['solver']['counts'],'unsettled/mismatched calls')
    v.require(all(counts[k]<=v.recipe()['limits'][k] for k in counts),'solver budget exceeded')
    return counts


def audit(plan,results):
    from scripts.run_warehouse_v628 import program_gate
    v.require([r['job'] for r in results]==plan['jobs'],'all ordered18 required')
    checked=[]
    for r in results:
        job=r['job'];folder=v.OUTPUT/'jobs'/job['job_id']
        if r['status']=='failed_or_unfinished':checked.append(dict(job=job,status=r['status']));continue
        counts=ledger(folder,r)
        if r['status']!='completed':checked.append(dict(job=job,status=r['status'],counts=counts));continue
        v.require(v.pointer(r['program']['path'])==r['program'],'program hash changed')
        p=Program.from_dict(v.bound(r['program']),v.old.ROLE_SHA256[job['role']]);proofs=p.certificates()
        parts=data.load_parts(plan,job['fold'],job['role']);pred={s:forward(p,a['x'],a['m']) for s,a in parts.items()}
        for s,a in parts.items():
            actual=p.distributions(a['x'],a['m'])
            v.require(all(np.array_equal(x,y) for x,y in zip(pred[s],actual,strict=True)),'independent hard/logit/probability replay differs')
            for purpose,selected in (('feedback',p.feedback_leaves),('explanation',p.explanation_leaves)):
                same_metrics(basic(p,a,pred[s],selected,purpose),r['metrics'][s][purpose])
        for purpose,selected,threshold in (('feedback',p.feedback_leaves,.93),('explanation',p.explanation_leaves,.95)):
            recomputed=set()
            for i in range(len(p.leaves)):
                metric=basic(p,parts['eligibility'],pred['eligibility'],{i},purpose)
                same_metrics(metric,p.report['eligibility'][str(i)][purpose])
                if p.report['fit_leaf_counts'][i] and eligible(metric,threshold):recomputed.add(i)
            v.require(set(selected)==recomputed,'new-data eligibility was inherited or differs')
        for i,leaf in enumerate(p.leaves):independent_dual(leaf,region(p.structure,i,job['capacity']),proofs[i])
        # Exact on-data core invariance in addition to whole-domain numeric path/payload checks.
        parent=ParentProgram.from_dict(v.bound(r['initial_parent']),p.actor_role_sha256)
        a=parts['validation'];original=parent.distributions(a['x'],a['m']);take=np.isin(original[0],list(parent.feedback_leaves|parent.explanation_leaves))
        v.require(np.array_equal(original[1][take],pred['validation'][1][take])
            and np.array_equal(original[2][take],pred['validation'][2][take]),'protected complete logits/probabilities changed')
        search=v.bound(r['search_report']);v.require(program_gate(r['metrics']['validation'],search)==r['gate'],'scientific gate differs')
        checked.append(dict(job=job,status='completed',counts=counts,protected_exact_forward=True,program=r['program']))
    return dict(schema='warehouse-v628-arithmetic-audit.v1',integrity_passed=True,jobs=checked,
        scope='independent_deterministic_replay_not_external_review_or_sealed_audit',
        actor_updates=0,PPO_joint_steps=0,pilot_opened=False)


def report(results,summary):
    rows=[]
    for r in results:
        m=r.get('metrics',{}).get('validation',{});f=m.get('feedback',{});e=m.get('explanation',{})
        ref=r.get('reference_metrics',{}).get('validation',{});base=ref.get('explanation',{}).get('coverage')
        search=v.bound(r['search_report']) if r.get('search_report') else {}
        rows.append(dict(**r['job'],status=r['status'],actual_leaves=r.get('leaf_count'),actual_depth=r.get('depth'),
            protected_leaves=r.get('protected_leaves'),feedback_coverage=f.get('coverage'),explanation_coverage=e.get('coverage'),
            expected_feedback=f.get('expected_actionable_coverage'),actual_feedback=f.get('actionable_coverage'),
            feedback_agreement=f.get('agreement'),explanation_agreement=e.get('agreement'),feedback_KL=f.get('kl'),explanation_KL=e.get('kl'),
            illegal=f.get('whole_tree_raw_illegal_count'),unproved=f.get('whole_tree_path_uncertified_count'),
            same_data_parent_explanation=base,delta_explanation=e.get('coverage')-base if e.get('coverage') is not None and base is not None else None,
            mandatory_exploration=search.get('mandatory_complete'),budget_limited=search.get('budget_limited'),
            leaf_calls=r.get('solver',{}).get('counts',{}).get('leaf'),node_calls=r.get('solver',{}).get('counts',{}).get('node'),
            proof_calls=r.get('solver',{}).get('counts',{}).get('proof'),passed=r['gate']['passed']))
    out=io.StringIO();w=csv.DictWriter(out,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (v.OUTPUT/'metrics-18.csv').write_text(out.getvalue())
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    lines=['# Warehouse v6.28 容量对照实测','',
        '| 容量 | 折 | 角色 | 实际叶/深 | F覆盖 | E覆盖 | 预期F | F/E一致 | KL(F/E) | 原E覆盖 | 通过 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| '+' | '.join([str(r['capacity']),str(r['fold']),r['role'],str(r['actual_leaves'])+'/'+str(r['actual_depth']),
            pct(r['feedback_coverage']),pct(r['explanation_coverage']),pct(r['expected_feedback']),pct(r['feedback_agreement'])+'/'+pct(r['explanation_agreement']),
            str(r['feedback_KL'])+'/'+str(r['explanation_KL']),pct(r['same_data_parent_explanation']),'是' if r['passed'] else '否'])+' |')
    collection=v.bound(v.pointer(v.OUTPUT/'collection/result.json'))
    outcomes=collection['outcomes'];quality={}
    for mode in ('sampled','argmax'):
        sample=[r for r in outcomes if r['case']['sampling_mode']==mode]
        quality[mode]=dict(episodes=len(sample),deliveries=float(np.mean([r['deliveries'] for r in sample])),
            collisions=float(np.mean([r['robot_collisions'] for r in sample])),shutdown_episodes=sum(r['shutdown'] for r in sample),
            shutdown_rate=float(np.mean([r['shutdown'] for r in sample])))
    write_new(v.OUTPUT/'common-NN-quality.json',dict(quality=quality,collection=v.pointer(v.OUTPUT/'collection/result.json'),
        same_for_all_capacities=True,development_only=True,actor_updates=0,PPO_joint_steps=0))
    lines+=['',f"选中容量：{summary['selected_capacity']}。",'',
        'F/E覆盖≥45%、预期F≥40%、F/E一致≥95%/97%、KL≤.20、整树非法及未证路径均为0，并完成全部可编辑区域检查。',
        '三折是数据分区，非三个独立NN；本轮冻结同一个v6.20双Actor，容量变化不会改变NN配送能力。',
        '原v6.26深7和本轮深9的比较包含深度与数据差异；三种新容量之间深度、数据、搜索预算相同。',
        '预算耗尽不能证明容量无效；失败/未完成单元的指标留空，不能视为零。',
        '程序只用于训练反馈或解释。独立确认、PPO、封存审计、留出与pilot均未执行。',
        '工程检查点开销单独验收，尚未独立测量。','']
    (v.OUTPUT/'results-18.md').write_text('\n'.join(lines))
    # Machine-readable curve preserves every role/fold rather than pooling failures.
    write_new(v.OUTPUT/'capacity-curve.json',dict(rows=rows,quality=quality,summary=summary))
    target=ROOT/'docs/warehouse_v628_results.md';v.require(not target.exists(),'report overwrite prohibited');target.write_text('\n'.join(lines))
    target=ROOT/'experiments/warehouse/evidence/warehouse-v628-capacity.json'
    write_new(target,dict(summary=v.pointer(v.OUTPUT/'summary.json'),metrics=v.pointer(v.OUTPUT/'metrics-18.csv'),scientific_pass=summary['scientific_pass']))
