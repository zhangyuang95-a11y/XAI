#!/usr/bin/env python3
"""Read-only arithmetic replay. Never optimizes, collects or updates an NN."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction as F
import csv,io,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v626_protocol as v
from backend.training.warehouse_v626_program import Program
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v626_certificate import public_polytope,polynomial,region
from backend.training.warehouse_v6_tree_routing import leaf_indices
from backend.training.warehouse_v622_program import masked_probabilities,_wilson


def independent_forward(program,x,m):
    t=program.structure;n=len(t['split_bias']);d=domain_table()['features']
    routes=leaf_indices(x,np.asarray(t['split_weights'],np.float32).reshape(n,x.shape[1]),t['split_bias'],t['left_refs'],t['right_refs'],n,
        6 if program.maximum_leaves==32 else 7)
    z=np.empty((len(x),5),np.float32)
    for i,model in enumerate(program.leaves):
        take=routes==i;b=np.asarray(model['bias'],np.float32);w=np.asarray(model['weights'],np.float32)
        out=np.broadcast_to(b,(int(take.sum()),5)).copy()
        for j,index in enumerate(model['support']):
            column=x[take,index]
            if model.get('normalized'):
                column=np.multiply(np.subtract(column,np.float32(d[index]['center']),dtype=np.float32),np.float32(d[index]['scale']),dtype=np.float32)
            out=np.add(out,np.multiply(column[:,None],w[None,:,j],dtype=np.float32),dtype=np.float32)
        z[take]=out
    return routes,z,masked_probabilities(z,m)


def basic(program,a,pred,selected,purpose):
    route,z,q=pred;p=a['p'];top=q.argmax(1);raw=z.argmax(1);nn=p.argmax(1)
    chosen=np.isin(route,list(selected));n=int(chosen.sum());mask=a['m']
    legal=mask[np.arange(len(mask)),raw]==1
    proofs=program.certificates();path=np.array([int(k) in proofs[int(i)] for k,i in zip(raw,route,strict=True)])
    matches=legal&path&(top==a['sampled']);expected=legal*path*p[np.arange(len(p)),top]
    if purpose=='feedback':matches&=a['sampled']==nn;expected*=top==nn
    with np.errstate(divide='ignore',invalid='ignore'):
        loss=np.sum(p*(np.log(np.maximum(p,np.float32(1e-8)))-np.log(np.maximum(q,np.float32(1e-8)))),axis=1)
    counts=Counter(r['episode'] for r,keep in zip(a['rows'],chosen,strict=True) if keep)
    agreement=float(np.sum((top==nn)&chosen)/n) if n else 0.
    return dict(rows=n,physical_frame_count=len(p),coverage=n/len(p),
        agreement=agreement,agreement_wilson_lower_90=_wilson(agreement,n),
        kl=float(loss[chosen].mean()) if n else None,
        expected_actionable_coverage=float(expected[chosen].sum()/len(p)),actionable_coverage=float(np.sum(chosen&matches)/len(p)),
        illegal_count=int(np.sum(chosen&~legal)),path_uncertified_count=int(np.sum(chosen&~path)),
        whole_tree_raw_illegal_count=int(np.sum(~legal)),whole_tree_path_uncertified_count=int(np.sum(~path)),
        distinct_episode_count=len(counts),distinct_scenario_seed_count=len({r['base_seed'] for r,keep in zip(a['rows'],chosen,strict=True) if keep}),
        max_episode_fraction=max(counts.values())/n if n else 0.)


def eligible(m,threshold):
    return (m['rows']>=20 and m['distinct_episode_count']>=3 and m['distinct_scenario_seed_count']>=3
        and m['max_episode_fraction']<=.5 and m['agreement']>=threshold
        and m['agreement_wilson_lower_90']>=threshold-.08 and m['kl'] is not None and m['kl']<=.25
        and m['illegal_count']==m['path_uncertified_count']==0)


def independent_dual(leaf,path,legal):
    """Second upper-bound implementation using exact interval residual sums."""
    for proof in leaf['certificate']['checks']:
        ids,A,b,bounds=public_polytope(leaf,path,proof['kind']=='path')
        c,constant=polynomial(leaf,proof['action'],leaf['witness'],ids)
        y=[F.from_float(float(value)) for value in proof['dual']]
        v.require(len(y)==len(b) and all(value>=0 for value in y),'bad nonnegative proof multiplier')
        dual=constant+sum((value*F.from_float(float(bound)) for value,bound in zip(y,b,strict=True)),F(0))
        for j,(lo,hi) in enumerate(bounds):
            residual=c[j]-sum((y[k]*F.from_float(float(A[k,j])) for k in range(len(y))),F(0))
            dual+=residual*F.from_float(float(hi if residual>=0 else lo))
        v.require([dual.numerator,dual.denominator]==proof['upper'] and dual < -F(*proof['required']),
            'independent exact dual upper bound failed')


def same_metrics(a,b):
    for key,expected in a.items():
        observed=b.get(key)
        if expected is None:v.require(observed is None,'unexpected nonempty metric '+key)
        elif isinstance(expected,int):v.require(type(observed) is int and observed==expected,'count mismatch '+key)
        else:v.require(type(observed) in (int,float) and math.isfinite(observed) and abs(observed-expected)<=1e-6,'metric mismatch '+key)


def audit(plan_path,results):
    from scripts.run_warehouse_v626 import load_parts
    plan=json.loads(Path(plan_path).read_bytes());checked=[];datasets={};bound=[]
    v.require([r['job'] for r in results]==plan['jobs'],'all 24 ordered units required')
    for result in results:
        spec=result['job'];directory=Path(plan_path).parent/'jobs'/spec['job_id'];key=f"fold-{spec['fold']}-{spec['role']}"
        events_path=directory/'checkpoint.json.calls.jsonl'
        # The checkpoint ledger filename is part of the unchanged CAS backend.
        if not events_path.exists():
            from backend.training.warehouse_v625_checkpoint import CheckpointSession
            events_path=Path(str(directory/'checkpoint.json')+'.ledger.jsonl')
        v.require(events_path.is_file(),'durable solver ledger missing')
        sequence=0;tail='0'*64;pending=set();counts=dict(leaf=0,node=0,proof=0)
        for line in events_path.read_bytes().splitlines():
            row=json.loads(line);sha=row.pop('record_sha256')
            v.require(row['sequence']==sequence+1 and row['previous_sha256']==tail and v.digest(row)==sha,'solver ledger hash chain differs')
            sequence+=1;tail=sha
            if row['event']=='call_started':
                token=row['data']['call'];v.require(not pending,'overlapping uncommitted solver calls')
                pending.add(token);counts[row['data']['metadata']['solver_kind']]+=1
            elif row['event']=='call_completed':
                token=row['data']['call'];v.require(token in pending,'solver settlement has no debit');pending.remove(token)
        v.require(not pending and counts==result['solver']['counts'],'solver debit counts differ')
        v.require(all(counts[k]<=v.recipe()['limits_A0'][k] for k in counts),'solver budget exceeded')
        if result['status']!='completed':
            checked.append(dict(job=spec,status=result['status'],counts=counts));continue
        ref=result['program'];v.require(v.pointer(ref['path'])==ref,'program hash changed')
        data=json.loads(Path(ref['path']).read_bytes());program=Program.from_dict(data,v.old.ROLE_SHA256[spec['role']])
        v.require(program.role==spec['role'] and program.arm==spec['arm'] and program.maximum_leaves==32,'program unit identity differs')
        if key not in datasets:datasets[key]=load_parts(plan,key)
        parts=datasets[key];pred={s:independent_forward(program,a['x'],a['m']) for s,a in parts.items()}
        for s,a in parts.items():
            observed=program.distributions(a['x'],a['m'])
            v.require(all(np.array_equal(x,y) for x,y in zip(pred[s],observed,strict=True)),'independent raw logits/probabilities differ')
            for purpose,selected in (('feedback',program.feedback_leaves),('explanation',program.explanation_leaves)):
                same_metrics(basic(program,a,pred[s],selected,purpose),result['metrics'][s][purpose])
        for purpose,selected,threshold in (('feedback',program.feedback_leaves,.93),('explanation',program.explanation_leaves,.95)):
            actual=set()
            for i in range(len(program.leaves)):
                m=basic(program,parts['eligibility'],pred['eligibility'],{i},purpose)
                same_metrics(m,program.report['eligibility'][str(i)][purpose])
                if program.report['fit_leaf_counts'][i] and eligible(m,threshold):actual.add(i)
            v.require(set(selected)==actual,'qualified leaves differ from independent eligibility')
        if spec['arm'] in ('C2','C3'):
            for i,leaf in enumerate(program.leaves):independent_dual(leaf,region(program.structure,i),program.certificates()[i])
        v.require(result['gate']==v.gate(result['metrics']['validation']),'reported scientific gate differs')
        checked.append(dict(job=spec,status='completed',counts=counts,program=ref))
        bound.append(v.pointer(events_path))
    return dict(schema='warehouse-v626-independent-arithmetic-audit.v1',integrity_passed=True,
        audit_scope='separate_deterministic_arithmetic_replay_not_external_reviewer',
        plan=v.pointer(plan_path),source_sha256=plan['source_sha256'],jobs=checked,ledgers=bound,
        numerical_replay_includes_all_physical_frames=True,performance_does_not_block_science=True,**v.ZERO)


def report(folder,results,summary):
    from scripts.run_warehouse_v625 import write_new
    rows=[]
    for result in results:
        m=result.get('metrics',{}).get('validation',{});f=m.get('feedback',{});e=m.get('explanation',{})
        row={**result['job'],'status':result['status'],'feedback_coverage':f.get('coverage'),
            'explanation_coverage':e.get('coverage'),'expected_feedback_coverage':f.get('expected_actionable_coverage'),
            'actual_sample_usable_coverage':f.get('actionable_coverage'),'feedback_agreement':f.get('agreement'),
            'explanation_agreement':e.get('agreement'),'feedback_KL':f.get('kl'),'explanation_KL':e.get('kl'),
            'illegal':f.get('whole_tree_raw_illegal_count'),'uncertified':f.get('whole_tree_path_uncertified_count'),
            'passed':result['gate']['passed'],**{k+'_calls':value for k,value in result['solver']['counts'].items()},
            'elapsed_seconds':result['elapsed_seconds']}
        rows.append(row)
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (folder/'metrics-24.csv').open('x',newline='') as h:h.write(out.getvalue())
    fmt=lambda value:'—' if value is None else f'{100*value:.2f}%'
    num=lambda value:'—' if value is None else f'{value:.4f}'
    lines=['# Warehouse v6.26 A0：24项固定拓扑实测','',
        '| 折 | 角色 | 方法 | 反馈覆盖 | 解释覆盖 | 预期反馈 | 反馈一致 | 解释一致 | F/E KL | 非法/未证 | 通过 |',
        '|---|---|---|---:|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        lines.append('| '+' | '.join([str(r['fold']),r['role'],r['arm'],fmt(r['feedback_coverage']),fmt(r['explanation_coverage']),
            fmt(r['expected_feedback_coverage']),fmt(r['feedback_agreement']),fmt(r['explanation_agreement']),
            num(r['feedback_KL'])+'/'+num(r['explanation_KL']),str(r['illegal'])+'/'+str(r['uncertified']),'是' if r['passed'] else '否'])+' |')
    lines+=['',f"C3逐角色/折全部通过：{summary['scientific_pass']}；下一阶段：{summary['next_stage']}。",'',
        '门槛：反馈/解释覆盖各45%，预期反馈40%，一致率95%/97%，KL≤0.20，整树非法/未证均0。',
        '全部为旧275局、原三折上的开发描述。各方法从同一S1拓扑重新拟合，不能把旧S1原系数成绩当成本轮C0。',
        '没有NN更新或新环境轨迹；配送14.93、碰撞2.84、停机1/75仍为历史开发记录，未在本轮重测。',
        '拟合覆盖/预期反馈/采样可用覆盖均不是实际接受训练梯度。实际Actor更新和程序梯度为0。',
        '各结果JSON同时包含任务来源、五动作、低电量、协作冲突等分层指标。工程性能独立记录，不能阻止模型指标报告。','']
    with (folder/'results-24.md').open('x') as h:h.write('\n'.join(lines))
