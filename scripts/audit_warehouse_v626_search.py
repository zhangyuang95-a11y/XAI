"""Numerical replay of six fitted hard-tree search results, no optimization."""
from __future__ import annotations
import csv,io,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v626_protocol as v
from scripts.audit_warehouse_v626 import independent_forward,basic,same_metrics,eligible,independent_dual
from backend.training.warehouse_v626_program import Program
from backend.training.warehouse_v626_certificate import region


def audit(path,results):
    from scripts.run_warehouse_v626 import load_parts
    from scripts.run_warehouse_v626_search import recipe
    from backend.training.warehouse_v626_search import score_better
    plan=json.loads(Path(path).read_bytes());a0=v.bound(plan['A0_plan']);checked=[]
    v.require([r['job'] for r in results]==plan['jobs'],'six complete ordered search units required')
    for result in results:
        spec=result['job'];folder=Path(path).parent/'jobs'/spec['job_id']
        counts=dict(leaf=0,node=0,proof=0);pending=set();sequence=0;tail='0'*64
        for line in (folder/'checkpoint.json.ledger.jsonl').read_bytes().splitlines():
            r=json.loads(line);sha=r.pop('record_sha256')
            v.require(r['sequence']==sequence+1 and r['previous_sha256']==tail and v.digest(r)==sha,'search ledger chain changed')
            sequence+=1;tail=sha
            if r['event']=='call_started':
                v.require(not pending,'parallel unsettled solver work');pending.add(r['data']['call'])
                counts[r['data']['metadata']['solver_kind']]+=1
            elif r['event']=='call_completed':
                v.require(r['data']['call'] in pending,'unpaid solver settlement');pending.remove(r['data']['call'])
        v.require(not pending and counts==result['solver']['counts'] and all(counts[k]<=recipe()['limits'][k] for k in counts),'solver work differs from budget')
        if result['status']!='completed':checked.append(dict(job=spec,status=result['status'],solver_counts=counts));continue
        data=v.bound(result['program']);program=Program.from_dict(data,v.old.ROLE_SHA256[spec['role']])
        v.require(program.maximum_leaves==plan['maximum_leaves'] and program.arm=='C3' and program.role==spec['role'],'wrong search asset binding')
        best=Program.from_dict(v.bound(result['search_best']),v.old.ROLE_SHA256[spec['role']])
        v.require(best.report is None and v.digest(best.structure)==v.digest(program.structure),'fit-selected artifact changed at final fitting')
        search=v.bound(result['search_report']);v.require(search['selection_uses_only_fit'] is True and search['validation_used_for_search'] is False,'search evidence scope changed')
        score=search['initial_score']
        for item in search['trace']:
            if item['accepted']:
                v.require(score_better(item['score'],score),'accepted candidate did not improve registered fit-only objective');score=item['score']
        v.require(list(score)==search['score'],'fit selection trace lost its best score')
        parts=load_parts(a0,spec['job_id']);pred={s:independent_forward(program,a['x'],a['m']) for s,a in parts.items()}
        for s,a in parts.items():
            current=program.distributions(a['x'],a['m'])
            v.require(all(np.array_equal(x,y) for x,y in zip(pred[s],current,strict=True)),'search program frame outputs differ')
            for purpose,selected in (('feedback',program.feedback_leaves),('explanation',program.explanation_leaves)):
                same_metrics(basic(program,a,pred[s],selected,purpose),result['metrics'][s][purpose])
        for purpose,selected,threshold in (('feedback',program.feedback_leaves,.93),('explanation',program.explanation_leaves,.95)):
            actual=set()
            for i in range(len(program.leaves)):
                m=basic(program,parts['eligibility'],pred['eligibility'],{i},purpose)
                same_metrics(m,program.report['eligibility'][str(i)][purpose])
                if program.report['fit_leaf_counts'][i] and eligible(m,threshold):actual.add(i)
            v.require(actual==set(selected),'independent eligibility differs')
        for i,leaf in enumerate(program.leaves):independent_dual(leaf,region(program.structure,i,program.maximum_leaves),program.certificates()[i])
        v.require(result['gate']==v.gate(result['metrics']['validation']),'scientific gate changed')
        checked.append(dict(job=spec,status='completed',solver_counts=counts,program=result['program']))
    return dict(schema='warehouse-v626-search-arithmetic-audit.v1',integrity_passed=True,
        audit_scope='separate_deterministic_arithmetic_replay_not_external_reviewer',
        plan=v.pointer(path),source_sha256=plan['source_sha256'],jobs=checked,
        performance_does_not_block_science=True,**v.ZERO)


def report(folder,results,summary):
    rows=[]
    for r in results:
        a=r.get('metrics',{}).get('validation',{});f=a.get('feedback',{});e=a.get('explanation',{})
        rows.append(dict(**r['job'],status=r['status'],leaves=r.get('leaf_count'),feedback_coverage=f.get('coverage'),
            explanation_coverage=e.get('coverage'),expected_feedback_coverage=f.get('expected_actionable_coverage'),
            feedback_agreement=f.get('agreement'),explanation_agreement=e.get('agreement'),feedback_KL=f.get('kl'),
            explanation_KL=e.get('kl'),illegal=f.get('whole_tree_raw_illegal_count'),uncertified=f.get('whole_tree_path_uncertified_count'),
            passed=r['gate']['passed'],**{k+'_calls':n for k,n in r['solver']['counts'].items()},elapsed_seconds=r['elapsed_seconds']))
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (folder/'metrics-six.csv').open('x',newline='') as h:h.write(out.getvalue())
    pct=lambda v:'—' if v is None else f'{100*v:.2f}%';num=lambda v:'—' if v is None else f'{v:.4f}'
    lines=[f"# Warehouse v6.26 {summary['stage']} 实测",'',
        '| 折 | 角色 | 叶数 | 反馈覆盖 | 解释覆盖 | 预期反馈 | F/E一致 | F/E KL | 非法/未证 | 通过 |',
        '|---|---|---:|---:|---:|---:|---|---|---|---|']
    for r in rows:lines.append('| '+' | '.join([str(r['fold']),r['role'],str(r['leaves']),pct(r['feedback_coverage']),pct(r['explanation_coverage']),
        pct(r['expected_feedback_coverage']),pct(r['feedback_agreement'])+'/'+pct(r['explanation_agreement']),num(r['feedback_KL'])+'/'+num(r['explanation_KL']),
        str(r['illegal'])+'/'+str(r['uncertified']),'是' if r['passed'] else '否'])+' |')
    lines+=['',f"全部六单元通过：{summary['scientific_pass']}；下一阶段：{summary['next_stage']}。",'',
        '开发门槛未降低：F/E覆盖45%，预期反馈40%，一致95%/97%，KL≤.20，整树非法/未证0。',
        '只使用旧275局、原三折。结构/支持选择仅使用fit内部场景分区；最终fit重拟合后校准、资格、验证。',
        '本阶段NN哈希保持原v6.20，环境重置和NN/PPO更新均0。配送14.93、碰撞2.84、停机1/75仍为旧开发记录。',
        '工程性能单独验收；当前没有开放pilot、消耗新确认场景或触碰封存集。','']
    with (folder/'results-six.md').open('x') as h:h.write('\n'.join(lines))
