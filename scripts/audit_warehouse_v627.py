"""Separate deterministic arithmetic replay; no optimization or trajectories."""
from __future__ import annotations
import csv,io,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v627_protocol as v
from backend.training.warehouse_v627_program import CertifiedSemanticProgram,inner_parts
from backend.training.warehouse_v626_certificate import region
from backend.training.warehouse_v626_program import kl
from scripts.audit_warehouse_v626 import independent_forward,basic,same_metrics,eligible,independent_dual


def replay(program,parts):
    predictions={s:independent_forward(program,a['x'],a['m']) for s,a in parts.items()}
    for s,a in parts.items():
        result=program.distributions(a['x'],a['m'])
        v.require(all(np.array_equal(x,y) for x,y in zip(result,predictions[s],strict=True)),
                  'stored hard-tree forward differs from independent arithmetic')
    for purpose,selected,threshold in (('feedback',program.feedback_leaves,.93),('explanation',program.explanation_leaves,.95)):
        actual=set()
        for i in range(len(program.leaves)):
            m=basic(program,parts['eligibility'],predictions['eligibility'],{i},purpose)
            recorded=program.report['eligibility'][str(i)][purpose]
            same_metrics({key:m[key] for key in recorded if key in m},recorded)
            if program.report['fit_leaf_counts'][i] and eligible(m,threshold):actual.add(i)
        v.require(actual==set(selected),'exact final eligibility rule differs')
    counts=np.bincount(predictions['fit'][0],minlength=len(program.leaves)).tolist()
    v.require(counts==program.report['fit_leaf_counts'],'fitted region support differs')
    for i,leaf in enumerate(program.leaves):
        independent_dual(leaf,region(program.structure,i,program.maximum_leaves),program.certificates()[i])
    return predictions


def audit(path,results):
    from scripts.run_warehouse_v626 import load_parts
    from backend.training.warehouse_v626_search import score_better
    plan=json.loads(Path(path).read_bytes());parent=v.bound(plan['parent_plan']);checked=[]
    v.require([r['job'] for r in results]==plan['jobs'],'all six ordered units required')
    for result in results:
        spec=result['job'];folder=Path(path).parent/'jobs'/spec['job_id'];counts=dict(leaf=0,node=0,proof=0)
        gradient_reserved=0;actual_gradient=0;pending=set();sequence=0;tail='0'*64
        for line in (folder/'checkpoint.json.ledger.jsonl').read_bytes().splitlines():
            row=json.loads(line);sha=row.pop('record_sha256')
            v.require(row['sequence']==sequence+1 and row['previous_sha256']==tail and v.digest(row)==sha,'solver ledger hash chain changed')
            sequence+=1;tail=sha
            if row['event']=='call_started':
                v.require(not pending,'overlapping unpaid solver work');pending.add(row['data']['call'])
                meta=row['data']['metadata'];counts[meta['solver_kind']]+=1
                steps=meta.get('reserved_gradient_steps',0)
                v.require(type(steps) is int and 0<=steps<=200,'invalid gradient-step reservation')
                gradient_reserved+=steps
            elif row['event']=='call_completed':
                v.require(row['data']['call'] in pending,'unpaid solver settlement');pending.remove(row['data']['call'])
        v.require(not pending and counts==result['solver']['counts'] and all(counts[k]<=v.recipe()['limits'][k] for k in counts),
                  'solver work exceeds categorized limits')
        v.require(gradient_reserved==result['solver']['surrogate_gradient_steps_reserved'] and gradient_reserved<=20000,
                  'surrogate gradient steps exceed predebits')
        for key,value in v.ZERO.items():v.require(type(result.get(key)) is type(value) and result[key]==value,'unexpected experiment authority')
        if result['status']!='completed':
            checked.append(dict(job=spec,status=result['status'],counts=counts,gradient_steps_reserved=gradient_reserved));continue
        v.require(result['elapsed_seconds']<5400,'unit exceeded physical time budget')
        parts=load_parts(parent,spec['job_id'])
        program=CertifiedSemanticProgram.from_dict(v.bound(result['program']),v.old.ROLE_SHA256[spec['role']])
        v.require(program.role==spec['role'] and program.maximum_leaves==plan['maximum_leaves'],'wrong model/capacity binding')
        predictions=replay(program,parts)
        for s,a in parts.items():
            for purpose,selected in (('feedback',program.feedback_leaves),('explanation',program.explanation_leaves)):
                same_metrics(basic(program,a,predictions[s],selected,purpose),result['metrics'][s][purpose])
        best=CertifiedSemanticProgram.from_dict(v.bound(result['search_best']),v.old.ROLE_SHA256[spec['role']])
        v.require(v.digest(best.structure)==v.digest(program.structure),'final refitting altered the selected topology')
        inner=inner_parts(parts['fit']);pred=replay(best,inner)
        f=basic(best,inner['validation'],pred['validation'],best.feedback_leaves,'feedback')
        e=basic(best,inner['validation'],pred['validation'],best.explanation_leaves,'explanation')
        expected=(e['coverage'],f['expected_actionable_coverage'],-float(kl(inner['validation']['p'],pred['validation'][2]).mean()),-len(best.leaves))
        search=v.bound(result['search_report'])
        v.require(search['selection_uses_only_fit'] is True and search['validation_used_for_search'] is False
            and search['exact_eligibility_rule'] is True and search['no_stagnation_stop'] is True
            and search['actual_search_seconds']<4800 and search['gradient_steps_reserved']==gradient_reserved,'search protocol differs')
        v.require(np.allclose(expected,search['score'],rtol=0,atol=1e-6),'fit-selected best does not reproduce its exact qualified score')
        score=None
        for item in search['trace']:
            better=score is None or score_better(item['score'],score)
            v.require(better==item['improved_best'] and item['actual_inner_qualification'] is True,'best trace differs')
            if better:score=item['score']
        v.require(score==search['score'],'lost globally best inner score')
        expected_gate=v.gate(result['metrics']['validation'])
        expected_gate['checks']['four_starts_completed_mandatory_exploration']=search['mandatory_completed']==[0,1,2,3]
        expected_gate['passed']=all(expected_gate['checks'].values())
        v.require(expected_gate==result['gate'],'development admission changed')
        checked.append(dict(job=spec,status='completed',counts=counts,gradient_steps_reserved=gradient_reserved,
                            program=result['program'],inner_score_replayed=list(expected)))
    return dict(schema='warehouse-v627-arithmetic-audit.v1',integrity_passed=True,plan=v.pointer(path),
        source_sha256=plan['source_sha256'],jobs=checked,
        scope='independent_deterministic_arithmetic_not_external_review_or_sealed_audit',**v.ZERO)


def report(folder,results,summary):
    rows=[]
    for r in results:
        a=r.get('metrics',{}).get('validation',{});f=a.get('feedback',{});e=a.get('explanation',{})
        rows.append(dict(**r['job'],status=r['status'],leaves=r.get('leaf_count'),
            feedback_coverage=f.get('coverage'),explanation_coverage=e.get('coverage'),
            expected_feedback=f.get('expected_actionable_coverage'),actual_sample_usable=f.get('actionable_coverage'),
            feedback_agreement=f.get('agreement'),explanation_agreement=e.get('agreement'),
            feedback_KL=f.get('kl'),explanation_KL=e.get('kl'),
            illegal=f.get('whole_tree_raw_illegal_count'),unproved=f.get('whole_tree_path_uncertified_count'),
            passed=r['gate']['passed'],**{key+'_calls':value for key,value in r['solver']['counts'].items()},
            surrogate_gradient_steps_reserved=r['solver']['surrogate_gradient_steps_reserved'],elapsed_seconds=r['elapsed_seconds']))
    s=io.StringIO();writer=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (folder/'metrics-six.csv').open('x',newline='') as f:f.write(s.getvalue())
    pct=lambda x:'—' if x is None else f'{x*100:.2f}%';num=lambda x:'—' if x is None else f'{x:.4f}'
    lines=[f"# Warehouse v6.27 {summary['stage']} 实测",'',
        '| 折 | 角色 | 实际总叶数 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 | 通过 |',
        '|---|---|---:|---:|---:|---:|---|---|---|---|']
    for r in rows:lines.append('| '+' | '.join([str(r['fold']),r['role'],str(r['leaves']),pct(r['feedback_coverage']),
        pct(r['explanation_coverage']),pct(r['expected_feedback']),pct(r['feedback_agreement'])+'/'+pct(r['explanation_agreement']),
        num(r['feedback_KL'])+'/'+num(r['explanation_KL']),str(r['illegal'])+'/'+str(r['unproved']),'是' if r['passed'] else '否'])+' |')
    lines+=['',f"全部通过：{summary['scientific_pass']}；下一阶段：{summary['next_stage']}。",'',
        '逐单元门槛：F/E覆盖45%、预期F40%、一致95%/97%、KL≤.20、整树非法/未证0。',
        '语义树与合法性/空分支全部计入32/64总容量；仅fit内部四分区搜索，资格规则与最终相同。',
        '本阶段程序的Torch梯度仅用于拟合外部程序，Actor更新、实际外部程序训练梯度和PPO联合步均0。',
        'NN始终为原v6.20完整双Actor；配送14.93、碰撞2.84、停机1/75仍为旧开发记录，本轮未重测。',
        '没有新确认数据、封存审计、留出测试或pilot；工程性能单列，完整runner开销尚未独立测量。','']
    with (folder/'results-six.md').open('x') as f:f.write('\n'.join(lines))
