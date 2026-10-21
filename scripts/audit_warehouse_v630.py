#!/usr/bin/env python3
"""Independent arithmetic replay of fixed confidence partitions, no fitting."""
from pathlib import Path
import csv,io,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v630_partition_search as search
from backend.training import warehouse_v630_partition_proposals as proposals
from backend.training.warehouse_v630_program import Program
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import basic,eligible,same_metrics,independent_dual


def ledger(folder,result):
    counts=dict(leaf=0,node=0,proof=0);pending=set();sequence=0;tail='0'*64
    for raw in (folder/'checkpoint.json.ledger.jsonl').read_bytes().splitlines():
        row=json.loads(raw);sha=row.pop('record_sha256')
        v.require(row['sequence']==sequence+1 and row['previous_sha256']==tail and v.digest(row)==sha,'v630 ledger chain differs')
        sequence+=1;tail=sha
        if row['event']=='call_started':
            v.require(not pending,'overlapping confidence fits');pending.add(row['data']['call'])
            counts[row['data']['metadata']['solver_kind']]+=1
        elif row['event']=='call_completed':
            call=row['data']['call'];v.require(call in pending,'confidence fit settlement has no debit');pending.remove(call)
    v.require(not pending and counts==result['solver']['counts']
        and all(counts[k]<=v.recipe()['limits'][k] for k in counts),'confidence fit counts, pending work or budget differs')
    return counts


def audited_program(program,parts,expected):
    pred={split:forward(program,a['x'],a['m']) for split,a in parts.items()}
    for split,a in parts.items():
        direct=program.distributions(a['x'],a['m'])
        v.require(all(np.array_equal(x,y) for x,y in zip(pred[split],direct,strict=True)),'v630 independent forward differs')
        for purpose,selected in [('feedback',program.feedback_leaves),('explanation',program.explanation_leaves)]:
            same_metrics(basic(program,a,pred[split],selected,purpose),expected[split][purpose])
    for purpose,selected,threshold in [('feedback',program.feedback_leaves,.93),('explanation',program.explanation_leaves,.95)]:
        qualified=set()
        for leaf in range(len(program.leaves)):
            actual=basic(program,parts['eligibility'],pred['eligibility'],{leaf},purpose)
            same_metrics(actual,program.report['eligibility'][str(leaf)][purpose])
            if program.report['fit_leaf_counts'][leaf] and eligible(actual,threshold):qualified.add(leaf)
        v.require(qualified==set(selected),'refined child eligibility does not reproduce')
    for leaf,model in enumerate(program.leaves):independent_dual(model,search.region(program.structure,leaf,256),program.certificates()[leaf])
    return pred


def audit_search(plan,result,source,report):
    job=result['job'];scope=job['scope'];arrays,cases,refs=v.fit_data(plan,job['fold'],job['role']);parts=search.lane_parts(arrays,cases)
    v.require(report['input_arrays']==refs and report['scope']==scope and report['mandatory_complete'] is True,'search provenance or mandatory sweep differs')
    base=search.from_previous(source);base.maximum_leaves=256;base.validate()
    if scope=='control':
        v.require(report['inspected_leaves']==[] and report['candidates']==[] and result['solver']['counts']==dict(leaf=0,node=0,proof=0),
            'control fit or candidate search is not zero')
        return dict(control_no_fits=True)
    v.require(report['inspected_leaves']==list(range(len(source.leaves))) and report['source_leaf_count']==len(source.leaves),'confidence first sweep incomplete')
    stored=v.bound(report['option_tables']);pred={s:base.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
    signals={s:search.frame_signals(base,parts[s],value) for s,value in pred.items()}
    risk=search.risk_subsets(parts['eligibility']['rows']);groups={};local={}
    allowed=set(search.semantic.verify(base)['protected_descendants']) if scope=='protected' else set(range(len(base.leaves)))
    records=report['proposal_records'];v.require(len(records)==len(base.leaves),'missing local predicate records')
    fits=0
    for leaf,record in enumerate(records):
        v.require(record['source_leaf']==leaf,'predicate source leaf reordered')
        if leaf not in allowed:v.require(record['fit_called'] is False and record.get('reason')=='outside_registered_scope','out-of-scope classifier fit')
        if record['fit_called']:
            fits+=1;take=pred['fit'][0]==leaf;target=parts['fit']['p'][take]
            from hashlib import sha256
            expected=search.digest(dict(shape=list(target.shape),bytes_sha256=sha256(target.tobytes()).hexdigest()))
            v.require(record['source_target_sha256']==expected and record['fit_frames']==int(take.sum())
                and record['classifier_outputs_discarded'] is True,'CART target provenance differs')
        options,details=search.local_table(base,leaf,record['proposals'],parts,pred,signals,risk)
        groups[f'{leaf:04d}']=options;local[f'{leaf:04d}']=details
    v.require(fits==result['solver']['counts']['node'] and search.digest(groups)==report['fixed_option_tables_sha256']
        and search.digest(groups)==search.digest(stored['groups']) and search.digest(local)==search.digest(stored['local']),
        'independent local tables or actual classifier count differs')
    n=len(parts['validation']['rows']);best=None
    v.require(len(report['candidates'])==4,'four fixed allocation preferences required')
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={group:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)
            +weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for group,options in groups.items()}
        allocation=proposals.allocate(weighted,256);record=report['candidates'][index]
        v.require(record['weight']==weight and record['allocation']==allocation,'fixed-table allocation does not reproduce')
        partitions={int(group):local[group][option]['structure'] for group,option in allocation['selected'].items()}
        program,derivations=search.assemble(base,partitions);search.qualify_current(program,parts);metrics=search.measure(program,parts['validation'])
        reliable={key:sum(local[group][option]['totals'][key] for group,option in allocation['selected'].items()) for key in ('E','F','expected_F')}
        passed=reliable['E']/n>=.45 and reliable['F']/n>=.45 and reliable['expected_F']/n>=.40
        score=(int(passed),*search.score_metrics(metrics,metrics['explanation']['whole_tree_kl'],len(program.leaves)))
        v.require(record['metrics']==metrics and record['reliable_selection_counts']==reliable and record['reliable_selection_passed']==passed
            and record['score']==list(score) and search.digest(record['derivations'])==search.digest(derivations),'candidate score, reliability or exact derivation differs')
        if best is None or score>best[0]:best=(score,index,program)
    saved=Program.from_dict(v.bound(result['search_program']),source.actor_role_sha256)
    v.require(best[1]==report['best_candidate'] and saved.to_dict()==best[2].to_dict()
        and report['reliable_selection_passed']==report['candidates'][best[1]]['reliable_selection_passed'],'selected fit-only candidate differs')
    return dict(local_tables_replayed=True,fixed_additive_allocations_replayed=True,node_fits=fits,selected=best[1])


def audit(plan,results):
    from scripts.run_warehouse_v630 import gate
    v.require(len(plan['jobs'])==len(results)==18 and [r['job'] for r in results]==plan['jobs'],'all18 ordered results required')
    dependency=v.bound(v.pointer(v.OUTPUT/'dependency-cleared.json'));v.require(v.old_completion()==dependency['completion'],'upstream close evidence changed')
    old=v.bound(plan['previous_plan']);data_plan=v.bound(old['previous_plan']);checked=[]
    for result in results:
        job=result['job'];folder=v.OUTPUT/'jobs'/job['job_id'];row=dict(job=job,status=result['status']);checked.append(row)
        if result.get('solver'):row['counts']=ledger(folder,result)
        if result['status']!='completed':continue
        source,binding=v.source_program(plan,job['fold'],job['role']);v.require(result['source']==binding,'import source differs')
        report=v.bound(result['search_report']);row['search']=audit_search(plan,result,source,report)
        program=Program.from_dict(v.bound(result['program']),source.actor_role_sha256)
        reference=Program.from_dict(v.bound(result['reference']),source.actor_role_sha256)
        parts=data.load_parts(data_plan,job['fold'],job['role'])
        pred=audited_program(program,parts,result['metrics']);ref=audited_program(reference,parts,result['reference_metrics'])
        for split,a in parts.items():
            initial=forward(source,a['x'],a['m'])
            v.require(np.array_equal(initial[1],pred[split][1]) and np.array_equal(initial[2],pred[split][2])
                and np.array_equal(initial[1],ref[split][1]) and np.array_equal(initial[2],ref[split][2]),'frozen whole-surrogate distribution changed')
        v.require(result['gate']==gate(result['metrics']['validation'],report,job['scope'])
            and result['leaf_count']==len(program.leaves) and result['depth']==search.structure_info(program.structure,256)[1]['depth'],
            'completed gate or program complexity differs')
        row.update(program=result['program'],all_old_distributions_preserved=True,independent_numeric_and_qualification_replay=True)
    quality=v.common_quality(plan)
    return dict(schema='warehouse-v630-independent-arithmetic-audit.v1',integrity_passed=True,jobs=checked,common_quality=quality,
        scope='deterministic_replay_not_external_review_or_final_sealed_audit',classifier_fits=0,NN_updates=0,environment_resets=0,pilot_opened=False)


def report(results,summary):
    rows=[]
    for r in results:
        m=r.get('metrics',{}).get('validation',{});f=m.get('feedback',{});e=m.get('explanation',{})
        rows.append(dict(**r['job'],status=r['status'],leaves=r.get('leaf_count'),depth=r.get('depth'),
            feedback_coverage=f.get('coverage'),explanation_coverage=e.get('coverage'),expected_feedback=f.get('expected_actionable_coverage'),
            feedback_agreement=f.get('agreement'),explanation_agreement=e.get('agreement'),feedback_KL=f.get('kl'),explanation_KL=e.get('kl'),
            illegal=e.get('whole_tree_raw_illegal_count'),unproved=e.get('whole_tree_path_uncertified_count'),passed=r['gate']['passed']))
    stream=io.StringIO();writer=csv.DictWriter(stream,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (v.OUTPUT/'metrics-18.csv').write_text(stream.getvalue())
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    lines=['# v6.30 冻结输出分区：18项开发指标','',
        '| 方法 | 折/角色 | 状态 | 叶/深 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 | 通过 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| '+' | '.join([row['name'],str(row['fold'])+'/'+row['role'],row['status'],str(row['leaves'])+'/'+str(row['depth']),
            pct(row['feedback_coverage']),pct(row['explanation_coverage']),pct(row['expected_feedback']),
            pct(row['feedback_agreement'])+'/'+pct(row['explanation_agreement']),str(row['feedback_KL'])+'/'+str(row['explanation_KL']),
            str(row['illegal'])+'/'+str(row['unproved']),str(row['passed'])])+' |')
    lines+=['',f"选定方法：{summary['selected_method']}。全部方法共享冻结NN，分区不改变原程序的完整分布。",
        '共同已花费开发数据；不是新独立确认、NN训练、封存审计、留出或真人效果。pilot关闭。']
    (v.OUTPUT/'results-18.md').write_text('\n'.join(lines)+'\n')
    target=ROOT/'docs/warehouse_v630_results.md';v.require(not target.exists(),'v630 result report overwrite prohibited');target.write_text('\n'.join(lines)+'\n')
