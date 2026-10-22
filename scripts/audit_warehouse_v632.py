#!/usr/bin/env python3
"""Read-only replay of registered v632 options, allocation, ledgers and outputs.

No classifier/leaf fitting, no LP solving, no environment or NN execution.
This audits spent-data development evidence, never sealed scientific admission.
"""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v, v628_data as data
from backend.training import warehouse_v632_search as search
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v626_program import _support_cost
from backend.training.warehouse_v630_partition_proposals import allocate
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v625 import write_new


def ledger(path, expected):
    tail='0'*64;pending=None;count=0
    for index,line in enumerate(path.read_text().splitlines(),1):
        row=json.loads(line);sha=row.pop('record_sha256')
        v.require(row['sequence']==index and row['previous_sha256']==tail and v.digest(row)==sha,'optimizer ledger chain differs')
        tail=sha;d=row['data']
        if row['event']=='call_started':
            count+=1;m=d['metadata']
            v.require(pending is None and d['kind']=='node' and d['call']==count and
                      _support_cost(m['source_indices'])<=3 and len(m['source_indices'])==len(set(m['source_indices'])) and
                      m['method']==search.sparse.VERSION,'invalid sparse support call')
            pending=count
        else:
            v.require(row['event']=='call_completed' and d['kind']=='node' and d['call']==pending,'unsettled optimizer ledger')
            pending=None
    v.require(pending is None and expected==dict(node=count,leaf=0,proof=0) and count<=12544,'actual optimization counts differ')
    return count


def audit(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use numerical audit already exists')
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_twelve_settled'] and len(summary['units'])==12,'all twelve jobs must first settle')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan']);collection=v.bound(prior['collection'])
    rows=[];refs=[planref,summaryref,plan['actor']]
    for ref in summary['units']:
        r=v.bound(ref);spec=r['spec'];v.require(r['plan']==planref and spec in plan['jobs'],'unknown job binding')
        if r['status']!='completed':
            rows.append(dict(job=spec['id'],status=r['status'],integrity_passed=False,reason=r.get('error')));continue
        sr=v.bound(spec['source_result']);source=Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
        program=Program.from_dict(v.bound(r['program']),source.actor_role_sha256)
        report=v.bound(r['fit_report']);base,options=search.sparse.collapse_protected(source)
        v.require(base.to_dict()==report['collapsed_base'] and {str(k):t for k,t in options.items()}==report['original_subtrees'],
                  'protected collapse or original retained options differ')
        cases,a,arrayrefs=sampled_fit(dp,collection,spec['fold'],spec['role']);parts=search.old.lane_parts(a,cases)
        pred={s:base.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
        signals={s:search.old.frame_signals(base,parts[s],p) for s,p in pred.items()}
        subsets=search.old.risk_subsets(parts['eligibility']['rows']);groups={};local={}
        v.require(len(report['proposal_records'])==len(base.leaves),'missing confidence regions')
        for leaf,record in enumerate(report['proposal_records']):
            v.require(record['source_leaf']==leaf and record['oblique']==spec['oblique'] and
                      record['fit_frames']==int((pred['fit'][0]==leaf).sum()) and
                      record['original_option_sha256']==v.digest(options[leaf]),'confidence provenance differs')
            trees={v.digest(t):t for t in [*record['proposals'],options[leaf]]}
            g,d=search.old.local_table(base,leaf,[trees[k] for k in sorted(trees)],parts,pred,signals,subsets)
            groups[f'{leaf:04d}']=g;local[f'{leaf:04d}']=d
        v.require(groups==report['groups'] and local==report['local'],'fixed option metrics differ')
        n=len(parts['validation']['x']);best=None
        for index,weight in enumerate((0.,.5,1.,2.)):
            weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(
                o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
            allocation=allocate(weighted,256)
            p,_=search.old.assemble(base,{int(g):local[g][i]['structure'] for g,i in allocation['selected'].items()})
            search.old.qualify_current(p,parts);metrics=search.old.measure(p,parts['validation'])
            counts={k:sum(local[g][i]['totals'][k] for g,i in allocation['selected'].items()) for k in ('E','F','expected_F')}
            reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
            score=(int(reliable),*search.old.score_metrics(metrics,metrics['explanation']['whole_tree_kl'],len(p.leaves)))
            expected=dict(weight=weight,allocation=allocation,reliable_counts=counts,
                          reliable_selection_passed=reliable,metrics=metrics,score=list(score))
            v.require(expected==report['candidates'][index],'finite allocation or candidate score differs')
            if best is None or score>best[0]:best=(score,index,p)
        v.require(best[1]==report['selected'] and v.digest(best[2].structure)==v.digest(program.structure) and
                  all(search.sparse.payload(a)==search.sparse.payload(b) for a,b in zip(best[2].leaves,program.leaves,strict=True)),
                  'final program differs from the fit-only selection')
        outer=data.load_parts(dp,spec['fold'],spec['role']);independent=audited_program(program,outer,r['metrics'])
        for split,a in outer.items():
            before=source.distributions(a['x'],a['m'])
            v.require(all(np.array_equal(x,y) for x,y in zip(before[1:],independent[split][1:])), 'frozen q changed')
        calls=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'])
        rows.append(dict(job=spec['id'],status='completed',integrity_passed=True,node_fits=calls,
            fixed_option_tables_replayed=True,all_four_allocations_replayed=True,fit_only_selection_replayed=True,
            independent_program_frames=sum(len(a['x']) for a in outer.values()),independent_leaf_certificates=len(program.leaves),
            strict_numeric_passed=r['strict_numeric_passed'],reliable_fit_only_passed=r['reliable_fit_only_passed']))
        refs.extend([ref,r['program'],r['fit_report'],spec['source_result'],sr['program'],*arrayrefs])
        print(spec['id'],'read-only replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    for ref in refs:v.require(v.pointer(ref['path'])==ref,'audited immutable input changed')
    result=dict(schema='warehouse-v632-readonly-replay.v1',source=v.pointer(__file__),plan=planref,summary=summaryref,
        units=rows,integrity_passed=all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,new_environment_resets=0,
        model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444)
    return v.pointer(folder/'independent-replay.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(audit(parser.parse_args().run.resolve())),flush=True)
