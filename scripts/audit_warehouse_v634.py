#!/usr/bin/env python3
"""Parallel read-only replay of all templates, fixed tables and final outputs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v634_search as search
from backend.training.warehouse_v626_search import regions,decode
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v630_partition_proposals import allocate
from backend.training.warehouse_v628_program import payload,measure
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def ledger(path,expected,selective):
    tail='0'*64;pending=None;count=0;new_targets={p:0 for p in search.precision.PRECISIONS};old_calls=0
    for i,line in enumerate(path.read_text().splitlines(),1):
        row=json.loads(line);sha=row.pop('record_sha256')
        v.require(row['sequence']==i and row['previous_sha256']==tail and v.digest(row)==sha,'ledger chain differs')
        tail=sha;d=row['data']
        if row['event']=='call_started':
            count+=1;m=d['metadata'];v.require(pending is None and d['kind']=='node' and d['call']==count and
                m['method']==search.legacy.VERSION and search.precision._support_cost(m['source_indices'])<=3,'invalid classifier call')
            if 'precision_target' in m:
                v.require(selective and m['proposal_objective']==search.precision.VERSION and m['precision_target'] in new_targets,'unregistered precision fit')
                new_targets[m['precision_target']]+=1
            else:old_calls+=1
            pending=count
        else:
            v.require(row['event']=='call_completed' and d['kind']=='node' and d['call']==pending,'unsettled call');pending=None
    v.require(pending is None and expected==dict(node=count,leaf=0,proof=0) and count<=37632,'call count differs')
    return dict(total=count,entropy_kernel_calls=old_calls,precision_kernel_calls={str(k):n for k,n in new_targets.items()})


def training_records(base,arrays,leaf,template):
    take=base.leaf_indices(arrays['x'])==leaf;x=arrays['x'][take];m=arrays['m'][take];p=arrays['p'][take]
    v.require(template['source_leaf']==leaf and template['fit_frames']==len(x),'template exposure differs')
    _,_,q=base.distributions(x,m);y=(q.argmax(1)==p.argmax(1)).astype(float)
    full=max(template['proposals'],key=lambda t:t['leaf_count'])
    rows=[r for r in regions(decode(full),x) if r[0] is not None]
    records=template['records'];v.require(len(rows)==len(records),'template fit node inventory differs')
    for (node,path,indices),record in zip(rows,records,strict=True):
        v.require(list(path)==record['route'] and len(indices)==record['frames'] and
                  int(np.count_nonzero(node.weight))==record['source_features'] and record['optimizer_supports']<=7,
                  'template node provenance differs')
        go=search.precision.split_scores(x[indices],np.asarray(node.weight,np.float32)[None],np.array([node.bias],np.float32))[:,0]>search.precision.HARD_SPLIT_TIE_EPSILON
        v.require(min(go.sum(),(~go).sum())>=40,'template child support below40')
        if template['version']==search.precision.VERSION:
            target=template['precision_target'];score=search.precision.score_split(y[indices],go,target)
            v.require(record['fit_high_precision_mass']==-score[0] and record['weighted_entropy']==score[1] and
                      record['precision_target']==target,'actual hard-route precision mass differs')
        else:
            w=np.where(y[indices]==0,4.,1.)
            entropy=sum(float(search.legacy._entropy(np.sum(w[t]*y[indices][t]),np.sum(w[t]))) for t in (go,~go))
            v.require(record['weighted_entropy']==entropy,'entropy template hard-route score differs')


def unit(folder,ref):
    folder=Path(folder);planref=v.pointer(folder/'plan.json');plan=v.bound(planref);r=v.bound(ref);spec=r['spec']
    v.require(r['status']=='completed' and r['plan']==planref and spec in plan['jobs'],'incomplete or unbound unit')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    sr=v.bound(spec['source_result']);source=Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
    p=Program.from_dict(v.bound(r['program']),source.actor_role_sha256);report=v.bound(r['fit_report'])
    base,originals=search.legacy.collapse_protected(source)
    v.require(base.to_dict()==report['collapsed_base'] and {str(k):v for k,v in originals.items()}==report['original_subtrees'], 'collapsed source differs')
    cases,a,refs=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);parts=search.old.lane_parts(a,cases)
    pred={s:base.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
    signals={s:search.old.frame_signals(base,parts[s],x) for s,x in pred.items()};subsets=search.old.risk_subsets(parts['eligibility']['rows'])
    groups={};local={};v.require(len(report['proposal_records'])==len(base.leaves),'missing template groups')
    template_nodes=0
    for i,record in enumerate(report['proposal_records']):
        v.require(record['source_leaf']==i and record['original_option_sha256']==v.digest(originals[i]),'template group identity differs')
        templates=record['templates'];v.require(len(templates)==(3 if spec['selective'] else 1),'wrong template count')
        v.require(templates[0]['version']==search.legacy.VERSION,'entropy control missing')
        if spec['selective']:
            v.require([t['precision_target'] for t in templates[1:]]==list(search.precision.PRECISIONS) and
                      all(t['version']==search.precision.VERSION for t in templates[1:]),'precision targets differ')
        for template in templates:
            training_records(base,parts['fit'],i,template);template_nodes+=len(template['records'])
        options={v.digest(t):t for t in [originals[i],*(t for template in templates for t in template['proposals'])]}
        g,d=search.old.local_table(base,i,[options[k] for k in sorted(options)],parts,pred,signals,subsets)
        groups[f'{i:04d}']=g;local[f'{i:04d}']=d
    v.require(groups==report['groups'] and local==report['local'],'frozen tables differ')
    n=len(parts['validation']['x']);best=None
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256);q,_=search.old.assemble(base,{int(g):local[g][i]['structure'] for g,i in allocation['selected'].items()})
        search.old.qualify_current(q,parts);metrics=measure(q,parts['validation'])
        counts={k:sum(local[g][i]['totals'][k] for g,i in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*search.old.score_metrics(metrics,metrics['explanation']['whole_tree_kl'],len(q.leaves)))
        expected=dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=metrics,score=list(score))
        v.require(expected==report['candidates'][index],'candidate allocation or score differs')
        if best is None or score>best[0]:best=(score,index,q)
    v.require(best[1]==report['selected'] and v.digest(best[2].structure)==v.digest(p.structure) and
              all(payload(a)==payload(b) for a,b in zip(best[2].leaves,p.leaves,strict=True)),'final selection differs')
    outer=data.load_parts(dp,spec['fold'],spec['role']);independent=audited_program(p,outer,r['metrics'])
    for split,a in outer.items():
        reference=source.distributions(a['x'],a['m']);v.require(all(np.array_equal(x,y) for x,y in zip(reference[1:],independent[split][1:])),'full q changed')
        v.require(measure(source,a)==r['baseline_metrics'][split]==sr['metrics'][split],'baseline metrics differ')
    calls=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'],spec['selective'])
    v.require(r['numerical_passed']==strict(r['metrics']['validation']) and r['development_passed']==(r['numerical_passed'] and report['reliable_selection_passed']),'gate differs')
    return dict(job=spec['id'],input=ref,integrity_passed=True,template_node_training_scores_replayed=template_nodes,
        complete_fixed_tables_and_four_allocations_replayed=True,actual_solver_calls=calls,
        independent_program_frames=sum(len(a['x']) for a in outer.values()),independent_leaf_certificates=len(p.leaves),
        numerical_passed=r['numerical_passed'],development_passed=r['development_passed'])


def main(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use audit already exists')
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_twelve_settled'] and all(v.bound(r)['status']=='completed' for r in summary['units']),'need twelve actual complete units')
    auditplan=dict(source=v.pointer(__file__),plan=planref,summary=summaryref,maximum_workers=3,optimizer_candidates_refitted=False)
    write_new(folder/'audit-plan.json',auditplan);(folder/'audit-plan.json').chmod(0o444)
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(unit,str(folder),r) for r in summary['units']]):
            r=f.result();rows.append(r);print(r['job'],'independent replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==auditplan['source'],'actor or audit source changed')
    result=dict(schema='warehouse-v634-independent-replay.v1',audit_plan=v.pointer(folder/'audit-plan.json'),units=sorted(rows,key=lambda r:r['job']),
        integrity_passed=len(rows)==12 and all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,new_environment_resets=0,
        model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444)
    print(json.dumps(v.pointer(folder/'independent-replay.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
