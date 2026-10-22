#!/usr/bin/env python3
"""Read-only replay of learned-root proofs, finite options and final choice."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction as F
from itertools import product
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v635_program as language,warehouse_v635_search as search,warehouse_v635_subtrees as build
from backend.training.warehouse_v630_program import Program as Previous
from backend.training.warehouse_v626_search import decode,encode,replace,regions
from backend.training.warehouse_v628_certificate import region
from backend.training.warehouse_v626_certificate import verify as verify_leaf
from backend.training.warehouse_v630_partition_proposals import allocate
from scripts.diagnose_warehouse_v634_free_roots import free_roots
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.audit_warehouse_v630 import audited_program
from scripts.audit_warehouse_v626 import independent_dual
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def physical_bounds(structure):
    checked=0;domain=build.domain_table()['features'];tree,_=build.structure_info(structure,256)
    for weights,bias in zip(tree['split_weights'],tree['split_bias'],strict=True):
        proof=build.guards.evidence(weights,bias)
        if proof['kind']!=build.guards.KIND:continue
        ids=proof['context_indices'];bit=build.guards.MASK_INDICES[proof['action']]
        exact=max(F(float(bias))+sum(F(float(weights[i]))*F(value) for i,value in zip(ids,corner,strict=True))
                  for corner in product(*[(domain[i]['lower'],domain[i]['upper']) for i in ids]))
        operations=4*len(domain)+16;u=F(1,2**52)
        magnitude=abs(F(float(bias)))+abs(F(float(weights[bit])))+sum(max(abs(F(float(weights[i]))*F(domain[i]['lower'])),abs(F(float(weights[i]))*F(domain[i]['upper']))) for i in ids)
        error=operations*u/(1-operations*u)*magnitude
        v.require([exact.numerator,exact.denominator]==proof['zero_mask_upper'] and [error.numerator,error.denominator]==proof['fp64_error_bound'] and
                  exact+error<F(build.HARD_SPLIT_TIE_EPSILON),'independent zero-mask physical proof failed')
        checked+=1
    return checked


def ledger(path,expected,limits,spent=None):
    rows=[json.loads(line) for line in path.read_text().splitlines()];tail='0'*64;actual={k:0 for k in limits};pending=None;new={k:0 for k in limits}
    for i,row in enumerate(rows,1):
        sha=row.pop('record_sha256');v.require(row['sequence']==i and row['previous_sha256']==tail and v.digest(row)==sha,'ledger chain differs')
        tail=sha;d=row['data']
        if row['event']=='engineering_prefix_debit':
            v.require(i==1,'prefix debit not first')
            if spent is not None:
                v.require(d==spent,'engineering debit differs');parent=v.bound(spent['source'])
                v.require(parent['counts']==spent['counts'] and parent['elapsed_seconds']==spent['seconds'],'parent spent counts differ')
                ledger(Path(spent['ledger']['path']),parent['counts'],limits)
            else:v.require(d['counts']==actual and d['seconds']==0 and d['source'] is None,'unexpected earlier debit')
            actual=dict(d['counts'])
        elif row['event']=='call_started':
            kind=d['kind'];actual[kind]+=1;new[kind]+=1
            v.require(pending is None and d['call']==actual[kind] and actual[kind]<=limits[kind],'call sequence or limit differs');pending=(kind,d['call'])
        elif row['event']=='call_completed':v.require(pending==(d['kind'],d['call']),'incomplete call');pending=None
        else:v.require(row['event']=='progress','unrecognized ledger event')
    v.require(pending is None and actual==expected,'cumulative actual call count differs')
    return dict(cumulative=actual,new_calls=new)


def option_context(source,root,option):
    tree=replace(decode(source.structure),root,decode(option['structure']));structure,paths=encode(tree);structure=build.plain(structure)
    legal=build.guards.leaf_actions(structure,256);ids=[i for i,path in enumerate(paths) if path[:len(root)]==root]
    v.require(len(ids)==len(option['models'])==len(option['paths']),'option leaf inventory differs')
    proofs=physical_bounds(structure)
    for j,i in enumerate(ids):
        expected=region(structure,i,256);given=option['paths'][j]
        v.require(given['path']==expected and given['legal']==sorted(legal[i]),'option full-path context or logical fact differs')
        verify_leaf(option['models'][j],expected,legal[i],option['models'][j]['certificate']);independent_dual(option['models'][j],expected,legal[i])
    return proofs


def unit(folder,ref):
    folder=Path(folder);planref=v.pointer(folder/'plan.json');plan=v.bound(planref);r=v.bound(ref);spec=r['spec']
    v.require(r['status']=='completed' and r['plan']==planref and spec in plan['jobs'],'incomplete or unbound unit')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan']);outer=data.load_parts(dp,spec['fold'],spec['role'])
    sr=v.bound(spec['source_result']);previous=Previous.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
    source=language.import_previous(previous);report=v.bound(r['search_report']);final=language.Program.from_dict(v.bound(r['program']),source.target_actor_role_sha256)
    roots,paths=free_roots(previous);v.require([list(x) for x in roots]==spec['structural_roots'],'root list differs from structural inventory')
    cases,a,refs=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=search.metrics.lane_parts(a,cases)
    masks=search.metrics.risk_subsets(lanes['eligibility']['rows']);train,_=build.internal_split(outer['fit']['rows']);groups={};tables={}
    physical_checks=0;template_nodes=0
    for i in language.protection(source)['protected_descendants']:
        path=paths[i];parts,take=search.root_parts(source,path,lanes);option=build.original_option(source,path)
        score=search.option_score(option,parts,[mask[take['eligibility']] for mask in masks]);key=f'p{i:04d}'
        groups[key]=[dict(id='fixed',leaves=1,score=score['score'])];tables[key]={'fixed':score}
    for index,root in enumerate(roots):
        key=f'r{index:04d}';obj=report['option_objects'][key];v.require(obj['root']==list(root),'option root differs')
        parts,take=search.root_parts(source,root,lanes);groups[key]=[];tables[key]={}
        record=report['root_records'][index];trparts,trmask=search.root_parts(source,root,{'fit':outer['fit']})
        indices=np.flatnonzero(trmask['fit']&train);x=outer['fit']['x'][indices];nn=outer['fit']['p'][indices].argmax(1)
        v.require(record['root']==list(root) and record['root_training_frames']==len(indices),'root training exposure differs')
        for name,option in obj['options'].items():
            if name=='old':v.require(option==build.original_option(source,root),'original subtree option changed')
            else:
                physical_checks+=option_context(source,root,option)
                template=next(t for t in record['templates'] if t.get('option_id')==name)
                nodes=[row for row in regions(decode(option['structure']),x) if row[0] is not None]
                v.require(len(nodes)==len(template['records']),'learned template node count differs')
                for (node,path,rows),evidence in zip(nodes,template['records'],strict=True):
                    go=build.split_scores(x[rows],np.asarray(node.weight,np.float32)[None],np.array([node.bias],np.float32))[:,0]>build.HARD_SPLIT_TIE_EPSILON
                    v.require(evidence['route']==list(path) and evidence['frames']==len(rows) and evidence['right_frames']==int(go.sum()) and
                        evidence['right_NN_agreement']==float((nn[rows[go]]==evidence['action']).mean()) and
                        evidence['right_NN_agreement']>=evidence['precision_target'],'NN action-condition training score differs')
                    template_nodes+=1
            score=search.option_score(option,parts,[mask[take['eligibility']] for mask in masks])
            groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
    v.require(groups==report['groups'] and tables==report['tables'],'fixed root tables differ')
    n=len(lanes['validation']['x']);best=None
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256);chosen={tuple(report['option_objects'][g]['root']):report['option_objects'][g]['options'][name]
            for g,name in allocation['selected'].items() if g in report['option_objects']}
        program=build.assemble(source,chosen);language.qualify(program,lanes);observed=language.measure(program,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*search.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(program.leaves)))
        expected=dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=observed,score=list(score))
        v.require(expected==report['candidates'][index] and len(program.leaves)==allocation['leaves'],'allocation or score differs')
        if best is None or score>best[0]:best=(score,index,program)
    v.require(best[1]==report['selected'] and v.digest(best[2].structure)==v.digest(final.structure) and
        all(language.payload(a)==language.payload(b) for a,b in zip(best[2].leaves,final.leaves,strict=True)),'final fit-only choice differs')
    audited_program(final,outer,r['metrics']);physical_checks+=physical_bounds(final.structure)
    for split,a in outer.items():v.require(language.measure(previous,a)==r['baseline_metrics'][split]==sr['metrics'][split],'source baseline differs')
    calls=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'],spec['limits'],spec.get('engineering_spent'))
    v.require(r['numerical_passed']==strict(r['metrics']['validation']) and r['development_passed']==(r['numerical_passed'] and report['reliable_selection_passed']),'gate differs')
    return dict(job=spec['id'],input=ref,integrity_passed=True,independent_zero_mask_bounds=physical_checks,training_action_nodes=template_nodes,
        full_option_tables_and_all_allocations_replayed=True,solver_counts=calls,independent_program_frames=sum(len(a['x']) for a in outer.values()),
        independent_leaf_certificates=len(final.leaves),numerical_passed=r['numerical_passed'],development_passed=r['development_passed'])


def main(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use replay already exists')
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need six actual completed programs')
    auditplan=dict(source=v.pointer(__file__),plan=planref,summary=summaryref,maximum_workers=3,optimizer_candidates_refitted=False)
    write_new(folder/'audit-plan-02.json',auditplan);(folder/'audit-plan-02.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(unit,str(folder),ref) for ref in summary['units']]):
            r=f.result();rows.append(r);print(r['job'],'independent replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==auditplan['source'],'NN or replay source changed')
    result=dict(schema='warehouse-v635-independent-replay.v1',audit_plan=v.pointer(folder/'audit-plan-02.json'),units=sorted(rows,key=lambda r:r['job']),
        integrity_passed=len(rows)==6 and all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,new_environment_resets=0,
        model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444)
    print(json.dumps(v.pointer(folder/'independent-replay.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
