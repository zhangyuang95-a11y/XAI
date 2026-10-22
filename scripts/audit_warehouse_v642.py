#!/usr/bin/env python3
"""Read-only replay of paired depth contracts, options, proofs and allocation.

Stored context predicates are checked against their recorded public directions
and training entropy. This does not claim independent global tree optimality
or refit unretained optimizer trajectories.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v642_search as search,warehouse_v642_program as language,warehouse_v635_program as previous
from backend.training.warehouse_v626_search import decode,encode,regions,replace
from backend.training.warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from scripts.audit_warehouse_v642_program import audited_program
from scripts.audit_warehouse_v635 import ledger
from scripts.audit_warehouse_v626 import independent_dual
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def context_evidence(training,bank,template,option,known,remaining):
    c=search.context;target=training['p'].astype(float);target/=target.sum(1,keepdims=True)
    tree=decode(option['structure']);node_map={p:(node,rows) for node,p,rows in regions(tree,training['x']) if node is not None}
    v.require(template['bank']==bank and template['training_frames']==len(target) and
        template['complete_NN_soft_targets'] and template['source_actions_never_executed'],'context target or geometry evidence differs')
    context_paths=set();guard_paths=set()
    for record in template['records']:
        path=tuple(record['route']);node,rows=node_map[path];context_paths.add(path)
        v.require(list(node.weight)==record['stored_weight'] and node.bias==record['stored_bias'] and len(rows)==record['frames'],'stored context predicate differs')
        direction=record['direction_provenance'];w=np.asarray(node.weight,np.float32)
        if direction['origin']=='public_axis':
            ids=np.flatnonzero(w);v.require(len(ids)==1 and w[ids[0]]==1 and not c.domain_table()['features'][ids[0]]['constant'],'unregistered public axis')
        else:
            expected=dict(direction);expected.pop('origin');v.require(expected in bank and list(w)==direction['weight'],'unregistered borrowed public direction')
        go=split_scores(training['x'][rows],w[None],np.array([node.bias],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
        v.require(record['left_frames']==int((~go).sum()) and record['right_frames']==int(go.sum()) and min(int(go.sum()),int((~go).sum()))>=160,'context child exposure differs')
        before=float(c.entropy(target[rows].sum(0),len(rows)))
        after=float(c.entropy(target[rows[go]].sum(0),int(go.sum()))+c.entropy(target[rows[~go]].sum(0),int((~go).sum())))
        v.require(record['entropy_before']==before and record['entropy_after']==after and before-after>1e-8,'stored training entropy differs')
    # Terminals may be leaves or the dedicated, counted legal-bit split.
    routes={p:rows for _,p,rows in regions(tree,training['x'])}
    for item in template['terminals']:
        path=tuple(item['route']);rows=routes[path];counts=np.bincount(target[rows].argmax(1),minlength=5)
        mode=int(counts.argmax()) if len(rows) else 4
        legal=set(known)
        for depth in range(len(path)):
            node,_=node_map[path[:depth]];proof=search.build.guards.evidence(node.weight,node.bias)
            legal.update(proof[('right' if path[depth] else 'left')+'_proven'])
        v.require(item['frames']==len(rows) and item['NN_argmax_counts']==counts.tolist() and item['mode']==mode and item['known_actions']==sorted(legal),'terminal NN mode or path facts differ')
        guarded=bool(len(rows) and mode not in legal and remaining-len(path)>0)
        v.require(item['physical_guard_added']==guarded,'terminal physical guard inventory differs')
        if guarded:
            guard_paths.add(path);node,_=node_map[path];w,b=search.build.guards.old.predicate([mode])
            v.require(list(node.weight)==list(w) and node.bias==b and node.left is None and node.right is None,'terminal guard changed or uncounted')
        else:v.require(path not in node_map,'unguarded terminal has unexplained split')
    v.require(context_paths.isdisjoint(guard_paths) and context_paths|guard_paths==set(node_map),'unrecorded context/guard nodes')
    attempts=template['attempts'];v.require(sum(a['found_split'] for a in attempts)==len(context_paths) and
        all(a['directions_evaluated']<=173+80 and a['threshold_evaluations']<=4*a['directions_evaluated'] for a in attempts),'node call/direction accounting differs')
    return dict(context_nodes=len(context_paths),physical_guard_nodes=len(guard_paths),node_search_calls=len(attempts))


def replay(source,outer,lanes,report,depth):
    base=language.import_previous(source,depth);fixed,roots,paths,risk=search.scan(base,lanes)
    v.require(report['maximum_depth']==depth and report['frozen_leaves']==sorted(fixed) and report['selection_frames']==len(lanes['validation']['x']) and
        [r['root'] for r in report['root_records']]==[list(p) for p in roots],'capacity, frozen set or free root scope differs')
    groups={};tables={};options={};contexts=[];proofs=0;tr,_=search.build.internal_split(outer['fit']['rows'])
    for i,item in fixed.items():
        key=f'p{i:04d}';score=item['score'];groups[key]=[dict(id='fixed',leaves=1,score=score['score'])];tables[key]={'fixed':score}
    for index,root in enumerate(roots):
        key=f'r{index:04d}';record=report['root_records'][index];obj=report['option_objects'][key]
        v.require(obj['root']==list(root) and obj['options']['old']==search.build.original_option(base,root),'old finite option changed')
        parts,masks=search.old.root_parts(base,root,lanes);_,takes=search.old.root_parts(base,root,{'fit':outer['fit']})
        idx=np.flatnonzero(takes['fit']&tr);a=outer['fit'];training={k:a[k][idx] for k in ('x','m','p')}
        prefix=obj['options']['old']['paths'][0]['path']['path'][:len(root)];known={4}
        for p in prefix:known.update(search.build.guards.evidence(p['weight'],p['bias'])[p['branch']+'_proven'])
        bank=search.context.borrowed_bank(base,root,training['x']) if len(idx) else []
        v.require(record['training_frames']==len(idx) and record['known_actions']==sorted(known) and record['bank']==bank,'source training exposure differs')
        names={'old'}
        for template,d in zip(record['templates'],(2,3),strict=True):
            v.require(template['context_depth']==d,'context recipe differs')
            if depth-len(root)<1 or len(idx)<80:
                v.require(template['skipped']=='no_remaining_depth_or_minimum80_fit_frames','skipped context differs');continue
            name=f'context-{d}';names.add(name);v.require(template['option_id']==name,'template binding differs')
            contexts.append(context_evidence(training,bank,template,obj['options'][name],known,depth-len(root)))
        v.require(set(obj['options'])==names,'missing or extra context option')
        options[key]=obj;groups[key]=[];tables[key]={}
        for name,option in sorted(obj['options'].items()):
            # Compile the exact prefix, verify every final stored leaf/path.
            full=search.build.plain(encode(replace(decode(base.structure),root,decode(option['structure'])))[0])
            _,fullpaths=encode(decode(full));legal=search.build.guards.leaf_actions(full,256,depth)
            ids=[i for i,p in enumerate(fullpaths) if p[:len(root)]==root]
            for j,i in enumerate(ids):
                path=language.region(full,i,256,depth);entry=option['paths'][j]
                v.require(entry['path']==path and entry['legal']==sorted(legal[i]),'option full path context differs')
                independent_dual(option['models'][j],path,legal[i]);proofs+=1
            score=search.old.option_score(option,parts,[m[masks['eligibility']] for m in risk]);tables[key][name]=score
            groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']))
    v.require(groups==report['groups'] and tables==report['tables'],'finite option table differs')
    n=report['selection_frames'];best=None
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in objects] for g,objects in groups.items()}
        allocation=search.allocate(weighted,256);selected={tuple(options[g]['root']):options[g]['options'][name] for g,name in allocation['selected'].items() if g in options}
        out=search.build.assemble(base,selected);language.qualify(out,lanes);observed=language.measure(out,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*search.old.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(out.leaves)))
        v.require(report['candidates'][index]==dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=observed,score=list(score)),'allocation or comparison differs')
        if best is None or score>best[0]:best=(score,out,index)
    v.require(best[2]==report['selected'] and report['reliable_selection_passed']==report['candidates'][best[2]]['reliable_selection_passed'],'chosen allocation differs')
    return best[1],dict(stored_option_leaf_proofs=proofs,context_nodes=sum(x['context_nodes'] for x in contexts),physical_guard_nodes=sum(x['physical_guard_nodes'] for x in contexts),node_search_calls=sum(x['node_search_calls'] for x in contexts))


def unit(folder,ref):
    plan=v.bound(v.pointer(Path(folder)/'plan.json'));r=v.bound(ref);spec=r['spec'];depth=spec['maximum_depth']
    v.require(r['status']=='completed' and spec in plan['jobs'],'need completed declared arm')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    source_result=v.bound(spec['source_result']);source=previous.Program.from_dict(v.bound(source_result['program']),old['actor_role_sha256'][spec['role']])
    outer=data.load_parts(dp,spec['fold'],spec['role']);cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=search.old.metrics.lane_parts(a,cases)
    report=v.bound(r['search_report']);chosen,checks=replay(source,outer,lanes,report,depth)
    final=language.Program.from_dict(v.bound(r['program']),source.target_actor_role_sha256)
    v.require(chosen.structure==final.structure and all(language.payload(a)==language.payload(b) for a,b in zip(chosen.leaves,final.leaves,strict=True)),'final structure/payload differs')
    audited_program(final,outer,r['metrics']);a=outer['validation'];take=np.isin(source.leaf_indices(a['x']),spec['frozen_leaves'])
    before=source.distributions(a['x'],a['m']);after=final.distributions(a['x'],a['m'])
    v.require(all(np.array_equal(x[take],y[take]) for x,y in zip(before[1:],after[1:],strict=True)),'frozen output changed')
    counts=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'],spec['limits'])
    v.require(counts['cumulative']['node']==checks['node_search_calls'] and strict(r['metrics']['validation'])==r['numerical_passed'] and
        r['development_passed']==(r['numerical_passed'] and report['reliable_selection_passed']),'call count or gate differs')
    return dict(job=spec['id'],input=ref,integrity_passed=True,checks=checks,counts=counts,program_frames=sum(len(a['x']) for a in outer.values()),
        leaf_certificates=len(final.leaves),optimizer_trajectories_refitted=False,global_tree_optimality_claimed=False)


def main(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use audit already exists');plan=v.bound(v.pointer(folder/'plan.json'));summary=v.bound(v.pointer(folder/'result.json'))
    v.require(summary['all_twelve_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need 12 completed arms before full replay')
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            pair=[j for j in plan['jobs'] if j['fold']==fold and j['role']==role]
            v.require(len(pair)==2 and {j['maximum_depth'] for j in pair}=={9,12} and pair[0]['limits']==pair[1]['limits'] and pair[0]['source_result']==pair[1]['source_result'],'capacity arms are not matched')
    ap=dict(source=v.pointer(__file__),plan=v.pointer(folder/'plan.json'),summary=v.pointer(folder/'result.json'),maximum_workers=3,optimizer_refits=0)
    write_new(folder/'audit-plan.json',ap);(folder/'audit-plan.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(unit,str(folder),ref) for ref in summary['units']]):
            row=future.result();rows.append(row);print(row['job'],'replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==ap['source'],'NN or audit source changed')
    result=dict(schema='warehouse-v642-depth-matched-independent-replay.v1',audit_plan=v.pointer(folder/'audit-plan.json'),units=sorted(rows,key=lambda r:r['job']),integrity_passed=len(rows)==12,
        optimizer_refits=0,NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444)
    print(json.dumps(v.pointer(folder/'independent-replay.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
