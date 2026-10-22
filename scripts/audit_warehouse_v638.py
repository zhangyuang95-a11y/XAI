#!/usr/bin/env python3
"""Independent read-only replay of all fixed distribution/partition options."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v638_joint_options as search,warehouse_v635_program as language
from scripts.audit_warehouse_v635 import ledger,option_context,physical_bounds
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def unit(folder,ref):
    folder=Path(folder);planref=v.pointer(folder/'plan.json');plan=v.bound(planref);r=v.bound(ref);spec=r['spec']
    v.require(r['status']=='completed' and r['plan']==planref and spec in plan['jobs'],'incomplete/unbound joint unit')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan']);outer=data.load_parts(dp,spec['fold'],spec['role'])
    sr=v.bound(spec['source_result']);source=language.Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
    base=language.import_previous(source);final=language.Program.from_dict(v.bound(r['program']),source.target_actor_role_sha256);report=v.bound(r['search_report'])
    candidate=v.bound(spec['candidate_result']);v.require(candidate['spec']['source_result']==spec['source_result'],'candidate/source pairing differs')
    candidate_report=v.bound(candidate['search_report']);cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role'])
    lanes=search.choices.metrics.lane_parts(a,cases);risk=search.choices.metrics.risk_subsets(lanes['eligibility']['rows'])
    _,paths=search.encode(search.decode(base.structure));groups={};tables={};checked_options=0;physical=0
    v.require(len(report['records'])==len(paths)==len(candidate_report['records']),'leaf record count differs')
    for i,record in enumerate(report['records']):
        key=f'l{i:04d}';obj=report['option_objects'][key];root=paths[i]
        v.require(record['leaf']==i and obj['root']==list(root),'source leaf root differs')
        parts,masks=search.choices.root_parts(base,root,lanes);repeats=[m[masks['eligibility']] for m in risk]
        cr=candidate_report['records'][i];oldoption=search.build.original_option(base,root);oldscore=search.choices.option_score(oldoption,parts,repeats)
        frozen=cr['frozen_reliable_E_or_core'];v.require(cr['old_score']==oldscore and record['frozen_original_core_or_reliable_E']==frozen,'old score/frozen source differs')
        allowed={'old':source.leaves[i]}
        if 'candidate_model'in cr:allowed['refit']=cr['candidate_model']
        proposals={t['output']:t for t in record['proposals']}
        if frozen:v.require(set(obj['options'])=={'old'} and not proposals,'frozen source has new options')
        else:v.require(set(proposals)==set(allowed),'missing or unregistered output proposal')
        for label,p in proposals.items():
            v.require(p['model_sha256']==v.digest(language.payload(allowed[label])) and p['proposal']['source_leaf']==i and
                p['proposal']['fit_frames']==int((source.leaf_indices(lanes['fit']['x'])==i).sum()),'proposal output or fit exposure differs')
        groups[key]=[];tables[key]={}
        for name,option in obj['options'].items():
            if name=='old':v.require(option==oldoption,'old unsplit option changed')
            else:
                label=name.split('-',1)[0];v.require(label in proposals,'unknown fixed output label')
                v.require(name==label+'-'+v.digest(option['structure']) and option['structure'] in proposals[label]['proposal']['proposals'],'unknown finite confidence tree')
                v.require(all(language.payload(m)==language.payload(allowed[label]) for m in option['models']),'partition changed its fixed complete output')
                physical+=option_context(base,root,option);checked_options+=1
            score=search.choices.option_score(option,parts,repeats);groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
    v.require(groups==report['groups'] and tables==report['tables'],'complete finite joint tables differ')
    n=len(lanes['validation']['x']);best=None
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=search.allocate(weighted,256);selected={tuple(report['option_objects'][g]['root']):report['option_objects'][g]['options'][name] for g,name in allocation['selected'].items()}
        program=search.build.assemble(base,selected);language.qualify(program,lanes);observed=language.measure(program,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*search.choices.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(program.leaves)))
        expected=dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=observed,score=list(score))
        v.require(expected==report['candidates'][index],'joint finite allocation or gate differs')
        if best is None or score>best[0]:best=(score,index,program)
    v.require(best[1]==report['selected'] and v.digest(best[2].structure)==v.digest(final.structure) and
        all(language.payload(a)==language.payload(b) for a,b in zip(best[2].leaves,final.leaves,strict=True)),'final fit-only selection differs')
    audited_program(final,outer,r['metrics']);physical+=physical_bounds(final.structure)
    for split,a in outer.items():v.require(language.measure(source,a)==r['baseline_metrics'][split]==sr['metrics'][split],'uniform baseline differs')
    a=outer['validation']
    for task in (1,2,3):
        part=search.choices.subset(a,np.array([row['task']==task for row in a['rows']],bool))
        expected=dict(baseline=language.measure(source,part),candidate=language.measure(final,part),old_scenarios=len({row['base_seed'] for row in part['rows']}))
        v.require(expected==r['task_metrics'][str(task)],'fixed task slice differs')
    calls=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'],spec['limits']);v.require(r['counts']['leaf']==r['counts']['proof']==0,'fixed input experiment unexpectedly refitted coefficients')
    v.require(r['numerical_passed']==strict(r['metrics']['validation']) and r['development_passed']==(r['numerical_passed'] and report['reliable_selection_passed']),'outer gate differs')
    return dict(job=spec['id'],input=ref,integrity_passed=True,independent_program_frames=sum(len(a['x']) for a in outer.values()),independent_final_leaf_certificates=len(final.leaves),
        independent_fixed_partition_options=checked_options,independent_physical_bounds=physical,all_finite_tables_and_allocations_replayed=True,
        predicate_optimizers_refitted=False,predicate_optimizer_trajectory_independently_replayed=False,solver_counts=calls,numerical_passed=r['numerical_passed'],development_passed=r['development_passed'])


def main(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use replay already exists')
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need six completed joint programs')
    ap=dict(source=v.pointer(__file__),plan=planref,summary=summaryref,maximum_workers=3,optimizer_candidates_refitted=False)
    write_new(folder/'audit-plan.json',ap);(folder/'audit-plan.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(unit,str(folder),ref) for ref in summary['units']]):
            r=f.result();rows.append(r);print(r['job'],'independent joint replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'scientific source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==ap['source'],'NN/replay source changed')
    result=dict(schema='warehouse-v638-independent-joint-replay.v1',audit_plan=v.pointer(folder/'audit-plan.json'),units=sorted(rows,key=lambda r:r['job']),
        integrity_passed=len(rows)==6 and all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444);print(json.dumps(v.pointer(folder/'independent-replay.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
