#!/usr/bin/env python3
"""Read-only fitting-history, two-stage option, proof and fidelity replay.

Optimizer candidates are source/ledger bound, not refitted. Intermediate
optimizer parameters are not all retained; that scope is explicitly limited.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v640_alternation as method,warehouse_v639_joint_precision as search,warehouse_v635_program as language
from backend.training.warehouse_v626_certificate import verify as verify_leaf
from backend.training.warehouse_v626_program import evaluate_leaf
from backend.training.warehouse_v626_search import regions
from scripts.audit_warehouse_v635 import ledger,option_context,physical_bounds
from scripts.audit_warehouse_v626 import independent_dual
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def fitting_history(source,outer,lanes,candidate):
    core=set(language.protection(source)['protected_descendants']);legal=source.certificates()
    train,selection=method.fitting.internal_split(outer['fit']['rows']);route=source.leaf_indices(outer['fit']['x'])
    _,paths=search.encode(search.decode(source.structure));risk=search.choices.metrics.risk_subsets(lanes['eligibility']['rows']);trials=0;fitted=0;frozen_count=0;dense_calls=0
    v.require(candidate['source_structure_sha256']==v.digest(source.structure) and candidate['original_core']==sorted(core) and
        len(candidate['records'])==len(source.leaves),'conditional fitting source differs')
    for i,record in enumerate(candidate['records']):
        parts,masks=search.choices.root_parts(source,paths[i],lanes);repeats=[m[masks['eligibility']] for m in risk]
        score=search.choices.option_score(method.fitting.leaf_option(source,i,source.leaves[i]),parts,repeats)
        frozen=i in core or score['children'][0]['reliable_selected']['explanation']
        v.require(record['leaf']==i and record['old_score']==score and record['original_core']==(i in core) and record['frozen_reliable_E_or_core']==frozen,'conditional fitting/frozen gate differs')
        take=route==i
        if frozen:
            frozen_count+=1;v.require(record['status']=='exact_payload_preserved' and 'search'not in record and 'candidate_model'not in record,'frozen leaf fitted')
        elif 'search'in record:
            h=record['search'];v.require(h['fit_frames']==int((take&train).sum()) and h['selection_frames']==int((take&selection).sum()) and
                len(h['trials'])<=24,'conditional fit exposure or trial bound differs')
            seen=set();key=h['incumbent_key'];dense_calls+=bool(h['dense_proposal']['called'])
            for t in h['trials']:
                ident=(tuple(t['support']),t['witness']);v.require(ident not in seen and t['support']==sorted(set(t['support'])) and
                    method.fitting._support_cost(t['support'])<=3 and t['witness'] in legal[i],'invalid conditional support or witness')
                seen.add(ident);trials+=1
                if t['certified']:
                    accept=t['proxy']>key;v.require(t['accepted']==accept,'support comparison differs')
                    if accept:key=t['proxy']
                else:v.require(t['proxy'] is None,'uncertified support scored')
            v.require(key==h['selected_key'],'selected support history differs')
            if 'candidate_model'in record:
                model=record['candidate_model'];path=search.region(source.structure,i,256);fitted+=1
                v.require(not h['selected_original'] and model['support']==[-j for j in key[-2]] and model['witness']==-key[-1] and
                    record['retained_even_if_whole_leaf_zero_score'],'candidate support lost before joint evaluation')
                verify_leaf(model,path,legal[i],model['certificate']);independent_dual(model,path,legal[i])
                observed=search.choices.option_score(method.fitting.leaf_option(source,i,model),parts,repeats)
                v.require(observed==record['candidate_score'],'fitted whole-leaf score differs')
            else:v.require(h['selected_original'],'changed selected model not retained')
        else:v.require(record['status']=='no_internal_fit_exposure_keep_old' and
            (not np.any(take&train) or not np.any(take&selection)),'missing conditional fitting without exposure reason')
    return dict(conditional_support_trials=trials,conditional_final_models=fitted,frozen_exact_leaves=frozen_count,dense_proposal_calls=dense_calls)


def replay_stage(source,lanes,report,candidate,parent=None):
    base=language.import_previous(source);_,paths=search.encode(search.decode(base.structure));risk=search.choices.metrics.risk_subsets(lanes['eligibility']['rows'])
    groups={};tables={};checked=0;physical=0;precision_nodes=0
    v.require(len(report['records'])==len(paths),'stage source leaf count differs')
    for i,record in enumerate(report['records']):
        key=f'l{i:04d}';root=paths[i];obj=report['option_objects'][key];cr=candidate['records'][i]
        v.require(record['leaf']==i and obj['root']==list(root),'stage root differs')
        parts,masks=search.choices.root_parts(base,root,lanes);repeats=[m[masks['eligibility']] for m in risk]
        old=search.build.original_option(base,root);oldscore=search.choices.option_score(old,parts,repeats);frozen=cr['frozen_reliable_E_or_core']
        v.require(cr['old_score']==oldscore and record['frozen_original_core_or_reliable_E']==frozen,'stage old score/frozen set differs')
        allowed={'old':source.leaves[i]}
        if 'candidate_model'in cr:allowed['refit']=cr['candidate_model']
        proposals={p['output']:p for p in record['proposals']}
        if frozen:v.require(set(obj['options'])=={'old'} and not proposals,'frozen stage has new options')
        else:v.require(set(proposals)==set(allowed),'stage output model missing')
        if parent is not None:
            prior=parent['option_objects'][key]
            v.require(prior['root']==obj['root'] and all(name in obj['options'] and obj['options'][name]==option for name,option in prior['options'].items()),'union lost entropy option')
            prior_proposals={p['output']:p for p in parent['records'][i]['proposals']}
        take=source.leaf_indices(lanes['fit']['x'])==i;x=lanes['fit']['x'][take];m=lanes['fit']['m'][take];target=lanes['fit']['p'][take]
        for label,p in proposals.items():
            prop=p['proposal'];v.require(p['model_sha256']==v.digest(language.payload(allowed[label])) and prop['source_leaf']==i and prop['fit_frames']==len(x),'proposal target or exposure differs')
            if parent is not None:
                templates=prop['templates'];v.require(len(templates)==3 and templates[0]==prior_proposals[label]['proposal'] and
                    search.merge_proposals(templates[0],templates[1:])==prop,'union templates differ')
                _,q=evaluate_leaf(allowed[label],x,m);y=(q.argmax(1)==target.argmax(1)).astype(float)
                for template in templates[1:]:
                    maximal=max(template['proposals'],key=lambda t:t['leaf_count']);nodes=[r for r in regions(search.decode(maximal),x) if r[0] is not None]
                    v.require(len(nodes)==len(template['records']),'precision node inventory differs')
                    for (node,path,rows),e in zip(nodes,template['records'],strict=True):
                        go=search.build.split_scores(x[rows],np.asarray(node.weight,np.float32)[None],np.array([node.bias],np.float32))[:,0]>search.build.HARD_SPLIT_TIE_EPSILON
                        mass,entropy=search.precision.score_split(y[rows],go,template['precision_target'])
                        v.require(e['route']==list(path) and e['frames']==len(rows) and e['fit_high_precision_mass']==-mass and
                            e['weighted_entropy']==entropy,'precision training score differs');precision_nodes+=1
        groups[key]=[];tables[key]={}
        # Entropy-stage records predate canonical union ordering. Reproduce
        # their registered list order, never JSON key iteration order.
        order=[o['id'] for o in report['groups'][key]]
        v.require(len(order)==len(set(order)) and set(order)==set(obj['options']),'stage group/option inventory differs')
        for name in order:
            option=obj['options'][name]
            if name=='old':v.require(option==old,'old unsplit option differs')
            else:
                label=name.split('-',1)[0];v.require(label in proposals and name==label+'-'+v.digest(option['structure']) and
                    option['structure'] in proposals[label]['proposal']['proposals'],'unregistered fixed confidence tree')
                v.require(all(language.payload(model)==language.payload(allowed[label]) for model in option['models']),'stage changed fixed leaf output')
                physical+=option_context(base,root,option);checked+=1
            score=search.choices.option_score(option,parts,repeats)
            if parent is not None and name in prior['options']:v.require(score==parent['tables'][key][name],'inherited entropy score differs')
            groups[key].append(dict(id=name,leaves=score['leaves'],score=score['score']));tables[key][name]=score
    v.require(groups==report['groups'] and tables==report['tables'],'complete stage option tables differ')
    n=len(lanes['validation']['x']);best=None
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=search.allocate(weighted,256);selected={tuple(report['option_objects'][g]['root']):report['option_objects'][g]['options'][name] for g,name in allocation['selected'].items()}
        program=search.build.assemble(base,selected);language.qualify(program,lanes);observed=language.measure(program,lanes['validation'])
        counts={k:sum(tables[g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*search.choices.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(program.leaves)))
        expected=dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=observed,score=list(score))
        v.require(expected==report['candidates'][index],'stage finite allocation or gate differs')
        if best is None or score>best[0]:best=(score,index,program)
    v.require(best[1]==report['selected'] and report['reliable_selection_passed']==report['candidates'][best[1]]['reliable_selection_passed'],'stage selected method differs')
    return best[2],dict(independent_fixed_options=checked,independent_physical_bounds=physical,precision_training_nodes=precision_nodes)


def phase_counts(path,report,source_leaves,expected):
    counts={k:0 for k in expected};phases=[]
    for line in path.read_text().splitlines():
        row=json.loads(line);d=row['data']
        if row['event']=='call_started':counts[d['kind']]+=1
        if row['event']=='progress' and d.get('completed_leaves')==source_leaves:
            phases.append((d['stage'],dict(counts)))
    v.require(len(phases)==3 and phases[0]==('conditional_full_distribution_fit',report['fit_phase_counts']) and
        phases[1]==('joint_leaf_options',report['entropy_phase_cumulative_counts']) and phases[2]==('joint_leaf_options',expected),'phase boundary debit differs')
    v.require(report['coefficient_fit_calls']==expected['leaf'] and report['proof_optimizer_calls']==expected['proof'] and
        report['dense_proposal_optimizer_calls']==phases[0][1]['node'] and
        report['entropy_direction_optimizer_calls']==phases[1][1]['node']-phases[0][1]['node'] and
        report['precision_direction_optimizer_calls']==expected['node']-phases[1][1]['node'],'reported phase totals differ')


def unit(folder,ref):
    folder=Path(folder);planref=v.pointer(folder/'plan.json');plan=v.bound(planref);r=v.bound(ref);spec=r['spec']
    v.require(r['status']=='completed' and r['plan']==planref and spec in plan['jobs'],'incomplete/unbound alternating unit')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan']);outer=data.load_parts(dp,spec['fold'],spec['role'])
    sr=v.bound(spec['source_result']);source=language.Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']]);report=v.bound(r['search_report'])
    final=language.Program.from_dict(v.bound(r['program']),source.target_actor_role_sha256);cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role'])
    lanes=method.choices.metrics.lane_parts(a,cases);fit=fitting_history(source,outer,lanes,report['candidate_report'])
    _,entropy=replay_stage(source,lanes,report['entropy_report'],report['candidate_report'])
    chosen,union=replay_stage(source,lanes,report,report['candidate_report'],report['entropy_report'])
    v.require(v.digest(chosen.structure)==v.digest(final.structure) and all(language.payload(a)==language.payload(b) for a,b in zip(chosen.leaves,final.leaves,strict=True)),'final alternating choice differs')
    audited_program(final,outer,r['metrics'])
    for split,a in outer.items():v.require(language.measure(source,a)==r['baseline_metrics'][split]==sr['metrics'][split],'uniform v639 baseline differs')
    a=outer['validation']
    for task in (1,2,3):
        part=method.choices.subset(a,np.array([row['task']==task for row in a['rows']],bool))
        expected=dict(baseline=language.measure(source,part),candidate=language.measure(final,part),old_scenarios=len({row['base_seed'] for row in part['rows']}))
        v.require(expected==r['task_metrics'][str(task)],'fixed task slice differs')
    calls=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'],spec['limits']);phase_counts(Path(ref['path']).with_name('ledger.jsonl'),report,len(source.leaves),r['counts'])
    v.require(fit['dense_proposal_calls']==report['dense_proposal_optimizer_calls'] and r['numerical_passed']==strict(r['metrics']['validation']) and
        r['development_passed']==(r['numerical_passed'] and report['reliable_selection_passed']),'actual fitting count or gate differs')
    return dict(job=spec['id'],input=ref,integrity_passed=True,fitting=fit,entropy_stage=entropy,union_stage=union,solver_counts=calls,
        independent_program_frames=sum(len(a['x']) for a in outer.values()),independent_final_leaf_certificates=len(final.leaves),
        numerical_passed=r['numerical_passed'],development_passed=r['development_passed'],
        optimizer_trials_refitted=False,intermediate_optimizer_parameters_independently_replayed=False)


def main(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use replay already exists')
    pref=v.pointer(folder/'plan.json');plan=v.bound(pref);sref=v.pointer(folder/'result.json');summary=v.bound(sref)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need six actual completed alternating programs')
    ap=dict(source=v.pointer(__file__),plan=pref,summary=sref,maximum_workers=3,optimizer_refits=0)
    write_new(folder/'audit-plan.json',ap);(folder/'audit-plan.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(unit,str(folder),ref) for ref in summary['units']]):
            row=f.result();rows.append(row);print(row['job'],'independent alternating replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'scientific source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==ap['source'],'NN or replay source changed')
    result=dict(schema='warehouse-v640-independent-alternating-replay.v1',audit_plan=v.pointer(folder/'audit-plan.json'),units=sorted(rows,key=lambda r:r['job']),
        integrity_passed=len(rows)==6 and all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444);print(json.dumps(v.pointer(folder/'independent-replay.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
