#!/usr/bin/env python3
"""Read-only replay of v637 fixed leaf options, final proofs and gate metrics.

Optimizer trajectories/support selection are source- and ledger-bound; they
are not refitted. Complete stored OLD/FINAL finite option tables are rescored.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v637_leaf_refresh as search,warehouse_v635_program as language
from backend.training.warehouse_v626_certificate import verify as verify_leaf
from scripts.audit_warehouse_v635 import ledger,physical_bounds
from scripts.audit_warehouse_v626 import independent_dual
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def unit(folder,ref):
    folder=Path(folder);planref=v.pointer(folder/'plan.json');plan=v.bound(planref);r=v.bound(ref);spec=r['spec']
    v.require(r['status']=='completed' and r['plan']==planref and spec in plan['jobs'],'incomplete/unbound leaf experiment')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan']);outer=data.load_parts(dp,spec['fold'],spec['role'])
    sr=v.bound(spec['source_result']);source=language.Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
    final=language.Program.from_dict(v.bound(r['program']),source.target_actor_role_sha256);report=v.bound(r['search_report'])
    v.require(v.digest(source.structure)==v.digest(final.structure)==report['source_structure_sha256'] and
        v.digest(source.protection)==v.digest(final.protection)==report['source_protection_sha256'],'topology or original core contract changed')
    core=set(language.protection(source)['protected_descendants']);v.require(report['original_core']==sorted(core),'core inventory differs')
    cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=search.choices.metrics.lane_parts(a,cases)
    repeats=search.choices.metrics.risk_subsets(lanes['eligibility']['rows']);train,selection=search.internal_split(outer['fit']['rows'])
    fit_route=source.leaf_indices(outer['fit']['x']);_,paths=search.choices.encode(search.choices.decode(source.structure));legal=source.certificates()
    v.require(len(report['records'])==len(source.leaves)==len(final.leaves),'incomplete leaf records');totals=dict(E=0,F=0,expected_F=0.);trial_count=0;frozen_count=0;new_count=0
    for i,(incumbent,record) in enumerate(zip(source.leaves,report['records'],strict=True)):
        v.require(record['leaf']==i,'leaf record order differs');parts,masks=search.choices.root_parts(source,paths[i],lanes)
        risk=[m[masks['eligibility']] for m in repeats];old_score=search.choices.option_score(search.leaf_option(source,i,incumbent),parts,risk)
        frozen=i in core or old_score['children'][0]['reliable_selected']['explanation']
        v.require(record['old_score']==old_score and record['original_core']==(i in core) and record['frozen_reliable_E_or_core']==frozen,'old finite option or frozen predicate differs')
        chosen='old';score=old_score;model=incumbent
        if frozen:
            frozen_count+=1;v.require(record['status']=='exact_payload_preserved' and 'search' not in record,'frozen leaf was fitted')
        elif 'search' in record:
            history=record['search'];take=fit_route==i
            v.require(history['fit_frames']==int((take&train).sum()) and history['selection_frames']==int((take&selection).sum()) and
                len(history['trials'])<=search.TRIALS,'support fit exposure or trial bound differs')
            identities=set();key=history['incumbent_key']
            for trial in history['trials']:
                identity=(tuple(trial['support']),trial['witness']);v.require(identity not in identities and
                    search._support_cost(trial['support'])<=3 and trial['support']==sorted(set(trial['support'])) and trial['witness'] in legal[i],
                    'invalid duplicate support or witness');identities.add(identity);trial_count+=1
                if trial['certified']:
                    accepts=trial['proxy']>key;v.require(trial['accepted']==accepts,'support incumbent comparison differs')
                    if accepts:key=trial['proxy']
                else:v.require(trial['proxy'] is None,'uncertified trial has a proxy')
            v.require(key==history['selected_key'],'selected support key differs from history')
            if 'candidate_model' in record:
                candidate=record['candidate_model'];path=search.region(source.structure,i,256)
                v.require(not history['selected_original'] and candidate['support']==[-j for j in key[-2]] and
                    candidate['witness']==-key[-1],'final support differs from selected history')
                verify_leaf(candidate,path,legal[i],candidate['certificate']);independent_dual(candidate,path,legal[i])
                newscore=search.choices.option_score(search.leaf_option(source,i,candidate),parts,risk)
                v.require(record['candidate_score']==newscore,'stored refreshed option table differs')
                if tuple(newscore['score'])>tuple(old_score['score']):chosen='refreshed';score=newscore;model=candidate;new_count+=1
            else:v.require(history['selected_original'],'changed support lacks final fitted finite option')
        else:v.require(record['status']=='no_internal_fit_exposure_keep_old' and
            (not np.any((fit_route==i)&train) or not np.any((fit_route==i)&selection)),'missing fit search without a reason')
        v.require(record['selected']==chosen and language.payload(final.leaves[i])==language.payload(model),'final selected finite leaf differs')
        for k in totals:totals[k]+=score['totals'][k]
    v.require(totals==report['reliable_counts'],'reliable finite sum differs');n=len(lanes['validation']['x'])
    reliable=totals['E']/n>=.45 and totals['F']/n>=.45 and totals['expected_F']/n>=.40
    copy=deepcopy(final);copy.report=None;copy.feedback_leaves=frozenset();copy.explanation_leaves=frozenset();language.qualify(copy,lanes)
    v.require(reliable==report['reliable_selection_passed']==r['reliable_fit_only_passed'] and language.measure(copy,lanes['validation'])==report['fit_only_metrics'],'fit-only gate or full metrics differ')
    audited_program(final,outer,r['metrics'])
    for split,a in outer.items():v.require(language.measure(source,a)==r['baseline_metrics'][split]==sr['metrics'][split],'baseline differs')
    a=outer['validation']
    for task in (1,2,3):
        part=search.choices.subset(a,np.array([row['task']==task for row in a['rows']],bool))
        expected=dict(baseline=language.measure(source,part),candidate=language.measure(final,part),old_scenarios=len({row['base_seed'] for row in part['rows']}))
        v.require(expected==r['task_metrics'][str(task)] and expected['old_scenarios']==100,'fixed task slice differs')
    calls=ledger(Path(ref['path']).with_name('ledger.jsonl'),r['counts'],spec['limits'])
    v.require(r['numerical_passed']==strict(r['metrics']['validation']) and r['development_passed']==(r['numerical_passed'] and reliable),'outer gate differs')
    return dict(job=spec['id'],input=ref,integrity_passed=True,independent_program_frames=sum(len(a['x']) for a in outer.values()),
        independent_final_leaf_certificates=len(final.leaves),physical_guard_checks=physical_bounds(final.structure),
        support_trial_history_entries=trial_count,optimizer_trials_refitted=False,optimizer_candidate_parameters_independently_replayed=False,
        stored_final_finite_option_tables_independently_replayed=True,frozen_exact_leaves=frozen_count,changed_leaves=new_count,
        solver_counts=calls,numerical_passed=r['numerical_passed'],development_passed=r['development_passed'])


def main(folder):
    v.require(not (folder/'independent-replay.json').exists(),'single-use replay already exists')
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need six actual completed programs')
    auditplan=dict(source=v.pointer(__file__),plan=planref,summary=summaryref,maximum_workers=3,optimizer_candidates_refitted=False)
    write_new(folder/'audit-plan.json',auditplan);(folder/'audit-plan.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(unit,str(folder),ref) for ref in summary['units']]):
            r=f.result();rows.append(r);print(r['job'],'independent leaf replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==auditplan['source'],'NN or replay source changed')
    result=dict(schema='warehouse-v637-independent-leaf-replay.v1',audit_plan=v.pointer(folder/'audit-plan.json'),units=sorted(rows,key=lambda r:r['job']),
        integrity_passed=len(rows)==6 and all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'independent-replay.json',result);(folder/'independent-replay.json').chmod(0o444);print(json.dumps(v.pointer(folder/'independent-replay.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
