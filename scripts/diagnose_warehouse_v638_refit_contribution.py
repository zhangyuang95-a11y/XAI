#!/usr/bin/env python3
"""Post-hoc OLD-output-only finite control; no fit, new data or method lock."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v638_joint_options as s,warehouse_v635_program as language
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.audit_warehouse_v630 import audited_program
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def one(ref):
    r=v.bound(ref);spec=r['spec'];report=v.bound(r['search_report']);_,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    source=language.Program.from_dict(v.bound(v.bound(spec['source_result'])['program']),old['actor_role_sha256'][spec['role']]);base=language.import_previous(source)
    cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=s.choices.metrics.lane_parts(a,cases)
    groups={g:[o for o in options if not o['id'].startswith('refit-')] for g,options in report['groups'].items()}
    n=len(lanes['validation']['x']);candidates=[];best=None
    for weight in (0.,.5,1.,2.):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=s.allocate(weighted,256);selected={tuple(report['option_objects'][g]['root']):report['option_objects'][g]['options'][name] for g,name in allocation['selected'].items()}
        p=s.build.assemble(base,selected);language.qualify(p,lanes);observed=language.measure(p,lanes['validation'])
        counts={k:sum(report['tables'][g][name]['totals'][k] for g,name in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*s.choices.score_metrics(observed,observed['explanation']['whole_tree_kl'],len(p.leaves)))
        candidates.append(dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_passed=reliable,metrics=observed,score=list(score)))
        if best is None or score>best[0]:best=(score,p,len(candidates)-1)
    program=best[1];outer=data.load_parts(dp,spec['fold'],spec['role']);language.qualify(program,outer);m=language.measure(program,outer['validation'])
    audited_program(program,outer,{split:language.measure(program,part) for split,part in outer.items()})
    full=report['candidates'][report['selected']]
    return dict(job=spec['id'],input=ref,posthoc=True,old_only_candidates=candidates,old_only_selected=best[2],
        old_only_outer_metrics=m,full_outer_metrics=r['metrics']['validation'],old_only_numerical_passed=strict(m),
        old_only_reliable_passed=candidates[best[2]]['reliable_passed'],full_reliable_counts=full['reliable_counts'],selection_frames=n,
        full_minus_old_only_E_pp=100*(r['metrics']['validation']['explanation']['coverage']-m['explanation']['coverage']),
        fixed_control_is_not_independent_full_training_ablation=True,optimizer_refits=0,new_environment_resets=0)


def main(folder,output):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref);auditref=v.pointer(folder/'independent-replay.json')
    v.require(summary['all_six_settled'] and v.bound(auditref)['integrity_passed'],'need six complete independently replayed joint results')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    plan=dict(source=source,summary=summaryref,audit=auditref,posthoc=True,method_lock_authority=False,optimizer_refits=0,maximum_workers=3)
    write_new(output/'plan.json',plan);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,ref) for ref in summary['units']]):
            r=f.result();rows.append(r);print(r['job'],'fixed OLD-only control complete',flush=True)
    v.require(v.pointer(__file__)==source,'control report source changed')
    result=dict(schema='warehouse-v638-posthoc-fixed-output-contribution.v1',plan=v.pointer(output/'plan.json'),units=sorted(rows,key=lambda r:r['job']),
        new_fits=0,new_environment_resets=0,method_lock_authority=False,pilot_opened=False)
    write_new(output/'result.json',result);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
