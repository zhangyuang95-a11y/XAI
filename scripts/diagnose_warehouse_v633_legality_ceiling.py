#!/usr/bin/env python3
"""Exact fixed-path legality ceiling, separate from fitted accuracy.

Current certificates force every non-path-proven action below a legal
witness throughout a leaf. Thus raw argmax belongs to path-proven actions.
If the NN top action is outside that set, no coefficient/support update
within this exact certificate contract can match it at that frame.
No predicates, coefficients, eligibility sets or admission gates are changed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_certificate import region
from experiments.warehouse.v630_semantic_protection import verify
from scripts.run_warehouse_v625 import write_new


def one(ref):
    r=v.bound(ref);spec=r['spec'];_,old=v.checked_plan();dp=v.bound(v.bound(old['previous_plan'])['previous_plan'])
    p=Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']]);parts=data.load_parts(dp,spec['fold'],spec['role'])
    legal=p.certificates();protected=set(verify(p)['protected_descendants']);stats={};leaves=[]
    for split in ('eligibility','validation'):
        a=parts[split];routes,z,q=p.distributions(a['x'],a['m']);nn=a['p'].argmax(1);top=z.argmax(1)
        possible=np.array([int(action) in legal[int(i)] for i,action in zip(routes,nn,strict=True)])
        raw_proven=np.array([int(action) in legal[int(i)] for i,action in zip(routes,top,strict=True)])
        v.require(raw_proven.all() and np.array_equal(z.argmax(1),q.argmax(1)), 'raw argmax escaped the path-certified legal set')
        n=np.bincount(routes,minlength=len(p.leaves));correct=np.bincount(routes,weights=top==nn,minlength=len(p.leaves))
        possible_n=np.bincount(routes,weights=possible,minlength=len(p.leaves))
        stats[split]=dict(n=n,correct=correct,possible=possible_n,total=len(routes),possible_frames=int(possible.sum()))
    eq=stats['eligibility'];val=stats['validation'];possible_leaf=np.divide(eq['possible'],eq['n'],out=np.zeros(len(p.leaves)),where=eq['n']>0)>=.95
    for i in range(len(p.leaves)):
        count=int(eq['n'][i]);n=int(val['n'][i]);depth=len(region(p.structure,i,p.maximum_leaves)['path'])
        leaves.append(dict(leaf=i,protected=i in protected,path_guaranteed_legal_actions=sorted(legal[i]),depth=depth,
            eligibility_frames=count,eligibility_maximum_possible_agreement=float(eq['possible'][i]/count) if count else None,
            validation_frames=n,validation_actual_agreement=float(val['correct'][i]/n) if n else None,
            validation_maximum_possible_agreement=float(val['possible'][i]/n) if n else None,
            can_reach_95_percent_on_same_eligibility_pool=bool(possible_leaf[i]),
            validation_NN_top_outside_path_legal_frames=int(val['n'][i]-val['possible'][i])))
    return dict(job=spec['id'],input=ref,program=r['program'],leaves=leaves,
        whole_validation_actual_agreement=float(val['correct'].sum()/val['total']),
        whole_validation_maximum_possible_agreement=val['possible_frames']/val['total'],
        optimistic_validation_E_coverage_from_eligibility_ceiling=float(val['n'][possible_leaf].sum()/val['total']),
        unprotected_validation_fraction=float(sum(x['validation_frames'] for x in leaves if not x['protected'])/val['total']),
        unprotected_mismatches_due_to_outside_legal_frames=sum(x['validation_NN_top_outside_path_legal_frames'] for x in leaves if not x['protected']),
        ceiling_only_for_fixed_paths_and_existing_certificate_contract=True,program_changed=False)


def run(summary_path,output):
    ref=v.pointer(summary_path);summary=v.bound(ref)
    refs=[r for r in summary['units'] if v.bound(r)['spec'].get('arm') in (None,'oblique')]
    v.require(len(refs)==6 and all(v.bound(r)['status']=='completed' for r in refs),'need all six fixed actual programs')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    config=dict(source=source,summary=ref,inputs=refs,maximum_workers=3,diagnostic_only=True,
        logical_argument='certificate_dominates_every_action_not_in_path_legal_set_by_a_guaranteed_legal_witness',
        qualification_ceiling_threshold=.95,relaxed_for_optimistic_E_bound=['KL','minimum_exposure','repeat_stability','fitting_capacity'])
    write_new(output/'plan.json',config);(output/'plan.json').chmod(0o444)
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,r) for r in refs]):
            r=f.result();rows.append(r);print(r['job'],r['whole_validation_actual_agreement'],r['whole_validation_maximum_possible_agreement'],r['optimistic_validation_E_coverage_from_eligibility_ceiling'],flush=True)
    for r in refs:v.require(v.pointer(r['path'])==r,'fixed input changed')
    v.require(v.pointer(__file__)==source,'diagnostic source changed')
    result=dict(schema='warehouse-fixed-path-legality-accuracy-ceiling.v1',plan=v.pointer(output/'plan.json'),units=sorted(rows,key=lambda r:r['job']),
        new_fits=0,NN_updates=0,new_environment_resets=0,actual_qualification_or_gates_changed=False,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444);print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.summary.resolve(),a.output.resolve())
