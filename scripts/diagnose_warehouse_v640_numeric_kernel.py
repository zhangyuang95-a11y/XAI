#!/usr/bin/env python3
"""Cached-state numerical probes of legacy masked-CE objective/gradient.

No optimizer call, fitting, parameter update or environment reset. Compare
BLAS @ with explicit einsum and finite differences at recorded models.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logsumexp
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v635_program as language
from backend.training.warehouse_v626_domain import transform
from scripts.run_warehouse_v625 import write_new


def objective(flat,design,m,p,prior,*,stable):
    w=flat.reshape(5,design.shape[1]);z=np.einsum('ni,ai->na',design,w,optimize=False) if stable else design@w.T
    normal=logsumexp(np.where(m>.5,z,-np.inf),axis=1);q=np.exp(np.where(m>.5,z,-np.inf)-normal[:,None])
    value=float(np.mean(p.sum(1)*normal-np.sum(p*z,axis=1)));res=q*p.sum(1)[:,None]-p
    grad=np.einsum('na,ni->ai',res,design,optimize=False)/len(p) if stable else res.T@design/len(p)
    anchor=np.zeros_like(w);anchor[:,0]=prior
    return value+1e-4*float(np.sum((w-anchor)**2)),(grad+2e-4*(w-anchor)).ravel()


def one(ref):
    r=v.bound(ref);spec=r['spec'];_,old=v.checked_plan();dp=v.bound(v.bound(old['previous_plan'])['previous_plan'])
    a=data.load_parts(dp,spec['fold'],spec['role'])['fit'];program=language.Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']])
    route=program.leaf_indices(a['x']);counts=np.bincount(route,minlength=len(program.leaves));core=set(language.protection(program)['protected_descendants'])
    ids=sorted((i for i,n in enumerate(counts) if n>=40 and i not in core),key=lambda i:(-counts[i],i))[:5];rows=[]
    for i in ids:
        take=route==i;leaf=program.leaves[i];x=a['x'][take];m=a['m'][take];p=a['p'][take]
        design=np.column_stack((np.ones(len(x)),transform(x,leaf['support']).astype(float)))
        theta=np.column_stack((np.asarray(leaf['bias'],float),np.asarray(leaf['weights'],float))).ravel()
        prior=np.log((p.sum(0)+.2)/(m.sum(0)+1.));prior-=prior.mean()
        for probe in (0,1):
            values=theta.copy()
            if probe:values+=1e-3*np.sin(np.arange(len(values))+1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',RuntimeWarning);loss,grad=objective(values,design,m,p,prior,stable=False)
            stable,sg=objective(values,design,m,p,prior,stable=True);errors=[];eps=1e-5
            for j in range(len(values)):
                hi=values.copy();lo=values.copy();hi[j]+=eps;lo[j]-=eps
                fd=(objective(hi,design,m,p,prior,stable=True)[0]-objective(lo,design,m,p,prior,stable=True)[0])/(2*eps)
                errors.append(abs(float(sg[j])-fd))
            row=dict(leaf=i,fit_frames=int(take.sum()),probe=probe,support=leaf['support'],legacy_warning_messages=[str(w.message) for w in caught],
                loss_difference=abs(loss-stable),gradient_max_abs_difference=float(np.max(np.abs(grad-sg))),
                stable_gradient_max_finite_difference_error=max(errors),all_values_finite=bool(np.isfinite(loss) and np.isfinite(grad).all() and np.isfinite(stable) and np.isfinite(sg).all()))
            row['probe_passed']=row['all_values_finite'] and row['loss_difference']<=1e-9 and row['gradient_max_abs_difference']<=1e-9 and row['stable_gradient_max_finite_difference_error']<=1e-6
            rows.append(row)
    return dict(job=spec['id'],input=ref,probes=rows,all_probes_passed=bool(rows) and all(r['probe_passed'] for r in rows),
        inference_or_optimizer_parameters_changed=False,optimizer_calls=0)


def main(folder,output):
    sr=v.pointer(folder/'result.json');summary=v.bound(sr)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need six actual model outputs')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    plan=dict(source=source,summary=sr,created_utc=datetime.now(timezone.utc).isoformat(),data='old_fit_only',
        selection='five_largest_noncore_fit_exposed_leaves_per_unit_recorded_parameters_and_deterministic_perturbation',
        new_fits=0,optimizer_calls=0,new_environment_resets=0,maximum_workers=1)
    write_new(output/'plan.json',plan);rows=[]
    for ref in summary['units']:
        r=one(ref);rows.append(r);print(r['job'],'probes',len(r['probes']),'passed',r['all_probes_passed'],flush=True)
    v.require(v.pointer(__file__)==source,'numerical probe source changed')
    result=dict(schema='warehouse-v640-legacy-objective-numerical-probes.v1',plan=v.pointer(output/'plan.json'),units=rows,
        all_probes_passed=all(r['all_probes_passed'] for r in rows),new_fits=0,optimizer_calls=0,new_environment_resets=0,
        scope='finite_recorded_and_nearby_points_only_not_all_historical_optimizer_iterates',model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',result);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
