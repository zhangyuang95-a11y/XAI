#!/usr/bin/env python3
"""Read-only FIT-lane legality, fidelity and reliability failure decomposition."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v
from backend.training import warehouse_v635_program as language,warehouse_v636_search as search
from backend.training.warehouse_v630_program import Program as Previous
from backend.training.warehouse_v622_program import _eligible
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v625 import write_new


def one(ref):
    r=v.bound(ref);spec=r['spec'];_,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);parts=search.metrics.lane_parts(a,cases)
    subsets=search.metrics.risk_subsets(parts['eligibility']['rows']);out=[]
    for name,program in (('uniform_v633',language.import_previous(Previous.from_dict(v.bound(v.bound(spec['source_result'])['program']),old['actor_role_sha256'][spec['role']]))),
        ('context_v636',language.Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']]))):
        pred={s:program.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
        signals={s:search.metrics.frame_signals(program,parts[s],value) for s,value in pred.items()};n=len(parts['validation']['x'])
        count=np.bincount(pred['fit'][0],minlength=len(program.leaves));legal=program.certificates();children=[];accept=np.zeros(n,bool)
        for i in range(len(program.leaves)):
            risk=pred['eligibility'][0]==i;choose=pred['validation'][0]==i
            rm=search.metrics.row_metrics(signals['eligibility'],risk);sm=search.metrics.row_metrics(signals['validation'],choose)
            repeats=sum(_eligible(search.metrics.row_metrics(signals['eligibility'],risk&m),.95) for m in subsets)
            ok=bool(count[i] and _eligible(rm,.95) and repeats>=27 and sm['rows'] and sm['agreement']>=.97 and sm['kl']<=.20)
            if ok:accept|=choose
            reasons=[]
            if not count[i]:reasons.append('no_fit_exposure')
            if not _eligible(rm,.95):reasons.append('risk_leaf_ineligible')
            if repeats<27:reasons.append('risk_repeat_below27')
            if not sm['rows']:reasons.append('no_selection_rows')
            elif sm['agreement']<.97:reasons.append('selection_agreement_below97')
            if sm['kl'] is not None and sm['kl']>.20:reasons.append('selection_KL_above020')
            p=parts['validation']['p'][choose];q=pred['validation'][2][choose];nn=p.argmax(1);top=q.argmax(1)
            allowed=np.array([int(a) in legal[i] for a in nn],bool);match=top==nn
            children.append(dict(leaf=i,accepted=ok,reasons=reasons,risk=rm,selection=sm,repeated_passes=repeats,
                rows=int(choose.sum()),NN_mode_not_path_proven=int((~allowed).sum()),
                wrong_mode_with_path_proven=int((~match&allowed).sum()),wrong_mode_without_path_proof=int((~match&~allowed).sum()),
                mean_NN_margin=float(np.mean(np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2])) if len(p) else None))
        excluded=[c for c in children if not c['accepted']]
        counts={k:sum(c[k] for c in excluded) for k in ('rows','NN_mode_not_path_proven','wrong_mode_with_path_proven','wrong_mode_without_path_proof')}
        reason_rows={reason:sum(c['rows'] for c in excluded if reason in c['reasons']) for reason in sorted({t for c in excluded for t in c['reasons']})}
        if name=='context_v636':
            report=v.bound(r['search_report']);expected=report['candidates'][report['selected']]['reliable_counts']['E']
            v.require(int(accept.sum())==expected,'diagnostic reliability count differs from locked experiment')
        out.append(dict(method=name,fit_only_selection_frames=n,reliable_E=float(accept.mean()),excluded_selection_counts=counts,
            overlapping_reason_rows=reason_rows,leaves=children))
    return dict(job=spec['id'],input=ref,methods=out)


def main(folder,output):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'requires six actual programs')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    plan=dict(source=source,summary=summaryref,created_utc=datetime.now(timezone.utc).isoformat(),data='old_fit_lanes_only_no_outer_validation_labels',
        new_fits=0,new_environment_resets=0,maximum_workers=3)
    write_new(output/'plan.json',plan);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,ref) for ref in summary['units']]):
            row=f.result();rows.append(row);print(row['job'],'fit-error decomposition complete',flush=True)
    v.require(v.pointer(__file__)==source,'diagnostic source changed')
    result=dict(schema='warehouse-v636-fit-error-decomposition.v1',plan=v.pointer(output/'plan.json'),units=sorted(rows,key=lambda r:r['job']),new_fits=0,
        NN_updates=0,new_environment_resets=0,thresholds_changed=False,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',result);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
