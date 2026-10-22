#!/usr/bin/env python3
"""Read-only action-proof ceiling, NN margins and mode mixing diagnostics.

The zero-KL ideal is a NECESSARY-CONDITION upper diagnostic, never a model,
certificate, admission score or executable action rule.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v
from backend.training import warehouse_v635_program as language,warehouse_v635_search as choices
from backend.training.warehouse_v622_program import _eligible
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v625 import write_new


def one(ref):
    r=v.bound(ref);spec=r['spec'];_,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    cases,a,_=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);parts=choices.metrics.lane_parts(a,cases)
    program=language.Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']]);legal=program.certificates()
    pred={s:program.distributions(a['x'],a['m']) for s,a in parts.items() if s!='calibration'}
    signals={s:choices.metrics.frame_signals(program,parts[s],value) for s,value in pred.items()}
    known={s:np.array([int(a) in legal[int(i)] for a,i in zip(parts[s]['p'].argmax(1),pred[s][0],strict=True)],bool) for s in pred}
    ideal={s:dict(value,correct=known[s],losses=np.zeros(len(known[s])),illegal=np.zeros(len(known[s]),bool),unproved=np.zeros(len(known[s]),bool)) for s,value in signals.items()}
    repeated=choices.metrics.risk_subsets(parts['eligibility']['rows']);fit_count=np.bincount(pred['fit'][0],minlength=len(program.leaves))
    n=len(parts['validation']['x']);p=parts['validation']['p'];sort=np.sort(p,axis=1);nn=p.argmax(1);top=pred['validation'][2].argmax(1)
    margin=sort[:,-1]-sort[:,-2];prob=sort[:,-1];reliable=np.zeros(n,bool);optimistic=np.zeros(n,bool);children=[]
    for i in range(len(program.leaves)):
        risk=pred['eligibility'][0]==i;choose=pred['validation'][0]==i
        def passes(sig):
            rm=choices.metrics.row_metrics(sig['eligibility'],risk);sm=choices.metrics.row_metrics(sig['validation'],choose)
            rep=sum(_eligible(choices.metrics.row_metrics(sig['eligibility'],risk&mask),.95) for mask in repeated)
            return bool(fit_count[i] and _eligible(rm,.95) and rep>=27 and sm['rows'] and sm['agreement']>=.97 and sm['kl']<=.20)
        actual=passes(signals);upper=passes(ideal)
        if actual:reliable|=choose
        if upper:optimistic|=choose
        v.require(not actual or upper,'necessary action legality ceiling below actual')
        wrong=choose&(top!=nn);wrong_known=wrong&known['validation'];wrong_unknown=wrong&~known['validation']
        train=pred['fit'][0]==i;NNtrain=parts['fit']['p'][train].argmax(1);Qtrain=pred['fit'][2][train].argmax(1)
        children.append(dict(leaf=i,reliable=actual,optimistic_action_legal_ceiling_qualified=upper,selection_rows=int(choose.sum()),
            NN_mode_path_proven_fraction=float(known['validation'][choose].mean()) if choose.any() else None,
            wrong_known_rows=int(wrong_known.sum()),wrong_unproved_rows=int(wrong_unknown.sum()),
            known_error_NN_probability_median=float(np.median(prob[wrong_known])) if wrong_known.any() else None,
            known_error_NN_margin_median=float(np.median(margin[wrong_known])) if wrong_known.any() else None,
            known_errors_margin_below020=int(np.sum(wrong_known&(margin<.2))),
            known_errors_NN_probability_above090=int(np.sum(wrong_known&(prob>=.9))),
            fit_NN_action_counts=np.bincount(NNtrain,minlength=5).tolist(),fit_Q_action_counts=np.bincount(Qtrain,minlength=5).tolist(),
            fit_correct_NN_action_counts=np.bincount(NNtrain[NNtrain==Qtrain],minlength=5).tolist()))
    report=v.bound(r['search_report']);c=report['candidates'][report['selected']]
    v.require(int(reliable.sum())==c['reliable_counts']['E'],'actual reliability reconstruction differs')
    excluded=~reliable;wrong=excluded&(top!=nn);wk=wrong&known['validation'];wu=wrong&~known['validation']
    return dict(job=spec['id'],input=ref,selection_frames=n,actual_reliable_E=float(reliable.mean()),optimistic_action_legality_E=float(optimistic.mean()),
        excluded_frames=int(excluded.sum()),wrong_with_proven_NN_mode=int(wk.sum()),wrong_with_unproved_NN_mode=int(wu.sum()),
        known_error_NN_probability_median=float(np.median(prob[wk])) if wk.any() else None,
        known_error_NN_margin_median=float(np.median(margin[wk])) if wk.any() else None,
        known_errors_margin_below020=int(np.sum(wk&(margin<.2))),known_errors_NN_probability_above090=int(np.sum(wk&(prob>=.9))),
        unqualified_fit_mixed_NN_action_leaves=sum(not c['reliable'] and sum(x>0 for x in c['fit_NN_action_counts'])>=3 for c in children),
        leaves=children,optimistic_ceiling_does_not_assert_representability=True,gate_or_program_changed=False)


def main(folder,output):
    sr=v.pointer(folder/'result.json');summary=v.bound(sr);ar=v.pointer(folder/'independent-replay.json')
    v.require(summary['all_six_settled'] and v.bound(ar)['integrity_passed'],'need six completed audited programmes')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    plan=dict(source=source,summary=sr,audit=ar,created_utc=datetime.now(timezone.utc).isoformat(),data='old_fit_lanes_only',new_fits=0,
        new_environment_resets=0,method_selection_or_admission_authority=False,maximum_workers=1)
    write_new(output/'plan.json',plan);rows=[]
    for ref in summary['units']:
        row=one(ref);rows.append(row);print(row['job'],json.dumps({k:row[k] for k in ('actual_reliable_E','optimistic_action_legality_E','wrong_with_proven_NN_mode','wrong_with_unproved_NN_mode','known_error_NN_margin_median')}),flush=True)
    v.require(v.pointer(__file__)==source,'diagnostic source changed')
    result=dict(schema='warehouse-v640-action-proof-headroom.v1',plan=v.pointer(output/'plan.json'),units=rows,new_fits=0,new_environment_resets=0,
        gate_or_program_changes=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',result);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
