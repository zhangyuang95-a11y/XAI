#!/usr/bin/env python3
"""Parallel read-only final-evidence/finite-selection replay, no optimizer.

Optimizer candidates are not refitted. Their source-pinned sequence and call
ledger are checked; final distributions/proofs/eligibility and every frozen
confidence table/allocation are independently recomputed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v633_support_refresh as support,warehouse_v632_search as confidence
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v630_partition_proposals import allocate
from backend.training.warehouse_v628_program import measure,payload
from scripts.audit_warehouse_v630 import audited_program
from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
from scripts.run_warehouse_v633 import strict
from scripts.run_warehouse_v625 import write_new


def ledger(path,counts,limits):
    tail='0'*64;pending=None;actual={k:0 for k in limits};phases={}
    for i,line in enumerate(path.read_text().splitlines(),1):
        row=json.loads(line);sha=row.pop('record_sha256')
        v.require(row['sequence']==i and row['previous_sha256']==tail and v.digest(row)==sha,'ledger chain differs')
        tail=sha;d=row['data'];kind=d['kind'];phase=d['phase']
        v.require(kind in actual and phase in ('support','confidence'),'unexpected solver kind or phase')
        if row['event']=='call_started':
            actual[kind]+=1;v.require(pending is None and d['call']==actual[kind] and actual[kind]<=limits[kind],'call budget/sequence differs')
            v.require((phase=='support' and kind in ('leaf','proof')) or (phase=='confidence' and kind=='node'),'solver escaped its phase')
            pending=(kind,d['call'],phase);phases.setdefault(phase,{k:0 for k in limits})[kind]+=1
        else:
            v.require(row['event']=='call_completed' and pending==(kind,d['call'],phase),'unsettled solver call')
            pending=None
    v.require(pending is None and actual==counts,'actual optimizer counts differ')
    return phases


def unit(output,ref):
    folder=Path(output);planref=v.pointer(folder/'plan.json');plan=v.bound(planref);r=v.bound(ref);spec=r['spec']
    v.require(r['status']=='completed' and r['plan']==planref and spec in plan['jobs'],'incomplete or unbound unit')
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    sr=v.bound(spec['source_result']);source=Program.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
    path=Path(ref['path']).parent;rawref=v.pointer(path/'refreshed-unqualified.json')
    raw=Program.from_dict(v.bound(rawref),source.actor_role_sha256)
    control=Program.from_dict(v.bound(r['support_only_program']),source.actor_role_sha256)
    final=Program.from_dict(v.bound(r['program']),source.actor_role_sha256)
    parts=data.load_parts(dp,spec['fold'],spec['role']);report=v.bound(r['support_report'])
    protected=support.protected_indices(source);routes=source.leaf_indices(parts['fit']['x'])
    train,selection=support.internal_split(parts['fit']['rows'])
    v.require(raw.report is None and not raw.feedback_leaves and not raw.explanation_leaves,'unqualified boundary inherited evidence')
    v.require(v.digest(raw.structure)==v.digest(source.structure)==v.digest(control.structure),'support refresh changed topology')
    v.require([x['leaf'] for x in report['records']]==list(range(len(source.leaves))),'support search omitted/reordered leaves')
    v.require(report['train_scenes']==sorted({row['base_seed'] for row,keep in zip(parts['fit']['rows'],train,strict=True) if keep}) and
              report['selection_scenes']==sorted({row['base_seed'] for row,keep in zip(parts['fit']['rows'],selection,strict=True) if keep}),
              'support choice used different whole-scene lanes')
    changed=0
    for i,record in enumerate(report['records']):
        v.require(payload(raw.leaves[i])==payload(control.leaves[i]),'qualifying changed the refreshed leaf output')
        take=routes==i
        if i in protected:
            v.require(record['status']=='protected_exact_payload_preserved' and payload(raw.leaves[i])==payload(source.leaves[i]),'protected output changed')
        elif not (take&train).any() or not (take&selection).any():
            v.require(record['status']=='no_fit_lane_exposure_keep_incumbent' and payload(raw.leaves[i])==payload(source.leaves[i]),'missing-lane incumbent changed')
        else:
            trials=record['trials'];v.require(1<=len(trials)<=8 and len({(tuple(t['support']),t['witness']) for t in trials})==len(trials),'support trials differ or exceed8')
            v.require(all(support._support_cost(t['support'])<=3 and t['witness'] in source.certificates()[i] for t in trials),'illegal source/witness trial')
            v.require(record['fit_frames']==int((take&train).sum()) and record['selection_frames']==int((take&selection).sum()),'support exposure counts differ')
            if record['selected_original']:
                v.require(payload(raw.leaves[i])==payload(source.leaves[i]),'incumbent choice changed payload')
            else:
                accepted=[t for t in trials if t['accepted']];v.require(accepted,'changed support lacks accepted trial')
                chosen=accepted[-1];v.require(raw.leaves[i]['support']==chosen['support'] and raw.leaves[i]['witness']==chosen['witness'],'final support/witness differs from fit-only selection')
                changed+=1
    # Replay all confidence choices using only the original fit sublanes.
    confidence_report=v.bound(r['confidence_report']);base,originals=confidence.sparse.collapse_protected(raw)
    v.require(base.to_dict()==confidence_report['collapsed_base'],'collapsed base differs')
    cases,a,arrayrefs=sampled_fit(dp,v.bound(prior['collection']),spec['fold'],spec['role']);lanes=confidence.old.lane_parts(a,cases)
    pred={s:base.distributions(a['x'],a['m']) for s,a in lanes.items() if s!='calibration'}
    signals={s:confidence.old.frame_signals(base,lanes[s],p) for s,p in pred.items()};subsets=confidence.old.risk_subsets(lanes['eligibility']['rows'])
    groups={};local={}
    for i,record in enumerate(confidence_report['proposal_records']):
        v.require(record['source_leaf']==i and record['oblique'] is True and record['original_option_sha256']==v.digest(originals[i]),'confidence proposal provenance differs')
        options={v.digest(t):t for t in [*record['proposals'],originals[i]]}
        g,d=confidence.old.local_table(base,i,[options[k] for k in sorted(options)],lanes,pred,signals,subsets)
        groups[f'{i:04d}']=g;local[f'{i:04d}']=d
    v.require(groups==confidence_report['groups'] and local==confidence_report['local'],'frozen confidence tables differ')
    n=len(lanes['validation']['x']);best=None
    for index,weight in enumerate((0.,.5,1.,2.)):
        weighted={g:[dict(id=o['id'],leaves=o['leaves'],score=[o['score'][0]/(.45*n)+weight*(o['score'][2]/(.45*n)+o['score'][1]/(.40*n)),*o['score']]) for o in options] for g,options in groups.items()}
        allocation=allocate(weighted,256)
        p,_=confidence.old.assemble(base,{int(g):local[g][i]['structure'] for g,i in allocation['selected'].items()})
        confidence.old.qualify_current(p,lanes);metrics=measure(p,lanes['validation'])
        counts={k:sum(local[g][i]['totals'][k] for g,i in allocation['selected'].items()) for k in ('E','F','expected_F')}
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        score=(int(reliable),*confidence.old.score_metrics(metrics,metrics['explanation']['whole_tree_kl'],len(p.leaves)))
        expected=dict(weight=weight,allocation=allocation,reliable_counts=counts,reliable_selection_passed=reliable,metrics=metrics,score=list(score))
        v.require(expected==confidence_report['candidates'][index],'confidence allocation or candidate score differs')
        if best is None or score>best[0]:best=(score,index,p)
    v.require(best[1]==confidence_report['selected'] and v.digest(best[2].structure)==v.digest(final.structure) and
              all(payload(a)==payload(b) for a,b in zip(best[2].leaves,final.leaves,strict=True)),'final output differs from locked fit-only choice')
    cp=audited_program(control,parts,r['support_only_metrics']);fp=audited_program(final,parts,r['metrics'])
    for split,a in parts.items():
        v.require(measure(source,a)==r['baseline_metrics'][split]==sr['metrics'][split],'source baseline metric dictionary differs')
        v.require(all(np.array_equal(x,y) for x,y in zip(cp[split][1:],fp[split][1:])),'confidence subdivision changed full output')
    phases=ledger(path/'ledger.jsonl',r['counts'],spec['limits'])
    v.require(r['numerical_passed']==strict(r['metrics']['validation']) and
              r['development_passed']==(r['numerical_passed'] and confidence_report['reliable_selection_passed']), 'gates differ')
    return dict(job=spec['id'],integrity_passed=True,source_result=ref,changed_unprotected_leaves=changed,
        final_program_frames=2*sum(len(a['x']) for a in parts.values()),leaf_certificates=len(control.leaves)+len(final.leaves),
        full_confidence_tables_and_all_four_allocations_replayed=True,phase_solver_counts=phases,
        numerical_passed=r['numerical_passed'],development_passed=r['development_passed'],
        optimizer_candidates_refitted=False,NN_updates=0,new_environment_resets=0)


def main(folder):
    target=folder/'independent-replay.json';v.require(not target.exists(),'audit already sealed')
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in summary['units']),'need all six completed units')
    auditplan=dict(source=v.pointer(__file__),plan=planref,summary=summaryref,maximum_workers=3,
                   optimizer_candidates_refitted=False,scope='numeric_proofs_tables_allocation_and_source_pinned_call_history')
    write_new(folder/'audit-plan.json',auditplan);(folder/'audit-plan.json').chmod(0o444)
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(unit,str(folder),ref) for ref in summary['units']]
        for f in as_completed(fs):
            r=f.result();rows.append(r);print(r['job'],'independent replay passed',flush=True)
    for name,sha in plan['source_files'].items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and v.pointer(__file__)==auditplan['source'],'actor or audit source changed')
    result=dict(schema='warehouse-v633-independent-replay.v1',audit_plan=v.pointer(folder/'audit-plan.json'),units=sorted(rows,key=lambda r:r['job']),
        integrity_passed=len(rows)==6 and all(r['integrity_passed'] for r in rows),new_fits=0,NN_updates=0,
        new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(target,result);target.chmod(0o444);print(json.dumps(v.pointer(target)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
