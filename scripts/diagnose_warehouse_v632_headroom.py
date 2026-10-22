#!/usr/bin/env python3
"""Optimistic reliable E bounds on *fixed* v632 tables, no new predicates.

This deliberately drops every global fidelity/feedback gate, retaining risk
qualification and 27/30 stability. An upper bound below45% rules out changing
only the allocator on these tables. It never supplies an admitted program.
"""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v
from backend.training.warehouse_v622_program import _eligible
from backend.training.warehouse_v630_partition_proposals import allocate
from scripts.run_warehouse_v625 import write_new


def run(folder, output):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref);planref=v.pointer(folder/'plan.json');plan=v.bound(planref)
    v.require(summary['all_twelve_settled'],'wait for the original twelve fit jobs to settle')
    inputs={spec['id']:v.pointer(folder/spec['id']/'fit-report.json') for spec in plan['jobs']}
    output.mkdir(parents=True,exist_ok=False)
    registered=dict(source=v.pointer(__file__),original_plan=planref,original_summary=summaryref,inputs=inputs,
        risk_threshold=.95,risk_subsets=30,minimum_risk_passes=27,maximum_leaves=256,
        relaxed_for_upper_bound_only=['selection_agreement','selection_KL','global_fidelity','feedback_gates','fit_occurrence'],
        actual_method_or_gates_changed=False,fit_calls=0,NN_updates=0,new_environment_resets=0)
    write_new(output/'plan.json',registered);(output/'plan.json').chmod(0o444)
    results=[]
    for spec in plan['jobs']:
        report=v.bound(inputs[spec['id']]);groups={}
        for group,options in report['local'].items():
            group_options=[]
            for ident,option in options.items():
                score=sum(c['selection']['rows'] for c in option['children'] if
                          _eligible(c['risk'],.95) and c['risk_explanation_passes']>=27)
                group_options.append(dict(id=ident,leaves=len(option['children']),score=[score]))
            groups[group]=group_options
        best=allocate(groups,256)
        # Independent cost-only dense integer DP, without allocator IDs,
        # lexicographic ties or its tuple-valued state implementation.
        scores=np.full(257,-10**12,dtype=np.int64);scores[0]=0
        for group in sorted(groups):
            updated=np.full(257,-10**12,dtype=np.int64)
            for option in groups[group]:
                k=option['leaves'];updated[k:]=np.maximum(updated[k:],scores[:-k]+option['score'][0])
            scores=updated
        v.require(int(scores.max())==best['score'][0],'independent integer upper bound differs')
        n=report['candidates'][0]['metrics']['explanation']['physical_frame_count']
        result=dict(job=spec['id'],E_upper_rows=best['score'][0],physical_selection_rows=n,
            optimistic_reliable_E_upper=best['score'][0]/n,leaves=best['leaves'],allocation=best,
            allocator_alone_impossible_for_these_options=best['score'][0]/n<.45)
        results.append(result);print(spec['id'],f"{result['optimistic_reliable_E_upper']:.5%}",flush=True)
    for ref in inputs.values():v.require(v.pointer(ref['path'])==ref,'frozen option table changed')
    v.require(v.pointer(__file__)==registered['source'],'diagnostic source changed')
    out=dict(schema='warehouse-v632-fixed-table-optimistic-headroom.v1',plan=v.pointer(output/'plan.json'),units=results,
        actual_method_or_gates_changed=False,new_fits=0,NN_updates=0,new_environment_resets=0,
        model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',out);(output/'result.json').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.output.resolve())
