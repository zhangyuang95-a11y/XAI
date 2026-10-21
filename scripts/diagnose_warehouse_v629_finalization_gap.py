#!/usr/bin/env python3
"""Read-only settled-unit coverage decomposition; never changes selection.

Uses spent development frames only, no fitting, calibration, LP, environment
or NN updates. Inner/outer cross-evaluation is diagnostic, not new evidence
of independent generalization or authority to change the registered search.
"""
from copy import deepcopy
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v629_protocol as v,v628_data as data
from backend.training.warehouse_v629_program import Program
from backend.training.warehouse_v628_program import measure,qualify,metrics
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import basic,same_metrics
from scripts.run_warehouse_v625 import write_new
import numpy as np


def selected_metrics(program,arrays,feedback,explanation):
    direct=program.distributions(arrays['x'],arrays['m']);independent=forward(program,arrays['x'],arrays['m'])
    v.require(all(np.array_equal(x,y) for x,y in zip(direct,independent,strict=True)),'independent diagnostic forward differs')
    result={}
    for purpose,selected in [('feedback',feedback),('explanation',explanation)]:
        result[purpose]=metrics(program,arrays['rows'],(arrays['x'],arrays['m'],arrays['sampled']),arrays['p'],selected,purpose,direct)
        same_metrics(basic(program,arrays,independent,selected,purpose),result[purpose])
    return result


def run(job_id,output):
    path,plan=v.checked_plan();job=next(j for j in plan['jobs'] if j['job_id']==job_id)
    folder=v.OUTPUT/'jobs'/job_id;ref=v.pointer(folder/'result.json');r=v.bound(ref)
    v.require(v.bound(v.pointer(folder/'completed.json'))==dict(result_sha256=ref['sha256']) and r['status']=='completed'
        and r['job']==job and r['plan_sha256']==v.file_sha(path),'diagnose only a fully settled registered unit')
    refs=[v.pointer(path),ref,v.pointer(folder/'completed.json'),r['search_best'],r['program'],r['search_report']]
    b=Program.from_dict(v.bound(r['search_best']),plan['actor_role_sha256'][job['role']])
    f=Program.from_dict(v.bound(r['program']),b.actor_role_sha256)
    v.require(v.digest(b.structure)==v.digest(f.structure),'diagnostic requires the same pre/post finalization topology')
    oldplan=v.bound(plan['previous_plan']);inner=data.load_parts(oldplan,job['fold'],job['role'],inner=True)
    outer=data.load_parts(oldplan,job['fold'],job['role']);refs.append(plan['collection'])
    cells={}
    def save(name,p,a,fs=None,es=None):
        fs=p.feedback_leaves if fs is None else fs;es=p.explanation_leaves if es is None else es
        cells[name]=dict(feedback_leaves=sorted(fs),explanation_leaves=sorted(es),metrics=selected_metrics(p,a,fs,es))
    save('search_best_inner_original_qualification',b,inner['validation'])
    save('search_best_outer_original_qualification',b,outer['validation'])
    bo=qualify(deepcopy(b),outer);save('search_best_outer_requalified_without_refit',bo,outer['validation'])
    save('final_outer_search_best_requalified_leaf_sets',f,outer['validation'],bo.feedback_leaves,bo.explanation_leaves)
    save('final_outer_actual_qualification',f,outer['validation'])
    fi=qualify(deepcopy(f),inner);save('final_inner_requalified_without_refit',fi,inner['validation'])
    for purpose in ('feedback','explanation'):
        v.require(cells['final_outer_actual_qualification']['metrics'][purpose]==r['metrics']['validation'][purpose],
            'complete stored metric dictionary differs from exact diagnostic replay')
    inner_ids=set(b.explanation_leaves);outer_ids=set(bo.explanation_leaves);final_ids=set(f.explanation_leaves)
    leaf_changes=[]
    for i in range(len(b.leaves)):
        source=b.report['eligibility'][str(i)]['explanation'];new=bo.report['eligibility'][str(i)]['explanation']
        final=f.report['eligibility'][str(i)]['explanation']
        leaf_changes.append(dict(leaf=i,coefficients_changed=any(b.leaves[i][k]!=f.leaves[i][k]
                for k in ('support','weights','bias','temperature')),
            inner_qualified=i in inner_ids,outer_before_refit_qualified=i in outer_ids,outer_final_qualified=i in final_ids,
            inner_eligibility=source,outer_before_refit_eligibility=new,outer_final_eligibility=final))
    for item in refs:v.require(v.pointer(item['path'])==item,'settled diagnostic input changed')
    result=dict(version='warehouse-v629-read-only-finalization-gap.v1',job=job,inputs=refs,source_sha256=v.file_sha(__file__),
        frozen_scientific_source_sha256=plan['source_sha256'],scope='spent_development_diagnostic_only_not_method_selection',
        cells=cells,leaf_changes=leaf_changes,environment_resets=0,NN_updates=0,solver_calls=0,
        original_files_written=0,registered_candidate_or_admission_changed=False,pilot_opened=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    return v.pointer(output/'result.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(run(args.job,args.output),ensure_ascii=False))
