#!/usr/bin/env python3
"""One fixed engineering split on spent data, no search or admission.

The split is supplied by the saved prototype (leaf 10, battery 0.5), never
chosen from these metrics. Both programs are requalified independently with
the same existing partitions; all coefficients and temperatures stay frozen.
"""
from copy import deepcopy
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v629_protocol as v,v628_data as data,v630_semantic_protection as semantic
from backend.training.warehouse_v629_program import Program as Previous
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_program import qualify,measure
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import basic,same_metrics
from scripts.run_warehouse_v625 import write_new


def run(prototype_ref,output):
    path,plan=v.checked_plan();prototype=v.bound(prototype_ref);source=v.bound(prototype['source_result'])
    v.require(prototype['scientific_candidate'] is False and prototype['model_acceptance_passed'] is False
        and prototype['source_program']==source['program'] and prototype['scientific_plan']==v.pointer(path),
        'fixed engineering prototype source changed')
    before=Previous.from_dict(v.bound(source['program']),plan['actor_role_sha256']['robot_1'])
    after=Program.from_dict(v.bound(prototype['unqualified_prototype']),before.actor_role_sha256)
    oldplan=v.bound(plan['previous_plan']);rows=[];refs=[prototype_ref,prototype['source_result'],source['program'],
        prototype['unqualified_prototype'],v.pointer(path),plan['collection']]
    for scope in ('inner','outer'):
        parts=data.load_parts(oldplan,0,'robot_1',inner=scope=='inner')
        a=qualify(deepcopy(before),parts);b=qualify(deepcopy(after),parts)
        # Correct the old qualifier's diagnostic list: refined protected
        # descendants stay frozen too. No refit/calibration is run at all.
        b.report['protected_refit_and_calibration_skipped']=semantic.verify(b)['protected_descendants']
        b.report['qualification_only_no_leaf_fitting']=True;b.validate()
        cells={}
        for split,arrays in parts.items():
            old=a.distributions(arrays['x'],arrays['m']);new=b.distributions(arrays['x'],arrays['m'])
            independent=forward(b,arrays['x'],arrays['m'])
            v.require(all(np.array_equal(x,y) for x,y in zip(new,independent,strict=True))
                and np.array_equal(old[1],new[1]) and np.array_equal(old[2],new[2]),
                'fixed refinement altered distribution or independent forward')
            base=measure(a,arrays);refined=measure(b,arrays)
            for purpose,selected in [('feedback',b.feedback_leaves),('explanation',b.explanation_leaves)]:
                same_metrics(basic(b,arrays,independent,selected,purpose),refined[purpose])
            cells[split]=dict(before=base,after=refined,physical_frames=len(arrays['x']),
                full_program_logits_and_distributions_bitwise_equal=True)
        if scope=='outer':v.require(all(cells['validation']['before'][purpose]==source['metrics']['validation'][purpose]
                for purpose in ('feedback','explanation')),
            'fixed baseline qualification no longer matches its settled receipt')
        rows.append(dict(scope=scope,cells=cells,before_feedback_leaves=sorted(a.feedback_leaves),
            before_explanation_leaves=sorted(a.explanation_leaves),after_feedback_leaves=sorted(b.feedback_leaves),
            after_explanation_leaves=sorted(b.explanation_leaves)))
    for ref in refs:v.require(v.pointer(ref['path'])==ref,'fixed partition diagnostic input changed')
    result=dict(version='warehouse-v630-one-fixed-partition-diagnostic.v1',source_sha256=v.file_sha(__file__),
        fixed_predicate=prototype['fixed_engineering_example'],inputs=refs,rows=rows,
        scope='spent_development_fixed_engineering_example_not_search_method_or_independent_validation',
        new_candidate_registered=False,method_selected=False,solver_calls=0,NN_updates=0,new_environment_resets=0,
        current_v629_candidate_changed=False,model_acceptance_passed=False,pilot_opened=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    return v.pointer(output/'result.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prototype',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(v.pointer(args.prototype),args.output),ensure_ascii=False))
