"""Alternate full-NN leaf fitting with confidence structure selection.

Unlike v637, fits occur in the improved v639 regions. All fitted candidates
remain available until OLD/NEW confidence subregions are jointly evaluated.
No NN parameters, physical rules, source core or reliable E output change.
"""
from copy import deepcopy
import numpy as np
from . import warehouse_v637_leaf_refresh as fitting
from . import warehouse_v638_joint_options as entropy
from . import warehouse_v639_joint_precision as precision
from . import warehouse_v635_program as language
from .warehouse_v628_certificate import region
from .warehouse_v622_program import plain,digest
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-alternating-complete-distribution-leaves-and-confidence.v6.40'
choices=fitting.choices


def fit_candidates(source,outer,lanes,budget):
    source.validate();core=set(language.protection(source)['protected_descendants']);legal=source.certificates()
    a=outer['fit'];tr,se=fitting.internal_split(a['rows']);routes={s:source.leaf_indices(p['x']) for s,p in outer.items()}
    repeated=choices.metrics.risk_subsets(lanes['eligibility']['rows']);_,paths=choices.encode(choices.decode(source.structure));records=[]
    for i,incumbent in enumerate(source.leaves):
        budget.check();parts,masks=choices.root_parts(source,paths[i],lanes);risk=[r[masks['eligibility']] for r in repeated]
        score=choices.option_score(fitting.leaf_option(source,i,incumbent),parts,risk)
        frozen=i in core or score['children'][0]['reliable_selected']['explanation'];take=routes['fit']==i
        train,selection=take&tr,take&se;record=dict(leaf=i,original_core=i in core,frozen_reliable_E_or_core=bool(frozen),old_score=score)
        if frozen:record['status']='exact_payload_preserved'
        elif not train.any() or not selection.any():record['status']='no_internal_fit_exposure_keep_old'
        else:
            path=region(source.structure,i,256)
            selected,history=fitting.select_leaf(incumbent,fitting.arrays(a,train),fitting.arrays(a,selection),path,legal[i],budget,i)
            record['search']=history;record['status']='support_search_completed'
            if not history['selected_original']:
                model=fitting.fit_leaf(*fitting.arrays(a,take),arm='C3',path=path,legal=legal[i],budget=budget,initial=selected,final=True)
                cal=outer['calibration'];model=fitting.calibrate(model,*fitting.arrays(cal,routes['calibration']==i),arm='C3',path=path,legal=legal[i],budget=budget)
                record['candidate_model']=model
                # A whole-leaf zero score MUST NOT discard this model before
                # the joint confidence/leaf comparison (v637 failure mode).
                record['candidate_score']=choices.option_score(fitting.leaf_option(source,i,model),parts,risk)
                record['retained_even_if_whole_leaf_zero_score']=True
        records.append(record);budget.checkpoint(dict(stage='conditional_full_distribution_fit',completed_leaves=len(records),total_leaves=len(source.leaves),
            fitted_candidates=sum('candidate_model'in r for r in records)))
    return plain(dict(version=VERSION,records=records,source_structure_sha256=digest(source.structure),
        original_core=sorted(core),outer_qualification_not_consulted=True,NN_updates=0))


def run(source,outer,lanes,budget):
    candidates=fit_candidates(source,outer,lanes,budget);fit_counts=dict(budget.counts)
    _,entropy_report=entropy.run(source,outer,lanes,budget,candidates);entropy_counts=dict(budget.counts)
    out,union=precision.run(source,outer,lanes,budget,candidates,entropy_report)
    report=deepcopy(union);report.update(version=VERSION,candidate_report=candidates,entropy_report=entropy_report,
        fit_phase_counts=fit_counts,entropy_phase_cumulative_counts=entropy_counts,source_structure_sha256=digest(source.structure),
        old_and_fitted_models_jointly_selected_after_refinement=True,complete_NN_distribution_target=True,
        NN_updates=0,training_authority=False,pilot_opened=False)
    report.pop('candidate_models_never_refitted',None);report.pop('inherited_all_v638_options',None);report.pop('new_entropy_fits',None)
    report.update(coefficient_fit_calls=budget.counts['leaf'],proof_optimizer_calls=budget.counts['proof'],
        dense_proposal_optimizer_calls=fit_counts['node'],entropy_direction_optimizer_calls=entropy_counts['node']-fit_counts['node'],
        precision_direction_optimizer_calls=budget.counts['node']-entropy_counts['node'],
        candidate_coefficients_fixed_during_joint_partition_selection=True,inherited_all_internal_entropy_options=True)
    return out,plain(report)
