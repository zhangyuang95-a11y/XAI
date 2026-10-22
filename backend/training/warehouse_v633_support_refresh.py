"""Fit-only support/witness refresh, immutable original protected outputs.

External five-action distribution fitting only. No Actor optimizer, action
controller, seed collection, execution adapter or admission authority.
"""
from copy import deepcopy
import numpy as np
from .warehouse_v626_program import _trial, _rank, _proxy, _support_cost, calibrate, fit_leaf, evaluate_leaf
from .warehouse_v628_certificate import region
from .warehouse_v628_program import payload
from .warehouse_v630_training_program import validate_parts, protected_indices
from .warehouse_v622_program import digest, plain
from scripts.run_warehouse_v626 import internal_split
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-residual-support-witness-refresh.v6.33'


def limits(program):
    n=len(program.leaves)-len(protected_indices(program))
    # 8 support/witness trials x <=8 counterexample rounds, plus <=8 final
    # full-fit rounds. Each round <=4 proof LPs; five temperatures <=20 LPs.
    return dict(leaf=72*n,proof=320*n,node=12544)


def arrays(a,mask):
    return tuple(a[k][mask] for k in ('x','m','p'))


def select_leaf(incumbent, train, selection, path, legal, budget):
    """At most eight distinct support/witness fits; incumbent always retained.

    Ranking uses residuals against the complete masked NN distribution on
    TRAIN only. Candidate choice uses whole-scene internal FIT selection.
    Calibration, qualification and outer validation do not enter this API.
    """
    x,m,p=train
    require(len(x)>0 and len(selection[0])>0,'leaf support selection needs both fit lanes')
    best=deepcopy(incumbent);seen=set();trials=[]
    def key(leaf):
        return (*_proxy(leaf,selection),-_support_cost(leaf['support']),tuple(-i for i in leaf['support']),-leaf['witness'])
    original_key=key(best)

    def attempt(support,witness):
        nonlocal best
        support=tuple(sorted(support));identity=(support,witness)
        if identity in seen or len(seen)>=8 or _support_cost(support)>3:return False
        require(witness in legal and len(support)==len(set(support)),'invalid support/witness proposal')
        seen.add(identity)
        fitted=_trial(x,m,p,support,witness,path,legal,True,budget,initial=incumbent)
        record=dict(support=list(support),witness=int(witness),certified=fitted is not None,
                    proxy=None if fitted is None else list(key(fitted)))
        if fitted is not None and key(fitted)>key(best):best=fitted;record['accepted']=True
        else:record['accepted']=False
        trials.append(record)
        return True

    attempt(incumbent['support'],incumbent['witness'])
    alternative=max(legal,key=lambda a:(float(p[:,a].sum()),-a))
    if alternative!=incumbent['witness']:attempt(incumbent['support'],alternative)
    # Conditional residual ranking is recomputed after every accepted model.
    # Exhausting this finite budget does not assert global support optimality.
    while len(seen)<8:
        budget.check();support=list(best['support']);ranking=_rank(x,p,m,best,support)
        candidates=[]
        if _support_cost(support)<3:
            candidates.extend((sorted(support+[i]),best['witness']) for i in ranking[:2]
                              if _support_cost(support+[i])<=3)
        if support:
            strength=np.linalg.norm(np.asarray(best['weights'],float),axis=0)
            slots=sorted(range(len(support)),key=lambda j:(strength[j],support[j]))
            for i in ranking:
                for j in slots:
                    retained=[k for n,k in enumerate(support) if n!=j]
                    if _support_cost(retained+[i])<=3:
                        candidates.append((sorted(retained+[i]),best['witness']))
        # If no source can enter (e.g. constant public state), stop; no random
        # retries, teacher action labels or hidden features are introduced.
        chosen=next(((s,w) for s,w in candidates if (tuple(s),w) not in seen),None)
        if chosen is None:break
        attempt(*chosen)
    return best,plain(dict(version=VERSION,trials=trials,incumbent_key=list(original_key),selected_key=list(key(best)),
        selected_original=digest(payload(best))==digest(payload(incumbent)),
        complete_NN_distribution_target=True,fit_frames=len(x),selection_frames=len(selection[0]),
        optimality_gap='unknown_outside_the_eight_conditional_trials'))


def refresh(program,parts,budget):
    routes=validate_parts(program,parts);source=deepcopy(program);out=deepcopy(program)
    out.feedback_leaves=frozenset();out.explanation_leaves=frozenset();out.report=None
    protected=protected_indices(program);legal=program.certificates();a=parts['fit']
    train,select=internal_split(a['rows']);records=[]
    for i,leaf in enumerate(source.leaves):
        budget.check();take=routes['fit']==i
        if i in protected:
            records.append(dict(leaf=i,status='protected_exact_payload_preserved'));continue
        tr,sl=take&train,take&select
        if not tr.any() or not sl.any():
            records.append(dict(leaf=i,status='no_fit_lane_exposure_keep_incumbent',train_frames=int(tr.sum()),selection_frames=int(sl.sum())))
            continue
        path=region(source.structure,i,source.maximum_leaves)
        selected,record=select_leaf(leaf,arrays(a,tr),arrays(a,sl),path,legal[i],budget)
        record.update(leaf=i,status='support_search_complete')
        if not record['selected_original']:
            fitted=fit_leaf(*arrays(a,take),arm='C3',path=path,legal=legal[i],budget=budget,
                            initial=selected,final=True)
            cal=parts['calibration'];cal_take=routes['calibration']==i
            out.leaves[i]=calibrate(fitted,*arrays(cal,cal_take),arm='C3',path=path,legal=legal[i],budget=budget)
            require(out.leaves[i]['support']==selected['support'] and out.leaves[i]['witness']==selected['witness'],
                    'final full-fit changed the selected support or witness')
        record['final_payload_sha256']=digest(payload(out.leaves[i]));records.append(record)
    for i in protected:
        require(payload(out.leaves[i])==payload(source.leaves[i]),'original protected output changed')
    require(digest(out.structure)==digest(source.structure),'leaf support refresh changed topology')
    # Do not inspect outer eligibility to select subsequent confidence
    # predicates. Qualification is deliberately deferred to the owning runner
    # after its fit-only candidate selection has finished.
    out.validate()
    return out,plain(dict(version=VERSION,records=records,source_structure_sha256=digest(source.structure),
        source_protection_sha256=digest(source.protection),protected_descendants=sorted(protected),
        train_scenes=sorted({r['base_seed'] for r,keep in zip(a['rows'],train,strict=True) if keep}),
        selection_scenes=sorted({r['base_seed'] for r,keep in zip(a['rows'],select,strict=True) if keep}),
        qualification_and_validation_do_not_select_parameters=True,NN_updates=0,training_authority=False,pilot_opened=False))
