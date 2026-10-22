"""Balanced contextual surrogate trees, full NN distribution leaf targets.

CART only proposes public predicates. Every terminal movement mode receives
an explicit counted physical guard if its path does not already prove it.
No program output is supplied to the physical environment or Actor.
"""
import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.tree import DecisionTreeClassifier
from .warehouse_v635_subtrees import (guards,language,Node,plain,encode,decode,replace,region,
    structure_info,_support_cost,internal_split,fit_option,original_option,local_predictions,assemble)
from .warehouse_v626_domain import domain_table
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-context-first-guarded-subtrees.v6.36'
DEPTHS=(2,3)
MINIMUM_CONTEXT_FRAMES=160


def learn_tree(arrays,known,remaining,budget,*,context_depth,root):
    require(context_depth in DEPTHS and remaining>=0,'unknown contextual subtree contract')
    x,m,p=(arrays[k] for k in ('x','m','p'));nn=p.argmax(1)
    require(x.ndim==2 and m.shape==p.shape==(len(x),5) and np.isfinite(p).all() and
        np.all(p[m==0]==0) and np.allclose(p.sum(1),1.,atol=1e-6,rtol=0),'need complete masked NN targets')
    depth=min(context_depth,max(0,remaining-1));features=domain_table()['features']
    available=[i for i,d in enumerate(features) if not d['constant'] and _support_cost([i])<=3]
    records=[];terminal=[];model=None
    if depth>0 and len(x)>=2*MINIMUM_CONTEXT_FRAMES and len(np.unique(nn))>1:
        model=DecisionTreeClassifier(criterion='entropy',max_depth=depth,min_samples_leaf=MINIMUM_CONTEXT_FRAMES,random_state=0)
        def fit():
            model.fit(x[:,available],nn)
            return OptimizeResult(success=True,status=0,nit=int(model.tree_.node_count),fun=float(np.mean(model.predict(x[:,available])!=nn)))
        budget.solve('node',dict(method=VERSION,root=list(root),context_depth=depth,training_frames=len(x),
            target='complete_masked_NN_argmax_multiclass_context',min_samples_leaf=MINIMUM_CONTEXT_FRAMES,
            purity_required_before_context_split=False,random_state_is_fixed_engineering_control=True),fit)
    def finish(rows,legal,route,slots):
        counts=np.bincount(nn[rows],minlength=5);action=int(np.argmax(counts)) if len(rows) else 4
        record=dict(route=list(route),frames=len(rows),NN_argmax_counts=counts.tolist(),movement_mode=action,
            inherited_actions=sorted(legal),physical_guard_added=bool(len(rows) and action not in legal and slots>0))
        terminal.append(record)
        if not record['physical_guard_added']:return None
        w,b=guards.old.predicate([action]);require(action in guards.evidence(w,b)['right_proven'],'terminal physical guard unproved')
        return Node(w,b,None,None)
    def convert(node,rows,legal,route,slots):
        budget.check()
        if model is None or model.tree_.children_left[node]<0 or slots<=1:
            return finish(rows,legal,route,slots)
        tree=model.tree_;feature=available[int(tree.feature[node])]
        w=np.zeros(x.shape[1],np.float32);w[feature]=1.;bias=float(np.float32(-tree.threshold[node]))
        # Recompute the stored float32 hard route, never trust CART's float64
        # thresholds or leaf assignment as the deployed surrogate route.
        go=split_scores(x[rows],w[None],np.array([bias],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
        proof=guards.evidence(w,bias)
        records.append(dict(route=list(route),frames=len(rows),feature=feature,bias=bias,left_frames=int((~go).sum()),
            right_frames=int(go.sum()),NN_argmax_counts=np.bincount(nn[rows],minlength=5).tolist(),
            left_NN_argmax_counts=np.bincount(nn[rows[~go]],minlength=5).tolist(),
            right_NN_argmax_counts=np.bincount(nn[rows[go]],minlength=5).tolist()))
        return Node(tuple(map(float,w)),bias,
            convert(int(tree.children_left[node]),rows[~go],legal|set(proof['left_proven']),route+(0,),slots-1),
            convert(int(tree.children_right[node]),rows[go],legal|set(proof['right_proven']),route+(1,),slots-1))
    subtree=convert(0,np.arange(len(x)),set(known),(),remaining)
    structure,paths=encode(subtree);_,info=structure_info(plain(structure),256)
    require(info['depth']<=remaining and len(paths)<=2**min(remaining,context_depth+1),'context subtree exceeds prospective bound')
    return subtree,plain(dict(version=VERSION,root=list(root),context_depth=context_depth,effective_context_depth=depth,
        minimum_context_frames=MINIMUM_CONTEXT_FRAMES,training_frames=len(x),records=records,terminals=terminal,
        no_execution_actions=True,all_leaf_outputs_must_be_fitted_to_complete_NN=True,leaf_probabilities_are_not_CART_class_votes=True))
