"""Budgeted public confidence predicates and finite additive leaf allocation.

Components only: the caller still owns registered data partitions, search
authority, exact hard-route scoring/certification and independent admission.
The learned classifier's predictions never select environment actions or
replace the surrogate's full five-action distribution.
"""
from itertools import product
from math import isfinite
from hashlib import sha256
import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.tree import DecisionTreeClassifier
from .warehouse_v626_search import Node,encode
from .warehouse_v628_certificate import region
from .warehouse_v622_program import digest,plain
from .warehouse_v6_actor import FEATURE_NAMES
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-public-confidence-partition-components.v1'


def prunings(tree):
    """Only stop here or retain both child subtrees; no subtree promotion."""
    def depth(node):return 0 if node is None else 1+max(depth(node.left),depth(node.right))
    require(depth(tree)<=3,'confidence pruning template exceeds fixed depth three')
    def enumerate_nodes(node):
        if node is None:return [None]
        return [None]+[Node(node.weight,node.bias,left,right)
            for left,right in product(enumerate_nodes(node.left),enumerate_nodes(node.right))]
    options=enumerate_nodes(tree)
    seen={}
    for node in options:
        value=plain(encode(node)[0]);seen.setdefault(digest(value),value)
    return [seen[key] for key in sorted(seen)]


def propose(program,arrays,source_leaf,budget):
    """One budgeted CART fit, then all its depth<=3 prunings. No scoring.

    The caller must supply FIT-ONLY physical frames from the registered
    source. Outputs only predicates, not leaf probabilities or qualification.
    """
    program.validate()
    require(type(source_leaf) is int and 0<=source_leaf<len(program.leaves),'invalid confidence source leaf')
    path=region(program.structure,source_leaf,program.maximum_leaves)
    depth=min(3,9-len(path['path']));take=program.leaf_indices(arrays['x'])==source_leaf
    budget.check()
    if depth<1 or int(take.sum())<80:
        return dict(version=VERSION,source_leaf=source_leaf,fit_called=False,fit_frames=int(take.sum()),
            proposals=[plain(encode(None)[0])],no_qualification_or_action_output=True)
    x=np.asarray(arrays['x'][take],np.float32);m=np.asarray(arrays['m'][take],np.float32)
    target=np.asarray(arrays['p'][take],np.float32)
    require(x.ndim==2 and x.shape[1]==len(FEATURE_NAMES) and m.shape==target.shape==(len(x),5)
        and np.isfinite(x).all() and np.isfinite(target).all() and np.all(target>=0)
        and np.allclose(target.sum(1),1.,atol=1e-6,rtol=0) and np.all(target[m==0]==0),
        'confidence training needs complete masked NN probabilities')
    _,_,q=program.distributions(x,m);labels=(q.argmax(1)==target.argmax(1)).astype(np.int64)
    model=DecisionTreeClassifier(criterion='entropy',max_depth=depth,min_samples_leaf=40,
        class_weight={0:4.,1:1.},random_state=0)
    def fit():
        model.fit(x,labels)
        return OptimizeResult(success=True,status=0,nit=int(model.tree_.node_count),
            fun=float(np.mean(model.predict(x)!=labels)))
    budget.solve('node',dict(method=VERSION,source_leaf=source_leaf,target='same_surrogate_vs_complete_NN_argmax_agreement',
        maximum_depth=depth,min_samples_leaf=40,error_class_weight=4.,random_state_is_fixed_engineering_control=True),fit)
    tree=model.tree_
    def convert(i):
        if int(tree.children_left[i])<0:return None
        feature=int(tree.feature[i]);weights=np.zeros(len(FEATURE_NAMES),np.float32);weights[feature]=1.
        return Node(tuple(map(float,weights)),float(np.float32(-tree.threshold[i])),
            convert(int(tree.children_left[i])),convert(int(tree.children_right[i])))
    proposed=prunings(convert(0))
    return dict(version=VERSION,source_leaf=source_leaf,fit_called=True,fit_frames=len(x),
        fit_correct_count=int(labels.sum()),proposals=proposed,
        source_target_sha256=digest(dict(shape=list(target.shape),bytes_sha256=sha256(target.tobytes()).hexdigest())),
        classifier_outputs_discarded=True,no_qualification_or_action_output=True,
        final_scoring_must_use_stored_float32_shared_hard_router=True)


def allocate(groups,maximum_leaves):
    """Exact lexicographic additive optimum for this supplied finite table.

    Each option supplies total local leaf count and a fixed score vector.
    This does not optimize unknown predicates, nonlinear global gates, or
    max/min across scenarios. Those remain the registered caller's job.
    """
    require(type(maximum_leaves) is int and 1<=maximum_leaves<=256 and groups,'invalid finite leaf allocation budget')
    keys=sorted(groups);dimension=None;clean={}
    for key in keys:
        require(type(key) is str and key and groups[key],'empty or unsafe allocation group')
        options=[];identities=set()
        for option in groups[key]:
            require(set(option)=={'id','leaves','score'} and type(option['id']) is str and option['id']
                and option['id'] not in identities and type(option['leaves']) is int and 1<=option['leaves']<=256,
                'invalid or duplicate finite option')
            score=tuple(option['score']);dimension=len(score) if dimension is None else dimension
            require(score and len(score)==dimension and all(type(v) in (int,float) and isfinite(v) for v in score),
                'inconsistent or nonfinite additive allocation score')
            identities.add(option['id']);options.append(dict(id=option['id'],leaves=option['leaves'],score=score))
        clean[key]=sorted(options,key=lambda o:o['id'])
    # Cost -> (score, ordered chosen option IDs). Retain canonical ties.
    states={0:((0,)*dimension,())}
    for key in keys:
        next_states={}
        for cost,(score,chosen) in states.items():
            for option in clean[key]:
                n=cost+option['leaves']
                if n>maximum_leaves:continue
                value=(tuple(a+b for a,b in zip(score,option['score'],strict=True)),chosen+(option['id'],))
                old=next_states.get(n)
                if old is None or value[0]>old[0] or value[0]==old[0] and value[1]<old[1]:next_states[n]=value
        require(next_states,'finite options cannot fit within the declared leaf budget');states=next_states
    cost,(score,chosen)=min(states.items(),key=lambda item:(tuple(-v for v in item[1][0]),item[0],item[1][1]))
    return dict(version=VERSION,leaves=cost,score=list(score),selected=dict(zip(keys,chosen,strict=True)),
        table_sha256=digest(groups),maximum_leaves=maximum_leaves,
        optimality_scope='only_supplied_fixed_options_and_additive_scores',
        unknown_predicate_or_nonlinear_gate_optimality_gap='unknown',training_authority=False)
