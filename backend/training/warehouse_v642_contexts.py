"""Public sparse logit-margin directions and soft-NN context trees.

Directions are learned surrogate coefficients, not instructions. The complete
frozen NN supplies soft targets. Explicit terminal physical guards and every
empty branch count against the same depth/leaf bounds.
"""
import numpy as np
from scipy.optimize import OptimizeResult
from .warehouse_v642_subtrees import (Node,guards,plain,encode,decode,region,structure_info,_support_cost)
from .warehouse_v626_domain import domain_table
from .warehouse_v622_program import digest
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-borrowed-margin-soft-NN-contexts.v6.42'
CONTEXT_DEPTHS=(2,3)
MINIMUM_LEAF_FRAMES=160


def logit_directions(model):
    """Public affine directions; storage rounding is rescored, not called exact."""
    domain=domain_table()['features'];support=model['support'];w=np.asarray(model['weights'],float);b=np.asarray(model['bias'],float);out={}
    for a in range(5):
        for c in range(a+1,5):
            values=w[a]-w[c];raw=np.zeros(len(domain),float);bias=float(b[a]-b[c])
            for i,value in zip(support,values,strict=True):
                raw[i]=value*domain[i]['scale'];bias-=value*domain[i]['scale']*domain[i]['center']
            if np.max(np.abs(raw))<1e-10:continue
            scale=max(1.,float(np.abs(raw).max()),abs(bias));raw=(raw/scale).astype(np.float32);bias=float(np.float32(bias/scale))
            used=np.flatnonzero(raw)
            if raw[used[0]]<0:raw=-raw;bias=-bias
            require(_support_cost(used)<=3,'borrowed direction exceeds public source budget')
            key=digest(dict(w=raw.tolist(),b=bias));out.setdefault(key,dict(weight=raw.tolist(),bias=bias,actions=[a,c],model_payload_sha256=digest({k:v for k,v in model.items() if k!='certificate'})))
    return [out[k] for k in sorted(out)]


def borrowed_bank(source,root,training_x):
    _,paths=encode(decode(source.structure));ids=[i for i,p in enumerate(paths) if p[:len(root)]==root]
    count=np.bincount(source.leaf_indices(training_x),minlength=len(paths));chosen=sorted(ids,key=lambda i:(-count[i],i))[:8];out={}
    for i in chosen:
        for direction in logit_directions(source.leaves[i]):
            key=digest(dict(w=direction['weight'],b=direction['bias']));out.setdefault(key,dict(direction,source_leaf=i,source_fit_frames=int(count[i])))
    require(len(out)<=80,'borrowed bank exceeds fixed exposure-ranked eight-model budget')
    return [out[k] for k in sorted(out)]


def entropy(mass,total):
    mass=np.asarray(mass,float);total=np.asarray(total,float)
    return -np.sum(mass*np.log(np.maximum(mass/np.expand_dims(total,-1),1e-300)),axis=-1)


def learn_tree(arrays,known,remaining,budget,*,context_depth,root,bank):
    require(context_depth in CONTEXT_DEPTHS and remaining>=1,'invalid context recipe')
    x,m,p=(arrays[k] for k in ('x','m','p'))
    require(len(x)>0 and p.shape==m.shape==(len(x),5) and np.isfinite(p).all() and np.all(p[m==0]==0) and
        np.allclose(p.sum(1),1.,atol=1e-6,rtol=0),'context requires complete masked NN targets')
    target=p.astype(float);target/=target.sum(1,keepdims=True);nn=target.argmax(1);records=[];terminals=[];attempts=[];domain=domain_table()['features']
    axes=[]
    for i,d in enumerate(domain):
        if not d['constant'] and _support_cost([i])<=3:
            w=np.zeros(x.shape[1],np.float32);w[i]=1.;axes.append(dict(weight=w.tolist(),bias=0.,origin='public_axis'))
    directions=[*axes,*[dict(d,origin='borrowed_leaf_logit_margin') for d in bank]]
    require(len(directions)<=len(domain)+80,'context direction bound exceeded')
    def finish(rows,legal,path,slots):
        counts=np.bincount(nn[rows],minlength=5);action=int(counts.argmax()) if len(rows) else 4
        guarded=bool(len(rows) and action not in legal and slots>0)
        terminals.append(dict(route=list(path),frames=len(rows),NN_argmax_counts=counts.tolist(),mode=action,
            known_actions=sorted(legal),physical_guard_added=guarded))
        if not guarded:return None
        w,b=guards.old.predicate([action]);return Node(w,b,None,None)
    def grow(rows,levels,slots,legal,path):
        budget.check()
        if levels==0 or slots<=1 or len(rows)<2*MINIMUM_LEAF_FRAMES:return finish(rows,legal,path,slots)
        before=float(entropy(target[rows].sum(0),len(rows)));best=None;evaluations=0;threshold_evaluations=0
        def search():
            nonlocal best,evaluations,threshold_evaluations
            for direction in directions:
                budget.check();weights=np.asarray(direction['weight'],np.float32);bias=float(np.float32(direction['bias']))
                scores=split_scores(x[rows],weights[None],np.array([bias],np.float32))[:,0]
                order=np.argsort(scores,kind='stable');s=scores[order];count=np.arange(1,len(rows))
                valid=np.flatnonzero((s[:-1]<s[1:])&(count>=MINIMUM_LEAF_FRAMES)&(len(rows)-count>=MINIMUM_LEAF_FRAMES))
                evaluations+=1
                if not len(valid):continue
                sums=np.cumsum(target[rows[order]],axis=0);total=sums[-1]
                cost=entropy(sums[valid],valid+1)+entropy(total-sums[valid],len(rows)-valid-1)
                for j in sorted(range(len(valid)),key=lambda j:(float(cost[j]),int(valid[j])))[:4]:
                    threshold_evaluations+=1
                    k=int(valid[j]);cut=(float(s[k])+float(s[k+1]))/2;bb=float(np.float32(bias-cut))
                    go=split_scores(x[rows],weights[None],np.array([bb],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
                    if min(int(go.sum()),int((~go).sum()))<MINIMUM_LEAF_FRAMES:continue
                    actual=float(entropy(target[rows[go]].sum(0),int(go.sum()))+entropy(target[rows[~go]].sum(0),int((~go).sum())))
                    key=(actual,_support_cost(np.flatnonzero(weights)),digest(dict(w=weights.tolist(),b=bb)))
                    if before-actual<=1e-8:continue
                    if best is None or key<best[0]:best=(key,weights,bb,go,direction)
            return OptimizeResult(success=best is not None,fun=before if best is None else best[0][0],nit=evaluations)
        budget.solve('node',dict(method=VERSION,root=list(root),route=list(path),training_frames=len(rows),
            target='complete_masked_NN_soft_probability_entropy',maximum_directions=len(directions),maximum_cuts_per_direction=4,
            minimum_child_frames=MINIMUM_LEAF_FRAMES,context_depth=context_depth),search)
        attempts.append(dict(route=list(path),frames=len(rows),directions_evaluated=evaluations,threshold_evaluations=threshold_evaluations,found_split=best is not None))
        if best is None:return finish(rows,legal,path,slots)
        key,w,b,go,d=best;proof=guards.evidence(w,b)
        records.append(dict(route=list(path),frames=len(rows),left_frames=int((~go).sum()),right_frames=int(go.sum()),
            direction_origin=d['origin'],direction_provenance=d,stored_weight=w.tolist(),stored_bias=b,
            entropy_before=before,entropy_after=key[0],directions_evaluated=evaluations))
        return Node(tuple(map(float,w)),b,
            grow(rows[~go],levels-1,slots-1,legal|set(proof['left_proven']),path+(0,)),
            grow(rows[go],levels-1,slots-1,legal|set(proof['right_proven']),path+(1,)))
    tree=grow(np.arange(len(x)),min(context_depth,remaining-1),remaining,set(known),())
    structure,paths=encode(tree);_,info=structure_info(plain(structure),256)
    require(info['depth']<=remaining and len(paths)<=2**min(remaining,context_depth+1),'new context exceeds depth/leaf accounting')
    return tree,plain(dict(version=VERSION,root=list(root),context_depth=context_depth,training_frames=len(x),bank=bank,
        records=records,terminals=terminals,attempts=attempts,complete_NN_soft_targets=True,source_actions_never_executed=True))
