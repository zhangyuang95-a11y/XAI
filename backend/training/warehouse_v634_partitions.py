"""High-precision coverage predicates, no execution actions or new leaf output.

The bounded v632 logistic direction fitter is reused unchanged. New training
threshold objectives target .98/.99 agreement mass instead of entropy alone.
Qualification/selection/final gates remain separate and unchanged.
"""
import numpy as np
from scipy.special import expit
from . import warehouse_v632_partitions as old
from .warehouse_v626_search import Node
from .warehouse_v630_partition_proposals import prunings
from .warehouse_v626_domain import domain_table,transform
from .warehouse_v626_program import _support_cost
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from .warehouse_v622_program import digest,plain
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-high-precision-coverage-predicates.v6.34'
PRECISIONS=(.98,.99)


def score_split(y,go,precision):
    weights=np.where(y==0,4.,1.)
    entropy=sum(float(old._entropy(np.sum(weights[t]*y[t]),np.sum(weights[t]))) for t in (go,~go))
    mass=sum(int(t.sum()) for t in (go,~go) if t.any() and float(y[t].mean())>=precision)
    return -mass,entropy


def thresholds(scores,labels,precision,minimum=40):
    """Four best fit-only cuts; exact stored routing is scored by the caller."""
    scores=np.asarray(scores,np.float32);labels=np.asarray(labels,float)
    require(scores.shape==labels.shape and np.isfinite(scores).all() and np.isin(labels,[0.,1.]).all(),
            'invalid precision-threshold training rows')
    order=np.argsort(scores,kind='stable');s,y=scores[order],labels[order];n=len(s)
    indices=np.flatnonzero((s[:-1]<s[1:])&(np.arange(n-1)+1>=minimum)&(n-np.arange(n-1)-1>=minimum))
    if not len(indices):return []
    count=indices+1;correct=np.cumsum(y);left=correct[indices];right=correct[-1]-left
    mass=count*(left/count>=precision)+(n-count)*(right/(n-count)>=precision)
    weights=np.where(y==0,4.,1.);w=np.cumsum(weights);positive=np.cumsum(weights*y)
    entropy=old._entropy(positive[indices],w[indices])+old._entropy(positive[-1]-positive[indices],w[-1]-w[indices])
    ranked=sorted(range(len(indices)),key=lambda j:(-int(mass[j]),float(entropy[j]),int(indices[j])))
    answer=[]
    for j in ranked:
        k=int(indices[j]);cut=float(np.float32((float(s[k])+float(s[k+1]))/2))
        if cut not in answer:answer.append(cut)
        if len(answer)==4:break
    return answer


def propose(source,arrays,leaf,budget,precision):
    require(precision in PRECISIONS,'unregistered precision training target')
    path=old.region(source.structure,leaf,256);depth=min(3,9-len(path['path']))
    take=source.leaf_indices(arrays['x'])==leaf
    x,m,p=(arrays[k][take] for k in ('x','m','p'))
    require(p.shape==m.shape==(len(x),5) and np.isfinite(p).all() and np.all(p[m==0]==0)
            and np.allclose(p.sum(1),1.,atol=1e-6,rtol=0),'need complete masked NN distribution')
    _,_,q=source.distributions(x,m);y=(q.argmax(1)==p.argmax(1)).astype(float)
    d=domain_table()['features'];available=[i for i,a in enumerate(d) if not a['constant']]
    u=transform(x,available).astype(float);records=[]
    def grow(rows,remaining,route):
        budget.check()
        if remaining==0 or len(rows)<80 or np.all(y[rows]==y[rows][0]):return None
        local=u[rows];labels=y[rows];valid=np.flatnonzero(np.ptp(local,axis=0)>1e-8);best=None;seen=set()
        def consider(weights,bias=0.,origin='axis'):
            nonlocal best
            linear=split_scores(x[rows],weights[None],np.zeros(1,np.float32))[:,0]
            scores=linear+float(np.float32(bias))
            for cut in thresholds(scores,labels,precision):
                b=np.float32(bias-cut)
                # Exactly the shared router: identical float64 dot reduction,
                # followed by addition of the new stored float32 bias.
                go=linear+float(b)>HARD_SPLIT_TIE_EPSILON
                if min(go.sum(),(~go).sum())<40:continue
                key=(*score_split(labels,go,precision),int(np.count_nonzero(weights)),digest(dict(w=weights.tolist(),b=float(b))))
                if best is None or key<best[0]:best=(key,weights,b,go,origin)
        for j in valid:
            weights=np.zeros(x.shape[1],np.float32);weights[available[j]]=1.;consider(weights)
        def fit(support):
            ids=[available[j] for j in support];identity=tuple(sorted(support))
            if identity in seen or not 0<_support_cost(ids)<=3:return None
            seen.add(identity)
            theta,_=old._fit(local,labels,support,budget,dict(source_leaf=leaf,route=list(route),
                training_frames=len(rows),source_indices=ids,proposal_objective=VERSION,precision_target=precision))
            weights=np.zeros(x.shape[1],np.float32);bias=float(theta[0])
            for k,j in enumerate(ids):
                weights[j]=np.float32(theta[k+1]*d[j]['scale']);bias-=float(theta[k+1])*d[j]['scale']*d[j]['center']
            consider(weights,np.float32(bias),'sparse_logistic');return theta
        support=[];theta=None
        while len(support)<3:
            pred=np.full(len(rows),np.average(labels,weights=np.where(labels==0,4.,1.))) if theta is None else expit(theta[0]+old._mv(local[:,support],theta[1:]))
            residual=np.where(labels==0,4.,1.)*(labels-pred)
            corr=np.abs(old._mv(local.T,residual))/np.maximum(np.linalg.norm(local,axis=0),1e-10)
            choices=[j for j in valid if j not in support and _support_cost([available[k] for k in support+[j]])<=3]
            if not choices:break
            support.append(min(choices,key=lambda j:(-corr[j],available[j])));theta=fit(support)
            if theta is None:break
        if theta is not None:
            residual=np.where(labels==0,4.,1.)*(labels-expit(theta[0]+old._mv(local[:,support],theta[1:])))
            corr=np.abs(old._mv(local.T,residual))/np.maximum(np.linalg.norm(local,axis=0),1e-10);exchanges=[]
            for remove in support:
                for new in valid:
                    candidate=sorted([j for j in support if j!=remove]+[int(new)])
                    if len(candidate)!=len(set(candidate)) or tuple(candidate) in seen or _support_cost([available[k] for k in candidate])>3:continue
                    exchanges.append((-float(corr[new]),available[remove],available[new],candidate))
            calls=0
            for *_,candidate in sorted(exchanges):
                if tuple(candidate) in seen:continue
                fit(candidate);calls+=1
                if calls==4:break
        if best is None:return None
        key,weights,b,go,origin=best
        records.append(dict(route=list(route),frames=len(rows),origin=origin,source_features=int(np.count_nonzero(weights)),
            optimizer_supports=len(seen),fit_high_precision_mass=-key[0],weighted_entropy=key[1],precision_target=precision))
        return Node(tuple(map(float,weights)),float(b),grow(rows[~go],remaining-1,route+(0,)),grow(rows[go],remaining-1,route+(1,)))
    tree=grow(np.arange(len(x)),depth,()) if len(x) else None
    return plain(dict(version=VERSION,source_leaf=leaf,fit_frames=len(x),precision_target=precision,records=records,
        proposals=prunings(tree),classifier_outputs_discarded=True,execution_actions_produced=False))
