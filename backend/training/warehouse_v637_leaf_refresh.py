"""Distribution-aware sparse leaf support search; no execution controller.

A dense GROUP-regularized multinomial relaxation proposes public sources.
Only <=3-source leaves with independently checkable path proofs may survive.
Already reliable explanation leaves and the original protected core stay exact.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .warehouse_v626_domain import domain_table,transform
from .warehouse_v626_program import _trial,_rank,_proxy,_support_cost,fit_leaf,calibrate,evaluate_leaf
from .warehouse_v628_certificate import region
from .warehouse_v622_program import plain,digest
from . import warehouse_v635_program as language,warehouse_v635_search as choices
from scripts.run_warehouse_v626 import internal_split
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-group-relaxation-certified-sparse-leaf-refresh.v6.37'
TRIALS=24


def dense_objective(flat,u,m,p):
    """Full masked target distribution CE + smooth group penalty, analytic grad."""
    w=flat.reshape(5,u.shape[1]+1);z=w[:,0]+np.einsum('ni,ai->na',u,w[:,1:],optimize=False)
    normal=logsumexp(np.where(m>0,z,-np.inf),axis=1)
    q=np.exp(np.where(m>0,z,-np.inf)-normal[:,None]);res=q*p.sum(1)[:,None]-p
    g=np.empty_like(w);g[:,0]=res.mean(0);g[:,1:]=np.einsum('na,ni->ai',res,u,optimize=False)/len(u)
    norm=np.sqrt(np.sum(w[:,1:]**2,axis=0)+1e-8)
    value=np.mean(p.sum(1)*normal-np.sum(p*z,axis=1))+.003*np.sum(norm-1e-4)+1e-4*np.sum(w**2)
    g[:,1:]+=.003*w[:,1:]/norm;g+=2e-4*w
    require(np.isfinite(value) and np.isfinite(g).all(),'nonfinite dense distribution proposal')
    return float(value),g.ravel()


def dense_sources(x,m,p,budget,metadata):
    ids=[i for i,d in enumerate(domain_table()['features']) if not d['constant']]
    u=transform(x,ids).astype(float);vary=np.ptp(u,axis=0)>1e-8;ids=[i for i,yes in zip(ids,vary,strict=True) if yes];u=u[:,vary]
    if not ids:return [],dict(called=False,reason='no_varying_public_sources')
    start=np.zeros((5,len(ids)+1));start[:,0]=np.log((p.sum(0)+.2)/(m.sum(0)+1.));start[:,0]-=start[:,0].mean()
    res=budget.solve('node',dict(method=VERSION,proposal_only=True,complete_masked_NN_targets=True,
        penalty_group=.003,penalty_L2=.0001,maximum_iterations=120,source_indices=ids,**metadata),
        lambda:minimize(dense_objective,start.ravel(),args=(u,m,p),jac=True,method='L-BFGS-B',
            bounds=[(-32.,32.)]*start.size,options=dict(maxiter=120,ftol=1e-8)))
    require(np.isfinite(res.x).all(),'nonfinite dense support proposal coefficients')
    weights=res.x.reshape(start.shape)[:,1:];strength=np.linalg.norm(weights,axis=0)
    ranking=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-strength[j],ids[j]))]
    return ranking,plain(dict(called=True,solver_success=bool(res.success),iterations=int(res.nit),objective=float(res.fun),
        source_strength=[dict(feature=ids[j],norm=float(strength[j])) for j in sorted(range(len(ids)),key=lambda j:(-strength[j],ids[j]))],
        dense_output_never_stored_as_program=True,not_a_global_support_optimality_claim=True))


def select_leaf(incumbent,train,selection,path,legal,budget,leaf):
    x,m,p=train;best=deepcopy(incumbent);seen=set();trials=[];candidates=[best]
    def key(model):return (*_proxy(model,selection),-_support_cost(model['support']),tuple(-i for i in model['support']),-model['witness'])
    initial=key(best);dense,proposal=dense_sources(x,m,p,budget,dict(leaf=leaf,training_frames=len(x)))
    def attempt(support,witness):
        nonlocal best
        support=tuple(sorted(support));identity=(support,int(witness))
        if identity in seen or len(seen)>=TRIALS or _support_cost(support)>3 or len(set(support))!=len(support):return False
        require(witness in legal,'unknown physical witness');seen.add(identity)
        fitted=_trial(x,m,p,support,int(witness),path,legal,True,budget,initial=incumbent)
        record=dict(support=list(support),witness=int(witness),certified=fitted is not None,proxy=None if fitted is None else list(key(fitted)))
        if fitted is not None:
            candidates.append(fitted)
            if key(fitted)>key(best):best=fitted;record['accepted']=True
            else:record['accepted']=False
        trials.append(record);return True
    attempt(incumbent['support'],incumbent['witness'])
    alt=max(legal,key=lambda a:(float(p[:,a].sum()),-a))
    if alt!=incumbent['witness']:attempt(incumbent['support'],alt)
    attempt([],incumbent['witness'])
    prefix=[]
    for i in dense:
        if _support_cost(prefix+[i])<=3:
            prefix.append(i);attempt(prefix,incumbent['witness'])
        if _support_cost(prefix)==3:break
    residual=_rank(x,p,m,best,best['support'])
    pool=list(dict.fromkeys([*dense[:4],*residual[:4]]))
    for i in pool[:4]:attempt([i],incumbent['witness'])
    # A width-two conditional support beam; all candidates fit full five-action
    # probabilities, not teacher/handwritten task labels or dense class votes.
    for size in (2,3):
        beam=sorted((c for c in candidates if len(c['support'])==size-1),key=key,reverse=True)[:2]
        for parent in beam:
            rank=list(dict.fromkeys([*dense[:6],*_rank(x,p,m,parent,parent['support'])[:4]]))
            added=0
            for i in rank:
                if i in parent['support']:continue
                if attempt([*parent['support'],i],parent['witness']):added+=1
                if added==3:break
    while len(seen)<TRIALS:
        budget.check();support=best['support'];rank=list(dict.fromkeys([*dense[:10],*_rank(x,p,m,best,support)[:8]]))
        proposed=[]
        if _support_cost(support)<3:proposed.extend((sorted([*support,i]),best['witness']) for i in rank if i not in support)
        for i in rank:
            if i in support:continue
            for j in support:proposed.append((sorted([k for k in support if k!=j]+[i]),best['witness']))
        proposal_next=next(((s,w) for s,w in proposed if (tuple(s),w) not in seen and _support_cost(s)<=3),None)
        if proposal_next is None:break
        attempt(*proposal_next)
    return best,plain(dict(leaf=leaf,dense_proposal=proposal,trials=trials,incumbent_key=list(initial),selected_key=list(key(best)),
        selected_original=digest(language.payload(best))==digest(language.payload(incumbent)),fit_frames=len(x),selection_frames=len(selection[0])))


def arrays(a,take):return tuple(a[k][take] for k in ('x','m','p'))


def leaf_option(program,leaf,model):
    return dict(structure=plain(dict(root_ref=-1,split_weights=np.zeros((0,len(domain_table()['features'])),np.float32),
        split_bias=[],left_refs=[],right_refs=[],leaf_count=1,feature_dim=len(domain_table()['features']))),models=[model],
        paths=[dict(path=region(program.structure,leaf,256),legal=sorted(program.certificates()[leaf]))])


def run(source,outer,lanes,budget):
    source.validate();out=deepcopy(source);out.report=None;out.feedback_leaves=frozenset();out.explanation_leaves=frozenset()
    core=set(language.protection(source)['protected_descendants']);legal=source.certificates();a=outer['fit']
    tr,se=internal_split(a['rows']);routes={s:source.leaf_indices(p['x']) for s,p in outer.items()}
    repeated=choices.metrics.risk_subsets(lanes['eligibility']['rows']);records=[]
    _,paths=choices.encode(choices.decode(source.structure))
    for i,incumbent in enumerate(source.leaves):
        budget.check();parts,masks=choices.root_parts(source,paths[i],lanes)
        repeat=[r[masks['eligibility']] for r in repeated]
        old_score=choices.option_score(leaf_option(source,i,incumbent),parts,repeat)
        frozen=i in core or old_score['children'][0]['reliable_selected']['explanation']
        take=routes['fit']==i;train,selection=take&tr,take&se
        record=dict(leaf=i,original_core=i in core,frozen_reliable_E_or_core=bool(frozen),old_score=old_score,selected='old')
        if frozen:record['status']='exact_payload_preserved'
        elif not train.any() or not selection.any():record['status']='no_internal_fit_exposure_keep_old'
        else:
            path=region(source.structure,i,256)
            selected,search=select_leaf(incumbent,arrays(a,train),arrays(a,selection),path,legal[i],budget,i);record['search']=search
            record['status']='support_search_completed'
            if not search['selected_original']:
                fitted=fit_leaf(*arrays(a,take),arm='C3',path=path,legal=legal[i],budget=budget,initial=selected,final=True)
                cal=outer['calibration'];ct=routes['calibration']==i
                fitted=calibrate(fitted,*arrays(cal,ct),arm='C3',path=path,legal=legal[i],budget=budget)
                score=choices.option_score(leaf_option(source,i,fitted),parts,repeat)
                record['candidate_model']=fitted;record['candidate_score']=score
                # Additive lexicographic finite selection. Accepted old E never
                # changes; each remaining leaf retains its old safe option.
                def option_key(s):return tuple(s['score'])
                if option_key(score)>option_key(old_score):out.leaves[i]=fitted;record['selected']='refreshed'
        records.append(record)
        budget.checkpoint(dict(stage='leaf_refresh',completed_leaves=len(records),total_leaves=len(source.leaves),
            changed=sum(r['selected']=='refreshed' for r in records)))
    out.validate();require(digest(out.structure)==digest(source.structure),'leaf experiment changed topology')
    for r in records:
        if r['frozen_reliable_E_or_core']:require(language.payload(out.leaves[r['leaf']])==language.payload(source.leaves[r['leaf']]),'frozen reliable/core payload changed')
    totals={k:sum((r.get('candidate_score') if r['selected']=='refreshed' else r['old_score'])['totals'][k] for r in records) for k in ('E','F','expected_F')}
    n=len(lanes['validation']['x']);reliable=totals['E']/n>=.45 and totals['F']/n>=.45 and totals['expected_F']/n>=.40
    language.qualify(out,lanes)
    return out,plain(dict(version=VERSION,records=records,reliable_counts=totals,reliable_selection_passed=reliable,
        fit_only_metrics=language.measure(out,lanes['validation']),original_core=sorted(core),source_structure_sha256=digest(source.structure),
        source_protection_sha256=digest(source.protection),structure_unchanged=True,NN_updates=0,training_authority=False,pilot_opened=False))
