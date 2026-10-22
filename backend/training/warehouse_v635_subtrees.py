"""Learn non-executing, action-qualified subtrees inside unprotected roots.

Condition labels come exclusively from the complete masked NN distribution.
Leaf outputs fit that full distribution; no action anchors are added to the
Actor or applied after its forward pass. NN execution is never called here.
"""
from copy import deepcopy
import numpy as np
from scipy.special import expit
from . import warehouse_v635_guards as guards,warehouse_v635_program as language
from . import warehouse_v632_partitions as directions
from .warehouse_v634_partitions import thresholds
from .warehouse_v626_search import Node,decode,encode,replace
from .warehouse_v626_domain import domain_table,transform
from .warehouse_v626_program import fit_leaf,calibrate,evaluate_leaf,_support_cost
from .warehouse_v628_certificate import region,structure_info
from .warehouse_v622_program import digest,plain,_route,_eligible
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from experiments.warehouse.v626_protocol import require
from scripts.run_warehouse_v626 import internal_split

VERSION='warehouse-learned-action-qualified-unprotected-subtrees.v6.35'
PRECISIONS=(.98,.99)


def learn_tree(arrays,known,remaining,budget,*,precision,root):
    """A counted physical mask can share a node with <=2 learned sources."""
    require(precision in PRECISIONS and remaining>=0,'unknown learned subtree contract')
    x,m,p=(arrays[k] for k in ('x','m','p'));nn=p.argmax(1);domain=domain_table()['features'];records=[]
    available=[i for i,d in enumerate(domain) if not d['constant'] and i not in guards.MASK_INDICES]
    u=transform(x,available).astype(float)
    def grow(rows,depth,legal,path):
        if depth==0 or len(rows)<80:return None
        if np.all(nn[rows]==nn[rows][0]) and int(nn[rows][0]) in legal:return None
        best=None
        for action in range(4):
            budget.check();limit=3 if action in legal else 2
            eligible=rows[m[rows,action]==1];labels=(nn[eligible]==action).astype(float)
            if len(eligible)<40 or not labels.any():continue
            local=u[eligible];vary=np.flatnonzero(np.ptp(local,axis=0)>1e-8);seen=set();linear_cache={}
            def consider(weights,bias):
                nonlocal best
                # Both orientations are evaluated; only the RIGHT branch is
                # the proposed action-qualified region.
                for sign in (1.,-1.):
                    ww=(np.asarray(weights,np.float32)*np.float32(sign)).astype(np.float32);bb=float(np.float32(bias*sign))
                    if action not in legal:ww,bb=guards.fuse(action,ww,bb)
                    else:ww=tuple(map(float,ww))
                    key_weights=np.asarray(ww,np.float32).tobytes()
                    if key_weights not in linear_cache:
                        linear_cache[key_weights]=split_scores(x[rows],np.asarray(ww,np.float32)[None],np.zeros(1,np.float32))[:,0]
                    go=linear_cache[key_weights]+float(np.float32(bb))>HARD_SPLIT_TIE_EPSILON
                    n=int(go.sum());correct=(nn[rows][go]==action)
                    if n<40 or float(correct.mean())<precision:continue
                    require(np.all(m[rows[go],action]==1),'learned right branch contains a physical zero bit')
                    expected=float(p[rows[go],action].sum())
                    key=(-n,-expected,len(np.flatnonzero(ww)),digest(dict(w=ww,b=bb)),action)
                    if best is None or key<best[0]:best=(key,ww,bb,go,action,len(seen))
            if labels.all():
                consider(np.zeros(x.shape[1],np.float32),1.)
            else:
                def consider_direction(w,bias):
                    scores=split_scores(x[eligible],np.asarray(w,np.float32)[None],np.array([bias],np.float32))[:,0]
                    for cut in thresholds(scores,labels,precision):consider(w,float(np.float32(bias-cut)))
                for j in vary:
                    i=available[j]
                    if _support_cost([i])>limit:continue
                    w=np.zeros(x.shape[1],np.float32);w[i]=1.;consider_direction(w,0.)
                def fit(support):
                    ids=[available[j] for j in support];identity=tuple(sorted(support))
                    if identity in seen or _support_cost(ids)>limit:return None
                    seen.add(identity)
                    theta,_=directions._fit(local,labels,support,budget,dict(root=list(root),route=list(path),
                        action=action,label_target='complete_masked_NN_argmax_is_action',source_indices=ids,training_frames=len(eligible),
                        proposal_objective=VERSION,precision_target=precision))
                    w=np.zeros(x.shape[1],np.float32);bias=float(theta[0])
                    for k,i in enumerate(ids):
                        w[i]=np.float32(theta[k+1]*domain[i]['scale']);bias-=float(theta[k+1])*domain[i]['scale']*domain[i]['center']
                    consider_direction(w,float(np.float32(bias)));return theta
                support=[];theta=None
                while _support_cost([available[j] for j in support])<limit:
                    pred=np.full(len(eligible),np.average(labels,weights=np.where(labels==0,4.,1.))) if theta is None else expit(theta[0]+directions._mv(local[:,support],theta[1:]))
                    residual=np.where(labels==0,4.,1.)*(labels-pred)
                    corr=np.abs(directions._mv(local.T,residual))/np.maximum(np.linalg.norm(local,axis=0),1e-10)
                    choices=[j for j in vary if j not in support and _support_cost([available[k] for k in support+[j]])<=limit]
                    if not choices:break
                    support.append(min(choices,key=lambda j:(-corr[j],available[j])));theta=fit(support)
                    if theta is None:break
                if theta is not None:
                    residual=np.where(labels==0,4.,1.)*(labels-expit(theta[0]+directions._mv(local[:,support],theta[1:])))
                    corr=np.abs(directions._mv(local.T,residual))/np.maximum(np.linalg.norm(local,axis=0),1e-10);swaps=[]
                    for remove in support:
                        for new in vary:
                            s=sorted([j for j in support if j!=remove]+[int(new)])
                            if len(s)!=len(set(s)) or tuple(s) in seen or _support_cost([available[j] for j in s])>limit:continue
                            swaps.append((-float(corr[new]),available[remove],available[new],s))
                    called=0
                    for *_,s in sorted(swaps):
                        if tuple(s) in seen:continue
                        fit(s);called+=1
                        if called==4:break
        if best is None:return None
        key,w,b,go,action,calls=best;proof=guards.evidence(w,b)
        require(action in legal or action in proof['right_proven'],'chosen action has no path legality proof')
        records.append(dict(route=list(path),action=action,frames=len(rows),right_frames=int(go.sum()),
            right_NN_agreement=float((nn[rows[go]]==action).mean()),precision_target=precision,
            chosen_predicate=dict(weight=list(w),bias=b),inherited_actions=sorted(legal)))
        # Every right child is a full-distribution leaf. The left child can
        # learn another region, including another context for the same action.
        return Node(tuple(w),b,grow(rows[~go],depth-1,legal|set(proof['left_proven']),path+(0,)),None)
    tree=grow(np.arange(len(x)),min(4,remaining),set(known),())
    return tree,plain(dict(version=VERSION,root=list(root),precision_target=precision,training_frames=len(x),records=records,
        no_execution_actions=True,all_leaf_outputs_must_be_fitted_to_complete_NN=True))


def fit_option(source,root,subtree,parts,budget):
    """Fit only this root's leaves in the full path context, no eligibility."""
    tree=replace(decode(source.structure),root,subtree);structure,paths=encode(tree);structure=plain(structure)
    structure_info(structure,256);legal=guards.leaf_actions(structure,256)
    ids=[i for i,path in enumerate(paths) if path[:len(root)]==root]
    local_structure,local_paths=encode(subtree);local_structure=plain(local_structure)
    require([path[len(root):] for i,path in enumerate(paths) if i in ids]==local_paths,'subtree output order differs')
    tr,se=internal_split(parts['fit']['rows'])
    # Use the complete root prefix to take rows, not the old leaf partition.
    old_tree=decode(source.structure);_,old_paths=encode(old_tree)
    old_ids=[i for i,path in enumerate(old_paths) if path[:len(root)]==root]
    takes={s:np.isin(source.leaf_indices(a['x']),old_ids) for s,a in parts.items()}
    local_router,local_info=structure_info(local_structure,256)
    routes={s:_route(local_router,a['x'][takes[s]],local_info['depth']) for s,a in parts.items()}
    models=[];evidence=[]
    for local_id,global_id in enumerate(ids):
        budget.check();a=parts['fit'];within=np.flatnonzero(takes['fit']);selected=within[routes['fit']==local_id]
        training=selected[tr[selected]];selection=selected[se[selected]];path=region(structure,global_id,256)
        chosen=fit_leaf(a['x'][training],a['m'][training],a['p'][training],arm='C3',path=path,legal=legal[global_id],budget=budget,
            selection=(a['x'][selection],a['m'][selection],a['p'][selection]))
        fitted=fit_leaf(a['x'][selected],a['m'][selected],a['p'][selected],arm='C3',path=path,legal=legal[global_id],budget=budget,initial=chosen,final=True)
        cal=parts['calibration'];indices=np.flatnonzero(takes['calibration'])[routes['calibration']==local_id]
        fitted=calibrate(fitted,cal['x'][indices],cal['m'][indices],cal['p'][indices],arm='C3',path=path,legal=legal[global_id],budget=budget)
        models.append(fitted);evidence.append(dict(path=path,legal=sorted(legal[global_id]),train_frames=len(training),selection_frames=len(selection),fullfit_frames=len(selected),calibration_frames=len(indices)))
    return plain(dict(structure=local_structure,models=models,paths=evidence,root=list(root),training_only_no_eligibility=True))


def original_option(source,root):
    from .warehouse_v626_search import at
    subtree=at(decode(source.structure),root);structure,_=encode(subtree);_,paths=encode(decode(source.structure))
    ids=[i for i,path in enumerate(paths) if path[:len(root)]==root];legal=source.certificates()
    return plain(dict(structure=structure,models=[deepcopy(source.leaves[i]) for i in ids],root=list(root),
        paths=[dict(path=region(source.structure,i,256),legal=sorted(legal[i])) for i in ids],training_only_no_eligibility=True))


def local_predictions(option,x,m):
    t,info=structure_info(option['structure'],256);route=_route(t,x,info['depth'])
    z=np.empty((len(x),5),np.float32);q=np.empty_like(z)
    for i,model in enumerate(option['models']):
        take=route==i;z[take],q[take]=evaluate_leaf(model,x[take],m[take])
    return route,z,q


def assemble(source,options):
    tree=decode(source.structure)
    for root,option in sorted(options.items()):tree=replace(tree,root,decode(option['structure']))
    structure,paths=encode(tree);structure=plain(structure);structure_info(structure,256)
    out=language.Program(source.role,source.actor_role_sha256,structure,[],'C3',256,protection=deepcopy(source.protection),
        target_actor_role_sha256=source.target_actor_role_sha256)
    new_legal=out.certificates();_,old_paths=encode(decode(source.structure));old_legal=source.certificates()
    for i,path in enumerate(paths):
        root=next((root for root in options if path[:len(root)]==root),None)
        if root is None:
            j=old_paths.index(path);model=source.leaves[j];old_path=region(source.structure,j,256);known=old_legal[j]
        else:
            option=options[root];_,relative=encode(decode(option['structure']));j=relative.index(path[len(root):])
            model=option['models'][j];old_path=option['paths'][j]['path'];known=option['paths'][j]['legal']
        out.leaves.append(language.rebind(model,old_path,region(structure,i,256),known,new_legal[i]))
    return out.validate()
