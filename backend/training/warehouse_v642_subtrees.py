"""Learn non-executing, action-qualified subtrees inside unprotected roots.

Condition labels come exclusively from the complete masked NN distribution.
Leaf outputs fit that full distribution; no action anchors are added to the
Actor or applied after its forward pass. NN execution is never called here.
"""
from copy import deepcopy
import numpy as np
from . import warehouse_v642_guards as guards,warehouse_v642_program as language
from .warehouse_v626_search import Node,decode,encode,replace
from .warehouse_v626_program import fit_leaf,calibrate,evaluate_leaf,_support_cost
from .warehouse_v642_certificate import region,structure_info
from .warehouse_v622_program import digest,plain,_route,_eligible
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from experiments.warehouse.v626_protocol import require
from scripts.run_warehouse_v626 import internal_split

VERSION='warehouse-learned-action-qualified-unprotected-subtrees.v6.42'
def fit_option(source,root,subtree,parts,budget):
    """Fit only this root's leaves in the full path context, no eligibility."""
    tree=replace(decode(source.structure),root,subtree);structure,paths=encode(tree);structure=plain(structure)
    structure_info(structure,256,source.maximum_depth);legal=guards.leaf_actions(structure,256,source.maximum_depth)
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
        training=selected[tr[selected]];selection=selected[se[selected]];path=region(structure,global_id,256,source.maximum_depth)
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
        paths=[dict(path=region(source.structure,i,256,source.maximum_depth),legal=sorted(legal[i])) for i in ids],training_only_no_eligibility=True))


def local_predictions(option,x,m):
    t,info=structure_info(option['structure'],256);route=_route(t,x,info['depth'])
    z=np.empty((len(x),5),np.float32);q=np.empty_like(z)
    for i,model in enumerate(option['models']):
        take=route==i;z[take],q[take]=evaluate_leaf(model,x[take],m[take])
    return route,z,q


def assemble(source,options):
    tree=decode(source.structure)
    for root,option in sorted(options.items()):tree=replace(tree,root,decode(option['structure']))
    structure,paths=encode(tree);structure=plain(structure);structure_info(structure,256,source.maximum_depth)
    out=language.Program(source.role,source.actor_role_sha256,structure,[],'C3',256,protection=deepcopy(source.protection),
        target_actor_role_sha256=source.target_actor_role_sha256,maximum_depth=source.maximum_depth)
    new_legal=out.certificates();_,old_paths=encode(decode(source.structure));old_legal=source.certificates()
    for i,path in enumerate(paths):
        root=next((root for root in options if path[:len(root)]==root),None)
        if root is None:
            j=old_paths.index(path);model=source.leaves[j];old_path=region(source.structure,j,256,source.maximum_depth);known=old_legal[j]
        else:
            option=options[root];_,relative=encode(decode(option['structure']));j=relative.index(path[len(root):])
            model=option['models'][j];old_path=option['paths'][j]['path'];known=option['paths'][j]['legal']
        out.leaves.append(language.rebind(model,old_path,region(structure,i,256,source.maximum_depth),known,new_legal[i]))
    return out.validate()
