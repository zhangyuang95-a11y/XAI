"""Whole-tree proposals, atomic subtree rebuilding and a finite beam.

Only outer-fit scenes are visible. Torch optimizes SURROGATE partitions,
never an Actor. Stored hard routing, not a fuzzy mixture, scores proposals.
"""
from __future__ import annotations
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from scipy.optimize import OptimizeResult
import torch
from .warehouse_v627_program import (Semantic,semantic_dict,semantic_from_dict,semantic_regions,
    semantic_replace,refresh_actions,compile_tree,fit_semantic,s1_semantic,CertifiedSemanticProgram)
from .warehouse_v626_program import evaluate_leaf,kl
from .warehouse_v626_domain import domain_table,transform
from .warehouse_v626_search import score_better
from .warehouse_v622_program import digest,plain
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from .warehouse_v625_checkpoint import BudgetExhausted

MAXIMUM_GRADIENT_STEPS = 20000
GRADIENT_STEPS_PER_CALL = 200


def compiled_costs(program, x, m, p):
    costs=[];cursor=0
    for action in semantic_dict(program.semantic)['actions']:
        if action == 4:
            _,q=evaluate_leaf(program.kernel.leaves[cursor],x,m);cursor+=1
        else:
            _,fallback=evaluate_leaf(program.kernel.leaves[cursor],x,m)
            _,good=evaluate_leaf(program.kernel.leaves[cursor+1],x,m);cursor+=2
            q=np.where(m[:,action,None]>0,good,fallback)
        costs.append(kl(p,q))
    return np.column_stack(costs)


def sparse_weights(values, features):
    out=np.zeros_like(values)
    for row in range(len(values)):
        total=0
        for j in sorted(range(len(features)),key=lambda i:(-abs(values[row,i]),features[i])):
            cost=2 if FEATURE_NAMES[features[j]]=='public_hard_risk_context' else 1
            if total+cost<=3 and values[row,j]!=0:
                out[row,j]=values[row,j];total+=cost
            if total==3:break
    return out


def torch_tree_proposal(ctx, current, seed, *, steps=200):
    """Hard forward + sigmoid derivative, with dense support exploration."""
    tree=current.semantic;data=semantic_dict(tree);t=data['structure'];n=len(t['split_bias'])
    if not n:return tree,0.,0
    d=domain_table()['features'];features=[i for i,v in enumerate(d) if not v['constant']]
    center=np.array([v['center'] for v in d]);scale=np.array([v['scale'] for v in d])
    weights=np.asarray(t['split_weights'],np.float32);bias=np.asarray(t['split_bias'],np.float32)
    normalized=weights[:,features]/scale[features]
    normalized_bias=bias+(weights.astype(float)*center).sum(1)
    dense=torch.nn.Parameter(torch.tensor(normalized,dtype=torch.float32))
    beta=torch.nn.Parameter(torch.tensor(normalized_bias,dtype=torch.float32))
    optimizer=torch.optim.Adam([dense,beta],lr=.015)
    x=ctx.a['x'];u=torch.tensor(transform(x,features));cost=torch.tensor(compiled_costs(current,x,ctx.a['m'],ctx.a['p']),dtype=torch.float32)
    rng=torch.Generator(device='cpu');rng.manual_seed(seed)
    # Exact path products use the same stored raw-f32 / shared fp64 router.
    from .warehouse_v626_certificate import structure_info
    _,info=structure_info(t,64)
    paths=[info['leaf_paths'][i] for i in range(len(data['actions']))];last=0.
    for step in range(steps):
        ctx.budget.check()
        ids=torch.randperm(len(x),generator=rng)[:min(512,len(x))]
        actual=sparse_weights(dense.detach().numpy(),features)
        projected=torch.tensor(actual);st=projected+(dense-dense.detach())
        raw=np.zeros_like(weights);raw[:,features]=(actual*scale[features]).astype(np.float32)
        raw_bias=(beta.detach().numpy()-(raw.astype(float)*center).sum(1)).astype(np.float32)
        hard=torch.tensor((split_scores(x[ids.numpy()],raw,raw_bias)>HARD_SPLIT_TIE_EPSILON).astype(np.float32))
        soft=torch.sigmoid((u[ids]@st.T+beta)/.5)
        decisions=hard+(soft-soft.detach())
        reach=[]
        for path in paths:
            r=torch.ones(len(ids))
            for item in path:
                v=decisions[:,item['node']];r=r*(v if item['branch']=='right' else 1-v)
            reach.append(r)
        probabilities=torch.stack(reach,1)
        # A routing inconsistency can never be hidden by a soft mixture.
        if not torch.equal(probabilities.detach().sum(1),torch.ones(len(ids))):
            raise ValueError('hard semantic route is not a partition')
        loss=(probabilities*cost[ids]).sum(1).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():dense.clamp_(-32,32);beta.clamp_(-32,32)
        last=float(loss.detach())
    actual=sparse_weights(dense.detach().numpy(),features)
    raw=np.zeros_like(weights);raw[:,features]=(actual*scale[features]).astype(np.float32)
    raw_bias=(beta.detach().numpy()-(raw.astype(float)*center).sum(1)).astype(np.float32)
    new=deepcopy(data);new['structure']['split_weights']=raw.tolist();new['structure']['split_bias']=raw_bias.tolist()
    result=refresh_actions(semantic_from_dict(new),x,ctx.a['p'])
    compile_tree(result,ctx.maximum)
    return result,last,steps


def seed_number(value):return int(digest(value)[:8],16)%2147483647


def seed_tree(ctx, index, initial):
    if index==0:
        return s1_semantic(initial,ctx.a['x'],ctx.a['p'],ctx.maximum)
    out={}
    def solve():
        out['tree']=fit_semantic(ctx.a['x'],ctx.a['p'],ctx.maximum//2,5 if ctx.maximum==32 else 6,
            seed_number([ctx.role,index,'semantic-start']),random=index>=2)
        return OptimizeResult(success=True,status=0,nit=1,fun=0.)
    ctx.budget.solve('node',dict(kind='semantic_initialization',start=index),solve)
    return refresh_actions(out['tree'],ctx.a['x'],ctx.a['p'])


def gradient_proposal(ctx, program, index, gradient_used):
    if ctx.budget.gradient_steps+GRADIENT_STEPS_PER_CALL>MAXIMUM_GRADIENT_STEPS:
        raise BudgetExhausted('registered surrogate gradient-step budget exhausted')
    out={};seed=seed_number([semantic_dict(program.semantic),index,'whole-gradient'])
    def solve():
        out['tree'],loss,actual=torch_tree_proposal(ctx,program,seed,steps=GRADIENT_STEPS_PER_CALL)
        return OptimizeResult(success=True,status=0,nit=actual,fun=loss)
    # Reserve all steps BEFORE starting, including failed/interrupted calls.
    ctx.budget.solve('node',dict(kind='surrogate_whole_tree_gradient',seed=seed,
        reserved_gradient_steps=GRADIENT_STEPS_PER_CALL),solve)
    return out['tree']


def rebuild_proposals(ctx, current, index):
    tree=current.semantic;x,p=ctx.a['x'],ctx.a['p'];maximum=ctx.maximum//2;depth=5 if ctx.maximum==32 else 6
    entries=semantic_regions(tree,x);count=len(semantic_dict(tree)['actions'])
    cost=compiled_costs(current,x,ctx.a['m'],p)
    row_cost=np.empty(len(x))
    leaf_id=0
    for node,path,ids in entries:
        if node.weights is None:row_cost[ids]=cost[ids,leaf_id];leaf_id+=1
    # Rebuild root and the ancestors of high-error terminals, including full
    # depth terminals. This is not restricted to missing legal-action bits.
    paths=[()]
    terminals=sorted([(float(row_cost[ids].sum()),path) for node,path,ids in entries if node.weights is None],reverse=True)
    for _,path in terminals:
        ancestor=path[:-2] if len(path)>=2 else ()
        if ancestor not in paths:paths.append(ancestor)
        if len(paths)>=5:break
    result=[]
    for path in paths:
        node,_,ids=next(v for v in entries if v[1]==path)
        old_count=len(semantic_dict(node)['actions']);capacity=maximum-count+old_count
        if len(ids)<40 or depth-len(path)<1:continue
        out={}
        def solve():
            out['tree']=fit_semantic(x[ids],p[ids],capacity,depth-len(path),
                seed_number([semantic_dict(tree),index,list(path),'rebuild']))
            return OptimizeResult(success=True,status=0,nit=1,fun=0.)
        ctx.budget.solve('node',dict(kind='whole_subtree_rebuild',path=list(path),capacity=capacity),solve)
        result.append((semantic_replace(tree,path,out['tree']),'rebuild:'+str(path)))
    # Atomic donor pruning plus receiver reconstruction, one complete candidate.
    donors=[]
    for node,path,ids in entries:
        if node.weights is None or len(semantic_dict(node)['actions'])!=2:continue
        mean=p[ids].mean(0) if len(ids) else np.ones(5)/5
        donors.append((float(kl(p[ids],np.tile(mean,(len(ids),1))).sum()-row_cost[ids].sum()),path,ids))
    if donors:
        _,donor,ids=min(donors,key=lambda t:(t[0],t[1]))
        reduced=semantic_replace(tree,donor,Semantic(action=int(p[ids].sum(0).argmax()) if len(ids) else 4))
        for _,receiver in terminals:
            if receiver[:len(donor)]==donor:continue
            path=receiver if len(receiver)<depth else receiver[:-2]
            if donor[:len(path)]==path:continue
            entries2=semantic_regions(reduced,x)
            match=next((v for v in entries2 if v[1]==path),None)
            if match is None:continue
            node,_,ids=match;capacity=maximum-len(semantic_dict(reduced)['actions'])+len(semantic_dict(node)['actions'])
            if len(ids)<40:continue
            out={}
            def solve():
                out['tree']=fit_semantic(x[ids],p[ids],capacity,depth-len(path),seed_number([index,list(path),'exchange']))
                return OptimizeResult(success=True,status=0,nit=1,fun=0.)
            ctx.budget.solve('node',dict(kind='atomic_leaf_budget_exchange',donor=list(donor),receiver=list(path)),solve)
            result.append((semantic_replace(reduced,path,out['tree']),'exchange:'+str(donor)+':'+str(path)))
            break
    return result


class SearchPaused(RuntimeError):pass


def optimize(ctx, initial, *, rounds=14, beam_width=4, resume=False, pause_after_evaluations=0):
    store=ctx.budget.session.store
    def store_program(p):return store.put_json(p.to_dict())
    if resume:
        state=ctx.budget.session.state
        if state.get('algorithm')!='v627-semantic-beam.v1':raise ValueError('wrong search state')
        machine=deepcopy(state['machine']);ctx.cache=OrderedDict((key,store.get_json(ref)) for key,ref in state['cache'])
        ctx.cache_hits=state['cache_hits'];ctx.evaluations=state['evaluations']
        best=CertifiedSemanticProgram.from_dict(store.get_json(state['best']),ctx.actor_sha)
        best_score=tuple(state['best_score']);trace=state['trace']
    else:
        machine=dict(phase='seeds',seed=0,round=0,frontier=[],queue=[],cursor=0,
            next_frontier=[],gradient_used=0,mandatory_completed=[])
        best=None;best_score=None;trace=[]
    stopped='maximum_registered_rounds'
    def checkpoint(status='running'):
        ctx.best=best;ctx.best_score=best_score;ctx.trace=trace
        ctx.budget.checkpoint(dict(algorithm='v627-semantic-beam.v1',machine=machine,
            best=store_program(best) if best is not None else None,best_score=list(best_score) if best_score is not None else None,
            cache=[[key,store.put_json(value)] for key,value in ctx.cache.items()],cache_hits=ctx.cache_hits,
            evaluations=ctx.evaluations,trace=trace),status=status)
    def accept(semantic,kind):
        nonlocal best,best_score
        candidate,score,m=ctx.evaluate(semantic);ref=store_program(candidate)
        improved=best is None or score_better(score,best_score)
        if improved:best,best_score=candidate,score
        trace.append(dict(kind=kind,round=machine['round'],score=list(score),improved_best=improved,
            actual_inner_qualification=True,program_sha256=ref['sha256']))
        return dict(program=ref,score=list(score))
    try:
        while machine['phase']!='completed':
            if machine['phase']=='seeds':
                if machine['seed']>=4:
                    machine.update(phase='beam',cursor=0,queue=[],next_frontier=[])
                    machine['frontier']=sorted(machine['frontier'],key=lambda a:tuple(a['score']),reverse=True)[:beam_width]
                    checkpoint();continue
                start=machine['seed']
                tree=seed_tree(ctx,start,initial)
                item=accept(tree,'start:'+str(start))
                current=CertifiedSemanticProgram.from_dict(store.get_json(item['program']),ctx.actor_sha)
                # Every start gets both mandatory global and subtree exploration
                # before any stagnation decision. No two-stale early stop exists.
                proposal=gradient_proposal(ctx,current,start,machine['gradient_used'])
                machine['gradient_used']+=GRADIENT_STEPS_PER_CALL
                candidates=[item,accept(proposal,'mandatory-gradient:'+str(start))]
                rebuilt=rebuild_proposals(ctx,current,start)
                if not rebuilt:rebuilt=[(tree,'degenerate-root-rebuild')]
                candidates.append(accept(rebuilt[0][0],'mandatory-'+rebuilt[0][1]))
                machine['frontier'].append(max(candidates,key=lambda a:tuple(a['score'])))
                machine['mandatory_completed'].append(start);machine['seed']+=1;checkpoint()
            else:
                if machine['round']>=rounds:machine['phase']='completed';checkpoint();break
                if not machine['queue']:
                    queue=[]
                    for k,item in enumerate(machine['frontier']):
                        current=CertifiedSemanticProgram.from_dict(store.get_json(item['program']),ctx.actor_sha)
                        proposal=gradient_proposal(ctx,current,[machine['round'],k],machine['gradient_used'])
                        machine['gradient_used']+=GRADIENT_STEPS_PER_CALL
                        queue.append(dict(semantic=semantic_dict(proposal),kind=f'beam-gradient:{k}'))
                        for tree,label in rebuild_proposals(ctx,current,[machine['round'],k]):
                            queue.append(dict(semantic=semantic_dict(tree),kind=label))
                    # Persist complete generated queue before consuming it.
                    machine.update(queue=queue,cursor=0,next_frontier=[]);checkpoint()
                if machine['cursor']<len(machine['queue']):
                    item=machine['queue'][machine['cursor']]
                    machine['next_frontier'].append(accept(semantic_from_dict(item['semantic']),item['kind']))
                    machine['cursor']+=1;checkpoint()
                else:
                    # Frontier may move downhill; the separately saved best never
                    # does. This permits multiple coordinated partition changes.
                    pool=machine['next_frontier'];seen=set();frontier=[]
                    for item in sorted(pool,key=lambda a:tuple(a['score']),reverse=True):
                        key=item['program']['sha256']
                        if key not in seen:frontier.append(item);seen.add(key)
                        if len(frontier)==beam_width:break
                    if not frontier:stopped='no_distinct_candidate';machine['phase']='completed'
                    else:machine.update(frontier=frontier,queue=[],cursor=0,next_frontier=[],round=machine['round']+1)
                    checkpoint()
            if pause_after_evaluations and ctx.evaluations>=pause_after_evaluations:
                checkpoint('paused');raise SearchPaused('explicit full-candidate safe pause')
    except BudgetExhausted as error:
        stopped=str(error)
        if best is None:raise
    return best,dict(score=list(best_score),trace=trace,stopped=stopped,evaluations=ctx.evaluations,
        mandatory_completed=machine['mandatory_completed'],gradient_steps_reserved=ctx.budget.gradient_steps,
        cache_hits=ctx.cache_hits,selection_uses_only_fit=True,validation_used_for_search=False,
        exact_eligibility_rule=True,no_stagnation_stop=True)
