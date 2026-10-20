"""Fit-only hard-tree alternating optimization and certified capacity growth.

All predicates are proposed from full-NN distribution costs, never applied to
environment execution. Explicit legal guards are immutable. Candidate models
are accepted only after complete hard routing and exact certificate replay.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .warehouse_v626_program import Program,fit_leaf,evaluate_leaf,kl,_proxy
from .warehouse_v626_certificate import region,structure_info,binding,verify
from .warehouse_v626_domain import domain_table,transform
from .warehouse_v626_budget import Budget
from .warehouse_v625_checkpoint import BudgetExhausted
from .warehouse_v622_program import digest,plain,MASK_INDICES
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON,explicit_mask_branches


@dataclass(frozen=True)
class Node:
    weight:tuple
    bias:float
    left:object=None
    right:object=None


def decode(t):
    def walk(ref):
        if ref<0:return None
        return Node(tuple(map(float,t['split_weights'][ref])),float(t['split_bias'][ref]),
            walk(int(t['left_refs'][ref])),walk(int(t['right_refs'][ref])))
    return walk(int(t['root_ref']))


def encode(tree):
    weights=[];bias=[];left=[];right=[];leaves=[]
    def walk(node,path):
        if node is None:
            index=len(leaves);leaves.append(path);return -1-index
        i=len(weights);weights.append(node.weight);bias.append(node.bias);left.append(0);right.append(0)
        left[i]=walk(node.left,path+(0,));right[i]=walk(node.right,path+(1,));return i
    root=walk(tree,())
    return dict(root_ref=root,split_weights=np.array(weights,np.float32).reshape(-1,len(FEATURE_NAMES)),
        split_bias=np.array(bias,np.float32),left_refs=np.array(left),right_refs=np.array(right),
        leaf_count=len(leaves),feature_dim=len(FEATURE_NAMES)),leaves


def replace(tree,path,new):
    if not path:return new
    if tree is None:raise ValueError('unreachable replacement')
    return Node(tree.weight,tree.bias,replace(tree.left,path[1:],new) if path[0]==0 else tree.left,
        replace(tree.right,path[1:],new) if path[0]==1 else tree.right)


def at(tree,path):
    for bit in path:tree=tree.right if bit else tree.left
    return tree


def regions(tree,x):
    def walk(node,path,rows):
        yield node,path,rows
        if node is not None:
            go=split_scores(x[rows],np.array(node.weight,np.float32)[None],np.array([node.bias],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
            yield from walk(node.left,path+(0,),rows[~go]);yield from walk(node.right,path+(1,),rows[go])
    return list(walk(tree,(),np.arange(len(x))))


class SearchBudget(Budget):
    def __init__(self,path,identity,limits,seconds,*,maximum_leaves=32,resume=False,started_at=None):
        from .warehouse_v626_checkpoint import CheckpointSession
        self.limits=dict(limits)
        if set(limits)!={'leaf','node','proof'} or any(type(n) is not int or n<0 for n in limits.values()):raise ValueError('explicit categorized limits required')
        self.session=CheckpointSession(path,{**identity,'solver_limits':self.limits,'budget_contract':'v626-categorized.v1'},
            max_calls=sum(limits.values()),wall_seconds=seconds,resume=resume,started_at=started_at)
        self.counts={k:0 for k in limits};self.pending={};self.phase='fit'
        if resume:
            for row in self.session.events():
                if row['event']=='call_started':self.counts[row['data']['metadata']['solver_kind']]+=1
        self.reserved=dict(leaf=8*maximum_leaves,proof=52*maximum_leaves,node=0)
        self.final_leaf_reserve=8*maximum_leaves
    def consume(self,kind,metadata=None):
        if self.phase!='final' and self.counts[kind]>=self.limits[kind]-self.reserved[kind]:
            raise BudgetExhausted('search leaves final '+kind+' reserve untouched')
        return super().consume(kind,metadata)


class Context:
    def __init__(self,parts,spec,budget,maximum_leaves):
        from scripts.run_warehouse_v626 import internal_split
        self.a=parts['fit'];self.spec=spec;self.budget=budget;self.maximum=maximum_leaves
        self.train,self.selection=internal_split(self.a['rows']);self.cache=OrderedDict();self.cache_hits=0;self.cache_evictions=0
        self.evaluations=0;self.trace=[]
        self.initial_score=None

    def evaluate(self,tree):
        t,_=encode(tree);t,info=structure_info(t,self.maximum)
        program=Program(self.spec['role'],self.spec['actor_role_sha256'],t,[],'C3',self.maximum)
        routes=program.leaf_indices(self.a['x']);legal=program.certificates();score=np.zeros(3)
        for i in range(t['leaf_count']):
            self.budget.check();tr=(routes==i)&self.train;se=(routes==i)&self.selection;path=region(t,i,self.maximum)
            key=digest(dict(path=path['path'],legal=sorted(legal[i]),
                train=sha256(np.flatnonzero(tr).astype('<i8').tobytes()).hexdigest(),
                selection=sha256(np.flatnonzero(se).astype('<i8').tobytes()).hexdigest()))
            if key in self.cache:
                self.cache_hits+=1;leaf=deepcopy(self.cache[key])
                # Only the full-tree/leaf ID binding changed. The complete
                # numeric domain, rows and legal action set are byte-identical.
                leaf['certificate']['binding_sha256']=binding(leaf,path,legal[i],True)
                verify(leaf,path,legal[i],leaf['certificate'])
            else:
                leaf=fit_leaf(self.a['x'][tr],self.a['m'][tr],self.a['p'][tr],arm='C3',path=path,legal=legal[i],
                    budget=self.budget,selection=(self.a['x'][se],self.a['m'][se],self.a['p'][se]))
                self.cache[key]=deepcopy(leaf)
                if len(self.cache)>512:self.cache.popitem(last=False);self.cache_evictions+=1
            program.leaves.append(leaf)
            local=_proxy(leaf,(self.a['x'][se],self.a['m'][se],self.a['p'][se]))
            score[0]+=local[0];score[1]+=local[1];score[2]+=local[2]*int(se.sum())
        program.validate();n=int(self.selection.sum());self.evaluations+=1
        return program,(float(score[0]/n),float(score[1]/n),float(score[2]/n),-t['leaf_count'])

    def state(self,current,score,queue,cursor,round_id):
        store=self.budget.session.store
        encoded=current.to_dict();encoded['leaves']=[store.put_json(value) for value in encoded['leaves']]
        return dict(algorithm='v626_hard_TAO_v1',current=encoded,current_leaves_are_CAS_refs=True,score=list(score),initial_score=self.initial_score,
            cache=[[key,store.put_json(value)] for key,value in self.cache.items()],cache_values_are_CAS_refs=True,cache_hits=self.cache_hits,
            cache_evictions=self.cache_evictions,evaluations=self.evaluations,trace=self.trace,
            queue=queue,cursor=cursor,round=round_id,deterministic_no_random_draws=True,
            inner_training_scenarios=sorted({r['base_seed'] for r,keep in zip(self.a['rows'],self.train,strict=True) if keep}),
            inner_selection_scenarios=sorted({r['base_seed'] for r,keep in zip(self.a['rows'],self.selection,strict=True) if keep}))

    def restore(self,state):
        if state.get('algorithm')!='v626_hard_TAO_v1':raise ValueError('unknown search checkpoint')
        if state.get('current_leaves_are_CAS_refs') is not True or state.get('cache_values_are_CAS_refs') is not True:raise ValueError('unknown state object encoding')
        self.cache=OrderedDict((key,self.budget.session.store.get_json(ref)) for key,ref in state['cache'])
        self.cache_hits=state['cache_hits'];self.cache_evictions=state['cache_evictions']
        self.evaluations=state['evaluations'];self.trace=state['trace']
        self.initial_score=state['initial_score']
        encoded=deepcopy(state['current']);encoded['leaves']=[self.budget.session.store.get_json(ref) for ref in encoded['leaves']]
        program=Program.from_dict(encoded,self.spec['actor_role_sha256'])
        if program.maximum_leaves!=self.maximum or program.role!=self.spec['role']:raise ValueError('checkpoint role or capacity differs')
        return program,tuple(state['score'])


def score_better(new,old):
    for a,b in zip(new,old,strict=True):
        if a>b+1e-9:return True
        if a<b-1e-9:return False
    return False


def subtree_probabilities(program,node_ref,indices,x,m):
    t=program.structure;refs=np.full(len(indices),node_ref,dtype=int);local=x[indices]
    for _ in range(8):
        rows=np.flatnonzero(refs>=0)
        if not len(rows):break
        nodes=refs[rows];scores=split_scores(local,t['split_weights'],t['split_bias'])
        right=scores[rows,nodes]>HARD_SPLIT_TIE_EPSILON
        refs[rows]=np.where(right,np.asarray(t['right_refs'])[nodes],np.asarray(t['left_refs'])[nodes])
    if np.any(refs>=0):raise ValueError('subtree route exceeded registered depth')
    q=np.empty((len(indices),5),np.float32)
    for leaf in np.unique(-1-refs):
        take=(-1-refs)==leaf
        _,q[take]=evaluate_leaf(program.leaves[int(leaf)],local[take],m[indices][take])
    return q


def node_candidates(ctx,current,tree,path):
    node=at(tree,path);x,m,p=ctx.a['x'],ctx.a['m'],ctx.a['p']
    if node is None:return []
    if any(explicit_mask_branches(np.array(node.weight,np.float32),node.bias,MASK_INDICES)):return []
    entry=next((rows for _,where,rows in regions(tree,x) if where==path),None)
    ids=entry[ctx.train[entry]]
    if len(ids)<40:return []
    t=current.structure;ref=t['root_ref']
    for bit in path:ref=int(t['right_refs'][ref] if bit else t['left_refs'][ref])
    ql=subtree_probabilities(current,int(t['left_refs'][ref]),ids,x,m)
    qr=subtree_probabilities(current,int(t['right_refs'][ref]),ids,x,m)
    difference=kl(p[ids],ql)-kl(p[ids],qr);care=np.abs(difference)>1e-7
    if care.sum()<20:return []
    ids=ids[care];target=(difference[care]>0).astype(float);weight=np.abs(difference[care])
    weight/=max(float(weight.mean()),1e-12)
    d=domain_table()['features'];features=[i for i,v in enumerate(d) if not v['constant']]
    u=transform(x[ids],features).astype(float);centered=u-u.mean(0)
    proxy=np.abs(np.einsum('ni,n->i',centered,weight*(target-.5),optimize=False))/np.maximum(np.linalg.norm(centered,axis=0),1e-12)
    picked=[];cost=0
    for j in sorted(range(len(features)),key=lambda j:(-proxy[j],features[j])):
        i=features[j];c=2 if FEATURE_NAMES[i]=='public_hard_risk_context' else 1
        if np.linalg.norm(centered[:,j])>1e-9 and cost+c<=3:picked.append(i);cost+=c
        if cost==3:break
    supports=[tuple(np.flatnonzero(node.weight).tolist()),tuple(sorted(picked))];out=[];seen=set()
    for support in supports:
        if not support or support in seen:continue
        seen.add(support);design=np.column_stack((np.ones(len(ids)),transform(x[ids],support).astype(float)))
        def objective(theta):
            logits=design@theta
            loss=np.mean(weight*(np.logaddexp(0,logits)-target*logits))+.001*np.sum(theta[1:]**2)
            grad=design.T@(weight*(expit(logits)-target))/len(target);grad[1:]+=.002*theta[1:]
            return float(loss),grad
        def solve():return minimize(objective,np.zeros(len(support)+1),jac=True,method='L-BFGS-B',
            bounds=[(-32.,32.)]*(len(support)+1),options={'maxiter':100,'ftol':1e-7})
        result=ctx.budget.solve('node',dict(path=list(path),support=list(support)),solve)
        if result.success:
            w=np.zeros(len(FEATURE_NAMES),np.float32)
            for j,i in enumerate(support):w[i]=np.float32(result.x[j+1]*d[i]['scale'])
            b=np.float32(result.x[0]-sum(float(w[i])*d[i]['center'] for i in support))
            out.append(Node(tuple(map(float,w)),float(b),node.left,node.right))
    return out


def growth_candidates(ctx,current,tree):
    x,m,p=ctx.a['x'],ctx.a['m'],ctx.a['p'];routes=current.leaf_indices(x);proofs=current.certificates()
    candidates=[];_,leaf_paths=encode(tree)
    ranked=[]
    for i,path in enumerate(leaf_paths):
        take=(routes==i)&ctx.train
        if len(path)>=7 or int(take.sum())<40:continue
        z,q=evaluate_leaf(current.leaves[i],x[take],m[take]);top=p[take].argmax(1)
        failure=float(np.sum(kl(p[take],q)))
        ranked.append((failure,i,path,top))
    for _,i,path,top in sorted(ranked,key=lambda item:(-item[0],item[1]))[:4]:
        take=(routes==i)&ctx.train
        choices=[a for a in range(4) if a not in proofs[i] and np.any(top==a)]
        choices.sort(key=lambda a:(-int(np.sum(top==a)),a))
        for action in choices:
            w=np.zeros(len(FEATURE_NAMES),np.float32);w[MASK_INDICES[action]]=1.
            candidates.append((replace(tree,path,Node(tuple(map(float,w)),-.5)),f'grow-legal-{i}-{action}'))
        # A data-selected axis predicate may accompany a new legal guard when
        # two remaining depths exist. It is still counted as two real nodes.
        if len(path)<=5 and choices:
            target=(top==choices[0]).astype(float);a=ctx.a['x'][take]
            d=domain_table()['features'];valid=[j for j,v in enumerate(d) if not v['constant']]
            u=transform(a,valid).astype(float);u-=u.mean(0)
            corr=np.abs(np.einsum('ni,n->i',u,target-target.mean(),optimize=False))
            for k in sorted(range(len(valid)),key=lambda k:(-corr[k],valid[k]))[:2]:
                j=valid[k];threshold=float(np.float32(np.median(a[:,j])))
                if np.min(a[:,j])==np.max(a[:,j]):continue
                w=np.zeros(len(FEATURE_NAMES),np.float32);w[j]=1.
                guard=np.zeros(len(FEATURE_NAMES),np.float32);guard[MASK_INDICES[choices[0]]]=1.
                replacement=Node(tuple(map(float,w)),-threshold,None,Node(tuple(map(float,guard)),-.5))
                candidates.append((replace(tree,path,replacement),f'grow-feature-{i}-{j}'))
    return [(candidate,label) for candidate,label in candidates if encode(candidate)[0]['leaf_count']<=64]


class SearchPaused(RuntimeError):pass


def optimize(ctx,initial,*,rounds=14,pause_after_evaluations=0,resume=False):
    """Exact queue/cursor state machine; only explicit safe pauses resume."""
    if resume:
        saved=ctx.budget.session.state;current,score=ctx.restore(saved)
        machine=deepcopy(saved['machine']);tree=decode(current.structure)
    else:
        tree=decode(initial);current,score=ctx.evaluate(tree)
        ctx.initial_score=list(score)
        machine=dict(sweep=0,stage='growth' if ctx.maximum==64 else 'nodes',stale=0,improved=False,
            paths=None,path_cursor=0,queue=None,cursor=0,skipped_after_growth_accept=[],growth_accepted=0)
    def checkpoint(status='running'):
        ctx.best=current;ctx.best_score=score
        state=ctx.state(current,score,machine['queue'] or [],machine['cursor'],machine['sweep'])
        state['machine']=deepcopy(machine);ctx.budget.checkpoint(state,status=status)
    checkpoint()
    stopped='maximum_rounds'
    try:
        while machine['sweep']<rounds:
            if machine['stage']=='growth':
                if machine['queue'] is None:
                    machine['queue']=[dict(structure=plain(encode(c)[0]),label=label) for c,label in growth_candidates(ctx,current,tree)]
                    machine['cursor']=0
                if machine['cursor']<len(machine['queue']):
                    item=machine['queue'][machine['cursor']];candidate=decode(item['structure'])
                    proposed,new=ctx.evaluate(candidate);accepted=score_better(new,score)
                    ctx.trace.append(dict(round=machine['sweep'],kind=item['label'],score=list(new),accepted=accepted))
                    machine['cursor']+=1
                    if accepted:
                        tree,current,score=candidate,proposed,new;machine['improved']=True
                        machine['skipped_after_growth_accept']=machine['queue'][machine['cursor']:]
                        machine['growth_accepted']+=1
                        machine['stage']='nodes' if machine['growth_accepted']>=4 or len(current.leaves)>=64 else 'growth'
                        machine['queue']=None
                    checkpoint()
                else:machine['stage']='nodes';machine['queue']=None
            else:
                if machine['paths'] is None:
                    paths=[path for node,path,_ in regions(tree,ctx.a['x']) if node is not None]
                    paths.sort(key=lambda path:(-len(path),path));machine['paths']=[list(p) for p in paths]
                if machine['path_cursor']>=len(machine['paths']):
                    machine['stale']=0 if machine['improved'] else machine['stale']+1
                    if machine['stale']>=2:stopped='two_complete_fit_only_stagnant_sweeps';break
                    machine.update(sweep=machine['sweep']+1,stage='growth' if ctx.maximum==64 else 'nodes',
                        improved=False,paths=None,path_cursor=0,queue=None,cursor=0,skipped_after_growth_accept=[],growth_accepted=0)
                    checkpoint();continue
                path=tuple(machine['paths'][machine['path_cursor']])
                if machine['queue'] is None:
                    nodes=node_candidates(ctx,current,tree,path)
                    machine['queue']=[dict(structure=plain(encode(replace(tree,path,node))[0]),label='TAO-node') for node in nodes]
                    machine['cursor']=0
                if machine['cursor']<len(machine['queue']):
                    item=machine['queue'][machine['cursor']];candidate=decode(item['structure'])
                    proposed,new=ctx.evaluate(candidate);accepted=score_better(new,score)
                    ctx.trace.append(dict(round=machine['sweep'],kind='TAO-node',path=list(path),score=list(new),accepted=accepted))
                    machine['cursor']+=1
                    if accepted:tree,current,score=candidate,proposed,new;machine['improved']=True
                    checkpoint()
                else:
                    machine['path_cursor']+=1;machine['queue']=None;machine['cursor']=0
            if pause_after_evaluations and ctx.evaluations>=pause_after_evaluations:
                checkpoint('paused');raise SearchPaused('explicit complete-candidate safe pause')
    except BudgetExhausted as error:
        stopped=str(error)
    return current,dict(score=list(score),initial_score=ctx.initial_score,stopped=stopped,evaluations=ctx.evaluations,
        cache_hits=ctx.cache_hits,cache_evictions=ctx.cache_evictions,trace=ctx.trace,
        selection_uses_only_fit=True,validation_used_for_search=False)
