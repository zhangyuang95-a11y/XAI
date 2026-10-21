"""Finite protected-region repair. Every proposal is a non-executing program.

Only the four fit-internal partitions enter this module. The outer eligibility
and validation arrays are deliberately absent from its constructor.
"""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import time
import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.tree import DecisionTreeRegressor
from .warehouse_v626_search import Node, encode, decode, replace, regions, at
from .warehouse_v626_program import fit_leaf, calibrate, kl
from .warehouse_v626_budget import Budget as BaseBudget
from .warehouse_v625_checkpoint import BudgetExhausted, DeadlineExceeded
from .warehouse_v628_checkpoint import CheckpointSession
from .warehouse_v628_certificate import structure_info,region,rebind
from .warehouse_v628_program import Program,qualify,measure,protected_indices,path_key
from .warehouse_v622_program import digest,plain,MASK_INDICES
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v626_domain import domain_table,transform


class Budget(BaseBudget):
    def __init__(self,path,identity,*,resume=False,started_at=None):
        self.limits=dict(leaf=80000,node=4096,proof=400000)
        self.session=CheckpointSession(path,{**identity,'solver_limits':self.limits,
            'budget_contract':'v628-common-max256-reserve.v1'},max_calls=484096,
            wall_seconds=6000,resume=resume,started_at=started_at)
        self.counts={k:0 for k in self.limits};self.pending={};self.phase='fit';self.final_leaf_reserve=0
        if resume:
            for e in self.session.events():
                if e['event']=='call_started':self.counts[e['data']['metadata']['solver_kind']]+=1
        self.reserved=dict(leaf=2048,node=0,proof=13312)

    def consume(self,kind,metadata=None):
        if self.phase!='final' and self.counts[kind]>=self.limits[kind]-self.reserved[kind]:
            raise BudgetExhausted('common maximum-256 final '+kind+' reserve reached')
        return super().consume(kind,metadata)


def descendant(path,ancestor):return path[:len(ancestor)]==ancestor


def score_metrics(result,full_kl,count):
    f,e=result['feedback'],result['explanation']
    feasibility=[f['agreement']/.95,e['agreement']/.97,
        .20/max(f['kl'] if f['kl'] is not None else 1e9,1e-12),
        .20/max(e['kl'] if e['kl'] is not None else 1e9,1e-12)]
    legal=all(m[k]==0 for m in (f,e) for k in ('whole_tree_raw_illegal_count','whole_tree_path_uncertified_count'))
    coverage=sorted((f['coverage']/.45,e['coverage']/.45,f['expected_actionable_coverage']/.40))
    return (int(legal),min(1.,min(feasibility)),*coverage,-float(full_kl),-count)


class Context:
    def __init__(self,parts,parent,budget):
        if set(parts)!={'fit','calibration','eligibility','validation'}:raise ValueError('four inner partitions required')
        self.parts=parts;self.a=parts['fit'];self.parent=parent;self.budget=budget
        self.maximum=parent.maximum_leaves;self.cache=OrderedDict();self.cache_hits=0;self.evaluations=0
        self.trace=[];self.protected_paths=[]
        _,paths=encode(decode(parent.structure))
        self.protected_paths=[paths[i] for i in protected_indices(parent)]

    def editable(self,tree):
        return [(node,path,ids) for node,path,ids in regions(tree,self.a['x'])
            if not any(descendant(p,path) for p in self.protected_paths)]

    def evaluate(self,tree,*,preserve_parent=False):
        t,_=structure_info(encode(tree)[0],self.maximum)
        out=Program(self.parent.role,self.parent.actor_role_sha256,t,[],'C3',self.maximum,
            protection=deepcopy(self.parent.protection))
        legal=out.certificates();route=out.leaf_indices(self.a['x'])
        se=self.parts['validation'];sr=out.leaf_indices(se['x'])
        pr=self.parts['calibration'];cr=out.leaf_indices(pr['x']);before=[]
        parent_paths={path_key(region(self.parent.structure,i,self.maximum)):i for i in range(len(self.parent.leaves))}
        protected={digest(e['path']) for e in self.parent.protection['entries']}
        for i in range(t['leaf_count']):
            self.budget.check();path=region(t,i,self.maximum);pk=path_key(path)
            if pk in protected or preserve_parent and pk in parent_paths:
                j=parent_paths[pk]
                leaf=rebind(self.parent.leaves[j],region(self.parent.structure,j,self.maximum),path,
                    self.parent.certificates()[j],legal[i]);before.append(deepcopy(leaf))
            else:
                take=route==i;scoretake=sr==i;caltake=cr==i
                key=digest(dict(path=path['path'],legal=sorted(legal[i]),
                    rows=sha256(np.flatnonzero(take).astype('<i8').tobytes()).hexdigest()))
                if key in self.cache:
                    saved=deepcopy(self.cache[key]);leaf=rebind(saved['leaf'],saved['path'],path,legal[i],legal[i]);self.cache_hits+=1
                    before.append(saved['before'])
                else:
                    leaf=fit_leaf(self.a['x'][take],self.a['m'][take],self.a['p'][take],arm='C3',path=path,
                        legal=legal[i],budget=self.budget,
                        selection=(se['x'][scoretake],se['m'][scoretake],se['p'][scoretake]))
                    uncal=deepcopy(leaf);before.append(uncal)
                    leaf=calibrate(leaf,pr['x'][caltake],pr['m'][caltake],pr['p'][caltake],arm='C3',
                        path=path,legal=legal[i],budget=self.budget)
                    self.cache[key]=dict(leaf=deepcopy(leaf),path=path,before=uncal)
                    if len(self.cache)>512:self.cache.popitem(last=False)
            out.leaves.append(leaf)
        qualify(out,self.parts,before);metrics=measure(out,se)
        score=score_metrics(metrics,kl(se['p'],out.distributions(se['x'],se['m'])[2]).mean(),len(out.leaves))
        self.evaluations+=1
        return out,score,metrics

    def proposals(self,program,path):
        """One fully enumerated local check: split, legal guard, rebuild, prune.

        Structure targets and repair priorities read inner fit only. A full
        distribution CART is just one proposal; all final routing/proofs remain
        our own hard sparse language, with every empty/guard leaf counted.
        """
        tree=decode(program.structure);node=at(tree,path)
        entry=next(ids for _,p,ids in regions(tree,self.a['x']) if p==path)
        if not len(entry):return [],'empty_fit_region'
        x,p=self.a['x'][entry],self.a['p'][entry];out=[]
        available=9-len(path)
        if node is not None:out.append((replace(tree,path,None),'prune'))
        if available<1:return out,'depth_limit'
        _,paths=encode(tree);local=sum(descendant(p0,path) for p0 in paths)
        allowance=self.maximum-(len(paths)-local)
        if allowance<2:return out,'capacity_limit'
        # All classifier optimisation calls, including unsuccessful ones, predebit.
        for leaves in sorted({2,min(4,allowance),min(8,allowance)}):
            def solve():
                model=DecisionTreeRegressor(max_leaf_nodes=leaves,max_depth=available,
                    min_samples_leaf=20,random_state=0)
                model.fit(x,p)
                return OptimizeResult(success=True,status=0,fun=float(np.mean((model.predict(x)-p)**2)),model=model,nit=1)
            result=self.budget.solve('node',dict(path=list(path),operation='distribution_subtree',leaves=leaves),solve)
            t=result.model.tree_
            def walk(j):
                if t.children_left[j]<0:return None
                w=np.zeros(len(FEATURE_NAMES),np.float32);w[t.feature[j]]=1
                return Node(tuple(map(float,w)),float(np.float32(-t.threshold[j])),walk(t.children_left[j]),walk(t.children_right[j]))
            replacement=walk(0)
            if replacement is not None:out.append((replace(tree,path,replacement),'distribution_rebuild_'+str(leaves)))
        # Explicit physical bits permit mobile witnesses without an unsafe fallback.
        top=p.argmax(1)
        for action in sorted(set(top.tolist())-{4}):
            w=np.zeros(len(FEATURE_NAMES),np.float32);w[MASK_INDICES[action]]=1
            out.append((replace(tree,path,Node(tuple(map(float,w)),-.5)),'legal_guard_'+str(action)))
        # Disagreement-derived sparse direction, followed by quantile candidates.
        z,q=program.distributions(x,self.a['m'][entry])[1:]
        residual=p-q;valid=[i for i,v in enumerate(domain_table()['features']) if not v['constant']]
        u=transform(x,valid).astype(float);u-=u.mean(0)
        importance=np.linalg.norm(u.T@residual,axis=1)/np.maximum(np.linalg.norm(u,axis=0),1e-12)
        for j in sorted(range(len(valid)),key=lambda j:(-importance[j],valid[j]))[:3]:
            feature=valid[j]
            if np.min(x[:,feature])==np.max(x[:,feature]):continue
            for quantile in (.25,.5,.75):
                w=np.zeros(len(FEATURE_NAMES),np.float32);w[feature]=1
                out.append((replace(tree,path,Node(tuple(map(float,w)),float(np.float32(-np.quantile(x[:,feature],quantile))))),
                    'residual_axis_'+str(feature)+'_'+str(quantile)))
        unique={}
        for c,label in out:
            try:t,_=structure_info(encode(c)[0],self.maximum)
            except ValueError:continue
            unique.setdefault(digest(t),(plain(t),label))
        return list(unique.values()),'enumerated'

    def checkpoint(self,machine,beam,best,status='running'):
        store=self.budget.session.store
        def put(p):return store.put_json(p.to_dict())
        self.budget.checkpoint(dict(algorithm='protected_region_subtree_refinement_v1',machine=machine,
            beam=[dict(program=put(p),score=list(s)) for p,s in beam],best=dict(program=put(best[0]),score=list(best[1])),
            cache=[[k,store.put_json(v)] for k,v in self.cache.items()],cache_hits=self.cache_hits,
            evaluations=self.evaluations,trace=self.trace),status=status)


def optimize(ctx,*,resume=False,pause_after=0):
    if resume:
        s=ctx.budget.session.state
        if s['algorithm']!='protected_region_subtree_refinement_v1':raise ValueError('unknown resume algorithm')
        store=ctx.budget.session.store
        def get(e):return Program.from_dict(store.get_json(e['program']),ctx.parent.actor_role_sha256),tuple(e['score'])
        beam=[get(e) for e in s['beam']];best=get(s['best']);machine=s['machine']
        ctx.cache=OrderedDict((k,store.get_json(v)) for k,v in s['cache']);ctx.cache_hits=s['cache_hits']
        ctx.evaluations=s['evaluations'];ctx.trace=s['trace']
    else:
        p,score,_=ctx.evaluate(decode(ctx.parent.structure),preserve_parent=True)
        beam=[(p,score)];best=(p,score)
        # Mandatory paths reference parent topology and are stable at the anchor.
        mandatory=[list(path) for _,path,_ in ctx.editable(decode(p.structure))]
        machine=dict(round=0,beam_cursor=0,paths=mandatory,path_cursor=0,queue=None,cursor=0,
            mandatory_paths=mandatory,mandatory_checked=[],mandatory_complete=False,operation_statuses=[],pending_frontier=[])
    stopped='registered_14_rounds'
    try:
        ctx.checkpoint(machine,beam,best)
        while machine['round']<14:
            if machine['beam_cursor']>=len(beam):
                if machine['round']==0:
                    machine['mandatory_complete']=len(machine['mandatory_checked'])==len(machine['mandatory_paths'])
                if not machine['pending_frontier']:break
                candidates=[(Program.from_dict(v['program'],ctx.parent.actor_role_sha256),tuple(v['score'])) for v in machine['pending_frontier']]
                # Anchor always retained; exploratory candidate need not beat best.
                distinct={digest(p.structure):(p,s) for p,s in candidates}
                ranked=sorted(distinct.values(),key=lambda q:q[1],reverse=True)
                beam=ranked[:3]
                machine.update(round=machine['round']+1,beam_cursor=0,paths=None,path_cursor=0,
                    queue=None,cursor=0,pending_frontier=[])
                continue
            current=beam[machine['beam_cursor']][0]
            if machine['paths'] is None:
                editable=ctx.editable(decode(current.structure))
                route,z,q=current.distributions(ctx.a['x'],ctx.a['m']);loss=kl(ctx.a['p'],q)
                editable.sort(key=lambda row:(-float(loss[row[2]].sum()),len(row[1]),row[1]))
                machine['paths']=[list(p) for _,p,_ in editable]
            if machine['path_cursor']>=len(machine['paths']):
                machine.update(beam_cursor=machine['beam_cursor']+1,paths=None,path_cursor=0,queue=None,cursor=0);continue
            path=tuple(machine['paths'][machine['path_cursor']])
            if machine['queue'] is None:
                proposals,status=ctx.proposals(current,path)
                machine['queue']=[dict(structure=t,label=label) for t,label in proposals];machine['cursor']=0
                machine['operation_statuses'].append(dict(round=machine['round'],path=list(path),status=status,proposals=len(proposals)))
            if machine['cursor']>=len(machine['queue']):
                if machine['round']==0:machine['mandatory_checked'].append(list(path))
                machine.update(path_cursor=machine['path_cursor']+1,queue=None,cursor=0)
                ctx.checkpoint(machine,beam,best);continue
            item=machine['queue'][machine['cursor']]
            try:
                proposed,score,metrics=ctx.evaluate(decode(item['structure']))
                accepted=score>best[1]
                if accepted:best=(proposed,score)
                frontier=machine['pending_frontier'];frontier.append(dict(program=proposed.to_dict(),score=list(score)))
                frontier.sort(key=lambda q:q['score'],reverse=True);del frontier[3:]
                ctx.trace.append(dict(round=machine['round'],path=list(path),operation=item['label'],score=list(score),best_updated=accepted))
            except ValueError as error:
                ctx.trace.append(dict(round=machine['round'],path=list(path),operation=item['label'],rejected=str(error)))
            machine['cursor']+=1;ctx.checkpoint(machine,beam,best)
            if pause_after and ctx.evaluations>=pause_after:
                ctx.checkpoint(machine,beam,best,'paused');return best[0],dict(paused=True)
    except (BudgetExhausted,DeadlineExceeded) as error:stopped=str(error)
    complete=machine['mandatory_complete'] or machine['round']==0 and len(machine['mandatory_checked'])==len(machine['mandatory_paths'])
    return best[0],dict(stopped=stopped,mandatory_complete=complete,mandatory_checked=machine['mandatory_checked'],
        mandatory_paths=machine['mandatory_paths'],operation_statuses=machine['operation_statuses'],
        best_score=list(best[1]),evaluations=ctx.evaluations,cache_hits=ctx.cache_hits,trace=ctx.trace,
        outer_validation_used=False,budget_limited='budget' in stopped or 'deadline' in stopped or 'reserve' in stopped)
