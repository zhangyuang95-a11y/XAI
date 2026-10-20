"""Complete-domain sparse model trees, exclusively surrogate evidence.

The complete frozen NN is never replaced by this class. Old C0/C1 kernels
are dispatched unchanged; C2/C3 have separately versioned normalization and
certificate semantics. No environment is constructed by this module.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from . import warehouse_v625_program as legacy
from .warehouse_v622_program import (_eligible, _route, gate_metrics, digest, plain,
    masked_probabilities, MASK_INDICES)
from .warehouse_v6_actor import FEATURE_NAMES, FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256
from .warehouse_v6_tree_routing import certified_leaf_actions
from .warehouse_v626_domain import domain_table,domain_sha,transform,validate_rows
from .warehouse_v626_certificate import structure_info,region,certify,verify,coefficient_data,forward_error

VERSION='warehouse-role-program.v6.26-domain-complete'
ARMS=('C0','C1','C2','C3')


class LegacyBudget:
    def __init__(self,budget):self.budget=budget
    def check(self):self.budget.check()
    def consume(self):return self.budget.consume('leaf',{'kernel':'v625_unchanged'})
    def finish(self,token,*,success,detail):self.budget.finish(token,success=success,detail=detail)


def evaluate_leaf(leaf,x,m):
    if leaf.get('version')==legacy.LEAF_VERSION:return legacy.evaluate_leaf(leaf,x,m)
    support,b,w=coefficient_data(leaf)
    u=transform(x,support)
    logits=np.broadcast_to(b,(len(x),5)).copy()
    for j in range(len(support)):
        logits=np.add(logits,np.multiply(u[:,j,None],w[None,:,j],dtype=np.float32),dtype=np.float32)
    return logits,masked_probabilities(logits,m)


def kl(p,q):
    return np.sum(p.astype(float)*(np.log(np.maximum(p.astype(float),1e-300))
        -np.log(np.maximum(q.astype(float),1e-300))),axis=-1)


def _support_cost(s):return sum(2 if FEATURE_NAMES[i]=='public_hard_risk_context' else 1 for i in s)


def _rank(x,p,m,model,excluded=()):
    if not len(x):return []
    _,q=evaluate_leaf(model,x,m);res=p.astype(float)-q.astype(float)
    d=domain_table()['features'];ids=[i for i,v in enumerate(d) if not v['constant'] and i not in excluded]
    # Rank over ALL publicly bounded variables, never just the old whitelist.
    u=transform(x,ids).astype(float);u-=u.mean(0)
    norms=np.linalg.norm(u,axis=0)
    scores=np.linalg.norm(np.einsum('ni,na->ia',u,res,optimize=False),axis=1)/np.maximum(norms,1e-12)
    return [ids[j] for j in sorted(range(len(ids)),key=lambda j:(-scores[j],ids[j])) if norms[j]>1e-12]


def _proxy(model,selection):
    x,m,p=selection
    if not len(x):return (0.,0.,-float(model.get('objective',0.)))
    z,q=evaluate_leaf(model,x,m);top=z.argmax(1);match=top==p.argmax(1)
    agreement=float(match.mean());loss=float(kl(p,q).mean())
    return (float(len(x)) if agreement>=.97 and loss<=.20 else 0.,
        float(np.sum(p[np.arange(len(p)),top]*match)) if agreement>=.93 and loss<=.25 else 0.,-loss)


def _trial(x,m,p,support,witness,path,legal,use_path,budget,initial=None):
    k=len(support);d=domain_table()['features'];u=transform(x,support).astype(float)
    design=np.column_stack((np.ones(len(x)),u))
    prior=np.log((p.sum(0)+.2)/(m.sum(0)+1.));prior-=prior.mean()
    start=np.zeros((5,k+1));start[:,0]=prior
    if initial is not None and initial.get('version')==VERSION and initial['support']==list(support):
        start[:,0]=initial['bias'];start[:,1:]=initial['weights']
    # A universally feasible constant starting point remains available even
    # when an LP relaxation is empty or a support trial fails to converge.
    start[:,1:]=0.
    forbidden=sorted(set(range(5))-set(legal))
    worst=dict(version=VERSION,domain_sha256=domain_sha(),normalized=True,support=list(support),
        bias=[32.]*5,weights=[[32.]*k for _ in range(5)])
    margin=.001+2*float(forward_error(worst).max())+1e-5
    if forbidden:start[witness,0]=max(start[witness,0],float(start[forbidden,0].max())+margin+.01)
    start-=start.mean(0)
    center=np.zeros((k+1,5*(k+1)))
    for j in range(k+1):center[j,j::k+1]=1.
    if use_path:
        points=[x.mean(0)] if len(x) else [np.array([(v['lower']+v['upper'])/2 for v in d])]
    else:
        points=[]
        for corner in product(*[(d[i]['lower'],d[i]['upper']) for i in support]):
            row=np.zeros(len(FEATURE_NAMES));row[list(support)]=corner;points.append(row)
    def objective(flat):
        values=flat.reshape(5,k+1);logits=design@values.T
        if len(x):
            normal=logsumexp(np.where(m>.5,logits,-np.inf),axis=1)
            q=np.exp(np.where(m>.5,logits,-np.inf)-normal[:,None])
            ce=float(np.mean(p.sum(1)*normal-np.sum(p*logits,axis=1)))
            grad=(q*p.sum(1)[:,None]-p).T@design/len(x)
        else:ce=0.;grad=np.zeros_like(values)
        anchor=np.zeros_like(values);anchor[:,0]=prior
        return ce+1e-4*float(np.sum((values-anchor)**2)),(grad+2e-4*(values-anchor)).ravel()
    for round_id in range(8):
        matrix=[]
        for point in points:
            local=np.array([1.,*[(float(point[i])-d[i]['center'])*d[i]['scale'] for i in support]])
            for a in forbidden:
                row=np.zeros((5,k+1));row[witness]=local;row[a]=-local;matrix.append(row.ravel())
        matrix=np.asarray(matrix).reshape(len(matrix),5*(k+1))
        constraints=[{'type':'eq','fun':lambda v:center@v,'jac':lambda v:center}]
        if len(matrix):constraints.append({'type':'ineq','fun':lambda v:matrix@v-margin,'jac':lambda v:matrix})
        def solve():return minimize(objective,start.ravel(),method='SLSQP',jac=True,
            bounds=[(-32.,32.)]*start.size,constraints=constraints,options={'maxiter':200,'ftol':1e-6})
        res=budget.solve('leaf',{'support':list(support),'witness':witness,'cut_round':round_id},solve)
        if not res.success or not np.isfinite(res.fun):return None
        values=res.x.reshape(5,k+1).astype(np.float32)
        leaf=dict(version=VERSION,domain_sha256=domain_sha(),normalized=True,support=list(support),
            bias=values[:,0].tolist(),weights=values[:,1:].tolist(),witness=int(witness),objective=float(res.fun))
        proof,counterexamples=certify(leaf,path,legal,use_path=use_path,budget=budget)
        if proof['passed']:
            leaf['certificate']=proof;return leaf
        for example in counterexamples:
            point=np.array([(v['lower']+v['upper'])/2 for v in d]);
            for i,value in example.items():point[i]=value
            points.append(point)
        start=res.x.reshape(5,k+1)
    return None


def fit_leaf(x,m,p,*,arm,path,legal,budget,selection=None,initial=None,final=False):
    if arm not in ARMS:raise ValueError('unknown v626 method')
    budget.check()
    if arm in ('C0','C1'):
        return legacy.fit_leaf(x,m,p,FEATURE_NAMES,legal,mode='P0' if arm=='C0' else 'P1',
            support=initial['support'] if final else None,
            witness=initial['witness'] if final else None,budget=LegacyBudget(budget))
    if selection is None:selection=(x,m,p)
    prior=np.log((p.sum(0)+.2)/(m.sum(0)+1.));top=int(prior.argmax())
    witness=initial['witness'] if final else top if top in legal else max(legal,key=lambda a:(float(p[:,a].sum()),-a))
    constant=_trial(x,m,p,tuple(initial['support']) if final else (),witness,path,legal,arm=='C3',budget,initial)
    if final:
        if constant is None:raise ValueError('fixed final support did not converge with a valid certificate')
        return constant
    if constant is None:raise ValueError('constant incumbent could not be fitted/certified')
    candidates=[constant];seen={()};singles=[];pairs=[];triples=[]
    def attempt(s,parent):
        s=tuple(sorted(s))
        if s in seen or len(s)>3 or _support_cost(s)>3 or len(seen)>=8:return None
        seen.add(s)
        leaf=_trial(x,m,p,s,witness,path,legal,arm=='C3',budget,parent)
        if leaf is not None:candidates.append(leaf)
        return leaf
    for i in _rank(x,p,m,constant)[:2]:
        leaf=attempt((i,),constant)
        if leaf is not None:singles.append(leaf)
    for parent in sorted(singles,key=lambda q:_proxy(q,selection),reverse=True):
        for i in _rank(x,p,m,parent,parent['support']):
            if _support_cost([*parent['support'],i])<=3:
                leaf=attempt([*parent['support'],i],parent)
                if leaf is not None:pairs.append(leaf)
                break
    for parent in sorted(pairs,key=lambda q:_proxy(q,selection),reverse=True):
        for i in _rank(x,p,m,parent,parent['support']):
            if _support_cost([*parent['support'],i])<=3:
                leaf=attempt([*parent['support'],i],parent)
                if leaf is not None:triples.append(leaf)
                break
    if triples and len(seen)<8:
        parent=max(triples,key=lambda q:_proxy(q,selection))
        weakest=int(np.argmin(np.linalg.norm(np.asarray(parent['weights']),axis=0)))
        retained=[i for j,i in enumerate(parent['support']) if j!=weakest]
        for i in _rank(x,p,m,parent,parent['support']):
            if _support_cost([*retained,i])<=3:attempt([*retained,i],parent);break
    result=max(candidates,key=lambda q:(_proxy(q,selection),-len(q['support']),tuple(-i for i in q['support'])))
    result['selection_report']=dict(support_trials=len(seen),fit_internal_selection_only=True,
        candidate_supports=[q['support'] for q in candidates],proxy_score=list(_proxy(result,selection)))
    return result


def calibrate(leaf,x,m,p,*,arm,path,legal,budget):
    if arm in ('C0','C1'):
        return legacy.calibrate_leaf(leaf,x,m,p,FEATURE_NAMES,legal)[0]
    choices=[]
    for order,t in enumerate((.8,.9,1.,1.1,1.2)):
        trial=deepcopy(leaf)
        trial['bias']=(np.array(leaf['bias'],np.float32)/np.float32(t)).tolist()
        trial['weights']=(np.array(leaf['weights'],np.float32)/np.float32(t)).tolist()
        proof,_=certify(trial,path,legal,use_path=arm=='C3',budget=budget)
        if proof['passed']:
            trial['certificate']=proof
            _,q=evaluate_leaf(trial,x,m)
            loss=float(kl(p,q).mean()) if len(x) else abs(t-1)
            choices.append((loss,order,trial,t))
    if not choices:raise ValueError('no recertified temperature')
    _,_,result,t=min(choices,key=lambda a:a[:2]);result['temperature']=t
    return result


@dataclass
class Program:
    role:str
    actor_role_sha256:str
    structure:dict
    leaves:list
    arm:str
    maximum_leaves:int=32
    feedback_leaves:frozenset=frozenset()
    explanation_leaves:frozenset=frozenset()
    report:dict=None

    def leaf_indices(self,x):
        tree,info=structure_info(self.structure,self.maximum_leaves)
        return _route(tree,np.asarray(x,np.float32),info['depth'])

    def certificates(self):
        t,_=structure_info(self.structure,self.maximum_leaves)
        return certified_leaf_actions(t['split_weights'],t['split_bias'],t['left_refs'],t['right_refs'],
            len(t['split_bias']),t['leaf_count'],MASK_INDICES,explicit=True,wait_action=4)

    def distributions(self,x,m):
        x=validate_rows(x);m=np.asarray(m,np.float32)
        if (m.shape!=(len(x),5) or np.any((m!=0)&(m!=1)) or np.any(m[:,4]!=1)
                or not np.array_equal(x[:,MASK_INDICES],m)):
            raise ValueError('encoded legal bits and physical masks disagree')
        routes=self.leaf_indices(x)
        z,q=np.empty((len(x),5),np.float32),np.empty((len(x),5),np.float32)
        for i,leaf in enumerate(self.leaves):
            take=routes==i;z[take],q[take]=evaluate_leaf(leaf,x[take],m[take])
        return routes,z,q

    def validate(self):
        t,info=structure_info(self.structure,self.maximum_leaves)
        if self.arm not in ARMS or self.role not in ('robot_1','robot_2') or len(self.leaves)!=t['leaf_count']:
            raise ValueError('program identity/shape mismatch')
        proofs=self.certificates()
        for i,leaf in enumerate(self.leaves):
            if self.arm in ('C0','C1'):
                if legacy.certify_leaf(leaf,FEATURE_NAMES,proofs[i])!=leaf['certificate'] or not leaf['certificate']['passed']:
                    raise ValueError('invalid legacy control certificate')
                if self.arm=='C0' and leaf['support']:raise ValueError('variable constant control')
            else:
                verify(leaf,region(t,i,self.maximum_leaves),proofs[i],leaf['certificate'])
                if leaf['certificate']['use_path'] != (self.arm=='C3'):raise ValueError('wrong certificate language')
        for selected in (self.feedback_leaves,self.explanation_leaves):
            if not isinstance(selected,frozenset) or not selected<=set(range(len(self.leaves))):raise ValueError('invalid eligibility set')
        if self.report is not None:
            r=self.report
            if (r['domain_sha256']!=domain_sha() or r['actor_role_sha256']!=self.actor_role_sha256
                    or r['structure_sha256']!=digest(t) or r['external_program_selects_execution_action'] is not False):
                raise ValueError('lost model/domain/source evidence')
            for key in ('split_episode_ids','split_scenario_ids'):
                groups=r[key]
                if set(groups)!=set(legacy.SPLITS) or any(set(groups[a])&set(groups[b]) for a in groups for b in groups if a<b):
                    raise ValueError('evidence partitions overlap')
            for purpose,selected,threshold in (('feedback',self.feedback_leaves,.93),('explanation',self.explanation_leaves,.95)):
                actual=frozenset(i for i in range(len(self.leaves)) if r['fit_leaf_counts'][i] and _eligible(r['eligibility'][str(i)][purpose],threshold))
                if actual!=selected:raise ValueError('qualification differs from eligibility evidence')
        return self

    def to_dict(self):
        self.validate()
        return plain(dict(version=VERSION,role=self.role,actor_role_sha256=self.actor_role_sha256,
            structure=self.structure,leaves=self.leaves,arm=self.arm,maximum_leaves=self.maximum_leaves,
            feedback_leaves=sorted(self.feedback_leaves),explanation_leaves=sorted(self.explanation_leaves),
            domain_sha256=domain_sha(),feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            action_physics_config_sha256=ACTION_PHYSICS_CONFIG_SHA256,report=self.report))

    @classmethod
    def from_dict(cls,a,actor_role_sha256):
        keys={'version','role','actor_role_sha256','structure','leaves','arm','maximum_leaves',
            'feedback_leaves','explanation_leaves','domain_sha256','feature_schema_sha256','action_physics_config_sha256','report'}
        if (set(a)!=keys or a.get('version')!=VERSION or a.get('actor_role_sha256')!=actor_role_sha256
                or a.get('domain_sha256')!=domain_sha() or a.get('feature_schema_sha256')!=FEATURE_SCHEMA_SHA256
                or a.get('action_physics_config_sha256')!=ACTION_PHYSICS_CONFIG_SHA256):raise ValueError('unknown or mismatched program binding')
        for key in ('feedback_leaves','explanation_leaves'):
            if a[key]!=sorted(set(a[key])) or any(type(i) is not int for i in a[key]):raise ValueError('invalid serialized qualified IDs')
        return cls(a['role'],actor_role_sha256,a['structure'],a['leaves'],a['arm'],a['maximum_leaves'],
            frozenset(a['feedback_leaves']),frozenset(a['explanation_leaves']),a['report']).validate()


def metrics(program,rows,arrays,p,selected,purpose,predictions=None):
    return legacy._metrics(program,rows,arrays,p,selected,purpose,predictions=predictions)


def complete_program(program,parts,budget):
    """Only this finalizer reads cal/elig/validation, after topology is locked."""
    t,info=structure_info(program.structure,program.maximum_leaves);proofs=program.certificates()
    routes={s:program.leaf_indices(a['x']) for s,a in parts.items()}
    counts=np.bincount(routes['fit'],minlength=t['leaf_count']).tolist()
    r=dict(domain_sha256=domain_sha(),actor_role_sha256=program.actor_role_sha256,structure_sha256=digest(t),
        external_program_selects_execution_action=False,full_NN_distribution_targets=True,
        split_episode_ids={s:sorted({str(v['episode']) for v in a['rows']}) for s,a in parts.items()},
        split_scenario_ids={s:sorted({str(v['base_seed']) for v in a['rows']}) for s,a in parts.items()},
        fit_leaf_counts=counts,eligibility={},calibration_before=[])
    for i,leaf in enumerate(program.leaves):
        a=parts['fit'];take=routes['fit']==i
        fitted=fit_leaf(a['x'][take],a['m'][take],a['p'][take],arm=program.arm,
            path=region(t,i,program.maximum_leaves),legal=proofs[i],budget=budget,initial=leaf,final=True)
        r['calibration_before'].append(deepcopy(fitted))
        a=parts['calibration'];take=routes['calibration']==i
        program.leaves[i]=calibrate(fitted,a['x'][take],a['m'][take],a['p'][take],arm=program.arm,
            path=region(t,i,program.maximum_leaves),legal=proofs[i],budget=budget)
    e=parts['eligibility'];pred=program.distributions(e['x'],e['m']);chosen={'feedback':set(),'explanation':set()}
    arr=(e['x'],e['m'],e['sampled'])
    for i in range(len(program.leaves)):
        r['eligibility'][str(i)]={p:metrics(program,e['rows'],arr,e['p'],{i},p,pred) for p in chosen}
        for purpose,threshold in (('feedback',.93),('explanation',.95)):
            if counts[i] and _eligible(r['eligibility'][str(i)][purpose],threshold):chosen[purpose].add(i)
    program.feedback_leaves=frozenset(chosen['feedback']);program.explanation_leaves=frozenset(chosen['explanation'])
    program.report=r
    results={}
    for name,a in parts.items():
        budget.check();pred=program.distributions(a['x'],a['m']);arr=(a['x'],a['m'],a['sampled'])
        results[name]={p:metrics(program,a['rows'],arr,a['p'],sel,p,pred) for p,sel in (
            ('feedback',program.feedback_leaves),('explanation',program.explanation_leaves))}
        results[name]['gate']=gate_metrics(results[name],phase='development')
    program.validate()
    return program,results
