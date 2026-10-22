"""Sparse borrowed geometry, soft-target routing, and real frozen root scope."""
import numpy as np
from backend.training import warehouse_v641_contexts as c,warehouse_v641_search as s,warehouse_v635_program as language
from backend.training.warehouse_v622_program import MASK_INDICES,_route

class Budget:
    def __init__(self):self.counts={'node':0}
    def check(self):pass
    def solve(self,kind,metadata,callback):self.counts[kind]+=1;return callback()


def features():
    d=c.domain_table()['features'];ids=[i for i,a in enumerate(d) if not a['constant'] and a['lower']==0 and a['upper']==1 and i not in MASK_INDICES and c._support_cost([i])==1][:2]
    return d,ids


def test_borrowed_logit_direction_is_public_sparse_affine_proposal():
    d,ids=features();model=dict(support=ids,bias=[.3,-.3,0,0,0],weights=[[1.5,-.5],[-1.5,.5],[0,0],[0,0],[0,0]])
    direction=next(v for v in c.logit_directions(model) if v['actions']==[0,1]);x=np.repeat(np.array([a['center'] for a in d],np.float32)[None],160,axis=0)
    x[:,ids[0]]=np.linspace(0,1,160);x[:,ids[1]]=np.linspace(1,0,160)
    actual=c.split_scores(x,np.asarray(direction['weight'],np.float32)[None],np.array([direction['bias']],np.float32))[:,0]
    expected=.6+3*(x[:,ids[0]]-d[ids[0]]['center'])*d[ids[0]]['scale']-(x[:,ids[1]]-d[ids[1]]['center'])*d[ids[1]]['scale']
    bias=.6-3*d[ids[0]]['center']*d[ids[0]]['scale']+d[ids[1]]['center']*d[ids[1]]['scale']
    scale=max(1.,3*d[ids[0]]['scale'],d[ids[1]]['scale'],abs(bias))
    assert np.allclose(scale*actual,expected,atol=2e-6) and c._support_cost(np.flatnonzero(direction['weight']))<=3


def test_soft_NN_context_and_terminal_guard_count_all_leaves():
    d,ids=features();x=np.repeat(np.array([a['center'] for a in d],np.float32)[None],480,axis=0);x[:,MASK_INDICES]=1
    x[:,ids[0]]=np.linspace(0,1,480,dtype=np.float32);m=np.ones((480,5),np.float32);nn=np.where(x[:,ids[0]]>.5,0,4)
    p=np.zeros_like(m);p[:,0]=.02;p[:,4]=.02;p[np.arange(480),nn]=.98
    budget=Budget();tree,record=c.learn_tree(dict(x=x,m=m,p=p),{4},3,budget,context_depth=2,root=(),bank=[])
    structure,_=c.encode(tree);t,info=c.structure_info(c.plain(structure),256);route=_route(t,x,info['depth']);known=c.guards.leaf_actions(structure,256)
    assert len(record['records'])==1 and record['records'][0]['entropy_after']<record['records'][0]['entropy_before']
    assert budget.counts['node']==1 and info['depth']==2 and t['leaf_count']==3
    assert all(int(a) in known[int(i)] for a,i in zip(nn,route,strict=True)) and record['source_actions_never_executed']


def test_real_frozen_core_reliable_E_and_free_roots_form_disjoint_cover():
    from experiments.warehouse import v630_protocol as v
    from scripts.diagnose_warehouse_v629_qualification_stability import sampled_fit
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    folder=v.ROOT/'output/warehouse_v640/warehouse-v640-alternating-regions-01';r=v.bound(v.pointer(folder/'fold-0-robot_2/result.json'))
    source=language.Program.from_dict(v.bound(r['program']),old['actor_role_sha256']['robot_2']);before=v.digest(source.to_dict())
    cases,a,_=sampled_fit(dp,v.bound(prior['collection']),0,'robot_2');lanes=s.old.metrics.lane_parts(a,cases)
    fixed,roots,paths,_=s.scan(source,lanes);covered=set(fixed)
    for root in roots:
        ids={i for i,p in enumerate(paths) if p[:len(root)]==root};assert ids and not covered&ids;covered|=ids
    assert covered==set(range(len(source.leaves)))
    head=v.bound(v.pointer(v.ROOT/'output/warehouse_v640/diagnostics/action-headroom-20261022-01/result.json'))
    item=next(x for x in head['units'] if x['job']=='fold-0-robot_2');expected=set(language.protection(source)['protected_descendants'])|{x['leaf'] for x in item['leaves'] if x['reliable']}
    assert set(fixed)==expected and v.digest(source.to_dict())==before
