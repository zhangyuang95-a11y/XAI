"""Small numerical end-to-end synthetic root, no rollout or admission claim."""
from copy import deepcopy
import numpy as np
from backend.training import warehouse_v635_program as language,warehouse_v635_subtrees as s
from backend.training.warehouse_v630_program import Program as Previous
from scripts.diagnose_warehouse_v634_free_roots import free_roots
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import independent_dual
from experiments.warehouse import v630_protocol as v


class Budget:
    def __init__(self):self.counts=dict(node=0,leaf=0,proof=0)
    def check(self):pass
    def solve(self,kind,metadata,fn):self.counts[kind]+=1;return fn()


def test_learn_fit_and_assemble_preserves_core_and_proves_new_movement_leaf():
    _,plan=v.checked_plan();folder=v.ROOT/'output/warehouse_v633/warehouse-v633-support-refresh-01'
    r=v.bound(v.pointer(folder/'fold-0-robot_2/result.json'))
    old=Previous.from_dict(v.bound(r['program']),plan['actor_role_sha256']['robot_2']);source=language.import_previous(old)
    roots,paths=free_roots(old);ref=v.bound(plan['collection'])['cases']['dev-0300']['arrays']
    with np.load(ref['path']) as z:x=z['x'][:,1].copy();m=z['m'][:,1].copy()
    route=source.leaf_indices(x);choice=None
    for root in roots:
        if len(root)>6:continue
        ids=[i for i,p in enumerate(paths) if p[:len(root)]==root];rows=np.flatnonzero(np.isin(route,ids))
        if not len(rows):continue
        path=s.region(source.structure,ids[0],256)['path'][:len(root)]
        known={4}
        for item in path:known.update(s.guards.evidence(item['weight'],item['bias'])[item['branch']+'_proven'])
        actions=[a for a in range(4) if a not in known and m[rows[0],a]==1]
        if actions:choice=(root,rows[0],path,known,actions[0]);break
    assert choice is not None
    root,index,path,known,action=choice;used={j for item in path for j,w in enumerate(item['weight']) if w}
    d=s.domain_table()['features'];feature=next(i for i,a in enumerate(d) if i not in used and i not in s.guards.MASK_INDICES
        and a['lower']==0 and a['upper']==1 and s._support_cost([i])==1)
    xx=np.repeat(x[index:index+1],240,axis=0);mm=np.repeat(m[index:index+1],240,axis=0)
    xx[:,feature]=np.linspace(0,1,240,dtype=np.float32);target=np.zeros((240,5),np.float32)
    top=np.where(xx[:,feature]>.5,action,4);target[:,4]=.06;target[:,action]=.06;target[np.arange(240),top]=.94
    parts={}
    for split in ('fit','calibration','eligibility','validation'):
        parts[split]=dict(x=xx.copy(),m=mm.copy(),p=target.copy(),sampled=top.astype(int),
            rows=[dict(episode=f'{split}-{i//10}',base_seed=f'{split}-{i//10}',task=i//80+1) for i in range(240)])
    budget=Budget();template,record=s.learn_tree(parts['fit'],known,9-len(root),budget,precision=.99,root=root)
    assert template is not None and record['records'] and record['records'][0]['action']==action
    assert action in s.guards.evidence(template.weight,template.bias)['right_proven']
    baseline=source.to_dict();option=s.fit_option(source,root,template,parts,budget)
    out=s.assemble(source,{root:option});assert source.to_dict()==baseline
    language.qualify(out,parts)
    assert language.Program.from_dict(out.to_dict(),source.target_actor_role_sha256).to_dict()==out.to_dict()
    pred=out.distributions(xx,mm);assert np.mean(pred[2].argmax(1)==top)>.99
    assert all(np.array_equal(a,b) for a,b in zip(pred,forward(out,xx,mm)))
    before=source.distributions(x,m);after=out.distributions(x,m)
    selected=np.isin(route,[i for i,p in enumerate(paths) if p[:len(root)]==root])
    assert np.array_equal(before[1][~selected],after[1][~selected]) and np.array_equal(before[2][~selected],after[2][~selected])
    for i,model in enumerate(out.leaves):independent_dual(model,s.region(out.structure,i,256),out.certificates()[i])
    assert budget.counts['node']>0 and budget.counts['leaf']>0
