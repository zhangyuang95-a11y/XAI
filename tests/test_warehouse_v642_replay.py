"""Stored context training arithmetic replay rejects falsified evidence."""
from copy import deepcopy
import numpy as np
import pytest
from backend.training import warehouse_v642_contexts as c
from backend.training.warehouse_v622_program import MASK_INDICES,plain
from scripts.audit_warehouse_v642 import context_evidence


def test_recorded_entropy_guards_and_exposure_replay():
    class Budget:
        def check(self):pass
        def solve(self,kind,metadata,callback):return callback()
    domain=c.domain_table()['features'];i=next(i for i,d in enumerate(domain) if not d['constant'] and d['lower']==0 and d['upper']==1 and i not in MASK_INDICES and c._support_cost([i])==1)
    x=np.array([[d['center'] for d in domain]]*480,np.float32);x[:,MASK_INDICES]=1;x[:,i]=np.linspace(0,1,480)
    m=np.ones((480,5),np.float32);p=np.zeros_like(m);p[:,0]=.02;p[:,4]=.02;p[np.arange(480),np.where(x[:,i]>.5,0,4)]=.98
    a=dict(x=x,m=m,p=p);tree,record=c.learn_tree(a,{4},3,Budget(),context_depth=2,root=(),bank=[])
    option=dict(structure=plain(c.encode(tree)[0]));result=context_evidence(a,[],record,option,{4},3)
    assert result==dict(context_nodes=1,physical_guard_nodes=1,node_search_calls=1)
    tampered=deepcopy(record);tampered['records'][0]['entropy_after']+=.001
    with pytest.raises(RuntimeError):context_evidence(a,[],tampered,option,{4},3)
