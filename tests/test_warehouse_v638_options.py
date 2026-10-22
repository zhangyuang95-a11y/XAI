"""Exact real-asset prefix extension; no coefficient optimization or rollout."""
import numpy as np
from backend.training import warehouse_v638_joint_options as s
from experiments.warehouse import v630_protocol as v
from scripts.audit_warehouse_v626 import independent_dual
from scripts.audit_warehouse_v628 import forward


def test_joint_partition_reuses_real_calibrated_outputs_with_exact_proofs():
    _,p=v.checked_plan();folder=v.ROOT/'output/warehouse_v636/warehouse-v636-context-first-01'
    r=v.bound(v.pointer(folder/'fold-0-robot_2/result.json'));source=s.language.Program.from_dict(v.bound(r['program']),p['actor_role_sha256']['robot_2'])
    core=set(s.language.protection(source)['protected_descendants']);_,paths=s.encode(s.decode(source.structure))
    i=next(i for i,path in enumerate(paths) if i not in core and len(path)<9)
    d=s.build.domain_table()['features'];feature=next(i for i,a in enumerate(d) if not a['constant'] and a['lower']==0 and a['upper']==1)
    w=np.zeros(len(d),np.float32);w[feature]=1.
    subtree=s.build.Node(tuple(map(float,w)),-.5,None,None)
    option=s.compile_option(source,i,source.leaves[i],subtree);out=s.build.assemble(source,{paths[i]:option})
    assert len(out.leaves)==len(source.leaves)+1
    for model,path in zip(option['models'],option['paths'],strict=True):independent_dual(model,path['path'],path['legal'])
    ref=v.bound(p['collection'])['cases']['dev-0300']['arrays']
    with np.load(ref['path']) as z:x=z['x'][:,1];m=z['m'][:,1]
    a=source.distributions(x,m);b=out.distributions(x,m)
    assert np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2])
    assert all(np.array_equal(x,y) for x,y in zip(b,forward(out,x,m)))
