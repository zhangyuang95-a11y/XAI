"""Depth-declared independent forward, exact dual and eligibility replay."""
import numpy as np
from experiments.warehouse import v630_protocol as v
from backend.training.warehouse_v642_certificate import region
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v6_tree_routing import leaf_indices
from backend.training.warehouse_v622_program import masked_probabilities
from scripts.audit_warehouse_v626 import basic,eligible,same_metrics,independent_dual

def forward(p,x,m):
    t=p.structure;n=len(t['split_bias']);d=domain_table()['features']
    route=leaf_indices(x,t['split_weights'],t['split_bias'],t['left_refs'],t['right_refs'],n,p.maximum_depth)
    logits=np.empty((len(x),5),np.float32)
    for i,leaf in enumerate(p.leaves):
        take=route==i;z=np.broadcast_to(np.array(leaf['bias'],np.float32),(int(take.sum()),5)).copy()
        weights=np.array(leaf['weights'],np.float32)
        for j,index in enumerate(leaf['support']):
            column=np.multiply(np.subtract(x[take,index],np.float32(d[index]['center']),dtype=np.float32),np.float32(d[index]['scale']),dtype=np.float32)
            z=np.add(z,np.multiply(column[:,None],weights[None,:,j],dtype=np.float32),dtype=np.float32)
        logits[take]=z
    return route,logits,masked_probabilities(logits,m)


def audited_program(program,parts,expected):
    pred={split:forward(program,a['x'],a['m']) for split,a in parts.items()}
    for split,a in parts.items():
        direct=program.distributions(a['x'],a['m'])
        v.require(all(np.array_equal(x,y) for x,y in zip(pred[split],direct,strict=True)),'v630 independent forward differs')
        for purpose,selected in [('feedback',program.feedback_leaves),('explanation',program.explanation_leaves)]:
            same_metrics(basic(program,a,pred[split],selected,purpose),expected[split][purpose])
    for purpose,selected,threshold in [('feedback',program.feedback_leaves,.93),('explanation',program.explanation_leaves,.95)]:
        qualified=set()
        for leaf in range(len(program.leaves)):
            actual=basic(program,parts['eligibility'],pred['eligibility'],{leaf},purpose)
            same_metrics(actual,program.report['eligibility'][str(leaf)][purpose])
            if program.report['fit_leaf_counts'][leaf] and eligible(actual,threshold):qualified.add(leaf)
        v.require(qualified==set(selected),'refined child eligibility does not reproduce')
    for leaf,model in enumerate(program.leaves):independent_dual(model,region(program.structure,leaf,program.maximum_leaves,program.maximum_depth),program.certificates()[leaf])
    return pred

