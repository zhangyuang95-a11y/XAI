from copy import deepcopy
from fractions import Fraction
import numpy as np
import pytest
from backend.training import warehouse_v631_guards as g,warehouse_v629_guards as old
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v622_program import MASK_INDICES
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v626_search import Node,encode
from backend.training.warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON

FEATURE=next(i for i,d in enumerate(domain_table()['features']) if not d['constant'] and i not in MASK_INDICES)

@pytest.mark.parametrize('action',range(4))
@pytest.mark.parametrize('greater',[False,True])
def test_false_bit_cannot_reach_claimed_branch(action,greater):
    d=domain_table()['features'][FEATURE];threshold=(d['lower']+d['upper'])/2
    w,b=g.predicate(action,FEATURE,threshold,greater=greater);p=g.branch_proof(w,b)
    assert p['right_proven']==[action] and not p['left_proven']
    # All movement-mask configurations and physical feature endpoints/interior.
    for bits in range(16):
        x=np.zeros((101,len(FEATURE_NAMES)),np.float32)
        x[:,FEATURE]=np.linspace(d['lower'],d['upper'],101,dtype=np.float32)
        x[:,MASK_INDICES]=np.array([*(int(bool(bits&(1<<a))) for a in range(4)),1],np.float32)
        right=split_scores(x,np.asarray([w],np.float32),np.asarray([b],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
        if not bits&(1<<action):assert not right.any()
        else:assert right.any() and not right.all()
    bound=p['zero_bit_upper_bound'];assert Fraction(bound['numerator'],bound['denominator'])<0
    assert old.branch_proof(w,b)['right_proven']==[]

def test_reject_tampered_bound_domain_and_action():
    d=domain_table()['features'][FEATURE];w,b=g.predicate(3,FEATURE,(d['lower']+d['upper'])/2)
    original=g.branch_proof(w,b)
    for key,value in [('right_proven',[0]),('domain_sha256','0'*64),('zero_bit_upper_bound',{'numerator':0,'denominator':1})]:
        evidence=deepcopy(original);evidence[key]=value
        with pytest.raises(ValueError):g.verify_branch(w,b,evidence)
    changed=list(w);changed[MASK_INDICES[3]]=3.
    assert g.branch_proof(changed,b)['right_proven']==[]

def test_legacy_proofs_and_tree_accounting_unchanged():
    w,b=old.predicate((0,2));assert g.branch_proof(w,b)==old.branch_proof(w,b)
    d=domain_table()['features'][FEATURE];w,b=g.predicate(3,FEATURE,(d['lower']+d['upper'])/2)
    tree=encode(Node(w,b))[0]
    assert tree['leaf_count']==2 and g.leaf_actions(tree)==(frozenset({4}),frozenset({3,4}))

def test_binary_mask_domain_is_mandatory():
    domains=domain_table()['features'];x=np.asarray([[(d['lower']+d['upper'])/2 for d in domains]],np.float32)
    x[:,MASK_INDICES]=[0,0,0,1,1];g.validate_guard_rows(x)
    x[0,MASK_INDICES[3]]=.5
    with pytest.raises(ValueError):g.validate_guard_rows(x)
