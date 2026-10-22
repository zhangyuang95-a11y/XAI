"""Check the fused primitive across every versioned public feature domain."""
from fractions import Fraction
import numpy as np
import pytest
from backend.training import warehouse_v631_guards as g
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v622_program import MASK_INDICES
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON

@pytest.mark.parametrize('greater',[False,True])
def test_all_public_domains_have_exact_false_bit_margin(greater):
    ds=domain_table()['features'];count=0
    for feature,d in enumerate(ds):
        if d['constant'] or feature in MASK_INDICES:continue
        w,b=g.predicate(3,feature,(d['lower']+d['upper'])/2,greater=greater)
        p=g.branch_proof(w,b);count+=1
        # Independently reconstruct the zero-bit maximum at the two endpoints.
        a=Fraction(float(np.float32(w[feature])));bias=Fraction(float(np.float32(b)))
        upper=max(bias+a*Fraction(float(v)) for v in (d['lower'],d['upper']))
        stored=p['zero_bit_upper_bound'];assert upper==Fraction(stored['numerator'],stored['denominator'])
        error=p['fp64_rounding_bound'];assert upper+Fraction(error['numerator'],error['denominator'])<Fraction(HARD_SPLIT_TIE_EPSILON)
        base=np.asarray([[(v['lower']+v['upper'])/2 for v in ds]],np.float32)
        x=np.repeat(base,5,axis=0);x[:,feature]=np.linspace(d['lower'],d['upper'],5,dtype=np.float32)
        for bit in (0,1):
            x[:,MASK_INDICES]=[1,1,1,bit,1];g.validate_guard_rows(x)
            right=split_scores(x,np.asarray([w],np.float32),np.asarray([b],np.float32))[:,0]>HARD_SPLIT_TIE_EPSILON
            if bit==0:assert not right.any()
            else:assert right.any() and not right.all()
    assert count>100

def test_public_domain_assumption_cannot_be_bypassed():
    ds=domain_table()['features'];feature=next(i for i,d in enumerate(ds) if not d['constant'] and i not in MASK_INDICES)
    x=np.asarray([[(d['lower']+d['upper'])/2 for d in ds]],np.float32);x[:,MASK_INDICES]=[1,1,1,1,1]
    x[0,feature]=ds[feature]['upper']+100
    with pytest.raises(ValueError):g.validate_guard_rows(x)
