"""Physical proof and version/target binding checks; no environment rollout."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import numpy as np
import pytest
from backend.training import warehouse_v635_guards as g,warehouse_v635_program as p
from backend.training.warehouse_v630_program import Program as Old
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from experiments.warehouse import v630_protocol as v


def test_all_movement_bits_two_context_signs_and_public_box_corners_are_safe():
    d=domain_table()['features'];ids=[i for i,a in enumerate(d) if not a['constant'] and i not in g.MASK_INDICES and
        g._support_cost([i])==1][:2];assert len(ids)==2
    for action in range(4):
        for signs in product((-1,1),repeat=2):
            w=np.zeros(len(d),np.float32)
            for j,sign in zip(ids,signs):w[j]=np.float32(sign*d[j]['scale']*.7)
            weights,bias=g.fuse(action,w,-.13);proof=g.evidence(weights,bias)
            assert proof['kind']==g.KIND and proof['right_proven']==[action] and not proof['left_proven']
            upper=F(*proof['zero_mask_upper']);error=F(*proof['fp64_error_bound'])
            assert upper+error<F(HARD_SPLIT_TIE_EPSILON)
            rows=np.array([[x['center'] for x in d] for _ in range(8)],np.float32);rows[:,g.MASK_INDICES]=1.
            for k,(bit,values) in enumerate(product((0,1),product(*[(d[j]['lower'],d[j]['upper']) for j in ids]))):
                rows[k,g.MASK_INDICES[action]]=bit;rows[k,ids]=values
            scores=split_scores(rows,np.array(weights,np.float32)[None],np.array([bias],np.float32))[:,0]
            assert np.all(scores[rows[:,g.MASK_INDICES[action]]==0]<=HARD_SPLIT_TIE_EPSILON)
            # Independent rational corner max reconstructs the zero-bit bound.
            exact=F(float(np.float32(bias)))+sum(max(F(float(weights[j]))*F(d[j]['lower']),F(float(weights[j]))*F(d[j]['upper'])) for j in ids)
            assert exact==upper


def test_source_overflow_nonphysical_context_and_unsafe_changed_bias_are_rejected():
    d=domain_table()['features'];ids=[i for i,a in enumerate(d) if not a['constant'] and i not in g.MASK_INDICES and g._support_cost([i])==1][:3]
    w=np.zeros(len(d),np.float32);w[ids]=1.
    with pytest.raises(ValueError):g.fuse(0,w,0.)
    w[:]=0.;w[g.MASK_INDICES[1]]=1.
    with pytest.raises(ValueError):g.fuse(0,w,0.)
    w[:]=0.;w[ids[:2]]=1.;weights,bias=g.fuse(0,w,0.)
    assert g.evidence(weights,1000.)['kind']!=g.KIND
    x=np.array([[a['center'] for a in d]],np.float32);x[:,g.MASK_INDICES]=1.;x[:,g.MASK_INDICES[0]]=.5
    with pytest.raises(ValueError):g.validate_rows_with_masks(x)


def test_real_program_import_preserves_q_and_keeps_anchor_separate_from_target_hash():
    _,plan=v.checked_plan();root=v.ROOT/'output/warehouse_v633/warehouse-v633-support-refresh-01'
    r=v.bound(v.pointer(root/'fold-0-robot_2/result.json'));old=Old.from_dict(v.bound(r['program']),plan['actor_role_sha256']['robot_2'])
    new=p.import_previous(old);ref=v.bound(plan['collection'])['cases']['dev-0300']['arrays']
    with np.load(ref['path']) as z:x=z['x'][:,1];m=z['m'][:,1]
    a=old.distributions(x,m);b=new.distributions(x,m)
    assert all(np.array_equal(i,j) for i,j in zip(a,b))
    asset=new.to_dict();assert p.Program.from_dict(asset,old.actor_role_sha256).to_dict()==asset
    assert new.report is None and not new.feedback_leaves and not new.explanation_leaves
    with pytest.raises((ValueError,RuntimeError)):p.Program.from_dict(asset,'0'*64)
    with pytest.raises((ValueError,RuntimeError)):Old.from_dict(asset,old.actor_role_sha256)
    damaged=deepcopy(asset);damaged['guard_evidence']['structure_sha256']='0'*64
    with pytest.raises((ValueError,RuntimeError)):p.Program.from_dict(damaged,old.actor_role_sha256)
    # A later current NN has its own binding; it does not rewrite the frozen
    # original anchor or authorize fidelity without a new evidence owner.
    new.target_actor_role_sha256='a'*64;new.validate();changed=new.to_dict()
    assert p.Program.from_dict(changed,'a'*64).actor_role_sha256==old.actor_role_sha256
    with pytest.raises((ValueError,RuntimeError)):p.Program.from_dict(changed,old.actor_role_sha256)
