"""Versioned canonical (public threshold AND legal bit) program predicates.

These predicates never choose environment actions. A movement fact is granted
only after an exact bound proves the right branch impossible with its bit zero.
Historical guard readers and their evidence are unchanged.
"""
from fractions import Fraction
import numpy as np
from . import warehouse_v629_guards as previous
from .warehouse_v622_program import MASK_INDICES,digest
from .warehouse_v626_domain import domain_table,domain_sha,validate_rows
from .warehouse_v628_certificate import structure_info
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import HARD_SPLIT_TIE_EPSILON

VERSION='warehouse-canonical-threshold-and-legal-bit.v6.31'
KIND='canonical_public_threshold_and_legal_bit'

def rational(value):
    return dict(numerator=value.numerator,denominator=value.denominator)

def validate_guard_rows(x):
    rows=np.asarray(x,np.float32);validate_rows(rows)
    if rows.ndim!=2 or rows.shape[1]!=len(FEATURE_NAMES) or np.any((rows[:,MASK_INDICES]!=0)&(rows[:,MASK_INDICES]!=1)):
        raise ValueError('fused guard requires binary physical mask features')
    return rows

def _canonical(weights,bias):
    w=np.asarray(weights,np.float32);b=float(np.float32(bias))
    if w.shape!=(len(FEATURE_NAMES),) or not np.isfinite(w).all() or not np.isfinite(b):
        raise ValueError('invalid fused guard coefficients')
    used=np.flatnonzero(w).tolist();bits=[i for i in used if i in MASK_INDICES[:4]]
    if len(used)!=2 or len(bits)!=1:return None
    bit=bits[0];feature=next(i for i in used if i!=bit)
    if feature in MASK_INDICES or float(w[bit])!=4.:return None
    d=domain_table()['features'][feature];scale=float(np.float32(d['scale']))
    if d['constant'] or abs(float(w[feature]))!=scale:return None
    a=Fraction(float(w[feature]));b=Fraction(b);lower=Fraction(float(d['lower']));upper=Fraction(float(d['upper']))
    threshold=-(b+4)/a
    if not lower<threshold<upper:return None
    zero_upper=b+max(a*lower,a*upper)
    # Conservative FP64 dot/add error, including all zero coefficient entries.
    unit=Fraction(1,2**52);operations=4*len(FEATURE_NAMES)+16
    gamma=operations*unit/(1-operations*unit)
    error=gamma*(abs(b)+4+max(abs(a*lower),abs(a*upper)))
    if zero_upper+error>=Fraction(HARD_SPLIT_TIE_EPSILON):return None
    return dict(kind=KIND,action=MASK_INDICES.index(bit),feature=feature,
        comparison='greater' if a>0 else 'less',effective_raw_threshold=rational(threshold),
        zero_bit_upper_bound=rational(zero_upper),fp64_rounding_bound=rational(error),
        domain_sha256=domain_sha(),left_proven=[],right_proven=[MASK_INDICES.index(bit)])

def branch_proof(weights,bias):
    evidence=_canonical(weights,bias)
    if evidence is None:return previous.branch_proof(weights,bias)
    verify_branch(weights,bias,evidence)
    return evidence

def verify_branch(weights,bias,evidence):
    if evidence.get('kind')!=KIND:return previous.verify_branch(weights,bias,evidence)
    # Reconstruct from stored float32 coefficients and public domain constants;
    # supplied threshold/bound values never authorize a physical action.
    expected=_canonical(weights,bias)
    if expected is None or evidence!=expected:raise ValueError('fused legality evidence changed or overclaims a bit')
    return True

def predicate(action,feature,threshold,*,greater=True):
    if type(action) is not int or action not in range(4) or type(feature) is not int or not 0<=feature<len(FEATURE_NAMES):
        raise ValueError('invalid canonical action or public source')
    if feature in MASK_INDICES or type(greater) is not bool or not np.isfinite(threshold):
        raise ValueError('invalid public threshold clause')
    d=domain_table()['features'][feature]
    if d['constant'] or not d['lower']<threshold<d['upper']:raise ValueError('threshold outside nonconstant public domain')
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[MASK_INDICES[action]]=4.
    w[feature]=np.float32(d['scale'])*(1 if greater else -1)
    b=float(np.float32(-4.-float(w[feature])*float(threshold)))
    if branch_proof(w,b).get('kind')!=KIND:raise ValueError('stored threshold has no canonical legality margin')
    return tuple(map(float,w)),b

def tree_evidence(structure,maximum_leaves=256):
    t,_=structure_info(structure,maximum_leaves)
    return dict(version=VERSION,domain_sha256=domain_sha(),structure_sha256=digest(t),
        nodes=[branch_proof(w,b) for w,b in zip(t['split_weights'],t['split_bias'],strict=True)])

def leaf_actions(structure,maximum_leaves=256):
    t,_=structure_info(structure,maximum_leaves);e=tree_evidence(t,maximum_leaves);out=[None]*t['leaf_count']
    def walk(ref,proven):
        if ref<0:
            i=-1-ref
            if out[i] is not None:raise ValueError('duplicate fused proof leaf')
            out[i]=frozenset(proven);return
        p=e['nodes'][ref]
        walk(int(t['left_refs'][ref]),proven|set(p['left_proven']))
        walk(int(t['right_refs'][ref]),proven|set(p['right_proven']))
    walk(int(t['root_ref']),{4})
    if any(v is None for v in out):raise ValueError('unreachable fused proof leaf')
    return tuple(out)
