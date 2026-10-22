"""Exact public interval proofs for (physical bit AND learned context).

These are surrogate predicates only. One movement bit plus at most two
public source units can certify the right branch in a single counted node.
No physical fact comes from fitted labels, solver status or sampled states.
"""
from fractions import Fraction as F
import math
import numpy as np
from . import warehouse_v629_guards as old
from .warehouse_v626_domain import domain_table,domain_sha,validate_rows
from .warehouse_v626_program import _support_cost
from .warehouse_v622_program import MASK_INDICES,digest
from .warehouse_v628_certificate import structure_info
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import HARD_SPLIT_TIE_EPSILON

VERSION='warehouse-bounded-mask-public-linear-guard.v6.35'
KIND='bounded_mask_and_public_linear_context'


def pair(v):return [v.numerator,v.denominator]


def evidence(weights,bias):
    w=np.asarray(weights,np.float32);b=float(np.float32(bias))
    if w.shape!=(len(FEATURE_NAMES),) or not np.isfinite(w).all() or not np.isfinite(b):raise ValueError('invalid guard coefficients')
    legacy=old.branch_proof(w,b)
    if legacy['kind']!='unrecognized_no_new_facts':return legacy
    used=np.flatnonzero(w).tolist();bits=[i for i in used if i in MASK_INDICES]
    if len(bits)!=1 or bits[0] not in MASK_INDICES[:4] or w[bits[0]]<=0 or _support_cost(used)>3:return legacy
    bit=bits[0];context=[i for i in used if i!=bit];d=domain_table()['features']
    if not context or any(d[i]['constant'] for i in context):return legacy
    upper=F(b);magnitude=abs(F(b))+abs(F(float(w[bit])))
    for i in context:
        lo,hi=F(d[i]['lower']),F(d[i]['upper']);a=F(float(w[i]))
        upper+=max(a*lo,a*hi);magnitude+=max(abs(a*lo),abs(a*hi))
    operations=4*len(FEATURE_NAMES)+16;u=F(1,2**52);error=operations*u/(1-operations*u)*magnitude
    if upper+error>=F(HARD_SPLIT_TIE_EPSILON):return legacy
    return dict(kind=KIND,action=MASK_INDICES.index(bit),context_indices=context,
        zero_mask_upper=pair(upper),fp64_error_bound=pair(error),effective_context_bias=pair(F(b)+F(float(w[bit]))),
        domain_sha256=domain_sha(),left_proven=[],right_proven=[MASK_INDICES.index(bit)])


def fuse(action,context_weights,context_bias):
    if type(action) is not int or action not in range(4):raise ValueError('invalid movement action')
    w=np.asarray(context_weights,np.float32).copy();b=float(np.float32(context_bias));used=np.flatnonzero(w).tolist()
    if w.shape!=(len(FEATURE_NAMES),) or not np.isfinite(w).all() or not np.isfinite(b) or set(used)&set(MASK_INDICES) or _support_cost(used)>2:
        raise ValueError('context exceeds two public sources or contains physical bits')
    if not used:
        # A physical-only split is still counted and uses the old exact proof.
        return old.predicate([action]) if b>HARD_SPLIT_TIE_EPSILON else (tuple(map(float,w)),b)
    d=domain_table()['features'];magnitude=abs(b)+sum(max(abs(float(w[i])*d[i]['lower']),abs(float(w[i])*d[i]['upper'])) for i in used)
    big=float(2**max(1,int(math.ceil(math.log2(2*magnitude+2)))))
    w[MASK_INDICES[action]]=np.float32(big);bias=float(np.float32(b-big))
    if evidence(w,bias).get('kind')!=KIND:raise ValueError('stored fused node cannot prove physical legality')
    return tuple(map(float,w)),bias


def validate_rows_with_masks(x):
    x=validate_rows(x)
    if np.any((x[:,MASK_INDICES]!=0)&(x[:,MASK_INDICES]!=1)):raise ValueError('guard requires binary physical bits')
    return x


def tree_evidence(structure,maximum=256):
    tree,_=structure_info(structure,maximum)
    return dict(version=VERSION,domain_sha256=domain_sha(),structure_sha256=digest(tree),
                nodes=[evidence(w,b) for w,b in zip(tree['split_weights'],tree['split_bias'],strict=True)])


def leaf_actions(structure,maximum=256):
    tree,_=structure_info(structure,maximum);nodes=tree_evidence(tree,maximum)['nodes'];out=[None]*tree['leaf_count']
    def walk(ref,known):
        if ref<0:
            i=-1-ref
            if out[i] is not None:raise ValueError('duplicate proof leaf')
            out[i]=frozenset(known);return
        row=nodes[ref];walk(int(tree['left_refs'][ref]),known|set(row['left_proven']));walk(int(tree['right_refs'][ref]),known|set(row['right_proven']))
    walk(int(tree['root_ref']),{4})
    if any(x is None for x in out):raise ValueError('unreachable proof leaf')
    return tuple(out)
