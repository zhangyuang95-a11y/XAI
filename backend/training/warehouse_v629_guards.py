"""Versioned physical-bit conjunction proofs for NON-EXECUTING programs.

Only canonical unit-weight conjunctions of two or three public legal bits
extend the old dedicated-bit language. All sixteen physical move-mask inputs
are independently replayed with exact rational arithmetic. No learned numeric
predicate is promoted into a legality fact. Historical readers are unchanged.
"""
from fractions import Fraction
from itertools import product
import numpy as np
from .warehouse_v622_program import MASK_INDICES, digest, plain
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import explicit_mask_branches, HARD_SPLIT_TIE_EPSILON
from .warehouse_v628_certificate import structure_info

VERSION = 'warehouse-canonical-legal-conjunction.v6.29'


def predicate(actions, *, negative=False):
    actions = tuple(sorted(actions))
    if (len(actions) not in (1, 2, 3) or len(set(actions)) != len(actions)
            or any(type(a) is not int or a not in range(4) for a in actions)):
        raise ValueError('legal guard needs one to three distinct movement bits')
    sign = -1 if negative else 1
    w = np.zeros(len(FEATURE_NAMES), np.float32)
    w[[MASK_INDICES[a] for a in actions]] = sign
    return tuple(map(float, w)), float(sign * -(len(actions) - .5))


def branch_proof(weights, bias):
    w = np.asarray(weights, np.float32)
    b = float(np.float32(bias))
    if w.shape != (len(FEATURE_NAMES),) or not np.isfinite(w).all() or not np.isfinite(b):
        raise ValueError('invalid legal guard input')
    used = np.flatnonzero(w).tolist()
    left, right = explicit_mask_branches(w, b, MASK_INDICES)
    kind, actions, sign = 'unrecognized_no_new_facts', [], 0
    if left or right:
        kind = 'legacy_dedicated_bit'
    elif (len(used) in (2, 3) and all(i in MASK_INDICES[:4] for i in used)
            and len(set(map(float, w[used]))) == 1 and abs(float(w[used[0]])) == 1):
        sign = int(w[used[0]])
        if b == -sign * (len(used) - .5):
            kind = 'canonical_conjunction'
            actions = sorted(MASK_INDICES.index(i) for i in used)
            left, right = (frozenset(actions), frozenset()) if sign < 0 else (frozenset(), frozenset(actions))
    result = dict(kind=kind, actions=actions, sign=sign,
        left_proven=sorted(left), right_proven=sorted(right))
    verify_branch(w, b, result)
    return result


def verify_branch(weights, bias, evidence):
    """Reconstruct evidence independently; no solver or statistical tolerance.

    Canonical scores are exact half-integers under the shared FP64 router, so
    their distance from its tie epsilon is at least 0.4999999999. Other splits
    grant no additional actions, even if a stronger proof might be possible.
    """
    w = np.asarray(weights, np.float32)
    used = np.flatnonzero(w).tolist()
    expected = dict(kind='unrecognized_no_new_facts', actions=[], sign=0,
        left_proven=[], right_proven=[])
    old = explicit_mask_branches(w, bias, MASK_INDICES)
    if any(old):
        expected.update(kind='legacy_dedicated_bit', left_proven=sorted(old[0]), right_proven=sorted(old[1]))
    else:
        recognized = False
        for sign in (-1, 1):
            if (len(used) in (2, 3) and set(used) <= set(MASK_INDICES[:4])
                    and all(float(w[i]) == sign for i in used)
                    and Fraction(float(bias)) == -sign * (Fraction(len(used)) - Fraction(1, 2))):
                recognized = True
                expected.update(kind='canonical_conjunction', sign=sign,
                    actions=sorted(MASK_INDICES.index(i) for i in used))
        if recognized:
            branch_masks = {0: [], 1: []}
            for bits in product((0, 1), repeat=4):
                score = Fraction(float(bias)) + sum(
                    Fraction(float(w[MASK_INDICES[a]])) * bits[a] for a in range(4))
                branch_masks[int(score > Fraction(HARD_SPLIT_TIE_EPSILON))].append(bits)
            for branch, key in ((0, 'left_proven'), (1, 'right_proven')):
                rows = branch_masks[branch]
                expected[key] = [a for a in range(4) if rows and all(bits[a] for bits in rows)]
    if evidence != expected:
        raise ValueError('canonical legal guard proof changed or overclaims a bit')
    return True


def tree_evidence(structure, maximum_leaves=64):
    t, _ = structure_info(structure, maximum_leaves)
    return dict(version=VERSION, structure_sha256=digest(t),
        nodes=[branch_proof(w, b) for w, b in zip(t['split_weights'], t['split_bias'], strict=True)])


def leaf_actions(structure, maximum_leaves=64):
    t, _ = structure_info(structure, maximum_leaves)
    evidence = tree_evidence(t, maximum_leaves)
    result = [None] * t['leaf_count']
    def walk(ref, proven):
        if ref < 0:
            i = -1 - ref
            if result[i] is not None:
                raise ValueError('duplicate leaf in physical proof')
            result[i] = frozenset(proven)
            return
        e = evidence['nodes'][ref]
        walk(int(t['left_refs'][ref]), proven | set(e['left_proven']))
        walk(int(t['right_refs'][ref]), proven | set(e['right_proven']))
    walk(int(t['root_ref']), {4})
    if any(v is None for v in result):
        raise ValueError('unreachable physical proof leaf')
    return tuple(result)
