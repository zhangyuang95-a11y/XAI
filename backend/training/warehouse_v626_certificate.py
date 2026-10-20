"""Path-conditioned linear-leaf proofs with independently checkable duals.

LP is a proposal engine only. A nonnegative dual plus a box-bounded residual
is checked using exact rationals, including stored normalization constants.
No LP success/infeasibility flag is accepted as a proof by itself.
"""
from __future__ import annotations
from fractions import Fraction as F
from itertools import product
import numpy as np
from scipy.optimize import linprog
from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import HARD_SPLIT_TIE_EPSILON
from .warehouse_v622_program import normalize_structure, digest
from .warehouse_v626_domain import domain_table, domain_sha

VERSION = 'warehouse-path-public-polytope-farkas.v6.26'
MARGIN = .001


def fraction(x): return F.from_float(float(x))


def structure_info(structure, maximum_leaves=32):
    if type(maximum_leaves) is not int or maximum_leaves not in (32, 64): raise ValueError('unknown versioned capacity')
    tree, info = normalize_structure(structure, enforce_limits=False)
    if info['leaf_count'] > maximum_leaves or info['depth'] > (6 if maximum_leaves == 32 else 7) or info['maximum_source_features_per_node'] > 3:
        raise ValueError('program exceeds bound depth/leaf/source limits')
    return tree, info


def region(tree, leaf, maximum_leaves=32):
    tree, info = structure_info(tree, maximum_leaves)
    return dict(path=[dict(branch=p['branch'], weight=tree['split_weights'][p['node']].tolist(),
        bias=float(tree['split_bias'][p['node']])) for p in info['leaf_paths'][leaf]],
        tree_sha256=digest(tree), leaf=leaf)


def coefficient_data(leaf):
    if leaf.get('version')!='warehouse-role-program.v6.26-domain-complete':raise ValueError('unknown normalized leaf language')
    support = leaf['support']
    if (not isinstance(support, list) or support != sorted(set(support)) or len(support) > 3
            or any(type(i) is not int or not 0 <= i < len(FEATURE_NAMES) for i in support)
            or sum(2 if FEATURE_NAMES[i] == 'public_hard_risk_context' else 1 for i in support) > 3):
        raise ValueError('invalid sparse source support')
    b, w = np.asarray(leaf['bias'], np.float32), np.asarray(leaf['weights'], np.float32)
    if not support and w.shape == (0,): w = w.reshape(5, 0)
    if (b.shape != (5,) or w.shape != (5, len(support)) or not np.isfinite(b).all() or not np.isfinite(w).all()
            or np.max(np.abs(b), initial=0) > 40.00001 or np.max(np.abs(w), initial=0) > 40.00001
            or not np.array_equal(np.asarray(leaf['bias'], np.float64), b.astype(np.float64))
            or not np.array_equal(np.asarray(leaf['weights'], np.float64), w.astype(np.float64))):
        raise ValueError('stored coefficients must be exact bounded float32 values')
    if leaf.get('domain_sha256') != domain_sha() or leaf.get('normalized') is not True:
        raise ValueError('unknown domain or normalized arithmetic')
    return support, b, w


def forward_error(leaf):
    support, b, w = coefficient_data(leaf)
    d = domain_table()['features']; eps = np.finfo(np.float32).eps
    mag, extra = np.abs(b.astype(float)), np.zeros(5)
    for j, i in enumerate(support):
        item = d[i]; bound = max(abs(item['lower']), abs(item['upper']))
        v = (bound + abs(item['center'])) * item['scale']
        en = (2*eps/(1-2*eps))*v + 4*float(np.finfo(np.float32).tiny)
        mag += np.abs(w[:, j].astype(float))*(v+en)
        extra += np.abs(w[:, j].astype(float))*en
    n = 2*len(support)+2
    return np.nextafter(extra+(n*eps/(1-n*eps))*mag+n*float(np.finfo(np.float32).tiny), np.inf) if support else np.zeros(5)


def polynomial(leaf, action, witness, indices):
    support, b, w = coefficient_data(leaf)
    d = domain_table()['features']; out = {i:F(0) for i in indices}
    constant = fraction(b[action])-fraction(b[witness])
    for j, i in enumerate(support):
        value = (fraction(w[action,j])-fraction(w[witness,j]))*fraction(d[i]['scale'])
        out[i] += value; constant -= value*fraction(d[i]['center'])
    return [out[i] for i in indices], constant


def public_polytope(leaf, path, use_path):
    used = set(leaf['support'])
    if use_path:
        for item in path['path']: used.update(np.flatnonzero(item['weight']).tolist())
    indices = sorted(used); d = domain_table()['features']
    bounds = [(d[i]['lower'],d[i]['upper']) for i in indices]
    A, rhs = [], []
    if use_path:
        for item in path['path']:
            weight=np.asarray(item['weight'],np.float32)
            magnitude = abs(item['bias'])+sum(abs(float(weight[i]))*max(abs(d[i]['lower']),abs(d[i]['upper'])) for i in np.flatnonzero(weight))
            # Conservative fp64 summation error of the shared 173-column kernel.
            n=2*len(FEATURE_NAMES)+2; u=np.finfo(np.float64).eps
            error=n*u/(1-n*u)*magnitude+float(np.finfo(float).tiny)*n
            if item['branch']=='left':
                A.append([float(weight[i]) for i in indices])
                rhs.append(float(np.nextafter(HARD_SPLIT_TIE_EPSILON-item['bias']+error,np.inf)))
            elif item['branch']=='right':
                A.append([-float(weight[i]) for i in indices])
                rhs.append(float(np.nextafter(item['bias']-HARD_SPLIT_TIE_EPSILON+error,np.inf)))
            else: raise ValueError('unknown hard route branch')
    return indices,np.asarray(A,float).reshape(len(A),len(indices)),np.asarray(rhs,float),bounds


def dual_upper(c, constant, A, rhs, bounds, y):
    if len(y)!=len(A) or any(not np.isfinite(v) or v<0 for v in y): raise ValueError('invalid nonnegative Farkas vector')
    residual=list(c); total=constant
    for j, multiplier in enumerate(y):
        q=fraction(multiplier);total+=q*fraction(rhs[j])
        for i in range(len(residual)):residual[i]-=q*fraction(A[j,i])
    for r,(lo,hi) in zip(residual,bounds,strict=True): total+=max(r*fraction(lo),r*fraction(hi))
    return total


def binding(leaf,path,legal,use_path):
    return digest(dict(domain_sha256=domain_sha(),support=leaf['support'],bias=leaf['bias'],weights=leaf['weights'],
        normalized=True,witness=leaf['witness'],path=path,legal=sorted(legal),use_path=use_path))


def certify(leaf,path,legal,*,use_path,budget=None):
    support,_,_=coefficient_data(leaf); witness=leaf['witness']
    if type(witness) is not int or witness not in legal or 4 not in legal: raise ValueError('unproven legal witness')
    error=forward_error(leaf); checks=[]; counterexamples=[]
    for action in sorted(set(range(5))-set(legal)):
        indices,A,rhs,bounds=public_polytope(leaf,path,False)
        c,constant=polynomial(leaf,action,witness,indices)
        upper=dual_upper(c,constant,A,rhs,bounds,[])
        required=fraction(MARGIN)+fraction(error[action])+fraction(error[witness])
        y=[];kind='box';point=None
        if upper >= -required and use_path:
            indices,A,rhs,bounds=public_polytope(leaf,path,True)
            c,constant=polynomial(leaf,action,witness,indices)
            if len(indices):
                def solve():
                    return linprog(-np.array(list(map(float,c))),A_ub=A if len(A) else None,
                        b_ub=rhs if len(A) else None,bounds=bounds,method='highs',
                        options={'time_limit':1.,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
                result=budget.solve('proof',{'leaf':path['leaf'],'action':action},solve) if budget is not None else solve()
                if result.success:
                    y=np.maximum(0.,-np.asarray(result.ineqlin.marginals)).tolist() if len(A) else []
                    upper=dual_upper(c,constant,A,rhs,bounds,y);kind='path'
                    point=dict(zip(indices,map(float,result.x),strict=True))
                else:
                    # A solver's infeasibility claim is not trusted. The box
                    # proof remains valid; if too weak, reject this candidate.
                    kind='box';indices,A,rhs,bounds=public_polytope(leaf,path,False)
                    c,constant=polynomial(leaf,action,witness,indices);y=[]
                    upper=dual_upper(c,constant,A,rhs,bounds,[])
        passed=upper < -required
        checks.append(dict(action=action,kind=kind,dual=y,upper=[upper.numerator,upper.denominator],
                           required=[required.numerator,required.denominator],passed=passed))
        if not passed:
            if point is None:
                point={i:(bounds[j][1] if c[j]>0 else bounds[j][0]) for j,i in enumerate(indices)}
            counterexamples.append(point)
    proof=dict(version=VERSION,binding_sha256=binding(leaf,path,legal,use_path),
        use_path=use_path,passed=all(x['passed'] for x in checks),checks=checks)
    return proof,counterexamples


def verify(leaf,path,legal,proof):
    """No LP call: reconstruct every inequality from stored coefficients."""
    if (proof.get('version')!=VERSION or proof.get('passed') is not True or type(proof.get('use_path')) is not bool
            or proof.get('binding_sha256')!=binding(leaf,path,legal,proof.get('use_path'))
            or [v['action'] for v in proof.get('checks',[])]!=sorted(set(range(5))-set(legal))):
        raise ValueError('stale or incomplete certificate')
    error=forward_error(leaf)
    for check in proof['checks']:
        if check['kind'] not in ('box','path') or check['kind']=='path' and not proof['use_path']:raise ValueError('invalid proof domain')
        ids,A,rhs,bounds=public_polytope(leaf,path,check['kind']=='path')
        c,k=polynomial(leaf,check['action'],leaf['witness'],ids)
        upper=dual_upper(c,k,A,rhs,bounds,check['dual'])
        required=fraction(MARGIN)+fraction(error[check['action']])+fraction(error[leaf['witness']])
        if ([upper.numerator,upper.denominator]!=check['upper'] or [required.numerator,required.denominator]!=check['required']
                or not upper < -required or check['passed'] is not True):raise ValueError('invalid exact rational upper bound')
    return True
