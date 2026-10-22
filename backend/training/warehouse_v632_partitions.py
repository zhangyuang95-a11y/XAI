"""Fit-only sparse oblique confidence trees; never an action controller.

Predicates predict disagreement between a *fixed* surrogate and full NN.
They only subdivide regions whose complete five-action output is identical.
Hard float32 routing, public-source accounting and old proofs remain binding.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .warehouse_v626_domain import domain_table, transform
from .warehouse_v626_program import _support_cost
from .warehouse_v626_search import Node, at, decode, encode, replace
from .warehouse_v628_certificate import region, structure_info
from .warehouse_v628_program import ParentProgram, payload
from .warehouse_v630_program import Program
from .warehouse_v630_certificate import extend
from .warehouse_v630_partition_proposals import prunings
from .warehouse_v630_partition_search import semantic
from .warehouse_v622_program import digest, plain
from .warehouse_v6_tree_routing import split_scores, HARD_SPLIT_TIE_EPSILON
from experiments.warehouse.v626_protocol import require, bound

VERSION = 'warehouse-sparse-oblique-confidence.v6.32'


def _mv(a, b):
    # Accelerate/NumPy matmul on this host has emitted spurious FP warnings
    # even for bounded small arrays. Explicit reductions avoid that backend;
    # finite objectives/gradients are still required, never warning-suppressed.
    return np.einsum('ij,j->i', a, b, optimize=False)


def collapse_protected(source):
    """Recover original protected prefixes, with their original exact proofs.

    Return each former confidence subtree as an option, including its empty
    branches. Other regions retain the source payload/path. No q changes.
    """
    source.validate()
    tree = decode(source.structure)
    _, old_paths = encode(tree)
    parent = ParentProgram.from_dict(bound(source.protection['parent']), source.actor_role_sha256)
    originals = {}
    replacements = {}
    for entry in source.protection['entries']:
        path = tuple(0 if p['branch'] == 'left' else 1 for p in entry['path'])
        originals[path] = plain(encode(at(tree, path))[0])
        replacements[path] = entry['parent_leaf']
        tree = replace(tree, path, None)
    structure, paths = encode(tree)
    out = Program(source.role, source.actor_role_sha256, plain(structure), [], 'C3', 256,
                  protection=deepcopy(source.protection))
    legal = out.certificates()
    old_legal, parent_legal = source.certificates(), parent.certificates()
    original_options = {}
    for i, path in enumerate(paths):
        if path in replacements:
            j = replacements[path]
            leaf, _ = extend(parent.leaves[j], region(parent.structure, j, 64),
                             region(structure, i, 256), parent_legal[j], legal[i])
            original_options[i] = originals[path]
        else:
            j = old_paths.index(path)
            leaf, _ = extend(source.leaves[j], region(source.structure, j, source.maximum_leaves),
                             region(structure, i, 256), old_legal[j], legal[i])
            original_options[i] = plain(encode(None)[0])
        out.leaves.append(leaf)
    out.validate()
    return out, original_options


def _entropy(positive, total):
    p = np.divide(positive, total, out=np.zeros_like(np.asarray(positive, float)), where=total > 0)
    return -total * (np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.) +
                     np.where(p < 1, (1-p) * np.log(np.maximum(1-p, 1e-300)), 0.))


def _threshold(scores, labels, minimum=40):
    """Best weighted entropy threshold, fitted only on supplied training rows."""
    scores = np.asarray(scores, np.float32)
    order = np.argsort(scores, kind='stable')
    s, y = scores[order], labels[order]
    w = np.where(y == 0, 4., 1.)
    total, correct = np.cumsum(w), np.cumsum(w*y)
    indices = np.flatnonzero((s[:-1] < s[1:]) &
                            (np.arange(len(s)-1)+1 >= minimum) &
                            (len(s)-np.arange(len(s)-1)-1 >= minimum))
    if not len(indices):
        return None
    cost = _entropy(correct[indices], total[indices]) + _entropy(correct[-1]-correct[indices], total[-1]-total[indices])
    k = int(indices[int(np.argmin(cost))])
    threshold = np.float32((float(s[k])+float(s[k+1]))/2)
    return float(_entropy(correct[-1], total[-1])-cost.min()), threshold


def _fit(u, y, support, budget, metadata):
    a = np.column_stack([np.ones(len(u)), u[:, support]])
    w = np.where(y == 0, 4., 1.); w /= w.sum()
    def objective(theta):
        z = _mv(a, theta)
        value = np.sum(w*(np.logaddexp(0., z)-y*z)) + 1e-3*np.sum(theta[1:]**2)/2
        grad = _mv(a.T, w*(expit(z)-y)); grad[1:] += 1e-3*theta[1:]
        require(np.isfinite(value) and np.isfinite(grad).all(), 'nonfinite confidence objective or gradient')
        return float(value), grad
    result = budget.solve('node', dict(method=VERSION, support=list(map(int, support)), **metadata),
        lambda: minimize(objective, np.zeros(len(support)+1), jac=True, method='L-BFGS-B',
                         bounds=[(-32., 32.)]*(len(support)+1), options=dict(maxiter=80, ftol=1e-10)))
    require(np.isfinite(result.x).all(), 'nonfinite sparse confidence classifier')
    return result.x, float(objective(result.x)[0])


def propose(source, arrays, leaf, budget, *, oblique=True):
    """Greedy <=3-source logistic directions plus axis candidates at each node.

    Finite forward support selection (three fits) followed by four conditional
    residual exchanges. Classifier outputs are discarded; true hard routes
    decide the split. Every optimizer call is predebited, including failures.
    """
    path = region(source.structure, leaf, 256)
    depth = min(3, 9-len(path['path']))
    take = source.leaf_indices(arrays['x']) == leaf
    x = arrays['x'][take]; m = arrays['m'][take]; target = arrays['p'][take]
    require(target.shape == m.shape == (len(x), 5) and np.isfinite(target).all() and
            np.all(target[m == 0] == 0) and np.allclose(target.sum(1), 1., atol=1e-6, rtol=0),
            'confidence target must be complete physical masked NN probabilities')
    _, _, q = source.distributions(x, m)
    y = (q.argmax(1) == target.argmax(1)).astype(float)
    d = domain_table()['features']; available = [i for i, a in enumerate(d) if not a['constant']]
    u = transform(x, available).astype(float)
    records = []

    def grow(rows, remaining, route):
        budget.check()
        if remaining == 0 or len(rows) < 80 or np.all(y[rows] == y[rows][0]):
            return None
        local = u[rows]; labels = y[rows]
        valid = np.flatnonzero(np.ptp(local, axis=0) > 1e-8)
        best = None

        def consider(weights, bias=0., origin='axis'):
            nonlocal best
            scores = split_scores(x[rows], weights[None], np.array([bias], np.float32))[:, 0]
            answer = _threshold(scores, labels)
            if answer is None:
                return
            _, threshold = answer
            bb = np.float32(bias-threshold)
            go = split_scores(x[rows], weights[None], np.array([bb], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            if min(go.sum(), (~go).sum()) < 40:
                return
            w = np.where(labels == 0, 4., 1.)
            entropy = sum(float(_entropy(np.sum(w[a]*labels[a]), np.sum(w[a]))) for a in (go, ~go))
            key = (entropy, int(np.count_nonzero(weights)), digest(dict(w=weights.tolist(), b=float(bb))))
            if best is None or key < best[0]:
                best = (key, weights, bb, go, origin)

        for j in valid:
            weights = np.zeros(x.shape[1], np.float32); weights[available[j]] = 1.
            consider(weights)
        supports = []; theta = None; seen = set()

        def fit_support(support):
            nonlocal theta
            ids = [available[j] for j in support]
            key = tuple(sorted(support))
            if key in seen or not 0 < _support_cost(ids) <= 3:
                return None
            seen.add(key)
            fitted, loss = _fit(local, labels, support, budget,
                dict(source_leaf=leaf, route=list(route), training_frames=len(rows), source_indices=ids))
            weights = np.zeros(x.shape[1], np.float32)
            bias = float(fitted[0])
            for k, j in enumerate(ids):
                weights[j] = np.float32(fitted[k+1]*d[j]['scale'])
                bias -= float(fitted[k+1])*d[j]['scale']*d[j]['center']
            consider(weights, np.float32(bias), 'sparse_logistic')
            return fitted, loss

        if oblique:
            support = []
            while len(support) < 3:
                pred = np.full(len(rows), np.average(labels, weights=np.where(labels == 0, 4., 1.))) if theta is None else expit(theta[0]+_mv(local[:, support], theta[1:]))
                residual = np.where(labels == 0, 4., 1.)*(labels-pred)
                correlation = np.abs(_mv(local.T, residual))/np.maximum(np.linalg.norm(local, axis=0), 1e-10)
                choices = [j for j in valid if j not in support and _support_cost([available[k] for k in support+[j]]) <= 3]
                if not choices:
                    break
                chosen = min(choices, key=lambda j: (-correlation[j], available[j]))
                support.append(chosen)
                fitted = fit_support(support)
                if fitted is None:
                    break
                theta = fitted[0]; supports.append(tuple(support))
            if theta is not None and support:
                residual = np.where(labels == 0, 4., 1.)*(labels-expit(theta[0]+_mv(local[:, support], theta[1:])))
                correlation = np.abs(_mv(local.T, residual))/np.maximum(np.linalg.norm(local, axis=0), 1e-10)
                exchanges = []
                for old in support:
                    for new in valid:
                        candidate = sorted([j for j in support if j != old]+[int(new)])
                        if len(candidate) != len(set(candidate)) or tuple(candidate) in seen or _support_cost([available[k] for k in candidate]) > 3:
                            continue
                        exchanges.append((-float(correlation[new]), available[old], available[new], candidate))
                used = 0
                for *_, candidate in sorted(exchanges):
                    if tuple(candidate) in seen:
                        continue
                    fit_support(candidate); used += 1
                    if used == 4:
                        break
        if best is None:
            return None
        key, weights, bias, go, origin = best
        records.append(dict(route=list(route), frames=len(rows), origin=origin,
                            source_features=int(np.count_nonzero(weights)), optimizer_supports=len(seen), weighted_entropy=key[0]))
        return Node(tuple(map(float, weights)), float(bias),
                    grow(rows[~go], remaining-1, route+(0,)), grow(rows[go], remaining-1, route+(1,)))

    tree = grow(np.arange(len(x)), depth, ()) if len(x) else None
    return dict(version=VERSION, source_leaf=leaf, fit_frames=len(x), records=records,
                proposals=prunings(tree), oblique=oblique, classifier_outputs_discarded=True,
                NN_target_used_only_as_fit_label=True, execution_actions_produced=False)
