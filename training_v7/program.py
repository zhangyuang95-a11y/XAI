"""Sparse, certified external programs. Nothing here executes environment actions.

All fit decisions use fit scenes. Calibration only sets temperature; eligibility
only declares leaves. Validation cannot change a tree, output or leaf set.
Proofs are independently checked using rational dual bounds, not LP status.
"""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as Q
from itertools import combinations
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp
from .contracts import MethodSpec, digest, source_support, program_gates


class BudgetExceeded(RuntimeError): pass


class FitBudget:
    def __init__(self, journal=None):
        self.start = time.monotonic(); self.phase = 'search'; self.counts = {'leaf': 0, 'proof': 0}
        self.final_start = None
        self.journal = journal; self.events = []

    def begin_final(self):
        self.phase = 'final'; self.final_start = time.monotonic()

    def debit(self, kind):
        s = MethodSpec()
        now = time.monotonic()
        deadline = s.program_fit_seconds if self.phase == 'search' else s.program_fit_seconds + s.program_final_seconds
        limit = s.leaf_calls if kind == 'leaf' else s.proof_calls
        if self.phase == 'search': limit -= s.final_leaf_call_reserve if kind == 'leaf' else s.final_proof_call_reserve
        if (now - self.start >= deadline or (self.final_start is not None and now-self.final_start >= s.program_final_seconds)
                or self.counts[kind] >= limit):
            raise BudgetExceeded(kind)
        self.counts[kind] += 1
        event = dict(kind=kind, ordinal=self.counts[kind], phase=self.phase, started=True)
        self.events.append(event)
        if self.journal:
            self.journal(event)  # Must durably append before a solver is called.
        return event


def normalize(x, role):
    lo = np.array([f.lower for f in role.features], np.float32)
    hi = np.array([f.upper for f in role.features], np.float32)
    x = np.asarray(x, np.float32)
    if (not np.isfinite(x).all() or (x < lo).any() or (x > hi).any()):
        raise ValueError('program data outside registered physical domain')
    center = (lo.astype(float) + hi) / 2
    scale = np.divide(2., hi.astype(float) - lo, out=np.zeros_like(center), where=hi != lo)
    return (x.astype(float) - center) * scale


def probabilities(logits, masks):
    if logits.shape != masks.shape or not masks.any(-1).all():
        raise ValueError('program masks differ')
    z = np.where(masks, logits, -np.inf)
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def kl_rows(p, q):
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('nonfinite distributions')
    positive = p > 0
    result = np.zeros_like(p, dtype=float)
    result[positive] = p[positive] * (np.log(p[positive]) - np.log(np.maximum(q[positive], 1e-300)))
    return np.maximum(result.sum(-1), 0.)


def checked_data(data, role):
    required = {'x', 'mask', 'p', 'scene', 'sampled'}
    if set(data) != required: raise ValueError('complete labelled program data required')
    x, mask, p = data['x'], data['mask'], data['p']
    if (x.ndim != 2 or x.shape[1] != len(role.features) or mask.shape != p.shape or
            p.shape != (len(x), len(role.actions)) or len(data['scene']) != len(x)
            or len(data['sampled']) != len(x) or not np.isin(mask, [0, 1]).all()
            or not np.array_equal(x[:, list(role.legal_indices)], mask)
            or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1., atol=1e-6)
            or np.any(p[~mask.astype(bool)] != 0) or len(x) == 0):
        raise ValueError('invalid full masked NN labels or public legal bits')
    normalize(x, role)
    return data


def subset(data, selected):
    return {k: np.asarray(v)[selected] for k, v in data.items()}


def right_branch(x, split):
    # Avoid NumPy's weak scalar promotion rounding a midpoint threshold to
    # float32 during fitting while the stored program routes in float64.
    return np.einsum('ni,i->n', np.asarray(x, dtype=float)[:, split['indices']],
                     np.asarray(split['weights'], dtype=float), optimize=False) > float(split['threshold'])


def scene_partition(data, fraction=.8):
    scenes = sorted(set(map(str, data['scene'])), key=lambda x: digest(x))
    if len(scenes) < 2: raise ValueError('scene-disjoint selection needs at least two scenes')
    train = set(scenes[:max(1, min(len(scenes) - 1, int(len(scenes) * fraction)))])
    selected = np.array([str(s) in train for s in data['scene']])
    return subset(data, selected), subset(data, ~selected)


def _polytope(role, path, indices):
    bounds = [(role.features[i].lower, role.features[i].upper) for i in indices]
    A, rhs = [], []
    for split in path:
        row = np.zeros(len(indices)); magnitude = abs(split['threshold'])
        for i, w in zip(split['indices'], split['weights'], strict=True):
            row[indices.index(i)] = w
            f = role.features[i]; magnitude += abs(w) * max(abs(f.lower), abs(f.upper))
        # Routing uses float64; include a conservative dot-product error.
        error = np.nextafter(32 * np.finfo(float).eps * magnitude + np.finfo(float).tiny, np.inf)
        if split['right']: A.append(-row); rhs.append(-split['threshold'] + error)
        else: A.append(row); rhs.append(split['threshold'] + error)
    return np.asarray(A).reshape(len(A), len(indices)), np.array(rhs), bounds


def _dual_upper(c, constant, A, rhs, bounds, multipliers):
    if len(multipliers) != len(A) or any(not math.isfinite(v) or v < 0 for v in multipliers):
        raise ValueError('invalid Farkas multipliers')
    residual = [Q.from_float(float(v)) for v in c]; total = Q.from_float(float(constant))
    for row, b, y in zip(A, rhs, multipliers, strict=True):
        y = Q.from_float(float(y)); total += y * Q.from_float(float(b))
        residual = [v - y * Q.from_float(float(a)) for v, a in zip(residual, row, strict=True)]
    for v, (lo, hi) in zip(residual, bounds, strict=True):
        total += max(v * Q.from_float(float(lo)), v * Q.from_float(float(hi)))
    return total


def _difference(role, leaf, action, witness, indices):
    c = np.zeros(len(indices)); constant = float(leaf['bias'][action]) - float(leaf['bias'][witness])
    for j, i in enumerate(leaf['support']):
        f = role.features[i]; scale = 0. if f.upper == f.lower else 2 / (f.upper - f.lower)
        value = (leaf['weights'][action][j] - leaf['weights'][witness][j]) * scale
        c[indices.index(i)] += value; constant -= value * ((f.lower + f.upper) / 2)
    return c, constant


def legal_facts(role, path):
    facts = {role.wait_index}
    for split in path:
        if (len(split['indices']) == 1 and split['weights'] == [1.] and split['threshold'] == .5
                and split['right'] and split['indices'][0] in role.legal_indices):
            facts.add(role.legal_indices.index(split['indices'][0]))
    return facts


def certify(role, leaf, path, budget):
    legal = legal_facts(role, path)
    witness = leaf['witness']
    if witness not in legal: raise ValueError('unproved witness')
    indices = sorted(set(leaf['support']).union(*(s['indices'] for s in path)))
    checks = []
    for action in sorted(set(range(len(role.actions))) - legal):
        budget.debit('proof')
        c, constant = _difference(role, leaf, action, witness, indices)
        A, rhs, bounds = _polytope(role, path, indices)
        # Conservative forward and normalization error, well above float64
        # arithmetic accumulation; stored coefficients are exact float32.
        error = .001 + 1e-5 * (1 + abs(constant) + sum(abs(c)))
        y = [0.] * len(A)
        upper = _dual_upper(c, constant, A, rhs, bounds, y)
        if upper >= -Q.from_float(error) and indices:
            result = linprog(-c, A_ub=A if len(A) else None, b_ub=rhs if len(A) else None,
                             bounds=bounds, method='highs', options={'time_limit': 1.})
            if result.success:
                y = np.maximum(0., -result.ineqlin.marginals).tolist() if len(A) else []
                upper = _dual_upper(c, constant, A, rhs, bounds, y)
        if upper >= -Q.from_float(error): return None
        checks.append(dict(action=action, dual=y, upper=[upper.numerator, upper.denominator], error=error))
    return dict(binding=digest(dict(role=role.sha256, leaf=leaf, path=path)), checks=checks)


def verify_certificate(role, leaf, path, proof):
    if proof is None or proof['binding'] != digest(dict(role=role.sha256, leaf=leaf, path=path)):
        raise ValueError('missing or stale legality proof')
    legal = legal_facts(role, path)
    if leaf['witness'] not in legal: raise ValueError('invalid witness')
    if [c['action'] for c in proof['checks']] != sorted(set(range(len(role.actions))) - legal):
        raise ValueError('incomplete legality proof')
    indices = sorted(set(leaf['support']).union(*(s['indices'] for s in path)))
    for check in proof['checks']:
        c, constant = _difference(role, leaf, check['action'], leaf['witness'], indices)
        A, rhs, bounds = _polytope(role, path, indices)
        upper = _dual_upper(c, constant, A, rhs, bounds, check['dual'])
        error = .001 + 1e-5 * (1 + abs(constant) + sum(abs(c)))
        if check['error'] != error or check['upper'] != [upper.numerator, upper.denominator] or upper >= -Q.from_float(error):
            raise ValueError('invalid rational proof or forward error')


def _fit_leaf(data, role, path, budget, *, selection=None, fixed_support=None, fixed_witness=None):
    x = normalize(data['x'], role); p = data['p']; masks = data['mask'].astype(bool)
    prior = p.mean(0); allowed = sorted(legal_facts(role, path))
    witness = max(allowed, key=lambda a: (prior[a], -a)) if fixed_witness is None else fixed_witness
    bias = np.log(np.maximum(prior, 1e-8)); bias -= bias.mean()
    # A certified constant fallback is allowed only as a surrogate, never as an
    # environment decision. Search can replace it with a legal fitted leaf.
    fallback = dict(support=[], bias=[0. if a == witness else -20. for a in range(len(prior))],
                    weights=[[] for _ in prior], witness=witness)
    target_residual = p - probabilities(np.broadcast_to(bias, p.shape), masks)
    scores = np.max(np.abs(np.einsum('ni,nj->ij', x, target_residual, optimize=False)), axis=1)
    ranked = [int(i) for i in np.argsort(-scores, kind='stable') if role.features[i].lower != role.features[i].upper]
    # Residual-driven forward supports followed by support exchanges. Sources,
    # rather than just encoded columns, count against the common limit.
    supports = [()]; current = []
    for i in ranked:
        if len(source_support(role.features, [*current, i])) <= 3:
            current.append(i); supports.append(tuple(sorted(current)))
        if len(current) == 3: break
    for i in ranked[:12]:
        for j in range(len(current)):
            s = tuple(sorted(set(current[:j] + current[j + 1:] + [i])))
            if s not in supports and len(source_support(role.features, s)) <= 3:
                supports.append(s)
            if len(supports) >= 8: break
        if len(supports) >= 8: break
    best = fallback; best_key = (-1., -1., -math.inf)
    for support in (supports[:8] if fixed_support is None else [tuple(fixed_support)]):
        z = np.column_stack((np.ones(len(x)), x[:, support])); actions = len(prior)
        initial = np.zeros((actions, z.shape[1])); initial[:, 0] = bias
        def objective(flat):
            w = flat.reshape(actions, z.shape[1]); logits = z @ w.T
            q = probabilities(logits, masks)
            value = kl_rows(p, q).mean() + 1e-5 * np.square(w[:, 1:]).sum()
            gradient = (q - p).T @ z / len(z)
            gradient[:, 1:] += 2e-5 * w[:, 1:]
            return value, gradient.ravel()
        budget.debit('leaf')
        result = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B', bounds=[(-40., 40.)] * initial.size,
                          options={'maxiter': 60, 'ftol': 1e-8})
        if not np.isfinite(result.x).all(): continue
        w = result.x.reshape(actions, -1).astype(np.float32)
        leaf = dict(support=list(support), bias=w[:, 0].tolist(), weights=w[:, 1:].tolist(), witness=witness)
        proof = certify(role, leaf, path, budget)
        if proof is None: continue
        verify_certificate(role, leaf, path, proof)
        eval_data = selection if selection is not None and len(selection['x']) else data
        q, top = leaf_distribution(leaf, role, eval_data['x'], eval_data['mask'])
        agree = top == eval_data['p'].argmax(-1)
        mean_kl = float(kl_rows(eval_data['p'], q).mean())
        support_ok = len(eval_data['x']) >= 160 and len(set(eval_data['scene'])) >= 27
        e = support_ok and agree.mean() >= .97 and mean_kl <= .10
        f = support_ok and agree.mean() >= .95 and mean_kl <= .10
        expected = float((agree * eval_data['p'][np.arange(len(top)), eval_data['p'].argmax(-1)]).mean()) if f else 0.
        key = (float(e), expected, -mean_kl)
        if key > best_key: best, best_key = leaf, key
    proof = certify(role, best, path, budget)
    if proof is None: raise RuntimeError('even fallback leaf lacks a legality certificate')
    return best, proof


def leaf_distribution(leaf, role, x, mask):
    z = normalize(x, role)[:, leaf['support']]
    weights = np.asarray(leaf['weights'], dtype=float).reshape(len(role.actions), len(leaf['support']))
    logits = z @ weights.T + leaf['bias']
    return probabilities(logits, np.asarray(mask, bool)), logits.argmax(-1)


class Program:
    def __init__(self, role, tree, actor_sha256, *, temperature=1., feedback=(), explanation=(), protection=None, anchor_actor_sha256=None):
        self.role, self.tree, self.actor_sha256 = role, deepcopy(tree), actor_sha256
        self.temperature = temperature; self.feedback = tuple(feedback); self.explanation = tuple(explanation)
        self.protection = deepcopy(protection or {})
        self.anchor_actor_sha256 = anchor_actor_sha256 or actor_sha256
        self.validate()

    def validate(self):
        leaves, max_depth = [], 0
        def visit(node, path):
            nonlocal max_depth
            if len(path) > 12: raise ValueError('program exceeds depth limit')
            max_depth = max(max_depth, len(path))
            if 'leaf' in node:
                if set(node) != {'leaf', 'proof'}: raise ValueError('unknown leaf node fields')
                leaf = node['leaf']; support = leaf['support']; a = len(self.role.actions)
                if (set(leaf) != {'support', 'bias', 'weights', 'witness'} or
                        any(type(i) is not int or not 0 <= i < len(self.role.features) for i in support)
                        or support != sorted(set(support)) or len(source_support(self.role.features, support)) > 3
                        or np.asarray(leaf['bias']).shape != (a,)
                        or np.asarray(leaf['weights']).size != a * len(support)
                        or not np.isfinite(leaf['bias']).all() or not np.isfinite(leaf['weights']).all()
                        or np.max(np.abs(leaf['bias']), initial=0.) > 40
                        or np.max(np.abs(leaf['weights']), initial=0.) > 40):
                    raise ValueError('invalid bounded sparse leaf')
                verify_certificate(self.role, leaf, path, node['proof'])
                leaves.append((node, path))
            else:
                if set(node) != {'split', 'left', 'right'}: raise ValueError('unknown split node fields')
                split = node['split']
                if (set(split) != {'indices', 'weights', 'threshold'} or
                        any(type(i) is not int or not 0 <= i < len(self.role.features) for i in split['indices'])
                        or split['indices'] != sorted(set(split['indices']))
                        or len(split['indices']) != len(split['weights']) or len(source_support(self.role.features, split['indices'])) > 3
                        or not np.isfinite(split['weights']).all() or not math.isfinite(split['threshold'])):
                    raise ValueError('invalid sparse split')
                for right, name in ((False, 'left'), (True, 'right')):
                    visit(node[name], [*path, {**split, 'right': right}])
        visit(self.tree, [])
        if (len(leaves) > 256 or max_depth > 12 or len(self.actor_sha256) != 64 or len(self.anchor_actor_sha256) != 64
                or not .1 <= self.temperature <= 10.
                or not set(self.feedback).union(self.explanation) <= set(range(len(leaves)))):
            raise ValueError('program exceeds locked language or eligibility')
        self._leaves = leaves
        protected = {digest(path): digest(node['leaf']) for node, path in leaves if digest(path) in self.protection}
        if protected != self.protection: raise ValueError('protected fit-only output or original path changed')
        return self

    def distributions(self, x, mask):
        x = np.asarray(x); mask = np.asarray(mask, bool)
        q = np.empty(mask.shape); top = np.empty(len(x), int); ids = np.empty(len(x), int)
        cursor = [0]
        def visit(node, selected):
            if 'leaf' in node:
                leaf_id = cursor[0]; cursor[0] += 1
                if selected.any():
                    leaf = node['leaf']; z = normalize(x[selected], self.role)[:, leaf['support']]
                    w = np.asarray(leaf['weights']).reshape(len(self.role.actions), len(leaf['support']))
                    logits = z @ w.T + leaf['bias']
                    q[selected] = probabilities(logits / self.temperature, mask[selected])
                    top[selected] = logits.argmax(-1); ids[selected] = leaf_id
                return
            s = node['split']; right = right_branch(x, s)
            visit(node['left'], selected & ~right); visit(node['right'], selected & right)
        visit(self.tree, np.ones(len(x), bool))
        return q, top, ids

    def as_dict(self):
        return dict(version='three-domain-certified-program.v1', role_sha256=self.role.sha256,
                    actor_sha256=self.actor_sha256, tree=deepcopy(self.tree), temperature=self.temperature,
                    anchor_actor_sha256=self.anchor_actor_sha256,
                    feedback=list(self.feedback), explanation=list(self.explanation),
                    protection=deepcopy(self.protection),
                    maximum_depth=12, maximum_leaves=256, runtime_control_allowed=False)

    @classmethod
    def from_dict(cls, role, value, *, actor_sha256):
        expected = {'version', 'role_sha256', 'actor_sha256', 'anchor_actor_sha256', 'tree', 'temperature', 'feedback',
                    'explanation', 'protection', 'maximum_depth', 'maximum_leaves', 'runtime_control_allowed'}
        if (set(value) != expected or value['version'] != 'three-domain-certified-program.v1'
                or value['role_sha256'] != role.sha256 or value['actor_sha256'] != actor_sha256
                or value['maximum_depth'] != 12 or value['maximum_leaves'] != 256
                or value['runtime_control_allowed'] is not False):
            raise ValueError('program language, role, Actor or complexity binding differs')
        return cls(role, value['tree'], actor_sha256, temperature=value['temperature'],
                   feedback=value['feedback'], explanation=value['explanation'], protection=value['protection'],
                   anchor_actor_sha256=value['anchor_actor_sha256'])


def metrics(program, data):
    checked_data(data, program.role)
    q, top, ids = program.distributions(data['x'], data['mask'])
    p = data['p']; agreement = top == p.argmax(-1); kl = kl_rows(p, q)
    result = dict(illegal_actions=int((~data['mask'][np.arange(len(top)), top].astype(bool)).sum()),
                  unproved_paths=0, full_tree_agreement=float(agreement.mean()), full_tree_kl=float(kl.mean()),
                  frames=len(top), scenes=len(set(data['scene'])), support_passed=True)
    for purpose, leaves in (('feedback', program.feedback), ('explanation', program.explanation)):
        selected = np.isin(ids, leaves)
        result[purpose + '_coverage'] = float(selected.mean())
        result[purpose + '_agreement'] = float(agreement[selected].mean()) if selected.any() else 0.
        result[purpose + '_kl'] = float(kl[selected].mean()) if selected.any() else 1e300
        result[purpose + '_available_coverage'] = float((selected & agreement & (top == data['sampled'])).mean())
        if purpose == 'feedback':
            result['expected_feedback_coverage'] = float((selected * agreement * p[np.arange(len(p)), p.argmax(-1)]).mean())
        for leaf_id in leaves:
            rows = ids == leaf_id
            if rows.sum() < 160 or len(set(data['scene'][rows])) < 27:
                result['support_passed'] = False
    result['passed'] = program_gates(result)
    return result


def qualify(program, data):
    checked_data(data, program.role)
    q, top, ids = program.distributions(data['x'], data['mask'])
    kl, agree = kl_rows(data['p'], q), top == data['p'].argmax(-1)
    feedback, explanation = [], []
    for i in range(len(program._leaves)):
        selected = ids == i
        if selected.sum() < 160 or len(set(data['scene'][selected])) < 27: continue
        if kl[selected].mean() <= .1:
            if agree[selected].mean() >= .95: feedback.append(i)
            if agree[selected].mean() >= .97: explanation.append(i)
    return Program(program.role, program.tree, program.actor_sha256, temperature=program.temperature,
                   feedback=feedback, explanation=explanation, protection=program.protection,
                   anchor_actor_sha256=program.anchor_actor_sha256)


def selection_key(program, data):
    report = metrics(qualify(program, data), data)
    return (report['explanation_coverage'], report['expected_feedback_coverage'],
            -report['full_tree_kl'], -len(program._leaves))


def fit_program(role, partitions, actor_sha256, budget, *, previous=None):
    if previous is not None: raise ValueError('new Actor needs a new seed topology; use refit_program only within its locked training pair')
    if set(partitions) != {'fit', 'calibration', 'eligibility', 'validation'}:
        raise ValueError('four explicit episode-disjoint partitions required')
    sets = []
    for data in partitions.values():
        checked_data(data, role); sets.append(set(map(str, data['scene'])))
    if any(a & b for a, b in combinations(sets, 2)): raise ValueError('program partition scene leakage')
    train, selection = scene_partition(partitions['fit'])
    from sklearn.tree import DecisionTreeRegressor
    # New BC Actors have no historical S1. Generate the common seed topology
    # from full NN distributions on fit-training scenes, then compile explicit
    # legality branches. No qualification/validation label chooses this seed.
    template = DecisionTreeRegressor(max_depth=4, max_leaf_nodes=16,
                                    min_samples_leaf=160, random_state=0)
    template.fit(train['x'], train['p'])
    def constant(path):
        leaf = dict(support=[], bias=[0. if a == role.wait_index else -20. for a in range(len(role.actions))],
                    weights=[[] for _ in role.actions], witness=role.wait_index)
        return dict(leaf=leaf, proof=certify(role, leaf, path, budget))
    def seed(source, path, indices, validation_indices):
        t = template.tree_
        if t.children_left[source] != t.children_right[source]:
            i, threshold = int(t.feature[source]), float(t.threshold[source])
            split = dict(indices=[i], weights=[1.], threshold=threshold)
            right = right_branch(train['x'][indices], split)
            vr = right_branch(selection['x'][validation_indices], split)
            return dict(split=split,
                left=seed(t.children_left[source], [*path, {**split, 'right': False}], indices[~right], validation_indices[~vr]),
                right=seed(t.children_right[source], [*path, {**split, 'right': True}], indices[right], validation_indices[vr]))
        if not len(indices): return constant(path)
        top = int(train['p'][indices].mean(0).argmax())
        if top not in legal_facts(role, path):
            split = dict(indices=[role.legal_indices[top]], weights=[1.], threshold=.5)
            right = right_branch(train['x'][indices], split)
            vr = right_branch(selection['x'][validation_indices], split)
            children = []
            for branch in (False, True):
                child_path = [*path, {**split, 'right': branch}]
                ids = indices[right == branch]; vids = validation_indices[vr == branch]
                if len(ids):
                    leaf, proof = _fit_leaf(subset(train, ids), role, child_path, budget, selection=subset(selection, vids))
                    children.append(dict(leaf=leaf, proof=proof))
                else: children.append(constant(child_path))
            return dict(split=split, left=children[0], right=children[1])
        leaf, proof = _fit_leaf(subset(train, indices), role, path, budget, selection=subset(selection, validation_indices))
        return dict(leaf=leaf, proof=proof)
    tree = seed(0, [], np.arange(len(train['x'])), np.arange(len(selection['x'])))
    seed_program = Program(role, tree, actor_sha256)
    sq, stop, sids = seed_program.distributions(selection['x'], selection['mask'])
    protected_nodes = set()
    def protect(node, path):
        if 'leaf' in node:
            for i, (_, expected_path) in enumerate(seed_program._leaves):
                selected = sids == i
                if (path == expected_path and selected.sum() >= 160 and len(set(selection['scene'][selected])) >= 27
                        and (stop[selected] == selection['p'][selected].argmax(-1)).mean() >= .97
                        and kl_rows(selection['p'][selected], sq[selected]).mean() <= .10):
                    protected_nodes.add(id(node))
            return
        for right, child in ((False, 'left'), (True, 'right')):
            protect(node[child], [*path, {**node['split'], 'right': right}])
    protect(tree, [])
    leaf_count = len(seed_program._leaves)
    while leaf_count < 256:
        proposed = []
        def candidates(node, path, indices, ordinal):
            if 'leaf' not in node:
                s = node['split']; right = right_branch(train['x'][indices], s)
                candidates(node['left'], [*path, {**s, 'right': False}], indices[~right], ordinal)
                candidates(node['right'], [*path, {**s, 'right': True}], indices[right], ordinal)
                return
            identity = ordinal[0]; ordinal[0] += 1
            if id(node) in protected_nodes or len(path) >= 12 or len(indices) < 320: return
            data = subset(train, indices); target = data['p']; oldq, _ = leaf_distribution(node['leaf'], role, data['x'], data['mask'])
            error = float(kl_rows(target, oldq).sum())
            if error < 1e-8: return
            for i, feature in enumerate(role.features):
                values = data['x'][:, i]
                if feature.lower == feature.upper or np.min(values) == np.max(values): continue
                thresholds = [.5] if i in role.legal_indices else np.unique(np.quantile(values, [.2, .4, .6, .8]))
                for threshold in thresholds:
                    right = values.astype(float) > float(threshold)
                    if right.sum() < 160 or (~right).sum() < 160: continue
                    cost = 0.
                    for selected in (right, ~right):
                        mean = target[selected].mean(0)
                        cost += float(kl_rows(target[selected], np.broadcast_to(mean, target[selected].shape)).sum())
                    proposed.append((error - cost, identity, i, float(threshold), node, path, indices, right))
        candidates(tree, [], np.arange(len(train['x'])), [0])
        if not proposed: break
        accepted = False
        for _, _, i, threshold, node, path, indices, right in sorted(proposed, key=lambda v: (-v[0], v[1], v[2], v[3]))[:8]:
            split = dict(indices=[i], weights=[1.], threshold=threshold)
            try:
                children = []
                for branch in (False, True):
                    child_data = subset(train, indices[right == branch])
                    sel = right_branch(selection['x'], split)
                    for item in path:
                        hit = right_branch(selection['x'], item)
                        sel = sel & (hit if item['right'] else ~hit)
                    if not branch:
                        sel = ~right_branch(selection['x'], split)
                        for item in path:
                            hit = right_branch(selection['x'], item)
                            sel &= hit if item['right'] else ~hit
                    child, cert = _fit_leaf(child_data, role, [*path, {**split, 'right': branch}], budget,
                                            selection=subset(selection, sel))
                    children.append(dict(leaf=child, proof=cert))
                trial = deepcopy(tree)
                # Evaluate proposed replacement on fit-internal selection only.
                old = deepcopy(node); node.clear(); node.update(split=split, left=children[0], right=children[1])
                candidate = Program(role, tree, actor_sha256)
                old_program = Program(role, trial, actor_sha256)
                key, oldkey = selection_key(candidate, selection), selection_key(old_program, selection)
                if key > oldkey:
                    leaf_count += 1; accepted = True; break
                node.clear(); node.update(old)
            except BudgetExceeded:
                break
        if not accepted: break
    budget.begin_final()
    # Refit all unprotected outputs using all fit scenes, then re-prove.
    ordinal = [0]
    def refit(node, path, selected):
        if 'leaf' in node:
            identity = ordinal[0]; ordinal[0] += 1
            if id(node) not in protected_nodes and selected.any():
                node['leaf'], node['proof'] = _fit_leaf(subset(partitions['fit'], selected), role, path, budget)
            return
        s = node['split']; right = right_branch(partitions['fit']['x'], s)
        refit(node['left'], [*path, {**s, 'right': False}], selected & ~right)
        refit(node['right'], [*path, {**s, 'right': True}], selected & right)
    refit(tree, [], np.ones(len(partitions['fit']['x']), bool))
    protection = {}
    def capture_protection(node, path):
        if 'leaf' in node:
            if id(node) in protected_nodes: protection[digest(path)] = digest(node['leaf'])
            return
        for right, child in ((False, 'left'), (True, 'right')):
            capture_protection(node[child], [*path, {**node['split'], 'right': right}])
    capture_protection(tree, [])
    program = Program(role, tree, actor_sha256, protection=protection)
    # Calibration chooses temperature only; argmax and certificate stay bound.
    scored = []
    for temperature in (.75, 1., 1.25, 1.5):
        candidate = Program(role, tree, actor_sha256, temperature=temperature, protection=protection)
        q, _, _ = candidate.distributions(partitions['calibration']['x'], partitions['calibration']['mask'])
        scored.append((float(kl_rows(partitions['calibration']['p'], q).mean()), temperature))
    temperature = min(scored)[1]
    program = qualify(Program(role, tree, actor_sha256, temperature=temperature, protection=protection), partitions['eligibility'])
    program.validate()
    return program, metrics(program, partitions['validation'])


def refit_program(previous, partitions, actor_sha256, budget):
    """Online output-only refit: exact topology, support and witness stay fixed."""
    budget.begin_final()
    for data in partitions.values(): checked_data(data, previous.role)
    sets = [set(map(str, d['scene'])) for d in partitions.values()]
    if any(a & b for a, b in combinations(sets, 2)): raise ValueError('program partition leakage')
    tree = deepcopy(previous.tree); fit = partitions['fit']; ordinal = [0]
    def visit(node, path, selected):
        if 'leaf' in node:
            identity = ordinal[0]; ordinal[0] += 1
            if digest(path) not in previous.protection and selected.any():
                old = deepcopy(node['leaf'])
                leaf, proof = _fit_leaf(subset(fit, selected), previous.role, path, budget,
                    fixed_support=old['support'], fixed_witness=old['witness'])
                if leaf['support'] != old['support'] or leaf['witness'] != old['witness']:
                    # A fallback cannot silently change the locked language.
                    return
                node.update(leaf=leaf, proof=proof)
            return
        split = node['split']; right = right_branch(fit['x'], split)
        visit(node['left'], [*path, {**split, 'right': False}], selected & ~right)
        visit(node['right'], [*path, {**split, 'right': True}], selected & right)
    visit(tree, [], np.ones(len(fit['x']), bool))
    scored = []
    for temperature in (.75, 1., 1.25, 1.5):
        trial = Program(previous.role, tree, actor_sha256, temperature=temperature, protection=previous.protection,
                        anchor_actor_sha256=previous.anchor_actor_sha256)
        q, _, _ = trial.distributions(partitions['calibration']['x'], partitions['calibration']['mask'])
        scored.append((float(kl_rows(partitions['calibration']['p'], q).mean()), temperature))
    program = Program(previous.role, tree, actor_sha256, temperature=min(scored)[1], protection=previous.protection,
                      anchor_actor_sha256=previous.anchor_actor_sha256)
    program = qualify(program, partitions['eligibility'])
    return program, metrics(program, partitions['validation'])
