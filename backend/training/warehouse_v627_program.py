"""Semantic partitions compiled into certified, non-executing surrogates.

The v626 arithmetic/proof language is reused explicitly, never reinterpreted.
A v627 envelope additionally binds the semantic topology and its compiler.
"""
from __future__ import annotations
from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from .warehouse_v6_actor import FEATURE_NAMES
from .warehouse_v6_tree_routing import split_scores, HARD_SPLIT_TIE_EPSILON, explicit_mask_branches
from .warehouse_v622_program import digest, plain, MASK_INDICES, _eligible, _wilson
from .warehouse_v626_domain import domain_table, domain_sha
from .warehouse_v626_certificate import structure_info, region, binding, verify
from .warehouse_v626_program import Program, fit_leaf, calibrate, metrics, kl
from .warehouse_v626_search import Node, encode, decode, regions, replace, at, SearchBudget

VERSION = 'warehouse-role-program.v6.27-semantic-compiled'
COMPILER = 'semantic-primary-guard-with-counted-wait-fallback.v1'
PARTITION = 'v627-fit-55-10-15-20-whole-scenario-task-stratified.v1'


class Budget(SearchBudget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_steps = sum(e['data']['metadata'].get('reserved_gradient_steps',0)
            for e in self.session.events() if e['event']=='call_started')

    def consume(self, kind, metadata=None):
        steps=(metadata or {}).get('reserved_gradient_steps',0)
        if type(steps) is not int or steps<0 or steps>200 or self.gradient_steps+steps>20000:
            from .warehouse_v625_checkpoint import BudgetExhausted
            raise BudgetExhausted('surrogate gradient step budget exhausted')
        token=super().consume(kind,metadata)
        self.gradient_steps+=steps
        return token

    def summary(self):
        return {**super().summary(),'surrogate_gradient_steps_reserved':self.gradient_steps}


@dataclass(frozen=True)
class Semantic:
    weights: tuple | None = None
    bias: float = 0.
    left: object = None
    right: object = None
    action: int = 4


def semantic_dict(tree):
    actions = []
    def walk(node):
        if node.weights is None:
            if type(node.action) is not int or not 0 <= node.action < 5:
                raise ValueError('unknown semantic action')
            actions.append(node.action)
            return None
        return Node(node.weights, node.bias, walk(node.left), walk(node.right))
    return plain(dict(structure=encode(walk(tree))[0], actions=actions))


def semantic_from_dict(data):
    if set(data) != {'structure', 'actions'}:
        raise ValueError('unknown semantic tree format')
    t, info = structure_info(data['structure'], 64)
    actions = data['actions']
    if len(actions) != t['leaf_count']:
        raise ValueError('semantic outputs and terminals differ')
    def walk(ref):
        if ref < 0:
            action = actions[-1-ref]
            if type(action) is not int or not 0 <= action < 5:
                raise ValueError('invalid learned semantic action')
            return Semantic(action=action)
        return Semantic(tuple(map(float, t['split_weights'][ref])), float(t['split_bias'][ref]),
                        walk(int(t['left_refs'][ref])), walk(int(t['right_refs'][ref])))
    return walk(t['root_ref'])


def semantic_regions(tree, x):
    def walk(node, path, rows):
        yield node, path, rows
        if node.weights is not None:
            go = split_scores(x[rows], np.asarray(node.weights, np.float32)[None],
                              np.asarray([node.bias], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
            yield from walk(node.left, path+(0,), rows[~go])
            yield from walk(node.right, path+(1,), rows[go])
    return list(walk(tree, (), np.arange(len(x))))


def semantic_replace(tree, path, replacement):
    if not path:
        return replacement
    if tree.weights is None:
        raise ValueError('unreachable semantic replacement')
    return Semantic(tree.weights, tree.bias,
                    semantic_replace(tree.left, path[1:], replacement) if path[0] == 0 else tree.left,
                    semantic_replace(tree.right, path[1:], replacement) if path[0] == 1 else tree.right)


def compile_tree(tree, maximum_leaves):
    """All guards, fallback/empty leaves and semantic nodes count in full."""
    if maximum_leaves not in (32, 64):
        raise ValueError('unknown compiled capacity')
    semantic = semantic_dict(tree)
    _, info = structure_info(semantic['structure'], 64)
    if info['leaf_count'] > maximum_leaves//2 or info['depth'] > (5 if maximum_leaves == 32 else 6):
        raise ValueError('semantic topology exceeds reserved guard capacity')
    mapping = []
    def walk(node):
        if node.weights is None:
            start = len(mapping)
            if node.action == 4:
                mapping.append((4, 'ordinary'))
                return None
            w = np.zeros(len(FEATURE_NAMES), np.float32)
            w[MASK_INDICES[node.action]] = 1.
            mapping.extend([(node.action, 'fallback'), (node.action, 'primary')])
            return Node(tuple(map(float, w)), -.5)
        return Node(node.weights, node.bias, walk(node.left), walk(node.right))
    compiled = encode(walk(tree))[0]
    structure_info(compiled, maximum_leaves)
    return compiled, mapping


class CertifiedSemanticProgram:
    def __init__(self, semantic, kernel):
        self.semantic, self.kernel = semantic, kernel

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def to_dict(self):
        compiled, _ = compile_tree(self.semantic, self.kernel.maximum_leaves)
        if digest(compiled) != digest(self.kernel.structure) or self.kernel.arm != 'C3':
            raise ValueError('program is not the exact compiled semantic topology')
        return dict(version=VERSION, compiler=COMPILER, domain_sha256=domain_sha(),
                    semantic=semantic_dict(self.semantic), kernel=self.kernel.to_dict(),
                    external_program_selects_execution_action=False)

    @classmethod
    def from_dict(cls, data, actor_role_sha256):
        if (set(data) != {'version', 'compiler', 'domain_sha256', 'semantic', 'kernel',
                          'external_program_selects_execution_action'} or data['version'] != VERSION
                or data['compiler'] != COMPILER or data['domain_sha256'] != domain_sha()
                or data['external_program_selects_execution_action'] is not False):
            raise ValueError('unknown or mismatched semantic program envelope')
        result = cls(semantic_from_dict(data['semantic']), Program.from_dict(data['kernel'], actor_role_sha256))
        result.to_dict()
        return result


def inner_parts(a):
    """Only outer FIT is allowed here. No outer evidence is read."""
    groups = defaultdict(list)
    for r in a['rows']:
        key = str(r['base_seed'])
        if key not in groups[int(r['task'])]:
            groups[int(r['task'])].append(key)
    selected = {s:set() for s in ('fit', 'calibration', 'eligibility', 'validation')}
    for task, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda seed:digest([PARTITION, task, seed]))
        n = len(ordered)
        if n < 8:
            raise ValueError('insufficient whole scenarios for four-way fit partition')
        ncal, nelig, nscore = max(1, round(.10*n)), max(1, round(.15*n)), max(1, round(.20*n))
        ntrain = n-ncal-nelig-nscore
        cursor = 0
        for s, count in zip(selected, (ntrain, ncal, nelig, nscore), strict=True):
            selected[s].update(ordered[cursor:cursor+count]); cursor += count
    out = {}
    for s, scenes in selected.items():
        take = np.asarray([str(r['base_seed']) in scenes for r in a['rows']])
        out[s] = {k:a[k][take] for k in ('x', 'm', 'p', 'sampled')}
        out[s]['rows'] = [r for r, keep in zip(a['rows'], take, strict=True) if keep]
    if sum(len(v['rows']) for v in out.values()) != len(a['rows']):
        raise ValueError('fit scenarios overlap or are lost')
    return out


def qualification(program, a, predictions):
    """Exact SAME eligibility rule as the final program, no coverage proxy."""
    route, z, q = predictions
    nn, top, raw = a['p'].argmax(1), q.argmax(1), z.argmax(1)
    legal = a['m'][np.arange(len(raw)), raw] > 0
    proofs = program.certificates()
    proven = np.asarray([int(action) in proofs[int(leaf)] for action, leaf in zip(raw, route, strict=True)])
    loss = np.sum(a['p']*(np.log(np.maximum(a['p'], np.float32(1e-8))) -
                           np.log(np.maximum(q, np.float32(1e-8)))), axis=1)
    report = {}
    for i in range(len(program.leaves)):
        take = route == i; n = int(take.sum())
        episodes = Counter(str(r['episode']) for r, keep in zip(a['rows'], take, strict=True) if keep)
        scenes = {str(r['base_seed']) for r, keep in zip(a['rows'], take, strict=True) if keep}
        agreement = float(np.mean(top[take] == nn[take])) if n else 0.
        m = dict(rows=n, agreement=agreement, agreement_wilson_lower_90=_wilson(agreement, n),
                 kl=float(loss[take].mean()) if n else None, distinct_episode_count=len(episodes),
                 distinct_scenario_seed_count=len(scenes), max_episode_fraction=max(episodes.values())/n if n else 0.,
                 illegal_count=int(np.sum(take & ~legal)), path_uncertified_count=int(np.sum(take & ~proven)))
        report[str(i)] = {'feedback':dict(m), 'explanation':dict(m)}
    return report


def qualify_and_score(program, parts, before):
    fit_routes = program.leaf_indices(parts['fit']['x'])
    counts = np.bincount(fit_routes, minlength=len(program.leaves)).tolist()
    elig = qualification(program, parts['eligibility'], program.distributions(parts['eligibility']['x'], parts['eligibility']['m']))
    program.feedback_leaves = frozenset(i for i in range(len(counts)) if counts[i] and _eligible(elig[str(i)]['feedback'], .93))
    program.explanation_leaves = frozenset(i for i in range(len(counts)) if counts[i] and _eligible(elig[str(i)]['explanation'], .95))
    program.report = dict(domain_sha256=domain_sha(), actor_role_sha256=program.actor_role_sha256,
        structure_sha256=digest(program.structure), external_program_selects_execution_action=False,
        full_NN_distribution_targets=True,
        split_episode_ids={s:sorted({str(r['episode']) for r in a['rows']}) for s,a in parts.items()},
        split_scenario_ids={s:sorted({str(r['base_seed']) for r in a['rows']}) for s,a in parts.items()},
        fit_leaf_counts=counts, eligibility=elig, calibration_before=before)
    program.validate()
    a = parts['validation']; pred = program.distributions(a['x'], a['m'])
    result = {purpose:metrics(program, a['rows'], (a['x'],a['m'],a['sampled']), a['p'], selected, purpose, pred)
              for purpose,selected in (('feedback',program.feedback_leaves), ('explanation',program.explanation_leaves))}
    score = (result['explanation']['coverage'], result['feedback']['expected_actionable_coverage'],
             -float(kl(a['p'],pred[2]).mean()), -len(program.leaves))
    return score, result


def primary(p):
    return int(np.sum(p,axis=0).argmax()) if len(p) else 4


def fit_semantic(x, p, leaves, depth, seed, *, random=False):
    if depth == 0 or leaves < 2 or len(x) < 40:
        return Semantic(action=primary(p))
    model = (ExtraTreeRegressor if random else DecisionTreeRegressor)(
        max_leaf_nodes=leaves, max_depth=depth, min_samples_leaf=20,
        random_state=seed, max_features=.6 if random else None)
    model.fit(x,p)
    t = model.tree_
    def walk(i):
        if t.children_left[i] == -1:
            return Semantic(action=int(np.asarray(t.value[i]).reshape(-1).argmax()))
        w = np.zeros(len(FEATURE_NAMES), np.float32); w[t.feature[i]] = 1.
        return Semantic(tuple(map(float,w)),float(np.float32(-t.threshold[i])),walk(t.children_left[i]),walk(t.children_right[i]))
    return walk(0)


def refresh_actions(tree, x, p):
    if tree.weights is None:
        return Semantic(action=primary(p))
    right = split_scores(x,np.asarray(tree.weights,np.float32)[None],np.asarray([tree.bias],np.float32))[:,0] > HARD_SPLIT_TIE_EPSILON
    return Semantic(tree.weights,tree.bias,refresh_actions(tree.left,x[~right],p[~right]),refresh_actions(tree.right,x[right],p[right]))


def s1_semantic(structure, x, p, maximum):
    """Discard legacy certificates; EVERY resulting topology is recompiled."""
    def strip(node, depth, rows):
        if node is None or depth >= (5 if maximum == 32 else 6):
            return Semantic(action=primary(p[rows]))
        right = split_scores(x[rows],np.asarray(node.weight,np.float32)[None],np.asarray([node.bias],np.float32))[:,0] > HARD_SPLIT_TIE_EPSILON
        if any(explicit_mask_branches(np.asarray(node.weight,np.float32),node.bias,MASK_INDICES)):
            # Only an initialization, not a legal program: promote the fit-mass
            # larger side, discard ALL certificates, then compile new guards.
            child = node.right if right.sum() >= (~right).sum() else node.left
            return strip(child,depth,rows)
        return Semantic(node.weight,node.bias,strip(node.left,depth+1,rows[~right]),strip(node.right,depth+1,rows[right]))
    tree = strip(decode(structure),0,np.arange(len(x)))
    # Prune whole semantic subtrees until the reserved guard budget fits.
    while len(semantic_dict(tree)['actions']) > maximum//2:
        candidates = []
        for node,path,ids in semantic_regions(tree,x):
            if node.weights is None:continue
            local = len(semantic_dict(node)['actions'])
            if local <= 1:continue
            mean = p[ids].mean(0) if len(ids) else np.ones(5)/5
            cost = float(np.sum(p[ids]*(np.log(np.maximum(p[ids],1e-8))-np.log(np.maximum(mean,1e-8)))))
            candidates.append((cost/(local-1),path,ids))
        _,path,ids = min(candidates,key=lambda a:(a[0],a[1]))
        tree = semantic_replace(tree,path,Semantic(action=primary(p[ids])))
    return refresh_actions(tree,x,p)


class Context:
    def __init__(self, fit, role, actor_sha, maximum, budget):
        self.parts = inner_parts(fit); self.a = self.parts['fit']
        self.role,self.actor_sha,self.maximum,self.budget = role,actor_sha,maximum,budget
        self.cache = OrderedDict(); self.cache_hits = 0; self.evaluations = 0

    def evaluate(self, semantic):
        t,_ = compile_tree(semantic,self.maximum)
        program = Program(self.role,self.actor_sha,t,[],'C3',self.maximum)
        routes = {s:program.leaf_indices(a['x']) for s,a in self.parts.items()}
        proofs = program.certificates(); before = []
        for i in range(t['leaf_count']):
            self.budget.check();path = region(t,i,self.maximum)
            key = digest(dict(path=path['path'],legal=sorted(proofs[i])))
            if key in self.cache:
                self.cache_hits += 1; model,fitted = deepcopy(self.cache[key])
                for leaf in (model,fitted):
                    leaf['certificate']['binding_sha256'] = binding(leaf,path,proofs[i],True)
                    verify(leaf,path,proofs[i],leaf['certificate'])
            else:
                a = self.parts['fit'];take = routes['fit']==i
                fitted = fit_leaf(a['x'][take],a['m'][take],a['p'][take],arm='C3',path=path,
                                  legal=proofs[i],budget=self.budget)
                a = self.parts['calibration'];take = routes['calibration']==i
                model = calibrate(fitted,a['x'][take],a['m'][take],a['p'][take],arm='C3',path=path,
                                  legal=proofs[i],budget=self.budget)
                self.cache[key] = deepcopy((model,fitted))
                if len(self.cache)>512:self.cache.popitem(last=False)
            program.leaves.append(model); before.append(fitted)
        score,metrics0 = qualify_and_score(program,self.parts,before)
        self.evaluations += 1
        return CertifiedSemanticProgram(semantic,program),score,metrics0
