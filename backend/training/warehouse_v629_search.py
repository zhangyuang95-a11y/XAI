"""Cumulative, local, certified repairs of v6.26/v6.28's protected anchor.

All structure/support proposals read inner fit only. Inner calibration,
eligibility and score remain distinct whole-scene partitions. Outer evidence
is never accepted here. A queue stores local replacements, not stale whole
trees: each candidate is applied to the latest accepted working program.
"""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.tree import DecisionTreeRegressor
from .warehouse_v626_search import Node, encode, decode, replace, at, regions
from .warehouse_v626_program import fit_leaf, calibrate, kl
from .warehouse_v626_budget import Budget as BaseBudget
from .warehouse_v625_checkpoint import BudgetExhausted, DeadlineExceeded
from .warehouse_v628_checkpoint import CheckpointSession
from .warehouse_v628_certificate import structure_info, region, rebind
from .warehouse_v628_search import Context as PreviousContext, score_metrics, descendant
from .warehouse_v629_program import Program, qualify, measure, protected_indices, path_key
from .warehouse_v629_guards import predicate
from .warehouse_v622_program import digest, plain

ALGORITHM = 'cumulative_local_certified_repair_v1'


class Budget(BaseBudget):
    def __init__(self, path, identity, *, resume=False, started_at=None):
        self.limits = dict(leaf=80000, node=4096, proof=400000)
        self.session = CheckpointSession(path, {**identity, 'solver_limits': self.limits,
            'budget_contract': 'v629-cumulative-common-max256-reserve.v1'},
            max_calls=484096, wall_seconds=6000, resume=resume, started_at=started_at)
        self.counts = {k: 0 for k in self.limits}
        self.pending = {}; self.phase = 'fit'; self.final_leaf_reserve = 0
        if resume:
            for event in self.session.events():
                if event['event'] == 'call_started':
                    self.counts[event['data']['metadata']['solver_kind']] += 1
        self.reserved = dict(leaf=2048, node=0, proof=13312)

    def consume(self, kind, metadata=None):
        if self.phase != 'final' and self.counts[kind] >= self.limits[kind] - self.reserved[kind]:
            raise BudgetExhausted('v629 common final ' + kind + ' reserve reached')
        return super().consume(kind, metadata)

    def check(self):
        super().check()
        # Cumulative billed time includes resumed work and checkpoint charges.
        # Leave 20 seconds before the independent hard search timer for a safe
        # return and seal; final fitting still owns its separate 1200 seconds.
        if self.phase != 'final' and self.session.elapsed_seconds >= 4770:
            raise DeadlineExceeded('v629 cumulative soft search deadline reached')


class Context(PreviousContext):
    def __init__(self, parts, parent, budget, *, guard_bundles=True):
        super().__init__(parts, parent, budget)
        self.guard_bundles = guard_bundles
        self.reused_leaves = 0

    def root_paths(self, program):
        paths = [path for _, path, _ in self.editable(decode(program.structure))]
        return [path for path in paths if not any(descendant(path, other) and path != other for other in paths)]

    def evaluate(self, tree, *, anchor=None, preserve_parent=False):
        """Only changed numeric leaf paths are fitted, all others exact-rebound.

        This preserves independent local gains, beyond the permanently protected
        core. The numerical proof may be reused only for identical paths/legal
        sets; full-tree qualification and scoring are always recomputed.
        """
        anchor = self.parent if anchor is None else anchor
        t, _ = structure_info(encode(tree)[0], self.maximum)
        out = Program(anchor.role, anchor.actor_role_sha256, t, [], 'C3', self.maximum,
            protection=deepcopy(anchor.protection))
        legal, old_legal = out.certificates(), anchor.certificates()
        old_paths = {path_key(region(anchor.structure, i, self.maximum)): i for i in range(len(anchor.leaves))}
        protected = {digest(e['path']) for e in anchor.protection['entries']}
        route = out.leaf_indices(self.a['x'])
        score_data = self.parts['validation']; sr = out.leaf_indices(score_data['x'])
        cal = self.parts['calibration']; cr = out.leaf_indices(cal['x']); before = []
        for i in range(t['leaf_count']):
            self.budget.check(); path = region(t, i, self.maximum); key_path = path_key(path)
            if key_path in old_paths:
                j = old_paths[key_path]
                leaf = rebind(anchor.leaves[j], region(anchor.structure, j, self.maximum),
                    path, old_legal[j], legal[i])
                self.reused_leaves += 1; before.append(deepcopy(leaf))
            elif key_path in protected:
                raise ValueError('proposal lost a protected anchor path')
            else:
                take, score_take, cal_take = route == i, sr == i, cr == i
                key = digest(dict(language='v629-canonical-guard', path=path['path'], legal=sorted(legal[i]),
                    rows=sha256(np.flatnonzero(take).astype('<i8').tobytes()).hexdigest()))
                if key in self.cache:
                    saved = deepcopy(self.cache[key])
                    leaf = rebind(saved['leaf'], saved['path'], path, legal[i], legal[i])
                    self.cache_hits += 1; before.append(saved['before'])
                else:
                    leaf = fit_leaf(self.a['x'][take], self.a['m'][take], self.a['p'][take], arm='C3',
                        path=path, legal=legal[i], budget=self.budget,
                        selection=(score_data['x'][score_take], score_data['m'][score_take], score_data['p'][score_take]))
                    uncal = deepcopy(leaf); before.append(uncal)
                    leaf = calibrate(leaf, cal['x'][cal_take], cal['m'][cal_take], cal['p'][cal_take],
                        arm='C3', path=path, legal=legal[i], budget=self.budget)
                    self.cache[key] = dict(leaf=deepcopy(leaf), path=path, before=uncal)
                    if len(self.cache) > 512: self.cache.popitem(last=False)
                # A stored cache leaf is not trusted without independent replay.
            out.leaves.append(leaf)
        qualify(out, self.parts, before)
        metrics = measure(out, score_data)
        score = score_metrics(metrics,
            kl(score_data['p'], out.distributions(score_data['x'], score_data['m'])[2]).mean(), len(out.leaves))
        self.evaluations += 1
        return out, score, metrics

    def proposals(self, program, path):
        # Existing bounded CART, single-bit guards, prune and residual axes.
        candidates, status = super().proposals(program, path)
        if not self.guard_bundles: return candidates, status
        tree = decode(program.structure)
        ids = next(ids for _, p, ids in regions(tree, self.a['x']) if p == path)
        remaining = 9 - len(path)
        if not len(ids) or remaining < 1: return candidates, status
        x, m, p = (self.a[k][ids] for k in ('x', 'm', 'p'))
        route, _, _ = program.distributions(x, m); proven = program.certificates()
        missing = np.array([int(a) not in proven[int(i)] for a, i in zip(p.argmax(1), route, strict=True)])
        ranked = []
        for k in (2, 3):
            for actions in combinations(range(4), k):
                take = np.all(m[:, actions] == 1, axis=1)
                if not np.any(take): continue
                covered = take & missing & np.isin(p.argmax(1), actions)
                ranked.append((float(p[covered][:, actions].sum()), int(take.sum()), actions))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        for _, _, actions in ranked[:6]:
            w, b = predicate(actions); guard = Node(w, b)
            candidates.append((plain(encode(replace(tree, path, guard))[0]), 'bundle_' + ''.join(map(str, actions))))
        # Three+one legal tests certify all four moves in two levels, with every
        # failed guard and empty fallback still represented as counted leaves.
        if remaining >= 2 and np.any(np.all(m[:, :4] == 1, axis=1)):
            for omitted in range(4):
                actions = tuple(a for a in range(4) if a != omitted)
                w, b = predicate(actions); w1, b1 = predicate((omitted,))
                replacement = Node(w, b, None, Node(w1, b1))
                candidates.append((plain(encode(replace(tree, path, replacement))[0]), 'all4_bundle_' + str(omitted)))
        # Combine a guard with a distribution proposal immediately. Requiring
        # a weak intermediate guard to be accepted first would recreate a local
        # optimization barrier. These small CART fits all debit the node ledger.
        if remaining >= 2:
            for _, _, actions in ranked[:3]:
                take = np.all(m[:, actions] == 1, axis=1)
                if int(take.sum()) < 40: continue
                for leaves in (2, 4):
                    def solve():
                        model = DecisionTreeRegressor(max_leaf_nodes=leaves, max_depth=remaining-1,
                            min_samples_leaf=20, random_state=0).fit(x[take], p[take])
                        return OptimizeResult(success=True, status=0, fun=float(np.mean((model.predict(x[take])-p[take])**2)),
                            model=model, nit=1)
                    result = self.budget.solve('node', dict(path=list(path), operation='guarded_distribution',
                        actions=list(actions), leaves=leaves), solve)
                    t = result.model.tree_
                    def walk(i):
                        if t.children_left[i] < 0: return None
                        w = np.zeros(x.shape[1], np.float32); w[t.feature[i]] = 1
                        return Node(tuple(map(float, w)), float(np.float32(-t.threshold[i])),
                            walk(t.children_left[i]), walk(t.children_right[i]))
                    w, b = predicate(actions)
                    replacement = Node(w, b, None, walk(0))
                    candidates.append((plain(encode(replace(tree, path, replacement))[0]),
                        'guarded_distribution_' + ''.join(map(str, actions)) + '_' + str(leaves)))
        unique = {}
        for candidate, label in candidates:
            try: t, _ = structure_info(candidate, self.maximum)
            except ValueError: continue
            unique.setdefault(digest(t), (plain(t), label))
        return list(unique.values()), 'enumerated_with_canonical_bundles'

    def checkpoint(self, machine, beam, best, status='running'):
        store = self.budget.session.store
        def put(p): return store.put_json(p.to_dict())
        self.budget.checkpoint(dict(algorithm=ALGORITHM, machine=deepcopy(machine),
            beam=[dict(program=put(p), score=list(s)) for p, s in beam],
            best=dict(program=put(best[0]), score=list(best[1])),
            cache=[[k, store.put_json(value)] for k, value in self.cache.items()],
            cache_hits=self.cache_hits, reused_leaves=self.reused_leaves, evaluations=self.evaluations, trace=self.trace), status=status)


def optimize(ctx, *, resume=False, pause_after=0, rounds=14):
    if resume:
        state = ctx.budget.session.state
        if state['algorithm'] != ALGORITHM: raise ValueError('unknown v629 resume algorithm')
        store = ctx.budget.session.store
        def get(e): return Program.from_dict(store.get_json(e['program']), ctx.parent.actor_role_sha256), tuple(e['score'])
        beam = [get(e) for e in state['beam']]; best = get(state['best']); machine = deepcopy(state['machine'])
        ctx.cache = OrderedDict((k, store.get_json(v)) for k, v in state['cache'])
        for key in ('cache_hits', 'evaluations', 'trace', 'reused_leaves'): setattr(ctx, key, deepcopy(state[key]))
    else:
        program, score, _ = ctx.evaluate(decode(ctx.parent.structure), preserve_parent=True)
        beam = [(program, score)]; best = beam[0]
        roots = [list(path) for path in ctx.root_paths(program)]
        machine = dict(round=0, beam_cursor=0, paths=None, path_cursor=0, queue=None, cursor=0,
            mandatory_paths=roots, mandatory_checked=[], mandatory_complete=not roots, pending_frontier=[],
            configuration=dict(rounds=rounds, guard_bundles=ctx.guard_bundles))
    if machine['configuration'] != dict(rounds=rounds, guard_bundles=ctx.guard_bundles):
        raise ValueError('resume search settings differ')
    stopped = 'registered_rounds_complete'
    try:
        ctx.checkpoint(machine, beam, best)
        while machine['round'] < rounds:
            if machine['beam_cursor'] >= len(beam):
                distinct = {digest(best[0].to_dict()): best}
                for value in machine['pending_frontier']:
                    p = Program.from_dict(value['program'], ctx.parent.actor_role_sha256)
                    distinct.setdefault(digest(p.to_dict()), (p, tuple(value['score'])))
                ranked = sorted(distinct.values(), key=lambda item: item[1], reverse=True)
                beam = ranked[:3]
                machine.update(round=machine['round']+1, beam_cursor=0, paths=None, path_cursor=0,
                    queue=None, cursor=0, pending_frontier=[])
                continue
            current, current_score = beam[machine['beam_cursor']]
            if machine['paths'] is None:
                editable = ctx.editable(decode(current.structure))
                _, _, q = current.distributions(ctx.a['x'], ctx.a['m']); loss = kl(ctx.a['p'], q)
                roots = set(ctx.root_paths(current))
                editable.sort(key=lambda row: (row[1] not in roots, -float(loss[row[2]].sum()), len(row[1]), row[1]))
                machine['paths'] = [list(path) for _, path, _ in editable]
            if machine['path_cursor'] >= len(machine['paths']):
                machine.update(beam_cursor=machine['beam_cursor']+1, paths=None, path_cursor=0, queue=None, cursor=0)
                continue
            path = tuple(machine['paths'][machine['path_cursor']])
            # Earlier accepted pruning may remove descendants scheduled in this
            # sweep. New descendants are enumerated on the following sweep.
            valid_paths = {p for _, p, _ in ctx.editable(decode(current.structure))}
            if path not in valid_paths:
                machine.update(path_cursor=machine['path_cursor']+1, queue=None, cursor=0)
                continue
            if machine['queue'] is None:
                proposals, status = ctx.proposals(current, path)
                machine['queue'] = [dict(subtree=plain(encode(at(decode(t), path))[0]), label=label)
                    for t, label in proposals]
                machine['cursor'] = 0
                ctx.trace.append(dict(round=machine['round'], path=list(path), status=status, proposals=len(proposals)))
            if machine['cursor'] >= len(machine['queue']):
                if (machine['round'] == 0 and machine['beam_cursor'] == 0 and list(path) in machine['mandatory_paths']
                        and list(path) not in machine['mandatory_checked']):
                    machine['mandatory_checked'].append(list(path))
                    machine['mandatory_complete'] = len(machine['mandatory_checked']) == len(machine['mandatory_paths'])
                machine.update(path_cursor=machine['path_cursor']+1, queue=None, cursor=0)
                ctx.checkpoint(machine, beam, best); continue
            item = machine['queue'][machine['cursor']]
            try:
                tree = replace(decode(current.structure), path, decode(item['subtree']))
                proposed, score, _ = ctx.evaluate(tree, anchor=current)
                accepted = score > current_score
                global_improvement = score > best[1]
                if accepted:
                    # This is the essential change: subsequent proposals are
                    # applied to this accepted complete program in THIS sweep.
                    beam[machine['beam_cursor']] = (proposed, score)
                if global_improvement: best = (proposed, score)
                frontier = machine['pending_frontier']
                fingerprint = digest(proposed.to_dict())
                if not any(v['fingerprint'] == fingerprint for v in frontier):
                    frontier.append(dict(program=proposed.to_dict(), score=list(score), fingerprint=fingerprint, path=list(path)))
                    # At most one exploratory alternative per local repair path,
                    # so three variants of one path cannot crowd out diversity.
                    frontier.sort(key=lambda value: value['score'], reverse=True)
                    seen = set(); diverse = []
                    for value in frontier:
                        key = tuple(value['path'])
                        if key not in seen: diverse.append(value); seen.add(key)
                    frontier[:] = diverse[:3]
                ctx.trace.append(dict(round=machine['round'], path=list(path), operation=item['label'],
                    score=list(score), accepted=accepted, best_updated=global_improvement,
                    applied_to_sha256=digest(current.to_dict()), result_sha256=fingerprint))
            except ValueError as error:
                ctx.trace.append(dict(round=machine['round'], path=list(path), operation=item['label'], rejected=str(error)))
            machine['cursor'] += 1; ctx.checkpoint(machine, beam, best)
            if pause_after and ctx.evaluations >= pause_after:
                ctx.checkpoint(machine, beam, best, 'paused')
                return best[0], dict(paused=True)
    except (BudgetExhausted, DeadlineExceeded) as error:
        stopped = str(error)
    return best[0], dict(stopped=stopped, mandatory_complete=machine['mandatory_complete'],
        mandatory_paths=machine['mandatory_paths'], mandatory_checked=machine['mandatory_checked'],
        best_score=list(best[1]), evaluations=ctx.evaluations, cache_hits=ctx.cache_hits,
        reused_leaves=ctx.reused_leaves, trace=ctx.trace, outer_validation_used=False,
        budget_limited=any(s in stopped for s in ('budget', 'deadline', 'reserve')))
