"""Experimental static compilation for exact native current_report metrics.

No caller/pipeline is changed. This is statistics only: it does not verify NN
target provenance, issue evidence, execute NN or grant training authority.
All current-NN p, sampled, argmax, KL and qualification are recomputed per call.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from . import warehouse_v629_online_evidence as native
from .warehouse_v642_program import Program
from .warehouse_v642_online_evidence import source_files as native_source_files
from .warehouse_v625_program import FEATURE_NAMES, ACTIONS, _wilson
from experiments.warehouse.v626_protocol import require

VERSION = 'warehouse-v642-exact-static-metrics-cache.v1'
ROOT = Path(__file__).resolve().parents[2]


def source_files():
    values = native_source_files()
    name = 'backend/training/warehouse_v642_static_metrics.py'
    values[name] = sha256((ROOT / name).read_bytes()).hexdigest()
    return values


def _plain(value):
    if isinstance(value, (set, frozenset)): return sorted(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _digest(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_sha(array):
    a = np.asarray(array); result = sha256()
    result.update(a.dtype.str.encode()); result.update(json.dumps(list(a.shape)).encode())
    result.update(memoryview(np.ascontiguousarray(a))); return result.hexdigest()


def _part_binding(a):
    x, m, rows = a['x'], a['m'], a['rows']
    require(isinstance(x, np.ndarray) and isinstance(m, np.ndarray) and x.dtype == m.dtype == np.float32
        and x.ndim == 2 and x.shape[1] == len(FEATURE_NAMES) and m.shape == (len(x), 5)
        and len(x) > 0 and type(rows) is list and len(rows) == len(x), 'invalid static pool shape/type')
    return dict(x_sha256=_array_sha(x), m_sha256=_array_sha(m), rows_sha256=_digest(rows))


def _reference(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'explicit complete program reference required')
    path = Path(ref['path'])
    require(path.is_file() and not path.is_symlink() and sha256(path.read_bytes()).hexdigest() == ref['sha256'],
        'static program/dependency bytes changed')
    return json.loads(path.read_bytes())


def _dependency_refs(ref):
    refs, seen = [], set()
    def visit(current):
        identity = (current['path'], current['sha256'])
        if identity in seen: return
        seen.add(identity); value = _reference(current); refs.append(deepcopy(current))
        def walk(v):
            if type(v) is dict:
                if set(v) == {'path', 'sha256'} and type(v['path']) is str and type(v['sha256']) is str:
                    visit(v)
                else:
                    for item in v.values(): walk(item)
            elif type(v) is list:
                for item in v: walk(item)
        walk(value)
    visit(ref)
    return tuple(refs)


def _readonly(value):
    a = np.asarray(value); a.flags.writeable = False; return a


class _StaticPart:
    def __init__(self, program, a, proofs):
        self.x, self.m = _readonly(a['x'].copy()), _readonly(a['m'].copy())
        self.rows = deepcopy(a['rows']); self.n = len(self.rows)
        self.route, raw, self.q = map(_readonly, program.distributions(self.x, self.m))
        self.raw, self.top = _readonly(raw.argmax(-1)), _readonly(self.q.argmax(-1))
        self.legal = _readonly(self.m[np.arange(len(self.m)), self.raw] > 0)
        self.path = _readonly(np.asarray([int(action) in proofs[int(leaf)]
            for action, leaf in zip(self.raw, self.route, strict=True)]))
        self.valid = _readonly(self.legal & self.path)
        self.masked_illegal = _readonly(self.m[np.arange(len(self.m)), self.top] == 0)
        conflict = self.x[:, FEATURE_NAMES.index('same_target_collision_streak_capped')] > 0
        near = self.x[:, FEATURE_NAMES.index('robots_shortest_distance')] * np.float32(42) <= 2
        low = self.x[:, FEATURE_NAMES.index('self_charger_return_margin')] * np.float32(50) <= 4
        low_battery = self.x[:, FEATURE_NAMES.index('self_battery_fraction')] <= np.float32(.2)
        self.strata = {k: _readonly(v) for k, v in dict(ordinary=~(conflict | near | low), conflict=conflict,
            near_robots=near, low_return_margin=low, low_battery_le_20_percent=low_battery).items()}
        self.singletons = {i: self._selection({i}) for i in range(len(program.leaves))}

    def _selection(self, selected):
        chosen = _readonly(np.isin(self.route, tuple(selected)))
        n = int(chosen.sum())
        frequencies = Counter(str(r['episode']) for r, keep in zip(self.rows, chosen, strict=True) if keep)
        scenes = {str(r['base_seed']) for r, keep in zip(self.rows, chosen, strict=True) if keep}
        return dict(chosen=chosen, rows=n, distinct_episode_count=len(frequencies),
            distinct_scenario_seed_count=len(scenes), max_episode_fraction=max(frequencies.values()) / n if n else 0.)

    def selection(self, selected):
        require(all(type(i) is int and i in self.singletons for i in selected), 'invalid selected leaf IDs')
        return self.singletons[next(iter(selected))] if len(selected) == 1 else self._selection(selected)

    def current(self, a):
        p, sampled = a['p'], a['sampled']
        require(type(p) is np.ndarray and p.dtype == np.float32 and p.shape == (self.n, 5)
            and type(sampled) is np.ndarray and sampled.dtype == np.int64 and sampled.shape == (self.n,),
            'current NN targets/sampled shape or type differs')
        # These expressions, dtypes and reduction order are exactly native.
        nn = p.argmax(-1)
        kl = np.sum(p * (np.log(np.maximum(p, np.float32(1.e-8))) - np.log(np.maximum(self.q, np.float32(1.e-8)))), axis=-1)
        return dict(p=p, sampled=sampled, nn=nn, kl=kl)

    def metric(self, current, selected, purpose):
        require(purpose in ('feedback', 'explanation'), 'invalid native metric purpose')
        p, sampled, nn, kl = (current[k] for k in ('p', 'sampled', 'nn', 'kl'))
        top, legal, path, valid, m = self.top, self.legal, self.path, self.valid, self.m
        fixed = self.selection(selected); chosen, n = fixed['chosen'], fixed['rows']
        matched, expected = valid & (top == sampled), valid * p[np.arange(len(p)), top]
        if purpose == 'feedback':
            matched &= sampled == nn
            expected *= top == nn
        agreement = float((top[chosen] == nn[chosen]).mean()) if n else 0.
        result = dict(sample_count=self.n, physical_frame_count=self.n, excluded_frames=0,
            rows=n, coverage=n / self.n, actionable_rows=int((chosen & matched).sum()),
            actionable_coverage=float((chosen & matched).sum() / self.n),
            expected_actionable_rows=float(expected[chosen].sum()), expected_actionable_coverage=float(expected[chosen].sum() / self.n),
            agreement=agreement, agreement_wilson_lower_90=_wilson(agreement, n),
            kl=float(kl[chosen].mean()) if n else None, illegal_count=int((chosen & ~legal).sum()),
            masked_illegal_count=int((chosen & self.masked_illegal).sum()), path_uncertified_count=int((chosen & ~path).sum()),
            whole_tree_raw_illegal_count=int((~legal).sum()), whole_tree_path_uncertified_count=int((~path).sum()),
            certified_rows=int((chosen & valid).sum()), distinct_episode_count=fixed['distinct_episode_count'],
            distinct_scenario_seed_count=fixed['distinct_scenario_seed_count'], max_episode_fraction=fixed['max_episode_fraction'],
            whole_tree_agreement=float((top == nn).mean()), whole_tree_kl=float(kl.mean()))
        result['per_nn_action'] = {action: dict(physical_frames=int((nn == i).sum()), rows=int((chosen & (nn == i)).sum()),
            qualified_coverage=float((chosen & (nn == i)).sum() / (nn == i).sum()) if (nn == i).any() else None,
            agreement=float((top[chosen & (nn == i)] == nn[chosen & (nn == i)]).mean()) if (chosen & (nn == i)).any() else None,
            actual_actionable_rows=int((chosen & matched & (nn == i)).sum()),
            expected_actionable_rows=float(expected[chosen & (nn == i)].sum())) for i, action in enumerate(ACTIONS)}
        result['public_strata'] = {}
        for label, take in self.strata.items():
            selected_rows, denominator = take & chosen, int(take.sum()); support = int(selected_rows.sum())
            result['public_strata'][label] = dict(rows=support, physical_frames=denominator,
                qualified_coverage=support / denominator if denominator else None,
                agreement=float((top[selected_rows] == nn[selected_rows]).mean()) if support else None,
                kl=float(kl[selected_rows].mean()) if support else None,
                illegal_count=int((selected_rows & ~legal).sum()), actual_actionable_rows=int((selected_rows & matched).sum()),
                expected_actionable_rows=float(expected[selected_rows].sum()))
        return result


class StaticMetricsCache:
    """One validated immutable program, with four exact static partitions.

    Each evaluate still needs independent current-NN provenance verification
    by its caller. A matching statistics cache is never an evidence capability.
    """
    def __init__(self, program_ref, parts):
        require(set(parts) == set(native.SPLITS), 'all four partitions required for static binding')
        self.program_ref = deepcopy(program_ref); self.dependencies = _dependency_refs(program_ref)
        encoded = _reference(program_ref)
        self._program = Program.from_dict(encoded, encoded['target_actor_role_sha256'])
        self._program_state_sha = _digest(vars(self._program)); self.sources = source_files()
        self.bindings = {s: _part_binding(a) for s, a in parts.items()}
        self.cache_key = _digest(dict(version=VERSION, complete_program=self.program_ref,
            dependencies=self.dependencies, source_files=self.sources, partitions=self.bindings, numpy_version=np.__version__))
        counts = np.bincount(self._program.leaf_indices(parts['fit']['x']), minlength=len(self._program.leaves)).tolist()
        self._counts = tuple(counts); proofs = self._program.certificates()
        self._parts = {s: _StaticPart(self._program, parts[s], proofs) for s in ('eligibility', 'validation')}
        self._episodes = {s: sorted({r['episode'] for r in a['rows']}) for s, a in parts.items()}
        self._scenes = {s: sorted({r['base_seed'] for r in a['rows']}) for s, a in parts.items()}

    def verify(self, parts):
        require(source_files() == self.sources and _digest(vars(self._program)) == self._program_state_sha,
            'static metrics source or in-memory program changed')
        for ref in self.dependencies: _reference(ref)
        require(set(parts) == set(native.SPLITS) and {s: _part_binding(a) for s, a in parts.items()} == self.bindings,
            'static metrics x/m/complete rows binding changed')

    def current_report(self, parts):
        self.verify(parts)
        e = self._parts['eligibility']; dynamic = e.current(parts['eligibility'])
        eligibility = {str(i): {purpose: e.metric(dynamic, {i}, purpose) for purpose in ('feedback', 'explanation')}
            for i in range(len(self._program.leaves))}
        feedback = frozenset(i for i, count in enumerate(self._counts) if count and native._eligible(eligibility[str(i)]['feedback'], .93))
        explanation = frozenset(i for i, count in enumerate(self._counts) if count and native._eligible(eligibility[str(i)]['explanation'], .95))
        a = self._parts['validation']; dynamic = a.current(parts['validation'])
        validation = {purpose: a.metric(dynamic, selected, purpose) for purpose, selected in
            (('feedback', feedback), ('explanation', explanation))}
        return native.plain(dict(fit_leaf_counts=list(self._counts), eligibility=eligibility,
            feedback_leaves=sorted(feedback), explanation_leaves=sorted(explanation), validation=validation,
            gate=native.gate(validation), split_episode_ids=deepcopy(self._episodes), split_scenario_ids=deepcopy(self._scenes)))
