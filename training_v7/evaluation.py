"""Actual complete-scene metrics and quality-only pair selection."""
from fractions import Fraction
import numpy as np
from .contracts import DOMAINS


def _mean(rows, key):
    if not rows: raise ValueError('no completed evaluation episodes')
    return sum(_value(r, key) for r in rows) / len(rows)


def _value(row, key):
    if key + '_numerator' in row:
        return Fraction(row[key + '_numerator'], row[key + '_denominator'])
    return Fraction(int(row[key])) if isinstance(row[key], (bool, np.bool_)) else Fraction(str(row[key]))


def _score(value):
    return dict(score=float(value), score_fraction=[value.numerator, value.denominator])


def quality(domain, rows, reference, *, formal=False):
    if domain not in DOMAINS or not rows or len(rows) != len(reference):
        raise ValueError('matched complete-scene quality records required')
    keys = [(r['seed'], r['task'], r.get('trajectory', 0), r.get('pairing', 'nn_nn')) for r in rows]
    if len(keys) != len(set(keys)) or keys != [(r['seed'], r['task'], r.get('trajectory', 0), r.get('pairing', 'nn_nn')) for r in reference]:
        raise ValueError('unmatched scenarios or duplicated trajectories')
    if any(r.get('complete') is not True for r in [*rows, *reference]):
        raise ValueError('partial episodes cannot contribute quality or negative costs')
    if domain == 'warehouse':
        delivery, collision, failed = (_mean(rows, k) for k in ('deliveries', 'collisions', 'failed'))
        refd, refc = (_mean(reference, k) for k in ('deliveries', 'collisions'))
        minimum = Fraction(5) if formal else max(Fraction('5.5'), Fraction('.95') * refd)
        maximum = Fraction(8) if formal else min(Fraction(6), refc + 1)
        failure_max = Fraction('.1' if formal else '.05')
        return dict(passed=delivery >= minimum and collision <= maximum and failed <= failure_max,
            delivery=float(delivery), collision=float(collision), failure_rate=float(failed),
            failures=sum(int(r['failed']) for r in rows), episodes=len(rows),
            gates=dict(minimum_delivery=float(minimum), maximum_collision=float(maximum), maximum_failure=float(failure_max)),
            **_score(delivery - Fraction('.25') * collision - 5 * failed))
    if domain == 'pong':
        summaries = []
        for task in (1, 2, 3):
            subset = [r for r in rows if r['task'] == task]; ref = [r for r in reference if r['task'] == task]
            if not subset: raise ValueError('missing task in Pong evaluation')
            capacity, wait = _mean(subset, 'capacity'), _mean(subset, 'wait')
            passed = (capacity >= Fraction('.9') and min(_value(r, 'capacity') for r in subset) >= Fraction('.8')
                      and wait <= Fraction('.1') and max(_value(r, 'wait') for r in subset) <= Fraction('.2'))
            if not formal: passed &= capacity >= Fraction('.95') * _mean(ref, 'capacity')
            summaries.append(dict(task=task, capacity=float(capacity), wait=float(wait),
                minimum_capacity=min(r['capacity'] for r in subset), maximum_wait=max(r['wait'] for r in subset), passed=bool(passed)))
        return dict(passed=all(s['passed'] for s in summaries), tasks=summaries,
                    **_score(_mean(rows, 'capacity') - _mean(rows, 'wait')))
    summaries = []
    for pairing in ('nn_nn', 'offline_human_proxy'):
        subset = [r for r in rows if r['pairing'] == pairing]
        ref = [r for r in reference if r['pairing'] == pairing]
        if not subset: raise ValueError('both NN and explicitly labelled offline proxy evaluations required')
        completed = sum(r['completed'] == 5 for r in subset)
        rate = Fraction(completed, len(subset)); ref_rate = Fraction(sum(r['completed'] == 5 for r in ref), len(ref))
        passed = rate >= Fraction('.8' if formal else '.95')
        if not formal: passed &= rate >= Fraction('.95') * ref_rate
        summaries.append(dict(pairing=pairing, completed=completed, episodes=len(subset), completion_rate=float(rate), passed=bool(passed)))
    exact = min(Fraction(s['completed'], s['episodes']) for s in summaries)
    return dict(passed=all(s['passed'] for s in summaries), pairings=summaries,
                **_score(exact), human_participants=0)


def select_quality_only(pairs):
    # Program fields are deliberately not accepted at this selection interface.
    required = {'pair_id', 'checkpoint_id', 'raw', 'regularized'}
    if not pairs or any(set(p) != required for p in pairs): raise ValueError('only quality may select a pair')
    eligible = [p for p in pairs if p['raw']['passed'] and p['regularized']['passed']]
    if not eligible: raise ValueError('no pair passes quality in both treatments')
    exact = lambda q: Fraction(*q['score_fraction']) if 'score_fraction' in q else Fraction(str(q['score']))
    return min(eligible, key=lambda p: (-min(exact(p['raw']), exact(p['regularized'])),
                                      str(p['pair_id']), p['checkpoint_id']))


def common_budget_lock(results):
    expected_domains = set(DOMAINS)
    for budget in (200000, 300000, 400000):
        if budget not in results: raise ValueError('missing next common budget evidence')
        current = results[budget]
        if set(current) != expected_domains or any(set(current[d]) != {0, 1, 2} for d in current):
            raise ValueError('all nine complete development pairs required; no pooling')
        if all(current[d][i]['passed'] is True for d in current for i in current[d]):
            if any(b > budget for b in results): raise ValueError('continued budget search after first joint success')
            return budget
    raise ValueError('all registered common budgets failed; candidate closed')


def scene_bootstrap(rows, statistic, *, seed=20261024, draws=10000):
    rng = np.random.default_rng(seed)
    groups = {}
    for r in rows: groups.setdefault((r['seed'], r['task']), []).append(r)
    by_task = {t: [k for k in groups if k[1] == t] for t in (1, 2, 3)}
    if any(not keys for keys in by_task.values()): raise ValueError('balanced task strata required')
    values = []
    for _ in range(draws):
        sampled = [r for keys in by_task.values() for j in rng.integers(0, len(keys), len(keys)) for r in groups[keys[j]]]
        values.append(statistic(sampled))
    return dict(lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)),
                independent_scenes=len(groups), draws=draws, unit='complete_scene', seed=seed)
