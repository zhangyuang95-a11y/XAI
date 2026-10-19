"""Fit-only synthetic checks; no environment, rollout or sealed data."""
from dataclasses import replace
from hashlib import sha256
import inspect
import itertools
import json

import numpy as np
import pytest

from backend.training import warehouse_v622_search as search
from backend.training import warehouse_program_search as old
from backend.training.warehouse_v619_tree import _Decision, _Region, _Leaf, _export


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
NAMES = ("position", "context", "public_hard_risk_context", *("legal_"+a.lower() for a in ACTIONS))


def data(n=480, split=False):
    x = np.zeros((n, len(NAMES)), np.float32)
    x[:, 0] = np.resize(np.arange(24, dtype=np.float32)/24, n)
    m = np.ones((n, 5), np.float32)
    x[:, 3:] = m
    chosen = np.where(x[:, 0] < .5, 2, 3) if split else np.full(n, 4)
    z = np.zeros((n, 5), np.float32); z[np.arange(n), chosen] = 5
    p = np.exp(z-z.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    return x, m, p, ["fit:"+str(i//24) for i in range(n)]


def config(**kwargs):
    base = dict(initializations=1, max_rounds=2, max_leaf_optimization_calls=100,
        axis_features=2, semantic_candidates_per_node=2, proposals_per_round=12,
        template_count=1, template_max_leaf_optimization_calls=200,
        table_max_leaf_optimization_calls=40, template_candidates_per_growth=4, max_depth=3, max_leaves=8)
    base.update(kwargs)
    return replace(search.PRODUCTION_SEARCH_CONFIG, **base)


def fit(arm="A32", rows=None, cfg=None):
    return search.fit_structure(arm, *(data() if rows is None else rows), feature_names=NAMES,
        actor_role_sha256="a"*64, config=config() if cfg is None else cfg)


def prove(structure, rows, depth=6, leaves=32):
    x, m = rows[:2]
    search._verify_export(structure, x, m, NAMES, depth, leaves)
    mask = np.array([[bool(bits & (1 << a)) for a in range(4)]+[True] for bits in range(16)], np.float32)
    unseen = np.zeros((len(mask)*3, len(NAMES)), np.float32)
    unseen[:, 0] = np.tile([0, .5, 1], 16)
    unseen[:, 3:] = np.repeat(mask, 3, axis=0)
    search._verify_export(structure, unseen, np.repeat(mask, 3, axis=0), NAMES, depth, leaves)


def test_versioned_configuration_does_not_change_old_limits_or_sources():
    before = sha256(inspect.getsource(old).encode()).hexdigest()
    assert search.PRODUCTION_SEARCH_CONFIG.max_depth == 6
    assert search.PRODUCTION_SEARCH_CONFIG.max_leaves == 32
    assert search.PRODUCTION_SEARCH_CONFIG.max_leaf_optimization_calls == 19968
    assert search.PRODUCTION_SEARCH_CONFIG.wall_time_seconds == 5400
    assert old.PRODUCTION_SEARCH_CONFIG.max_depth == 5
    assert old.PRODUCTION_SEARCH_CONFIG.max_leaves == 16
    assert sha256(inspect.getsource(old).encode()).hexdigest() == before
    for update in ({"max_depth": 7}, {"max_leaves": 33}, {"max_rounds": 61},
                   {"max_leaf_optimization_calls": 19969}, {"wall_time_seconds": 5401},
                   {"template_count": 5}, {"table_max_leaf_optimization_calls": 509}):
        with pytest.raises(ValueError): config(**update)


def test_A32_reduced_capacity_has_same_M2_structure_and_probabilities():
    rows, cfg = data(split=True), config(max_depth=3, max_leaves=8)
    actual, report = fit(rows=rows, cfg=cfg)
    legacy_config = old.SearchConfig(**{name: getattr(cfg, name) for name in old.SearchConfig.__dataclass_fields__})
    expected, legacy_report = old.fit_legal_structure(*rows, feature_names=NAMES,
        actor_role_sha256="a"*64, method=old.ROBUST_METHOD, config=legacy_config)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert report["fit_masked_kl"] == legacy_report["fit_masked_kl"]
    assert report["leaf_optimization_calls"] == legacy_report["leaf_optimization_calls"]
    prove(actual, rows)


def test_A32_global_rounds_calls_solver_ledger_and_final_reserve():
    structure, report = fit(rows=data(split=True), cfg=config(initializations=8, max_rounds=7,
        max_leaf_optimization_calls=90))
    assert len(report["search_runs"]) == 8
    assert sum(len(r["rounds"]) for r in report["search_runs"]) == report["global_search_rounds"] <= 7
    assert report["leaf_optimization_calls"] == len(report["leaf_api_attempts"]) <= 90
    assert report["actual_convex_solver_calls"] == sum(e["actual_convex_solver_calls"] for e in report["leaf_api_attempts"])
    assert report["actual_convex_solver_calls"] <= 5*report["leaf_optimization_calls"]
    assert report["bootstrap_leaf_optimization_calls"]+sum(r["leaf_optimization_calls"]
        for r in report["search_runs"]) == report["leaf_optimization_calls"]
    assert report["bootstrap_actual_convex_solver_calls"]+sum(r["actual_convex_solver_calls"]
        for r in report["search_runs"]) == report["actual_convex_solver_calls"]
    assert report["final_refit_reserved_calls"] == 32
    assert report["maximum_end_to_end_calls"] == report["leaf_optimization_calls"]+32 <= 20000
    assert report["global_tree_optimality_gap"] is None
    assert report["input_identity"]["fit_target_sha256"] == sha256(rows_bytes(data(split=True)[2])).hexdigest()
    json.dumps(report, allow_nan=False)
    prove(structure, data(split=True))


def rows_bytes(value):
    return np.ascontiguousarray(value, dtype="<f4").tobytes()


def test_failed_API_attempts_are_spent_and_partial_tree_never_wins(monkeypatch):
    original, attempts = search.fit_constrained_leaf_distribution, []
    def solver(target, masks, legal, **kwargs):
        attempts.append(tuple(legal))
        if len(legal) > 1:
            error = RuntimeError("synthetic leaf failure")
            error.enumerated_action_solves = len(legal)
            raise error
        return original(target, masks, legal, **kwargs)
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", solver)
    structure, report = fit(rows=data(split=True), cfg=config(max_leaf_optimization_calls=12))
    assert report["leaf_optimization_calls"] == len(attempts) <= 12
    assert report["actual_convex_solver_calls"] == sum(map(len, attempts))
    assert any(not e["success"] for e in report["leaf_api_attempts"])
    prove(structure, data(split=True))


def test_deadline_preserves_completed_safe_incumbent_and_marks_unstarted_runs(monkeypatch):
    current = [0.]
    monkeypatch.setattr(search, "monotonic", lambda: current[0])
    original = search.fit_constrained_leaf_distribution
    def solver(*args, **kwargs):
        result = original(*args, **kwargs)
        current[0] = 11.
        return result
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", solver)
    structure, report = fit(cfg=config(initializations=8, wall_time_seconds=10))
    assert report["wall_time_budget_exhausted"] and report["complete_incumbent"]
    assert report["leaf_optimization_calls"] == 1
    assert all(r["stop_reason"] == "wall_time_budget" for r in report["search_runs"])
    assert report["global_search_rounds"] == 0
    prove(structure, data())


def test_deadline_inside_actual_solver_spends_the_attempt(monkeypatch):
    import scipy.optimize
    observed = []
    class CounterClock:
        def __init__(self): self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls >= 3: raise search._DeadlineExhausted("synthetic deadline")
    def solver(fun, initial, **kwargs):
        observed.append(1)
        fun(initial)
        fun(initial)
    monkeypatch.setattr(scipy.optimize, "minimize", solver)
    p = np.array([[.1, .1, .1, .1, .6]])
    with pytest.raises(search._DeadlineExhausted) as error:
        search.fit_constrained_leaf_distribution(p, np.ones_like(p), (4,), clock=CounterClock())
    assert len(observed) == error.value.enumerated_action_solves == 1
    assert error.value.report["wall_time_budget_exhausted"]


def test_no_fitted_root_returns_typed_failure_not_invented_logits(monkeypatch):
    def fail(*args, **kwargs):
        error = RuntimeError("no complete fit")
        error.enumerated_action_solves = 1
        raise error
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", fail)
    with pytest.raises(search.SearchIncompleteError) as error:
        fit()
    assert not error.value.report["complete_incumbent"]
    assert error.value.report["leaf_optimization_calls"] == 1


@pytest.mark.parametrize("arm", ["A32", "D32"])
@pytest.mark.parametrize("bad", ["illegal_mass", "illegal_mask", "wrong_encoded", "missing_episode", "nan_target"])
def test_only_complete_masked_fit_distribution_is_accepted(arm, bad):
    x, m, p, e = data()
    if bad == "illegal_mass":
        m[0, 0] = 0; x[0, 3] = 0; p[0, 0] = .01
    elif bad == "illegal_mask": m[0, 4] = 0
    elif bad == "wrong_encoded": x[0, 3] = 0
    elif bad == "missing_episode": e.pop()
    elif bad == "nan_target": p[0, 0] = np.nan
    with pytest.raises(ValueError): fit(arm, (x, m, p, e))


def test_full_distribution_targets_not_reduced_to_hard_labels(monkeypatch):
    rows = data()
    p = rows[2].copy()
    p[:, 0], p[:, 1], p[:, 4] = .05, .10, .85
    p[:, 2:4] = 0
    rows = rows[0], rows[1], p, rows[3]
    original, supplied = search.fit_constrained_leaf_distribution, []
    def solver(target, masks, legal, **kwargs):
        supplied.append(target.copy())
        return original(target, masks, legal, **kwargs)
    monkeypatch.setattr(search, "fit_constrained_leaf_distribution", solver)
    fit(rows=rows, cfg=config(max_leaf_optimization_calls=1))
    np.testing.assert_array_equal(supplied[0], p)
    assert np.all(np.count_nonzero(supplied[0], axis=1) == 3)


def _balanced(depth, threshold=0.):
    if not depth: return None
    return _Decision(tuple([1.]+[0.]*(len(NAMES)-1)), threshold,
        _balanced(depth-1, threshold-.1), _balanced(depth-1, threshold+.1))


def test_export_really_accepts_32_leaves_and_six_deep_paths():
    x, m, _, _ = data()
    for tree in (_balanced(5), _Decision(tuple([1.]+[0.]*(len(NAMES)-1)), 0., None, _balanced(5))):
        # The second 33-leaf template is pruned once to 32 while keeping depth6.
        if search._size(tree)[0] == 33:
            tree = search._replace(tree, (1, 0, 0, 0, 0), None)
        regions = list(search._FitContext(x, m, data()[2], np.array(data()[3]), NAMES, tuple(range(3, 8)),
            search._Clock(10), 1).regions(tree))
        leaves = [_Leaf(r, np.array([0, 0, 0, 0, 2], np.float32), {}, False, 0., 0., 0.)
                  for r in regions if r.tree is None]
        structure = _export(tree, leaves, len(NAMES))
        assert structure["leaf_count"] == 32
        prove(structure, data())
        if search._size(tree)[1] == 6:
            with pytest.raises(RuntimeError): prove(structure, data(), depth=5)


def test_D32_phase_budgets_frozen_outputs_certificate_and_JSON():
    rows = data(split=True)
    structure, report = fit("D32", rows, config(template_count=2, template_max_leaf_optimization_calls=250,
        table_max_leaf_optimization_calls=40, max_depth=3, max_leaves=8))
    assert report["complete_frozen_tables"] == 2
    assert report["template_generation_leaf_optimization_calls"] <= 250
    assert report["frozen_table_leaf_optimization_calls"] <= 40
    assert report["final_refit_reserved_calls"] == 0
    assert report["maximum_end_to_end_calls"] == report["leaf_optimization_calls"]
    assert report["certificate"]["restricted_table_gap"] == 0
    assert report["certificate"]["global_function_class_optimality_gap"] is None
    assert not report["certificate"]["continuous_leaf_output_optimality_claimed"]
    cert = structure["frozen_table_certificate"]
    assert cert["selected_leaf_logits_sha256"] == sha256(rows_bytes(structure["initial_leaf_logits"])).hexdigest()
    table = next(t for t in report["frozen_terminal_tables"] if t["template"] == report["selected_template"])
    assert table["frozen_terminal_table_sha256"] == cert["selected_table_sha256"]
    by_path = {tuple(r["path"]): r for r in table["records"]}
    expected = np.asarray([by_path[tuple(p)]["logits"] for p in cert["selected_terminal_paths"]], np.float32)
    np.testing.assert_array_equal(structure["initial_leaf_logits"], expected)
    assert structure["initial_leaf_logits"].dtype == np.float32
    for phase in ("generation", "terminal_table"):
        assert report[phase]["leaf_optimization_calls"] == len(report[phase]["leaf_api_attempts"])
        assert report[phase]["actual_convex_solver_calls"] == sum(e["actual_convex_solver_calls"]
            for e in report[phase]["leaf_api_attempts"])
    assert len(report["fit_leaf_reports"]) == structure["leaf_count"]
    assert all("table_node_id" in r for r in report["fit_leaf_reports"])
    assert structure["fit_leaf_reports"] == report["fit_leaf_reports"]
    assert structure["fit_target_sha256"] == report["input_identity"]["fit_target_sha256"]
    json.dumps(report, allow_nan=False)
    prove(structure, rows)


def test_D32_no_complete_table_is_typed_failure_not_fake_gap(monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError("synthetic table failure")
    monkeypatch.setattr(search, "_freeze_table", fail)
    with pytest.raises(search.SearchIncompleteError) as error:
        fit("D32")
    assert error.value.report["complete_generation_incumbent_available_but_not_a_DP_certificate"]
    assert not error.value.report["complete_incumbent"]


def test_D32_later_incomplete_table_keeps_only_previous_certified_family(monkeypatch):
    original, calls = search._freeze_table, []
    def freeze(ctx, tree):
        calls.append(1)
        if len(calls) == 2:
            ctx.clock.timed_out = True
            raise search._DeadlineExhausted("synthetic later deadline")
        return original(ctx, tree)
    monkeypatch.setattr(search, "_freeze_table", freeze)
    structure, report = fit("D32", cfg=config(template_count=2, template_max_leaf_optimization_calls=250))
    assert report["complete_frozen_tables"] == 1
    assert not report["certificate"]["all_requested_templates_completed_and_enumerated"]
    assert report["certificate"]["restricted_table_gap"] == 0
    assert report["wall_time_budget_exhausted"]
    prove(structure, data())


def test_D32_later_DP_deadline_does_not_certify_unscored_table(monkeypatch):
    original, calls = search._prune_frozen_tree, []
    def prune(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            kwargs["clock"].timed_out = True
            raise search._DeadlineExhausted("synthetic deadline inside DP")
        return original(*args, **kwargs)
    monkeypatch.setattr(search, "_prune_frozen_tree", prune)
    structure, report = fit("D32", cfg=config(template_count=2,
        template_max_leaf_optimization_calls=250))
    assert report["complete_frozen_tables"] == 1
    assert report["template_generation_reports"][1]["frozen_table_completed"]
    assert not report["template_generation_reports"][1]["dp_completed"]
    assert not report["certificate"]["all_requested_templates_completed_and_enumerated"]
    assert report["wall_time_budget_exhausted"]
    prove(structure, data())


def test_fixed_table_DP_matches_exhaustive_prunings_and_keeps_both_children():
    tree = _balanced(2)
    paths = [(), (0,), (0, 0), (0, 1), (1,), (1, 0), (1, 1)]
    scores = [(0, 0, -20, -1), (20, 10, -8, -1), (20, 18, -1, -1), (20, 18, -1, -1),
              (0, 0, -8, -1), (15, 12, -2, -1), (15, 12, -2, -1)]
    table = [{"path": list(path), "score": list(score)} for path, score in zip(paths, scores)]
    records = {path: score for path, score in zip(paths, scores)}
    def enumerate_prunings(node, path=()):
        result = [(1, records[path], (path,))]
        if node is not None:
            for l, r in itertools.product(enumerate_prunings(node.left, path+(0,)),
                                           enumerate_prunings(node.right, path+(1,))):
                result.append((l[0]+r[0], search._add_scores(l[1], r[1]), l[2]+r[2]))
        return result
    for maximum in (1, 2, 3, 4):
        winner, certificate = search._prune_frozen_tree(tree, table, maximum)
        expected = max(v[1] for v in enumerate_prunings(tree) if v[0] <= maximum)
        assert winner[0] == expected
        assert search._size(winner[1])[0] <= maximum
        assert certificate["complete_DP"]
        # Every retained path retains all ancestor decisions; a lone child
        # subtree can never be promoted past a physical guard.
        chosen = set(winner[2])
        assert all(not any(other != path and path[:len(other)] == other for other in chosen) for path in chosen)


def test_DP_64_leaf_template_is_globally_pruned_to_32_not_truncated():
    tree = _balanced(6)
    table = []
    def records(node, path=()):
        # Reward all 32 depth5 regions; expanding to depth6 provides no
        # additional first component and loses on the leaf-count tie break.
        score = (float(2**(6-len(path))) if len(path) >= 5 else 0., 0., 0., -1.)
        table.append({"path": list(path), "score": list(score)})
        if node is not None:
            records(node.left, path+(0,)); records(node.right, path+(1,))
    records(tree)
    winner, cert = search._prune_frozen_tree(tree, table, 32)
    assert len(table) == 127
    assert search._size(winner[1]) == (32, 5)
    assert winner[0][0] == 64
    assert cert["complete_DP"]


def test_three_source_cost_including_derived_context_is_enforced():
    ctx = search._FitContext(*data()[:3], np.array(data()[3]), NAMES, tuple(range(3, 8)), search._Clock(10), 100)
    region = next(ctx.regions(None))
    predicates = search._template_predicates(ctx, region, search.SEARCH_INITIALIZATIONS[0], config())
    assert all(search.source_feature_cost(w, NAMES) <= 3 for w, *_ in predicates)
