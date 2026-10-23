"""Exact synthetic statistics only: no NN forward, fitting or environment reset."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from backend.training import warehouse_v642_static_metrics as cache_module
from backend.training.warehouse_v642_program import import_previous
from backend.training.warehouse_v625_program import _metrics
from backend.training.warehouse_v629_online_evidence import current_report
from experiments.warehouse.v626_protocol import pointer


def strict_equal(a, b, path='root'):
    assert type(a) is type(b), (path, type(a), type(b))
    if type(a) is float: assert a.hex() == b.hex(), (path, a.hex(), b.hex())
    elif type(a) is dict:
        assert set(a) == set(b), path
        for k in a: strict_equal(a[k], b[k], path + '.' + k)
    elif type(a) in (list, tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)): strict_equal(x, y, path + f'[{i}]')
    else: assert a == b, (path, a, b)


@pytest.fixture
def prepared(tmp_path):
    spec = importlib.util.spec_from_file_location('static_metrics_synthetic_fixture', Path(__file__).with_name('test_warehouse_v629.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    previous, parts = module.fixture(tmp_path)
    program = import_previous(previous, 12)
    path = tmp_path / 'v642-program.json'; path.write_text(json.dumps(program.to_dict())); path.chmod(0o444)
    compiled = cache_module.StaticMetricsCache(pointer(path), parts)
    return program, parts, compiled, path


def test_complete_current_report_all_fields_and_float_hex_match_native(prepared):
    p, parts, compiled, _ = prepared
    strict_equal(compiled.current_report(parts), current_report(p, parts))
    assert all(compiled.bindings[s]['rows_sha256'] == cache_module._digest(a['rows']) for s, a in parts.items())


@pytest.mark.parametrize('purpose', ['feedback', 'explanation'])
@pytest.mark.parametrize('selected', [frozenset(), frozenset({0}), frozenset({1, 2}), frozenset({0, 1, 2})])
def test_empty_singleton_and_union_metrics_preserve_every_native_float(prepared, purpose, selected):
    p, parts, compiled, _ = prepared; a = parts['eligibility']; static = compiled._parts['eligibility']
    actual = static.metric(static.current(a), selected, purpose)
    expected = _metrics(p, a['rows'], (a['x'], a['m'], a['sampled']), a['p'], selected, purpose)
    strict_equal(actual, expected)


def test_current_p_sampled_and_qualification_are_not_cached(prepared):
    p, parts, compiled, _ = prepared; before = compiled.current_report(parts); key = compiled.cache_key
    changed = deepcopy(parts)
    for a in changed.values():
        a['p'][:] = np.float32(.2)
        a['sampled'][:] = 0
    actual = compiled.current_report(changed); strict_equal(actual, current_report(p, changed))
    assert actual != before and actual['feedback_leaves'] != before['feedback_leaves']
    assert compiled.cache_key == key
    assert not any(k in vars(compiled) for k in ('p', 'sampled', 'nn', 'feedback_leaves', 'explanation_leaves'))


@pytest.mark.parametrize('kind', ['x', 'm', 'rows', 'program_file', 'program_memory', 'parent', 'source'])
def test_all_static_changes_invalidate_cache(prepared, kind, monkeypatch):
    _, parts, compiled, path = prepared
    if kind in ('x', 'm'):
        parts['calibration'][kind][0, 0] += np.float32(.01)
    elif kind == 'rows': parts['fit']['rows'][0]['task'] = 42
    elif kind == 'program_file':
        path.chmod(0o644); path.write_bytes(path.read_bytes() + b' ')
    elif kind == 'program_memory': compiled._program.leaves[0]['bias'][4] += .01
    elif kind == 'parent':
        parent = Path(compiled._program.protection['parent']['path']); parent.write_bytes(parent.read_bytes() + b' ')
    else:
        sources = dict(compiled.sources); sources['synthetic-new-source'] = 'f' * 64
        monkeypatch.setattr(cache_module, 'source_files', lambda: sources)
    with pytest.raises((RuntimeError, ValueError)): compiled.current_report(parts)


def test_reuse_calls_no_certificates_route_or_program_forward(prepared, monkeypatch):
    p, parts, compiled, _ = prepared; expected = current_report(p, parts)
    def forbidden(*a, **kw): raise AssertionError('static scientific operation repeated')
    for name in ('certificates', 'leaf_indices', 'distributions'):
        monkeypatch.setattr(type(compiled._program), name, forbidden)
    strict_equal(compiled.current_report(parts), expected)


def test_array_order_and_dynamic_input_contract_are_explicit(prepared):
    p, parts, compiled, _ = prepared
    copied = deepcopy(parts)
    for a in copied.values():
        a['x'] = np.asfortranarray(a['x']); a['m'] = np.asfortranarray(a['m'])
    strict_equal(compiled.current_report(copied), current_report(p, copied))
    copied['validation']['p'] = copied['validation']['p'].astype(np.float64)
    with pytest.raises(RuntimeError, match='current NN targets'): compiled.current_report(copied)
