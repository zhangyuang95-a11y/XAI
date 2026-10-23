"""Software-only preparation checks; no new scenes, NN or evidence claims."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from backend.training import warehouse_v642_prepared_program as prepared_module
from backend.training.warehouse_v642_program import Program as NativeProgram, import_previous, qualify
from backend.training.warehouse_v629_online_evidence import current_report
from experiments.warehouse.v626_protocol import pointer


def strict_equal(a, b, path='root'):
    assert type(a) is type(b), (path, type(a), type(b))
    if type(a) is float: assert a.hex() == b.hex(), (path, a.hex(), b.hex())
    elif type(a) is dict:
        assert set(a) == set(b), path
        for key in a: strict_equal(a[key], b[key], path + '.' + key)
    elif type(a) in (tuple, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)): strict_equal(x, y, path + f'[{i}]')
    else: assert a == b, (path, a, b)


@pytest.fixture
def prepared(tmp_path):
    spec = importlib.util.spec_from_file_location('prepared_program_synthetic_fixture', Path(__file__).with_name('test_warehouse_v629.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    previous, parts = module.fixture(tmp_path)
    native = qualify(import_previous(previous, 12), parts)
    file = tmp_path / 'prepared-source.json'; file.write_text(json.dumps(native.to_dict(), sort_keys=True)); file.chmod(0o444)
    compiled = prepared_module.PreparedProgram.from_reference(pointer(file), target_actor_role_sha256=native.target_actor_role_sha256)
    return native, parts, compiled, file


def test_native_subclass_and_exact_serialization_forward_and_current_report(prepared):
    native, parts, compiled, _ = prepared
    assert isinstance(compiled, NativeProgram)
    strict_equal(compiled.to_dict(), native.to_dict())
    for split, part in parts.items():
        for actual, expected in zip(compiled.distributions(part['x'], part['m']), native.distributions(part['x'], part['m']), strict=True):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes(), split
        assert np.array_equal(compiled.leaf_indices(part['x']), native.leaf_indices(part['x']))
    strict_equal(current_report(compiled, parts), current_report(native, parts))
    assert compiled.preparation_binding()['caches_current_NN_qualification'] is False


def test_current_targets_are_recomputed_and_no_current_report_is_cached(prepared):
    native, parts, compiled, _ = prepared
    before = current_report(compiled, parts); changed = deepcopy(parts)
    for part in changed.values():
        part['p'][:] = np.float32(.2); part['sampled'][:] = 0
    after = current_report(compiled, changed)
    assert after != before
    strict_equal(after, current_report(native, changed))


@pytest.mark.parametrize('field', ['structure', 'leaf', 'report', 'protection', 'feedback', 'explanation', 'target', 'anchor', 'arm', 'depth', 'capacity', 'role', 'extra'])
def test_any_original_field_mutation_invalidates_preparation(prepared, field):
    _, _, compiled, _ = prepared
    if field == 'structure': compiled.structure['split_bias'][0] += .001
    elif field == 'leaf': compiled.leaves[0]['bias'][4] += .001
    elif field == 'report': compiled.report['fit_leaf_counts'][0] += 1
    elif field == 'protection': compiled.protection['entries'][0]['legal'].append(42)
    elif field == 'feedback': compiled.feedback_leaves = frozenset({999})
    elif field == 'explanation': compiled.explanation_leaves = frozenset({999})
    elif field == 'target': compiled.target_actor_role_sha256 = '0' * 64
    elif field == 'anchor': compiled.actor_role_sha256 = '0' * 64
    elif field == 'arm': compiled.arm = 'C2'
    elif field == 'depth': compiled.maximum_depth += 1
    elif field == 'capacity': compiled.maximum_leaves += 1
    elif field == 'role': compiled.role = 'robot_2' if compiled.role == 'robot_1' else 'robot_1'
    else: compiled.unregistered = True
    with pytest.raises(RuntimeError, match='fields or types'): compiled.to_dict()
    with pytest.raises(RuntimeError, match='fields or types'): compiled.validate()


@pytest.mark.parametrize('field', ['signed_zero', 'bool_for_int', 'int_for_float', 'tuple_for_list', 'set_for_frozenset', 'numpy_scalar'])
def test_type_and_float_bits_are_not_python_equality(prepared, field):
    _, _, compiled, _ = prepared
    if field == 'signed_zero':
        # A zero coefficient in this source is +0.0.  -0.0 compares == but hashes differently.
        vector = compiled.structure['split_weights'][0]
        i = next(i for i, value in enumerate(vector) if type(value) is float and value == 0.)
        assert float(vector[i]).hex() == '0x0.0p+0'; vector[i] = -0.0
    elif field == 'bool_for_int': compiled.maximum_depth = bool(compiled.maximum_depth)
    elif field == 'int_for_float':
        compiled.structure['split_weights'][0][0] = int(compiled.structure['split_weights'][0][0])
    elif field == 'tuple_for_list': compiled.leaves = tuple(compiled.leaves)
    elif field == 'set_for_frozenset': compiled.feedback_leaves = set(compiled.feedback_leaves)
    else: compiled.maximum_depth = np.int64(compiled.maximum_depth)
    with pytest.raises(RuntimeError): compiled.to_dict()


@pytest.mark.parametrize('kind', ['source_program', 'parent', 'missing_parent', 'symlink_parent', 'source_code'])
def test_real_source_and_transitive_parent_files_are_rehashed(prepared, kind, monkeypatch, tmp_path):
    _, _, compiled, file = prepared
    if kind == 'source_program':
        file.chmod(0o644); file.write_bytes(file.read_bytes() + b' ')
    elif kind in ('parent', 'missing_parent', 'symlink_parent'):
        parent = Path(compiled.protection['parent']['path'])
        if kind == 'parent': parent.write_bytes(parent.read_bytes() + b' ')
        elif kind == 'missing_parent': parent.unlink()
        else:
            moved = tmp_path / 'moved-parent.json'; moved.write_bytes(parent.read_bytes()); parent.unlink(); parent.symlink_to(moved)
    else:
        sources = dict(prepared_module.source_files()); sources['new-or-changed-source'] = 'f' * 64
        monkeypatch.setattr(prepared_module, 'source_files', lambda: sources)
    with pytest.raises(RuntimeError): compiled.to_dict()


def test_returned_payload_never_aliases_cache_or_original_fields(prepared):
    _, _, compiled, _ = prepared
    expected = compiled.to_dict(); result = compiled.to_dict(); result['leaves'][0]['bias'][0] += .5
    strict_equal(compiled.to_dict(), expected)
    assert compiled.validate() is compiled


def test_warm_reuse_performs_no_native_static_proof_work(prepared, monkeypatch):
    _, _, compiled, _ = prepared; expected = compiled.to_dict()
    def forbidden(*a, **kw): raise AssertionError('native static proof repeated')
    monkeypatch.setattr(NativeProgram, 'validate', forbidden)
    monkeypatch.setattr(NativeProgram, 'to_dict', forbidden)
    monkeypatch.setattr(NativeProgram, 'from_dict', forbidden)
    strict_equal(compiled.to_dict(), expected)
    assert compiled.validate() is compiled


def test_unbound_construction_wrong_target_and_unknown_subclass_refused(prepared):
    native, _, _, file = prepared
    with pytest.raises(RuntimeError): prepared_module.PreparedProgram()
    with pytest.raises(RuntimeError): prepared_module.PreparedProgram.from_dict(native.to_dict(), native.target_actor_role_sha256)
    with pytest.raises((RuntimeError, ValueError)):
        prepared_module.PreparedProgram.from_reference(pointer(file), target_actor_role_sha256='0' * 64)
    class Unknown(prepared_module.PreparedProgram): pass
    with pytest.raises(RuntimeError): Unknown.from_reference(pointer(file), target_actor_role_sha256=native.target_actor_role_sha256)


def test_hash_binding_preserves_supported_scalar_and_container_types():
    values = [None, False, 0, 0., -0., '0', [0], (0,), {0}, frozenset({0})]
    assert len({prepared_module._typed_sha(value) for value in values}) == len(values)
    assert prepared_module._typed_sha({'a': 1, 'b': 2}) == prepared_module._typed_sha({'b': 2, 'a': 1})
    with pytest.raises(RuntimeError): prepared_module._typed_sha(float('nan'))
    with pytest.raises(RuntimeError): prepared_module._typed_sha(np.float32(0))


def test_duplicate_and_nonfinite_source_json_rejected(tmp_path):
    for name, text in [('duplicate', '{"protection":null,"protection":null}'), ('nan', '{"protection":NaN}')]:
        file = tmp_path / (name + '.json'); file.write_text(text)
        with pytest.raises(RuntimeError):
            prepared_module.PreparedProgram.from_reference(pointer(file), target_actor_role_sha256='0' * 64)


def test_recursive_parent_closure_rechecks_actual_grandparent_bytes(tmp_path):
    grandparent = tmp_path / 'grandparent.json'; grandparent.write_text('{"version":"software-fixture"}')
    parent = tmp_path / 'parent.json'; parent.write_text(json.dumps({'protection': {'parent': pointer(grandparent)}}))
    child = tmp_path / 'child.json'; child.write_text(json.dumps({'protection': {'parent': pointer(parent)}}))
    closure = prepared_module._protection_closure(pointer(child))
    assert [p for p, _ in closure] == [str(child), str(parent), str(grandparent)]
    grandparent.write_bytes(grandparent.read_bytes() + b' ')
    with pytest.raises(RuntimeError, match='bytes changed'): prepared_module._protection_closure(pointer(child))


def test_cached_serialization_integrity_is_checked(prepared):
    _, _, compiled, _ = prepared
    # The normal preparation record is frozen; forced low-level corruption is
    # diagnostic only and must not make changed cached bytes reusable.
    object.__setattr__(compiled._prepared, 'serialized', compiled._prepared.serialized + b' ')
    with pytest.raises(RuntimeError, match='serialization changed'): compiled.to_dict()
