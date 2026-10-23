"""Portable evidence tests; no environment, NN training or new trial data."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import shutil

import numpy as np
import pytest

from backend.training import warehouse_v642_portable_program as portable
from backend.training import warehouse_v642_program as native
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v622_program import MASK_INDICES, plain


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False))
    path.chmod(0o444)
    return portable.pointer(path)


@pytest.fixture
def fixture(tmp_path):
    spec = importlib.util.spec_from_file_location('portable_synthetic_parent', Path(__file__).with_name('test_warehouse_v629.py'))
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    previous, parts = helper.fixture(tmp_path)
    previous.maximum_leaves = 256
    program = native.import_previous(previous, 12)
    program.target_actor_role_sha256 = 'b' * 64
    # Add a genuinely variable sparse linear leaf outside the protected leaf.
    i = 1
    leaf = deepcopy(program.leaves[i])
    leaf['support'] = [FEATURE_NAMES.index('self_battery_fraction')]
    leaf['weights'] = [[.25], [-.25], [0.], [0.], [.5]]
    from backend.training.warehouse_v626_certificate import certify
    leaf['certificate'], counter = certify(leaf, native.region(program.structure, i, 256, 12), program.certificates()[i], use_path=True)
    assert not counter
    program.leaves[i] = leaf
    for value in parts.values():
        value['x'][:, FEATURE_NAMES.index('self_battery_fraction')] = np.linspace(.02, .98, len(value['x']), dtype=np.float32)
    native.qualify(program, parts)
    assert program.explanation_leaves
    source = write(tmp_path / 'source-program.json', program.to_dict())
    archive = portable.build_archive(source, tmp_path / 'archive', target_actor_role_sha256='b' * 64)
    return program, parts, source, archive


def load(fixture, **kwargs):
    _, _, source, archive = fixture
    return portable.PortableProgram.load(archive, expected_source_program_sha256=source['sha256'],
        target_actor_role_sha256='b' * 64, **kwargs)


def audit_fixture(tmp_path, fixture, *, mutate=None):
    program, _, source, archive = fixture
    metric = dict(coverage=.8, actionable_coverage=.7, agreement=.99, kl=.01,
        sample_count=1200, rows=960, actionable_rows=840,
        illegal_count=0, whole_tree_raw_illegal_count=0, whole_tree_path_uncertified_count=0)
    value = dict(version=portable.AUDIT_VERSION, phase='final_independent_program_audit',
        independent=True, passed=True, program_archive_sha256=archive['sha256'], source_program_sha256=source['sha256'],
        role=program.role, anchor_actor_role_sha256='a' * 64, target_actor_role_sha256='b' * 64,
        explanation_eligible_leaves=sorted(program.explanation_leaves), sample_count=1200,
        one_use_claim_sha256='c' * 64, audit_partition_sha256='d' * 64,
        metrics={'feedback': deepcopy(metric), 'explanation': deepcopy(metric)}, synthetic_fixture_only=True)
    if mutate:
        mutate(value)
    return write(tmp_path / 'audit.json', value)


def test_original_json_bytes_and_every_frame_distribution_survive_source_removal(fixture):
    program, parts, source, archive = fixture
    manifest = portable.bound(archive)
    assert manifest['anchor_actor_role_sha256'] == 'a' * 64 and manifest['target_actor_role_sha256'] == 'b' * 64
    assert manifest['maximum_depth'] == 12 and manifest['maximum_leaves'] == 256
    assert len(manifest['assets']) == 2 and manifest['parent_dependencies']
    original_bytes = Path(source['path']).read_bytes()
    assert (Path(archive['path']).parent / manifest['assets'][source['sha256']]['path']).read_bytes() == original_bytes
    Path(source['path']).unlink()
    Path(program.protection['parent']['path']).unlink()
    reader = load(fixture)
    for split, data in parts.items():
        expected = program.distributions(data['x'], data['m'])
        actual = reader.distributions(data['x'], data['m'])
        assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True)), split
    assert not hasattr(reader, 'act') and not hasattr(reader, 'decide') and not hasattr(reader, 'step')


def test_all_five_physical_action_masks_match_original(fixture):
    program, parts, *_ = fixture
    reader = load(fixture)
    x = np.repeat(parts['validation']['x'][:1], 16, axis=0)
    x[:, MASK_INDICES] = np.asarray([(*[(i >> bit) & 1 for bit in range(4)], 1) for i in range(16)], np.float32)
    m = x[:, MASK_INDICES]
    expected, actual = program.distributions(x, m), reader.distributions(x, m)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert np.all(actual[2][m == 0] == 0)


def test_archive_can_move_to_another_root_without_changing_program_bytes(fixture, tmp_path):
    program, parts, source, archive = fixture
    relocated = tmp_path / 'different-runtime-root'
    shutil.copytree(Path(archive['path']).parent, relocated)
    shutil.rmtree(Path(archive['path']).parent)
    Path(source['path']).unlink()
    Path(program.protection['parent']['path']).unlink()
    moved = portable.pointer(relocated / 'archive.json')
    assert moved['sha256'] == archive['sha256']
    reader = portable.PortableProgram.load(moved, expected_source_program_sha256=source['sha256'], target_actor_role_sha256='b' * 64)
    data = parts['validation']
    assert all(np.array_equal(a, b) for a, b in zip(program.distributions(data['x'], data['m']),
                    reader.distributions(data['x'], data['m']), strict=True))


@pytest.mark.parametrize('kind', ['missing_parent', 'changed_parent', 'changed_program'])
def test_missing_or_tampered_dependency_fails_closed(fixture, kind):
    program, _, source, archive = fixture
    manifest = portable.bound(archive)
    key = source['sha256'] if kind == 'changed_program' else program.protection['parent']['sha256']
    file = Path(archive['path']).parent / manifest['assets'][key]['path']
    if kind == 'missing_parent':
        file.unlink()
    else:
        file.chmod(0o644); file.write_bytes(file.read_bytes() + b' '); file.chmod(0o444)
    with pytest.raises(RuntimeError, match='dependency'):
        load(fixture)


def test_target_hash_is_current_NN_not_the_original_protection_anchor(fixture):
    _, _, source, archive = fixture
    with pytest.raises(RuntimeError, match='model binding'):
        portable.PortableProgram.load(archive, expected_source_program_sha256=source['sha256'], target_actor_role_sha256='a' * 64)
    with pytest.raises(RuntimeError):
        portable.PortableProgram.load(archive, expected_source_program_sha256='e' * 64, target_actor_role_sha256='b' * 64)


def test_no_independent_audit_means_no_explanation(fixture):
    reader = load(fixture)
    data = fixture[1]['validation']
    assert reader.evidence(data['x'][0], data['m'][0], 'wait', current_actor_role_sha256='b' * 64) is None


def test_independent_audit_and_current_action_must_both_match(fixture, tmp_path):
    audit = audit_fixture(tmp_path, fixture)
    reader = load(fixture, independent_audit_ref=audit, expected_independent_audit_sha256=audit['sha256'])
    data = fixture[1]['validation']; selected = next(i for i, leaf in enumerate(fixture[0].leaf_indices(data['x']))
                                                 if int(leaf) in fixture[0].explanation_leaves)
    row, mask = data['x'][selected], data['m'][selected]
    evidence = reader.evidence(row, mask, 'wait', current_actor_role_sha256='b' * 64)
    assert evidence['eligible'] and evidence['nn_action'] == 'wait' and not evidence['unique_causal_motive_established']
    assert evidence['independent_audit_sha256'] == audit['sha256']
    assert evidence['anchor_actor_role_sha256'] != evidence['target_actor_role_sha256']
    assert reader.evidence(row, mask, 'up', current_actor_role_sha256='b' * 64) is None
    with pytest.raises(RuntimeError, match='role hash'):
        reader.evidence(row, mask, 'wait', current_actor_role_sha256='a' * 64)
    with pytest.raises(RuntimeError, match='trusted expected'):
        load(fixture, independent_audit_ref=audit)
    with pytest.raises(RuntimeError, match='missing'):
        load(fixture, expected_independent_audit_sha256=audit['sha256'])


@pytest.mark.parametrize('fault', ['failed', 'wrong_program', 'wrong_target', 'wrong_archive', 'unqualified_leaf', 'illegal', 'coverage'])
def test_independent_receipt_binding_and_formal_gates_reject(fixture, tmp_path, fault):
    def mutate(value):
        if fault == 'failed': value['passed'] = False
        elif fault == 'wrong_program': value['source_program_sha256'] = 'e' * 64
        elif fault == 'wrong_target': value['target_actor_role_sha256'] = 'e' * 64
        elif fault == 'wrong_archive': value['program_archive_sha256'] = 'e' * 64
        elif fault == 'unqualified_leaf': value['explanation_eligible_leaves'] = [1000]
        elif fault == 'illegal': value['metrics']['feedback']['whole_tree_raw_illegal_count'] = 1
        else: value['metrics']['explanation']['coverage'] = .19
    audit = audit_fixture(tmp_path, fixture, mutate=mutate)
    with pytest.raises(RuntimeError):
        load(fixture, independent_audit_ref=audit, expected_independent_audit_sha256=audit['sha256'])


@pytest.mark.parametrize('fault', ['proof', 'protection', 'capacity', 'leaf_support', 'feature_domain'])
def test_full_native_constraints_remain_enforced_before_packaging(fixture, tmp_path, fault):
    value = portable.bound(fixture[2])
    if fault == 'proof': value['leaves'][1]['certificate']['binding_sha256'] = '0' * 64
    elif fault == 'protection': value['protection']['entries'][0]['leaf_payload']['bias'][4] += .125
    elif fault == 'capacity': value['maximum_depth'] = 13
    elif fault == 'leaf_support': value['leaves'][1]['support'] = list(range(4))
    else: value['domain_sha256'] = '0' * 64
    source = write(tmp_path / 'bad.json', value)
    with pytest.raises((RuntimeError, ValueError)):
        portable.build_archive(source, tmp_path / 'rejected', target_actor_role_sha256='b' * 64)
    assert not (tmp_path / 'rejected').exists()


def test_reader_detects_parent_mutation_after_loading(fixture):
    reader = load(fixture)
    program, parts, _, archive = fixture
    file = Path(archive['path']).parent / portable.bound(archive)['assets'][program.protection['parent']['sha256']]['path']
    file.chmod(0o644); file.write_bytes(file.read_bytes() + b' '); file.chmod(0o444)
    with pytest.raises(RuntimeError, match='hash mismatch'):
        reader.distributions(parts['validation']['x'], parts['validation']['m'])


def test_spent_development_program_portable_forward_matches_all_validation_frames(tmp_path):
    """Old spent arrays only: implementation parity, never independent admission."""
    from experiments.warehouse import v630_protocol as old, v628_data
    source = portable.pointer(portable.ROOT / 'output/warehouse_v642/warehouse-v642-depth9-depth12-01/depth-12-fold-0-robot_2/program.json')
    value = portable.bound(source)
    original = native.Program.from_dict(value, value['target_actor_role_sha256'])
    _, plan = old.checked_plan(); previous = old.bound(plan['previous_plan']); data_plan = old.bound(previous['previous_plan'])
    data = v628_data.load_parts(data_plan, 0, 'robot_2')['validation']
    archive = portable.build_archive(source, tmp_path / 'spent-parity', target_actor_role_sha256=original.target_actor_role_sha256)
    reader = portable.PortableProgram.load(archive, expected_source_program_sha256=source['sha256'],
        target_actor_role_sha256=original.target_actor_role_sha256)
    expected = original.distributions(data['x'], data['m']); actual = reader.distributions(data['x'], data['m'])
    assert len(data['x']) > 1000
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))
