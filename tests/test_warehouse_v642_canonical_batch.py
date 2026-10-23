"""Pure numerical controls: no environment/reset/Actor training constructors."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from backend.training import warehouse_v642_canonical_batch as d
from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout


@pytest.fixture
def numerical():
    # These numbers seed only synthetic tensor values, never environments.
    rng = np.random.default_rng(642)
    f, h, g = d.runtime.FEATURE_DIM, d.runtime.RESIDUAL_WIDTH, d.runtime.GATE_WIDTH
    relation = f+d.runtime.GEOMETRY_DIM+5+d.runtime.ACTION_FEATURE_DIM
    shapes = {'tree.split.weight': (15, f), 'tree.split.bias': (15,),
        'tree.leaf_logits': (16, 5), 'tree.left_refs': (15,), 'tree.right_refs': (15,),
        'tree.node_count': (1,), 'tree.leaf_count': (1,), 'gate.0.weight': (g, f+4),
        'gate.0.bias': (g,), 'gate.2.weight': (1, g), 'gate.2.bias': (1,),
        'action_residual.0.weight': (h, relation), 'action_residual.0.bias': (h,),
        'action_residual.2.weight': (h, h), 'action_residual.2.bias': (h,),
        'action_residual.4.weight': (1, h), 'action_residual.4.bias': (1,)}
    weights = {key: (rng.standard_normal(shape)*.1).astype(np.float32) for key, shape in shapes.items()}
    for key in ('tree.split.weight', 'tree.split.bias', 'tree.left_refs', 'tree.right_refs', 'tree.node_count'):
        weights[key].fill(0)
    weights['tree.leaf_count'].fill(1)
    role = d.runtime.FrozenWarehouseV6Role(weights)
    names = {name: i for i, name in enumerate(OBSERVATION_FEATURE_NAMES)}
    points = list(layout.passable_positions)
    obs, masks = [], []
    for own in points:
        row = np.zeros(len(names), np.float32)
        for stem, position in [('self', own), ('partner', points[-1]),
                ('charger', layout.charger_position), ('order_0_pickup', points[0]),
                ('order_0_delivery', points[-1]), ('order_1_pickup', points[1]),
                ('order_1_delivery', points[-2])]:
            row[names[stem+'_row']] = position[0]/(layout.rows-1)
            row[names[stem+'_column']] = position[1]/(layout.cols-1)
        row[names['self_active']] = 1.
        row[names['self_battery_fraction']] = .8
        for r, line in enumerate(layout.tiles):
            for c, tile in enumerate(line):
                row[names[f'wall_row_{r}_column_{c}']] = tile == '#'
        masks.append([float(a == 'WAIT' or (own[0]+MOVE_DELTAS[a][0],
            own[1]+MOVE_DELTAS[a][1]) in points) for a in ACTIONS])
        obs.append(row)
    x = d.runtime.encode_actor_tree_features(np.asarray(obs), np.asarray(masks, np.float32))
    return role, x


def test_bitwise_comparison_does_not_accept_signed_zero_or_ulp():
    assert not d.bits_equal(np.array([0.], np.float32), np.array([-0.], np.float32))
    assert not d.bits_equal(np.array([1.], np.float32), np.nextafter(np.array([1.], np.float32), np.float32(2)))
    assert not d.bits_equal(np.array([1.], np.float32), np.array([1.], np.float64))


def test_original_transcription_and_shape_preserving_candidates_match_every_layer(numerical):
    role, x = numerical
    out = d.compare(role, x)
    for name in ('frame_matrices', 'frame_matrices_scalar_ufunc'):
        assert out['candidates'][name]['all_layers_bitwise_equal']
    # W-left changes the original residual inner matrix from five rows to one.
    assert not out['candidates']['w_left_vectors']['all_layers_bitwise_equal']
    assert out['candidates']['w_left_vectors']['layers']['residual_affine_0']['differing_elements']


@pytest.mark.parametrize('chunk', [1, 2, 7, 19, 256])
def test_exact_probability_chunks_and_complete_layer_streaming(numerical, chunk):
    role, x = numerical
    original = d.canonical_probabilities(role, x)
    assert d.bits_equal(original, d.candidate_probabilities(role, x, chunk_size=chunk))
    value = d.verify_all_layers(role, x, stored=original, chunk_size=chunk)
    assert value['stored_canonical_targets_bitwise_verified']
    assert value['candidates']['frame_matrices_scalar_ufunc']['all_layers_bitwise_equal']
    bad = original.copy()
    bad[0, 4] = np.nextafter(bad[0, 4], np.float32(2))
    with pytest.raises(RuntimeError, match='cache differs'):
        d.verify_all_layers(role, x, stored=bad, chunk_size=chunk)


def test_numeric_extreme_probes_rebuild_derived_features_without_environment(numerical):
    role, x = numerical
    p = d.numerical_probes(x, count=65)
    assert d.bits_equal(p, d.numerical_probes(x, count=65))
    assert not d.bits_equal(p, d.numerical_probes(x, count=65, control=643))
    d.validate_rows(p)
    assert np.isin(p[:, d.MASK_INDICES], (0, 1)).all()
    d.runtime.action_features_from_tree_encoded(p)
    assert d.compare(role, p)['candidates']['frame_matrices_scalar_ufunc']['all_layers_bitwise_equal']


@pytest.mark.parametrize('kind', ['mask', 'context', 'nonfinite', 'batch_original', 'version', 'method'])
def test_contract_and_version_rejections(numerical, kind):
    role, x = numerical
    x = x.copy()
    method = 'frame_matrices_scalar_ufunc'
    if kind == 'mask':
        x[0, d.MASK_INDICES[4]] = 0
    elif kind == 'context':
        x[0, -1] = 1-x[0, -1]
    elif kind == 'nonfinite':
        x[0, 0] = np.nan
    elif kind == 'batch_original':
        method = 'original_shape'
    elif kind == 'version':
        role.model_version = 'unregistered'
    else:
        method = 'unregistered'
    with pytest.raises((ValueError, RuntimeError)):
        d.forward_trace(role, x, method)


def test_partial_call_without_native_event_cannot_be_profiled(tmp_path):
    source = tmp_path/'result.json'
    source.write_text('{}')
    missing = dict(path=str(tmp_path/'absent-event.json'), sha256='0'*64)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        d.load_old_target(d.pointer(source), missing)


def test_probability_benchmark_is_numerical_only_and_rejects_wrong_cache(numerical):
    role, x = numerical
    out = d.benchmark_probabilities(role, x, stored=d.canonical_probabilities(role, x), chunk_sizes=(1, 7))
    assert out['all_chunks_bitwise_equal']
    assert out['stored_target_bitwise_verified']
    with pytest.raises(RuntimeError, match='cache differs'):
        d.benchmark_probabilities(role, x, stored=np.zeros((len(x), 5), np.float32), chunk_sizes=(7,))


def test_source_contract_is_explicit_and_no_original_module_is_written():
    sources = d.source_files()
    assert sources['backend/training/warehouse_v642_canonical_batch.py'] == d.file_sha(d.__file__)
    assert 'backend/training/warehouse_v6_actor.py' in sources
