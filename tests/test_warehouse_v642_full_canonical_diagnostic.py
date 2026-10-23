"""Synthetic numerical tests; no environment or actual research seed use."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import numpy as np
import pytest

from experiments.warehouse import v642_full_canonical_diagnostic as full


@pytest.fixture
def numeric():
    spec = importlib.util.spec_from_file_location('full_old_numerical_fixture',
        Path(__file__).with_name('test_warehouse_v642_canonical_batch.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    role, x = helper.numerical.__wrapped__()
    return role, x[:7]


def provenance(p, role):
    # Unit-test numeric controls only, not a generated experiment inventory.
    rows = [dict(action_seed=0, episode_id=1, frame=i) for i in range(len(p))]
    sampled = np.array([np.random.default_rng(full.independent_agent_seed(base_seed=r['action_seed'],
        episode_id=r['episode_id'], frame=r['frame'], agent_id=role)).choice(5, p=row)
        for r, row in zip(rows, p, strict=True)], dtype=np.int64)
    return rows, sampled


def test_two_chunk_sizes_all_layers_and_float_hex_signzero_draws_share_scalar(numeric):
    role, x = numeric
    p = full.numerical.canonical_probabilities(role, x)
    rows, sampled = provenance(p, 'robot_1')
    observer = full.ProbabilityObserver('robot_1', rows, p, sampled)
    scalar = full.ScalarOutputs(len(rows), sampled)
    result = full.numerical.verify_all_layers(role, x, stored=p, chunk_size=5,
        candidate_chunk_sizes=full.CHUNKS, probability_observer=observer, scalar_output_observer=scalar)
    assert set(result['candidates']) == {f'{full.METHOD}-chunk-{n}' for n in full.CHUNKS}
    assert all(a['all_layers_bitwise_equal'] for a in result['candidates'].values())
    assert observer.result()['all_probability_bits_hex_signzero_and_recorded_draws_equal']
    assert observer.result()['new_action_seed_allocation'] == 0
    assert full.numerical.bits_equal(scalar.probabilities, p)
    assert full.numerical.bits_equal(scalar.logits, np.stack([role.logits_encoded(row) for row in x]))


def test_observer_rejects_signed_zero_without_tolerance(numeric):
    role, x = numeric
    p = full.numerical.canonical_probabilities(role, x)
    rows, sampled = provenance(p, 'robot_1')
    bad = p.copy()
    i, j = np.argwhere(p == 0)[0]
    bad[i, j] = np.float32(-0.)
    candidates = {f'{full.METHOD}-chunk-{n}': bad for n in full.CHUNKS}
    observer = full.ProbabilityObserver('robot_1', rows, p, sampled)
    with pytest.raises(RuntimeError, match='bytes/sign-zero'):
        observer(p, candidates, 0)


def test_observer_records_actual_sampling_counterexample(numeric):
    role, x = numeric
    p = full.numerical.canonical_probabilities(role, x)
    rows, sampled = provenance(p, 'robot_2')
    sampled[0] = (sampled[0]+1)%5
    candidates = {f'{full.METHOD}-chunk-{n}': p for n in full.CHUNKS}
    observer = full.ProbabilityObserver('robot_2', rows, p, sampled)
    with pytest.raises(full.NumericalMismatch) as caught:
        observer(p, candidates, 0)
    assert caught.value.evidence['kind'] == 'recorded_sampling'
    assert caught.value.evidence['row_offset'] == 0
    assert len(caught.value.evidence['probability_float_hex']) == 5


def test_observer_rejects_skipped_rows_and_incomplete_coverage(numeric):
    role, x = numeric
    p = full.numerical.canonical_probabilities(role, x)
    rows, sampled = provenance(p, 'robot_1')
    observer = full.ProbabilityObserver('robot_1', rows, p, sampled)
    with pytest.raises(RuntimeError, match='skipped'):
        observer(p, {f'{full.METHOD}-chunk-{n}': p for n in full.CHUNKS}, 1)
    with pytest.raises(RuntimeError, match='not all'):
        observer.result()


def test_counterexample_keeps_exact_hex_and_signed_zero(numeric, tmp_path):
    role, x = numeric
    p = full.numerical.canonical_probabilities(role, x)
    rows, sampled = provenance(p, 'robot_1')
    i, j = np.argwhere(p == 0)[0]
    p[i, j] = np.float32(-0.)
    ref = full._counterexample(role, x, p, rows, tmp_path)
    value = full.bound(ref)
    assert value['layer'] == 'stored_probability'
    assert value['expected_float_hex'] != value['candidate_float_hex']
    assert value['expected_signbit'] != value['candidate_signbit']
    assert value['numerical_candidate_rejected']


def test_empty_wrong_call_inventory_cannot_authorize_a_run():
    with pytest.raises(RuntimeError, match='three explicit old'):
        full.inventory([])


def test_frozen_source_contract_contains_only_old_data_numerical_reader():
    sources = full.source_files()
    assert sources['experiments/warehouse/v642_full_canonical_diagnostic.py'] == full.file_sha(full.__file__)
    assert 'backend/training/warehouse_v6_actor.py' in sources
