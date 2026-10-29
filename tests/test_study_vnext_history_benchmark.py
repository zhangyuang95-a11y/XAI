"""Benchmark statistic/trace contracts only; never starts real-model runs."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('history_benchmark_contract',
    Path(__file__).parents[1]/'scripts/benchmark_study_vnext_history.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


def test_fixed_count_and_explicit_interpolated_percentiles():
    values=list(range(1,31));summary=b.measurements(values)
    assert summary['N']==30 and summary['P50']==15.5 and summary['P95']==pytest.approx(28.55)
    assert summary['mean']==15.5
    with pytest.raises(RuntimeError,match='exactly 30'):b.measurements(values[:-1])
    with pytest.raises(RuntimeError):b.measurements([float('nan')]*30)


@pytest.mark.parametrize('field',['complete_state_sha256','executed_AI_action','score','binding_sha256','public_response_sha256'])
def test_every_action_state_score_binding_difference_rejects_pair(field):
    first=[dict(complete_state_sha256='a',executed_AI_action='wait',score={'native':1},
        binding_sha256='b',public_response_sha256='c') for _ in range(40)]
    second=[dict(row) for row in first];second[23][field]='changed'
    with pytest.raises(RuntimeError,match='mismatch at action 23'):b.compare_traces(first,second)
    assert b.compare_traces(first,first)['compared_actions']==40
    with pytest.raises(RuntimeError,match='all 40'):b.compare_traces(first,first[:-1])


def test_exact_six_variants_and_fixed_scientific_free_benchmark_counts():
    assert len(b.MODELS)==6 and set(b.MODELS)=={(d,v) for d in ('warehouse','pong','kitchen') for v in ('raw','regularized')}
    assert b.WARMUP_ACTIONS==10 and b.SAMPLE_ACTIONS==30
