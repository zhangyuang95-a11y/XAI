"""CPU numeric tolerance must not hide changed actions or physics."""
from copy import deepcopy

import pytest

from scripts.check_manual_model_preview_http import compare_replay


def trace():
    return {'replay':{'records':[{'decision':{'action':'left','distributions':{
        'robot_2':{'logits':[75.2226791381836],'probabilities':[0.9947770833969116]}}},
        'state':{'turn':1,'deliveries':0}}]}}


def test_float32_cpu_reduction_difference_is_recorded():
    local=trace();remote=deepcopy(local)
    remote['replay']['records'][0]['decision']['distributions']['robot_2']['logits'][0]=75.22268676757812
    result=compare_replay(remote,local)
    assert result['NN_float_differences']==1
    assert result['exact_actions_bindings_and_physics'] is True


@pytest.mark.parametrize('kind',['action','physical_state','large_nn_error','nan_nn'])
def test_action_state_and_large_numeric_changes_are_rejected(kind):
    local=trace();remote=deepcopy(local);record=remote['replay']['records'][0]
    if kind=='action': record['decision']['action']='right'
    elif kind=='physical_state': record['state']['deliveries']=1
    else: record['decision']['distributions']['robot_2']['logits'][0]=90.0 if kind=='large_nn_error' else float('nan')
    with pytest.raises(AssertionError): compare_replay(remote,local)
