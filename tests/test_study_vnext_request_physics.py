"""Synthetic deterministic transitions; no model training or environment reset."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from runtime_v642.contracts import digest,require,RuntimeUnavailable
from study_vnext.history import full_history_verification
from study_vnext.request_physics import StudyWarehouseAdapter,REPLAY_VERSION,MAXIMUM_BYTES,MAXIMUM_WITNESSES


class Runtime:
    def __init__(self):self.current=ContextVar('synthetic_physics_request',default=None)
    @contextmanager
    def request(self):
        scope={};token=self.current.set(scope)
        try:yield scope
        finally:self.current.reset(token)


class ToyAdapter(StudyWarehouseAdapter):
    def __init__(self,binding):
        self.expected=binding;self.runtime=Runtime();self.physical_calls=0;self.decide_calls=0;self.fail=False
    def _actor(self,binding):require(binding==self.expected,'synthetic binding changed')
    def restore(self,state,binding):
        self._actor(binding)
        require(digest({k:v for k,v in state.items() if k!='state_sha256'})==state['state_sha256'],
            'synthetic private state or RNG changed')
        return deepcopy(state)
    def decide(self,state,binding):
        self.restore(state,binding);self.decide_calls+=1
        return dict(action='wait',policy_actions={'robot_1':'WAIT','robot_2':'WAIT'},state_sha256=state['state_sha256'])
    def _compute_transition(self,state,binding,human_action,decision):
        self.physical_calls+=1;actual=self.decide(state,binding)
        require(decision is None or decision==actual,'synthetic supplied decision changed')
        if self.fail:raise RuntimeError('synthetic physical failure')
        nxt=deepcopy(state);nxt['turn']+=1;nxt['rng']+=1
        nxt['state_sha256']=digest({k:v for k,v in nxt.items() if k!='state_sha256'})
        return nxt,dict(human_action=human_action,decision=actual,state=nxt,rewards={'team':1})


@pytest.fixture
def prepared(tmp_path):
    spec=importlib.util.spec_from_file_location('physics_toy_catalog',Path(__file__).with_name('test_runtime_v642_store.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    _,_,catalog=helper.package_at(tmp_path/'package');binding=catalog.binding('warehouse','regularized')
    adapter=ToyAdapter(binding);state=dict(turn=0,rng=123)
    state['state_sha256']=digest(state)
    return adapter,binding,state


def bundle(binding,state,result):
    nxt,record=result
    return dict(version=REPLAY_VERSION,binding=binding.to_dict(),initial_state=deepcopy(state),
        records=[deepcopy(record)],final_state=deepcopy(nxt),two_frozen_NN=False)


def test_same_request_deduplicates_only_identical_physics_with_fresh_decide_every_call(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request() as scope:
        original=adapter._advance(state,binding,'wait');decisions=adapter.decide_calls
        again=adapter._advance(state,binding,'wait',original[1]['decision'])
        assert again==original and adapter.physical_calls==1
        assert adapter.decide_calls==decisions+1
        proof=scope['study_warehouse_physics']
        assert proof['computed']==proof['reused']==1
        assert 'memo' not in scope  # Does not evict/change existing NN memo.
        assert adapter.replay(bundle(binding,state,original),binding)==original[0]
        assert adapter.physical_calls==1
    with adapter.runtime.request():adapter._advance(state,binding,'wait')
    assert adapter.physical_calls==2  # No cross-request physical/NN reuse.


def test_mutable_return_cannot_poison_immutable_request_witness(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request():
        expected=adapter._advance(state,binding,'wait');changed=adapter._advance(state,binding,'wait')
        changed[0]['rng']=999;changed[1]['decision']['action']='forged'
        assert adapter._advance(state,binding,'wait')==expected


def test_None_and_wait_are_distinct_inputs_and_untrusted_decisions_are_always_checked(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request():
        actual=adapter._advance(state,binding,'wait')
        adapter._advance(state,binding,None)
        assert adapter.physical_calls==2
        bad=deepcopy(actual[1]['decision']);bad['action']='up'
        with pytest.raises(RuntimeUnavailable,match='decision does not match'):
            adapter._advance(state,binding,'wait',bad)
        assert adapter.physical_calls==2


@pytest.mark.parametrize('fault',['decision','record_state','reward','final_state','initial_rng'])
def test_tampered_record_endpoint_or_private_RNG_is_rejected(prepared,fault):
    adapter,binding,state=prepared
    with adapter.runtime.request():
        current=bundle(binding,state,adapter._advance(state,binding,'wait'))
        if fault=='decision':current['records'][0]['decision']['action']='up'
        elif fault=='record_state':current['records'][0]['state']['rng']=999
        elif fault=='reward':current['records'][0]['rewards']['team']=999
        elif fault=='final_state':current['final_state']['rng']=999
        else:current['initial_state']['rng']=999
        with pytest.raises(RuntimeUnavailable):adapter.replay(current,binding)


def test_changed_current_NN_result_cannot_be_masked_by_physical_witness(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request():
        adapter._advance(state,binding,'wait');original=adapter.decide
        def changed(*args):
            result=original(*args);result['action']='up';return result
        adapter.decide=changed
        with pytest.raises(RuntimeUnavailable,match='differs from current bound NN'):
            adapter._advance(state,binding,'wait')


def test_full_state_binding_changes_and_failed_computations_never_reuse_witness(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request() as scope:
        adapter._advance(state,binding,'wait')
        changed=deepcopy(state);changed['rng']=456
        with pytest.raises(RuntimeUnavailable,match='state or RNG changed'):
            adapter._advance(changed,binding,'wait')
        other=replace(binding,accepted_receipt_sha256='f'*64)
        with pytest.raises(RuntimeUnavailable,match='binding changed'):adapter._advance(state,other,'wait')
        next_state=deepcopy(state);next_state['turn']=10
        next_state['state_sha256']=digest({k:v for k,v in next_state.items() if k!='state_sha256'})
        before=len(scope['study_warehouse_physics']['records']);adapter.fail=True
        with pytest.raises(RuntimeError,match='physical failure'):adapter._advance(next_state,binding,'wait')
        assert len(scope['study_warehouse_physics']['records'])==before


def test_strict_audit_and_missing_request_execute_original_physics_every_time(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request() as scope,full_history_verification():
        adapter._advance(state,binding,'wait');adapter._advance(state,binding,'wait')
        assert 'study_warehouse_physics' not in scope
    adapter._advance(state,binding,'wait');adapter._advance(state,binding,'wait')
    assert adapter.physical_calls==4


def test_request_cache_has_entry_and_byte_limits(prepared):
    adapter,binding,state=prepared
    with adapter.runtime.request() as scope:
        for turn in range(12):
            current=dict(turn=turn,rng=turn);current['state_sha256']=digest(current)
            adapter._advance(current,binding,'wait')
        proof=scope['study_warehouse_physics']
        assert len(proof['records'])<=MAXIMUM_WITNESSES and proof['bytes']<=MAXIMUM_BYTES
        oversized=dict(state,padding='x'*(MAXIMUM_BYTES+1))
        oversized['state_sha256']=digest({k:v for k,v in oversized.items() if k!='state_sha256'})
        before=adapter.physical_calls
        adapter._advance(oversized,binding,'wait');adapter._advance(oversized,binding,'wait')
        assert adapter.physical_calls==before+2
        assert proof['bytes']<=MAXIMUM_BYTES
