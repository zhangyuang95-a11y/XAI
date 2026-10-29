"""New-study-only request-local deterministic Warehouse replay witnesses.

The original frozen adapter, environment and model code are unchanged. Every
call checks the current request's bound NN decision. Each unique complete
pre-state/binding/human-command input executes the original transition once;
repeated audit calls compare full records and endpoints against that result.
No witness, mutable environment or forward output survives its HTTP request.
Strict stage/export verification always executes the original implementation.
"""
from collections import OrderedDict
from contextlib import nullcontext
import json

from runtime_v642.contracts import digest,require
from runtime_v642.store import encode
from ui.model_preview_fast_runtime import FastWarehouseAdapter
from ui.model_preview_warehouse import REPLAY_VERSION
from env.warehouse.navigation import ACTIONS
from .history import full_verification_enabled

VERSION='study-vnext-request-local-Warehouse-physical-witness.v1'
MAXIMUM_WITNESSES=8
MAXIMUM_BYTES=2*1024*1024


class StudyWarehouseAdapter(FastWarehouseAdapter):
    """Explicit local dependency; never patches an old adapter class/global."""
    def __init__(self,catalog,runtime,*,preloaded_actors=None):
        super().__init__(catalog,runtime)
        if preloaded_actors is not None:
            # These are the already verified read-only frozen NumPy models.
            # Copy the index rather than loading duplicate matrices or sharing
            # a mutable adapter cache with the legacy preview object.
            self._actors=dict(preloaded_actors)

    def _witness_scope(self):
        if full_verification_enabled():return None
        scope=self.runtime.current.get()
        if scope is None:return None
        return scope.setdefault('study_warehouse_physics',dict(
            records=OrderedDict(),bytes=0,computed=0,reused=0))

    @staticmethod
    def _transition_key(state,binding,human_action):
        return digest(dict(version=VERSION,complete_pre_state=state,
            complete_binding=binding.to_dict(),human_action=human_action))

    def _compute_transition(self,state,binding,human_action,decision):
        return super()._advance(state,binding,human_action,decision)

    def _advance(self,state,binding,human_action,decision=None):
        scope=self._witness_scope()
        if scope is None:
            return self._compute_transition(state,binding,human_action,decision)
        self._actor(binding)  # Existing package/binding checks, also on hits.
        # Always invoke decide, including every witness hit. Existing validated
        # request-local NN reuse remains unchanged; no cross-request reuse.
        actual=self.decide(state,binding)
        require(decision is None or actual==decision,
            'decision does not match exact bound pre-action frame')
        require(human_action is None or type(human_action) is str
            and human_action in tuple(a.lower() for a in ACTIONS),'unknown human physical command')
        key=self._transition_key(state,binding,human_action)
        if key in scope['records']:
            scope['records'].move_to_end(key)
            result=json.loads(scope['records'][key])
            require(result[1]['decision']==actual and result[1]['human_action']==human_action
                and result[0]==result[1]['state'],
                'physical witness differs from current bound NN or endpoint')
            scope['reused']+=1
            return result[0],result[1]
        nxt,record=self._compute_transition(state,binding,human_action,decision)
        require(record['decision']==actual and record['human_action']==human_action
            and record['state']==nxt,'fresh physical transition differs from current NN or endpoint')
        scope['computed']+=1
        payload=encode((nxt,record)).encode('utf-8')
        if len(payload)<=MAXIMUM_BYTES:
            scope['records'][key]=payload;scope['bytes']+=len(payload)
            while len(scope['records'])>MAXIMUM_WITNESSES or scope['bytes']>MAXIMUM_BYTES:
                _,old=scope['records'].popitem(last=False);scope['bytes']-=len(old)
        return nxt,record

    def replay(self,bundle,binding):
        scope=self._witness_scope()
        if scope is not None:
            require(type(bundle) is dict and bundle.get('version')==REPLAY_VERSION
                and bundle.get('binding')==binding.to_dict() and type(bundle.get('records')) is list
                and type(bundle.get('two_frozen_NN')) is bool,'unbound or malformed replay')
            previous=bundle['initial_state']
            for record in bundle['records']:
                require(type(record) is dict and (not bundle['two_frozen_NN'] or record.get('human_action') is None),
                    'NN-NN replay includes an unrecorded human takeover')
                key=self._transition_key(previous,binding,record.get('human_action'))
                if key in scope['records']:
                    verified=json.loads(scope['records'][key])
                    require(record==verified[1] and record.get('state')==verified[0],
                        'record differs from verified deterministic physical transition')
                previous=record.get('state')
            require(previous==bundle['final_state'],'replay final endpoint differs from its records')
        # All original record/NN/binding/RNG/endpoint comparisons still execute.
        return super().replay(bundle,binding)
