"""Formal training authority from an actually replayed three-stream budget lock.

This grants only preparation of fresh finite formal inventories and training.
No JSON passed flag, direct constructor, restart, chosen budget, model success,
program audit, holdout or release permission substitutes for the real reader.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from . import v642_long_training_protocol as protocol
from .v626_protocol import require,pointer,digest,file_sha

ROOT=Path(__file__).resolve().parents[2]
VERSION='warehouse-v642-budget-locked-formal-training-admission.v1'
PROPOSAL_VERSION='warehouse-v642-budget-locked-formal-seed-proposal.v1'
_TOKEN=object()
_CACHE={}
FILES=('experiments/warehouse/v642_formal_training_admission.py',
       'tests/test_warehouse_v642_formal_training_admission.py','docs/warehouse_v642_formal_training_admission.md')
DEVELOPMENT_FILES=('experiments/warehouse/v642_development_admission.py',
       'tests/test_warehouse_v642_development_admission.py','docs/warehouse_v642_development_admission.md')


def source_files():
    # Flat hashes are intentional: development.source_files -> quality ->
    # owner -> seed-plan -> this module must never recurse into itself.
    return {name:file_sha(ROOT/name) for name in (*FILES,*DEVELOPMENT_FILES)}


@dataclass(frozen=True,init=False)
class FormalTrainingAdmission:
    value:dict
    refs:tuple
    payload_sha256:str
    _token:object
    _budget_lock:object

    def __init__(self,*args,**kwargs):
        raise RuntimeError('read the actual three-stream development budget lock before formal inputs')

    @property
    def budget_lock_ref(self):return deepcopy(self._budget_lock.ref)

    @property
    def budget(self):return self.value['joint_step_budget']

    def verify(self):
        from . import v642_development_admission as development
        require(type(self._budget_lock) is development.BudgetLock and self._token is _TOKEN,
                'formal authorization requires a genuine independently read BudgetLock token')
        self._budget_lock.verify()  # All source files and every immutable input SHA, every time.
        require(self.value==_binding(self._budget_lock) and self.payload_sha256==digest(self.value)
            and self.refs==self._budget_lock.refs,'formal admission source, lock, budget or scope changed')
        return self

    def validate_proposal(self,p):
        self.verify()
        keys={'version','phase','stream_id','joint_step_budget','initialization_seed','optimizer_seed',
              'train_case_inventory','program_evidence_inventory','development_budget_lock'}
        require(type(p) is dict and set(p)==keys and p['version']==PROPOSAL_VERSION
            and p['phase']=='formal' and p['stream_id'] in protocol.FORMAL_STREAMS
            and type(p['joint_step_budget']) is int and p['joint_step_budget']==self.budget
            and p['development_budget_lock']==self.budget_lock_ref,
            'formal proposal must bind the exact real budget lock, fixed budget/stream, and no restart')
        return self


def _binding(lock):
    return dict(version=VERSION,development_budget_lock=deepcopy(lock.ref),
        development_lock_payload_sha256=lock.payload_sha256,joint_step_budget=lock.value['selected_joint_step_budget'],
        formal_streams=list(protocol.FORMAL_STREAMS),source_files=source_files(),source_sha256=digest(source_files()),
        actual_three_development_streams_independently_replayed=True,
        formal_inventory_must_be_fresh=True,formal_restart_authorized=False,
        finite_training_preparation_authorized=True,training_only=True,
        quality_selection_authorized=False,sealed_authority=False,model_acceptance_passed=False,pilot_opened=False)


def read_admission(budget_lock_ref):
    """FIRST call of every explicit formal entry, before other input reads.

    Cache only this returned token, never a parsed passed dictionary. Its
    verify method keeps complete source and all original refs SHA checks;
    the full independent development replay is performed by this first read.
    """
    require(type(budget_lock_ref) is dict and set(budget_lock_ref)=={'path','sha256'},
            'formal entry requires an explicit real development budget-lock pointer before other inputs')
    cache_key=digest(budget_lock_ref)
    if cache_key in _CACHE:
        token=_CACHE[cache_key]
        require(type(token) is FormalTrainingAdmission and token.budget_lock_ref==budget_lock_ref,
                'formal admission cache is not the genuine previously read token')
        return token.verify()
    from . import v642_development_admission as development
    lock=development.read_budget_lock(budget_lock_ref)
    require(type(lock) is development.BudgetLock,'formal entry accepts only the real BudgetLock reader token')
    lock.verify()
    token=object.__new__(FormalTrainingAdmission);value=_binding(lock)
    for field_name,item in dict(value=value,refs=lock.refs,payload_sha256=digest(value),_token=_TOKEN,_budget_lock=lock).items():
        object.__setattr__(token,field_name,item)
    token.verify();_CACHE[cache_key]=token
    return token
