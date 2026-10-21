"""Current-NN loss evidence for native v630 assets, never execution actions.

The original target replay, scene isolation, qualification and strict gates
are reused unchanged. A separate version prevents an old reader silently
accepting the refined program language. No owner/collection authority exists.
"""
from copy import deepcopy
from dataclasses import dataclass

from . import warehouse_v629_online_evidence as legacy
from .warehouse_v630_program import Program,VERSION as PROGRAM_VERSION
from experiments.warehouse.v626_protocol import require,pointer,bound,digest

VERSION='warehouse-v630-online-current-NN-evidence.v1'


def source_files():
    from experiments.warehouse import v630_training_admission as admission
    from experiments.warehouse.v626_protocol import file_sha
    files=admission.source_files()
    for name in ('backend/training/warehouse_v630_online_evidence.py',
                 'tests/test_warehouse_v630_online_evidence.py',
                 'tests/test_warehouse_v629_training_feedback.py',
                 'tests/test_warehouse_v629_online_evidence.py',
                 'tests/test_warehouse_v622_actor_training.py',
                 'docs/warehouse_v630_online_feedback_bridge.md'):
        files[name]=file_sha(admission.v.ROOT/name)
    return dict(sorted(files.items()))


@dataclass(frozen=True)
class OnlineEvidence(legacy.OnlineEvidence):
    def payload(self):
        value=super().payload()
        value.update(version=VERSION,program_asset_version=PROGRAM_VERSION,
                     immutable_protected_parent=deepcopy(self.program.protection['parent']))
        return value

    def verify(self,*,current_role_sha256,epoch_id):
        require(isinstance(self.program,Program),'v630 loss evidence requires its native program reader')
        return super().verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)


def build_evidence(program_ref,target_actor_ref,pool_ref,*,epoch_id):
    require(isinstance(epoch_id,str) and epoch_id and len(epoch_id)<=128,'explicit optimizer epoch ID required')
    require(pointer(target_actor_ref['path'])==target_actor_ref,'current target Actor pointer differs')
    encoded=bound(program_ref)
    program=Program.from_dict(encoded,encoded['actor_role_sha256'])
    actor=legacy.FrozenWarehouseV6Actor.load(target_actor_ref['path'],expected_sha256=target_actor_ref['sha256'])
    require(actor.model_version==legacy.MODEL_VERSION,'online targets must retain the complete original Actor architecture')
    parts,refs=legacy.load_pool(pool_ref,target_actor_ref,actor,program.role)
    report=legacy.current_report(program,parts)
    refs=[program_ref,target_actor_ref,program.protection['parent'],*refs]
    out=OnlineEvidence(program,report,deepcopy(program_ref),deepcopy(target_actor_ref),actor.role_sha256[program.role],
                       deepcopy(pool_ref),epoch_id,tuple(deepcopy(refs)),'')
    out=OnlineEvidence(**{**out.__dict__,'envelope_digest':digest(out.payload())})
    return out.verify(current_role_sha256=out.target_role_sha256,epoch_id=epoch_id)


def load_evidence(ref,*,current_role_sha256,epoch_id):
    encoded=bound(ref)
    require(encoded.get('version')==VERSION and encoded.get('program_asset_version')==PROGRAM_VERSION,
            'unknown v630 online evidence schema or program language')
    out=build_evidence(encoded['anchor_program'],encoded['current_target_actor'],encoded['pool'],epoch_id=encoded['epoch_id'])
    require(encoded==out.payload(),'stored v630 loss evidence does not independently replay')
    return out.verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)
