"""Durable finite v6.43 ownership primitives; scientific collector not enabled.

These components compile exact owner/claim identities and journal predebits
before an external initializer/reset may occur. They do not perform those
operations or authenticate physical traces by themselves. ``claimed_owner``
intentionally refuses before input I/O until the complete collector, current
program provider and recovery integration have undergone safety review.
No candidate claim is created by import, descriptor preparation or a ledger.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os

from . import v643_training_seed_plan as seeds
from . import v643_training_partitions as partitions
from .v643_risk_recipe import CANDIDATE, REFERENCE_SHA256
from .v626_protocol import require, pointer, bound, digest

ROOT = partitions.ROOT
OWNER_ROOT = ROOT/'output/warehouse_v643'/CANDIDATE
VERSION = 'warehouse-v643-finite-calibration-owner-components.v1'
OWNER_VERSION = 'warehouse-v643-finite-calibration-variant-owner.v1'
CLAIM_VERSION = 'warehouse-v643-finite-calibration-single-use-claim.v1'
DESCRIPTOR_VERSION = 'warehouse-v643-finite-calibration-owner-descriptor.v1'
JOURNAL_VERSION = 'warehouse-v643-finite-owner-journal.v1'


def source_files():
    return seeds.source_files()


def claim_path(plan_ref=None):
    # A different run ID/plan can never create a second attempt for this candidate.
    return OWNER_ROOT/'single-use-20k.claim.json'


def claimed_owner(*_args, **_kwargs):
    # Hard stop before plan reads, directories, claims, RNGs or environments.
    raise RuntimeError('v643 finite collector/recovery integration is not enabled; no claim or training')


def _exclusive_json(path, value):
    """Durable exclusive data writer; this helper is not a claim authority."""
    path=Path(path)
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    try:
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    except BaseException:
        # Preserve even partial files. An interrupted single-use write must
        # never turn into an implicit retry by deleting its reservation.
        raise
    return pointer(path)


def descriptor_document(plan):
    plan.verify();p=plan.partition.value
    require(p['reference_actor']['sha256']==REFERENCE_SHA256,'new owner must start at original complete v620 BC')
    require(plan.value['source_files']==source_files(),'owner implementation differs from frozen plan')
    return dict(schema=DESCRIPTOR_VERSION,candidate_id=CANDIDATE,plan=deepcopy(plan.ref),
        partition=deepcopy(plan.partition.ref),source_files=source_files(),source_sha256=digest(source_files()),
        reference_actor=deepcopy(p['reference_actor']),reference_buffer=deepcopy(p['reference_buffer']),
        program_templates=deepcopy(p['program_templates']),
        **{key:deepcopy(p[key]) for key in ('train_case_inventory','program_evidence_inventory','quality_case_inventory')},
        config=deepcopy(p['config']),variants=list(seeds.VARIANTS),per_variant_joint_step_limit=20000,
        zero_update_original_BC_start_required=True,old_20k_pass_required=False,
        collector_implementation='warehouse-v643-finite-calibration-pipeline.v1',
        public_admission_contract='original_BC_finite_inventories_exact_history_single_use_candidate_claim',
        training_authority=False,sealed_authority=False,pilot_opened=False)


def prepare_descriptor(plan):
    """Freeze reviewable source/data binding; does not burn a candidate claim."""
    value=descriptor_document(plan)
    folder=Path(plan.ref['path']).parent/'calibration'
    folder.mkdir(exist_ok=False)
    return _exclusive_json(folder/'descriptor.json',value)


def compile_claim_record(plan, descriptor_ref):
    """Read-only record compiler; only a future reviewed owner may write it."""
    plan.verify()
    require(Path(descriptor_ref['path'])==Path(plan.ref['path']).parent/'calibration/descriptor.json'
        and partitions._immutable(descriptor_ref)==descriptor_document(plan),'new source-bound descriptor differs')
    clearance=seeds.checked_clearance(plan,execution=True)
    require(not claim_path().exists(),'v643 candidate already consumed its only calibration attempt')
    return dict(schema=CLAIM_VERSION,candidate_id=CANDIDATE,plan=deepcopy(plan.ref),
        partition=deepcopy(plan.partition.ref),descriptor=deepcopy(descriptor_ref),
        source_sha256=plan.value['source_sha256'],clearance=deepcopy(clearance['clearance']),
        single_use_even_if_interrupted=True,automatic_resume=False,training_authority=False,pilot_opened=False)


def variant_owner_document(plan, descriptor_ref, claim_ref, variant):
    """Pure identity compiler; requires the exact global claim to exist."""
    require(variant in seeds.VARIANTS,'unknown paired variant')
    expected=compile_claim_record_without_unused_check(plan,descriptor_ref)
    require(Path(claim_ref['path'])==claim_path() and partitions._immutable(claim_ref)==expected,
            'old/forged/relocated candidate claim cannot bind a variant owner')
    p=plan.partition.value
    return dict(schema=OWNER_VERSION,candidate_id=CANDIDATE,variant=variant,claim=deepcopy(claim_ref),
        plan=deepcopy(plan.ref),descriptor=deepcopy(descriptor_ref),source_files=source_files(),
        source_sha256=plan.value['source_sha256'],pair_seed=p['initialization_seed'],
        program_templates=deepcopy(p['program_templates']),
        **{key:deepcopy(p[key]) for key in ('train_case_inventory','program_evidence_inventory','quality_case_inventory')})


def compile_claim_record_without_unused_check(plan, descriptor_ref):
    # Kept distinct so reading an existing claim never temporarily hides or
    # renames it merely to pass compile_claim_record's unused-attempt check.
    plan.verify()
    require(Path(descriptor_ref['path'])==Path(plan.ref['path']).parent/'calibration/descriptor.json'
        and partitions._immutable(descriptor_ref)==descriptor_document(plan),'descriptor differs')
    # Once the exclusive claim exists, its own metadata is a legitimate new
    # execution record. The full stat delta was checked immediately before it.
    clearance=seeds.checked_clearance(plan,execution=False)
    from datetime import datetime, timezone
    require(datetime.now(timezone.utc)<datetime.fromisoformat(clearance['value']['expires_utc']),
            'claimed finite owner history clearance expired')
    return dict(schema=CLAIM_VERSION,candidate_id=CANDIDATE,plan=deepcopy(plan.ref),
        partition=deepcopy(plan.partition.ref),descriptor=deepcopy(descriptor_ref),
        source_sha256=plan.value['source_sha256'],clearance=deepcopy(clearance['clearance']),
        single_use_even_if_interrupted=True,automatic_resume=False,training_authority=False,pilot_opened=False)


class ResetLedger:
    """Finite reset coordination, not permission to construct an environment.

    The future collector must use the actual reset hook and physics replayer;
    accepting a matching supplied row here does not prove the row's origin.
    Unbound predebits are deliberately not safe checkpoints or retry authority.
    """
    def __init__(self, plan, journal_path):
        plan.verify()
        self.plan=plan;self.failed=False;self.closed=False
        self.training=bound(plan.partition.value['train_case_inventory'])['cases']
        self.program=bound(plan.partition.value['program_evidence_inventory'])['cases']
        self.quality=bound(plan.partition.value['quality_case_inventory'])['cases']
        self.streams={slot:[c for c in self.training if c['slot']==slot] for slot in range(8)}
        self.cursors={v:[0]*8 for v in seeds.VARIANTS}
        self.program_cursors={v:0 for v in seeds.VARIANTS}
        self.quality_cursors={v:0 for v in ('reference',*seeds.VARIANTS)}
        self.initialized=set();self.pending={};self.records=[]
        self.path=Path(journal_path);self.handle=self.path.open('xb')

    def _append(self, kind, **values):
        require(not self.failed and not self.closed,'reset ledger is closed or failed')
        payload=dict(version=JOURNAL_VERSION,candidate_id=CANDIDATE,plan=self.plan.ref,
            sequence=len(self.records),previous_sha256=None if not self.records else self.records[-1]['sha256'],
            kind=kind,**values)
        record=dict(payload=payload,sha256=digest(payload))
        try:
            raw=(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
            self.handle.write(raw);self.handle.flush();os.fsync(self.handle.fileno())
        except BaseException:
            self.failed=True
            raise
        self.records.append(record)
        return record

    def predebit_initialization(self, variant):
        require(variant in seeds.VARIANTS and variant not in self.initialized,'initialization cannot repeat')
        record=self._append('initialization_predebit',variant=variant,
            initialization_seed=self.plan.partition.value['initialization_seed'],
            optimizer_seed=self.plan.partition.value['optimizer_seed'],reference_actor_sha256=REFERENCE_SHA256)
        self.initialized.add(variant)
        return deepcopy(record)

    def predebit_reset(self, variant, case, *, scope):
        require(scope in ('train','program_evidence','quality'),'unknown finite reset scope')
        require(variant in (('reference',*seeds.VARIANTS) if scope=='quality' else seeds.VARIANTS),
                'unknown reset policy')
        require(scope=='quality' or variant in self.initialized,'reset cannot precede its initialization predebit')
        require(variant not in self.pending,'bind the previous actual reset before another predebit')
        if scope=='train':
            require(type(case.get('slot')) is int and 0<=case['slot']<8,'unknown training slot')
            slot=case['slot'];cursor=self.cursors[variant][slot];stream=self.streams[slot]
        elif scope=='program_evidence':
            slot=None;cursor=self.program_cursors[variant];stream=self.program
        else:
            slot=None;cursor=self.quality_cursors[variant];stream=self.quality
        require(cursor<len(stream) and case==stream[cursor],'reset outside the exact finite case order')
        record=self._append('environment_constructor_and_reset_predebit',variant=variant,scope=scope,
                            case=deepcopy(case),cursor=cursor,slot=slot)
        if scope=='train':self.cursors[variant][slot]+=1
        elif scope=='program_evidence':self.program_cursors[variant]+=1
        else:self.quality_cursors[variant]+=1
        self.pending[variant]=record
        return deepcopy(record)

    def bind_actual_reset(self, variant, predebit_sha256, reset, *, initial_environment_ref):
        require(variant in self.pending and self.pending[variant]['sha256']==predebit_sha256,
                'actual reset has no unique pending predebit')
        pending=self.pending[variant]['payload'];case=pending['case']
        require(type(reset) is dict and type(reset.get('seed')) is int
            and reset['seed']==case['scenario_seed'] and type(reset.get('episode_id')) is int
            and reset['episode_id']==1,'actual reset seed or fresh physical episode ID differs')
        partitions._immutable(initial_environment_ref)
        record=self._append('actual_reset_bound',variant=variant,scope=pending['scope'],
            predebit_sha256=predebit_sha256,reset=deepcopy(reset),initial_environment=deepcopy(initial_environment_ref))
        del self.pending[variant]
        return deepcopy(record)

    def snapshot(self):
        require(not self.failed and not self.closed and not self.pending,'unbound or failed reset is not a safe boundary')
        self.handle.flush();os.fsync(self.handle.fileno())
        return dict(version=VERSION,candidate_id=CANDIDATE,plan=deepcopy(self.plan.ref),
            journal=pointer(self.path),journal_bytes=self.path.stat().st_size,journal_records=len(self.records),
            journal_head_sha256=None if not self.records else self.records[-1]['sha256'],
            initialized=sorted(self.initialized),reset_cursors=deepcopy(self.cursors),
            program_cursors=deepcopy(self.program_cursors),quality_cursors=deepcopy(self.quality_cursors),
            unbound_predebits=0,training_authority=False,resume_authority=False)

    def close(self):
        if not self.closed:
            self.handle.flush();os.fsync(self.handle.fileno());self.handle.close();self.closed=True
