"""Controller sequencing and source registration, never candidate training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pytest
from experiments.warehouse import v629_calibration_pipeline as c
from scripts import run_warehouse_v629_calibration as cli


def helper():
    spec=importlib.util.spec_from_file_location('pipeline_owner_fixture',Path(__file__).with_name('test_warehouse_v629_calibration_owner.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('stage',['prepare','audit_seeds','run','cli'])
def test_real_closed_A_B_gate_precedes_input_reads_and_all_work(tmp_path,monkeypatch,stage):
    monkeypatch.setattr(c.owner.seeds.partitions.admission.v,'OUTPUT',tmp_path/'missing-winner')
    def forbidden(*args,**kwargs):raise AssertionError('missing A/B cannot read a plan or construct a training owner')
    monkeypatch.setattr(c,'pointer',forbidden);monkeypatch.setattr(c.owner,'bound',forbidden)
    monkeypatch.setattr(c.owner,'initialize_from_reference',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):
        if stage=='cli':cli.main(['--run','--reference-directory',str(tmp_path/'reference'),'--plan',str(tmp_path/'absent-plan')])
        else:getattr(c,stage)(reference_directory=tmp_path/'reference',plan_ref=None)
    assert not list(tmp_path.iterdir())


@pytest.fixture
def preregistered(tmp_path,monkeypatch):
    h=helper();original=c.owner.audit_source_bound_seeds
    monkeypatch.setattr(c.owner,'audit_source_bound_seeds',lambda **kw:None)
    plan,reference=h.preparation.__wrapped__(tmp_path,monkeypatch)
    monkeypatch.setattr(c.owner,'audit_source_bound_seeds',original)
    contract=c.prepare(reference_directory=reference,plan_ref=plan.ref)
    return plan,reference,contract


def test_complete_controller_source_is_snapshotted_before_exact_audit(preregistered):
    plan,reference,contract=preregistered
    assert c.checked_contract(plan)==contract and not (Path(plan.ref['path']).parent/'seed-ledger').exists()
    value=c.bound(contract)
    assert value['source_files']==c.source_files() and value['config']==c.asdict(c.PPOConfig())
    clearance=c.audit_seeds(reference_directory=reference,plan_ref=plan.ref)
    assert c.bound(clearance)==dict(version=c.VERSION,controller_source_contract=contract,
        owner_source_clearance=c.pointer(Path(plan.ref['path']).with_name('owner-source-clearance.json')),training_authority=False)
    with pytest.raises(RuntimeError,match='before calibration history audit'):
        c.prepare(reference_directory=reference,plan_ref=plan.ref)
    assert not c.owner.claim_path().exists()


def test_changed_controller_snapshot_rejected_before_any_claim(preregistered):
    plan,reference,contract=preregistered;value=c.bound(contract)
    path=Path(plan.ref['path']).parent/'controller-source'/value['source_sha256']/'scripts/run_warehouse_v629_calibration.py'
    path.chmod(0o644);path.write_bytes(b'tampered controller')
    with pytest.raises(RuntimeError,match='snapshot changed'):
        c.audit_seeds(reference_directory=reference,plan_ref=plan.ref)
    assert not c.owner.claim_path().exists() and not (Path(plan.ref['path']).parent/'seed-ledger').exists()


class FakeOwner:
    """No tensors, RNG, environment, or feedback capability is constructed."""
    def __init__(self):self.config=c.PPOConfig();self.states={};self.events=[];self.refits={v:[] for v in c.owner.seeds.VARIANTS}
    def initialize(self,v):self.states[v]=SimpleNamespace(joint_steps=0);self.events.append(('init',v))
    def collect_rollout(self,v):
        s=self.states[v]
        assert self.refits[v]==[step for step in self.config.extraction_steps if step<=s.joint_steps]
        self.events.append(('collect',v,s.joint_steps));s.joint_steps+=8*self.config.next_collection_length(s.joint_steps)
    def update(self,v):self.events.append(('update',v,self.states[v].joint_steps))
    def checkpoint(self,v):
        step=self.states[v].joint_steps
        assert self.refits[v]==[s for s in self.config.extraction_steps if s<=step]
        self.events.append(('checkpoint',v,step));return {'fixture':v,'step':step}


def test_round_robin_calls_real_owner_methods_in_exact_boundary_order(monkeypatch):
    active=FakeOwner();seen=[];verifications=[]
    def refit(a,v,programs):
        step=a.states[v].joint_steps;seen.append((v,step,deepcopy(programs)));a.refits[v].append(step)
        a.events.append(('refit',v,step));return {r:f'{v}-{step}-{r}' for r in programs}
    monkeypatch.setattr(c,'scheduled_refit',refit)
    initial={'robot_1':'B-locked-one','robot_2':'B-locked-two'}
    result=c.drive(active,initial,lambda:verifications.append(True))
    assert initial=={'robot_1':'B-locked-one','robot_2':'B-locked-two'}
    assert [r[:2] for r in seen]==[(v,s) for s in (15000,16800,18600) for v in ('raw','regularized')]
    for v in ('raw','regularized'):
        own=[r for r in seen if r[0]==v]
        assert own[0][2]==initial and own[1][2]['robot_1']==f'{v}-15000-robot_1'
        assert own[2][2]['robot_1']==f'{v}-16800-robot_1'
        assert [row['step'] for row in result[v]]==[4000,15000,16800,18600,20000]
    updates=[e for e in active.events if e[0]=='update']
    assert updates and [e[1] for e in updates]==['raw','regularized']*(len(updates)//2)
    assert active.states['raw'].joint_steps==active.states['regularized'].joint_steps==20000
    assert len(verifications)>len(updates)


def test_refit_failure_propagates_without_retry_or_later_collection(monkeypatch):
    active=FakeOwner();calls=[]
    def fail(a,v,programs):calls.append((v,a.states[v].joint_steps));raise RuntimeError('synthetic refit failure')
    monkeypatch.setattr(c,'scheduled_refit',fail)
    with pytest.raises(RuntimeError,match='synthetic refit failure'):c.drive(active,{'robot_1':'one','robot_2':'two'},lambda:None)
    assert calls==[('raw',15000)] and active.states['raw'].joint_steps==15000 and active.states['regularized'].joint_steps<15000
    assert not any(e[0]=='checkpoint' and e[2]>=15000 for e in active.events)


def test_controller_cannot_silently_switch_to_formal_budget():
    active=FakeOwner();active.config=c.PPOConfig(joint_step_budget=200000)
    with pytest.raises(RuntimeError,match='fixed 20k'):c.drive(active,{},lambda:None)
    assert not active.events
