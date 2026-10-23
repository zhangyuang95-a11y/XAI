"""Synthetic lower-auditor fixtures only; never actual development evidence."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.warehouse import v642_formal_training_admission as f
from experiments.warehouse import v642_development_admission as d
from scripts.run_warehouse_v625 import write_new


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);write_new(path,value);path.chmod(0o444);return d.pointer(path)


def install_fake_development_auditor(tmp_path,monkeypatch,*,budget=200000):
    """The real lock issue/read/token code runs; ONLY its lower auditor is fake."""
    monkeypatch.setattr(d,'OUTPUT',tmp_path/'synthetic-development-admission')
    monkeypatch.setattr(f,'_CACHE',{})
    evidence=put(tmp_path/'synthetic-lower-audit-input.json',dict(synthetic_only=True))
    calls=[]
    def fake_lower_auditor(attempts):
        calls.append(deepcopy(attempts))
        return dict(version=d.VERSION,attempts=deepcopy(attempts),source_files=d.source_files(),
            groups=[dict(synthetic_only=True)],selected_joint_step_budget=budget,all_three_streams_passed=True,
            next_joint_step_budget=None,input_asset_refs=[evidence],formal_preparation_authorized=True,
            sealed_authority=False,model_acceptance_passed=False,pilot_opened=False,new_environment_resets=0,NN_updates=0)
    monkeypatch.setattr(d,'audit_attempts',fake_lower_auditor)
    ref=d.issue_budget_lock([dict(synthetic_lower_auditor_not_real_scientific_evidence=True)])
    return ref,evidence,calls


@pytest.fixture
def lock(tmp_path,monkeypatch):return install_fake_development_auditor(tmp_path,monkeypatch)


def proposal(ref,budget=200000):
    return dict(version=f.PROPOSAL_VERSION,phase='formal',stream_id='formal-0',joint_step_budget=budget,
        initialization_seed=700,optimizer_seed=701,train_case_inventory=dict(path='unread',sha256='a'*64),
        program_evidence_inventory=dict(path='unread',sha256='b'*64),development_budget_lock=ref)


def test_only_actual_reader_can_construct_or_cache_formal_token_and_retains_training_only_scope(lock):
    ref,evidence,calls=lock
    with pytest.raises(RuntimeError,match='actual three-stream'):f.FormalTrainingAdmission()
    token=f.read_admission(ref)
    assert type(token) is f.FormalTrainingAdmission and type(token._budget_lock) is d.BudgetLock
    assert token.budget==200000 and token.budget_lock_ref==ref and evidence in token.refs
    assert token.value['formal_streams']==['formal-0','formal-1','formal-2']
    assert token.value['training_only'] and not token.value['quality_selection_authorized']
    assert not token.value['sealed_authority'] and not token.value['model_acceptance_passed'] and not token.value['pilot_opened']
    assert len(calls)==2  # issue + actual read/replay; cached verification never invents replay.
    assert f.read_admission(ref) is token and len(calls)==2
    token.validate_proposal(proposal(ref))


@pytest.mark.parametrize('change',['budget','stream','phase','restart','lock','version'])
def test_formal_proposal_cannot_choose_budget_stream_restart_or_another_lock(lock,change):
    ref,_,_=lock;token=f.read_admission(ref);p=proposal(ref)
    if change=='budget':p['joint_step_budget']=300000
    elif change=='stream':p['stream_id']='formal-3'
    elif change=='phase':p['phase']='development'
    elif change=='restart':p['restart_declaration']=dict(passed=True)
    elif change=='lock':p['development_budget_lock']=dict(ref,sha256='0'*64)
    else:p['version']='warehouse-v642-long-pair-seed-proposal.v1'
    with pytest.raises(RuntimeError,match='exact real budget lock'):token.validate_proposal(p)


def test_cached_genuine_token_rehashes_all_original_inputs_and_rejects_mutation(lock):
    ref,evidence,calls=lock;token=f.read_admission(ref)
    path=Path(evidence['path']);path.chmod(0o644);path.write_bytes(b'{"synthetic_only":"changed"}');path.chmod(0o444)
    with pytest.raises(RuntimeError,match='changed'):f.read_admission(ref)
    assert len(calls)==2 and token.refs[2]==evidence


def test_cache_cannot_be_replaced_with_passed_dictionary(lock):
    ref,_,_=lock;f._CACHE[f.digest(ref)]=dict(passed=True)
    with pytest.raises(RuntimeError,match='genuine previously read token'):f.read_admission(ref)


def test_missing_lock_rejects_before_source_or_other_input_access(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('missing formal lock accessed a source or lower capability')
    monkeypatch.setattr(f,'source_files',forbidden);monkeypatch.setattr(d,'read_budget_lock',forbidden)
    with pytest.raises(RuntimeError,match='before other inputs'):f.read_admission(None)
    assert not list(tmp_path.iterdir())


def test_passed_JSON_cannot_replace_actual_independent_budget_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT',tmp_path/'fake-passed');monkeypatch.setattr(f,'_CACHE',{})
    ref=put(d.OUTPUT/'budget-lock.json',dict(version=d.VERSION,passed=True,all_three_streams_passed=True))
    with pytest.raises((RuntimeError,KeyError)):f.read_admission(ref)
    assert not f._CACHE


def test_flat_source_hashes_avoid_development_owner_seed_recursion():
    current=f.source_files()
    assert set(current)==set((*f.FILES,*f.DEVELOPMENT_FILES))
    full=d.source_files()
    assert all(full[k]==v for k,v in current.items())
