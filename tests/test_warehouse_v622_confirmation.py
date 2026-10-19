"""Synthetic metadata/fake-physics tests; no real scene, NN, fit, or seed claim."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from experiments.warehouse import v622_confirmation as c


@pytest.fixture(autouse=True)
def synthetic_winner_sources(monkeypatch):
    # Metadata fixtures declare a tiny scientific closure. Production reads
    # and requires the full unchanged v622 scientific source closure.
    monkeypatch.setattr(c.capacity,'source_closure_files',lambda root:{'fixture-source.py':c.file_sha(Path(root)/'fixture-source.py')})
    monkeypatch.setattr(c.recovery,'source_closure_files',lambda root:c.capacity.source_closure_files(root))
    monkeypatch.setattr(c.recovery,'validate_recipe',lambda root:'synthetic-recovery-recipe')
    monkeypatch.setattr(c,'required_history_traces',lambda root,bound:set())


def cases():
    rows=[];index=0
    for split,n in c.COUNTS.items():
        for i in range(n):
            base=800500+index if split!='pressure' else 900500+i//3
            task=i%3+1 if split!='pressure' else (i//3)%3+1
            rows.append({'case_id':f'{split}-{i:03d}','split':split,'base':base,'scenario_seed':base*10+task,
                'task':task,'action_seed':20000000+index,'sampling_mode':'argmax' if split=='fit' and i>=50 else 'sampled',
                'draw':i%3 if split=='pressure' else 0});index+=1
    return rows


def metrics():
    row={'coverage':.6,'agreement':.999,'kl':.01,'expected_actionable_coverage':.55,'whole_tree_raw_illegal_count':0,'whole_tree_path_uncertified_count':0}
    return {'feedback':dict(row),'explanation':dict(row)}


def bundle(root,stage='primary',passing='A32'):
    folder=root/'output/warehouse_v622'/('capacity-02' if stage=='primary' else 'fallback-02')
    folder.mkdir(parents=True)
    source=root/'fixture-source.py';source.write_text('# synthetic source\n')
    sources={'fixture-source.py':c.file_sha(source)}
    plan={'schema':'warehouse-v622-capacity-plan.v1','candidate_id':c.capacity.CANDIDATE_ID,'stage':stage,
        'jobs':c.capacity.jobs(stage),'folds':c.capacity.grouped_folds(),'gates':c.capacity.GATES,'limits':c.capacity.LIMITS,'budgets':c.capacity.round_budget_contract(),
        'scientific_retry_permitted':False,'identity':{'source_files_sha256':sources,'source_closure_sha256':c.digest(sources),
            'actor_role_sha256':c.capacity.ROLE_SHA256,'actor':{'sha256':c.capacity.ACTOR_SHA256},'recipe_sha256':'synthetic-recovery-recipe'},
        'environment_resets':0,'actor_updates':0,'PPO_joint_steps':0,'publishable':False,'pilot_opened':False}
    c.write_new(folder/'plan.json',plan);plan_sha=c.file_sha(folder/'plan.json');results=[];refs={};audited=[]
    c.write_new(folder/'plan.claim.json',{'sha256':plan_sha,'single_use':True})
    for job in plan['jobs']:
        d=folder/'jobs'/job['job_id'];d.mkdir(parents=True)
        raw={'method':c.capacity.METHODS[job['arm']],'actor_role_sha256':c.capacity.ROLE_SHA256[job['role']]}
        c.write_new(d/'program.json',raw);m=metrics()
        if job['arm']!=passing:m['explanation']['coverage']=.1
        gate=c.capacity.gate_metrics(m)
        result={'job':job,'status':'completed','plan_sha256':plan_sha,'actor_weights_unchanged':True,'excluded_frames':0,
            'metrics':{'validation':m},'program':c.pointer(d/'program.json'),'development_gate':gate,'leaf_count':20,
            'budget_audit':{'passed':True},'environment_resets':0,'actor_updates':0,'PPO_joint_steps':0,'publishable':False,'pilot_opened':False}
        c.write_new(d/'result.json',result);ref=c.pointer(d/'result.json');refs[job['job_id']]=ref;results.append(result)
        c.write_new(d/'completed.json',{'plan_sha256':plan_sha,'result_sha256':ref['sha256'],'passed':gate['passed']})
        audited.append({'job':job,'passed':True,'program_sha256':result['program']['sha256']})
    summary={'schema':'warehouse-v622-capacity-summary.v1','candidate_id':c.capacity.CANDIDATE_ID,'stage':stage,
        'plan_sha256':plan_sha,'jobs':refs,**c.capacity.summarize_units(results,stage=stage)}
    c.write_new(folder/'summary.json',summary)
    audit={'schema':'warehouse-v622-independent-capacity-audit.v1','passed':True,'candidate_id':c.capacity.CANDIDATE_ID,
        'stage':stage,'plan_sha256':plan_sha,'summary_sha256':c.file_sha(folder/'summary.json'),'source_closure_sha256':c.digest(sources),
        'job_count':len(results),'jobs':audited,'environment_resets':0,'actor_updates':0,'PPO_joint_steps':0,'publishable':False,'pilot_opened':False}
    c.write_new(folder/'independent-audit.json',audit)
    return {'plan':c.pointer(folder/'plan.json'),'summary':c.pointer(folder/'summary.json'),'audit':c.pointer(folder/'independent-audit.json')}


def test_exact491_schedule_and_pressure_reuse_are_explicit_without_reset():
    rows=cases();assert c.checked_cases(rows) is rows
    plan=c.exact_seed_plan(rows)
    assert len(plan['uses'])==4*491 and plan['authorization_granted'] is False
    assert len({r['scenario_seed'] for r in rows})==347
    assert sum(r['kind']=='action_rng' and r['consumed_as_random_draw'] for r in plan['uses'])==441
    assert c.recipe()['training_authorized'] is False


@pytest.mark.parametrize('change',[
    lambda rows:rows.pop(),
    lambda rows:rows[0].update(case_id='../escape'),
    lambda rows:rows[100].update(sampling_mode='argmax'),
    lambda rows:rows[200].update(action_seed=rows[0]['action_seed']),
    lambda rows:rows[275].update(draw=3),
    lambda rows:rows[275].update(base=rows[0]['base'],scenario_seed=rows[0]['scenario_seed']),
    lambda rows:rows[0].update(scenario_seed=240100),
])
def test_missing_overlap_sealed_path_and_purpose_cases_rejected(change):
    rows=cases();change(rows)
    with pytest.raises(ValueError):c.checked_cases(rows)


def test_winner_requires_complete_current02_independent_audit(tmp_path):
    refs=bundle(tmp_path)
    arm,_=c.checked_winner(tmp_path,{'primary':refs,'fallback':None},{})
    assert arm=='A32'
    p=Path(refs['audit']['path']);a=json.loads(p.read_bytes());a['job_count']=11;p.write_text(json.dumps(a));refs['audit']=c.pointer(p)
    with pytest.raises(RuntimeError,match='complete independent'):c.checked_winner(tmp_path,{'primary':refs,'fallback':None},{})


def test_old01_or_incomplete_child_cannot_unlock_fresh_confirmation(tmp_path):
    refs=bundle(tmp_path)
    first=c.capacity.jobs()[0]['job_id'];folder=tmp_path/'output/warehouse_v622/capacity-02/jobs'/first
    (folder/'completed.json').unlink()
    with pytest.raises((RuntimeError,FileNotFoundError)):c.checked_winner(tmp_path,{'primary':refs,'fallback':None},{})
    refs=deepcopy(refs);refs['plan']['path']=refs['plan']['path'].replace('capacity-02','capacity-01')
    with pytest.raises(RuntimeError):c.checked_winner(tmp_path,{'primary':refs,'fallback':None},{})


def test_no_winner_is_failclosed_before_actor_or_environment(tmp_path,monkeypatch):
    refs=bundle(tmp_path,passing=None)
    with pytest.raises(RuntimeError,match='no audited winner'):c.checked_winner(tmp_path,{'primary':refs,'fallback':None},{})
    with pytest.raises(RuntimeError,match='ordinary callers'):c.ConfirmationPlan(tmp_path,Path('x'),{},tmp_path,object())
    with pytest.raises(RuntimeError):c.collect(tmp_path,tmp_path/'missing-plan.json')


def test_recipe_binding_excludes_only_clearance_to_avoid_circular_hashes():
    config={'schema':c.SCHEMA,'cases':cases(),'seed_clearance':{'path':'first','sha256':'a'*64}}
    identity=c.configuration_identity(config);config['seed_clearance']={'path':'second','sha256':'b'*64}
    assert c.configuration_identity(config)==identity
    config['cases'][0]['action_seed']+=999999
    assert c.configuration_identity(config)!=identity


def test_cli_ordinary_missing_plan_and_contract_grant_no_authority(capsys):
    from scripts import run_warehouse_v622_confirmation as script
    with pytest.raises(SystemExit):script.main([])
    with pytest.raises(SystemExit):script.main(['--collect'])
    assert script.main(['--contract'])==0
    value=json.loads(capsys.readouterr().out)
    assert value['seed_allocation'] is False and value['recipe']['training_authorized'] is False


def test_prepare_and_load_replay_real_exact_inventory_delta_on_four_synthetic_roots(tmp_path,monkeypatch):
    from experiments.warehouse import exact_seed_audit as exact,seed_metadata_audit as metadata
    parent=tmp_path/'ICLR';parent.mkdir()
    for name in c.HISTORY_ROOT_NAMES:(parent/name).mkdir()
    root=parent/'XAI-v6-integration';refs=bundle(root)
    actor=root/'synthetic-reference.npz';actor.write_bytes(b'synthetic-not-a-model')
    monkeypatch.setattr(c.capacity,'ACTOR_SHA256',c.file_sha(actor))
    # Recreate only actor binding in the native plan, then resign descendants.
    plan_path=Path(refs['plan']['path']);plan=json.loads(plan_path.read_bytes());plan['identity']['actor']['sha256']=c.capacity.ACTOR_SHA256
    plan_path.write_text(json.dumps(plan));new_plan_sha=c.file_sha(plan_path)
    (plan_path.parent/'plan.claim.json').write_text(json.dumps({'sha256':new_plan_sha,'single_use':True}))
    summary_path=Path(refs['summary']['path']);summary=json.loads(summary_path.read_bytes());summary['plan_sha256']=new_plan_sha
    for ref in summary['jobs'].values():
        p=Path(ref['path']);value=json.loads(p.read_bytes());value['plan_sha256']=new_plan_sha;p.write_text(json.dumps(value));ref['sha256']=c.file_sha(p)
        complete=p.parent/'completed.json';value=json.loads(complete.read_bytes());value.update(plan_sha256=new_plan_sha,result_sha256=ref['sha256']);complete.write_text(json.dumps(value))
    summary_path.write_text(json.dumps(summary));ap=Path(refs['audit']['path']);value=json.loads(ap.read_bytes());value.update(plan_sha256=new_plan_sha,summary_sha256=c.file_sha(summary_path));ap.write_text(json.dumps(value))
    refs={k:c.pointer(Path(v['path'])) for k,v in refs.items()}
    repo=Path(__file__).resolve().parents[1]
    for name in c.NEW_SOURCES:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/name).read_bytes())
    ledger=root/'output/warehouse_v622/confirmation-synthetic-seed-ledger';ledger.mkdir(parents=True)
    rows=cases();seed_path=ledger/'exact-plan.json';c.write_new(seed_path,c.exact_seed_plan(rows))
    files=c.source_files(root,plan)
    config={'schema':c.SCHEMA,'candidate_id':c.capacity.CANDIDATE_ID,'run_id':'synthetic','winner':{'primary':refs,'fallback':None},
        'method':c.capacity.METHODS['A32'],'actor':{'path':str(actor),'sha256':c.capacity.ACTOR_SHA256,'role_sha256':c.capacity.ROLE_SHA256},
        'cases':rows,'recipe':c.recipe(),'source_closure_sha256':c.digest(files),'exact_seed_plan':c.pointer(seed_path),'seed_clearance':{}}
    proposal=ledger/'proposal.json';families=exact.compile_families(c.exact_seed_plan(rows))
    c.write_new(proposal,{'schema':'warehouse-seed-proposal.v1','candidate_id':c.capacity.CANDIDATE_ID,
        'families':[{k:v for k,v in f.items() if k not in ('intervals','count')} for f in families],'exclude_paths':[str(ledger)]})
    roots=[parent/name for name in c.HISTORY_ROOT_NAMES]
    inventory=metadata.audit(proposal,roots,ledger/'inventory')
    exact.assess(seed_path,ledger/'inventory/inventory.json',ledger/'exact')
    delta=exact.delta_inventory(inventory,proposal,output_directory=ledger/'delta')
    clearance={'schema':c.CLEARANCE_SCHEMA,'candidate_id':c.capacity.CANDIDATE_ID,'configuration_identity_sha256':c.configuration_identity(config),
        'cases_sha256':c.digest(rows),'source_closure_sha256':config['source_closure_sha256'],'approved_for_confirmation':True,'unresolved':[],
        'remaining_conflicts':[],'PPO_joint_steps':0,'actor_updates':0,'pilot_opened':False,'publishable':False,'sealed_evaluation_authorized':False,
        'valid_until_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'exact_evidence':c.pointer(ledger/'exact/result.json'),
        'final_delta':c.pointer(ledger/'delta/delta.json')}
    c.write_new(ledger/'clearance.json',clearance);config['seed_clearance']=c.pointer(ledger/'clearance.json');config_path=ledger/'configuration.json';c.write_new(config_path,config)
    prepared=c.prepare(root,config_path)
    assert c.load_plan(root,prepared)[1]['method']==c.capacity.METHODS['A32']
    assert not (prepared.parent/'collection').exists()
    # Opaque weight files are stat-only; a new name must block even though
    # metadata JSON paths and every content-read count remain unchanged.
    weight=parent/'XAI'/'output/warehouse-synthetic/new-model.npz';weight.parent.mkdir(parents=True);weight.write_bytes(b'opaque-not-read')
    with pytest.raises(RuntimeError,match='metadata changed'):c.load_plan(root,prepared,start_collection=True)
    weight.unlink()
    # A later historical execution file is a real snapshot delta, even though
    # this fixture never constructs an environment or uses its seed arguments.
    c.write_new(parent/'XAI'/'new-seed-report.json',{'seed':rows[0]['scenario_seed']})
    with pytest.raises(RuntimeError,match='metadata changed'):c.load_plan(root,prepared,start_collection=True)
    assert c.load_plan(root,prepared)[1]['arm']=='A32'  # Offline receipts retain their historical validity.


@pytest.mark.parametrize('override,bad_record',[(False,False),(True,False),(False,True)])
def test_collector_full_NN_draws_native_reset_journal_and_no_action_override(tmp_path,monkeypatch,override,bad_record):
    from experiments.warehouse import v622_confirmation as c
    from backend.training import warehouse_v6_actor as frozen
    from env.warehouse import environment as physics,public_observation_v6 as public,pure_nn
    from env.warehouse.decision_protocol import independent_agent_seed
    from env.warehouse.reset_audit_v614 import record_successful_reset
    from core.policy_contracts import ActionDistribution
    rows=cases();ref={'path':'synthetic-no-real-model','sha256':'a'*64,'role_sha256':{r:'b'*64 for r in c.capacity.ROLES}}
    output=tmp_path/'collection';output.mkdir();plan=tmp_path/'plan.json';plan.write_text('{}')
    cap=c.ConfirmationPlan(tmp_path,plan,{'cases':rows,'actor':ref,'source_files_sha256':{}},output,c._SEAL)
    (output/'reset-preauthorization.jsonl').touch()
    @contextmanager
    def fake_capability(*args):
        token=c._ACTIVE.set(cap)
        try:yield cap
        finally:c._ACTIVE.reset(token)
    monkeypatch.setattr(c,'collection_capability',fake_capability)
    class FakeEnv:
        resets=0
        def __init__(self,config):self.config=config
        def reset(self,*,seed):
            FakeEnv.resets+=1;self.state=SimpleNamespace(frame=0,episode_id=1,terminated=False,truncated=False,total_deliveries=0,robot_collision_events=0,shutdown_count=0)
            record_successful_reset(seed,1,self.config.map_layout_id,self.config.move_battery_cost,self.config.charge_per_wait)
        def get_state(self):return self.state
        def set_state(self,state):self.state=state
        def step(self,actions):
            self.state.frame=1;self.state.truncated=True
            self.state.proposed={r:('LEFT' if bad_record else actions[r]) for r in c.capacity.ROLES}
            return None,None,None,None,{'proposed_actions':actions,'action_override_agents':['robot_2'] if override else [],
                'executed_actions':{r:'WAIT' for r in c.capacity.ROLES}}
    class FakeActor:
        def logits(self,obs,masks):return np.tile(np.array([0,0,0,0,2],np.float32),(2,1))
        def act(self,state,*,base_seed,deterministic):
            masks=np.ones((2,5),np.float32);p=np.exp(self.logits(None,masks)-2);p/=p.sum(-1,keepdims=True)
            actions={};distributions={}
            for i,r in enumerate(c.capacity.ROLES):
                rng=independent_agent_seed(base_seed=base_seed,episode_id=state.episode_id,frame=state.frame,agent_id=r)
                index=int(p[i].argmax()) if deterministic else int(np.random.default_rng(rng).choice(5,p=p[i]))
                actions[r]=('UP','DOWN','LEFT','RIGHT','WAIT')[index]
                distributions[r]=SimpleNamespace(probabilities=tuple(float(x) for x in p[i]))
            return actions,distributions
    monkeypatch.setattr(physics,'WarehouseMultiAgentEnv',FakeEnv)
    monkeypatch.setattr(frozen.FrozenWarehouseV6Actor,'load',lambda *a,**kw:FakeActor())
    monkeypatch.setattr(public,'encode_public_observation',lambda state:np.zeros((2,141),np.float32))
    monkeypatch.setattr(public,'public_pre_action_snapshot',lambda state:{'frame':state.frame,'episode_id':state.episode_id,
        'agents':[{'agent_id':r,'last_action':getattr(state,'proposed',{}).get(r,'WAIT'),'last_executed_action':'WAIT'} for r in c.capacity.ROLES]})
    monkeypatch.setattr(pure_nn,'action_masks',lambda state:np.ones((2,5),np.float32))
    if override or bad_record:
        with pytest.raises(RuntimeError,match='controller' if override else 'different NN proposal'):c.collect(tmp_path,plan)
        assert FakeEnv.resets==1 and not (output/'completed.json').exists()
        return
    result=c.collect(tmp_path,plan)
    assert FakeEnv.resets==491 and cap.cursor[0]==491 and result['actual_reset_count']==491
    assert result['NN_action_checks']==982 and result['actor_updates']==result['PPO_joint_steps']==0
    assert result['post_policy_interventions']==0
    trace=json.loads((output/'validation-000.jsonl').read_text())
    assert trace['pre_state_public_sha256']==c.digest(trace['pre_state_public'])
    assert trace['post_state_public_sha256']==c.digest(trace['post_state_public'])
    assert any(json.loads(p.read_text())['proposed_actions']!={'robot_1':'WAIT','robot_2':'WAIT'} for p in output.glob('validation-*.jsonl'))
    native=[json.loads(s) for s in (output/'actual-resets.jsonl').read_text().splitlines()]
    assert [r['seed'] for r in native]==[r['scenario_seed'] for r in rows]


@pytest.mark.parametrize('bad_split,bad_kind',[('validation','deliveries'),('pressure','deliveries'),
    ('pressure','robot_collisions'),('pressure','shutdown'),('pressure','program_coverage'),('pressure','program_kl')])
def test_fit_cannot_hide_pressure_or_validation_failure_with_pooled_quality(tmp_path,monkeypatch,bad_split,bad_kind):
    """Real entry decision on synthetic rows; all scientific fits are stubs."""
    from backend.training import warehouse_v622_program as program
    root=tmp_path;path=root/'plan.json';path.write_text('{}')
    plan={'arm':'B32','method':c.capacity.METHODS['B32'],'actor':{'sha256':'synthetic'}}
    actor=SimpleNamespace(roles={r:object() for r in c.capacity.ROLES})
    parts={s:{r:[{'split':s}] for r in c.capacity.ROLES} for s in c.COUNTS}
    outcomes=[{'case':row,'deliveries':10,'robot_collisions':0,'shutdown':False} for row in cases()]
    if bad_kind in ('deliveries','robot_collisions','shutdown'):
        for row in outcomes:
            if row['case']['split']==bad_split:
                row[bad_kind]=True if bad_kind=='shutdown' else 0 if bad_kind=='deliveries' else 7
    collection=root/'collection';collection.mkdir();(collection/'result.json').write_text('{}')
    monkeypatch.setattr(c,'load_plan',lambda *a,**kw:(path,plan))
    monkeypatch.setattr(c,'collection_rows',lambda *a:(actor,parts,{'outcomes':outcomes}))
    monkeypatch.setattr(program,'compile_actor_structure',lambda *a:({'leaf_count':1},{}))
    monkeypatch.setattr(program,'program_structure_sha256',lambda *a:'synthetic')
    class FakeProgram:
        def save(self,p):c.write_new(p,{'synthetic':True})
    monkeypatch.setattr(program,'fit_program',lambda *a,**kw:FakeProgram())
    def fake_metrics(unused,unused_actor,rows):
        m=metrics()
        for v in m.values():v.update(illegal_count=0,path_uncertified_count=0,actionable_coverage=.6)
        if rows[0]['split']==bad_split:
            if bad_kind=='program_coverage':
                for v in m.values():v.update(coverage=.35,expected_actionable_coverage=.33,actionable_coverage=.33)
            if bad_kind=='program_kl':
                for v in m.values():v['kl']=.23
        return m
    monkeypatch.setattr(program,'program_metrics',fake_metrics)
    result=c.fit(root,path)
    assert result['native_passed'] is False
    assert result['quality_by_split']['validation']['episodes']==75
    assert result['quality_by_split']['pressure']['episodes']==216
    if bad_kind.startswith('program_'):
        # Both counterexamples satisfy the obsolete pressure gate and must
        # still fail this entry's same-as-validation strict precheck.
        m=result['roles']['robot_1']['metrics']['pressure']
        assert program.gate_metrics(m,phase='pressure')['passed'] is True
        assert result['roles']['robot_1']['pressure_gate']['passed'] is False
    else:
        assert result['quality_gates_by_split'][bad_split]['passed'] is False
        other='pressure' if bad_split=='validation' else 'validation'
        assert result['quality_gates_by_split'][other]['passed'] is True


@pytest.mark.parametrize('change',[
    lambda p:p['identity'].update(recipe_sha256='original-not-recovery'),
    lambda p:p.update(budgets={}),
    lambda p:p['identity']['source_files_sha256'].update(extra='f'*64),
])
def test_recovery_recipe_budgets_and_exact_source_identity_are_not_self_attested(tmp_path,change):
    refs=bundle(tmp_path);p=Path(refs['plan']['path']);value=json.loads(p.read_bytes());change(value);p.write_text(json.dumps(value))
    refs['plan']=c.pointer(p)
    with pytest.raises(RuntimeError):c.checked_winner(tmp_path,{'primary':refs,'fallback':None},{})
