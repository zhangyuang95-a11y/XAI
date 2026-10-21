"""Finite, source-bound collection AFTER a complete audited v629 winner.

No new cases are allocated or reset by importing this module. The allocation,
metadata audit, final history delta and single-use claim are explicit stages.
Only the frozen complete NN submits actions; programs are scoring evidence.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime,timedelta,timezone
from pathlib import Path
from dataclasses import dataclass
import gzip,json,os,re
import numpy as np
from . import v629_protocol as v, v629_confirmation_metrics as metrics
from . import seed_metadata_audit as metadata, exact_seed_audit as exact
from . import v628_seed_triage_bounded as triage
from .v628_data import HISTORY_NAMES
from scripts.run_warehouse_v625 import write_new

ROOT=v.ROOT
ACTIVE=ContextVar('warehouse_v629_fixed_confirmation',default=None)
FILES=('experiments/warehouse/v629_confirmation.py','experiments/warehouse/v629_confirmation_metrics.py',
    'experiments/warehouse/v628_seed_triage_bounded.py','scripts/run_warehouse_v629_confirmation.py',
    'scripts/audit_warehouse_v629_confirmation.py','scripts/measure_warehouse_v628_collection.py',
    'docs/warehouse_v629_confirmation_plan.md','tests/test_warehouse_v629_confirmation.py',
    'tests/test_warehouse_v629_confirmation_metrics.py')
INVENTORY_REF=dict(path=str(ROOT/'output/warehouse_v621/seed-ledger-03/rebound-01/inventory.json'),
    sha256='98ea8b4d0aed59978d8537e45b8677fa84af848416a64ccc6aa1b0d9edf62b19')


def recipe():
    return dict(schema='warehouse-v629-fixed-confirmation-recipe.v1',episodes=516,
        confirmation_scenes=300,pressure_scenes=72,pressure_draws=3,task_balance=True,
        program_fit_calls=0,program_qualification_fixed=True,temperature_fixed=True,
        action_source='complete_frozen_v620_dual_Actor_and_physical_masks',
        stream_derivation='source_pinned_independent_agent_seed_base_episode_frame_role',
        metadata_roots=list(HISTORY_NAMES),history_clearance_max_age_hours=24,
        program_gate=v.recipe()['gate'],minimum_deliveries=5.5,maximum_collisions=6,maximum_shutdown_rate=.05,
        primary_bootstrap='confirmation_four_role_endpoints_9875_one_sided_scene_paired',
        bootstrap_iterations=20000,bootstrap_numerical_control=0,
        claim_single_use_even_if_interrupted=True,final_admission=False,PPO_joint_steps=0,pilot_opened=False)


def source_files():
    files=v.source_files()
    for name in FILES:files[name]=v.file_sha(ROOT/name)
    return dict(sorted(files.items()))


def run_name(value):
    v.require(isinstance(value,str) and re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',value) and len(value)<=80,'invalid confirmation run ID')
    return value


def output_for(run_id):return ROOT/'output/warehouse_v629'/('confirmation-'+run_name(run_id))
def proposal_for(run_id):return ROOT/'output/warehouse_v629'/('confirmation-proposal-'+run_name(run_id))
def claim_path():return v.OUTPUT/'confirmation.claim.json'


def allocate(run_id,base_start,action_start):
    """Allocation is permitted only after an actual all-six audited winner."""
    winner=metrics.load_locked_bundle()
    v.require(not claim_path().exists(),'this candidate already consumed its one confirmation attempt')
    v.require(type(base_start) is int and type(action_start) is int,'explicit integer seed starts required')
    cases=[]
    for split,scenes,draws,offset in (('validation',300,1,0),('pressure',72,3,300)):
        for index in range(scenes):
            base=base_start+offset+index;task=index%3+1
            for draw in range(draws):
                cases.append(dict(case_id=f'{split}-{index:03d}-{draw}',split=split,base=base,
                    scenario_seed=10*base+task,task=task,action_seed=action_start+len(cases),sampling_mode='sampled',draw=draw))
    metrics.validate_cases(cases)
    folder=proposal_for(run_id);folder.mkdir(parents=True,exist_ok=False)
    value=dict(schema='warehouse-v629-confirmation-allocation.v1',run_id=run_id,cases=cases,
        locked_selection=winner.provenance,source_files=source_files(),recipe=recipe(),
        created_utc=datetime.now(timezone.utc).isoformat(),actual_environment_resets=0,
        new_collection_authority=False,training_authority=False,sealed_authority=False)
    write_new(folder/'allocation.json',value)
    write_new(folder/'allocation.claim.json',dict(sha256=v.file_sha(folder/'allocation.json')))
    return folder/'allocation.json'


def exact_plan(cases):
    metrics.validate_cases(cases);uses=[]
    for case in cases:
        purpose='v629_fixed_confirmation_'+case['split']
        for kind,argument,key in (('scenario','task_expansion_base','base'),('env','random.Random(config.seed)','scenario_seed'),
            ('env','reset(seed)','scenario_seed'),('action_rng','FrozenActor.act(base_seed)','action_seed')):
            row=dict(kind=kind,purpose=purpose,argument=argument,value=case[key],group=case['case_id'],reuse_tag=key+':'+str(case[key]))
            if kind=='action_rng':row['consumed_as_random_draw']=True
            uses.append(row)
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id=v.CANDIDATE+'-fixed-confirmation',
        uses=uses,cases_sha256=v.digest(cases),authorization_granted=False)


def prepare(allocation_path):
    winner=metrics.load_locked_bundle();allocation_path=Path(allocation_path).resolve()
    a=v.bound(v.pointer(allocation_path));run=run_name(a['run_id'])
    v.require(allocation_path==proposal_for(run)/'allocation.json'
        and json.loads(allocation_path.with_name('allocation.claim.json').read_bytes())==dict(sha256=v.file_sha(allocation_path)),
        'allocation must be this candidate\'s single-use recorded proposal')
    v.require(a['schema']=='warehouse-v629-confirmation-allocation.v1' and a['locked_selection']==winner.provenance
        and a['source_files']==source_files() and a['recipe']==recipe() and not claim_path().exists(),'allocation/selection/source changed or candidate already consumed')
    metrics.validate_cases(a['cases'])
    folder=output_for(run);folder.mkdir(parents=True,exist_ok=False)
    sources=source_files();source_sha=v.digest(sources)
    for name,sha in sources.items():
        target=folder/'source'/source_sha/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();v.require(v.file_sha(ROOT/name)==sha,'source changed preparing')
        target.write_bytes(raw);target.chmod(0o444)
    bound={}
    refs=[winner.actor,*winner.provenance.values()]
    for unit in winner.units:refs.extend([unit.program_ref,unit.reference_ref,unit.program.protection['parent'],unit.reference.protection['parent']])
    for ref in refs:
        v.require(v.pointer(ref['path'])==ref,'fixed prerequisite changed')
        v.require(ref['path'] not in bound or bound[ref['path']]==ref['sha256'],'conflicting prerequisite binding')
        bound[ref['path']]=ref['sha256']
    plan=dict(schema='warehouse-v629-fixed-confirmation-plan.v1',candidate_id=v.CANDIDATE,run_id=run,
        allocation=v.pointer(allocation_path),winner=winner.provenance,actor=winner.actor,
        actor_role_sha256=v.old.ROLE_SHA256,cases=a['cases'],recipe=recipe(),source_files=sources,source_sha256=source_sha,
        prerequisites=bound,environment_reset_ceiling=516,training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'plan.json',plan);write_new(folder/'plan.claim.json',dict(sha256=v.file_sha(folder/'plan.json'),single_use=True))
    return folder/'plan.json'


def checked_plan(path):
    path=Path(path).resolve();p=v.bound(v.pointer(path));folder=output_for(p['run_id'])
    v.require(path==folder/'plan.json' and json.loads((folder/'plan.claim.json').read_bytes())==dict(sha256=v.file_sha(path),single_use=True),
        'confirmation plan seal/location differs')
    winner=metrics.load_locked_bundle()
    v.require(p['schema']=='warehouse-v629-fixed-confirmation-plan.v1' and p['candidate_id']==v.CANDIDATE
        and p['winner']==winner.provenance and p['actor']==winner.actor and p['actor_role_sha256']==v.old.ROLE_SHA256
        and p['recipe']==recipe() and p['source_files']==source_files() and p['source_sha256']==v.digest(source_files()),'confirmation source or locked assets differ')
    metrics.validate_cases(p['cases'])
    v.require(v.bound(p['allocation'])['cases']==p['cases'],'confirmation cases changed')
    for name,sha in p['source_files'].items():
        snapshot=folder/'source'/p['source_sha256']/name
        v.require(v.file_sha(snapshot)==sha and not snapshot.stat().st_mode&0o222,'confirmation source snapshot changed')
    for name,sha in p['prerequisites'].items():v.require(v.file_sha(name)==sha,'fixed prerequisite changed')
    if claim_path().exists():
        v.require(v.bound(v.pointer(claim_path()))['plan']==v.pointer(path),'candidate confirmation already claimed by another run')
    return path,p,winner


def exclusions(plan):return [output_for(plan['run_id']),proposal_for(plan['run_id']),claim_path()]


def history_selection(plan,traces):
    stats={};selected=[];roots=[ROOT.parent/name for name in HISTORY_NAMES]
    for path in metadata.candidates(roots,exclusions(plan),registered_trace_paths=traces,selection_stats=stats):
        s=path.stat();selected.append(dict(path=str(path.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    return selected,stats['excluded_weights_name_stat_only']


def template_bindings(inventory,*,check_current):
    rows=[r for r in inventory if r['classification']=='source_template_environment_reservation']
    bindings={r['path']:r['sha256'] for r in rows}
    v.require(len(bindings)==len(rows),'duplicate historical template')
    if check_current:
        current={}
        for name in HISTORY_NAMES:
            path=ROOT.parent/name/'backend/training/seed_ledger.py'
            if path.is_file():
                raw=path.read_bytes()
                if b'500_004' in raw and b'10_500_204' in raw:current[str(path.resolve())]=v.file_sha(path)
        v.require(current==bindings,'historical source reservation changed after audit')
    return bindings


def audit_seeds(path):
    path,p,_=checked_plan(path);folder=path.parent/'seed-ledger';folder.mkdir(exist_ok=False)
    v.require(not claim_path().exists(),'cannot re-audit spent confirmation')
    exact_value=exact_plan(p['cases']);write_new(folder/'exact-plan.json',exact_value);exact.read_exact_plan(folder/'exact-plan.json')
    families=exact.compile_families(exact_value);traces=v.bound(INVENTORY_REF)['registered_trace_paths']
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=v.CANDIDATE+'-fixed-confirmation',
        families=[{k:r[k] for k in ('name','kind','values','query_only')} for r in families],
        exclude_paths=list(map(str,exclusions(p))),registered_trace_paths=traces)
    write_new(folder/'proposal.json',proposal)
    metadata.audit(folder/'proposal.json',[ROOT.parent/name for name in HISTORY_NAMES],folder/'bounded')
    triage.triage(folder/'bounded/inventory.json',folder/'proposal.json',folder/'triaged')
    result=exact.assess(folder/'exact-plan.json',folder/'triaged/result.json',folder/'exact',context_source_root=ROOT)
    v.require(not result['summary']['unresolved_paths'] and not any(any(row.get(k) for k in
        ('references_or_reservations','reservation_overlaps','audit_references_not_original_execution','excluded_weight_path_seed_hints'))
        for row in result['summary']['families'].values()),'historical seed conflict or unresolved metadata')
    selected,weights=history_selection(p,traces)
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory'] if r['classification']!='source_template_environment_reservation'}
    v.require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selected},'history changed during seed audit')
    template_bindings(result['inventory'],check_current=True)
    now=datetime.now(timezone.utc)
    clearance=dict(schema='warehouse-v629-confirmation-clearance.v1',plan=v.pointer(path),cases_sha256=v.digest(p['cases']),
        exact=v.pointer(folder/'exact/result.json'),registered_traces=traces,selection=selected,weight_paths=weights,
        issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),approved_finite_resets=516,
        training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'clearance.json',clearance);return clearance


def checked_clearance(path,p,*,execution=True):
    """Execution rechecks history+expiry; later audit replays bound evidence only."""
    folder=path.parent/'seed-ledger';c=v.bound(v.pointer(folder/'clearance.json'));e=v.bound(c['exact'])
    issued=datetime.fromisoformat(c['issued_utc']);expires=datetime.fromisoformat(c['expires_utc'])
    v.require(c['schema']=='warehouse-v629-confirmation-clearance.v1' and c['plan']==v.pointer(path)
        and c['cases_sha256']==v.digest(p['cases']) and c['approved_finite_resets']==516
        and issued.tzinfo is not None and expires.tzinfo is not None and expires-issued==timedelta(hours=24)
        and (not execution or issued<=datetime.now(timezone.utc)<expires),'confirmation clearance changed or expired')
    v.require(c['exact']==v.pointer(folder/'exact/result.json') and e['exact_plan']==v.pointer(folder/'exact-plan.json'),
        'exact evidence location differs')
    ep=v.bound(e['exact_plan']);v.require(ep==exact_plan(p['cases']) and e['exact_auditor_sha256']==v.file_sha(exact.__file__), 'exact plan/auditor differs')
    inv=v.bound(e['bounded_or_triaged_inventory']);projection=exact.project_inventory(inv,exact.compile_families(ep))
    if e.get('context_annotations'):projection=exact.apply_context(projection,v.bound(e['context_annotations']))
    v.require(v.digest(e['inventory'])==v.digest(projection['inventory']) and e['summary']==projection['summary']
        and not e['summary']['unresolved_paths'],'exact evidence failed replay')
    v.require(sorted(e['scan_roots'])==sorted(str((ROOT.parent/name).resolve()) for name in HISTORY_NAMES)
        and e['scan_roots']==inv['scan_roots'] and not inv['missing_roots']
        and all((ROOT.parent/name).is_dir() for name in HISTORY_NAMES),'all four history roots required')
    traces=v.bound(INVENTORY_REF)['registered_trace_paths']
    expected_exclusions=list(map(str,[folder/'proposal.json',folder/'bounded',*exclusions(p)]))
    v.require(c['registered_traces']==e['registered_trace_paths']==inv['registered_trace_paths']==traces
        and e['selection_scope']==inv['selection_scope']==metadata.SELECTION_SCOPE
        and e['read_model_weight_files'] is inv['read_model_weight_files'] is False
        and sorted(e['scope_exclusions'])==sorted(inv['scope_exclusions'])==sorted(expected_exclusions),
        'history trace registry/scope/exclusions differ')
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in e['inventory'] if r['classification']!='source_template_environment_reservation'}
    v.require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in c['selection']}
        and len(audited)==len(c['selection']),'audited inventory differs from clearance selection')
    template_bindings(e['inventory'],check_current=execution)
    v.require(not any(any(row.get(k) for k in ('references_or_reservations','reservation_overlaps',
        'audit_references_not_original_execution','excluded_weight_path_seed_hints')) for row in e['summary']['families'].values()),'seed evidence contains conflicts')
    if execution:
        current,weights=history_selection(p,traces)
        v.require(current==c['selection'] and weights==c['weight_paths'],'history changed after clearance')
    return c


@dataclass
class Capability:
    path: Path
    plan: dict
    clearance: dict
    exact_evidence: dict
    expires_utc: str
    cursor: int=0

    def before_reset(self,case,journal):
        v.require(ACTIVE.get() is self and self.cursor<len(self.plan['cases']) and case==self.plan['cases'][self.cursor],
            'reset outside current finite confirmation schedule')
        v.require(datetime.now(timezone.utc)<datetime.fromisoformat(self.expires_utc),'confirmation expired before next RNG constructor')
        v.require(v.pointer(self.clearance['path'])==self.clearance and v.pointer(self.exact_evidence['path'])==self.exact_evidence,
            'confirmation clearance/evidence changed before RNG constructor')
        v.require(source_files()==self.plan['source_files'],'source changed before RNG constructor')
        for name,sha in self.plan['prerequisites'].items():v.require(v.file_sha(name)==sha,'fixed asset changed before RNG constructor')
        journal.write(json.dumps(dict(index=self.cursor,case=case,plan_sha256=v.file_sha(self.path),
            clearance_sha256=self.clearance['sha256'],authorized_utc=datetime.now(timezone.utc).isoformat()))+'\n');journal.flush();os.fsync(journal.fileno())
        self.cursor+=1


@contextmanager
def capability(path):
    path,p,_=checked_plan(path);v.require(ACTIVE.get() is None,'confirmation capability cannot nest')
    clearance=checked_clearance(path,p);clearance_ref=v.pointer(path.parent/'seed-ledger/clearance.json')
    v.require(not claim_path().exists(),'candidate confirmation attempt already consumed')
    binding=dict(clearance=clearance_ref,exact_evidence=clearance['exact'],expires_utc=clearance['expires_utc'])
    write_new(claim_path(),dict(schema='warehouse-v629-single-confirmation-claim.v1',plan=v.pointer(path),cases_sha256=v.digest(p['cases']),
        **binding,started_utc=datetime.now(timezone.utc).isoformat(),history_delta_scope='at_first_claim_not_continuous_during_collection',
        owner_pid=os.getpid(),single_use_even_if_interrupted=True,PPO_joint_steps=0,pilot_opened=False))
    folder=path.parent/'collection';folder.mkdir(exist_ok=False)
    write_new(folder/'started.json',dict(plan=v.pointer(path),scheduled_resets=516,claim=v.pointer(claim_path()),**binding))
    cap=Capability(path,p,clearance_ref,clearance['exact'],clearance['expires_utc']);token=ACTIVE.set(cap)
    try:
        yield cap
        v.require(cap.cursor==516,'incomplete confirmation schedule')
        checked_plan(path)
    except BaseException as error:
        write_new(folder/'failed.json',dict(error_type=type(error).__name__,error=str(error),
            preauthorized_cases=cap.cursor,actual_reset_journal=str(folder/'actual-resets.jsonl'),
            actual_reset_count='see_retained_journal_not_inferred_from_cursor',automatic_retry=False,pilot_opened=False))
        raise
    finally:ACTIVE.reset(token)


def collect(path):
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,encode_actor_tree_features
    from backend.training.warehouse_v6_runner import _set_dual_nn_mode
    from backend.training.warehouse_v622_program import masked_probabilities
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from env.warehouse.public_action_features_v617 import validate_study_action_physics
    from env.warehouse.public_observation_v6 import encode_public_observation,public_pre_action_snapshot
    from env.warehouse.pure_nn import AGENTS,action_masks
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.decision_protocol import independent_agent_seed
    from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
    with capability(path) as cap:
        folder=cap.path.parent/'collection';ref=cap.plan['actor']
        actor=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],expected_role_sha256=cap.plan['actor_role_sha256'])
        cases={};outcomes=[];checks=0
        journal=folder/'reset-preauthorization.jsonl'
        with journal.open('x') as auth,capture_stage1_resets(folder/'actual-resets.jsonl') as resets:
            for case in cap.plan['cases']:
                cap.before_reset(case,auth)
                config=participant_study_config(seed=case['scenario_seed']);validate_study_action_physics(config)
                with stage1_reset_phase('pressure' if case['split']=='pressure' else 'program'):
                    env=WarehouseMultiAgentEnv(config);env.reset(seed=case['scenario_seed'])
                _set_dual_nn_mode(env);arrays={k:[] for k in ('x','m','p','sampled')};previous=None
                trace=folder/(case['case_id']+'.jsonl.gz')
                with gzip.open(trace,'xt') as h:
                    while not (env.state.terminated or env.state.truncated):
                        obs=encode_public_observation(env.state);mask=action_masks(env.state);before=public_pre_action_snapshot(env.state)
                        v.require(previous is None or previous==v.digest(before),'disconnected public confirmation trace')
                        actions,distributions=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
                        p=masked_probabilities(actor.logits(obs,mask),mask);indices=[]
                        for i,role in enumerate(AGENTS):
                            seed=independent_agent_seed(base_seed=case['action_seed'],episode_id=int(env.state.episode_id),frame=int(env.state.frame),agent_id=role)
                            index=int(np.random.default_rng(seed).choice(5,p=p[i]));indices.append(index)
                            v.require(actions[role]==ACTIONS[index] and mask[i,index]==1
                                and np.array_equal(np.array(distributions[role].probabilities,np.float32),p[i]),'action not from frozen complete NN')
                            checks+=1
                        frame=int(env.state.frame);_,_,_,_,info=env.step(actions);after=public_pre_action_snapshot(env.state);previous=v.digest(after)
                        v.require(info['proposed_actions']==actions and not info['action_override_agents']
                            and {a['agent_id']:a['last_action'] for a in after['agents']}==actions,'post-NN action override')
                        h.write(json.dumps(dict(case=case,frame=frame,pre_state_public=before,post_state_public=after,
                            proposed_actions=actions,actor_sha256=ref['sha256'],action_override_agents=info['action_override_agents']),separators=(',',':'))+'\n')
                        for key,value in (('x',encode_actor_tree_features(obs,mask)),('m',mask),('p',p),('sampled',indices)):arrays[key].append(value)
                target=folder/(case['case_id']+'.npz')
                with target.open('xb') as h:
                    np.savez_compressed(h,**{key:np.asarray(values,dtype=np.int64 if key=='sampled' else np.float32) for key,values in arrays.items()});h.flush();os.fsync(h.fileno())
                target.chmod(0o444);trace.chmod(0o444)
                cases[case['case_id']]=dict(arrays=v.pointer(target),trace=v.pointer(trace))
                outcomes.append(dict(case=case,complete=True,deliveries=int(env.state.total_deliveries),robot_collisions=int(env.state.robot_collision_events),
                    shutdown=bool(env.state.shutdown_count),physical_frames=len(arrays['x'])))
                if cap.cursor%25==0:print(json.dumps(dict(confirmation_collected=cap.cursor,total=516)),flush=True)
        v.require(len(resets)==516 and [r['seed'] for r in resets]==[c['scenario_seed'] for c in cap.plan['cases']],'actual reset journal differs')
        result=dict(schema='warehouse-v629-fixed-confirmation-collection.v1',plan=v.pointer(cap.path),actor=ref,cases=cases,outcomes=outcomes,
            clearance=cap.clearance,exact_evidence=cap.exact_evidence,claim=v.pointer(claim_path()),
            actual_resets=v.pointer(folder/'actual-resets.jsonl'),preauthorization=v.pointer(journal),actual_reset_count=516,
            NN_action_checks=checks,post_policy_interventions=0,actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
        write_new(folder/'result.json',result)
    write_new(folder/'completed.json',dict(result_sha256=v.file_sha(folder/'result.json')))
    return result
