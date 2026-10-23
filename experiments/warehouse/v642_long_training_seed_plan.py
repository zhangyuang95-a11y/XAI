"""Finite long-pair inventories and exact history clearance, never execution.

Real calibration replay precedes every caller proposal/plan read. Inventories
are explicit inputs, not searched or allocated by this module. Preparation,
indexed history audit and final-stat clearance are separate explicit calls.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
from collections import Counter
import os
import json

from . import v642_long_training_protocol as protocol
from . import v642_training_partitions as old_partitions
from . import v642_training_seed_plan as legacy
from . import seed_metadata_audit_indexed_v642 as metadata
from . import seed_triage_indexed_v642 as triage
from . import seed_interval_index_v642 as interval_source
from . import exact_seed_audit as exact
from . import v642_formal_training_admission as formal
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training.warehouse_v625_actor_training import PPOConfig,ReferenceTrainingBuffer
from backend.training.warehouse_v629_environment_replay import EnvironmentRolloutReplay
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
from scripts.run_warehouse_v625 import write_new

ROOT=protocol.ROOT
SOURCE_ROOT=Path(__file__).resolve().parents[2]
OUTPUT_ROOT=ROOT/'output/warehouse_v642/long-training-seed-plans'
VERSION='warehouse-v642-long-pair-seed-plan.v1'
PROPOSAL_VERSION='warehouse-v642-long-pair-seed-proposal.v1'
VARIANTS=('raw','regularized')
HISTORY_NAMES=legacy.confirmation.HISTORY_NAMES
REGISTRY_REF=deepcopy(legacy.confirmation.INVENTORY_REF)
EXECUTION_LEASE_DAYS_BY_BUDGET={200000:7,300000:10,400000:14}
RESTART_VERSION='warehouse-v642-long-common-start-restart.v1'


def source_files():
    files=protocol.source_files()
    files.update(formal.source_files())  # Flat hashes, never development.source_files recursion.
    names=('experiments/warehouse/v642_long_training_seed_plan.py',
           'tests/test_warehouse_v642_long_training_seed_plan.py','docs/warehouse_v642_long_training_seed_plan.md',
           'experiments/warehouse/seed_metadata_audit_indexed_v642.py',
           'experiments/warehouse/seed_triage_indexed_v642.py','experiments/warehouse/seed_interval_index_v642.py',
           'experiments/warehouse/exact_seed_audit.py')
    files.update({name:file_sha(SOURCE_ROOT/name) for name in names})
    return dict(sorted(files.items()))


def _asset(ref,*,readonly=True):
    path=Path(ref['path'])
    require(pointer(path)==ref and not path.is_symlink() and (not readonly or not path.stat().st_mode&0o222),
            'long-pair input missing, changed, symlinked or writable')
    return path


def _seed(value):
    require(type(value) is int and 0<=value<2**63 and not is_sealed_scenario_seed(value),
            'invalid or sealed original seed argument')
    return value


def validate_inventories(training,evidence,*,config,initialization_seed,optimizer_seed):
    require(type(config) is PPOConfig and config.joint_step_budget in protocol.BUDGETS,
            'long-pair inventories require a registered long budget')
    init,opt=_seed(initialization_seed),_seed(optimizer_seed)
    require(init!=opt,'initialization and optimizer streams overlap')
    require(set(training)=={'schema','cases'} and training['schema']==old_partitions.TRAIN_VERSION
            and type(training['cases']) is list and 8<=len(training['cases'])<=config.joint_step_budget,
            'invalid finite eight-slot training inventory')
    counts=protocol.Protocol().payload()['program_source_episode_counts']
    observed=old_partitions.observed
    require(set(evidence)=={'schema','scope','expected_counts','cases'}
            and evidence['schema']==observed.INVENTORY_VERSION and evidence['scope']==observed.SCOPE
            and evidence['expected_counts']==counts and all(type(n) is int for n in evidence['expected_counts'].values())
            and type(evidence['cases']) is list,
            'program inventory must contain the fixed 1650 disjoint evidence episodes')
    ids=set();scenes=set();values={init,opt};slots=Counter();actual=Counter()
    def accept(episode,base,scenario,action,task):
        require(type(episode) is str and episode and episode not in ids,'episode identity reused across partitions')
        require(type(task) is int and task in (1,2,3),'unknown task')
        base,scenario,action=_seed(base),_seed(scenario),_seed(action)
        require(scenario==10*base+task and base not in scenes,'whole scene reused or task expansion differs')
        arguments={base,scenario,action}
        require(len(arguments)==3 and not arguments&values,'scene/action/init/optimizer arguments overlap')
        ids.add(episode);scenes.add(base);values.update(arguments)
    for case in training['cases']:
        require(set(case)=={'episode_id','scenario_seed','task','action_seed','slot','partition'}
                and case['partition']=='train' and type(case['slot']) is int and 0<=case['slot']<8,
                'training inventory contains an unknown partition or slot')
        scenario=_seed(case['scenario_seed'])
        accept(case['episode_id'],scenario//10,scenario,case['action_seed'],case['task']);slots[case['slot']]+=1
    per_slot=config.joint_step_budget//8
    require(set(slots)==set(range(8))
            and min(slots.values())>=(per_slot+config.episode_horizon-1)//config.episode_horizon
            and max(slots.values())<=per_slot,'finite per-slot training stream is too short or exceeds the one-frame reset ceiling')
    for case in evidence['cases']:
        require(set(case)=={'episode','base_seed','scenario_seed','task','action_seed','split'}
                and case['split'] in counts,'unknown program evidence split')
        accept(case['episode'],case['base_seed'],case['scenario_seed'],case['action_seed'],case['task'])
        actual[case['split']]+=1
    require(dict(actual)==counts,'program evidence inventory count changed')
    return dict(training_episodes=len(training['cases']),episodes_per_slot={str(k):v for k,v in sorted(slots.items())},
        evidence_counts=dict(actual),complete_scene_values=sorted(scenes),all_seed_arguments=sorted(values),episode_ids=sorted(ids),
        raw_regularized_share_identical_training_and_evidence_inventories=True,
        exhausted_finite_inventory_fails_closed=True,history_freshness_verified=False,
        environment_authority=False,optimizer_authority=False)


def _known_spent(registration,readiness):
    values,ids=old_partitions.known_spent_arguments(readiness)
    precheck=bound(registration.value['calibration_precheck'])
    quality_plan=bound(precheck['plan'])
    for case in quality_plan['cases']:
        values.update(case[k] for k in ('base','scenario_seed','action_seed'));ids.add(case['case_id'])
    return values,ids


def declared_calls(value,training,evidence):
    uses=[];reservations=[];tag=VERSION+':'+value['phase']+':'+value['stream_id']+':'+str(value['joint_step_budget'])
    def use(kind,purpose,argument,seed,group,reuse,draw=None):
        # Only the common initialization/optimizer start intentionally repeats
        # across budgets. Scene/action reservations retain their budget tag.
        reuse_prefix=(VERSION+':'+value['phase']+':'+value['stream_id']) if reuse in ('paired-initialization','paired-optimizer') else tag
        row=dict(kind=kind,purpose=purpose,argument=argument,value=seed,group=group,reuse_tag=reuse_prefix+':'+reuse)
        if draw is not None:row['consumed_as_random_draw']=draw
        uses.append(row)
    for variant in VARIANTS:
        use('init','v642_long_imported_actors_new_critic_initialization','torch.default_generator.manual_seed',
            value['initialization_seed'],variant+':initialization','paired-initialization')
        use('action_rng','v642_long_PPO_shuffle_and_reference_mix','np.random.default_rng',
            value['optimizer_seed'],variant+':optimizer','paired-optimizer',True)
        rows=[dict(episode=c['episode_id'],base=c['scenario_seed']//10,scenario_seed=c['scenario_seed'],task=c['task'],
                   action_seed=c['action_seed'],scope='train',slot=c['slot']) for c in training['cases']]
        rows.extend(dict(episode=c['episode'],base=c['base_seed'],scenario_seed=c['scenario_seed'],task=c['task'],
                         action_seed=c['action_seed'],scope='program_'+c['split'],slot=None) for c in evidence['cases'])
        for case in rows:
            group=variant+':'+case['episode'];purpose='v642_long_'+case['scope']
            for kind,argument,key in (('scenario','task_expansion_base','base'),('env','random.Random(config.seed)','scenario_seed'),
                ('env','reset(seed)','scenario_seed'),('action_rng','independent_agent_seed(base_seed,episode_id=1,frame,agent_id)','action_seed')):
                use(kind,purpose,argument,case[key],group,case['episode']+':'+key,True if kind=='action_rng' else None)
            reservations.append(dict(variant=variant,**case))
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id=tag,uses=uses,
        planned_episode_construction='fresh_constructor_then_explicit_reset_for_each_used_case',planned_physical_episode_id=1,
        case_reservations_sha256=digest(reservations),paired_case_reuse_not_independent_evidence=True,
        inventory_reservation_is_not_actual_reset=True,authorization_granted=False),reservations


@dataclass(frozen=True)
class LongTrainingSeedPlan:
    ref:dict
    value:dict
    registration:object
    readiness:object
    refs:tuple
    formal_authorization:object=None

    @property
    def proposal(self):return self

    @property
    def config(self):return PPOConfig(joint_step_budget=self.value['joint_step_budget'])

    def verify(self):
        if self.formal_authorization is not None:
            require(type(self.formal_authorization) is formal.FormalTrainingAdmission,
                    'formal plan requires its actual independently read admission token')
            self.formal_authorization.verify()
        self.registration.verify();self.readiness.verify();p=self.value;folder=Path(self.ref['path']).parent
        require(pointer(self.ref['path'])==self.ref and bound(self.ref)==p and p['version']==VERSION
                and p['source_files']==source_files() and p['source_sha256']==digest(source_files())
                and p['protocol_registration_sha256']==self.registration.payload_sha256
                and p['calibration_precheck_ref']==self.registration.value['calibration_precheck']
                and p['readiness_sha256']==self.readiness.payload_sha256 and p['config']==asdict(self.config)
                and p['training_authority'] is p['sealed_authority'] is p['pilot_opened'] is False,
                'long-pair plan, prerequisite, configuration or source changed')
        require(bound(pointer(folder/'completed.json'))==dict(plan_sha256=self.ref['sha256']), 'long-pair plan seal changed')
        for ref in self.refs:_asset(ref,readonly=False)
        proposal=bound(p['proposal_ref'])
        if self.formal_authorization is None:
            require(p['phase']=='development' and p.get('formal_admission') is None,
                    'ordinary development reader cannot authorize a formal plan')
            expected=dict(version=PROPOSAL_VERSION,**{key:p[key] for key in
                ('phase','stream_id','joint_step_budget','initialization_seed','optimizer_seed',
                 'train_case_inventory','program_evidence_inventory')},
                 **({'restart_declaration':p['restart_declaration']} if p['restart_declaration'] is not None else {}))
        else:
            self.formal_authorization.validate_proposal(proposal)
            require(p['formal_admission']==self.formal_authorization.value
                    and p['development_budget_lock']==self.formal_authorization.budget_lock_ref
                    and p['restart_declaration'] is p['restart_proof'] is None,
                    'formal plan changed its lock or attempted a restart')
            expected=dict(version=formal.PROPOSAL_VERSION,development_budget_lock=self.formal_authorization.budget_lock_ref,
                **{key:p[key] for key in ('phase','stream_id','joint_step_budget','initialization_seed','optimizer_seed',
                   'train_case_inventory','program_evidence_inventory')})
        require(proposal==expected,'long plan differs from its immutable caller proposal')
        require(p['execution_lease_days']==EXECUTION_LEASE_DAYS_BY_BUDGET[p['joint_step_budget']]
                and p['seed_clearance_start_window_hours']==24,'long finite execution lease or start window changed')
        require(p['phase'] in ('development','formal') and p['stream_id'] in
                (protocol.DEVELOPMENT_STREAMS if p['phase']=='development' else protocol.FORMAL_STREAMS)
                and p['execution_output']==str(folder/'training')
                and p['global_claim_path']==str(OUTPUT_ROOT/'claims'/f'{p["phase"]}-{p["stream_id"]}-{p["joint_step_budget"]}.json'),
                'long-pair phase, output or global single-use claim differs')
        for key in ('proposal_ref','train_case_inventory','program_evidence_inventory'):_asset(p[key])
        t,e=bound(p['train_case_inventory']),bound(p['program_evidence_inventory'])
        checked=validate_inventories(t,e,config=self.config,initialization_seed=p['initialization_seed'],optimizer_seed=p['optimizer_seed'])
        require(p['restart_proof']==_verified_restart(p),'long registered restart proof changed')
        spent,ids=_known_spent(self.registration,self.readiness)
        require(not spent&set(checked['all_seed_arguments']) and not ids&set(checked['episode_ids']),
                'long-pair plan overlaps spent development, confirmation or calibration-quality scenes')
        declaration,cases=declared_calls(p,t,e)
        require(p['checked']==checked and bound(p['exact_plan'])==declaration and bound(p['case_reservations'])==cases
                and p['maximum_environment_constructors']==p['maximum_explicit_resets']==len(cases)
                and p['maximum_original_reset_invocations']==len(cases)
                and p['maximum_original_environment_seed_operations']==2*len(cases)
                and p['unconsumed_cursor']==dict(train={v:{str(i):0 for i in range(8)} for v in VARIANTS},program={v:0 for v in VARIANTS})
                and p['asset_manifest_sha256']==digest(self.refs),'long-pair inventory, arguments, cursors or ceiling changed')
        for name,sha in p['source_files'].items():
            path=folder/'source'/p['source_sha256']/name
            require(file_sha(path)==sha and not path.stat().st_mode&0o222,'long-pair source snapshot changed')
        return self


def prepare(*,calibration_precheck_ref,reference_directory,proposal_ref,run_id):
    return _prepare(calibration_precheck_ref=calibration_precheck_ref,reference_directory=reference_directory,
        proposal_ref=proposal_ref,run_id=run_id,formal_authorization=None)


def prepare_formal(*,budget_lock_ref,calibration_precheck_ref,reference_directory,proposal_ref,run_id):
    authority=formal.read_admission(budget_lock_ref)  # FIRST, before any other caller input.
    return _prepare(calibration_precheck_ref=calibration_precheck_ref,reference_directory=reference_directory,
        proposal_ref=proposal_ref,run_id=run_id,formal_authorization=authority)


def _prepare(*,calibration_precheck_ref,reference_directory,proposal_ref,run_id,formal_authorization):
    if formal_authorization is not None:formal_authorization.verify()
    registration=protocol.register_preparation(calibration_precheck_ref)  # FIRST, before caller proposal/output/seed reads.
    registration.verify();readiness=old_partitions.admission.load_readiness(reference_directory);readiness.verify()
    _asset(proposal_ref);p=bound(proposal_ref)
    keys={'version','phase','stream_id','joint_step_budget','initialization_seed','optimizer_seed',
          'train_case_inventory','program_evidence_inventory'}
    if formal_authorization is None:
        require(set(p) in (keys,keys|{'restart_declaration'}) and p['version']==PROPOSAL_VERSION and p['phase']=='development',
                'invalid explicit development long-pair proposal; formal needs its explicit budget-lock entry')
    else:formal_authorization.validate_proposal(p)
    require(p['phase'] in ('development','formal') and p['joint_step_budget'] in protocol.BUDGETS
            and p['stream_id'] in (protocol.DEVELOPMENT_STREAMS if p['phase']=='development' else protocol.FORMAL_STREAMS),
            'unregistered long phase, stream or budget')
    config=PPOConfig(joint_step_budget=p['joint_step_budget'])
    for key in ('train_case_inventory','program_evidence_inventory'):_asset(p[key])
    training,evidence=bound(p['train_case_inventory']),bound(p['program_evidence_inventory'])
    checked=validate_inventories(training,evidence,config=config,initialization_seed=p['initialization_seed'],optimizer_seed=p['optimizer_seed'])
    restart=_verified_restart(p)
    if restart is not None:
        fresh=set(checked['all_seed_arguments'])-{p['initialization_seed'],p['optimizer_seed']}
        require(not fresh&set(restart['spent_seed_arguments']) and not set(checked['episode_ids'])&set(restart['spent_episode_ids']),
                'a common-start restart cannot reuse old training/program scenes or action arguments')
    spent,ids=_known_spent(registration,readiness)
    require(not spent&set(checked['all_seed_arguments']) and not ids&set(checked['episode_ids']),
            'long-pair proposal overlaps spent development, confirmation or calibration-quality scenes')
    require(readiness.value['actor']['sha256']==protocol.REFERENCE_SHA256,
            'long continuation must preserve the registered complete v620 reference')
    run=legacy.confirmation.run_name(run_id);sources=source_files();source_sha=digest(sources)
    declaration,cases=declared_calls(p,training,evidence)
    refs=tuple(deepcopy([*registration.refs,*readiness.refs,proposal_ref,p['train_case_inventory'],p['program_evidence_inventory'],
                        *([] if restart is None else [restart['declaration'],*restart['allowed_metadata_refs']]),
                        *([] if formal_authorization is None else formal_authorization.refs)]))
    for ref in refs:_asset(ref,readonly=False)
    global_claim=OUTPUT_ROOT/'claims'/f'{p["phase"]}-{p["stream_id"]}-{p["joint_step_budget"]}.json'
    require(not global_claim.exists(),'this phase/stream/budget was already consumed even if interrupted')
    output=OUTPUT_ROOT/run;output.mkdir(parents=True,exist_ok=False)
    for name,sha in sources.items():
        raw=(SOURCE_ROOT/name).read_bytes();require(file_sha(SOURCE_ROOT/name)==sha,'long source changed during snapshot')
        path=output/'source'/source_sha/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        path.chmod(0o444)
    for name,value in (('exact-plan.json',declaration),('case-reservations.json',cases)):
        write_new(output/name,value);(output/name).chmod(0o444)
    exact.read_exact_plan(output/'exact-plan.json')
    value=dict(deepcopy(p),version=VERSION,run_id=run,proposal_ref=deepcopy(proposal_ref),config=asdict(config),
        restart_declaration=p.get('restart_declaration'),restart_proof=restart,
        execution_lease_days=EXECUTION_LEASE_DAYS_BY_BUDGET[p['joint_step_budget']],seed_clearance_start_window_hours=24,
        calibration_precheck_ref=deepcopy(calibration_precheck_ref),protocol_registration_sha256=registration.payload_sha256,
        readiness_sha256=readiness.payload_sha256,source_files=sources,source_sha256=source_sha,
        reference_actor=deepcopy(readiness.value['actor']),reference_role_sha256=dict(protocol.REFERENCE_ROLES),
        paired_initialization=dict(complete_reference_Actor_shared=True,new_critic_initialization_seed=p['initialization_seed'],
            empty_Actor_and_critic_optimizers_required=True,optimizer_random_stream_seed=p['optimizer_seed'],
            actual_initial_critic_and_optimizer_hashes_must_be_recorded_by_owner=True),
        checked=checked,asset_manifest_sha256=digest(refs),exact_plan=pointer(output/'exact-plan.json'),
        case_reservations=pointer(output/'case-reservations.json'),maximum_environment_constructors=len(cases),
        execution_output=str(output/'training'),
        global_claim_path=str(OUTPUT_ROOT/'claims'/f'{p["phase"]}-{p["stream_id"]}-{p["joint_step_budget"]}.json'),
        maximum_explicit_resets=len(cases),new_critic_initializations=2,optimizer_RNG_constructors=2,variants=list(VARIANTS),
        maximum_original_reset_invocations=len(cases),maximum_original_environment_seed_operations=2*len(cases),
        unconsumed_cursor=dict(train={v:{str(i):0 for i in range(8)} for v in VARIANTS},program={v:0 for v in VARIANTS}),
        history_freshness_verified=False,phase_budget_admission_and_single_use_owner_still_required=True,
        training_authority=False,sealed_authority=False,new_environment_resets=0,Actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
    if formal_authorization is not None:value['formal_admission']=deepcopy(formal_authorization.value)
    require(source_files()==sources,'long source changed while preparing plan')
    write_new(output/'plan.json',value);(output/'plan.json').chmod(0o444);ref=pointer(output/'plan.json')
    write_new(output/'completed.json',dict(plan_sha256=ref['sha256']));(output/'completed.json').chmod(0o444)
    return LongTrainingSeedPlan(ref,value,registration,readiness,refs,formal_authorization).verify()


def check(plan_ref,*,calibration_precheck_ref,reference_directory):
    return _check(plan_ref,calibration_precheck_ref=calibration_precheck_ref,reference_directory=reference_directory,
        formal_authorization=None)


def check_formal(plan_ref,*,budget_lock_ref,calibration_precheck_ref,reference_directory):
    authority=formal.read_admission(budget_lock_ref)  # FIRST, even before hashing/reading the plan.
    return _check(plan_ref,calibration_precheck_ref=calibration_precheck_ref,reference_directory=reference_directory,
        formal_authorization=authority)


def _check(plan_ref,*,calibration_precheck_ref,reference_directory,formal_authorization):
    if formal_authorization is not None:formal_authorization.verify()
    registration=protocol.register_preparation(calibration_precheck_ref)  # FIRST; never infer permission from unadmitted plan.
    registration.verify();readiness=old_partitions.admission.load_readiness(reference_directory);readiness.verify()
    _asset(plan_ref);p=bound(plan_ref)
    require(Path(plan_ref['path']).resolve()==OUTPUT_ROOT/legacy.confirmation.run_name(p['run_id'])/'plan.json',
            'long plan is outside its isolated namespace')
    refs=tuple(deepcopy([*registration.refs,*readiness.refs,p['proposal_ref'],p['train_case_inventory'],p['program_evidence_inventory'],
        *([] if p['restart_proof'] is None else [p['restart_proof']['declaration'],*p['restart_proof']['allowed_metadata_refs']]),
        *([] if formal_authorization is None else formal_authorization.refs)]))
    return LongTrainingSeedPlan(plan_ref,p,registration,readiness,refs,formal_authorization).verify()


def exclusions(plan):
    return [Path(plan.ref['path']).parent,*(Path(plan.value[k]['path']) for k in
            ('proposal_ref','train_case_inventory','program_evidence_inventory')),Path(plan.value['global_claim_path']),
            *([] if plan.value['restart_declaration'] is None else [Path(plan.value['restart_declaration']['path'])])]


def _verified_restart(p):
    ref=p.get('restart_declaration')
    if ref is None:
        require(p['phase']!='development' or p['joint_step_budget']==protocol.BUDGETS[0],
                'later development budgets need an independently verified common-start restart declaration')
        return None
    _asset(ref);d=bound(ref);budget=p['joint_step_budget']
    require(set(d)=={'version','phase','stream_id','from_budget','to_budget','initialization_seed','optimizer_seed','previous_assets'}
            and d['version']==RESTART_VERSION and d['phase']==p['phase']=='development'
            and d['stream_id']==p['stream_id'] and budget in protocol.BUDGETS[1:]
            and d['to_budget']==budget and d['from_budget']==protocol.BUDGETS[protocol.BUDGETS.index(budget)-1]
            and d['initialization_seed']==p['initialization_seed'] and d['optimizer_seed']==p['optimizer_seed'],
            'restart may only repeat the same development stream common start at the next registered budget')
    # Lazy import avoids the seed-plan -> owner -> assets -> seed-plan cycle.
    from . import v642_long_training_assets as assets
    parent=bound(d['previous_assets'])
    context=assets.finalization_context(d['previous_assets'],parent['reference_directory'])
    plan=context['plan'];prior=plan.value
    require(prior['phase']==p['phase'] and prior['stream_id']==p['stream_id'] and plan.config.joint_step_budget==d['from_budget']
            and prior['initialization_seed']==p['initialization_seed'] and prior['optimizer_seed']==p['optimizer_seed']
            and prior['reference_actor']['sha256']==protocol.REFERENCE_SHA256,
            'restart parent does not prove the same actual common Actor/critic/optimizer start')
    identity_keys=('roles','critic_sha256','full_initial_training_state_sha256','actor_optimizer_sha256',
                   'critic_optimizer_sha256','initial_joint_steps','initial_actor_updates','initial_critic_updates')
    starts=[{key:e[key] for key in identity_keys} for e in context['events'] if e['kind']=='initialized']
    require(len(starts)==2 and starts[0]==starts[1],'restart parent has inconsistent actual pair initialization')
    refs=[d['previous_assets'],plan.ref,prior['proposal_ref'],*context['assets']['inputs']]
    clearance=checked_clearance(plan,execution=False)['value'];prior_exact=bound(clearance['exact'])
    prior_inventory=bound(prior_exact['bounded_or_triaged_inventory'])
    audit_refs=[clearance['exact'],prior_exact['bounded_or_triaged_inventory'],prior_inventory['bounded_inventory'],prior_inventory['proposal']]
    if prior_exact.get('context_annotations'):audit_refs.append(prior_exact['context_annotations'])
    if clearance['restart_context'] is not None:audit_refs.append(clearance['restart_context'])
    refs.extend(audit_refs)
    spent=set(prior['checked']['all_seed_arguments']);ids=set(prior['checked']['episode_ids'])
    if prior['restart_proof'] is not None:
        refs.extend(prior['restart_proof']['allowed_metadata_refs'])
        audit_refs.extend(prior['restart_proof']['reference_only_audit_refs'])
        spent.update(prior['restart_proof']['spent_seed_arguments']);ids.update(prior['restart_proof']['spent_episode_ids'])
    unique={r['path']:r for r in refs}
    for item in unique.values():_asset(item)
    return dict(version=RESTART_VERSION,declaration=ref,previous_assets=d['previous_assets'],previous_plan=plan.ref,
        expected_initial_identity=starts[0],allowed_metadata_refs=[unique[k] for k in sorted(unique)],
        reference_only_audit_refs=sorted({r['path']:r for r in audit_refs}.values(),key=lambda r:r['path']),
        initialization_seed=p['initialization_seed'],optimizer_seed=p['optimizer_seed'],
        spent_seed_arguments=sorted(spent),spent_episode_ids=sorted(ids),
        only_common_initialization_and_optimizer_arguments_reused=True,scene_and_action_arguments_must_remain_fresh=True)


def _restart_context(evidence,proof):
    """Narrow new source-context rule; retain the untouched original audit."""
    if proof is None:return evidence,[]
    allowed={ref['path']:ref for ref in proof['allowed_metadata_refs']};result=deepcopy(evidence);removed=[];documents={}
    reference_only={ref['path'] for ref in proof['reference_only_audit_refs']}
    init,opt=proof['initialization_seed'],proof['optimizer_seed']
    for row in result['inventory']:
        ref=allowed.get(row['path'])
        if ref is None or row.get('sha256')!=ref['sha256']:continue
        path=Path(row['path']);_asset(ref)
        if path.suffix not in ('.json','.jsonl'):continue
        documents[row['path']]=([json.loads(line) for line in path.read_text().splitlines()] if path.suffix=='.jsonl' else json.loads(path.read_text()))
        kept=[]
        for match in row.get('detail',{}).get('candidate_matches',[]):
            route=match['path'];value=match['value'];body=documents[row['path']];parent=None;valid=False
            try:
                for key in route:parent=body;body=body[key]
                key=route[-1] if route else None
                valid=(body==value and ((row['path'] in reference_only and value in (init,opt))
                    or (type(key) is int and len(route)>=2 and route[-2] in ('all_seed_arguments','spent_seed_arguments') and value in (init,opt))
                    or (value==init and key in ('initialization_seed','pair_seed','new_critic_initialization_seed'))
                    or (value==opt and key in ('optimizer_seed','optimizer_random_stream_seed'))
                    or (value==opt and key=='seed' and type(parent) is dict and parent.get('kind')=='optimizer_RNG_predebit')
                    or (key=='value' and type(parent) is dict and (
                        (value==init and parent.get('kind')=='init' and parent.get('purpose')=='v642_long_imported_actors_new_critic_initialization')
                        or (value==opt and parent.get('kind')=='action_rng' and parent.get('purpose')=='v642_long_PPO_shuffle_and_reference_mix')))))
            except (KeyError,IndexError,TypeError):valid=False
            if valid:removed.append(dict(file=ref,path=route,value=value,reason='source_bound_same_stream_common_start_restart'))
            else:kept.append(match)
        if 'detail' in row:row['detail']['candidate_matches']=kept
    result['summary']=metadata.summarize(result['inventory'],result['families'],selection_stats=result.get('selection_stats'),
                                         missing_roots=result.get('missing_roots',()))
    result['initialization_and_optimizer_are_registered_restarts_not_fresh']=True
    return result,removed


def history_roots():return [ROOT.parent/name for name in HISTORY_NAMES]


def _history(plan,traces):
    stats={};selected=[]
    for path in metadata.candidates(history_roots(),exclusions(plan),registered_trace_paths=traces,selection_stats=stats):
        stat=path.stat();selected.append(dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return selected,stats['excluded_weights_name_stat_only']


def _templates(rows,*,current):
    recorded=legacy.template_bindings(rows,check_current=False)
    if current:
        actual={}
        for root in history_roots():
            path=root/'backend/training/seed_ledger.py'
            if path.is_file():
                raw=path.read_bytes()
                if b'500_004' in raw and b'10_500_204' in raw:actual[str(path.resolve())]=file_sha(path)
        require(actual==recorded,'long-pair historical source reservations changed')
    return recorded


def audit(plan):
    plan.verify();folder=Path(plan.ref['path']).parent/'seed-ledger';folder.mkdir(exist_ok=False)
    _asset(REGISTRY_REF);traces=bound(REGISTRY_REF)['registered_trace_paths']
    families=exact.compile_families(bound(plan.value['exact_plan']))
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=bound(plan.value['exact_plan'])['candidate_id'],
        families=[{k:r[k] for k in ('name','kind','values','query_only')} for r in families],
        exclude_paths=list(map(str,exclusions(plan))),registered_trace_paths=traces)
    write_new(folder/'proposal.json',proposal)
    metadata.audit(folder/'proposal.json',history_roots(),folder/'bounded')
    triage.triage(folder/'bounded/inventory.json',folder/'proposal.json',folder/'triaged')
    result=exact.assess(plan.value['exact_plan']['path'],folder/'triaged/result.json',folder/'exact',context_source_root=ROOT)
    reviewed,exceptions=_restart_context(result,plan.value['restart_proof'])
    legacy.conflict_free(reviewed);selected,weights=_history(plan,traces)
    context_ref=None
    if plan.value['restart_proof'] is not None:
        context=dict(version=RESTART_VERSION,exact=pointer(folder/'exact/result.json'),
            proof_sha256=digest(plan.value['restart_proof']),source_sha256=file_sha(__file__),intentional_restart_matches=exceptions,
            reviewed_inventory_sha256=digest(reviewed['inventory']),reviewed_summary=reviewed['summary'],
            reused_arguments_are_not_claimed_fresh=True,environment_resets=0,training_authority=False)
        write_new(folder/'restart-context.json',context);(folder/'restart-context.json').chmod(0o444)
        context_ref=pointer(folder/'restart-context.json')
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory']
             if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selected},'history changed during long-pair seed audit')
    templates=_templates(result['inventory'],current=True);plan.verify();now=datetime.now(timezone.utc)
    value=dict(version=VERSION,plan=plan.ref,registry=REGISTRY_REF,registered_traces=traces,
        exact=pointer(folder/'exact/result.json'),selection=selected,weight_paths=weights,templates=templates,
        restart_context=context_ref,
        issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),
        approved_case_reservations_sha256=plan.value['case_reservations']['sha256'],
        maximum_finite_reset_reservations=plan.value['maximum_explicit_resets'],actual_resets=0,
        training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'clearance.json',value);(folder/'clearance.json').chmod(0o444)
    for path in folder.rglob('*'):
        require(not path.is_symlink(),'long seed audit output contains a symlink')
        if path.is_file():path.chmod(0o444)
    return checked_clearance(plan)


def checked_clearance(plan,*,execution=True):
    plan.verify();folder=Path(plan.ref['path']).parent/'seed-ledger';ref=pointer(folder/'clearance.json');c=bound(ref);e=bound(c['exact'])
    issued=datetime.fromisoformat(c['issued_utc']);expires=datetime.fromisoformat(c['expires_utc'])
    require(c['version']==VERSION and c['plan']==plan.ref and c['registry']==REGISTRY_REF
            and pointer(REGISTRY_REF['path'])==REGISTRY_REF and issued.tzinfo and expires.tzinfo
            and expires-issued==timedelta(hours=24) and (not execution or issued<=datetime.now(timezone.utc)<expires)
            and c['approved_case_reservations_sha256']==plan.value['case_reservations']['sha256']
            and c['maximum_finite_reset_reservations']==plan.value['maximum_explicit_resets'] and c['actual_resets']==0
            and c['training_authority'] is c['sealed_authority'] is c['pilot_opened'] is False,
            'long-pair clearance scope, lifetime or finite ceiling differs')
    require(c['exact']==pointer(folder/'exact/result.json') and e['exact_plan']==plan.value['exact_plan']
            and e['exact_auditor_sha256']==file_sha(exact.__file__),'long-pair exact source or input differs')
    inv=bound(e['bounded_or_triaged_inventory']);families=exact.compile_families(bound(plan.value['exact_plan']))
    projection=exact.project_inventory(inv,families)
    if e.get('context_annotations'):projection=exact.apply_context(projection,bound(e['context_annotations']))
    require(e['summary']==projection['summary'] and digest(e['inventory'])==digest(projection['inventory']),
            'long-pair exact/context evidence does not replay')
    roots=sorted(str(root.resolve()) for root in history_roots());traces=bound(REGISTRY_REF)['registered_trace_paths']
    require(sorted(e['scan_roots'])==sorted(inv['scan_roots'])==roots and not inv['missing_roots']
            and len(roots)==4 and all(Path(root).is_dir() for root in roots)
            and e['selection_scope']==inv['selection_scope']==metadata.SELECTION_SCOPE
            and e['read_model_weight_files'] is inv['read_model_weight_files'] is False
            and e['scope_exclusions']==inv['scope_exclusions']
            and sorted(inv['scope_exclusions'])==sorted(map(str,[folder/'proposal.json',folder/'bounded',*exclusions(plan)]))
            and inv['auditor_sha256']==file_sha(metadata.__file__) and inv['triage_driver_sha256']==file_sha(triage.__file__)
            and inv['interval_index_source_sha256']==file_sha(interval_source.__file__)
            and c['registered_traces']==e['registered_trace_paths']==inv['registered_trace_paths']==traces,
            'long-pair four-root inventory, registry, exclusions or indexed auditor source changed')
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in e['inventory']
             if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in c['selection']}
            and len(audited)==len(c['selection']),'long-pair clearance metadata selection differs')
    reviewed,exceptions=_restart_context(e,plan.value['restart_proof'])
    if plan.value['restart_proof'] is None:
        require(c['restart_context'] is None,'unexpected long restart exception')
    else:
        require(c['restart_context']==pointer(folder/'restart-context.json')
                and bound(c['restart_context'])==dict(version=RESTART_VERSION,exact=c['exact'],
                    proof_sha256=digest(plan.value['restart_proof']),source_sha256=file_sha(__file__),
                    intentional_restart_matches=exceptions,reviewed_inventory_sha256=digest(reviewed['inventory']),
                    reviewed_summary=reviewed['summary'],reused_arguments_are_not_claimed_fresh=True,
                    environment_resets=0,training_authority=False),'long restart source-context does not independently replay')
    legacy.conflict_free(reviewed)
    if execution:
        selected,weights=_history(plan,traces)
        require(selected==c['selection'] and weights==c['weight_paths'],'history changed after long-pair clearance')
        require(_templates(e['inventory'],current=True)==c['templates'],'source reservations changed after long-pair clearance')
    return dict(clearance=ref,value=c)


def check_reference_buffer(proposal,buffer):
    proposal.verify();ref=proposal.readiness.value['reference']['buffer']
    require(isinstance(buffer,ReferenceTrainingBuffer) and buffer.source_path is not None
            and pointer(buffer.source_path)==ref and buffer.source_sha256==ref['sha256'],
            'long-pair reference KL requires the original audited common-fit buffer')
    original=ReferenceTrainingBuffer.load(ref['path'],ref['sha256'],protocol.REFERENCE_SHA256)
    import numpy as np
    require(np.array_equal(buffer.encoded,original.encoded) and np.array_equal(buffer.masks,original.masks)
            and buffer.episode_ids==original.episode_ids,'long-pair reference buffer changed')
    return buffer


def assemble_training_batch(proposal,reader,trace_ref,state):
    proposal.verify()
    require(type(reader) is EnvironmentRolloutReplay and reader.inventory_ref==proposal.value['train_case_inventory']
            and asdict(reader.config)==proposal.value['config'] and state.initialization_seed==proposal.value['initialization_seed']
            and state.reference_sha256==protocol.REFERENCE_SHA256,'long-pair replay reader or NN initialization differs')
    owner=bound(reader.owner_ref)
    require(owner['training_partition_proposal']==proposal.ref
            and owner['program_evidence_inventory']==proposal.value['program_evidence_inventory']
            and owner['train_case_inventory']==proposal.value['train_case_inventory'],'long owner does not bind the same inventories')
    allowed={c['episode_id']:c for c in bound(reader.inventory_ref)['cases']}
    require(all(allowed.get(e['case']['episode_id'])==e['case'] for e in bound(trace_ref)['events']),
            'program evidence or foreign cases cannot enter long-pair PPO')
    batch,audit=reader.assemble(trace_ref,state)
    require(all(row['episode_id'] in allowed for key in ('rows','complete_cost_rows') for row in batch[key]),
            'non-training scenes reached long-pair reward/cost buffers')
    proposal.verify()
    return batch,dict(physics_and_NN_replay=audit,partition_proposal=proposal.ref,
                      program_evidence_frames_in_gradient_buffers=0,training_authority=False)
