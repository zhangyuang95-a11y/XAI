"""Independent long-development readback and direct tensor export.

Requires actual supervised termination, finite-source/reset/update histories,
current-NN program evidence and full checkpoints. This is numerical training
completion, never task-quality, budget-lock, formal or release acceptance.
No training model/optimizer is constructed or updated. Recorded trajectories
and seeded action draws may be replayed; no new scene is generated.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
import io,json,sys

import numpy as np
import torch

from . import v642_long_training_owner as owner
from . import v642_recovered_assets as shared
from .v642_update_receipt_audit import audit_updates
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training import warehouse_v642_evidence_provider as provider
from backend.training import warehouse_v642_long_targets as materialization
from backend.training import warehouse_v642_long_evidence as storage_callback
from backend.training import warehouse_v629_environment_replay as physics
from backend.training import warehouse_v629_ppo_update as ppo
from backend.training.warehouse_v642_checkpoint_export import export_checkpoint_actor
from backend.training.warehouse_v622_actor_training import CHECKPOINT_SCHEMA
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,MODEL_VERSION,_tensor_names,role_sha256
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from scripts.run_warehouse_v625 import write_new

ROOT=owner.ROOT
VERSION='warehouse-v642-completed-budget-locked-long-assets.v2'
ROLES=('robot_1','robot_2')
VARIANTS=('raw','regularized')
_CACHE={}


class InputRefs(list):
    """Preserve the complete read set once, despite repeated source checks."""
    def __init__(self):
        super().__init__();self._by_path={}

    def append(self,ref):
        require(ref['path'] not in self._by_path or self._by_path[ref['path']]==ref,
                'conflicting content addresses in long audit read set')
        if ref['path'] not in self._by_path:
            self._by_path[ref['path']]=ref;super().append(ref)


def source_files():
    names=('experiments/warehouse/v642_long_training_assets.py',
           'scripts/audit_warehouse_v642_long_results.py',
           'tests/test_warehouse_v642_long_training_assets.py','docs/warehouse_v642_long_training_assets.md',
           'experiments/warehouse/v642_update_receipt_audit.py','experiments/warehouse/v642_recovered_assets.py',
           'backend/training/warehouse_v642_checkpoint_export.py')
    return {name:file_sha(ROOT/name) for name in names}


def asset(ref,inputs):
    path=Path(ref['path'])
    require(path.is_absolute() and path.resolve()==path and pointer(path)==ref
            and not path.is_symlink() and not path.stat().st_mode&0o222,'long asset missing, changed, relocated, writable or symlinked')
    inputs.append(deepcopy(ref));return path


def _time(value):
    result=datetime.fromisoformat(value);require(result.tzinfo is not None,'long audit timestamps require a timezone');return result


def audit_lease(claim,clearance,plan):
    start=_time(claim['started_utc']);expiry=_time(claim['expires_utc'])
    issued=_time(clearance['issued_utc']);deadline=_time(clearance['expires_utc'])
    days=owner.seeds.EXECUTION_LEASE_DAYS_BY_BUDGET[plan.config.joint_step_budget]
    require(deadline-issued==timedelta(hours=24) and issued<=start<deadline
            and days==plan.value['execution_lease_days']==claim['execution_lease_days']
            and expiry-start==timedelta(days=days) and claim['seed_clearance_start_deadline_utc']==clearance['expires_utc']
            and claim['automatic_renewal'] is False,'long startup freshness or finite execution lease differs')
    return start,expiry


def audit_terminal(result_ref,exit_ref,*,plan,reference_directory,calibration_precheck_ref,inputs):
    """Only the source-bound real Popen.wait receipt can admit long output."""
    asset(exit_ref,inputs);e=bound(exit_ref);base=Path(plan.ref['path']).parent
    require(e.get('version')=='warehouse-v642-budget-locked-long-supervisor.v2'
            and e.get('observed_by')=='source_bound_subprocess_wait'
            and type(e.get('exit_code')) is int and e['exit_code']==0 and e.get('result')==result_ref
            and 'result_binding_error' not in e and Path(exit_ref['path'])==base/'long-supervisor-exit.json'
            and not (base/'long-supervisor-failed.json').exists(),'actual bound long-worker OS wait exit0 required')
    asset(e['started'],inputs);started=bound(e['started'])
    cli=pointer(ROOT/'scripts/run_warehouse_v642_long_training.py')
    command=[str(Path(sys.executable).resolve()),cli['path'],'--run','--reference-directory',str(Path(reference_directory).resolve()),
        '--calibration-precheck',calibration_precheck_ref['path'],'--plan',plan.ref['path']]
    phase=plan.value['phase'];lock_ref=plan.value.get('development_budget_lock');admission=plan.value.get('formal_admission')
    if phase=='formal':command.extend(['--phase','formal','--formal-budget-lock',lock_ref['path']])
    require(e['started']==pointer(base/'long-supervisor-started.json')
            and started['version']==e['version'] and started['observed_by']=='source_bound_subprocess_launch'
            and e['command']==started['command']==command and e['supervisor_source']==started['supervisor_source']==cli
            and e['seed_plan']==started['seed_plan']==plan.ref and started['calibration_precheck']==calibration_precheck_ref
            and e['phase']==started['phase']==phase
            and e['development_budget_lock']==started['development_budget_lock']==lock_ref
            and e['formal_admission']==started['formal_admission']==admission
            and type(e['worker_pid']) is int and e['worker_pid']>0 and type(e['supervisor_pid']) is int
            and e['supervisor_pid']==started['supervisor_pid']>0 and e['worker_pid']!=e['supervisor_pid']
            and e['started_utc']==started['started_utc'] and _time(e['started_utc'])<=_time(e['finished_utc']),
            'long supervisor source, command, PID, phase or timing differs')
    log=asset(e['log'],inputs)
    require(log==base/'long-run.log' and started['log_path']==str(log),'long supervisor log differs')
    asset(e['completed'],inputs)
    require(e['completed']==pointer(Path(plan.value['execution_output'])/'completed.json'), 'long terminal completion relocated')
    return e


def checkpoint_state(ref,record,*,variant,step,plan,inputs):
    """Hash the full native state without model construction or RNG restore."""
    asset(ref,inputs)
    value=torch.load(io.BytesIO(Path(ref['path']).read_bytes()),map_location='cpu',weights_only=True)
    required={'schema','model_version','reference_path','reference_sha256','reference_role_sha256','initialization_seed',
              'model','actor_optimizer','critic_optimizer','beta','joint_steps','actor_updates','critic_updates',
              'rng','extra_rng_states','requires_grad'}
    require(set(value)==required and value['schema']==CHECKPOINT_SCHEMA and value['model_version']==MODEL_VERSION
            and type(value['joint_steps']) is int and value['joint_steps']==step
            and value['reference_sha256']==plan.value['reference_actor']['sha256']
            and value['reference_role_sha256']==plan.value['reference_role_sha256']
            and value['initialization_seed']==plan.value['initialization_seed']
            and set(value['beta'])==set(ROLES) and value['rng'] and value['requires_grad'],
            'full long checkpoint schema, reference, RNG or step differs')
    require(set(value['rng'])=={'python','numpy','torch_cpu'} and isinstance(value['rng']['torch_cpu'],torch.Tensor)
            and value['rng']['torch_cpu'].dtype==torch.uint8 and value['rng']['torch_cpu'].ndim==1
            and set(value['rng']['numpy'])=={'name','words','position','gaussian','cached'}
            and value['rng']['numpy']['words'].shape==(624,) and type(value['rng']['python']) is tuple,
            'long checkpoint is missing full Python/NumPy/Torch RNG state')
    state=SimpleNamespace(reference_path=value['reference_path'],reference_sha256=value['reference_sha256'],
        initialization_seed=value['initialization_seed'],beta=value['beta'],joint_steps=step,
        actor_updates=value['actor_updates'],critic_updates=value['critic_updates'],
        model=SimpleNamespace(state_dict=lambda:value['model'],named_parameters=lambda:((key,SimpleNamespace(requires_grad=flag))
            for key,flag in value['requires_grad'].items())),
        actor_optimizer=SimpleNamespace(state_dict=lambda:value['actor_optimizer']),
        critic_optimizer=SimpleNamespace(state_dict=lambda:value['critic_optimizer']))
    require(training_fingerprint(state)==record['full_training_state_sha256']
            and value['actor_updates']==record['core']['actor_updates']
            and value['critic_updates']==record['core']['critic_updates']
            and value['beta']==record['core']['beta'],'checkpoint full tensors/optimizer state differ from the recorded core')
    require(checkpoint_actor_hashes(value)==record['core']['actor_role_sha256'],
            'long checkpoint inference tensors differ from the native optimizer role hashes')
    critic=sha256()
    for name in ('reward_critic','cost_critic'):
        pairs=[(key,tensor) for key,tensor in value['model'].items() if key.startswith(name+'.')]
        require(pairs,'long full checkpoint omitted a critic network')
        for key,tensor in pairs:
            array=np.ascontiguousarray(tensor.detach().cpu().numpy(),dtype='<f4')
            critic.update(key.encode());critic.update(np.asarray(array.shape,dtype='<i8').tobytes());critic.update(array.tobytes())
    require(critic.hexdigest()==record['core']['critic_sha256'],'long checkpoint critic tensors differ from the native core')
    if step==4000:
        require(not value['actor_optimizer']['state'] and value['actor_updates']==0,'critic warmup altered Actor optimizer')
    return value


def checkpoint_actor_hashes(value):
    roles={}
    for role in ROLES:
        prefix='actors.'+role+'.';model=value['model']
        require(model[prefix+'tree.support_fixed'].shape==(1,) and model[prefix+'tree.support_fixed'].item()==1,
                'long checkpoint lost the original fixed Actor feature support')
        arrays={name:model[prefix+name].detach().cpu().numpy().astype(np.float32,copy=True) for name in _tensor_names(MODEL_VERSION)}
        mask=model[prefix+'tree.support_mask'].detach().cpu().numpy()
        require(mask.shape==arrays['tree.split.weight'].shape and np.isin(mask,(0,1)).all(),'long Actor support mask changed')
        arrays['tree.split.weight']=np.add(np.multiply(arrays['tree.split.weight'],mask.astype(np.float32),dtype=np.float32),
                                          np.float32(0.),dtype=np.float32)
        roles[role]=role_sha256(arrays,model_version=MODEL_VERSION)
    return roles


def audit_commit_journal(events,updates):
    pending={};accepted={v:[] for v in VARIANTS}
    for e in events:
        if e['kind']=='optimizer_update_predebit':
            v=e['variant'];require(v in VARIANTS and v not in pending,'duplicate pending long optimizer predebit')
            pending[v]=e
        elif e['kind']=='optimizer_update_accepted':
            v=e['variant'];require(v in pending,'long optimizer commit lacks a predebit')
            prior=pending.pop(v);row=bound(e['receipt'])
            require(prior['step']==e['step']==row['end_joint_step']
                    and prior['trace']['sha256']==row['batch_sha256'],'long accepted update differs from its predebit')
            accepted[v].append(e['receipt'])
    require(not pending and accepted==updates,'long accepted optimizer journal is incomplete or reordered')


def audit_native_layout(refs,config):
    for batch,ref in enumerate(refs):
        row=bound(ref);n=row['end_joint_step']-row['start_joint_step']
        require(len(row['epochs'])==config.epochs,'long PPO epoch count changed')
        for epoch,entry in enumerate(row['epochs']):
            offsets=list(range(0,n,config.minibatch_joint_frames))
            require(entry['epoch']==epoch and len(entry['updates'])==len(offsets),'long PPO minibatches missing or duplicated')
            for offset,update in zip(offsets,entry['updates'],strict=True):
                require(update['epoch_id']==f'batch-{batch}/epoch-{epoch}/minibatch-{offset}'
                        and update['physical_frames']==min(config.minibatch_joint_frames,n-offset),
                        'long native optimizer epoch identity or physical-frame normalization differs')


def audit_resets(events,resets,*,plan,rollouts,snapshots):
    train=bound(plan.value['train_case_inventory'])['cases'];program=bound(plan.value['program_evidence_inventory'])['cases']
    streams={slot:[c for c in train if c['slot']==slot] for slot in range(8)}
    cursors={v:[0]*8 for v in VARIANTS};program_cursor={v:0 for v in VARIANTS};used={v:[] for v in VARIANTS};index=0
    for position,e in enumerate(events):
        if e['kind']!='environment_constructor_and_reset_predebit':continue
        v=e['variant'];case=e['case'];require(v in VARIANTS and index<len(resets),'extra or unknown long reset predebit')
        require(position+1<len(events),'long reset predebit lacks its actual hook')
        after=events[position+1];reset=resets[index]
        if e['scope']=='train':
            slot=case['slot'];cursor=cursors[v][slot]
            require(cursor<len(streams[slot]) and case==streams[slot][cursor],'long training reset skipped/reused its finite slot sequence')
            cursors[v][slot]+=1;episode=case['episode_id'];used[v].append((v,case))
        else:
            cursor=program_cursor[v]
            require(e['scope']=='program_evidence' and cursor<len(program) and case==program[cursor],
                    'long program reset skipped/reused its evidence sequence')
            program_cursor[v]+=1;episode=case['episode']
        require(after['kind']=='actual_reset_bound' and after['variant']==v and after['scope']==e['scope']
                and after['episode']==episode and after['reset_hook_index']==index and after['reset']==reset
                and reset['seed']==case['scenario_seed'] and reset['episode_id']==1,
                'long reset hook does not follow its exact predebited original argument')
        index+=1
    require(index==len(resets)<=plan.value['maximum_explicit_resets'],'long reset hook count or ceiling differs')
    for v in VARIANTS:
        require(used[v]==rollouts[v]['reset_cases'] and snapshots[v][-1]['reset_cursors']==cursors[v]
                and program_cursor[v]==snapshots[v][-1]['program_source_cursor']==1650,
                'long actual reset consumption differs from replayed trajectories or checkpoint cursors')
    return cursors


def audit_rollouts(update_refs,*,variant,owner_ref,plan,initial_roles,inputs):
    """Replay actual frozen NN actions and stored full-environment dynamics."""
    allowed={c['episode_id']:c for c in bound(plan.value['train_case_inventory'])['cases']}
    environments=[None]*8;active=[None]*8;used=set();current=deepcopy(initial_roles);frames=0;draws=0
    resets=[];complete=0;shutdowns=0;completed_ids=[]
    for update_ref in update_refs:
        asset(update_ref,inputs);row=bound(update_ref);folder=Path(update_ref['path']).parent
        trace_ref=pointer(folder/'trace.json');asset(trace_ref,inputs);trace=bound(trace_ref)
        require(trace['schema']==owner.TRACE_VERSION and trace['partition']=='train' and trace['owner']==owner_ref
                and trace['config']==asdict(plan.config) and trace['start_joint_step']==row['start_joint_step']==frames
                and trace['end_joint_step']==row['end_joint_step'] and row['batch_sha256']==trace_ref['sha256']
                and len(trace['events'])==8*plan.config.next_collection_length(frames), 'long rollout does not bind its actual accepted optimizer batch')
        asset(trace['actor'],inputs)
        actor=FrozenWarehouseV6Actor.load(trace['actor']['path'],expected_sha256=trace['actor']['sha256'],expected_role_sha256=current)
        require(trace['rollout_state']['actor_role_sha256']==current,'rollout actor is not the current optimizer-prefix NN')
        records=trace['physics'];require(records['schema']==physics.PHYSICS_VERSION,'unknown long physics records')
        ends=[None]*8;new=set();batch_complete=0;batch_shutdowns=0
        for index,event in enumerate(trace['events']):
            slot=index%8;case=event['case'];episode=case['episode_id']
            require(event['slot']==case['slot']==slot and allowed.get(episode)==case,'long physical frame outside finite train inventory')
            if environments[slot] is None:
                require(episode not in used and episode in records['resets'],'long rollout reused an episode or omitted initial state')
                ref=records['resets'][episode];asset(ref,inputs)
                env=physics.restore_environment(bound(ref));environments[slot]=env;active[slot]=case
                require(env.state.frame==0 and not(env.state.terminated or env.state.truncated),'long reset snapshot is not a fresh frame zero')
                used.add(episode);new.add(episode);resets.append((variant,case))
            env=environments[slot]
            require(active[slot]==case and asdict(env.config)==asdict(physics.training_config(case['scenario_seed']))
                    and env.state.participant_controlled_agent_id=='robot_1'
                    and physics.public_pre_action_snapshot(env.state)==event['pre_state_public'],'long physical state/config disconnected')
            actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False);draws+=2
            require(actions==event['proposed_actions'] and event['action_override_agents']==[],'executed long action was not the bound full NN')
            _,rewards,terminated,truncated,info=env.step(actions)
            require(rewards==event['rewards'] and terminated is event['terminated'] and truncated is event['truncated']
                    and sorted(info['shutdowns'])==sorted(event['shutdowns']) and not info['action_override_agents']
                    and physics.public_pre_action_snapshot(env.state)==event['post_state_public'],'long transition, reward or physical scoring differs')
            if index>=len(trace['events'])-8:ends[slot]=physics.capture_environment(env)
            if terminated or truncated:
                completed_ids.append(episode)
                batch_complete+=1;batch_shutdowns+=int(terminated);environments[slot]=None;active[slot]=None
        require(set(records['resets'])==new and len(records['end_snapshots'])==8,'long rollout reset/end inventory changed')
        for index,ref in enumerate(records['end_snapshots']):
            asset(ref,inputs);require(bound(ref)==ends[index],'long private end state or RNG differs')
        require(batch_complete==row['complete_episodes'] and batch_shutdowns==row['shutdown_episodes'],
                'long update complete-episode costs differ from actual terminal events')
        complete+=batch_complete;shutdowns+=batch_shutdowns
        for epoch in row['epochs']:
            for update in epoch['updates']:
                if update['actor_updated']:
                    require(update['role_sha256_before']==current,'long optimizer role hashes do not form a chain')
                    current=deepcopy(update['role_sha256_after'])
        frames=row['end_joint_step']
    return dict(joint_frames=frames,replayed_role_actions=draws,reset_cases=resets,
                complete_episodes=complete,shutdown_episodes=shutdowns,final_role_sha256=current,
                completed_episode_ids=sorted(completed_ids),
                environment_constructors=0,new_environment_resets=0,new_scenario_streams=0,
                optimizer_reexecution=False,critic_baseline_gradients_not_reexecuted=True)


def audit_materialization(target_ref,*,owner_ref,inputs):
    """Require the new producer's complete storage proof, then replay NN elsewhere.

    Native target-schema compatibility alone does not establish that the new
    long owner used its registered storage producer. This distinct proof is
    mandatory for every scheduled refit and every feedback-bearing call.
    """
    checked=materialization.read_materialization(target_ref,expected_owner=owner_ref)
    require(checked['canonical_NN_replay_still_required'] is True,
            'storage proof cannot replace complete native NN replay')
    for ref in checked['inputs']:asset(ref,inputs)
    return checked


def audit_storage_callback(final,*,provider_payload,inputs):
    envelope=final.get('long_storage_callback')
    require(type(envelope) is dict and set(envelope)=={'payload','payload_sha256'},
            'regularized long checkpoint lacks its explicit storage callback')
    p=envelope['payload'];native=provider_payload
    require(envelope['payload_sha256']==digest(p) and p['version']==storage_callback.VERSION
            and p['source_files']==storage_callback.source_files() and p['owner']==native['owner']
            and p['native_provider_started']==native['started']
            and p['full_training_state_sha256']==final['full_training_state_sha256']
            and p['safe_snapshot_not_resume_authority'] is True
            and type(p['call_count']) is int and p['call_count']==native['call_count'],
            'long callback snapshot/source/native-provider binding differs')
    folder=Path(native['output'])/'long-storage-callback'
    require(p['started']==pointer(folder/'started.json'),'long callback header escaped its native owner')
    asset(p['started'],inputs)
    require(bound(p['started'])==dict(version=storage_callback.VERSION,source_files=p['source_files'],
            owner=native['owner'],native_provider_started=native['started'],native_sources=native['source_files'],
            materialization='new_unsealed_stage_only',program_static_validation=storage_callback.prepared.VERSION,
            NN_method_changed=False,pilot_opened=False),
            'long callback immutable header differs')
    validations=[ref for ref in native['events'] if bound(ref)['kind']=='validation']
    require(len(p['events'])==len(validations)==p['call_count']
            and sorted((folder/'calls').glob('*.json'))==[Path(ref['path']) for ref in p['events']],
            'long callback history omits or adds native validations')
    for index,(ref,event_ref) in enumerate(zip(p['events'],validations,strict=True)):
        asset(ref,inputs);item=bound(ref);event=bound(event_ref)
        require(ref==pointer(folder/'calls'/f'{index:06d}.json')
                and item==dict(version=storage_callback.VERSION,sequence_call=index,
                    source_sha256=digest(p['source_files']),previous=p['events'][index-1] if index else None,
                    native_validation=event_ref,target_package=event['target_package'],
                    provenance=event['long_materialization_provenance'],
                    native_NN_and_qualification_method_unchanged=True,optimizer_updates=0)
                and event['sequence_call']==index and event['long_storage_callback_started']==p['started'],
                'long callback/native validation/materialization chain differs')
    audit_prepared_bindings(p['prepared_programs'],validations,inputs=inputs)
    return p


def audit_prepared_bindings(recorded,validations,*,inputs):
    """Rebuild static preparations from real program refs, never cached gates."""
    refs={}
    for event_ref in validations:
        event=bound(event_ref)
        require(set(event['programs'])==set(ROLES),'both native programs required in the static preparation ledger')
        for ref in event['programs'].values():
            asset(ref,inputs);value=bound(ref)
            refs[(ref['path'],ref['sha256'],value['target_actor_role_sha256'])]=ref
    expected=[]
    for key,ref in sorted(refs.items()):
        program=storage_callback.prepared.PreparedProgram.from_reference(ref,target_actor_role_sha256=key[2])
        binding=program.preparation_binding()
        require(binding['caches_current_NN_qualification'] is binding['training_authority'] is False,
                'prepared static program cannot replace current-NN qualification')
        for parent in binding['protection_dependencies']:asset(parent,inputs)
        expected.append(binding)
    require(type(recorded) is list and recorded==expected,
            'prepared program source, complete fields or static proof do not independently reproduce')
    return expected


def audit_provider(final,*,pool,readiness,config,updates,inputs):
    p=final['provider']['payload']
    require(final['provider']['payload_sha256']==digest(p) and p['version']==provider.VERSION
            and p['source_files']==provider.source_files() and p['config']==asdict(config)
            and p['full_training_state_sha256']==final['full_training_state_sha256'], 'long final provider state differs')
    asset(p['started'],inputs)
    seed_plan=bound(bound(bound(p['owner'])['descriptor'])['seed_plan'])
    require(p['owner']==final['owner'] and p['variant']=='regularized' and p['pair_seed']==seed_plan['initialization_seed']
            and Path(p['output'])==provider.PROVIDER_ROOT/digest(p['owner'])
            and bound(p['started'])==dict(version=provider.VERSION,source_files=p['source_files'],owner=p['owner'],
                pair_seed=p['pair_seed'],variant=p['variant'],config=asdict(config),
                readiness_sha256=readiness.payload_sha256,training_authority=False),'long provider header or pair ownership differs')
    obj=provider.CurrentNNEvidenceProvider.__new__(provider.CurrentNNEvidenceProvider)
    values=dict(reference_directory=None,ready=readiness,owner_ref=p['owner'],output=Path(p['output']),
        pair_seed=p['pair_seed'],variant=p['variant'],config=config,sources=p['source_files'],events=p['events'],
        installed=p['installed'],used_epochs=set(p['used_epochs']),call_count=p['call_count'],latch=p['latch'],
        reason=p['reason'],programs=p['programs'],observed=pool,failed=False,started_ref=p['started'])
    for key,value in values.items():setattr(obj,key,value)
    obj.verify();callback=audit_storage_callback(final,provider_payload=p,inputs=inputs);validations={}
    for event_ref in p['events']:
        asset(event_ref,inputs);event=bound(event_ref)
        for ref in event['asset_refs']:asset(ref,inputs)
        if event['kind']!='validation':continue
        stored=audit_materialization(event['target_package'],owner_ref=p['owner'],inputs=inputs)
        require(event['long_materialization_provenance']==stored['provenance_ref']
                and event['long_storage_callback_started']==callback['started'],
                'long validation did not use its exact source-bound target producer')
        folder=Path(p['output'])/'calls'/f'{event["sequence_call"]:06d}'
        asset(event['target_actor'],inputs)
        actor=FrozenWarehouseV6Actor.load(event['target_actor']['path'],expected_sha256=event['target_actor']['sha256'])
        for role in ROLES:
            e=provider.pools.build_observed_evidence(event['programs'][role],pool,event['target_package'],epoch_id=event['epoch_id'])
            args={**e.__dict__,'asset_refs':tuple([*e.asset_refs,event_ref,*[item['binding'] for item in p['installed']
                if item['step']==max(item['step'] for item in p['installed'] if item['step']<event['step'])],p['owner']]),
                'provider_receipt_ref':event_ref,'feedback_enabled':not event['latch_after'],'envelope_digest':''}
            out=provider.ProviderEvidence(**args);out=provider.ProviderEvidence(**{**args,'envelope_digest':digest(out.payload())})
            out.verify(current_role_sha256=actor.role_sha256[role],epoch_id=event['epoch_id'])
            ref=pointer(folder/(role+'-evidence.json'));asset(ref,inputs)
            require(out.payload()==bound(ref),'long current-NN qualification/envelope does not independently replay')
        validations[event['epoch_id']]=event
    consumed=[]
    for ref in updates:
        row=bound(ref)
        if row['end_joint_step']<=config.joint_step_budget*.8:continue
        for epoch in row['epochs']:
            for update in epoch['updates']:
                event=validations[update['epoch_id']]
                require(event['step']==row['end_joint_step'],'long provider evidence belongs to another PPO batch')
                for role in ROLES:
                    e=bound(pointer(Path(p['output'])/'calls'/f'{event["sequence_call"]:06d}'/(role+'-evidence.json')))
                    require(e['current_target_role_sha256']==update['role_sha256_before'][role], 'long program used another current NN')
                if event['latch_after']:
                    require(update['program_loss']==0 and all(n==0 for n in update['program_gradient']['role_program_gradient_norm_before_shared_clip'].values()),
                            'long fidelity latch failed to suppress program gradient')
                consumed.append(update['epoch_id'])
    require(consumed==list(validations),'long provider epochs do not match accepted native optimizer updates')
    return dict(validations=len(consumed),full_current_NN_targets_replayed=True,
                explicit_storage_callback_audited=True,scientific_sample_count_added=0)


def _phase_admission(phase,formal_budget_lock_ref):
    require(phase in ('development','formal'),'unknown explicit long audit phase')
    if phase=='formal':
        # Before calibration, result, plan or model I/O; a JSON phase flag
        # cannot grant permission to read a formal numerical training chain.
        return owner.seeds.formal.read_admission(formal_budget_lock_ref).verify()
    require(formal_budget_lock_ref is None,'development audit cannot infer formal authority from a lock')
    return None


def audit_completion(*,reference_directory,calibration_precheck_ref,result_ref,supervisor_exit_ref,
                     phase='development',formal_budget_lock_ref=None):
    admission=_phase_admission(phase,formal_budget_lock_ref)
    owner.seeds.protocol.register_preparation(calibration_precheck_ref).verify()  # Before long-plan/output reads.
    inputs=InputRefs();asset(result_ref,inputs);r=bound(result_ref)
    require(r.get('version')==owner.VERSION and r.get('phase')==phase and r.get('numerical_training_completed') is True
            and r.get('model_acceptance_passed') is r.get('pilot_opened') is r.get('publishable') is False,
            'only completed nonpublication long-development or explicitly admitted formal results may be audited')
    lock=admission.budget_lock_ref if admission is not None else None
    binding=admission.value if admission is not None else None
    require(r.get('development_budget_lock')==lock and r.get('formal_admission')==binding,
            'long result changed its explicit formal budget-lock binding')
    plan=owner.load_seed_plan(reference_directory,calibration_precheck_ref,r['seed_plan'],
                              phase=phase,formal_budget_lock_ref=formal_budget_lock_ref);plan.verify()
    folder=Path(plan.value['execution_output']);budget=plan.config.joint_step_budget
    require(Path(result_ref['path'])==folder/'controller-completed.json' and not (folder/'failed.json').exists(),
            'long result relocated or failed')
    exit_record=audit_terminal(result_ref,supervisor_exit_ref,plan=plan,reference_directory=reference_directory,
                               calibration_precheck_ref=calibration_precheck_ref,inputs=inputs)
    require(owner.checked_sources(plan)==r['source_contract'],'long executor source contract changed')
    clearance=owner.seeds.checked_clearance(plan,execution=False)
    for ref in (plan.ref,r['source_contract'],clearance['clearance'],r['owner_completed']):asset(ref,inputs)
    done=bound(r['owner_completed']);asset(done['claim'],inputs);asset(done['descriptor'],inputs)
    claim=bound(done['claim']);descriptor=bound(done['descriptor']);sources=owner.source_files()
    audit_lease(claim,clearance['value'],plan)
    require(done['version']==descriptor['version']==claim['version']==owner.VERSION and done['training_finished'] is True
            and done['claim']==pointer(Path(plan.value['global_claim_path'])) and claim['plan']==plan.ref
            and claim['descriptor']==done['descriptor']==pointer(folder/'descriptor.json')
            and claim['clearance']==descriptor['clearance']==clearance['clearance']
            and claim['single_use_even_if_interrupted'] is True and claim['automatic_resume'] is claim['automatic_retry'] is False
            and claim['owner_pid']==exit_record['worker_pid']
            and _time(exit_record['started_utc'])<=_time(claim['started_utc'])<=_time(exit_record['finished_utc'])
            and _time(clearance['value']['issued_utc'])<=_time(claim['started_utc'])<_time(clearance['value']['expires_utc'])
            and claim['execution_lease_days']==plan.value['execution_lease_days']
            and _time(claim['expires_utc'])-_time(claim['started_utc'])==timedelta(days=plan.value['execution_lease_days'])
            and claim['seed_clearance_start_deadline_utc']==clearance['value']['expires_utc']
            and claim['automatic_renewal'] is False
            and descriptor['source_files']==sources and descriptor['seed_plan']==plan.ref
            and r['stream_id']==descriptor['stream_id']==claim['stream_id']==plan.value['stream_id']
            and descriptor['phase']==claim['phase']==plan.value['phase']==phase
            and descriptor['development_budget_lock']==claim['development_budget_lock']==done['development_budget_lock']
                ==plan.value.get('development_budget_lock')==lock
            and descriptor['formal_admission']==claim['formal_admission']==done['formal_admission']
                ==plan.value.get('formal_admission')==binding
            and claim['joint_step_budget']==budget
            and descriptor['reference_actor']==plan.readiness.value['actor'] and descriptor['config']==asdict(plan.config)
            and descriptor['per_variant_joint_step_limit']==budget and descriptor['every_budget_restarts_at_step_zero'] is True
            and done['per_variant_joint_steps']=={v:budget for v in VARIANTS},'long single-use claim, lease, source or budget differs')
    journal=asset(pointer(folder/'owner-journal.jsonl'),inputs)
    events=[json.loads(line) for line in journal.read_text().splitlines()]
    require([e['sequence'] for e in events]==list(range(len(events))) and events[0]['kind']=='long_training_claimed'
            and events[-1]['kind']=='paired_long_steps_finished' and all(e['version']==owner.VERSION and e['claim']==done['claim']
            and e['phase']==plan.value['phase'] and e['stream_id']==plan.value['stream_id']
            and e['budget']==budget and _time(claim['started_utc'])<=_time(e['utc'])<_time(claim['expires_utc'])
            for e in events),'long owner journal order, identity or lease differs')
    starts=[e for e in events if e['kind']=='initialized']
    require(len(starts)==2 and {e['variant'] for e in starts}==set(VARIANTS),'missing or repeated long initialization')
    identity_keys=('roles','critic_sha256','full_initial_training_state_sha256','actor_optimizer_sha256',
                   'critic_optimizer_sha256','initial_joint_steps','initial_actor_updates','initial_critic_updates')
    identities={e['variant']:{k:e[k] for k in identity_keys} for e in starts}
    require(identities['raw']==identities['regularized'] and all(identities['raw'][k]==0 for k in
        ('initial_joint_steps','initial_actor_updates','initial_critic_updates'))
        and identities['raw']['roles']==plan.value['reference_role_sha256'],'long paired Actor/critic/optimizer initialization differs')
    snapshots={};reports={};rollouts={};owners={};program_sources={}
    audit_commit_journal(events,done['updates'])
    steps=owner.checkpoint_steps(plan.config)
    for variant in VARIANTS:
        owners[variant]=pointer(folder/variant/'owner.json');asset(owners[variant],inputs)
        refs=r['checkpoints'][variant];require(len(refs)==5,'all long checkpoints required')
        values=[]
        for step,ref in zip(steps,refs,strict=True):
            asset(ref,inputs);record=bound(ref);core=record['core']
            require(ref==pointer(folder/variant/'checkpoints'/f'{step:06d}'/'owner-state.json')
                and record['version']==owner.VERSION and record['owner']==owners[variant] and record['claim']==done['claim']
                and record['source_files']==sources and record['reserved_joint_steps']==core['processed_joint_steps']==step
                and record['safe_batch_boundary'] is core['safe_batch_boundary'] is True and record['pending_optimizer_batches']==0
                and core['failed'] is False and core['variant']==variant and record['paired_initial_identity']==identities[variant]
                and core['config']==asdict(plan.config)
                and core['version']==ppo.VERSION and core['source_sha256']==file_sha(ppo.__file__)
                and core['reference_sha256']==plan.readiness.value['actor']['sha256']
                and core['reference_buffer_sha256']==plan.readiness.value['reference']['buffer']['sha256']
                and record['refits']==[s for s in plan.config.extraction_steps if s<=step], 'unsafe or mismatched full long checkpoint')
            checkpoint_state(record['training_state'],record,variant=variant,step=step,plan=plan,inputs=inputs);values.append(record)
            reader=record['reader'];require(reader['payload_sha256']==digest(reader['payload']), 'long reader checkpoint envelope changed')
            rp=reader['payload'];public=rp['public']
            require(rp['source_files']==physics.source_files() and rp['owner']==owners[variant]
                    and rp['inventory']==plan.value['train_case_inventory']
                    and rp['environments']==record['environments'] and rp['active']==record['active_cases']
                    and public['payload_sha256']==digest(public['payload'])
                    and public['payload']['full_training_state_sha256']==record['full_training_state_sha256']
                    and public['payload']['processed_joint_steps']==step
                    and public['payload']['safe_assembly_boundary'] is True,'long reader/environment state differs from full checkpoint')
            for ref in rp['asset_refs']:asset(ref,inputs)
        final=values[-1];require(final['updates']==done['updates'][variant], 'long final update list differs')
        require(final['core']['update_sequence']==len(final['updates'])
                and final['core']['seen_batch_receipts']==sorted(bound(ref)['batch_sha256'] for ref in final['updates']),
                'long core batch-consumption ledger differs from native optimizer history')
        for earlier in values[:-1]:require(earlier['updates']==final['updates'][:len(earlier['updates'])],'long checkpoint history is not a prefix')
        reports[variant]=audit_updates(final['updates'],variant=variant,expected_joint_steps=budget,core=final['core'])
        audit_native_layout(final['updates'],plan.config)
        require((variant=='raw' and all(n==0 for n in reports[variant]['accepted_program_updates'].values()))
            or (variant=='regularized' and all(n>0 for n in reports[variant]['accepted_program_updates'].values())),
            'long actual raw zero/reg both-role positive program gradients required')
        shared.audit_NN_hash_chain(final['updates'],values)
        require([e['receipt'] for e in events if e['kind']=='optimizer_update_accepted' and e['variant']==variant]==final['updates'],
                'long actual optimizer journal differs from checkpoint receipts')
        rollouts[variant]=audit_rollouts(final['updates'],variant=variant,owner_ref=owners[variant],plan=plan,
                                        initial_roles=identities[variant]['roles'],inputs=inputs)
        require(rollouts[variant]['joint_frames']==budget and rollouts[variant]['final_role_sha256']==final['core']['actor_role_sha256'],
                'long rollout/optimizer chain did not reach its full budget')
        require(rollouts[variant]['completed_episode_ids']==final['core']['seen_complete_episodes'],
                'long complete episode cost identities differ from physical outcomes')
        program_sources[variant]=pointer(folder/variant/'program-source/source.json');asset(program_sources[variant],inputs)
        pool=provider.pools.audit_observed_pool(program_sources[variant],reference_directory=reference_directory)
        require(final['observed_source']==program_sources[variant] and final['observed_audit_sha256']==pool.audit_sha256
            and final['program_source_cursor']==1650,'long program source pool or cursor differs')
        for ref in pool.refs:asset(ref,inputs)
        refits=[e for e in events if e['kind']=='scheduled_refit_bound' and e['variant']==variant]
        require([e['step'] for e in refits]==list(plan.config.extraction_steps),'long program refits missing, duplicated or unscheduled')
        for event,record in zip(refits,values[1:4],strict=True):
            asset(event['binding'],inputs);b=bound(event['binding']);asset(b['extraction'],inputs);result=bound(b['extraction'])
            seal=pointer(Path(event['binding']['path']).with_name('observed-source-binding.completed.json'));asset(seal,inputs)
            require(bound(seal)==dict(binding_sha256=event['binding']['sha256'])
                    and b['schema']=='warehouse-v642-observed-refit-source-binding.v1'
                    and b['source_files']==provider.pools.source_files() and b['owner']==owners[variant]
                    and b['observed_source']==program_sources[variant] and b['observed_audit_sha256']==pool.audit_sha256
                    and result['identity']['step']==event['step'] and result['identity']['owner']==owners[variant]
                    and result['identity']['variant']==variant and result['identity']['config']==asdict(plan.config)
                    and result['full_training_state_sha256']==record['full_training_state_sha256'],
                    'long scheduled refit lacks the actual completed NN/source-state binding')
            audit_materialization(b['target_package'],owner_ref=owners[variant],inputs=inputs)
            for role in ROLES:
                e=provider.pools.build_observed_evidence(result['roles'][role]['program'],pool,b['target_package'],epoch_id=f'install-{event["step"]}')
                require(e.target_role_sha256==record['core']['actor_role_sha256'][role]
                        and e.report==result['roles'][role]['current'],'long refit qualification/target NN does not replay')
                for ref in e.asset_refs:asset(ref,inputs)
        if variant=='regularized':
            reports[variant]['provider']=audit_provider(final,pool=pool,readiness=plan.readiness,
                                                       config=plan.config,updates=final['updates'],inputs=inputs)
            final_native=final['provider']['payload'];final_storage=final['long_storage_callback']['payload']
            for record in values:
                if record['core']['processed_joint_steps']==4000:
                    require(record['provider'] is record.get('long_storage_callback') is None,
                            'critic warmup cannot install external program feedback')
                    continue
                native=record['provider']['payload']
                step=record['core']['processed_joint_steps']
                expected_events=[ref for ref in final_native['events'] if bound(ref)['step']<=step]
                require(record['provider']['payload_sha256']==digest(native)
                        and native['source_files']==provider.source_files()
                        and native['events']==expected_events,
                        'checkpoint native provider is not a prefix of the independently audited history')
                callback=audit_storage_callback(record,provider_payload=native,inputs=inputs)
                expected_calls=[ref for ref in final_storage['events']
                                if bound(bound(ref)['native_validation'])['step']<=step]
                require(callback['events']==expected_calls,
                        'checkpoint storage callback is not a prefix of the independently audited history')
        else:
            require(all(record['provider'] is record.get('long_storage_callback') is None for record in values),
                    'raw cannot install program feedback or its callback')
        snapshots[variant]=values
    require(snapshots['raw'][0]['full_training_state_sha256']==snapshots['regularized'][0]['full_training_state_sha256'],
            'paired warmup full Actor/critic/optimizer state differs')
    reset_ref=pointer(folder/'actual-resets.jsonl');asset(reset_ref,inputs)
    resets=[json.loads(line) for line in Path(reset_ref['path']).read_text().splitlines()]
    binds=[e for e in events if e['kind']=='actual_reset_bound']
    require(len(resets)==done['actual_reset_count']==len(binds)<=plan.value['maximum_explicit_resets'], 'long actual reset ceiling or journal differs')
    predebits=[e for e in events if e['kind']=='environment_constructor_and_reset_predebit']
    require(len(predebits)==len(binds) and all(b['reset_hook_index']==i and b['reset']==resets[i]
        and predebits[i]['variant']==b['variant'] and predebits[i]['case']['scenario_seed']==resets[i]['seed']
        and predebits[i]['scope']==b['scope'] for i,b in enumerate(binds)),'long reset consumed unregistered arguments or lacks predebit')
    audit_resets(events,resets,plan=plan,rollouts=rollouts,snapshots=snapshots)
    require(done['accepted_program_updates']=={v:reports[v]['accepted_program_updates'] for v in VARIANTS},'long completion gradient counters differ')
    for ref in inputs:asset(ref,[])
    return dict(plan=plan,events=events,owners=owners,program_sources=program_sources,result=r,
        checkpoints=snapshots,receipt_audit=reports,rollout_audit=rollouts,input_asset_refs=inputs,actual_reset_count=len(resets))


def run(*,reference_directory,calibration_precheck_ref,result_ref,supervisor_exit_ref,output,
        phase='development',formal_budget_lock_ref=None):
    _phase_admission(phase,formal_budget_lock_ref)
    sources=source_files()
    audited=audit_completion(reference_directory=reference_directory,calibration_precheck_ref=calibration_precheck_ref,
                             result_ref=result_ref,supervisor_exit_ref=supervisor_exit_ref,
                             phase=phase,formal_budget_lock_ref=formal_budget_lock_ref)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);exports={};plan=audited['plan'];budget=plan.config.joint_step_budget
    for variant in VARIANTS:
        exports[variant]={}
        for label,index,step in (('warmup',0,4000),('extract-75',1,plan.config.extraction_steps[0]),
            ('extract-84',2,plan.config.extraction_steps[1]),('extract-93',3,plan.config.extraction_steps[2]),('final',-1,budget)):
            record=audited['checkpoints'][variant][index]
            value=export_checkpoint_actor(record['training_state'],output/f'{variant}-{label}.actor.npz',variant=variant,
                    candidate_id=VERSION+':'+plan.value['stream_id'],expected_joint_steps=step)
            require(value['role_sha256']==record['core']['actor_role_sha256'],'long exported tensors differ from live checkpoint NN')
            exports[variant][label]=value
    for role in ROLES:
        anchor=plan.value['reference_role_sha256'][role]
        require(all(exports[v]['warmup']['role_sha256'][role]==anchor for v in VARIANTS)
            and len({anchor,*[exports[v]['final']['role_sha256'][role] for v in VARIANTS]})==3,
            'long warmup altered reference or final paired Actor weights did not separately change')
    require(source_files()==sources,'long audit/export source changed')
    report=dict(version=VERSION,reference_directory=str(Path(reference_directory).resolve()),calibration_precheck=calibration_precheck_ref,
        result=result_ref,supervisor_exit=supervisor_exit_ref,seed_plan=plan.ref,phase=plan.value['phase'],stream_id=plan.value['stream_id'],
        development_budget_lock=plan.value.get('development_budget_lock'),formal_admission=plan.value.get('formal_admission'),
        joint_step_budget=budget,audit_source_files=sources,input_asset_refs=audited['input_asset_refs'],exports=exports,
        receipt_audit=audited['receipt_audit'],rollout_audit={v:{k:x for k,x in a.items() if k!='reset_cases'} for v,a in audited['rollout_audit'].items()},
        actual_reset_count=audited['actual_reset_count'],numerical_training_completed=True,
        independent_quality_and_final_NN_program_evaluation_required=True,budget_lock_issued=False,
        formal_training_authorized=False,sealed_authority=False,model_acceptance_passed=False,publishable=False,pilot_opened=False,
        new_environment_resets=0,new_scenario_streams=0,optimizer_steps=0)
    write_new(output/'report.json',report);(output/'report.json').chmod(0o444);ref=pointer(output/'report.json')
    _CACHE[ref['sha256']]=audited
    return ref


def _context(report_ref,*,formal_budget_lock_ref=None):
    asset(report_ref,[]);r=bound(report_ref)
    require(r['version']==VERSION and r['phase'] in ('development','formal') and r['audit_source_files']==source_files()
            and r['model_acceptance_passed'] is r['pilot_opened'] is r['publishable'] is False,
            'long export is not an audited nonpublication development artifact')
    admission=_phase_admission(r['phase'],formal_budget_lock_ref)
    require(r.get('development_budget_lock')==(None if admission is None else admission.budget_lock_ref)
            and r.get('formal_admission')==(None if admission is None else admission.value),
            'long exported formal authority differs from the explicit independently replayed lock')
    if report_ref['sha256'] not in _CACHE:
        _CACHE[report_ref['sha256']]=audit_completion(reference_directory=r['reference_directory'],
            calibration_precheck_ref=r['calibration_precheck'],result_ref=r['result'],supervisor_exit_ref=r['supervisor_exit'],
            phase=r['phase'],formal_budget_lock_ref=formal_budget_lock_ref)
    c=_CACHE[report_ref['sha256']]
    c['plan'].verify();owner.checked_sources(c['plan'])
    for ref in c['input_asset_refs']:asset(ref,[])
    require(c['plan'].ref==r['seed_plan'] and c['receipt_audit']==r['receipt_audit'],'long export audit did not reproduce')
    require(r['input_asset_refs']==c['input_asset_refs'] and r['actual_reset_count']==c['actual_reset_count']
            and r['stream_id']==c['plan'].value['stream_id'] and r['joint_step_budget']==c['plan'].config.joint_step_budget
            and r['rollout_audit']=={v:{k:x for k,x in value.items() if k!='reset_cases'}
                                    for v,value in c['rollout_audit'].items()},'long export report omitted or changed independently replayed evidence')
    return r,c


def read_assets(report_ref,*,formal_budget_lock_ref=None):
    r,c=_context(report_ref,formal_budget_lock_ref=formal_budget_lock_ref)
    return _checkpoint_assets(report_ref,r,c,'final')


def _checkpoint_assets(report_ref,r,c,checkpoint):
    labels={'warmup':0,'extract-75':1,'extract-84':2,'extract-93':3,'final':-1}
    require(checkpoint in labels,'unregistered exact long checkpoint')
    actors={};roles={};inputs=[report_ref,*r['input_asset_refs']];index=labels[checkpoint]
    for variant in VARIANTS:
        export=r['exports'][variant][checkpoint];asset(export['actor'],inputs)
        require(export['checkpoint']==c['checkpoints'][variant][index]['training_state']
                and export['role_sha256']==c['checkpoints'][variant][index]['core']['actor_role_sha256'],
                'long frozen export lost its exact checkpoint binding')
        FrozenWarehouseV6Actor.load(export['actor']['path'],expected_sha256=export['actor']['sha256'],expected_role_sha256=export['role_sha256'])
        actors[variant]=export['actor'];roles[variant]=export['role_sha256']
    actors['reference']=c['plan'].readiness.value['actor'];roles['reference']=c['plan'].value['reference_role_sha256']
    return dict(report=report_ref,checkpoint=checkpoint,actors=actors,actor_role_sha256=roles,inputs=inputs)


def read_formal_checkpoint(report_ref,checkpoint,*,formal_budget_lock_ref):
    """The predeclared post-feedback quality pool, never chosen by programs."""
    _phase_admission('formal',formal_budget_lock_ref)
    require(checkpoint in ('extract-84','extract-93','final'),
            'formal quality cannot select warmup or pre-feedback checkpoints')
    r,c=_context(report_ref,formal_budget_lock_ref=formal_budget_lock_ref)
    require(r['phase']=='formal','development models cannot enter formal quality selection')
    return _checkpoint_assets(report_ref,r,c,checkpoint)


def finalization_context(report_ref,reference_directory,*,formal_budget_lock_ref=None):
    r,c=_context(report_ref,formal_budget_lock_ref=formal_budget_lock_ref)
    require(str(Path(reference_directory).resolve())==r['reference_directory'],'long finalization uses another reference directory')
    return dict(c,assets=_checkpoint_assets(report_ref,r,c,'final'))
