"""Audited observed states and explicitly counterfactual current-NN targets.

No fresh scenes, fitting, training or execution actions are authorized here.
Source episodes are replayed from their stored complete environment/RNG;
relabelled draws are never claimed as actual current-policy trajectories.
The future finite owner still proves reset freshness and exclusion from NN
gradient buffers; a scope label alone cannot establish those facts.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict,field
from hashlib import sha256
from pathlib import Path
import gzip,json
import numpy as np
from . import warehouse_v629_online_evidence as online
from . import warehouse_v630_online_evidence as evidence
from . import warehouse_v630_target_cache as target_cache
from . import warehouse_v629_environment_replay as physics
from . import warehouse_v630_program_refit as refit
from .warehouse_v6_actor import FrozenWarehouseV6Actor,MODEL_VERSION,encode_actor_tree_features
from .warehouse_v622_program import masked_probabilities
from env.warehouse.public_observation_v6 import public_pre_action_snapshot,encode_public_observation
from env.warehouse.pure_nn import AGENTS,action_masks
from env.warehouse.navigation import ACTIONS
from env.warehouse.decision_protocol import independent_agent_seed
from experiments.warehouse import v630_training_admission as admission
from experiments.warehouse.v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v630-observed-program-states.v1'
INVENTORY_VERSION='warehouse-v630-program-evidence-case-inventory.v1'
EPISODE_VERSION='warehouse-v630-program-source-episode.v1'
TARGET_VERSION='warehouse-v630-counterfactual-NN-targets.v1'
EVIDENCE_VERSION='warehouse-v630-observed-current-NN-evidence.v2'
SCOPE='program_evidence_only_excluded_from_NN_gradient_buffers'
TARGET_ROOT=admission.v.ROOT/'output/warehouse_v630/training-evidence-targets'


def source_files():
    files=physics.source_files();files.update(refit.source_files())
    files['backend/training/warehouse_v630_observed_pool.py']=file_sha(__file__)
    for name in ('tests/test_warehouse_v630_observed_pool.py','docs/warehouse_v630_observed_pool_bridge.md',
                 'tests/test_warehouse_v629_observed_pool.py','tests/test_warehouse_v629_environment_replay.py',
                 'backend/training/warehouse_v630_target_cache.py','tests/test_warehouse_v630_target_cache.py',
                 'docs/warehouse_v630_target_cache.md'):
        files[name]=file_sha(admission.v.ROOT/name)
    return dict(sorted(files.items()))


def data_sha(parts):
    h=sha256()
    for split in online.SPLITS:
        part=parts[split];h.update(split.encode());h.update(digest(part['rows']).encode())
        for key in ('x','m','behavior_p','executed'):
            a=part[key];h.update(key.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ObservedPool:
    source_ref:dict
    owner_ref:dict
    parts:dict
    refs:tuple
    sources:dict
    readiness:object
    observed_data_sha256:str
    audit:dict
    audit_sha256:str
    _replayed_target_cache:object=field(default=None,init=False,repr=False,compare=False)

    def verify(self):
        self.readiness.verify()
        require(self.sources==source_files() and data_sha(self.parts)==self.observed_data_sha256
            and digest(self.audit)==self.audit_sha256 and self.audit['source']==self.source_ref and self.audit['owner']==self.owner_ref
            and self.audit['source_sha256']==digest(self.sources) and self.audit['observed_data_sha256']==self.observed_data_sha256
            and self.audit['asset_manifest_sha256']==digest(self.refs),'observed state pool, audit, binding or source mutated')
        for ref in self.refs:
            path=Path(ref['path']);require(pointer(path)==ref and not path.is_symlink() and not path.stat().st_mode&0o222,
                'observed source asset changed, missing or writable')
        return self


def audit_observed_pool(source_ref,*,reference_directory):
    ready=admission.load_readiness(reference_directory)  # Before any file/RNG replay.
    ready.verify();sources=source_files();source=bound(source_ref)
    require(set(source)=={'schema','scope','owner','inventory','cases'} and source['schema']==VERSION and source['scope']==SCOPE,
        'source pool is not exclusively registered program evidence')
    owner=bound(source['owner']);inventory=bound(source['inventory'])
    require(owner['program_evidence_inventory']==source['inventory'] and inventory['schema']==INVENTORY_VERSION
        and inventory['scope']==SCOPE and set(inventory)=={'schema','scope','expected_counts','cases'}
        and set(inventory['expected_counts'])==set(online.SPLITS),'source inventory differs from its finite owner')
    cases=inventory['cases'];ids=set();scenes=set();action_streams=set();counts={s:0 for s in online.SPLITS}
    from .warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
    for case in cases:
        require(set(case)=={'episode','base_seed','scenario_seed','task','action_seed','split'}
            and type(case['episode']) is str and case['episode'] and case['split'] in online.SPLITS
            and all(type(case[k]) is int and 0<=case[k]<2**63 for k in ('base_seed','scenario_seed','task','action_seed'))
            and case['task'] in (1,2,3) and case['scenario_seed']==10*case['base_seed']+case['task']
            and case['episode'] not in ids and case['base_seed'] not in scenes and case['action_seed'] not in action_streams,
            'source episodes or whole scenes are ambiguous, reused or unregistered')
        require(not any(is_sealed_scenario_seed(case[k]) for k in ('base_seed','scenario_seed','action_seed')),
            'source pool consumes a sealed audit/holdout argument')
        ids.add(case['episode']);scenes.add(case['base_seed']);action_streams.add(case['action_seed']);counts[case['split']]+=1
    require(counts==inventory['expected_counts'] and all(type(n) is int and n>0 for n in counts.values())
        and all(type(n) is int and n>0 for n in inventory['expected_counts'].values())
        and set(source['cases'])==ids,'source pool missing registered cases or partitions')
    refs=[deepcopy(source_ref),source['owner'],source['inventory']];parts={s:{k:[] for k in ('x','m','behavior_p','executed','rows')} for s in online.SPLITS}
    behavior_actors={};outcomes=[]
    for case in cases:
        item=source['cases'][case['episode']]
        require(set(item)=={'behavior_actor','trace','initial_environment','final_environment'},'unknown physical source episode record')
        refs.extend(deepcopy(list(item.values())))
        actor_ref=item['behavior_actor'];key=(actor_ref['path'],actor_ref['sha256'])
        if key not in behavior_actors:
            actor=FrozenWarehouseV6Actor.load(actor_ref['path'],expected_sha256=actor_ref['sha256'])
            require(actor.model_version==MODEL_VERSION,'source behavior NN architecture changed');behavior_actors[key]=actor
        actor=behavior_actors[key]
        env=physics.restore_environment(bound(item['initial_environment']))
        require(asdict(env.config)==asdict(physics.training_config(case['scenario_seed'])) and env.state.frame==0
            and not(env.state.terminated or env.state.truncated) and env.state.participant_controlled_agent_id=='robot_1',
            'source physical/reward config, initial frame or direct-NN mode differs')
        path=Path(item['trace']['path'])
        if path.suffix=='.gz':
            with gzip.open(path,'rt') as handle:trace=json.load(handle)
        else:trace=bound(item['trace'])
        require(set(trace)=={'schema','case','behavior_actor','events'} and trace['schema']==EPISODE_VERSION
            and trace['case']==case and trace['behavior_actor']==actor_ref and 0<len(trace['events'])<=120,
            'source trace identity or complete-frame inventory differs')
        part=parts[case['split']]
        for frame,event in enumerate(trace['events']):
            require(not(env.state.terminated or env.state.truncated) and env.state.frame==frame
                and public_pre_action_snapshot(env.state)==event['pre_state_public'],'source physical/public history disconnected')
            require(type(event['terminated']) is type(event['truncated']) is bool,'source terminal events need explicit boolean types')
            obs=encode_public_observation(env.state);m=action_masks(env.state);x=encode_actor_tree_features(obs,m)
            p=masked_probabilities(actor.logits(obs,m),m);actions=[]
            for i,role in enumerate(AGENTS):
                seed=independent_agent_seed(base_seed=case['action_seed'],episode_id=env.state.episode_id,frame=frame,agent_id=role)
                actions.append(int(np.random.default_rng(seed).choice(5,p=p[i])))
            proposed={role:ACTIONS[actions[i]] for i,role in enumerate(AGENTS)}
            require(event['proposed_actions']==proposed and event['action_override_agents']==[],'source action is not the recorded behavior NN')
            physical_id=env.state.episode_id
            _,rewards,terminated,truncated,info=env.step(proposed)
            require(event['rewards']==rewards and event['terminated']==terminated and event['truncated']==truncated
                and sorted(event['shutdowns'])==sorted(info['shutdowns']) and not info['action_override_agents']
                and public_pre_action_snapshot(env.state)==event['post_state_public'],'source reward/outcome/physics differs')
            for name,value in (('x',x),('m',m),('behavior_p',p),('executed',actions)):part[name].append(value)
            part['rows'].append(dict(episode=case['episode'],base_seed=str(case['base_seed']),task=case['task'],frame=frame,
                action_seed=case['action_seed'],episode_id=physical_id))
        require((env.state.terminated or env.state.truncated) and physics.capture_environment(env)==bound(item['final_environment']),
            'source episode is partial or final private state/RNG differs')
        outcomes.append(dict(case=case,behavior_actor=actor_ref,physical_frames=len(trace['events']),deliveries=env.state.total_deliveries,
            robot_collisions=env.state.robot_collision_events,shutdown=bool(env.state.shutdown_count)))
    for part in parts.values():
        for name in ('x','m','behavior_p','executed'):
            part[name]=np.asarray(part[name],np.int64 if name=='executed' else np.float32);part[name].flags.writeable=False
    observed_sha=data_sha(parts)
    audit=dict(version=VERSION,source=source_ref,owner=source['owner'],inventory=source['inventory'],source_sha256=digest(sources),
        observed_data_sha256=observed_sha,asset_manifest_sha256=digest(tuple(refs)),
        complete_source_episodes=len(cases),joint_physical_frames=sum(len(p['x']) for p in parts.values()),outcomes=outcomes,
        current_policy_task_quality_estimate=None,source_policy_outcomes_only=True,environment_constructors=0,new_environment_resets=0,
        owner_must_verify_original_reset_authority_freshness_and_exclusion_from_NN_gradients=True,training_authority=False)
    return ObservedPool(deepcopy(source_ref),deepcopy(source['owner']),parts,tuple(refs),sources,ready,observed_sha,audit,digest(audit)).verify()


def relabel_current_NN(observed,target_actor_ref,output):
    require(isinstance(observed,ObservedPool),'audited observed states required');observed.verify()
    require(pointer(target_actor_ref['path'])==target_actor_ref,'current target NN changed')
    actor=FrozenWarehouseV6Actor.load(target_actor_ref['path'],expected_sha256=target_actor_ref['sha256'])
    require(actor.model_version==MODEL_VERSION,'target NN architecture changed')
    output=Path(output).resolve();require(output.is_relative_to(TARGET_ROOT.resolve()) and output!=TARGET_ROOT.resolve(),
        'counterfactual targets need a new isolated output directory')
    output.mkdir(parents=True,exist_ok=False);pools={};differences={};actual={};refs=[]
    try:
        for split,part in observed.parts.items():
            file=output/(split+'-source-actual-actions.npz')
            with file.open('xb') as handle:np.savez_compressed(handle,executed=part['executed'])
            file.chmod(0o444);actual[split]=pointer(file);refs.append(actual[split])
        for i,role in enumerate(AGENTS):
            sources={};differences[role]={}
            for split,part in observed.parts.items():
                x,m=part['x'][:,i],part['m'][:,i];p=online.runtime_probabilities(actor,role,x,m)
                sampled=np.array([np.random.default_rng(independent_agent_seed(base_seed=row['action_seed'],episode_id=row['episode_id'],
                    frame=row['frame'],agent_id=role)).choice(5,p=prob) for row,prob in zip(part['rows'],p,strict=True)],np.int64)
                file=output/(role+'-'+split+'.npz')
                with file.open('xb') as handle:np.savez_compressed(handle,x=x,m=m,p=p,sampled=sampled)
                file.chmod(0o444);rows=output/(role+'-'+split+'-rows.json');write_new(rows,part['rows']);rows.chmod(0o444)
                sources[split]=dict(arrays=pointer(file),rows=pointer(rows));refs.extend(sources[split].values())
                differences[role][split]=int(np.count_nonzero(sampled!=part['executed'][:,i]))
            file=output/(role+'-pool.json');write_new(file,dict(schema=online.MANIFEST_VERSION,scope='online_program_training_evidence',
                target_actor=target_actor_ref,role=role,parts=sources));file.chmod(0o444);pools[role]=pointer(file);refs.append(pools[role])
        observed.verify();require(pointer(target_actor_ref['path'])==target_actor_ref,'NN changed while relabelling')
        result=dict(schema=TARGET_VERSION,observed_source=observed.source_ref,observed_audit_sha256=observed.audit_sha256,
            observed_data_sha256=observed.observed_data_sha256,owner=observed.owner_ref,target_actor=target_actor_ref,source_files=source_files(),
            pools=pools,actual_source_actions=actual,counterfactual_difference_counts=differences,asset_refs=refs,
            current_policy_trajectory=False,counterfactual_draws_not_executed=True,actual_actions_replaced=False,
            actual_PPO_feedback_counts_from_real_rollouts_only=True,current_policy_task_quality_estimate=None,
            uses_existing_recorded_seeds_for_counterfactual_numerical_replay=True,fresh_randomness_claimed=False,
            environment_resets=0,NN_updates=0,PPO_joint_steps=0,training_authority=False,pilot_opened=False)
        write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
        write_new(output/'completed.json',dict(result_sha256=file_sha(output/'result.json')));(output/'completed.json').chmod(0o444)
        return pointer(output/'result.json')
    except BaseException as error:
        write_new(output/'failed.json',dict(error=str(error),automatic_retry=False,training_authority=False));raise


def _replay_targets(observed,ref):
    observed.verify();p=bound(ref);require(p['schema']==TARGET_VERSION and p['observed_source']==observed.source_ref
        and p['observed_audit_sha256']==observed.audit_sha256 and p['observed_data_sha256']==observed.observed_data_sha256
        and p['owner']==observed.owner_ref and p['source_files']==source_files() and set(p['pools'])==set(AGENTS)
        and p['current_policy_trajectory'] is p['actual_actions_replaced'] is p['training_authority'] is False
        and p['counterfactual_draws_not_executed'] is p['actual_PPO_feedback_counts_from_real_rollouts_only'] is True
        and p['current_policy_task_quality_estimate'] is None and p['fresh_randomness_claimed'] is p['pilot_opened'] is False
        and p['uses_existing_recorded_seeds_for_counterfactual_numerical_replay'] is True
        and all(type(p[k]) is int and p[k]==0 for k in ('environment_resets','NN_updates','PPO_joint_steps')),
        'counterfactual target scope or source binding differs')
    require(bound(pointer(Path(ref['path']).with_name('completed.json')))==dict(result_sha256=ref['sha256']),'counterfactual target seal differs')
    require(set(p['actual_source_actions'])==set(online.SPLITS) and set(p['counterfactual_difference_counts'])==set(AGENTS)
        and all(set(values)==set(online.SPLITS) for values in p['counterfactual_difference_counts'].values()),'counterfactual part inventory differs')
    require(pointer(p['target_actor']['path'])==p['target_actor'],'target NN changed')
    actor=FrozenWarehouseV6Actor.load(p['target_actor']['path'],expected_sha256=p['target_actor']['sha256'])
    require(actor.model_version==MODEL_VERSION,'counterfactual target NN architecture changed')
    refs=[ref,pointer(Path(ref['path']).with_name('completed.json')),*p['asset_refs'],p['target_actor']]
    for item in refs:require(pointer(item['path'])==item,'counterfactual target asset missing or changed')
    parts={};expected_refs=[]
    for split,item in p['actual_source_actions'].items():
        expected_refs.append(item)
        with np.load(item['path'],allow_pickle=False) as z:
            require(z.files==['executed'] and np.array_equal(z['executed'],observed.parts[split]['executed']),'source actual actions were overwritten or relabelled')
    for i,role in enumerate(AGENTS):
        parts[role],role_refs=online.load_pool(p['pools'][role],p['target_actor'],actor,role);expected_refs.extend(role_refs)
        for split,a in parts[role].items():
            old=observed.parts[split]
            require(np.array_equal(a['x'],old['x'][:,i]) and np.array_equal(a['m'],old['m'][:,i]) and a['rows']==old['rows']
                and p['counterfactual_difference_counts'][role][split]==int(np.count_nonzero(a['sampled']!=old['executed'][:,i])),
                'counterfactual targets replaced observed states, provenance or actual-action counts')
    require({digest(item) for item in expected_refs}=={digest(item) for item in p['asset_refs']},'counterfactual asset manifest omitted consumed files')
    observed.verify()
    for item in refs:require(pointer(item['path'])==item,'counterfactual asset changed during replay')
    return p,refs,parts,actor.role_sha256


def checked_targets(observed,ref):
    observed.verify();cached=observed._replayed_target_cache
    if cached is not None and cached.target_ref==ref:
        cached.verify(observed)
        return deepcopy(cached.payload),deepcopy(list(cached.asset_refs))
    p,refs,parts,roles=_replay_targets(observed,ref)
    cached=target_cache._seal_after_replay(observed=observed,ref=ref,payload=p,
        parts=parts,role_hashes=roles,asset_refs=refs)
    # At most one target package stays resident. Never serialize this cache
    # as authority; a restored pool starts with no replay certificate.
    object.__setattr__(observed,'_replayed_target_cache',cached)
    return p,refs


@dataclass(frozen=True)
class ObservedOnlineEvidence(evidence.OnlineEvidence):
    observed:ObservedPool
    target_package_ref:dict

    def payload(self):
        value=super().payload()
        value.update(version=EVIDENCE_VERSION,observed_source=self.observed.source_ref,observed_audit_sha256=self.observed.audit_sha256,
            target_package=self.target_package_ref,asset_manifest_sha256=digest(self.asset_refs),
            validation_actions_are_counterfactual=True,actual_PPO_feedback_counts_from_real_rollouts_only=True)
        return value

    def verify(self,*,current_role_sha256,epoch_id):
        self.observed.verify();return super().verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)


def build_observed_evidence(program_ref,observed,targets_ref,*,epoch_id):
    require(isinstance(epoch_id,str) and epoch_id and len(epoch_id)<=128,'explicit optimizer epoch ID required')
    targets,refs=checked_targets(observed,targets_ref);cached=observed._replayed_target_cache
    cached.verify(observed);encoded=bound(program_ref)
    program=evidence.Program.from_dict(encoded,encoded['actor_role_sha256']);role=program.role
    # Qualification remains program- and epoch-specific. Only the unchanged
    # current-NN targets, already replayed above, avoid another NN forward.
    report=online.current_report(program,cached.parts[role])
    role_refs=[targets['pools'][role]];manifest=bound(targets['pools'][role])
    for split in online.SPLITS:
        role_refs.extend([manifest['parts'][split]['arrays'],manifest['parts'][split]['rows']])
    base=evidence.OnlineEvidence(program,report,deepcopy(program_ref),deepcopy(targets['target_actor']),
        cached.role_hashes[role],deepcopy(targets['pools'][role]),epoch_id,
        tuple(deepcopy([program_ref,targets['target_actor'],program.protection['parent'],*role_refs])), '')
    base=evidence.OnlineEvidence(**{**base.__dict__,'envelope_digest':digest(base.payload())})
    base.verify(current_role_sha256=base.target_role_sha256,epoch_id=epoch_id)
    args={**base.__dict__,'asset_refs':tuple(deepcopy([*base.asset_refs,*refs])),
        'observed':observed,'target_package_ref':deepcopy(targets_ref),'envelope_digest':''}
    value=ObservedOnlineEvidence(**args);value=ObservedOnlineEvidence(**{**args,'envelope_digest':digest(value.payload())})
    return value.verify(current_role_sha256=value.target_role_sha256,epoch_id=epoch_id)


def load_observed_evidence(ref,observed,*,current_role_sha256,epoch_id):
    encoded=bound(ref)
    require(encoded.get('version')==EVIDENCE_VERSION,'unknown observed evidence version')
    out=build_observed_evidence(encoded['anchor_program'],observed,encoded['target_package'],epoch_id=encoded['epoch_id'])
    require(out.payload()==encoded,'observed evidence source, metrics or counterfactual scope does not replay')
    return out.verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)


def refit_observed_pair_at_boundary(*,reference_directory,owner_ref,state,observed,targets_ref,program_refs,output,
        variant,epoch_id,config=physics.PPOConfig()):
    admission.load_readiness(reference_directory).verify()  # Before touching a pool.
    observed.verify();targets,_=checked_targets(observed,targets_ref);sources=source_files()
    require(targets['owner']==owner_ref,'observed program data belongs to another training owner')
    output=Path(output).resolve()
    try:
        result=refit.refit_pair_at_boundary(reference_directory=reference_directory,owner_ref=owner_ref,state=state,
            target_actor_ref=targets['target_actor'],program_refs=program_refs,pool_refs=targets['pools'],output=output,
            variant=variant,epoch_id=epoch_id,config=config)
        observed.verify();checked_targets(observed,targets_ref)
        require(source_files()==sources,'observed refit source changed during fitting')
        value=dict(schema='warehouse-v630-observed-refit-source-binding.v1',source_files=sources,owner=owner_ref,
            extraction=pointer(output/'result.json'),observed_source=observed.source_ref,observed_audit_sha256=observed.audit_sha256,
            target_package=targets_ref,validation_actionable_coverage_scope='counterfactual_current_NN_draws_on_fixed_observed_states',
            actual_PPO_feedback_counts_from_real_rollouts_only=True,current_policy_task_quality_estimate=None,
            NN_updates=0,PPO_joint_steps=0,environment_resets=0,training_authority=False,pilot_opened=False)
        write_new(output/'observed-source-binding.json',value)
        write_new(output/'observed-source-binding.completed.json',dict(binding_sha256=file_sha(output/'observed-source-binding.json')))
        return result,pointer(output/'observed-source-binding.json')
    except BaseException as error:
        if output.is_dir() and not (output/'observed-source-binding.failed.json').exists():
            write_new(output/'observed-source-binding.failed.json',dict(error=str(error),automatic_retry=False,training_authority=False))
        raise
