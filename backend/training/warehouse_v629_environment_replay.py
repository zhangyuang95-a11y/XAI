"""Replay recorded physics and rewards without constructors or fresh resets.

Private state is an audit/checkpoint input only. NN inputs remain the public
encoder used by PublicRolloutReplay. This module does not prove that a reset
was authorized or fresh: the finite Stage C owner must bind its actual reset
journal and historical clearance before collecting any scientific data.
"""
from copy import deepcopy
from dataclasses import fields,is_dataclass,asdict
from pathlib import Path
import math,random
from .warehouse_v629_rollout_replay import PublicRolloutReplay,training_fingerprint
from .warehouse_v625_actor_training import PPOConfig
from .warehouse_v6_runner import TRAINING_REWARD_OVERRIDES
from env.warehouse.domain import WarehouseConfig,WarehouseState,AgentState,DeliveryTask,participant_study_config
from env.warehouse.rewards import RewardConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import get_map_layout
from env.warehouse.public_observation_v6 import public_pre_action_snapshot
from env.warehouse.public_action_features_v617 import validate_study_action_physics
from env.warehouse.pure_nn import AGENTS
from experiments.warehouse import v629_training_admission as admission
from experiments.warehouse.v626_protocol import require,pointer,bound,file_sha,digest

VERSION='warehouse-v629-full-environment-audit.v1'
SNAPSHOT_VERSION='warehouse-v629-full-environment-snapshot.v1'
PHYSICS_VERSION='warehouse-v629-PPO-physics-records.v1'
CLASSES={c.__name__:c for c in (WarehouseConfig,WarehouseState,AgentState,DeliveryTask,RewardConfig)}
ENV_FIELDS={'config','layout','_rng','_episode_counter','state'}


def source_files():
    sources=admission.source_files()
    sources['backend/training/warehouse_v629_environment_replay.py']=file_sha(__file__)
    return dict(sorted(sources.items()))


def encode(value):
    """Explicit data-only types; no pickle, imports, callbacks or object hooks."""
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:
        require(math.isfinite(value),'nonfinite private environment value');return value
    if type(value) in (list,tuple):return dict(kind=type(value).__name__,items=[encode(x) for x in value])
    if type(value) is dict:
        require(all(type(k) is str for k in value),'environment dictionaries need string keys')
        return dict(kind='dict',items=[[k,encode(value[k])] for k in sorted(value)])
    if is_dataclass(value) and type(value).__name__ in CLASSES and type(value) is CLASSES[type(value).__name__]:
        return dict(kind=type(value).__name__,fields={f.name:encode(getattr(value,f.name)) for f in fields(value)})
    raise ValueError('unsupported private environment type')


def decode(value):
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:
        require(math.isfinite(value),'nonfinite snapshot value');return value
    require(type(value) is dict and type(value.get('kind')) is str,'unknown private snapshot type')
    kind=value['kind']
    if kind in ('list','tuple'):
        require(set(value)=={'kind','items'} and type(value['items']) is list,'malformed environment sequence')
        items=[decode(x) for x in value['items']];return items if kind=='list' else tuple(items)
    if kind=='dict':
        require(set(value)=={'kind','items'} and type(value['items']) is list,'malformed environment dictionary')
        pairs=value['items']
        require(all(type(p) is list and len(p)==2 and type(p[0]) is str for p in pairs)
            and [p[0] for p in pairs]==sorted({p[0] for p in pairs}),'duplicate or noncanonical environment dictionary')
        return {k:decode(v) for k,v in pairs}
    require(kind in CLASSES and set(value)=={'kind','fields'} and type(value['fields']) is dict
        and set(value['fields'])=={f.name for f in fields(CLASSES[kind])},'unknown or incomplete environment dataclass')
    return CLASSES[kind](**{k:decode(v) for k,v in value['fields'].items()})


def capture_environment(env):
    require(type(env) is WarehouseMultiAgentEnv and set(vars(env))==ENV_FIELDS
        and type(env._rng) is random.Random and type(env.state) is WarehouseState
        and type(env._episode_counter) is int and env._episode_counter==env.state.episode_id
        and env.layout==get_map_layout(env.config.map_layout_id),'unsupported or incomplete environment instance')
    validate_study_action_physics(env.config)
    require(not env.validate_state(env.state),'invalid private environment state')
    payload=dict(schema=SNAPSHOT_VERSION,config=encode(env.config),state=encode(env.state),
        rng=encode(env.get_rng_state()),episode_counter=env._episode_counter)
    return dict(payload=payload,payload_sha256=digest(payload))


def restore_environment(snapshot):
    require(type(snapshot) is dict and set(snapshot)=={'payload','payload_sha256'}
        and digest(snapshot['payload'])==snapshot['payload_sha256'],'environment snapshot integrity differs')
    p=snapshot['payload']
    require(set(p)=={'schema','config','state','rng','episode_counter'} and p['schema']==SNAPSHOT_VERSION,
        'unknown environment snapshot schema')
    config,state,rng=decode(p['config']),decode(p['state']),decode(p['rng'])
    require(type(config) is WarehouseConfig and type(state) is WarehouseState,'environment config/state type differs')
    # __new__ allocates empty objects without __init__/seed/reset. set_state
    # intentionally is not used: it would regenerate private goal bookkeeping.
    env=WarehouseMultiAgentEnv.__new__(WarehouseMultiAgentEnv)
    env.config=config;env.layout=get_map_layout(config.map_layout_id)
    env._rng=random.Random.__new__(random.Random);env._rng.setstate(rng)
    env._episode_counter=p['episode_counter'];env.state=state
    require(capture_environment(env)==snapshot,'environment snapshot does not roundtrip exactly')
    return env


def training_config(scenario_seed):
    return participant_study_config(seed=scenario_seed,**TRAINING_REWARD_OVERRIDES)


class EnvironmentRolloutReplay:
    """Full stored-environment replay followed by the independent NN reader.

    An inventory proves finite membership, not freshness/authority. The owner
    must create it only after A/B admission and its own exact seed audit.
    A failure permanently closes this reader, including partial consumption.
    """
    def __init__(self,owner_ref,inventory_ref,*,config=PPOConfig()):
        require(pointer(owner_ref['path'])==owner_ref and pointer(inventory_ref['path'])==inventory_ref,'missing physics owner/inventory')
        owner=bound(owner_ref);inventory=bound(inventory_ref)
        require(owner['train_case_inventory']==inventory_ref and inventory['schema']=='warehouse-v629-finite-training-case-inventory.v1',
            'physics inventory is not bound by the owner')
        cases=inventory['cases'];require(type(cases) is list and cases,'empty finite training inventory')
        self.cases={}
        for case in cases:
            require(set(case)=={'episode_id','scenario_seed','task','action_seed','slot','partition'}
                and type(case['episode_id']) is str and case['episode_id'] and case['partition']=='train'
                and type(case['slot']) is int and 0<=case['slot']<8 and type(case['task']) is int and case['task'] in (1,2,3)
                and all(type(case[k]) is int and 0<=case[k]<2**63 for k in ('scenario_seed','action_seed'))
                and case['episode_id'] not in self.cases,'invalid finite training case')
            self.cases[case['episode_id']]=deepcopy(case)
        require(len({c['scenario_seed'] for c in cases})==len(cases) and len({c['action_seed'] for c in cases})==len(cases),
            'independent training episodes reused a scene or action stream')
        from .warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
        require(not any(is_sealed_scenario_seed(c[k]) for c in cases for k in ('scenario_seed','action_seed')),
            'training inventory contains a sealed audit/holdout argument')
        self.owner_ref=deepcopy(owner_ref);self.inventory_ref=deepcopy(inventory_ref);self.config=config
        self.public=PublicRolloutReplay(owner_ref,config=config);self.environments=[None]*8;self.active=[None]*8
        self.used_episodes=set();self.asset_refs=[deepcopy(owner_ref),deepcopy(inventory_ref)];self.failed=False
        self.sources=source_files()

    def verify(self):
        require(not self.failed and source_files()==self.sources,'failed physics reader or changed source')
        for ref in self.asset_refs:require(pointer(ref['path'])==ref,'physics prerequisite asset missing or changed')

    def assemble(self,trace_ref,state):
        try:
            self.verify();require(pointer(trace_ref['path'])==trace_ref,'physical trace changed')
            trace=bound(trace_ref);events=trace['events'];records=trace['physics'];n=len(events)
            require(trace['owner']==self.owner_ref and trace['config']==asdict(self.config)
                and trace['start_joint_step']==self.public.processed_joint_steps and trace['end_joint_step']==state.joint_steps
                and n==8*self.config.next_collection_length(self.public.processed_joint_steps)
                and set(records)=={'schema','resets','end_snapshots'} and records['schema']==PHYSICS_VERSION
                and type(records['resets']) is dict and len(records['end_snapshots'])==8,'incomplete physical vector records')
            refs=deepcopy([trace_ref,*records['resets'].values(),*records['end_snapshots']])
            for ref in refs:require(pointer(ref['path'])==ref,'stored environment state changed')
            resets=set();before=training_fingerprint(state);complete=0;ends=[None]*8
            for index,event in enumerate(events):
                slot=index%8;case=event['case'];key=case['episode_id']
                require(event['slot']==slot and case['slot']==slot and self.cases.get(key)==case,'physics frame outside finite case inventory')
                if self.environments[slot] is None:
                    require(key not in self.used_episodes and key in records['resets'],'missing or reused initial physical state')
                    env=restore_environment(bound(records['resets'][key]));self.environments[slot]=env
                    require(env.state.frame==0 and not(env.state.terminated or env.state.truncated),'new physical episode does not start at zero')
                    self.active[slot]=deepcopy(case);self.used_episodes.add(key);resets.add(key)
                env=self.environments[slot]
                require(self.active[slot]==case and asdict(env.config)==asdict(training_config(case['scenario_seed']))
                    and env.state.participant_controlled_agent_id=='robot_1','episode, training physics/reward config or direct-NN mode differs')
                require(public_pre_action_snapshot(env.state)==event['pre_state_public'],'private/public pre-state disconnected')
                proposed=event['proposed_actions']
                require(set(proposed)==set(AGENTS),'physical action proposal missing a role')
                _,rewards,terminated,truncated,info=env.step(proposed)
                require(rewards==event['rewards'] and terminated==event['terminated'] and truncated==event['truncated']
                    and sorted(info['shutdowns'])==sorted(event['shutdowns']) and info['proposed_actions']==proposed
                    and not info['action_override_agents'] and event['action_override_agents']==[]
                    and public_pre_action_snapshot(env.state)==event['post_state_public'],'environment reward, outcome or physical transition differs')
                if index>=n-8:ends[slot]=capture_environment(env)
                if terminated or truncated:
                    complete+=1;self.environments[slot]=None;self.active[slot]=None
            require(resets==set(records['resets']),'unused or omitted reset snapshots')
            for slot,ref in enumerate(records['end_snapshots']):
                expected=bound(ref)
                require(expected==ends[slot],'physical end state or RNG differs')
            for ref in refs:require(pointer(ref['path'])==ref,'physical asset changed during replay')
            self.asset_refs.extend(refs);self.verify()
            require(training_fingerprint(state)==before,'environment replay changed NN/critic/optimizer state')
            batch,public_audit=self.public.assemble(trace_ref,state)
            self.verify()
            require(training_fingerprint(state)==before,'public replay changed NN/critic/optimizer state')
            return batch,dict(version=VERSION,owner=self.owner_ref,inventory=self.inventory_ref,trace=trace_ref,
                physics_asset_manifest_sha256=digest(refs),public_NN_audit=public_audit,physical_frames=n,
                complete_episodes=complete,source_sha256=digest(self.sources),private_state_used_as_NN_input=False,
                environment_constructors=0,new_environment_resets=0,new_scenario_streams=0,optimizer_steps=0,
                proves_environment_implementation_consistency_not_independent_physics_model=True,
                reset_freshness_and_training_authority_must_be_checked_by_owner=True,training_authorized=False)
        except BaseException:
            self.failed=True;raise

    def snapshot(self,state):
        self.verify()
        value=dict(version=VERSION,source_files=self.sources,owner=self.owner_ref,inventory=self.inventory_ref,
            public=self.public.snapshot(state),environments=[None if e is None else capture_environment(e) for e in self.environments],
            active=deepcopy(self.active),used_episodes=sorted(self.used_episodes),asset_refs=deepcopy(self.asset_refs),training_authorized=False)
        return dict(payload=value,payload_sha256=digest(value))

    @classmethod
    def restore(cls,snapshot,state,*,owner_ref,inventory_ref,config=PPOConfig()):
        p=snapshot['payload']
        require(snapshot['payload_sha256']==digest(p) and p['version']==VERSION and p['source_files']==source_files()
            and p['owner']==owner_ref and p['inventory']==inventory_ref and p['training_authorized'] is False,
            'physics restore source, owner or inventory differs')
        out=cls(owner_ref,inventory_ref,config=config)
        out.public=PublicRolloutReplay.restore(p['public'],state,owner_ref=owner_ref,config=config)
        require(len(p['environments'])==len(p['active'])==8,'physics restore needs eight slots')
        out.environments=[None if e is None else restore_environment(e) for e in p['environments']]
        out.active=deepcopy(p['active']);out.used_episodes=set(p['used_episodes']);out.asset_refs=deepcopy(p['asset_refs'])
        require(out.used_episodes==out.public.used_episodes and out.active==out.public.active,'physics/public episode history differs')
        for slot,env in enumerate(out.environments):
            require((env is None)==(out.active[slot] is None),'physics active slot differs')
            if env is not None:require(digest(public_pre_action_snapshot(env.state))==out.public.last_public[slot],
                'restored private/public histories differ')
            if env is not None:
                require(out.cases.get(out.active[slot]['episode_id'])==out.active[slot] and out.public.receipts,
                    'restored private state outside recorded finite inventory')
                last=bound(out.public.receipts[-1]);ref=last['physics']['end_snapshots'][slot]
                require(pointer(ref['path'])==ref and capture_environment(env)==bound(ref),'restored private state or RNG differs from recorded end state')
        out.verify();return out
