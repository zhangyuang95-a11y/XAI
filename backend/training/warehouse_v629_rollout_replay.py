"""Public-frame PPO assembly; no environment or training authority.

The admitted collector must bind finite train case/reset membership, original
environment rewards and physics configuration. This independent reader binds
the submitted trace to the current complete NN, reconstructs inputs/actions
and critic baselines, and carries real full cost episodes across PPO batches.
It cannot infer reset authorization or reward correctness from public states.
Action draws are replayed from the submitted episode's existing seed stream.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import math
import numpy as np
import torch
from backend.training.warehouse_v625_actor_training import PPOConfig,complete_episode_cost,reward_gae,critic_initialization_sha256
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,encode_actor_tree_features
from backend.training.warehouse_v6_actor_training import role_hashes
from backend.training.warehouse_v622_program import masked_probabilities
from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state,encode_public_observation
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS,action_masks
from experiments.warehouse.v626_protocol import require,pointer,bound,file_sha,digest

VERSION='warehouse-v629-public-rollout-replay.v1'
TRACE_VERSION='warehouse-v629-collected-PPO-public-frames.v1'
ROLES=tuple(AGENTS)


def training_fingerprint(state):
    """Bitwise full state identity, excluding only the fresh/resumed marker."""
    h=sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            a=value.detach().cpu().numpy();raw=np.ascontiguousarray(a).tobytes()
            h.update(('tensor:'+a.dtype.str+':'+str(a.shape)+':'+str(len(raw))+':').encode());h.update(raw)
        elif isinstance(value,dict):
            h.update(('dict:'+str(len(value))+':').encode())
            for key in sorted(value,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(value[key])
            h.update(b'end-dict;')
        elif isinstance(value,(list,tuple)):
            h.update((type(value).__name__+':'+str(len(value))+':').encode())
            for item in value:visit(item)
        else:
            import json
            require(value is None or type(value) in (str,int,float,bool),'unsupported full training state value')
            h.update((type(value).__name__+':'+json.dumps(value,allow_nan=False,separators=(',',':'))+';').encode())
    require(file_sha(state.reference_path)==state.reference_sha256,'full state reference artifact changed')
    visit(dict(model=state.model.state_dict(),actor_optimizer=state.actor_optimizer.state_dict(),
        critic_optimizer=state.critic_optimizer.state_dict(),trainability={n:p.requires_grad for n,p in state.model.named_parameters()},
        reference_sha256=state.reference_sha256,initialization_seed=state.initialization_seed,beta=state.beta,
        joint_steps=state.joint_steps,actor_updates=state.actor_updates,critic_updates=state.critic_updates))
    return h.hexdigest()


def critic_values(model,obs):
    with torch.no_grad():
        x=torch.as_tensor(np.asarray(obs,np.float32).reshape(1,-1))
        rewards=model.reward_critic(x)[0].cpu().numpy().copy()
        cost=float(model.cost_critic(x).sigmoid()[0,0])
    require(rewards.shape==(2,) and np.isfinite(rewards).all() and math.isfinite(cost),'nonfinite rollout critic')
    return rewards,cost


def arrays_to_lists(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:arrays_to_lists(v) for k,v in value.items()}
    if isinstance(value,list):return [arrays_to_lists(v) for v in value]
    return value


def restore_row(row):
    row=deepcopy(row)
    for key in ('obs','encoded','mask','old_logp','rewards','values','next_values'):
        row[key]=np.asarray(row[key],np.float32)
    row['actions']=np.asarray(row['actions'],np.int64)
    return row


class PublicRolloutReplay:
    """Finite trace consumer, not a collector or numerical update owner.

    Source receipts are consumed once. A partial failure closes this reader;
    only completed assembly boundaries can be explicitly restored alongside
    the owner's full state and admitted continuation capability.
    """
    def __init__(self,owner_ref,*,config=PPOConfig()):
        require(pointer(owner_ref['path'])==owner_ref,'missing bound rollout owner record')
        self.owner_ref=deepcopy(owner_ref)
        self.config=config;self.processed_joint_steps=0;self.pending=[[] for _ in range(8)]
        self.active=[None]*8;self.last_public=[None]*8;self.used_episodes=set();self.receipts=[];self.failed=False

    def _frame(self,event,actor,state,slot):
        require(type(event['slot']) is int and event['slot']==slot,'rollout vectors must contain slots zero through seven in order')
        case=event['case']
        require(set(case)=={'episode_id','scenario_seed','task','action_seed','slot','partition'}
            and case['partition']=='train' and type(case['slot']) is int and case['slot']==slot and type(case['task']) is int and case['task'] in (1,2,3)
            and isinstance(case['episode_id'],str) and case['episode_id']
            and all(type(case[k]) is int and 0<=case[k]<2**63 for k in ('scenario_seed','action_seed')),
            'invalid bound training case identity')
        pre=event['pre_state_public'];post=event['post_state_public']
        before=rebuild_public_pre_action_state(pre);after=rebuild_public_pre_action_state(post)
        require(after.episode_id==before.episode_id and after.frame==before.frame+1<=120
            and all(before.by_id(r).active for r in ROLES),'public physical frame does not advance once or continues after shutdown')
        if self.active[slot] is None:
            require(before.frame==0 and case['episode_id'] not in self.used_episodes,'new rollout episode must begin at zero and never repeat')
            self.active[slot]=deepcopy(case);self.used_episodes.add(case['episode_id'])
        else:
            require(case==self.active[slot] and digest(pre)==self.last_public[slot],'rollout public history or active scenario disconnected')
        require(before.frame==len(self.pending[slot]),'complete cost history has missing physical frames')
        obs=encode_public_observation(before);mask=action_masks(before);encoded=encode_actor_tree_features(obs,mask)
        p=masked_probabilities(actor.logits(obs,mask),mask);actions=[]
        for i,role in enumerate(ROLES):
            seed=independent_agent_seed(base_seed=case['action_seed'],episode_id=int(before.episode_id),frame=before.frame,agent_id=role)
            action=int(np.random.default_rng(seed).choice(5,p=p[i]));actions.append(action)
        requested={role:ACTIONS[actions[i]] for i,role in enumerate(ROLES)}
        require(event['proposed_actions']==requested and event['action_override_agents']==[]
            and {a['agent_id']:a['last_action'] for a in post['agents']}==requested,'executed proposal is not bound NN plus physical mask')
        for key in ('terminated','truncated'):require(type(event[key]) is bool,'physical terminal event missing')
        inactive=[role for role in ROLES if not after.by_id(role).active]
        require(type(event['shutdowns']) is list and sorted(event['shutdowns'])==inactive
            and event['terminated']==bool(inactive) and not(event['terminated'] and event['truncated'])
            and event['truncated']==(after.frame==120 and not inactive),'shutdown or horizon label differs from public terminal state')
        require(set(event['rewards'])==set(ROLES) and all(type(v) in (float,int) and math.isfinite(v) for v in event['rewards'].values()),
            'original environment reward receipt missing or nonfinite')
        values,cost=critic_values(state.model,obs);next_values,_=critic_values(state.model,encode_public_observation(after))
        row=dict(obs=obs,encoded=encoded,mask=mask,actions=np.asarray(actions,np.int64),
            old_logp=np.log(p[np.arange(2),actions]).astype(np.float32),rewards=np.asarray([event['rewards'][r] for r in ROLES],np.float32),
            values=values,next_values=next_values,cost_value=cost,shutdown=bool(inactive),terminated=event['terminated'],
            truncated=event['truncated'],episode_id=case['episode_id'],frame=before.frame,slot=slot)
        self.pending[slot].append(deepcopy(row));self.last_public[slot]=digest(post)
        complete=[]
        if row['terminated'] or row['truncated']:
            complete,_=complete_episode_cost(self.pending[slot],terminated=row['terminated'],truncated=row['truncated'])
            self.pending[slot]=[];self.active[slot]=None;self.last_public[slot]=None
        return row,complete

    def assemble(self,trace_ref,state):
        require(not self.failed,'failed rollout reader cannot silently resume')
        try:
            require(pointer(self.owner_ref['path'])==self.owner_ref,'rollout owner changed')
            require(pointer(trace_ref['path'])==trace_ref and trace_ref not in self.receipts,'rollout trace changed or consumed twice')
            trace=bound(trace_ref);start=self.processed_joint_steps;events=trace['events'];n=len(events)
            require(trace['schema']==TRACE_VERSION and trace['owner']==self.owner_ref and trace['partition']=='train' and trace['start_joint_step']==start
                and trace['end_joint_step']==state.joint_steps==start+n
                and 0<n==8*self.config.next_collection_length(start) and n%8==0
                and trace['config']==asdict(self.config),'rollout trace count, boundary or registered config differs')
            expected=dict(actor_role_sha256=role_hashes(state.model),critic_sha256=critic_initialization_sha256(state.model),
                actor_updates=state.actor_updates,critic_updates=state.critic_updates)
            require(trace['rollout_state']==expected,'rollout sampled using another Actor or critic version')
            actor_ref=trace['actor'];require(pointer(actor_ref['path'])==actor_ref,'rollout frozen Actor asset changed')
            actor=FrozenWarehouseV6Actor.load(actor_ref['path'],expected_sha256=actor_ref['sha256'],expected_role_sha256=expected['actor_role_sha256'])
            rows=[];complete=[]
            for i,event in enumerate(events):
                row,finished=self._frame(event,actor,state,i%8);rows.append(row);complete.extend(finished)
            shape=(n//8,8)
            rewards=np.asarray([r['rewards'] for r in rows]).reshape(*shape,2)
            values=np.asarray([r['values'] for r in rows]).reshape(*shape,2)
            next_values=np.asarray([r['next_values'] for r in rows]).reshape(*shape,2)
            terminal=np.asarray([r['terminated'] for r in rows]).reshape(shape)
            trunc=np.asarray([r['truncated'] for r in rows]).reshape(shape)
            advantages,returns=reward_gae(rewards,values,next_values,terminal,trunc,self.config)
            for row,a,r in zip(rows,advantages.reshape(n,2),returns.reshape(n,2),strict=True):
                row['reward_advantage']=a;row['reward_return']=r
            require(pointer(trace_ref['path'])==trace_ref and pointer(self.owner_ref['path'])==self.owner_ref
                and pointer(actor_ref['path'])==actor_ref and expected==dict(actor_role_sha256=role_hashes(state.model),
                critic_sha256=critic_initialization_sha256(state.model),actor_updates=state.actor_updates,critic_updates=state.critic_updates),
                'trace, NN or critic changed during independent replay')
            self.processed_joint_steps=state.joint_steps;self.receipts.append(deepcopy(trace_ref))
            batch=dict(start_joint_step=start,end_joint_step=state.joint_steps,rows=rows,complete_cost_rows=complete)
            audit=dict(version=VERSION,owner=deepcopy(self.owner_ref),trace=deepcopy(trace_ref),actor=actor_ref,rollout_state=expected,physical_frames=n,NN_action_checks=2*n,
                complete_episodes=len({r['episode_id'] for r in complete}),pending_frames=[len(p) for p in self.pending],
                partial_episodes_labeled_negative=0,environment_resets=0,actor_updates=0,PPO_joint_steps=0,
                training_authorized=False,reward_and_seed_authority_must_be_checked_by_owner=True)
            return batch,audit
        except BaseException:
            self.failed=True;raise

    def snapshot(self,state):
        require(not self.failed,'cannot snapshot failed or partially consumed rollout')
        require(pointer(self.owner_ref['path'])==self.owner_ref and state.joint_steps==self.processed_joint_steps,
            'snapshot owner or completed joint counter differs')
        value=dict(version=VERSION,source_sha256=file_sha(__file__),owner=deepcopy(self.owner_ref),
            full_training_state_sha256=training_fingerprint(state),config=asdict(self.config),processed_joint_steps=self.processed_joint_steps,
            pending=arrays_to_lists(self.pending),active=deepcopy(self.active),last_public=list(self.last_public),
            used_episodes=sorted(self.used_episodes),receipts=deepcopy(self.receipts),safe_assembly_boundary=True,authorizes_training=False)
        return dict(payload=value,payload_sha256=digest(value))

    @classmethod
    def restore(cls,snapshot,state,*,owner_ref,config=PPOConfig()):
        value=snapshot['payload']
        require(snapshot['payload_sha256']==digest(value) and value['version']==VERSION and value['source_sha256']==file_sha(__file__)
            and value['config']==asdict(config) and value['safe_assembly_boundary'] is True
            and state.resumed_training_state and value['processed_joint_steps']==state.joint_steps
            and value['owner']==owner_ref and pointer(owner_ref['path'])==owner_ref
            and value['full_training_state_sha256']==training_fingerprint(state),
            'rollout restore requires matching source and full completed training state')
        for ref in value['receipts']:require(pointer(ref['path'])==ref,'prior physical trace changed before restore')
        out=cls(owner_ref,config=config);out.processed_joint_steps=value['processed_joint_steps']
        out.pending=[[restore_row(row) for row in group] for group in value['pending']]
        out.active=deepcopy(value['active']);out.last_public=list(value['last_public']);out.used_episodes=set(value['used_episodes']);out.receipts=deepcopy(value['receipts'])
        require(len(out.pending)==len(out.active)==len(out.last_public)==8,'invalid eight-slot pending history')
        for i,group in enumerate(out.pending):
            require((out.active[i] is None)==(not group)==(out.last_public[i] is None)
                and len(group)<120 and [r['frame'] for r in group]==list(range(len(group)))
                and all(r['slot']==i and r['episode_id']==out.active[i]['episode_id'] and not r['terminated'] and not r['truncated'] for r in group),
                'pending episode snapshot is incomplete or inconsistent')
            if group:require(out.active[i]['episode_id'] in out.used_episodes,'active snapshot episode omitted from consumption ledger')
        return out
