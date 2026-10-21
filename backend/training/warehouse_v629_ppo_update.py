"""Environment-free continuation PPO update core, not a training capability.

Reuses the frozen reward GAE, complete-event cost and training-state helpers.
The admitted owner must supply audited rollouts, training-only buffers, finite
seed authority, persistence and recovery. This module constructs no RNG/env
and exports no command that can start a scientific candidate.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import torch
from torch.nn import functional as F
from .warehouse_v625_actor_training import (PPOConfig,ReferenceTrainingBuffer,
    complete_episode_cost,reward_gae,critic_initialization_sha256)
from .warehouse_v6_actor_training import role_hashes
from .warehouse_v6_actor import FEATURE_DIM
from .warehouse_v6_shutdown_cost import (ShutdownRiskDual,RiskDualConfig,
    joint_shutdown_surrogate,episode_balanced_cost_critic_loss)
from .warehouse_v622_actor_training import (TrainingState,ReferenceRow,
    masked_log_probabilities,reference_logits_encoded,reference_kl_loss,masked_reference_kl)
from .warehouse_v629_training_feedback import feedback_loss,coefficient
from env.warehouse.public_observation_v6 import OBSERVATION_DIM
from env.warehouse.pure_nn import AGENTS
from experiments.warehouse.v626_protocol import require,file_sha

VERSION='warehouse-v629-continuation-ppo-update.v1'
ROLES=tuple(AGENTS)


def tensor(value):return torch.as_tensor(np.asarray(value),dtype=torch.float32)


def validate_row(row,*,cost=False):
    require(isinstance(row,dict) and isinstance(row.get('episode_id'),str) and row['episode_id']
        and type(row.get('frame')) is int and 0<=row['frame']<120,'physical row identity missing')
    for key,shape in (('obs',(2,OBSERVATION_DIM)),('encoded',(2,FEATURE_DIM)),('mask',(2,5)),
            ('old_logp',(2,)),('reward_advantage',(2,)),('reward_return',(2,))):
        if cost and key in ('reward_advantage','reward_return'):continue
        value=np.asarray(row[key]);require(value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all(),
            'invalid physical row '+key)
    mask=np.asarray(row['mask']);actions=np.asarray(row['actions'])
    require(np.isin(mask,(0,1)).all() and (mask[:,4]==1).all() and actions.shape==(2,)
        and actions.dtype==np.int64 and np.all((actions>=0)&(actions<5))
        and np.all(mask[np.arange(2),actions]==1) and np.all(np.asarray(row['old_logp'])<=0),
        'physical action or behavior log probability invalid')
    for key in ('shutdown','terminated','truncated'):require(type(row[key]) is bool,'physical terminal flags required')
    require(not(row['terminated'] and row['truncated']) and row['shutdown']==row['terminated'],'real shutdown/termination mismatch')
    require(type(row['cost_value']) in (float,int) and math.isfinite(row['cost_value']) and 0<=row['cost_value']<=1,
        'cost baseline must be rollout-time sigmoid probability')
    return row


def complete_cost_rows(rows):
    """Reconstruct labels from real full events; do not trust supplied costs."""
    groups={}
    for row in rows:
        validate_row(row,cost=True);groups.setdefault(row['episode_id'],[]).append(row)
    output=[];outcomes={}
    for key,episode in groups.items():
        require([r['frame'] for r in episode]==list(range(len(episode))) and not any(
            r['terminated'] or r['truncated'] for r in episode[:-1]),'cost rows are not one contiguous complete episode')
        derived,outcome=complete_episode_cost(episode,terminated=episode[-1]['terminated'],truncated=episode[-1]['truncated'])
        for old,new in zip(episode,derived,strict=True):
            for field in ('cost_return','cost_advantage','episode_length'):
                require(field not in old or old[field]==new[field],'supplied cost labels differ from real episode events')
        output.extend(derived);outcomes[key]=outcome
    return output,outcomes


class PPOUpdateCore:
    """Four PPO epochs, with current-NN proof before each feedback update.

    This is a numerical kernel. Constructing it does not authorize collection,
    candidate initialization or scientific training. The source-bound owner
    must verify receipt and scenario membership before calling update_batch.
    """
    def __init__(self,state,reference_buffer,*,variant,rng,config=PPOConfig(),processed_joint_steps=0):
        require(isinstance(state,TrainingState) and isinstance(reference_buffer,ReferenceTrainingBuffer)
            and isinstance(rng,np.random.Generator) and variant in ('raw','regularized'), 'typed state, buffer, variant and caller-owned RNG required')
        require(type(processed_joint_steps) is int and 0<=processed_joint_steps<=state.joint_steps,'invalid processed rollout counter')
        reference_buffer.validate()
        require(reference_buffer.source_path is not None and file_sha(reference_buffer.source_path)==reference_buffer.source_sha256,
            'reference buffer requires actual bound training file')
        verified=ReferenceTrainingBuffer.load(reference_buffer.source_path,reference_buffer.source_sha256,state.reference_sha256)
        require(np.array_equal(verified.encoded,reference_buffer.encoded) and np.array_equal(verified.masks,reference_buffer.masks)
            and verified.episode_ids==reference_buffer.episode_ids,'reference buffer differs from bound file')
        require(all(group['lr']==config.actor_learning_rate for group in state.actor_optimizer.param_groups)
            and all(group['lr']==config.critic_learning_rate for group in state.critic_optimizer.param_groups),'registered optimizer learning rates required')
        self.state,self.buffer,self.rng,self.config,self.variant=state,reference_buffer,rng,config,variant
        self.processed_joint_steps=processed_joint_steps;self.seen_complete_episodes=set();self.seen_batch_receipts=set();self.risk=ShutdownRiskDual(RiskDualConfig())
        self.update_sequence=0;self.failed=False
        self.accepted_program_updates={r:0 for r in ROLES}

    def reference_mix(self,rows,batch_sha256):
        n=len(rows);require(n>0 and n%2==0,'reference mix requires even joint minibatch')
        old=self.rng.integers(len(self.buffer.encoded),size=n//2);current=self.rng.choice(n,size=n//2,replace=False)
        encoded=np.concatenate((self.buffer.encoded[old],np.asarray([rows[i]['encoded'] for i in current])))
        masks=np.concatenate((self.buffer.masks[old],np.asarray([rows[i]['mask'] for i in current])))
        provenance=[ReferenceRow('reference_train','train',self.buffer.episode_ids[i],self.buffer.source_sha256) for i in old]
        provenance.extend(ReferenceRow('current_train','train',rows[i]['episode_id'],batch_sha256) for i in current)
        return tensor(encoded),tensor(masks),provenance

    def _cost_loss(self,rows,outcomes,logits):
        if not rows:return logits[...,0].mul(0).sum(),None
        x=tensor([r['encoded'] for r in rows]);m=tensor([r['mask'] for r in rows])
        logp=masked_log_probabilities(self.state.model.outputs_encoded(x,m),m)
        actions=torch.as_tensor(np.asarray([r['actions'] for r in rows]),dtype=torch.long)
        chosen=logp.gather(-1,actions[...,None]).squeeze(-1)
        ids={key:i for i,key in enumerate(outcomes)};labels=torch.tensor([ids[r['episode_id']] for r in rows])
        loss,_=joint_shutdown_surrogate(chosen,tensor([r['old_logp'] for r in rows]),
            tensor([r['cost_advantage'] for r in rows]),labels,list(ids.values()),clip=self.config.clip,reference_horizon=120)
        cost_logits=self.state.model.cost_critic(tensor([r['obs'] for r in rows]).reshape(len(rows),-1)).squeeze(-1)
        critic=episode_balanced_cost_critic_loss(cost_logits,labels,{ids[key]:value for key,value in outcomes.items()})
        return loss,critic

    def _minibatch(self,rows,costs,outcomes,*,batch_sha256,actor_enabled,epoch_id,evidence_provider):
        state=self.state;x=tensor([r['encoded'] for r in rows]);m=tensor([r['mask'] for r in rows])
        actions=torch.as_tensor(np.asarray([r['actions'] for r in rows]),dtype=torch.long)
        evidence=None
        if actor_enabled and coefficient(state.joint_steps,self.config.joint_step_budget,self.variant)>0:
            require(callable(evidence_provider),'current-NN evidence provider required before feedback')
            before=role_hashes(state.model);critics=critic_initialization_sha256(state.model)
            counters=(state.joint_steps,state.actor_updates,state.critic_updates)
            evidence=evidence_provider(state,epoch_id)
            require(role_hashes(state.model)==before and critic_initialization_sha256(state.model)==critics
                and counters==(state.joint_steps,state.actor_updates,state.critic_updates),'evidence provider changed NN, critics or training counters')
        feedback=feedback_loss(state.model,x,m,actions,joint_steps=state.joint_steps,total_steps=self.config.joint_step_budget,
            variant=self.variant,epoch_id=epoch_id,evidence=evidence)
        logp=masked_log_probabilities(feedback.logits,m);chosen=logp.gather(-1,actions[...,None]).squeeze(-1)
        ratio=(chosen-tensor([r['old_logp'] for r in rows])).exp();adv=tensor([r['normalized_advantage'] for r in rows])
        reward=-torch.minimum(ratio*adv,ratio.clamp(1-self.config.clip,1+self.config.clip)*adv).mean(0).sum()
        entropy=-(logp.exp()*logp.masked_fill(m==0,0.)).sum(-1).mean(0).sum()
        predicted=state.model.reward_critic(tensor([r['obs'] for r in rows]).reshape(len(rows),-1))
        critic=self.config.critic_coefficient*F.mse_loss(predicted,tensor([r['reward_return'] for r in rows]))
        cost,cost_critic=self._cost_loss(costs,outcomes,feedback.logits)
        if cost_critic is not None:critic=critic+self.config.critic_coefficient*cost_critic
        reference_loss=None
        if actor_enabled:
            ref_x,ref_m,provenance=self.reference_mix(rows,batch_sha256)
            reference=reference_logits_encoded(state,ref_x)
            reference_loss=reference_kl_loss(state.model.outputs_encoded(ref_x,ref_m),reference,ref_m,provenance=provenance,beta=state.beta)
            total=reward+self.risk.multiplier*cost-self.config.entropy_coefficient*entropy+reference_loss+feedback.program_loss.loss
            require(bool(torch.isfinite(total)),'nonfinite total Actor objective; no optimizer step accepted')
        require(bool(torch.isfinite(critic)),'nonfinite critic objective; no optimizer step accepted')
        before=role_hashes(state.model);actor_before=state.actor_updates;critic_before=state.critic_updates
        parameters={r:[p.detach().clone() for p in state.model.actors[r].parameters()] for r in ROLES}
        state.critic_step(critic)
        report=dict(epoch_id=epoch_id,physical_frames=len(rows),actor_updated=False,critic_loss=float(critic.detach()),
            complete_cost_frames=len(costs),complete_cost_episodes=len(outcomes),partial_episodes_labeled_negative=0,
            program_gradient=feedback.diagnostics,accepted_positive_program_gradient={r:False for r in ROLES})
        if actor_enabled:
            feedback.verify_before_optimizer_step(state.model,current_epoch_id=epoch_id)
            def observed():
                with torch.no_grad():
                    value=masked_reference_kl(state.model.outputs_encoded(ref_x,ref_m),reference,ref_m).mean(0)
                return {r:float(value[i]) for i,r in enumerate(ROLES)}
            accepted=state.actor_step(total,observed_reference_kl=observed,max_grad_norm=self.config.maximum_gradient_norm)
            require(state.actor_updates==actor_before+1,'Actor optimizer did not accept exactly one update')
            after=role_hashes(state.model);clip=min(1.,self.config.maximum_gradient_norm/(accepted['gradient_norm']+1e-6))
            delta={r:math.sqrt(sum(float((p.detach()-old).double().square().sum()) for p,old in
                zip(state.model.actors[r].parameters(),parameters[r],strict=True))) for r in ROLES}
            positive={r:feedback.diagnostics['role_program_gradient_norm_before_shared_clip'][r]>0 and clip>0
                and delta[r]>0 and after[r]!=before[r] for r in ROLES}
            report.update(actor_updated=True,actor=accepted,role_sha256_before=before,role_sha256_after=after,
                role_weight_delta_l2=delta,common_gradient_clip_factor=clip,accepted_positive_program_gradient=positive,
                combined_update_not_causal_program_attribution=True,reference_loss=float(reference_loss.detach()),
                reward_loss=float(reward.detach()),shutdown_surrogate=float(cost.detach()),entropy=float(entropy.detach()),
                program_loss=float(feedback.program_loss.loss.detach()))
        else:
            require(state.actor_updates==actor_before and role_hashes(state.model)==before,'critic warmup changed an Actor')
        require(state.critic_updates==critic_before+1,'critic optimizer did not accept exactly one update')
        return report

    def update_batch(self,batch,*,batch_sha256,evidence_provider=None):
        """Consume one already audited rollout, fail permanently after an error."""
        require(not self.failed,'failed numerical attempt cannot silently resume')
        try:
            require(isinstance(batch_sha256,str) and len(batch_sha256)==64 and all(c in '0123456789abcdef' for c in batch_sha256),
                'owner-bound rollout receipt SHA required')
            require(batch_sha256 not in self.seen_batch_receipts,'rollout receipt already consumed')
            require(batch['start_joint_step']==self.processed_joint_steps and batch['end_joint_step']==self.state.joint_steps
                and 0<batch['end_joint_step']-batch['start_joint_step']==len(batch['rows'])<=1024
                and len(batch['rows'])%8==0 and batch['end_joint_step']<=self.config.joint_step_budget,
                'rollout step counters, row count or registered budget differ')
            require(not(batch['start_joint_step']<4000<batch['end_joint_step']),'rollout crossed exact critic-only boundary')
            rows=deepcopy(batch['rows']);identities=[]
            for row in rows:validate_row(row);identities.append((row['episode_id'],row['frame']))
            require(len(set(identities))==len(identities),'physical rollout frame duplicated')
            grouped={}
            for row in rows:grouped.setdefault(row['episode_id'],[]).append(row)
            for group in grouped.values():
                frames=[r['frame'] for r in group]
                require(frames==list(range(frames[0],frames[0]+len(frames))) and not any(
                    r['terminated'] or r['truncated'] for r in group[:-1]),'rollout continued or skipped frames within an episode')
            costs,outcomes=complete_cost_rows(deepcopy(batch['complete_cost_rows']))
            terminal_episodes={r['episode_id'] for r in rows if r['terminated'] or r['truncated']}
            require(terminal_episodes==set(outcomes),'rollout terminal episodes missing complete cost history or extra outcomes')
            current={key:row for key,row in zip(identities,rows,strict=True)}
            for row in costs:
                key=(row['episode_id'],row['frame'])
                if row['terminated'] or row['truncated']:require(key in current,'completed cost episode did not finish in this rollout')
                if key in current:
                    require(all(np.array_equal(row[k],current[key][k]) for k in ('obs','encoded','mask','actions','old_logp'))
                        and all(row[k]==current[key][k] for k in ('shutdown','terminated','truncated')),
                        'complete cost transition differs from actual PPO rollout')
                    require(row['cost_value']==current[key]['cost_value'],'complete cost baseline differs from rollout-time critic')
            require(not self.seen_complete_episodes&outcomes.keys(),'complete episode risk outcome counted twice')
            self.seen_batch_receipts.add(batch_sha256)
            self.seen_complete_episodes.update(outcomes)
            risk=self.risk.update([(True,outcome) for outcome in outcomes.values()])
            advantages=np.asarray([r['reward_advantage'] for r in rows],np.float32)
            normalized=(advantages-advantages.mean(0))/(advantages.std(0)+1e-8)
            for row,value in zip(rows,normalized,strict=True):row['normalized_advantage']=value
            report=dict(version=VERSION,batch_sha256=batch_sha256,start_joint_step=batch['start_joint_step'],end_joint_step=self.state.joint_steps,
                risk=risk,complete_episodes=len(outcomes),shutdown_episodes=sum(outcomes.values()),
                no_positive_shutdown_examples_observed=not any(outcomes.values()),partial_episodes_labeled_negative=0,epochs=[])
            actor_enabled=batch['start_joint_step']>=4000
            for epoch in range(self.config.epochs):
                order=self.rng.permutation(len(rows));updates=[]
                for offset in range(0,len(order),self.config.minibatch_joint_frames):
                    identity=f'batch-{self.update_sequence}/epoch-{epoch}/minibatch-{offset}'
                    chosen=[rows[i] for i in order[offset:offset+self.config.minibatch_joint_frames]]
                    update=self._minibatch(chosen,costs,outcomes,batch_sha256=batch_sha256,actor_enabled=actor_enabled,
                        epoch_id=identity,evidence_provider=evidence_provider)
                    updates.append(update)
                    for role in ROLES:self.accepted_program_updates[role]+=int(update['accepted_positive_program_gradient'][role])
                report['epochs'].append(dict(epoch=epoch,updates=updates))
            self.processed_joint_steps=self.state.joint_steps;self.update_sequence+=1
            report['accepted_positive_program_gradient_updates']={r:sum(int(u['accepted_positive_program_gradient'][r])
                for e in report['epochs'] for u in e['updates']) for r in ROLES}
            return report
        except BaseException:
            self.failed=True;raise

    def snapshot(self):
        """Caller stores this alongside full TrainingState; no restore authority."""
        return dict(version=VERSION,config=asdict(self.config),variant=self.variant,processed_joint_steps=self.processed_joint_steps,
            update_sequence=self.update_sequence,failed=self.failed,risk=self.risk.snapshot(),
            seen_complete_episodes=sorted(self.seen_complete_episodes),rng_state=deepcopy(self.rng.bit_generator.state),
            seen_batch_receipts=sorted(self.seen_batch_receipts),
            safe_batch_boundary=self.processed_joint_steps==self.state.joint_steps and not self.failed,
            source_sha256=file_sha(__file__),reference_sha256=self.state.reference_sha256,
            reference_buffer_sha256=self.buffer.source_sha256,actor_role_sha256=role_hashes(self.state.model),
            critic_sha256=critic_initialization_sha256(self.state.model),actor_updates=self.state.actor_updates,
            critic_updates=self.state.critic_updates,beta=dict(self.state.beta),accepted_program_updates=dict(self.accepted_program_updates))

    @classmethod
    def restore(cls,state,reference_buffer,value,*,variant,rng,config=PPOConfig()):
        """Only restore a completed boundary from a full training checkpoint.

        An owner still needs a new explicit continuation capability. This
        method does not resume an interrupted candidate or construct a RNG.
        """
        require(state.resumed_training_state and value['version']==VERSION and value['config']==asdict(config)
            and value['variant']==variant
            and value['source_sha256']==file_sha(__file__) and value['safe_batch_boundary'] is True
            and value['failed'] is False and value['processed_joint_steps']==state.joint_steps,
            'core resume requires the same source and completed full-training checkpoint')
        require(value['reference_sha256']==state.reference_sha256 and value['reference_buffer_sha256']==reference_buffer.source_sha256
            and value['actor_role_sha256']==role_hashes(state.model) and value['critic_sha256']==critic_initialization_sha256(state.model)
            and value['actor_updates']==state.actor_updates and value['critic_updates']==state.critic_updates
            and value['beta']==state.beta,'core and training-state checkpoint differ')
        ids=value['seen_complete_episodes']
        receipts=value['seen_batch_receipts']
        feedback=value['accepted_program_updates']
        require(isinstance(ids,list) and ids==sorted(set(ids)) and all(isinstance(s,str) and s for s in ids)
            and isinstance(receipts,list) and receipts==sorted(set(receipts)) and all(isinstance(s,str) and len(s)==64
                and all(c in '0123456789abcdef' for c in s) for s in receipts)
            and type(value['update_sequence']) is int and len(receipts)==value['update_sequence']>=0,'invalid restored episode/update ledger')
        require(set(feedback)==set(ROLES) and all(type(n) is int and 0<=n<=state.actor_updates for n in feedback.values())
            and (variant!='raw' or not any(feedback.values())),'invalid restored accepted-program-update counts')
        out=cls(state,reference_buffer,variant=value['variant'],rng=rng,config=config,processed_joint_steps=state.joint_steps)
        out.risk=ShutdownRiskDual.restore(value['risk'],RiskDualConfig());out.seen_complete_episodes=set(ids)
        out.seen_batch_receipts=set(receipts)
        out.accepted_program_updates=dict(feedback)
        out.update_sequence=value['update_sequence'];out.rng.bit_generator.state=deepcopy(value['rng_state'])
        return out
