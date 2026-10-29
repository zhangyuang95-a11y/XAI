"""Configurable v6.43 cost-critic numerical kernel; no rollout authority.

Only real complete TRAIN episodes enter the buffer. Critic replay retains the
observed behavior-policy outcome, not an independent current-policy risk label.
The buffer never retains actions, log probabilities, or Actor autograd graphs.
Risk multipliers consume each newly completed episode once per rollout batch;
critic replay cannot increase the observed episode count. The finite owner must
independently verify physical traces, source/seed capabilities and all run gates.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .warehouse_v6_shutdown_cost import RiskDualConfig, ShutdownRiskDual

VERSION = 'warehouse-v643-complete-episode-cost-core.v1'
BUFFER_VERSION = 'warehouse-v643-training-only-complete-episode-buffer.v1'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class RiskTrainingConfig:
    capacity_episodes: int = 256
    episode_horizon: int = 120
    episodes_per_update: int = 8
    learning_rate: float = 3e-4
    optimizer_epsilon: float = 1e-5
    maximum_gradient_norm: float = .5
    low_battery_columns: tuple = ()
    low_battery_fraction: float = .20
    risk: RiskDualConfig = field(default_factory=lambda:RiskDualConfig(
        target=.02,window_episodes=256,minimum_episodes=64,confidence=.90,
        release_upper=.01,initial_multiplier=2.))

    def __post_init__(self):
        check(type(self.capacity_episodes) is int and self.capacity_episodes>0
            and type(self.episode_horizon) is int and self.episode_horizon>0
            and type(self.episodes_per_update) is int and self.episodes_per_update>0,
            'positive finite episode limits required')
        check(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (
            self.learning_rate,self.optimizer_epsilon,self.maximum_gradient_norm)),
            'finite positive optimizer settings required')
        check(type(self.low_battery_columns) is tuple and all(type(i) is int and i>=0
            for i in self.low_battery_columns) and len(set(self.low_battery_columns))==len(self.low_battery_columns)
            and 0<self.low_battery_fraction<1,'invalid public low-battery feature columns')
        check(type(self.risk) is RiskDualConfig,'explicit immutable risk config required')
        check(type(self.risk.window_episodes) is int and type(self.risk.minimum_episodes) is int
            and all(type(v) in (int,float) and math.isfinite(v) for v in asdict(self.risk).values()),
            'finite registered risk dual settings required')


def detached_observation(value, observation_dim):
    if isinstance(value,torch.Tensor):
        value=value.detach().cpu().numpy()
    value=np.asarray(value)
    check(value.dtype==np.float32 and value.size==observation_dim
        and value.ndim in (1,2) and np.isfinite(value).all(),
        'finite public float32 critic observation required')
    return value.reshape(observation_dim).copy()


def completed_episode(rows, *, observation_dim, horizon, batch_sha256, end_joint_step):
    """Detach a whole episode and derive its label solely from terminal events.

    Extra rollout fields may be present, but only observations and event flags
    are retained. Supplying an action/logp tensor cannot give replay an Actor
    gradient path. A metadata claim is not physical trace verification.
    """
    check(type(rows) is list and rows and len(rows)<=horizon,'whole finite episode required')
    check(valid_sha(batch_sha256) and type(end_joint_step) is int and end_joint_step>0,
          'complete episode must have a finite hash-bound training boundary')
    episode=rows[0].get('episode_id')
    check(type(episode) is str and episode and len(episode)<=256,'episode identity required')
    observations=[]
    for frame,row in enumerate(rows):
        check(isinstance(row,dict) and row.get('episode_id')==episode
            and type(row.get('frame')) is int and row['frame']==frame,
            'complete episode must start at frame zero and be contiguous')
        check(row.get('partition','train')=='train','only TRAIN episode rows may enter critic replay')
        for name in ('shutdown','terminated','truncated'):
            check(type(row.get(name)) is bool,'real boolean episode events required')
        check(not(row['terminated'] and row['truncated']) and row['shutdown']==row['terminated'],
            'shutdown and physical termination disagree')
        check(frame==len(rows)-1 or not(row['terminated'] or row['truncated']),
            'episode has an earlier terminal event')
        observations.append(detached_observation(row.get('obs'),observation_dim))
    last=rows[-1]
    check(last['terminated'] != last['truncated'] and
          (not last['truncated'] or len(rows)==horizon),'incomplete episode cannot be labeled safe')
    outcome=last['shutdown']
    for row in rows:
        check('cost_return' not in row or row['cost_return']==float(outcome),
              'supplied cost target contradicts actual complete outcome')
    value=dict(episode_id=episode,observations=np.stack(observations).tolist(),shutdown=outcome,
        frames=len(rows),batch_sha256=batch_sha256,collected_joint_step=end_joint_step,
        outcome_scope='actual_behavior_policy_training_episode')
    value['episode_sha256']=digest(value)
    return value


def validate_saved_episode(value, *, observation_dim, horizon):
    keys={'episode_id','observations','shutdown','frames','batch_sha256','collected_joint_step',
          'outcome_scope','episode_sha256'}
    check(type(value) is dict and set(value)==keys,'invalid saved episode fields')
    raw={k:v for k,v in value.items() if k!='episode_sha256'}
    check(valid_sha(value['episode_sha256']) and digest(raw)==value['episode_sha256'],
          'saved training episode hash differs')
    check(type(value['episode_id']) is str and value['episode_id'] and type(value['shutdown']) is bool
        and type(value['frames']) is int and 0<value['frames']<=horizon
        and (value['shutdown'] or value['frames']==horizon)
        and valid_sha(value['batch_sha256'])
        and type(value['collected_joint_step']) is int and value['collected_joint_step']>0
        and value['outcome_scope']=='actual_behavior_policy_training_episode',
        'invalid complete episode identity or outcome')
    x=np.asarray(value['observations'],dtype=np.float32)
    check(x.shape==(value['frames'],observation_dim) and np.isfinite(x).all(),
          'saved critic observations differ')
    return deepcopy(value)


class RiskTrainingCore:
    """Training-only cost critic and risk dual, usable by a future finite owner.

    Critic parameters and optimizer must be disjoint from both Actors and reward
    critic. This kernel never updates those modules or computes their forwards.
    Sampling uses only the explicitly supplied NumPy RNG; no RNG is constructed.
    """
    def __init__(self,cost_critic,optimizer,*,forbidden_modules,observation_dim,rng,
                 config=RiskTrainingConfig()):
        check(isinstance(cost_critic,nn.Module) and isinstance(optimizer,torch.optim.Adam),
              'explicit cost critic and Adam optimizer required')
        check(type(observation_dim) is int and observation_dim>0 and isinstance(rng,np.random.Generator),
              'explicit critic dimension and caller-owned RNG required')
        check(type(config) is RiskTrainingConfig and all(i<observation_dim for i in config.low_battery_columns),
              'risk config or public feature domain differs')
        forbidden=tuple(forbidden_modules)
        check(forbidden and all(isinstance(m,nn.Module) for m in forbidden),
              'explicit Actor/reward modules must protect replay parameter ownership')
        parameters=tuple(cost_critic.parameters());owned={id(p) for p in parameters}
        protected={id(p) for m in forbidden for p in m.parameters()}
        opt=[p for group in optimizer.param_groups for p in group['params']]
        check(owned and not owned&protected and len(opt)==len({id(p) for p in opt})
            and {id(p) for p in opt}==owned,'critic replay optimizer must own only disjoint cost critic parameters')
        check(all(group['lr']==config.learning_rate and group['eps']==config.optimizer_epsilon
            and group.get('weight_decay',0)==0 and group['betas']==(.9,.999)
            and not any(group.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable'))
            and group.get('foreach') is None and group.get('fused') is None
            for group in optimizer.param_groups),
            'cost optimizer settings differ from the registered numerical config')
        # Actual Warehouse critics have Linear/ReLU layers. Disallow stochastic
        # or stateful normalization layers whose hidden RNG/statistics are absent
        # from the explicit complete numerical checkpoint contract.
        check(all(isinstance(m,(nn.Sequential,nn.Linear,nn.ReLU,nn.Tanh,nn.Sigmoid,nn.Identity))
            for m in cost_critic.modules()),'cost critic must use deterministic registered layers')
        self.critic,self.optimizer,self.forbidden=cost_critic,optimizer,forbidden
        self.observation_dim,self.rng,self.config=observation_dim,rng,config
        self.risk=ShutdownRiskDual(config.risk)
        self.episodes=OrderedDict();self.seen_episodes={};self.seen_batches=set()
        self.joint_step=0;self.critic_updates=0;self.last_risk_update=None

    def observe_batch(self,rows,*,batch_sha256,end_joint_step,partition):
        """Consume newly completed real episodes once, never partial negatives.

        Validation is atomic. Only one dual update occurs for the rollout batch,
        matching the existing update cadence. Replaying a critic minibatch does
        not call this method and cannot mint new shutdown observations.
        """
        check(partition=='train','program/validation/quality/holdout data cannot train the critic')
        check(type(rows) is list and valid_sha(batch_sha256) and batch_sha256 not in self.seen_batches,
              'new hash-bound complete-training batch required')
        check(type(end_joint_step) is int and end_joint_step>self.joint_step,
              'new physical training step boundary required')
        grouped=OrderedDict()
        for row in rows:
            check(isinstance(row,dict) and type(row.get('episode_id')) is str,'training episode identity required')
            grouped.setdefault(row['episode_id'],[]).append(row)
        episodes=[completed_episode(r,observation_dim=self.observation_dim,horizon=self.config.episode_horizon,
            batch_sha256=batch_sha256,end_joint_step=end_joint_step) for r in grouped.values()]
        check(all(e['episode_id'] not in self.seen_episodes for e in episodes),'complete episode already consumed')
        for e in episodes:
            self.episodes[e['episode_id']]=e;self.seen_episodes[e['episode_id']]=e['episode_sha256']
        while len(self.episodes)>self.config.capacity_episodes:self.episodes.popitem(last=False)
        self.last_risk_update=self.risk.update([(True,e['shutdown']) for e in episodes])
        self.seen_batches.add(batch_sha256);self.joint_step=end_joint_step
        return dict(version=VERSION,new_complete_episodes=len(episodes),
            new_shutdown_episodes=sum(e['shutdown'] for e in episodes),
            buffered_complete_episodes=len(self.episodes),buffered_shutdown_episodes=sum(e['shutdown'] for e in self.episodes.values()),
            partial_episodes_labeled_negative=0,risk=deepcopy(self.last_risk_update),
            historical_Actor_gradient_updates=0,training_authority=False)

    def _forward(self,episodes):
        for e in episodes:validate_saved_episode(e,observation_dim=self.observation_dim,horizon=self.config.episode_horizon)
        first=next(self.critic.parameters());device=first.device
        x=torch.tensor([r for e in episodes for r in e['observations']],dtype=torch.float32,device=device)
        logits=self.critic(x).reshape(-1)
        check(logits.shape==(len(x),) and bool(torch.isfinite(logits).all()),'nonfinite or invalid cost critic output')
        episode_losses=[];episode_briers=[];low_losses=[];low_briers=[];cursor=0;low_frames=0
        with torch.no_grad():probabilities=logits.sigmoid()
        for e in episodes:
            n=e['frames'];piece=logits[cursor:cursor+n];target=torch.full_like(piece,float(e['shutdown']))
            losses=F.binary_cross_entropy_with_logits(piece,target,reduction='none')
            briers=(probabilities[cursor:cursor+n]-target).square()
            episode_losses.append(losses.mean());episode_briers.append(briers.mean())
            if self.config.low_battery_columns:
                low=(x[cursor:cursor+n,list(self.config.low_battery_columns)]<=self.config.low_battery_fraction).any(-1)
                if bool(low.any()):
                    low_losses.append(losses[low].mean());low_briers.append(briers[low].mean());low_frames+=int(low.sum())
            cursor+=n
        loss=torch.stack(episode_losses).mean()
        diagnostics=dict(complete_episodes=len(episodes),shutdown_episodes=sum(e['shutdown'] for e in episodes),
            physical_observations=len(x),training_episode_mean_BCE=float(loss.detach()),
            training_episode_mean_Brier=float(torch.stack(episode_briers).mean()),
            low_battery_frames=low_frames,
            low_battery_episode_mean_BCE=float(torch.stack(low_losses).mean().detach()) if low_losses else None,
            low_battery_episode_mean_Brier=float(torch.stack(low_briers).mean()) if low_briers else None,
            episode_age_joint_steps=[self.joint_step-e['collected_joint_step'] for e in episodes],
            label_scope='actual_behavior_policy_outcomes_not_independent_current_policy_risk')
        return loss,diagnostics

    def update_critic(self,*,episode_count=None):
        check(self.episodes,'no complete TRAIN episode available for critic replay')
        count=self.config.episodes_per_update if episode_count is None else episode_count
        check(type(count) is int and count>0,'positive critic episode count required')
        ids=tuple(self.episodes);indices=self.rng.choice(len(ids),size=min(count,len(ids)),replace=False)
        selected=[self.episodes[ids[int(i)]] for i in indices]
        loss,diagnostics=self._forward(selected)
        protected={id(p):(p.detach().clone(),None if p.grad is None else p.grad.detach().clone())
                   for m in self.forbidden for p in m.parameters()}
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(tuple(self.critic.parameters()),self.config.maximum_gradient_norm)
        check(bool(torch.isfinite(norm)),'nonfinite cost critic gradient')
        self.optimizer.step();self.critic_updates+=1
        for m in self.forbidden:
            for p in m.parameters():
                old,grad=protected[id(p)]
                check(torch.equal(p.detach(),old) and ((grad is None and p.grad is None)
                    or (grad is not None and p.grad is not None and torch.equal(p.grad,grad))),
                    'critic replay changed a protected Actor/reward parameter or gradient')
        return dict(version=VERSION,critic_update=self.critic_updates,selected_episode_ids=[e['episode_id'] for e in selected],
            cost_critic_gradient_norm_before_clip=float(norm),risk_multiplier=self.risk.multiplier,
            risk_observed_complete_episode_count=len(self.seen_episodes),
            new_shutdown_observations_from_replay=0,historical_Actor_gradient_updates=0,
            **diagnostics,training_authority=False,model_acceptance_passed=False)

    def snapshot(self):
        return dict(version=VERSION,buffer_version=BUFFER_VERSION,config=asdict(self.config),
            observation_dim=self.observation_dim,joint_step=self.joint_step,critic_updates=self.critic_updates,
            episodes=deepcopy(list(self.episodes.values())),seen_episodes=deepcopy(self.seen_episodes),
            seen_batches=sorted(self.seen_batches),risk=self.risk.snapshot(),last_risk_update=deepcopy(self.last_risk_update),
            cost_critic_state={k:v.detach().cpu().clone() for k,v in self.critic.state_dict().items()},
            cost_optimizer_state=deepcopy(self.optimizer.state_dict()),
            rng_bit_generator_class=type(self.rng.bit_generator).__name__,rng_state=deepcopy(self.rng.bit_generator.state),
            Actor_or_reward_weights_included=False,training_authority=False)

    def restore(self,value):
        expected=set(self.snapshot())
        check(type(value) is dict and set(value)==expected and value['version']==VERSION
            and value['buffer_version']==BUFFER_VERSION and value['config']==asdict(self.config)
            and value['observation_dim']==self.observation_dim and value['training_authority'] is False
            and value['Actor_or_reward_weights_included'] is False,'risk numerical checkpoint contract differs')
        check(type(value['joint_step']) is int and value['joint_step']>=0
            and type(value['critic_updates']) is int and value['critic_updates']>=0
            and value['rng_bit_generator_class']==type(self.rng.bit_generator).__name__,
            'risk numerical counter or caller-owned RNG class differs')
        check(type(value['episodes']) is list,'saved complete episodes must be an ordered list')
        episodes=[validate_saved_episode(e,observation_dim=self.observation_dim,horizon=self.config.episode_horizon)
                  for e in value['episodes']]
        check(len(episodes)<=self.config.capacity_episodes and len({e['episode_id'] for e in episodes})==len(episodes)
            and type(value['seen_episodes']) is dict
            and all(type(k) is str and k and valid_sha(v) for k,v in value['seen_episodes'].items())
            and all(value['seen_episodes'].get(e['episode_id'])==e['episode_sha256']
                    and e['collected_joint_step']<=value['joint_step'] for e in episodes),
            'risk replay buffer or episode membership differs')
        check(type(value['seen_batches']) is list and len(set(value['seen_batches']))==len(value['seen_batches'])
            and all(valid_sha(x) for x in value['seen_batches'])
            and all(e['batch_sha256'] in value['seen_batches'] for e in episodes),'saved training batch identity differs')
        risk=ShutdownRiskDual.restore(value['risk'],self.config.risk)
        # Check tensor identities, shapes and numerical finiteness before load.
        # A shape-compatible NaN checkpoint must not poison the caller's model.
        state=value['cost_critic_state'];current=self.critic.state_dict()
        check(type(state) is dict and set(state)==set(current),'saved critic parameter identities differ')
        check(all(isinstance(state[k],torch.Tensor) and state[k].shape==current[k].shape
            and state[k].dtype==current[k].dtype and bool(torch.isfinite(state[k]).all())
            for k in current),'saved critic parameter shape, dtype or finiteness differs')
        optimizer=value['cost_optimizer_state'];expected_optimizer=self.optimizer.state_dict()
        check(type(optimizer) is dict and set(optimizer)=={'state','param_groups'}
            and type(optimizer['state']) is dict and type(optimizer['param_groups']) is list
            and len(optimizer['param_groups'])==len(expected_optimizer['param_groups']),
            'saved cost optimizer structure differs')
        for group,expected_group in zip(optimizer['param_groups'],expected_optimizer['param_groups']):
            check(type(group) is dict and group==expected_group,'saved optimizer config or parameter identities differ')
        parameters={identifier:parameter for group,live in zip(optimizer['param_groups'],self.optimizer.param_groups)
                    for identifier,parameter in zip(group['params'],live['params'])}
        check(set(optimizer['state'])<=set(parameters),'saved optimizer references an unowned parameter')
        for identifier,saved in optimizer['state'].items():
            check(type(saved) is dict and set(saved)=={'step','exp_avg','exp_avg_sq'},
                  'saved Adam moment fields differ')
            parameter=parameters[identifier]
            check(all(isinstance(saved[k],torch.Tensor) and saved[k].shape==parameter.shape
                and saved[k].dtype==parameter.dtype and bool(torch.isfinite(saved[k]).all())
                for k in ('exp_avg','exp_avg_sq'))
                and bool((saved['exp_avg_sq']>=0).all())
                and isinstance(saved['step'],torch.Tensor) and saved['step'].numel()==1
                and bool(torch.isfinite(saved['step']).all()) and float(saved['step'])>=0,
                'saved Adam moments or step are invalid')
        # Validate RNG state using the caller-owned generator and restore it
        # immediately. No fresh generator or global Torch RNG is constructed.
        old_rng=deepcopy(self.rng.bit_generator.state)
        try:
            self.rng.bit_generator.state=deepcopy(value['rng_state'])
        except (TypeError,ValueError,KeyError) as exc:
            raise ValueError('saved caller-owned RNG state differs') from exc
        finally:
            self.rng.bit_generator.state=old_rng
        old_critic={k:v.detach().clone() for k,v in current.items()}
        old_optimizer=deepcopy(expected_optimizer)
        try:
            self.critic.load_state_dict(state,strict=True)
            self.optimizer.load_state_dict(deepcopy(optimizer))
            self.rng.bit_generator.state=deepcopy(value['rng_state'])
        except Exception:
            self.critic.load_state_dict(old_critic,strict=True)
            self.optimizer.load_state_dict(old_optimizer)
            self.rng.bit_generator.state=old_rng
            raise
        self.episodes=OrderedDict((e['episode_id'],e) for e in episodes)
        self.seen_episodes=deepcopy(value['seen_episodes']);self.seen_batches=set(value['seen_batches'])
        self.risk=risk;self.last_risk_update=deepcopy(value['last_risk_update'])
        self.joint_step=value['joint_step'];self.critic_updates=value['critic_updates']
        return self


def make_risk_training_core(model,*,observation_dim,rng,config=RiskTrainingConfig()):
    """Explicit factory over caller-created modules; never constructs an Actor/env."""
    check(isinstance(getattr(model,'cost_critic',None),nn.Module)
        and isinstance(getattr(model,'actors',None),nn.Module)
        and isinstance(getattr(model,'reward_critic',None),nn.Module),'complete caller-owned critic/Actor modules required')
    optimizer=torch.optim.Adam(model.cost_critic.parameters(),lr=config.learning_rate,eps=config.optimizer_epsilon)
    return RiskTrainingCore(model.cost_critic,optimizer,forbidden_modules=(model.actors,model.reward_critic),
        observation_dim=observation_dim,rng=rng,config=config)
