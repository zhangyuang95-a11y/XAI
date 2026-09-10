"""Finite own-shutdown reward comparison; no runtime action intervention.

A genuine ContinuationTrainer supplies the exact Actor/Critic, both Adam states,
in-flight public-history environments, episode prefixes and all owned RNGs.
Only per-role returned training reward changes; this module grants no budget,
opens no checkpoint file and never masks/reselects a neural command.
"""
from __future__ import annotations
from collections import Counter
import copy
from pathlib import Path
import re
import numpy as np
import torch
from .warehouse_native_common import ROOT,digest,file_hash
from .warehouse_native_continuation_trainer import ContinuationTrainer, VERSION as SOURCE_VERSION, execution_sources as parent_sources
from .warehouse_native_credit_trainer import CreditRewardEnvironment,CREDIT_REWARD_VERSION
from .warehouse_native_partner_mix_trainer import _sha,_prefix
from .warehouse_native_public_feedback import VERSION as OBSERVER_VERSION
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_public_feedback_trainer import (PublicFeedbackTrainer,COUNTERS,EPISODE_FIELDS,
    _cpu,_integer,_finite,_validate_learning_state,_preserve_global_rng,_set_global_rng)
from .warehouse_native_revision_reward import REWARD_REVISION
from .warehouse_native_v2 import NativeV2Trainer
from env.warehouse_native.scenarios import reset_scenario

VERSION='warehouse-native-own-shutdown-trainer.v1'
PROTOCOL_VERSION='warehouse-native-own-shutdown-protocol.v1'
SHUTDOWN_REWARD_VERSION='warehouse-native-own-shutdown-reward.v1'
SHUTDOWN_ARMS=('beta0','beta1')
BETAS={'beta0':0.,'beta1':1.}
SHUTDOWN_SNAPSHOT_KEYS=frozenset({'own_shutdown_reward_version','own_shutdown_beta'})


def execution_sources():
    result=parent_sources();result[str(Path(__file__).relative_to(ROOT))]=file_hash(Path(__file__))
    return result


class OwnShutdownRewardEnvironment(CreditRewardEnvironment):
    def __init__(self,config=None,reward_config=None,collision_cost=.05,*,alpha=.5,beta=0.,mode='observed'):
        if type(beta) not in (int,float) or beta not in BETAS.values() or type(alpha) not in (int,float) or alpha!=.5:
            raise ValueError('Only registered own-shutdown beta and inherited alpha.5 are allowed')
        self.beta=float(beta)
        super().__init__(config,reward_config,collision_cost=collision_cost,alpha=alpha,mode=mode)

    def step(self,actions,*,decision_metadata=None):
        self._require_state()
        before={key:self.state.by_id(key).active for key in self.agent_ids}
        obs,credit,terminated,truncated,info=super().step(actions,decision_metadata=decision_metadata)
        shutdown={key:bool(before[key] and not self.state.by_id(key).active) for key in self.agent_ids}
        counts={key:0 for key in self.agent_ids}
        for event in info['events']:
            if event['event']=='shutdown':counts[event['agent_id']]+=1
        if any(counts[key]!=int(shutdown[key]) for key in self.agent_ids):
            raise RuntimeError('Actual active-to-shutdown transition differs from physical event')
        adjustments={key:-self.beta*int(shutdown[key]) for key in self.agent_ids}
        rewards={key:credit[key]+adjustments[key] for key in self.agent_ids}
        mean_adjustment=sum(adjustments.values())/len(self.agent_ids)
        original_components=copy.deepcopy(info['reward_components'])
        info['reward_components']={**original_components,'own_shutdown_cost_mean':mean_adjustment}
        info['own_shutdown_reward']={'version':SHUTDOWN_REWARD_VERSION,'beta':self.beta,
            'by_role_shutdowns':shutdown,'shutdown_count':sum(shutdown.values()),
            'adjustments':adjustments,'mean_adjustment':mean_adjustment,
            'credit_rewards':dict(credit),'rewards':dict(rewards),'original_reward_components':original_components}
        # The parent's original shared state.last_rewards and public history
        # hash stay untouched. Extra costs are private training reward/info only.
        return obs,rewards,terminated,truncated,info

    def snapshot(self):
        result=super().snapshot()
        result.update(own_shutdown_reward_version=SHUTDOWN_REWARD_VERSION,own_shutdown_beta=self.beta)
        return result

    def _commit_credit(self,candidate):
        self.state=candidate.state;self.set_rng_state(candidate.get_rng_state())
        self._episode_counter=candidate._episode_counter;self._history=copy.deepcopy(candidate._history)

    def restore(self,payload,*,require_feedback=False,require_credit=False,require_shutdown=False):
        if not isinstance(payload,dict) or type(require_shutdown) is not bool:
            raise ValueError('Invalid own-shutdown snapshot')
        keys={key for key in payload if key.startswith('own_shutdown_')}
        if keys:
            if (keys!=SHUTDOWN_SNAPSHOT_KEYS or payload.get('own_shutdown_reward_version')!=SHUTDOWN_REWARD_VERSION
                    or type(payload.get('own_shutdown_beta')) not in (int,float) or payload['own_shutdown_beta']!=self.beta):
                raise ValueError('Own-shutdown snapshot reward identity or beta differs')
        elif require_shutdown or payload.get('state',{}).get('frame')!=0 or payload.get('public_feedback_history',{}).get('valid'):
            raise ValueError('Own-shutdown marker required for confirmed or resumed state')
        candidate=CreditRewardEnvironment(self.config,copy.deepcopy(self.reward_config),self.collision_cost,alpha=self.alpha)
        candidate.restore({k:copy.deepcopy(v) for k,v in payload.items() if k not in SHUTDOWN_SNAPSHOT_KEYS},
            require_feedback=require_feedback,require_credit=require_credit)
        self._commit_credit(candidate)

    def restore_source_credit(self,payload,*,expected_snapshot_sha256):
        """Explicit first fork only; caller must first load the genuine source trainer."""
        _sha(expected_snapshot_sha256,'source snapshot')
        if not isinstance(payload,dict):raise ValueError('Source credit snapshot must be an object')
        if initialization_sha256(payload)!=expected_snapshot_sha256 or any(k.startswith('own_shutdown_') for k in payload):
            raise ValueError('Source credit snapshot is changed or already a shutdown revision')
        candidate=CreditRewardEnvironment(self.config,copy.deepcopy(self.reward_config),self.collision_cost,alpha=self.alpha)
        candidate.restore(copy.deepcopy(payload),require_feedback=True,require_credit=True)
        if candidate.done:raise ValueError('Training source must retain a live in-flight episode')
        self._commit_credit(candidate)

    def branch(self):
        result=type(self)(self.config,copy.deepcopy(self.reward_config),self.collision_cost,alpha=self.alpha,beta=self.beta)
        result.restore(self.snapshot(),require_feedback=True,require_credit=True,require_shutdown=True)
        return result


def _source(source,fixture):
    if type(source) is not ContinuationTrainer or type(fixture) is not bool or source.test_fixture is not fixture:
        raise ValueError('Only the genuine original ContinuationTrainer source is accepted')
    if source.mode!='observed' or source.branch!='own_credit' or source.alpha!=.5 or source.feedback_enabled:
        raise ValueError('Source learning condition differs')
    state=source.state_dict()
    if not fixture and (source.joint_steps!=1000000 or source.protocol['budget']['maximum_ppo_joint_steps']!=1000000
            or source.joint_steps not in source.protocol['evaluation']['checkpoints_ppo_steps']):
        raise ValueError('Production source must be the registered fixed1m endpoint')
    return state


def _protocol(source,state,*,checkpoint_sha,state_sha,ppo_cap,cycle_id,evaluation_checkpoints,fixture):
    _sha(checkpoint_sha,'source checkpoint');_sha(state_sha,'source state')
    if initialization_sha256(state)!=state_sha:raise ValueError('Source state differs from external semantic hash')
    _integer(ppo_cap,'Per-arm finite PPO cap',1)
    if ppo_cap%len(source.envs) or (not fixture and ppo_cap!=250000):raise ValueError('Own-shutdown requires fixed finite per-arm budget')
    if type(cycle_id) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',cycle_id):raise ValueError('Invalid cycle identity')
    endpoints=[50000,250000] if evaluation_checkpoints is None else copy.deepcopy(evaluation_checkpoints)
    if (type(endpoints) is not list or not endpoints or any(type(x) is not int or not 0<x<=ppo_cap or x%len(source.envs) for x in endpoints)
            or endpoints!=sorted(set(endpoints)) or endpoints[-1]!=ppo_cap or (not fixture and endpoints!=[50000,250000])):
        raise ValueError('Own-shutdown evaluation endpoints differ')
    identity={'trainer_version':SOURCE_VERSION,'branch':source.branch,'checkpoint_sha256':checkpoint_sha,
        'state_sha256':state_sha,'protocol_sha256':digest(source.protocol),'source_sha256':digest(source.sources),
        'joint_steps':state['joint_steps'],'cumulative_joint_steps':source.source_counters['joint_steps']+state['joint_steps']}
    lineage=copy.deepcopy(source.source_lineage);lineage.append(identity)
    return {'version':PROTOCOL_VERSION,'test_fixture':fixture,'cycle_id':cycle_id,'branch':source.branch,
        'shutdown_arms':list(SHUTDOWN_ARMS),'betas_by_arm':dict(BETAS),'own_shutdown_reward_version':SHUTDOWN_REWARD_VERSION,
        'public_feedback_version':OBSERVER_VERSION,'public_feedback_mode':'observed',
        'credit_reward_version':CREDIT_REWARD_VERSION,'delivery_credit_alpha':.5,
        **{k:copy.deepcopy(source.protocol[k]) for k in ('training','reward','collision_training_cost','seed')},
        'partners_by_branch':{source.branch:copy.deepcopy(source.protocol['partners_by_branch'][source.branch])},
        'source':identity,'source_lineage':lineage,'source_protocol':copy.deepcopy(source.protocol),'feedback_training_enabled':False,
        'budget':{'maximum_ppo_joint_steps_per_arm':ppo_cap,'maximum_ppo_joint_steps':ppo_cap,
            'requested_total_ppo_joint_steps':2*ppo_cap,'curriculum_generation_steps':0},
        'evaluation':{'checkpoints_ppo_steps':endpoints,'partners':['skilled','assertive','noisy'],'scenarios_per_partner':50,
            'deterministic':True,'horizon':120,'maximum_environment_steps':2*len(endpoints)*3*50*120,
            'reuse_source_zero_step_validation':True,'read_final_test':False,'training_credit_in_evaluation':False,
            'training_shutdown_in_evaluation':False},
        'continuation':{'inflight_episodes':'preserve_exactly','rng':'resume_without_reseeding',
            'learning_configuration':'own_shutdown_training_cost_only','critic_structure':'unchanged'}}


def make_protocol(source_trainer,*,source_checkpoint_sha256,source_state_sha256,cycle_id,
                  ppo_cap=250000,evaluation_checkpoints=None,test_fixture=False):
    state=_source(source_trainer,test_fixture)
    return _protocol(source_trainer,state,checkpoint_sha=source_checkpoint_sha256,state_sha=source_state_sha256,
        ppo_cap=ppo_cap,cycle_id=cycle_id,evaluation_checkpoints=evaluation_checkpoints,fixture=test_fixture)


class OwnShutdownTrainer(ContinuationTrainer):
    """Independent source copy; only own actual shutdown adds training cost."""

    def __init__(self, protocol, source_trainer, *, expected_source_state_sha256,
                 source_checkpoint_sha256, shutdown_arm, device="cpu", test_fixture=False):
        saved = _source(source_trainer, test_fixture)
        if type(shutdown_arm) is not str or shutdown_arm not in SHUTDOWN_ARMS:
            raise ValueError("Unknown own-shutdown arm")
        if torch.device(device) != source_trainer.device:
            raise ValueError("Own-shutdown requires the same actual training device")
        if not isinstance(protocol, dict): raise ValueError("A registered continuation protocol is required")
        expected = _protocol(source_trainer, saved, checkpoint_sha=source_checkpoint_sha256,
            state_sha=expected_source_state_sha256,
            ppo_cap=protocol.get("budget", {}).get("maximum_ppo_joint_steps_per_arm"),
            cycle_id=protocol.get("cycle_id"),
            evaluation_checkpoints=protocol.get("evaluation", {}).get("checkpoints_ppo_steps"), fixture=test_fixture)
        if initialization_sha256(protocol) != initialization_sha256(expected): raise ValueError("Own-shutdown must retain its exact registered single-factor contract")
        # Use the source's actual loader, never disguise the new payload as an old version.
        # Whole-object deepcopy keeps optimizer parameter links internal to the copy;
        # the genuine loader also stages brand-new parameters and Adam tensors.
        independent = copy.deepcopy(source_trainer)
        type(source_trainer).load_state_dict(independent, saved)
        self.__dict__.update(independent.__dict__)
        self.protocol, self.cfg = copy.deepcopy(protocol), copy.deepcopy(protocol["training"])
        self.shutdown_arm, self.beta = shutdown_arm, BETAS[shutdown_arm]
        migrated = []
        for snapshot in saved["envs"]:
            env = self._environment()
            env.restore_source_credit(snapshot, expected_snapshot_sha256=initialization_sha256(snapshot))
            migrated.append(env)
        self.envs = migrated
        self.sources = execution_sources()
        self.initialization_sha256, self.source_checkpoint_sha256 = expected_source_state_sha256, source_checkpoint_sha256
        self.cycle_id, self.source_lineage = protocol["cycle_id"], copy.deepcopy(protocol["source_lineage"])
        self.source_parent_counters = {k: saved[k] for k in COUNTERS}
        self.source_counters = {k: source_trainer.source_counters[k] + saved[k]
            for k in ("joint_steps", "optimizer_updates", "minibatch_updates", "actor_optimizer_steps", "critic_optimizer_steps")}
        self.source_frames = [env.state.frame for env in self.envs]
        self.source_completed_episodes_sha256 = initialization_sha256(saved["completed_episodes"])
        self.source_round_episode_prefixes_sha256 = initialization_sha256(saved["round_episode_prefixes"])
        self.round_episode_prefixes = [_prefix(env, self.episode_returns[i], self.episode_reward_components[i])
            for i, env in enumerate(self.envs)]
        self.completed_episodes = []
        for key in COUNTERS: setattr(self, key, 0)
        self.elapsed_seconds, self.feedback_enabled, self._collecting = 0., False, False

    def _environment(self):
        return OwnShutdownRewardEnvironment(reward_config=self.protocol["reward"],
            collision_cost=self.protocol["collision_training_cost"], alpha=self.alpha, beta=self.beta)

    def collect(self, time_steps):
        _integer(time_steps, "Rollout length", 1)
        if self.joint_steps + time_steps * len(self.envs) > self.protocol["budget"]["maximum_ppo_joint_steps"]:
            raise ValueError("Collect exceeds this finite continuation cycle")
        self._collecting = True
        try: batch = PublicFeedbackTrainer.collect(self, time_steps)
        finally: self._collecting = False
        for record in batch["transition_records"]:
            record.update(version=VERSION, cycle_id=self.cycle_id, branch=self.branch,
                shutdown_arm=self.shutdown_arm, own_shutdown_beta=self.beta, own_shutdown_reward_version=SHUTDOWN_REWARD_VERSION,
                delivery_credit_alpha=self.alpha, source_cumulative_joint_steps=self.source_counters["joint_steps"],
                cumulative_joint_step=self.source_counters["joint_steps"] + record["joint_step"])
        return batch

    def _bindings(self):
        return {"version": VERSION, "cycle_id": self.cycle_id, "branch": self.branch,
            "shutdown_arm": self.shutdown_arm, "own_shutdown_beta": self.beta, "own_shutdown_reward_version": SHUTDOWN_REWARD_VERSION,
            "credit_reward_version": CREDIT_REWARD_VERSION, "delivery_credit_alpha": self.alpha,
            "public_feedback_version": OBSERVER_VERSION, "public_feedback_mode": "observed",
            "training_device": self.device.type, "protocol": copy.deepcopy(self.protocol),
            "protocol_sha256": digest(self.protocol), "scenario_manifest_sha256": digest(self.scenarios),
            "sources": copy.deepcopy(self.sources), "source_sha256": digest(self.sources),
            "initialization_sha256": self.initialization_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_lineage": copy.deepcopy(self.source_lineage), "source_counters": copy.deepcopy(self.source_counters),
            "source_parent_counters": copy.deepcopy(self.source_parent_counters),
            "source_completed_episodes_sha256": self.source_completed_episodes_sha256,
            "source_round_episode_prefixes_sha256": self.source_round_episode_prefixes_sha256,
            "source_frames": self.source_frames.copy(), "obs_dim": 197, "state_dim": 354,
            "feature_names": list(self.envs[0].feature_names), "reward_revision": REWARD_REVISION,
            "test_fixture": self.test_fixture, "curriculum_generation_steps": 0, "feedback_enabled": False}

    def state_dict(self):
        if execution_sources() != self.sources: raise ValueError("Continuation execution sources changed")
        result = {**self._bindings(), "model": _cpu(self.model.state_dict()), "optimizers": _cpu(self.optimizers.state_dict()),
            "rng": copy.deepcopy(self.rng.bit_generator.state), **_cpu(self._owned_rng_state),
            "envs": [env.snapshot() for env in self.envs], "completed_episodes": _cpu(self.completed_episodes),
            "elapsed_seconds": self.elapsed_seconds, "round_episode_prefixes": copy.deepcopy(self.round_episode_prefixes)}
        for key in (*COUNTERS, *EPISODE_FIELDS): result[key] = _cpu(getattr(self, key))
        result["episode_reward_components"] = [dict(x) for x in self.episode_reward_components]
        return result

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or execution_sources() != self.sources:
            raise ValueError("Invalid checkpoint or changed continuation sources")
        if any(initialization_sha256(payload.get(k)) != initialization_sha256(v) for k, v in self._bindings().items()):
            raise ValueError("Continuation identity/device/source/learning contract differs")
        p, count = copy.deepcopy(payload), len(self.envs)
        for key in COUNTERS: _integer(p.get(key), key)
        if (p["joint_steps"] % count or p["joint_steps"] > self.protocol["budget"]["maximum_ppo_joint_steps"]
                or p["optimizer_updates"] > p["minibatch_updates"] or p["actor_optimizer_steps"] > p["minibatch_updates"]
                or p["critic_optimizer_steps"] != p["minibatch_updates"] or p["last_evaluated_joint_steps"] > p["joint_steps"]):
            raise ValueError("Invalid continuation counters")
        _finite(p.get("elapsed_seconds"), "elapsed time")
        if p["elapsed_seconds"] < 0: raise ValueError("Negative elapsed time")
        _validate_learning_state(p.get("model"), p.get("optimizers"), self.cfg,
            {role: self.source_counters[role + "_optimizer_steps"] + p[role + "_optimizer_steps"] for role in ("actor", "critic")})
        for key in (*EPISODE_FIELDS, "envs", "round_episode_prefixes"):
            if not isinstance(p.get(key), (list, np.ndarray)) or len(p[key]) != count:
                raise ValueError("Invalid episode vector: " + key)
        returns = np.asarray(p["episode_returns"])
        if returns.shape != (count,) or returns.dtype != np.float64 or not np.isfinite(returns).all():
            raise ValueError("Invalid episode returns")
        envs = []
        for i, saved in enumerate(p["envs"]):
            kind, role = p["partner_kinds"][i], p["program_roles"][i]
            if (kind not in self.protocol["partners_by_branch"][self.branch] or type(role) is not int
                    or role not in ((-1,) if kind == "selfplay" else (0, 1)) or p["scenario_ids"][i] not in self._train_ids):
                raise ValueError("Invalid episode partner/scenario")
            entry = next(x for x in self.scenarios["splits"]["train"] if x["id"] == p["scenario_ids"][i])
            baseline = self._environment(); reset_scenario(baseline, entry)
            expected = {"source": "original_train", "curriculum_id": None, "category": None, "start_frame": 0,
                "prefix": {"team_deliveries": baseline.state.total_deliveries,
                    "individual_deliveries": [a.deliveries_completed for a in baseline.state.agents],
                    "shutdowns": baseline.state.shutdown_count, "collisions": baseline.state.robot_collision_events}}
            if p["episode_context"][i] != expected: raise ValueError("Original episode context differs")
            env = self._environment(); env.restore(saved, require_feedback=True, require_credit=True, require_shutdown=True)
            if (env.done or env.state.frame > self.source_frames[i] + p["joint_steps"] // count
                    or (env.state.frame and not env.public_history()["valid"])):
                raise ValueError("Invalid confirmed in-flight state")
            components = p["episode_reward_components"][i]
            if not isinstance(components, dict): raise ValueError("Invalid reward accounting")
            for key, value in components.items():
                if type(key) is not str: raise ValueError("Invalid component name")
                _finite(value, key)
            prefix = p["round_episode_prefixes"][i]
            if not isinstance(prefix, dict) or set(prefix) != set(_prefix(env, 0., {})):
                raise ValueError("Invalid inherited episode prefix")
            for key in ("frame", "team_deliveries", "shutdowns", "collisions"): _integer(prefix[key], key)
            if prefix["frame"] > env.state.frame or not isinstance(prefix["individual_deliveries"], list) or len(prefix["individual_deliveries"]) != 2:
                raise ValueError("Invalid prefix frame/deliveries")
            for value in prefix["individual_deliveries"]: _integer(value, "prefix delivery")
            _finite(prefix["return"], "prefix return")
            if not isinstance(prefix["reward_components"], dict): raise ValueError("Invalid prefix rewards")
            for value in prefix["reward_components"].values(): _finite(value, "prefix reward")
            envs.append(env)
        if not isinstance(p.get("completed_episodes"), list): raise ValueError("Invalid episode journal")
        initialization_sha256(p["completed_episodes"])
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(p["rng"])
        owned = {key: p[key] for key in ("python_rng", "numpy_rng", "torch_rng")}
        if self.device.type == "mps": owned["mps_rng"] = p["mps_rng"]
        elif "mps_rng" in p: raise ValueError("CPU cannot restore MPS RNG")
        for key in ("torch_rng", *(("mps_rng",) if self.device.type == "mps" else ())):
            value = owned[key]
            if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.device.type != "cpu" or value.ndim != 1:
                raise ValueError("Invalid RNG tensor")
        with _preserve_global_rng(self.device):
            _set_global_rng(owned, self.device)
            model, optimizers = self._new_learning_state(p["model"], p["optimizers"])
        self.model, self.optimizers, self.envs, self.rng = model, optimizers, envs, rng
        self._owned_rng_state = _cpu(owned)
        for key in (*COUNTERS, *EPISODE_FIELDS, "elapsed_seconds", "completed_episodes", "round_episode_prefixes"):
            setattr(self, key, copy.deepcopy(p[key]))
        self.episode_returns = returns.copy()
        self.episode_reward_components = [Counter(x) for x in p["episode_reward_components"]]

    def save(self, path):
        """Exclusive state-only checkpoint; orchestration owns its durable envelope."""
        payload = self.state_dict()
        with Path(path).open("xb") as stream: torch.save(payload, stream)

    def export(self, path):
        if execution_sources() != self.sources: raise ValueError("Continuation execution sources changed")
        metadata = self._bindings()
        for key in ("version", "sources", "protocol", "source_frames", "source_completed_episodes_sha256"):
            metadata.pop(key)
        metadata.update(experiment_version=VERSION, joint_steps=self.joint_steps,
            optimizer_updates=self.optimizer_updates, minibatch_updates=self.minibatch_updates,
            initialization="exact_genuine_source_with_inflight_episodes", candidate=True)
        return self.model.export_npz(path, metadata)
