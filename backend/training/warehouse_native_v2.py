"""Prepared v2 foundation, independent from the completed original run.

The CLI refuses all sampling until an explicit additional-budget flag and cap are
provided. Merely preparing this implementation does not authorize that flag.
Curriculum generation and PPO reserve from the same crash-safe budget. Programs
only produce legal training snapshots or act as ordinary partners; their action
labels, paths and target choices are never Actor targets or observation features.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import math
from pathlib import Path
import random
import signal
import shutil
import time

import numpy as np
import torch

from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.partners import partner_action, partner_actions
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor
from env.warehouse_native.scenarios import generate_manifest, reset_scenario, scenario_fingerprint
from .warehouse_native import NativeTrainer, gae, ppo_actor_objective, atomic_torch_save, acknowledge_update
from .warehouse_native_common import ROOT, atomic_json, canonical, digest, file_hash, source_hashes
from .warehouse_native_evaluation import evaluate, capability

VERSION="warehouse-native-foundation.v2"
REWARD_VERSION="warehouse-native-score-pbrs.v2"
CURRICULUM_VERSION="warehouse-native-legal-curriculum.v2"
PROTOCOL_PATH=Path(__file__).with_name("warehouse_native_v2_protocol.json")


def load_v2_protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def v2_source_hashes():
    result=source_hashes()
    result[str(Path(__file__).relative_to(ROOT))]=file_hash(__file__)
    result[str(PROTOCOL_PATH.relative_to(ROOT))]=file_hash(PROTOCOL_PATH)
    return result


class V2RewardEnvironment(NativeWarehouseEnv):
    """Identical public physics/score, with an explicitly different RL reward."""
    def __init__(self,config=None,reward_config=None):
        super().__init__(config)
        self.reward_config=copy.deepcopy(reward_config or load_v2_protocol()["reward"])
        if self.reward_config["version"]!=REWARD_VERSION:
            raise ValueError("Unknown v2 training reward")

    def step(self,actions,*,decision_metadata=None):
        before_score=self.native_score
        obs,_,terminated,truncated,info=super().step(actions,decision_metadata=decision_metadata)
        old_components=info["reward_components"]
        cfg=self.reward_config
        components={"native_score_increment":cfg["native_score_scale"]*(self.native_score-before_score),
            "static_wall_commands":-cfg["static_wall_command_cost"]*len(info["invalid_moves"]),
            "potential":cfg["potential_scale"]*(cfg["gamma"]*info["potential_after"]-info["potential_before"])}
        reward=float(sum(components.values()))
        rewards={key:reward for key in self.agent_ids}
        self.state.last_rewards=dict(rewards)
        info.update(reward_version=REWARD_VERSION,reward_components=components,
            original_foundation_reward_components=old_components,participant_score_unchanged=True)
        return obs,rewards,terminated,truncated,info

    def snapshot(self):
        payload=super().snapshot()
        payload["training_reward_version"]=REWARD_VERSION
        payload["training_reward_config"]=copy.deepcopy(self.reward_config)
        return payload

    def restore(self,payload):
        # Original physics snapshots are valid scenario/curriculum starts.
        if "training_reward_version" in payload and (payload["training_reward_version"]!=REWARD_VERSION or payload.get("training_reward_config")!=self.reward_config):
            raise ValueError("V2 reward snapshot mismatch")
        super().restore(payload)

    def branch(self):
        result=type(self)(self.config,self.reward_config)
        result.restore(self.snapshot())
        return result


class IndependentOptimizers:
    """Disjoint clipping and Adam states; the other loss cannot shrink a group."""
    def __init__(self,model,cfg):
        self.model,self.cfg=model,cfg
        self.actor_parameters=tuple(model.actor.parameters())
        self.critic_parameters=tuple(model.critic.parameters())
        if set(map(id,self.actor_parameters))&set(map(id,self.critic_parameters)):
            raise ValueError("Actor and Critic optimizers must have disjoint parameters")
        self.actor=torch.optim.Adam(self.actor_parameters,lr=cfg["actor_learning_rate"],eps=1e-5)
        self.critic=torch.optim.Adam(self.critic_parameters,lr=cfg["critic_learning_rate"],eps=1e-5)

    @staticmethod
    def _norm(values):
        return math.sqrt(sum(float(value.detach().square().sum()) for value in values if value is not None))

    def step(self,actor_loss,critic_loss,*,train_actor=True):
        if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite v2 loss")
        self.actor.zero_grad(set_to_none=True);self.critic.zero_grad(set_to_none=True)
        if train_actor:actor_loss.backward()
        critic_loss.backward()
        actor_norm=float(torch.nn.utils.clip_grad_norm_(self.actor_parameters,self.cfg["actor_gradient_norm"],error_if_nonfinite=True))
        critic_norm=float(torch.nn.utils.clip_grad_norm_(self.critic_parameters,self.cfg["critic_gradient_norm"],error_if_nonfinite=True))
        actor_after=self._norm(p.grad for p in self.actor_parameters)
        critic_after=self._norm(p.grad for p in self.critic_parameters)
        actor_before=[p.detach().clone() for p in self.actor_parameters]
        critic_before=[p.detach().clone() for p in self.critic_parameters]
        # Skip rather than apply old Adam momentum on program-only minibatches.
        if train_actor:self.actor.step()
        self.critic.step()
        return {"actor_gradient_preclip":actor_norm,"critic_gradient_preclip":critic_norm,
            "actor_gradient_postclip":actor_after,"critic_gradient_postclip":critic_after,
            "actor_clip_scale":min(1.,self.cfg["actor_gradient_norm"]/max(actor_norm,1e-20)),
            "critic_clip_scale":min(1.,self.cfg["critic_gradient_norm"]/max(critic_norm,1e-20)),
            "actor_parameter_update_norm":self._norm(p.detach()-before for p,before in zip(self.actor_parameters,actor_before)),
            "critic_parameter_update_norm":self._norm(p.detach()-before for p,before in zip(self.critic_parameters,critic_before)),
            "actor_optimizer_applied":int(train_actor)}

    def state_dict(self):
        return {"actor":self.actor.state_dict(),"critic":self.critic.state_dict()}

    def load_state_dict(self,payload):
        self.actor.load_state_dict(payload["actor"]);self.critic.load_state_dict(payload["critic"])


class TrainingBudget:
    """Durable reservation upper bound; a crashed reservation remains spent."""
    def __init__(self,path,cap):
        self.path=Path(path);self.cap=int(cap)
        if not 0<self.cap<=500000:raise ValueError("Additional budget must be in 1..500000")
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._lock():
            if not self.path.exists():
                atomic_json(self.path,{"version":VERSION,"cap":self.cap,"reserved":{"curriculum":0,"ppo":0}})
            self.read()

    def _lock(self):
        class Lock:
            def __init__(self,path):self.path=path
            def __enter__(self):
                self.stream=self.path.open("a");fcntl.flock(self.stream,fcntl.LOCK_EX);return self
            def __exit__(self,*args):self.stream.close()
        return Lock(self.path.with_suffix(".lock"))

    def read(self):
        value=json.loads(self.path.read_text())
        if value["version"]!=VERSION or value["cap"]!=self.cap or set(value["reserved"])!={"curriculum","ppo"}:
            raise ValueError("V2 budget version/cap mismatch")
        if any(type(n) is not int or n<0 for n in value["reserved"].values()) or sum(value["reserved"].values())>self.cap:
            raise ValueError("Invalid v2 budget ledger")
        return value

    @property
    def remaining(self):return self.cap-sum(self.read()["reserved"].values())

    def reserve(self,kind,amount):
        if kind not in ("curriculum","ppo") or type(amount) is not int or amount<0:
            raise ValueError("Invalid sampling reservation")
        with self._lock():
            value=self.read()
            if sum(value["reserved"].values())+amount>self.cap:raise ValueError("V2 training budget exceeded")
            value["reserved"][kind]+=amount
            atomic_json(self.path,value)
            return value


def curriculum_categories(env):
    """Only public event/geometry facts; no reference goals, actions or labels."""
    result=[];state=env.state
    distance=lambda a,b:shortest_path_distance(a,b,env.config.map_layout_id)
    if any(a.carrying_task_id is None and any(t.status=="available" and 0<distance(a.position,t.pickup_position)<=2 for t in state.tasks) for a in state.agents):
        result.append("near_pickup")
    if any(a.carrying_task_id for a in state.agents):result.append("carrying")
    if any(a.battery<=60 and distance(a.position,env.layout.charger_position)<=4 for a in state.agents):result.append("low_battery_near_charger")
    if distance(state.agents[0].position,state.agents[1].position)<=3 and any(sum(env.layout.is_passable((a.position[0]+d[0],a.position[1]+d[1])) for d in MOVE_DELTAS.values())<=2 for a in state.agents):
        result.append("narrow_encounter")
    return result


class CurriculumBuilder:
    """Resumable real transitions from train starts; never handwritten states."""
    def __init__(self,protocol,scenarios):
        self.protocol,self.scenarios=protocol,scenarios
        self.cfg=protocol["curriculum"]
        if self.cfg["source_split"]!="train" or self.cfg["teacher_action_labels"] or self.cfg["manual_state_edits"]:
            raise ValueError("Curriculum must use real train-only states without labels")
        self.rng=np.random.default_rng(np.random.SeedSequence([protocol["seed"],301]))
        self.env=NativeWarehouseEnv();self.current_entry=None
        self.steps=0;self.entries=[];self.counts=Counter();self.seen=set()
        # Only read held-out initial fingerprints, never run their trajectories.
        self.heldout={e["fingerprint"] for split,entries in scenarios["splits"].items() if split!="train" for e in entries}
        self.heldout_exclusions=0

    def advance(self,steps):
        if steps<0 or self.steps+steps>self.cfg["maximum_generation_steps"]:
            raise ValueError("Curriculum generation budget exceeded")
        for _ in range(steps):
            if self.current_entry is None or self.env.done:
                choices=self.scenarios["splits"]["train"]
                self.current_entry=choices[int(self.rng.integers(len(choices)))]
                reset_scenario(self.env,self.current_entry)
            self.env.step(partner_actions(self.env,self.cfg["generator_partner"],self.rng))
            self.steps+=1
            if self.env.done:continue
            if scenario_fingerprint(self.env) in self.heldout:
                self.heldout_exclusions+=1
                continue
            categories=[c for c in curriculum_categories(self.env) if self.counts[c]<self.cfg["entries_per_category"]]
            fingerprint=self.env.fingerprint()
            if not categories or fingerprint in self.seen:continue
            category=min(categories,key=lambda c:(self.counts[c],self.cfg["categories"].index(c)))
            self.entries.append({"id":f"curriculum_{len(self.entries):05d}","source_id":self.current_entry["id"],"source_initial_fingerprint":self.current_entry["fingerprint"],"source_split":"train","category":category,"frame":self.env.state.frame,"physical_fingerprint":fingerprint,"initial_content_fingerprint":scenario_fingerprint(self.env),"snapshot":self.env.snapshot()})
            self.seen.add(fingerprint);self.counts[category]+=1

    def bank(self):
        return {"version":CURRICULUM_VERSION,"scenario_manifest_sha256":digest(self.scenarios),"generation_steps":self.steps,"generator_partner":self.cfg["generator_partner"],"generator_source_sha256":file_hash(ROOT/"env/warehouse_native/partners.py"),"counts":dict(self.counts),"entries":copy.deepcopy(self.entries),"teacher_action_labels":False,"manual_state_edits":False,"source_split":"train","heldout_initial_exclusions":self.heldout_exclusions,"generalization_scope":"initial_state_split_not_full_trajectory_isolation"}

    def state_dict(self):
        return {"version":CURRICULUM_VERSION,"protocol_sha256":digest(self.protocol),"scenario_manifest_sha256":digest(self.scenarios),"sources":v2_source_hashes(),"steps":self.steps,"entries":copy.deepcopy(self.entries),"counts":dict(self.counts),"heldout_exclusions":self.heldout_exclusions,"rng":copy.deepcopy(self.rng.bit_generator.state),"current_source_id":self.current_entry["id"] if self.current_entry else None,"environment":self.env.snapshot() if self.current_entry else None}

    def load_state_dict(self,payload):
        if payload["version"]!=CURRICULUM_VERSION or payload["protocol_sha256"]!=digest(self.protocol) or payload["scenario_manifest_sha256"]!=digest(self.scenarios) or payload["sources"]!=v2_source_hashes():
            raise ValueError("Curriculum generation version/provenance mismatch")
        self.steps=int(payload["steps"]);self.entries=copy.deepcopy(payload["entries"]);self.counts=Counter(payload["counts"])
        self.heldout_exclusions=int(payload["heldout_exclusions"])
        self.seen={entry["physical_fingerprint"] for entry in self.entries}
        self.rng.bit_generator.state=copy.deepcopy(payload["rng"])
        source_id=payload["current_source_id"]
        self.current_entry=next((e for e in self.scenarios["splits"]["train"] if e["id"]==source_id),None)
        if source_id is not None and self.current_entry is None:raise ValueError("Curriculum source is not in train")
        if self.current_entry:self.env.restore(payload["environment"])
        validate_curriculum_bank(self.bank(),self.scenarios)


def validate_curriculum_bank(bank,scenarios,*,replay=False):
    if bank["version"]!=CURRICULUM_VERSION or bank["scenario_manifest_sha256"]!=digest(scenarios) or bank["source_split"]!="train" or bank["teacher_action_labels"] or bank["manual_state_edits"]:
        raise ValueError("Invalid curriculum bank provenance")
    if bank["generator_source_sha256"]!=file_hash(ROOT/"env/warehouse_native/partners.py"):
        raise ValueError("Curriculum generator changed")
    if bank["generator_partner"]!="skilled" or type(bank["generation_steps"]) is not int or not 0<=bank["generation_steps"]<=10000:
        raise ValueError("Invalid curriculum generation accounting")
    if bank["counts"]!=dict(Counter(e["category"] for e in bank["entries"])) or len(bank["entries"])>bank["generation_steps"]:
        raise ValueError("Invalid curriculum category accounting")
    sources={entry["id"]:entry for entry in scenarios["splits"]["train"]}
    heldout={e["fingerprint"] for split,entries in scenarios["splits"].items() if split!="train" for e in entries}
    ids=set();fingerprints=set()
    for entry in bank["entries"]:
        if set(entry)!={"id","source_id","source_initial_fingerprint","source_split","category","frame","physical_fingerprint","initial_content_fingerprint","snapshot"}:
            raise ValueError("Unexpected curriculum field, including possible teacher labels")
        source=sources.get(entry["source_id"])
        if source is None or source["fingerprint"]!=entry["source_initial_fingerprint"] or entry["source_split"]!="train":
            raise ValueError("Curriculum entry crosses scenario split")
        if entry["id"] in ids or entry["physical_fingerprint"] in fingerprints:
            raise ValueError("Duplicate curriculum entry")
        ids.add(entry["id"]);fingerprints.add(entry["physical_fingerprint"])
        env=NativeWarehouseEnv();env.restore(entry["snapshot"])
        initial_content=scenario_fingerprint(env)
        if initial_content!=entry["initial_content_fingerprint"] or initial_content in heldout:
            raise ValueError("Curriculum state overlaps held-out initial content")
        if env.done or env.state.frame!=entry["frame"] or env.fingerprint()!=entry["physical_fingerprint"] or entry["category"] not in curriculum_categories(env):
            raise ValueError("Invalid curriculum physical state/category")
        if replay:
            witness=NativeWarehouseEnv();reset_scenario(witness,source)
            for _ in range(entry["frame"]):witness.step(partner_actions(witness,bank["generator_partner"]))
            if digest(witness.snapshot())!=digest(entry["snapshot"]):raise ValueError("Curriculum snapshot has no matching real-transition witness")


def curriculum_probability(ppo_steps,cfg):
    if ppo_steps>=cfg["zero_at_ppo_steps"]:return 0.
    if ppo_steps<=cfg["constant_until_ppo_steps"]:return cfg["initial_probability"]
    return cfg["initial_probability"]*(cfg["zero_at_ppo_steps"]-ppo_steps)/(cfg["zero_at_ppo_steps"]-cfg["constant_until_ppo_steps"])


class PublicActionAudit:
    """Read submitted NN commands and public neighbors; return actions unchanged."""
    def __init__(self,actor,feature_names):
        self.actor=actor;self.names={name:i for i,name in enumerate(feature_names)}
        self.counts=Counter()

    def act(self,observations,deterministic=True):
        actions,probabilities=self.actor.act(observations,deterministic=deterministic)
        # Standard frozen evaluator controls robot_2 with the neural candidate.
        action=actions["robot_2"];row=observations["robot_2"]
        if row[self.names["self.active"]]>.5:
            self.counts["active_neural_commands"]+=1
            self.counts["static_wall_commands"]+=int(action in MOVE_DELTAS and row[self.names[f"self.neighbor.{action}.passable"]]<.5)
        return actions,probabilities


def evaluate_v2(actor,scenarios,feature_names):
    """Same frozen selection episodes plus actual static-wall command counts."""
    summaries={};rows=[]
    for partner in ("skilled","assertive","noisy"):
        audited=PublicActionAudit(actor,feature_names)
        result=evaluate(audited,scenarios,partners=(partner,))
        summary=result["summary"][partner]
        counts=audited.counts
        summary.update(active_neural_commands=counts["active_neural_commands"],neural_static_wall_commands=counts["static_wall_commands"],neural_static_wall_rate=counts["static_wall_commands"]/max(1,counts["active_neural_commands"]))
        summaries[partner]=summary;rows.extend(result["rows"])
    return {"summary":summaries,"rows":rows,"deterministic_actor":True,"metric":"mean_team_deliveries","action_audit":"unchanged_actor_outputs_and_public_neighbor_passability"}


def diagnostic_review(report,joint_steps,protocol):
    cfg=protocol["diagnostic_checkpoints"];skilled=report["summary"]["skilled"]
    reasons=[]
    if skilled["mean_ai_deliveries"]<cfg["mean_ai_delivery_warning"]:reasons.append("low_neural_delivery_contribution")
    if skilled["neural_static_wall_rate"]>=cfg["static_wall_rate_warning"]:reasons.append("high_static_wall_command_rate")
    return {"joint_steps":joint_steps,"source":"fixed_validation_skilled_episodes","reasons":reasons,"pause_required":joint_steps>=cfg["stop_for_review_ppo_steps"] and bool(reasons),"candidate_gate_unchanged":True,"not_a_checkpoint_selection_metric":True}


def checkpoint_selection_due(ppo_steps,budget_remaining,environments,interval):
    """A temporary throughput probe cannot add a new selection checkpoint."""
    return ppo_steps>0 and (ppo_steps%interval==0 or budget_remaining<environments)


class NativeV2Trainer(NativeTrainer):
    """Reuse only v1's plain batched observations/forward; own losses and state."""
    def __init__(self,protocol,scenarios,bank,*,device="cpu"):
        self.protocol,self.scenarios,self.bank=copy.deepcopy(protocol),copy.deepcopy(scenarios),copy.deepcopy(bank)
        if self.protocol["version"]!=VERSION:raise ValueError("V2 protocol required")
        validate_curriculum_bank(bank,scenarios)
        self.cfg=self.protocol["training"];self.device=torch.device(device);self.seed=self.protocol["seed"]
        random.seed(self.seed);np.random.seed(self.seed);torch.manual_seed(self.seed)
        self.rng=np.random.default_rng(np.random.SeedSequence([self.seed,401]))
        self.curriculum_rng=np.random.default_rng(np.random.SeedSequence([self.seed,402]))
        self.envs=[V2RewardEnvironment(reward_config=protocol["reward"]) for _ in range(self.cfg["environments"])]
        count=len(self.envs)
        self.partner_kinds=[""]*count;self.program_roles=[-1]*count;self.scenario_ids=[""]*count
        self.episode_context=[None]*count;self.episode_returns=np.zeros(count,dtype=np.float64)
        self.episode_reward_components=[Counter() for _ in range(count)]
        self.completed_episodes=[];self.episode_count=0;self.joint_steps=0
        self.optimizer_updates=0;self.minibatch_updates=0;self.best=None;self.elapsed_seconds=0.
        self.feedback_enabled=False
        for i in range(count):self.reset_one(i)
        self.model=NativeActorCritic(self.envs[0].observation_size,len(self.envs[0].global_state())).to(self.device)
        self.optimizers=IndependentOptimizers(self.model,self.cfg)

    def reset_one(self,i):
        probability=curriculum_probability(self.joint_steps,self.protocol["curriculum"])
        if self.bank["entries"] and self.curriculum_rng.random()<probability:
            entry=self.bank["entries"][int(self.curriculum_rng.integers(len(self.bank["entries"])))]
            self.envs[i].restore(entry["snapshot"])
            self.scenario_ids[i]=entry["source_id"]
            self.episode_context[i]={"start_kind":"legal_curriculum","curriculum_id":entry["id"],"category":entry["category"],"start_frame":entry["frame"]}
        else:
            entries=self.scenarios["splits"]["train"]
            entry=entries[int(self.rng.integers(len(entries)))]
            reset_scenario(self.envs[i],entry)
            self.scenario_ids[i]=entry["id"]
            self.episode_context[i]={"start_kind":"original_train","curriculum_id":None,"category":None,"start_frame":0}
        mix=self.protocol["partners"]
        self.partner_kinds[i]=str(self.rng.choice(list(mix),p=list(mix.values())))
        self.program_roles[i]=-1 if self.partner_kinds[i]=="selfplay" else int(self.rng.integers(2))
        self.episode_returns[i]=0.;self.episode_reward_components[i]=Counter();self.episode_count+=1
        state=self.envs[i].state
        self.episode_context[i]["prefix"]={"team_deliveries":state.total_deliveries,"individual_deliveries":[a.deliveries_completed for a in state.agents],"shutdowns":state.shutdown_count,"collisions":state.robot_collision_events}

    def collect(self,time_steps):
        keys=("observations","states","actions","old_log_probs","values","rewards","dones","trainable")
        rows={key:[] for key in keys};audit={"neural_submitted":0,"program_submitted":0,"neural_overrides":0}
        for _ in range(time_steps):
            obs,states=self.arrays()
            programs=[None if role<0 else partner_action(env,f"robot_{role+1}",kind,self.rng) for env,kind,role in zip(self.envs,self.partner_kinds,self.program_roles)]
            probabilities,values=self.infer(obs,states)
            actions=(self.rng.random(probabilities.shape[:2])[...,None]>probabilities.cumsum(-1)).sum(-1).clip(0,4)
            old=np.log(np.take_along_axis(probabilities,actions[...,None],-1)[...,0].clip(1e-12))
            trainable=np.ones(actions.shape,dtype=np.float32);rewards=np.zeros(actions.shape,dtype=np.float32);dones=np.zeros(len(self.envs),dtype=np.float32)
            for i,(env,role) in enumerate(zip(self.envs,self.program_roles)):
                submitted={f"robot_{j+1}":ACTIONS[int(actions[i,j])] for j in range(2)}
                for j,a in enumerate(env.state.agents):trainable[i,j]=float(a.active)
                if role>=0:
                    trainable[i,role]=0;submitted[f"robot_{role+1}"]=programs[i];audit["program_submitted"]+=1
                for j in range(2):
                    if trainable[i,j]:
                        assert submitted[f"robot_{j+1}"]==ACTIONS[int(actions[i,j])]
                        audit["neural_submitted"]+=1
                _,reward,terminated,truncated,info=env.step(submitted)
                rewards[i]=[reward[key] for key in env.agent_ids]
                self.episode_returns[i]+=float(rewards[i].mean());self.episode_reward_components[i].update(info["reward_components"]);self.joint_steps+=1
                if terminated or truncated:
                    dones[i]=1
                    prefix=self.episode_context[i]["prefix"]
                    self.completed_episodes.append({"joint_step":self.joint_steps,"scenario_id":self.scenario_ids[i],"partner":self.partner_kinds[i],"program_role":role,"return":self.episode_returns[i],"reward_components":dict(self.episode_reward_components[i]),"team_deliveries":env.state.total_deliveries-prefix["team_deliveries"],"individual_deliveries":[a.deliveries_completed-prefix["individual_deliveries"][j] for j,a in enumerate(env.state.agents)],"shutdowns":env.state.shutdown_count-prefix["shutdowns"],"collisions":env.state.robot_collision_events-prefix["collisions"],"length":env.state.frame-self.episode_context[i]["start_frame"],"terminal_frame":env.state.frame,"curriculum_context":copy.deepcopy(self.episode_context[i]),"metric_scope":"post_reset_learner_segment_only"})
                    self.reset_one(i)
            for key,value in zip(keys,(obs,states,actions,old,values,rewards,dones,trainable)):rows[key].append(value)
        rows={key:np.asarray(value) for key,value in rows.items()}
        obs,state=self.arrays();_,bootstrap=self.infer(obs,state)
        rows["advantages"],rows["returns"]=gae(rows["rewards"],rows["values"],rows["dones"],bootstrap,self.cfg["gamma"],self.cfg["gae_lambda"])
        rows["audit"]=audit
        return rows

    def update(self,batch):
        obs=batch["observations"].reshape(-1,self.model.obs_dim)
        states=np.repeat(batch["states"][:,:,None,:],2,axis=2).reshape(-1,self.model.state_dim)
        roles=np.tile([0,1],len(obs)//2);trainable=batch["trainable"].reshape(-1)
        advantages=batch["advantages"].reshape(-1).copy();selected=trainable>0
        if selected.any():advantages[selected]=(advantages[selected]-advantages[selected].mean())/(advantages[selected].std()+1e-8)
        data={"obs":obs,"states":states,"roles":roles,"trainable":trainable,"actions":batch["actions"].reshape(-1),"old":batch["old_log_probs"].reshape(-1),"advantages":advantages,"returns":batch["returns"].reshape(-1)}
        data={key:torch.as_tensor(value,device=self.device) for key,value in data.items()};metrics=[]
        for _ in range(self.cfg["epochs"]):
            permutation=self.rng.permutation(len(obs))
            for start in range(0,len(obs),self.cfg["minibatch"]):
                index=torch.as_tensor(permutation[start:start+self.cfg["minibatch"]],device=self.device)
                logits=self.model.actor_logits(data["obs"][index])
                actor_loss,part=ppo_actor_objective(logits,data["actions"][index],data["old"][index],data["advantages"][index],data["trainable"][index],clip=self.cfg["clip"],entropy=self.cfg["entropy"])
                critic_loss=.5*(self.model.values(data["states"][index],data["roles"][index])-data["returns"][index]).square().mean()
                result=self.optimizers.step(actor_loss,critic_loss,train_actor=bool(data["trainable"][index].bool().any()))
                metrics.append({**part,**result,"critic_loss":float(critic_loss.detach())});self.minibatch_updates+=1
        self.optimizer_updates+=1
        return {key:float(np.mean([value.get(key,0.) for value in metrics])) for key in set().union(*(value.keys() for value in metrics))}

    def state_dict(self):
        result={"version":VERSION,"model":self.model.state_dict(),"optimizers":self.optimizers.state_dict(),"obs_dim":self.model.obs_dim,"state_dim":self.model.state_dim,"protocol":self.protocol,"scenario_manifest_hash":digest(self.scenarios),"curriculum_bank_hash":digest(self.bank),"curriculum_generation_steps":self.bank["generation_steps"],"sources":v2_source_hashes(),"rng":copy.deepcopy(self.rng.bit_generator.state),"curriculum_rng":copy.deepcopy(self.curriculum_rng.bit_generator.state),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"envs":[env.snapshot() for env in self.envs],"episode_reward_components":[dict(value) for value in self.episode_reward_components]}
        for key in ("joint_steps","optimizer_updates","minibatch_updates","partner_kinds","program_roles","scenario_ids","episode_context","episode_returns","episode_count","best","elapsed_seconds"):
            result[key]=copy.deepcopy(getattr(self,key))
        if self.device.type=="mps":result["mps_rng"]=torch.mps.get_rng_state()
        return result

    def load_state_dict(self,payload):
        if payload["version"]!=VERSION or payload["protocol"]!=self.protocol or payload["scenario_manifest_hash"]!=digest(self.scenarios) or payload["curriculum_bank_hash"]!=digest(self.bank) or payload["sources"]!=v2_source_hashes():
            raise ValueError("V2 checkpoint protocol/source/curriculum mismatch")
        if len(payload["envs"])!=len(self.envs):raise ValueError("V2 environment count mismatch")
        self.model.load_state_dict(payload["model"]);self.optimizers.load_state_dict(payload["optimizers"])
        for key in ("joint_steps","optimizer_updates","minibatch_updates","partner_kinds","program_roles","scenario_ids","episode_context","episode_returns","episode_count","best","elapsed_seconds"):
            setattr(self,key,copy.deepcopy(payload[key]))
        self.episode_reward_components=[Counter(value) for value in payload["episode_reward_components"]]
        for env,snapshot in zip(self.envs,payload["envs"]):
            if snapshot.get("training_reward_version")!=REWARD_VERSION:raise ValueError("Missing persisted v2 reward version")
            env.restore(snapshot)
        self.rng.bit_generator.state=copy.deepcopy(payload["rng"]);self.curriculum_rng.bit_generator.state=copy.deepcopy(payload["curriculum_rng"])
        random.setstate(payload["python_rng"]);np.random.set_state(payload["numpy_rng"]);torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type=="mps" and "mps_rng" in payload:torch.mps.set_rng_state(payload["mps_rng"].cpu())

    def export(self,path):
        return self.model.export_npz(path,{"experiment_version":VERSION,"joint_steps":self.joint_steps,"curriculum_generation_steps":self.bank["generation_steps"],"optimizer_updates":self.optimizer_updates,"protocol_sha256":digest(self.protocol),"scenario_manifest_sha256":digest(self.scenarios),"curriculum_bank_sha256":digest(self.bank),"source_sha256":digest(v2_source_hashes()),"initialization":"fresh_random","candidate":True,"feedback_enabled":False,"feature_names":self.envs[0].feature_names,"reward_version":REWARD_VERSION})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--additional-budget-authorized",action="store_true")
    parser.add_argument("--approved-total-joint-steps",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--device",choices=("cpu","mps"),default="cpu")
    parser.add_argument("--stop-after-ppo-steps",type=int)
    parser.add_argument("--resume-after-diagnostic-review",action="store_true")
    args=parser.parse_args(argv)
    # Fail before directory creation, scenario generation, sampling or updates.
    if not args.additional_budget_authorized or args.approved_total_joint_steps is None:
        parser.error("V2 is prepared only. Explicit user approval of an additional total budget is required before starting.")
    if not 0<args.approved_total_joint_steps<=500000:parser.error("Approved v2 cap must not exceed 500000")
    protocol=load_v2_protocol();output=args.output.resolve();n=protocol["training"]["environments"]
    if args.stop_after_ppo_steps is not None and (args.stop_after_ppo_steps<=0 or args.stop_after_ppo_steps%n):parser.error("Stop boundary must be a positive environment-batch multiple")
    if args.resume:
        if not (output/"run.json").exists():parser.error("Resume requires an existing authorized v2 run")
    elif output.exists() and any(output.iterdir()):parser.error("New v2 run requires an empty directory")
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/"run.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error("This v2 run is already active")
    torch.set_num_threads(1)
    if args.device=="mps" and not torch.backends.mps.is_available():parser.error("MPS is unavailable")
    if args.resume:
        record=json.loads((output/"run.json").read_text())
        if record["approved_total_joint_steps"]!=args.approved_total_joint_steps or record["sources"]!=v2_source_hashes() or record["protocol_sha256"]!=digest(protocol):parser.error("V2 resume budget/source/protocol mismatch")
        scenarios=json.loads((output/"scenarios.json").read_text())
        if record["scenario_sha256"]!=digest(scenarios):parser.error("V2 scenario manifest changed")
    else:
        scenarios=generate_manifest(counts=protocol["splits"],seed=protocol["seed"])
        atomic_json(output/"scenarios.json",scenarios);atomic_json(output/"protocol.json",protocol)
        atomic_json(output/"run.json",{"version":VERSION,"status":"authorized_v2_preparing_curriculum","approved_total_joint_steps":args.approved_total_joint_steps,"explicit_budget_flag_recorded":True,"sources":v2_source_hashes(),"protocol_sha256":digest(protocol),"scenario_sha256":digest(scenarios),"device":args.device,"feedback_enabled":False})
        for relative,expected_hash in v2_source_hashes().items():
            destination=output/"source_bundle"/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,destination)
            if file_hash(destination)!=expected_hash:raise ValueError("Source changed while archiving v2 run")
    budget=TrainingBudget(output/"sampling_budget.json",args.approved_total_joint_steps)
    stopped=False
    def signal_stop(*_):
        nonlocal stopped
        stopped=True
    old_handlers={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        review_path=output/"diagnostic_review_required.json"
        if review_path.exists():
            review=json.loads(review_path.read_text())
            if review["pause_required"] and not args.resume_after_diagnostic_review:
                parser.error("V2 is paused for diagnostic review; review the saved evidence before an explicitly acknowledged resume")
            if review["pause_required"]:
                atomic_json(output/"diagnostic_review_acknowledged.json",{"review_sha256":digest(review),"explicit_resume_review_flag_recorded":True,"budget_unchanged":args.approved_total_joint_steps})
        bank_path=output/"curriculum_bank.json"
        if not bank_path.exists():
            builder=CurriculumBuilder(protocol,scenarios)
            latest=output/"curriculum_generation_latest.json"
            if latest.exists():builder.load_state_dict(json.loads(latest.read_text()))
            ceiling=min(protocol["curriculum"]["maximum_generation_steps"],args.approved_total_joint_steps)
            while budget.read()["reserved"]["curriculum"]<ceiling:
                count=min(protocol["curriculum"]["generation_chunk"],ceiling-budget.read()["reserved"]["curriculum"],budget.remaining)
                if count<=0:break
                budget.reserve("curriculum",count);builder.advance(count)
                atomic_json(latest,builder.state_dict())
                if stopped:return 130
            bank=builder.bank();validate_curriculum_bank(bank,scenarios)
            atomic_json(bank_path,bank)
        bank=json.loads(bank_path.read_text());validate_curriculum_bank(bank,scenarios)
        trainer=NativeV2Trainer(protocol,scenarios,bank,device=args.device)
        latest=output/"latest_checkpoint.pt"
        if latest.exists():
            payload=torch.load(latest,map_location=args.device,weights_only=False)
            trainer.load_state_dict(payload);acknowledge_update(output,payload)
        else:
            trainer.export(output/"initial_actor.npz");atomic_torch_save(latest,trainer.state_dict())
        reserved=budget.read()["reserved"]
        if reserved["ppo"]<trainer.joint_steps or reserved["curriculum"]<bank["generation_steps"]:raise ValueError("Budget ledger is older than persisted sampling")
        for kind in ("reference","random"):
            path=output/f"validation_{kind}.json"
            if not path.exists():atomic_json(path,evaluate(kind,scenarios["splits"]["validation"]))
        reference=json.loads((output/"validation_reference.json").read_text());random_report=json.loads((output/"validation_random.json").read_text())
        while budget.remaining>=n:
            if args.stop_after_ppo_steps is not None and trainer.joint_steps>=args.stop_after_ppo_steps:break
            next_checkpoint=(trainer.joint_steps//protocol["training"]["checkpoint_interval"]+1)*protocol["training"]["checkpoint_interval"]
            limit=min(next_checkpoint,trainer.joint_steps+(budget.remaining//n)*n,args.stop_after_ppo_steps or 10**12)
            count=min(protocol["training"]["rollout_steps"],(limit-trainer.joint_steps)//n)
            if count<=0:break
            budget.reserve("ppo",count*n);started=time.perf_counter()
            batch=trainer.collect(count);metrics=trainer.update(batch)
            trainer.elapsed_seconds+=time.perf_counter()-started
            record={"version":VERSION,"joint_steps":trainer.joint_steps,"curriculum_generation_steps":bank["generation_steps"],"optimizer_updates":trainer.optimizer_updates,"metrics":metrics,"execution_audit":batch["audit"],"budget":budget.read()}
            payload=trainer.state_dict();payload["budget_at_checkpoint"]=budget.read();payload["acknowledgement"]={"record":record,"episodes":list(trainer.completed_episodes)}
            atomic_torch_save(latest,payload);acknowledge_update(output,payload);trainer.completed_episodes.clear()
            at_limit=trainer.joint_steps==limit
            select=checkpoint_selection_due(trainer.joint_steps,budget.remaining,n,protocol["training"]["checkpoint_interval"])
            if at_limit and not select:
                trainer.export(output/f"diagnostics/probe_actor_{trainer.joint_steps:07d}.npz")
                print(canonical({"event":"v2_sampling_probe","ppo_joint_steps":trainer.joint_steps,"curriculum_generation_steps":bank["generation_steps"],"budget":budget.read(),"ppo_joint_steps_per_training_second":trainer.joint_steps/max(trainer.elapsed_seconds,1e-9),"throughput_excludes_curriculum_and_validation":True,"used_for_checkpoint_selection":False}),flush=True)
            if at_limit and select:
                actor_path=output/f"checkpoints/actor_{trainer.joint_steps:07d}.npz";trainer.export(actor_path)
                actor=NumPyNativeActor(actor_path);report=evaluate_v2(actor,scenarios["splits"]["validation"],trainer.envs[0].feature_names)
                report["capability"]=capability(report,reference,random_report,protocol)
                report.update(joint_steps=trainer.joint_steps,actor_sha256=actor.artifact_sha256)
                score=float(np.mean([value["mean_team_deliveries"] for value in report["summary"].values()]));shutdowns=float(np.mean([value["mean_shutdowns"] for value in report["summary"].values()]))
                candidate={"score":score,"shutdowns":shutdowns,"joint_steps":trainer.joint_steps,"actor":str(actor_path.relative_to(output)),"actor_sha256":actor.artifact_sha256,"capability":report["capability"]}
                if trainer.best is None or (score,-shutdowns,-trainer.joint_steps)>(trainer.best["score"],-trainer.best["shutdowns"],-trainer.best["joint_steps"]):trainer.best=candidate
                atomic_json(output/f"validation/step_{trainer.joint_steps:07d}.json",report)
                review=diagnostic_review(report,trainer.joint_steps,protocol)
                if trainer.joint_steps>=protocol["diagnostic_checkpoints"]["first_review_ppo_steps"]:
                    atomic_json(output/f"diagnostics/step_{trainer.joint_steps:07d}.json",review)
                atomic_torch_save(output/f"checkpoints/checkpoint_{trainer.joint_steps:07d}.pt",trainer.state_dict());atomic_torch_save(latest,trainer.state_dict())
                atomic_json(output/"progress.json",{"status":"v2_candidate_training","joint_steps":trainer.joint_steps,"curriculum_generation_steps":bank["generation_steps"],"budget":budget.read(),"best":trainer.best})
                print(canonical({"event":"v2_checkpoint","joint_steps":trainer.joint_steps,"budget":budget.read(),"capability":report["capability"]}),flush=True)
                acknowledged=output/"diagnostic_review_acknowledged.json"
                if review["pause_required"] and not acknowledged.exists():
                    atomic_json(review_path,review)
                    atomic_json(output/"progress.json",{"status":"v2_paused_for_diagnostic_review","joint_steps":trainer.joint_steps,"budget":budget.read(),"best":trainer.best,"formal_ready":False,"review":review})
                    return 0
            if stopped:return 130
        atomic_json(output/"progress.json",{"status":"v2_paused" if budget.remaining>=n else "v2_budget_complete_candidate","joint_steps":trainer.joint_steps,"curriculum_generation_steps":bank["generation_steps"],"budget":budget.read(),"best":trainer.best,"formal_ready":False})
        return 0
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__=="__main__":raise SystemExit(main())
