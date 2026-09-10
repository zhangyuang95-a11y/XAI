"""Explicit paired RCPD continuations of a qualified, frozen v2 foundation.

Prepare copies existing evidence without sampling. Run requires separate
additional-budget authorization. This module never changes the frozen v2
trainer, reward, curriculum, policy, or protocol. Tiny fixtures are accessible
only through explicit Python test arguments and cannot become CLI runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch

from env.warehouse_native.feedback import FeedbackManager
from env.warehouse_native.policy import NumPyNativeActor
from .warehouse_native import acknowledge_update, atomic_torch_save, ppo_actor_objective, reserve_sampling
from .warehouse_native_common import ROOT, atomic_json, canonical, digest, file_hash
from .warehouse_native_evaluation import capability, summarize
from .warehouse_native_v2 import (NativeV2Trainer, VERSION as FOUNDATION_VERSION,
    REWARD_VERSION, load_v2_protocol, v2_source_hashes, validate_curriculum_bank)
from . import warehouse_native_feedback_run as shared


VERSION = "warehouse-native-v2-paired-continuation.v1"
FEEDBACK_PROTOCOL_VERSION = "warehouse-native-v2-feedback-protocol.v1"
BRANCHES = ("control", "feedback")
DEVELOPMENT_PARTNERS = shared.DEVELOPMENT_PARTNERS
state_digest = shared.state_digest
collect_neural_dataset = shared.collect_neural_dataset


def feedback_protocol():
    """Independent treatment specification; never merged into foundation JSON."""
    return {"version":FEEDBACK_PROTOCOL_VERSION,"feedback":{
        "depths":[4,6,8],"leaves":[16,32,64],"target":"detached_tree_distribution",
        "direction":"KL(nn||tree)","complexity":"candidate_selection_only",
        "lambda_max":.01,"ramp_steps":100000,"extract_interval":50000,
        "minimum_fidelity":.9,"minimum_critical_fidelity":.85,"maximum_mean_kl":.35,
        "maximum_performance_drop_fraction":.1},
        "explanation":{"categories":["narrow_passage","shared_pickup","shared_charger"]},
        "additional_long_training_authorized":False,"formal_explanation_ready":False}


def execution_sources():
    paths = [Path(__file__),Path(shared.__file__),ROOT/"env/warehouse_native/feedback.py"]
    paths += sorted((ROOT/"core").glob("*.py"))
    return {str(path.relative_to(ROOT)):file_hash(path) for path in paths}


def manager_config(native_protocol,treatment):
    combined = {"seed":native_protocol["seed"],"warmup_gate":native_protocol["warmup_gate"],
        "feedback":treatment["feedback"],"explanation":treatment["explanation"]}
    return shared.feedback_config(combined)


@contextmanager
def preserve_rng(trainer=None):
    state = (random.getstate(),np.random.get_state(),torch.get_rng_state())
    mps = torch.mps.get_rng_state() if trainer is not None and trainer.device.type=="mps" else None
    local = None if trainer is None else (copy.deepcopy(trainer.rng.bit_generator.state),
        copy.deepcopy(trainer.curriculum_rng.bit_generator.state))
    try:
        yield
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        if mps is not None:torch.mps.set_rng_state(mps)
        if local is not None:
            trainer.rng.bit_generator.state=local[0];trainer.curriculum_rng.bit_generator.state=local[1]


class V2FeedbackTrainer(NativeV2Trainer):
    """Keep v2 sampling/physics/dual optimizers; add NN-only Actor KL."""
    def __init__(self,protocol,scenarios,bank,*,treatment=None,branch="control",device="cpu",test_fixture=False):
        if branch not in BRANCHES:raise ValueError("Unknown continuation branch")
        super().__init__(protocol,scenarios,bank,device=device)
        self.treatment=copy.deepcopy(treatment or feedback_protocol())
        if self.treatment["version"]!=FEEDBACK_PROTOCOL_VERSION:raise ValueError("Unknown feedback protocol")
        self.branch=branch;self.test_fixture=bool(test_fixture)
        self.feedback_enabled=branch=="feedback"
        self.feedback=FeedbackManager(self.envs[0].feature_names,manager_config(protocol,self.treatment)) if self.feedback_enabled else None
        self.continuation=None

    def update(self,batch):
        # This exact path also avoids zero-valued graph additions changing any
        # floating-point operation or optimizer ordering in the paired control.
        if not self.feedback_enabled or self.feedback.current_lambda<=0:
            return {**super().update(batch),"feedback_loss":0.,"feedback_gradient_norm":0.,
                "feedback_rows":0.,"feedback_kl":0.,"feedback_lambda":0.}
        obs=batch["observations"].reshape(-1,self.model.obs_dim)
        states=np.repeat(batch["states"][:,:,None,:],2,axis=2).reshape(-1,self.model.state_dim)
        roles=np.tile([0,1],len(obs)//2);trainable=batch["trainable"].reshape(-1)
        advantages=batch["advantages"].reshape(-1).copy();selected=trainable>0
        if selected.any():advantages[selected]=(advantages[selected]-advantages[selected].mean())/(advantages[selected].std()+1e-8)
        data={"obs":obs,"states":states,"roles":roles,"trainable":trainable,"actions":batch["actions"].reshape(-1),
            "old":batch["old_log_probs"].reshape(-1),"advantages":advantages,"returns":batch["returns"].reshape(-1)}
        data={key:torch.as_tensor(value,device=self.device) for key,value in data.items()};metrics=[]
        for _ in range(self.cfg["epochs"]):
            permutation=self.rng.permutation(len(obs))
            for start in range(0,len(obs),self.cfg["minibatch"]):
                index=torch.as_tensor(permutation[start:start+self.cfg["minibatch"]],device=self.device)
                logits=self.model.actor_logits(data["obs"][index])
                actor_loss,part=ppo_actor_objective(logits,data["actions"][index],data["old"][index],
                    data["advantages"][index],data["trainable"][index],clip=self.cfg["clip"],entropy=self.cfg["entropy"])
                critic_loss=.5*(self.model.values(data["states"][index],data["roles"][index])-data["returns"][index]).square().mean()
                mask=data["trainable"][index].bool();feedback_loss=logits.sum()*0
                feedback_metrics={"feedback_loss":0.,"feedback_gradient_norm":0.,"feedback_rows":int(mask.sum()),"feedback_kl":0.}
                if mask.any():
                    feedback_loss,evidence=self.feedback.loss(logits[mask],data["obs"][index][mask].detach().cpu().numpy())
                    gradients=torch.autograd.grad(feedback_loss,tuple(self.model.actor.parameters()),retain_graph=True,allow_unused=True)
                    norm=torch.sqrt(sum((g.square().sum() for g in gradients if g is not None),logits.new_zeros(())))
                    feedback_metrics.update(feedback_loss=float(feedback_loss.detach()),feedback_gradient_norm=float(norm.detach()),feedback_kl=evidence.get("kl",0.))
                result=self.optimizers.step(actor_loss+feedback_loss,critic_loss,train_actor=bool(mask.any()))
                metrics.append({**part,**result,**feedback_metrics,"critic_loss":float(critic_loss.detach()),
                    "feedback_lambda":self.feedback.current_lambda});self.minibatch_updates+=1
        self.optimizer_updates+=1
        return {key:float(np.mean([value.get(key,0.) for value in metrics])) for key in set().union(*(value.keys() for value in metrics))}

    def state_dict(self):
        return {"version":VERSION,"native_state":super().state_dict(),"treatment":copy.deepcopy(self.treatment),
            "execution_sources":execution_sources(),"branch":self.branch,"test_fixture":self.test_fixture,
            "feedback_enabled":self.feedback_enabled,"feedback_state":self.feedback.state_dict() if self.feedback_enabled else None,
            "continuation":copy.deepcopy(self.continuation)}

    def load_state_dict(self,payload):
        if (payload["version"]!=VERSION or payload["treatment"]!=self.treatment
                or payload["execution_sources"]!=execution_sources() or payload["branch"]!=self.branch
                or payload["test_fixture"]!=self.test_fixture or payload["feedback_enabled"]!=self.feedback_enabled):
            raise ValueError("V2 continuation identity, feedback mode, or source mismatch")
        super().load_state_dict(payload["native_state"])
        self.continuation=copy.deepcopy(payload["continuation"])
        if self.feedback_enabled:self.feedback.load_state_dict(payload["feedback_state"])
        elif payload["feedback_state"] is not None:raise ValueError("Control cannot contain feedback state")

    def export(self,path):
        context=self.continuation or {}
        return self.model.export_npz(path,{"experiment_version":VERSION,"joint_steps":self.joint_steps,
            "curriculum_generation_steps":self.bank["generation_steps"],"optimizer_updates":self.optimizer_updates,
            "protocol_sha256":digest(self.protocol),"scenario_manifest_sha256":digest(self.scenarios),
            "curriculum_bank_sha256":digest(self.bank),"source_sha256":digest(v2_source_hashes()),
            "continuation_source_sha256":digest(execution_sources()),"feedback_protocol_sha256":digest(self.treatment),
            "initialization":"continued_exact_frozen_v2_foundation","candidate":True,"test_fixture":self.test_fixture,
            "feedback_enabled":self.feedback_enabled,"feature_names":self.envs[0].feature_names,"reward_version":REWARD_VERSION,
            "pair_id":context.get("pair_id"),"foundation_joint_steps":context.get("foundation_joint_steps"),
            "foundation_actor_sha256":context.get("foundation_actor_sha256"),"formal_ready":False})


def verified_report(path,scenarios,*,allow_test_fixture=False):
    report=json.loads(Path(path).read_text())
    if report.get("test_fixture") and not allow_test_fixture:raise ValueError("Synthetic evidence cannot qualify a real foundation")
    expected={scene["id"]:scene["fingerprint"] for scene in scenarios}
    for partner in DEVELOPMENT_PARTNERS:
        rows=[row for row in report["rows"] if row["partner"]==partner]
        if len(rows)!=len(expected) or {row["scenario_id"] for row in rows}!=set(expected):
            raise ValueError("Validation is not the exact frozen scene set")
        if any(row["initial_fingerprint"]!=expected[row["scenario_id"]] or row["nn_action_overrides"]!=0 for row in rows):
            raise ValueError("Validation scene or neural action provenance mismatch")
        actual=summarize(rows);claimed=report["summary"][partner]
        if any(key not in claimed or canonical(value)!=canonical(claimed[key]) for key,value in actual.items()):
            raise ValueError("Validation summary does not match recorded episodes")
    return report


def fork_payload(payload,branch,manager,*,pair_id,additional_steps,treatment,actor_sha256,test_fixture=False):
    if branch not in BRANCHES or payload["version"]!=FOUNDATION_VERSION or payload.get("feedback_enabled",False):
        raise ValueError("Only an unchanged pure v2 foundation can be forked")
    if payload.get("feedback_state") is not None:raise ValueError("Foundation already contains feedback")
    return {"version":VERSION,"native_state":copy.deepcopy(payload),"treatment":copy.deepcopy(treatment),
        "execution_sources":execution_sources(),"branch":branch,"test_fixture":bool(test_fixture),
        "feedback_enabled":branch=="feedback","feedback_state":manager.state_dict() if branch=="feedback" else None,
        "continuation":{"version":VERSION,"pair_id":pair_id,"branch":branch,
            "foundation_joint_steps":payload["joint_steps"],"foundation_actor_sha256":actor_sha256,
            "additional_budget":additional_steps,"additional_joint_steps":0,"initial_rng_offset":0,
            "last_refresh_step":None,"latest_gate":None,"best_continuation":None,
            "pending_refresh_step":payload["joint_steps"],"auxiliary_actual_joint_steps":0,"audit_events":[]}}


def prepare_pair(foundation,checkpoint,output,additional_steps,*,expected_protocol=None,treatment=None,allow_test_fixture=False):
    foundation,checkpoint,output=Path(foundation).resolve(),Path(checkpoint).resolve(),Path(output).resolve()
    if output==foundation or output in foundation.parents or (foundation in output.parents and output.name in {"checkpoints","validation"}):
        raise ValueError("Pair output must not replace foundation artifacts")
    if output.exists() and any(p.name!="prepare_report.json" for p in output.iterdir()):raise ValueError("Fresh pair output required")
    output.mkdir(parents=True,exist_ok=True)
    evidence={"foundation_checkpoint":str(checkpoint),"test_fixture":bool(allow_test_fixture)}
    try:
        native=load_v2_protocol() if expected_protocol is None else copy.deepcopy(expected_protocol)
        treatment=copy.deepcopy(treatment or feedback_protocol())
        if (native!=load_v2_protocol() or treatment!=feedback_protocol()) and not allow_test_fixture:
            raise ValueError("Production preparation requires the frozen protocols")
        n=native["training"]["environments"]
        if type(additional_steps) is not int or additional_steps<=0 or additional_steps%n:raise ValueError("Equal positive batched additional budget required")
        if treatment["feedback"]["extract_interval"]%n:raise ValueError("Extraction interval must align with environment count")
        if not checkpoint.is_relative_to(foundation/"checkpoints"):raise ValueError("Archived foundation checkpoint required")
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        scenarios=json.loads((foundation/"scenarios.json").read_text());bank=json.loads((foundation/"curriculum_bank.json").read_text())
        if payload["version"]!=FOUNDATION_VERSION or payload["protocol"]!=native or payload["sources"]!=v2_source_hashes():
            raise ValueError("Foundation source or protocol changed")
        if payload.get("test_fixture") and not allow_test_fixture:raise ValueError("Fixture checkpoint cannot qualify production")
        if payload.get("feedback_enabled",False) or payload.get("feedback_state") is not None:raise ValueError("Foundation is not pure RL")
        if payload["scenario_manifest_hash"]!=digest(scenarios) or payload["curriculum_bank_hash"]!=digest(bank):
            raise ValueError("Foundation scene or curriculum binding failed")
        validate_curriculum_bank(bank,scenarios)
        if payload["curriculum_generation_steps"]!=bank["generation_steps"]:raise ValueError("Curriculum accounting mismatch")
        step=payload["joint_steps"];evidence["foundation_joint_steps"]=step
        if step<native["warmup_gate"]["minimum_joint_steps"] or step%n:raise ValueError("Foundation has not completed the registered warmup")
        if step+bank["generation_steps"]>native["authorized_run"]["total_training_environment_steps_max"]:
            raise ValueError("Foundation total PPO plus curriculum exceeds its registered budget")
        if set(payload["optimizers"])!={"actor","critic"}:raise ValueError("Both frozen Adam states are required")
        actor_path=foundation/f"checkpoints/actor_{step:07d}.npz";actor=NumPyNativeActor(actor_path)
        shared._same_actor(payload,actor)
        if (actor.metadata.get("experiment_version")!=FOUNDATION_VERSION or actor.metadata.get("joint_steps")!=step
                or actor.metadata.get("feedback_enabled") is not False or actor.metadata.get("curriculum_bank_sha256")!=digest(bank)
                or actor.metadata.get("protocol_sha256")!=digest(native) or actor.metadata.get("source_sha256")!=digest(v2_source_hashes())):
            raise ValueError("Actor metadata does not bind this pure v2 foundation")
        report_path=foundation/f"validation/step_{step:07d}.json"
        report=verified_report(report_path,scenarios["splits"]["validation"],allow_test_fixture=allow_test_fixture)
        reference=verified_report(foundation/"validation_reference.json",scenarios["splits"]["validation"],allow_test_fixture=allow_test_fixture)
        random_report=verified_report(foundation/"validation_random.json",scenarios["splits"]["validation"],allow_test_fixture=allow_test_fixture)
        if report.get("actor_sha256")!=actor.sha256 or report.get("joint_steps")!=step:raise ValueError("Validation is bound to a different Actor")
        gate=capability(report,reference,random_report,native)
        warmup=shared.warmup_capability(report,reference,native,step)
        evidence.update(capability=gate,warmup_capability=warmup,validation_summary=report["summary"],
            reference_summary=reference["summary"],random_summary=random_report["summary"],foundation_actor_sha256=actor.sha256)
        if not gate["eligible"] or not warmup["eligible"]:raise ValueError("Foundation capability gate failed")
        sets=[]
        for name in ("train","extraction","validation"):
            scenes=scenarios["splits"][name];fingerprints=[scene["fingerprint"] for scene in scenes]
            if len(scenes)<2 or len(fingerprints)!=len(set(fingerprints)) or any(not scene["id"].startswith(name+"_") for scene in scenes):
                raise ValueError("Invalid independent development pool")
            sets.append(set(fingerprints))
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError("Development physical scenes overlap")
    except (ValueError,KeyError,OSError,RuntimeError) as error:
        result={"version":VERSION,"status":"blocked","reason":str(error),**evidence,"training_started":False,
            "environment_transitions":0,"additional_training_authorized":False,"formal_ready":False}
        atomic_json(output/"prepare_report.json",result);return result
    interval=treatment["feedback"]["extract_interval"];refreshes=math.ceil(additional_steps/interval)+1
    count=min(32,len(scenarios["splits"]["train"]),len(scenarios["splits"]["extraction"]))
    per_refresh=scenarios["configuration"]["horizon"]*(3*len(scenarios["splits"]["validation"])+2*count)
    plan={"version":VERSION,"foundation_checkpoint_sha256":file_hash(checkpoint),"foundation_actor_sha256":actor.sha256,
        "foundation_joint_steps":step,"foundation_curriculum_generation_steps":bank["generation_steps"],
        "additional_joint_steps_per_branch":additional_steps,"target_absolute_joint_steps":step+additional_steps,
        "training_sources":v2_source_hashes(),"execution_sources":execution_sources(),
        "scenario_manifest_sha256":digest(scenarios),"curriculum_bank_sha256":digest(bank),
        "protocol_sha256":digest(native),"feedback_protocol_sha256":digest(treatment),"capability":gate,
        "warmup_capability":warmup,"foundation_directory":str(foundation),"foundation_checkpoint":str(checkpoint),
        "paired_initial_rng_offset":{b:0 for b in BRANCHES},"curriculum_schedule":"absolute_foundation_plus_additional_ppo_steps",
        "curriculum_regenerated":False,"test_fixture":bool(allow_test_fixture),
        "extraction":{"fit_pool":"train","selection_pool":"extraction","scenarios_per_pool":count,
            "selection_pool_role":"reused_development_not_final_test","refresh_interval":interval,
            "profiles":list(shared.EXTRACTION_PROFILES),"sampling":"same_frozen_actor_stochastic_actions_and_soft_probabilities"},
        "validation":{"pool":"validation","role":"reused_development_model_selection","final_test_read":False,
            "explanation_test_read":False,"play_read":False},
        "auxiliary_sampling":{"separate_from_optimizer_joint_steps":True,"actual_count_scope":"completed_checkpointed_audits_only",
            "cap_per_branch":per_refresh*(refreshes+2),"planned_refreshes":refreshes,"crash_recovery_refresh_allowance":2},
        "additional_training_authorized":False,"status":"prepared_not_authorized","formal_ready":False,
        "preserved_checkpoint_fields":{key:state_digest(value) for key,value in payload.items()}}
    atomic_json(output/"protocol.json",native);atomic_json(output/"feedback_protocol.json",treatment)
    atomic_json(output/"scenarios.json",scenarios);atomic_json(output/"curriculum_bank.json",bank)
    for source,dest in ((checkpoint,"foundation_checkpoint.pt"),(actor_path,"foundation_actor.npz"),
            (report_path,"foundation_validation.json"),(foundation/"validation_reference.json","validation_reference.json"),
            (foundation/"validation_random.json","validation_random.json")):
        shutil.copyfile(source,output/dest)
    frozen=("foundation_checkpoint.pt","foundation_actor.npz","foundation_validation.json","validation_reference.json",
        "validation_random.json","protocol.json","feedback_protocol.json","scenarios.json","curriculum_bank.json")
    plan["frozen_files"]={name:file_hash(output/name) for name in frozen};plan["pair_id"]=digest(plan)
    manager=FeedbackManager(actor.metadata["feature_names"],manager_config(native,treatment))
    for branch in BRANCHES:
        folder=output/branch;folder.mkdir()
        initial=fork_payload(payload,branch,manager,pair_id=plan["pair_id"],additional_steps=additional_steps,
            treatment=treatment,actor_sha256=actor.sha256,test_fixture=allow_test_fixture)
        assert state_digest(initial["native_state"])==state_digest(payload)
        atomic_torch_save(folder/"initial_checkpoint.pt",initial)
        reserve_sampling(folder/"sampling_budget.json",0,cap=additional_steps)
        reserve_sampling(folder/"auxiliary_budget.json",0,cap=plan["auxiliary_sampling"]["cap_per_branch"])
    atomic_json(output/"pair.json",plan)
    result={"version":VERSION,"status":"prepared_not_authorized","pair_id":plan["pair_id"],"test_fixture":plan["test_fixture"],
        "training_started":False,"environment_transitions":0,"additional_training_authorized":False,
        "additional_joint_steps_per_branch":additional_steps,"formal_ready":False}
    atomic_json(output/"prepare_report.json",result);return result


def refresh_feedback(trainer,context,output,plan,scenarios,reference,random_report):
    """Reuse frozen-Actor evidence collection while preserving all training RNG."""
    before=[digest(env.snapshot()) for env in trainer.envs]
    with preserve_rng(trainer):
        result=shared.refresh_feedback(trainer,context,output,plan,scenarios,reference,random_report)
    if before!=[digest(env.snapshot()) for env in trainer.envs]:raise AssertionError("Feedback audit changed live training environments")
    return result


def save_complete(output,trainer,context,acknowledgement=None):
    context["additional_joint_steps"]=trainer.joint_steps-context["foundation_joint_steps"]
    trainer.continuation=copy.deepcopy(context);payload=trainer.state_dict()
    if acknowledgement is not None:payload["acknowledgement"]=acknowledgement
    atomic_torch_save(Path(output)/"latest_checkpoint.pt",payload)
    acknowledge_update(output,payload)
    return payload


def run_branch(pair,branch,*,additional_budget_authorized=False,device="cpu",stop_after_additional=None,allow_test_fixture=False):
    if not additional_budget_authorized:raise PermissionError("Additional control/feedback training has not been authorized")
    if branch not in BRANCHES:raise ValueError("Unknown paired continuation branch")
    pair=Path(pair).resolve();plan=json.loads((pair/"pair.json").read_text())
    if plan.get("test_fixture") and not allow_test_fixture:raise PermissionError("Fixture pairs cannot run through production entry points")
    if digest({k:v for k,v in plan.items() if k!="pair_id"})!=plan["pair_id"]:raise ValueError("Prepared pair description changed")
    if plan["version"]!=VERSION or plan["training_sources"]!=v2_source_hashes() or plan["execution_sources"]!=execution_sources():
        raise ValueError("Continuation source contract changed")
    expected={"foundation_checkpoint.pt","foundation_actor.npz","foundation_validation.json","validation_reference.json",
        "validation_random.json","protocol.json","feedback_protocol.json","scenarios.json","curriculum_bank.json"}
    if set(plan["frozen_files"])!=expected or any(file_hash(pair/p)!=h for p,h in plan["frozen_files"].items()):
        raise ValueError("Prepared evidence changed")
    native=json.loads((pair/"protocol.json").read_text());treatment=json.loads((pair/"feedback_protocol.json").read_text())
    scenes=json.loads((pair/"scenarios.json").read_text());bank=json.loads((pair/"curriculum_bank.json").read_text())
    if (digest(native)!=plan["protocol_sha256"] or digest(treatment)!=plan["feedback_protocol_sha256"]
            or digest(scenes)!=plan["scenario_manifest_sha256"] or digest(bank)!=plan["curriculum_bank_sha256"]):
        raise ValueError("Prepared protocol, scenes, or curriculum changed")
    if not plan["capability"]["eligible"] or not plan["warmup_capability"]["eligible"]:raise ValueError("Unqualified pair cannot run")
    output=pair/branch;cap=plan["additional_joint_steps_per_branch"];base=plan["foundation_joint_steps"]
    n=native["training"]["environments"];stop=cap if stop_after_additional is None else stop_after_additional
    if type(stop) is not int or not 0<=stop<=cap or stop%n:raise ValueError("Invalid additional-step stop boundary")
    interval=treatment["feedback"]["extract_interval"]
    rollout_joint_steps=native["training"]["rollout_steps"]*n
    if stop not in (0,cap) and (stop%interval)%rollout_joint_steps:
        raise ValueError("Temporary stop must preserve a scheduled PPO update boundary")
    locks=[]
    try:
        for path in (pair/"pair.lock",Path(plan["foundation_directory"])/"run.lock"):
            handle=path.open("a+");locks.append(handle)
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError("Foundation or paired continuation is already running") from None
        execution={"device":str(device),"torch_version":str(torch.__version__),"numpy_version":np.__version__}
        execution_path=pair/"execution.json"
        if execution_path.exists() and json.loads(execution_path.read_text())!=execution:raise ValueError("Paired execution device or libraries differ")
        atomic_json(execution_path,execution)
        reference=verified_report(pair/"validation_reference.json",scenes["splits"]["validation"],allow_test_fixture=allow_test_fixture)
        random_report=verified_report(pair/"validation_random.json",scenes["splits"]["validation"],allow_test_fixture=allow_test_fixture)
        foundation_report=verified_report(pair/"foundation_validation.json",scenes["splits"]["validation"],allow_test_fixture=allow_test_fixture)
        if not capability(foundation_report,reference,random_report,native)["eligible"]:raise ValueError("Foundation capability no longer verified")
        initial=torch.load(output/"initial_checkpoint.pt",map_location="cpu",weights_only=False)
        if {k:state_digest(v) for k,v in initial["native_state"].items()}!=plan["preserved_checkpoint_fields"]:
            raise ValueError("Initial dual-optimizer/RNG/physical state differs from the common foundation")
        path=output/"latest_checkpoint.pt"
        payload=torch.load(path,map_location="cpu",weights_only=False) if path.exists() else initial
        context=copy.deepcopy(payload["continuation"]);absolute=payload["native_state"]["joint_steps"]
        if (context["pair_id"]!=plan["pair_id"] or context["branch"]!=branch or context["additional_budget"]!=cap
                or context["foundation_joint_steps"]!=base or context["additional_joint_steps"]!=absolute-base):
            raise ValueError("Continuation identity or counters changed")
        log=output/"training.jsonl"
        if log.exists() and any(json.loads(line)["joint_steps"]>absolute for line in log.read_text().splitlines() if line.strip()):
            raise ValueError("Newer acknowledged update exists; refusing replay")
        trainer=V2FeedbackTrainer(native,scenes,bank,treatment=treatment,branch=branch,device=device,test_fixture=plan["test_fixture"])
        trainer.load_state_dict(payload);acknowledge_update(output,payload)
        reserved=json.loads((output/"sampling_budget.json").read_text())["reserved_joint_steps"];completed=absolute-base
        if reserved<completed:raise ValueError("Sampling ledger is older than persisted transitions")
        discarded=reserved-completed;target=base+min(stop,completed+cap-reserved)
        atomic_json(output/"authorization.json",{"version":VERSION,"additional_budget_authorized":True,
            "per_branch_additional_budget":cap,"auxiliary_budget_cap":plan["auxiliary_sampling"]["cap_per_branch"],
            "pid":os.getpid(),"branch":branch,"pair_id":plan["pair_id"],"test_fixture":plan["test_fixture"]})
        stopped=False
        def request_stop(signum,frame):
            nonlocal stopped
            stopped=True
        handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            if context["pending_refresh_step"] is not None:
                if context["pending_refresh_step"]!=trainer.joint_steps:raise ValueError("Pending audit belongs to a different checkpoint")
                refresh_feedback(trainer,context,output,plan,scenes,reference,random_report);save_complete(output,trainer,context)
            def progress(status):
                return {"version":VERSION,"status":status,"branch":branch,"test_fixture":plan["test_fixture"],
                    "joint_steps":trainer.joint_steps,"foundation_joint_steps":base,"additional_joint_steps":trainer.joint_steps-base,
                    "foundation_curriculum_generation_steps":bank["generation_steps"],"additional_curriculum_generation_steps":0,
                    "additional_budget":cap,"auxiliary_actual_joint_steps":context["auxiliary_actual_joint_steps"],
                    "reserved_additional_upper_bound":reserved,"discarded_after_crash_upper_bound":discarded,
                    "latest_gate":context["latest_gate"],"best_continuation":context["best_continuation"],"formal_ready":False}
            if trainer.joint_steps>=target:
                result=progress("budget_complete" if completed==cap else "no_available_sampling_budget")
                atomic_json(output/"progress.json",result);return result
            while trainer.joint_steps<target:
                interval=treatment["feedback"]["extract_interval"]
                next_refresh=base+((trainer.joint_steps-base)//interval+1)*interval
                boundary=min(target,next_refresh)
                count=min(native["training"]["rollout_steps"],(boundary-trainer.joint_steps)//n)
                if count<=0:raise ValueError("Incompatible sampling boundary")
                if trainer.feedback_enabled:
                    gate=context["latest_gate"]
                    trainer.feedback.update(trainer.joint_steps,gate["validation_score"],
                        capability_eligible=gate["capability_eligible"],reference_score=gate["reference_score"])
                tick=time.perf_counter();reserved=reserve_sampling(output/"sampling_budget.json",count*n,cap=cap)
                batch=trainer.collect(count);collected=time.perf_counter();metrics=trainer.update(batch);elapsed=time.perf_counter()-tick
                trainer.elapsed_seconds+=elapsed
                record={"joint_steps":trainer.joint_steps,"foundation_joint_steps":base,"additional_joint_steps":trainer.joint_steps-base,
                    "branch":branch,"test_fixture":plan["test_fixture"],"optimizer_updates":trainer.optimizer_updates,
                    "curriculum_generation_steps":bank["generation_steps"],"collection_seconds":collected-tick,
                    "update_seconds":elapsed-(collected-tick),"metrics":metrics,"execution_audit":batch["audit"],
                    "reserved_additional_upper_bound":reserved,"discarded_after_crash_upper_bound":discarded,
                    "feedback_lambda":trainer.feedback.current_lambda if trainer.feedback_enabled else 0.}
                refresh_due=trainer.joint_steps in (next_refresh,base+cap)
                if refresh_due:context["pending_refresh_step"]=trainer.joint_steps
                save_complete(output,trainer,context,{"record":record,"episodes":list(trainer.completed_episodes)})
                trainer.completed_episodes.clear()
                if refresh_due:
                    refresh_feedback(trainer,context,output,plan,scenes,reference,random_report)
                    saved=save_complete(output,trainer,context)
                    atomic_torch_save(output/f"checkpoints/checkpoint_{trainer.joint_steps:07d}.pt",saved)
                result=progress("paused_after_complete_update" if stopped else "running")
                atomic_json(output/"progress.json",result)
                if stopped:return result
            result=progress("budget_complete" if trainer.joint_steps-base==cap else "paused_or_sampling_reservations_exhausted")
            atomic_json(output/"progress.json",result);return result
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    finally:
        for handle in reversed(locks):handle.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--foundation",type=Path);parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--additional-steps",type=int)
    parser.add_argument("--branch",choices=(*BRANCHES,"both"),default="both")
    parser.add_argument("--additional-budget-authorized",action="store_true")
    parser.add_argument("--device",choices=("cpu","mps"),default="cpu");parser.add_argument("--stop-after-additional",type=int)
    args=parser.parse_args(argv)
    if args.prepare:
        if args.foundation is None or args.checkpoint is None or args.additional_steps is None:
            parser.error("Preparation requires foundation, archived checkpoint, and equal additional steps")
        report=prepare_pair(args.foundation,args.checkpoint,args.output,args.additional_steps)
        print(canonical(report),flush=True);return 0 if report["status"]=="prepared_not_authorized" else 2
    if not args.additional_budget_authorized:parser.error("Additional feedback/control training is not authorized")
    if args.additional_steps is not None or args.foundation is not None or args.checkpoint is not None:
        parser.error("Run uses the frozen prepared pair and cannot change its source or budget")
    torch.set_num_threads(1)
    for branch in BRANCHES if args.branch=="both" else (args.branch,):
        report=run_branch(args.output,branch,additional_budget_authorized=True,device=args.device,stop_after_additional=args.stop_after_additional)
        print(canonical(report),flush=True)
        if report["status"]=="paused_after_complete_update":return 130
    return 0


if __name__=="__main__":raise SystemExit(main())
