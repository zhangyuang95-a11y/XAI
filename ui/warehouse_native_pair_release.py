"""Read-only inspection of the frozen v2 paired runner's actual output layout.

The outer release index binds existing files; it is not a replacement training
report. Initial .pt files are HASHED ONLY. Equality of Adam, RNG and environment
contents is evidence of the bound runner's pre-run checks, not an independent
pickle/tensor comparison here. No optimization, tree fitting or checkpoint load.
"""
from __future__ import annotations

from hashlib import sha256
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from backend.training import warehouse_native_v2_feedback_run as runner
from backend.training.warehouse_native_common import canonical,digest,file_hash
from backend.training.warehouse_native_evaluation import capability,critical_groups
from backend.training.warehouse_native_v2 import load_v2_protocol,v2_source_hashes,validate_curriculum_bank,REWARD_VERSION
from core.program import ExecutableProgram
from core.policy_program_regularizer import program_complexity
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.feedback import FeedbackManager
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor,ACTIONS
from env.warehouse_native.scenarios import reset_scenario
from ui import warehouse_native_release as release

VERSION="warehouse-native-v2-pair-end.v1"
SELECTION_VERSION="warehouse-native-v2-pair-selection.v1"
FROZEN_FILES={"foundation_checkpoint.pt","foundation_actor.npz","foundation_validation.json","validation_reference.json",
    "validation_random.json","protocol.json","feedback_protocol.json","scenarios.json","curriculum_bank.json"}
NATIVE_FIELDS={"version","model","optimizers","obs_dim","state_dim","protocol","scenario_manifest_hash","curriculum_bank_hash",
    "curriculum_generation_steps","sources","rng","curriculum_rng","python_rng","numpy_rng","torch_rng","envs",
    "episode_reward_components","joint_steps","optimizer_updates","minibatch_updates","partner_kinds","program_roles",
    "scenario_ids","episode_context","episode_returns","episode_count","best","elapsed_seconds"}
SAME_INITIAL_SCOPE="bound_frozen_runner_checked_native_fields_before_training; checkpoint_pickle_not_read_or_independently_recomputed"


def _equal(a,b,message):
    if canonical(a)!=canonical(b):raise ValueError(message)


def _number(value,minimum=0,maximum=None):
    if type(value) not in (int,float) or not math.isfinite(value) or value<minimum or (maximum is not None and value>maximum):
        raise ValueError("Invalid finite paired metric")
    return float(value)


def _hash(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):raise ValueError("Invalid paired SHA-256")


class BoundPair:
    """Only read indexed files at their actual unchanged runner-relative names."""
    def __init__(self,record,root):
        release._reject_fixture(record)
        if record["version"]!=VERSION:raise ValueError("Unsupported paired endpoint index")
        self.root=Path(root).resolve();self.plan_path=release.bound_path(self.root,record["pair"])
        if self.plan_path.name!="pair.json":raise ValueError("Pair plan must retain its runner filename")
        self.directory=self.plan_path.parent;self.files=record["files"];self.used=set();self.bound={}
        for name,contract in self.files.items():
            logical=Path(name)
            if logical.is_absolute() or ".." in logical.parts or str(logical)!=name:raise ValueError("Invalid paired output name")
            path=release.bound_path(self.root,contract)
            if path!=(self.directory/logical).resolve():raise ValueError("Pair index changes the original output layout")
            self.bound[name]=path
        self.plan=release._json(self.plan_path);release._reject_fixture(self.plan)

    def path(self,name):
        if name not in self.bound:raise ValueError("Missing paired evidence: "+name)
        self.used.add(name);return self.bound[name]

    def json(self,name):
        value=release._json(self.path(name));release._reject_fixture(value);return value

    def rows(self,name):return [release.parse_json(line) for line in self.path(name).read_text().splitlines() if line.strip()]


def verify_actor_metadata(actor,protocol,scenarios,plan=None,step=None,branch=None):
    m=actor.metadata;release._finite(m);release._reject_fixture(m)
    if protocol!=load_v2_protocol():raise ValueError("Paired continuation requires the unchanged v2 foundation protocol")
    required={"experiment_version":runner.VERSION,"protocol_sha256":digest(protocol),
        "scenario_manifest_sha256":digest(scenarios),"source_sha256":digest(v2_source_hashes()),
        "continuation_source_sha256":digest(runner.execution_sources()),"feedback_protocol_sha256":digest(runner.feedback_protocol()),
        "initialization":"continued_exact_frozen_v2_foundation","candidate":True,"reward_version":REWARD_VERSION,"formal_ready":False}
    for key,value in required.items():_equal(m[key],value,"Paired Actor metadata mismatch: "+key)
    if type(m["feedback_enabled"]) is not bool:raise ValueError("Paired Actor feedback mode must be Boolean")
    release._integer(m["joint_steps"],1);release._integer(m["optimizer_updates"],1)
    if plan is not None:
        for key,value in {"pair_id":plan["pair_id"],"foundation_joint_steps":plan["foundation_joint_steps"],
                "foundation_actor_sha256":plan["foundation_actor_sha256"],"curriculum_bank_sha256":plan["curriculum_bank_sha256"],
                "curriculum_generation_steps":plan["foundation_curriculum_generation_steps"]}.items():
            _equal(m[key],value,"Paired Actor common-source mismatch: "+key)
    if step is not None and m["joint_steps"]!=step:raise ValueError("Paired Actor step differs from development directory")
    if branch is not None and m["feedback_enabled"]!=(branch=="feedback"):raise ValueError("Paired Actor branch identity mismatch")
    return {"experiment_version":runner.VERSION,"feedback_enabled":m["feedback_enabled"]}


def verify_plan(plan,protocol,scenarios):
    release._reject_fixture(plan);release._finite(plan)
    if plan["version"]!=runner.VERSION or digest({k:v for k,v in plan.items() if k!="pair_id"})!=plan["pair_id"]:
        raise ValueError("Pair plan digest/version changed")
    if protocol!=load_v2_protocol() or plan["training_sources"]!=v2_source_hashes() or plan["execution_sources"]!=runner.execution_sources():
        raise ValueError("Pair source or protocol is not the frozen executed runner")
    treatment=runner.feedback_protocol();n=protocol["training"]["environments"];interval=treatment["feedback"]["extract_interval"]
    base=plan["foundation_joint_steps"];cap=plan["additional_joint_steps_per_branch"]
    generated=plan["foundation_curriculum_generation_steps"]
    release._integer(base,protocol["warmup_gate"]["minimum_joint_steps"]);release._integer(cap,n);release._integer(generated)
    if base%n or cap%n or plan["target_absolute_joint_steps"]!=base+cap or base+generated>protocol["authorized_run"]["total_training_environment_steps_max"]:
        raise ValueError("Pair absolute/additional/foundation budget mismatch")
    # No new cap is invented: the frozen runner explicitly authorizes cap for
    # each branch. It cannot be inferred from the original foundation allowance.
    for key,value in {"protocol_sha256":digest(protocol),"feedback_protocol_sha256":digest(treatment),
        "scenario_manifest_sha256":digest(scenarios),"paired_initial_rng_offset":{"control":0,"feedback":0},
        "curriculum_schedule":"absolute_foundation_plus_additional_ppo_steps","curriculum_regenerated":False,
        "test_fixture":False,"additional_training_authorized":False,"status":"prepared_not_authorized","formal_ready":False}.items():
        _equal(plan[key],value,"Frozen pair preparation differs: "+key)
    count=min(32,len(scenarios["splits"]["train"]),len(scenarios["splits"]["extraction"]))
    expected_extraction={"fit_pool":"train","selection_pool":"extraction","scenarios_per_pool":count,
        "selection_pool_role":"reused_development_not_final_test","refresh_interval":interval,
        "profiles":list(runner.shared.EXTRACTION_PROFILES),"sampling":"same_frozen_actor_stochastic_actions_and_soft_probabilities"}
    _equal(plan["extraction"],expected_extraction,"Pair extraction source or schedule changed")
    _equal(plan["validation"],{"pool":"validation","role":"reused_development_model_selection","final_test_read":False,
        "explanation_test_read":False,"play_read":False},"Pair selection reads a forbidden pool")
    refreshes=math.ceil(cap/interval)+1
    upper=scenarios["configuration"]["horizon"]*(3*len(scenarios["splits"]["validation"])+2*count)
    _equal(plan["auxiliary_sampling"],{"separate_from_optimizer_joint_steps":True,"actual_count_scope":"completed_checkpointed_audits_only",
        "cap_per_branch":upper*(refreshes+2),"planned_refreshes":refreshes,"crash_recovery_refresh_allowance":2},"Auxiliary budget differs from frozen runner")
    fields=plan["preserved_checkpoint_fields"]
    if not NATIVE_FIELDS<=set(fields) or set(fields)-NATIVE_FIELDS-{"mps_rng","budget_at_checkpoint","acknowledgement","test_fixture"}:
        raise ValueError("Common checkpoint lacks dual-optimizer/RNG/environment source fields")
    for value in fields.values():_hash(value)
    for key in ("foundation_checkpoint_sha256","foundation_actor_sha256","curriculum_bank_sha256"):_hash(plan[key])
    if set(plan["frozen_files"])!=FROZEN_FILES:raise ValueError("Pair preparation files differ from actual runner")
    return [base,*[base+value for value in range(interval,cap+1,interval)],*([base+cap] if cap%interval else [])]


def verify_branch_log(rows,progress,authorization,budget,aux_budget,plan,protocol,branch,base_updates):
    release._finite([rows,progress,authorization,budget,aux_budget]);release._reject_fixture([rows,progress,authorization])
    base=plan["foundation_joint_steps"];cap=plan["additional_joint_steps_per_branch"];n=protocol["training"]["environments"]
    interval=runner.feedback_protocol()["feedback"]["extract_interval"];rollout=n*protocol["training"]["rollout_steps"]
    expected_auth={"version":runner.VERSION,"additional_budget_authorized":True,"per_branch_additional_budget":cap,
        "auxiliary_budget_cap":plan["auxiliary_sampling"]["cap_per_branch"],"branch":branch,"pair_id":plan["pair_id"],"test_fixture":False}
    for key,value in expected_auth.items():_equal(authorization[key],value,"Additional branch authorization mismatch: "+key)
    release._integer(authorization["pid"],1)
    expected_progress={"version":runner.VERSION,"status":"budget_complete","branch":branch,"test_fixture":False,
        "joint_steps":base+cap,"foundation_joint_steps":base,"additional_joint_steps":cap,
        "foundation_curriculum_generation_steps":plan["foundation_curriculum_generation_steps"],"additional_curriculum_generation_steps":0,
        "additional_budget":cap,"reserved_additional_upper_bound":cap,"discarded_after_crash_upper_bound":0,"formal_ready":False}
    for key,value in expected_progress.items():_equal(progress[key],value,"Incomplete or unequal paired endpoint: "+key)
    _equal(budget,{"cap":cap,"reserved_joint_steps":cap},"Additional sampling budget incomplete or changed")
    if aux_budget["cap"]!=plan["auxiliary_sampling"]["cap_per_branch"]:raise ValueError("Auxiliary sampling cap mismatch")
    release._integer(aux_budget["reserved_joint_steps"],0,aux_budget["cap"])
    previous=base;updates=base_updates;neural=0;active=[];by_step={}
    for row in rows:
        release._reject_fixture(row);step=row["joint_steps"];release._integer(step,base+1,base+cap)
        next_refresh=base+((previous-base)//interval+1)*interval
        expected_step=previous+min(rollout,next_refresh-previous,base+cap-previous)
        if step!=expected_step or step%n or row["optimizer_updates"]!=updates+1:raise ValueError("Paired log skips, duplicates or splits a scheduled PPO update")
        for key,value in {"foundation_joint_steps":base,"additional_joint_steps":step-base,"branch":branch,"test_fixture":False,
                "curriculum_generation_steps":plan["foundation_curriculum_generation_steps"],"reserved_additional_upper_bound":step-base,
                "discarded_after_crash_upper_bound":0}.items():_equal(row[key],value,"Paired update accounting mismatch: "+key)
        audit=row["execution_audit"]
        for key in ("neural_submitted","program_submitted","neural_overrides"):release._integer(audit[key])
        if audit["neural_overrides"] or not 0<audit["neural_submitted"]+audit["program_submitted"]<=2*(step-previous):raise ValueError("Paired on-policy action provenance failed")
        metrics=row["metrics"];lam=_number(row["feedback_lambda"],0,.01)
        for key in ("feedback_loss","feedback_gradient_norm","feedback_rows","feedback_kl","feedback_lambda"):_number(metrics[key],0 if key!="feedback_kl" else -1e-6)
        if not np.isclose(metrics["feedback_lambda"],lam,atol=1e-12):raise ValueError("Logged feedback coefficients disagree")
        if branch=="control" or lam==0:
            if lam or any(metrics[key]!=0 for key in ("feedback_loss","feedback_gradient_norm","feedback_rows","feedback_kl")):
                raise ValueError("Control/inactive update reports feedback use")
        else:
            minibatches=math.ceil(2*(step-previous)/protocol["training"]["minibatch"])
            expected_rows=audit["neural_submitted"]/minibatches
            if not np.isclose(metrics["feedback_rows"],expected_rows,atol=1e-6):
                raise ValueError("Feedback rows do not equal actual neural-only PPO minibatch rows")
            if not np.isclose(metrics["feedback_loss"],lam*metrics["feedback_kl"],rtol=2e-4,atol=1e-7):raise ValueError("Weighted actual KL and loss disagree")
            if metrics["feedback_rows"]>0 and metrics["feedback_gradient_norm"]>0 and metrics["feedback_kl"]>0 and metrics["feedback_loss"]>0:
                if audit["neural_submitted"]==0:raise ValueError("Feedback gradient without neural sampled rows")
                active.append(step)
        _number(row["collection_seconds"]);_number(row["update_seconds"])
        by_step[step]={"before":previous,"updates":updates+1,"row":row}
        previous=step;updates+=1;neural+=audit["neural_submitted"]
    if previous!=base+cap or not rows:raise ValueError("Paired log does not reach the true complete endpoint")
    return {"joint_steps":previous,"additional_joint_steps":cap,"optimizer_updates":updates,"neural_rows":neural,
        "nonzero_feedback_updates":active,"updates_by_step":by_step}


def _saved_dataset(path):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={"observations","probabilities","episode_ids","groups_json"}:raise ValueError("Unexpected actual extraction NPZ fields")
        result={key:f[key].copy() for key in ("observations","probabilities","episode_ids")}
        result["groups"]=release.parse_json(str(f["groups_json"].item()))
    x,p=result["observations"],result["probabilities"]
    if x.dtype!=np.float32 or p.dtype!=np.float32 or x.ndim!=2 or p.shape!=(len(x),5) or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("Invalid real extraction arrays")
    if result["episode_ids"].shape!=(len(x),) or result["episode_ids"].dtype.kind not in "US" or len(result["groups"])!=len(x):
        raise ValueError("Extraction row provenance missing")
    return result


def replay_extraction(traces,actor,scenes,*,pool,step,seed):
    """Replay the actual collector convention, including neural RNG and masks.

    This is inspection of recorded training-pool evidence, never PPO collection
    or a new performance evaluation. No program action enters an Actor row.
    """
    if len(traces)!=len(scenes):raise ValueError("Extraction trace does not cover exact registered pool prefix")
    observations=[];probabilities=[];ids=[];groups=[];transitions=0
    for index,(trace,scene) in enumerate(zip(traces,scenes)):
        profile=runner.shared.EXTRACTION_PROFILES[index%4];role=-1 if profile=="selfplay" else index%2
        for key,value in {"scenario_id":scene["id"],"fingerprint":scene["fingerprint"],"partner":profile,"program_role":role}.items():
            _equal(trace[key],value,"Recorded extraction source/profile mismatch")
        if not scene["id"].startswith(pool+"_"):raise ValueError("Extraction reads a forbidden split")
        env=NativeWarehouseEnv();reset_scenario(env,scene)
        actor_rng=np.random.default_rng(np.random.SeedSequence([seed,step,index,91]))
        partner_rng=np.random.default_rng(np.random.SeedSequence([seed,step,index,92]))
        if len(trace["decisions"])>env.config.horizon:raise ValueError("Extraction exceeds physical episode horizon")
        for decision in trace["decisions"]:
            if env.done:raise ValueError("Recorded extraction continues beyond termination")
            obs=env.observations();program=None if role<0 else partner_action(env,f"robot_{role+1}",profile,partner_rng)
            proposed,dist=actor.act(obs,False,actor_rng);submitted=dict(proposed)
            if role>=0:submitted[f"robot_{role+1}"]=program
            row_indices=[]
            for i,agent in enumerate(env.state.agents):
                if i==role or not agent.active:continue
                row_indices.append(len(observations));observations.append(obs[agent.agent_id]);probabilities.append(dist[agent.agent_id])
                ids.append(f"{pool}/{step}/{scene['id']}");groups.append(critical_groups(env,agent.agent_id))
            _,_,_,_,info=env.step(submitted);transitions+=1
            _equal(decision,{"frame":env.state.frame,"proposed":proposed,"submitted":submitted,"executed":info["executed_actions"],
                "neural_row_indices":row_indices},"Recorded extraction is not the frozen Actor's actual trajectory")
        if not env.done:raise ValueError("Extraction trace stops before the registered full episode")
    return {"observations":np.asarray(observations,dtype=np.float32).reshape(-1,actor.obs_dim),
        "probabilities":np.asarray(probabilities,dtype=np.float32).reshape(-1,5),"episode_ids":np.asarray(ids),
        "groups":groups,"transitions":transitions}


def verify_tree_evidence(pair,prefix,actor,scenarios,plan,protocol,fit):
    provenance=pair.json(prefix+"trajectory_provenance.json")
    for key,value in {"actor_sha256":actor.sha256,"selection_pool":"extraction_reused_development","final_test_read":False,
            "teacher_rows_in_fit":0,"selection_trace_row_indices_refer_to_unfiltered_pool":True}.items():
        _equal(provenance[key],value,"Tree trajectory provenance mismatch: "+key)
    n=plan["extraction"]["scenarios_per_pool"];step=actor.metadata["joint_steps"]
    train=replay_extraction(provenance["train"],actor,scenarios["splits"]["train"][:n],pool="train",step=step,seed=protocol["seed"])
    selection=replay_extraction(provenance["selection"],actor,scenarios["splits"]["extraction"][:n],pool="extraction",step=step,seed=protocol["seed"]+100)
    seen={row.tobytes() for row in train["observations"]};keep=np.asarray([row.tobytes() not in seen for row in selection["observations"]],dtype=bool)
    if provenance["selection_overlap_rows_removed"]!=int((~keep).sum()):raise ValueError("Extraction overlap count is false")
    filtered={key:selection[key][keep] for key in ("observations","probabilities","episode_ids")}
    filtered["groups"]=[value for value,present in zip(selection["groups"],keep) if present]
    for name,expected in (("train_rows.npz",train),("selection_rows.npz",filtered)):
        saved=_saved_dataset(pair.path(prefix+name))
        for key in ("observations","probabilities","episode_ids"):
            if not np.array_equal(saved[key],expected[key]):raise ValueError("Saved tree-fit data differ from actual neural trajectory: "+key)
        _equal(saved["groups"],expected["groups"],"Saved critical tree labels differ from physical state")
    manager=FeedbackManager(actor.metadata["feature_names"],runner.manager_config(protocol,runner.feedback_protocol()))
    if "selected" not in fit:
        if fit.get("reliable") is not False or fit.get("actor_sha256")!=actor.sha256 or not isinstance(fit.get("reason"),str):
            raise ValueError("Unrecognized failed extraction record")
        return {"reliable":False,"program":None,"transitions":train["transitions"]+selection["transitions"]}
    x=train["observations"];vx=filtered["observations"]
    y=train["probabilities"]/train["probabilities"].sum(-1,keepdims=True)
    vy=filtered["probabilities"]/filtered["probabilities"].sum(-1,keepdims=True)
    for key,value in {"version":runner.shared.FEEDBACK_VERSION if hasattr(runner.shared,"FEEDBACK_VERSION") else "warehouse_native_rcpd_feedback_v1",
        "step":step,"source_actor_sha256":actor.sha256,"source":"same_frozen_actor_soft_probabilities_on_neural_trajectory_observations",
        "train_rows":len(x),"validation_rows":len(vx),"train_episodes":len(set(train["episode_ids"])),
        "validation_episodes":len(set(filtered["episode_ids"])),"episode_overlap":0,"exact_observation_overlap":0,
        "complexity_has_actor_gradient":False,"explanation_qualified":False,"intervention_direction_not_tested_here":True,
        "training_observation_sha256":sha256(x.tobytes()).hexdigest(),"validation_observation_sha256":sha256(vx.tobytes()).hexdigest(),
        "training_probabilities_sha256":sha256(y.tobytes()).hexdigest(),"validation_probabilities_sha256":sha256(vy.tobytes()).hexdigest()}.items():
        _equal(fit[key],value,"Tree fit report differs from raw trajectory evidence: "+key)
    if len(x)<manager.config.minimum_training_rows or len(vx)<manager.config.minimum_validation_rows or min(len(set(train["episode_ids"])),len(set(filtered["episode_ids"])))<2:
        raise ValueError("Successful extraction lacks minimum independent rows")
    program=ExecutableProgram.load_json(pair.path(prefix+"program.json"));release._reject_fixture(program.to_dict())
    if program.metadata["native_source_actor_sha256"]!=actor.sha256 or tuple(program.action_names)!=ACTIONS or tuple(program.feature_names)!=tuple(actor.metadata["feature_names"]):
        raise ValueError("Program is bound to a different Actor or feature/action contract")
    prediction=manager._predict(program,vx);agree=prediction.argmax(-1)==vy.argmax(-1)
    mean_kl=float(np.mean(np.sum(vy*(np.log(vy.clip(1e-8))-np.log(prediction.clip(1e-8))),axis=-1)))
    selected=fit["selected"];critical={}
    _equal(program.metadata["native_feedback_config"],asdict(manager.config),"Serialized tree uses a different feedback configuration")
    if program.metadata["native_feedback_version"]!="warehouse_native_rcpd_feedback_v1":raise ValueError("Tree feedback version changed")
    complexity=program_complexity(program,max_depth=max(manager.config.depths),max_leaf_count=max(manager.config.leaves),max_predicate_count=max(manager.config.leaves)-1)
    _equal(selected["complexity"],complexity.to_dict(),"Chosen tree complexity differs from actual serialized tree")
    for group in manager.config.critical_groups:
        mask=np.asarray([group in value for value in filtered["groups"]])
        critical[group]={"rows":int(mask.sum()),"episodes":len({str(e) for e,m in zip(filtered["episode_ids"],mask) if m}),
            "fidelity":float(agree[mask].mean()) if mask.any() else None}
    if not np.isclose(selected["fidelity"],float(agree.mean()),atol=1e-7) or not np.isclose(selected["mean_kl"],mean_kl,atol=1e-7):
        raise ValueError("Tree reliability summary differs from same-Actor soft probabilities")
    _equal(selected["critical"],critical,"Tree critical reliability differs from actual physical groups")
    reliable=bool(agree.mean()>=manager.config.minimum_fidelity and mean_kl<=manager.config.maximum_mean_kl and
        all(value["rows"]>=manager.config.minimum_critical_rows and value["fidelity"] is not None and value["fidelity"]>=manager.config.minimum_critical_fidelity for value in critical.values()))
    if fit["reliable"] is not reliable or selected["reliable"] is not reliable:raise ValueError("Tree fit claimed a false reliability gate")
    for key,value in {**selected,"feedback_eligible":reliable,"explanation_eligible":False}.items():
        _equal(program.metadata["metrics"][key],value,"Serialized tree reliability differs from its fit report")
    candidates=fit["candidates"]
    if [(c["depth_cap"],c["leaf_cap"]) for c in candidates]!=[(d,l) for d in manager.config.depths for l in manager.config.leaves]:raise ValueError("Tree candidate search differs from frozen feedback protocol")
    eligible=[c for c in candidates if c["reliable"]]
    chosen=min(eligible,key=lambda c:(c["complexity"]["loss"],c["mean_kl"],-c["fidelity"],c["depth_cap"],c["leaf_cap"])) if eligible else min(candidates,key=lambda c:c["selection_objective"])
    _equal(chosen,selected,"Recorded compact tree selection disagrees with candidate table")
    return {"reliable":reliable,"program":program,"transitions":train["transitions"]+selection["transitions"]}


def verify_logged_lambdas(manager,step,until,rows,score,reference,warmed):
    for end,item in rows.items():
        if not step<end<=until:continue
        # The lambda was calculated at the update START. Its ending checkpoint's
        # newly extracted tree does not exist yet and cannot explain this loss.
        expected=0. if manager is None else manager.update(item["before"],score,
            capability_eligible=warmed,reference_score=reference)["lambda"]
        if not np.isclose(item["row"]["feedback_lambda"],expected,atol=1e-12):raise ValueError("Feedback update uses wrong refresh/step/ramp")


def verify_pair_endpoint(record,root,protocol,scenarios):
    pair=BoundPair(record,root);plan=pair.plan;due=verify_plan(plan,protocol,scenarios)
    for name,expected in plan["frozen_files"].items():
        if file_hash(pair.path(name))!=expected:raise ValueError("Prepared pair evidence changed: "+name)
    if file_hash(pair.path("foundation_checkpoint.pt"))!=plan["foundation_checkpoint_sha256"]:raise ValueError("Common checkpoint SHA differs from plan")
    # Byte binding only: no Torch checkpoint archive/pickle is inspected.
    _equal(pair.json("protocol.json"),protocol,"Pair protocol differs from release")
    _equal(pair.json("feedback_protocol.json"),runner.feedback_protocol(),"Pair feedback specification changed")
    _equal(pair.json("scenarios.json"),scenarios,"Pair scenarios differ from release")
    bank=pair.json("curriculum_bank.json");validate_curriculum_bank(bank,scenarios)
    if digest(bank)!=plan["curriculum_bank_sha256"] or bank["generation_steps"]!=plan["foundation_curriculum_generation_steps"]:raise ValueError("Pair curriculum identity/count differs")
    foundation=NumPyNativeActor(pair.path("foundation_actor.npz"));release._reject_fixture(foundation.metadata)
    for key,value in {"experiment_version":protocol["version"],"joint_steps":plan["foundation_joint_steps"],
        "feedback_enabled":False,"protocol_sha256":digest(protocol),"scenario_manifest_sha256":digest(scenarios),
        "curriculum_bank_sha256":digest(bank),"source_sha256":digest(v2_source_hashes()),"curriculum_generation_steps":bank["generation_steps"]}.items():
        _equal(foundation.metadata[key],value,"Pair foundation Actor metadata mismatch: "+key)
    if foundation.sha256!=plan["foundation_actor_sha256"]:raise ValueError("Pair foundation Actor identity mismatch")
    base_updates=foundation.metadata["optimizer_updates"];release._integer(base_updates,1)
    execution=pair.json("execution.json")
    if set(execution)!={"device","torch_version","numpy_version"} or any(not isinstance(v,str) or not v for v in execution.values()):raise ValueError("Missing common paired execution environment")
    reports={}
    for name in ("foundation_validation.json","validation_reference.json","validation_random.json"):
        report=pair.json(name)
        if report["deterministic_actor"] is not True or report["metric"]!="mean_team_deliveries":raise ValueError("Unknown actual runner evaluation convention")
        reports[name]=release.verified_report(report,scenarios["splits"]["validation"],scenarios["configuration"]["horizon"],expected_count=50)
    source_report=pair.json("foundation_validation.json")
    if source_report["actor_sha256"]!=foundation.sha256 or source_report["joint_steps"]!=plan["foundation_joint_steps"]:raise ValueError("Foundation validation identity mismatch")
    if len({file_hash(pair.path(name)) for name in reports})!=3:raise ValueError("Foundation/baseline reports are reused")
    reference=reports["validation_reference.json"];random_report=reports["validation_random.json"]
    gate=capability(reports["foundation_validation.json"],reference,random_report,protocol)
    warmup=runner.shared.warmup_capability(reports["foundation_validation.json"],reference,protocol,plan["foundation_joint_steps"])
    if not gate["eligible"] or not warmup["eligible"]:raise ValueError("Actual recorded foundation capability/warmup gate failed")
    _equal(plan["capability"],gate,"Pair preparation capability differs from raw rows")
    _equal(plan["warmup_capability"],warmup,"Pair preparation warmup differs from raw rows")
    results={};treatment=runner.feedback_protocol();base=plan["foundation_joint_steps"];horizon=scenarios["configuration"]["horizon"]
    for branch in runner.BRANCHES:
        folder=branch+"/";progress=pair.json(folder+"progress.json")
        pair.path(folder+"initial_checkpoint.pt");pair.path(folder+"latest_checkpoint.pt")
        log=verify_branch_log(pair.rows(folder+"training.jsonl"),progress,pair.json(folder+"authorization.json"),
            pair.json(folder+"sampling_budget.json"),pair.json(folder+"auxiliary_budget.json"),plan,protocol,branch,base_updates)
        manager=FeedbackManager(foundation.metadata["feature_names"],runner.manager_config(protocol,treatment)) if branch=="feedback" else None
        candidates=[];total_aux=0;reserved_previous=0;latest_gate=None;rows=log["updates_by_step"]
        for index,step in enumerate(due):
            prefix=folder+f"development/step_{step:07d}/"
            actor=NumPyNativeActor(pair.path(prefix+"actor.npz"));verify_actor_metadata(actor,protocol,scenarios,plan,step,branch)
            expected_updates=base_updates if step==base else rows[step]["updates"]
            if actor.metadata["optimizer_updates"]!=expected_updates:raise ValueError("Exported Actor update count differs from committed log")
            if step==base and any(not np.array_equal(actor.weights[key],foundation.weights[key]) for key in actor.weights):raise ValueError("Initial branch export differs from common foundation weights")
            if step!=base:pair.path(folder+f"checkpoints/checkpoint_{step:07d}.pt")
            report=pair.json(prefix+"validation.json");event=pair.json(prefix+"refresh.json")
            if report["actor_sha256"]!=actor.sha256 or report["joint_steps"]!=step or report["dataset_role"]!="repeated_development_validation_not_final_test" or report["deterministic_actor"] is not True or report["metric"]!="mean_team_deliveries":
                raise ValueError("Actual development report source mismatch")
            evaluated=release.verified_report(report,scenarios["splits"]["validation"],horizon,expected_count=50)
            got=capability(evaluated,reference,random_report,protocol);warmed=runner.shared.warmup_capability(evaluated,reference,protocol,step)
            _equal(report["capability"],got,"Development capability differs from raw rows")
            _equal(report["warmup_capability"],warmed,"Development warmup differs from raw rows")
            score=float(np.mean([v["mean_team_deliveries"] for v in evaluated["summary"].values()]))
            ref=float(np.mean([v["mean_team_deliveries"] for v in reference["summary"].values()]))
            shutdowns=float(np.mean([v["mean_shutdowns"] for v in evaluated["summary"].values()]))
            actual=sum(row["steps"] for row in report["rows"])
            if manager is not None:
                fit=pair.json(prefix+"fit.json");_equal(event["tree_fit"],fit,"Refresh uses a different fit report")
                tree=verify_tree_evidence(pair,prefix,actor,scenarios,plan,protocol,fit);actual+=tree["transitions"]
                # Match fit's effect on the manager without refitting a tree.
                manager.current_lambda=0.;manager.reliable=tree["reliable"]
                if tree["program"] is not None:
                    manager.program=tree["program"];manager.last_fit_step=step;manager.last_step=step
                expected_gate=manager.update(step,score,capability_eligible=warmed["eligible"],reference_score=ref)
            else:
                if event["tree_fit"] is not None:raise ValueError("Pure RL control contains tree feedback")
                expected_gate={"active":False,"lambda":0.,"reason":"pure_rl_control"}
            _equal(event["gate"],expected_gate,"Recorded feedback gate/ramp disagrees with preceding evidence")
            for key,value in {"step":step,"actor_sha256":actor.sha256,"validation_score":score,"reference_score":ref,
                "warmup_eligible":warmed["eligible"],"actual_auxiliary_joint_steps":actual,"training_dataset_used":"train_only",
                "selection_data_reused_as_development":True}.items():_equal(event[key],value,"Refresh accounting mismatch: "+key)
            _number(event["seconds"])
            upper=horizon*(150+(2*plan["extraction"]["scenarios_per_pool"] if branch=="feedback" else 0))
            reserved=event["reserved_auxiliary_upper_bound"];release._integer(reserved,0,plan["auxiliary_sampling"]["cap_per_branch"])
            if reserved<reserved_previous+upper or (reserved-reserved_previous)%upper:raise ValueError("Auxiliary reservations do not cover actual refreshes")
            reserved_previous=reserved;total_aux+=actual
            latest_gate={"validation_score":score,"reference_score":ref,"capability_eligible":warmed["eligible"]}
            candidate={"score":score,"shutdowns":shutdowns,"joint_steps":step,"actor":f"development/step_{step:07d}/actor.npz",
                "actor_sha256":actor.sha256,"capability":got}
            candidates.append({**candidate,"validation_sha256":file_hash(pair.path(prefix+"validation.json"))})
            until=due[index+1] if index+1<len(due) else step
            verify_logged_lambdas(manager,step,until,rows,score,ref,warmed["eligible"])
        best=max(candidates,key=lambda c:(c["score"],-c["shutdowns"],-c["joint_steps"]))
        _equal(progress["best_continuation"],{k:v for k,v in best.items() if k!="validation_sha256"},"Endpoint selected a different development Actor")
        _equal(progress["latest_gate"],latest_gate,"Endpoint gate does not match last actual refresh")
        if progress["auxiliary_actual_joint_steps"]!=total_aux:raise ValueError("Endpoint auxiliary actual count differs from raw recorded audits")
        aux=pair.json(folder+"auxiliary_budget.json")
        if aux["reserved_joint_steps"]!=reserved_previous:raise ValueError("Unaccounted auxiliary reservation after final audit")
        max_lost=2*horizon*(150+(2*plan["extraction"]["scenarios_per_pool"] if branch=="feedback" else 0))
        usual=sum(horizon*(150+(2*plan["extraction"]["scenarios_per_pool"] if branch=="feedback" else 0)) for _ in due)
        if reserved_previous-usual>max_lost:raise ValueError("Auxiliary crash allowance exceeded")
        results[branch]={"best":best,"registered_candidates":len(candidates),"candidates":candidates,
            "additional_joint_steps":log["additional_joint_steps"],"optimizer_updates":log["optimizer_updates"],"neural_rows":log["neural_rows"],
            "nonzero_feedback_updates":log["nonzero_feedback_updates"],"auxiliary_actual_joint_steps":total_aux,
            "auxiliary_reserved_upper_bound":reserved_previous}
    if results["control"]["optimizer_updates"]!=results["feedback"]["optimizer_updates"]:raise ValueError("Paired actual PPO update counts differ")
    # The actual runner writes these exact families, not user-selected subsets.
    known=set(pair.bound);allowed=set(pair.used)
    if known-allowed:raise ValueError("Unexpected indexed pair evidence; use only the actual selected release chain")
    for name in pair.used:
        release.bound_path(pair.root,pair.files[name])
    release.bound_path(pair.root,record["pair"])
    return {"kind":VERSION,"pair_id":plan["pair_id"],"pair_sha256":file_hash(pair.plan_path),"branches":results,
        "baseline_sha256":{"reference":file_hash(pair.path("validation_reference.json")),"random":file_hash(pair.path("validation_random.json"))},
        "total_ppo_joint_steps":plan["target_absolute_joint_steps"],"curriculum_generation_steps":bank["generation_steps"],
        "foundation_joint_steps":base,"additional_joint_steps_per_branch":plan["additional_joint_steps_per_branch"],
        "common_initial_state_scope":SAME_INITIAL_SCOPE,"checkpoint_tensor_equality_recomputed":False,
        "feedback_gradient_scope":"actual_committed_batch_metrics_and_bound_runner; no_saved_PPO_minibatches_for_independent_gradient_recomputation",
        "interaction_budget_scope":"equal_actual_PPO_steps_and_updates; auxiliary_extraction_interactions_disclosed_separately_not_equal_total_interactions",
        "pair_artifacts":{name:{"path":str(pair.bound[name]),"sha256":file_hash(pair.bound[name])} for name in sorted(pair.used)}}


def verify_pair_selection(selection,protocol,scenarios,actor,validation_sha256,endpoint):
    release._reject_fixture(selection)
    if endpoint is None or endpoint.get("kind")!=VERSION:raise ValueError("Missing verified complete paired endpoint")
    branch=selection["branch"]
    if branch not in runner.BRANCHES:raise ValueError("Unknown selected pair branch")
    for key,value in {"version":SELECTION_VERSION,"pair_id":endpoint["pair_id"],"pair_sha256":endpoint["pair_sha256"],
        "split":"validation","rule":protocol["checkpoint_selection"],"selection_inputs":["validation"],"frozen_before_final_test":True,
        "protocol_sha256":digest(protocol),"scenario_manifest_sha256":digest(scenarios)}.items():
        _equal(selection[key],value,"Paired recorded selection mismatch: "+key)
    verify_actor_metadata(actor,protocol,scenarios,step=actor.metadata["joint_steps"],branch=branch)
    best=endpoint["branches"][branch]["best"]
    if actor.metadata["pair_id"]!=endpoint["pair_id"] or best["actor_sha256"]!=actor.sha256 or best["joint_steps"]!=actor.metadata["joint_steps"] or best["validation_sha256"]!=validation_sha256:
        raise ValueError("Selected paired Actor is not the actual branch validation winner")
    if selection["selected_actor_sha256"]!=actor.sha256 or selection["selected_joint_steps"]!=best["joint_steps"]:raise ValueError("Selected paired identity changed")
    effective=[step for step in endpoint["branches"][branch]["nonzero_feedback_updates"] if step<=best["joint_steps"]]
    if branch=="feedback" and not effective:raise ValueError("Selected Actor has no preceding confirmed nonzero KL/gradient update; later feedback cannot qualify it")
    return {"branch":branch,"selected_step":best["joint_steps"],"registered_candidates":endpoint["branches"][branch]["registered_candidates"],
        "effective_feedback_updates_before_selected_actor":len(effective),"scope":"recorded_validation_selection_only; no_unrecorded_human_selection_claim"}
