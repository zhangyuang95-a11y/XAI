"""Fail-closed local A/B release inspection; never creates or qualifies fixtures.

All artifacts are relative paths plus SHA-256, contained in the manifest
directory. Raw explanation rows use JSON ``{version, actor_sha256,
program_sha256, scenario_manifest_sha256, split, rows, trajectories, fit, selection}``.
Rows carry episode_id/frame/scenario_id/agent_id/snapshot/observation/groups.
Each trajectory binds its initial fingerprint and actual player-action prefix;
the same Actor replays it. Stochastic extraction also saves actor_rng_state.
The fit block declares train and/or extraction subsets and uses the same raw
trajectory contract; exact fit/holdout observation overlap is recomputed.
Parity rows use an allow_pickle=False NPZ with observations, scenario_ids and
snapshots (JSON strings), from calibration only. No checkpoint pickle is read:
PyTorch inference is reconstructed from the exact exported neural tensors.

Training endpoints bind the complete JSONL update log, budget, run and progress
artifacts; selection cannot shorten the actual registered candidate history.
Completed v1/v2 foundations and the frozen v2 paired continuation have separate
endpoint contracts. Arbitrary early-stop releases are not supported. This
certifies recorded technical evidence for local_pilot. It does
not prove unrecorded human selection never occurred or explanation efficacy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash, load_protocol, source_hashes
from backend.training.warehouse_native_evaluation import capability, critical_groups, summarize
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.explanation import NativeExplainer, REQUIRED_CATEGORIES
from env.warehouse_native.policy import ACTIONS
from env.warehouse_native.runtime import NativeRuntime
from env.warehouse_native.scenarios import SPLIT_NAMES, scenario_fingerprint
from ui.warehouse_native_bank import NativeQuestionBank

VERSION="warehouse-native-local-ab-release.v1"
SELECTION_VERSION="warehouse-native-validation-selection.v1"
RAW_EXPLANATION_VERSION="warehouse-native-explanation-rows.v1"
PARITY_VERSION="warehouse-native-release-numpy-parity.v1"
ARTIFACT_NAMES=("actor","scenarios","protocol","validation","reference","random","final_test",
    "final_reference","final_random","parity","explanation","program","bank","task_calibration",
    "analysis_protocol","selection","explanation_rows","parity_rows","task_calibration_rows","training_end")
PARTNERS=("skilled","assertive","noisy")
GENERALIZATION={"initial_states_disjoint":True,"shared_topology":True,
    "trajectory_overlap_excluded":False,"human_explanation_effect_validated":False}


def analysis_protocol():
    return {"version":"warehouse-native-local-ab-analysis.v1",
        "primary":{"metric":"task2_mean_deliveries","aggregation":"arithmetic_mean_of_three_unique_ended_rounds",
            "missing_rounds":"do_not_impute","explicit_early_end":"retain_record"},
        "rounds_per_task":{"task1":3,"task2":3},"task_order":["XY","YX"],
        "condition_access":{group:{"task1_explanation":group=="A","task1_review_explanation":group=="A",
            "task2_explanation":False,"task2_old_answers":False} for group in ("A","B")},
        "same_actor_ab":True,"stage_difference":"descriptive_only","detour_metrics":False,
        "namespace":"local_pilot","formal_ready":False,"questionnaire":{"next_action":4,"wait_three":4}}


def release_source_hashes():
    paths={Path(__file__),ROOT/"ui/warehouse_native_pair_release.py",ROOT/"ui/warehouse_native_answer_verification.py",
        ROOT/"backend/training/warehouse_native_answer_audit.py",ROOT/"ui/warehouse_native_server.py",ROOT/"ui/warehouse_native_bank.py",
        ROOT/"ui/warehouse_native_export.py",ROOT/"ui/warehouse_view.py",
        ROOT/"backend/training/warehouse_native_task_calibration.py",
        ROOT/"backend/training/warehouse_native_v2_feedback_run.py",
        ROOT/"backend/training/warehouse_native_feedback_run.py"}
    paths.update((ROOT/"env/warehouse_native").glob("*.py"))
    paths.update((ROOT/"env/warehouse").glob("*.py"))
    paths.update((ROOT/"core").glob("*.py"))
    paths.update(p for p in (ROOT/"ui/warehouse_native").iterdir() if p.is_file())
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}


def _finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError("Non-finite JSON value")
    if isinstance(value,dict):
        for item in value.values():_finite(item)
    elif isinstance(value,list):
        for item in value:_finite(item)


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError("Duplicate JSON key")
        result[key]=value
    return result


def parse_json(text):
    def invalid(value):raise ValueError("Non-finite JSON constant: "+value)
    value=json.loads(text,parse_constant=invalid,object_pairs_hook=_pairs);_finite(value)
    return value


def _json(path):return parse_json(Path(path).read_text())


def _reject_fixture(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ("test_fixture","fixture","synthetic_fixture") and item:
                raise ValueError("Fixture evidence cannot open a real local study")
            _reject_fixture(item)
    elif isinstance(value,list):
        for item in value:_reject_fixture(item)


def bound_path(root,contract):
    if not isinstance(contract,dict) or set(contract)!={"path","sha256"}:raise ValueError("Artifact requires only path and SHA-256")
    relative=Path(contract["path"])
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:raise ValueError("Artifact path must stay inside release directory")
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():raise ValueError("Artifact path escapes or is missing")
    expected=contract["sha256"]
    if not isinstance(expected,str) or len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):raise ValueError("Invalid artifact SHA-256")
    if file_hash(path)!=expected:raise ValueError("Artifact SHA-256 mismatch")
    return path


def _integer(value,minimum=0,maximum=None):
    if type(value) is not int or value<minimum or (maximum is not None and value>maximum):raise ValueError("Invalid integer metric")


def verify_scenarios(scenarios):
    if set(scenarios["splits"])!=set(SPLIT_NAMES):raise ValueError("Unexpected scenario splits")
    ids=set();fingerprints=set()
    for split,scenes in scenarios["splits"].items():
        for scene in scenes:
            if scene["id"] in ids or not scene["id"].startswith(split+"_"):raise ValueError("Duplicate or incorrectly named scenario")
            env=NativeWarehouseEnv();env.restore(scene["snapshot"])
            if asdict(env.config)!=scenarios["configuration"]:raise ValueError("Scenario top-level configuration differs from physical snapshots")
            if env.state.frame!=0 or scenario_fingerprint(env)!=scene["fingerprint"]:raise ValueError("Scenario is not its bound initial physical state")
            if scene["fingerprint"] in fingerprints:raise ValueError("Initial physical states overlap")
            ids.add(scene["id"]);fingerprints.add(scene["fingerprint"])
    expected={"validation":50,"final_test":100,"explanation_test":100}
    if any(len(scenarios["splits"][key])!=count for key,count in expected.items()):raise ValueError("Frozen validation/final/explanation pools have wrong sizes")
    if len(scenarios["splits"]["play"])<7 or len(scenarios["splits"]["train"])<2:raise ValueError("Missing fixed study or training scenes")
    if "split" in scenarios and scenarios["split"]!=scenarios["splits"]:raise ValueError("Conflicting scenario alias")
    return {"scenes":len(ids),"initial_fingerprints":len(fingerprints),"limits":dict(GENERALIZATION)}


def verified_report(report,scenes,horizon,*,expected_count):
    _finite(report);_reject_fixture(report)
    expected={scene["id"]:scene["fingerprint"] for scene in scenes}
    if len(scenes)!=expected_count or len(expected)!=expected_count:raise ValueError("Wrong exact evaluation pool")
    rows=report["rows"]
    if len(rows)!=len(PARTNERS)*expected_count:raise ValueError("Unexpected evaluation row count")
    if any(row.get("partner") not in PARTNERS for row in rows):raise ValueError("Unexpected evaluation partner")
    for row in rows:
        if row["scenario_id"] not in expected or row["initial_fingerprint"]!=expected[row["scenario_id"]]:raise ValueError("Evaluation initial physical fingerprint mismatch")
        for key in ("team_deliveries","ai_deliveries","collisions","shutdowns","invalid_moves","max_repeated_configuration","nn_action_overrides"):_integer(row[key])
        _integer(row["steps"],1,horizon)
        if type(row["neural_role"]) is not int or row["neural_role"]!=1 or type(row["ai_active_end"]) is not bool:raise ValueError("Wrong role or nonboolean survival")
        individual=row["individual_deliveries"]
        if not isinstance(individual,list) or len(individual)!=2:raise ValueError("Both individual contributions required")
        for value in individual:_integer(value)
        if sum(individual)!=row["team_deliveries"] or individual[1]!=row["ai_deliveries"]:raise ValueError("Individual and team contribution mismatch")
        if row["nn_action_overrides"]!=0:raise ValueError("Neural actions were overridden")
        if type(row["native_score"]) not in (int,float):raise ValueError("Invalid score")
        if not isinstance(row["charging_steps"],list) or len(row["charging_steps"])!=2:raise ValueError("Charging metrics missing")
        for count in row["charging_steps"]:_integer(count,0,row["steps"])
    recomputed={}
    for partner in PARTNERS:
        local=[row for row in rows if row["partner"]==partner]
        if len(local)!=expected_count or {row["scenario_id"] for row in local}!=set(expected):raise ValueError("Missing or repeated evaluation scenario")
        summary=summarize(local);claimed=report["summary"][partner]
        if any(key not in claimed or canonical(value)!=canonical(claimed[key]) for key,value in summary.items()):raise ValueError("Evaluation summary differs from raw rows")
        recomputed[partner]=summary
    return {"rows":rows,"summary":recomputed}


def _training_sources(protocol):
    if protocol==load_protocol():return source_hashes()
    from backend.training.warehouse_native_v2 import load_v2_protocol,v2_source_hashes
    if protocol==load_v2_protocol():return v2_source_hashes()
    raise ValueError("Training protocol is not an unchanged frozen native foundation protocol")


def evaluation_contract(kind):
    if kind not in ("neural","reference","random"):raise ValueError("Unknown evaluation controller")
    return {"actor_kind":kind,"neural_role":1,"partners":list(PARTNERS),"seed_base":17000,
        "deterministic_actor":True,"reference_policy":"skilled","random_policy":"uniform_five_actions",
        "sources":{p:file_hash(ROOT/p) for p in ("backend/training/warehouse_native_evaluation.py","env/warehouse_native/partners.py")}}


def verify_training_end(record,root,protocol,scenarios):
    if record["version"]!="warehouse-native-training-end.v1":raise ValueError("Unsupported training-end contract")
    documents={name:_json(bound_path(root,record[name])) for name in ("run","budget","progress")}
    for value in documents.values():_reject_fixture(value)
    log_path=bound_path(root,record["training_log"])
    rows=[parse_json(line) for line in log_path.read_text().splitlines() if line.strip()]
    run,budget,progress=(documents[name] for name in ("run","budget","progress"))
    if run["sources"]!=_training_sources(protocol) or run["protocol_sha256"]!=digest(protocol) or run["feedback_enabled"] is not False:
        raise ValueError("Training-end source or protocol binding mismatch")
    if run.get("scenario_sha256",run.get("scene_sha256"))!=digest(scenarios):raise ValueError("Training-end scenario binding mismatch")
    previous=0;updates=0;n=protocol["training"]["environments"]
    for row in rows:
        _reject_fixture(row);step=row["joint_steps"];_integer(step,1)
        if step<=previous or step%n or row["optimizer_updates"]!=updates+1:raise ValueError("Training log is not a monotonic complete update sequence")
        audit=row["execution_audit"]
        for name in ("neural_submitted","program_submitted","neural_overrides"):_integer(audit[name])
        if audit["neural_overrides"] or not 0<audit["neural_submitted"]+audit["program_submitted"]<=2*(step-previous):raise ValueError("Training sampling provenance or action accounting mismatch")
        previous=step;updates+=1
    if not rows or progress["joint_steps"]!=previous:raise ValueError("Training progress does not equal the complete raw log endpoint")
    if protocol["version"]=="warehouse-native-foundation.v2":
        if progress["status"]!="v2_budget_complete_candidate":raise ValueError("V2 foundation is not at a recorded completed budget endpoint")
        cap=run["approved_total_joint_steps"];generated=progress["curriculum_generation_steps"]
        _integer(generated)
        if budget["cap"]!=cap or set(budget["reserved"])!={"ppo","curriculum"}:raise ValueError("V2 budget ledger mismatch")
        if previous>budget["reserved"]["ppo"] or generated>budget["reserved"]["curriculum"] or sum(budget["reserved"].values())>cap or cap-sum(budget["reserved"].values())>=n:
            raise ValueError("V2 completed/reserved budget accounting mismatch")
        bank=_json(bound_path(root,record["curriculum_bank"]))
        if bank["generation_steps"]!=generated or bank["scenario_manifest_sha256"]!=digest(scenarios):raise ValueError("Curriculum generation source accounting mismatch")
        if cap>protocol["authorized_run"]["total_training_environment_steps_max"]:raise ValueError("V2 budget exceeds frozen maximum")
    else:
        cap=run["budget"];generated=0
        if progress["status"]!="foundation_budget_complete" or budget["cap"]!=cap or not previous<=budget["reserved_joint_steps"]<=cap or cap-budget["reserved_joint_steps"]>=n:
            raise ValueError("Foundation completion or sampling budget mismatch")
        if cap>protocol["authorized_run"]["joint_steps_max"]:raise ValueError("Foundation exceeds frozen maximum")
    return {"total_ppo_joint_steps":previous,"curriculum_generation_steps":generated,"optimizer_updates":updates,
        "scope":"completed_v1_or_v2_foundations_only; paired and early-stop releases require their own source contract"}


def verify_selection(selection,root,protocol,scenarios,actor,validation_sha256,training_end=None):
    _reject_fixture(selection)
    if (selection["version"]!=SELECTION_VERSION or selection["split"]!="validation"
            or selection["rule"]!=protocol["checkpoint_selection"] or selection["frozen_before_final_test"] is not True
            or selection["selection_inputs"]!=["validation"] or selection["protocol_sha256"]!=digest(protocol)
            or selection["scenario_manifest_sha256"]!=digest(scenarios)):
        raise ValueError("Invalid frozen validation-only selection contract")
    total=selection["total_ppo_joint_steps"];generated=selection["curriculum_generation_steps"]
    if training_end is None or total!=training_end["total_ppo_joint_steps"] or generated!=training_end["curriculum_generation_steps"]:
        raise ValueError("Selection endpoint is not bound to actual complete training and budget records")
    _integer(total,1);_integer(generated)
    interval=protocol["training"]["checkpoint_interval"]
    if protocol["version"]=="warehouse-native-foundation.v2":
        if total+generated>protocol["authorized_run"]["total_training_environment_steps_max"]:raise ValueError("Foundation exceeds frozen combined budget")
    elif generated or total>protocol["authorized_run"]["joint_steps_max"]:raise ValueError("Foundation exceeds frozen PPO budget")
    due=sorted(set([*range(interval,total+1,interval),total]))
    candidates=selection["candidates"]
    if [item["joint_steps"] for item in candidates]!=due:raise ValueError("Selection omits a registered checkpoint or includes a probe")
    rankings=[]
    for item in candidates:
        path=bound_path(root,item["validation"]);report=_json(path)
        if report["evaluation"]!=evaluation_contract("neural"):raise ValueError("Selection evaluation controller metadata mismatch")
        if report.get("joint_steps")!=item["joint_steps"] or report.get("actor_sha256")!=item["actor_sha256"]:raise ValueError("Selection candidate binding mismatch")
        result=verified_report(report,scenarios["splits"]["validation"],scenarios["configuration"]["horizon"],expected_count=50)
        score=float(np.mean([r["mean_team_deliveries"] for r in result["summary"].values()]))
        shutdowns=float(np.mean([r["mean_shutdowns"] for r in result["summary"].values()]))
        rankings.append(((score,-shutdowns,-item["joint_steps"]),item))
    selected=max(rankings,key=lambda item:item[0])[1]
    if (selected["actor_sha256"]!=actor.sha256 or selected["joint_steps"]!=actor.metadata.get("joint_steps")
            or selected["validation"]["sha256"]!=validation_sha256
            or selection["selected_actor_sha256"]!=actor.sha256 or selection["selected_joint_steps"]!=selected["joint_steps"]):
        raise ValueError("Actor is not the recorded validation-selected candidate")
    return {"selected_step":selected["joint_steps"],"registered_candidates":len(candidates),
        "scope":"recomputed_recorded_selection_only_not_proof_of_unrecorded_choices"}


def _trajectory_rows(block,runtime,scenarios,allowed_splits):
    """Verify each state against its actual frozen source and NN trajectory."""
    scenes={scene["id"]:scene for split in allowed_splits for scene in scenarios["splits"][split]}
    trajectories=block["trajectories"];rows=block["rows"]
    if not 1<=len(trajectories)<=400 or not 1<=len(rows)<=50000:raise ValueError("Raw trajectory evidence exceeds bounded contract")
    if len({t["episode_id"] for t in trajectories})!=len(trajectories):raise ValueError("Repeated raw trajectory ID")
    if sum(len(t["player_actions"]) for t in trajectories)>50000:raise ValueError("Raw trajectory replay exceeds bounded audit steps")
    requested={t["episode_id"]:set() for t in trajectories}
    for row in rows:
        _integer(row["frame"],0,scenarios["configuration"]["horizon"])
        if row["episode_id"] not in requested or row["agent_id"]!="robot_2":raise ValueError("Unknown raw trajectory or role")
        requested[row["episode_id"]].add(row["frame"])
    verified={};audit_steps=0
    for trace in trajectories:
        sid=trace["scenario_id"];episode=trace["episode_id"]
        if sid not in scenes or trace["initial_fingerprint"]!=scenes[sid]["fingerprint"]:raise ValueError("Raw trajectory uses a different initial scene or forbidden split")
        actions=trace["player_actions"]
        if not requested[episode] or max(requested[episode])!=len(actions) or any(a not in ACTIONS for a in actions):raise ValueError("Raw actions do not match the requested complete prefix")
        env=runtime.environment(scenes[sid])
        mode=trace.get("neural_execution","deterministic")
        if mode not in ("deterministic","stochastic"):raise ValueError("Unsupported actual neural trajectory mode")
        actor_rng=None
        if mode=="stochastic":
            actor_rng=np.random.default_rng(0)
            actor_rng.bit_generator.state=trace["actor_rng_state"]
        for frame in range(len(actions)+1):
            if frame in requested[episode]:verified[(episode,frame)]=(sid,env.snapshot())
            if frame<len(actions):
                if mode=="deterministic":runtime.step(env,actions[frame])
                else:
                    if env.done:raise ValueError("Raw trajectory continues beyond physical termination")
                    proposed,_=runtime.actor.act(env.observations(),deterministic=False,rng=actor_rng)
                    env.step({"robot_1":actions[frame],"robot_2":proposed["robot_2"]})
                audit_steps+=1
    observations=[];groups=[];ids=[];seen=set()
    for row in rows:
        sid,snapshot=verified[(row["episode_id"],row["frame"])]
        if row["scenario_id"]!=sid or digest(snapshot)!=digest(row["snapshot"]):raise ValueError("Raw snapshot is not reachable from its claimed frozen initial scene")
        env=NativeWarehouseEnv();env.restore(snapshot);obs=env.observations()["robot_2"]
        if not np.array_equal(obs,np.asarray(row["observation"],dtype=np.float32)):raise ValueError("Explanation observation differs from physical state")
        expected=sorted(critical_groups(env,"robot_2"))
        if sorted(row["groups"])!=expected:raise ValueError("Critical labels differ from public geometry")
        key=(sid,digest(snapshot),row["agent_id"])
        if key in seen:raise ValueError("Duplicate explanation physical evidence")
        seen.add(key);observations.append(obs);groups.append(expected);ids.append(sid)
    return np.asarray(observations,dtype=np.float32),groups,ids,audit_steps


def explanation_fidelity(raw,runtime,program,scenarios):
    _finite(raw);_reject_fixture(raw)
    if (raw["version"]!=RAW_EXPLANATION_VERSION or raw["split"]!="explanation_test"
            or raw["actor_sha256"]!=runtime.actor.sha256 or raw["scenario_manifest_sha256"]!=digest(scenarios)):
        raise ValueError("Explanation raw source binding mismatch")
    matrix,groups,ids,audit_steps=_trajectory_rows(raw,runtime,scenarios,("explanation_test",))
    fit=raw["fit"]
    if not fit["splits"] or set(fit["splits"])-{"train","extraction"}:raise ValueError("Forbidden fit source split")
    fit_matrix,_,fit_ids,fit_steps=_trajectory_rows(fit,runtime,scenarios,fit["splits"])
    selection=raw["selection"]
    if selection["splits"]!=["extraction"]:raise ValueError("Forbidden tree-selection source split")
    selection_matrix,_,selection_ids,selection_steps=_trajectory_rows(selection,runtime,scenarios,selection["splits"])
    fingerprint=lambda row:sha256(np.asarray(row,dtype="<f4").tobytes()).hexdigest()
    heldout={fingerprint(row) for row in matrix}
    if {fingerprint(row) for row in fit_matrix}&heldout:raise ValueError("Exact explanation fit/holdout observations overlap")
    if {fingerprint(row) for row in selection_matrix}&heldout:raise ValueError("Exact tree-selection/holdout observations overlap")
    if len(matrix)<10 or min(len(set(fit_ids)),len(set(selection_ids)))<2:raise ValueError("Insufficient independent explanation evidence")
    neural=runtime.actor.logits(matrix).argmax(-1)
    names=runtime.actor.metadata["feature_names"]
    tree=np.array([ACTIONS.index(program.predict(dict(zip(names,map(float,row))))) for row in matrix])
    agree=neural==tree;critical={}
    for group in REQUIRED_CATEGORIES:
        mask=np.array([group in value for value in groups])
        critical[group]={"rows":int(mask.sum()),"scenarios":len({s for s,m in zip(ids,mask) if m}),
            "fidelity":float(agree[mask].mean()) if mask.any() else 0.}
    by_action={action:{"rows":int((neural==i).sum()),"fidelity":float(agree[neural==i].mean()) if (neural==i).any() else None} for i,action in enumerate(ACTIONS)}
    passed=bool(agree.mean()>=.9 and all(v["scenarios"]>=10 and v["fidelity"]>=.85 for v in critical.values()))
    return {"passed":passed,"rows":len(matrix),"action_fidelity":float(agree.mean()),"critical":critical,"by_action":by_action,
        "observation_sha256":sha256(matrix.tobytes()).hexdigest(),"fit_observation_sha256":sha256(fit_matrix.tobytes()).hexdigest(),
        "selection_observation_sha256":sha256(selection_matrix.tobytes()).hexdigest(),"selection_rows":len(selection_matrix),
        "exact_observation_overlap":0,"selection_holdout_overlap":0,"replayed_audit_steps":audit_steps+fit_steps+selection_steps}


def verify_parity(path,runtime,scenarios):
    with np.load(path,allow_pickle=False) as raw:
        if set(raw.files)!={"observations","scenario_ids","snapshots"}:raise ValueError("Parity raw fields mismatch")
        observations=raw["observations"].copy();ids=raw["scenario_ids"].tolist();snapshots=raw["snapshots"].tolist()
    if observations.ndim!=2 or observations.shape[1]!=runtime.actor.obs_dim or not 10<=len(observations)<=50000:
        raise ValueError("Invalid parity observation dimensions or sample count")
    if not np.isfinite(observations).all() or len(ids)!=len(observations) or len(snapshots)!=len(ids):raise ValueError("Invalid parity rows")
    allowed={scene["id"] for scene in scenarios["splits"]["calibration"]}
    coverage={key:0 for key in ("normal","zero_battery","charger","carrying","collision","timer_boundary")}
    for sid,encoded,obs in zip(ids,snapshots,observations):
        if sid not in allowed:raise ValueError("Parity may use only calibration scenes")
        snapshot=parse_json(encoded);env=NativeWarehouseEnv();env.restore(snapshot)
        if not any(np.array_equal(obs,value) for value in env.observations().values()):raise ValueError("Parity observation differs from physical snapshot")
        state=env.state
        coverage["normal"]+=int(not env.done and 0<state.frame<env.config.horizon-1)
        coverage["zero_battery"]+=int(any(a.battery==0 for a in state.agents))
        coverage["charger"]+=int(any(a.position==env.layout.charger_position for a in state.agents))
        coverage["carrying"]+=int(any(a.carrying_task_id is not None for a in state.agents))
        coverage["collision"]+=int(state.robot_collision_events>0)
        coverage["timer_boundary"]+=int(state.frame>=env.config.horizon-1)
    # Exact exported weights, not an arbitrary report's claimed Torch logits.
    import torch
    from env.warehouse_native.policy import NativeActorCritic
    with torch.random.fork_rng(devices=[]):
        model=NativeActorCritic(runtime.actor.obs_dim,runtime.actor.state_dim)
        model.actor.load_state_dict({name:torch.from_numpy(value.copy()) for name,value in runtime.actor.weights.items()})
        with torch.no_grad():a=model.actor_logits(torch.from_numpy(observations.astype(np.float32))).numpy()
    b=runtime.actor.logits(observations)
    softmax=lambda logits:np.exp(logits-logits.max(-1,keepdims=True))/np.exp(logits-logits.max(-1,keepdims=True)).sum(-1,keepdims=True)
    maximum=float(np.max(np.abs(a-b)));probability=float(np.max(np.abs(softmax(a)-softmax(b))))
    equal=bool(np.array_equal(a.argmax(-1),b.argmax(-1)))
    return {"passed":maximum<=1e-4 and probability<=1e-4 and equal and all(coverage.values()),
        "rows":len(a),"max_raw_logit_error":maximum,"max_probability_error":probability,
        "all_argmax_equal":equal,"coverage":coverage,"optimizer_steps":0}


def verify_task_calibration(report,rows,scenarios,scenario_file_hash):
    from backend.training import warehouse_native_task_calibration as calibration
    _reject_fixture(report);_reject_fixture(rows)
    if report["protocol"]!=calibration.PROTOCOL or report["protocol_sha256"]!=digest(calibration.PROTOCOL):raise ValueError("Task matching protocol changed")
    if report["scenario_file_sha256"]!=scenario_file_hash or report["evaluated_splits"]!=["play"] or report["participant_data_read"] is not False or report["final_test_rollouts"]!=0:
        raise ValueError("Task calibration source or split mismatch")
    expected=scenarios["splits"]["play"][:7]
    for row in rows:
        _integer(row["play_index"],0,6);scene=expected[row["play_index"]]
        role="practice" if row["play_index"]==0 else "X" if row["play_index"]<=3 else "Y"
        if row["scene_id"]!=scene["id"] or row["scene_fingerprint"]!=scene["fingerprint"] or row["initial_snapshot_sha256"]!=digest(scene["snapshot"]) or row["task_set"]!=role:
            raise ValueError("Task calibration substituted a study scene")
        metrics=row["metrics"]
        for name in ("deliveries","collisions","shutdowns","invalid_moves","steps","environment_step_calls"):_integer(metrics[name])
        if metrics["steps"]!=metrics["environment_step_calls"] or metrics["steps"]>scenarios["configuration"]["horizon"]:raise ValueError("Task calibration step accounting mismatch")
        if metrics["deliveries"]!=sum(metrics["deliveries_by_robot"].values()):raise ValueError("Task calibration contribution mismatch")
    actual=calibration.summarize(rows)
    if any(canonical(report.get(key))!=canonical(value) for key,value in actual.items()):raise ValueError("Task calibration summary differs from raw episodes")
    expected_sources={str(p.relative_to(ROOT)):file_hash(p) for p in sorted({Path(calibration.__file__).resolve(),
        *(ROOT/"env/warehouse_native").glob("*.py"),*(ROOT/"env/warehouse").glob("*.py")})}
    if report["source_sha256"]!=expected_sources:raise ValueError("Task calibration source changed")
    if actual["matching_status"]!="passed":raise ValueError("Fixed task matching failed")
    return actual


def verify_behavioral_artifact(acceptance,acceptance_path,release_root,program_path,runtime,scenarios):
    """Bind and independently replay the actual raw answers, before eligibility.

    No acceptance boolean grants access to this check. The private renderer
    harness used by the verifier has no public question-answer eligibility.
    """
    from ui.warehouse_native_answer_verification import verify_answer_report
    path=bound_path(Path(acceptance_path).resolve().parent,acceptance["behavioral_answer_artifact"])
    if not path.is_relative_to(Path(release_root).resolve()):raise ValueError("Behavioral answer artifact escapes release")
    report=_json(path);_reject_fixture(report)
    result=verify_answer_report(report,program_path,runtime,scenarios)
    if not result["passed"]:raise ValueError("Actual behavioral answer text/trajectory checks failed: "+str([c for c in result["cases"] if not c["passed"]][:5]))
    return result


@dataclass(frozen=True)
class ReleaseAudit:
    signature:str
    eligible:bool
    checks:dict
    artifacts:dict
    analysis:dict
    public_summary:dict

    @property
    def study_ready(self):return self.eligible


def load_release(path,actor_path,scenarios,explainer,question_bank):
    checks={};artifacts={};analysis={};root=Path(path).resolve().parent;manifest={};documents={};runtime=None
    sources=release_source_hashes()
    def check(name,operation):
        try:
            detail=operation()
            checks[name]={"passed":True,"detail":detail};return detail
        except Exception as error:
            checks[name]={"passed":False,"reason":str(error)};return None
    def manifest_check():
        nonlocal manifest
        manifest=_json(path);_reject_fixture(manifest)
        if manifest["version"]!=VERSION or manifest["namespace"]!="local_pilot" or manifest["formal_ready"] is not False:raise ValueError("Invalid local-study release schema")
        if set(manifest["artifacts"])!=set(ARTIFACT_NAMES):raise ValueError("Missing or unexpected release artifacts")
        if manifest["sources"]!=sources or manifest["generalization"]!=GENERALIZATION:raise ValueError("Release sources or generalization limits changed")
        return {"version":VERSION}
    check("manifest",manifest_check)
    for name in ARTIFACT_NAMES:
        def bind(name=name):
            artifact=bound_path(root,manifest["artifacts"][name]);artifacts[name]={"path":str(artifact),"sha256":file_hash(artifact)}
            if name not in ("actor","parity_rows","task_calibration_rows"):
                value=_json(artifact);_reject_fixture(value);documents[name]=value
            return {"sha256":artifacts[name]["sha256"]}
        check("artifact_"+name,bind)
    def actor_check():
        nonlocal runtime
        if Path(actor_path).resolve()!=Path(artifacts["actor"]["path"]):raise ValueError("Server Actor path differs from release")
        runtime=NativeRuntime(actor_path);_finite(runtime.actor.metadata);_reject_fixture(runtime.actor.metadata)
        if runtime.actor.sha256!=manifest["actor_sha256"] or runtime.signature!=manifest["runtime_signature"]:raise ValueError("Actor/runtime signature mismatch")
        _integer(runtime.actor.metadata["joint_steps"],1)
        return {"actor_sha256":runtime.actor.sha256}
    check("actor_runtime",actor_check)
    def scenarios_check():
        if digest(scenarios)!=digest(documents["scenarios"]) or digest(scenarios)!=manifest["scenario_manifest_sha256"]:raise ValueError("Server scenarios differ from release")
        return verify_scenarios(scenarios)
    check("scenarios",scenarios_check)
    def protocol_check():
        expected=_training_sources(documents["protocol"])
        if runtime.actor.metadata.get("experiment_version")=="warehouse-native-v2-paired-continuation.v1":
            from ui.warehouse_native_pair_release import verify_actor_metadata
            verify_actor_metadata(runtime.actor,documents["protocol"],scenarios)
        elif runtime.actor.metadata.get("experiment_version")!=documents["protocol"]["version"]:
            raise ValueError("Unknown foundation or paired continuation source contract")
        if expected!=manifest["training_sources"] or runtime.actor.metadata.get("source_sha256")!=digest(expected):raise ValueError("Actor training source chain differs from frozen runtime sources")
        if runtime.actor.metadata.get("protocol_sha256")!=digest(documents["protocol"]) or runtime.actor.metadata.get("scenario_manifest_sha256")!=digest(scenarios):raise ValueError("Actor protocol or scenario binding mismatch")
        return {"protocol":documents["protocol"]["version"]}
    check("training_protocol",protocol_check)
    def endpoint_check():
        if runtime.actor.metadata.get("experiment_version")=="warehouse-native-v2-paired-continuation.v1":
            from ui.warehouse_native_pair_release import verify_pair_endpoint
            return verify_pair_endpoint(documents["training_end"],root,documents["protocol"],scenarios)
        return verify_training_end(documents["training_end"],root,documents["protocol"],scenarios)
    training_end=check("training_endpoint",endpoint_check)
    results={}
    for name,split,count in (("validation","validation",50),("reference","validation",50),("random","validation",50),
            ("final_test","final_test",100),("final_reference","final_test",100),("final_random","final_test",100)):
        def evaluation(name=name,split=split,count=count):
            report=documents[name]
            kind="reference" if name in ("reference","final_reference") else "random" if name in ("random","final_random") else "neural"
            paired_validation=(name in ("validation","reference","random") and training_end is not None and training_end.get("kind")=="warehouse-native-v2-pair-end.v1")
            if paired_validation:
                if name=="validation":
                    candidates=[c for b in training_end["branches"].values() for c in b["candidates"]]
                    if not any(c["validation_sha256"]==artifacts[name]["sha256"] and c["actor_sha256"]==runtime.actor.sha256 for c in candidates):
                        raise ValueError("Validation is not an actual source-bound paired development report")
                elif artifacts[name]["sha256"]!=training_end["baseline_sha256"][name]:
                    raise ValueError("Release baseline differs from the actual paired foundation baseline")
            else:
                if report["evaluation"]!=evaluation_contract(kind):raise ValueError("Evaluation baseline/controller identity or source differs from this artifact role")
                if report["scenario_manifest_sha256"]!=digest(scenarios) or report["runtime_signature"]!=runtime.signature:raise ValueError("Evaluation source binding mismatch")
            if name in ("validation","final_test") and (report["actor_sha256"]!=runtime.actor.sha256 or report["joint_steps"]!=runtime.actor.metadata["joint_steps"]):raise ValueError("Evaluation uses another neural Actor")
            if split=="final_test" and (report["selection_sha256"]!=artifacts["selection"]["sha256"] or report["used_for_selection"] is not False):raise ValueError("Final report is not bound to prior validation selection")
            result=verified_report(report,scenarios["splits"][split],scenarios["configuration"]["horizon"],expected_count=count)
            results[name]=result;return {"summary":result["summary"]}
        check(name,evaluation)
    for label,names in (("validation_capability",("validation","reference","random")),("final_capability",("final_test","final_reference","final_random"))):
        def gate(names=names):
            result=capability(*(results[name] for name in names),documents["protocol"])
            if not result["eligible"]:raise ValueError("Capability failed: "+",".join(k for k,v in result["checks"].items() if not v))
            return result
        check(label,gate)
    check("baseline_artifacts_distinct",lambda:True if len({artifacts[name]["sha256"] for name in ("reference","random","final_reference","final_random")})==4 else (_ for _ in ()).throw(ValueError("Baseline artifacts were swapped or reused")))
    def selection_check():
        if runtime.actor.metadata.get("experiment_version")=="warehouse-native-v2-paired-continuation.v1":
            from ui.warehouse_native_pair_release import verify_pair_selection
            return verify_pair_selection(documents["selection"],documents["protocol"],scenarios,runtime.actor,artifacts["validation"]["sha256"],training_end)
        return verify_selection(documents["selection"],root,documents["protocol"],scenarios,runtime.actor,artifacts["validation"]["sha256"],training_end)
    check("frozen_selection",selection_check)
    def behavioral_check():
        return verify_behavioral_artifact(documents["explanation"],artifacts["explanation"]["path"],root,
            artifacts["program"]["path"],runtime,scenarios)
    check("behavioral_answer_recomputed",behavioral_check)
    def explanation_check():
        if type(explainer) is not NativeExplainer:raise ValueError("Actual NativeExplainer required")
        rebuilt=NativeExplainer(artifacts["program"]["path"],artifacts["explanation"]["path"],runtime.actor.sha256)
        if not rebuilt.eligible or not explainer.eligible or rebuilt.signature!=explainer.signature:raise ValueError("Actual explanation evidence has not passed or changed")
        rebuilt._assert_current(runtime)
        raw=documents["explanation_rows"]
        if raw["program_sha256"]!=artifacts["program"]["sha256"]:raise ValueError("Raw explanation program mismatch")
        fidelity=explanation_fidelity(raw,runtime,rebuilt.program,scenarios)
        accepted=documents["explanation"]["metrics"]
        if documents["explanation"]["holdout"]["holdout_observation_sha256"]!=fidelity["observation_sha256"]:
            raise ValueError("Accepted explanation holdout observations differ from bound raw rows")
        if documents["explanation"]["holdout"]["fit_observation_sha256"]!=fidelity["fit_observation_sha256"]:
            raise ValueError("Accepted extraction fit observations differ from bound raw rows")
        if (documents["explanation"]["holdout"]["selection_observation_sha256"]!=fidelity["selection_observation_sha256"]
                or documents["explanation"]["holdout"]["selection_rows"]!=fidelity["selection_rows"]):
            raise ValueError("Accepted extraction selection observations differ from bound raw rows")
        if fidelity["rows"]!=accepted["rows"] or not np.isclose(fidelity["action_fidelity"],accepted["action_fidelity"],atol=1e-12):raise ValueError("Accepted explanation summary differs from raw rows")
        for group in REQUIRED_CATEGORIES:
            if fidelity["critical"][group]!=accepted["critical"][group]:raise ValueError("Accepted critical fidelity differs from raw rows")
        if not fidelity["passed"]:raise ValueError("Raw explanation fidelity or critical coverage failed")
        for artifact in rebuilt.bound_artifacts:
            if not artifact.resolve().is_relative_to(root):raise ValueError("Explanation dependent artifact escapes release")
        return fidelity
    check("explanation_recomputed",explanation_check)
    def bank_check():
        if type(question_bank) is not NativeQuestionBank:raise ValueError("Actual NativeQuestionBank required")
        rebuilt=NativeQuestionBank(artifacts["bank"]["path"],runtime,scenarios)
        if rebuilt.test_fixture or not rebuilt.eligible or not question_bank.eligible or rebuilt.signature!=question_bank.signature:raise ValueError("Question bank failed real answer recomputation or binding")
        return rebuilt.checks
    check("bank_recomputed",bank_check)
    def parity_check():
        report=documents["parity"]
        if (report["version"]!=PARITY_VERSION or report["actor_sha256"]!=runtime.actor.sha256
                or report["runtime_signature"]!=runtime.signature or report["raw_sha256"]!=artifacts["parity_rows"]["sha256"]):raise ValueError("Parity raw binding mismatch")
        result=verify_parity(artifacts["parity_rows"]["path"],runtime,scenarios)
        if not result["passed"] or report["rows"]!=result["rows"]:raise ValueError("Actual same-weight Torch/NumPy parity or boundary coverage failed")
        return result
    check("parity_recomputed",parity_check)
    def task_check():
        report=documents["task_calibration"]
        if report["artifacts"]["episodes.jsonl"]!=artifacts["task_calibration_rows"]["sha256"]:raise ValueError("Task calibration raw episode hash mismatch")
        rows=[parse_json(line) for line in Path(artifacts["task_calibration_rows"]["path"]).read_text().splitlines() if line.strip()]
        return verify_task_calibration(report,rows,scenarios,artifacts["scenarios"]["sha256"])
    check("task_calibration_recomputed",task_check)
    def analysis_check():
        nonlocal analysis
        analysis=documents["analysis_protocol"]
        if analysis!=analysis_protocol():raise ValueError("Analysis or stage-permission protocol differs from frozen local study")
        return dict(analysis)
    check("analysis_protocol",analysis_check)
    check("sources_unchanged_during_load",lambda:True if release_source_hashes()==sources else (_ for _ in ()).throw(ValueError("Source changed during release inspection")))
    eligible=all(value["passed"] for value in checks.values())
    signature=digest({"version":VERSION,"manifest":manifest,"sources":sources,"artifacts":{k:v["sha256"] for k,v in artifacts.items()}})
    public={"version":VERSION,"signature":signature,"study_ready":eligible,"status":"local_study_ready" if eligible else "candidate_blocked","formal_ready":False}
    return ReleaseAudit(signature,eligible,checks,artifacts,analysis,public)


def manifest_template():
    return {"version":VERSION,"namespace":"local_pilot","formal_ready":False,"actor_sha256":None,
        "runtime_signature":None,"scenario_manifest_sha256":None,"sources":release_source_hashes(),
        "training_sources":None,"generalization":dict(GENERALIZATION),
        "artifacts":{name:{"path":None,"sha256":None} for name in ARTIFACT_NAMES},
        "analysis_protocol_template":analysis_protocol(),"template_only_missing_evidence":True,
        "note":"Unfilled template; never evidence of capability or release eligibility."}


def main(argv=None):
    parser=argparse.ArgumentParser(description="Read-only local A/B release inspection; no training or fixture authorization")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest",type=Path);mode.add_argument("--template",type=Path,help="Write a new incomplete template; never overwrite or release")
    parser.add_argument("--actor",type=Path)
    args=parser.parse_args(argv)
    if args.template:
        if args.actor:parser.error("Template mode does not load an Actor")
        try:
            with args.template.open("x") as stream:stream.write(json.dumps(manifest_template(),ensure_ascii=False,indent=2)+"\n")
        except OSError as error:
            print(canonical({"study_ready":False,"formal_ready":False,"reason":str(error)}));return 2
        print(canonical({"study_ready":False,"formal_ready":False,"status":"incomplete_template_written"}));return 0
    if args.actor is None:parser.error("Inspection requires --actor")
    try:
        manifest=_json(args.manifest);_reject_fixture(manifest)
        root=args.manifest.resolve().parent;paths={k:bound_path(root,v) for k,v in manifest["artifacts"].items()}
        scenarios=_json(paths["scenarios"]);runtime=NativeRuntime(args.actor)
        _reject_fixture(runtime.actor.metadata)
        explainer=NativeExplainer(paths["program"],paths["explanation"],runtime.actor.sha256)
        bank=NativeQuestionBank(paths["bank"],runtime,scenarios)
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as error:
        print(canonical({"study_ready":False,"formal_ready":False,"reason":str(error)}));return 2
    report=load_release(args.manifest,args.actor,scenarios,explainer,bank)
    print(canonical({**report.public_summary,"checks":report.checks}));return 0 if report.eligible else 2


if __name__=="__main__":raise SystemExit(main())
