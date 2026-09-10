"""Six bounded RCPD candidates from two-role ordinary and branch NN labels.

The caller supplies already verified development records, not a model or an
environment. Ordinary reliability controls the manager; counterfactual goals
are reported separately and never grant independent explanation eligibility.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import math

import numpy as np

from . import warehouse_family_intervention_fit as pure
from . import warehouse_native_expanded_rcpd as exact
from .warehouse_native_common import ROOT, digest, jsonable
from env.warehouse_native.feedback import FeedbackConfig

VERSION = "warehouse-family-dual-role-branch-teacher-rcpd.v1"
DATA_VERSION = "warehouse-family-dual-role-branch-teacher-data.v1"
CANDIDATES = tuple((depth, leaves, structure) for depth,leaves in ((6,32),(8,64),(12,256))
                   for structure in (0.,.5))
ROLES = ("robot_1", "robot_2")
GROUPS = pure.GROUPS


def feedback_config():
    return FeedbackConfig(depths=(6, 8, 12), leaves=(32, 64, 256))


def contract():
    return dict(version=VERSION, data_version=DATA_VERSION, candidates=[list(c) for c in CANDIDATES],
        maximum_rcpd_calls=6, maximum_sklearn_fits=52, min_samples_leaf=8,
        action_structure_weight=[0.,.5], counterfactual_loss_weight=.2,
        counterfactual_changed_pair_weight=1., importance_weight_scale=8.,
        fit_view_weights="1+8*NN_top_two_margin per view; changed pair multiplier1",
        fit_presentation="all original rows plus two views per physically effective pair; no row removal",
        feedback_config=asdict(feedback_config()), role_scope=list(ROLES),
        minimum_fidelity=.9, minimum_non_wait_fidelity=.85, minimum_critical_fidelity=.85,
        minimum_critical_scenarios=10, maximum_mean_kl=.35,
        minimum_direction_fidelity=.85, minimum_effective_direction_scenarios=10,
        ordinary_reliability_scope="only base executed_NN_command rows, pooled across actual roles",
        manager_reliability="ordinary_reliable; CF85 is not a prerequisite for soft training",
        teacher_target="ordinary and all-row and both-role fidelity/direction goals; development only",
        selection="simplest if all goals pass; else ordinary-reliable candidates by CF then complexity; else diagnostic only",
        internal_split_group="whole physical initial scene fingerprint, all roles/partners together",
        labels="same frozen NN five-action soft probabilities; no program labels",
        structure_labels="optional NN argmax one-hot derived from same soft targets; leaves retain original five probabilities",
        prediction_semantics=pure.prediction.VERSION, independent_explanation_qualified=False)


def config(candidate):
    if tuple(candidate) not in CANDIDATES:
        raise ValueError("Candidate is outside the six predeclared capacity/structure settings")
    depth,leaves,structure=candidate
    return pure.RCPDConfig(max_depth=depth,max_leaf_nodes=leaves,max_predicates=None,
        min_samples_leaf=8,complexity_penalty=.001,random_seed=260910,regularization_lambda=.01,
        action_structure_weight=structure,counterfactual_changed_pair_weight=1.,importance_weight_scale=8.,counterfactual_loss_weight=.2)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Missing exact SHA256: " + name)
    return value


def _data_digest(data):
    value = {k:v for k,v in data.items() if k not in ("observations", "probabilities")}
    value = {**value, **{name:dict(shape=list(data[name].shape), dtype=str(data[name].dtype),
        sha256=sha256(np.ascontiguousarray(data[name]).tobytes()).hexdigest())
        for name in ("observations", "probabilities")}}
    return digest(value)


def validate_data(data, *, pool):
    """Validate the in-memory label contract, without pretending to replay physics."""
    if pool not in ("train", "selection") or not isinstance(data, dict):
        raise ValueError("Explicit train/selection datasets required")
    x, y = data.get("observations"), data.get("probabilities")
    if (not isinstance(x,np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 197
            or not isinstance(y,np.ndarray) or y.dtype != np.float32 or y.shape != (len(x),5)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or (y<0).any() or (y>1).any() or not np.allclose(y.sum(1),1.,atol=1e-6,rtol=0)):
        raise ValueError("Finite actual float32 observed197/five-action soft labels required")
    if len(x) < (64 if pool=="train" else 32):
        raise ValueError("Original minimum training64/selection32 rows is unchanged")
    keys=("actions","episode_ids","scene_fingerprints","groups","kind","roles","row_sources")
    if any(not isinstance(data.get(k),(list,tuple)) or len(data[k])!=len(x) for k in keys):
        raise ValueError("Every label requires matching scene/role/context metadata")
    actions=[pure.ACTIONS[i] for i in y.argmax(1)]
    if list(data["actions"]) != actions:
        raise ValueError("Hard labels must be the saved five-probability argmax")
    if set(data["roles"]) != set(ROLES) or not isinstance(data.get("pairs"),list):
        raise ValueError("Both real roles and explicit pairs required")
    if any(not any(r==role and k=="base" for r,k in zip(data["roles"],data["kind"])) for role in ROLES):
        raise ValueError("Both roles require actual ordinary NN rows")
    seen_rows=set(); labels={}; contexts={};episodes={}
    for i, source in enumerate(data["row_sources"]):
        fp=_sha(data["scene_fingerprints"][i],"scene fingerprint")
        role=data["roles"][i]; kind=data["kind"][i]; groups=data["groups"][i]
        if (kind not in ("base","counterfactual") or not isinstance(groups,(list,tuple))
                or len(groups)!=len(set(groups)) or not set(groups)<=set(GROUPS)
                or not isinstance(data["episode_ids"][i],str) or not data["episode_ids"][i]):
            raise ValueError("Invalid episode/kind/critical group")
        required=("context_id","scene_id","frame","target_role","phase","other_action","anchor_id","role_semantics")
        if not isinstance(source,dict) or any(k not in source for k in required):
            raise ValueError("Missing actual row source context")
        if (not isinstance(source["context_id"],str) or not source["context_id"]
                or type(source["frame"]) is not int or not 0<=source["frame"]<=120
                or source["target_role"]!=role or source["phase"]!=kind):
            raise ValueError("Row source frame/role/phase differs")
        if kind=="base":
            if source["role_semantics"]!="executed_NN_command":
                raise ValueError("Only actual NN-controlled ordinary rows are admitted")
        elif (source["role_semantics"]!="isolated_next_NN_query"
                or source["other_action"] not in pure.ACTIONS
                or not isinstance(source["anchor_id"],str) or not source["anchor_id"]):
            raise ValueError("Counterfactual endpoints are NN queries, never executed PPO labels")
        identity=(source["context_id"],role,kind,source["frame"],source["anchor_id"],source["other_action"])
        if identity in seen_rows:raise ValueError("Same original query row repeated in basic statistics")
        seen_rows.add(identity)
        context=(fp,str(source["scene_id"]),data["episode_ids"][i])
        if source["context_id"] in contexts and contexts[source["context_id"]]!=context:
            raise ValueError("Context changes physical initial scene or episode")
        contexts[source["context_id"]]=context
        episode=data["episode_ids"][i]
        if episode in episodes and episodes[episode]!=fp:
            raise ValueError("An episode crosses physical initial scenes")
        episodes[episode]=fp
        key=x[i].tobytes()
        if key in labels and labels[key]!=y[i].tobytes():
            raise ValueError("Identical Actor observation has conflicting saved soft labels")
        labels[key]=y[i].tobytes()
    seen_pairs=set(); referenced=set()
    for pair in data["pairs"]:
        if not isinstance(pair,dict):raise ValueError("Explicit pair record required")
        a,b=pair.get("baseline_index"),pair.get("changed_index")
        if type(a) is not int or type(b) is not int or not (0<=a<len(x) and 0<=b<len(x)) or a==b:
            raise ValueError("Invalid original pair indices")
        sa,sb=data["row_sources"][a],data["row_sources"][b]
        if ((a,b) in seen_pairs or any(data["kind"][i]!="counterfactual" for i in (a,b))
                or sa["other_action"]!="WAIT" or sb["other_action"]=="WAIT"
                or any(sa[k]!=sb[k] for k in ("context_id","frame","anchor_id","target_role"))
                or data["episode_ids"][a]!=data["episode_ids"][b]
                or pair.get("target_role")!=sa["target_role"]
                or pair.get("anchor_id")!=sa["anchor_id"]
                or pair.get("scene_fingerprint")!=data["scene_fingerprints"][a]
                or data["scene_fingerprints"][a]!=data["scene_fingerprints"][b]
                or list(pair.get("groups",[]))!=list(data["groups"][a])
                or list(data["groups"][a])!=list(data["groups"][b])):
            raise ValueError("Pair must link WAIT/change for the same real role and anchor")
        if (type(pair.get("physical_effect")) is not bool or type(pair.get("nn_changed")) is not bool
                or pair["nn_changed"]!=(actions[a]!=actions[b])):
            raise ValueError("Pair flags must preserve original physical receipt and actual NN argmax change")
        seen_pairs.add((a,b));referenced.update((a,b))
    # Nonterminal endpoints with a terminal counterpart may remain unpaired.
    return dict(data_sha256=_data_digest(data), rows=len(x), scenes=len(set(data["scene_fingerprints"])),
        pairs=len(data["pairs"]), paired_endpoint_rows=len(referenced), original_rows_removed=0,
        physical_effect_scope="bound caller collection receipt; not recomputed physics here")


def _subset(data, indices):
    indices=list(indices);mapping={old:new for new,old in enumerate(indices)}
    result={"observations":data["observations"][indices],"probabilities":data["probabilities"][indices]}
    for k in ("actions","episode_ids","scene_fingerprints","groups","kind","roles","row_sources"):
        result[k]=[data[k][i] for i in indices]
    result["pairs"]=[{**p,"baseline_index":mapping[p["baseline_index"]],"changed_index":mapping[p["changed_index"]]}
        for p in data["pairs"] if p["baseline_index"] in mapping and p["changed_index"] in mapping]
    return result


def metrics(program, data, feature_names):
    result=pure.evaluate(program,data,feature_names)
    result["by_role"]={role:pure.evaluate(program,_subset(data,[i for i,r in enumerate(data["roles"]) if r==role]),feature_names)
        for role in ROLES}
    return result


def _meets(value, minimum, scenes=1):
    return (value["rows"]>0 and value["scenarios"]>=scenes
        and value["fidelity"] is not None and math.isfinite(value["fidelity"]) and value["fidelity"]>=minimum)


def _fidelity_gate(value):
    checks={"overall":_meets(value["overall"],.9),"non_wait":_meets(value["non_wait"],.85),
        "mean_kl":math.isfinite(value["mean_kl"]) and value["mean_kl"]<=.35}
    for group in GROUPS:
        checks[group]=_meets(value["critical"][group],.85,10)
        checks[group+"_non_wait"]=_meets(value["critical"][group]["non_wait"],.85,10)
    return {"checks":checks,"passed":all(checks.values())}


def _direction_gate(value):
    checks={name:_meets(value["direction"][name],.85,10) for name in ("all",*GROUPS)}
    return {"checks":checks,"passed":all(checks.values())}


def gates(selection_metrics, ordinary_metrics):
    ordinary=_fidelity_gate(ordinary_metrics);all_rows=_fidelity_gate(selection_metrics)
    cf=_direction_gate(selection_metrics)
    by_role={r:{"fidelity":_fidelity_gate(selection_metrics["by_role"][r]),
                "direction":_direction_gate(selection_metrics["by_role"][r])} for r in ROLES}
    return dict(ordinary=ordinary,all_rows=all_rows,counterfactual=cf,by_role=by_role,
        ordinary_reliable=ordinary["passed"],cf_reliable=cf["passed"],
        teacher_target_met=ordinary["passed"] and all_rows["passed"] and cf["passed"]
            and all(v[k]["passed"] for v in by_role.values() for k in ("fidelity","direction")))


def execution_sources():
    files=[Path(__file__),Path(pure.__file__),Path(exact.__file__),Path(pure.prediction.__file__),
        ROOT/'env/warehouse_native/feedback.py',ROOT/'env/warehouse_native/policy.py',
        ROOT/'backend/training/warehouse_native_common.py',ROOT/'backend/training/warehouse_native_continuation_rcpd.py']
    files+=list((ROOT/'core').glob('*.py'))
    return {str(p.relative_to(ROOT)):pure.hashed(p) for p in files}


def _rate_or_zero(value):
    return value["fidelity"] if value["fidelity"] is not None else 0.


def choose(records):
    complete=[r for r in records if r["gates"]["teacher_target_met"]]
    ordinary=[r for r in records if r["gates"]["ordinary_reliable"]]
    simplest=lambda r:(r["complexity"]["loss"],r["selection_metrics"]["mean_kl"],r["index"])
    if complete:return min(complete,key=simplest),"all_targets_simplest"
    if ordinary:
        return min(ordinary,key=lambda r:(-_rate_or_zero(r["selection_metrics"]["direction"]["all"]),
            -min(_rate_or_zero(r["selection_metrics"]["direction"][g]) for g in GROUPS),*simplest(r))),"ordinary_reliable_cf_diagnostic_order"
    return min(records,key=lambda r:(-_rate_or_zero(r["ordinary_metrics"]["overall"]),
        r["ordinary_metrics"]["mean_kl"],*simplest(r))),"unreliable_diagnostic_only"


def validate_manager_state(state, *, feature_names, actor_bindings, step, expected_evidence_sha256=None):
    """Pure structure/binding/gate check; caller must anchor the actual fit bytes."""
    manager=exact.ExactProgramManager(feature_names,feedback_config());manager.load_state_dict(state)
    report=manager.last_fit_report;binding=report.get("observed197_bindings",{})
    if (report.get("version")!=VERSION or report.get("step")!=step
            or report.get("actor_bindings")!=actor_bindings
            or report.get("source_actor_sha256")!=actor_bindings["actor_sha256"]
            or manager.current_lambda!=0 or manager.last_step!=step or manager.last_fit_step!=step
            or manager.program is None or manager.last_gate.get("active") is not False
            or binding.get("actor_sha256")!=actor_bindings["actor_sha256"]
            or binding.get("actor_parameters_sha256")!=actor_bindings["actor_parameters_sha256"]
            or binding.get("cumulative_fit_step")!=step or binding.get("config_sha256")!=digest(contract())):
        raise ValueError("New teacher source/clock/config/closed schedule binding differs")
    computed=gates(report["selection_metrics"],report["ordinary_metrics"])
    if (report.get("gates")!=computed or manager.reliable is not computed["ordinary_reliable"]
            or report.get("reliable") is not manager.reliable
            or any(report.get(k) is not computed[k] for k in ("ordinary_reliable","cf_reliable","teacher_target_met"))
            or report.get("extraction_config")!=jsonable(contract())
            or report.get("explanation_qualified") is not False or report.get("release_ready") is not False):
        raise ValueError("Manager reliability must be the recomputed ordinary trajectory gate")
    meta=manager.program.metadata
    if (meta.get("native_feedback_version")!=VERSION or meta.get("native_source_actor_sha256")!=binding["actor_sha256"]
            or digest(meta.get("native_feedback_config"))!=digest(asdict(feedback_config()))
            or meta.get("observed197_bindings")!=binding or meta.get("prediction_semantics")!=pure.prediction.VERSION
            or tuple(manager.program.feature_names)!=tuple(feature_names)
            or meta.get("reliability_scope")!="ordinary_executed_nn_trajectories"
            or meta.get("role_scope")!=list(ROLES) or meta.get("action_constraint_reason_features")
            or meta.get("metrics",{}).get("explanation_eligible") is not False
            or meta.get("metrics",{}).get("feedback_eligible") is not manager.reliable):
        raise ValueError("New teacher program metadata differs from its actual manager")
    if expected_evidence_sha256 is not None:
        _sha(expected_evidence_sha256,"fit evidence")
        if digest({"binding":binding,"fit_report":report})!=expected_evidence_sha256:
            raise ValueError("Teacher fit evidence hash differs")
    return manager


def fit_teacher(train, selection, *, feature_names, actor_bindings, step, output,
                allow_test_fixture=False, prior_manager_state=None):
    """One new output, exactly six real RCPD calls; interrupted fits never retry."""
    if type(allow_test_fixture) is not bool or type(step) is not int or step<0:
        raise ValueError("Explicit fixture scope and cumulative fit clock required")
    if len(feature_names)!=197 or len(set(feature_names))!=197 or any(not isinstance(x,str) or not x for x in feature_names):
        raise ValueError("Full ordered197 feature names required")
    for key in ("actor_sha256","actor_parameters_sha256","protocol_sha256","source_sha256","runtime_signature"):
        _sha(actor_bindings.get(key),key)
    inputs={"train":validate_data(train,pool="train"),"selection":validate_data(selection,pool="selection")}
    overlap={"scene_fingerprints":len(set(train["scene_fingerprints"])&set(selection["scene_fingerprints"])),
        "episode_ids":len(set(train["episode_ids"])&set(selection["episode_ids"])),
        "exact_observations":len({x.tobytes() for x in train["observations"]}&{x.tobytes() for x in selection["observations"]})}
    if any(overlap.values()):raise ValueError("Cross-pool overlap rejected without deleting rows: "+str(overlap))
    if not allow_test_fixture and (inputs["train"]["scenes"]!=32 or inputs["selection"]["scenes"]!=16):
        raise ValueError("Production fixed development pools require32/16 physical initial scenes")
    output=Path(output).resolve()
    if output.exists():raise ValueError("Output already exists; incomplete fit attempts are not repeated")
    manager=exact.ExactProgramManager(feature_names,feedback_config())
    if prior_manager_state is not None:manager.load_state_dict(prior_manager_state)
    if step<manager.last_step:raise ValueError("Teacher cumulative fit clock cannot move backwards")
    manager.current_lambda=0.;manager.reliable=False
    sources=execution_sources();binding=dict(config_sha256=digest(contract()),
        training_data_sha256=inputs["train"]["data_sha256"],selection_data_sha256=inputs["selection"]["data_sha256"],
        actor_sha256=actor_bindings["actor_sha256"],actor_parameters_sha256=actor_bindings["actor_parameters_sha256"],cumulative_fit_step=step)
    views={k:pure.samples(d) for k,d in (("train",train),("selection",selection))}
    for pool,data in (("train",train),("selection",selection)):
        episode_scene=dict(zip(data["episode_ids"],data["scene_fingerprints"]))
        for row in views[pool]:row["split_scene"]=episode_scene[row["episode"]]
    output.mkdir(parents=True,exist_ok=False)
    pure.put(output/'request.json',dict(version=VERSION,contract=contract(),inputs=inputs,actor_bindings=actor_bindings,
        feature_names=list(feature_names),step=step,source_hashes=sources,overlap=overlap,
        presentation_rows={k:len(v) for k,v in views.items()},test_fixture=allow_test_fixture,
        prior_manager_state_sha256=digest(prior_manager_state) if prior_manager_state is not None else None))
    records=[];programs=[]
    try:
        for index,(depth,leaves,structure) in enumerate(CANDIDATES):
            cfg=config((depth,leaves,structure))
            pure.put(output/f'fit_{index:02d}.request.json',dict(index=index,config=asdict(cfg),maximum_sklearn_fits=depth,retry_allowed=False))
            with pure._isolated_host_rng():
                result=pure._NativeRCPD(cfg).fit(views["train"],
                    lambda row:dict(zip(pure.ACTIONS,map(float,row["probabilities"]))),
                    lambda row:dict(zip(feature_names,map(float,row["obs"]))),
                    validation_states=views["selection"],split_group_provider=lambda row:row["split_scene"],
                    counterfactual_pair_provider=lambda row:row["pair"],
                    program_metadata=dict(native_feedback_version=VERSION,native_source_actor_sha256=binding["actor_sha256"],
                        prediction_semantics=pure.prediction.VERSION,runtime_controller="native_neural_actor_only",
                        role_scope=list(ROLES),test_fixture=allow_test_fixture))
            program=pure.canonical_program(result.program,feature_names)
            sm=metrics(program,selection,feature_names)
            base=_subset(selection,[i for i,k in enumerate(selection["kind"]) if k=="base"])
            om=metrics(program,base,feature_names)
            record=dict(index=index,depth_cap=depth,leaf_cap=leaves,action_structure_weight=structure,
                config=asdict(cfg),selection_metrics=sm,ordinary_metrics=om,
                train_pool_metrics=metrics(program,train,feature_names),gates=gates(sm,om),
                complexity=pure.program_complexity(program,max_depth=12,max_leaf_count=256,max_predicate_count=255).to_dict(),
                core_extraction_summary=list(result.extraction_summary))
            records.append(record);programs.append(program)
            pure.put(output/f'program_{index:02d}.json',program.to_dict());pure.put(output/f'fit_{index:02d}.report.json',record)
        chosen,selection_reason=choose(records);gate=chosen["gates"];program=programs[chosen["index"]]
        report=dict(version=VERSION,step=step,source_actor_sha256=binding["actor_sha256"],actor_bindings=deepcopy(actor_bindings),
            observed197_bindings=binding,ordinary_reliable=gate["ordinary_reliable"],cf_reliable=gate["cf_reliable"],
            teacher_target_met=gate["teacher_target_met"],gates=gate,reliable=gate["ordinary_reliable"],
            reliability_scope="ordinary_executed_nn_trajectories",selection_reason=selection_reason,
            selected=chosen,candidates=records,selection_metrics=chosen["selection_metrics"],ordinary_metrics=chosen["ordinary_metrics"],
            train_rows=len(train["actions"]),validation_rows=len(selection["actions"]),overlap=overlap,rows_removed=0,
            actor_training_updates=0,environment_steps=0,NN_queries=0,PT_loads=0,actual_rcpd_calls=6,
            sklearn_fits_in_completed_core_loops=52,counts_scope="fixed core depth loops; no other fit invocation",
            execution_sources=sources,extraction_config=contract(),prediction_semantics=pure.prediction.VERSION,
            train_pool_includes_core_internal_validation=True,role_scope=list(ROLES),test_fixture=allow_test_fixture,
            independent_acceptance_executed=False,explanation_qualified=False,release_ready=False)
        meta={**deepcopy(program.metadata),"native_feedback_config":asdict(feedback_config()),
            "observed197_bindings":binding,"reliability_scope":"ordinary_executed_nn_trajectories",
            "metrics":{**deepcopy(program.metadata.get("metrics",{})),"reliable":gate["ordinary_reliable"],
                "feedback_eligible":gate["ordinary_reliable"],"explanation_eligible":False,"feedback_weight":0.,
                "ordinary_reliable":gate["ordinary_reliable"],"cf_reliable":gate["cf_reliable"],
                "teacher_target_met":gate["teacher_target_met"],"feedback_ineligibility_reasons":[] if gate["ordinary_reliable"] else ["ordinary_trajectory_reliability_failed"],
                "explanation_ineligibility_reasons":["independent_explanation_acceptance_not_granted"]}}
        manager.program=pure.canonical_program(program,feature_names,meta);manager.reliable=gate["ordinary_reliable"]
        manager.last_step=manager.last_fit_step=step;manager.last_fit_report=jsonable(report)
        manager.last_gate={"active":False,"lambda":0.,"step":step,"reason":"fit_complete_capability_and_schedule_required"}
        evidence=digest({"binding":binding,"fit_report":manager.last_fit_report})
        result=jsonable(dict(version=VERSION,reliable=manager.reliable,ordinary_reliable=manager.reliable,
            cf_reliable=gate["cf_reliable"],teacher_target_met=gate["teacher_target_met"],fit_report=manager.last_fit_report,
            program=manager.program.to_dict(),manager_state=manager.state_dict(),evidence_sha256=evidence,
            prediction_semantics=pure.prediction.VERSION,test_fixture=allow_test_fixture,qualified=False,
            explanation_qualified=False,release_ready=False))
        validate_manager_state(result["manager_state"],feature_names=feature_names,actor_bindings=actor_bindings,
            step=step,expected_evidence_sha256=evidence)
        if sources!=execution_sources() or any(_data_digest(d)!=inputs[k]["data_sha256"] for k,d in (("train",train),("selection",selection))):
            raise ValueError("Source or supplied data mutated during fitting")
        pure.put(output/'program.json',result["program"]);pure.put(output/'fit_result.json',result)
        pure.put(output/'report.json',{**report,"status":"completed","fit_result_sha256":pure.hashed(output/'fit_result.json'),
            "program_sha256":pure.hashed(output/'program.json'),"request_sha256":pure.hashed(output/'request.json')})
        return result
    except BaseException as error:
        pure.put(output/'failure.json',dict(version=VERSION,error_type=type(error).__name__,message=str(error),
            completed_rcpd_calls=len(records),pending_fit_not_retried=True,partial_fit_count_may_be_unknown=True,qualified=False))
        raise
