"""Independent grouped RCPD producer using the unchanged two-role data contract.

Only original critical-group membership is added to core sample views. It can
change the internal whole-scene validation split and candidate-depth scoring;
sklearn tree growth still uses the original weighted multi-output squared error.
No Actor, environment, or new label is constructed by this module.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from . import warehouse_family_branch_teacher_fit as original
from . import warehouse_family_intervention_fit as pure
from . import warehouse_native_expanded_rcpd as exact
from .warehouse_native_common import ROOT, digest, jsonable

VERSION = "warehouse-family-dual-role-grouped-branch-teacher-rcpd.v1"
DATA_VERSION = original.DATA_VERSION
CANDIDATES = original.CANDIDATES
ROLES = original.ROLES
GROUPS = original.GROUPS
feedback_config = original.feedback_config
config = original.config
validate_data = original.validate_data
metrics = original.metrics
gates = original.gates
choose = original.choose
_sha = original._sha
_data_digest = original._data_digest
_subset = original._subset
_NativeRCPD = pure._NativeRCPD


def contract():
    return {**original.contract(), "version": VERSION,
        "group_source": "unchanged original row groups; pair endpoints retain their anchor groups",
        "core_interaction_groups": list(GROUPS),
        "core_interaction_loss_weight": config(CANDIDATES[0]).interaction_loss_weight,
        "core_group_effect": "internal whole-scene validation stratification and candidate-depth scoring",
        "sklearn_growth_objective": "unchanged margin-weighted soft/argmax-structure multi-output squared error",
        "no_group_reweighting_or_row_filtering": True,
        "ungrouped_producer": original.VERSION}


def execution_sources():
    return {**original.execution_sources(),
        str(Path(__file__).resolve().relative_to(ROOT)): pure.hashed(__file__)}


def samples(data):
    """Keep the existing exact view order/weights, attaching original row provenance."""
    views = pure.samples(data)
    origins = [(i, None) for i in range(len(data["actions"]))]
    for pair_index, pair in enumerate(data["pairs"]):
        if not pair["physical_effect"]:
            continue
        a, b = pair["baseline_index"], pair["changed_index"]
        if (tuple(data["groups"][a]) != tuple(data["groups"][b])
                or tuple(data["groups"][a]) != tuple(pair["groups"])):
            raise ValueError("Pair views must retain the original same-anchor critical groups")
        origins.extend(((a, pair_index), (b, pair_index)))
    if len(views) != len(origins):
        raise ValueError("Original view presentation changed")
    for view, (row_index, pair_index) in zip(views, origins):
        view.update(groups=tuple(data["groups"][row_index]),
            split_scene=data["scene_fingerprints"][row_index],
            source_row_index=row_index, source_pair_index=pair_index,
            row_source=deepcopy(data["row_sources"][row_index]))
    return views


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
    views={k:samples(d) for k,d in (("train",train),("selection",selection))}
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
                result=_NativeRCPD(cfg).fit(views["train"],
                    lambda row:dict(zip(pure.ACTIONS,map(float,row["probabilities"]))),
                    lambda row:dict(zip(feature_names,map(float,row["obs"]))),
                    validation_states=views["selection"],split_group_provider=lambda row:row["split_scene"],
                    counterfactual_pair_provider=lambda row:row["pair"],
                    group_provider=lambda row:row["groups"],interaction_groups=GROUPS,
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
