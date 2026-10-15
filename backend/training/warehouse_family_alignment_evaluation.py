"""Actual alignment-trainer admission and fixed original public-physics validation.

No old Actor or protocol version is substituted. This alignment producer has its
own changed learning-rate and feedback schedule contract. Evaluation delegates episode
execution, durable reservations and original capability statistics unchanged.
The read path checks acknowledged saved rows/traces without inference or reset.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import os

import numpy as np

from . import warehouse_family_alignment_trainer as trainer
from . import warehouse_native_continuation_evaluation as continuation
from . import warehouse_native_partner_mix_evaluation as compact
from . import warehouse_native_public_feedback_evaluation as physical
from . import warehouse_native_shutdown_evaluation as shutdown
from . import warehouse_family_branch_teacher_fit as ordinary
from . import warehouse_family_intervention_fit as pure
from . import warehouse_native_program_batch as prediction
from .warehouse_native_compact_validation import execute_episodes
from .warehouse_native_shutdown_result import _trace as verify_saved_trace
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_family_feedback_cycle_source import gate_values
from .warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from backend.warehouse_family_explanation import actor_parameter_sha256
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import NumPyNativeActor, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION
from core.program import ExecutableProgram

VERSION = "warehouse-family-alignment-evaluation.v1"
TEACHER_CHECK_VERSION = "warehouse-family-alignment-teacher-check.v1"
PARTNERS = physical.PARTNERS
STEP_BUDGET = 3 * 50 * 120
BINDINGS = (*shutdown.REQUIRED_BINDINGS, "feedback_branch", "actor_parameters_sha256",
    "entry_actor_sha256", "entry_actor_parameters_sha256", "entry_report_sha256", "teacher_alignment_sha256",
    "auxiliary_binding_sha256", "teacher_fit_sha256", "program_sha256",
    "auxiliary_observations_sha256", "initial_learning_state_sha256")


def execution_sources():
    result = trainer.execution_sources()
    result.update(continuation.execution_sources())
    for path in (Path(__file__), ROOT/'backend/warehouse_family_explanation.py',
                 ROOT/'backend/training/warehouse_native_shutdown_result.py', Path(shutdown.__file__),
                 Path(ordinary.__file__), Path(pure.__file__), Path(prediction.__file__)):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result



def _validate_protocol(protocol, *, fixture=False):
    # The actual producer owns the complete new 50k continuation protocol.
    # Never recast it as the older 20k protocol to reuse old admission.
    trainer.validate_protocol(protocol, fixture=fixture)
    continuation._gates(protocol)


def _alignment_bindings(metadata, protocol):
    paths = {"entry_actor_sha256": "entry_actor_sha256",
        "entry_actor_parameters_sha256": "entry_actor_parameters_sha256",
        "entry_report_sha256": "entry_report_sha256",
        "teacher_alignment_sha256": "teacher_alignment_sha256",
        "auxiliary_binding_sha256": "auxiliary_binding_sha256",
        "teacher_fit_sha256": "teacher_fit_semantic_sha256",
        "program_sha256": "teacher_program_semantic_sha256"}
    for key, field in paths.items():
        if key not in metadata or field not in protocol or not compact._same(metadata[key], protocol[field]):
            raise ValueError("Alignment export differs from its actual protocol: " + key)
    auxiliary = (protocol["auxiliary_binding"]["observations_sha256"]
        if protocol["feedback_branch"] == "feedback" else trainer.semantic(np.empty((0,2,197), np.float32)))
    if (metadata.get("feedback_branch") != protocol["feedback_branch"]
            or metadata.get("native_collector_feedback_enabled") is not False
            or metadata.get("auxiliary_observations_sha256") != auxiliary
            or not compact._same(metadata.get("feature_names"), protocol["feature_names"])):
        raise ValueError("Actual alignment arm, auxiliary pool or public feature order differs")


def validate_actor(actor, protocol, expected_actor_sha256, *, allow_test_fixture=False):
    """Zero-forward admission of the genuine alignment export and full protocol."""
    if (type(actor) is not NumPyNativeActor or not compact._sha(expected_actor_sha256)
            or type(allow_test_fixture) is not bool or not isinstance(protocol, dict)
            or protocol.get("version") != trainer.PROTOCOL_VERSION
            or protocol.get("test_fixture") is not allow_test_fixture):
        raise ValueError("A genuine alignment NumPy Actor, protocol, hash and fixture scope are required")
    if actor.artifact_sha256 != expected_actor_sha256 or file_hash(actor.path) != expected_actor_sha256:
        raise ValueError("Actual Actor bytes differ from the caller anchor")
    metadata = actor.metadata; config = collaborative_study_config()
    expected = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "experiment_version": trainer.VERSION, "protocol_sha256": digest(protocol),
        "source_sha256": digest(trainer.execution_sources()), "obs_dim": 197, "state_dim": 354,
        "hidden": 128, "architecture": "two_hidden_layer_tanh", "actions": list(physical.ACTIONS),
        "feature_names": list(observation_names(config)) + list(HISTORY_FEATURE_NAMES),
        "action_masks": False, "runtime_action_override": False, "test_fixture": allow_test_fixture,
        "public_feedback_mode": "observed", "public_feedback_version": physical.OBSERVER_VERSION,
        "branch": "own_credit", "delivery_credit_alpha": .5,
        "credit_reward_version": shutdown.CREDIT_REWARD_VERSION,
        "own_shutdown_reward_version": shutdown.SHUTDOWN_REWARD_VERSION,
        "cycle_id": protocol.get("cycle_id"), "candidate": True, "explanation_qualified": False}
    for key, value in expected.items():
        if not compact._same(metadata.get(key), value): raise ValueError("Alignment Actor contract differs: " + key)
    if any(metadata.get(key, False) is not False for key in ("study_ready", "formal_ready", "release_ready")):
        raise ValueError("Alignment candidate export cannot grant study or release readiness")
    for key in ("obs_dim", "state_dim", "hidden", "joint_steps"):
        if type(metadata.get(key)) is not int: raise ValueError("Actor dimensions/step must be exact integers")
    for key in BINDINGS:
        nullable = key in ("teacher_alignment_sha256", "auxiliary_binding_sha256", "teacher_fit_sha256", "program_sha256")
        if nullable and metadata.get("feedback_branch") == "control":
            if key not in metadata or metadata[key] is not None:
                raise ValueError("Control must have no teacher or auxiliary evidence: " + key)
        elif key.endswith("sha256") and key != "actor_sha256" and not compact._sha(metadata.get(key)):
            raise ValueError("Missing alignment Actor source binding: " + key)
    if actor_parameter_sha256(actor) != metadata["actor_parameters_sha256"]:
        raise ValueError("Actual six Actor parameter arrays differ from their semantic hash")
    with np.load(actor.path, allow_pickle=False) as archive:
        if (set(archive.files) != {*actor.weights, "metadata_json"}
                or not compact._same(metadata, json.loads(str(archive["metadata_json"].item())))
                or any(not np.array_equal(archive[k], v) for k, v in actor.weights.items())):
            raise ValueError("Cached Actor arrays differ from the anchored NPZ")
    # The producer validates its own new LR/schedule protocol; it is never
    # recast as an inherited producer to get past old admission.
    _validate_protocol(protocol, fixture=allow_test_fixture)
    _alignment_bindings(metadata, protocol)
    for key in ("source_report", "team_reference"):
        if digest(protocol.get(key)) != protocol.get(key+"_sha256"):
            raise ValueError("Immutable source/baseline report binding differs")
    base = protocol
    source, ev = protocol.get("source", {}), protocol.get("evaluation", {})
    original = protocol.get("source_protocol", {})
    arm, beta = metadata.get("shutdown_arm"), metadata.get("own_shutdown_beta")
    if (arm not in shutdown.ARMS or type(beta) not in (int, float) or beta != shutdown.ARMS[arm]
            or protocol.get("shutdown_arm") != arm
            or protocol.get("branch") != "own_credit" or protocol.get("delivery_credit_alpha") != .5
            or protocol.get("public_feedback_mode") != "observed"
            or protocol.get("public_feedback_version") != physical.OBSERVER_VERSION
            or protocol.get("reward") != physical.REWARD or protocol.get("collision_training_cost") != .05
            or original.get("version") != "warehouse-family-stable-feedback-protocol.v1"
            or source.get("trainer_version") != "warehouse-family-stable-feedback-trainer.v1"
            or source.get("protocol_sha256") != digest(original)
            or source.get("checkpoint_sha256") != metadata["source_checkpoint_sha256"]
            or source.get("state_sha256") != metadata["initialization_sha256"]):
        raise ValueError("Retained public physics, training reward or genuine source differs")
    cap, endpoints = base.get("budget", {}).get("maximum_ppo_joint_steps"), ev.get("checkpoints_ppo_steps")
    if (type(cap) is not int or cap <= 0 or base["budget"].get("maximum_ppo_joint_steps_per_arm") != cap
            or base["budget"].get("curriculum_generation_steps") != 0
            or type(endpoints) is not list or not endpoints or endpoints != sorted(set(endpoints))
            or any(type(x) is not int or not 0 < x <= cap for x in endpoints) or endpoints[-1] != cap
            or not 0 <= metadata["joint_steps"] <= cap):
        raise ValueError("Branch finite cap and ordered endpoint contract differs")
    expected_evaluation = {"partners": list(PARTNERS), "scenarios_per_partner": 50, "deterministic": True,
        "horizon": 120, "read_final_test": False, "training_credit_in_evaluation": False,
        "training_shutdown_in_evaluation": False, "maximum_environment_steps": len(endpoints)*STEP_BUDGET,
        "reuse_source_zero_step_validation": True}
    if any(not compact._same(ev.get(k), v) for k, v in expected_evaluation.items()):
        raise ValueError("Original fixed 3-by-50-by-120 validation contract differs")
    lineage, inherited = metadata.get("source_lineage"), metadata.get("source_counters", {}).get("joint_steps")
    if (type(inherited) is not int or inherited < 0 or inherited != source.get("cumulative_joint_steps")
            or not isinstance(lineage, list) or not lineage or lineage != base.get("source_lineage")
            or lineage[-1] != source):
        raise ValueError("Original source lineage/cumulative clock differs")
    prior = None
    for entry in lineage:
        if (not isinstance(entry, dict) or entry.get("branch") != "own_credit"
                or any(not compact._sha(entry.get(k)) for k in ("checkpoint_sha256", "state_sha256", "protocol_sha256", "source_sha256"))
                or type(entry.get("joint_steps")) is not int or type(entry.get("cumulative_joint_steps")) is not int
                or not 0 <= entry["joint_steps"] <= entry["cumulative_joint_steps"]
                or (prior is not None and prior+entry["joint_steps"] != entry["cumulative_joint_steps"])):
            raise ValueError("Source lineage accounting or hashes differ")
        prior = entry["cumulative_joint_steps"]
    condition, strength = metadata.get("feedback_branch"), metadata.get("feedback_lambda")
    if (condition not in ("control", "feedback") or metadata.get("feedback_enabled") is not (condition == "feedback")
            or type(strength) not in (int, float) or not 0 <= strength <= protocol["feedback"]["maximum_total_lambda"]
            or (condition == "control" and strength != 0)):
        raise ValueError("Recorded branch feedback condition/lambda differs")
    bindings = {key: expected_actor_sha256 if key == "actor_sha256" else deepcopy(metadata[key]) for key in BINDINGS}
    return {"version": VERSION, "actor_sha256": expected_actor_sha256, "actor_bindings": bindings,
        "metadata": deepcopy(metadata), "metadata_sha256": digest(metadata), "gates": continuation._gates(protocol),
        "source_report": deepcopy(protocol["source_report"]),
        "source_joint_steps": inherited, "cumulative_joint_steps": inherited+metadata["joint_steps"],
        "training_stage": "alignment_feedback", "configuration": asdict(config),
        "qualification_evaluated": False, "explanation_qualified": False, "release_ready": False,
        "test_fixture": allow_test_fixture}


def _material(actor_path, scenarios, protocol, expected, reference_report, random_report):
    actor_path = Path(actor_path).expanduser().resolve()
    if file_hash(actor_path) != expected: raise ValueError("Actor bytes differ before NPZ loading")
    actor = NumPyNativeActor(actor_path)
    admitted = validate_actor(actor, protocol, expected)
    metadata, bindings = admitted["metadata"], admitted["actor_bindings"]
    if metadata["joint_steps"] not in protocol["evaluation"]["checkpoints_ppo_steps"]:
        raise ValueError("Capability evaluation requires a registered alignment checkpoint endpoint")
    config = collaborative_study_config()
    scenes = physical._scenarios(scenarios, False, config)
    reference, random = physical._baseline(reference_report, scenes), physical._baseline(random_report, scenes)
    if digest(reference) != protocol["team_reference_sha256"]:
        raise ValueError("Evaluation team reference differs from the immutable branch protocol")
    sources, weights = execution_sources(), shutdown._weights_digest(actor)
    identity = {"version": VERSION, "actor_bindings": bindings, "actor_metadata_sha256": digest(metadata),
        "validation_entries_sha256": digest(scenes), "protocol_sha256": digest(protocol), "sources": sources,
        "configuration": asdict(config), "episode_count": 150, "step_budget": STEP_BUDGET,
        "reference_sha256": digest(reference), "random_sha256": digest(random), "test_fixture": False,
        "validation_reward": "original_shared_r1", "training_delivery_credit_alpha": .5,
        "cycle_id": protocol["cycle_id"], "training_stage": "alignment_feedback",
        "shutdown_arm": metadata["shutdown_arm"], "own_shutdown_beta": metadata["own_shutdown_beta"],
        "feedback_branch": metadata["feedback_branch"], "feedback_lambda": metadata["feedback_lambda"]}
    contexts = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            context = {"evaluation_version": VERSION, "mode": "observed", "branch": "own_credit",
                "cycle_id": protocol["cycle_id"], "training_stage": "alignment_feedback",
                "shutdown_arm": metadata["shutdown_arm"], "own_shutdown_beta": metadata["own_shutdown_beta"],
                "feedback_branch": metadata["feedback_branch"], "partner": partner,
                "scenario_id": scene["id"], "scenario_index": index, "initial_fingerprint": scene["fingerprint"],
                "episode_index": len(contexts), "seed": 17000+index, "horizon": 120,
                "maximum_environment_steps": 120, "test_fixture": False, "actor_bindings": bindings}
            context["operation_id"] = digest({"evaluation": digest(identity), "episode": context})
            contexts.append(context)
    def unchanged():
        if (execution_sources() != sources or not compact._same(actor.metadata, metadata)
                or shutdown._weights_digest(actor) != weights or file_hash(actor.path) != expected):
            raise ValueError("Evaluation Actor, weights or sources changed")
    return actor, scenes, config, admitted, reference, random, identity, contexts, unchanged


def _summary(rows, admitted, reference, random, identity, reserved):
    report = compact._summary(rows, admitted["metadata"], admitted["actor_bindings"],
        admitted["gates"], reference, random, admitted["test_fixture"], identity, reserved)
    report.update(version=VERSION, validation_reward="original_shared_r1", training_delivery_credit_alpha=.5,
        cycle_id=identity["cycle_id"], training_stage="alignment_feedback", shutdown_arm=identity["shutdown_arm"],
        training_own_shutdown_beta=identity["own_shutdown_beta"], feedback_branch=identity["feedback_branch"],
        feedback_lambda=identity["feedback_lambda"], remaining_reservable_steps=STEP_BUDGET-reserved,
        explanation_qualified=False, release_ready=False)
    if report["status"] == "completed" and not admitted["test_fixture"]:
        strict = gate_values(report, admitted["source_report"], team_reference=reference)
        initial_team = sum(admitted["source_report"]["summary"][p]["mean_team_deliveries"] for p in PARTNERS) / len(PARTNERS)
        team_retained = strict["validation_score"] >= .9 * initial_team
        strict["diagnostics"]["team_retention"] = {"current_mean": strict["validation_score"],
            "source_mean": initial_team, "minimum_mean": .9 * initial_team, "passed": team_retained,
            "anchor": "fixed_source_NN"}
        strict["capability_eligible"] = strict["capability_eligible"] and team_retained
        strict["update_kwargs"]["capability_eligible"] = strict["capability_eligible"]
        report["strict_guard"] = strict
        report["strict_capability_eligible"] = strict["capability_eligible"]
    else:
        report["strict_guard"] = {"capability_eligible": False, "reason": "incomplete_or_test_fixture"}
        report["strict_capability_eligible"] = False
    # Original ordinary capability statistics do not grant an explanation/release.
    report.update(formal_ready=False, explanation_qualified=False, release_ready=False)
    return report


def evaluate(actor_path, scenarios, protocol, output, *, expected_actor_sha256,
             reference_report, random_report, before_episode, on_episode, confirmed_operation_ids=None):
    """Exactly 150 original episodes; caller callbacks reserve and acknowledge each."""
    if not callable(before_episode) or not callable(on_episode):
        raise ValueError("Production evaluation requires caller reservation and acknowledgment callbacks")
    if confirmed_operation_ids is not None and (not isinstance(confirmed_operation_ids, (set, frozenset, tuple, list))
            or any(type(x) is not str for x in confirmed_operation_ids)):
        raise ValueError("External confirmed operation IDs required")
    actor, scenes, config, admitted, reference, random, identity, contexts, unchanged = _material(
        actor_path, scenarios, protocol, expected_actor_sha256, reference_report, random_report)
    rows, manifest = execute_episodes(actor, scenes, config=config, contexts=contexts, identity=identity,
        output=output, step_budget=STEP_BUDGET,
        dynamics={"reward": deepcopy(protocol["reward"]), "initialization": {"source_joint_steps": admitted["source_joint_steps"]}},
        validate_unchanged=unchanged, before_episode=before_episode, on_episode=on_episode,
        confirmed_operation_ids=confirmed_operation_ids)
    unchanged()
    report = _summary(rows, admitted, reference, random, identity, manifest["reserved_environment_steps"])
    report_path = Path(output).expanduser().resolve()/"report.json"
    if report_path.exists():
        if not compact._same(report, json.loads(report_path.read_bytes())):
            raise ValueError("Existing report differs from recomputed acknowledged rows")
    else: compact._put(report_path, report)
    return report


def read_existing(actor_path, scenarios, protocol, output, *, expected_actor_sha256,
                  reference_report, random_report, confirmed_operation_ids, expected_report_sha256=None):
    """Read complete acknowledged rows, verify traces and recompute original gates."""
    if not isinstance(confirmed_operation_ids, (set, frozenset, tuple, list)):
        raise ValueError("External acknowledgment IDs are required for existing episodes")
    _, _, config, admitted, reference, random, identity, contexts, unchanged = _material(
        actor_path, scenarios, protocol, expected_actor_sha256, reference_report, random_report)
    output = Path(output).expanduser().resolve()
    manifest_bytes = (output/"manifest.json").read_bytes(); manifest = json.loads(manifest_bytes)
    if (manifest.get("version") != VERSION or manifest.get("status") != "completed"
            or not compact._same(manifest.get("identity"), identity) or len(manifest.get("episodes", [])) != len(contexts)):
        raise ValueError("Only the complete original alignment validation matrix may be read")
    rows = []
    for entry, context in zip(manifest["episodes"], contexts):
        if context["operation_id"] not in confirmed_operation_ids: raise ValueError("Saved episode lacks external ACK")
        row = compact._read_entry(output, entry, context)
        verify_saved_trace((output/entry["trace"]["path"]).read_bytes(), context, row); rows.append(row)
    if (manifest.get("reserved_environment_steps") != STEP_BUDGET
            or manifest.get("actual_environment_steps") != sum(row["steps"] for row in rows)):
        raise ValueError("Saved evaluation budget accounting differs")
    report = _summary(rows, admitted, reference, random, identity, STEP_BUDGET)
    report_path = output/"report.json"
    if report_path.exists():
        if expected_report_sha256 is not None and file_hash(report_path) != expected_report_sha256:
            raise ValueError("Saved report bytes differ from the caller anchor")
        if not compact._same(report, json.loads(report_path.read_bytes())):
            raise ValueError("Saved report differs from recomputed original episode statistics/gates")
    elif expected_report_sha256 is not None: raise ValueError("Externally anchored report is missing")
    unchanged()
    if (output/"manifest.json").read_bytes() != manifest_bytes: raise ValueError("Manifest changed while reading")
    return report


def _array_binding(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
        "sha256": sha256(value.tobytes()).hexdigest()}


def _teacher_material(actor_path, protocol, selection_data, program, expected_actor_sha256,
                      selection_data_sha256):
    actor_path = Path(actor_path).expanduser().resolve()
    if file_hash(actor_path) != expected_actor_sha256:
        raise ValueError("Teacher-check Actor bytes differ before loading")
    actor = NumPyNativeActor(actor_path)
    admitted = validate_actor(actor, protocol, expected_actor_sha256)
    metadata = admitted["metadata"]
    if metadata["feedback_branch"] != "feedback":
        raise ValueError("Teacher checking belongs only to the feedback arm")
    if metadata["joint_steps"] not in protocol["evaluation"]["checkpoints_ppo_steps"]:
        raise ValueError("Teacher reliability requires a registered alignment boundary")
    contract = ordinary.validate_data(selection_data, pool="selection")
    if (not compact._sha(selection_data_sha256) or contract["data_sha256"] != selection_data_sha256
            or selection_data_sha256 != protocol["auxiliary_binding"]["source_selection_data_sha256"]):
        raise ValueError("Complete original selection data differs from its caller anchor")
    if (type(program) is not ExecutableProgram
            or tuple(program.feature_names) != tuple(metadata["feature_names"])
            or tuple(program.action_names) != tuple(physical.ACTIONS)
            or program.metadata.get("action_legality_features")
            or program.metadata.get("action_constraint_reason_features")
            or digest(program.to_dict()) != metadata["program_sha256"]):
        raise ValueError("The fixed, unmasked teacher program differs from the actual trainer binding")
    indices = np.asarray([i for i,k in enumerate(selection_data["kind"]) if k == "base"], np.int64)
    if len(indices) != 9289:
        raise ValueError("The fixed 390m development selection must retain all 9289 ordinary rows")
    base = ordinary._subset(selection_data, indices.tolist())
    observations = base["observations"]
    sources = execution_sources()
    request = {"version": TEACHER_CHECK_VERSION, "actor_sha256": expected_actor_sha256,
        "actor_parameters_sha256": metadata["actor_parameters_sha256"],
        "actor_metadata_sha256": digest(metadata), "protocol_sha256": digest(protocol),
        "joint_steps": metadata["joint_steps"], "selection_data_sha256": selection_data_sha256,
        "teacher_program_semantic_sha256": digest(program.to_dict()),
        "observations": _array_binding(observations), "row_indices": _array_binding(indices),
        "source_selection_rows": len(selection_data["actions"]), "ordinary_rows": len(indices),
        "sources": sources, "scope": "fixed development base rows; current NN labels; soft-feedback admission only",
        "probability_semantics": "NumPyNativeActor.logits float32; original max-subtracted exponential softmax",
        "prediction_semantics": prediction.VERSION, "maximum_NN_batches": 1,
        "maximum_tree_routes": 1, "environment_steps": 0, "explanation_qualified": False}
    def unchanged():
        if (execution_sources() != sources or file_hash(actor.path) != expected_actor_sha256
                or actor_parameter_sha256(actor) != metadata["actor_parameters_sha256"]
                or not compact._same(actor.metadata, metadata)
                or ordinary._data_digest(selection_data) != selection_data_sha256
                or digest(program.to_dict()) != request["teacher_program_semantic_sha256"]
                or _array_binding(observations) != request["observations"]
                or _array_binding(indices) != request["row_indices"]):
            raise ValueError("Teacher-check inputs, Actor or execution sources changed")
    return actor, base, indices, request, unchanged


def _ordinary_statistics(base, nn_probabilities, tree_probabilities):
    count = len(base["actions"])
    for value in (nn_probabilities, tree_probabilities):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (count,5)
                or not np.isfinite(value).all() or (value<0).any() or (value>1).any()
                or not np.allclose(value.sum(1), 1., atol=1e-6, rtol=0)):
            raise ValueError("Finite actual five-action probability arrays required")
    # Only this private ordinary copy receives newly queried NN truth. Source
    # selection labels, physical receipts and pair eligibility stay untouched.
    current = {**base, "probabilities": nn_probabilities,
        "actions": [physical.ACTIONS[i] for i in nn_probabilities.argmax(1)]}
    metrics = pure.statistics(current, [physical.ACTIONS[i] for i in tree_probabilities.argmax(1)])
    metrics["mean_kl"] = float(np.mean(np.sum(nn_probabilities * (
        np.log(nn_probabilities.clip(1e-8))-np.log(tree_probabilities.clip(1e-8))), axis=-1)))
    return metrics, ordinary._fidelity_gate(metrics)


def _teacher_report(request, base, nn_probabilities, tree_probabilities, evidence):
    metrics, gate = _ordinary_statistics(base, nn_probabilities, tree_probabilities)
    return {"version": TEACHER_CHECK_VERSION, "status": "completed", "passed": bool(gate["passed"]),
        **{k: request[k] for k in ("actor_parameters_sha256", "teacher_program_semantic_sha256",
            "protocol_sha256", "joint_steps", "selection_data_sha256")},
        "ordinary_metrics": metrics, "ordinary_gate": gate, "evidence_sha256": digest(evidence),
    }


def check_teacher(actor_path, protocol, selection_data, program, output, *, expected_actor_sha256,
                  selection_data_sha256):
    """One current NN batch on fixed base rows; no environment, fitting or gradients.

    A previous completed check is verified from saved evidence. Any directory
    without a completed report is pending/failed and is never sampled again.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        if (output/"report.json").exists():
            return read_teacher_check(actor_path, protocol, selection_data, program, output,
                expected_actor_sha256=expected_actor_sha256, selection_data_sha256=selection_data_sha256)
        raise ValueError("Pending teacher check cannot automatically repeat a NN query")
    actor, base, indices, request, unchanged = _teacher_material(actor_path, protocol, selection_data,
        program, expected_actor_sha256, selection_data_sha256)
    output.mkdir(parents=True, exist_ok=False)
    compact._put(output/"request.json", request)
    counts = {"NN_batches_attempted": 0, "NN_batches_completed": 0, "NN_query_rows_attempted": 0,
        "NN_query_rows_completed": 0, "tree_routes_attempted": 0, "tree_routes_completed": 0,
        "environment_steps": 0, "fit_calls": 0, "PT_loads": 0}
    compact._put(output/"pending.json", {"version": TEACHER_CHECK_VERSION, "status": "pending",
        "request_sha256": file_hash(output/"request.json"), "reserved_NN_query_rows": len(indices),
        "reserved_NN_batches": 1, "automatic_retry": False})
    try:
        unchanged()
        counts.update(NN_batches_attempted=1, NN_query_rows_attempted=len(indices))
        logits = actor.logits(base["observations"])
        probabilities = np.exp(logits-logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        counts.update(NN_batches_completed=1, NN_query_rows_completed=len(indices), tree_routes_attempted=1)
        tree_probabilities = prediction.predict(program, base["observations"], actor.metadata["feature_names"])
        counts["tree_routes_completed"] = 1
        unchanged()
        with (output/"predictions.npz").open("xb") as stream:
            np.savez(stream, nn_probabilities=probabilities, tree_probabilities=tree_probabilities, row_indices=indices)
            stream.flush(); os.fsync(stream.fileno())
        execution = {"version": TEACHER_CHECK_VERSION, "status": "completed", "counts": counts,
            "request_sha256": file_hash(output/"request.json"),
            "predictions_sha256": file_hash(output/"predictions.npz")}
        compact._put(output/"execution.json", execution)
        evidence = {k: file_hash(output/name) for k,name in (("request_sha256","request.json"),
            ("predictions_sha256","predictions.npz"), ("execution_sha256","execution.json"))}
        report = _teacher_report(request, base, probabilities, tree_probabilities, evidence)
        unchanged()
        compact._put(output/"report.json", report)
        return report
    except BaseException as error:
        compact._put(output/"failure.json", {"version": TEACHER_CHECK_VERSION, "status": "failed",
            "error_type": type(error).__name__, "message": str(error), "counts": counts,
            "automatic_retry": False, "reserved_NN_query_rows": len(indices)})
        raise


def read_teacher_check(actor_path, protocol, selection_data, program, output, *, expected_actor_sha256,
                       selection_data_sha256, expected_report_sha256=None):
    """Verify saved current NN/tree probabilities and recompute the original gate; zero forward."""
    _, base, indices, request, unchanged = _teacher_material(actor_path, protocol, selection_data,
        program, expected_actor_sha256, selection_data_sha256)
    output = Path(output).expanduser().resolve()
    report_path = output/"report.json"
    if not report_path.exists() or (output/"failure.json").exists():
        raise ValueError("Only a completed, nonfailed teacher check may be reused")
    if expected_report_sha256 is not None and file_hash(report_path) != expected_report_sha256:
        raise ValueError("Saved teacher report differs from its caller anchor")
    names = ("request.json", "pending.json", "predictions.npz", "execution.json", "report.json")
    hashes = {name:file_hash(output/name) for name in names}
    if not compact._same(json.loads((output/"request.json").read_bytes()), request):
        raise ValueError("Saved teacher request differs from the actual immutable inputs")
    pending = json.loads((output/"pending.json").read_bytes())
    expected_pending = {"version": TEACHER_CHECK_VERSION, "status": "pending",
        "request_sha256": hashes["request.json"], "reserved_NN_query_rows": len(indices),
        "reserved_NN_batches": 1, "automatic_retry": False}
    if not compact._same(pending, expected_pending):
        raise ValueError("Original teacher-check reservation differs")
    with np.load(output/"predictions.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"nn_probabilities", "tree_probabilities", "row_indices"}:
            raise ValueError("Saved teacher prediction archive schema differs")
        nn_probabilities, tree_probabilities = saved["nn_probabilities"], saved["tree_probabilities"]
        if _array_binding(saved["row_indices"]) != request["row_indices"]:
            raise ValueError("Saved prediction rows differ from actual original base-row indices")
    execution = json.loads((output/"execution.json").read_bytes())
    expected_execution = {"version": TEACHER_CHECK_VERSION, "status": "completed",
        "request_sha256": hashes["request.json"], "predictions_sha256": hashes["predictions.npz"],
        "counts": {"NN_batches_attempted": 1, "NN_batches_completed": 1,
            "NN_query_rows_attempted": len(indices), "NN_query_rows_completed": len(indices),
            "tree_routes_attempted": 1, "tree_routes_completed": 1,
            "environment_steps": 0, "fit_calls": 0, "PT_loads": 0}}
    if not compact._same(execution, expected_execution):
        raise ValueError("Saved NN/tree execution receipt differs")
    evidence = {"request_sha256": hashes["request.json"], "predictions_sha256": hashes["predictions.npz"],
        "execution_sha256": hashes["execution.json"]}
    report = _teacher_report(request, base, nn_probabilities, tree_probabilities, evidence)
    if not compact._same(report, json.loads(report_path.read_bytes())):
        raise ValueError("Saved teacher report differs from recomputed probabilities and original gate")
    unchanged()
    if any(file_hash(output/name) != expected for name,expected in hashes.items()):
        raise ValueError("Teacher-check evidence changed while reading")
    return report
