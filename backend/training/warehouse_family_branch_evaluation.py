"""Actual branch-trainer admission and fixed original public-physics validation.

No old Actor or protocol version is substituted. Evaluation delegates episode
execution, durable reservations and original capability statistics unchanged.
The read path checks acknowledged saved rows/traces without inference or reset.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from . import warehouse_family_branch_trainer as trainer
from . import warehouse_native_continuation_evaluation as continuation
from . import warehouse_native_partner_mix_evaluation as compact
from . import warehouse_native_public_feedback_evaluation as physical
from . import warehouse_native_shutdown_evaluation as shutdown
from .warehouse_native_compact_validation import execute_episodes
from .warehouse_native_shutdown_result import _trace as verify_saved_trace
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from backend.warehouse_family_explanation import actor_parameter_sha256
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import NumPyNativeActor, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION

VERSION = "warehouse-family-branch-evaluation.v1"
PARTNERS = physical.PARTNERS
STEP_BUDGET = 3 * 50 * 120
BINDINGS = (*shutdown.REQUIRED_BINDINGS, "feedback_branch", "actor_parameters_sha256")


def execution_sources():
    result = trainer.execution_sources()
    result.update(continuation.execution_sources())
    for path in (Path(__file__), ROOT/'backend/warehouse_family_explanation.py',
                 ROOT/'backend/training/warehouse_native_shutdown_result.py', Path(shutdown.__file__)):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def validate_actor(actor, protocol, expected_actor_sha256, *, allow_test_fixture=False):
    """Zero-forward admission of the genuine branch export and full protocol."""
    if (type(actor) is not NumPyNativeActor or not compact._sha(expected_actor_sha256)
            or type(allow_test_fixture) is not bool or not isinstance(protocol, dict)
            or protocol.get("version") != trainer.PROTOCOL_VERSION
            or protocol.get("test_fixture") is not allow_test_fixture):
        raise ValueError("A genuine branch NumPy Actor, protocol, hash and fixture scope are required")
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
        if not compact._same(metadata.get(key), value): raise ValueError("Branch Actor contract differs: " + key)
    for key in ("obs_dim", "state_dim", "hidden", "joint_steps"):
        if type(metadata.get(key)) is not int: raise ValueError("Actor dimensions/step must be exact integers")
    for key in BINDINGS:
        if key.endswith("sha256") and key != "actor_sha256" and not compact._sha(metadata.get(key)):
            raise ValueError("Missing branch Actor source binding: " + key)
    if actor_parameter_sha256(actor) != metadata["actor_parameters_sha256"]:
        raise ValueError("Actual six Actor parameter arrays differ from their semantic hash")
    with np.load(actor.path, allow_pickle=False) as archive:
        if (set(archive.files) != {*actor.weights, "metadata_json"}
                or any(not np.array_equal(archive[k], v) for k, v in actor.weights.items())):
            raise ValueError("Cached Actor arrays differ from the anchored NPZ")
    base = protocol.get("native_protocol")
    if not isinstance(base, dict) or base.get("version") != trainer.native.PROTOCOL_VERSION:
        raise ValueError("The real native ShutdownContinuation protocol is required")
    for key, value in base.items():
        if key not in ("version", "feedback_training_enabled") and not compact._same(protocol.get(key), value):
            raise ValueError("Branch wrapper changed the native learning contract: " + key)
    if (base.get("feedback_training_enabled") is not False or protocol.get("feedback_training_enabled") is not True
            or protocol.get("feedback_branches") != ["control", "feedback"]
            or protocol.get("explanation_qualification_granted") is not False
            or protocol.get("guard") != {"full_and_warmup": True, "per_partner_nn_retention": .9,
                "team_drop_guard": "unchanged_ExactProgramManager"}
            or protocol.get("feedback") != {"direction": "KL(nn||tree)", "maximum_total_lambda": .01,
                "ordinary_fraction": .5, "branch_fraction": .5, "refresh_interval_joint_steps": 50000,
                "teacher_version": trainer.TEACHER_VERSION, "action_controller": "neural_actor_only",
                "runtime_action_override": False, "branch_rows_supply_ppo_targets": False}):
        raise ValueError("Actual branch feedback objective or strict guard differs")
    trainer._feedback_config(protocol.get("feedback_config"), allow_test_fixture)
    for key in ("source_report", "team_reference"):
        if digest(protocol.get(key)) != protocol.get(key+"_sha256"):
            raise ValueError("Immutable source/baseline report binding differs")
    source, ev = base.get("source", {}), base.get("evaluation", {})
    original = base.get("source_protocol", {})
    arm, beta = metadata.get("shutdown_arm"), metadata.get("own_shutdown_beta")
    if (arm not in shutdown.ARMS or type(beta) not in (int, float) or beta != shutdown.ARMS[arm]
            or base.get("shutdown_arm") != arm or base.get("own_shutdown_beta") != beta
            or base.get("branch") != "own_credit" or base.get("delivery_credit_alpha") != .5
            or base.get("public_feedback_mode") != "observed" or base.get("public_feedback_version") != physical.OBSERVER_VERSION
            or base.get("reward") != physical.REWARD or base.get("collision_training_cost") != .05
            or source.get("trainer_version") != trainer.native.VERSION
            or original.get("version") != trainer.native.PROTOCOL_VERSION
            or source.get("protocol_sha256") != digest(original)
            or source.get("checkpoint_sha256") != metadata["source_checkpoint_sha256"]
            or source.get("state_sha256") != metadata["initialization_sha256"]):
        raise ValueError("Retained public physics, training reward or genuine source differs")
    for key in ("training", "reward", "collision_training_cost", "seed", "partners_by_branch"):
        if not compact._same(base.get(key), original.get(key)):
            raise ValueError("Inherited training contract differs: " + key)
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
            or type(strength) not in (int, float) or not 0 <= strength <= .01
            or (condition == "control" and strength != 0)):
        raise ValueError("Recorded branch feedback condition/lambda differs")
    bindings = {key: expected_actor_sha256 if key == "actor_sha256" else deepcopy(metadata[key]) for key in BINDINGS}
    return {"version": VERSION, "actor_sha256": expected_actor_sha256, "actor_bindings": bindings,
        "metadata": deepcopy(metadata), "metadata_sha256": digest(metadata), "gates": continuation._gates(protocol),
        "source_joint_steps": inherited, "cumulative_joint_steps": inherited+metadata["joint_steps"],
        "training_stage": "branch_feedback", "configuration": asdict(config),
        "qualification_evaluated": False, "explanation_qualified": False, "release_ready": False,
        "test_fixture": allow_test_fixture}


def _material(actor_path, scenarios, protocol, expected, reference_report, random_report):
    actor_path = Path(actor_path).expanduser().resolve()
    if file_hash(actor_path) != expected: raise ValueError("Actor bytes differ before NPZ loading")
    actor = NumPyNativeActor(actor_path)
    admitted = validate_actor(actor, protocol, expected)
    metadata, bindings = admitted["metadata"], admitted["actor_bindings"]
    if metadata["joint_steps"] not in protocol["evaluation"]["checkpoints_ppo_steps"]:
        raise ValueError("Capability evaluation requires a registered branch checkpoint endpoint")
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
        "cycle_id": protocol["cycle_id"], "training_stage": "branch_feedback",
        "shutdown_arm": metadata["shutdown_arm"], "own_shutdown_beta": metadata["own_shutdown_beta"],
        "feedback_branch": metadata["feedback_branch"], "feedback_lambda": metadata["feedback_lambda"]}
    contexts = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            context = {"evaluation_version": VERSION, "mode": "observed", "branch": "own_credit",
                "cycle_id": protocol["cycle_id"], "training_stage": "branch_feedback",
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
        admitted["gates"], reference, random, False, identity, reserved)
    report.update(version=VERSION, validation_reward="original_shared_r1", training_delivery_credit_alpha=.5,
        cycle_id=identity["cycle_id"], training_stage="branch_feedback", shutdown_arm=identity["shutdown_arm"],
        training_own_shutdown_beta=identity["own_shutdown_beta"], feedback_branch=identity["feedback_branch"],
        feedback_lambda=identity["feedback_lambda"], remaining_reservable_steps=STEP_BUDGET-reserved,
        explanation_qualified=False, release_ready=False)
    return report


def evaluate(actor_path, scenarios, branch_protocol, output, *, expected_actor_sha256,
             reference_report, random_report, before_episode, on_episode, confirmed_operation_ids=None):
    """Exactly 150 original episodes; caller callbacks reserve and acknowledge each."""
    if not callable(before_episode) or not callable(on_episode):
        raise ValueError("Production evaluation requires caller reservation and acknowledgment callbacks")
    if confirmed_operation_ids is not None and (not isinstance(confirmed_operation_ids, (set, frozenset, tuple, list))
            or any(type(x) is not str for x in confirmed_operation_ids)):
        raise ValueError("External confirmed operation IDs required")
    actor, scenes, config, admitted, reference, random, identity, contexts, unchanged = _material(
        actor_path, scenarios, branch_protocol, expected_actor_sha256, reference_report, random_report)
    rows, manifest = execute_episodes(actor, scenes, config=config, contexts=contexts, identity=identity,
        output=output, step_budget=STEP_BUDGET,
        dynamics={"reward": deepcopy(branch_protocol["reward"]), "initialization": {"source_joint_steps": admitted["source_joint_steps"]}},
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


def read_existing(actor_path, scenarios, branch_protocol, output, *, expected_actor_sha256,
                  reference_report, random_report, confirmed_operation_ids, expected_report_sha256=None):
    """Read complete acknowledged rows, verify traces and recompute original gates."""
    if not isinstance(confirmed_operation_ids, (set, frozenset, tuple, list)):
        raise ValueError("External acknowledgment IDs are required for existing episodes")
    _, _, config, admitted, reference, random, identity, contexts, unchanged = _material(
        actor_path, scenarios, branch_protocol, expected_actor_sha256, reference_report, random_report)
    output = Path(output).expanduser().resolve()
    manifest_bytes = (output/"manifest.json").read_bytes(); manifest = json.loads(manifest_bytes)
    if (manifest.get("version") != VERSION or manifest.get("status") != "completed"
            or not compact._same(manifest.get("identity"), identity) or len(manifest.get("episodes", [])) != len(contexts)):
        raise ValueError("Only the complete original branch validation matrix may be read")
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
