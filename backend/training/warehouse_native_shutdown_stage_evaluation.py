"""Independent validation of retained-beta continuation and soft-feedback stages.

The Actor's actual new schema is admitted directly. Shared r1 public physics,
197 observed features, all five unmodified commands, and the original fixed
validation partners/gates remain unchanged. No training, PT load or final pool.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .warehouse_native_common import ROOT, digest, file_hash
from . import warehouse_native_shutdown_evaluation as previous
from .warehouse_native_shutdown_result import _trace as verify_saved_trace
from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_native_shutdown_feedback_trainer as feedback
from . import warehouse_native_continuation_evaluation as continuation
from . import warehouse_native_partner_mix_evaluation as compact
from . import warehouse_native_public_feedback_evaluation as physical
from .warehouse_native_compact_validation import execute_episodes
from .warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import NumPyNativeActor, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION

VERSION = "warehouse-native-own-shutdown-stage-evaluation.v2"
PARTNERS = physical.PARTNERS
REQUIRED_BINDINGS = previous.REQUIRED_BINDINGS
ACTOR_PROTOCOLS = {native.VERSION: native.PROTOCOL_VERSION, feedback.VERSION: feedback.PROTOCOL_VERSION}
SHUTDOWN_REWARD_VERSION = previous.SHUTDOWN_REWARD_VERSION


def execution_sources():
    result = previous.execution_sources()
    result.update(native.execution_sources()); result.update(feedback.execution_sources())
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    path = ROOT / "backend/training/warehouse_native_shutdown_result.py"
    result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def required_bindings(protocol):
    if protocol.get("version") == feedback.PROTOCOL_VERSION:
        return (*REQUIRED_BINDINGS, "feedback_branch", "actor_parameters_sha256")
    if protocol.get("version") == native.PROTOCOL_VERSION: return REQUIRED_BINDINGS
    raise ValueError("Unknown retained-beta training stage")


def _gates(protocol): return continuation._gates(protocol)


def _contract(actor, protocol, expected, fixture, config):
    if (type(fixture) is not bool or type(protocol) is not dict or protocol.get("test_fixture") is not fixture
            or type(actor) is not NumPyNativeActor):
        raise ValueError("A genuine NumPy Actor and explicit retained-beta stage protocol are required")
    if type(expected) is not dict or not set(required_bindings(protocol)) <= set(expected):
        raise ValueError("Independent stage Actor bindings are required")
    for key, value in expected.items():
        if key.endswith("sha256") and not compact._sha(value): raise ValueError("Malformed independent Actor binding")
    version = expected["experiment_version"]
    if ACTOR_PROTOCOLS.get(version) != protocol.get("version"):
        raise ValueError("Actor is not the genuine exported stage identity")
    stage = "feedback" if version == feedback.VERSION else "continuation"
    base = protocol.get("native_protocol") if stage == "feedback" else protocol
    if not isinstance(base, dict) or base.get("version") != native.PROTOCOL_VERSION:
        raise ValueError("The actual native continuation protocol is required")
    if stage == "feedback":
        if (protocol.get("feedback_training_enabled") is not True or protocol.get("feedback_branches") != ["control", "feedback"]
                or protocol.get("explanation_qualification_granted") is not False
                or protocol.get("feedback") != feedback.feedback_contract()):
            raise ValueError("The feedback stage's actual soft-loss contract differs")
        for key, value in base.items():
            if key not in ("version", "feedback_training_enabled") and not compact._same(protocol.get(key), value):
                raise ValueError("Feedback changed the native learning contract: " + key)
    if stage == "feedback": feedback.validated_feedback_config(protocol.get("feedback_config"), fixture)
    if base.get("feedback_training_enabled") is not False:
        raise ValueError("Native continuation cannot apply feedback")
    source, ev = base.get("source", {}), base.get("evaluation", {})
    arm, beta, branch, step, cycle = (expected[k] for k in ("shutdown_arm", "own_shutdown_beta", "branch", "joint_steps", "cycle_id"))
    if (arm not in previous.ARMS or type(beta) not in (int, float) or beta != previous.ARMS[arm]
            or base.get("shutdown_arm") != arm or base.get("own_shutdown_beta") != beta
            or base.get("own_shutdown_reward_version") != SHUTDOWN_REWARD_VERSION
            or branch != "own_credit" or base.get("branch") != branch or base.get("delivery_credit_alpha") != .5
            or base.get("public_feedback_mode") != "observed" or base.get("public_feedback_version") != physical.OBSERVER_VERSION
            or base.get("credit_reward_version") != previous.CREDIT_REWARD_VERSION):
        raise ValueError("Retained alpha, beta, arm or observation mode differs")
    cap, endpoints = base.get("budget", {}).get("maximum_ppo_joint_steps"), ev.get("checkpoints_ppo_steps")
    if (type(cap) is not int or cap <= 0 or base["budget"].get("maximum_ppo_joint_steps_per_arm") != cap
            or base["budget"].get("curriculum_generation_steps") != 0
            or type(endpoints) is not list or not endpoints or endpoints != sorted(set(endpoints))
            or any(type(s) is not int or not 0 < s <= cap for s in endpoints) or endpoints[-1] != cap
            or type(step) is not int or step not in endpoints or type(cycle) is not str or not cycle
            or cycle != protocol.get("cycle_id") or expected["protocol_sha256"] != digest(protocol)):
        raise ValueError("Stage finite cap, cycle or registered endpoint differs")
    if (ev.get("partners") != list(PARTNERS) or ev.get("scenarios_per_partner") != 50
            or ev.get("deterministic") is not True or ev.get("horizon") != 120 or ev.get("read_final_test") is not False
            or ev.get("training_credit_in_evaluation") is not False or ev.get("training_shutdown_in_evaluation") is not False
            or ev.get("maximum_environment_steps") != len(endpoints)*3*50*120
            or ev.get("reuse_source_zero_step_validation") is not True):
        raise ValueError("The original deterministic development matrix differs")
    if (not fixture and config != collaborative_study_config()) or type(config.horizon) is not int or not 1 <= config.horizon <= 120:
        raise ValueError("Public validation configuration changed")
    if not compact._same(base.get("reward"), physical.REWARD) or base.get("collision_training_cost") != .05:
        raise ValueError("Validation must retain original shared r1 reward and physical score")
    original = base.get("source_protocol", {})
    source_versions = {previous.TRAINER_VERSION: previous.PROTOCOL_VERSION, native.VERSION: native.PROTOCOL_VERSION}
    if (source_versions.get(source.get("trainer_version")) != original.get("version")
            or source.get("protocol_sha256") != digest(original)
            or source.get("branch") != branch or source.get("shutdown_arm") != arm or source.get("own_shutdown_beta") != beta
            or source.get("own_shutdown_reward_version") != SHUTDOWN_REWARD_VERSION
            or source.get("checkpoint_sha256") != expected["source_checkpoint_sha256"]
            or source.get("state_sha256") != expected["initialization_sha256"]):
        raise ValueError("The genuine retained-beta source identity differs")
    for key in ("training", "reward", "collision_training_cost", "seed"):
        if not compact._same(base.get(key), original.get(key)): raise ValueError("Inherited learning configuration differs: " + key)
    if not compact._same(base.get("partners_by_branch"), {branch: original.get("partners_by_branch", {}).get(branch)}):
        raise ValueError("Inherited partner mixture differs")
    metadata = actor.metadata
    required = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": 197, "state_dim": 354, "hidden": 128, "architecture": "two_hidden_layer_tanh",
        "actions": list(physical.ACTIONS), "action_masks": False, "runtime_action_override": False,
        "public_feedback_mode": "observed", "public_feedback_version": physical.OBSERVER_VERSION,
        "feature_names": list(observation_names(config)) + list(HISTORY_FEATURE_NAMES), "test_fixture": fixture,
        "credit_reward_version": previous.CREDIT_REWARD_VERSION, "delivery_credit_alpha": .5,
        "shutdown_arm": arm, "own_shutdown_beta": beta, "own_shutdown_reward_version": SHUTDOWN_REWARD_VERSION}
    if stage == "feedback":
        condition = expected["feedback_branch"]
        if condition not in ("control", "feedback"): raise ValueError("Unknown feedback branch")
        required.update(feedback_branch=condition, feedback_enabled=condition == "feedback")
        strength, maximum = metadata.get("feedback_lambda"), protocol.get("feedback_config", {}).get("lambda_max")
        if (type(strength) not in (int, float) or type(maximum) not in (int, float) or not 0 <= strength <= maximum
                or (condition == "control" and strength != 0)):
            raise ValueError("Recorded feedback lambda differs")
    else:
        required["feedback_enabled"] = False
        if any(k in metadata for k in ("feedback_branch", "feedback_lambda")): raise ValueError("Native export contains feedback identity")
    for key, value in required.items():
        if not compact._same(metadata.get(key), value): raise ValueError("Actor public/learning contract differs: " + key)
    for key, value in expected.items():
        actual = actor.artifact_sha256 if key == "actor_sha256" else metadata.get(key)
        if not compact._same(actual, value): raise ValueError("Actor source binding differs: " + key)
    if file_hash(actor.path) != expected["actor_sha256"]: raise ValueError("Actual Actor bytes differ")
    with np.load(actor.path, allow_pickle=False) as archive:
        if any(not np.array_equal(archive[name], value) for name, value in actor.weights.items()):
            raise ValueError("Cached Actor weights differ from the bound NPZ")
    inherited, lineage = metadata.get("source_counters", {}).get("joint_steps"), metadata.get("source_lineage")
    if (type(inherited) is not int or inherited < 0 or inherited != source.get("cumulative_joint_steps")
            or not isinstance(lineage, list) or not lineage or not all(isinstance(x, dict) for x in lineage)
            or not compact._same(lineage, base.get("source_lineage")) or not compact._same(lineage[-1], source)):
        raise ValueError("Explicit inherited source accounting differs")
    previous_steps = None; entered_shutdown = False
    old_versions = ("warehouse-native-delivery-credit-trainer.v1", continuation.TRAINER_VERSION)
    for item in lineage:
        kind = item.get("trainer_version")
        if (kind not in (*old_versions, previous.TRAINER_VERSION, native.VERSION) or item.get("branch") != branch
                or any(not compact._sha(item.get(k)) for k in ("checkpoint_sha256", "state_sha256", "protocol_sha256", "source_sha256"))
                or type(item.get("joint_steps")) is not int or type(item.get("cumulative_joint_steps")) is not int
                or not 0 <= item["joint_steps"] <= item["cumulative_joint_steps"]
                or (previous_steps is not None and previous_steps+item["joint_steps"] != item["cumulative_joint_steps"])):
            raise ValueError("Actual ancestry identity or cumulative counter differs")
        if kind not in old_versions:
            entered_shutdown = True
            if (item.get("shutdown_arm") != arm or item.get("own_shutdown_beta") != beta
                    or item.get("own_shutdown_reward_version") != SHUTDOWN_REWARD_VERSION):
                raise ValueError("Ancestral shutdown reward changed")
        elif entered_shutdown: raise ValueError("Ancestry regresses to a prior training identity")
        previous_steps = item["cumulative_joint_steps"]
    return deepcopy(metadata), _gates(protocol), inherited, stage


def validate_actor(actor, expected_bindings, protocol, config=None, allow_test_fixture=False):
    """Zero-forward admission for future consumers; never grants capability."""
    config = config or collaborative_study_config()
    metadata, gates, inherited, stage = _contract(actor, protocol, expected_bindings, allow_test_fixture, config)
    return {"version": VERSION, "metadata": metadata, "metadata_sha256": digest(metadata),
        "actor_sha256": actor.artifact_sha256, "weights_sha256": previous._weights_digest(actor),
        "actor_bindings": deepcopy(expected_bindings), "protocol_sha256": digest(protocol),
        "configuration": asdict(config), "training_stage": stage,
        "source_joint_steps": inherited, "cumulative_joint_steps": inherited+metadata["joint_steps"],
        "execution_sources": execution_sources(), "qualification_evaluated": False,
        "release_ready": False, "explanation_qualified": False, "test_fixture": allow_test_fixture}


def _material(actor_or_path, scenarios, protocol, step_budget, expected, reference_report, random_report, fixture, config):
    if type(step_budget) is not int or step_budget < 0 or type(fixture) is not bool: raise ValueError("Explicit finite evaluation budget required")
    if (reference_report is None) != (random_report is None) or (not fixture and reference_report is None):
        raise ValueError("Both original reference/random records are required")
    config = config or collaborative_study_config()
    actor = NumPyNativeActor(Path(actor_or_path).resolve()) if isinstance(actor_or_path, (str, Path)) else actor_or_path
    metadata, gates, inherited, stage = _contract(actor, protocol, expected, fixture, config)
    scenes = physical._scenarios(scenarios, fixture, config)
    reference = physical._baseline(reference_report, scenes) if reference_report is not None else None
    random = physical._baseline(random_report, scenes) if random_report is not None else None
    sources, weights = execution_sources(), previous._weights_digest(actor)
    stage_fields = {"training_stage": stage, "shutdown_arm": expected["shutdown_arm"], "own_shutdown_beta": expected["own_shutdown_beta"]}
    if stage == "feedback": stage_fields.update(feedback_branch=metadata["feedback_branch"], feedback_lambda=metadata["feedback_lambda"])
    identity = {"version": VERSION, "actor_bindings": deepcopy(expected), "actor_metadata_sha256": digest(metadata),
        "validation_entries_sha256": digest(scenes), "protocol_sha256": digest(protocol), "sources": sources,
        "configuration": asdict(config), "episode_count": len(scenes)*len(PARTNERS), "step_budget": step_budget,
        "reference_sha256": digest(reference) if reference is not None else None,
        "random_sha256": digest(random) if random is not None else None, "test_fixture": fixture,
        "validation_reward": "original_shared_r1", "training_delivery_credit_alpha": .5, "cycle_id": protocol["cycle_id"], **stage_fields}
    contexts = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            context = {"evaluation_version": VERSION, "mode": "observed", "branch": expected["branch"], "cycle_id": protocol["cycle_id"],
                **stage_fields, "partner": partner, "scenario_id": scene["id"], "scenario_index": index,
                "initial_fingerprint": scene["fingerprint"], "episode_index": len(contexts), "seed": 17000+index,
                "horizon": config.horizon, "maximum_environment_steps": config.horizon, "test_fixture": fixture,
                "actor_bindings": deepcopy(expected)}
            context["operation_id"] = digest({"evaluation": digest(identity), "episode": context}); contexts.append(context)
    def unchanged():
        if (execution_sources() != sources or not compact._same(actor.metadata, metadata)
                or previous._weights_digest(actor) != weights or file_hash(actor.path) != expected["actor_sha256"]):
            raise ValueError("Actual evaluation Actor, cached weights or execution sources changed")
    return actor, scenes, config, metadata, gates, inherited, reference, random, identity, contexts, unchanged


def _summary(rows, metadata, expected, gates, reference, random, fixture, identity, reserved, step_budget):
    report = compact._summary(rows, metadata, expected, gates, reference, random, fixture, identity, reserved)
    report.update(version=VERSION, validation_reward="original_shared_r1", training_delivery_credit_alpha=.5,
        remaining_reservable_steps=step_budget-reserved, cycle_id=identity["cycle_id"],
        training_stage=identity["training_stage"], shutdown_arm=identity["shutdown_arm"], training_own_shutdown_beta=identity["own_shutdown_beta"])
    if identity["training_stage"] == "feedback":
        report.update(feedback_branch=identity["feedback_branch"], feedback_lambda=identity["feedback_lambda"])
    return report


def evaluate(actor_or_path, scenarios, protocol, output, step_budget, *, expected_bindings,
             reference_report, random_report, before_episode=None, on_episode=None,
             confirmed_operation_ids=None, allow_test_fixture=False, config=None):
    if any(c is not None and not callable(c) for c in (before_episode, on_episode)): raise ValueError("Callbacks must be callable")
    if confirmed_operation_ids is not None and (not isinstance(confirmed_operation_ids, (set, frozenset, tuple, list))
            or any(type(x) is not str for x in confirmed_operation_ids)): raise ValueError("Confirmed operation IDs required")
    actor, scenes, config, metadata, gates, inherited, ref, random, identity, contexts, unchanged = _material(
        actor_or_path, scenarios, protocol, step_budget, expected_bindings, reference_report, random_report, allow_test_fixture, config)
    rows, manifest = execute_episodes(actor, scenes, config=config, contexts=contexts, identity=identity, output=output,
        step_budget=step_budget, dynamics={"reward": deepcopy(protocol["reward"]), "initialization": {"source_joint_steps": inherited}},
        validate_unchanged=unchanged, before_episode=before_episode, on_episode=on_episode, confirmed_operation_ids=confirmed_operation_ids)
    unchanged()
    report = _summary(rows, metadata, expected_bindings, gates, ref, random, allow_test_fixture, identity,
        manifest["reserved_environment_steps"], step_budget)
    compact._put(Path(output).expanduser().resolve()/"report.json", report)
    return report


def read_completed(actor_or_path, scenarios, protocol, output, step_budget, *, expected_bindings,
                   reference_report, random_report, confirmed_operation_ids, expected_report_sha256,
                   allow_test_fixture=False, config=None):
    """Recompute a complete bound report from acknowledged saved rows; no step/forward."""
    output = Path(output).expanduser().resolve()
    raw = (output/"report.json").read_bytes()
    if sha256(raw).hexdigest() != expected_report_sha256: raise ValueError("External validation report binding differs")
    actor, scenes, config, metadata, gates, inherited, ref, random, identity, contexts, unchanged = _material(
        actor_or_path, scenarios, protocol, step_budget, expected_bindings, reference_report, random_report, allow_test_fixture, config)
    if not isinstance(confirmed_operation_ids, (set, frozenset, list, tuple)): raise ValueError("External confirmed IDs required")
    manifest = json.loads((output/"manifest.json").read_bytes())
    if (manifest.get("version") != VERSION or manifest.get("status") != "completed"
            or not compact._same(manifest.get("identity"), identity) or len(manifest.get("episodes", [])) != len(contexts)):
        raise ValueError("Only a complete original validation matrix may be read")
    rows = []
    for entry, context in zip(manifest["episodes"], contexts):
        if context["operation_id"] not in confirmed_operation_ids: raise ValueError("Saved row lacks independent acknowledgment")
        row = compact._read_entry(output, entry, context)
        verify_saved_trace((output/entry["trace"]["path"]).read_bytes(), context, row)
        rows.append(row)
    reserved = len(contexts)*config.horizon
    if (manifest.get("reserved_environment_steps") != reserved or reserved > step_budget
            or manifest.get("actual_environment_steps") != sum(r["steps"] for r in rows)):
        raise ValueError("Stored finite validation accounting differs")
    report = _summary(rows, metadata, expected_bindings, gates, ref, random, allow_test_fixture, identity, reserved, step_budget)
    if not compact._same(report, json.loads(raw)): raise ValueError("Saved report differs from recomputed rows/gates")
    unchanged()
    if (output/"report.json").read_bytes() != raw: raise ValueError("Report changed while reading")
    return deepcopy(report)
