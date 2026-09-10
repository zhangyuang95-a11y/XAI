"""Stored-record admission and fixed-endpoint comparison for own-shutdown runs.

No checkpoint deserialization, Actor construction/forward, physics or training.
PT files are read only as opaque bytes. Qualifications are recomputed from
hash-bound stored development rows, not reexecuted performance. A future fork
must still invoke the genuine trainer loader before using its complete state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from .warehouse_native_common import ROOT, digest, canonical
from .warehouse_native_cycle_budget import CycleBudget, FILENAME
from .warehouse_native_partner_mix_budget import _bytes as safe_bytes
from . import warehouse_native_shutdown_evaluation as evaluation
from . import warehouse_native_partner_mix_evaluation as compact
from . import warehouse_native_continuation_run as original
from . import warehouse_native_shutdown_run as runner
from .warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import ACTIONS, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION

VERSION = "warehouse-native-own-shutdown-result.v1"
ARMS = ("beta0", "beta1")


def _same(a, b, reason):
    if canonical(a) != canonical(b): raise ValueError(reason)


def _int(value, minimum=0):
    if type(value) is not int or value < minimum: raise ValueError("Invalid recorded integer")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json_bytes(raw):
    def invalid(value): raise ValueError("Nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


class _Inputs:
    def __init__(self):
        self.records = {}
        self.pt_hashes = set()

    def raw(self, root, relative, expected=None, size=None):
        relative = Path(relative)
        if (relative.is_absolute() or any(p in ("", ".", "..") for p in str(relative).split("/"))
                or "\\" in str(relative)):
            raise ValueError("Evidence path must be safe and relative")
        path = Path(root) / relative
        raw = safe_bytes(path)
        actual = sha256(raw).hexdigest()
        if expected is not None and actual != expected: raise ValueError("Bound evidence SHA differs")
        if size is not None and len(raw) != size: raise ValueError("Bound evidence size differs")
        old = self.records.get(str(path))
        binding = {"sha256": actual, "size": len(raw)}
        if old is not None and old != binding: raise ValueError("Input changed while reading")
        self.records[str(path)] = binding
        if path.suffix == ".pt": self.pt_hashes.add(str(path))
        return raw

    def json(self, root, relative, expected=None):
        return _json_bytes(self.raw(root, relative, expected))

    def bound(self, root, binding):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size"}:
            raise ValueError("Complete artifact binding required")
        return self.raw(root, binding["path"], binding["sha256"], binding["size"])

    def unchanged(self):
        for name, binding in self.records.items():
            raw = safe_bytes(Path(name))
            if len(raw) != binding["size"] or sha256(raw).hexdigest() != binding["sha256"]:
                raise ValueError("Input changed during record verification")

    def counts(self):
        return {"checkpoint_byte_hashes": len(self.pt_hashes), "checkpoint_hash_count_scope": "distinct explicitly verified PT paths",
            "checkpoint_decodes": 0, "actor_constructions": 0, "neural_forwards": 0,
            "environment_steps": 0, "optimizer_updates": 0, "final_test_execution": 0}


def _configuration(scenes, fixture):
    value = scenes.get("configuration", {})
    horizon = _int(value.get("horizon"), 1)
    if horizon > 120 or (not fixture and horizon != 120): raise ValueError("Public horizon differs")
    config = collaborative_study_config(horizon=horizon)
    _same(value, asdict(config), "Original public configuration differs")
    return config


def _prepare_record(root, read, fixture):
    if type(fixture) is not bool: raise ValueError("Explicit fixture scope required")
    prepared = read.json(root, "prepared.json")
    if (prepared.get("version") != runner.VERSION or prepared.get("test_fixture") is not fixture
            or prepared.get("arms") != list(ARMS) or prepared.get("branch") != "own_credit"):
        raise ValueError("Shutdown run, arms or fixture scope differs")
    identity = {k: v for k, v in prepared.items() if k not in ("created_unix", "formal_ready", "identity")}
    _same(prepared.get("identity"), identity, "Prepared identity differs")
    if (root / "retired_before_sampling.json").exists(): raise ValueError("Retired preparation cannot supply a source")
    sources = runner.sources()
    _same(prepared.get("runtime_sources"), sources, "Current shutdown execution closure differs")
    for name, expected in sources.items():
        read.raw(ROOT, name, expected)
        read.raw(root / "source_snapshot", name, expected)
    if prepared.get("authorization_record_sha256") != runner.AUTHORIZATION_SHA:
        raise ValueError("Authorization anchor differs")
    read.raw(runner.AUTHORIZATION.parent, runner.AUTHORIZATION.name, runner.AUTHORIZATION_SHA)
    read.raw(root, "authorization.json", runner.AUTHORIZATION_SHA)
    read.raw(root, "proposal.md", prepared["proposal_sha256"])
    read.raw(runner.PROPOSAL.parent, runner.PROPOSAL.name, prepared["proposal_sha256"])
    protocol, scenes = read.json(root, "protocol.json"), read.json(root, "scenarios.json")
    _same(digest(protocol), prepared["protocol_sha256"], "Protocol hash differs")
    _same(digest(scenes), prepared["scenario_manifest_sha256"], "Scenario manifest hash differs")
    if (protocol.get("version") != evaluation.PROTOCOL_VERSION or protocol.get("test_fixture") is not fixture
            or protocol.get("cycle_id") != prepared["cycle_id"] or protocol.get("branch") != "own_credit"
            or protocol.get("shutdown_arms") != list(ARMS) or protocol.get("betas_by_arm") != evaluation.ARMS
            or protocol.get("delivery_credit_alpha") != .5
            or protocol.get("own_shutdown_reward_version") != evaluation.SHUTDOWN_REWARD_VERSION
            or protocol.get("feedback_training_enabled") is not False
            or protocol.get("public_feedback_mode") != "observed"
            or protocol.get("public_feedback_version") != evaluation.physical.OBSERVER_VERSION
            or protocol.get("credit_reward_version") != evaluation.CREDIT_REWARD_VERSION):
        raise ValueError("Actual own-shutdown learning identity differs")
    config = _configuration(scenes, fixture)
    entries = scenes.get("splits", {}).get("validation", [])
    if (not isinstance(entries, list) or not entries or (not fixture and len(entries) != 50)
            or len({x["id"] for x in entries}) != len(entries)
            or len({x["fingerprint"] for x in entries}) != len(entries)):
        raise ValueError("Fixed validation scenes are missing or duplicated")
    for i, scene in enumerate(entries):
        if (scene.get("id") != f"validation_{i:04d}" or scene.get("test_fixture", False) is not fixture
                or not compact._sha(scene.get("fingerprint"))): raise ValueError("Validation scene identity differs")
    cap = _int(prepared.get("primary_endpoint"), 1)
    endpoints = prepared.get("validation_endpoints")
    if (type(endpoints) is not list or not endpoints or endpoints != sorted(set(endpoints))
            or any(type(x) is not int or not 0 < x <= cap for x in endpoints) or endpoints[-1] != cap
            or (not fixture and (cap != 250000 or endpoints != [50000, 250000]))):
        raise ValueError("Registered fixed endpoints differ")
    budget = protocol.get("budget", {})
    _same(budget, {"maximum_ppo_joint_steps_per_arm": cap, "maximum_ppo_joint_steps": cap,
        "requested_total_ppo_joint_steps": 2 * cap, "curriculum_generation_steps": 0}, "Finite PPO contract differs")
    ev = protocol.get("evaluation", {})
    for key, expected in {"checkpoints_ppo_steps": endpoints, "partners": list(evaluation.PARTNERS),
            "scenarios_per_partner": 50, "deterministic": True, "read_final_test": False,
            "reuse_source_zero_step_validation": True, "training_credit_in_evaluation": False,
            "training_shutdown_in_evaluation": False, "horizon": 120,
            "maximum_environment_steps": 2 * len(endpoints) * 3 * 50 * 120}.items():
        _same(ev.get(key), expected, "Fixed evaluation contract differs: " + key)
    maximum = 3 * len(entries) * config.horizon
    _same(prepared["budget_caps"], {a: {"ppo": cap, "evaluation": maximum * len(endpoints)} for a in ARMS},
          "Finite arm caps differ")
    evaluation._gates(protocol)
    _same(protocol.get("reward"), evaluation.physical.REWARD, "Original public reward differs")
    _same(protocol.get("collision_training_cost"), .05, "Original collision cost differs")
    source = protocol["source"]
    if (source.get("checkpoint_sha256") != prepared["source_checkpoint_sha256"]
            or source.get("state_sha256") != prepared["source_state_sha256"]
            or source.get("trainer_version") != evaluation.continuation.TRAINER_VERSION
            or source.get("branch") != "own_credit"
            or digest(protocol.get("source_protocol")) != source.get("protocol_sha256")
            or protocol["source_protocol"].get("version") != evaluation.continuation.PROTOCOL_VERSION
            or (not fixture and (source.get("joint_steps") != 1000000 or source.get("cumulative_joint_steps") != 2230000))):
        raise ValueError("Fixed genuine source lineage differs")
    lineage = protocol.get("source_lineage")
    if not isinstance(lineage, list) or not lineage or lineage[-1] != source: raise ValueError("Source lineage missing")
    previous = None
    for item in lineage:
        for key in ("checkpoint_sha256", "state_sha256", "protocol_sha256", "source_sha256"):
            if not compact._sha(item.get(key)): raise ValueError("Invalid source lineage hash")
        if (item.get("branch") != "own_credit" or item.get("trainer_version") not in
                ("warehouse-native-delivery-credit-trainer.v1", evaluation.continuation.TRAINER_VERSION)
                or _int(item.get("joint_steps")) > _int(item.get("cumulative_joint_steps"))
                or (previous is not None and previous + item["joint_steps"] != item["cumulative_joint_steps"])):
            raise ValueError("Source cumulative clock differs")
        previous = item["cumulative_joint_steps"]
    baselines = read.json(root, "baselines.json", prepared["baselines_sha256"])
    return prepared, protocol, scenes, config, baselines


def _source_record(prepared, protocol, scenes, read, fixture):
    descriptor = prepared["source_descriptor"]
    if descriptor.get("branch") != "own_credit" or descriptor.get("run_version") != original.VERSION:
        raise ValueError("The immediate source must be a genuine Continuation record")
    if not fixture and (Path(descriptor["run"]).resolve() != runner.PRODUCTION_SOURCE.resolve() or descriptor["step"] != 1000000):
        raise ValueError("Production comparison changed the fixed failed source")
    actual, _, old_protocol, old_scenes = original._registered_source(descriptor["run"], "own_credit", descriptor["step"],
        expected=descriptor, allow_test_fixture=fixture)
    _same(actual, descriptor, "Frozen source descriptor changed")
    _same(old_protocol, protocol["source_protocol"], "Immediate source protocol differs")
    _same(old_scenes, scenes, "Inherited scenes changed")
    for key in ("training", "reward", "collision_training_cost", "seed", "partners_by_branch"):
        _same(protocol.get(key), old_protocol.get(key), "Unchanged source learning field differs: " + key)
        if key not in protocol or key not in old_protocol: raise ValueError("Inherited learning field missing: " + key)
    _same(prepared["baselines_sha256"], descriptor["baselines_sha256"], "Source baselines changed")
    _same(protocol["source"]["joint_steps"], descriptor["step"], "Source step differs from its acknowledgment")
    report = read.json(Path(descriptor["run"]), Path("branches/own_credit/validation") /
        f"step_{descriptor['step']:07d}" / "report.json", descriptor["validation_report_sha256"])
    _same(protocol["source"]["cumulative_joint_steps"], report["total_actor_training_joint_steps"], "Source cumulative clock differs from original report")
    _same(protocol["source"]["source_sha256"], report["actor_bindings"]["source_sha256"], "Source execution hash differs")
    if not fixture and (report["capability"]["eligible"] or report["warmup_capability"]["eligible"]):
        raise ValueError("Registered fixed source no longer matches the failed-source repair trigger")
    read.raw(Path(descriptor["run"]), descriptor["checkpoint"]["path"], descriptor["checkpoint"]["sha256"])


@contextmanager
def _context(output, fixture):
    root = Path(output).expanduser().absolute()
    if root.is_symlink(): raise ValueError("Run root cannot be an alias")
    root = root.resolve()
    read = _Inputs()
    prepared, protocol, scenes, config, baselines = _prepare_record(root, read, fixture)
    _source_record(prepared, protocol, scenes, read, fixture)
    ledger = CycleBudget(root, prepared["identity"])
    with ledger.lease():
        account = ledger.read()
        if any(op["status"] != "acknowledged" for op in account["operations"].values()):
            raise ValueError("Pending or abandoned sampling prevents source admission")
        read.raw(root, FILENAME)
        for cells in account["branches"].values():
            for cell in cells.values():
                for key in ("initial_head", "head"):
                    item = cell[key]
                    if item: read.raw(root, item["path"], item["sha256"])
        yield root, read, prepared, protocol, scenes, config, baselines, account
        read.unchanged()


def _actor(read, root, relative, bindings, protocol, prepared, config, fixture):
    raw = read.raw(root, relative, bindings["actor_sha256"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        shapes = {"0.weight": (128, 197), "0.bias": (128,), "2.weight": (128, 128),
            "2.bias": (128,), "4.weight": (5, 128), "4.bias": (5,)}
        if set(data.files) != {"metadata_json", *shapes} or len(data.files) != 7: raise ValueError("Actor tensor schema differs")
        for key, shape in shapes.items():
            if data[key].dtype != np.float32 or data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError("Actor array shape or finite float32 contract differs")
        metadata = _json_bytes(str(data["metadata_json"].item()))
    required = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": 197, "state_dim": 354, "hidden": 128, "architecture": "two_hidden_layer_tanh",
        "actions": list(ACTIONS), "action_masks": False, "runtime_action_override": False,
        "feature_names": list(observation_names(config)) + list(HISTORY_FEATURE_NAMES),
        "public_feedback_version": evaluation.physical.OBSERVER_VERSION, "public_feedback_mode": "observed",
        "credit_reward_version": evaluation.CREDIT_REWARD_VERSION, "delivery_credit_alpha": .5,
        "own_shutdown_reward_version": evaluation.SHUTDOWN_REWARD_VERSION, "test_fixture": fixture,
        "source_lineage": protocol["source_lineage"], "source_sha256": digest(_trainer_sources()),
    }
    for key, value in required.items(): _same(metadata.get(key), value, "Actual Actor contract differs: " + key)
    for key, value in bindings.items():
        if key != "actor_sha256": _same(metadata.get(key), value, "Actor metadata binding differs: " + key)
    if metadata.get("source_counters", {}).get("joint_steps") != protocol["source"]["cumulative_joint_steps"]:
        raise ValueError("Actor inherited counter differs")
    return metadata


def _trainer_sources():
    from .warehouse_native_shutdown_trainer import execution_sources
    return execution_sources()


def _trace(raw, context, row):
    frames = [_json_bytes(line) for line in gzip.decompress(raw).splitlines() if line]
    if len(frames) != row["steps"]: raise ValueError("Trace length differs from confirmed row")
    physical = []
    previous = None
    for index, record in enumerate(frames):
        for key, value in context.items(): _same(record.get(key), value, "Trace episode binding differs")
        if (record.get("frame") != index or record.get("actual_steps_completed") != index
                or record.get("nn_action_overrides") != 0 or record["before"]["frame"] != index
                or record["after"]["frame"] != index + 1): raise ValueError("Trace frame or NN audit differs")
        if previous is not None: _same(record["before"], previous, "Trace chain is discontinuous")
        previous = record["after"]
        for key in ("policy_actions", "submitted_actions", "executed_actions"):
            if set(record[key]) != {"robot_1", "robot_2"} or any(a not in ACTIONS for a in record[key].values()):
                raise ValueError("Recorded action alphabet differs")
        if (record["policy_actions"]["robot_2"] != record["submitted_actions"]["robot_2"]
                or record.get("program_action") != record["submitted_actions"]["robot_1"]):
            raise ValueError("Neural proposal was overridden or program action misattributed")
        for role in ("robot_1", "robot_2"):
            probabilities = np.asarray(record["action_distributions"][role])
            observation = np.asarray(record["actor_observations"][role])
            if (probabilities.shape != (5,) or not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or not np.isclose(probabilities.sum(), 1., atol=1e-5)
                    or observation.shape != (197,) or not np.isfinite(observation).all()
                    or ACTIONS[int(probabilities.argmax())] != record["policy_actions"][role]):
                raise ValueError("Stored deterministic neural distribution differs")
        physical.append({"before": record["before"], "after": record["after"],
            "requested_actions": record["submitted_actions"], "executed_actions": record["executed_actions"], "events": record["events"]})
    if row.get("physical_trajectory_keys") != list(evaluation.physical.PHYSICAL_TRAJECTORY_KEYS):
        raise ValueError("Physical trajectory contract differs")
    _same(row.get("physical_trajectory_sha256"), digest(physical), "Physical trajectory hash differs")


def _endpoint(context, arm, step, fixture):
    root, read, prepared, protocol, scenes, config, baselines, account = context
    if arm not in ARMS or type(step) is not int or step not in prepared["validation_endpoints"]:
        raise ValueError("Unknown arm or unregistered source endpoint")
    candidates = [(opid, op) for opid, op in account["operations"].items() if op["request"]["kind"] == "ppo"
        and op["request"]["branch"] == arm and op["completion"]["checkpoint"]["step"] == step]
    if len(candidates) != 1: raise ValueError("Source needs exactly one acknowledged endpoint checkpoint")
    opid, op = candidates[0]; checkpoint = op["completion"]["checkpoint"]
    prefix = Path("branches") / arm
    if not Path(checkpoint["path"]).is_relative_to(prefix / "checkpoints") or op["completion"]["actual_steps"] != op["request"]["steps"]:
        raise ValueError("Checkpoint or PPO amount differs")
    read.raw(root, checkpoint["path"], checkpoint["sha256"])
    folder = prefix / "validation" / f"step_{step:07d}"
    report, manifest = read.json(root, folder / "report.json"), read.json(root, folder / "manifest.json")
    n = len(scenes["splits"]["validation"]); maximum = 3 * n * config.horizon
    if (report.get("version") != evaluation.VERSION or report.get("status") != "completed"
            or report.get("test_fixture") is not fixture or report.get("branch") != "own_credit"
            or report.get("shutdown_arm") != arm or report.get("additional_joint_steps") != step
            or manifest.get("version") != evaluation.VERSION or manifest.get("status") != "completed"
            or len(report.get("rows", [])) != 3 * n or len(manifest.get("episodes", [])) != 3 * n):
        raise ValueError("Fixed endpoint validation is incomplete or belongs to another arm")
    bindings = report["actor_bindings"]
    if set(bindings) != set(evaluation.REQUIRED_BINDINGS): raise ValueError("Complete exact Actor bindings required")
    wanted = {"experiment_version": evaluation.TRAINER_VERSION, "branch": "own_credit", "shutdown_arm": arm,
        "own_shutdown_beta": evaluation.ARMS[arm], "joint_steps": step, "cycle_id": prepared["cycle_id"],
        "protocol_sha256": prepared["protocol_sha256"], "scenario_manifest_sha256": prepared["scenario_manifest_sha256"],
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"], "initialization_sha256": prepared["source_state_sha256"]}
    for key, value in wanted.items(): _same(bindings.get(key), value, "Endpoint source binding differs: " + key)
    for key in evaluation.REQUIRED_BINDINGS:
        if key.endswith("sha256") and not compact._sha(bindings[key]): raise ValueError("Invalid Actor hash")
    actor_relative = prefix / "actors" / f"actor_{step:07d}.npz"
    metadata = _actor(read, root, actor_relative, bindings, protocol, prepared, config, fixture)
    identity = {"version": evaluation.VERSION, "actor_bindings": deepcopy(bindings), "actor_metadata_sha256": digest(metadata),
        "validation_entries_sha256": digest(scenes["splits"]["validation"]), "protocol_sha256": digest(protocol),
        "sources": evaluation.execution_sources(), "configuration": asdict(config), "episode_count": 3 * n,
        "step_budget": maximum, "reference_sha256": digest(baselines["reference"]), "random_sha256": digest(baselines["random"]),
        "test_fixture": fixture, "validation_reward": "original_shared_r1", "training_delivery_credit_alpha": .5,
        "cycle_id": protocol["cycle_id"], "shutdown_arm": arm, "training_own_shutdown_beta": evaluation.ARMS[arm]}
    _same(report["identity"], identity, "Validation identity differs")
    _same(manifest["identity"], identity, "Validation manifest identity differs")
    total = 0
    for index, (stored, row) in enumerate(zip(manifest["episodes"], report["rows"])):
        partner, si = evaluation.PARTNERS[index // n], index % n
        scene = scenes["splits"]["validation"][si]
        expected_context = {"evaluation_version": evaluation.VERSION, "mode": "observed", "branch": "own_credit",
            "cycle_id": protocol["cycle_id"], "shutdown_arm": arm, "own_shutdown_beta": evaluation.ARMS[arm],
            "partner": partner, "scenario_id": scene["id"], "scenario_index": si, "initial_fingerprint": scene["fingerprint"],
            "episode_index": index, "seed": 17000 + si, "horizon": config.horizon,
            "maximum_environment_steps": config.horizon, "test_fixture": fixture, "actor_bindings": deepcopy(bindings)}
        expected_context["operation_id"] = digest({"evaluation": digest(identity), "episode": expected_context})
        _same(stored.get("context"), expected_context, "Registered episode context differs")
        for key, value in expected_context.items(): _same(row.get(key), value, "Row context differs")
        amount = _int(row.get("steps"), 1)
        if (stored.get("status") != "completed" or stored.get("actual_steps") != amount or amount > config.horizon
                or row.get("neural_role") != 1 or row.get("nn_action_overrides") != 0
                or row.get("raw_neural_actions_submitted") != amount): raise ValueError("Saved row action or completion audit differs")
        operation = account["operations"].get(expected_context["operation_id"])
        if (not operation or operation["status"] != "acknowledged" or operation["request"]["kind"] != "evaluation"
                or operation["request"]["branch"] != arm or operation["request"]["steps"] != config.horizon
                or operation["completion"]["actual_steps"] != amount): raise ValueError("Episode lacks its external acknowledgment")
        for kind in ("row", "trace"):
            if Path(stored[kind]["path"]).name != stored[kind]["path"]: raise ValueError("Episode filename escapes its folder")
        row_raw = read.bound(root / folder, stored["row"])
        _same(_json_bytes(row_raw), row, "Original raw row differs from report")
        receipt = operation["completion"]["checkpoint"]
        if receipt["path"] != str(folder / stored["row"]["path"]) or receipt["sha256"] != stored["row"]["sha256"]:
            raise ValueError("Episode confirmation binds another file")
        _trace(read.bound(root / folder, stored["trace"]), expected_context, row)
        total += amount
    if manifest["actual_environment_steps"] != total or manifest["reserved_environment_steps"] != maximum:
        raise ValueError("Actual/reserved validation steps differ")
    rebuilt = compact._summary(report["rows"], metadata, bindings, evaluation._gates(protocol), baselines["reference"],
        baselines["random"], fixture, identity, maximum)
    rebuilt.update(version=evaluation.VERSION, validation_reward="original_shared_r1", training_delivery_credit_alpha=.5,
        remaining_reservable_steps=0, cycle_id=protocol["cycle_id"], shutdown_arm=arm, training_own_shutdown_beta=evaluation.ARMS[arm])
    _same(report, rebuilt, "Original report differs from recomputed statistics and gates")
    parity_relative = prefix / "actors" / f"actor_{step:07d}.json"
    parity = read.json(root, parity_relative)
    if (parity.get("checkpoint_sha256") != checkpoint["sha256"] or parity.get("argmax_equal") is not True
            or type(parity.get("maximum_absolute_error")) not in (int, float)
            or not 0 <= parity["maximum_absolute_error"] <= 1e-4): raise ValueError("Checkpoint export parity receipt differs")
    _same(parity.get("actor"), {"path": str(actor_relative), "sha256": bindings["actor_sha256"],
        "size": read.records[str(root / actor_relative)]["size"]}, "Parity Actor binding differs")
    descriptor = {"run": str(root), "run_version": runner.VERSION, "branch": "own_credit", "shutdown_arm": arm,
        "own_shutdown_beta": evaluation.ARMS[arm], "step": step, "cycle_id": prepared["cycle_id"],
        "prepared_sha256": read.records[str(root / "prepared.json")]["sha256"], "protocol_sha256": digest(protocol),
        "scenario_manifest_sha256": digest(scenes), "baselines_sha256": prepared["baselines_sha256"],
        "checkpoint": deepcopy(checkpoint), "operation_id": opid, "operation_sha256": digest(op),
        "validation_report_sha256": read.records[str(root / folder / "report.json")]["sha256"],
        "validation_manifest_sha256": read.records[str(root / folder / "manifest.json")]["sha256"],
        "actor_sha256": bindings["actor_sha256"], "parity_sha256": read.records[str(root / parity_relative)]["sha256"],
        "test_fixture": fixture, "checkpoint_bytes_verified": True, "checkpoint_payload_verification_required_before_fork": True,
        "capability_eligible": report["capability"]["eligible"], "warmup_eligible": report["warmup_capability"]["eligible"],
        "next_stage_capability_eligible": bool(not fixture and (report["capability"]["eligible"] or report["warmup_capability"]["eligible"])),
        "release_ready": False}
    return descriptor, report


def registered_source(output, shutdown_arm, step, allow_test_fixture=False):
    """Strict stored source receipt. Failure of capability does not erase provenance."""
    with _context(output, allow_test_fixture) as context:
        descriptor, _ = _endpoint(context, shutdown_arm, step, allow_test_fixture)
        descriptor["inspection_counts"] = context[1].counts()
        return deepcopy(descriptor), deepcopy(context[2]), deepcopy(context[3]), deepcopy(context[4])


def select_endpoint(reports, *, test_fixture=False):
    """Pure predeclared ranking; this helper never grants source authenticity."""
    eligible = [arm for arm in ARMS if reports[arm]["capability"]["eligible"] or reports[arm]["warmup_capability"]["eligible"]]
    if test_fixture: eligible = []
    selected = max(eligible, key=lambda a: (reports[a]["capability"]["eligible"], reports[a]["primary_value"], a == "beta0")) if eligible else None
    return {"selected_arm": selected, "next_stage_capability_eligible": selected is not None,
        "rule": "warmup_or_full_then_full_priority_then_equal_partner_nn_deliveries_tie_beta0",
        "scope": "development_endpoint_choice_not_final_test_or_explanation_qualification", "release_ready": False}


def _comparison(reports):
    left, right = reports["beta0"], reports["beta1"]
    partners = {}
    for partner in evaluation.PARTNERS:
        a, b = left["summary"][partner], right["summary"][partner]
        partners[partner] = {"beta0_nn_deliveries": a["mean_ai_deliveries"], "beta1_nn_deliveries": b["mean_ai_deliveries"],
            "nn_delivery_difference": b["mean_ai_deliveries"] - a["mean_ai_deliveries"],
            "beta0_team_deliveries": a["mean_team_deliveries"], "beta1_team_deliveries": b["mean_team_deliveries"],
            "team_delivery_difference": b["mean_team_deliveries"] - a["mean_team_deliveries"],
            "beta0_survival": a["ai_active_end_rate"], "beta1_survival": b["ai_active_end_rate"],
            "survival_difference": b["ai_active_end_rate"] - a["ai_active_end_rate"],
            "beta0_collisions_per_step": a["collisions_per_step"], "beta1_collisions_per_step": b["collisions_per_step"]}
    differences = []
    for a, b in zip(left["rows"], right["rows"]):
        for key in ("partner", "scenario_id", "scenario_index", "initial_fingerprint", "seed"):
            _same(a[key], b[key], "Paired episodes use different actual scenarios")
        differences.append({**{k: a[k] for k in ("partner", "scenario_id", "initial_fingerprint", "seed")},
            "nn_delivery_difference": b["ai_deliveries"] - a["ai_deliveries"],
            "team_delivery_difference": b["team_deliveries"] - a["team_deliveries"],
            "survival_difference": int(b["ai_active_end"]) - int(a["ai_active_end"]),
            "collisions_difference": b["collisions"] - a["collisions"], "native_score_difference": b["native_score"] - a["native_score"]})
    return {"beta0_primary": left["primary_value"], "beta1_primary": right["primary_value"],
        "beta1_minus_beta0": right["primary_value"] - left["primary_value"], "partners": partners, "paired_rows": differences}


def compare_fixed(output, *, allow_test_fixture=False):
    with _context(output, allow_test_fixture) as context:
        root, read, prepared, protocol, scenes, config, baselines, account = context
        reports, descriptors, missing = {}, {}, []
        for step in prepared["validation_endpoints"]:
            reports[step], descriptors[step] = {}, {}
            for arm in ARMS:
                path = root / "branches" / arm / "validation" / f"step_{step:07d}" / "report.json"
                checkpoints = [op for op in account["operations"].values() if op["request"]["kind"] == "ppo"
                    and op["request"]["branch"] == arm and op["completion"]["checkpoint"]["step"] == step]
                if not path.exists() and not checkpoints:
                    missing.append({"shutdown_arm": arm, "step": step}); continue
                if not path.exists():
                    missing.append({"shutdown_arm": arm, "step": step, "reason": "validation_not_complete"}); continue
                descriptors[step][arm], reports[step][arm] = _endpoint(context, arm, step, allow_test_fixture)
        complete = not missing
        if complete:
            for arm in ARMS:
                for kind in ("ppo", "evaluation"):
                    cell = account["branches"][arm][kind]
                    if cell["remaining"] != 0 or cell["reserved"] != cell["cap"] or (kind == "ppo" and cell["acknowledged"] != cell["cap"]):
                        raise ValueError("Both complete arms must exhaust their registered equal budgets")
            actual = sum(r["environment_steps"] for arms in reports.values() for r in arms.values())
            if actual != account["totals"]["evaluation"]["acknowledged"]: raise ValueError("Reports omit or duplicate acknowledged evaluation steps")
        main = prepared["primary_endpoint"]
        return {"version": VERSION, "status": "fixed_endpoint_complete" if complete else "incomplete",
            "cycle_id": prepared["cycle_id"], "test_fixture": allow_test_fixture, "primary_endpoint": main,
            "primary": _comparison(reports[main]) if complete else None,
            "descriptive_endpoints": {str(s): _comparison(v) for s, v in reports.items() if s != main and set(v) == set(ARMS)},
            "selection": select_endpoint(reports[main], test_fixture=allow_test_fixture) if complete else None,
            "missing_endpoints": missing, "source_descriptors": descriptors,
            "qualification": {a: {"capability": reports[main][a]["capability"], "warmup_capability": reports[main][a]["warmup_capability"]} for a in reports[main]},
            "budget": deepcopy(account["totals"]), "input_ledger_sha256": read.records[str(root / FILENAME)]["sha256"],
            "inspection_counts": read.counts(), "result_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "stored_row_recomputation_and_bound_trace_checks_not_physics_neural_or_gradient_reexecution",
            "release_ready": False, "formal_ready": False, "website_model_changed": False}


def run(output, summary_output, *, allow_test_fixture=False):
    source, target = Path(output).resolve(), Path(summary_output).expanduser().absolute()
    if target.exists(): raise FileExistsError("Keep earlier summaries; choose a new output directory")
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Summary output must be separate from the source run")
    result = compare_fixed(source, allow_test_fixture=allow_test_fixture)
    target.mkdir(parents=True, exist_ok=False)
    runner.write_json(target / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--allow-test-fixture", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.run, args.output, allow_test_fixture=args.allow_test_fixture), sort_keys=True))


if __name__ == "__main__": main()
