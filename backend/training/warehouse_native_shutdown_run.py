"""Fixed-endpoint own-shutdown reward comparison from one genuine learner.

New rewards are learned, never used as an action override. Two independently
resumable arms retain the same initial Actor, Critic, Adam, environments and RNG.
This runner neither grants explanation qualification nor changes a web model.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_partner_mix_run import (
    write_bytes, write_json, decode, save_batch, export_actor, AUTHORIZATION, AUTHORIZATION_SHA,
)
from .warehouse_native_cycle_budget import CycleBudget
from . import warehouse_native_continuation_run as original

VERSION = "warehouse-native-own-shutdown-run.v1"
ARMS = ("beta0", "beta1")
PROPOSAL = ROOT / "docs/warehouse_native_own_shutdown_round.md"
PRODUCTION_SOURCE = ROOT / "output/warehouse_native/continuation_03_own_20260909"


def _json(path):
    return json.loads(Path(path).read_bytes())


def _same(a, b, reason):
    if digest(a) != digest(b):
        raise ValueError(reason)


def sources():
    from .warehouse_native_shutdown_trainer import execution_sources
    from .warehouse_native_shutdown_evaluation import execution_sources as evaluation_sources
    result = original.sources()
    result.update(execution_sources())
    result.update(evaluation_sources())
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _source(source_run, source_step, device, fixture, expected=None):
    source_run = Path(source_run).expanduser().resolve()
    if not fixture and (source_run != PRODUCTION_SOURCE.resolve() or source_step != 1000000):
        raise ValueError("This comparison retains the fixed failed continuation03 million-step endpoint")
    source, descriptor, scenes = original.load_source(source_run, "own_credit", source_step,
        device=device, expected_descriptor=expected, allow_test_fixture=fixture)
    if not fixture:
        report = _json(source_run / "branches/own_credit/validation/step_1000000/report.json")
        if report["capability"]["eligible"] or report["warmup_capability"]["eligible"]:
            raise ValueError("The fixed source already qualifies; the registered repair trigger is absent")
        if source.source_counters["joint_steps"] + source.joint_steps != 2230000:
            raise ValueError("Source cumulative learning clock differs")
    return source, descriptor, scenes


def _trainer(prepared, protocol, source, arm):
    from .warehouse_native_shutdown_trainer import OwnShutdownTrainer
    return OwnShutdownTrainer(protocol, source, shutdown_arm=arm,
        expected_source_state_sha256=prepared["source_state_sha256"],
        source_checkpoint_sha256=prepared["source_checkpoint_sha256"],
        device=prepared["device"], test_fixture=prepared["test_fixture"])


def prepare(output, *, source_run=PRODUCTION_SOURCE, source_step=1000000, device="mps",
            cycle_id=None, allow_test_fixture=False, ppo_cap=250000, evaluation_checkpoints=None):
    from .warehouse_native_shutdown_trainer import make_protocol
    output = Path(output).expanduser().resolve()
    source_run = Path(source_run).expanduser().resolve()
    if output.exists():
        raise FileExistsError("A reward comparison requires a new directory")
    if output == source_run or output in source_run.parents or source_run in output.parents:
        raise ValueError("The new ledger must be separate from its source run")
    if type(allow_test_fixture) is not bool or device not in ("cpu", "mps"):
        raise ValueError("Explicit fixture scope and device required")
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError("Local autonomous training authorization changed")
    source, descriptor, scenes = _source(source_run, source_step, device, allow_test_fixture)
    state = source.state_dict()
    state_sha = initialization_sha256(state)
    cycle_id = cycle_id or output.name
    endpoints = [50000, 250000] if evaluation_checkpoints is None else evaluation_checkpoints
    protocol = make_protocol(source, source_checkpoint_sha256=descriptor["checkpoint"]["sha256"],
        source_state_sha256=state_sha, cycle_id=cycle_id, test_fixture=allow_test_fixture,
        ppo_cap=ppo_cap, evaluation_checkpoints=endpoints)
    count = len(source.envs)
    probe = min(4096, ppo_cap)
    probe -= probe % count
    if probe < count:
        raise ValueError("Probe must contain a full environment batch")
    maximum = len(endpoints) * 3 * len(scenes["splits"]["validation"]) * scenes["configuration"]["horizon"]
    runtime_sources = sources()
    prepared = {"version": VERSION, "cycle_id": cycle_id, "device": device,
        "branch": "own_credit", "arms": list(ARMS), "source_descriptor": descriptor,
        "source_checkpoint_sha256": descriptor["checkpoint"]["sha256"], "source_state_sha256": state_sha,
        "protocol_sha256": digest(protocol), "scenario_manifest_sha256": digest(scenes),
        "baselines_sha256": file_hash(source_run / "baselines.json"), "runtime_sources": runtime_sources,
        "authorization_record_sha256": AUTHORIZATION_SHA, "proposal_sha256": file_hash(PROPOSAL),
        "budget_caps": {arm: {"ppo": ppo_cap, "evaluation": maximum} for arm in ARMS},
        "primary_endpoint": ppo_cap, "validation_endpoints": list(endpoints), "probe_endpoint": probe,
        "test_fixture": allow_test_fixture, "created_unix": time.time(), "formal_ready": False,
        "selection": "fixed_endpoints_only_full_gate_then_warmup_then_neural_deliveries_tie_beta0"}
    prepared["identity"] = {k: copy.deepcopy(v) for k, v in prepared.items()
        if k not in ("created_unix", "formal_ready")}
    # Validate both real forks before persisting a production ledger.
    matched = {}
    for arm in ARMS:
        learner = _trainer(prepared, protocol, source, arm)
        saved = learner.state_dict()
        fields = ("model", "optimizers", "rng", "python_rng", "numpy_rng", "torch_rng",
            "partner_kinds", "program_roles", "scenario_ids", "episode_context",
            "episode_returns", "episode_reward_components")
        if device == "mps":
            fields = (*fields, "mps_rng")
        matched[arm] = {key: initialization_sha256(saved[key]) for key in fields}
        for key, value in matched[arm].items():
            _same(value, initialization_sha256(state[key]), "Reward fork changed " + key)
        for before, after in zip(state["envs"], saved["envs"]):
            _same(before, {k: v for k, v in after.items()
                if k not in ("own_shutdown_reward_version", "own_shutdown_beta")},
                "Reward fork changed physical state, history or RNG")
        del learner
    _same(matched["beta0"], matched["beta1"], "Paired initial learning states differ")
    if sources() != runtime_sources:
        raise ValueError("Execution sources changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol.json", protocol), ("prepared.json", prepared), ("scenarios.json", scenes)):
        write_json(output / name, value)
    write_bytes(output / "baselines.json", (source_run / "baselines.json").read_bytes())
    write_bytes(output / "authorization.json", AUTHORIZATION.read_bytes())
    write_bytes(output / "proposal.md", PROPOSAL.read_bytes())
    for name in runtime_sources:
        write_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
    ledger = CycleBudget.create(output, prepared["identity"])
    with ledger.lease():
        for arm in ARMS:
            learner = _trainer(prepared, protocol, source, arm)
            checkpoint = output / "branches" / arm / "checkpoints/initial.pt"
            atomic_torch_save(checkpoint, {"version": VERSION, "cycle_id": cycle_id,
                "branch": "own_credit", "shutdown_arm": arm, "operation_id": None,
                "trainer": learner.state_dict(), "new_environment_steps": 0})
            ledger.initialize_head("ppo", arm, str(checkpoint.relative_to(output)), file_hash(checkpoint))
            reference = output / "branches" / arm / "initial_evaluation_reference.json"
            write_json(reference, {"source_descriptor": descriptor, "new_environment_steps": 0})
            ledger.initialize_head("evaluation", arm, str(reference.relative_to(output)), file_hash(reference))
            del learner
    write_json(output / "initialization_check.json", {"identical_learning_state": True, "matched": matched,
        "source_state_sha256": state_sha, "new_environment_steps": 0})
    return {"status": "prepared", "output": str(output), "cycle_id": cycle_id,
        "per_arm_ppo_cap": ppo_cap, "validation_endpoints": endpoints, "probe_endpoint": probe,
        "total_auxiliary_cap": 2 * maximum, "formal_ready": False}


def _check_prepared(output, prepared, fixture):
    if (prepared["version"] != VERSION or prepared.get("test_fixture") is not fixture
            or prepared["arms"] != list(ARMS) or prepared["branch"] != "own_credit"
            or prepared["runtime_sources"] != sources()):
        raise ValueError("Reward comparison identity or execution sources changed")
    for key, value in prepared["identity"].items():
        _same(prepared.get(key), value, "Prepared identity differs: " + key)
    if prepared["authorization_record_sha256"] != AUTHORIZATION_SHA or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError("Local authorization changed")
    for name, key in (("authorization.json", "authorization_record_sha256"),
            ("baselines.json", "baselines_sha256"), ("proposal.md", "proposal_sha256")):
        if file_hash(output / name) != prepared[key]:
            raise ValueError("Bound comparison file changed: " + name)
    for name, sha in prepared["runtime_sources"].items():
        if file_hash(output / "source_snapshot" / name) != sha:
            raise ValueError("Archived execution source changed: " + name)


def read_prepared(output, *, allow_test_fixture=False):
    output = Path(output).expanduser().resolve()
    prepared = _json(output / "prepared.json")
    _check_prepared(output, prepared, allow_test_fixture)
    protocol, scenes = _json(output / "protocol.json"), _json(output / "scenarios.json")
    _same(digest(protocol), prepared["protocol_sha256"], "Frozen protocol changed")
    _same(digest(scenes), prepared["scenario_manifest_sha256"], "Frozen scene manifest changed")
    descriptor = prepared["source_descriptor"]
    source, actual, original_scenes = _source(descriptor["run"], descriptor["step"], prepared["device"],
        allow_test_fixture, expected=descriptor)
    _same(actual, descriptor, "Restored source descriptor changed")
    _same(initialization_sha256(source.state_dict()), prepared["source_state_sha256"], "Full source state differs")
    _same(scenes, original_scenes, "Reward comparison cannot alter scenarios")
    n = 3 * len(scenes["splits"]["validation"]) * scenes["configuration"]["horizon"]
    expected_caps = {arm: {"ppo": protocol["budget"]["maximum_ppo_joint_steps_per_arm"],
        "evaluation": n * len(prepared["validation_endpoints"])} for arm in ARMS}
    _same(prepared["budget_caps"], expected_caps, "Finite comparison caps differ")
    _same(protocol["evaluation"]["checkpoints_ppo_steps"], prepared["validation_endpoints"], "Fixed endpoints differ")
    initial = _json(output / "initialization_check.json")
    if initial["identical_learning_state"] is not True or initial["source_state_sha256"] != prepared["source_state_sha256"]:
        raise ValueError("Initial fork verification is incomplete")
    return prepared, protocol, scenes, source


def _envelope(payload, trainer, arm):
    if (payload.get("version") != VERSION or payload.get("branch") != "own_credit"
            or payload.get("shutdown_arm") != arm or payload.get("cycle_id") != trainer.protocol["cycle_id"]):
        raise ValueError("Checkpoint belongs to another reward arm or runtime")


def train_to(output, trainer, ledger, arm, target):
    if arm not in ARMS or trainer.shutdown_arm != arm:
        raise ValueError("Learner and ledger arm differ")
    n = trainer.cfg["environments"]
    while trainer.joint_steps < target:
        if any(ledger.read()["branches"][arm][kind]["pending"] for kind in ("ppo", "evaluation")):
            raise ValueError("Unconfirmed operation requires diagnosis; never resample")
        amount = min(n * trainer.cfg["rollout_steps"], target - trainer.joint_steps, ledger.remaining("ppo", arm))
        amount -= amount % n
        if amount <= 0:
            raise ValueError("Finite budget cannot reach this endpoint")
        head = ledger.head("ppo", arm)
        if trainer.joint_steps != head["step"]:
            raise ValueError("Learner clock differs from acknowledged head")
        before = initialization_sha256(trainer.state_dict())
        old = decode(output / head["path"], head["sha256"])
        _envelope(old, trainer, arm)
        _same(initialization_sha256(old["trainer"]), before, "Learner differs from confirmed predecessor")
        operation_id = "ppo_" + uuid.uuid4().hex
        reservation = ledger.reserve("ppo", arm, amount, operation_id, expected_step=trainer.joint_steps)
        if not reservation["execution_permitted"]:
            raise ValueError("Repeated reservation is not execution permission")
        started = time.monotonic()
        batch, metrics = trainer.train_chunk(amount // n)
        if (trainer.joint_steps != head["step"] + amount or len(batch["transition_records"]) != amount
                or batch["audit"]["neural_overrides"] != 0
                or any(row.get("shutdown_arm") != arm for row in batch["transition_records"])
                or any(not np.isfinite(v) for v in metrics.values())):
            raise ValueError("Actual reward training differs from its reservation or NN contract")
        evidence = save_batch(output, arm, operation_id, batch)
        episodes = copy.deepcopy(trainer.completed_episodes)
        trainer.completed_episodes.clear()
        checkpoint = output / "branches" / arm / "checkpoints" / f"step_{trainer.joint_steps:07d}_{operation_id}.pt"
        payload = {"version": VERSION, "cycle_id": trainer.protocol["cycle_id"], "branch": "own_credit",
            "shutdown_arm": arm, "operation_id": operation_id, "trainer": trainer.state_dict(),
            "audit": batch["audit"], "metrics": metrics, "evidence": evidence, "actual_steps": amount,
            "completed_episodes": episodes, "before_state_sha256": before,
            "elapsed_seconds": time.monotonic() - started}
        atomic_torch_save(checkpoint, payload)
        ledger.ack(operation_id, str(checkpoint.relative_to(output)), file_hash(checkpoint), amount)
        write_json(output / "progress.json", {"status": "training", "shutdown_arm": arm,
            "branch_ppo_steps": trainer.joint_steps, "target": target, "last_metrics": metrics,
            "last_chunk_seconds": payload["elapsed_seconds"], "ledger": ledger.read()}, replace=True)
        print(json.dumps({"event": "ppo_ack", "shutdown_arm": arm, "steps": trainer.joint_steps,
            "seconds": round(payload["elapsed_seconds"], 3), "neural_overrides": 0}), flush=True)


def evaluate_boundary(output, trainer, ledger, arm, scenes):
    from .warehouse_native_shutdown_evaluation import evaluate, REQUIRED_BINDINGS
    actor = export_actor(output, trainer, arm, ledger.head("ppo", arm))
    bindings = {key: actor.metadata[key] for key in REQUIRED_BINDINGS if key != "actor_sha256"}
    bindings["actor_sha256"] = actor.artifact_sha256
    baselines = _json(output / "baselines.json")
    confirmed = {opid for opid, op in ledger.read()["operations"].items()
        if op["status"] == "acknowledged" and op["request"]["kind"] == "evaluation"
        and op["request"]["branch"] == arm}
    def reserve(context):
        operation = ledger.reserve("evaluation", arm, context["horizon"], context["operation_id"])
        if not operation["execution_permitted"]:
            raise ValueError("Evaluation was already reserved")
    def acknowledge(row):
        ledger.ack(row["operation_id"], str(Path(row["row_path"]).relative_to(output)),
            row["row_sha256"], row["actual_steps"])
    maximum = 3 * len(scenes["splits"]["validation"]) * scenes["configuration"]["horizon"]
    return evaluate(actor, scenes["splits"]["validation"], trainer.protocol,
        output / "branches" / arm / "validation" / f"step_{trainer.joint_steps:07d}", maximum,
        expected_bindings=bindings, reference_report=baselines["reference"], random_report=baselines["random"],
        before_episode=reserve, on_episode=acknowledge, confirmed_operation_ids=confirmed,
        allow_test_fixture=trainer.test_fixture)


def advance(output, *, until=None, allow_test_fixture=False):
    output = Path(output).expanduser().resolve()
    raw = _json(output / "prepared.json")
    until = raw["primary_endpoint"] if until is None else until
    if type(until) is not int or until not in (raw["probe_endpoint"], *raw["validation_endpoints"]):
        raise ValueError("Use only a registered probe or validation endpoint")
    ledger = CycleBudget(output, raw["identity"])
    with ledger.lease():
        if any(cell["pending"] for cells in ledger.read()["branches"].values() for cell in cells.values()):
            raise ValueError("Unconfirmed operation requires diagnosis; no automatic repeat")
        prepared, protocol, scenes, source = read_prepared(output, allow_test_fixture=allow_test_fixture)
        reports = {}
        for arm in ARMS:
            trainer = _trainer(prepared, protocol, source, arm)
            head = ledger.head("ppo", arm)
            payload = decode(output / head["path"], head["sha256"])
            _envelope(payload, trainer, arm)
            for binding in payload.get("evidence", {}).values():
                original._check_file(output, binding)
            trainer.load_state_dict(payload["trainer"])
            if trainer.joint_steps != head["step"] or trainer.joint_steps > until:
                raise ValueError("Requested endpoint precedes this arm's confirmed state")
            reports[arm] = {}
            for target in sorted(set((prepared["probe_endpoint"], *prepared["validation_endpoints"]))):
                if target > until or target < trainer.joint_steps:
                    continue
                train_to(output, trainer, ledger, arm, target)
                if target in prepared["validation_endpoints"]:
                    report = evaluate_boundary(output, trainer, ledger, arm, scenes)
                    reports[arm][str(target)] = {k: report[k] for k in
                        ("primary_value", "capability", "warmup_capability", "summary")}
                    print(json.dumps({"event": "validation_complete", "shutdown_arm": arm, "steps": target,
                        "primary_value": report["primary_value"], "capability": report["capability"],
                        "warmup_capability": report["warmup_capability"]}), flush=True)
            del trainer
        result = {"status": "fixed_endpoint_completed" if until == prepared["primary_endpoint"] else "boundary_completed",
            "cycle_id": protocol["cycle_id"], "until": until, "ledger": ledger.read(),
            "validation_reports": reports, "formal_ready": False, "website_model_changed": False}
        write_json(output / f"completion_{until:07d}.json", result, replace=True)
        write_json(output / "progress.json", result, replace=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--until", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.prepare:
        result = prepare(args.output, device=args.device)
    else:
        class Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, value):
                for stream in self.streams: stream.write(value); stream.flush()
                return len(value)
            def flush(self):
                for stream in self.streams: stream.flush()
        output = Path(args.output).expanduser().resolve()
        with (output / "stdout.log").open("a") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
            result = advance(output, until=args.until)
    print(json.dumps({k: v for k, v in result.items() if k not in ("ledger", "validation_reports")}), flush=True)


if __name__ == "__main__":
    main()
