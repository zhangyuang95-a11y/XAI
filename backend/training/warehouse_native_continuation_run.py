"""Finite continuation from an acknowledged, registered validation checkpoint.

The original source learner is reconstructed through its genuine run/trainer
classes, including recursive ancestry. No algorithm, alpha, partner mix or
in-flight state is changed. Each cycle owns its finite ledger. This is a local
learning workflow, not model publication or an explanation qualification.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import re
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

VERSION = "warehouse-native-continuation-run.v1"
CREDIT_RUN_VERSION = "warehouse-native-credit-run.v1"


def sources():
    from .warehouse_native_continuation_trainer import execution_sources
    from .warehouse_native_continuation_evaluation import execution_sources as evaluation_sources
    result = execution_sources(); result.update(evaluation_sources())
    for name in ("warehouse_native_continuation_run.py", "warehouse_native_cycle_budget.py",
                 "warehouse_native_partner_mix_budget.py", "warehouse_native_partner_mix_run.py",
                 "warehouse_native_credit_run.py", "warehouse_native_credit_result.py",
                 "warehouse_native_partner_mix_result.py"):
        path = Path(__file__).with_name(name); result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _same(a, b, reason):
    if digest(a) != digest(b): raise ValueError(reason)


def _json(path): return json.loads(Path(path).read_bytes())


def _relative(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root):
        raise ValueError("Evidence path escapes source run")
    return path


def _check_file(root, binding):
    path = _relative(root, binding["path"])
    if file_hash(path) != binding["sha256"]: raise ValueError("Bound source evidence changed: " + str(path))
    if "size" in binding and path.stat().st_size != binding["size"]: raise ValueError("Evidence size differs")
    return path


def _report_record(run, branch, step, prepared, protocol, scenarios, account, fixture):
    """Pure stored-row/report checks, never a fresh validation or Actor forward."""
    from . import warehouse_native_partner_mix_evaluation as compact
    from . import warehouse_native_credit_evaluation as credit_evaluation
    from . import warehouse_native_continuation_evaluation as continuation_evaluation
    credit = prepared["version"] == CREDIT_RUN_VERSION
    evaluator = credit_evaluation if credit else continuation_evaluation
    folder = run / "branches" / branch / "validation" / f"step_{step:07d}"
    report, manifest = _json(folder / "report.json"), _json(folder / "manifest.json")
    entries = scenarios["splits"]["validation"]; n = len(entries)
    if (not fixture and n != 50) or n < 1: raise ValueError("Invalid validation matrix")
    if (report.get("version") != evaluator.VERSION or report.get("status") != "completed"
            or manifest.get("status") != "completed" or report.get("branch") != branch
            or report.get("test_fixture") is not fixture or report.get("additional_joint_steps") != step
            or len(report.get("rows", [])) != 3 * n or len(manifest.get("episodes", [])) != 3 * n):
        raise ValueError("Registered source validation is incomplete")
    _same(report["identity"], manifest["identity"], "Source validation identities differ")
    bindings = report["actor_bindings"]
    for name in ("protocol_sha256", "scenario_manifest_sha256", "source_checkpoint_sha256"):
        _same(bindings[name], prepared[name], "Validation source differs: " + name)
    if (bindings["branch"] != branch or bindings["joint_steps"] != step
            or bindings["initialization_sha256"] != prepared["source_state_sha256"]
            or bindings["experiment_version"] != evaluator.TRAINER_VERSION):
        raise ValueError("Validation checkpoint ancestry differs")
    actor_path = run / "branches" / branch / "actors" / f"actor_{step:07d}.npz"
    if file_hash(actor_path) != bindings["actor_sha256"]: raise ValueError("Validated Actor bytes changed")
    with np.load(io.BytesIO(actor_path.read_bytes()), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    if metadata.get("test_fixture", False) is not fixture: raise ValueError("Actor fixture boundary differs")
    for key, value in bindings.items():
        if key != "actor_sha256": _same(metadata.get(key), value, "Actor metadata binding differs: " + key)
    _same(report["identity"]["actor_metadata_sha256"], digest(metadata), "Actor metadata hash differs")
    total = 0
    for index, (stored, row) in enumerate(zip(manifest["episodes"], report["rows"])):
        partner, scenario_index = evaluator.PARTNERS[index // n], index % n
        context = stored["context"]; operation = account["operations"].get(context["operation_id"])
        horizon = manifest["identity"]["configuration"]["horizon"]
        if (stored["status"] != "completed" or row["partner"] != partner or row["scenario_index"] != scenario_index
                or row["scenario_id"] != entries[scenario_index]["id"] or row["seed"] != 17000 + scenario_index
                or type(row["steps"]) is not int or not 0 < row["steps"] <= horizon
                or row["nn_action_overrides"] != 0 or row["raw_neural_actions_submitted"] != row["steps"]
                or row["neural_role"] != 1 or not operation or operation["status"] != "acknowledged"
                or operation["request"]["branch"] != branch or operation["request"]["kind"] != "evaluation"
                or operation["request"]["steps"] != horizon or operation["completion"]["actual_steps"] != row["steps"]):
            raise ValueError("Source validation row is not a confirmed fixed-matrix episode")
        for key, value in context.items(): _same(row.get(key), value, "Validation row context differs")
        row_path = _check_file(folder, stored["row"]); _check_file(folder, stored["trace"])
        _same(_json(row_path), row, "Original validation row differs")
        completed = operation["completion"]["checkpoint"]
        if completed["path"] != str(row_path.relative_to(run)) or completed["sha256"] != stored["row"]["sha256"]:
            raise ValueError("Validation row was not acknowledged by its run")
        total += row["steps"]
    maximum = 3 * n * manifest["identity"]["configuration"]["horizon"]
    if manifest["reserved_environment_steps"] != maximum or manifest["actual_environment_steps"] != total:
        raise ValueError("Source validation accounting differs")
    baselines = _json(run / "baselines.json")
    gates = evaluator._gates(protocol)
    rebuilt = compact._summary(report["rows"], metadata, bindings, gates, baselines["reference"], baselines["random"],
                               fixture, report["identity"], manifest["reserved_environment_steps"])
    rebuilt.update(version=evaluator.VERSION, validation_reward="original_shared_r1",
        training_delivery_credit_alpha=metadata["delivery_credit_alpha"], remaining_reservable_steps=0)
    if not credit: rebuilt["cycle_id"] = protocol["cycle_id"]
    _same(report, rebuilt, "Original source report aggregates or gates differ")
    return report, actor_path


def _registered_source(run, branch, step, *, expected=None, allow_test_fixture=False):
    run = Path(run).expanduser().resolve()
    if not isinstance(branch, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", branch) is None:
        raise ValueError("Invalid source branch")
    if type(step) is not int or step <= 0: raise ValueError("Positive registered source step required")
    prepared, protocol, scenes = (_json(run / name) for name in ("prepared.json", "protocol.json", "scenarios.json"))
    if prepared.get("version") not in (CREDIT_RUN_VERSION, VERSION): raise ValueError("Unsupported genuine source run")
    fixture = prepared.get("test_fixture", False)
    if type(fixture) is not bool or fixture != allow_test_fixture or protocol.get("test_fixture", False) is not fixture:
        raise ValueError("Source fixture/production boundary differs")
    endpoints = prepared["validation_endpoints"]
    if step not in endpoints or step not in protocol["evaluation"]["checkpoints_ppo_steps"]:
        raise ValueError("Source step is not a registered validation endpoint")
    if digest(protocol) != prepared["protocol_sha256"] or digest(scenes) != prepared["scenario_manifest_sha256"]:
        raise ValueError("Source protocol or scenarios changed")
    ledger = CycleBudget(run, prepared["identity"])
    with ledger.lease():
        account = ledger.read()
        if branch not in account["branches"]: raise ValueError("Source branch is absent")
        if any(account["branches"][branch][kind]["pending"] for kind in ("ppo", "evaluation")):
            raise ValueError("Source has unconfirmed operations")
        candidates = [(key, op) for key, op in account["operations"].items()
            if op["status"] == "acknowledged" and op["request"]["kind"] == "ppo"
            and op["request"]["branch"] == branch and op["completion"]["checkpoint"]["step"] == step]
        if len(candidates) != 1: raise ValueError("Source lacks one acknowledged checkpoint at this boundary")
        opid, operation = candidates[0]; checkpoint = operation["completion"]["checkpoint"]
        path = _check_file(run, checkpoint)
        report, actor_path = _report_record(run, branch, step, prepared, protocol, scenes, account, fixture)
        parity = _json(actor_path.with_suffix(".json"))
        if (parity["checkpoint_sha256"] != checkpoint["sha256"] or parity.get("argmax_equal") is not True
                or not 0 <= parity["maximum_absolute_error"] <= 1e-4 or parity["actor"]["sha256"] != file_hash(actor_path)):
            raise ValueError("Source Actor lacks its confirmed checkpoint parity receipt")
        folder = actor_path.parent.parent / "validation" / f"step_{step:07d}"
        descriptor = {"run": str(run), "run_version": prepared["version"], "branch": branch, "step": step,
            "prepared_sha256": file_hash(run / "prepared.json"), "protocol_sha256": digest(protocol),
            "scenario_manifest_sha256": digest(scenes), "baselines_sha256": file_hash(run / "baselines.json"),
            "checkpoint": copy.deepcopy(checkpoint), "operation_id": opid, "operation_sha256": digest(operation),
            "validation_report_sha256": file_hash(folder / "report.json"),
            "validation_manifest_sha256": file_hash(folder / "manifest.json"),
            "actor_sha256": file_hash(actor_path), "parity_sha256": file_hash(actor_path.with_suffix(".json")),
            "test_fixture": fixture}
        if expected is not None: _same(descriptor, expected, "Selected source ancestry or records changed")
    return descriptor, prepared, protocol, scenes


def load_source(source_run, source_branch, source_step, *, device, expected_descriptor=None,
                allow_test_fixture=False, _seen=()):
    """Rebuild original class ancestry; never relabel a saved trainer payload."""
    run = Path(source_run).expanduser().resolve()
    if str(run) in _seen or len(_seen) >= 64: raise ValueError("Cyclic or excessive source ancestry")
    seen = (*_seen, str(run))
    descriptor, prepared, protocol, scenes = _registered_source(run, source_branch, source_step,
        expected=expected_descriptor, allow_test_fixture=allow_test_fixture)
    if prepared["device"] != device: raise ValueError("Continuation must keep the original training device")
    if prepared["version"] == CREDIT_RUN_VERSION:
        from . import warehouse_native_credit_run as original
        from .warehouse_native_credit_trainer import CreditTrainer
        saved, original_protocol, original_scenes, state, context = original.read_prepared(run)
        _same(original_protocol, protocol, "Genuine credit protocol changed")
        _same(original_scenes, scenes, "Genuine credit scenes changed")
        source = CreditTrainer(protocol, scenes, state, source_context=context, branch=source_branch,
            expected_source_state_sha256=saved["source_state_sha256"],
            source_checkpoint_sha256=saved["source_checkpoint_sha256"], device=device, test_fixture=allow_test_fixture)
    else:
        saved, original_protocol, original_scenes, ancestor = read_prepared(run, allow_test_fixture=allow_test_fixture, _seen=seen)
        from .warehouse_native_continuation_trainer import ContinuationTrainer
        source = ContinuationTrainer(protocol, ancestor,
            expected_source_state_sha256=saved["source_state_sha256"],
            source_checkpoint_sha256=saved["source_checkpoint_sha256"], device=device, test_fixture=allow_test_fixture)
    path = _relative(run, descriptor["checkpoint"]["path"])
    payload = decode(path, descriptor["checkpoint"]["sha256"])
    if (payload.get("version") != prepared["version"] or payload.get("branch") != source_branch
            or payload.get("operation_id") != descriptor["operation_id"]
            or payload.get("audit", {}).get("neural_overrides") != 0):
        raise ValueError("Acknowledged source envelope is not its genuine run output")
    for binding in payload.get("evidence", {}).values(): _check_file(run, binding)
    source.load_state_dict(payload["trainer"])
    if source.joint_steps != source_step: raise ValueError("Source trainer does not match registered step")
    return source, descriptor, scenes


def _check_prepared_files(output, prepared):
    if prepared["authorization_record_sha256"] != AUTHORIZATION_SHA or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError("Autonomous authorization changed")
    if prepared["version"] != VERSION or prepared["runtime_sources"] != sources(): raise ValueError("Continuation sources changed")
    for key, value in prepared["identity"].items(): _same(prepared.get(key), value, "Cycle identity differs: " + key)
    for name, key in (("authorization.json", "authorization_record_sha256"), ("baselines.json", "baselines_sha256")):
        if file_hash(output / name) != prepared[key]: raise ValueError("Bound cycle file changed: " + name)
    for name, sha in prepared["runtime_sources"].items():
        if file_hash(output / "source_snapshot" / name) != sha: raise ValueError("Archived runtime source changed")


def read_prepared(output, *, allow_test_fixture=False, _seen=()):
    output = Path(output).expanduser().resolve(); prepared = _json(output / "prepared.json")
    _check_prepared_files(output, prepared)
    if prepared.get("test_fixture") is not allow_test_fixture: raise ValueError("Cycle fixture boundary differs")
    protocol, scenes = _json(output / "protocol.json"), _json(output / "scenarios.json")
    if digest(protocol) != prepared["protocol_sha256"] or digest(scenes) != prepared["scenario_manifest_sha256"]:
        raise ValueError("Frozen continuation protocol/scenarios changed")
    descriptor = prepared["source_descriptor"]
    source, actual, source_scenes = load_source(descriptor["run"], descriptor["branch"], descriptor["step"],
        device=prepared["device"], expected_descriptor=descriptor, allow_test_fixture=allow_test_fixture, _seen=_seen)
    if initialization_sha256(source.state_dict()) != prepared["source_state_sha256"]: raise ValueError("Source learner state changed")
    _same(scenes, source_scenes, "Continuation cannot change scenarios")
    cap = protocol["budget"]["maximum_ppo_joint_steps"]
    expected_caps = {prepared["branch"]: {"ppo": cap, "evaluation": len(prepared["validation_endpoints"]) * 3 * len(scenes["splits"]["validation"]) * scenes["configuration"]["horizon"]}}
    if (prepared["validation_endpoints"] != protocol["evaluation"]["checkpoints_ppo_steps"]
            or prepared["primary_endpoint"] != cap or prepared["branch"] != source.branch
            or prepared["cycle_id"] != protocol["cycle_id"] or type(prepared["stop_when_warmup_ready"]) is not bool):
        raise ValueError("Prepared continuation endpoints or stop criterion differ")
    _same(prepared["budget_caps"], expected_caps, "Prepared cycle caps differ")
    initial = _json(output / "initialization_check.json")
    if initial.get("identical_learning_state") is not True or initial["source_state_sha256"] != prepared["source_state_sha256"]:
        raise ValueError("Continuation initialization is incomplete")
    return prepared, protocol, scenes, source


def prepare(output, *, source_run, source_branch, source_step, ppo_cap=500000,
            validation_interval=100000, cycle_id=None, device="mps", allow_test_fixture=False, stop_when_warmup_ready=True):
    from .warehouse_native_continuation_trainer import ContinuationTrainer, make_protocol
    output = Path(output).expanduser().resolve(); source_run = Path(source_run).expanduser().resolve()
    if output.exists(): raise FileExistsError("Use a new finite cycle directory")
    if output == source_run or source_run in output.parents or output in source_run.parents:
        raise ValueError("New cycle must be separate from its source")
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA: raise ValueError("Autonomous training authorization changed")
    if type(ppo_cap) is not int or ppo_cap < 16 or type(validation_interval) is not int or validation_interval <= 0:
        raise ValueError("Positive finite PPO cap and validation interval required")
    if device not in ("cpu", "mps") or type(allow_test_fixture) is not bool or type(stop_when_warmup_ready) is not bool: raise ValueError("Invalid device/fixture boundary")
    source, descriptor, scenes = load_source(source_run, source_branch, source_step, device=device, allow_test_fixture=allow_test_fixture)
    n = source.cfg["environments"]
    if ppo_cap % n or validation_interval % n: raise ValueError("Cap and interval must align with environment batch size")
    endpoints = list(range(validation_interval, ppo_cap, validation_interval)) + [ppo_cap]
    probe = min(4096, ppo_cap); probe -= probe % n
    if probe < n: raise ValueError("A complete probe batch is required")
    state = source.state_dict(); state_sha = initialization_sha256(state)
    cycle_id = cycle_id or output.name
    protocol = make_protocol(source, source_checkpoint_sha256=descriptor["checkpoint"]["sha256"],
        source_state_sha256=state_sha, ppo_cap=ppo_cap, cycle_id=cycle_id,
        evaluation_checkpoints=endpoints, test_fixture=allow_test_fixture)
    runtime_sources = sources(); branch = source.branch
    caps = {branch: {"ppo": ppo_cap, "evaluation": len(endpoints) * 3 * len(scenes["splits"]["validation"]) * scenes["configuration"]["horizon"]}}
    prepared = {"version": VERSION, "cycle_id": cycle_id, "device": device, "branch": branch,
        "source_descriptor": descriptor, "source_checkpoint_sha256": descriptor["checkpoint"]["sha256"],
        "source_state_sha256": state_sha, "protocol_sha256": digest(protocol), "scenario_manifest_sha256": digest(scenes),
        "baselines_sha256": file_hash(source_run / "baselines.json"), "runtime_sources": runtime_sources,
        "authorization_record_sha256": AUTHORIZATION_SHA, "budget_caps": caps, "primary_endpoint": ppo_cap,
        "validation_endpoints": endpoints, "probe_endpoint": probe, "test_fixture": allow_test_fixture,
        "stop_when_warmup_ready": stop_when_warmup_ready,
        "created_unix": time.time(), "formal_ready": False, "selection": "caller_selected_registered_validation_checkpoint_no_final_test"}
    prepared["identity"] = {key: copy.deepcopy(value) for key, value in prepared.items() if key not in ("created_unix", "formal_ready", "selection")}
    trainer = ContinuationTrainer(protocol, source, expected_source_state_sha256=state_sha,
        source_checkpoint_sha256=prepared["source_checkpoint_sha256"], device=device, test_fixture=allow_test_fixture)
    saved = trainer.state_dict(); matched = {}
    for key in ("model", "optimizers", "envs", "rng", "python_rng", "numpy_rng", "torch_rng", "partner_kinds",
                "program_roles", "scenario_ids", "episode_context", "episode_returns", "episode_reward_components"):
        matched[key] = initialization_sha256(saved[key]); _same(matched[key], initialization_sha256(state[key]), "Fork changed " + key)
    if device == "mps": _same(initialization_sha256(saved["mps_rng"]), initialization_sha256(state["mps_rng"]), "Fork changed MPS RNG")
    if sources() != runtime_sources: raise ValueError("Sources changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol); write_json(output / "prepared.json", prepared)
    write_json(output / "scenarios.json", scenes); write_bytes(output / "baselines.json", (source_run / "baselines.json").read_bytes())
    write_bytes(output / "authorization.json", AUTHORIZATION.read_bytes())
    for name in runtime_sources: write_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
    ledger = CycleBudget.create(output, prepared["identity"])
    with ledger.lease():
        checkpoint = output / "branches" / branch / "checkpoints/initial.pt"
        atomic_torch_save(checkpoint, {"version": VERSION, "cycle_id": cycle_id, "branch": branch,
            "operation_id": None, "trainer": saved, "new_environment_steps": 0})
        ledger.initialize_head("ppo", branch, str(checkpoint.relative_to(output)), file_hash(checkpoint))
        initial_eval = output / "branches" / branch / "initial_evaluation_reference.json"
        write_json(initial_eval, {"source_descriptor": descriptor, "new_environment_steps": 0})
        ledger.initialize_head("evaluation", branch, str(initial_eval.relative_to(output)), file_hash(initial_eval))
    write_json(output / "initialization_check.json", {"identical_learning_state": True, "matched": matched,
        "source_state_sha256": state_sha, "new_environment_steps": 0})
    return {"status": "prepared", "output": str(output), "branch": branch, "cycle_id": cycle_id,
        "ppo_cap": ppo_cap, "validation_endpoints": endpoints, "probe_endpoint": probe}


def train_to(output, trainer, ledger, branch, target):
    n = trainer.cfg["environments"]
    while trainer.joint_steps < target:
        if any(ledger.read()["branches"][branch][kind]["pending"] for kind in ("ppo", "evaluation")):
            raise ValueError("Unconfirmed cycle operation requires diagnosis; never resample")
        amount = min(n * trainer.cfg["rollout_steps"], target - trainer.joint_steps, ledger.remaining("ppo", branch))
        amount -= amount % n
        if amount <= 0: raise ValueError("Reserved budget cannot reach the fixed endpoint")
        head = ledger.head("ppo", branch)
        if trainer.joint_steps != head["step"]: raise ValueError("Learner counter differs from acknowledged head")
        before = initialization_sha256(trainer.state_dict())
        old = decode(output / head["path"], head["sha256"])
        if (old["version"] != VERSION or old["branch"] != branch or old.get("cycle_id") != trainer.protocol["cycle_id"]
                or initialization_sha256(old["trainer"]) != before):
            raise ValueError("Learner state differs from the confirmed predecessor")
        opid = "ppo_" + uuid.uuid4().hex
        reservation = ledger.reserve("ppo", branch, amount, opid, expected_step=trainer.joint_steps)
        if not reservation["execution_permitted"]: raise ValueError("Repeated reservation is not execution permission")
        started = time.monotonic(); batch, metrics = trainer.train_chunk(amount // n)
        if (trainer.joint_steps != head["step"] + amount or len(batch["transition_records"]) != amount
                or batch["audit"]["neural_overrides"] != 0 or any(not np.isfinite(v) for v in metrics.values())):
            raise ValueError("Actual training differs from its reservation or NN contract")
        evidence = save_batch(output, branch, opid, batch)
        episodes = copy.deepcopy(trainer.completed_episodes); trainer.completed_episodes.clear()
        checkpoint = output / "branches" / branch / "checkpoints" / (f"step_{trainer.joint_steps:07d}_" + opid + ".pt")
        payload = {"version": VERSION, "cycle_id": trainer.protocol["cycle_id"], "branch": branch,
            "operation_id": opid, "trainer": trainer.state_dict(), "audit": batch["audit"], "metrics": metrics,
            "evidence": evidence, "actual_steps": amount, "completed_episodes": episodes,
            "before_state_sha256": before, "elapsed_seconds": time.monotonic() - started}
        atomic_torch_save(checkpoint, payload)
        ledger.ack(opid, str(checkpoint.relative_to(output)), file_hash(checkpoint), amount)
        progress = {"status": "training", "branch": branch, "cycle_id": trainer.protocol["cycle_id"],
            "branch_ppo_steps": trainer.joint_steps, "target": target, "last_metrics": metrics,
            "last_chunk_seconds": payload["elapsed_seconds"], "ledger": ledger.read()}
        write_json(output / "progress.json", progress, replace=True)
        print(json.dumps({"event": "ppo_ack", "branch": branch, "steps": trainer.joint_steps,
            "seconds": round(payload["elapsed_seconds"], 3), "neural_overrides": batch["audit"]["neural_overrides"]}), flush=True)


def evaluate_boundary(output, trainer, ledger, branch, scenes):
    from .warehouse_native_continuation_evaluation import evaluate
    actor = export_actor(output, trainer, branch, ledger.head("ppo", branch))
    bindings = {key: actor.metadata[key] for key in ("experiment_version", "branch", "protocol_sha256",
        "scenario_manifest_sha256", "initialization_sha256", "source_sha256", "source_checkpoint_sha256", "joint_steps", "cycle_id")}
    bindings["actor_sha256"] = actor.artifact_sha256
    baselines = _json(output / "baselines.json")
    confirmed = {opid for opid, op in ledger.read()["operations"].items() if op["status"] == "acknowledged"
                 and op["request"]["kind"] == "evaluation" and op["request"]["branch"] == branch}
    def reserve(context):
        op = ledger.reserve("evaluation", branch, context["horizon"], context["operation_id"])
        if not op["execution_permitted"]: raise ValueError("Evaluation already reserved")
    def ack(result):
        ledger.ack(result["operation_id"], str(Path(result["row_path"]).relative_to(output)), result["row_sha256"], result["actual_steps"])
    maximum = 3 * len(scenes["splits"]["validation"]) * scenes["configuration"]["horizon"]
    return evaluate(actor, scenes["splits"]["validation"], trainer.protocol,
        output / "branches" / branch / "validation" / f"step_{trainer.joint_steps:07d}", maximum,
        expected_bindings=bindings, reference_report=baselines["reference"], random_report=baselines["random"],
        before_episode=reserve, on_episode=ack, confirmed_operation_ids=confirmed,
        allow_test_fixture=trainer.test_fixture)


def advance(output, *, until=None, allow_test_fixture=False):
    from .warehouse_native_continuation_trainer import ContinuationTrainer
    output = Path(output).expanduser().resolve()
    raw = _json(output / "prepared.json")
    until = raw["primary_endpoint"] if until is None else until
    if type(until) is not int or until not in (raw["probe_endpoint"], *raw["validation_endpoints"]):
        raise ValueError("Use only the registered probe or validation boundaries")
    ledger = CycleBudget(output, raw["identity"])
    with ledger.lease():
        branch = raw["branch"]
        if any(ledger.read()["branches"][branch][kind]["pending"] for kind in ("ppo", "evaluation")):
            raise ValueError("Unconfirmed cycle operation requires diagnosis; no implicit recovery sampling")
        prepared, protocol, scenes, source = read_prepared(output, allow_test_fixture=allow_test_fixture, _seen=(str(output),))
        trainer = ContinuationTrainer(protocol, source, expected_source_state_sha256=prepared["source_state_sha256"],
            source_checkpoint_sha256=prepared["source_checkpoint_sha256"], device=prepared["device"], test_fixture=allow_test_fixture)
        head = ledger.head("ppo", branch); payload = decode(output / head["path"], head["sha256"])
        if payload["version"] != VERSION or payload["branch"] != branch or payload["cycle_id"] != protocol["cycle_id"]:
            raise ValueError("Wrong continuation checkpoint")
        for binding in payload.get("evidence", {}).values(): _check_file(output, binding)
        trainer.load_state_dict(payload["trainer"])
        if trainer.joint_steps != head["step"] or trainer.joint_steps > until: raise ValueError("Invalid requested continuation boundary")
        reports = {}; ready = False
        for target in sorted(set((prepared["probe_endpoint"], *prepared["validation_endpoints"]))):
            if target > until or target < trainer.joint_steps: continue
            train_to(output, trainer, ledger, branch, target)
            if target in prepared["validation_endpoints"]:
                report = evaluate_boundary(output, trainer, ledger, branch, scenes); reports[str(target)] = report
                ready = report["capability"]["eligible"] or report["warmup_capability"]["eligible"]
                print(json.dumps({"event": "validation_complete", "steps": target, "primary_value": report["primary_value"],
                    "capability": report["capability"], "warmup_capability": report["warmup_capability"]}), flush=True)
                if ready and prepared["stop_when_warmup_ready"]: break
        status = "warmup_ready" if ready and prepared["stop_when_warmup_ready"] else ("fixed_endpoint_completed" if trainer.joint_steps == prepared["primary_endpoint"] else "boundary_completed")
        result = {"status": status, "cycle_id": protocol["cycle_id"], "branch": branch, "until": trainer.joint_steps,
            "ledger": ledger.read(), "validation_reports": {key: {k:v for k,v in report.items() if k not in ("rows", "artifacts")}
                for key,report in reports.items()}, "formal_ready": False, "website_model_changed": False}
        write_json(output / f"completion_{trainer.joint_steps:07d}.json", result, replace=True)
        write_json(output / "progress.json", result, replace=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--prepare", action="store_true"); mode.add_argument("--run", action="store_true")
    parser.add_argument("--source-run"); parser.add_argument("--source-branch"); parser.add_argument("--source-step", type=int)
    parser.add_argument("--ppo-cap", type=int, default=500000); parser.add_argument("--validation-interval", type=int, default=100000)
    parser.add_argument("--cycle-id"); parser.add_argument("--device", choices=("cpu", "mps"), default="mps"); parser.add_argument("--until", type=int)
    args = parser.parse_args(argv); torch.set_num_threads(1)
    if args.prepare:
        if not args.source_run or not args.source_branch or args.source_step is None: parser.error("Preparation needs a registered source run, branch and step")
        result = prepare(args.output, source_run=args.source_run, source_branch=args.source_branch, source_step=args.source_step,
            ppo_cap=args.ppo_cap, validation_interval=args.validation_interval, cycle_id=args.cycle_id, device=args.device)
    else:
        output = Path(args.output).expanduser().resolve()
        class Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, value):
                for stream in self.streams: stream.write(value); stream.flush()
                return len(value)
            def flush(self):
                for stream in self.streams: stream.flush()
        with (output / "stdout.log").open("a") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
            result = advance(output, until=args.until)
    print(json.dumps({k:v for k,v in result.items() if k not in ("ledger", "validation_reports")}), flush=True)


if __name__ == "__main__": main()
