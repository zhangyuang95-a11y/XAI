"""Autonomous finite continuation experiment on delivery credit assignment.

This round keeps the actual source weights, optimizer moments, in-flight
episodes and public task unchanged. It is not a RCPD feedback experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native import atomic_torch_save
from backend.training.warehouse_native_public_feedback_initialization import initialization_sha256
from backend.training.warehouse_native_partner_mix_run import (
    write_bytes, write_json, decode, save_batch, export_actor,
    AUTHORIZATION, AUTHORIZATION_SHA,
)

VERSION = "warehouse-native-credit-run.v1"
BRANCHES = ("team_credit", "own_credit")
PARENT = ROOT / "output/warehouse_native/partner_mix_20260908"
ENDPOINTS = (50000, 250000)
CAPS = {b: {"ppo": 250000, "evaluation": 36000} for b in BRANCHES}


def sources():
    from .warehouse_native_credit_trainer import execution_sources
    result = execution_sources()
    for name in ("warehouse_native_credit_run.py", "warehouse_native_cycle_budget.py",
                 "warehouse_native_credit_evaluation.py", "warehouse_native_partner_mix_run.py",
                 "warehouse_native_partner_mix_budget.py"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    # The new evaluator can use a standalone compact I/O implementation; it
    # explicitly declares its execution closure rather than relying on a glob.
    from .warehouse_native_credit_evaluation import execution_sources as evaluation_sources
    result.update(evaluation_sources())
    return result


def _inputs(prepared):
    source = decode(Path(prepared["source_checkpoint"]), prepared["source_checkpoint_sha256"])["trainer"]
    original = decode(Path(prepared["original_source_checkpoint"]), prepared["original_source_checkpoint_sha256"])["trainer"]
    if initialization_sha256(source) != prepared["source_state_sha256"]:
        raise ValueError("Source learner state differs")
    if initialization_sha256(original) != prepared["original_source_state_sha256"]:
        raise ValueError("Original source learner state differs")
    context = {"protocol": copy.deepcopy(source["protocol"]), "original_source_state": original,
        "original_source_state_sha256": prepared["original_source_state_sha256"],
        "original_source_checkpoint_sha256": prepared["original_source_checkpoint_sha256"]}
    return source, context


def _trainer(protocol, scenarios, source, context, branch, prepared):
    from .warehouse_native_credit_trainer import CreditTrainer
    return CreditTrainer(protocol, scenarios, source, source_context=context, branch=branch,
        expected_source_state_sha256=prepared["source_state_sha256"],
        source_checkpoint_sha256=prepared["source_checkpoint_sha256"], device=prepared["device"])


def prepare(output, *, device="mps"):
    from .warehouse_native_credit_trainer import make_protocol
    from .warehouse_native_cycle_budget import CycleBudget
    output = Path(output).expanduser().resolve()
    if output.exists(): raise FileExistsError("Use a new cycle directory")
    if output == PARENT or PARENT in output.parents or output in PARENT.parents:
        raise ValueError("The new cycle must be separate from the completed experiment")
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError("Open-ended local training authorization differs")
    previous_result_path = PARENT / "result.json"
    previous_result = json.loads(previous_result_path.read_bytes())
    if previous_result.get("status") != "fixed_endpoint_complete":
        raise ValueError("The preceding result has not passed fixed-endpoint aggregation")
    # This design is triggered by the completed fixed comparison, never by a
    # conveniently selected intermediate score. The result schema is bound
    # below, and its producer already validates all four episode matrices.
    previous_prepared = json.loads((PARENT / "prepared.json").read_bytes())
    from .warehouse_native_partner_mix_budget import PartnerMixBudget
    previous_ledger = PartnerMixBudget(PARENT, previous_prepared["identity"])
    account = previous_ledger.read()
    if any(account["branches"][b]["ppo"]["acknowledged"] != 100000
           or account["branches"][b]["ppo"]["pending"]
           or account["branches"][b]["evaluation"]["reserved"] != 36000
           or account["branches"][b]["evaluation"]["pending"] for b in ("baseline_mix", "active_mix")):
        raise ValueError("The preceding fixed-endpoint comparison is incomplete")
    reports = [json.loads((PARENT / "branches" / b / "validation/step_0100000/report.json").read_bytes())
               for b in ("baseline_mix", "active_mix")]
    if any(r["status"] != "completed" or r["episodes"] != 150 for r in reports):
        raise ValueError("Both preceding final validations must be complete")
    if any(r["warmup_capability"]["eligible"] or r["capability"]["eligible"] for r in reports):
        raise ValueError("A base candidate is already ready; consider the RCPD stage instead")
    survival_ok = all(reports[1]["summary"][p]["ai_active_end_rate"] -
        reports[0]["summary"][p]["ai_active_end_rate"] >= -.05 - 1e-12
        for p in ("skilled", "assertive", "noisy"))
    if reports[1]["primary_value"] - reports[0]["primary_value"] >= .5 and survival_ok:
        raise ValueError("The registered mixture improvement passed; review that result before changing credit")
    head = previous_ledger.head("ppo", "baseline_mix")
    source_path = PARENT / head["path"]
    envelope = decode(source_path, head["sha256"])
    source = envelope["trainer"]
    if source["branch"] != "baseline_mix" or source["joint_steps"] != 100000:
        raise ValueError("Only the original-mixture fixed100k source is registered")
    state_sha = initialization_sha256(source)
    protocol = make_protocol(source["protocol"], source_checkpoint_sha256=head["sha256"],
                             source_state_sha256=state_sha)
    scenarios_raw = (PARENT / "scenarios.json").read_bytes()
    scenarios = json.loads(scenarios_raw)
    baselines_raw = (PARENT / "baselines.json").read_bytes()
    binding = sources()
    prepared = {"version": VERSION, "device": device, "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": head["sha256"], "source_state_sha256": state_sha,
        "original_source_checkpoint": previous_prepared["source_checkpoint"],
        "original_source_checkpoint_sha256": previous_prepared["source_checkpoint_sha256"],
        "original_source_state_sha256": previous_prepared["source_state_sha256"],
        "previous_result": str(previous_result_path), "previous_result_sha256": file_hash(previous_result_path),
        "protocol_sha256": digest(protocol), "scenario_manifest_sha256": digest(scenarios),
        "baselines_sha256": file_hash(PARENT / "baselines.json"), "runtime_sources": binding,
        "authorization_record_sha256": AUTHORIZATION_SHA, "budget_caps": copy.deepcopy(CAPS),
        "primary_endpoint": 250000, "validation_endpoints": list(ENDPOINTS),
        "created_unix": time.time(), "formal_ready": False,
        "allocation": "agent_selected_finite_round_under_user_open_authorization",
        "selection": "fixed_baseline_mix100k_after_failed_mixture_comparison_not_best_checkpoint"}
    prepared["identity"] = {key: copy.deepcopy(prepared[key]) for key in
        ("version", "device", "source_checkpoint_sha256", "source_state_sha256",
         "original_source_checkpoint_sha256", "original_source_state_sha256", "previous_result_sha256",
         "protocol_sha256", "scenario_manifest_sha256", "baselines_sha256", "runtime_sources",
         "authorization_record_sha256", "budget_caps", "primary_endpoint")}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "prepared.json", prepared)
    write_bytes(output / "scenarios.json", scenarios_raw)
    write_bytes(output / "baselines.json", baselines_raw)
    write_bytes(output / "authorization.json", AUTHORIZATION.read_bytes())
    write_bytes(output / "preceding_result.json", previous_result_path.read_bytes())
    for name in binding: write_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
    source, context = _inputs(prepared)
    ledger = CycleBudget.create(output, prepared["identity"])
    matched = {}
    with ledger.lease():
        for branch in BRANCHES:
            trainer = _trainer(protocol, scenarios, source, context, branch, prepared)
            saved = trainer.state_dict()
            fields = ("model", "optimizers", "rng", "python_rng", "numpy_rng", "torch_rng",
                      "partner_kinds", "program_roles", "scenario_ids", "episode_context",
                      "episode_returns", "episode_reward_components")
            matched[branch] = {k: initialization_sha256(saved[k]) for k in fields}
            if device == "mps": matched[branch]["mps_rng"] = initialization_sha256(saved["mps_rng"])
            for key, value in matched[branch].items():
                if value != initialization_sha256(source[key]):
                    raise ValueError("Credit fork changed learning/in-flight state: " + key)
            # Reward version metadata can change. Actual physical state and
            # environment RNG must be preserved by the new environment import.
            for old_env, new_env in zip(source["envs"], saved["envs"]):
                for key in ("version", "configuration", "state", "rng", "episode_counter",
                            "public_feedback_version", "public_feedback_mode", "public_feedback_history"):
                    if key in old_env and initialization_sha256(old_env[key]) != initialization_sha256(new_env[key]):
                        raise ValueError("Credit migration changed original environment " + key)
            path = output / "branches" / branch / "checkpoints/initial.pt"
            atomic_torch_save(path, {"version": VERSION, "branch": branch, "trainer": saved,
                                    "operation_id": None, "new_environment_steps": 0})
            ledger.initialize_head("ppo", branch, str(path.relative_to(output)), file_hash(path))
            initial_eval = output / "branches" / branch / "initial_evaluation_reference.json"
            reference_path = PARENT / "branches/baseline_mix/validation/step_0100000/report.json"
            write_json(initial_eval, {"source": str(reference_path), "sha256": file_hash(reference_path),
                                      "new_environment_steps": 0})
            ledger.initialize_head("evaluation", branch, str(initial_eval.relative_to(output)), file_hash(initial_eval))
            del trainer
    if matched[BRANCHES[0]] != matched[BRANCHES[1]]: raise ValueError("Initial neural forks differ")
    write_json(output / "initialization_check.json", {"identical_learning_state": True,
        "matched": matched, "source_checkpoint_sha256": head["sha256"], "new_environment_steps": 0})
    return {"status": "prepared", "output": str(output), "device": device}


def read_prepared(output):
    prepared = json.loads((output / "prepared.json").read_bytes())
    if prepared["version"] != VERSION or prepared["runtime_sources"] != sources():
        raise ValueError("Credit run execution sources changed")
    if any(prepared.get(k) != v for k, v in prepared["identity"].items()):
        raise ValueError("Credit run identity differs")
    for name, key in (("authorization.json", "authorization_record_sha256"),
                      ("baselines.json", "baselines_sha256"), ("preceding_result.json", "previous_result_sha256")):
        if file_hash(output / name) != prepared[key]: raise ValueError("Bound file changed: " + name)
    protocol = json.loads((output / "protocol.json").read_bytes())
    scenarios = json.loads((output / "scenarios.json").read_bytes())
    if digest(protocol) != prepared["protocol_sha256"] or digest(scenarios) != prepared["scenario_manifest_sha256"]:
        raise ValueError("Frozen protocol/scenarios changed")
    initial = json.loads((output / "initialization_check.json").read_bytes())
    if initial.get("identical_learning_state") is not True or initial["source_checkpoint_sha256"] != prepared["source_checkpoint_sha256"]:
        raise ValueError("Credit fork preparation is incomplete")
    source, context = _inputs(prepared)
    return prepared, protocol, scenarios, source, context


def train_to(output, trainer, ledger, branch, target):
    count = trainer.cfg["environments"]
    while trainer.joint_steps < target:
        if ledger.read()["branches"][branch]["ppo"]["pending"]:
            raise ValueError("Unconfirmed training operation requires recovery diagnosis")
        amount = min(count * trainer.cfg["rollout_steps"], target - trainer.joint_steps, ledger.remaining("ppo", branch))
        amount -= amount % count
        if amount <= 0: raise ValueError("Cycle cannot reach the registered endpoint within its reservation")
        opid = "ppo_" + uuid.uuid4().hex
        old_step = trainer.joint_steps
        reservation = ledger.reserve("ppo", branch, amount, opid, expected_step=old_step)
        if not reservation["execution_permitted"]: raise ValueError("Repeated reservation is not execution permission")
        start = time.monotonic()
        batch, metrics = trainer.train_chunk(amount // count)
        if (trainer.joint_steps != old_step + amount or len(batch["transition_records"]) != amount
                or batch["audit"]["neural_overrides"] != 0):
            raise ValueError("Actual neural training transitions differ from the reserved operation")
        if any(not np.isfinite(value) for value in metrics.values()): raise FloatingPointError("Nonfinite training update")
        evidence = save_batch(output, branch, opid, batch)
        episodes = copy.deepcopy(trainer.completed_episodes)
        trainer.completed_episodes.clear()
        path = output / "branches" / branch / "checkpoints" / (f"step_{trainer.joint_steps:07d}_" + opid + ".pt")
        payload = {"version": VERSION, "branch": branch, "operation_id": opid, "trainer": trainer.state_dict(),
            "audit": batch["audit"], "metrics": metrics, "evidence": evidence, "completed_episodes": episodes,
            "actual_steps": amount, "elapsed_seconds": time.monotonic() - start}
        atomic_torch_save(path, payload)
        ledger.ack(opid, str(path.relative_to(output)), file_hash(path), amount)
        write_json(output / "progress.json", {"status": "training", "branch": branch,
            "branch_ppo_steps": trainer.joint_steps, "target": target, "ledger": ledger.read(),
            "last_metrics": metrics, "last_chunk_seconds": payload["elapsed_seconds"]}, replace=True)
        print(json.dumps({"event": "ppo_ack", "branch": branch, "steps": trainer.joint_steps,
            "seconds": round(payload["elapsed_seconds"], 3), "neural_submitted": batch["audit"]["neural_submitted"]}), flush=True)


def evaluate_boundary(output, trainer, ledger, branch, scenarios):
    from .warehouse_native_credit_evaluation import evaluate
    actor = export_actor(output, trainer, branch, ledger.head("ppo", branch))
    bindings = {key: actor.metadata[key] for key in ("experiment_version", "branch", "protocol_sha256",
        "scenario_manifest_sha256", "initialization_sha256", "source_sha256", "source_checkpoint_sha256", "joint_steps")}
    bindings["actor_sha256"] = actor.artifact_sha256
    baselines = json.loads((output / "baselines.json").read_bytes())
    confirmed = {opid for opid, op in ledger.read()["operations"].items()
                 if op["status"] == "acknowledged" and op["request"]["kind"] == "evaluation" and op["request"]["branch"] == branch}
    def reserve(context):
        op = ledger.reserve("evaluation", branch, 120, context["operation_id"])
        if not op["execution_permitted"]: raise ValueError("Evaluation has already been reserved")
    def ack(result):
        path = Path(result["row_path"])
        ledger.ack(result["operation_id"], str(path.relative_to(output)), result["row_sha256"], result["actual_steps"])
    report = evaluate(actor, scenarios["splits"]["validation"], trainer.protocol,
        output / "branches" / branch / "validation" / f"step_{trainer.joint_steps:07d}", 18000,
        expected_bindings=bindings, reference_report=baselines["reference"], random_report=baselines["random"],
        before_episode=reserve, on_episode=ack, confirmed_operation_ids=confirmed)
    print(json.dumps({"event": "validation_complete", "branch": branch, "steps": trainer.joint_steps,
        "primary_value": report["primary_value"], "capability": report["capability"]}), flush=True)
    return report


def advance(output, *, until):
    from .warehouse_native_cycle_budget import CycleBudget
    if until not in (4096, *ENDPOINTS): raise ValueError("Use the registered probe or validation endpoints")
    output = Path(output).expanduser().resolve()
    prepared, protocol, scenarios, source, context = read_prepared(output)
    ledger = CycleBudget(output, prepared["identity"])
    started = time.monotonic()
    reports = {}
    with ledger.lease():
        for branch in BRANCHES:
            trainer = _trainer(protocol, scenarios, source, context, branch, prepared)
            head = ledger.head("ppo", branch)
            payload = decode(output / head["path"], head["sha256"])
            if payload["version"] != VERSION or payload["branch"] != branch: raise ValueError("Wrong credit checkpoint branch")
            trainer.load_state_dict(payload["trainer"])
            if trainer.joint_steps != head["step"]: raise ValueError("Ledger and learner counters disagree")
            if trainer.joint_steps > until: raise ValueError("The requested endpoint precedes the confirmed checkpoint")
            for target in (4096, *ENDPOINTS):
                if target > until or target < trainer.joint_steps: continue
                train_to(output, trainer, ledger, branch, target)
                if target in ENDPOINTS:
                    reports[branch + "/" + str(target)] = evaluate_boundary(output, trainer, ledger, branch, scenarios)
            del trainer
        result = {"status": "fixed_endpoint_completed" if until == ENDPOINTS[-1] else "boundary_completed",
            "until": until, "elapsed_seconds": time.monotonic() - started, "ledger": ledger.read(),
            "validation_reports": {key: {k: v for k, v in value.items() if k not in ("rows", "artifacts")}
                                   for key, value in reports.items()}, "formal_ready": False, "website_model_changed": False}
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
    parser.add_argument("--until", type=int, choices=(4096, *ENDPOINTS), default=250000)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = prepare(args.output, device=args.device) if args.prepare else advance(args.output, until=args.until)
    print(json.dumps({k: v for k, v in result.items() if k not in ("ledger", "validation_reports")}), flush=True)


if __name__ == "__main__": main()
