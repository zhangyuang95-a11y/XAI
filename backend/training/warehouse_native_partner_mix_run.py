"""Run a versioned, matched partner-mixture experiment under user autonomy.

Each experiment retains its own finite ledger and fixed endpoints. This runner
does not overwrite older training, select final-test scenes or deploy a model.
"""
from __future__ import annotations

import argparse
import copy
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
import torch

from backend.training.warehouse_native_common import ROOT, digest, file_hash, canonical
from backend.training.warehouse_native import atomic_torch_save
from backend.training.warehouse_native_public_feedback_initialization import initialization_sha256
from env.warehouse_native.policy import NumPyNativeActor

VERSION = "warehouse-native-partner-mix-run.v1"
BRANCHES = ("baseline_mix", "active_mix")
PARENT = ROOT / "output/warehouse_native/public_history_pair_prepared_20260908"
SOURCE = PARENT / "branches/observed/checkpoints/ppo_5fcaf50bd2a248daac22326703018ac0.pt"
SOURCE_SHA = "8e9f3c1ba35ac9f913ea9512c818f903f24691d2f8bc17eec7bbcd0737bd6ebd"
AUTHORIZATION = ROOT / "output/warehouse_native/autonomous_training_authorization_20260908/authorization.json"
AUTHORIZATION_SHA = "1faea3405110c8730614531752a18bb73937002dbfdc2bb803f876f72d8fd962"


def sources():
    from .warehouse_native_partner_mix_trainer import execution_sources
    result = execution_sources()
    for name in ("warehouse_native_partner_mix_run.py", "warehouse_native_partner_mix_budget.py",
                 "warehouse_native_partner_mix_evaluation.py"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def write_bytes(path, raw, *, replace=False):
    path = Path(path)
    if path.exists() and not replace: raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def write_json(path, value, *, replace=False):
    write_bytes(path, (canonical(value) + "\n").encode(), replace=replace)


def bound(root, path):
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path), "size": path.stat().st_size}


def decode(path, expected):
    if file_hash(path) != expected: raise ValueError("Source file changed: " + str(path))
    return torch.load(path, map_location="cpu", weights_only=False)


def _trainer(protocol, scenarios, state, branch, prepared):
    from .warehouse_native_partner_mix_trainer import PartnerMixTrainer
    return PartnerMixTrainer(protocol, scenarios, state, branch=branch,
        expected_source_state_sha256=prepared["source_state_sha256"],
        source_checkpoint_sha256=prepared["source_checkpoint_sha256"], device=prepared["device"])


def prepare(output, *, device="mps"):
    from .warehouse_native_partner_mix_trainer import make_protocol
    from .warehouse_native_partner_mix_budget import PartnerMixBudget
    output = Path(output).expanduser().resolve()
    if output.exists(): raise FileExistsError("Use a new experiment output")
    if PARENT == output or PARENT in output.parents or output in PARENT.parents:
        raise ValueError("New experiment must be separate from old training")
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA: raise ValueError("Autonomous authorization changed")
    if device not in ("cpu", "mps"): raise ValueError("Expected local CPU or MPS device")
    source = decode(SOURCE, SOURCE_SHA)
    if source["branch"] != "observed" or source["trainer"]["joint_steps"] != 100000:
        raise ValueError("This protocol fixes the recorded observed 100k endpoint")
    state = source["trainer"]; state_sha = initialization_sha256(state)
    protocol = make_protocol(state["protocol"], source_checkpoint_sha256=SOURCE_SHA,
                             source_state_sha256=state_sha)
    scenes_raw = (PARENT / "scenarios.json").read_bytes(); scenarios = json.loads(scenes_raw)
    baselines_raw = (PARENT / "baselines.json").read_bytes()
    source_binding = sources()
    prepared = {"version": VERSION, "device": device, "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": SOURCE_SHA, "source_state_sha256": state_sha,
        "protocol_sha256": digest(protocol), "scenario_manifest_sha256": digest(scenarios),
        "baselines_sha256": sha256(baselines_raw).hexdigest(),
        "sources": source_binding, "authorization_sha256": AUTHORIZATION_SHA,
        "ppo_cap_per_branch": 100000, "evaluation_cap_per_branch": 36000,
        "registered_validation_endpoints": [50000, 100000], "primary_endpoint": 100000,
        "zero_step_report_reused": str(PARENT / "branches/observed/validation/step_0100000.json"),
        "zero_step_report_sha256": "878e482fd9c867fcbacacd7792c7c1ccefc6aa6298c42bfdce75db7b3221b44a",
        "created_unix": time.time(), "formal_ready": False}
    prepared["identity"] = {key: copy.deepcopy(prepared[key]) for key in
        ("version", "device", "source_checkpoint_sha256", "source_state_sha256", "protocol_sha256",
         "scenario_manifest_sha256", "baselines_sha256", "sources", "authorization_sha256", "primary_endpoint")}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_bytes(output / "scenarios.json", scenes_raw)
    write_bytes(output / "baselines.json", baselines_raw)
    write_bytes(output / "authorization.json", AUTHORIZATION.read_bytes())
    write_json(output / "prepared.json", prepared)
    for name in source_binding:
        write_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
    ledger = PartnerMixBudget.create(output, prepared["identity"])
    identities = {}
    with ledger.lease():
        for branch in BRANCHES:
            trainer = _trainer(protocol, scenarios, state, branch, prepared)
            saved = trainer.state_dict()
            identities[branch] = {key: initialization_sha256(saved[key]) for key in
                ("model", "optimizers", "envs", "rng", "python_rng", "numpy_rng", "torch_rng",
                 "partner_kinds", "program_roles", "episode_context")}
            if device == "mps": identities[branch]["mps_rng"] = initialization_sha256(saved["mps_rng"])
            for key in identities[branch]:
                if identities[branch][key] != initialization_sha256(state[key]):
                    raise ValueError("Fork changed original learning or in-flight state: " + key)
            path = output / "branches" / branch / "checkpoints/initial.pt"
            atomic_torch_save(path, {"version": VERSION, "branch": branch, "trainer": saved,
                                    "operation_id": None, "new_environment_steps": 0})
            ledger.initialize_head("ppo", branch, str(path.relative_to(output)), file_hash(path))
            initial_eval = output / "branches" / branch / "initial_evaluation_reference.json"
            write_json(initial_eval, {"scope": "original_record_reference_no_new_episode", "new_steps": 0,
                "source": prepared["zero_step_report_reused"], "sha256": prepared["zero_step_report_sha256"]})
            ledger.initialize_head("evaluation", branch, str(initial_eval.relative_to(output)), file_hash(initial_eval))
            del trainer
    if identities[BRANCHES[0]] != identities[BRANCHES[1]]:
        raise ValueError("The matched forks differ before any new training")
    write_json(output / "initialization_check.json", {"identical": True, "digests": identities,
        "source_checkpoint_sha256": SOURCE_SHA, "new_environment_steps": 0, "new_optimizer_updates": 0})
    return {"status": "prepared", "output": str(output), "device": device}


def read_prepared(output):
    prepared = json.loads((output / "prepared.json").read_bytes())
    if prepared["version"] != VERSION or prepared["sources"] != sources():
        raise ValueError("Experiment execution sources changed")
    if file_hash(output / "authorization.json") != prepared["authorization_sha256"]:
        raise ValueError("Experiment authorization changed")
    if file_hash(output / "baselines.json") != prepared["baselines_sha256"]:
        raise ValueError("Frozen baseline records changed")
    if any(prepared.get(k) != v for k, v in prepared["identity"].items()):
        raise ValueError("Prepared identity differs from its bindings")
    initialization = json.loads((output / "initialization_check.json").read_bytes())
    if initialization.get("identical") is not True or initialization.get("source_checkpoint_sha256") != prepared["source_checkpoint_sha256"]:
        raise ValueError("Matched fork preparation was not completed")
    protocol = json.loads((output / "protocol.json").read_bytes())
    scenarios = json.loads((output / "scenarios.json").read_bytes())
    if digest(protocol) != prepared["protocol_sha256"] or digest(scenarios) != prepared["scenario_manifest_sha256"]:
        raise ValueError("Frozen protocol or scenarios changed")
    source = decode(Path(prepared["source_checkpoint"]), prepared["source_checkpoint_sha256"])["trainer"]
    if initialization_sha256(source) != prepared["source_state_sha256"]: raise ValueError("Source state differs")
    return prepared, protocol, scenarios, source


def save_batch(output, branch, operation_id, batch):
    directory = output / "branches" / branch / "rollouts"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: value for key, value in batch.items() if isinstance(value, np.ndarray)})
    arrays = directory / (operation_id + ".npz"); write_bytes(arrays, buffer.getvalue())
    records = batch["transition_records"]
    if not records: raise ValueError("Missing real neural sampling evidence")
    raw = ("\n".join(canonical(record) for record in records) + "\n").encode()
    trace = directory / (operation_id + ".jsonl.gz")
    write_bytes(trace, gzip.compress(raw, compresslevel=1, mtime=0))
    return {"arrays": bound(output, arrays), "trace": bound(output, trace)}


def train_to(output, trainer, ledger, branch, target):
    count = trainer.cfg["environments"]
    while trainer.joint_steps < target:
        if ledger.read()["branches"][branch]["ppo"]["pending"]:
            raise ValueError("Unconfirmed PPO reservation requires diagnosis; not replayed automatically")
        amount = min(count * trainer.cfg["rollout_steps"], target - trainer.joint_steps, ledger.remaining("ppo", branch))
        amount -= amount % count
        if amount <= 0: raise ValueError("Reserved budget cannot reach the fixed endpoint")
        op_id = "ppo_" + uuid.uuid4().hex
        operation = ledger.reserve("ppo", branch, amount, op_id, expected_step=trainer.joint_steps)
        if not operation["execution_permitted"]: raise ValueError("Duplicate reservation is not execution permission")
        start = time.monotonic()
        previous_steps = trainer.joint_steps
        batch, metrics = trainer.train_chunk(amount // count)
        if trainer.joint_steps != previous_steps + amount or len(batch["transition_records"]) != amount:
            raise ValueError("Actual training transitions disagree with the reserved chunk")
        if batch["audit"]["neural_overrides"] != 0: raise ValueError("NN action execution failed")
        if any(not np.isfinite(value) for value in metrics.values()): raise FloatingPointError("Nonfinite update metrics")
        evidence = save_batch(output, branch, op_id, batch)
        episodes = copy.deepcopy(trainer.completed_episodes); trainer.completed_episodes.clear()
        path = output / "branches" / branch / "checkpoints" / (f"step_{trainer.joint_steps:07d}_" + op_id + ".pt")
        payload = {"version": VERSION, "branch": branch, "operation_id": op_id,
            "trainer": trainer.state_dict(), "audit": batch["audit"], "metrics": metrics,
            "evidence": evidence, "completed_episodes": episodes, "actual_steps": amount,
            "elapsed_seconds": time.monotonic() - start}
        atomic_torch_save(path, payload)
        ledger.ack(op_id, str(path.relative_to(output)), file_hash(path), amount)
        write_json(output / "progress.json", {"status": "training", "branch": branch,
            "branch_ppo_steps": trainer.joint_steps, "target": target, "ledger": ledger.read(),
            "last_metrics": metrics, "last_chunk_seconds": payload["elapsed_seconds"]}, replace=True)
        print(json.dumps({"event": "ppo_ack", "branch": branch, "steps": trainer.joint_steps,
            "seconds": round(payload["elapsed_seconds"], 3), "neural_submitted": batch["audit"]["neural_submitted"]}), flush=True)


def export_actor(output, trainer, branch, head):
    path = output / "branches" / branch / "actors" / f"actor_{trainer.joint_steps:07d}.npz"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        record = json.loads(receipt.read_bytes())
        if record["checkpoint_sha256"] != head["sha256"] or file_hash(path) != record["actor"]["sha256"]:
            raise ValueError("Export differs from the confirmed checkpoint")
        return NumPyNativeActor(path)
    if path.exists(): raise ValueError("An unfinished export needs diagnosis")
    trainer.export(path); actor = NumPyNativeActor(path)
    obs, _ = trainer.arrays(); obs = obs.reshape(-1, 197)
    with torch.no_grad(): actual = trainer.model.actor_logits(torch.as_tensor(obs, device=trainer.device)).cpu().numpy()
    exported = actor.logits(obs); error = float(np.max(np.abs(actual - exported)))
    equal = bool(np.array_equal(actual.argmax(-1), exported.argmax(-1)))
    if error > 1e-4 or not equal: raise ValueError("Export parity failed")
    write_json(receipt, {"actor": bound(output, path), "checkpoint_sha256": head["sha256"],
        "observations": len(obs), "maximum_absolute_error": error, "argmax_equal": equal,
        "scope": "current saved environments only; not complete boundary qualification"})
    return actor


def evaluate_boundary(output, prepared, trainer, ledger, branch, scenarios):
    from .warehouse_native_partner_mix_evaluation import evaluate
    actor = export_actor(output, trainer, branch, ledger.head("ppo", branch))
    metadata = actor.metadata
    bindings = {key: metadata[key] for key in ("experiment_version", "branch", "protocol_sha256",
        "scenario_manifest_sha256", "initialization_sha256", "source_sha256", "source_checkpoint_sha256", "joint_steps")}
    bindings["actor_sha256"] = actor.artifact_sha256
    baselines = json.loads((output / "baselines.json").read_bytes())
    account = ledger.read()
    confirmed = {key for key, value in account["operations"].items()
                 if value["status"] == "acknowledged" and value["request"]["kind"] == "evaluation" and value["request"]["branch"] == branch}
    def reserve(context):
        op = ledger.reserve("evaluation", branch, 120, context["operation_id"])
        if not op["execution_permitted"]: raise ValueError("Evaluation request has already been reserved")
    def acknowledge(result):
        path = Path(result["row_path"])
        ledger.ack(result["operation_id"], str(path.relative_to(output)), result["row_sha256"], result["actual_steps"])
    report = evaluate(actor, scenarios["splits"]["validation"], trainer.protocol,
        output / "branches" / branch / "validation" / f"step_{trainer.joint_steps:07d}", 18000,
        expected_bindings=bindings, reference_report=baselines["reference"], random_report=baselines["random"],
        before_episode=reserve, on_episode=acknowledge, confirmed_operation_ids=confirmed)
    print(json.dumps({"event": "validation_complete", "branch": branch, "steps": trainer.joint_steps,
        "primary_value": report.get("primary_value"), "capability": report.get("capability")}), flush=True)
    return report


def advance(output, *, until):
    from .warehouse_native_partner_mix_budget import PartnerMixBudget
    output = Path(output).expanduser().resolve()
    if until not in (4096, 50000, 100000): raise ValueError("Use the registered probe or validation endpoints")
    prepared, protocol, scenarios, source = read_prepared(output)
    ledger = PartnerMixBudget(output, prepared["identity"])
    started = time.monotonic(); reports = {}
    with ledger.lease():
        for branch in BRANCHES:
            trainer = _trainer(protocol, scenarios, source, branch, prepared)
            head = ledger.head("ppo", branch)
            payload = decode(output / head["path"], head["sha256"])
            if payload["version"] != VERSION or payload["branch"] != branch: raise ValueError("Checkpoint branch differs")
            trainer.load_state_dict(payload["trainer"])
            if trainer.joint_steps != head["step"]: raise ValueError("Checkpoint and ledger disagree")
            if trainer.joint_steps > until: raise ValueError("Requested endpoint precedes the confirmed training head")
            for target in (4096, 50000, 100000):
                if target > until or target < trainer.joint_steps: continue
                train_to(output, trainer, ledger, branch, target)
                if target in prepared["registered_validation_endpoints"]:
                    reports[branch + "/" + str(target)] = evaluate_boundary(output, prepared, trainer, ledger, branch, scenarios)
            del trainer
        result = {"status": "fixed_endpoint_completed" if until == 100000 else "boundary_completed",
            "until": until, "elapsed_seconds": time.monotonic() - started, "ledger": ledger.read(),
            "validation_reports": {key: {k: v for k, v in value.items() if k not in ("rows", "artifacts")}
                                   for key, value in reports.items()}, "formal_ready": False, "website_model_changed": False}
        write_json(output / f"completion_{until:07d}.json", result, replace=True)
        write_json(output / "progress.json", result, replace=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--until", type=int, choices=(4096, 50000, 100000), default=100000)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = prepare(args.output, device=args.device) if args.prepare else advance(args.output, until=args.until)
    print(json.dumps({key: value for key, value in result.items() if key not in ("ledger", "validation_reports")}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
