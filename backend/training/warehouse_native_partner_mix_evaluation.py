"""Fixed observed-mode validation with compact, durable per-episode evidence.

This is a new experiment identity. It reuses the frozen physical episode loop,
not the old public-history experiment's checkpoint whitelist or control arm.
Saved reference/random rows are read only; no baseline is sampled here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

from backend.training.warehouse_native_common import canonical, digest
from backend.training import warehouse_native_public_feedback_evaluation as physical
from backend.training.warehouse_native_evaluation import summarize, capability
from backend.training.warehouse_native_feedback_run import warmup_capability
from backend.training.warehouse_native_public_feedback import HISTORY_FEATURE_NAMES
from env.warehouse.domain import collaborative_study_config
from env.warehouse.layouts import get_map_layout
from env.warehouse_native.observations import observation_names
from env.warehouse_native.policy import NumPyNativeActor, NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION

VERSION = "warehouse-native-partner-mix-evaluation.v1"
TRAINER_VERSION = "warehouse-native-partner-mix-trainer.v1"
PROTOCOL_VERSION = "warehouse-native-partner-mix-protocol.v1"
PARTNERS = physical.PARTNERS
BRANCHES = ("baseline_mix", "active_mix")
ROOT = Path(__file__).resolve().parents[2]
REQUIRED_BINDINGS = ("actor_sha256", "experiment_version", "branch", "protocol_sha256",
    "scenario_manifest_sha256", "initialization_sha256", "source_sha256", "source_checkpoint_sha256", "joint_steps")


def execution_sources():
    paths = set((ROOT / "env/warehouse").glob("*.py"))
    paths.update((ROOT / "env/warehouse_native").glob("*.py"))
    paths.update((ROOT / "core").glob("*.py"))
    for name in ("warehouse_native_common.py", "warehouse_native_evaluation.py", "warehouse_native_feedback_run.py",
                 "warehouse_native_public_feedback.py", "warehouse_native_public_feedback_evaluation.py",
                 "warehouse_native_revision_reward.py", "warehouse_native_v2.py"):
        paths.add(ROOT / "backend/training" / name)
    paths.add(Path(__file__))
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def _same(a, b):
    return canonical(a) == canonical(b)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _contract(actor, protocol, expected, fixture, config):
    if type(fixture) is not bool or type(protocol) is not dict or protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("An explicit partner-mix protocol is required")
    if protocol.get("test_fixture", False) is not fixture:
        raise ValueError("Protocol fixture provenance differs")
    if not fixture and type(actor) is not NumPyNativeActor:
        raise ValueError("Production validation requires the actual NumPy Actor")
    if type(expected) is not dict or not set(REQUIRED_BINDINGS) <= set(expected):
        raise ValueError("Independent Actor and experiment bindings are required")
    for key in REQUIRED_BINDINGS:
        if key.endswith("sha256") and not _sha(expected[key]):
            raise ValueError("Malformed expected hash: " + key)
    if (expected["protocol_sha256"] != digest(protocol) or expected["experiment_version"] != TRAINER_VERSION
            or expected["branch"] not in BRANCHES or type(expected["joint_steps"]) is not int
            or expected["joint_steps"] < 0):
        raise ValueError("Actor protocol, branch or additional-step binding differs")
    metadata = getattr(actor, "metadata", None)
    if type(metadata) is not dict or not callable(getattr(actor, "act", None)):
        raise ValueError("A bound Actor is required")
    required = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": 197, "state_dim": 354, "hidden": 128, "architecture": "two_hidden_layer_tanh",
        "actions": list(physical.ACTIONS), "action_masks": False, "runtime_action_override": False,
        "public_feedback_mode": "observed", "public_feedback_version": physical.OBSERVER_VERSION,
        "feature_names": list(observation_names(config)) + list(HISTORY_FEATURE_NAMES)}
    for name, value in required.items():
        if not _same(metadata.get(name), value):
            raise ValueError("Actor observation/action contract differs: " + name)
    if metadata.get("test_fixture", False) is not fixture:
        raise ValueError("Actor fixture provenance differs")
    for name, value in expected.items():
        got = getattr(actor, "artifact_sha256", None) if name == "actor_sha256" else metadata.get(name)
        if not _same(got, value):
            raise ValueError("Expected Actor binding differs: " + name)
    source = protocol.get("source", {})
    inherited = metadata.get("source_counters", {}).get("joint_steps")
    history = metadata.get("source_public_history_counters", {}).get("joint_steps")
    r1 = metadata.get("source_r1_counters", {}).get("joint_steps")
    if (any(type(v) is not int or v < 0 for v in (inherited, history, r1))
            or inherited != history + r1 or type(source.get("joint_steps")) is not int
            or source["joint_steps"] != history
            or source.get("state_sha256") != expected["initialization_sha256"]
            or source.get("checkpoint_sha256") != expected["source_checkpoint_sha256"]):
        raise ValueError("Inherited source counters or original checkpoint binding differ")
    if not fixture and (inherited, history, r1) != (380000, 100000, 280000):
        raise ValueError("Production warm start must preserve the observed380k ancestry")
    original = protocol.get("source_protocol", {})
    gates = {"candidate_gate": protocol.get("absolute_gate", original.get("absolute_gate")),
             "warmup_gate": protocol.get("warmup_gate", original.get("warmup_gate"))}
    if not _same(gates, {"candidate_gate": physical.ABSOLUTE_GATE, "warmup_gate": physical.WARMUP_GATE}):
        raise ValueError("Frozen capability and warmup gates cannot change")
    if not _same(protocol.get("reward"), physical.REWARD) or protocol.get("collision_training_cost") != .05:
        raise ValueError("Validation reward revision differs")
    evaluation = protocol.get("evaluation", {})
    if (evaluation.get("partners") != list(PARTNERS) or evaluation.get("deterministic") is not True
            or evaluation.get("scenarios_per_partner", evaluation.get("episodes_per_partner")) != 50
            or evaluation.get("read_final_test") is not False):
        raise ValueError("Only the original fixed deterministic validation matrix is permitted")
    if not fixture and config != collaborative_study_config():
        raise ValueError("Production validation configuration cannot change")
    if type(config.horizon) is not int or not 1 <= config.horizon <= 120:
        raise ValueError("Invalid validation horizon")
    return deepcopy(metadata), gates, inherited


def _put(path, value):
    raw = canonical(value).encode()
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        if temporary.exists(): temporary.unlink()
    return {"path": path.name, "sha256": sha256(raw).hexdigest(), "size": len(raw)}


def _file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Episode evidence must be a regular local file")
    raw = path.read_bytes()
    return {"path": path.name, "sha256": sha256(raw).hexdigest(), "size": len(raw)}


@contextmanager
def _locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise ValueError("Evaluation output is already in use") from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_entry(output, entry, context):
    if entry.get("status") != "completed" or not _same(entry.get("context"), context):
        raise ValueError("Incomplete or differently bound episode cannot be resampled")
    for role in ("row", "trace"):
        item = entry.get(role, {})
        if type(item) is not dict or set(item) != {"path", "sha256", "size"}:
            raise ValueError("Missing durable episode artifact")
        if Path(item["path"]).name != item["path"] or _file(output / item["path"]) != item:
            raise ValueError("Durable episode artifact changed")
    row = json.loads((output / entry["row"]["path"]).read_text())
    if (any(not _same(row.get(k), v) for k, v in context.items())
            or row.get("steps") != entry.get("actual_steps") or "trace" in row or "physical_trace" in row):
        raise ValueError("Durable episode row differs from its context")
    return row


def _summary(rows, metadata, bindings, gates, reference, random, fixture, identity, reserved):
    summaries = {}
    for partner in PARTNERS:
        selected = [row for row in rows if row["partner"] == partner]
        item = summarize(selected)
        steps = sum(r["steps"] for r in selected)
        if selected:
            item.update(environment_steps=steps, collisions_per_step=sum(r["collisions"] for r in selected) / steps,
                mean_program_deliveries=float(np.mean([r["program_deliveries"] for r in selected])),
                mean_longest_consecutive_collisions=float(np.mean([r["longest_consecutive_collisions"] for r in selected])),
                max_consecutive_collisions=max(r["longest_consecutive_collisions"] for r in selected),
                mean_longest_steps_without_task_progress=float(np.mean([r["longest_steps_without_task_progress"] for r in selected])),
                full_battery_waits_by_role=[sum(r["full_battery_waits_by_role"][i] for r in selected) for i in (0, 1)],
                full_battery_charger_waits_by_role=[sum(r["full_battery_charger_waits_by_role"][i] for r in selected) for i in (0, 1)],
                active_end_rate_by_role=[float(np.mean([r["active_end_by_role"][i] for r in selected])) for i in (0, 1)],
                double_wait_steps=sum(r["double_wait_steps"] for r in selected),
                joint_submitted_wait_steps=sum(r["joint_submitted_wait_steps"] for r in selected),
                wall_invalid_by_role=[sum(r["wall_invalid_by_role"][i] for r in selected) for i in (0, 1)])
        summaries[partner] = item
    complete = len(rows) == identity["episode_count"]
    inherited = metadata["source_counters"]["joint_steps"]
    report = {"version": VERSION, "status": "completed" if complete else "incomplete",
        "summary": summaries, "rows": rows, "mode": "observed", "branch": bindings["branch"],
        "actor_bindings": deepcopy(bindings), "identity": identity, "episodes": len(rows),
        "primary_metric": "equal_partner_mean_of_nn_deliveries",
        "primary_value": float(np.mean([s["mean_ai_deliveries"] for s in summaries.values()])) if complete else None,
        "source_joint_steps": inherited, "additional_joint_steps": bindings["joint_steps"],
        "total_actor_training_joint_steps": inherited + bindings["joint_steps"],
        "environment_steps": sum(r["steps"] for r in rows), "reserved_environment_steps": reserved,
        "nn_action_overrides": sum(r["nn_action_overrides"] for r in rows),
        "raw_neural_actions_submitted": sum(r["raw_neural_actions_submitted"] for r in rows),
        "training_steps": 0, "optimizer_updates": 0, "final_test_read": False, "formal_ready": False,
        "test_fixture": fixture, "deterministic_actor": True,
        "qualification_evaluated": complete and reference is not None,
        "action_audit": "program_decides_first; deterministic_NN_argmax_submitted_unchanged; physical_cancellations_separate"}
    if complete and reference is not None:
        report["capability"] = capability(report, reference, random, gates)
        report["warmup_capability"] = warmup_capability(report, reference, gates, inherited + bindings["joint_steps"])
        if fixture:
            report["capability"]["eligible"] = report["warmup_capability"]["eligible"] = False
            report["capability"]["status"] = "test_fixture_not_a_candidate"
    else:
        report["capability"] = {"eligible": False, "status": "incomplete_or_baselines_missing"}
        report["warmup_capability"] = {"eligible": False, "status": "incomplete_or_baselines_missing"}
    return report


def evaluate(actor_or_path, scenarios, protocol, output, step_budget, *, expected_bindings,
             reference_report, random_report, before_episode=None, on_episode=None,
             confirmed_operation_ids=None, allow_test_fixture=False, config=None):
    """Evaluate a new bound Actor on the unchanged development validation set.

    ``step_budget`` caps cumulative non-refundable horizon reservations in this
    output, not just the latest call. ``before_episode`` runs after the local
    reservation and before reset; the caller reserves its own ledger there.
    ``on_episode`` receives durable absolute row/trace paths and hashes before
    it acknowledges the external ledger. On resume every saved complete row
    must appear in caller-authenticated ``confirmed_operation_ids``. Incomplete
    reservations are never retried or refunded by this evaluator.
    """
    if type(step_budget) is not int or step_budget < 0:
        raise ValueError("An integer validation step budget is required")
    if any(callback is not None and not callable(callback) for callback in (before_episode, on_episode)):
        raise ValueError("Episode callbacks must be callable")
    if confirmed_operation_ids is not None and (not isinstance(confirmed_operation_ids, (set, frozenset, list, tuple))
            or any(not isinstance(op, str) for op in confirmed_operation_ids)):
        raise ValueError("Confirmed operation IDs must come from the retained external ledger")
    config = config or collaborative_study_config()
    actor_path = Path(actor_or_path).resolve() if isinstance(actor_or_path, (str, Path)) else None
    actor = NumPyNativeActor(actor_path) if actor_path is not None else actor_or_path
    metadata, gates, inherited = _contract(actor, protocol, expected_bindings, allow_test_fixture, config)
    if actor_path is None and type(actor) is NumPyNativeActor:
        actor_path = Path(actor.path).resolve()
    scenes = physical._scenarios(scenarios, allow_test_fixture, config)
    if (reference_report is None) != (random_report is None) or (not allow_test_fixture and reference_report is None):
        raise ValueError("Both retained baselines are required in production")
    reference = physical._baseline(reference_report, scenes) if reference_report is not None else None
    random = physical._baseline(random_report, scenes) if random_report is not None else None
    sources = execution_sources()
    identity = {"version": VERSION, "actor_bindings": deepcopy(expected_bindings), "actor_metadata_sha256": digest(metadata),
        "validation_entries_sha256": digest(scenes), "protocol_sha256": digest(protocol), "sources": sources,
        "configuration": asdict(config), "episode_count": len(scenes) * len(PARTNERS), "step_budget": step_budget,
        "reference_sha256": digest(reference) if reference is not None else None,
        "random_sha256": digest(random) if random is not None else None, "test_fixture": allow_test_fixture}
    contexts = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            context = {"evaluation_version": VERSION, "mode": "observed", "branch": expected_bindings["branch"],
                "partner": partner, "scenario_id": scene["id"], "scenario_index": index,
                "initial_fingerprint": scene["fingerprint"], "episode_index": len(contexts), "seed": 17000 + index,
                "horizon": config.horizon, "maximum_environment_steps": config.horizon,
                "test_fixture": allow_test_fixture, "actor_bindings": deepcopy(expected_bindings)}
            context["operation_id"] = digest({"evaluation": digest(identity), "episode": context})
            contexts.append(context)
    output = Path(output).expanduser().absolute()
    if output.is_symlink(): raise ValueError("Evaluation output cannot be a symlink")
    output = output.resolve()
    with _locked(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not _same(manifest.get("identity"), identity):
                raise ValueError("Existing evaluation bindings or reserved budget differ")
        else:
            if set(p.name for p in output.iterdir()) != {".lock"}:
                raise ValueError("Existing unbound evaluation output cannot be adopted")
            manifest = {"version": VERSION, "identity": identity, "episodes": [], "reserved_environment_steps": 0,
                "actual_environment_steps": 0, "status": "running"}
            _put(manifest_path, manifest)
        entries = manifest.get("episodes")
        if not isinstance(entries, list) or len(entries) > len(contexts):
            raise ValueError("Invalid saved episode matrix")
        if manifest.get("reserved_environment_steps") != len(entries) * config.horizon:
            raise ValueError("Reserved step accounting differs")
        rows = []
        for entry, context in zip(entries, contexts):
            rows.append(_read_entry(output, entry, context))
            if confirmed_operation_ids is None or context["operation_id"] not in confirmed_operation_ids:
                raise ValueError("Saved complete episode requires external ledger acknowledgment before continuing")
        if manifest.get("actual_environment_steps") != sum(r["steps"] for r in rows):
            raise ValueError("Confirmed actual step accounting differs")
        dynamics = {"reward": deepcopy(protocol["reward"]), "initialization": {"source_joint_steps": inherited}}
        for context in contexts[len(rows):]:
            if manifest["reserved_environment_steps"] + config.horizon > step_budget:
                break
            if execution_sources() != sources or not _same(actor.metadata, metadata):
                raise ValueError("Actor metadata or execution source changed before an episode")
            if actor_path is not None and sha256(actor_path.read_bytes()).hexdigest() != expected_bindings["actor_sha256"]:
                raise ValueError("Frozen Actor bytes changed before an episode")
            entry = {"context": context, "status": "reserved", "actual_steps": None}
            manifest["episodes"].append(entry)
            manifest["reserved_environment_steps"] += config.horizon
            _put(manifest_path, manifest)
            admitted = 0; completed = 0; full = [0, 0]; charger_full = [0, 0]
            double = joint_wait = 0; last_after = None
            name = f"{context['episode_index']:04d}_{context['partner']}_{context['scenario_id']}"
            trace_path = output / (name + ".jsonl.gz")
            temporary = output / (name + ".jsonl.gz.incomplete")
            row_path = output / (name + ".json")
            try:
                physical._callback(before_episode, context)
                with temporary.open("xb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as compressed:
                        def before_step(value):
                            nonlocal admitted
                            if admitted >= config.horizon: raise ValueError("Episode would exceed its permanent reservation")
                            admitted += 1
                        def on_step(record):
                            nonlocal completed, double, joint_wait, last_after
                            completed += 1
                            if record["submitted_actions"]["robot_2"] != record["policy_actions"]["robot_2"]:
                                raise ValueError("The NN command was overwritten")
                            agents = record["before"]["agents"]
                            charger = tuple(get_map_layout(config.map_layout_id).charger_position)
                            for i, key in enumerate(("robot_1", "robot_2")):
                                is_full_wait = (agents[i]["active"] and agents[i]["battery"] >= 100. - 1e-8
                                                and record["submitted_actions"][key] == "WAIT")
                                full[i] += int(is_full_wait)
                                charger_full[i] += int(is_full_wait and tuple(agents[i]["position"]) == charger)
                            both = all(record["submitted_actions"][key] == "WAIT" for key in ("robot_1", "robot_2"))
                            joint_wait += int(both); double += int(both and all(a["active"] for a in agents))
                            last_after = record["after"]
                            compressed.write(canonical(record).encode() + b"\n")
                        row = physical._episode(actor, scenes[context["scenario_index"]], context["partner"],
                            "observed", dynamics, config, context, before_step, on_step, False)
                    raw.flush(); os.fsync(raw.fileno())
                if admitted != completed or row["steps"] != completed:
                    raise ValueError("Physical steps and written trace rows differ")
                os.replace(temporary, trace_path)
                descriptor = os.open(output, os.O_RDONLY)
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
                row.update(full_battery_waits_by_role=full, full_battery_charger_waits_by_role=charger_full,
                    double_wait_steps=double, joint_submitted_wait_steps=joint_wait,
                    active_end_by_role=[bool(a["active"]) for a in last_after["agents"]],
                    wait_metric_definition="Full battery>=100 before an active role submits WAIT; double_wait requires both roles active.")
                row_item = _put(row_path, row)
                committed = deepcopy(manifest)
                committed["episodes"][-1].update(status="completed", actual_steps=completed,
                    row=row_item, trace=_file(trace_path))
                committed["actual_environment_steps"] += completed
                _put(manifest_path, committed)
                manifest = committed
                entry = manifest["episodes"][-1]
                rows.append(row)
                physical._callback(on_episode, {**context, "actual_steps": completed,
                    "row_path": str(row_path), "row_sha256": row_item["sha256"],
                    "trace_path": str(trace_path), "trace_sha256": entry["trace"]["sha256"]})
            except BaseException as error:
                # A completed row remains durable even if the caller's ack
                # fails. It cannot be skipped later without an external ack.
                if entry["status"] != "completed":
                    entry.update(status="incomplete", actual_steps=completed,
                        admitted_step_calls=admitted, completed_trace_rows=completed)
                    manifest["actual_environment_steps"] += completed
                manifest.update(status="interrupted", error=f"{type(error).__name__}: {error}")
                _put(manifest_path, manifest)
                raise
        manifest["status"] = "completed" if len(rows) == len(contexts) else "budget_exhausted"
        _put(manifest_path, manifest)
        report = _summary(rows, metadata, expected_bindings, gates, reference, random,
            allow_test_fixture, identity, manifest["reserved_environment_steps"])
        report["remaining_reservable_steps"] = step_budget - manifest["reserved_environment_steps"]
        _put(output / "report.json", report)
        return report
