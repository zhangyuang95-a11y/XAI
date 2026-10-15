"""Fixed held-out robot-2 fidelity and effective one-step intervention audit.

Real registered runtime families only. Saved-record verification recalculates
statistics and tree outputs; it is explicitly not a second NN/physics replay.
This component never grants participant, explanation or release qualification.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
import gzip
import json
import os
import re

import numpy as np

from backend import warehouse_runtime_family as registry
from backend import warehouse_family_explanation as renderer
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from core.program import ExecutableProgram
from env.warehouse.layouts import get_map_layout
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS

VERSION = "warehouse-native-family-heldout-explanation-audit.v2"
PARTNERS = ("skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
AGENT_PHYSICS = ("agent_id", "position", "battery", "active", "carrying_task_id", "deliveries_completed",
                 "last_battery_delta", "steps_since_charging", "charger_wait_streak")
TASK_PHYSICS = ("task_id", "pickup_position", "delivery_position", "status", "carrier_agent_id",
                "created_frame", "claimed_frame", "delivered_frame")
JOINT_PHYSICS = ("next_task_index", "total_deliveries", "terminated", "truncated", "terminal_reason")


def contract():
    return {"version": VERSION, "split": "explanation_test", "scenes": 100,
        "partners": list(PARTNERS), "horizon": 120, "seed": "17000+scenario_index",
        "evaluated_role": "robot_2", "base_step_cap": 36000, "counterfactual_step_cap": 18000,
        "total_step_cap": 54000, "intervene": "preframe%10==0 and critical_groups nonempty",
        "intervention_actions": list(ACTIONS), "branch_steps": 1,
        "effective_pair": "physical_projection differs AND next NN action differs from WAIT branch",
        "physical_projection": {"agents": list(AGENT_PHYSICS), "tasks_and_completed_tasks": list(TASK_PHYSICS),
            "joint": list(JOINT_PHYSICS), "excluded": "command/history/heading/collision-counter/score logs"},
        "direction_correct": "tree agrees with NN at both WAIT and changed endpoints",
        "neural_action_selection": "NumPyNativeActor.act deterministic: float32 softmax probabilities.argmax; logits argmax diagnostic only",
        "fidelity": .90, "critical_fidelity": .85, "nonwait_fidelity": .85,
        "critical_nonwait_fidelity": .85, "critical_scenes": 10, "critical_nonwait_scenes": 10,
        "direction_fidelity": .85, "effective_scenes_per_group": 10,
        "empty_evidence_passes": False, "tree_controls_runtime": False,
        "qualification_evaluated": False}


def execution_sources():
    sources = renderer.explanation_sources()
    sources[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    for name in ("backend/training/warehouse_native_evaluation.py", "env/warehouse_native/partners.py",
                 "env/warehouse/layouts.py"):
        value = file_hash(ROOT / name)
        if name in sources and sources[name] != value: raise ValueError("Audit source closure differs")
        sources[name] = value
    return sources


def input_bindings(runtime, program_path, scenarios):
    identity = registry.verify(runtime, allow_test_fixture=runtime.test_fixture)
    return {"runtime_signature": runtime.signature, "runtime_family": identity["family"],
        "runtime_version": identity["runtime_version"], "actor_sha256": runtime.actor_sha256,
        "actor_metadata_sha256": digest(runtime.actor.metadata), "program_sha256": file_hash(program_path),
        "protocol_sha256": runtime.protocol_sha256, "scenario_manifest_sha256": digest(scenarios),
        "contract_sha256": digest(contract()), "sources_sha256": digest(execution_sources())}


def _equal(a, b, message):
    if digest(a) != digest(b): raise ValueError(message)


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _put(path, value, *, replace=False):
    raw = canonical(value).encode()
    if path.suffix == ".gz": raw = gzip.compress(raw, compresslevel=1, mtime=0)
    target = path.with_name(path.name + ".pending") if replace else path
    with target.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if replace: os.replace(target, path)
    _sync(path.parent)


def _read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def physical_projection(view):
    return {"agents": [{key: a[key] for key in AGENT_PHYSICS} for a in view["agents"]],
        **{pool: [{key: task[key] for key in TASK_PHYSICS} for task in view[pool]] for pool in ("tasks", "completed_tasks")},
        **{key: view[key] for key in JOINT_PHYSICS}}


def _groups(snapshot):
    """Reapply the public geometry definition to saved state, no environment."""
    state = snapshot["state"]
    agents = [SimpleNamespace(**{**x, "position": tuple(x["position"])}) for x in state["agents"]]
    tasks = [SimpleNamespace(**{**x, "pickup_position": tuple(x["pickup_position"]),
        "delivery_position": tuple(x["delivery_position"])}) for x in state["tasks"]]
    public = SimpleNamespace(agents=agents, tasks=tasks,
        by_id=lambda role: next(a for a in agents if a.agent_id == role))
    config = SimpleNamespace(**snapshot["configuration"])
    return critical_groups(SimpleNamespace(state=public, config=config,
        layout=get_map_layout(config.map_layout_id)), "robot_2")


class AuditFailure(ValueError):
    def __init__(self, reason, report):
        super().__init__(reason); self.report = report


class _Meter:
    def __init__(self, root, execution_id, before, after, caps):
        self.root, self.execution_id, self.before, self.after = root, execution_id, before, after
        self.caps = caps; self.pending = None; self.entries = []
        self.counts = {"base_steps": 0, "counterfactual_steps": 0, "base_attempts": 0,
            "counterfactual_attempts": 0, "acknowledged_steps": 0, "numpy_logits_calls": 0,
            "numpy_logits_rows": 0, "neural_updates": 0, "tree_fits": 0, "torch_loads": 0}

    def save_pending(self): _put(self.root / "pending.json", self.pending, replace=True)

    def step(self, env, runtime, program, before_decision, player, context):
        phase = context["phase"]
        if self.pending is not None: raise ValueError("unacknowledged_step_no_retry")
        if self.counts[phase + "_attempts"] >= self.caps[phase]: raise ValueError("phase_budget_exhausted")
        index = len(self.entries)
        before = env.snapshot(); before_view = env.public_view()
        submitted = {"robot_1": player, "robot_2": before_decision["neural_action"]}
        ctx = {**context, "operation_id": f"{self.execution_id}:{phase}:{index:06d}",
            "before_sha256": digest(before), "submitted_actions": submitted, "reserved_steps": 1}
        self.pending = {**ctx, "status": "awaiting_permission"}; self.save_pending()
        if self.before is not None and self.before(deepcopy(ctx)) is not True:
            raise ValueError("before_step_not_permitted")
        self.pending["status"] = "reserved"; self.save_pending()
        self.counts[phase + "_attempts"] += 1
        _, rewards, terminated, truncated, info = env.step(submitted)
        self.counts[phase + "_steps"] += 1
        self.pending.update(status="executed_unacknowledged", actual_steps=1); self.save_pending()
        _equal(info["requested_actions"], submitted, "NN_submission_changed")
        record = {**ctx, "before": before, "after": env.snapshot(), "before_view": before_view,
            "after_view": env.public_view(), "decision": before_decision,
            "groups": _groups(before),
            "executed_actions": info["executed_actions"], "events": info["events"],
            "info": info, "rewards": rewards, "terminated": bool(terminated), "truncated": bool(truncated),
            "next_decision": (_decision(env, runtime, program, self)
                if phase == "counterfactual" and not env.done else None)}
        path = self.root / "steps" / f"{index:06d}.json.gz"
        _put(path, record)
        entry = {"path": str(path.relative_to(self.root)), "sha256": file_hash(path),
            "operation_id": ctx["operation_id"], "phase": phase, "actual_steps": 1}
        completed = {**ctx, "actual_steps": 1, "record_path": str(path.resolve()),
            "record_sha256": entry["sha256"], "status": "executed_unacknowledged"}
        self.pending = completed; self.save_pending()
        if self.after is not None and self.after(deepcopy(completed)) is not True:
            raise ValueError("after_step_not_acknowledged")
        _put(self.root / "acks" / f"{index:06d}.json", entry)
        self.entries.append(entry); self.counts["acknowledged_steps"] += 1
        (self.root / "pending.json").unlink(); _sync(self.root)
        self.pending = None
        return record

    def report(self):
        return {"counts": deepcopy(self.counts), "pending_operation": deepcopy(self.pending),
            "automatic_retry": False, "caps": self.caps,
            "accounting_complete": self.pending is None}


def _decision(env, runtime, program, meter):
    before = digest(env.snapshot()); roles = sorted(env.agent_ids)
    observations = env.observations(); matrix = np.stack([observations[r] for r in roles])
    meter.counts["numpy_logits_calls"] += 1; meter.counts["numpy_logits_rows"] += len(roles)
    logits = runtime.actor.logits(matrix)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True)); probabilities /= probabilities.sum(axis=1, keepdims=True)
    index = roles.index("robot_2"); obs = observations["robot_2"].astype(np.float32)
    features = dict(zip(env.feature_names, map(float, obs)))
    result = {"frame": env.state.frame, "role": "robot_2", "observation": obs.tolist(),
        "observation_sha256": digest(obs.tolist()), "logits": logits[index].tolist(),
        "probabilities": probabilities[index].tolist(), "neural_action": ACTIONS[int(probabilities[index].argmax())],
        "logits_argmax_diagnostic": ACTIONS[int(logits[index].argmax())],
        "tree_action": program.predict(features), "tree_probabilities": program.predict_proba(features),
        "actor_sha256": runtime.actor_sha256, "runtime_signature": runtime.signature}
    if digest(env.snapshot()) != before: raise ValueError("decision_mutated_state")
    return result


def _preflight(runtime, program_path, scenarios, bindings, allow_fixture, permitted, before, after, execution_id):
    if permitted is not True: raise ValueError("execution_not_explicitly_permitted")
    identity = registry.verify(runtime, allow_test_fixture=allow_fixture)
    if type(execution_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", execution_id):
        raise ValueError("execution_id_required")
    if (before is None) != (after is None) or (before is not None and (not callable(before) or not callable(after))):
        raise ValueError("both_callbacks_required")
    if not allow_fixture and before is None: raise ValueError("production_durable_callbacks_required")
    _equal(bindings, input_bindings(runtime, program_path, scenarios), "external_bindings_differ")
    if scenarios.get("test_fixture", False) is not allow_fixture: raise ValueError("fixture_scope_differs")
    _equal(scenarios["configuration"], asdict(runtime.config), "configuration_differs")
    scenes = scenarios["splits"]["explanation_test"]
    if not scenes or (not allow_fixture and (len(scenes) != 100 or runtime.config.horizon != 120)):
        raise ValueError("fixed_100_scene_matrix_required")
    if allow_fixture and (len(scenes) > 2 or runtime.config.horizon > 10):
        raise ValueError("fixture_requires_at_most_two_short_scenes")
    if not allow_fixture and digest(scenarios) != runtime.actor.metadata["scenario_manifest_sha256"]:
        raise ValueError("registered_manifest_differs")
    ids = {s["id"] for s in scenes}; fingerprints = {s["fingerprint"] for s in scenes}
    if len(ids) != len(scenes) or len(fingerprints) != len(scenes): raise ValueError("duplicate_scenes")
    if any(not s["id"].startswith("explanation_test_") or s["snapshot"]["state"]["frame"] != 0 for s in scenes):
        raise ValueError("original_heldout_starts_required")
    excluded = {s["fingerprint"] for name, pool in scenarios["splits"].items() if name != "explanation_test" for s in pool}
    if fingerprints & excluded: raise ValueError("heldout_pool_overlap")
    explainer = renderer.FamilyExplainer(program_path, expected_program_sha256=bindings["program_sha256"],
        runtime=runtime, allow_test_fixture=allow_fixture)
    return identity, deepcopy(scenes), explainer.program


def audit(runtime, program_path, scenarios, *, expected_bindings, output, execution_id,
          execution_permitted=False, allow_test_fixture=False, before_step=None, after_step=None):
    identity, scenes, program = _preflight(runtime, program_path, scenarios, expected_bindings,
        allow_test_fixture, execution_permitted, before_step, after_step, execution_id)
    output = Path(output).resolve()
    if output.exists(): raise ValueError("audit_output_already_exists_no_retry")
    caps = {"base": 36000, "counterfactual": 18000}
    if allow_test_fixture: caps = {"base": len(scenes) * 3 * runtime.config.horizon, "counterfactual": len(scenes) * 3 * 5}
    output.mkdir(parents=True); (output / "steps").mkdir(); (output / "acks").mkdir(); _sync(output.parent)
    meter = _Meter(output, execution_id, before_step, after_step, caps)
    header = {"version": VERSION, "test_fixture": allow_test_fixture, "bindings": deepcopy(expected_bindings),
        "contract": contract(), "configuration": asdict(runtime.config), "sources": execution_sources(),
        "scenes": scenes, "program": program.to_dict(), "program_file_text": Path(program_path).read_bytes().decode("utf-8"),
        "qualification_evaluated": False, "explanation_eligible": False, "participant_enabled": False}
    _put(output / "inputs.json", header)
    try:
        private = registry.fresh_instance(runtime, allow_test_fixture=allow_test_fixture,
            expected_family=identity["family"], expected_signature=runtime.signature)
        for partner in PARTNERS:
            for index, scene in enumerate(scenes):
                env = private.environment(scene); rng = np.random.default_rng(17000 + index)
                while not env.done:
                    snapshot = env.snapshot(); decision = _decision(env, private, program, meter)
                    groups = critical_groups(env, "robot_2")
                    player = partner_action(env, "robot_1", partner, rng)
                    _equal(snapshot, env.snapshot(), "program_partner_mutated_environment")
                    context = {"phase": "base", "partner": partner, "scenario_index": index,
                        "scenario_id": scene["id"], "initial_fingerprint": scene["fingerprint"],
                        "seed": 17000 + index, "frame": env.state.frame}
                    meter.step(env, private, program, decision, player, context)
                    if snapshot["state"]["frame"] % 10 == 0 and groups:
                        for action in ACTIONS:
                            branch = private.from_snapshot(snapshot)
                            meter.step(branch, private, program, decision, action,
                                {**context, "phase": "counterfactual", "intervention_action": action})
                    if env.state.frame > private.config.horizon: raise ValueError("trajectory_exceeds_horizon")
        _equal(input_bindings(runtime, program_path, scenarios), expected_bindings, "inputs_changed_during_audit")
        registry.verify(private, allow_test_fixture=allow_test_fixture, expected_signature=runtime.signature)
        manifest = {"version": VERSION, "header_sha256": file_hash(output / "inputs.json"),
            "entries": meter.entries, "execution": meter.report(), "qualified": False}
        # Verify raw records and matrix before marking completed; reader adds no NN/physics steps.
        report = _recompute(output, header, manifest)
        _put(output / "report.json", report); manifest["report_sha256"] = file_hash(output / "report.json")
        _put(output / "manifest.json", manifest)
        return {**report, "manifest_sha256": file_hash(output / "manifest.json"), "output": str(output)}
    except BaseException as error:
        failure = {"version": VERSION, "reason": str(error), "execution": meter.report(),
            "qualification_evaluated": False, "passed": False, "automatic_retry": False}
        try: _put(output / "failed.json", failure)
        except Exception: failure["failure_report_persistence_failed"] = True
        if not isinstance(error, Exception): error.audit_report = failure; raise
        raise AuditFailure(str(error), failure) from error


def _check_decision(record, program, bindings):
    if record is None: return
    obs = np.asarray(record["observation"], np.float32); logits = np.asarray(record["logits"], np.float32)
    probs = np.asarray(record["probabilities"], np.float32)
    if obs.shape != (197,) or logits.shape != (5,) or probs.shape != (5,) or not all(np.isfinite(x).all() for x in (obs, logits, probs)):
        raise ValueError("invalid_raw_NN_evidence")
    expected = np.exp(logits - logits.max()); expected /= expected.sum()
    if not np.allclose(probs, expected, atol=1e-7, rtol=1e-6): raise ValueError("stored_distribution_differs_from_logits")
    if record["neural_action"] != ACTIONS[int(expected.argmax())]: raise ValueError("stored_action_differs_from_probability_argmax")
    if record["logits_argmax_diagnostic"] != ACTIONS[int(logits.argmax())]: raise ValueError("logits_diagnostic_differs")
    features = dict(zip(program.feature_names, map(float, obs)))
    _equal(record["tree_action"], program.predict(features), "saved_tree_prediction_differs")
    _equal(record["tree_probabilities"], program.predict_proba(features), "saved_tree_distribution_differs")
    if (record["role"] != "robot_2" or record["actor_sha256"] != bindings["actor_sha256"]
            or record["runtime_signature"] != bindings["runtime_signature"]
            or record["observation_sha256"] != digest(obs.tolist())): raise ValueError("decision_binding_differs")


def _rate(rows, key): return sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.


def _stats(rows, key):
    return {"rows": len(rows), "scenes": len({r["fingerprint"] for r in rows}), "fidelity": _rate(rows, key)}


def _summarize(rows, pairs, full):
    fidelity = {"overall": _stats(rows, "correct"),
        "nonwait": _stats([r for r in rows if r["action"] != "WAIT"], "correct"),
        "by_action": {a: _stats([r for r in rows if r["action"] == a], "correct") for a in ACTIONS},
        "by_group": {g: _stats([r for r in rows if g in r["groups"]], "correct") for g in GROUPS},
        "by_group_nonwait": {g: _stats([r for r in rows if g in r["groups"] and r["action"] != "WAIT"], "correct") for g in GROUPS},
        "by_partner": {p: _stats([r for r in rows if r["partner"] == p], "correct") for p in PARTNERS}}
    effective = [r for r in pairs if r["physical_effect"] and r["nn_changed"]]
    direction = {"eligible_pairs": len(effective), "all_pairs": len(pairs),
        "no_physical_effect_pairs": sum(not r["physical_effect"] for r in pairs),
        "unchanged_NN_pairs": sum(not r["nn_changed"] for r in pairs),
        "overall": _stats(effective, "correct"),
        "by_group": {g: _stats([r for r in effective if g in r["groups"]], "correct") for g in GROUPS},
        "by_player_action": {a: _stats([r for r in effective if r["action"] == a], "correct") for a in ACTIONS},
        "by_partner": {p: _stats([r for r in effective if r["partner"] == p], "correct") for p in PARTNERS}}
    checks = {"fidelity_overall": bool(rows) and fidelity["overall"]["fidelity"] >= .9,
        "fidelity_nonwait": fidelity["nonwait"]["rows"] > 0 and fidelity["nonwait"]["fidelity"] >= .85,
        "direction_overall": bool(effective) and direction["overall"]["fidelity"] >= .85,
        "full_production_matrix": full}
    for group in GROUPS:
        checks["fidelity_" + group] = fidelity["by_group"][group]["scenes"] >= 10 and fidelity["by_group"][group]["fidelity"] >= .85
        checks["fidelity_nonwait_" + group] = fidelity["by_group_nonwait"][group]["scenes"] >= 10 and fidelity["by_group_nonwait"][group]["fidelity"] >= .85
        checks["direction_" + group] = direction["by_group"][group]["scenes"] >= 10 and direction["by_group"][group]["fidelity"] >= .85
    return {"fidelity": fidelity, "intervention_direction": direction, "checks": checks,
        "passed": all(checks.values()), "evaluated_role": "robot_2", "robot_1_policy_fidelity_evaluated": False,
        "qualification_evaluated": False, "explanation_eligible": False, "participant_enabled": False,
        "no_effect_pairs_counted_as_success": False}


def _recompute(output, header, manifest):
    """Hash-bound saved facts only; neither restore/step nor NN load/forward."""
    _equal(header["contract"], contract(), "contract_changed")
    _equal(header["sources"], execution_sources(), "sources_changed")
    if digest(header["sources"]) != header["bindings"]["sources_sha256"]: raise ValueError("source_binding_changed")
    if header["version"] != VERSION or manifest["version"] != VERSION: raise ValueError("audit_version_differs")
    if any(header.get(k) is not False for k in ("qualification_evaluated", "explanation_eligible", "participant_enabled")):
        raise ValueError("component_cannot_grant_qualification")
    if sha256(header["program_file_text"].encode()).hexdigest() != header["bindings"]["program_sha256"]:
        raise ValueError("original_program_bytes_differ")
    program = ExecutableProgram.from_dict(header["program"])
    raw = json.loads(header["program_file_text"])
    _equal(program.to_dict(), raw["program"] if raw.get("version") == "warehouse_native_rcpd_feedback_v1" else raw, "program_wrapper_differs")
    entries = manifest["entries"]; base = {}; anchors = {}; branches = {}; rows = []; physical_steps = {"base": 0, "counterfactual": 0}
    for index, entry in enumerate(entries):
        relative = f"steps/{index:06d}.json.gz"
        if entry["path"] != relative or entry["actual_steps"] != 1: raise ValueError("noncanonical_step_entry")
        path = output / relative
        if path.is_symlink() or file_hash(path) != entry["sha256"]: raise ValueError("step_hash_differs")
        _equal(_read(output / "acks" / f"{index:06d}.json"), entry, "durable_ack_differs")
        record = _read(path); phase = record["phase"]
        if phase not in physical_steps or record["operation_id"] != entry["operation_id"] or phase != entry["phase"]:
            raise ValueError("operation_binding_differs")
        physical_steps[phase] += 1
        if digest(record["before"]) != record["before_sha256"]: raise ValueError("before_hash_differs")
        before, after = record["before"]["state"], record["after"]["state"]
        if before["terminated"] or before["truncated"] or after["frame"] != before["frame"] + 1:
            raise ValueError("invalid_transition_frames")
        if record["frame"] != before["frame"] or record["decision"]["frame"] != before["frame"]:
            raise ValueError("decision_frame_differs")
        _equal(record["groups"], _groups(record["before"]), "saved_groups_differ")
        _check_decision(record["decision"], program, header["bindings"])
        _check_decision(record["next_decision"], program, header["bindings"])
        if (record["submitted_actions"]["robot_2"] != record["decision"]["neural_action"]
                or record["info"]["requested_actions"] != record["submitted_actions"]): raise ValueError("NN_override_in_saved_trace")
        _equal(record["executed_actions"], record["info"]["executed_actions"], "physical_actions_differ")
        _equal(record["events"], record["info"]["events"], "events_differ")
        for name, state in (("before_view", before), ("after_view", after)):
            _equal(physical_projection(record[name]), physical_projection(state), "physical_projection_differs_from_snapshot")
        _equal([record["terminated"], record["truncated"]], [after["terminated"], after["truncated"]], "terminal_flags_differ")
        key = (record["partner"], record["scenario_index"])
        scene = header["scenes"][key[1]]
        if key[0] not in PARTNERS or (record["scenario_id"], record["initial_fingerprint"], record["seed"]) != (scene["id"], scene["fingerprint"], 17000 + key[1]):
            raise ValueError("scene_binding_differs")
        if phase == "base":
            sequence = base.setdefault(key, {"steps": 0, "last_after_sha256": None, "done": False})
            if before["frame"] != sequence["steps"]: raise ValueError("incomplete_base_prefix")
            if sequence["steps"]:
                _equal(digest(record["before"]), sequence["last_after_sha256"], "base_snapshot_chain_broken")
            else:
                # Initial observed wrapper is extra; its physical state/RNG must retain source bytes.
                for field in ("state", "rng", "episode_counter"):
                    _equal(record["before"][field], scene["snapshot"][field], "initial_state_differs")
            if record["next_decision"] is not None: raise ValueError("unexpected_base_next_decision")
            sequence.update(steps=sequence["steps"] + 1, last_after_sha256=digest(record["after"]),
                done=bool(after["terminated"] or after["truncated"]))
            if not record["frame"] % 10 and record["groups"]:
                anchors[(*key, record["frame"])] = {"before_sha256": digest(record["before"]),
                    "decision_sha256": digest(record["decision"]), "groups": record["groups"],
                    "fingerprint": record["initial_fingerprint"]}
            rows.append({"fingerprint": scene["fingerprint"], "partner": key[0], "groups": record["groups"],
                "action": record["decision"]["neural_action"], "correct": record["decision"]["neural_action"] == record["decision"]["tree_action"]})
        else:
            action = record["intervention_action"]
            if record["submitted_actions"]["robot_1"] != action or action not in ACTIONS: raise ValueError("intervention_action_differs")
            anchor = (*key, before["frame"])
            pool = branches.setdefault(anchor, {})
            if action in pool: raise ValueError("duplicate_branch")
            if bool(record["next_decision"] is None) != bool(after["terminated"] or after["truncated"]): raise ValueError("missing_next_decision")
            if record["next_decision"] is not None and record["next_decision"]["frame"] != after["frame"]: raise ValueError("next_decision_frame_differs")
            if anchor not in anchors: raise ValueError("branch_precedes_unregistered_base_anchor")
            _equal(digest(record["before"]), anchors[anchor]["before_sha256"], "branch_did_not_start_from_anchor")
            _equal(digest(record["decision"]), anchors[anchor]["decision_sha256"], "branch_initial_NN_differs")
            pool[action] = {"physical_sha256": digest(physical_projection(record["after_view"])),
                "next": None if record["next_decision"] is None else {
                    "neural_action": record["next_decision"]["neural_action"], "tree_action": record["next_decision"]["tree_action"]}}
    expected_base = {(p, i) for p in PARTNERS for i in range(len(header["scenes"]))}
    if set(base) != expected_base: raise ValueError("incomplete_partner_scene_matrix")
    pairs = []
    for key, sequence in base.items():
        if not sequence["done"]: raise ValueError("base_episode_not_complete")
        if sequence["steps"] > header["configuration"]["horizon"]: raise ValueError("base_horizon_exceeded")
    for anchor, source in anchors.items():
        pool = branches.get(anchor, {})
        if set(pool) != set(ACTIONS): raise ValueError("five_branch_matrix_incomplete")
        baseline = pool["WAIT"]
        for action in ACTIONS:
            if action == "WAIT": continue
            changed = pool[action]
            if baseline["next"] is None or changed["next"] is None: continue
            b, c = baseline["next"], changed["next"]
            pairs.append({"fingerprint": source["fingerprint"], "partner": anchor[0], "groups": source["groups"],
                "action": action, "physical_effect": baseline["physical_sha256"] != changed["physical_sha256"],
                "nn_changed": b["neural_action"] != c["neural_action"],
                "correct": b["tree_action"] == b["neural_action"] and c["tree_action"] == c["neural_action"]})
    if set(branches) != set(anchors): raise ValueError("unexpected_branch_anchors")
    counts = manifest["execution"]["counts"]; caps = manifest["execution"]["caps"]
    expected_caps = {"base": 36000, "counterfactual": 18000} if not header["test_fixture"] else {
        "base": len(header["scenes"]) * 3 * header["configuration"]["horizon"], "counterfactual": len(header["scenes"]) * 3 * 5}
    _equal(caps, expected_caps, "fixed_caps_differ")
    for phase, actual in physical_steps.items():
        if actual != counts[phase + "_steps"] or actual != counts[phase + "_attempts"] or actual > caps[phase]: raise ValueError("actual_phase_accounting_differs")
    if counts["acknowledged_steps"] != len(entries) or manifest["execution"]["pending_operation"] is not None: raise ValueError("unconfirmed_records")
    if any(counts[k] != 0 for k in ("neural_updates", "tree_fits", "torch_loads")): raise ValueError("unexpected_training_or_fit")
    full = len(header["scenes"]) == 100 and header["configuration"]["horizon"] == 120 and not header["test_fixture"]
    return {"version": VERSION, "test_fixture": header["test_fixture"], "bindings": header["bindings"],
        "statistics": _summarize(rows, pairs, full), "base_rows": len(rows), "branch_anchor_count": len(anchors),
        "episodes": len(base), "zero_NN_overrides": True, "execution": manifest["execution"],
        "saved_recalculation_scope": "hash-bound recorded logits/observations, scalar tree outputs, groups, matrices and metrics; not independent NN or physical replay",
        "qualification_evaluated": False, "explanation_eligible": False, "participant_enabled": False}


def read_saved_report(output, *, expected_manifest_sha256, expected_bindings):
    output = Path(output).resolve()
    if (output / "pending.json").exists() or (output / "failed.json").exists(): raise ValueError("incomplete_audit_no_retry")
    if file_hash(output / "manifest.json") != expected_manifest_sha256: raise ValueError("manifest_anchor_differs")
    manifest = _read(output / "manifest.json"); header = _read(output / "inputs.json")
    if file_hash(output / "inputs.json") != manifest["header_sha256"]: raise ValueError("header_hash_differs")
    _equal(header["bindings"], expected_bindings, "external_bindings_differ")
    recomputed = _recompute(output, header, manifest)
    if file_hash(output / "report.json") != manifest["report_sha256"]: raise ValueError("report_hash_differs")
    _equal(recomputed, _read(output / "report.json"), "saved_metrics_differ")
    return recomputed
