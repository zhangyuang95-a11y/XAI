"""One-shot observed197 final evaluation; never a qualification grant.

The caller supplies externally verified selection and preflight evidence. This
module binds that evidence, runs a fixed matrix, and preserves actual records.
It neither selects a model nor decodes checkpoints. A permanent weight/physical
matrix key excludes output paths, selection prose and NPZ packaging metadata.
An interrupted or failed key is consumed forever; only completed records have
an offline reader. Synthetic fixtures require explicit opt-in AND an isolated
registry. No CLI or production fixture switch is provided.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import sys
import uuid

import numpy as np

from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash, jsonable
from backend.training.warehouse_native_evaluation import summarize
from backend.training.warehouse_native_public_feedback_evaluation import _outputs, REWARD
from backend.training.warehouse_native_revision_provenance import _absolute, _bytes, _open_dir, _relative, _json
from backend.warehouse_public_history_runtime import PublicHistoryRuntime, runtime_sources
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor, ACTIONS

VERSION = "warehouse-observed197-final-evaluation.v1"
SOURCE = "backend/training/warehouse_public_history_final_evaluation.py"
PRODUCTION_REGISTRY_ROOT = ROOT / "output/warehouse_native/.warehouse_public_history_final_evaluation"
REGISTRY_ROOT = PRODUCTION_REGISTRY_ROOT
PARTNERS = ("skilled", "assertive", "noisy")
JOBS = (("final_test", "neural"), ("final_reference", "reference"), ("final_random", "random"))
COUNT_SCOPE = "admitted_native_step_calls_including_raised_calls_not_completed_physics_proof"
_STEP_CODE = NativeWarehouseEnv.step.__code__


def evaluation_sources():
    return {**runtime_sources(), **{p: file_hash(ROOT / p) for p in (SOURCE,
        "backend/training/warehouse_native_common.py", "backend/training/warehouse_native_evaluation.py",
        "backend/training/warehouse_native_revision_provenance.py")}}


def _encoded(value): return (canonical(value) + "\n").encode()
def _bind(path, raw): return {"path": _relative(path), "sha256": sha256(raw).hexdigest(), "size": len(raw)}
def _equal(a, b, reason):
    if canonical(a) != canonical(b): raise ValueError(reason)


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA-256")
    return value


def _int(value, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError("Invalid exact integer")
    return value


def _create(path):
    path = _absolute(path); parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            try: os.mkdir(part, 0o700, dir_fd=parent); os.fsync(parent)
            except FileExistsError: pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = child
        os.mkdir(path.name, 0o700, dir_fd=parent); os.fsync(parent)
    finally: os.close(parent)
    return path


def _put(root, relative, raw, *, replace=False):
    parts = Path(_relative(relative)).parts; descriptors = [_open_dir(root)]
    temporary = ".pending_" + uuid.uuid4().hex
    try:
        for part in parts[:-1]:
            try: os.mkdir(part, 0o700, dir_fd=descriptors[-1])
            except FileExistsError: pass
            descriptors.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptors[-1]))
        directory = descriptors[-1]
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if replace:
            _bytes(Path(root) / relative)
            os.replace(temporary, parts[-1], src_dir_fd=directory, dst_dir_fd=directory)
        else:
            os.link(temporary, parts[-1], src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            os.unlink(temporary, dir_fd=directory)
        for descriptor in reversed(descriptors): os.fsync(descriptor)
    finally:
        try: os.unlink(temporary, dir_fd=descriptors[-1])
        except FileNotFoundError: pass
        for descriptor in reversed(descriptors): os.close(descriptor)


def _sources_current(sources):
    if not isinstance(sources, dict): raise ValueError("Explicit execution sources required")
    required = evaluation_sources()
    if any(sources.get(p) != h for p, h in required.items()): raise ValueError("Incomplete or changed evaluator sources")
    for path, expected in sources.items():
        _relative(path); _sha(expected)
        if sha256(_bytes(ROOT / path)).hexdigest() != expected: raise ValueError("Source changed: " + path)


def _contract(horizon, count):
    return {"version": VERSION, "observation_size": 197, "public_feedback_mode": "observed",
        "controllers": [kind for _, kind in JOBS], "partners": list(PARTNERS), "evaluated_role": "robot_2",
        "program_role": "robot_1", "episodes_per_partner": count, "horizon": horizon,
        "seed_base": 17000, "controller_rng_stream": 101, "partner_rng_stream": 202,
        "reference_policy": "skilled", "random_policy": "uniform_five_actions",
        "neural_policy": "deterministic_unmasked_argmax", "reward": REWARD, "collision_cost": .05,
        "metric": "mean_team_deliveries", "used_for_selection": False}


def _inputs(runtime, scenarios, protocol, selection_raw, preflight, sources, fixture):
    if type(fixture) is not bool or type(runtime) is not PublicHistoryRuntime or type(runtime.actor) is not NumPyNativeActor:
        raise ValueError("The actual observed197 runtime and explicit fixture boundary are required")
    runtime.verify_binding()
    if runtime.test_fixture is not fixture or runtime.actor.metadata.get("test_fixture", False) is not fixture:
        raise ValueError("Fixture and production runtime cannot be mixed")
    _sources_current(sources)
    _equal(protocol, runtime.protocol, "Runtime protocol differs")
    metadata = runtime.actor.metadata
    _int(metadata["joint_steps"], 1)
    if runtime.actor.obs_dim != 197 or runtime.actor.state_dim != 354: raise ValueError("Only actual197/354 is supported")
    if digest(scenarios) != metadata["scenario_manifest_sha256"]: raise ValueError("Actor scenario manifest differs")
    if scenarios.get("test_fixture", False) is not fixture or preflight.get("test_fixture", False) is not fixture:
        raise ValueError("All fixture inputs must be explicitly marked")
    _equal(scenarios["configuration"], asdict(runtime.config), "Final configuration differs from runtime")
    horizon = _int(runtime.config.horizon, 1, 120); scenes = scenarios["splits"]["final_test"]
    count = _int(len(scenes), 1, 100)
    if not fixture and (count != 100 or horizon != 120): raise ValueError("Production final matrix is exactly 100 by 120")
    if len({s["fingerprint"] for s in scenes}) != count: raise ValueError("Final physical starts must be distinct")
    if "split" in scenarios: _equal(scenarios["split"], scenarios["splits"], "Scenario aliases differ")
    for index, scene in enumerate(scenes):
        _sha(scene["fingerprint"])
        snapshot = scene["snapshot"]
        if (scene["id"] != f"final_test_{index:04d}" or "config" in snapshot
                or any(k.startswith("public_feedback") for k in snapshot)):
            raise ValueError("Final scene order or raw-history schema differs")
        _equal(snapshot["configuration"], scenarios["configuration"], "Snapshot configuration differs")
        state = snapshot["state"]
        if (type(state["frame"]) is not int or state["frame"] != 0
                or state["terminated"] is not False or state["truncated"] is not False):
            raise ValueError("Final scenes must be unfinished original frame zero")
    other = {scene["fingerprint"] for name, pool in scenarios["splits"].items() if name != "final_test" for scene in pool}
    if other.intersection(s["fingerprint"] for s in scenes): raise ValueError("Final physical scenes overlap another split")
    if not isinstance(selection_raw, bytes): raise ValueError("Original selection bytes are required")
    selection = _json(selection_raw)
    selection_binding = {"selected_step": metadata["joint_steps"], "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": digest(protocol), "scenario_manifest_sha256": digest(scenarios),
        "selection_split": "validation", "final_test_used": False}
    for key, value in selection_binding.items(): _equal(selection.get(key), value, "Selection differs: " + key)
    _sha(preflight["signature"])
    if not isinstance(preflight.get("checks"), dict) or not preflight["checks"] or any(
            not isinstance(v, dict) or v.get("passed") is not True for v in preflight["checks"].values()):
        raise ValueError("All externally established preflight checks must pass")
    bindings = {key: selection_binding[key] for key in ("actor_sha256", "protocol_sha256", "scenario_manifest_sha256")}
    bindings.update(selection_sha256=sha256(selection_raw).hexdigest(), runtime_signature=runtime.signature)
    _equal(preflight.get("bindings"), bindings, "Preflight input bindings differ")
    # Weight semantics deliberately exclude file metadata, selection prose and output.
    consumption = {"actor_weights_sha256": runtime._weight_digest(), "feature_names_sha256": digest(metadata["feature_names"]),
        "final_physical_fingerprints": [s["fingerprint"] for s in scenes],
        "configuration": scenarios["configuration"], "evaluation_contract": _contract(horizon, count)}
    identity = {**bindings, "selected_step": metadata["joint_steps"],
        "selected_cumulative_step": metadata["source_counters"]["joint_steps"] + metadata["joint_steps"],
        "actor_metadata_sha256": digest(metadata), "consumption": consumption, "test_fixture": fixture}
    return identity, digest(consumption), scenes, count * len(PARTNERS) * horizon


class _Meter:
    """Thread-local native-call accounting, including a call that raises."""
    def __init__(self, limit): self.limit, self.count, self.rejected = limit, 0, 0
    def __enter__(self):
        if sys.getprofile() is not None: raise ValueError("An existing profiler cannot be replaced")
        def observe(frame, event, arg):
            if event == "call" and frame.f_code is _STEP_CODE:
                if self.count >= self.limit:
                    self.rejected += 1; raise RuntimeError("Final reserved step limit exhausted")
                self.count += 1
        self.observe = observe; sys.setprofile(observe); return self
    def __exit__(self, *args): sys.setprofile(None)


@contextmanager
def _claim(key):
    try: _create(REGISTRY_ROOT)
    except FileExistsError: pass
    directory = _open_dir(REGISTRY_ROOT)
    try:
        try: os.mkdir(key, 0o700, dir_fd=directory)
        except FileExistsError: raise ValueError("final_evaluation_already_reserved_no_retry") from None
        os.fsync(directory)
    finally: os.close(directory)
    root = _absolute(REGISTRY_ROOT) / key; directory = _open_dir(root)
    fd = os.open("attempt.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); os.fsync(fd); os.fsync(directory)
        yield root
    finally: os.close(fd); os.close(directory)


def _episode(runtime, controller, scene, partner, index):
    env = runtime.environment(scene)
    actor_rng = np.random.default_rng(np.random.SeedSequence([17000 + index, 101]))
    partner_rng = np.random.default_rng(np.random.SeedSequence([17000 + index, 202]))
    waits, blocked, gains, charges = [0, 0], [0, 0], [0., 0.], [0, 0]
    repeated = Counter(); trace = sha256(); steps = 0
    if env.public_history()["valid"]: raise ValueError("Final start has invented history")
    while not env.done:
        before = env.get_state(); snapshot_hash = digest(env.snapshot()); obs = env.observations()
        if any(np.asarray(value).shape != (197,) for value in obs.values()): raise ValueError("Non197 Actor observation")
        # Partner does not receive the neural output or current submitted action.
        program = partner_action(env, "robot_1", partner, partner_rng)
        if program not in ACTIONS or digest(env.snapshot()) != snapshot_hash:
            raise ValueError("Program partner changed the prestate")
        if controller == "neural": proposals, probabilities = _outputs(runtime.actor, obs)
        elif controller == "reference":
            proposals = {key: partner_action(env, key, "skilled", actor_rng) for key in env.agent_ids}; probabilities = None
        elif controller == "random":
            proposals = {key: ACTIONS[int(actor_rng.integers(len(ACTIONS)))] for key in env.agent_ids}; probabilities = None
        else: raise ValueError("Unknown controller")
        if program not in ACTIONS or digest(env.snapshot()) != snapshot_hash: raise ValueError("Decision changed the prestate")
        submitted = {"robot_1": program, "robot_2": proposals["robot_2"]}
        _, rewards, terminated, truncated, info = env.step(submitted); steps += 1
        if info["requested_actions"] != submitted: raise ValueError("A submitted command was overwritten")
        after = env.get_state()
        for i, old in enumerate(before.agents):
            new = after.by_id(old.agent_id)
            waits[i] += submitted[old.agent_id] == "WAIT"
            blocked[i] += submitted[old.agent_id] != "WAIT" and info["executed_actions"][old.agent_id] == "WAIT"
            gain = max(0., new.battery - old.battery); gains[i] += gain; charges[i] += gain > 0
        repeated[digest({"positions": [a.position for a in after.agents], "cargo": [a.carrying_task_id for a in after.agents],
            "tasks": [(t.task_id, t.status) for t in after.tasks], "delivered": after.total_deliveries})] += 1
        trace.update(_encoded({"snapshot_before_sha256": snapshot_hash, "snapshot_after_sha256": digest(env.snapshot()),
            "observations": obs, "probabilities": probabilities, "proposed_actions": proposals,
            "submitted_actions": submitted, "executed_actions": info["executed_actions"], "events": info["events"],
            "rewards": rewards, "terminated": terminated, "truncated": truncated}))
    state = env.get_state()
    return jsonable({"scenario_id": scene["id"], "initial_fingerprint": scene["fingerprint"], "partner": partner,
        "controller": controller, "neural_role": 1, "seed": 17000 + index, "steps": steps,
        "team_deliveries": state.total_deliveries, "individual_deliveries": [a.deliveries_completed for a in state.agents],
        "ai_deliveries": state.agents[1].deliveries_completed, "ai_active_end": bool(state.agents[1].active),
        "collisions": state.robot_collision_events, "shutdowns": state.shutdown_count, "native_score": float(state.user_score),
        "legacy_score": None, "legacy_score_status": "not_replayed_against_original_planner", "waits": waits,
        "blocked_moves": blocked, "charge_gain": gains, "charging_steps": charges, "invalid_moves": state.invalid_move_count,
        "max_repeated_configuration": max(repeated.values()), "terminal_reason": state.terminal_reason,
        "nn_action_overrides": 0, "neural_action_calls": steps if controller == "neural" else 0,
        "neural_logit_rows": 2 * steps if controller == "neural" else 0,
        "observation_size": 197, "final_public_history": env.public_history(), "transition_digest": trace.hexdigest()})


def _evaluate(runtime, controller, scenes, on_episode):
    rows = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            row = _episode(runtime, controller, scene, partner, index)
            on_episode(row); rows.append(row)
    return {"rows": rows, "summary": {p: summarize([r for r in rows if r["partner"] == p]) for p in PARTNERS},
        "deterministic_actor": True, "metric": "mean_team_deliveries"}


def _verified_report(raw, scenes, horizon, controller):
    rows = raw["rows"]
    if len(rows) != len(PARTNERS) * len(scenes): raise ValueError("Incomplete final row matrix")
    for row, (partner, index, scene) in zip(rows, ((p, i, s) for p in PARTNERS for i, s in enumerate(scenes))):
        for key, value in {"partner": partner, "scenario_id": scene["id"], "initial_fingerprint": scene["fingerprint"],
                "controller": controller, "neural_role": 1, "seed": 17000 + index, "observation_size": 197,
                "nn_action_overrides": 0}.items(): _equal(row[key], value, "Final row binding differs: " + key)
        steps = _int(row["steps"], 1, horizon)
        for key in ("team_deliveries", "ai_deliveries", "collisions", "shutdowns", "invalid_moves", "max_repeated_configuration"):
            _int(row[key])
        if type(row["ai_active_end"]) is not bool or type(row["native_score"]) not in (int, float): raise ValueError("Invalid score/survival")
        individual = row["individual_deliveries"]
        if not isinstance(individual, list) or len(individual) != 2: raise ValueError("Missing contribution counts")
        for value in individual: _int(value)
        if sum(individual) != row["team_deliveries"] or individual[1] != row["ai_deliveries"]: raise ValueError("Contribution mismatch")
        for key in ("charging_steps", "waits", "blocked_moves"):
            if not isinstance(row[key], list) or len(row[key]) != 2: raise ValueError("Missing role counts")
            for value in row[key]: _int(value, 0, steps)
        _equal(row["neural_action_calls"], steps if controller == "neural" else 0, "NN call count differs")
        _equal(row["neural_logit_rows"], 2 * row["neural_action_calls"], "NN rows differ")
        if row["final_public_history"].get("valid") is not True: raise ValueError("Completed history is missing")
        _sha(row["transition_digest"])
    recomputed = {p: summarize([r for r in rows if r["partner"] == p]) for p in PARTNERS}
    _equal(raw["summary"], recomputed, "Summary differs from original rows")
    _equal(raw["deterministic_actor"], True, "Execution convention differs")
    _equal(raw["metric"], "mean_team_deliveries", "Metric differs")
    return sum(r["steps"] for r in rows)


def _initial_jobs(limit):
    return [{"job": i, "role": role, "controller": kind, "limit": limit, "status": "not_started",
        "actual_environment_steps": 0, "rejected_step_attempts": 0, "episodes": [], "report": None}
        for i, (role, kind) in enumerate(JOBS)]


def _envelope(identity, kind):
    return {"version": VERSION, "controller": kind, "identity": identity, "used_for_selection": False,
        "qualification_granted": False, "release_ready": False, "test_fixture": identity["test_fixture"]}


def verify_receipt(receipt, *, runtime, scenarios, protocol, selection_raw, preflight, source_binding,
                   reports, report_bindings, reservation, journal, allow_test_fixture=False):
    """Portable record verification, not a second physical or neural evaluation.

    Caller anchors must authenticate the receipt and its bound bytes. This does
    not prove historical physics or re-derive the throwaway transition digest.
    """
    identity, key, scenes, limit = _inputs(runtime, scenarios, protocol, selection_raw, preflight, source_binding, allow_test_fixture)
    roles = {role for role, _ in JOBS}; total = limit * len(JOBS)
    if set(reports) != roles or set(report_bindings) != roles: raise ValueError("Incomplete final reports")
    expected_common = {"version": VERSION, "key": key, "identity": identity, "source_binding": source_binding,
        "preflight_sha256": digest(preflight), "preflight_signature": preflight["signature"],
        "reserved_environment_steps": total, "optimizer_updates": 0, "automatic_retry": False,
        "qualification_granted": False, "release_ready": False, "test_fixture": allow_test_fixture}
    for document in (reservation, receipt):
        for field, value in expected_common.items(): _equal(document[field], value, "Record binding differs: " + field)
    _equal(reservation["status"], "reserved_no_refund", "Reservation must remain consumed")
    _equal(reservation["jobs"], _initial_jobs(limit), "Original reservation was changed")
    _absolute(reservation["output_identity"])
    if not isinstance(reservation["created_at"], str): raise ValueError("Missing reservation timestamp")
    for field, value in {"version": VERSION, "key": key, "status": "completed", "reason": None,
            "automatic_retry": False}.items(): _equal(journal[field], value, "Journal terminal differs")
    _equal(receipt["status"], "completed", "Only confirmed completion may be read")
    _equal(receipt["reason"], None, "Completion cannot retain a failure")
    _equal(receipt["count_scope"], COUNT_SCOPE, "Count scope differs")
    _equal(receipt["reservation"], _bind("reservation.json", _encoded(reservation)), "Reservation byte binding differs")
    _equal(receipt["journal"], _bind("journal.json", _encoded(journal)), "Journal byte binding differs")
    _equal(receipt["jobs"], journal["jobs"], "Final jobs differ from durable journal")
    if len(journal["jobs"]) != len(JOBS): raise ValueError("Missing controller jobs")
    actual = 0
    for index, (role, kind) in enumerate(JOBS):
        job = journal["jobs"][index]; raw = reports[role]
        for field, value in _envelope(identity, kind).items(): _equal(raw[field], value, "Report envelope differs")
        for field, value in {"job": index, "role": role, "controller": kind, "limit": limit,
                "status": "completed", "rejected_step_attempts": 0}.items(): _equal(job[field], value, "Job differs: " + field)
        steps = _verified_report(raw, scenes, runtime.config.horizon, kind)
        _equal(job["actual_environment_steps"], steps, "Observed steps differ from original rows")
        _int(steps, 1, limit); actual += steps
        binding = _bind(role + ".json", _encoded(raw))
        _equal(job["report"], binding, "Report bytes differ")
        _equal(report_bindings[role], {k: binding[k] for k in ("sha256", "size")}, "External report binding differs")
        expected_episodes = [{**_bind(f"episodes/{role}/{i:04d}.json", _encoded(row)), "steps": row["steps"]}
            for i, row in enumerate(raw["rows"])]
        _equal(job["episodes"], expected_episodes, "Raw rows differ from confirmed episodes")
    _equal(receipt["actual_environment_steps"], actual, "Actual total differs")
    _equal(receipt["confirmed_environment_steps"], actual, "Confirmed total differs")
    return {"passed": True, "recorded_actual_environment_steps": actual, "reserved_environment_steps": total,
        "environment_steps": 0, "nn_forward_calls": 0, "checkpoint_decodes": 0, "optimizer_updates": 0,
        "qualification_granted": False, "release_ready": False, "test_fixture": allow_test_fixture,
        "scope": "anchored_original_rows_and_accounting_not_independent_physics_or_neural_replay"}


def run_final_evaluations(runtime, scenarios, protocol, selection_raw, *, output, preflight,
                          source_binding, allow_test_fixture=False):
    output = _absolute(output)
    if allow_test_fixture and _absolute(REGISTRY_ROOT) == _absolute(PRODUCTION_REGISTRY_ROOT):
        raise ValueError("Fixtures require an isolated registry")
    if output.exists() or output.is_symlink(): raise ValueError("Final output must be wholly new")
    for protected in (_absolute(REGISTRY_ROOT), Path(runtime.actor.path).resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Final output overlaps a protected input or registry")
    # Check existing ancestors without creating anything before qualification.
    ancestor = output.parent
    while not ancestor.exists() and not ancestor.is_symlink(): ancestor = ancestor.parent
    descriptor = _open_dir(ancestor); os.close(descriptor)
    if sys.getprofile() is not None: raise ValueError("An existing profiler cannot be replaced")
    source_binding, preflight, scenarios, protocol = map(deepcopy, (source_binding, preflight, scenarios, protocol))
    identity, key, scenes, limit = _inputs(runtime, scenarios, protocol, selection_raw, preflight, source_binding, allow_test_fixture)
    # Restore/check fingerprints after all passed preconditions, but before any
    # sampling reservation. These are known initial states, never trajectories.
    for scene in scenes:
        env = runtime.environment(scene)
        if env.done or env.state.frame != 0 or env.public_history()["valid"]: raise ValueError("Invalid final initial state")
    jobs = _initial_jobs(limit)
    common = {"version": VERSION, "key": key, "identity": identity, "source_binding": source_binding,
        "preflight_sha256": digest(preflight), "preflight_signature": preflight["signature"],
        "reserved_environment_steps": len(JOBS) * limit, "optimizer_updates": 0, "automatic_retry": False,
        "qualification_granted": False, "release_ready": False, "test_fixture": allow_test_fixture}
    reservation = {**common, "status": "reserved_no_refund", "jobs": deepcopy(jobs), "output_identity": str(output),
        "created_at": datetime.now(timezone.utc).isoformat()}
    reservation_raw = _encoded(reservation)
    journal = {"version": VERSION, "key": key, "status": "reserved", "reason": None,
        "automatic_retry": False, "jobs": deepcopy(jobs)}
    meters = {}; reports = {}; output_created = False; failure = None
    with _claim(key) as registry:
        # Exclusive directory creation itself is an irrevocable reservation,
        # even if this first durable record fails or the process is killed.
        _put(registry, "reservation.json", reservation_raw)
        _put(registry, "preflight.json", _encoded(preflight))
        _put(registry, "selection.json", selection_raw)
        _put(registry, "journal.json", _encoded(journal))
        def persist(path, raw, replace=False):
            _put(registry, path, raw, replace=replace)
            if output_created: _put(output, path, raw, replace=replace)
        def save_journal(candidate):
            persist("journal.json", _encoded(candidate), replace=True)
        try:
            _create(output); output_created = True
            for path in ("reservation.json", "preflight.json", "selection.json", "journal.json"):
                _put(output, path, _bytes(registry / path))
            for index, (role, kind) in enumerate(JOBS):
                _sources_current(source_binding); runtime.verify_binding()
                candidate = deepcopy(journal); candidate["status"] = "running"
                candidate["jobs"][index].update(status="running", actual_environment_steps=None)
                save_journal(candidate); journal = candidate
                meter = meters[role] = _Meter(limit)
                def on_episode(row):
                    nonlocal journal
                    index_row = len(journal["jobs"][index]["episodes"])
                    path = f"episodes/{role}/{index_row:04d}.json"; raw = _encoded(row)
                    persist(path, raw)  # Raw completion is durable before acknowledgment.
                    candidate = deepcopy(journal)
                    candidate["jobs"][index]["episodes"].append({**_bind(path, raw), "steps": row["steps"]})
                    candidate["jobs"][index]["actual_environment_steps"] = meter.count
                    save_journal(candidate); journal = candidate
                with meter: raw = _evaluate(runtime, kind, scenes, on_episode)
                raw.update(_envelope(identity, kind))
                steps = _verified_report(raw, scenes, runtime.config.horizon, kind)
                if steps != meter.count: raise ValueError("Original row steps differ from actual native calls")
                _sources_current(source_binding); runtime.verify_binding()
                binding = _bind(role + ".json", _encoded(raw)); persist(role + ".json", _encoded(raw))
                candidate = deepcopy(journal)
                candidate["jobs"][index].update(status="completed", actual_environment_steps=meter.count,
                    rejected_step_attempts=meter.rejected, report=binding)
                save_journal(candidate); journal = candidate; reports[role] = raw
            _sources_current(source_binding); runtime.verify_binding()
        except BaseException as error:
            failure = error
        # Recover confirmations only from the durable primary ledger. If it is
        # unreadable or writing fails, raise and leave the consumed key intact.
        journal = _json(_bytes(registry / "journal.json"))
        journal["status"] = "completed" if failure is None else "failed" if isinstance(failure, Exception) else "interrupted"
        journal["reason"] = None if failure is None else f"{type(failure).__name__}: {failure}"
        for job in journal["jobs"]:
            meter = meters.get(job["role"])
            if meter is not None:
                job.update(actual_environment_steps=meter.count, rejected_step_attempts=meter.rejected)
            if job["status"] == "running": job["status"] = journal["status"]
        save_journal(journal)
        receipt = {**common, "status": journal["status"], "reason": journal["reason"],
            "reservation": _bind("reservation.json", reservation_raw),
            "journal": _bind("journal.json", _bytes(registry / "journal.json")), "jobs": deepcopy(journal["jobs"]),
            "actual_environment_steps": sum(m.count for m in meters.values()),
            "confirmed_environment_steps": sum(e["steps"] for j in journal["jobs"] for e in j["episodes"]),
            "count_scope": COUNT_SCOPE}
        if failure is None:
            try:
                verify_receipt(receipt, runtime=runtime, scenarios=scenarios, protocol=protocol, selection_raw=selection_raw,
                    preflight=preflight, source_binding=source_binding, reports=reports,
                    report_bindings={j["role"]: {k: j["report"][k] for k in ("sha256", "size")} for j in journal["jobs"]},
                    reservation=reservation, journal=journal, allow_test_fixture=allow_test_fixture)
            except BaseException as error:
                failure = error
                journal.update(status="failed" if isinstance(error, Exception) else "interrupted",
                    reason=f"{type(error).__name__}: {error}")
                save_journal(journal)
                receipt.update(status=journal["status"], reason=journal["reason"],
                    journal=_bind("journal.json", _bytes(registry / "journal.json")))
        persist("receipt.json", _encoded(receipt))
        result = {"status": receipt["status"], "key": key, "registry_path": str(registry),
            "output_created": output_created, "receipt": receipt, "reports": reports, "reason": receipt["reason"],
            "reserved_environment_steps": receipt["reserved_environment_steps"],
            "actual_environment_steps": receipt["actual_environment_steps"], "qualification_granted": False,
            "release_ready": False, "test_fixture": allow_test_fixture}
        if failure is not None and not isinstance(failure, Exception): raise failure
        return result


def read_completed(output, *, expected_receipt_sha256, runtime, scenarios, protocol, selection_raw,
                   preflight, source_binding, allow_test_fixture=False):
    """Read confirmed original output without registry access or re-evaluation."""
    _sha(expected_receipt_sha256); output = _absolute(output)
    raw = _bytes(output / "receipt.json")
    if sha256(raw).hexdigest() != expected_receipt_sha256: raise ValueError("External receipt anchor differs")
    receipt = _json(raw)
    if receipt.get("status") != "completed": raise ValueError("Only completed final records can be read; no retry")
    documents = []
    for name, key in (("reservation.json", "reservation"), ("journal.json", "journal")):
        bound_raw = _bytes(output / name)
        _equal(_bind(name, bound_raw), receipt[key], "Original " + name + " bytes differ")
        documents.append(_json(bound_raw))
    reservation, journal = documents
    if _bytes(output / "selection.json") != selection_raw: raise ValueError("Original selection bytes differ")
    if _bytes(output / "preflight.json") != _encoded(preflight): raise ValueError("Original preflight bytes differ")
    reports = {}; bindings = {}
    for role, _ in JOBS:
        raw = _bytes(output / (role + ".json")); reports[role] = _json(raw)
        bindings[role] = {"sha256": sha256(raw).hexdigest(), "size": len(raw)}
    report = verify_receipt(receipt, runtime=runtime, scenarios=scenarios, protocol=protocol, selection_raw=selection_raw,
        preflight=preflight, source_binding=source_binding, reports=reports, report_bindings=bindings,
        reservation=reservation, journal=journal, allow_test_fixture=allow_test_fixture)
    for job in journal["jobs"]:
        for entry in job["episodes"]:
            raw = _bytes(output / _relative(entry["path"]))
            _equal(_bind(entry["path"], raw), {k: entry[k] for k in ("path", "sha256", "size")}, "Confirmed episode bytes differ")
    return {"receipt": deepcopy(receipt), "reports": deepcopy(reports), "verification": report}
