"""One-use final evaluation for genuine registered warehouse runtime families.

The frozen final numerical/trajectory functions and safe durable I/O are used
unchanged. Only admission, family envelopes, genuine private construction and
registry routing are new. No class substitution, metadata relabelling, source
patching or final-model selection is performed. This module grants no release
or explanation qualification.

Consumption shares the ORIGINAL registry and physical/seed contract, excluding
this module version, runtime family, output path and NPZ packaging. Therefore a
previously consumed same-weights/same-matrix key cannot move to this new entry.
Caller-authenticated preflight and selection bytes are prerequisites, not
proofs recreated here. Synthetic fixtures need an explicit isolated registry.
"""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import sys

from backend import warehouse_runtime_family as family_registry
from backend.training import warehouse_public_history_final_evaluation as original
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_revision_provenance import _absolute, _bytes, _open_dir, _relative, _json
from env.warehouse_native.policy import NumPyNativeActor

VERSION = "warehouse-runtime-family-final-evaluation.v1"
SOURCE = "backend/training/warehouse_family_final_evaluation.py"
PRODUCTION_REGISTRY_ROOT = original.PRODUCTION_REGISTRY_ROOT
PARTNERS, JOBS, COUNT_SCOPE = original.PARTNERS, original.JOBS, original.COUNT_SCOPE
# These are the actual frozen functions. No type-guarded old runner is invoked.
_encoded, _bind, _equal, _sha, _int = original._encoded, original._bind, original._equal, original._sha, original._int
_create, _put, _contract = original._create, original._put, original._contract
_Meter, _episode, _evaluate = original._Meter, original._episode, original._evaluate
_verified_report, _initial_jobs = original._verified_report, original._initial_jobs


def evaluation_sources():
    sources = original.evaluation_sources()
    for name, value in family_registry.execution_sources().items():
        if name in sources and sources[name] != value: raise ValueError("Final runtime sources disagree")
        sources[name] = value
    sources[SOURCE] = file_hash(ROOT / SOURCE)
    return sources


def _registry_path(value, fixture):
    if type(fixture) is not bool: raise ValueError("Explicit fixture scope required")
    production = _absolute(PRODUCTION_REGISTRY_ROOT)
    if not fixture:
        if value is not None and _absolute(value) != production:
            raise ValueError("Production final registry is permanent and cannot be redirected")
        return production
    if value is None: raise ValueError("Fixtures require an explicitly isolated registry")
    path = _absolute(value)
    if path == production or path.is_relative_to(production) or production.is_relative_to(path):
        raise ValueError("Fixtures cannot use or overlap the production final registry")
    return path


def _runtime_current(runtime, original_runtime, identity):
    for value in (runtime, original_runtime):
        report = family_registry.verify(value, allow_test_fixture=identity['test_fixture'],
            expected_family=identity['runtime_family'], expected_signature=identity['runtime_signature'])
        if report['registry_sources_sha256'] != identity['registry_sources_sha256']:
            raise ValueError("Runtime family sources changed during final work")

def _sources_current(sources):
    if not isinstance(sources, dict): raise ValueError("Explicit execution sources required")
    required = evaluation_sources()
    if any(sources.get(p) != h for p, h in required.items()): raise ValueError("Incomplete or changed evaluator sources")
    for path, expected in sources.items():
        _relative(path); _sha(expected)
        if sha256(_bytes(ROOT / path)).hexdigest() != expected: raise ValueError("Source changed: " + path)



def _inputs(runtime, scenarios, protocol, selection_raw, preflight, sources, fixture):
    family = family_registry.verify(runtime, allow_test_fixture=fixture)
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
        "actor_metadata_sha256": digest(metadata), "consumption": consumption, "test_fixture": fixture,
        "runtime_family": family["family"], "runtime_version": family["runtime_version"],
        "runtime_registry_version": family_registry.VERSION,
        "registry_sources_sha256": family["registry_sources_sha256"]}
    return identity, digest(consumption), scenes, count * len(PARTNERS) * horizon



@contextmanager
def _claim(key, registry_root):
    try: _create(registry_root)
    except FileExistsError: pass
    directory = _open_dir(registry_root)
    try:
        try: os.mkdir(key, 0o700, dir_fd=directory)
        except FileExistsError: raise ValueError("final_evaluation_already_reserved_no_retry") from None
        os.fsync(directory)
    finally: os.close(directory)
    root = _absolute(registry_root) / key; directory = _open_dir(root)
    fd = os.open("attempt.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); os.fsync(fd); os.fsync(directory)
        yield root
    finally: os.close(fd); os.close(directory)



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
                          source_binding, allow_test_fixture=False, registry_root=None):
    output = _absolute(output)
    registry_root = _registry_path(registry_root, allow_test_fixture)
    family_registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    original_runtime = runtime
    if output.exists() or output.is_symlink(): raise ValueError("Final output must be wholly new")
    for protected in (registry_root, Path(runtime.actor.path).resolve()):
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
    with _claim(key, registry_root) as registry:
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
            runtime = family_registry.fresh_instance(original_runtime, allow_test_fixture=allow_test_fixture,
                expected_family=identity["runtime_family"], expected_signature=identity["runtime_signature"])
            _create(output); output_created = True
            for path in ("reservation.json", "preflight.json", "selection.json", "journal.json"):
                _put(output, path, _bytes(registry / path))
            for index, (role, kind) in enumerate(JOBS):
                _sources_current(source_binding); _runtime_current(runtime, original_runtime, identity)
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
                _sources_current(source_binding); _runtime_current(runtime, original_runtime, identity)
                binding = _bind(role + ".json", _encoded(raw)); persist(role + ".json", _encoded(raw))
                candidate = deepcopy(journal)
                candidate["jobs"][index].update(status="completed", actual_environment_steps=meter.count,
                    rejected_step_attempts=meter.rejected, report=binding)
                save_journal(candidate); journal = candidate; reports[role] = raw
            _sources_current(source_binding); _runtime_current(runtime, original_runtime, identity)
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
            "output_created": output_created, "runtime_family": identity["runtime_family"], "receipt": receipt, "reports": reports, "reason": receipt["reason"],
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
