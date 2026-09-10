"""Durable, bounded driver for the genuine family bilingual answer audit.

Generation and verification are different permanent phases. A completed phase
is read and authenticated, never rerun. An interrupted phase cannot be retried
automatically even if a report or some acknowledged steps were already saved.
This driver preserves the original audit's conclusions and grants no eligibility.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import fcntl
import gzip
import json
import os
import re

from backend import warehouse_runtime_family as registry
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from env.warehouse_native.policy import ACTIONS
from ui import warehouse_family_answer_verification as audit

VERSION = "warehouse-native-family-answer-persistent-run.v1"
PHASES = ("generate", "verify")
PHASE_CAP = 12000


def sources():
    result = audit.answer_verification_sources()
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _equal(a, b, reason):
    if digest(a) != digest(b): raise ValueError(reason)


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _write(path, value, *, replace=False, raw=False):
    data = value if raw else canonical(value).encode()
    if path.suffix == ".gz": data = gzip.compress(data, compresslevel=1, mtime=0)
    temporary = path.with_name(path.name + ".writing") if replace else path
    with temporary.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    if replace: os.replace(temporary, path)
    _sync(path.parent)


def _read(path):
    if path.is_symlink(): raise ValueError("Symlinked journal artifact is not accepted")
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _binding(root, path):
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path), "size": path.stat().st_size}


def _bound(root, binding, expected_path):
    if set(binding) != {"path", "sha256", "size"} or binding["path"] != expected_path:
        raise ValueError("Journal artifact identity differs")
    path = root / expected_path
    if path.is_symlink() or file_hash(path) != binding["sha256"] or path.stat().st_size != binding["size"]:
        raise ValueError("Journal artifact bytes changed")
    return _read(path)


@contextmanager
def _locked(root):
    with (root / ".lock").open("a+b") as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise ValueError("Another driver operation is active") from error
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _phase_state(phase):
    return {"version": VERSION, "phase": phase, "status": "prepared", "cap": PHASE_CAP,
        "reserved_steps": 0, "acknowledged_steps": 0, "pending": None, "report": None,
        "automatic_retry": False, "refund_allowed": False}


class DriverFailure(ValueError):
    def __init__(self, reason, report):
        super().__init__(reason); self.report = report


class _Journal:
    def __init__(self, root, phase, plan):
        self.root, self.phase, self.plan = root, phase, plan
        self.folder = root / phase; self.state_path = self.folder / "state.json"
        self.state = _read(self.state_path)
        self.execution_id = plan["execution_id"] + "-" + phase

    def _commit(self, state):
        _write(self.state_path, state, replace=True)
        self.state = deepcopy(state)

    def before(self, context):
        _equal(_read(self.state_path), self.state, "Concurrent or changed phase state")
        index = self.state["reserved_steps"]
        if (self.state["status"] != "running" or self.state["pending"] is not None
                or index != self.state["acknowledged_steps"] or index >= PHASE_CAP):
            raise ValueError("Phase exhausted, interrupted or awaiting confirmation")
        if (context.get("operation_id") != f"{self.execution_id}:{index}"
                or context.get("phase") not in ("fixed_prefix", "renderer", "independent_oracle")
                or type(context.get("frame")) is not int or context["frame"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(context.get("before_sha256", "")))
                or set(context.get("submitted_actions", {})) != {"robot_1", "robot_2"}
                or any(a not in ACTIONS for a in context["submitted_actions"].values())):
            raise ValueError("Unexpected original audit operation context")
        reservation = {"version": VERSION, "driver_phase": self.phase, "reserved_steps": 1,
            "context": deepcopy(context), "status": "permanently_reserved"}
        path = self.folder / "operations" / f"{index:05d}.reservation.json"
        _write(path, reservation)
        pending = {"index": index, "reservation": _binding(self.root, path)}
        updated = deepcopy(self.state); updated.update(reserved_steps=index + 1, pending=pending)
        self._commit(updated)
        return True

    def after(self, completed):
        _equal(_read(self.state_path), self.state, "Concurrent or changed phase state")
        pending = self.state["pending"]
        if self.state["status"] != "running" or pending is None:
            raise ValueError("Completion lacks a permanent reservation")
        index = pending["index"]
        reservation = _bound(self.root, pending["reservation"], f"{self.phase}/operations/{index:05d}.reservation.json")
        for key, value in reservation["context"].items(): _equal(completed.get(key), value, "After-step context differs from reservation")
        if completed.get("actual_steps") != 1 or type(completed["actual_steps"]) is not int or completed.get("status") != "executed_unacknowledged":
            raise ValueError("Invalid original audit completion")
        _validate_record(completed)
        record_path = self.folder / "operations" / f"{index:05d}.record.json.gz"
        _write(record_path, completed)
        ack = {"version": VERSION, "driver_phase": self.phase, "operation_id": completed["operation_id"],
            "reservation": pending["reservation"], "record": _binding(self.root, record_path), "actual_steps": 1}
        _write(self.folder / "operations" / f"{index:05d}.ack.json", ack)
        updated = deepcopy(self.state); updated.update(acknowledged_steps=index + 1, pending=None)
        self._commit(updated)
        return True


def _validate_record(record):
    if digest(record["before"]) != record["before_sha256"]: raise ValueError("Full before-state hash differs")
    if record["before"]["state"]["frame"] != record["frame"] or record["after"]["state"]["frame"] != record["frame"] + 1:
        raise ValueError("Physical confirmation frame differs")
    _equal(record["info"]["requested_actions"], record["submitted_actions"], "Confirmed submitted actions differ")
    for key in ("rewards", "info"):
        if not isinstance(record[key], dict): raise ValueError("Incomplete original step evidence")
    for flag in ("terminated", "truncated"):
        if type(record[flag]) is not bool or record[flag] is not record["after"]["state"][flag]:
            raise ValueError("Confirmed termination differs")


def _journal_state(root, phase, plan):
    state = _read(root / phase / "state.json")
    if (state.get("version") != VERSION or state.get("phase") != phase or state.get("cap") != PHASE_CAP
            or state.get("automatic_retry") is not False or state.get("refund_allowed") is not False):
        raise ValueError("Phase budget contract differs")
    for key in ("reserved_steps", "acknowledged_steps"):
        if type(state[key]) is not int or not 0 <= state[key] <= PHASE_CAP: raise ValueError("Invalid journal counters")
    if state["status"] not in ("prepared", "completed") or state["pending"] is not None:
        raise ValueError("Interrupted phase cannot automatically replay or refund")
    if state["reserved_steps"] != state["acknowledged_steps"]: raise ValueError("Unconfirmed reserved operations")
    expected_files = set(); phases = {}
    for index in range(state["acknowledged_steps"]):
        prefix = f"{phase}/operations/{index:05d}"
        names = [prefix + suffix for suffix in (".reservation.json", ".record.json.gz", ".ack.json")]
        expected_files.update(Path(n).name for n in names)
        ack = _read(root / names[2]); reservation = _bound(root, ack["reservation"], names[0]); record = _bound(root, ack["record"], names[1])
        if (ack["version"] != VERSION or ack["driver_phase"] != phase or ack["actual_steps"] != 1
                or type(ack["actual_steps"]) is not int or reservation["version"] != VERSION
                or reservation["driver_phase"] != phase or reservation["reserved_steps"] != 1
                or reservation["status"] != "permanently_reserved"):
            raise ValueError("Saved operation phase or budget differs")
        operation_id = f"{plan['execution_id']}-{phase}:{index}"
        if ack["operation_id"] != operation_id or record["operation_id"] != operation_id:
            raise ValueError("Saved operation ID differs")
        for key, value in reservation["context"].items(): _equal(record.get(key), value, "Saved step differs from reservation")
        if record["status"] != "executed_unacknowledged" or type(record["actual_steps"]) is not int or record["actual_steps"] != 1:
            raise ValueError("Saved full record has an invalid actual count")
        _validate_record(record); subphase = record["phase"]
        phases[subphase] = phases.get(subphase, 0) + 1
    if {p.name for p in (root / phase / "operations").iterdir()} != expected_files:
        raise ValueError("Unexpected or missing operation bytes, possibly an interrupted write")
    if state["status"] == "prepared":
        if state["reserved_steps"] or state["report"] is not None or (root / phase / "report.json").exists():
            raise ValueError("Prepared phase already consumed operations or report")
        return state, None
    report = _bound(root, state["report"], f"{phase}/report.json")
    _check_report(report, phase, plan, state, phases)
    return state, report


def _check_report(report, phase, plan, state, phases):
    _equal(report["bindings"], plan["audit_bindings"], "Original report input bindings differ")
    _equal(report["sources"], plan["audit_sources"], "Original report sources differ")
    if (report.get("version") != audit.VERSION or report.get("test_fixture") is not plan["test_fixture"]
            or any(report.get(k) is not False for k in ("release_eligible", "formal_ready", "research_qualification_evaluated"))
            or type(report.get("passed")) is not bool): raise ValueError("Original audit result scope differs")
    execution = report["execution"]; counts = execution["actual_execution"]
    if (execution["auxiliary_step_budget"] != PHASE_CAP or execution["pending_operation"] is not None
            or execution["accounting_complete"] is not True or execution["automatic_retry"] is not False
            or execution["remaining_step_attempt_budget"] != PHASE_CAP-state["reserved_steps"]):
        raise ValueError("Original report budget or confirmation status differs")
    for key in ("environment_steps", "environment_step_attempts", "acknowledged_steps"):
        if type(counts[key]) is not int or counts[key] != state["acknowledged_steps"]:
            raise ValueError("Original API actual counts differ from durable journal")
    _equal(counts["steps_by_phase"], phases, "Original API phase counts differ from durable journal")
    if counts["neural_updates"] != 0 or counts["torch_loads"] != 0: raise ValueError("Unexpected training or checkpoint decoding")
    if phase == "generate" and (report["passed"] is not False or report["independently_verified"] is not False):
        raise ValueError("Generation cannot self-declare independent verification")


def _plan(runtime, program_path, scenarios, expected_bindings, execution_id, fixture):
    return {"version": VERSION, "audit_version": audit.VERSION, "execution_id": execution_id,
        "audit_bindings": deepcopy(expected_bindings), "runtime_family": registry.family(runtime).name,
        "runtime_signature": runtime.signature, "test_fixture": fixture,
        "source_hashes": sources(), "audit_sources": audit.answer_verification_sources(),
        "phase_caps": {phase: PHASE_CAP for phase in PHASES}, "total_cap": PHASE_CAP*2,
        "program_sha256": file_hash(program_path), "scenarios_sha256": digest(scenarios),
        "qualification_evaluated": False, "automatic_retry": False, "refund_allowed": False}


def _initialize(root, plan, program_path, scenarios):
    _write(root / "program.json", Path(program_path).read_bytes(), raw=True)
    _write(root / "scenarios.json", scenarios)
    _write(root / "plan.json", plan)
    for phase in PHASES:
        (root / phase / "operations").mkdir(parents=True)
        _write(root / phase / "state.json", _phase_state(phase))
    _write(root / "manifest.json", {"version": VERSION, "plan_sha256": digest(plan),
        "program": _binding(root, root / "program.json"), "scenarios": _binding(root, root / "scenarios.json")})


def _invoke(root, phase, plan, runtime, program_path, scenarios, generated):
    journal = _Journal(root, phase, plan)
    updated = deepcopy(journal.state); updated["status"] = "running"; journal._commit(updated)
    try:
        options = {"expected_bindings": plan["audit_bindings"], "step_budget": PHASE_CAP,
            "execution_permitted": True, "execution_id": journal.execution_id,
            "allow_test_fixture": plan["test_fixture"], "before_step": journal.before, "after_step": journal.after}
        if phase == "generate": result = audit.generate_answer_report(runtime, program_path, scenarios, **options)
        else: result = audit.verify_answer_report(deepcopy(generated), program_path, runtime, scenarios, **options)
        registry.verify(runtime, allow_test_fixture=plan["test_fixture"], expected_signature=plan["runtime_signature"])
        _equal(audit.input_bindings(runtime, program_path, scenarios), plan["audit_bindings"], "Runtime inputs changed during phase")
        _equal(sources(), plan["source_hashes"], "Driver or verifier changed during phase")
        # A report can only be committed after all original callback operations.
        if journal.state["pending"] is not None: raise ValueError("API returned with an unconfirmed operation")
        phases = {}
        for index in range(journal.state["acknowledged_steps"]):
            row = _read(root / phase / "operations" / f"{index:05d}.record.json.gz")
            phases[row["phase"]] = phases.get(row["phase"], 0) + 1
        _check_report(result, phase, plan, journal.state, phases)
        if phase == "verify" and result.get("input_report_sha256") != digest(generated):
            raise ValueError("Verification did not bind the saved generation report")
        path = root / phase / "report.json"; _write(path, result)
        updated = deepcopy(journal.state); updated.update(status="completed", report=_binding(root, path))
        journal._commit(updated)
        return result
    except BaseException as error:
        evidence = {"version": VERSION, "driver_phase": phase, "reason": str(error),
            "original_api_report": getattr(error, "report", getattr(error, "audit_report", None)),
            "automatic_retry": False, "refund_allowed": False, "qualification_evaluated": False}
        try:
            durable = _read(journal.state_path); evidence["durable_journal"] = durable
            failed = deepcopy(durable); failed["status"] = "failed"
            journal._commit(failed); _write(root / phase / "failure.json", evidence)
        except BaseException as persistence:
            evidence["accounting_incomplete"] = True; evidence["persistence_error"] = str(persistence)
        if not isinstance(error, Exception): error.driver_report = evidence; raise
        raise DriverFailure(str(error), evidence) from error


def run(runtime, program_path, scenarios, *, expected_bindings, output, execution_id,
        execution_permitted=False, allow_test_fixture=False, through="verify"):
    """Explicitly start, continue after generation, or read authenticated completion.

    Each original API owns its inference and physical evaluation. These journal
    callbacks grant at most one permanently reserved environment step each.
    There is no retry/refund for a running, failed or partially persisted phase.
    """
    if execution_permitted is not True: raise ValueError("Explicit execution permission is required")
    if through not in PHASES or type(execution_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", execution_id):
        raise ValueError("Invalid phase or execution ID")
    registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    _equal(audit.input_bindings(runtime, program_path, scenarios), expected_bindings, "External audit bindings differ")
    plan = _plan(runtime, program_path, scenarios, expected_bindings, execution_id, allow_test_fixture)
    root = Path(output).expanduser().resolve(); created = not root.exists()
    if created: root.mkdir(parents=True, exist_ok=False); _sync(root.parent)
    with _locked(root):
        if created: _initialize(root, plan, program_path, scenarios)
        _equal(_read(root / "plan.json"), plan, "Saved source, input, execution ID or budget changed")
        manifest = _read(root / "manifest.json")
        if manifest["version"] != VERSION or manifest["plan_sha256"] != digest(plan): raise ValueError("Plan manifest changed")
        _bound(root, manifest["program"], "program.json")
        if file_hash(root / "program.json") != plan["program_sha256"]: raise ValueError("Frozen program changed")
        _equal(_bound(root, manifest["scenarios"], "scenarios.json"), scenarios, "Frozen scenarios changed")
        generated_state, generated = _journal_state(root, "generate", plan)
        verification_state, verified = _journal_state(root, "verify", plan)
        if generated is None:
            if verification_state["status"] != "prepared": raise ValueError("Verification precedes generation")
            generated = _invoke(root, "generate", plan, runtime, program_path, scenarios, None)
            _journal_state(root, "generate", plan)
        if verified is not None and verified.get("input_report_sha256") != digest(generated):
            raise ValueError("Cached verification refers to a different raw report")
        if through == "verify" and verified is None:
            verified = _invoke(root, "verify", plan, runtime, program_path, scenarios, generated)
            _journal_state(root, "verify", plan)
        states = {phase: _read(root / phase / "state.json") for phase in PHASES}
        return {"version": VERSION, "status": "completed" if verified is not None else "generated",
            "generated_report": generated, "verification_report": verified,
            "reports": {phase: states[phase]["report"] for phase in PHASES},
            "reserved_steps": {phase: states[phase]["reserved_steps"] for phase in PHASES},
            "acknowledged_steps": {phase: states[phase]["acknowledged_steps"] for phase in PHASES},
            "phase_caps": plan["phase_caps"], "total_cap": plan["total_cap"],
            "qualification_evaluated": False, "participant_enabled": False, "release_eligible": False,
            "scope": "Durable orchestration of original reports only; the driver does not declare an independent pass"}
