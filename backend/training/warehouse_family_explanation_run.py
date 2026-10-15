"""Permanent bounded journal for the genuine held-out family explanation audit.

The audit owns all inference and physics. Its full raw transition is already
fsynced before this driver acknowledges it. This layer independently reserves
each step, validates that raw evidence, and never reruns an interrupted audit.
Saved metrics are recomputed by the original reader, not trusted from booleans.
"""
from copy import deepcopy
from pathlib import Path
import re

from backend import warehouse_runtime_family as registry
from backend.training import warehouse_family_explanation_audit as audit
from backend.training import warehouse_family_answer_run as io
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from env.warehouse_native.policy import ACTIONS

VERSION = "warehouse-native-family-heldout-persistent-run.v1"
PHASES = ("base", "counterfactual")


def sources():
    result = audit.execution_sources()
    result[str(Path(io.__file__).relative_to(ROOT))] = file_hash(Path(io.__file__))
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _equal(a, b, message):
    if digest(a) != digest(b): raise ValueError(message)


def _caps(scenarios, fixture):
    if not fixture: return {"base": 36000, "counterfactual": 18000}
    count, horizon = len(scenarios["splits"]["explanation_test"]), scenarios["configuration"]["horizon"]
    if not 1 <= count <= 2 or not 1 <= horizon <= 10:
        raise ValueError("Fixtures require at most two short scenes")
    return {"base": count * 3 * horizon, "counterfactual": count * 3 * 5}


def _plan(runtime, program_path, scenarios, bindings, execution_id, fixture):
    return {"version": VERSION, "audit_version": audit.VERSION, "execution_id": execution_id,
        "bindings": deepcopy(bindings), "sources": sources(), "caps": _caps(scenarios, fixture),
        "runtime_family": registry.family(runtime).name, "runtime_signature": runtime.signature,
        "program_sha256": file_hash(program_path), "scenarios_sha256": digest(scenarios),
        "test_fixture": fixture, "automatic_retry": False, "refund_allowed": False,
        "qualification_evaluated": False}


def _initial_state(plan):
    return {"version": VERSION, "plan_sha256": digest(plan), "status": "prepared",
        "reserved": {p: 0 for p in PHASES}, "acknowledged": {p: 0 for p in PHASES},
        "pending": None, "manifest_sha256": None}


class Journal:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.path = root / "state.json"
        self.state = io._read(self.path)

    def commit(self, state):
        io._write(self.path, state, replace=True); self.state = deepcopy(state)

    def before(self, context):
        _equal(io._read(self.path), self.state, "Journal changed before reservation")
        state = self.state; phase = context.get("phase"); index = sum(state["reserved"].values())
        if (state["status"] != "running" or state["pending"] is not None or phase not in PHASES
                or state["reserved"] != state["acknowledged"] or state["reserved"][phase] >= self.plan["caps"][phase]):
            raise ValueError("Interrupted, unconfirmed or exhausted held-out audit")
        if (context.get("operation_id") != f"{self.plan['execution_id']}:{phase}:{index:06d}"
                or type(context.get("reserved_steps")) is not int or context["reserved_steps"] != 1
                or type(context.get("frame")) is not int or context["frame"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(context.get("before_sha256", "")))
                or set(context.get("submitted_actions", {})) != {"robot_1", "robot_2"}
                or any(a not in ACTIONS for a in context["submitted_actions"].values())):
            raise ValueError("Unexpected original audit reservation")
        reservation = {"version": VERSION, "context": deepcopy(context), "permanent": True}
        path = self.root / "reservations" / f"{index:06d}.json"; io._write(path, reservation)
        changed = deepcopy(state); changed["reserved"][phase] += 1
        changed["pending"] = {"index": index, "reservation": io._binding(self.root, path)}
        self.commit(changed)
        return True

    def after(self, completed):
        _equal(io._read(self.path), self.state, "Journal changed before confirmation")
        if self.state["status"] != "running" or self.state["pending"] is None:
            raise ValueError("Completion has no permanent reservation")
        pending = self.state["pending"]; index = pending["index"]
        reserved = io._bound(self.root, pending["reservation"], f"reservations/{index:06d}.json")
        context = reserved["context"]
        for key, value in context.items(): _equal(completed.get(key), value, "Completion differs from reservation")
        if (completed.get("status") != "executed_unacknowledged"
                or type(completed.get("actual_steps")) is not int or completed["actual_steps"] != 1):
            raise ValueError("Original audit did not complete exactly one step")
        expected = self.root / "audit" / "steps" / f"{index:06d}.json.gz"
        if Path(completed.get("record_path", "")) != expected or expected.is_symlink():
            raise ValueError("Raw transition belongs outside the reserved audit")
        if file_hash(expected) != completed["record_sha256"]:
            raise ValueError("Original fsynced raw transition differs")
        row = io._read(expected); _validate_raw(row, context)
        ack = {"version": VERSION, "reservation": pending["reservation"],
            "record": io._binding(self.root, expected), "completion": deepcopy(completed)}
        io._write(self.root / "confirmations" / f"{index:06d}.json", ack)
        changed = deepcopy(self.state); changed["acknowledged"][context["phase"]] += 1; changed["pending"] = None
        self.commit(changed)
        return True


def _validate_raw(row, context):
    for key, value in context.items(): _equal(row.get(key), value, "Raw transition differs from reserved context")
    if (digest(row["before"]) != context["before_sha256"]
            or row["before"]["state"]["frame"] != context["frame"]
            or row["after"]["state"]["frame"] != context["frame"] + 1):
        raise ValueError("Raw before/after frame or hash differs")
    _equal(row["info"]["requested_actions"], context["submitted_actions"], "Physical input differs")
    if row["decision"]["neural_action"] != context["submitted_actions"]["robot_2"]:
        raise ValueError("Submitted robot 2 command differs from the actual neural output")
    for flag in ("terminated", "truncated"):
        if type(row[flag]) is not bool or row[flag] is not row["after"]["state"][flag]:
            raise ValueError("Raw terminal status differs")


def _read_completed(root, plan):
    state = io._read(root / "state.json")
    if (state.get("version") != VERSION or state.get("plan_sha256") != digest(plan)
            or state.get("status") != "completed" or state.get("pending") is not None
            or state.get("reserved") != state.get("acknowledged")):
        raise ValueError("Interrupted or unconfirmed audit cannot be rerun")
    counts = {p: 0 for p in PHASES}
    expected_names = set()
    for index in range(sum(state["acknowledged"].values())):
        name = f"{index:06d}.json"; expected_names.add(name)
        ack = io._read(root / "confirmations" / name)
        reservation = io._bound(root, ack["reservation"], "reservations/" + name)
        context = reservation["context"]; phase = context["phase"]
        if (ack["version"] != VERSION or reservation["version"] != VERSION or reservation["permanent"] is not True
                or phase not in PHASES or context["operation_id"] != f"{plan['execution_id']}:{phase}:{index:06d}"
                or type(context["reserved_steps"]) is not int or context["reserved_steps"] != 1):
            raise ValueError("Saved reservation contract differs")
        row = io._bound(root, ack["record"], f"audit/steps/{index:06d}.json.gz")
        _validate_raw(row, context)
        completion = ack["completion"]
        for key, value in context.items(): _equal(completion.get(key), value, "Saved confirmation context differs")
        if (completion["record_sha256"] != ack["record"]["sha256"]
                or Path(completion["record_path"]) != root / ack["record"]["path"]
                or completion["status"] != "executed_unacknowledged"
                or type(completion["actual_steps"]) is not int or completion["actual_steps"] != 1):
            raise ValueError("Saved confirmation differs from the original raw bytes")
        counts[phase] += 1
    for folder in ("reservations", "confirmations"):
        if {p.name for p in (root / folder).iterdir()} != expected_names:
            raise ValueError("Unexpected or missing journal records")
    if counts != state["acknowledged"] or any(counts[p] > plan["caps"][p] for p in PHASES):
        raise ValueError("Saved journal counts differ from permanent bounds")
    report = audit.read_saved_report(root / "audit", expected_manifest_sha256=state["manifest_sha256"],
        expected_bindings=plan["bindings"])
    _equal(io._read(root / "audit/inputs.json")["sources"], audit.execution_sources(), "Saved audit source hashes differ")
    for phase in PHASES:
        if report["execution"]["counts"][phase + "_steps"] != counts[phase]:
            raise ValueError("Recomputed raw matrix and journal step counts differ")
    return {"version": VERSION, "status": "completed", "audit_report": report,
        "audit_manifest_sha256": state["manifest_sha256"], "reserved_steps": state["reserved"],
        "acknowledged_steps": counts, "caps": plan["caps"], "qualification_evaluated": False,
        "participant_enabled": False, "explanation_eligible": False}


def run(runtime, program_path, scenarios, *, expected_bindings, output, execution_id,
        execution_permitted=False, allow_test_fixture=False):
    if execution_permitted is not True: raise ValueError("Explicit execution scope is required")
    if type(execution_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", execution_id):
        raise ValueError("Invalid execution ID")
    registry.verify(runtime, allow_test_fixture=allow_test_fixture)
    _equal(audit.input_bindings(runtime, program_path, scenarios), expected_bindings, "External audit bindings differ")
    plan = _plan(runtime, program_path, scenarios, expected_bindings, execution_id, allow_test_fixture)
    root = Path(output).expanduser().absolute()
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("Driver output must not traverse a symlink")
    new = not root.exists()
    if new: root.mkdir(parents=True, exist_ok=False); io._sync(root.parent)
    with io._locked(root):
        if not new:
            _equal(io._read(root / "plan.json"), plan, "Saved input, source or finite budget changed")
            return _read_completed(root, plan)
        io._write(root / "plan.json", plan)
        for folder in ("reservations", "confirmations"): (root / folder).mkdir()
        io._write(root / "state.json", _initial_state(plan))
        journal = Journal(root, plan); started = deepcopy(journal.state); started["status"] = "running"; journal.commit(started)
        try:
            report = audit.audit(runtime, program_path, scenarios, expected_bindings=expected_bindings,
                output=root / "audit", execution_id=execution_id, execution_permitted=True,
                allow_test_fixture=allow_test_fixture, before_step=journal.before, after_step=journal.after)
            _equal(sources(), plan["sources"], "Execution sources changed during audit")
            registry.verify(runtime, allow_test_fixture=allow_test_fixture, expected_signature=runtime.signature)
            _equal(audit.input_bindings(runtime, program_path, scenarios), expected_bindings, "Input artifacts changed during audit")
            if journal.state["pending"] is not None: raise ValueError("Audit returned before a permanent acknowledgement")
            done = deepcopy(journal.state); done.update(status="completed", manifest_sha256=report["manifest_sha256"])
            journal.commit(done)
            return _read_completed(root, plan)
        except BaseException as error:
            failed = deepcopy(journal.state); failed["status"] = "failed"
            try:
                journal.commit(failed)
                io._write(root / "failure.json", {"version": VERSION, "reason": str(error),
                    "original_report": getattr(error, "report", getattr(error, "audit_report", None)),
                    "automatic_retry": False, "refund_allowed": False, "qualification_evaluated": False})
            except BaseException as persistence: error.journal_persistence_failure = str(persistence)
            raise
