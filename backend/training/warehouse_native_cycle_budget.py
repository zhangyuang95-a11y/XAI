"""Reusable finite-cycle reservations with immutable protocol/source/cap binding.

Only local accounting is provided; an external runner must validate authorization
before creating or using a production ledger. No old ledger or model is loaded.
The mature byte/path/JSON helpers are imported without mutating their module;
all ledger filenames, identities, branch caps and version tags are independent.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import uuid

from .warehouse_native_partner_mix_budget import (
    _absolute, _bytes, _digest, _directory, _encoded, _equal, _integer, _pairs, _sha,
)

VERSION = "warehouse-native-finite-cycle-budget.v1"
FILENAME = "cycle_budget.json"
LOCKFILE = "cycle_budget.lock"
KINDS = ("ppo", "evaluation")


def _write(root, value, *, create=False):
    fd = _directory(root); temp = ".cycle_pending_" + uuid.uuid4().hex
    try:
        handle = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(handle, "wb") as stream:
            stream.write(_encoded(value)); stream.flush(); os.fsync(stream.fileno())
        if create:
            os.link(temp, FILENAME, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            os.unlink(temp, dir_fd=fd)
        else:
            _bytes(root / FILENAME)
            os.replace(temp, FILENAME, src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        try: os.unlink(temp, dir_fd=fd)
        except FileNotFoundError: pass
        os.close(fd)


def _cell(caps):
    return {kind: {"cap": cap, "reserved": 0, "acknowledged": 0, "remaining": cap,
        "initial_head": None, "head": None, "pending": None} for kind, cap in caps.items()}


def _binding(identity, caps):
    if not isinstance(identity, dict) or not identity:
        raise ValueError("A complete cycle identity is required")
    _encoded(identity)
    _sha(identity.get("protocol_sha256"))
    sources = identity.get("runtime_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Runtime sources must be bound in the cycle identity")
    for name, digest in sources.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or Path(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("Runtime source names must be safe repository-relative paths")
        _sha(digest)
    if "test_fixture" in identity and type(identity["test_fixture"]) is not bool:
        raise ValueError("Fixture marker must be Boolean")
    embedded = identity.get("budget_caps")
    if caps is None: caps = embedded
    elif "budget_caps" in identity: _equal(caps, embedded, "Explicit and identity budget caps differ")
    if not isinstance(caps, dict) or not caps:
        raise ValueError("Explicit finite branch caps are required")
    for branch, limits in caps.items():
        if not isinstance(branch, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", branch) is None:
            raise ValueError("Invalid branch name")
        if not isinstance(limits, dict) or set(limits) != set(KINDS):
            raise ValueError("Every branch must explicitly bind PPO and evaluation caps")
        _integer(limits["ppo"], 1)
        _integer(limits["evaluation"], 0)
    return deepcopy(identity), deepcopy(caps)


class CycleBudget:
    """Read-only open; caps and identity supplied by a separately approved protocol.

    This component does not authorize sampling. New directories create separate,
    finite cycle ledgers, never a continuation of or refund to a previous cycle.
    Pending operations consume quota until explicitly diagnosed and abandoned;
    abandonment never returns quota or permits the same operation to run again.
    """
    def __init__(self, root, identity, *, caps=None):
        self.root = _absolute(root)
        self.identity, self._caps = _binding(identity, caps)
        self._lease_fd = None
        self.read()

    @property
    def caps(self): return deepcopy(self._caps)

    @property
    def branches(self): return tuple(self._caps)

    @classmethod
    def create(cls, root, identity, *, caps=None):
        root = _absolute(root)
        identity, caps = _binding(identity, caps)
        parent = _directory(root, create=True); lock = None
        try:
            if os.path.lexists(root / FILENAME): raise ValueError("Cycle ledger already exists")
            lock = os.open(LOCKFILE, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); os.fsync(lock); os.fsync(parent)
            branches = {branch: _cell(limits) for branch, limits in caps.items()}
            _write(root, {"version": VERSION, "root": str(root), "identity": identity,
                "identity_sha256": _digest(identity), "caps": caps, "caps_sha256": _digest(caps),
                "revision": 0, "branches": branches, "operations": {},
                "totals": cls._totals(branches, caps)}, create=True)
        finally:
            if lock is not None: os.close(lock)
            os.close(parent)
        return cls(root, identity, caps=caps)

    @staticmethod
    def _totals(branches, caps):
        return {kind: {"cap": sum(value[kind] for value in caps.values()),
            **{field: sum(branches[branch][kind][field] for branch in caps)
               for field in ("reserved", "acknowledged", "remaining")}} for kind in KINDS}

    @contextmanager
    def lease(self):
        if self._lease_fd is not None: raise ValueError("Nested ledger lease is forbidden")
        parent = _directory(self.root)
        try: fd = os.open(LOCKFILE, os.O_RDWR | os.O_NOFOLLOW, dir_fd=parent)
        finally: os.close(parent)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
                raise ValueError("Budget lock must be a single regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lease_fd = fd
            self.read()
            yield self
        finally:
            self._lease_fd = None; os.close(fd)

    def _leased(self):
        if self._lease_fd is None: raise ValueError("An exclusive budget lease is required")
        info = os.stat(self.root / LOCKFILE, follow_symlinks=False)
        held = os.fstat(self._lease_fd)
        if (info.st_dev, info.st_ino) != (held.st_dev, held.st_ino): raise ValueError("Budget lock was replaced")

    def _target(self, path, branch=None):
        path = Path(path)
        if not path.is_absolute(): path = self.root / path
        path = _absolute(path)
        if not path.is_relative_to(self.root) or path == self.root: raise ValueError("Checkpoint escapes the round root")
        relative = str(path.relative_to(self.root))
        if branch is not None and Path(relative).parts[:2] != ("branches", branch):
            raise ValueError("Checkpoint/evidence belongs to a different branch")
        if relative in (FILENAME, LOCKFILE) or any(p.startswith(".cycle_pending_") for p in Path(relative).parts):
            raise ValueError("Checkpoint cannot be a ledger or temporary ledger file")
        return path, relative

    def _checkpoint(self, path, expected, *, durable=False, branch=None):
        _sha(expected); target, relative = self._target(path, branch)
        if sha256(_bytes(target, durable=durable)).hexdigest() != expected: raise ValueError("Checkpoint/evidence SHA-256 differs")
        if durable:
            parent = target.parent
            while parent.is_relative_to(self.root):
                fd = _directory(parent)
                try: os.fsync(fd)
                finally: os.close(fd)
                if parent == self.root: break
                parent = parent.parent
        return relative

    def read(self):
        raw = _bytes(self.root / FILENAME)
        def invalid(value): raise ValueError("Non-finite ledger JSON")
        data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
        _equal(data["version"], VERSION, "Budget version differs")
        _equal(data["root"], str(self.root), "Budget belongs to a different root")
        _equal(data["identity"], self.identity, "Round protocol/source/authorization identity differs")
        _equal(data["identity_sha256"], _digest(self.identity), "Round identity digest differs")
        _equal(data["caps"], self._caps, "Cycle branch caps differ")
        _equal(data["caps_sha256"], _digest(self._caps), "Cycle caps digest differs")
        _integer(data["revision"])
        if set(data["branches"]) != set(self._caps) or not isinstance(data["operations"], dict): raise ValueError("Invalid branch or operation set")
        rebuilt = {b: _cell(caps) for b, caps in self._caps.items()}
        for branch in self._caps:
            if set(data["branches"][branch]) != set(KINDS): raise ValueError("Invalid budget kind")
            for kind in KINDS:
                initial = data["branches"][branch][kind]["initial_head"]
                if initial is not None:
                    if set(initial) != {"path", "sha256", "step"} or type(initial["step"]) is not int or initial["step"] != 0:
                        raise ValueError("Initial checkpoint must be a zero-step binding")
                    self._target(initial["path"], branch); _sha(initial["sha256"])
                rebuilt[branch][kind].update(initial_head=deepcopy(initial), head=deepcopy(initial))
        ordered = sorted(data["operations"].items(), key=lambda kv: kv[1]["sequence"])
        for sequence, (opid, operation) in enumerate(ordered, 1):
            self._opid(opid); _equal(operation["opid"], opid, "Operation ID differs")
            _equal(operation["sequence"], sequence, "Operation sequence has a gap")
            request = operation["request"]
            if set(request) != {"kind", "branch", "steps", "expected_step"}: raise ValueError("Invalid reservation request")
            kind, branch, amount = request["kind"], request["branch"], request["steps"]
            self._kind(kind, branch); _integer(amount, 1, self._caps[branch][kind])
            if request["expected_step"] is not None: _integer(request["expected_step"])
            _equal(operation["request_sha256"], _digest(request), "Reservation request digest differs")
            cell = rebuilt[branch][kind]
            if cell["head"] is None or cell["pending"] is not None: raise ValueError("Reservation lacks a head or follows unresolved sampling")
            _equal(operation["old_head"], cell["head"], "Reservation source checkpoint differs")
            if request["expected_step"] is not None and request["expected_step"] != cell["head"]["step"]:
                raise ValueError("Reservation expected step differs")
            cell["reserved"] += amount; cell["remaining"] -= amount
            if cell["remaining"] < 0: raise ValueError("Round sampling cap exceeded")
            status = operation["status"]
            if status == "pending":
                if operation["completion"] is not None or operation["abandon_reason"] is not None: raise ValueError("Pending reservation has completion data")
                cell["pending"] = opid
            elif status == "abandoned":
                if operation["completion"] is not None or not isinstance(operation["abandon_reason"], str) or not operation["abandon_reason"].strip():
                    raise ValueError("Abandoned reservation lacks diagnostic reason")
            elif status == "acknowledged":
                complete = operation["completion"]
                if set(complete) != {"actual_steps", "checkpoint"} or operation["abandon_reason"] is not None:
                    raise ValueError("Invalid checkpoint acknowledgement")
                actual = complete["actual_steps"]; _integer(actual, 1, amount)
                checkpoint = complete["checkpoint"]
                if set(checkpoint) != {"path", "sha256", "step"}: raise ValueError("Invalid checkpoint head")
                self._target(checkpoint["path"], branch); _sha(checkpoint["sha256"])
                _equal(checkpoint["step"], cell["head"]["step"] + actual, "Acknowledged checkpoint step differs")
                cell["acknowledged"] += actual; cell["head"] = deepcopy(checkpoint)
            else: raise ValueError("Unknown reservation status")
        _equal(data["branches"], rebuilt, "Ledger counters differ from reservations and confirmations")
        totals = self._totals(rebuilt, self._caps)
        _equal(data["totals"], totals, "Global round totals differ")
        revisions = sum(rebuilt[b][k]["initial_head"] is not None for b in self._caps for k in KINDS)
        revisions += len(ordered) + sum(op["status"] != "pending" for _, op in ordered)
        _equal(data["revision"], revisions, "Ledger revision differs from actual mutations")
        checked = set()
        for branch in self._caps:
            for kind in KINDS:
                head = rebuilt[branch][kind]["head"]
                if head is not None and (head["path"], head["sha256"]) not in checked:
                    self._checkpoint(head["path"], head["sha256"], branch=branch); checked.add((head["path"], head["sha256"]))
        return deepcopy(data)

    def _kind(self, kind, branch):
        if not isinstance(kind, str) or not isinstance(branch, str) or kind not in KINDS or branch not in self._caps:
            raise ValueError("Unknown branch or budget kind")

    @staticmethod
    def _opid(opid):
        if not isinstance(opid, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", opid) is None:
            raise ValueError("Invalid operation ID")

    def _save(self, data):
        self._leased(); data["revision"] += 1
        data["totals"] = self._totals(data["branches"], self._caps)
        _write(self.root, data)

    def initialize_head(self, kind, branch, path, sha256):
        self._leased(); self._kind(kind, branch); data = self.read(); cell = data["branches"][branch][kind]
        relative = self._checkpoint(path, sha256, durable=True, branch=branch)
        initial = {"path": relative, "sha256": sha256, "step": 0}
        if cell["initial_head"] is not None:
            _equal(cell["initial_head"], initial, "Initial checkpoint cannot be changed")
            return deepcopy(cell["initial_head"])
        if cell["reserved"]: raise ValueError("Cannot initialize a consumed budget")
        cell.update(initial_head=initial, head=deepcopy(initial)); self._save(data)
        return deepcopy(initial)

    def reserve(self, kind, branch, steps, opid, expected_step=None):
        self._leased(); self._kind(kind, branch); self._opid(opid); _integer(steps, 1, self._caps[branch][kind])
        if expected_step is not None: _integer(expected_step)
        data = self.read(); request = {"kind": kind, "branch": branch, "steps": steps, "expected_step": expected_step}
        if opid in data["operations"]:
            existing = data["operations"][opid]
            _equal(existing["request"], request, "Operation ID reused with another request")
            return {**deepcopy(existing), "execution_permitted": False}
        cell = data["branches"][branch][kind]
        if cell["head"] is None: raise ValueError("Initialize the source checkpoint before sampling")
        if cell["pending"] is not None: raise ValueError("Unconfirmed sampling requires explicit diagnosis before continuing")
        if expected_step is not None and expected_step != cell["head"]["step"]: raise ValueError("Stale checkpoint step")
        if steps > cell["remaining"]: raise ValueError("This round's fixed sampling cap is exhausted")
        operation = {"sequence": len(data["operations"]) + 1, "opid": opid, "request": request,
            "request_sha256": _digest(request), "old_head": deepcopy(cell["head"]), "status": "pending",
            "completion": None, "abandon_reason": None}
        data["operations"][opid] = operation
        cell["reserved"] += steps; cell["remaining"] -= steps; cell["pending"] = opid
        self._save(data)
        return {**deepcopy(operation), "execution_permitted": True}

    def ack(self, opid, checkpoint_path, checkpoint_sha256, actual_steps):
        self._leased(); self._opid(opid); data = self.read()
        operation = data["operations"][opid]; request = operation["request"]
        _integer(actual_steps, 1, request["steps"])
        relative = self._checkpoint(checkpoint_path, checkpoint_sha256, durable=True, branch=request["branch"])
        completion = {"actual_steps": actual_steps, "checkpoint": {"path": relative,
            "sha256": checkpoint_sha256, "step": operation["old_head"]["step"] + actual_steps}}
        if operation["status"] == "acknowledged":
            _equal(operation["completion"], completion, "Acknowledgement retry differs")
            return deepcopy(operation)
        if operation["status"] != "pending": raise ValueError("Abandoned sampling cannot be acknowledged or rerun")
        cell = data["branches"][request["branch"]][request["kind"]]
        if cell["pending"] != opid: raise ValueError("Pending operation binding differs")
        operation.update(status="acknowledged", completion=completion)
        cell["acknowledged"] += actual_steps; cell.update(head=deepcopy(completion["checkpoint"]), pending=None)
        self._save(data)
        return deepcopy(operation)

    def abandon(self, opid, reason):
        self._leased(); self._opid(opid)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000: raise ValueError("Explicit diagnostic reason required")
        data = self.read(); operation = data["operations"][opid]
        if operation["status"] == "abandoned":
            _equal(operation["abandon_reason"], reason, "Abandonment retry differs"); return deepcopy(operation)
        if operation["status"] != "pending": raise ValueError("Confirmed sampling cannot be abandoned")
        request = operation["request"]
        operation.update(status="abandoned", abandon_reason=reason)
        data["branches"][request["branch"]][request["kind"]]["pending"] = None
        self._save(data)
        return deepcopy(operation)

    def remaining(self, kind, branch):
        self._kind(kind, branch); return self.read()["branches"][branch][kind]["remaining"]

    def head(self, kind, branch):
        self._kind(kind, branch); return self.read()["branches"][branch][kind]["head"]


Budget = CycleBudget
