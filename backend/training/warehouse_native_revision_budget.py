"""Prepared-only budget reservations for a separately authorized v2 revision.

This module does not change or activate the paused trainer. One parent has one
canonical registry, independent of child output directories. Reservations are
durable upper bounds: a crash never refunds them. A lease also takes the frozen
trainer's existing ``run.lock`` for the entire child job.

The original frozen CLI does not read this registry. These guards coordinate
revision entrants and exclude simultaneous use of the original CLI, but cannot
prevent someone manually running that CLI after the lease has been released.
No real run is opened, changed or trained merely by importing this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading


VERSION = "warehouse-native-revision-budget.v1"
PARENT_VERSION = "warehouse-native-foundation.v2"
AUTHORIZED_TOTAL = 500_000
PARENT_RESERVED = {"curriculum": 10_000, "ppo": 200_000}
PARENT_BASELINE = sum(PARENT_RESERVED.values())
REVISION_MAXIMUM = AUTHORIZED_TOTAL - PARENT_BASELINE
KINDS = frozenset(PARENT_RESERVED)
CHILD_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


class RevisionBudgetError(ValueError):
    """An unsafe or incompatible budget operation was rejected."""


class RevisionLeaseBusy(RevisionBudgetError):
    """The parent is owned by another revision or the frozen original trainer."""


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RevisionBudgetError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise RevisionBudgetError(f"Non-finite JSON value: {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RevisionBudgetError("Invalid budget JSON") from exc


def _no_symlinks(path):
    """Reject aliases and broken links, including existing path ancestors."""
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RevisionBudgetError(f"Budget paths cannot contain symlinks: {component}")


def _regular_bytes(path):
    _no_symlinks(path)
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise RevisionBudgetError(f"Budget path is not a regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RevisionBudgetError(f"Budget path is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _durable_json(path, value):
    """Flush content and directory entry before acknowledging a reservation."""
    _no_symlinks(path)
    content = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".registry-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RevisionBudget:
    """One child ceiling within the original 500k additional authorization.

    Construction and ``read()`` never write. If no registry or child exists,
    ``read()`` returns an explicitly marked preview. Only an authorized
    ``lease()`` creates them. ``remaining`` is the smaller of this child's
    unreserved allowance and the globally unreserved allowance.

    Usage (only after explicit approval of the revised training protocol)::

        budget = RevisionBudget(parent_run, "revision_01",
                                revision_authorized=True)
        with budget.lease():
            budget.reserve("curriculum", 16)  # commit before those steps
            # Perform at most 16 steps; never refund after exceptions.
    """

    def __init__(self, parent_run, child_id, requested_cap=REVISION_MAXIMUM,
                 revision_authorized=False):
        if type(revision_authorized) is not bool:
            raise RevisionBudgetError("revision_authorized must be an explicit boolean")
        if not isinstance(child_id, str) or not CHILD_ID.fullmatch(child_id):
            raise RevisionBudgetError("Invalid revision child ID")
        if not _integer(requested_cap, positive=True) or requested_cap > REVISION_MAXIMUM:
            raise RevisionBudgetError(f"Child cap must be in 1..{REVISION_MAXIMUM}")
        parent = Path(os.path.abspath(os.fspath(parent_run)))
        _no_symlinks(parent)
        if not parent.is_dir():
            raise RevisionBudgetError("Parent run must be an existing directory")
        self.parent_run = parent
        self.child_id = child_id
        self.requested_cap = requested_cap
        self.revision_authorized = revision_authorized
        self.parent_budget_path = parent / "sampling_budget.json"
        self.path = parent.parent / ".warehouse_native_revision_budget" / parent.name / "registry.json"
        self.registry_path = self.path
        self.lock_path = self.path.with_suffix(".lock")
        self._lease_fd = None
        self._lease_owner = None
        self._lease_guard = threading.Lock()
        self._created_registry_observed = False
        self._expected_parent_sha = self._parent_sha()
        _no_symlinks(self.path)
        # Existing binding must be checked even when only constructing a reader.
        if self.path.exists():
            self._load()

    def _parent_sha(self):
        raw = _regular_bytes(self.parent_budget_path)
        value = _strict_json(raw)
        if (not isinstance(value, dict) or set(value) != {"version", "cap", "reserved"}
                or value["version"] != PARENT_VERSION
                or type(value["cap"]) is not int or value["cap"] != AUTHORIZED_TOTAL
                or not isinstance(value["reserved"], dict)
                or set(value["reserved"]) != KINDS
                or any(not _integer(amount) for amount in value["reserved"].values())
                or value["reserved"] != PARENT_RESERVED):
            raise RevisionBudgetError("Parent sampling budget differs from the frozen 210000-step baseline")
        return hashlib.sha256(raw).hexdigest()

    def _check_parent(self):
        current = self._parent_sha()
        if current != self._expected_parent_sha:
            raise RevisionBudgetError("Parent sampling budget bytes changed after revision binding")
        return current

    def _initial(self):
        return {"version": VERSION, "parent_run": str(self.parent_run),
                "parent_sampling_budget_sha256": self._expected_parent_sha,
                "parent_reserved": dict(PARENT_RESERVED), "parent_baseline": PARENT_BASELINE,
                "authorized_total": AUTHORIZED_TOTAL, "revision": 0, "children": {}}

    def _load(self):
        self._check_parent()
        if not self.path.exists():
            if self._created_registry_observed:
                raise RevisionBudgetError("Previously observed revision registry was removed")
            return self._initial()
        value = _strict_json(_regular_bytes(self.path))
        initial = self._initial()
        if not isinstance(value, dict) or set(value) != set(initial):
            raise RevisionBudgetError("Invalid revision registry schema")
        for key in set(initial) - {"children", "revision"}:
            if type(value[key]) is not type(initial[key]) or value[key] != initial[key]:
                raise RevisionBudgetError(f"Revision registry binding mismatch: {key}")
        if not _integer(value["revision"]) or not isinstance(value["children"], dict):
            raise RevisionBudgetError("Invalid registry revision or children")
        for child_id, child in value["children"].items():
            if (not isinstance(child_id, str) or not CHILD_ID.fullmatch(child_id)
                    or not isinstance(child, dict) or set(child) != {"cap", "reserved"}
                    or not _integer(child["cap"], positive=True)
                    or child["cap"] > REVISION_MAXIMUM
                    or not isinstance(child["reserved"], dict) or set(child["reserved"]) != KINDS
                    or any(not _integer(amount) for amount in child["reserved"].values())
                    or sum(child["reserved"].values()) > child["cap"]):
                raise RevisionBudgetError("Invalid child reservation or cap")
        if self._total_reserved(value) > AUTHORIZED_TOTAL:
            raise RevisionBudgetError("Original total authorization exceeded")
        if self.child_id in value["children"] and value["children"][self.child_id]["cap"] != self.requested_cap:
            raise RevisionBudgetError("Existing child cap cannot be changed")
        self._created_registry_observed = True
        return value

    @staticmethod
    def _total_reserved(value):
        return value["parent_baseline"] + sum(sum(child["reserved"].values())
                                              for child in value["children"].values())

    @contextmanager
    def _registry_lock(self):
        _no_symlinks(self.lock_path)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RevisionBudgetError("Registry lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _require_lease(self):
        if (not self.revision_authorized or self._lease_fd is None
                or self._lease_owner != (os.getpid(), threading.get_ident())):
            raise RevisionBudgetError("Reservation requires the authorized parent lease in this process/thread")
        descriptor = os.fstat(self._lease_fd)
        current = os.stat(self.parent_run / "run.lock", follow_symlinks=False)
        if (descriptor.st_dev, descriptor.st_ino) != (current.st_dev, current.st_ino):
            raise RevisionBudgetError("Parent run lock was replaced during lease")

    @contextmanager
    def lease(self):
        if not self.revision_authorized:
            raise PermissionError("Explicit revision authorization is required before any budget write")
        if not self._lease_guard.acquire(blocking=False):
            raise RevisionBudgetError("Nested revision leases are not allowed")
        descriptor = None
        try:
            lock_path = self.parent_run / "run.lock"
            _no_symlinks(lock_path)
            if not stat.S_ISREG(lock_path.stat(follow_symlinks=False).st_mode):
                raise RevisionBudgetError("Parent run lock must be a regular file")
            # The original trainer created this file. Never create or write parent files.
            descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RevisionBudgetError("Parent run lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RevisionLeaseBusy("Parent run is already leased by another training job") from exc
            self._lease_fd = descriptor
            self._lease_owner = (os.getpid(), threading.get_ident())
            self._check_parent()
            _no_symlinks(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._registry_lock():
                value = self._load()
                if self.child_id not in value["children"]:
                    value["children"][self.child_id] = {
                        "cap": self.requested_cap, "reserved": {"curriculum": 0, "ppo": 0}}
                    value["revision"] += 1
                    _durable_json(self.path, value)
                    self._created_registry_observed = True
            yield self
        finally:
            self._lease_fd = None
            self._lease_owner = None
            if descriptor is not None:
                os.close(descriptor)
            self._lease_guard.release()

    def read(self):
        value = self._load()
        child = value["children"].get(self.child_id)
        registered = child is not None
        child_reserved = dict(child["reserved"] if registered else {"curriculum": 0, "ppo": 0})
        global_remaining = AUTHORIZED_TOTAL - self._total_reserved(value)
        return {**copy.deepcopy(value), "registry_path": str(self.path), "child_id": self.child_id,
                "child_registered": registered, "read_only_preview": not registered,
                "child_reserved": child_reserved, "global_remaining": global_remaining,
                "remaining": min(self.requested_cap - sum(child_reserved.values()), global_remaining)}

    @property
    def remaining(self):
        return self.read()["remaining"]

    @property
    def global_remaining(self):
        return self.read()["global_remaining"]

    @property
    def child_reserved(self):
        return self.read()["child_reserved"]

    @property
    def child_reserved_total(self):
        return sum(self.child_reserved.values())

    def reserve(self, kind, amount):
        self._require_lease()
        if kind not in KINDS or not _integer(amount, positive=True):
            raise RevisionBudgetError("Reservation must have a valid kind and positive integer amount")
        with self._registry_lock():
            value = self._load()
            child = value["children"][self.child_id]
            if (self._total_reserved(value) + amount > AUTHORIZED_TOTAL
                    or sum(child["reserved"].values()) + amount > child["cap"]):
                raise RevisionBudgetError("Revision reservation exceeds child or original total authorization")
            child["reserved"][kind] += amount
            value["revision"] += 1
            _durable_json(self.path, value)
        return self.read()
