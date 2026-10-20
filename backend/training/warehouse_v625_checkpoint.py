"""Small, content-addressed checkpoints and a predebit fit ledger.

Only an explicitly paused, completely committed transaction can resume.  A
process crash deliberately fails closed: a last good snapshot never refunds
work performed after it.  The store uses JSON and allow_pickle=False arrays;
it neither deserializes executable objects nor contains environment state.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time

import numpy as np


SCHEMA = "warehouse-v625-incremental-checkpoint.v1"
LEDGER_SCHEMA = "warehouse-v625-call-ledger.v1"
COMMIT_CHARGE_SECONDS = 1.0
_NONSTRING_KEY_RE = re.compile(r'(?:^|[{,])"(?:-?[0-9]|true"|false"|null")')


class CheckpointError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class DeadlineExceeded(BudgetExhausted):
    pass


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def canonical_bytes(value):
    # The C encoder already traverses native dict/list/tuple/scalar trees.
    # Calling plain on every native element first only duplicates that work.
    # Legacy plain also stringifies *keys before sorting*: numeric keys sort
    # differently and bool/None keys use Python spellings. Preserve that rare
    # case with the original path; ordinary string-key scientific objects use
    # the byte-equivalent fast encoder, with NumPy values converted on demand.
    try:
        encoded = json.dumps(value, default=plain, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        if _NONSTRING_KEY_RE.search(encoded) is None:
            return encoded.encode("utf-8")
    except (TypeError, ValueError):
        pass
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise CheckpointError("missing or symlink checkpoint object: " + str(path))


def _directory_fsync(path, fault):
    """Publish directory-entry changes durably on supported local POSIX FS."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fault("before_directory_fsync", path=str(path))
        os.fsync(descriptor)
        fault("after_directory_fsync", path=str(path))
    finally:
        os.close(descriptor)


class ContentStore:
    """Immutable logical objects, durably published in incremental CAS packs.

    Fit calls remain immediately journaled. Object bytes may buffer while the
    session is dirty; a safe commit first flushes every pending object, its
    offset index, and the pack registry. A crash before that rejects resume.
    """

    def __init__(self, root, *, fault_hook=None):
        self.root = Path(root)
        if self.root.is_symlink():
            raise CheckpointError("object directory must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.fault_hook = fault_hook
        self.written_objects = self.written_bytes = self.reused_objects = 0
        self.io_seconds = 0.
        self._verified_graphs = set()
        self._pending, self._locations, self._pack_arrays = {}, {}, {}
        self._packs, self.registry_ref = [], None
        self.logical_objects_added = self.logical_bytes_added = 0

    def _fault(self, event, **context):
        if self.fault_hook is not None:
            self.fault_hook(event, context)

    def _fsync(self, handle):
        self._fault("before_fsync", path=str(handle.name))
        os.fsync(handle.fileno())
        self._fault("after_fsync", path=str(handle.name))

    def _put(self, raw, kind):
        identity = sha256(raw).hexdigest()
        ref = {"sha256": identity, "kind": kind, "size": len(raw)}
        key = (identity, kind)
        if key in self._pending or key in self._locations:
            self.reused_objects += 1
            return ref
        self._pending[key] = raw
        self.logical_objects_added += 1
        self.logical_bytes_added += len(raw)
        return ref

    def _publish(self, raw, kind):
        """Publish one physical pack/index/registry, never a logical small file."""
        identity = sha256(raw).hexdigest()
        ref = {"sha256": identity, "kind": kind, "size": len(raw)}
        path = self.root / (identity + "." + kind)
        if path.exists() or path.is_symlink():
            self._physical_read(ref)
            self.reused_objects += 1
            return ref
        self._fault("before_object_write", path=str(path), size=len(raw))
        temporary = self.root / (identity + "." + str(os.getpid()) + ".tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                self._fsync(handle)
            # Only this job owns its object store. Never replace a committed
            # object; link is an atomic, no-overwrite publication operation.
            try:
                os.link(temporary, path)
            except FileExistsError:
                self._physical_read(ref)
            _directory_fsync(self.root, self._fault)
            self.written_objects += 1
            self.written_bytes += len(raw)
            self._fault("after_object_write", path=str(path), size=len(raw))
        finally:
            temporary.unlink(missing_ok=True)
        return ref

    @staticmethod
    def _validate_ref(ref):
        if (not isinstance(ref, dict) or set(ref) != {"sha256", "kind", "size"}
                or ref["kind"] not in ("json", "npy")
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in ref["sha256"])
                or type(ref["size"]) is not int or ref["size"] < 0):
            raise CheckpointError("invalid content-addressed reference")

    def _physical_read(self, ref):
        self._validate_ref(ref)
        path = self.root / (ref["sha256"] + "." + ref["kind"])
        _regular(path)
        raw = path.read_bytes()
        if len(raw) != ref["size"] or sha256(raw).hexdigest() != ref["sha256"]:
            raise CheckpointError("content-addressed object hash or size mismatch")
        return raw

    def _read(self, ref):
        self._validate_ref(ref)
        key = (ref["sha256"], ref["kind"])
        if key in self._pending:
            raw = self._pending[key]
        elif key in self._locations:
            data_sha, size, offset = self._locations[key]
            raw = memoryview(self._pack_arrays[data_sha])[offset:offset+size].tobytes()
        else:
            raise CheckpointError("missing logical checkpoint object")
        if len(raw) != ref["size"]:
            raise CheckpointError("logical checkpoint object size mismatch")
        # Pending immutable bytes were hashed at entry; committed slices were
        # independently verified while opening the pack registry.
        return raw

    def flush(self):
        """Durably publish the new batch; all physical files are hash named."""
        if not self._pending:
            return self.registry_ref
        begin = time.monotonic()
        try:
            entries, offset, pieces = [], 0, []
            for (identity, kind), raw in self._pending.items():
                entries.append([identity, kind, len(raw), offset])
                pieces.append(raw)
                offset += len(raw)
            values = np.frombuffer(b"".join(pieces), dtype=np.uint8)
            buffer = io.BytesIO()
            np.save(buffer, values, allow_pickle=False)
            data_ref = self._publish(buffer.getvalue(), "npy")
            index_ref = self._publish(canonical_bytes({
                "schema": "warehouse-v625-CAS-packed-index.v1", "data": data_ref,
                "entries": entries}), "json")
            packs = [*self._packs, index_ref]
            registry = self._publish(canonical_bytes({
                "schema": "warehouse-v625-CAS-pack-registry.v1", "packs": packs}), "json")
            self._pack_arrays[data_ref["sha256"]] = values
            for identity, kind, size, position in entries:
                self._locations[(identity, kind)] = (data_ref["sha256"], size, position)
            self._pending.clear()
            self._packs, self.registry_ref = packs, registry
            return registry
        finally:
            self.io_seconds += time.monotonic()-begin

    def load_registry(self, registry_ref):
        """Verify all physical and logical hashes once on a fresh session."""
        if self._pending or self._locations or self.registry_ref is not None:
            raise CheckpointError("pack registry must open into an empty object store")
        begin = time.monotonic()
        try:
            if registry_ref.get("kind") != "json":
                raise CheckpointError("CAS registry must be a JSON object")
            registry = json.loads(self._physical_read(registry_ref))
            if (registry.get("schema") != "warehouse-v625-CAS-pack-registry.v1"
                    or not isinstance(registry.get("packs"), list)):
                raise CheckpointError("invalid CAS pack registry")
            locations, arrays = {}, {}
            for index_ref in registry["packs"]:
                if index_ref.get("kind") != "json":
                    raise CheckpointError("CAS offset index must be a JSON object")
                index = json.loads(self._physical_read(index_ref))
                if (index.get("schema") != "warehouse-v625-CAS-packed-index.v1"
                        or not isinstance(index.get("entries"), list)):
                    raise CheckpointError("invalid CAS packed index")
                data_ref = index["data"]
                if data_ref.get("kind") != "npy":
                    raise CheckpointError("CAS data pack must be a NumPy byte array")
                values = np.load(io.BytesIO(self._physical_read(data_ref)), allow_pickle=False)
                if values.dtype != np.uint8 or values.ndim != 1 or values.dtype.hasobject:
                    raise CheckpointError("CAS data pack must be a flat uint8 array")
                arrays[data_ref["sha256"]] = values
                cursor = 0
                for identity, kind, size, offset in index["entries"]:
                    ref = {"sha256": identity, "kind": kind, "size": size}
                    self._validate_ref(ref)
                    if type(offset) is not int or offset != cursor or offset+size > len(values):
                        raise CheckpointError("CAS pack offsets overlap or exceed their data")
                    raw = memoryview(values)[offset:offset+size]
                    if sha256(raw).hexdigest() != identity or (identity, kind) in locations:
                        raise CheckpointError("CAS logical object hash or duplicate differs")
                    locations[(identity, kind)] = (data_ref["sha256"], size, offset)
                    cursor += size
                if cursor != len(values):
                    raise CheckpointError("CAS pack has unindexed trailing bytes")
            self._locations, self._pack_arrays = locations, arrays
            self._packs, self.registry_ref = registry["packs"], registry_ref
        except (ValueError, KeyError, TypeError) as error:
            raise CheckpointError("invalid packed checkpoint bytes") from error
        finally:
            self.io_seconds += time.monotonic()-begin

    def physical_path(self, ref):
        """Physical carrier for explicit missing/corrupt-object fault tests."""
        self._validate_ref(ref)
        location = self._locations.get((ref["sha256"], ref["kind"]))
        if location is None:
            raise CheckpointError("logical object has not been durably flushed")
        return self.root/(location[0]+".npy")

    def put_json(self, value):
        begin = time.monotonic()
        try:
            return self._put(canonical_bytes(value), "json")
        finally:
            self.io_seconds += time.monotonic()-begin

    def get_json(self, ref):
        begin = time.monotonic()
        if ref.get("kind") != "json":
            raise CheckpointError("expected JSON object")
        try:
            return json.loads(self._read(ref))
        except (ValueError, UnicodeError) as error:
            raise CheckpointError("invalid stored JSON") from error
        finally:
            self.io_seconds += time.monotonic()-begin

    def put_array(self, value):
        begin = time.monotonic()
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError("checkpoint arrays must not contain Python objects")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        try:
            return self._put(buffer.getvalue(), "npy")
        finally:
            self.io_seconds += time.monotonic()-begin

    def get_array(self, ref):
        begin = time.monotonic()
        if ref.get("kind") != "npy":
            raise CheckpointError("expected array object")
        try:
            return np.load(io.BytesIO(self._read(ref)), allow_pickle=False)
        except (ValueError, OSError) as error:
            raise CheckpointError("invalid stored array") from error
        finally:
            self.io_seconds += time.monotonic()-begin

    def verify_graph(self, value):
        """Verify every reachable typed reference, including evicted cache refs."""
        begin, prior_io = time.monotonic(), self.io_seconds
        seen = set()

        def walk(item):
            if isinstance(item, dict):
                if set(item) == {"sha256", "kind", "size"}:
                    key = (item["sha256"], item["kind"])
                    if key in seen or key in self._verified_graphs:
                        return
                    seen.add(key)
                    if item["kind"] == "json":
                        walk(self.get_json(item))
                    else:
                        self._read(item)
                    self._verified_graphs.add(key)
                else:
                    for child in item.values():
                        walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)
        try:
            walk(value)
            return len(seen)
        finally:
            # get_json already contributes its own interval. Account the rest
            # (including array reads/hashes and graph walking) exactly once.
            self.io_seconds += max(0., time.monotonic()-begin-(self.io_seconds-prior_io))


class CheckpointSession:
    """Single-owner durable ledger, cumulative clock, and small commit file."""

    def __init__(self, path, identity, *, max_calls=19968, wall_seconds=4800.,
                 resume=False, clock=time.monotonic, fault_hook=None, started_at=None):
        if (type(max_calls) is not int or not 0 < max_calls <= 19968
                or type(wall_seconds) not in (int, float)
                or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 4800):
            raise ValueError("checkpoint budget exceeds registered search ceilings")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(str(self.path) + ".ledger.jsonl")
        self.dirty_path = Path(str(self.path) + ".dirty")
        self.identity = plain(identity)
        self.identity_sha256 = digest(self.identity)
        self.max_calls, self.wall_seconds = max_calls, float(wall_seconds)
        self.clock, self.fault_hook = clock, fault_hook
        now = clock()
        if started_at is not None and (type(started_at) not in (int, float)
                or not math.isfinite(started_at) or started_at > now):
            raise ValueError("checkpoint start must be an earlier point on its monotonic clock")
        self.started = now if started_at is None else float(started_at)
        self.elapsed_prior = self.actual_prior = 0.
        self.actual_prior_uncertainty_seconds = 0.
        self.commit_charge_seconds = self.checkpoint_io_seconds = 0.
        self.ledger_io_seconds = self.manifest_io_seconds = 0.
        self.io_prior_upper_seconds = self.io_prior_lower_seconds = 0.
        self.calls, self.sequence = 0, 0
        self.tail = "0" * 64
        self.pending = {}
        self.state = None
        self.closed = False
        self.committed_summary = None
        self.store = ContentStore(str(self.path) + ".objects", fault_hook=fault_hook)
        if resume:
            if self.dirty_path.exists() or self.dirty_path.is_symlink():
                raise CheckpointError("dirty/crashed computation cannot resume")
            _regular(self.path)
            manifest = json.loads(self.path.read_bytes())
            if (manifest.get("schema") != SCHEMA or manifest.get("status") != "paused"
                    or manifest.get("identity_sha256") != self.identity_sha256
                    or manifest.get("budget") != self.budget):
                raise CheckpointError("only an identical actively paused run can resume")
            self._load_ledger()
            if (self.pending or manifest.get("ledger_tail") != self.tail
                    or manifest.get("ledger_sequence") != self.sequence
                    or manifest.get("calls") != self.calls):
                raise CheckpointError("unsettled or post-snapshot ledger work cannot resume")
            self.store.load_registry(manifest["object_pack_registry"])
            self.store.verify_graph(manifest["state"])
            self.state = self.store.get_json(manifest["state"])
            self.elapsed_prior = float(manifest["elapsed_upper_bound_seconds"])
            self.actual_prior = float(manifest["actual_elapsed_at_snapshot_seconds"])
            self.actual_prior_uncertainty_seconds = (
                float(manifest["actual_elapsed_at_snapshot_upper_bound_seconds"])
                - self.actual_prior + COMMIT_CHARGE_SECONDS)
            self.commit_charge_seconds = float(manifest["commit_charge_seconds"])
            self.checkpoint_io_seconds = float(manifest["checkpoint_io_seconds"])
            self.io_prior_lower_seconds = float(manifest["checkpoint_total_io_lower_bound_at_snapshot_seconds"])
            self.io_prior_upper_seconds = (
                float(manifest["checkpoint_total_io_upper_bound_at_snapshot_seconds"])
                + COMMIT_CHARGE_SECONDS)
            if (not all(math.isfinite(value) and value >= 0 for value in
                    (self.elapsed_prior, self.actual_prior, self.actual_prior_uncertainty_seconds,
                     self.io_prior_lower_seconds, self.io_prior_upper_seconds))
                    or self.io_prior_lower_seconds > self.io_prior_upper_seconds):
                raise CheckpointError("invalid cumulative checkpoint accounting bounds")
            if self.elapsed_prior >= self.wall_seconds:
                raise CheckpointError("paused checkpoint has exhausted its budget")
        else:
            if any(p.exists() or p.is_symlink() for p in (self.path, self.ledger_path, self.dirty_path)):
                raise CheckpointError("fresh checkpoint cannot overwrite prior work")
        with self.dirty_path.open("xb") as handle:
            handle.write(canonical_bytes({"identity_sha256": self.identity_sha256}))
            handle.flush()
            self._sync(handle)
        _directory_fsync(self.path.parent, self._fault)
        if resume:
            self._append("resumed", {"prior_elapsed_upper_bound_seconds": self.elapsed_prior})
        else:
            self._append("created", {"schema": LEDGER_SCHEMA,
                                     "identity_sha256": self.identity_sha256,
                                     "budget": self.budget})

    @property
    def budget(self):
        return {"maximum_calls": self.max_calls, "wall_seconds": self.wall_seconds}

    @property
    def elapsed_seconds(self):
        return self.elapsed_prior + self.clock() - self.started

    @property
    def actual_elapsed_seconds(self):
        return self.actual_prior + self.clock() - self.started

    def _fault(self, event, **context):
        if self.fault_hook is not None:
            self.fault_hook(event, context)

    def _sync(self, handle):
        self._fault("before_fsync", path=str(handle.name))
        os.fsync(handle.fileno())
        self._fault("after_fsync", path=str(handle.name))

    def _append(self, event, data):
        begin = time.monotonic()
        body = {"sequence": self.sequence + 1, "previous_sha256": self.tail,
                "event": event, "data": plain(data),
                "elapsed_seconds": self.elapsed_seconds}
        row = dict(body, record_sha256=digest(body))
        if self.ledger_path.is_symlink():
            raise CheckpointError("ledger must not be a symlink")
        created = not self.ledger_path.exists()
        with self.ledger_path.open("ab") as handle:
            handle.write(canonical_bytes(row) + b"\n")
            handle.flush()
            self._sync(handle)
        if created:
            _directory_fsync(self.ledger_path.parent, self._fault)
        self.sequence, self.tail = body["sequence"], row["record_sha256"]
        self.ledger_io_seconds += time.monotonic()-begin
        return row

    def _load_ledger(self):
        begin = time.monotonic()
        _regular(self.ledger_path)
        for raw in self.ledger_path.read_bytes().splitlines():
            try:
                row = json.loads(raw)
                record = row.pop("record_sha256")
                if (row["sequence"] != self.sequence + 1
                        or row["previous_sha256"] != self.tail or digest(row) != record):
                    raise ValueError("broken ledger chain")
                event, data = row["event"], row["data"]
                if self.sequence == 0 and (event != "created" or data.get("identity_sha256") != self.identity_sha256
                                          or data.get("budget") != self.budget):
                    raise ValueError("ledger identity differs")
                if event == "call_started":
                    if data["call"] != self.calls + 1 or self.pending:
                        raise ValueError("invalid call predebit")
                    self.calls += 1
                    self.pending[self.calls] = data
                elif event == "call_completed":
                    if data["call"] not in self.pending:
                        raise ValueError("completion without predebit")
                    self.pending.pop(data["call"])
                self.sequence, self.tail = row["sequence"], record
            except (ValueError, KeyError, TypeError) as error:
                raise CheckpointError("corrupt or incomplete durable call ledger") from error
        if not self.sequence or self.calls > self.max_calls:
            raise CheckpointError("empty or over-budget call ledger")
        self.ledger_io_seconds += time.monotonic()-begin

    def check(self, *, reserve_seconds=0.):
        if self.closed:
            raise CheckpointError("checkpoint session is closed")
        if self.elapsed_seconds + reserve_seconds >= self.wall_seconds:
            raise DeadlineExceeded("registered cumulative search deadline exhausted")

    def consume(self, metadata=None):
        self.check()
        if self.pending:
            raise CheckpointError("previous fit call was not settled")
        if self.calls >= self.max_calls:
            raise BudgetExhausted("registered predebited leaf API budget exhausted")
        call = self.calls + 1
        data = {"call": call, "metadata": plain(metadata or {})}
        # Durable predebit is completed before the caller may enter its solver.
        self._append("call_started", data)
        self.calls = call
        self.pending[call] = data
        self.check()
        return call

    def finish(self, token=None, *, success=True, detail=None):
        if token is None and len(self.pending) == 1:
            token = next(iter(self.pending))
        if token not in self.pending:
            raise CheckpointError("unknown or already settled fit call")
        self._append("call_completed", {"call": token, "success": bool(success),
                                         "detail": plain(detail or {})})
        self.pending.pop(token)

    def events(self):
        begin = time.monotonic()
        try:
            return [json.loads(line) for line in self.ledger_path.read_bytes().splitlines()]
        finally:
            self.ledger_io_seconds += time.monotonic()-begin

    def commit(self, state, *, status="paused"):
        if status not in ("running", "paused", "completed") or self.pending:
            raise CheckpointError("only settled candidate boundaries can commit")
        self.check(reserve_seconds=COMMIT_CHARGE_SECONDS)
        begin = self.clock()
        ref = self.store.put_json(state)
        self.store.verify_graph(ref)
        registry = self.store.flush()
        self.check(reserve_seconds=COMMIT_CHARGE_SECONDS)
        self.checkpoint_io_seconds += self.clock() - begin
        self._append("commit", {"status": status, "state": ref,
                                "object_pack_registry": registry})
        actual_snapshot = self.actual_elapsed_seconds
        billed_snapshot = self.elapsed_seconds
        manifest = {"schema": SCHEMA, "status": status,
            "identity_sha256": self.identity_sha256, "budget": self.budget,
            "state": ref, "ledger_tail": self.tail, "ledger_sequence": self.sequence,
            "object_pack_registry": registry,
            "calls": self.calls, "actual_elapsed_at_snapshot_seconds": actual_snapshot,
            "actual_elapsed_at_snapshot_upper_bound_seconds": actual_snapshot+self.actual_prior_uncertainty_seconds,
            "elapsed_upper_bound_seconds": billed_snapshot + COMMIT_CHARGE_SECONDS,
            "commit_charge_seconds": self.commit_charge_seconds + COMMIT_CHARGE_SECONDS,
            "checkpoint_io_seconds": self.checkpoint_io_seconds,
            "checkpoint_total_io_at_snapshot_seconds": self.io_prior_upper_seconds+self.store.io_seconds+self.ledger_io_seconds+self.manifest_io_seconds,
            "checkpoint_total_io_lower_bound_at_snapshot_seconds": self.io_prior_lower_seconds+self.store.io_seconds+self.ledger_io_seconds+self.manifest_io_seconds,
            "checkpoint_total_io_upper_bound_at_snapshot_seconds": self.io_prior_upper_seconds+self.store.io_seconds+self.ledger_io_seconds+self.manifest_io_seconds,
            "resume_scope": "explicit_pause_only_no_dirty_or_inflight_work"}
        temporary = Path(str(self.path) + ".new")
        small_begin = self.clock()
        try:
            with temporary.open("xb") as handle:
                handle.write(canonical_bytes(manifest))
                handle.flush()
                self._sync(handle)
            self._fault("before_manifest_replace", path=str(self.path))
            os.replace(temporary, self.path)
            _directory_fsync(self.path.parent, self._fault)
            self._fault("after_manifest_replace", path=str(self.path))
            if status != "running":
                self._fault("before_dirty_remove", path=str(self.dirty_path))
                self.dirty_path.unlink()
                _directory_fsync(self.path.parent, self._fault)
            actual_small = self.clock() - small_begin
            self.manifest_io_seconds += actual_small
            if actual_small > COMMIT_CHARGE_SECONDS or self.elapsed_seconds >= self.wall_seconds:
                # A late commit is never resumable or a successful completion.
                self.dirty_path.write_bytes(b"late commit")
                raise DeadlineExceeded("checkpoint small commit exceeded its conservative charge/deadline")
        except BaseException:
            if not self.dirty_path.exists():
                self.dirty_path.write_bytes(b"failed commit")
            raise
        finally:
            temporary.unlink(missing_ok=True)
        self.commit_charge_seconds += COMMIT_CHARGE_SECONDS
        # Replace the measured small tail with its declared one-second upper
        # charge. Do not add both the actual tail and the full charge again.
        self.elapsed_prior += COMMIT_CHARGE_SECONDS-actual_small
        if status == "running":
            return manifest
        self.committed_summary = self.summary()
        self.closed = True
        return manifest

    def snapshot(self, state):
        """Durable diagnostic boundary; remains dirty and is not resumable."""
        return self.commit(state, status="running")

    def summary(self):
        actual = self.actual_elapsed_seconds
        current_io = self.store.io_seconds+self.ledger_io_seconds+self.manifest_io_seconds
        return {"calls": self.calls, "pending_calls": len(self.pending),
                "elapsed_upper_bound_seconds": self.elapsed_seconds,
                "actual_elapsed_seconds": actual,
                "actual_elapsed_lower_bound_seconds": actual,
                "actual_elapsed_upper_bound_seconds": actual+self.actual_prior_uncertainty_seconds,
                "actual_elapsed_scope": "measured_current_segment_plus_prior_snapshot_lower_bounds",
                "prior_unmeasured_commit_tail_upper_seconds": self.actual_prior_uncertainty_seconds,
                "commit_charge_seconds": self.commit_charge_seconds,
                "checkpoint_io_seconds": self.checkpoint_io_seconds,
                "checkpoint_total_io_seconds": self.io_prior_upper_seconds+current_io,
                "checkpoint_segment_io_seconds": current_io,
                "checkpoint_total_io_lower_bound_seconds": self.io_prior_lower_seconds+current_io,
                "checkpoint_total_io_upper_bound_seconds": self.io_prior_upper_seconds+current_io,
                "checkpoint_io_scope": "measured_current_segment_plus_prior_IO_bounds_and_pause_tail_reserve",
                "object_io_seconds": self.store.io_seconds,
                "ledger_io_seconds": self.ledger_io_seconds,
                "manifest_io_seconds": self.manifest_io_seconds,
                "ledger_tail": self.tail, "ledger_sequence": self.sequence,
                "store_written_objects": self.store.written_objects,
                "store_written_bytes": self.store.written_bytes,
                "store_reused_objects": self.store.reused_objects,
                "store_pending_objects": len(self.store._pending),
                "store_logical_objects_added": self.store.logical_objects_added,
                "store_logical_bytes_added": self.store.logical_bytes_added,
                "store_pack_count": len(self.store._packs)}

    def close(self):
        """Abandoning a live session retains dirty evidence and prohibits resume."""
        self.closed = True

    @contextmanager
    def hard_deadline(self):
        """POSIX main-thread timer; runner must also supervise the process.

        Every successful exit is checked after the operation. A C extension
        which delays Python signal delivery therefore cannot report success.
        """
        self.check()
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
            raise CheckpointError("strict local deadline needs POSIX main-thread supervision")
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] or previous_timer[1]:
            raise CheckpointError("an existing process alarm cannot be repurposed")
        def expired(signum, frame):
            raise DeadlineExceeded("hard registered search deadline reached")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, self.wall_seconds-self.elapsed_seconds))
        try:
            yield
            if not self.closed:
                self.check()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, previous_handler)
