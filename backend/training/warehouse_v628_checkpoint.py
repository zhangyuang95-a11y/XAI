"""v6.28 categorized solver budget constructor; immutable v6.25 CAS methods.

Only constructor admission changes. CAS, fsync, hash-chain, refusal and
cumulative resume implementations are inherited without changing old limits.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from .warehouse_v625_checkpoint import (CheckpointSession as OriginalCheckpointSession,
    plain,digest,ContentStore,CheckpointError,_regular,COMMIT_CHARGE_SECONDS,SCHEMA,
    LEDGER_SCHEMA,canonical_bytes,_directory_fsync)


class CheckpointSession(OriginalCheckpointSession):
    def __init__(self, path, identity, *, max_calls=484096, wall_seconds=6000.,
                 resume=False, clock=time.monotonic, fault_hook=None, started_at=None):
        if (type(max_calls) is not int or not 0 < max_calls <= 484096
                or type(wall_seconds) not in (int, float)
                or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 6000):
            raise ValueError("v628 categorized checkpoint budget exceeds 484096/6000 ceilings")
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

