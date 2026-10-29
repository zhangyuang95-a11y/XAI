"""Bounded process-local certificates for immutable committed history only.

No current Actor inference or program qualification is cached here. All persisted frame
bodies are decoded and hash-chain checked on every read. The real run head and
its incoming physical transition are independently recomputed every time, even
for a request selecting an earlier frame. Proof IDs cover only frames before
that head; transaction rollback cannot publish new proof IDs.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import threading

from runtime_v642.contracts import digest, require
from runtime_v642.store import Store as CoreStore, _all, check, decode

VERSION = 'study-vnext-immutable-history-certificates.v1'
_STRICT = ContextVar('study_vnext_original_full_history_verification', default=False)
_PENDING = ContextVar('study_vnext_transaction_local_history_proofs', default=())


def full_verification_enabled():
    return _STRICT.get()


@contextmanager
def full_history_verification():
    token = _STRICT.set(True)
    try:
        yield
    finally:
        _STRICT.reset(token)


def original_full_verification(method):
    @wraps(method)
    def call(*args, **kwargs):
        with full_history_verification():
            return method(*args, **kwargs)
    return call


class HistoryCertificates:
    """Only proof identities and exact-byte demo parse results, never NN output.

    A certificate is eligible for cross-request reuse only after the database
    transaction that verified it commits. A staged certificate can be reused
    inside the same database transaction, but is discarded after any failure.
    """
    def __init__(self, *, maximum_prefixes=4096, maximum_demonstrations=16,
                 maximum_demonstration_bytes=16*1024*1024):
        require(type(maximum_prefixes) is int and maximum_prefixes > 0
            and type(maximum_demonstrations) is int and maximum_demonstrations > 0
            and type(maximum_demonstration_bytes) is int and maximum_demonstration_bytes > 0,
            'positive bounded history cache capacities required')
        self.maximum_prefixes = maximum_prefixes
        self.maximum_demonstrations = maximum_demonstrations
        self.maximum_demonstration_bytes = maximum_demonstration_bytes
        self._prefixes = OrderedDict()
        self._positions = {}
        self._demonstrations = OrderedDict()
        self._demonstration_sizes = {}
        self._demonstration_bytes = 0
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        pending = dict(prefixes=set(), demonstrations=OrderedDict(), demonstration_bytes=0)
        token = _PENDING.set((*_PENDING.get(), (self, pending)))
        try:
            yield
        except BaseException:
            # The outer DB context has rolled back or failed to confirm commit.
            raise
        else:
            with self._lock:
                for key in pending['prefixes']:
                    self._prefixes[key] = None
                    self._prefixes.move_to_end(key)
                    self._positions[key[:2]] = key
                for key, (value, size) in pending['demonstrations'].items():
                    self._demonstration_bytes -= self._demonstration_sizes.get(key, 0)
                    self._demonstrations[key] = deepcopy(value)
                    self._demonstration_sizes[key] = size
                    self._demonstration_bytes += size
                    self._demonstrations.move_to_end(key)
                self._trim()
        finally:
            _PENDING.reset(token)

    def _pending(self):
        for cache, pending in reversed(_PENDING.get()):
            if cache is self:
                return pending
        return None

    def _trim(self):
        while len(self._prefixes) > self.maximum_prefixes:
            key, _ = self._prefixes.popitem(last=False)
            if self._positions.get(key[:2]) == key:
                self._positions.pop(key[:2])
        while len(self._demonstrations) > self.maximum_demonstrations \
                or self._demonstration_bytes > self.maximum_demonstration_bytes:
            key, _ = self._demonstrations.popitem(last=False)
            self._demonstration_bytes -= self._demonstration_sizes.pop(key)

    def contains(self, key):
        pending = self._pending()
        if pending is not None and key in pending['prefixes']:
            return True
        with self._lock:
            if key not in self._prefixes:
                return False
            self._prefixes.move_to_end(key)
            return True

    def stage_prefixes(self, keys):
        pending = self._pending()
        # Unowned direct helper calls may verify, but cannot publish proofs.
        if pending is not None:
            pending['prefixes'].update(keys)

    def conflicts(self, key):
        pending = self._pending()
        if pending is not None and any(old[:2] == key[:2] and old != key for old in pending['prefixes']):
            return True
        with self._lock:
            previous = self._positions.get(key[:2])
            return previous is not None and previous != key

    def parsed_demonstration(self, row, binding):
        """Reuse only an identical persisted body that was structurally checked.

        The caller still performs a fresh NN/physics replay whenever requested
        or in full-verification mode. Returned dictionaries are deep copies.
        """
        key = digest(dict(version=VERSION, binding=binding.to_dict(), raw_persisted_row=row))
        pending = self._pending()
        if not full_verification_enabled():
            if pending is not None and key in pending['demonstrations']:
                return deepcopy(pending['demonstrations'][key][0])
            with self._lock:
                if key in self._demonstrations:
                    self._demonstrations.move_to_end(key)
                    return deepcopy(self._demonstrations[key])
        bundle = decode(row['bundle_json'])
        require(row['binding_sha256'] == digest(binding.to_dict())
            and decode(row['binding_json']) == binding.to_dict()
            and digest(bundle) == row['bundle_sha256'] and bundle.get('two_frozen_NN') is True
            and bundle.get('binding') == binding.to_dict()
            and type(bundle.get('records')) is list and bundle['records'],
            'saved preparation demonstration differs from bound NNs')
        # Bound both the shared LRU and the transaction-local staging area.
        # Oversized valid bundles still return normally; they are never cached.
        size = sum(len(value.encode('utf-8')) for value in row.values() if type(value) is str)
        if pending is not None and size <= self.maximum_demonstration_bytes:
            old = pending['demonstrations'].pop(key, None)
            if old is not None:
                pending['demonstration_bytes'] -= old[1]
            pending['demonstrations'][key] = (deepcopy(bundle), size)
            pending['demonstration_bytes'] += size
            while len(pending['demonstrations']) > self.maximum_demonstrations \
                    or pending['demonstration_bytes'] > self.maximum_demonstration_bytes:
                _, (_, removed_size) = pending['demonstrations'].popitem(last=False)
                pending['demonstration_bytes'] -= removed_size
        return bundle

    def counts(self):
        with self._lock:
            return dict(prefixes=len(self._prefixes), demonstrations=len(self._demonstrations))

    @property
    def demonstration_bytes(self):
        with self._lock:
            return self._demonstration_bytes


def _recompute_frame(store, bound, run, initial, frames, index):
    """The same frame and physical transition checks as the original verifier."""
    actual = frames[index]
    previous = None if index == 0 else frames[index - 1]
    expected = store._frame(bound, actual['state'], transition=actual['transition'],
        previous=None if previous is None else previous['frame_sha256'])
    require(actual == expected, 'saved state/public/NN decision differs from bound runtime')
    if previous is None:
        require(actual['state'] == initial and actual['transition'] is None, 'initial frame differs')
        return
    bundle = actual['transition']
    require(type(bundle) is dict and bundle.get('binding') == bound.binding.to_dict()
        and bundle.get('two_frozen_NN') is False and bundle['initial_state'] == previous['state']
        and bundle['final_state'] == actual['state'] and len(bundle['records']) == 1
        and bundle['records'][0]['decision'] == previous['decision'], 'stored human transition differs')
    record = bundle['records'][0]
    require(type(record.get('human_action')) is str, 'missing recorded human replacement')
    replayed = bound.adapter.replay(bundle, bound.binding)
    stepped = bound.adapter.step(previous['state'], record['human_action'], bound.binding,
                                 decision=previous['decision'])
    require(replayed == stepped == actual['state'], 'bound NN/physics transition replay differs')


def verified_run(store, cache, db, bound, run, *, through=None):
    """Read every byte/hash chain; reuse static proofs, recompute the real head.

    The full verifier is used on a cold/unrecognized history. Normal append-only
    growth has a certified prefix and verifies only the new suffix, selected
    historical endpoint and live head. No proof is reusable without matching
    every raw row byte from the original run origin through that prefix.
    """
    bound.verify()  # Never let a proof replace current asset/binding validation.
    if full_verification_enabled():
        frames = CoreStore._verified_run(store, db, bound, run)
        check(through is None or type(through) is int and 0 <= through < len(frames), 'frame_not_found', 404)
        return frames if through is None else frames[:through + 1]
    require(run['task'] in (1, 2, 3), 'invalid saved task')
    rows = _all(db, 'SELECT * FROM v642_frames WHERE run_id=? ORDER BY turn', (run['id'],))
    require(rows and all(row['turn'] == i for i, row in enumerate(rows)), 'noncontiguous persisted run')
    check(through is None or type(through) is int and 0 <= through < len(rows), 'frame_not_found', 404)
    initial = decode(run['initial_state_json'])
    require(digest(initial) == run['initial_state_sha256'] and initial['turn'] == 0
        and initial['task'] == run['task']
        and initial.get('seed', initial.get('snapshot', {}).get('seed')) == run['seed'], 'run origin differs')
    origin = {key: run[key] for key in ('id', 'instance_id', 'task', 'seed',
        'initial_state_json', 'initial_state_sha256', 'created')}
    context = digest(dict(version=VERSION, run_origin=origin, binding=bound.binding.to_dict()))
    prefix = context
    frames, prefixes = [], []
    previous = None
    for row in rows:
        actual = store._saved_frame(row)  # All JSON bodies + frame hash, every read.
        require(actual['state']['task'] == run['task']
            and actual['binding'] == bound.binding.to_dict()
            and actual['binding_sha256'] == digest(bound.binding.to_dict()),
            'task or binding changed in persisted frame')
        require(actual['previous_frame_sha256'] == previous, 'stored frame hash chain differs')
        prefix = digest(dict(previous_raw_prefix=prefix, raw_persisted_row=row))
        prefixes.append((context, len(frames), prefix))
        frames.append(actual);previous = actual['frame_sha256']
    require(frames[0]['state'] == initial and frames[0]['transition'] is None, 'initial frame differs')
    head = len(frames) - 1
    require(frames[head]['state'] == decode(run['state_json'])
        and (run['status'] == 'complete') == frames[head]['state']['terminal'], 'run head/status differs')
    # Only a prior frame's *executed outgoing* decision can have a reusable
    # certificate. The current head is excluded even if its body was seen.
    known = next((index for index in range(head - 1, -1, -1) if cache.contains(prefixes[index])), None)
    if known is None or any(cache.conflicts(key) for key in prefixes[:head]):
        full = CoreStore._verified_run(store, db, bound, run)
        require(full == frames, 'original full verification and decoded history differ')
    else:
        selected = head if through is None else through
        fresh = set(range(known + 1, len(frames))) | {head, selected}
        for index in sorted(fresh):
            _recompute_frame(store, bound, run, initial, frames, index)
    cache.stage_prefixes(prefixes[:head])
    return frames if through is None else frames[:through + 1]
