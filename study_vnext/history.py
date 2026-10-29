"""Bounded process-local certificates for immutable committed history only.

No current Actor inference or program qualification is cached here. All persisted
raw bytes and the hash chain are checked on every read. The real run head and
its incoming physical transition are independently recomputed every time, even
for a request selecting an earlier frame. Proof IDs cover only frames before
that head; transaction rollback cannot publish new proof IDs.
"""
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import json
import math
import struct
import threading

from runtime_v642.contracts import digest, require
from runtime_v642.store import Store as CoreStore, _all, check, decode, encode

VERSION = 'study-vnext-immutable-history-certificates.v2-static-record-parsing'
_STRICT = ContextVar('study_vnext_original_full_history_verification', default=False)
_PENDING = ContextVar('study_vnext_transaction_local_history_proofs', default=())


def raw_row_fingerprint(row):
    """Hash exact SQL values without JSON escaping their embedded JSON bodies.

    Field names, type tags, lengths and exact UTF-8/raw bytes are included.
    Every source byte is read on every request, including fields outside the
    mathematical frame hash. The returned byte count bounds cached payloads.
    """
    require(type(row) is dict and all(type(k) is str for k in row), 'raw SQL row must use string fields')
    result = sha256(b'study-vnext-raw-SQL-row.v1\0')
    total = 0
    for key in sorted(row):
        name = key.encode('utf-8');value = row[key]
        if value is None:tag,raw = b'N',b''
        elif type(value) is str:tag,raw = b'S',value.encode('utf-8')
        elif type(value) is bytes:tag,raw = b'B',value
        elif type(value) is bool:tag,raw = b'T',b'1' if value else b'0'
        elif type(value) is int:tag,raw = b'I',str(value).encode('ascii')
        elif type(value) is float:
            require(math.isfinite(value), 'nonfinite raw SQL value')
            tag,raw = b'F',struct.pack('!d',value)
        else:
            require(False, 'unsupported raw SQL field type')
        result.update(len(name).to_bytes(8,'big'));result.update(name)
        result.update(tag);result.update(len(raw).to_bytes(8,'big'));result.update(raw)
        total += len(name)+len(raw)+17
    return result.hexdigest(),total


def _clone_canonical_json(payload):
    # Only this module's encode(..., allow_nan=False) creates these immutable
    # bytes from already strictly parsed native dictionaries. A single C JSON
    # clone avoids mutable aliases and deepcopy's Python recursion overhead.
    # Untrusted DB bytes always go through the original strict parser on miss.
    return json.loads(payload)


@dataclass(frozen=True,slots=True)
class DemonstrationSummary:
    frame_count: int
    bundle_sha256: str
    binding_sha256: str
    two_frozen_NN: bool


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
                 maximum_demonstration_bytes=16*1024*1024,
                 maximum_frames=4096, maximum_frame_bytes=16*1024*1024):
        require(type(maximum_prefixes) is int and maximum_prefixes > 0
            and type(maximum_demonstrations) is int and maximum_demonstrations > 0
            and type(maximum_demonstration_bytes) is int and maximum_demonstration_bytes > 0
            and type(maximum_frames) is int and maximum_frames > 0
            and type(maximum_frame_bytes) is int and maximum_frame_bytes > 0,
            'positive bounded history cache capacities required')
        self.maximum_prefixes = maximum_prefixes
        self.maximum_demonstrations = maximum_demonstrations
        self.maximum_demonstration_bytes = maximum_demonstration_bytes
        self.maximum_frames = maximum_frames
        self.maximum_frame_bytes = maximum_frame_bytes
        self._prefixes = OrderedDict()
        self._positions = {}
        self._demonstrations = OrderedDict()
        self._demonstration_sizes = {}
        self._demonstration_summaries = {}
        self._demonstration_bytes = 0
        self._frames = OrderedDict()
        self._frame_sizes = {}
        self._frame_bytes = 0
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        pending = dict(prefixes=set(), demonstrations=OrderedDict(), demonstration_bytes=0,
                       demonstration_summaries={}, frames=OrderedDict(), frame_bytes=0)
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
                    self._demonstrations[key] = value  # Canonical immutable bytes.
                    self._demonstration_sizes[key] = size
                    self._demonstration_summaries[key] = pending['demonstration_summaries'][key]
                    self._demonstration_bytes += size
                    self._demonstrations.move_to_end(key)
                for key, (value,size) in pending['frames'].items():
                    self._frame_bytes -= self._frame_sizes.get(key,0)
                    self._frames[key] = value
                    self._frame_sizes[key] = size
                    self._frame_bytes += size
                    self._frames.move_to_end(key)
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
            self._demonstration_summaries.pop(key)
        while len(self._frames) > self.maximum_frames or self._frame_bytes > self.maximum_frame_bytes:
            key,_ = self._frames.popitem(last=False)
            self._frame_bytes -= self._frame_sizes.pop(key)

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
        raw_sha,source_size = raw_row_fingerprint(row)
        key = (digest(binding.to_dict()),raw_sha)
        pending = self._pending()
        if not full_verification_enabled():
            if pending is not None and key in pending['demonstrations']:
                return _clone_canonical_json(pending['demonstrations'][key][0])
            with self._lock:
                if key in self._demonstrations:
                    self._demonstrations.move_to_end(key)
                    return _clone_canonical_json(self._demonstrations[key])
        bundle = decode(row['bundle_json'])
        require(row['binding_sha256'] == digest(binding.to_dict())
            and decode(row['binding_json']) == binding.to_dict()
            and digest(bundle) == row['bundle_sha256'] and bundle.get('two_frozen_NN') is True
            and bundle.get('binding') == binding.to_dict()
            and type(bundle.get('records')) is list and bundle['records'],
            'saved preparation demonstration differs from bound NNs')
        # Bound both the shared LRU and the transaction-local staging area.
        # Oversized valid bundles still return normally; they are never cached.
        payload = encode(bundle).encode('utf-8')
        size = max(source_size,len(payload))
        if pending is not None and size <= self.maximum_demonstration_bytes:
            old = pending['demonstrations'].pop(key, None)
            if old is not None:
                pending['demonstration_bytes'] -= old[1]
            pending['demonstrations'][key] = (payload, size)
            pending['demonstration_summaries'][key] = DemonstrationSummary(
                len(bundle['records']),row['bundle_sha256'],row['binding_sha256'],True)
            pending['demonstration_bytes'] += size
            while len(pending['demonstrations']) > self.maximum_demonstrations \
                    or pending['demonstration_bytes'] > self.maximum_demonstration_bytes:
                removed_key, (_, removed_size) = pending['demonstrations'].popitem(last=False)
                pending['demonstration_bytes'] -= removed_size
                pending['demonstration_summaries'].pop(removed_key)
        return bundle

    def demonstration_summary(self,row,binding):
        """Return typed immutable metadata; never fabricate a partial bundle."""
        raw_sha,_=raw_row_fingerprint(row);key=(digest(binding.to_dict()),raw_sha)
        pending=self._pending()
        if not full_verification_enabled():
            if pending is not None and key in pending['demonstration_summaries']:
                return pending['demonstration_summaries'][key]
            with self._lock:
                if key in self._demonstration_summaries:
                    self._demonstrations.move_to_end(key)
                    return self._demonstration_summaries[key]
        bundle=self.parsed_demonstration(row,binding)
        return DemonstrationSummary(len(bundle['records']),row['bundle_sha256'],row['binding_sha256'],True)

    def parsed_frame(self, key):
        """Clone a previously committed, strictly parsed and verified DB frame."""
        pending = self._pending()
        if pending is not None and key in pending['frames']:
            return _clone_canonical_json(pending['frames'][key][0])
        with self._lock:
            if key not in self._frames:
                return None
            self._frames.move_to_end(key)
            return _clone_canonical_json(self._frames[key])

    def stage_frame(self, key, frame, source_size):
        pending = self._pending()
        if pending is None or key in pending['frames']:
            return
        with self._lock:
            if key in self._frames:
                return
        payload = encode(frame).encode('utf-8');size = max(source_size,len(payload))
        if size > self.maximum_frame_bytes:
            return
        pending['frames'][key] = (payload,size);pending['frame_bytes'] += size
        while len(pending['frames']) > self.maximum_frames or pending['frame_bytes'] > self.maximum_frame_bytes:
            _,(_,removed_size) = pending['frames'].popitem(last=False)
            pending['frame_bytes'] -= removed_size

    def counts(self):
        with self._lock:
            return dict(prefixes=len(self._prefixes), demonstrations=len(self._demonstrations))

    @property
    def demonstration_bytes(self):
        with self._lock:
            return self._demonstration_bytes

    @property
    def frame_bytes(self):
        with self._lock:
            return self._frame_bytes


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


class _VerifiedFrames(Sequence):
    """Private lazy copies of certified records; mutable outputs never alias LRU.

    A normal view only reads the live head. Replay explicitly materializes the
    sequence at its API boundary; strict exports never construct this class.
    Unaccessed historical records retain only their exact raw SQL row and proof
    identity for this transaction, not a shared mutable parsed tree.
    """
    def __init__(self,store,cache,rows,keys,sizes,head_frame,*,length=None):
        self.store,self.cache,self.rows,self.keys,self.sizes = store,cache,rows,keys,sizes
        self.length = len(rows) if length is None else length
        self.materialized = {len(rows)-1:head_frame}

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(self.length))]
        if index < 0:index += self.length
        if not 0 <= index < self.length:raise IndexError(index)
        if index not in self.materialized:
            frame=self.cache.parsed_frame(self.keys[index])
            if frame is None:frame=self.store._saved_frame(self.rows[index])
            self.materialized[index]=frame
        return self.materialized[index]

    def __eq__(self,other):
        return list(self)==other if isinstance(other,list) else list(self)==list(other)

    def stage_accessed(self):
        for index,frame in self.materialized.items():
            if index < len(self.rows)-1:
                self.cache.stage_frame(self.keys[index],frame,self.sizes[index])


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
    binding = bound.binding.to_dict();binding_sha = digest(binding)
    context,_ = raw_row_fingerprint(dict(origin, cache_version=VERSION, complete_binding_sha256=binding_sha))
    prefix = context
    prefixes, parse_keys, source_sizes = [], [], []
    head = len(rows)-1
    previous = None
    for index,row in enumerate(rows):
        row_sha,source_size = raw_row_fingerprint(row)  # Every raw byte, every read.
        key = (context,row_sha)
        require(row['binding_sha256'] == binding_sha,
            'task or binding changed in persisted frame')
        require(row['previous_frame_sha256'] == previous, 'stored frame hash chain differs')
        prefix = sha256(bytes.fromhex(prefix)+bytes.fromhex(row_sha)).hexdigest()
        prefixes.append((context,index,prefix))
        parse_keys.append(key);source_sizes.append(source_size)
        previous = row['frame_sha256']
    # The head always uses the original strict parser and canonical frame-hash
    # check. It is never fetched from, or published into, the static parse LRU.
    current=store._saved_frame(rows[head])
    require(current['state']['task'] == run['task'] and current['binding'] == binding
        and current['state'] == decode(run['state_json'])
        and (run['status'] == 'complete') == current['state']['terminal'], 'run head/status differs')
    # Only a prior frame's *executed outgoing* decision can have a reusable
    # certificate. The current head is excluded even if its body was seen.
    known = next((index for index in range(head - 1, -1, -1) if cache.contains(prefixes[index])), None)
    if known is None or any(cache.conflicts(key) for key in prefixes[:head]):
        frames=CoreStore._verified_run(store, db, bound, run)
        for index in range(head):cache.stage_frame(parse_keys[index],frames[index],source_sizes[index])
    else:
        frames=_VerifiedFrames(store,cache,rows,parse_keys,source_sizes,current)
        require(frames[0]['state'] == initial and frames[0]['transition'] is None, 'initial frame differs')
        selected = head if through is None else through
        fresh = set(range(known + 1, len(frames))) | {head, selected}
        for index in sorted(fresh):
            require(frames[index]['state']['task'] == run['task'] and frames[index]['binding'] == binding
                and frames[index]['binding_sha256'] == binding_sha,
                'task or binding changed in persisted frame')
            _recompute_frame(store, bound, run, initial, frames, index)
        frames.stage_accessed()
    cache.stage_prefixes(prefixes[:head])
    if through is not None and isinstance(frames,_VerifiedFrames):
        frames.length=through+1
        return frames
    return frames if through is None else frames[:through+1]
