"""Exact, source-bound reuse of a once fully validated native v642 program.

This is not an evidence issuer, NN target validator or execution policy.  Only
native ``validate`` / ``to_dict`` static proof work is memoized.  Forward,
metrics, current-NN qualification and all scientific gates remain native.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
import json
import math

from .warehouse_v642_program import Program as NativeProgram
from .warehouse_v642_online_evidence import source_files as native_source_files
from experiments.warehouse.v626_protocol import require

VERSION = 'warehouse-v642-complete-static-program-preparation.v1'
ROOT = Path(__file__).resolve().parents[2]
_FIELDS = tuple(f.name for f in fields(NativeProgram))


def source_files():
    values = native_source_files()
    name = 'backend/training/warehouse_v642_prepared_program.py'
    values[name] = sha256((ROOT / name).read_bytes()).hexdigest()
    return dict(sorted(values.items()))


def _typed(value):
    """Lossless type-and-value binding; JSON equality alone merges 1/1.0/True."""
    kind = type(value)
    if value is None: return ['none']
    if kind is bool: return ['bool', value]
    if kind is int: return ['int', str(value)]
    if kind is float:
        require(math.isfinite(value), 'nonfinite prepared program field')
        return ['float', value.hex()]
    if kind is str: return ['str', value]
    if kind is dict:
        items = [[_typed(k), _typed(v)] for k, v in value.items()]
        items.sort(key=lambda item: json.dumps(item[0], separators=(',', ':'), ensure_ascii=True))
        return ['dict', items]
    if kind in (list, tuple): return [kind.__name__, [_typed(v) for v in value]]
    if kind in (set, frozenset):
        items = [_typed(v) for v in value]
        items.sort(key=lambda v: json.dumps(v, separators=(',', ':'), ensure_ascii=True))
        return [kind.__name__, items]
    raise RuntimeError('unsupported prepared program field type: ' + kind.__name__)


def _typed_sha(value):
    return sha256(json.dumps(_typed(value), separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def _state_sha(program):
    return _typed_sha(tuple((name, getattr(program, name)) for name in _FIELDS))


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate program JSON key')
        value[key] = item
    return value


def _loads(data):
    def nonfinite(value): raise RuntimeError('nonfinite program JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=nonfinite)


def _read(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'} and type(ref['path']) is str
        and type(ref['sha256']) is str and len(ref['sha256']) == 64
        and all(c in '0123456789abcdef' for c in ref['sha256']), 'complete program reference required')
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
        and path.resolve() == path, 'program dependency missing, noncanonical or symlinked')
    data = path.read_bytes()
    require(sha256(data).hexdigest() == ref['sha256'], 'prepared program dependency bytes changed')
    return _loads(data)


def _protection_closure(ref):
    """Only real recursive protection parent references; never trust cached JSON."""
    refs, seen, active = [], {}, set()
    def visit(current):
        path = current.get('path') if type(current) is dict else None
        require(path not in active, 'cyclic program protection parent')
        if path in seen:
            require(current == seen[path], 'conflicting program protection parent')
            return
        value = _read(current); seen[path] = deepcopy(current); active.add(path)
        refs.append((current['path'], current['sha256']))
        require(type(value) is dict, 'program dependency must be a JSON object')
        protection = value.get('protection')
        if protection is not None:
            require(type(protection) is dict and 'parent' in protection, 'incomplete protection parent binding')
            visit(protection['parent'])
        active.remove(path)
    visit(ref)
    return tuple(refs)


@dataclass(frozen=True)
class _Preparation:
    version: str
    fields: tuple
    field_sha256: str
    references: tuple
    sources: tuple
    serialized: bytes
    serialized_sha256: str
    serialized_typed_sha256: str


class PreparedProgram(NativeProgram):
    """Native subclass whose static validation is reusable only while identical.

    Construct with ``from_reference``.  Every ``validate`` / ``to_dict`` checks
    all original fields, actual program/parent bytes and the full source set.
    Other methods are inherited unchanged and calculate their outputs afresh.
    This class neither changes nor certifies current NN evidence.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError('prepared program requires an explicit immutable source reference')

    @classmethod
    def from_dict(cls, *args, **kwargs):
        raise RuntimeError('prepared program requires from_reference, not an unbound dictionary')

    @classmethod
    def from_reference(cls, program_ref, *, target_actor_role_sha256):
        require(cls is PreparedProgram, 'unknown prepared program subclass')
        sources = tuple(source_files().items())
        references = _protection_closure(program_ref)
        encoded = _read(program_ref)
        native = NativeProgram.from_dict(encoded, target_actor_role_sha256)
        native.validate()
        serialized = native.to_dict()
        # Python == would overlook signed zero and numeric type differences.
        require(_typed_sha(serialized) == _typed_sha(encoded), 'native program serialization changes source type or value')
        data = json.dumps(serialized, separators=(',', ':'), allow_nan=False).encode()
        require(_typed_sha(_loads(data)) == _typed_sha(serialized), 'program JSON is not exactly round-trippable')
        out = object.__new__(cls)
        for name in _FIELDS: object.__setattr__(out, name, deepcopy(getattr(native, name)))
        seal = _Preparation(VERSION, _FIELDS, _state_sha(out), references, sources, data,
            sha256(data).hexdigest(), _typed_sha(serialized))
        object.__setattr__(out, '_prepared', seal)
        return out.verify_binding()

    def verify_binding(self):
        require(type(self) is PreparedProgram and type(getattr(self, '_prepared', None)) is _Preparation,
            'missing complete program preparation')
        seal = self._prepared
        require(seal.version == VERSION and seal.fields == _FIELDS
            and set(vars(self)) == set(_FIELDS) | {'_prepared'} and _state_sha(self) == seal.field_sha256,
            'prepared program fields or types changed')
        require(tuple(source_files().items()) == seal.sources, 'prepared program semantic sources changed')
        require(seal.references and _protection_closure(dict(zip(('path', 'sha256'), seal.references[0]))) == seal.references,
            'prepared program protection dependency closure changed')
        require(sha256(seal.serialized).hexdigest() == seal.serialized_sha256,
            'prepared program cached serialization changed')
        return self

    def validate(self):
        return self.verify_binding()

    def to_dict(self):
        self.verify_binding()
        return _loads(self._prepared.serialized)

    def preparation_binding(self):
        self.verify_binding()
        seal = self._prepared
        return dict(version=VERSION, program=dict(zip(('path', 'sha256'), seal.references[0])),
            protection_dependencies=[dict(path=p, sha256=s) for p, s in seal.references[1:]],
            source_files=dict(seal.sources), complete_field_sha256=seal.field_sha256,
            native_serialized_sha256=seal.serialized_sha256,
            target_actor_role_sha256=self.target_actor_role_sha256,
            immutable_anchor_role_sha256=self.actor_role_sha256,
            caches_current_NN_qualification=False, training_authority=False)
