"""Immutable runtime identities; experimental group is never an Actor input."""
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
import json

DOMAINS = ('warehouse', 'pong', 'kitchen')
ROLES = ('human', 'ai')
GROUP_VARIANTS = MappingProxyType({'A': 'regularized', 'B': 'regularized', 'C': 'raw'})


class RuntimeUnavailable(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeUnavailable(message)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ProgramBinding:
    role: str
    native_role: str
    reader_schema: str
    archive_sha256: str
    source_program_sha256: str
    independent_audit_sha256: str | None
    audit_scope: str = 'independent_final_program_audit'

    def __post_init__(self):
        require(self.role in ROLES and isinstance(self.native_role, str) and bool(self.native_role)
            and isinstance(self.reader_schema, str) and bool(self.reader_schema)
            and is_sha(self.archive_sha256) and is_sha(self.source_program_sha256)
            and self.audit_scope in ('independent_final_program_audit', 'carried_selected_actor_development_audit')
            and (self.independent_audit_sha256 is None or is_sha(self.independent_audit_sha256)), 'invalid program binding')

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__slots__}


@dataclass(frozen=True, slots=True)
class PolicyBinding:
    release_id: str
    policy_id: str
    domain: str
    variant: str
    model_schema: str
    method_schema: str
    actor_sha256: str
    human_actor_sha256: str
    ai_actor_sha256: str
    feature_schema_sha256: str
    action_physics_config_sha256: str
    programs: tuple[ProgramBinding, ...]
    accepted_receipt_sha256: str

    def __post_init__(self):
        require(all(isinstance(getattr(self, key), str) and getattr(self, key) for key in
            ('release_id', 'policy_id', 'model_schema', 'method_schema')) and self.domain in DOMAINS
            and self.variant in ('raw', 'regularized'), 'invalid policy identity')
        require(all(is_sha(getattr(self, key)) for key in ('actor_sha256', 'human_actor_sha256', 'ai_actor_sha256',
            'feature_schema_sha256', 'action_physics_config_sha256', 'accepted_receipt_sha256')),
            'invalid policy artifact or role hash')
        require(self.human_actor_sha256 != self.ai_actor_sha256, 'two Actor roles must have independent weights')
        require(type(self.programs) is tuple and all(type(p) is ProgramBinding for p in self.programs)
            and tuple(p.role for p in self.programs) == (ROLES if self.variant == 'regularized' else ()),
            'regularized needs both ordered role programs; raw exposes no explanation programs')

    def role_hash(self, role):
        require(role in ROLES, 'unknown Actor role')
        return self.human_actor_sha256 if role == 'human' else self.ai_actor_sha256

    def identity(self):
        # The acceptance receipt binds this identity. Excluding its own hash
        # avoids a circular binding without excluding any runtime model field.
        return {key: ([p.to_dict() for p in self.programs] if key == 'programs' else getattr(self, key))
                for key in self.__slots__ if key != 'accepted_receipt_sha256'}

    @property
    def identity_sha256(self):
        return digest(self.identity())

    def to_dict(self):
        return {**self.identity(), 'accepted_receipt_sha256': self.accepted_receipt_sha256}

    @classmethod
    def from_dict(cls, value):
        require(type(value) is dict and set(value) == set(cls.__slots__), 'unknown policy binding fields')
        return cls(**{**value, 'programs': tuple(ProgramBinding(**p) for p in value['programs'])})
