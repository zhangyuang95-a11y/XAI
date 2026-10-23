"""Read-only content resolution and schema-specific formal admission injection."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
import json

from .contracts import DOMAINS, ROLES, PolicyBinding, RuntimeUnavailable, freeze, require, is_sha

VERSION = 'runtime-v642-policy-catalog.v1'
FORMAL_PHASES = ('method_lock', 'budget_lock', 'development_streams', 'formal_pairs', 'quality_selection',
    'program_preaudit', 'independent_program_audit', 'heldout', 'execution_consistency', 'engineering_gate')


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate catalog/evidence JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise RuntimeUnavailable('nonfinite catalog/evidence JSON')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    require(type(value) is dict, 'catalog/evidence must be a JSON object')
    return value


@dataclass(frozen=True, slots=True)
class AssetRecord:
    path: str
    dependencies: tuple[str, ...]

    def __post_init__(self):
        path = Path(self.path)
        require(not path.is_absolute() and self.path and '..' not in path.parts and path.as_posix() == self.path,
                'asset path must remain inside its content-addressed package')
        require(type(self.dependencies) is tuple and self.dependencies == tuple(sorted(set(self.dependencies)))
                and all(is_sha(s) for s in self.dependencies), 'invalid dependency inventory')


class AssetResolver:
    """Explicit SHA dependency graph, with no network or training-path fallback."""
    def __init__(self, directory, records):
        directory = Path(directory)
        require(directory.is_dir() and not directory.is_symlink(), 'missing or symlinked package root')
        self.root = directory.resolve()
        self.records = MappingProxyType({key: AssetRecord(row['path'], tuple(row['dependencies'])) for key, row in records.items()})
        require(self.records and all(is_sha(key) for key in self.records), 'invalid asset hash inventory')
        require(len({record.path for record in self.records.values()}) == len(self.records), 'one package path has conflicting content hashes')
        for key, record in self.records.items():
            require(set(record.dependencies) <= self.records.keys(), 'asset has a missing content-addressed dependency')

    def path(self, key):
        require(is_sha(key) and key in self.records, 'required asset hash is absent')
        record = self.records[key]
        path = self.root / record.path
        require(path.resolve().is_relative_to(self.root) and path.is_file()
            and not any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(self.root))
            and not path.stat().st_mode & 0o222, 'missing, writable or symlinked package asset')
        require(sha256(path.read_bytes()).hexdigest() == key, 'package asset hash mismatch')
        return path

    def pointer(self, key):
        return {'path': str(self.path(key)), 'sha256': key}

    def json(self, key):
        return freeze(decode(self.path(key).read_bytes()))

    def closure(self, roots):
        seen, active = set(), set()
        def visit(key):
            require(key not in active, 'cyclic package dependency graph')
            if key in seen:
                return
            self.path(key)
            active.add(key)
            for dependency in self.records[key].dependencies:
                visit(dependency)
            active.remove(key); seen.add(key)
        for root in roots:
            visit(root)
        return frozenset(seen)


@dataclass(frozen=True, slots=True)
class FormalContext:
    release_id: str
    domain: str
    receipt_sha256: str
    receipt: object
    bindings: tuple[PolicyBinding, PolicyBinding]
    assets: AssetResolver


@dataclass(frozen=True, slots=True)
class FormalVerdict:
    passed: bool
    schema: str
    receipt_sha256: str
    reason: str

    def __post_init__(self):
        require(type(self.passed) is bool and isinstance(self.schema, str) and is_sha(self.receipt_sha256)
                and isinstance(self.reason, str), 'invalid formal verifier result')


class VerifierRegistry:
    """No default scientific verifier exists; unknown receipt schemas fail closed."""
    def __init__(self, verifiers=None):
        self._verifiers = MappingProxyType(dict(verifiers or {}))
        require(all(isinstance(k, str) and k and callable(v) for k, v in self._verifiers.items()), 'invalid explicit verifier registry')

    def verify(self, context):
        receipt = context.receipt
        schema = receipt.get('version', '')
        if schema not in self._verifiers:
            return FormalVerdict(False, schema, context.receipt_sha256, 'unknown_formal_acceptance_schema')
        # These generic invariants never delegate away to a permissive hook.
        require(receipt.get('stage') == 'formal_release_acceptance' and receipt.get('passed') is True
            and receipt.get('release_id') == context.release_id and receipt.get('domain') == context.domain
            and receipt.get('training_joint_steps') in (200000, 300000, 400000)
            and type(receipt.get('training_joint_steps')) is int
            and receipt.get('formal_pair_count') == 3 and type(receipt.get('formal_pair_count')) is int
            and receipt.get('development_stream_count') == 3 and type(receipt.get('development_stream_count')) is int
            and receipt.get('frozen_models') is True,
            'calibration, partial training or placeholder receipt cannot grant pilot admission')
        require(dict(receipt.get('policy_identity_sha256', {})) == {b.variant: b.identity_sha256 for b in context.bindings},
                'formal receipt is not bound to both exact policies')
        phases = receipt.get('phase_evidence')
        require(hasattr(phases, 'keys') and set(phases) == set(FORMAL_PHASES)
                and all(is_sha(phases[name]) for name in FORMAL_PHASES), 'full formal evidence inventory is required')
        context.assets.closure(phases.values())
        for binding in context.bindings:
            require(all(p.independent_audit_sha256 is not None for p in binding.programs),
                    'formal regularized policy lacks its independent program audit')
        verdict = self._verifiers[schema](context)
        require(type(verdict) is FormalVerdict and verdict.schema == schema and verdict.receipt_sha256 == context.receipt_sha256,
                'injected verifier returned an unbound or unknown result')
        return verdict


class PolicyCatalog:
    def __init__(self, manifest, resolver, *, verifiers=None):
        require(manifest.get('version') == VERSION and isinstance(manifest.get('release_id'), str)
                and manifest['release_id'] and set(manifest.get('policies', {})) == set(DOMAINS), 'incomplete three-domain v642 catalog')
        self.release_id = manifest['release_id']
        self.assets = resolver
        self.verifiers = verifiers if verifiers is not None else VerifierRegistry()
        require(type(self.verifiers) is VerifierRegistry, 'explicit read-only verifier registry required')
        values = {}
        for domain in DOMAINS:
            pair = manifest['policies'][domain]
            require(set(pair) == {'raw', 'regularized'}, 'paired raw/regularized policies required')
            raw, reg = (PolicyBinding.from_dict(pair[v]) for v in ('raw', 'regularized'))
            require(all(b.release_id == self.release_id and b.domain == domain and b.variant == v
                    for b, v in ((raw, 'raw'), (reg, 'regularized'))), 'policy and catalog release identity differ')
            require(raw.policy_id != reg.policy_id and raw.actor_sha256 != reg.actor_sha256
                and len({raw.human_actor_sha256, raw.ai_actor_sha256, reg.human_actor_sha256, reg.ai_actor_sha256}) == 4,
                'raw and regularized need independent paired Actor artifacts and role hashes')
            require(all(getattr(raw, key) == getattr(reg, key) for key in ('model_schema', 'method_schema',
                'feature_schema_sha256', 'action_physics_config_sha256')), 'paired policy observation, architecture or method differs')
            require(raw.accepted_receipt_sha256 == reg.accepted_receipt_sha256, 'paired policies need the same formal receipt')
            values[domain] = MappingProxyType({'raw': raw, 'regularized': reg})
        self.policies = MappingProxyType(values)
        require(len({b.policy_id for pair in values.values() for b in pair.values()}) == 2 * len(DOMAINS),
                'policy IDs must be unique across all three domains')
        self.verify_assets()

    @classmethod
    def load(cls, path, *, expected_sha256, verifiers=None):
        path = Path(path)
        require(is_sha(expected_sha256) and path.is_file() and not path.is_symlink()
                and not path.stat().st_mode & 0o222 and sha256(path.read_bytes()).hexdigest() == expected_sha256,
                'v642 catalog hash mismatch, missing or writable')
        manifest = decode(path.read_bytes())
        out = cls(manifest, AssetResolver(path.parent, manifest['assets']), verifiers=verifiers)
        out._catalog_ref = {'path': str(path.resolve()), 'sha256': expected_sha256}
        return out

    def verify_assets(self):
        if hasattr(self, '_catalog_ref'):
            ref = self._catalog_ref
            require(sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], 'v642 catalog changed after binding')
        roots = set()
        for pair in self.policies.values():
            for binding in pair.values():
                roots.update((binding.actor_sha256, binding.accepted_receipt_sha256))
                for p in binding.programs:
                    roots.update((p.archive_sha256, p.source_program_sha256))
                    if p.independent_audit_sha256:
                        roots.add(p.independent_audit_sha256)
        require(self.assets.closure(roots) == self.assets.records.keys(), 'catalog contains missing or unreferenced assets')

    def binding(self, domain, variant):
        self.verify_assets()
        require(domain in self.policies and variant in ('raw', 'regularized'), 'unknown domain or policy variant')
        return self.policies[domain][variant]

    def formal_verdict(self, domain):
        self.verify_assets()
        require(domain in DOMAINS, 'unknown formal domain')
        pair = self.policies[domain]
        ref = pair['regularized'].accepted_receipt_sha256
        return self.verifiers.verify(FormalContext(self.release_id, domain, ref, self.assets.json(ref),
            (pair['raw'], pair['regularized']), self.assets))

    def require_pilot(self):
        verdicts = tuple(self.formal_verdict(domain) for domain in DOMAINS)
        require(all(v.passed for v in verdicts), 'three-domain formal acceptance is incomplete; pilot remains closed')
        return verdicts

    def program(self, binding, role):
        """Warehouse portable evidence only; foreign languages need their DI reader."""
        from backend.training.warehouse_v642_portable_program import PortableProgram
        require(binding == self.binding(binding.domain, binding.variant) and role in ROLES, 'unregistered program binding')
        if binding.variant == 'raw':
            return None
        p = next(p for p in binding.programs if p.role == role)
        require(binding.domain == 'warehouse' and p.reader_schema == 'warehouse-role-program.v6.42-explicit-depth-capacity',
                'program reader schema is not registered by this v642 reader')
        reader = PortableProgram.load(self.assets.pointer(p.archive_sha256),
            expected_source_program_sha256=p.source_program_sha256, target_actor_role_sha256=binding.role_hash(role),
            independent_audit_ref=None if p.independent_audit_sha256 is None else self.assets.pointer(p.independent_audit_sha256),
            expected_independent_audit_sha256=p.independent_audit_sha256)
        require(reader._manifest['role'] == p.native_role, 'program native role differs from immutable binding')
        # The archive reader verifies its internal bytes. Also require every
        # byte it consumed to belong to the catalog's declared SHA closure.
        archive_folder = self.assets.path(p.archive_sha256).parent
        completed = archive_folder / 'completed.json'
        completed_sha = sha256(completed.read_bytes()).hexdigest()
        needed = {**{key: archive_folder / item['path'] for key, item in reader._manifest['assets'].items()},
                  completed_sha: completed}
        closure = self.assets.closure((p.archive_sha256,))
        require(set(needed) <= closure and all(self.assets.path(key) == path for key, path in needed.items()),
                'portable archive consumed an undeclared or relocated dependency')
        if p.independent_audit_sha256:
            require(p.archive_sha256 in self.assets.closure((p.independent_audit_sha256,)),
                    'independent audit dependency graph does not bind its program archive')
        return reader
