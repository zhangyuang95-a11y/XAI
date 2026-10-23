"""Content-addressed, read-only v642 surrogate evidence with explicit resolution.

Original program and parent JSON bytes are never rewritten. Source paths are
provenance labels only: the resolver reads package-local assets by SHA. No
environment actions, training, fitting, audit authorization or publishing are
provided by this module.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import os

import numpy as np

from . import warehouse_v642_program as native
from .warehouse_v6_actor import FEATURE_NAMES, FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256
from .warehouse_v626_domain import domain_sha
from .warehouse_v626_certificate import verify as verify_leaf
from .warehouse_v622_program import plain
from env.warehouse.navigation import ACTIONS
from experiments.warehouse.v626_protocol import require, pointer, bound, file_sha
from scripts.run_warehouse_v625 import write_new

VERSION = 'warehouse-v642-portable-program-archive.v1'
AUDIT_VERSION = 'warehouse-v642-portable-independent-program-audit.v1'
VALIDATION_VERSION = 'warehouse-v642-portable-explicit-resolver-validation.v1'
digest = native.digest
ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = (
    'backend/training/warehouse_v642_portable_program.py',
    'tests/test_warehouse_v642_portable_program.py',
    'docs/warehouse_v642_portable_program.md',
)
SEMANTIC_NAMES = (
    'backend/training/warehouse_v642_program.py',
    'backend/training/warehouse_v642_guards.py',
    'backend/training/warehouse_v642_certificate.py',
    'backend/training/warehouse_v629_guards.py',
    'backend/training/warehouse_v628_program.py',
    'backend/training/warehouse_v628_certificate.py',
    'backend/training/warehouse_v626_program.py',
    'backend/training/warehouse_v626_domain.py',
    'backend/training/warehouse_v626_certificate.py',
    'backend/training/warehouse_v625_program.py',
    'backend/training/warehouse_v622_program.py',
    'backend/training/warehouse_v619_leaf_fit.py',
    'backend/training/warehouse_v6_actor.py',
    'backend/training/warehouse_v6_tree_routing.py',
    'env/warehouse/public_observation_v6.py',
    'env/warehouse/public_action_features_v614.py',
    'env/warehouse/public_action_features_v617.py',
    'env/warehouse/layouts.py',
    'env/warehouse/navigation.py',
)


def source_files():
    return {name: file_sha(ROOT / name) for name in (*SOURCE_NAMES, *SEMANTIC_NAMES)}


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key in portable evidence')
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON in portable evidence')))
    require(type(value) is dict, 'portable evidence must contain a JSON object')
    return value


def _parent_refs(value):
    """Only explicit parent contracts are transitive runtime dependencies."""
    refs = {}
    for key in ('protection', 'semantic_protection'):
        item = value.get(key)
        if isinstance(item, dict) and 'parent' in item:
            ref = item['parent']
            require(type(ref) is dict and set(ref) == {'path', 'sha256'}
                    and isinstance(ref['path'], str) and is_sha(ref['sha256']), 'invalid protected parent reference')
            refs[digest(ref)] = deepcopy(ref)
    return [refs[key] for key in sorted(refs)]


class ContentResolver:
    """Resolve only the sealed dependency inventory; never source-path fallback."""
    def __init__(self, directory, assets):
        self.directory = Path(directory).resolve() if directory is not None else None
        self.assets = deepcopy(assets)
        self.memory = None

    @classmethod
    def from_memory(cls, raw_by_sha, refs_by_sha):
        out = cls(None, {key: dict(path=f'assets/{key}/asset.json', source_refs=refs_by_sha[key]) for key in raw_by_sha})
        out.memory = dict(raw_by_sha)
        return out

    def raw(self, ref):
        require(type(ref) is dict and set(ref) == {'path', 'sha256'} and is_sha(ref['sha256']), 'invalid source reference')
        key = ref['sha256']
        require(key in self.assets, 'missing content-addressed dependency')
        item = self.assets[key]
        require(type(item) is dict and set(item) == {'path', 'source_refs'}
                and item['path'] == f'assets/{key}/asset.json' and ref in item['source_refs'],
                'dependency pointer differs from the sealed source alias map')
        if self.memory is not None:
            raw = self.memory[key]
        else:
            path = self.directory / item['path']
            require(path.resolve().is_relative_to(self.directory) and path.is_file() and not path.is_symlink()
                    and not path.stat().st_mode & 0o222, 'missing, writable or symlinked portable dependency')
            raw = path.read_bytes()
        require(sha256(raw).hexdigest() == key, 'portable dependency content hash mismatch')
        return raw

    def json(self, ref):
        return _decode(self.raw(ref))

    def verify(self):
        require(self.assets, 'empty portable dependency inventory')
        for key, item in self.assets.items():
            require(is_sha(key) and type(item.get('source_refs')) is list and item['source_refs']
                    and len({digest(ref) for ref in item['source_refs']}) == len(item['source_refs']),
                    'invalid or repeated dependency aliases')
            for ref in item['source_refs']:
                require(ref['sha256'] == key, 'dependency alias hash differs')
                self.raw(ref)


def _protection(program, resolver):
    """Same native protection equations, with one explicit parent resolver."""
    contract = program.protection
    require(type(contract) is dict and set(contract) == {'parent', 'parent_version', 'entries'}, 'missing original protection contract')
    parent_data = resolver.json(contract['parent'])
    # Native v642 deliberately protects a v626 parent; do not interpret an
    # unsupported recursive program version as if it had v626 semantics.
    parent = native.ParentProgram.from_dict(parent_data, program.actor_role_sha256)
    require(contract == native.make_protection(parent, contract['parent']) and parent.role == program.role,
            'original protected regions changed')
    tree, info = native.structure_info(program.structure, program.maximum_leaves, program.maximum_depth)
    legal = program.certificates()
    require(legal == native.guards.leaf_actions(tree, program.maximum_leaves, program.maximum_depth), 'guard fact map differs')
    paths = [native.region(tree, i, program.maximum_leaves, program.maximum_depth)['path'] for i in range(len(program.leaves))]
    all_ids, entries = set(), []
    for item in contract['entries']:
        prefix = item['path']
        ids = [i for i, path in enumerate(paths) if len(path) >= len(prefix) and digest(path[:len(prefix)]) == digest(prefix)]
        require(ids and not all_ids.intersection(ids), 'missing or overlapping original prefix')
        for i in ids:
            require(native.payload(program.leaves[i]) == item['leaf_payload'] and set(item['legal']) <= set(legal[i]),
                    'original protected output or physical fact changed')
        all_ids.update(ids)
        entries.append(dict(parent_leaf=item['parent_leaf'], descendant_leaves=ids))
    return dict(version='warehouse-original-prefix-protection-with-bounded-guards.v1', parent=contract['parent'],
        protected_descendants=sorted(all_ids), entries=entries, structure_sha256=digest(tree), depth=info['depth'])


def validate_program(data, target_role_sha256, resolver):
    """Native from_dict/validate checks, without absolute-path parent loading."""
    keys = {'version', 'role', 'actor_role_sha256', 'target_actor_role_sha256', 'structure', 'leaves', 'arm',
        'maximum_leaves', 'maximum_depth', 'protection', 'report', 'feedback_leaves', 'explanation_leaves',
        'domain_sha256', 'feature_schema_sha256', 'action_physics_config_sha256', 'guard_evidence',
        'semantic_protection', 'external_program_selects_execution_action'}
    require(set(data) == keys and data['version'] == native.VERSION and is_sha(target_role_sha256)
            and data['target_actor_role_sha256'] == target_role_sha256 and is_sha(data['actor_role_sha256'])
            and data['domain_sha256'] == domain_sha() and data['feature_schema_sha256'] == FEATURE_SCHEMA_SHA256
            and data['action_physics_config_sha256'] == ACTION_PHYSICS_CONFIG_SHA256
            and data['external_program_selects_execution_action'] is False, 'unknown or mismatched portable v642 program')
    for key in ('feedback_leaves', 'explanation_leaves'):
        require(type(data[key]) is list and data[key] == sorted(set(data[key])) and all(type(i) is int for i in data[key]),
                'invalid serialized eligibility')
    p = native.Program(data['role'], data['actor_role_sha256'], deepcopy(data['structure']), deepcopy(data['leaves']), data['arm'],
        data['maximum_leaves'], frozenset(data['feedback_leaves']), frozenset(data['explanation_leaves']),
        deepcopy(data['report']), deepcopy(data['protection']), target_role_sha256, data['maximum_depth'])
    tree, _ = native.structure_info(p.structure, p.maximum_leaves, p.maximum_depth)
    require(p.arm == 'C3' and p.role in ('robot_1', 'robot_2') and len(p.leaves) == tree['leaf_count'], 'invalid bounded program identity')
    legal = p.certificates()
    for i, leaf in enumerate(p.leaves):
        verify_leaf(leaf, native.region(tree, i, p.maximum_leaves, p.maximum_depth), legal[i], leaf['certificate'])
        require(leaf['certificate']['use_path'] is True, 'path-conditioned proof required')
    for ids in (p.feedback_leaves, p.explanation_leaves):
        require(all(type(i) is int and 0 <= i < len(p.leaves) for i in ids), 'invalid eligibility IDs')
    if p.report is None:
        require(not p.feedback_leaves and not p.explanation_leaves, 'qualification cannot be inherited')
    else:
        report = p.report
        require(report['maximum_depth'] == p.maximum_depth and report['maximum_leaves'] == p.maximum_leaves
                and report['domain_sha256'] == domain_sha() and report['actor_role_sha256'] == p.actor_role_sha256
                and report['target_actor_role_sha256'] == target_role_sha256 and report['structure_sha256'] == digest(tree)
                and report['external_program_selects_execution_action'] is False, 'qualification target, tree or schema differs')
        for key in ('split_episode_ids', 'split_scenario_ids'):
            groups = report[key]
            require(set(groups) == set(native.legacy.SPLITS) and all(not set(groups[a]) & set(groups[b])
                    for a in groups for b in groups if a < b), 'evidence partitions overlap')
        require(len(report['fit_leaf_counts']) == len(p.leaves) and set(report['eligibility']) == set(map(str, range(len(p.leaves)))),
                'incomplete eligibility evidence')
        for purpose, ids, threshold in (('feedback', p.feedback_leaves, .93), ('explanation', p.explanation_leaves, .95)):
            expected = frozenset(i for i in range(len(p.leaves)) if report['fit_leaf_counts'][i]
                and native._eligible(report['eligibility'][str(i)][purpose], threshold))
            require(expected == ids, 'qualified leaf set differs from evidence')
    require(data['guard_evidence'] == native.guards.tree_evidence(p.structure, p.maximum_leaves, p.maximum_depth)
            and data['semantic_protection'] == _protection(p, resolver), 'physical/protection evidence differs')
    return p


def _dependencies(program_ref):
    raw_by_sha, refs_by_sha, edges, active = {}, {}, [], set()
    def visit(ref):
        key = ref['sha256']
        require(is_sha(key) and key not in active, 'invalid hash or cyclic protected dependency')
        require(pointer(ref['path']) == ref, 'original program dependency changed')
        raw = Path(ref['path']).read_bytes()
        aliases = refs_by_sha.setdefault(key, [])
        if ref not in aliases:
            aliases.append(deepcopy(ref))
        if key in raw_by_sha:
            require(raw_by_sha[key] == raw, 'same-hash dependency bytes differ')
            return
        raw_by_sha[key] = raw
        active.add(key)
        for parent in _parent_refs(_decode(raw)):
            edges.append(dict(source_sha256=key, parent=parent))
            visit(parent)
        active.remove(key)
    visit(program_ref)
    return raw_by_sha, refs_by_sha, edges


def build_archive(program_ref, output, *, target_actor_role_sha256):
    """Package existing bytes only; this neither runs nor grants an audit."""
    raw, aliases, edges = _dependencies(program_ref)
    resolver = ContentResolver.from_memory(raw, aliases)
    resolver.verify()
    data = resolver.json(program_ref)
    program = validate_program(data, target_actor_role_sha256, resolver)
    sources = source_files()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key, value in raw.items():
        path = output / 'assets' / key / 'asset.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        path.chmod(0o444)
    manifest = dict(version=VERSION, validation_version=VALIDATION_VERSION, source_program=deepcopy(program_ref),
        source_program_sha256=program_ref['sha256'], role=program.role,
        anchor_actor_role_sha256=program.actor_role_sha256, target_actor_role_sha256=target_actor_role_sha256,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256, domain_sha256=domain_sha(),
        action_physics_config_sha256=ACTION_PHYSICS_CONFIG_SHA256,
        maximum_depth=program.maximum_depth, maximum_leaves=program.maximum_leaves,
        assets=resolver.assets, parent_dependencies=edges, source_files=sources,
        original_program_bytes_unchanged=True, source_paths_are_provenance_only=True,
        external_program_selects_execution_action=False, independent_audit_included=False,
        scientific_acceptance=False, publishable=False)
    file = output / 'archive.json'
    write_new(file, manifest); file.chmod(0o444)
    write_new(output / 'completed.json', dict(archive_sha256=file_sha(file)))
    (output / 'completed.json').chmod(0o444)
    return pointer(file)


def _audit(receipt_ref, expected_sha256, archive_ref, manifest, program):
    if receipt_ref is None:
        require(expected_sha256 is None, 'required independent audit receipt is missing')
        return None
    require(is_sha(expected_sha256) and receipt_ref['sha256'] == expected_sha256 and pointer(receipt_ref['path']) == receipt_ref,
            'independent audit needs its trusted expected content hash')
    path = Path(receipt_ref['path'])
    require(not path.stat().st_mode & 0o222, 'independent audit receipt is writable')
    value = _decode(path.read_bytes())
    require(value['version'] == AUDIT_VERSION and value['phase'] == 'final_independent_program_audit'
            and value['independent'] is True and value['passed'] is True
            and value['program_archive_sha256'] == archive_ref['sha256']
            and value['source_program_sha256'] == manifest['source_program_sha256']
            and value['role'] == program.role and value['anchor_actor_role_sha256'] == program.actor_role_sha256
            and value['target_actor_role_sha256'] == program.target_actor_role_sha256
            and is_sha(value['one_use_claim_sha256']) and is_sha(value['audit_partition_sha256']),
            'independent audit receipt phase, model, program or archive binding differs')
    leaves = value['explanation_eligible_leaves']
    require(type(leaves) is list and leaves == sorted(set(leaves)) and leaves
            and all(type(i) is int and i in program.explanation_leaves for i in leaves),
            'independent audit added unsupported explanation leaves')
    require(type(value['sample_count']) is int and value['sample_count'] >= 1000, 'independent audit sample support is insufficient')
    require(set(value['metrics']) == {'feedback', 'explanation'}, 'independent audit needs both purpose metrics')
    for purpose, minimum in (('feedback', .85), ('explanation', .90)):
        m = value['metrics'][purpose]
        require(all(type(m[k]) is int for k in ('sample_count', 'rows', 'actionable_rows'))
                and m['sample_count'] == value['sample_count'] and 0 < m['actionable_rows'] <= m['rows'] <= m['sample_count']
                and m['coverage'] == m['rows'] / m['sample_count']
                and m['actionable_coverage'] == m['actionable_rows'] / m['sample_count'],
                'independent audit coverage does not reproduce from its physical frame counts')
        require(all(type(m[k]) in (float, int) and math.isfinite(m[k]) for k in ('coverage', 'actionable_coverage', 'agreement', 'kl')),
                'nonfinite or invalid independent audit metric')
        require(.20 <= m['actionable_coverage'] <= m['coverage'] <= 1 and minimum <= m['agreement'] <= 1 and 0 <= m['kl'] <= .35
                and all(type(m[k]) is int and m[k] == 0 for k in ('illegal_count', 'whole_tree_raw_illegal_count', 'whole_tree_path_uncertified_count')),
                'independent audit formal fidelity/coverage/legality gate failed')
    return deepcopy(value)


class PortableProgram:
    """A surrogate-analysis reader, not an Actor or action-selection policy."""
    @classmethod
    def load(cls, archive_ref, *, expected_source_program_sha256, target_actor_role_sha256,
             independent_audit_ref=None, expected_independent_audit_sha256=None):
        require(pointer(archive_ref['path']) == archive_ref and not Path(archive_ref['path']).stat().st_mode & 0o222,
                'portable archive manifest changed or writable')
        manifest = _decode(Path(archive_ref['path']).read_bytes())
        folder = Path(archive_ref['path']).parent
        require(bound(pointer(folder / 'completed.json')) == dict(archive_sha256=archive_ref['sha256']), 'portable archive completion seal differs')
        require(manifest['version'] == VERSION and manifest['validation_version'] == VALIDATION_VERSION
                and manifest['source_files'] == source_files() and manifest['source_program_sha256'] == expected_source_program_sha256
                and manifest['source_program']['sha256'] == expected_source_program_sha256
                and manifest['target_actor_role_sha256'] == target_actor_role_sha256
                and manifest['domain_sha256'] == domain_sha() and manifest['feature_schema_sha256'] == FEATURE_SCHEMA_SHA256
                and manifest['action_physics_config_sha256'] == ACTION_PHYSICS_CONFIG_SHA256
                and manifest['original_program_bytes_unchanged'] is manifest['source_paths_are_provenance_only'] is True
                and manifest['external_program_selects_execution_action'] is manifest['independent_audit_included'] is False
                and manifest['scientific_acceptance'] is manifest['publishable'] is False, 'portable archive schema, method or model binding differs')
        resolver = ContentResolver(folder, manifest['assets'])
        resolver.verify()
        data = resolver.json(manifest['source_program'])
        program = validate_program(data, target_actor_role_sha256, resolver)
        require(manifest['role'] == program.role and manifest['anchor_actor_role_sha256'] == program.actor_role_sha256
                and manifest['maximum_depth'] == program.maximum_depth and manifest['maximum_leaves'] == program.maximum_leaves,
                'portable archive capacity or anchor differs')
        visited, edges = set(), []
        def visit(ref):
            key = ref['sha256']
            if key in visited: return
            visited.add(key)
            for parent in _parent_refs(resolver.json(ref)):
                edges.append(dict(source_sha256=key, parent=parent))
                visit(parent)
        visit(manifest['source_program'])
        require(visited == set(manifest['assets']) and edges == manifest['parent_dependencies'], 'portable transitive dependency inventory differs')
        out = cls.__new__(cls)
        out._manifest, out._archive_ref, out._resolver = deepcopy(manifest), deepcopy(archive_ref), resolver
        out._program, out._data = program, deepcopy(data)
        out._audit_ref, out._expected_audit_sha256 = deepcopy(independent_audit_ref), expected_independent_audit_sha256
        out._audit = _audit(independent_audit_ref, expected_independent_audit_sha256, archive_ref, manifest, program)
        return out

    def verify_assets(self):
        require(pointer(self._archive_ref['path']) == self._archive_ref
                and _decode(Path(self._archive_ref['path']).read_bytes()) == self._manifest
                and self._manifest['source_files'] == source_files(), 'portable archive or reader sources changed')
        require(bound(pointer(Path(self._archive_ref['path']).with_name('completed.json')))
                == dict(archive_sha256=self._archive_ref['sha256']), 'portable archive completion seal changed')
        self._resolver.verify()
        if self._audit_ref is not None:
            require(pointer(self._audit_ref['path']) == self._audit_ref, 'independent audit receipt changed or missing')

    def distributions(self, x, m):
        """Pure offline numerical comparison; never sends an environment action."""
        self.verify_assets()
        return self._program.distributions(x, m)

    def evidence(self, x, m, executed_NN_action, *, current_actor_role_sha256):
        self.verify_assets()
        require(current_actor_role_sha256 == self._manifest['target_actor_role_sha256'], 'explanation current NN role hash differs')
        if self._audit is None:
            return None
        x, m = np.asarray(x, np.float32), np.asarray(m, np.float32)
        require(x.shape == (len(FEATURE_NAMES),) and m.shape == (5,), 'one pre-action public frame is required')
        if isinstance(executed_NN_action, str):
            require(executed_NN_action.upper() in ACTIONS, 'unknown executed NN action')
            action = ACTIONS.index(executed_NN_action.upper())
        else:
            require(type(executed_NN_action) is int and 0 <= executed_NN_action < 5, 'invalid executed NN action index')
            action = executed_NN_action
        routes, raw, q = self._program.distributions(x[None], m[None])
        leaf, top, raw_top = int(routes[0]), int(q[0].argmax()), int(raw[0].argmax())
        if leaf not in self._audit['explanation_eligible_leaves'] or top != action:
            return None
        require(m[action] == 1 and raw_top in self._program.certificates()[leaf], 'program explanation lacks current physical/path legality')
        path = native.region(self._program.structure, leaf, self._program.maximum_leaves, self._program.maximum_depth)['path']
        public_path = [dict(branch=item['branch'], bias=item['bias'],
            terms=[dict(feature=FEATURE_NAMES[i], weight=weight) for i, weight in enumerate(item['weight']) if weight]) for item in path]
        return dict(version=VERSION, eligible=True, role=self._program.role, leaf_id=leaf,
            source_program_sha256=self._manifest['source_program_sha256'], program_archive_sha256=self._archive_ref['sha256'],
            anchor_actor_role_sha256=self._program.actor_role_sha256, target_actor_role_sha256=current_actor_role_sha256,
            independent_audit_sha256=self._expected_audit_sha256, nn_action=ACTIONS[action].lower(), path=public_path,
            program_probabilities=q[0].tolist(), audited_metrics=deepcopy(self._audit['metrics']['explanation']),
            describes_observed_policy_pattern=True, unique_causal_motive_established=False,
            external_program_selects_execution_action=False)
