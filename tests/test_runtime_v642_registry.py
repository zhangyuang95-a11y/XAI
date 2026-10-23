"""DI wiring with explicit fake formal verifiers; no scientific admission."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
import importlib.util
import json
import shutil

import pytest

from runtime_v642 import PolicyBinding, ProgramBinding, PolicyCatalog, VerifierRegistry, FormalVerdict, RuntimeRegistry, RuntimeUnavailable
from runtime_v642.catalog import AssetResolver, DOMAINS, FORMAL_PHASES, VERSION
from runtime_v642.contracts import ROLES

FAKE_SCHEMA = 'explicit-synthetic-formal-verifier-for-unit-tests.v1'


def hash_text(text):
    return sha256(text.encode()).hexdigest()


class Package:
    def __init__(self, root): self.root, self.records = root, {}
    def add(self, value, dependencies=()):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = sha256(raw).hexdigest(); path = self.root / 'assets' / key / 'asset.json'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o444)
        self.records[key] = dict(path=path.relative_to(self.root).as_posix(), dependencies=sorted(set(dependencies)))
        return key
    def catalog(self, manifest, verifiers=None):
        value = deepcopy(manifest); value['assets'] = deepcopy(self.records)
        raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        path = self.root / 'catalog.json'
        if path.exists(): path.chmod(0o644)
        path.write_bytes(raw); path.chmod(0o444)
        return PolicyCatalog.load(path, expected_sha256=sha256(raw).hexdigest(), verifiers=verifiers)


@pytest.fixture
def package(tmp_path):
    p = Package(tmp_path)
    manifest = dict(version=VERSION, release_id='synthetic-v642-release-unit-tests-only', policies={})
    for domain in DOMAINS:
        pair = {}
        for variant in ('raw', 'regularized'):
            programs = []
            if variant == 'regularized':
                for role in ROLES:
                    source = p.add(dict(synthetic_source_program=True, domain=domain, role=role))
                    archive = p.add(dict(synthetic_archive=True, domain=domain, role=role), (source,))
                    audit = p.add(dict(synthetic_program_audit=True, domain=domain, role=role))
                    programs.append(ProgramBinding(role, 'robot_1' if role == 'human' else 'robot_2',
                        'warehouse-role-program.v6.42-explicit-depth-capacity' if domain == 'warehouse' else domain + '-legacy-program.v5',
                        archive, source, audit))
            actor = p.add(dict(synthetic_actor_bytes=True, domain=domain, variant=variant))
            pair[variant] = PolicyBinding(manifest['release_id'], f'{domain}-{variant}-synthetic', domain, variant,
                f'{domain}-synthetic-model.v1', 'synthetic-method.v1', actor, hash_text(f'{domain}-{variant}-human'),
                hash_text(f'{domain}-{variant}-ai'), hash_text(domain + '-features'), hash_text(domain + '-physics'),
                tuple(programs), 'e' * 64)
        phases = {phase: p.add(dict(synthetic_formal_phase=True, phase=phase, domain=domain)) for phase in FORMAL_PHASES}
        receipt = p.add(dict(version=FAKE_SCHEMA, stage='formal_release_acceptance', passed=True,
            release_id=manifest['release_id'], domain=domain, training_joint_steps=200000,
            formal_pair_count=3, development_stream_count=3, frozen_models=True,
            policy_identity_sha256={v: b.identity_sha256 for v, b in pair.items()}, phase_evidence=phases,
            synthetic_unit_test_only_not_actual_scientific_evidence=True), phases.values())
        manifest['policies'][domain] = {v: replace(b, accepted_receipt_sha256=receipt).to_dict() for v, b in pair.items()}
    return p, manifest


class FakeAdapter:
    def __init__(self, domain): self.DOMAIN, self.calls = domain, []
    def forbidden(self, *a, **kw): raise AssertionError('binding tests must never execute a domain')
    initial_state = demonstration = decide = step = restore = replay = counterfactual = forbidden


def registry(catalog, legacy=None):
    return RuntimeRegistry(catalog, adapters={d: FakeAdapter(d) for d in DOMAINS}, legacy_reader=legacy)


def fake_verifiers(calls=None):
    def fake(context):
        if calls is not None: calls.append(context.domain)
        return FormalVerdict(True, FAKE_SCHEMA, context.receipt_sha256, 'explicit fake for unit tests; not scientific acceptance')
    return VerifierRegistry({FAKE_SCHEMA: fake})


def test_A_B_same_binding_C_independent_and_bindings_frozen(package):
    p, m = package; catalog = p.catalog(m); r = registry(catalog)
    for domain in DOMAINS:
        a, b, c = (r.bind(domain, group, mode='preview') for group in ('A', 'B', 'C'))
        assert a.binding is b.binding and a.binding == b.binding
        assert a.binding.actor_sha256 != c.binding.actor_sha256
        assert a.binding.human_actor_sha256 != c.binding.human_actor_sha256
        assert a.binding.ai_actor_sha256 != c.binding.ai_actor_sha256
        assert 'group' not in a.binding.to_dict()
        with pytest.raises(FrozenInstanceError): a.binding.actor_sha256 = c.binding.actor_sha256
        with pytest.raises(FrozenInstanceError): a.binding.programs[0].archive_sha256 = 'f' * 64
        with pytest.raises(TypeError): catalog.policies[domain]['raw'] = a.binding
        assert not a.adapter.calls


def test_unknown_schema_is_closed_even_when_all_metadata_says_passed(package):
    p, m = package; catalog = p.catalog(m)
    assert not catalog.formal_verdict('warehouse').passed
    assert catalog.formal_verdict('warehouse').reason == 'unknown_formal_acceptance_schema'
    with pytest.raises(RuntimeUnavailable, match='pilot remains closed'):
        registry(catalog).bind('warehouse', 'A', mode='pilot')


def test_only_explicit_fake_verifier_accepts_synthetic_formal_fixture(package):
    p, m = package; calls = []; catalog = p.catalog(m, fake_verifiers(calls))
    bound = registry(catalog).bind('warehouse', 'A', mode='pilot')
    assert calls == list(DOMAINS) and bound.mode == 'pilot'
    assert not bound.adapter.calls


def replace_receipt(p, manifest, domain, mutate):
    m = deepcopy(manifest); old = m['policies'][domain]['raw']['accepted_receipt_sha256']
    receipt = json.loads((p.root / p.records[old]['path']).read_text())
    mutate(receipt)
    dependencies = p.records[old]['dependencies']
    new = p.add(receipt, dependencies)
    del p.records[old]
    for v in ('raw', 'regularized'): m['policies'][domain][v]['accepted_receipt_sha256'] = new
    return m


@pytest.mark.parametrize('fault', ['20k', 'calibration', 'missing_phase', 'wrong_policy', 'missing_pair'])
def test_even_registered_fake_cannot_bypass_generic_formal_requirements(package, fault):
    p, m = package
    def mutate(value):
        if fault == '20k': value['training_joint_steps'] = 20000
        elif fault == 'calibration': value['stage'] = 'paired_20k_calibration'
        elif fault == 'missing_phase': value['phase_evidence'].pop('heldout')
        elif fault == 'wrong_policy': value['policy_identity_sha256']['raw'] = 'f' * 64
        else: value['formal_pair_count'] = 1
    m = replace_receipt(p, m, 'warehouse', mutate)
    catalog = p.catalog(m, fake_verifiers())
    with pytest.raises(RuntimeUnavailable): catalog.require_pilot()


def test_boolean_verifier_result_is_rejected(package):
    p, m = package
    catalog = p.catalog(m, VerifierRegistry({FAKE_SCHEMA: lambda context: True}))
    with pytest.raises(RuntimeUnavailable, match='unbound or unknown'):
        catalog.require_pilot()


@pytest.mark.parametrize('fault', ['deleted', 'tampered'])
def test_missing_or_tampered_asset_invalidates_existing_binding(package, fault):
    p, m = package; catalog = p.catalog(m); r = registry(catalog)
    bound = r.bind('warehouse', 'A', mode='test')
    path = p.root / p.records[bound.binding.actor_sha256]['path']
    if fault == 'deleted': path.unlink()
    else: path.chmod(0o644); path.write_bytes(path.read_bytes() + b' '); path.chmod(0o444)
    with pytest.raises(RuntimeUnavailable): bound.verify()


@pytest.mark.parametrize('fault', ['missing_dependency', 'cycle', 'traversal', 'wrong_raw_role'])
def test_dependency_inventory_and_pair_identity_are_strict(package, fault):
    p, m = package
    raw = m['policies']['warehouse']['raw']; reg = m['policies']['warehouse']['regularized']
    if fault == 'missing_dependency': p.records[raw['actor_sha256']]['dependencies'] = ['f' * 64]
    elif fault == 'cycle':
        p.records[raw['actor_sha256']]['dependencies'] = [reg['actor_sha256']]
        p.records[reg['actor_sha256']]['dependencies'] = [raw['actor_sha256']]
    elif fault == 'traversal': p.records[raw['actor_sha256']]['path'] = '../outside.json'
    else: raw['human_actor_sha256'] = reg['human_actor_sha256']
    with pytest.raises(RuntimeUnavailable): p.catalog(m)


def test_legacy_interface_is_read_export_only(package):
    p, m = package
    class Legacy:
        def read_instance(self, instance_id): return dict(id=instance_id, release_id='old', frames=[dict(turn=0)])
        def export_instance(self, instance_id): return dict(id=instance_id, archived=True)
    r = registry(p.catalog(m), Legacy())
    value = r.archived_instance('old-id')
    assert value['release_id'] == 'old' and r.archived_instance('old-id', export=True)['archived']
    with pytest.raises(TypeError): value['frames'][0]['turn'] = 1
    assert not hasattr(r, 'step') and not hasattr(r, 'resume_legacy')


def test_group_and_task_gate_never_loads_program_for_B_C_or_Task1(package, monkeypatch):
    p, m = package; catalog = p.catalog(m); r = registry(catalog)
    monkeypatch.setattr(catalog, 'program', lambda *args: pytest.fail('forbidden group/task read program'))
    for group, task in (('B', 2), ('C', 2), ('A', 1), ('A', 3)):
        assert r.bind('warehouse', group, mode='test').explanation_program(task=task) is None


def test_missing_independent_audit_blocks_explanation_and_pilot(package):
    p, m = package
    for program in m['policies']['warehouse']['regularized']['programs']:
        del p.records[program['independent_audit_sha256']]
        program['independent_audit_sha256'] = None
    identities = {v: PolicyBinding.from_dict(row).identity_sha256 for v, row in m['policies']['warehouse'].items()}
    m = replace_receipt(p, m, 'warehouse', lambda receipt: receipt.update(policy_identity_sha256=identities))
    catalog = p.catalog(m, fake_verifiers())
    assert registry(catalog).bind('warehouse', 'A', mode='test').explanation_program(task=2) is None
    with pytest.raises(RuntimeUnavailable, match='independent program audit'):
        catalog.require_pilot()


def test_portable_program_loader_uses_target_role_and_trusted_audit_binding(tmp_path):
    """Use actual synthetic portable bytes, without environments or NN inference."""
    spec = importlib.util.spec_from_file_location('runtime_portable_fixture', Path(__file__).with_name('test_warehouse_v642_portable_program.py'))
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    program, parts, source, archive = fixture = helper.fixture.__wrapped__(tmp_path)
    from runtime_v642.catalog import AssetResolver
    # A dedicated minimal catalog stub exercises only this typed loader; it
    # cannot grant pilot and is not presented as a full accepted release.
    records = {}
    archive_folder = Path(archive['path']).parent
    for path in archive_folder.rglob('*'):
        if path.is_file():
            key = sha256(path.read_bytes()).hexdigest()
            records[key] = dict(path=path.relative_to(tmp_path).as_posix(), dependencies=[])
    records[archive['sha256']]['dependencies'] = sorted(set(records) - {archive['sha256']})
    resolver = AssetResolver(tmp_path, records)
    b = PolicyBinding('synthetic', 'warehouse-reg-synthetic', 'warehouse', 'regularized', 'synthetic-model', 'synthetic-method',
        'e' * 64, 'b' * 64, 'f' * 64, '1' * 64, '2' * 64,
        (ProgramBinding('human', 'robot_1', 'warehouse-role-program.v6.42-explicit-depth-capacity', archive['sha256'], source['sha256'], None),
         ProgramBinding('ai', 'robot_2', 'warehouse-role-program.v6.42-explicit-depth-capacity', archive['sha256'], source['sha256'], None)), '3' * 64)
    catalog = PolicyCatalog.__new__(PolicyCatalog); catalog.assets = resolver
    catalog.binding = lambda domain, variant: b
    reader = catalog.program(b, 'human')
    data = parts['validation']
    assert reader.evidence(data['x'][0], data['m'][0], 'wait', current_actor_role_sha256='b' * 64) is None
    with pytest.raises(RuntimeError): catalog.program(b, 'ai')
    audit = helper.audit_fixture(tmp_path, fixture)
    resolver.records = type(resolver.records)({**dict(resolver.records), audit['sha256']:
        type(next(iter(resolver.records.values())))(Path(audit['path']).relative_to(tmp_path).as_posix(), (archive['sha256'],))})
    b = replace(b, programs=(replace(b.programs[0], independent_audit_sha256=audit['sha256']), b.programs[1]))
    reader = catalog.program(b, 'human')
    assert reader.evidence(data['x'][0], data['m'][0], 'wait', current_actor_role_sha256='b' * 64)['eligible']
    # Files still exist and native proof validation passes, but removing the
    # declared archive closure must stop this higher-level runtime catalog.
    resolver.records = type(resolver.records)({**dict(resolver.records), archive['sha256']:
        type(next(iter(resolver.records.values())))(records[archive['sha256']]['path'], ())})
    with pytest.raises(RuntimeUnavailable, match='undeclared or relocated'):
        catalog.program(b, 'human')
