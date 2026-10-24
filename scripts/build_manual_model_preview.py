"""Package exact historical Actors for manual play, never scientific admission."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, MODEL_VERSION, FEATURE_SCHEMA_SHA256
from env.warehouse.public_action_features_v617 import ACTION_PHYSICS_CONFIG_SHA256
from runtime_v642.catalog import AssetResolver, PolicyCatalog, VERSION
from runtime_v642.contracts import PolicyBinding, ProgramBinding
from runtime_v642.legacy_domain import domain_contract, PROGRAM_SCHEMAS, ORIGINAL_ACCEPTANCE_SHA

RELEASE = 'policylens-manual-model-preview-20261024.v1'
DESTINATION = ROOT / 'releases' / RELEASE


def build():
    DESTINATION.mkdir(parents=True, exist_ok=False)
    records = {}

    def copy(path, expected=None):
        source = ROOT / path
        raw = source.read_bytes()
        key = sha256(raw).hexdigest()
        if expected is not None and key != expected:
            raise ValueError('source asset identity differs: ' + str(path))
        if key not in records:
            target = DESTINATION / 'assets' / key / source.name
            target.parent.mkdir(parents=True)
            target.write_bytes(raw)
            target.chmod(0o444)
            records[key] = dict(path=target.relative_to(DESTINATION).as_posix(), dependencies=[])
        return key

    def add(value, dependencies=()):
        raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
        key = sha256(raw).hexdigest()
        target = DESTINATION / 'assets' / key / 'preview-receipt.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
        target.chmod(0o444)
        records[key] = dict(path=target.relative_to(DESTINATION).as_posix(), dependencies=sorted(dependencies))
        return key

    manifest = dict(version=VERSION, release_id=RELEASE, policies={},
                    purpose='manual_researcher_preview', pilot_opened=False, model_acceptance_passed=False)
    old = json.loads((ROOT / 'releases/policylens-three-domain-20261016.v6.0/catalog.json').read_text())
    carried = json.loads((ROOT / 'output/warehouse_v620/carried-domain-evidence-01/manifest.json').read_text())
    for domain in ('pong', 'kitchen'):
        dependencies = set()
        cfg = old['runtime_configs'][domain]
        dependencies.add(copy(cfg['path'], cfg['sha256']))
        for entry in old['policies'][domain].values():
            for name in ('actor', 'training_report', 'acceptance'):
                dependencies.add(copy(entry[name + '_path'], entry[name + '_sha256']))
            for role, path in zip(('human', 'ai'), entry['program_paths']):
                dependencies.add(copy(path, entry['program_sha256'][path if domain == 'pong' else role]))
        original = ORIGINAL_ACCEPTANCE_SHA[domain]
        if domain == 'kitchen':
            receipt = json.loads((DESTINATION / records[original]['path']).read_text())
            needed = {receipt[k] for k in ('pair_report_sha256', 'original_pair_report_sha256',
                'final_program_validation_receipt_sha256', 'heldout_receipt_sha256')}
            for item in carried['files']:
                if item['sha256'] in needed:
                    dependencies.add(copy(item['archive_path'], item['sha256']))
        records[original]['dependencies'] = sorted(dependencies - {original})
        unaccepted = add(dict(version='manual-preview-not-formal-admission.v1', passed=False,
            domain=domain, historical_acceptance_sha256=original), (original,))
        pair = {}
        for variant, entry in old['policies'][domain].items():
            programs = []
            for role, path in zip(('human', 'ai'), entry['program_paths']):
                key = entry['program_sha256'][path if domain == 'pong' else role]
                programs.append(ProgramBinding(role, role, PROGRAM_SCHEMAS[domain], key, key, original,
                    audit_scope='carried_selected_actor_development_audit'))
            pair[variant] = PolicyBinding(RELEASE, entry['policy_id'], domain, variant,
                **domain_contract(domain), actor_sha256=entry['actor_sha256'],
                human_actor_sha256=entry['role_sha256']['human'], ai_actor_sha256=entry['role_sha256']['ai'],
                programs=tuple(programs), accepted_receipt_sha256=unaccepted).to_dict()
        manifest['policies'][domain] = pair

    source = ROOT / 'output/warehouse_v642/engineering/accelerated-postchecks-08/audited-assets'
    expected = dict(raw='ca7bfddfca1eb89d3c73935cd22d3828600009053e80663cca58e66de9703677',
                    regularized='2ff0ed2778ecc14b3cd170aef4eb0410ab6a30382e9109b94fb0531874c93ab1')
    program_root = ROOT / 'output/warehouse_v642/training-final-programs/warehouse-v642-accelerated-final-08/regularized'
    programs = []
    for role, native in (('human', 'robot_1'), ('ai', 'robot_2')):
        key = copy(program_root / native / 'program.json')
        programs.append(ProgramBinding(role, native, 'warehouse-role-program.v6.42-explicit-depth-capacity', key, key, None))
    receipt = add(dict(version='manual-preview-not-formal-admission.v1', passed=False, domain='warehouse',
        model_version='v6.42', joint_steps=dict(raw=20000, regularized=20000),
        calibration_passed=False, reason='raw_shutdown_rate_5_of_75_exceeds_5_percent',
        scientific_evaluation=False, explanations_available=False),
        [p.source_program_sha256 for p in programs])
    pair = {}
    for variant in ('raw', 'regularized'):
        path = source / (variant + '-final.actor.npz')
        copy(path, expected[variant])
        actor = FrozenWarehouseV6Actor.load(path, expected_sha256=expected[variant])
        pair[variant] = PolicyBinding(RELEASE, 'warehouse-' + variant + '-v642-20k-preview', 'warehouse', variant,
            MODEL_VERSION, 'warehouse-v642-v620-preserving-calibration', expected[variant],
            actor.role_sha256['robot_1'], actor.role_sha256['robot_2'], FEATURE_SCHEMA_SHA256,
            ACTION_PHYSICS_CONFIG_SHA256, tuple(programs) if variant == 'regularized' else (), receipt).to_dict()
    manifest['policies']['warehouse'] = pair
    manifest['assets'] = records
    PolicyCatalog(manifest, AssetResolver(DESTINATION, records))
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    (DESTINATION / 'catalog.json').write_bytes(raw)
    (DESTINATION / 'catalog.json').chmod(0o444)
    print(json.dumps(dict(catalog=str(DESTINATION / 'catalog.json'), sha256=sha256(raw).hexdigest(),
                         assets=len(records), pilot_opened=False)))


if __name__ == '__main__':
    build()
