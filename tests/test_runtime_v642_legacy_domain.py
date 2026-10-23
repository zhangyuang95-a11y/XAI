"""Carried P/K software replay from archived QA snapshots, never heldout runs."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import pytest

from domains.pong import nn_policy as pn, turnbased as pp
from domains.kitchen import neural_policy as kn, engine_v5_frozen as kp
from runtime_v642.catalog import AssetResolver, PolicyCatalog, VERSION
from runtime_v642.contracts import PolicyBinding, ProgramBinding, RuntimeUnavailable, digest
from runtime_v642.legacy_domain import (
    ROOT, CONFIG_SHA, ORIGINAL_ACCEPTANCE_SHA, PROGRAM_SCHEMAS,
    PongAdapter, KitchenAdapter, domain_contract, load_kitchen_actor,
    FrozenLegacyCore, _saved_kitchen_scene,
)
from runtime_v642.legacy_acceptance import verify_original_assets, REVALIDATION_SCHEMAS, runtime_source_files, EXECUTION_CASE_VERSION
from runtime_v642.registry import BoundDomain

# Used only to reconstruct a seed already present in the immutable historical
# development receipt. It is never used for heldout or a fresh scene argument.
HISTORICAL_PONG_SCHEDULE = pp._schedule


@pytest.fixture(autouse=True)
def forbid_new_scenes_and_old_global_policy_selection(monkeypatch):
    def forbidden(*a, **kw): raise AssertionError('DI software test invoked old selector or fresh scene')
    for obj, names in ((pp, ('initial_state', '_schedule', 'decide', '_plan')),
                       (pn, ('load_model', 'load_bound_model', 'choose')),
                       (kp, ('initial_state', '_scenario', 'decide', 'human_advisor'))):
        for name in names: monkeypatch.setattr(obj, name, forbidden)
    monkeypatch.setattr(kn.FrozenActors, '__init__', forbidden)


@pytest.fixture(scope='module')
def originals():
    old = json.loads((ROOT / 'releases/policylens-three-domain-20261016.v6.0/catalog.json').read_text())
    carried_path = ROOT / 'output/warehouse_v620/carried-domain-evidence-01/manifest.json'
    if not carried_path.is_file(): pytest.skip('spent original receipt archive absent; do not regenerate')
    carried = json.loads(carried_path.read_text())
    out = dict(catalog=old, carried=carried, states={})
    for domain, name in (('pong', 'configs/study_v3_qa_cases.json'), ('kitchen', 'domains/kitchen/neural_qa_cases.json')):
        data = json.loads((ROOT / name).read_text()); cases = data['cases'] if type(data) is dict else data
        unique = {digest(c['state']): c['state'] for c in cases if c['domain'] == domain and not c['state']['terminal']}
        assert len(unique) == 4
        out['states'][domain] = list(unique.values())
    return out


def make_catalog(root, originals, *, revalidation=None, cases_factory=None):
    records = {}; old = originals['catalog']
    def copy(path, expected):
        raw = (ROOT / path).read_bytes(); assert sha256(raw).hexdigest() == expected
        target = root / 'assets' / expected / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): assert target.read_bytes() == raw
        else: target.write_bytes(raw); target.chmod(0o444)
        records[expected] = dict(path=target.relative_to(root).as_posix(), dependencies=[])
        return expected
    def add(value, deps=()):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode(); key = sha256(raw).hexdigest()
        path = root / 'assets' / key / 'asset.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw); path.chmod(0o444)
        records[key] = dict(path=path.relative_to(root).as_posix(), dependencies=sorted(set(deps))); return key
    manifest = dict(version=VERSION, release_id='unit-carried-runtime-migration-not-scientifically-accepted', policies={})
    for domain in ('pong', 'kitchen'):
        domain_assets = set(); pair = {}
        cfg = old['runtime_configs'][domain]; domain_assets.add(copy(cfg['path'], cfg['sha256']))
        for entry in old['policies'][domain].values():
            for a, h in (('actor_path', 'actor_sha256'), ('training_report_path', 'training_report_sha256'), ('acceptance_path', 'acceptance_sha256')):
                domain_assets.add(copy(entry[a], entry[h]))
            for role, path in zip(('human', 'ai'), entry['program_paths']):
                domain_assets.add(copy(path, entry['program_sha256'][path if domain == 'pong' else role]))
        if domain == 'kitchen':
            receipt = json.loads((root / records[ORIGINAL_ACCEPTANCE_SHA[domain]]['path']).read_text())
            needed = {receipt[k] for k in ('pair_report_sha256', 'original_pair_report_sha256',
                'final_program_validation_receipt_sha256', 'heldout_receipt_sha256')}
            for item in originals['carried']['files']:
                if item['sha256'] in needed: domain_assets.add(copy(item['archive_path'], item['sha256']))
        original_sha = ORIGINAL_ACCEPTANCE_SHA[domain]
        records[original_sha]['dependencies'] = sorted(domain_assets - {original_sha})
        unaccepted = add(dict(version='unit-unaccepted-carried-runtime', passed=False, domain=domain), [original_sha])
        for variant in ('raw', 'regularized'):
            entry = old['policies'][domain][variant]; programs = []
            for role, path in zip(('human', 'ai'), entry['program_paths']):
                psha = entry['program_sha256'][path if domain == 'pong' else role]
                programs.append(ProgramBinding(role, role, PROGRAM_SCHEMAS[domain], psha, psha, original_sha,
                    audit_scope='carried_selected_actor_development_audit'))
            pair[variant] = PolicyBinding(manifest['release_id'], entry['policy_id'], domain, variant,
                **domain_contract(domain), actor_sha256=entry['actor_sha256'], human_actor_sha256=entry['role_sha256']['human'],
                ai_actor_sha256=entry['role_sha256']['ai'], programs=tuple(programs), accepted_receipt_sha256=unaccepted).to_dict()
        manifest['policies'][domain] = pair
    source = add(dict(unit_unused_warehouse_program=True)); fake_receipt = add(dict(version='unit-unknown-warehouse', passed=False))
    pair = {}
    for v in ('raw', 'regularized'):
        actor = add(dict(unit_warehouse_actor_not_executed=True, variant=v))
        programs = tuple(ProgramBinding(r, n, 'warehouse-role-program.v6.42-explicit-depth-capacity', source,
            source, None) for r, n in (('human', 'robot_1'), ('ai', 'robot_2'))) if v == 'regularized' else ()
        pair[v] = PolicyBinding(manifest['release_id'], 'warehouse-' + v + '-unit', 'warehouse', v,
            'unit-only', 'unit-only', actor, digest(v + '-h'), digest(v + '-a'), digest('features'), digest('physics'),
            programs, fake_receipt).to_dict()
    manifest['policies']['warehouse'] = pair
    if revalidation is not None:
        domain, mutate = revalidation; old_sha = manifest['policies'][domain]['raw']['accepted_receipt_sha256']
        r = dict(version=REVALIDATION_SCHEMAS[domain], stage='carried_v5_release_revalidation', passed=True,
            release_id=manifest['release_id'], domain=domain, original_acceptance_sha256=ORIGINAL_ACCEPTANCE_SHA[domain],
            runtime_config_sha256=CONFIG_SHA[domain], source_files=runtime_source_files(domain),
            policy_identity_sha256={v: PolicyBinding.from_dict(b).identity_sha256 for v, b in manifest['policies'][domain].items()},
            heldout_rerun=False, new_training=False, new_human_results=False, execution_cases_sha256=[])
        case_refs = []
        if cases_factory is not None:
            resolver = AssetResolver(root, records)
            for case in cases_factory(domain, manifest, resolver):
                case_refs.append(add(case, [case['provenance']['source_sha256']]))
            r['execution_cases_sha256'] = case_refs
        mutate(r); new_sha = add(r, [*records[old_sha]['dependencies'], *case_refs]); del records[old_sha]
        for b in manifest['policies'][domain].values(): b['accepted_receipt_sha256'] = new_sha
    manifest['assets'] = records
    return PolicyCatalog(manifest, AssetResolver(root, records))


@pytest.fixture
def catalog(tmp_path, originals): return make_catalog(tmp_path, originals)


@pytest.mark.parametrize('domain,adapter_type', [('pong', PongAdapter), ('kitchen', KitchenAdapter)])
def test_real_original_receipts_pass_hash_and_original_numeric_checks(catalog, domain, adapter_type):
    result = verify_original_assets(catalog.assets, tuple(catalog.policies[domain].values()))
    assert result['passed'] and result['heldout_evaluated']
    assert not catalog.formal_verdict(domain).passed  # unknown migration receipt remains closed
    assert catalog.binding(domain, 'regularized').programs[1].audit_scope == 'carried_selected_actor_development_audit'


@pytest.mark.parametrize('domain,adapter_type', [('pong', PongAdapter), ('kitchen', KitchenAdapter)])
def test_all_eight_real_old_snapshots_dual_NN_actions_and_physics_exact(catalog, originals, domain, adapter_type):
    adapter = adapter_type(catalog)
    for variant in ('raw', 'regularized'):
        binding = catalog.binding(domain, variant)
        payload = json.loads(catalog.assets.path(binding.actor_sha256).read_text()) if domain == 'pong' else None
        actor = load_kitchen_actor(catalog.assets.path(binding.actor_sha256), binding.actor_sha256) if domain == 'kitchen' else None
        for old in originals['states'][domain]:
            state = adapter.recorded_state(old, binding, provenance=dict(kind='existing_QA_snapshot', snapshot_sha256=digest(old)))
            native = deepcopy(old)
            for _ in range(3):
                if native['terminal']: break
                decision = adapter.decide(state, binding)
                if domain == 'pong':
                    choices = {r: pn.choose_from_payload(native, r, payload) for r in ('human', 'ai')}
                    expected = pp.step_pair(native, choices['human'], choices['ai'])
                else:
                    choices = {r: actor.choose(native, r)[0] for r in ('human', 'ai')}
                    spec = kn.action_spec(choices['ai'])
                    expected = kp.step(native, choices['human'], {**spec, 'policy_origin': 'kitchen_nn',
                        'action_id': choices['ai'], 'reason_code': 'neural_selection', 'memory': {}})
                assert decision['policy_actions'] == choices and decision['ai_action_overrides'] == 0
                state, record = adapter._advance(state, binding, None, decision)
                native = adapter.restore(state, binding)
                assert native == expected and record['selected_actions'] == choices


@pytest.mark.parametrize('domain,adapter_type', [('pong', PongAdapter), ('kitchen', KitchenAdapter)])
def test_demo_takeover_restore_replay_and_counterfactual_same_binding(catalog, originals, domain, adapter_type):
    adapter = adapter_type(catalog); binding = catalog.binding(domain, 'regularized')
    old = originals['states'][domain][0]
    state = adapter.recorded_state(old, binding, provenance=dict(kind='old_QA_frame'))
    demo = adapter.demonstration(binding, state=state, steps=3)
    assert adapter.replay(demo, binding) == demo['final_state']
    commands = [r['selected_actions']['human'] for r in demo['records']]
    counter = adapter.counterfactual(state, commands, binding)
    assert counter['final_state'] == demo['final_state'] and adapter.replay(counter, binding) == counter['final_state']
    decision = adapter.decide(state, binding)
    legal = pp.legal_actions(old) if domain == 'pong' else kp.legal_actions(old)
    human = next(a for a in legal if a != decision['human_action'])
    nxt, record = adapter._advance(state, binding, human, decision)
    assert record['selected_actions'] == dict(human=human, ai=decision['action'])
    assert adapter.restore(state, binding) == old
    public = adapter.public_state(nxt, binding)
    assert public['binding'] == binding.to_dict() and 'snapshot' not in public
    if domain == 'kitchen':
        assert 'handoff_grace' not in public and 'handoff_grace_turns' not in public['rule_metadata']
        assert public['ai']['policy']['actor_sha256'] == binding.actor_sha256
        assert all(r.get('location') != 'plan' for r in public['ai']['current_cooking'])


@pytest.mark.parametrize('domain,adapter_type', [('pong', PongAdapter), ('kitchen', KitchenAdapter)])
def test_A_task2_historical_evidence_only_and_B_C_cannot_explain(catalog, originals, domain, adapter_type):
    adapter = adapter_type(catalog); binding = catalog.binding(domain, 'regularized')
    state = adapter.recorded_state(originals['states'][domain][0], binding, provenance=dict(kind='old_QA_frame'))
    a = BoundDomain(binding, adapter, catalog, 'A', 'test')
    reader = a.explanation_program(task=2)
    for old in originals['states'][domain]:
        state = adapter.recorded_state(old, binding, provenance=dict(kind='old_QA_frame'))
        decision = adapter.decide(state, binding); evidence = adapter.explanation(state, a)
        if evidence is not None:
            assert evidence['nn_action'] == decision['action'] and evidence['new_independent_audit_claimed'] is False
            assert type(evidence['path']) is list and evidence['audited_metrics']['passed'] is True
        wrong = 'not-an-NN-action'
        assert reader.evidence(old, wrong, current_actor_role_sha256=binding.ai_actor_sha256) is None
    assert adapter.explanation(state, BoundDomain(binding, adapter, catalog, 'B', 'test')) is None
    raw = catalog.binding(domain, 'raw')
    other = adapter.recorded_state(originals['states'][domain][0], raw, provenance=dict(kind='old_QA_frame'))
    assert adapter.explanation(other, BoundDomain(raw, adapter, catalog, 'C', 'test')) is None


@pytest.mark.parametrize('domain,adapter_type', [('pong', PongAdapter), ('kitchen', KitchenAdapter)])
@pytest.mark.parametrize('fault', ['state', 'decision', 'snapshot', 'switch', 'actor_asset', 'receipt_asset'])
def test_tamper_missing_assets_and_policy_switch_refused(catalog, originals, domain, adapter_type, fault):
    adapter = adapter_type(catalog); binding = catalog.binding(domain, 'regularized')
    state = adapter.recorded_state(originals['states'][domain][0], binding, provenance=dict(kind='old_QA_frame'))
    decision = adapter.decide(state, binding)
    if fault == 'state': state['turn'] += 1
    elif fault == 'snapshot': state['snapshot']['turn'] += 1
    elif fault == 'switch': binding = catalog.binding(domain, 'raw')
    elif fault in ('actor_asset', 'receipt_asset'):
        path = catalog.assets.path(binding.actor_sha256 if fault == 'actor_asset' else ORIGINAL_ACCEPTANCE_SHA[domain])
        path.chmod(0o644); path.write_bytes(path.read_bytes() + b' '); path.chmod(0o444)
    else: decision['action'] = 'forged'
    with pytest.raises((RuntimeUnavailable, ValueError)):
        adapter.step(state, 'wait', binding, decision=decision)


@pytest.mark.parametrize('domain', ['pong', 'kitchen'])
@pytest.mark.parametrize('fault', ['incomplete', 'failed', 'foreign_source', 'heldout_rerun', 'wrong_policy'])
def test_known_revalidation_schema_is_real_verifier_not_passed_flag(tmp_path, originals, domain, fault):
    def mutate(r):
        if fault == 'failed': r['passed'] = False
        elif fault == 'foreign_source': r['source_files'] = {}
        elif fault == 'heldout_rerun': r['heldout_rerun'] = True
        elif fault == 'wrong_policy': r['policy_identity_sha256']['raw'] = 'f' * 64
    catalog = make_catalog(tmp_path, originals, revalidation=(domain, mutate))
    with pytest.raises(RuntimeUnavailable): catalog.formal_verdict(domain)


def test_neutral_kitchen_practice_is_explicit_non_model_exception():
    info = KitchenAdapter.neutral_practice_metadata()
    assert info['labelled_exception'] and not info['model_evaluation'] and not info['frozen_teammate_experiment']


def historical_cases(domain, manifest, resolver):
    original = json.loads(resolver.path(ORIGINAL_ACCEPTANCE_SHA[domain]).read_text())
    config = json.loads(resolver.path(CONFIG_SHA[domain]).read_text())
    for variant in ('raw', 'regularized'):
        binding = PolicyBinding.from_dict(manifest['policies'][domain][variant])
        core = FrozenLegacyCore(resolver, binding, CONFIG_SHA[domain])
        for task in (1, 2, 3):
            if domain == 'kitchen':
                native = _saved_kitchen_scene(1000, task, config['scenarios']['1000'][str(task)])
                provenance = dict(kind='pinned_saved_scene', source_sha256=CONFIG_SHA[domain])
            else:
                seed = next(row['seed'] for row in original['development'][variant]['rows'] if row['task'] == task)
                assert not 890000 <= seed < 890100
                human, ai = (6, 2) if task == 3 else (2, 6)
                native = pp._new_state(HISTORICAL_PONG_SCHEDULE(seed, task, config), seed=seed,
                    task=task, human=human, ai=ai, config=config)
                provenance = dict(kind='historical_development_schedule', source_sha256=ORIGINAL_ACCEPTANCE_SHA[domain])
            choices = core.choose(native)[0]
            for control in ('nn', 'human'):
                actions = dict(choices); command = None
                if control == 'human':
                    legal = pp.legal_actions(native) if domain == 'pong' else kp.legal_actions(native)
                    command = next(a for a in legal if a != choices['human']); actions['human'] = command
                yield dict(version=EXECUTION_CASE_VERSION, domain=domain, variant=variant,
                    provenance=provenance, native_initial_state=native, human_commands=[command],
                    expected_native_states=[core.transition(native, actions)])


@pytest.mark.parametrize('domain', ['pong', 'kitchen'])
def test_true_verifier_replays_all_12_recorded_domain_paths_without_new_training_or_holdout(tmp_path, originals, domain, monkeypatch):
    if domain == 'pong':
        # Permit only the exact unchanged historical schedule implementation.
        # _spent_source independently rejects seeds absent from the old receipt.
        monkeypatch.setattr(pp, '_schedule', HISTORICAL_PONG_SCHEDULE)
    catalog = make_catalog(tmp_path, originals, revalidation=(domain, lambda r: None), cases_factory=historical_cases)
    verdict = catalog.formal_verdict(domain)
    assert verdict.passed and verdict.schema == REVALIDATION_SCHEMAS[domain]
    assert 'no new heldout or human effect claim' in verdict.reason
    # This positive software revalidation is not a three-domain release.
    with pytest.raises(RuntimeUnavailable, match='pilot remains closed'): catalog.require_pilot()
