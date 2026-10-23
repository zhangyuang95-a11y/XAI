"""Revalidate pinned historical v5 evidence without rerunning its heldout set.

The only waiver is of Warehouse-specific new training stages. Original domain
quality/program criteria, actual asset hashes and new runtime replay remain
mandatory. A passed flag or injected callback cannot manufacture this gate.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import math

from core.program import ExecutableProgram
from study_v3.policies import _candidate_training_valid, _selected_feedback_matches, _heldout_quality_valid

from .catalog import FormalVerdict, decode
from .contracts import digest, require, is_sha
from .legacy_domain import (
    ROOT, CONFIG_SHA, ORIGINAL_ACCEPTANCE_SHA, PROGRAM_SCHEMAS,
    FrozenLegacyCore, PongAdapter, KitchenAdapter, _saved_kitchen_scene,
)

REVALIDATION_SCHEMAS = {d: f'runtime-v642-carried-{d}-revalidation.v1' for d in CONFIG_SHA}
EXECUTION_CASE_VERSION = 'runtime-v642-carried-recorded-execution-case.v1'
UPSTREAM_RELEASE_ID = 'policylens-three-domain-20261014.v5.0'
QA_SOURCE_SHA = dict(pong='2170cbf412f1ad1eb52480da1739b767c9ad774472decd5e73a46ba76411f2ad',
    kitchen='140e902fb114a5cf1baa92591c68ee049a39435d1cfd454a3bcae5f6b232500a')


def _metric(metric, *, fidelity, kl, count='sample_count'):
    require(type(metric) is dict and metric.get('passed') is True
        and type(metric.get(count)) is int and metric[count] >= 1000
        and all(type(metric.get(k)) in (int, float) and math.isfinite(metric[k])
            for k in ('action_fidelity', 'mean_kl_divergence'))
        and fidelity <= metric['action_fidelity'] <= 1 and 0 <= metric['mean_kl_divergence'] <= kl,
        'historical program metrics fail their original domain gate')


def verify_original_assets(assets, bindings):
    require(len(bindings) == 2 and {b.variant for b in bindings} == {'raw', 'regularized'}
        and len({b.domain for b in bindings}) == 1 and bindings[0].domain in CONFIG_SHA, 'invalid carried policy pair')
    domain = bindings[0].domain; pair = {b.variant: b for b in bindings}
    acceptance_sha = ORIGINAL_ACCEPTANCE_SHA[domain]
    original = decode(assets.path(acceptance_sha).read_bytes())
    require(original.get('release_id', original.get('release')) == UPSTREAM_RELEASE_ID
        and original.get('passed') is True and original.get('heldout_evaluated') is True
        and original.get('post_policy_override_count') == 0
        and original.get('joint_steps') == dict(raw=200000, regularized=200000), 'historical v5 acceptance is incomplete')
    seeds = original.get('pair_seeds')
    require(type(seeds) is list and len(seeds) == len(set(seeds)) == 3 and original.get('selected_pair_seed') in seeds
        and _candidate_training_valid(domain, original, seeds) and _selected_feedback_matches(domain, original)
        and _heldout_quality_valid(domain, original), 'historical paired training or heldout numeric evidence fails')
    for variant, binding in pair.items():
        require(original['artifact_sha256'][variant] == binding.actor_sha256, 'carried Actor differs from accepted v5 artifact')
        assets.path(binding.actor_sha256)
        assets.path(original['training_report_sha256'][variant])
    require(original.get('raw_program_feedback_rows') == dict(human=0, ai=0)
        and set(original.get('regularized_program_feedback_rows', {})) == {'human', 'ai'}
        and all(type(v) is int and v > 0 for v in original['regularized_program_feedback_rows'].values()), 'historical feedback differs')
    reg = pair['regularized']
    for role in ('human', 'ai'):
        p = next(p for p in reg.programs if p.role == role)
        require(p.native_role == role and p.reader_schema == PROGRAM_SCHEMAS[domain]
            and p.audit_scope == 'carried_selected_actor_development_audit'
            and p.independent_audit_sha256 == acceptance_sha
            and p.archive_sha256 == p.source_program_sha256 == original['program_sha256']['regularized'][role],
            'legacy program or explicit historical-audit scope differs')
        payload = decode(assets.path(p.source_program_sha256).read_bytes())
        program = ExecutableProgram.from_dict(payload['program'] if domain == 'pong' else payload)
        require(program.root.depth() <= 5 and program.root.leaf_count() <= 16, 'legacy program complexity exceeds original bound')
        if domain == 'pong':
            audit = payload.get('selected_actor_development_audit')
            require(audit == original['selected_actor_development_program_audit'][role]
                and payload.get('selected_actor_artifact_sha256') == reg.actor_sha256
                and payload.get('selected_role_sha256') == reg.role_hash(role)
                and payload.get('role') == role and payload.get('policy_id') == reg.policy_id
                and audit.get('purpose') == ('explanation' if role == 'ai' else 'training_feedback'),
                'Pong selected-Actor development audit provenance differs')
            _metric(audit, fidelity=.95 if role == 'ai' else .75, kl=.10 if role == 'ai' else .60)
        else:
            audit = original['final_role_program_development_audit'][role]
            _metric(audit, fidelity=.80, kl=.35, count='samples')
            require(audit.get('role') == role and audit.get('actor_sha256') == reg.actor_sha256
                and audit.get('role_sha256') == reg.role_hash(role) and audit.get('program_sha256') == p.source_program_sha256
                and audit.get('development_seeds') == list(range(41000, 41020))
                and program.metadata.get('role') == role
                and program.metadata.get('program_phase') == 'final_frozen_actor_training_only_reextraction'
                and program.metadata.get('actor_sha256') == reg.actor_sha256
                and program.metadata.get('role_sha256') == reg.role_hash(role)
                and program.metadata.get('runtime_control_allowed') is False, 'Kitchen final selected-role program provenance differs')
    if domain == 'pong':
        require(all(original['development'][v]['passed'] is True and original['runtime_parity'][v] is True
            for v in ('raw', 'regularized')), 'Pong historical development/parity failed')
    else:
        require(original.get('verification_errors') == [] and original.get('pre_audit_quality_passed') is True
            and original['final_ai_program_development_audit'] == original['final_role_program_development_audit']['ai'],
            'Kitchen historical finalization failed')
        validation = decode(assets.path(original['final_program_validation_receipt_sha256']).read_bytes())
        heldout = decode(assets.path(original['heldout_receipt_sha256']).read_bytes())
        for key in ('pair_report_sha256', 'original_pair_report_sha256'): assets.path(original[key])
        require(validation.get('status') == 'completed' and validation.get('passed') is True
            and validation.get('selected_pair_seed') == original['selected_pair_seed']
            and validation.get('program_sha256') == original['program_sha256']['regularized']
            and heldout.get('status') == 'completed' and heldout.get('selected_pair_seed') == original['selected_pair_seed']
            and heldout.get('actor_sha256') == original['artifact_sha256']
            and heldout.get('final_program_validation_receipt_sha256') == original['final_program_validation_receipt_sha256']
            and heldout.get('heldout') == original['heldout'] and heldout.get('seed_start') == 50000
            and heldout.get('seed_count') == 100, 'Kitchen original one-use program/heldout receipt differs')
    return original


def runtime_source_files(domain):
    require(domain in CONFIG_SHA, 'unknown carried source domain')
    names = ['runtime_v642/contracts.py', 'runtime_v642/catalog.py', 'runtime_v642/registry.py',
        'runtime_v642/legacy_domain.py', 'runtime_v642/legacy_acceptance.py', 'core/program.py', 'study_v3/policies.py']
    names += (['domains/pong/nn_policy.py', 'domains/pong/turnbased.py'] if domain == 'pong' else
        ['domains/kitchen/neural_policy.py', 'domains/kitchen/engine_v5_frozen.py'])
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}


def _contains_snapshot(value, snapshot):
    if type(value) is dict:
        return value == snapshot or any(_contains_snapshot(v, snapshot) for v in value.values())
    if type(value) is list: return any(_contains_snapshot(v, snapshot) for v in value)
    return False


def _spent_source(assets, domain, case, original):
    p = case.get('provenance', {}); initial = case['native_initial_state']
    seed, task = initial.get('seed'), initial['task']
    require(not (domain == 'pong' and type(seed) is int and 890000 <= seed < 890100)
        and not (domain == 'kitchen' and type(seed) is int and 50000 <= seed < 50100),
        'runtime migration must not rerun an already used heldout scene')
    if p.get('kind') == 'archived_full_snapshot':
        require(p.get('source_sha256') == QA_SOURCE_SHA[domain], 'unknown historical full-snapshot source')
        source = decode(assets.path(p['source_sha256']).read_bytes()) if domain == 'pong' else None
        if domain == 'kitchen':
            import json
            source = json.loads(assets.path(p['source_sha256']).read_bytes())
        require(_contains_snapshot(source, initial), 'execution initial frame is absent from pinned spent source')
    elif domain == 'kitchen' and p.get('kind') == 'pinned_saved_scene':
        require(p.get('source_sha256') == CONFIG_SHA[domain] and type(seed) is int and seed in range(1000, 1024),
            'only existing spent development menus may be reconstructed')
        config = decode(assets.path(CONFIG_SHA[domain]).read_bytes())
        scene = config.get('scenarios', {}).get(str(seed), {}).get(str(task))
        require(type(scene) is dict and initial == _saved_kitchen_scene(seed, task, scene), 'saved Kitchen reconstruction differs')
    elif domain == 'pong' and p.get('kind') == 'historical_development_schedule':
        require(p.get('source_sha256') == ORIGINAL_ACCEPTANCE_SHA[domain]
            and (task, seed) in {(row['task'], row['seed']) for result in original['development'].values() for row in result['rows']},
            'Pong scene was not in the already consumed development evidence')
        from domains.pong import turnbased as physics
        config = decode(assets.path(CONFIG_SHA[domain]).read_bytes())
        human, ai = (6, 2) if task == 3 else (2, 6)
        require(initial == physics._new_state(physics._schedule(seed, task, config), seed=seed,
            task=task, human=human, ai=ai, config=config), 'historical Pong schedule reconstruction differs')
    else:
        require(False, 'migration needs an explicitly pinned spent snapshot; fresh scenarios are forbidden')


def verify_revalidation(context):
    """Read-only true verifier: recomputes NN/physics, never trusts passed JSON."""
    domain = context.domain; schema = REVALIDATION_SCHEMAS[domain]
    require(context.catalog is not None, 'carried verifier requires the real explicit catalog')
    r = decode(context.assets.path(context.receipt_sha256).read_bytes())
    require(r.get('version') == schema and r.get('stage') == 'carried_v5_release_revalidation' and r.get('passed') is True
        and r.get('release_id') == context.release_id and r.get('domain') == domain
        and r.get('original_acceptance_sha256') == ORIGINAL_ACCEPTANCE_SHA[domain]
        and r.get('runtime_config_sha256') == CONFIG_SHA[domain]
        and r.get('source_files') == runtime_source_files(domain)
        and r.get('policy_identity_sha256') == {b.variant: b.identity_sha256 for b in context.bindings}
        and r.get('heldout_rerun') is False and r.get('new_training') is False and r.get('new_human_results') is False,
        'carried revalidation source, scope or policy identity differs')
    original = verify_original_assets(context.assets, context.bindings)
    cases = r.get('execution_cases_sha256')
    require(type(cases) is list and 12 <= len(cases) <= 96 and len(cases) == len(set(cases))
        and all(is_sha(s) for s in cases), 'complete recorded execution cases required')
    adapter = PongAdapter(context.catalog) if domain == 'pong' else KitchenAdapter(context.catalog)
    pair = {b.variant: b for b in context.bindings}; coverage = set()
    for case_sha in cases:
        case = decode(context.assets.path(case_sha).read_bytes()); variant = case.get('variant')
        require(case.get('version') == EXECUTION_CASE_VERSION and case.get('domain') == domain and variant in pair,
            'unregistered migration case domain/variant')
        commands, expected = case.get('human_commands'), case.get('expected_native_states')
        require(type(commands) is list and 1 <= len(commands) <= 12 and type(expected) is list
            and len(commands) == len(expected) and (all(c is None for c in commands) or all(type(c) is str for c in commands)),
            'invalid finite NN/takeover migration case')
        _spent_source(context.assets, domain, case, original)
        binding = pair[variant]; initial = adapter.recorded_state(case['native_initial_state'], binding, provenance=case['provenance'])
        current = deepcopy(initial); records = []; nn_pair = commands[0] is None
        for command, target in zip(commands, expected):
            decision = adapter.decide(current, binding)
            current, record = adapter._advance(current, binding, command, decision); records.append(record)
            require(adapter.restore(current, binding) == target and record['ai_action_overrides'] == 0,
                'new carried runtime differs from archived exact NN/physics transition')
        from .legacy_domain import REPLAY_VERSION
        bundle = dict(version=REPLAY_VERSION, binding=binding.to_dict(), initial_state=initial,
            records=records, final_state=current, two_frozen_NN=nn_pair)
        require(adapter.replay(bundle, binding) == current, 'recorded resume/replay equivalence failed')
        if nn_pair:
            require(adapter.demonstration(binding, state=initial, steps=len(commands)) == bundle,
                'two-NN demonstration differs from archived execution')
        else:
            require(adapter.counterfactual(initial, commands, binding) == bundle, 'bound counterfactual differs from recorded execution')
        coverage.add((variant, initial['task'], 'nn' if nn_pair else 'human'))
    require(coverage == {(v, task, control) for v in pair for task in (1, 2, 3) for control in ('nn', 'human')},
        'migration lacks a variant/task/control execution path')
    return FormalVerdict(True, schema, context.receipt_sha256,
        'pinned v5 evidence and actual migrated NN/physics replay verified; no new heldout or human effect claim')
