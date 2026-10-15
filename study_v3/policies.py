"""Immutable, per-instance bindings for the three-domain frozen NN study."""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from threading import RLock

from . import RELEASE_ID
from .registry import MODULES, engine
from .release_catalog import (UPSTREAM_RELEASE_ID, ReleaseAssetUnavailable,
                              warehouse_evidence, warehouse_policy_artifacts,
                              warehouse_evidence_revision,
                              warehouse_release_spec,
                              warehouse_runtime_source_closure_sha256)


ROOT = Path(__file__).resolve().parents[1]
GROUP_VARIANT = {'A': 'regularized', 'B': 'regularized', 'C': 'raw'}
WAREHOUSE_ROLES = frozenset(('robot_1', 'robot_2'))
_WAREHOUSE_ACCEPTANCE_CACHE: set[tuple] = set()
_WAREHOUSE_ACCEPTANCE_CACHE_LOCK = RLock()


class PolicyUnavailable(RuntimeError):
    """A required frozen policy or its audit evidence is unavailable."""


def policy_id(domain: str, group: str) -> str:
    if domain not in MODULES or group not in GROUP_VARIANT:
        raise ValueError('invalid_policy_assignment')
    return _catalog_entry(domain, GROUP_VARIANT[group])['policy_id']


def _path(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PolicyUnavailable('policy_artifact_path_missing')
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise PolicyUnavailable('policy_artifact_outside_repository')
    return path


def _digest(value: str) -> str:
    path = _path(value)
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PolicyUnavailable('policy_artifact_missing') from exc


def _catalog_entry(domain: str, variant: str) -> dict:
    module = engine(domain)
    try:
        catalog = (warehouse_policy_artifacts() if domain == 'warehouse'
                   else module.policy_artifacts(RELEASE_ID))
    except (AttributeError, OSError, RuntimeError, ValueError,
            ReleaseAssetUnavailable) as exc:
        raise PolicyUnavailable('policy_catalog_unavailable') from exc
    entry = catalog.get(variant)
    expected = f'{domain}-{variant}-v6' if domain == 'warehouse' else f'{domain}-{variant}-v5'
    if not isinstance(entry, dict) or entry.get('policy_id') != expected:
        raise PolicyUnavailable('policy_catalog_incomplete')
    return entry


def binding_for_group(domain: str, group: str) -> dict:
    """Verify and freeze the exact Actor pair before a new instance is stored."""
    variant = GROUP_VARIANT[group]
    entry = _catalog_entry(domain, variant)
    return _binding_from_entry(domain, variant, entry, entry['policy_id'])


def _binding_from_entry(domain: str, variant: str, entry: dict, ident: str) -> dict:
    if entry.get('policy_id') != ident:
        raise PolicyUnavailable('policy_catalog_incomplete')
    expected = entry.get('actor_sha256')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise PolicyUnavailable('actor_hash_missing')
    actual = _digest(entry.get('actor_path'))
    if actual != expected:
        raise PolicyUnavailable('actor_hash_mismatch')
    roles = entry.get('role_sha256')
    if not isinstance(roles, dict) or any(
        not isinstance(roles.get(role), str) or len(roles[role]) != 64
        or any(c not in '0123456789abcdef' for c in roles[role])
        for role in ('human', 'ai')
    ):
        raise PolicyUnavailable('role_hash_missing')
    program_paths = entry.get('program_paths') or []
    if variant == 'regularized' and len(program_paths) != 2:
        raise PolicyUnavailable('role_programs_missing')
    program_hashes = {path: _digest(path) for path in program_paths}
    binding = {'policy_id': ident, 'domain': domain, 'variant': variant,
               'actor_path': entry['actor_path'], 'actor_sha256': actual,
               'human_actor_sha256': roles['human'], 'ai_actor_sha256': roles['ai'],
               'program_hashes': program_hashes}
    if domain == 'warehouse' and ident.endswith('-v6'):
        from backend.training.warehouse_v6_actor import MODEL_VERSION, FEATURE_SCHEMA_SHA256
        if (entry.get('model_version') != MODEL_VERSION
                or entry.get('feature_schema_sha256') != FEATURE_SCHEMA_SHA256):
            raise PolicyUnavailable('warehouse_v6_actor_schema_mismatch')
        binding['actor_model_version'] = MODEL_VERSION
        binding['feature_schema_sha256'] = FEATURE_SCHEMA_SHA256
        try:
            _validated_v6_actor(str(_path(binding['actor_path'])), binding['actor_sha256'],
                                binding['human_actor_sha256'], binding['ai_actor_sha256'])
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise PolicyUnavailable('warehouse_v6_actor_artifact_invalid') from exc
    elif domain == 'warehouse' and entry.get('model_version') == _structured_model_version():
        from backend.training.warehouse_structured_v5 import (
            STRUCTURED_FEATURE_SCHEMA_SHA256, STRUCTURED_MODEL_VERSION)
        if (entry.get('model_version') != STRUCTURED_MODEL_VERSION or
                entry.get('feature_schema_sha256') != STRUCTURED_FEATURE_SCHEMA_SHA256):
            raise PolicyUnavailable('structured_actor_schema_mismatch')
        binding['actor_model_version'] = STRUCTURED_MODEL_VERSION
        binding['feature_schema_sha256'] = STRUCTURED_FEATURE_SCHEMA_SHA256
        try:
            _validated_structured_actor(
                str(_path(binding['actor_path'])), binding['actor_sha256'],
                binding['human_actor_sha256'], binding['ai_actor_sha256'])
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise PolicyUnavailable('structured_actor_artifact_invalid') from exc
    elif domain == 'warehouse':
        from env.warehouse.pure_nn import DUAL_MODEL_VERSION
        if entry.get('model_version') != DUAL_MODEL_VERSION:
            raise PolicyUnavailable('warehouse_actor_model_unknown')
    return binding


def _structured_model_version() -> str:
    from backend.training.warehouse_structured_v5 import STRUCTURED_MODEL_VERSION
    return STRUCTURED_MODEL_VERSION


@lru_cache(maxsize=8)
def _validated_v6_actor(path: str, artifact_sha256: str,
                        robot_1_sha256: str, robot_2_sha256: str) -> None:
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    FrozenWarehouseV6Actor.load(path, expected_sha256=artifact_sha256,
                                expected_role_sha256={'robot_1': robot_1_sha256,
                                                      'robot_2': robot_2_sha256})


@lru_cache(maxsize=8)
def _validated_structured_actor(path: str, artifact_sha256: str,
                                robot_1_sha256: str, robot_2_sha256: str) -> None:
    from backend.training.warehouse_structured_v5 import FrozenStructuredDualNN
    FrozenStructuredDualNN.load(
        path, expected_sha256=artifact_sha256,
        expected_role_sha256={"robot_1": robot_1_sha256,
                              "robot_2": robot_2_sha256})


def binding_for_warehouse_structured(ident: str) -> dict:
    """Bind an explicit candidate without changing active A/B/C assignment."""
    variants = {'warehouse-structured-raw-v54': 'raw',
                'warehouse-structured-regularized-v54': 'regularized'}
    variant = variants.get(ident)
    if variant is None:
        raise PolicyUnavailable('invalid_structured_policy_id')
    module = engine('warehouse')
    try:
        entry = module.structured_policy_artifacts()[variant]
    except (AttributeError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise PolicyUnavailable('structured_policy_catalog_unavailable') from exc
    binding = _binding_from_entry('warehouse', variant, entry, ident)
    if not entry.get('readiness') or not _accepted(entry, binding):
        raise PolicyUnavailable('structured_policy_not_accepted')
    return binding


def verify_binding(binding: dict) -> dict:
    """Fail closed if saved identity and current pinned assets ever diverge."""
    domain = binding.get('domain')
    variant = binding.get('variant')
    if domain not in MODULES or variant not in ('raw', 'regularized'):
        raise PolicyUnavailable('invalid_saved_policy')
    if domain == 'warehouse' and binding.get('policy_id', '').startswith(
            'warehouse-structured-'):
        current = binding_for_warehouse_structured(binding['policy_id'])
    else:
        group = 'C' if variant == 'raw' else 'A'
        current = binding_for_group(domain, group)
    for key in ('policy_id', 'actor_path', 'actor_sha256',
                'human_actor_sha256', 'ai_actor_sha256', 'program_hashes',
                'actor_model_version', 'feature_schema_sha256'):
        if current.get(key) != binding.get(key):
            raise PolicyUnavailable('saved_policy_changed')
    return current


def policy_manifest() -> dict:
    """Return verifiable provenance, including explicit missing-asset status."""
    domains = {}
    for domain in MODULES:
        domains[domain] = {}
        for variant in ('regularized', 'raw'):
            try:
                entry = _catalog_entry(domain, variant)
                binding = binding_for_group(domain, 'C' if variant == 'raw' else 'A')
                paths = [entry.get('training_report_path'), entry.get('acceptance_path')]
                hashes = {path: _digest(path) for path in paths if path}
                domains[domain][variant] = {**binding, 'evidence_sha256': hashes,
                    'accepted': _accepted(entry, binding)}
            except (PolicyUnavailable, AttributeError, OSError, ValueError) as exc:
                domains[domain][variant] = {'policy_id': f'{domain}-{variant}-' +
                                             ('v6' if domain == 'warehouse' else 'v5'),
                                             'accepted': False, 'error': str(exc)}
    return domains


def _candidate_training_valid(domain: str, acceptance: dict, seeds: list) -> bool:
    candidates = acceptance.get('candidate_training')
    if not isinstance(candidates, list) or len(candidates) != 3:
        return False
    if {item.get('pair_seed') for item in candidates if isinstance(item, dict)} != set(seeds):
        return False
    for item in candidates:
        if domain == 'warehouse':
            variants = item.get('variants') or {}
            raw, regularized = variants.get('raw') or {}, variants.get('regularized') or {}
            schedules = [row.get('scenario_schedule_sha256') for row in (raw, regularized)]
            initial = item.get('initial_role_sha256') or {}
            changed = (_warehouse_role_hashes_valid(initial)
                       and all(_warehouse_candidate_variant_valid(row, initial, variant)
                               for variant, row in (('raw', raw), ('regularized', regularized))))
            feedback = raw.get('feedback_rows') or {}
            regularized_feedback = regularized.get('feedback_rows') or {}
            overrides = all(row.get('post_policy_interventions') == 0
                            for row in (raw, regularized))
        elif domain == 'pong':
            raw, regularized = item.get('raw') or {}, item.get('regularized') or {}
            schedules = [row.get('seed_schedule_sha256') for row in (raw, regularized)]
            changed = (raw.get('bc_checkpoint_sha256') == regularized.get('bc_checkpoint_sha256')
                       and raw.get('initial_actor_sha256') == regularized.get('initial_actor_sha256')
                       and all(type(value) in (int, float) and value > 0
                               for row in (raw, regularized)
                               for value in (row.get('actor_weight_change_l2') or {}).values())
                       and all(len(row.get('actor_weight_change_l2') or {}) == 2
                               for row in (raw, regularized)))
            feedback = raw.get('program_feedback_rows') or {}
            regularized_feedback = regularized.get('program_feedback_rows') or {}
            overrides = all(row.get('post_policy_override_count') == 0
                            for row in (raw, regularized))
        elif domain == 'kitchen':
            steps = item.get('joint_steps') or {}
            schedules_map = item.get('expanded_scenario_seed_sha256') or {}
            schedules = [schedules_map.get('raw'), schedules_map.get('regularized')]
            raw, regularized = ({'joint_steps': steps.get(variant)} for variant in ('raw', 'regularized'))
            changes = item.get('actor_l2_change') or {}
            changed = (isinstance(item.get('shared_initial_bc_actor_sha256'), str)
                       and all(set((changes.get(variant) or {})) == {'human', 'ai'}
                               and all(type(value) in (int, float) and value > 0
                                       for value in changes[variant].values())
                               for variant in ('raw', 'regularized')))
            feedback = (item.get('program_feedback_rows') or {}).get('raw') or {}
            regularized_feedback = (item.get('program_feedback_rows') or {}).get('regularized') or {}
            overrides = item.get('post_policy_override_count') == 0
        else:
            return False
        if (raw.get('joint_steps') != 200000 or regularized.get('joint_steps') != 200000
                or not all(isinstance(value, str) and len(value) == 64 for value in schedules)
                or schedules[0] != schedules[1] or not changed or not overrides
                or len(feedback) != 2 or any(type(value) is not int or value != 0
                                              for value in feedback.values())
                or len(regularized_feedback) != 2
                or any(type(value) is not int or value <= 0
                       for value in regularized_feedback.values())):
            return False
    return True


def _warehouse_role_hashes_valid(hashes: dict) -> bool:
    return (isinstance(hashes, dict) and set(hashes) == WAREHOUSE_ROLES
            and all(isinstance(value, str) and len(value) == 64
                    and all(char in '0123456789abcdef' for char in value)
                    for value in hashes.values()))


def _warehouse_candidate_variant_valid(row: dict, initial: dict, variant: str) -> bool:
    """Check the candidate's shared BC start and audited role-row accounting."""
    if not isinstance(row, dict) or row.get('initial_role_sha256') != initial:
        return False
    selected = row.get('selected_role_sha256')
    changes = row.get('actor_weight_delta_l2')
    changed_parameters = row.get('actor_weight_changed_parameter_count')
    if (not _warehouse_role_hashes_valid(selected)
            or not isinstance(changes, dict) or set(changes) != WAREHOUSE_ROLES
            or not isinstance(changed_parameters, dict)
            or set(changed_parameters) != WAREHOUSE_ROLES
            or row.get('both_roles_changed') is not True
            or type(row.get('joint_steps')) is not int
            or row.get('actor_decisions') != 2 * row['joint_steps']):
        return False
    for role in WAREHOUSE_ROLES:
        delta, count = changes[role], changed_parameters[role]
        if (selected[role] == initial[role]
                or type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0
                or type(count) is not int or count <= 0):
            return False
    counters = [row.get(field) for field in ('feedback_rows', 'selected_feedback_rows',
                'feedback_window_actor_rows', 'zero_action_coverage_rows',
                'physical_no_legal_action_rows')]
    if any(not isinstance(values, dict) or set(values) != WAREHOUSE_ROLES
           or any(type(value) is not int or value < 0 for value in values.values())
           for values in counters):
        return False
    feedback, selected_feedback, exposure, zero, no_legal = counters
    for role in WAREHOUSE_ROLES:
        if (selected_feedback[role] > feedback[role]
                or feedback[role] > exposure[role]
                or zero[role] != exposure[role] - feedback[role]
                or no_legal[role] != 0):
            return False
        if variant == 'raw' and any(values[role] != 0 for values in counters):
            return False
        if variant == 'regularized' and (feedback[role] <= 0 or exposure[role] <= 0):
            return False
    return True


def _selected_feedback_matches(domain: str, acceptance: dict) -> bool:
    selected = next((item for item in acceptance['candidate_training']
                     if item['pair_seed'] == acceptance['selected_pair_seed']), None)
    if selected is None:
        return False
    if domain == 'warehouse':
        rows = selected['variants']
        raw = rows['raw']['feedback_rows']
        regularized = rows['regularized']['feedback_rows']
        selected_feedback = rows['regularized']['selected_feedback_rows']
        if (selected_feedback != acceptance.get('selected_regularized_feedback_rows')
                or set(selected_feedback) != {'robot_1', 'robot_2'}
                or any(type(value) is not int or value <= 0
                       for value in selected_feedback.values())
                or any(selected_feedback[role] > regularized[role]
                       for role in selected_feedback)):
            return False
    elif domain == 'pong':
        raw = selected['raw']['program_feedback_rows']
        regularized = selected['regularized']['program_feedback_rows']
    elif domain == 'kitchen':
        raw = selected['program_feedback_rows']['raw']
        regularized = selected['program_feedback_rows']['regularized']
    else:
        return False
    return (raw == acceptance.get('raw_program_feedback_rows')
            and regularized == acceptance.get('regularized_program_feedback_rows'))


def _heldout_quality_valid(domain: str, acceptance: dict) -> bool:
    heldout = acceptance.get('heldout') or {}
    try:
        if domain == 'warehouse':
            base_seeds = set(range(250000, 250100))
            expected = {task: {seed * 10 + task for seed in base_seeds}
                        for task in (1, 2, 3)}
            for variant in ('raw', 'regularized'):
                result = heldout[variant]
                rows = result['episodes']
                summary = result['summary']
                observed = {
                    'mean_deliveries': sum(row['deliveries'] for row in rows) / len(rows),
                    'mean_collisions': sum(row['collisions'] for row in rows) / len(rows),
                    'shutdown_episode_rate': sum(bool(row['shutdown_episode'])
                                                 for row in rows) / len(rows),
                }
                if (len(rows) != 300 or any(len(result['tasks'][str(task)]['episodes']) != 100
                                            for task in (1, 2, 3))
                        or any({(row['seed'], row['scenario_seed']) for row in rows
                                if row['task'] == task} != {(seed, seed * 10 + task)
                                                           for seed in base_seeds}
                               for task in (1, 2, 3))
                        or any({row['seed'] for row in result['tasks'][str(task)]['episodes']}
                               != expected[task] for task in (1, 2, 3))
                        or any(abs(summary[key] - value) > 1e-6
                               for key, value in observed.items())
                        or summary['mean_deliveries'] < 5
                        or summary['mean_collisions'] > 8
                        or summary['shutdown_episode_rate'] > .10):
                    return False
            return True
        if domain == 'pong':
            expected = {(task, seed) for task in (1, 2, 3)
                        for seed in range(890000, 890100)}
            for variant in ('raw', 'regularized'):
                result = heldout[variant]
                if (len(result['rows']) != 300 or len(result['summaries']) != 3
                        or {(row['task'], row['seed']) for row in result['rows']} != expected
                        or {row['task'] for row in result['summaries']} != {1, 2, 3}):
                    return False
                for summary in result['summaries']:
                    task_rows = [row for row in result['rows']
                                 if row['task'] == summary['task']]
                    observed = {
                        'mean_capacity_percent': sum(row['capacity_percent'] for row in task_rows) / len(task_rows),
                        'min_capacity_percent': min(row['capacity_percent'] for row in task_rows),
                        'mean_ai_wait_rate': sum(row['ai_wait_rate'] for row in task_rows) / len(task_rows),
                        'max_ai_wait_rate': max(row['ai_wait_rate'] for row in task_rows),
                    }
                    if (summary['mean_capacity_percent'] < 90
                            or summary['min_capacity_percent'] < 80
                            or summary['mean_ai_wait_rate'] > .10
                            or summary['max_ai_wait_rate'] > .20
                            or len(task_rows) != 100
                            or any(abs(summary[key] - value) > 1e-6
                                   for key, value in observed.items())):
                        return False
            return True
        if domain == 'kitchen':
            for variant in ('raw', 'regularized'):
                for partner in ('nn', 'proxy'):
                    result = heldout[variant][partner]
                    episodes, successes = result['episodes'], result['successes']
                    if (type(episodes) is not int or episodes != 300
                            or type(successes) is not int or not 240 <= successes <= episodes
                            or not isinstance(result['failures'], list)
                            or len(result['failures']) != episodes - successes
                            or type(result['completion_rate']) not in (int, float)
                            or not math.isfinite(result['completion_rate'])
                            or abs(result['completion_rate'] - successes / episodes) > 1e-6
                            or result['partner'] != partner
                            or result['seeds'] != list(range(50000, 50100))):
                        return False
            return True
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return False


def _development_and_audit_valid(domain: str, acceptance: dict) -> bool:
    """Require the final frozen pair's model and explanation checks explicitly."""
    try:
        if domain == 'warehouse':
            selected = next(item for item in acceptance['development']
                            if item['pair_seed'] == acceptance['selected_pair_seed'])
            audits = selected['selected_program_audit']
            return (all(selected[variant]['passed'] is True
                        for variant in ('raw', 'regularized'))
                    and audits == acceptance.get('final_program_audit')
                    and all(_warehouse_selective_audit_valid(
                        audits[role], role, acceptance, explanation=role == 'robot_2')
                            for role in ('robot_1', 'robot_2')))
        if domain == 'pong':
            audits = acceptance['selected_actor_development_program_audit']
            return (all(acceptance['development'][variant]['passed'] is True
                        and acceptance['runtime_parity'][variant] is True
                        for variant in ('raw', 'regularized'))
                    and all(_program_audit_metrics_valid(
                        audits[role], minimum_fidelity=.95 if role == 'ai' else .75,
                        maximum_kl=.10 if role == 'ai' else .60,
                        minimum_samples=1000)
                        and audits[role].get('purpose') ==
                            ('explanation' if role == 'ai' else 'training_feedback')
                        for role in ('human', 'ai')))
        if domain == 'kitchen':
            audits = acceptance['final_role_program_development_audit']
            return (acceptance['pre_audit_quality_passed'] is True
                    and acceptance['verification_errors'] == []
                    and all(_program_audit_metrics_valid(
                        audits[role], minimum_fidelity=.80, maximum_kl=.35,
                        minimum_samples=1000, sample_key='samples')
                        and audits[role].get('role') == role
                        and audits[role].get('development_seeds') == list(range(41000, 41020))
                        and audits[role].get('actor_sha256') ==
                            acceptance['artifact_sha256']['regularized']
                        and audits[role].get('role_sha256') ==
                            _catalog_entry('kitchen', 'regularized')['role_sha256'][role]
                        and audits[role].get('program_sha256') ==
                            acceptance['program_sha256']['regularized'][role]
                        for role in ('human', 'ai')))
    except (KeyError, StopIteration, TypeError, ValueError):
        return False
    return False


def _program_audit_metrics_valid(audit: dict, *, minimum_fidelity: float,
                                 maximum_kl: float, minimum_samples: int,
                                 sample_key: str = 'sample_count') -> bool:
    if not isinstance(audit, dict) or audit.get('passed') is not True:
        return False
    count = audit.get(sample_key)
    fidelity = audit.get('action_fidelity')
    kl = audit.get('mean_kl_divergence')
    return (type(count) is int and count >= minimum_samples
            and type(fidelity) in (int, float) and math.isfinite(fidelity)
            and minimum_fidelity <= fidelity <= 1.0
            and type(kl) in (int, float) and math.isfinite(kl)
            and 0.0 <= kl <= maximum_kl)


def _warehouse_selective_metric_valid(metric: dict, *, agreement: float,
                                      sample_count: int) -> bool:
    """Check the measured coverage and fidelity, never a self-reported pass flag."""
    try:
        rows = metric['rows']
        coverage = metric['coverage']
        fidelity = metric['agreement']
        kl = metric['kl']
        illegal = metric['illegal_count']
        return (type(rows) is int and 0 < rows <= sample_count
                and type(illegal) is int and illegal == 0
                and all(type(value) in (int, float) for value in (coverage, fidelity, kl))
                and .20 <= coverage <= 1.0
                and abs(coverage - rows / sample_count) <= 1e-6
                and agreement <= fidelity <= 1.0
                and 0.0 <= kl <= .35)
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def _warehouse_selective_audit_valid(audit: dict, role: str, acceptance: dict,
                                     *, explanation: bool) -> bool:
    """The final frozen program must pass a new, fixed-leaf development audit."""
    try:
        sample_count = audit['sample_count']
        measured = audit['validation']
        global_metric = measured['global']
        expected_role_sha = acceptance['role_sha256']['regularized'][role]
        expected_program_sha = acceptance['program_sha256']['regularized'][role]
        sparse_schema = acceptance.get('program_schema_version')
        sparse_ok = (sparse_schema is None or
                     (sparse_schema == 'warehouse-sparse-oblique-v5.0'
                      and acceptance.get('max_features_per_node') == 3
                      and audit.get('max_features_per_node') == 3))
        base = (audit['audit_source'] == 'final_independent_development_rollouts'
                and sparse_ok
                and audit['development_seeds'] == list(range(240100, 240124))
                and audit['role'] == role
                and audit['actor_role_sha256'] == expected_role_sha
                and audit['program_sha256'] == expected_program_sha
                and type(sample_count) is int and sample_count >= 1000
                and type(audit['program_depth']) is int and 0 <= audit['program_depth'] <= 5
                and type(audit['program_leaf_count']) is int
                and 1 <= audit['program_leaf_count'] <= 16
                and type(global_metric['rows']) is int
                and global_metric['rows'] == sample_count
                and audit['feedback_eligible'] is True
                and _warehouse_selective_metric_valid(measured['feedback'],
                    agreement=.85, sample_count=sample_count))
        if not base:
            return False
        if explanation:
            return (audit['explanation_eligible'] is True
                    and _warehouse_selective_metric_valid(measured['explanation'],
                        agreement=.90, sample_count=sample_count))
        return audit['explanation_eligible'] is False
    except (KeyError, TypeError, ValueError):
        return False


def _warehouse_once_receipts_valid(acceptance: dict, entry: dict | None = None) -> bool:
    """Bind the one-use final audit and fresh test receipts to the selected pair."""
    try:
        selected = next(item for item in acceptance['candidate_training']
                        if item['pair_seed'] == acceptance['selected_pair_seed'])
        variants = selected['variants']
        selection = {
            'release': RELEASE_ID,
            'selected_pair_seed': selected['pair_seed'],
            'selected_role_sha256': acceptance['role_sha256'],
            'scenario_schedule_sha256': {
                variant: variants[variant]['scenario_schedule_sha256']
                for variant in ('raw', 'regularized')},
            'selected_program_sha256': acceptance['program_sha256']['regularized'],
            'heldout_scenario_seed_rule': 'enrollment_seed_times_10_plus_task',
            'heldout_seed_ranges': {str(task): [250000, 250099]
                                    for task in (1, 2, 3)},
        }
        if acceptance.get('actor_model_version') is not None:
            selection['actor_model_version'] = acceptance['actor_model_version']
            selection['feature_schema_sha256'] = acceptance['feature_schema_sha256']
        if (acceptance['heldout_scenario_seed_rule'] != selection['heldout_scenario_seed_rule']
                or acceptance['heldout_seed_ranges'] != selection['heldout_seed_ranges']
                or acceptance.get('all_candidate_regularized_feedback_passed') is not True):
            return False
        acceptance_path = (entry if entry is not None else
                           _catalog_entry('warehouse', 'regularized'))['acceptance_path']
        if not acceptance_path.endswith('_acceptance.json'):
            return False
        receipt_prefix = acceptance_path[:-len('_acceptance.json')]
        receipts = (
            (receipt_prefix + '_final_program_audit_receipt.json',
             {**selection, 'final_program_audit_seeds': list(range(240100, 240124))},
             'audit', 'audit_sha256', acceptance['final_program_audit']),
            (receipt_prefix + '_heldout_receipt.json',
             selection, 'heldout', 'heldout_sha256', acceptance['heldout']),
        )
        for path, expected_selection, evidence_key, digest_key, evidence in receipts:
            receipt = json.loads(_path(path).read_text())
            if (receipt.get('status') != 'completed'
                    or receipt.get('selection') != expected_selection
                    or receipt.get(evidence_key) != evidence
                    or receipt.get(digest_key) != sha256(
                        json.dumps(evidence, sort_keys=True).encode()).hexdigest()):
                return False
        return True
    except (KeyError, StopIteration, TypeError, ValueError, OSError,
            PolicyUnavailable):
        return False


def _accepted(entry: dict, binding: dict) -> bool:
    if (binding.get('domain') == 'warehouse'
            and binding.get('policy_id') in
            warehouse_release_spec(RELEASE_ID).policy_ids.values()):
        return _accepted_warehouse_v6(entry, binding)
    try:
        report_path = entry['training_report_path']
        acceptance_path = entry['acceptance_path']
        report_digest = _digest(report_path)
        acceptance = json.loads(_path(acceptance_path).read_text())
        steps = acceptance.get('joint_steps') or {}
        seeds = acceptance.get('pair_seeds') or []
        variant = binding['variant']
        artifact_hashes = acceptance.get('artifact_sha256') or {}
        report_hashes = acceptance.get('training_report_sha256') or {}
        accepted_program_hashes = (acceptance.get('program_sha256') or {}).get('regularized') or {}
        program_roles = ('robot_1', 'robot_2') if binding['domain'] == 'warehouse' else ('human', 'ai')
        program_paths = entry.get('program_paths') or []
        accepted_roles = (acceptance.get('role_sha256') or {}).get(variant)
        structured = (binding['domain'] == 'warehouse' and
                      entry.get('model_version') == _structured_model_version())
        structured_contract = True
        if structured:
            from backend.training.warehouse_structured_v5 import (
                STRUCTURED_FEATURE_SCHEMA_SHA256, STRUCTURED_MODEL_VERSION)
            structured_contract = (
                binding.get('actor_model_version') == STRUCTURED_MODEL_VERSION
                and binding.get('feature_schema_sha256') == STRUCTURED_FEATURE_SCHEMA_SHA256
                and entry.get('model_version') == STRUCTURED_MODEL_VERSION
                and entry.get('feature_schema_sha256') == STRUCTURED_FEATURE_SCHEMA_SHA256
                and acceptance.get('actor_model_version') == STRUCTURED_MODEL_VERSION
                and acceptance.get('feature_schema_sha256') == STRUCTURED_FEATURE_SCHEMA_SHA256
                and acceptance.get('pair_seeds') == [6101, 6102, 6103]
                and acceptance.get('program_schema_version') == 'warehouse-sparse-oblique-v5.0'
                and acceptance.get('max_features_per_node') == 3)
        elif binding['domain'] == 'warehouse':
            # A structured binding cannot be checked under the old MLP contract
            # after a catalog or configuration change.
            from env.warehouse.pure_nn import DUAL_MODEL_VERSION
            structured_contract = (entry.get('model_version') == DUAL_MODEL_VERSION and
                                   binding.get('actor_model_version') is None and
                                   binding.get('feature_schema_sha256') is None)
        roles_match = (binding['domain'] != 'warehouse' or accepted_roles == {
            'robot_1': binding['human_actor_sha256'],
            'robot_2': binding['ai_actor_sha256']})
        programs_match = (variant == 'raw' or
            len(program_paths) == 2 and set(accepted_program_hashes) == set(program_roles)
            and all(accepted_program_hashes[role] == binding['program_hashes'].get(path)
                    for role, path in zip(program_roles, program_paths)))
        raw_feedback = acceptance.get('raw_program_feedback_rows') or {}
        regularized_feedback = acceptance.get('regularized_program_feedback_rows') or {}
        feedback_valid = (isinstance(raw_feedback, dict) and isinstance(regularized_feedback, dict)
                          and len(raw_feedback) == 2 and len(regularized_feedback) == 2
                          and all(type(value) is int and value == 0 for value in raw_feedback.values())
                          and all(type(value) is int and value > 0 for value in regularized_feedback.values()))
        expected_release = RELEASE_ID if binding['domain'] == 'warehouse' else UPSTREAM_RELEASE_ID
        return (acceptance.get('release_id', acceptance.get('release')) == expected_release
                and acceptance.get('passed') is True
                and acceptance.get('heldout_evaluated') is True
                and len(seeds) == 3
                and len(set(seeds)) == 3
                and acceptance.get('selected_pair_seed') in seeds
                and _candidate_training_valid(binding['domain'], acceptance, seeds)
                and _selected_feedback_matches(binding['domain'], acceptance)
                and steps.get('raw') == 200000
                and steps.get('regularized') == 200000
                and artifact_hashes.get(variant) == binding['actor_sha256']
                and report_hashes.get(variant) == report_digest
                and roles_match and programs_match and feedback_valid
                and structured_contract
                and _development_and_audit_valid(binding['domain'], acceptance)
                and (binding['domain'] != 'warehouse'
                     or _warehouse_once_receipts_valid(acceptance, entry))
                and _heldout_quality_valid(binding['domain'], acceptance)
                and acceptance.get('post_policy_override_count') == 0)
    except (KeyError, OSError, ValueError, TypeError, PolicyUnavailable):
        return False


def _accepted_warehouse_v6(entry: dict, binding: dict) -> bool:
    """Apply the versioned Warehouse gate to the packaged evidence only."""
    try:
        if (entry.get('policy_id') != binding['policy_id']
                or entry.get('actor_sha256') != binding['actor_sha256']
                or entry.get('role_sha256') != {
                    'human': binding['human_actor_sha256'],
                    'ai': binding['ai_actor_sha256']}
                or entry.get('model_version') != binding['actor_model_version']
                or entry.get('feature_schema_sha256') !=
                binding['feature_schema_sha256']
                or _digest(entry['training_report_path']) !=
                entry['training_report_sha256']
                or _digest(entry['acceptance_path']) !=
                entry['acceptance_sha256']):
            return False
        source_closure = warehouse_runtime_source_closure_sha256()
        entries = warehouse_policy_artifacts()
        if entry != entries.get(binding['variant']):
            return False
        evidence = warehouse_evidence()
        revision = warehouse_evidence_revision()
        identity = sha256(json.dumps({'entry': entry, 'binding': binding},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        cache_key = (revision, source_closure, identity)
        with _WAREHOUSE_ACCEPTANCE_CACHE_LOCK:
            if cache_key in _WAREHOUSE_ACCEPTANCE_CACHE:
                return True
        acceptance = json.loads(_path(entry['acceptance_path']).read_text())
        if acceptance.get('source_closure_sha256') != source_closure:
            return False
        from experiments.warehouse.acceptance import validate_acceptance
        passed = validate_acceptance(
            acceptance, release_id=RELEASE_ID,
            entries=entries, evidence=evidence, repository_root=ROOT)
        if passed:
            with _WAREHOUSE_ACCEPTANCE_CACHE_LOCK:
                _WAREHOUSE_ACCEPTANCE_CACHE.add(cache_key)
        return passed
    except (ImportError, KeyError, OSError, ValueError, TypeError,
            PolicyUnavailable, ReleaseAssetUnavailable):
        return False


def all_policies_accepted() -> bool:
    manifest = policy_manifest()
    for variants in manifest.values():
        raw, regularized = variants['raw'], variants['regularized']
        if not raw['accepted'] or not regularized['accepted']:
            return False
        if raw['actor_sha256'] == regularized['actor_sha256']:
            return False
        if (raw['human_actor_sha256'] == regularized['human_actor_sha256']
                or raw['ai_actor_sha256'] == regularized['ai_actor_sha256']):
            return False
        for item in (raw, regularized):
            if item['human_actor_sha256'] == item['ai_actor_sha256']:
                return False
    return True
