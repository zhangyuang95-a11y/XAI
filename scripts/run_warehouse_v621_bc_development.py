#!/usr/bin/env python3
"""Finite, gated common-BC development; no fresh precheck or PPO authority."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v621_bc_protocol as protocol


def pointer(path):
    target = Path(path).resolve(strict=True)
    return {'path': str(target), 'sha256': sha256(target.read_bytes()).hexdigest()}


def read_pointer(value):
    actual = pointer(value['path'])
    if actual != value:
        raise RuntimeError('development evidence changed')
    return json.loads(Path(actual['path']).read_bytes())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def prepare(*, source, clearance, summary, audit, output):
    from backend.training.warehouse_v621_bc_capability import (
        SCHEMA, model_implementation_sha256, _checked_prerequisites)
    protocol.checked_configuration(ROOT)
    if source != protocol.source_sha256(ROOT):
        raise RuntimeError('development source changed before registration')
    spec = {'schema': SCHEMA, 'candidate_id': protocol.diagnostics.CANDIDATE_ID,
        'batch_id': protocol.BATCH, 'source_closure_sha256': source,
        'source_files_sha256': protocol.source_files(ROOT),
        'model_implementation_sha256': model_implementation_sha256(ROOT),
        'recipe': pointer(ROOT/'configs/warehouse_v621_bc_recipe.json'),
        'seed_plan': pointer(ROOT/'configs/warehouse_v621_bc_seed_plan.json'),
        'diagnostics_summary': pointer(summary), 'diagnostics_independent_audit': pointer(audit),
        'seed_clearance': pointer(clearance),
        'output_directory': str(ROOT/'output/warehouse_v621'/protocol.BATCH),
        'PPO_joint_steps': 0, 'fresh_precheck_authorized': False, 'publishable': False,
        'backup_authorized_by_main_quality_failure_only': True}
    _checked_prerequisites(ROOT, spec)
    clear = read_pointer(spec['seed_clearance'])
    if (clear.get('approved_for_bc_development') is not True
            or clear.get('source_closure_sha256') != source
            or clear.get('recipe_sha256') != spec['recipe']['sha256']
            or clear.get('seed_plan_sha256') != spec['seed_plan']['sha256']):
        raise RuntimeError('exact source-bound seed clearance missing')
    write_new(output, spec)
    return spec


def execute(spec_path, initialization, recipe):
    import torch
    from backend.training.warehouse_v621_bc_capability import load_and_authorize
    from backend.training.warehouse_v621_bc_pipeline import build_shared_cache, development_model_job
    from env.warehouse.reset_audit_v614 import capture_stage1_resets, stage1_reset_phase
    torch.set_num_threads(2)
    begin = perf_counter()
    with load_and_authorize(ROOT, spec_path, recipe, initialization) as plan:
        journal = plan.output_directory/'native-successful-resets.jsonl'
        with capture_stage1_resets(journal) as actual:
            with stage1_reset_phase('bc'):
                result = build_shared_cache(plan) if recipe == 'shared_cache' else development_model_job(plan)
        if ([row['seed'] for row in actual] != [row['scenario_seed'] for row in plan.schedule]
                or len(actual) != plan.reset_count):
            raise RuntimeError('native successful resets differ from exact preauthorized schedule')
        result.update({'success_journal': pointer(journal), 'actual_reset_count': len(actual),
            'preauthorization_journal': pointer(plan.journal_path),
            'elapsed_seconds': perf_counter()-begin, 'specification': pointer(spec_path)})
        target = plan.output_directory/('cache.manifest.json' if recipe == 'shared_cache' else 'result.json')
        write_new(target, result)
    write_new(plan.output_directory/'completed.json', {'result': pointer(target),
        'capability_completed': pointer(plan.output_directory/'capability.completed.json'),
        'source_closure_sha256': plan.source_closure_sha256, 'actual_reset_count': len(actual),
        'PPO_joint_steps': 0, 'publishable': False})
    if recipe == 'shared_cache':
        for asset in (*result['teacher_assets'].values(), result['critic_asset']):
            Path(asset['path']).chmod(0o444)
    return result


def checked_result(spec, initialization, recipe):
    folder = Path(spec['output_directory'])/f'init-{initialization}'/recipe
    completed = json.loads((folder/'completed.json').read_bytes())
    result = read_pointer(completed['result'])
    capability = read_pointer(completed['capability_completed'])
    if ((folder/'capability.failed.json').exists() or result.get('status') != 'completed'
            or result.get('schema') != 'warehouse-v621-bc-development-result.v1'
            or result.get('initialization') != initialization or result.get('recipe') != recipe
            or result.get('source_closure_sha256') != spec['source_closure_sha256']
            or result.get('seed_plan_sha256') != spec['seed_plan']['sha256']
            or result.get('model_implementation_sha256') != spec['model_implementation_sha256']
            or result.get('actual_reset_count') != 1005 or completed.get('actual_reset_count') != 1005
            or capability.get('preauthorized_resets') != 1005
            or result.get('PPO_joint_steps') != 0 or result.get('publishable') is not False
            or result.get('fresh_precheck_episodes') != 0):
        raise RuntimeError('BC result is incomplete or differs from immutable parent')
    native = Path(result['success_journal']['path']).read_bytes()
    if sha256(native).hexdigest() != result['success_journal']['sha256']:
        raise RuntimeError('development native reset journal changed')
    from backend.training.warehouse_v621_bc_capability import reset_schedule
    schedule = reset_schedule(initialization, recipe)
    if [json.loads(line)['seed'] for line in native.splitlines()] != [row['scenario_seed'] for row in schedule]:
        raise RuntimeError('development native reset schedule changed')
    pre = Path(result['preauthorization_journal']['path']).read_bytes()
    if (sha256(pre).hexdigest() != result['preauthorization_journal']['sha256']
            or capability.get('preauthorization_journal_sha256') != result['preauthorization_journal']['sha256']
            or [{key: json.loads(line)[key] for key in ('phase', 'scenario_seed', 'action_seed')}
                for line in pre.splitlines()] != schedule):
        raise RuntimeError('development reset preauthorization chain changed')
    actor = result['actor']
    if pointer(actor['path'])['sha256'] != actor['sha256']:
        raise RuntimeError('development Actor bytes changed')
    if recipe == 'additive_control':
        from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor as Actor
        from backend.training.warehouse_v6_program import WarehouseV6Program as Program
    else:
        from backend.training.warehouse_v621_actor import FrozenWarehouseV621Actor as Actor
        from backend.training.warehouse_v621_program import WarehouseV621Program as Program
    frozen_actor = Actor.load(actor['path'], expected_sha256=actor['sha256'], expected_role_sha256=actor['role_sha256'])
    if recipe != 'additive_control' and any(role.core_residual_bound != (.35 if recipe == 'shared_b035' else .50)
            for role in frozen_actor.roles.values()):
        raise RuntimeError('development Actor residual bound belongs to a different recipe')
    if set(result['roles']) != {'robot_1', 'robot_2'}:
        raise RuntimeError('development needs evidence for both roles')
    for role, evidence in result['roles'].items():
        if pointer(evidence['program']['path']) != evidence['program']:
            raise RuntimeError('development program bytes changed')
        native = Program.load(evidence['program']['path'], expected_sha256=evidence['program']['sha256'],
            actor_role_sha256=actor['role_sha256'][role])
        if native.role != role or any(native.report['validation'][purpose] != evidence['metrics']['validation'][purpose]
                for purpose in ('feedback', 'explanation')):
            raise RuntimeError('development program role or validation report differs')
        if evidence['development_gate'] != protocol.diagnostics.gate_metrics(evidence['metrics']['validation']):
            raise RuntimeError('development program gate does not recompute')
        if recipe != 'additive_control':
            probe = evidence['gradient_probe']
            raw, reg = probe['variants']['raw'], probe['variants']['regularized']
            expected_probe = (raw['feedback_rows'] == 0 and raw['weighted_loss'] == 0.
                and raw['maximum_gradient_norm'] == 0. and reg['feedback_rows'] > 0
                and type(reg['maximum_gradient_norm']) in (int, float)
                and math.isfinite(reg['maximum_gradient_norm']) and reg['maximum_gradient_norm'] > 0.)
            if (probe['passed'] is not expected_probe or probe.get('role') != role
                    or probe.get('actor_parameters_unchanged') is not True or probe.get('optimizer_steps') != 0
                    or probe['current_fidelity']['current_actor_role_sha256'] != actor['role_sha256'][role]
                    or probe['current_fidelity']['program_sha256'] != native.sha256()):
                raise RuntimeError('development gradient probe does not recompute or match bound Actor')
    collection = read_pointer(result['development_collection'])
    actual = [row for row in collection['outcomes'] if row['case']['original_split'] == 'validation']
    if len(actual) != 75 or collection['cases'] != protocol.diagnostics.source_cases():
        raise RuntimeError('development outcome cohort changed')
    computed = {'mean_deliveries': sum(row['deliveries'] for row in actual)/75,
        'mean_collisions': sum(row['robot_collisions'] for row in actual)/75,
        'shutdown_episode_rate': sum(row['shutdown'] for row in actual)/75}
    quality = result['quality']
    if (set(quality) != set(computed) or any(type(quality[key]) not in (int, float)
            or not math.isfinite(quality[key]) or quality[key] != computed[key] for key in computed)):
        raise RuntimeError('BC quality measurements do not match recorded outcomes')
    expected_quality = quality['mean_deliveries'] >= 5.5 and quality['mean_collisions'] <= 6. and quality['shutdown_episode_rate'] <= .05
    if type(result['quality_passed']) is not bool or result['quality_passed'] != expected_quality:
        raise RuntimeError('BC quality gate does not recompute')
    expected_program = all(row['development_gate']['passed'] for row in result['roles'].values())
    expected_gradient = all(row['gradient_probe']['passed'] for row in result['roles'].values())
    if (result['program_passed'] is not expected_program or result['gradient_probe_passed'] is not expected_gradient
            or result['passed'] is not (expected_quality and expected_program and expected_gradient)):
        raise RuntimeError('BC combined candidate gate does not recompute')
    return result, pointer(folder/'result.json')


def summarize(spec_path, *, backup=False):
    spec = json.loads(Path(spec_path).read_bytes())
    if spec['source_closure_sha256'] != protocol.source_sha256(ROOT):
        raise RuntimeError('development source changed before decision')
    rows, refs = {}, {}
    required = ('additive_control', 'shared_b035', 'shared_b050') if backup else ('additive_control', 'shared_b035')
    for recipe in required:
        rows[recipe], refs[recipe] = [], {}
        for initialization in protocol.INITIALIZATIONS:
            result, ref = checked_result(spec, initialization, recipe)
            rows[recipe].append(result)
            refs[recipe][str(initialization)] = ref
    candidate = 'shared_b050' if backup else 'shared_b035'
    passed = all(r['passed'] is True for r in rows[candidate])
    quality_failed = any(r['quality_passed'] is False for r in rows['shared_b035'])
    if backup and not quality_failed:
        raise RuntimeError('backup cannot run or select when every main quality gate passed')
    if passed:
        next_stage = 'lock_common_bc_and_implement_fresh_precheck'
    elif not backup and quality_failed:
        next_stage = 'conditional_shared_b050_development'
    else:
        next_stage = 'candidate_closed_development_gate_failed'
    summary = {'schema': 'warehouse-v621-bc-development-summary.v1',
        'candidate_id': protocol.diagnostics.CANDIDATE_ID, 'specification': pointer(spec_path),
        'source_closure_sha256': spec['source_closure_sha256'], 'results': refs,
        'candidate_recipe': candidate, 'all_three_candidate_initializations_passed': passed,
        'main_quality_failure': quality_failed, 'next_stage': next_stage,
        'fresh_precheck_authorized': False, 'PPO_joint_steps': 0, 'publishable': False, 'pilot_opened': False}
    directory = Path(spec['output_directory'])
    write_new(directory/('backup.summary.json' if backup else 'main.summary.json'), summary)
    if next_stage == 'conditional_shared_b050_development':
        backup_spec = {**spec, 'main_results': refs['shared_b035']}
        write_new(directory/'backup.spec.json', backup_spec)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--job', choices=('shared_cache', *protocol.RECIPES))
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--initialization', type=int, choices=protocol.INITIALIZATIONS)
    parser.add_argument('--backup', action='store_true')
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--seed-clearance', type=Path)
    parser.add_argument('--diagnostics-summary', type=Path)
    parser.add_argument('--diagnostics-audit', type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if not all((args.expected_source_sha256, args.seed_clearance, args.diagnostics_summary, args.diagnostics_audit)):
            parser.error('prepare requires source, exact clearance and complete independently audited diagnostics')
        result = prepare(source=args.expected_source_sha256, clearance=args.seed_clearance,
            summary=args.diagnostics_summary, audit=args.diagnostics_audit, output=args.spec)
    elif args.job:
        if args.initialization is None or args.backup:
            parser.error('job requires initialization; backup jobs use the independently generated backup spec')
        result = execute(args.spec, args.initialization, args.job)
    else:
        result = summarize(args.spec, backup=args.backup)
    print(json.dumps({key: result[key] for key in ('status', 'quality', 'passed', 'next_stage',
        'actual_reset_count', 'PPO_joint_steps', 'publishable') if key in result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
