"""Engineering-only draft tests: synthetic records; real reset/fit forbidden."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.training import warehouse_v620_bc_capability as cap
from backend.training import warehouse_v6_runner as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path), 'sha256': cap.file_sha(path)}


def native_cv_fixture(tmp_path, monkeypatch):
    from experiments.warehouse import v620_protocol as v
    from scripts.run_warehouse_development_cv import (
        ROLE_SHA256, development_gate, search_budget_audit, summarize_results,
    )
    actor = tmp_path / 'synthetic-actor.bin'
    actor.write_bytes(b'not a scientific model; engineering fixture only')
    monkeypatch.setattr(v, 'ACTOR_SHA256', cap.file_sha(actor))
    script = tmp_path / 'scripts/run_warehouse_development_cv.py'
    script.write_text('# source-pinned synthetic CV fixture\n')
    recipe_path = tmp_path / v.RECIPE_PATH
    write(recipe_path, {'schema': 'warehouse-v620-development-recipe.v2',
                        'cv': {'batch_id': v.CV_BATCH_ID,
                               'round_budget_contract': v.round_budget_contract()}})
    approved = cap.approved_cv_lock_record()
    approved['script_sha256'] = cap.file_sha(script)
    approved['recipe_sha256'] = cap.file_sha(recipe_path)
    write(tmp_path / cap.CV_LOCK_PATH, approved)
    data = tmp_path / 'synthetic-input.json'
    write(data, {'synthetic_data_only': True})
    directory = tmp_path / v.OUTPUT_ROOT / v.CV_BATCH_ID
    identity = {'source_closure_sha256': cap.CV_SOURCE_SHA256,
        'script_sha256': cap.file_sha(script), 'recipe_sha256': cap.file_sha(recipe_path),
        'actor_path': str(actor), 'actor_sha256': cap.file_sha(actor),
        'actor_role_sha256': ROLE_SHA256,
        'inputs': {'files_sha256': {str(data): cap.file_sha(data)}}}
    plan = {'schema': v.CV_PLAN_SCHEMA, 'candidate_id': v.CANDIDATE_ID,
        'batch_id': v.CV_BATCH_ID, 'round_budget_contract': v.round_budget_contract(),
        'output_directory': str(directory), 'identity': identity, 'jobs': v.jobs(),
        'folds': v.grouped_folds(), 'fold_assignment_sha256': v.digest(v.grouped_folds()),
        'gates': v.GATES, 'selection_arms': ['M1', 'M2'],
        'environment_resets': 0, 'actor_updates': 0, 'PPO_joint_steps': 0,
        'publishable': False, 'admission_evidence': False, 'fresh_validation_evidence': False}
    plan_ptr = write(directory / 'plan.json', plan)
    results, references = [], {}
    for job in v.jobs():
        job_dir = directory / 'jobs' / job['job_id']
        rounds = [30, 30, 30, 20, 0, 0, 0, 0] if job['arm'] == 'M0' else [4, 4, 4, 4, 4, 4, 3, 3]
        search = {'search_runs': [{'rounds': [{}] * count} for count in rounds],
                  'leaf_optimization_calls': 100}
        if job['arm'] != 'M0':
            search.update(round_budget_scope='global_per_role_fit_across_all_initializations',
                          global_search_rounds=sum(rounds), global_search_rounds_remaining=30-sum(rounds))
        program_report = {'fit_tree': search, 'leaf_optimization_calls_outside_search': 0}
        budget = search_budget_audit(program_report, job['arm'])
        assert budget['passed'] is True
        program = write(job_dir / 'program.json', {'report': program_report})
        write(job_dir / 'search-budget-audit.json', budget)
        metrics = {'feedback': {'coverage': .2, 'expected_actionable_coverage': .2,
                                'agreement': .99, 'kl': .1,
                                'whole_tree_raw_illegal_count': 0,
                                'whole_tree_path_uncertified_count': 0},
                   'explanation': {'coverage': .2, 'agreement': .99, 'kl': .1}}
        gate = development_gate(metrics)
        result = {'schema': 'warehouse-development-cv-job-result.v1', 'job': job,
            'plan_sha256': plan_ptr['sha256'], 'identity_sha256': v.digest(identity),
            'program_path': program['path'], 'program_sha256': program['sha256'],
            'actor_weights_unchanged': True, 'excluded_frames': 0,
            'fold_assignment_sha256': plan['folds'][job['fold']]['assignment_sha256'],
            'development_gate': gate, 'metrics': {'validation': metrics}, 'leaf_count': 2,
            'search_budget_audit': budget,
            'publishable': False, 'admission_evidence': False, 'fresh_validation_evidence': False,
            'environment_resets': 0, 'PPO_joint_steps': 0}
        ptr = write(job_dir / 'result.json', result)
        write(job_dir / 'completed.json', {'result_sha256': ptr['sha256'],
            'program_sha256': program['sha256'], 'plan_sha256': plan_ptr['sha256'],
            'passed': gate['passed']})
        results.append(result)
        references[job['job_id']] = ptr
    return write(directory / 'summary.json', {'schema': 'warehouse-development-cv-summary.v1',
        'candidate_id': v.CANDIDATE_ID, 'plan_sha256': plan_ptr['sha256'],
        'job_results': references, **summarize_results(results)})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_SOURCE_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'source_closure_sha256', lambda: 'a' * 64)
    monkeypatch.setattr(runner, '_LOCKED_EXECUTION_USED_IN_PROCESS', False)
    main = tmp_path / 'scripts/run_warehouse_v620_shared_bc_pilot.py'
    main.parent.mkdir(parents=True)
    main.write_text('# saved synthetic entry\n')
    monkeypatch.setattr(sys.modules['__main__'], '__file__', str(main))
    schedule, index = {}, 0
    for name, count in cap.STAGE_COUNTS:
        schedule[name] = [{'scenario_seed': 2000000000 + index + j,
                           'action_seed': 2100000000 + index + j} for j in range(count)]
        index += count
    cv_summary = native_cv_fixture(tmp_path, monkeypatch)
    clearance = write(tmp_path / 'seed-clearance.json', {
        'schema': 'warehouse-v620-bc-seed-clearance.v1', 'approved_for_shared_bc': True,
        'seed_binding_sha256': cap.digest(cap.seed_binding(schedule, 2200000000)),
        'conflict_count': 0})
    recipe = write(tmp_path / 'recipe.json', cap.recipe(.01))
    paired = write(tmp_path / 'configs/warehouse_v620_shared_bc_seed_plan.json', {
        'schema': 'warehouse-v620-paired-bc-seeds.v1', 'candidate_id': cap.CANDIDATE_ID,
        'pair_seed': 6160, 'gamma_candidates': [.01, .03],
        'seed_binding': cap.seed_binding(schedule, 2200000000)})
    raw = {'schema': cap.SCHEMA, 'candidate_id': cap.CANDIDATE_ID,
        'pair_seed': 6160, 'gamma': .01, 'run_id': 'pair-6160-gamma-001',
        'bc_seed_schedule': schedule, 'critic_seed': 2200000000,
        'source_closure_sha256': 'a' * 64, 'recipe': recipe,
        'cv_summary': cv_summary, 'seed_clearance': clearance, 'paired_seed_plan': paired,
        'output_directory': str(tmp_path / 'output/warehouse_v620_bc_pilot' /
                                cap.CANDIDATE_ID / 'pair-6160-gamma-001')}
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from backend.training import warehouse_v6_program
    from backend.training import warehouse_v618_selective_tree

    def forbidden(*args, **kwargs):
        raise AssertionError('engineering draft must not reset or scientifically fit')

    monkeypatch.setattr(WarehouseMultiAgentEnv, 'reset', forbidden)
    monkeypatch.setattr(warehouse_v6_program, 'fit_program', forbidden)
    monkeypatch.setattr(warehouse_v618_selective_tree, 'fit_independent_structure', forbidden)
    return raw, main


def authorize(raw, main, tmp_path):
    plan = cap.validate_spec(raw, runner=runner, main_path=main)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6160)
        initial = runner.role_hashes(runner.WarehouseV6Actor())
    authorization = cap.authorization_fields(plan, runner=runner, initial_roles=initial)
    path = tmp_path / 'authorization.json'
    write(path, authorization)
    return plan, path


def test_exact_1270_plus_two_reset_and_rng_contract(setup):
    raw, main = setup
    plan = cap.validate_spec(raw, runner=runner, main_path=main)
    resets = cap.expected_resets(plan)
    assert len(resets) == 1272
    assert sum(r['seed'] == plan.critic_seed for r in resets) == 2
    with pytest.raises(RuntimeError, match='opaque'):
        plan.scenarios('initial_public_teacher')
    with cap._activate_verified_plan(plan):
        assert plan.scenarios('topology_fit_sampled') == plan.scenarios('joint_augmentation_sampled')
        assert len(plan.action_seeds('deep_4_argmax')) == 50
        plan.assert_immutable(runner, raw)


@pytest.mark.parametrize('change', ['wrong_gamma', 'wrong_candidate', 'wrong_init',
    'scene_reuse', 'rng_reuse', 'critic_overlap', 'critic_shuffle_overlap', 'bc_shuffle_overlap',
    'stage_missing', 'wrong_source', 'path_escape', 'clearance_tamper', 'cv_tamper',
    'incomplete_cv', 'recipe_tamper', 'forged_summary_route', 'job_completion_tamper'])
def test_bad_plans_reject_before_any_training_or_reset(setup, change):
    raw, main = setup
    raw = deepcopy(raw)
    first, second = raw['bc_seed_schedule']['initial_public_teacher'][:2]
    if change == 'wrong_gamma': raw['gamma'] = .02
    elif change == 'wrong_candidate': raw['candidate_id'] += '-x'
    elif change == 'wrong_init': raw['pair_seed'] = 6150
    elif change == 'scene_reuse': second['scenario_seed'] = first['scenario_seed']
    elif change == 'rng_reuse': second['action_seed'] = first['action_seed']
    elif change == 'critic_overlap': raw['critic_seed'] = first['scenario_seed']
    elif change == 'critic_shuffle_overlap': first['scenario_seed'] = raw['critic_seed'] + 1
    elif change == 'bc_shuffle_overlap': first['action_seed'] = 6160 + 31
    elif change == 'stage_missing': raw['bc_seed_schedule'].pop('deep_4_argmax')
    elif change == 'wrong_source': raw['source_closure_sha256'] = 'b' * 64
    elif change == 'path_escape': raw['output_directory'] = '/tmp/unregistered-output'
    elif change == 'clearance_tamper': Path(raw['seed_clearance']['path']).write_text('{}')
    elif change == 'cv_tamper': Path(raw['cv_summary']['path']).write_text('{}')
    elif change == 'recipe_tamper': Path(raw['recipe']['path']).write_text('{}')
    elif change == 'incomplete_cv':
        path = Path(raw['cv_summary']['path'])
        decision = json.loads(path.read_bytes())
        decision['job_results'].pop(next(iter(decision['job_results'])))
        raw['cv_summary'] = write(path, decision)
    elif change == 'forged_summary_route':
        path = Path(raw['cv_summary']['path'])
        summary = json.loads(path.read_bytes())
        summary['chosen_method_arm'] = 'M1'
        raw['cv_summary'] = write(path, summary)
    elif change == 'job_completion_tamper':
        path = Path(raw['cv_summary']['path'])
        summary = json.loads(path.read_bytes())
        job_ptr = next(iter(summary['job_results'].values()))
        (Path(job_ptr['path']).parent / 'completed.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError)):
        cap.validate_spec(raw, runner=runner, main_path=main)


@pytest.mark.parametrize('mutation', ['cv01_location', 'plan_v1', 'old_source',
                                     'missing_contract', 'legacy_m1_contract'])
def test_only_corrected_cv02_v2_lock_can_enable_bc(setup, mutation):
    raw, main = setup
    summary_path = Path(raw['cv_summary']['path'])
    summary = json.loads(summary_path.read_bytes())
    plan_path = summary_path.parent / 'plan.json'
    plan = json.loads(plan_path.read_bytes())
    if mutation == 'cv01_location':
        old_summary = summary_path.parent.with_name('cv-01') / 'summary.json'
        raw['cv_summary'] = write(old_summary, summary)
    else:
        if mutation == 'plan_v1': plan['schema'] = 'warehouse-development-cv-plan.v1'
        elif mutation == 'old_source':
            plan['identity']['source_closure_sha256'] = '93581bbb94775934bd434f1175f706cd3581280ce089021dfadfde3ac2e03c1c'
        elif mutation == 'missing_contract': plan.pop('round_budget_contract')
        elif mutation == 'legacy_m1_contract':
            plan['round_budget_contract']['M1']['maximum_total_rounds'] = 240
        summary['plan_sha256'] = write(plan_path, plan)['sha256']
        raw['cv_summary'] = write(summary_path, summary)
    with pytest.raises(RuntimeError):
        cap.validate_spec(raw, runner=runner, main_path=main)


@pytest.mark.parametrize('mutation', ['global_31_rounds', 'wrong_global_counter',
    'wrong_remaining_counter', 'per_start_scope', 'leaf_api_over_budget',
    'missing_search_runs', 'legacy_m0_start_31', 'audit_artifact_lie'])
def test_budget_is_recomputed_from_actual_program_report_not_passed_flag(setup, mutation):
    raw, main = setup
    summary_path = Path(raw['cv_summary']['path'])
    summary = json.loads(summary_path.read_bytes())
    arm = 'M0' if mutation == 'legacy_m0_start_31' else 'M1'
    job_id = next(name for name in summary['job_results'] if name.endswith(arm))
    result_path = Path(summary['job_results'][job_id]['path'])
    result = json.loads(result_path.read_bytes())
    program_path = Path(result['program_path'])
    program = json.loads(program_path.read_bytes())
    search = program['report']['fit_tree']
    if mutation == 'global_31_rounds':
        search['search_runs'][-1]['rounds'].append({})
        search['global_search_rounds'] = 31
        search['global_search_rounds_remaining'] = -1
    elif mutation == 'wrong_global_counter': search['global_search_rounds'] = 29
    elif mutation == 'wrong_remaining_counter': search['global_search_rounds_remaining'] = 1
    elif mutation == 'per_start_scope': search['round_budget_scope'] = 'per_start'
    elif mutation == 'leaf_api_over_budget': program['report']['leaf_optimization_calls_outside_search'] = 20000
    elif mutation == 'missing_search_runs': search.pop('search_runs')
    elif mutation == 'legacy_m0_start_31': search['search_runs'][0]['rounds'].append({})
    elif mutation == 'audit_artifact_lie':
        write(result_path.parent / 'search-budget-audit.json', {'passed': True})
    result['program_sha256'] = write(program_path, program)['sha256']
    summary['job_results'][job_id] = write(result_path, result)
    completed_path = result_path.parent / 'completed.json'
    completed = json.loads(completed_path.read_bytes())
    completed['result_sha256'] = summary['job_results'][job_id]['sha256']
    completed['program_sha256'] = result['program_sha256']
    write(completed_path, completed)
    raw['cv_summary'] = write(summary_path, summary)
    # All pointer hashes and self-reported passed flags remain coherent. The
    # new verifier still rejects the actual serialized search histories.
    with pytest.raises(RuntimeError):
        cap.validate_spec(raw, runner=runner, main_path=main)


def test_bc_execution_requires_saved_main_and_exact_authorization(setup, tmp_path, monkeypatch):
    raw, main = setup
    plan, authorization = authorize(raw, main, tmp_path)
    monkeypatch.setattr(sys.modules['__main__'], '__file__', str(tmp_path / 'not-authorized.py'))
    with pytest.raises(FileNotFoundError):
        cap.execute(raw, authorization_receipt_path=authorization, runner=runner, main_path=main)
    monkeypatch.setattr(sys.modules['__main__'], '__file__', str(main))
    changed = json.loads(authorization.read_bytes())
    changed['PPO_joint_steps'] = 1
    write(authorization, changed)
    with pytest.raises(RuntimeError, match='authorization'):
        cap.execute(raw, authorization_receipt_path=authorization, runner=runner, main_path=main)
    assert not plan.output_directory.exists()


def synthetic_rows(seeds):
    from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
    from env.warehouse.layouts import STUDY_MAP_LAYOUT
    from env.warehouse.public_observation_v6 import encode_public_observation
    from env.warehouse.pure_nn import action_masks
    state = WarehouseState(episode_id=620, frame=1,
        agents=[AgentState('robot_1', (1, 0), 70.),
                AgentState('robot_2', STUDY_MAP_LAYOUT.robot_start_positions[1], 70.)],
        tasks=[DeliveryTask('task_0', (0, 3), (3, 2)),
               DeliveryTask('task_1', (1, 0), (3, 1))], next_task_index=2)
    obs = torch.from_numpy(np.repeat(encode_public_observation(state)[None], len(seeds), 0))
    masks = torch.from_numpy(np.repeat(action_masks(state)[None], len(seeds), 0))
    return (obs, masks, masks.argmax(-1).long(), torch.zeros(len(seeds)),
            tuple(f'nn-visit:{seed}' for seed in seeds), tuple(seeds))


def install_synthetic_engine(monkeypatch):
    from env.warehouse.reset_audit_v614 import record_successful_reset
    from env.warehouse.layouts import STUDY_MAP_LAYOUT
    from backend.training import warehouse_v6_collision_critic as critic_module
    calls = {'bc': [], 'risk': [], 'visits': [], 'teacher': [], 'critic': [],
             'shared_shuffle': [], 'risk_shuffle': []}

    def reset_record(seed):
        record_successful_reset(seed, 620, STUDY_MAP_LAYOUT.layout_id, 2., 10.)

    def teacher(episodes, seed, *, scenario_seeds=None):
        assert scenario_seeds is not None and len(scenario_seeds) == episodes
        calls['teacher'].append(list(scenario_seeds))
        for scene in scenario_seeds: reset_record(scene)
        rows = synthetic_rows(scenario_seeds)
        return (*rows[:3], {'row_episode_ids': list(rows[4]),
                            'row_scenario_seeds': list(rows[5])})

    def visits(model, seeds, *, action_seed_by_scene=None, **kwargs):
        assert set(action_seed_by_scene) == set(seeds)
        assert not set(action_seed_by_scene.values()) & set(seeds)
        assert kwargs['public_labels'] is True
        assert kwargs['include_execution_metadata'] is False
        calls['visits'].append((list(seeds), dict(action_seed_by_scene)))
        for scene in seeds: reset_record(scene)
        rows = synthetic_rows(seeds)
        return rows, {'episodes': len(seeds), 'physical_frames': len(seeds), 'excluded_frames': 0}

    def init_trees(model, *args, **kwargs):
        # No sklearn fit or model training: retain the randomly initialized
        # valid tree to test wiring and frozen parity only.
        assert all(p.requires_grad for actor in model.actors.values() for p in actor.gate.parameters())
        return {role: {'sklearn_version': 'synthetic-no-fit'} for role in runner.AGENTS}

    def epochs(model, *args, _bc_plan=None, **kwargs):
        cap.require_active_plan(_bc_plan)
        calls['bc'].append(kwargs['epochs'])
        calls['shared_shuffle'].append((kwargs['generator'].initial_seed(), id(kwargs['generator'])))
        for actor in model.actors.values():
            with torch.no_grad(): actor.gate[-1].bias.add_(.01)
        return .5

    def risk_epochs(model, critic, parts, *, _bc_plan=None, **kwargs):
        cap.require_active_plan(_bc_plan)
        calls['risk'].append(kwargs['epochs'])
        calls['risk_shuffle'].append(kwargs['generator_seed'])
        return {'synthetic_only': True, 'epochs': kwargs['epochs']}

    def critic(seed, **kwargs):
        calls['critic'].append(seed)
        reset_record(seed); reset_record(seed)
        return torch.nn.Linear(1, 1), {'validation_passed': True, 'synthetic_only': True}

    monkeypatch.setattr(runner, '_teacher_rows', teacher)
    monkeypatch.setattr(runner, '_collect_bc_visits', visits)
    monkeypatch.setattr(runner, '_initialize_public_trees', init_trees)
    monkeypatch.setattr(runner, '_bc_epochs', epochs)
    monkeypatch.setattr(runner, '_risk_bc_epochs', risk_epochs)
    monkeypatch.setattr(critic_module, 'train_collision_critic', critic)
    monkeypatch.setattr(runner, '_refresh_qualified_topology',
        lambda model, rows, **kwargs: {'roles': {r: {} for r in runner.AGENTS}, 'support_locked': False})
    monkeypatch.setattr(runner, 'evaluate', lambda *a, **k: pytest.fail('BC-only cannot evaluate'))
    return calls


def test_complete_bc_engine_skips_install_and_fresh_precheck(setup, tmp_path, monkeypatch):
    raw, main = setup
    plan, authorization = authorize(raw, main, tmp_path)
    calls = install_synthetic_engine(monkeypatch)
    result = cap.execute(raw, authorization_receipt_path=authorization, runner=runner, main_path=main)
    assert result['status'] == 'bc_completed_not_admitted'
    report = json.loads(Path(result['report_path']).read_bytes())
    receipt = json.loads(Path(result['receipt_path']).read_bytes())
    assert calls['bc'] == [20, 10, 10, 8]
    assert calls['risk'] == [8, 8, 6, 6, 6, 6, 6]
    assert calls['risk_shuffle'] == [6361, 6461, 6561, 6562, 6563, 6564, 6659]
    assert {seed for seed, _ in calls['shared_shuffle']} == {6191}
    assert len({identity for _, identity in calls['shared_shuffle']}) == 1
    assert calls['critic'] == [2200000000]
    assert sum(len(seeds) for seeds in calls['teacher']) + sum(len(seeds) for seeds, _ in calls['visits']) == 1270
    assert receipt['actual_reset_count'] == 1272
    assert receipt['reset_full_sequence_matches'] is True
    assert receipt['authorization_unchanged'] is True
    assert report['mid_bc_extracted_tree_installed'] is False
    assert report['fresh_validation_episodes'] == report['PPO_joint_steps'] == 0
    assert report['publishable'] is False
    assert report['declared_training_rng_schedule'] == cap.training_rng_schedule(2200000000)
    assert report['critic_environment_reset_seeds'] == [2200000000, 2200000000]
    assert not (plan.output_directory / 'final-bc-internal-precheck.json').exists()
    assert set(report['bc_data_seed_blocks']) == {name for name, _ in cap.STAGE_COUNTS}
    assert (plan.output_directory / 'final-bc.actor.npz').is_file()


def test_interruption_writes_prefix_receipt_and_cannot_retry(setup, tmp_path, monkeypatch):
    raw, main = setup
    plan, authorization = authorize(raw, main, tmp_path)
    install_synthetic_engine(monkeypatch)
    monkeypatch.setattr(runner, '_bc_epochs', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        cap.execute(raw, authorization_receipt_path=authorization, runner=runner, main_path=main)
    receipt = json.loads((plan.output_directory / 'bc-pilot.receipt.json').read_bytes())
    assert receipt['status'] == 'failed'
    assert receipt['actual_reset_count'] == 100
    assert receipt['reset_prefix_matches'] is True
    with pytest.raises(RuntimeError, match='unavailable'):
        cap.execute(raw, authorization_receipt_path=authorization, runner=runner, main_path=main)


def test_opaque_capability_not_recipe_flag_grants_engine_access(setup, tmp_path):
    raw, main = setup
    plan, _ = authorize(raw, main, tmp_path)
    with pytest.raises(RuntimeError, match='opaque'):
        cap.require_active_plan(plan)
    forged = cap.BCPilotPlan(plan.spec_sha256, plan.candidate_id, plan.pair_seed,
        plan.gamma, plan.run_id, plan.stage_cases, plan.critic_seed,
        plan.output_directory, plan.recipe_file_sha256,
        plan.expected_source_closure_sha256, plan.bound_files, object())
    with pytest.raises(RuntimeError, match='invalid'):
        with cap._activate_verified_plan(forged): pass


def test_new_scene_and_action_sequences_reach_actual_collectors(setup, monkeypatch):
    from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
    from env.warehouse.layouts import STUDY_MAP_LAYOUT
    scenes, action_rngs = [], []

    class FakeEnvironment:
        def reset(self, seed):
            scenes.append(seed)
            self.state = WarehouseState(episode_id=620, frame=1,
                agents=[AgentState('robot_1', (1, 0), 70.),
                        AgentState('robot_2', STUDY_MAP_LAYOUT.robot_start_positions[1], 70.)],
                tasks=[DeliveryTask('task_0', (0, 3), (3, 2)),
                       DeliveryTask('task_1', (1, 0), (3, 1))], next_task_index=2)

        def get_state(self): return deepcopy(self.state)
        def set_state(self, state): self.state = deepcopy(state)
        def step(self, actions):
            self.state.frame += 1
            self.state.truncated = True
            return None, None, None, None, {'proposed_actions': actions, 'action_override_agents': []}

    def act(state, *, base_seed, deterministic):
        action_rngs.append(base_seed)
        return ({role: 'WAIT' for role in runner.AGENTS},
                {role: SimpleNamespace(probabilities=np.array([0., 0., 0., 0., 1.]))
                 for role in runner.AGENTS})

    monkeypatch.setattr(runner, '_environment', lambda **kwargs: FakeEnvironment())
    monkeypatch.setattr(runner.FrozenWarehouseV6Actor, 'from_model', lambda model: SimpleNamespace(act=act))
    monkeypatch.setattr(runner, 'public_lookahead_teacher', lambda obs, masks: np.array([4, 4]))
    _, report = runner._collect_bc_visits(None, [10, 50], deterministic=False,
        deduplicate_collisions=False, action_seed_by_scene={10: 90, 50: 120})
    assert scenes == [10, 50]
    assert action_rngs == [90, 120]
    assert report['independent_action_seed_by_scene'] == {10: 90, 50: 120}
    scenes.clear()
    rows = runner._teacher_rows(2, 999, scenario_seeds=[60, 80])
    assert scenes == [60, 80]
    assert rows[-1]['row_scenario_seeds'] == [60, 80]


def test_both_epoch_implementations_use_alignment_and_default_keeps_old_loss(setup, tmp_path, monkeypatch):
    raw, main = setup
    plan, _ = authorize(raw, main, tmp_path)
    from backend.training import warehouse_bc_alignment as alignment
    from backend.training import warehouse_v6_collision_critic as collision
    real_loss = alignment.actor_shared_bc_alignment_loss
    used = []

    def traced_loss(*args, **kwargs):
        used.append((kwargs['recipe'].gamma, kwargs['phase']))
        return real_loss(*args, **kwargs)

    def forbidden_old(*args, **kwargs):
        raise AssertionError('new BC must not double count old distillation')

    monkeypatch.setattr(alignment, 'actor_shared_bc_alignment_loss', traced_loss)
    monkeypatch.setattr(runner, '_bc_tree_distillation', forbidden_old)
    monkeypatch.setattr(collision, 'risk_matrix', lambda critic, obs, mask: torch.zeros(len(obs), 5, 5))
    rows = synthetic_rows([12])
    model = runner.WarehouseV6Actor()
    before = runner.role_hashes(model)
    with cap._activate_verified_plan(plan):
        runner._bc_epochs(model, torch.optim.SGD(model.actors.parameters(), lr=.001),
            *rows[:3], epochs=1, generator=torch.Generator().manual_seed(1), _bc_plan=plan)
        report = runner._risk_bc_epochs(model, torch.nn.Linear(1, 1), [rows], epochs=1,
            generator_seed=1, margin_weight=0, near_increment=0,
            low_return_increment=0, _bc_plan=plan)
    assert used == [(.01, 'shared_bc'), (.01, 'shared_bc')]
    assert report['end']['core_intrinsic']['external_program_feedback'] == 0
    assert all(before[role] != runner.role_hashes(model)[role] for role in runner.AGENTS)
    old_calls = []
    monkeypatch.setattr(alignment, 'actor_shared_bc_alignment_loss',
                        lambda *a, **k: pytest.fail('legacy BC cannot call new alignment'))
    monkeypatch.setattr(runner, '_bc_tree_distillation',
                        lambda model, encoded, masks, logits: old_calls.append(True) or logits.new_zeros(()))
    runner._bc_epochs(model, torch.optim.SGD(model.actors.parameters(), lr=.001),
        *rows[:3], epochs=1, generator=torch.Generator().manual_seed(1))
    assert old_calls == [True]


@pytest.mark.parametrize("interrupt", [False, True])
def test_deleted_authorization_still_saves_actual_reset_receipt(setup, tmp_path, monkeypatch, interrupt):
    raw, main = setup
    plan, authorization = authorize(raw, main, tmp_path)
    install_synthetic_engine(monkeypatch)
    original = runner._bc_epochs
    def remove_authorization(*args, **kwargs):
        if authorization.exists():
            authorization.unlink()
        if interrupt:
            raise KeyboardInterrupt("synthetic interruption after authorization deletion")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, '_bc_epochs', remove_authorization)
    with pytest.raises(KeyboardInterrupt if interrupt else FileNotFoundError):
        cap.execute(raw, authorization_receipt_path=authorization, runner=runner, main_path=main)
    receipt = json.loads((plan.output_directory / 'bc-pilot.receipt.json').read_bytes())
    assert receipt['status'] == 'failed'
    assert receipt['authorization_unchanged'] is False
    assert 'FileNotFoundError' in receipt['authorization_hash_error']
    assert receipt['actual_reset_count'] == (100 if interrupt else 1272)
    assert receipt['reset_prefix_matches'] is True
    assert Path(receipt['journal_path']).is_file()
    assert receipt['journal_sha256'] == cap.file_sha(Path(receipt['journal_path']))
    assert 'KeyboardInterrupt' in receipt['error'] if interrupt else 'FileNotFoundError' in receipt['error']
