"""Finite BC entry gates and synthetic collectors; no native reset or training."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_warehouse_v621_bc_development as entry
from backend.training import warehouse_v621_bc_pipeline as pipeline
from backend.training import warehouse_v621_bc_capability as capability
from experiments.warehouse import v621_bc_protocol as protocol
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
    return entry.pointer(path)


@pytest.fixture
def summary_fixture(tmp_path, monkeypatch):
    spec = {'source_closure_sha256': 'a'*64, 'output_directory': str(tmp_path)}
    path = tmp_path/'spec.json'
    write(path, spec)
    monkeypatch.setattr(protocol, 'source_sha256', lambda _: 'a'*64)
    rows = {(seed, recipe): {'quality_passed': True, 'program_passed': True,
        'gradient_probe_passed': True, 'passed': True} for seed in protocol.INITIALIZATIONS
        for recipe in protocol.RECIPES}
    for seed in protocol.INITIALIZATIONS:
        rows[seed, 'additive_control']['passed'] = False
    def checked(_, seed, recipe):
        if (seed, recipe) not in rows:
            raise RuntimeError('synthetic incomplete job')
        return deepcopy(rows[seed, recipe]), {'path': f'/synthetic/{seed}/{recipe}', 'sha256': 'f'*64}
    monkeypatch.setattr(entry, 'checked_result', checked)
    return path, rows


def test_all_three_candidate_initializations_required_and_control_is_diagnostic(summary_fixture):
    path, _ = summary_fixture
    result = entry.summarize(path)
    assert result['all_three_candidate_initializations_passed']
    assert result['next_stage'] == 'lock_common_bc_and_implement_fresh_precheck'
    assert not result['fresh_precheck_authorized'] and not result['pilot_opened']
    assert result['PPO_joint_steps'] == 0 and not result['publishable']


@pytest.mark.parametrize('failure', ['program', 'gradient', 'quality'])
def test_any_one_of_three_main_failures_prevents_lock(summary_fixture, failure):
    path, rows = summary_fixture
    bad = rows[protocol.INITIALIZATIONS[-1], 'shared_b035']
    bad[failure+'_passed' if failure != 'gradient' else 'gradient_probe_passed'] = False
    bad['passed'] = False
    result = entry.summarize(path)
    assert not result['all_three_candidate_initializations_passed']
    if failure == 'quality':
        assert result['next_stage'] == 'conditional_shared_b050_development'
        assert (path.parent/'backup.spec.json').exists()
    else:
        assert result['next_stage'] == 'candidate_closed_development_gate_failed'
        assert not (path.parent/'backup.spec.json').exists()


def test_incomplete_main_cannot_select_or_activate_backup(summary_fixture):
    path, rows = summary_fixture
    del rows[protocol.INITIALIZATIONS[-1], 'shared_b035']
    with pytest.raises(RuntimeError, match='incomplete'):
        entry.summarize(path)
    assert not (path.parent/'main.summary.json').exists()
    assert not (path.parent/'backup.spec.json').exists()


@pytest.mark.parametrize('main_program_passed', [True, False])
def test_backup_summary_rejects_no_main_quality_failure(summary_fixture, main_program_passed):
    path, rows = summary_fixture
    for seed in protocol.INITIALIZATIONS:
        rows[seed, 'shared_b035'].update(program_passed=main_program_passed,
            passed=main_program_passed, quality_passed=True)
    with pytest.raises(RuntimeError):
        entry.summarize(path, backup=True)
    assert not (path.parent/'backup.summary.json').exists()


def test_backup_is_single_conditional_recipe_and_still_needs_all_three(summary_fixture):
    path, rows = summary_fixture
    rows[protocol.INITIALIZATIONS[0], 'shared_b035'].update(quality_passed=False, passed=False)
    rows[protocol.INITIALIZATIONS[-1], 'shared_b050'].update(program_passed=False, passed=False)
    result = entry.summarize(path, backup=True)
    assert not result['all_three_candidate_initializations_passed']
    assert result['next_stage'] == 'candidate_closed_development_gate_failed'
    assert not result['fresh_precheck_authorized']


def test_ordinary_pipeline_operations_rejected_before_model_or_environment_creation(tmp_path):
    plan = SimpleNamespace(recipe='shared_b035', initialization=protocol.INITIALIZATIONS[0],
                           output_directory=tmp_path)
    for function, args in ((pipeline.create_model, (plan,)),
                           (pipeline.environment, (plan, 'unknown', 1, 2)),
                           (pipeline.train_common_start, (plan,)),
                           (pipeline.gradient_probe, (plan, None, None, None, []))):
        with pytest.raises(RuntimeError, match='ordinary'):
            function(*args)
    assert not list(tmp_path.iterdir())


class SyntheticActor:
    """Known complete logits and the same declared per-role sampling contract."""
    def __init__(self, corrupt=None):
        self.corrupt = corrupt

    def logits(self, obs, masks):
        return np.asarray([[.2, -.5, .3, 1.2, .7], [.5, .3, -.2, 1., .1]], np.float32)

    def act(self, state, *, base_seed, deterministic=False):
        masks = state.mask
        z = np.where(masks > .5, self.logits(state.obs, masks).astype(np.float64), -np.inf)
        p = np.exp(z-z.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        actions, distributions = {}, {}
        for index, role in enumerate(AGENTS):
            seed = independent_agent_seed(base_seed=base_seed, episode_id=state.episode_id,
                                          frame=state.frame, agent_id=role)
            chosen = int(p[index].argmax()) if deterministic else int(np.random.default_rng(seed).choice(5, p=p[index]))
            actions[role] = ACTIONS[chosen]
            distributions[role] = SimpleNamespace(probabilities=tuple(p[index]))
        if self.corrupt == 'action':
            original = ACTIONS.index(actions[AGENTS[0]])
            actions[AGENTS[0]] = ACTIONS[4 if original != 4 else 3]
        if self.corrupt == 'distribution':
            distributions[AGENTS[0]].probabilities = tuple([.2]*5)
        return actions, distributions


class SyntheticEnvironment:
    """Finite state fixture, with no native environment construction or reset."""
    def __init__(self, case, *, rewrite=False):
        self.case, self.rewrite = case, rewrite
        self.executed = []
        self.state = SimpleNamespace(frame=0, episode_id=7, terminated=False, truncated=False,
            total_deliveries=6, robot_collision_events=2, shutdown_count=0,
            obs=np.zeros((2, 141), np.float32),
            mask=np.asarray([[0, 1, 1, 1, 1], [1, 1, 0, 1, 1]], np.float32))

    def step(self, actions):
        self.executed.append(dict(actions))
        self.state.frame += 1
        self.state.terminated = self.state.frame == 2
        return None, None, None, None, {'proposed_actions': dict(actions),
                                       'action_override_agents': [AGENTS[0]] if self.rewrite else []}


def install_synthetic_collector(monkeypatch, tmp_path, *, corrupt=None, rewrite=False):
    cases = protocol.diagnostics.source_cases()
    environments = []
    def create(plan, phase, scenario, action):
        assert phase == 'spent_development_evaluation'
        case = next(c for c in cases if c['scenario_seed'] == scenario)
        assert action == case['action_seed']
        result = SyntheticEnvironment(case, rewrite=rewrite)
        environments.append(result)
        return result
    monkeypatch.setattr(pipeline, 'environment', create)
    monkeypatch.setattr(pipeline, 'encode_public_observation', lambda state: state.obs.copy())
    monkeypatch.setattr(pipeline, 'action_masks', lambda state: state.mask.copy())
    monkeypatch.setattr(pipeline, 'public_pre_action_snapshot', lambda state: {
        'frame': state.frame, 'episode_id': state.episode_id,
        'terminated': state.terminated, 'deliveries': state.total_deliveries})
    plan = SimpleNamespace(output_directory=tmp_path)
    artifact = {'sha256': 'a'*64, 'role_sha256': {AGENTS[0]: 'b'*64, AGENTS[1]: 'c'*64}}
    return plan, SyntheticActor(corrupt), artifact, environments


def test_spent_collection_checks_complete_NN_RNG_and_keeps_every_trace_frame(monkeypatch, tmp_path):
    plan, actor, artifact, environments = install_synthetic_collector(monkeypatch, tmp_path)
    parts, report = pipeline.collect_spent_development(plan, actor, artifact)
    assert len(environments) == len(report['traces']) == 275
    assert report['NN_action_checks'] == 1100
    assert report['quality_episodes'] == 75
    assert report['quality'] == {'mean_deliveries': 6., 'mean_collisions': 2., 'shutdown_episode_rate': 0.}
    assert report['quality_passed']
    assert report['omitted_frames'] == report['post_policy_interventions'] == 0
    assert not report['fresh_validation_evidence']
    assert report['physical_frames_per_role'] == {'fit': 200, 'calibration': 50, 'eligibility': 150, 'validation': 150}
    for env in environments:
        reference = report['traces'][str(env.case['base'])]
        raw = open(reference['path'], 'rb').read()
        assert sha256(raw).hexdigest() == reference['sha256']
        traces = [json.loads(line) for line in raw.splitlines()]
        assert [row['frame'] for row in traces] == [0, 1]
        for frame, trace in enumerate(traces):
            assert trace['executed_actions'] == env.executed[frame]
            assert protocol.digest(trace['pre_state_public']) == trace['pre_state_public_sha256']
            assert protocol.digest(trace['post_state_public']) == trace['post_state_public_sha256']
            assert trace['pre_state_public']['frame'] == frame
            assert trace['post_state_public']['frame'] == frame+1
            if frame:
                assert traces[frame-1]['post_state_public_sha256'] == trace['pre_state_public_sha256']
            for role in AGENTS:
                z = trace['roles'][role]
                assert z['sampled_action'] == env.executed[frame][role]
                assert z['rng_seed'] == independent_agent_seed(base_seed=env.case['action_seed'],
                    episode_id=7, frame=frame, agent_id=role)
                assert z['actor_role_sha256'] == artifact['role_sha256'][role]
                assert z['mask'][ACTIONS.index(z['sampled_action'])] == 1


@pytest.mark.parametrize('corrupt,rewrite', [('action', False), ('distribution', False), (None, True)])
def test_spent_collection_rejects_NN_RNG_distribution_or_execution_override(monkeypatch, tmp_path, corrupt, rewrite):
    plan, actor, artifact, _ = install_synthetic_collector(monkeypatch, tmp_path, corrupt=corrupt, rewrite=rewrite)
    with pytest.raises(RuntimeError, match='differs|rewrote'):
        pipeline.collect_spent_development(plan, actor, artifact)
    assert not (tmp_path/'development.collection.json').exists()


def test_spent_collection_rejects_disconnected_public_state_chain(monkeypatch, tmp_path):
    plan, actor, artifact, _ = install_synthetic_collector(monkeypatch, tmp_path)
    calls = [0]
    def disconnected(state):
        calls[0] += 1
        return {'frame': state.frame, 'episode_id': state.episode_id,
                'unphysical_external_mutation': int(calls[0] >= 3)}
    monkeypatch.setattr(pipeline, 'public_pre_action_snapshot', disconnected)
    with pytest.raises(RuntimeError, match='chain disconnected'):
        pipeline.collect_spent_development(plan, actor, artifact)
    assert not (tmp_path/'development.collection.json').exists()


def test_gradient_probe_computes_positive_real_gradient_without_changing_actor(monkeypatch, tmp_path):
    # Load only test fixture builders; these create tensors/public arrays and
    # execute synthetic convex fits, with no native environment or optimizer.
    import importlib.util
    source = __import__('pathlib').Path(__file__).with_name('test_warehouse_v621_program.py')
    spec = importlib.util.spec_from_file_location('_v621_program_synthetic_fixtures', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    active, actor, rows, program = module.fit()
    monkeypatch.setattr(pipeline, 'require', lambda *_: None)
    before = {name: tensor.clone() for name, tensor in active.state_dict().items()}
    result = pipeline.gradient_probe(SimpleNamespace(), active, actor, program, rows[3])
    assert result['passed']
    assert result['variants']['raw']['maximum_gradient_norm'] == 0.
    assert result['variants']['raw']['weighted_loss'] == 0.
    assert result['variants']['regularized']['maximum_gradient_norm'] > 0.
    assert result['optimizer_steps'] == 0 and result['actor_parameters_unchanged']
    for name, tensor in active.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert all(parameter.grad is None for parameter in active.parameters())


@pytest.fixture
def completed_result_fixture(tmp_path, monkeypatch):
    """A fully synthetic receipt chain backed by genuine typed tensor assets."""
    import importlib.util
    from pathlib import Path
    from backend.training import warehouse_v621_actor as fa
    from backend.training import warehouse_v621_actor_training as ta
    from backend.training import warehouse_v621_program as pg
    from env.warehouse.layouts import STUDY_MAP_LAYOUT
    source = Path(__file__).with_name('test_warehouse_v621_program.py')
    fixture_spec = importlib.util.spec_from_file_location('_v621_receipt_tensor_fixtures', source)
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    active = fixture_module.model()
    snapshot = fa.FrozenWarehouseV621Actor.from_model(active)
    seed, recipe = protocol.INITIALIZATIONS[0], 'shared_b035'
    spec = {'source_closure_sha256': 'a'*64, 'seed_plan': {'sha256': 'b'*64},
            'model_implementation_sha256': 'c'*64, 'output_directory': str(tmp_path)}
    folder = tmp_path/f'init-{seed}'/recipe
    folder.mkdir(parents=True)
    artifact = ta.export_actor(active, folder/'final-bc.actor.npz', pair_seed=seed,
                               variant='raw', candidate_id=protocol.diagnostics.CANDIDATE_ID)
    templates = fixture_module.parts()[0]
    by_split = {name: [] for name in protocol.diagnostics.SPLITS}
    outcomes = []
    for index, case in enumerate(protocol.diagnostics.source_cases()):
        row = dict(templates[index % len(templates)], frame=0,
            episode=f'development-spent-base-{case["base"]}', base_seed=case['scenario_seed'],
            base=case['base'], task=case['task'])
        by_split[case['original_split']].append(row)
        outcomes.append({'case': case, 'deliveries': 6, 'robot_collisions': 2,
                         'shutdown': False, 'physical_frames': 1})
    parts = [by_split[name] for name in protocol.diagnostics.SPLITS]
    roles = {}
    monkeypatch.setattr(pipeline, 'require', lambda *_: None)
    for role in AGENTS:
        program = pg.fit_shared_program(role, snapshot, *parts)
        target = folder/(role+'.program.json')
        program.save(target)
        metrics = {name: pg.program_metrics(program, snapshot, rows) for name, rows in by_split.items()}
        roles[role] = {'program': entry.pointer(target), 'metrics': metrics,
            'development_gate': protocol.diagnostics.gate_metrics(metrics['validation']),
            'gradient_probe': pipeline.gradient_probe(SimpleNamespace(), active, snapshot,
                                                      program, by_split['validation'])}
    schedule = capability.reset_schedule(seed, recipe)
    native = folder/'native-successful-resets.jsonl'
    native.write_text(''.join(json.dumps({'phase': 'bc', 'seed': row['scenario_seed'],
        'episode_id': i+1, 'map_layout_id': STUDY_MAP_LAYOUT.layout_id,
        'move_battery_cost': 2., 'charge_per_wait': 10.})+'\n' for i, row in enumerate(schedule)))
    pre = folder/'reset-preauthorization.jsonl'
    pre.write_text(''.join(json.dumps({'schema': 'warehouse-v621-bc-reset-preauthorization.v1',
        'index': i, 'initialization': seed, 'recipe': recipe, **row,
        'source_closure_sha256': spec['source_closure_sha256'],
        'seed_plan_sha256': spec['seed_plan']['sha256'], 'consumed_before_actual_reset': True,
        'not_an_actual_reset_receipt': True})+'\n' for i, row in enumerate(schedule)))
    collection = {'schema': 'warehouse-v621-spent-development-trajectories.v1', 'actor': artifact,
        'cases': protocol.diagnostics.source_cases(), 'outcomes': outcomes,
        'quality': {'mean_deliveries': 6., 'mean_collisions': 2., 'shutdown_episode_rate': 0.},
        'quality_passed': True, 'quality_episodes': 75, 'post_policy_interventions': 0,
        'omitted_frames': 0, 'fresh_validation_evidence': False,
        'purpose': 'spent_scene_development_only',
        'physical_frames_per_role': {name: len(rows) for name, rows in by_split.items()}}
    collection_ref = write(folder/'development.collection.json', collection)
    training = {'both_roles_changed': True, 'external_program_feedback': 0,
        'external_program_actor_updates': 0, 'training_episode_exposure': 1270,
        'new_nn_visitation_episodes': 730, 'reused_cached_teacher_episodes': 540}
    training_ref = write(folder/'bc.training.json', training)
    spec_ref = write(tmp_path/'synthetic.spec.json', spec)
    result = {'schema': 'warehouse-v621-bc-development-result.v1', 'status': 'completed',
        'candidate_id': protocol.diagnostics.CANDIDATE_ID, 'batch_id': protocol.BATCH,
        'recipe': recipe, 'initialization': seed,
        'source_closure_sha256': spec['source_closure_sha256'],
        'seed_plan_sha256': spec['seed_plan']['sha256'],
        'model_implementation_sha256': spec['model_implementation_sha256'],
        'output_directory': str(folder), 'actor': artifact, 'roles': roles,
        'quality': dict(collection['quality']), 'quality_passed': True,
        'program_passed': all(r['development_gate']['passed'] for r in roles.values()),
        'gradient_probe_passed': all(r['gradient_probe']['passed'] for r in roles.values()),
        'training_report': training_ref, 'development_collection': collection_ref,
        'actual_reset_count': 1005, 'success_journal': entry.pointer(native),
        'preauthorization_journal': entry.pointer(pre), 'specification': spec_ref,
        'actor_updates_during_development_evaluation': 0, 'PPO_joint_steps': 0,
        'fresh_precheck_episodes': 0, 'fresh_validation_evidence': False,
        'publishable': False, 'pilot_opened': False}
    result['passed'] = result['quality_passed'] and result['program_passed'] and result['gradient_probe_passed']
    write(folder/'capability.started.json', {'schema': 'warehouse-v621-bc-development-capability.v1',
        'initialization': seed, 'recipe': recipe, 'source_closure_sha256': spec['source_closure_sha256'],
        'specification_sha256': spec_ref['sha256'], 'single_use_even_if_interrupted': True,
        'scheduled_reset_count': 1005, 'PPO_joint_steps': 0, 'publishable': False})
    cap = {'schema': 'warehouse-v621-bc-capability-completed.v1', 'initialization': seed,
        'recipe': recipe, 'preauthorized_resets': 1005,
        'preauthorization_journal_sha256': entry.pointer(pre)['sha256'],
        'requires_native_actual_reset_reconciliation': True, 'PPO_joint_steps': 0,
        'fresh_precheck_authorized': False, 'publishable': False}
    cap_ref = write(folder/'capability.completed.json', cap)
    def publish():
        result_ref = write(folder/'result.json', result)
        write(folder/'completed.json', {'result': result_ref, 'capability_completed': cap_ref,
            'source_closure_sha256': spec['source_closure_sha256'], 'actual_reset_count': 1005,
            'PPO_joint_steps': 0, 'publishable': False})
    publish()
    assert entry.checked_result(spec, seed, recipe)[0]['passed']
    return spec, seed, recipe, folder, result, publish


@pytest.mark.parametrize('change', ['missing_role', 'combined_gate', 'gradient_gate',
                                   'program_metrics', 'probe_hash', 'quality_measurement',
                                   'native_journal', 'failed', 'uncompleted'])
def test_checked_result_recomputes_and_binds_complete_native_evidence(completed_result_fixture, change):
    spec, seed, recipe, folder, result, publish = completed_result_fixture
    if change == 'missing_role':
        del result['roles'][AGENTS[1]]
    elif change == 'combined_gate':
        result['passed'] = False
    elif change == 'gradient_gate':
        result['gradient_probe_passed'] = False
    elif change == 'program_metrics':
        result['roles'][AGENTS[0]]['metrics']['validation']['feedback']['coverage'] = .99
    elif change == 'probe_hash':
        result['roles'][AGENTS[0]]['gradient_probe']['current_fidelity']['program_sha256'] = '0'*64
    elif change == 'quality_measurement':
        result['quality']['mean_deliveries'] = 7.
    elif change == 'native_journal':
        (folder/'native-successful-resets.jsonl').write_text('{}\n')
    elif change == 'failed':
        write(folder/'capability.failed.json', {'error': 'synthetic interruption'})
    else:
        (folder/'capability.completed.json').unlink()
    publish()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        entry.checked_result(spec, seed, recipe)
