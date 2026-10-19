"""Explicit-factory BC development. No import has environment or training effects.

BC labels are public observations only. Every model visit executes its frozen
dual NN, and all physical frames are retained. External programs never update
an Actor during this stage. PPO and fresh validation are separate capabilities.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (OBSERVATION_FEATURE_NAMES,
    encode_public_observation, public_pre_action_snapshot)
from experiments.warehouse import v621_bc_protocol as protocol
from .warehouse_v6_actor import (FEATURE_DIM, FEATURE_NAMES, FrozenWarehouseV6Actor,
    _role_arrays, role_sha256)
from .warehouse_v6_actor_training import (SparseHardTree, WarehouseV6Actor,
    export_actor as export_control, policy_features_torch)
from .warehouse_v621_compile import compile_actor_structure
from .warehouse_v6_tree_routing import leaf_indices, certified_leaf_actions


def require(plan, operation):
    from .warehouse_v621_bc_capability import require_active_plan
    require_active_plan(plan, operation=operation)
    plan.assert_immutable()


def array_sha(**arrays):
    result = sha256()
    for name, item in sorted(arrays.items()):
        value = np.ascontiguousarray(item)
        result.update(name.encode())
        result.update(value.dtype.str.encode())
        result.update(np.asarray(value.shape, dtype='<i8').tobytes())
        result.update(value.tobytes())
    return result.hexdigest()


def frozen(model):
    if getattr(model, 'model_version', '') == 'warehouse-dual-actor.v6.18-quality-selective':
        return FrozenWarehouseV6Actor.from_model(model)
    from .warehouse_v621_actor import FrozenWarehouseV621Actor
    return FrozenWarehouseV621Actor.from_model(model)


def create_model(plan):
    """Equal primary NN tensors; architectural tensors have different semantics."""
    require(plan, 'construct')
    recipe, initialization = plan.recipe, plan.initialization
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization)
        control = WarehouseV6Actor()
    if recipe == 'additive_control':
        return control, {role: array_sha(**{name: tensor.detach().numpy()
            for name, tensor in control.actors[role].action_residual.state_dict().items()}) for role in AGENTS}
    if recipe not in ('shared_b035', 'shared_b050'):
        raise ValueError('unknown registered BC recipe')
    from .warehouse_v621_actor_training import WarehouseV621Actor
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization)
        model = WarehouseV621Actor(core_residual_bound=.35 if recipe == 'shared_b035' else .50)
    shared = {}
    for role in AGENTS:
        model.actors[role].action_residual.load_state_dict(control.actors[role].action_residual.state_dict())
        shared[role] = array_sha(**{name: tensor.detach().numpy()
            for name, tensor in model.actors[role].action_residual.state_dict().items()})
    return model, shared


def environment(plan, phase, scene_seed, action_seed):
    require(plan, 'collect')
    plan.authorize_reset(phase, int(scene_seed), None if action_seed is None else int(action_seed))
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=int(scene_seed))
    marked = env.get_state()
    marked.participant_controlled_agent_id = AGENTS[0]
    env.set_state(marked)
    return env


def teacher_rows(plan, stage, cases):
    from .warehouse_v6_teacher_lookahead import public_lookahead_teacher
    observations, masks, labels, episodes, scenes = [], [], [], [], []
    outcomes = []
    for case in cases:
        env = environment(plan, stage, case['scenario_seed'], case['action_seed'])
        while not (env.state.terminated or env.state.truncated):
            obs, mask = encode_public_observation(env.state), action_masks(env.state)
            label = public_lookahead_teacher(obs, mask)
            actions = {role: ACTIONS[int(label[i])] for i, role in enumerate(AGENTS)}
            observations.append(obs); masks.append(mask); labels.append(label)
            episodes.append(f'public-teacher:{case["scenario_seed"]}'); scenes.append(case['scenario_seed'])
            _, _, _, _, info = env.step(actions)
            if info['proposed_actions'] != actions or info['action_override_agents']:
                raise RuntimeError('public BC teacher actions were rewritten')
        outcomes.append(outcome(env))
    rows = (torch.from_numpy(np.stack(observations)), torch.from_numpy(np.stack(masks)),
        torch.from_numpy(np.stack(labels)).long(), torch.zeros(len(labels)), tuple(episodes), tuple(scenes))
    return rows, {'episodes': len(cases), 'physical_frames': len(labels), 'excluded_frames': 0,
        'data_sha256': rows_sha(rows), 'outcomes': outcomes, 'labels_source': 'pre_action_public_only'}


def outcome(env):
    return {'deliveries': int(env.state.total_deliveries),
        'robot_collisions': int(env.state.robot_collision_events),
        'shutdown': bool(env.state.shutdown_count), 'physical_frames': int(env.state.frame)}


def rows_sha(rows):
    return array_sha(**{str(i): row.detach().numpy() if isinstance(row, torch.Tensor)
                       else np.asarray(row) for i, row in enumerate(rows)})


def save_rows(path, rows):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **{str(i): row.detach().numpy() if isinstance(row, torch.Tensor)
            else np.asarray(row) for i, row in enumerate(rows)})


def load_rows(path):
    with np.load(path, allow_pickle=False) as data:
        return (*(torch.from_numpy(data[str(i)].copy()) for i in range(4)),
            tuple(map(str, data['4'].tolist())), tuple(map(int, data['5'].tolist())))


def nn_visits(plan, model, stage, cases, *, deterministic=False):
    from .warehouse_v6_teacher_lookahead import public_lookahead_teacher
    actor = frozen(model)
    observations, masks, labels, hard, episodes, scenes, outcomes = [], [], [], [], [], [], []
    for case in cases:
        env = environment(plan, stage, case['scenario_seed'], case['action_seed'])
        while not (env.state.terminated or env.state.truncated):
            obs, mask = encode_public_observation(env.state), action_masks(env.state)
            label = public_lookahead_teacher(obs, mask)
            chosen, _ = actor.act(env.state, base_seed=case['action_seed'], deterministic=deterministic)
            before = env.state.robot_collision_events
            _, _, _, _, info = env.step(chosen)
            if info['proposed_actions'] != chosen or info['action_override_agents']:
                raise RuntimeError('a BC NN action was rewritten')
            collided = env.state.robot_collision_events > before
            observations.append(obs); masks.append(mask); labels.append(label)
            hard.append(5. if collided and deterministic else 3. if collided else 0.)
            episodes.append(f'nn-visit:{case["scenario_seed"]}'); scenes.append(case['scenario_seed'])
        outcomes.append(outcome(env))
    rows = (torch.from_numpy(np.stack(observations)), torch.from_numpy(np.stack(masks)),
        torch.from_numpy(np.stack(labels)).long(), torch.tensor(hard, dtype=torch.float32),
        tuple(episodes), tuple(scenes))
    return rows, {'episodes': len(cases), 'physical_frames': len(labels), 'excluded_frames': 0,
        'data_sha256': rows_sha(rows), 'outcomes': outcomes,
        'actions_source': 'frozen_dual_nn_plus_physical_mask', 'post_policy_interventions': 0}


def install_public_topologies(model, rows, *, shared):
    from sklearn.tree import DecisionTreeClassifier
    encoded = policy_features_torch(rows[0], rows[1]).numpy()
    reports = {}
    for role_index, role in enumerate(AGENTS):
        classifier = DecisionTreeClassifier(criterion='log_loss', max_depth=4 if shared else 5,
            max_leaf_nodes=8 if shared else 16, min_samples_leaf=80,
            random_state=19+role_index)
        classifier.fit(encoded[:, role_index], rows[2][:, role_index].numpy())
        if not shared:
            reports[role] = model.actors[role].initialize_tree_from_classifier(classifier)
            continue
        # Eight learned classifier leaves leave capacity for eight physical guards.
        holder = model.actors[role].tree
        SparseHardTree.initialize_from_classifier(holder, classifier)
        names = ('split.weight', 'split.bias', 'left_refs', 'right_refs', 'node_count', 'leaf_count', 'leaf_logits')
        arrays = {'tree.'+name: holder.state_dict()[name].detach().numpy().astype(np.float32) for name in names}
        source = SimpleNamespace(feature_names=FEATURE_NAMES, feature_dim=FEATURE_DIM,
            role_sha256=array_sha(**arrays), weights=arrays)
        structure, report = compile_actor_structure(source)
        if structure is None:
            raise RuntimeError('registered initial public tree unexpectedly exceeds guard capacity')
        x = encoded[:, role_index]
        route = leaf_indices(x, structure['split_weights'], structure['split_bias'],
            structure['left_refs'], structure['right_refs'], len(structure['split_weights']), 5)
        count = structure['leaf_count']
        core = np.zeros(count, bool)
        anchors = np.asarray([ACTIONS.index(row['frozen_preferred_action'])
                              for row in report['compiled_leaf_origins']], np.int64)
        logits = np.zeros((count, len(ACTIONS)), np.float32)
        diagnostics = []
        for leaf in range(count):
            take = route == leaf
            label = rows[2][:, role_index].numpy()[take]
            counts = np.bincount(label, minlength=5)
            logits[leaf] = np.log((counts+1.)/(counts.sum()+5.))
            _, support = np.unique(np.asarray(rows[4])[take], return_counts=True)
            agreement = float((label == anchors[leaf]).mean()) if len(label) else 0.
            origin = report['compiled_leaf_origins'][leaf]
            core[leaf] = (origin['guard_branch'] != 'illegal_false_wait' and len(label) >= 80
                and len(support) >= 8 and support.max(initial=0)/max(1, len(label)) <= .20
                and agreement >= .75)
            diagnostics.append({'leaf': leaf, 'training_rows': len(label), 'training_episodes': len(support),
                'teacher_agreement': agreement, 'core': bool(core[leaf]), 'action': ACTIONS[anchors[leaf]]})
        if not core.any():
            raise RuntimeError('registered initial public tree has no supported static core')
        model.actors[role].install_structure(structure, core, anchors, logits)
        reports[role] = {'purpose': 'public_teacher_initialization_before_joint_bc_not_external_program',
            'compiler': report, 'leaves': diagnostics,
            'training_core_coverage': float(core[route].mean()), 'teacher_data_sha256': rows_sha(rows)}
    return reports


def shared_bc_loss(model, encoded, masks, labels, weights):
    """All training rows drive full CE and consistency in both gradient directions."""
    full = model.outputs_encoded(encoded, masks)
    trees, membership = [], []
    for index, role in enumerate(AGENTS):
        detail = model.actors[role].details_encoded(encoded[:, index])
        trees.append(detail['tree_logits'])
        membership.append(detail['core_membership'])
    tree = torch.stack(trees, dim=1)
    legal_max = tree.masked_fill(masks <= .5, -torch.inf).max(-1, keepdim=True).values
    tree = (tree-legal_max).masked_fill(masks <= .5, -1.e9)
    w = weights.detach()/weights.detach().mean()
    ce = nn.functional.cross_entropy(full.reshape(-1, 5), labels.reshape(-1), reduction='none').reshape(-1, 2)
    tree_ce = nn.functional.cross_entropy(tree.reshape(-1, 5), labels.reshape(-1), reduction='none').reshape(-1, 2)
    lp, lq = torch.log_softmax(full, -1), torch.log_softmax(tree, -1)
    to_full = (lq.detach().exp()*(lq.detach()-lp)).sum(-1)
    to_tree = (lp.detach().exp()*(lp.detach()-lq)).sum(-1)
    core = torch.stack(membership, dim=1)
    occupancy = torch.relu(.50-core.mean(0)).square().mean()
    components = {'full_task': (ce*w).mean(), 'tree_task': (tree_ce*w).mean(),
        'tree_to_full_kl': (to_full*w).mean(), 'full_to_tree_kl': (to_tree*w).mean(),
        'core_occupancy_penalty': occupancy}
    loss = components['full_task']+.2*components['tree_task']+.2*components['tree_to_full_kl']+.2*components['full_to_tree_kl']+.05*occupancy
    return loss, full, components


def bc_epochs(plan, model, parts, *, epochs, generator, lr, batch_size,
              critic=None, near_increment=0.):
    require(plan, 'bc_update')
    from .warehouse_bc_alignment import BCAlignmentRecipe, actor_shared_bc_alignment_loss
    from .warehouse_v6_collision_critic import risk_matrix
    obs, masks, labels, hard = (torch.cat([rows[i] for rows in parts]) for i in range(4))
    encoded = policy_features_torch(obs, masks)
    weights = torch.ones_like(labels, dtype=torch.float32)
    risk = None
    if critic is not None:
        near = obs[:, 0, OBSERVATION_FEATURE_NAMES.index('robots_shortest_distance')]*42 <= 2
        margin = obs[:, :, OBSERVATION_FEATURE_NAMES.index('self_charger_return_margin')]*50
        low = (margin >= 0) & (margin <= 6)
        weights = 1.+near_increment*near.float()[:, None]+8.*low.float()+hard[:, None]
        risk = risk_matrix(critic, obs, masks)
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=lr)
    control = getattr(model, 'model_version', '') == 'warehouse-dual-actor.v6.18-quality-selective'
    epoch_reports = []
    role_gradient_maxima = dict.fromkeys(AGENTS, 0.)
    for epoch in range(epochs):
        require(plan, 'bc_update')
        totals, seen = {}, 0
        for indices in torch.randperm(len(obs), generator=generator).split(batch_size):
            if control:
                detail = actor_shared_bc_alignment_loss(model, encoded[indices], masks[indices], labels[indices],
                    recipe=BCAlignmentRecipe(.03), phase='shared_bc', sample_weights=weights[indices])
                loss = detail.total
                logits = model.outputs_encoded(encoded[indices], masks[indices])
                components = {'full_task': detail.full_task, 'tree_task': detail.internal_tree_task,
                    'full_to_tree_kl': detail.full_to_internal_tree_kl, 'residual_penalty': detail.residual_penalty}
            else:
                loss, logits, components = shared_bc_loss(model, encoded[indices], masks[indices], labels[indices], weights[indices])
            if risk is not None:
                p = torch.softmax(logits, -1)
                collision = (p[:, 0, :, None]*p[:, 1, None, :]*risk[indices]).sum((1, 2)).mean()
                loss = loss+2.*collision
                components['offline_collision_risk'] = collision
            optimizer.zero_grad()
            loss.backward()
            for role in AGENTS:
                squared = sum((parameter.grad.detach().double().square().sum()
                    for parameter in model.actors[role].parameters() if parameter.grad is not None),
                    torch.zeros((), dtype=torch.float64))
                role_gradient_maxima[role] = max(role_gradient_maxima[role], float(squared.sqrt()))
            nn.utils.clip_grad_norm_(model.actors.parameters(), .5)
            optimizer.step()
            seen += len(indices)
            for name, value in {'total': loss, **components}.items():
                totals[name] = totals.get(name, 0.)+float(value.detach())*len(indices)
        for actor in model.actors.values():
            actor.tree.refresh_sparse_support()
        # Exact frozen forward validates guards, core bounds and route parity.
        check = frozen(model)
        with torch.no_grad():
            reference = model.outputs_encoded(encoded[:256], masks[:256]).numpy()
        raw = check.logits(obs[:256].numpy(), masks[:256].numpy())
        legal = masks[:256].numpy() > .5
        if control:
            actual = np.where(legal, raw, -1.e9)
        else:
            actual = np.where(legal, raw-np.where(legal, raw, -np.inf).max(-1, keepdims=True), -1.e9)
        if not np.allclose(reference, actual, atol=3.e-5, rtol=3.e-5):
            raise RuntimeError('full Torch/frozen NN parity failed after BC epoch')
        epoch_reports.append({'epoch': epoch+1, 'all_epoch_rows': seen,
            'row_weighted_mean_losses': {name: total/seen for name, total in totals.items()}})
    return {'epochs': epochs, 'training_rows': len(obs), 'data_sha256': rows_sha((obs, masks, labels, hard)),
        'all_physical_training_frames_retained': True, 'epoch_reports': epoch_reports,
        'role_maximum_bc_gradient_norm': role_gradient_maxima,
        'external_program_feedback': 0, 'external_program_gradient': 0}


def file_pointer(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def build_shared_cache(plan):
    require(plan, 'critic_update')
    from .warehouse_v6_collision_critic import train_collision_critic, collision_critic_sha256
    seeds = protocol.seed_plan()['initializations'][str(plan.initialization)]
    assets, stages = {}, {}
    for stage in protocol.TEACHER_STAGES:
        rows, report = teacher_rows(plan, stage, seeds['stages'][stage])
        path = plan.output_directory/(stage+'.npz')
        save_rows(path, rows)
        assets[stage] = {**file_pointer(path), 'data_sha256': rows_sha(rows)}
        stages[stage] = report
    # The unchanged critic creates exactly two same-seed environments. Consume
    # both tickets first; an interruption spends both and cannot be retried.
    for seed in seeds['critic_environment_reset_seeds']:
        plan.authorize_reset('critic_setup', seed, None)
    critic, report = train_collision_critic(seeds['critic_seed'], **protocol.recipe()['critic'])
    if report['validation_passed'] is not True:
        raise RuntimeError('shared offline event critic failed validation')
    path = plan.output_directory/'collision-critic.pt'
    with path.open('xb') as handle:
        torch.save(critic.state_dict(), handle)
    report_path = plan.output_directory/'collision-critic.report.json'
    write_new(report_path, report)
    return {'schema': 'warehouse-v621-bc-shared-cache.v1', 'status': 'completed',
        'initialization': plan.initialization, 'init': plan.initialization,
        'source_closure_sha256': plan.source_closure_sha256,
        'seed_plan_sha256': plan.seed_plan_sha256,
        'model_implementation_sha256': plan.model_implementation_sha256,
        'actual_reset_count': plan.reset_count, 'teacher_assets': assets,
        'teacher_reports': stages, 'critic_asset': {**file_pointer(path),
            'model_sha256': collision_critic_sha256(critic)}, 'critic_report': file_pointer(report_path),
        'actor_updates': 0, 'PPO_joint_steps': 0, 'publishable': False}


def load_shared_cache(plan):
    from .warehouse_v6_collision_critic import CollisionCritic, collision_critic_sha256
    folder = plan.output_directory.parent/'shared_cache'
    manifest = json.loads((folder/'cache.manifest.json').read_bytes())
    if (manifest['status'] != 'completed' or manifest['initialization'] != plan.initialization
            or manifest['source_closure_sha256'] != plan.source_closure_sha256
            or manifest['seed_plan_sha256'] != plan.seed_plan_sha256
            or manifest['model_implementation_sha256'] != plan.model_implementation_sha256
            or manifest['actual_reset_count'] != 542):
        raise RuntimeError('shared teacher cache identity differs from BC plan')
    for asset in (*manifest['teacher_assets'].values(), manifest['critic_asset'], manifest['critic_report']):
        if file_pointer(asset['path'])['sha256'] != asset['sha256']:
            raise RuntimeError('shared teacher or critic cache bytes changed')
    teacher = {stage: load_rows(asset['path']) for stage, asset in manifest['teacher_assets'].items()}
    if any(rows_sha(rows) != manifest['teacher_assets'][stage]['data_sha256'] for stage, rows in teacher.items()):
        raise RuntimeError('shared public teacher cache arrays changed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol.seed_plan()['initializations'][str(plan.initialization)]['rng']['critic_cache_load_torch'])
        critic = CollisionCritic()
    critic.load_state_dict(torch.load(manifest['critic_asset']['path'], map_location='cpu', weights_only=True))
    critic.eval()
    if collision_critic_sha256(critic) != manifest['critic_asset']['model_sha256']:
        raise RuntimeError('shared event critic model changed')
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return teacher, critic, file_pointer(folder/'cache.manifest.json')


def train_common_start(plan):
    require(plan, 'bc_update')
    teacher, critic, cache = load_shared_cache(plan)
    model, common = create_model(plan)
    initial = {role: array_sha(**{name: value.detach().numpy() for name, value
        in model.actors[role].state_dict().items()}) for role in AGENTS}
    topology = install_public_topologies(model, teacher['initial_public_teacher'],
        shared=plan.recipe != 'additive_control')
    start_parameters = {role: array_sha(**{name: value.detach().numpy() for name, value
        in model.actors[role].named_parameters() if value.requires_grad}) for role in AGENTS}
    assigned = protocol.seed_plan()['initializations'][str(plan.initialization)]
    rng = assigned['rng']
    generator = torch.Generator().manual_seed(rng['initial_bc_and_dagger_torch_shuffle'])
    stages = []

    def fit(name, parts, epochs, *, later=False, generator_seed=None, near=0., lr=None):
        current_generator = generator if generator_seed is None else torch.Generator().manual_seed(generator_seed)
        report = bc_epochs(plan, model, parts, epochs=epochs, generator=current_generator,
            lr=lr if lr is not None else (.0001 if later else .0003),
            batch_size=128 if later else 256, critic=critic if later else None, near_increment=near)
        stages.append({'stage': name, 'fit': report})
        print(json.dumps({'stage': name, 'recipe': plan.recipe, 'initialization': plan.initialization,
            'final_epoch_mean_losses': report['epoch_reports'][-1]['row_weighted_mean_losses']}), flush=True)

    def visit(name):
        data, report = nn_visits(plan, model, name, assigned['stages'][name], deterministic='argmax' in name)
        stages.append({'stage': name, 'visitation': report})
        return data

    early = [teacher['initial_public_teacher']]
    fit('initial_public_bc', early, 20)
    for number in (1, 2):
        early.append(visit(f'initial_nn_dagger_{number}'))
        fit(f'initial_nn_dagger_{number}_fit', early, 10)
    augmentation = [visit('joint_augmentation_sampled'), visit('joint_augmentation_argmax')]
    fit('joint_augmentation_public_bc', augmentation, 8, lr=.0001)
    risk_visit = visit('risk_nn_visitation')
    fit('risk_public_bc', [teacher['risk_public_teacher'], risk_visit, *augmentation], 8,
        later=True, generator_seed=rng['risk_torch_shuffle'])
    large_visit = visit('large_nn_visitation')
    fit('large_public_bc', [teacher['large_public_teacher'], large_visit, *augmentation], 8,
        later=True, generator_seed=rng['large_torch_shuffle'], near=2.)
    deep = [teacher['deep_public_teacher'], *augmentation]
    for number in range(1, 5):
        deep.extend((visit(f'deep_{number}_sampled'), visit(f'deep_{number}_argmax')))
        fit(f'deep_{number}_fit', deep, 6, later=True,
            generator_seed=rng['deep_torch_shuffle'][number-1], near=2.)
    fit('final_standard_calibration', deep, 6, later=True, generator_seed=rng['final_torch_shuffle'], near=2.)
    after = {role: array_sha(**{name: value.detach().numpy() for name, value
        in model.actors[role].state_dict().items()}) for role in AGENTS}
    trained_parameters = {role: array_sha(**{name: value.detach().numpy() for name, value
        in model.actors[role].named_parameters() if value.requires_grad}) for role in AGENTS}
    gradients = {role: max(row['fit']['role_maximum_bc_gradient_norm'][role]
        for row in stages if 'fit' in row) for role in AGENTS}
    if any(start_parameters[role] == trained_parameters[role] or gradients[role] <= 0. for role in AGENTS):
        raise RuntimeError('one role received no BC parameter updates')
    require(plan, 'export')
    path = plan.output_directory/'final-bc.actor.npz'
    if plan.recipe == 'additive_control':
        artifact = export_control(model, path, pair_seed=plan.initialization, variant='raw',
            candidate_id=protocol.diagnostics.CANDIDATE_ID+'-development-control')
    else:
        from .warehouse_v621_actor_training import export_actor
        artifact = export_actor(model, path, pair_seed=plan.initialization, variant='raw',
            candidate_id=protocol.diagnostics.CANDIDATE_ID+'-development-common-bc')
    path.chmod(0o444)
    checkpoint = plan.output_directory/'final-bc.checkpoint.pt'
    with checkpoint.open('xb') as handle:
        torch.save({'schema': 'warehouse-v621-common-bc-checkpoint.v1',
            'model_version': model.model_version, 'recipe': plan.recipe,
            'initialization': plan.initialization, 'source_closure_sha256': plan.source_closure_sha256,
            'seed_plan_sha256': plan.seed_plan_sha256, 'model_state': model.state_dict(),
            'actor_artifact': artifact, 'raw_and_regularized_must_clone_exactly': True,
            'PPO_joint_steps': 0}, handle)
    report = {'shared_teacher_cache': cache, 'shared_initial_primary_nn_sha256': common,
        'initial_state_sha256': initial, 'trained_state_sha256': after,
        'post_install_bc_start_parameter_sha256': start_parameters,
        'trained_parameter_sha256': trained_parameters, 'role_maximum_bc_gradient_norm': gradients,
        'both_roles_changed': True, 'public_tree_initialization': topology, 'stages': stages,
        'training_episode_exposure': 1270, 'new_nn_visitation_episodes': 730,
        'reused_cached_teacher_episodes': 540, 'external_program_feedback': 0,
        'external_program_actor_updates': 0, 'checkpoint': file_pointer(checkpoint)}
    write_new(plan.output_directory/'bc.training.json', report)
    return model, artifact, report


def collect_spent_development(plan, actor, artifact):
    """New trajectories on already spent scene/RNG arguments: development only."""
    from env.warehouse.decision_protocol import independent_agent_seed
    assigned = protocol.seed_plan()['development_evaluation']['cases']
    parts = {split: {role: [] for role in AGENTS} for split in protocol.diagnostics.SPLITS}
    outcomes, trace_refs, action_checks = [], {}, 0
    trace_folder = plan.output_directory/'development-traces'
    trace_folder.mkdir(exist_ok=False)
    for index, case in enumerate(assigned):
        env = environment(plan, 'spent_development_evaluation', case['scenario_seed'], case['action_seed'])
        deterministic = case['sampling_mode'] == 'argmax'
        trace = trace_folder/f'episode-{index:03d}.jsonl'
        previous_public_sha = None
        with trace.open('x') as handle:
            while not (env.state.terminated or env.state.truncated):
                obs, mask = encode_public_observation(env.state), action_masks(env.state)
                public_before = public_pre_action_snapshot(env.state)
                before_sha = protocol.digest(public_before)
                if previous_public_sha is not None and previous_public_sha != before_sha:
                    raise RuntimeError('development public physical state chain disconnected')
                frame, episode_id = int(env.state.frame), int(env.state.episode_id)
                actions, distributions = actor.act(env.state, base_seed=case['action_seed'], deterministic=deterministic)
                complete_logits = actor.logits(obs, mask)
                if not np.isfinite(complete_logits).all():
                    raise RuntimeError('nonfinite full NN logits in development execution')
                if getattr(actor, 'model_version', '') == 'warehouse-dual-actor.v6.18-quality-selective':
                    selected = np.where(mask > .5, complete_logits, -1.e9)
                    p = np.exp(selected-selected.max(-1, keepdims=True))
                    p /= p.sum(-1, keepdims=True)
                else:
                    from .warehouse_v621_actor import masked_probabilities
                    p = masked_probabilities(complete_logits, mask)
                recorded_roles = {}
                for role_index, role in enumerate(AGENTS):
                    rng_seed = independent_agent_seed(base_seed=case['action_seed'], episode_id=episode_id,
                        frame=frame, agent_id=role)
                    expected = int(p[role_index].argmax()) if deterministic else int(
                        np.random.default_rng(rng_seed).choice(len(ACTIONS), p=p[role_index]))
                    if (actions[role] != ACTIONS[expected] or mask[role_index, expected] != 1
                            or not np.allclose(distributions[role].probabilities, p[role_index], atol=2.e-6)):
                        raise RuntimeError('development action differs from bound frozen NN and physical mask')
                    row = {'obs': obs[role_index], 'mask': mask[role_index],
                        'episode': f'development-spent-base-{case["base"]}',
                        'base_seed': case['scenario_seed'], 'base': case['base'], 'task': case['task'],
                        'frame': frame, 'episode_id': episode_id, 'sampled_action': actions[role],
                        'nn_probabilities': p[role_index], 'rng_seed': rng_seed}
                    parts[case['original_split']][role].append(row)
                    recorded_roles[role] = {**row, 'obs': row['obs'].tolist(), 'mask': row['mask'].tolist(),
                        'nn_probabilities': row['nn_probabilities'].tolist(),
                        'actor_role_sha256': artifact['role_sha256'][role]}
                    action_checks += 1
                _, _, _, _, info = env.step(actions)
                if info['proposed_actions'] != actions or info['action_override_agents']:
                    raise RuntimeError('development environment rewrote a bound NN action')
                public_after = public_pre_action_snapshot(env.state)
                after_sha = protocol.digest(public_after)
                previous_public_sha = after_sha
                handle.write(json.dumps({'case': case, 'frame': frame, 'episode_id': episode_id,
                    'actor_sha256': artifact['sha256'], 'roles': recorded_roles,
                    'pre_state_public': public_before, 'pre_state_public_sha256': before_sha,
                    'post_state_public': public_after, 'post_state_public_sha256': after_sha,
                    'executed_actions': actions, 'post_policy_interventions': 0},
                    sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
        trace_refs[str(case['base'])] = file_pointer(trace)
        outcomes.append({'case': case, **outcome(env)})
    validation = [row for row in outcomes if row['case']['original_split'] == 'validation']
    if len(validation) != 75:
        raise RuntimeError('development quality does not contain all original 75 natural episodes')
    quality = {'mean_deliveries': float(np.mean([r['deliveries'] for r in validation])),
        'mean_collisions': float(np.mean([r['robot_collisions'] for r in validation])),
        'shutdown_episode_rate': float(np.mean([r['shutdown'] for r in validation]))}
    quality_passed = (quality['mean_deliveries'] >= 5.5 and quality['mean_collisions'] <= 6.
        and quality['shutdown_episode_rate'] <= .05)
    evidence = {'schema': 'warehouse-v621-spent-development-trajectories.v1', 'actor': artifact,
        'cases': assigned, 'outcomes': outcomes, 'traces': trace_refs, 'quality': quality,
        'quality_passed': quality_passed, 'quality_episodes': 75, 'NN_action_checks': action_checks,
        'post_policy_interventions': 0, 'omitted_frames': 0,
        'fresh_validation_evidence': False, 'purpose': 'spent_scene_development_only',
        'physical_frames_per_role': {split: len(rows[AGENTS[0]]) for split, rows in parts.items()}}
    write_new(plan.output_directory/'development.collection.json', evidence)
    return parts, evidence


def gradient_probe(plan, model, actor, program, validation):
    require(plan, 'gradient_probe')
    from .warehouse_v621_program import current_actor_fidelity, differentiable_feedback_loss
    original = model
    original.zero_grad(set_to_none=True)
    original_before = array_sha(**{name: tensor.detach().numpy() for name, tensor in original.state_dict().items()})
    model = deepcopy(original)
    model.freeze_structure(150000)
    if frozen(model).role_sha256 != actor.role_sha256:
        raise RuntimeError('post-freeze gradient probe clone changed the bound NN policy')
    role = program.role
    fidelity = current_actor_fidelity(program, actor, validation, purpose='feedback')
    before = array_sha(**{name: tensor.detach().numpy() for name, tensor in model.state_dict().items()})
    model.zero_grad(set_to_none=True)
    counts = {variant: {'feedback_rows': 0, 'maximum_gradient_norm': 0., 'nonzero_gradient_batches': 0,
        'weighted_loss': 0.} for variant in ('raw', 'regularized')}
    for start in range(0, len(validation), 512):
        rows = validation[start:start+512]
        obs = torch.from_numpy(np.stack([r['obs'] for r in rows]))
        mask = torch.from_numpy(np.stack([r['mask'] for r in rows]))
        encoded = policy_features_torch(obs, mask)
        actions = torch.tensor([ACTIONS.index(r['sampled_action']) for r in rows])
        for variant in counts:
            logits = model.role_logits(role, obs, mask)
            loss, report = differentiable_feedback_loss(program, logits, encoded, mask, actions,
                current_fidelity=fidelity, variant=variant)
            loss = .01*loss
            gradients = torch.autograd.grad(loss, tuple(p for p in model.actors[role].parameters()
                if p.requires_grad), allow_unused=True)
            norm = float(sum((value.detach().double().square().sum() for value in gradients if value is not None),
                             torch.zeros((), dtype=torch.float64)).sqrt())
            row = counts[variant]
            row['feedback_rows'] += report['feedback_rows']
            row['weighted_loss'] += float(loss.detach())*len(rows)/len(validation)
            row['maximum_gradient_norm'] = max(row['maximum_gradient_norm'], norm)
            row['nonzero_gradient_batches'] += int(norm > 0.)
    after = array_sha(**{name: tensor.detach().numpy() for name, tensor in model.state_dict().items()})
    original_after = array_sha(**{name: tensor.detach().numpy() for name, tensor in original.state_dict().items()})
    if (before != after or original_before != original_after
            or any(p.grad is not None for p in (*model.parameters(), *original.parameters()))):
        raise RuntimeError('gradient-only diagnostic modified Actor weights or gradient buffers')
    passed = (counts['raw']['feedback_rows'] == 0 and counts['raw']['weighted_loss'] == 0.
        and counts['raw']['maximum_gradient_norm'] == 0.
        and counts['regularized']['feedback_rows'] > 0
        and counts['regularized']['maximum_gradient_norm'] > 0.)
    return {'schema': 'warehouse-v621-rcpd-gradient-probe.v1', 'role': role,
        'passed': passed, 'current_fidelity': fidelity, 'variants': counts,
        'actor_parameters_unchanged': True, 'optimizer_steps': 0,
        'post_75_percent_structure_and_core_logits_frozen_clone': True,
        'clone_virtual_frozen_at_step': 150000, 'actual_PPO_joint_steps': 0,
        'program_coefficient': .01,
        'real_PPO_feedback_or_human_benefit_claimed': False}


def validation_strata(actor, program, rows):
    """Diagnostic strata do not select leaves or discard difficult frames."""
    from env.warehouse.public_action_features_v617 import encode_actor_tree_features
    x = encode_actor_tree_features(np.stack([r['obs'] for r in rows]), np.stack([r['mask'] for r in rows]))
    leaves = program.leaf_indices(x)
    probabilities = np.stack([r['nn_probabilities'] for r in rows])
    nn = probabilities.argmax(-1)
    top = program.leaf_logits[leaves].argmax(-1)
    sampled = np.asarray([ACTIONS.index(r['sampled_action']) for r in rows])
    f = np.isin(leaves, list(program.feedback_leaves))
    e = np.isin(leaves, list(program.explanation_leaves))
    matches = top == nn
    actionable = f & matches & (sampled == nn)
    expected = probabilities[np.arange(len(rows)), top]*f*matches
    role_weights = actor.roles[program.role].weights
    core = (role_weights['core_leaf_mask'][leaves] > .5
        if 'core_leaf_mask' in role_weights else np.zeros(len(rows), bool))
    def measured(take):
        n = int(take.sum())
        return {'physical_frames': n, 'core_frames': int((core & take).sum()),
            'feedback_qualified_frames': int((f & take).sum()),
            'explanation_qualified_frames': int((e & take).sum()),
            'feedback_coverage': float(f[take].mean()) if n else None,
            'explanation_coverage': float(e[take].mean()) if n else None,
            'actual_feedback_coverage': float(actionable[take].mean()) if n else None,
            'expected_feedback_coverage': float(expected[take].mean()) if n else None,
            'mean_nn_top_probability': float(probabilities[take].max(-1).mean()) if n else None}
    return {'nn_top_action': {action: measured(nn == i) for i, action in enumerate(ACTIONS)},
        'task': {str(task): measured(np.asarray([r['task'] == task for r in rows])) for task in (1, 2, 3)},
        'moving_nn_top': measured(nn != ACTIONS.index('WAIT')),
        'wait_share_of_explanation_qualified': float(np.sum(e & (nn == ACTIONS.index('WAIT')))/e.sum()) if e.any() else None,
        'purpose': 'all_frame_descriptive_checks_not_extra_selection_gates'}


def development_model_job(plan):
    model, artifact, training = train_common_start(plan)
    actor = frozen(model)
    if actor.role_sha256 != artifact['role_sha256']:
        raise RuntimeError('exported development Actor differs from current frozen NN')
    parts, collection = collect_spent_development(plan, actor, artifact)
    require(plan, 'program_fit')
    roles = {}
    for role in AGENTS:
        selected = [parts[split][role] for split in protocol.diagnostics.SPLITS]
        if plan.recipe == 'additive_control':
            from .warehouse_v6_program import fit_program, WarehouseV6Program
            from scripts.run_warehouse_v619_cached_comparison import program_metrics
            from .warehouse_program_search import ROBUST_METHOD
            program = fit_program(role, actor, *selected, method=ROBUST_METHOD)
            path = plan.output_directory/(role+'.program.json')
            program.save(path)
            program = WarehouseV6Program.load(path, expected_sha256=file_pointer(path)['sha256'],
                actor_role_sha256=artifact['role_sha256'][role])
            probe = {'applicable': False, 'reason': 'diagnostic_control_not_new_candidate', 'passed': False}
        else:
            from .warehouse_v621_program import fit_shared_program, WarehouseV621Program, program_metrics
            program = fit_shared_program(role, actor, *selected)
            path = plan.output_directory/(role+'.program.json')
            program.save(path)
            program = WarehouseV621Program.load(path, expected_sha256=file_pointer(path)['sha256'],
                actor_role_sha256=artifact['role_sha256'][role])
            probe = gradient_probe(plan, model, actor, program, parts['validation'][role])
        metrics = {split: program_metrics(program, actor, parts[split][role]) for split in protocol.diagnostics.SPLITS}
        roles[role] = {'program': file_pointer(path), 'metrics': metrics,
            'development_gate': protocol.diagnostics.gate_metrics(metrics['validation']),
            'validation_strata': validation_strata(actor, program, parts['validation'][role]),
            'gradient_probe': probe}
    if sha256(Path(artifact['path']).read_bytes()).hexdigest() != artifact['sha256']:
        raise RuntimeError('frozen development Actor changed during program fitting')
    program_passed = all(row['development_gate']['passed'] for row in roles.values())
    gradient_passed = all(row['gradient_probe']['passed'] for row in roles.values())
    return {'schema': 'warehouse-v621-bc-development-result.v1', 'status': 'completed',
        'candidate_id': protocol.diagnostics.CANDIDATE_ID, 'batch_id': protocol.BATCH,
        'recipe': plan.recipe, 'initialization': plan.initialization,
        'source_closure_sha256': plan.source_closure_sha256,
        'seed_plan_sha256': plan.seed_plan_sha256,
        'model_implementation_sha256': plan.model_implementation_sha256,
        'output_directory': str(plan.output_directory), 'actor': artifact,
        'training_report': file_pointer(plan.output_directory/'bc.training.json'),
        'development_collection': file_pointer(plan.output_directory/'development.collection.json'),
        'quality': collection['quality'], 'quality_passed': collection['quality_passed'],
        'program_passed': program_passed, 'gradient_probe_passed': gradient_passed,
        'passed': collection['quality_passed'] and program_passed and gradient_passed,
        'roles': roles, 'actual_reset_count': plan.reset_count,
        'actor_updates_during_development_evaluation': 0, 'PPO_joint_steps': 0,
        'fresh_precheck_episodes': 0, 'fresh_validation_evidence': False,
        'publishable': False, 'pilot_opened': False}
