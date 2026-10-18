"""Actual small PPO/physics checks; none of these fixtures are admission runs."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import (
    FEATURE_DIM, GEOMETRY_DIM, FrozenWarehouseV6Actor, WarehouseV6Actor, role_hashes,
)
from backend.training.warehouse_v6_program import (
    INDEPENDENT_SELECTIVE_METHOD, WarehouseV6Program,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS


@pytest.fixture(autouse=True)
def _two_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _wait_model():
    model = WarehouseV6Actor()
    with torch.no_grad():
        for actor in model.actors.values():
            actor.refresh_sparse_support()
            # A known neural policy used only to give this unit test enough
            # actual sampled actions agreeing with the synthetic WAIT program.
            for parameter in actor.action_residual.parameters():
                parameter.zero_()
            actor.action_residual[0].weight[0, FEATURE_DIM+GEOMETRY_DIM+4] = 1.
            actor.action_residual[2].weight[0, 0] = 1.
            actor.action_residual[4].weight[0, 0] = 4.
    return model


def _physics_episode(model, monkeypatch, *, steps=12):
    # Replace only the environment factory dependency. No training capability
    # or acceptance marker is armed. collect_episode itself samples both NNs
    # and executes real physical transitions.
    monkeypatch.setattr(runner, '_environment', lambda **_kwargs:
        WarehouseMultiAgentEnv(participant_study_config(**runner.TRAINING_REWARD_OVERRIDES)))
    episode, summary = runner.collect_episode(
        model, seed=987618, generator=torch.Generator().manual_seed(618),
        max_steps=steps, episode_index=0)
    assert summary['post_policy_interventions'] == 0
    fixed = {role: [{
        'obs': row['obs'][index].copy(), 'mask': row['mask'][index].copy(),
        'episode': row['episode_key'], 'base_seed': row['seed'],
        'episode_id': row['episode_id'], 'frame': row['frame'],
    } for row in episode['records']] for index, role in enumerate(AGENTS)}
    return episode, fixed


def _programs(model):
    digests = role_hashes(model)
    return {role: WarehouseV6Program(
        role=role, actor_role_sha256=digests[role], root_ref=-1,
        split_weights=np.zeros((0, FEATURE_DIM), np.float32),
        split_bias=np.zeros(0, np.float32), left_refs=np.zeros(0, np.int64),
        right_refs=np.zeros(0, np.int64),
        leaf_logits=np.array([[0., 0., 0., 0., 5.]], np.float32),
        feedback_leaves=frozenset({0}), explanation_leaves=frozenset({0}),
        report={'fit_method': INDEPENDENT_SELECTIVE_METHOD,
                'validation': {'feedback_passed': True, 'explanation_passed': True}},
    ) for role in AGENTS}


def test_raw_real_ppo_never_calls_teacher_or_core_feedback(monkeypatch):
    model = _wait_model()
    episode, _ = _physics_episode(model, monkeypatch)
    before = role_hashes(model)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Raw PPO reached teacher/program/auxiliary CE')

    monkeypatch.setattr(runner, 'core_intrinsic_loss', forbidden, raising=False)
    monkeypatch.setattr(runner, '_bc_tree_distillation', forbidden)
    monkeypatch.setattr(runner, '_teacher_rows', forbidden)
    monkeypatch.setattr(runner, '_program_feedback', forbidden)
    monkeypatch.setattr(torch.nn.functional, 'cross_entropy', forbidden)
    result = runner.ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4),
                              [episode], None, progress=.95, cost_multiplier=0., epochs=2)
    assert result['program_coefficient'] == 0
    assert result['core_intrinsic_loss'] == 0
    assert all(result['feedback_rows'][role] == 0 and
               result['feedback_gradient_norm'][role] == 0 for role in AGENTS)
    assert all(before[role] != role_hashes(model)[role] for role in AGENTS)


def test_regularized_revalidates_current_actor_each_epoch_and_both_roles_get_gradient(monkeypatch):
    model = _wait_model()
    episode, fixed = _physics_episode(model, monkeypatch)
    programs = _programs(model)
    checked = []
    actual_validation = runner.current_actor_fidelity

    def validation(program, actor, rows, **kwargs):
        checked.append((program.role, actor.role_sha256[program.role],
                        tuple(row['validation_rng_seed'] for row in rows)))
        return actual_validation(program, actor, rows, **kwargs)

    monkeypatch.setattr(runner, 'current_actor_fidelity', validation)
    result = runner.ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4),
        [episode], programs, progress=.95, cost_multiplier=0., epochs=2,
        program_validation_rows=fixed)
    assert len(checked) == 2*len(AGENTS)
    for role in AGENTS:
        role_checks = [item for item in checked if item[0] == role]
        assert role_checks[0][1] != role_checks[1][1]
        assert role_checks[0][2] == role_checks[1][2]
        assert result['feedback_rows'][role] > 0
        assert result['feedback_gradient_norm'][role] > 0
        assert result['feedback_disabled_epochs'][role] == 0
        assert all(row['passed'] for row in result['current_actor_validation'][role])
        assert result['feedback_window_actor_rows'][role] == 2*len(episode['records'])


def test_invalid_or_missing_current_validation_disables_all_feedback(monkeypatch):
    model = _wait_model()
    episode, fixed = _physics_episode(model, monkeypatch)
    programs = _programs(model)
    programs[AGENTS[0]].report['validation']['feedback_passed'] = False
    fixed[AGENTS[1]] = []
    result = runner.ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4),
        [episode], programs, progress=.95, cost_multiplier=0., epochs=1,
        program_validation_rows=fixed)
    assert all(result['feedback_rows'][role] == 0 and
               result['feedback_gradient_norm'][role] == 0 and
               result['feedback_disabled_epochs'][role] == 1 for role in AGENTS)


def test_feedback_requires_qualified_program_action_current_argmax_and_actual_sample(monkeypatch):
    model = _wait_model()
    episode, fixed = _physics_episode(model, monkeypatch)
    role = AGENTS[0]
    rows = episode['records'][:3]
    obs = torch.from_numpy(np.stack([row['obs'][0] for row in rows]))
    mask = torch.from_numpy(np.stack([row['mask'][0] for row in rows]))
    logits = model.role_logits(role, obs, mask)
    samples = logits.detach().argmax(-1)
    # An actual non-argmax sample cannot receive the old argmax evidence.
    alternative = int(torch.nonzero(mask[1] > .5)[0])
    assert alternative != ACTIONS.index('WAIT')
    samples[1] = alternative
    loss, count, kl = runner._program_feedback(
        logits, obs, mask, samples, _programs(model)[role], coefficient=.01)
    assert count == 2 and loss > 0 and kl > 0
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.count_nonzero(gradient[1]) == 0
    assert gradient[[0, 2]].abs().sum() > 0


def test_diagnostics_are_read_only_and_freeze_resume_keeps_leaf_learning(monkeypatch):
    model = _wait_model()
    episode, _ = _physics_episode(model, monkeypatch)
    # Distinct synthetic scene IDs exercise receipt structure only; this test
    # does not claim independent empirical program or quality validation.
    buffers = {role: [{'obs': row['obs'][index], 'mask': row['mask'][index],
                      'base_seed': 990000+scene}
                     for scene in range(40) for row in episode['records'][:2]]
               for index, role in enumerate(AGENTS)}
    before = deepcopy(model.state_dict())
    for step in (48000, 96000, 144000):
        result = runner._refresh_qualified_topology(model, buffers, stage=f'ppo_{step}')
        assert result['actor_role_sha256_before'] == result['actor_role_sha256_after']
        assert result['actor_rebuilt'] is False and result['auxiliary_ce_updates'] == 0
        assert result['guard_invariants_passed'] and result['support_locked']
        for name, value in before.items():
            assert torch.equal(value, model.state_dict()[name])
    for actor in model.actors.values():
        actor.gate[-1].bias.data.fill_(.4)
        actor.set_topology_trainable(False)
        actor.set_certified_core_logits_trainable(False)
    frozen_splits = runner._role_topology_hashes(model)
    frozen_support = runner._role_support_hashes(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Establish both optimizer momentum and a mid-training state snapshot.
    runner.ppo_update(model, optimizer, [episode], None, progress=.8,
                      cost_multiplier=0., epochs=1, topology_frozen=True)
    saved_model, saved_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    restored = WarehouseV6Actor()
    restored.load_state_dict(saved_model)
    for actor in restored.actors.values():
        actor.set_topology_trainable(False)
        actor.set_certified_core_logits_trainable(False)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(saved_optimizer)
    runner.ppo_update(model, optimizer, [episode], None, progress=.9,
                      cost_multiplier=0., epochs=1, topology_frozen=True)
    runner.ppo_update(restored, restored_optimizer, [episode], None, progress=.9,
                      cost_multiplier=0., epochs=1, topology_frozen=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert runner._role_topology_hashes(model) == frozen_splits
    assert runner._role_support_hashes(model) == frozen_support
    for role in AGENTS:
        assert not torch.equal(saved_model[f'actors.{role}.tree.leaf_logits'],
                               model.actors[role].tree.leaf_logits)
