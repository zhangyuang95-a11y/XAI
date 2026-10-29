"""Synthetic numerical tests: no environment, scene inventory or Actor action."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from backend.training.warehouse_v643_risk_training import (
    RiskTrainingConfig, RiskTrainingCore, make_risk_training_core,
)
from backend.training.warehouse_v6_shutdown_cost import RiskDualConfig


@pytest.fixture(autouse=True)
def single_thread_numerics():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actors = nn.ModuleDict({name: nn.Linear(2, 2)
                                    for name in ('robot_1', 'robot_2')})
        self.reward_critic = nn.Linear(2, 1)
        self.cost_critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))


def config(**kwargs):
    return RiskTrainingConfig(episode_horizon=4, low_battery_columns=(1,), **kwargs)


def make_core(seed=31, settings=None):
    torch.manual_seed(seed)
    model = ToyModel()
    core = make_risk_training_core(model, observation_dim=2, rng=np.random.default_rng(seed),
                                   config=settings or config())
    return model, core


def identity(text):
    return sha256(text.encode()).hexdigest()


def episode(name, shutdown=False, frames=None, obs=None):
    frames = frames or (2 if shutdown else 4)
    rows = []
    for frame in range(frames):
        observation = np.array([1.0 if shutdown else -1.0, .10 if shutdown else .75],
                               dtype=np.float32) if obs is None else obs
        rows.append(dict(episode_id=name, frame=frame, obs=observation,
                         shutdown=shutdown and frame == frames - 1,
                         terminated=shutdown and frame == frames - 1,
                         truncated=not shutdown and frame == frames - 1,
                         partition='train'))
    return rows


def add(core, rows, name='batch-1', step=120):
    return core.observe_batch(rows, batch_sha256=identity(name), end_joint_step=step,
                              partition='train')


def equal_state(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_complete_train_episodes_preserve_warmup_positives_and_count_once():
    settings = config(capacity_episodes=3, risk=RiskDualConfig(
        target=.02, window_episodes=256, minimum_episodes=2, release_upper=.01,
        initial_multiplier=2.0))
    _, core = make_core(settings=settings)
    observed = add(core, episode('warmup-failure', True) + episode('safe-1'))
    assert observed['new_complete_episodes'] == 2
    assert observed['buffered_shutdown_episodes'] == 1
    assert observed['risk']['phase'] == 'increase'
    assert core.risk.multiplier == 2.1
    risk = deepcopy(core.risk.snapshot())
    for _ in range(6):
        result = core.update_critic()
        assert result['new_shutdown_observations_from_replay'] == 0
        assert result['risk_observed_complete_episode_count'] == 2
        assert result['historical_Actor_gradient_updates'] == 0
    assert core.risk.snapshot() == risk
    add(core, episode('safe-2'), name='batch-2', step=240)
    assert core.episodes['warmup-failure']['shutdown'] is True
    add(core, episode('safe-3'), name='batch-3', step=360)
    assert tuple(core.episodes) == ('safe-1', 'safe-2', 'safe-3')
    assert len(core.seen_episodes) == 4
    # FIFO retention has a declared finite boundary; eviction cannot allow an
    # old positive to be counted as a fresh episode again.
    with pytest.raises(ValueError, match='already consumed'):
        add(core, episode('warmup-failure', True), name='batch-4', step=480)


@pytest.mark.parametrize('change', ['partial', 'early_terminal', 'nontrain', 'contradiction'])
def test_incomplete_or_nontraining_inputs_are_rejected_atomically(change):
    _, core = make_core()
    rows = episode('bad')
    if change == 'partial':
        rows = rows[:2]
    elif change == 'early_terminal':
        rows[1].update(shutdown=True, terminated=True)
    elif change == 'nontrain':
        rows[2]['partition'] = 'validation'
    else:
        rows[0]['cost_return'] = 1.0
    before = core.snapshot()
    with pytest.raises(ValueError):
        add(core, rows)
    equal_state(before, core.snapshot())
    with pytest.raises(ValueError, match='cannot train'):
        core.observe_batch(episode('quality'), batch_sha256=identity('quality'),
                           end_joint_step=120, partition='quality')


def test_actual_torch_learning_changes_only_cost_critic_and_detaches_history():
    settings = config(learning_rate=.01)
    model, core = make_core(settings=settings)
    public_input = torch.tensor([1.0, .10], requires_grad=True)
    historical_action = model.actors['robot_1'](public_input)
    historical_logp = historical_action.log_softmax(0)
    failure_rows = episode('failure', True, obs=public_input)
    for row in failure_rows:
        row.update(action=historical_action, logp=historical_logp)
    rows = failure_rows + episode('safe')
    for parameter in model.actors.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    for parameter in model.reward_critic.parameters():
        parameter.grad = torch.full_like(parameter, -.25)
    protected = [(p, p.detach().clone(), p.grad.clone())
                 for module in (model.actors, model.reward_critic) for p in module.parameters()]
    original_critic = {k: v.clone() for k, v in model.cost_critic.state_dict().items()}
    add(core, rows)
    initial = core._forward(list(core.episodes.values()))[1]['training_episode_mean_BCE']
    for _ in range(80):
        result = core.update_critic()
        assert np.isfinite(result['cost_critic_gradient_norm_before_clip'])
        assert result['cost_critic_gradient_norm_before_clip'] > 0
        assert result['historical_Actor_gradient_updates'] == 0
    final = core._forward(list(core.episodes.values()))[1]['training_episode_mean_BCE']
    assert final < initial * .10
    assert any(not torch.equal(value, original_critic[key])
               for key, value in model.cost_critic.state_dict().items())
    for parameter, value, gradient in protected:
        assert torch.equal(parameter, value)
        assert torch.equal(parameter.grad, gradient)
    assert public_input.grad is None
    assert set(core.episodes['failure']) == {
        'episode_id', 'observations', 'shutdown', 'frames', 'batch_sha256',
        'collected_joint_step', 'outcome_scope', 'episode_sha256'}
    assert result['label_scope'] == 'actual_behavior_policy_outcomes_not_independent_current_policy_risk'
    assert result['low_battery_frames'] == 2


def test_loss_is_episode_balanced_not_frame_balanced():
    _, core = make_core()
    with torch.no_grad():
        for parameter in core.critic.parameters():
            parameter.zero_()
        core.critic[-1].bias.fill_(2.0)
    add(core, episode('short-failure', True, frames=1) + episode('long-safe'))
    loss, metrics = core._forward(list(core.episodes.values()))
    expected = (F.softplus(torch.tensor(-2.0)) + F.softplus(torch.tensor(2.0))) / 2
    frame_weighted = (F.softplus(torch.tensor(-2.0)) + 4 * F.softplus(torch.tensor(2.0))) / 5
    assert torch.allclose(loss, expected, atol=1e-7)
    assert not torch.allclose(loss, frame_weighted)
    assert metrics['complete_episodes'] == 2
    assert metrics['physical_observations'] == 5


def test_duplicate_batch_and_complete_episode_cannot_reenter_the_risk_window():
    _, core = make_core()
    add(core, episode('first', True))
    before = core.snapshot()
    with pytest.raises(ValueError, match='new hash-bound'):
        add(core, episode('second'), step=240)
    with pytest.raises(ValueError, match='already consumed'):
        add(core, episode('first', True), name='new-batch', step=240)
    equal_state(before, core.snapshot())
    empty = add(core, [], name='empty', step=240)
    assert empty['new_complete_episodes'] == 0
    assert empty['risk']['phase'] == 'no_new_complete_episode'
    assert len(core.risk.completed) == 1


def test_numeric_checkpoint_restores_sampling_optimizer_and_next_update_exactly():
    _, core = make_core()
    add(core, episode('a', True) + episode('b') + episode('c') + episode('d', True))
    for _ in range(3):
        core.update_critic(episode_count=2)
    snapshot = core.snapshot()
    model2, restored = make_core(seed=999)
    protected = deepcopy(model2.actors.state_dict())
    restored.restore(snapshot)
    assert snapshot['Actor_or_reward_weights_included'] is False
    assert snapshot['training_authority'] is False
    for _ in range(4):
        equal_state(core.update_critic(episode_count=2), restored.update_critic(episode_count=2))
        equal_state(core.snapshot(), restored.snapshot())
    equal_state(protected, model2.actors.state_dict())


@pytest.mark.parametrize('tamper', ['episode', 'config', 'batch', 'risk', 'critic',
                                   'optimizer', 'moment', 'rng'])
def test_tampered_checkpoint_is_rejected_before_model_mutation(tamper):
    _, core = make_core()
    add(core, episode('a', True) + episode('b'))
    core.update_critic()
    value = core.snapshot()
    if tamper == 'episode':
        value['episodes'][0]['shutdown'] = False
    elif tamper == 'config':
        value['config']['risk']['target'] = .5
    elif tamper == 'batch':
        value['seen_batches'] = []
    elif tamper == 'risk':
        value['risk']['multiplier'] = float('nan')
    elif tamper == 'critic':
        next(iter(value['cost_critic_state'].values())).view(-1)[0] = float('nan')
    elif tamper == 'optimizer':
        value['cost_optimizer_state']['param_groups'][0]['lr'] = .1
    elif tamper == 'moment':
        next(iter(value['cost_optimizer_state']['state'].values()))['exp_avg'].view(-1)[0] = float('nan')
    else:
        value['rng_state'] = {'bit_generator': 'invalid'}
    _, restored = make_core(seed=71)
    before = restored.snapshot()
    with pytest.raises(ValueError):
        restored.restore(value)
    equal_state(before, restored.snapshot())


def test_disjoint_parameter_ownership_and_deterministic_layers_are_required():
    model, _ = make_core()
    settings = config()
    bad_optimizer = torch.optim.Adam(list(model.cost_critic.parameters()) +
                                     list(model.actors.parameters()),
                                     lr=settings.learning_rate, eps=settings.optimizer_epsilon)
    with pytest.raises(ValueError, match='only disjoint cost critic'):
        RiskTrainingCore(model.cost_critic, bad_optimizer,
                         forbidden_modules=(model.actors, model.reward_critic),
                         observation_dim=2, rng=np.random.default_rng(0), config=settings)
    model.cost_critic = nn.Sequential(nn.Linear(2, 8), nn.Dropout(.1), nn.Linear(8, 1))
    with pytest.raises(ValueError, match='deterministic registered'):
        make_risk_training_core(model, observation_dim=2, rng=np.random.default_rng(0),
                                config=settings)
