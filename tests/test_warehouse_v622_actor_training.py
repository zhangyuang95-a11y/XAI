"""Synthetic tensors/artifacts only: no environment, rollout, or scientific fit."""
from copy import deepcopy
from hashlib import sha256
import random

import numpy as np
import pytest
import torch

from backend.training import warehouse_v622_actor_training as t
from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, FEATURE_DIM, _role_arrays,
)
from backend.training.warehouse_v6_actor_training import (
    WarehouseV6Actor, export_actor, role_hashes,
)
from env.warehouse.public_action_features_v617 import encode_actor_tree_features
from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES
from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.fixture
def artifact(tmp_path):
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(0)
        model = WarehouseV6Actor()
    with torch.no_grad():
        for role in t.ROLES:
            tree = model.actors[role].tree
            tree.split.weight.zero_(); tree.split.bias.zero_()
            tree.left_refs.zero_(); tree.right_refs.zero_()
            tree.node_count[0] = 0; tree.leaf_count[0] = 1
            tree.leaf_logits.zero_()
            tree.leaf_logits[0] = torch.tensor([.2, -.1, .3, -.2, .1])
            model.actors[role].gate[-1].bias.fill_(.4)
    pointer = export_actor(model, tmp_path/'reference.npz', pair_seed=0,
                           variant='raw', candidate_id='synthetic-only')
    return model, pointer


def imported(artifact, seed=1):
    _, p = artifact
    return t.initialize_from_reference(p['path'], p['sha256'], p['role_sha256'],
        initialization_seed=seed)


def physical_rows():
    names = {n: i for i, n in enumerate(OBSERVATION_FEATURE_NAMES)}
    points = list(layout.passable_positions)
    obs, masks = [], []
    for own in points[:4]:
        row = np.zeros(len(names), np.float32)
        for endpoint, position in [('self', own), ('partner', points[-1]),
                ('charger', layout.charger_position), ('order_0_pickup', points[0]),
                ('order_0_delivery', points[-1]), ('order_1_pickup', points[1]),
                ('order_1_delivery', points[-2])]:
            row[names[endpoint+'_row']] = position[0]/(layout.rows-1)
            row[names[endpoint+'_column']] = position[1]/(layout.cols-1)
        row[names['self_active']] = 1.; row[names['self_battery_fraction']] = .8
        for r, line in enumerate(layout.tiles):
            for c, tile in enumerate(line):
                row[names[f'wall_row_{r}_column_{c}']] = tile == '#'
        mask = [float(a == 'WAIT' or (own[0]+MOVE_DELTAS[a][0],
                own[1]+MOVE_DELTAS[a][1]) in points) for a in ACTIONS]
        obs.append(row); masks.append(mask)
    masks = np.asarray(masks, np.float32)
    x = encode_actor_tree_features(np.asarray(obs), masks)
    return torch.from_numpy(np.stack([x, x], 1)), torch.from_numpy(np.stack([masks, masks], 1))


def observer(state, x, masks):
    with torch.no_grad():
        values = t.masked_reference_kl(state.model.outputs_encoded(x, masks),
            t.reference_logits_encoded(state, x), masks).mean(0)
    return {r: max(0., float(values[i])) for i, r in enumerate(t.ROLES)}


def test_import_all_npz_actor_weights_fresh_critic_and_exact_export(artifact, tmp_path):
    original, pointer = artifact
    before = torch.get_rng_state().clone()
    state = imported(artifact)
    assert torch.equal(before, torch.get_rng_state())
    assert role_hashes(state.model) == pointer['role_sha256']
    assert not state.actor_optimizer.state and not state.critic_optimizer.state
    assert state.initialization_report['resumed_training_state'] is False
    assert state.initialization_report['old_critic_or_optimizer_imported'] is False
    paired = imported(artifact)
    for name, value in state.model.state_dict().items():
        assert torch.equal(value, paired.model.state_dict()[name]), name
    first = next(state.model.actors[t.ROLES[0]].parameters())
    second = next(paired.model.actors[t.ROLES[0]].parameters())
    assert first.data_ptr() != second.data_ptr()  # same start, independent storage
    for role in t.ROLES:
        for key, value in _role_arrays(original.actors[role]).items():
            np.testing.assert_array_equal(value, _role_arrays(state.model.actors[role])[key])
    assert any(not torch.equal(a, b) for a, b in zip(original.cost_critic.parameters(), state.model.cost_critic.parameters()))
    x, masks = physical_rows()
    assert t.verify_initial_equivalence(state, x, masks)['masked_argmax_mismatches'] == 0
    out = t.export_current_actor(state, tmp_path/'export.npz', pair_seed=0,
        variant='regularized', candidate_id='synthetic-only')
    restored = FrozenWarehouseV6Actor.load(out['path'], expected_sha256=out['sha256'])
    assert restored.role_sha256 == pointer['role_sha256']
    for r in t.ROLES:
        np.testing.assert_array_equal(restored.roles[r].logits_encoded(x[:, 0].numpy()),
            state.reference.roles[r].logits_encoded(x[:, 0].numpy()))
    with pytest.raises(FileExistsError):
        t.export_current_actor(state, out['path'], pair_seed=0, variant='raw', candidate_id='synthetic-only')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        t.initialize_from_reference(pointer['path'], '0'*64, pointer['role_sha256'], initialization_seed=1)


def test_reference_kl_direction_mask_and_gradient():
    logits = torch.tensor([[1., 2., 99., -1., .5]], requires_grad=True)
    reference = torch.tensor([[2., 1., -99., 0., .5]], requires_grad=True)
    masks = torch.tensor([[1., 1., 0., 1., 1.]], requires_grad=True)
    loss = t.masked_reference_kl(logits, reference, masks).sum()
    loss.backward()
    p = torch.softmax(logits.detach().masked_fill(masks.detach() == 0, -torch.inf), -1)
    q = torch.softmax(reference.detach().masked_fill(masks.detach() == 0, -torch.inf), -1)
    torch.testing.assert_close(logits.grad, p-q)
    assert logits.grad[0, 2] == 0 and reference.grad is None and masks.grad is None


def provenance(sources=('reference_train','current_train'), split='train'):
    return [t.ReferenceRow(s, split, 'synthetic-episode-'+str(i), 'a'*64) for i, s in enumerate(sources)]


def test_reference_training_only_equal_mix_and_beta_roles():
    logits = torch.zeros(2, 2, 5, requires_grad=True)
    masks = torch.ones_like(logits)
    result = t.reference_kl_loss(logits, logits.detach(), masks,
        provenance=provenance(), beta={r: 1. for r in t.ROLES})
    assert float(result.detach()) == 0
    for bad in [provenance(split='validation'), provenance(('current_train','current_train')),
                [t.ReferenceRow('reference_train','train','ep','bad'), provenance()[1]]]:
        with pytest.raises(ValueError):
            t.reference_kl_loss(logits, logits.detach(), masks, provenance=bad, beta={r:1. for r in t.ROLES})
    beta = t.updated_reference_beta({r: 1. for r in t.ROLES}, {t.ROLES[0]: .02, t.ROLES[1]: 0.})
    assert beta[t.ROLES[0]] == pytest.approx(np.exp(.05))
    assert beta[t.ROLES[1]] == pytest.approx(np.exp(-.05))
    bounded = t.updated_reference_beta({r:100. for r in t.ROLES}, {r:1.e300 for r in t.ROLES})
    assert set(bounded.values()) == {100.}
    with pytest.raises(ValueError):
        t.updated_reference_beta(beta, {r:float('nan') for r in t.ROLES})


def test_program_loss_real_nn_gradient_all_physical_denominator_and_no_target_gradient():
    logits = torch.tensor([[.1, 0., 0., 0., .2]]*4, requires_grad=True)
    q = torch.tensor([[.1, .1, .1, .1, .6]]*4, requires_grad=True)
    mask = torch.ones_like(logits, requires_grad=True)
    result = t.program_kl_loss(logits, q, mask, torch.tensor([4,4,0,4]),
        torch.tensor([True,False,True,True]), raw_program_top=torch.tensor([4,4,4,0]), variant='regularized')
    assert result.selected.tolist() == [True,False,False,False]
    result.loss.backward()
    expected = torch.zeros_like(logits)
    expected[0] = .01/4*(torch.softmax(logits.detach()[0], -1)-q.detach()[0])
    torch.testing.assert_close(logits.grad, expected)
    assert q.grad is None and mask.grad is None
    assert torch.count_nonzero(logits.grad) > 0  # same argmax, different distribution
    raw_logits = logits.detach().clone().requires_grad_()
    raw = t.program_kl_loss(raw_logits, None, mask.detach(), None, None,
        raw_program_top=None, variant='raw')
    raw.loss.backward()
    assert raw.loss.item() == 0 and raw.coefficient == 0 and not raw.selected.any()
    assert torch.count_nonzero(raw_logits.grad) == 0


def test_program_pre_mask_action_legality_and_joint_roles():
    logits = torch.zeros(2, 2, 5, requires_grad=True)
    masks = torch.ones_like(logits); masks[..., 0] = 0
    q = masks/4
    loss = t.program_kl_loss(logits, q, masks, torch.ones(2,2,dtype=torch.long),
        torch.ones(2,2,dtype=torch.bool), raw_program_top=torch.tensor([[0,1],[1,1]]), variant='regularized')
    assert loss.selected.tolist() == [[False,True],[True,True]]
    assert loss.selected_counts.tolist() == [1,2]
    with pytest.raises(ValueError, match='physically masked'):
        t.program_kl_loss(logits, torch.ones_like(q)/5, masks, torch.ones(2,2,dtype=torch.long),
            torch.ones(2,2,dtype=torch.bool), raw_program_top=torch.ones(2,2,dtype=torch.long), variant='regularized')


def test_complete_episode_shutdown_targets_never_label_partial_as_negative():
    assert t.episode_shutdown_returns(np.array([False,False,True]), terminated=True, truncated=False).tolist() == [1,1,1]
    negative = t.episode_shutdown_returns(np.zeros(120,dtype=bool), terminated=False, truncated=True)
    assert negative.sum() == 0  # no positive event is legitimate, not manufactured
    batch = t.complete_shutdown_batch([{'episode_id':'synthetic-complete',
        'shutdown_events':np.zeros(120,dtype=bool), 'terminated':False, 'truncated':True}])
    assert batch['no_positive_shutdown_examples_observed'] is True
    assert batch['cost_critic_validated'] is False and batch['partial_episodes_labeled_negative'] == 0
    with pytest.raises(ValueError):
        t.episode_shutdown_returns(np.zeros(119,dtype=bool), terminated=False, truncated=True)
    with pytest.raises(ValueError):
        t.episode_shutdown_returns(np.array([True,False]), terminated=True, truncated=False)
    predictions = torch.zeros_like(negative, requires_grad=True)
    t.shutdown_critic_loss(predictions, negative).backward()
    assert predictions.grad is not None


def test_warmup_only_updates_critic_and_post_step_observer(artifact):
    state = imported(artifact)
    x, masks = physical_rows()
    original = role_hashes(state.model); beta = deepcopy(state.beta)
    state.advance_joint_steps(3999)
    loss = state.model.outputs_encoded(x, masks)[..., 4].mean()
    with pytest.raises(RuntimeError, match='critic updates only'):
        state.actor_step(loss, observed_reference_kl=lambda: observer(state,x,masks))
    obs = torch.zeros(4, state.model.cost_critic[0].in_features)
    targets = torch.zeros(4, 1)
    state.critic_step(t.shutdown_critic_loss(state.model.cost_critic(obs), targets))
    assert state.critic_updates == 1 and state.actor_updates == 0
    assert original == role_hashes(state.model) and state.beta == beta
    state.advance_joint_steps(1)
    with pytest.raises(ValueError, match='observer'):
        state.actor_step(loss, observed_reference_kl={r:.01 for r in t.ROLES})
    observed = []
    def after():
        observed.append(role_hashes(state.model) != original)
        return observer(state,x,masks)
    state.actor_step(loss, observed_reference_kl=after)
    assert observed == [True] and state.actor_updates == 1


def test_checkpoint_restores_optimizer_critics_beta_rng_and_next_update(artifact, tmp_path):
    state = imported(artifact); state.advance_joint_steps(4000)
    x, masks = physical_rows()
    state.actor_step(state.model.outputs_encoded(x,masks)[...,4].mean(),
        observed_reference_kl=lambda: observer(state,x,masks))
    obs = torch.zeros(2, state.model.cost_critic[0].in_features)
    state.critic_step(state.model.cost_critic(obs).square().mean())
    state.extra_rng_states = {'action_generator': np.random.default_rng(0).bit_generator.state}
    ptr = t.save_checkpoint(state, tmp_path/'training.pt')
    expected_draws = (random.random(), np.random.random(), torch.rand(3))
    expected_beta = deepcopy(state.beta)
    state.actor_step(state.model.outputs_encoded(x,masks)[...,3].mean(),
        observed_reference_kl=lambda: observer(state,x,masks))
    expected = deepcopy(state.model.state_dict())
    expected_optimizer = deepcopy(state.actor_optimizer.state_dict())
    resumed = t.load_checkpoint(ptr['path'], ptr['sha256'])
    assert resumed.resumed_training_state and resumed.actor_updates == 1 and resumed.critic_updates == 1
    assert resumed.joint_steps == 4000 and resumed.beta == expected_beta
    assert resumed.extra_rng_states == state.extra_rng_states
    actual_draws = (random.random(), np.random.random(), torch.rand(3))
    assert actual_draws[:2] == expected_draws[:2]
    assert torch.equal(actual_draws[2], expected_draws[2])
    resumed.actor_step(resumed.model.outputs_encoded(x,masks)[...,3].mean(),
        observed_reference_kl=lambda: observer(resumed,x,masks))
    for name, tensor in resumed.model.state_dict().items():
        assert torch.equal(tensor, expected[name]), name
    for key, value in resumed.actor_optimizer.state_dict()['state'].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, expected_optimizer['state'][key][name])
    with pytest.raises(RuntimeError, match='checkpoint hash mismatch'):
        t.load_checkpoint(ptr['path'], '0'*64)
