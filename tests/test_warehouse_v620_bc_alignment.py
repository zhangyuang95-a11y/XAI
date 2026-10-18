"""BC-only gradient, physical-mask and unchanged-inference contracts."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from backend.training.warehouse_bc_alignment import (
    ALIGNMENT_RECIPE_VERSION, BCAlignmentRecipe,
    actor_shared_bc_alignment_loss, shared_bc_alignment_loss,
)
from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, MODEL_VERSION, V617_MODEL_VERSION,
)
from backend.training.warehouse_v6_actor_training import (
    WarehouseV6Actor, policy_features_torch, role_hashes,
)
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.pure_nn import AGENTS, action_masks


def tensors():
    full = torch.tensor([[[1., 3., 0., -1., 0.], [2., 1., 0., 4., 0.]],
                         [[0., 2., 4., 1., 0.], [1., 3., 2., 0., 0.]]],
                        requires_grad=True)
    tree = torch.tensor([[[3., 1., 0., -1., 0.], [0., 1., 0., 3., 0.]],
                         [[0., 2., 3., 1., 0.], [1., 4., 2., 0., 0.]]],
                        requires_grad=True)
    primary = torch.tensor([[[1., 2., 0., 0., -1.], [0., 1., 0., 3., 0.]],
                            [[0., 1., 4., 1., 0.], [1., 2., 1., 0., 0.]]],
                           requires_grad=True)
    masks = torch.ones_like(full)
    masks[..., 2] = 0
    labels = torch.tensor([[0, 3], [1, 1]])
    return full, tree, primary, masks, labels


def loss(*args, gamma=.01, **kwargs):
    return shared_bc_alignment_loss(*args, recipe=BCAlignmentRecipe(gamma),
                                    phase="shared_bc", **kwargs)


@pytest.mark.parametrize("gamma", [.01, .03])
def test_two_immutable_recipes_change_only_gamma(gamma):
    recipe = BCAlignmentRecipe(gamma)
    report = recipe.as_dict()
    assert report["version"] == ALIGNMENT_RECIPE_VERSION
    assert report["internal_tree_task_weight"] == .2
    assert report["full_to_internal_tree_kl_weight"] == .2
    assert report["ppo_auxiliary_loss"] == 0
    assert report["external_program_targets_or_gradients"] is False
    assert report["residual_penalty_weight"] == gamma
    with pytest.raises(AttributeError):
        recipe.gamma = .02


@pytest.mark.parametrize("gamma", [0, .02, -.01, True, "0.01", float("nan"), float("inf")])
def test_unregistered_gamma_rejected(gamma):
    with pytest.raises(ValueError, match="0.01 or 0.03"):
        BCAlignmentRecipe(gamma)


@pytest.mark.parametrize("phase", ["ppo", "raw", "regularized", "evaluation", None])
def test_bc_auxiliary_cannot_enter_other_phases(phase):
    with pytest.raises(ValueError, match="never PPO"):
        shared_bc_alignment_loss(*tensors(), recipe=BCAlignmentRecipe(.01), phase=phase)
    # Reject before even inspecting the model; no side effects or environment.
    with pytest.raises(ValueError, match="never PPO"):
        actor_shared_bc_alignment_loss(None, None, None, None,
                                        recipe=BCAlignmentRecipe(.01), phase=phase)


def test_full_distribution_kl_detaches_only_complete_nn_target():
    full, tree, primary, masks, labels = tensors()
    result = loss(full, tree, primary, masks, labels)
    result.full_to_internal_tree_kl.backward()
    assert full.grad is None
    assert primary.grad is None
    assert tree.grad is not None and tree.grad.abs().sum() > 0
    assert torch.count_nonzero(tree.grad[masks == 0]) == 0
    # Same argmax does not erase distribution differences.
    full2 = tree.detach().clone().requires_grad_()
    full2.data[..., 4] += .7
    assert loss(full2, tree.detach(), primary.detach(), masks, labels).full_to_internal_tree_kl > 0


def test_wrong_teacher_region_has_exactly_zero_residual_penalty_and_gradient():
    full, tree, primary, masks, labels = tensors()
    labels = torch.full_like(labels, 4)  # all masked tree decisions disagree
    result = loss(full, tree, primary, masks, labels)
    assert result.correct_tree_fraction == 0
    assert result.residual_penalty == 0
    result.residual_penalty.backward()
    assert torch.count_nonzero(primary.grad) == 0
    assert tree.grad is None  # Correctness selector never routes a gradient.
    assert full.grad is None


def test_legal_centering_ignores_common_shifts_and_illegal_logits():
    full, tree, primary, masks, labels = tensors()
    baseline = loss(full, tree, primary, masks, labels)
    shifted = primary + 113.0
    changed_full = full.masked_fill(masks == 0, 123456.0)
    changed_tree = tree.masked_fill(masks == 0, -98765.0)
    shifted = shifted.masked_fill(masks == 0, -54321.0)
    changed = loss(changed_full, changed_tree, shifted, masks, labels)
    torch.testing.assert_close(baseline.residual_penalty, changed.residual_penalty)
    torch.testing.assert_close(baseline.total, changed.total)
    baseline.total.backward()
    for logits in (full, tree, primary):
        assert torch.count_nonzero(logits.grad[masks == 0]) == 0
        assert logits.grad[masks > 0].abs().sum() > 0


def test_one_legal_action_has_zero_kl_ce_and_policy_relevant_residual():
    full, tree, primary, masks, labels = tensors()
    masks.zero_()
    masks[..., 4] = 1
    result = loss(full, tree, primary, masks, torch.full_like(labels, 4))
    assert result.total == 0
    assert result.correct_tree_fraction == 1


def test_weighting_and_role_accounting_use_all_joint_rows():
    args = tensors()
    weights = torch.tensor([[0., 3.], [1., 0.]], requires_grad=True)
    result = loss(*args, sample_weights=weights)
    constant = loss(*args, sample_weights=100 * weights)
    torch.testing.assert_close(result.total, constant.total)
    torch.testing.assert_close(result.per_role_components.mean(0), torch.stack([
        result.full_task, result.internal_tree_task,
        result.full_to_internal_tree_kl, result.residual_penalty]))
    result.total.backward()
    assert weights.grad is None
    assert set(result.metrics()["per_role_components"]) == set(AGENTS)


@pytest.mark.parametrize("case", ["illegal_teacher", "empty_mask", "nonbinary_mask", "nan_logits",
                                  "wrong_labels", "shape", "zero_weights", "negative_weights"])
def test_malformed_public_training_records_fail_closed(case):
    args = list(tensors())
    weights = None
    if case == "illegal_teacher":
        args[4][0, 0] = 2
    elif case == "empty_mask":
        args[3][0, 0] = 0
    elif case == "nonbinary_mask":
        args[3][0, 0, 0] = .5
    elif case == "nan_logits":
        args[1].data[0, 0, 0] = float("nan")
    elif case == "wrong_labels":
        args[4] = args[4].float()
    elif case == "shape":
        args[1] = args[1][:, 0]
    elif case == "zero_weights":
        weights = torch.zeros(2, 2)
    elif case == "negative_weights":
        weights = torch.tensor([[1., -1.], [1., 1.]])
    with pytest.raises(ValueError):
        loss(*args, sample_weights=weights)


def public_rows():
    # Construct valid public states without resetting or evaluating a scene.
    states = [WarehouseState(
        episode_id=620, frame=frame,
        agents=[AgentState("robot_1", position, 70.),
                AgentState("robot_2", STUDY_MAP_LAYOUT.robot_start_positions[1], 70.)],
        tasks=[DeliveryTask("task_0", (0, 3), (3, 2)),
               DeliveryTask("task_1", (1, 0), (3, 1))], next_task_index=2)
        for frame, position in ((1, (1, 0)), (2, (1, 1)), (3, (5, 2)))]
    obs = torch.from_numpy(np.stack([encode_public_observation(state) for state in states]))
    masks = torch.from_numpy(np.stack([action_masks(state) for state in states]))
    return obs, masks, policy_features_torch(obs, masks)


def test_independent_roles_jointly_update_without_mutating_forward_or_shared_start(monkeypatch):
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from backend.training import warehouse_v6_program

    def forbidden(*args, **kwargs):
        raise AssertionError("BC loss must not reset environments or fit external programs")

    monkeypatch.setattr(WarehouseMultiAgentEnv, "reset", forbidden)
    monkeypatch.setattr(warehouse_v6_program, "fit_program", forbidden)
    torch.manual_seed(620)
    model = WarehouseV6Actor()
    obs, masks, encoded = public_rows()
    for index, role in enumerate(AGENTS):
        with torch.no_grad():
            model.actors[role].gate[-1].bias.fill_(.4 + .2 * index)
        model.actors[role].set_tree_branch_enabled(True)
    labels = masks.argmax(-1).long()
    before = role_hashes(model)
    before_state = deepcopy(model.state_dict())
    before_forward = model.outputs_encoded(encoded, masks).detach().clone()
    result = actor_shared_bc_alignment_loss(model, encoded, masks, labels,
        recipe=BCAlignmentRecipe(.01), phase="shared_bc")
    # Computing a training objective leaves Actor state and actions intact.
    assert torch.equal(before_forward, model.outputs_encoded(encoded, masks))
    for key, value in before_state.items():
        assert torch.equal(value, model.state_dict()[key])
    result.total.backward()
    for role in AGENTS:
        actor = model.actors[role]
        assert actor.tree.leaf_logits.grad.abs().sum() > 0
        assert actor.tree.split.weight.grad.abs().sum() > 0
        assert actor.action_residual[0].weight.grad.abs().sum() > 0
        assert actor.gate[-1].bias.grad.abs().sum() > 0
    torch.optim.SGD(model.actors.parameters(), lr=.001).step()
    after = role_hashes(model)
    assert all(before[role] != after[role] for role in AGENTS)
    raw, regularized = WarehouseV6Actor(), WarehouseV6Actor()
    raw.load_state_dict(deepcopy(model.state_dict()))
    regularized.load_state_dict(deepcopy(model.state_dict()))
    assert role_hashes(raw) == role_hashes(regularized) == after
    assert len(set(after.values())) == 2
    frozen = FrozenWarehouseV6Actor.from_model(model)
    assert frozen.model_version == MODEL_VERSION
    frozen_masked = np.where(masks.numpy() > .5,
                             frozen.logits(obs.numpy(), masks.numpy()), -1e9)
    np.testing.assert_allclose(frozen_masked,
        model.outputs_encoded(encoded, masks).detach().numpy(), atol=1e-4, rtol=0)


def test_historical_bounded_actor_does_not_silently_enter_new_bc_adapter():
    _, masks, encoded = public_rows()
    model = WarehouseV6Actor(model_version=V617_MODEL_VERSION)
    with pytest.raises(ValueError, match="additive Actor"):
        actor_shared_bc_alignment_loss(model, encoded, masks, masks.argmax(-1),
            recipe=BCAlignmentRecipe(.01), phase="shared_bc")
