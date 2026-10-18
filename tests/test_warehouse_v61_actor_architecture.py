"""V6.1 role tree initialization and frozen relational Actor parity."""
from __future__ import annotations

import numpy as np
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, MAX_SPLIT_FEATURES,
    policy_features_torch,
)
from backend.training.warehouse_v6_runner import (
    _initialize_public_trees, _teacher_rows,
)
from env.warehouse.pure_nn import AGENTS
from tests.warehouse_v614_test_scopes import isolated_evaluation_scope


def test_public_label_trees_keep_sparse_independent_splits_and_frozen_parity():
    with isolated_evaluation_scope():
        obs, mask, labels, _ = _teacher_rows(4, 992000)
    torch.manual_seed(29)
    model = WarehouseV6Actor()
    fitted = _initialize_public_trees(model, obs, mask, labels)
    role_weights = []
    for role in AGENTS:
        actor = model.actors[role]
        weight = actor.tree.split.weight.detach().numpy()
        assert np.max(np.count_nonzero(weight, axis=-1)) <= MAX_SPLIT_FEATURES
        assert actor.tree.split.weight.requires_grad
        assert actor.tree.split.bias.requires_grad
        assert fitted[role]["training_leaf_occupancy"] >= 2
        role_weights.append(weight)
    assert not np.array_equal(*role_weights)
    with torch.no_grad():
        features = policy_features_torch(obs[:12], mask[:12])
        predicted = model.outputs_encoded(features, mask[:12]).numpy()
    frozen = FrozenWarehouseV6Actor.from_model(model)
    actual = np.where(mask[:12].numpy() > 0.5,
                      frozen.logits(obs[:12].numpy(), mask[:12].numpy()), -1e9)
    np.testing.assert_allclose(predicted, actual, atol=1e-4, rtol=0)
