"""A trained tree must route identically in single, batched, and frozen inference."""

import numpy as np
import torch

from backend.training.warehouse_v6_actor import (
    FEATURE_DIM, FrozenWarehouseV6Actor, WarehouseV6Actor,
)
from env.warehouse.pure_nn import AGENTS


def test_hard_split_tie_is_batch_invariant_and_matches_frozen_actor():
    model = WarehouseV6Actor()
    with torch.no_grad():
        for role in AGENTS:
            tree = model.actors[role].tree
            tree.split.weight.zero_()
            tree.split.bias.fill_(-1.0)
            tree.split.weight[0, 0] = 1.0
            tree.split.bias[0] = 0.0
    frozen = FrozenWarehouseV6Actor.from_model(model)

    rows = np.zeros((513, FEATURE_DIM), dtype=np.float32)
    rows[:, 0] = 5.0e-11  # Numeric threshold tie: route left by contract.
    rows[-1, 0] = 2.0e-10  # A score outside the tie band routes right.
    for role in AGENTS:
        tree = model.actors[role].tree
        with torch.no_grad():
            single = np.concatenate([
                tree.path(torch.from_numpy(row[None])).argmax(-1).numpy()
                for row in rows
            ])
            batched = tree.path(torch.from_numpy(rows)).argmax(-1).numpy()
        deployed = frozen.roles[role].leaf(rows)
        np.testing.assert_array_equal(single, batched)
        np.testing.assert_array_equal(batched, deployed)
        assert deployed[0] != deployed[-1]
