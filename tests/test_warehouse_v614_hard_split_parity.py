"""A trained tree must route identically in single, batched, and frozen inference."""

from hashlib import sha256
import json

import numpy as np
import torch
import pytest

from backend.training.warehouse_v6_actor import (
    FEATURE_DIM, FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor,
)
from backend.training.warehouse_v614_certificate_tree import (
    FeatureSchema, SparseProgramTree, to_current_core_tree_result,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_action_features_v617 import TREE_FEATURE_NAMES


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
        weights = frozen.roles[role].weights
        certificate = SparseProgramTree(
            FeatureSchema(tuple(TREE_FEATURE_NAMES), tuple(ACTIONS)),
            weights["tree.split.weight"], weights["tree.split.bias"],
            weights["tree.left_refs"].astype(np.int64),
            weights["tree.right_refs"].astype(np.int64),
            int(weights["tree.node_count"][0]),
            int(weights["tree.leaf_count"][0]),
            weights["tree.leaf_logits"],
            # This generic route test uses the current Actor. Qualification
            # sidecars are not part of its exported neural policy tensors.
            np.zeros(16, dtype=np.bool_),
            {}, frozenset(),
        )
        np.testing.assert_array_equal(certificate.leaf_indices(rows), deployed)
        core = to_current_core_tree_result(certificate)
        np.testing.assert_array_equal(core.leaf_indices(rows), deployed)


def test_pre_routing_contract_artifact_is_rejected(tmp_path):
    source = tmp_path / "current.npz"
    export_actor(WarehouseV6Actor(), source, pair_seed=1,
                 variant="raw", candidate_id="routing-contract-test")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.pop("hard_split_routing_version")
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    old = tmp_path / "pre-routing-contract.npz"
    np.savez_compressed(old, **arrays)
    with pytest.raises(RuntimeError, match="routing contract mismatch"):
        FrozenWarehouseV6Actor.load(
            old, expected_sha256=sha256(old.read_bytes()).hexdigest())
