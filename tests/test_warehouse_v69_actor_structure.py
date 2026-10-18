"""Irregular hard-tree topology, sparse splits, and frozen Actor parity."""
from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.training.warehouse_v6_actor import (
    FEATURE_DIM, MODEL_VERSION, TREE_DEPTH, TREE_LEAVES, FrozenWarehouseV6Actor,
    FrozenWarehouseV6Role, WarehouseV6Actor, _role_arrays, export_actor,
)
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v617 import encode_actor_tree_features
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.pure_nn import AGENTS, action_masks


SELF_ROW_INDEX = 45


def _public_state(frame: int, position: tuple[int, int]) -> WarehouseState:
    """Valid study-map pre-action state, built without a scenario reset."""
    return WarehouseState(
        episode_id=69, frame=frame,
        agents=[AgentState("robot_1", position, 70.0),
                AgentState("robot_2", STUDY_MAP_LAYOUT.robot_start_positions[1],
                           70.0)],
        tasks=[DeliveryTask("task_0", (0, 3), (3, 2)),
               DeliveryTask("task_1", (1, 0), (3, 1))], next_task_index=2)


def _public_rows() -> tuple[np.ndarray, np.ndarray]:
    states = (_public_state(10, (1, 0)), _public_state(80, (1, 0)),
              _public_state(80, (5, 2)))
    return (np.stack([encode_public_observation(state) for state in states]),
            np.stack([action_masks(state) for state in states]))


def _classifier() -> SimpleNamespace:
    """Root's left leaf is early; its right child has two leaves."""
    values = np.zeros((5, 1, 3), dtype=np.float64)
    values[1, 0] = [10, 0, 0]
    values[3, 0] = [0, 10, 0]
    values[4, 0] = [0, 0, 10]
    return SimpleNamespace(
        classes_=np.arange(3),
        tree_=SimpleNamespace(
            children_left=np.asarray([1, -1, 3, -1, -1]),
            children_right=np.asarray([2, -1, 4, -1, -1]),
            feature=np.asarray([0, -2, SELF_ROW_INDEX, -2, -2]),
            threshold=np.asarray([0.4, -2, 0.6, -2, -2]),
            value=values,
            weighted_n_node_samples=np.asarray([30, 10, 20, 10, 10]),
        ),
    )


def _irregular_model() -> WarehouseV6Actor:
    torch.manual_seed(69)
    model = WarehouseV6Actor()
    for role in AGENTS:
        model.actors[role].initialize_tree_from_classifier(_classifier())
    return model


def _encoded_rows() -> np.ndarray:
    obs, mask = _public_rows()
    rows = encode_actor_tree_features(obs[:, 0], mask[:, 0])
    assert rows.shape == (3, FEATURE_DIM)
    return rows


def test_irregular_tree_keeps_early_leaves_and_torch_frozen_routes():
    model = _irregular_model()
    rows = _encoded_rows()
    for role in AGENTS:
        tree = model.actors[role].tree
        assert tree.node_count.item() == 2
        assert tree.leaf_count.item() == 3
        np.testing.assert_array_equal(tree.left_refs[:2].numpy(), [-1, -2])
        np.testing.assert_array_equal(tree.right_refs[:2].numpy(), [1, -3])
        with torch.no_grad():
            path = tree.path(torch.from_numpy(rows)).numpy()
        np.testing.assert_array_equal(path.argmax(axis=-1), [0, 1, 2])
        frozen = FrozenWarehouseV6Role(_role_arrays(model.actors[role]))
        np.testing.assert_array_equal(frozen.leaf(rows), [0, 1, 2])
        with torch.no_grad():
            expected = model.actors[role](torch.from_numpy(rows)).numpy()
        np.testing.assert_allclose(frozen.logits_encoded(rows), expected,
                                   atol=1e-4, rtol=0)


def test_visited_split_parameters_receive_gradients_with_hard_forward():
    model = _irregular_model()
    tree = model.actors[AGENTS[0]].tree
    assert tree.split.weight.requires_grad and tree.split.bias.requires_grad
    rows = _encoded_rows()
    rows[0, 0] = 0.4
    rows[1, SELF_ROW_INDEX] = 0.6
    encoded = torch.from_numpy(rows)
    logits, path = tree(encoded)
    np.testing.assert_array_equal(path.detach().numpy().sum(axis=-1), [1, 1, 1])
    np.testing.assert_array_equal(path.detach().numpy().argmax(axis=-1), [0, 1, 2])
    weight_grad, bias_grad = torch.autograd.grad(
        logits[:, 1].sum(), (tree.split.weight, tree.split.bias))
    assert weight_grad[0, 0].abs() > 0
    assert weight_grad[1, SELF_ROW_INDEX].abs() > 0
    assert bias_grad[:2].abs().min() > 0
    assert torch.count_nonzero(weight_grad[2:]) == 0


def test_equal_leaf_logits_cancel_split_gradients():
    model = _irregular_model()
    tree = model.actors[AGENTS[0]].tree
    rows = _encoded_rows()
    rows[0, 0] = 0.4
    rows[1, SELF_ROW_INDEX] = 0.6
    encoded = torch.from_numpy(rows)
    with torch.no_grad():
        tree.leaf_logits[:] = torch.arange(len(ACTIONS), dtype=torch.float32)
    equal_logits, path = tree(encoded)
    np.testing.assert_array_equal(path.detach().numpy().sum(axis=-1), [1, 1, 1])
    equal_weight_grad, equal_bias_grad = torch.autograd.grad(
        equal_logits[:, 0].sum(), (tree.split.weight, tree.split.bias))
    assert torch.count_nonzero(equal_weight_grad) == 0
    assert torch.count_nonzero(equal_bias_grad) == 0

    with torch.no_grad():
        tree.leaf_logits[0, 0] += 1.0
        tree.leaf_logits[1, 0] -= 1.0
        tree.leaf_logits[2, 0] += 2.0
    distinct_logits, path = tree(encoded)
    np.testing.assert_array_equal(path.detach().numpy().argmax(axis=-1), [0, 1, 2])
    weight_grad, bias_grad = torch.autograd.grad(
        distinct_logits[:, 0].sum(), (tree.split.weight, tree.split.bias))
    assert weight_grad[0, 0].abs() > 0
    assert weight_grad[1, SELF_ROW_INDEX].abs() > 0
    assert bias_grad[:2].abs().min() > 0
    assert torch.count_nonzero(weight_grad[2:]) == 0


def test_active_splits_are_sparse_and_frozen_validation_rejects_dense_split():
    model = WarehouseV6Actor()
    role = model.actors[AGENTS[0]]
    with torch.no_grad():
        effective = role.tree.effective_weights().numpy()
    assert np.count_nonzero(effective, axis=-1).max() <= 3
    weights = _role_arrays(role)
    assert np.count_nonzero(weights["tree.split.weight"], axis=-1).max() <= 3
    weights["tree.split.weight"][0, :4] = 1.0
    with pytest.raises(ValueError, match="feature limit"):
        FrozenWarehouseV6Role(weights)


@pytest.mark.parametrize("change, error", [
    (lambda w: w["tree.left_refs"].__setitem__(0, 2), "split invalid"),
    (lambda w: w["tree.left_refs"].__setitem__(0, -4), "split invalid"),
    (lambda w: w["tree.right_refs"].__setitem__(0, 0.5), "split invalid"),
    (lambda w: w["tree.right_refs"].__setitem__(0, -1), "leaf invalid"),
    (lambda w: w["tree.node_count"].__setitem__(0, 1.5), "size mismatch"),
    (lambda w: w["tree.leaf_count"].__setitem__(0, 4), "size mismatch"),
])
def test_frozen_tree_rejects_malformed_topology(change, error):
    weights = _role_arrays(_irregular_model().actors[AGENTS[0]])
    change(weights)
    with pytest.raises(ValueError, match=error):
        FrozenWarehouseV6Role(weights)


def test_tree_depth_limit_accepts_five_and_rejects_six():
    assert TREE_DEPTH == 5 and TREE_LEAVES == 16
    baseline = _role_arrays(_irregular_model().actors[AGENTS[0]])

    def chain(nodes: int) -> dict[str, np.ndarray]:
        weights = {name: value.copy() for name, value in baseline.items()}
        weights["tree.left_refs"].fill(0)
        weights["tree.right_refs"].fill(0)
        weights["tree.node_count"][0] = nodes
        weights["tree.leaf_count"][0] = nodes + 1
        for index in range(nodes):
            weights["tree.left_refs"][index] = -1 - index
            weights["tree.right_refs"][index] = (
                index + 1 if index + 1 < nodes else -1 - nodes)
        return weights

    FrozenWarehouseV6Role(chain(5))
    with pytest.raises(ValueError, match="too deep"):
        FrozenWarehouseV6Role(chain(6))


def test_shallow_and_depth_five_paths_match_frozen_leaves_and_logits():
    model = _irregular_model()
    actor = model.actors[AGENTS[0]]
    tree = actor.tree
    with torch.no_grad():
        tree.split.weight.zero_()
        tree.split.bias.zero_()
        tree.left_refs.zero_()
        tree.right_refs.zero_()
        tree.node_count[0] = 5
        tree.leaf_count[0] = 6
        for index, threshold in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
            tree.split.weight[index, 0] = 1.0
            tree.split.bias[index] = -threshold
            tree.left_refs[index] = -1 - index
            tree.right_refs[index] = index + 1 if index < 4 else -6
            tree.leaf_logits[index, 0] = float(index)
        tree.leaf_logits[5, 0] = 5.0
    states = [_public_state(frame, (1, 0))
              for frame in (6, 18, 30, 42, 54, 72)]
    rows = encode_actor_tree_features(
        np.stack([encode_public_observation(state)[0] for state in states]),
        np.stack([action_masks(state)[0] for state in states]))
    with torch.no_grad():
        path = tree.path(torch.from_numpy(rows)).numpy()
        expected = actor(torch.from_numpy(rows)).numpy()
    frozen = FrozenWarehouseV6Role(_role_arrays(actor))
    np.testing.assert_array_equal(path.argmax(axis=-1), np.arange(6))
    np.testing.assert_array_equal(frozen.leaf(rows), np.arange(6))
    np.testing.assert_allclose(frozen.logits_encoded(rows), expected,
                               atol=1e-4, rtol=0)


def test_single_leaf_topology_and_reused_node_are_checked():
    weights = _role_arrays(_irregular_model().actors[AGENTS[0]])
    weights["tree.node_count"][0] = 0
    weights["tree.leaf_count"][0] = 1
    frozen = FrozenWarehouseV6Role(weights)
    np.testing.assert_array_equal(frozen.leaf(_encoded_rows()), [0, 0, 0])

    weights["tree.node_count"][0] = 2
    weights["tree.leaf_count"][0] = 3
    weights["tree.left_refs"][0] = 1
    weights["tree.right_refs"][0] = 1
    with pytest.raises(ValueError, match="node invalid"):
        FrozenWarehouseV6Role(weights)


def test_export_load_hashes_and_full_torch_frozen_parity(tmp_path):
    model = _irregular_model()
    path = tmp_path / "v69-actor.npz"
    artifact = export_actor(model, path, pair_seed=69,
                            variant="raw", candidate_id="tree-structure-test")
    assert artifact["model_version"] == MODEL_VERSION
    frozen = FrozenWarehouseV6Actor.load(
        path, expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    obs, mask = _public_rows()
    with torch.no_grad():
        expected = model.outputs(torch.from_numpy(obs), torch.from_numpy(mask)).numpy()
    np.testing.assert_allclose(np.where(mask > 0.5, frozen.logits(obs, mask),
                                        -1e9), expected, atol=1e-4, rtol=0)
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        FrozenWarehouseV6Actor.load(path, expected_sha256="0" * 64)

    # A new file digest alone cannot authorize modified topology tensors.
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload[f"{AGENTS[0]}.tree.left_refs"][0] = -4
    corrupted = tmp_path / "corrupt-role.npz"
    np.savez_compressed(corrupted, **payload)
    with pytest.raises(ValueError, match="split invalid"):
        FrozenWarehouseV6Actor.load(
            corrupted, expected_sha256=sha256(corrupted.read_bytes()).hexdigest())

    payload[f"{AGENTS[0]}.tree.left_refs"][0] = -1
    payload[f"{AGENTS[0]}.tree.split.bias"][0] += 1.0
    unchanged_claim = tmp_path / "stale-role-digest.npz"
    np.savez_compressed(unchanged_claim, **payload)
    with pytest.raises(RuntimeError, match="embedded role hashes mismatch"):
        FrozenWarehouseV6Actor.load(
            unchanged_claim, expected_sha256=sha256(unchanged_claim.read_bytes()).hexdigest())


def test_torch_state_dict_preserves_irregular_routes_and_split_trainability(tmp_path):
    model = _irregular_model()
    path = tmp_path / "actor-state.pt"
    torch.save(model.state_dict(), path)
    restored = WarehouseV6Actor()
    restored.load_state_dict(torch.load(path, weights_only=True))
    rows = _encoded_rows()
    for role in AGENTS:
        tree = restored.actors[role].tree
        assert tree.node_count.item() == 2 and tree.leaf_count.item() == 3
        assert tree.split.weight.requires_grad and tree.split.bias.requires_grad
        with torch.no_grad():
            expected = model.actors[role](torch.from_numpy(rows)).numpy()
            actual = restored.actors[role](torch.from_numpy(rows)).numpy()
        np.testing.assert_array_equal(actual, expected)
        frozen = FrozenWarehouseV6Role(_role_arrays(restored.actors[role]))
        np.testing.assert_array_equal(frozen.leaf(rows), [0, 1, 2])
        np.testing.assert_allclose(frozen.logits_encoded(rows), actual,
                                   atol=1e-4, rtol=0)
