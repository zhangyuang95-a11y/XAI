"""Synthetic, no-reset tests for the inactive shared-structure Actor."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from backend.training import warehouse_v621_actor as frozen
from backend.training import warehouse_v621_actor_training as training
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v619_leaf_fit import float32_preference_proof
from backend.training.warehouse_v6_tree_routing import leaf_indices


def structure():
    w = np.zeros((2, frozen.FEATURE_DIM), np.float32)
    w[0, frozen.TREE_MASK_INDICES[3]] = 1
    w[1, frozen.FEATURE_NAMES.index("self_row")] = 1
    return dict(root_ref=0, split_weights=w, split_bias=np.array([-.5, -.5], np.float32),
        left_refs=np.array([-1, -2]), right_refs=np.array([1, -3]), leaf_count=3)


def model(bound=.35):
    result = training.WarehouseV621Actor(core_residual_bound=bound)
    for role in result.actors:
        result.install_structure(role, structure(), [1, 1, 0], [4, 3, -1], np.zeros((3, 5), np.float32))
    return result


def encoded():
    # Build public numeric fixtures directly: no environment construction,
    # reset, rollout, dataset or scientific scenario seed is involved.
    from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
    from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
    from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES
    from env.warehouse.public_action_features_v617 import encode_actor_tree_features
    names = {name: i for i, name in enumerate(OBSERVATION_FEATURE_NAMES)}
    points = list(layout.passable_positions)
    observations, legal_masks = [], []
    for own in points:
        for active in (0., 1.):
            obs = np.zeros(frozen.OBSERVATION_DIM, np.float32)
            for endpoint, point in (("self", own), ("partner", points[-1]),
                    ("charger", layout.charger_position),
                    ("order_0_pickup", points[0]), ("order_0_delivery", points[-1]),
                    ("order_1_pickup", points[1]), ("order_1_delivery", points[-2])):
                obs[names[endpoint+"_row"]] = point[0]/(layout.rows-1)
                obs[names[endpoint+"_column"]] = point[1]/(layout.cols-1)
            obs[names["self_active"]] = active
            obs[names["self_battery_fraction"]] = .8
            for r, row in enumerate(layout.tiles):
                for c, tile in enumerate(row):
                    obs[names[f"wall_row_{r}_column_{c}"]] = tile == "#"
            mask = []
            for action in ACTIONS:
                if action == "WAIT":
                    mask.append(1.)
                else:
                    dr, dc = MOVE_DELTAS[action]
                    mask.append(float(active and (own[0]+dr, own[1]+dc) in points))
            observations.append(obs)
            legal_masks.append(mask)
    masks = np.asarray(legal_masks, np.float32)
    return encode_actor_tree_features(np.asarray(observations), masks), masks


@pytest.mark.parametrize("bound", [.35, .50])
def test_torch_numpy_hard_route_logits_bounds_and_unseen_masks(bound):
    active = model(bound)
    snapshot = frozen.FrozenWarehouseV621Actor.from_model(active)
    x, masks = encoded()
    for name, actor in active.actors.items():
        details = actor.details_encoded(torch.from_numpy(x))
        role = snapshot.roles[name]
        np.testing.assert_array_equal(details["leaf_indices"].numpy(), role.leaf(x))
        np.testing.assert_allclose(details["logits"].detach().numpy(), role.logits_encoded(x), atol=1.e-5, rtol=1.e-5)
        np.testing.assert_allclose(details["tree_logits"].detach().numpy(), role.tree_logits_encoded(x), atol=1.e-6)
        residual = role.effective_residual_encoded(x)
        core = role.weights["core_leaf_mask"][role.leaf(x)] > .5
        assert np.max(np.abs(residual[core])) <= np.float32(bound)
        chosen = role.logits_encoded(x).argmax(-1)
        anchors = role.weights["core_actions"][role.leaf(x)].astype(int)
        np.testing.assert_array_equal(chosen[core], anchors[core])
        assert np.all(masks[np.arange(len(masks))[core], chosen[core]] == 1)
        for i in np.flatnonzero(role.weights["core_leaf_mask"]):
            assert float32_preference_proof(role.weights["tree.leaf_logits"][i],
                int(role.weights["core_actions"][i]), role.certified_actions()[i])["passed"]


def test_core_membership_anchor_and_second_install_cannot_change():
    active = model()
    actor = active.actors["robot_1"]
    with pytest.raises(RuntimeError, match="only be installed once"):
        actor.install_structure(structure(), [0, 0, 0], [-1, -1, -1], np.zeros((3, 5)))
    with torch.no_grad():
        actor.tree.core_leaf_mask[0] = 0
    with pytest.raises(RuntimeError, match="contract changed"):
        actor(torch.from_numpy(encoded()[0]))


def test_noncore_has_unbounded_full_residual_and_core_has_no_escape_gate():
    active = model()
    actor = active.actors["robot_1"]
    with torch.no_grad():
        actor.action_residual[-1].weight.zero_()
        actor.action_residual[-1].bias.fill_(12)
    x, _ = encoded()
    result = actor.details_encoded(torch.from_numpy(x))
    core = result["core_membership"].detach().numpy() > .5
    residual = result["residual"].detach().numpy()
    assert np.max(np.abs(residual[core])) <= .35
    assert np.all(residual[~core] == 12)
    assert not hasattr(actor, "gate")


def test_guard_forward_is_immutable_even_if_optimizer_momentum_changes_raw_guard():
    active = model()
    actor = active.actors["robot_1"]
    x, _ = encoded()
    before = actor.tree.effective_weights()[0].detach().clone()
    bias = actor.tree.effective_bias()[0].detach().clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
    output = actor(torch.from_numpy(x))
    output[:, 0].sum().backward()
    assert torch.count_nonzero(actor.tree.split.weight.grad[0]) == 0
    assert actor.tree.split.bias.grad[0] == 0
    # Simulate stale momentum; effective forward still uses immutable buffers.
    optimizer.state[actor.tree.split.weight] = {"step": torch.tensor(1.),
        "exp_avg": torch.ones_like(actor.tree.split.weight),
        "exp_avg_sq": torch.ones_like(actor.tree.split.weight)}
    optimizer.step()
    assert torch.equal(actor.tree.effective_weights()[0], before)
    assert torch.equal(actor.tree.effective_bias()[0], bias)


def test_75_percent_freeze_and_checkpoint_restore_preserve_core_tree_and_bound():
    active = model(.50)
    active.freeze_structure(150000)
    x, _ = encoded()
    before = frozen.FrozenWarehouseV621Actor.from_model(active)
    checkpoint = deepcopy(active.state_dict())
    # Different constructor bound must restore the checkpoint's registered b.
    restored = training.WarehouseV621Actor(core_residual_bound=.35)
    restored.load_state_dict(checkpoint)
    after = frozen.FrozenWarehouseV621Actor.from_model(restored)
    assert before.role_sha256 == after.role_sha256
    assert restored.actors["robot_1"].tree.core_residual_bound == .50
    for r in active.actors:
        tree = restored.actors[r].tree
        assert not tree.split.weight.requires_grad
        assert not tree.split.bias.requires_grad
        assert int(tree.topology_frozen_at_step[0]) == 150000
        with pytest.raises(RuntimeError, match="cannot be unfrozen"):
            tree.set_topology_trainable(True)
        with pytest.raises(RuntimeError, match="cannot refresh"):
            tree.refresh_sparse_support()
        core_before = tree.effective_leaf_logits().detach().clone()[tree.core_leaf_mask.bool()]
        with torch.no_grad():
            tree.leaf_logits.add_(100)
            tree.extra_margin.add_(20)
        assert torch.equal(core_before, tree.effective_leaf_logits()[tree.core_leaf_mask.bool()])
        np.testing.assert_array_equal(before.roles[r].leaf(x), after.roles[r].leaf(x))


def test_shared_actor_keeps_independent_role_parameters_and_gradients():
    active = model()
    x, masks = encoded()
    joint = torch.from_numpy(np.stack((x, x), 1))
    legal = torch.from_numpy(np.stack((masks, masks), 1))
    before = training.role_hashes(active)
    output = active.outputs_encoded(joint, legal)
    output[:, 0, 4].sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in active.actors["robot_1"].parameters())
    assert all(p.grad is None or not torch.count_nonzero(p.grad) for p in active.actors["robot_2"].parameters())
    assert before["robot_1"] != before["robot_2"]
    assert active.actors["robot_1"].action_residual[0].weight.data_ptr() != active.actors["robot_2"].action_residual[0].weight.data_ptr()


def test_soft_alignment_has_gradient_to_complete_actor_not_only_tree():
    active = model()
    actor = active.actors["robot_1"]
    x, masks = encoded()
    details = actor.details_encoded(torch.from_numpy(x))
    fill = torch.finfo(torch.float32).min
    tree = details["tree_logits"].masked_fill(torch.from_numpy(masks) == 0, fill)
    full = details["logits"].masked_fill(torch.from_numpy(masks) == 0, fill)
    q = torch.softmax(tree.detach(), -1)
    loss = (torch.xlogy(q, q)-q*torch.log_softmax(full, -1)).sum(-1).mean()
    loss.backward()
    assert sum(float(p.grad.square().sum()) for p in actor.action_residual.parameters() if p.grad is not None) > 0


def test_export_load_strict_hash_metadata_dtype_padding_and_core_path(tmp_path):
    active = model()
    path = tmp_path/"actor.npz"
    receipt = training.export_actor(active, path, pair_seed=0, variant="raw", candidate_id="synthetic")
    snapshot = frozen.FrozenWarehouseV621Actor.load(path, expected_sha256=receipt["sha256"],
        expected_role_sha256=receipt["role_sha256"])
    assert snapshot.model_version == frozen.MODEL_VERSION
    assert snapshot.act.__func__ is frozen.FrozenWarehouseV621Actor.act
    assert snapshot.act.__func__ is not FrozenWarehouseV6Actor.act
    with pytest.raises(FileExistsError):
        training.export_actor(active, path, pair_seed=0, variant="raw", candidate_id="synthetic")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        frozen.FrozenWarehouseV621Actor.load(path, expected_sha256="0"*64)
    with pytest.raises(RuntimeError, match="role hashes"):
        frozen.FrozenWarehouseV621Actor.load(path, expected_sha256=receipt["sha256"],
            expected_role_sha256={"robot_1": "0"*64, "robot_2": "0"*64})
    values = training.role_arrays(active.actors["robot_1"])
    bad = deepcopy(values)
    bad["core_actions"][1] = 2  # LEFT has no guard on that core path.
    with pytest.raises(ValueError, match="path or margin"):
        frozen.FrozenWarehouseV621Role(bad)
    bad = deepcopy(values)
    bad["tree.split.weight"] = bad["tree.split.weight"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        frozen.FrozenWarehouseV621Role(bad)
    bad = deepcopy(values)
    bad["tree.split.bias"][-1] = 1
    with pytest.raises(ValueError, match="unused"):
        frozen.FrozenWarehouseV621Role(bad)
    with pytest.raises(ValueError):
        snapshot.roles["robot_1"].weights["core_leaf_mask"][0] = 0


def test_reject_uninstalled_illegal_structure_source_cost_and_new_version_masquerade():
    active = training.WarehouseV621Actor()
    with pytest.raises(RuntimeError, match="installed"):
        frozen.FrozenWarehouseV621Actor.from_model(active)
    s = structure()
    s["split_weights"][1, :4] = 1
    with pytest.raises(ValueError, match="feature"):
        active.install_structure("robot_1", s, [1, 1, 0], [4, 3, -1], np.zeros((3, 5)))
    for bound in (.08, .36, True):
        with pytest.raises(ValueError):
            training.WarehouseV621Actor(core_residual_bound=bound)


def test_bound_implies_both_direction_KL_not_expected_feedback_probability():
    active = model()
    role = frozen.FrozenWarehouseV621Actor.from_model(active).roles["robot_1"]
    x, masks = encoded()
    core = role.weights["core_leaf_mask"][role.leaf(x)].astype(bool)
    tree, full = role.tree_logits_encoded(x)[core], role.logits_encoded(x)[core]
    legal = masks[core]
    # Deterministic synthetic perturbations in the complete centered box.
    e = np.linspace(-.05, .05, 5)[None]
    program = tree+e
    def probs(z):
        z = np.where(legal > 0, np.asarray(z, np.float64), -np.inf)
        z -= np.max(z, -1, keepdims=True)
        p = np.exp(z)
        return p/p.sum(-1, keepdims=True)
    p, q = probs(full), probs(program)
    kl = np.sum(p*(np.log(np.maximum(p, 1.e-300))-np.log(np.maximum(q, 1.e-300))), -1)
    reverse = np.sum(q*(np.log(np.maximum(q, 1.e-300))-np.log(np.maximum(p, 1.e-300))), -1)
    assert np.max(kl) <= .08+1.e-6 and np.max(reverse) <= .08+1.e-6
    assert 1/(1+4*np.exp(-(frozen.required_core_margin(.35)-2*.35))) < .30


def test_paired_primary_copy_is_equal_but_never_shared():
    first, second = model(), model()
    a, b = first.actors["robot_1"], second.actors["robot_1"]
    report = b.load_primary_state(a.action_residual.state_dict())
    assert report["independent_parameter_copy"]
    for left, right in zip(a.action_residual.parameters(), b.action_residual.parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()


def test_frozen_topology_uses_snapshot_despite_stale_raw_parameters_and_rejects_flag_escape():
    active = model()
    active.freeze_structure(150000)
    actor = active.actors["robot_1"]
    before = training.role_arrays(actor)
    with torch.no_grad():
        actor.tree.split.weight.add_(100)
        actor.tree.split.bias.add_(100)
    after = training.role_arrays(actor)
    np.testing.assert_array_equal(before["tree.split.weight"], after["tree.split.weight"])
    np.testing.assert_array_equal(before["tree.split.bias"], after["tree.split.bias"])
    with torch.no_grad():
        actor.tree.structure_frozen[0] = False
    with pytest.raises(RuntimeError, match="flag was cleared"):
        actor.tree.assert_static_contract()


def test_frozen_numpy_import_does_not_import_torch():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c",
        "import sys; import backend.training.warehouse_v621_actor; assert 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_joint_forward_rejects_physical_mask_mismatch():
    active = model()
    x, masks = encoded()
    joint = torch.from_numpy(np.stack((x, x), 1))
    legal = torch.from_numpy(np.stack((masks, masks), 1))
    legal[..., 0] = 1-legal[..., 0]
    with pytest.raises(ValueError, match="physical mask"):
        active.outputs_encoded(joint, legal)


def test_extreme_finite_logits_masked_probabilities_are_exactly_legal_and_double():
    raw = np.array([[-1.e12, -1.e12, -1.e12, -1.e12, -1.e12]], np.float32)
    masks = np.array([[0, 1, 0, 1, 1]], np.float32)
    probs = frozen.masked_probabilities(raw, masks)
    assert probs.dtype == np.float64
    np.testing.assert_array_equal(probs[masks == 0], 0.)
    np.testing.assert_array_equal(probs[masks == 1], np.full(3, 1/3))
    logits = training.masked_policy_logits(torch.from_numpy(raw), torch.from_numpy(masks))
    assert torch.all(torch.isfinite(logits))
    torch.testing.assert_close(logits[0, [1, 3, 4]], torch.zeros(3))
    assert masks[0, int(logits.argmax(-1))] == 1
    np.testing.assert_allclose(torch.softmax(logits, -1).numpy(), probs, atol=2e-8)
    for illegal in (np.full((1, 5), np.nan), np.full((1, 5), np.inf)):
        with pytest.raises(ValueError):
            frozen.masked_probabilities(illegal, masks)
        with pytest.raises(ValueError):
            training.masked_policy_logits(torch.from_numpy(illegal), torch.from_numpy(masks))


def test_new_act_exact_independent_rng_reconstruction_for_negative_unbounded_NN(monkeypatch):
    from types import SimpleNamespace
    from env.warehouse.decision_protocol import independent_agent_seed
    active = model()
    for actor in active.actors.values():
        with torch.no_grad():
            actor.action_residual[-1].weight.zero_()
            actor.action_residual[-1].bias.fill_(-1.e12)
    snapshot = frozen.FrozenWarehouseV621Actor.from_model(active)
    x, masks = encoded()
    candidate = np.flatnonzero(snapshot.roles["robot_1"].leaf(x) == 2)[0]
    obs = np.stack((x[candidate, :frozen.OBSERVATION_DIM],)*2)
    mask = np.stack((masks[candidate],)*2)
    monkeypatch.setattr(frozen, "encode_public_observation", lambda state: obs)
    monkeypatch.setattr(frozen, "action_masks", lambda state: mask)
    state = SimpleNamespace(episode_id=10, frame=2)
    chosen, distributions = snapshot.act(state, base_seed=0)
    expected = frozen.masked_probabilities(snapshot.logits(obs, mask), mask)
    for i, role in enumerate(("robot_1", "robot_2")):
        seed = independent_agent_seed(base_seed=0, episode_id=10, frame=2, agent_id=role)
        index = int(np.random.default_rng(seed).choice(5, p=expected[i]))
        assert chosen[role] == frozen.ACTIONS[index]
        assert mask[i, index] == 1
        np.testing.assert_array_equal(distributions[role].probabilities, expected[i])
        np.testing.assert_array_equal(expected[i][mask[i] == 0], 0.)


def test_numpy_and_torch_overflow_from_finite_assets_fail_closed():
    active = model()
    with torch.no_grad():
        for actor in active.actors.values():
            actor.action_residual[0].weight.fill_(torch.finfo(torch.float32).max)
    snapshot = frozen.FrozenWarehouseV621Actor.from_model(active)
    x, _ = encoded()
    with pytest.raises(RuntimeError, match="overflowed"):
        active.actors["robot_1"](torch.from_numpy(x))
    with pytest.raises(RuntimeError, match="nonfinite"):
        snapshot.roles["robot_1"].logits_encoded(x)


def test_normal_weights_probability_stabilization_preserves_historical_distribution():
    from backend.training.warehouse_v619_tree import masked_probabilities as historical
    active = model()
    role = frozen.FrozenWarehouseV621Actor.from_model(active).roles["robot_1"]
    x, masks = encoded()
    raw = role.logits_encoded(x)
    current = frozen.masked_probabilities(raw, masks)
    np.testing.assert_allclose(current, historical(raw, masks), atol=1.e-14, rtol=1.e-13)
    old_runtime = np.where(masks > .5, raw, np.float32(-1.e9))
    old_runtime = np.exp(old_runtime-old_runtime.max(-1, keepdims=True))
    old_runtime /= old_runtime.sum(-1, keepdims=True)
    np.testing.assert_allclose(current, old_runtime, atol=2.e-7, rtol=2.e-7)
