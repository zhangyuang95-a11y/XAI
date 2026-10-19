"""Synthetic pipeline integration; no environment, rollout, or authorization."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.training import warehouse_v621_bc_pipeline as pipeline
from backend.training import warehouse_v621_actor_training as training
from backend.training import warehouse_v621_actor as frozen
from env.warehouse.layouts import STUDY_MAP_LAYOUT as LAYOUT
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES, OBSERVATION_DIM


def teacher_fixture():
    """XOR public labels force eight classifier leaves with no env/reset."""
    own = next(point for point in LAYOUT.passable_positions if all(
        LAYOUT.is_passable((point[0]+dr, point[1]+dc))
        for dr, dc in (MOVE_DELTAS["UP"], MOVE_DELTAS["RIGHT"])))
    index = {name: i for i, name in enumerate(OBSERVATION_FEATURE_NAMES)}
    rows, labels, episodes = [], [], []
    for episode in range(10):
        for bits in range(8):
            for _ in range(8):
                obs = np.zeros(OBSERVATION_DIM, np.float32)
                for row, tiles in enumerate(LAYOUT.tiles):
                    for col, tile in enumerate(tiles):
                        obs[index[f"wall_row_{row}_column_{col}"]] = tile == "#"
                for name, point in {"self": own, "partner": LAYOUT.passable_positions[-1],
                    "charger": LAYOUT.charger_position,
                    "order_0_pickup": LAYOUT.passable_positions[0],
                    "order_0_delivery": LAYOUT.passable_positions[1],
                    "order_1_pickup": LAYOUT.passable_positions[-2],
                    "order_1_delivery": LAYOUT.passable_positions[-1]}.items():
                    obs[index[name+"_row"]] = point[0]/(LAYOUT.rows-1)
                    obs[index[name+"_column"]] = point[1]/(LAYOUT.cols-1)
                obs[index["self_active"]] = obs[index["partner_active"]] = 1
                obs[index["self_battery_fraction"]] = obs[index["partner_battery_fraction"]] = .8
                for bit, name in enumerate(("elapsed_fraction", "self_steps_since_charging", "partner_steps_since_charging")):
                    obs[index[name]] = (bits >> bit) & 1
                rows.append(np.stack((obs, obs)))
                action = 0 if bits.bit_count() % 2 == 0 else 3
                labels.append([action, action])
                episodes.append(f"synthetic-public-{episode}")
    mask = np.asarray([float(LAYOUT.is_passable((own[0]+MOVE_DELTAS[a][0], own[1]+MOVE_DELTAS[a][1])))
        if a in MOVE_DELTAS else 1. for a in ACTIONS], np.float32)
    masks = np.broadcast_to(mask, (len(rows), 2, 5)).copy()
    return (torch.from_numpy(np.asarray(rows)), torch.from_numpy(masks),
        torch.tensor(labels, dtype=torch.long), torch.zeros(len(rows)), tuple(episodes),
        tuple([0]*len(rows)))


def shared_model():
    model = training.WarehouseV621Actor()
    rows = teacher_fixture()
    report = pipeline.install_public_topologies(model, rows, shared=True)
    return model, rows, report


def test_ordinary_calls_cannot_construct_reset_update_or_export_without_capability():
    with pytest.raises(RuntimeError, match="authority"):
        pipeline.create_model(SimpleNamespace(recipe="shared_b035", initialization=0))
    with pytest.raises(RuntimeError, match="authority"):
        pipeline.environment(None, "synthetic", 0, 0)
    with pytest.raises(RuntimeError, match="authority"):
        pipeline.bc_epochs(None, None, [], epochs=1, generator=None, lr=.01, batch_size=1)
    with pytest.raises(RuntimeError, match="authority"):
        pipeline.train_common_start(None)


def test_control_main_backup_copy_equal_primary_with_independent_storage(monkeypatch):
    monkeypatch.setattr(pipeline, "require", lambda *args: None)
    models, identities = [], []
    # Synthetic initializer zero is outside registered scientific entries.
    for recipe in ("additive_control", "shared_b035", "shared_b050"):
        model, identity = pipeline.create_model(SimpleNamespace(recipe=recipe, initialization=0))
        models.append(model)
        identities.append(identity)
    assert identities[0] == identities[1] == identities[2]
    for role in pipeline.AGENTS:
        for tensors in zip(*(m.actors[role].action_residual.parameters() for m in models), strict=True):
            assert all(torch.equal(tensors[0], item) for item in tensors[1:])
            assert len({item.data_ptr() for item in tensors}) == 3


def test_public_eight_leaf_initialization_charges_guards_and_has_static_supported_core():
    model, rows, reports = shared_model()
    for role in pipeline.AGENTS:
        detail = reports[role]
        # Three XOR decisions produce eight leaves; each moving leaf adds its
        # complete guard, including the empty illegal branch, to the budget.
        assert detail["compiler"]["original_capacity"]["leaf_count"] == 8
        assert model.actors[role].tree.leaf_count.item() == 16
        snapshot = frozen.FrozenWarehouseV621Actor.from_model(model).roles[role]
        assert len(snapshot.certified_actions()) == 16
        assert snapshot.weights["core_leaf_mask"].sum() == 8
        assert detail["training_core_coverage"] == 1.
        for leaf in detail["leaves"]:
            if leaf["core"]:
                assert leaf["training_rows"] >= 80
                assert leaf["training_episodes"] == 10
                assert leaf["teacher_agreement"] == 1
        assert model.actors[role].tree.core_leaf_mask.sum() > 0
        assert detail["purpose"] == "public_teacher_initialization_before_joint_bc_not_external_program"
    assert rows[0].shape[0] == 640


def test_public_initializer_rejects_missing_episode_support_instead_of_dropping_constraint():
    model = training.WarehouseV621Actor()
    rows = list(teacher_fixture())
    rows[4] = tuple(["one-synthetic-episode"]*len(rows[0]))
    with pytest.raises(RuntimeError, match="no supported static core"):
        pipeline.install_public_topologies(model, rows, shared=True)


def test_shared_loss_both_KL_directions_have_the_declared_real_gradients():
    model, rows, _ = shared_model()
    obs, masks, labels = rows[:3]
    encoded = pipeline.policy_features_torch(obs[:128], masks[:128])
    loss, full, components = pipeline.shared_bc_loss(model, encoded, masks[:128], labels[:128], torch.ones((128, 2)))
    primary = tuple(model.actors["robot_1"].action_residual.parameters())
    tree = tuple(p for p in model.actors["robot_1"].tree.parameters() if p.requires_grad)
    primary_grad = torch.autograd.grad(components["tree_to_full_kl"], primary,
        allow_unused=True, retain_graph=True)
    assert sum(float(g.square().sum()) for g in primary_grad if g is not None) > 0
    tree_grad = torch.autograd.grad(components["full_to_tree_kl"], tree,
        allow_unused=True, retain_graph=True)
    assert sum(float(g.square().sum()) for g in tree_grad if g is not None) > 0
    detached_direction = torch.autograd.grad(components["full_to_tree_kl"], primary,
        allow_unused=True, retain_graph=True)
    assert all(g is None or not torch.count_nonzero(g) for g in detached_direction)
    expected = (components["full_task"]+.2*components["tree_task"]
        +.2*components["tree_to_full_kl"]+.2*components["full_to_tree_kl"]
        +.05*components["core_occupancy_penalty"])
    torch.testing.assert_close(loss, expected)
    assert full.shape == (128, 2, 5)


def test_shared_and_frozen_full_forward_same_physical_mask_without_runtime_program():
    model, rows, _ = shared_model()
    obs, masks = rows[:2]
    encoded = pipeline.policy_features_torch(obs[:16], masks[:16])
    with torch.no_grad():
        actual = model.outputs_encoded(encoded, masks[:16]).numpy()
    snapshot = pipeline.frozen(model)
    expected = frozen.masked_policy_logits(
        snapshot.logits(obs[:16].numpy(), masks[:16].numpy()), masks[:16].numpy())
    np.testing.assert_allclose(actual, expected, atol=3.e-5, rtol=3.e-5)
    assert snapshot.model_version == frozen.MODEL_VERSION


def test_bc_configuration_matches_implemented_losses_and_capacity():
    from experiments.warehouse import v621_bc_protocol
    recipe = v621_bc_protocol.recipe()
    assert recipe["initial_shared_topology"]["maximum_classifier_leaves"] == 8
    assert recipe["initial_shared_topology"]["maximum_classifier_depth"] == 4
    assert recipe["shared_losses"] == {"full_task_ce": 1., "tree_task_ce": .2,
        "tree_to_full_kl": .2, "full_to_tree_kl": .2,
        "training_core_occupancy_weight": .05, "training_core_occupancy_target": .5,
        "condition_on_current_teacher_correctness": False}
    assert recipe["external_program_actor_updates_during_bc"] == 0
    assert recipe["PPO_joint_steps"] == 0 and not recipe["pilot_opened"]


def test_training_update_proof_ignores_topology_installation_only(monkeypatch, tmp_path):
    """Regression: installation is not evidence either NN received BC updates."""
    rows = teacher_fixture()
    plan = SimpleNamespace(recipe="shared_b035", initialization=0,
        source_closure_sha256="a"*64, seed_plan_sha256="b"*64, output_directory=tmp_path)
    monkeypatch.setattr(pipeline, "require", lambda *args: None)
    assigned = {"rng": {"initial_bc_and_dagger_torch_shuffle": 0,
        "risk_torch_shuffle": 0, "large_torch_shuffle": 0,
        "deep_torch_shuffle": [0]*4, "final_torch_shuffle": 0},
        "stages": {name: [] for name, _ in pipeline.protocol.STAGES}}
    monkeypatch.setattr(pipeline.protocol, "seed_plan", lambda: {"initializations": {"0": assigned}})
    monkeypatch.setattr(pipeline, "load_shared_cache", lambda _: (
        {name: rows for name in pipeline.protocol.TEACHER_STAGES}, None, {}))
    monkeypatch.setattr(pipeline, "nn_visits", lambda *args, **kwargs: (rows, {"synthetic": True}))
    def no_updates(*args, **kwargs):
        return {"epoch_reports": [{"row_weighted_mean_losses": {"synthetic_no_op": 0}}],
            "role_maximum_bc_gradient_norm": dict.fromkeys(pipeline.AGENTS, 1.)}
    monkeypatch.setattr(pipeline, "bc_epochs", no_updates)
    with pytest.raises(RuntimeError, match="(no BC|no .*parameter updates|received no)"):
        pipeline.train_common_start(plan)


def test_one_synthetic_bc_epoch_updates_both_roles_retains_all_frames_and_guard_parity(monkeypatch):
    model, rows, _ = shared_model()
    monkeypatch.setattr(pipeline, "require", lambda *args: None)
    monkeypatch.setattr(pipeline, "environment", lambda *args, **kwargs: pytest.fail("no reset in tensor test"))
    before = training.role_hashes(model)
    report = pipeline.bc_epochs(SimpleNamespace(), model, [rows], epochs=1,
        generator=torch.Generator().manual_seed(0), lr=.001, batch_size=128)
    after = training.role_hashes(model)
    assert all(before[role] != after[role] for role in pipeline.AGENTS)
    assert all(report["role_maximum_bc_gradient_norm"][role] > 0 for role in pipeline.AGENTS)
    assert report["all_physical_training_frames_retained"]
    assert report["epoch_reports"][0]["all_epoch_rows"] == len(rows[0])
    assert report["external_program_feedback"] == report["external_program_gradient"] == 0
    assert model.actors["robot_1"].tree.core_leaf_mask.sum() == 8
