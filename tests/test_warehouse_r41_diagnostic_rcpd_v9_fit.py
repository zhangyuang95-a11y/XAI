from __future__ import annotations

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training import warehouse_r41_diagnostic_rcpd_v9_fit as subject
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from env.warehouse_native.policy import NumPyNativeActor


ACTOR = (
    "output/warehouse_native/r41_active_2m_20260911/"
    "boundaries/step_2000000/actor.npz"
)


def config(pair_mass_fraction=0.35):
    return {
        "version": subject.CONFIG_VERSION,
        "pair_mass_fraction": pair_mass_fraction,
        "duplicate_power": 0.5,
        "balance_power": 0.5,
        "maximum_row_weight": 30.0,
        "model": {
            "learning_rate": 0.1,
            "max_iter": 2,
            "max_leaf_nodes": 4,
            "min_samples_leaf": 5,
            "l2_regularization": 0.1,
            "max_depth": 3,
            "max_bins": 32,
            "random_state": 1941,
        },
    }


def rows():
    actor = NumPyNativeActor(ACTOR)
    count = 30
    rng = np.random.default_rng(41)
    observations = rng.normal(size=(count, actor.obs_dim)).astype(np.float32)
    # Keep map and coordinate fields public-valid for the v9 relation mapper.
    names = actor.metadata["feature_names"]
    index = {name: i for i, name in enumerate(names)}
    for row in range(6):
        for column in range(7):
            observations[:, index[f"map.{row}.{column}.passable"]] = 1.0
    for name in ("self.row", "other.row"):
        observations[:, index[name]] = rng.integers(0, 6, count) / 5
    for name in ("self.column", "other.column"):
        observations[:, index[name]] = rng.integers(0, 7, count) / 6
    for task in range(2):
        for endpoint in ("pickup", "delivery"):
            observations[:, index[f"task.{task}.{endpoint}_row"]] = rng.integers(0, 6, count) / 5
            observations[:, index[f"task.{task}.{endpoint}_column"]] = rng.integers(0, 7, count) / 6
    scenes = np.asarray(["a" * 64] * 15 + ["b" * 64] * 15, dtype="S64")
    actions = np.arange(count, dtype=np.uint8) % 5
    kinds = np.full(count, "ordinary", dtype="S16")
    anchors = np.full(count, "", dtype="S240")
    branches = np.full(count, "", dtype="S8")
    physical = np.asarray([f"{1000+i:064x}" for i in range(count)], dtype="S64")
    # Two complete intervention anchors, with WAIT and changed Actor actions.
    for base, anchor in ((0, "anchor-a"), (15, "anchor-b")):
        for offset, branch in enumerate(("WAIT", "UP", "DOWN", "LEFT", "RIGHT")):
            row = base + offset
            kinds[row] = b"intervention"
            anchors[row] = anchor.encode()
            branches[row] = branch.encode()
        actions[base:base + 5] = np.asarray((4, 0, 1, 2, 3), dtype=np.uint8)
    arrays = {
        "observations": observations,
        "probabilities": np.eye(5, dtype=np.float32)[actions],
        "action_indices": actions,
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": np.asarray(
            [f"{2000 + (i // 2):064x}" for i in range(count)], dtype="S64"),
        "scene_fingerprints": scenes,
        "episode_ids": np.asarray([
            "episode-a:skilled" if i < 15 else "episode-b:assertive"
            for i in range(count)
        ], dtype="S180"),
        "frames": np.arange(count, dtype=np.int16),
        "group_bits": np.asarray([1 + (i % 7) for i in range(count)], dtype=np.uint8),
        "kinds": kinds,
        "anchor_ids": anchors,
        "branch_actions": branches,
        "physical_hashes": physical,
        "source_state_hashes": np.asarray(
            [f"{3000+i:064x}" for i in range(count)], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.zeros(count, dtype=np.bool_),
        "split_validation": np.zeros(count, dtype=np.bool_),
    }
    return actor, arrays


def test_weights_are_fit_only_finite_and_pair_stratified():
    _, arrays = rows()
    mask = np.arange(30) < 15
    weights, pairs, bits, audit = subject.build_fit_weights(
        arrays, mask,
        scene_families={"a" * 64: "conflict_family_01",
                        "b" * 64: "conflict_family_02"},
        config=config(),
    )
    assert len(pairs) == 4
    assert bits.shape == (4,)
    assert np.all(np.isfinite(weights[mask])) and np.all(weights[mask] > 0)
    assert np.all(weights[~mask] == 0)
    assert audit["effective_pair_count"] == 4
    assert audit["requested_pair_mass"] == pytest.approx(15 * 0.35)
    assert audit["validation_labels_used"] is False


def test_fit_exports_one_traceable_public_program_without_runtime_control():
    actor, arrays = rows()
    relations = R41DiagnosticPublicRelationsV9(actor.metadata["feature_names"])
    program, diagnostics = subject.fit_program(
        arrays, np.ones(30, dtype=np.bool_), relations=relations,
        scene_families={"a" * 64: "conflict_family_01",
                        "b" * 64: "conflict_family_02"},
        config=config(), binding_sha256="f" * 64,
    )
    probabilities = program.predict_proba_batch(arrays["observations"])
    assert probabilities.shape == (30, 5)
    np.testing.assert_allclose(probabilities.sum(1), 1.0, rtol=0, atol=2e-12)
    assert diagnostics["sklearn_explicit_actions_equal"] is True
    payload = program.to_dict()
    assert payload["metadata"]["runtime_action_override"] is False
    assert all(item["mix_weight"] == 0.0 for item in payload["specialists"])
    used = " ".join(program.complexity()["unique_split_feature_names"]).casefold()
    assert "logit" not in used and "hidden" not in used


def test_config_and_masks_are_strict():
    _, arrays = rows()
    bad = config()
    bad["model"]["max_leaf_nodes"] = 999
    with pytest.raises(ValueError):
        subject.normalize_config(bad)
    with pytest.raises(ValueError):
        subject.build_fit_weights(
            arrays, np.ones(29, dtype=np.bool_),
            scene_families={"a" * 64: "conflict_family_01"},
            config=config())
