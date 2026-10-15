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


def config(pair_pool_multiplier=16.0, wait_endpoint_share=0.62, *,
           replacement=False, minimum_rows=1, minimum_scenes=1,
           minimum_episodes=1):
    return {
        "version": subject.CONFIG_VERSION,
        "pair_pool_multiplier": pair_pool_multiplier,
        "wait_endpoint_share": wait_endpoint_share,
        "shared_pickup_replacement": {
            "version": subject.REPLACEMENT_VERSION,
            "enabled": replacement,
            "group": "shared_pickup",
            "combination": "replacement",
            "route": {
                "feature_name": subject.REPLACEMENT_ROUTE_FEATURE,
                "operator": ">", "threshold": 0.5,
            },
            "fit_source": "fit_only_public_route_rows",
            "estimator": "fit_only_weighted_majority",
            "minimum_rows_per_partition": minimum_rows,
            "minimum_scenes_per_partition": minimum_scenes,
            "minimum_episodes_per_partition": minimum_episodes,
        },
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


def activate_replacement_route(actor, arrays, selected):
    observations = arrays["observations"].copy()
    index = {name: i for i, name in enumerate(actor.metadata["feature_names"])}
    for task in range(2):
        observations[:, index[f"task.{task}.available"]] = 0.0
        observations[:, index[f"task.{task}.carried_other"]] = 0.0
    observations[:, index["history.valid"]] = 0.0
    observations[:, index["history.self.move_canceled"]] = 0.0
    observations[:, index["history.other.submitted.WAIT"]] = 0.0
    for row in selected:
        observations[row, index["task.0.exists"]] = 1.0
        observations[row, index["task.0.available"]] = 1.0
        observations[row, index["task.0.carried_other"]] = 1.0
        observations[row, index["task.0.delivery.self.path_distance"]] = 0.0
        observations[row, index["task.0.pickup.self.path_distance"]] = 0.0
        observations[row, index["task.0.pickup.other.path_distance"]] = 0.0
        observations[row, index["history.valid"]] = 1.0
        observations[row, index["history.self.move_canceled"]] = 1.0
        observations[row, index["history.other.submitted.WAIT"]] = 1.0
        observations[row, index["other.battery"]] = .8
    result = dict(arrays)
    result["observations"] = observations
    return result


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
    assert audit["pair_mass_after_redistribution"] == pytest.approx(
        audit["pair_mass_before_redistribution"])
    assert audit["pair_mass_preserved"] is True
    assert audit["validation_labels_used"] is False


def test_held_fold_label_values_are_never_read_or_validated():
    _, arrays = rows()
    mask = np.arange(30) < 15
    kwargs = {
        "scene_families": {"a" * 64: "conflict_family_01",
                           "b" * 64: "conflict_family_02"},
        "config": config(),
    }
    expected = subject.build_fit_weights(arrays, mask, **kwargs)
    poisoned = dict(arrays)
    poisoned["action_indices"] = arrays["action_indices"].copy()
    poisoned["action_indices"][~mask] = np.uint8(255)
    poisoned["group_bits"] = arrays["group_bits"].copy()
    poisoned["group_bits"][~mask] = np.uint8(255)
    actual = subject.build_fit_weights(poisoned, mask, **kwargs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert actual[3] == expected[3]


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
    trace = program.trace(dict(zip(relations.base_feature_names,
                                   arrays["observations"][0])))
    assert trace["triggered_routes"] == []
    assert all(item["triggered"] is False for item in trace["routes"])
    assert all(item["program_trace"] is None for item in trace["routes"])
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


def test_replacement_specialist_uses_fit_route_labels_and_held_poison_is_inert():
    actor, arrays = rows()
    arrays = activate_replacement_route(actor, arrays, (0, 1, 2, 15, 16))
    arrays["action_indices"] = arrays["action_indices"].copy()
    arrays["action_indices"][:3] = np.uint8(4)
    arrays["probabilities"] = np.eye(5, dtype=np.float32)[arrays["action_indices"]]
    relations = R41DiagnosticPublicRelationsV9(actor.metadata["feature_names"])
    mask = np.arange(30) < 15
    kwargs = {
        "relations": relations,
        "scene_families": {"a" * 64: "conflict_family_01",
                           "b" * 64: "conflict_family_02"},
        "config": config(replacement=True),
        "binding_sha256": "e" * 64,
    }
    expected, diagnostics = subject.fit_program(arrays, mask, **kwargs)
    poisoned = dict(arrays)
    poisoned["action_indices"] = arrays["action_indices"].copy()
    poisoned["action_indices"][~mask] = np.uint8(255)
    poisoned["probabilities"] = arrays["probabilities"].copy()
    poisoned["probabilities"][~mask] = np.nan
    actual, poisoned_diagnostics = subject.fit_program(poisoned, mask, **kwargs)
    assert actual.to_dict() == expected.to_dict()
    assert poisoned_diagnostics == diagnostics
    payload = expected.to_dict()
    pickup = payload["specialists"][1]
    assert pickup["combination"] == "replacement"
    assert pickup["mix_weight"] == 1.0
    assert pickup["route"]["feature_name"] == subject.REPLACEMENT_ROUTE_FEATURE
    assert diagnostics["shared_pickup_replacement"]["fit"][
        "held_labels_used"] is False


def test_replacement_candidate_fails_when_either_fold_partition_lacks_support():
    actor, arrays = rows()
    arrays = activate_replacement_route(actor, arrays, (0, 15))
    relations = R41DiagnosticPublicRelationsV9(actor.metadata["feature_names"])
    mask = np.arange(30) < 15
    with pytest.raises(subject.ReplacementSupportError, match="fit support"):
        subject.fit_program(
            arrays, mask, relations=relations,
            scene_families={"a" * 64: "conflict_family_01",
                            "b" * 64: "conflict_family_02"},
            config=config(replacement=True, minimum_rows=2),
            binding_sha256="d" * 64,
        )
