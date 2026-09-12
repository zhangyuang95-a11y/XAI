import numpy as np
import pytest

from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from env.warehouse_native.policy import NumPyNativeActor


ACTOR = (
    "output/warehouse_native/r41_active_2m_20260911/"
    "boundaries/step_2000000/actor.npz"
)


def transformer():
    actor = NumPyNativeActor(ACTOR)
    return actor, R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])


def test_registry_is_stable_and_public_only():
    actor, subject = transformer()
    assert len(subject.base_feature_names) == actor.obs_dim == 197
    assert len(subject.feature_names) == 349
    assert len(subject.derived_feature_names) == 152
    lowered = " ".join(subject.feature_names).lower()
    assert "logit" not in lowered
    assert "hidden" not in lowered
    assert subject.contract()["actor_logits_input"] is False
    assert subject.contract()["intervention_metadata_input"] is False


def test_batch_and_mapping_are_exactly_consistent():
    _, subject = transformer()
    rng = np.random.default_rng(11)
    values = rng.normal(size=(4, 197)).astype(np.float32)
    batch = subject.transform_batch(values)
    mapping = subject.transform_mapping(dict(zip(subject.base_feature_names, values[2])))
    mapped = np.asarray([mapping[name] for name in subject.feature_names], dtype=np.float32)
    assert batch.shape == (4, 349)
    np.testing.assert_array_equal(batch[2], mapped)


def test_critical_predicates_match_registered_public_definitions():
    _, subject = transformer()
    index = {name: i for i, name in enumerate(subject.base_feature_names)}
    values = np.zeros((3, 197), dtype=np.float32)
    values[:, index["self.battery"]] = .8
    values[:, index["other.battery"]] = .8
    values[:, index["charger.self.path_distance"]] = 1
    values[:, index["charger.other.path_distance"]] = 1
    for row in (1, 2):
        for action in ("UP", "DOWN", "LEFT"):
            values[row, index[f"self.neighbor.{action}.passable"]] = 1
        values[row, index["other.path_distance"]] = 1
    for action in ("UP", "DOWN"):
        values[0, index[f"self.neighbor.{action}.passable"]] = 1
    values[0, index["other.path_distance"]] = np.float32(3 / 41)
    values[1, index["task.0.available"]] = 1
    values[1, index["task.0.pickup.self.path_distance"]] = np.float32(4 / 41)
    values[1, index["task.0.pickup.other.path_distance"]] = np.float32(4 / 41)
    values[2, index["self.battery"]] = .3
    values[2, index["other.battery"]] = .8
    values[2, index["charger.self.path_distance"]] = np.float32(5 / 41)
    values[2, index["charger.other.path_distance"]] = np.float32(5 / 41)
    masks = subject.critical_masks(values)
    assert masks["narrow_passage"].tolist() == [True, False, False]
    assert masks["shared_pickup"].tolist() == [False, True, False]
    assert masks["shared_charger"].tolist() == [False, False, True]


def test_nonfinite_and_missing_values_are_rejected():
    _, subject = transformer()
    values = np.zeros((1, 197), dtype=np.float32)
    values[0, 4] = np.nan
    with pytest.raises(ValueError):
        subject.transform_batch(values)
    with pytest.raises(ValueError):
        subject.transform_mapping({})


def test_registry_rejects_forbidden_and_incomplete_inputs():
    _, subject = transformer()
    with pytest.raises(ValueError):
        R41DiagnosticPublicRelationsV8((*subject.base_feature_names, "actor.logits.0"))
    with pytest.raises(ValueError):
        R41DiagnosticPublicRelationsV8(tuple(
            name for name in subject.base_feature_names
            if name != "other.path_distance"
        ))
