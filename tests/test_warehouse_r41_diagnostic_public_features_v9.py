import numpy as np
import pytest

from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from env.warehouse_native.policy import NumPyNativeActor


ACTOR = (
    "output/warehouse_native/r41_active_2m_20260911/"
    "boundaries/step_2000000/actor.npz"
)


def transformer():
    actor = NumPyNativeActor(ACTOR)
    return actor, R41DiagnosticPublicRelationsV9(actor.metadata["feature_names"])


def test_registry_is_public_and_extends_v8():
    actor, subject = transformer()
    assert len(subject.base_feature_names) == actor.obs_dim == 197
    assert len(subject.feature_names) > 349
    assert subject.feature_names[:349] == (
        *subject.base_feature_names, *subject.v8.derived_feature_names)
    lowered = " ".join(subject.feature_names).casefold()
    for forbidden in ("logit", "hidden", "physical_hash", "scene_fingerprint"):
        assert forbidden not in lowered
    contract = subject.contract()
    assert contract["prediction_inputs"] == ["public_observation"]
    assert contract["scene_identifier_input"] is False
    assert contract["action_label_input"] is False


def test_batch_mapping_parity_and_critical_predicates():
    _, subject = transformer()
    rng = np.random.default_rng(19)
    values = rng.normal(size=(4, 197)).astype(np.float32)
    # Public map and normalized coordinates must remain valid.
    index = {name: i for i, name in enumerate(subject.base_feature_names)}
    for row in range(6):
        for column in range(7):
            values[:, index[f"map.{row}.{column}.passable"]] = rng.integers(0, 2, 4)
    for name, size in (("self.row", 6), ("other.row", 6)):
        values[:, index[name]] = rng.integers(0, size, 4) / (size - 1)
    for name, size in (("self.column", 7), ("other.column", 7)):
        values[:, index[name]] = rng.integers(0, size, 4) / (size - 1)
    batch = subject.transform_batch(values)
    mapping = subject.transform_mapping(dict(zip(subject.base_feature_names, values[2])))
    mapped = np.asarray([mapping[name] for name in subject.feature_names], dtype=np.float32)
    np.testing.assert_array_equal(batch[2], mapped)
    assert batch.shape == (4, len(subject.feature_names))
    v8_masks = subject.v8.critical_masks(values)
    masks = subject.critical_masks(values)
    for name in v8_masks:
        np.testing.assert_array_equal(masks[name], v8_masks[name])


def test_missing_nonfinite_and_forbidden_registry_are_rejected():
    _, subject = transformer()
    values = np.zeros((1, 197), dtype=np.float32)
    values[0, 2] = np.nan
    with pytest.raises(ValueError):
        subject.transform_batch(values)
    with pytest.raises(ValueError):
        subject.transform_mapping({})
    with pytest.raises(ValueError):
        R41DiagnosticPublicRelationsV9((*subject.base_feature_names, "actor.hidden.0"))
