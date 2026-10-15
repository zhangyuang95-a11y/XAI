import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v12_fit as subject
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    GROUPS,
    assemble_public_tree_program_v9,
)


def _base_feature_names() -> tuple[str, ...]:
    required = sorted(R41DiagnosticPublicRelationsV9._required_names())
    fillers = [f"public.filler.{index}"
               for index in range(197 - len(required))]
    names = tuple((*required, *fillers))
    assert len(names) == len(set(names)) == 197
    return names


def _config() -> dict:
    return {
        "version": subject.CONFIG_VERSION,
        "pair_pool_multiplier": 32.0,
        "wait_endpoint_share": 0.5,
        "model": {
            "learning_rate": 0.1, "max_iter": 120,
            "max_leaf_nodes": 127, "min_samples_leaf": 10,
            "l2_regularization": 0.1, "max_depth": 16,
            "max_bins": 255, "random_state": 1941,
        },
        "shared_pickup_replacement": {
            "version": subject.REPLACEMENT_VERSION,
            "enabled": True, "group": "shared_pickup",
            "combination": "replacement",
            "route": {
                "feature_name": subject.REPLACEMENT_ROUTE_FEATURE,
                "operator": ">", "threshold": 0.5,
            },
            "fit_source": "fit_only_public_route_rows",
            "estimator": "fit_only_weighted_majority",
            "minimum_rows_per_partition": 2,
            "minimum_scenes_per_partition": 2,
            "minimum_episodes_per_partition": 2,
        },
        "shared_charger_endpoint_specialist": {
            "version": subject.CHARGER_SPECIALIST_VERSION,
            "enabled": True, "group": "shared_charger",
            "combination": "weighted_average", "mix_weight": 1.0,
            "route": {
                "feature_name": subject.CHARGER_ROUTE_FEATURE,
                "operator": ">", "threshold": 0.5,
            },
            "fit_source": (
                "fit_partition_effective_shared_charger_pair_endpoints"),
            "estimator": "hist_gradient_boosting_classifier",
            "endpoint_population": "unique_pair_endpoint_rows",
            "endpoint_wait_share": 0.8,
            "sample_weight_source": (
                "fit_only_global_weights_with_specialist_0.8_wait_share"),
            "model": {
                "learning_rate": 0.1, "max_iter": 100,
                "max_leaf_nodes": 63, "min_samples_leaf": 10,
                "l2_regularization": 0.1, "max_depth": 12,
                "max_bins": 255, "random_state": 1941,
            },
        },
    }


def test_v12_config_freezes_fit_only_charger_specialist():
    config = subject.normalize_config(_config())
    specialist = config["shared_charger_endpoint_specialist"]
    assert config["wait_endpoint_share"] == 0.5
    assert specialist["endpoint_wait_share"] == 0.8
    assert specialist["route"] == {
        "feature_name": "derived.critical.shared_charger",
        "operator": ">", "threshold": 0.5,
    }
    assert specialist["mix_weight"] == 1.0
    assert specialist["fit_source"].startswith("fit_partition_")


@pytest.mark.parametrize("mutation", ["share", "route", "extra"])
def test_v12_config_rejects_unregistered_specialist_variants(mutation):
    config = _config()
    if mutation == "share":
        config["shared_charger_endpoint_specialist"][
            "endpoint_wait_share"] = 0.7
    elif mutation == "route":
        config["shared_charger_endpoint_specialist"]["route"][
            "feature_name"] = "derived.critical.shared_pickup"
    else:
        config["shared_charger_endpoint_specialist"]["unregistered"] = True
    with pytest.raises(ValueError):
        subject.normalize_config(config)


def test_charger_endpoint_population_is_unique_and_charger_only():
    pairs = np.asarray([[1, 2], [1, 3], [4, 5], [2, 6]], dtype=np.int64)
    charger = 1 << GROUPS.index("shared_charger")
    pickup = 1 << GROUPS.index("shared_pickup")
    bits = np.asarray([charger, charger | pickup, pickup, charger], dtype=np.uint8)
    assert subject.shared_charger_endpoint_indices(pairs, bits).tolist() == [
        1, 2, 3, 6]


def test_fit_replaces_only_the_traceable_charger_component(monkeypatch):
    relations = R41DiagnosticPublicRelationsV9(_base_feature_names())
    dummy = subject.v9.constant_component(relations)
    base = assemble_public_tree_program_v9(
        relations.base_feature_names, dummy, dummy, dummy, dummy,
        mix_weights={group: 0.0 for group in GROUPS},
        metadata={"runtime_action_override": False},
    )
    monkeypatch.setattr(
        subject.v9, "fit_program", lambda *args, **kwargs: (
            base, {"base": "diagnostic"}))
    monkeypatch.setattr(
        subject, "_fit_charger_specialist", lambda *args, **kwargs: (
            dummy, {"held_labels_used": False}))
    program, diagnostics = subject.fit_program(
        {}, np.ones(1, dtype=np.bool_), relations=relations,
        scene_families={}, config=_config(), binding_sha256="a" * 64)
    assert program.mix_weights["shared_charger"] == 1.0
    assert program.combinations["shared_charger"] == "weighted_average"
    assert next(route for route in program.routes
                if route["feature_name"] == subject.CHARGER_ROUTE_FEATURE) == {
        "feature_name": subject.CHARGER_ROUTE_FEATURE,
        "operator": ">", "threshold": 0.5,
    }
    assert program.mix_weights["narrow_passage"] == 0.0
    assert program.mix_weights["shared_pickup"] == 0.0
    assert diagnostics["shared_charger_endpoint_specialist"][
        "held_labels_used"] is False
    assert program.to_dict()["metadata"]["runtime_action_override"] is False
