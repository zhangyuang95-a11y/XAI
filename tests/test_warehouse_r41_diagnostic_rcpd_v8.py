from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v8 as subject
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as weights
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    VERSION as BOOSTED_VERSION,
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    AGGREGATION,
    R41DiagnosticPublicTreeProgramV8,
    assemble_public_tree_program_v8,
)


def config():
    model = {
        "learning_rate": 0.1,
        "max_iter": 20,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 5,
        "l2_regularization": 0.1,
        "max_depth": None,
        "max_bins": 64,
        "random_state": 17,
    }
    return {
        "version": subject.CONFIG_VERSION,
        "pair_pool_multiplier": 8.0,
        "use_action_factor": True,
        "models": {
            "base": deepcopy(model),
            "narrow_passage": {**model, "random_state": 19},
            "shared_pickup": {**model, "random_state": 23},
            "shared_charger": {**model, "random_state": 29},
        },
        "mix_weights": {
            "narrow_passage": 0.2,
            "shared_pickup": 0.3,
            "shared_charger": 0.4,
        },
    }


def base_feature_names():
    required = sorted(R41DiagnosticPublicRelationsV8._required_names())
    fillers = [f"public.filler.{index}"
               for index in range(197 - len(required))]
    return tuple((*required, *fillers))


def tree_program(relations, *, component, binding, fit_config):
    trees = []
    for output_index in range(5):
        trees.append({
            "iteration": 0,
            "output_index": output_index,
            "nodes": [{"kind": "leaf", "value": float(output_index == 0)}],
        })
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_VERSION,
        "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(subject.CLASSES),
        "action_names": list(subject.ACTIONS),
        "output_kind": "multiclass_logits",
        "baseline": [0.0] * 5,
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1,
        "trees": trees,
        "metadata": {
            "diagnostic_rcpd_version": subject.VERSION,
            "diagnostic_rcpd_binding_sha256": binding,
            "component": component,
            "fit_rows": 25,
            "fit_config": fit_config,
            "prediction_input": "349 deterministic public features",
            "validation_labels_used_for_fit": False,
            "actor_logits_used_as_program_input": False,
            "actor_hidden_states_used": False,
            "runtime_action_override": False,
        },
    })


def wrapped_program(cfg=None, binding="b" * 64):
    cfg = config() if cfg is None else cfg
    relations = R41DiagnosticPublicRelationsV8(base_feature_names())
    parts = {
        name: tree_program(
            relations, component=name, binding=binding,
            fit_config=cfg["models"][name])
        for name in subject.COMPONENTS
    }
    program = assemble_public_tree_program_v8(
        relations.base_feature_names,
        parts["base"], parts["narrow_passage"],
        parts["shared_pickup"], parts["shared_charger"],
        mix_weights=cfg["mix_weights"], routes=subject._fixed_routes(),
        metadata={
            "diagnostic_rcpd_version": subject.VERSION,
            "diagnostic_rcpd_binding_sha256": binding,
            "fit_config_sha256": subject.digest(cfg),
            "public_feature_contract_sha256": subject.digest(relations.contract()),
            "pair_weight_contract_sha256": subject.digest(weights.contract()),
            "actions": list(subject.ACTIONS),
            "classes": list(subject.CLASSES),
            "native_source_actor_sha256": "a" * 64,
            "source_actor_parameters_sha256": "c" * 64,
            "source_full_manifest_bindings": {
                "actor_sha256": "a" * 64,
                "protocol_sha256": "d" * 64,
                "manifest_sha256": "e" * 64,
            },
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
            "formal_ready": False,
        },
    )
    return relations, program


def test_contract_has_no_final_input_and_fixes_public_protocol():
    value = subject.contract()
    assert value["source_rows"]["final_rows_accessed"] is False
    assert value["source_rows"]["final_labels_accessed"] is False
    assert value["prediction_inputs"]["base_public_features"] == 197
    assert value["prediction_inputs"]["expanded_public_features"] == 349
    assert value["prediction_inputs"]["actor_logits"] is False
    assert value["prediction_inputs"]["actor_hidden_state"] is False
    assert value["actions"] == list(subject.ACTIONS)
    assert value["classes"] == list(subject.CLASSES)
    assert value["routes"] == subject._fixed_routes()
    assert value["aggregation"] == AGGREGATION
    assert value["weights"]["contract_sha256"] == subject.digest(weights.contract())


def test_disclosed_fit_config_is_strict_and_normalized():
    value = config()
    assert subject.normalize_config(value) == value
    reordered = json.loads(json.dumps(value, sort_keys=True))
    assert subject.normalize_config(reordered) == value
    for mutation in (
        lambda item: item.update(extra=True),
        lambda item: item.update(pair_pool_multiplier=0.0),
        lambda item: item["models"]["base"].update(max_bins=256),
        lambda item: item["mix_weights"].update(shared_pickup=1.1),
        lambda item: item["models"]["base"].update(unknown=1),
    ):
        changed = deepcopy(value)
        mutation(changed)
        with pytest.raises(ValueError):
            subject.normalize_config(changed)


def minimal_arrays(prefix, split):
    count = len(split)
    result = {}
    for name in subject._ROW_FIELDS:
        if name == "observations":
            result[name] = np.zeros((count, 197), dtype=np.float32)
        elif name == "probabilities":
            result[name] = np.full((count, 5), 0.2, dtype=np.float32)
        elif name == "action_indices":
            result[name] = np.zeros(count, dtype=np.uint8)
        elif name == "weights":
            result[name] = np.ones(count, dtype=np.float32)
        elif name == "frames":
            result[name] = np.arange(count, dtype=np.int16)
        elif name == "group_bits":
            result[name] = np.zeros(count, dtype=np.uint8)
        elif name in ("submitted_equal", "trajectory_done"):
            result[name] = np.ones(count, dtype=np.bool_)
        elif name == "split_validation":
            result[name] = np.asarray(split, dtype=np.bool_)
        else:
            width = {
                "observation_hashes": 64, "scene_fingerprints": 64,
                "episode_ids": 180, "kinds": 16, "anchor_ids": 240,
                "branch_actions": 8, "physical_hashes": 64,
                "source_state_hashes": 64,
            }[name]
            result[name] = np.asarray(
                [f"{prefix}{index}" for index in range(count)],
                dtype=f"S{width}")
    return result


def test_validation_wins_exact_observation_overlap():
    prior = minimal_arrays("prior", [False, True, False])
    expansion = minimal_arrays("new", [False, True, False])
    prior["observation_hashes"][:] = np.asarray([b"a", b"b", b"c"], dtype="S64")
    expansion["observation_hashes"][:] = np.asarray([b"d", b"b", b"e"], dtype="S64")
    prior["split_validation"][:] = np.asarray([False, True, False])
    combined, accounting = subject._validation_wins_merge(
        prior, expansion, layout="expansion_only",
        expansion_fingerprints={"not-used-by-merge"})
    assert accounting["fit_rows_removed_for_exact_validation_overlap"] == 1
    assert combined["split_validation"].tolist() == [False, False, False, True, False]
    fit_hashes = set(map(bytes, combined["observation_hashes"][
        ~combined["split_validation"]]))
    validation_hashes = set(map(bytes, combined["observation_hashes"][
        combined["split_validation"]]))
    assert fit_hashes.isdisjoint(validation_hashes)
    assert validation_hashes == {b"b"}


def test_component_fit_masks_never_include_validation_labels():
    arrays = {
        "split_validation": np.asarray([False, False, False, True, True]),
        "group_bits": np.asarray([1, 0, 0, 7, 7], dtype=np.uint8),
    }
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    masks = subject._component_fit_masks(arrays, pairs)
    assert masks["base"].tolist() == [True, True, True, False, False]
    assert masks["narrow_passage"].tolist() == [True, True, False, False, False]
    assert all(not mask[3:].any() for mask in masks.values())


def metric(value=0.9, scenes=10):
    return {
        "overall": {"fidelity": value, "rows": 100, "scenes": scenes},
        "nonwait": {"fidelity": value, "rows": 80, "scenes": scenes},
        "critical": {
            group: {"fidelity": value, "rows": 20, "scenes": scenes}
            for group in subject.GROUPS
        },
        "effective_intervention_direction": {
            "fidelity": value, "pairs": 30, "scenes": scenes,
            "by_group": {
                group: {"fidelity": value, "pairs": 10, "scenes": scenes}
                for group in subject.GROUPS
            },
        },
        "mean_kl": 0.1,
    }


def test_every_registered_hard_gate_is_required():
    assert subject._gate(metric())["passed"] is True
    checks = list(subject._gate(metric())["checks"])
    for check in checks:
        changed = metric()
        if check == "overall":
            changed["overall"]["fidelity"] = 0.899
        elif check == "nonwait":
            changed["nonwait"]["fidelity"] = 0.899
        elif check == "effective_intervention_direction":
            changed[check]["fidelity"] = 0.849
        elif check in subject.GROUPS:
            changed["critical"][check]["fidelity"] = 0.849
        else:
            group = check.removeprefix("effective_intervention_direction_")
            changed["effective_intervention_direction"]["by_group"][group][
                "fidelity"] = 0.849
        assert subject._gate(changed)["passed"] is False, check
    too_few = metric(scenes=9)
    assert subject._gate(too_few)["passed"] is False


def test_program_identity_binds_actions_classes_routes_aggregation_and_sources():
    cfg = config()
    relations, program = wrapped_program(cfg)
    source_identity = {
        "native_source_actor_sha256": "a" * 64,
        "source_actor_parameters_sha256": "c" * 64,
        "source_full_manifest_bindings": {
            "actor_sha256": "a" * 64,
            "protocol_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
        },
    }
    subject._validate_program_identity(
        program, relations=relations, config=cfg, binding="b" * 64,
        source_identity=source_identity)
    payload = program.to_dict()
    payload["action_names"][0] = "WAIT"
    with pytest.raises(ValueError):
        R41DiagnosticPublicTreeProgramV8.from_dict(payload)
    payload = program.to_dict()
    payload["metadata"]["pair_weight_contract_sha256"] = "0" * 64
    changed = R41DiagnosticPublicTreeProgramV8.from_dict(payload)
    with pytest.raises(ValueError):
        subject._validate_program_identity(
            changed, relations=relations, config=cfg, binding="b" * 64,
            source_identity=source_identity)
    for key, value in (
        ("native_source_actor_sha256", "0" * 64),
        ("source_actor_parameters_sha256", "1" * 64),
        ("source_full_manifest_bindings", {"tampered": True}),
        ("runtime_controller", "frozen_neural_actor_only"),
        ("program_feedback_into_actor", True),
    ):
        payload = program.to_dict()
        payload["metadata"][key] = value
        changed = R41DiagnosticPublicTreeProgramV8.from_dict(payload)
        with pytest.raises(ValueError):
            subject._validate_program_identity(
                changed, relations=relations, config=cfg, binding="b" * 64,
                source_identity=source_identity)


def test_reader_rejects_wrong_report_hash_before_reading_artifacts(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    actual = hashlib.sha256(report.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="report hash differs"):
        subject.read_saved_report(
            tmp_path, expected_report_sha256="0" * 64,
            actor_path=report, protocol_path=report, manifest_path=report,
            designation_path=report, expansion_registry_path=report,
            expected_expansion_registry_sha256=actual,
            expected_prior_v7_report_sha256=actual,
            previous_development_path=report,
            expected_expansion_rows_sha256=actual,
            expected_config_sha256=actual,
        )


def test_producer_has_no_final_cli_and_never_serializes_pickle():
    source = open(subject.__file__, encoding="utf-8").read()
    assert 'parser.add_argument("--final' not in source
    assert "pickle.dump" not in source
    assert "joblib.dump" not in source
    assert subject.MAX_PROGRAM_BYTES > 0
