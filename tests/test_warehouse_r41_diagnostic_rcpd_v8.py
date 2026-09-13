from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
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
    assert value["config_selection"]["direct_config_input"] is False
    assert value["config_selection"][
        "selector_report_scope_and_selected_config_copied"] is True
    assert value["effective_pair_group_assignment"].startswith(
        "ordinary pre-action source anchor")


def test_active_v8_chain_uses_fixed_designation_and_frozen_manifest_closure():
    assert subject.designation_binding.EXPECTED_DESIGNATION_SHA256 == (
        "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
    )
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_designation_v2.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_designation_v2_binding.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_frozen_manifest_v2.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_prior_rows_v8.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_expansion_rows_v8.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rows_v8.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v8_outer_split.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v8_fit_selector.py" in sources
    assert "scripts/build_warehouse_r41_diagnostic_designation_v2.py" in sources
    assert "env/warehouse/transition_outcome.py" in sources
    assert not [path for path in sources if any(token in path for token in (
        "admission", "release", "preflight", "final_once",
        "fresh_final_holdout", "explanation_audit",
    ))]


def test_source_closure_rejects_overlapping_hash_disagreement(monkeypatch):
    monkeypatch.setattr(
        subject, "local_source_hashes", lambda paths: {"shared.py": "a" * 64})
    monkeypatch.setattr(
        subject.designation_binding.designation, "source_closure",
        lambda: {"shared.py": "b" * 64})
    with pytest.raises(RuntimeError, match="closure hash disagreement"):
        subject.producer_sources()


def test_build_and_reader_accept_only_reauthenticated_row_receipts():
    for function in (subject.build, subject.read_saved_report):
        parameters = inspect.signature(function).parameters
        assert "expected_prior_rows_report_sha256" in parameters
        assert "expected_expansion_rows_report_sha256" in parameters
        assert "expected_prior_v7_report_sha256" not in parameters
        assert "expected_expansion_rows_sha256" not in parameters
    parameters = inspect.signature(subject.build).parameters
    assert "prior_rows_output" in parameters
    assert "expansion_rows_output" in parameters
    assert "selector_evidence" in parameters
    assert "expected_selector_report_sha256" in parameters
    assert "config_path" not in parameters
    assert "expected_config_sha256" not in parameters
    assert "prior_v7_output" not in parameters
    assert "expansion_rows_path" not in parameters


def test_complete_binding_requires_authenticated_previous_and_row_semantics(
        tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"fixed")
    paths = {
        name: artifact for name in (
            "actor", "protocol", "manifest", "designation",
            "prior_reauth_report", "prior_source_report", "prior_rows",
            "expansion_registry", "expansion_registry_report",
            "expansion_reauth_report", "expansion_collection_report",
            "expansion_rows",
            "previous_development", "selector_report", "selector_scope",
            "selector_selected_config",
        )
    }

    class Actor:
        metadata = {
            "actor_parameters_sha256": "p" * 64,
            "feature_names": ["feature"],
        }

    class Relations:
        feature_names = ("feature",)

        @staticmethod
        def contract():
            return {"relations": True}

    authenticated = {
        "actor": Actor(),
        "protocol": {"protocol": True},
        "designation": {"designation": True},
        "expansion": {"content_sha256": "e" * 64},
        "prior_report": {"bindings": {
            "source_v7_report_semantic_sha256": "s" * 64,
            "source_v7_rows_semantic_sha256": "r" * 64,
        }},
        "expansion_rows_report": {"bindings": {
            "expansion_report_semantic_sha256": "g" * 64,
            "source_collection_report_semantic_sha256": "c" * 64,
            "source_rows_semantic_sha256": "x" * 64,
        }},
        "previous_development": {"scenes": [{"id": "dev"}]},
        "relations": Relations(),
        "config": config(),
        "selector": {
            "report": {"bindings": {
                "fresh_outer_registry_file_sha256": "u" * 64,
                "fresh_outer_report_file_sha256": "v" * 64,
            }},
            "scope": {"content_sha256": "w" * 64},
            "selected_config_record": {"selected": True},
        },
        "source_full_manifest_bindings": {"manifest": "m" * 64},
    }
    monkeypatch.setattr(subject.weight_api, "contract", lambda: {"weights": True})
    monkeypatch.setattr(subject.weight_api, "__file__", str(artifact))
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    bindings = subject._bindings(
        paths=paths, authenticated=authenticated,
        sources={"artifact": subject.file_hash(artifact)})

    assert bindings["previous_development_semantic_sha256"] \
        == subject.digest(authenticated["previous_development"])
    assert bindings["prior_v7_source_report_semantic_sha256"] == "s" * 64
    assert bindings["prior_v7_rows_semantic_sha256"] == "r" * 64
    assert bindings["expansion_registry_report_semantic_sha256"] == "g" * 64
    assert (bindings["expansion_source_collection_report_semantic_sha256"]
            == "c" * 64)
    assert bindings["expansion_rows_semantic_sha256"] == "x" * 64
    assert bindings["fit_selector_fresh_outer_registry_file_sha256"] == "u" * 64
    assert bindings["fit_selector_fresh_outer_report_file_sha256"] == "v" * 64
    assert bindings["fit_selector_selected_config_sha256"] == subject.digest(
        authenticated["config"])
    assert bindings["fit_config_file_sha256"] == subject._canonical_json_file_sha256(
        authenticated["config"])

    without_previous = dict(authenticated)
    without_previous.pop("previous_development")
    with pytest.raises(KeyError):
        subject._bindings(
            paths=paths, authenticated=without_previous,
            sources={"artifact": subject.file_hash(artifact)})


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


def intervention_pair_arrays(*, split, source_bits, wait_bits, branch_bits):
    arrays = minimal_arrays("pair", [split, split, split])
    arrays["scene_fingerprints"][:] = b"scene"
    arrays["episode_ids"][:] = b"scene:skilled"
    arrays["kinds"][:] = np.asarray(
        [b"ordinary", b"intervention", b"intervention"], dtype="S16")
    arrays["anchor_ids"][:] = b"scene:skilled:0"
    arrays["branch_actions"][:] = np.asarray(
        [b"", b"WAIT", b"RIGHT"], dtype="S8")
    arrays["physical_hashes"][:] = np.asarray(
        [b"", b"a" * 64, b"b" * 64], dtype="S64")
    arrays["group_bits"][:] = np.asarray(
        [source_bits, wait_bits, branch_bits], dtype=np.uint8)
    arrays["action_indices"][:] = np.asarray([0, 0, 1], dtype=np.uint8)
    pairs = np.asarray([[1, 2]], dtype=np.int64)
    return arrays, pairs


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
    masks = subject._component_fit_masks(
        arrays, pairs, np.asarray([1], dtype=np.uint8))
    assert masks["base"].tolist() == [True, True, True, False, False]
    assert masks["narrow_passage"].tolist() == [True, True, False, False, False]
    assert all(not mask[3:].any() for mask in masks.values())


def test_pair_groups_use_ordinary_source_when_wait_endpoint_differs():
    arrays, pairs = intervention_pair_arrays(
        split=False, source_bits=1, wait_bits=2, branch_bits=0)
    pair_bits = subject._pair_group_bits(arrays, pairs)
    assert pair_bits.tolist() == [1]
    assert pair_bits.tolist() != arrays["group_bits"][pairs[:, 0]].tolist()

    masks = subject._component_fit_masks(arrays, pairs, pair_bits)
    assert masks["narrow_passage"].tolist() == [True, True, True]
    # The WAIT row itself is a shared-pickup state, but that post-action flag
    # must not pull the other endpoint into the shared-pickup specialist.
    assert masks["shared_pickup"].tolist() == [False, True, False]


def test_pair_groups_fall_back_to_endpoint_union_without_source_row():
    arrays, pairs = intervention_pair_arrays(
        split=False, source_bits=1, wait_bits=2, branch_bits=4)
    arrays["anchor_ids"][0] = b""
    assert subject._pair_group_bits(arrays, pairs).tolist() == [6]

    # A same-text ordinary anchor on the other side of the development split
    # must not supply validation group membership.
    arrays, pairs = intervention_pair_arrays(
        split=True, source_bits=1, wait_bits=2, branch_bits=4)
    arrays["split_validation"][0] = False
    assert subject._pair_group_bits(arrays, pairs).tolist() == [6]


def test_metrics_and_strata_share_source_pair_groups_and_reject_gaps():
    arrays, pairs = intervention_pair_arrays(
        split=True, source_bits=1, wait_bits=2, branch_bits=0)
    pair_bits = subject._pair_group_bits(arrays, pairs)
    probabilities = np.full((3, 5), 0.025, dtype=np.float64)
    probabilities[np.arange(3), arrays["action_indices"]] = 0.9

    metrics = subject._metrics(
        probabilities, arrays, pairs=pairs, pair_group_bits=pair_bits)
    direction = metrics["effective_intervention_direction"]["by_group"]
    assert direction["narrow_passage"]["pairs"] == 1
    assert direction["narrow_passage"]["fidelity"] == 1.0
    assert direction["shared_pickup"]["pairs"] == 0

    strata = subject._strata(
        probabilities, arrays, {"scene": "family"},
        pairs=pairs, pair_group_bits=pair_bits)
    assert strata["critical"]["narrow_passage"]["effective_pairs"] == 1
    assert strata["critical"]["shared_pickup"]["effective_pairs"] == 0
    # Row strata remain current-state facts even though pair direction uses the
    # ordinary pre-action source state.
    assert strata["critical"]["shared_pickup"]["rows"] == 1

    with pytest.raises(ValueError, match="metric coverage differs"):
        subject._metrics(
            probabilities, arrays,
            pairs=np.empty((0, 2), dtype=np.int64),
            pair_group_bits=np.empty(0, dtype=np.uint8))
    with pytest.raises(ValueError, match="group assignment differs"):
        subject._metrics(
            probabilities, arrays, pairs=pairs,
            pair_group_bits=np.asarray([2], dtype=np.uint8))
    with pytest.raises(ValueError, match="stratum pair coverage differs"):
        subject._strata(
            probabilities, arrays, {"scene": "family"},
            pairs=np.empty((0, 2), dtype=np.int64),
            pair_group_bits=np.empty(0, dtype=np.uint8))


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
            expansion_report_path=report,
            expected_expansion_report_sha256=actual,
            expected_prior_rows_report_sha256=actual,
            previous_development_path=report,
            expected_expansion_rows_report_sha256=actual,
            expected_selector_report_sha256=actual,
        )


def test_producer_has_no_final_cli_and_never_serializes_pickle():
    source = open(subject.__file__, encoding="utf-8").read()
    assert 'parser.add_argument("--final' not in source
    assert 'parser.add_argument("--config"' not in source
    assert 'parser.add_argument("--selector-evidence"' in source
    assert "pickle.dump" not in source
    assert "joblib.dump" not in source
    assert subject.MAX_PROGRAM_BYTES > 0


def _toctou_build_inputs(tmp_path):
    paths = {}
    for name in (
        "actor", "protocol", "manifest", "designation",
        "expansion_registry", "expansion_report", "previous",
    ):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode("ascii"))
        paths[name] = path
    selector = tmp_path / "selector"
    selector.mkdir()
    selector_scope = selector / "fit_scope.json"
    selector_scope.write_text("{}\n", encoding="utf-8")
    selector_config = selector / "selected_config.json"
    selector_config.write_text("{}\n", encoding="utf-8")
    selector_report = selector / "report.json"
    selector_report.write_text(json.dumps({
        "evidence_artifacts": {
            "fit_scope.json": subject.file_hash(selector_scope),
            "selected_config.json": subject.file_hash(selector_config),
        },
    }, sort_keys=True) + "\n", encoding="utf-8")
    paths["selector"] = selector
    paths["selector_report"] = selector_report
    paths["selector_scope"] = selector_scope
    paths["selector_selected_config"] = selector_config
    paths["manifest_validation"] = tmp_path / "validation.json"
    paths["manifest_validation"].write_bytes(b"validation\n")
    paths["designation_actor"] = paths["actor"]
    paths["designation_protocol"] = paths["protocol"]
    for name in ("training_ledger", "dual_evaluation", "failure_closeout"):
        paths["designation_" + name] = tmp_path / ("designation_" + name)
        paths["designation_" + name].write_bytes((name + "\n").encode("ascii"))
    prior = tmp_path / "prior"
    prior.mkdir()
    for name in ("report.json", "rows.npz", "source_v7_report.json"):
        (prior / name).write_bytes(("prior-" + name + "\n").encode("ascii"))
    expansion = tmp_path / "expansion"
    expansion.mkdir()
    for name in (
            "report.json", "source_collection_report.json",
            "expansion_rows.npz"):
        (expansion / name).write_bytes(
            ("expansion-" + name + "\n").encode("ascii"))
    paths["prior"] = prior
    paths["expansion"] = expansion
    return paths


def _patch_toctou_manifest_validation(paths, monkeypatch):
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_VALIDATION_SHA256",
        subject.file_hash(paths["manifest_validation"]))
    components = {
        "actor": paths["actor"], "protocol": paths["protocol"],
        "training_ledger": paths["designation_training_ledger"],
        "dual_evaluation": paths["designation_dual_evaluation"],
        "failure_closeout": paths["designation_failure_closeout"],
    }
    monkeypatch.setattr(
        subject, "_resolved_designation_components", lambda path: components)

    def expected_hashes(**kwargs):
        return {
            "actor": subject.file_hash(paths["actor"]),
            "protocol": subject.file_hash(paths["protocol"]),
            "manifest": subject.file_hash(paths["manifest"]),
            "manifest_validation": subject.file_hash(paths["manifest_validation"]),
            "designation": subject.file_hash(paths["designation"]),
            "expansion_registry": subject.file_hash(paths["expansion_registry"]),
            "expansion_registry_report": subject.file_hash(paths["expansion_report"]),
            "selector_report": subject.file_hash(paths["selector_report"]),
            "selector_scope": subject.file_hash(paths["selector_scope"]),
            "selector_selected_config": subject.file_hash(
                paths["selector_selected_config"]),
            "previous_development": subject.file_hash(paths["previous"]),
            "prior_reauth_report": subject.file_hash(paths["prior"] / "report.json"),
            "prior_rows": subject.file_hash(paths["prior"] / "rows.npz"),
            "prior_source_report": subject.file_hash(
                paths["prior"] / "source_v7_report.json"),
            "expansion_reauth_report": subject.file_hash(
                paths["expansion"] / "report.json"),
            "expansion_collection_report": subject.file_hash(
                paths["expansion"] / "source_collection_report.json"),
            "expansion_rows": subject.file_hash(
                paths["expansion"] / "expansion_rows.npz"),
            "designation_actor": subject.file_hash(paths["actor"]),
            "designation_protocol": subject.file_hash(paths["protocol"]),
            "designation_training_ledger": subject.file_hash(
                paths["designation_training_ledger"]),
            "designation_dual_evaluation": subject.file_hash(
                paths["designation_dual_evaluation"]),
            "designation_failure_closeout": subject.file_hash(
                paths["designation_failure_closeout"]),
        }

    monkeypatch.setattr(subject, "_snapshot_expected_hashes", expected_hashes)


def _toctou_build_kwargs(paths, output):
    return {
        "actor_path": paths["actor"],
        "protocol_path": paths["protocol"],
        "manifest_path": paths["manifest"],
        "designation_path": paths["designation"],
        "expansion_registry_path": paths["expansion_registry"],
        "expected_expansion_registry_sha256": subject.file_hash(
            paths["expansion_registry"]),
        "expansion_report_path": paths["expansion_report"],
        "expected_expansion_report_sha256": subject.file_hash(
            paths["expansion_report"]),
        "prior_rows_output": paths["prior"],
        "expected_prior_rows_report_sha256": subject.file_hash(
            paths["prior"] / "report.json"),
        "previous_development_path": paths["previous"],
        "expansion_rows_output": paths["expansion"],
        "expected_expansion_rows_report_sha256": subject.file_hash(
            paths["expansion"] / "report.json"),
        "selector_evidence": paths["selector"],
        "expected_selector_report_sha256": subject.file_hash(
            paths["selector_report"]),
        "output": output,
    }


def test_selector_bootstrap_pins_report_scope_and_selected_config(tmp_path):
    paths = _toctou_build_inputs(tmp_path)
    originals, expected = subject._selector_snapshot_inputs(
        paths["selector"],
        expected_report_sha256=subject.file_hash(paths["selector_report"]))
    assert originals == {
        "selector_report": paths["selector_report"],
        "selector_scope": paths["selector_scope"],
        "selector_selected_config": paths["selector_selected_config"],
    }
    assert expected == {
        name: subject.file_hash(path) for name, path in originals.items()
    }


def test_selector_bootstrap_rejects_unbound_selected_config(tmp_path):
    paths = _toctou_build_inputs(tmp_path)
    report = json.loads(paths["selector_report"].read_text(encoding="utf-8"))
    del report["evidence_artifacts"]["selected_config.json"]
    paths["selector_report"].write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact registry differs"):
        subject._selector_snapshot_inputs(
            paths["selector"], expected_report_sha256=subject.file_hash(
                paths["selector_report"]))


def test_build_snapshots_sources_before_authentication_and_rejects_change(
        tmp_path, monkeypatch):
    paths = _toctou_build_inputs(tmp_path)
    output = tmp_path / "candidate"
    _patch_toctou_manifest_validation(paths, monkeypatch)
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return {"producer.py": ("a" if calls == 1 else "b") * 64}

    monkeypatch.setattr(subject, "producer_sources", changing_sources)
    monkeypatch.setattr(subject, "_authenticate_inputs", lambda **kwargs: {})
    with pytest.raises(RuntimeError, match="changed during authentication"):
        subject.build(**_toctou_build_kwargs(paths, output))
    assert not output.exists()
    assert not (tmp_path / ".candidate.lock").exists()


def test_build_rechecks_sources_after_staging_and_before_publication(
        tmp_path, monkeypatch):
    paths = _toctou_build_inputs(tmp_path)
    output = tmp_path / "candidate"
    _patch_toctou_manifest_validation(paths, monkeypatch)
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return {"producer.py": ("a" if calls <= 2 else "b") * 64}

    def build_into(destination, **kwargs):
        (destination / "report.json").write_text("{}\n", encoding="utf-8")
        return {"status": "passed"}

    monkeypatch.setattr(subject, "producer_sources", changing_sources)
    monkeypatch.setattr(subject, "_authenticate_inputs", lambda **kwargs: {})
    monkeypatch.setattr(subject, "_build_into", build_into)
    with pytest.raises(RuntimeError, match="changed during publication"):
        subject.build(**_toctou_build_kwargs(paths, output))
    assert not output.exists()
    assert not (tmp_path / ".candidate.lock").exists()


def test_build_rejects_fixed_input_change_during_authenticate_inputs(
        tmp_path, monkeypatch):
    paths = _toctou_build_inputs(tmp_path)
    output = tmp_path / "candidate"
    _patch_toctou_manifest_validation(paths, monkeypatch)

    def mutate_input(**kwargs):
        paths["actor"].write_bytes(b"changed\n")
        return {}

    monkeypatch.setattr(subject, "producer_sources", lambda: {"p.py": "a" * 64})
    monkeypatch.setattr(subject, "_authenticate_inputs", mutate_input)
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        subject.build(**_toctou_build_kwargs(paths, output))
    assert not output.exists()
    assert not (tmp_path / ".candidate.lock").exists()


def test_build_rejects_manifest_validation_change_before_publication(
        tmp_path, monkeypatch):
    paths = _toctou_build_inputs(tmp_path)
    output = tmp_path / "candidate"
    _patch_toctou_manifest_validation(paths, monkeypatch)

    def build_into(destination, **kwargs):
        (destination / "report.json").write_text("{}\n", encoding="utf-8")
        paths["manifest_validation"].write_bytes(b"changed\n")
        return {"status": "passed"}

    monkeypatch.setattr(subject, "producer_sources", lambda: {"p.py": "a" * 64})
    monkeypatch.setattr(subject, "_authenticate_inputs", lambda **kwargs: {})
    monkeypatch.setattr(subject, "_build_into", build_into)
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        subject.build(**_toctou_build_kwargs(paths, output))
    assert not output.exists()
    assert not (tmp_path / ".candidate.lock").exists()


@pytest.mark.parametrize(
    "component", ("training_ledger", "dual_evaluation", "failure_closeout"))
def test_build_rejects_designation_component_drift_without_output(
        component, tmp_path, monkeypatch):
    paths = _toctou_build_inputs(tmp_path)
    output = tmp_path / "candidate"
    _patch_toctou_manifest_validation(paths, monkeypatch)

    def build_into(destination, **kwargs):
        (destination / "report.json").write_text("{}\n", encoding="utf-8")
        paths["designation_" + component].write_bytes(b"changed\n")
        return {"status": "passed"}

    monkeypatch.setattr(subject, "producer_sources", lambda: {"p.py": "a" * 64})
    monkeypatch.setattr(subject, "_authenticate_inputs", lambda **kwargs: {})
    monkeypatch.setattr(subject, "_build_into", build_into)
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        subject.build(**_toctou_build_kwargs(paths, output))
    assert not output.exists()
    assert not (tmp_path / ".candidate.lock").exists()


def test_designation_build_script_source_drift_aborts_without_output(
        tmp_path, monkeypatch):
    paths = _toctou_build_inputs(tmp_path)
    output = tmp_path / "candidate"
    _patch_toctou_manifest_validation(paths, monkeypatch)
    calls = 0

    monkeypatch.setattr(
        subject, "local_source_hashes", lambda roots: {"producer.py": "p" * 64})

    def changing_designation_closure():
        nonlocal calls
        calls += 1
        return {
            "scripts/build_warehouse_r41_diagnostic_designation_v2.py":
                ("a" if calls == 1 else "b") * 64,
        }

    monkeypatch.setattr(
        subject.designation_binding.designation, "source_closure",
        changing_designation_closure)
    monkeypatch.setattr(subject, "_authenticate_inputs", lambda **kwargs: {})
    with pytest.raises(RuntimeError, match="changed during authentication"):
        subject.build(**_toctou_build_kwargs(paths, output))
    assert not output.exists()
    assert not (tmp_path / ".candidate.lock").exists()
