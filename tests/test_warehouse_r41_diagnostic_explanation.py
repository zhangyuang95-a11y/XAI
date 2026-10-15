from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training import warehouse_r41_diagnostic_rcpd as rcpd
from backend.training import warehouse_r41_diagnostic_question_bank as question_bank
from backend.warehouse_alignment_online_explanation import parse_question
from backend.warehouse_r41_online_explanation import (
    DIAGNOSTIC_VERSION,
    R41DiagnosticOnlineAlignmentExplainer,
    diagnostic_explanation_sources,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    CONFLICT_VALIDATION_VERSION,
    FULL_SCENE_MANIFEST_VERSION,
    PORTABLE_RUNTIME_MANIFEST_VERSION,
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_diagnostic_model_tree import (
    R41DiagnosticModelTreeProgram,
    VERSION as MODEL_TREE_VERSION,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "output/warehouse_native/r41_active_2m_20260911/boundaries/step_2000000/actor.npz"
PROTOCOL = ROOT / "output/warehouse_native/r41_active_2m_20260911/protocol.json"


def _manifest(path: Path, **extra):
    value = {
        "version": FULL_SCENE_MANIFEST_VERSION,
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "splits": {},
        **extra,
    }
    value["content_sha256"] = digest(value)
    path.write_text(canonical(value) + "\n", encoding="utf-8")
    return value


def _runtime(path: Path, value):
    protocol = json.loads(PROTOCOL.read_text())
    content = deepcopy(value); claimed = content.pop("content_sha256")
    return R41DiagnosticOnlineAlignmentRuntime(
        ACTOR.resolve(), training_protocol_path=PROTOCOL.resolve(),
        manifest_path=path.resolve(), expected_actor_sha256=file_hash(ACTOR),
        expected_training_protocol_file_sha256=file_hash(PROTOCOL),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(value),
    )


def test_real_terminal_actor_loads_from_training_protocol_and_portable_signature_is_stable(tmp_path):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires terminal r4.1 Actor")
    full_path = tmp_path / "full.json"
    full = _manifest(full_path)
    first = _runtime(full_path, full)
    portable_path = tmp_path / "portable.json"
    source = first.source_full_manifest_bindings
    portable = _manifest(portable_path,
        version=PORTABLE_RUNTIME_MANIFEST_VERSION,
        source_full_manifest_version=FULL_SCENE_MANIFEST_VERSION,
        source_conflict_validation_version=CONFLICT_VALIDATION_VERSION,
        source_conflict_validation_sha256="4" * 64,
        source_full_manifest_file_sha256=source["manifest_file_sha256"],
        source_full_manifest_content_sha256=source["manifest_content_sha256"],
        source_full_manifest_semantic_sha256=source["manifest_semantic_sha256"])
    second = _runtime(portable_path, portable)
    assert first.actor_sha256 == "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
    assert first.signature == second.signature
    assert first.runtime_manifest_signature != second.runtime_manifest_signature
    assert first.verify_binding() == first.signature


def test_runtime_cache_reauthenticates_changed_manifest_bytes(tmp_path):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires terminal r4.1 Actor")
    path = tmp_path / "manifest.json"
    manifest = _manifest(path)
    runtime = _runtime(path, manifest)
    original = path.read_text(encoding="utf-8")
    path.write_text(original + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="external binding changed"):
        runtime.verify_binding()


@pytest.mark.parametrize(("changed", "message"), (
    ({"version": "warehouse-r41-diagnostic-conflict-scene-manifest.v2"},
     "manifest version"),
    ({"diagnostic_contract_version": "warehouse-r41-diagnostic-conflict.v1"},
     "source contract"),
    ({"diagnostic_conflict_graph_sha256": "0" * 64}, "source contract"),
))
def test_runtime_rejects_legacy_or_changed_full_contract(tmp_path, changed, message):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires terminal r4.1 Actor")
    path = tmp_path / "manifest.json"
    manifest = _manifest(path, **changed)
    with pytest.raises(ValueError, match=message):
        _runtime(path, manifest)


def test_portable_runtime_rejects_legacy_full_manifest_parent(tmp_path):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires terminal r4.1 Actor")
    path = tmp_path / "portable.json"
    manifest = _manifest(
        path,
        version=PORTABLE_RUNTIME_MANIFEST_VERSION,
        source_full_manifest_version=(
            "warehouse-r41-diagnostic-conflict-scene-manifest.v2"),
        source_conflict_validation_version=CONFLICT_VALIDATION_VERSION,
        source_conflict_validation_sha256="4" * 64,
        source_full_manifest_file_sha256="1" * 64,
        source_full_manifest_content_sha256="2" * 64,
        source_full_manifest_semantic_sha256="3" * 64,
    )
    with pytest.raises(ValueError, match="source contract"):
        _runtime(path, manifest)


def test_observation_hash_components_never_cross_fit_partition():
    def row(obs, anchor="", action="WAIT", groups=()):
        return {"observation": np.asarray(obs, dtype=np.float32), "anchor": anchor,
                "action": action, "groups": groups, "kind": "ordinary",
                "branch_action": "", "physical_hash": ""}
    rows = []
    for index in range(3000):
        rows.append(row([index, index % 7], f"a{index // 5}",
                        rcpd.ACTIONS[index % len(rcpd.ACTIONS)],
                        (rcpd.GROUPS[index % len(rcpd.GROUPS)],)))
    # Exact duplicates must follow their source observation's assignment.
    rows += [row([index, index % 7], action=rcpd.ACTIONS[index % len(rcpd.ACTIONS)],
                 groups=(rcpd.GROUPS[index % len(rcpd.GROUPS)],))
             for index in range(100)]
    split = rcpd._partition(rows)
    assert np.array_equal(split, rcpd._partition(rows))
    by_hash = {}
    for item, validation in zip(rows, split):
        by_hash.setdefault(rcpd._obs_hash(item["observation"]), set()).add(bool(validation))
    assert all(len(values) == 1 for values in by_hash.values())
    assert .15 <= float(split.mean()) <= .25
    for action in rcpd.ACTIONS:
        selected = np.asarray([item["action"] == action for item in rows])
        assert .15 <= float(split[selected].mean()) <= .25


def test_diagnostic_gates_keep_nonwait_at_ninety_percent():
    metrics = {
        "overall": {"fidelity": .91}, "nonwait": {"fidelity": .899},
        "critical": {name: {"scenes": 10, "fidelity": .90} for name in rcpd.GROUPS},
        "effective_intervention_direction": {"scenes": 10, "fidelity": .90},
    }
    assert rcpd._gate(metrics)["passed"] is False
    metrics["nonwait"]["fidelity"] = .90
    assert rcpd._gate(metrics)["passed"] is True


def test_diagnostic_leaf_fit_budget_records_slow_registered_candidates():
    assert rcpd.LOGISTIC_MAX_ITER == 3000
    assert rcpd.contract()["leaf"]["maximum_iterations"] == 3000


def test_english_current_task_shortcut_parses_as_goal():
    assert parse_question("Which task is Robot 2 moving toward?") == {
        "intent": "goal", "focus": "next"}


def test_diagnostic_explainer_has_distinct_complete_source_scope(tmp_path):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires terminal r4.1 Actor")
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path)
    runtime = _runtime(manifest_path, manifest)
    metadata = {"diagnostic_rcpd_version": rcpd.VERSION,
         "diagnostic_rcpd_binding_sha256": "0" * 64,
         "source_full_manifest_bindings": runtime.source_full_manifest_bindings,
         "native_source_actor_sha256": runtime.actor_sha256,
         "source_actor_parameters_sha256": runtime.actor.metadata[
             "actor_parameters_sha256"],
         "program_family": "axis_router_sparse_public_linear_softmax_leaves",
         "candidate_profile": dict(rcpd.MODEL_TREE_PROFILES[0]),
         "router_depth_cap": 8, "router_leaf_cap": 32,
         "router_min_samples_leaf": 64,
         "action_legality_features": {},
         "action_constraint_reason_features": {},
         "runtime_controller": "native_neural_actor_only",
         "runtime_action_override": False,
         "program_feedback_into_actor": False,
         "ppo_joint_steps": 0, "optimizer_updates": 0,
         "formal_ready": False}
    program = R41DiagnosticModelTreeProgram.from_dict({
        "version": MODEL_TREE_VERSION,
        "action_names": list(ACTIONS),
        "feature_names": list(runtime.actor.metadata["feature_names"]),
        "router": {"depth": 0, "leaf_count": 1,
                   "nodes": [{"kind": "leaf", "model_index": 0}]},
        "leaf_models": [{"router_node": 0, "classes": [2],
                         "feature_indices": [], "coefficients": [[]],
                         "intercepts": [0.0]}],
        "metadata": metadata,
    })
    program_path = tmp_path / "program.json"
    program_path.write_text(canonical(program.to_dict()) + "\n", encoding="utf-8")
    explainer = R41DiagnosticOnlineAlignmentExplainer(
        program_path, expected_program_sha256=file_hash(program_path), runtime=runtime)
    assert explainer.contract_report["version"] == DIAGNOSTIC_VERSION
    assert "backend/warehouse_r41_diagnostic_online_runtime.py" in explainer.sources
    assert "backend/warehouse_r41_diagnostic_model_tree.py" in explainer.sources
    assert "env/warehouse_native/r41_diagnostic_conflict.py" in explainer.sources
    assert explainer.sources == diagnostic_explanation_sources()
    explainer._assert_current(runtime)


def test_model_tree_schema_rejects_internal_prediction_metadata():
    payload = {
        "version": MODEL_TREE_VERSION, "action_names": list(ACTIONS),
        "feature_names": ["public"],
        "router": {"depth": 0, "leaf_count": 1,
                   "nodes": [{"kind": "leaf", "model_index": 0}]},
        "leaf_models": [{"router_node": 0, "classes": [0],
                         "feature_indices": [], "coefficients": [[]],
                         "intercepts": [0.0]}],
        "metadata": {"actor_logits": [1.0]},
    }
    with pytest.raises(ValueError, match="prohibited internal metadata"):
        R41DiagnosticModelTreeProgram.from_dict(payload)


def test_model_tree_fit_is_independent_of_validation_labels():
    rng = np.random.default_rng(41041)
    count = 1500
    observations = rng.normal(size=(count, 197)).astype(np.float32)
    labels = np.argmax(np.stack((observations[:, 0], -observations[:, 0],
        observations[:, 1], -observations[:, 1], observations[:, 2]), axis=1),
        axis=1).astype(np.uint8)
    validation = np.zeros(count, dtype=np.bool_); validation[::5] = True
    probabilities = np.full((count, len(ACTIONS)), .025, dtype=np.float32)
    probabilities[np.arange(count), labels] = .9
    group_bits = np.zeros(count, np.uint8)
    kinds = np.full(count, "ordinary", dtype="U16")
    changed = np.zeros(count, np.bool_)
    arrays = {
        "observations": observations, "probabilities": probabilities,
        "action_indices": labels.copy(),
        "weights": rcpd._sample_weights(labels, validation, group_bits,
                                         kinds == "intervention", changed),
        "split_validation": validation, "kinds": kinds,
        "group_bits": group_bits,
        "scene_fingerprints": np.full(count, "a" * 64, dtype="U64"),
        "anchor_ids": np.full(count, "", dtype="U240"),
        "branch_actions": np.full(count, "", dtype="U8"),
        "physical_hashes": np.full(count, "", dtype="U64"),
    }
    actor = SimpleNamespace(artifact_sha256="b" * 64,
        metadata={"feature_names": [f"public.{i}" for i in range(197)],
                  "actor_parameters_sha256": "c" * 64})
    first = rcpd._fit_one(actor, arrays, rcpd.MODEL_TREE_PROFILES[0],
                          "d" * 64, {"manifest_file_sha256": "e" * 64})
    altered = {key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in arrays.items()}
    altered["action_indices"][validation] = \
        (altered["action_indices"][validation] + 1) % len(ACTIONS)
    altered["probabilities"][validation] = \
        np.roll(altered["probabilities"][validation], 1, axis=1)
    altered["weights"] = rcpd._sample_weights(
        altered["action_indices"], validation, group_bits,
        kinds == "intervention", changed)
    assert np.array_equal(arrays["weights"][~validation],
                          altered["weights"][~validation])
    second = rcpd._fit_one(actor, altered, rcpd.MODEL_TREE_PROFILES[0],
                           "d" * 64, {"manifest_file_sha256": "e" * 64})
    assert first["program"] == second["program"]
    assert first["fit_diagnostics"] == second["fit_diagnostics"]


def test_question_pool_rejects_overlap_with_any_other_split():
    questions = [{"fingerprint": f"{i:064x}"} for i in range(36)]
    manifest = {"splits": {"question_bank": questions,
                            "final_test": [{"fingerprint": questions[0]["fingerprint"]}]}}
    with pytest.raises(ValueError, match="overlap"):
        question_bank._pool(manifest)
