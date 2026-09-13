from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pytest

from backend.warehouse_alignment_online_explanation import _MAIN_ANSWER_FORBIDDEN
from backend.warehouse_alignment_online_runtime import digest, file_hash
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    VERSION as BOOSTED_TREE_VERSION,
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_online_explanation_v9 import (
    EXACT_ACTIONS,
    QUICK_QUESTIONS,
    RCPD_VERSION,
    R41DiagnosticOnlineAlignmentExplainerV9,
    explanation_access,
    explanation_sources,
    make_artifact_binding,
    validate_artifact_binding,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    CONFLICT_VALIDATION_VERSION,
    FULL_SCENE_MANIFEST_VERSION,
    PORTABLE_RUNTIME_MANIFEST_VERSION,
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    GROUPS,
    R41DiagnosticPublicTreeProgramV9,
)
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
ACTOR = (ROOT / "output/warehouse_native/r41_active_2m_20260911/"
         "boundaries/step_2000000/actor.npz")
PROTOCOL = ROOT / "output/warehouse_native/r41_active_2m_20260911/protocol.json"
DEVELOPMENT = (ROOT / "output/warehouse_native/"
               "r41_diagnostic_development_expansion_v8_20260912/"
               "development_expansion.json")


def _portable_manifest(path: Path) -> dict:
    value = {
        "version": PORTABLE_RUNTIME_MANIFEST_VERSION,
        "source_full_manifest_version": FULL_SCENE_MANIFEST_VERSION,
        "source_conflict_validation_version": CONFLICT_VALIDATION_VERSION,
        "source_conflict_validation_sha256": "4" * 64,
        "source_full_manifest_file_sha256": "1" * 64,
        "source_full_manifest_content_sha256": "2" * 64,
        "source_full_manifest_semantic_sha256": "3" * 64,
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
    }
    value["content_sha256"] = digest(value)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return value


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires the frozen diagnostic Actor")
    directory = tmp_path_factory.mktemp("r41_v9_explanation_runtime")
    manifest_path = directory / "portable_manifest.json"
    manifest = _portable_manifest(manifest_path)
    protocol = json.loads(PROTOCOL.read_text())
    return R41DiagnosticOnlineAlignmentRuntime(
        ACTOR.resolve(), training_protocol_path=PROTOCOL.resolve(),
        manifest_path=manifest_path.resolve(),
        expected_actor_sha256=file_hash(ACTOR),
        expected_training_protocol_file_sha256=file_hash(PROTOCOL),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=manifest["content_sha256"],
        expected_manifest_semantic_sha256=digest(manifest),
        allow_test_fixture=True,
    )


@pytest.fixture(scope="module")
def development_scene():
    if not DEVELOPMENT.is_file():
        pytest.skip("requires a development-only scene")
    value = json.loads(DEVELOPMENT.read_text())
    assert value["final_audit_rows_access"] is False
    return deepcopy(value["fit_supplement"][0])


def _component(base_names, action: str) -> R41DiagnosticBoostedTreeProgram:
    relations = R41DiagnosticPublicRelationsV9(base_names)
    selected = EXACT_ACTIONS.index(action)
    split_index = relations.feature_names.index("self.battery")
    trees = []
    for output_index in range(len(EXACT_ACTIONS)):
        leaf = 6.0 if output_index == selected else 0.0
        trees.append({
            "iteration": 0, "output_index": output_index,
            "nodes": [
                {"kind": "split", "feature_index": split_index,
                 "threshold": .5, "missing_go_to_left": True,
                 "left": 1, "right": 2},
                {"kind": "leaf", "value": leaf},
                {"kind": "leaf", "value": leaf},
            ],
        })
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_TREE_VERSION, "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(range(5)), "action_names": list(EXACT_ACTIONS),
        "output_kind": "multiclass_logits", "baseline": [0.0] * 5,
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1, "trees": trees, "metadata": {},
    })


def _explainer(tmp_path: Path, runtime, action: str):
    relations = R41DiagnosticPublicRelationsV9(
        tuple(runtime.actor.metadata["feature_names"]))
    component = _component(relations.base_feature_names, action)
    program = R41DiagnosticPublicTreeProgramV9.from_programs(
        relations.base_feature_names, component, component, component, component,
        mix_weights={group: .25 for group in GROUPS},
        metadata={
            "version": RCPD_VERSION,
            "binding_sha256": "a" * 64,
            "fit_config_sha256": "b" * 64,
            "public_feature_contract_sha256": digest(relations.contract()),
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
            "formal_ready": False,
        },
    )
    path = tmp_path / ("program_" + action + ".json")
    path.write_text(program.to_json() + "\n")
    return R41DiagnosticOnlineAlignmentExplainerV9(
        path, expected_program_sha256=file_hash(path), runtime=runtime,
        allow_test_fixture=True)


def _plain(result, language):
    assert set(result) == {"answer", "evidence_detail"}
    assert _MAIN_ANSWER_FORBIDDEN.search(result["answer"]) is None
    assert len(re.findall(r"[。！？]|(?<!\d)[.!?](?!\d)", result["answer"])) <= 2
    assert ("绑定帧" in result["evidence_detail"] if language == "zh"
            else "Bound frame" in result["evidence_detail"])


def _access(**changes):
    value = {
        "condition": "A", "stage": "task1", "surface": "live",
        "request_kind": "new_question", "active_run_id": "run-1",
        "bound_run_id": "run-1", "selected_frame": 1,
        "available_frames": [0, 1], "round_closed": False,
    }
    value.update(changes)
    return value


def test_renderer_source_boundary_is_public_runtime_only():
    sources = explanation_sources()
    assert "backend/warehouse_r41_diagnostic_public_features_v9.py" in sources
    assert "backend/warehouse_r41_diagnostic_public_tree_program_v9.py" in sources
    assert not [name for name in sources
                if "training/" in name or "outer" in name
                or "final" in name or "holdout" in name]


def test_access_allows_a_task1_live_history_and_review_but_isolates_b_and_task2():
    assert explanation_access(_access())["allowed"] is True
    assert explanation_access(_access(
        surface="history", selected_frame=0))["allowed"] is True
    assert explanation_access(_access(
        surface="round_review", round_closed=True))["allowed"] is True
    assert explanation_access(_access(condition="B"))["reason"] == \
        "condition_b_replay_only"
    assert explanation_access(_access(stage="task2"))["reason"] == \
        "task2_explanations_closed"
    assert explanation_access(_access(
        stage="task2", request_kind="read_answer"))["allowed"] is False
    assert explanation_access(_access(
        bound_run_id="other"))["reason"] == "cross_run_binding_denied"


def test_hash_only_artifact_binding_rejects_tampering(runtime):
    value = make_artifact_binding(
        actor_sha256=runtime.actor_sha256, program_sha256="5" * 64,
        runtime_signature=runtime.signature,
        runtime_manifest_sha256=runtime.manifest_file_sha256,
        candidate_lock_sha256="6" * 64,
        public_feature_contract_sha256="7" * 64)
    assert validate_artifact_binding(
        value, actor_sha256=runtime.actor_sha256, program_sha256="5" * 64,
        runtime_signature=runtime.signature,
        runtime_manifest_sha256=runtime.manifest_file_sha256,
        public_feature_contract_sha256="7" * 64) == value
    changed = deepcopy(value)
    changed["candidate_lock_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="binding_mismatch"):
        validate_artifact_binding(
            changed, actor_sha256=runtime.actor_sha256,
            program_sha256="5" * 64, runtime_signature=runtime.signature,
            runtime_manifest_sha256=runtime.manifest_file_sha256,
            public_feature_contract_sha256="7" * 64)


def test_six_shortcuts_and_free_input_are_plain_bilingual(
    tmp_path, runtime, development_scene,
):
    record = runtime.step(runtime.environment(deepcopy(development_scene)), "WAIT")
    explainer = _explainer(
        tmp_path, runtime, record["submitted_actions"]["robot_2"])
    for language in ("zh", "en"):
        for index, question in enumerate(QUICK_QUESTIONS[language]):
            focus = "next" if index in (4, 5) else "executed"
            result = explainer.answer_study({
                "question": question, "focus": focus, "language": language,
                "frame": 1,
            }, record, runtime, access_context=_access())
            _plain(result, language)
    free = explainer.answer_study({
        "question": "Why did Robot 2 choose that move just now?",
        "focus": "executed", "language": "en", "frame": 1,
    }, record, runtime, access_context=_access())
    _plain(free, "en")
    with pytest.raises(PermissionError, match="task2_explanations_closed"):
        explainer.answer_study({
            "question": QUICK_QUESTIONS["zh"][0], "focus": "executed",
            "language": "zh", "frame": 1,
        }, record, runtime, access_context=_access(stage="task2"))


def test_program_mismatch_withholds_purpose_and_counterfactual_is_isolated(
    tmp_path, runtime, development_scene,
):
    env = runtime.environment(deepcopy(development_scene))
    record = runtime.step(env, "WAIT")
    actual = record["submitted_actions"]["robot_2"]
    different = next(action for action in EXACT_ACTIONS if action != actual)
    explainer = _explainer(tmp_path, runtime, different)
    result = explainer.answer_study({
        "question": QUICK_QUESTIONS["zh"][0], "focus": "executed",
        "language": "zh", "frame": 1,
    }, record, runtime, access_context=_access())
    assert "是为了" not in result["answer"]
    assert "近似程序路径" not in result["evidence_detail"]

    before_snapshot = digest(env.snapshot())
    before_rng = digest(env.get_rng_state())
    result = explainer.answer_study({
        "question": "如果我等待三步会怎样？", "focus": "next",
        "language": "zh", "frame": 1,
    }, record, runtime, access_context=_access())
    assert "真实回合保持不变" in result["answer"]
    assert digest(env.snapshot()) == before_snapshot
    assert digest(env.get_rng_state()) == before_rng


def test_environment_blocking_is_separate_from_selected_action(
    tmp_path, runtime, development_scene, monkeypatch,
):
    env = runtime.environment(deepcopy(development_scene))
    for agent, position in zip(env.state.agents, ((5, 3), (5, 2))):
        agent.position = position
        agent.battery = 100.0
        agent.active = True

    def right_policy(observations, deterministic=True):
        distribution = np.asarray((0, 0, 0, 1, 0), dtype=np.float32)
        return ({agent_id: "RIGHT" for agent_id in observations},
                {agent_id: distribution.copy() for agent_id in observations})

    monkeypatch.setattr(runtime.actor, "act", right_policy)
    record = runtime.step(env, "WAIT")
    assert record["submitted_actions"]["robot_2"] == "RIGHT"
    assert record["executed_actions"]["robot_2"] == "WAIT"
    explainer = _explainer(tmp_path, runtime, "RIGHT")
    result = explainer.answer_study({
        "question": QUICK_QUESTIONS["zh"][2], "focus": "executed",
        "language": "zh", "frame": 1,
    }, record, runtime, access_context=_access())
    assert "机器人2选择了向右" in result["answer"]
    assert "环境取消了它的移动" in result["answer"]
    assert "实际留在原地" in result["answer"]
    _plain(result, "zh")
