from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from backend.warehouse_alignment_online_runtime import digest, file_hash
from backend.warehouse_alignment_online_explanation import (
    _agent, _best_progress, _MAIN_ANSWER_FORBIDDEN, parse_question,
)
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    VERSION as BOOSTED_TREE_VERSION,
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_online_explanation_v8 import (
    EXACT_ACTIONS,
    QUICK_QUESTIONS,
    RCPD_VERSION,
    R41DiagnosticOnlineAlignmentExplainer,
    _goal_purpose_supported,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    CONFLICT_VALIDATION_VERSION,
    FULL_SCENE_MANIFEST_VERSION,
    PORTABLE_RUNTIME_MANIFEST_VERSION,
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    GROUPS,
    R41DiagnosticPublicTreeProgramV8,
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
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return value


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if not ACTOR.is_file() or not PROTOCOL.is_file():
        pytest.skip("requires the frozen r4.1 diagnostic Actor")
    directory = tmp_path_factory.mktemp("r41_v8_explanation_runtime")
    manifest_path = directory / "portable_manifest.json"
    manifest = _portable_manifest(manifest_path)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    return R41DiagnosticOnlineAlignmentRuntime(
        ACTOR.resolve(),
        training_protocol_path=PROTOCOL.resolve(),
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
        pytest.skip("requires the frozen development-only v8 expansion")
    payload = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    assert payload["final_audit_rows_access"] is False
    assert payload["program_access"] is False
    return deepcopy(payload["fit_supplement"][0])


def _component(feature_names, action: str,
               action_names=EXACT_ACTIONS) -> R41DiagnosticBoostedTreeProgram:
    relations = R41DiagnosticPublicRelationsV8(feature_names)
    split_index = relations.feature_names.index("self.battery")
    selected = action_names.index(action)
    trees = []
    for output_index in range(5):
        value = 6.0 if output_index == selected else 0.0
        trees.append({
            "iteration": 0,
            "output_index": output_index,
            "nodes": [
                {"kind": "split", "feature_index": split_index,
                 "threshold": 0.5, "missing_go_to_left": True,
                 "left": 1, "right": 2},
                {"kind": "leaf", "value": value},
                {"kind": "leaf", "value": value},
            ],
        })
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_TREE_VERSION,
        "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(range(5)),
        "action_names": list(action_names),
        "output_kind": "multiclass_logits",
        "baseline": [0.0] * 5,
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1,
        "trees": trees,
        "metadata": {},
    })


def _program(runtime, action: str,
             action_names=EXACT_ACTIONS) -> R41DiagnosticPublicTreeProgramV8:
    feature_names = tuple(runtime.actor.metadata["feature_names"])
    component = _component(feature_names, action, action_names)
    return R41DiagnosticPublicTreeProgramV8.from_programs(
        feature_names,
        component,
        component,
        component,
        component,
        mix_weights={group: 0.25 for group in GROUPS},
        metadata={
            "native_source_actor_sha256": runtime.actor_sha256,
            "source_actor_parameters_sha256": runtime.actor.metadata[
                "actor_parameters_sha256"],
            "source_full_manifest_bindings": runtime.source_full_manifest_bindings,
            "diagnostic_rcpd_version": RCPD_VERSION,
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
            "formal_ready": False,
            "action_legality_features": {},
            "action_constraint_reason_features": {},
        },
    )


def _explainer(tmp_path: Path, runtime, action: str,
               action_names=EXACT_ACTIONS):
    program = _program(runtime, action, action_names)
    path = tmp_path / f"program_{action}_{action_names[0]}.json"
    path.write_text(program.to_json() + "\n", encoding="utf-8")
    return R41DiagnosticOnlineAlignmentExplainer(
        path.resolve(), expected_program_sha256=file_hash(path), runtime=runtime,
        allow_test_fixture=True,
    )


def _record(runtime, scene, player_action="WAIT"):
    env = runtime.environment(deepcopy(scene))
    return runtime.step(env, player_action)


def _assert_plain_short(result, language):
    assert set(result) == {"answer", "evidence_detail"}
    assert result["answer"].strip() and result["evidence_detail"].strip()
    assert _MAIN_ANSWER_FORBIDDEN.search(result["answer"]) is None
    endings = re.findall(r"[。！？]|(?<!\d)[.!?](?!\d)", result["answer"])
    assert len(endings) <= 2
    assert ("绑定帧" in result["evidence_detail"] if language == "zh"
            else "Bound frame" in result["evidence_detail"])


def test_online_v8_import_is_dependency_light_and_purpose_gate_is_conjunctive():
    code = (
        "import sys; import backend.warehouse_r41_diagnostic_online_explanation_v8; "
        "assert not [name for name in sys.modules "
        "if name.startswith(('torch','sklearn','backend.training'))]"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    complete = {
        "actual_public_progress": True,
        "chosen_branch_reproduced": True,
        "supported": True,
    }
    assert _goal_purpose_supported(True, complete) is True
    for key in complete:
        changed = dict(complete)
        changed[key] = False
        assert _goal_purpose_supported(True, changed) is False
    assert _goal_purpose_supported(False, complete) is False


def test_exact_five_action_mapping_is_enforced_externally(
    tmp_path, runtime, development_scene,
):
    record = _record(runtime, development_scene)
    action = record["submitted_actions"]["robot_2"]
    reordered = ("DOWN", "UP", "LEFT", "RIGHT", "WAIT")
    with pytest.raises(ValueError, match="action authority"):
        _explainer(tmp_path, runtime, action, reordered)


def test_six_quick_questions_are_bilingual_frame_bound_and_plain(
    tmp_path, runtime, development_scene,
):
    record = _record(runtime, development_scene)
    action = record["submitted_actions"]["robot_2"]
    explainer = _explainer(tmp_path, runtime, action)
    assert parse_question(QUICK_QUESTIONS["en"][4]) == {
        "intent": "goal", "focus": "next",
    }
    for language in ("zh", "en"):
        for index, question in enumerate(QUICK_QUESTIONS[language]):
            focus = "next" if index in (4, 5) else "executed"
            result = explainer.answer({
                "question": question,
                "focus": focus,
                "language": language,
                "frame": record["after"]["state"]["frame"],
            }, record, runtime)
            _assert_plain_short(result, language)


def test_structured_trace_paths_stay_in_detail_and_mismatch_omits_them(
    tmp_path, runtime, development_scene,
):
    record = _record(runtime, development_scene)
    actual = record["submitted_actions"]["robot_2"]
    matching = _explainer(tmp_path, runtime, actual)
    request = {"question": "机器人2刚才为什么这样行动？",
               "focus": "executed", "language": "zh", "frame": 1}
    agreed = matching.answer(request, record, runtime)
    assert "近似程序路径" in agreed["evidence_detail"]
    assert "self.battery" in agreed["evidence_detail"]
    assert "近似程序" not in agreed["answer"]

    different = next(action for action in EXACT_ACTIONS if action != actual)
    mismatch = _explainer(tmp_path, runtime, different)
    disagreed = mismatch.answer(request, record, runtime)
    assert "不一致" in disagreed["evidence_detail"]
    assert "未显示或使用其路径" in disagreed["evidence_detail"]
    assert "近似程序路径" not in disagreed["evidence_detail"]
    assert "self.battery" not in disagreed["evidence_detail"]
    assert "是为了" not in disagreed["answer"]


def test_goal_purpose_requires_confirmed_progress_agreement_and_physical_support(
    tmp_path, runtime, development_scene, monkeypatch,
):
    env = runtime.environment(deepcopy(development_scene))
    selected = None
    for _ in range(30):
        if env.done:
            break
        record = runtime.step(env, "WAIT")
        before = runtime.from_snapshot(record["before"])
        after = runtime.from_snapshot(record["after"])
        action = record["submitted_actions"]["robot_2"]
        progress = _best_progress(before, tuple(_agent(after).position))
        event = any(item.get("agent_id") == "robot_2"
                    and item.get("event") in ("pickup", "delivery")
                    for item in record["events"])
        if (action != "WAIT" and record["executed_actions"]["robot_2"] == action
                and progress and not event):
            selected = (record, action)
            break
    if selected is None:
        pytest.skip("development scene did not expose a clean progress frame")
    record, action = selected
    explainer = _explainer(tmp_path, runtime, action)
    request = {"question": "机器人2刚才为什么这样行动？",
               "focus": "executed", "language": "zh",
               "frame": record["after"]["state"]["frame"]}
    supported = explainer.answer(request, record, runtime)
    assert "是为了靠近" in supported["answer"]
    assert "隔离物理对照" in supported["evidence_detail"]
    assert "对照支持=是" in supported["evidence_detail"]

    import backend.warehouse_r41_diagnostic_online_explanation_v8 as module

    real = module._physical_progress_counterfactual

    def unsupported(**kwargs):
        value = real(**kwargs)
        value["supported"] = False
        return value

    monkeypatch.setattr(module, "_physical_progress_counterfactual", unsupported)
    withheld = explainer.answer(request, record, runtime)
    assert "是为了" not in withheld["answer"]
    assert "现有证据无法可靠说明更具体的原因" in withheld["answer"]
    assert "对照支持=否" in withheld["evidence_detail"]


def test_collision_separates_chosen_action_from_environment_stopped_motion(
    tmp_path, runtime, development_scene, monkeypatch,
):
    env = runtime.environment(deepcopy(development_scene))
    for agent, position in zip(env.state.agents, ((5, 3), (5, 2))):
        agent.position = position
        agent.battery = 100.0
        agent.active = True

    def right_policy(observations, deterministic=True):
        assert deterministic is True
        distribution = np.asarray((0, 0, 0, 1, 0), dtype=np.float32)
        return ({agent_id: "RIGHT" for agent_id in observations},
                {agent_id: distribution.copy() for agent_id in observations})

    monkeypatch.setattr(runtime.actor, "act", right_policy)
    record = runtime.step(env, "WAIT")
    assert record["submitted_actions"]["robot_2"] == "RIGHT"
    assert record["executed_actions"]["robot_2"] == "WAIT"
    explainer = _explainer(tmp_path, runtime, "RIGHT")
    result = explainer.answer({
        "question": "我们刚才为什么发生碰撞？",
        "focus": "executed", "language": "zh", "frame": 1,
    }, record, runtime)
    assert "机器人2选择了向右" in result["answer"]
    assert "环境取消了它的移动" in result["answer"]
    assert "实际留在原地" in result["answer"]
    _assert_plain_short(result, "zh")


def test_wait_three_and_arbitrary_history_are_isolated(
    tmp_path, runtime, development_scene,
):
    env = runtime.environment(deepcopy(development_scene))
    history = [runtime.step(env, "WAIT") for _ in range(3)]
    first = history[0]
    action = first["submitted_actions"]["robot_2"]
    explainer = _explainer(tmp_path, runtime, action)
    live_snapshot_hash = digest(env.snapshot())
    live_rng_hash = digest(env.get_rng_state())
    first_hash = digest(first)
    actor_hash = digest({name: sha256(value.tobytes()).hexdigest()
                         for name, value in runtime.actor.weights.items()})
    program_hash = digest(explainer.program.to_dict())
    result = explainer.answer({
        "question": "如果我等待三步会怎样？",
        "focus": "next", "language": "zh", "frame": 1,
    }, first, runtime)
    assert "依次选择" in result["answer"]
    assert "真实回合保持不变" in result["answer"]
    assert "反事实使用同一冻结策略的隔离副本" in result["evidence_detail"]
    assert digest(first) == first_hash
    assert digest({name: sha256(value.tobytes()).hexdigest()
                   for name, value in runtime.actor.weights.items()}) == actor_hash
    assert digest(explainer.program.to_dict()) == program_hash
    assert digest(env.snapshot()) == live_snapshot_hash
    assert digest(env.get_rng_state()) == live_rng_hash
    assert history[-1]["after"]["state"]["frame"] == 3

    historical = explainer.answer({
        "question": "机器人2在第1帧为什么这样行动？",
        "focus": "executed", "language": "zh", "frame": 1,
    }, first, runtime)
    assert "绑定帧：1" in historical["evidence_detail"]
    with pytest.raises(ValueError, match="frame_mismatch"):
        explainer.answer({
            "question": "机器人2刚才为什么这样行动？",
            "focus": "executed", "language": "zh", "frame": 3,
        }, first, runtime)
